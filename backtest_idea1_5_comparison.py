"""Ideas 1 + 5 Combined Entry Signal Comparison — MSTR & MSTU

Compares the CURRENT entry gate:
    u1        = (ehma3 > 0.7) & (hb3 >= 2)
    tf1_entry = u1 & ((above_ma30 ^ clean_10d) | v_recent)

vs IDEAS 1+5 COMBINED — both filters applied simultaneously:
    Idea 5 tightens U1:
        ehma5[i]  = mean(err_hi[i-4 : i+1])
        u1        = (ehma3 > 0.7) & (hb3 >= 2) & (ehma5 > 0.3)
    Idea 1 tightens the XOR trend gate:
        bull_regime = above_ma30 & ma30_slope_pos
        tf1_entry   = u1 & ((bull_regime ^ clean_10d) | v_recent)

  Idea 5: blocks entries where the 3-day error spike has no 5-day
  structural backing — filters isolated bear bounces.
  Idea 1: blocks entries above a DECLINING MA30 — filters rallies in
  a downtrend that haven't yet re-established a rising trend.
  Together they target two distinct populations of bad entries.

Runs the three UI periods for MSTR and MSTU:
  Bear  : 2025-06-01 → 2026-05-31   (MSTR) / 2025-06-04 → 2026-05-31 (MSTU)
  Bull  : 2024-06-05 → 2025-06-14   (both)
  Full  : 2024-06-01 → 2026-05-31   (both)

Stop losses: MSTR −3%  (SL5 regime-adaptive re-entry), MSTU −7%.
"""

import sys, os, time, math
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import joblib

from paths import DAILY_MODEL_CT

# ── Constants ──────────────────────────────────────────────────────────────
WARMUP       = 35
INITIAL_CAP  = 100_000.0
TAX_RATE     = 0.35

# UI period definitions  (end_iso, start_iso) matching the live dashboard calls
PERIODS = {
    "Bear" : ("2026-05-31", "2025-06-01"),
    "Bull" : ("2025-06-14", "2024-06-05"),
    "Full" : ("2026-05-31", "2024-06-01"),
}
MSTU_BEAR_START = "2025-06-04"   # MSTU uses a slightly different bear start


# ══════════════════════════════════════════════════════════════════════════
# 1.  Build CT predictions  (mirrors _build_backtest_preds in the app)
# ══════════════════════════════════════════════════════════════════════════
def build_preds(fetch_start_iso: str, fetch_end_iso: str):
    """Return (preds_df, raw_df) or None on failure."""
    try:
        AD = joblib.load(str(DAILY_MODEL_CT))
    except Exception as e:
        print(f"[ERROR] Could not load CT model: {e}"); return None

    # BTC daily
    try:
        d_btc = yf.download("BTC-USD", start=fetch_start_iso, end=fetch_end_iso,
                             progress=False, auto_adjust=True)
        if isinstance(d_btc.columns, pd.MultiIndex):
            d_btc.columns = [c[0] for c in d_btc.columns]
        d_btc.index = pd.DatetimeIndex(d_btc.index).tz_localize(None).normalize()
    except Exception as e:
        print(f"[ERROR] BTC download failed: {e}"); return None
    if d_btc.empty or "Close" not in d_btc.columns:
        print("[ERROR] BTC data empty"); return None

    df = pd.DataFrame({
        "btc_close":  d_btc["Close"],
        "btc_high":   d_btc["High"],
        "btc_low":    d_btc["Low"],
        "btc_volume": d_btc.get("Volume", pd.Series(dtype=float)),
    })
    raw_df = df[["btc_close","btc_high","btc_low","btc_volume"]].copy()

    # Macro
    _MACRO = {"eth":"ETH-USD","spx":"^GSPC","ndx":"^IXIC",
              "vix":"^VIX","gold":"GC=F","dxy":"DX-Y.NYB","tnx":"^TNX"}
    for nm, sym in _MACRO.items():
        try:
            _d = yf.download(sym, start=fetch_start_iso, end=fetch_end_iso,
                             progress=False, auto_adjust=True)
            if isinstance(_d.columns, pd.MultiIndex):
                _d.columns = [c[0] for c in _d.columns]
            _d.index = pd.DatetimeIndex(_d.index).tz_localize(None).normalize()
            df[f"{nm}_close"] = _d["Close"].reindex(df.index).ffill(limit=7)
        except Exception:
            pass

    # On-chain
    _ONCHAIN = ["hash-rate","difficulty","n-transactions","miners-revenue",
                "n-unique-addresses","transaction-fees-usd","mempool-size",
                "estimated-transaction-volume-usd","market-cap",
                "avg-block-size","cost-per-transaction"]
    for _m in _ONCHAIN:
        try:
            _r = requests.get(
                f"https://api.blockchain.info/charts/{_m}",
                params={"timespan":"3years","format":"json","sampled":"true"},
                timeout=20)
            _vals = _r.json().get("values", [])
            _s = pd.Series(
                {pd.Timestamp(_v["x"], unit="s").normalize(): _v["y"] for _v in _vals},
                name=f"oc_{_m.replace('-','_')}", dtype=float)
            _s = _s[~_s.index.duplicated(keep="last")].sort_index()
            _s.index = pd.DatetimeIndex(_s.index).tz_localize(None)
            df[_s.name] = _s.reindex(df.index).ffill(limit=7)
        except Exception:
            pass

    # Coinbase premium
    _CB_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    _cb_rows: list = []
    _cb_cur = pd.Timestamp(fetch_start_iso)
    _cb_end = pd.Timestamp(fetch_end_iso)
    while _cb_cur <= _cb_end:
        _cb_chunk = min(_cb_cur + pd.Timedelta(days=299), _cb_end)
        try:
            _r2 = requests.get(_CB_URL, params={
                "granularity": 86400,
                "start": _cb_cur.strftime("%Y-%m-%dT00:00:00Z"),
                "end":   (_cb_chunk + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z"),
            }, timeout=30)
            if _r2.status_code == 200:
                _cb_rows.extend(_r2.json())
        except Exception:
            pass
        _cb_cur = _cb_chunk + pd.Timedelta(days=1)
        time.sleep(0.1)
    if _cb_rows:
        _cb_df = pd.DataFrame(_cb_rows, columns=["ts","low","high","open","close","volume"])
        _cb_df["date"] = pd.to_datetime(_cb_df["ts"], unit="s").dt.normalize()
        _cb_df = _cb_df.drop_duplicates("date").set_index("date").sort_index()
        _cb_close = _cb_df["close"].reindex(df.index).astype(float)
        c_ref = df["btc_close"]
        _prem = (_cb_close - c_ref) / c_ref * 100
        df["cb_premium"]     = _prem
        df["cb_premium_ma3"] = _prem.rolling(3).mean()
        df["cb_premium_z7"]  = (_prem - _prem.rolling(7).mean()) / _prem.rolling(7).std()
    else:
        df["cb_premium"] = df["cb_premium_ma3"] = df["cb_premium_z7"] = 0.0

    # Feature engineering (mirrors app exactly)
    c  = df["btc_close"]; h = df["btc_high"]
    l_ = df["btc_low"];   v = df["btc_volume"]
    ret = np.log(c).diff()
    feat = pd.DataFrame(index=df.index)
    for k in [1,3,5,7,14,30]: feat[f"ret_{k}"]  = ret.rolling(k).sum()
    for k in [5,10,20,30]:    feat[f"vol_{k}"]  = ret.rolling(k).std()
    prev_c = c.shift(1)
    tr = pd.concat([(h-l_),(h-prev_c).abs(),(l_-prev_c).abs()], axis=1).max(axis=1)
    for k in [7,14,30]: feat[f"atr_{k}"] = tr.rolling(k).mean()/c
    feat["range_today"] = (h-l_)/c
    feat["range_ma7"]   = ((h-l_)/c).rolling(7).mean()
    feat["range_ma30"]  = ((h-l_)/c).rolling(30).mean()
    feat["range_std30"] = ((h-l_)/c).rolling(30).std()
    gain = c.diff().clip(lower=0).rolling(14).mean()
    loss = (-c.diff().clip(upper=0)).rolling(14).mean()
    feat["rsi_14"] = 100 - 100/(1 + gain/loss.replace(0, np.nan))
    e12 = c.ewm(span=12,adjust=False).mean(); e26 = c.ewm(span=26,adjust=False).mean()
    macd = e12 - e26
    feat["macd"]      = macd/c
    feat["macd_sig"]  = macd.ewm(span=9,adjust=False).mean()/c
    feat["macd_hist"] = (macd - macd.ewm(span=9,adjust=False).mean())/c
    ma20 = c.rolling(20).mean(); sd20 = c.rolling(20).std()
    feat["bb_width"]   = (4*sd20)/ma20
    feat["dist_hi_30"] = c/c.rolling(30).max() - 1
    feat["dist_lo_30"] = c/c.rolling(30).min() - 1
    feat["dist_hi_90"] = c/c.rolling(90).max() - 1
    feat["vol_chg_1"]    = np.log(v).diff()
    feat["vol_z_20"]     = (np.log(v)-np.log(v).rolling(20).mean())/np.log(v).rolling(20).std()
    feat["vol_ma_ratio"] = v/v.rolling(20).mean()
    dow = df.index.dayofweek
    for i in range(6): feat[f"dow_{i}"] = (dow==i).astype(float)
    for nm in ["spx","ndx","vix","gold","dxy","tnx","eth"]:
        col = f"{nm}_close"
        if col not in df.columns: continue
        s = df[col]; lr = np.log(s).diff()
        for k in [1,5,20]: feat[f"{nm}_ret_{k}"] = lr.rolling(k).sum()
        feat[f"{nm}_vol_20"] = lr.rolling(20).std()
    for corr_nm, corr_col in [("spx","spx_close"),("ndx","ndx_close"),
                                ("gold","gold_close"),("dxy","dxy_close")]:
        if corr_col in df.columns:
            feat[f"btc_{corr_nm}_corr_30"] = ret.rolling(30).corr(np.log(df[corr_col]).diff())
    for col in [x for x in df.columns if x.startswith("oc_")]:
        s = df[col].astype(float); sl = np.log(s.replace(0, np.nan))
        feat[f"{col}_d1"]  = sl.diff(1); feat[f"{col}_d7"] = sl.diff(7)
        feat[f"{col}_z30"] = (sl - sl.rolling(30).mean())/sl.rolling(30).std()
    nh = h.shift(-1); nl = l_.shift(-1)
    y_hi = (nh-c)/c; y_lo = (c-nl)/c
    feat["y_hi_ema3"] = y_hi.shift(1).ewm(span=3,adjust=False).mean()
    feat["y_lo_ema3"] = y_lo.shift(1).ewm(span=3,adjust=False).mean()
    feat["y_hi_ema7"] = y_hi.shift(1).ewm(span=7,adjust=False).mean()
    feat["y_lo_ema7"] = y_lo.shift(1).ewm(span=7,adjust=False).mean()
    p3h = h.shift(1).rolling(3).max(); p3l = l_.shift(1).rolling(3).min()
    feat["above_3d_high"]  = (c > p3h).astype(float)
    feat["below_3d_low"]   = (c < p3l).astype(float)
    feat["bo_strength_up"] = (c/p3h - 1).clip(lower=0)
    feat["bo_strength_dn"] = (1 - c/p3l).clip(lower=0)
    yhl = y_hi.shift(1); yll = y_lo.shift(1)
    feat["y_hi_surprise"] = yhl - yhl.ewm(span=7,adjust=False).mean()
    feat["y_lo_surprise"] = yll - yll.ewm(span=7,adjust=False).mean()
    nr = ret.clip(upper=0)
    feat["dn_vol_5"]  = nr.rolling(5).std()
    feat["dn_vol_20"] = nr.rolling(20).std()
    sma50 = c.rolling(50).mean()
    feat["below_sma50"]    = (c < sma50).astype(float)
    feat["below_sma50_5d"] = feat["below_sma50"].rolling(5).min().fillna(0)
    for _cb_col in ["cb_premium","cb_premium_ma3","cb_premium_z7"]:
        feat[_cb_col] = df[_cb_col].fillna(0.0)

    # Ensemble predictions
    fc = AD["feat_cols"]
    feat = feat.replace([np.inf, -np.inf], np.nan)
    for col in fc:
        if col not in feat.columns: feat[col] = np.nan
    F = feat[fc].dropna()
    if F.empty:
        print("[ERROR] No feature rows after dropna"); return None

    if AD.get("ensemble") and AD.get("constituents"):
        yhi = np.mean([con["m_hi"].predict(F) for con in AD["constituents"]], axis=0)
        ylo = np.mean([con["m_lo"].predict(F) for con in AD["constituents"]], axis=0)
        if AD.get("blended") and float(AD.get("alpha", 1.0)) < 1.0:
            a = float(AD["alpha"])
            yhi = a*yhi + (1-a)*float(AD.get("mu_hi", 0))
            ylo = a*ylo + (1-a)*float(AD.get("mu_lo", 0))
    else:
        yhi = AD["hi_model"].predict(F)
        ylo = AD["lo_model"].predict(F)

    c_vals = c.reindex(F.index).values
    ph = c_vals * (1 + np.clip(yhi, 0, None))
    pl = c_vals * (1 - np.clip(ylo, 0, None))
    idx_arr = np.asarray(F.index, dtype="datetime64[ns]")
    nd = np.empty(len(F), dtype="datetime64[ns]")
    nd[:-1] = idx_arr[1:]; nd[-1] = idx_arr[-1] + np.timedelta64(1, "D")
    preds_df = pd.DataFrame(
        {"close_asof": c_vals, "pred_high": ph, "pred_low": pl},
        index=pd.DatetimeIndex(nd, name="target_date"),
    )
    preds_df = preds_df[~preds_df.index.duplicated(keep="last")]
    return preds_df, raw_df


# ══════════════════════════════════════════════════════════════════════════
# 2.  Core signal arrays from predictions + actual BTC H/L
# ══════════════════════════════════════════════════════════════════════════
def compute_signals(comp: pd.DataFrame, entry_mode: str):
    """Return dict of signal arrays. entry_mode: 'current' or 'idea1+5'."""
    N       = len(comp)
    c_asof  = comp["close_asof"].values.astype(float)
    pred_hi = comp["pred_high"].values.astype(float)
    pred_lo = comp["pred_low"].values.astype(float)
    act_hi  = comp["actual_high"].values.astype(float)
    act_lo  = comp["actual_low"].values.astype(float)

    err_hi = (act_hi - pred_hi) / c_asof * 100
    err_lo = (pred_lo - act_lo) / c_asof * 100
    hi_brk = (act_hi > pred_hi).astype(int)
    lo_brk = (act_lo < pred_lo).astype(int)

    ehma3 = np.zeros(N); elma3 = np.zeros(N)
    hb3   = np.zeros(N, dtype=int); lb3 = np.zeros(N, dtype=int)
    for i in range(N):
        s = max(0, i-2)
        ehma3[i] = np.mean(err_hi[s:i+1]); elma3[i] = np.mean(err_lo[s:i+1])
        hb3[i]   = int(np.sum(hi_brk[s:i+1])); lb3[i] = int(np.sum(lo_brk[s:i+1]))

    # Idea 5: 5-bar rolling error average for broader structural context
    ehma5 = np.array([
        float(np.mean(err_hi[max(0, i-4):i+1])) for i in range(N)
    ])

    if entry_mode == "idea1+5":
        # Idea 5: tighten U1 with 5-day error average
        u1 = (ehma3 > 0.7) & (hb3 >= 2) & (ehma5 > 0.3)
    else:
        u1 = (ehma3 > 0.7) & (hb3 >= 2)

    d1 = (lb3 >= 2)    & (elma3 > 0.5)
    d2 = ehma3 < -0.75
    d3 = np.zeros(N, dtype=bool)
    for i in range(1, N):
        consec = 0
        for k in range(i-1, -1, -1):
            if hi_brk[k]: consec += 1
            else: break
        if consec >= 3 and lo_brk[i]:
            d3[i] = True

    ma30 = np.full(N, np.nan)
    for i in range(N):
        w = min(30, i+1)
        ma30[i] = np.mean(c_asof[max(0, i-w+1):i+1])
    above_ma30     = c_asof > ma30
    ma30_slope_pos = np.zeros(N, dtype=bool)
    for i in range(N):
        if i >= 5 and np.isfinite(ma30[i]) and np.isfinite(ma30[i-5]):
            ma30_slope_pos[i] = ma30[i] > ma30[i-5]
    bull_regime = above_ma30 & ma30_slope_pos

    clean_10d = np.zeros(N, dtype=bool)
    for i in range(N):
        lo_i = max(0, i-7)
        clean_10d[i] = not bool(np.any(d1[lo_i:i] | d2[lo_i:i]))

    _DN_NORM_W    = 30
    roll_ehi_norm = np.array([
        float(np.mean(err_hi[max(0, i-_DN_NORM_W+1):i+1])) for i in range(N)
    ])
    dn_score_arr = np.zeros(N)
    for i in range(N):
        norm = max(abs(roll_ehi_norm[i]), 0.01)
        dn_score_arr[i] = (
            (-ehma3[i] / norm)                          * 0.30 +
            (lb3[i]    / 3.0)                           * 0.30 +
            (elma3[i]  / max(abs(elma3[i]), 0.10))      * 0.20 +
            float(lo_brk[i])                            * 0.20
        )
    v_rev_bar = (dn_score_arr > 0.8) & (err_lo > 3.0)
    v_recent  = np.zeros(N, dtype=bool)
    for i in range(N):
        v_recent[i] = bool(np.any(v_rev_bar[max(0, i-2):i+1]))

    if entry_mode == "idea1+5":
        # Idea 1: tighten XOR gate — require bull_regime (rising MA30) not just above_ma30
        tf1_entry = u1 & ((bull_regime ^ clean_10d) | v_recent)
    else:
        tf1_entry = u1 & ((above_ma30 ^ clean_10d) | v_recent)

    return dict(
        d2=d2, d3=d3, bull_regime=bull_regime, tf1_entry=tf1_entry,
        above_ma30=above_ma30, v_recent=v_recent,
    )


# ══════════════════════════════════════════════════════════════════════════
# 3.  Asset backtest loop (MSTR / MSTU with stop-loss + SL5 re-entry)
# ══════════════════════════════════════════════════════════════════════════
def run_asset_backtest(
    preds_df, raw_df,
    asset_ticker: str,
    start_iso: str,
    end_iso: str,
    stop_pct: float,       # 0.03 for MSTR, 0.07 for MSTU
    entry_mode: str,       # "current" or "idea1+5"
):
    end_dt   = pd.Timestamp(end_iso)
    start_dt = pd.Timestamp(start_iso)
    pre_dt   = start_dt - pd.Timedelta(days=60)

    p = preds_df.loc[(preds_df.index >= pre_dt) & (preds_df.index <= end_dt)].copy()
    if len(p) < WARMUP + 3:
        return None

    p["actual_high"]  = raw_df["btc_high"].reindex(p.index).values
    p["actual_low"]   = raw_df["btc_low"].reindex(p.index).values
    p["actual_close"] = raw_df["btc_close"].reindex(p.index).values
    comp = p.dropna(subset=["actual_high","actual_low","actual_close"]).reset_index()
    N = len(comp)
    if N < WARMUP + 3:
        return None

    dates = pd.DatetimeIndex(comp["target_date"])
    _bt0  = max(WARMUP, int(dates.searchsorted(start_dt)))
    if N - _bt0 < 3:
        return None

    # Fetch asset prices
    try:
        d_asset = yf.download(
            asset_ticker,
            start=pre_dt.strftime("%Y-%m-%d"),
            end=(end_dt + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
            progress=False, auto_adjust=True,
        )
        if isinstance(d_asset.columns, pd.MultiIndex):
            d_asset.columns = [c[0] for c in d_asset.columns]
        d_asset.index = pd.DatetimeIndex(d_asset.index).tz_localize(None).normalize()
    except Exception as e:
        print(f"[ERROR] {asset_ticker} download failed: {e}"); return None
    if d_asset.empty or "Close" not in d_asset.columns:
        return None

    asset_raw = d_asset["Close"].sort_index()
    asset_all = asset_raw.reindex(
        pd.date_range(asset_raw.index[0], max(asset_raw.index[-1], end_dt), freq="D")
    ).ffill()
    # For MSTU: backward-fill pre-inception dates with the first available price
    if asset_ticker == "MSTU":
        asset_all = asset_all.bfill()
    asset_px = asset_all.reindex(dates).ffill().bfill().values.astype(float)

    sigs = compute_signals(comp, entry_mode)
    d2         = sigs["d2"]
    d3         = sigs["d3"]
    bull_regime= sigs["bull_regime"]
    tf1_entry  = sigs["tf1_entry"]
    above_ma30 = sigs["above_ma30"]
    v_recent   = sigs["v_recent"]

    # ── Trade loop ──────────────────────────────────────────────────────
    nav   = INITIAL_CAP; pos = "CASH"; qty = 0.0
    e_price = e_nav = e_date = e_trigger = None
    e_reentry = False; stop_px = 0.0
    from_sl = False; bars_since_sl = 0
    trades: list = []; nav_arr = np.full(N, np.nan)

    for i in range(N):
        price = asset_px[i]
        if i < _bt0:
            nav_arr[i] = INITIAL_CAP; continue
        if not np.isfinite(price) or price <= 0:
            nav_arr[i] = qty * asset_px[i-1] if pos == "LONG" and i > 0 else nav
            continue
        if pos == "LONG":
            cur = qty * price
            if price <= stop_px:   # stop triggered on close
                nav = qty * price
                trades.append(dict(
                    entry_date=e_date, entry_price=e_price, entry_nav=e_nav,
                    entry_trigger=e_trigger, exit_date=dates[i], exit_price=price,
                    exit_nav=nav, pnl_pct=(price/e_price-1)*100,
                    pnl_abs=nav-e_nav, exit_signal=f"SL-{int(stop_pct*100)}%",
                    duration_days=(dates[i]-e_date).days, stop_triggered=True,
                    was_reentry=e_reentry,
                ))
                pos = "CASH"; qty = 0.0; stop_px = 0.0; e_reentry = False
                from_sl = True; bars_since_sl = 0
            else:
                should_exit = bool(d3[i] or (d2[i] and not bull_regime[i]))
                if should_exit:
                    nav = cur
                    trades.append(dict(
                        entry_date=e_date, entry_price=e_price, entry_nav=e_nav,
                        entry_trigger=e_trigger, exit_date=dates[i], exit_price=price,
                        exit_nav=nav, pnl_pct=(price/e_price-1)*100,
                        pnl_abs=nav-e_nav, exit_signal="D3" if d3[i] else "D2 (bear)",
                        duration_days=(dates[i]-e_date).days, stop_triggered=False,
                        was_reentry=e_reentry,
                    ))
                    pos = "CASH"; qty = 0.0; stop_px = 0.0; e_reentry = False
                    from_sl = False
                else:
                    nav = cur
        else:
            if from_sl:
                bars_since_sl += 1
            _exit_at_i = d3[i] or (d2[i] and not bull_regime[i])
            _sl_ok = (not from_sl or bool(bull_regime[i]) or bars_since_sl >= 10)
            if tf1_entry[i] and _sl_ok and (from_sl or not _exit_at_i):
                e_reentry = bool(from_sl)
                qty = nav / price; e_price = price; e_date = dates[i]
                e_nav = nav; pos = "LONG"; stop_px = price * (1 - stop_pct)
                from_sl = False; bars_since_sl = 0
                if v_recent[i]:
                    e_trigger = "U1 + V-reversal"
                elif above_ma30[i]:
                    e_trigger = "U1 + ↑MA30"
                else:
                    e_trigger = "U1 + Clean 7d"
        nav_arr[i] = qty * price if pos == "LONG" else nav

    if pos == "LONG" and np.isfinite(asset_px[N-1]) and asset_px[N-1] > 0:
        nav_arr[N-1] = qty * asset_px[N-1]

    nav_series = pd.Series(nav_arr[_bt0:], index=dates[_bt0:]).ffill()
    bh_px0     = asset_px[_bt0]
    bh_series  = pd.Series(INITIAL_CAP * asset_px[_bt0:] / bh_px0, index=dates[_bt0:])

    # ── Statistics ───────────────────────────────────────────────────────
    final_nav = float(nav_series.iloc[-1])
    final_bh  = float(bh_series.iloc[-1])
    strat_ret = (final_nav / INITIAL_CAP - 1) * 100
    bh_ret    = (final_bh  / INITIAL_CAP - 1) * 100
    wins      = [t for t in trades if t["pnl_pct"] > 0]
    losses    = [t for t in trades if t["pnl_pct"] <= 0]
    win_rate  = 100*len(wins)/len(trades) if trades else 0.0
    rm        = nav_series.cummax()
    max_dd    = float(((nav_series - rm)/rm*100).min())
    days_in   = sum(t["duration_days"] for t in trades)
    tot_days  = max(1, (dates[N-1] - dates[_bt0]).days)
    rf_daily  = (1.045)**(1/252) - 1
    dr_s      = nav_series.pct_change().fillna(0)
    exc_s     = dr_s - rf_daily
    sharpe    = float(exc_s.mean()/exc_s.std()*np.sqrt(252)) if exc_s.std()>0 else 0.0
    _annual_net: dict = {}
    for t in trades:
        yr = pd.Timestamp(t["exit_date"]).year
        _annual_net[yr] = _annual_net.get(yr, 0.0) + t["pnl_abs"]
    total_tax   = sum(TAX_RATE * max(0.0, v) for v in _annual_net.values())
    after_tax_ret = ((final_nav - total_tax) / INITIAL_CAP - 1) * 100
    n_sl        = sum(1 for t in trades if t.get("stop_triggered"))

    return dict(
        strat_ret=strat_ret, bh_ret=bh_ret, max_dd=max_dd,
        sharpe=sharpe, win_rate=win_rate,
        n_trades=len(trades), n_sl=n_sl,
        time_in=100*days_in/tot_days,
        after_tax_ret=after_tax_ret,
        trades=trades,
    )



# ══════════════════════════════════════════════════════════════════════════
# 4.  Main: run all periods x assets x modes and print comparison
#     Data fetch matches UI exactly: fetch_start = period_start - 200 days
# ══════════════════════════════════════════════════════════════════════════

def _build_for_period(period_start_iso: str, period_end_iso: str):
    """Per-period preds build matching run_mstr_backtest / run_mstu_backtest:
    fetch_start = period_start - 200 calendar days  (ensures dist_hi_90 warm)
    fetch_end   = period_end   + 3 calendar days
    """
    start_dt = pd.Timestamp(period_start_iso)
    end_dt   = pd.Timestamp(period_end_iso)
    fs = (start_dt - pd.Timedelta(days=200)).strftime("%Y-%m-%d")
    fe = (end_dt   + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    print(f"    window: {fs} -> {fe}", flush=True)
    return build_preds(fs, fe)


def _print_trades(trades: list, asset: str, period: str, mode: str):
    """Compact trade log for cross-checking against the UI trade table."""
    label = f"  Trade log -- {asset}  {period}  ({mode})"
    print(label)
    if not trades:
        print("    (no trades)")
        return
    hdr = (f"    {'#':>2}  {'Entry date':>12}  {'Exit date':>12}  "
           f"{'EntryPx':>9}  {'ExitPx':>9}  {'P&L%':>7}  "
           f"{'Trigger':<22}  {'Exit signal'}")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for idx, t in enumerate(trades, 1):
        e_d = pd.Timestamp(t["entry_date"]).strftime("%Y-%m-%d")
        x_d = pd.Timestamp(t["exit_date"]).strftime("%Y-%m-%d")
        trig = t.get("entry_trigger") or "-"
        sig  = t.get("exit_signal", "-")
        sl_flag = " [SL]" if t.get("stop_triggered") else ""
        print(f"    {idx:>2}  {e_d:>12}  {x_d:>12}  "
              f"{t['entry_price']:>9.2f}  {t['exit_price']:>9.2f}  "
              f"{t['pnl_pct']:>+6.1f}%  "
              f"{trig:<22}  {sig}{sl_flag}")
    print()


def main():
    SEP = "=" * 70
    print("\n" + SEP)
    print("  Ideas 1+5 Combined Comparison -- MSTR & MSTU (all UI periods)")
    print("  Idea 1: bull_regime XOR gate  |  Idea 5: ehma5 > 0.3 on U1")
    print("  Fetch: per-period (period_start - 200d -> period_end + 3d)")
    print("         Matches run_mstr_backtest / run_mstu_backtest exactly")
    print(SEP)

    # UI period definitions
    # Bear MSTR starts 2025-06-01, MSTU 2025-06-04; use 2025-06-01 as fetch key
    # (3-day difference is irrelevant -- both are well inside 200-day warmup).
    PERIOD_FETCH = {
        "Bear": ("2025-06-01", "2026-05-31"),
        "Bull": ("2024-06-05", "2025-06-14"),
        "Full": ("2024-06-01", "2026-05-31"),
    }

    print("\nBuilding CT predictions (one per period, matching UI fetch windows) ...")
    period_data: dict = {}
    for pname, (pstart, pend) in PERIOD_FETCH.items():
        print(f"\n  [{pname}]", flush=True)
        res = _build_for_period(pstart, pend)
        if res is None:
            print(f"  [FATAL] Could not build preds for {pname}. Aborting.")
            return
        preds_df, raw_df = res
        print(f"  -> {len(preds_df)} bars  "
              f"({preds_df.index[0].date()} -> {preds_df.index[-1].date()})")
        period_data[pname] = (preds_df, raw_df)

    # Asset configs: (label, ticker, stop_pct, period_name -> (end, start))
    configs = [
        ("MSTR", "MSTR", 0.03, {
            "Bear": ("2026-05-31", "2025-06-01"),
            "Bull": ("2025-06-14", "2024-06-05"),
            "Full": ("2026-05-31", "2024-06-01"),
        }),
        ("MSTU", "MSTU", 0.07, {
            "Bear": ("2026-05-31", "2025-06-04"),
            "Bull": ("2025-06-14", "2024-06-05"),
            "Full": ("2026-05-31", "2024-06-01"),
        }),
    ]

    print("\n\n" + SEP)
    print("  SUMMARY TABLE")
    print(SEP)

    for asset_label, ticker, stop_pct, periods in configs:
        print(f"\n  {'-'*68}")
        print(f"  {asset_label}  (stop-loss {int(stop_pct*100)}%)")
        print(f"  {'-'*68}")
        hdr = (f"  {'Period':<6}  {'Mode':<10}  {'Return':>8}  {'B&H':>8}  "
               f"{'MaxDD':>7}  {'Sharpe':>6}  {'WinR%':>6}  "
               f"{'Trd':>4}  {'SLs':>4}  {'TiM%':>5}  {'AfterTax':>9}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        for period_name, (end_iso, start_iso) in periods.items():
            preds_df, raw_df = period_data[period_name]
            r_cur = run_asset_backtest(preds_df, raw_df, ticker,
                                       start_iso, end_iso, stop_pct, "current")
            r_i15 = run_asset_backtest(preds_df, raw_df, ticker,
                                       start_iso, end_iso, stop_pct, "idea1+5")
            for mode, r in [("current", r_cur), ("idea1+5", r_i15)]:
                if r is None:
                    print(f"  {period_name:<6}  {mode:<10}  [NO DATA]")
                    continue
                print(
                    f"  {period_name:<6}  {mode:<10}"
                    f"  {r['strat_ret']:>+7.1f}%"
                    f"  {r['bh_ret']:>+7.1f}%"
                    f"  {r['max_dd']:>+6.1f}%"
                    f"  {r['sharpe']:>6.2f}"
                    f"  {r['win_rate']:>5.0f}%"
                    f"  {r['n_trades']:>4}"
                    f"  {r['n_sl']:>4}"
                    f"  {r['time_in']:>5.0f}%"
                    f"  {r['after_tax_ret']:>+8.1f}%"
                )
            if r_cur and r_i15:
                dr = r_i15["strat_ret"] - r_cur["strat_ret"]
                dd = r_i15["max_dd"]    - r_cur["max_dd"]
                ds = r_i15["sharpe"]    - r_cur["sharpe"]
                print(
                    f"  {'':6}  {'delta':>10}"
                    f"  {dr:>+6.1f}pp"
                    f"  {'':>8}"
                    f"  {dd:>+5.1f}pp"
                    f"  {ds:>+5.2f}"
                    f"  {'':>6}"
                    f"  {r_i15['n_trades']-r_cur['n_trades']:>+4}"
                    f"  {r_i15['n_sl']-r_cur['n_sl']:>+4}"
                )
            print()

    print(SEP)
    print("  Legend: B&H=Buy&Hold  MaxDD=Max Drawdown  WinR=Win Rate")
    print("  Trd=# Trades  SLs=Stop-Loss exits  TiM=Time in Market")
    print("  AfterTax = 35% tax on annual net gains")
    print(SEP + "\n")

    # ── Trade logs for UI cross-check ────────────────────────────────────────
    print("\n" + SEP)
    print("  TRADE LOGS (current implementation) -- cross-check vs UI")
    print(SEP + "\n")

    for asset_label, ticker, stop_pct, periods in configs:
        for period_name, (end_iso, start_iso) in periods.items():
            preds_df, raw_df = period_data[period_name]
            r = run_asset_backtest(preds_df, raw_df, ticker,
                                   start_iso, end_iso, stop_pct, "current")
            if r:
                _print_trades(r["trades"], asset_label, period_name, "current")


if __name__ == "__main__":
    main()
