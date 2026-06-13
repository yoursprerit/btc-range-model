"""Idea 3 Entry Signal Comparison — MSTR & MSTU

Compares the CURRENT entry gate:
    tf1_entry = u1 & ((above_ma30 ^ clean_10d) | v_recent)

vs IDEA 3 — MA30 slope requirement on the above_ma30 entry path:
    above_ma30_valid = above_ma30 & ma30_slope_pos
    tf1_entry = u1 & ((above_ma30_valid ^ clean_10d) | v_recent)

above_ma30_valid requires BOTH price > MA30 AND MA30 is rising.
This blocks entries when price has bounced above a still-declining MA30,
which is typical of bear-market dead-cat bounces, while leaving all
genuine bull-trend entries (where MA30 is also rising) untouched.

NOTE: above_ma30_valid == bull_regime (above_ma30 & ma30_slope_pos),
so Idea 3 and Idea 1 use an algebraically identical gate.  This script
confirms that equivalence empirically and reports a per-period diagnostic
showing how many bars carry above_ma30=True but ma30_slope_pos=False
(the exact population Idea 3 targets).

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
    """Return dict of signal arrays. entry_mode: 'current' or 'idea3'."""
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

    if entry_mode == "idea3":
        # Idea 3: require MA30 also rising for the above_ma30 entry path.
        # above_ma30_valid = above_ma30 & ma30_slope_pos = bull_regime (same variable).
        above_ma30_valid = above_ma30 & ma30_slope_pos   # identical to bull_regime
        tf1_entry = u1 & ((above_ma30_valid ^ clean_10d) | v_recent)
    else:
        # Current / baseline
        tf1_entry = u1 & ((above_ma30 ^ clean_10d) | v_recent)

    # Diagnostic counts (returned for reporting)
    # Bars where above_ma30=True but ma30_slope_pos=False — the "declining-above" population
    declining_above = above_ma30 & ~ma30_slope_pos
    # Of those, how many would have triggered a current-mode entry?
    current_gate  = u1 & ((above_ma30 ^ clean_10d) | v_recent)
    idea3_gate    = u1 & ((bull_regime ^ clean_10d) | v_recent)
    entry_diff    = int(np.sum(current_gate != idea3_gate))
    n_declining   = int(np.sum(declining_above))

    return dict(
        d2=d2, d3=d3, bull_regime=bull_regime, tf1_entry=tf1_entry,
        above_ma30=above_ma30, v_recent=v_recent,
        _n_declining_above=n_declining,
        _n_entry_diff=entry_diff,
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
    entry_mode: str,       # "current" or "idea1"
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
# 4.  Main: run all periods × assets × modes and print comparison
# ══════════════════════════════════════════════════════════════════════════
def _period_signals(preds_df, raw_df, start_iso, end_iso):
    """Build comp slice and compute signals for the diagnostic section."""
    end_dt   = pd.Timestamp(end_iso)
    start_dt = pd.Timestamp(start_iso)
    pre_dt   = start_dt - pd.Timedelta(days=60)
    p = preds_df.loc[(preds_df.index >= pre_dt) & (preds_df.index <= end_dt)].copy()
    p["actual_high"]  = raw_df["btc_high"].reindex(p.index).values
    p["actual_low"]   = raw_df["btc_low"].reindex(p.index).values
    p["actual_close"] = raw_df["btc_close"].reindex(p.index).values
    comp = p.dropna(subset=["actual_high","actual_low","actual_close"]).reset_index()
    if len(comp) < WARMUP + 3:
        return None
    return compute_signals(comp, "idea3")   # diagnostics are mode-independent


def main():
    print("\n" + "═"*70)
    print("  Idea 3 Comparison — MSTR & MSTU (all UI periods)")
    print("  MA30 slope requirement: above_ma30_valid = above_ma30 & ma30_slope_pos")
    print("═"*70)

    GLOBAL_FETCH_START = "2023-11-01"
    GLOBAL_FETCH_END   = "2026-06-14"

    print(f"\nFetching data {GLOBAL_FETCH_START} → {GLOBAL_FETCH_END} …")
    result = build_preds(GLOBAL_FETCH_START, GLOBAL_FETCH_END)
    if result is None:
        print("[FATAL] Could not build predictions. Aborting.")
        return
    preds_df, raw_df = result
    print(f"  Predictions built: {len(preds_df)} bars  "
          f"({preds_df.index[0].date()} → {preds_df.index[-1].date()})")

    # ── Diagnostic: bars where above_ma30=True but ma30 declining (Idea 3 target) ──
    print("\n  Diagnostic — 'declining-above-MA30' population per period")
    print("  (bars where above_ma30=True but ma30_slope_pos=False = Idea 3 target)")
    print(f"  {'Period':<8}  {'Total bars':>10}  {'Decl-above':>10}  "
          f"{'Entry diff':>10}  {'Note'}")
    all_periods = {
        "Bear": ("2026-05-31", "2025-06-01"),
        "Bull": ("2025-06-14", "2024-06-05"),
        "Full": ("2026-05-31", "2024-06-01"),
    }
    for pname, (end_iso, start_iso) in all_periods.items():
        sigs = _period_signals(preds_df, raw_df, start_iso, end_iso)
        if sigs is None:
            continue
        # total bars in the backtest window (after warmup)
        end_dt   = pd.Timestamp(end_iso)
        start_dt = pd.Timestamp(start_iso)
        pre_dt   = start_dt - pd.Timedelta(days=60)
        p = preds_df.loc[(preds_df.index >= pre_dt) & (preds_df.index <= end_dt)].copy()
        p["actual_high"]  = raw_df["btc_high"].reindex(p.index).values
        p["actual_low"]   = raw_df["btc_low"].reindex(p.index).values
        p["actual_close"] = raw_df["btc_close"].reindex(p.index).values
        comp = p.dropna(subset=["actual_high","actual_low","actual_close"]).reset_index()
        _bt0 = max(WARMUP, int(pd.DatetimeIndex(comp["target_date"]).searchsorted(start_dt)))
        n_bars = len(comp) - _bt0
        nd = sigs["_n_declining_above"]
        ed = sigs["_n_entry_diff"]
        note = "identical to Idea 1" if ed == 0 else f"{ed} entry bar(s) differ"
        print(f"  {pname:<8}  {n_bars:>10}  {nd:>10}  {ed:>10}  {note}")
    print()

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

    for asset_label, ticker, stop_pct, periods in configs:
        print(f"{'─'*70}")
        print(f"  {asset_label}  (stop-loss {int(stop_pct*100)}%)")
        print(f"{'─'*70}")
        hdr = f"{'Period':<8}  {'Mode':<9}  {'Return':>8}  {'B&H':>8}  "
        hdr += f"{'MaxDD':>7}  {'Sharpe':>6}  {'WinR%':>6}  {'Trd':>4}  {'SLs':>4}  "
        hdr += f"{'TiM%':>5}  {'AfterTax':>9}"
        print(hdr)
        print("  " + "─"*(len(hdr)-2))

        for period_name, (end_iso, start_iso) in periods.items():
            for mode in ("current", "idea3"):
                r = run_asset_backtest(
                    preds_df, raw_df, ticker,
                    start_iso, end_iso, stop_pct, mode
                )
                if r is None:
                    print(f"  {period_name:<8}  {mode:<9}  [NO DATA]")
                    continue
                row = (
                    f"  {period_name:<8}  {mode:<9}"
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
                print(row)
            # Delta row
            r_cur = run_asset_backtest(preds_df, raw_df, ticker, start_iso, end_iso, stop_pct, "current")
            r_i3  = run_asset_backtest(preds_df, raw_df, ticker, start_iso, end_iso, stop_pct, "idea3")
            if r_cur and r_i3:
                delta_ret = r_i3["strat_ret"] - r_cur["strat_ret"]
                delta_dd  = r_i3["max_dd"]    - r_cur["max_dd"]
                delta_sh  = r_i3["sharpe"]    - r_cur["sharpe"]
                sign_r = "+" if delta_ret >= 0 else ""
                sign_d = "+" if delta_dd  >= 0 else ""
                sign_s = "+" if delta_sh  >= 0 else ""
                print(f"  {'':8}  {'Δ idea3':<9}"
                      f"  {sign_r}{delta_ret:>6.1f}pp"
                      f"  {'':>8}"
                      f"  {sign_d}{delta_dd:>5.1f}pp"
                      f"  {sign_s}{delta_sh:>5.2f}"
                      f"  {'':>6}"
                      f"  {r_i3['n_trades']-r_cur['n_trades']:>+4}"
                      f"  {r_i3['n_sl']-r_cur['n_sl']:>+4}")
            print()

    print("═"*70)
    print("  Legend: B&H=Buy&Hold  MaxDD=Max Drawdown  WinR%=Win Rate")
    print("  Trd=# Trades  SLs=Stop-Loss exits  TiM%=Time in Market")
    print("  AfterTax = 35% tax on annual net gains")
    print("  Idea 3 formula: above_ma30_valid = above_ma30 & ma30_slope_pos = bull_regime")
    print("  → algebraically identical to Idea 1; delta should be 0.0pp everywhere.")
    print("═"*70 + "\n")


if __name__ == "__main__":
    main()
