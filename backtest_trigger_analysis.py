#!/usr/bin/env python3
"""
Entry-Trigger Analysis: U1+MA30+clean7d combined vs separate triggers (MSTR & MSTU)
=====================================================================================

Questions answered:
1. Is "U1+↑MA30+clean7d" (both above_ma30 AND clean_7d simultaneously) more
   prone to loss than the single-filter triggers for MSTR / MSTU?

2. Would blocking entries where BOTH above_ma30 AND clean_7d fire improve results?

Uses the EXACT same data pipeline as the app's run_mstr_backtest() /
run_mstu_backtest():
  - yfinance daily BTC-USD (same as _build_backtest_preds)
  - 200-day pre-period warmup (ensures dist_hi_90 is fully warm at bt start)
  - Real Coinbase premium from API
  - Same-bar execution (after-hours for MSTR/MSTU)
  - Fixed -3% stop (MSTR) / -7% stop (MSTU) + SL5 regime-adaptive re-entry

Periods match the locked UI backtests:
  Bear  Jun 2025 – May 2026
  Bull  Jun 2024 – Jun 2025
  Full  Jun 2024 – May 2026
"""

import sys, warnings, joblib, requests, time as _time
warnings.filterwarnings("ignore")
from pathlib import Path
from collections import defaultdict

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import yfinance as yf

# ── Model (same path as app's DAILY_MODEL_CT) ────────────────────────────────
_MODEL_PATH = _ROOT / "models" / "inference_assets_ct.joblib"
if not _MODEL_PATH.exists():
    _MODEL_PATH = _ROOT / "artifacts" / "artifacts.pkl"

print(f"Loading CT model from: {_MODEL_PATH}")
AD = joblib.load(str(_MODEL_PATH))
print(f"  model keys: {list(AD.keys())[:6]}\n")

INITIAL_CAPITAL = 100_000.0
WARMUP = 35

# Locked periods matching UI — note: 200-day pre-fetch warmup computed per-period
PERIODS = {
    "Bear (Jun25→May26)":  ("2025-06-01", "2026-05-31"),
    "Bull (Jun24→Jun25)":  ("2024-06-05", "2025-06-14"),
    "Full (Jun24→May26)":  ("2024-06-01", "2026-05-31"),
}

ASSETS = {
    "MSTR": {"ticker": "MSTR",  "sl_pct": 0.03, "stop_label": "Fixed -3%"},
    "MSTU": {"ticker": "MSTU",  "sl_pct": 0.07, "stop_label": "Fixed -7%"},
}

_MACRO_SYMS = {
    "eth": "ETH-USD", "spx": "^GSPC", "ndx": "^IXIC",
    "vix": "^VIX",   "gold": "GC=F",  "dxy": "DX-Y.NYB", "tnx": "^TNX",
}
_ONCHAIN = [
    "hash-rate", "difficulty", "n-transactions", "miners-revenue",
    "n-unique-addresses", "transaction-fees-usd", "mempool-size",
    "estimated-transaction-volume-usd", "market-cap",
    "avg-block-size", "cost-per-transaction",
]


# ── Data helpers ──────────────────────────────────────────────────────────────
def _yf_dl(ticker, start, end):
    d = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] for c in d.columns]
    d.index = pd.DatetimeIndex(d.index).tz_localize(None).normalize()
    return d


_DATA_CACHE: dict = {}


def _build_backtest_preds(fetch_start, fetch_end):
    """Exact replica of app's _build_backtest_preds() — yfinance BTC + Coinbase premium."""
    cache_key = (fetch_start, fetch_end)
    if cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key]

    print(f"  Fetching {fetch_start} → {fetch_end} …")

    # 1. BTC daily (yfinance, same source as research script)
    try:
        d_btc = yf.download("BTC-USD", start=fetch_start, end=fetch_end,
                             progress=False, auto_adjust=True)
        if isinstance(d_btc.columns, pd.MultiIndex):
            d_btc.columns = [c[0] for c in d_btc.columns]
        d_btc.index = pd.DatetimeIndex(d_btc.index).tz_localize(None).normalize()
    except Exception as e:
        print(f"  [ERR] BTC fetch: {e}"); return None
    if d_btc.empty or "Close" not in d_btc.columns:
        print("  [ERR] BTC data empty"); return None

    df = pd.DataFrame({
        "btc_close":  d_btc["Close"],
        "btc_high":   d_btc["High"],
        "btc_low":    d_btc["Low"],
        "btc_volume": d_btc.get("Volume", pd.Series(dtype=float)),
    })
    raw_df = df[["btc_close", "btc_high", "btc_low", "btc_volume"]].copy()

    # 2. Macro data
    for nm, sym in _MACRO_SYMS.items():
        try:
            _d = yf.download(sym, start=fetch_start, end=fetch_end,
                              progress=False, auto_adjust=True)
            if isinstance(_d.columns, pd.MultiIndex):
                _d.columns = [c[0] for c in _d.columns]
            _d.index = pd.DatetimeIndex(_d.index).tz_localize(None).normalize()
            df[f"{nm}_close"] = _d["Close"].reindex(df.index).ffill(limit=7)
        except Exception:
            pass

    # 3. On-chain (blockchain.info)
    print("    On-chain …", end=" ", flush=True)
    ok = 0
    for m in _ONCHAIN:
        try:
            r = requests.get(
                f"https://api.blockchain.info/charts/{m}",
                params={"timespan": "3years", "format": "json", "sampled": "true"},
                timeout=20)
            vals = r.json().get("values", [])
            s = pd.Series(
                {pd.Timestamp(v["x"], unit="s").normalize(): v["y"] for v in vals},
                name=f"oc_{m.replace('-','_')}", dtype=float)
            s = s[~s.index.duplicated(keep="last")].sort_index()
            s.index = pd.DatetimeIndex(s.index).tz_localize(None)
            df[s.name] = s.reindex(df.index).ffill(limit=7)
            ok += 1
        except Exception:
            pass
    print(f"{ok}/{len(_ONCHAIN)} OK")

    # 4. Coinbase premium (same as _build_backtest_preds in app)
    _CB_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    _cb_rows: list = []
    _cb_cur = pd.Timestamp(fetch_start)
    _cb_end = pd.Timestamp(fetch_end)
    while _cb_cur <= _cb_end:
        _cb_chunk = min(_cb_cur + pd.Timedelta(days=299), _cb_end)
        try:
            _r2 = requests.get(_CB_URL, params={
                "granularity": 86400,
                "start": _cb_cur.strftime("%Y-%m-%dT00:00:00Z"),
                "end": (_cb_chunk + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z"),
            }, timeout=30)
            if _r2.status_code == 200:
                _cb_rows.extend(_r2.json())
        except Exception:
            pass
        _cb_cur = _cb_chunk + pd.Timedelta(days=1)
        _time.sleep(0.1)
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
        print(f"    Coinbase premium: {int(_prem.notna().sum())} days")
    else:
        df["cb_premium"] = df["cb_premium_ma3"] = df["cb_premium_z7"] = 0.0
        print("    Coinbase premium: unavailable (set to 0)")

    # 5. Feature engineering (identical to app's _build_backtest_preds)
    c   = df["btc_close"]; h = df["btc_high"]
    l_  = df["btc_low"];   v = df["btc_volume"]
    ret = np.log(c).diff()
    feat = pd.DataFrame(index=df.index)
    for k in [1,3,5,7,14,30]: feat[f"ret_{k}"] = ret.rolling(k).sum()
    for k in [5,10,20,30]:    feat[f"vol_{k}"] = ret.rolling(k).std()
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
        s = df[col].astype(float); sl2 = np.log(s.replace(0, np.nan))
        feat[f"{col}_d1"]  = sl2.diff(1); feat[f"{col}_d7"] = sl2.diff(7)
        feat[f"{col}_z30"] = (sl2 - sl2.rolling(30).mean())/sl2.rolling(30).std()
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

    # 6. Predictions (ensemble, no direction_head — matches app)
    fc = AD["feat_cols"]
    feat = feat.replace([np.inf, -np.inf], np.nan)
    for col in fc:
        if col not in feat.columns: feat[col] = np.nan
    F = feat[fc].dropna()
    if F.empty:
        print("  [ERR] feature matrix empty"); return None
    print(f"    CT predictions: {len(F)} rows")

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
    nd[:-1] = idx_arr[1:]; nd[-1] = idx_arr[-1] + np.timedelta64(1,"D")
    preds_df = pd.DataFrame(
        {"close_asof": c_vals, "pred_high": ph, "pred_low": pl},
        index=pd.DatetimeIndex(nd, name="target_date"),
    )
    preds_df = preds_df[~preds_df.index.duplicated(keep="last")]

    _DATA_CACHE[cache_key] = (preds_df, raw_df)
    return preds_df, raw_df


# ── Signal computation (identical to app) ────────────────────────────────────
def compute_signals(comp):
    N  = len(comp)
    ca = comp["close_asof"].values.astype(float)
    ph = comp["pred_high"].values.astype(float)
    pl = comp["pred_low"].values.astype(float)
    ah = comp["actual_high"].values.astype(float)
    al = comp["actual_low"].values.astype(float)

    err_hi = (ah - ph) / ca * 100
    err_lo = (pl - al) / ca * 100
    hi_brk = (ah > ph).astype(int)
    lo_brk = (al < pl).astype(int)

    ehma3 = np.zeros(N); elma3 = np.zeros(N)
    hb3   = np.zeros(N, dtype=int); lb3 = np.zeros(N, dtype=int)
    for i in range(N):
        s = max(0, i-2)
        ehma3[i] = np.mean(err_hi[s:i+1]); elma3[i] = np.mean(err_lo[s:i+1])
        hb3[i]   = int(np.sum(hi_brk[s:i+1])); lb3[i] = int(np.sum(lo_brk[s:i+1]))

    u1 = (ehma3 > 0.7) & (hb3 >= 2)
    d1 = (lb3 >= 2) & (elma3 > 0.5)
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
        ma30[i] = np.mean(ca[max(0, i-w+1):i+1])
    above_ma30     = ca > ma30
    ma30_slope_pos = np.zeros(N, dtype=bool)
    for i in range(5, N):
        if np.isfinite(ma30[i]) and np.isfinite(ma30[i-5]):
            ma30_slope_pos[i] = ma30[i] > ma30[i-5]
    bull_regime = above_ma30 & ma30_slope_pos

    # clean_10d: variable name from app (but checks 7 bars — named clean_7d in research script)
    clean_7d = np.zeros(N, dtype=bool)
    for i in range(N):
        lo_i = max(0, i-7)
        clean_7d[i] = not bool(np.any(d1[lo_i:i] | d2[lo_i:i]))

    roll_norm = np.array([float(np.mean(err_hi[max(0, i-29):i+1])) for i in range(N)])
    dn_score  = np.zeros(N)
    for i in range(N):
        norm = max(abs(roll_norm[i]), 0.01)
        dn_score[i] = (
            (-ehma3[i] / norm)                           * 0.30 +
            (lb3[i] / 3.0)                               * 0.30 +
            (elma3[i] / max(abs(elma3[i]), 0.10))        * 0.20 +
            float(lo_brk[i])                             * 0.20
        )
    v_rev_bar = (dn_score > 0.8) & (err_lo > 3.0)
    v_recent  = np.zeros(N, dtype=bool)
    for i in range(N):
        v_recent[i] = bool(np.any(v_rev_bar[max(0, i-2):i+1]))

    tf1_entry = u1 & (above_ma30 | clean_7d | v_recent)

    return dict(
        N=N, ca=ca, u1=u1, d1=d1, d2=d2, d3=d3,
        above_ma30=above_ma30, bull_regime=bull_regime,
        clean_7d=clean_7d, v_recent=v_recent,
        tf1_entry=tf1_entry,
    )


# ── Backtest loop — same-bar execution matching app's run_mstr/mstu_backtest ─
def run_asset_backtest(dates, asset_px, sigs, bt_start, sl_pct, cap, exclude_combined):
    N          = len(dates)
    d2         = sigs["d2"]; d3 = sigs["d3"]
    bull       = sigs["bull_regime"]
    above_ma30 = sigs["above_ma30"]
    clean_7d   = sigs["clean_7d"]
    v_recent   = sigs["v_recent"]
    tf1_entry  = sigs["tf1_entry"]

    _bt0 = max(WARMUP, int(pd.DatetimeIndex(dates).searchsorted(bt_start)))

    nav      = cap; pos = "CASH"; qty = 0.0
    e_price  = e_nav = e_date = e_trigger = None
    e_reentry = False
    stop_px   = 0.0
    from_sl   = False; bars_since_sl = 0
    trades    = []; nav_arr = np.full(N, np.nan)

    for i in range(N):
        price = asset_px[i]
        if i < _bt0:
            nav_arr[i] = cap; continue
        if not np.isfinite(price) or price <= 0:
            nav_arr[i] = qty * asset_px[i-1] if pos == "LONG" and i > 0 else nav
            continue

        if pos == "LONG":
            cur = qty * price
            if price <= stop_px:
                nav = qty * price
                trades.append(dict(
                    entry_date=e_date, entry_price=e_price, entry_nav=e_nav,
                    entry_trigger=e_trigger, exit_date=dates[i],
                    exit_price=price, exit_nav=nav,
                    pnl_pct=(price/e_price-1)*100, pnl_abs=nav-e_nav,
                    exit_signal="SL", duration_days=(dates[i]-e_date).days,
                    stop_triggered=True, was_reentry=e_reentry,
                ))
                pos = "CASH"; qty = 0.0; stop_px = 0.0; e_reentry = False
                from_sl = True; bars_since_sl = 0
            else:
                # Same-bar TF2 regime-adaptive exit (matching app logic exactly)
                should_exit = bool(d3[i] or (d2[i] and not bull[i]))
                if should_exit:
                    nav = cur
                    trades.append(dict(
                        entry_date=e_date, entry_price=e_price, entry_nav=e_nav,
                        entry_trigger=e_trigger, exit_date=dates[i],
                        exit_price=price, exit_nav=nav,
                        pnl_pct=(price/e_price-1)*100, pnl_abs=nav-e_nav,
                        exit_signal="D3" if d3[i] else "D2",
                        duration_days=(dates[i]-e_date).days,
                        stop_triggered=False, was_reentry=e_reentry,
                    ))
                    pos = "CASH"; qty = 0.0; stop_px = 0.0; e_reentry = False
                    from_sl = False
                else:
                    nav = cur
        else:
            if from_sl:
                bars_since_sl += 1
            _exit_at_i = d3[i] or (d2[i] and not bull[i])
            # SL5: bull regime → re-enter immediately; bear → 10-bar cooldown
            _sl_ok     = not from_sl or bool(bull[i]) or bars_since_sl >= 10
            _combined  = bool(above_ma30[i]) and bool(clean_7d[i])
            _can_enter = tf1_entry[i] and _sl_ok and (from_sl or not _exit_at_i)
            # Version C: block entries where BOTH above_ma30 AND clean_7d fire (unless V-rev)
            if exclude_combined and _combined and not v_recent[i]:
                _can_enter = False

            if _can_enter:
                e_reentry = bool(from_sl)
                qty = nav / price; e_price = price; e_date = dates[i]
                e_nav = nav; pos = "LONG"; stop_px = price * (1 - sl_pct)
                from_sl = False; bars_since_sl = 0
                # Trigger label — same logic as app
                if v_recent[i] and not above_ma30[i] and not clean_7d[i]:
                    e_trigger = "U1+V-reversal"
                elif above_ma30[i] and clean_7d[i]:
                    e_trigger = "U1+↑MA30+clean7d"   # COMBINED — what user is asking about
                elif above_ma30[i]:
                    e_trigger = "U1+↑MA30"
                else:
                    e_trigger = "U1+clean7d"

        nav_arr[i] = qty * price if pos == "LONG" else nav

    if pos == "LONG" and np.isfinite(asset_px[N-1]) and asset_px[N-1] > 0:
        nav_arr[N-1] = qty * asset_px[N-1]

    return trades, nav_arr


# ── Formatting helpers ────────────────────────────────────────────────────────
def fmt_pnl(v):
    return f"+{v:.1f}%" if v >= 0 else f"{v:.1f}%"


def trigger_stats(trades):
    by_trig = defaultdict(list)
    for t in trades:
        by_trig[t["entry_trigger"]].append(t["pnl_pct"])
    out = {}
    for trig, pnls in by_trig.items():
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        out[trig] = dict(
            n=len(pnls), wins=len(wins), losses=len(losses),
            win_rate=100*len(wins)/len(pnls) if pnls else 0,
            avg_pnl=float(np.mean(pnls)) if pnls else 0,
            best=float(max(pnls)) if pnls else 0,
            worst=float(min(pnls)) if pnls else 0,
        )
    return out


TRIGGER_ORDER = ["U1+↑MA30", "U1+↑MA30+clean7d", "U1+clean7d", "U1+V-reversal"]


def print_trigger_table(stats_dict):
    hdr = f"  {'Trigger':<26} {'n':>4} {'W/L':>6} {'Win%':>6} {'AvgP&L':>8} {'Best':>8} {'Worst':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr)-2))
    for trig in TRIGGER_ORDER + [k for k in stats_dict if k not in TRIGGER_ORDER]:
        if trig not in stats_dict:
            continue
        s = stats_dict[trig]
        wl = f"{s['wins']}/{s['losses']}"
        print(f"  {trig:<26} {s['n']:>4} {wl:>6} {s['win_rate']:>5.0f}%"
              f"  {fmt_pnl(s['avg_pnl']):>8} {fmt_pnl(s['best']):>8} {fmt_pnl(s['worst']):>8}")


def fishers_p(n_wins_a, n_a, n_wins_b, n_b):
    if n_a < 2 or n_b < 2:
        return float("nan")
    p_a = n_wins_a / n_a; p_b = n_wins_b / n_b
    p_pool = (n_wins_a + n_wins_b) / (n_a + n_b)
    if p_pool in (0.0, 1.0):
        return float("nan")
    se = (p_pool * (1 - p_pool) * (1/n_a + 1/n_b)) ** 0.5
    if se == 0:
        return float("nan")
    from scipy.stats import norm
    return float(2 * norm.sf(abs((p_a - p_b) / se)))


# ── Main analysis loop ────────────────────────────────────────────────────────
pooled_trades: dict = {"MSTR": defaultdict(list), "MSTU": defaultdict(list)}

for period_name, (start_iso, end_iso) in PERIODS.items():
    print(f"\n{'='*72}")
    print(f"  PERIOD: {period_name}")
    print(f"{'='*72}")

    start_dt = pd.Timestamp(start_iso)
    end_dt   = pd.Timestamp(end_iso)
    # 200-day pre-fetch warmup — ensures dist_hi_90 is fully warm at bt start
    fetch_start = (start_dt - pd.Timedelta(days=200)).strftime("%Y-%m-%d")
    fetch_end   = (end_dt   + pd.Timedelta(days=3)).strftime("%Y-%m-%d")

    ext = _build_backtest_preds(fetch_start, fetch_end)
    if ext is None:
        print("  [SKIP] data fetch failed"); continue
    preds_df, raw_df = ext

    # Filter preds to 60-day pre-period (for signal warmup) + backtest period
    pre_dt = start_dt - pd.Timedelta(days=60)
    preds  = preds_df.loc[(preds_df.index >= pre_dt) & (preds_df.index <= end_dt)].copy()
    preds["actual_high"]  = raw_df["btc_high"].reindex(preds.index).values
    preds["actual_low"]   = raw_df["btc_low"].reindex(preds.index).values
    preds["actual_close"] = raw_df["btc_close"].reindex(preds.index).values
    comp = preds.dropna(subset=["actual_high","actual_low","actual_close"]).reset_index()
    if len(comp) < WARMUP + 3:
        print("  [SKIP] insufficient bars"); continue

    dates = pd.DatetimeIndex(comp["target_date"])
    sigs  = compute_signals(comp)

    for asset_name, acfg in ASSETS.items():
        ticker  = acfg["ticker"]
        sl_pct  = acfg["sl_pct"]
        sl_lbl  = acfg["stop_label"]

        # Fetch asset prices
        try:
            d_asset = yf.download(ticker, start=fetch_start,
                                  end=(end_dt + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
                                  progress=False, auto_adjust=True)
            if isinstance(d_asset.columns, pd.MultiIndex):
                d_asset.columns = [c[0] for c in d_asset.columns]
            d_asset.index = pd.DatetimeIndex(d_asset.index).tz_localize(None).normalize()
        except Exception as e:
            print(f"  [SKIP] {asset_name}: {e}"); continue
        if d_asset.empty or "Close" not in d_asset.columns:
            print(f"  [SKIP] {asset_name}: empty"); continue

        px_raw = d_asset["Close"].sort_index()
        # MSTU backward-fill for pre-inception dates
        if ticker == "MSTU":
            INCEPTION = pd.Timestamp("2024-09-18")
            real_px = px_raw.loc[px_raw.index >= INCEPTION]
            if len(real_px):
                first_real_val = float(real_px.iloc[0])
                px_raw.loc[px_raw.index < INCEPTION] = first_real_val
        px_all = px_raw.reindex(
            pd.date_range(px_raw.index[0], max(px_raw.index[-1], end_dt), freq="D")
        ).ffill()
        asset_px = px_all.reindex(dates).ffill().bfill().values.astype(float)

        # Version A: current logic (allows combined, labels combined separately)
        trades_a, nav_a = run_asset_backtest(
            dates, asset_px, sigs, bt_start=start_dt,
            sl_pct=sl_pct, cap=INITIAL_CAPITAL, exclude_combined=False,
        )
        # Version C: block entries when BOTH above_ma30 AND clean_7d simultaneously
        trades_c, nav_c = run_asset_backtest(
            dates, asset_px, sigs, bt_start=start_dt,
            sl_pct=sl_pct, cap=INITIAL_CAPITAL, exclude_combined=True,
        )

        bh_idx  = max(WARMUP, int(dates.searchsorted(start_dt)))
        bh_px0  = asset_px[bh_idx] if np.isfinite(asset_px[bh_idx]) else 1
        bh_ret  = (asset_px[-1]/bh_px0 - 1)*100 if bh_px0 > 0 else 0

        ret_a   = (next((nav_a[j] for j in range(len(nav_a)-1,-1,-1) if np.isfinite(nav_a[j])), INITIAL_CAPITAL) / INITIAL_CAPITAL - 1)*100
        ret_c   = (next((nav_c[j] for j in range(len(nav_c)-1,-1,-1) if np.isfinite(nav_c[j])), INITIAL_CAPITAL) / INITIAL_CAPITAL - 1)*100

        ts_a = trigger_stats(trades_a)
        COMBINED = "U1+↑MA30+clean7d"
        MA30_ONLY = "U1+↑MA30"

        print(f"\n  ── {asset_name} ({sl_lbl} stop + SL5)  ──────────────────────────")
        print(f"  B&H return:           {fmt_pnl(bh_ret)}")
        print(f"  Strategy (current):   {fmt_pnl(ret_a)}   ({len(trades_a)} closed trades)")
        print(f"  Strategy (no-combined): {fmt_pnl(ret_c)}   ({len(trades_c)} closed trades)")

        print(f"\n  TRIGGER BREAKDOWN (current labelling):")
        print_trigger_table(ts_a)

        # Per-trigger trade details
        if COMBINED in ts_a:
            combined_trades = [t for t in trades_a if t["entry_trigger"] == COMBINED]
            print(f"\n  COMBINED trigger trades (U1+↑MA30+clean7d)  n={len(combined_trades)}:")
            for t in combined_trades:
                outcome = "WIN ✓" if t["pnl_pct"] > 0 else "LOSS ✗"
                print(f"    {t['entry_date'].date()} → {t['exit_date'].date()}"
                      f"  {fmt_pnl(t['pnl_pct']):>7}  {outcome}  exit={t['exit_signal']}"
                      f"  dur={t['duration_days']}d"
                      + ("  [re-entry]" if t.get("was_reentry") else ""))
        else:
            print(f"\n  No combined trigger trades in this period.")

        # Statistical test
        if COMBINED in ts_a and MA30_ONLY in ts_a:
            sc = ts_a[COMBINED]; sm = ts_a[MA30_ONLY]
            p  = fishers_p(sc["wins"], sc["n"], sm["wins"], sm["n"])
            print(f"\n  STAT TEST — combined vs MA30-only win rate:")
            print(f"    U1+↑MA30+clean7d: {sc['wins']}/{sc['n']} wins ({sc['win_rate']:.0f}%)")
            print(f"    U1+↑MA30 only:    {sm['wins']}/{sm['n']} wins ({sm['win_rate']:.0f}%)")
            pstr = f"{p:.4f}" if not np.isnan(p) else "N/A (too few samples)"
            sig  = "YES p<0.05" if (not np.isnan(p) and p < 0.05) else ("marginal p<0.10" if (not np.isnan(p) and p < 0.10) else "NO")
            print(f"    p-value: {pstr}  →  Statistically significant: {sig}")

        # Impact of blocking combined
        print(f"\n  IMPACT of blocking combined entries:")
        blocked = [t for t in trades_a if t["entry_trigger"] == COMBINED]
        print(f"    Trades blocked: {len(blocked)}")
        for t in blocked:
            print(f"      {t['entry_date'].date()}  {fmt_pnl(t['pnl_pct'])}  ({t['exit_signal']})")
        delta = ret_c - ret_a
        print(f"    Return: {fmt_pnl(ret_c)} vs current {fmt_pnl(ret_a)}"
              f"  Δ={'+' if delta>=0 else ''}{delta:.1f}pp  "
              f"({'BETTER' if delta > 1 else 'WORSE' if delta < -1 else 'NEUTRAL'})")

        # Accumulate for pooled analysis
        for t in trades_a:
            pooled_trades[asset_name][t["entry_trigger"]].append(t["pnl_pct"])

# ── Cross-period pooled summary ───────────────────────────────────────────────
print(f"\n\n{'='*72}")
print("  CROSS-PERIOD POOLED — ALL TRIGGER TRADES COMBINED")
print(f"{'='*72}")

for asset_name in ASSETS:
    print(f"\n  {asset_name}:")
    hdr = f"  {'Trigger':<26} {'n':>4} {'W/L':>6} {'Win%':>6} {'AvgP&L':>8} {'Best':>8} {'Worst':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr)-2))
    all_trigs = pooled_trades[asset_name]
    for trig in TRIGGER_ORDER + [k for k in all_trigs if k not in TRIGGER_ORDER]:
        pnls = all_trigs.get(trig, [])
        if not pnls:
            continue
        wins = [p for p in pnls if p > 0]
        loss = [p for p in pnls if p <= 0]
        wl = f"{len(wins)}/{len(loss)}"
        print(f"  {trig:<26} {len(pnls):>4} {wl:>6} {100*len(wins)/len(pnls):>5.0f}%"
              f"  {fmt_pnl(np.mean(pnls)):>8} {fmt_pnl(max(pnls)):>8} {fmt_pnl(min(pnls)):>8}")

    COMBINED  = "U1+↑MA30+clean7d"
    MA30_ONLY = "U1+↑MA30"
    if COMBINED in all_trigs and MA30_ONLY in all_trigs:
        pc = all_trigs[COMBINED]; pm = all_trigs[MA30_ONLY]
        wc = len([p for p in pc if p > 0]); wm = len([p for p in pm if p > 0])
        p  = fishers_p(wc, len(pc), wm, len(pm))
        pstr = f"{p:.4f}" if not np.isnan(p) else "N/A"
        print(f"\n  Pooled stat test — combined vs MA30-only: p={pstr}")
        if not np.isnan(p):
            print(f"  {'Statistically significant (p<0.05)' if p < 0.05 else 'NOT statistically significant'}")

print("\nDONE.\n")
