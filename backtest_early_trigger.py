#!/usr/bin/env python3
"""
Early U1 Trigger Analysis  (v2 — corrected framing)
=====================================================
Question: When the pre-conditions for U1 are partially met at yesterday's close —
  • err_hi_ma3 > 0.7%   (3-day avg high-side error already elevated)
  • hi_breaks_3d == 1   (only 1 of last 3 bars has broken pred_high)
...and today's INTRADAY price breaks through pred_high, does entering EARLY
(at the breakout level, intraday) outperform the standard convention of waiting
for today's bar to close and executing at next-day's close?

CORRECTED framing (v2)
-----------------------
hi_break is computed from actual_high (the bar's intraday peak), NOT from close.
So any intraday breach of pred_high makes hi_brk[T] = 1 regardless of where the
bar settles. This means the relevant split is NOT "close > pred_hi vs close ≤ pred_hi"
but instead:

  CASE A/B  (SYNC cases): The intraday break causes hb3[T] to reach 2 (because the
            prior single break was within T-2 or T-1). Standard U1 fires at bar-T close
            (u1[T] = True). Both EARLY and STANDARD take this trade — question is purely
            about execution price.

  CASE C    (EXTRA cases): The intraday break causes hb3[T] to stay at 1 (prior break
            was at T-3, now rolled off). Standard U1 does NOT fire. EARLY would take
            a trade that STANDARD never takes.

Four modes compared
-------------------
  STANDARD    : u1[T] fires at bar-T close → execute T+1 close.  (baseline)

  EARLY-SYNC  : Fires early ONLY for CASE A/B (where u1[T] is True at close).
                Entry at max(open_T, pred_high_T) — the breakout price proxy.
                Same set of trades as STANDARD, 1 bar earlier execution.
                Pure "does earlier entry price improve performance?" test.

  EARLY-EXTRA : Fires early ONLY for CASE C (where u1[T] would NOT fire at close).
                Takes trades that STANDARD misses entirely.

  EARLY-ALL   : Combines EARLY-SYNC + EARLY-EXTRA.

Exit logic is identical across all modes: D2 or D3 fires at bar close → exit next bar.

Diagnostic
----------
Per candidate bar: whether STANDARD also fires (SYNC vs EXTRA), entry price advantage
vs T+1 close, and forward returns from both entry points.

Periods
-------
  OOS Bear   Sep 2025 → May 2026  (BTC −33%, fully out-of-sample)
  Bull IS    Sep 2024 → Sep 2025  (BTC +93%, in-sample — directional cross-check only)
"""

import sys, os, warnings, joblib, requests
warnings.filterwarnings("ignore")
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import yfinance as yf

# ─── Load model ───────────────────────────────────────────────────────────────
_CT_PATH = _ROOT / "models" / "inference_assets_ct.joblib"
if not _CT_PATH.exists():
    _CT_PATH = _ROOT / "artifacts" / "artifacts.pkl"
print(f"Loading model: {_CT_PATH}")
AD = joblib.load(str(_CT_PATH))

_MACRO_SYMS = {
    "eth": "ETH-USD", "spx": "^GSPC", "ndx": "^IXIC",
    "vix": "^VIX",   "gold": "GC=F",  "dxy": "DX-Y.NYB", "tnx": "^TNX",
}
_ONCHAIN_METRICS = [
    "hash-rate", "difficulty", "n-transactions", "miners-revenue",
    "n-unique-addresses", "transaction-fees-usd", "mempool-size",
    "estimated-transaction-volume-usd", "market-cap",
    "avg-block-size", "cost-per-transaction",
]
_CB_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"


# ─── Data helpers ─────────────────────────────────────────────────────────────
def _yf_single(sym, start, end):
    d = yf.download(sym, start=start, end=end, progress=False, auto_adjust=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] for c in d.columns]
    d.index = pd.DatetimeIndex(d.index).tz_localize(None).normalize()
    return d


def _fetch_onchain(metric):
    col = f"oc_{metric.replace('-','_')}"
    try:
        r = requests.get(
            f"https://api.blockchain.info/charts/{metric}",
            params={"timespan": "3years", "format": "json", "sampled": "true"},
            timeout=25,
        )
        r.raise_for_status()
        vals = r.json().get("values", [])
        s = pd.Series(
            {pd.Timestamp(d["x"], unit="s").normalize(): d["y"] for d in vals},
            name=col, dtype=float,
        )
        return s[~s.index.duplicated(keep="last")].sort_index()
    except Exception as e:
        print(f"    [warn] {metric}: {e}")
        return pd.Series(dtype=float, name=col)


def _fetch_coinbase_daily(start_str, end_str):
    import time as _time
    rows = []
    cur = pd.Timestamp(start_str)
    end = pd.Timestamp(end_str)
    while cur <= end:
        chunk_end = min(cur + pd.Timedelta(days=299), end)
        try:
            r = requests.get(_CB_URL, params={
                "granularity": 86400,
                "start": cur.isoformat() + "Z",
                "end":   (chunk_end + pd.Timedelta(days=1)).isoformat() + "Z",
            }, timeout=30)
            if r.status_code == 200:
                rows.extend(r.json())
        except Exception as e:
            print(f"    [warn] coinbase: {e}")
        cur = chunk_end + pd.Timedelta(days=1)
        _time.sleep(0.35)
    if not rows:
        return pd.Series(dtype=float, name="coinbase_close")
    tmp = pd.DataFrame(rows, columns=["ts","low","high","open","close","volume"])
    tmp["date"] = pd.to_datetime(tmp["ts"], unit="s").dt.normalize()
    tmp = tmp.drop_duplicates("date").set_index("date").sort_index()
    return tmp["close"].rename("coinbase_close")


def fetch_daily_data(fetch_start, fetch_end):
    print(f"  Fetching {fetch_start} → {fetch_end} …")
    btc = _yf_single("BTC-USD", fetch_start, fetch_end)
    frames = {
        "btc_open":   btc["Open"],
        "btc_close":  btc["Close"],
        "btc_high":   btc["High"],
        "btc_low":    btc["Low"],
        "btc_volume": btc["Volume"],
    }
    for nm, sym in _MACRO_SYMS.items():
        try:
            frames[f"{nm}_close"] = _yf_single(sym, fetch_start, fetch_end)["Close"]
        except Exception as e:
            print(f"    [warn] {sym}: {e}")

    df = pd.DataFrame(frames).sort_index().ffill(limit=5)
    df.index.name = "date"

    print("  On-chain …", end=" ", flush=True)
    ok = 0
    for m in _ONCHAIN_METRICS:
        s = _fetch_onchain(m)
        if not s.empty:
            s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
            df[s.name] = s.reindex(df.index).ffill(limit=7)
            ok += 1
    print(f"{ok}/{len(_ONCHAIN_METRICS)} ok")

    print("  Coinbase premium …", end=" ", flush=True)
    cb = _fetch_coinbase_daily(fetch_start, fetch_end)
    if not cb.empty:
        df["coinbase_close"] = cb.reindex(df.index).ffill(limit=3)
        print(f"{df['coinbase_close'].notna().sum()} rows")
    else:
        print("unavailable (zero-fill)")
    return df


# ─── CT predictions (vectorised) ──────────────────────────────────────────────
def build_ct_predictions(df):
    c = df["btc_close"]; h = df["btc_high"]; l_ = df["btc_low"]; v = df["btc_volume"]
    ret = np.log(c).diff()
    f = pd.DataFrame(index=df.index)
    for k in [1,3,5,7,14,30]: f[f"ret_{k}"] = ret.rolling(k).sum()
    for k in [5,10,20,30]:    f[f"vol_{k}"] = ret.rolling(k).std()
    prev_c = c.shift(1)
    tr = pd.concat([(h-l_),(h-prev_c).abs(),(l_-prev_c).abs()],axis=1).max(axis=1)
    for k in [7,14,30]: f[f"atr_{k}"] = tr.rolling(k).mean()/c
    f["range_today"]  = (h-l_)/c
    f["range_ma7"]    = ((h-l_)/c).rolling(7).mean()
    f["range_ma30"]   = ((h-l_)/c).rolling(30).mean()
    f["range_std30"]  = ((h-l_)/c).rolling(30).std()
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    f["rsi_14"]   = 100 - 100/(1 + gain/loss.replace(0, np.nan))
    e12 = c.ewm(span=12,adjust=False).mean(); e26 = c.ewm(span=26,adjust=False).mean()
    macd = e12 - e26
    f["macd"]     = macd/c
    f["macd_sig"] = macd.ewm(span=9,adjust=False).mean()/c
    f["macd_hist"]= (macd - macd.ewm(span=9,adjust=False).mean())/c
    ma20 = c.rolling(20).mean(); sd20 = c.rolling(20).std()
    f["bb_width"]   = (4*sd20)/ma20
    f["dist_hi_30"] = c/c.rolling(30).max()-1
    f["dist_lo_30"] = c/c.rolling(30).min()-1
    f["dist_hi_90"] = c/c.rolling(90).max()-1
    f["vol_chg_1"]  = np.log(v).diff()
    f["vol_z_20"]   = (np.log(v)-np.log(v).rolling(20).mean())/np.log(v).rolling(20).std()
    f["vol_ma_ratio"]= v/v.rolling(20).mean()
    dow = df.index.dayofweek
    for i in range(6): f[f"dow_{i}"] = (dow==i).astype(float)
    for nm in ["spx","ndx","vix","gold","dxy","tnx","eth"]:
        col = f"{nm}_close"
        if col not in df.columns: continue
        lr = np.log(df[col]).diff()
        for k in [1,5,20]: f[f"{nm}_ret_{k}"] = lr.rolling(k).sum()
        f[f"{nm}_vol_20"] = lr.rolling(20).std()
    for cn, cc in [("spx","spx_close"),("ndx","ndx_close"),
                   ("gold","gold_close"),("dxy","dxy_close")]:
        if cc in df.columns:
            f[f"btc_{cn}_corr_30"] = ret.rolling(30).corr(np.log(df[cc]).diff())
    for col in [x for x in df.columns if x.startswith("oc_")]:
        s = df[col].astype(float)
        sl = np.log(s.replace(0, np.nan))
        f[f"{col}_d1"]  = sl.diff(1)
        f[f"{col}_d7"]  = sl.diff(7)
        f[f"{col}_z30"] = (sl - sl.rolling(30).mean())/sl.rolling(30).std()
    nh, nl = h.shift(-1), l_.shift(-1)
    y_hi = (nh-c)/c; y_lo = (c-nl)/c
    f["y_hi_ema3"] = y_hi.shift(1).ewm(span=3,adjust=False).mean()
    f["y_lo_ema3"] = y_lo.shift(1).ewm(span=3,adjust=False).mean()
    f["y_hi_ema7"] = y_hi.shift(1).ewm(span=7,adjust=False).mean()
    f["y_lo_ema7"] = y_lo.shift(1).ewm(span=7,adjust=False).mean()
    prev_3_hi = h.shift(1).rolling(3).max(); prev_3_lo = l_.shift(1).rolling(3).min()
    f["above_3d_high"]  = (c > prev_3_hi).astype(float)
    f["below_3d_low"]   = (c < prev_3_lo).astype(float)
    f["bo_strength_up"] = (c/prev_3_hi - 1).clip(lower=0)
    f["bo_strength_dn"] = (1 - c/prev_3_lo).clip(lower=0)
    _yla = y_hi.shift(1); _ylab = y_lo.shift(1)
    f["y_hi_surprise"] = _yla  - _yla.ewm(span=7,adjust=False).mean()
    f["y_lo_surprise"] = _ylab - _ylab.ewm(span=7,adjust=False).mean()
    neg_ret = ret.clip(upper=0)
    f["dn_vol_5"]  = neg_ret.rolling(5).std()
    f["dn_vol_20"] = neg_ret.rolling(20).std()
    sma50 = c.rolling(50).mean()
    f["below_sma50"]    = (c < sma50).astype(float)
    f["below_sma50_5d"] = f["below_sma50"].rolling(5).min().fillna(0)

    # Coinbase premium — required by model; zero-fill when unavailable
    if "coinbase_close" in df.columns and df["coinbase_close"].notna().any():
        _cb   = df["coinbase_close"]
        _prem = (_cb - c) / c * 100
        f["cb_premium"]     = _prem
        f["cb_premium_ma3"] = _prem.rolling(3).mean()
        f["cb_premium_z7"]  = (
            (_prem - _prem.rolling(7).mean()) / _prem.rolling(7).std()
        )
    else:
        f["cb_premium"] = f["cb_premium_ma3"] = f["cb_premium_z7"] = 0.0

    fc = AD["feat_cols"]
    f  = f.replace([np.inf,-np.inf], np.nan)
    for col in fc:
        if col not in f.columns: f[col] = np.nan
    F = f[fc].dropna()

    if F.empty:
        raise RuntimeError("Feature matrix empty after dropna — check data quality")

    if AD.get("ensemble") and AD.get("constituents"):
        yhi = np.mean([co["m_hi"].predict(F) for co in AD["constituents"]], axis=0)
        ylo = np.mean([co["m_lo"].predict(F) for co in AD["constituents"]], axis=0)
    else:
        yhi = AD["hi_model"].predict(F)
        ylo = AD["lo_model"].predict(F)

    c_vals       = c.reindex(F.index).values
    pred_hi_vals = c_vals * (1 + np.clip(yhi, 0, None))
    pred_lo_vals = c_vals * (1 - np.clip(ylo, 0, None))

    idx_arr    = np.asarray(F.index, dtype="datetime64[ns]")
    next_dates = np.empty(len(F), dtype="datetime64[ns]")
    next_dates[:-1] = idx_arr[1:]
    next_dates[-1]  = idx_arr[-1] + np.timedelta64(1,"D")

    result = pd.DataFrame(
        {"close_asof": c_vals, "pred_high": pred_hi_vals, "pred_low": pred_lo_vals},
        index=pd.DatetimeIndex(next_dates, name="target_date"),
    )
    return result[~result.index.duplicated(keep="last")]


# ─── Signal computation ────────────────────────────────────────────────────────
def compute_signals(comp):
    N = len(comp)
    c_asof  = comp["close_asof"].values.astype(float)
    pred_hi = comp["pred_high"].values.astype(float)
    pred_lo = comp["pred_low"].values.astype(float)
    act_hi  = comp["actual_high"].values.astype(float)
    act_lo  = comp["actual_low"].values.astype(float)

    # hi_break uses actual_high (intraday peak), NOT close
    err_hi = (act_hi - pred_hi) / c_asof * 100
    err_lo = (pred_lo - act_lo) / c_asof * 100
    hi_brk = (act_hi > pred_hi).astype(int)
    lo_brk = (act_lo < pred_lo).astype(int)

    ehma3 = np.zeros(N); elma3 = np.zeros(N)
    hb3   = np.zeros(N, dtype=int); lb3 = np.zeros(N, dtype=int)
    for i in range(N):
        s = max(0, i-2)
        ehma3[i] = np.mean(err_hi[s:i+1])
        elma3[i] = np.mean(err_lo[s:i+1])
        hb3[i]   = int(np.sum(hi_brk[s:i+1]))
        lb3[i]   = int(np.sum(lo_brk[s:i+1]))

    u1 = (ehma3 > 0.7) & (hb3 >= 2)   # fires at bar close; uses actual_high for hb3
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
        ma30[i] = np.mean(c_asof[max(0, i-w+1):i+1])
    above_ma30 = c_asof > ma30

    ma30_slope_pos = np.zeros(N, dtype=bool)
    for i in range(5, N):
        if np.isfinite(ma30[i]) and np.isfinite(ma30[i-5]):
            ma30_slope_pos[i] = ma30[i] > ma30[i-5]
    bull_regime = above_ma30 & ma30_slope_pos

    clean_10d = np.zeros(N, dtype=bool)
    for i in range(N):
        lo_i = max(0, i-7)
        clean_10d[i] = not bool(np.any(d1[lo_i:i] | d2[lo_i:i]))

    # Pre-state for early trigger: at end of bar i-1, setup is partially met.
    # If today's actual_high > pred_hi[i], the early trigger fires intraday.
    # We classify by whether STANDARD also fires at bar i close (u1[i]).
    early_setup = np.zeros(N, dtype=bool)   # pre-state satisfied
    for i in range(1, N):
        early_setup[i] = (ehma3[i-1] > 0.7) and (hb3[i-1] == 1)

    return dict(
        u1=u1, d1=d1, d2=d2, d3=d3,
        above_ma30=above_ma30, bull_regime=bull_regime,
        clean_10d=clean_10d, early_setup=early_setup,
        c_asof=c_asof, ehma3=ehma3, hb3=hb3,
        hi_brk=hi_brk, lo_brk=lo_brk,
        pred_hi=pred_hi, pred_lo=pred_lo,
        err_hi=err_hi, err_lo=err_lo,
    )


# ─── Backtest engine ──────────────────────────────────────────────────────────
# EARLY-SYNC  : pre-state met + intraday break + STANDARD also fires at close
#               → same trade as STANDARD but entered at breakout price on bar T
# EARLY-EXTRA : pre-state met + intraday break + STANDARD does NOT fire at close
#               → Case C: extra trade that STANDARD misses
# EARLY-ALL   : EARLY-SYNC ∪ EARLY-EXTRA
_MODES = ("STANDARD", "EARLY-SYNC", "EARLY-EXTRA", "EARLY-ALL")


def run_backtest(df_raw, preds, start_iso, end_iso,
                 mode="STANDARD", initial_capital=100_000.0):
    WARMUP = 35
    start_dt = pd.Timestamp(start_iso)
    end_dt   = pd.Timestamp(end_iso)

    p = preds.loc[(preds.index >= start_dt - pd.Timedelta(days=45)) &
                  (preds.index <= end_dt)].copy()
    p["actual_high"]  = df_raw["btc_high"].reindex(p.index).values
    p["actual_low"]   = df_raw["btc_low"].reindex(p.index).values
    p["actual_close"] = df_raw["btc_close"].reindex(p.index).values
    p["actual_open"]  = df_raw["btc_open"].reindex(p.index).values
    comp = (p.dropna(subset=["actual_high","actual_low","actual_close","actual_open"])
             .reset_index())
    comp = comp[comp["target_date"] >= start_dt].reset_index(drop=True)
    N = len(comp)
    if N < WARMUP + 3:
        return None

    dates   = pd.DatetimeIndex(comp["target_date"])
    sigs    = compute_signals(comp)
    u1          = sigs["u1"]
    d1          = sigs["d1"]; d2 = sigs["d2"]; d3 = sigs["d3"]
    above_ma30  = sigs["above_ma30"]
    clean_10d   = sigs["clean_10d"]
    early_setup = sigs["early_setup"]
    pred_hi     = sigs["pred_hi"]

    act_hi = comp["actual_high"].values.astype(float)
    act_cl = comp["actual_close"].values.astype(float)
    act_op = comp["actual_open"].values.astype(float)

    # Early entry price proxy: max(open_T, pred_high_T)
    # Rationale: if open already above pred_hi → gapped through, enter at open;
    # otherwise enter exactly at the breakout threshold level.
    early_px = np.maximum(act_op, pred_hi)

    nav     = initial_capital
    pos     = "CASH"
    btc_qty = 0.0
    e_price = e_nav = e_date = e_trigger = None
    trades  = []
    nav_arr = np.full(N, np.nan)

    for i in range(N):
        if i < WARMUP:
            nav_arr[i] = initial_capital
            continue

        si = i - 1  # prior bar index for signal read

        # ── Signals ──────────────────────────────────────────────────────────
        std_signal = bool(u1[si] and (above_ma30[si] or clean_10d[si])) if si >= 0 else False
        std_exit   = bool(d2[si] or d3[si]) if si >= 0 else False

        if mode == "STANDARD":
            _enter = std_signal
            _exit  = std_exit
            exec_price = act_cl[i]
            trig_label = "STD-U1"

        else:
            intraday_break  = bool(act_hi[i] > pred_hi[i])
            gate_ok         = bool(above_ma30[i] or clean_10d[i])
            # SYNC: same trade as STANDARD would take at bar-T close (u1 + gate both True)
            early_std_fires  = bool(early_setup[i] and intraday_break and u1[i] and gate_ok)
            # EXTRA: early fires but STANDARD would NOT fire at bar-T close (u1 or gate fails)
            early_std_misses = bool(early_setup[i] and intraday_break and not (u1[i] and gate_ok))

            if mode == "EARLY-SYNC":
                fire_early = early_std_fires
            elif mode == "EARLY-EXTRA":
                fire_early = early_std_misses
            else:  # EARLY-ALL
                fire_early = early_std_fires or early_std_misses

            _enter = fire_early or (std_signal and not fire_early)
            _exit  = std_exit
            exec_price = early_px[i] if fire_early else act_cl[i]
            trig_label = "EARLY" if fire_early else "STD-U1"

        # ── Position update ───────────────────────────────────────────────────
        if pos == "LONG":
            if _exit:
                nav = btc_qty * act_cl[i]
                trades.append(dict(
                    entry_date=e_date, entry_price=e_price, entry_nav=e_nav,
                    entry_trigger=e_trigger,
                    exit_date=dates[i], exit_price=act_cl[i], exit_nav=nav,
                    pnl_pct=(act_cl[i]/e_price - 1)*100,
                    pnl_abs=nav - e_nav,
                    exit_signal="D3" if (si >= 0 and d3[si]) else "D2",
                    duration_days=(dates[i]-e_date).days,
                ))
                pos = "CASH"; btc_qty = 0.0
            else:
                nav = btc_qty * act_cl[i]
        else:
            if _enter and not _exit:
                btc_qty = nav / exec_price
                e_price = exec_price; e_date = dates[i]
                e_nav = nav; pos = "LONG"
                e_trigger = trig_label

        nav_arr[i] = btc_qty * act_cl[i] if pos == "LONG" else nav

    if pos == "LONG":
        nav_arr[N-1] = btc_qty * act_cl[N-1]

    nav_series = pd.Series(nav_arr[WARMUP:], index=dates[WARMUP:]).ffill()
    bh_series  = pd.Series(
        initial_capital * act_cl[WARMUP:] / act_cl[WARMUP],
        index=dates[WARMUP:],
    )
    final_nav = float(nav_series.iloc[-1])
    final_bh  = float(bh_series.iloc[-1])
    wins = [t for t in trades if t["pnl_pct"] > 0]
    days_in  = sum(t["duration_days"] for t in trades)
    tot_days = max(1, (dates[N-1] - dates[WARMUP]).days)
    rm = nav_series.cummax()
    max_dd = float(((nav_series - rm) / rm * 100).min())

    return dict(
        mode=mode, start=str(dates[WARMUP].date()), end=str(dates[N-1].date()),
        final_nav=final_nav, final_bh=final_bh,
        strat_ret=(final_nav/initial_capital - 1)*100,
        bh_ret=(final_bh/initial_capital - 1)*100,
        alpha_abs=final_nav - final_bh,
        n_trades=len(trades), n_wins=len(wins),
        win_rate=100*len(wins)/len(trades) if trades else 0.0,
        max_dd=max_dd, time_in=100*days_in/tot_days,
        trades=trades, nav_series=nav_series, bh_series=bh_series,
        open_pos=pos == "LONG",
    )


# ─── Per-event diagnostic ─────────────────────────────────────────────────────
def early_trigger_diagnostics(df_raw, preds, start_iso, end_iso):
    """
    For each bar where early trigger fires, shows:
      • Whether STANDARD also fires (SYNC) or not (EXTRA)
      • Entry price: early (max(open_T, pred_hi_T)) vs standard (close_T+1)
      • Price advantage: + means early gets cheaper price
      • Forward returns: 1d/3d/5d from each entry point
    """
    start_dt = pd.Timestamp(start_iso)
    end_dt   = pd.Timestamp(end_iso)

    p = preds.loc[(preds.index >= start_dt - pd.Timedelta(days=45)) &
                  (preds.index <= end_dt)].copy()
    p["actual_high"]  = df_raw["btc_high"].reindex(p.index).values
    p["actual_low"]   = df_raw["btc_low"].reindex(p.index).values
    p["actual_close"] = df_raw["btc_close"].reindex(p.index).values
    p["actual_open"]  = df_raw["btc_open"].reindex(p.index).values
    comp = (p.dropna(subset=["actual_high","actual_low","actual_close","actual_open"])
             .reset_index())
    comp = comp[comp["target_date"] >= start_dt].reset_index(drop=True)
    N = len(comp)

    sigs = compute_signals(comp)
    early_setup = sigs["early_setup"]
    pred_hi     = sigs["pred_hi"]
    u1          = sigs["u1"]
    above_ma30  = sigs["above_ma30"]
    clean_10d   = sigs["clean_10d"]

    act_hi = comp["actual_high"].values.astype(float)
    act_cl = comp["actual_close"].values.astype(float)
    act_op = comp["actual_open"].values.astype(float)
    dates  = pd.DatetimeIndex(comp["target_date"])

    WARMUP = 35
    rows = []
    for i in range(WARMUP, N-5):
        if not early_setup[i]:
            continue
        if not (act_hi[i] > pred_hi[i]):   # no intraday break → early trigger can't fire
            continue

        gate_ok   = bool(above_ma30[i] or clean_10d[i])
        std_fires = bool(u1[i] and gate_ok)   # same gate as STANDARD strategy
        case      = "SYNC" if std_fires else "EXTRA"
        early_entry  = max(act_op[i], pred_hi[i])
        # Standard execution price: T+1 close (same as what STANDARD strategy uses)
        std_entry    = act_cl[i+1]

        # Price advantage: positive = early entry is cheaper (better)
        px_adv = (std_entry / early_entry - 1) * 100

        # Forward returns from early entry: measured from early_entry price
        fwd_e = {h: (act_cl[min(i+h, N-1)] / early_entry - 1) * 100 for h in [1,3,5]}
        # Forward returns from standard entry: measured from std_entry price
        fwd_s = {h: (act_cl[min(i+1+h, N-1)] / std_entry - 1) * 100 for h in [1,3,5]}

        rows.append(dict(
            date=str(dates[i].date()),
            case=case,
            std_entry_gate=f"{'↑MA30' if above_ma30[i] else ''}{'c10d' if clean_10d[i] else ''}".strip() or "none",
            open_T=round(act_op[i], 0),
            pred_hi_T=round(pred_hi[i], 0),
            close_T=round(act_cl[i], 0),
            early_entry=round(early_entry, 0),
            std_entry=round(std_entry, 0),
            px_adv=round(px_adv, 2),
            fwd_1d_early=round(fwd_e[1], 2),
            fwd_1d_std=round(fwd_s[1], 2),
            fwd_3d_early=round(fwd_e[3], 2),
            fwd_3d_std=round(fwd_s[3], 2),
            fwd_5d_early=round(fwd_e[5], 2),
            fwd_5d_std=round(fwd_s[5], 2),
        ))

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
PERIODS = {
    "OOS Bear (Sep25→May26, BTC −33%)": {
        "start": "2025-09-17", "end": "2026-05-27",
        "fetch_start": "2025-06-01", "fetch_end": "2026-05-28",
    },
    "Bull IS (Sep24→Sep25, BTC +93%)": {
        "start": "2024-09-17", "end": "2025-09-17",
        "fetch_start": "2024-06-01", "fetch_end": "2025-09-20",
    },
}

all_results = {}

for period_name, pcfg in PERIODS.items():
    print(f"\n{'═'*72}")
    print(f"PERIOD: {period_name}")
    print(f"{'═'*72}")

    df_raw = fetch_daily_data(pcfg["fetch_start"], pcfg["fetch_end"])
    print("  Building CT predictions …", end=" ", flush=True)
    preds  = build_ct_predictions(df_raw)
    print(f"→ {len(preds)} bars")

    period_res = {}
    for mode in _MODES:
        r = run_backtest(df_raw, preds, pcfg["start"], pcfg["end"], mode=mode)
        period_res[mode] = r
    all_results[period_name] = period_res

    # ── Performance table ──────────────────────────────────────────────────
    print(f"\n  {'─'*82}")
    print(f"  {'Mode':<14} {'NAV':>10} {'Ret%':>8} {'B&H%':>8} {'Alpha$':>12} "
          f"{'Trd':>5} {'Win%':>6} {'MaxDD%':>8} {'TiM%':>6}")
    print(f"  {'─'*82}")
    for mode, r in period_res.items():
        if r is None:
            print(f"  {mode:<14}  ERROR"); continue
        flag = " ◄ BEST" if r["alpha_abs"] == max(
            rv["alpha_abs"] for rv in period_res.values() if rv) else ""
        print(
            f"  {mode:<14} ${r['final_nav']:>9,.0f} {r['strat_ret']:>+7.1f}% "
            f"{r['bh_ret']:>+7.1f}% ${r['alpha_abs']:>+11,.0f} "
            f"{r['n_trades']:>4}  {r['win_rate']:>5.0f}% {r['max_dd']:>+7.1f}% "
            f"{r['time_in']:>5.0f}%{flag}"
        )

    # ── Candidate event diagnostic ─────────────────────────────────────────
    print(f"\n  ── Early Trigger Candidates ({period_name}) ──")
    diag = early_trigger_diagnostics(df_raw, preds, pcfg["start"], pcfg["end"])

    if diag.empty:
        print("  No candidates found.")
    else:
        sync  = diag[diag["case"] == "SYNC"]
        extra = diag[diag["case"] == "EXTRA"]
        print(f"\n  Total candidates: {len(diag)}  │  "
              f"SYNC (Standard also fires): {len(sync)}  │  "
              f"EXTRA (Standard misses): {len(extra)}")

        # Entry price advantage summary
        adv = diag["px_adv"]
        sadv = sync["px_adv"]
        print(f"\n  Entry price advantage (std_entry/early_entry − 1, + = early cheaper):")
        print(f"    ALL  : mean {adv.mean():+.2f}%  median {adv.median():+.2f}%  "
              f">0 (early cheaper): {(adv>0).sum()}/{len(diag)}")
        if not sync.empty:
            print(f"    SYNC : mean {sadv.mean():+.2f}%  median {sadv.median():+.2f}%  "
                  f">0: {(sadv>0).sum()}/{len(sync)}")

        # Forward-return breakdown by case
        for case_label, sub in [("SYNC  (same trades, earlier entry)", sync),
                                  ("EXTRA (Case C — extra trades only)", extra)]:
            if sub.empty: continue
            print(f"\n  Forward returns — {case_label} (n={len(sub)}):")
            print(f"  {'Horizon':<8} {'Early mean':>11} {'Std mean':>11} "
                  f"{'Delta':>9}  {'Early>Std':>10}")
            for h in [1, 3, 5]:
                em  = sub[f"fwd_{h}d_early"].mean()
                sm  = sub[f"fwd_{h}d_std"].mean()
                win = (sub[f"fwd_{h}d_early"] > sub[f"fwd_{h}d_std"]).sum()
                print(f"  {h}d       {em:>+10.2f}% {sm:>+10.2f}% "
                      f"{em-sm:>+8.2f}%  {win}/{len(sub)}")

        # Per-event table
        print(f"\n  Per-event table (entry price: EARLY = max(open,pred_hi), STD = close_T+1):")
        print(f"  {'Date':12} {'Case':5} {'OpenT':>8} {'PredHi':>8} {'CloseT':>8} "
              f"{'EarlyPx':>9} {'StdPx':>9} {'Adv%':>6} "
              f"{'1d-E%':>7} {'1d-S%':>7} {'3d-E%':>7} {'3d-S%':>7}")
        print(f"  {'─'*96}")
        for _, row in diag.iterrows():
            print(
                f"  {row['date']:12} {row['case']:5} "
                f"{row['open_T']:>8,.0f} {row['pred_hi_T']:>8,.0f} {row['close_T']:>8,.0f} "
                f"{row['early_entry']:>9,.0f} {row['std_entry']:>9,.0f} "
                f"{row['px_adv']:>+5.2f}% "
                f"{row['fwd_1d_early']:>+6.2f}% {row['fwd_1d_std']:>+6.2f}% "
                f"{row['fwd_3d_early']:>+6.2f}% {row['fwd_3d_std']:>+6.2f}%"
            )

    # ── Trade logs ─────────────────────────────────────────────────────────
    for mode, r in period_res.items():
        if r is None or not r["trades"]: continue
        print(f"\n  Trade log — {mode}")
        print(f"  {'#':>2}  {'Entry':12} {'Exit':12} {'Trig':10} "
              f"{'EntryPx':>9} {'ExitPx':>9} {'P&L%':>7} {'Days':>5}")
        for j, t in enumerate(r["trades"], 1):
            mk = "✓" if t["pnl_pct"] > 0 else "✗"
            print(
                f"  {j:>2}  {str(t['entry_date'].date()):12} "
                f"{str(t['exit_date'].date()):12} "
                f"{t.get('entry_trigger','?'):<10} "
                f"{t['entry_price']:>9,.0f} {t['exit_price']:>9,.0f} "
                f"{mk}{t['pnl_pct']:>+5.1f}% {t['duration_days']:>5}d"
            )
        if r["open_pos"]:
            print("  [open position at period end]")

# ─── Cross-period summary ──────────────────────────────────────────────────────
print(f"\n{'═'*72}")
print("CROSS-PERIOD SUMMARY — Does early trigger improve performance?")
print(f"{'═'*72}\n")

pnames = list(all_results.keys())
short  = ["OOS-Bear", "Bull-IS"]
hdr    = "  " + f"{'Mode':<14}"
for sn in short:
    hdr += f"  {sn+' Alpha$':>15}  {sn+' NAV%':>9}"
print(hdr)
print("  " + "─"*70)

for mode in _MODES:
    row = f"  {mode:<14}"
    for pn in pnames:
        r = all_results[pn].get(mode)
        row += f"  ${r['alpha_abs']:>+14,.0f}  {r['strat_ret']:>+8.1f}%" if r else f"  {'N/A':>15}  {'N/A':>9}"
    print(row)

print(f"\n  {'─'*70}")
print("  Verdict by period:")
for pn, sn in zip(pnames, short):
    rs = {m: all_results[pn].get(m) for m in _MODES}
    std = rs["STANDARD"]["alpha_abs"]
    print(f"\n  {sn}:")
    for mode in _MODES[1:]:
        r = rs[mode]
        delta = r["alpha_abs"] - std
        verdict = "BETTER" if delta > 0 else "WORSE"
        print(f"    {mode:<14} vs STANDARD: ${delta:>+,.0f} alpha  ({verdict})")

print("\n✓ Done.")
