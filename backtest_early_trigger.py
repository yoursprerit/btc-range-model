#!/usr/bin/env python3
"""
Early U1 Trigger Analysis
==========================
Question: When the pre-conditions for U1 are already partially met —
  • err_hi_ma3 > 0.7%  (3-day avg high-side error already elevated)
  • hi_breaks_3d == 1  (only 1 of last 3 bars has broken pred_high)
...and today's INTRADAY price breaks through today's pred_high, does entering
EARLY (intraday, at the breakout level) outperform waiting for the bar to close
and executing at next day's close (standard TF1/TF2 convention)?

Three entry modes compared
--------------------------
  STANDARD : Signal fires at bar-T close (hi_breaks_3d becomes 2).
             Execute at bar T+1 close. (current live strategy)

  EARLY-ALL : Trigger fires intraday on bar T whenever:
              prev_ehma3 > 0.7 AND prev_hb3 == 1 AND high_T > pred_high_T
              Enter at max(open_T, pred_high_T) — the breakout-level proxy.
              Includes BOTH confirmed breaks (close_T > pred_high_T) and
              false intraday breaks (close_T ≤ pred_high_T).

  EARLY-CONF: Same intraday trigger but only accepted when close_T > pred_high_T
              (bar closes above the ceiling — same bars as STANDARD but 1 bar earlier
              in execution time). Enter at max(open_T, pred_high_T).

Exits: D2 or D3 fires at bar close → execute at next bar close (unchanged for all modes).

Additional diagnostics
-----------------------
  • False-break rate: how often does intraday break NOT survive to close?
  • Forward-return profile: 1d / 3d / 5d after early vs standard entry
  • Per-trade comparison on confirmed-break days

Periods tested
--------------
  OOS Bear   Sep 2025 → May 2026  (BTC −33%)
  Bull IS    Sep 2024 → Sep 2025  (BTC +93%)
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
    """Coinbase Exchange daily candles → close price Series."""
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
            print(f"    [warn] coinbase chunk: {e}")
        cur = chunk_end + pd.Timedelta(days=1)
        _time.sleep(0.35)
    if not rows:
        return pd.Series(dtype=float, name="coinbase_close")
    tmp = pd.DataFrame(rows, columns=["ts","low","high","open","close","volume"])
    tmp["date"] = pd.to_datetime(tmp["ts"], unit="s").dt.normalize()
    tmp = tmp.drop_duplicates("date").set_index("date").sort_index()
    return tmp["close"].rename("coinbase_close")


def fetch_daily_data(fetch_start, fetch_end):
    """Daily OHLCV (BTC open included) + macro + on-chain + Coinbase premium."""
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
        n_ok = df["coinbase_close"].notna().sum()
        print(f"{n_ok} rows")
    else:
        print("unavailable (will zero-fill)")

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

    # Coinbase premium (required by model; zero-fill when unavailable)
    if "coinbase_close" in df.columns and df["coinbase_close"].notna().any():
        _cb   = df["coinbase_close"]
        _prem = (_cb - c) / c * 100
        f["cb_premium"]     = _prem
        f["cb_premium_ma3"] = _prem.rolling(3).mean()
        f["cb_premium_z7"]  = (
            (_prem - _prem.rolling(7).mean()) / _prem.rolling(7).std()
        )
    else:
        f["cb_premium"]     = 0.0
        f["cb_premium_ma3"] = 0.0
        f["cb_premium_z7"]  = 0.0

    fc = AD["feat_cols"]
    f  = f.replace([np.inf,-np.inf], np.nan)
    for col in fc:
        if col not in f.columns: f[col] = np.nan
    F = f[fc].dropna()

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

    u1 = (ehma3 > 0.7) & (hb3 >= 2)   # live threshold
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

    # Early-trigger pre-state: at end of bar i-1, is the setup partially met?
    # prev_ehma3 > 0.7 AND prev_hb3 == 1
    # → if today's high also breaks pred_high, early U1 fires
    early_setup = np.zeros(N, dtype=bool)
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
_MODES = ("STANDARD", "EARLY-CONF", "EARLY-ALL")

def run_backtest(df_raw, preds, start_iso, end_iso,
                 mode="STANDARD", initial_capital=100_000.0):
    """
    mode = "STANDARD"   : signal at bar-T close → execute T+1 close
    mode = "EARLY-CONF" : intraday entry at max(open_T, pred_high_T) when
                          prev_ehma3>0.7, prev_hb3==1, high_T>pred_hi_T,
                          AND close_T>pred_hi_T (confirmed break).
    mode = "EARLY-ALL"  : same but includes false intraday breaks (close_T ≤ pred_hi_T).
    """
    WARMUP = 35
    start_dt = pd.Timestamp(start_iso)
    end_dt   = pd.Timestamp(end_iso)

    p = preds.loc[(preds.index >= start_dt - pd.Timedelta(days=45)) &
                  (preds.index <= end_dt)].copy()
    closes = df_raw["btc_close"]
    highs  = df_raw["btc_high"]
    lows   = df_raw["btc_low"]
    opens  = df_raw["btc_open"]

    p["actual_high"]  = highs.reindex(p.index).values
    p["actual_low"]   = lows.reindex(p.index).values
    p["actual_close"] = closes.reindex(p.index).values
    p["actual_open"]  = opens.reindex(p.index).values
    comp = (p.dropna(subset=["actual_high","actual_low","actual_close","actual_open"])
             .reset_index())
    comp = comp[comp["target_date"] >= start_dt].reset_index(drop=True)
    N = len(comp)
    if N < WARMUP + 3:
        return None

    dates   = pd.DatetimeIndex(comp["target_date"])
    sigs    = compute_signals(comp)

    u1          = sigs["u1"]
    d1          = sigs["d1"]
    d2          = sigs["d2"]
    d3          = sigs["d3"]
    above_ma30  = sigs["above_ma30"]
    bull_regime = sigs["bull_regime"]
    clean_10d   = sigs["clean_10d"]
    early_setup = sigs["early_setup"]
    pred_hi     = sigs["pred_hi"]

    act_hi  = comp["actual_high"].values.astype(float)
    act_cl  = comp["actual_close"].values.astype(float)
    act_op  = comp["actual_open"].values.astype(float)

    # Early entry price proxy: max(open, pred_high)
    # If open already above ceiling → gapped up, enter at open
    # If open below ceiling → enter exactly at breakout level
    early_px = np.maximum(act_op, pred_hi)

    # Entry/exit logic varies by mode
    nav     = initial_capital
    pos     = "CASH"
    btc_qty = 0.0
    e_price = e_nav = e_date = e_trigger = None
    trades  = []
    nav_arr = np.full(N, np.nan)

    for i in range(N):
        price = act_cl[i]   # default exec price (close of bar i)

        if i < WARMUP:
            nav_arr[i] = initial_capital
            continue

        si = i - 1  # signal bar index (previous close)

        # ── Determine entry/exit signals at bar i ────────────────────────────
        if mode == "STANDARD":
            # Signal from prior bar (si), execute at current close
            _enter = bool(u1[si] and (above_ma30[si] or clean_10d[si])) if si >= 0 else False
            _exit  = bool(d2[si] or d3[si]) if si >= 0 else False
            exec_price = act_cl[i]

        else:
            # Early modes: check if THIS bar fires early trigger
            # Early entry: the current bar is the "trigger bar"
            # - pre-state from bar i-1: early_setup[i] = True
            # - today's intraday: act_hi[i] > pred_hi[i]
            intraday_break = bool(act_hi[i] > pred_hi[i])
            close_confirms = bool(act_cl[i] > pred_hi[i])

            if mode == "EARLY-CONF":
                fire_early = bool(early_setup[i] and intraday_break and close_confirms)
            else:  # EARLY-ALL
                fire_early = bool(early_setup[i] and intraday_break)

            # Standard entry from prior bar (si) — might also fire on the same day
            std_enter = bool(u1[si] and (above_ma30[si] or clean_10d[si])) if si >= 0 else False

            # Combined: early fire takes priority on current bar
            _enter = fire_early or std_enter
            exec_price = early_px[i] if fire_early else act_cl[i]

            # Exit: standard (from prior bar's signal)
            _exit = bool(d2[si] or d3[si]) if si >= 0 else False

        # ── Update position ──────────────────────────────────────────────────
        if pos == "LONG":
            cur = btc_qty * act_cl[i]   # mark-to-market at close
            if _exit:
                exit_px = act_cl[i]
                nav = btc_qty * exit_px
                trades.append(dict(
                    entry_date=e_date, entry_price=e_price, entry_nav=e_nav,
                    entry_trigger=e_trigger,
                    exit_date=dates[i], exit_price=exit_px, exit_nav=nav,
                    pnl_pct=(exit_px/e_price - 1)*100,
                    pnl_abs=nav - e_nav,
                    exit_signal="D3" if d3[si] else "D2",
                    duration_days=(dates[i]-e_date).days,
                ))
                pos = "CASH"; btc_qty = 0.0
            else:
                nav = cur
        else:
            if _enter and not _exit:
                btc_qty = nav / exec_price
                e_price = exec_price
                e_date  = dates[i]
                e_nav   = nav
                pos     = "LONG"
                e_trigger = "EARLY" if (mode != "STANDARD" and
                             early_setup[i] and act_hi[i] > pred_hi[i]) else "STD-U1"

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


# ─── Early-trigger diagnostic ─────────────────────────────────────────────────
def early_trigger_diagnostics(df_raw, preds, start_iso, end_iso):
    """
    For each candidate bar where early trigger fires (pre-state met + intraday break),
    compute:
      • Whether close confirmed the break (close > pred_high)
      • Entry price advantage vs next-day close
      • 1d / 3d / 5d forward returns from early vs standard entry
    """
    start_dt = pd.Timestamp(start_iso)
    end_dt   = pd.Timestamp(end_iso)

    p = preds.loc[(preds.index >= start_dt - pd.Timedelta(days=45)) &
                  (preds.index <= end_dt)].copy()
    closes = df_raw["btc_close"]
    highs  = df_raw["btc_high"]
    lows   = df_raw["btc_low"]
    opens  = df_raw["btc_open"]

    p["actual_high"]  = highs.reindex(p.index).values
    p["actual_low"]   = lows.reindex(p.index).values
    p["actual_close"] = closes.reindex(p.index).values
    p["actual_open"]  = opens.reindex(p.index).values
    comp = (p.dropna(subset=["actual_high","actual_low","actual_close","actual_open"])
             .reset_index())
    comp = comp[comp["target_date"] >= start_dt].reset_index(drop=True)
    N = len(comp)

    sigs = compute_signals(comp)
    early_setup = sigs["early_setup"]
    pred_hi     = sigs["pred_hi"]

    act_hi = comp["actual_high"].values.astype(float)
    act_cl = comp["actual_close"].values.astype(float)
    act_op = comp["actual_open"].values.astype(float)
    dates  = pd.DatetimeIndex(comp["target_date"])

    rows = []
    WARMUP = 35
    for i in range(WARMUP, N-5):
        if not early_setup[i]:
            continue
        intraday_break = bool(act_hi[i] > pred_hi[i])
        if not intraday_break:
            continue

        close_confirmed = bool(act_cl[i] > pred_hi[i])
        early_entry     = max(act_op[i], pred_hi[i])
        std_entry       = act_cl[i+1]  # next bar's close (standard execution)

        # Forward returns from both entry points
        fwd_ret_1d_early  = (act_cl[i+1]/early_entry - 1)*100
        fwd_ret_3d_early  = (act_cl[min(i+3, N-1)]/early_entry - 1)*100
        fwd_ret_5d_early  = (act_cl[min(i+5, N-1)]/early_entry - 1)*100
        fwd_ret_1d_std    = (act_cl[min(i+2, N-1)]/std_entry - 1)*100
        fwd_ret_3d_std    = (act_cl[min(i+4, N-1)]/std_entry - 1)*100
        fwd_ret_5d_std    = (act_cl[min(i+6, N-1)]/std_entry - 1)*100

        # Entry price advantage (positive = early gets better price)
        px_advantage_pct  = (std_entry/early_entry - 1)*100

        rows.append(dict(
            date=str(dates[i].date()),
            close_confirmed=close_confirmed,
            early_entry=round(early_entry, 1),
            std_entry=round(std_entry, 1),
            px_advantage_pct=round(px_advantage_pct, 3),
            fwd_1d_early=round(fwd_ret_1d_early, 2),
            fwd_3d_early=round(fwd_ret_3d_early, 2),
            fwd_5d_early=round(fwd_ret_5d_early, 2),
            fwd_1d_std=round(fwd_ret_1d_std, 2),
            fwd_3d_std=round(fwd_ret_3d_std, 2),
            fwd_5d_std=round(fwd_ret_5d_std, 2),
            open_T=round(act_op[i], 1),
            pred_hi_T=round(pred_hi[i], 1),
            close_T=round(act_cl[i], 1),
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

    # ── Run all three modes ────────────────────────────────────────────────
    period_res = {}
    for mode in _MODES:
        r = run_backtest(df_raw, preds, pcfg["start"], pcfg["end"], mode=mode)
        period_res[mode] = r

    all_results[period_name] = period_res

    # ── Performance table ──────────────────────────────────────────────────
    print(f"\n  {'─'*80}")
    print(f"  {'Mode':<14} {'NAV':>10} {'Ret%':>8} {'B&H%':>8} {'Alpha$':>12} "
          f"{'Trades':>7} {'Win%':>7} {'MaxDD%':>8} {'TiM%':>6}")
    print(f"  {'─'*80}")
    for mode, r in period_res.items():
        if r is None:
            print(f"  {mode:<14}  ERROR"); continue
        print(
            f"  {mode:<14} ${r['final_nav']:>9,.0f} {r['strat_ret']:>+7.1f}% "
            f"{r['bh_ret']:>+7.1f}% ${r['alpha_abs']:>+11,.0f} "
            f"{r['n_trades']:>6}  {r['win_rate']:>5.0f}% {r['max_dd']:>+7.1f}% "
            f"{r['time_in']:>5.0f}%"
        )

    # ── Early trigger diagnostics ──────────────────────────────────────────
    print(f"\n  ── Early Trigger Diagnostic ({period_name}) ──")
    diag = early_trigger_diagnostics(df_raw, preds, pcfg["start"], pcfg["end"])

    if diag.empty:
        print("  No early trigger candidates found in this period.")
    else:
        n_total      = len(diag)
        n_confirmed  = diag["close_confirmed"].sum()
        n_false      = n_total - n_confirmed
        false_rate   = n_false / n_total * 100

        print(f"\n  Early trigger candidates : {n_total}")
        print(f"  Close confirmed (high bar): {n_confirmed} ({100-false_rate:.0f}%)")
        print(f"  False intraday breaks     : {n_false} ({false_rate:.0f}%)")

        # Entry price advantage
        adv = diag["px_advantage_pct"]
        print(f"\n  Entry price advantage (std_entry/early_entry − 1) :")
        print(f"    Mean  : {adv.mean():+.3f}% (+ = early gets cheaper price)")
        print(f"    Median: {adv.median():+.3f}%")
        print(f"    >0 (early cheaper): {(adv > 0).sum()}/{n_total} days "
              f"({100*(adv>0).mean():.0f}%)")

        # Forward returns — confirmed breaks only
        conf = diag[diag["close_confirmed"]]
        false_b = diag[~diag["close_confirmed"]]
        if not conf.empty:
            print(f"\n  Forward returns — CONFIRMED breaks (n={len(conf)}):")
            print(f"  {'Horizon':<10} {'Early mean':>11} {'Std mean':>11} {'Delta':>9}")
            for h in [1, 3, 5]:
                em = conf[f"fwd_{h}d_early"].mean()
                sm = conf[f"fwd_{h}d_std"].mean()
                print(f"  {h}d          {em:>+10.2f}% {sm:>+10.2f}% {em-sm:>+8.2f}%")

        if not false_b.empty:
            print(f"\n  Forward returns — FALSE intraday breaks (n={len(false_b)}):")
            print(f"  {'Horizon':<10} {'Early mean':>11} {'Std mean':>11}")
            for h in [1, 3, 5]:
                em = false_b[f"fwd_{h}d_early"].mean()
                sm = false_b[f"fwd_{h}d_std"].mean()
                print(f"  {h}d          {em:>+10.2f}% {sm:>+10.2f}%")

        # Per-event table
        print(f"\n  Per-event table:")
        print(f"  {'Date':12} {'Conf':5} {'EarlyPx':>10} {'StdPx':>10} "
              f"{'Adv%':>7} {'1d-E%':>7} {'1d-S%':>7} {'3d-E%':>7} {'3d-S%':>7}")
        print(f"  {'─'*82}")
        for _, row in diag.iterrows():
            conf_str = "YES" if row["close_confirmed"] else "no"
            print(
                f"  {row['date']:12} {conf_str:5} "
                f"{row['early_entry']:>10,.0f} {row['std_entry']:>10,.0f} "
                f"{row['px_advantage_pct']:>+6.2f}% "
                f"{row['fwd_1d_early']:>+6.2f}% {row['fwd_1d_std']:>+6.2f}% "
                f"{row['fwd_3d_early']:>+6.2f}% {row['fwd_3d_std']:>+6.2f}%"
            )

    # ── Trade log ──────────────────────────────────────────────────────────
    for mode, r in period_res.items():
        if r is None or not r["trades"]: continue
        print(f"\n  Trade log — {mode}")
        print(f"  {'#':>2}  {'Entry':12} {'Exit':12} {'Trigger':10} "
              f"{'EntryPx':>10} {'ExitPx':>10} {'P&L%':>7} {'Days':>5}")
        for j, t in enumerate(r["trades"], 1):
            mk = "✓" if t["pnl_pct"] > 0 else "✗"
            print(
                f"  {j:>2}  {str(t['entry_date'].date()):12} "
                f"{str(t['exit_date'].date()):12} "
                f"{t.get('entry_trigger','?'):<10} "
                f"{t['entry_price']:>10,.0f} {t['exit_price']:>10,.0f} "
                f"{mk}{t['pnl_pct']:>+6.1f}% {t['duration_days']:>5}d"
            )
        if r["open_pos"]:
            print(f"  [open position at period end]")

# ─── Cross-period summary ──────────────────────────────────────────────────────
print(f"\n{'═'*72}")
print("CROSS-PERIOD SUMMARY — Does early trigger help?")
print(f"{'═'*72}\n")

pnames = list(all_results.keys())
print(f"  {'Mode':<14}  ", end="")
for pn in pnames:
    short = "OOS-Bear" if "Bear" in pn else "Bull-IS"
    print(f"  {short} Alpha$  {short} NAV%", end="")
print()
print(f"  {'─'*90}")

for mode in _MODES:
    print(f"  {mode:<14}", end="")
    for pn in pnames:
        r = all_results[pn].get(mode)
        if r:
            print(f"  ${r['alpha_abs']:>+11,.0f}  {r['strat_ret']:>+8.1f}%", end="")
        else:
            print(f"  {'N/A':>13}  {'N/A':>8}", end="")
    print()

print(f"\n  Verdict:")
for pn in pnames:
    rs = {m: all_results[pn].get(m) for m in _MODES}
    if not all(rs.values()):
        continue
    alphas = {m: rs[m]["alpha_abs"] for m in _MODES}
    best   = max(alphas, key=alphas.get)
    std_a  = alphas["STANDARD"]
    ec_a   = alphas["EARLY-CONF"]
    ea_a   = alphas["EARLY-ALL"]
    print(f"\n  {pn}:")
    print(f"    EARLY-CONF vs STANDARD : ${ec_a - std_a:>+,.0f} alpha delta "
          f"({'BETTER' if ec_a > std_a else 'WORSE'})")
    print(f"    EARLY-ALL  vs STANDARD : ${ea_a - std_a:>+,.0f} alpha delta "
          f"({'BETTER' if ea_a > std_a else 'WORSE'})")
    print(f"    Best mode: {best}")

print("\n✓ Done.")
