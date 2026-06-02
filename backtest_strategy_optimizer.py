"""
Multi-Strategy Backtest Optimizer
==================================
Tests several TF strategy variants on two periods:
  - OOS  : Sep 2025 → May 2026  (out-of-sample, bear/consolidation, BTC −33%)
  - IS   : Sep 2024 → Sep 2025  (in-sample, strong bull, BTC +93%)

Goal: find a strategy that outperforms B&H in BOTH regimes.

Strategies tested
-----------------
TF1  baseline   : Entry U1+(↑MA30|clean10d)         ; Exit D2|D3
TF2  regime-exit: Same entry                          ; Bull→D3 only | Bear→D2|D3
TF3  strong-D2  : Same entry                          ; Bull→D2<-1.5%|D3 | Bear→D2<-1.0%|D3
TF4  confirm-D2 : Same entry                          ; Exit (2×consec D2)|D3
TF5  combined   : Regime-adaptive entry AND exit:
     BULL regime: entry = U1+↑MA30 (no clean10d shortcut) + D3-only exit
     BEAR regime: entry = U1+(↑MA30|clean10d)            + D2|D3 exit

Regime definition (BULL): above_ma30 = True AND MA30 slope (5-bar) > 0
"""

import sys, os, warnings, joblib, requests
warnings.filterwarnings("ignore")
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import yfinance as yf

# ─── Load CT model artefact ───────────────────────────────────────────────────
_CT_PATH = _ROOT / "models" / "inference_assets_ct.joblib"
if not _CT_PATH.exists():
    _CT_PATH = _ROOT / "artifacts" / "artifacts.pkl"
print(f"Loading model from: {_CT_PATH}")
AD = joblib.load(str(_CT_PATH))
print(f"  keys: {list(AD.keys())[:8]}")

# ─── Fetch daily data ─────────────────────────────────────────────────────────
_MACRO_SYMS = {
    "btc": "BTC-USD",
    "eth": "ETH-USD", "spx": "^GSPC", "ndx": "^IXIC",
    "vix": "^VIX",  "gold": "GC=F",  "dxy": "DX-Y.NYB", "tnx": "^TNX",
}
_ONCHAIN_METRICS = [
    "hash-rate", "difficulty", "n-transactions", "miners-revenue",
    "n-unique-addresses", "transaction-fees-usd", "mempool-size",
    "estimated-transaction-volume-usd", "market-cap",
    "avg-block-size", "cost-per-transaction",
]


def _yf_download_single(sym: str, start: str, end: str) -> pd.DataFrame:
    """Download a single Yahoo ticker, flatten MultiIndex if present."""
    d = yf.download(sym, start=start, end=end, progress=False, auto_adjust=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] for c in d.columns]
    d.index = pd.DatetimeIndex(d.index).tz_localize(None).normalize()
    return d


def _fetch_onchain(metric: str, fetch_start: str) -> pd.Series:
    """Fetch a blockchain.info metric as daily Series."""
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


def fetch_daily_data(fetch_start: str, fetch_end: str) -> pd.DataFrame:
    """Build a daily-bar DataFrame (BTC + macro + on-chain).
    fetch_start should be 90+ days before the backtest start for feature warmup.
    """
    print(f"\n  Fetching daily data {fetch_start} → {fetch_end} …")
    frames = {}

    # BTC OHLCV
    d = _yf_download_single("BTC-USD", fetch_start, fetch_end)
    frames["btc_close"]  = d["Close"]
    frames["btc_high"]   = d["High"]
    frames["btc_low"]    = d["Low"]
    frames["btc_volume"] = d["Volume"]

    # Macro closes (individual downloads to avoid MultiIndex issues)
    for name, sym in list(_MACRO_SYMS.items())[1:]:   # skip btc
        try:
            d2 = _yf_download_single(sym, fetch_start, fetch_end)
            frames[f"{name}_close"] = d2["Close"]
        except Exception as e:
            print(f"    [warn] {sym}: {e}")

    df = pd.DataFrame(frames)
    df.index.name = "date"
    df = df.sort_index().ffill(limit=5)

    # On-chain
    print("  Fetching on-chain …", end=" ", flush=True)
    oc_ok = 0
    for m in _ONCHAIN_METRICS:
        s = _fetch_onchain(m, fetch_start)
        if not s.empty:
            s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
            s = s.reindex(df.index).ffill(limit=7)
            df[s.name] = s
            oc_ok += 1
    print(f"{oc_ok}/{len(_ONCHAIN_METRICS)} metrics OK")
    return df


# ─── CT batch predictions ─────────────────────────────────────────────────────
def build_ct_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised CT H/L predictions. Returns DataFrame indexed by TARGET date."""
    c  = df["btc_close"]
    h  = df["btc_high"]
    l_ = df["btc_low"]
    v  = df["btc_volume"]
    ret = np.log(c).diff()

    f = pd.DataFrame(index=df.index)
    for k in [1,3,5,7,14,30]: f[f"ret_{k}"] = ret.rolling(k).sum()
    for k in [5,10,20,30]:    f[f"vol_{k}"] = ret.rolling(k).std()
    prev_c = c.shift(1)
    tr = pd.concat([(h-l_),(h-prev_c).abs(),(l_-prev_c).abs()],axis=1).max(axis=1)
    for k in [7,14,30]: f[f"atr_{k}"] = tr.rolling(k).mean()/c
    f["range_today"] = (h-l_)/c
    f["range_ma7"]   = ((h-l_)/c).rolling(7).mean()
    f["range_ma30"]  = ((h-l_)/c).rolling(30).mean()
    f["range_std30"] = ((h-l_)/c).rolling(30).std()
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    f["rsi_14"] = 100 - 100/(1 + gain/loss.replace(0, np.nan))
    e12 = c.ewm(span=12,adjust=False).mean()
    e26 = c.ewm(span=26,adjust=False).mean()
    macd = e12 - e26
    f["macd"]      = macd/c
    f["macd_sig"]  = macd.ewm(span=9,adjust=False).mean()/c
    f["macd_hist"] = (macd - macd.ewm(span=9,adjust=False).mean())/c
    ma20 = c.rolling(20).mean(); sd20 = c.rolling(20).std()
    f["bb_width"]   = (4*sd20)/ma20
    f["dist_hi_30"] = c/c.rolling(30).max()-1
    f["dist_lo_30"] = c/c.rolling(30).min()-1
    f["dist_hi_90"] = c/c.rolling(90).max()-1
    f["vol_chg_1"]  = np.log(v).diff()
    f["vol_z_20"]   = (np.log(v)-np.log(v).rolling(20).mean())/np.log(v).rolling(20).std()
    f["vol_ma_ratio"] = v/v.rolling(20).mean()
    dow = df.index.dayofweek
    for i in range(6): f[f"dow_{i}"] = (dow==i).astype(float)
    for nm in ["spx","ndx","vix","gold","dxy","tnx","eth"]:
        col = f"{nm}_close"
        if col not in df.columns:
            continue
        s = df[col]
        lr = np.log(s).diff()
        for k in [1,5,20]: f[f"{nm}_ret_{k}"] = lr.rolling(k).sum()
        f[f"{nm}_vol_20"] = lr.rolling(20).std()
    for corr_nm, corr_col in [("spx","spx_close"),("ndx","ndx_close"),
                               ("gold","gold_close"),("dxy","dxy_close")]:
        if corr_col in df.columns:
            f[f"btc_{corr_nm}_corr_30"] = ret.rolling(30).corr(
                np.log(df[corr_col]).diff())
    for col in [x for x in df.columns if x.startswith("oc_")]:
        s = df[col].astype(float)
        s_log = np.log(s.replace(0, np.nan))
        f[f"{col}_d1"]  = s_log.diff(1)
        f[f"{col}_d7"]  = s_log.diff(7)
        f[f"{col}_z30"] = (s_log - s_log.rolling(30).mean())/s_log.rolling(30).std()
    nh, nl = h.shift(-1), l_.shift(-1)
    y_hi = (nh-c)/c; y_lo = (c-nl)/c
    f["y_hi_ema3"] = y_hi.shift(1).ewm(span=3,adjust=False).mean()
    f["y_lo_ema3"] = y_lo.shift(1).ewm(span=3,adjust=False).mean()
    f["y_hi_ema7"] = y_hi.shift(1).ewm(span=7,adjust=False).mean()
    f["y_lo_ema7"] = y_lo.shift(1).ewm(span=7,adjust=False).mean()
    prev_3_hi = h.shift(1).rolling(3).max()
    prev_3_lo = l_.shift(1).rolling(3).min()
    f["above_3d_high"]  = (c > prev_3_hi).astype(float)
    f["below_3d_low"]   = (c < prev_3_lo).astype(float)
    f["bo_strength_up"] = (c/prev_3_hi - 1).clip(lower=0)
    f["bo_strength_dn"] = (1 - c/prev_3_lo).clip(lower=0)
    _yla  = y_hi.shift(1)
    _ylab = y_lo.shift(1)
    f["y_hi_surprise"] = _yla  - _yla.ewm(span=7,adjust=False).mean()
    f["y_lo_surprise"] = _ylab - _ylab.ewm(span=7,adjust=False).mean()
    neg_ret = ret.clip(upper=0)
    f["dn_vol_5"]  = neg_ret.rolling(5).std()
    f["dn_vol_20"] = neg_ret.rolling(20).std()
    sma50 = c.rolling(50).mean()
    f["below_sma50"]    = (c < sma50).astype(float)
    f["below_sma50_5d"] = f["below_sma50"].rolling(5).min().fillna(0)

    fc = AD["feat_cols"]
    f  = f.replace([np.inf,-np.inf], np.nan)

    # Align to model feat_cols: add missing as NaN, keep order
    for col in fc:
        if col not in f.columns:
            f[col] = np.nan
    F = f[fc].dropna()

    missing_pct = sum(f[col].isna().all() for col in fc)
    print(f"  Feature matrix: {len(F)} rows, {len(fc)} cols "
          f"({missing_pct} cols all-NaN)")
    if F.empty:
        raise RuntimeError("Feature matrix empty after dropna — check data quality")

    # Predict
    if AD.get("ensemble") and AD.get("constituents"):
        yhi = np.mean([con["m_hi"].predict(F) for con in AD["constituents"]], axis=0)
        ylo = np.mean([con["m_lo"].predict(F) for con in AD["constituents"]], axis=0)
        if AD.get("blended") and float(AD.get("alpha",1.0)) < 1.0:
            a = float(AD["alpha"])
            yhi = a*yhi + (1-a)*float(AD.get("mu_hi",0))
            ylo = a*ylo + (1-a)*float(AD.get("mu_lo",0))
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


# ─── Compute signal arrays ────────────────────────────────────────────────────
def compute_signals(comp: pd.DataFrame) -> dict:
    """Compute all signal arrays from a completed prediction+actual DataFrame."""
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

    u1 = (ehma3 > 0.5)  & (hb3 >= 2)
    d1 = (lb3   >= 2)   & (elma3 > 0.5)
    d2 = ehma3 < -1.0
    d2s = ehma3 < -1.5   # strict D2

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
        ma30[i] = np.mean(c_asof[max(0, i-w+1): i+1])
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

    d2_confirmed = np.zeros(N, dtype=bool)
    for i in range(1, N):
        d2_confirmed[i] = bool(d2[i] and d2[i-1])

    return dict(
        u1=u1, d1=d1, d2=d2, d2s=d2s, d3=d3,
        above_ma30=above_ma30, bull_regime=bull_regime,
        clean_10d=clean_10d, d2_confirmed=d2_confirmed,
        c_asof=c_asof, ehma3=ehma3,
    )


# ─── Backtest engine ──────────────────────────────────────────────────────────
def run_backtest(
    df_raw: pd.DataFrame,
    preds:  pd.DataFrame,
    start_iso: str,
    end_iso:   str,
    strategy:  str  = "TF1",
    initial_capital: float = 100_000.0,
) -> dict:
    """
    Backtest a strategy variant on the given window.
    1-bar lag: signal on bar i → execute at bar i+1 close.
    """
    WARMUP = 35

    start_dt = pd.Timestamp(start_iso)
    end_dt   = pd.Timestamp(end_iso)

    # Join predictions with actual prices
    p = preds.loc[(preds.index >= start_dt - pd.Timedelta(days=45)) &
                  (preds.index <= end_dt)].copy()
    closes = df_raw["btc_close"]
    highs  = df_raw["btc_high"]
    lows   = df_raw["btc_low"]
    p["actual_high"]  = highs.reindex(p.index).values
    p["actual_low"]   = lows.reindex(p.index).values
    p["actual_close"] = closes.reindex(p.index).values
    comp = (p.dropna(subset=["actual_high","actual_low","actual_close"])
             .reset_index())
    comp = comp[comp["target_date"] >= start_dt].reset_index(drop=True)
    N = len(comp)
    if N < WARMUP + 3:
        return None

    dates   = pd.DatetimeIndex(comp["target_date"])
    exec_px = comp["actual_close"].values.astype(float)

    sigs = compute_signals(comp)
    u1, d1, d2, d2s, d3 = sigs["u1"], sigs["d1"], sigs["d2"], sigs["d2s"], sigs["d3"]
    above_ma30  = sigs["above_ma30"]
    bull_regime = sigs["bull_regime"]
    clean_10d   = sigs["clean_10d"]
    d2_conf     = sigs["d2_confirmed"]

    # ── Common entry condition (same for TF1–TF4; differs for TF5) ────────────
    base_entry = u1 & (above_ma30 | clean_10d)

    # Pre-compute static exit arrays for TF1 and TF4
    tf1_exit = d2 | d3
    tf4_exit = d2_conf | d3

    nav      = initial_capital
    pos      = "CASH"
    btc_qty  = 0.0
    e_price = e_nav = e_date = e_trigger = None
    trades  = []
    nav_arr = np.full(N, np.nan)

    for i in range(N):
        si    = i - 1
        price = exec_px[i]

        if i < WARMUP:
            nav_arr[i] = initial_capital
            continue

        # Resolve dynamic entry/exit
        if si >= 0:
            if strategy == "TF1":
                _enter = bool(base_entry[si])
                _exit  = bool(tf1_exit[si])
                _xlbl  = "D3" if d3[si] else "D2"
            elif strategy == "TF2":
                _enter = bool(base_entry[si])
                # In BULL regime: only D3 exits; in BEAR: D2 or D3
                _exit  = bool(d3[si] or (d2[si] and not bull_regime[si]))
                _xlbl  = "D3" if d3[si] else "D2"
            elif strategy == "TF3":
                _enter = bool(base_entry[si])
                # In BULL: need strict D2 (<-1.5%) or D3; in BEAR: D2 (<-1.0%) or D3
                _exit  = bool(d3[si] or (d2s[si] if bull_regime[si] else d2[si]))
                _xlbl  = "D3" if d3[si] else "D2"
            elif strategy == "TF4":
                _enter = bool(base_entry[si])
                _exit  = bool(tf4_exit[si])
                _xlbl  = "D3" if d3[si] else "D2×2"
            elif strategy == "TF5":
                # BULL: entry only U1+above_ma30 (no clean10d shortcut); exit D3 only
                # BEAR: entry U1+(above_ma30|clean10d); exit D2|D3
                if bull_regime[si]:
                    _enter = bool(u1[si] and above_ma30[si])
                    _exit  = bool(d3[si])
                else:
                    _enter = bool(base_entry[si])
                    _exit  = bool(d2[si] or d3[si])
                _xlbl  = "D3" if d3[si] else "D2"
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
        else:
            _enter = _exit = False
            _xlbl  = "?"

        if pos == "LONG":
            cur = btc_qty * price
            if _exit:
                nav = cur
                trades.append(dict(
                    entry_date=e_date, entry_price=e_price, entry_nav=e_nav,
                    entry_trigger=e_trigger,
                    exit_date=dates[i], exit_price=price, exit_nav=nav,
                    pnl_pct=(price/e_price - 1)*100,
                    pnl_abs=nav - e_nav,
                    exit_signal=_xlbl,
                    duration_days=(dates[i]-e_date).days,
                ))
                pos = "CASH"; btc_qty = 0.0
            else:
                nav = cur
        else:
            if _enter:
                btc_qty = nav / price
                e_price = price; e_date = dates[i]
                e_nav   = nav;   pos   = "LONG"
                if si >= 0 and above_ma30[si] and clean_10d[si]:
                    e_trigger = "U1+↑MA30+c10d"
                elif si >= 0 and above_ma30[si]:
                    e_trigger = "U1+↑MA30"
                else:
                    e_trigger = "U1+c10d"

        nav_arr[i] = btc_qty * price if pos == "LONG" else nav

    if pos == "LONG":
        nav_arr[N-1] = btc_qty * exec_px[N-1]

    nav_series = pd.Series(nav_arr[WARMUP:], index=dates[WARMUP:]).ffill()
    bh_series  = pd.Series(
        initial_capital * exec_px[WARMUP:] / exec_px[WARMUP],
        index=dates[WARMUP:],
    )
    final_nav = float(nav_series.iloc[-1])
    final_bh  = float(bh_series.iloc[-1])
    wins      = [t for t in trades if t["pnl_pct"] > 0]
    days_in   = sum(t["duration_days"] for t in trades)
    tot_days  = max(1, (dates[N-1] - dates[WARMUP]).days)
    rm        = nav_series.cummax()
    max_dd    = float(((nav_series - rm) / rm * 100).min())

    return dict(
        strategy  = strategy,
        start     = str(dates[WARMUP].date()),
        end       = str(dates[N-1].date()),
        final_nav = final_nav,   final_bh = final_bh,
        strat_ret = (final_nav/initial_capital - 1)*100,
        bh_ret    = (final_bh /initial_capital - 1)*100,
        alpha_abs = final_nav - final_bh,
        n_trades  = len(trades),
        n_wins    = len(wins),
        win_rate  = 100*len(wins)/len(trades) if trades else 0.0,
        max_dd    = max_dd,
        time_in   = 100*days_in/tot_days,
        trades    = trades,
        nav_series = nav_series,
        bh_series  = bh_series,
        open_pos  = pos == "LONG",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
STRATEGIES = ["TF1", "TF2", "TF3", "TF4", "TF5"]

PERIODS = {
    "OOS (Mar26→May26)": {
        "start": "2026-03-01", "end": "2026-05-27",
        "fetch_start": "2025-12-01", "fetch_end": "2026-05-28",
        "regime": "bear",
    },
    "Prior-Year Bull+93% (Sep24→Sep25)": {
        "start": "2024-09-17", "end": "2025-09-17",
        "fetch_start": "2024-06-01", "fetch_end": "2025-09-20",
        "regime": "bull",
    },
}

results = {}

for period_name, pcfg in PERIODS.items():
    print(f"\n{'═'*72}")
    print(f"PERIOD: {period_name}")
    print(f"{'═'*72}")
    df_raw = fetch_daily_data(pcfg["fetch_start"], pcfg["fetch_end"])
    print("  Building CT predictions …", end=" ", flush=True)
    preds  = build_ct_predictions(df_raw)
    print(f"→ {len(preds)} bars")

    results[period_name] = {}
    for strat in STRATEGIES:
        r = run_backtest(df_raw, preds, pcfg["start"], pcfg["end"], strat)
        results[period_name][strat] = r

    # Per-period results table
    print(f"\n  {'─'*78}")
    print(f"  {'Strategy':<8} {'NAV':>10} {'Ret%':>8} {'B&H%':>8} {'Alpha$':>12} "
          f"{'Trd':>5} {'Win%':>7} {'MaxDD%':>8} {'TiM%':>6}")
    print(f"  {'─'*78}")
    for strat, r in results[period_name].items():
        if r is None:
            print(f"  {strat:<8}  ERROR"); continue
        best = r["alpha_abs"] == max(
            rv["alpha_abs"] for rv in results[period_name].values() if rv)
        print(
            f"  {strat:<8} ${r['final_nav']:>9,.0f} {r['strat_ret']:>+7.1f}% "
            f"{r['bh_ret']:>+7.1f}% ${r['alpha_abs']:>+11,.0f} "
            f"{r['n_trades']:>4}  {r['win_rate']:>5.0f}% {r['max_dd']:>+7.1f}% "
            f"{r['time_in']:>5.0f}%"
            + ("  ◄ BEST" if best else "")
        )

# ─── Cross-period summary ─────────────────────────────────────────────────────
pnames = list(results.keys())
p1, p2 = pnames[0], pnames[1]

print(f"\n{'═'*72}")
print("CROSS-PERIOD SUMMARY — Outperforms B&H in BOTH periods?")
print(f"{'═'*72}")
print(f"\n  {'Strategy':<8}  {'Bear Alpha':>13}  {'Bull Alpha':>13}  "
      f"{'Total':>13}  {'Both?':>6}")
print(f"  {'─'*65}")

both_winners = []
for strat in STRATEGIES:
    r1 = results[p1].get(strat)
    r2 = results[p2].get(strat)
    if not r1 or not r2: continue
    a1, a2 = r1["alpha_abs"], r2["alpha_abs"]
    both = (a1 > 0 and a2 > 0)
    if both:
        both_winners.append((strat, a1+a2))
    print(
        f"  {strat:<8}  ${a1:>+12,.0f}  ${a2:>+12,.0f}  "
        f"${a1+a2:>+12,.0f}  {'✓ YES' if both else '✗ NO ':>6}"
    )

# ─── Winner detail ────────────────────────────────────────────────────────────
if both_winners:
    best_strat = max(both_winners, key=lambda x: x[1])[0]
    print(f"\n★  WINNER: {best_strat} — outperforms B&H in BOTH periods "
          f"(total alpha ${dict(both_winners)[best_strat]:+,.0f})")
else:
    # No strategy beats B&H in both — show the one with best total alpha
    totals = []
    for strat in STRATEGIES:
        r1 = results[p1].get(strat)
        r2 = results[p2].get(strat)
        if r1 and r2:
            totals.append((strat, r1["alpha_abs"]+r2["alpha_abs"]))
    best_strat = max(totals, key=lambda x: x[1])[0]
    print(f"\n[!] No strategy outperforms B&H in BOTH periods.")
    print(f"    Best combined alpha: {best_strat} "
          f"(${dict(totals)[best_strat]:+,.0f})")

# ─── Trade log for winner in each period ─────────────────────────────────────
print(f"\n{'─'*72}")
print(f"TRADE LOG — {best_strat}")
for pname in pnames:
    r = results[pname].get(best_strat)
    if r is None or not r["trades"]: continue
    print(f"\n  {pname}  [NAV ${r['final_nav']:,.0f}  B&H ${r['final_bh']:,.0f}  "
          f"Alpha ${r['alpha_abs']:+,.0f}]")
    print(f"  {'#':>3}  {'Entry':12}  {'Exit':12}  {'P&L%':>7}  {'Days':>5}  "
          f"Trigger          Exit")
    for j, t in enumerate(r["trades"], 1):
        mk = "✓" if t["pnl_pct"] > 0 else "✗"
        print(
            f"  {j:>3}  {str(t['entry_date'].date()):12}  "
            f"{str(t['exit_date'].date()):12}  "
            f"{mk}{t['pnl_pct']:>+5.1f}%  "
            f"{t['duration_days']:>5}  "
            f"{t.get('entry_trigger','?'):<16} "
            f"{t.get('exit_signal','?')}"
        )
    if r["open_pos"]:
        print(f"  {'':>3}  Open position at period end")

print("\n✓ Done.")
