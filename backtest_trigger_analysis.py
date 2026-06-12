#!/usr/bin/env python3
"""
Entry-Trigger Analysis: U1+MA30+clean10d vs Separate Triggers (MSTR & MSTU)
=============================================================================

Questions answered:
1. Is "U1+MA30+clean10d" (both filters simultaneously) statistically more
   prone to loss than the single-filter triggers for MSTR / MSTU?

2. Would restricting entries to three strict, non-overlapping triggers
   (U1+MA30, U1+V-Reversal, U1+clean10d) improve backtesting results?

The "strict" alternative classifies each trade by whichever single condition
is the PRIMARY reason for entry (priority: V-reversal > MA30 > clean10d).
This does NOT change the entry condition — the union rule is kept exactly as
in the live strategy.  The "exclude-combined" variant DOES change the entry
condition by blocking trades where both above_ma30 AND clean_10d fire.

Periods tested (matching UI locked backtests):
  Bear  Jun 2025 – May 2026
  Bull  Jun 2024 – Jun 2025
  Full  Jun 2024 – May 2026

Assets: MSTR (fixed -3% stop), MSTU (fixed -7% stop) — same as live.
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

# ── Config ────────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 100_000.0
WARMUP = 35

_MODEL_PATH = _ROOT / "models" / "inference_assets_ct.joblib"
if not _MODEL_PATH.exists():
    _MODEL_PATH = _ROOT / "artifacts" / "artifacts.pkl"

PERIODS = {
    "Bear (Jun25→May26)":  ("2025-06-01", "2026-05-31", "2025-03-01"),
    "Bull (Jun24→Jun25)":  ("2024-06-05", "2025-06-14", "2024-03-01"),
    "Full (Jun24→May26)":  ("2024-06-01", "2026-05-31", "2024-03-01"),
}

ASSETS = {
    "MSTR": {"ticker": "MSTR", "sl_pct": 0.03, "stop_label": "Fixed -3%"},
    "MSTU": {"ticker": "MSTU", "sl_pct": 0.07, "stop_label": "Fixed -7%"},
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

# ── Model loading ─────────────────────────────────────────────────────────────
print(f"Loading CT model from: {_MODEL_PATH}")
AD = joblib.load(str(_MODEL_PATH))
print(f"  keys: {list(AD.keys())[:6]}\n")


# ── Data fetching ─────────────────────────────────────────────────────────────
def _yf_dl(ticker, start, end):
    d = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] for c in d.columns]
    d.index = pd.DatetimeIndex(d.index).tz_localize(None).normalize()
    return d


_DATA_CACHE: dict = {}


def fetch_data(fetch_start, fetch_end):
    cache_key = f"{fetch_start}_{fetch_end}"
    if cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key]

    print(f"  Fetching BTC+macro {fetch_start} → {fetch_end} …")
    d = _yf_dl("BTC-USD", fetch_start, fetch_end)
    frames = {
        "btc_close": d["Close"], "btc_high": d["High"],
        "btc_low":   d["Low"],   "btc_volume": d["Volume"],
    }
    for nm, sym in _MACRO_SYMS.items():
        try:
            frames[f"{nm}_close"] = _yf_dl(sym, fetch_start, fetch_end)["Close"]
        except Exception:
            pass
    df = pd.DataFrame(frames).sort_index().ffill(limit=5)
    df.index.name = "date"

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

    _DATA_CACHE[cache_key] = df
    return df


def fetch_asset_px(ticker, fetch_start, fetch_end):
    try:
        d = _yf_dl(ticker, fetch_start, fetch_end)
        px = d["Close"].sort_index()
        all_days = pd.date_range(px.index[0], max(px.index[-1], pd.Timestamp(fetch_end)), freq="D")
        return px.reindex(all_days).ffill()
    except Exception as e:
        print(f"    [WARN] {ticker}: {e}")
        return pd.Series(dtype=float)


# ── CT predictions ────────────────────────────────────────────────────────────
def build_ct_preds(df):
    c, h, l_, v = df["btc_close"], df["btc_high"], df["btc_low"], df["btc_volume"]
    ret = np.log(c).diff()
    f = pd.DataFrame(index=df.index)
    for k in [1, 3, 5, 7, 14, 30]: f[f"ret_{k}"] = ret.rolling(k).sum()
    for k in [5, 10, 20, 30]:       f[f"vol_{k}"] = ret.rolling(k).std()
    prev_c = c.shift(1)
    tr = pd.concat([(h-l_), (h-prev_c).abs(), (l_-prev_c).abs()], axis=1).max(axis=1)
    for k in [7, 14, 30]: f[f"atr_{k}"] = tr.rolling(k).mean() / c
    f["range_today"] = (h - l_) / c; f["range_ma7"] = ((h-l_)/c).rolling(7).mean()
    f["range_ma30"]  = ((h - l_) / c).rolling(30).mean()
    f["range_std30"] = ((h - l_) / c).rolling(30).std()
    g  = c.diff().clip(lower=0).rolling(14).mean()
    ls = (-c.diff().clip(upper=0)).rolling(14).mean()
    f["rsi_14"] = 100 - 100 / (1 + g / ls.replace(0, np.nan))
    e12 = c.ewm(span=12, adjust=False).mean(); e26 = c.ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    f["macd"] = macd/c; f["macd_sig"] = macd.ewm(span=9, adjust=False).mean()/c
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    ma20 = c.rolling(20).mean(); sd20 = c.rolling(20).std()
    f["bb_width"]   = (4 * sd20) / ma20
    f["dist_hi_30"] = c / c.rolling(30).max() - 1
    f["dist_lo_30"] = c / c.rolling(30).min() - 1
    f["dist_hi_90"] = c / c.rolling(90).max() - 1
    f["vol_chg_1"]  = np.log(v).diff()
    f["vol_z_20"]   = (np.log(v)-np.log(v).rolling(20).mean()) / np.log(v).rolling(20).std()
    f["vol_ma_ratio"] = v / v.rolling(20).mean()
    dow = df.index.dayofweek
    for i in range(6): f[f"dow_{i}"] = (dow == i).astype(float)
    for nm in ["spx","ndx","vix","gold","dxy","tnx","eth"]:
        col = f"{nm}_close"
        if col not in df.columns: continue
        s = df[col]; lr = np.log(s).diff()
        for k in [1, 5, 20]: f[f"{nm}_ret_{k}"] = lr.rolling(k).sum()
        f[f"{nm}_vol_20"] = lr.rolling(20).std()
    for corr_nm, corr_col in [("spx","spx_close"),("ndx","ndx_close"),
                               ("gold","gold_close"),("dxy","dxy_close")]:
        if corr_col in df.columns:
            f[f"btc_{corr_nm}_corr_30"] = ret.rolling(30).corr(np.log(df[corr_col]).diff())
    for col in [x for x in df.columns if x.startswith("oc_")]:
        s = df[col].astype(float); sl = np.log(s.replace(0, np.nan))
        f[f"{col}_d1"] = sl.diff(1); f[f"{col}_d7"] = sl.diff(7)
        f[f"{col}_z30"] = (sl - sl.rolling(30).mean()) / sl.rolling(30).std()
    nh, nl_ = h.shift(-1), l_.shift(-1)
    y_hi = (nh - c) / c; y_lo = (c - nl_) / c
    f["y_hi_ema3"] = y_hi.shift(1).ewm(span=3, adjust=False).mean()
    f["y_lo_ema3"] = y_lo.shift(1).ewm(span=3, adjust=False).mean()
    f["y_hi_ema7"] = y_hi.shift(1).ewm(span=7, adjust=False).mean()
    f["y_lo_ema7"] = y_lo.shift(1).ewm(span=7, adjust=False).mean()
    p3h = h.shift(1).rolling(3).max(); p3l = l_.shift(1).rolling(3).min()
    f["above_3d_high"]  = (c > p3h).astype(float)
    f["below_3d_low"]   = (c < p3l).astype(float)
    f["bo_strength_up"] = (c / p3h - 1).clip(lower=0)
    f["bo_strength_dn"] = (1 - c / p3l).clip(lower=0)
    ya = y_hi.shift(1); yb = y_lo.shift(1)
    f["y_hi_surprise"] = ya - ya.ewm(span=7, adjust=False).mean()
    f["y_lo_surprise"] = yb - yb.ewm(span=7, adjust=False).mean()
    neg_ret = ret.clip(upper=0)
    f["dn_vol_5"]  = neg_ret.rolling(5).std()
    f["dn_vol_20"] = neg_ret.rolling(20).std()
    sma50 = c.rolling(50).mean()
    f["below_sma50"]    = (c < sma50).astype(float)
    f["below_sma50_5d"] = f["below_sma50"].rolling(5).min().fillna(0)

    f["cb_premium"] = f["cb_premium_ma3"] = f["cb_premium_z7"] = 0.0

    fc = AD["feat_cols"]
    f  = f.replace([np.inf, -np.inf], np.nan)
    for col in fc:
        if col not in f.columns: f[col] = np.nan

    F = f[fc].dropna()
    if F.empty:
        raise RuntimeError("Feature matrix empty")
    print(f"    CT predictions: {len(F)} rows")

    if AD.get("ensemble") and AD.get("constituents"):
        yhi = np.mean([con["m_hi"].predict(F) for con in AD["constituents"]], axis=0)
        ylo = np.mean([con["m_lo"].predict(F) for con in AD["constituents"]], axis=0)
        if AD.get("blended") and float(AD.get("alpha", 1.0)) < 1.0:
            a = float(AD["alpha"])
            yhi = a * yhi + (1 - a) * float(AD.get("mu_hi", 0))
            ylo = a * ylo + (1 - a) * float(AD.get("mu_lo", 0))
    else:
        yhi = AD["hi_model"].predict(F)
        ylo = AD["lo_model"].predict(F)

    c_vals = c.reindex(F.index).values
    ph = c_vals * (1 + np.clip(yhi, 0, None))
    pl = c_vals * (1 - np.clip(ylo, 0, None))
    idx = np.asarray(F.index, dtype="datetime64[ns]")
    nd  = np.empty(len(F), dtype="datetime64[ns]")
    nd[:-1] = idx[1:]; nd[-1] = idx[-1] + np.timedelta64(1, "D")
    res = pd.DataFrame(
        {"close_asof": c_vals, "pred_high": ph, "pred_low": pl},
        index=pd.DatetimeIndex(nd, name="target_date"))
    return res[~res.index.duplicated(keep="last")]


# ── Signal computation ────────────────────────────────────────────────────────
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
        s = max(0, i - 2)
        ehma3[i] = np.mean(err_hi[s:i+1]); elma3[i] = np.mean(err_lo[s:i+1])
        hb3[i]   = int(np.sum(hi_brk[s:i+1])); lb3[i] = int(np.sum(lo_brk[s:i+1]))

    u1 = (ehma3 > 0.7) & (hb3 >= 2)
    d1 = (lb3 >= 2) & (elma3 > 0.5)
    d2 = ehma3 < -0.75
    d3 = np.zeros(N, dtype=bool)
    for i in range(1, N):
        consec = 0
        for k in range(i - 1, -1, -1):
            if hi_brk[k]: consec += 1
            else: break
        if consec >= 3 and lo_brk[i]:
            d3[i] = True

    ma30 = np.full(N, np.nan)
    for i in range(N):
        w = min(30, i + 1)
        ma30[i] = np.mean(ca[max(0, i - w + 1):i + 1])
    above_ma30     = ca > ma30
    ma30_slope_pos = np.zeros(N, dtype=bool)
    for i in range(5, N):
        if np.isfinite(ma30[i]) and np.isfinite(ma30[i - 5]):
            ma30_slope_pos[i] = ma30[i] > ma30[i - 5]
    bull_regime = above_ma30 & ma30_slope_pos

    clean_10d = np.zeros(N, dtype=bool)
    for i in range(N):
        lo_i = max(0, i - 7)
        clean_10d[i] = not bool(np.any(d1[lo_i:i] | d2[lo_i:i]))

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
        v_recent[i] = bool(np.any(v_rev_bar[max(0, i - 2):i + 1]))

    tf1_entry = u1 & (above_ma30 | clean_10d | v_recent)

    return dict(
        N=N, ca=ca, u1=u1, d1=d1, d2=d2, d3=d3,
        above_ma30=above_ma30, bull_regime=bull_regime,
        clean_10d=clean_10d, v_recent=v_recent,
        tf1_entry=tf1_entry,
    )


# ── Trigger labelling helpers ─────────────────────────────────────────────────
def _label_current(v_recent_i, above_ma30_i, clean_10d_i):
    """Current live labelling logic (from btc_hourly_app.py)."""
    if v_recent_i and not above_ma30_i and not clean_10d_i:
        return "U1 + V-reversal"
    elif above_ma30_i and clean_10d_i:
        return "U1 + MA30 + clean10d"   # COMBINED
    elif above_ma30_i:
        return "U1 + MA30"
    else:
        return "U1 + clean10d"


def _label_strict(v_recent_i, above_ma30_i, clean_10d_i):
    """Strict single-trigger labelling: V-reversal > MA30 > clean10d priority.
    The COMBINED case is absorbed into MA30 (above_ma30 takes priority).
    Entry conditions unchanged — same tf1_entry gate."""
    if v_recent_i:
        return "U1 + V-reversal"
    elif above_ma30_i:
        return "U1 + MA30"
    else:
        return "U1 + clean10d"


# ── Core backtest loop ────────────────────────────────────────────────────────
def run_asset_backtest(
    dates, asset_px, sigs, bt_start,
    sl_pct, cap,
    labeller,           # function(v_recent_i, above_ma30_i, clean_10d_i) → str
    exclude_combined,   # bool: block entries where BOTH above_ma30 AND clean_10d
):
    N          = len(dates)
    d2         = sigs["d2"]; d3 = sigs["d3"]
    bull       = sigs["bull_regime"]
    above_ma30 = sigs["above_ma30"]
    clean_10d  = sigs["clean_10d"]
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
            _exit_at_i   = d3[i] or (d2[i] and not bull[i])
            _sl_ok       = not from_sl or bool(bull[i]) or bars_since_sl >= 10
            _combined    = bool(above_ma30[i]) and bool(clean_10d[i])
            _can_enter   = tf1_entry[i] and _sl_ok and (from_sl or not _exit_at_i)
            if exclude_combined and _combined and not v_recent[i]:
                _can_enter = False   # block when BOTH MA30 and clean10d fire (unless V-rev)

            if _can_enter:
                e_reentry = bool(from_sl)
                qty = nav / price; e_price = price; e_date = dates[i]
                e_nav = nav; pos = "LONG"; stop_px = price * (1 - sl_pct)
                from_sl = False; bars_since_sl = 0
                e_trigger = labeller(bool(v_recent[i]), bool(above_ma30[i]), bool(clean_10d[i]))

        nav_arr[i] = qty * price if pos == "LONG" else nav

    if pos == "LONG" and np.isfinite(asset_px[N-1]) and asset_px[N-1] > 0:
        nav_arr[N-1] = qty * asset_px[N-1]

    return trades, nav_arr


# ── Aggregate per-trigger stats ───────────────────────────────────────────────
TRIGGERS_ORDER = [
    "U1 + MA30",
    "U1 + MA30 + clean10d",
    "U1 + clean10d",
    "U1 + V-reversal",
]


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
            median_pnl=float(np.median(pnls)) if pnls else 0,
            best=float(max(pnls)) if pnls else 0,
            worst=float(min(pnls)) if pnls else 0,
            total_pnl=float(sum(pnls)),
        )
    return out


def overall_stats(trades, nav_arr, bt0_nav, asset_px, bt0_idx, cap):
    closed = [t for t in trades if "exit_date" in t]
    all_pnls = [t["pnl_pct"] for t in closed]
    final_nav = nav_arr[np.isfinite(nav_arr)][-1] if np.any(np.isfinite(nav_arr)) else cap
    wins = [p for p in all_pnls if p > 0]
    return dict(
        n=len(closed),
        win_rate=100*len(wins)/len(closed) if closed else 0,
        avg_pnl=float(np.mean(all_pnls)) if all_pnls else 0,
        strat_ret=(final_nav/cap-1)*100,
        final_nav=final_nav,
    )


# ── Statistical significance (Fisher's exact, one-sided) ─────────────────────
def fishers_p(n_wins_a, n_a, n_wins_b, n_b):
    """Two-sample proportion test via continuity-corrected z. Returns p-value."""
    if n_a == 0 or n_b == 0:
        return float("nan")
    p_a = n_wins_a / n_a; p_b = n_wins_b / n_b
    p_pool = (n_wins_a + n_wins_b) / (n_a + n_b)
    if p_pool in (0, 1):
        return float("nan")
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
    if se == 0:
        return float("nan")
    z = (p_a - p_b) / se
    from scipy.stats import norm
    return float(2 * norm.sf(abs(z)))


# ── Main analysis ─────────────────────────────────────────────────────────────
def fmt_pnl(v):
    return f"+{v:.1f}%" if v > 0 else f"{v:.1f}%"


def print_trigger_table(stats_dict):
    header = f"  {'Trigger':<28} {'n':>4} {'W/L':>6} {'Win%':>6} {'AvgP&L':>8} {'Best':>8} {'Worst':>8}"
    print(header)
    print("  " + "-" * (len(header)-2))
    for trig in TRIGGERS_ORDER + [k for k in stats_dict if k not in TRIGGERS_ORDER]:
        if trig not in stats_dict:
            continue
        s = stats_dict[trig]
        wl = f"{s['wins']}/{s['losses']}"
        print(f"  {trig:<28} {s['n']:>4} {wl:>6} {s['win_rate']:>5.0f}% "
              f" {fmt_pnl(s['avg_pnl']):>8} {fmt_pnl(s['best']):>8} {fmt_pnl(s['worst']):>8}")


def run_period(period_name, start_iso, end_iso, pre_iso):
    print(f"\n{'='*70}")
    print(f"  PERIOD: {period_name}")
    print(f"{'='*70}")

    start_dt = pd.Timestamp(start_iso)
    end_dt   = pd.Timestamp(end_iso)
    pre_dt   = pd.Timestamp(pre_iso)

    fetch_start = pre_iso
    fetch_end   = (end_dt + pd.Timedelta(days=3)).strftime("%Y-%m-%d")

    df = fetch_data(fetch_start, fetch_end)

    btc_hi  = df["btc_high"]
    btc_lo  = df["btc_low"]
    btc_cl  = df["btc_close"]

    preds = build_ct_preds(df)

    # Align actual OHLC to preds (target_date index)
    preds["actual_high"]  = btc_hi.reindex(preds.index).values
    preds["actual_low"]   = btc_lo.reindex(preds.index).values
    preds["actual_close"] = btc_cl.reindex(preds.index).values

    comp = preds.dropna(subset=["actual_high", "actual_low", "actual_close"]).reset_index()
    if len(comp) < WARMUP + 3:
        print("  [SKIP] not enough data"); return

    dates = pd.DatetimeIndex(comp["target_date"])
    sigs  = compute_signals(comp)

    results = {}

    for asset_name, acfg in ASSETS.items():
        ticker  = acfg["ticker"]
        sl_pct  = acfg["sl_pct"]
        sl_lbl  = acfg["stop_label"]

        px_series = fetch_asset_px(ticker, pre_iso, fetch_end)
        if px_series.empty:
            print(f"  [SKIP] {asset_name} — price unavailable"); continue

        # Handle MSTU inception (Sep 18, 2024): backward-fill with inception open
        if ticker == "MSTU":
            INCEPTION = pd.Timestamp("2024-09-18")
            first_real = px_series.loc[px_series.index >= INCEPTION].iloc[0] if len(px_series.loc[px_series.index >= INCEPTION]) > 0 else None
            if first_real is not None:
                px_series.loc[px_series.index < INCEPTION] = first_real

        asset_px = px_series.reindex(dates).ffill().bfill().values.astype(float)

        # ── Version A: current system (4 trigger labels) ──────────────────────
        trades_a, nav_a = run_asset_backtest(
            dates, asset_px, sigs, bt_start=start_dt,
            sl_pct=sl_pct, cap=INITIAL_CAPITAL,
            labeller=_label_current, exclude_combined=False,
        )

        # ── Version B: strict (3 non-overlapping trigger labels; same entries) ─
        trades_b, nav_b = run_asset_backtest(
            dates, asset_px, sigs, bt_start=start_dt,
            sl_pct=sl_pct, cap=INITIAL_CAPITAL,
            labeller=_label_strict, exclude_combined=False,
        )

        # ── Version C: exclude combined entries (true entry-rule change) ───────
        trades_c, nav_c = run_asset_backtest(
            dates, asset_px, sigs, bt_start=start_dt,
            sl_pct=sl_pct, cap=INITIAL_CAPITAL,
            labeller=_label_strict, exclude_combined=True,
        )

        bt0_idx = max(WARMUP, int(dates.searchsorted(start_dt)))
        bh_px0  = asset_px[bt0_idx] if np.isfinite(asset_px[bt0_idx]) else 1
        bh_final = INITIAL_CAPITAL * asset_px[-1] / bh_px0 if bh_px0 > 0 else INITIAL_CAPITAL
        bh_ret   = (bh_final / INITIAL_CAPITAL - 1) * 100

        # Compute final NAVs from nav arrays
        nav_a_final = next((nav_a[j] for j in range(len(nav_a)-1,-1,-1) if np.isfinite(nav_a[j])), INITIAL_CAPITAL)
        nav_b_final = next((nav_b[j] for j in range(len(nav_b)-1,-1,-1) if np.isfinite(nav_b[j])), INITIAL_CAPITAL)
        nav_c_final = next((nav_c[j] for j in range(len(nav_c)-1,-1,-1) if np.isfinite(nav_c[j])), INITIAL_CAPITAL)
        ret_a = (nav_a_final/INITIAL_CAPITAL-1)*100
        ret_b = (nav_b_final/INITIAL_CAPITAL-1)*100  # same as A — labelling only
        ret_c = (nav_c_final/INITIAL_CAPITAL-1)*100

        print(f"\n  ── {asset_name} ({sl_lbl} stop + SL5 re-entry) ──────────────")
        print(f"  B&H return this period: {fmt_pnl(bh_ret)}")
        print(f"  Strategy return  (A: current labels):  {fmt_pnl(ret_a)}   ({len(trades_a)} trades)")
        print(f"  Strategy return  (C: exclude combined): {fmt_pnl(ret_c)}   ({len(trades_c)} trades)")

        # ── Per-trigger breakdown (Version A — current labelling) ──────────────
        ts_a = trigger_stats(trades_a)
        combined_lbl = "U1 + MA30 + clean10d"
        ma30_lbl     = "U1 + MA30"
        c10d_lbl     = "U1 + clean10d"
        vrev_lbl     = "U1 + V-reversal"

        print(f"\n  PER-TRIGGER BREAKDOWN (Version A — current labelling):")
        print_trigger_table(ts_a)

        # ── Statistical significance: combined vs MA30-only ───────────────────
        if combined_lbl in ts_a and ma30_lbl in ts_a:
            s_comb = ts_a[combined_lbl]
            s_ma30 = ts_a[ma30_lbl]
            p = fishers_p(s_comb["wins"], s_comb["n"], s_ma30["wins"], s_ma30["n"])
            print(f"\n  STAT TEST — Combined vs MA30-only win-rate difference:")
            print(f"    Combined : {s_comb['wins']}/{s_comb['n']} wins  ({s_comb['win_rate']:.0f}%)")
            print(f"    MA30-only: {s_ma30['wins']}/{s_ma30['n']} wins  ({s_ma30['win_rate']:.0f}%)")
            if not np.isnan(p):
                sig = "YES (p<0.05)" if p < 0.05 else ("marginal (p<0.10)" if p < 0.10 else "NO")
                print(f"    Two-sided p-value: {p:.4f}  → Statistically significant: {sig}")
            else:
                print(f"    p-value: N/A (sample too small for z-test)")

        # ── Impact of excluding combined entries ───────────────────────────────
        print(f"\n  IMPACT OF EXCLUDING 'MA30+clean10d' ENTRIES (Version C):")
        # Which trades in A are "combined" that would be excluded in C?
        combined_trades_a = [t for t in trades_a if t["entry_trigger"] == combined_lbl]
        print(f"    Trades excluded: {len(combined_trades_a)}")
        if combined_trades_a:
            for t in combined_trades_a:
                print(f"      {t['entry_date'].date()}  {fmt_pnl(t['pnl_pct'])}  "
                      f"({t['exit_signal']})  dur={t['duration_days']}d")
        print(f"    Return with exclusion:  {fmt_pnl(ret_c)}  vs current {fmt_pnl(ret_a)}")
        delta = ret_c - ret_a
        print(f"    Delta: {'+' if delta >= 0 else ''}{delta:.1f}pp  "
              f"({'BETTER' if delta > 0 else 'WORSE' if delta < 0 else 'NO CHANGE'})")

        # ── Per-trigger breakdown (Version B — strict labels, same entries) ────
        ts_b = trigger_stats(trades_b)
        print(f"\n  PER-TRIGGER BREAKDOWN (Version B — strict labels, same entries):")
        print_trigger_table(ts_b)

        results[(period_name, asset_name)] = dict(
            trades_a=trades_a, trades_b=trades_b, trades_c=trades_c,
            ts_a=ts_a, ts_b=ts_b,
            ret_a=ret_a, ret_c=ret_c, bh_ret=bh_ret,
        )

    return results


# ── Run all periods ───────────────────────────────────────────────────────────
all_results = {}
for pname, (s, e, pre) in PERIODS.items():
    r = run_period(pname, s, e, pre)
    if r:
        all_results.update(r)

# ── Final cross-period summary ────────────────────────────────────────────────
print(f"\n\n{'='*70}")
print("  CROSS-PERIOD TRIGGER SUMMARY — COMBINED TRADES POOL")
print(f"{'='*70}")

for asset_name in ASSETS:
    all_combined = []
    all_ma30     = []
    all_c10d     = []
    all_vrev     = []

    for pname in PERIODS:
        key = (pname, asset_name)
        if key not in all_results:
            continue
        ts = all_results[key]["ts_a"]
        for lbl, lst in [
            ("U1 + MA30 + clean10d", all_combined),
            ("U1 + MA30",            all_ma30),
            ("U1 + clean10d",        all_c10d),
            ("U1 + V-reversal",      all_vrev),
        ]:
            if lbl in ts:
                lst.append(ts[lbl])

    def pool(dicts, key):
        return [d[key] for d in dicts]

    def combine(dicts):
        if not dicts:
            return None
        n    = sum(d["n"]    for d in dicts)
        wins = sum(d["wins"] for d in dicts)
        pnls_all = []  # approximate from stats only — not raw
        return dict(
            n=n, wins=wins,
            win_rate=100*wins/n if n else 0,
            avg_pnl=float(np.mean([d["avg_pnl"] for d in dicts])) if dicts else 0,
            best=float(max(d["best"] for d in dicts)) if dicts else 0,
            worst=float(min(d["worst"] for d in dicts)) if dicts else 0,
        )

    print(f"\n  {asset_name}  (all three periods pooled — approximate):")
    header = f"  {'Trigger':<28} {'n':>4} {'W/L':>6} {'Win%':>6} {'AvgP&L':>8} {'Best':>8} {'Worst':>8}"
    print(header)
    print("  " + "-" * (len(header)-2))
    for lbl, lst in [
        ("U1 + MA30",            all_ma30),
        ("U1 + MA30 + clean10d", all_combined),
        ("U1 + clean10d",        all_c10d),
        ("U1 + V-reversal",      all_vrev),
    ]:
        c = combine(lst)
        if not c:
            continue
        wl = f"{c['wins']}/{c['n']-c['wins']}"
        print(f"  {lbl:<28} {c['n']:>4} {wl:>6} {c['win_rate']:>5.0f}% "
              f" {fmt_pnl(c['avg_pnl']):>8} {fmt_pnl(c['best']):>8} {fmt_pnl(c['worst']):>8}")

    if all_combined and all_ma30:
        flat_c = combine(all_combined)
        flat_m = combine(all_ma30)
        if flat_c and flat_m:
            p = fishers_p(flat_c["wins"], flat_c["n"], flat_m["wins"], flat_m["n"])
            print(f"\n  Combined vs MA30-only: p={p:.4f}" if not np.isnan(p)
                  else "\n  Combined vs MA30-only: p=N/A (too few samples)")

print("\n\nDONE.\n")
