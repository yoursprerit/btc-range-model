#!/usr/bin/env python3
"""
COUNTERFACTUAL THRESHOLD-WINDOW EVALUATION  (read-only, nothing touches the app)
===============================================================================
Question
--------
The live BTC/MSTR/MSTU signature thresholds (U1, D2, D1, D3, V-reversal, MA trend
filter, per-asset stops) were chosen to maximize FULL-period backtest return over
Jun 2024 -> May 31 2026.

Counterfactual: if instead they were chosen to maximize the FULL period ending
Feb 28 2026 (which is ALSO the CT model's training cutoff), how would the OOS
backtest (Mar 2026 -> present) compare with the existing documented OOS results?

Why this matters
----------------
The CT H/L model is trained through Feb 28 2026, so Mar 2026 -> present is the
model's genuine out-of-sample window.  BUT the *thresholds* were tuned on the full
period through May 31 2026 -- i.e. they were allowed to peek at Mar-May 2026, the
very OOS window whose numbers the docs advertise.  Refitting the thresholds only
through Feb 28 makes Mar-> present truly out-of-sample for BOTH the model AND the
thresholds.  The gap between the two = the threshold "look-ahead" premium baked
into the documented OOS numbers.

Engine is identical to scripts/compare_variants_full.py (same CT artefact, same
feature build, same signal math, same UI-exact backtests) -- only the U1/D2/stop
thresholds are parametrized and the optimization/eval windows are varied.
"""

import sys, warnings, itertools
warnings.filterwarnings("ignore")
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

try:
    import sklearn._loss._loss as _e
    if "_loss" not in sys.modules: sys.modules["_loss"] = _e
    import sklearn._loss.loss, sklearn._loss.link
except Exception: pass

import json, joblib
import numpy as np
import pandas as pd

DATA_DIR   = _ROOT / "data" / "backtest"
MODEL_PATH = _ROOT / "models" / "inference_assets_ct.joblib"

WARMUP          = 35
INITIAL_CAPITAL = 100_000.0
CAP             = INITIAL_CAPITAL

# ---- structural (non-tuned) signal constants, held at live values --------------
D1_ERRLO_MIN = 0.5      # D1 low-band accumulation
V_DN_SCORE   = 0.8      # V-reversal capitulation score
V_ERRLO_MIN  = 3.0      # V-reversal single-day low undershoot

# ------------------------------------------------------------------ data loaders
def _load_csv(fname):
    df = pd.read_csv(DATA_DIR / fname, index_col=0, parse_dates=True)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    df.columns = [c.lower() for c in df.columns]
    return df

btc_csv      = _load_csv("btc_usd_daily.csv")
mstr_csv     = _load_csv("mstr_daily.csv")
mstu_csv     = _load_csv("mstu_synthetic_daily.csv")
raw_feat_csv = _load_csv("raw_features_daily.csv")
manifest     = json.loads((DATA_DIR / "manifest.json").read_text())
AD           = joblib.load(str(MODEL_PATH))


# ------------------------------------------------------- CT predictions (as UI)
def _build_preds(start_dt, end_dt, AD):
    pre_dt  = start_dt - pd.Timedelta(days=60)
    fetch_s = start_dt - pd.Timedelta(days=200)
    fetch_e = end_dt   + pd.Timedelta(days=3)
    df = raw_feat_csv.loc[(raw_feat_csv.index >= fetch_s) &
                          (raw_feat_csv.index <= fetch_e)].copy()
    c = df["btc_close"]; h = df["btc_high"]; l_ = df["btc_low"]; v = df["btc_volume"]
    ret = np.log(c).diff(); feat = pd.DataFrame(index=df.index)
    for k in [1,3,5,7,14,30]: feat[f"ret_{k}"] = ret.rolling(k).sum()
    for k in [5,10,20,30]:    feat[f"vol_{k}"] = ret.rolling(k).std()
    prev_c = c.shift(1)
    tr = pd.concat([(h-l_),(h-prev_c).abs(),(l_-prev_c).abs()], axis=1).max(axis=1)
    for k in [7,14,30]: feat[f"atr_{k}"] = tr.rolling(k).mean() / c
    feat["range_today"] = (h-l_)/c
    feat["range_ma7"]   = ((h-l_)/c).rolling(7).mean()
    feat["range_ma30"]  = ((h-l_)/c).rolling(30).mean()
    feat["range_std30"] = ((h-l_)/c).rolling(30).std()
    gain = c.diff().clip(lower=0).rolling(14).mean()
    loss = (-c.diff().clip(upper=0)).rolling(14).mean()
    feat["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    e12 = c.ewm(span=12, adjust=False).mean(); e26 = c.ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    feat["macd"] = macd / c
    feat["macd_sig"] = macd.ewm(span=9, adjust=False).mean() / c
    feat["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    ma20 = c.rolling(20).mean(); sd20 = c.rolling(20).std()
    feat["bb_width"] = (4*sd20) / ma20
    feat["dist_hi_30"] = c / c.rolling(30).max() - 1
    feat["dist_lo_30"] = c / c.rolling(30).min() - 1
    feat["dist_hi_90"] = c / c.rolling(90).max() - 1
    feat["vol_chg_1"] = np.log(v).diff()
    feat["vol_z_20"] = (np.log(v) - np.log(v).rolling(20).mean()) / np.log(v).rolling(20).std()
    feat["vol_ma_ratio"] = v / v.rolling(20).mean()
    dow = df.index.dayofweek
    for i in range(6): feat[f"dow_{i}"] = (dow == i).astype(float)
    for nm in ["spx","ndx","vix","gold","dxy","tnx","eth"]:
        col = f"{nm}_close"
        if col not in df.columns: continue
        s = df[col]; lr = np.log(s).diff()
        for k in [1,5,20]: feat[f"{nm}_ret_{k}"] = lr.rolling(k).sum()
        feat[f"{nm}_vol_20"] = lr.rolling(20).std()
    for cnm, ccol in [("spx","spx_close"),("ndx","ndx_close"),("gold","gold_close"),("dxy","dxy_close")]:
        if ccol in df.columns:
            feat[f"btc_{cnm}_corr_30"] = ret.rolling(30).corr(np.log(df[ccol]).diff())
    for col in [x for x in df.columns if x.startswith("oc_")]:
        s = df[col].astype(float); sl = np.log(s.replace(0, np.nan))
        feat[f"{col}_d1"] = sl.diff(1); feat[f"{col}_d7"] = sl.diff(7)
        feat[f"{col}_z30"] = (sl - sl.rolling(30).mean()) / sl.rolling(30).std()
    nh = h.shift(-1); nl = l_.shift(-1)
    y_hi = (nh - c) / c; y_lo = (c - nl) / c
    feat["y_hi_ema3"] = y_hi.shift(1).ewm(span=3, adjust=False).mean()
    feat["y_lo_ema3"] = y_lo.shift(1).ewm(span=3, adjust=False).mean()
    feat["y_hi_ema7"] = y_hi.shift(1).ewm(span=7, adjust=False).mean()
    feat["y_lo_ema7"] = y_lo.shift(1).ewm(span=7, adjust=False).mean()
    p3h = h.shift(1).rolling(3).max(); p3l = l_.shift(1).rolling(3).min()
    feat["above_3d_high"] = (c > p3h).astype(float)
    feat["below_3d_low"]  = (c < p3l).astype(float)
    feat["bo_strength_up"] = (c / p3h - 1).clip(lower=0)
    feat["bo_strength_dn"] = (1 - c / p3l).clip(lower=0)
    yhl = y_hi.shift(1); yll = y_lo.shift(1)
    feat["y_hi_surprise"] = yhl - yhl.ewm(span=7, adjust=False).mean()
    feat["y_lo_surprise"] = yll - yll.ewm(span=7, adjust=False).mean()
    nr = ret.clip(upper=0)
    feat["dn_vol_5"] = nr.rolling(5).std(); feat["dn_vol_20"] = nr.rolling(20).std()
    sma50 = c.rolling(50).mean()
    feat["below_sma50"] = (c < sma50).astype(float)
    feat["below_sma50_5d"] = feat["below_sma50"].rolling(5).min().fillna(0)
    for cb in ["cb_premium", "cb_premium_ma3", "cb_premium_z7"]:
        feat[cb] = df[cb].fillna(0.0)
    fc = AD["feat_cols"]; feat = feat.replace([np.inf, -np.inf], np.nan)
    for col in fc:
        if col not in feat.columns: feat[col] = np.nan
    F = feat[fc].dropna()
    if AD.get("ensemble") and AD.get("constituents"):
        yhi = np.mean([con["m_hi"].predict(F) for con in AD["constituents"]], axis=0)
        ylo = np.mean([con["m_lo"].predict(F) for con in AD["constituents"]], axis=0)
        if AD.get("blended") and float(AD.get("alpha", 1.0)) < 1.0:
            a = float(AD["alpha"])
            yhi = a*yhi + (1-a)*float(AD.get("mu_hi", 0))
            ylo = a*ylo + (1-a)*float(AD.get("mu_lo", 0))
    else:
        yhi = AD["hi_model"].predict(F); ylo = AD["lo_model"].predict(F)
    c_vals = c.reindex(F.index).values
    ph = c_vals * (1 + np.clip(yhi, 0, None)); pl = c_vals * (1 - np.clip(ylo, 0, None))
    idx = np.asarray(F.index, dtype="datetime64[ns]")
    nd = np.empty(len(F), dtype="datetime64[ns]"); nd[:-1] = idx[1:]; nd[-1] = idx[-1] + np.timedelta64(1,"D")
    preds = pd.DataFrame({"close_asof": c_vals, "pred_high": ph, "pred_low": pl},
                         index=pd.DatetimeIndex(nd, name="target_date"))
    preds = preds[~preds.index.duplicated(keep="last")]
    preds2 = preds.loc[(preds.index >= pre_dt) & (preds.index <= end_dt)].copy()
    preds2["actual_high"]  = h.reindex(preds2.index).values
    preds2["actual_low"]   = l_.reindex(preds2.index).values
    preds2["actual_close"] = c.reindex(preds2.index).values
    comp = preds2.dropna(subset=["actual_high","actual_low","actual_close"]).reset_index()
    return comp


# ---------- signal computation, U1_min & D2_max parametrized -------------------
def compute_signals(comp, u1_min, d2_max):
    N = len(comp)
    c_asof = comp["close_asof"].values.astype(float)
    ph = comp["pred_high"].values.astype(float); pl = comp["pred_low"].values.astype(float)
    ah = comp["actual_high"].values.astype(float); al = comp["actual_low"].values.astype(float)
    err_hi = (ah - ph) / c_asof * 100
    err_lo = (pl - al) / c_asof * 100
    hi_brk = (ah > ph).astype(int); lo_brk = (al < pl).astype(int)
    ehma3 = np.zeros(N); elma3 = np.zeros(N)
    hb3 = np.zeros(N, dtype=int); lb3 = np.zeros(N, dtype=int)
    for i in range(N):
        s = max(0, i-2)
        ehma3[i] = np.mean(err_hi[s:i+1]); elma3[i] = np.mean(err_lo[s:i+1])
        hb3[i] = int(np.sum(hi_brk[s:i+1])); lb3[i] = int(np.sum(lo_brk[s:i+1]))
    u1 = (ehma3 > u1_min) & (hb3 >= 2)
    d1 = (lb3 >= 2) & (elma3 > D1_ERRLO_MIN)
    d2 = ehma3 < d2_max
    d3 = np.zeros(N, dtype=bool)
    for i in range(1, N):
        c2 = 0
        for k in range(i-1, -1, -1):
            if hi_brk[k]: c2 += 1
            else: break
        if c2 >= 3 and lo_brk[i]: d3[i] = True
    ma30 = np.full(N, np.nan)
    for i in range(N):
        w = min(30, i+1); ma30[i] = np.mean(c_asof[max(0, i-w+1):i+1])
    above_ma30 = c_asof > ma30
    ma30_slope_pos = np.zeros(N, dtype=bool)
    for i in range(N):
        if i >= 5 and np.isfinite(ma30[i]) and np.isfinite(ma30[i-5]):
            ma30_slope_pos[i] = ma30[i] > ma30[i-5]
    bull_regime = above_ma30 & ma30_slope_pos
    clean_7d = np.zeros(N, dtype=bool)
    for i in range(N):
        lo_i = max(0, i-7); clean_7d[i] = not bool(np.any(d1[lo_i:i] | d2[lo_i:i]))
    roll_norm = np.array([float(np.mean(err_hi[max(0,i-29):i+1])) for i in range(N)])
    dn_score = np.zeros(N)
    for i in range(N):
        norm = max(abs(roll_norm[i]), 0.01)
        dn_score[i] = ((-ehma3[i]/norm)*0.30 + (lb3[i]/3.0)*0.30 +
                       (elma3[i]/max(abs(elma3[i]),0.10))*0.20 + float(lo_brk[i])*0.20)
    v_rev_bar = (dn_score > V_DN_SCORE) & (err_lo > V_ERRLO_MIN)
    v_recent = np.zeros(N, dtype=bool)
    for i in range(N):
        v_recent[i] = bool(np.any(v_rev_bar[max(0, i-2):i+1]))
    tf_current = u1 & ((above_ma30 ^ clean_7d) | v_recent)   # live unified Standard-MA gate
    return dict(d2=d2, d3=d3, bull_regime=bull_regime, tf=tf_current)


# ------------------------------------------------ backtests (UI-exact) ---------
def run_btc(dates, px, sigs, bt_start, bt_end):
    N = len(dates); d2 = sigs["d2"]; d3 = sigs["d3"]; bull = sigs["bull_regime"]; tf = sigs["tf"]
    i0 = max(WARMUP, int(pd.DatetimeIndex(dates).searchsorted(bt_start)))
    iN = int(pd.DatetimeIndex(dates).searchsorted(bt_end, side="right"))
    nav = CAP; pos = "CASH"; qty = 0.0; e_price = None; trades = []; nav_arr = np.full(N, np.nan)
    for i in range(N):
        price = px[i]
        if i < i0 or i >= iN:
            nav_arr[i] = (qty*price if pos=="LONG" else nav); continue
        if not np.isfinite(price) or price <= 0:
            nav_arr[i] = qty*px[i-1] if pos=="LONG" else nav; continue
        if pos == "LONG":
            if bool(d3[i] or (d2[i] and not bull[i])):
                nav = qty*price; trades.append((price/e_price-1)*100); pos="CASH"; qty=0.0
            else: nav = qty*price
        else:
            if tf[i] and not (d3[i] or (d2[i] and not bull[i])):
                qty = nav/price; e_price = price; pos="LONG"
        nav_arr[i] = qty*price if pos=="LONG" else nav
    if pos == "LONG": nav = qty*px[min(iN,N)-1]
    return trades, nav


def run_sl(dates, px, sigs, sl_pct, bt_start, bt_end):
    N = len(dates); d2 = sigs["d2"]; d3 = sigs["d3"]; bull = sigs["bull_regime"]; tf = sigs["tf"]
    i0 = max(WARMUP, int(pd.DatetimeIndex(dates).searchsorted(bt_start)))
    iN = int(pd.DatetimeIndex(dates).searchsorted(bt_end, side="right"))
    nav = CAP; pos = "CASH"; qty = 0.0; stop_px = 0.0; from_sl = False; bars_since_sl = 0
    e_price = None; trades = []; nav_arr = np.full(N, np.nan)
    use_stop = sl_pct is not None
    for i in range(N):
        price = px[i]
        if i < i0 or i >= iN:
            nav_arr[i] = (qty*price if pos=="LONG" else nav); continue
        if not np.isfinite(price) or price <= 0:
            nav_arr[i] = qty*px[i-1] if pos=="LONG" else nav; continue
        if pos == "LONG":
            if use_stop and price <= stop_px:
                nav = qty*price; trades.append((price/e_price-1)*100)
                pos="CASH"; qty=0.0; stop_px=0.0; from_sl=True; bars_since_sl=0
            elif bool(d3[i] or (d2[i] and not bull[i])):
                nav = qty*price; trades.append((price/e_price-1)*100)
                pos="CASH"; qty=0.0; stop_px=0.0; from_sl=False
            else: nav = qty*price
        else:
            if from_sl: bars_since_sl += 1
            _exit_at_i = d3[i] or (d2[i] and not bull[i])
            _sl_ok = (not from_sl or bool(bull[i]) or bars_since_sl >= 10)
            if tf[i] and _sl_ok and (from_sl or not _exit_at_i):
                qty = nav/price; e_price = price; pos="LONG"
                stop_px = price*(1-sl_pct) if use_stop else 0.0
                from_sl=False; bars_since_sl=0
        nav_arr[i] = qty*price if pos=="LONG" else nav
    if pos == "LONG": nav = qty*px[min(iN,N)-1]
    return trades, nav


def ret_of(nav): return (nav/CAP - 1)*100

# ------------------------------------------------------------------ build once
FULL_START = pd.Timestamp("2024-06-01")
DATA_END   = pd.Timestamp("2026-07-10")
comp = _build_preds(FULL_START, DATA_END, AD)
dates = pd.DatetimeIndex(comp["target_date"]); N = len(comp)
btc_px  = btc_csv["close"].reindex(dates).ffill().bfill().values.astype(float)
_mstr = mstr_csv["close"].sort_index().reindex(pd.date_range(mstr_csv.index[0], mstr_csv.index[-1], freq="D")).ffill()
mstr_px = _mstr.reindex(dates).ffill().bfill().values.astype(float)
_mstu = mstu_csv["close"].sort_index().reindex(pd.date_range(mstu_csv.index[0], mstu_csv.index[-1], freq="D")).ffill()
mstu_px = _mstu.reindex(dates).ffill().bfill().values.astype(float)

print(f"comp {N} rows {dates[0].date()} -> {dates[-1].date()}")

# ------------------------------------------------------------------ grids
U1_GRID = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1]
D2_GRID = [-0.5, -0.75, -1.0, -1.3, -1.5, -1.8, -2.1]
MSTR_STOPS = [None, 0.03, 0.05, 0.07]
MSTU_STOPS = [None, 0.03, 0.05, 0.07, 0.10]

# cache signals per (u1,d2)
_sig_cache = {}
def sigs_for(u1, d2):
    key = (u1, d2)
    if key not in _sig_cache: _sig_cache[key] = compute_signals(comp, u1, d2)
    return _sig_cache[key]

def perf(u1, d2, mstr_sl, mstu_sl, start, end):
    s = sigs_for(u1, d2)
    _, nb = run_btc(dates, btc_px, s, start, end)
    _, nr = run_sl(dates, mstr_px, s, mstr_sl, start, end)
    _, nu = run_sl(dates, mstu_px, s, mstu_sl, start, end)
    rb, rr, ru = ret_of(nb), ret_of(nr), ret_of(nu)
    return rb, rr, ru

def optimize(train_end, label):
    """Shared U1/D2 across assets; per-asset best stop. Objective = sum of 3 full-period returns."""
    best = None
    for u1, d2 in itertools.product(U1_GRID, D2_GRID):
        # per-asset best stop on the training window
        best_r = {"MSTR": (-1e9, None), "MSTU": (-1e9, None)}
        s = sigs_for(u1, d2)
        _, nb = run_btc(dates, btc_px, s, FULL_START, train_end); rb = ret_of(nb)
        for sl in MSTR_STOPS:
            _, nr = run_sl(dates, mstr_px, s, sl, FULL_START, train_end); r = ret_of(nr)
            if r > best_r["MSTR"][0]: best_r["MSTR"] = (r, sl)
        for sl in MSTU_STOPS:
            _, nu = run_sl(dates, mstu_px, s, sl, FULL_START, train_end); r = ret_of(nu)
            if r > best_r["MSTU"][0]: best_r["MSTU"] = (r, sl)
        total = rb + best_r["MSTR"][0] + best_r["MSTU"][0]
        cfg = dict(u1=u1, d2=d2, mstr_sl=best_r["MSTR"][1], mstu_sl=best_r["MSTU"][1],
                   train=dict(BTC=rb, MSTR=best_r["MSTR"][0], MSTU=best_r["MSTU"][0]), total=total)
        if best is None or total > best["total"]: best = cfg
    print(f"\n[{label}] optimize on FULL Jun-2024 -> {train_end.date()}")
    print(f"  winning shared thresholds: U1 > {best['u1']:+.2f} / D2 < {best['d2']:+.2f}"
          f" | MSTR stop {best['mstr_sl']} | MSTU stop {best['mstu_sl']}")
    print(f"  train-window return  BTC {best['train']['BTC']:+.1f}%  MSTR {best['train']['MSTR']:+.1f}%  MSTU {best['train']['MSTU']:+.1f}%")
    return best

# ------------------------------------------------------------------ run
OOS_START = pd.Timestamp("2026-03-01")   # day after model training cutoff (Feb 28 2026)
OOS_END   = DATA_END

CFG_LIVE = dict(u1=1.3, d2=-1.3, mstr_sl=0.03, mstu_sl=0.03, name="LIVE (documented)")

print("="*92)
print("  COUNTERFACTUAL: threshold optimization window   Feb-28-2026  vs  May-31-2026")
print("  OOS eval window (post model-training cutoff): 2026-03-01 -> 2026-07-10, fresh $100k")
print("="*92)

cfg_may = optimize(pd.Timestamp("2026-05-31"), "OPT-through-MAY-31")
cfg_feb = optimize(pd.Timestamp("2026-02-28"), "OPT-through-FEB-28")

def show_oos(cfg, name):
    rb, rr, ru = perf(cfg["u1"], cfg["d2"], cfg["mstr_sl"], cfg["mstu_sl"], OOS_START, OOS_END)
    print(f"  {name:<34} U1>{cfg['u1']:+.2f}/D2<{cfg['d2']:+.2f} "
          f"stops M{cfg['mstr_sl']}/U{cfg['mstu_sl']:} ->  "
          f"BTC {rb:+7.1f}%   MSTR {rr:+8.1f}%   MSTU {ru:+8.1f}%")
    return rb, rr, ru

print("\n" + "="*92)
print("  OOS BACKTEST RESULTS  (2026-03-01 -> 2026-07-10, fresh capital at OOS start)")
print("="*92)
# B&H over OOS
def bh(px):
    i0 = max(WARMUP, int(dates.searchsorted(OOS_START)))
    iN = int(dates.searchsorted(OOS_END, side="right"))-1
    return (px[iN]/px[i0]-1)*100
print(f"  Buy & Hold over OOS:  BTC {bh(btc_px):+.1f}%   MSTR {bh(mstr_px):+.1f}%   MSTU {bh(mstu_px):+.1f}%\n")
r_live = show_oos(CFG_LIVE, "LIVE / documented (opt thru May)")
r_may  = show_oos(cfg_may,  "Re-derived opt-thru-May-31")
r_feb  = show_oos(cfg_feb,  "Counterfactual opt-thru-Feb-28")

print("\n" + "-"*92)
print("  OOS delta: counterfactual(Feb) MINUS live(documented)")
print(f"    BTC {r_feb[0]-r_live[0]:+.1f}pp   MSTR {r_feb[1]-r_live[1]:+.1f}pp   MSTU {r_feb[2]-r_live[2]:+.1f}pp")
print("-"*92)

# also report full-through-May and full-through-Feb for both configs (context)
print("\n  FULL-period return by config (context):")
for cfg, nm in [(CFG_LIVE,"live"), (cfg_may,"opt-May"), (cfg_feb,"opt-Feb")]:
    a = perf(cfg["u1"],cfg["d2"],cfg["mstr_sl"],cfg["mstu_sl"], FULL_START, pd.Timestamp("2026-05-31"))
    b = perf(cfg["u1"],cfg["d2"],cfg["mstr_sl"],cfg["mstu_sl"], FULL_START, pd.Timestamp("2026-02-28"))
    print(f"    {nm:<9} thru-May BTC{a[0]:+7.1f}/MSTR{a[1]:+7.1f}/MSTU{a[2]:+7.1f}   "
          f"thru-Feb BTC{b[0]:+7.1f}/MSTR{b[1]:+7.1f}/MSTU{b[2]:+7.1f}")


# =====================================================================================
#  ROBUSTNESS — is the Feb-optimum a knife-edge, and what drives the OOS gap?
# =====================================================================================
def _perf(cfg, start, end): return perf(cfg["u1"], cfg["d2"], cfg["mstr_sl"], cfg["mstu_sl"], start, end)

FEB = pd.Timestamp("2026-02-28"); MAY = pd.Timestamp("2026-05-31"); JUL = DATA_END

# enumerate ALL (U1, D2, MSTR-stop, MSTU-stop) configs, scored on the FEB-train objective
rows = []
for u1, d2 in itertools.product(U1_GRID, D2_GRID):
    s = sigs_for(u1, d2)
    _, nb = run_btc(dates, btc_px, s, FULL_START, FEB); rb = ret_of(nb)
    for msl, usl in itertools.product(MSTR_STOPS, MSTU_STOPS):
        _, nr = run_sl(dates, mstr_px, s, msl, FULL_START, FEB)
        _, nu = run_sl(dates, mstu_px, s, usl, FULL_START, FEB)
        rows.append(dict(u1=u1, d2=d2, mstr_sl=msl, mstu_sl=usl,
                         feb_total=rb + ret_of(nr) + ret_of(nu)))
dfc = pd.DataFrame(rows).sort_values("feb_total", ascending=False).reset_index(drop=True)

print("\n" + "="*92)
print("  TOP-10 configs by FEB-train objective -> their genuine OOS (Mar1->Jul10) returns")
print("="*92)
print(f"  {'rk':>2} {'U1':>5} {'D2':>6} {'Mst':>5} {'Ust':>5} {'febTot':>7} | {'BTC':>6} {'MSTR':>7} {'MSTU':>7} {'sum':>7}")
for i in range(10):
    c = dfc.iloc[i].to_dict(); ob, om, ou = _perf(c, OOS_START, JUL)
    print(f"  {i+1:>2} {c['u1']:>+5.2f} {c['d2']:>+6.2f} {str(c['mstr_sl']):>5} {str(c['mstu_sl']):>5} "
          f"{c['feb_total']:>7.0f} | {ob:>+6.1f} {om:>+7.1f} {ou:>+7.1f} {ob+om+ou:>+7.1f}")
ndec = max(1, len(dfc)//10)
oos_sums = [sum(_perf(dfc.iloc[i].to_dict(), OOS_START, JUL)) for i in range(ndec)]
print(f"\n  Top-decile ({ndec} configs) OOS sum:  median {np.median(oos_sums):+.1f}%   mean {np.mean(oos_sums):+.1f}%")
mask = ((dfc.u1==1.3)&(dfc.d2==-1.3)&(dfc.mstr_sl==0.03)&(dfc.mstu_sl==0.03))
print(f"  LIVE config rank on FEB objective: {int(dfc.index[mask][0])+1} of {len(dfc)}  "
      f"(feb_total {dfc[mask].feb_total.values[0]:.0f} vs top {dfc.feb_total.iloc[0]:.0f})")

print("\n" + "="*92)
print("  D2 EXIT SWEEP (U1=+1.3, stops 3/3): FEB-train objective vs genuine OOS(Mar1->Jul10)")
print("  -> the entire optimization-window gap lives in this one threshold")
print("="*92)
print(f"  {'D2':>6} {'febTrainSum':>12} {'oosSum':>9}")
for d2 in D2_GRID:
    c = dict(u1=1.3, d2=d2, mstr_sl=0.03, mstu_sl=0.03)
    ft = sum(_perf(c, FULL_START, FEB)); oo = sum(_perf(c, OOS_START, JUL))
    tag = "  <- FEB argmax" if d2 == cfg_feb["d2"] else ("  <- LIVE (opt-thru-May)" if d2 == -1.3 else "")
    print(f"  {d2:>+6.2f} {ft:>12.0f} {oo:>+9.1f}{tag}")
