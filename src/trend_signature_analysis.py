"""
Trend Signature Pattern Analysis
=================================
Mines all historical model predictions vs actuals for early-detection
signatures of significant uptrends and downtrends.

Sources used:
  - artifacts/artifacts.pkl  (test-set predictions + actuals, Sep 2025 – May 2026)
  - models/inference_assets_ct.joblib   (CT daily H/L + direction head)
  - models/inference_assets_7d_cone.joblib
  - models/inference_assets_14d_cone.joblib
  - models/inference_assets_3class.joblib
  - Live Yahoo Finance fetch for BTC close history (the full available window)
  - runtime/bookmarks.json   (ground-truth drastic event labels)
"""

import sys, warnings, json
from pathlib import Path
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import joblib
import yfinance as yf

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)
pd.set_option("display.float_format", "{:.4f}".format)

# ─────────────────────────────────────────────────────────────────────────────
# 0. LOAD ARTEFACTS
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("LOADING ARTIFACTS …")
print("=" * 70)

arts      = joblib.load(ROOT / "artifacts" / "artifacts.pkl")
ct_assets = joblib.load(ROOT / "models" / "inference_assets_ct.joblib")
a7d       = joblib.load(ROOT / "models" / "inference_assets_7d_cone.joblib")
a14d      = joblib.load(ROOT / "models" / "inference_assets_14d_cone.joblib")
a3c       = joblib.load(ROOT / "models" / "inference_assets_3class.joblib")
bookmarks = json.loads((ROOT / "runtime" / "bookmarks.json").read_text())

test_idx  = pd.DatetimeIndex(arts["test_index"])
high_true = np.array(arts["high_true"])
low_true  = np.array(arts["low_true"])
close_te  = np.array(arts["close_te"])

# Predicted high/low (price level) for each constituent model
ridge_phi = np.array(arts["rep"]["ridge"]["pred_hi"])
ridge_plo = np.array(arts["rep"]["ridge"]["pred_lo"])
gbm_phi   = np.array(arts["rep"]["gbm"]["pred_hi"])
gbm_plo   = np.array(arts["rep"]["gbm"]["pred_lo"])
rf_phi    = np.array(arts["rep"]["rf"]["pred_hi"])
rf_plo    = np.array(arts["rep"]["rf"]["pred_lo"])

# Ensemble average (simple mean of the 3)
ens_phi = (ridge_phi + gbm_phi + rf_phi) / 3
ens_plo = (ridge_plo + gbm_plo + rf_plo) / 3

print(f"Test window: {test_idx[0].date()} → {test_idx[-1].date()}  ({len(test_idx)} bars)")

# ─────────────────────────────────────────────────────────────────────────────
# 1. FETCH FULL BTC CLOSE HISTORY FROM YAHOO
# ─────────────────────────────────────────────────────────────────────────────
print("\n>>> Fetching BTC-USD daily close from Yahoo Finance …")
raw_yf = yf.download("BTC-USD", start="2019-01-01", progress=False, auto_adjust=True)
if isinstance(raw_yf.columns, pd.MultiIndex):
    raw_yf.columns = [c[0] for c in raw_yf.columns]
raw_yf.index = pd.to_datetime(raw_yf.index).tz_localize(None).normalize()
btc_full = raw_yf["Close"].dropna().sort_index()
btc_high_full = raw_yf["High"].dropna().sort_index()
btc_low_full  = raw_yf["Low"].dropna().sort_index()
print(f"   BTC daily close: {btc_full.index.min().date()} → {btc_full.index.max().date()}  ({len(btc_full)} rows)")

# ─────────────────────────────────────────────────────────────────────────────
# 2. BUILD TEST DATAFRAME WITH ALL PREDICTIONS vs ACTUALS
# ─────────────────────────────────────────────────────────────────────────────
df = pd.DataFrame({
    "close":     close_te,
    "high_true": high_true,
    "low_true":  low_true,
    "ens_phi":   ens_phi,
    "ens_plo":   ens_plo,
    "ridge_phi": ridge_phi,
    "ridge_plo": ridge_plo,
    "gbm_phi":   gbm_phi,
    "gbm_plo":   gbm_plo,
    "rf_phi":    rf_phi,
    "rf_plo":    rf_plo,
}, index=test_idx)

# Predicted midpoint (regime expectation of close)
df["ens_mid"]   = (df["ens_phi"] + df["ens_plo"]) / 2
df["ridge_mid"] = (df["ridge_phi"] + df["ridge_plo"]) / 2

# ─────────────────────────────────────────────────────────────────────────────
# 3. COMPUTE DIVERGENCE / BAND-POSITION FEATURES
# ─────────────────────────────────────────────────────────────────────────────
sigma_hi = ct_assets["sigma_hi"]   # ~0.0161
sigma_lo = ct_assets["sigma_lo"]   # ~0.0186

# --- A: Error features: actual vs predicted
df["err_hi_pct"]   = (df["high_true"] - df["ens_phi"]) / df["close"] * 100   # + means actual exceeded prediction
df["err_lo_pct"]   = (df["ens_plo"]   - df["low_true"]) / df["close"] * 100  # + means actual went lower than predicted
df["err_close_pct"]= (df["close"] - df["ens_mid"]) / df["ens_mid"] * 100     # + means close above midpoint

# --- B: Daily direction (actual return from prev close)
df["ret_1d"]        = df["close"].pct_change() * 100     # percent
df["ret_1d_log"]    = np.log(df["close"]).diff()

# --- C: Position of actual close within the prediction band
# Normalized position: -1 = at ens_plo, 0 = at midpoint, +1 = at ens_phi
df["band_width"]    = df["ens_phi"] - df["ens_plo"]
df["band_pos"]      = (df["close"] - df["ens_plo"]) / df["band_width"].replace(0, np.nan) * 2 - 1  # [-1, +1]

# --- D: Band cross signals
# How far did actual HIGH go above predicted HIGH (upper band break)?
df["hi_band_break"] = (df["high_true"] > df["ens_phi"]).astype(int)
df["lo_band_break"] = (df["low_true"]  < df["ens_plo"]).astype(int)
df["hi_exceed_pct"] = np.maximum(0, (df["high_true"] - df["ens_phi"]) / df["close"] * 100)
df["lo_exceed_pct"] = np.maximum(0, (df["ens_plo"]   - df["low_true"]) / df["close"] * 100)

# --- E: Model disagreement (inter-model spread)
df["phi_spread"]    = df[["ridge_phi","gbm_phi","rf_phi"]].max(axis=1) - df[["ridge_phi","gbm_phi","rf_phi"]].min(axis=1)
df["plo_spread"]    = df[["ridge_plo","gbm_plo","rf_plo"]].max(axis=1) - df[["ridge_plo","gbm_plo","rf_plo"]].min(axis=1)
df["phi_spread_pct"]= df["phi_spread"] / df["close"] * 100
df["plo_spread_pct"]= df["plo_spread"] / df["close"] * 100
df["model_disagree"]= df["phi_spread_pct"] + df["plo_spread_pct"]

# --- F: Rolling windows (momentum of errors)
for w in [3, 5, 7, 10]:
    df[f"err_hi_ma{w}"]      = df["err_hi_pct"].rolling(w).mean()
    df[f"err_lo_ma{w}"]      = df["err_lo_pct"].rolling(w).mean()
    df[f"err_close_ma{w}"]   = df["err_close_pct"].rolling(w).mean()
    df[f"band_pos_ma{w}"]    = df["band_pos"].rolling(w).mean()
    df[f"hi_breaks_{w}d"]    = df["hi_band_break"].rolling(w).sum()
    df[f"lo_breaks_{w}d"]    = df["lo_band_break"].rolling(w).sum()
    df[f"ret_ma{w}"]         = df["ret_1d"].rolling(w).mean()

# --- G: Cumulative error (trending bias signal)
df["err_close_cum3"]  = df["err_close_pct"].rolling(3).sum()
df["err_close_cum5"]  = df["err_close_pct"].rolling(5).sum()
df["err_hi_cum3"]     = df["err_hi_pct"].rolling(3).sum()
df["err_lo_cum3"]     = df["err_lo_pct"].rolling(3).sum()

# --- H: Direction sequence (consecutive up/down days)
def consecutive_streak(series):
    """Return today's streak length: +N if N consecutive up days, -N if N consecutive down days."""
    out = np.zeros(len(series), dtype=int)
    for i in range(1, len(series)):
        d = np.sign(series.iloc[i])
        if d == 0:
            out[i] = 0
        elif d == np.sign(series.iloc[i-1]) and out[i-1] != 0:
            out[i] = out[i-1] + d
        else:
            out[i] = d
    return pd.Series(out, index=series.index)

df["streak"] = consecutive_streak(df["ret_1d"])

# --- I: Band position streak (consecutive above/below midpoint)
df["band_pos_sign"] = np.sign(df["band_pos"])
df["band_streak"]   = consecutive_streak(df["band_pos"])

# --- J: Model bias streak (consecutive over/under-predict high)
df["err_hi_sign"]   = np.sign(df["err_hi_pct"])
df["err_hi_streak"] = consecutive_streak(df["err_hi_pct"].apply(np.sign).rename("x"))

# ─────────────────────────────────────────────────────────────────────────────
# 4. IDENTIFY SIGNIFICANT TREND EVENTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("4. IDENTIFYING SIGNIFICANT TREND EVENTS")
print("=" * 70)

# --- Method A: Bookmarked dates
bm_up   = pd.to_datetime([b["date"] for b in bookmarks.get("Drastic Climbs",[])])
bm_dn   = pd.to_datetime([b["date"] for b in bookmarks.get("Drastic Drops",[])])
bm_grad = pd.to_datetime([b["date"] for b in bookmarks.get("Gradual Climb",[])])

print(f"\nBookmarked Drastic Climbs:  {[d.date() for d in bm_up]}")
print(f"Bookmarked Drastic Drops:   {[d.date() for d in bm_dn]}")
print(f"Bookmarked Gradual Climbs:  {[d.date() for d in bm_grad]}")

# --- Method B: Systematic detection on full BTC history
# Define a "significant" single-day move as >4% (roughly 2.5σ on BTC)
# Define a "significant" multi-day trend as ≥10% over ≤7 days
def detect_trend_events(close_series, single_day_thresh=4.0, multiday_thresh=10.0, window=7):
    """
    Returns DataFrame of all significant up/down events:
      big_up, big_dn: single-day moves
      trend_up, trend_dn: multi-day moves
    """
    ret1 = close_series.pct_change() * 100
    ret_w = close_series.pct_change(window) * 100

    big_up   = ret1[ret1 > single_day_thresh]
    big_dn   = ret1[ret1 < -single_day_thresh]
    trend_up = ret_w[ret_w > multiday_thresh]
    trend_dn = ret_w[ret_w < -multiday_thresh]

    return {
        "big_up":   big_up.dropna(),
        "big_dn":   big_dn.dropna(),
        "trend_up": trend_up.dropna(),
        "trend_dn": trend_dn.dropna(),
    }

events_full = detect_trend_events(btc_full)
# Filter to test period only for signal comparison
events_test = {k: v[v.index.isin(test_idx)] for k, v in events_full.items()}

print(f"\nSystematic events in TEST period ({test_idx[0].date()} → {test_idx[-1].date()}):")
for k, v in events_test.items():
    print(f"  {k}: {len(v)} events | dates: {[d.date() for d in v.index[:15]]}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. PRE-EVENT WINDOW ANALYSIS (WHAT SIGNALS APPEARED N DAYS BEFORE TREND?)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("5. PRE-EVENT SIGNAL WINDOWS (N=1..5 days before event)")
print("=" * 70)

def pre_event_snapshot(df_test, event_dates, label, lookback=5):
    """
    For each event date, extract signal snapshots from lookback days before.
    Returns a DataFrame of pre-event rows with their signal values.
    """
    rows = []
    for ed in event_dates:
        if ed not in df_test.index:
            # Find nearest date in test index
            candidates = df_test.index[df_test.index <= ed]
            if len(candidates) == 0:
                continue
            ed = candidates[-1]
        pos = df_test.index.get_loc(ed)
        for lag in range(lookback, 0, -1):
            if pos - lag < 0:
                continue
            row = df_test.iloc[pos - lag].copy()
            row["event_date"] = ed
            row["event_label"] = label
            row["days_before"] = lag
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

sig_cols = [
    "err_hi_pct", "err_lo_pct", "err_close_pct",
    "band_pos", "band_streak",
    "hi_band_break", "lo_band_break",
    "err_hi_ma3", "err_hi_ma5",
    "err_close_ma3", "err_close_ma5",
    "band_pos_ma3", "band_pos_ma5",
    "hi_breaks_3d", "lo_breaks_3d",
    "hi_breaks_5d", "lo_breaks_5d",
    "ret_1d", "ret_ma3", "ret_ma5",
    "streak", "model_disagree",
    "err_close_cum3", "err_close_cum5",
    "err_hi_cum3", "err_lo_cum3",
]

# Combine all big trend events
up_dates   = list(events_test["big_up"].index) + list(events_test["trend_up"].index)
dn_dates   = list(events_test["big_dn"].index) + list(events_test["trend_dn"].index)
up_dates   = sorted(set(up_dates))
dn_dates   = sorted(set(dn_dates))

# Also add bookmarked dates that fall in test period
for d in bm_up:
    if d in test_idx and d not in up_dates:
        up_dates.append(d)
for d in bm_dn:
    if d in test_idx and d not in dn_dates:
        dn_dates.append(d)

pre_up = pre_event_snapshot(df, up_dates, "UPTREND")
pre_dn = pre_event_snapshot(df, dn_dates, "DOWNTREND")

print(f"\nUptrend events (test period): {len(up_dates)}")
print(f"Downtrend events (test period): {len(dn_dates)}")

if not pre_up.empty:
    print("\n--- Mean signal values N days BEFORE big UPTREND ---")
    for lag in range(5, 0, -1):
        sub = pre_up[pre_up["days_before"] == lag][sig_cols]
        if sub.empty: continue
        mean_vals = sub.mean()
        print(f"\n  T-{lag} days before uptrend (N={len(sub)}):")
        for col in sig_cols[:12]:
            print(f"    {col:30s}: {mean_vals[col]:+.4f}")

if not pre_dn.empty:
    print("\n--- Mean signal values N days BEFORE big DOWNTREND ---")
    for lag in range(5, 0, -1):
        sub = pre_dn[pre_dn["days_before"] == lag][sig_cols]
        if sub.empty: continue
        mean_vals = sub.mean()
        print(f"\n  T-{lag} days before downtrend (N={len(sub)}):")
        for col in sig_cols[:12]:
            print(f"    {col:30s}: {mean_vals[col]:+.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. COMPARE SIGNALS: UP vs DOWN vs NORMAL DAYS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("6. SIGNAL DISTRIBUTIONS: UPTREND vs DOWNTREND vs NORMAL (at T=1 day before)")
print("=" * 70)

# Label each test day
df["event_type"] = "NORMAL"
for d in up_dates:
    if d in df.index:
        # Mark T-1 and T-2 before event as "pre-uptrend"
        pos = df.index.get_loc(d)
        for lag in [1, 2]:
            if pos - lag >= 0:
                df.iloc[pos - lag, df.columns.get_loc("event_type")] = "PRE_UP"
        df.at[d, "event_type"] = "UPTREND"

for d in dn_dates:
    if d in df.index:
        pos = df.index.get_loc(d)
        for lag in [1, 2]:
            if pos - lag >= 0:
                cur = df.iloc[pos - lag]["event_type"]
                if cur == "NORMAL":
                    df.iloc[pos - lag, df.columns.get_loc("event_type")] = "PRE_DN"
        if df.at[d, "event_type"] == "NORMAL":
            df.at[d, "event_type"] = "DOWNTREND"

cat_counts = df["event_type"].value_counts()
print(f"\nDay categories:\n{cat_counts}\n")

key_signals = [
    "err_close_pct", "band_pos", "err_hi_pct", "err_lo_pct",
    "hi_band_break", "lo_band_break",
    "err_close_ma3", "band_pos_ma3", "hi_breaks_3d", "lo_breaks_3d",
    "streak", "model_disagree", "err_close_cum3",
]

print("Signal comparison by category (mean values):")
print(f"{'Signal':35s}  {'PRE_UP':>10} {'PRE_DN':>10} {'UPTREND':>10} {'DOWNTREND':>10} {'NORMAL':>10}")
print("-" * 90)
for sig in key_signals:
    if sig not in df.columns: continue
    vals = {}
    for cat in ["PRE_UP","PRE_DN","UPTREND","DOWNTREND","NORMAL"]:
        sub = df[df["event_type"] == cat][sig].dropna()
        vals[cat] = sub.mean() if len(sub) else np.nan
    print(f"{sig:35s}  {vals['PRE_UP']:+10.4f} {vals['PRE_DN']:+10.4f} {vals['UPTREND']:+10.4f} {vals['DOWNTREND']:+10.4f} {vals['NORMAL']:+10.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. 7-DAY CONE MODEL: BAND-CROSSING ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("7. 7-DAY CONE MODEL: WHEN DOES ACTUAL PRICE BREAK THE CONE?")
print("=" * 70)

regime_edges   = a7d["regime_edges"]    # two tercile thresholds for range_ma30
band_pct_7d    = a7d["band_pct"]        # 0.105
regime_stats_7d = a7d["regime_stats"]   # {0:..., 1:..., 2:...}

print(f"7d cone band: ±{band_pct_7d*100:.1f}%")
print(f"Regime edges (range_ma30): {regime_edges}")
for r, stats in regime_stats_7d.items():
    print(f"  Regime {r}: {stats}")

print(f"\n14d cone band: ±{a14d['band_pct']*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 8. DETAILED SIGNATURE PROFILES AROUND EACH BOOKMARKED EVENT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("8. DETAILED SIGNATURE PROFILES FOR EACH BOOKMARKED EVENT")
print("=" * 70)

all_bookmarked = (
    [(d, "DRASTIC_UP")   for d in bm_up]  +
    [(d, "DRASTIC_DN")   for d in bm_dn]  +
    [(d, "GRADUAL_UP")   for d in bm_grad]
)

sig_display = [
    "close", "ret_1d",
    "err_close_pct", "err_close_ma3", "err_close_ma5",
    "band_pos", "band_pos_ma3", "band_streak",
    "err_hi_pct", "err_lo_pct",
    "hi_band_break", "lo_band_break",
    "hi_breaks_3d", "hi_breaks_5d",
    "lo_breaks_3d", "lo_breaks_5d",
    "streak", "model_disagree",
    "err_hi_cum3", "err_lo_cum3",
]

for event_date, event_lbl in sorted(all_bookmarked, key=lambda x: x[0]):
    print(f"\n{'─'*60}")
    print(f"EVENT: {event_lbl}  on  {event_date.date()}")
    print(f"{'─'*60}")

    # Get window: 7 days before → 3 days after event
    window_start = event_date - pd.Timedelta(days=7)
    window_end   = event_date + pd.Timedelta(days=3)
    mask = (df.index >= window_start) & (df.index <= window_end)
    window = df[mask][sig_display].copy()

    if window.empty:
        print("  [Not in test period — using full-history close only]")
        # Show raw price return around this event
        wd = btc_full[(btc_full.index >= window_start) & (btc_full.index <= window_end)]
        print(f"  BTC Close around event:\n{wd.to_string()}")
        continue

    print(window.to_string())

    # Highlight what was happening in the 3 days before
    pre3 = df[mask & (df.index < event_date)].tail(3)
    if not pre3.empty:
        print(f"\n  PRE-EVENT SIGNALS (3 days before):")
        for col in ["err_close_ma3","band_pos_ma3","hi_breaks_3d","lo_breaks_3d","streak","model_disagree"]:
            if col in pre3.columns:
                vals = pre3[col].values
                print(f"    {col:30s}: {vals}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. STATISTICAL SIGNIFICANCE TEST: ARE PRE-EVENT SIGNALS DIFFERENT FROM NORMAL?
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("9. STATISTICAL SIGNIFICANCE OF PRE-EVENT SIGNALS")
print("=" * 70)

from scipy import stats as scipy_stats

def test_signal_diff(df, signal_col, cat_a, cat_b):
    a = df[df["event_type"] == cat_a][signal_col].dropna()
    b = df[df["event_type"] == cat_b][signal_col].dropna()
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan, len(a), len(b)
    u, p = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
    return a.mean() - b.mean(), p, len(a), len(b)

print(f"\n{'Signal':35s}  {'ΔMean(PRE_UP-NORMAL)':>22} {'p-val':>8}  {'ΔMean(PRE_DN-NORMAL)':>22} {'p-val':>8}")
print("-" * 110)
for sig in key_signals:
    if sig not in df.columns: continue
    dm_up, p_up, n_up, n_n  = test_signal_diff(df, sig, "PRE_UP",   "NORMAL")
    dm_dn, p_dn, n_dn, _    = test_signal_diff(df, sig, "PRE_DN",   "NORMAL")
    sig_up = "**" if (not np.isnan(p_up) and p_up < 0.05) else ("*" if (not np.isnan(p_up) and p_up < 0.10) else "  ")
    sig_dn = "**" if (not np.isnan(p_dn) and p_dn < 0.05) else ("*" if (not np.isnan(p_dn) and p_dn < 0.10) else "  ")
    dm_up_str = f"{dm_up:+.4f}" if not np.isnan(dm_up) else "    N/A"
    dm_dn_str = f"{dm_dn:+.4f}" if not np.isnan(dm_dn) else "    N/A"
    p_up_str  = f"{p_up:.4f}" if not np.isnan(p_up) else "   N/A"
    p_dn_str  = f"{p_dn:.4f}" if not np.isnan(p_dn) else "   N/A"
    print(f"{sig:35s}  {dm_up_str:>20}{sig_up}  {p_up_str:>8}  {dm_dn_str:>20}{sig_dn}  {p_dn_str:>8}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. COMPOSITE EARLY-WARNING SCORES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("10. COMPOSITE EARLY-WARNING SCORES")
print("=" * 70)

# UPTREND score: high when conditions favour an imminent uptrend
# Based on signals that showed highest discriminative power
df["score_up"] = (
    df["err_close_ma3"]         * 0.30 +   # systematic under-prediction → bullish
    df["band_pos_ma3"]          * 0.25 +   # price hugging upper band → bullish
    df["hi_breaks_3d"]          * 0.20 +   # repeated high-band breaks → bullish
    df["err_hi_ma3"]            * 0.15 +   # actual H exceeds prediction → bullish
    (df["streak"].clip(0,5))    * 0.10     # consecutive up days → momentum
)

# DOWNTREND score: high when conditions favour an imminent downtrend
df["score_dn"] = (
    (-df["err_close_ma3"])      * 0.30 +   # systematic over-prediction → bearish
    (-df["band_pos_ma3"])       * 0.25 +   # price hugging lower band → bearish
    df["lo_breaks_3d"]          * 0.20 +   # repeated low-band breaks → bearish
    df["err_lo_ma3"]            * 0.15 +   # actual L undershoots → bearish
    (-df["streak"].clip(-5,0))  * 0.10     # consecutive down days → momentum
)

# Normalize 0→1
for col in ["score_up","score_dn"]:
    mn, mx = df[col].min(), df[col].max()
    df[f"{col}_norm"] = (df[col] - mn) / (mx - mn + 1e-9)

print("\nTop 10 days by UPTREND score (test period):")
top_up = df.nlargest(10, "score_up_norm")[["close","ret_1d","score_up_norm","band_pos_ma3",
                                           "err_close_ma3","hi_breaks_3d","streak"]]
print(top_up.to_string())

print("\nTop 10 days by DOWNTREND score (test period):")
top_dn = df.nlargest(10, "score_dn_norm")[["close","ret_1d","score_dn_norm","band_pos_ma3",
                                           "err_close_ma3","lo_breaks_3d","streak"]]
print(top_dn.to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 11. PATTERN CLASSIFIER ACCURACY EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("11. PATTERN CLASSIFIER ACCURACY — DOES HIGH SCORE PRECEDE ACTUAL TREND?")
print("=" * 70)

# Build ground-truth: did the next 3 days see a significant move?
df["fwd_ret_3d"]   = df["close"].shift(-3) / df["close"] - 1
df["fwd_big_up"]   = (df["fwd_ret_3d"] > 0.04).astype(int)
df["fwd_big_dn"]   = (df["fwd_ret_3d"] < -0.04).astype(int)

# --- Evaluate uptrend score
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
print("\nUptrend score vs. next-3d return >4%:")
print(f"{'Threshold':>12} {'Triggered':>10} {'Hit Rate':>10} {'Base Rate':>10} {'Lift':>8}")
base_up = df["fwd_big_up"].mean()
for thr in thresholds:
    triggered = df["score_up_norm"] >= thr
    if triggered.sum() < 5: continue
    hit_rate = df.loc[triggered, "fwd_big_up"].mean()
    lift = hit_rate / base_up if base_up > 0 else np.nan
    print(f"{thr:12.1f} {triggered.sum():10d} {hit_rate:10.3f} {base_up:10.3f} {lift:8.2f}x")

print("\nDowntrend score vs. next-3d return < -4%:")
print(f"{'Threshold':>12} {'Triggered':>10} {'Hit Rate':>10} {'Base Rate':>10} {'Lift':>8}")
base_dn = df["fwd_big_dn"].mean()
for thr in thresholds:
    triggered = df["score_dn_norm"] >= thr
    if triggered.sum() < 5: continue
    hit_rate = df.loc[triggered, "fwd_big_dn"].mean()
    lift = hit_rate / base_dn if base_dn > 0 else np.nan
    print(f"{thr:12.1f} {triggered.sum():10d} {hit_rate:10.3f} {base_dn:10.3f} {lift:8.2f}x")

# ─────────────────────────────────────────────────────────────────────────────
# 12. CHRONOLOGICAL REPLAY OF KEY EVENTS (SHOW SIGNALS IN REAL TIME)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("12. CHRONOLOGICAL SIGNAL REPLAY — KEY EVENT WINDOWS")
print("=" * 70)

# Events to replay: all known drastic events ± 5 days
key_events = {
    "2025-06-22 DRASTIC DROP":  "2025-06-22",
    "2025-06-23 DRASTIC CLIMB": "2025-06-23",
    "2025-07-10 DRASTIC CLIMB": "2025-07-10",
    "2025-07-13 DRASTIC CLIMB": "2025-07-13",
    "2025-10-07 DRASTIC DROP":  "2025-10-07",
    "2026-01-31 DRASTIC DROP":  "2026-01-31",
    "2026-02-03 DRASTIC DROP":  "2026-02-03",
    "2026-02-04 DRASTIC DROP":  "2026-02-04",
    "2026-02-05 DRASTIC DROP":  "2026-02-05",
    "2026-02-06 DRASTIC CLIMB": "2026-02-06",
    "2026-05-04 GRADUAL CLIMB": "2026-05-04",
}

show_cols = [
    "close", "ret_1d", "streak",
    "err_close_pct", "err_close_ma3",
    "band_pos", "band_pos_ma3",
    "hi_band_break", "lo_band_break",
    "hi_breaks_3d", "lo_breaks_3d",
    "score_up_norm", "score_dn_norm",
]

for event_name, event_date_str in key_events.items():
    event_date = pd.Timestamp(event_date_str)
    w_start = event_date - pd.Timedelta(days=5)
    w_end   = event_date + pd.Timedelta(days=2)
    mask    = (df.index >= w_start) & (df.index <= w_end)
    w       = df[mask][show_cols]

    if w.empty:
        # Show from btc_full for context
        wbt = btc_full[(btc_full.index >= w_start) & (btc_full.index <= w_end)]
        print(f"\n{'─'*60}")
        print(f"[{event_name}]  (not in test period)")
        r1 = wbt.pct_change() * 100
        for d, c, r in zip(wbt.index, wbt.values, r1.values):
            marker = " ◄ EVENT" if d.date() == event_date.date() else ""
            print(f"  {d.date()}  close={c:,.0f}  ret={r:+.2f}%{marker}")
        continue

    print(f"\n{'─'*60}")
    print(f"[{event_name}]")
    print(f"{'─'*60}")

    # Print day-by-day
    for i, row in w.iterrows():
        marker = " ◄ EVENT" if i.date() == event_date.date() else ""
        pre_mrk = " [PRE-EVENT]" if (event_date - i).days in [1,2,3] else ""
        print(
            f"  {i.date()}{marker}{pre_mrk}  "
            f"close={row['close']:>8,.0f}  "
            f"ret={row['ret_1d']:+6.2f}%  "
            f"streak={int(row['streak']):+3d}  "
            f"err_cl={row['err_close_ma3']:+7.3f}  "
            f"band={row['band_pos_ma3']:+6.3f}  "
            f"hiBreak3d={int(row['hi_breaks_3d']) if not np.isnan(row['hi_breaks_3d']) else 'NA':>2}  "
            f"loBreak3d={int(row['lo_breaks_3d']) if not np.isnan(row['lo_breaks_3d']) else 'NA':>2}  "
            f"↑score={row['score_up_norm']:.3f}  "
            f"↓score={row['score_dn_norm']:.3f}"
        )

# ─────────────────────────────────────────────────────────────────────────────
# 13. FINAL SUMMARY: CONCLUSIVE SIGNATURE PATTERNS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("13. FINAL SUMMARY — CONCLUSIVE SIGNATURE PATTERNS")
print("=" * 70)

# Run a proper forward-return evaluation for each signal threshold combo
print("\n--- UPTREND EARLY-WARNING CHECKLIST (signal measured T-1, T-2, T-3) ---")

# For each signal, compute hit rate at various lags before big up moves
for lag in [1, 2, 3]:
    df_lag = df.copy()
    df_lag["target_up"] = (df["fwd_ret_3d"].shift(-(lag-1)) > 0.04).astype(int) if lag > 0 else df["fwd_big_up"]

    print(f"\n  Lag={lag} day(s) ahead (signal today → big move within {3+lag-1}d):")
    signals_to_test = [
        ("err_close_ma3 > +0.5", df["err_close_ma3"] > 0.5),
        ("band_pos_ma3 > +0.3",  df["band_pos_ma3"]  > 0.3),
        ("hi_breaks_3d >= 2",    df["hi_breaks_3d"]  >= 2),
        ("hi_breaks_5d >= 3",    df["hi_breaks_5d"]  >= 3),
        ("streak >= +3",         df["streak"]        >= 3),
        ("err_hi_ma3 > +0.5",    df["err_hi_ma3"]    > 0.5),
        ("score_up_norm > 0.5",  df["score_up_norm"] > 0.5),
    ]
    for sig_name, sig_mask in signals_to_test:
        n_trig  = sig_mask.sum()
        if n_trig < 3: continue
        n_hit   = df.loc[sig_mask, "fwd_big_up"].sum()
        hit_rt  = n_hit / n_trig
        base    = df["fwd_big_up"].mean()
        lift    = hit_rt / base if base > 0 else 0
        print(f"    {sig_name:35s}: trig={n_trig:4d}  hit={n_hit:3d}  rate={hit_rt:.3f}  lift={lift:.2f}x")

print("\n--- DOWNTREND EARLY-WARNING CHECKLIST ---")
for lag in [1, 2, 3]:
    df_lag = df.copy()
    print(f"\n  Lag={lag} day(s) ahead:")
    signals_to_test = [
        ("err_close_ma3 < -0.5", df["err_close_ma3"] < -0.5),
        ("band_pos_ma3 < -0.3",  df["band_pos_ma3"]  < -0.3),
        ("lo_breaks_3d >= 2",    df["lo_breaks_3d"]  >= 2),
        ("lo_breaks_5d >= 3",    df["lo_breaks_5d"]  >= 3),
        ("streak <= -3",         df["streak"]        <= -3),
        ("err_lo_ma3 > +0.5",    df["err_lo_ma3"]    > 0.5),
        ("score_dn_norm > 0.5",  df["score_dn_norm"] > 0.5),
    ]
    for sig_name, sig_mask in signals_to_test:
        n_trig  = sig_mask.sum()
        if n_trig < 3: continue
        n_hit   = df.loc[sig_mask, "fwd_big_dn"].sum()
        hit_rt  = n_hit / n_trig
        base    = df["fwd_big_dn"].mean()
        lift    = hit_rt / base if base > 0 else 0
        print(f"    {sig_name:35s}: trig={n_trig:4d}  hit={n_hit:3d}  rate={hit_rt:.3f}  lift={lift:.2f}x")

print("\n--- 3-CLASS DAY-TYPE MODEL CALIBRATION ---")
print(f"Quiet threshold: {a3c['quiet_threshold']:.4f} (= {a3c['quiet_threshold']*100:.2f}%)")
print(f"Class labels:    {a3c['class_labels']}")
print(f"Regime edges:    {a3c['regime_edges']}")
cm = a3c["calibration_meta"]
print(f"Train: {cm['train_start']} → {cm['train_end']}  (n={cm['train_n']})")
print(f"Test:  {cm['test_start']} → {cm['test_end']}  (n={cm['test_n']})")
print(f"Test accuracy: {cm['test_accuracy_pct']:.1f}%")
print(f"Selective ({cm['selective']}): see notes")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
