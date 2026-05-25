"""Research script:
  1. Does adding energy/commodity features (crude oil, natural gas, XLE)
     improve 14-day BTC forward-return prediction?
  2. What quantile band level historically identifies decisive BTC breakouts
     (uptrend / downtrend)?
"""
import sys, warnings
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone

warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

HORIZON = 14

# ────────────────────────────────────────────────────────
# 1. FETCH DATA
# ────────────────────────────────────────────────────────
print(">>> Fetching data from Yahoo Finance …")
tickers = {
    "BTC":  "BTC-USD",
    "SPX":  "^GSPC",
    "NDX":  "^IXIC",
    "VIX":  "^VIX",
    "GOLD": "GC=F",
    "DXY":  "DX-Y.NYB",
    "TNX":  "^TNX",
    "ETH":  "ETH-USD",
    # --- Energy / Commodities ---
    "CL":   "CL=F",    # Crude Oil (WTI front-month futures)
    "NG":   "NG=F",    # Natural Gas front-month futures
    "XLE":  "XLE",     # Energy Select Sector SPDR ETF
    "BNO":  "BNO",     # Brent Oil ETF (proxy for Brent crude)
}

def _dl(sym, start="2017-01-01"):
    d = yf.download(sym, start=start, progress=False, auto_adjust=True)
    if d.empty:
        return None
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] for c in d.columns]
    d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
    return d["Close"].sort_index().rename(sym)

raw = {}
for name, sym in tickers.items():
    s = _dl(sym)
    if s is not None:
        raw[name] = s
        print(f"  {name:6s} ({sym:15s}): {len(s)} bars  "
              f"{s.index.min().date()} → {s.index.max().date()}")
    else:
        print(f"  {name:6s} ({sym:15s}): *** EMPTY ***")

df = pd.DataFrame(raw).sort_index().loc["2019-01-01":]
df = df.ffill(limit=5)

# ────────────────────────────────────────────────────────
# 2. BUILD FEATURES
# ────────────────────────────────────────────────────────
print("\n>>> Engineering features …")
c    = df["BTC"]
h_s  = c.rolling(2).max()    # approximate daily high
l_s  = c.rolling(2).min()    # approximate daily low
ret  = np.log(c).diff()

# Volatility regime feature (same as production)
rng  = ret.abs()              # proxy range
rm30 = rng.rolling(30).mean()

def log_ret_features(s, name, ks=(1, 5, 10, 20)):
    r = np.log(s).diff()
    out = {}
    for k in ks:
        out[f"{name}_ret_{k}"] = r.rolling(k).sum()
    out[f"{name}_vol_20"] = r.rolling(20).std()
    return out

feats = {}

# BTC features
for k in [1, 3, 5, 7, 14]:
    feats[f"btc_ret_{k}"] = ret.rolling(k).sum()
for k in [5, 10, 20, 30]:
    feats[f"btc_vol_{k}"] = ret.rolling(k).std()
feats["btc_rsi_14"] = _rsi = pd.Series(dtype=float, index=c.index)
delta = c.diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = (-delta.clip(upper=0)).rolling(14).mean()
rs    = gain / loss.replace(0, np.nan)
feats["btc_rsi_14"] = 100 - 100 / (1 + rs)
feats["range_ma30"] = rm30

# Macro features (existing)
for nm in ["SPX", "NDX", "VIX", "GOLD", "DXY", "TNX", "ETH"]:
    if nm in df.columns:
        feats.update(log_ret_features(df[nm], nm.lower()))

# --- ENERGY / COMMODITY features (new) ---
energy_added = []
for nm in ["CL", "NG", "XLE", "BNO"]:
    if nm in df.columns:
        feats.update(log_ret_features(df[nm], nm.lower()))
        energy_added.append(nm)

print(f"  Energy features added for: {energy_added}")

# Target
feats_df = pd.DataFrame(feats, index=df.index)
feats_df["y_logret_14"] = np.log(c.shift(-HORIZON) / c)
feats_df = feats_df.replace([np.inf, -np.inf], np.nan).dropna()
print(f"  Feature matrix: {feats_df.shape}")

# ────────────────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT  (same convention as production)
# ────────────────────────────────────────────────────────
TODAY      = pd.Timestamp("today").normalize()
test_start = TODAY - pd.DateOffset(months=8)
train_end  = test_start - pd.Timedelta(days=HORIZON)

train = feats_df.loc[:train_end]
test  = feats_df.loc[test_start:]
print(f"\n  TRAIN {train.index.min().date()} → {train.index.max().date()}  n={len(train)}")
print(f"  TEST  {test.index.min().date()} → {test.index.max().date()}   n={len(test)}")

FEAT_COLS = [c for c in feats_df.columns if c != "y_logret_14"]
X_tr = train[FEAT_COLS]; y_tr = train["y_logret_14"]
X_te = test[FEAT_COLS];  y_te = test["y_logret_14"]

# Baseline regime-cone (existing, no features)
edges = np.quantile(train["range_ma30"], [1/3, 2/3])
def to_regime(x):
    return np.searchsorted(edges, np.asarray(x, dtype=float), side="right").clip(0, 2)

train_r = to_regime(train["range_ma30"])
test_r  = to_regime(test["range_ma30"])

regime_med = {r: float(np.median(y_tr.values[train_r == r])) for r in range(3)}
baseline_pred_te  = np.array([regime_med[r] for r in test_r])
baseline_pred_tr  = np.array([regime_med[r] for r in train_r])

def mape(pred, actual):
    return float(np.mean(np.abs(pred - actual) / np.abs(actual)) * 100)

def dir_acc(pred, actual):
    return float(100 * (np.sign(pred) == np.sign(actual)).mean())

def band_cov(pred, actual, band=0.16):
    return float(100 * ((actual >= pred - band) & (actual <= pred + band)).mean())

print("\n" + "="*70)
print("  PART 1: FEATURE ABLATION — does adding energy features help?")
print("="*70)

results_mape = {}
results_dir  = {}
results_cov  = {}

# --- Baseline: regime cone (no features) ---
bm_te = mape(baseline_pred_te, y_te.values)
bd_te = dir_acc(baseline_pred_te, y_te.values)
bc_te = band_cov(baseline_pred_te, y_te.values)
results_mape["Regime cone (baseline)"]   = bm_te
results_dir["Regime cone (baseline)"]    = bd_te
results_cov["Regime cone (baseline)"]    = bc_te
print(f"\n  Regime cone (baseline): MAPE={bm_te:.2f}%  DirAcc={bd_te:.1f}%  Cov={bc_te:.1f}%")

# --- Model helper ---
def fit_eval(X_tr_, y_tr_, X_te_, y_te_, label, n_est=200, depth=3, lr=0.05):
    m = Pipeline([("sc", StandardScaler()),
                  ("m", GradientBoostingRegressor(
                      n_estimators=n_est, max_depth=depth,
                      learning_rate=lr, subsample=0.8, random_state=42))])
    m.fit(X_tr_, y_tr_)
    pred = m.predict(X_te_)
    ma = mape(pred, y_te_.values)
    da = dir_acc(pred, y_te_.values)
    co = band_cov(pred, y_te_.values)
    results_mape[label] = ma
    results_dir[label]  = da
    results_cov[label]  = co
    print(f"  {label:40s}: MAPE={ma:.2f}%  DirAcc={da:.1f}%  Cov={co:.1f}%")
    return m, pred

# Feature groups
btc_only   = [f for f in FEAT_COLS if f.startswith("btc_") or f == "range_ma30"]
macro_only  = [f for f in FEAT_COLS if any(f.startswith(nm.lower()+"_") for nm in ["spx","ndx","vix","gold","dxy","tnx","eth"])]
energy_only = [f for f in FEAT_COLS if any(f.startswith(nm.lower()+"_") for nm in ["cl","ng","xle","bno"])]
no_energy   = [f for f in FEAT_COLS if f not in energy_only]
all_feats   = FEAT_COLS

print(f"\n  Feature counts: BTC-only={len(btc_only)}  Macro={len(macro_only)}  "
      f"Energy={len(energy_only)}  No-energy={len(no_energy)}  All={len(all_feats)}")

fit_eval(X_tr[btc_only],   y_tr, X_te[btc_only],   y_te, "GBM  BTC-only features")
fit_eval(X_tr[no_energy],  y_tr, X_te[no_energy],  y_te, "GBM  All excl. energy")
fit_eval(X_tr[all_feats],  y_tr, X_te[all_feats],  y_te, "GBM  All incl. energy")
fit_eval(X_tr[energy_only+btc_only], y_tr, X_te[energy_only+btc_only], y_te,
         "GBM  BTC + energy only")

# Correlation analysis
print("\n  Energy feature correlations with 14-day BTC log-return (TRAIN):")
energy_feat_corrs = []
for f in energy_only:
    if f in train.columns:
        corr = float(train[f].corr(y_tr))
        energy_feat_corrs.append((abs(corr), corr, f))
energy_feat_corrs.sort(reverse=True)
for _, corr, f in energy_feat_corrs[:12]:
    print(f"    {f:30s}  corr = {corr:+.4f}")

# ────────────────────────────────────────────────────────
# PART 2: BREAKOUT THRESHOLD ANALYSIS
# ────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PART 2: BREAKOUT THRESHOLDS — what band level signals decisive moves?")
print("="*70)

y_14 = feats_df["y_logret_14"]
# Full dataset (train + test) for historical analysis
# (for thresholds we use train only to avoid look-ahead)
y_train_vals = y_tr.values

print(f"\n  Distribution of 14-day BTC log-returns (TRAIN, n={len(y_train_vals):,}):")
for q in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
    pct = np.quantile(y_train_vals, q)
    tag = ""
    if q <= 0.10: tag = " ← decisive DOWN"
    elif q <= 0.20: tag = " ← moderate DOWN"
    elif q >= 0.90: tag = " ← decisive UP"
    elif q >= 0.80: tag = " ← moderate UP"
    print(f"    q{q*100:5.1f}%  →  {pct*100:+7.2f}%{tag}")

# What % of time does BTC move more than X% in 14 days?
print("\n  Frequency of 14-day moves exceeding threshold (TRAIN):")
for thresh in [5, 8, 10, 12, 15, 18, 20, 25]:
    up_pct   = float(100 * (y_train_vals > np.log(1 + thresh/100)).mean())
    down_pct = float(100 * (y_train_vals < np.log(1 - thresh/100)).mean())
    print(f"    >{thresh:2d}% UP:  {up_pct:5.1f}%  |  >{thresh:2d}% DOWN:  {down_pct:5.1f}%")

# Regime-specific thresholds
print("\n  Regime-specific decisive-move thresholds (q10/q90 per regime, TRAIN):")
regime_labels = ["low vol", "mid vol", "high vol"]
for r in range(3):
    sub = y_train_vals[train_r == r]
    q10 = np.quantile(sub, 0.10); q90 = np.quantile(sub, 0.90)
    q20 = np.quantile(sub, 0.20); q80 = np.quantile(sub, 0.80)
    print(f"\n  Regime {r} [{regime_labels[r]:8s}]  n={len(sub):4d}")
    print(f"    Decisive DOWN (q10): {q10*100:+.1f}%  |  Moderate DOWN (q20): {q20*100:+.1f}%")
    print(f"    Decisive UP   (q90): {q90*100:+.1f}%  |  Moderate UP   (q80): {q80*100:+.1f}%")

# What's the optimal "decisive breakout" band for each regime?
print("\n  Suggested breakout thresholds (what %-move in 14 days is 'unusual'?):")
print("  [Using q20 for moderate, q10 for decisive — so only 10-20% of days qualify]")
for r in range(3):
    sub = y_train_vals[train_r == r]
    q10 = float(np.quantile(sub, 0.10)); q90 = float(np.quantile(sub, 0.90))
    # Convert to % moves (exp - 1)
    down_dec = (np.exp(q10) - 1) * 100
    up_dec   = (np.exp(q90) - 1) * 100
    print(f"    [{regime_labels[r]:8s}]  decisive DOWN < {down_dec:+.1f}%  |  decisive UP > {up_dec:+.1f}%")

print("\n  Summary table:")
print(f"  {'':30s}  {'Decisive DOWN':>15s}  {'Decisive UP':>15s}")
for r in range(3):
    sub = y_train_vals[train_r == r]
    q10 = float(np.quantile(sub, 0.10)); q90 = float(np.quantile(sub, 0.90))
    print(f"  {regime_labels[r]:30s}  {(np.exp(q10)-1)*100:>+14.1f}%  {(np.exp(q90)-1)*100:>+14.1f}%")

print("\n  Overall (regime-agnostic) decisive thresholds:")
q10_all = float(np.quantile(y_train_vals, 0.10)); q90_all = float(np.quantile(y_train_vals, 0.90))
q05_all = float(np.quantile(y_train_vals, 0.05)); q95_all = float(np.quantile(y_train_vals, 0.95))
print(f"    q10/q90: {(np.exp(q10_all)-1)*100:+.1f}%  /  {(np.exp(q90_all)-1)*100:+.1f}%")
print(f"    q05/q95: {(np.exp(q05_all)-1)*100:+.1f}%  /  {(np.exp(q95_all)-1)*100:+.1f}%")
print(f"\n  INTERPRETATION:")
print(f"  → A 14-day move EXCEEDING ±{abs((np.exp(q10_all)-1)*100):.0f}% (q10/q90) is in the outer 20% of history.")
print(f"  → A 14-day move EXCEEDING ±{abs((np.exp(q05_all)-1)*100):.0f}% (q05/q95) is truly extreme (outer 10%).")
print(f"  → For uptrend signal: close price +{(np.exp(q90_all)-1)*100:.0f}% above 14-day-ago level → top-decile bull run.")
print(f"  → For downtrend signal: close price {(np.exp(q10_all)-1)*100:.0f}% below 14-day-ago level → top-decile bear move.")
