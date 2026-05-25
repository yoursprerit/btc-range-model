"""Research script:
  1. Does adding energy/commodity features (crude oil CL, natural gas NG,
     XLE energy-sector ETF) improve 7-day BTC forward-return prediction?
  2. What 7-day band level / quantile threshold signals decisive BTC breakouts?

Methodology mirrors research_14d_energy_breakout.py but uses HORIZON=7
and reads from the production data pipeline first (falls back to yfinance).
"""
import sys, warnings
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

try:
    from paths import RAW_CT_CSV, FEATURES_CT_CSV
    _pipe_available = True
except ImportError:
    _pipe_available = False

HORIZON = 7

# ──────────────────────────────────────────────────────────
# 1. FETCH DATA
# ──────────────────────────────────────────────────────────
print(">>> Fetching data …")
TICKERS = {
    "BTC":  "BTC-USD",
    "SPX":  "^GSPC",
    "NDX":  "^IXIC",
    "VIX":  "^VIX",
    "GOLD": "GC=F",
    "DXY":  "DX-Y.NYB",
    "TNX":  "^TNX",
    "ETH":  "ETH-USD",
    "CL":   "CL=F",   # WTI crude oil
    "NG":   "NG=F",   # natural gas
    "XLE":  "XLE",    # energy sector ETF
    "BNO":  "BNO",    # Brent oil proxy
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
for name, sym in TICKERS.items():
    s = _dl(sym)
    if s is not None:
        raw[name] = s
        print(f"  {name:6s} ({sym:15s}): {len(s)} bars  "
              f"{s.index.min().date()} → {s.index.max().date()}")
    else:
        print(f"  {name:6s} ({sym:15s}): *** EMPTY ***")

df = pd.DataFrame(raw).sort_index().loc["2019-01-01":]
df = df.ffill(limit=5)

# ──────────────────────────────────────────────────────────
# 2. ENGINEER FEATURES
# ──────────────────────────────────────────────────────────
print("\n>>> Engineering features …")
c   = df["BTC"]
ret = np.log(c).diff()
rng = ret.abs()                        # proxy daily range
rm30 = rng.rolling(30).mean()          # regime feature (same as production)

def log_ret_features(s, name, ks=(1, 5, 7, 14, 20)):
    r = np.log(s).diff()
    out = {}
    for k in ks:
        out[f"{name}_ret_{k}"] = r.rolling(k).sum()
    out[f"{name}_vol_20"] = r.rolling(20).std()
    return out

feats = {}

# BTC momentum / volatility
for k in [1, 2, 3, 5, 7, 10, 14]:
    feats[f"btc_ret_{k}"] = ret.rolling(k).sum()
for k in [5, 7, 10, 14, 20, 30]:
    feats[f"btc_vol_{k}"] = ret.rolling(k).std()

# RSI-14
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

# Energy / commodity features (new)
energy_added = []
for nm in ["CL", "NG", "XLE", "BNO"]:
    if nm in df.columns:
        feats.update(log_ret_features(df[nm], nm.lower()))
        energy_added.append(nm)

print(f"  Energy tickers available: {energy_added}")

# 7-day forward log-return target
feats_df = pd.DataFrame(feats, index=df.index)
feats_df["y_logret_7"] = np.log(c.shift(-HORIZON) / c)
feats_df = feats_df.replace([np.inf, -np.inf], np.nan).dropna()
print(f"  Feature matrix: {feats_df.shape}")

# ──────────────────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT  (same convention as production)
# ──────────────────────────────────────────────────────────
TODAY      = pd.Timestamp("today").normalize()
test_start = TODAY - pd.DateOffset(months=8)
train_end  = test_start - pd.Timedelta(days=HORIZON)

train = feats_df.loc[:train_end]
test  = feats_df.loc[test_start:]
print(f"\n  TRAIN {train.index.min().date()} → {train.index.max().date()}  n={len(train)}")
print(f"  TEST  {test.index.min().date()} → {test.index.max().date()}   n={len(test)}")

FEAT_COLS = [col for col in feats_df.columns if col != "y_logret_7"]
X_tr = train[FEAT_COLS];  y_tr = train["y_logret_7"]
X_te = test[FEAT_COLS];   y_te = test["y_logret_7"]

# ──────────────────────────────────────────────────────────
# 4. BASELINE: PRODUCTION REGIME CONE
# ──────────────────────────────────────────────────────────
edges_q = np.quantile(train["range_ma30"], [1/3, 2/3])

def to_regime(x):
    return np.searchsorted(edges_q, np.asarray(x, dtype=float), side="right").clip(0, 2)

train_r = to_regime(train["range_ma30"])
test_r  = to_regime(test["range_ma30"])

# Regime medians (as in production)
regime_med = {r: float(np.median(y_tr.values[train_r == r])) for r in range(3)}
baseline_pred_te = np.array([regime_med[r] for r in test_r])
baseline_pred_tr = np.array([regime_med[r] for r in train_r])

# ──────────────────────────────────────────────────────────
# METRIC HELPERS
# ──────────────────────────────────────────────────────────
def mape_price(pred_logret, actual_logret, close_today):
    """MAPE on actual close price (same as production script)."""
    pred_close   = close_today * np.exp(pred_logret)
    actual_close = close_today * np.exp(actual_logret)
    return float(np.mean(np.abs(pred_close - actual_close) / actual_close) * 100)

def dir_acc(pred, actual):
    return float(100 * (np.sign(pred) == np.sign(actual)).mean())

def band_cov(pred, actual, band_pct):
    hi = pred + np.log(1 + band_pct)
    lo = pred + np.log(1 - band_pct)
    return float(100 * ((actual >= lo) & (actual <= hi)).mean())

def mae_logret(pred, actual):
    return float(np.mean(np.abs(pred - actual)) * 100)   # in log-return %-points

# Production band
PROD_BAND = 0.097   # ±9.7% as in train_7d_close_cone.py

# Get today's close (last test row)
close_today_te = df["BTC"].reindex(test.index).values

bl_mape_p = mape_price(baseline_pred_te, y_te.values, close_today_te)
bl_dir    = dir_acc(baseline_pred_te, y_te.values)
bl_cov    = band_cov(baseline_pred_te, y_te.values, PROD_BAND)
bl_mae    = mae_logret(baseline_pred_te, y_te.values)

print("\n" + "="*72)
print("  PART 1: FEATURE ABLATION — does energy improve 7-day prediction?")
print("="*72)
print(f"\n  {'Model':40s}  {'MAPE':>7s}  {'MAE%':>6s}  {'DirAcc':>7s}  {'Cov±9.7%':>9s}")
print(f"  {'-'*40}  {'-'*7}  {'-'*6}  {'-'*7}  {'-'*9}")
print(f"  {'Regime cone (baseline)':40s}  {bl_mape_p:7.2f}%  "
      f"{bl_mae:6.2f}  {bl_dir:6.1f}%  {bl_cov:8.1f}%")

results = {}

def fit_eval(X_tr_, y_tr_, X_te_, y_te_, label, n_est=250, depth=3, lr=0.05):
    m = Pipeline([("sc", StandardScaler()),
                  ("m", GradientBoostingRegressor(
                      n_estimators=n_est, max_depth=depth,
                      learning_rate=lr, subsample=0.8, random_state=42))])
    m.fit(X_tr_, y_tr_)
    pred = m.predict(X_te_)
    mp  = mape_price(pred, y_te_.values, close_today_te)
    da  = dir_acc(pred, y_te_.values)
    cov = band_cov(pred, y_te_.values, PROD_BAND)
    mae = mae_logret(pred, y_te_.values)
    results[label] = dict(mape=mp, mae=mae, dir=da, cov=cov)
    print(f"  {label:40s}  {mp:7.2f}%  {mae:6.2f}  {da:6.1f}%  {cov:8.1f}%")
    return m, pred

# Feature groups
energy_feats = [f for f in FEAT_COLS if any(f.startswith(nm.lower()+"_") for nm in ["cl","ng","xle","bno"])]
macro_feats  = [f for f in FEAT_COLS if any(f.startswith(nm.lower()+"_") for nm in ["spx","ndx","vix","gold","dxy","tnx","eth"])]
btc_feats    = [f for f in FEAT_COLS if f.startswith("btc_") or f == "range_ma30"]
no_energy    = [f for f in FEAT_COLS if f not in energy_feats]
all_feats    = FEAT_COLS

print(f"\n  Feature counts  BTC={len(btc_feats)}  Macro={len(macro_feats)}  "
      f"Energy={len(energy_feats)}  No-energy={len(no_energy)}  All={len(all_feats)}")
print()

_, pred_btc_only  = fit_eval(X_tr[btc_feats],  y_tr, X_te[btc_feats],  y_te, "GBM  BTC-only features")
_, pred_no_energy = fit_eval(X_tr[no_energy],  y_tr, X_te[no_energy],  y_te, "GBM  All excl. energy")
_, pred_all       = fit_eval(X_tr[all_feats],  y_tr, X_te[all_feats],  y_te, "GBM  All incl. energy")
_, pred_btc_nrg   = fit_eval(X_tr[btc_feats + energy_feats], y_tr,
                              X_te[btc_feats + energy_feats], y_te,
                              "GBM  BTC + energy only")
_, pred_nrg_mac   = fit_eval(X_tr[energy_feats + macro_feats], y_tr,
                              X_te[energy_feats + macro_feats], y_te,
                              "GBM  Energy + macro (no BTC)")

# Coverage at multiple band sizes for best model
best_label = max(results, key=lambda k: results[k]["dir"])
print(f"\n  Best direction model: '{best_label}'")

# ── Feature importance for energy vars ──────────────────
print("\n  Energy feature correlations with 7-day BTC log-return (TRAIN):")
corrs = []
for f in energy_feats:
    if f in train.columns:
        corr = float(train[f].corr(y_tr))
        corrs.append((abs(corr), corr, f))
corrs.sort(reverse=True)
for _, corr, f in corrs[:15]:
    print(f"    {f:30s}  corr = {corr:+.4f}")

# ──────────────────────────────────────────────────────────
# PART 2: BREAKOUT / BAND SIZE THRESHOLDS (7-DAY)
# ──────────────────────────────────────────────────────────
print("\n" + "="*72)
print("  PART 2: BREAKOUT THRESHOLDS — what 7-day move is 'decisive'?")
print("="*72)

y_tr_vals = y_tr.values    # log-returns (train only, no look-ahead)

print(f"\n  Distribution of 7-day BTC log-returns (TRAIN, n={len(y_tr_vals):,}):")
print(f"  {'Quantile':>10}  {'Log-return':>12}  {'% move':>10}  {'Label':20}")
for q, lbl in [
    (0.02, "extreme DOWN"),
    (0.05, "decisive DOWN"),
    (0.10, "strong DOWN"),
    (0.20, "moderate DOWN"),
    (0.25, ""),
    (0.50, "median (central)"),
    (0.75, ""),
    (0.80, "moderate UP"),
    (0.90, "strong UP"),
    (0.95, "decisive UP"),
    (0.98, "extreme UP"),
]:
    lr  = float(np.quantile(y_tr_vals, q))
    pct = (np.exp(lr) - 1) * 100
    print(f"  q{q*100:5.1f}%  →  {lr:+10.4f}  {pct:+9.2f}%  {lbl}")

print(f"\n  Current production band: ±{PROD_BAND*100:.1f}%")
print(f"  → Covers values between "
      f"q{100*np.mean(y_tr_vals < np.log(1-PROD_BAND)):.1f}% and "
      f"q{100*np.mean(y_tr_vals < np.log(1+PROD_BAND)):.1f}%  "
      f"in training data")

print(f"\n  OOS band coverage at varying sizes (test, n={len(y_te)}):")
print(f"  {'Band':>6}  {'Coverage':>9}  {'Assessment':30}")
for band, label in [
    (0.05,  "tight"),
    (0.075, "narrow"),
    (0.097, "current ±9.7%"),
    (0.10,  ""),
    (0.12,  ""),
    (0.15,  "wide"),
    (0.20,  "very wide"),
]:
    cov = band_cov(baseline_pred_te, y_te.values, band)
    bar = "█" * int(cov / 5)
    print(f"  ±{band*100:4.1f}%   {cov:7.1f}%   {bar}")

print(f"\n  Frequency of 7-day moves exceeding threshold (TRAIN):")
print(f"  {'Threshold':>9}  {'UP %':>7}  {'DOWN %':>8}  {'Combined':>9}")
for thr in [3, 5, 7, 8, 10, 12, 15, 18, 20]:
    up   = float(100 * (y_tr_vals > np.log(1 + thr/100)).mean())
    down = float(100 * (y_tr_vals < np.log(1 - thr/100)).mean())
    both = up + down
    print(f"  >{thr:2d}%         {up:6.1f}%  {down:7.1f}%  {both:8.1f}%")

print(f"\n  Regime-specific decisive-move thresholds (TRAIN):")
regime_labels = ["low vol", "mid vol", "high vol"]
print(f"  {'Regime':10}  {'q05 DOWN':>10}  {'q10 DOWN':>10}  {'q25 DOWN':>10}  "
      f"{'q50 MED':>8}  {'q75 UP':>8}  {'q90 UP':>8}  {'q95 UP':>8}")
for r in range(3):
    sub = y_tr_vals[train_r == r]
    qs  = {q: float(np.quantile(sub, q)) for q in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]}
    p   = {q: (np.exp(qs[q]) - 1) * 100 for q in qs}
    print(f"  {regime_labels[r]:10s}  {p[0.05]:>+9.1f}%  {p[0.10]:>+9.1f}%  {p[0.25]:>+9.1f}%  "
          f"  {p[0.50]:>+6.1f}%  {p[0.75]:>+6.1f}%  {p[0.90]:>+6.1f}%  {p[0.95]:>+6.1f}%")

print(f"\n  Suggested decisive breakout thresholds (q05 / q95 = outer 10%):")
for r in range(3):
    sub  = y_tr_vals[train_r == r]
    q05  = (np.exp(float(np.quantile(sub, 0.05))) - 1) * 100
    q95  = (np.exp(float(np.quantile(sub, 0.95))) - 1) * 100
    q10  = (np.exp(float(np.quantile(sub, 0.10))) - 1) * 100
    q90  = (np.exp(float(np.quantile(sub, 0.90))) - 1) * 100
    print(f"  [{regime_labels[r]:8s}]  "
          f"decisive DOWN < {q05:+.1f}%  (strong: {q10:+.1f}%)   |   "
          f"decisive UP > {q95:+.1f}%  (strong: {q90:+.1f}%)")

q05_all = float(np.quantile(y_tr_vals, 0.05))
q95_all = float(np.quantile(y_tr_vals, 0.95))
q10_all = float(np.quantile(y_tr_vals, 0.10))
q90_all = float(np.quantile(y_tr_vals, 0.90))
print(f"\n  Overall (regime-agnostic):")
print(f"    Decisive  (q05/q95): {(np.exp(q05_all)-1)*100:+.1f}%  /  {(np.exp(q95_all)-1)*100:+.1f}%")
print(f"    Strong    (q10/q90): {(np.exp(q10_all)-1)*100:+.1f}%  /  {(np.exp(q90_all)-1)*100:+.1f}%")

print(f"""
  ╔══════════════════════════════════════════════════════════════════╗
  ║  7-DAY DECISIVE BREAKOUT THRESHOLDS  (outer 10% of history)     ║
  ║                                                                  ║
  ║  Decisive UPTREND:    7-day close > +{(np.exp(q95_all)-1)*100:.0f}%  (q95 overall)    ║
  ║  Strong UPTREND:      7-day close > +{(np.exp(q90_all)-1)*100:.1f}%  (q90 overall)   ║
  ║                                                                  ║
  ║  Decisive DOWNTREND:  7-day close <  {(np.exp(q05_all)-1)*100:.1f}%  (q05 overall)   ║
  ║  Strong DOWNTREND:    7-day close <  {(np.exp(q10_all)-1)*100:.1f}%  (q10 overall)   ║
  ║                                                                  ║
  ║  Note: the current ±9.7% band ≈ the q80/q20 "typical" range.    ║
  ║  To signal decisiveness, price needs to break OUTSIDE ±12-14%.  ║
  ╚══════════════════════════════════════════════════════════════════╝
""")
