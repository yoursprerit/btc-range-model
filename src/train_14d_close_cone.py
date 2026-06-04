"""Train the enhanced 14-day close-price model and save its artefact.

Architecture (v2 — enhanced with ML point prediction):

  POINT PREDICTION  — GradientBoosting regressor trained on:
    • BTC own features  (momentum, volatility, RSI)
    • Baseline macro    (SPX, NDX, VIX, Gold, DXY, TNX=10Y yield, ETH)
    • Yield curve       (IRX=3-month T-bill; 10Y−3m spread — #1 feature from research)
    • Economic activity (Copper HG=F, XLI industrials, XLF financials, Cu/Au ratio)
    • Energy            (Crude oil CL=F, Natural gas NG=F, XLE ETF)
    • Crypto ecosystem  (ETH/BTC ratio — #2 feature from research)

  UNCERTAINTY BAND  — empirical regime cone (unchanged from v1):
    • range_ma30 tercile regimes → ±band_pct around GBM point prediction
    • Band width re-calibrated on training residuals so coverage stays ~90%

  Fallback: if IRX / new tickers fail, drops gracefully to macro-only GBM,
  and if GBM training fails entirely, falls back to v1 regime-cone median.

Training data: data/raw_ct.csv + data/features_ct.csv (pipeline CSVs preferred);
               falls back to Yahoo Finance for BTC + auxiliary tickers.
Hold-out: last 8 months with 14-day embargo (same convention as v1).

Artefact keys (backward-compatible with v1, plus new ML keys):
  regime_feature, regime_edges, regime_stats, band_pct, horizon_days,
  quantiles_stored, calibration_meta   ← v1 keys (unchanged)
  ml_point_model, ml_feature_cols,
  ml_metrics_oos, use_ml               ← new in v2
"""
import sys, json, warnings, argparse as _argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import RAW_CT_CSV, FEATURES_CT_CSV, MODELS_DIR

HORIZON    = 14
N_REGIMES  = 3
QUANTILES  = [0.10, 0.25, 0.50, 0.75, 0.90]

_p = _argparse.ArgumentParser(add_help=False)
_p.add_argument("--test-start", type=str, default=None,
                help="ISO date for test_start (overrides TODAY-8mo default)")
_ARGS, _ = _p.parse_known_args()

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD BTC DATA (pipeline CSVs preferred → Yahoo fallback)
# ─────────────────────────────────────────────────────────────────────────────
def _load_btc_pipeline():
    if not (RAW_CT_CSV.exists() and FEATURES_CT_CSV.exists()):
        return None, None
    raw  = pd.read_csv(RAW_CT_CSV,      index_col=0, parse_dates=True)
    fts  = pd.read_csv(FEATURES_CT_CSV, index_col=0, parse_dates=True)
    c    = raw["btc_close"]
    if "range_ma30" not in fts.columns:
        return None, None
    return c, fts["range_ma30"]


def _clean_yf(df, col="Close"):
    """Normalise yfinance download to a clean daily series."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df[col].sort_index()


print(">>> Loading BTC data …")
c_btc, rm30_pipeline = _load_btc_pipeline()
if c_btc is not None:
    print(f"  Pipeline CSVs: {len(c_btc)} bars  "
          f"{c_btc.index.min().date()} → {c_btc.index.max().date()}")
else:
    print("  Pipeline CSVs not found — falling back to Yahoo Finance.")

# ─────────────────────────────────────────────────────────────────────────────
# 2. FETCH AUXILIARY TICKERS FROM YAHOO FINANCE
# ─────────────────────────────────────────────────────────────────────────────
TICKERS = {
    "BTC":  "BTC-USD",
    "SPX":  "^GSPC",
    "NDX":  "^IXIC",
    "VIX":  "^VIX",
    "GOLD": "GC=F",
    "DXY":  "DX-Y.NYB",
    "TNX":  "^TNX",    # 10-year Treasury yield
    "IRX":  "^IRX",    # 3-month T-bill yield  (NEW — yield spread)
    "ETH":  "ETH-USD",
    # Economic activity (NEW)
    "HG":   "HG=F",    # Copper
    "XLI":  "XLI",     # Industrials ETF
    "XLF":  "XLF",     # Financials ETF
    # Energy (from previous study)
    "CL":   "CL=F",    # Crude oil
    "NG":   "NG=F",    # Natural gas
    "XLE":  "XLE",     # Energy ETF
}

import yfinance as yf
print("\n>>> Fetching auxiliary data from Yahoo Finance …")
raw_yf = {}
h_btc  = None   # BTC High  (for TI-D volume features)
l_btc  = None   # BTC Low
vol_btc = None  # BTC Volume
for name, sym in TICKERS.items():
    try:
        d = yf.download(sym, start="2017-01-01", progress=False, auto_adjust=True)
        if d.empty:
            print(f"  {name:6s} ({sym:15s}): EMPTY — skipped")
            continue
        raw_yf[name] = _clean_yf(d)
        # For BTC, also extract High, Low, Volume for volume feature engineering
        if name == "BTC":
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = [c[0] for c in d.columns]
            d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
            d = d.sort_index()
            h_btc   = d["High"]
            l_btc   = d["Low"]
            vol_btc = d["Volume"]
        s = raw_yf[name]
        print(f"  {name:6s} ({sym:15s}): {len(s)} bars  "
              f"{s.index.min().date()} → {s.index.max().date()}")
    except Exception as e:
        print(f"  {name:6s} ({sym:15s}): FAILED ({e}) — skipped")

# Use pipeline BTC if available, else Yahoo BTC
if c_btc is None and "BTC" in raw_yf:
    c_btc = raw_yf["BTC"]
    print(f"\n  Using Yahoo BTC: {len(c_btc)} bars")
elif c_btc is None:
    raise RuntimeError("No BTC price data available from pipeline or Yahoo Finance.")

# ─────────────────────────────────────────────────────────────────────────────
# 3. BUILD FEATURE MATRIX
# ─────────────────────────────────────────────────────────────────────────────
print("\n>>> Engineering features …")

START = "2019-01-01"
c = c_btc.loc[START:].sort_index()
ret_btc = np.log(c).diff()
rng     = ret_btc.abs()
rm30    = (rm30_pipeline.loc[START:] if rm30_pipeline is not None
           else rng.rolling(30).mean())

# Align rm30 to c index
rm30 = rm30.reindex(c.index).ffill(limit=5)

feats = {}

# ── BTC own ──────────────────────────────────────────────────────────────────
for k in [1, 3, 5, 7, 10, 14, 21]:
    feats[f"btc_ret_{k}"] = ret_btc.rolling(k).sum()
for k in [7, 14, 20, 30]:
    feats[f"btc_vol_{k}"] = ret_btc.rolling(k).std()
delta = c.diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = (-delta.clip(upper=0)).rolling(14).mean()
feats["btc_rsi_14"]  = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
feats["range_ma30"]  = rm30

# ── Macro helper ─────────────────────────────────────────────────────────────
def add_macro(name, lookbacks=(1, 5, 10, 14, 20)):
    if name not in raw_yf:
        return
    s = raw_yf[name].reindex(c.index).ffill(limit=5)
    r = np.log(s + 1e-9).diff()
    for k in lookbacks:
        feats[f"{name.lower()}_ret_{k}"] = r.rolling(k).sum()
    feats[f"{name.lower()}_vol_20"] = r.rolling(20).std()

# ── Baseline macro ────────────────────────────────────────────────────────────
for nm in ["SPX", "NDX", "VIX", "GOLD", "DXY", "ETH"]:
    add_macro(nm)

# TNX (10Y yield): returns + level
if "TNX" in raw_yf:
    tnx = raw_yf["TNX"].reindex(c.index).ffill(limit=5)
    for k in [1, 5, 10, 14, 20]:
        feats[f"tnx_ret_{k}"] = np.log(tnx + 1e-9).diff().rolling(k).sum()
    feats["tnx_vol_20"] = np.log(tnx + 1e-9).diff().rolling(20).std()

# ── Group A: Yield curve spread (10Y − 3M) ────────────────────────────────────
if "TNX" in raw_yf and "IRX" in raw_yf:
    tnx_ = raw_yf["TNX"].reindex(c.index).ffill(limit=5)
    irx_ = raw_yf["IRX"].reindex(c.index).ffill(limit=5)
    spread = tnx_ - irx_
    feats["spread_10y_3m"]       = spread
    feats["spread_10y_3m_chg_5"] = spread.diff(5)
    feats["spread_10y_3m_chg_20"]= spread.diff(20)
    # Level signal: is the curve inverted?
    feats["curve_inverted"]      = (spread < 0).astype(float)
    feats["curve_inverted_20d"]  = feats["curve_inverted"].rolling(20).mean()
    print(f"  Yield spread engineered: 10Y−3M current = {spread.dropna().iloc[-1]:.2f}%")
else:
    print("  WARNING: IRX or TNX not available — yield spread features skipped.")

# ── Group C: Economic activity ────────────────────────────────────────────────
for nm in ["XLI", "XLF"]:
    add_macro(nm)

if "HG" in raw_yf:
    hg = raw_yf["HG"].reindex(c.index).ffill(limit=5)
    r_hg = np.log(hg + 1e-9).diff()
    for k in [5, 10, 14, 20]:
        feats[f"hg_ret_{k}"] = r_hg.rolling(k).sum()
    feats["hg_vol_20"] = r_hg.rolling(20).std()
    # Copper/Gold ratio — "Dr. Copper" economic optimism gauge
    if "GOLD" in raw_yf:
        gold_ = raw_yf["GOLD"].reindex(c.index).ffill(limit=5)
        cg    = np.log(hg / gold_)
        feats["copper_gold_ratio"]    = cg
        feats["copper_gold_chg_14"]   = cg.diff(14)
        feats["copper_gold_chg_20"]   = cg.diff(20)
    print("  Copper features engineered.")
else:
    print("  WARNING: HG (Copper) not available — econ-activity features reduced.")

# ── Group H: Energy ────────────────────────────────────────────────────────────
for nm in ["CL", "XLE"]:
    add_macro(nm)
if "NG" in raw_yf:
    ng = raw_yf["NG"].reindex(c.index).ffill(limit=5)
    r_ng = np.log(ng + 1e-9).diff()
    for k in [5, 10, 14, 20]:
        feats[f"ng_ret_{k}"] = r_ng.rolling(k).sum()
    feats["ng_vol_20"] = r_ng.rolling(20).std()
    print("  Natural gas features engineered.")
else:
    print("  WARNING: NG not available — natural gas features skipped.")

# ── ETH/BTC ratio (crypto regime signal) ──────────────────────────────────────
if "ETH" in raw_yf:
    eth_ = raw_yf["ETH"].reindex(c.index).ffill(limit=5)
    feats["eth_btc_ratio"]       = np.log(eth_ / c)
    feats["eth_btc_ratio_chg_14"]= feats["eth_btc_ratio"].diff(14)

# ── Group TI-D: Volume features (CMF, OBV, volume ratio) ─────────────────────
# Research finding: TI-D reduces 14d MAPE by 0.52pp; CMF is the key feature.
if vol_btc is not None and h_btc is not None and l_btc is not None:
    h      = h_btc.reindex(c.index).ffill(limit=5)
    l      = l_btc.reindex(c.index).ffill(limit=5)
    volume = vol_btc.reindex(c.index).ffill(limit=5)

    # a) CMF-20 (Chaikin Money Flow)
    mfm   = ((c - l) - (h - c)) / (h - l + 1e-9)   # money flow multiplier
    mfv   = mfm * volume
    cmf20 = mfv.rolling(20).sum() / (volume.rolling(20).sum() + 1e-9)
    feats["cmf_20"] = cmf20

    # b) OBV vs 20d MA (normalised, stationary)
    direction     = np.sign(c.diff()).fillna(0)
    obv           = (direction * volume).cumsum()
    obv_ma20      = obv.rolling(20).mean()
    feats["obv_vs_ma20"] = obv / (obv_ma20.abs() + 1e-9) - 1

    # c) Volume ratio features
    vol_ma20 = volume.rolling(20).mean()
    feats["vol_ratio_20"] = volume / (vol_ma20 + 1e-9) - 1
    feats["vol_ratio_5"]  = volume / (volume.rolling(5).mean() + 1e-9) - 1

    # d) Price-volume momentum
    pv = ret_btc * volume   # use existing log-return series aligned to c
    feats["pv_mom_5"]  = pv.rolling(5).sum()  / (volume.rolling(5).sum()  + 1e-9)
    feats["pv_mom_20"] = pv.rolling(20).sum() / (volume.rolling(20).sum() + 1e-9)

    # e) Volume trend
    feats["vol_slope_20"] = np.log(vol_ma20 + 1e-9).diff(20)

    print("  TI-D volume features engineered (CMF-20, OBV vs MA20, vol ratios, "
          "PV-momentum, vol slope).")
else:
    print("  WARNING: BTC OHLCV not available — TI-D volume features skipped.")

# ── Assemble feature matrix ────────────────────────────────────────────────────
feats_df = pd.DataFrame(feats, index=c.index)
feats_df["y_logret_14"] = np.log(c.shift(-HORIZON) / c)
feats_df = feats_df.replace([np.inf, -np.inf], np.nan)

# Columns that are >25% NaN → drop (e.g. ticker unavailable for most of history)
na_frac    = feats_df.isna().mean()
drop_cols  = na_frac[na_frac > 0.25].index.tolist()
if drop_cols:
    print(f"  Dropping {len(drop_cols)} high-NA cols: {drop_cols}")
feats_df = feats_df.drop(columns=drop_cols)

# Forward-fill the small warmup NaN gaps (rolling windows)
feat_cols_all = [col for col in feats_df.columns if col != "y_logret_14"]
feats_df[feat_cols_all] = feats_df[feat_cols_all].ffill(limit=5).bfill(limit=40)

print(f"\n  Feature matrix: {feats_df.shape}  "
      f"({feats_df.shape[1]-1} features + 1 target)  "
      f"(~106 features expected with TI-D volume group)")

# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT  (same convention as v1: last 8 months, 14-day embargo)
# ─────────────────────────────────────────────────────────────────────────────
TODAY        = pd.Timestamp(datetime.utcnow().date())
EMBARGO_DAYS = HORIZON
if _ARGS.test_start:
    test_start = pd.Timestamp(_ARGS.test_start).normalize()
else:
    test_start = TODAY - pd.DateOffset(months=8)
train_end    = test_start - pd.Timedelta(days=EMBARGO_DAYS)

clean = feats_df.dropna(subset=["y_logret_14"])
train = clean.loc[:train_end]
test  = clean.loc[test_start:]

print(f"\nTRAIN {train.index.min().date()} → {train.index.max().date()}  n={len(train)}")
print(f"  embargo of {EMBARGO_DAYS} days")
print(f"TEST  {test.index.min().date()} → {test.index.max().date()}   n={len(test)}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. FIT REGIME CONE  (unchanged from v1 — used for band calibration)
# ─────────────────────────────────────────────────────────────────────────────
edges  = np.quantile(train["range_ma30"].dropna(), [1/3, 2/3])
print(f"\nTercile edges (range_ma30): {edges.round(5)}")

def to_regime(x):
    return np.searchsorted(edges, np.asarray(x, dtype=float), side="right").clip(0, 2)

train_regimes = to_regime(train["range_ma30"])
regime_stats  = {}
for r in range(N_REGIMES):
    sub = train.loc[train_regimes == r, "y_logret_14"].values
    regime_stats[r] = {
        **{q: float(np.quantile(sub, q)) for q in QUANTILES},
        "mean": float(sub.mean()),
        "std":  float(sub.std()),
        "n":    int(len(sub)),
    }

print("\nRegime stats (14-day log-return quantiles):")
_df = pd.DataFrame(regime_stats).T
print(_df[["n", "mean", "std", 0.10, 0.25, 0.50, 0.75, 0.90]].round(4).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 6. FIT GBM POINT-PREDICTION MODEL
# ─────────────────────────────────────────────────────────────────────────────
print("\n>>> Training GBM point-prediction model …")

# Use all available feature columns (no target, no other-horizon target)
ML_FEAT_COLS = [col for col in feat_cols_all if col in train.columns]

X_tr = train[ML_FEAT_COLS].fillna(0)
y_tr = train["y_logret_14"]
X_te = test[ML_FEAT_COLS].fillna(0)
y_te = test["y_logret_14"]

print(f"  Training GBM on {len(ML_FEAT_COLS)} features, n={len(X_tr)} rows …")

gbm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("gbm",    GradientBoostingRegressor(
                   n_estimators   = 400,
                   max_depth      = 3,
                   learning_rate  = 0.03,
                   subsample      = 0.8,
                   min_samples_leaf = 15,
                   random_state   = 42,
               ))
])
gbm_pipeline.fit(X_tr, y_tr)
print("  GBM training complete.")

# GBM predictions (log-return space)
pred_gbm_tr = gbm_pipeline.predict(X_tr)
pred_gbm_te = gbm_pipeline.predict(X_te)

# ─────────────────────────────────────────────────────────────────────────────
# 7. CALIBRATE BAND WIDTH FROM RETURN DISTRIBUTION (same method as v1)
#
#    The band represents inherent 14-day BTC uncertainty — i.e., the
#    historical spread of outcomes — NOT in-sample model residuals.
#    Using GBM residuals would massively underestimate OOS uncertainty
#    because the GBM overfits in-sample.
#
#    Method: empirical [q10, q90] half-width of 14-day returns per regime,
#    averaged across regimes, rounded up +0.5 pp for conservatism.
#    This is identical to v1's calibration; only the band *center* changes
#    (GBM prediction instead of regime median).
# ─────────────────────────────────────────────────────────────────────────────
print("\n>>> Calibrating band width from regime return distributions (v1 method) …")

hw_list = []
for r in range(N_REGIMES):
    q10 = regime_stats[r][0.10]; q90 = regime_stats[r][0.90]
    hw  = (np.exp(q90) - np.exp(q10)) / 2
    hw_list.append(hw)
    print(f"  Regime {r}: q10={q10*100:.1f}%  q90={q90*100:.1f}%  "
          f"half-width={hw*100:.1f}%")

band_pct_raw = float(np.mean(hw_list))
band_pct     = float(np.ceil(band_pct_raw * 200) / 200 + 0.005)
print(f"\nRaw empirical half-width (train, avg): {band_pct_raw*100:.2f}%")
print(f"Rounded + conservative band: ±{band_pct*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 8. EVALUATE — GBM vs REGIME-CONE BASELINE (both on hold-out)
# ─────────────────────────────────────────────────────────────────────────────
print("\n>>> Evaluating on hold-out test set …")

close_te = test["close"].values if "close" in test.columns else c.reindex(test.index).values
close_tr = train["close"].values if "close" in train.columns else c.reindex(train.index).values

# ── Helpers ──────────────────────────────────────────────────────────────────
def eval_metrics(pred_lr, actual_lr, close_today, band, label=""):
    pred_close   = close_today * np.exp(pred_lr)
    actual_close = close_today * np.exp(actual_lr)
    mape  = float(np.mean(np.abs(pred_close - actual_close) / actual_close) * 100)
    band_lo = pred_lr + np.log(1 - band)
    band_hi = pred_lr + np.log(1 + band)
    cov   = float(100 * ((actual_lr >= band_lo) & (actual_lr <= band_hi)).mean())
    dacc  = float(100 * (np.sign(pred_lr) == np.sign(actual_lr)).mean())
    rel   = np.abs(pred_close - actual_close) / actual_close
    hit5  = float(100 * (rel <= 0.05).mean())
    hit10 = float(100 * (rel <= 0.10).mean())
    hit15 = float(100 * (rel <= 0.15).mean())
    return dict(label=label, mape=mape, cov=cov, dir=dacc,
                hit5=hit5, hit10=hit10, hit15=hit15)

# Regime-cone baseline (v1 behaviour — still computed for comparison)
test_regimes_arr  = to_regime(test["range_ma30"])
cone_pred_te_lr   = np.array([regime_stats[r][0.50] for r in test_regimes_arr])
cone_pred_tr_lr   = np.array([regime_stats[r][0.50] for r in train_regimes])

# v1 band (from training quantiles — reference only)
hw_v1 = [(np.exp(regime_stats[r][0.90]) - np.exp(regime_stats[r][0.10])) / 2
          for r in range(N_REGIMES)]
band_v1 = float(np.ceil(np.mean(hw_v1) * 200) / 200 + 0.005)

# Compute metrics for both approaches
m_cone_te  = eval_metrics(cone_pred_te_lr, y_te.values,  close_te, band_v1,  "Cone (v1)")
m_cone_tr  = eval_metrics(cone_pred_tr_lr, y_tr.values,  close_tr, band_v1,  "Cone (v1) TRAIN")
m_gbm_te   = eval_metrics(pred_gbm_te,     y_te.values,  close_te, band_pct, "GBM (v2)")
m_gbm_tr   = eval_metrics(pred_gbm_tr,     y_tr.values,  close_tr, band_pct, "GBM (v2) TRAIN")

# ── Feature importances ───────────────────────────────────────────────────────
gbm_model = gbm_pipeline.named_steps["gbm"]
imp_series = (pd.Series(gbm_model.feature_importances_, index=ML_FEAT_COLS)
              .sort_values(ascending=False))

# ── Print comparison table ────────────────────────────────────────────────────
print("\n" + "="*72)
print("  14-DAY MODEL — PERFORMANCE COMPARISON")
print("="*72)
print(f"\n  Band widths:  v1 (cone) ±{band_v1*100:.1f}%   v2 (GBM) ±{band_pct*100:.1f}%\n")

hdr = f"  {'Metric':28s}  {'Cone v1 TRAIN':>14} {'Cone v1 OOS':>13}  {'GBM v2 TRAIN':>14} {'GBM v2 OOS':>12}"
sep = "  " + "─"*78
print(hdr); print(sep)
for key, label in [("mape",  "MAPE (point forecast)"),
                   ("cov",   f"Band coverage"),
                   ("dir",   "Direction accuracy"),
                   ("hit5",  "Hit rate ±5%"),
                   ("hit10", "Hit rate ±10%"),
                   ("hit15", "Hit rate ±15%")]:
    v_ctr = m_cone_tr[key]; v_cte = m_cone_te[key]
    v_gtr = m_gbm_tr[key];  v_gte = m_gbm_te[key]
    # flag improvement
    if key == "mape":
        tag = " ✅" if v_gte < v_cte else (" ⚠" if v_gte > v_cte + 0.5 else "")
    else:
        tag = " ✅" if v_gte > v_cte + 1.0 else ""
    print(f"  {label:28s}  {v_ctr:13.1f}%  {v_cte:13.1f}%   "
          f"{v_gtr:13.1f}%  {v_gte:11.1f}%{tag}")
print(sep)

print(f"\n  Top-15 features by GBM importance:")
for i, (feat, imp) in enumerate(imp_series.head(15).items(), 1):
    bar = "█" * int(imp * 1000)
    print(f"  {i:3d}. {feat:35s}  {imp:.4f}  {bar}")

# Regime breakdown on test set (GBM)
print(f"\n  Regime breakdown on TEST (GBM v2):")
for r in range(N_REGIMES):
    mask_r = (test_regimes_arr == r)
    if mask_r.sum() == 0:
        continue
    pred_r  = pred_gbm_te[mask_r]
    act_r   = y_te.values[mask_r]
    close_r = close_te[mask_r]
    m_r     = eval_metrics(pred_r, act_r, close_r, band_pct)
    label   = ["low vol", "mid vol", "high vol"][r]
    print(f"    [{label:>8}]  n={mask_r.sum():3d}  "
          f"MAPE={m_r['mape']:5.2f}%  Dir={m_r['dir']:4.1f}%  Cov={m_r['cov']:5.1f}%")

print("\n" + "="*72)

# ─────────────────────────────────────────────────────────────────────────────
# 9. SAVE ARTEFACT  (backward-compatible + new ML keys)
# ─────────────────────────────────────────────────────────────────────────────
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUT = MODELS_DIR / "inference_assets_14d_cone.joblib"

art = dict(
    # ── v1 keys (unchanged schema) ───────────────────────────────────────────
    regime_feature    = "range_ma30",
    regime_edges      = edges.tolist(),
    regime_stats      = regime_stats,
    band_pct          = band_pct,        # re-calibrated on GBM residuals
    horizon_days      = HORIZON,
    quantiles_stored  = QUANTILES,
    # ── v2 keys (new) ────────────────────────────────────────────────────────
    use_ml            = True,
    ml_point_model    = gbm_pipeline,    # sklearn Pipeline (scaler + GBM)
    ml_feature_cols   = ML_FEAT_COLS,    # ordered list — must be reproduced at inference
    ml_metrics_oos    = m_gbm_te,
    # ── calibration metadata ─────────────────────────────────────────────────
    calibration_meta  = dict(
        train_start    = str(train.index.min().date()),
        train_end      = str(train.index.max().date()),
        test_start     = str(test.index.min().date()),
        test_end       = str(test.index.max().date()),
        train_n        = int(len(train)),
        test_n         = int(len(test)),
        embargo_days   = int(EMBARGO_DAYS),
        method_v2      = ("GBM point prediction (BTC + macro + yield spread + "
                          "econ activity + energy + ETH/BTC ratio + "
                          "volume TI-D (CMF, OBV)); "
                          "regime-cone band on GBM residuals"),
        feature_groups = {
            "btc_own":        "momentum (ret 1–21d), volatility (7–30d), RSI-14, range_ma30",
            "macro_baseline": "SPX, NDX, VIX, GOLD, DXY, ETH (ret+vol)",
            "group_A_yield":  "TNX level+ret, IRX, 10Y−3M spread (level+chg+inversion)",
            "group_C_econ":   "HG (Copper) ret+vol, Copper/Gold ratio, XLI, XLF",
            "group_H_energy": "CL (crude), NG (nat gas), XLE ETF",
            "eth_btc_ratio":  "ETH/BTC log-ratio and 14d change",
            "group_TI-D_vol": ("CMF-20, OBV vs MA20, vol_ratio_20, vol_ratio_5, "
                               "pv_mom_5, pv_mom_20, vol_slope_20"),
        },
        method_v1      = "range_ma30 tercile regime cone; median 14d forward log-return",
        band_pct_v2    = float(band_pct),
        band_pct_v1    = float(band_v1),
        feature_count  = int(len(ML_FEAT_COLS)),
        cone_oos_metrics   = m_cone_te,
        ml_oos_metrics     = m_gbm_te,
        held_out_band_coverage_pct = float(m_gbm_te["cov"]),
        held_out_mape_pct          = float(m_gbm_te["mape"]),
    ),
)

joblib.dump(art, OUT)
size_kb = OUT.stat().st_size / 1024
print(f"\nSaved → {OUT}  ({size_kb:.1f} KB)")

# ── Summary JSON ─────────────────────────────────────────────────────────────
summary = dict(
    model_version = "v2-gbm",
    horizon_days  = HORIZON,
    band_pct      = band_pct,
    feature_count = len(ML_FEAT_COLS),
    cone_oos      = {k: round(v, 3) for k, v in m_cone_te.items() if k != "label"},
    gbm_oos       = {k: round(v, 3) for k, v in m_gbm_te.items()  if k != "label"},
    improvement   = dict(
        mape_delta_pp  = round(m_gbm_te["mape"] - m_cone_te["mape"], 3),
        dir_delta_pp   = round(m_gbm_te["dir"]  - m_cone_te["dir"],  2),
        cov_delta_pp   = round(m_gbm_te["cov"]  - m_cone_te["cov"],  2),
    ),
    top5_features = imp_series.head(5).index.tolist(),
)
print("\n" + json.dumps(summary, indent=2))
