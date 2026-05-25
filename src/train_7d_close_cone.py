"""Train the enhanced 7-day close-price model and save its artefact.

Architecture (v2 — upgraded from pure regime-cone to GBM + cone hybrid):

  POINT PREDICTION  — GradientBoosting regressor trained on:
    • BTC own features  (momentum, volatility, RSI)
    • Baseline macro    (SPX, NDX, VIX, Gold, DXY, TNX=10Y yield, ETH)
    • Yield curve       (IRX=3-month T-bill; 10Y−3m spread)
    • Economic activity (Copper HG=F, XLI industrials, XLF financials, Cu/Au ratio)
    • Energy            (Crude oil CL=F, Natural gas NG=F, XLE ETF)
    • Crypto ecosystem  (ETH/BTC ratio)
    • TI-B Momentum     (RSI-7/21/30, Stochastic %K/%D, Williams %R, CCI, ROC — #1 new group)
    • OC-B Miner/flow   (Miners revenue, mempool, NVT proxy, miner stress — #2 new group)

  UNCERTAINTY BAND  — empirical regime cone (unchanged from v1):
    • range_ma30 tercile regimes → ±band_pct around GBM point prediction
    • Band width calibrated from empirical q10/q90 of 7-day return distribution

  Fallback: if any on-chain or ticker data fails, trains on whatever is available.

Training data: data/raw_ct.csv + data/features_ct.csv (pipeline CSVs preferred);
               falls back to Yahoo Finance for BTC + auxiliary tickers.
Hold-out: last 8 months with 7-day embargo (same convention as v1).

Artefact keys (backward-compatible with v1, plus new ML keys):
  regime_feature, regime_edges, regime_stats, band_pct, horizon_days,
  quantiles_stored, calibration_meta   ← v1 keys (unchanged)
  ml_point_model, ml_feature_cols,
  ml_metrics_oos, use_ml               ← new in v2
"""
import sys, json, warnings, time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import requests

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import RAW_CT_CSV, FEATURES_CT_CSV, MODELS_DIR

HORIZON           = 7
N_REGIMES         = 3
QUANTILES         = [0.10, 0.25, 0.50, 0.75, 0.90]
BAND_PCT_FALLBACK = 0.097   # ±9.7% if cone calibration gives an out-of-range value

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD BTC DATA (pipeline CSVs preferred → Yahoo fallback)
# ─────────────────────────────────────────────────────────────────────────────
def _load_btc_pipeline():
    if not (RAW_CT_CSV.exists() and FEATURES_CT_CSV.exists()):
        return None, None
    raw = pd.read_csv(RAW_CT_CSV,      index_col=0, parse_dates=True)
    fts = pd.read_csv(FEATURES_CT_CSV, index_col=0, parse_dates=True)
    c   = raw["btc_close"]
    if "range_ma30" not in fts.columns:
        return None, None
    return c, fts["range_ma30"]


def _clean_yf(df, col="Close"):
    """Normalise yfinance download to a clean daily series."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df[col].sort_index()


def _clean_yf_ohlcv(df):
    """Normalise yfinance download to a clean daily OHLCV DataFrame."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df.sort_index()


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
    "IRX":  "^IRX",    # 3-month T-bill yield (yield spread)
    "ETH":  "ETH-USD",
    "HG":   "HG=F",    # Copper
    "XLI":  "XLI",     # Industrials ETF
    "XLF":  "XLF",     # Financials ETF
    "CL":   "CL=F",    # Crude oil
    "NG":   "NG=F",    # Natural gas
    "XLE":  "XLE",     # Energy ETF
}

import yfinance as yf
print("\n>>> Fetching auxiliary data from Yahoo Finance …")
raw_yf = {}
btc_ohlcv = None
for name, sym in TICKERS.items():
    try:
        d = yf.download(sym, start="2017-01-01", progress=False, auto_adjust=True)
        if d.empty:
            print(f"  {name:6s} ({sym:15s}): EMPTY — skipped")
            continue
        raw_yf[name] = _clean_yf(d)
        # Save full OHLCV for BTC (needed for TI-B indicators)
        if name == "BTC":
            btc_ohlcv = _clean_yf_ohlcv(d)
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
# 3. FETCH ON-CHAIN DATA  (blockchain.com public API, no key)
# ─────────────────────────────────────────────────────────────────────────────
print("\n>>> Fetching on-chain data from blockchain.com …")

OC_METRICS = {
    "tx_fees_usd":  "transaction-fees-usd",
    "miners_rev":   "miners-revenue",
    "mempool_sz":   "mempool-size",
    "tx_vol_usd":   "estimated-transaction-volume-usd",
}

def fetch_blockchain_info(metric_slug, start="2018-01-01", retries=3):
    url = f"https://api.blockchain.info/charts/{metric_slug}"
    params = {"timespan": "all", "sampled": "false", "format": "json", "cors": "true"}
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                vals = r.json()["values"]
                df = pd.DataFrame(vals)
                df["date"] = pd.to_datetime(df["x"], unit="s").dt.normalize()
                # Sub-daily sources (mempool) have many rows per day — take last
                s = df.groupby("date")["y"].last().sort_index()
                return s.loc[start:]
            else:
                print(f"    HTTP {r.status_code} for {metric_slug}")
        except Exception as e:
            print(f"    Attempt {attempt+1} failed for {metric_slug}: {e}")
            time.sleep(2 ** attempt)
    return None

onchain_raw = {}
for name, slug in OC_METRICS.items():
    s = fetch_blockchain_info(slug)
    if s is not None:
        onchain_raw[name] = s
        print(f"  {name:16s}: {len(s)} daily bars  "
              f"{s.index.min().date()} → {s.index.max().date()}")
    else:
        print(f"  {name:16s}: FAILED (will skip)")
    time.sleep(0.3)   # be kind to the API

# ─────────────────────────────────────────────────────────────────────────────
# 4. TECHNICAL INDICATOR HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def compute_rsi(close, period):
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename(f"rsi_{period}")

def compute_stoch(high, low, close, k=14, d=3):
    lo_k  = low.rolling(k).min()
    hi_k  = high.rolling(k).max()
    pct_k = 100 * (close - lo_k) / (hi_k - lo_k + 1e-9)
    pct_d = pct_k.rolling(d).mean()
    return pct_k.rename("stoch_k"), pct_d.rename("stoch_d")

# ─────────────────────────────────────────────────────────────────────────────
# 5. BUILD FEATURE MATRIX
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

# ── GROUP BASE: BTC own ───────────────────────────────────────────────────────
for k in [1, 3, 5, 7, 10, 14, 21]:
    feats[f"btc_ret_{k}"] = ret_btc.rolling(k).sum()
for k in [7, 14, 20, 30]:
    feats[f"btc_vol_{k}"] = ret_btc.rolling(k).std()
delta = c.diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = (-delta.clip(upper=0)).rolling(14).mean()
feats["btc_rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
feats["range_ma30"] = rm30

# ── GROUP BASE: Macro helper ──────────────────────────────────────────────────
def add_macro(name, lookbacks=(1, 5, 10, 14, 20)):
    if name not in raw_yf:
        return
    s = raw_yf[name].reindex(c.index).ffill(limit=5)
    r = np.log(s + 1e-9).diff()
    for k in lookbacks:
        feats[f"{name.lower()}_ret_{k}"] = r.rolling(k).sum()
    feats[f"{name.lower()}_vol_20"] = r.rolling(20).std()

# Baseline macro
for nm in ["SPX", "NDX", "VIX", "GOLD", "DXY", "ETH"]:
    add_macro(nm)

# TNX (10Y yield): returns + level
if "TNX" in raw_yf:
    tnx = raw_yf["TNX"].reindex(c.index).ffill(limit=5)
    for k in [1, 5, 10, 14, 20]:
        feats[f"tnx_ret_{k}"] = np.log(tnx + 1e-9).diff().rolling(k).sum()
    feats["tnx_vol_20"] = np.log(tnx + 1e-9).diff().rolling(20).std()

# ── GROUP BASE: Yield curve spread (10Y − 3M) ────────────────────────────────
if "TNX" in raw_yf and "IRX" in raw_yf:
    tnx_ = raw_yf["TNX"].reindex(c.index).ffill(limit=5)
    irx_ = raw_yf["IRX"].reindex(c.index).ffill(limit=5)
    spread = tnx_ - irx_
    feats["spread_10y_3m"]        = spread
    feats["spread_10y_3m_chg_5"]  = spread.diff(5)
    feats["spread_10y_3m_chg_20"] = spread.diff(20)
    print(f"  Yield spread engineered: 10Y−3M current = {spread.dropna().iloc[-1]:.2f}%")
else:
    print("  WARNING: IRX or TNX not available — yield spread features skipped.")

# ── GROUP BASE: Economic activity ─────────────────────────────────────────────
for nm in ["XLI", "XLF"]:
    add_macro(nm)

if "HG" in raw_yf:
    hg   = raw_yf["HG"].reindex(c.index).ffill(limit=5)
    r_hg = np.log(hg + 1e-9).diff()
    for k in [1, 5, 10, 14, 20]:
        feats[f"hg_ret_{k}"] = r_hg.rolling(k).sum()
    feats["hg_vol_20"] = r_hg.rolling(20).std()
    if "GOLD" in raw_yf:
        gold_ = raw_yf["GOLD"].reindex(c.index).ffill(limit=5)
        cg    = np.log(hg / gold_)
        feats["copper_gold_ratio"]    = cg
        feats["copper_gold_chg_20"]   = cg.diff(20)
    print("  Copper features engineered.")
else:
    print("  WARNING: HG (Copper) not available — econ-activity features reduced.")

# ── GROUP BASE: Energy ────────────────────────────────────────────────────────
for nm in ["CL", "XLE"]:
    add_macro(nm)
if "NG" in raw_yf:
    ng   = raw_yf["NG"].reindex(c.index).ffill(limit=5)
    r_ng = np.log(ng + 1e-9).diff()
    for k in [1, 5, 10, 14, 20]:
        feats[f"ng_ret_{k}"] = r_ng.rolling(k).sum()
    feats["ng_vol_20"] = r_ng.rolling(20).std()
    print("  Natural gas features engineered.")
else:
    print("  WARNING: NG not available — natural gas features skipped.")

# ── GROUP BASE: ETH/BTC ratio (crypto regime signal) ─────────────────────────
if "ETH" in raw_yf:
    eth_ = raw_yf["ETH"].reindex(c.index).ffill(limit=5)
    feats["eth_btc_ratio"]        = np.log(eth_ / c)
    feats["eth_btc_ratio_chg_14"] = feats["eth_btc_ratio"].diff(14)

# ── GROUP TI-B: Momentum (#1 new group for 7d, +8.1pp direction) ─────────────
print("  Engineering TI-B Momentum features …")

# Determine BTC H/L/V series for TI-B
# Try Yahoo OHLCV first; fallback: approximate H≈L≈close (TI-B degrades gracefully)
if btc_ohlcv is not None:
    h_btc  = btc_ohlcv["High"].loc[START:].reindex(c.index).ffill(limit=5)
    lo_btc = btc_ohlcv["Low"].loc[START:].reindex(c.index).ffill(limit=5)
else:
    h_btc  = c.copy()
    lo_btc = c.copy()
    print("    WARNING: BTC OHLCV not available — H/L approximated from Close for TI-B")

feats["rsi_7"]  = compute_rsi(c, 7)
feats["rsi_21"] = compute_rsi(c, 21)
feats["rsi_30"] = compute_rsi(c, 30)

stoch_k, stoch_d = compute_stoch(h_btc, lo_btc, c, k=14, d=3)
feats["stoch_k"]      = stoch_k
feats["stoch_d"]      = stoch_d
feats["stoch_kd_diff"] = stoch_k - stoch_d

highest_high_14 = h_btc.rolling(14).max()
lowest_low_14   = lo_btc.rolling(14).min()
feats["williams_r"] = -100 * (highest_high_14 - c) / (highest_high_14 - lowest_low_14 + 1e-9)

tp        = (h_btc + lo_btc + c) / 3
sma_tp_20 = tp.rolling(20).mean()
mad_tp_20 = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
feats["cci_20"] = (tp - sma_tp_20) / (0.015 * mad_tp_20 + 1e-9)

feats["roc_7"]  = c / c.shift(7)  - 1
feats["roc_14"] = c / c.shift(14) - 1

print("    TI-B Momentum: rsi_7/21/30, stoch_k/d, stoch_kd_diff, williams_r, cci_20, roc_7/14")

# ── GROUP OC-B: Miner/flow (#2 new group for 7d, +7.2pp direction) ───────────
print("  Engineering OC-B Miner/flow features …")

OC_NAME_MAP = {
    "tx_fees_usd": "oc_tx_fees_usd",
    "miners_rev":  "oc_miners_rev",
    "mempool_sz":  "oc_mempool_sz",
    "tx_vol_usd":  "oc_tx_vol_usd",
}

for raw_name, feat_prefix in OC_NAME_MAP.items():
    if raw_name not in onchain_raw:
        continue
    s        = onchain_raw[raw_name].reindex(c.index).ffill(limit=5)
    log_diff = np.log(s.clip(lower=1e-9)).diff()
    feats[f"{feat_prefix}_ret_7"]  = log_diff.rolling(7).sum()
    feats[f"{feat_prefix}_ret_30"] = log_diff.rolling(30).sum()
    feats[f"{feat_prefix}_vol_20"] = log_diff.rolling(20).std()
    print(f"    {feat_prefix}: ret_7, ret_30, vol_20")

# NVT proxy: nvt = log(close * 19_500_000 / (tx_vol_usd + 1e-9))
if "tx_vol_usd" in onchain_raw:
    tv_usd = onchain_raw["tx_vol_usd"].reindex(c.index).ffill(limit=5)
    nvt    = np.log(c * 19_500_000 / (tv_usd + 1e-9))
    feats["oc_nvt_proxy_chg_30"] = nvt.diff(30)
    print("    oc_nvt_proxy_chg_30 engineered")

# Miner stress: log(miners_rev / (tx_fees_usd + 1e-9)) — revenue-to-fees ratio
if "miners_rev" in onchain_raw and "tx_fees_usd" in onchain_raw:
    mr  = onchain_raw["miners_rev"].reindex(c.index).ffill(limit=5)
    tf  = onchain_raw["tx_fees_usd"].reindex(c.index).ffill(limit=5)
    ms  = np.log(mr / (tf + 1e-9))
    feats["oc_miner_stress_chg_30"] = ms.diff(30)
    print("    oc_miner_stress_chg_30 engineered")

# ── Assemble feature matrix ───────────────────────────────────────────────────
feats_df = pd.DataFrame(feats, index=c.index)
feats_df["close"]       = c
feats_df["y_logret_7"]  = np.log(c.shift(-HORIZON) / c)
feats_df = feats_df.replace([np.inf, -np.inf], np.nan)

# Columns that are >25% NaN → drop (e.g. ticker unavailable for most of history)
na_frac   = feats_df.drop(columns=["y_logret_7", "close"]).isna().mean()
drop_cols = na_frac[na_frac > 0.25].index.tolist()
if drop_cols:
    print(f"  Dropping {len(drop_cols)} high-NA cols: {drop_cols}")
feats_df = feats_df.drop(columns=drop_cols)

# Forward-fill the small warmup NaN gaps (rolling windows)
feat_cols_all = [col for col in feats_df.columns
                 if col not in ("y_logret_7", "close")]
feats_df[feat_cols_all] = feats_df[feat_cols_all].ffill(limit=5).bfill(limit=40)

print(f"\n  Feature matrix: {feats_df.shape}  "
      f"({feats_df.shape[1]-2} features + close + 1 target)")

# ─────────────────────────────────────────────────────────────────────────────
# 6. TRAIN / TEST SPLIT  (same convention as v1: last 8 months, 7-day embargo)
# ─────────────────────────────────────────────────────────────────────────────
TODAY        = pd.Timestamp(datetime.utcnow().date())
EMBARGO_DAYS = HORIZON   # = 7
test_start   = TODAY      - pd.DateOffset(months=8)
train_end    = test_start - pd.Timedelta(days=EMBARGO_DAYS)

clean = feats_df.dropna(subset=["y_logret_7"])
train = clean.loc[:train_end]
test  = clean.loc[test_start:]

print(f"\nTRAIN {train.index.min().date()} → {train.index.max().date()}  n={len(train)}")
print(f"  embargo of {EMBARGO_DAYS} days")
print(f"TEST  {test.index.min().date()} → {test.index.max().date()}   n={len(test)}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. FIT REGIME CONE  (unchanged from v1 — used for band calibration)
# ─────────────────────────────────────────────────────────────────────────────
edges  = np.quantile(train["range_ma30"].dropna(), [1/3, 2/3])
print(f"\nTercile edges (range_ma30): {edges.round(5)}")

def to_regime(x):
    return np.searchsorted(edges, np.asarray(x, dtype=float), side="right").clip(0, 2)

train_regimes = to_regime(train["range_ma30"])
regime_stats  = {}
for r in range(N_REGIMES):
    sub = train.loc[train_regimes == r, "y_logret_7"].values
    regime_stats[r] = {
        **{q: float(np.quantile(sub, q)) for q in QUANTILES},
        "mean": float(sub.mean()),
        "std":  float(sub.std()),
        "n":    int(len(sub)),
    }

print("\nRegime stats (7-day log-return quantiles):")
_df = pd.DataFrame(regime_stats).T
print(_df[["n", "mean", "std", 0.10, 0.25, 0.50, 0.75, 0.90]].round(4).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 8. FIT GBM POINT-PREDICTION MODEL
# ─────────────────────────────────────────────────────────────────────────────
print("\n>>> Training GBM point-prediction model …")

ML_FEAT_COLS = [col for col in feat_cols_all if col in train.columns]

X_tr = train[ML_FEAT_COLS].fillna(0)
y_tr = train["y_logret_7"]
X_te = test[ML_FEAT_COLS].fillna(0)
y_te = test["y_logret_7"]

print(f"  Training GBM on {len(ML_FEAT_COLS)} features, n={len(X_tr)} rows …")

gbm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("gbm",    GradientBoostingRegressor(
                   n_estimators     = 400,
                   max_depth        = 3,
                   learning_rate    = 0.03,
                   subsample        = 0.8,
                   min_samples_leaf = 15,
                   random_state     = 42,
               ))
])
gbm_pipeline.fit(X_tr, y_tr)
print("  GBM training complete.")

pred_gbm_tr = gbm_pipeline.predict(X_tr)
pred_gbm_te = gbm_pipeline.predict(X_te)

# ─────────────────────────────────────────────────────────────────────────────
# 9. CALIBRATE BAND WIDTH FROM RETURN DISTRIBUTION (same method as v1)
#
#    The band represents inherent 7-day BTC uncertainty — empirical spread of
#    historical outcomes, NOT in-sample model residuals (which would massively
#    underestimate OOS uncertainty due to GBM overfitting).
#
#    Method: empirical [q10, q90] half-width of 7-day returns per regime,
#    averaged across regimes, rounded up +0.5 pp for conservatism.
#    If the result falls outside [0.07, 0.14], clip to BAND_PCT_FALLBACK.
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
if band_pct < 0.07 or band_pct > 0.14:
    print(f"Band pct {band_pct*100:.1f}% out of range [7%, 14%] — "
          f"clipping to fallback {BAND_PCT_FALLBACK*100:.1f}%")
    band_pct = BAND_PCT_FALLBACK
print(f"Final band: ±{band_pct*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 10. EVALUATE — GBM v2 vs REGIME-CONE v1 BASELINE (both on hold-out)
# ─────────────────────────────────────────────────────────────────────────────
print("\n>>> Evaluating on hold-out test set …")

close_te = test["close"].values if "close" in test.columns else c.reindex(test.index).values
close_tr = train["close"].values if "close" in train.columns else c.reindex(train.index).values

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
test_regimes_arr = to_regime(test["range_ma30"])
cone_pred_te_lr  = np.array([regime_stats[r][0.50] for r in test_regimes_arr])
cone_pred_tr_lr  = np.array([regime_stats[r][0.50] for r in train_regimes])

# v1 band (from training quantiles — same calibration as above, reference)
hw_v1   = [(np.exp(regime_stats[r][0.90]) - np.exp(regime_stats[r][0.10])) / 2
           for r in range(N_REGIMES)]
band_v1 = float(np.ceil(np.mean(hw_v1) * 200) / 200 + 0.005)
if band_v1 < 0.07 or band_v1 > 0.14:
    band_v1 = BAND_PCT_FALLBACK

m_cone_te = eval_metrics(cone_pred_te_lr, y_te.values,  close_te, band_v1,  "Cone (v1)")
m_cone_tr = eval_metrics(cone_pred_tr_lr, y_tr.values,  close_tr, band_v1,  "Cone (v1) TRAIN")
m_gbm_te  = eval_metrics(pred_gbm_te,     y_te.values,  close_te, band_pct, "GBM (v2)")
m_gbm_tr  = eval_metrics(pred_gbm_tr,     y_tr.values,  close_tr, band_pct, "GBM (v2) TRAIN")

# ── Feature importances ───────────────────────────────────────────────────────
gbm_model  = gbm_pipeline.named_steps["gbm"]
imp_series = (pd.Series(gbm_model.feature_importances_, index=ML_FEAT_COLS)
              .sort_values(ascending=False))

# ── Print comparison table ────────────────────────────────────────────────────
print("\n" + "="*72)
print("  7-DAY MODEL — PERFORMANCE COMPARISON")
print("="*72)
print(f"\n  Band widths:  v1 (cone) ±{band_v1*100:.1f}%   v2 (GBM) ±{band_pct*100:.1f}%\n")

hdr = f"  {'Metric':28s}  {'Cone v1 TRAIN':>14} {'Cone v1 OOS':>13}  {'GBM v2 TRAIN':>14} {'GBM v2 OOS':>12}"
sep = "  " + "─"*78
print(hdr); print(sep)
for key, label in [("mape",  "MAPE (point forecast)"),
                   ("cov",   "Band coverage"),
                   ("dir",   "Direction accuracy"),
                   ("hit5",  "Hit rate ±5%"),
                   ("hit10", "Hit rate ±10%"),
                   ("hit15", "Hit rate ±15%")]:
    v_ctr = m_cone_tr[key]; v_cte = m_cone_te[key]
    v_gtr = m_gbm_tr[key];  v_gte = m_gbm_te[key]
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

print(f"\n  Regime breakdown on TEST (GBM v2):")
for r in range(N_REGIMES):
    mask_r  = (test_regimes_arr == r)
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
# 11. SAVE ARTEFACT  (backward-compatible + new ML keys)
# ─────────────────────────────────────────────────────────────────────────────
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUT = MODELS_DIR / "inference_assets_7d_cone.joblib"

art = dict(
    # ── v1 keys (unchanged schema) ───────────────────────────────────────────
    regime_feature   = "range_ma30",
    regime_edges     = edges.tolist(),
    regime_stats     = regime_stats,
    band_pct         = band_pct,
    horizon_days     = HORIZON,
    quantiles_stored = QUANTILES,
    # ── v2 keys (new) ────────────────────────────────────────────────────────
    use_ml           = True,
    ml_point_model   = gbm_pipeline,    # sklearn Pipeline (scaler + GBM)
    ml_feature_cols  = ML_FEAT_COLS,    # ordered list — must be reproduced at inference
    ml_metrics_oos   = m_gbm_te,
    # ── calibration metadata ─────────────────────────────────────────────────
    calibration_meta = dict(
        train_start    = str(train.index.min().date()),
        train_end      = str(train.index.max().date()),
        test_start     = str(test.index.min().date()),
        test_end       = str(test.index.max().date()),
        train_n        = int(len(train)),
        test_n         = int(len(test)),
        embargo_days   = int(EMBARGO_DAYS),
        method_v2      = ("GBM point prediction (BTC + macro + yield spread + "
                          "econ activity + energy + ETH/BTC ratio + "
                          "TI-B momentum + OC-B miner/flow); "
                          "regime-cone band from empirical q10/q90 return distribution"),
        method_v1      = "range_ma30 tercile regime cone; median 7d forward log-return",
        band_pct_v2    = float(band_pct),
        band_pct_v1    = float(band_v1),
        feature_count  = int(len(ML_FEAT_COLS)),
        cone_oos_metrics  = m_cone_te,
        ml_oos_metrics    = m_gbm_te,
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
        mape_delta_pp = round(m_gbm_te["mape"] - m_cone_te["mape"], 3),
        dir_delta_pp  = round(m_gbm_te["dir"]  - m_cone_te["dir"],  2),
        cov_delta_pp  = round(m_gbm_te["cov"]  - m_cone_te["cov"],  2),
    ),
    top5_features = imp_series.head(5).index.tolist(),
)
print("\n" + json.dumps(summary, indent=2))
