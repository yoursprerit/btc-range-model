"""Train the 14-day close-price *regime cone* and save its artefact.

Approach (mirrors train_7d_close_cone.py; horizon extended to 14 days):

  • Regime indicator: range_ma30 (30-day average daily range, already in
    the CT pipeline's feature matrix).
  • Bin training-set days into terciles of range_ma30.
  • For each tercile, record the empirical quantiles of the 14-day
    forward log-return.
  • At inference, classify today's range_ma30 into a regime, take the
    regime's median forward log-return → multiply against today's close,
    and apply a fixed band (the average empirical [q10, q90] half-width
    on the held-out test tail, rounded up for conservatism).

Training data: data/raw_ct.csv + data/features_ct.csv (12:00-UTC anchored).
Hold-out: last 8 months — same convention as the rest of the repo.
Embargo:  14 days (= horizon) between train_end and test_start.

If data/raw_ct.csv or data/features_ct.csv are missing, falls back to
fetching BTC daily prices from Yahoo Finance so the script can always run.
"""
import sys, json, joblib, warnings
from datetime import datetime
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from paths import RAW_CT_CSV, FEATURES_CT_CSV, MODELS_DIR, DATA_DIR

HORIZON   = 14
N_REGIMES = 3
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]

# ─────────────────────────────────────────────────────────────────────
# 1. LOAD DATA  (pipeline CSVs preferred; Yahoo fallback)
# ─────────────────────────────────────────────────────────────────────
def _load_from_pipeline():
    """Return (close_series, range_ma30_series) from existing CSV artefacts."""
    if not (RAW_CT_CSV.exists() and FEATURES_CT_CSV.exists()):
        return None, None
    raw = pd.read_csv(RAW_CT_CSV, index_col=0, parse_dates=True)
    fts = pd.read_csv(FEATURES_CT_CSV, index_col=0, parse_dates=True)
    c   = raw["btc_close"]
    if "range_ma30" not in fts.columns:
        return None, None
    rm30 = fts["range_ma30"]
    return c, rm30


def _load_from_yahoo():
    """Fetch BTC-USD daily from Yahoo Finance; compute range_ma30."""
    import yfinance as yf
    print("  [fallback] Fetching BTC-USD from Yahoo Finance …")
    btc = yf.download("BTC-USD", start="2017-01-01", progress=False, auto_adjust=True)
    if btc.empty:
        raise RuntimeError("Yahoo Finance returned empty data.")
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = [c[0] for c in btc.columns]
    btc.index = pd.to_datetime(btc.index).tz_localize(None).normalize()
    c    = btc["Close"].sort_index()
    h    = btc["High"].sort_index()
    lo   = btc["Low"].sort_index()
    rng  = (h - lo) / c
    rm30 = rng.rolling(30).mean()
    print(f"  Yahoo: {len(c)} daily bars  "
          f"{c.index.min().date()} → {c.index.max().date()}")
    return c, rm30


print(">>> Loading data …")
c, rm30 = _load_from_pipeline()
if c is None:
    print("  Pipeline CSVs not found — using Yahoo Finance fallback.")
    c, rm30 = _load_from_yahoo()
else:
    print(f"  Pipeline CSVs loaded: {len(c)} bars  "
          f"{c.index.min().date()} → {c.index.max().date()}")

# ─────────────────────────────────────────────────────────────────────
# 2. BUILD TARGET  y = log(close[t+14] / close[t])
# ─────────────────────────────────────────────────────────────────────
data = pd.DataFrame({"close": c, "range_ma30": rm30})
data["y_logret_14"] = np.log(c.shift(-HORIZON) / c)
data = data.replace([np.inf, -np.inf], np.nan).dropna(
    subset=["range_ma30", "y_logret_14"])
data = data.loc["2019-01-01":]
print(f"Dataset: {data.shape}   {data.index.min().date()} → {data.index.max().date()}")

# ─────────────────────────────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT  with HORIZON-day embargo
# ─────────────────────────────────────────────────────────────────────
TODAY        = pd.Timestamp(datetime.utcnow().date())
EMBARGO_DAYS = HORIZON   # = 14
test_start   = TODAY    - pd.DateOffset(months=8)
train_end    = test_start - pd.Timedelta(days=EMBARGO_DAYS)

train = data.loc[: train_end]
test  = data.loc[test_start:]

print(f"\nTRAIN {train.index.min().date()} → {train.index.max().date()}  n={len(train)}")
print(f"  embargo of {EMBARGO_DAYS} days")
print(f"TEST  {test.index.min().date()} → {test.index.max().date()}   n={len(test)}")

# ─────────────────────────────────────────────────────────────────────
# 4. FIT REGIME CONE  (terciles of range_ma30 on training set)
# ─────────────────────────────────────────────────────────────────────
edges = np.quantile(train["range_ma30"], [1/3, 2/3])
print(f"\nTercile edges (range_ma30): {edges.round(5)}")

def to_regime(x):
    return np.searchsorted(edges, np.asarray(x, dtype=float), side="right")

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

# ─────────────────────────────────────────────────────────────────────
# 5. CALIBRATE BAND WIDTH on TRAINING SET
#    The empirical [q10, q90] half-width = (q90 - q10) / 2 per regime.
#    We use the training-weighted average and round up for conservatism.
# ─────────────────────────────────────────────────────────────────────
hw_list = []
for r in range(N_REGIMES):
    q10 = regime_stats[r][0.10]; q90 = regime_stats[r][0.90]
    # Half-width in price-space around the median: convert log-return spread
    # to an approximate ±% band.
    hw = (np.exp(q90) - np.exp(q10)) / 2
    hw_list.append(hw)

band_pct_raw = float(np.mean(hw_list))
# Round to nearest 0.5 % for clean display; add 1 pp for conservatism.
band_pct = float(np.ceil(band_pct_raw * 200) / 200 + 0.005)
print(f"\nRaw empirical half-width (train, avg across regimes): {band_pct_raw*100:.2f}%")
print(f"Rounded + conservative band: ±{band_pct*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────
# 6. EVALUATE ON TEST SET
# ─────────────────────────────────────────────────────────────────────
test_regimes  = to_regime(test["range_ma30"])
median_logret = np.array([regime_stats[r][0.50] for r in test_regimes])
pred_close_te = test["close"].values * np.exp(median_logret)
actual_close_te = test["close"].values * np.exp(test["y_logret_14"].values)

# ── MAPE on the 14-day forward close ──────────────────────────────────
mape_te = float(np.mean(np.abs(pred_close_te - actual_close_te) / actual_close_te) * 100)

# ── Band coverage (does the realized 14d return fall inside ±band_pct?) ──
band_lo = np.log(1 - band_pct) + median_logret
band_hi = np.log(1 + band_pct) + median_logret
inside  = (test["y_logret_14"].values >= band_lo) & (test["y_logret_14"].values <= band_hi)
coverage_pct = float(100 * inside.mean())

# ── Direction accuracy (does the sign of our 14-day median match realized?) ──
sign_pred = np.sign(median_logret)
sign_real = np.sign(test["y_logret_14"].values)
dir_acc   = float(100 * (sign_pred == sign_real).mean())

# ── Hit rates within ±5 % / ±10 % / ±15 % of actual ─────────────────
rel_err = np.abs(pred_close_te - actual_close_te) / actual_close_te
hit5  = float(100 * (rel_err <= 0.05).mean())
hit10 = float(100 * (rel_err <= 0.10).mean())
hit15 = float(100 * (rel_err <= 0.15).mean())

# ── Also evaluate on TRAIN (in-sample sanity) ─────────────────────────
train_regimes_ = to_regime(train["range_ma30"])
med_train = np.array([regime_stats[r][0.50] for r in train_regimes_])
pred_train    = train["close"].values * np.exp(med_train)
actual_train  = train["close"].values * np.exp(train["y_logret_14"].values)
mape_train    = float(np.mean(np.abs(pred_train - actual_train) / actual_train) * 100)

band_lo_tr = np.log(1 - band_pct) + med_train
band_hi_tr = np.log(1 + band_pct) + med_train
inside_tr  = ((train["y_logret_14"].values >= band_lo_tr) &
              (train["y_logret_14"].values <= band_hi_tr))
cov_train  = float(100 * inside_tr.mean())

rel_err_tr = np.abs(pred_train - actual_train) / actual_train
hit5_tr  = float(100 * (rel_err_tr <= 0.05).mean())
hit10_tr = float(100 * (rel_err_tr <= 0.10).mean())
hit15_tr = float(100 * (rel_err_tr <= 0.15).mean())
dir_tr   = np.sign(med_train) == np.sign(train["y_logret_14"].values)
diracc_tr = float(100 * dir_tr.mean())

metrics = dict(
    horizon_days      = HORIZON,
    band_pct          = band_pct,

    train_n           = int(len(train)),
    train_mape_pct    = mape_train,
    train_coverage_pct= cov_train,
    train_hit5_pct    = hit5_tr,
    train_hit10_pct   = hit10_tr,
    train_hit15_pct   = hit15_tr,
    train_dir_acc_pct = diracc_tr,

    test_n            = int(len(test)),
    test_mape_pct     = mape_te,
    test_coverage_pct = coverage_pct,
    test_hit5_pct     = hit5,
    test_hit10_pct    = hit10,
    test_hit15_pct    = hit15,
    test_dir_acc_pct  = dir_acc,
)

print("\n" + "="*62)
print("  14-DAY CLOSE CONE — PERFORMANCE REPORT")
print("="*62)
print(f"\n  Band width:   ±{band_pct*100:.1f}%")
print(f"\n  ┌{'─'*30}┬{'─'*13}┬{'─'*13}┐")
print(f"  │{'Metric':^30}│{'TRAIN':^13}│{'TEST (OOS)':^13}│")
print(f"  ├{'─'*30}┼{'─'*13}┼{'─'*13}┤")
print(f"  │{'Rows':^30}│{len(train):^13,}│{len(test):^13,}│")
print(f"  │{'MAPE (point forecast)':^30}│{mape_train:^12.2f}%│{mape_te:^12.2f}%│")
print(f"  │{'Band coverage':^30}│{cov_train:^12.1f}%│{coverage_pct:^12.1f}%│")
print(f"  │{'Hit rate ±5%':^30}│{hit5_tr:^12.1f}%│{hit5:^12.1f}%│")
print(f"  │{'Hit rate ±10%':^30}│{hit10_tr:^12.1f}%│{hit10:^12.1f}%│")
print(f"  │{'Hit rate ±15%':^30}│{hit15_tr:^12.1f}%│{hit15:^12.1f}%│")
print(f"  │{'Direction accuracy':^30}│{diracc_tr:^12.1f}%│{dir_acc:^12.1f}%│")
print(f"  └{'─'*30}┴{'─'*13}┴{'─'*13}┘")

# Regime breakdown on test
print("\n  Regime breakdown on TEST:")
for r in range(N_REGIMES):
    mask = (test_regimes == r)
    n_r  = mask.sum()
    if n_r == 0:
        continue
    inside_r = inside[mask].mean() * 100
    mape_r   = (np.abs(pred_close_te[mask] - actual_close_te[mask])
                / actual_close_te[mask]).mean() * 100
    label = ["low vol", "mid vol", "high vol"][r]
    print(f"    [{label:>8}]  n={n_r:3d}  "
          f"MAPE={mape_r:5.2f}%  coverage={inside_r:5.1f}%  "
          f"med_14d_ret={regime_stats[r][0.50]*100:+.2f}%")

# Context vs 7-day model
print("\n  Context: 7-day cone had ~1.5-3% MAPE and 88% band coverage.")
print(f"  14-day cone has wider uncertainty: "
      f"{mape_te:.2f}% MAPE, {coverage_pct:.1f}% coverage at ±{band_pct*100:.1f}%.")
print("="*62)

# ─────────────────────────────────────────────────────────────────────
# 7. SAVE ARTEFACT
# ─────────────────────────────────────────────────────────────────────
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUT = MODELS_DIR / "inference_assets_14d_cone.joblib"

art = dict(
    regime_feature    = "range_ma30",
    regime_edges      = edges.tolist(),
    regime_stats      = regime_stats,
    band_pct          = band_pct,
    horizon_days      = HORIZON,
    quantiles_stored  = QUANTILES,
    calibration_meta  = dict(
        train_start   = str(train.index.min().date()),
        train_end     = str(train.index.max().date()),
        test_start    = str(test.index.min().date()),
        test_end      = str(test.index.max().date()),
        train_n       = int(len(train)),
        test_n        = int(len(test)),
        embargo_days  = int(EMBARGO_DAYS),
        method        = ("range_ma30 tercile regime cone; "
                         "median 14-day forward log-return"),
        metrics       = metrics,
        held_out_band_coverage_pct = float(coverage_pct),
        held_out_mape_pct          = float(mape_te),
    ),
)

joblib.dump(art, OUT)
print(f"\nSaved → {OUT}")
print(json.dumps(metrics, indent=2))
