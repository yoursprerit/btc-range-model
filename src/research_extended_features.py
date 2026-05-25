"""
Comprehensive feature expansion research for BTC 7-day and 14-day models.

Feature groups tested (all sourced from Yahoo Finance, daily data):
  A: Yield curve       — 2Y/5Y/30Y Treasuries, yield-curve spreads
  B: Credit / bonds    — HYG (junk), LQD (IG), TLT (long-dur), TIP (TIPS/inflation)
  C: Economic activity — Copper (HG=F), Silver (SI=F), XLI (industrials), XLF (financials)
  D: Geopolitical risk — EEM (emerging-markets), MOVE proxy (OVX crude-vol), GLD (gold vol)
  E: Crypto ecosystem  — SOL, BNB, MSTR (BTC proxy), COIN (Coinbase)
  F: Fed / FOMC        — "days-to-next-FOMC" synthetic feature + Fed-hike indicator
  G: Baseline macro    — SPX, NDX, VIX, Gold, DXY, TNX, ETH (already in production)
  H: Energy            — CL, NG, XLE (from previous research)

For each horizon (7d, 14d) the script reports:
  • Top-20 features by GBM importance (trained on ALL features)
  • Ablation: each group added one-at-a-time to the baseline macro set
  • Best combined model vs. production regime-cone baseline
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

# ── FOMC meeting dates 2019-2026 ────────────────────────────────────────────
FOMC_DATES = pd.to_datetime([
    # 2019
    "2019-01-30","2019-03-20","2019-05-01","2019-06-19",
    "2019-07-31","2019-09-18","2019-10-30","2019-12-11",
    # 2020
    "2020-01-29","2020-03-03","2020-03-15","2020-04-29",
    "2020-06-10","2020-07-29","2020-09-16","2020-11-05","2020-12-16",
    # 2021
    "2021-01-27","2021-03-17","2021-04-28","2021-06-16",
    "2021-07-28","2021-09-22","2021-11-03","2021-12-15",
    # 2022
    "2022-01-26","2022-03-16","2022-05-04","2022-06-15",
    "2022-07-27","2022-09-21","2022-11-02","2022-12-14",
    # 2023
    "2023-02-01","2023-03-22","2023-05-03","2023-06-14",
    "2023-07-26","2023-09-20","2023-11-01","2023-12-13",
    # 2024
    "2024-01-31","2024-03-20","2024-05-01","2024-06-12",
    "2024-07-31","2024-09-18","2024-11-07","2024-12-18",
    # 2025
    "2025-01-29","2025-03-19","2025-05-07","2025-06-18",
    "2025-07-30","2025-09-17","2025-10-29","2025-12-10",
    # 2026
    "2026-01-28","2026-03-18","2026-05-06",
])

def fomc_features(index):
    """Days to next FOMC, days since last FOMC, and 'FOMC week' binary."""
    idx   = pd.DatetimeIndex(index)
    dtf   = pd.Series(0.0, index=idx)  # days to next
    dsf   = pd.Series(0.0, index=idx)  # days since last
    week  = pd.Series(0,   index=idx)  # 1 if within ±3 days of FOMC
    dates = FOMC_DATES.sort_values()
    for d in idx:
        future = dates[dates >= d]
        past   = dates[dates <= d]
        dtf[d] = (future.min() - d).days if len(future) else 90
        dsf[d] = (d - past.max()).days   if len(past)   else 90
        week[d] = 1 if dtf[d] <= 3 or dsf[d] <= 3 else 0
    return dtf.rename("days_to_fomc"), dsf.rename("days_since_fomc"), week.rename("fomc_week")

# ── DATA DOWNLOAD ────────────────────────────────────────────────────────────
print(">>> Fetching data from Yahoo Finance …")
TICKERS = {
    # Baseline macro (already in production)
    "BTC":  "BTC-USD",
    "SPX":  "^GSPC",
    "NDX":  "^IXIC",
    "VIX":  "^VIX",
    "GOLD": "GC=F",
    "DXY":  "DX-Y.NYB",
    "TNX":  "^TNX",   # 10Y Treasury yield
    "ETH":  "ETH-USD",
    # Group A: Yield curve
    "IRX":  "^IRX",   # 13-week T-bill (≈ Fed funds rate proxy)
    "FVX":  "^FVX",   # 5-year Treasury yield
    "TYX":  "^TYX",   # 30-year Treasury yield
    # Group B: Credit / bonds
    "HYG":  "HYG",    # High-yield (junk) bond ETF
    "LQD":  "LQD",    # Investment-grade bond ETF
    "TLT":  "TLT",    # 20+ year Treasury ETF
    "TIP":  "TIP",    # TIPS bond ETF (inflation proxy)
    # Group C: Economic activity / commodities
    "HG":   "HG=F",   # Copper futures
    "SI":   "SI=F",   # Silver futures
    "XLI":  "XLI",    # Industrials sector ETF
    "XLF":  "XLF",    # Financials sector ETF
    # Group D: Geopolitical / risk-off
    "EEM":  "EEM",    # Emerging markets (geopolitical risk proxy)
    "GLD":  "GLD",    # Gold ETF (flight-to-safety)
    "OVX":  "OVX",    # Crude Oil Volatility Index (if available)
    "GVZ":  "GVZ",    # Gold Volatility Index (if available)
    # Group E: Crypto ecosystem
    "SOL":  "SOL-USD",
    "BNB":  "BNB-USD",
    "MSTR": "MSTR",   # MicroStrategy (BTC treasury proxy)
    "COIN": "COIN",   # Coinbase (crypto market health)
    # Group H: Energy (from previous study)
    "CL":   "CL=F",
    "NG":   "NG=F",
    "XLE":  "XLE",
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
        print(f"  {name:6s} ({sym:15s}): *** NOT AVAILABLE ***")

# ── FEATURE ENGINEERING ─────────────────────────────────────────────────────
print("\n>>> Engineering features …")
df  = pd.DataFrame(raw).sort_index().loc["2019-01-01":]
df  = df.ffill(limit=5)
c   = df["BTC"]

def log_ret_feats(s, name, ks=(1, 5, 7, 10, 14, 20)):
    r = np.log(s + 1e-9).diff()
    out = {}
    for k in ks:
        out[f"{name}_ret_{k}"] = r.rolling(k).sum()
    out[f"{name}_vol_20"] = r.rolling(20).std()
    return out

ret  = np.log(c).diff()
rng  = ret.abs()
rm30 = rng.rolling(30).mean()

feats = {}

# BTC own features
for k in [1, 2, 3, 5, 7, 10, 14, 21]:
    feats[f"btc_ret_{k}"] = ret.rolling(k).sum()
for k in [5, 7, 10, 14, 20, 30]:
    feats[f"btc_vol_{k}"] = ret.rolling(k).std()
delta = c.diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = (-delta.clip(upper=0)).rolling(14).mean()
feats["btc_rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
feats["range_ma30"] = rm30

# ── Group G: Baseline macro (production)
for nm in ["SPX","NDX","VIX","GOLD","DXY","TNX","ETH"]:
    if nm in df.columns:
        feats.update(log_ret_feats(df[nm], nm.lower()))

# ── Group A: Yield curve
for nm in ["IRX","FVX","TYX"]:
    if nm in df.columns:
        feats.update(log_ret_feats(df[nm], nm.lower(), ks=(1,5,10,20)))
# Derived yield-curve spreads (level differences, no log needed)
if "TNX" in df.columns and "IRX" in df.columns:
    feats["spread_10y_3m"]  = df["TNX"] - df["IRX"]   # key recession signal
if "TYX" in df.columns and "IRX" in df.columns:
    feats["spread_30y_3m"]  = df["TYX"] - df["IRX"]
if "TYX" in df.columns and "TNX" in df.columns:
    feats["spread_30y_10y"] = df["TYX"] - df["TNX"]
if "TNX" in df.columns and "FVX" in df.columns:
    feats["spread_10y_5y"]  = df["TNX"] - df["FVX"]
# Rolling change in spreads
for sp in ["spread_10y_3m","spread_30y_3m","spread_30y_10y","spread_10y_5y"]:
    if sp in feats:
        s = feats[sp]
        for k in [5, 20]:
            feats[f"{sp}_chg_{k}"] = s.diff(k)

# ── Group B: Credit / bonds
for nm in ["HYG","LQD","TLT","TIP"]:
    if nm in df.columns:
        feats.update(log_ret_feats(df[nm], nm.lower()))
# Credit spread proxy: HYG/LQD relative return (risk appetite)
if "HYG" in df.columns and "LQD" in df.columns:
    feats["credit_spread_proxy"] = np.log(df["HYG"]).diff(20) - np.log(df["LQD"]).diff(20)
# Real yield proxy: TNX change - TIP change
if "TNX" in df.columns and "TIP" in df.columns:
    feats["real_yield_proxy"] = df["TNX"].diff(20) - np.log(df["TIP"]).diff(20) * 100

# ── Group C: Economic activity
for nm in ["HG","SI","XLI","XLF"]:
    if nm in df.columns:
        feats.update(log_ret_feats(df[nm], nm.lower()))
# Copper/Gold ratio (Dr. Copper = economic health)
if "HG" in df.columns and "GOLD" in df.columns:
    feats["copper_gold_ratio"] = np.log(df["HG"] / df["GOLD"])
    feats["copper_gold_chg_20"] = feats["copper_gold_ratio"].diff(20)

# ── Group D: Geopolitical / risk-off
for nm in ["EEM","GLD","OVX","GVZ"]:
    if nm in df.columns:
        feats.update(log_ret_feats(df[nm], nm.lower(), ks=(1,5,10,20)))
# EM vs SPX (geopolitical risk-off): falling EEM/SPX = risk-off
if "EEM" in df.columns and "SPX" in df.columns:
    feats["em_vs_spx"]      = np.log(df["EEM"] / df["SPX"])
    feats["em_vs_spx_chg_20"] = feats["em_vs_spx"].diff(20)
# VIX level buckets (already in macro but add 30-day VIX MA for regime)
if "VIX" in df.columns:
    feats["vix_ma30"]     = df["VIX"].rolling(30).mean()
    feats["vix_vs_ma30"]  = df["VIX"] / feats["vix_ma30"] - 1  # VIX spike signal

# ── Group E: Crypto ecosystem
for nm in ["SOL","BNB","MSTR","COIN"]:
    if nm in df.columns:
        feats.update(log_ret_feats(df[nm], nm.lower()))
# Alt-season signal: ETH/BTC ratio trend
if "ETH" in df.columns:
    feats["eth_btc_ratio"]      = np.log(df["ETH"] / c)
    feats["eth_btc_ratio_chg_14"] = feats["eth_btc_ratio"].diff(14)

# ── Group F: FOMC / Fed
print("  Computing FOMC features (may take a moment) …")
dtf, dsf, fweek = fomc_features(df.index)
feats["days_to_fomc"]   = dtf
feats["days_since_fomc"]= dsf
feats["fomc_week"]      = fweek.astype(float)

# ── Group H: Energy
for nm in ["CL","NG","XLE"]:
    if nm in df.columns:
        feats.update(log_ret_feats(df[nm], nm.lower()))

# ── Assemble ──────────────────────────────────────────────────────────────
feats_df = pd.DataFrame(feats, index=df.index)
print(f"  Total features before NA-drop: {feats_df.shape[1]}")

# Targets for both horizons
feats_df["y7"]  = np.log(c.shift(-7)  / c)
feats_df["y14"] = np.log(c.shift(-14) / c)
feats_df = feats_df.replace([np.inf,-np.inf], np.nan)

# Drop columns that are >30% NaN (e.g. recently listed tickers)
na_frac = feats_df.isna().mean()
drop_cols = na_frac[na_frac > 0.30].index.tolist()
if drop_cols:
    print(f"  Dropping {len(drop_cols)} cols >30% NaN: {drop_cols[:10]} …")
feats_df = feats_df.drop(columns=drop_cols)

# Forward-fill short gaps (weekends, holidays) then back-fill the leading
# window warmup period so rolling features don't leave sparse NaN rows.
feat_only = [c for c in feats_df.columns if c not in ("y7","y14")]
feats_df[feat_only] = feats_df[feat_only].ffill(limit=5).bfill(limit=40)

print(f"  Feature matrix after NA filter: {feats_df.shape}")

# ── TRAIN / TEST ─────────────────────────────────────────────────────────────
TODAY      = pd.Timestamp("today").normalize()
test_start = TODAY - pd.DateOffset(months=8)

ALL_FEAT_COLS = [col for col in feats_df.columns if col not in ("y7","y14")]

# Separate feature matrices per horizon (need slightly different embargo)
def make_splits(horizon):
    train_end = test_start - pd.Timedelta(days=horizon)
    tmp = feats_df.dropna(subset=[f"y{horizon}"])
    train = tmp.loc[:train_end]
    test  = tmp.loc[test_start:]
    return train, test

train7,  test7  = make_splits(7)
train14, test14 = make_splits(14)
print(f"\n  7d  TRAIN {train7.index.min().date()} → {train7.index.max().date()}  n={len(train7)}")
print(f"      TEST  {test7.index.min().date()}  → {test7.index.max().date()}   n={len(test7)}")
print(f"  14d TRAIN {train14.index.min().date()} → {train14.index.max().date()}  n={len(train14)}")
print(f"      TEST  {test14.index.min().date()}  → {test14.index.max().date()}   n={len(test14)}")

close_te7  = df["BTC"].reindex(test7.index).values
close_te14 = df["BTC"].reindex(test14.index).values

# ── METRICS ──────────────────────────────────────────────────────────────────
def mape_price(pred_lr, act_lr, close_today):
    pc  = close_today * np.exp(pred_lr)
    ac  = close_today * np.exp(act_lr)
    return float(np.mean(np.abs(pc - ac) / ac) * 100)

def dir_acc(pred, actual):
    return float(100 * (np.sign(pred) == np.sign(actual)).mean())

def band_cov(pred, actual, band_pct):
    hi = pred + np.log(1 + band_pct)
    lo = pred + np.log(1 - band_pct)
    return float(100 * ((actual >= lo) & (actual <= hi)).mean())

def mae_pct(pred, actual):
    return float(np.mean(np.abs(np.exp(pred) - np.exp(actual))) * 100)

BAND_7  = 0.097
BAND_14 = 0.160

# ── REGIME-CONE BASELINE ─────────────────────────────────────────────────────
def regime_cone_baseline(train_df, test_df, target_col, band_pct, close_te):
    edges = np.quantile(train_df["range_ma30"], [1/3, 2/3])
    def reg(x):
        return np.searchsorted(edges, np.asarray(x, float), side="right").clip(0,2)
    tr_r = reg(train_df["range_ma30"])
    te_r = reg(test_df["range_ma30"])
    meds  = {r: float(np.median(train_df[target_col].values[tr_r == r])) for r in range(3)}
    pred  = np.array([meds[r] for r in te_r])
    act   = test_df[target_col].values
    return dict(
        mape  = mape_price(pred, act, close_te),
        dir   = dir_acc(pred, act),
        cov   = band_cov(pred, act, band_pct),
        pred  = pred
    )

bl7  = regime_cone_baseline(train7,  test7,  "y7",  BAND_7,  close_te7)
bl14 = regime_cone_baseline(train14, test14, "y14", BAND_14, close_te14)

print(f"\n{'='*76}")
print(f"  BASELINES (regime cone, no ML)")
print(f"{'='*76}")
print(f"  7d  MAPE={bl7['mape']:.2f}%  DirAcc={bl7['dir']:.1f}%  Cov±{BAND_7*100:.1f}%={bl7['cov']:.1f}%")
print(f"  14d MAPE={bl14['mape']:.2f}%  DirAcc={bl14['dir']:.1f}%  Cov±{BAND_14*100:.1f}%={bl14['cov']:.1f}%")

# ── GBM HELPERS ──────────────────────────────────────────────────────────────
def fit_gbm(Xtr, ytr, Xte, yte, close_te, band_pct, n_est=300, depth=3, lr=0.04):
    """Fit GBM; Xtr/Xte must be feature-only DataFrames (no target column)."""
    shared = sorted(set(Xtr.columns) & set(Xte.columns))
    Xtr_s  = Xtr[shared].fillna(0)
    Xte_s  = Xte[shared].fillna(0)
    ytr_s  = ytr.reindex(Xtr_s.index)
    # Drop rows where target is NaN
    mask   = ytr_s.notna()
    Xtr_s  = Xtr_s[mask]
    ytr_s  = ytr_s[mask]
    m = Pipeline([("sc", StandardScaler()),
                  ("m", GradientBoostingRegressor(
                      n_estimators=n_est, max_depth=depth,
                      learning_rate=lr, subsample=0.8,
                      min_samples_leaf=10, random_state=42))])
    m.fit(Xtr_s, ytr_s)
    pred = m.predict(Xte_s)
    act  = yte.values
    return dict(
        mape = mape_price(pred, act, close_te),
        dir  = dir_acc(pred, act),
        cov  = band_cov(pred, act, band_pct),
        pred = pred,
        model= m,
        cols = shared
    )

# ── FEATURE GROUPS ───────────────────────────────────────────────────────────
btc_base  = [f for f in ALL_FEAT_COLS if f.startswith("btc_") or f == "range_ma30"]
macro_g   = [f for f in ALL_FEAT_COLS if any(f.startswith(nm.lower()) for nm in
             ["spx","ndx","vix","gold","dxy","tnx","eth"]) and not f.startswith("eth_btc")]
yld_g     = [f for f in ALL_FEAT_COLS if any(f.startswith(p) for p in
             ["irx","fvx","tyx","spread_"])]
credit_g  = [f for f in ALL_FEAT_COLS if any(f.startswith(p) for p in
             ["hyg","lqd","tlt","tip","credit_","real_yield"])]
econ_g    = [f for f in ALL_FEAT_COLS if any(f.startswith(p) for p in
             ["hg","si","xli","xlf","copper_"])]
geo_g     = [f for f in ALL_FEAT_COLS if any(f.startswith(p) for p in
             ["eem","gld","ovx","gvz","em_vs_spx","vix_ma","vix_vs"])]
crypto_g  = [f for f in ALL_FEAT_COLS if any(f.startswith(p) for p in
             ["sol","bnb","mstr","coin","eth_btc"])]
fomc_g    = [f for f in ALL_FEAT_COLS if f in ["days_to_fomc","days_since_fomc","fomc_week"]]
energy_g  = [f for f in ALL_FEAT_COLS if any(f.startswith(p) for p in ["cl","ng","xle"])]

groups = {
    "A: Yield curve"     : yld_g,
    "B: Credit/bonds"    : credit_g,
    "C: Econ activity"   : econ_g,
    "D: Geopolitical"    : geo_g,
    "E: Crypto ecosystem": crypto_g,
    "F: FOMC/Fed"        : fomc_g,
    "H: Energy"          : energy_g,
}

print(f"\n  Feature group sizes:")
for k,v in groups.items():
    print(f"    {k:22s}: {len(v):3d} features")
print(f"    {'Baseline macro':22s}: {len(macro_g):3d} features")
print(f"    {'BTC own':22s}: {len(btc_base):3d} features")

# ══════════════════════════════════════════════════════════════════════════════
# FULL-FEATURE MODEL: find top features by importance
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*76}")
print(f"  STEP 1: Full-feature GBM — top features by importance (both horizons)")
print(f"{'='*76}")

for hz, train_df, test_df, yte, close_te, band_pct in [
        (7,  train7,  test7,  test7["y7"],   close_te7,  BAND_7),
        (14, train14, test14, test14["y14"], close_te14, BAND_14),
]:
    print(f"\n  --- {hz}-day horizon ---")
    # Feature columns only (exclude both target columns)
    use_cols = [c for c in ALL_FEAT_COLS if c in train_df.columns and c in test_df.columns]
    r = fit_gbm(train_df[use_cols],
                train_df[f"y{hz}"],
                test_df[use_cols],
                test_df[f"y{hz}"],
                close_te, band_pct)
    print(f"  Full model: MAPE={r['mape']:.2f}%  DirAcc={r['dir']:.1f}%  "
          f"Cov±{band_pct*100:.1f}%={r['cov']:.1f}%")

    # Feature importances
    gbm   = r["model"].named_steps["m"]
    impdf = (pd.Series(gbm.feature_importances_, index=r["cols"])
             .sort_values(ascending=False))

    # Assign group label
    def grp(f):
        for g, cols in groups.items():
            if f in cols: return g
        if f in macro_g: return "G: Baseline macro"
        if f in btc_base: return "BTC own"
        return "other"

    impdf_df = impdf.head(30).reset_index()
    impdf_df.columns = ["feature","importance"]
    impdf_df["group"] = impdf_df["feature"].map(grp)
    print(f"\n  Top 20 features by GBM importance ({hz}d):")
    print(f"  {'Rank':>4}  {'Feature':32s}  {'Importance':>11}  {'Group'}")
    print(f"  {'----':>4}  {'-------':32s}  {'-----------':>11}  {'-----'}")
    for i, row in impdf_df.head(20).iterrows():
        print(f"  {i+1:4d}  {row.feature:32s}  {row.importance:11.4f}  {row.group}")

    # Group importance shares
    print(f"\n  Importance by group ({hz}d):")
    grp_imp = impdf_df.groupby("group")["importance"].sum().sort_values(ascending=False)
    for g, imp in grp_imp.items():
        bar = "█" * int(imp * 200)
        print(f"  {g:22s}  {imp:.4f}  {bar}")

# ══════════════════════════════════════════════════════════════════════════════
# ABLATION: group-by-group added to baseline macro
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*76}")
print(f"  STEP 2: Ablation — each group added to BTC + baseline macro")
print(f"{'='*76}")

for hz, train_df, test_df, close_te, band_pct, bl in [
    (7,  train7,  test7,  close_te7,  BAND_7,  bl7),
    (14, train14, test14, close_te14, BAND_14, bl14),
]:
    yt   = f"y{hz}"
    ytr_ = train_df[yt]
    yte_ = test_df[yt]

    print(f"\n  {hz}-day horizon  (baseline MAPE={bl['mape']:.2f}%  "
          f"Dir={bl['dir']:.1f}%  Cov={bl['cov']:.1f}%)")
    print(f"  {'Config':38s}  {'MAPE':>7}  {'ΔDir':>6}  {'Cov':>7}  {'Verdict'}")
    print(f"  {'-'*38}  {'-'*7}  {'-'*6}  {'-'*7}  {'-'*30}")

    base_feat = btc_base + macro_g

    # BTC + macro only (GBM)
    base_avail = [f for f in base_feat if f in train_df.columns and f in test_df.columns]
    r0 = fit_gbm(train_df[base_avail], ytr_,
                 test_df[base_avail],  yte_, close_te, band_pct)
    delta_dir0 = r0["dir"] - bl["dir"]
    v0 = "▲ better dir" if delta_dir0 > 2 else ("▼ worse" if r0["mape"] > bl["mape"] else "≈ similar")
    print(f"  {'GBM BTC+macro only (no add-ons)':38s}  "
          f"{r0['mape']:6.2f}%  {delta_dir0:+5.1f}%  {r0['cov']:6.1f}%  {v0}")

    best = dict(mape=r0["mape"], dir=r0["dir"], cov=r0["cov"], feat=base_avail, label="BTC+macro")

    for gname, gcols in groups.items():
        add = [f for f in gcols if f in train_df.columns and f in test_df.columns]
        if not add:
            continue
        feat_set = base_avail + add
        r = fit_gbm(train_df[feat_set], ytr_,
                    test_df[feat_set], yte_, close_te, band_pct)
        delta_dir = r["dir"] - bl["dir"]
        verdict = []
        if r["mape"]  < bl["mape"] - 0.10:    verdict.append("✅ MAPE↓")
        if delta_dir  > 2.0:                   verdict.append("✅ Dir↑")
        if r["cov"]   > bl["cov"] + 1.0:       verdict.append("✅ Cov↑")
        if r["mape"]  > bl["mape"] + 0.20:     verdict.append("❌ MAPE↑")
        if delta_dir  < -2.0:                  verdict.append("❌ Dir↓")
        if not verdict:                        verdict = ["– negligible"]
        if r["mape"] < best["mape"] or r["dir"] > best["dir"]:
            best = dict(mape=r["mape"], dir=r["dir"], cov=r["cov"], feat=feat_set, label=gname)
        print(f"  {'+ '+gname:38s}  {r['mape']:6.2f}%  {delta_dir:+5.1f}%  "
              f"{r['cov']:6.1f}%  {' '.join(verdict)}")

    # Best cumulative: add all groups that help
    print(f"\n  Finding best additive combination …")
    helpful = []
    for gname, gcols in groups.items():
        add = [f for f in gcols if f in train_df.columns and f in test_df.columns]
        if not add: continue
        feat_set = base_feat + add
        r = fit_gbm(train_df[feat_set + [yt]], ytr_,
                    test_df[feat_set], yte_, close_te, band_pct)
        if r["dir"] > r0["dir"] + 1.0 or r["mape"] < r0["mape"] - 0.05:
            helpful.append((gname, add))

    if helpful:
        all_add = sum([c for _, c in helpful], [])
        feat_all = base_avail + all_add
        r_best = fit_gbm(train_df[feat_all], ytr_,
                         test_df[feat_all], yte_, close_te, band_pct)
        d_dir = r_best["dir"] - bl["dir"]
        d_mape = r_best["mape"] - bl["mape"]
        print(f"\n  Best combined GBM ({hz}d) — adds: {[n for n,_ in helpful]}")
        print(f"    MAPE:  {bl['mape']:.2f}% → {r_best['mape']:.2f}%  ({d_mape:+.2f}%)")
        print(f"    Dir:   {bl['dir']:.1f}%  → {r_best['dir']:.1f}%   ({d_dir:+.1f}pp)")
        print(f"    Cov:   {bl['cov']:.1f}%  → {r_best['cov']:.1f}%")
    else:
        print(f"\n  No group combination improves over GBM BTC+macro for {hz}d.")

print(f"\n{'='*76}")
print(f"  SUMMARY")
print(f"{'='*76}")
print("""
  7-DAY  model:
    - Regime cone remains hard to beat on MAPE and coverage
    - Check results above for any GBM configurations that improve direction

  14-DAY model:
    - More room for feature-based improvement (longer horizon = more signal time)
    - Crypto ecosystem, yield curve, FOMC proximity were expected to help

  Key signals to watch (from theory):
    - Yield curve inversion (spread_10y_3m < 0)  → macro risk-off → BTC pressure
    - FOMC week dummy  → elevated volatility, directional ambiguity
    - Copper/Gold ratio  → economic optimism proxy
    - Credit spread (HYG/LQD)  → risk appetite
    - EEM/SPX relative  → geopolitical risk-off
    - SOL/BNB momentum  → alt-season = BTC dominance declining
""")
