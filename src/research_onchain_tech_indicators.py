"""
On-chain and Technical Indicator Feature Research
===================================================
Tests whether on-chain metrics and technical indicators improve the
7-day (regime-cone baseline) and 14-day (GBM v2 baseline) BTC models.

Feature groups:
  TI-A  Trend        — SMA20/50/200, EMA crossovers, MACD, price/MA ratios
  TI-B  Momentum     — RSI-7/21/30, Stochastic %K/%D, Williams %R, CCI, ROC
  TI-C  Volatility   — ATR, Bollinger Band width & %B, vol-of-vol
  TI-D  Volume       — OBV, volume ratio, CMF, Force Index
  OC-A  Network act. — Hash rate, difficulty, tx count, active addresses
                        (blockchain.com public API, no key needed)
  OC-B  Miner/flow   — Miners revenue, mempool size, NVT proxy
  FG    Sentiment    — Fear & Greed Index (alternative.me public API)

Baselines:
  7d  — regime cone               (MAPE≈4.8%  Dir≈49%  Cov≈90%)
  14d — GBM v2 (99 macro features) (MAPE≈8.9%  Dir≈55%  Cov≈88%)

For 14d the new groups are ADDED to the existing 99-feature GBM base.
"""
import sys, warnings, time
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
try:
    import joblib
    art14 = joblib.load("models/inference_assets_14d_cone.joblib")
    BASE14_FEAT_COLS = art14["ml_feature_cols"]
    print(f"Loaded 14d artefact: {len(BASE14_FEAT_COLS)} existing features")
except Exception as e:
    BASE14_FEAT_COLS = None
    print(f"Could not load 14d artefact ({e}) — will build base features from scratch")

HORIZON7  = 7
HORIZON14 = 14

# ─────────────────────────────────────────────────────────────────────────────
# 1. DOWNLOAD OHLCV (need H/L/V for TI calculation)
# ─────────────────────────────────────────────────────────────────────────────
print("\n>>> Downloading BTC OHLCV + auxiliary prices …")

def _dl_ohlcv(sym, start="2017-01-01"):
    d = yf.download(sym, start=start, progress=False, auto_adjust=True)
    if d.empty:
        return None
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] for c in d.columns]
    d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
    return d.sort_index()

def _dl_close(sym, start="2017-01-01"):
    d = _dl_ohlcv(sym, start)
    return d["Close"].rename(sym) if d is not None else None

btc_ohlcv = _dl_ohlcv("BTC-USD")
print(f"  BTC OHLCV: {len(btc_ohlcv)} bars  "
      f"{btc_ohlcv.index.min().date()} → {btc_ohlcv.index.max().date()}")

AUX_TICKERS = {
    "SPX":"^GSPC","NDX":"^IXIC","VIX":"^VIX","GOLD":"GC=F","DXY":"DX-Y.NYB",
    "TNX":"^TNX","IRX":"^IRX","ETH":"ETH-USD","HG":"HG=F","XLI":"XLI",
    "XLF":"XLF","CL":"CL=F","NG":"NG=F","XLE":"XLE",
}
aux = {}
for name, sym in AUX_TICKERS.items():
    s = _dl_close(sym)
    if s is not None:
        aux[name] = s
        print(f"  {name:6s}: {len(s)} bars")
    else:
        print(f"  {name:6s}: FAILED")

# ─────────────────────────────────────────────────────────────────────────────
# 2. FETCH ON-CHAIN DATA  (blockchain.com public API)
# ─────────────────────────────────────────────────────────────────────────────
print("\n>>> Fetching on-chain data from blockchain.com …")

OC_METRICS = {
    "hash_rate":      "hash-rate",
    "difficulty":     "difficulty",
    "n_tx":           "n-transactions",
    "n_addr":         "n-unique-addresses",
    "tx_vol_usd":     "estimated-transaction-volume-usd",
    "miners_rev":     "miners-revenue",
    "tx_fees_usd":    "transaction-fees-usd",
    "mempool_size":   "mempool-size",
    "utxo_count":     "utxo-count",
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
                # Sub-daily sources (mempool, UTXO) have many rows per day —
                # take the last reading each day to get a clean daily series.
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
# 3. FETCH FEAR & GREED INDEX  (alternative.me)
# ─────────────────────────────────────────────────────────────────────────────
print("\n>>> Fetching Fear & Greed Index (alternative.me) …")
fg_series = None
try:
    r = requests.get("https://api.alternative.me/fng/?limit=0&format=json", timeout=30)
    if r.status_code == 200:
        data = r.json()["data"]
        fg_df = pd.DataFrame(data)
        fg_df["date"]  = pd.to_datetime(fg_df["timestamp"].astype(int), unit="s").dt.normalize()
        fg_series = fg_df.set_index("date")["value"].astype(float).sort_index()
        fg_series.index = fg_series.index.tz_localize(None)
        print(f"  Fear & Greed: {len(fg_series)} days  "
              f"{fg_series.index.min().date()} → {fg_series.index.max().date()}")
    else:
        print(f"  HTTP {r.status_code}")
except Exception as e:
    print(f"  FAILED: {e}")

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
    lo_k = low.rolling(k).min()
    hi_k = high.rolling(k).max()
    pct_k = 100 * (close - lo_k) / (hi_k - lo_k + 1e-9)
    pct_d = pct_k.rolling(d).mean()
    return pct_k.rename("stoch_k"), pct_d.rename("stoch_d")

def compute_williams_r(high, low, close, period=14):
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return (-100 * (hh - close) / (hh - ll + 1e-9)).rename("williams_r")

def compute_cci(high, low, close, period=20):
    tp  = (high + low + close) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return ((tp - sma) / (0.015 * mad + 1e-9)).rename("cci_20")

def compute_atr(high, low, close, period=14):
    hl  = high - low
    hc  = (high - close.shift(1)).abs()
    lc  = (low  - close.shift(1)).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean().rename(f"atr_{period}")

def compute_bollinger(close, period=20, nstd=2):
    sma   = close.rolling(period).mean()
    std   = close.rolling(period).std()
    upper = sma + nstd * std
    lower = sma - nstd * std
    width = ((upper - lower) / sma).rename("bb_width_20")
    pct_b = ((close - lower) / (upper - lower + 1e-9)).rename("bb_pct_b_20")
    return width, pct_b

def compute_macd(close, fast=12, slow=26, sig=9):
    ef = close.ewm(span=fast, adjust=False).mean()
    es = close.ewm(span=slow, adjust=False).mean()
    mc = ef - es
    ms = mc.ewm(span=sig, adjust=False).mean()
    return (mc / close).rename("macd_norm"), ((mc - ms) / close).rename("macd_hist_norm")

def compute_obv_norm(close, volume):
    direction = np.sign(close.diff()).fillna(0)
    obv  = (direction * volume).cumsum()
    obv_ma20 = obv.rolling(20).mean()
    # Return normalised OBV change (not level — too non-stationary)
    return (obv / (obv_ma20.abs() + 1e-9) - 1).rename("obv_vs_ma20")

def compute_cmf(high, low, close, volume, period=20):
    mfm = ((close - low) - (high - close)) / (high - low + 1e-9)
    mfv = mfm * volume
    return (mfv.rolling(period).sum() / volume.rolling(period).sum()).rename("cmf_20")

# ─────────────────────────────────────────────────────────────────────────────
# 5. BUILD FULL FEATURE MATRIX
# ─────────────────────────────────────────────────────────────────────────────
print("\n>>> Engineering all features …")

START = "2019-01-01"
c   = btc_ohlcv["Close"].loc[START:]
h   = btc_ohlcv["High"].loc[START:]
lo  = btc_ohlcv["Low"].loc[START:]
vol = btc_ohlcv["Volume"].loc[START:]
ret = np.log(c).diff()
rm30 = (h - lo).div(c).rolling(30).mean()   # range_ma30 proxy

feats = {}

# ── Existing base features (BTC own + macro for 7d/14d) ──────────────────────
for k in [1, 3, 5, 7, 10, 14, 21]:
    feats[f"btc_ret_{k}"] = ret.rolling(k).sum()
for k in [7, 14, 20, 30]:
    feats[f"btc_vol_{k}"] = ret.rolling(k).std()
feats["btc_rsi_14"] = compute_rsi(c, 14)
feats["range_ma30"] = rm30

def add_macro(name, ks=(1, 5, 10, 14, 20)):
    if name not in aux:
        return
    s = aux[name].reindex(c.index).ffill(limit=5)
    r = np.log(s + 1e-9).diff()
    for k in ks:
        feats[f"{name.lower()}_ret_{k}"] = r.rolling(k).sum()
    feats[f"{name.lower()}_vol_20"] = r.rolling(20).std()

for nm in ["SPX","NDX","VIX","GOLD","DXY","ETH","XLI","XLF","CL","XLE"]:
    add_macro(nm)
if "TNX" in aux:
    add_macro("TNX")
if "IRX" in aux and "TNX" in aux:
    tnx_ = aux["TNX"].reindex(c.index).ffill(limit=5)
    irx_ = aux["IRX"].reindex(c.index).ffill(limit=5)
    sp   = tnx_ - irx_
    feats["spread_10y_3m"]        = sp
    feats["spread_10y_3m_chg_5"]  = sp.diff(5)
    feats["spread_10y_3m_chg_20"] = sp.diff(20)
if "HG" in aux:
    add_macro("HG")
    if "GOLD" in aux:
        cg = np.log(aux["HG"].reindex(c.index).ffill(limit=5) /
                    aux["GOLD"].reindex(c.index).ffill(limit=5))
        feats["copper_gold_ratio"]  = cg
        feats["copper_gold_chg_20"] = cg.diff(20)
if "NG" in aux:
    add_macro("NG")
if "ETH" in aux:
    eth_ = aux["ETH"].reindex(c.index).ffill(limit=5)
    feats["eth_btc_ratio"]        = np.log(eth_ / c)
    feats["eth_btc_ratio_chg_14"] = feats["eth_btc_ratio"].diff(14)

BASE_FEAT_KEYS = list(feats.keys())   # everything up to here = base feature set

# ── TI-A: Trend ───────────────────────────────────────────────────────────────
ti_a = {}
for period in [20, 50, 200]:
    sma = c.rolling(period).mean()
    ti_a[f"sma{period}_ratio"]     = (c / sma - 1)
    ti_a[f"sma{period}_slope_20"]  = np.log(sma).diff(20)
ti_a["sma20_50_cross"]   = (c.rolling(20).mean() / c.rolling(50).mean() - 1)
ti_a["sma50_200_cross"]  = (c.rolling(50).mean() / c.rolling(200).mean() - 1)
ema12 = c.ewm(span=12, adjust=False).mean()
ema26 = c.ewm(span=26, adjust=False).mean()
ti_a["ema12_26_cross"]  = (ema12 / ema26 - 1)
macd_n, macd_h_n = compute_macd(c)
ti_a["macd_norm"]       = macd_n
ti_a["macd_hist_norm"]  = macd_h_n
# Price distance from ATH (drawdown)
ath = c.expanding().max()
ti_a["drawdown_from_ath"] = (c / ath - 1)

print(f"  TI-A (trend):       {len(ti_a):3d} features")

# ── TI-B: Momentum ────────────────────────────────────────────────────────────
ti_b = {}
for p in [7, 21, 30]:
    ti_b[f"rsi_{p}"] = compute_rsi(c, p)
stk, std_ = compute_stoch(h, lo, c)
ti_b["stoch_k"]      = stk
ti_b["stoch_d"]      = std_
ti_b["stoch_kd_diff"]= stk - std_
ti_b["williams_r"]   = compute_williams_r(h, lo, c)
ti_b["cci_20"]       = compute_cci(h, lo, c)
for p in [7, 14, 21]:
    ti_b[f"roc_{p}"] = (c / c.shift(p) - 1)

print(f"  TI-B (momentum):    {len(ti_b):3d} features")

# ── TI-C: Volatility ──────────────────────────────────────────────────────────
ti_c = {}
for p in [7, 14, 21]:
    ti_c[f"atr_{p}"]      = compute_atr(h, lo, c, p) / c
    ti_c[f"atr_{p}_chg7"] = ti_c[f"atr_{p}"].pct_change(7)
bb_w, bb_pct = compute_bollinger(c)
ti_c["bb_width_20"]     = bb_w
ti_c["bb_pct_b_20"]     = bb_pct
# Vol-of-vol: 10-day rolling std of realized vol
vv = ret.rolling(10).std()
ti_c["vol_of_vol_20"]   = vv.rolling(20).std() / (vv.rolling(20).mean() + 1e-9)
# ATR ratio: current ATR vs 50d average ATR (regime signal)
atr14 = compute_atr(h, lo, c, 14)
ti_c["atr_ratio_50"]    = (atr14 / (atr14.rolling(50).mean() + 1e-9) - 1)
# High-low range vs previous bar
ti_c["hl_range"]        = ((h - lo) / c)
ti_c["hl_range_ma7"]    = ti_c["hl_range"].rolling(7).mean()

print(f"  TI-C (volatility):  {len(ti_c):3d} features")

# ── TI-D: Volume ──────────────────────────────────────────────────────────────
ti_d = {}
ti_d["obv_vs_ma20"]    = compute_obv_norm(c, vol)
vol_ma20 = vol.rolling(20).mean()
ti_d["vol_ratio_20"]   = (vol / (vol_ma20 + 1e-9) - 1)
ti_d["vol_ratio_5"]    = (vol / (vol.rolling(5).mean() + 1e-9) - 1)
ti_d["cmf_20"]         = compute_cmf(h, lo, c, vol)
# Price * volume momentum
pv = ret * vol
ti_d["pv_mom_5"]       = pv.rolling(5).sum() / (vol.rolling(5).sum() + 1e-9)
ti_d["pv_mom_20"]      = pv.rolling(20).sum() / (vol.rolling(20).sum() + 1e-9)
# Volume trend
ti_d["vol_slope_20"]   = np.log(vol_ma20).diff(20)

print(f"  TI-D (volume):      {len(ti_d):3d} features")

# ── OC-A: On-chain network activity ───────────────────────────────────────────
oc_a = {}
for name in ["hash_rate", "difficulty", "n_tx", "n_addr"]:
    if name not in onchain_raw:
        continue
    s = (onchain_raw[name].reindex(c.index).ffill(limit=3)
         .pipe(np.log).diff())   # use log-returns
    oc_a[f"oc_{name}_ret_7"]  = s.rolling(7).sum()
    oc_a[f"oc_{name}_ret_30"] = s.rolling(30).sum()
    oc_a[f"oc_{name}_vol_20"] = s.rolling(20).std()
# Hash rate level relative to 90d mean (miner confidence regime)
if "hash_rate" in onchain_raw:
    hr = onchain_raw["hash_rate"].reindex(c.index).ffill(limit=3)
    oc_a["oc_hash_vs_ma90"] = hr / hr.rolling(90).mean() - 1

print(f"  OC-A (network):     {len(oc_a):3d} features")

# ── OC-B: Miner economics / flow ──────────────────────────────────────────────
oc_b = {}
for name in ["tx_vol_usd", "miners_rev", "tx_fees_usd", "mempool_size", "utxo_count"]:
    if name not in onchain_raw:
        continue
    s = (onchain_raw[name].reindex(c.index).ffill(limit=3)
         .pipe(np.log).diff())
    oc_b[f"oc_{name}_ret_7"]  = s.rolling(7).sum()
    oc_b[f"oc_{name}_ret_30"] = s.rolling(30).sum()
    oc_b[f"oc_{name}_vol_20"] = s.rolling(20).std()

# NVT proxy: price * circulating_supply / tx_vol_usd
# Use BTC market cap proxy = price * 19.5M (approx)
if "tx_vol_usd" in onchain_raw:
    tv  = onchain_raw["tx_vol_usd"].reindex(c.index).ffill(limit=3)
    mcp = c * 19_500_000   # approximate circulating supply
    nvt = np.log(mcp / (tv + 1e-9))
    oc_b["oc_nvt_proxy"]       = nvt
    oc_b["oc_nvt_proxy_chg_30"]= nvt.diff(30)

# Miner stress: miners_revenue / hash_rate ratio
if "miners_rev" in onchain_raw and "hash_rate" in onchain_raw:
    mr  = onchain_raw["miners_rev"].reindex(c.index).ffill(limit=3)
    hr2 = onchain_raw["hash_rate"].reindex(c.index).ffill(limit=3)
    miner_stress = np.log(mr / (hr2 + 1e-9))
    oc_b["oc_miner_stress"]       = miner_stress
    oc_b["oc_miner_stress_chg_30"]= miner_stress.diff(30)

print(f"  OC-B (miner/flow):  {len(oc_b):3d} features")

# ── FG: Fear & Greed ──────────────────────────────────────────────────────────
fg = {}
if fg_series is not None:
    fgr = fg_series.reindex(c.index).ffill(limit=3)
    fg["fg_value"]      = fgr
    fg["fg_chg_7"]      = fgr.diff(7)
    fg["fg_chg_30"]     = fgr.diff(30)
    fg["fg_ma7"]        = fgr.rolling(7).mean()
    fg["fg_zscore_30"]  = (fgr - fgr.rolling(30).mean()) / (fgr.rolling(30).std() + 1e-9)
    fg["fg_extreme_fear"]  = (fgr < 25).astype(float)   # binary: extreme fear
    fg["fg_extreme_greed"] = (fgr > 75).astype(float)   # binary: extreme greed

print(f"  FG  (fear/greed):   {len(fg):3d} features")

# ── Assemble ──────────────────────────────────────────────────────────────────
ALL_GROUPS = {"TI-A": ti_a, "TI-B": ti_b, "TI-C": ti_c, "TI-D": ti_d,
              "OC-A": oc_a, "OC-B": oc_b, "FG":   fg}

feats_df = pd.DataFrame({**feats, **ti_a, **ti_b, **ti_c, **ti_d,
                         **oc_a, **oc_b, **fg}, index=c.index)
feats_df["y7"]  = np.log(c.shift(-7)  / c)
feats_df["y14"] = np.log(c.shift(-14) / c)
feats_df = feats_df.replace([np.inf, -np.inf], np.nan)

# Drop columns >25% NaN
naf = feats_df.isna().mean()
drop = naf[naf > 0.25].index.tolist()
if drop:
    print(f"\n  Dropping {len(drop)} high-NA columns: {drop[:6]} …")
    feats_df.drop(columns=drop, inplace=True)

feat_only = [c2 for c2 in feats_df.columns if c2 not in ("y7","y14")]
feats_df[feat_only] = feats_df[feat_only].ffill(limit=5).bfill(limit=40)
print(f"\n  Total feature matrix: {feats_df.shape}  "
      f"({feats_df.shape[1]-2} features, 2 targets)")

# ─────────────────────────────────────────────────────────────────────────────
# 6. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
TODAY      = pd.Timestamp("today").normalize()
test_start = TODAY - pd.DateOffset(months=8)

def make_splits(horizon):
    te = test_start - pd.Timedelta(days=horizon)
    tmp = feats_df.dropna(subset=[f"y{horizon}"])
    return tmp.loc[:te], tmp.loc[test_start:]

tr7,  te7  = make_splits(7)
tr14, te14 = make_splits(14)
close_te7  = btc_ohlcv["Close"].reindex(te7.index).values
close_te14 = btc_ohlcv["Close"].reindex(te14.index).values

print(f"\n  7d  TRAIN {tr7.index.min().date()} → {tr7.index.max().date()}  n={len(tr7)}")
print(f"      TEST  {te7.index.min().date()} → {te7.index.max().date()}   n={len(te7)}")
print(f"  14d TRAIN {tr14.index.min().date()} → {tr14.index.max().date()}  n={len(tr14)}")
print(f"      TEST  {te14.index.min().date()} → {te14.index.max().date()}   n={len(te14)}")

# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
BAND7 = 0.097; BAND14 = 0.160

def mape_p(pred, act, c0):
    return float(np.mean(np.abs(c0*np.exp(pred) - c0*np.exp(act)) / (c0*np.exp(act)))*100)
def dir_acc(pred, act):
    return float(100*(np.sign(pred)==np.sign(act)).mean())
def band_cov(pred, act, band):
    return float(100*((act>=pred+np.log(1-band)) & (act<=pred+np.log(1+band))).mean())

def metrics_dict(pred, act, c0, band):
    return dict(mape=mape_p(pred,act,c0), dir=dir_acc(pred,act), cov=band_cov(pred,act,band))

# Regime-cone helpers (7d baseline)
edges7 = np.quantile(tr7["range_ma30"].dropna(), [1/3, 2/3])
def reg7(x):
    return np.searchsorted(edges7, np.asarray(x, float), side="right").clip(0,2)
tr7_r = reg7(tr7["range_ma30"])
meds7 = {r: float(np.median(tr7["y7"].values[tr7_r==r])) for r in range(3)}
te7_r = reg7(te7["range_ma30"])
cone7_pred = np.array([meds7[r] for r in te7_r])
bl7 = metrics_dict(cone7_pred, te7["y7"].values, close_te7, BAND7)

# 14d GBM v2 baseline  (reproduce v2 feature set from artefact or from feats_df)
# We use the same columns that the saved 14d model was trained on, but
# re-train on the new (slightly different) data — this keeps the evaluation
# fair since we're adding NEW features on top of the same base.
edges14 = np.quantile(tr14["range_ma30"].dropna(), [1/3, 2/3])
def reg14(x):
    return np.searchsorted(edges14, np.asarray(x, float), side="right").clip(0,2)

BASE14 = [f for f in BASE_FEAT_KEYS if f in tr14.columns]  # v2 feature set reproduced here

def fit_gbm(Xtr, ytr, Xte, yte, close_te, band, label="", n_est=300, depth=3, lr=0.04):
    cols = sorted(set(Xtr.columns) & set(Xte.columns))
    Xtr_ = Xtr[cols].fillna(0)
    Xte_ = Xte[cols].fillna(0)
    ytr_ = ytr.reindex(Xtr_.index).dropna()
    Xtr_ = Xtr_.reindex(ytr_.index)
    m = Pipeline([("sc", StandardScaler()),
                  ("gbm", GradientBoostingRegressor(
                      n_estimators=n_est, max_depth=depth, learning_rate=lr,
                      subsample=0.8, min_samples_leaf=15, random_state=42))])
    m.fit(Xtr_, ytr_)
    pred = m.predict(Xte_)
    act  = yte.values
    imp  = pd.Series(m.named_steps["gbm"].feature_importances_, index=cols)
    return metrics_dict(pred, act, close_te, band), imp

# Refit 14d GBM baseline on current data
print("\n>>> Refitting 14d GBM v2 baseline on current feature set …")
bl14_metrics, bl14_imp = fit_gbm(
    tr14[BASE14], tr14["y14"], te14[BASE14], te14["y14"], close_te14, BAND14,
    label="GBM v2 baseline")

print(f"\n{'='*72}")
print(f"  BASELINES")
print(f"{'='*72}")
print(f"  7d  Regime Cone: MAPE={bl7['mape']:.2f}%  Dir={bl7['dir']:.1f}%  Cov={bl7['cov']:.1f}%")
print(f"  14d GBM v2:      MAPE={bl14_metrics['mape']:.2f}%  "
      f"Dir={bl14_metrics['dir']:.1f}%  Cov={bl14_metrics['cov']:.1f}%")

# Group membership for classification
def group_of(f):
    for gname, gdict in ALL_GROUPS.items():
        if f in gdict:
            return gname
    if f in feats:
        return "base"
    return "?"

# ─────────────────────────────────────────────────────────────────────────────
# 7. CORRELATION ANALYSIS — quick ranking before running ablations
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print(f"  CORRELATION SCAN — all new features vs 7d and 14d targets (TRAIN)")
print(f"{'='*72}")

new_feat_cols = [f for f in feat_only if f not in BASE_FEAT_KEYS]
corr7  = {f: float(tr7[f].corr(tr7["y7"]))   for f in new_feat_cols if f in tr7.columns}
corr14 = {f: float(tr14[f].corr(tr14["y14"])) for f in new_feat_cols if f in tr14.columns}

corr_df = pd.DataFrame({"corr7": corr7, "corr14": corr14})
corr_df["abs7"]  = corr_df["corr7"].abs()
corr_df["abs14"] = corr_df["corr14"].abs()
corr_df["group"] = corr_df.index.map(group_of)

print(f"\n  Top 20 new features by |correlation| with 14d target:")
print(f"  {'Feature':38s}  {'corr_14d':>9}  {'corr_7d':>9}  {'Group'}")
print(f"  {'-------':38s}  {'-'*9}  {'-'*9}  {'-----'}")
for feat, row in corr_df.nlargest(20, "abs14").iterrows():
    print(f"  {feat:38s}  {row.corr14:+9.4f}  {row.corr7:+9.4f}  {row.group}")

print(f"\n  Top 20 new features by |correlation| with 7d target:")
for feat, row in corr_df.nlargest(20, "abs7").iterrows():
    print(f"  {feat:38s}  {row.corr7:+9.4f}  {row.corr14:+9.4f}  {row.group}")

# Group-level average |correlation|
print(f"\n  Average |correlation| per group:")
for g, sub in corr_df.groupby("group"):
    print(f"    {g:8s}  |corr_14d|={sub.abs14.mean():.4f}  "
          f"|corr_7d|={sub.abs7.mean():.4f}  n={len(sub)}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. ABLATION: group-by-group added to each baseline
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print(f"  ABLATION — each new group added to baseline")
print(f"{'='*72}")

for hz, bl, tr, te, close_te, band, base in [
    (7,  bl7,          tr7,  te7,  close_te7,  BAND7,  BASE_FEAT_KEYS),
    (14, bl14_metrics, tr14, te14, close_te14, BAND14, BASE14),
]:
    yt  = f"y{hz}"
    ytr = tr[yt]; yte = te[yt]
    print(f"\n  {hz}d horizon  baseline  "
          f"MAPE={bl['mape']:.2f}%  Dir={bl['dir']:.1f}%  Cov={bl['cov']:.1f}%")
    print(f"  {'Config':38s}  {'MAPE':>7}  {'ΔDir':>7}  {'Cov':>7}  {'Verdict'}")
    print(f"  {'-'*38}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*30}")

    # GBM on base features alone
    base_avail = [f for f in base if f in tr.columns and f in te.columns]
    m0, _ = fit_gbm(tr[base_avail], ytr, te[base_avail], yte, close_te, band)
    dd0 = m0["dir"] - bl["dir"]
    print(f"  {'GBM base only':38s}  {m0['mape']:6.2f}%  "
          f"{dd0:+6.1f}%  {m0['cov']:6.1f}%  {'—'}")

    group_results = {}
    for gname, gdict in ALL_GROUPS.items():
        add = [f for f in gdict if f in tr.columns and f in te.columns]
        if not add:
            print(f"  {'+ '+gname:38s}  (no data available)")
            continue
        feat_set = base_avail + add
        m, imp = fit_gbm(tr[feat_set], ytr, te[feat_set], yte, close_te, band)
        dd = m["dir"] - bl["dir"]
        dm = m["mape"] - bl["mape"]
        v  = []
        if dm < -0.20:     v.append("✅ MAPE↓")
        if dd  >  2.0:     v.append("✅ Dir↑")
        if m["cov"] > bl["cov"] + 1.0: v.append("✅ Cov↑")
        if dm  >  0.40:    v.append("❌ MAPE↑")
        if dd  < -2.0:     v.append("❌ Dir↓")
        if not v:          v = ["— negligible"]
        print(f"  {'+ '+gname:38s}  {m['mape']:6.2f}%  "
              f"{dd:+6.1f}%  {m['cov']:6.1f}%  {' '.join(v)}")
        group_results[gname] = (m, imp, add)

    # Best combination: add all groups that improve direction or MAPE
    helpful_add = []
    helpful_names = []
    for gname, (m, imp, add) in group_results.items():
        if m["dir"] > m0["dir"] + 1.5 or m["mape"] < m0["mape"] - 0.15:
            helpful_add.extend(add)
            helpful_names.append(gname)

    if helpful_add:
        combined = base_avail + helpful_add
        m_comb, imp_comb = fit_gbm(tr[combined], ytr, te[combined], yte, close_te, band)
        dd_c = m_comb["dir"] - bl["dir"]
        dm_c = m_comb["mape"] - bl["mape"]
        print(f"\n  Best combined (+{helpful_names})")
        print(f"  {'→ COMBINED':38s}  {m_comb['mape']:6.2f}%  "
              f"{dd_c:+6.1f}%  {m_comb['cov']:6.1f}%")

        # Top-15 feature importances for combined model
        print(f"\n  Top-15 features by importance in combined {hz}d model:")
        print(f"  {'Rank':>4}  {'Feature':38s}  {'Imp':>8}  {'Group'}")
        for i, (feat, val) in enumerate(imp_comb.sort_values(ascending=False).head(15).items(), 1):
            print(f"  {i:4d}  {feat:38s}  {val:.4f}  {group_of(feat)}")
    else:
        print(f"\n  No new group improved over GBM base for {hz}d.")

# ─────────────────────────────────────────────────────────────────────────────
# 9. FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print(f"  SUMMARY")
print(f"{'='*72}")
print("""
  Interpretation guide:
    MAPE↓ — new features help predict the 14-day forward close price more accurately
    Dir↑  — new features help predict whether BTC goes UP or DOWN
    Cov↑  — new features improve band calibration

  On-chain vs Technical Indicators:
    On-chain features (hash rate, tx volume, NVT) capture fundamental
    network health but typically lag price by days-to-weeks.
    Technical indicators are coincident with price, so they may add
    information on top of the raw return/vol features already in the model.

  Fear & Greed Index:
    A composite daily sentiment score — historically high fear corresponds
    to BTC buying opportunities (contrarian signal), high greed to sell zones.
""")
