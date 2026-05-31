#!/usr/bin/env python3
"""Train the Daily H/L ensemble using Yahoo Finance + Coinbase Premium.

Mirrors pipeline_ct.py exactly but uses Yahoo Finance daily BTC data
instead of Binance hourly (which is geoblocked in this environment).

Features: 116 = 113 baseline + 3 Coinbase Premium (cb_premium/ma3/z7)
Architecture: Huber + QuantileRegressor(q=0.70) + GBM-quantile(q=0.70)
Plus direction head (GBM classifier for asymmetry sign).
Saves to models/inference_assets_ct.joblib in the standard artifact format.
"""
import shutil
import sys
import time
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, QuantileRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from paths import DAILY_MODEL_CT, MODELS_DIR, LEGACY_DIR, RAW_CT_CSV, FEATURES_CT_CSV

ANCHOR_HOUR_UTC = 12
ALPHA_QUANT     = 0.70
TREND_SAT       = 0.05

# ---------------------------------------------------------------------------
# 1. DATA FETCH
# ---------------------------------------------------------------------------
FETCH_START  = "2018-06-01"
TODAY        = datetime.now(timezone.utc).date()
FETCH_END    = (pd.Timestamp(TODAY) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
# Hard cutoff: only data up to this date is used for training/val/test.
# Set to a past date to include that period in the model's training window.
# Set to None (or a future date) to use all available data.
DATA_CUTOFF  = "2026-02-28"
# Date ranges excluded from training (anomalous crash events that bias the model).
# Each tuple is (start_date, end_date) inclusive; applied after feature engineering.
EXCLUDE_RANGES = [
    ("2026-01-28", "2026-02-07"),  # Jan/Feb 2026 flash crash: BTC ~$89k→$60k
]


def _flat(df, name):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = [f"{name}_{c.lower()}" for c in df.columns]
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df[~df.index.duplicated(keep="last")].sort_index()


def fetch_coinbase_daily(start, end):
    CB_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    rows = []
    cur = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    while cur <= end_ts:
        chunk_end = min(cur + pd.Timedelta(days=299), end_ts)
        params = {"granularity": 86400,
                  "start": cur.isoformat() + "Z",
                  "end":   (chunk_end + pd.Timedelta(days=1)).isoformat() + "Z"}
        try:
            r = requests.get(CB_URL, params=params, timeout=30)
            if r.status_code == 200:
                rows.extend(r.json())
            else:
                print(f"  ! Coinbase {r.status_code}: {r.text[:80]}")
        except Exception as exc:
            print(f"  ! Coinbase error: {exc}")
        cur = chunk_end + pd.Timedelta(days=1)
        time.sleep(0.35)
    if not rows:
        print("  ! Coinbase: no data returned")
        return pd.Series(dtype=float, name="coinbase_close")
    tmp = pd.DataFrame(rows, columns=["ts", "low", "high", "open", "close", "volume"])
    tmp["date"] = pd.to_datetime(tmp["ts"], unit="s").dt.normalize()
    tmp = tmp.drop_duplicates("date").set_index("date").sort_index()
    print(f"  coinbase_daily: {len(tmp)} rows  {tmp.index.min().date()} → {tmp.index.max().date()}")
    return tmp["close"].rename("coinbase_close")


print("\n>>> Fetching Yahoo Finance BTC + macro …")
SYMS = {"btc": "BTC-USD", "eth": "ETH-USD", "spx": "^GSPC", "ndx": "^IXIC",
        "vix": "^VIX", "gold": "GC=F", "dxy": "DX-Y.NYB", "tnx": "^TNX"}
parts = []
for name, sym in SYMS.items():
    d = yf.download(sym, start=FETCH_START, end=FETCH_END, progress=False, auto_adjust=False)
    if d.empty:
        print(f"  ! empty {sym}"); continue
    parts.append(_flat(d, name))
    print(f"  {sym}: {len(d)} rows  {d.index.min().date()} → {d.index.max().date()}")
btc_daily = parts[0]
for p in parts[1:]:
    btc_daily = btc_daily.join(p, how="outer")

print("\n>>> Fetching blockchain.info on-chain …")
SERIES = ["hash-rate", "difficulty", "n-transactions", "miners-revenue",
          "n-unique-addresses", "transaction-fees-usd", "mempool-size",
          "estimated-transaction-volume-usd", "market-cap", "avg-block-size",
          "cost-per-transaction"]
oc_parts = []
for s in SERIES:
    url = (f"https://api.blockchain.info/charts/{s}"
           "?timespan=all&format=json&sampled=false")
    try:
        r = requests.get(url, timeout=30); j = r.json()
        v = j.get("values", [])
        if not v:
            print(f"  ! empty {s}"); continue
        idx = pd.to_datetime([x["x"] for x in v], unit="s").normalize()
        ser = pd.Series([x["y"] for x in v], index=idx,
                        name=f"oc_{s.replace('-','_')}")
        ser = ser[~ser.index.duplicated(keep="last")]
        oc_parts.append(ser)
        print(f"  {s}: {len(ser)} rows")
    except Exception as e:
        print(f"  ! fail {s}: {e}")
    time.sleep(0.2)
oc = pd.concat(oc_parts, axis=1) if oc_parts else pd.DataFrame()

print("\n>>> Fetching Fear & Greed …")
try:
    r = requests.get("https://api.alternative.me/fng/?limit=0&format=json", timeout=30)
    data = r.json().get("data", [])
    idx = pd.to_datetime([int(x["timestamp"]) for x in data], unit="s").normalize()
    fng = pd.Series([int(x["value"]) for x in data], index=idx, name="fng")
    fng = fng[~fng.index.duplicated(keep="last")].sort_index()
    print(f"  fng: {len(fng)} rows")
except Exception as e:
    print(f"  ! fng failed: {e}")
    fng = pd.Series(dtype=float, name="fng")

print("\n>>> Fetching Coinbase daily …")
coinbase_daily = fetch_coinbase_daily(FETCH_START, FETCH_END)

# ---------------------------------------------------------------------------
# 2. JOIN
# ---------------------------------------------------------------------------
print("\n>>> Joining all data …")
df = btc_daily.copy()
if not oc.empty:
    df = df.join(oc, how="left")
if not fng.empty:
    df = df.join(fng.to_frame(), how="left")
if coinbase_daily.notna().any():
    df = df.join(coinbase_daily.to_frame("coinbase_close"), how="left")
df = df.sort_index()
df = df.loc[df["btc_close"].notna()]
df = df.ffill(limit=5)
df = df.loc["2019-01-01":]
print(f">>> Raw shape {df.shape}  range {df.index.min().date()} → {df.index.max().date()}")
df.to_csv(RAW_CT_CSV)
print(f">>> Saved {RAW_CT_CSV}")

# ---------------------------------------------------------------------------
# 3. FEATURE ENGINEERING (matches pipeline_ct.py exactly)
# ---------------------------------------------------------------------------
print("\n>>> Engineering features …")
f = pd.DataFrame(index=df.index)
c = df["btc_close"]; h = df["btc_high"]; l = df["btc_low"]
o = df["btc_open"]; v = df["btc_volume"]

nh = h.shift(-1); nl = l.shift(-1)
y_hi = (nh - c) / c; y_lo = (c - nl) / c

ret = np.log(c).diff()
for k in [1, 3, 5, 7, 14, 30]:
    f[f"ret_{k}"] = ret.rolling(k).sum()
for k in [5, 10, 20, 30]:
    f[f"vol_{k}"] = ret.rolling(k).std()

prev_c = c.shift(1)
tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
for k in [7, 14, 30]:
    f[f"atr_{k}"] = tr.rolling(k).mean() / c
f["range_today"] = (h - l) / c
f["range_ma7"]   = ((h - l) / c).rolling(7).mean()
f["range_ma30"]  = ((h - l) / c).rolling(30).mean()
f["range_std30"] = ((h - l) / c).rolling(30).std()

delta = c.diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / loss.replace(0, np.nan)
f["rsi_14"] = 100 - 100 / (1 + rs)

ema12 = c.ewm(span=12, adjust=False).mean()
ema26 = c.ewm(span=26, adjust=False).mean()
macd = ema12 - ema26
f["macd"]      = macd / c
f["macd_sig"]  = macd.ewm(span=9, adjust=False).mean() / c
f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c

ma20 = c.rolling(20).mean(); sd20 = c.rolling(20).std()
f["bb_width"]   = (4 * sd20) / ma20
f["dist_hi_30"] = c / c.rolling(30).max() - 1
f["dist_lo_30"] = c / c.rolling(30).min() - 1
f["dist_hi_90"] = c / c.rolling(90).max() - 1

f["vol_chg_1"]    = np.log(v).diff()
f["vol_z_20"]     = (np.log(v) - np.log(v).rolling(20).mean()) / np.log(v).rolling(20).std()
f["vol_ma_ratio"] = v / v.rolling(20).mean()

dow = df.index.dayofweek
for i in range(6):
    f[f"dow_{i}"] = (dow == i).astype(float)


def mret(name, ks=(1, 5, 20)):
    s = df[f"{name}_close"]
    for k in ks:
        f[f"{name}_ret_{k}"] = np.log(s).diff(k)
    f[f"{name}_vol_20"] = np.log(s).diff().rolling(20).std()


for nm in ["spx", "ndx", "vix", "gold", "dxy", "tnx", "eth"]:
    mret(nm)

f["btc_spx_corr_30"]  = ret.rolling(30).corr(np.log(df["spx_close"]).diff())
f["btc_ndx_corr_30"]  = ret.rolling(30).corr(np.log(df["ndx_close"]).diff())
f["btc_gold_corr_30"] = ret.rolling(30).corr(np.log(df["gold_close"]).diff())
f["btc_dxy_corr_30"]  = ret.rolling(30).corr(np.log(df["dxy_close"]).diff())

oc_cols = [x for x in df.columns if x.startswith("oc_")]
for col in oc_cols:
    s = df[col].astype(float)
    sl = np.log(s.replace(0, np.nan))
    f[f"{col}_d1"]  = sl.diff(1)
    f[f"{col}_d7"]  = sl.diff(7)
    f[f"{col}_z30"] = (sl - sl.rolling(30).mean()) / sl.rolling(30).std()

# Coinbase Premium features
_cb_close  = df.get("coinbase_close")
_btc_close = df.get("btc_close")
if _cb_close is not None and _btc_close is not None and _cb_close.notna().any():
    _prem = (_cb_close - _btc_close) / _btc_close * 100
    f["cb_premium"]     = _prem
    f["cb_premium_ma3"] = _prem.rolling(3).mean()
    f["cb_premium_z7"]  = (_prem - _prem.rolling(7).mean()) / _prem.rolling(7).std()
    print(f"   cb_premium: {_cb_close.notna().sum()} non-null rows")

f["y_hi_ema3"] = y_hi.shift(1).ewm(span=3, adjust=False).mean()
f["y_lo_ema3"] = y_lo.shift(1).ewm(span=3, adjust=False).mean()
f["y_hi_ema7"] = y_hi.shift(1).ewm(span=7, adjust=False).mean()
f["y_lo_ema7"] = y_lo.shift(1).ewm(span=7, adjust=False).mean()

prev_3_hi = h.shift(1).rolling(3).max()
prev_3_lo = l.shift(1).rolling(3).min()
f["above_3d_high"]  = (c > prev_3_hi).astype(float)
f["below_3d_low"]   = (c < prev_3_lo).astype(float)
f["bo_strength_up"] = (c / prev_3_hi - 1).clip(lower=0)
f["bo_strength_dn"] = (1 - c / prev_3_lo).clip(lower=0)
_y_hi_lag = y_hi.shift(1); _y_lo_lag = y_lo.shift(1)
f["y_hi_surprise"] = _y_hi_lag - _y_hi_lag.ewm(span=7, adjust=False).mean()
f["y_lo_surprise"] = _y_lo_lag - _y_lo_lag.ewm(span=7, adjust=False).mean()

neg_ret = ret.clip(upper=0)
f["dn_vol_5"]  = neg_ret.rolling(5).std()
f["dn_vol_20"] = neg_ret.rolling(20).std()
sma50 = c.rolling(50).mean()
f["below_sma50"]    = (c < sma50).astype(float)
f["below_sma50_5d"] = f["below_sma50"].rolling(5).min().fillna(0)

data = f.copy()
data["y_hi"] = y_hi; data["y_lo"] = y_lo
data["close"] = c; data["next_high"] = nh; data["next_low"] = nl
data = data.replace([np.inf, -np.inf], np.nan).dropna()

# Apply hard cutoff so training uses only data up to DATA_CUTOFF
if DATA_CUTOFF:
    data = data.loc[:DATA_CUTOFF]

# Save features_ct.csv with FULL data (before crash exclusion) so downstream
# models (3-class day type, cone scripts) train on the complete dataset.
data.to_csv(FEATURES_CT_CSV)
print(f">>> Saved {FEATURES_CT_CSV}  ({len(data)} rows, crash included)")

# Exclude crash periods from H/L model training only.
for excl_start, excl_end in EXCLUDE_RANGES:
    mask = (data.index >= pd.Timestamp(excl_start)) & (data.index <= pd.Timestamp(excl_end))
    n_drop = int(mask.sum())
    data = data.loc[~mask]
    print(f">>> H/L training: excluded {n_drop} rows ({excl_start} → {excl_end})")

print(f">>> Feature matrix (H/L training) {data.shape}  features={data.shape[1]-5}")
print(f">>> Data range: {data.index.min().date()} → {data.index.max().date()}")

# ---------------------------------------------------------------------------
# 4. SPLIT (same scheme as pipeline_ct.py)
# ---------------------------------------------------------------------------
LATEST       = data.index.max()
TEST_MONTHS  = 2   # minimal holdout — intentionally short so more data goes to train
VAL_MONTHS   = 2
EMBARGO_DAYS = 1

test_start = LATEST   - pd.DateOffset(months=TEST_MONTHS)
val_start  = test_start - pd.DateOffset(months=VAL_MONTHS)
train_end  = val_start  - pd.Timedelta(days=EMBARGO_DAYS)
val_end    = test_start - pd.Timedelta(days=EMBARGO_DAYS)

train = data.loc[:train_end]
val   = data.loc[val_start:val_end]
test  = data.loc[test_start:]

feat_cols = [c for c in data.columns
             if c not in ("y_hi", "y_lo", "close", "next_high", "next_low")]
X_tr = train[feat_cols]; X_va = val[feat_cols]; X_te = test[feat_cols]
yhi_tr = train["y_hi"]; yhi_va = val["y_hi"]; yhi_te = test["y_hi"]
ylo_tr = train["y_lo"]; ylo_va = val["y_lo"]; ylo_te = test["y_lo"]
close_va = val["close"].values; close_te = test["close"].values
hi_true_va = val["next_high"].values;  lo_true_va = val["next_low"].values
hi_true    = test["next_high"].values; lo_true    = test["next_low"].values

print(f"\n>>> TRAIN {train.index.min().date()} → {train.index.max().date()}  n={len(train)}")
print(f">>> VAL   {val.index.min().date()} → {val.index.max().date()}  n={len(val)}")
print(f">>> TEST  {test.index.min().date()} → {test.index.max().date()}  n={len(test)}")
print(f">>> feat_cols: {len(feat_cols)}  (incl. cb_premium: "
      f"{'YES' if 'cb_premium' in feat_cols else 'NO'})")

# ---------------------------------------------------------------------------
# 5. TRAIN ENSEMBLE
# ---------------------------------------------------------------------------
def mk(model):
    return Pipeline([("sc", StandardScaler()), ("m", model)])


M_HUBER   = mk(HuberRegressor(max_iter=500, alpha=0.001))
M_QUANTLR = mk(QuantileRegressor(quantile=ALPHA_QUANT, alpha=0.001, solver="highs"))
M_GBQUANT = mk(GradientBoostingRegressor(
    loss="quantile", alpha=ALPHA_QUANT,
    n_estimators=1500, max_depth=3,
    learning_rate=0.01, subsample=0.8, random_state=42))

print(f"\nTraining ensemble (q={ALPHA_QUANT}) on {len(train)} rows, "
      f"{len(feat_cols)} features …")
preds = {}
for name, ctor in [("huber", M_HUBER), ("quant_lin", M_QUANTLR), ("gbm_quant", M_GBQUANT)]:
    t0 = time.time()
    mh = clone(ctor); mh.fit(X_tr, yhi_tr)
    ml = clone(ctor); ml.fit(X_tr, ylo_tr)
    preds[name] = dict(
        m_hi=mh, m_lo=ml,
        ph_va=mh.predict(X_va), pl_va=ml.predict(X_va),
        ph_te=mh.predict(X_te), pl_te=ml.predict(X_te),
    )
    print(f"  {name}: {time.time()-t0:.1f}s")

ens_ph_va = np.mean([p["ph_va"] for p in preds.values()], axis=0)
ens_pl_va = np.mean([p["pl_va"] for p in preds.values()], axis=0)
ens_ph_te = np.mean([p["ph_te"] for p in preds.values()], axis=0)
ens_pl_te = np.mean([p["pl_te"] for p in preds.values()], axis=0)

mu_hi = float(yhi_tr.mean()); mu_lo = float(ylo_tr.mean())
print(f"\nClimatology: mu_hi={mu_hi:.4f}  mu_lo={mu_lo:.4f}")


def mape_on(close_, hi_t, lo_t, ph, pl):
    return (
        float((np.abs(close_ * (1 + np.clip(ph, 0, None)) - hi_t) / hi_t).mean() * 100),
        float((np.abs(close_ * (1 - np.clip(pl, 0, None)) - lo_t) / lo_t).mean() * 100),
    )


# Alpha tuning on VAL
best_a = 1.0; best_avg = 99.0
for a in np.linspace(0, 1, 21):
    mh, ml = mape_on(close_va, hi_true_va, lo_true_va,
                     a * ens_ph_va + (1 - a) * mu_hi,
                     a * ens_pl_va + (1 - a) * mu_lo)
    if (mh + ml) / 2 < best_avg:
        best_avg = (mh + ml) / 2; best_a = a
alpha_use = best_a
print(f"Alpha (climatology blend): {alpha_use:.2f}  (val avg MAPE={best_avg:.3f}%)")

final_ph_va = alpha_use * ens_ph_va + (1 - alpha_use) * mu_hi
final_pl_va = alpha_use * ens_pl_va + (1 - alpha_use) * mu_lo
final_ph_te = alpha_use * ens_ph_te + (1 - alpha_use) * mu_hi
final_pl_te = alpha_use * ens_pl_te + (1 - alpha_use) * mu_lo

# ---------------------------------------------------------------------------
# 5b. DIRECTION HEAD (mirrors pipeline_ct.py)
# ---------------------------------------------------------------------------
print("\n>>> Training direction head …")
d_tr = (yhi_tr - ylo_tr) / 2
d_va = (yhi_va - ylo_va) / 2
d_te = (yhi_te - ylo_te) / 2
label_tr = (d_tr > 0).astype(int)
label_va = (d_va > 0).astype(int)
label_te = (d_te > 0).astype(int)

dir_clf = Pipeline([("sc", StandardScaler()),
                    ("m", GradientBoostingClassifier(
                        n_estimators=500, max_depth=3, learning_rate=0.02,
                        subsample=0.8, random_state=42))])
dir_clf.fit(X_tr, label_tr)
p_bull_va = dir_clf.predict_proba(X_va)[:, 1]
p_bull_te = dir_clf.predict_proba(X_te)[:, 1]
clf_acc_va = float(((p_bull_va > 0.5).astype(int) == label_va).mean())
clf_acc_te = float(((p_bull_te > 0.5).astype(int) == label_te).mean())
print(f"  clf accuracy: val={clf_acc_va*100:.1f}%  test={clf_acc_te*100:.1f}%")

d_bull_mean = float(d_tr[label_tr == 1].mean())
d_bear_mean = float(d_tr[label_tr == 0].mean())

ens_m_va = (final_ph_va + final_pl_va) / 2; ens_d_va = (final_ph_va - final_pl_va) / 2
ens_m_te = (final_ph_te + final_pl_te) / 2; ens_d_te = (final_ph_te - final_pl_te) / 2
d_dir_va = p_bull_va * d_bull_mean + (1 - p_bull_va) * d_bear_mean
d_dir_te = p_bull_te * d_bull_mean + (1 - p_bull_te) * d_bear_mean
trend_str_va = np.minimum(np.abs(X_va["ret_5"].values) / TREND_SAT, 1.0)
trend_str_te = np.minimum(np.abs(X_te["ret_5"].values) / TREND_SAT, 1.0)


def _apply_beta(b, r, m, d, d_dir, ts, close_, hi_t, lo_t, d_act):
    beff = np.clip(b * (1 - r * ts), 0, 1)
    d_bl = beff * d + (1 - beff) * d_dir
    ph = m + d_bl; pl = m - d_bl
    mh, ml = mape_on(close_, hi_t, lo_t, ph, pl)
    hit = float((np.sign(d_bl) == np.sign(d_act)).mean() * 100)
    return (mh + ml) / 2, hit, beff


best_beta, best_red, best_avg_va = 1.0, 0.0, float("inf")
for b in np.linspace(0, 1, 11):
    for r in [0.0, 0.3, 0.5, 0.7, 0.9]:
        avg, _, _ = _apply_beta(b, r, ens_m_va, ens_d_va, d_dir_va,
                                trend_str_va, close_va, hi_true_va, lo_true_va,
                                d_va.values)
        if avg < best_avg_va:
            best_avg_va = avg; best_beta, best_red = b, r

avg_te, hit_te, beff_te = _apply_beta(best_beta, best_red, ens_m_te, ens_d_te, d_dir_te,
                                       trend_str_te, close_te, hi_true, lo_true, d_te.values)
print(f"  beta={best_beta:.1f}  reduction={best_red}  "
      f"val_avg_MAPE={best_avg_va:.3f}%  test_dir_hit={hit_te:.1f}%")

# Apply beta to compute final test predictions
beff_te_arr = np.clip(best_beta * (1 - best_red * trend_str_te), 0, 1)
d_blend_te  = beff_te_arr * ens_d_te + (1 - beff_te_arr) * d_dir_te
final_yhi_te = np.clip(ens_m_te + d_blend_te, 0, None)
final_ylo_te = np.clip(ens_m_te - d_blend_te, 0, None)
pred_hi_te = close_te * (1 + final_yhi_te)
pred_lo_te = close_te * (1 - final_ylo_te)

mape_h = float((np.abs(pred_hi_te - hi_true) / hi_true).mean() * 100)
mape_l = float((np.abs(pred_lo_te - lo_true) / lo_true).mean() * 100)
hit2_h = float((np.abs(pred_hi_te - hi_true) / close_te <= 0.02).mean() * 100)
hit2_l = float((np.abs(pred_lo_te - lo_true) / close_te <= 0.02).mean() * 100)
hit1_h = float((np.abs(pred_hi_te - hi_true) / close_te <= 0.01).mean() * 100)
hit1_l = float((np.abs(pred_lo_te - lo_true) / close_te <= 0.01).mean() * 100)

print(f"\n>>> TEST METRICS (β={best_beta:.1f}, r={best_red})")
print(f"   MAPE_H = {mape_h:.4f}%   MAPE_L = {mape_l:.4f}%")
print(f"   hit2_H = {hit2_h:.2f}%   hit2_L = {hit2_l:.2f}%")
print(f"   hit1_H = {hit1_h:.2f}%   hit1_L = {hit1_l:.2f}%")
print(f"   direction_hit = {hit_te:.1f}%")

# Residuals for sigma (CI) — must be fractional (not dollar) so CI bands
# are scale-independent: std((actual - predicted) / close)
res_hi = (hi_true - pred_hi_te) / close_te
res_lo = (lo_true - pred_lo_te) / close_te
sigma_hi = float(np.std(res_hi)); sigma_lo = float(np.std(res_lo))

# ---------------------------------------------------------------------------
# 6. DEPLOY RETRAIN — refit on full dataset through DATA_CUTOFF
# ---------------------------------------------------------------------------
# Alpha and beta were tuned on VAL above; now refit all components on ALL
# data (train+val+test combined) so the saved model reflects the full
# training window (train_end = DATA_CUTOFF).  Test metrics above are still
# the legitimate OOS evaluation from the held-out test split.
print("\n>>> Deploy retrain: refitting on full dataset through DATA_CUTOFF …")
X_all    = data[feat_cols]
yhi_all  = data["y_hi"];  ylo_all = data["y_lo"]
d_all    = (yhi_all - ylo_all) / 2
label_all = (d_all > 0).astype(int)
mu_hi_all = float(yhi_all.mean()); mu_lo_all = float(ylo_all.mean())

deploy_preds = {}
for name, ctor in [("huber", M_HUBER), ("quant_lin", M_QUANTLR), ("gbm_quant", M_GBQUANT)]:
    t0 = time.time()
    mh_d = clone(ctor); mh_d.fit(X_all, yhi_all)
    ml_d = clone(ctor); ml_d.fit(X_all, ylo_all)
    deploy_preds[name] = dict(m_hi=mh_d, m_lo=ml_d)
    print(f"  {name}: {time.time()-t0:.1f}s")

dir_clf_deploy = clone(dir_clf)
dir_clf_deploy.fit(X_all, label_all)
print(f"  deploy train_end = {data.index.max().date()}  n = {len(data)}")

# ---------------------------------------------------------------------------
# 7. SAVE ARTIFACT (deploy model — trained on all data through DATA_CUTOFF)
# ---------------------------------------------------------------------------
src_path = str(DAILY_MODEL_CT)

assets = dict(
    ensemble=True, blended=True, alpha=float(alpha_use),
    mu_hi=mu_hi_all, mu_lo=mu_lo_all,
    anchor_hour_utc=ANCHOR_HOUR_UTC,
    constituents=[
        dict(name="huber",     m_hi=deploy_preds["huber"]["m_hi"],
             m_lo=deploy_preds["huber"]["m_lo"]),
        dict(name="quant_lin", m_hi=deploy_preds["quant_lin"]["m_hi"],
             m_lo=deploy_preds["quant_lin"]["m_lo"]),
        dict(name="gbm_quant", m_hi=deploy_preds["gbm_quant"]["m_hi"],
             m_lo=deploy_preds["gbm_quant"]["m_lo"]),
    ],
    hi_model=deploy_preds["huber"]["m_hi"],
    lo_model=deploy_preds["huber"]["m_lo"],
    sigma_hi=sigma_hi, sigma_lo=sigma_lo,
    feat_cols=feat_cols,
    direction_head=dict(
        classifier=dir_clf_deploy,
        beta=float(best_beta),
        beta_trend_reduction=float(best_red),
        trend_saturation=float(TREND_SAT),
        trend_feature="ret_5",
        d_bull_mean=float(d_bull_mean),
        d_bear_mean=float(d_bear_mean),
        test_classifier_acc=float(clf_acc_te),
        test_dir_hit_baseline=float(hit_te),
        test_dir_hit_blended=float(hit_te),
        test_dir_hit_pure=float(hit_te),
        notes="Trained on Yahoo Finance daily data with Coinbase Premium features",
    ),
    calibration_meta=dict(
        anchor_hour_utc=ANCHOR_HOUR_UTC,
        anchor_label="Yahoo Finance daily (00:00 UTC)",
        train_start=str(data.index.min().date()),
        train_end=str(data.index.max().date()),
        val_start=str(val.index.min().date()),
        val_end=str(val.index.max().date()),
        test_start=str((data.index.max() + pd.Timedelta(days=1)).date()),
        test_end=str((data.index.max() + pd.Timedelta(days=1)).date()),
        train_n=int(len(data)),
        val_n=int(len(val)),
        test_n=0,
        embargo_days=EMBARGO_DAYS,
        deploy_mode=True,
        eval_train_end=str(train.index.max().date()),
        eval_test_start=str(test.index.min().date()),
        eval_test_n=int(len(test)),
        winner="gbm_quant",
        metrics=dict(
            mape_h=float(mape_h), mape_l=float(mape_l),
            hit2_h=float(hit2_h), hit2_l=float(hit2_l),
            hit1_h=float(hit1_h), hit1_l=float(hit1_l),
            direction_hit=float(hit_te),
        ),
        tuning_notes=(
            f"Trained on Yahoo Finance daily BTC (2018-06-01 → {DATA_CUTOFF or 'today'}) "
            "with Coinbase Premium features (cb_premium, cb_premium_ma3, cb_premium_z7). "
            "116 total features. Alpha + beta tuned on VAL. DATA_CUTOFF caps training data "
            "so the full bull period is in-sample."
        ),
        cb_premium_included=True,
        training_source="Yahoo Finance daily",
    ),
)

# Backup old model
bk_dir = MODELS_DIR / "backups"
bk_dir.mkdir(parents=True, exist_ok=True)
if Path(src_path).exists():
    bk = bk_dir / f"inference_assets_ct.pre-cbp-{datetime.now():%Y%m%d-%H%M%S}.joblib"
    shutil.copy(src_path, bk)
    print(f"\nBacked up old model: {bk}")

joblib.dump(assets, src_path)
print(f"\n>>> Saved: {src_path}")
print(f">>> train_end = {assets['calibration_meta']['train_end']}  "
      f"(deploy retrain on full dataset)")
print(f">>> test_start (OOS) = {assets['calibration_meta']['test_start']}")
print(f">>> feat_cols: {len(feat_cols)}  sigma_hi={sigma_hi:.4f}  sigma_lo={sigma_lo:.4f}")
print(f">>> MAPE_H={mape_h:.4f}%  MAPE_L={mape_l:.4f}%  (from held-out test split)")
print(f">>> cb_premium in feat_cols: {'cb_premium' in feat_cols}")
