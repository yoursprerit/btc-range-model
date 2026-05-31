#!/usr/bin/env python3
"""Generate raw_ct.csv and features_ct.csv from Yahoo Finance data.

These files are required by src/train_3class_day_type.py. Mirrors the data
fetching and feature engineering in train_daily_yahoo.py but skips model
training so it's fast and doesn't overwrite the saved H/L model.

Clips raw_ct.csv to full available data (so shift(-1) works on last row).
Clips features_ct.csv to DATA_CUTOFF so training stays in the intended window.
"""
import sys
import time
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from paths import RAW_CT_CSV, FEATURES_CT_CSV

FETCH_START = "2018-06-01"
FETCH_END   = (pd.Timestamp("today") + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
DATA_CUTOFF = "2026-02-28"


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
                  "end": (chunk_end + pd.Timedelta(days=1)).isoformat() + "Z"}
        try:
            r = requests.get(CB_URL, params=params, timeout=30)
            if r.status_code == 200:
                rows.extend(r.json())
            else:
                print(f"  ! Coinbase {r.status_code}")
        except Exception as exc:
            print(f"  ! Coinbase error: {exc}")
        cur = chunk_end + pd.Timedelta(days=1)
        time.sleep(0.35)
    if not rows:
        return pd.Series(dtype=float, name="coinbase_close")
    tmp = pd.DataFrame(rows, columns=["ts", "low", "high", "open", "close", "volume"])
    tmp["date"] = pd.to_datetime(tmp["ts"], unit="s").dt.normalize()
    tmp = tmp.drop_duplicates("date").set_index("date").sort_index()
    print(f"  coinbase_daily: {len(tmp)} rows  {tmp.index.min().date()} → {tmp.index.max().date()}")
    return tmp["close"].rename("coinbase_close")


# 1. Fetch data
print("\n>>> Fetching Yahoo Finance BTC + macro …")
SYMS = {"btc": "BTC-USD", "eth": "ETH-USD", "spx": "^GSPC", "ndx": "^IXIC",
        "vix": "^VIX", "gold": "GC=F", "dxy": "DX-Y.NYB", "tnx": "^TNX"}
parts = []
for name, sym in SYMS.items():
    d = yf.download(sym, start=FETCH_START, end=FETCH_END, progress=False, auto_adjust=False)
    if d.empty:
        print(f"  ! empty {sym}"); continue
    parts.append(_flat(d, name))
    print(f"  {sym}: {len(d)} rows")
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

# 2. Join
print("\n>>> Joining …")
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
print(f">>> Raw shape {df.shape}  {df.index.min().date()} → {df.index.max().date()}")

# Save raw (full, unclipped — needed for shift(-1) on last row)
df.to_csv(RAW_CT_CSV)
print(f">>> Saved {RAW_CT_CSV}")

# 3. Feature engineering
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

if DATA_CUTOFF:
    data = data.loc[:DATA_CUTOFF]

print(f">>> Feature matrix {data.shape}  features={data.shape[1]-5}")
print(f">>> Feature range: {data.index.min().date()} → {data.index.max().date()}")

data.to_csv(FEATURES_CT_CSV)
print(f">>> Saved {FEATURES_CT_CSV}")
print("\nDone.")
