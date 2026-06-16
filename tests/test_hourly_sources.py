"""Verify hourly data sources before building the full pipeline."""
import warnings, time, json
warnings.filterwarnings("ignore")
import pandas as pd
import requests
import yfinance as yf

# 1. Hourly BTC + macro from Yahoo
print("=== Yahoo hourly (interval=60m, period=2y) ===")
for sym in ["BTC-USD","ETH-USD","^GSPC","^IXIC","^VIX","GC=F","DX-Y.NYB","^TNX"]:
    try:
        d = yf.download(sym, period="2y", interval="60m",
                        progress=False, auto_adjust=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = [c[0] for c in d.columns]
        print(f"  {sym:14s}  rows={len(d):6d}  "
              f"first={d.index.min()} last={d.index.max()}  "
              f"cols={list(d.columns)}")
    except Exception as e:
        print(f"  {sym}: FAIL {e}")

# 2. Fear & Greed Index — daily
print("\n=== Fear & Greed (alternative.me, free, daily) ===")
try:
    r = requests.get("https://api.alternative.me/fng/?limit=0", timeout=20)
    j = r.json()
    print(f"  records: {len(j['data'])}  "
          f"first={j['data'][-1]['timestamp']}  last={j['data'][0]['timestamp']}")
    print(f"  example: {j['data'][0]}")
except Exception as e:
    print(f"  FAIL {e}")

# 3. Binance hourly (no key) — primary source for BTC.
# Try api.binance.us first: api.binance.com returns HTTP 451 from US-hosted
# infrastructure (e.g. Streamlit Community Cloud), the same host order the app uses.
print("\n=== Binance klines (no key) ===")
for host in ("https://api.binance.us", "https://api.binance.com"):
    try:
        r = requests.get(f"{host}/api/v3/klines",
                         params={"symbol":"BTCUSDT","interval":"1h","limit":1000},
                         timeout=20)
        if r.status_code != 200:
            print(f"  {host}: HTTP {r.status_code}")
            continue
        j = r.json()
        print(f"  {host}  rows: {len(j)}  first ts: {j[0][0]}  last ts: {j[-1][0]}")
        print(f"  example: time={pd.to_datetime(j[-1][0], unit='ms')} close={j[-1][4]}")
        break
    except Exception as e:
        print(f"  {host}: FAIL {e}")
