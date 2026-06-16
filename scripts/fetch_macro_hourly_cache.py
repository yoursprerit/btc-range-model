"""Fetch up to 729 days of hourly BTC + macro data from Yahoo Finance and
save a merged snapshot to data/macro_hourly_cache.csv.

Why this exists: yfinance hard-caps 1-hour bars at ~730 days — there is no
API call, on Streamlit Cloud or anywhere else, that can ever return hourly
macro data (VIX, SPX, NDX, DXY, TNX, Gold, ETH) older than that, no matter
when it's made. app/btc_hourly_app.py's fetch_data() lives entirely inside
that rolling window, so the historical-replay tab's minimum selectable date
is bounded by it.

This script takes a one-time snapshot of the *current* 729-day window and
saves it permanently. Unlike data/binance_hourly_btc.csv (which is
regenerable any time via src/fetch_binance_hourly.py and so is gitignored),
this file captures data that becomes permanently unobtainable once it rolls
out of yfinance's live window — it must be committed to the repo, not
regenerated on demand.

fetch_data() loads this cache and only uses rows strictly older than its own
live fetch's earliest timestamp (see app/btc_hourly_app.py). Live data always
wins on overlap, so this cache can only extend history backward — it never
touches or alters any row inside the live window that feeds current
predictions, signals, or backtests.

Run this periodically (e.g. every few months) to checkpoint freshly-expiring
history before it rolls out of the 729-day window; each run's new rows merge
with whatever the cache already has, so coverage only ever grows.

Usage:
    python scripts/fetch_macro_hourly_cache.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import MACRO_HOURLY_CACHE_CSV

# Must match SYMS in app/btc_hourly_app.py's fetch_data().
SYMS = {"btc": "BTC-USD", "eth": "ETH-USD", "spx": "^GSPC", "ndx": "^IXIC",
        "vix": "^VIX", "gold": "GC=F", "dxy": "DX-Y.NYB", "tnx": "^TNX"}


def _flat(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Mirrors app/btc_hourly_app.py's _flat() so the cache's column shape
    matches fetch_data()'s live frame exactly."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = [f"{name}_{c.lower()}" for c in df.columns]
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    df.index = idx
    return df[~df.index.duplicated(keep="last")].sort_index()


def fetch_snapshot() -> pd.DataFrame:
    """Mirrors the df-building section of fetch_data() (pre F&G / coinbase
    join) for the maximal 729-day yfinance window."""
    parts = {}
    for name, sym in SYMS.items():
        raw = yf.download(sym, period="729d", interval="60m",
                          progress=False, auto_adjust=False)
        if raw.empty:
            raise RuntimeError(f"yfinance returned no data for {sym}")
        parts[name] = _flat(raw, name)
    btc = parts["btc"]
    grid = pd.date_range(btc.index.min().floor("h"), btc.index.max().floor("h"),
                         freq="h")
    df = pd.DataFrame(index=grid)
    for name, d in parts.items():
        agg = {f"{name}_open": "first", f"{name}_high": "max", f"{name}_low": "min",
               f"{name}_close": "last", f"{name}_volume": "last"}
        d = d.resample("h").agg(agg)
        d[f"{name}_volume"] = d[f"{name}_volume"].replace(0, np.nan)
        d = d.reindex(grid).ffill(limit=168 if name not in ("btc", "eth") else 4)
        df = df.join(d)
    df = df.dropna(subset=["btc_close"])
    df.index.name = "timestamp_utc"
    return df


def main():
    print(">>> Fetching 729d of hourly BTC + macro from Yahoo Finance ...")
    fresh = fetch_snapshot()
    print(f">>> Fresh range: {fresh.index.min()} -> {fresh.index.max()}  "
          f"({len(fresh)} rows)")

    if MACRO_HOURLY_CACHE_CSV.exists():
        existing = pd.read_csv(MACRO_HOURLY_CACHE_CSV, index_col="timestamp_utc",
                               parse_dates=True)
        combined = pd.concat([existing, fresh]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        print(f">>> Merged with existing cache ({len(existing)} rows) "
              f"-> {len(combined)} rows")
    else:
        combined = fresh

    MACRO_HOURLY_CACHE_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(MACRO_HOURLY_CACHE_CSV)
    print(f">>> Saved {MACRO_HOURLY_CACHE_CSV}  "
          f"({combined.index.min()} -> {combined.index.max()}, {len(combined)} rows)")


if __name__ == "__main__":
    main()
