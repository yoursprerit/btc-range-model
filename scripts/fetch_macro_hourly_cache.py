"""Fetch up to 729 days of hourly BTC + macro + Coinbase-premium data and
save a merged snapshot to data/macro_hourly_cache.csv.

Why this exists: two of fetch_data()'s feeds have hard ceilings shorter than
the model needs for historical replay, and neither can ever be extended by
asking the live API harder:
  - yfinance hard-caps 1-hour bars at ~730 days (BTC, ETH, SPX, NDX, VIX,
    Gold, DXY, TNX).
  - _fetch_coinbase_hourly() self-limits to the last 365 days. cb_premium /
    cb_premium_ma3 / cb_premium_z7 are required hourly-model features, so any
    date older than 365 days has NaN cb_premium and fails valid_mask — this
    was the actual binding constraint on min_date even after the yfinance
    macro cache was added (Coinbase's API itself has no such limit; it's a
    self-imposed window in the live-fetch code, kept short there to bound
    the ~30-request pagination cost on every cache refresh).

This script takes a one-time snapshot of the *current* 729-day window (the
longest of the two) for all of: BTC, the 7 macro symbols, and Coinbase
BTC-USD close. Unlike data/binance_hourly_btc.csv (regenerable any time via
src/fetch_binance_hourly.py, hence gitignored), this file captures data that
becomes permanently unobtainable once it rolls out of yfinance's live
window — it must be committed to the repo, not regenerated on demand.

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
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import MACRO_HOURLY_CACHE_CSV

# Must match SYMS in app/btc_hourly_app.py's fetch_data().
SYMS = {"btc": "BTC-USD", "eth": "ETH-USD", "spx": "^GSPC", "ndx": "^IXIC",
        "vix": "^VIX", "gold": "GC=F", "dxy": "DX-Y.NYB", "tnx": "^TNX"}

CB_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"


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


def fetch_coinbase_hourly(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.Series:
    """Paginated Coinbase hourly close fetch over an arbitrary [start, end]
    window. Same request shape as app/btc_hourly_app.py's
    _fetch_coinbase_hourly(), just parameterized instead of fixed to the
    last 365 days — Coinbase's API has no shorter limit of its own."""
    rows: list = []
    cur = start_ts
    while cur < end_ts:
        chunk_end = min(cur + pd.Timedelta(hours=299), end_ts)
        try:
            r = requests.get(CB_URL, params={
                "granularity": 3600,
                "start": cur.isoformat() + "Z",
                "end":   (chunk_end + pd.Timedelta(hours=1)).isoformat() + "Z",
            }, timeout=20)
            if r.status_code == 200:
                rows.extend(r.json())
        except Exception:
            pass
        cur = chunk_end + pd.Timedelta(hours=1)
        time.sleep(0.12)
    if not rows:
        return pd.Series(dtype=float, name="coinbase_close")
    cb = pd.DataFrame(rows, columns=["ts", "low", "high", "open", "close", "volume"])
    cb.index = pd.to_datetime(cb["ts"], unit="s").dt.floor("h")
    cb = cb[~cb.index.duplicated(keep="last")].sort_index()
    return cb["close"].astype(float).rename("coinbase_close")


def fetch_snapshot() -> pd.DataFrame:
    """Mirrors the df-building section of fetch_data() (pre F&G join) for
    the maximal 729-day yfinance window, plus the matching Coinbase window."""
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

    print(">>> Fetching matching Coinbase hourly window ...")
    cb_close = fetch_coinbase_hourly(df.index.min(), df.index.max())
    df["coinbase_close"] = cb_close.reindex(df.index)

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
        # combine_first (not concat+sort_index+drop_duplicates): sort_index()'s
        # default quicksort isn't stable for duplicate index values, so it can
        # silently keep the wrong side of a duplicate row. combine_first takes
        # fresh's value at every index where fresh has one, falling back to
        # existing only where fresh has no row (or a null cell) — no ambiguity.
        combined = fresh.combine_first(existing).sort_index()
        print(f">>> Merged with existing cache ({len(existing)} rows) "
              f"-> {len(combined)} rows")
    else:
        combined = fresh

    MACRO_HOURLY_CACHE_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(MACRO_HOURLY_CACHE_CSV)
    print(f">>> Saved {MACRO_HOURLY_CACHE_CSV}  "
          f"({combined.index.min()} -> {combined.index.max()}, {len(combined)} rows)")
    cb_coverage = combined["coinbase_close"].notna().sum()
    print(f">>> coinbase_close coverage: {cb_coverage}/{len(combined)} rows")


if __name__ == "__main__":
    main()
