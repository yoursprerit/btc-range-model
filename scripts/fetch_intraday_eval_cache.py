"""Snapshot hourly OHLC for the sleeves the intraday-drop eval covers.

yfinance caps 1-hour bars at ~730 days, so history rolls out of the window and
cannot be re-fetched later.  This merges a fresh pull into
``data/intraday_eval/hourly_<TICKER>.csv`` (existing rows kept, new rows win on
overlap) so ``scripts/eval_intraday_drop.py`` stays reproducible.

    python scripts/fetch_intraday_eval_cache.py            # all tickers
    python scripts/fetch_intraday_eval_cache.py SOXL NUGT
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

_REPO = Path(__file__).resolve().parent.parent
OUT = _REPO / "data" / "intraday_eval"
TICKERS = ["SOXL", "NUGT", "MSTU", "WGMI"]


def fetch(tk: str) -> pd.DataFrame:
    d = yf.download(tk, period="730d", interval="1h", progress=False, auto_adjust=False)
    if d is None or d.empty:        # e.g. MSTU: listed inside the window
        start = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=725)).strftime("%Y-%m-%d")
        d = yf.download(tk, start=start, interval="1h", progress=False, auto_adjust=False)
    if d is None or d.empty:
        raise RuntimeError(f"{tk}: no hourly data")
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] for c in d.columns]
    d = d.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    d.index = pd.DatetimeIndex(d.index).tz_convert("America/New_York").tz_localize(None)
    d.index.name = "ts_et"
    return d.dropna(subset=["close"])


def main(argv):
    OUT.mkdir(parents=True, exist_ok=True)
    for tk in (argv or TICKERS):
        new = fetch(tk)
        p = OUT / f"hourly_{tk}.csv"
        if p.exists():
            old = pd.read_csv(p, index_col=0, parse_dates=True)
            new = pd.concat([old[~old.index.isin(new.index)], new]).sort_index()
        new.to_csv(p)
        print(f"{tk}: {len(new)} bars {new.index[0]} → {new.index[-1]}")


if __name__ == "__main__":
    main(sys.argv[1:])
