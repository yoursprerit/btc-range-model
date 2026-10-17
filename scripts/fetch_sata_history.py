"""Refresh SATA's daily price history (``data/sata/sata_daily.csv``).

SATA (Strive's variable-rate preferred) is the idle-cash park every book holds.
The replays used to model it as a par-stable coupon; this file lets them earn
the REAL total return — ``overall_core.load_sata_returns`` reads it as
``(close + dividend) / previous close − 1`` per session.

Yahoo keeps the full daily history (listed 2025-11-06) and the dividends, so a
run re-pulls everything and merges it over the committed file: rows already
on disk are kept where Yahoo has no row (a feed outage never erases history),
the live pull wins on overlap.  Best-effort by design — any failure leaves the
committed file untouched and exits 0, so the publish workflow never breaks on
a quote outage (``--strict`` makes a failure exit 1 instead).

    python scripts/fetch_sata_history.py            # refresh in place
    python scripts/fetch_sata_history.py --strict   # CI check: fail loudly
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
# kept in step with overall_core.SATA / SATA_DATA_PATH — not imported, so this
# stays a dependency-light fetch (importing the engine loads the models)
TICKER = "SATA"
SATA_DATA_PATH = _REPO / "data" / "sata" / "sata_daily.csv"

COLS = ["open", "high", "low", "close", "volume", "dividend"]


def fetch(ticker: str = TICKER) -> pd.DataFrame:
    """Full daily history from Yahoo (unadjusted closes + cash dividends)."""
    import yfinance as yf
    h = yf.Ticker(ticker).history(period="max", auto_adjust=False)
    if h is None or h.empty:
        raise RuntimeError(f"Yahoo returned no rows for {ticker}")
    h.index = pd.DatetimeIndex(h.index).tz_localize(None).normalize()
    out = pd.DataFrame({
        "open": h["Open"], "high": h["High"], "low": h["Low"],
        "close": h["Close"], "volume": h.get("Volume"),
        "dividend": h.get("Dividends", 0.0),
    }).astype(float)
    out = out[out["close"].notna() & (out["close"] > 0)]
    out.index.name = "date"
    return out[COLS].sort_index()


def merge(old: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """The live pull over the committed rows: new rows win on overlap, old rows
    survive where the pull has none."""
    if old is None or old.empty:
        return new
    both = new.combine_first(old)
    return both[COLS].sort_index()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(SATA_DATA_PATH))
    ap.add_argument("--strict", action="store_true", help="exit 1 on failure")
    args = ap.parse_args()
    out = Path(args.out)
    try:
        old = (pd.read_csv(out, index_col=0, parse_dates=True)
               if out.exists() else None)
        new = fetch()
        df = merge(old, new)
        if old is not None and len(df) < len(old):
            raise RuntimeError(f"merged history shorter than on disk ({len(df)} < {len(old)})")
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, float_format="%.6f")
        print(f"SATA history: {len(df)} sessions {df.index[0].date()} → "
              f"{df.index[-1].date()} (last close {df['close'].iloc[-1]:.2f}) → {out}")
        return 0
    except Exception as exc:                      # best-effort by design
        print(f"WARNING: SATA history not refreshed: {exc}", file=sys.stderr)
        return 1 if args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
