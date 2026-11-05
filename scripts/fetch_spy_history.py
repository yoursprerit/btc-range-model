"""Refresh SPY's daily price history (``data/spy/spy_daily.csv``).

SPY is the Strategy P&L section's passive benchmark ("SPY Buy & Hold"):
``overall_core.load_spy_equity`` compounds ``(close + dividend) / previous
close − 1`` per session from this file.  Same best-effort merge as
``fetch_sata_history.py`` (committed rows survive a feed outage; failures exit
0 unless ``--strict``).

    python scripts/fetch_spy_history.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_sata_history import fetch, merge   # noqa: E402

SPY_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "spy" / "spy_daily.csv"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(SPY_DATA_PATH))
    ap.add_argument("--strict", action="store_true", help="exit 1 on failure")
    args = ap.parse_args()
    out = Path(args.out)
    try:
        old = (pd.read_csv(out, index_col=0, parse_dates=True)
               if out.exists() else None)
        df = merge(old, fetch("SPY"))
        if old is not None and len(df) < len(old):
            raise RuntimeError(f"merged history shorter than on disk ({len(df)} < {len(old)})")
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, float_format="%.6f")
        print(f"SPY history: {len(df)} sessions {df.index[0].date()} → "
              f"{df.index[-1].date()} (last close {df['close'].iloc[-1]:.2f}) → {out}")
        return 0
    except Exception as exc:
        print(f"WARNING: SPY history not refreshed: {exc}", file=sys.stderr)
        return 1 if args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
