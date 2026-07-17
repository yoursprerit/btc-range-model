"""Publish today's target book (Option C, publisher side).

Runs the full Overall engine and writes the live-adjusted target allocation as a
compact, optionally-signed JSON artifact that a local executor
(``scripts/ibkr_execute_book.py``) consumes to trade an IBKR paper account —
without ever running the model itself.

This is the "cloud decides" half of Option C.  It's transport-agnostic: it just
writes a file.  Wire it to a transport however you like — e.g. the GitHub Action
``.github/workflows/publish-target-book.yml`` runs this and commits the artifact
to the feature branch, so the executor can ``git pull`` (or fetch the raw URL)
and act on it.

    OVERALL_BOOK_SECRET=… python scripts/publish_target_book.py            # default profile
    python scripts/publish_target_book.py --profile Balanced --stdout

Signing: set ``OVERALL_BOOK_SECRET`` (env) to HMAC-sign the artifact so the
executor can verify authenticity before trading.  Without it the book is written
unsigned (a warning is printed).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "app"))
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO))

import overall_core as oc                         # noqa: E402
import target_book as tb                          # noqa: E402
from ibkr_common import is_trading_day             # noqa: E402  (shared weekend/holiday guard)
from ibkr_rebalance import compute_target_book     # noqa: E402  (reuse the exact app path)

DEFAULT_OUT = _REPO / "data" / "overall" / "target_book.json"


def main() -> int:
    ap = argparse.ArgumentParser(description="Publish the Overall target book (Option C)")
    ap.add_argument("--profile", default=oc.DEFAULT_PROFILE,
                    choices=list(oc.RISK_PROFILES), help="risk profile (default: app default)")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help=f"output JSON path (default {DEFAULT_OUT})")
    ap.add_argument("--stdout", action="store_true",
                    help="also print the artifact to stdout")
    ap.add_argument("--no-sign", action="store_true",
                    help="do not sign even if OVERALL_BOOK_SECRET is set")
    ap.add_argument("--force", action="store_true",
                    help="publish even on a weekend / US market holiday")
    args = ap.parse_args()

    # Don't publish when the US market is closed: no new daily bar has formed, so
    # a holiday/weekend run would only re-emit the last trading day's book with a
    # fresh timestamp. Uses the SAME weekend + US-holiday calendar the executor
    # trusts (ibkr_common.is_trading_day), so publisher and executor never
    # disagree about what's a trading day.
    today = pd.Timestamp.now(tz="America/New_York").tz_localize(None)
    ok, why = is_trading_day(today)
    if not ok and not args.force:
        print(f"Skipping publish: {why} (US market closed; use --force to override). "
              "No book written.")
        return 0

    secret = None if args.no_sign else os.environ.get("OVERALL_BOOK_SECRET")

    print(f"Computing target book (live universe run, ~30–90s) — profile {args.profile}…",
          flush=True)
    book = compute_target_book(args.profile)

    payload = tb.build_payload(
        as_of=book.as_of, profile=args.profile, weights=book.weights,
        cash_weight=book.cash_weight, exec_price=book.exec_price, actions=book.actions)
    text = tb.dumps(payload, secret)

    if not secret:
        print("WARNING: OVERALL_BOOK_SECRET not set — writing an UNSIGNED book. "
              "The executor cannot verify authenticity.", file=sys.stderr)
    else:
        print("Signed with OVERALL_BOOK_SECRET (HMAC-SHA256).")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)

    print(f"\nTarget book (as-of {book.as_of}, profile {args.profile}):")
    for k in sorted(book.weights, key=lambda k: -book.weights[k]):
        print(f"  {k:5s} {book.weights[k]*100:5.1f}%")
    print(f"  CASH  {book.cash_weight*100:5.1f}%")
    print(f"\nWrote {out}")
    if args.stdout:
        print("\n" + text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
