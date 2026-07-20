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

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "app"))
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO))

import overall_core as oc                         # noqa: E402
import target_book as tb                          # noqa: E402
from ibkr_rebalance import compute_target_book     # noqa: E402  (reuse the exact app path)

DEFAULT_OUT = _REPO / "data" / "overall" / "target_book.json"
DEFAULT_OUT_LIVE = _REPO / "data" / "overall" / "target_book_live.json"
SATA_KEY = "SATA"


def main() -> int:
    ap = argparse.ArgumentParser(description="Publish the Overall target books (Option C)")
    ap.add_argument("--profile", default=oc.DEFAULT_PROFILE,
                    choices=list(oc.RISK_PROFILES), help="risk profile (default: app default)")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help=f"PAPER book output path (default {DEFAULT_OUT})")
    ap.add_argument("--live-out", default=str(DEFAULT_OUT_LIVE),
                    help=f"LIVE book output path (default {DEFAULT_OUT_LIVE})")
    ap.add_argument("--stdout", action="store_true",
                    help="also print the paper artifact to stdout")
    ap.add_argument("--no-sign", action="store_true",
                    help="do not sign even if OVERALL_BOOK_SECRET is set")
    args = ap.parse_args()

    # The book is published EVERY day, weekends and US market holidays included:
    # Bitcoin trades continuously, so BTC-driven signals (and the resulting
    # target weights) keep moving while the US equity market is closed. The
    # weekend/holiday guard lives solely on the EXECUTOR side
    # (ibkr_execute_book.py / ibkr_rebalance.py via ibkr_common.is_trading_day),
    # so orders are still only ever placed on US market days — but when the
    # executor wakes up on the next trading morning it acts on a book computed
    # from the freshest data, not one frozen at the prior week's close.
    secret = None if args.no_sign else os.environ.get("OVERALL_BOOK_SECRET")

    print(f"Computing target book (live universe run, ~30–90s) — profile {args.profile}…",
          flush=True)
    book = compute_target_book(args.profile)

    if not secret:
        print("WARNING: OVERALL_BOOK_SECRET not set — writing UNSIGNED books. "
              "The executor cannot verify authenticity.", file=sys.stderr)
    else:
        print("Signed with OVERALL_BOOK_SECRET (HMAC-SHA256).")

    # ── PAPER book — undeployed remainder is held as CASH (no SATA leg) ──────
    paper = tb.build_payload(
        as_of=book.as_of, profile=args.profile, weights=book.weights,
        cash_weight=book.cash_weight, exec_price=book.exec_price,
        actions=book.actions, book_mode="paper")

    # ── LIVE book — the SAME risk weights, but the idle remainder is parked in
    # a real SATA position (Strive preferred, ~13% yield) instead of cash. ─────
    live_weights = dict(book.weights)
    live_exec = dict(book.exec_price)
    if book.cash_weight > 1e-6:
        live_weights[SATA_KEY] = live_weights.get(SATA_KEY, 0.0) + book.cash_weight
        try:
            sata_px = oc.fetch_sata().get("price")
        except Exception:
            sata_px = None
        live_exec[SATA_KEY] = float(sata_px) if sata_px else float(oc.SATA["par"])
    live = tb.build_payload(
        as_of=book.as_of, profile=args.profile, weights=live_weights,
        cash_weight=0.0, exec_price=live_exec, actions=book.actions,
        book_mode="live", generated_at_utc=paper["generated_at_utc"])

    outdir = Path(args.out).parent
    outdir.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(tb.dumps(paper, secret))
    Path(args.live_out).write_text(tb.dumps(live, secret))

    print(f"\nTarget books (as-of {book.as_of}, profile {args.profile}):")
    print("  PAPER (idle → cash):")
    for k in sorted(book.weights, key=lambda k: -book.weights[k]):
        print(f"    {k:5s} {book.weights[k]*100:5.1f}%")
    print(f"    CASH  {book.cash_weight*100:5.1f}%")
    print("  LIVE  (idle → SATA):")
    for k in sorted(live_weights, key=lambda k: -live_weights[k]):
        tag = "  ← idle park" if k == SATA_KEY else ""
        print(f"    {k:5s} {live_weights[k]*100:5.1f}%{tag}")
    print(f"\nWrote {args.out}\n      {args.live_out}")
    if args.stdout:
        print("\n" + tb.dumps(paper, secret))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
