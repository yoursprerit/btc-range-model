"""Execute a published target book against IBKR paper (Option C, executor side).

Consumes the JSON artifact produced by ``scripts/publish_target_book.py`` and
rebalances an IBKR **paper** account to match it — WITHOUT running the model.
This is the lightweight "local executes" half of Option C: it needs only
``ib_async`` + ``pandas`` and the shared ``ibkr_common`` / ``target_book``
modules — none of the scientific / model stack.

    # preview (default — no orders); reads the book, diffs vs your positions:
    python scripts/ibkr_execute_book.py --file data/overall/target_book.json

    # from a URL (e.g. the raw artifact on the feature branch):
    python scripts/ibkr_execute_book.py --url https://…/target_book.json

    # actually place the paper orders:
    OVERALL_BOOK_SECRET=… python scripts/ibkr_execute_book.py --file … --execute

Safety rails (same posture as the all-in-one rebalancer)
--------------------------------------------------------
* ``--dry-run`` is the DEFAULT; orders require ``--execute``.
* Paper-account guard (id must start with ``DU``) unless ``--allow-nonpaper``.
* If ``OVERALL_BOOK_SECRET`` is set (or ``--require-signature`` is passed) the
  book's HMAC signature MUST verify, or the run aborts — a tampered/forged book
  can never trade the account.
* Freshness guards: rejects a stale signal bar or a book generated too long ago,
  and skips weekends / US holidays.
* Sizing uses the book's publish-time execution prices, so the executor makes no
  market-data calls; market orders still fill at the live price.
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "app"))
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO))

import target_book as tb                           # noqa: E402  (light: stdlib + pandas)
import ibkr_symbols as sym                          # noqa: E402
from ibkr_common import (                           # noqa: E402
    DEFAULT_PORT, Broker, build_order_plan, is_trading_day, print_plan,
)


def _load_book(args) -> dict:
    if args.file:
        return tb.loads(Path(args.file).read_text())
    if args.url:
        with urllib.request.urlopen(args.url, timeout=30) as r:   # noqa: S310 (user-supplied URL)
            return tb.loads(r.read().decode())
    return tb.loads(sys.stdin.read())


def _print_book(payload: dict) -> None:
    weights = payload.get("weights", {})
    print(f"\nPublished book — profile {payload.get('profile')} — as-of "
          f"{payload.get('as_of')} (generated {payload.get('generated_at_utc')}):")
    if not weights:
        print("  (fully in cash — no deployed positions)")
    for k in sorted(weights, key=lambda k: -weights[k]):
        px = payload.get("exec_price", {}).get(k)
        pxs = f"@ {px:.2f}" if px else "(no price)"
        print(f"  {k:5s} → {sym.trade_symbol(k) or '?':5s} {weights[k]*100:5.1f}%  {pxs}")
    print(f"  CASH  {payload.get('cash_weight', 0.0)*100:5.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description="Execute a published target book against IBKR paper")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--file", help="path to a target-book JSON")
    src.add_argument("--url", help="URL to fetch the target-book JSON from")
    # (no source → read JSON from stdin)
    ap.add_argument("--execute", action="store_true",
                    help="actually place orders (default is a no-order dry-run)")
    ap.add_argument("--band", type=float, default=0.01,
                    help="no-trade band as a fraction of net-liq (default 0.01 = 1%%)")
    ap.add_argument("--fractional", action="store_true",
                    help="allow fractional shares (default: whole shares)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT,
                    help=f"IB Gateway API port (default {DEFAULT_PORT} = paper)")
    ap.add_argument("--client-id", type=int, default=18)
    ap.add_argument("--fill-timeout", type=float, default=60.0,
                    help="seconds to wait for each order leg to fill")
    ap.add_argument("--max-age-hours", type=float, default=12.0,
                    help="reject a book generated more than this many hours ago")
    ap.add_argument("--require-signature", action="store_true",
                    help="abort unless the book carries a valid signature")
    ap.add_argument("--allow-nonpaper", action="store_true",
                    help="permit a non-DU account (DANGEROUS — disables the paper guard)")
    ap.add_argument("--force", action="store_true",
                    help="ignore the weekend/holiday & freshness guards")
    args = ap.parse_args()

    payload = _load_book(args)
    _print_book(payload)

    # ── signature ────────────────────────────────────────────────────────────
    secret = os.environ.get("OVERALL_BOOK_SECRET")
    ok_sig, sig_why = tb.verify_signature(payload, secret)
    print(f"Signature: {sig_why}")
    if not ok_sig:
        print("ABORT: signature check failed. No orders placed.")
        return 1
    if args.require_signature and not payload.get("signature"):
        print("ABORT: --require-signature set but the book is unsigned.")
        return 1

    # ── trading day + freshness ──────────────────────────────────────────────
    today = pd.Timestamp.now(tz="America/New_York").tz_localize(None)
    ok_day, day_why = is_trading_day(today)
    if not ok_day and not args.force:
        print(f"Not trading: {day_why} (use --force to override).")
        return 0
    ok_val, val_why = tb.validate(payload, today, max_gen_age_hours=args.max_age_hours)
    print(f"Freshness: {val_why}")
    if not ok_val and not args.force:
        print("ABORT: book failed freshness validation (use --force to override).")
        return 1

    weights = payload.get("weights", {})
    exec_price = payload.get("exec_price", {})

    # ── dry-run: connect only to diff against live positions ─────────────────
    if not args.execute:
        print("\n[dry-run] Connecting only to read positions for the plan preview…")
        try:
            broker = Broker(args.host, args.port, args.client_id, args.allow_nonpaper)
        except Exception as e:
            print(f"[dry-run] Could not connect to IB Gateway ({e}).")
            print("[dry-run] Showing the published book only; run with a gateway to "
                  "diff against live positions.")
            return 0
        try:
            net_liq = broker.net_liq()
            current = broker.positions_by_key()
            orders = build_order_plan(weights, exec_price, net_liq, current,
                                      args.band, args.fractional)
            print(f"\nAccount {broker.account} (PAPER).")
            print_plan(orders, net_liq)
            print("\n[dry-run] No orders transmitted. Re-run with --execute to trade.")
        finally:
            broker.disconnect()
        return 0

    # ── execute ──────────────────────────────────────────────────────────────
    broker = Broker(args.host, args.port, args.client_id, args.allow_nonpaper)
    try:
        net_liq = broker.net_liq()
        current = broker.positions_by_key()
        orders = build_order_plan(weights, exec_price, net_liq, current,
                                  args.band, args.fractional)
        print(f"\nAccount {broker.account} (PAPER).")
        print_plan(orders, net_liq)
        if orders:
            print("\nTransmitting orders (sells → buys)…")
            broker.place(orders, args.fractional, args.fill_timeout)
        print("\n✓ Rebalance complete.")
    finally:
        broker.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
