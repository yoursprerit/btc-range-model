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
import executed_book as eb                          # noqa: E402
import ibkr_symbols as sym                          # noqa: E402
from ibkr_common import (                           # noqa: E402
    DEFAULT_PORT, Broker, build_order_plan, is_trading_day, print_plan,
)

DEFAULT_REPORT = _REPO / "data" / "overall" / "executed_book.json"


def _write_report(broker, payload: dict, mode: str, trades: list[dict],
                  out_path: str, secret, account_mode: str = "paper") -> None:
    """Write the execution report (trades + current positions) the Executed Book
    page reads. Best-effort — a failure here never fails the rebalance."""
    try:
        net_liq = broker.net_liq()
    except Exception:
        net_liq = 0.0
    cash = broker.cash()
    try:
        positions = broker.portfolio_snapshot()
    except Exception:
        positions = []
    report = eb.build_payload(
        as_of=payload.get("as_of", ""), profile=payload.get("profile", ""),
        mode=mode, account=broker.account, net_liq=net_liq, cash=cash,
        trades=trades, positions=positions, account_mode=account_mode)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tb.dumps(report, secret))
    print(f"Wrote execution report → {out_path} "
          f"({len(trades)} trade(s), {len(positions)} position(s))")


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
    ap.add_argument("--force", action="store_true",
                    help="ignore the weekend/holiday & freshness guards")
    # ── account mode (paper vs LIVE) ─────────────────────────────────────────
    ap.add_argument("--account-mode", choices=["paper", "live"], default="paper",
                    help="paper (default) requires a DU… account; live requires a "
                         "real account + --confirm-live (+ matching --expected-account)")
    ap.add_argument("--expected-account",
                    help="pin the exact account id; the run aborts if the connected "
                         "account differs (strongly recommended for live)")
    ap.add_argument("--confirm-live", action="store_true",
                    help="required with --account-mode live (you are trading REAL money)")
    # ── live safety limits ───────────────────────────────────────────────────
    ap.add_argument("--max-deploy-frac", type=float, default=1.0,
                    help="cap total deployed weight at this fraction of net-liq "
                         "(e.g. 0.25 = never deploy more than 25%%); the rest stays cash")
    ap.add_argument("--max-order-notional", type=float, default=0.0,
                    help="clamp any single order to this dollar cap (0 = no cap)")
    ap.add_argument("--kill-switch-file",
                    help="if this file exists (or env IBKR_TRADING_DISABLED is set), "
                         "abort before placing any order")
    ap.add_argument("--report-out",
                    help="execution-report output path (default: executed_book.json, "
                         "or executed_book_live.json in live mode)")
    ap.add_argument("--no-report", action="store_true",
                    help="do not write the execution report")
    args = ap.parse_args()

    live = args.account_mode == "live"
    report_out = args.report_out or str(
        _REPO / "data" / "overall" / ("executed_book_live.json" if live
                                      else "executed_book.json"))

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

    weights = dict(payload.get("weights", {}))
    exec_price = payload.get("exec_price", {})

    # ── live-mode banner + confirmation ──────────────────────────────────────
    if live:
        print("\n*** LIVE ACCOUNT MODE — REAL MONEY *** "
              f"(deploy cap {args.max_deploy_frac*100:.0f}% of NAV"
              + (f", per-order cap ${args.max_order_notional:,.0f}"
                 if args.max_order_notional else "") + ")")

    # ── exposure cap: scale weights so total deployed ≤ max_deploy_frac ──────
    total_w = sum(weights.values())
    if args.max_deploy_frac < 1.0 and total_w > args.max_deploy_frac:
        scale = args.max_deploy_frac / total_w
        weights = {k: w * scale for k, w in weights.items()}
        print(f"Exposure cap: scaling deployed weight {total_w*100:.0f}% → "
              f"{args.max_deploy_frac*100:.0f}% of NAV (× {scale:.3f}); rest stays cash.")

    # ── kill switch — a manual, instant halt independent of cron/systemd ─────
    if os.environ.get("IBKR_TRADING_DISABLED") or (
            args.kill_switch_file and Path(args.kill_switch_file).exists()):
        print("ABORT: trading is DISABLED (kill switch active). No orders placed.")
        return 0

    account_kwargs = dict(account_mode=args.account_mode,
                          expected_account=args.expected_account,
                          confirm_live=args.confirm_live)
    tag = args.account_mode.upper()

    # ── dry-run: connect only to diff against live positions ─────────────────
    if not args.execute:
        print("\n[dry-run] Connecting only to read positions for the plan preview…")
        try:
            broker = Broker(args.host, args.port, args.client_id, **account_kwargs)
        except Exception as e:
            print(f"[dry-run] Could not connect / guard failed ({e}).")
            print("[dry-run] Showing the published book only.")
            return 0
        try:
            net_liq = broker.net_liq()
            current = broker.positions_by_key()
            orders = build_order_plan(weights, exec_price, net_liq, current,
                                      args.band, args.fractional, args.max_order_notional)
            print(f"\nAccount {broker.account} ({tag}).")
            print_plan(orders, net_liq)
            if not args.no_report:
                planned = [dict(key=o.key, symbol=o.symbol, action=o.action,
                                qty=o.qty, price=o.price, status="PLANNED",
                                filled=0.0, avg_fill_price=0.0, reason=o.reason)
                           for o in orders]
                _write_report(broker, payload, "dry-run", planned,
                              report_out, secret, args.account_mode)
            print("\n[dry-run] No orders transmitted. Re-run with --execute to trade.")
        finally:
            broker.disconnect()
        return 0

    # ── execute ──────────────────────────────────────────────────────────────
    broker = Broker(args.host, args.port, args.client_id, **account_kwargs)
    try:
        net_liq = broker.net_liq()
        current = broker.positions_by_key()
        orders = build_order_plan(weights, exec_price, net_liq, current,
                                  args.band, args.fractional, args.max_order_notional)
        print(f"\nAccount {broker.account} ({tag}).")
        print_plan(orders, net_liq)
        fills: list[dict] = []
        if orders:
            print("\nTransmitting orders (sells → buys)…")
            fills = broker.place(orders, args.fractional, args.fill_timeout)
        if not args.no_report:
            # positions read AFTER the fills → the report shows the resulting book
            _write_report(broker, payload, "execute", fills, report_out, secret,
                          args.account_mode)
        print("\n✓ Rebalance complete.")
    finally:
        broker.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
