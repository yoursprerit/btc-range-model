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
from datetime import date
from pathlib import Path

import pandas as pd

# Redirected stdout on Windows (a Task Scheduler pipe, `> out.txt`, a PowerShell
# `|`) is encoded with the ANSI codepage, not UTF-8 — so the arrows and rules in
# the printout below raise UnicodeEncodeError and abort the run, even though the
# identical command works when typed at a console. Make our own streams UTF-8 so
# the output never depends on how the caller redirected us.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):   # already-wrapped or non-reconfigurable stream
        pass

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "app"))
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO))

import target_book as tb                           # noqa: E402  (light: stdlib + pandas)
import executed_book as eb                          # noqa: E402
import ibkr_symbols as sym                          # noqa: E402
from ibkr_common import (                           # noqa: E402
    DEFAULT_PORT, DEFAULT_SLIPPAGE_CAP, ORDER_MARKETABLE_LIMIT, ORDER_TYPES,
    Broker, build_order_plan, is_trading_day, print_plan,
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
    # dated copy (executed_archive/<as_of>[_live].json) — the live report file
    # only ever holds the LAST run, so without this the Executed Book page's
    # Historical tab would have nothing to browse. Best-effort by design.
    arch = eb.archive_report(report, out, secret)
    if arch:
        print(f"  archived as-of record → {arch}")
    if not secret:
        # Silently emitting an unsigned report is how "signature mismatch"
        # shows up in the app hours later, with nothing in the run to explain
        # it. Say so at the point it happens.
        print("  WARN: report is UNSIGNED (OVERALL_BOOK_SECRET not set in this "
              "shell). The Executed Book page will report a signature error. "
              "Open a new PowerShell so setx reaches it, then re-run with "
              "--refresh-report.")


def _push_report(out_path: str) -> None:
    """Commit and push the execution report, the way ibkr_execute_daily.ps1 does.

    A manual run otherwise writes a perfectly good report that never leaves the
    laptop, so the cloud app keeps showing whatever the last scheduled run
    published. On any failure this resets the file rather than leaving the
    branch diverged -- a stale report is recoverable, a wedged repo is a chore.
    """
    import subprocess
    out = Path(out_path).resolve()
    # the report itself PLUS the dated as-of record written beside it — the
    # Historical tab is only as complete as what gets pushed.
    rels = [str(out.relative_to(_REPO)),
            str(eb.archive_dir(out).relative_to(_REPO))]
    def _git(*a):
        return subprocess.run(("git",) + a, cwd=_REPO, capture_output=True, text=True)
    branch = (_git("rev-parse", "--abbrev-ref", "HEAD").stdout or "main").strip()
    _git("add", *rels)
    if _git("diff", "--cached", "--quiet", "--", *rels).returncode == 0:
        print("  report unchanged — nothing to publish")
        return
    _git("-c", "user.name=ibkr-executor", "-c", "user.email=executor@localhost",
         "commit", "-q", "-m", f"chore(ibkr): execution report {date.today()}")
    r = _git("push", "origin", f"HEAD:{branch}")
    if r.returncode == 0:
        print(f"  published execution report to origin/{branch}")
    else:
        print(f"  WARN: could not push ({r.stderr.strip().splitlines()[-1] if r.stderr.strip() else 'unknown'})")
        print("  see IBKR_OPTION_C_WINDOWS.md section 8 — rolling back the commit")
        _git("reset", "--hard", f"origin/{branch}")


def _load_book(args, default_path: str) -> dict:
    if args.file:
        return tb.loads(Path(args.file).read_text())
    if args.url:
        with urllib.request.urlopen(args.url, timeout=30) as r:   # noqa: S310 (user-supplied URL)
            return tb.loads(r.read().decode())
    if not sys.stdin.isatty():                 # data piped in → read it
        return tb.loads(sys.stdin.read())
    return tb.loads(Path(default_path).read_text())   # else the per-mode default book


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
                    help="seconds to wait for each order leg to fill (MOC ignores "
                         "this and waits for the 4:00 PM ET auction)")
    ap.add_argument("--order-type", choices=list(ORDER_TYPES),
                    default=ORDER_MARKETABLE_LIMIT,
                    help="marketable-limit (default): a limit priced through the "
                         "touch by --slippage-cap — fills like a market order but "
                         "caps what a bad quote can cost; moc: market-on-close, "
                         "filled in the 4:00 PM ET auction (the price the engine "
                         "books against, so it removes the drift between live and "
                         "backtest); market: unprotected, no price ceiling")
    ap.add_argument("--slippage-cap", type=float, default=DEFAULT_SLIPPAGE_CAP,
                    help=f"how far THROUGH the touch a marketable limit is priced "
                         f"(default {DEFAULT_SLIPPAGE_CAP} = "
                         f"{DEFAULT_SLIPPAGE_CAP*100:.1f}%%)")
    ap.add_argument("--max-age-hours", type=float, default=36.0,
                    help="reject a book generated more than this many hours ago "
                         "(default 36 — spans the 7:15-AM-CT publish anchor to "
                         "the next day's 2:30-PM-CT executor slot (31.25 h) with "
                         "slack, so yesterday's book still trades when today's "
                         "publish was withheld by a failed audit)")
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
    ap.add_argument("--refresh-report", action="store_true",
                    help="place NO orders: connect, read the account's real "
                         "positions and today's fills straight from IBKR, and "
                         "rewrite a SIGNED execution report. Use after late or "
                         "partial fills print past the sending run's timeout, "
                         "or to re-sign a report written without the secret")
    ap.add_argument("--push-report", action="store_true",
                    help="git commit + push the execution report after writing "
                         "it, the way the scheduled wrapper does (a manual run "
                         "otherwise leaves the cloud app showing a stale report)")
    ap.add_argument("--outside-rth", action="store_true",
                    help="allow fills outside regular trading hours: stamps "
                         "outsideRth on the orders and forces marketable-limit "
                         "(IBKR REJECTS market and MOC orders outside RTH). "
                         "Manual use only -- the scheduled 2:30 PM CT wrapper "
                         "never passes this, so automated runs are unchanged")
    ap.add_argument("--market-data-type", type=int, default=1,
                    choices=(1, 2, 3, 4),
                    help="IBKR market-data type: 1 live (default), 2 frozen, "
                         "3 delayed, 4 delayed-frozen. Delayed is FREE and needs "
                         "no subscription, but IBKR only serves it when asked -- "
                         "use 3 if quotes come back empty (error 10089)")
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
    default_book = str(_REPO / "data" / "overall" / (
        "target_book_live.json" if live else "target_book.json"))

    payload = _load_book(args, default_book)
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

    # ── exposure cap: cap RISK assets at max_deploy_frac; route the freed weight
    # to SATA on a live book (idle → yield park) or to cash otherwise. SATA itself
    # is not counted as risk, so a live book's idle park isn't scaled down. ─────
    risk_keys = [k for k in weights if k != "SATA"]
    risk_total = sum(weights[k] for k in risk_keys)
    if args.max_deploy_frac < 1.0 and risk_total > args.max_deploy_frac:
        scale = args.max_deploy_frac / risk_total
        freed = 0.0
        for k in risk_keys:
            nw = weights[k] * scale
            freed += weights[k] - nw
            weights[k] = nw
        if "SATA" in weights:
            weights["SATA"] = weights.get("SATA", 0.0) + freed
            sink = "SATA"
        else:
            sink = "cash"
        print(f"Exposure cap: risk {risk_total*100:.0f}% → {args.max_deploy_frac*100:.0f}% "
              f"of NAV; freed {freed*100:.0f}% → {sink}.")

    # ── kill switch — a manual, instant halt independent of cron/systemd ─────
    if os.environ.get("IBKR_TRADING_DISABLED") or (
            args.kill_switch_file and Path(args.kill_switch_file).exists()):
        print("ABORT: trading is DISABLED (kill switch active). No orders placed.")
        return 0

    account_kwargs = dict(account_mode=args.account_mode,
                          expected_account=args.expected_account,
                          confirm_live=args.confirm_live,
                          market_data_type=args.market_data_type)
    tag = args.account_mode.upper()

    # ── report refresh: no orders, just re-state reality ─────────────────────
    if args.refresh_report:
        print("\n[refresh-report] Reading positions and today's fills — "
              "no orders will be placed.")
        try:
            broker = Broker(args.host, args.port, args.client_id, **account_kwargs)
        except Exception as e:
            print(f"ABORT: could not connect to IB Gateway ({e}).")
            return 1
        try:
            fills = broker.day_fills_by_symbol()
            print(f"  {len(fills)} symbol(s) with executions today")
            # mode stays 'execute': orders really were sent for this book, and
            # the app's badge should reflect that rather than reading 'dry-run'.
            _write_report(broker, payload, "execute", fills, report_out, secret,
                          account_mode=args.account_mode)
        finally:
            broker.disconnect()
        if args.push_report:
            _push_report(report_out)
        return 0

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
            print(f"\nTransmitting orders ({args.order_type})…")
            fills = broker.place(orders, args.fractional, args.fill_timeout,
                                 order_type=args.order_type,
                                 slippage_cap=args.slippage_cap,
                                 outside_rth=args.outside_rth)
        if not args.no_report:
            # positions read AFTER the fills → the report shows the resulting book
            _write_report(broker, payload, "execute", fills, report_out, secret,
                          args.account_mode)
        print("\n✓ Rebalance complete.")
    finally:
        broker.disconnect()
    if args.push_report:
        _push_report(report_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
