"""Daily IBKR **paper** rebalancer for the Overall Trading strategy.

Pushes the Overall app's *live-adjusted* target book into an Interactive Brokers
paper account and reconciles it against the account's current positions, once per
trading day.  It reuses the exact same computation the Streamlit app renders — so
the paper account mirrors the "Recommended now (live-adjusted)" allocation you
see on screen — and never re-implements any signal logic:

    results   = overall_core.run_universe()                     # every strategy, live
    opt       = overall_core.optimize_weights(... DEFAULT_PROFILE ...)  # app-default blend
    spot      = overall_core.fetch_spot();  apply_spot(results, spot)
    exits     = overall_core.live_exit_keys(results, spot, include_entries=True)
    gate_live = overall_core.signal_gated_allocation(results, weights, force_exit=exits)
    targets   = gate_live["target"]        # {signal-key: portfolio weight}

The remaining (undeployed) weight is **held as cash** — there is no SATA leg in
the live book, by design.  Sizing uses each traded instrument's own live quote
(so the BTC sleeve is sized on IBIT's price, not spot BTC), IBKR is used only to
read positions / net-liquidation and to place orders, and orders are sequenced
**sells-first** so freed capital funds the buys.

    # see what it WOULD do — places no orders (this is the default):
    python scripts/ibkr_rebalance.py

    # actually place the paper orders:
    python scripts/ibkr_rebalance.py --execute

Safety rails
------------
* ``--dry-run`` is the DEFAULT.  Orders are transmitted only with ``--execute``.
* The connected account MUST be a paper account (id starts with ``DU``) or the
  run aborts — it can never touch a live account unless ``--allow-nonpaper`` is
  passed deliberately.
* A per-name **no-trade band** (``--band``, default 1% of net-liq) suppresses
  churn from tiny drifts.
* Weekends and US market holidays are skipped; a stale signal bar aborts the run.

This script is intentionally standalone and imports nothing from the Streamlit
UI — only ``overall_core`` (the headless engine).  ``ib_async`` is imported
lazily so ``--dry-run`` fully works on a host without it installed / without a
gateway (it prints the intended orders and the reconciliation it *would* do).
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "app"))
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO))

import overall_core as oc                       # noqa: E402  headless engine
import ibkr_symbols as sym                       # noqa: E402
from ibkr_common import (                        # noqa: E402  shared broker/order plumbing
    DEFAULT_PORT, Broker, build_order_plan, is_trading_day, print_plan,
    signal_is_fresh,
)


# ════════════════════════════════════════════════════════════════════════════
# 1. TARGET BOOK — reproduce the app's live-adjusted allocation, headless.
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class TargetBook:
    as_of: str                       # freshest completed signal bar (YYYY-MM-DD)
    weights: dict[str, float]        # signal-key → target portfolio weight (0..1)
    cash_weight: float               # undeployed remainder → held as cash
    exec_price: dict[str, float]     # traded-symbol live price used for sizing
    actions: list[dict]              # the gate's per-asset action rows (for logging)


def compute_target_book(profile: str = None, results: list | None = None) -> TargetBook:
    """Run the full engine and return today's live-adjusted target weights.

    Mirrors ``app/overall_app.py``'s live path exactly: optimise for the chosen
    risk profile (default = the app's ``DEFAULT_PROFILE``), overlay live spot,
    then gate the allocation with the live-exit override
    (``include_entries=True``) so a name whose live price has already broken its
    trend is dropped before we trade it.

    ``results`` — an already-computed ``oc.run_universe()`` output may be passed
    in (the publisher runs the universe once, audits its freshness, and only
    then builds the book from the SAME audited results — no second run that
    could silently differ from what was audited).
    """
    profile = profile or oc.DEFAULT_PROFILE
    prof = oc.RISK_PROFILES.get(profile) or oc.RISK_PROFILES[oc.DEFAULT_PROFILE]
    caps = oc.caps_for(profile)

    if results is None:
        results = oc.run_universe()
    if not results:
        raise RuntimeError("run_universe() returned no instruments — check data feeds")

    rets = oc.returns_matrix(results)
    pos = oc.position_matrix(results, rets.index)
    opt = oc.optimize_weights(rets, caps=caps, pos=pos, sata_daily=oc.SATA_DAILY,
                              mdd_floor=prof["mdd_floor"], objective=prof["objective"],
                              fundamental=True)

    # live spot overlay + live-exit override — identical to the app.
    spot = oc.fetch_spot()
    oc.apply_spot(results, spot)
    live_exits = oc.live_exit_keys(results, spot, include_entries=True)
    gate = oc.signal_gated_allocation(results, opt["optimal"]["weights"],
                                      caps=caps, force_exit=live_exits)

    targets = {k: float(w) for k, w in gate["target"].items() if w and w > 0}

    # Size on the price of the instrument we ACTUALLY trade (IBIT for the BTC
    # sleeve, the ticker itself otherwise) — not the signal's underlying (spot
    # BTC ≈ $100k would mis-size an IBIT position).
    trade_syms = {k: sym.trade_symbol(k) for k in targets if sym.trade_symbol(k)}
    px_raw = oc.fetch_spot({k: s for k, s in trade_syms.items()})
    exec_price = {k: v["price"] for k, v in px_raw.items() if v and v.get("price")}

    as_of = max((str(pd.Timestamp(r["as_of"]).date()) for r in results), default="")
    return TargetBook(as_of=as_of, weights=targets, cash_weight=float(gate["sata"]),
                      exec_price=exec_price, actions=gate["actions"])


# ════════════════════════════════════════════════════════════════════════════
# 2. CLI  (guards, Order, build_order_plan and Broker live in ibkr_common)
# ════════════════════════════════════════════════════════════════════════════
def _print_book(book: TargetBook) -> None:
    print(f"\nTarget book (as-of {book.as_of}, live-adjusted):")
    if not book.weights:
        print("  (fully in cash — no deployed positions today)")
    for k in sorted(book.weights, key=lambda k: -book.weights[k]):
        px = book.exec_price.get(k)
        pxs = f"@ {px:.2f}" if px else "(no price)"
        print(f"  {k:5s} → {sym.trade_symbol(k):5s} {book.weights[k]*100:5.1f}%  {pxs}")
    print(f"  CASH  {book.cash_weight*100:5.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description="Daily IBKR paper rebalancer for the Overall strategy")
    ap.add_argument("--execute", action="store_true",
                    help="actually place orders (default is a no-order dry-run)")
    ap.add_argument("--profile", default=oc.DEFAULT_PROFILE,
                    choices=list(oc.RISK_PROFILES), help="risk profile (default: app default)")
    ap.add_argument("--band", type=float, default=0.01,
                    help="no-trade band as a fraction of net-liq (default 0.01 = 1%%)")
    ap.add_argument("--fractional", action="store_true",
                    help="allow fractional shares (default: whole shares)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT,
                    help=f"IB Gateway API port (default {DEFAULT_PORT} = paper)")
    ap.add_argument("--client-id", type=int, default=17)
    ap.add_argument("--fill-timeout", type=float, default=60.0,
                    help="seconds to wait for each order leg to fill")
    ap.add_argument("--allow-nonpaper", action="store_true",
                    help="permit a non-DU account (DANGEROUS — disables the paper guard)")
    ap.add_argument("--force", action="store_true",
                    help="ignore the weekend/holiday trading-day guard")
    args = ap.parse_args()

    today = pd.Timestamp.now(tz="America/New_York").tz_localize(None)
    ok, why = is_trading_day(today)
    if not ok and not args.force:
        print(f"Not trading: {why} (use --force to override).")
        return 0

    print(f"=== IBKR paper rebalance — profile {args.profile} — {today.date()} ===")
    print("Computing target book (live universe run, ~30–90s)…", flush=True)
    book = compute_target_book(args.profile)

    fresh, fwhy = signal_is_fresh(book.as_of, today)
    _print_book(book)
    if not fresh and not args.force:
        print(f"\nABORT: {fwhy} (use --force to override). No orders placed.")
        return 1
    print(f"Signal freshness: {fwhy}")

    if not args.execute:
        print("\n[dry-run] Connecting only to read positions for the plan preview…")
        try:
            broker = Broker(args.host, args.port, args.client_id,
                        account_mode=("any" if args.allow_nonpaper else "paper"))
        except Exception as e:                   # no gateway on this host → preview vs flat
            print(f"[dry-run] Could not connect to IB Gateway ({e}).")
            print("[dry-run] Showing the target book only; run with a gateway to diff "
                  "against live positions.")
            return 0
        try:
            net_liq = broker.net_liq()
            current = broker.positions_by_key()
            orders = build_order_plan(book.weights, book.exec_price, net_liq,
                                      current, args.band, args.fractional)
            print(f"\nAccount {broker.account} (PAPER).")
            print_plan(orders, net_liq)
            print("\n[dry-run] No orders transmitted. Re-run with --execute to trade.")
        finally:
            broker.disconnect()
        return 0

    # ── execute ───────────────────────────────────────────────────────────────
    broker = Broker(args.host, args.port, args.client_id,
                        account_mode=("any" if args.allow_nonpaper else "paper"))
    try:
        net_liq = broker.net_liq()
        current = broker.positions_by_key()
        orders = build_order_plan(book.weights, book.exec_price, net_liq,
                                  current, args.band, args.fractional)
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
