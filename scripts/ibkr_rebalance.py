"""Daily IBKR **paper** rebalancer for the Overall Trading strategy.

Pushes the Overall app's *live-adjusted* target book into an Interactive Brokers
paper account and reconciles it against the account's current positions, once per
trading day.  It reuses the exact same computation the Streamlit app renders — so
the paper account mirrors the "Recommended now (live-adjusted)" allocation you
see on screen — and never re-implements any signal logic:

    results   = overall_core.run_universe()                     # every strategy, live
    opt       = overall_core.optimize_weights(... Aggressive ...)  # app-default blend
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
sys.path.insert(0, str(_REPO))

import overall_core as oc                       # noqa: E402  headless engine
import ibkr_symbols as sym                       # noqa: E402

# IB Gateway default sockets: paper 4002 / live 4001 (TWS would be 7497 / 7496).
DEFAULT_PORT = 4002
PAPER_ACCT_PREFIX = "DU"          # IBKR paper account ids start with DU
STALE_BAR_DAYS = 4                # abort if the freshest signal bar is older than this

# US equity-market full-day closures (NYSE/Nasdaq).  Extend as needed; the cron
# wrapper already fires only on weekdays, this is the holiday backstop.
US_MARKET_HOLIDAYS = {
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
    "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
}


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


def compute_target_book(profile: str = None) -> TargetBook:
    """Run the full engine and return today's live-adjusted target weights.

    Mirrors ``app/overall_app.py``'s live path exactly: optimise for the chosen
    risk profile (default = the app's ``DEFAULT_PROFILE``), overlay live spot,
    then gate the allocation with the live-exit override
    (``include_entries=True``) so a name whose live price has already broken its
    trend is dropped before we trade it.
    """
    profile = profile or oc.DEFAULT_PROFILE
    prof = oc.RISK_PROFILES.get(profile) or oc.RISK_PROFILES[oc.DEFAULT_PROFILE]
    caps = oc.caps_for(profile)

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
# 2. TRADING-DAY / FRESHNESS GUARDS
# ════════════════════════════════════════════════════════════════════════════
def is_trading_day(day: pd.Timestamp) -> tuple[bool, str]:
    """(tradeable?, reason) — False on weekends and US market holidays."""
    if day.weekday() >= 5:
        return False, f"{day.date()} is a weekend"
    if str(day.date()) in US_MARKET_HOLIDAYS:
        return False, f"{day.date()} is a US market holiday"
    return True, "trading day"


def signal_is_fresh(as_of: str, today: pd.Timestamp) -> tuple[bool, str]:
    """Guard against trading on a stale signal (e.g. a dead data feed left the
    last completed bar days old).  Allows the normal 1–3 day gap over weekends."""
    if not as_of:
        return False, "no signal as-of date"
    age = (today.normalize() - pd.Timestamp(as_of).normalize()).days
    if age > STALE_BAR_DAYS:
        return False, f"signal bar {as_of} is {age}d old (> {STALE_BAR_DAYS}d)"
    return True, f"signal bar {as_of} ({age}d old)"


# ════════════════════════════════════════════════════════════════════════════
# 3. ORDER PLAN — diff target shares vs current IBKR positions.
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class Order:
    key: str
    symbol: str
    action: str            # BUY / SELL
    qty: float             # shares (absolute; whole unless --fractional)
    price: float           # reference price used for sizing / value
    reason: str

    @property
    def value(self) -> float:
        return self.qty * self.price


def build_order_plan(book: TargetBook, net_liq: float,
                     current: dict[str, float], band: float,
                     fractional: bool) -> list[Order]:
    """Turn target weights + current share counts into BUY/SELL orders.

    ``current`` maps signal-key → shares held now (already translated from IBKR
    symbols).  A name held but no longer in the target book is fully closed.  A
    per-name no-trade band (``band`` × net-liq) suppresses tiny rebalances.
    """
    orders: list[Order] = []
    keys = set(book.weights) | set(current)
    band_value = band * net_liq

    for key in sorted(keys):
        symbol = sym.trade_symbol(key)
        if not symbol:
            continue
        price = book.exec_price.get(key)
        w = book.weights.get(key, 0.0)
        held = current.get(key, 0.0)

        if w > 0 and not price:
            # can't size without a price — skip rather than guess (logged upstream)
            continue

        target_shares = (w * net_liq / price) if (w > 0 and price) else 0.0
        if not fractional:
            target_shares = float(round(target_shares))

        # for a full close we may have no fresh price; fall back to avg cost proxy
        px = price or 0.0
        delta = target_shares - held
        delta_value = abs(delta) * px if px else float("inf")

        if abs(delta) < (1e-6 if fractional else 0.5):
            continue                              # already at target (share-level)
        if w > 0 and delta_value < band_value:
            continue                              # inside the no-trade band → leave it

        if delta > 0:
            orders.append(Order(key, symbol, "BUY", abs(delta), px,
                                f"→ {w*100:.1f}% (hold {held:g}→{target_shares:g})"))
        else:
            reason = ("close (no longer in book)" if w <= 0
                      else f"→ {w*100:.1f}% (hold {held:g}→{target_shares:g})")
            orders.append(Order(key, symbol, "SELL", abs(delta), px, reason))

    # sells first (free buying power), then buys; largest value first within each.
    orders.sort(key=lambda o: (0 if o.action == "SELL" else 1, -o.value))
    return orders


# ════════════════════════════════════════════════════════════════════════════
# 4. IBKR SESSION — connect, read account, place orders (lazy ib_async import).
# ════════════════════════════════════════════════════════════════════════════
class Broker:
    """Thin ``ib_async`` wrapper: paper-account guard, positions, net-liq, orders."""

    def __init__(self, host: str, port: int, client_id: int, allow_nonpaper: bool):
        from ib_async import IB
        self.ib = IB()
        self.ib.connect(host, port, clientId=client_id, timeout=30)
        accts = self.ib.managedAccounts()
        self.account = accts[0] if accts else ""
        if not allow_nonpaper and not self.account.startswith(PAPER_ACCT_PREFIX):
            self.ib.disconnect()
            raise RuntimeError(
                f"Connected account '{self.account}' is not a paper account "
                f"(expected '{PAPER_ACCT_PREFIX}…'). Point IB Gateway at the paper "
                f"login, or pass --allow-nonpaper to override (NOT recommended).")

    def net_liq(self) -> float:
        for v in self.ib.accountSummary(self.account):
            if v.tag == "NetLiquidation":
                return float(v.value)
        # fallback: sum of accountValues
        for v in self.ib.accountValues(self.account):
            if v.tag == "NetLiquidation" and v.currency in ("USD", "BASE"):
                return float(v.value)
        raise RuntimeError("could not read NetLiquidation from the account")

    def positions_by_key(self) -> dict[str, float]:
        """Current holdings as signal-key → shares (foreign symbols ignored)."""
        out: dict[str, float] = {}
        for p in self.ib.positions(self.account):
            key = sym.key_for_symbol(p.contract.symbol)
            if key is not None:
                out[key] = out.get(key, 0.0) + float(p.position)
        return out

    def place(self, orders: list[Order], fractional: bool, wait: float) -> None:
        """Transmit market orders, sells first, waiting for each leg to fill."""
        from ib_async import MarketOrder
        sells = [o for o in orders if o.action == "SELL"]
        buys = [o for o in orders if o.action == "BUY"]
        for leg_name, legs in (("SELL", sells), ("BUY", buys)):
            trades = []
            for o in legs:
                contract = sym.ibkr_contract(o.symbol)
                self.ib.qualifyContracts(contract)
                qty = o.qty if fractional else float(round(o.qty))
                if qty <= 0:
                    continue
                order = MarketOrder(o.action, qty)
                trades.append((o, self.ib.placeOrder(contract, order)))
                print(f"    sent  {o.action:4s} {qty:g} {o.symbol}")
            # wait for this leg to fill before starting the next (buys need cash)
            deadline = wait
            while deadline > 0 and any(not t.isDone() for _, t in trades):
                self.ib.sleep(1.0)
                deadline -= 1.0
            for o, t in trades:
                st = t.orderStatus.status
                filled = t.orderStatus.filled
                print(f"    {leg_name} {o.symbol}: {st} filled={filled:g}")

    def disconnect(self) -> None:
        try:
            self.ib.disconnect()
        except Exception:
            pass


# ════════════════════════════════════════════════════════════════════════════
# 5. CLI
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


def _print_plan(orders: list[Order], net_liq: float) -> None:
    print(f"\nOrder plan (net-liq ${net_liq:,.0f}):")
    if not orders:
        print("  ✓ already within the no-trade band — nothing to do")
        return
    for o in orders:
        print(f"  {o.action:4s} {o.qty:8g} {o.symbol:5s} "
              f"~${o.value:12,.0f}   {o.reason}")


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
            broker = Broker(args.host, args.port, args.client_id, args.allow_nonpaper)
        except Exception as e:                   # no gateway on this host → preview vs flat
            print(f"[dry-run] Could not connect to IB Gateway ({e}).")
            print("[dry-run] Showing the target book only; run with a gateway to diff "
                  "against live positions.")
            return 0
        try:
            net_liq = broker.net_liq()
            current = broker.positions_by_key()
            orders = build_order_plan(book, net_liq, current, args.band, args.fractional)
            print(f"\nAccount {broker.account} (PAPER).")
            _print_plan(orders, net_liq)
            print("\n[dry-run] No orders transmitted. Re-run with --execute to trade.")
        finally:
            broker.disconnect()
        return 0

    # ── execute ───────────────────────────────────────────────────────────────
    broker = Broker(args.host, args.port, args.client_id, args.allow_nonpaper)
    try:
        net_liq = broker.net_liq()
        current = broker.positions_by_key()
        orders = build_order_plan(book, net_liq, current, args.band, args.fractional)
        print(f"\nAccount {broker.account} (PAPER).")
        _print_plan(orders, net_liq)
        if orders:
            print("\nTransmitting orders (sells → buys)…")
            broker.place(orders, args.fractional, args.fill_timeout)
        print("\n✓ Rebalance complete.")
    finally:
        broker.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
