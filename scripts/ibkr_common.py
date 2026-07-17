"""Shared IBKR plumbing for the paper-trading tools.

Both entry points depend on this:

* ``ibkr_rebalance.py`` — the all-in-one rebalancer (Option A): runs the engine
  AND trades on the same host.
* ``ibkr_execute_book.py`` — the Option-C executor: trades a *pre-computed*
  target book (published by the cloud side) without running the model locally.

Keeping the order-diff, the ``ib_async`` session wrapper and the trading-day /
freshness guards here means there is exactly ONE implementation of "turn target
weights into orders and place them", so the two entry points can never drift.

Nothing here imports ``overall_core`` or the model stack — this module is
deliberately lightweight so the executor can run on a minimal box next to IB
Gateway.  ``ib_async`` is imported lazily inside :class:`Broker` /
:func:`ibkr_contract` so dry-runs work on a host without it installed.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import ibkr_symbols as sym

# IB Gateway default sockets: paper 4002 / live 4001 (TWS would be 7497 / 7496).
DEFAULT_PORT = 4002
PAPER_ACCT_PREFIX = "DU"          # IBKR paper account ids start with DU
STALE_BAR_DAYS = 4                # abort if the freshest signal bar is older than this

# US equity-market full-day closures (NYSE/Nasdaq).  Extend as needed; the cron
# wrapper / workflow already fire only on weekdays — this is the holiday backstop.
US_MARKET_HOLIDAYS = {
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
    "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
}


# ════════════════════════════════════════════════════════════════════════════
# TRADING-DAY / FRESHNESS GUARDS
# ════════════════════════════════════════════════════════════════════════════
def is_trading_day(day: pd.Timestamp) -> tuple[bool, str]:
    """(tradeable?, reason) — False on weekends and US market holidays."""
    if day.weekday() >= 5:
        return False, f"{day.date()} is a weekend"
    if str(day.date()) in US_MARKET_HOLIDAYS:
        return False, f"{day.date()} is a US market holiday"
    return True, "trading day"


def signal_is_fresh(as_of: str, today: pd.Timestamp,
                    max_days: int = STALE_BAR_DAYS) -> tuple[bool, str]:
    """Guard against trading on a stale signal (e.g. a dead data feed left the
    last completed bar days old).  Allows the normal 1–3 day gap over weekends."""
    if not as_of:
        return False, "no signal as-of date"
    age = (today.normalize() - pd.Timestamp(as_of).normalize()).days
    if age > max_days:
        return False, f"signal bar {as_of} is {age}d old (> {max_days}d)"
    return True, f"signal bar {as_of} ({age}d old)"


# ════════════════════════════════════════════════════════════════════════════
# ORDER PLAN — diff target shares vs current IBKR positions.
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class Order:
    key: str
    symbol: str
    action: str            # BUY / SELL
    qty: float             # shares (absolute; whole unless fractional)
    price: float           # reference price used for sizing / value
    reason: str

    @property
    def value(self) -> float:
        return self.qty * self.price


def build_order_plan(weights: dict[str, float], exec_price: dict[str, float],
                     net_liq: float, current: dict[str, float], band: float,
                     fractional: bool, max_order_notional: float = 0.0) -> list[Order]:
    """Turn target weights + current share counts into BUY/SELL orders.

    ``weights``    signal-key → target portfolio weight (0..1).
    ``exec_price`` signal-key → price of the instrument we actually trade
                   (IBIT for the BTC sleeve, the ticker itself otherwise).
    ``current``    signal-key → shares held now (already translated from IBKR
                   symbols).  A name held but not in ``weights`` is fully closed.
    A per-name no-trade band (``band`` × net-liq) suppresses tiny rebalances.
    ``max_order_notional`` (>0) clamps any single order's size to that dollar cap
    — a fat-finger / bug backstop that matters most for live trading.
    """
    orders: list[Order] = []
    keys = set(weights) | set(current)
    band_value = band * net_liq

    for key in sorted(keys):
        symbol = sym.trade_symbol(key)
        if not symbol:
            continue
        price = exec_price.get(key)
        w = weights.get(key, 0.0)
        held = current.get(key, 0.0)

        if w > 0 and not price:
            # can't size a new/updated target without a price — skip rather than
            # guess (the caller logs which names were dropped for lack of price).
            continue

        target_shares = (w * net_liq / price) if (w > 0 and price) else 0.0
        if not fractional:
            target_shares = float(round(target_shares))

        px = price or 0.0                       # a full close may have no price
        delta = target_shares - held
        delta_value = abs(delta) * px if px else float("inf")

        if abs(delta) < (1e-6 if fractional else 0.5):
            continue                            # already at target (share-level)
        if w > 0 and delta_value < band_value:
            continue                            # inside the no-trade band → leave it

        order_qty = abs(delta)
        clamped = ""
        if max_order_notional and px and order_qty * px > max_order_notional:
            cap_qty = max_order_notional / px
            order_qty = cap_qty if fractional else float(int(cap_qty))
            clamped = f" [clamped to ${max_order_notional:,.0f}]"
            if order_qty < (1e-6 if fractional else 0.5):
                continue                        # cap too small to place a share

        if delta > 0:
            orders.append(Order(key, symbol, "BUY", order_qty, px,
                                f"→ {w*100:.1f}% (hold {held:g}→{target_shares:g}){clamped}"))
        else:
            reason = ("close (no longer in book)" if w <= 0
                      else f"→ {w*100:.1f}% (hold {held:g}→{target_shares:g})")
            orders.append(Order(key, symbol, "SELL", order_qty, px, reason + clamped))

    # sells first (free buying power), then buys; largest value first within each.
    orders.sort(key=lambda o: (0 if o.action == "SELL" else 1, -o.value))
    return orders


def print_plan(orders: list[Order], net_liq: float) -> None:
    print(f"\nOrder plan (net-liq ${net_liq:,.0f}):")
    if not orders:
        print("  ✓ already within the no-trade band — nothing to do")
        return
    for o in orders:
        print(f"  {o.action:4s} {o.qty:8g} {o.symbol:5s} "
              f"~${o.value:12,.0f}   {o.reason}")


# ════════════════════════════════════════════════════════════════════════════
# IBKR SESSION — connect, read account, place orders (lazy ib_async import).
# ════════════════════════════════════════════════════════════════════════════
class Broker:
    """Thin ``ib_async`` wrapper: paper-account guard, positions, net-liq, orders."""

    def __init__(self, host: str, port: int, client_id: int,
                 account_mode: str = "paper", expected_account: str | None = None,
                 confirm_live: bool = False):
        """Connect and enforce the account guard for the chosen mode.

        account_mode:
          * ``paper`` — the connected account MUST start with ``DU`` (default).
          * ``live``  — the account must NOT be paper; additionally requires
            ``confirm_live=True`` and, if ``expected_account`` is given, an EXACT
            match, so a real account can never be traded by accident or the wrong
            real account traded by mistake.
          * ``any``   — no check (escape hatch; not used by the executor).
        """
        from ib_async import IB
        self.ib = IB()
        self.ib.connect(host, port, clientId=client_id, timeout=30)
        accts = self.ib.managedAccounts()
        self.account = accts[0] if accts else ""
        self.account_mode = account_mode
        self._enforce_account_guard(account_mode, expected_account, confirm_live)

    def _enforce_account_guard(self, mode: str, expected: str | None,
                               confirm_live: bool) -> None:
        acct = self.account
        is_paper = acct.startswith(PAPER_ACCT_PREFIX)
        if mode == "any":
            return
        if mode == "paper":
            if not is_paper:
                self._abort(f"account '{acct}' is not a paper account "
                            f"('{PAPER_ACCT_PREFIX}…'). Point IB Gateway at the paper "
                            f"login, or run with --account-mode live for a real account.")
            return
        if mode == "live":
            if not confirm_live:
                self._abort("live mode requires --confirm-live (you are about to "
                            "trade REAL money).")
            if is_paper:
                self._abort(f"--account-mode live but the connected account '{acct}' "
                            f"is a PAPER account. Point the gateway at the live login.")
            if expected and acct != expected:
                self._abort(f"connected account '{acct}' does not match the expected "
                            f"live account '{expected}'. Refusing to trade the wrong account.")
            return
        self._abort(f"unknown account mode {mode!r}")

    def _abort(self, msg: str) -> None:
        try:
            self.ib.disconnect()
        except Exception:
            pass
        raise RuntimeError(msg)

    def net_liq(self) -> float:
        for v in self.ib.accountSummary(self.account):
            if v.tag == "NetLiquidation":
                return float(v.value)
        for v in self.ib.accountValues(self.account):
            if v.tag == "NetLiquidation" and v.currency in ("USD", "BASE"):
                return float(v.value)
        raise RuntimeError("could not read NetLiquidation from the account")

    def cash(self) -> float:
        """Total cash value (best-effort; 0.0 if the tag isn't present)."""
        for v in self.ib.accountSummary(self.account):
            if v.tag == "TotalCashValue":
                return float(v.value)
        return 0.0

    def portfolio_snapshot(self) -> list[dict]:
        """Current holdings with cost/market data, one dict per position.

        Uses ``ib.portfolio()`` (richer than ``positions()``): shares, average
        cost, and — when the paper account has market data — market price/value
        and unrealised P&L. Fields that IBKR can't supply come back as 0.0, which
        the UI renders as “—”. Foreign symbols keep ``key=None``."""
        out = []
        for item in self.ib.portfolio(self.account):
            c = item.contract
            out.append(dict(
                key=sym.key_for_symbol(c.symbol), symbol=c.symbol,
                shares=float(getattr(item, "position", 0.0) or 0.0),
                avg_cost=float(getattr(item, "averageCost", 0.0) or 0.0),
                market_price=float(getattr(item, "marketPrice", 0.0) or 0.0),
                market_value=float(getattr(item, "marketValue", 0.0) or 0.0),
                unrealized_pnl=float(getattr(item, "unrealizedPNL", 0.0) or 0.0)))
        return out

    def positions_by_key(self) -> dict[str, float]:
        """Current holdings as signal-key → shares (foreign symbols ignored)."""
        out: dict[str, float] = {}
        for p in self.ib.positions(self.account):
            key = sym.key_for_symbol(p.contract.symbol)
            if key is not None:
                out[key] = out.get(key, 0.0) + float(p.position)
        return out

    def place(self, orders: list[Order], fractional: bool, wait: float) -> list[dict]:
        """Transmit market orders, sells first, waiting for each leg to fill.

        Returns one result dict per transmitted order (action, symbol, qty, the
        reference price, final status, filled quantity and average fill price) so
        the caller can build the execution report."""
        from ib_async import MarketOrder
        results: list[dict] = []
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
                trades.append((o, qty, self.ib.placeOrder(contract, order)))
                print(f"    sent  {o.action:4s} {qty:g} {o.symbol}")
            deadline = wait                      # wait for this leg before the next
            while deadline > 0 and any(not t.isDone() for _, _, t in trades):
                self.ib.sleep(1.0)
                deadline -= 1.0
            for o, qty, t in trades:
                st = t.orderStatus.status
                filled = float(t.orderStatus.filled or 0.0)
                avg = float(t.orderStatus.avgFillPrice or 0.0)
                print(f"    {leg_name} {o.symbol}: {st} filled={filled:g}")
                results.append(dict(key=o.key, symbol=o.symbol, action=o.action,
                                    qty=float(qty), price=float(o.price), status=st,
                                    filled=filled, avg_fill_price=avg, reason=o.reason))
        return results

    def disconnect(self) -> None:
        try:
            self.ib.disconnect()
        except Exception:
            pass
