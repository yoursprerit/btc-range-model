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

import math
from dataclasses import dataclass

import pandas as pd

import ibkr_symbols as sym

# IB Gateway default sockets: paper 4002 / live 4001 (TWS would be 7497 / 7496).
DEFAULT_PORT = 4002
PAPER_ACCT_PREFIX = "DU"          # IBKR paper account ids start with DU
STALE_BAR_DAYS = 4                # abort if the freshest signal bar is older than this

# ── order types ─────────────────────────────────────────────────────────────
# MARKETABLE_LIMIT (default): a LIMIT priced THROUGH the touch by
# ``slippage_cap`` — it fills like a market order under normal conditions but
# caps what a stale quote, a flash dislocation or a bad print can cost.  A naked
# MARKET order has no such ceiling, which is the only reason to prefer the limit.
# MOC: market-on-close, filled in the official closing auction — the venue's
# deepest liquidity, and the price the engine/backtest actually books against
# (both assume a close fill), so it removes the structural drift between live
# results and every backtest number in the repo.  It cannot be used after the
# exchange's entry cutoff, and nothing fills until 4:00 PM ET.
ORDER_MARKET = "market"
ORDER_MARKETABLE_LIMIT = "marketable-limit"
ORDER_MOC = "moc"
ORDER_TYPES = (ORDER_MARKETABLE_LIMIT, ORDER_MOC, ORDER_MARKET)

# How far THROUGH the touch a marketable limit is priced. 0.5% clears the spread
# on everything in the traded universe (including the leveraged/thin sleeves —
# MSTU, NUGT, SOXL, ERX, WGMI, REMX, ARTY, SATA) while still refusing to chase a
# quote that has blown out far past it.
DEFAULT_SLIPPAGE_CAP = 0.005

# NYSE/Nasdaq stop accepting MOC entries at 15:50 ET; the auction prints at
# 16:00 ET.  The executor's 2:30-PM-CT (3:30 PM ET) slot sits 20 minutes ahead
# of the cutoff — which is precisely what makes MOC usable at all (it was not,
# at the old 8:45-AM-CT slot).
MOC_CUTOFF_ET = (15, 50)
US_CLOSE_ET = (16, 0)
ET_TZ = "America/New_York"
# After the auction prints, give IBKR a moment to report the fills back.
MOC_CLOSE_BUFFER_S = 120.0
# Hard ceiling on the post-submit wait, so a stuck run can never hang forever.
MOC_MAX_WAIT_S = 2400.0

# US equity-market full-day closures (NYSE/Nasdaq).  Extend as needed.  The
# publisher workflow runs EVERY day (BTC trades on weekends/holidays too), so
# this weekend+holiday guard on the EXECUTOR side is what keeps order placement
# restricted to US market days.
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


# ════════════════════════════════════════════════════════════════════════════
# ORDER PRICING / ROUTING — pure helpers (no ib_async), so they are unit-tested
# without a broker connection.
# ════════════════════════════════════════════════════════════════════════════
def _finite(x) -> float | None:
    """A usable positive price, or None (IBKR reports absent fields as nan/-1)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v <= 0:
        return None
    return v


def round_to_tick(price: float, action: str, min_tick: float = 0.01) -> float:
    """Round a limit onto the venue's tick grid, always in the MARKETABLE
    direction (up for a BUY, down for a SELL) so rounding can never pull the
    limit back inside the spread and turn a marketable order passive."""
    steps = price / min_tick
    steps = (math.ceil(steps - 1e-9) if action == "BUY"
             else math.floor(steps + 1e-9))
    return round(max(steps, 1) * min_tick, 4)


def marketable_limit_price(action: str, quote: dict,
                           cap: float = DEFAULT_SLIPPAGE_CAP
                           ) -> tuple[float | None, str]:
    """(limit price, basis) for a marketable limit — or (None, why) when the
    quote is unusable and the caller must fall back.

    A BUY prices off the ASK and a SELL off the BID (the side we must cross),
    falling back to last, then the prior close.  ``cap`` is then applied
    THROUGH that reference — a BUY limit sits ``cap`` above the ask — so the
    order crosses immediately in a normal book but stops dead if the market has
    run further than ``cap`` away.  That ceiling is the whole point: it is the
    difference between paying the spread and paying whatever a broken quote
    happens to print.

    The ``close`` fallback is the prior session's close, so on a gap day a limit
    derived from it can land unmarketable — acceptable only because
    :meth:`Broker._escalate_unfilled` re-sends anything that fails to fill as a
    market order in the same run.  On a box with delayed-only market data,
    widen ``cap`` so the first attempt still crosses.
    """
    if action not in ("BUY", "SELL"):
        return None, f"unknown action {action!r}"
    if not (0 < cap < 1):
        return None, f"slippage cap {cap} out of range (0 < cap < 1)"
    touch = "ask" if action == "BUY" else "bid"
    for field in (touch, "last", "close"):
        ref = _finite(quote.get(field))
        if ref is None:
            continue
        raw = ref * (1 + cap) if action == "BUY" else ref * (1 - cap)
        # sub-$1 names quote in hundredths of a cent; everything in this
        # universe is far above that, but the grid still has to be right.
        tick = 0.01 if ref >= 1.0 else 0.0001
        return round_to_tick(raw, action, tick), field
    return None, "no usable bid/ask/last/close"


def unfilled_remainder(qty: float, filled: float, fractional: bool = False) -> float:
    """Shares still outstanding when a marketable limit's wait expired.

    Used to escalate the leftover to a market order: a limit that does not fill
    leaves the account holding the WRONG exposure until the next daily run,
    which is a far worse outcome than paying the spread.  Escalating means the
    price ceiling costs nothing in fill certainty — it only ever caps the
    FIRST attempt.  Whole-share accounts round the remainder down, so a
    rounding crumb never becomes a 1-share order."""
    rem = float(qty) - float(filled or 0.0)
    if not fractional:
        rem = float(math.floor(rem + 1e-9))
    return rem if rem > 0 else 0.0


def _et_now(now=None) -> pd.Timestamp:
    n = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if n.tzinfo is None:
        n = n.tz_localize("UTC")
    return n.tz_convert(ET_TZ)


def moc_entry_open(now=None, cutoff: tuple[int, int] = MOC_CUTOFF_ET
                   ) -> tuple[bool, str]:
    """(accepting?, reason) — can a MOC order still be entered right now?

    Past the exchange cutoff a MOC is rejected outright, so the caller must
    route around it rather than discover the rejection order by order."""
    et = _et_now(now)
    edge = et.normalize() + pd.Timedelta(hours=cutoff[0], minutes=cutoff[1])
    stamp = et.strftime("%H:%M ET")
    if et > edge:
        return False, (f"{stamp} is past the {cutoff[0]:02d}:{cutoff[1]:02d} ET "
                       f"MOC entry cutoff")
    return True, f"{stamp}, before the {cutoff[0]:02d}:{cutoff[1]:02d} ET MOC cutoff"


def moc_wait_seconds(now=None, buffer_s: float = MOC_CLOSE_BUFFER_S,
                     max_wait_s: float = MOC_MAX_WAIT_S) -> float:
    """How long to wait for MOC fills: until the 4:00 PM ET auction prints plus
    a reporting buffer, clamped to ``max_wait_s``.

    MOC is the one order type whose fills are unknowable at submit time, so the
    executor has to hold the connection open to write a truthful execution
    report instead of one that says "Submitted" for every leg."""
    et = _et_now(now)
    close = et.normalize() + pd.Timedelta(hours=US_CLOSE_ET[0],
                                          minutes=US_CLOSE_ET[1])
    secs = (close - et).total_seconds() + buffer_s
    return float(min(max(secs, 0.0), max_wait_s))


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
                 confirm_live: bool = False, market_data_type: int = 1):
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
        # 1 = live (default), 2 = frozen, 3 = delayed, 4 = delayed-frozen.
        # Delayed data is free and needs no subscription, but IBKR only serves
        # it when the client asks — without this call an unsubscribed account
        # gets NO quote at all and every marketable limit degrades to a market
        # order. Left at 1 unless the caller opts in, so a subscribed account is
        # unaffected.
        if int(market_data_type) != 1:
            self.ib.reqMarketDataType(int(market_data_type))
            print(f"  market data type {market_data_type} requested "
                  f"({'delayed' if int(market_data_type) in (3, 4) else 'frozen'})")
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

    def day_fills_by_symbol(self) -> list[dict]:
        """Today's executions aggregated per symbol, in the shape the execution
        report's ``trades`` list expects.

        Read back from the broker rather than from what this process happened to
        send, so a report written after the fact reflects what ACTUALLY filled --
        including remainders that printed long after the sending run's
        fill-timeout expired (routine outside RTH, where a limit can sit working
        for hours).
        """
        agg: dict[str, dict] = {}
        try:
            fills = self.ib.fills()
        except Exception as e:                       # pragma: no cover
            print(f"    could not read fills: {e}")
            return []
        for f in fills:
            ex = getattr(f, "execution", None)
            con = getattr(f, "contract", None)
            if ex is None or con is None:
                continue
            symbol = getattr(con, "symbol", "") or ""
            shares = float(getattr(ex, "shares", 0.0) or 0.0)
            price = float(getattr(ex, "price", 0.0) or 0.0)
            side = "BUY" if str(getattr(ex, "side", "")).upper().startswith("B") else "SELL"
            a = agg.setdefault(f"{symbol}:{side}", {
                "key": sym.key_for_symbol(symbol) or symbol, "symbol": symbol,
                "action": side, "qty": 0.0, "price": 0.0, "status": "Filled",
                "filled": 0.0, "avg_fill_price": 0.0, "order_type": "",
                "limit_price": 0.0, "reason": "reconstructed from broker fills",
                "_notional": 0.0})
            a["filled"] += shares
            a["_notional"] += shares * price
        out = []
        for a in agg.values():
            n = a.pop("_notional")
            a["avg_fill_price"] = (n / a["filled"]) if a["filled"] else 0.0
            a["qty"] = a["filled"]
            a["price"] = a["avg_fill_price"]
            out.append(a)
        return sorted(out, key=lambda t: -abs(t["qty"] * t["price"]))

    def positions_by_key(self) -> dict[str, float]:
        """Current holdings as signal-key → shares (foreign symbols ignored)."""
        out: dict[str, float] = {}
        for p in self.ib.positions(self.account):
            key = sym.key_for_symbol(p.contract.symbol)
            if key is not None:
                out[key] = out.get(key, 0.0) + float(p.position)
        return out

    def quote(self, contract) -> dict:
        """Snapshot bid/ask/last/close for a qualified contract (empty on any
        failure — a missing market-data subscription must degrade, not raise)."""
        try:
            tickers = self.ib.reqTickers(contract)
        except Exception as e:
            print(f"    quote {contract.symbol}: unavailable ({e})")
            return {}
        if not tickers:
            return {}
        t = tickers[0]
        return {"bid": getattr(t, "bid", None), "ask": getattr(t, "ask", None),
                "last": getattr(t, "last", None), "close": getattr(t, "close", None)}

    def _build_order(self, o: Order, qty: float, order_type: str,
                     slippage_cap: float, contract, outside_rth: bool = False):
        """(ib_async order, type actually used, limit price or 0.0).

        Falls back to a plain MARKET order when a marketable limit cannot be
        priced (no market-data subscription, or a dead quote).  That preference
        is deliberate: an unpriceable limit that never fills leaves the account
        holding the WRONG exposure until the next run — an unbounded tracking
        error — whereas market slippage in this universe is bounded and small.

        ``outside_rth`` flips both halves of that reasoning, because IBKR does
        not accept a MARKET order outside regular hours at all — it is rejected,
        not merely risky.  So an extended-hours run stamps ``outsideRth`` on a
        LIMIT and, when no quote can be had, prices off the book's own
        publish-time ``exec_price`` rather than degrading to an order the venue
        will refuse.  A leg that cannot be priced even then is SKIPPED and
        reported, never sent blind.
        """
        from ib_async import LimitOrder, MarketOrder, Order as IBOrder
        if order_type == ORDER_MOC:
            return IBOrder(action=o.action, totalQuantity=qty,
                           orderType="MOC", tif="DAY"), ORDER_MOC, 0.0
        if order_type == ORDER_MARKETABLE_LIMIT:
            lmt, basis = marketable_limit_price(o.action, self.quote(contract),
                                                slippage_cap)
            if lmt is None and outside_rth:
                # Extended hours with no live quote: the book's own exec_price
                # is a real, recent print for this symbol and beats sending
                # nothing. Widen the cap yourself if it fails to cross.
                lmt, basis = marketable_limit_price(
                    o.action, {"last": o.price}, slippage_cap)
                if lmt is not None:
                    basis = "book exec_price"
            if lmt is not None:
                order = LimitOrder(o.action, qty, lmt)
                if outside_rth:
                    order.outsideRth = True
                print(f"    limit {o.symbol}: {o.action} ≤ ${lmt:,.4f} "
                      f"({basis} {slippage_cap*100:.2f}% through)"
                      + (" [outside RTH]" if outside_rth else ""))
                return order, ORDER_MARKETABLE_LIMIT, float(lmt)
            if outside_rth:
                print(f"    SKIP {o.symbol}: {basis}, and a MARKET order is "
                      f"rejected outside RTH — nothing sent for this leg")
                return None, ORDER_MARKETABLE_LIMIT, 0.0
            print(f"    WARN {o.symbol}: {basis} — falling back to MARKET "
                  f"(no price ceiling on this leg)")
        if outside_rth:
            print(f"    SKIP {o.symbol}: MARKET orders are rejected outside RTH")
            return None, ORDER_MARKET, 0.0
        return MarketOrder(o.action, qty), ORDER_MARKET, 0.0

    def _resolve_order_type(self, order_type: str, fractional: bool,
                            now=None, outside_rth: bool = False) -> tuple[str, bool]:
        """(effective order type, fractional) after the routing guards.

        MOC is refused after the exchange cutoff (it would simply be rejected)
        and cannot carry fractional quantities, so both fall back rather than
        fail the run."""
        if order_type not in ORDER_TYPES:
            raise ValueError(f"unknown order type {order_type!r} "
                             f"(want one of {', '.join(ORDER_TYPES)})")
        if outside_rth and order_type != ORDER_MARKETABLE_LIMIT:
            # Neither MOC (there is no auction) nor MARKET (rejected by the
            # venue) means anything outside regular hours.
            print(f"  {order_type} is unavailable outside RTH — using "
                  f"{ORDER_MARKETABLE_LIMIT}.")
            return ORDER_MARKETABLE_LIMIT, fractional
        if order_type != ORDER_MOC:
            return order_type, fractional
        ok, why = moc_entry_open(now)
        if not ok:
            print(f"  MOC unavailable: {why} — falling back to "
                  f"{ORDER_MARKETABLE_LIMIT}.")
            return ORDER_MARKETABLE_LIMIT, fractional
        if fractional:
            print("  MOC does not accept fractional quantities — rounding to "
                  "whole shares for this run.")
            fractional = False
        print(f"  MOC routing: {why}; fills print in the 4:00 PM ET auction.")
        return ORDER_MOC, fractional

    def _escalate_unfilled(self, trades: list, fractional: bool,
                           wait: float, outside_rth: bool = False) -> list:
        """Cancel any limit that did not fill in time and re-send the remainder
        as a MARKET order; returns the new trade tuples to report alongside.

        This is what makes the marketable limit strictly better than a bare
        market order rather than a trade-off: the limit caps the first attempt,
        and anything it fails to fill still gets done in the same run instead
        of leaving the book mismatched until tomorrow.
        """
        from ib_async import MarketOrder
        if outside_rth:
            # The market-order safety net does not exist outside RTH; an
            # unfilled extended-hours limit simply stays unfilled.
            stuck = [o.symbol for o, _q, used, _l, t in trades
                     if not t.isDone() and used == ORDER_MARKETABLE_LIMIT]
            if stuck:
                print(f"    outside RTH: leaving {len(stuck)} unfilled "
                      f"limit(s) working ({', '.join(stuck)}) — no MARKET "
                      f"escalation is possible")
            return []
        extra = []
        for o, qty, used, _lmt, t in trades:
            if t.isDone() or used != ORDER_MARKETABLE_LIMIT:
                continue
            rem = unfilled_remainder(qty, t.orderStatus.filled, fractional)
            try:
                self.ib.cancelOrder(t.order)
            except Exception as e:
                print(f"    WARN {o.symbol}: cancel failed ({e}) — not escalating")
                continue
            self.ib.sleep(1.0)
            if rem <= 0:
                continue
            print(f"    ESCALATE {o.symbol}: limit unfilled after {wait:g}s — "
                  f"re-sending {rem:g} as MARKET")
            contract = sym.ibkr_contract(o.symbol)
            self.ib.qualifyContracts(contract)
            esc = Order(o.key, o.symbol, o.action, rem, o.price,
                        f"{o.reason} (escalated from unfilled limit)")
            extra.append((esc, rem, ORDER_MARKET, 0.0,
                          self.ib.placeOrder(contract, MarketOrder(o.action, rem))))
        if extra:
            deadline = wait
            while deadline > 0 and any(not t.isDone() for *_, t in extra):
                self.ib.sleep(1.0)
                deadline -= 1.0
        return extra

    def place(self, orders: list[Order], fractional: bool, wait: float,
              order_type: str = ORDER_MARKETABLE_LIMIT,
              slippage_cap: float = DEFAULT_SLIPPAGE_CAP,
              now=None, outside_rth: bool = False) -> list[dict]:
        """Transmit the order plan, sells first, waiting for fills.

        ``order_type`` is one of :data:`ORDER_TYPES`; see the module header for
        why the marketable limit is the default and what MOC buys.  Returns one
        result dict per transmitted order (action, symbol, qty, reference price,
        the order type actually used and its limit, final status, filled
        quantity and average fill price) so the caller can build the execution
        report.
        """
        order_type, fractional = self._resolve_order_type(order_type, fractional,
                                                          now, outside_rth)
        results: list[dict] = []
        sells = [o for o in orders if o.action == "SELL"]
        buys = [o for o in orders if o.action == "BUY"]

        # MOC legs all print in the SAME auction, so sequencing sells before
        # buys buys nothing — it would only burn minutes off the entry cutoff.
        # Submit everything, then wait once for the close.
        legs = ([("MOC", sells + buys)] if order_type == ORDER_MOC
                else [("SELL", sells), ("BUY", buys)])

        for leg_name, leg_orders in legs:
            trades = []
            for o in leg_orders:
                contract = sym.ibkr_contract(o.symbol)
                self.ib.qualifyContracts(contract)
                qty = o.qty if fractional else float(round(o.qty))
                if qty <= 0:
                    continue
                order, used, lmt = self._build_order(o, qty, order_type,
                                                     slippage_cap, contract,
                                                     outside_rth)
                if order is None:            # unsendable outside RTH — skipped
                    continue
                trades.append((o, qty, used, lmt, self.ib.placeOrder(contract, order)))
                print(f"    sent  {o.action:4s} {qty:g} {o.symbol} [{used}]"
                      + (" outsideRth" if outside_rth else ""))
            if not trades:
                continue
            if order_type == ORDER_MOC:
                deadline = moc_wait_seconds(now)
                print(f"    waiting {deadline/60:.0f} min for the closing "
                      f"auction to print…")
            else:
                deadline = wait              # wait for this leg before the next
            while deadline > 0 and any(not t.isDone() for *_, t in trades):
                self.ib.sleep(1.0)
                deadline -= 1.0
            if order_type == ORDER_MARKETABLE_LIMIT:
                trades += self._escalate_unfilled(trades, fractional, wait,
                                                  outside_rth)
            for o, qty, used, lmt, t in trades:
                st = t.orderStatus.status
                filled = float(t.orderStatus.filled or 0.0)
                avg = float(t.orderStatus.avgFillPrice or 0.0)
                print(f"    {leg_name} {o.symbol}: {st} filled={filled:g}")
                results.append(dict(key=o.key, symbol=o.symbol, action=o.action,
                                    qty=float(qty), price=float(o.price), status=st,
                                    filled=filled, avg_fill_price=avg,
                                    order_type=used, limit_price=lmt,
                                    reason=o.reason))
        return results

    def disconnect(self) -> None:
        try:
            self.ib.disconnect()
        except Exception:
            pass
