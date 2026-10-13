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
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import ibkr_safety as safety
import ibkr_symbols as sym

# IB Gateway default sockets: paper 4002 / live 4001 (TWS would be 7497 / 7496).
DEFAULT_PORT = 4002
DEFAULT_LIVE_PORT = 4001
PAPER_ACCT_PREFIX = "DU"          # IBKR paper account ids start with DU
STALE_BAR_DAYS = 4                # abort if the freshest signal bar is older than this

# ── exposure / funding guards ───────────────────────────────────────────────
# The executor's book never asks for more than 100% of net liquidation, so any
# plan that would end above ~1x NAV is a bug in the inputs (a bad positions
# read, a duplicate run), not a decision. 1.02 leaves room for whole-share
# rounding and a fill printing through the sizing price.
DEFAULT_MAX_GROSS_FRAC = 1.02
# Per-order sanity limits (independent of the book's own weights). A single BUY
# larger than this share of net liquidation, or a plan with more orders than the
# universe could need, means a wrong price, a wrong NAV or a wrong plan -- not a
# decision. SELLs are bounded separately: never more than is held.
DEFAULT_MAX_BUY_FRAC = 0.40
DEFAULT_MAX_ORDERS = 25
# How far the PC clock may be from IBKR's server clock. The MOC cutoff, the
# trading-day test and the session test all read the PC clock.
MAX_CLOCK_SKEW_S = 30.0
# Buys are funded from settled cash + the proceeds the sell leg actually
# realised. The buffer keeps a fill printing above its limit from tipping the
# account into a margin loan.
FUNDING_BUFFER = 0.01
# How long to wait for IBKR to deliver the position subscription before
# concluding the account really is flat.
POSITION_SETTLE_TRIES = 5
POSITION_SETTLE_SLEEP_S = 1.0
# A positions read is untrustworthy when the account reports materially more
# gross position value than the read accounts for (tolerance as a fraction of
# net liquidation).
POSITION_READ_TOL = 0.02
# One login serving several accounts to several client processes at once can be
# slow to deliver the per-account "account updates" (the prices behind
# ``portfolio()``). The positions read is then compared against an EMPTY
# portfolio and refused. Re-request it a few times before giving up, and as a
# last resort price the positions off live quotes -- an independent check, not a
# weaker one.
ACCOUNT_SYNC_RETRIES = 2
ACCOUNT_SYNC_WAIT_S = 8.0
# Every request to the gateway that ``ib_async`` offers only as a BLOCKING call
# (it waits for an answer with no deadline) is run through ``Broker._bounded``
# with one of these, so a gateway that never answers cannot hang the run.
GATEWAY_REQUEST_TIMEOUT_S = 20.0
# A quote is reused for this long within one run (the pre-flight's drift check
# and the buy leg's funding budget ask for the same names minutes apart).
QUOTE_CACHE_TTL_S = 60.0

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

# How long each order leg is given to fill before the run moves on.
DEFAULT_FILL_TIMEOUT_S = 60.0
# Outside regular hours that is far too short, and not because fills are merely
# slower: the BUY leg is budgeted from what the SELL leg actually realised, so a
# sell still working when the wait expires funds nothing and every buy comes
# back SKIPPED-FUNDING — which is exactly how the 2026-09-10 after-hours run
# ended up placing two-share buys against a $19k sell that had not printed yet.
# Inside RTH a stuck limit is rescued by the MARKET escalation; out here there
# is no such escalation, so the wait IS the mechanism. Waiting longer is close
# to free: the loop ends as soon as every leg is done.
AFTER_HOURS_FILL_TIMEOUT_S = 300.0

# NYSE/Nasdaq stop accepting MOC entries at 15:50 ET; the auction prints at
# 16:00 ET.  The executor's 2:30-PM-CT (3:30 PM ET) slot sits 20 minutes ahead
# of the cutoff — which is precisely what makes MOC usable at all (it was not,
# at the old 8:45-AM-CT slot).
MOC_CUTOFF_ET = (15, 50)
US_CLOSE_ET = (16, 0)
# US equities keep trading past the 4:00 PM ET close: IBKR works an ``outsideRth``
# order through the extended session, which ends at 20:00 ET.  That window is the
# only reason a MISSED run can still be caught up — past it nothing works the
# order, and by the next session the book is stale by construction.
EXT_CLOSE_ET = (20, 0)
ET_TZ = "America/New_York"
# After the auction prints, give IBKR a moment to report the fills back.
MOC_CLOSE_BUFFER_S = 120.0
# Hard ceiling on the post-submit wait, so a stuck run can never hang forever.
# It must cover the whole stretch from the scheduled start to the auction print:
# the 2:15 PM CT task (3:15 PM ET) places its orders about 2 minutes in and then
# waits ~44 minutes for the 4:00 PM ET close plus the reporting buffer. The old
# 40-minute ceiling expired ~3 minutes BEFORE the auction, so the execution
# report was written while every leg still read "Submitted" and the run exited
# "not fully filled". 60 minutes covers a start as early as ~3:00 PM ET.
MOC_MAX_WAIT_S = 3600.0

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

# Scheduled 1:00 PM ET early closes (the day after Thanksgiving, Christmas Eve).
# Exchanges stop accepting MOC entries 10 minutes before the bell, so on these
# days the cutoff is 12:50 ET — long before the 3:30 PM ET executor slot.  A
# strict-MOC run therefore skips the session instead of sending orders the
# exchange would reject.  Extend as needed.
US_EARLY_CLOSES = {
    "2026-11-27": (13, 0), "2026-12-24": (13, 0), "2027-11-26": (13, 0),
}
MOC_CUTOFF_LEAD_MIN = 10          # minutes before the bell MOC entry stops

# The tables above are hand-maintained and end at some year. Past that year the
# executor would treat every market holiday as a trading day and send MOC orders
# into a closed market, so it refuses to trade (see ``calendar_covers``) rather
# than guess.
CALENDAR_LAST_YEAR = max(int(d[:4]) for d in (*US_MARKET_HOLIDAYS, *US_EARLY_CLOSES))


def calendar_covers(day) -> bool:
    """Is *day* inside the years the holiday / early-close tables cover?"""
    return pd.Timestamp(day).year <= CALENDAR_LAST_YEAR


def calendar_warning(today, within_days: int = 120) -> str | None:
    """A heads-up (not a block) when the tables are close to running out."""
    end = pd.Timestamp(year=CALENDAR_LAST_YEAR, month=12, day=31)
    left = (end - pd.Timestamp(today).normalize()).days
    if left < 0:
        return (f"the market-holiday table ends in {CALENDAR_LAST_YEAR}; trading "
                "is refused until US_MARKET_HOLIDAYS / US_EARLY_CLOSES are extended")
    if left <= within_days:
        return (f"the market-holiday table ends {end:%b %d, %Y} ({left} days) — "
                "extend US_MARKET_HOLIDAYS / US_EARLY_CLOSES in scripts/ibkr_common.py")
    return None


# ════════════════════════════════════════════════════════════════════════════
# TRADING-DAY / FRESHNESS GUARDS
# ════════════════════════════════════════════════════════════════════════════
# IBKR signals "this P&L is not available" by sending Double.MAX_VALUE
# (~1.8e308) rather than omitting the field, so a raw float() of it renders as
# an absurd number. Anything at that magnitude is the sentinel, not a P&L.
_PNL_UNAVAILABLE = 1e30


def consistent_cash(net_liq: float, positions: list[dict], cash: float,
                    tol: float = 0.02) -> float:
    """The cash figure an execution report should record.

    ``TotalCashValue`` and ``NetLiquidation`` are separate account-summary
    reads, and the cash tag can lag the portfolio: right after a MOC fill the
    positions and net liq include it while cash still shows the pre-fill
    balance (U23154574, 2026-10-13: cash overstated by ~$100k, 21.8% of net
    liq). When ``net_liq - sum(market_value)`` disagrees with the reported cash
    by more than ``tol`` of net liq, the implied figure is returned instead.
    Needs a real net liq and priced positions; otherwise the reading is kept.
    Short positions carry a negative market value, so a plain sum is right."""
    if net_liq <= 0 or not positions:
        return cash
    if any(not float(p.get("market_value") or 0.0) for p in positions):
        return cash                       # unpriced name: implied cash is unreliable
    implied = net_liq - sum(float(p["market_value"]) for p in positions)
    return implied if abs(cash - implied) > tol * net_liq else cash


def _pnl_value(raw) -> float | None:
    """A P&L field from IBKR as a float, or None when IBKR withholds it.

    None and 0.0 mean different things here — "the broker did not report it"
    versus "the position really has booked nothing" — so an unavailable value
    is never flattened to zero."""
    if raw is None or raw == "":
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if val != val or abs(val) >= _PNL_UNAVAILABLE:     # NaN or the sentinel
        return None
    return val


def is_trading_day(day: pd.Timestamp) -> tuple[bool, str]:
    """(tradeable?, reason) — False on weekends and US market holidays."""
    if day.weekday() >= 5:
        return False, f"{day.date()} is a weekend"
    if str(day.date()) in US_MARKET_HOLIDAYS:
        return False, f"{day.date()} is a US market holiday"
    return True, "trading day"


def prev_trading_day(day: pd.Timestamp, max_back: int = 10) -> pd.Timestamp:
    """The last US trading session STRICTLY before *day* (weekends/holidays skipped)."""
    d = pd.Timestamp(day).normalize()
    for _ in range(max_back):
        d -= pd.Timedelta(days=1)
        if is_trading_day(d)[0]:
            return d
    raise RuntimeError(f"no trading day found within {max_back} days before {day}")


def sessions_behind(bar: pd.Timestamp, want: pd.Timestamp,
                    max_back: int = 20) -> int:
    """How many US trading sessions *bar* sits behind *want* (0 if equal/ahead)."""
    sessions = 0
    d = pd.Timestamp(want).normalize()
    bar = pd.Timestamp(bar).normalize()
    while d > bar and sessions < max_back:
        d = prev_trading_day(d)
        sessions += 1
    return sessions


def bar_is_current(as_of: str, today: pd.Timestamp, *,
                   equity_close: str | None = None) -> tuple[bool, str]:
    """(ok, reason) — is this book the one this session is supposed to trade?

    The publisher computes a book from a COMPLETED close and the executor
    trades it during the next session, so a book whose EQUITY basis is older
    than the last completed session is stale — yesterday's decision, priced off
    yesterday's close, re-traded into a market that has already moved (and, if
    the account already holds it, traded a second time).  That is what the
    freshness window alone could not catch: a book generated at 7:15 AM CT is
    still inside a 36-hour window at 2:30 PM CT the FOLLOWING day.

    A withheld publish (failed audit, engine error) therefore means NO TRADE,
    not "trade the last book we have" — the intended fallback all along.

    TWO CALENDARS.  ``as_of`` is the freshest bar across the whole universe, and
    Bitcoin prints one every day, so the publisher (which runs 7 days a week)
    stamps a WEEKEND ``as_of`` on the Saturday, Sunday and Monday books — while
    their equity basis is Friday's close either way, because equities do not
    trade on weekends.  Judging ``as_of`` against the previous *equity* session
    therefore refused the Monday book as "AHEAD" and, after every US market
    holiday, refused the next session's book too: a whole session traded
    nothing (2026-08-24).  So when the book carries the publisher's
    signature-covered ``signal_basis.equity_close`` stamp, gate on THAT — the
    exact 4:00-PM-ET close the book was built from — and let ``as_of`` carry
    the 24/7 sleeves through the weekend, which is the entire reason the
    publisher runs on weekends at all.

    Books published before 2026-07-23 carry no ``signal_basis``, so
    ``equity_close`` is None for them and the original ``as_of`` rule stands."""
    if not as_of:
        return False, "no signal as-of date"
    bar = pd.Timestamp(as_of).normalize()
    want = prev_trading_day(today)

    if equity_close:
        basis = pd.Timestamp(equity_close).normalize()
        # a bar dated past today is a broken publisher, not a weekend sleeve
        if bar > pd.Timestamp(today).normalize():
            return False, (f"signal bar {bar.date()} is in the FUTURE (today is "
                           f"{pd.Timestamp(today).normalize().date()}) — the book "
                           f"is not tradeable")
        if basis == want:
            return True, (f"equity basis {basis.date()} is the last completed "
                          f"session" + (f" (signal bar {bar.date()} carries the "
                                        f"24/7 sleeves)" if bar != want else ""))
        if basis > want:
            return False, (f"equity basis {basis.date()} is AHEAD of the last "
                           f"completed session ({want.date()}) — the book is not "
                           f"tradeable yet")
        return False, (f"STALE book: equity basis {basis.date()} is "
                       f"{sessions_behind(basis, want)} session(s) behind the "
                       f"last completed session ({want.date()})")

    if bar == want:
        return True, f"signal bar {bar.date()} is the last completed session"
    if bar > want:
        return False, (f"signal bar {bar.date()} is AHEAD of the last completed "
                       f"session ({want.date()}) — the book is not tradeable yet")
    return False, (f"STALE book: signal bar {bar.date()} is "
                   f"{sessions_behind(bar, want)} session(s) behind the last "
                   f"completed session ({want.date()})")


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


def executable_qty(action: str, qty: float, fractional: bool) -> float:
    """The quantity that can actually be SENT for a planned order.

    Fractional trading sends *qty* as is.  Whole-share trading (which MOC always
    is) cannot send a fraction, and the two directions must round differently:
    a SELL rounds DOWN — rounding 22.8672 held shares up to 23 sells a share the
    account does not own (rejected, or a short) — while a BUY rounds to nearest,
    which can overshoot the target by under half a share.  A SELL whose whole-
    share part is zero returns 0.0: the fractional dust is left for the owner to
    sell by hand, since no MOC order can close it."""
    q = float(qty)
    if fractional:
        return q
    if action == "SELL":
        return float(math.floor(q + 1e-9))
    return float(round(q))


def build_order_plan(weights: dict[str, float], exec_price: dict[str, float],
                     net_liq: float, current: dict[str, float], band: float,
                     fractional: bool, max_order_notional: float = 0.0,
                     skipped: list | None = None) -> list[Order]:
    """Turn target weights + current share counts into BUY/SELL orders.

    ``weights``    signal-key → target portfolio weight (0..1).
    ``exec_price`` signal-key → price of the instrument we actually trade
                   (IBIT for the BTC sleeve, the ticker itself otherwise).
    ``current``    signal-key → shares held now (already translated from IBKR
                   symbols).  A name held but not in ``weights`` is fully closed.
    A per-name no-trade band (``band`` × net-liq) suppresses tiny rebalances.
    ``max_order_notional`` (>0) clamps any single order's size to that dollar cap
    — a fat-finger / bug backstop that matters most for live trading.

    Without ``fractional`` every order is a WHOLE number of shares and the plan
    already says so (see :func:`executable_qty`): the preview then shows exactly
    what will be sent, and a SELL never exceeds what is held.  A name whose only
    remaining difference is fractional dust that cannot be sold in whole shares
    is left out of the plan; if a *skipped* list is passed, it is described
    there (``{"key", "symbol", "held", "reason"}``) so the caller can say so.
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
            if (not fractional and w <= 0 and held > 1e-6 and skipped is not None):
                skipped.append(dict(key=key, symbol=symbol, held=held,
                                    reason=f"{held:g} share(s) left, less than one "
                                           "whole share"))
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

        if not fractional:
            whole = executable_qty("BUY" if delta > 0 else "SELL", order_qty, False)
            if whole < 0.5:
                # e.g. 0.87 of a share left after the rest was closed: nothing a
                # whole-share order can do. Say so rather than drop it silently.
                if skipped is not None:
                    skipped.append(dict(key=key, symbol=symbol, held=held,
                                        reason=f"{abs(delta):g} share(s) cannot "
                                               "be traded in whole shares"))
                continue
            order_qty = whole

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
# ════════════════════════════════════════════════════════════════════════════
# EXPOSURE / FUNDING GUARDS
# ════════════════════════════════════════════════════════════════════════════
# The August-18 incident these exist to prevent: three duplicate runs each read
# the account as flat, each bought the whole book again, and the account ended
# at 3.4x NAV on a $2.17M margin loan. Every one of those rounds was arithmetic
# the plan builder could not question, because it was handed `current = {}` and
# sizes off net liquidation — which does NOT fall when you buy on margin, so
# each round looked exactly as affordable as the first.


class PositionReadError(RuntimeError):
    """The broker's positions read could not be trusted — never trade on it."""


def position_read_is_trustworthy(read_value: float, gross_value: float,
                                 net_liq: float, tol: float = POSITION_READ_TOL
                                 ) -> tuple[bool, str]:
    """(ok, reason) — does the positions read account for the account's holdings?

    *read_value* is the gross value the read itself explains; *gross_value* is
    what the account reports independently (IBKR's ``GrossPositionValue``).  A
    read that misses material value is a subscription that has not arrived yet,
    not a flat account — and treating it as flat is what buys the book twice.

    A zero/absent ``GrossPositionValue`` cannot confirm anything, so the read
    passes only when it is itself empty; that keeps a genuinely flat account
    (day one) tradeable while refusing the dangerous direction."""
    if net_liq <= 0:
        return False, "net liquidation is zero or unreadable"
    if gross_value <= 0:
        if read_value > 0:
            return True, ("account reports no gross position value; "
                          f"read explains ${read_value:,.0f} — using the read")
        return True, "account reports no positions and the read agrees (flat)"
    missing = gross_value - read_value
    if missing > tol * net_liq:
        return False, (f"positions read explains ${read_value:,.0f} of the "
                       f"${gross_value:,.0f} the account reports "
                       f"(${missing:,.0f} missing, > {tol*100:.0f}% of NAV) — "
                       "the position subscription is incomplete")
    return True, (f"positions read matches the account "
                  f"(${read_value:,.0f} vs ${gross_value:,.0f} reported)")


def projected_exposure(orders: list[Order], current: dict[str, float],
                       exec_price: dict[str, float], net_liq: float) -> dict:
    """Gross exposure before and after *orders*, valued at the book's prices."""
    def _px(key: str) -> float:
        return float(exec_price.get(key) or 0.0)

    gross_before = sum(abs(sh) * _px(k) for k, sh in current.items())
    buy = sum(o.qty * (o.price or _px(o.key)) for o in orders if o.action == "BUY")
    sell = sum(o.qty * (o.price or _px(o.key)) for o in orders if o.action == "SELL")
    gross_after = gross_before + buy - sell
    return {"gross_before": gross_before, "buy_notional": buy,
            "sell_notional": sell, "gross_after": gross_after,
            "net_liq": float(net_liq),
            "leverage_after": (gross_after / net_liq) if net_liq > 0 else 0.0}


def check_exposure(exposure: dict, max_gross_frac: float = DEFAULT_MAX_GROSS_FRAC
                   ) -> tuple[bool, str]:
    """(ok, reason) — refuse a plan that would leave the account above NAV.

    The book is an unlevered allocation: weights sum to at most 1.0, so a plan
    landing materially above 1x net liquidation cannot be what the book asked
    for."""
    lev = exposure["leverage_after"]
    if exposure["net_liq"] <= 0:
        return False, "net liquidation is zero or unreadable"
    if lev > max_gross_frac:
        return False, (f"plan would leave gross exposure at "
                       f"${exposure['gross_after']:,.0f} = {lev:.2f}x net liq "
                       f"(cap {max_gross_frac:.2f}x) — the book is unlevered, so "
                       "this means the positions read or the run is duplicated")
    return True, (f"projected gross ${exposure['gross_after']:,.0f} = {lev:.2f}x "
                  f"net liq (cap {max_gross_frac:.2f}x)")


def fit_buys_to_funding(buys: list[Order], available: float, fractional: bool,
                        buffer: float = FUNDING_BUFFER,
                        price_of=None) -> tuple[list[Order], list[Order], str]:
    """(sendable, dropped, note) — scale BUY orders to the cash that funds them.

    Buys are paid for out of settled cash plus whatever the sell leg actually
    realised.  When the plan asks for more than that, every buy is scaled by the
    same factor so the allocation keeps its shape, and names that fall below one
    share are dropped rather than rounded up.  This is the backstop that turns
    "buy the whole book again" into "buy what the account can pay for" — the
    duplicate rounds would have been trimmed to nothing.

    ``price_of`` supplies the price the budget is measured at — by default the
    book's own ``exec_price``, but the caller passes a LIVE quote (see
    :meth:`Broker.funding_price`).  That distinction matters: the book prices
    yesterday's close, so on a market that gapped up, shares sized at the book
    price cost more than the book says and the account borrows the difference.
    Pricing the budget at what the order can actually pay removes that gap.
    """
    px = price_of or (lambda o: float(o.price or 0.0))
    want = sum(o.qty * px(o) for o in buys)
    budget = max(0.0, available) * (1.0 - buffer)
    if want <= budget or want <= 0:
        return list(buys), [], (f"buys ${want:,.0f} funded from ${available:,.0f} "
                                "available — no trimming")
    scale = budget / want
    sendable, dropped = [], []
    for o in buys:
        qty = o.qty * scale
        # never let whole-share rounding push the spend back over the budget
        if not fractional:
            qty = float(int(qty))
        if qty < (1e-6 if fractional else 1.0):
            dropped.append(o)
            continue
        sendable.append(Order(o.key, o.symbol, o.action, qty, o.price,
                              f"{o.reason} [trimmed to funding: "
                              f"{qty:g} of {o.qty:g}]"))
    note = (f"buys ${want:,.0f} exceed ${available:,.0f} available — scaled to "
            f"{scale*100:.0f}% (${budget:,.0f}); {len(dropped)} name(s) dropped")
    return sendable, dropped, note


def validate_book_math(weights: dict[str, float], exec_price: dict[str, float],
                       max_total: float = 1.02) -> tuple[bool, str]:
    """(ok, reason) — is the book itself arithmetically sane?

    A corrupted or mis-generated book (NaN weight, negative weight, weights
    summing past 1, a zero/absent price on a name we are asked to buy) must
    never reach the order planner: every one of those turns into a wrong order
    size rather than an error."""
    total = 0.0
    for k, w in weights.items():
        try:
            w = float(w)
        except (TypeError, ValueError):
            return False, f"weight for {k} is not a number ({w!r})"
        if not math.isfinite(w):
            return False, f"weight for {k} is not finite ({w})"
        if w < 0:
            return False, f"negative weight for {k} ({w:.4f}) — the book is long-only"
        total += w
        if w > 0:
            px = exec_price.get(k)
            try:
                px = float(px)
            except (TypeError, ValueError):
                px = 0.0
            if not math.isfinite(px) or px <= 0:
                return False, (f"{k} has weight {w*100:.1f}% but no usable "
                               f"execution price ({exec_price.get(k)!r})")
    if total > max_total:
        return False, (f"weights sum to {total*100:.1f}% (> {max_total*100:.0f}%) — "
                       "the book is unlevered by construction, so this is corrupt")
    return True, f"book math OK (weights sum to {total*100:.1f}%)"


def check_account_state(net_liq: float, cash: float, gross_value: float,
                        allow_margin: bool = False,
                        max_gross_frac: float = DEFAULT_MAX_GROSS_FRAC,
                        cash_tol: float = 0.005) -> tuple[bool, str]:
    """(ok, reason) — is the account in a state the book can be traded into?

    Two red flags before a single order is planned: the account already carries
    a margin loan (cash below ``-cash_tol`` x NAV — a few dollars of fee debit
    is not a loan), or it is already geared past the cap.  Both
    mean the account is not where the last run left it, so sizing a fresh book
    against it would compound the problem rather than correct it — exactly the
    state the duplicate runs left behind."""
    if net_liq <= 0:
        return False, "net liquidation is zero or unreadable"
    if cash < -abs(cash_tol) * net_liq and not allow_margin:
        return False, (f"account already carries a margin loan (cash "
                       f"${cash:,.0f}) — refusing to trade until it is flat "
                       "(--allow-margin to override)")
    if gross_value > max_gross_frac * net_liq:
        return False, (f"account is already at {gross_value/net_liq:.2f}x net liq "
                       f"before trading (cap {max_gross_frac:.2f}x) — reduce it "
                       "before running the book")
    return True, (f"account OK (cash ${cash:,.0f}, gross ${gross_value:,.0f} = "
                  f"{gross_value/net_liq:.2f}x net liq)")


def check_margin_health(broker, net_liq: float, min_cushion: float
                        ) -> tuple[bool, str]:
    """(ok, reason) — is the broker's own risk system quiet?  Best-effort: a
    broker that cannot report excess liquidity passes (see
    :func:`ibkr_safety.margin_problem`)."""
    read = getattr(broker, "excess_liquidity", None)
    if min_cushion <= 0 or not callable(read):
        return True, "margin-health check disabled"
    try:
        ex = read()
    except Exception:
        ex = None
    if not isinstance(ex, (int, float)):
        return True, "margin health not verified (excess liquidity unreadable)"
    why = safety.margin_problem(ex, net_liq, min_cushion)
    if why:
        return False, why
    return True, f"margin healthy (excess liquidity ${ex:,.0f})"


def check_turnover(orders: list[Order], net_liq: float,
                   max_frac: float = 1.5) -> tuple[bool, str]:
    """(ok, reason) — refuse a plan that churns implausibly much of the account.

    A daily rebalance moves a few percent of NAV; a full liquidate-and-rebuy is
    ~2x. Anything past the cap means the plan was built against the wrong
    picture of the account."""
    traded = sum(o.qty * (o.price or 0.0) for o in orders)
    if net_liq <= 0:
        return False, "net liquidation is zero or unreadable"
    frac = traded / net_liq
    if frac > max_frac:
        return False, (f"plan trades ${traded:,.0f} = {frac:.2f}x net liq "
                       f"(cap {max_frac:.2f}x) — implausible for a daily "
                       "rebalance; check the positions read")
    return True, f"turnover ${traded:,.0f} = {frac:.2f}x net liq"


def market_session_open(now=None, allow_outside: bool = False) -> tuple[bool, str]:
    """(ok, reason) — is the US regular session open right now?

    Orders sent outside regular hours fill against a thin book (or, for MOC and
    MARKET, are rejected outright), and the executor's own slot sits inside the
    session — so an after-hours run is a scheduling accident, not a decision."""
    et = _et_now(now)
    open_t = et.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = et.replace(hour=US_CLOSE_ET[0], minute=US_CLOSE_ET[1],
                         second=0, microsecond=0)
    if open_t <= et <= close_t:
        return True, f"regular session open ({et:%H:%M} ET)"
    if allow_outside:
        return True, f"outside regular hours ({et:%H:%M} ET) — allowed explicitly"
    return False, (f"outside regular trading hours ({et:%H:%M} ET; session is "
                   "09:30–16:00 ET) — use --outside-rth to trade anyway")


# ── missed-slot catch-up ────────────────────────────────────────────────────
# What ``catch_up_state`` can report.  Three values because the caller must do
# three different things, and collapsing any two of them into a bool is what
# would turn "the market is open" into an after-hours order.
CATCH_UP_OPEN = "open"     # regular hours — the ordinary run works, do nothing
CATCH_UP_LATE = "late"     # past the close, still inside the extended session
CATCH_UP_SHUT = "shut"     # nothing would fill; the day is simply missed


def catch_up_state(now=None) -> tuple[str, str]:
    """(state, reason) — can a run that MISSED its scheduled slot still trade?

    The scheduled executor sits at 2:30 PM CT (3:30 PM ET), and a host that is
    switched off at that moment simply never places the day's book.  By the next
    morning that book is stale BY CONSTRUCTION — ``bar_is_current`` refuses it,
    and rightly so: the publisher has since built a newer one off a newer close,
    and re-trading the old one would act on a decision the engine has already
    replaced.  So the missed decision is still THIS session's decision only for
    the remainder of the same trading day, in the extended session out to
    20:00 ET, where an ``outsideRth`` limit can still fill.

    The three outcomes:

    ``CATCH_UP_OPEN``  regular hours — there is nothing to catch up; the normal
                       run places the book the normal way.
    ``CATCH_UP_LATE``  past the 16:00 close, inside the extended session — the
                       book is still tradeable, as ``outsideRth`` limits (IBKR
                       rejects MARKET and MOC there, so routing must switch).
    ``CATCH_UP_SHUT``  before the open, past 20:00 ET, or not a trading day —
                       no order placed now would fill, so the session is missed
                       and tomorrow's book supersedes this one.

    Pre-market (04:00–09:30 ET) reports SHUT rather than LATE on purpose: the
    day's 2:30 PM CT slot has not been missed yet, it simply has not arrived,
    and the book on disk at that hour belongs to the session that is about to
    open.
    """
    et = _et_now(now)
    ok_day, day_why = is_trading_day(et.tz_localize(None))
    if not ok_day:
        return CATCH_UP_SHUT, f"{day_why} — there is no session to catch up"
    if market_session_open(et)[0]:
        return CATCH_UP_OPEN, (f"regular session is open ({et:%H:%M} ET) — "
                               "nothing to catch up")
    # Wall-clock edges, the way market_session_open() builds its own: a session
    # boundary is a local time, not an offset from midnight.
    close_t = et.replace(hour=US_CLOSE_ET[0], minute=US_CLOSE_ET[1],
                         second=0, microsecond=0)
    ext_t = et.replace(hour=EXT_CLOSE_ET[0], minute=EXT_CLOSE_ET[1],
                       second=0, microsecond=0)
    ext_stamp = f"{EXT_CLOSE_ET[0]:02d}:{EXT_CLOSE_ET[1]:02d} ET"
    if et <= close_t:
        return CATCH_UP_SHUT, (f"{et:%H:%M} ET is before the 09:30 open — the "
                               "day's slot has not been missed, only not "
                               "reached yet")
    if et <= ext_t:
        return CATCH_UP_LATE, (f"{et:%H:%M} ET is past the 16:00 close but "
                               f"inside the extended session (to {ext_stamp})")
    return CATCH_UP_SHUT, (f"{et:%H:%M} ET is past the {ext_stamp} end of the "
                           "extended session — no order placed now would fill")


def fill_timeout(explicit: float | None,
                 outside_rth: bool) -> tuple[float, str]:
    """(seconds, reason) — how long to give each order leg to fill.

    An explicit ``--fill-timeout`` always wins; otherwise the default follows
    the session, because the two cases are not the same problem.  Inside regular
    hours the wait only decides how soon an unfilled limit is escalated to a
    MARKET order, so 60s is plenty.  Outside them there is no escalation, and
    the sell leg's fills are what funds the buy leg — so the wait is the only
    thing standing between a slow sell and a run whose every buy is dropped for
    lack of funding.
    """
    if explicit is not None:
        return float(explicit), f"{float(explicit):.0f}s (--fill-timeout)"
    if outside_rth:
        return AFTER_HOURS_FILL_TIMEOUT_S, (
            f"{AFTER_HOURS_FILL_TIMEOUT_S:.0f}s (outside RTH — the buy leg is "
            "funded by what the sells realise, so they are given time to fill)")
    return DEFAULT_FILL_TIMEOUT_S, f"{DEFAULT_FILL_TIMEOUT_S:.0f}s"


def extended_hours_tif(now=None) -> tuple[str, str]:
    """(tif, goodTillDate) for a limit that has to work the extended session.

    A DAY order is good until the end of the REGULAR session, so one sent after
    16:00 ET arrives already expired and IBKR cancels it on the spot — which is
    what error 10349 ("Order TIF was set to DAY based on order preset") is: the
    Gateway's own order preset supplying DAY to an order that carried no TIF of
    its own.  Setting the TIF explicitly on the order overrides that preset.

    GTD pinned to the 20:00 ET end of the extended session is the right TIF
    rather than GTC: it works for the rest of tonight and then expires by
    itself, so nothing is left resting against tomorrow's plan.  A stray GTC
    from an after-hours run would still be live at the next day's 2:30 PM CT
    slot, which sizes its orders from POSITIONS and cannot see it — the exact
    shape of a double execution.

    The stamp is IBKR's UTC form, ``yyyymmdd-hh:mm:ss`` with a DASH between the
    date and the time.  Its zoned form takes only legacy names (``US/Eastern``);
    an IANA name is rejected with error 343 and the order is cancelled on
    arrival, so the instant is expressed in UTC and no name is involved at all.
    """
    et = _et_now(now)
    end = et.replace(hour=EXT_CLOSE_ET[0], minute=EXT_CLOSE_ET[1],
                     second=0, microsecond=0)
    return "GTD", end.tz_convert("UTC").strftime("%Y%m%d-%H:%M:%S")


def price_drift(book_price: float, live_price: float) -> float:
    """|live − book| / book, or 0.0 when either side is unusable."""
    try:
        b, l = float(book_price), float(live_price)
    except (TypeError, ValueError):
        return 0.0
    if not (math.isfinite(b) and math.isfinite(l)) or b <= 0 or l <= 0:
        return 0.0
    return abs(l - b) / b


def realised_weights(current: dict[str, float], price: dict[str, float],
                     net_liq: float) -> dict[str, float]:
    """Post-trade holdings as portfolio weights, valued at *price*."""
    if net_liq <= 0:
        return {}
    return {k: abs(sh) * float(price.get(k) or 0.0) / net_liq
            for k, sh in current.items()}


def weight_drift(realised: dict[str, float], target: dict[str, float]
                 ) -> list[tuple[str, float, float, float]]:
    """[(key, target, realised, drift_pp)] sorted by |drift| descending."""
    keys = set(realised) | set(target)
    rows = [(k, float(target.get(k, 0.0)), float(realised.get(k, 0.0)),
             (float(realised.get(k, 0.0)) - float(target.get(k, 0.0))) * 100)
            for k in keys]
    return sorted(rows, key=lambda r: -abs(r[3]))


@dataclass
class Guards:
    """The trading limits both entry points enforce, in one place.

    Defaults are the production settings; the CLIs expose each as a flag so an
    operator can widen one deliberately, in the open, rather than by editing
    code."""
    max_gross_frac: float = DEFAULT_MAX_GROSS_FRAC
    max_turnover_frac: float = 1.5
    max_price_drift: float = 0.25
    slippage_cap: float = DEFAULT_SLIPPAGE_CAP
    allow_margin: bool = False
    outside_rth: bool = False
    max_buy_frac: float = DEFAULT_MAX_BUY_FRAC
    max_orders: int = DEFAULT_MAX_ORDERS
    min_margin_cushion: float = safety.DEFAULT_MIN_MARGIN_CUSHION

    @classmethod
    def from_args(cls, args) -> "Guards":
        """Build from an argparse namespace, falling back to the defaults for
        any flag a given entry point does not expose."""
        g = cls()
        for f in ("max_gross_frac", "max_turnover_frac", "max_price_drift",
                  "slippage_cap", "allow_margin", "outside_rth",
                  "max_buy_frac", "max_orders", "min_margin_cushion"):
            v = getattr(args, f, None)
            if v is not None:
                setattr(g, f, v)
        return g


def gross_value(broker, current: dict[str, float],
                exec_price: dict[str, float]) -> float:
    """Gross position value on the broker's own marks, book prices as a fallback.

    Book prices alone would UNDER-count a name that is held but no longer in the
    book (it has no price there) — and under-counting exposure is the direction
    that lets a geared account look fine."""
    for read in (getattr(broker, "gross_position_value", None),
                 getattr(broker, "read_value", None)):
        if read is None:
            continue
        try:
            v = float(read())
            if v > 0:
                return v
        except Exception:
            continue
    return sum(abs(sh) * float(exec_price.get(k) or 0.0) for k, sh in current.items())


def price_drift_check(broker, orders: list[Order], tol: float,
                      slippage_cap: float = DEFAULT_SLIPPAGE_CAP
                      ) -> tuple[bool, str]:
    """Compare each order's sizing price against a live quote.

    A corporate action between publish and execution (a split above all) leaves
    the book's ``exec_price`` off by a multiple, so every share count computed
    from it is wrong by that same multiple. Best-effort: a name that cannot be
    quoted is not a reason to block the run, but a name quoting far from its
    sizing price is."""
    if tol <= 0:
        return True, "price-drift check disabled"
    worst = (0.0, "")
    checked = 0
    for o in orders:
        q = broker.quote_by_symbol(o.symbol)      # cached: the funding budget
        if not q:                                 # asks for the same names later
            continue
        live = next((q.get(k) for k in ("last", "close", "ask", "bid")
                     if _finite(q.get(k))), None)
        if live is None:
            continue
        checked += 1
        d = price_drift(o.price, live)
        if d > worst[0]:
            worst = (d, f"{o.symbol} sized at ${o.price:,.2f} but quoting "
                        f"${float(live):,.2f} ({d*100:.0f}% away)")
    if not checked:
        return True, "price-drift check skipped (no quotes available)"
    if worst[0] > tol:
        return False, (f"{worst[1]} — beyond the {tol*100:.0f}% limit; a split or "
                       "a bad book price would size every order wrong")
    return True, (f"sizing prices within {tol*100:.0f}% of the market "
                  f"({checked} checked, worst {worst[0]*100:.1f}%)")


def live_price(quote: dict) -> float | None:
    """The price the drift checks compare with: last, else close, ask, bid."""
    return next((float(quote[k]) for k in ("last", "close", "ask", "bid")
                 if _finite(quote.get(k))), None)


def soft_drift_flags(broker, orders: list[Order], soft_tol: float
                     ) -> dict[str, str]:
    """{symbol: reason} for BUYS of unlevered names whose live price is more than
    *soft_tol* from the book's sizing price.

    This is the tighter, gentler sibling of :func:`price_drift_check`: that one
    aborts the whole run at 25 % (a split), this one only holds the BUY of the
    one name — sells are never held, and leveraged / single-stock / crypto names
    (whose ordinary day can exceed it) are exempt.  Uses the cached quotes, so it
    costs no extra market-data calls; a name that cannot be quoted is not
    flagged."""
    if soft_tol <= 0:
        return {}
    out: dict[str, str] = {}
    for o in orders:
        if o.action != "BUY" or sym.is_volatile(o.symbol):
            continue
        try:
            q = broker.quote_by_symbol(o.symbol)
        except Exception:
            continue
        live = live_price(q or {})
        if live is None:
            continue
        d = price_drift(o.price, live)
        if d > soft_tol:
            out[o.symbol] = (f"{o.symbol} quotes ${live:,.2f} vs the book's "
                             f"${float(o.price):,.2f} ({d*100:.0f}% away, soft "
                             f"limit {soft_tol*100:.0f}%) — a wrong book price "
                             "would size the buy wrong")
    return out


def preflight(broker, orders: list[Order], current: dict[str, float],
              exec_price: dict[str, float], net_liq: float,
              guards: Guards) -> tuple[bool, str]:
    """Every check that must pass between building the plan and sending it.

    Each one answers a way the 2026-08-18 duplicate-execution incident could
    repeat: an account that is already geared, a plan that would gear it, a plan
    that churns the whole book, a sizing price the market has left behind, or a
    run that landed outside the session.  All are cheap and read-only, and BOTH
    entry points (the Option-C executor and the Option-A rebalancer) run the
    same set — they place orders through the same code, so they must refuse the
    same things.  Returns (ok, reason); reason is empty when everything passed.
    """
    print("\nPre-flight:")
    try:
        cash = broker.cash()
    except Exception:
        cash = 0.0
    gross = gross_value(broker, current, exec_price)
    exposure = projected_exposure(orders, current, exec_price, net_liq)

    checks = [
        check_account_state(net_liq, cash, gross, guards.allow_margin,
                            guards.max_gross_frac),
        check_margin_health(broker, net_liq, guards.min_margin_cushion),
        market_session_open(allow_outside=guards.outside_rth),
        check_order_sanity(orders, current, net_liq, guards.max_buy_frac,
                           guards.max_orders),
        check_turnover(orders, net_liq, guards.max_turnover_frac),
        check_exposure(exposure, guards.max_gross_frac),
        price_drift_check(broker, orders, guards.max_price_drift,
                          guards.slippage_cap),
    ]
    skew_fn = getattr(broker, "clock_skew", None)
    if callable(skew_fn):
        try:
            checks.append(check_clock(skew_fn()))
        except Exception as e:
            checks.append((True, f"clock not verified ({e})"))
    # Informational only: the stress gate and NAV brake never abort the run —
    # they withhold BUYS at placement time, so sells (de-risking) still go.
    lim = getattr(broker, "safety", None)
    if isinstance(lim, safety.SafetyLimits):
        try:
            if getattr(broker, "buy_block", None):
                print(f"  ! NAV brake: {broker.buy_block} — buys will be withheld")
            st = broker.stress_assessment(orders)
            if st is not None:
                state = (f"BUYS WILL BE WITHHELD ({st.block_buys})" if st.block_buys
                         else "no market-wide stress")
                print(f"  {'!' if st.block_buys else '✓'} {state}"
                      + (f" [{'; '.join(st.notes)}]" if st.notes else ""))
                for why in st.names.values():
                    print(f"  ! buy withheld: {why}")
            for why in soft_drift_flags(broker, orders, lim.soft_price_drift).values():
                print(f"  ! buy withheld: {why}")
        except Exception as e:
            print(f"  · stress check unavailable ({e}) — not blocking")
    failed = ""
    for ok, why in checks:
        print(f"  {'✓' if ok else '✗'} {why}")
        if not ok and not failed:
            failed = why
    return (not failed), failed


def post_trade_check(broker, weights: dict[str, float],
                     exec_price: dict[str, float], guards: Guards,
                     max_drift_pp: float = 5.0) -> tuple[bool, str]:
    """Re-read the account AFTER trading and say whether it landed on target:
    ``(ok, reason)``.

    Nothing is auto-corrected here -- a second corrective round of orders is
    exactly the reflex that compounds a bad read.  This reports, loudly, and
    returns a verdict the caller turns into an exit code and an alert, so a run
    that ended badly cannot look identical to a good one."""
    print("\nPost-trade:")
    try:
        net_liq = broker.net_liq()
        after = broker.positions_by_key()
    except Exception as e:
        print(f"  could not verify the resulting account ({e})")
        return False, f"could not verify the resulting account ({e})"
    gross = gross_value(broker, after, exec_price)
    lev = gross / net_liq if net_liq > 0 else 0.0
    print(f"  {'✗' if lev > guards.max_gross_frac else '✓'} gross ${gross:,.0f} = "
          f"{lev:.2f}x net liq ${net_liq:,.0f}")
    problems = []
    if lev > guards.max_gross_frac:
        print("  WARNING: the account is geared above the cap after this run — "
              "do NOT re-run the executor; reconcile the account first.")
        problems.append(f"gross {lev:.2f}x net liq (cap {guards.max_gross_frac:.2f}x)")
    rows = weight_drift(realised_weights(after, exec_price, net_liq), weights)
    for key, tgt, got, dpp in rows[:3]:
        print(f"    {key:6} target {tgt*100:5.1f}%  actual {got*100:5.1f}%  "
              f"drift {dpp:+.1f} pp")
    if rows and abs(rows[0][3]) > max_drift_pp:
        print(f"  WARNING: largest allocation drift is {rows[0][3]:+.1f} pp on "
              f"{rows[0][0]} — check the fills above.")
        problems.append(f"{rows[0][0]} drifted {rows[0][3]:+.1f} pp from target")
    if problems:
        return False, "; ".join(problems)
    return True, "account landed on target"


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


def session_close_et(day) -> tuple[int, int]:
    """(hour, minute) ET of the closing auction on *day* — 16:00, or the 13:00
    early close on the handful of days the exchanges shut at 1 PM."""
    return US_EARLY_CLOSES.get(str(pd.Timestamp(day).date()), US_CLOSE_ET)


def moc_cutoff_et(day) -> tuple[int, int]:
    """(hour, minute) ET after which a MOC entry is rejected on *day*."""
    close = session_close_et(day)
    if close == US_CLOSE_ET:
        return MOC_CUTOFF_ET
    t = close[0] * 60 + close[1] - MOC_CUTOFF_LEAD_MIN
    return t // 60, t % 60


def moc_entry_open(now=None, cutoff: tuple[int, int] | None = None
                   ) -> tuple[bool, str]:
    """(accepting?, reason) — can a MOC order still be entered right now?

    Past the exchange cutoff a MOC is rejected outright, so the caller must
    route around it rather than discover the rejection order by order.  The
    cutoff follows the session: 15:50 ET normally, 12:50 ET on a 1 PM early
    close.  An explicit *cutoff* overrides that."""
    et = _et_now(now)
    if cutoff is None:
        cutoff = moc_cutoff_et(et)
    edge = et.normalize() + pd.Timedelta(hours=cutoff[0], minutes=cutoff[1])
    stamp = et.strftime("%H:%M ET")
    if et > edge:
        return False, (f"{stamp} is past the {cutoff[0]:02d}:{cutoff[1]:02d} ET "
                       f"MOC entry cutoff")
    return True, f"{stamp}, before the {cutoff[0]:02d}:{cutoff[1]:02d} ET MOC cutoff"


def moc_wait_seconds(now=None, buffer_s: float = MOC_CLOSE_BUFFER_S,
                     max_wait_s: float = MOC_MAX_WAIT_S) -> float:
    """How long to wait for MOC fills: until the closing auction prints plus
    a reporting buffer, clamped to ``max_wait_s``.

    MOC is the one order type whose fills are unknowable at submit time, so the
    executor has to hold the connection open to write a truthful execution
    report instead of one that says "Submitted" for every leg."""
    et = _et_now(now)
    h, m = session_close_et(et)
    close = et.normalize() + pd.Timedelta(hours=h, minutes=m)
    secs = (close - et).total_seconds() + buffer_s
    return float(min(max(secs, 0.0), max_wait_s))


def moc_wait_capped(now=None, buffer_s: float = MOC_CLOSE_BUFFER_S,
                    max_wait_s: float = MOC_MAX_WAIT_S) -> bool:
    """True when :func:`moc_wait_seconds` would stop waiting BEFORE the auction
    has printed (the ceiling bit) — the report would then be written pre-fill."""
    et = _et_now(now)
    h, m = session_close_et(et)
    close = et.normalize() + pd.Timedelta(hours=h, minutes=m)
    return ((close - et).total_seconds() + buffer_s) > max_wait_s


def print_plan(orders: list[Order], net_liq: float) -> None:
    print(f"\nOrder plan (net-liq ${net_liq:,.0f}):")
    if not orders:
        print("  ✓ already within the no-trade band — nothing to do")
        return
    for o in orders:
        print(f"  {o.action:4s} {o.qty:8g} {o.symbol:5s} "
              f"~${o.value:12,.0f}   {o.reason}")


# ════════════════════════════════════════════════════════════════════════════
# MULTI-ACCOUNT + STRICT-MOC GUARDS — pure helpers, unit-tested without a broker.
# ════════════════════════════════════════════════════════════════════════════
# Strategy Logic V2 caps SATA at 30 % of the book (overall_core.SATA_CAP; a test
# keeps the two equal -- this module must stay free of the model stack).  New
# books carry the cap they were built under (``sata_cap``, signed).  A book
# published BEFORE the cap carries none, and a stale pre-cap book can still be
# the one that trades if the next publish is withheld -- so the executor applies
# this default to such a book rather than park 40 % (or, under a deploy cap,
# 75 %) of an account in a single preferred.
DEFAULT_SATA_CAP = 0.30


def effective_sata_cap(book_cap, default_cap) -> tuple[float | None, str]:
    """(cap, where it came from).  The book's own cap wins; a book without one
    falls back to *default_cap* (``None`` or negative = no fallback, the old
    behaviour).  A malformed book cap is treated as 0 -- all idle capital stays
    cash -- the cautious reading of a signed value that cannot be parsed."""
    if book_cap is not None:
        try:
            cap = float(book_cap)
            if math.isfinite(cap):
                return max(cap, 0.0), "book"
        except (TypeError, ValueError):
            pass
        return 0.0, "book (unreadable -> 0)"
    if default_cap is not None and float(default_cap) >= 0:
        return float(default_cap), "executor default (book predates the V2 SATA cap)"
    return None, "none"


def trim_sata(weights: dict[str, float], cap: float) -> tuple[dict[str, float], float]:
    """Cap the SATA weight at *cap*; the excess is simply not assigned, i.e.
    cash.  Returns (new weights, excess trimmed)."""
    out = dict(weights)
    excess = max(out.get("SATA", 0.0) - cap, 0.0)
    if excess > 1e-12:
        out["SATA"] = cap
    else:
        excess = 0.0
    return out, excess


def apply_substitutions(weights: dict[str, float], subs: dict[str, str]
                        ) -> tuple[dict[str, float], list[tuple[str, str, float]]]:
    """Re-assign a restricted instrument's weight to its tradeable substitute.

    ``subs`` maps a signal key the account cannot hold (UGL in an IBKR IRA) to
    the key it trades instead (GLDM).  The weight moves one-for-one -- same
    capital, so a 2x sleeve becomes 1x exposure -- and ADDS to the substitute if
    the book already holds it (GLDM and UGL are both on when gold trends).
    Total deployed weight is unchanged.  Returns ``(new weights, moves)`` where
    each move is ``(from_key, to_key, weight)``.  The input is not modified."""
    out = dict(weights)
    moves: list[tuple[str, str, float]] = []
    for src, dst in subs.items():
        w = out.pop(src, 0.0)
        if w > 0:
            out[dst] = out.get(dst, 0.0) + w
            moves.append((src, dst, w))
    return out, moves


def parse_substitutions(items) -> dict[str, str]:
    """``["UGL=GLDM", ...]`` -> ``{"UGL": "GLDM"}``; raises ValueError on a bad
    pair, an unmapped key, or a chain/loop (a substitute that is itself
    substituted)."""
    subs: dict[str, str] = {}
    for it in items or []:
        src, sep, dst = str(it).partition("=")
        src, dst = src.strip().upper(), dst.strip().upper()
        if not sep or not src or not dst or src == dst:
            raise ValueError(f"bad substitution {it!r} (want KEY=KEY, e.g. UGL=GLDM)")
        for k in (src, dst):
            if not sym.trade_symbol(k):
                raise ValueError(f"substitution {it!r}: {k} is not a tradeable key")
        subs[src] = dst
    if any(d in subs for d in subs.values()):
        raise ValueError("a substitute may not itself be substituted")
    return subs


def apply_deploy_cap(weights: dict[str, float], max_deploy_frac: float
                     ) -> tuple[dict[str, float], dict | None]:
    """Cap the TOTAL the book may deploy at *max_deploy_frac* of the account.

    Every asset in the book -- SATA included -- is scaled by the same factor so
    the weights sum to the cap, and everything left over simply stays CASH
    (a weight that is not assigned is not traded into).  It is a ceiling, not a
    target: a book that already deploys less than the cap is left as it is.

    Returns ``(new weights, info)`` where info is None when no scaling was
    needed, else ``{"before", "after", "scale"}``.  The input is not modified."""
    out = dict(weights)
    total = sum(v for v in out.values() if v > 0)
    if max_deploy_frac >= 1.0 or total <= max_deploy_frac or total <= 0:
        return out, None
    scale = max_deploy_frac / total
    out = {k: v * scale for k, v in out.items()}
    return out, dict(before=total, after=max_deploy_frac, scale=scale)


def check_clock(skew_s, limit: float = MAX_CLOCK_SKEW_S) -> tuple[bool, str]:
    """(ok, reason) -- is the PC clock close enough to IBKR's server clock?

    *skew_s* is local minus server in seconds (None = could not be measured,
    which is reported but is not a failure: the gateway answered everything
    else)."""
    if skew_s is None:
        return True, "clock not verified (the gateway did not return its time)"
    if abs(skew_s) > limit:
        return False, (f"this PC's clock is {skew_s:+.0f}s from IBKR's server clock "
                       f"(limit {limit:g}s) -- the MOC cutoff and trading-day tests "
                       "would be wrong; sync Windows time (w32tm /resync)")
    return True, f"clock within {abs(skew_s):.1f}s of IBKR's server"


def check_order_sanity(orders: list, current: dict[str, float], net_liq: float,
                       max_buy_frac: float = DEFAULT_MAX_BUY_FRAC,
                       max_orders: int = DEFAULT_MAX_ORDERS) -> tuple[bool, str]:
    """(ok, reason) -- limits that hold whatever the book says.

    * no SELL larger than the shares held (a short, or a rejected order);
    * no single BUY larger than *max_buy_frac* of net liquidation (a wrong
      price or NAV sizes a buy absurdly; SELLs are bounded by holdings);
    * no more than *max_orders* orders in one run."""
    if max_orders and len(orders) > max_orders:
        return False, (f"{len(orders)} orders in one run (limit {max_orders}) -- "
                       "implausible for a daily rebalance")
    for o in orders:
        if o.action == "SELL":
            held = float(current.get(o.key, 0.0))
            if o.qty > held + 1e-6:
                return False, (f"SELL {o.qty:g} {o.symbol} exceeds the {held:g} "
                               "shares held -- refusing to sell what the account "
                               "does not own")
        elif o.action == "BUY" and net_liq > 0 and max_buy_frac:
            frac = o.qty * (o.price or 0.0) / net_liq
            if frac > max_buy_frac:
                return False, (f"BUY {o.qty:g} {o.symbol} is {frac*100:.0f}% of net "
                               f"liq (limit {max_buy_frac*100:.0f}%) -- check the "
                               "book price and the account value")
    return True, f"{len(orders)} order(s) within the per-order limits"


def runtime_versions() -> str:
    """One line naming the interpreter and the libraries the run used, so a
    behaviour change after an upgrade can be traced from the log."""
    import platform
    from importlib import metadata
    parts = [f"python {platform.python_version()}"]
    for pkg in ("ib_async", "pandas", "numpy"):
        try:
            parts.append(f"{pkg} {metadata.version(pkg)}")
        except Exception:
            parts.append(f"{pkg} ?")
    return ", ".join(parts)


def print_dust(skipped: list[dict]) -> None:
    """Tell the operator what the whole-share plan had to leave alone."""
    for d in skipped or []:
        print(f"  note: {d['symbol']} — {d['reason']} (holding {d['held']:g}); "
              "sell the fractional remainder by hand in TWS if you want it gone")


class MocUnavailable(RuntimeError):
    """Strict-MOC run that cannot send its orders (past the exchange cutoff).

    Raised BEFORE any order is built.  A strict-MOC run never degrades to a
    limit or market order: a session that cannot be traded in the closing
    auction is skipped, and tomorrow's book supersedes today's."""


def select_account(accounts: list[str], requested: str | None,
                   account_mode: str) -> tuple[str | None, str]:
    """(account, reason) — which of the login's managed accounts to trade.

    One IB Gateway login can expose several accounts, and ``managedAccounts()``
    lists them in no order you control.  Taking ``[0]`` silently — the old
    behaviour — trades whichever account IBKR happens to list first, which is
    only acceptable when the login has exactly one.  So:

    * an explicit *requested* id must be in the list (exact match) — otherwise
      the answer is None and the reason names what IS available;
    * with no request and a single account, that account is used;
    * with no request and several accounts, a LIVE run refuses (None) — it will
      not guess which real account to trade.  Paper keeps the old ``[0]``."""
    accts = [a for a in (accounts or []) if a]
    if requested:
        if requested in accts:
            return requested, f"account {requested} selected"
        return None, (f"account '{requested}' is not managed by this login "
                      f"(available: {', '.join(accts) or 'none'})")
    if not accts:
        return None, "the gateway reported no managed accounts (not logged in?)"
    if len(accts) == 1:
        return accts[0], f"single account {accts[0]}"
    if account_mode == "live":
        return None, (f"this login manages {len(accts)} accounts "
                      f"({', '.join(accts)}) — pass --account to say which one "
                      "to trade; a live run will not guess")
    return accts[0], f"first of {len(accts)} accounts: {accts[0]}"


def working_orders(trades, account: str) -> list:
    """The not-yet-done orders in *trades* that belong to *account*.

    An order with no account stamp is treated as belonging to the account being
    checked (the safe reading: it blocks rather than slips through)."""
    out = []
    for t in trades or []:
        try:
            if t.isDone():
                continue
            acct = getattr(t.order, "account", "") or ""
        except Exception:
            continue
        if not acct or acct == account:
            out.append(t)
    return out


def check_no_working_orders(trades, account: str) -> tuple[bool, str]:
    """(ok, reason) — refuse to send a book into an account that already has
    orders working.

    The duplicate-run lock keys on the execution REPORT, which a MOC run only
    writes after the 4:00 PM auction — so a second run at 3:40 PM would find no
    report and send the whole book again, on top of MOC orders that cannot be
    cancelled after the cutoff.  Orders still working at the broker are the
    ground truth for that window."""
    live = working_orders(trades, account)
    if not live:
        return True, f"no working orders on {account}"
    def _desc(t) -> str:
        qty = getattr(t.order, "totalQuantity", None)
        q = f"{qty:g} " if isinstance(qty, (int, float)) else ""
        return f"{t.order.action} {q}{t.contract.symbol}"
    names = ", ".join(_desc(t) for t in live[:6])
    more = f" (+{len(live) - 6} more)" if len(live) > 6 else ""
    return False, (f"{account} already has {len(live)} working order(s): "
                   f"{names}{more} — a previous run's orders are still live and "
                   "sending the book again would double them. Wait for them to "
                   "fill/expire, or cancel them in TWS, then re-run "
                   "(--ignore-working-orders to override)")


def send_lock_path(report_path, as_of) -> Path:
    """``…/<account dir>/.send_locks/<as_of>.lock`` next to the report."""
    return (Path(report_path).parent / ".send_locks"
            / f"{pd.Timestamp(as_of).date()}.lock")


def acquire_send_lock(report_path, as_of, account: str,
                      force: bool = False) -> tuple[bool, str]:
    """Atomically claim the right to TRANSMIT *as_of*'s book for this account.

    Created immediately before orders go out and left in place afterwards: it
    is the record that this bar's orders were sent, readable the moment they
    are — hours before a MOC report exists.  ``O_EXCL`` makes two simultaneous
    runs race to a single winner.  ``force`` (``--force-rerun``) replaces it.
    The directory is untracked/ignored, so a ``git reset --hard`` after a
    failed report push cannot erase it."""
    import json as _json
    import os as _os
    from datetime import datetime, timezone
    p = send_lock_path(report_path, as_of)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = _json.dumps({"account": account, "as_of": str(pd.Timestamp(as_of).date()),
                        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
                        "pid": _os.getpid()})
    flags = _os.O_WRONLY | _os.O_CREAT | (_os.O_TRUNC if force else _os.O_EXCL)
    try:
        fd = _os.open(str(p), flags)
    except FileExistsError:
        try:
            prior = _json.loads(p.read_text())
            when = prior.get("locked_at_utc", "?")
        except Exception:
            when = "?"
        return False, (f"orders for signal bar {pd.Timestamp(as_of).date()} were "
                       f"already transmitted for {account} at {when} "
                       f"({p.name}) — refusing to send them again. If that run "
                       "genuinely placed nothing, re-run with --force-rerun")
    with _os.fdopen(fd, "w") as f:
        f.write(body)
    return True, f"send lock taken ({p.name})"


def skipped_result(o: Order, status: str, why: str, qty: float | None = None) -> dict:
    """Report row for a leg that was deliberately NOT sent, so the execution
    report says what was withheld and why instead of the leg just vanishing."""
    return dict(key=o.key, symbol=o.symbol, action=o.action,
                qty=float(o.qty if qty is None else qty), price=float(o.price),
                status=status, filled=0.0, avg_fill_price=0.0, order_type="",
                limit_price=0.0, reason=f"{o.reason} [{why}]")


def broker_message(trade, status: str | None = None) -> str:
    """The broker's last explanatory message for an order that did not fill.

    IBKR states why it refused or cancelled an order (restricted product, IRA
    rule, no permission, price outside the collar…) in the trade log; the
    execution report used to drop it.  Empty for a clean fill, or when the
    broker said nothing."""
    try:
        st = status if status is not None else trade.orderStatus.status
        if st == "Filled":
            return ""
        for entry in reversed(getattr(trade, "log", None) or []):
            msg = (getattr(entry, "message", "") or "").strip()
            if msg:
                return msg[:300]
    except Exception:
        pass
    return ""


def moc_rejections(trades) -> list[tuple[str, str, str]]:
    """(symbol, status, last log message) for MOC orders the broker already
    refused or cancelled — visible within seconds of submission, long before
    the auction, which is the window in which a human can still act."""
    bad = []
    for *_, t in trades:
        try:
            st = t.orderStatus.status
            if st in ("Cancelled", "ApiCancelled", "Inactive"):
                msg = (t.log[-1].message if getattr(t, "log", None) else "") or ""
                bad.append((t.contract.symbol, st, msg))
        except Exception:
            continue
    return bad


# ════════════════════════════════════════════════════════════════════════════
# IBKR SESSION — connect, read account, place orders (lazy ib_async import).
# ════════════════════════════════════════════════════════════════════════════
class Broker:
    """Thin ``ib_async`` wrapper: paper-account guard, positions, net-liq, orders."""

    def __init__(self, host: str, port: int, client_id: int,
                 account_mode: str = "paper", expected_account: str | None = None,
                 confirm_live: bool = False, market_data_type: int = 1,
                 account: str | None = None):
        """Connect and enforce the account guard for the chosen mode.

        account_mode:
          * ``paper`` — the connected account MUST start with ``DU`` (default).
          * ``live``  — the account must NOT be paper; additionally requires
            ``confirm_live=True`` and, if ``expected_account`` is given, an EXACT
            match, so a real account can never be traded by accident or the wrong
            real account traded by mistake.
          * ``any``   — no check (escape hatch; not used by the executor).

        account: which of the login's managed accounts to trade (see
          :func:`select_account`).  Every read and every order below is scoped
          to ``self.account``; orders carry it explicitly so the gateway cannot
          route them to a different account on the same login.  In live mode a
          login with several accounts REQUIRES it.
        """
        from ib_async import IB
        self.ib = IB()
        # ``account=`` tells ib_async which account is the main one for the
        # account-update subscription; with several accounts on one login it
        # must not be left to the gateway's default.
        self.ib.connect(host, port, clientId=client_id, timeout=20,
                        account=account or "")
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
        accts = list(self.ib.managedAccounts() or [])
        chosen, why = select_account(accts, account, account_mode)
        if chosen is None:
            self._abort(why)
        self.accounts = accts
        self.account = chosen
        print(f"  {why}" + (f" (login manages {len(accts)})" if len(accts) > 1 else ""))
        self.account_mode = account_mode
        # stamped on every order this run sends (set by the caller, e.g.
        # "btc-range:2026-10-09") so the bot's orders are recognisable in TWS
        # and in audits
        self.order_ref = ""
        self._enforce_account_guard(account_mode, expected_account or account,
                                    confirm_live)

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
        """NetLiquidation in USD. The executor sizes every order in dollars, so
        an account whose base currency is not USD is refused rather than read
        as if its number were dollars."""
        seen = None
        for v in self.ib.accountSummary(self.account):
            if v.tag == "NetLiquidation":
                cur = (getattr(v, "currency", "") or "USD").upper()
                if cur == "USD":
                    return float(v.value)
                seen = cur
        for v in self.ib.accountValues(self.account):
            if v.tag == "NetLiquidation" and \
                    (getattr(v, "currency", "") or "USD").upper() == "USD":
                return float(v.value)
        if seen:
            raise RuntimeError(f"the account's NetLiquidation is in {seen}, not USD "
                               "-- the executor sizes in dollars; refusing to trade")
        raise RuntimeError("could not read NetLiquidation from the account")

    def clock_skew(self) -> float | None:
        """This PC's clock minus IBKR's server clock, in seconds (None if the
        gateway did not return its time)."""
        try:
            done, t = self._bounded("reqCurrentTime", timeout=GATEWAY_REQUEST_TIMEOUT_S)
        except Exception:
            return None
        if not done or t is None:
            return None
        t = pd.Timestamp(t)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        return float((pd.Timestamp.now(tz="UTC") - t).total_seconds())

    def cash(self) -> float:
        """Total cash value (best-effort; 0.0 if the tag isn't present)."""
        for v in self.ib.accountSummary(self.account):
            if v.tag == "TotalCashValue":
                return float(v.value)
        return 0.0

    def excess_liquidity(self) -> float | None:
        """ExcessLiquidity in USD — the cushion IBKR liquidates on — or None."""
        for src in (lambda: self.ib.accountSummary(self.account),
                    lambda: self.ib.accountValues(self.account)):
            try:
                for v in src():
                    if v.tag == "ExcessLiquidity" and \
                            (getattr(v, "currency", "") or "USD").upper() in ("USD", "BASE"):
                        return float(v.value)
            except Exception:
                continue
        return None

    def realized_pnl(self) -> float | None:
        """The account's SESSION realized P&L, or None if IBKR won't say.

        IBKR reports realized P&L per trading session, not since inception — so
        read straight after a rebalance (which is when the executor writes its
        report) this is what THAT rebalance banked: the gain/loss booked by
        every trim and close it placed.  Any manual trade made in the same
        session lands in it too.

        Account-level on purpose.  A name closed out ENTIRELY disappears from
        ``ib.portfolio()`` (the wrapper drops a contract the moment its size
        hits zero), so per-position realized P&L can never see a full exit —
        only this figure covers the whole session."""
        for v in self.ib.accountSummary(self.account):
            if v.tag == "RealizedPnL" and v.currency in ("USD", "BASE", ""):
                return _pnl_value(v.value)
        for v in self.ib.accountValues(self.account):
            if v.tag == "RealizedPnL" and v.currency in ("USD", "BASE"):
                return _pnl_value(v.value)
        return None

    def portfolio_snapshot(self) -> list[dict]:
        """Current holdings with cost/market data, one dict per position.

        Uses ``ib.portfolio()`` (richer than ``positions()``): shares, average
        cost, and — when the paper account has market data — market price/value
        and unrealised P&L. Fields that IBKR can't supply come back as 0.0, which
        the UI renders as “—”. Foreign symbols keep ``key=None``.

        ``realized_pnl`` is the session figure IBKR attributes to that position:
        after a rebalance that TRIMMED a name, it is what the sold slice booked
        (the remaining shares keep their average cost, so the trim is otherwise
        invisible in this snapshot).  ``None`` when IBKR withholds it — which is
        not the same as a real zero, so it is never coerced to 0.0."""
        out = []
        self.positions_by_key()          # settle the subscription first: a report
        items = list(self.ib.portfolio(self.account))  # written off a half-arrived
        if not items and self._positions_raw():        # read understates the book
            self.refresh_account_updates()
            items = list(self.ib.portfolio(self.account))
        if not items and self._positions_raw():
            return self._snapshot_from_positions()
        for item in items:
            c = item.contract
            out.append(dict(
                key=sym.key_for_symbol(c.symbol), symbol=c.symbol,
                shares=float(getattr(item, "position", 0.0) or 0.0),
                avg_cost=float(getattr(item, "averageCost", 0.0) or 0.0),
                market_price=float(getattr(item, "marketPrice", 0.0) or 0.0),
                market_value=float(getattr(item, "marketValue", 0.0) or 0.0),
                unrealized_pnl=_pnl_value(getattr(item, "unrealizedPNL", None)),
                realized_pnl=_pnl_value(getattr(item, "realizedPNL", None))))
        return out

    def _snapshot_from_positions(self) -> list[dict]:
        """Holdings rebuilt from the positions read and live quotes, for when
        the account's portfolio subscription never arrived.  Shares and average
        cost are the broker's own; the mark is a live quote (0.0 -- shown as
        "—" -- when none is available); realised P&L is unknown, not zero."""
        print("    portfolio subscription unavailable — report built from "
              "positions + live quotes")
        out = []
        for p in self._positions_raw():
            c = p.contract
            shares = float(getattr(p, "position", 0.0) or 0.0)
            if abs(shares) < 1e-9:
                continue
            avg = float(getattr(p, "avgCost", 0.0) or 0.0)
            q = self.quote(c)
            bid, ask = _finite(q.get("bid")), _finite(q.get("ask"))
            mid = (bid + ask) / 2.0 if (bid and ask) else None
            px = next((v for v in (_finite(q.get("last")), mid,
                                   _finite(q.get("close"))) if v), None)
            out.append(dict(
                key=sym.key_for_symbol(c.symbol), symbol=c.symbol, shares=shares,
                avg_cost=avg, market_price=float(px or 0.0),
                market_value=float(px * shares) if px else 0.0,
                unrealized_pnl=((px - avg) * shares) if px and avg else None,
                realized_pnl=None))
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
            # One login, several accounts: ``ib.fills()`` is the whole login's,
            # so without this filter account B's report would book account A's
            # trades as its own.
            fill_acct = getattr(ex, "acctNumber", "") or ""
            if fill_acct and fill_acct != self.account:
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

    def gross_position_value(self) -> float:
        """IBKR's own ``GrossPositionValue`` for the account (0.0 if absent).

        Read independently of the position subscription, so it can be used to
        catch a positions read that has not arrived yet."""
        for tag in ("GrossPositionValue",):
            for v in self.ib.accountSummary(self.account):
                if v.tag == tag:
                    try:
                        return abs(float(v.value))
                    except (TypeError, ValueError):
                        return 0.0
        return 0.0

    def _positions_raw(self) -> list:
        return list(self.ib.positions(self.account))

    def positions_by_key(self, settle: bool = True) -> dict[str, float]:
        """Current holdings as signal-key → shares (foreign symbols ignored).

        ``ib_async`` fills its position cache asynchronously after connect, so a
        read taken too early comes back EMPTY — indistinguishable, to the order
        planner, from a flat account.  With *settle* on, the subscription is
        requested explicitly and re-read until it stops changing (or turns up
        non-empty), which removes the race; :meth:`holdings` then verifies the
        result against the account's own gross value before anything is traded.
        """
        if settle:
            try:
                self._bounded("reqPositions", timeout=GATEWAY_REQUEST_TIMEOUT_S)
            except Exception:
                pass                      # already subscribed — the sleep still helps
            last = None
            for _ in range(POSITION_SETTLE_TRIES):
                raw = self._positions_raw()
                snap = tuple(sorted((p.contract.symbol, float(p.position)) for p in raw))
                if raw and snap == last:
                    break                 # non-empty and stable → settled
                last = snap
                self.ib.sleep(POSITION_SETTLE_SLEEP_S)
        out: dict[str, float] = {}
        for p in self._positions_raw():
            key = sym.key_for_symbol(p.contract.symbol)
            if key is not None:
                out[key] = out.get(key, 0.0) + float(p.position)
        return out

    def read_value(self) -> float:
        """What the position read itself accounts for, on IBKR's own marks.

        Valued from ``ib.portfolio()`` (market value where the account has data,
        cost basis otherwise) rather than from the book's prices: a name being
        CLOSED is not in the book at all, so book prices would under-count the
        read and make the verification below misfire on a perfectly good read.
        """
        total = 0.0
        for item in self.ib.portfolio(self.account):
            mv = abs(float(getattr(item, "marketValue", 0.0) or 0.0))
            if not mv:                    # no market-data subscription → cost
                mv = abs(float(getattr(item, "position", 0.0) or 0.0)) * \
                     abs(float(getattr(item, "averageCost", 0.0) or 0.0))
            total += mv
        return total

    def _bounded(self, name: str, *args, timeout: float):
        """Run the ib_async request *name* with a DEADLINE: (finished, result).

        ``IB.reqAccountUpdates`` / ``reqAllOpenOrders`` / ``reqPositions`` are
        "blocking" calls that wait for the gateway's answer with no timeout, so
        a gateway that never answers (it does not serve some accounts' account
        updates on a multi-account login) hangs the run forever.  The ``…Async``
        twin run under ``wait_for`` returns (False, None) instead.  Objects with
        no async twin (the unit-test fakes) fall back to the plain call."""
        fn = getattr(self.ib, name + "Async", None)
        if fn is None:
            return True, getattr(self.ib, name)(*args)
        import asyncio
        from ib_async import util
        try:
            return True, util.run(fn(*args), timeout=timeout)
        except (asyncio.TimeoutError, TimeoutError):
            return False, None

    def _qualify(self, contract) -> None:
        """Fill in the contract's details, with a deadline (raises TimeoutError)."""
        done, _ = self._bounded("qualifyContracts", contract,
                                timeout=GATEWAY_REQUEST_TIMEOUT_S)
        if not done:
            raise TimeoutError(f"no contract details for {contract.symbol} "
                               f"within {GATEWAY_REQUEST_TIMEOUT_S:g}s")

    def refresh_account_updates(self) -> None:
        """Ask the gateway again for this account's values and portfolio.

        Bounded by :data:`ACCOUNT_SYNC_WAIT_S`: harmless when the subscription is
        healthy, and when the gateway does not answer it just returns so the
        caller can fall back to live quotes."""
        try:
            done, _ = self._bounded("reqAccountUpdates", self.account,
                                    timeout=ACCOUNT_SYNC_WAIT_S)
            if not done:
                print(f"    no answer from the gateway within "
                      f"{ACCOUNT_SYNC_WAIT_S:g}s")
        except Exception as e:
            print(f"    could not re-request account updates ({e})")
        self.ib.sleep(1.0)

    def read_value_from_quotes(self) -> float | None:
        """The positions valued at live quotes, or None if any cannot be priced.

        Independent of the account-updates subscription (positions come from
        ``reqPositions``, prices from market data), so it can stand in when that
        subscription never arrives.  None -- not a partial sum -- when a held
        name has no usable quote: an understated value would wrongly fail the
        cross-check, an assumed one would wrongly pass it."""
        total = 0.0
        for p in self._positions_raw():
            shares = float(getattr(p, "position", 0.0) or 0.0)
            if abs(shares) < 1e-9:
                continue
            q = self.quote(p.contract)
            bid, ask = _finite(q.get("bid")), _finite(q.get("ask"))
            mid = (bid + ask) / 2.0 if (bid and ask) else None
            price = next((v for v in (_finite(q.get("last")), mid,
                                      _finite(q.get("close"))) if v), None)
            if price is None:
                return None
            mult = getattr(p.contract, "multiplier", "") or "1"
            try:
                mult = float(mult)
            except (TypeError, ValueError):
                mult = 1.0
            total += abs(shares) * price * (mult or 1.0)
        return total

    def holdings(self, net_liq: float | None = None,
                 tol: float = POSITION_READ_TOL) -> dict[str, float]:
        """Current holdings, VERIFIED against the account's gross position value.

        Raises :class:`PositionReadError` when the read misses material value —
        the state that let three duplicate runs each buy the whole book on top
        of the one already held.  Refusing to trade on an unverifiable read is
        always safe: the next scheduled run re-reads it.
        """
        current = self.positions_by_key()          # settled
        nl = float(net_liq if net_liq is not None else self.net_liq())
        try:
            value = self.read_value()
        except Exception:
            value = 0.0
        gross = self.gross_position_value()
        ok, why = position_read_is_trustworthy(value, gross, nl, tol)
        if not ok and self._positions_raw():
            # The account holds positions but the portfolio (their prices) did
            # not arrive. First give the subscription another chance...
            for attempt in range(1, ACCOUNT_SYNC_RETRIES + 1):
                print(f"  Positions: {why}\n  re-requesting account updates "
                      f"(attempt {attempt}/{ACCOUNT_SYNC_RETRIES})…")
                self.refresh_account_updates()
                try:
                    value = self.read_value()
                except Exception:
                    value = 0.0
                ok, why = position_read_is_trustworthy(value, gross, nl, tol)
                if ok:
                    break
            if not ok:
                # ...then price the positions off live quotes. Still an
                # independent comparison against the account's own gross value.
                qv = self.read_value_from_quotes()
                if qv is None:
                    why += " (and the positions could not be priced from live quotes)"
                else:
                    ok_q, why_q = position_read_is_trustworthy(qv, gross, nl, tol)
                    if ok_q:
                        ok = True
                        why = (f"{why_q} [valued at live quotes: the account's "
                               "portfolio subscription did not arrive]")
                    else:
                        why = why_q + " [valued at live quotes]"
        print(f"  Positions: {len(current)} name(s) — {why}")
        if not ok:
            raise PositionReadError(why)
        return current

    def working_orders(self) -> list:
        """Orders still working for this account, whichever API client or TWS
        window placed them (``reqAllOpenOrders`` sees them all)."""
        try:
            done, trades = self._bounded("reqAllOpenOrders",
                                         timeout=GATEWAY_REQUEST_TIMEOUT_S)
            if not done:
                raise TimeoutError(f"no answer within {GATEWAY_REQUEST_TIMEOUT_S:g}s")
        except Exception as e:
            raise RuntimeError(f"could not read the account's working orders "
                               f"({e}) — refusing to send blind") from e
        return working_orders(trades, self.account)

    def quote(self, contract) -> dict:
        """Snapshot bid/ask/last/close for a qualified contract (empty on any
        failure — a missing market-data subscription must degrade, not raise)."""
        try:
            done, tickers = self._bounded("reqTickers", contract,
                                          timeout=GATEWAY_REQUEST_TIMEOUT_S)
            if not done:
                print(f"    quote {contract.symbol}: no answer within "
                      f"{GATEWAY_REQUEST_TIMEOUT_S:g}s")
                return {}
        except Exception as e:
            print(f"    quote {contract.symbol}: unavailable ({e})")
            return {}
        if not tickers:
            return {}
        t = tickers[0]
        return {"bid": getattr(t, "bid", None), "ask": getattr(t, "ask", None),
                "last": getattr(t, "last", None), "close": getattr(t, "close", None),
                "halted": getattr(t, "halted", None)}

    def quote_by_symbol(self, symbol: str) -> dict:
        """Quote for *symbol*, cached briefly (empty dict on any failure).

        The pre-flight's price-drift check and the buy leg's funding budget ask
        for the same names minutes apart; one short-lived cache keeps them
        consistent and halves the market-data calls, while the TTL means a leg
        that took a while still re-prices."""
        cache = getattr(self, "_quotes", None)
        if cache is None:
            cache = self._quotes = {}
        now = time.monotonic()
        hit = cache.get(symbol)
        if hit and (now - hit[0]) < QUOTE_CACHE_TTL_S:
            return hit[1]
        try:
            contract = sym.ibkr_contract(symbol)
            self._qualify(contract)
            q = self.quote(contract)
        except Exception:
            q = {}
        cache[symbol] = (now, q)
        return q

    # ── market-stress & execution-quality guards (see ibkr_safety) ───────────
    # All of these read ``self.safety``; a Broker without one (a unit-test
    # double, or an entry point that never configured it) behaves exactly as
    # it did before the guards existed.
    def _alert(self, msg: str) -> None:
        cb = getattr(getattr(self, "safety", None), "alert", None)
        try:
            (cb or safety.send_alert)(msg)
        except Exception:
            print(msg)

    def _halt_reason(self) -> str | None:
        """The kill switch, re-read between orders.  Sticky: once it has fired
        the run stays halted even if the file is removed mid-run."""
        if getattr(self, "_halted", None):
            return self._halted
        check = getattr(getattr(self, "safety", None), "halt_check", None)
        if check is None:
            return None
        try:
            why = check()
        except Exception:
            why = None
        if why:
            self._halted = why
            self._alert(f"🛑 KILL SWITCH — {why}: no further orders will be "
                        f"sent and working orders are being cancelled "
                        f"({self.account}).")
        return why

    def _alert_moc_rejections(self, trades: list, alerted: set) -> None:
        """Print AND alert on every MOC the broker refused, once per symbol.

        The old behaviour printed the refusal to the log only; the first message
        a human got was the end-of-run summary AFTER the auction — hours after
        the 15:50 ET entry cutoff, the only window in which anyone could still
        fix it.  Strict MOC has no fallback, so a refused leg is a wrong
        position until the next run."""
        for symbol, st, msg in moc_rejections(trades):
            if symbol in alerted:
                continue
            alerted.add(symbol)
            line = (f"MOC {symbol} was {st} by the broker"
                    + (f": {msg}" if msg else ""))
            print(f"    !! {line} — nothing was substituted; act in TWS before "
                  "the cutoff")
            self._alert(f"🔴 {self.account}: {line}. Strict MOC has no fallback: "
                        "the position is now WRONG until fixed by hand in TWS "
                        f"before the {MOC_CUTOFF_ET[0]}:{MOC_CUTOFF_ET[1]:02d} ET "
                        "entry cutoff, or until the next run.")

    def _unsafe_quote(self, symbol: str, action: str, quote: dict) -> str | None:
        lim = getattr(self, "safety", None)
        if lim is None:
            return None
        return safety.quote_problem(quote, action, lim.max_spread,
                                    lim.sell_spread_mult, lim.outside_rth)

    def _cancel_working(self, trades: list) -> None:
        """Best-effort cancel of every not-yet-done order in *trades*."""
        for *_, t in trades:
            try:
                if not t.isDone():
                    self.ib.cancelOrder(t.order)
            except Exception as e:
                print(f"    WARN: cancel failed ({e})")

    def index_quote(self, symbol: str, exchange: str) -> dict:
        """Quote for an index (VIX…); empty on any failure — no subscription,
        no feed — so a missing index never blocks a rebalance."""
        try:
            from ib_async import Index
            contract = Index(symbol, exchange)
            self._qualify(contract)
            return self.quote(contract)
        except Exception:
            return {}

    def stress_assessment(self, orders: list[Order]) -> "safety.Stress | None":
        """Market-wide and per-name stress for the names about to be bought.
        None when no safety limits are configured.  Cached for the run."""
        lim = getattr(self, "safety", None)
        if lim is None:
            return None
        if getattr(self, "_stress", None) is not None:
            return self._stress
        names = {o.symbol: self.quote_by_symbol(o.symbol)
                 for o in orders if o.action == "BUY"} if lim.stress_name_move > 0 else {}
        spy = self.quote_by_symbol("SPY") if lim.stress_spy_move > 0 else None
        vix = self.index_quote("VIX", "CBOE") if lim.stress_vix > 0 else None
        self._stress = safety.stress_assessment(spy, vix, names, lim)
        return self._stress

    def _buy_block(self, buys: list[Order]) -> tuple[str | None, dict[str, str]]:
        """(reason ALL buys are blocked, {symbol: reason} for single names)."""
        why_all = getattr(self, "buy_block", None)        # e.g. the NAV brake
        st = self.stress_assessment(buys)
        if st is None:
            return why_all, {}
        per_name = soft_drift_flags(self, buys, self.safety.soft_price_drift)
        per_name.update(st.names)                         # a stress reason wins
        return why_all or st.block_buys, per_name

    def funding_price(self, o: Order,
                      slippage_cap: float = DEFAULT_SLIPPAGE_CAP) -> float:
        """The worst price per share *o* could pay (BUY) or receive (SELL).

        Taken from the live quote and priced exactly the way the order will be:
        a buy at the ask plus the slippage cap, a sell at the bid minus it — the
        ceiling on what a buy can cost and the floor on what a sell can realise.
        Both are the conservative side, so a budget built from them cannot be
        overspent by the fills themselves.

        Falls back to the book's ``exec_price`` when the name cannot be quoted
        (no market-data subscription, a dead feed): the same behaviour as before
        the live pricing existed, and no worse."""
        lmt, _basis = marketable_limit_price(o.action, self.quote_by_symbol(o.symbol),
                                             slippage_cap)
        if lmt is None or not math.isfinite(lmt) or lmt <= 0:
            return float(o.price or 0.0)
        return float(lmt)

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
                tif_note = ""
                if outside_rth:
                    order.outsideRth = True
                    # Without an explicit TIF the Gateway preset supplies DAY,
                    # which is already expired after 16:00 ET — the order is
                    # cancelled on arrival (error 10349).
                    order.tif, order.goodTillDate = extended_hours_tif()
                    tif_note = (f", {order.tif} to {EXT_CLOSE_ET[0]:02d}:"
                                f"{EXT_CLOSE_ET[1]:02d} ET")
                print(f"    limit {o.symbol}: {o.action} ≤ ${lmt:,.4f} "
                      f"({basis} {slippage_cap*100:.2f}% through)"
                      + (f" [outside RTH{tif_note}]" if outside_rth else ""))
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
                            now=None, outside_rth: bool = False,
                            moc_strict: bool = False) -> tuple[str, bool]:
        """(effective order type, fractional) after the routing guards.

        MOC is refused after the exchange cutoff (it would simply be rejected)
        and cannot carry fractional quantities.  By default both fall back
        rather than fail the run.  With *moc_strict* there is NO fallback: the
        run is MOC or it is nothing — a late run raises :class:`MocUnavailable`
        before a single order is built, and quantities are whole shares."""
        if order_type not in ORDER_TYPES:
            raise ValueError(f"unknown order type {order_type!r} "
                             f"(want one of {', '.join(ORDER_TYPES)})")
        if moc_strict:
            if order_type != ORDER_MOC:
                raise ValueError("strict MOC requires --order-type moc")
            if outside_rth:
                raise MocUnavailable("strict MOC cannot be combined with "
                                     "--outside-rth: there is no auction "
                                     "outside regular hours")
            ok, why = moc_entry_open(now)
            if not ok:
                raise MocUnavailable(f"{why} — strict MOC: nothing is sent "
                                     "(no limit/market fallback)")
            print(f"  MOC routing (strict): {why}; whole shares only, fills "
                  "print in the closing auction.")
            return ORDER_MOC, False
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
                      f"escalation is possible; they expire at the "
                      f"{EXT_CLOSE_ET[0]:02d}:{EXT_CLOSE_ET[1]:02d} ET end of "
                      f"the extended session")
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
            try:
                self._qualify(contract)
            except Exception as e:
                print(f"    WARN {o.symbol}: cannot escalate ({e})")
                continue
            # A MARKET order has no price ceiling, so it is the one order that
            # must not chase a halted, crossed or blown-out book — leave the
            # remainder unfilled; the next run re-plans from real positions.
            halt = self._halt_reason()
            bad = None
            if not halt and getattr(self, "safety", None) is not None:
                bad = self._unsafe_quote(o.symbol, o.action, self.quote(contract))
            if halt or bad:
                why = f"kill switch ({halt})" if halt else bad
                print(f"    NOT escalating {o.symbol}: {why} — {rem:g} left unfilled")
                if bad:
                    self._alert(f"⚠️ {o.action} {o.symbol} ({self.account}): "
                                f"{rem:g} shares left unfilled, not escalated to a "
                                f"market order — {bad}.")
                continue
            esc = Order(o.key, o.symbol, o.action, rem, o.price,
                        f"{o.reason} (escalated from unfilled limit)")
            mkt = MarketOrder(o.action, rem)
            mkt.account = self.account
            if getattr(self, "order_ref", ""):
                mkt.orderRef = self.order_ref
            extra.append((esc, rem, ORDER_MARKET, 0.0,
                          self.ib.placeOrder(contract, mkt)))
        if extra:
            deadline = wait
            while deadline > 0 and any(not t.isDone() for *_, t in extra):
                self.ib.sleep(1.0)
                deadline -= 1.0
        return extra

    def place(self, orders: list[Order], fractional: bool, wait: float,
              order_type: str = ORDER_MARKETABLE_LIMIT,
              slippage_cap: float = DEFAULT_SLIPPAGE_CAP,
              now=None, outside_rth: bool = False,
              allow_margin: bool = False,
              funding_buffer: float = FUNDING_BUFFER,
              moc_strict: bool = False) -> list[dict]:
        """Transmit the order plan, sells first, waiting for fills.

        ``order_type`` is one of :data:`ORDER_TYPES`; see the module header for
        why the marketable limit is the default and what MOC buys.  Returns one
        result dict per transmitted order (action, symbol, qty, reference price,
        the order type actually used and its limit, final status, filled
        quantity and average fill price) so the caller can build the execution
        report.

        **Funding.** The sell leg is transmitted and awaited first, and unless
        ``allow_margin`` is set the buy leg is then trimmed to what the account
        can actually pay for: settled cash plus the proceeds the sells REALLY
        realised (not the proceeds they were expected to realise).  Sells that
        did not fill therefore shrink the buys instead of being financed with a
        margin loan, and a plan built on a bad positions read cannot spend money
        the account does not have.  Trimmed-away names come back in the results
        as ``SKIPPED-FUNDING`` so the execution report shows what was not bought.
        """
        order_type, fractional = self._resolve_order_type(order_type, fractional,
                                                          now, outside_rth,
                                                          moc_strict)
        results: list[dict] = []
        sells = [o for o in orders if o.action == "SELL"]
        buys = [o for o in orders if o.action == "BUY"]
        self.funding_note = ""
        self._send_ref = {}          # (action, symbol) -> touch the order was priced off
        buys = self._withhold_blocked_buys(buys, results)

        cash0 = None
        if not allow_margin:
            try:
                cash0 = self.cash()
            except Exception as e:
                # Fail CLOSED: an unreadable cash balance is not permission to
                # spend. Buys are then funded from realised sell proceeds only,
                # which is the conservative reading of an account we cannot see.
                cash0 = 0.0
                print(f"    WARN: could not read cash ({e}) — funding buys from "
                      "realised sell proceeds only")

        # MOC legs all print in the SAME auction, so sequencing sells before
        # buys buys nothing — it would only burn minutes off the entry cutoff.
        # Submit everything, then wait once for the close. Nothing has filled by
        # then either, so the funding check uses the EXPECTED sell proceeds.
        def _px(o: Order) -> float:
            return self.funding_price(o, slippage_cap)

        if order_type == ORDER_MOC:
            if cash0 is not None:
                # nothing fills until the auction, so the sells' contribution is
                # an estimate — taken at the bid side, the floor on what they
                # can realise, so the budget errs small
                expected = sum(o.qty * _px(o) for o in sells)
                buys, dropped, note = fit_buys_to_funding(
                    buys, cash0 + expected, fractional, funding_buffer, _px)
                self._note_funding(note, dropped, results)
            results += self._run_leg("MOC", sells + buys, fractional, wait,
                                     order_type, slippage_cap, now, outside_rth)
            return results

        sell_results = self._run_leg("SELL", sells, fractional, wait, order_type,
                                     slippage_cap, now, outside_rth)
        results += sell_results

        lim = getattr(self, "safety", None)
        trip = (safety.fills_tripwire(sell_results, self.__dict__.get("_send_ref", {}),
                                      lim.max_fill_slippage) if lim else None)
        if trip and buys:
            print(f"    !! FILL-QUALITY TRIPWIRE — {trip}. The {len(buys)} buy(s) "
                  "are NOT sent.")
            self._alert(f"🔴 Fill-quality tripwire ({self.account}): {trip}. "
                        "Remaining buys withheld — check the market and the account.")
            results += [skipped_result(o, "SKIPPED-TRIPWIRE", "fill-quality tripwire")
                        for o in buys]
            return results

        if cash0 is not None:
            realised = sum(r["filled"] * (r["avg_fill_price"] or r["price"])
                           for r in sell_results)
            unfilled = [r for r in sell_results
                        if r["filled"] + 1e-9 < r["qty"]]
            if unfilled:
                print(f"    {len(unfilled)} sell(s) did not fill in full — the buy "
                      "leg is funded from what actually settled, not what was planned")
            available = cash0 + realised
            print(f"    Funding: cash ${cash0:,.0f} + realised sells "
                  f"${realised:,.0f} = ${available:,.0f}")
            buys, dropped, note = fit_buys_to_funding(buys, available, fractional,
                                                      funding_buffer, _px)
            self._note_funding(note, dropped, results)

        results += self._run_leg("BUY", buys, fractional, wait, order_type,
                                 slippage_cap, now, outside_rth)
        return results

    def _withhold_blocked_buys(self, buys: list[Order],
                               results: list[dict]) -> list[Order]:
        """Drop the buys the stress gate / NAV brake forbids (sells are never
        touched — de-risking must always be possible) and record each one."""
        if not buys or getattr(self, "safety", None) is None:
            return buys
        why_all, per_name = self._buy_block(buys)
        st = getattr(self, "_stress", None)
        if st is not None and st.notes:
            print("    Stress check: " + "; ".join(st.notes))
        keep: list[Order] = []
        dropped: list[tuple[Order, str]] = []
        for o in buys:
            why = why_all or per_name.get(o.symbol)
            (dropped.append((o, why)) if why else keep.append(o))
        if dropped:
            reasons = sorted({w for _, w in dropped})
            print(f"    !! BUYS WITHHELD ({len(dropped)}): " + "; ".join(reasons)
                  + " — sells proceed.")
            self._alert(f"⚠️ Buys withheld on {self.account}: "
                        + "; ".join(reasons)
                        + f". {len(dropped)} buy(s) not sent; sells proceed.")
            results += [skipped_result(o, "SKIPPED-STRESS", w) for o, w in dropped]
        return keep

    def _note_funding(self, note: str, dropped: list[Order],
                      results: list[dict]) -> None:
        """Log the funding decision and record dropped buys in the report."""
        print(f"    {note}")
        self.funding_note = note
        for o in dropped:
            print(f"    skipped BUY {o.qty:g} {o.symbol} — not funded")
            results.append(dict(key=o.key, symbol=o.symbol, action=o.action,
                                qty=float(o.qty), price=float(o.price),
                                status="SKIPPED-FUNDING", filled=0.0,
                                avg_fill_price=0.0, order_type="", limit_price=0.0,
                                reason=f"{o.reason} [not funded]"))

    def _run_leg(self, leg_name: str, leg_orders: list[Order], fractional: bool,
                 wait: float, order_type: str, slippage_cap: float, now,
                 outside_rth: bool) -> list[dict]:
        """Transmit one leg, wait for it, escalate the unfilled part, report."""
        results: list[dict] = []
        trades = []
        for o in leg_orders:
            halt = self._halt_reason()
            if halt:
                print(f"    SKIP {o.action} {o.symbol}: kill switch ({halt})")
                results.append(skipped_result(o, "SKIPPED-KILLSWITCH",
                                              "kill switch"))
                continue
            contract = sym.ibkr_contract(o.symbol)
            try:
                self._qualify(contract)
            except Exception as e:
                # No contract details = no order for this name. Say so in the
                # report like any other leg that could not be sent, and carry on
                # with the rest rather than hang or abort half-way.
                print(f"    SKIP {o.action} {o.symbol}: {e}")
                results.append(dict(key=o.key, symbol=o.symbol, action=o.action,
                                    qty=float(o.qty), price=float(o.price),
                                    status="SKIPPED-UNQUALIFIED", filled=0.0,
                                    avg_fill_price=0.0, order_type="",
                                    limit_price=0.0,
                                    reason=f"{o.reason} [contract not resolved]"))
                continue
            qty = executable_qty(o.action, o.qty, fractional)
            if qty <= 0:
                if not fractional and o.qty > 0:
                    print(f"    SKIP {o.action} {o.qty:g} {o.symbol}: less than one "
                          "whole share — fractional dust is left for you to sell "
                          "by hand")
                    results.append(dict(key=o.key, symbol=o.symbol, action=o.action,
                                        qty=float(o.qty), price=float(o.price),
                                        status="SKIPPED-FRACTIONAL", filled=0.0,
                                        avg_fill_price=0.0, order_type="",
                                        limit_price=0.0,
                                        reason=f"{o.reason} [fractional dust]"))
                continue
            if getattr(self, "safety", None) is not None:
                q = self.quote(contract)
                bad = self._unsafe_quote(o.symbol, o.action, q)
                if bad:
                    print(f"    SKIP {o.action} {o.symbol}: {bad} — nothing sent "
                          "for this leg")
                    self._alert(f"⚠️ {o.action} {o.symbol} skipped on "
                                f"{self.account}: {bad}.")
                    results.append(skipped_result(o, "SKIPPED-UNSAFE-QUOTE", bad,
                                                  qty))
                    continue
                ref = safety.touch_price(o.action, q)
                if ref:
                    self.__dict__.setdefault("_send_ref", {})[(o.action, o.symbol)] = ref
            order, used, lmt = self._build_order(o, qty, order_type,
                                                 slippage_cap, contract,
                                                 outside_rth)
            if order is None:
                # Unsendable outside RTH: no quote AND no usable book price, so
                # there is nothing to price a limit off and MARKET is rejected.
                # Record it the way an unfunded buy is recorded — a leg that
                # silently vanishes from the report leaves the position open
                # with nothing anywhere saying why (a full close of a name the
                # book has dropped carries no exec_price, so it is exactly the
                # leg most likely to land here).
                results.append(dict(key=o.key, symbol=o.symbol, action=o.action,
                                    qty=float(qty), price=float(o.price),
                                    status="SKIPPED-UNPRICEABLE", filled=0.0,
                                    avg_fill_price=0.0, order_type="",
                                    limit_price=0.0,
                                    reason=f"{o.reason} [no price outside RTH]"))
                continue
            # Explicit, on every order: a login with several accounts routes an
            # unstamped order to its DEFAULT account, which is not necessarily
            # the one this run was told to trade.
            order.account = self.account
            if getattr(self, "order_ref", ""):
                order.orderRef = self.order_ref
            trades.append((o, qty, used, lmt, self.ib.placeOrder(contract, order)))
            self.orders_sent = getattr(self, "orders_sent", 0) + 1
            print(f"    sent  {o.action:4s} {qty:g} {o.symbol} [{used}] "
                  f"-> {self.account}" + (" outsideRth" if outside_rth else ""))
        if not trades:
            return results
        if order_type == ORDER_MOC:
            # A MOC the broker refuses (restricted product, IRA rule, no
            # permission) shows up within seconds; saying so NOW leaves the
            # operator the minutes before the cutoff to act, instead of
            # discovering it from the 4 PM report.
            self.ib.sleep(5.0)
            alerted: set[str] = set()
            self._alert_moc_rejections(trades, alerted)
            deadline = moc_wait_seconds(now)
            print(f"    waiting {deadline/60:.0f} min for the closing "
                  f"auction to print…")
            if moc_wait_capped(now):
                print(f"    !! the {MOC_MAX_WAIT_S/60:.0f}-minute wait ceiling ends BEFORE "
                      "the auction prints (this run started very early): the report "
                      "will be written before the fills — the after-the-close "
                      "reconcile task restates it from the broker's real fills.")
        else:
            deadline = wait              # wait for this leg before the next
        ticks = 0
        while deadline > 0 and any(not t.isDone() for *_, t in trades):
            if self._halt_reason():
                break
            self.ib.sleep(1.0)
            deadline -= 1.0
            ticks += 1
            if order_type == ORDER_MOC and ticks % 60 == 0:
                # a refusal can arrive minutes after submission; still worth
                # hearing about while the cutoff has not passed
                self._alert_moc_rejections(trades, alerted)
        if self._halt_reason():
            self._cancel_working(trades)
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
                                reason=o.reason,
                                broker_msg=broker_message(t, st)))
        return results

    def disconnect(self) -> None:
        try:
            self.ib.disconnect()
        except Exception:
            pass
