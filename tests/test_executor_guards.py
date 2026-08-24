"""Production guards for the IBKR executor.

These exist because of a real incident: on 2026-08-18 three runs each read the
account as flat, each re-bought the whole target book on top of the holdings
already there, and the paper account ended at 3.4x net liquidation on a $2.17M
margin loan (see IBKR_PAPER_TRADING.md). Every test here pins one of the checks
that makes that sequence impossible — an unverifiable positions read, a stale
book, a duplicate run, a levered plan, and buys funded past the cash that pays
for them.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import ibkr_common as ic  # noqa: E402


# ── the positions read ──────────────────────────────────────────────────────
def test_empty_read_against_a_funded_account_is_refused():
    """The exact August-18 state: the account holds ~$955k, the read says none."""
    ok, why = ic.position_read_is_trustworthy(0.0, 955_000.0, 995_000.0)
    assert not ok and "incomplete" in why


def test_partial_read_is_refused_too():
    # the 22:48 report saw 10 names but only ~$650k of ~$955k
    ok, why = ic.position_read_is_trustworthy(651_000.0, 955_000.0, 995_000.0)
    assert not ok and "missing" in why


def test_a_genuinely_flat_account_still_trades():
    ok, why = ic.position_read_is_trustworthy(0.0, 0.0, 1_000_000.0)
    assert ok and "flat" in why


def test_small_gap_inside_tolerance_passes():
    ok, _ = ic.position_read_is_trustworthy(950_000.0, 960_000.0, 1_000_000.0)
    assert ok, "1% of NAV is rounding, not a missing subscription"


# ── the stale book ──────────────────────────────────────────────────────────
def test_todays_book_is_the_one_that_trades():
    ok, why = ic.bar_is_current("2026-08-17", pd.Timestamp("2026-08-18"))
    assert ok and "last completed session" in why


def test_yesterdays_book_is_refused():
    """What the 36-hour freshness window let through."""
    ok, why = ic.bar_is_current("2026-08-16", pd.Timestamp("2026-08-18"))
    assert not ok and "STALE" in why and "1 session" in why


def test_weekend_gap_is_not_stale():
    # Monday 2026-08-17 trades Friday 2026-08-14's bar
    ok, _ = ic.bar_is_current("2026-08-14", pd.Timestamp("2026-08-17"))
    assert ok


def test_a_bar_from_the_future_is_refused():
    ok, why = ic.bar_is_current("2026-08-18", pd.Timestamp("2026-08-18"))
    assert not ok and "AHEAD" in why


# ── the two calendars: a weekend as_of over a Friday equity basis ───────────
# 2026-08-24 (Monday): the publisher runs 7 days a week for the 24/7 sleeves, so
# the book handed to that afternoon's executor was stamped as_of 2026-08-23
# (Sunday) while its equity basis was Friday's close. Judging as_of against the
# previous EQUITY session called that "AHEAD of the last completed session" and
# the executor traded nothing — every Monday, and every day after a US holiday.
def test_monday_book_with_a_weekend_bar_trades_on_its_friday_equity_basis():
    ok, why = ic.bar_is_current("2026-08-23", pd.Timestamp("2026-08-24"),
                                equity_close="2026-08-21")
    assert ok, why
    assert "equity basis 2026-08-21 is the last completed session" in why
    assert "24/7 sleeves" in why


def test_the_day_after_a_holiday_trades_too():
    """2026-09-07 is Labor Day: Tuesday's book carries a Monday as_of over
    Friday's close, which the as_of rule refused just like a Monday."""
    ok, why = ic.bar_is_current("2026-09-07", pd.Timestamp("2026-09-08"),
                                equity_close="2026-09-04")
    assert ok, why


def test_a_withheld_publish_is_still_refused_on_its_equity_basis():
    """The guard's whole point survives: a book whose equity basis is behind the
    last completed session is stale no matter what its as_of says. Tuesday
    2026-08-25 holding Monday's book (as_of Sunday, basis Friday) because
    Tuesday's publish was withheld — Monday's close has happened since."""
    ok, why = ic.bar_is_current("2026-08-23", pd.Timestamp("2026-08-25"),
                                equity_close="2026-08-21")
    assert not ok and "STALE" in why and "1 session" in why


def test_an_equity_basis_ahead_of_the_session_is_refused():
    ok, why = ic.bar_is_current("2026-08-18", pd.Timestamp("2026-08-18"),
                                equity_close="2026-08-18")
    assert not ok and "AHEAD" in why


def test_a_bar_dated_past_today_is_refused_even_with_a_good_basis():
    """A broken publisher, not a weekend sleeve — the basis must not excuse it."""
    ok, why = ic.bar_is_current("2026-08-25", pd.Timestamp("2026-08-24"),
                                equity_close="2026-08-21")
    assert not ok and "FUTURE" in why


def test_books_without_a_basis_stamp_keep_the_as_of_rule():
    """Published before 2026-07-23 → no signal_basis; nothing changes for them."""
    ok, _ = ic.bar_is_current("2026-08-17", pd.Timestamp("2026-08-18"))
    assert ok
    ok, why = ic.bar_is_current("2026-08-16", pd.Timestamp("2026-08-18"))
    assert not ok and "STALE" in why


def test_prev_trading_day_skips_holidays():
    # 2026-09-07 is Labor Day → the session before Tuesday the 8th is Friday the 4th
    assert ic.prev_trading_day(pd.Timestamp("2026-09-08")).date() == \
        pd.Timestamp("2026-09-04").date()


# ── exposure / turnover / account state ─────────────────────────────────────
def _orders(*specs):
    return [ic.Order(k, k, a, q, px, "") for k, a, q, px in specs]


def test_duplicate_buy_round_is_refused_by_the_exposure_cap():
    """Buying the book again on top of the book already held → ~2x NAV."""
    current = {"SOXX": 184.0}
    orders = _orders(("SOXX", "BUY", 184.0, 559.98))
    exp = ic.projected_exposure(orders, current, {"SOXX": 559.98}, 200_000.0)
    assert exp["leverage_after"] == pytest.approx(1.03, abs=0.01)
    ok, why = ic.check_exposure(exp, ic.DEFAULT_MAX_GROSS_FRAC)
    assert not ok and "duplicated" in why


def test_a_normal_rebalance_passes_the_exposure_cap():
    current = {"SOXX": 100.0}
    orders = _orders(("SOXX", "SELL", 20.0, 500.0), ("GDX", "BUY", 100.0, 90.0))
    exp = ic.projected_exposure(orders, current, {"SOXX": 500.0, "GDX": 90.0}, 100_000.0)
    ok, _ = ic.check_exposure(exp)
    assert ok and exp["leverage_after"] < 1.0


def test_existing_margin_loan_blocks_the_run():
    ok, why = ic.check_account_state(914_643, -2_168_660, 3_083_303)
    assert not ok and "margin loan" in why


def test_already_geared_account_blocks_the_run():
    ok, why = ic.check_account_state(914_643, 10_000, 3_083_303)
    assert not ok and "already at 3.37x" in why


def test_healthy_account_passes():
    ok, _ = ic.check_account_state(1_000_000, 39_253, 955_000)
    assert ok


def test_turnover_cap_catches_a_plan_built_on_a_bad_read():
    orders = _orders(("SOXX", "BUY", 1000.0, 560.0))      # $560k on a $200k account
    ok, why = ic.check_turnover(orders, 200_000.0)
    assert not ok and "implausible" in why
    ok, _ = ic.check_turnover(_orders(("SOXX", "BUY", 10.0, 560.0)), 200_000.0)
    assert ok


# ── book math ───────────────────────────────────────────────────────────────
def test_book_math_rejects_levered_or_broken_books():
    assert not ic.validate_book_math({"A": 0.7, "B": 0.7}, {"A": 1, "B": 1})[0]
    assert not ic.validate_book_math({"A": -0.1}, {"A": 1})[0]
    assert not ic.validate_book_math({"A": float("nan")}, {"A": 1})[0]
    assert not ic.validate_book_math({"A": 0.5}, {})[0]          # no price
    assert not ic.validate_book_math({"A": 0.5}, {"A": 0.0})[0]  # zero price
    ok, why = ic.validate_book_math({"A": 0.6, "B": 0.4}, {"A": 10.0, "B": 20.0})
    assert ok and "100.0%" in why


# ── session hours ───────────────────────────────────────────────────────────
def _et(stamp):
    return pd.Timestamp(stamp, tz=ic.ET_TZ)


def test_after_hours_run_is_refused_by_default():
    # the 2026-08-17 run that placed orders at 18:48 ET
    ok, why = ic.market_session_open(_et("2026-08-17 18:48"))
    assert not ok and "outside regular trading hours" in why


def test_the_executor_slot_is_inside_the_session():
    ok, _ = ic.market_session_open(_et("2026-08-18 15:30"))       # 2:30 PM CT
    assert ok


def test_outside_rth_is_still_possible_when_asked_for():
    ok, why = ic.market_session_open(_et("2026-08-17 18:48"), allow_outside=True)
    assert ok and "allowed explicitly" in why


# ── price drift (corporate actions) ─────────────────────────────────────────
def test_price_drift_spots_a_split():
    assert ic.price_drift(560.0, 140.0) == pytest.approx(0.75)
    assert ic.price_drift(560.0, 559.0) < 0.01
    assert ic.price_drift(0.0, 100.0) == 0.0      # unusable → no false positive


# ── funding: buys can only spend what the sells realised ────────────────────
def test_buys_are_trimmed_to_the_cash_that_funds_them():
    buys = _orders(("SOXX", "BUY", 184.0, 560.0), ("GDX", "BUY", 1947.0, 92.0))
    want = 184 * 560 + 1947 * 92
    sendable, dropped, note = ic.fit_buys_to_funding(buys, want / 4, fractional=False)
    sent = sum(o.qty * o.price for o in sendable)
    assert sent <= want / 4, "must never spend more than the funding available"
    assert "scaled to" in note
    assert all("trimmed to funding" in o.reason for o in sendable)


def test_unfunded_names_are_dropped_not_rounded_up():
    buys = _orders(("OIH", "BUY", 3.0, 430.0))
    sendable, dropped, _ = ic.fit_buys_to_funding(buys, 100.0, fractional=False)
    assert sendable == [] and [o.symbol for o in dropped] == ["OIH"]


def test_a_funded_plan_is_left_alone():
    buys = _orders(("SOXX", "BUY", 10.0, 100.0))
    sendable, dropped, note = ic.fit_buys_to_funding(buys, 10_000.0, fractional=False)
    assert [o.qty for o in sendable] == [10.0] and not dropped
    assert "no trimming" in note


def test_the_duplicate_round_would_have_been_trimmed_to_nothing():
    """With $39k cash and no sells, the $2.19M of duplicate buys shrink to ~2%."""
    buys = _orders(("GDX", "BUY", 5841.0, 90.28), ("ARTY", "BUY", 4800.0, 74.96),
                   ("SOXX", "BUY", 552.0, 538.64))
    sendable, _, _ = ic.fit_buys_to_funding(buys, 39_253.0, fractional=False)
    spent = sum(o.qty * o.price for o in sendable)
    assert spent <= 39_253.0
    assert spent < 40_000, "no margin loan is possible from a $39k cash balance"


# ── the placement sequence itself: sells settle before buys are sized ───────
class _FakeTrade:
    def __init__(self, filled, price):
        self.orderStatus = type("S", (), {"status": "Filled" if filled else "Cancelled",
                                          "filled": filled, "avgFillPrice": price})()

    def isDone(self):
        return True


class _FakeIB:
    """Enough of ib_async to drive Broker.place() without a connection."""
    def __init__(self, fill_ratio=1.0):
        self.fill_ratio = fill_ratio
        self.sent: list[tuple] = []

    def qualifyContracts(self, contract):
        return [contract]

    def placeOrder(self, contract, order):
        qty = float(getattr(order, "totalQuantity", 0.0))
        self.sent.append((getattr(order, "action", "?"), contract.symbol, qty))
        return _FakeTrade(qty * self.fill_ratio, 100.0)

    def sleep(self, _s):
        return None

    def reqTickers(self, contract):
        return []


@pytest.fixture(autouse=True)
def _no_ib_async(monkeypatch):
    """``ib_async`` is not installed on a test box — stub the two places
    :meth:`Broker.place` reaches for it (contract + order construction). The
    sequencing and funding logic under test is untouched."""
    monkeypatch.setattr(ic.sym, "ibkr_contract",
                        lambda symbol: type("C", (), {"symbol": symbol})())

    def _stub_order(self, o, qty, order_type, slippage_cap, contract,
                    outside_rth=False):
        return (type("O", (), {"action": o.action, "totalQuantity": qty})(),
                order_type, 0.0)

    monkeypatch.setattr(ic.Broker, "_build_order", _stub_order)


def _broker(fill_ratio=1.0, cash=0.0):
    b = object.__new__(ic.Broker)
    b.ib = _FakeIB(fill_ratio)
    b.account = "DU1"
    b.account_mode = "paper"
    b.cash = lambda: cash
    return b


def test_sells_are_sent_and_settled_before_any_buy():
    b = _broker(cash=0.0)
    orders = _orders(("SOXX", "SELL", 100.0, 100.0), ("GDX", "BUY", 50.0, 100.0))
    b.place(orders, fractional=False, wait=0.0, order_type=ic.ORDER_MARKET)
    actions = [a for a, _, _ in b.ib.sent]
    assert actions == ["SELL", "BUY"], "the buy leg must not precede the sell leg"


def test_buys_shrink_when_the_sells_do_not_fill():
    """A sell that only half-fills funds only half the buy — never a margin loan."""
    b = _broker(fill_ratio=0.5, cash=0.0)
    orders = _orders(("SOXX", "SELL", 100.0, 100.0), ("GDX", "BUY", 100.0, 100.0))
    b.place(orders, fractional=False, wait=0.0, order_type=ic.ORDER_MARKET)
    bought = [q for a, _, q in b.ib.sent if a == "BUY"]
    assert bought and bought[0] <= 50, f"bought {bought} on 5,000 of proceeds"


def test_no_cash_and_no_sells_means_no_buys():
    """The August-18 duplicate round: $39k cash, nothing sold, $2.19M of buys."""
    b = _broker(cash=39_253.0)
    orders = _orders(("GDX", "BUY", 5841.0, 90.28), ("SOXX", "BUY", 552.0, 538.64))
    results = b.place(orders, fractional=False, wait=0.0, order_type=ic.ORDER_MARKET)
    spent = sum(q * 90.28 for a, sym_, q in b.ib.sent if a == "BUY" and sym_ == "GDX")
    spent += sum(q * 538.64 for a, sym_, q in b.ib.sent if a == "BUY" and sym_ == "SOXX")
    assert spent <= 39_253.0
    assert any(r["status"] == "SKIPPED-FUNDING" or "trimmed" in (r["reason"] or "")
               for r in results), "the report must show what was not bought"


def test_allow_margin_restores_the_old_behaviour():
    b = _broker(cash=0.0)
    orders = _orders(("GDX", "BUY", 100.0, 100.0))
    b.place(orders, fractional=False, wait=0.0, order_type=ic.ORDER_MARKET,
            allow_margin=True)
    assert [q for a, _, q in b.ib.sent if a == "BUY"] == [100.0]


# ── the wiring: holdings() refuses to hand back an unverifiable read ────────
class _PortfolioIB(_FakeIB):
    def __init__(self, positions, portfolio):
        super().__init__()
        self._positions, self._portfolio = positions, portfolio

    def positions(self, _account=None):
        return self._positions

    def portfolio(self, _account=None):
        return self._portfolio

    def reqPositions(self):
        return None


def _pos(symbol, shares):
    return type("P", (), {"contract": type("C", (), {"symbol": symbol})(),
                          "position": shares, "avgCost": 100.0})()


def _item(symbol, shares, mv):
    return type("I", (), {"contract": type("C", (), {"symbol": symbol})(),
                          "position": shares, "averageCost": 100.0,
                          "marketValue": mv})()


def _wired(positions, portfolio, gross):
    b = object.__new__(ic.Broker)
    b.ib = _PortfolioIB(positions, portfolio)
    b.account = "DU1"
    b.gross_position_value = lambda: gross
    return b


def test_holdings_raises_when_the_read_is_empty_but_the_account_is_not():
    b = _wired([], [], gross=955_000.0)
    with pytest.raises(ic.PositionReadError, match="incomplete"):
        b.holdings(net_liq=995_000.0)


def test_holdings_returns_a_verified_read():
    b = _wired([_pos("SOXX", 184.0)], [_item("SOXX", 184.0, 103_000.0)],
               gross=103_000.0)
    assert b.holdings(net_liq=995_000.0) == {"SOXX": 184.0}


def test_holdings_values_a_name_being_closed_too():
    """A held name that is no longer in the book must not look like missing value."""
    b = _wired([_pos("PBW", 8114.0)], [_item("PBW", 8114.0, 281_000.0)],
               gross=281_000.0)
    assert b.holdings(net_liq=995_000.0) == {"PBW": 8114.0}


def test_a_small_fee_debit_is_not_a_margin_loan():
    ok, _ = ic.check_account_state(1_000_000, -120.0, 950_000)
    assert ok, "a fee debit must not block the daily run"
    ok, _ = ic.check_account_state(1_000_000, -50_000.0, 950_000)
    assert not ok


def test_an_unreadable_cash_balance_fails_closed():
    """Not being able to see the cash is not permission to spend it."""
    b = _broker(cash=0.0)
    def _boom():
        raise RuntimeError("no TotalCashValue")
    b.cash = _boom
    orders = _orders(("GDX", "BUY", 100.0, 100.0))
    b.place(orders, fractional=False, wait=0.0, order_type=ic.ORDER_MARKET)
    assert [q for a, _, q in b.ib.sent if a == "BUY"] == [], \
        "with no readable cash and no sell proceeds, nothing may be bought"


# ── the funding budget is priced off the live quote, not yesterday's close ──
class _QuotingIB(_FakeIB):
    """Serves a quote, so the funding budget prices the fill it will really get."""
    def __init__(self, bid, ask, fill_ratio=1.0):
        super().__init__(fill_ratio)
        self.bid, self.ask = bid, ask

    def reqTickers(self, contract):
        return [type("T", (), {"bid": self.bid, "ask": self.ask,
                               "last": self.ask, "close": self.ask})()]


def _quoting_broker(bid, ask, cash=0.0, fill_ratio=1.0):
    b = object.__new__(ic.Broker)
    b.ib = _QuotingIB(bid, ask, fill_ratio)
    b.account = "DU1"
    b.cash = lambda: cash
    return b


def test_funding_price_is_the_ceiling_a_buy_can_pay():
    b = _quoting_broker(bid=549.0, ask=550.0)
    o = ic.Order("SOXX", "SOXX", "BUY", 10.0, 500.0, "")     # book says $500
    px = b.funding_price(o, slippage_cap=0.005)
    assert px == pytest.approx(550.0 * 1.005, abs=0.01), "buy budgets at ask+cap"


def test_funding_price_is_the_floor_a_sell_can_realise():
    b = _quoting_broker(bid=549.0, ask=550.0)
    o = ic.Order("SOXX", "SOXX", "SELL", 10.0, 500.0, "")
    px = b.funding_price(o, slippage_cap=0.005)
    assert px == pytest.approx(549.0 * 0.995, abs=0.01), "sell budgets at bid-cap"


def test_funding_price_falls_back_to_the_book_when_unquotable():
    b = _broker()                       # _FakeIB.reqTickers returns []
    o = ic.Order("SOXX", "SOXX", "BUY", 10.0, 500.0, "")
    assert b.funding_price(o) == 500.0


def test_a_gap_up_no_longer_borrows_the_difference():
    """The residual this change closes: book $500, market $550, $50k of cash."""
    b = _quoting_broker(bid=549.0, ask=550.0, cash=50_000.0)
    orders = _orders(("SOXX", "BUY", 100.0, 500.0))          # $50k at book prices
    b.place(orders, fractional=False, wait=0.0, order_type=ic.ORDER_MARKET)
    qty = sum(q for a, _, q in b.ib.sent if a == "BUY")
    worst_case_spend = qty * 550.0 * 1.005
    assert worst_case_spend <= 50_000.0, (
        f"{qty:g} shares could cost ${worst_case_spend:,.0f} against $50k of cash")
    assert qty > 0, "a gap up should trim the order, not cancel the strategy"


def test_a_gap_down_still_buys_the_whole_plan():
    b = _quoting_broker(bid=449.0, ask=450.0, cash=50_000.0)
    orders = _orders(("SOXX", "BUY", 100.0, 500.0))
    b.place(orders, fractional=False, wait=0.0, order_type=ic.ORDER_MARKET)
    assert [q for a, _, q in b.ib.sent if a == "BUY"] == [100.0]


def test_quotes_are_cached_within_a_run():
    b = _quoting_broker(bid=549.0, ask=550.0)
    calls = []
    inner = b.ib.reqTickers
    b.ib.reqTickers = lambda c: (calls.append(c) or inner(c))
    b.quote_by_symbol("SOXX"); b.quote_by_symbol("SOXX")
    assert len(calls) == 1, "the drift check and the funding budget share one quote"


# ── one implementation, both entry points ───────────────────────────────────
def test_guards_from_args_falls_back_to_the_production_defaults():
    class _Args:                       # an entry point that exposes only some
        max_gross_frac = 1.5
        allow_margin = True
    g = ic.Guards.from_args(_Args())
    assert g.max_gross_frac == 1.5 and g.allow_margin is True
    assert g.max_turnover_frac == ic.Guards().max_turnover_frac
    assert g.max_price_drift == ic.Guards().max_price_drift
    assert g.outside_rth is False, "an unset flag must never loosen a guard"


def test_both_trading_entry_points_run_the_shared_preflight():
    """They place orders through the same code, so they must refuse the same
    things — a regression here is how the two paths drift apart."""
    root = Path(__file__).resolve().parent.parent / "scripts"
    for name in ("ibkr_execute_book.py", "ibkr_rebalance.py"):
        src = (root / name).read_text()
        assert "preflight(" in src, f"{name} does not run the shared pre-flight"
        assert "post_trade_check(" in src, f"{name} does not verify after trading"
        assert "broker.holdings(" in src, f"{name} uses an unverified positions read"
        assert "positions_by_key()" not in src.split("def main")[-1], \
            f"{name} still sizes off a raw positions read"


def test_the_shared_preflight_stops_a_levered_plan(monkeypatch):
    """End-to-end through preflight(), the way both entry points call it."""
    b = _quoting_broker(bid=559.0, ask=560.0, cash=39_253.0)
    b.gross_position_value = lambda: 955_000.0
    current = {"SOXX": 184.0}
    orders = _orders(("SOXX", "BUY", 184.0, 560.0))        # the duplicate round
    ok, why = ic.preflight(b, orders, current, {"SOXX": 560.0}, 200_000.0,
                           ic.Guards(outside_rth=True))
    assert not ok and ("already at" in why or "net liq" in why)


# ── IBKR P&L fields: the Double.MAX_VALUE "not available" sentinel ──────────
# IBKR does not omit a P&L it cannot supply — it sends Double.MAX_VALUE
# (~1.8e308). A raw float() of that lands in the execution report and renders
# as an absurd number in the app, so it has to be read as "unknown" instead.

def test_the_max_value_sentinel_reads_as_unknown_not_a_number():
    assert ic._pnl_value(1.7976931348623157e308) is None
    assert ic._pnl_value(-1.7976931348623157e308) is None


def test_real_pnl_values_survive_including_zero():
    assert ic._pnl_value(127.6) == pytest.approx(127.6)
    assert ic._pnl_value(-12.4) == pytest.approx(-12.4)
    assert ic._pnl_value(0.0) == 0.0             # a real zero, not "unknown"
    assert ic._pnl_value("127.6") == pytest.approx(127.6)


def test_missing_and_unparsable_pnl_reads_as_unknown():
    for bad in (None, "", "n/a", float("nan"), object()):
        assert ic._pnl_value(bad) is None
