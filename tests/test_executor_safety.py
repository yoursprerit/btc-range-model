"""Market-stress and execution-quality guards for the IBKR executor.

Each test pins one way a broken market (or a broken run) used to reach the
broker: a halted or crossed book, fills far through the touch, a crash day, an
account far below where the last run left it, a kill switch thrown mid-run.
The other half pins the promise that none of it changes an ordinary day: no
safety limits configured means the old behaviour, and an unmeasurable input
(no VIX feed, no quote) never blocks a rebalance.
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import ibkr_common as ic  # noqa: E402
import ibkr_safety as ibs  # noqa: E402

NAN = float("nan")


# ── 2. per-symbol quote sanity ───────────────────────────────────────────────
def _q(bid=99.9, ask=100.1, halted=NAN):
    return dict(bid=bid, ask=ask, last=100.0, close=100.0, halted=halted)


def test_a_normal_quote_is_fine():
    assert ibs.quote_problem(_q(), "BUY", 0.02) is None


def test_an_empty_or_partial_quote_is_unknown_not_bad():
    assert ibs.quote_problem({}, "BUY", 0.02) is None
    assert ibs.quote_problem({"bid": -1, "ask": NAN, "last": 100}, "BUY", 0.02) is None


def test_a_halted_symbol_is_refused_both_ways():
    for action in ("BUY", "SELL"):
        assert "halt" in ibs.quote_problem(_q(halted=1.0), action, 0.02)
    assert "volatility" in ibs.quote_problem(_q(halted=2.0), "BUY", 0.02)


def test_a_zero_halt_flag_is_not_a_halt():
    assert ibs.quote_problem(_q(halted=0.0), "BUY", 0.02) is None


def test_a_crossed_book_is_refused():
    assert "crossed" in ibs.quote_problem(_q(bid=101.0, ask=100.0), "BUY", 0.02)


def test_a_locked_book_is_fine():
    assert ibs.quote_problem(_q(bid=100.0, ask=100.0), "BUY", 0.02) is None


def test_a_wide_spread_blocks_a_buy_but_a_sell_gets_twice_the_room():
    wide = _q(bid=97.0, ask=100.0)                    # ~3.05%
    assert "spread" in ibs.quote_problem(wide, "BUY", 0.02)
    assert ibs.quote_problem(wide, "SELL", 0.02) is None
    assert "spread" in ibs.quote_problem(_q(bid=90.0, ask=100.0), "SELL", 0.02)


def test_zero_switches_the_spread_check_off_but_not_the_halt_check():
    wide = _q(bid=50.0, ask=100.0)
    assert ibs.quote_problem(wide, "BUY", 0) is None
    assert ibs.quote_problem(_q(halted=1.0), "BUY", 0) is not None


def test_outside_rth_only_a_halt_counts():
    """Extended-hours books are routinely crossed and wide."""
    assert ibs.quote_problem(_q(bid=101.0, ask=100.0), "BUY", 0.02,
                             outside_rth=True) is None
    assert ibs.quote_problem(_q(halted=1.0), "BUY", 0.02, outside_rth=True)


# ── 5. fill-quality tripwire ─────────────────────────────────────────────────
def _fill(action, symbol, avg, filled=10.0, order_type="marketable-limit"):
    return dict(action=action, symbol=symbol, filled=filled, qty=filled,
                avg_fill_price=avg, order_type=order_type)


def test_slippage_is_signed_against_us():
    assert ibs.adverse_slippage("BUY", 100.0, 101.0) == pytest.approx(0.01)
    assert ibs.adverse_slippage("BUY", 100.0, 99.0) == pytest.approx(-0.01)
    assert ibs.adverse_slippage("SELL", 100.0, 99.0) == pytest.approx(0.01)


def test_fills_inside_the_limit_do_not_trip():
    refs = {("SELL", "X"): 100.0}
    assert ibs.fills_tripwire([_fill("SELL", "X", 99.6)], refs, 0.01) is None


def test_a_fill_far_through_the_touch_trips_it():
    refs = {("SELL", "X"): 100.0}
    why = ibs.fills_tripwire([_fill("SELL", "X", 97.0)], refs, 0.01)
    assert why and "X" in why and "3.00%" in why


def test_a_better_than_touch_fill_never_trips():
    refs = {("BUY", "X"): 100.0}
    assert ibs.fills_tripwire([_fill("BUY", "X", 95.0)], refs, 0.01) is None


def test_moc_unfilled_and_unreferenced_fills_are_never_judged():
    refs = {("SELL", "X"): 100.0}
    assert ibs.fills_tripwire([_fill("SELL", "X", 50.0, order_type="moc")], refs, 0.01) is None
    assert ibs.fills_tripwire([_fill("SELL", "X", 50.0, filled=0.0)], refs, 0.01) is None
    assert ibs.fills_tripwire([_fill("SELL", "Y", 50.0)], refs, 0.01) is None
    assert ibs.fills_tripwire([_fill("SELL", "X", 50.0)], refs, 0) is None


# ── 1. stress gate ───────────────────────────────────────────────────────────
def _lim(**kw):
    return ibs.SafetyLimits(**kw)


def test_an_ordinary_day_is_not_stress():
    st = ibs.stress_assessment({"last": 99.0, "close": 100.0}, {"last": 22.0},
                               {"X": {"last": 95.0, "close": 100.0}}, _lim())
    assert st.block_buys is None and not st.names


def test_a_crash_in_spy_blocks_all_buys():
    st = ibs.stress_assessment({"last": 90.0, "close": 100.0}, None, {}, _lim())
    assert "SPY -10.0%" in st.block_buys
    assert st.buy_block_for("ANY")


def test_a_vix_spike_blocks_all_buys():
    st = ibs.stress_assessment(None, {"last": 61.0}, {}, _lim())
    assert "VIX 61.0" in st.block_buys


def test_one_name_gapping_blocks_only_that_buy():
    st = ibs.stress_assessment(None, None, {"X": {"last": 55.0, "close": 100.0},
                                            "Y": {"last": 101.0, "close": 100.0}}, _lim())
    assert st.block_buys is None
    assert st.buy_block_for("X") and st.buy_block_for("Y") is None


def test_unmeasurable_inputs_pass_and_say_so():
    st = ibs.stress_assessment(None, None, {}, _lim())
    assert st.block_buys is None
    assert any("not measured" in n for n in st.notes)


def test_each_stress_rule_switches_off_with_zero():
    st = ibs.stress_assessment({"last": 50.0, "close": 100.0}, {"last": 90.0},
                               {"X": {"last": 1.0, "close": 100.0}},
                               _lim(stress_spy_move=0, stress_vix=0, stress_name_move=0))
    assert st.block_buys is None and not st.names


# ── 6. NAV brake + kill switch ───────────────────────────────────────────────
def test_nav_brake_fires_only_on_a_big_drop():
    assert ibs.nav_drop(90_000, 100_000, 0.15) is None
    assert "15" in ibs.nav_drop(80_000, 100_000, 0.15) or ibs.nav_drop(80_000, 100_000, 0.15)
    assert ibs.nav_drop(100_000, 100_000, 0.15) is None
    assert ibs.nav_drop(50_000, 100_000, 0) is None          # off
    assert ibs.nav_drop(50_000, None, 0.15) is None          # no baseline
    assert ibs.nav_drop(120_000, 100_000, 0.15) is None      # up is never a brake


def _report(tmp_path, net_liq, age_days=1):
    p = tmp_path / "executed_book.json"
    stamp = datetime.now(timezone.utc) - timedelta(days=age_days)
    p.write_text(json.dumps({"net_liq": net_liq,
                             "generated_at_utc": stamp.isoformat(timespec="seconds")}))
    return p


def test_baseline_reads_last_runs_net_liq(tmp_path):
    assert ibs.read_baseline_net_liq(_report(tmp_path, 123_456.0)) == 123_456.0


def test_a_stale_missing_or_broken_report_is_no_baseline(tmp_path):
    assert ibs.read_baseline_net_liq(_report(tmp_path, 1e5, age_days=30)) is None
    assert ibs.read_baseline_net_liq(tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert ibs.read_baseline_net_liq(bad) is None
    zero = tmp_path / "zero.json"
    zero.write_text(json.dumps({"net_liq": 0.0, "generated_at_utc": "2026-01-01T00:00:00+00:00"}))
    assert ibs.read_baseline_net_liq(zero) is None


def test_halt_reason_reads_file_and_env(tmp_path, monkeypatch):
    monkeypatch.delenv("IBKR_TRADING_DISABLED", raising=False)
    f = tmp_path / "STOP"
    assert ibs.halt_reason(str(f)) is None and ibs.halt_reason(None) is None
    f.write_text("")
    assert "STOP" in ibs.halt_reason(str(f))
    f.unlink()
    monkeypatch.setenv("IBKR_TRADING_DISABLED", "1")
    assert ibs.halt_reason(None)


def test_arm_applies_the_nav_brake_and_alerts(tmp_path):
    sent = []
    b = object.__new__(ic.Broker)
    b.account = "U1"

    class A:                      # the argparse defaults
        max_spread = ibs.DEFAULT_MAX_SPREAD
        max_fill_slippage = ibs.DEFAULT_MAX_FILL_SLIPPAGE
        stress_spy_move = ibs.DEFAULT_STRESS_SPY_MOVE
        stress_vix = ibs.DEFAULT_STRESS_VIX
        stress_name_move = ibs.DEFAULT_STRESS_NAME_MOVE
        max_nav_drop = ibs.DEFAULT_MAX_NAV_DROP
        outside_rth = False
    orig = ibs.send_alert
    ibs.send_alert = sent.append
    try:
        ibs.arm(b, A, 70_000, _report(tmp_path, 100_000.0))
        assert "NAV brake" in b.buy_block and sent
        b2 = object.__new__(ic.Broker)
        b2.account = "U1"
        ibs.arm(b2, A, 99_000, _report(tmp_path, 100_000.0))
        assert getattr(b2, "buy_block", None) is None
        assert isinstance(b2.safety, ibs.SafetyLimits)
    finally:
        ibs.send_alert = orig


# ── broker integration: the same fake-IB pattern as test_executor_guards ────
class _Trade:
    def __init__(self, qty, price, done=True):
        self.order = object()
        self._done = done
        self.orderStatus = type("S", (), {"status": "Filled" if done else "Submitted",
                                          "filled": qty if done else 0.0,
                                          "avgFillPrice": price})()

    def isDone(self):
        return self._done


class _Sent(tuple):
    """(action, symbol, qty) — tuple-compatible with the existing assertions —
    plus ``.lmt``, the limit price if the order carried one."""
    def __new__(cls, action, symbol, qty, lmt=None):
        self = super().__new__(cls, (action, symbol, qty))
        self.lmt = lmt
        return self


class _IB:
    """Quotes and fill prices per symbol; records what was sent / cancelled."""
    def __init__(self, quotes=None, fills=None, on_sleep=None, done=True):
        self.quotes = quotes or {}
        self.fills = fills or {}
        self.sent, self.cancelled = [], []
        self._cur = None
        self.on_sleep = on_sleep
        self.done = done

    def qualifyContracts(self, contract):
        self._cur = contract.symbol
        return [contract]

    def reqTickers(self, contract):
        q = self.quotes.get(contract.symbol)
        return [type("T", (), q)()] if q else []

    def placeOrder(self, contract, order):
        qty = float(getattr(order, "totalQuantity", 0.0))
        self.sent.append(_Sent(order.action, contract.symbol, qty,
                               getattr(order, "lmtPrice", None)))
        return _Trade(qty, self.fills.get(contract.symbol, 100.0), self.done)

    def cancelOrder(self, order):
        self.cancelled.append(order)

    def sleep(self, _s):
        if self.on_sleep:
            self.on_sleep()


@pytest.fixture(autouse=True)
def _no_ib_async(monkeypatch):
    monkeypatch.setattr(ic.sym, "ibkr_contract",
                        lambda symbol: type("C", (), {"symbol": symbol})())

    def _stub_order(self, o, qty, order_type, slippage_cap, contract,
                    outside_rth=False):
        return (type("O", (), {"action": o.action, "totalQuantity": qty})(),
                order_type, 0.0)

    monkeypatch.setattr(ic.Broker, "_build_order", _stub_order)
    monkeypatch.setattr(ic.Broker, "index_quote", lambda self, s, e: {})

    class _MarketOrder:                      # stands in for ib_async.MarketOrder
        def __init__(self, action, qty):
            self.action, self.totalQuantity = action, qty

    class _LimitOrder:                       # …and ib_async.LimitOrder
        def __init__(self, action, qty, lmt):
            self.action, self.totalQuantity, self.lmtPrice = action, qty, lmt
    monkeypatch.setitem(sys.modules, "ib_async",
                        type(sys)("ib_async"))
    sys.modules["ib_async"].MarketOrder = _MarketOrder
    sys.modules["ib_async"].LimitOrder = _LimitOrder


def _tk(bid=99.9, ask=100.1, last=100.0, close=100.0, halted=NAN):
    return dict(bid=bid, ask=ask, last=last, close=close, halted=halted)


def _broker(ib, cash=1e6, safety=True, **lim_kw):
    b = object.__new__(ic.Broker)
    b.ib = ib
    b.account = "DU1"
    b.account_mode = "paper"
    b.cash = lambda: cash
    if safety:
        b.safety = ibs.SafetyLimits(alert=lambda m: b.alerts.append(m), **lim_kw)
        b.alerts = []
    return b


def _orders(*rows):
    return [ic.Order(s, s, a, q, p, "") for s, a, q, p in rows]


def _statuses(results):
    return {(r["action"], r["symbol"]): r["status"] for r in results}


def test_without_safety_limits_nothing_changes():
    """The promise: a Broker that never configured the guards trades as before."""
    ib = _IB({"X": _tk(halted=1.0), "SPY": _tk(last=50.0)})
    b = _broker(ib, safety=False)
    res = b.place(_orders(("X", "BUY", 5, 100.0)), False, 0.0, order_type=ic.ORDER_MARKET)
    assert ib.sent == [("BUY", "X", 5.0)]
    assert _statuses(res) == {("BUY", "X"): "Filled"}


def test_an_ordinary_day_sends_everything():
    ib = _IB({"X": _tk(), "Y": _tk(), "SPY": _tk(last=99.5)})
    b = _broker(ib)
    res = b.place(_orders(("X", "SELL", 3, 100.0), ("Y", "BUY", 5, 100.0)), False, 0.0,
                  order_type=ic.ORDER_MARKET)
    assert sorted(ib.sent) == [("BUY", "Y", 5.0), ("SELL", "X", 3.0)]
    assert not b.alerts
    assert not any(r["status"].startswith("SKIPPED") for r in res)


def test_a_halted_name_is_skipped_and_reported_the_rest_still_trade():
    ib = _IB({"X": _tk(halted=1.0), "Y": _tk(), "SPY": _tk()})
    b = _broker(ib)
    res = b.place(_orders(("X", "SELL", 3, 100.0), ("Y", "BUY", 5, 100.0)), False, 0.0,
                  order_type=ic.ORDER_MARKET)
    st = _statuses(res)
    assert st[("SELL", "X")] == "SKIPPED-UNSAFE-QUOTE"
    assert st[("BUY", "Y")] == "Filled"
    assert ib.sent == [("BUY", "Y", 5.0)]
    assert any("halted" in a for a in b.alerts)


def test_a_wide_spread_buy_is_skipped():
    ib = _IB({"X": _tk(bid=90.0, ask=100.0), "SPY": _tk()})
    b = _broker(ib)
    res = b.place(_orders(("X", "BUY", 5, 100.0)), False, 0.0, order_type=ic.ORDER_MARKET)
    assert _statuses(res) == {("BUY", "X"): "SKIPPED-UNSAFE-QUOTE"} and not ib.sent


def test_spy_crash_withholds_buys_but_sells_proceed():
    ib = _IB({"X": _tk(), "Y": _tk(), "SPY": _tk(last=88.0, close=100.0)})
    b = _broker(ib)
    res = b.place(_orders(("X", "SELL", 3, 100.0), ("Y", "BUY", 5, 100.0)), False, 0.0,
                  order_type=ic.ORDER_MARKET)
    st = _statuses(res)
    assert st[("BUY", "Y")] == "SKIPPED-STRESS" and st[("SELL", "X")] == "Filled"
    assert ib.sent == [("SELL", "X", 3.0)]
    assert any("SPY" in a for a in b.alerts)


def test_stress_also_withholds_moc_buys():
    ib = _IB({"X": _tk(), "Y": _tk(), "SPY": _tk(last=88.0, close=100.0)})
    b = _broker(ib)
    b.place(_orders(("X", "SELL", 3, 100.0), ("Y", "BUY", 5, 100.0)), False, 0.0,
            order_type=ic.ORDER_MOC)
    assert [a for a, *_ in ib.sent] == ["SELL"]


def test_vix_spike_withholds_buys(monkeypatch):
    ib = _IB({"Y": _tk(), "SPY": _tk()})
    b = _broker(ib)
    monkeypatch.setattr(ic.Broker, "index_quote", lambda self, s, e: {"last": 75.0})
    res = b.place(_orders(("Y", "BUY", 5, 100.0)), False, 0.0, order_type=ic.ORDER_MARKET)
    assert _statuses(res) == {("BUY", "Y"): "SKIPPED-STRESS"} and not ib.sent


def test_a_name_that_gapped_is_not_bought_but_others_are():
    ib = _IB({"X": _tk(last=50.0, close=100.0, bid=49.9, ask=50.1), "Y": _tk(),
              "SPY": _tk()})
    b = _broker(ib)
    res = b.place(_orders(("X", "BUY", 5, 100.0), ("Y", "BUY", 5, 100.0)), False, 0.0,
                  order_type=ic.ORDER_MARKET)
    st = _statuses(res)
    assert st[("BUY", "X")] == "SKIPPED-STRESS" and st[("BUY", "Y")] == "Filled"


def test_the_nav_brake_withholds_buys_and_lets_sells_go():
    ib = _IB({"X": _tk(), "Y": _tk(), "SPY": _tk()})
    b = _broker(ib)
    b.buy_block = "NAV brake — 20% below the last run"
    res = b.place(_orders(("X", "SELL", 3, 100.0), ("Y", "BUY", 5, 100.0)), False, 0.0,
                  order_type=ic.ORDER_MARKET)
    st = _statuses(res)
    assert st == {("SELL", "X"): "Filled", ("BUY", "Y"): "SKIPPED-STRESS"}


def test_bad_sell_fills_trip_the_wire_and_no_buys_are_sent():
    ib = _IB({"X": _tk(bid=100.0, ask=100.1), "Y": _tk(), "SPY": _tk()},
             fills={"X": 94.0})                      # sold 6% below the bid
    b = _broker(ib)
    res = b.place(_orders(("X", "SELL", 3, 100.0), ("Y", "BUY", 5, 100.0)), False, 0.0,
                  order_type=ic.ORDER_MARKET)
    st = _statuses(res)
    assert st[("SELL", "X")] == "Filled" and st[("BUY", "Y")] == "SKIPPED-TRIPWIRE"
    assert ib.sent == [("SELL", "X", 3.0)]
    assert any("tripwire" in a.lower() for a in b.alerts)


def test_good_sell_fills_leave_the_buys_alone():
    ib = _IB({"X": _tk(bid=100.0), "Y": _tk(), "SPY": _tk()}, fills={"X": 99.9})
    b = _broker(ib)
    b.place(_orders(("X", "SELL", 3, 100.0), ("Y", "BUY", 5, 100.0)), False, 0.0,
            order_type=ic.ORDER_MARKET)
    assert ("BUY", "Y", 5.0) in ib.sent


def test_the_kill_switch_is_checked_between_orders(tmp_path):
    stop = tmp_path / "STOP"
    ib = _IB({"X": _tk(), "Y": _tk(), "Z": _tk(), "SPY": _tk()})
    orig = ib.placeOrder

    def place_then_halt(contract, order):
        t = orig(contract, order)
        stop.write_text("")                      # thrown right after the first order
        return t
    ib.placeOrder = place_then_halt
    b = _broker(ib)
    b.safety.halt_check = lambda: ibs.halt_reason(str(stop))
    res = b.place(_orders(("X", "SELL", 1, 100.0), ("Y", "SELL", 1, 100.0),
                          ("Z", "BUY", 1, 100.0)), False, 0.0, order_type=ic.ORDER_MARKET)
    st = _statuses(res)
    assert st[("SELL", "X")] == "Filled"
    assert st[("SELL", "Y")] == "SKIPPED-KILLSWITCH" and st[("BUY", "Z")] == "SKIPPED-KILLSWITCH"
    assert len(ib.sent) == 1
    assert any("KILL SWITCH" in a for a in b.alerts)


def test_the_kill_switch_cancels_orders_still_working(tmp_path):
    stop = tmp_path / "STOP"
    ib = _IB({"X": _tk(), "SPY": _tk()}, on_sleep=lambda: stop.write_text(""), done=False)
    b = _broker(ib)
    b.safety.halt_check = lambda: ibs.halt_reason(str(stop))
    b.place(_orders(("X", "SELL", 1, 100.0)), False, 5.0, order_type=ic.ORDER_MARKETABLE_LIMIT)
    assert ib.cancelled, "an order left working when the switch is thrown must be cancelled"
    assert len(ib.sent) == 1, "…and nothing is escalated to a market order"


def test_an_unfilled_limit_is_not_escalated_into_a_halted_book():
    ib = _IB({"X": _tk(), "SPY": _tk()}, done=False)
    b = _broker(ib)
    orig = ib.sleep

    def halt_after_send(_s=None):
        ib.quotes["X"] = _tk(halted=1.0)           # the name halts while we wait
        orig(0)
    ib.sleep = halt_after_send
    b.place(_orders(("X", "BUY", 1, 100.0)), False, 2.0, order_type=ic.ORDER_MARKETABLE_LIMIT)
    assert len(ib.sent) == 1, "no MARKET order into a halted name"
    assert any("left unfilled" in a for a in b.alerts)


def test_an_unfilled_limit_still_escalates_in_a_healthy_book():
    ib = _IB({"X": _tk(), "SPY": _tk()}, done=False)
    b = _broker(ib)
    b.place(_orders(("X", "BUY", 1, 100.0)), False, 2.0, order_type=ic.ORDER_MARKETABLE_LIMIT)
    assert len(ib.sent) == 2, "ordinary behaviour kept: the remainder is re-sent"
    # …but as a WIDER LIMIT off the current ask, never as a market order
    assert ib.sent[-1].lmt == pytest.approx(100.1 * (1 + ic.ESCALATION_SLIPPAGE_CAP), abs=0.01)
