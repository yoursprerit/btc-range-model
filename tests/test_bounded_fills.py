"""Every order a marketable-limit run sends carries a price — the two paths
that used to degrade to a naked MARKET order no longer do.

1. A leg whose quote cannot price a limit inside regular hours is SKIPPED
   (``SKIPPED-UNPRICEABLE``) and alerted, not sent as MARKET.
2. A limit unfilled after the wait is re-sent ONCE as a wider limit —
   ``ESCALATION_SLIPPAGE_CAP`` through the THEN-CURRENT touch — and a
   remainder that fails that too is cancelled and alerted.  No MARKET order.

An explicit ``--order-type market`` is untouched: that is a choice, not a
fallback.
"""
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ibkr_common as ic                      # noqa: E402
import ibkr_safety as ibs                     # noqa: E402

NAN = float("nan")


# ── a fake ib_async: real-shaped LimitOrder / MarketOrder, no stubbed _build_order
class _Trade:
    def __init__(self, order, qty, fill_px, done):
        self.order = order
        self._done = done
        self.orderStatus = type("S", (), {"status": "Filled" if done else "Submitted",
                                          "filled": qty if done else 0.0,
                                          "avgFillPrice": fill_px})()

    def isDone(self):
        return self._done


class _IB:
    def __init__(self, quotes, done=True, done_on_escalation=True):
        self.quotes, self.sent, self.cancelled = quotes, [], []
        self.done, self.done_esc = done, done_on_escalation

    def qualifyContracts(self, contract):
        return [contract]

    def reqTickers(self, contract):
        q = self.quotes.get(contract.symbol)
        return [type("T", (), q)()] if q else []

    def placeOrder(self, contract, order):
        self.sent.append((order, contract.symbol))
        done = self.done if len(self.sent) == 1 else self.done_esc
        return _Trade(order, float(order.totalQuantity), 100.0, done)

    def cancelOrder(self, order):
        self.cancelled.append(order)

    def sleep(self, _s):
        pass


@pytest.fixture(autouse=True)
def _fake_ib_async(monkeypatch):
    mod = type(sys)("ib_async")

    class MarketOrder:
        def __init__(self, action, qty):
            self.action, self.totalQuantity, self.orderType = action, qty, "MKT"

    class LimitOrder:
        def __init__(self, action, qty, lmt):
            self.action, self.totalQuantity, self.lmtPrice, self.orderType = action, qty, lmt, "LMT"

    class Order:
        def __init__(self, **kw):
            self.__dict__.update(kw); self.totalQuantity = kw.get("totalQuantity")
    mod.MarketOrder, mod.LimitOrder, mod.Order = MarketOrder, LimitOrder, Order
    monkeypatch.setitem(sys.modules, "ib_async", mod)
    monkeypatch.setattr(ic.sym, "ibkr_contract",
                        lambda symbol: type("C", (), {"symbol": symbol})())
    monkeypatch.setattr(ic.Broker, "index_quote", lambda self, s, e: {})


def _tk(bid=99.9, ask=100.1, last=100.0, close=100.0, halted=NAN):
    return dict(bid=bid, ask=ask, last=last, close=close, halted=halted)


def _broker(ib, safety=True):
    b = object.__new__(ic.Broker)
    b.ib, b.account, b.account_mode = ib, "U1", "live"
    b.cash = lambda: 1e6
    b.alerts = []
    if safety:
        b.safety = ibs.SafetyLimits(alert=lambda m: b.alerts.append(m))
    else:
        b._alert = lambda m: b.alerts.append(m)
    return b


def _orders(*rows):
    return [ic.Order(s, s, a, q, p, "") for s, a, q, p in rows]


def _types(ib):
    return [o.orderType for o, _ in ib.sent]


# ── 1. no quote → skip, not MARKET ───────────────────────────────────────────
def test_unpriceable_leg_is_skipped_inside_rth_not_sent_as_market():
    ib = _IB({"X": {}, "SPY": _tk()})                     # the feed knows nothing about X
    b = _broker(ib)
    res = b.place(_orders(("X", "BUY", 5, 100.0)), False, 0.0,
                  order_type=ic.ORDER_MARKETABLE_LIMIT)
    assert ib.sent == []
    assert res[0]["status"] == "SKIPPED-UNPRICEABLE"
    assert "no usable quote" in res[0]["reason"]
    assert any("No market order was sent" in a for a in b.alerts)


def test_unpriceable_leg_is_skipped_even_without_safety_limits():
    ib = _IB({"X": {}})
    b = _broker(ib, safety=False)
    res = b.place(_orders(("X", "SELL", 5, 100.0)), False, 0.0,
                  order_type=ic.ORDER_MARKETABLE_LIMIT)
    assert ib.sent == [] and res[0]["status"] == "SKIPPED-UNPRICEABLE"


def test_a_quoted_leg_still_goes_out_as_a_capped_limit():
    ib = _IB({"X": _tk(), "SPY": _tk()})
    b = _broker(ib)
    b.place(_orders(("X", "BUY", 5, 100.0), ("Y", "SELL", 0, 1.0)), False, 0.0,
            order_type=ic.ORDER_MARKETABLE_LIMIT)
    assert _types(ib) == ["LMT"]
    assert ib.sent[0][0].lmtPrice == pytest.approx(100.1 * 1.005, abs=0.01)


def test_explicit_market_order_type_is_untouched():
    ib = _IB({"X": _tk(), "SPY": _tk()})
    b = _broker(ib)
    b.place(_orders(("X", "BUY", 5, 100.0)), False, 0.0, order_type=ic.ORDER_MARKET)
    assert _types(ib) == ["MKT"]


# ── 2. escalation is a wider LIMIT, never MARKET ─────────────────────────────
def test_unfilled_limit_escalates_to_a_wider_limit_off_the_current_touch():
    ib = _IB({"X": _tk(), "SPY": _tk()}, done=False)
    b = _broker(ib)

    def drift(_s):                                       # the ask runs away while we wait
        ib.quotes["X"] = _tk(bid=101.9, ask=102.1, last=102.0)
    ib.sleep = drift
    b.place(_orders(("X", "BUY", 10, 100.0)), False, 2.0, order_type=ic.ORDER_MARKETABLE_LIMIT)
    assert _types(ib) == ["LMT", "LMT"], "re-sent as a LIMIT, not MKT"
    first, second = ib.sent[0][0], ib.sent[1][0]
    assert first.lmtPrice == pytest.approx(100.1 * 1.005, abs=0.01)
    assert second.lmtPrice == pytest.approx(102.1 * (1 + ic.ESCALATION_SLIPPAGE_CAP), abs=0.01)
    assert second.totalQuantity == 10
    assert len(ib.cancelled) == 1


def test_sell_escalation_widens_downward_off_the_bid():
    ib = _IB({"X": _tk(), "SPY": _tk()}, done=False)
    b = _broker(ib)
    b.place(_orders(("X", "SELL", 10, 100.0)), False, 2.0, order_type=ic.ORDER_MARKETABLE_LIMIT)
    assert _types(ib) == ["LMT", "LMT"]
    assert ib.sent[1][0].lmtPrice == pytest.approx(99.9 * (1 - ic.ESCALATION_SLIPPAGE_CAP), abs=0.01)


def test_escalation_cap_is_wider_than_the_first_attempt_and_clears_the_spread_gate():
    assert ic.ESCALATION_SLIPPAGE_CAP > ic.DEFAULT_SLIPPAGE_CAP
    # a book the quote-sanity gate admits (≤2 % spread on buys) is crossed by it
    assert ic.ESCALATION_SLIPPAGE_CAP >= ibs.DEFAULT_MAX_SPREAD / 2


def test_second_attempt_unfilled_is_cancelled_and_alerted_not_marketed():
    ib = _IB({"X": _tk(), "SPY": _tk()}, done=False, done_on_escalation=False)
    b = _broker(ib)
    res = b.place(_orders(("X", "BUY", 10, 100.0)), False, 1.0, order_type=ic.ORDER_MARKETABLE_LIMIT)
    assert _types(ib) == ["LMT", "LMT"]                  # exactly two attempts, both priced
    assert len(ib.cancelled) == 2                         # …and both cancelled
    assert any("not sent as a market order" in a for a in b.alerts)
    assert all(r["order_type"] != ic.ORDER_MARKET for r in res)


def test_no_escalation_when_the_quote_has_vanished_or_gone_stale():
    ib = _IB({"X": _tk(), "SPY": _tk()}, done=False)
    b = _broker(ib)

    def feed_dies(_s):
        ib.quotes["X"] = dict(bid=NAN, ask=NAN, last=NAN, close=100.0, halted=NAN)
    ib.sleep = feed_dies
    b.place(_orders(("X", "BUY", 10, 100.0)), False, 2.0, order_type=ic.ORDER_MARKETABLE_LIMIT)
    assert _types(ib) == ["LMT"], "no second order off a prior-session close"
    assert any("left unfilled" in a and "No market order" in a for a in b.alerts)


def test_no_escalation_into_a_blown_out_book():
    ib = _IB({"X": _tk(), "SPY": _tk()}, done=False)
    b = _broker(ib)

    def blow_out(_s):
        ib.quotes["X"] = _tk(bid=95.0, ask=105.0)         # 10 % spread
    ib.sleep = blow_out
    b.place(_orders(("X", "BUY", 10, 100.0)), False, 2.0, order_type=ic.ORDER_MARKETABLE_LIMIT)
    assert _types(ib) == ["LMT"]
    assert any("spread" in a for a in b.alerts)


def test_no_market_order_is_ever_built_under_marketable_limit():
    """Belt and braces: grep the routing code — under the default type the only
    place a MarketOrder is constructed is the explicit ``market`` branch."""
    src = (ROOT / "scripts" / "ibkr_common.py").read_text()
    esc = src[src.index("    def _escalate_unfilled"):src.index("    def place(self, orders")]
    assert "MarketOrder" not in esc
    assert "falling back to MARKET" not in src
