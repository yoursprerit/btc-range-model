"""Two live-trading fixes found in the first multi-account preview.

1. Fractional holdings + whole-share MOC: a SELL used to be rounded UP (22.8672
   held -> 23 sent), selling a share the account does not own.
2. A slow per-account "portfolio" subscription made the positions cross-check
   compare against $0 and refuse a perfectly good account.
"""
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "app"))

import ibkr_common as ic  # noqa: E402


# ── 1. whole-share sizing ───────────────────────────────────────────────────
def test_sells_round_down_buys_round_to_nearest():
    assert ic.executable_qty("SELL", 22.8672, False) == 22
    assert ic.executable_qty("SELL", 263.165, False) == 263
    assert ic.executable_qty("SELL", 0.87, False) == 0
    assert ic.executable_qty("BUY", 31.9419, False) == 32
    assert ic.executable_qty("BUY", 10.2, False) == 10
    assert ic.executable_qty("SELL", 22.8672, True) == pytest.approx(22.8672)


def test_plan_sells_whole_shares_only_and_reports_the_dust():
    prices = {"XLE": 65.08, "GRID": 180.89}
    held = {"XLE": 354.165, "GRID": 22.0581, "XOP": 22.8672}
    skipped = []
    orders = ic.build_order_plan({"XLE": 0.0182, "GRID": 0.125}, prices, 77_641.0,
                                 held, 0.01, fractional=False, skipped=skipped)
    by = {o.symbol: o for o in orders}
    # XLE: hold 354.165 -> target round(0.0182*77641/65.08)=22 -> sell 332.165 -> 332
    assert by["XLE"].action == "SELL" and by["XLE"].qty == float(int(by["XLE"].qty))
    assert by["XLE"].qty <= 354.165
    # GRID bought in whole shares
    assert by["GRID"].action == "BUY" and by["GRID"].qty == float(int(by["GRID"].qty))
    # XOP is not in the book: close it, but only the 22 whole shares
    assert by["XOP"].action == "SELL" and by["XOP"].qty == 22.0


def test_dust_is_reported_not_silently_dropped():
    skipped = []
    orders = ic.build_order_plan({}, {}, 50_000.0, {"XLE": 0.87}, 0.01,
                                 fractional=False, skipped=skipped)
    assert orders == []
    assert skipped and skipped[0]["symbol"] == "XLE" and skipped[0]["held"] == 0.87
    skipped = []
    ic.build_order_plan({}, {}, 50_000.0, {"XLE": 0.3}, 0.01, fractional=False,
                        skipped=skipped)
    assert skipped and "less than one whole share" in skipped[0]["reason"]


def test_whole_holdings_plan_is_unchanged():
    prices = {"XLE": 50.0, "GRID": 100.0}
    a = ic.build_order_plan({"XLE": 0.3, "GRID": 0.2}, prices, 100_000.0,
                            {"XLE": 100.0}, 0.01, fractional=False)
    # XLE target 0.3*100000/50 = 600 (hold 100 -> buy 500); GRID 0.2*100000/100 = 200
    assert sorted((o.symbol, o.action, o.qty) for o in a) == [("GRID", "BUY", 200.0),
                                                              ("XLE", "BUY", 500.0)]


def test_random_fractional_books_never_oversell():
    rng = random.Random(7)
    keys = ["XLE", "GRID", "ERX", "XOP", "SATA"]
    for _ in range(300):
        held = {k: rng.choice([0.0, rng.uniform(0.1, 400.0), float(rng.randint(1, 300))])
                for k in keys}
        w = {k: rng.uniform(0, 0.4) for k in keys if rng.random() < 0.6}
        px = {k: rng.uniform(20, 300) for k in keys}
        for o in ic.build_order_plan(w, px, 80_000.0, held, 0.0, fractional=False):
            assert o.qty == float(int(o.qty)) and o.qty >= 1
            if o.action == "SELL":
                assert o.qty <= held[o.key] + 1e-9, (o, held[o.key])


class _Trade:
    def __init__(self, order, symbol):
        self.order, self.contract = order, type("C", (), {"symbol": symbol})()
        self.orderStatus = type("S", (), {"status": "Filled", "filled": order.totalQuantity,
                                          "avgFillPrice": 1.0})()
        self.log = []

    def isDone(self):
        return True


class _SendIB:
    def __init__(self):
        self.placed = []

    def qualifyContracts(self, *_):
        pass

    def placeOrder(self, contract, order):
        self.placed.append((contract.symbol, order.action, order.totalQuantity))
        return _Trade(order, contract.symbol)

    def sleep(self, _s):
        pass


def _send_broker():
    b = object.__new__(ic.Broker)
    b.ib, b.account, b.account_mode = _SendIB(), "U1", "live"
    b.cash = lambda: 1e9
    return b


def test_send_time_safety_net_floors_a_sell_and_reports_dust(monkeypatch):
    pytest.importorskip("ib_async")
    monkeypatch.setattr(ic.sym, "ibkr_contract",
                        lambda s: type("C", (), {"symbol": s})())
    b = _send_broker()
    orders = [ic.Order("OIH", "OIH", "SELL", 22.8672, 100.0, "close"),
              ic.Order("XLE", "XLE", "SELL", 0.4, 65.0, "dust")]
    res = b.place(orders, fractional=False, wait=0.0, order_type=ic.ORDER_MOC,
                  now=__import__("pandas").Timestamp("2026-10-12 15:30", tz="America/New_York"),
                  moc_strict=True)
    assert b.ib.placed == [("OIH", "SELL", 22.0)]
    dust = [r for r in res if r["status"] == "SKIPPED-FRACTIONAL"]
    assert len(dust) == 1 and dust[0]["symbol"] == "XLE"


# ── 2. a slow account-updates subscription ──────────────────────────────────
class _Contract:
    def __init__(self, symbol):
        self.symbol, self.multiplier = symbol, ""


class _SyncIB:
    """A gateway whose portfolio arrives only after N refresh requests."""

    def __init__(self, positions, gross, nl, arrives_after=None, quotes=None):
        self._pos = positions               # {symbol: (shares, avg)}
        self.gross, self.nl = gross, nl
        self.arrives_after = arrives_after  # refresh count at which it arrives
        self.refreshes = 0
        self.quotes = quotes or {}
        self.tickers_asked = 0

    def reqPositions(self):
        pass

    def sleep(self, _s):
        pass

    def reqAccountUpdates(self, _acct):
        self.refreshes += 1

    def positions(self, _a=None):
        return [type("P", (), {"contract": _Contract(s), "position": sh,
                               "avgCost": avg})() for s, (sh, avg) in self._pos.items()]

    def portfolio(self, _a=None):
        if self.arrives_after is None or self.refreshes < self.arrives_after:
            return []
        return [type("I", (), {"contract": _Contract(s), "position": sh,
                               "averageCost": avg, "marketValue": sh * self.quotes.get(s, avg),
                               "marketPrice": self.quotes.get(s, avg),
                               "unrealizedPNL": 0.0, "realizedPNL": 0.0})()
                for s, (sh, avg) in self._pos.items()]

    def accountSummary(self, _a=None):
        return [type("V", (), {"tag": "GrossPositionValue", "value": str(self.gross),
                               "currency": "USD"})(),
                type("V", (), {"tag": "NetLiquidation", "value": str(self.nl),
                               "currency": "USD"})()]

    def reqTickers(self, contract):
        self.tickers_asked += 1
        px = self.quotes.get(contract.symbol)
        if px is None:
            return []
        return [type("T", (), {"bid": px - 0.01, "ask": px + 0.01, "last": px,
                               "close": px})()]


def _bk(ib):
    b = object.__new__(ic.Broker)
    b.ib, b.account, b.account_mode = ib, "U1", "live"
    return b


POS = {"XLE": (100.0, 60.0), "GRID": (50.0, 170.0)}      # 100*65 + 50*180 = 15,500
QUOTES = {"XLE": 65.0, "GRID": 180.0}


def test_healthy_portfolio_needs_no_retry_or_quotes():
    ib = _SyncIB(POS, gross=15_500.0, nl=50_000.0, arrives_after=0, quotes=QUOTES)
    cur = _bk(ib).holdings(50_000.0)
    assert set(cur) >= {"XLE", "GRID"}
    assert ib.refreshes == 0 and ib.tickers_asked == 0


def test_portfolio_arriving_on_a_retry_is_accepted():
    ib = _SyncIB(POS, gross=15_500.0, nl=50_000.0, arrives_after=2, quotes=QUOTES)
    cur = _bk(ib).holdings(50_000.0)
    assert set(cur) >= {"XLE", "GRID"}
    assert ib.refreshes == 2 and ib.tickers_asked == 0


def test_portfolio_that_never_arrives_is_verified_against_live_quotes(capsys):
    ib = _SyncIB(POS, gross=15_500.0, nl=50_000.0, arrives_after=None, quotes=QUOTES)
    cur = _bk(ib).holdings(50_000.0)
    assert set(cur) >= {"XLE", "GRID"}
    assert ib.refreshes == ic.ACCOUNT_SYNC_RETRIES
    assert "valued at live quotes" in capsys.readouterr().out


def test_quotes_that_disagree_with_the_account_still_refuse():
    """The independent check is still a check: positions worth far less than
    the account reports are an incomplete read."""
    ib = _SyncIB(POS, gross=40_000.0, nl=50_000.0, arrives_after=None, quotes=QUOTES)
    with pytest.raises(ic.PositionReadError):
        _bk(ib).holdings(50_000.0)


def test_unpriceable_positions_still_refuse():
    ib = _SyncIB(POS, gross=15_500.0, nl=50_000.0, arrives_after=None, quotes={})
    with pytest.raises(ic.PositionReadError, match="could not be priced"):
        _bk(ib).holdings(50_000.0)


def test_one_unpriceable_name_is_not_a_partial_pass():
    ib = _SyncIB(POS, gross=15_500.0, nl=50_000.0, arrives_after=None,
                 quotes={"XLE": 65.0})
    assert _bk(ib).read_value_from_quotes() is None
    with pytest.raises(ic.PositionReadError):
        _bk(ib).holdings(50_000.0)


def test_flat_account_does_not_retry():
    ib = _SyncIB({}, gross=0.0, nl=50_000.0, arrives_after=None)
    assert _bk(ib).holdings(50_000.0) == {}
    assert ib.refreshes == 0


def test_report_snapshot_falls_back_to_positions_and_quotes():
    ib = _SyncIB(POS, gross=15_500.0, nl=50_000.0, arrives_after=None, quotes=QUOTES)
    snap = _bk(ib).portfolio_snapshot()
    by = {r["symbol"]: r for r in snap}
    assert by["XLE"]["shares"] == 100.0 and by["XLE"]["avg_cost"] == 60.0
    assert by["XLE"]["market_value"] == pytest.approx(6_500.0)
    assert by["XLE"]["unrealized_pnl"] == pytest.approx(500.0)
    assert by["XLE"]["realized_pnl"] is None        # unknown, not zero


def test_orchestrator_children_are_unbuffered_and_staggered_wider():
    import ibkr_execute_accounts as acc
    cfg, _ = acc.validate_config({"accounts": [{"id": "U1111111"}]})
    assert cfg.stagger_s >= 10
    assert "PYTHONUNBUFFERED" in (ROOT / "scripts" / "ibkr_execute_accounts.py").read_text()


# ── 3. a gateway that never answers must not hang the run ───────────────────
class _SilentGatewayIB(_SyncIB):
    """The blocking calls would wait forever; only the *Async twins are safe to
    use, and they never complete -- exactly an unserved account-updates feed."""

    def reqAccountUpdates(self, _acct):
        raise AssertionError("blocking reqAccountUpdates would hang the run")

    def reqPositions(self):
        raise AssertionError("blocking reqPositions would hang the run")

    def reqAllOpenOrders(self):
        raise AssertionError("blocking reqAllOpenOrders would hang the run")

    async def _never(self):
        import asyncio
        await asyncio.Event().wait()

    def reqAccountUpdatesAsync(self, _acct):
        self.refreshes += 1
        return self._never()

    def reqPositionsAsync(self):
        return self._never()

    def reqAllOpenOrdersAsync(self):
        return self._never()


def test_bounded_returns_instead_of_waiting_forever():
    pytest.importorskip("ib_async")
    import time
    ib = _SilentGatewayIB(POS, 15_500.0, 50_000.0)
    t0 = time.monotonic()
    done, res = _bk(ib)._bounded("reqAccountUpdates", "U1", timeout=0.3)
    assert (done, res) == (False, None) and time.monotonic() - t0 < 3


def test_bounded_passes_a_normal_answer_through():
    pytest.importorskip("ib_async")

    class _OK(_SyncIB):
        async def _v(self):
            return ["answer"]

        def reqAllOpenOrdersAsync(self):
            return self._v()
    done, res = _bk(_OK(POS, 1.0, 1.0))._bounded("reqAllOpenOrders", timeout=5)
    assert done and res == ["answer"]


def test_silent_gateway_falls_back_to_quotes_without_hanging(monkeypatch):
    pytest.importorskip("ib_async")
    import time
    monkeypatch.setattr(ic, "ACCOUNT_SYNC_WAIT_S", 0.3)
    monkeypatch.setattr(ic, "GATEWAY_REQUEST_TIMEOUT_S", 0.3)
    ib = _SilentGatewayIB(POS, gross=15_500.0, nl=50_000.0, arrives_after=None,
                          quotes=QUOTES)
    t0 = time.monotonic()
    cur = _bk(ib).holdings(50_000.0)
    assert set(cur) >= {"XLE", "GRID"}
    assert ib.refreshes == ic.ACCOUNT_SYNC_RETRIES
    assert time.monotonic() - t0 < 15, "the run must not wait on a silent gateway"


def test_silent_gateway_with_no_quotes_refuses_rather_than_hangs(monkeypatch):
    pytest.importorskip("ib_async")
    monkeypatch.setattr(ic, "ACCOUNT_SYNC_WAIT_S", 0.2)
    monkeypatch.setattr(ic, "GATEWAY_REQUEST_TIMEOUT_S", 0.2)
    ib = _SilentGatewayIB(POS, gross=15_500.0, nl=50_000.0, quotes={})
    with pytest.raises(ic.PositionReadError):
        _bk(ib).holdings(50_000.0)


def test_working_orders_fails_closed_when_the_gateway_is_silent(monkeypatch):
    pytest.importorskip("ib_async")
    monkeypatch.setattr(ic, "GATEWAY_REQUEST_TIMEOUT_S", 0.2)
    with pytest.raises(RuntimeError, match="refusing to send blind"):
        _bk(_SilentGatewayIB(POS, 1.0, 1.0)).working_orders()


def test_no_blocking_gateway_call_is_left_unbounded():
    """Guard against a new un-deadlined call sneaking back in."""
    src = (ROOT / "scripts" / "ibkr_common.py").read_text()
    for call in ("self.ib.reqAccountUpdates(", "self.ib.reqAllOpenOrders(",
                 "self.ib.reqPositions(", "self.ib.reqTickers(",
                 "self.ib.qualifyContracts("):
        assert call not in src, f"{call} blocks with no timeout; use _bounded"


def test_a_silent_gateway_cannot_hang_quotes_or_contract_lookup(monkeypatch):
    pytest.importorskip("ib_async")
    import time

    class _Silent(_SilentGatewayIB):
        def reqTickers(self, *_a, **_k):
            raise AssertionError("blocking reqTickers would hang")

        def reqTickersAsync(self, *_a, **_k):
            return self._never()

        def qualifyContracts(self, *_a):
            raise AssertionError("blocking qualifyContracts would hang")

        def qualifyContractsAsync(self, *_a):
            return self._never()
    monkeypatch.setattr(ic, "GATEWAY_REQUEST_TIMEOUT_S", 0.3)
    b = _bk(_Silent(POS, 1.0, 1.0))
    t0 = time.monotonic()
    assert b.quote(_Contract("XLE")) == {}
    with pytest.raises(TimeoutError):
        b._qualify(_Contract("XLE"))
    assert time.monotonic() - t0 < 10


def test_unresolvable_contract_skips_that_order_and_sends_the_rest(monkeypatch):
    pytest.importorskip("ib_async")
    monkeypatch.setattr(ic.sym, "ibkr_contract", lambda s: _Contract(s))
    b = _send_broker()

    def _qualify(contract):
        if contract.symbol == "BAD":
            raise TimeoutError("no contract details for BAD")
    b._qualify = _qualify
    res = b.place([ic.Order("BAD", "BAD", "BUY", 5.0, 10.0, "t"),
                   ic.Order("OK", "OK", "BUY", 5.0, 10.0, "t")],
                  fractional=False, wait=0.0, order_type=ic.ORDER_MOC,
                  now=__import__("pandas").Timestamp("2026-10-12 15:30", tz="America/New_York"),
                  moc_strict=True)
    assert b.ib.placed == [("OK", "BUY", 5.0)]
    assert [r["status"] for r in res if r["symbol"] == "BAD"] == ["SKIPPED-UNQUALIFIED"]
