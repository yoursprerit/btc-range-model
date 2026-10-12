"""Run-level health: partial runs, margin trouble, and runs that quietly do nothing.

The market-stress guards decide what a run may send.  These pin what the run
says about itself afterwards: it died half-way, the broker is already in
trouble, the kill switch was left on, or every leg was withheld.  None of them
is allowed to change an ordinary day.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ibkr_common as ic  # noqa: E402
import ibkr_execute_book as ex  # noqa: E402
import ibkr_safety as ibs  # noqa: E402
from test_executor_multi_account import (  # noqa: E402,F401  (env is a fixture)
    ACCT, _FakeBroker, _argv, _run, env)


# ── margin health ────────────────────────────────────────────────────────────
def test_a_healthy_or_unlevered_account_passes():
    assert ibs.margin_problem(95_000, 100_000, 0.05) is None
    assert ibs.margin_problem(5_000, 100_000, 0.05) is None        # exactly the floor


def test_a_margin_deficiency_is_refused():
    assert "deficiency" in ibs.margin_problem(0, 100_000, 0.05)
    assert "deficiency" in ibs.margin_problem(-1_200, 100_000, 0.05)


def test_a_thin_cushion_is_refused():
    why = ibs.margin_problem(3_000, 100_000, 0.05)
    assert why and "3.0%" in why


def test_unreadable_or_disabled_is_not_a_problem():
    assert ibs.margin_problem(None, 100_000, 0.05) is None
    assert ibs.margin_problem(float("nan"), 100_000, 0.05) is None
    assert ibs.margin_problem(-5, 100_000, 0) is None
    assert ibs.margin_problem(-5, 0, 0.05) is None


def test_the_preflight_check_reads_the_brokers_excess_liquidity():
    ok, _ = ic.check_margin_health(SimpleNamespace(excess_liquidity=lambda: 80_000),
                                   100_000, 0.05)
    assert ok
    ok, why = ic.check_margin_health(SimpleNamespace(excess_liquidity=lambda: -10),
                                     100_000, 0.05)
    assert not ok and "deficiency" in why


def test_a_broker_that_cannot_report_it_passes():
    for b in (SimpleNamespace(), SimpleNamespace(excess_liquidity=lambda: None),
              SimpleNamespace(excess_liquidity=lambda: 1 / 0)):
        assert ic.check_margin_health(b, 100_000, 0.05)[0]
    assert ic.check_margin_health(SimpleNamespace(excess_liquidity=lambda: -10),
                                  100_000, 0)[0], "0 switches it off"


def test_guards_default_the_cushion_but_leave_old_callers_alone():
    assert ic.Guards().min_margin_cushion == ibs.DEFAULT_MIN_MARGIN_CUSHION
    g = ic.Guards.from_args(SimpleNamespace(min_margin_cushion=0.2))
    assert g.min_margin_cushion == 0.2


def test_excess_liquidity_is_read_from_the_account_summary():
    b = object.__new__(ic.Broker)
    b.account = "U1"
    row = lambda tag, v, cur="USD": SimpleNamespace(tag=tag, value=v, currency=cur)  # noqa: E731
    b.ib = SimpleNamespace(accountSummary=lambda a: [row("NetLiquidation", "1"),
                                                     row("ExcessLiquidity", "12345.5")],
                           accountValues=lambda a: [])
    assert b.excess_liquidity() == 12345.5
    b.ib = SimpleNamespace(accountSummary=lambda a: [], accountValues=lambda a: [])
    assert b.excess_liquidity() is None


# ── market-wide halt ─────────────────────────────────────────────────────────
def test_a_halted_spy_blocks_buys():
    st = ibs.stress_assessment({"last": 100.0, "close": 100.0, "halted": 1.0}, None,
                               {}, ibs.SafetyLimits())
    assert st.block_buys and "halted" in st.block_buys


def test_a_normal_spy_does_not():
    st = ibs.stress_assessment({"last": 100.0, "close": 100.0, "halted": float("nan")},
                               None, {}, ibs.SafetyLimits())
    assert st.block_buys is None


# ── a run that sends nothing says so ─────────────────────────────────────────
def test_nothing_sent_is_called_out_only_when_orders_were_planned():
    skipped = [dict(status="SKIPPED-STRESS"), dict(status="SKIPPED-UNSAFE-QUOTE")]
    assert "NONE was sent" in ibs.nothing_sent(skipped, 2)
    assert ibs.nothing_sent(skipped, 0) is None, "an empty plan is a normal quiet day"
    assert ibs.nothing_sent(skipped + [dict(status="Filled")], 3) is None
    assert ibs.nothing_sent([dict(status="Submitted")], 1) is None


# ── main() end to end ────────────────────────────────────────────────────────
@pytest.fixture
def alerts(monkeypatch):
    sent = []
    monkeypatch.setattr(ibs, "send_alert", sent.append)
    return sent


def test_an_ordinary_run_raises_no_alert(env, monkeypatch, alerts):
    _, book = env
    assert _run(monkeypatch, _argv(book)) == 0
    assert alerts == []


def test_a_crash_after_orders_were_sent_is_a_loud_partial_run(env, monkeypatch, alerts):
    tmp, book = env

    class Dies(_FakeBroker):
        def place(self, orders, fractional, wait, **kw):
            self.orders_sent = 1                    # the sell leg went out…
            raise ConnectionError("gateway dropped")        # …then the link died
        def day_fills_by_symbol(self): return []

    monkeypatch.setattr(ex, "Broker", Dies)
    assert _run(monkeypatch, _argv(book)) == ex.EXIT_POST_TRADE
    assert any("PARTIAL RUN" in a and "gateway dropped" in a for a in alerts)
    assert (tmp / "accounts" / ACCT / "executed_book_live.json").exists(), \
        "the partial run is still recorded"


def test_a_crash_before_anything_was_sent_still_raises_as_before(env, monkeypatch, alerts):
    _, book = env

    class Dies(_FakeBroker):
        def place(self, orders, fractional, wait, **kw):
            raise ConnectionError("gateway dropped")

    monkeypatch.setattr(ex, "Broker", Dies)
    with pytest.raises(ConnectionError):
        _run(monkeypatch, _argv(book))
    assert not any("PARTIAL" in a for a in alerts)


def test_every_leg_withheld_is_alerted(env, monkeypatch, alerts):
    _, book = env

    class Withholds(_FakeBroker):
        def place(self, orders, fractional, wait, **kw):
            return [dict(key=o.key, symbol=o.symbol, action=o.action, qty=o.qty,
                         price=o.price, status="SKIPPED-STRESS", filled=0.0,
                         avg_fill_price=0.0, order_type="", limit_price=0.0,
                         reason=o.reason) for o in orders]

    monkeypatch.setattr(ex, "Broker", Withholds)
    _run(monkeypatch, _argv(book))
    assert any("NONE was sent" in a for a in alerts)


def test_a_margin_deficiency_aborts_with_an_alert_and_sends_nothing(env, monkeypatch, alerts):
    _, book = env

    class InTrouble(_FakeBroker):
        def excess_liquidity(self): return -2_500.0

    monkeypatch.setattr(ex, "Broker", InTrouble)
    assert _run(monkeypatch, _argv(book)) == ex.EXIT_ABORT
    (b,) = InTrouble.instances
    assert b.placed == []
    assert any("excess liquidity" in a for a in alerts)


def test_the_margin_check_can_be_switched_off(env, monkeypatch, alerts):
    _, book = env

    class InTrouble(_FakeBroker):
        def excess_liquidity(self): return -2_500.0

    monkeypatch.setattr(ex, "Broker", InTrouble)
    assert _run(monkeypatch, _argv(book, "--min-margin-cushion", "0")) == 0


def test_a_forgotten_kill_switch_is_not_silent(env, monkeypatch, alerts, tmp_path):
    _, book = env
    stop = tmp_path / "STOP"
    stop.write_text("")
    assert _run(monkeypatch, _argv(book, "--kill-switch-file", str(stop))) == 0
    assert _FakeBroker.instances == []
    assert any("kill switch" in a.lower() for a in alerts)
