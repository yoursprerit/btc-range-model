"""``ibkr_execute_book.main()`` driven end to end against a fake broker, for the
live multi-account / strict-MOC path: what it does when it is late, when the
account is busy, and where it writes."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "app"))

import ibkr_common as ic  # noqa: E402
import ibkr_execute_book as ex  # noqa: E402

ACCT = "U2222222"


class _FakeBroker:
    instances: list = []
    working: list = []
    fill_ratio = 1.0

    def __init__(self, host, port, client_id, **kw):
        type(self).instances.append(self)
        self.kw, self.port = kw, port
        self.account = kw.get("account")
        self.placed = []
        self.disconnected = False

    def net_liq(self): return 100_000.0
    def cash(self): return 100_000.0
    def holdings(self, _nl=None): return {}
    def gross_position_value(self): return 0.0
    def read_value(self): return 0.0
    def quote_by_symbol(self, _s): return {}
    def portfolio_snapshot(self): return []
    def realized_pnl(self): return None
    def positions_by_key(self, settle=True): return {}
    def working_orders(self): return list(type(self).working)
    def disconnect(self): self.disconnected = True

    def place(self, orders, fractional, wait, **kw):
        self.placed.append((orders, kw))
        return [dict(key=o.key, symbol=o.symbol, action=o.action, qty=o.qty,
                     price=o.price, status="Filled" if self.fill_ratio >= 1 else "Submitted",
                     filled=o.qty * self.fill_ratio, avg_fill_price=o.price,
                     order_type="moc", limit_price=0.0, reason=o.reason)
                for o in orders]


@pytest.fixture
def env(tmp_path, monkeypatch):
    _FakeBroker.instances, _FakeBroker.working, _FakeBroker.fill_ratio = [], [], 1.0
    monkeypatch.setattr(ex, "Broker", _FakeBroker)
    monkeypatch.setattr(ex, "ACCOUNTS_DIR", tmp_path / "accounts")
    monkeypatch.setattr(ex, "_push_report", lambda *_: None)
    monkeypatch.setattr(ic, "market_session_open", lambda *a, **k: (True, "open"))
    monkeypatch.setattr(ic, "bar_is_current", lambda *a, **k: (True, "current"))
    monkeypatch.setattr(ex, "bar_is_current", lambda *a, **k: (True, "current"))
    monkeypatch.setattr(ex, "is_trading_day", lambda d: (True, "trading day"))
    monkeypatch.setattr(ex.tb, "validate", lambda *a, **k: (True, "fresh"))
    monkeypatch.setattr(ic, "moc_entry_open", lambda now=None, cutoff=None: (True, "15:31 ET"))
    monkeypatch.delenv("IBKR_TRADING_DISABLED", raising=False)
    monkeypatch.delenv("OVERALL_BOOK_SECRET", raising=False)
    book = tmp_path / "book.json"
    book.write_text(json.dumps({
        "schema": "overall-target-book/v1", "as_of": "2026-10-09", "book_mode": "live",
        "generated_at_utc": "2026-10-12T12:15:00+00:00", "profile": "Balanced",
        "weights": {"GDX": 0.5}, "cash_weight": 0.5, "exec_price": {"GDX": 100.0},
    }))
    return tmp_path, book


def _argv(book, *extra):
    return ["--file", str(book), "--execute", "--account-mode", "live",
            "--confirm-live", "--account", ACCT, "--order-type", "moc",
            "--strict-moc", "--no-push-report", "--port", "4001", *extra]


def _run(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["ibkr_execute_book.py", *argv])
    return ex.main()


def test_happy_path_is_strict_moc_to_the_pinned_account(env, monkeypatch):
    tmp, book = env
    assert _run(monkeypatch, _argv(book)) == 0
    (b,) = _FakeBroker.instances
    assert b.kw["account"] == ACCT and b.kw["expected_account"] == ACCT
    assert b.kw["account_mode"] == "live" and b.kw["confirm_live"] is True
    (orders, kw), = b.placed
    assert kw["order_type"] == "moc" and kw["moc_strict"] is True
    assert kw["outside_rth"] is False
    assert [o.symbol for o in orders] == ["GDX"] and orders[0].qty == 500
    # the report lands in this account's own directory
    rep = tmp / "accounts" / ACCT / "executed_book_live.json"
    assert rep.exists()
    assert (tmp / "accounts" / ACCT / "executed_archive" / "2026-10-09_live.json").exists()
    assert json.loads(rep.read_text())["account"] == ACCT


def test_a_late_run_skips_the_session_and_never_connects(env, monkeypatch):
    _, book = env
    monkeypatch.setattr(ic, "moc_entry_open",
                        lambda now=None, cutoff=None: (False, "15:55 ET is past the 15:50 ET MOC entry cutoff"))
    assert _run(monkeypatch, _argv(book)) == ex.EXIT_MOC_SKIPPED
    assert _FakeBroker.instances == [], "a late strict-MOC run must not touch the account"


def test_catch_up_is_ignored_under_strict_moc(env, monkeypatch, capsys):
    _, book = env
    monkeypatch.setattr(ic, "moc_entry_open",
                        lambda now=None, cutoff=None: (False, "late"))
    rc = _run(monkeypatch, _argv(book, "--catch-up", "--outside-rth"))
    assert rc == ex.EXIT_MOC_SKIPPED and _FakeBroker.instances == []
    assert "ignoring --catch-up" in capsys.readouterr().out


def test_strict_moc_refuses_other_order_types(env, monkeypatch):
    _, book = env
    argv = [a if a != "moc" else "market" for a in _argv(book)]
    assert _run(monkeypatch, argv) == ex.EXIT_ABORT
    assert _FakeBroker.instances == []


def test_working_orders_abort_before_anything_is_sent(env, monkeypatch):
    _, book = env
    t = type("T", (), {"order": type("O", (), {"account": ACCT, "action": "BUY",
                                               "totalQuantity": 500.0})(),
                       "contract": type("C", (), {"symbol": "GDX"})(),
                       "isDone": lambda self: False})()
    _FakeBroker.working = [t]
    assert _run(monkeypatch, _argv(book)) == ex.EXIT_ABORT
    assert _FakeBroker.instances[0].placed == []


def test_a_second_run_for_the_same_bar_is_locked_out(env, monkeypatch):
    tmp, book = env
    assert _run(monkeypatch, _argv(book)) == 0
    # simulate the report not being there yet (a MOC run writes it after 4 PM)
    for p in (tmp / "accounts" / ACCT).glob("executed_*"):
        p.unlink() if p.is_file() else [q.unlink() for q in p.iterdir()]
    assert _run(monkeypatch, _argv(book)) == ex.EXIT_ABORT
    assert len(_FakeBroker.instances[1].placed) == 0


def test_force_rerun_overrides_the_lock(env, monkeypatch):
    tmp, book = env
    assert _run(monkeypatch, _argv(book)) == 0
    for p in (tmp / "accounts" / ACCT).glob("executed_*"):
        p.unlink() if p.is_file() else [q.unlink() for q in p.iterdir()]
    assert _run(monkeypatch, _argv(book, "--force-rerun")) == 0


def test_unfilled_moc_exits_4_and_reports_it(env, monkeypatch, capsys):
    _, book = env
    _FakeBroker.fill_ratio = 0.0
    assert _run(monkeypatch, _argv(book)) == ex.EXIT_MOC_UNFILLED
    assert "NOT FULLY FILLED" in capsys.readouterr().out


def test_account_and_expected_account_must_agree(env, monkeypatch):
    _, book = env
    assert _run(monkeypatch, _argv(book, "--expected-account", "U1111111")) == ex.EXIT_ABORT
    assert _FakeBroker.instances == []


def test_live_defaults_to_the_live_gateway_port(env, monkeypatch):
    _, book = env
    argv = [a for a in _argv(book)]
    i = argv.index("--port")
    del argv[i:i + 2]
    _run(monkeypatch, argv)
    assert _FakeBroker.instances[0].port == ic.DEFAULT_LIVE_PORT


def test_dry_run_places_nothing(env, monkeypatch):
    _, book = env
    argv = [a for a in _argv(book) if a != "--execute"]
    assert _run(monkeypatch, argv) == 0
    assert _FakeBroker.instances[0].placed == []


def test_a_live_run_refuses_the_paper_book(env, monkeypatch):
    tmp, book = env
    d = json.loads(book.read_text())
    d["book_mode"] = "paper"
    book.write_text(json.dumps(d))
    assert _run(monkeypatch, _argv(book)) == ex.EXIT_ABORT
    assert _FakeBroker.instances == []
    assert _run(monkeypatch, _argv(book, "--allow-paper-book")) == 0
