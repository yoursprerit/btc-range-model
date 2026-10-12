"""Production-hardening batch: calendar expiry, clock skew, currency, per-order
sanity limits, the post-trade verdict, order tags, atomic report writes, report
publishing that commits only the reports, gateway-connect retry, log rotation
and least-privilege scheduling."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import executed_book as eb  # noqa: E402
import ibkr_common as ic  # noqa: E402
import ibkr_execute_accounts as acc  # noqa: E402
import ibkr_execute_book as ex  # noqa: E402
from test_executor_multi_account import (  # noqa: E402
    _FakeBroker, _argv, _book, _run)
from test_executor_multi_account import env as _executor_env  # noqa: E402,F401


@pytest.fixture
def env(_executor_env):
    """The fake-broker executor environment from test_executor_multi_account."""
    return _executor_env


# ── market calendar expiry ──────────────────────────────────────────────────
def test_calendar_covers_up_to_its_last_year():
    assert ic.calendar_covers(f"{ic.CALENDAR_LAST_YEAR}-12-31")
    assert not ic.calendar_covers(f"{ic.CALENDAR_LAST_YEAR + 1}-01-04")


def test_calendar_tables_have_not_run_out_for_the_coming_months():
    """Time-bomb on purpose: when this fails, extend US_MARKET_HOLIDAYS and
    US_EARLY_CLOSES in ibkr_common.py (and the mirror in preflight_option_c.py)."""
    ahead = pd.Timestamp.now() + pd.Timedelta(days=120)
    assert ic.calendar_covers(ahead), (
        f"the holiday table ends in {ic.CALENDAR_LAST_YEAR}; extend it")


def test_calendar_warning_appears_only_near_the_end():
    last = ic.CALENDAR_LAST_YEAR
    assert ic.calendar_warning(f"{last}-03-01") is None
    assert "ends" in ic.calendar_warning(f"{last}-11-15")
    assert "refused" in ic.calendar_warning(f"{last + 1}-02-01")


def test_executor_refuses_to_trade_past_the_calendar(env, monkeypatch):
    _, book = env
    monkeypatch.setattr(ic, "CALENDAR_LAST_YEAR", 2020)
    assert _run(monkeypatch, _argv(book)) == ex.EXIT_ABORT
    assert _FakeBroker.instances == []
    # an explicit --force is the documented override
    assert _run(monkeypatch, _argv(book, "--force")) in (0, ex.EXIT_ABORT)


# ── clock skew ──────────────────────────────────────────────────────────────
def test_check_clock():
    assert ic.check_clock(0.4)[0]
    assert ic.check_clock(None)[0] and "not verified" in ic.check_clock(None)[1]
    ok, why = ic.check_clock(-95.0)
    assert not ok and "w32tm" in why
    assert ic.check_clock(40.0, limit=60)[0]


def test_a_skewed_clock_blocks_trading(env, monkeypatch):
    _, book = env
    monkeypatch.setattr(_FakeBroker, "clock_skew", lambda self: 120.0, raising=False)
    assert _run(monkeypatch, _argv(book)) == ex.EXIT_ABORT
    assert _FakeBroker.instances[-1].placed == []


def test_a_healthy_clock_trades(env, monkeypatch):
    _, book = env
    monkeypatch.setattr(_FakeBroker, "clock_skew", lambda self: 0.8, raising=False)
    assert _run(monkeypatch, _argv(book)) == 0


# ── currency ────────────────────────────────────────────────────────────────
class _SummaryIB:
    def __init__(self, rows, values=()):
        self.rows, self.values = rows, values

    def accountSummary(self, _a=None):
        return [type("V", (), dict(tag=t, value=v, currency=c))() for t, v, c in self.rows]

    def accountValues(self, _a=None):
        return [type("V", (), dict(tag=t, value=v, currency=c))() for t, v, c in self.values]


def _bk(ib):
    b = object.__new__(ic.Broker)
    b.ib, b.account = ib, "U1"
    return b


def test_net_liq_accepts_usd():
    assert _bk(_SummaryIB([("NetLiquidation", "12345.6", "USD")])).net_liq() == 12345.6


def test_net_liq_refuses_a_non_usd_account():
    with pytest.raises(RuntimeError, match="EUR"):
        _bk(_SummaryIB([("NetLiquidation", "9000", "EUR")])).net_liq()


def test_net_liq_falls_back_to_account_values_in_usd():
    ib = _SummaryIB([], values=[("NetLiquidation", "777", "USD")])
    assert _bk(ib).net_liq() == 777.0


# ── per-order sanity ────────────────────────────────────────────────────────
def _o(action, qty, price=100.0, key="XLE"):
    return ic.Order(key, key, action, qty, price, "t")


def test_a_sell_larger_than_the_holding_is_refused():
    ok, why = ic.check_order_sanity([_o("SELL", 50)], {"XLE": 40.0}, 100_000.0)
    assert not ok and "does not own" in why
    assert ic.check_order_sanity([_o("SELL", 40)], {"XLE": 40.0}, 100_000.0)[0]


def test_an_oversized_buy_is_refused():
    ok, why = ic.check_order_sanity([_o("BUY", 500, 100.0)], {}, 100_000.0, 0.40)
    assert not ok and "50%" in why
    assert ic.check_order_sanity([_o("BUY", 300, 100.0)], {}, 100_000.0, 0.40)[0]


def test_too_many_orders_is_refused():
    many = [_o("BUY", 1, 1.0, key=f"K{i}") for i in range(30)]
    ok, why = ic.check_order_sanity(many, {}, 100_000.0, max_orders=25)
    assert not ok and "30 orders" in why
    assert ic.check_order_sanity(many, {}, 100_000.0, max_orders=0)[0]


def test_executor_aborts_on_an_oversized_buy_but_not_a_normal_one(env, monkeypatch):
    _, book = env
    _book(book, {"GDX": 0.60}, 0.40)
    assert _run(monkeypatch, _argv(book, "--max-buy-frac", "0.40")) == ex.EXIT_ABORT
    assert _FakeBroker.instances[-1].placed == []
    _book(book, {"GDX": 0.30}, 0.70)
    assert _run(monkeypatch, _argv(book, "--max-buy-frac", "0.40")) == 0


# ── post-trade verdict ──────────────────────────────────────────────────────
def test_post_trade_drift_is_a_failure_not_a_footnote(env, monkeypatch, capsys):
    _, book = env
    # fills happen but the account does not show them: it did not land on target
    monkeypatch.setattr(_FakeBroker, "positions_by_key",
                        lambda self, settle=True: {}, raising=True)
    assert _run(monkeypatch, _argv(book)) == ex.EXIT_POST_TRADE
    assert "POST-TRADE CHECK FAILED" in capsys.readouterr().out


def test_unfilled_orders_keep_their_own_exit_code(env, monkeypatch):
    _, book = env
    _FakeBroker.fill_ratio = 0.0
    assert _run(monkeypatch, _argv(book)) == ex.EXIT_MOC_UNFILLED


def test_post_trade_check_returns_a_verdict():
    class B:
        def net_liq(self): return 100_000.0
        def positions_by_key(self, settle=True): return {"XLE": 100.0}
        def gross_position_value(self): return 0.0
        def read_value(self): return 0.0
    g = ic.Guards()
    ok, _ = ic.post_trade_check(B(), {"XLE": 0.10}, {"XLE": 100.0}, g)
    assert ok                                         # 10 000 / 100 000 = target
    ok, why = ic.post_trade_check(B(), {"XLE": 0.50}, {"XLE": 100.0}, g)
    assert not ok and "drifted" in why


def test_runner_treats_exit_5_as_a_failed_account():
    r = acc.Result(acc.Account("U1"), rc=ex.EXIT_POST_TRADE)
    assert acc.overall_exit([r], []) == acc.EXIT_FAILED
    assert "did NOT land" in acc.describe(5)


# ── order tag ───────────────────────────────────────────────────────────────
def test_orders_carry_a_recognisable_reference(env, monkeypatch):
    _, book = env
    assert _run(monkeypatch, _argv(book)) == 0
    assert _FakeBroker.instances[-1].order_ref == "btc-range:2026-10-09"


def test_order_ref_reaches_the_ib_order(monkeypatch):
    pytest.importorskip("ib_async")
    monkeypatch.setattr(ic.sym, "ibkr_contract", lambda s: type("C", (), {"symbol": s})())
    sent = []

    class _T:
        def __init__(self, o):
            self.order = o
            self.orderStatus = type("S", (), dict(status="Filled", filled=o.totalQuantity,
                                                  avgFillPrice=1.0))()
            self.log = []
            self.contract = type("C", (), {"symbol": "X"})()

        def isDone(self): return True

    class _IB:
        def qualifyContracts(self, *_): pass
        def sleep(self, _s): pass
        def placeOrder(self, c, o):
            sent.append(o)
            return _T(o)
    b = object.__new__(ic.Broker)
    b.ib, b.account, b.account_mode, b.cash = _IB(), "U1", "live", (lambda: 1e9)
    b.order_ref = "btc-range:2026-10-09"
    b.place([ic.Order("GDX", "GDX", "BUY", 5.0, 10.0, "t")], fractional=False, wait=0.0,
            order_type=ic.ORDER_MOC, now=pd.Timestamp("2026-10-12 15:30", tz="America/New_York"),
            moc_strict=True)
    assert [o.orderRef for o in sent] == ["btc-range:2026-10-09"]


# ── atomic report writes ────────────────────────────────────────────────────
def test_write_atomic_replaces_the_file_whole(tmp_path):
    p = tmp_path / "sub" / "r.json"
    eb.write_atomic(p, '{"a": 1}')
    assert json.loads(p.read_text()) == {"a": 1}
    eb.write_atomic(p, '{"a": 2}')
    assert json.loads(p.read_text()) == {"a": 2}
    assert [x.name for x in p.parent.iterdir()] == ["r.json"]      # no temp left


def test_a_failed_write_leaves_the_old_report_intact(tmp_path, monkeypatch):
    p = tmp_path / "r.json"
    eb.write_atomic(p, '{"ok": true}')

    def boom(*_a, **_k):
        raise OSError("disk full")
    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        eb.write_atomic(p, '{"half":')
    monkeypatch.undo()
    assert json.loads(p.read_text()) == {"ok": True}
    assert [x.name for x in tmp_path.iterdir()] == ["r.json"]


def test_reports_and_archives_use_the_atomic_writer():
    src = (ROOT / "app" / "executed_book.py").read_text() + \
        (ROOT / "scripts" / "ibkr_execute_book.py").read_text()
    assert "out.write_text(" not in src and "p.write_text(tb.dumps" not in src


# ── publishing reports: only the reports ────────────────────────────────────
def _git(cwd, *a):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a],
                          cwd=cwd, capture_output=True, text=True, check=True).stdout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    remote, work = tmp_path / "remote.git", tmp_path / "work"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(remote)], check=True)
    work.mkdir()
    _git(work, "init", "-q", "-b", "main")
    (work / "README").write_text("x")
    _git(work, "add", "README")
    _git(work, "commit", "-q", "-m", "init")
    _git(work, "remote", "add", "origin", str(remote))
    _git(work, "push", "-q", "-u", "origin", "main")
    monkeypatch.setattr(acc, "_REPO", work)
    monkeypatch.setattr(acc, "ACCOUNTS_DIR", work / "data" / "overall" / "accounts")
    d = acc.ACCOUNTS_DIR / "U1111111"
    (d / "executed_archive").mkdir(parents=True)
    (d / "executed_book_live.json").write_text("{}")
    (d / "executed_archive" / "2026-10-09_live.json").write_text("{}")
    return work, remote


def test_push_reports_commits_only_the_reports(repo):
    work, remote = repo
    (work / "notes.txt").write_text("unrelated work in progress")
    _git(work, "add", "notes.txt")                          # staged, NOT ours
    assert acc.push_reports([acc.Account("U1111111")], branch="main") is True
    pushed = _git(remote, "show", "--name-only", "--format=", "main").split()
    assert sorted(pushed) == ["data/overall/accounts/U1111111/executed_archive/2026-10-09_live.json",
                              "data/overall/accounts/U1111111/executed_book_live.json"]
    assert "notes.txt" in _git(work, "diff", "--cached", "--name-only")   # still staged


def test_push_reports_refuses_a_detached_head(repo):
    work, remote = repo
    _git(work, "checkout", "-q", "--detach")
    before = _git(remote, "rev-parse", "main")
    assert acc.push_reports([acc.Account("U1111111")]) is False
    assert _git(remote, "rev-parse", "main") == before


def test_push_reports_with_nothing_new_does_nothing(repo):
    work, remote = repo
    assert acc.push_reports([acc.Account("U1111111")], branch="main") is True
    head = _git(remote, "rev-parse", "main")
    assert acc.push_reports([acc.Account("U1111111")], branch="main") is True
    assert _git(remote, "rev-parse", "main") == head


# ── gateway connect retry ───────────────────────────────────────────────────
def _cfg():
    cfg, _ = acc.validate_config({"accounts": [{"id": "U1111111"}]})
    return cfg


class _Clock:
    """A clock that only moves when the (fake) sleep is called."""
    def __init__(self):
        self.t, self.naps = 0.0, []

    def now(self):
        return self.t

    def sleep(self, s):
        self.naps.append(s)
        self.t += s


def test_gateway_retries_until_it_answers(monkeypatch):
    calls, clk = [], _Clock()

    def once(_cfg):
        calls.append(1)
        return (["U1111111"], "ok") if len(calls) >= 3 else (None, "not logged in")
    monkeypatch.setattr(acc, "_connect_once", once)
    monkeypatch.setattr(ic, "moc_entry_open", lambda now=None, cutoff=None: (True, "x"))
    accts, _ = acc.verify_gateway(_cfg(), retry_s=600, every_s=30,
                                  sleep=clk.sleep, clock=clk.now)
    assert accts == ["U1111111"] and len(calls) == 3 and clk.naps == [30, 30]


def test_no_retry_for_a_preview_or_check(monkeypatch):
    calls = []
    monkeypatch.setattr(acc, "_connect_once", lambda c: (calls.append(1), (None, "down"))[1])
    accts, why = acc.verify_gateway(_cfg(), retry_s=0)
    assert accts is None and len(calls) == 1 and "down" in why


def test_retry_gives_up_after_its_budget(monkeypatch):
    clk = _Clock()
    monkeypatch.setattr(acc, "_connect_once", lambda c: (None, "down"))
    monkeypatch.setattr(ic, "moc_entry_open", lambda now=None, cutoff=None: (True, "x"))
    accts, why = acc.verify_gateway(_cfg(), retry_s=90, every_s=30,
                                    sleep=clk.sleep, clock=clk.now)
    assert accts is None and "gave up" in why
    assert clk.naps == [30, 30, 30]              # tries at 0, 30, 60, 90s -- then out


def test_retry_stops_when_the_moc_cutoff_is_too_close(monkeypatch):
    monkeypatch.setattr(acc, "_connect_once", lambda c: (None, "down"))
    monkeypatch.setattr(ic, "moc_entry_open", lambda now=None, cutoff=None: (False, "late"))
    clk = _Clock()
    accts, why = acc.verify_gateway(_cfg(), retry_s=600, every_s=30,
                                    sleep=clk.sleep, clock=clk.now)
    assert accts is None and clk.naps == [] and "gave up" in why


# ── log rotation, versions, least privilege ─────────────────────────────────
def test_rotate_log_rolls_big_logs_and_keeps_three(tmp_path):
    log = tmp_path / "U1.log"
    for gen in range(5):
        log.write_text("x" * 200)
        acc.rotate_log(log, max_bytes=100, keep=3)
        assert not log.exists()
        log.write_text(f"gen{gen}")
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["U1.log", "U1.log.1", "U1.log.2", "U1.log.3"][:len(names)]
    assert "U1.log.4" not in names


def test_small_logs_are_left_alone(tmp_path):
    log = tmp_path / "U1.log"
    log.write_text("short")
    acc.rotate_log(log, max_bytes=100)
    assert log.read_text() == "short" and len(list(tmp_path.iterdir())) == 1


def test_runtime_versions_line():
    line = ic.runtime_versions()
    assert line.startswith("python ") and "pandas" in line and "ib_async" in line


def test_scheduled_tasks_do_not_run_elevated():
    setup = (ROOT / "scripts" / "setup_windows_option_c.ps1").read_text()
    assert "RunLevel Highest" not in setup and setup.count("RunLevel Limited") == 2
