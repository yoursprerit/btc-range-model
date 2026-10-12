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
        self.pos = {}                    # what the account holds after "fills"
        self.disconnected = False

    def net_liq(self): return 100_000.0
    def cash(self): return 100_000.0
    def holdings(self, _nl=None): return {}
    def gross_position_value(self): return 0.0
    def read_value(self): return 0.0
    def quote_by_symbol(self, _s): return {}
    def portfolio_snapshot(self): return []
    def realized_pnl(self): return None
    def positions_by_key(self, settle=True): return dict(self.pos)
    def working_orders(self): return list(type(self).working)
    def disconnect(self): self.disconnected = True

    def place(self, orders, fractional, wait, **kw):
        self.placed.append((orders, kw))
        for o in orders:                 # a real account reflects its fills
            d = o.qty * type(self).fill_ratio * (1 if o.action == "BUY" else -1)
            self.pos[o.key] = self.pos.get(o.key, 0.0) + d
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
    # the test books put 30-60 % in one name; the per-order NAV limit has its own
    # tests, so open it here
    return ["--file", str(book), "--execute", "--account-mode", "live",
            "--confirm-live", "--account", ACCT, "--order-type", "moc",
            "--strict-moc", "--no-push-report", "--port", "4001",
            "--max-buy-frac", "1.0", *extra]


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


# ── the deploy cap (TOTAL across all assets) and the V2 SATA cap ────────────
def _book(path, weights, cash, sata_cap=None):
    d = json.loads(path.read_text())
    d.update(weights=weights, cash_weight=cash,
             exec_price={k: 100.0 for k in weights})
    if sata_cap is None:
        d.pop("sata_cap", None)
    else:
        d["sata_cap"] = sata_cap
    path.write_text(json.dumps(d))


def _planned(monkeypatch, book, *extra, deploy="0.25"):
    assert _run(monkeypatch, _argv(book, "--max-deploy-frac", deploy, *extra)) == 0
    (orders, _kw), = _FakeBroker.instances[-1].placed
    return {o.symbol: o.qty * o.price / 100_000.0 for o in orders}


def test_deploy_cap_is_the_total_across_every_asset_sata_included(env, monkeypatch):
    """25 % means 25 % of the account in TOTAL -- risk assets AND SATA -- and the
    other 75 % is left as cash. (SATA 30 % + risk 60 % + cash 10 % scales by
    25/90.)"""
    _, book = env
    _book(book, {"GDX": 0.60, "SATA": 0.30}, 0.10, sata_cap=0.30)
    got = _planned(monkeypatch, book)
    assert sum(got.values()) == pytest.approx(0.25, abs=0.005)
    assert got["GDX"] == pytest.approx(0.60 * 25 / 90, abs=0.005)
    assert got["SATA"] == pytest.approx(0.30 * 25 / 90, abs=0.005)


def test_a_pre_cap_book_is_capped_at_30_then_scaled_to_the_deploy_cap(env, monkeypatch):
    """The first live preview's book (SATA 40 %, no sata_cap): SATA is trimmed to
    30 % first, then everything is scaled to the 25 % total."""
    _, book = env
    _book(book, {"GDX": 0.60, "SATA": 0.40}, 0.0, sata_cap=None)
    got = _planned(monkeypatch, book)
    assert sum(got.values()) == pytest.approx(0.25, abs=0.005)
    assert got["SATA"] == pytest.approx(0.30 * 25 / 90, abs=0.005)


def test_without_the_sata_fallback_it_is_still_capped_in_total(env, monkeypatch):
    _, book = env
    _book(book, {"GDX": 0.60, "SATA": 0.40}, 0.0, sata_cap=None)
    got = _planned(monkeypatch, book, "--default-sata-cap", "-1")
    assert sum(got.values()) == pytest.approx(0.25, abs=0.005)
    assert got["SATA"] == pytest.approx(0.40 * 0.25, abs=0.005)


def test_nothing_is_added_when_the_book_deploys_less_than_the_cap(env, monkeypatch):
    """A ceiling, not a target: no scaling UP, and no SATA park for the rest."""
    _, book = env
    _book(book, {"GDX": 0.10, "SATA": 0.05}, 0.85, sata_cap=0.30)
    got = _planned(monkeypatch, book)
    assert got["GDX"] == pytest.approx(0.10, abs=0.005)
    assert got["SATA"] == pytest.approx(0.05, abs=0.005)


def test_a_paper_style_book_without_sata_is_scaled_to_the_cap(env, monkeypatch):
    _, book = env
    _book(book, {"GDX": 0.60}, 0.40)
    got = _planned(monkeypatch, book)
    assert got == {"GDX": pytest.approx(0.25, abs=0.005)}


def test_no_deploy_cap_leaves_the_book_alone(env, monkeypatch):
    _, book = env
    _book(book, {"GDX": 0.60, "SATA": 0.30}, 0.10, sata_cap=0.30)
    got = _planned(monkeypatch, book, deploy="1.0")
    assert got["GDX"] == pytest.approx(0.60, abs=0.005)
    assert got["SATA"] == pytest.approx(0.30, abs=0.005)


def test_log_says_what_the_cap_did(env, monkeypatch, capsys):
    _, book = env
    _book(book, {"GDX": 0.60, "SATA": 0.30}, 0.10, sata_cap=0.30)
    _planned(monkeypatch, book)
    out = capsys.readouterr().out
    assert "Deploy cap 25% of NAV" in out and "SATA included" in out
    assert "stays cash" in out and "freed" not in out


def test_pre_cap_book_without_a_deploy_cap_is_trimmed_to_30(env, monkeypatch):
    _, book = env
    _book(book, {"GDX": 0.60, "SATA": 0.40}, 0.0, sata_cap=None)
    got = _planned(monkeypatch, book, deploy="1.0")
    assert got["GDX"] == pytest.approx(0.60, abs=0.005)
    assert got["SATA"] == pytest.approx(0.30, abs=0.005)


def test_a_books_own_cap_beats_the_default(env, monkeypatch):
    _, book = env
    _book(book, {"GDX": 0.50, "SATA": 0.40}, 0.10, sata_cap=0.50)
    got = _planned(monkeypatch, book, deploy="1.0")
    assert got["SATA"] == pytest.approx(0.40, abs=0.005)      # not trimmed to 30 %


def test_apply_deploy_cap_unit():
    w = {"GRID": 0.30, "XLE": 0.20, "SATA": 0.30}
    out, info = ic.apply_deploy_cap(w, 0.25)
    assert sum(out.values()) == pytest.approx(0.25)
    assert out["GRID"] / out["XLE"] == pytest.approx(0.30 / 0.20)     # proportional
    assert info["scale"] == pytest.approx(0.25 / 0.80) and info["before"] == pytest.approx(0.80)
    assert w == {"GRID": 0.30, "XLE": 0.20, "SATA": 0.30}            # input untouched
    assert ic.apply_deploy_cap(w, 1.0)[1] is None                    # no cap
    same, info = ic.apply_deploy_cap({"GRID": 0.10}, 0.25)           # under the cap
    assert same == {"GRID": 0.10} and info is None
    assert ic.apply_deploy_cap({}, 0.25) == ({}, None)


def test_effective_sata_cap_rules():
    assert ic.effective_sata_cap(0.25, 0.30) == (0.25, "book")
    cap, src = ic.effective_sata_cap(None, 0.30)
    assert cap == 0.30 and "default" in src
    assert ic.effective_sata_cap(None, -1) == (None, "none")
    assert ic.effective_sata_cap(None, None) == (None, "none")
    cap, src = ic.effective_sata_cap("garbage", 0.30)       # unreadable: cautious
    assert cap == 0.0 and "unreadable" in src
    assert ic.effective_sata_cap(float("nan"), 0.30)[0] == 0.0


def test_trim_sata_only_touches_an_excess():
    w, ex = ic.trim_sata({"GDX": 0.5, "SATA": 0.4}, 0.30)
    assert w["SATA"] == pytest.approx(0.30) and ex == pytest.approx(0.10) and w["GDX"] == 0.5
    w, ex = ic.trim_sata({"GDX": 0.5, "SATA": 0.2}, 0.30)
    assert w["SATA"] == 0.2 and ex == 0.0
    assert ic.trim_sata({"GDX": 1.0}, 0.30)[1] == 0.0


def test_default_cap_matches_the_engines_cap():
    """ibkr_common cannot import the model stack, so the number is copied --
    this keeps the copy honest."""
    oc = pytest.importorskip("overall_core")
    assert ic.DEFAULT_SATA_CAP == oc.SATA_CAP
