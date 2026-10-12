"""Is this the book THIS account should trade — and is a refused MOC reported in time?

Three promises, each pinned here:

* a published book for the wrong profile never trades (pinned), and a profile
  switch or a jump in NEW target weight holds the BUYS for a day while sells
  always go (a stop-out to cash is the safe direction);
* an unlevered ETF quoting 10 %+ from its book price has its buy held without
  aborting the run — and the volatile names are exempt, so a normal leveraged
  crash day cannot hold anything up;
* a MOC the broker refuses raises an alert immediately, not after the auction.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ibkr_common as ic  # noqa: E402
import ibkr_execute_accounts as acc  # noqa: E402
import ibkr_execute_book as ex  # noqa: E402
import ibkr_safety as ibs  # noqa: E402
import ibkr_symbols as sym  # noqa: E402
import report_vault as rv  # noqa: E402
from test_executor_multi_account import (  # noqa: E402,F401  (env is a fixture)
    ACCT, _FakeBroker, _argv, _run, env)
from test_executor_safety import _IB, _broker, _orders, _statuses, _tk  # noqa: E402,F401
from test_executor_safety import _no_ib_async  # noqa: E402,F401  (autouse stubs)


# ── book_change / book_hold ──────────────────────────────────────────────────
def _state(as_of="2026-10-08", profile="Balanced", **w):
    return ibs.next_book_state(None, as_of, profile, w)


def test_no_previous_book_means_nothing_to_compare():
    assert ibs.book_change(None, "2026-10-09", "Balanced", {"GDX": 0.5}) is None
    hold, notes = ibs.book_hold(None, 0.6, False, False)
    assert hold is None and "nothing to compare" in notes[0]


def test_adds_and_cuts_are_measured_against_the_last_distinct_book():
    ch = ibs.book_change(_state(GDX=0.5, SOXX=0.1), "2026-10-09", "Balanced",
                         {"GDX": 0.2, "XLE": 0.4})
    assert ch["adds"] == pytest.approx(0.4) and ch["cuts"] == pytest.approx(0.4)
    assert ch["base_as_of"] == "2026-10-08" and not ch["profile_changed"]


def test_a_rerun_of_the_same_day_is_judged_against_the_book_before_it():
    s1 = _state(GDX=0.1)
    seen = ibs.next_book_state(s1, "2026-10-09", "Balanced", {"GDX": 0.9})
    again = ibs.book_change(seen, "2026-10-09", "Balanced", {"GDX": 0.9})
    assert again["adds"] == pytest.approx(0.8) and again["base_as_of"] == "2026-10-08"
    # …and re-seeing the same day keeps that comparison, it does not erase it
    seen2 = ibs.next_book_state(seen, "2026-10-09", "Balanced", {"GDX": 0.9})
    assert seen2["prev_weights"] == {"GDX": 0.1}


def test_a_big_add_holds_the_buys():
    ch = ibs.book_change(_state(GDX=0.1), "2026-10-09", "Balanced", {"GDX": 0.9})
    hold, _ = ibs.book_hold(ch, 0.6, False, False)
    assert hold and "80%" in hold


def test_an_ordinary_change_is_left_alone():
    ch = ibs.book_change(_state(GDX=0.3), "2026-10-09", "Balanced", {"GDX": 0.45})
    assert ibs.book_hold(ch, 0.6, False, False)[0] is None


def test_a_large_cut_is_never_held_only_noted():
    ch = ibs.book_change(_state(GDX=0.9), "2026-10-09", "Balanced", {})
    hold, notes = ibs.book_hold(ch, 0.6, False, False)
    assert hold is None and any("de-risk" in n for n in notes)


def test_a_profile_switch_holds_buys_unless_the_profile_is_pinned():
    ch = ibs.book_change(_state(GDX=0.3), "2026-10-09", "Growth", {"GDX": 0.3})
    assert ch["profile_changed"]
    hold, _ = ibs.book_hold(ch, 0.6, False, profile_pinned=False)
    assert hold and "Balanced → Growth" in hold
    assert ibs.book_hold(ch, 0.6, False, profile_pinned=True)[0] is None


def test_the_override_and_the_off_switch():
    ch = ibs.book_change(_state(GDX=0.1), "2026-10-09", "Balanced", {"GDX": 0.9})
    hold, notes = ibs.book_hold(ch, 0.6, True, False)
    assert hold is None and any("ACCEPTED" in n for n in notes)
    assert ibs.book_hold(ch, 0, False, False) == (None, ["book check disabled"])


def test_state_round_trips_atomically_and_garbage_is_no_state(tmp_path):
    p = ibs.book_state_path(tmp_path / "executed_book_live.json")
    assert p.name.startswith(".") and p.parent == tmp_path
    assert ibs.write_book_state(p, _state(GDX=0.5))
    assert ibs.read_book_state(p)["weights"] == {"GDX": 0.5}
    assert not list(tmp_path.glob("*.tmp"))
    p.write_text("{nope")
    assert ibs.read_book_state(p) is None
    assert ibs.read_book_state(tmp_path / "missing") is None
    assert ibs.write_book_state(tmp_path / "no" / "such" / "dir" / ".s.json", {}) is True


def test_non_finite_weights_are_ignored_not_trusted():
    st = ibs.next_book_state(None, "d", "p", {"A": float("nan"), "B": "x", "C": 0.2})
    assert st["weights"] == {"C": 0.2}


def test_the_vault_never_seals_the_state_file(tmp_path):
    (tmp_path / "accounts" / "U1").mkdir(parents=True)
    (tmp_path / "accounts" / "U1" / "executed_book_live.json").write_text("{}")
    ibs.write_book_state(tmp_path / "accounts" / "U1" / ".executed_book_live.bookstate.json", {"a": 1})
    assert rv.seal_tree(tmp_path, tmp_path / "vault", "pw") == 1


# ── soft price drift ─────────────────────────────────────────────────────────
class _Quotes:
    def __init__(self, q): self.q = q
    def quote_by_symbol(self, s): return self.q.get(s, {})


def _buy(symbol, price=100.0):
    return ic.Order(symbol, symbol, "BUY", 10.0, price, "")


def test_an_unlevered_name_far_from_its_book_price_is_flagged():
    flags = ic.soft_drift_flags(_Quotes({"XLE": {"last": 88.0}}), [_buy("XLE")], 0.10)
    assert "XLE" in flags and "12%" in flags["XLE"]


def test_a_move_inside_the_limit_is_fine():
    assert ic.soft_drift_flags(_Quotes({"XLE": {"last": 95.0}}), [_buy("XLE")], 0.10) == {}


def test_volatile_names_are_exempt_even_at_20_percent():
    q = _Quotes({s: {"last": 80.0} for s in ("SOXL", "MSTU", "IBIT", "NUGT")})
    assert ic.soft_drift_flags(q, [_buy(s) for s in ("SOXL", "MSTU", "IBIT", "NUGT")], 0.10) == {}


def test_sells_unquotable_and_disabled_are_never_flagged():
    sell = ic.Order("XLE", "XLE", "SELL", 5.0, 100.0, "")
    assert ic.soft_drift_flags(_Quotes({"XLE": {"last": 50.0}}), [sell], 0.10) == {}
    assert ic.soft_drift_flags(_Quotes({}), [_buy("XLE")], 0.10) == {}
    assert ic.soft_drift_flags(_Quotes({"XLE": {"last": 50.0}}), [_buy("XLE")], 0) == {}


def test_every_symbol_in_the_volatile_set_is_a_real_tradeable_name():
    assert sym.VOLATILE_SYMBOLS <= set(sym.TRADE_SYMBOL.values())
    assert sym.is_volatile("soxl") and not sym.is_volatile("GLDM")


def test_the_buy_is_held_the_sell_goes_and_the_run_is_not_aborted():
    ib = _IB({"XLE": _tk(bid=87.9, ask=88.1, last=88.0, close=88.0),
              "GLDM": _tk(), "SPY": _tk()})
    b = _broker(ib)
    res = b.place(_orders(("GLDM", "SELL", 3, 100.0), ("XLE", "BUY", 5, 100.0)),
                  False, 0.0, order_type=ic.ORDER_MARKET)
    st = _statuses(res)
    assert st[("SELL", "GLDM")] == "Filled" and st[("BUY", "XLE")] == "SKIPPED-STRESS"
    assert ib.sent == [("SELL", "GLDM", 3.0)]
    assert any("XLE" in a for a in b.alerts)


def test_a_leveraged_name_in_the_same_spot_still_buys():
    ib = _IB({"SOXL": _tk(bid=87.9, ask=88.1, last=88.0, close=88.0), "SPY": _tk()})
    b = _broker(ib)
    b.place(_orders(("SOXL", "BUY", 5, 100.0)), False, 0.0, order_type=ic.ORDER_MARKET)
    assert ib.sent == [("BUY", "SOXL", 5.0)]


def test_the_hard_abort_is_unchanged():
    ok, why = ic.price_drift_check(_Quotes({"XLE": {"last": 60.0}}), [_buy("XLE")], 0.25)
    assert not ok and "XLE" in why


def test_no_safety_limits_means_no_soft_hold():
    ib = _IB({"XLE": _tk(bid=87.9, ask=88.1, last=88.0, close=88.0)})
    b = _broker(ib, safety=False)
    b.place(_orders(("XLE", "BUY", 5, 100.0)), False, 0.0, order_type=ic.ORDER_MARKET)
    assert ib.sent == [("BUY", "XLE", 5.0)]


# ── MOC rejections are alerted immediately ───────────────────────────────────
class _MocTrade:
    def __init__(self, symbol, status):
        self.order = object()
        self.contract = SimpleNamespace(symbol=symbol)
        self.log = [SimpleNamespace(message="not allowed in an IRA")]
        self.orderStatus = SimpleNamespace(status=status, filled=0.0, avgFillPrice=0.0)
    def isDone(self): return self.orderStatus.status == "Filled"


def _moc_broker(statuses):
    ib = _IB({"X": _tk(), "Y": _tk(), "SPY": _tk()})
    ib.placeOrder = lambda c, o: (ib.sent.append((o.action, c.symbol, 1.0))
                                  or _MocTrade(c.symbol, statuses[c.symbol]))
    return _broker(ib), ib


def test_a_refused_moc_alerts_at_once_naming_the_symbol_and_the_cutoff(monkeypatch):
    monkeypatch.setattr(ic, "moc_entry_open", lambda now=None, cutoff=None: (True, "ok"))
    monkeypatch.setattr(ic, "moc_wait_seconds", lambda now=None, **k: 3)
    b, ib = _moc_broker({"X": "Inactive", "Y": "Submitted"})
    b.place(_orders(("X", "BUY", 1, 100.0), ("Y", "BUY", 1, 100.0)), False, 0.0,
            order_type=ic.ORDER_MOC)
    refused = [a for a in b.alerts if "MOC" in a]
    assert len(refused) == 1 and "X" in refused[0] and "15:50 ET" in refused[0]
    assert "WRONG" in refused[0] and "Y was" not in refused[0]


def test_a_late_refusal_is_caught_by_the_recheck_and_each_symbol_alerts_once(monkeypatch):
    monkeypatch.setattr(ic, "moc_entry_open", lambda now=None, cutoff=None: (True, "ok"))
    monkeypatch.setattr(ic, "moc_wait_seconds", lambda now=None, **k: 200)
    b, ib = _moc_broker({"X": "Submitted", "Y": "Submitted"})
    trades, calls = [], {"n": 0}
    place = ib.placeOrder
    ib.placeOrder = lambda c, o: (trades.append(place(c, o)) or trades[-1])

    def sleep(_s):
        calls["n"] += 1
        if calls["n"] == 75:                       # the broker refuses X ~75 s in
            trades[0].orderStatus.status = "Cancelled"
    ib.sleep = sleep
    b.place(_orders(("X", "BUY", 1, 100.0), ("Y", "BUY", 1, 100.0)), False, 0.0,
            order_type=ic.ORDER_MOC)
    assert calls["n"] > 150, "the wait ran past several one-minute rechecks"
    assert len([a for a in b.alerts if "MOC X" in a]) == 1, \
        "caught by a recheck, and alerted once however many rechecks follow"


def test_no_refusal_no_moc_alert(monkeypatch):
    monkeypatch.setattr(ic, "moc_entry_open", lambda now=None, cutoff=None: (True, "ok"))
    monkeypatch.setattr(ic, "moc_wait_seconds", lambda now=None, **k: 3)
    b, _ = _moc_broker({"X": "Submitted", "Y": "Submitted"})
    b.place(_orders(("X", "BUY", 1, 100.0)), False, 0.0, order_type=ic.ORDER_MOC)
    assert not [a for a in b.alerts if "MOC" in a]


# ── accounts config ──────────────────────────────────────────────────────────
def _cfg(**acct):
    return {"accounts": [{"id": "U1111111", **acct}]}


def test_expected_profile_is_accepted_and_passed_to_the_child():
    cfg, errs = acc.validate_config(_cfg(expected_profile=" Balanced "))
    assert not errs and cfg.accounts[0].expected_profile == "Balanced"
    argv = acc.child_argv(cfg, cfg.accounts[0], 0, execute=True)
    assert argv[argv.index("--expected-profile") + 1] == "Balanced"


def test_unpinned_accounts_pass_no_flag_and_bad_types_are_rejected():
    cfg, errs = acc.validate_config(_cfg())
    assert not errs and "--expected-profile" not in acc.child_argv(
        cfg, cfg.accounts[0], 0, execute=True)
    assert acc.validate_config(_cfg(expected_profile=5))[1]


# ── main() end to end ────────────────────────────────────────────────────────
@pytest.fixture
def alerts(monkeypatch):
    sent = []
    monkeypatch.setattr(ibs, "send_alert", sent.append)
    return sent


def _book(path, as_of, weights, profile="Balanced"):
    d = json.loads(Path(path).read_text())
    d.update(as_of=as_of, profile=profile, weights=weights,
             cash_weight=round(1 - sum(weights.values()), 6),
             exec_price={k: 100.0 for k in weights} or {"GDX": 100.0})
    Path(path).write_text(json.dumps(d))


def _state_file(tmp):
    return tmp / "accounts" / ACCT / ".executed_book_live.bookstate.json"


def test_a_pinned_profile_that_does_not_match_never_connects(env, monkeypatch, alerts):
    _, book = env
    assert _run(monkeypatch, _argv(book, "--expected-profile", "Growth")) == ex.EXIT_ABORT
    assert _FakeBroker.instances == []
    assert any("pinned to 'Growth'" in a for a in alerts)


def test_a_pinned_profile_that_matches_trades_normally(env, monkeypatch, alerts):
    _, book = env
    assert _run(monkeypatch, _argv(book, "--expected-profile", "Balanced")) == 0
    assert alerts == []


def test_the_first_run_records_the_book_and_holds_nothing(env, monkeypatch, alerts):
    tmp, book = env
    assert _run(monkeypatch, _argv(book)) == 0
    assert ibs.read_book_state(_state_file(tmp))["weights"] == {"GDX": 0.5}
    assert not getattr(_FakeBroker.instances[0], "buy_block", None)
    assert alerts == []


def test_a_dry_run_records_nothing(env, monkeypatch, alerts):
    tmp, book = env
    argv = [a for a in _argv(book) if a != "--execute"]
    _run(monkeypatch, argv)
    assert not _state_file(tmp).exists()


def test_a_big_jump_holds_the_buys_alerts_and_the_next_day_trades(env, monkeypatch, alerts):
    tmp, book = env
    _book(book, "2026-10-08", {"GDX": 0.1})
    assert _run(monkeypatch, _argv(book)) == 0
    _book(book, "2026-10-09", {"GDX": 0.9})
    assert _run(monkeypatch, _argv(book)) == 0
    held = _FakeBroker.instances[-1]
    assert "book check" in held.buy_block and "80%" in held.buy_block
    assert any("Buys are held" in a for a in alerts)
    # the book was recorded, so the same weights the next day are an ordinary day
    _book(book, "2026-10-12", {"GDX": 0.9})
    assert _run(monkeypatch, _argv(book)) == 0
    assert not getattr(_FakeBroker.instances[-1], "buy_block", None)


def test_the_override_trades_the_big_jump(env, monkeypatch, alerts):
    _, book = env
    _book(book, "2026-10-08", {"GDX": 0.1})
    _run(monkeypatch, _argv(book))
    _book(book, "2026-10-09", {"GDX": 0.9})
    assert _run(monkeypatch, _argv(book, "--accept-large-change")) == 0
    assert not getattr(_FakeBroker.instances[-1], "buy_block", None)
    assert not any("Buys are held" in a for a in alerts)


def test_a_profile_switch_holds_the_buys(env, monkeypatch, alerts):
    _, book = env
    _book(book, "2026-10-08", {"GDX": 0.3}, profile="Balanced")
    _run(monkeypatch, _argv(book))
    _book(book, "2026-10-09", {"GDX": 0.3}, profile="Growth")
    _run(monkeypatch, _argv(book))
    assert "Balanced → Growth" in _FakeBroker.instances[-1].buy_block


def test_a_big_de_risk_goes_through_and_is_reported(env, monkeypatch, alerts):
    _, book = env
    _book(book, "2026-10-08", {"GDX": 0.9})
    _run(monkeypatch, _argv(book))
    _book(book, "2026-10-09", {"GDX": 0.05})
    assert _run(monkeypatch, _argv(book)) == 0
    assert not getattr(_FakeBroker.instances[-1], "buy_block", None)
    assert any("large de-risk" in a for a in alerts)


def test_the_book_check_can_be_switched_off(env, monkeypatch, alerts):
    _, book = env
    _book(book, "2026-10-08", {"GDX": 0.1})
    _run(monkeypatch, _argv(book))
    _book(book, "2026-10-09", {"GDX": 0.9})
    _run(monkeypatch, _argv(book, "--max-book-add", "0"))
    assert not getattr(_FakeBroker.instances[-1], "buy_block", None)


def test_a_refresh_report_ignores_the_profile_pin(env, monkeypatch, alerts):
    _, book = env
    argv = [a for a in _argv(book) if a != "--execute"] + [
        "--refresh-report", "--expected-profile", "Growth"]

    class B(_FakeBroker):
        def day_fills_by_symbol(self): return []
    monkeypatch.setattr(ex, "Broker", B)
    _run(monkeypatch, argv)
    assert B.instances, "a refresh reads the account; it places no orders, so the pin is moot"
