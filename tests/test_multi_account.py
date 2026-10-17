"""Multi-account, strict-MOC live trading (Option C, one IB Gateway login).

Everything here is broker-free: ``ib_async`` is replaced by a tiny fake, so the
rules that decide WHICH real account gets orders — and that those orders are
market-on-close or nothing — are pinned without a gateway.
"""
import sys
import textwrap
import time
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "app"))

import ibkr_common as ic  # noqa: E402
import ibkr_execute_accounts as acc  # noqa: E402

ET = "America/New_York"


def _et(y, m, d, h, mi):
    return pd.Timestamp(year=y, month=m, day=d, hour=h, minute=mi, tz=ET)


# ── choosing the account ────────────────────────────────────────────────────
def test_explicit_account_must_be_managed():
    got, why = ic.select_account(["U1", "U2"], "U3", "live")
    assert got is None and "not managed" in why and "U1, U2" in why


def test_explicit_account_is_selected_exactly():
    assert ic.select_account(["U1", "U2"], "U2", "live")[0] == "U2"


def test_single_account_needs_no_flag():
    assert ic.select_account(["U1"], None, "live")[0] == "U1"


def test_live_refuses_to_guess_between_accounts():
    got, why = ic.select_account(["U1", "U2"], None, "live")
    assert got is None and "pass --account" in why


def test_paper_keeps_the_legacy_first_account():
    assert ic.select_account(["DU1", "DU2"], None, "paper")[0] == "DU1"


def test_no_accounts_means_not_logged_in():
    got, why = ic.select_account([], None, "live")
    assert got is None and "no managed accounts" in why


# ── Broker construction against a fake gateway ──────────────────────────────
class _FakeIB:
    managed: list[str] = []
    connect_kwargs: dict = {}

    def __init__(self):
        self.disconnected = False

    def connect(self, host, port, clientId, timeout, account=""):
        type(self).connect_kwargs = dict(host=host, port=port, clientId=clientId,
                                         account=account)

    def managedAccounts(self):
        return list(type(self).managed)

    def disconnect(self):
        self.disconnected = True


@pytest.fixture
def fake_ib(monkeypatch):
    mod = type(sys)("ib_async")
    mod.IB = _FakeIB
    monkeypatch.setitem(sys.modules, "ib_async", mod)
    _FakeIB.managed = ["U1111111", "U2222222"]
    return _FakeIB


def test_broker_pins_the_requested_account_and_tells_the_gateway(fake_ib):
    b = ic.Broker("h", 4001, 41, account_mode="live", confirm_live=True,
                  account="U2222222", expected_account="U2222222")
    assert b.account == "U2222222"
    assert fake_ib.connect_kwargs["account"] == "U2222222"


def test_broker_live_multi_login_without_account_aborts(fake_ib):
    with pytest.raises(RuntimeError, match="pass --account"):
        ic.Broker("h", 4001, 41, account_mode="live", confirm_live=True)


def test_broker_wrong_account_aborts_and_disconnects(fake_ib):
    with pytest.raises(RuntimeError, match="not managed"):
        ic.Broker("h", 4001, 41, account_mode="live", confirm_live=True,
                  account="U9999999")


def test_broker_expected_account_mismatch_aborts(fake_ib):
    with pytest.raises(RuntimeError, match="does not match"):
        ic.Broker("h", 4001, 41, account_mode="live", confirm_live=True,
                  account="U1111111", expected_account="U2222222")


def test_broker_live_refuses_a_paper_account(fake_ib):
    fake_ib.managed = ["DU123"]
    with pytest.raises(RuntimeError, match="PAPER"):
        ic.Broker("h", 4001, 41, account_mode="live", confirm_live=True,
                  account="DU123")


def test_broker_live_needs_confirm(fake_ib):
    with pytest.raises(RuntimeError, match="--confirm-live"):
        ic.Broker("h", 4001, 41, account_mode="live", account="U1111111")


# ── orders carry the account; fills are per-account ─────────────────────────
class _Trade:
    def __init__(self, order, symbol):
        self.order = order
        self.contract = type("C", (), {"symbol": symbol})()
        self.orderStatus = type("S", (), {"status": "Filled", "filled": order.totalQuantity,
                                          "avgFillPrice": 100.0})()
        self.log = []

    def isDone(self):
        return True


class _OrderIB:
    def __init__(self):
        self.placed = []

    def qualifyContracts(self, *_):
        return None

    def placeOrder(self, contract, order):
        self.placed.append(order)
        return _Trade(order, contract.symbol)

    def sleep(self, _s):
        return None


def _broker(account="U2222222"):
    b = object.__new__(ic.Broker)
    b.ib = _OrderIB()
    b.account = account
    b.account_mode = "live"
    b.cash = lambda: 1e9
    return b


@pytest.fixture
def stub_contract(monkeypatch):
    monkeypatch.setattr(ic.sym, "ibkr_contract",
                        lambda symbol: type("C", (), {"symbol": symbol})())


def _orders(*rows):
    return [ic.Order(k, k, a, q, p, "t") for k, a, q, p in rows]


def test_every_moc_order_is_stamped_with_the_account(stub_contract):
    pytest.importorskip("ib_async")
    b = _broker()
    b.place(_orders(("GDX", "BUY", 10.0, 90.0), ("SOXX", "SELL", 5.0, 500.0)),
            fractional=False, wait=0.0, order_type=ic.ORDER_MOC,
            now=_et(2026, 10, 12, 15, 31), moc_strict=True,
            slippage_cap=0.005)
    assert b.ib.placed, "orders should have been sent"
    assert {o.account for o in b.ib.placed} == {"U2222222"}
    assert {o.orderType for o in b.ib.placed} == {"MOC"}


def test_fills_are_filtered_to_the_account(stub_contract):
    def fill(acct, sym_, shares, side="BOT"):
        return type("F", (), {
            "contract": type("C", (), {"symbol": sym_})(),
            "execution": type("E", (), {"acctNumber": acct, "shares": shares,
                                        "price": 100.0, "side": side})()})()
    b = _broker("U2222222")
    b.ib.fills = lambda: [fill("U1111111", "GDX", 50), fill("U2222222", "GDX", 7),
                          fill("U2222222", "SOXX", 3, "SLD")]
    got = {(t["symbol"], t["action"]): t["filled"] for t in b.day_fills_by_symbol()}
    assert got == {("GDX", "BUY"): 7.0, ("SOXX", "SELL"): 3.0}


# ── strict MOC: no fallback ─────────────────────────────────────────────────
def test_strict_moc_late_run_sends_nothing(stub_contract):
    b = _broker()
    with pytest.raises(ic.MocUnavailable, match="cutoff"):
        b.place(_orders(("GDX", "BUY", 10.0, 90.0)), fractional=False, wait=0.0,
                order_type=ic.ORDER_MOC, now=_et(2026, 10, 12, 15, 55),
                moc_strict=True)
    assert b.ib.placed == []


def test_non_strict_moc_still_falls_back_for_the_paper_path():
    b = _broker()
    t, _ = b._resolve_order_type(ic.ORDER_MOC, False, _et(2026, 10, 12, 15, 55))
    assert t == ic.ORDER_MARKETABLE_LIMIT


def test_strict_requires_moc_and_no_outside_rth():
    b = _broker()
    with pytest.raises(ValueError):
        b._resolve_order_type(ic.ORDER_MARKET, False, _et(2026, 10, 12, 15, 0),
                              moc_strict=True)
    with pytest.raises(ic.MocUnavailable):
        b._resolve_order_type(ic.ORDER_MOC, False, _et(2026, 10, 12, 15, 0),
                              outside_rth=True, moc_strict=True)


def test_strict_moc_forces_whole_shares():
    b = _broker()
    t, frac = b._resolve_order_type(ic.ORDER_MOC, True, _et(2026, 10, 12, 15, 0),
                                    moc_strict=True)
    assert t == ic.ORDER_MOC and frac is False


# ── early-close days ────────────────────────────────────────────────────────
def test_early_close_moves_the_moc_cutoff_to_1250():
    assert ic.moc_cutoff_et(_et(2026, 11, 27, 9, 0)) == (12, 50)
    assert ic.moc_cutoff_et(_et(2026, 11, 25, 9, 0)) == ic.MOC_CUTOFF_ET
    ok, _ = ic.moc_entry_open(_et(2026, 11, 27, 12, 40))
    assert ok
    ok, why = ic.moc_entry_open(_et(2026, 11, 27, 15, 30))   # the usual slot
    assert not ok and "12:50" in why


def test_early_close_auction_wait_ends_at_one_oclock():
    secs = ic.moc_wait_seconds(_et(2026, 11, 27, 12, 30), buffer_s=0)
    assert secs == pytest.approx(30 * 60)


# ── duplicate protection in the window the report can't cover ───────────────
class _WT:
    def __init__(self, acct, done=False, symbol="GDX"):
        self.order = type("O", (), {"account": acct, "action": "BUY",
                                    "totalQuantity": 5.0})()
        self.contract = type("C", (), {"symbol": symbol})()
        self._done = done

    def isDone(self):
        return self._done


def test_working_orders_block_a_second_send():
    ok, why = ic.check_no_working_orders([_WT("U2222222")], "U2222222")
    assert not ok and "working order" in why and "GDX" in why


def test_other_accounts_and_finished_orders_do_not_block():
    ok, _ = ic.check_no_working_orders(
        [_WT("U1111111"), _WT("U2222222", done=True)], "U2222222")
    assert ok


def test_unstamped_working_order_blocks_conservatively():
    assert not ic.check_no_working_orders([_WT("")], "U2222222")[0]


def test_send_lock_is_taken_once_per_bar_and_account(tmp_path):
    report = tmp_path / "executed_book_live.json"
    ok, _ = ic.acquire_send_lock(report, "2026-10-09", "U1")
    assert ok
    ok2, why = ic.acquire_send_lock(report, "2026-10-09", "U1")
    assert not ok2 and "already transmitted" in why
    assert ic.acquire_send_lock(report, "2026-10-12", "U1")[0]      # next bar
    assert ic.acquire_send_lock(report, "2026-10-09", "U1", force=True)[0]


def test_send_lock_is_per_report_directory(tmp_path):
    a, b = tmp_path / "U1", tmp_path / "U2"
    assert ic.acquire_send_lock(a / "r.json", "2026-10-09", "U1")[0]
    assert ic.acquire_send_lock(b / "r.json", "2026-10-09", "U2")[0]


def test_moc_rejections_surface_immediately():
    class T:
        def __init__(self, st, msg):
            self.orderStatus = type("S", (), {"status": st})()
            self.contract = type("C", (), {"symbol": "GDX"})()
            self.log = [type("L", (), {"message": msg})()]
    bad = ic.moc_rejections([(None, 1, 0, 0, T("Inactive", "IRA cannot short")),
                             (None, 1, 0, 0, T("Submitted", ""))])
    assert bad == [("GDX", "Inactive", "IRA cannot short")]


# ── config validation ───────────────────────────────────────────────────────
def _cfg(**over):
    raw = {"accounts": [{"id": "U1111111", "type": "margin"},
                        {"id": "U2222222", "type": "ira"}]}
    raw.update(over)
    return raw


def test_example_config_is_valid():
    cfg, errs = acc.load_config(ROOT / "deploy" / "ibkr_accounts.example.json")
    assert not errs and cfg and [a.id for a in cfg.enabled()] == ["U1111111", "U2222222"]


def test_valid_config_defaults():
    cfg, errs = acc.validate_config(_cfg())
    assert not errs and cfg.port == ic.DEFAULT_LIVE_PORT
    assert cfg.accounts[0].limits["max_deploy_frac"] == 1.0


def test_paper_account_is_rejected():
    _, errs = acc.validate_config({"accounts": [{"id": "DU1234567"}]})
    assert any("paper" in e for e in errs)


@pytest.mark.parametrize("bad", ["", "1234567", "U12", "u1234567", "X1234567", "DF1234567"])
def test_malformed_ids_are_rejected(bad):
    _, errs = acc.validate_config({"accounts": [{"id": bad}]})
    assert errs


def test_duplicate_ids_and_unknown_keys_are_rejected():
    _, errs = acc.validate_config({"accounts": [{"id": "U1111111"},
                                                {"id": "U1111111"}]})
    assert any("duplicate" in e for e in errs)
    _, errs = acc.validate_config({"accounts": [{"id": "U1111111", "typo": 1}]})
    assert any("unknown setting" in e for e in errs)


def test_ira_cannot_be_allowed_margin():
    _, errs = acc.validate_config({"accounts": [
        {"id": "U1111111", "type": "ira", "allow_margin": True}]})
    assert any("margin account" in e for e in errs)


@pytest.mark.parametrize("key,val", [("max_deploy_frac", 0), ("max_deploy_frac", 1.5),
                                     ("band", -0.1), ("max_order_notional", -1),
                                     ("max_gross_frac", 3.0)])
def test_limits_are_range_checked(key, val):
    _, errs = acc.validate_config({"accounts": [{"id": "U1111111", key: val}]})
    assert errs


def test_all_accounts_disabled_is_an_error():
    _, errs = acc.validate_config({"accounts": [{"id": "U1111111", "enabled": False}]})
    assert any("no account is enabled" in e for e in errs)


def test_missing_and_malformed_files(tmp_path):
    assert acc.load_config(tmp_path / "nope.json")[1]
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    assert "not valid JSON" in acc.load_config(p)[1][0]


# ── the command line each account's executor gets ───────────────────────────
def test_child_argv_is_strict_moc_pinned_and_signed():
    cfg, _ = acc.validate_config(_cfg(order_type="moc"))
    argv = acc.child_argv(cfg, cfg.accounts[1], 1, execute=True, python="py")
    s = " ".join(argv)
    for want in ("--account U2222222", "--expected-account U2222222",
                 "--account-mode live", "--confirm-live", "--order-type moc",
                 "--strict-moc", "--require-signature", "--no-push-report",
                 "--client-id 41", "--port 4001", "--execute"):
        assert want in s, want
    assert "--allow-margin" not in argv
    assert "--catch-up" not in argv and "--outside-rth" not in argv


def test_preview_has_no_execute_flag_and_keeps_the_real_report():
    cfg, _ = acc.validate_config(_cfg())
    argv = acc.child_argv(cfg, cfg.accounts[0], 0, execute=False)
    assert "--execute" not in argv
    assert "--no-report" in argv, "a preview must not overwrite the last real report"


def test_margin_is_only_passed_when_opted_in():
    cfg, _ = acc.validate_config({"accounts": [
        {"id": "U1111111", "type": "margin", "allow_margin": True}]})
    assert "--allow-margin" in acc.child_argv(cfg, cfg.accounts[0], 0, execute=True)


def test_client_ids_are_stable_per_account():
    """An account keeps its client id even when a sibling is disabled."""
    cfg, _ = acc.validate_config({"accounts": [
        {"id": "U1111111", "enabled": False}, {"id": "U2222222"}]})
    run = cfg.enabled()[0]
    idx = {a.id: i for i, a in enumerate(cfg.accounts)}[run.id]
    assert f"--client-id {cfg.client_id_base + idx}" in " ".join(
        acc.child_argv(cfg, run, idx, execute=True))


# ── availability + exit codes ───────────────────────────────────────────────
def test_missing_accounts_are_skipped_not_substituted():
    cfg, _ = acc.validate_config(_cfg())
    ok, missing = acc.split_by_availability(cfg, ["U1111111", "U7777777"])
    assert [a.id for a in ok] == ["U1111111"] and [a.id for a in missing] == ["U2222222"]


def _res(rc):
    return acc.Result(acc.Account("U1"), rc=rc)


def test_overall_exit_codes():
    assert acc.overall_exit([_res(0), _res(0)], []) == acc.EXIT_OK
    assert acc.overall_exit([_res(3), _res(3)], []) == acc.EXIT_SKIPPED
    assert acc.overall_exit([_res(0), _res(3)], []) == acc.EXIT_OK
    assert acc.overall_exit([_res(0), _res(1)], []) == acc.EXIT_FAILED
    assert acc.overall_exit([_res(4)], []) == acc.EXIT_FAILED
    assert acc.overall_exit([_res(None)], []) == acc.EXIT_FAILED
    assert acc.overall_exit([_res(0)], [acc.Account("U9")]) == acc.EXIT_FAILED


# ── the runner really runs the accounts together ────────────────────────────
def test_accounts_run_concurrently_and_failures_are_isolated(tmp_path, monkeypatch):
    fake = tmp_path / "fake_exec.py"
    fake.write_text(textwrap.dedent("""
        import sys, time
        a = sys.argv[sys.argv.index("--account") + 1]
        print("start", a, flush=True)
        time.sleep(1.5)
        print("end", a, flush=True)
        sys.exit(1 if a.endswith("2") else 0)
    """))
    monkeypatch.setattr(acc, "EXECUTOR", fake)
    monkeypatch.setattr(acc, "LOG_DIR", tmp_path / "logs")
    cfg, errs = acc.validate_config({"stagger_s": 0, "accounts": [
        {"id": "U1111111"}, {"id": "U2222222"}, {"id": "U3333333"}]})
    assert not errs
    t0 = time.monotonic()
    results = acc.run_all(cfg, cfg.enabled(), execute=True)
    elapsed = time.monotonic() - t0
    assert elapsed < 3.5, f"accounts ran one after another ({elapsed:.1f}s)"
    assert {r.acct.id: r.rc for r in results} == {"U1111111": 0, "U2222222": 1,
                                                  "U3333333": 0}
    assert acc.overall_exit(results, []) == acc.EXIT_FAILED
    assert (tmp_path / "logs" / "U2222222.log").read_text().count("end U2222222") == 1


def test_a_hung_child_is_killed(tmp_path, monkeypatch):
    fake = tmp_path / "hang.py"
    fake.write_text("import time\ntime.sleep(60)\n")
    monkeypatch.setattr(acc, "EXECUTOR", fake)
    monkeypatch.setattr(acc, "LOG_DIR", tmp_path / "logs")
    cfg, _ = acc.validate_config({"accounts": [{"id": "U1111111"}]})
    cfg.child_timeout_s = 1.0
    t0 = time.monotonic()
    (r,) = acc.run_all(cfg, cfg.enabled(), execute=True)
    assert time.monotonic() - t0 < 10 and r.rc is None and "killed" in r.note


# ── the executor's own wiring ───────────────────────────────────────────────
def test_executor_source_is_wired_for_strict_moc_and_accounts():
    src = (ROOT / "scripts" / "ibkr_execute_book.py").read_text()
    for needle in ("--strict-moc", "--account", "acquire_send_lock",
                   "check_no_working_orders", "moc_strict=args.strict_moc",
                   "EXIT_MOC_SKIPPED", "ACCOUNTS_DIR"):
        assert needle in src, needle


# ── the Executed Book page can find per-account reports ─────────────────────
def test_per_account_reports_are_discovered_and_isolated(tmp_path):
    import executed_book as eb
    for acct in ("U2222222", "U1111111"):
        d = tmp_path / "accounts" / acct
        (d / "executed_archive").mkdir(parents=True)
        (d / "executed_book_live.json").write_text("{}")
    (tmp_path / "accounts" / "U3333333").mkdir()          # empty: not listed
    paths = eb.account_report_paths(tmp_path)
    assert [eb.account_id_of(p) for p in paths] == ["U1111111", "U2222222"]
    # archive naming works per account without touching the module's keying
    a = eb.archive_path(paths[0], "2026-10-09")
    assert a.parent == tmp_path / "accounts" / "U1111111" / "executed_archive"
    assert a.name == "2026-10-09_live.json"
    assert eb.archive_dir(paths[0]) != eb.archive_dir(paths[1])
    assert eb.account_id_of(tmp_path / "executed_book_live.json") is None


def test_duplicate_run_lock_is_per_account(tmp_path):
    import executed_book as eb
    a = tmp_path / "accounts" / "U1" / "executed_book_live.json"
    b = tmp_path / "accounts" / "U2" / "executed_book_live.json"
    payload = eb.build_payload(as_of="2026-10-09", profile="Balanced",
                               mode="execute", account="U1", net_liq=1.0,
                               cash=1.0, trades=[], positions=[],
                               account_mode="live")
    eb.archive_report(payload, a, None)
    assert eb.completed_run(a, "2026-10-09") is not None
    assert eb.completed_run(b, "2026-10-09") is None, \
        "account U2 must not be locked out by U1's execution"


def test_market_data_type_is_validated_and_passed_through():
    cfg, errs = acc.validate_config({"market_data_type": 3,
                                     "accounts": [{"id": "U1111111"}]})
    assert not errs
    assert "--market-data-type 3" in " ".join(
        acc.child_argv(cfg, cfg.accounts[0], 0, execute=True))
    _, errs = acc.validate_config({"market_data_type": 9,
                                   "accounts": [{"id": "U1111111"}]})
    assert errs
