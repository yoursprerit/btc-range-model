"""The dead-man's switch (heartbeat pings) and the after-the-close reconcile."""
import http.server
import json
import sys
import threading
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "app"))

import executed_book as eb  # noqa: E402
import ibkr_common as ic  # noqa: E402
import ibkr_execute_accounts as acc  # noqa: E402
import ibkr_heartbeat as hb  # noqa: E402
import ibkr_reconcile_accounts as rc  # noqa: E402

ET = "America/New_York"


# ═══ heartbeat module ═══════════════════════════════════════════════════════
class _Recorder(http.server.BaseHTTPRequestHandler):
    seen: list = []
    status = 200

    def _go(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n).decode() if n else ""
        type(self).seen.append((self.command, self.path, body))
        self.send_response(type(self).status)
        self.end_headers()

    do_GET = do_POST = _go

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    _Recorder.seen, _Recorder.status = [], 200
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Recorder)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/ping/abc"
    srv.shutdown()


def test_healthchecks_style_paths_and_body(server):
    assert hb.ping("start", "go", url=server)
    assert hb.ping("ok", "all fine", url=server)
    assert hb.ping("fail", "bad", url=server)
    assert [(m, p) for m, p, _ in _Recorder.seen] == [
        ("POST", "/ping/abc/start"), ("POST", "/ping/abc"), ("POST", "/ping/abc/fail")]
    assert [b for *_, b in _Recorder.seen] == ["go", "all fine", "bad"]


def test_kuma_style_uses_status_query(server):
    assert hb.ping("ok", "fine", url=server, style="kuma")
    assert hb.ping("fail", "broke", url=server, style="kuma")
    assert hb.ping("start", url=server, style="kuma") is False       # no start signal
    paths = [p for _, p, _ in _Recorder.seen]
    assert paths[0].startswith("/ping/abc?status=up&msg=fine")
    assert paths[1].startswith("/ping/abc?status=down&msg=broke")
    assert len(paths) == 2


def test_url_comes_from_the_environment(server, monkeypatch):
    monkeypatch.setenv("IBKR_HEARTBEAT_URL", server)
    assert hb.ping("ok", "x")
    monkeypatch.setenv("MY_OTHER", server)
    assert hb.ping("ok", "x", env_var="MY_OTHER")


def test_no_or_bad_url_is_a_quiet_no_op(monkeypatch):
    monkeypatch.delenv("IBKR_HEARTBEAT_URL", raising=False)
    assert hb.ping("ok", "x") is False
    for bad in ("", "not a url", "ftp://x/y", "http://example.com/abc", "https://"):
        assert hb.ping("ok", "x", url=bad) is False
    assert hb.valid_url("https://hc-ping.com/abc") and hb.valid_url("http://localhost:1/a")
    assert not hb.valid_url("http://hc-ping.com/abc")                # bearer token: https only


def test_a_dead_monitor_never_raises_or_hangs():
    assert hb.ping("ok", "x", url="http://127.0.0.1:9/none", timeout=1, retries=1) is False


def test_server_errors_are_retried_then_give_up(server):
    _Recorder.status = 500
    assert hb.ping("ok", "x", url=server, retries=2) is False
    assert len(_Recorder.seen) == 3


def test_mask_account():
    assert hb.mask_account("U1234567") == "…4567" and hb.mask_account("U1") == "U1"


# ═══ the runner pings around a real run ═════════════════════════════════════
@pytest.fixture
def pings(monkeypatch):
    got = []
    monkeypatch.setattr(hb, "ping", lambda kind, msg="", **k: got.append((kind, msg)) or True)
    return got


def _runner_cfg(tmp_path, kill=None):
    raw = {"accounts": [{"id": "U1111111"}]}
    if kill:
        raw["kill_switch_file"] = str(kill)
    p = tmp_path / "accts.json"
    p.write_text(json.dumps(raw))
    return str(p)


@pytest.fixture
def ready(monkeypatch):
    monkeypatch.setenv("OVERALL_BOOK_SECRET", "s3cret")
    monkeypatch.delenv("IBKR_TRADING_DISABLED", raising=False)
    monkeypatch.setattr(ic, "is_trading_day", lambda d: (True, "trading day"))
    monkeypatch.setattr(ic, "moc_entry_open", lambda now=None, cutoff=None: (True, "15:20 ET"))
    monkeypatch.setattr(acc, "alert", lambda m: None)
    monkeypatch.setattr(acc, "push_reports", lambda a, b=None: True)
    monkeypatch.setattr(acc, "verify_gateway", lambda cfg, **k: (["U1111111"], "ok"))
    monkeypatch.setattr(acc, "run_all", lambda cfg, accts, execute: [
        acc.Result(accts[0], rc=0)])


def test_a_good_run_pings_start_then_ok(tmp_path, ready, pings):
    assert acc.main(["--execute", "--config", _runner_cfg(tmp_path)]) == acc.EXIT_OK
    assert [k for k, _ in pings] == ["start", "ok"]
    assert "…1111" in pings[-1][1] and "U1111111" not in pings[-1][1]


def test_a_failed_account_pings_fail(tmp_path, ready, pings, monkeypatch):
    monkeypatch.setattr(acc, "run_all", lambda cfg, accts, execute: [
        acc.Result(accts[0], rc=1)])
    assert acc.main(["--execute", "--config", _runner_cfg(tmp_path)]) == acc.EXIT_FAILED
    assert [k for k, _ in pings] == ["start", "fail"]


def test_a_weekend_is_a_successful_nothing_to_do_run(tmp_path, ready, pings, monkeypatch):
    monkeypatch.setattr(ic, "is_trading_day", lambda d: (False, "2026-10-10 is a weekend"))
    assert acc.main(["--execute", "--config", _runner_cfg(tmp_path)]) == acc.EXIT_SKIPPED
    assert [k for k, _ in pings] == ["start", "ok"]            # fired as scheduled: no false alarm
    assert "weekend" in pings[-1][1]


def test_an_armed_kill_switch_is_not_an_alarm(tmp_path, ready, pings):
    ks = tmp_path / "STOP"
    ks.write_text("")
    assert acc.main(["--execute", "--config", _runner_cfg(tmp_path, ks)]) == acc.EXIT_SKIPPED
    assert [k for k, _ in pings] == ["start", "ok"]


def test_an_armed_kill_switch_still_tells_a_human(tmp_path, ready, pings, monkeypatch):
    """Not an alarm for the dead-man's switch (the task did fire), but never
    silent: a switch thrown in a panic and forgotten loses every later session."""
    sent = []
    monkeypatch.setattr(acc, "alert", sent.append)
    ks = tmp_path / "STOP"
    ks.write_text("")
    acc.main(["--execute", "--config", _runner_cfg(tmp_path, ks)])
    assert len(sent) == 1 and "kill switch" in sent[0].lower() and "ACTIVE" in sent[0]


def test_starting_after_the_cutoff_pings_fail(tmp_path, ready, pings, monkeypatch):
    monkeypatch.setattr(ic, "moc_entry_open", lambda now=None, cutoff=None: (False, "late"))
    assert acc.main(["--execute", "--config", _runner_cfg(tmp_path)]) == acc.EXIT_SKIPPED
    assert [k for k, _ in pings] == ["start", "fail"]


def test_an_unreachable_gateway_pings_fail(tmp_path, ready, pings, monkeypatch):
    monkeypatch.setattr(acc, "verify_gateway", lambda cfg, **k: (None, "gateway down"))
    assert acc.main(["--execute", "--config", _runner_cfg(tmp_path)]) == acc.EXIT_SETUP
    assert [k for k, _ in pings] == ["start", "fail"] and "gateway down" in pings[-1][1]


def test_a_config_error_pings_fail(tmp_path, ready, pings):
    bad = tmp_path / "bad.json"
    bad.write_text("{nope")
    assert acc.main(["--execute", "--config", str(bad)]) == acc.EXIT_SETUP
    assert [k for k, _ in pings] == ["start", "fail"]


def test_a_crash_pings_fail_and_still_raises(tmp_path, ready, pings, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("kaboom")
    monkeypatch.setattr(acc, "run_all", boom)
    with pytest.raises(RuntimeError):
        acc.main(["--execute", "--config", _runner_cfg(tmp_path)])
    assert [k for k, _ in pings] == ["start", "fail"] and "kaboom" in pings[-1][1]


def test_previews_and_checks_never_ping(tmp_path, ready, pings):
    """A hand-run preview must not stand in for the missing scheduled run."""
    acc.main(["--config", _runner_cfg(tmp_path)])
    acc.main(["--check", "--config", _runner_cfg(tmp_path)])
    assert pings == []


# ═══ reconcile: pure helpers ════════════════════════════════════════════════
def _fill(sym, action, filled, avg=10.0):
    return dict(symbol=sym, action=action, filled=filled, avg_fill_price=avg, qty=filled,
                price=avg, status="Filled", key=sym, order_type="", limit_price=0.0,
                reason="reconstructed from broker fills")


def _plan(sym, action, qty, filled=None, status="Filled"):
    return dict(symbol=sym, action=action, qty=qty, price=10.0, key=sym, status=status,
                filled=qty if filled is None else filled, avg_fill_price=10.0,
                order_type="moc", limit_price=0.0, reason="→ 8.3%")


def test_unfilled_legs_names_the_gaps():
    gaps = rc.unfilled_legs([_plan("GRID", "BUY", 14), _plan("XLE", "SELL", 294)],
                            [_fill("GRID", "BUY", 14), _fill("XLE", "SELL", 100)])
    assert gaps == ["SELL XLE: 100 of 294 filled"]
    assert rc.unfilled_legs([_plan("GRID", "BUY", 14, status="SKIPPED-FUNDING")], []) == []


def test_merge_keeps_the_plan_and_takes_fills_from_the_broker():
    prior = [_plan("GRID", "BUY", 14, filled=0, status="Submitted"),
             _plan("XOP", "BUY", 6, status="SKIPPED-FUNDING")]
    out = rc.merge_trades(prior, [_fill("GRID", "BUY", 14, 181.2), _fill("ZZZ", "BUY", 3)])
    g = [t for t in out if t["symbol"] == "GRID"][0]
    assert g["filled"] == 14 and g["avg_fill_price"] == 181.2 and g["status"] == "Filled"
    assert g["reason"] == "→ 8.3%" and g["order_type"] == "moc"           # plan kept
    assert [t for t in out if t["symbol"] == "XOP"][0]["status"] == "SKIPPED-FUNDING"
    assert "no planned order" in [t for t in out if t["symbol"] == "ZZZ"][0]["reason"]


def test_partial_and_unfilled_statuses():
    out = rc.merge_trades([_plan("A", "BUY", 10), _plan("B", "BUY", 10)],
                          [_fill("A", "BUY", 4)])
    assert {t["symbol"]: t["status"] for t in out} == {"A": "Partial", "B": "Unfilled"}


def test_needs_rewrite_only_when_the_broker_disagrees():
    assert not rc.needs_rewrite([_plan("A", "BUY", 5)], [_fill("A", "BUY", 5)])
    assert rc.needs_rewrite([_plan("A", "BUY", 5, filled=0)], [_fill("A", "BUY", 5)])
    assert rc.needs_rewrite([_plan("A", "BUY", 5)], [_fill("A", "BUY", 5), _fill("Q", "SELL", 1)])


def _accounts(*ids):
    cfg, _ = acc.validate_config({"accounts": [{"id": i} for i in ids]})
    return cfg


def _lock(tmp, acct, as_of, when_utc):
    d = tmp / acct / ".send_locks"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{as_of}.lock").write_text(json.dumps(
        dict(account=acct, as_of=as_of, locked_at_utc=when_utc.isoformat())))


@pytest.fixture
def accts_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(acc, "ACCOUNTS_DIR", tmp_path / "accounts")
    return tmp_path / "accounts"


def test_todays_locks_only_counts_orders_sent_today(accts_dir):
    cfg = _accounts("U1111111", "U2222222", "U3333333")
    now = pd.Timestamp("2026-10-12 16:15", tz=ET)
    _lock(accts_dir, "U1111111", "2026-10-09", pd.Timestamp("2026-10-12 19:30", tz="UTC"))
    _lock(accts_dir, "U2222222", "2026-10-08", pd.Timestamp("2026-10-11 19:30", tz="UTC"))  # yesterday
    (accts_dir / "U3333333" / ".send_locks").mkdir(parents=True)
    (accts_dir / "U3333333" / ".send_locks" / "junk.lock").write_text("{not json")
    assert rc.todays_locks(cfg.enabled(), now) == {"U1111111": "2026-10-09"}


def test_the_newest_lock_of_the_day_wins(accts_dir):
    cfg = _accounts("U1111111")
    _lock(accts_dir, "U1111111", "2026-10-09", pd.Timestamp("2026-10-12 18:00", tz="UTC"))
    _lock(accts_dir, "U1111111", "2026-10-10", pd.Timestamp("2026-10-12 19:00", tz="UTC"))
    assert rc.todays_locks(cfg.enabled(), pd.Timestamp("2026-10-12 17:00", tz=ET)) == \
        {"U1111111": "2026-10-10"}


# ═══ reconcile: one account against a fake broker ═══════════════════════════
class _Order:
    def __init__(self, action, qty):
        self.action, self.totalQuantity = action, qty


class _WT:
    def __init__(self, sym, action, qty):
        self.order, self.contract = _Order(action, qty), type("C", (), {"symbol": sym})()


class _Broker:
    account = "U1111111"

    def __init__(self, fills=(), working=(), fail_working=False):
        self._fills, self._working, self._fail = list(fills), list(working), fail_working
        self.disconnected = False

    def day_fills_by_symbol(self): return list(self._fills)

    def working_orders(self):
        if self._fail:
            raise RuntimeError("no answer")
        return list(self._working)

    def net_liq(self): return 77_000.0
    def cash(self): return 1_000.0
    def portfolio_snapshot(self): return []
    def realized_pnl(self): return None
    def disconnect(self): self.disconnected = True


def _report_for(acct_id, trades, accts_dir, as_of="2026-10-09"):
    rep = accts_dir / acct_id / "executed_book_live.json"
    payload = eb.build_payload(as_of=as_of, profile="Balanced", mode="execute",
                               account=acct_id, net_liq=77_000.0, cash=1_000.0,
                               trades=trades, positions=[], account_mode="live")
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(payload))
    eb.archive_report(payload, rep, None)
    return rep


def _recon(cfg, broker, *, dry=False, secret="s3cret", as_of="2026-10-09"):
    a = cfg.accounts[0]

    def make(*args, **kw):
        if isinstance(broker, Exception):
            raise broker
        return broker
    return rc.reconcile_account(cfg, a, as_of, 100, dry_run=dry, secret=secret,
                                broker_factory=make)


def test_a_clean_account_reconciles_without_touching_the_report(accts_dir):
    cfg = _accounts("U1111111")
    rep = _report_for("U1111111", [_plan("GRID", "BUY", 14)], accts_dir)
    before = rep.read_text()
    b = _Broker(fills=[_fill("GRID", "BUY", 14)])
    f = _recon(cfg, b)
    assert f.ok and not f.rewritten and rep.read_text() == before and b.disconnected


def test_a_missing_report_is_rebuilt_from_the_brokers_fills(accts_dir):
    cfg = _accounts("U1111111")
    f = _recon(cfg, _Broker(fills=[_fill("GRID", "BUY", 14), _fill("XLE", "SELL", 60)]))
    assert not f.ok and f.rewritten
    assert "did not finish" in f.problems[0]
    rep = accts_dir / "U1111111" / "executed_book_live.json"
    d = json.loads(rep.read_text())
    assert d["account"] == "U1111111" and d["as_of"] == "2026-10-09"
    assert {t["symbol"] for t in d["trades"]} == {"GRID", "XLE"}
    assert d.get("signature")                                    # signed
    assert (accts_dir / "U1111111" / "executed_archive" / "2026-10-09_live.json").exists()


def test_a_missing_report_without_the_secret_is_flagged_not_written(accts_dir):
    cfg = _accounts("U1111111")
    f = _recon(cfg, _Broker(fills=[_fill("GRID", "BUY", 14)]), secret=None)
    assert not f.rewritten and any("cannot be signed" in p for p in f.problems)
    assert not (accts_dir / "U1111111" / "executed_book_live.json").exists()


def test_an_unfilled_leg_is_a_problem(accts_dir):
    cfg = _accounts("U1111111")
    _report_for("U1111111", [_plan("GRID", "BUY", 14), _plan("XLE", "SELL", 60)], accts_dir)
    f = _recon(cfg, _Broker(fills=[_fill("GRID", "BUY", 14)]))
    assert not f.ok and any("SELL XLE: 0 of 60 filled" in p for p in f.problems)


def test_late_fills_are_written_into_the_report(accts_dir):
    cfg = _accounts("U1111111")
    rep = _report_for("U1111111", [_plan("GRID", "BUY", 14, filled=0, status="Submitted")],
                      accts_dir)
    f = _recon(cfg, _Broker(fills=[_fill("GRID", "BUY", 14, 182.0)]))
    assert f.ok and f.rewritten
    t = json.loads(rep.read_text())["trades"][0]
    assert t["filled"] == 14 and t["status"] == "Filled" and t["reason"] == "→ 8.3%"


def test_no_executions_at_the_broker_never_overwrites_a_real_record(accts_dir):
    cfg = _accounts("U1111111")
    rep = _report_for("U1111111", [_plan("GRID", "BUY", 14)], accts_dir)
    before = rep.read_text()
    f = _recon(cfg, _Broker(fills=[]))
    assert not f.ok and not f.rewritten and rep.read_text() == before
    assert any("NO executions" in p for p in f.problems)


def test_orders_still_working_are_a_problem(accts_dir):
    cfg = _accounts("U1111111")
    _report_for("U1111111", [_plan("GRID", "BUY", 14)], accts_dir)
    f = _recon(cfg, _Broker(fills=[_fill("GRID", "BUY", 14)],
                            working=[_WT("SATA", "BUY", 65)]))
    assert not f.ok and "still working" in f.problems[0] and "BUY 65 SATA" in f.problems[0]


def test_an_unreadable_working_orders_list_is_a_problem(accts_dir):
    cfg = _accounts("U1111111")
    _report_for("U1111111", [_plan("GRID", "BUY", 14)], accts_dir)
    f = _recon(cfg, _Broker(fills=[_fill("GRID", "BUY", 14)], fail_working=True))
    assert any("working orders" in p for p in f.problems)


def test_a_connection_failure_is_reported_not_raised(accts_dir):
    f = _recon(_accounts("U1111111"), ConnectionError("refused"))
    assert not f.ok and "could not connect" in f.problems[0]


def test_dry_run_changes_nothing(accts_dir):
    cfg = _accounts("U1111111")
    f = _recon(cfg, _Broker(fills=[_fill("GRID", "BUY", 14)]), dry=True)
    assert not f.rewritten and any("would be rewritten" in n for n in f.notes)
    assert not (accts_dir / "U1111111" / "executed_book_live.json").exists()


# ═══ reconcile: the whole job ═══════════════════════════════════════════════
@pytest.fixture
def job(tmp_path, accts_dir, monkeypatch):
    cfgp = tmp_path / "accts.json"
    cfgp.write_text(json.dumps({"accounts": [{"id": "U1111111"}]}))
    state = dict(pings=[], alerts=[], pushed=[])
    monkeypatch.setattr(rc.hbeat, "ping", lambda k, m="", **kw: state["pings"].append((k, m, kw)) or True)
    monkeypatch.setattr(acc, "alert", lambda m: state["alerts"].append(m))
    monkeypatch.setattr(acc, "push_reports", lambda a, b=None: state["pushed"].append([x.id for x in a]) or True)
    monkeypatch.setattr(acc, "verify_gateway", lambda cfg, **k: (["U1111111"], "ok"))
    monkeypatch.setattr(ic, "is_trading_day", lambda d: (True, "trading day"))
    monkeypatch.setenv("OVERALL_BOOK_SECRET", "s3cret")
    now = pd.Timestamp("2026-10-12 16:20", tz=ET)
    monkeypatch.setattr(rc, "_now_et", lambda: now)
    state.update(cfg=str(cfgp), now=now, dir=accts_dir)
    return state


def _sent_today(job, acct="U1111111"):
    _lock(job["dir"], acct, "2026-10-09", job["now"].tz_convert("UTC") - pd.Timedelta(hours=1))


def test_job_with_no_orders_sent_today_is_a_benign_nothing_to_do(job):
    assert rc.main(["--config", job["cfg"]]) == rc.EXIT_NOTHING
    assert [k for k, *_ in job["pings"]] == ["start", "ok"]


def test_job_before_the_auction_prints_does_nothing_and_flags_it(job, monkeypatch):
    _sent_today(job)
    monkeypatch.setattr(rc, "_now_et", lambda: pd.Timestamp("2026-10-12 15:30", tz=ET))
    assert rc.main(["--config", job["cfg"]]) == rc.EXIT_NOTHING
    assert [k for k, *_ in job["pings"]] == ["start", "fail"]      # a scheduling mistake
    assert rc.main(["--config", job["cfg"], "--force"]) in (rc.EXIT_PROBLEM, rc.EXIT_OK)


def test_job_all_good(job, monkeypatch):
    _sent_today(job)
    monkeypatch.setattr(rc, "reconcile_account", lambda *a, **k: rc.Finding("U1111111", "2026-10-09"))
    assert rc.main(["--config", job["cfg"]]) == rc.EXIT_OK
    assert [k for k, *_ in job["pings"]] == ["start", "ok"] and job["alerts"] == []
    assert job["pings"][0][2]["env_var"] == "IBKR_RECONCILE_HEARTBEAT_URL"


def test_job_with_a_problem_alerts_and_pings_fail(job, monkeypatch):
    _sent_today(job)
    monkeypatch.setattr(rc, "reconcile_account", lambda *a, **k: rc.Finding(
        "U1111111", "2026-10-09", problems=["SELL XLE: 0 of 60 filled"]))
    assert rc.main(["--config", job["cfg"]]) == rc.EXIT_PROBLEM
    assert [k for k, *_ in job["pings"]] == ["start", "fail"]
    assert len(job["alerts"]) == 1 and "SELL XLE" in job["alerts"][0]
    assert "U1111111" not in job["pings"][-1][1]                   # masked in the heartbeat


def test_job_pushes_only_rebuilt_reports(job, monkeypatch):
    monkeypatch.setenv("IBKR_PUBLISH_REPORTS", "1")
    _sent_today(job)
    monkeypatch.setattr(rc, "reconcile_account", lambda *a, **k: rc.Finding(
        "U1111111", "2026-10-09", rewritten=True))
    assert rc.main(["--config", job["cfg"]]) == rc.EXIT_OK
    assert job["pushed"] == [["U1111111"]]
    job["pushed"].clear()
    monkeypatch.setattr(rc, "reconcile_account", lambda *a, **k: rc.Finding("U1111111", "x"))
    rc.main(["--config", job["cfg"]])
    assert job["pushed"] == []


def test_reports_are_never_published_unless_opted_in(job, monkeypatch):
    _sent_today(job)
    monkeypatch.delenv("IBKR_PUBLISH_REPORTS", raising=False)
    monkeypatch.setattr(rc, "reconcile_account", lambda *a, **k: rc.Finding(
        "U1111111", "2026-10-09", rewritten=True))
    assert rc.main(["--config", job["cfg"]]) == rc.EXIT_OK
    assert job["pushed"] == []


def test_job_gateway_down(job, monkeypatch):
    _sent_today(job)
    monkeypatch.setattr(acc, "verify_gateway", lambda cfg, **k: (None, "down"))
    assert rc.main(["--config", job["cfg"]]) == rc.EXIT_SETUP
    assert [k for k, *_ in job["pings"]] == ["start", "fail"] and job["alerts"]


def test_job_dry_run_is_silent(job, monkeypatch):
    _sent_today(job)
    monkeypatch.setattr(rc, "reconcile_account", lambda *a, **k: rc.Finding(
        "U1111111", "x", rewritten=True))
    rc.main(["--config", job["cfg"], "--dry-run"])
    assert job["pings"] == [] and job["pushed"] == []


def test_gateway_retry_does_not_give_up_just_because_the_market_closed(monkeypatch):
    """The 2:15 run stops retrying near the MOC cutoff; the 4:15 job must not."""
    monkeypatch.setattr(acc, "_connect_once", lambda c: (None, "down"))
    monkeypatch.setattr(ic, "moc_entry_open", lambda now=None, cutoff=None: (False, "late"))
    naps = []
    cfg, _ = acc.validate_config({"accounts": [{"id": "U1111111"}]})
    accts, _ = acc.verify_gateway(cfg, retry_s=60, every_s=30, sleep=naps.append,
                                  clock=iter([0, 0, 30, 60, 90, 120]).__next__,
                                  respect_cutoff=False)
    assert accts is None and len(naps) >= 1


# ═══ scripts / scheduling ═══════════════════════════════════════════════════
def test_reconcile_task_is_registered_with_limited_rights_and_scripts_are_ascii():
    setup = (ROOT / "scripts" / "setup_windows_option_c.ps1").read_text()
    assert "IBKR Option C LIVE reconcile" in setup and "ReconcileTaskTime" in setup
    for f in ("setup_windows_option_c.ps1", "ibkr_reconcile_live_accounts.ps1"):
        raw = (ROOT / "scripts" / f).read_bytes()
        assert all(b < 128 for b in raw), f"{f} must stay pure ASCII"


def test_reconcile_never_places_or_cancels_orders():
    src = (ROOT / "scripts" / "ibkr_reconcile_accounts.py").read_text()
    for forbidden in (".place(", "placeOrder", "cancelOrder", "MarketOrder", "reqGlobalCancel"):
        assert forbidden not in src
