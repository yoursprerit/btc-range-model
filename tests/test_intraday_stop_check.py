"""The 2:00-PM-CT intraday stop check (``scripts/intraday_stop_check.py``).

Pins: which sleeves are evaluated and against what level, when a live quote is
trusted, the CT window, the action rewrite, that a force-exit really removes a
held sleeve from the gated book, and the whole job end to end on synthetic
books — a breach re-publishes both signed books in place, a clear day touches
nothing, a re-run is idempotent, and every guard leaves the books alone.
"""
import inspect
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import freshness as fr                      # noqa: E402
import intraday_stop_check as isc           # noqa: E402
import overall_core as oc                   # noqa: E402
import target_book as tb                    # noqa: E402
import ibkr_rebalance as rb                 # noqa: E402
from test_strategy_logic_v2 import _res     # noqa: E402

NOW = pd.Timestamp("2026-10-19 19:00", tz="UTC")        # Mon 2:00 PM CDT
SECRET = "test-secret"


def _held(key, *, stop=None, trail=None, entry=100.0, bar=100.0, in_pos=True):
    r = _res(key, in_pos=in_pos, tone="hold")
    r["pos"].update(entry_px=entry, stop_px=stop if stop else trail, trail_px=trail)
    r["bar_close"] = bar
    return r


def _q(px, **kw):
    return dict(price=px, quote_time=NOW - pd.Timedelta(minutes=2),
                session_closed=False, src="yahoo", **kw)


# ── what is evaluated, and against what ──────────────────────────────────────
def test_stop_level_fixed_trail_flat_and_unstopped():
    assert isc.stop_level(_held("A", stop=95.0)) == (95.0, "fixed")
    assert isc.stop_level(_held("A", trail=90.0)) == (90.0, "trail")      # engine mirrors trail into stop_px
    assert isc.stop_level(_held("A")) == (None, None)                     # held, no stop (e.g. SOXL)
    assert isc.stop_level(_held("A", stop=95.0, in_pos=False)) == (None, None)


def test_breach_is_at_or_below_the_level():
    res = [_held("A", stop=95.0), _held("B", stop=95.0), _held("C", trail=90.0)]
    ev = isc.evaluate_stops(res, {"A": _q(95.0), "B": _q(95.01), "C": _q(89.0)}, now=NOW)
    assert [t["key"] for t in ev["triggered"]] == ["A", "C"]
    assert [w["key"] for w in ev["watching"]] == ["B"]
    assert ev["triggered"][1]["kind"] == "trail"
    assert ev["watching"][0]["dist_pct"] == pytest.approx(0.0105, abs=1e-3)


def test_only_sleeves_in_the_published_book_are_checked():
    res = [_held("A", stop=95.0), _held("B", stop=95.0)]
    ev = isc.evaluate_stops(res, {"A": _q(80.0), "B": _q(80.0)}, held_keys={"B"}, now=NOW)
    assert [t["key"] for t in ev["triggered"]] == ["B"]


@pytest.mark.parametrize("quote, why", [
    ({}, "no live quote"),
    ({"price": None}, "no live quote"),
    (dict(price=90.0, session_closed=True), "session closed"),
    (dict(price=90.0, session_closed=False,
          quote_time=NOW - pd.Timedelta(minutes=45)), "min old"),
    (dict(price=40.0, session_closed=False,
          quote_time=NOW - pd.Timedelta(minutes=1)), "implausible"),
])
def test_an_untrustworthy_quote_never_triggers(quote, why):
    ev = isc.evaluate_stops([_held("A", stop=95.0)], {"A": quote}, now=NOW)
    assert ev["triggered"] == [] and ev["watching"] == []
    assert why in ev["skipped"][0]["reason"]


def test_backup_feed_quote_without_a_timestamp_is_still_usable():
    ev = isc.evaluate_stops([_held("A", stop=95.0)],
                            {"A": dict(price=90.0, src="nasdaq")}, now=NOW)
    assert ev["triggered"][0]["src"] == "nasdaq"


# ── window ───────────────────────────────────────────────────────────────────
def test_window_is_bounded_before_the_executor_slot():
    ct = lambda s: pd.Timestamp(s, tz=fr.CT)             # noqa: E731
    assert isc.in_window(ct("2026-10-19 14:00"), "13:45", "14:12")[0]
    assert not isc.in_window(ct("2026-10-19 13:30"), "13:45", "14:12")[0]
    ok, why = isc.in_window(ct("2026-10-19 14:13"), "13:45", "14:12")
    assert not ok and "executor" in why


def test_amend_marks_the_stopped_sleeve_close():
    acts = [dict(key="A", action="HOLD", decision="LONG", target=0.0, in_pos=True),
            dict(key="B", action="HOLD", decision="LONG", target=0.2, in_pos=True)]
    out = isc.amend_actions(acts, [dict(key="A", live=90.0, kind="fixed", level=95.0)])
    assert out[0]["action"] == "CLOSE" and out[0]["target"] == 0.0
    assert "STOP" in out[0]["decision"]
    assert out[1]["action"] == "HOLD" and acts[0]["action"] == "HOLD"     # input untouched


# ── the allocator really drops a force-exited held sleeve ────────────────────
def test_force_exit_removes_a_held_sleeve_and_pins_the_rest():
    w = {"AAA": 0.3, "BBB": 0.3}
    held = [_res("AAA", in_pos=True, tone="hold"), _res("BBB", in_pos=True, tone="hold")]
    prev = {"AAA": 0.28, "BBB": 0.27}               # within the 8 pp adds-only band of the 0.30 caps
    base = oc.signal_gated_allocation(held, w, prev_weights=prev)
    out = oc.signal_gated_allocation(held, w, prev_weights=prev, force_exit={"AAA"})
    assert base["target"]["AAA"] == pytest.approx(0.28)
    assert out["target"].get("AAA", 0.0) == 0.0
    assert out["target"]["BBB"] == pytest.approx(base["target"]["BBB"])    # not reshuffled
    assert out["idle"] > base["idle"]


def test_compute_target_book_accepts_force_exit():
    assert "force_exit" in inspect.signature(rb.compute_target_book).parameters


# ── the job, end to end ──────────────────────────────────────────────────────
class _FakeBook:
    def __init__(self, weights, cash, actions):
        self.as_of = "2026-10-16"; self.weights = weights; self.cash_weight = cash
        self.exec_price = {k: 50.0 for k in weights}; self.actions = actions
        self.equity_close = "2026-10-16"; self.sata_cap = 0.30


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("OVERALL_BOOK_SECRET", SECRET)
    monkeypatch.setattr(fr, "now_utc", lambda: NOW)
    monkeypatch.setattr(oc, "fetch_sata", lambda: {"price": 25.0})
    monkeypatch.setattr(fr, "audit_universe", lambda *a, **k: dict(
        passed=True, checked_at_utc=NOW.isoformat(), stale_apps=[], rows=[]))
    results = [_held("AAA", stop=95.0), _held("BBB", stop=90.0), _held("CCC")]
    monkeypatch.setattr(oc, "run_universe", lambda *a, **k: results)
    spot = {"AAA": _q(92.0), "BBB": _q(99.0), "CCC": _q(10.0)}
    monkeypatch.setattr(oc, "fetch_spot", lambda *a, **k: spot)
    calls = {}

    def fake_ctb(profile, results=None, live_adjust=True, prev_weights=None, force_exit=None):
        calls.update(profile=profile, prev=dict(prev_weights), force=set(force_exit))
        keep = {k: w for k, w in prev_weights.items() if k not in force_exit}
        acts = [dict(key=k, action="HOLD", decision="LONG", target=keep.get(k, 0.0),
                     in_pos=True) for k in ("AAA", "BBB", "CCC")]
        return _FakeBook(keep, 1.0 - sum(keep.values()), acts)
    monkeypatch.setattr(rb, "compute_target_book", fake_ctb)

    def write_books(extra=None):
        for name, mode in (("target_book.json", "paper"), ("target_book_live.json", "live")):
            p = tb.build_payload(as_of="2026-10-16", profile="Balanced",
                                 weights={"AAA": 0.2, "BBB": 0.25, "CCC": 0.2, "SATA": 0.15},
                                 cash_weight=0.2, exec_price={"AAA": 92.0},
                                 book_mode=mode, sata_cap=0.30,
                                 generated_at_utc="2026-10-19T12:20:00+00:00")
            p.update(extra or {})
            (tmp_path / name).write_text(tb.dumps(p, SECRET))
    write_books()
    argv = ["--out", str(tmp_path / "target_book.json"),
            "--live-out", str(tmp_path / "target_book_live.json"),
            "--marker-out", str(tmp_path / "marker.json")]
    return dict(tmp=tmp_path, argv=argv, calls=calls, write_books=write_books,
                monkeypatch=monkeypatch)


def _marker(env):
    return json.loads((env["tmp"] / "marker.json").read_text())


def test_a_breach_republishes_both_signed_books_in_place(env):
    assert isc.main(env["argv"]) == 0
    assert env["calls"]["force"] == {"AAA"}
    assert env["calls"]["prev"] == {"AAA": 0.2, "BBB": 0.25, "CCC": 0.2}   # morning book, SATA dropped
    for name in ("target_book.json", "target_book_live.json"):
        b = tb.loads((env["tmp"] / name).read_text())
        assert tb.verify_signature(b, SECRET)[0]
        assert "AAA" not in b["weights"] and b["weights"]["BBB"] == pytest.approx(0.25)
        assert b["intraday_stop"]["stopped_keys"] == ["AAA"]
        assert b["intraday_stop"]["morning_generated_at_utc"] == "2026-10-19T12:20:00+00:00"
        assert {a["key"]: a["action"] for a in b["actions"]}["AAA"] == "CLOSE"
        assert b["generated_at_utc"] == NOW.isoformat(timespec="seconds")   # the amendment's own time
    assert tb.loads((env["tmp"] / "target_book.json").read_text())["book_mode"] == "paper"
    assert tb.loads((env["tmp"] / "target_book_live.json").read_text())["book_mode"] == "live"
    assert not (env["tmp"] / "target_book_prev.json").exists()           # same-day: no rotation
    m = _marker(env)
    assert m["status"] == "stopped" and m["books_rewritten"] is True
    assert m["triggered"][0]["key"] == "AAA"


def test_rerun_is_idempotent(env):
    isc.main(env["argv"])
    first = (env["tmp"] / "target_book.json").read_text()
    assert isc.main(env["argv"]) == 0
    assert _marker(env)["status"] == "already_applied"
    assert (env["tmp"] / "target_book.json").read_text() == first


def test_no_breach_leaves_the_books_byte_identical(env):
    env["monkeypatch"].setattr(oc, "fetch_spot", lambda *a, **k: {
        "AAA": _q(99.0), "BBB": _q(99.0), "CCC": _q(10.0)})
    before = {n: (env["tmp"] / n).read_text() for n in ("target_book.json", "target_book_live.json")}
    assert isc.main(env["argv"]) == 0
    assert {n: (env["tmp"] / n).read_text() for n in before} == before
    m = _marker(env)
    assert m["status"] == "clear" and m["books_rewritten"] is False
    assert m["watching"][0]["key"] == "AAA"                                # nearest to its stop first


def test_dry_run_writes_nothing(env):
    before = (env["tmp"] / "target_book.json").read_text()
    assert isc.main(env["argv"] + ["--dry-run"]) == 0
    assert (env["tmp"] / "target_book.json").read_text() == before
    assert not (env["tmp"] / "marker.json").exists()


def test_outside_the_window_or_a_holiday_does_nothing(env):
    env["monkeypatch"].setattr(fr, "now_utc", lambda: pd.Timestamp("2026-10-19 19:30", tz="UTC"))
    assert isc.main(env["argv"]) == 0                                      # 2:30 PM CT: executor's slot
    env["monkeypatch"].setattr(fr, "now_utc", lambda: pd.Timestamp("2026-10-17 19:00", tz="UTC"))
    assert isc.main(env["argv"]) == 0                                      # Saturday
    assert not (env["tmp"] / "marker.json").exists()
    assert "AAA" in tb.loads((env["tmp"] / "target_book.json").read_text())["weights"]


def test_yesterdays_book_is_not_amended(env):
    env["write_books"]()
    p = env["tmp"] / "target_book.json"
    b = tb.loads(p.read_text()); b["generated_at_utc"] = "2026-10-16T12:20:00+00:00"
    p.write_text(tb.dumps(b, SECRET))
    assert isc.main(env["argv"]) == 0
    assert _marker(env)["status"] == "no_current_book"
    assert "AAA" in tb.loads(p.read_text())["weights"]


def test_a_bad_signature_or_missing_secret_aborts_without_writing(env):
    env["monkeypatch"].setenv("OVERALL_BOOK_SECRET", "wrong")
    assert isc.main(env["argv"]) == 1
    assert _marker(env)["status"] == "error"
    env["monkeypatch"].delenv("OVERALL_BOOK_SECRET")
    assert isc.main(env["argv"]) == 1
    assert "AAA" in tb.loads((env["tmp"] / "target_book.json").read_text())["weights"]


def test_a_failed_freshness_audit_acts_on_nothing(env):
    env["monkeypatch"].setattr(fr, "audit_universe", lambda *a, **k: dict(
        passed=False, checked_at_utc=NOW.isoformat(), stale_apps=["GRID"], rows=[]))
    before = (env["tmp"] / "target_book.json").read_text()
    assert isc.main(env["argv"]) == 1
    assert _marker(env)["status"] == "audit_failed"
    assert (env["tmp"] / "target_book.json").read_text() == before
