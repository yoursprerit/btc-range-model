"""Unit tests for the publisher's withheld-streak counter and gate reporting
(``scripts/publish_target_book.py``).

Context: a freshness audit that WITHHOLDS the target book is the designed safe
path, but it used to fail the workflow exactly like a crash — so on 2026-09-01 a
red X meant either "a feed is briefly behind" or "the engine is broken" and only
reading the log told you which.  The streak counter lets a first-cycle withhold
annotate loudly while a withhold that persists into a second cycle goes red.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "app"))
sys.path.insert(0, str(_ROOT / "scripts"))


def _load():
    spec = importlib.util.spec_from_file_location(
        "publish_target_book", _ROOT / "scripts" / "publish_target_book.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ptb = _load()

CT_NOON = "2026-09-01T17:00:00Z"          # 12:00 CT on Tue 2026-09-01


def _audit_file(tmp_path, **target_book):
    p = tmp_path / "daily_audit.json"
    p.write_text(json.dumps({"target_book": target_book}))
    return p


def test_first_withheld_cycle_counts_as_one(tmp_path):
    p = _audit_file(tmp_path, published=True)
    since, days = ptb._withheld_streak(p, False, pd.Timestamp(CT_NOON))
    assert (since, days) == ("2026-09-01", 1)


def test_streak_grows_across_cycles_not_across_runs(tmp_path):
    """Several catch-up slots fire per cycle; they must not inflate the count."""
    p = _audit_file(tmp_path, published=False, withheld_since_ct="2026-09-01")
    # a second run on the SAME CT day is still cycle 1
    assert ptb._withheld_streak(p, False, pd.Timestamp(CT_NOON))[1] == 1
    # the next CT day is cycle 2 — this is what escalates the job to red
    assert ptb._withheld_streak(p, False, pd.Timestamp("2026-09-02T17:00:00Z"))[1] == 2
    assert ptb._withheld_streak(p, False, pd.Timestamp("2026-09-04T17:00:00Z"))[1] == 4


def test_a_publish_resets_the_streak(tmp_path):
    p = _audit_file(tmp_path, published=False, withheld_since_ct="2026-08-28")
    assert ptb._withheld_streak(p, True, pd.Timestamp(CT_NOON)) == (None, 0)
    # and the next withhold starts a fresh streak from today
    p2 = _audit_file(tmp_path, published=True, withheld_since_ct="2026-08-28")
    assert ptb._withheld_streak(p2, False, pd.Timestamp(CT_NOON)) == ("2026-09-01", 1)


def test_streak_degrades_safely_on_a_missing_or_corrupt_trail(tmp_path):
    missing = tmp_path / "nope.json"
    assert ptb._withheld_streak(missing, False, pd.Timestamp(CT_NOON)) == ("2026-09-01", 1)
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert ptb._withheld_streak(bad, False, pd.Timestamp(CT_NOON)) == ("2026-09-01", 1)
    weird = _audit_file(tmp_path, published=False, withheld_since_ct="tomorrow-ish")
    assert ptb._withheld_streak(weird, False, pd.Timestamp(CT_NOON)) == ("2026-09-01", 1)


def test_a_future_dated_streak_never_reports_a_negative_count(tmp_path):
    p = _audit_file(tmp_path, published=False, withheld_since_ct="2026-12-25")
    since, days = ptb._withheld_streak(p, False, pd.Timestamp(CT_NOON))
    assert days == 1 and since == "2026-09-01"


def test_gate_decisions_are_attached_per_app(monkeypatch):
    """The miners sleeve reads the SAME dataset as GLDM, so its row must carry
    the gold dataset's verdict rather than none at all."""
    monkeypatch.setattr(ptb, "_gate_decisions", lambda: {
        "GLDM": dict(decision="fallback_snapshot", source="snapshot",
                     served_through="2026-08-28", qc_passed=False,
                     failed_checks=["no_missing_recent_sessions"], note=""),
        "SOXX": dict(decision="refreshed", source="yahoo-live",
                     served_through="2026-08-31", qc_passed=True,
                     failed_checks=[], note=""),
    })
    audit = {"rows": [{"app": "GLDM"}, {"app": "GDXM"}, {"app": "SOXX"},
                      {"app": "BTC"}]}
    rows = {r["app"]: r for r in ptb._attach_gate_decisions(audit)["rows"]}
    assert rows["GLDM"]["data_gate"]["decision"] == "fallback_snapshot"
    assert rows["GDXM"]["data_gate"]["decision"] == "fallback_snapshot"
    assert rows["SOXX"]["data_gate"]["decision"] == "refreshed"
    assert "data_gate" not in rows["BTC"]        # BTC has no gated dataset
