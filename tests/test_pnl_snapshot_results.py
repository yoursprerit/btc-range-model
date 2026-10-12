"""The CI-built snapshot also serves the V1/V2 universe results to the app."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import pnl_snapshot as ps  # noqa: E402

_VIN = dict(btc_sha="a", btc_to="2026-10-12", sleeves_sha="b", n_sleeves=8)


def _snap(**kw):
    return dict(results={"v1": [{"key": "A"}], "v2": [{"key": "B"}]},
                vintage_in=dict(_VIN), built_at="2026-10-12T20:00:00+00:00", **kw)


def test_served_when_vintage_matches(monkeypatch):
    monkeypatch.setattr(ps.ov, "data_vintage", lambda: dict(_VIN))
    hit = ps.get_results(_snap(), "V2")
    assert hit["results"] == [{"key": "B"}]
    assert hit["computed_at"] == pd.Timestamp("2026-10-12T20:00:00+00:00")


def test_rejected_when_data_changed_since_build(monkeypatch):
    monkeypatch.setattr(ps.ov, "data_vintage", lambda: dict(_VIN, sleeves_sha="c"))
    assert ps.get_results(_snap(), "v2") is None


def test_absent_snapshot_unknown_view_or_old_schema(monkeypatch):
    monkeypatch.setattr(ps.ov, "data_vintage", lambda: dict(_VIN))
    assert ps.get_results(None, "v2") is None
    assert ps.get_results(_snap(), "combined") is None      # the app splices v1+v2
    old = dict(vintage_in=dict(_VIN), built_at="2026-10-12T20:00:00+00:00")
    assert ps.get_results(old, "v2") is None                # pre-results snapshot
