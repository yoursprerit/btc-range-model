"""The P&L section's benchmark table and growth chart must describe the same
measurement, and must both respond to the date pickers.

Both are driven by one dict of equity curves, each sliced at the chosen start
and re-based at ``sub.iloc[0]``.  That is exact arithmetic, but it quietly
changes MEANING when a curve's history begins after the chosen start: the
curve is anchored at its own first bar instead, so its row stops responding to
the start-date picker while every other row moves, and its line enters the
chart mid-plot at break-even.  The as-published record hits this constantly —
it begins the day the current strategy version was stamped.

These tests pin the arithmetic (table == chart, always) and the detection of
the short-history case that the UI must disclose.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
import overall_core as ov  # noqa: E402

IDX = pd.bdate_range("2026-01-01", periods=160)


def _curve(seed, n=160, start=0):
    rng = np.random.default_rng(seed)
    r = pd.Series(rng.normal(0.0015, 0.01, n - start), index=IDX[start:])
    return (1 + r).cumprod()


def _table_value(curve, start, end):
    """What the benchmark table prints for one curve."""
    return ov.slice_metrics(curve.loc[:pd.Timestamp(end)], start)


def _chart_value(curve, start, end):
    """Where that curve's line ends on the growth chart, as a return."""
    sub = curve.loc[:pd.Timestamp(end)].loc[pd.Timestamp(start):]
    if len(sub) < 2:
        return None
    return float((sub / sub.iloc[0]).iloc[-1] - 1)


def test_table_and_chart_agree_on_every_curve_and_window():
    curves = {"full": _curve(1), "other": _curve(2), "late": _curve(3, start=100)}
    windows = [(IDX[0], IDX[-1]), (IDX[20], IDX[80]), (IDX[110], IDX[-1])]
    for start, end in windows:
        for name, cv in curves.items():
            m = _table_value(cv, start, end)
            c = _chart_value(cv, start, end)
            if m is None or c is None:
                continue
            assert abs(m["total_ret"] - c) < 1e-12, (name, start, end)


def test_a_full_history_curve_tracks_the_start_date():
    cv = _curve(1)
    a = _table_value(cv, IDX[0], IDX[-1])["total_ret"]
    b = _table_value(cv, IDX[40], IDX[-1])["total_ret"]
    assert a != b, "moving the start date must change a full-history row"


def test_a_short_history_curve_stops_tracking_the_start_date():
    # the bug the UI has to disclose: this row is identical for two very
    # different windows, because both anchor at the curve's own first bar
    late = _curve(3, start=100)
    a = _table_value(late, IDX[0], IDX[-1])
    b = _table_value(late, IDX[50], IDX[-1])
    assert a["total_ret"] == b["total_ret"]
    assert a["start"] == b["start"] == IDX[100]


def test_short_history_is_detectable_by_the_rule_the_ui_uses():
    # the UI marks a row when its anchor is later than the window's anchor
    full, late = _curve(1), _curve(3, start=100)
    start, end = IDX[0], IDX[-1]
    window = _table_value(full, start, end)          # the selected source
    assert _table_value(full, start, end)["start"] == window["start"]
    assert _table_value(late, start, end)["start"] > window["start"]


def test_a_start_inside_every_curve_makes_them_comparable():
    full, late = _curve(1), _curve(3, start=100)
    start, end = IDX[120], IDX[-1]
    assert _table_value(full, start, end)["start"] == \
           _table_value(late, start, end)["start"] == IDX[120]


def test_end_date_moves_every_row():
    cv = _curve(1)
    a = _table_value(cv, IDX[0], IDX[80])["total_ret"]
    b = _table_value(cv, IDX[0], IDX[-1])["total_ret"]
    assert a != b, "moving the end date must change the measurement"
