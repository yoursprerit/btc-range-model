"""Unit tests for ``daily_edge_by_sleeve`` (app/overall_core.py) — the
per-sleeve / per-parent-cluster split of the P&L section's headline
"Daily edge (invested)" metric.

Synthetic returns and daily-weight matrices only (no network, no Streamlit):
the additivity to the headline (sleeve edges + SATA = the invested-day
expectancy ``daily_win_stats`` reports on the same book), the cluster roll-up,
the invested-day denominator (fully-idle days are left out, exactly as the
headline leaves them out), the anchor-bar convention, the held-day win/loss
read, business-day-only SATA accrual and the degenerate cases.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import overall_core as oc  # noqa: E402


# Mon 2026-06-01 … Mon 2026-06-08: six weekdays plus one weekend
IDX = pd.date_range("2026-06-01", periods=8, freq="D")
KEYS = ["XLE", "XOP", "GLDM"]
CLUSTER = {"XLE": "XLE", "XOP": "XLE", "GLDM": "GLDM"}
SATA = 0.0002


def _book():
    rets = pd.DataFrame(
        [[0.010, 0.020, -0.005],     # anchor bar — must not count
         [0.020, -0.010, 0.004],
         [-0.015, 0.030, 0.000],
         [0.000, 0.000, 0.000],      # fully-idle day (weights all 0)
         [0.005, 0.010, -0.020],
         [0.012, -0.004, 0.006],     # Saturday
         [-0.008, 0.002, 0.001],     # Sunday
         [0.030, 0.015, -0.010]],
        index=IDX, columns=KEYS)
    w = pd.DataFrame(
        [[0.30, 0.10, 0.20],
         [0.30, 0.10, 0.20],
         [0.25, 0.00, 0.30],
         [0.00, 0.00, 0.00],
         [0.00, 0.00, 0.40],
         [0.20, 0.10, 0.00],
         [0.20, 0.10, 0.00],
         [0.20, 0.10, 0.10]],
        index=IDX, columns=KEYS)
    sata_w = 1.0 - w.sum(axis=1)
    return rets, w, sata_w


def test_sleeve_edges_plus_sata_sum_to_the_headline_expectancy():
    rets, w, sata_w = _book()
    d = oc.daily_edge_by_sleeve(rets, w, sata_w, IDX[0], cluster_of=CLUSTER,
                                sata_daily=SATA)
    parts = sum(v["edge"] for v in d["per_key"].values()) + d["sata"]["edge"]
    assert parts == pytest.approx(d["expectancy"], abs=1e-15)
    # …and that expectancy IS the metric the headline reports on the book's
    # own equity curve, conditioned on the same invested-day mask
    daily = oc.pnl_daily_replay(rets, w, sata_w, IDX[0], sata_daily=SATA)
    dw = oc.daily_win_stats(oc._equity(daily["ret"]), IDX[0],
                            active=w.sum(axis=1))
    assert d["expectancy"] == pytest.approx(dw["expectancy"], abs=1e-15)
    assert d["n_active"] == dw["n_active"] == 6      # 7 post-anchor bars − 1 idle


def test_cluster_edge_is_the_sum_of_its_members():
    rets, w, sata_w = _book()
    d = oc.daily_edge_by_sleeve(rets, w, sata_w, IDX[0], cluster_of=CLUSTER,
                                sata_daily=SATA)
    xle = d["per_cluster"]["XLE"]
    assert sorted(xle["members"]) == ["XLE", "XOP"]
    assert xle["edge"] == pytest.approx(
        d["per_key"]["XLE"]["edge"] + d["per_key"]["XOP"]["edge"], abs=1e-15)
    assert xle["total"] == pytest.approx(
        d["per_key"]["XLE"]["total"] + d["per_key"]["XOP"]["total"], abs=1e-15)
    assert d["per_cluster"]["GLDM"]["members"] == ["GLDM"]
    assert d["per_cluster"]["GLDM"]["edge"] == d["per_key"]["GLDM"]["edge"]
    clusters = sum(v["edge"] for v in d["per_cluster"].values()) + d["sata"]["edge"]
    assert clusters == pytest.approx(d["expectancy"], abs=1e-15)


def test_cluster_held_days_count_any_member_with_weight():
    rets, w, sata_w = _book()
    d = oc.daily_edge_by_sleeve(rets, w, sata_w, IDX[0], cluster_of=CLUSTER,
                                sata_daily=SATA)
    # XOP is flat on Jun 03; XLE still carries the cluster that day, and on
    # Jun 05 neither does — 5 of the 6 invested days
    assert d["per_key"]["XOP"]["n_held"] == 4
    assert d["per_key"]["XLE"]["n_held"] == 5
    assert d["per_cluster"]["XLE"]["n_held"] == 5


def test_edge_averages_over_every_invested_day_but_held_read_over_held_days():
    rets, w, sata_w = _book()
    d = oc.daily_edge_by_sleeve(rets, w, sata_w, IDX[0], cluster_of=CLUSTER,
                                sata_daily=SATA)
    g = d["per_key"]["GLDM"]
    # GLDM's post-anchor contributions on invested days: Jun 02 .2×.004,
    # Jun 03 .3×0, Jun 05 .4×−.02, Jun 06/07 0 (no weight), Jun 08 .1×−.01
    contrib = [0.0008, 0.0, -0.008, 0.0, 0.0, -0.001]
    assert g["total"] == pytest.approx(sum(contrib), abs=1e-15)
    assert g["edge"] == pytest.approx(sum(contrib) / 6, abs=1e-15)
    assert g["n_held"] == 4                      # Jun 02, 03, 05, 08
    assert g["edge_held"] == pytest.approx(sum(contrib) / 4, abs=1e-15)
    assert g["wins"] == 1 and g["win_rate"] == pytest.approx(0.25)
    assert g["avg_win"] == pytest.approx(0.0008, abs=1e-15)
    assert g["avg_loss"] == pytest.approx(-0.0045, abs=1e-15)


def test_anchor_bar_is_the_cost_basis_and_does_not_count():
    rets, w, sata_w = _book()
    d = oc.daily_edge_by_sleeve(rets, w, sata_w, IDX[0], sata_daily=0.0)
    # a start on the anchor bar vs one bar later: the anchor's own (large)
    # returns never enter, so XOP's total is unchanged when the window opens
    # on Jun 01 or Jun 02 … but moving the start to Jun 02 makes Jun 02 the
    # new anchor and drops its contribution
    later = oc.daily_edge_by_sleeve(rets, w, sata_w, IDX[1], sata_daily=0.0)
    assert d["per_key"]["XOP"]["total"] == pytest.approx(
        later["per_key"]["XOP"]["total"] + 0.10 * -0.010, abs=1e-15)


def test_sata_accrues_on_business_days_only_and_counts_cash_held():
    rets, w, sata_w = _book()
    d = oc.daily_edge_by_sleeve(rets, w, sata_w, IDX[0], sata_daily=SATA)
    # invested days: Jun 02 (cash .4), 03 (.45), 05 (.6), 06/07 weekend (0),
    # 08 (.6)
    expect = SATA * (0.4 + 0.45 + 0.6 + 0.6)
    assert d["sata"]["total"] == pytest.approx(expect, abs=1e-15)
    assert d["sata"]["n_held"] == 4
    assert d["sata"]["edge"] == pytest.approx(expect / 6, abs=1e-15)


def test_no_sata_when_the_record_holds_plain_cash():
    rets, w, sata_w = _book()
    d = oc.daily_edge_by_sleeve(rets, w, sata_w, IDX[0], sata_daily=0.0)
    assert d["sata"]["edge"] == 0.0 and d["sata"]["n_held"] == 0
    assert sum(v["edge"] for v in d["per_key"].values()) == pytest.approx(
        d["expectancy"], abs=1e-15)


def test_a_key_outside_the_cluster_map_is_its_own_cluster():
    rets, w, sata_w = _book()
    d = oc.daily_edge_by_sleeve(rets, w, sata_w, IDX[0],
                                cluster_of={"XOP": "XLE"}, sata_daily=SATA)
    assert set(d["per_cluster"]) == {"XLE", "GLDM"}
    assert d["per_cluster"]["XLE"]["members"] == ["XLE", "XOP"]


def test_window_end_is_whatever_the_frames_were_cut_to():
    rets, w, sata_w = _book()
    cut = IDX[4]
    d = oc.daily_edge_by_sleeve(rets.loc[:cut], w.loc[:cut], sata_w.loc[:cut],
                                IDX[0], sata_daily=SATA)
    assert d["end"] == cut and d["n_active"] == 3


def test_degenerate_windows_return_none():
    rets, w, sata_w = _book()
    assert oc.daily_edge_by_sleeve(rets.iloc[:1], w, sata_w, IDX[0]) is None
    idle = w * 0.0
    assert oc.daily_edge_by_sleeve(rets, idle, 1.0 - idle.sum(axis=1),
                                   IDX[0], sata_daily=SATA) is None


def test_never_held_sleeve_has_zero_edge_and_no_held_read():
    rets, w, sata_w = _book()
    w = w.copy(); w["GLDM"] = 0.0
    d = oc.daily_edge_by_sleeve(rets, w, 1.0 - w.sum(axis=1), IDX[0],
                                sata_daily=SATA)
    g = d["per_key"]["GLDM"]
    assert g["edge"] == 0.0 and g["n_held"] == 0
    assert g["edge_held"] is None and g["win_rate"] is None
