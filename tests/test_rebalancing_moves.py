"""Unit tests for overall_core.rebalancing_moves.

Regression cover for two bugs in the Overall cockpit's "Rebalancing moves" line
(reported 2026-07-25 from the deployed app):
  1. the arrow was computed from RAW weights while the endpoints were rounded, so
     rows did not add up (17% → 10% shown as ▼8pt, 18% → 22% as ▲5pt);
  2. the "from" side was the engine's synthetic current book instead of the
     published Target Book, so it contradicted the donut above it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import overall_core as oc  # noqa: E402


def _by_key(moves):
    return {m["key"]: m for m in moves}


def test_delta_is_always_the_difference_of_the_displayed_endpoints():
    # the exact values behind the reported line: 17.4%→9.9% and 17.6%→22.4%
    moves = oc.rebalancing_moves({"REMX": 0.174, "XLE": 0.176}, 0.0,
                                 {"REMX": 0.099, "XLE": 0.224}, 0.0)
    for m in moves:
        assert m["delta_pct"] == m["to_pct"] - m["from_pct"], m
    got = _by_key(moves)
    assert (got["REMX"]["from_pct"], got["REMX"]["to_pct"], got["REMX"]["delta_pct"]) == (17, 10, -7)
    assert (got["XLE"]["from_pct"], got["XLE"]["to_pct"], got["XLE"]["delta_pct"]) == (18, 22, 4)


def test_moves_that_round_to_zero_points_are_omitted():
    # 18.0% → 18.4% is a real but sub-point move: rendering "18% → 18% ▲0pt"
    # would be noise, so the row is dropped.
    assert oc.rebalancing_moves({"OIH": 0.180}, 0.0, {"OIH": 0.184}, 0.0) == []
    # an exactly unchanged weight is likewise absent
    assert oc.rebalancing_moves({"SOXX": 0.30}, 0.0, {"SOXX": 0.30}, 0.0) == []


def test_retired_instrument_is_kept_so_it_can_be_sold_down():
    # a book published before a universe change still names ETHA; the move must
    # survive (sell to zero) rather than vanish.
    moves = _by_key(oc.rebalancing_moves({"ETHA": 0.05, "SOXX": 0.30}, 0.0,
                                         {"SOXX": 0.30, "ETH": 0.05}, 0.0))
    assert moves["ETHA"]["to_pct"] == 0 and moves["ETHA"]["delta_pct"] == -5
    assert moves["ETH"]["from_pct"] == 0 and moves["ETH"]["delta_pct"] == 5
    assert "SOXX" not in moves                      # unchanged → omitted


def test_idle_cash_row_is_last_and_uses_the_idle_key():
    moves = oc.rebalancing_moves({"SOXX": 0.30}, 0.133, {"SOXX": 0.40}, 0.0)
    assert moves[-1]["key"] == "SATA"
    assert (moves[-1]["from_pct"], moves[-1]["to_pct"], moves[-1]["delta_pct"]) == (13, 0, -13)
    assert moves[0]["key"] == "SOXX"


def test_ordering_is_by_target_weight_and_empty_inputs_are_safe():
    moves = oc.rebalancing_moves({}, 0.0, {"A": 0.10, "B": 0.30, "C": 0.20}, 0.0)
    assert [m["key"] for m in moves] == ["B", "C", "A"]
    assert oc.rebalancing_moves({}, 0.0, {}, 0.0) == []
    assert oc.rebalancing_moves({}, None, {}, None) == []


def test_reproduces_the_published_book_vs_live_target_case():
    # published Current Targetbook (2026-07-25) vs that morning's live target
    base = {"SOXX": 0.30, "XLE": 0.184, "OIH": 0.180, "SOXL": 0.10,
            "ERX": 0.10, "REMX": 0.0026}
    target = {"SOXX": 0.30, "XLE": 0.224, "OIH": 0.180, "SOXL": 0.10,
              "ERX": 0.10, "REMX": 0.096}
    got = _by_key(oc.rebalancing_moves(base, 0.1334, target, 0.0))
    assert got["XLE"]["delta_pct"] == 4          # 18 → 22, not the old ▲5pt
    assert (got["REMX"]["from_pct"], got["REMX"]["to_pct"]) == (0, 10)   # not 17%
    assert got["SATA"]["delta_pct"] == -13
    assert "OIH" not in got and "SOXX" not in got                 # unchanged
    for m in got.values():
        assert m["delta_pct"] == m["to_pct"] - m["from_pct"]
