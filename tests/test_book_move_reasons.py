"""Unit tests for overall_core.book_move_reasons.

Covers the "Previous → Current Targetbook" explainer added to the Overall
cockpit's Rebalancing-moves section: each whole-point weight move between the
two *published* books must carry a reason derived from the payloads' recorded
actions and priority scores (entry fired / signal exited / retired / cap-bound
/ priority re-tilt / water-fill redistribution / SATA residual).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import overall_core as oc  # noqa: E402


def _book(weights, cash=0.0, actions=None):
    return {"schema": "overall-target-book/v1", "weights": dict(weights),
            "cash_weight": cash, "actions": actions}


def _by_key(moves):
    return {m["key"]: m for m in moves}


def test_identical_books_produce_no_moves():
    b = _book({"SOXX": 0.30, "XLE": 0.20}, cash=0.50)
    assert oc.book_move_reasons(b, b) == []


def test_fresh_entry_cites_the_decision_and_priority():
    prev = _book({}, cash=1.0, actions=[])
    cur = _book({"GDX": 0.20}, cash=0.80, actions=[
        {"key": "GDX", "action": "OPEN", "decision": "LONG — ENTER",
         "in_pos": False, "priority": 0.7}])
    got = _by_key(oc.book_move_reasons(prev, cur))
    assert (got["GDX"]["from_pct"], got["GDX"]["to_pct"]) == (0, 20)
    assert "fresh entry" in got["GDX"]["reason"]
    assert "LONG — ENTER" in got["GDX"]["reason"]
    assert "0.70" in got["GDX"]["reason"]
    # the funding came out of idle cash — SATA row explains the residual
    assert got["SATA"]["delta_pct"] == -20
    assert "idle cash" in got["SATA"]["reason"]


def test_exit_cites_the_current_books_recorded_decision():
    prev = _book({"GDX": 0.20}, cash=0.80, actions=[
        {"key": "GDX", "action": "HOLD", "decision": "LONG — HOLDING",
         "in_pos": True, "priority": 0.6}])
    cur = _book({}, cash=1.0, actions=[
        {"key": "GDX", "action": "STAND ASIDE",
         "decision": "STAND ASIDE — EXIT ACTIVE (D2 momentum fade)",
         "in_pos": False, "priority": None}])
    got = _by_key(oc.book_move_reasons(prev, cur))
    assert got["GDX"]["delta_pct"] == -20
    assert "signal exited" in got["GDX"]["reason"]
    assert "EXIT ACTIVE (D2 momentum fade)" in got["GDX"]["reason"]
    assert got["SATA"]["delta_pct"] == 20
    assert "idle cash" in got["SATA"]["reason"]


def test_instrument_missing_from_current_universe_reads_as_retired():
    # an ETHA-era previous book after the ETH swap: the current book's actions
    # list no longer knows the key at all
    prev = _book({"ETHA": 0.10}, cash=0.90)
    cur = _book({}, cash=1.0, actions=[
        {"key": "ETH", "action": "STAND ASIDE", "decision": "FLAT — NO SIGNAL",
         "in_pos": False, "priority": None}])
    got = _by_key(oc.book_move_reasons(prev, cur))
    assert "retired" in got["ETHA"]["reason"]


def test_weight_filled_to_its_cap_is_reported_as_cap_bound():
    prev = _book({"SOXX": 0.25}, cash=0.75)
    cur = _book({"SOXX": 0.30}, cash=0.70)
    got = _by_key(oc.book_move_reasons(prev, cur, caps={"SOXX": 0.30}))
    assert "30% cap" in got["SOXX"]["reason"]


def test_resize_agreeing_with_own_priority_cites_the_tilt():
    prev = _book({"XLE": 0.18}, cash=0.82, actions=[
        {"key": "XLE", "action": "HOLD", "decision": "LONG — HOLDING",
         "in_pos": True, "priority": 0.60}])
    cur = _book({"XLE": 0.22}, cash=0.78, actions=[
        {"key": "XLE", "action": "HOLD", "decision": "LONG — HOLDING",
         "in_pos": True, "priority": 0.85}])
    got = _by_key(oc.book_move_reasons(prev, cur, caps={"XLE": 0.30}))
    assert "rose" in got["XLE"]["reason"]
    assert "0.60 → 0.85" in got["XLE"]["reason"]


def test_resize_against_own_priority_is_attributed_to_the_waterfill():
    # weight rose although its own priority fell — the competing names' tilts
    # must have freed the weight, so the reason blames the redistribution
    prev = _book({"REMX": 0.05, "XLE": 0.25}, cash=0.70, actions=[
        {"key": "REMX", "action": "HOLD", "decision": "LONG — HOLDING",
         "in_pos": True, "priority": 0.30},
        {"key": "XLE", "action": "HOLD", "decision": "LONG — HOLDING",
         "in_pos": True, "priority": 0.90}])
    cur = _book({"REMX": 0.10, "XLE": 0.20}, cash=0.70, actions=[
        {"key": "REMX", "action": "HOLD", "decision": "LONG — HOLDING",
         "in_pos": True, "priority": 0.28},
        {"key": "XLE", "action": "HOLD", "decision": "LONG — HOLDING",
         "in_pos": True, "priority": 0.70}])
    got = _by_key(oc.book_move_reasons(prev, cur,
                                       caps={"REMX": 0.30, "XLE": 0.30}))
    assert "water-fill" in got["REMX"]["reason"]
    assert "into" in got["REMX"]["reason"]
    # XLE's cut agrees with its own falling priority
    assert "fell" in got["XLE"]["reason"]


def test_legacy_books_without_actions_degrade_gracefully():
    prev = _book({"SOXX": 0.10}, cash=0.90)
    cur = _book({"SOXX": 0.20, "OIH": 0.15}, cash=0.65)
    got = _by_key(oc.book_move_reasons(prev, cur,
                                       caps={"SOXX": 0.30, "OIH": 0.30}))
    assert "reason" in got["SOXX"] and got["SOXX"]["reason"]
    assert "fresh entry" in got["OIH"]["reason"]


def test_sata_weight_inside_the_weights_dict_counts_as_idle():
    # LIVE books park idle capital as a SATA weight, PAPER books as cash_weight
    prev = _book({"SOXX": 0.30, "SATA": 0.70})
    cur = _book({"SOXX": 0.30}, cash=0.70)
    assert oc.book_move_reasons(prev, cur) == []
