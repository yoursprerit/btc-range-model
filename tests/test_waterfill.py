"""Unit tests for overall_core's cap-aware water-fill (_cap_normalise/_waterfill).

Regression cover for a bug found 2026-07-25 via the published Target Book of
Jul 24: when redistributing the overflow from capped names, a recipient that
hit ITS OWN cap mid-redistribution had the excess silently dropped (the old
``np.minimum`` clip), and the loop then exited because nothing was strictly
above cap.  Result: the published live book parked 13.3% in SATA while XLE
(18.4%) and REMX (0.26%) both sat far below their 30% caps — capital the
design says belonged in the sleeves with room.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import overall_core as oc  # noqa: E402


def _fill(raw: dict, caps: dict) -> dict:
    return oc._waterfill(raw, caps)


def test_spill_recycles_when_a_recipient_hits_its_own_cap():
    # A's overflow spills onto B and C; B clips at its small cap while C still
    # has room.  The old code dropped B's clipped share (leaving ~47% idle);
    # the excess must instead land on C.
    raw = {"A": 0.60, "B": 0.09, "C": 0.02}
    caps = {"A": 0.30, "B": 0.10, "C": 0.70}
    w = _fill(raw, caps)
    assert abs(sum(w.values()) - 1.0) < 1e-9          # caps sum 1.10 → feasible
    assert w["A"] == 0.30 and w["B"] == 0.10
    assert abs(w["C"] - 0.60) < 1e-9


def test_jul24_published_book_shape_deploys_fully():
    # The Jul 24, 2026 live book: four sleeves pinned at their Balanced caps,
    # XLE/REMX far below theirs — yet 13.3% was left in SATA.  With the fix the
    # same shape (big tilted anchors overflowing onto two small-weight names)
    # deploys 100%, since the caps sum to 1.28.
    raw = {"SOXX": 0.45, "SOXL": 0.13, "XLE": 0.15, "OIH": 0.17,
           "ERX": 0.09, "REMX": 0.01}
    caps = {"SOXX": 0.30, "SOXL": 0.10, "XLE": 0.30, "OIH": 0.18,
            "ERX": 0.10, "REMX": 0.30}
    w = _fill(raw, caps)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    for k in w:
        assert w[k] <= caps[k] + 1e-9
    # the recycled spill goes to the names with room, biggest raw weight first
    assert w["XLE"] > 0.15 and w["REMX"] > 0.01


def test_infeasible_caps_still_leave_the_remainder_to_cash():
    # n·cap < 1: nothing can absorb the book — everyone at cap, rest to cash.
    w = _fill({"A": 0.5, "B": 0.5}, {"A": 0.30, "B": 0.10})
    assert w["A"] == 0.30 and w["B"] == 0.10
    assert abs(sum(w.values()) - 0.40) < 1e-9


def test_no_cap_binding_is_a_plain_normalise():
    w = _fill({"A": 0.2, "B": 0.1, "C": 0.1}, {"A": 0.9, "B": 0.9, "C": 0.9})
    assert abs(w["A"] - 0.50) < 1e-9
    assert abs(w["B"] - 0.25) < 1e-9
    assert abs(w["C"] - 0.25) < 1e-9


def test_cap_normalise_converges_for_long_cap_chains():
    # a descending cap ladder forces the spill through many re-cap passes; the
    # n+1 pass bound must still fully deploy (caps sum comfortably > 1)
    n = 12
    raw = np.array([2.0 ** -i for i in range(n)])
    caps = np.array([0.09 + 0.01 * i for i in range(n)])
    assert caps.sum() > 1
    w = oc._cap_normalise(raw, caps)
    assert abs(w.sum() - 1.0) < 1e-9
    assert (w <= caps + 1e-9).all()
