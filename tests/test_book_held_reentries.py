"""A sleeve the published book HOLDS that the engine shows flat with a live buy
(action OPEN) is an exit + immediate re-entry — e.g. GLDM/UGL's −3% stop firing
at the close while the dual-MA trend is still long.  It must not be announced
as a "new committed entry" the book doesn't carry (observed 2026-09-28)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import overall_core as oc  # noqa: E402


def _book(*rows):
    return dict(actions=list(rows))


def _brow(key, action, in_pos):
    return dict(key=key, action=action, in_pos=in_pos, target=0.1)


def _arow(key, action, in_pos):
    return dict(key=key, action=action, in_pos=in_pos)


def test_held_sleeve_reopening_is_a_reentry():
    book = _book(_brow("GLDM", "HOLD", True), _brow("UGL", "HOLD", True))
    gate = dict(actions=[_arow("UGL", "OPEN", False), _arow("GLDM", "OPEN", False)])
    assert oc.book_held_reentries(book, gate) == ["GLDM", "UGL"]


def test_genuinely_new_entries_are_not_reentries():
    book = _book(_brow("ARTY", "WATCH", False), _brow("PBW", "STAND ASIDE", False),
                 _brow("XLE", "OPEN", False))
    gate = dict(actions=[_arow("ARTY", "OPEN", False), _arow("PBW", "OPEN", False),
                         _arow("XLE", "OPEN", False), _arow("NEW", "OPEN", False)])
    assert oc.book_held_reentries(book, gate) == []


def test_still_held_or_closing_rows_are_ignored():
    book = _book(_brow("GLDM", "HOLD", True), _brow("OIH", "CLOSE", True))
    gate = dict(actions=[_arow("GLDM", "HOLD", True), _arow("OIH", "OPEN", False)])
    assert oc.book_held_reentries(book, gate) == []


def test_no_book():
    assert oc.book_held_reentries(None, dict(actions=[_arow("GLDM", "OPEN", False)])) == []
