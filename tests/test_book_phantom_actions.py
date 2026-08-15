"""Published-book instructions today's engine cannot reproduce.

The Overall action plan pins its Action / Signal / Priority cells to the
*published* Current Targetbook so they match the donut and never drift
intraday.  Existing flags cover the book running BEHIND the engine (a signal
that committed at a close after the morning publish).  These tests pin the
other direction: a book row the engine cannot derive from the SAME as-of bar,
which is what a publish-time data vintage the pinned snapshot no longer matches
produces — the ARTY 2026-08-14 case, published ``CLOSE`` / ``EXIT NEXT BAR —
D3 exhaustion`` while every live surface read ``LONG — HOLDING`` with no D3.

The states the ordinary publish→execute lifecycle DOES produce must stay
unflagged, or the plan cries wolf every morning.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import overall_core as oc  # noqa: E402


def _book(*rows):
    return dict(schema="overall-target-book/v1", as_of="2026-08-14",
                actions=list(rows))


def _brow(key, action, decision="", exits_next_bar=False, in_pos=True,
          target=0.0):
    return dict(key=key, action=action, decision=decision, target=target,
                in_pos=in_pos, priority=None, exits_next_bar=exits_next_bar)


def _gate(*rows):
    return dict(actions=list(rows))


def _arow(key, action, in_pos, decision="LONG — HOLDING", exits_next_bar=False,
          state=None):
    a = dict(key=key, action=action, in_pos=in_pos, decision=decision,
             state=state or ("HOLD" if in_pos else "FLAT"))
    if exits_next_bar:
        a["exits_next_bar"] = True
    return a


# ── the ARTY case ────────────────────────────────────────────────────────
def test_phantom_exit_when_book_closes_a_position_the_engine_holds():
    book = _book(_brow("ARTY", "CLOSE", "EXIT NEXT BAR — D3 exhaustion",
                       exits_next_bar=True))
    gate = _gate(_arow("ARTY", "HOLD", True))
    out = oc.book_phantom_actions(book, gate)
    assert out == dict(exits=["ARTY"], entries=[])


def test_phantom_exit_detected_from_exits_next_bar_alone():
    """Legacy books recorded HOLD next to a zero weight + exits_next_bar; the
    plan decodes that as a CLOSE instruction, so it counts here too."""
    book = _book(_brow("ARTY", "HOLD", "EXIT NEXT BAR — D3 exhaustion",
                       exits_next_bar=True))
    out = oc.book_phantom_actions(book, _gate(_arow("ARTY", "HOLD", True)))
    assert out["exits"] == ["ARTY"]


def test_phantom_entry_when_book_opens_a_sleeve_still_flat():
    book = _book(_brow("PBW", "OPEN", "ENTER — PURE-REGIME BUY",
                       in_pos=False, target=0.2))
    gate = _gate(_arow("PBW", "STAND ASIDE", False, "FLAT — NO SIGNAL"))
    out = oc.book_phantom_actions(book, gate)
    assert out == dict(exits=[], entries=["PBW"])


# ── everything the normal lifecycle produces stays silent ────────────────
def test_agreeing_book_and_engine_are_not_flagged():
    book = _book(_brow("SOXX", "HOLD", "LONG — HOLDING", target=0.3),
                 _brow("GDX", "CLOSE", "EXIT NEXT BAR — D2 momentum fade",
                       exits_next_bar=True))
    gate = _gate(_arow("SOXX", "HOLD", True),
                 _arow("GDX", "CLOSE", True, "EXIT NEXT BAR — D2 momentum fade",
                       exits_next_bar=True))
    assert oc.book_phantom_actions(book, gate) == dict(exits=[], entries=[])


def test_book_behind_the_engine_is_not_a_phantom():
    """The publish-lag direction — book HOLD, exit committed at a later close —
    is the masked-exit flag's job, not this one."""
    book = _book(_brow("GDX", "HOLD", "LONG — HOLDING", target=0.3))
    gate = _gate(_arow("GDX", "CLOSE", True, "EXIT NEXT BAR — D2 momentum fade",
                       exits_next_bar=True))
    assert oc.book_phantom_actions(book, gate) == dict(exits=[], entries=[])


def test_executed_book_exit_is_not_a_phantom():
    """Once the book's CLOSE has been executed the sim is flat — the row must
    not keep warning that the exit 'never existed'."""
    book = _book(_brow("ARTY", "CLOSE", "EXIT NEXT BAR — D3 exhaustion",
                       exits_next_bar=True))
    gate = _gate(_arow("ARTY", "STAND ASIDE", False, "FLAT — NO SIGNAL"))
    assert oc.book_phantom_actions(book, gate)["exits"] == []


def test_executed_book_entry_is_not_a_phantom():
    book = _book(_brow("PBW", "OPEN", "ENTER — PURE-REGIME BUY",
                       in_pos=False, target=0.2))
    gate = _gate(_arow("PBW", "HOLD", True))
    assert oc.book_phantom_actions(book, gate)["entries"] == []


def test_live_driven_exit_or_entry_is_not_a_phantom():
    """A live-price exit/entry IS today's engine agreeing with the book on the
    intraday read — flagged elsewhere in red/green, never as a mismatch."""
    book = _book(_brow("ARTY", "CLOSE", "EXIT NEXT BAR — D3 exhaustion",
                       exits_next_bar=True),
                 _brow("PBW", "OPEN", "ENTER — PURE-REGIME BUY", in_pos=False))
    gate = _gate(_arow("ARTY", "HOLD", True),
                 _arow("PBW", "STAND ASIDE", False, "FLAT — NO SIGNAL"))
    out = oc.book_phantom_actions(book, gate, live_exits={"ARTY"},
                                  live_entries={"PBW"})
    assert out == dict(exits=[], entries=[])


def test_unpublished_or_new_instrument_is_never_flagged():
    gate = _gate(_arow("ARTY", "HOLD", True))
    assert oc.book_phantom_actions(None, gate) == dict(exits=[], entries=[])
    assert oc.book_phantom_actions({}, gate) == dict(exits=[], entries=[])
    # instrument added to the universe after the last publish — no book row
    book = _book(_brow("SOXX", "HOLD", "LONG — HOLDING", target=0.3))
    assert oc.book_phantom_actions(book, gate) == dict(exits=[], entries=[])
