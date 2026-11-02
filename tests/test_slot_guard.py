"""DST-proof 2:20 PM CT slot: the guard window, the trigger set, and the preflight task audit."""
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import slot_guard as sg  # noqa: E402
import preflight_option_c as pf  # noqa: E402


def utc(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


# --- the guard window: 14:20 America/Chicago on both sides of DST ------------------
@pytest.mark.parametrize("now,expected", [
    ("2026-10-30T19:20", True),    # CDT: 14:20 CT
    ("2026-10-30T20:20", False),   # CDT: 15:20 CT -- the second (winter) trigger is skipped
    ("2026-11-02T19:20", False),   # CST: 13:20 CT -- the 2026-11-02 bug, now refused
    ("2026-11-02T20:20", True),    # CST: 14:20 CT
    ("2026-11-02T20:21", True),    # a slightly late fire still counts
    ("2026-11-02T20:50", True),    # +30 min edge
    ("2026-11-02T20:51", False),
    ("2026-03-09T19:20", True),    # first CDT Monday after spring-forward
    ("2026-03-09T20:20", False),
    ("2026-03-06T19:20", False),   # CST Friday before: 13:20 CT
    ("2026-03-06T20:20", True),
])
def test_in_slot_on_both_sides_of_dst(now, expected):
    assert sg.in_slot(utc(now), "14:20")[0] is expected


def test_exactly_one_trigger_fires_per_day_all_year():
    """With triggers at 19:20 and 20:20 UTC, exactly one passes on every day of the year."""
    import datetime as dt
    d0 = dt.date(2026, 1, 1)
    for i in range(365):
        d = d0 + dt.timedelta(days=i)
        hits = [h for h in (19, 20) if sg.in_slot(
            datetime(d.year, d.month, d.day, h, 20, tzinfo=timezone.utc), "14:20")[0]]
        assert len(hits) == 1, d


# --- trigger times to register -----------------------------------------------------
def _in_zone(tz, fn):
    old = os.environ.get("TZ")
    os.environ["TZ"] = tz
    time.tzset()
    try:
        return fn()
    finally:
        if old is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = old
        time.tzset()


def test_central_pc_needs_a_single_local_trigger():
    assert _in_zone("America/Chicago", lambda: sg.local_trigger_times("14:20", year=2026)) == ["14:20"]


def test_non_dst_pc_gets_both_local_times():
    assert _in_zone("America/Phoenix", lambda: sg.local_trigger_times("14:20", year=2026)) == ["12:20", "13:20"]


def test_eastern_pc_is_one_local_trigger_too():
    assert _in_zone("America/New_York", lambda: sg.local_trigger_times("14:20", year=2026)) == ["15:20"]


def test_cli_exit_codes(capsys):
    assert sg.main(["--slot", "14:20", "--now", "2026-11-02T20:20"]) == 0
    assert sg.main(["--slot", "14:20", "--now", "2026-11-02T19:20"]) == 10
    assert sg.main(["--slot", "25:99"]) == 2
    capsys.readouterr()


# --- preflight DST audit of the registered task ------------------------------------
def _xml(boundaries, args='-File x.ps1 -SlotCT "14:20"'):
    trig = "".join(f"<CalendarTrigger><StartBoundary>{b}</StartBoundary>"
                   f"<ScheduleByDay><DaysInterval>1</DaysInterval></ScheduleByDay></CalendarTrigger>"
                   for b in boundaries)
    return ('<?xml version="1.0" encoding="UTF-16"?>'
            '<Task xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">'
            f"<Triggers>{trig}</Triggers><Actions><Exec><Command>powershell.exe</Command>"
            f"<Arguments>{args}</Arguments></Exec></Actions></Task>")


CHI = ZoneInfo("America/Chicago")


def test_utc_pinned_trigger_is_flagged_as_the_nov_2_bug():
    st, detail = pf.analyze_task_triggers(_xml(["2026-08-17T19:20:00Z"]), "14:20", CHI)
    assert st == "FAIL" and "winter" in detail and "UTC-PINNED" in detail


def test_pinned_offset_form_is_flagged_too():
    st, _ = pf.analyze_task_triggers(_xml(["2026-08-17T14:20:00-05:00"]), "14:20", CHI)
    assert st == "FAIL"


def test_local_trigger_on_a_central_pc_passes_even_without_the_guard():
    st, _ = pf.analyze_task_triggers(_xml(["2026-08-17T14:20:00"], args="-File x.ps1"), "14:20", CHI)
    assert st == "PASS"


def test_phoenix_pc_with_both_triggers_needs_the_guard():
    phx = ZoneInfo("America/Phoenix")
    two = ["2026-08-17T12:20:00", "2026-08-17T13:20:00"]
    assert pf.analyze_task_triggers(_xml(two), "14:20", phx)[0] == "PASS"
    assert pf.analyze_task_triggers(_xml(two, args="-File x.ps1"), "14:20", phx)[0] == "FAIL"


def test_task_with_no_trigger_fails():
    assert pf.analyze_task_triggers(_xml([]), "14:20", CHI)[0] == "FAIL"


# --- the wrappers carry the guard ---------------------------------------------------
@pytest.mark.parametrize("name", ["ibkr_execute_live_accounts.ps1", "ibkr_reconcile_live_accounts.ps1"])
def test_wrappers_call_the_guard_and_stay_ascii(name):
    text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
    assert "slot_guard.py" in text and "$SlotCT" in text and "exit 0" in text
    assert text.isascii()


def test_setup_registers_guarded_multi_trigger_tasks():
    text = (ROOT / "scripts" / "setup_windows_option_c.ps1").read_text(encoding="utf-8")
    assert text.isascii()
    assert "-SlotCT `\"$LiveTaskTime`\"" in text and "New-CentralSlotTriggers $LiveTaskTime" in text
    assert "-SlotCT `\"$ReconcileTaskTime`\"" in text
