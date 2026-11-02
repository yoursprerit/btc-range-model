"""DST-proof clock slot for the Windows scheduled tasks.

The live executor must run at 2:20 PM US Central whatever the PC's own time
zone is and whichever side of a US daylight-saving change we are on. A Task
Scheduler trigger is only a wall-clock hint -- one pinned to UTC ("Synchronize
across time zones") slid to 1:20 PM CT on 2026-11-02. So the trigger is allowed
to fire at EVERY local time the slot can map to over the year, and the wrapper
asks this module whether *America/Chicago* is actually inside the slot window.

CLI (stdlib only, no repo imports):
    slot_guard.py --slot 14:20                 exit 0 inside the window, 10 outside
    slot_guard.py --slot 14:20 --local-times   print the local HH:MM triggers to register
    --now ISO8601 overrides the clock (tests / dry runs); --tz overrides America/Chicago.

Exit 2 on a usage error. The wrappers fail OPEN on anything but 0 / 10: a run at
the wrong minute is recoverable (the runner has its own cutoff and duplicate
guards), a skipped trading day is not.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timezone

try:
    from zoneinfo import ZoneInfo
except ImportError:                                        # pragma: no cover
    ZoneInfo = None                                        # type: ignore

DEFAULT_TZ = "America/Chicago"
BEFORE_MIN = 5      # a trigger that wakes a few minutes early still counts
AFTER_MIN = 30      # < 60: the same trigger an hour off (wrong DST side) never counts
EXIT_IN, EXIT_OUT, EXIT_USAGE = 0, 10, 2


def parse_slot(slot: str) -> tuple[int, int]:
    hh, mm = (int(x) for x in slot.split(":"))
    if not (0 <= hh < 24 and 0 <= mm < 60):
        raise ValueError(slot)
    return hh, mm


def in_slot(now: datetime, slot: str, tz: str = DEFAULT_TZ,
            before: int = BEFORE_MIN, after: int = AFTER_MIN) -> tuple[bool, str]:
    """Is *now* (tz-aware) within [slot - before, slot + after] on the wall clock of *tz*?"""
    hh, mm = parse_slot(slot)
    local = now.astimezone(ZoneInfo(tz))
    anchor = local.replace(hour=hh, minute=mm, second=0, microsecond=0)
    delta = (local - anchor).total_seconds() / 60.0
    ok = -before <= delta <= after
    detail = (f"{local:%Y-%m-%d %H:%M %Z} ({tz}) is {delta:+.0f} min from the "
              f"{slot} slot; window -{before}/+{after} min")
    return ok, detail


def local_trigger_times(slot: str, tz: str = DEFAULT_TZ, year: int | None = None) -> list[str]:
    """Distinct machine-local HH:MM at which *slot* (wall clock of *tz*) lands,
    sampled on the 1st and 15th of every month so the weeks where the machine's
    zone and *tz* change clocks on different days are covered too."""
    hh, mm = parse_slot(slot)
    year = year or date.today().year
    zone = ZoneInfo(tz)
    seen: set[str] = set()
    for month in range(1, 13):
        for day in (1, 15):
            central = datetime(year, month, day, hh, mm, tzinfo=zone)
            seen.add(central.astimezone().strftime("%H:%M"))   # .astimezone() = machine zone
    return sorted(seen)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--slot", required=True, help="HH:MM on the wall clock of --tz")
    p.add_argument("--tz", default=DEFAULT_TZ)
    p.add_argument("--now", help="ISO-8601 instant (default: now); naive = UTC")
    p.add_argument("--local-times", action="store_true",
                   help="print the machine-local trigger times instead of checking")
    a = p.parse_args(argv)
    try:
        parse_slot(a.slot)
        if ZoneInfo is None:
            raise RuntimeError("zoneinfo unavailable")
        if a.local_times:
            print("\n".join(local_trigger_times(a.slot, a.tz)))
            return 0
        now = datetime.fromisoformat(a.now) if a.now else datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        ok, detail = in_slot(now, a.slot, a.tz)
    except Exception as e:                                 # noqa: BLE001
        print(f"slot_guard: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_USAGE
    print(("IN " if ok else "OUT ") + detail)
    return EXIT_IN if ok else EXIT_OUT


if __name__ == "__main__":
    sys.exit(main())
