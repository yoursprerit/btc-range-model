"""Machine-checkable half of the Option C pre-flight (see IBKR_OPTION_C_WINDOWS.md).

Everything here is stdlib-only and broker-free, so it runs before the venv is
complete and without IB Gateway up.  The PowerShell wrapper
(``setup_windows_option_c.ps1 -Preflight``) adds the Windows-only checks --
scheduled task, TCP reachability, autostart shortcut, git credential -- and
merges the two reports.

Each check prints one tab-separated line::

    PASS<TAB>name<TAB>detail
    WARN<TAB>name<TAB>detail        # works, but you probably want to fix it
    FAIL<TAB>name<TAB>detail        # will break the 2:30 PM CT run

Exit code is the number of FAILs (0 = ready to trade), so a caller can just
test the exit status.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BOOK = REPO / "data" / "overall" / "target_book.json"

# Mirrors app/target_book.py: the executor's own limits, restated here so this
# script stays importable with nothing but the stdlib.
MAX_BAR_AGE_DAYS = 4
MAX_GEN_AGE_HOURS = 36.0

# Mirrors scripts/ibkr_common.py: since the 2026-08-18 duplicate-execution
# incident the executor also requires the book's bar to be the LAST COMPLETED
# SESSION (a 36-hour-fresh book can still be a day-old decision), and refuses a
# bar it has already executed. Both are restated here so the operator learns it
# now rather than from an ABORT when the 2:30 PM CT task fires.
EXECUTED_ARCHIVE = REPO / "data" / "overall" / "executed_archive"
US_MARKET_HOLIDAYS = {
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
    "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
}


def _prev_trading_day(day):
    """The last US trading session strictly before *day* (stdlib mirror of
    ibkr_common.prev_trading_day)."""
    d = day
    for _ in range(10):
        d -= timedelta(days=1)
        if d.weekday() < 5 and d.isoformat() not in US_MARKET_HOLIDAYS:
            return d
    return day

_results: list[tuple[str, str, str]] = []


def _add(status: str, name: str, detail: str) -> None:
    _results.append((status, name, detail))
    print(f"{status}\t{name}\t{detail}")


def _canonical(payload: dict) -> bytes:
    """Byte-for-byte the encoding app/target_book.py signs over."""
    body = {k: v for k, v in payload.items() if k != "signature"}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()


def check_unicode_io() -> None:
    """Regression guard for the 2026-08-17 outage: the executor prints arrows
    and box rules, and a stdout that cannot encode them kills the run mid-plan.
    Only ever reproduced under Task Scheduler, where stdout is a pipe."""
    enc = (getattr(sys.stdout, "encoding", None) or "").lower()
    try:
        "→─—".encode(enc or "ascii")
    except (UnicodeEncodeError, LookupError):
        _add("FAIL", "unicode-stdout",
             f"stdout encoding {enc!r} cannot encode the arrows the executor "
             f"prints -- set PYTHONUTF8=1 (the daily wrapper does this)")
        return
    _add("PASS", "unicode-stdout", f"stdout encoding {enc} handles the plan glyphs")


def check_deps() -> None:
    for mod, why in (("pandas", "the executor and every shared book module import it"),
                     ("ib_async", "the broker layer")):
        try:
            m = __import__(mod)
            _add("PASS", f"import-{mod}", f"{getattr(m, '__version__', 'present')}")
        except ImportError:
            _add("FAIL", f"import-{mod}",
                 f"not installed ({why}) -- run setup with -Venv")


def check_book() -> dict | None:
    if not BOOK.exists():
        _add("FAIL", "book-present", f"{BOOK} is missing -- publish one and git pull")
        return None
    try:
        payload = json.loads(BOOK.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        _add("FAIL", "book-present", f"{BOOK} unreadable: {e}")
        return None
    _add("PASS", "book-present", f"as-of {payload.get('as_of')}, "
                                 f"profile {payload.get('profile')}")

    audit = payload.get("signal_audit")
    if audit is None:
        _add("WARN", "book-audit", "no signal_audit stamp (book predates the audit)")
    elif audit.get("passed"):
        _add("PASS", "book-audit", "publish-time signal-freshness audit passed")
    else:
        stale = ", ".join(audit.get("stale_apps") or []) or "unknown"
        _add("FAIL", "book-audit", f"audit FAILED at publish (stale: {stale}) -- "
                                   f"the executor refuses to trade this book")

    as_of = payload.get("as_of")
    try:
        from zoneinfo import ZoneInfo
        today = datetime.now(ZoneInfo("America/New_York")).date()
    except ImportError:                                    # pragma: no cover
        today = datetime.now(timezone.utc).date()
    if as_of:
        # ET, like the executor: after 20:00 ET the UTC date has already rolled,
        # and judging the bar by it would report an age the run never sees
        bar_age = (today - datetime.strptime(as_of, "%Y-%m-%d").date()).days
        if bar_age > MAX_BAR_AGE_DAYS:
            _add("FAIL", "book-bar-age",
                 f"signal bar {as_of} is {bar_age}d old (limit {MAX_BAR_AGE_DAYS}d)")
        else:
            _add("PASS", "book-bar-age", f"signal bar {as_of} is {bar_age}d old")

    if as_of:
        # the one bar this session may trade
        want = _prev_trading_day(today)
        bar = datetime.strptime(as_of, "%Y-%m-%d").date()
        if bar == want:
            _add("PASS", "book-current-bar",
                 f"signal bar {as_of} is the last completed session")
        elif bar > want:
            _add("WARN", "book-current-bar",
                 f"signal bar {as_of} is ahead of the last completed session "
                 f"({want}) -- not tradeable yet")
        else:
            _add("FAIL", "book-current-bar",
                 f"STALE: signal bar {as_of} is behind the last completed session "
                 f"({want}) -- the executor will refuse to trade it. Today's "
                 f"publish has not landed; git pull, or check the publisher")

        # already traded? the archived execution report is the executor's lock
        rec = EXECUTED_ARCHIVE / f"{as_of}.json"
        if rec.exists():
            _add("WARN", "book-already-executed",
                 f"{rec.name} exists -- this bar was already executed, so the "
                 f"next run will be a no-op (--refresh-report re-states the "
                 f"account without trading; --force-rerun only if that run "
                 f"placed nothing)")
        else:
            _add("PASS", "book-already-executed",
                 f"no execution report for {as_of} yet -- this bar is untraded")

    gen = payload.get("generated_at_utc")
    if gen:
        gen_ts = datetime.fromisoformat(gen)
        if gen_ts.tzinfo is None:
            gen_ts = gen_ts.replace(tzinfo=timezone.utc)
        age_h = (datetime.now(timezone.utc) - gen_ts).total_seconds() / 3600.0
        # The book is republished each morning, so judge it against the NEXT
        # 2:30 PM CT slot rather than right now -- a book that is fine at
        # midday can still be stale by the time the task fires.
        if age_h > MAX_GEN_AGE_HOURS:
            _add("FAIL", "book-gen-age",
                 f"generated {age_h:.1f}h ago (limit {MAX_GEN_AGE_HOURS}h) -- "
                 f"today's publish has not landed; git pull")
        elif age_h > MAX_GEN_AGE_HOURS - 6:
            _add("WARN", "book-gen-age",
                 f"generated {age_h:.1f}h ago -- within {MAX_GEN_AGE_HOURS}h now, "
                 f"but close to the limit at the next 2:30 PM CT fire")
        else:
            _add("PASS", "book-gen-age", f"generated {age_h:.1f}h ago")
    return payload


def check_secret(payload: dict | None) -> None:
    """The check that would have caught the 2026-08-17 rotation mistake: not
    'is a secret set' but 'does THIS secret actually verify THIS book'."""
    secret = os.environ.get("OVERALL_BOOK_SECRET")
    if not secret:
        # The caller (setup_windows_option_c.ps1 -Preflight) hydrates this from
        # the persisted USER value when the shell lacks it, so reaching here
        # means it is genuinely not set anywhere -- the scheduled task will not
        # see it either.
        _add("FAIL", "secret-set",
             "OVERALL_BOOK_SECRET is not set for this user -- the executor "
             "SKIPS verification rather than failing, so the book would be "
             "traded unverified. Set it with setx (section 4).")
        return
    if secret != secret.strip():
        _add("WARN", "secret-set",
             "value has leading/trailing whitespace -- the comparison is over "
             "raw bytes, so this will not match the publisher")
    elif os.environ.get("IBKR_SECRET_FROM_PERSISTED"):
        # Persisted but absent from the calling shell: harmless for the
        # scheduled run (Task Scheduler reads the persisted value), but a
        # manual dry-run from THIS window would skip verification silently.
        _add("WARN", "secret-set",
             f"present ({len(secret)} chars) but NOT in the shell you ran this "
             f"from -- the scheduled task is fine; re-open PowerShell before "
             f"running the executor by hand, or it verifies nothing")
    else:
        _add("PASS", "secret-set", f"present ({len(secret)} chars)")

    if payload is None:
        return
    sig = (payload.get("signature") or {}).get("value")
    if not sig:
        _add("FAIL", "secret-verifies",
             "the book carries no signature but a secret is configured")
        return
    expect = hmac.new(secret.encode(), _canonical(payload), hashlib.sha256).hexdigest()
    if hmac.compare_digest(expect, str(sig)):
        _add("PASS", "secret-verifies", "HMAC matches -- this secret signed this book")
    else:
        _add("FAIL", "secret-verifies",
             "HMAC MISMATCH: this secret did NOT sign this book. The publisher's "
             "GitHub repo secret and this host disagree -- see "
             "IBKR_OPTION_C_WINDOWS.md section 4 (rotating the secret)")


def check_slot() -> None:
    """The scheduled trigger fires in LOCAL time; the target is 2:30 PM US
    Central. Report what this machine's 14:30 actually maps to."""
    try:
        from zoneinfo import ZoneInfo
    except ImportError:                                    # pragma: no cover
        _add("WARN", "task-slot", "zoneinfo unavailable -- cannot check the timezone")
        return
    task_time = os.environ.get("IBKR_TASK_TIME", "14:30")
    try:
        hh, mm = (int(x) for x in task_time.split(":"))
    except ValueError:
        _add("WARN", "task-slot", f"unparseable task time {task_time!r}")
        return
    local_today = datetime.now().astimezone().replace(
        hour=hh, minute=mm, second=0, microsecond=0)
    ct = local_today.astimezone(ZoneInfo("America/Chicago"))
    et = local_today.astimezone(ZoneInfo("America/New_York"))
    detail = (f"{task_time} local = {ct:%H:%M} US Central = {et:%H:%M} ET "
              f"(machine zone {local_today.tzname()})")
    if (ct.hour, ct.minute) == (14, 30):
        _add("PASS", "task-slot", detail)
    elif 8 <= et.hour < 16 or (et.hour == 15 and et.minute < 60):
        _add("WARN", "task-slot", detail + " -- inside RTH but not the 2:30 CT slot")
    else:
        _add("FAIL", "task-slot", detail + " -- OUTSIDE US market hours; orders "
                                           "placed then are rejected. Re-register "
                                           "the task with the right -TaskTime")


def check_kill_switch() -> None:
    path = os.environ.get("IBKR_KILL_SWITCH_FILE")
    if not path:
        _add("WARN", "kill-switch", "IBKR_KILL_SWITCH_FILE not set -- no emergency "
                                    "stop and no way to run a no-order rehearsal")
    elif Path(path).exists():
        _add("FAIL", "kill-switch", f"ARMED ({path} exists) -- trading is disabled. "
                                    f"Delete that file to re-enable.")
    else:
        _add("PASS", "kill-switch", f"disarmed (arm by creating {path})")


def main() -> int:
    print("=== Option C pre-flight (python checks) ===")
    check_unicode_io()
    check_deps()
    payload = check_book()
    check_secret(payload)
    check_slot()
    check_kill_switch()
    fails = sum(1 for s, _, _ in _results if s == "FAIL")
    warns = sum(1 for s, _, _ in _results if s == "WARN")
    print(f"=== {len(_results) - fails - warns} pass, {warns} warn, {fails} fail ===")
    return fails


if __name__ == "__main__":
    sys.exit(main())
