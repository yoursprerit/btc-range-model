"""Signal-freshness bookkeeping shared by every app, the Overall engine and the
daily publisher — the ONE place that knows *when each app's signals close*.

Semantics (kept in lock-step with the engines):

* **Bitcoin (BTC / MSTR / MSTU)** — the BTC app trades 12:00-UTC-anchored daily
  bars (7:00 AM CT in summer, 6:00 AM CT in winter).  Bar *D* covers
  ``[D 12:00 UTC, D+1 12:00 UTC)`` and its signals become available the moment
  the bar closes at ``D+1 12:00 UTC``.  The engines report ``as_of`` as the bar
  START date *D*, so the freshest possible ``as_of`` is *today−1* once 12:00 UTC
  has passed (else *today−2*).

* **Every other app (GLDM / SOXX / GRID / XLE / REMX / WGMI / PBW / ARTY)** —
  signals are generated upon the US market close, 4:00 PM ET, of the bar's
  session date.  The freshest completed bar is the most recent NYSE session
  whose 4:00 PM ET close has passed (weekends and exchange holidays skipped).
  During market hours a partial *today* bar may also appear — that is *fresher*
  than the last completed close, never staler.

The audit (``audit_universe``) compares each signal app's actual ``as_of``
against the freshest completed bar it *could* have, and flags anything older as
STALE.  The same functions drive the per-app "signals generated from …" captions
and the 🕵️ Daily Audit tab, so the times shown everywhere always agree.

A tiny JSON log (``runtime/freshness_log.json``) records, per app, when its page
last refreshed and which close it displayed; the Daily Audit tab reads it.  All
writes are best-effort and atomic — a failed write never breaks an app render.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_DIR = _REPO_ROOT / "runtime"
REFRESH_LOG = RUNTIME_DIR / "freshness_log.json"
DAILY_AUDIT_JSON = _REPO_ROOT / "data" / "overall" / "daily_audit.json"

CT = "America/Chicago"
ET = "America/New_York"

# Which close governs each SIGNAL app (parent key).  Anything not listed is a
# US-equity app (signals from the 4:00 PM ET market close).
PARENT_CLASS = {"BTC": "crypto"}

# The daily publish is scheduled ~15 min after the Bitcoin bar close so the
# Overall strategy sees BTC's fresh 12:00-UTC signals AND every equity app's
# prior-session close.  Expressed in CT because that is how the user thinks of
# it ("7:15 AM Central"); the workflow guard resolves DST via America/Chicago.
SCHEDULED_PUBLISH_CT = "≈7:15 AM US Central (15 min after the Bitcoin bar close)"


def now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _as_utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def fmt_ct(ts, seconds: bool = False) -> str:
    """A timestamp in US Central, e.g. ``Jul 21, 2026 at 7:15 AM CDT``."""
    if ts is None:
        return "—"
    try:
        t = _as_utc(ts).tz_convert(CT)
    except Exception:
        return str(ts)
    fmt = "%b %d, %Y at %I:%M:%S %p %Z" if seconds else "%b %d, %Y at %I:%M %p %Z"
    return t.strftime(fmt).replace(" 0", " ")


# ════════════════════════════════════════════════════════════════════════════
# US (NYSE) trading calendar — weekends + the ten full-day exchange holidays.
# ════════════════════════════════════════════════════════════════════════════
def _easter(year: int) -> date:
    """Gregorian Easter Sunday (anonymous algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    d = date(year + (month == 12), (month % 12) + 1, 1) - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d: date) -> date:
    if d.weekday() == 5:                      # Saturday → observed Friday
        return d - timedelta(days=1)
    if d.weekday() == 6:                      # Sunday → observed Monday
        return d + timedelta(days=1)
    return d


def us_market_holidays(year: int) -> set[date]:
    """NYSE full-day closures for ``year`` (regular schedule; one-off special
    closures such as national days of mourning are not modelled — they would
    surface as a one-day STALE flag, which is the safe direction)."""
    hol = {
        _observed(date(year, 1, 1)),                    # New Year's Day
        _nth_weekday(year, 1, 0, 3),                    # MLK Day (3rd Mon Jan)
        _nth_weekday(year, 2, 0, 3),                    # Washington's Birthday
        _easter(year) - timedelta(days=2),              # Good Friday
        _last_weekday(year, 5, 0),                      # Memorial Day
        _observed(date(year, 7, 4)),                    # Independence Day
        _nth_weekday(year, 9, 0, 1),                    # Labor Day
        _nth_weekday(year, 11, 3, 4),                   # Thanksgiving
        _observed(date(year, 12, 25)),                  # Christmas
    }
    if year >= 2022:
        hol.add(_observed(date(year, 6, 19)))           # Juneteenth
    # A Jan-1 that falls on Saturday is observed the PRIOR Dec-31 (belongs to
    # the previous year); NYSE then has no Jan-1 holiday that year at all.
    if date(year, 1, 1).weekday() == 5:
        hol.discard(date(year - 1, 12, 31))
        hol.discard(date(year, 1, 1))
    return hol


def is_us_trading_day(d) -> bool:
    d = pd.Timestamp(d).date()
    return d.weekday() < 5 and d not in us_market_holidays(d.year)


def _prev_trading_day(d: date) -> date:
    d -= timedelta(days=1)
    while not is_us_trading_day(d):
        d -= timedelta(days=1)
    return d


# ════════════════════════════════════════════════════════════════════════════
# Expected freshest bar per asset class
# ════════════════════════════════════════════════════════════════════════════
def expected_equity_asof(now=None) -> pd.Timestamp:
    """Session date of the most recent COMPLETED US market close (4:00 PM ET)."""
    now_et = _as_utc(now or now_utc()).tz_convert(ET)
    d = now_et.date()
    if is_us_trading_day(d) and (now_et.hour, now_et.minute) >= (16, 0):
        return pd.Timestamp(d)
    return pd.Timestamp(_prev_trading_day(d))


def expected_crypto_asof(now=None) -> pd.Timestamp:
    """Bar-START date of the most recent completed 12:00-UTC Bitcoin bar (the
    engines' ``as_of`` convention: bar D closes at D+1 12:00 UTC)."""
    n = _as_utc(now or now_utc())
    close_day = n.date() if n.hour >= 12 else (n.date() - timedelta(days=1))
    return pd.Timestamp(close_day - timedelta(days=1))


def expected_asof(kind: str, now=None) -> pd.Timestamp:
    return (expected_crypto_asof(now) if kind == "crypto"
            else expected_equity_asof(now))


def close_moment(kind: str, asof) -> pd.Timestamp:
    """The wall-clock CLOSE datetime implied by a bar's ``as_of`` date — the
    moment those signals were generated.  Crypto: bar start *D* closes at
    ``D+1 12:00 UTC``; equity: session *D* closes at ``D 16:00 ET``."""
    d = pd.Timestamp(asof).tz_localize(None).normalize()
    if kind == "crypto":
        return (d + pd.Timedelta(days=1, hours=12)).tz_localize("UTC")
    return (d + pd.Timedelta(hours=16)).tz_localize(ET).tz_convert("UTC")


def close_label(kind: str, asof, now=None) -> str:
    """User-friendly description of the close a bar's signals come from, e.g.
    ``Jul 21, 2026, 7:00 AM CDT (Bitcoin daily-bar close, 12:00 UTC)`` or
    ``Jul 18, 2026, 4:00 PM ET (US market close)``.  A bar whose close is still
    in the future (an in-progress intraday bar) is labelled as such."""
    cm = close_moment(kind, asof)
    if kind == "crypto":
        lbl = f"{cm.tz_convert(CT).strftime('%b %d, %Y, %I:%M %p %Z').replace(' 0', ' ')} " \
              f"(Bitcoin daily-bar close, 12:00 UTC)"
    else:
        lbl = f"{cm.tz_convert(ET).strftime('%b %d, %Y, %I:%M %p %Z').replace(' 0', ' ')} " \
              f"(US market close)"
    n = _as_utc(now or now_utc())
    if cm > n:
        lbl += " — bar still in progress"
    return lbl


def signal_close_caption(kind: str, asof, now=None, extra: str = "") -> str:
    """The standard two-part freshness line every app shows under its title:
    the close its signals are generated from + when this page's data was last
    refreshed.  Formatted identically everywhere so it always matches the
    🕵️ Daily Audit tab."""
    n = now or now_utc()
    parts = [f"📅 Signals generated from the close of **{close_label(kind, asof, n)}**",
             f"🔄 Page data refreshed **{fmt_ct(n, seconds=True)}**"]
    if extra:
        parts.append(extra)
    return " · ".join(parts)


# ════════════════════════════════════════════════════════════════════════════
# The audit — is every signal app on its freshest possible bar?
# ════════════════════════════════════════════════════════════════════════════
def audit_universe(results: list[dict], now=None,
                   parent_order: list[str] | None = None) -> dict:
    """Freshness audit over Overall-engine result dicts (one per instrument,
    grouped by their parent signal app).

    For each signal app: ``actual`` = the newest ``as_of`` bar its instruments
    carry; ``expected`` = the freshest completed bar for its asset class right
    now.  ``fresh`` ⇔ ``actual ≥ expected`` (a partial intraday bar counts as
    fresh).  Returns per-app rows plus an overall ``passed`` verdict and the
    list of stale asset keys — the exact set the Overall app must flag."""
    n = _as_utc(now or now_utc())
    groups: dict[str, list[dict]] = {}
    for r in results or []:
        groups.setdefault(r.get("parent", r.get("key")), []).append(r)
    order = [k for k in (parent_order or list(groups))if k in groups]
    order += [k for k in groups if k not in order]

    rows, stale_parents, stale_assets = [], [], []
    for pk in order:
        grp = groups[pk]
        kind = PARENT_CLASS.get(pk, "us_equity")
        actual = max(pd.Timestamp(r["as_of"]).tz_localize(None).normalize()
                     for r in grp)
        exp = expected_asof(kind, n)
        fresh = actual >= exp
        age = max((exp - actual).days, 0)
        keys = [r["key"] for r in grp]
        if not fresh:
            stale_parents.append(pk)
            stale_assets.extend(keys)
        rows.append(dict(
            app=pk, asset_class=kind, instruments=keys,
            actual_asof=str(actual.date()), expected_asof=str(exp.date()),
            actual_close=close_label(kind, actual, n),
            expected_close=close_label(kind, exp, n),
            fresh=bool(fresh), age_days=int(age),
        ))
    return dict(
        checked_at_utc=n.isoformat(timespec="seconds"),
        checked_at_ct=fmt_ct(n, seconds=True),
        passed=not stale_parents,
        stale_apps=stale_parents, stale_assets=stale_assets,
        rows=rows,
    )


# ════════════════════════════════════════════════════════════════════════════
# Refresh log — per-app "page last refreshed / close displayed" bookkeeping.
# ════════════════════════════════════════════════════════════════════════════
def record_refresh(app_key: str, **fields) -> None:
    """Merge one app's freshness record into ``runtime/freshness_log.json``.
    Atomic replace, fully best-effort — never raises into an app render."""
    try:
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        log = read_refresh_log()
        entry = dict(fields)
        entry["recorded_at_utc"] = now_utc().isoformat(timespec="seconds")
        entry["recorded_at_ct"] = fmt_ct(now_utc(), seconds=True)
        log[app_key] = entry
        fd, tmp = tempfile.mkstemp(dir=str(RUNTIME_DIR), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(log, f, indent=1, default=str)
            os.replace(tmp, REFRESH_LOG)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
    except Exception:
        pass


def read_refresh_log() -> dict:
    try:
        return json.loads(REFRESH_LOG.read_text())
    except Exception:
        return {}


def load_daily_audit() -> dict | None:
    """The last scheduled-run audit artifact (written by the 7:15-AM-CT
    publisher, committed to the repo).  ``None`` if not present/readable."""
    try:
        return json.loads(DAILY_AUDIT_JSON.read_text())
    except Exception:
        return None
