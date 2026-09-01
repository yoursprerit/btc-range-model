"""Signal-freshness bookkeeping shared by every app, the Overall engine and the
daily publisher — the ONE place that knows *when each app's signals close*.

Semantics (kept in lock-step with the engines):

* **Bitcoin (BTC / MSTR / MSTU / ETH)** — the BTC app trades 12:00-UTC-anchored daily
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

Recovering a session the price feed is withholding
-------------------------------------------------
Yahoo's daily series lags its own intraday series, and on shared egress IPs
(Streamlit Community Cloud, GitHub Actions) its responses are rate-limited or
stale for hours — which strands every equity app one session behind and trips the
STALE-SIGNALS alert.  Two helpers here repair that with real data rather than a
relaxed audit, and the daily fetchers apply them in this order:

1. ``backfill_sessions_from_hourly`` — rebuild the missing session from Yahoo's
   *hourly* series (a different pipeline that usually still has it);
2. ``merge_missing_sessions`` — if Yahoo is withholding it on both feeds, take it
   from an INDEPENDENT provider (``app/market_fallback.py``, Nasdaq's keyless
   quote API), with a split-scale guard because the two providers differ in
   split/dividend adjustment.

3. ``repair_missing_closes`` — when the withheld close belongs to a traded
   SIBLING (SOXL / OIH / ERX / UGL / NUGT) the session row survives on the
   primary's price and only that cell is null, so it is repaired cell-by-cell
   from the same two sources rather than forward-filled into a fabricated price.

``repair_daily_frame`` applies all three in that order and is what the daily
fetchers call.

Every one of them is strictly additive: they never overwrite a value the primary
feed has, never add an in-progress or non-trading session, and no-op when nothing
is missing — so a genuinely stale feed still fails the audit.

Crucially the hole they look for is any COMPLETED session missing from the recent
window, not merely a short tail: a feed that WITHDRAWS an already-published close
(Yahoo began serving a null for the 2026-08-28 bar of nine tickers on
2026-09-01) leaves the frame's newest session perfectly current with the gap
buried inside it, which a tail-only test cannot see.  See ``missing_sessions``.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import date, datetime, timedelta
from functools import lru_cache
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

# The publish runs ONCE daily at ≈7:15 AM US Central — ~15 min after the
# Bitcoin bar close (12:00 UTC = 7:00 AM CDT), so the Overall strategy sees
# BTC's fresh 7:00-AM-CT signals AND every equity app's prior-session 4:00 PM
# ET close.  The daily audit runs ONCE, before the book is written; the
# published book then stays FROZEN until the next morning's cycle — only the
# UI's 🚀 publish button (a manual workflow dispatch) replaces it intraday.
# GitHub cron is best-effort, so the cycle has several catch-up slots; the
# workflow guard retries until the day's book publishes.
SCHEDULED_PUBLISH_CT = ("once daily at ≈7:15 AM US Central (≈15 min after the "
                        "7:00-AM-CT Bitcoin daily-bar close), with catch-up "
                        "retries until published; frozen until the next "
                        "morning unless manually re-published from the UI")


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


@lru_cache(maxsize=64)
def us_market_holidays(year: int) -> frozenset:
    """NYSE full-day closures for ``year`` (regular schedule; one-off special
    closures such as national days of mourning are not modelled — they would
    surface as a one-day STALE flag, which is the safe direction).

    Cached and returned immutable: the interior-gap repair walks up to
    ``REPAIR_LOOKBACK_SESSIONS`` sessions per frame and would otherwise rebuild
    this set hundreds of times per fetch, across every sleeve, on every load."""
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
    return frozenset(hol)


def is_us_trading_day(d) -> bool:
    d = pd.Timestamp(d).date()
    return d.weekday() < 5 and d not in us_market_holidays(d.year)


def _prev_trading_day(d: date) -> date:
    d -= timedelta(days=1)
    while not is_us_trading_day(d):
        d -= timedelta(days=1)
    return d


def _nth_prev_trading_day(d: date, n: int) -> date:
    """The session ``n`` trading days before ``d`` (``n`` <= 0 → ``d``)."""
    for _ in range(max(0, int(n))):
        d = _prev_trading_day(d)
    return d


def us_sessions_between(start, end) -> list:
    """Every NYSE session date in ``[start, end]`` inclusive, oldest first."""
    a = pd.Timestamp(start).date()
    b = pd.Timestamp(end).date()
    out = []
    while a <= b:
        if is_us_trading_day(a):
            out.append(pd.Timestamp(a))
        a += timedelta(days=1)
    return out


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


# ── completed-bars-only mode (the once-daily publisher) ──────────────────────
# The published Target Book's data basis is PINNED to the publish day's
# 7:15-AM-Central anchor, no matter what wall-clock time the publish actually
# runs: every equity ticker from the last 4:00-PM-ET market close BEFORE the
# anchor (i.e. the previous session — post-close signal changes belong only to
# the live "Recommended Live Possible Targetbook" view until the next
# morning), and BTC/MSTR/MSTU/ETH from the day's 7:00-AM-CT (12:00 UTC) bar close
# (the BTC engine already enforces its own bar close).  Live Yahoo daily feeds
# include an in-progress *today* row during market hours — and a completed
# *today* row after 4:00 PM ET — but neither may leak into a published book
# (e.g. a delayed catch-up publish or the UI's 🚀 manual publish fired
# mid-session or after the close).  The publisher sets this env flag; the
# daily fetchers then trim every US bar after the anchor's basis session.
COMPLETED_BARS_ENV = "OVERALL_COMPLETED_BARS_ONLY"


def publish_anchor_ct(now=None) -> pd.Timestamp:
    """The publish day's 7:15-AM-America/Chicago anchor (as a UTC timestamp).
    The Target Book's data basis is evaluated AT this moment for the whole CT
    day, so a publish at 9 AM, 2 PM or 9 PM CT all produce the same book the
    7:15 AM run would have."""
    n = _as_utc(now or now_utc()).tz_convert(CT)
    return (n.normalize() + pd.Timedelta(hours=7, minutes=15)).tz_convert("UTC")


def publish_pending(generated_at_utc, now=None) -> bool:
    """True when *now* is already past today's 7:15-AM-CT publish anchor but
    the published book at hand was generated BEFORE it — i.e. today's scheduled
    publish hasn't landed yet and the book shown is still yesterday's.  GitHub
    cron fires are routinely delivered late (or dropped), so this is the state
    the Targetbook donuts sit in every morning between the anchor and the
    moment the day's publish commit actually lands.  Falsy/unparsable input →
    False (no notice rather than a wrong one)."""
    if not generated_at_utc:
        return False
    try:
        gen = _as_utc(generated_at_utc)
        n = _as_utc(now or now_utc())
    except Exception:
        return False
    return bool(gen < publish_anchor_ct(n) <= n)


def completed_bars_only() -> bool:
    return os.environ.get(COMPLETED_BARS_ENV, "").strip().lower() in (
        "1", "true", "yes", "on")


def last_completed_session(index, now=None) -> pd.Timestamp | None:
    """The newest COMPLETED US session present in a daily index — i.e. the
    newest entry at or before ``expected_equity_asof(now)``.

    This — not ``index.max()`` — is how "is this daily frame behind?" must be
    measured: during US market hours Yahoo's daily series carries an
    in-progress *today* row even while a prior session's completed close is
    still missing from it (observed 2026-07-27: Mon's partial bar present,
    Fri Jul 24 absent), so ``index.max()`` masks the gap and the backstops
    below never fire.  ``None`` when the index is empty/unparseable or holds
    no completed session."""
    try:
        idx = pd.DatetimeIndex(index).normalize()
        if idx.tz is not None:
            idx = idx.tz_localize(None)
    except Exception:
        return None
    if not len(idx):
        return None
    idx = idx[idx <= expected_equity_asof(now)]
    last = idx.max() if len(idx) else None
    return None if last is None or pd.isna(last) else pd.Timestamp(last)


def drop_in_progress_us_bar(df: pd.DataFrame, now=None) -> pd.DataFrame:
    """Drop trailing daily rows whose US-session 4:00-PM-ET close hasn't
    happened yet (the in-progress *today* bar during market hours).  No-op on
    an empty frame or when every bar is already complete."""
    if df is None or len(df) == 0:
        return df
    cutoff = expected_equity_asof(now)
    try:
        idx = pd.DatetimeIndex(df.index).normalize()
        if idx.tz is not None:
            idx = idx.tz_localize(None)
    except Exception:
        return df
    return df.loc[idx <= cutoff]


# ── intraday backstop: rebuild a daily bar Yahoo's daily feed is missing ─────
# Yahoo serves the daily (`interval=1d`) and intraday (`interval=1h`) series from
# different pipelines, and the daily one lags: the newest session is routinely
# absent or carries a null close for a while after the 4:00 PM ET close, and on
# heavily-shared egress IPs (Streamlit Community Cloud) that lag can persist for
# hours because responses are rate-limited/cached upstream.  Since `_chart` drops
# null-close rows, the whole app then sits one session behind and the Overall
# cockpit raises its STALE-SIGNALS alert even though the data is obtainable — the
# HOURLY series usually still has the session (observed 2026-07-25: daily stuck on
# Jul 23 while hourly had Jul 24 20:00).
#
# This rebuilds those missing COMPLETED sessions by aggregating the hourly bars,
# so the fix is real data rather than a relaxed audit.  It never invents an
# in-progress session (the cap is ``expected_equity_asof``), never rewrites a
# session the daily feed already has, and returns the frame untouched if the
# hourly series is no fresher — in which case the audit correctly still flags
# staleness.
_AGG_BY_SUFFIX = {"open": "first", "high": "max", "low": "min",
                  "close": "last", "volume": "sum"}

# How far back the repair helpers look for holes.  Kept in lock-step with the
# data gate's ``no_missing_recent_sessions`` window (``data_gate.
# _RECENT_SESSION_DAYS``): the repair must cover exactly the sessions the gate
# refuses to lose, or a hole inside that window makes the gate reject every
# future fetch and the sleeve freezes on its pinned snapshot for good.
REPAIR_LOOKBACK_SESSIONS = 150


def et_session_dates(idx) -> pd.DatetimeIndex:
    """Map intraday (tz-naive UTC) timestamps to their US/Eastern session date."""
    i = pd.DatetimeIndex(idx)
    i = i.tz_localize("UTC") if i.tz is None else i.tz_convert("UTC")
    return pd.DatetimeIndex(i.tz_convert(ET).normalize().tz_localize(None))


def missing_sessions(index, now=None,
                     lookback: int = REPAIR_LOOKBACK_SESSIONS) -> list:
    """COMPLETED US sessions the frame LACKS inside the recent window, oldest first.

    This is the gap test the repair helpers below run on, and it deliberately
    looks at the whole window rather than only the tail.  Two different feed
    defects produce a hole:

    * the newest session simply has not arrived yet (the classic lag), and
    * an ALREADY-PUBLISHED session is withdrawn — Yahoo starts serving a null
      close for a bar it served correctly the day before, and ``_chart`` drops
      null-close rows.  Observed 2026-09-01, when the 2026-08-28 close vanished
      for GLDM/GRID/REMX/WGMI/PBW/ARTY/NUGT/OIH/ERX while 08-31 was present.

    Only the first is visible to a tail test (``last_completed_session(...) <
    expected_equity_asof(...)``): in the second the frame's newest session is
    perfectly current and the hole sits INSIDE the history, so the tail test
    reports "nothing to do", every backstop no-ops, and the data gate's
    ``no_missing_recent_sessions`` check then refuses the frame — stranding the
    sleeve on its pinned snapshot until someone intervenes.

    The window is bounded by ``lookback`` sessions before the cutoff and by the
    frame's own first session, so a short history is never asked to grow
    backwards.
    """
    try:
        idx = pd.DatetimeIndex(index).normalize()
        if idx.tz is not None:
            idx = idx.tz_localize(None)
    except Exception:
        return []
    if not len(idx):
        return []
    cutoff = expected_equity_asof(now)
    window_start = pd.Timestamp(_nth_prev_trading_day(cutoff.date(), lookback))
    first = max(window_start, pd.Timestamp(idx.min()))
    if first > cutoff:
        return []
    have = set(idx)
    return [d for d in us_sessions_between(first, cutoff) if d not in have]


def missing_closes(df: pd.DataFrame, cols, now=None,
                   lookback: int = REPAIR_LOOKBACK_SESSIONS) -> dict:
    """``{column: [session dates whose close is NULL]}`` inside the recent window.

    The row-level helpers above only see a session that is missing ENTIRELY —
    i.e. one the frame's PRIMARY price column dropped.  When the withdrawn close
    belongs to a traded SIBLING (SOXL, OIH, ERX, UGL, NUGT …) the session row
    survives on the primary's price and the sibling's cell is merely null, which
    the fetchers used to forward-fill — silently fabricating a traded price
    equal to the previous session's.  Observed 2026-09-01: the live XLE fetch
    carried Thursday's 414.95 / 103.10 as OIH's and ERX's Friday 08-28 closes
    (true closes 418.31 / 104.02).  The committed snapshot escaped only because
    it had been pinned on 08-29, before the feed withdrew those bars — a refresh
    landing during the withdrawal would have pinned the fabrication, and no
    quality check can catch it (a 0% move is perfectly plausible).  These are
    the prices the target book's execution legs read, so they get repaired from
    a real source, never invented.
    """
    out: dict = {}
    if df is None or len(df) == 0:
        return out
    try:
        idx = pd.DatetimeIndex(df.index).normalize()
        if idx.tz is not None:
            idx = idx.tz_localize(None)
    except Exception:
        return out
    cutoff = expected_equity_asof(now)
    window_start = pd.Timestamp(_nth_prev_trading_day(cutoff.date(), lookback))
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        s.index = idx
        s = s[(s.index >= window_start) & (s.index <= cutoff)]
        bad = [d for d in s.index[s.isna()] if is_us_trading_day(d)]
        if bad:
            out[c] = bad
    return out


def sessions_from_hourly(hourly: pd.DataFrame, columns, price_col: str):
    """Aggregate an intraday frame into daily US-session bars (OHLCV rules).

    Returns ``None`` when the hourly frame carries no usable primary price.
    """
    if hourly is None or hourly.empty:
        return None
    h = hourly.copy()
    h["_sess"] = et_session_dates(h.index)
    agg = {}
    for c in columns:
        if c not in h.columns:
            continue
        agg[c] = _AGG_BY_SUFFIX.get(str(c).rsplit("_", 1)[-1], "last")
    if price_col not in agg:
        return None
    return h.groupby("_sess").agg(agg)


def _nearest_prior_close(series: pd.Series, when) -> float | None:
    """The newest non-null value at or before ``when`` (else the newest of all).

    The split-scale guard compares a candidate against this rather than against
    the series' final value, so an INTERIOR session is judged against the level
    that actually prevailed around it instead of against a level several
    sessions away.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if not len(s):
        return None
    prior = s[s.index <= pd.Timestamp(when)]
    ref = prior if len(prior) else s
    v = float(ref.iloc[-1])
    return v if v > 0 else None


def backfill_sessions_from_hourly(daily: pd.DataFrame, hourly: pd.DataFrame,
                                  price_col: str, now=None) -> pd.DataFrame:
    """Add every COMPLETED US session the daily frame lacks, aggregated from
    ``hourly`` — tail gaps AND interior holes alike (see ``missing_sessions``).
    No-op when nothing is missing, when ``hourly`` is empty or does not cover the
    gap, or when the synthesized session has no primary-price data."""
    if daily is None or daily.empty or hourly is None or hourly.empty:
        return daily
    want = missing_sessions(daily.index, now)
    if not want:
        return daily                            # already complete — nothing to do
    rows = sessions_from_hourly(hourly, list(daily.columns), price_col)
    if rows is None or rows.empty:
        return daily
    rows = rows[rows.index.isin(want)].dropna(subset=[price_col])
    rows = rows[~rows.index.isin(daily.index)]
    if rows.empty:
        return daily
    out = pd.concat([daily, rows.reindex(columns=daily.columns)]).sort_index()
    return out[~out.index.duplicated(keep="last")]


def merge_missing_sessions(daily: pd.DataFrame, alt: pd.DataFrame, price_col: str,
                          now=None, max_rel_jump: float = 0.60) -> pd.DataFrame:
    """Add COMPLETED US sessions that ``daily`` lacks, taken from an alternate
    DAILY-indexed frame (a second provider — see ``app/market_fallback.py``).

    Used after the hourly backstop, when Yahoo is withholding sessions entirely.
    Guards, in order:

    * never touches a session ``daily`` already has, and never adds one past the
      last completed close (``expected_equity_asof``) or on a non-trading day;
    * requires the primary price column to be present and non-null;
    * **split-scale guard** — a candidate whose close differs from the nearest
      KNOWN close at or before that session by more than ``max_rel_jump`` is
      DROPPED.  The two providers differ in split/dividend adjustment, so this is
      what stops a raw, unadjusted print being spliced onto an adjusted series (a
      10-for-1 split shows up as a ~90% gap; a genuine leveraged-ETF day stays
      well inside 60%).
    """
    if daily is None or daily.empty or alt is None or alt.empty:
        return daily
    a = alt.copy()
    a.index = pd.DatetimeIndex(a.index).normalize()
    a = a[~a.index.duplicated(keep="last")].sort_index()
    if price_col not in a.columns:
        return daily
    want = [d for d in missing_sessions(daily.index, now) if d in a.index]
    if not want:
        return daily
    rows = a.loc[want].dropna(subset=[price_col])
    if rows.empty:
        return daily
    ref = pd.to_numeric(daily[price_col], errors="coerce")
    ref.index = pd.DatetimeIndex(daily.index).normalize()
    keep = []
    for d in rows.index:
        base = _nearest_prior_close(ref, d)
        keep.append(base is None
                    or abs(float(rows.loc[d, price_col]) / base - 1.0) <= max_rel_jump)
    rows = rows[keep]
    if rows.empty:
        return daily
    out = pd.concat([daily, rows.reindex(columns=daily.columns)]).sort_index()
    return out[~out.index.duplicated(keep="last")]


def repair_missing_closes(daily: pd.DataFrame, alt: pd.DataFrame, cols,
                          now=None, max_rel_jump: float = 0.60) -> pd.DataFrame:
    """Fill NULL cells of ``cols`` from ``alt``, cell by cell, inside the recent
    window — the sibling-level counterpart of ``merge_missing_sessions``.

    Only ever writes where the frame holds no value, only on completed trading
    sessions the frame already carries, and only when the candidate passes the
    same split-scale guard against the column's nearest prior known value.  The
    matching OHLCV cells of a repaired close are filled alongside it so the bar
    stays internally consistent; anything the alternate feed cannot supply is
    left NULL for the data gate to judge.
    """
    if daily is None or len(daily) == 0 or alt is None or len(alt) == 0:
        return daily
    holes = missing_closes(daily, list(cols), now)
    if not holes:
        return daily
    a = alt.copy()
    try:
        a.index = pd.DatetimeIndex(a.index).normalize()
    except Exception:
        return daily
    a = a[~a.index.duplicated(keep="last")].sort_index()
    out = daily.copy()
    out.index = pd.DatetimeIndex(out.index).normalize()
    for col, days in holes.items():
        if col not in a.columns:
            continue
        stem = str(col).rsplit("_", 1)[0]
        ref = pd.to_numeric(out[col], errors="coerce")
        for d in days:
            if d not in a.index:
                continue
            v = pd.to_numeric(pd.Series([a.loc[d, col]]), errors="coerce").iloc[0]
            if pd.isna(v) or float(v) <= 0:
                continue
            base = _nearest_prior_close(ref, d)
            if base is not None and abs(float(v) / base - 1.0) > max_rel_jump:
                continue                        # raw/unadjusted print — refuse
            for suffix in _AGG_BY_SUFFIX:
                c2 = f"{stem}_{suffix}"
                if c2 in out.columns and c2 in a.columns and pd.isna(out.at[d, c2]):
                    out.at[d, c2] = a.at[d, c2]
    return out


def repair_daily_frame(df: pd.DataFrame, price_col: str, *,
                       traded_cols=(), macro_cols=(),
                       fetch_hourly=None, fetch_alt=None,
                       now=None, ffill_limit: int = 5) -> pd.DataFrame:
    """Repair a freshly-fetched daily frame, then forward-fill the macro columns.

    One shared implementation of the resolution order documented in
    ``app/market_fallback.py`` — Yahoo daily → Yahoo hourly → Nasdaq daily —
    applied to BOTH kinds of hole: whole sessions the primary price column
    dropped, and null closes on traded siblings.  Each network fetch is made
    lazily and only while something is still missing, so a healthy frame costs
    no extra requests.

    ``fetch_hourly`` / ``fetch_alt`` are zero-argument callables returning an
    hourly frame and an alternate daily frame (either may be ``None``).  Failures
    are swallowed: the repair is best-effort, and whatever it cannot fix is left
    for the freshness audit and the data gate to catch rather than papered over.
    """
    traded_cols = [c for c in traded_cols if c in getattr(df, "columns", [])]
    macro_cols = [c for c in macro_cols if c in getattr(df, "columns", [])]

    def _ffill(frame):
        if macro_cols:
            frame[macro_cols] = frame[macro_cols].ffill(limit=ffill_limit)
        return frame

    if df is None or len(df) == 0:
        return df
    repaired_sessions: list = []
    repaired_cells: dict = {}

    def _stamp(frame):
        """Record what was synthesized rather than served by the primary feed.

        A rebuilt bar is real data but not the OFFICIAL print — an hourly
        aggregate closes on the last intraday bar, which lands within ~0.1% of
        the 4:00-PM-ET close, and a second provider carries its own adjustment
        basis.  Downstream (``data_gate``) prefers a pinned snapshot's exact
        value for any session flagged here, so a repair can never silently
        RESTATE history that was already pinned from the official print.
        """
        frame.attrs["repaired_sessions"] = [str(pd.Timestamp(d).date())
                                            for d in repaired_sessions]
        frame.attrs["repaired_cells"] = {
            c: [str(pd.Timestamp(d).date()) for d in ds]
            for c, ds in repaired_cells.items() if ds}
        return frame

    try:
        def _outstanding():
            return (missing_sessions(df.index, now),
                    missing_closes(df, traded_cols, now))

        gaps, holes = _outstanding()
        if not gaps and not holes:
            return _stamp(_ffill(df))
        def _absorb(before_gaps, before_holes):
            """Note what the step just fixed, and report what is still open."""
            after_gaps, after_holes = _outstanding()
            still = set(after_gaps)
            repaired_sessions.extend(d for d in before_gaps if d not in still)
            for c, ds in before_holes.items():
                left = set(after_holes.get(c, []))
                fixed = [d for d in ds if d not in left]
                if fixed:
                    repaired_cells.setdefault(c, []).extend(fixed)
            return after_gaps, after_holes

        if fetch_hourly is not None:
            h = fetch_hourly()
            if h is not None and len(h):
                df = backfill_sessions_from_hourly(df, h, price_col, now=now)
                agg = sessions_from_hourly(h, list(df.columns), price_col)
                if agg is not None and len(agg):
                    df = repair_missing_closes(df, agg, traded_cols, now=now)
            gaps, holes = _absorb(gaps, holes)
        if (gaps or holes) and fetch_alt is not None:
            a = fetch_alt(gaps[0] if gaps else min(
                (d for ds in holes.values() for d in ds), default=None))
            if a is not None and len(a):
                df = merge_missing_sessions(df, a, price_col, now=now)
                df = repair_missing_closes(df, a, traded_cols, now=now)
            _absorb(gaps, holes)
    except Exception:
        pass                # best-effort; the audit still flags what is left
    return _stamp(_ffill(df))


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


def next_session_date(kind: str, asof) -> pd.Timestamp:
    """The bar AFTER ``asof`` — the session an "exits/enters next bar" decision
    executes on.  Crypto bars run every calendar day; equity sessions skip
    weekends and NYSE holidays."""
    d = pd.Timestamp(asof).tz_localize(None).normalize()
    if kind == "crypto":
        return d + pd.Timedelta(days=1)
    nd = d.date() + timedelta(days=1)
    while not is_us_trading_day(nd):
        nd += timedelta(days=1)
    return pd.Timestamp(nd)


def next_close_label(kind: str, asof, now=None) -> str:
    """When a pending "next bar" decision actually executes: the close moment
    of the bar AFTER ``asof``, phrased relative to now so "next bar" stops
    reading as "tomorrow" when it means the session already underway — e.g.
    ``today, Aug 4, 4:00 PM EDT`` mid-session, or ``Aug 5, 4:00 PM EST`` after
    the close.  The strategies decide at a close and execute at the NEXT
    close, so this is the sell/buy moment of an exits/enters-next-bar flag."""
    cm = close_moment(kind, next_session_date(kind, asof))
    tz = CT if kind == "crypto" else ET
    local = cm.tz_convert(tz)
    lbl = local.strftime("%b %d, %I:%M %p %Z").replace(" 0", " ")
    today = _as_utc(now or now_utc()).tz_convert(tz).date()
    if local.date() == today:
        return "today, " + lbl
    if local.date() == today + timedelta(days=1):
        return "tomorrow, " + lbl
    return lbl


# the live IBKR executor's once-daily rebalance slot (crontab `30 14 * * 1-5`
# with CRON_TZ=America/Chicago — 2:30 PM US Central / 3:30 PM ET, 30 minutes
# before the 3:00-PM-CT equity close; see IBKR_PAPER_TRADING.md): committed
# signal changes are TRADED here, the session after they commit, while the
# engine/backtest books them at the pending bar's close.  Display-only —
# nothing schedules off this.
REBALANCE_CT = (14, 30)


def rebalance_label() -> str:
    """The executor's daily slot as shown in the UI, e.g. ``2:30 PM CT`` —
    derived from ``REBALANCE_CT`` so the copy can never drift from the slot
    the systemd timer / crontab actually fires."""
    h, m = REBALANCE_CT
    return (pd.Timestamp(2000, 1, 1, h, m).strftime("%I:%M %p").lstrip("0")
            + " CT")


def exit_execution_note(kind: str, asof, now=None) -> str:
    """Both timelines of a pending "exits next bar" decision, in one phrase:
    when the LIVE EXECUTOR actually sells (the ≈2:30 PM CT rebalance of the
    pending session — past tense once that moment has passed) and when the
    ENGINE/backtest books the exit (the pending bar's close).  Answers "it said
    exits next bar yesterday — why is it still LONG?": the executor sold at the
    afternoon rebalance; the engine's position runs to the close by design."""
    nxt_close = next_close_label(kind, asof, now)
    if kind == "crypto":
        return f"engine books the exit at the bar close — {nxt_close}"
    sess = next_session_date(kind, asof)
    reb = sess.tz_localize(CT) + pd.Timedelta(hours=REBALANCE_CT[0],
                                              minutes=REBALANCE_CT[1])
    n = _as_utc(now or now_utc())
    n_ct = n.tz_convert(CT)
    day = ("today" if reb.date() == n_ct.date() else
           "tomorrow" if reb.date() == (n_ct + pd.Timedelta(days=1)).date() else
           reb.strftime("%b %d").replace(" 0", " "))
    verb = "sold" if n >= reb.tz_convert("UTC") else "sells"
    return (f"executor {verb} at {day}'s ≈{rebalance_label()} rebalance · "
            f"engine books the exit at the close — {nxt_close}")


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
                   parent_order: list[str] | None = None,
                   expected_now=None) -> dict:
    """Freshness audit over Overall-engine result dicts (one per instrument,
    grouped by their parent signal app).

    For each signal app: ``actual`` = the newest ``as_of`` bar its instruments
    carry; ``expected`` = the freshest completed bar for its asset class right
    now.  ``fresh`` ⇔ ``actual ≥ expected`` (a partial intraday bar counts as
    fresh).  A parent listed in ``parent_order`` with NO instruments in
    ``results`` at all (its sleeve failed to load) also FAILS the audit — a
    silently-reduced universe must never publish a Target Book whose optimiser
    re-normalised over the missing app's weight.  ``expected_now`` (optional)
    evaluates the *expected* freshest bars at a different moment than the
    check itself — the publisher passes its 7:15-AM-CT anchor so a post-close
    publish still expects (and gets) the pre-anchor session, not the close
    that just landed.  Returns per-app rows plus an overall ``passed`` verdict
    and the list of stale asset keys — the exact set the Overall app must
    flag."""
    n = _as_utc(now or now_utc())
    exp_n = _as_utc(expected_now) if expected_now is not None else n
    groups: dict[str, list[dict]] = {}
    for r in results or []:
        groups.setdefault(r.get("parent", r.get("key")), []).append(r)
    order = [k for k in (parent_order or list(groups))if k in groups]
    order += [k for k in groups if k not in order]

    rows, stale_parents, stale_assets = [], [], []
    for pk in (parent_order or []):
        if pk in groups:
            continue                       # loaded fine — audited below
        kind = PARENT_CLASS.get(pk, "us_equity")
        exp = expected_asof(kind, exp_n)
        stale_parents.append(pk)
        rows.append(dict(
            app=pk, asset_class=kind, instruments=[],
            actual_asof="—", expected_asof=str(exp.date()),
            actual_close="⛔ app failed to load — no signals returned",
            expected_close=close_label(kind, exp, exp_n),
            fresh=False, age_days=0,
        ))
    for pk in order:
        grp = groups[pk]
        kind = PARENT_CLASS.get(pk, "us_equity")
        actual = max(pd.Timestamp(r["as_of"]).tz_localize(None).normalize()
                     for r in grp)
        exp = expected_asof(kind, exp_n)
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
            expected_close=close_label(kind, exp, exp_n),
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
    """The last scheduled-run audit artifact (written by the once-daily
    headless publisher, committed to the repo).  ``None`` if not present/readable."""
    try:
        return json.loads(DAILY_AUDIT_JSON.read_text())
    except Exception:
        return None


def freshest_signal_record(app_key: str, log: dict | None = None,
                           daily_audit: dict | None = None) -> dict | None:
    """The best freshness evidence for ONE signal app, merged across every
    source that can attest to when its signals were last generated:

    * the app page's own render entry in the refresh log;
    * the scheduled once-daily publisher's committed audit row
      (``data/overall/daily_audit.json``);
    * the 🧭 Overall app's live-render per-app audit rows — the Overall app
      re-runs the full engine (re-fetching every sleeve's data) on each render
      and records ``audit_rows`` under its ``OVERALL`` log entry, so its row
      for this app is exactly as authoritative as the app's own page render.

    The winner is the record with the newest ``as_of`` bar (the actual
    freshness evidence), tie-broken by the newest record time — so one source
    still sitting on an older close can never mask another source that already
    generated signals from a newer one.  This is what keeps the 🕵️ Daily Audit
    tab consistent with the Overall app after a close rolls over: the Overall
    render proves the new close's signals exist even when the individual app
    pages haven't re-rendered and the morning publisher hasn't re-run.

    Returns ``dict(source, as_of, recorded_at_utc, recorded_at_ct)`` with
    ``source`` ∈ {``app``, ``scheduled``, ``overall``} and ``as_of`` a
    normalized tz-naive Timestamp; ``None`` when no source has a usable record.
    """
    log = read_refresh_log() if log is None else (log or {})
    da = (load_daily_audit() or {}) if daily_audit is None else (daily_audit or {})
    cands: list[dict] = []

    def _add(source: str, recorded_at, recorded_ct, asof) -> None:
        if not recorded_at or not asof or str(asof).strip() in ("—", "-", ""):
            return
        try:
            rec = _as_utc(recorded_at)
            a = pd.Timestamp(asof).tz_localize(None).normalize()
        except Exception:
            return
        cands.append(dict(source=source, as_of=a, recorded_at_utc=rec,
                          recorded_at_ct=recorded_ct or fmt_ct(rec, seconds=True)))

    e = log.get(app_key) or {}
    _add("app", e.get("recorded_at_utc"), e.get("recorded_at_ct"), e.get("as_of"))

    for r in (da.get("audit") or {}).get("rows") or []:
        if r.get("app") == app_key:
            _add("scheduled", da.get("generated_at_utc"),
                 da.get("generated_at_ct"), r.get("actual_asof"))

    ov = log.get("OVERALL") or {}
    for r in ov.get("audit_rows") or []:
        if r.get("app") == app_key:
            _add("overall", ov.get("recorded_at_utc"),
                 ov.get("recorded_at_ct"), r.get("actual_asof"))

    if not cands:
        return None
    return max(cands, key=lambda c: (c["as_of"], c["recorded_at_utc"]))
