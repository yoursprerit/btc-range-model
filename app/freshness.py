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

Both are strictly additive: they never overwrite a session the primary feed has,
never add an in-progress or non-trading session, and no-op when the frame is
already current — so a genuinely stale feed still fails the audit.
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


def et_session_dates(idx) -> pd.DatetimeIndex:
    """Map intraday (tz-naive UTC) timestamps to their US/Eastern session date."""
    i = pd.DatetimeIndex(idx)
    i = i.tz_localize("UTC") if i.tz is None else i.tz_convert("UTC")
    return pd.DatetimeIndex(i.tz_convert(ET).normalize().tz_localize(None))


def backfill_sessions_from_hourly(daily: pd.DataFrame, hourly: pd.DataFrame,
                                  price_col: str, now=None) -> pd.DataFrame:
    """Append any COMPLETED US sessions the daily frame lacks, aggregated from
    ``hourly``.  No-op when the daily frame is already current, when ``hourly`` is
    empty/no fresher, or when the synthesized session has no primary-price data."""
    if daily is None or daily.empty or hourly is None or hourly.empty:
        return daily
    cutoff = expected_equity_asof(now)
    # Completed sessions only: an in-progress *today* row (which Yahoo's daily
    # feed serves during market hours even while a prior close is missing)
    # must not mask the gap — see last_completed_session.
    last = last_completed_session(daily.index, now)
    if last is not None and last >= cutoff:
        return daily                            # already current — nothing to do
    h = hourly.copy()
    h["_sess"] = et_session_dates(h.index)
    want = [d for d in sorted(set(h["_sess"]))
            if (last is None or last < d) and d <= cutoff and is_us_trading_day(d)]
    if not want:
        return daily
    agg = {}
    for c in daily.columns:
        if c not in h.columns:
            continue
        agg[c] = _AGG_BY_SUFFIX.get(str(c).rsplit("_", 1)[-1], "last")
    if price_col not in agg:
        return daily
    rows = h[h["_sess"].isin(want)].groupby("_sess").agg(agg)
    rows = rows.dropna(subset=[price_col])
    rows = rows[~rows.index.isin(daily.index)]
    if rows.empty:
        return daily
    out = pd.concat([daily, rows.reindex(columns=daily.columns)]).sort_index()
    return out[~out.index.duplicated(keep="last")]


def merge_missing_sessions(daily: pd.DataFrame, alt: pd.DataFrame, price_col: str,
                          now=None, max_rel_jump: float = 0.60) -> pd.DataFrame:
    """Append COMPLETED US sessions that ``daily`` lacks, taken from an alternate
    DAILY-indexed frame (a second provider — see ``app/market_fallback.py``).

    Used after the hourly backstop, when Yahoo is withholding sessions entirely.
    Guards, in order:

    * never touches a session ``daily`` already has, and never adds one past the
      last completed close (``expected_equity_asof``) or on a non-trading day;
    * requires the primary price column to be present and non-null;
    * **split-scale guard** — a candidate whose close differs from the newest
      known close by more than ``max_rel_jump`` is DROPPED.  The two providers
      differ in split/dividend adjustment, so this is what stops a raw,
      unadjusted print being spliced onto an adjusted series (a 10-for-1 split
      shows up as a ~90% gap; a genuine leveraged-ETF day stays well inside 60%).
    """
    if daily is None or daily.empty or alt is None or alt.empty:
        return daily
    cutoff = expected_equity_asof(now)
    # Completed sessions only, for the same reason as the hourly backstop.
    last = last_completed_session(daily.index, now)
    if last is not None and last >= cutoff:
        return daily
    a = alt.copy()
    a.index = pd.DatetimeIndex(a.index).normalize()
    a = a[~a.index.duplicated(keep="last")].sort_index()
    want = [d for d in a.index
            if (last is None or last < d) and d <= cutoff
            and d not in daily.index and is_us_trading_day(d)]
    if not want or price_col not in a.columns:
        return daily
    rows = a.loc[want].dropna(subset=[price_col])
    if rows.empty:
        return daily
    ref = pd.to_numeric(daily[price_col], errors="coerce").dropna()
    if len(ref):
        base = float(ref.iloc[-1])
        if base > 0:
            keep = (rows[price_col].astype(float) / base - 1.0).abs() <= max_rel_jump
            rows = rows[keep]
    if rows.empty:
        return daily
    out = pd.concat([daily, rows.reindex(columns=daily.columns)]).sort_index()
    return out[~out.index.duplicated(keep="last")]


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
