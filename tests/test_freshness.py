"""Unit tests for app/freshness.py — the shared signal-freshness source of truth.

Everything here is pure clock/calendar logic (no network, no Streamlit), pinned
to fixed instants so the tests are deterministic in any timezone.
"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import freshness as fr  # noqa: E402


# ── NYSE holiday calendar ────────────────────────────────────────────────────
def test_holidays_2026():
    h = fr.us_market_holidays(2026)
    assert date(2026, 1, 1) in h          # New Year's Day (Thursday)
    assert date(2026, 1, 19) in h         # MLK Day
    assert date(2026, 4, 3) in h          # Good Friday (Easter = Apr 5)
    assert date(2026, 5, 25) in h         # Memorial Day
    assert date(2026, 6, 19) in h         # Juneteenth
    assert date(2026, 7, 3) in h          # July 4 observed (Sat → Fri)
    assert date(2026, 11, 26) in h        # Thanksgiving
    assert date(2026, 12, 25) in h        # Christmas
    assert date(2026, 7, 4) not in h      # the Saturday itself is not a session anyway


def test_holidays_2025_good_friday():
    assert date(2025, 4, 18) in fr.us_market_holidays(2025)   # Easter 2025 = Apr 20


def test_trading_day():
    assert fr.is_us_trading_day("2026-07-21")        # Tuesday
    assert not fr.is_us_trading_day("2026-07-19")    # Sunday
    assert not fr.is_us_trading_day("2026-07-03")    # July-4th observed


# ── expected freshest bars ───────────────────────────────────────────────────
def test_expected_crypto_asof_rolls_at_12utc():
    # Bar D covers [D 12:00, D+1 12:00) UTC; as_of is the bar START date.
    before = pd.Timestamp("2026-07-21T11:59:00Z")
    after = pd.Timestamp("2026-07-21T12:01:00Z")
    assert fr.expected_crypto_asof(before) == pd.Timestamp("2026-07-19")
    assert fr.expected_crypto_asof(after) == pd.Timestamp("2026-07-20")


def test_expected_equity_asof_rolls_at_4pm_et():
    # Tue Jul 21 2026: before 4:00 PM ET the freshest completed close is Monday's.
    before = pd.Timestamp("2026-07-21T19:59:00Z")   # 3:59 PM EDT
    after = pd.Timestamp("2026-07-21T20:01:00Z")    # 4:01 PM EDT
    assert fr.expected_equity_asof(before) == pd.Timestamp("2026-07-20")
    assert fr.expected_equity_asof(after) == pd.Timestamp("2026-07-21")


def test_expected_equity_asof_skips_weekend_and_holiday():
    # Mon Jul 6 2026, pre-open: Fri Jul 3 was the July-4th closure → Thu Jul 2.
    n = pd.Timestamp("2026-07-06T12:30:00Z")
    assert fr.expected_equity_asof(n) == pd.Timestamp("2026-07-02")
    # Sunday: freshest completed close is Friday's (a normal Friday).
    n = pd.Timestamp("2026-07-19T15:00:00Z")
    assert fr.expected_equity_asof(n) == pd.Timestamp("2026-07-17")


def test_close_moment_conventions():
    # crypto bar starting Jul 20 closes Jul 21 12:00 UTC
    cm = fr.close_moment("crypto", "2026-07-20")
    assert cm == pd.Timestamp("2026-07-21T12:00:00Z")
    # equity session Jul 20 closes 4 PM ET = 20:00 UTC in July (EDT)
    cm = fr.close_moment("us_equity", "2026-07-20")
    assert cm == pd.Timestamp("2026-07-20T20:00:00Z")


def test_close_label_mentions_ct_and_et():
    assert "7:00 AM CDT" in fr.close_label("crypto", "2026-07-20",
                                           now="2026-07-21T13:00:00Z")
    assert "4:00 PM EDT" in fr.close_label("us_equity", "2026-07-20",
                                           now="2026-07-21T13:00:00Z")


def test_close_label_flags_in_progress_bar():
    lbl = fr.close_label("us_equity", "2026-07-21", now="2026-07-21T15:00:00Z")
    assert "in progress" in lbl


# ── the audit ────────────────────────────────────────────────────────────────
def _mk(parent, key, asof):
    return dict(parent=parent, key=key, as_of=asof)


def test_audit_passes_when_everything_fresh():
    now = "2026-07-21T12:20:00Z"      # just after the BTC bar close, pre-open
    res = [_mk("BTC", "BTC", "2026-07-20"), _mk("BTC", "MSTR", "2026-07-20"),
           _mk("SOXX", "SOXX", "2026-07-20"), _mk("GLDM", "GLDM", "2026-07-20")]
    aud = fr.audit_universe(res, now=now, parent_order=["BTC", "GLDM", "SOXX"])
    assert aud["passed"] and not aud["stale_assets"]
    assert [r["app"] for r in aud["rows"]] == ["BTC", "GLDM", "SOXX"]


def test_audit_flags_stale_app_and_its_instruments():
    now = "2026-07-21T12:20:00Z"
    res = [_mk("BTC", "BTC", "2026-07-18"), _mk("BTC", "MSTU", "2026-07-18"),
           _mk("SOXX", "SOXX", "2026-07-20")]
    aud = fr.audit_universe(res, now=now)
    assert not aud["passed"]
    assert aud["stale_apps"] == ["BTC"]
    assert set(aud["stale_assets"]) == {"BTC", "MSTU"}
    btc = next(r for r in aud["rows"] if r["app"] == "BTC")
    assert btc["age_days"] == 2 and not btc["fresh"]


def test_audit_accepts_fresher_than_expected_partial_bar():
    # intraday: today's in-progress equity bar (as_of == today) is fresh
    now = "2026-07-21T15:00:00Z"
    aud = fr.audit_universe([_mk("SOXX", "SOXX", "2026-07-21")], now=now)
    assert aud["passed"]


def test_audit_fails_when_a_signal_app_is_missing_entirely():
    """A parent listed in parent_order with NO instruments in results (its
    sleeve failed to load) must fail the audit — a reduced universe would
    publish a book whose optimiser re-normalised over the missing weight."""
    now = "2026-07-21T12:20:00Z"
    res = [_mk("BTC", "BTC", "2026-07-20"), _mk("SOXX", "SOXX", "2026-07-20")]
    aud = fr.audit_universe(res, now=now, parent_order=["BTC", "GRID", "SOXX"])
    assert not aud["passed"]
    assert "GRID" in aud["stale_apps"]
    grid = next(r for r in aud["rows"] if r["app"] == "GRID")
    assert not grid["fresh"] and grid["instruments"] == []
    assert "failed to load" in grid["actual_close"]


# ── completed-bars-only mode (the publisher's close-based book) ──────────────
def test_drop_in_progress_us_bar_trims_partial_today():
    # Tue Jul 21 2026, 11 AM ET: today's bar is in progress → dropped;
    # Monday's completed close stays.
    now = pd.Timestamp("2026-07-21T15:00:00Z")
    df = pd.DataFrame({"px_close": [1.0, 2.0, 3.0]},
                      index=pd.to_datetime(["2026-07-17", "2026-07-20",
                                            "2026-07-21"]))
    out = fr.drop_in_progress_us_bar(df, now)
    assert list(out.index) == list(pd.to_datetime(["2026-07-17", "2026-07-20"]))


def test_drop_in_progress_us_bar_keeps_all_after_close():
    now = pd.Timestamp("2026-07-21T20:30:00Z")     # 4:30 PM EDT — today closed
    df = pd.DataFrame({"px_close": [1.0, 2.0]},
                      index=pd.to_datetime(["2026-07-20", "2026-07-21"]))
    assert len(fr.drop_in_progress_us_bar(df, now)) == 2


def test_publish_anchor_is_715_ct_all_day():
    """Any wall-clock moment of a CT day maps to that day's 7:15 AM CT anchor
    (12:15 UTC during CDT)."""
    anchor = pd.Timestamp("2026-07-21T12:15:00Z")
    for now in ("2026-07-21T12:20:00Z",    # right after the morning cycle
                "2026-07-21T18:00:00Z",    # mid-session
                "2026-07-21T22:00:00Z",    # after the 4 PM ET close
                "2026-07-22T02:00:00Z"):   # 9 PM CDT — still the same CT day
        assert fr.publish_anchor_ct(now) == anchor, now


def test_publish_pending_states():
    """Pending ⇔ now is past today's 7:15-AM-CT anchor AND the book predates
    it; before the anchor (or with today's book / bad input) → not pending."""
    yday_book = "2026-07-29T14:30:11+00:00"          # yesterday's publish
    # 8:09 AM CDT, past today's 12:15 UTC anchor, book still yesterday's → pending
    assert fr.publish_pending(yday_book, now="2026-07-30T13:09:00Z")
    # 6:00 AM CDT — before today's anchor, yesterday's book is the expected one
    assert not fr.publish_pending(yday_book, now="2026-07-30T11:00:00Z")
    # today's book already landed → not pending
    assert not fr.publish_pending("2026-07-30T12:20:00+00:00",
                                  now="2026-07-30T13:09:00Z")
    # missing / unparsable stamps never warn
    assert not fr.publish_pending(None, now="2026-07-30T13:09:00Z")
    assert not fr.publish_pending("not-a-date", now="2026-07-30T13:09:00Z")


def test_anchor_trim_excludes_same_day_close_after_market():
    """Publishing AFTER the 4 PM ET close must still exclude today's close —
    post-close signal changes belong only to the live view."""
    now = pd.Timestamp("2026-07-21T21:30:00Z")      # 5:30 PM EDT, market closed
    df = pd.DataFrame({"px_close": [1.0, 2.0]},
                      index=pd.to_datetime(["2026-07-20", "2026-07-21"]))
    out = fr.drop_in_progress_us_bar(df, fr.publish_anchor_ct(now))
    assert list(out.index) == [pd.Timestamp("2026-07-20")]


def test_audit_expected_now_anchors_post_close_publish():
    """After the close, plain now expects today's bar (stale verdict for a
    yesterday-close book) but expected_now=anchor accepts it — and the anchor
    still demands the day's completed BTC bar."""
    now = "2026-07-21T21:30:00Z"                    # post-close
    anchor = fr.publish_anchor_ct(now)
    res = [_mk("SOXX", "SOXX", "2026-07-20"), _mk("BTC", "BTC", "2026-07-20")]
    assert not fr.audit_universe(res, now=now)["passed"]
    aud = fr.audit_universe(res, now=now, expected_now=anchor)
    assert aud["passed"]
    stale_btc = [_mk("BTC", "BTC", "2026-07-19")]   # yesterday's BTC bar → stale
    assert not fr.audit_universe(stale_btc, now=now, expected_now=anchor)["passed"]


def test_completed_bars_only_env_flag(monkeypatch):
    monkeypatch.delenv(fr.COMPLETED_BARS_ENV, raising=False)
    assert not fr.completed_bars_only()
    monkeypatch.setenv(fr.COMPLETED_BARS_ENV, "1")
    assert fr.completed_bars_only()
    monkeypatch.setenv(fr.COMPLETED_BARS_ENV, "0")
    assert not fr.completed_bars_only()


def test_audit_before_btc_bar_close_expects_two_day_old_start():
    # 11:00 UTC: the freshest completed BTC bar STARTED two days ago.
    now = "2026-07-21T11:00:00Z"
    aud = fr.audit_universe([_mk("BTC", "BTC", "2026-07-19")], now=now)
    assert aud["passed"]
    aud = fr.audit_universe([_mk("BTC", "BTC", "2026-07-18")], now=now)
    assert not aud["passed"]


# ── refresh log round-trip ───────────────────────────────────────────────────
def test_record_refresh_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(fr, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(fr, "REFRESH_LOG", tmp_path / "freshness_log.json")
    fr.record_refresh("SOXX", kind="us_equity", as_of="2026-07-20")
    fr.record_refresh("BTC", kind="crypto", as_of="2026-07-20")
    log = fr.read_refresh_log()
    assert log["SOXX"]["as_of"] == "2026-07-20"
    assert log["BTC"]["kind"] == "crypto"
    assert "recorded_at_utc" in log["SOXX"]


# ── freshest_signal_record — the Daily Audit page's per-app source merge ─────
def test_freshest_record_overall_render_wins_after_close_rolls_over():
    """Regression: after the 4 PM ET close the morning scheduled run and the
    app's own page render both sit on yesterday's close, but the Overall app
    has re-rendered — re-running the whole engine — and audited the app on
    TODAY's close.  The Daily Audit tab must use that newest evidence, so it
    can never show 🚨 STALE while the Overall app shows ✅ Fresh."""
    log = {"SOXX": {"as_of": "2026-07-30",
                    "recorded_at_utc": "2026-07-31T14:00:00+00:00",
                    "recorded_at_ct": "Jul 31, 2026 at 9:00:00 AM CDT"},
           "OVERALL": {"recorded_at_utc": "2026-07-31T21:10:00+00:00",
                       "recorded_at_ct": "Jul 31, 2026 at 4:10:00 PM CDT",
                       "audit_rows": [
                           {"app": "SOXX", "actual_asof": "2026-07-31"},
                           {"app": "GRID", "actual_asof": "2026-07-31"}]}}
    da = {"generated_at_utc": "2026-07-31T13:05:03+00:00",
          "generated_at_ct": "Jul 31, 2026 at 8:05:03 AM CDT",
          "audit": {"rows": [{"app": "SOXX", "actual_asof": "2026-07-30"}]}}
    best = fr.freshest_signal_record("SOXX", log=log, daily_audit=da)
    assert best["source"] == "overall"
    assert best["as_of"] == pd.Timestamp("2026-07-31")
    # 5:10 PM CDT, past the close: today's bar is the expected one → fresh
    assert best["as_of"] >= fr.expected_asof("us_equity", "2026-07-31T22:10:00Z")
    # an app only present in the Overall render's audit rows still resolves
    grid = fr.freshest_signal_record("GRID", log=log, daily_audit=da)
    assert grid["source"] == "overall" and grid["as_of"] == pd.Timestamp("2026-07-31")


def test_freshest_record_ties_broken_by_record_time():
    # Same as_of everywhere → the most recent record wins (scheduled run here).
    log = {"SOXX": {"as_of": "2026-07-30",
                    "recorded_at_utc": "2026-07-31T09:00:00+00:00",
                    "recorded_at_ct": "early"}}
    da = {"generated_at_utc": "2026-07-31T13:05:03+00:00",
          "generated_at_ct": "later",
          "audit": {"rows": [{"app": "SOXX", "actual_asof": "2026-07-30"}]}}
    best = fr.freshest_signal_record("SOXX", log=log, daily_audit=da)
    assert best["source"] == "scheduled"
    assert best["recorded_at_ct"] == "later"


def test_freshest_record_newer_asof_beats_newer_record_time():
    # A later page render on a rate-limited feed (older bar) must not mask the
    # scheduled run that already generated signals from the newer close.
    log = {"SOXX": {"as_of": "2026-07-29",
                    "recorded_at_utc": "2026-07-31T20:00:00+00:00",
                    "recorded_at_ct": "later-but-staler"}}
    da = {"generated_at_utc": "2026-07-31T13:05:03+00:00",
          "generated_at_ct": "morning run",
          "audit": {"rows": [{"app": "SOXX", "actual_asof": "2026-07-30"}]}}
    best = fr.freshest_signal_record("SOXX", log=log, daily_audit=da)
    assert best["source"] == "scheduled"
    assert best["as_of"] == pd.Timestamp("2026-07-30")


def test_freshest_record_degrades_safely():
    # no sources at all
    assert fr.freshest_signal_record("SOXX", log={}, daily_audit={}) is None
    # failed-to-load rows ("—") and unparsable stamps are skipped, not crashed on
    log = {"OVERALL": {"recorded_at_utc": "2026-07-31T21:10:00+00:00",
                       "audit_rows": [{"app": "GRID", "actual_asof": "—"}]}}
    da = {"generated_at_utc": "not-a-date",
          "audit": {"rows": [{"app": "GRID", "actual_asof": "2026-07-30"}]}}
    assert fr.freshest_signal_record("GRID", log=log, daily_audit=da) is None
    # a lone valid app entry still resolves (and fills recorded_at_ct itself)
    log = {"GRID": {"as_of": "2026-07-30",
                    "recorded_at_utc": "2026-07-31T14:00:00+00:00"}}
    best = fr.freshest_signal_record("GRID", log=log, daily_audit={})
    assert best["source"] == "app" and "2026" in best["recorded_at_ct"]


def test_record_refresh_never_raises(monkeypatch, tmp_path):
    # unwritable dir → still no exception (best-effort by design)
    monkeypatch.setattr(fr, "RUNTIME_DIR", tmp_path / "no" / "such" / "\0bad")
    fr.record_refresh("X", foo=1)   # must not raise


# ── hourly → daily backstop (Yahoo's daily feed lagging its intraday feed) ───
def _hourly_session(day: str, closes, *, tz_hours=(14, 15, 16, 20)):
    """A few tz-naive-UTC hourly bars inside one US session."""
    idx = [pd.Timestamp(f"{day}T{h:02d}:00:00") for h in tz_hours]
    n = len(idx)
    return pd.DataFrame(
        {"px_open": closes[:n], "px_high": [c + 1 for c in closes[:n]],
         "px_low": [c - 1 for c in closes[:n]], "px_close": closes[:n],
         "px_volume": [10] * n, "vix_close": [20] * n},
        index=pd.DatetimeIndex(idx))


def _daily(days, closes):
    return pd.DataFrame(
        {"px_open": closes, "px_high": [c + 2 for c in closes],
         "px_low": [c - 2 for c in closes], "px_close": closes,
         "px_volume": [100] * len(days), "vix_close": [21] * len(days)},
        index=pd.DatetimeIndex([pd.Timestamp(d) for d in days]))


def test_backfill_rebuilds_the_session_the_daily_feed_is_missing():
    # Sat Jul 25 → last completed session is Fri Jul 24, but daily stops Jul 23.
    daily = _daily(["2026-07-22", "2026-07-23"], [100.0, 101.0])
    hourly = _hourly_session("2026-07-24", [102.0, 103.0, 104.0, 105.5])
    out = fr.backfill_sessions_from_hourly(daily, hourly, "px_close",
                                           now="2026-07-25T14:00:00Z")
    assert str(out.index.max().date()) == "2026-07-24"
    row = out.loc[pd.Timestamp("2026-07-24")]
    assert row["px_close"] == 105.5          # last hourly close of the session
    assert row["px_open"] == 102.0           # first
    assert row["px_high"] == 106.5           # max of highs
    assert row["px_low"] == 101.0            # min of lows
    assert row["px_volume"] == 40            # summed
    assert list(out.columns) == list(daily.columns)


def test_backfill_is_noop_when_daily_already_current():
    daily = _daily(["2026-07-23", "2026-07-24"], [100.0, 101.0])
    hourly = _hourly_session("2026-07-24", [9.0, 9.0, 9.0, 9.0])
    out = fr.backfill_sessions_from_hourly(daily, hourly, "px_close",
                                           now="2026-07-25T14:00:00Z")
    assert len(out) == 2
    assert out.loc[pd.Timestamp("2026-07-24"), "px_close"] == 101.0  # not overwritten


def test_backfill_never_invents_an_incomplete_or_nontrading_session():
    daily = _daily(["2026-07-23", "2026-07-24"], [100.0, 101.0])
    # Sat Jul 25 has hourly ticks but is not a trading day, and Mon Jul 27 has
    # not closed yet at Mon 14:00 UTC (10:00 ET) — neither may be appended.
    hourly = pd.concat([_hourly_session("2026-07-25", [1.0, 1.0, 1.0, 1.0]),
                        _hourly_session("2026-07-27", [2.0, 2.0, 2.0, 2.0])])
    out = fr.backfill_sessions_from_hourly(daily, hourly, "px_close",
                                           now="2026-07-27T14:00:00Z")
    assert str(out.index.max().date()) == "2026-07-24"


def test_backfill_degrades_safely():
    daily = _daily(["2026-07-22", "2026-07-23"], [100.0, 101.0])
    now = "2026-07-25T14:00:00Z"
    # empty / None hourly, and hourly with no usable price column → unchanged
    assert len(fr.backfill_sessions_from_hourly(daily, pd.DataFrame(), "px_close", now=now)) == 2
    assert fr.backfill_sessions_from_hourly(daily, None, "px_close", now=now) is daily
    assert fr.backfill_sessions_from_hourly(None, pd.DataFrame(), "px_close", now=now) is None
    bad = _hourly_session("2026-07-24", [1.0, 1.0, 1.0, 1.0]).rename(columns={"px_close": "other"})
    assert len(fr.backfill_sessions_from_hourly(daily, bad, "px_close", now=now)) == 2


def test_last_completed_session_ignores_partial_today_row():
    # Mon Jul 27 2026, 11:30 ET (market open): the in-progress Monday row must
    # not count — the newest COMPLETED session present is Thu Jul 23.
    now = "2026-07-27T15:30:00Z"
    idx = pd.to_datetime(["2026-07-22", "2026-07-23", "2026-07-27"])
    assert fr.last_completed_session(idx, now) == pd.Timestamp("2026-07-23")
    # degenerate inputs → None
    assert fr.last_completed_session(pd.DatetimeIndex([]), now) is None
    assert fr.last_completed_session(["not-a-date"], now) is None
    # only post-cutoff rows present → None
    assert fr.last_completed_session(pd.to_datetime(["2026-07-27"]), now) is None


def test_backfill_recovers_session_masked_by_partial_today_row():
    """Regression for 2026-07-27: during Monday market hours Yahoo's daily feed
    served an in-progress Mon row while Friday's completed close was still
    missing.  index.max() (= Mon) masked the gap, the backstops never fired,
    and the publish audit failed all day.  The gap must be detected and Friday
    rebuilt from hourly, leaving the partial Monday row untouched."""
    now = "2026-07-27T15:30:00Z"                     # Mon 11:30 ET, market open
    daily = _daily(["2026-07-22", "2026-07-23", "2026-07-27"],
                   [100.0, 101.0, 103.0])
    hourly = _hourly_session("2026-07-24", [102.0, 103.0, 104.0, 105.5])
    out = fr.backfill_sessions_from_hourly(daily, hourly, "px_close", now=now)
    assert pd.Timestamp("2026-07-24") in out.index
    assert out.loc[pd.Timestamp("2026-07-24"), "px_close"] == 105.5
    assert out.loc[pd.Timestamp("2026-07-27"), "px_close"] == 103.0  # kept
    assert list(out.index) == sorted(out.index)


def test_backfill_at_anchor_repairs_basis_session_after_close():
    # Post-close catch-up publish (Mon 5:30 PM ET): the frame has Monday's
    # completed row but Friday is still missing.  Evaluated at the 7:15-AM-CT
    # anchor (the publisher's basis), Friday must still be rebuilt.
    anchor = fr.publish_anchor_ct("2026-07-27T22:30:00Z")
    daily = _daily(["2026-07-23", "2026-07-27"], [100.0, 103.0])
    hourly = _hourly_session("2026-07-24", [102.0, 103.0, 104.0, 105.0])
    out = fr.backfill_sessions_from_hourly(daily, hourly, "px_close", now=anchor)
    assert pd.Timestamp("2026-07-24") in out.index
    assert out.loc[pd.Timestamp("2026-07-24"), "px_close"] == 105.0


def test_et_session_dates_maps_utc_ticks_to_the_right_session():
    # 20:00 UTC = 16:00 ET (same day); 00:30 UTC = 20:30 ET the PREVIOUS day.
    idx = pd.DatetimeIndex([pd.Timestamp("2026-07-24T20:00:00"),
                            pd.Timestamp("2026-07-25T00:30:00")])
    got = [str(d.date()) for d in fr.et_session_dates(idx)]
    assert got == ["2026-07-24", "2026-07-24"]


# ── secondary-provider backfill (Nasdaq fallback when Yahoo withholds a bar) ──
def test_merge_missing_sessions_recovers_a_withheld_session():
    daily = _daily(["2026-07-22", "2026-07-23"], [100.0, 101.0])
    alt = _daily(["2026-07-23", "2026-07-24"], [101.0, 104.0])   # 2nd provider
    out = fr.merge_missing_sessions(daily, alt, "px_close", now="2026-07-25T14:00:00Z")
    assert str(out.index.max().date()) == "2026-07-24"
    assert out.loc[pd.Timestamp("2026-07-24"), "px_close"] == 104.0
    # the session both providers had is NOT overwritten by the fallback
    assert out.loc[pd.Timestamp("2026-07-23"), "px_close"] == 101.0


def test_merge_missing_sessions_split_scale_guard():
    # A raw/unadjusted print (10x the adjusted level) must be refused, since the
    # two providers differ in split adjustment.
    daily = _daily(["2026-07-22", "2026-07-23"], [100.0, 101.0])
    alt = _daily(["2026-07-24"], [1010.0])
    out = fr.merge_missing_sessions(daily, alt, "px_close", now="2026-07-25T14:00:00Z")
    assert str(out.index.max().date()) == "2026-07-23"      # rejected
    # a plausible move on the same day IS accepted
    ok = fr.merge_missing_sessions(daily, _daily(["2026-07-24"], [120.0]),
                                   "px_close", now="2026-07-25T14:00:00Z")
    assert ok.loc[pd.Timestamp("2026-07-24"), "px_close"] == 120.0


def test_merge_missing_sessions_respects_cutoff_and_calendar():
    daily = _daily(["2026-07-23", "2026-07-24"], [100.0, 101.0])
    # Sat Jul 25 is not a session; Mon Jul 27 has not closed at 14:00 UTC.
    alt = _daily(["2026-07-25", "2026-07-27"], [102.0, 103.0])
    out = fr.merge_missing_sessions(daily, alt, "px_close", now="2026-07-27T14:00:00Z")
    assert str(out.index.max().date()) == "2026-07-24"


def test_merge_recovers_session_masked_by_partial_today_row():
    # Same 2026-07-27 regression as the hourly backstop, on the second-provider
    # path: the partial Monday row must not stop Friday being spliced in.
    now = "2026-07-27T15:30:00Z"
    daily = _daily(["2026-07-22", "2026-07-23", "2026-07-27"],
                   [100.0, 101.0, 103.0])
    alt = _daily(["2026-07-24"], [104.0])
    out = fr.merge_missing_sessions(daily, alt, "px_close", now=now)
    assert out.loc[pd.Timestamp("2026-07-24"), "px_close"] == 104.0
    assert out.loc[pd.Timestamp("2026-07-27"), "px_close"] == 103.0  # kept


def test_merge_missing_sessions_degrades_safely():
    daily = _daily(["2026-07-22", "2026-07-23"], [100.0, 101.0])
    now = "2026-07-25T14:00:00Z"
    assert fr.merge_missing_sessions(daily, None, "px_close", now=now) is daily
    assert len(fr.merge_missing_sessions(daily, pd.DataFrame(), "px_close", now=now)) == 2
    alt = _daily(["2026-07-24"], [104.0]).rename(columns={"px_close": "nope"})
    assert len(fr.merge_missing_sessions(daily, alt, "px_close", now=now)) == 2


# ── interior-gap repair (a session the feed WITHDRAWS, not one it lags on) ───
def _sib(days, px, sib, sib_name="soxl"):
    """Daily frame with a primary + one traded sibling + one macro column."""
    return pd.DataFrame(
        {"px_open": px, "px_high": [c + 2 for c in px], "px_low": [c - 2 for c in px],
         "px_close": px, "px_volume": [100] * len(days),
         f"{sib_name}_open": sib, f"{sib_name}_high": sib, f"{sib_name}_low": sib,
         f"{sib_name}_close": sib, f"{sib_name}_volume": [10] * len(days),
         "vix_close": [21] * len(days)},
        index=pd.DatetimeIndex([pd.Timestamp(d) for d in days]))


def test_missing_sessions_finds_an_interior_hole():
    """Regression for 2026-09-01.  Yahoo withdrew the already-published
    2026-08-28 close (served a null; `_chart` drops null-close rows) while still
    serving 08-31, so the frame's newest session was perfectly current and the
    hole sat INSIDE the history.  The old tail test
    (`last_completed_session < expected_equity_asof`) reported "nothing to do",
    every backstop no-oped, and the gate then refused the frame forever."""
    now = "2026-09-01T13:00:00Z"                 # Tue, pre-open; expected = Mon 08-31
    idx = pd.to_datetime(["2026-08-26", "2026-08-27", "2026-08-31"])
    assert fr.expected_equity_asof(now) == pd.Timestamp("2026-08-31")
    # the tail looks entirely healthy — this is exactly what fooled the backstops
    assert fr.last_completed_session(idx, now) == pd.Timestamp("2026-08-31")
    assert fr.missing_sessions(idx, now) == [pd.Timestamp("2026-08-28")]


def test_missing_sessions_is_empty_on_a_complete_frame():
    now = "2026-09-01T13:00:00Z"
    idx = pd.to_datetime(["2026-08-26", "2026-08-27", "2026-08-28", "2026-08-31"])
    assert fr.missing_sessions(idx, now) == []
    # never demands history older than the frame itself, nor a future session
    assert fr.missing_sessions(pd.to_datetime(["2026-08-31"]), now) == []
    assert fr.missing_sessions(pd.DatetimeIndex([]), now) == []


def test_backfill_repairs_an_interior_hole_from_hourly():
    now = "2026-09-01T13:00:00Z"
    daily = _daily(["2026-08-26", "2026-08-27", "2026-08-31"], [100.0, 101.0, 103.0])
    hourly = _hourly_session("2026-08-28", [102.0, 103.0, 104.0, 105.5])
    out = fr.backfill_sessions_from_hourly(daily, hourly, "px_close", now=now)
    assert out.loc[pd.Timestamp("2026-08-28"), "px_close"] == 105.5
    assert out.loc[pd.Timestamp("2026-08-31"), "px_close"] == 103.0   # untouched
    assert list(out.index) == sorted(out.index)


def test_merge_repairs_an_interior_hole_from_second_provider():
    now = "2026-09-01T13:00:00Z"
    daily = _daily(["2026-08-26", "2026-08-27", "2026-08-31"], [100.0, 101.0, 103.0])
    alt = _daily(["2026-08-28"], [102.5])
    out = fr.merge_missing_sessions(daily, alt, "px_close", now=now)
    assert out.loc[pd.Timestamp("2026-08-28"), "px_close"] == 102.5
    assert out.loc[pd.Timestamp("2026-08-31"), "px_close"] == 103.0


def test_interior_split_scale_guard_uses_the_nearest_prior_close():
    """An interior candidate is judged against the level that prevailed AROUND
    it, not against the frame's final close several sessions later."""
    now = "2026-09-01T13:00:00Z"
    daily = _daily(["2026-08-26", "2026-08-27", "2026-08-31"], [100.0, 101.0, 103.0])
    # a raw/unadjusted print ~10x the prevailing level is refused
    out = fr.merge_missing_sessions(daily, _daily(["2026-08-28"], [1010.0]),
                                    "px_close", now=now)
    assert pd.Timestamp("2026-08-28") not in out.index
    # a plausible one is accepted
    ok = fr.merge_missing_sessions(daily, _daily(["2026-08-28"], [102.0]),
                                   "px_close", now=now)
    assert ok.loc[pd.Timestamp("2026-08-28"), "px_close"] == 102.0


# ── traded-sibling close repair (never forward-fill an execution price) ──────
def test_missing_closes_spots_a_withheld_sibling_close():
    """2026-09-01: OIH/ERX had a null 08-28 close while XLE (the primary) did
    not, so the session row survived and only the sibling cell was empty — the
    row-level helpers cannot see that at all."""
    now = "2026-09-01T13:00:00Z"
    df = _sib(["2026-08-27", "2026-08-28", "2026-08-31"],
              [101.0, 102.0, 103.0], [50.0, float("nan"), 52.0])
    assert fr.missing_sessions(df.index, now) == []          # no session missing
    assert fr.missing_closes(df, ["soxl_close"], now) == {
        "soxl_close": [pd.Timestamp("2026-08-28")]}


def test_repair_missing_closes_fills_the_whole_bar_from_the_alt_feed():
    now = "2026-09-01T13:00:00Z"
    df = _sib(["2026-08-27", "2026-08-28", "2026-08-31"],
              [101.0, 102.0, 103.0], [50.0, float("nan"), 52.0])
    alt = _sib(["2026-08-28"], [102.0], [51.5])
    out = fr.repair_missing_closes(df, alt, ["soxl_close"], now=now)
    assert out.loc[pd.Timestamp("2026-08-28"), "soxl_close"] == 51.5
    assert out.loc[pd.Timestamp("2026-08-28"), "soxl_open"] == 51.5   # bar stays whole
    assert out.loc[pd.Timestamp("2026-08-31"), "soxl_close"] == 52.0  # untouched


def test_repair_missing_closes_refuses_a_raw_unadjusted_print():
    now = "2026-09-01T13:00:00Z"
    df = _sib(["2026-08-27", "2026-08-28", "2026-08-31"],
              [101.0, 102.0, 103.0], [50.0, float("nan"), 52.0])
    out = fr.repair_missing_closes(df, _sib(["2026-08-28"], [102.0], [500.0]),
                                   ["soxl_close"], now=now)
    assert pd.isna(out.loc[pd.Timestamp("2026-08-28"), "soxl_close"])


def test_repair_daily_frame_ffills_macro_but_never_a_traded_close():
    """The bug this closes: traded siblings were lumped in with the macro
    columns and forward-filled, so OIH/ERX carried Thursday's close as Friday's
    — a fabricated execution price that the data gate cannot detect (a 0% move
    is perfectly plausible) and that any refresh landing during the withdrawal
    would have pinned."""
    now = "2026-09-01T13:00:00Z"
    df = _sib(["2026-08-27", "2026-08-28", "2026-08-31"],
              [101.0, 102.0, 103.0], [50.0, float("nan"), 52.0])
    df.loc[pd.Timestamp("2026-08-28"), "vix_close"] = float("nan")
    out = fr.repair_daily_frame(
        df, "px_close", traded_cols=["soxl_close"], macro_cols=["vix_close"],
        fetch_hourly=None, fetch_alt=None, now=now)
    # macro gaps are still bridged …
    assert out.loc[pd.Timestamp("2026-08-28"), "vix_close"] == 21.0
    # … but an unrepairable traded close is left NULL for the gate to judge,
    # never invented from the prior session
    assert pd.isna(out.loc[pd.Timestamp("2026-08-28"), "soxl_close"])


def test_repair_daily_frame_uses_hourly_then_the_second_provider():
    now = "2026-09-01T13:00:00Z"
    daily = _daily(["2026-08-26", "2026-08-27", "2026-08-31"], [100.0, 101.0, 103.0])
    calls = []

    def _hourly():
        calls.append("hourly")
        return _hourly_session("2026-08-28", [102.0, 103.0, 104.0, 105.5])

    def _alt(since):
        calls.append(("alt", since))
        return _daily(["2026-08-28"], [99.0])

    out = fr.repair_daily_frame(daily, "px_close", macro_cols=["vix_close"],
                                fetch_hourly=_hourly, fetch_alt=_alt, now=now)
    assert out.loc[pd.Timestamp("2026-08-28"), "px_close"] == 105.5
    assert calls == ["hourly"]          # hourly sufficed — no second-provider hit


def test_repair_daily_frame_makes_no_network_call_on_a_healthy_frame():
    now = "2026-09-01T13:00:00Z"
    daily = _daily(["2026-08-26", "2026-08-27", "2026-08-28", "2026-08-31"],
                   [100.0, 101.0, 102.0, 103.0])

    def _boom(*a, **k):
        raise AssertionError("must not fetch when nothing is missing")

    out = fr.repair_daily_frame(daily, "px_close", macro_cols=["vix_close"],
                                fetch_hourly=_boom, fetch_alt=_boom, now=now)
    assert len(out) == 4


def test_repair_daily_frame_degrades_safely_on_a_failing_source():
    now = "2026-09-01T13:00:00Z"
    daily = _daily(["2026-08-26", "2026-08-27", "2026-08-31"], [100.0, 101.0, 103.0])

    def _boom(*a, **k):
        raise RuntimeError("feed down")

    out = fr.repair_daily_frame(daily, "px_close", macro_cols=["vix_close"],
                                fetch_hourly=_boom, fetch_alt=_boom, now=now)
    assert len(out) == 3                       # unrepaired, but never raised
    assert fr.missing_sessions(out.index, now) == [pd.Timestamp("2026-08-28")]


# ── pending "next bar" execution moment (next_session_date / next_close_label) ─
def test_next_session_date_skips_weekends_and_holidays():
    # Fri Jul 2 2026 → Mon Jul 6 (Jul 3 is the observed July-4th closure)
    assert fr.next_session_date("us_equity", "2026-07-02") == \
        pd.Timestamp("2026-07-06")
    # plain weekday → next weekday
    assert fr.next_session_date("us_equity", "2026-08-03") == \
        pd.Timestamp("2026-08-04")
    # crypto bars run every calendar day
    assert fr.next_session_date("crypto", "2026-08-01") == \
        pd.Timestamp("2026-08-02")


def test_next_close_label_says_today_while_that_session_is_underway():
    # signal committed at Mon Aug 3's close; Tue Aug 4 10:30 ET mid-session —
    # the pending exit sells at TODAY's 4 PM close, and the label says so
    lbl = fr.next_close_label("us_equity", "2026-08-03",
                              now="2026-08-04T14:30:00Z")
    assert lbl.startswith("today, ")
    assert "Aug 4" in lbl and "4:00 PM" in lbl


def test_next_close_label_says_tomorrow_after_the_close():
    # looking on Mon Aug 3 after its close: the next bar is tomorrow's session
    lbl = fr.next_close_label("us_equity", "2026-08-03",
                              now="2026-08-03T22:00:00Z")
    assert lbl.startswith("tomorrow, ")
    assert "Aug 4" in lbl


def test_next_close_label_crypto_uses_ct_bar_close():
    # crypto bar D closes D+1 12:00 UTC → next bar (D+1) closes D+2 12:00 UTC
    lbl = fr.next_close_label("crypto", "2026-08-01",
                              now="2026-08-02T18:00:00Z")
    assert "Aug 3" in lbl and "7:00 AM" in lbl


def test_exit_execution_note_past_tense_after_the_afternoon_rebalance():
    # exit committed Mon Aug 3; Tue Aug 4 14:45 CT (19:45 UTC) — the 2:30 PM CT
    # rebalance has run: the executor has SOLD, the engine books today's close
    note = fr.exit_execution_note("us_equity", "2026-08-03",
                                  now="2026-08-04T19:45:00Z")
    assert note.startswith("executor sold at today's ≈2:30 PM CT rebalance")
    assert "today, Aug 4, 4:00 PM" in note


def test_exit_execution_note_future_tense_before_the_rebalance():
    # Tue Aug 4 10:00 CT (15:00 UTC) — before the 2:30 PM CT rebalance
    note = fr.exit_execution_note("us_equity", "2026-08-03",
                                  now="2026-08-04T15:00:00Z")
    assert note.startswith("executor sells at today's ≈2:30 PM CT rebalance")


def test_exit_execution_note_tomorrow_after_the_signal_close():
    # Mon Aug 3 evening: the pending session is tomorrow
    note = fr.exit_execution_note("us_equity", "2026-08-03",
                                  now="2026-08-03T22:00:00Z")
    assert note.startswith("executor sells at tomorrow's ≈2:30 PM CT rebalance")
    assert "tomorrow, Aug 4, 4:00 PM" in note


def test_exit_execution_note_crypto_keeps_bar_close_only():
    note = fr.exit_execution_note("crypto", "2026-08-01",
                                  now="2026-08-02T18:00:00Z")
    assert note.startswith("engine books the exit at the bar close")


def test_rebalance_label_matches_the_scheduled_slot():
    # the UI copy is derived from REBALANCE_CT, so the slot and the label can
    # never drift apart when the schedule moves
    assert fr.REBALANCE_CT == (14, 30)
    assert fr.rebalance_label() == "2:30 PM CT"
