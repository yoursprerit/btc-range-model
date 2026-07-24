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


def test_record_refresh_never_raises(monkeypatch, tmp_path):
    # unwritable dir → still no exception (best-effort by design)
    monkeypatch.setattr(fr, "RUNTIME_DIR", tmp_path / "no" / "such" / "\0bad")
    fr.record_refresh("X", foo=1)   # must not raise
