"""Unit tests for the execution-report as-of archive (executed_archive/).

The archive is the Executed Book page's 🕰️ Historical tab source: every
rebalance overwrites data/overall/executed_book.json, so a dated copy keyed by
the signal bar it executed is what makes past runs viewable at all. Written by
the executor on each run (scripts/ibkr_execute_book.py), backfilled from git
history by scripts/backfill_executed_archive.py.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import executed_book as eb  # noqa: E402
import target_book as tb    # noqa: E402


def _payload(as_of="2026-08-17", gen="2026-08-18T19:30:29+00:00",
             account_mode="paper", qty=12.0):
    return eb.build_payload(
        as_of=as_of, profile="Balanced", mode="execute", account="DU1234567",
        net_liq=914643.28, cash=38426.39,
        trades=[dict(key="SOXX", symbol="SOXX", action="BUY", qty=qty,
                     price=559.12, status="Filled", filled=qty,
                     avg_fill_price=559.30)],
        positions=[dict(key="SOXX", symbol="SOXX", shares=qty, avg_cost=559.30,
                        market_price=560.0, market_value=qty * 560.0,
                        unrealized_pnl=8.4)],
        generated_at_utc=gen, account_mode=account_mode)


def test_archive_path_is_keyed_by_as_of_and_account_mode(tmp_path):
    paper = tmp_path / "executed_book.json"
    live = tmp_path / "executed_book_live.json"
    assert eb.archive_path(paper, "2026-08-17") == \
        tmp_path / eb.ARCHIVE_DIRNAME / "2026-08-17.json"
    # a live run must never land on a paper run's record
    assert eb.archive_path(live, "2026-08-17") == \
        tmp_path / eb.ARCHIVE_DIRNAME / "2026-08-17_live.json"
    # timestamps normalise to the plain date
    assert eb.archive_path(paper, "2026-08-17T19:30:29+00:00").name == "2026-08-17.json"


def test_archive_report_writes_a_loadable_signed_record(tmp_path):
    out = eb.archive_report(_payload(), tmp_path / "executed_book.json",
                            secret="s3cret")
    assert out is not None and out.exists()
    back = json.loads(out.read_text())
    assert back["as_of"] == "2026-08-17"
    assert back["trades"][0]["symbol"] == "SOXX"
    ok, why = tb.verify_signature(back, "s3cret")
    assert ok, why


def test_rerun_for_same_signal_bar_wins(tmp_path):
    # a late --refresh-report (or a manual after-hours top-up) replaces that
    # bar's record — last run wins, matching the account's end state
    report = tmp_path / "executed_book.json"
    eb.archive_report(_payload(gen="2026-08-18T19:30:29+00:00", qty=12.0), report)
    out = eb.archive_report(_payload(gen="2026-08-18T21:05:00+00:00", qty=15.0), report)
    back = json.loads(out.read_text())
    assert back["generated_at_utc"] == "2026-08-18T21:05:00+00:00"
    assert back["trades"][0]["qty"] == 15.0


def test_archive_report_never_raises(tmp_path):
    # no as_of → skipped; unwritable target → best-effort None (never fails a
    # rebalance)
    assert eb.archive_report({"trades": []}, tmp_path / "executed_book.json") is None
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")
    assert eb.archive_report(_payload(), blocked / "executed_book.json") is None


def test_archived_records_are_newest_first_and_mode_scoped(tmp_path):
    paper = tmp_path / "executed_book.json"
    live = tmp_path / "executed_book_live.json"
    eb.archive_report(_payload(as_of="2026-08-14", gen="2026-08-15T19:00:00+00:00"), paper)
    eb.archive_report(_payload(as_of="2026-08-17", gen="2026-08-18T19:30:29+00:00"), paper)
    eb.archive_report(_payload(as_of="2026-08-17", gen="2026-08-18T19:31:00+00:00",
                               account_mode="live"), live)

    recs = eb.archived_records(paper)
    assert [r["as_of"] for r in recs] == ["2026-08-17", "2026-08-14"]
    # the executor trades the morning AFTER the signal bar — the historical
    # picker offers that execution date, not the bar
    assert recs[0]["executed_on"] == "2026-08-18"
    assert all(r["payload"]["account_mode"] == "paper" for r in recs)

    live_recs = eb.archived_records(live)
    assert [r["as_of"] for r in live_recs] == ["2026-08-17"]
    assert live_recs[0]["payload"]["account_mode"] == "live"


def test_archived_records_skip_junk_and_missing_dirs(tmp_path):
    report = tmp_path / "executed_book.json"
    assert eb.archived_records(report) == []          # no archive dir yet
    eb.archive_report(_payload(), report)
    d = eb.archive_dir(report)
    (d / "2026-08-15.json").write_text("{not json")            # unreadable
    (d / "2026-08-14.json").write_text(json.dumps({"schema": "other/v1"}))
    (d / "notes.txt").write_text("ignored")
    recs = eb.archived_records(report)
    assert [r["as_of"] for r in recs] == ["2026-08-17"]


def test_record_for_snaps_back_to_the_standing_run(tmp_path):
    report = tmp_path / "executed_book.json"
    for as_of, gen in (("2026-08-13", "2026-08-14T19:00:00+00:00"),
                       ("2026-08-14", "2026-08-15T19:00:00+00:00")):
        eb.archive_report(_payload(as_of=as_of, gen=gen), report)
    recs = eb.archived_records(report)

    # an exact hit
    assert eb.record_for(recs, "2026-08-14")["executed_on"] == "2026-08-14"
    # the weekend after the Friday run — the Friday book was what stood
    assert eb.record_for(recs, "2026-08-16")["executed_on"] == "2026-08-15"
    # before the archive begins → nothing to show
    assert eb.record_for(recs, "2026-08-01") is None
    # a timestamp normalises to its date
    assert eb.record_for(recs, "2026-08-15T09:30:00")["as_of"] == "2026-08-14"


def test_completed_run_is_the_duplicate_run_lock(tmp_path):
    report = tmp_path / "executed_book.json"
    assert eb.completed_run(report, "2026-08-17") is None      # nothing executed yet
    eb.archive_report(_payload(as_of="2026-08-17"), report)
    prior = eb.completed_run(report, "2026-08-17")
    assert prior and prior["as_of"] == "2026-08-17"
    # a different bar is not locked, and neither account mode locks the other
    assert eb.completed_run(report, "2026-08-18") is None
    assert eb.completed_run(tmp_path / "executed_book_live.json", "2026-08-17") is None


def test_a_dry_run_does_not_lock_the_bar(tmp_path):
    # a dry-run places no orders, so it must not stop the real run that follows
    report = tmp_path / "executed_book.json"
    dry = _payload(as_of="2026-08-17")
    dry["mode"] = "dry-run"
    eb.archive_report(dry, report)
    assert eb.completed_run(report, "2026-08-17") is None


# ── account reset ───────────────────────────────────────────────────────────
def test_a_reset_hides_the_previous_account_and_survives_a_backfill(tmp_path):
    report = tmp_path / "executed_book.json"
    eb.archive_report(_payload(as_of="2026-08-16", gen="2026-08-17T19:00:00+00:00"), report)
    eb.archive_report(_payload(as_of="2026-08-17", gen="2026-08-18T19:00:00+00:00"), report)
    assert len(eb.archived_records(report)) == 2

    eb.write_reset(report, "2026-08-17", reason="paper account reset")
    # the old account's runs are gone from the record...
    assert eb.archived_records(report) == []
    # ...even if the files themselves come back (a backfill from git history)
    assert eb.archive_path(report, "2026-08-17").exists()
    assert eb.archived_records(report) == []


def test_a_reset_does_not_hide_runs_after_the_cutoff(tmp_path):
    report = tmp_path / "executed_book.json"
    eb.write_reset(report, "2026-08-17", reason="paper account reset")
    eb.archive_report(_payload(as_of="2026-08-18", gen="2026-08-19T19:00:00+00:00"), report)
    recs = eb.archived_records(report)
    assert [r["as_of"] for r in recs] == ["2026-08-18"], \
        "the new account's first run must start the record"


def test_a_reset_releases_the_duplicate_run_lock_for_retired_bars(tmp_path):
    # the new account has never traded that bar, so the lock must not fire
    report = tmp_path / "executed_book.json"
    eb.archive_report(_payload(as_of="2026-08-17"), report)
    assert eb.completed_run(report, "2026-08-17") is not None
    eb.write_reset(report, "2026-08-17")
    assert eb.completed_run(report, "2026-08-17") is None


def test_reset_marker_is_per_account_mode(tmp_path):
    paper, live = tmp_path / "executed_book.json", tmp_path / "executed_book_live.json"
    eb.write_reset(paper, "2026-08-17")
    assert eb.read_reset(paper) is not None
    assert eb.read_reset(live) is None, "resetting paper must not retire live history"


def test_the_marker_is_not_mistaken_for_a_run(tmp_path):
    report = tmp_path / "executed_book.json"
    eb.write_reset(report, "2026-08-01")
    eb.archive_report(_payload(as_of="2026-08-17"), report)
    assert [r["as_of"] for r in eb.archived_records(report)] == ["2026-08-17"]


def test_a_dry_run_never_overwrites_an_executed_record(tmp_path):
    """What an after-hours dry-run did to the real 2026-08-17 record: replaced
    the fills with a list of PLANNED trades."""
    report = tmp_path / "executed_book.json"
    eb.archive_report(_payload(as_of="2026-08-17", qty=462.0), report)
    preview = _payload(as_of="2026-08-17", gen="2026-08-19T01:18:14+00:00", qty=271.0)
    preview["mode"] = "dry-run"
    preview["trades"][0]["status"] = "PLANNED"
    assert eb.archive_report(preview, report) is None, "a preview is not a record"
    back = json.loads(eb.archive_path(report, "2026-08-17").read_text())
    assert back["mode"] == "execute" and back["trades"][0]["qty"] == 462.0


# ════════════════════════════════════════════════════════════════════════════
# REPORT AGE — counted in trading SESSIONS, not wall-clock hours
# ════════════════════════════════════════════════════════════════════════════
# The executor runs once per trading day at 2:30 PM CT (= 3:30 PM ET), so a
# Friday report is the newest one that can exist until Monday afternoon. The old
# 48-hour window could not express that: Fri 2:30 PM CT → Mon 2:30 PM CT is 72 h,
# so a flawless week still flagged its own report stale from Sunday afternoon on
# ("report is 3.1 days old (> 2d)"), while any window wide enough to cover the
# weekend would have hidden a missed midweek session.
import pandas as pd  # noqa: E402


FRIDAY_RUN = "2026-08-21T19:30:47+00:00"        # Fri 2:30 PM CT, the real one


def _report(gen=FRIDAY_RUN, as_of="2026-08-20"):
    return _payload(as_of=as_of, gen=gen)


def test_fridays_report_is_fresh_all_weekend():
    for when in ("2026-08-22 11:00", "2026-08-23 18:00"):      # Sat, Sun
        ok, why = eb.validate(_report(), pd.Timestamp(when))
        assert ok, f"{when}: {why}"


def test_fridays_report_is_fresh_on_monday_before_the_slot():
    """Someone looking at 9 AM Monday: nothing newer can exist yet."""
    ok, why = eb.validate(_report(), pd.Timestamp("2026-08-24 09:00"))
    assert ok, why


def test_a_missed_monday_slot_goes_stale_that_afternoon():
    """2026-08-24: Monday's run never happened, so from ~4:15 PM ET the newest
    report is a session behind — which is exactly what should show yellow."""
    ok, why = eb.validate(_report(), pd.Timestamp("2026-08-24 17:41"))
    assert not ok
    assert "1 session" in why and "2026-08-21" in why and "2026-08-24" in why


def test_a_missed_midweek_session_is_not_hidden():
    """What a wider hour-window would have swallowed: Wednesday evening with
    Tuesday's report still the newest."""
    ok, why = eb.validate(_report(gen="2026-08-18T19:30:00+00:00"),
                          pd.Timestamp("2026-08-19 17:00"))
    assert not ok and "1 session" in why


def test_todays_own_report_is_fresh():
    ok, why = eb.validate(_report(gen="2026-08-24T19:30:00+00:00"),
                          pd.Timestamp("2026-08-24 17:41"))
    assert ok, why


def test_a_holiday_does_not_age_the_report():
    """2026-09-07 is Labor Day: Friday's report is still the newest one that
    can exist when the market reopens Tuesday morning."""
    ok, why = eb.validate(_report(gen="2026-09-04T19:30:00+00:00"),
                          pd.Timestamp("2026-09-08 09:00"))
    assert ok, why


def test_the_expected_session_flips_at_the_slot_plus_grace():
    before = eb.expected_report_session(pd.Timestamp("2026-08-24 15:00"))
    after = eb.expected_report_session(pd.Timestamp("2026-08-24 16:20"))
    assert str(before.date()) == "2026-08-21"      # Friday — nothing newer yet
    assert str(after.date()) == "2026-08-24"       # today's run should have landed


def test_a_wrong_schema_is_still_refused():
    ok, why = eb.validate({"schema": "something-else/v1"}, pd.Timestamp("2026-08-24"))
    assert not ok and "unexpected schema" in why
