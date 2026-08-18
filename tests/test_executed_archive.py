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
