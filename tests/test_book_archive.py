"""Unit tests for the target-book as-of archive (book_archive/<as_of>.json).

The archive is the Historical View's source for "the book actually published
from this bar's committed signals" — written by the daily publisher on every
publish, backfilled from git history by scripts/backfill_book_archive.py.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import target_book as tb  # noqa: E402


def _payload(as_of="2026-07-24", gen="2026-07-25T14:17:22+00:00"):
    return tb.build_payload(as_of=as_of, profile="Balanced",
                            weights={"SOXX": 0.30, "REMX": 0.10, "SATA": 0.60},
                            cash_weight=0.0, exec_price={"SOXX": 245.1},
                            generated_at_utc=gen, book_mode="live")


def test_archive_path_is_keyed_by_as_of_date(tmp_path):
    book = tmp_path / "target_book_live.json"
    p = tb.archive_path(book, "2026-07-24")
    assert p == tmp_path / tb.ARCHIVE_DIRNAME / "2026-07-24.json"
    # timestamps normalise to the plain date
    assert tb.archive_path(book, "2026-07-24T12:00:00+00:00").name == "2026-07-24.json"


def test_archive_book_writes_a_loadable_signed_record(tmp_path):
    book = tmp_path / "target_book_live.json"
    out = tb.archive_book(_payload(), book, secret="s3cret")
    assert out is not None and out.exists()
    back = json.loads(out.read_text())
    assert back["as_of"] == "2026-07-24"
    ok, why = tb.verify_signature(back, "s3cret")
    assert ok, why


def test_republish_for_same_as_of_wins(tmp_path):
    # an intraday re-publish replaces that day's record — last publish wins,
    # matching what the executor actually traded
    book = tmp_path / "target_book_live.json"
    tb.archive_book(_payload(gen="2026-07-24T12:15:00+00:00"), book)
    late = _payload(gen="2026-07-24T21:44:31+00:00")
    late["weights"] = {"SOXX": 0.25, "SATA": 0.75}
    out = tb.archive_book(late, book)
    back = json.loads(out.read_text())
    assert back["generated_at_utc"] == "2026-07-24T21:44:31+00:00"
    assert back["weights"]["SOXX"] == 0.25


def test_archive_book_never_raises(tmp_path):
    # no as_of → skipped; unwritable target → best-effort None (never blocks
    # a publish)
    assert tb.archive_book({"weights": {}}, tmp_path / "b.json") is None
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")
    assert tb.archive_book(_payload(), blocked / "b.json") is None
