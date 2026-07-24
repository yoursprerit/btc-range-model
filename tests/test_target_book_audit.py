"""Unit tests for the target book's signal-audit gate (app/target_book.py).

A book whose publish-time signal-freshness audit FAILED must never validate
(the executor refuses it); books without the stamp (published before the audit
existed) stay accepted for back-compatibility.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import target_book as tb  # noqa: E402

# validate() measures generation age against the REAL clock, so the fixture
# must be clock-relative (a hardcoded date rots into a false "stale" failure).
NOW = pd.Timestamp.now(tz="UTC")
TODAY = NOW.tz_localize(None).normalize()


def _payload(**extra):
    p = tb.build_payload(
        as_of=str(NOW.date()), profile="Balanced", weights={"SOXX": 0.3},
        cash_weight=0.7, exec_price={"SOXX": 250.0},
        generated_at_utc=NOW.isoformat(timespec="seconds"))
    p.update(extra)
    return p


def test_validate_accepts_unstamped_book():
    ok, why = tb.validate(_payload(), TODAY)
    assert ok, why


def test_validate_accepts_audit_passed_book():
    p = _payload(signal_audit=dict(passed=True, stale_apps=[],
                                   checked_at_utc="2026-07-21T12:16:00+00:00"))
    ok, why = tb.validate(p, TODAY)
    assert ok, why


def test_validate_rejects_audit_failed_book():
    p = _payload(signal_audit=dict(passed=False, stale_apps=["BTC", "XLE"],
                                   checked_at_utc="2026-07-21T12:16:00+00:00"))
    ok, why = tb.validate(p, TODAY)
    assert not ok
    assert "audit FAILED" in why and "BTC" in why


def test_signature_covers_audit_stamp():
    """The HMAC must cover signal_audit, so the verdict can't be tampered with."""
    secret = "s3cret"
    p = tb.sign(_payload(signal_audit=dict(passed=True, stale_apps=[])), secret)
    ok, _ = tb.verify_signature(p, secret)
    assert ok
    p["signal_audit"]["passed"] = False          # tamper
    ok, why = tb.verify_signature(p, secret)
    assert not ok and "MISMATCH" in why


# ── generation-age window — must span the once-daily 7:15-AM-CT cycle ─────────
def _payload_aged(hours_ago: float) -> dict:
    now = pd.Timestamp.now(tz="UTC")
    gen = now - pd.Timedelta(hours=hours_ago)
    p = tb.build_payload(
        as_of=str(gen.date()), profile="Balanced", weights={"SOXX": 0.3},
        cash_weight=0.7, exec_price={"SOXX": 250.0},
        generated_at_utc=gen.isoformat(timespec="seconds"))
    return p


def test_validate_accepts_yesterdays_book():
    """The book publishes ONCE daily; yesterday's book (≈25h old) must still
    validate so the executor can fall back to it when today's publish was
    withheld by a failed audit."""
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    ok, why = tb.validate(_payload_aged(25.0), today)
    assert ok, why


def test_validate_rejects_book_older_than_a_cycle():
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    ok, why = tb.validate(_payload_aged(31.0), today)
    assert not ok
    assert "stale" in why


# ── previous-book rotation (the "Previous Targetbook" the UI shows) ───────────
def test_rotate_prev_preserves_outgoing_book(tmp_path):
    book = tmp_path / "target_book.json"
    book.write_text('{"schema": "overall-target-book/v1", "as_of": "2026-07-23"}')
    assert tb.rotate_prev(book)
    prev = tb.prev_path(book)
    assert prev.name == "target_book_prev.json"
    assert prev.read_text() == book.read_text()


def test_rotate_prev_missing_book_is_noop(tmp_path):
    book = tmp_path / "target_book.json"
    assert not tb.rotate_prev(book)
    assert not tb.prev_path(book).exists()
