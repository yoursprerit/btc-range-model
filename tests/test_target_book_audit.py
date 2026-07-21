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

TODAY = pd.Timestamp("2026-07-21")


def _payload(**extra):
    p = tb.build_payload(
        as_of="2026-07-21", profile="Balanced", weights={"SOXX": 0.3},
        cash_weight=0.7, exec_price={"SOXX": 250.0},
        generated_at_utc=pd.Timestamp("2026-07-21T12:18:00Z").isoformat())
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
