"""The multi-account live runner's configurable order type (the 2:20 PM CT slot).

MOC stays the default — existing configs and the strict-MOC contract
(tests/test_multi_account.py) are untouched.  ``marketable-limit`` / ``market``
trade at the touch in the regular session: no ``--strict-moc``, a regular-session
window instead of the MOC cutoff, and a visible warning when the 2 PM intraday
stop check did not run.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "app"))

import ibkr_common as ic                     # noqa: E402
import ibkr_execute_accounts as acc          # noqa: E402

ET = "America/New_York"


def _et(s):
    return pd.Timestamp(s, tz=ET)


def _cfg(**kw):
    return {"accounts": [{"id": "U1111111"}], **kw}


# ── config ───────────────────────────────────────────────────────────────────
def test_default_is_unchanged_strict_moc():
    cfg, errs = acc.validate_config(_cfg())
    assert not errs and cfg.order_type == ic.ORDER_MOC
    argv = acc.child_argv(cfg, cfg.accounts[0], 0, execute=True)
    assert "--strict-moc" in argv and argv[argv.index("--order-type") + 1] == "moc"
    assert "--slippage-cap" not in argv


@pytest.mark.parametrize("ot", [ic.ORDER_MARKETABLE_LIMIT, ic.ORDER_MARKET])
def test_regular_session_order_types_drop_strict_moc(ot):
    cfg, errs = acc.validate_config(_cfg(order_type=ot, slippage_cap=0.004,
                                         fill_timeout_s=90))
    assert not errs
    argv = acc.child_argv(cfg, cfg.accounts[0], 0, execute=True)
    assert argv[argv.index("--order-type") + 1] == ot
    assert "--strict-moc" not in argv
    assert argv[argv.index("--slippage-cap") + 1] == "0.004"
    assert argv[argv.index("--fill-timeout") + 1] == "90.0"
    # everything that guards a live account is still pinned
    for want in ("--require-signature", "--confirm-live", "--expected-account"):
        assert want in argv


def test_fill_timeout_is_only_passed_when_set():
    cfg, _ = acc.validate_config(_cfg(order_type="market"))
    assert "--fill-timeout" not in acc.child_argv(cfg, cfg.accounts[0], 0, execute=True)


@pytest.mark.parametrize("bad", [dict(order_type="limit"), dict(order_type="MOO"),
                                 dict(slippage_cap=0), dict(slippage_cap=0.2),
                                 dict(fill_timeout_s=-1), dict(fill_timeout_s=9999)])
def test_bad_order_settings_are_rejected(bad):
    cfg, errs = acc.validate_config(_cfg(**bad))
    assert cfg is None and errs


def test_order_type_is_case_insensitive():
    cfg, errs = acc.validate_config(_cfg(order_type=" Marketable-Limit "))
    assert not errs and cfg.order_type == ic.ORDER_MARKETABLE_LIMIT


def test_every_flag_the_runner_passes_exists_in_the_executor():
    src = (ROOT / "scripts" / "ibkr_execute_book.py").read_text()
    known = set(re.findall(r'add_argument\(\s*"(--[a-z-]+)"', src))
    for ot in ic.ORDER_TYPES:
        cfg, _ = acc.validate_config(_cfg(order_type=ot, fill_timeout_s=60))
        argv = acc.child_argv(cfg, cfg.accounts[0], 0, execute=True)
        flags = {a for a in argv if a.startswith("--")}
        assert flags <= known, flags - known


# ── the regular-session window (2:20 PM CT = 3:20 PM ET) ─────────────────────
def test_the_2_20_pm_ct_slot_is_inside_the_window():
    ok, why = ic.regular_entry_open(_et("2026-10-19 15:20"))
    assert ok and "16:00 ET" in why


@pytest.mark.parametrize("now, frag", [
    ("2026-10-19 09:00", "before the 09:30 ET open"),
    ("2026-10-19 15:51", "within 10 min"),
    ("2026-10-19 16:05", "within 10 min"),
])
def test_outside_the_window_is_refused(now, frag):
    ok, why = ic.regular_entry_open(_et(now))
    assert not ok and frag in why


def test_the_window_follows_an_early_close():
    # 2026-11-27 is the 1 PM ET early close: 3:20 PM ET is long past it
    assert not ic.regular_entry_open(_et("2026-11-27 15:20"))[0]
    assert ic.regular_entry_open(_et("2026-11-27 12:30"))[0]
    assert not ic.regular_entry_open(_et("2026-11-27 12:55"))[0]


def test_moc_window_is_independent_of_the_new_helper():
    assert ic.moc_entry_open(_et("2026-10-19 15:20"))[0]
    assert not ic.moc_entry_open(_et("2026-10-19 15:51"))[0]


# ── the stop-check marker ────────────────────────────────────────────────────
def _marker(tmp_path, when, status="clear", **kw):
    p = tmp_path / "m.json"
    p.write_text(json.dumps(dict(schema="x", checked_at_utc=when, status=status, **kw)))
    return p


TODAY = pd.Timestamp("2026-10-19").date()


def test_marker_ok_when_the_check_ran_today(tmp_path):
    for st in ("clear", "stopped", "already_applied"):
        s, d = acc.stop_check_status(_marker(tmp_path, "2026-10-19T19:01:00+00:00", st), TODAY)
        assert s == "ok" and "14:01 CT" in d


def test_marker_flags_a_missing_stale_or_failed_check(tmp_path):
    assert acc.stop_check_status(tmp_path / "absent.json", TODAY)[0] == "missing"
    s, d = acc.stop_check_status(_marker(tmp_path, "2026-10-16T19:01:00+00:00"), TODAY)
    assert s == "missing" and "not today" in d
    for st in ("error", "audit_failed", "no_current_book"):
        s, d = acc.stop_check_status(
            _marker(tmp_path, "2026-10-19T19:01:00+00:00", st, reason="boom"), TODAY)
        assert s == "failed" and "boom" in d
    bad = tmp_path / "bad.json"; bad.write_text("{nope")
    assert acc.stop_check_status(bad, TODAY)[0] == "missing"
