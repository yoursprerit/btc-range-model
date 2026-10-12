"""The post-submit wait must reach the auction print — and the runner must not
kill an executor that is still waiting for it.

2026-10-12: the 2:15 PM CT task placed its orders ~2 minutes in, then waited the
old 40-minute ceiling, which expired at ~2:57 PM CT, three minutes BEFORE the
3:00 PM CT auction. The execution report was written with every leg still
"Submitted" and the run exited "not fully filled".
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "app"))

import ibkr_common as ic  # noqa: E402
import ibkr_execute_accounts as acc  # noqa: E402


def _et(ts):
    return pd.Timestamp(ts, tz=ic.ET_TZ)


def test_the_scheduled_run_waits_through_the_auction_print():
    """Orders in at ~15:17:30 ET (2:15 PM CT task + start-up): the wait must run to
    16:00 ET plus the reporting buffer."""
    now = _et("2026-10-12 15:17:30")
    secs = ic.moc_wait_seconds(now)
    assert secs == pytest.approx(42.5 * 60 + ic.MOC_CLOSE_BUFFER_S, abs=1)
    end = now + pd.Timedelta(seconds=secs)
    assert end >= _et("2026-10-12 16:00:00") + pd.Timedelta(seconds=ic.MOC_CLOSE_BUFFER_S)
    assert not ic.moc_wait_capped(now)


def test_the_old_ceiling_would_have_stopped_short_of_the_close():
    """The regression: with the 2400 s ceiling the same run ended at 15:57:30 ET."""
    now = _et("2026-10-12 15:17:30")
    old = ic.moc_wait_seconds(now, max_wait_s=2400.0)
    assert now + pd.Timedelta(seconds=old) < _et("2026-10-12 16:00:00")
    assert ic.moc_wait_capped(now, max_wait_s=2400.0)


def test_an_early_start_is_flagged_as_capped():
    assert ic.moc_wait_capped(_et("2026-10-12 13:00:00"))          # report would be pre-auction
    assert not ic.moc_wait_capped(_et("2026-10-12 15:45:00"))


def test_the_early_close_and_late_start_waits_are_unchanged():
    assert ic.moc_wait_seconds(_et("2026-11-27 12:30:00"), buffer_s=0) == pytest.approx(1800)
    assert ic.moc_wait_seconds(_et("2026-08-04 16:30:00")) == 0.0
    assert ic.moc_wait_seconds(_et("2026-08-04 09:30:00"), max_wait_s=600.0) == 600.0


# ── the runner's child timeout can never undercut a full wait ────────────────
def _cfg(**kw):
    return {"accounts": [{"id": "U1111111"}], **kw}


def test_the_default_timeout_outlasts_the_wait():
    cfg, errs = acc.validate_config(_cfg())
    assert not errs and cfg.child_timeout_s >= ic.MOC_MAX_WAIT_S + 300


def test_the_dataclass_default_is_safe_too():
    assert acc.Config().child_timeout_s >= ic.MOC_MAX_WAIT_S + 300


def test_an_old_config_value_is_raised_not_trusted():
    """A config copied from the previous example carried 3000 s."""
    cfg, errs = acc.validate_config(_cfg(child_timeout_s=3000))
    assert not errs and cfg.child_timeout_s >= ic.MOC_MAX_WAIT_S + 300


def test_a_longer_timeout_is_kept():
    cfg, _ = acc.validate_config(_cfg(child_timeout_s=7200))
    assert cfg.child_timeout_s == 7200


def test_a_nonsense_timeout_is_still_rejected():
    assert acc.validate_config(_cfg(child_timeout_s=60))[1]


def test_the_shipped_example_config_carries_a_safe_timeout():
    import json
    raw = json.loads((ROOT / "deploy" / "ibkr_accounts.example.json").read_text())
    assert raw["child_timeout_s"] >= ic.MOC_MAX_WAIT_S + 300


def test_the_systemd_unit_outlasts_the_wait():
    import re
    unit = (ROOT / "deploy" / "systemd" / "ibkr-executor.service").read_text()
    assert int(re.search(r"TimeoutStartSec=(\d+)", unit).group(1)) > ic.MOC_MAX_WAIT_S


# ── an early start says so, instead of silently writing a pre-fill report ────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_executor_safety import _IB, _broker, _orders, _tk  # noqa: E402
from test_executor_safety import _no_ib_async  # noqa: E402,F401  (autouse stubs)


def _place_moc(monkeypatch, capped):
    monkeypatch.setattr(ic, "moc_entry_open", lambda now=None, cutoff=None: (True, "ok"))
    monkeypatch.setattr(ic, "moc_wait_seconds", lambda now=None, **k: 2)
    monkeypatch.setattr(ic, "moc_wait_capped", lambda now=None, **k: capped)
    b = _broker(_IB({"X": _tk(), "SPY": _tk()}))
    b.place(_orders(("X", "BUY", 1, 100.0)), False, 0.0, order_type=ic.ORDER_MOC)


def test_a_capped_wait_is_called_out_in_the_log(monkeypatch, capsys):
    _place_moc(monkeypatch, capped=True)
    out = capsys.readouterr().out
    assert "ceiling ends BEFORE the auction prints" in out and "reconcile" in out


def test_a_normal_wait_prints_no_warning(monkeypatch, capsys):
    _place_moc(monkeypatch, capped=False)
    assert "ceiling ends BEFORE" not in capsys.readouterr().out
