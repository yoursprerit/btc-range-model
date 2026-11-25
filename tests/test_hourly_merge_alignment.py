"""Regression: fetch_hourly must keep the primary's newest hourly bars when the
macro series stamp their bars at different minutes (ARTY went stale a day)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import ticker_core as tc    # noqa: E402
import ticker_config        # noqa: E402
import gldm_core as gc      # noqa: E402


def _bars(stamps, base):
    idx = pd.to_datetime(stamps, format="mixed")
    v = [base + i for i in range(len(idx))]
    return pd.DataFrame({"open": v, "high": v, "low": v, "close": v,
                         "volume": 1.0}, index=idx)


PRIMARY = ["2026-11-24 19:30", "2026-11-24 20:30", "2026-11-25 14:30", "2026-11-25 15:15"]
MACRO = ["2026-11-24 19:57", "2026-11-24 20:56", "2026-11-25 14:57", "2026-11-25 15:15:54"]


def _fake(primary_sym):
    def chart(sym, interval, start=None, range_=None, timeout=30):
        return _bars(PRIMARY if sym == primary_sym else MACRO,
                     100.0 if sym == primary_sym else 1.0)
    return chart


def test_ticker_hourly_keeps_latest_primary_bar(monkeypatch):
    cfg = ticker_config.get_config("ARTY")
    monkeypatch.setattr(tc, "_chart", _fake(cfg.primary_symbol))
    h = tc.fetch_hourly(cfg)
    assert h.index.max() == pd.Timestamp("2026-11-25 15:00")
    assert pd.Timestamp("2026-11-25 14:00") in h.index
    assert h["px_close"].notna().all()


def test_gldm_hourly_keeps_latest_primary_bar(monkeypatch):
    monkeypatch.setattr(gc, "_chart", _fake(gc.PRIMARY_SYMBOL))
    h = gc.fetch_hourly()
    assert h.index.max() == pd.Timestamp("2026-11-25 15:00")
    assert h["gldm_close"].notna().all()
