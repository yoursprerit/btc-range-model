"""Unit tests for the secondary (Nasdaq) daily feed — no network required."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import market_fallback as mf  # noqa: E402


def test_supports_only_plain_us_tickers():
    for ok in ("SOXX", "soxx", "MSTR", "GLDM", "ETHA", "IBIT"):
        assert mf.supports(ok), ok
    # indices, futures and FX quotes are NOT served by this provider
    for bad in ("^GSPC", "^VIX", "^TNX", "GC=F", "DX-Y.NYB", "BTC-USD", "", None):
        assert not mf.supports(bad), bad


def test_num_parses_nasdaq_display_values():
    assert mf._num("$91.67") == 91.67
    assert mf._num("10,952,030") == 10952030.0
    assert mf._num("527.01") == 527.01
    for empty in (None, "", "N/A", "--", "abc"):
        assert pd.isna(mf._num(empty)), empty


def test_daily_ohlcv_unsupported_symbol_short_circuits(monkeypatch):
    # must not even attempt a request for a symbol the provider can't serve
    def _boom(*a, **k):
        raise AssertionError("network attempted for an unsupported symbol")
    monkeypatch.setattr(mf.urllib.request, "urlopen", _boom)
    assert mf.daily_ohlcv("^VIX", "2026-07-01").empty


def test_daily_ohlcv_returns_empty_on_error(monkeypatch):
    mf._cache.clear()
    monkeypatch.setattr(mf.urllib.request, "urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("down")))
    assert mf.daily_ohlcv("SOXX", "2026-07-01").empty


def test_merge_frame_skips_unsupported_and_prefixes(monkeypatch):
    mf._cache.clear()
    frame = pd.DataFrame(
        {"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [10.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-07-24")]))
    monkeypatch.setattr(mf, "daily_ohlcv",
                        lambda sym, start, end=None: frame if mf.supports(sym) else pd.DataFrame())
    out = mf.merge_frame({"px": "SOXX", "vix": "^VIX"}, start="2026-07-01")
    assert "px_close" in out.columns
    assert not any(c.startswith("vix_") for c in out.columns)   # index skipped
