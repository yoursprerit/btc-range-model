"""Read-time price guard for the four CT sleeves (app/btc_ct_engine.py).

BTC/MSTR/MSTU/ETH never pass through app/data_gate.py — their prices come from
the committed data/backtest/ vintage, which the engine consumed on the stated
assumption that scripts/validate_refreshed_data.py had already cleared it.  In
2026-08 that assumption failed: mstu_synthetic_daily.csv carried a spliced
+917.81% bar (MSTU really moved +5.71%) and every sleeve read it, because the
one check that would have caught it — data_gate.MAX_ABS_TRADED_MOVE, whose own
comment names "a large-ratio split splice" — only ran on the OTHER 14 sleeves.

These tests cover the guard that now runs at load time, and the policy around
it: a defective sibling is dropped so its parent still loads, but a defective
BTC — the signal asset every sleeve is gated on — fails loudly instead of
publishing a book built on fabricated prices.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

import btc_ct_engine as ct  # noqa: E402
import data_gate as dg      # noqa: E402

IDX = pd.bdate_range("2024-01-01", periods=300)


def _px(seed=5, n=300):
    rng = np.random.default_rng(seed)
    return 100 * np.cumprod(1 + rng.normal(0.0004, 0.02, n))


# ── the check itself ───────────────────────────────────────────────────────

def test_threshold_tracks_the_data_gate():
    # data_gate is the source of truth; the engine's fallback literal must not
    # silently drift away from it
    assert ct._MAX_ABS_TRADED_MOVE == dg.MAX_ABS_TRADED_MOVE


def test_ordinary_history_passes():
    assert ct._px_defect(_px(), IDX) is None


def test_catches_a_spliced_scale_break():
    a = _px()
    a[-4:] = a[-4:] * 9.900208            # two scales in one series
    why = ct._px_defect(a, IDX)
    assert why and "corrupted splice" in why
    assert str(IDX[-4].date()) in why


def test_reports_the_worst_bar_not_merely_the_first():
    a = _px()
    a[100:] = a[100:] * 2.0               # ~+100%
    a[200:] = a[200:] * 12.0              # ~+1100% — the worst
    why = ct._px_defect(a, IDX)
    # the bar reported is the LARGEST break, not the first one encountered
    assert str(IDX[200].date()) in why
    assert float(why.split("%")[0].replace(",", "").lstrip("+")) > 1000


def test_a_nan_hole_is_skipped_not_read_as_a_gap():
    # a pending fill or a late-starting sleeve leaves NaN; comparing across the
    # hole would invent a gap-sized return and reject good data
    a = _px()
    a[50:60] = np.nan
    assert ct._px_defect(a, IDX) is None


def test_late_starting_sleeve_passes():
    a = _px()
    a[:120] = np.nan
    assert ct._px_defect(a, IDX) is None


def _step(a, i, size):
    """Rescale everything from ``i`` so the series takes exactly one step of
    ``size`` there — no snap-back on the following bar to muddy the test."""
    a = a.copy()
    a[i:] = a[i:] * (1 + size) / (a[i] / a[i - 1])
    return a


def test_a_move_just_inside_the_bound_passes():
    a = _step(_px(n=10), 5, dg.MAX_ABS_TRADED_MOVE * 0.9)
    assert ct._px_defect(a, IDX[:10]) is None


def test_a_move_just_outside_the_bound_is_caught():
    a = _step(_px(n=10), 5, dg.MAX_ABS_TRADED_MOVE * 1.1)
    assert ct._px_defect(a, IDX[:10]) is not None


def test_degenerate_inputs_are_described_not_crashed():
    assert ct._px_defect(np.array([1.0]), IDX[:1]) == "fewer than 2 bars"
    assert ct._px_defect(np.full(5, np.nan), IDX[:5]) == "no usable prices"
    assert ct._px_defect(np.zeros(5), IDX[:5]) == "no usable prices"


# ── the policy: drop a sibling, refuse on BTC ──────────────────────────────

def _wire(monkeypatch, tmp_path, bad=None, factor=9.900208):
    """_load_prices against synthetic sleeves, optionally corrupting one."""
    monkeypatch.setattr(ct.T, "DATA", tmp_path)      # no synthetic CSV → fallback
    series = {}
    for i, k in enumerate(("MSTR", "MSTU", "ETH")):
        a = _px(seed=i + 1)
        if k == bad:
            a[-4:] = a[-4:] * factor
        series[k] = pd.Series(a, index=IDX)
    monkeypatch.setattr(ct.T, "load_asset", lambda k: series[k])
    btc = _px(seed=9)
    if bad == "BTC":
        btc[-4:] = btc[-4:] * factor
    comp = pd.DataFrame({"actual_close": btc}, index=IDX)
    return comp


def test_a_defective_sibling_is_dropped_and_the_others_survive(monkeypatch,
                                                               tmp_path):
    comp = _wire(monkeypatch, tmp_path, bad="MSTU")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = ct._load_prices(IDX, comp)
    assert "MSTU" not in out
    assert {"BTC", "MSTR", "ETH"} <= set(out)
    assert any("MSTU sleeve dropped" in str(w.message) for w in caught)


def test_a_clean_universe_keeps_every_sleeve(monkeypatch, tmp_path):
    comp = _wire(monkeypatch, tmp_path, bad=None)
    out = ct._load_prices(IDX, comp)
    assert {"BTC", "MSTR", "MSTU", "ETH"} == set(out)


def test_a_defective_btc_refuses_to_run(monkeypatch, tmp_path):
    # BTC is the signal asset every sleeve is gated on — dropping it silently
    # would leave the book trading on the rest off a fabricated signal
    comp = _wire(monkeypatch, tmp_path, bad="BTC")
    with pytest.raises(RuntimeError, match="BTC price series rejected"):
        ct._load_prices(IDX, comp)


# ── the real failure, replayed ─────────────────────────────────────────────

def test_the_shipped_mstu_vintage_is_clean():
    syn = pd.read_csv(ROOT / "data/backtest/mstu_synthetic_daily.csv",
                      parse_dates=["Date"]).set_index("Date")["close"]
    assert ct._px_defect(syn.to_numpy(float), syn.index) is None


# The exact bytes that shipped in mstu_synthetic_daily.csv across the splice —
# pre-split scale through 08-21, post-split from 08-24 (git be2aca0 and earlier).
# Held as a literal rather than reconstructed from the corrected file, so the
# regression asserts the real +917.81% and not an approximation of it.
_SHIPPED_SPLICE = {
    "2026-08-19": 1.8386000394821167,
    "2026-08-20": 2.3949999809265137,
    "2026-08-21": 2.7950000762939453,
    "2026-08-24": 28.447900772094727,      # <- +917.81%; no market did that
    "2026-08-25": 28.145000457763672,
    "2026-08-26": 29.385000228881836,
}


def test_the_2026_08_mstu_splice_would_now_be_rejected():
    s = pd.Series(_SHIPPED_SPLICE)
    s.index = pd.DatetimeIndex(s.index)
    why = ct._px_defect(s.to_numpy(float), s.index)
    assert why, "the series that actually shipped must not pass the guard"
    assert "+917.81%" in why and "2026-08-24" in why
