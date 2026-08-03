"""Unit tests for ``daily_win_stats`` (app/overall_core.py) — the invested-days
winning read behind the P&L section's "Winning days (invested)" metric.

Synthetic equity curves only (no network, no Streamlit): the active-day
conditioning (idle SATA-yield days must not pad the rate), the flat-day
convention (in the denominator, never a win), the Wilson 95% interval, the
payoff / expectancy identities, the ``slice_metrics``-matching windowing, and
the degenerate cases (<2 bars, no active days, no losing days).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import overall_core as oc  # noqa: E402


IDX = pd.date_range("2026-06-01", periods=9, freq="D")


def _equity(rets):
    """Equity curve whose pct_change reproduces ``rets`` exactly (first bar 1.0)."""
    return pd.Series(np.cumprod([1.0] + list(1.0 + np.asarray(rets, float))),
                     index=pd.date_range("2026-06-01", periods=len(rets) + 1,
                                         freq="D"))


def test_all_days_without_active_mask():
    eq = _equity([0.02, -0.01, 0.03, 0.0, -0.02])
    d = oc.daily_win_stats(eq, eq.index[0])
    assert d["n_days"] == 5 and d["n_active"] == 5
    assert d["wins"] == 2                       # the 0.0 day is NOT a win
    assert d["win_rate"] == pytest.approx(0.4)
    assert d["expectancy"] == pytest.approx(np.mean([0.02, -0.01, 0.03, 0.0, -0.02]))


def test_active_mask_excludes_cash_days():
    # two real trading days buried in tiny always-positive "SATA" days: the
    # all-days rate is 5/6 but only the two invested days should count
    rets = [0.0005, 0.03, 0.0005, -0.01, 0.0005, 0.0005]
    eq = _equity(rets)
    active = pd.Series(0.0, index=eq.index)
    active.iloc[2] = 1.0                        # earns rets[1] = +0.03
    active.iloc[4] = 0.6                        # earns rets[3] = -0.01 (weight, not flag)
    d = oc.daily_win_stats(eq, eq.index[0], active=active)
    assert d["n_days"] == 6
    assert d["n_active"] == 2 and d["wins"] == 1
    assert d["win_rate"] == pytest.approx(0.5)
    assert d["avg_win"] == pytest.approx(0.03)
    assert d["avg_loss"] == pytest.approx(-0.01)
    assert d["payoff"] == pytest.approx(3.0)
    assert d["expectancy"] == pytest.approx(0.01)


def test_expectancy_identity():
    rets = [0.02, -0.01, 0.03, 0.0, -0.02, 0.01]
    d = oc.daily_win_stats(_equity(rets), "2026-06-01")
    n_loss = sum(r < 0 for r in rets)
    assert d["expectancy"] == pytest.approx(
        (d["wins"] * d["avg_win"] + n_loss * d["avg_loss"]) / d["n_active"])


def test_wilson_interval_known_value():
    lo, hi = oc._wilson(6, 10)                  # classic 6/10 check
    assert lo == pytest.approx(0.3127, abs=2e-3)
    assert hi == pytest.approx(0.8318, abs=2e-3)
    d = oc.daily_win_stats(_equity([0.01] * 6 + [-0.01] * 4), "2026-06-01")
    assert (d["ci_lo"], d["ci_hi"]) == (lo, hi)
    assert d["ci_lo"] < d["win_rate"] < d["ci_hi"]


def test_windowing_matches_slice_metrics():
    rets = [0.02, -0.01, 0.03, -0.02, 0.01]
    eq = _equity(rets)
    # anchor on the 3rd bar, close on the 5th → returns on bars 4 and 5 only
    d = oc.daily_win_stats(eq, eq.index[2], end=eq.index[4])
    assert d["n_days"] == 2
    assert d["wins"] == 1 and d["win_rate"] == pytest.approx(0.5)
    assert d["expectancy"] == pytest.approx(np.mean([0.03, -0.02]))


def test_degenerate_cases():
    assert oc.daily_win_stats(_equity([0.01]).iloc[:1], "2026-06-01") is None
    # active never > 0 → counted days but no rate to report
    eq = _equity([0.01, -0.01])
    d = oc.daily_win_stats(eq, eq.index[0],
                           active=pd.Series(0.0, index=eq.index))
    assert d["n_active"] == 0 and d["win_rate"] is None
    assert d["expectancy"] is None
    # no losing days → payoff undefined, avg_loss 0.0
    d = oc.daily_win_stats(_equity([0.01, 0.02]), "2026-06-01")
    assert d["payoff"] is None and d["avg_loss"] == 0.0
    assert d["win_rate"] == pytest.approx(1.0)


def test_active_index_mismatch_is_safe():
    # active series missing dates (e.g. sleeve starts later) → missing = flat
    eq = _equity([0.02, -0.01, 0.03])
    active = pd.Series(1.0, index=eq.index[-1:])   # only the last day flagged
    d = oc.daily_win_stats(eq, eq.index[0], active=active)
    assert d["n_active"] == 1 and d["wins"] == 1
