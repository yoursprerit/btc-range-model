"""``btc_ct_engine.live_diagnostics`` — the "will it trigger?" explainer.

``run_btc_ct`` says what the signal IS.  This says *why*, and what the next bar
would have to do to change it — the question people actually ask about a live
signal, and the one an assistant cannot answer by reading engine source.

The contract that matters most here is the U1 impossibility check.  U1 needs at
least 2 high-breaks within a 3-bar window, so the count reachable on the next
bar is ``breaks_in_last_2 + 1``.  When the last two bars broke no highs, no
entry can fire at the next close *however far the bar runs* — a definite "no",
not a hedge.  Getting that inverted would produce a confidently wrong answer
about a trade, so it is pinned in both directions against hand-built inputs.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "app"))

ce = pytest.importorskip("btc_ct_engine",
                         reason="CT engine needs the trained model artefacts")


@pytest.fixture(scope="module")
def diag():
    return ce.live_diagnostics(lookback=6)


def test_reports_the_bar_convention_and_the_latest_completed_bar(diag):
    """Every consumer needs to know these dates are bar STARTS — a bar labelled
    today is the one still open, closing tomorrow at 12:00Z."""
    assert "12:00" in diag["bar_convention"]
    assert "7:00 AM" in diag["bar_convention"]
    assert "START" in diag["bar_convention"]
    assert diag["latest_completed_bar_start"] == diag["recent_bars"][-1]["bar_start_date"]


def test_exposes_the_live_gate_thresholds(diag):
    th = diag["thresholds"]
    assert th["U1_ERRHI_MIN_pct"] == ce.U1_ERRHI_MIN
    assert th["D2_ERRHI_MAX_pct"] == ce.D2_ERRHI_MAX
    assert th["U1_min_high_breaks_in_3"] == 2
    assert set(th["gate_by_asset"]) == {"BTC", "MSTR", "MSTU", "ETH"}


def test_current_state_matches_the_engines_own_signal_arrays(diag):
    """These must be READ from compute_sigs_pure, never re-derived — a second
    implementation would drift from the gates the app actually trades."""
    import backtest_trailing_stop as T

    rf = T.load_raw_features()
    comp = T.prep(rf, T.build_preds_offline(rf), "2025-01-01",
                  str(rf.index[-1].date()))
    sigs = ce.compute_sigs_pure(comp)
    i = len(comp) - 1
    cur = diag["current_state"]
    assert cur["u1"] == bool(sigs["u1"][i])
    assert cur["hb3"] == int(sigs["hb3"][i])
    assert cur["above_ma30"] == bool(sigs["above_ma30"][i])
    assert cur["bull_regime"] == bool(sigs["bull_regime"][i])
    assert cur["entry_ma_gate"] == bool(sigs["tf2_entry_ma"][i])
    assert cur["ehma3_pct"] == pytest.approx(float(sigs["ehma3"][i]), abs=1e-3)


def test_recent_bars_carry_predicted_vs_realised_high_and_low(diag):
    bar = diag["recent_bars"][-1]
    for field in ("pred_high", "actual_high", "pred_low", "actual_low",
                  "err_hi_pct", "high_break", "hb3", "ehma3"):
        assert field in bar, field
    # err_hi is the realised overshoot of the predicted high, in % of the
    # reference close — the quantity U1 averages.
    expected = (bar["actual_high"] - bar["pred_high"]) / bar["close_asof"] * 100
    assert bar["err_hi_pct"] == pytest.approx(expected, abs=1e-2)
    assert bar["high_break"] == (bar["actual_high"] > bar["pred_high"])


def test_next_bar_requirement_is_stated_as_overshoot_not_price(diag):
    """The next bar's pred_high is not knowable until its features exist, so a
    price target would be fabricated. A % above prediction is honest."""
    req = diag["next_bar_entry_requirement"]
    assert isinstance(req["needs_err_hi_pct_above"], float)
    assert "% of that bar's reference close" in req["explanation"]
    assert "not knowable" in req["caveat"]


def test_u1_impossibility_is_decided_by_the_3_bar_break_count(monkeypatch):
    """The decisive branch, pinned both ways against hand-built signal arrays.

    hb3 counts high-breaks over a 3-bar window, so next bar it can reach at
    most (breaks in the last 2) + 1.  With 0 recent breaks, 2 is unreachable
    and an entry is impossible; with 1, it is reachable and must not be
    reported as impossible.
    """
    import pandas as pd

    def _fake_comp(n=40):
        # A DatetimeIndex, because live_diagnostics dates its window from the
        # raw-feature frame's own index (rf.index[-1].date()).
        idx = pd.date_range("2026-01-01", periods=n, freq="D")
        return pd.DataFrame({
            "target_date": idx,
            "close_asof": np.full(n, 100.0),
            "pred_high": np.full(n, 101.0), "pred_low": np.full(n, 99.0),
            "actual_high": np.full(n, 100.5), "actual_low": np.full(n, 99.5),
            "actual_close": np.full(n, 100.0),
        }, index=idx)

    for breaks, expect_possible in ((0, False), (1, True), (2, True)):
        comp = _fake_comp()
        # Break the predicted high on the last `breaks` bars.
        if breaks:
            comp.loc[comp.index[-breaks:], "actual_high"] = 105.0

        monkeypatch.setattr(ce.T, "load_raw_features", lambda: comp)
        monkeypatch.setattr(ce.T, "build_preds_offline", lambda _rf: None)
        monkeypatch.setattr(ce.T, "prep", lambda *a, **k: comp)

        out = ce.live_diagnostics(lookback=3)
        req = out["next_bar_entry_requirement"]
        assert req["high_breaks_in_last_2_bars"] == breaks
        assert req["u1_possible_next_bar"] is expect_possible, (
            f"{breaks} recent break(s) → expected possible={expect_possible}")
        if expect_possible:
            assert req["u1_impossible_reason"] is None
        else:
            assert "at most 1" in req["u1_impossible_reason"]
            assert "regardless of how high" in req["u1_impossible_reason"]
