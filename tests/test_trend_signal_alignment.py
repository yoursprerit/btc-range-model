"""The trend engine's long/flat signal must be the one every surface reads.

``backtest_ticker.build_predictions`` drops any bar whose ridge FEATURES are
incomplete — a single NaN feature removes the whole row.  That is correct for
the H/L model, but the trend family's SMAs (``_rolling_mean``) count bars
POSITIONALLY, so every dropped row slides the rolling windows one bar further
back.  The simulation then walks a different trend signal than
``trend_long_now`` — which is what the decision label, the trend chart, the
live-exit check and the published book all read.

Observed 2026-08: five SOXX rows were dropped in April (a 14-session run-up
with no down day left ``rsi_14`` undefined), sliding the 100-day slow SMA to
521.32 against a true 526.44.  The simulation still saw the 25/100 pair crossed
UP, so it never executed the 2026-08-27 death cross: ``in_pos_now`` stayed True
and the decision stayed the in-position "EXIT NEXT BAR — BELOW TREND".  The
Overall action plan therefore re-flagged SOXX and SOXL "⚠️ exits next bar"
every day — days after the executor had actually sold them (2026-08-28) and
after they had dropped out of the published Target Book.

No network: every check runs on the repo's pinned snapshot CSVs.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "app"))

import backtest_ticker as bt      # noqa: E402
import freshness as fr            # noqa: E402
import overall_core as ov         # noqa: E402
import ticker_config as tcfg      # noqa: E402
import ticker_core as tc          # noqa: E402

TREND_KEYS = [k for k, c in tcfg.CONFIGS.items() if c.is_trend]


def _snapshot(cfg) -> pd.DataFrame:
    """The pinned daily snapshot for one sleeve, completed sessions only."""
    csv = Path(tc.cache_paths(cfg)["daily"])
    if not csv.exists():
        pytest.skip(f"pinned snapshot missing: {csv}")
    df = pd.read_csv(csv, index_col=0, parse_dates=True)
    if "px_close" not in df.columns:
        pref = f"{cfg.primary_symbol.lower()}_"
        df = df.rename(columns={c: "px_" + c[len(pref):]
                                for c in df.columns if c.startswith(pref)})
    df = fr.drop_in_progress_us_bar(df)
    if df is None or len(df) < 300:
        pytest.skip("snapshot too short")
    return df


# ── the signal itself ────────────────────────────────────────────────────
@pytest.mark.parametrize("key", TREND_KEYS)
def test_preds_trend_long_matches_the_contiguous_history(key):
    """``preds["trend_long"]`` IS ``trend_long_array`` on the ungapped frame.

    Not "close enough on the recent tail": every retained bar must carry the
    signal the ungapped history produces for that date, warm-up included.
    """
    cfg = tcfg.CONFIGS[key]
    hist = _snapshot(cfg)
    preds = bt.build_predictions(cfg, hist)
    assert "trend_long" in preds.columns

    canonical = pd.Series(
        np.asarray(bt.trend_long_array(cfg, hist["px_close"].to_numpy(float)), bool),
        index=hist.index)
    aligned = canonical.reindex(pd.to_datetime(preds["target_date"])).to_numpy()
    assert not pd.isna(aligned).any()
    np.testing.assert_array_equal(preds["trend_long"].to_numpy(bool), aligned)


@pytest.mark.parametrize("key", TREND_KEYS)
def test_last_bar_signal_agrees_with_trend_long_now(key):
    """The newest simulated bar and ``trend_long_now`` can never disagree —
    that disagreement is precisely what stranded SOXX in a permanent
    "exits next bar"."""
    cfg = tcfg.CONFIGS[key]
    hist = _snapshot(cfg)
    preds = bt.build_predictions(cfg, hist)
    assert bool(preds["trend_long"].to_numpy(bool)[-1]) is bt.trend_long_now(cfg, hist)


def test_simulate_regime_prefers_the_precomputed_signal():
    """A punched close series must NOT decide the trades when the gap-free
    signal is on the frame: ``simulate_regime`` follows ``trend_long``."""
    cfg = tcfg.CONFIGS["SOXX"]
    n = 400
    dates = pd.bdate_range("2024-01-01", periods=n)
    # a price path whose naive SMA read is long throughout …
    close = pd.Series(np.linspace(100.0, 300.0, n), index=dates)
    preds = pd.DataFrame({"target_date": dates, "px_close": close.to_numpy(),
                          "soxl_close": close.to_numpy()}, index=dates)
    naive = bt.trend_long_array(cfg, preds["px_close"].to_numpy(float))
    assert naive[-1] and naive[-2]

    # … while the gap-free signal says the trend broke two bars ago.
    flagged = preds.copy()
    flagged["trend_long"] = naive
    flagged.iloc[-2:, flagged.columns.get_loc("trend_long")] = False

    held = bt.simulate_regime(cfg, preds, None, "px_close", oos_start="2024-06-01")
    exited = bt.simulate_regime(cfg, flagged, None, "px_close", oos_start="2024-06-01")
    assert held["in_pos_now"] is True
    assert exited["in_pos_now"] is False

    # an explicit ma_window (the sweep path) still overrides both
    swept = bt.simulate_regime(cfg, flagged, None, "px_close", ma_window=20,
                               oos_start="2024-06-01")
    assert swept["in_pos_now"] is True


# ── the invariant the bug broke ──────────────────────────────────────────
@pytest.mark.parametrize("key", TREND_KEYS)
def test_position_cannot_outlive_a_broken_trend(key, monkeypatch):
    """A trend sleeve may only still be open on the newest bar if the trend was
    long at the bar BEFORE it — ``simulate_regime`` exits on
    ``long_at_close[i-1]``.  A held position under a two-bar-old break is the
    stranded state that renders "exits next bar" forever."""
    cfg = tcfg.CONFIGS[key]
    hist = _snapshot(cfg)
    canonical = bt.trend_long_array(cfg, hist["px_close"].to_numpy(float))

    monkeypatch.setattr(ov, "_load_daily", lambda _cfg, _h=hist: _h.copy())
    results = ov.run_asset(ov.overall_config(key))
    assert results, f"{key}: engine returned no instruments"

    for r in results:
        if not r["pos"]["in_pos"]:
            continue
        assert bool(canonical[-2]), (
            f"{r['key']} is still held though the trend broke at "
            f"{hist.index[-2].date()} — it should have exited on the "
            f"{hist.index[-1].date()} bar")


@pytest.mark.parametrize("key", TREND_KEYS)
def test_pending_exit_flag_implies_an_open_position(key, monkeypatch):
    """``exits_next_bar`` is an IN-POSITION state.  A flat sleeve must read
    FLAT — BELOW TREND, never the pending-exit label the action plan shades
    red and the published book records as CLOSE."""
    cfg = tcfg.CONFIGS[key]
    hist = _snapshot(cfg)
    monkeypatch.setattr(ov, "_load_daily", lambda _cfg, _h=hist: _h.copy())

    for r in ov.run_asset(ov.overall_config(key)):
        dec = r["decision"]
        if dec.get("exits_next_bar"):
            assert r["pos"]["in_pos"], (
                f"{r['key']} flags a pending exit while flat: {dec['label']}")
        if not r["pos"]["in_pos"]:
            assert "EXIT NEXT BAR" not in dec["label"]


# ── the root cause: RSI over a window with no down bar ───────────────────
@pytest.mark.parametrize("mod", ["ticker_core", "gldm_core"])
def test_rsi_is_defined_without_a_down_bar(mod):
    """Zero average loss is RSI 100, not NaN — a NaN feature drops the whole
    bar out of ``build_predictions`` and punches the price series."""
    m = __import__(mod)
    up = pd.Series(np.arange(1, 41, dtype=float))
    down = pd.Series(np.arange(40, 0, -1, dtype=float))
    flat = pd.Series(np.full(40, 10.0))

    assert m._rsi(up).iloc[-1] == pytest.approx(100.0)
    assert m._rsi(down).iloc[-1] == pytest.approx(0.0)
    assert m._rsi(flat).iloc[-1] == pytest.approx(50.0)
    for s in (up, down, flat):
        assert m._rsi(s).iloc[14:].notna().all()
        assert m._rsi(s).iloc[:13].isna().all()      # warm-up stays NaN


@pytest.mark.parametrize("key", TREND_KEYS)
def test_no_bar_is_dropped_by_an_unbroken_run_up(key):
    """The specific SOXX gap: a stretch with no down day must not remove bars
    from the frame the strategy walks."""
    cfg = tcfg.CONFIGS[key]
    hist = _snapshot(cfg)
    preds = bt.build_predictions(cfg, hist)
    kept = set(pd.to_datetime(preds["target_date"]))

    delta = hist["px_close"].diff()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    # features are shifted one bar forward, so a NaN at bar D would drop D+1
    zero_loss = hist.index[(loss == 0) & loss.notna()]
    shifted = [hist.index[i + 1] for d in zero_loss
               if (i := hist.index.get_loc(d)) + 1 < len(hist.index)]
    for d in shifted:
        if d < min(kept):
            continue                                  # predates the OOS frame
        assert d in kept, f"{key}: {d.date()} dropped by a zero-loss RSI window"
