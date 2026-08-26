"""Unit tests for the equal-weight buy & hold replay (app/overall_core.py) —
the P&L section's third selectable performance source, the "do nothing"
alternative next to the walk-forward replay and the as-published record.

Synthetic results only (no network, no Streamlit).  The contracts that matter
for the UI: the curve must be the SAME curve as the ``bh_equal`` benchmark line
the Growth charts already draw (the P&L section highlights that row as the
selected source, so a second definition would show two different numbers under
one name); the weights must stack to 100% with nothing in cash; and the exact
per-sleeve attribution must reconcile to the curve, since the 📊 P&L-by-asset
and 📆 Daily P&L toggles run off it unchanged.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import overall_core as oc  # noqa: E402


def _result(key, dates, bh, pos):
    """One traded-instrument result of the shape ``run_universe`` returns —
    only the fields the replay and the per-asset reads actually touch."""
    strat = 100.0 * np.cumprod(1 + np.r_[0.0, np.diff(bh) / bh[:-1]] * pos)
    ret = pd.Series(np.diff(strat) / strat[:-1], index=dates[1:]).rename(key)
    return dict(key=key, name=key, kind="core", parent=key, accent="#000",
                emoji="•", dates=dates, ret=ret,
                pos_series=pd.Series(np.asarray(pos, float), index=dates),
                pos=dict(in_pos=False),
                r=dict(bh=bh, dates=list(dates), trade_log=[]))


def _universe(seed=7, n_days=200, late=60):
    """Three sleeves, one of which starts ``late`` bars in — so the
    renormalise-over-available-sleeves path is always exercised."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n_days)
    out = []
    for key, skip in (("AAA", 0), ("BBB", 0), ("CCC", late)):
        d = idx[skip:]
        bh = 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.012, len(d)))
        bh[0] = 100.0
        out.append(_result(key, d, bh, (rng.random(len(d)) > 0.4).astype(float)))
    return out, idx


def test_curve_is_identical_to_the_bh_equal_benchmark():
    # the P&L section highlights the "Equal-weight Buy & Hold" benchmark row as
    # the selected source — the two must be the same series, not merely close
    results, _ = _universe()
    rets = oc.returns_matrix(results)
    pos = oc.position_matrix(results, rets.index)
    bm = oc.benchmarks(rets, results, pos=pos, sata_daily=oc.SATA_DAILY)
    rep = oc.equal_weight_bh_replay(results, index=rets.index)
    assert rep is not None
    ref = bm["bh_equal"]["equity"]
    assert list(rep["equity"].index) == list(ref.index)
    assert np.max(np.abs(rep["equity"].to_numpy() - ref.to_numpy())) == 0.0


def test_weights_stack_to_100pct_and_nothing_sits_in_cash():
    results, _ = _universe()
    rets = oc.returns_matrix(results)
    rep = oc.equal_weight_bh_replay(results, index=rets.index)
    w = rep["weights"]
    assert list(w.columns) == list(rets.columns)
    wsum = w.sum(axis=1)
    assert np.allclose(wsum[wsum > 0].to_numpy(), 1.0)
    # buy & hold never parks capital — the SATA leg is flat zero, so the
    # coupon can never leak into the curve
    assert float(rep["sata"].abs().max()) == 0.0
    assert rep["n_assets"] == 3


def test_late_starting_sleeve_joins_at_its_first_bar():
    # before CCC has data the equal split is over the two sleeves that do;
    # after it, all three — the renormalisation, not a cash stub
    results, idx = _universe(late=60)
    rets = oc.returns_matrix(results)
    rep = oc.equal_weight_bh_replay(results, index=rets.index)
    w = rep["weights"]
    early, late = w.index[1], w.index[-1]
    assert w.loc[early, "CCC"] == 0.0
    assert np.isclose(w.loc[early, "AAA"], 0.5)
    assert np.allclose(w.loc[late].to_numpy(), 1.0 / 3)


def test_attribution_and_daily_pnl_reconcile_to_the_curve():
    # the 📊 P&L-by-asset and 📆 Daily P&L toggles are fed this triple
    # unchanged, so their parts must sum to the headline slice metrics
    results, _ = _universe()
    rets = oc.returns_matrix(results)
    rep = oc.equal_weight_bh_replay(results, index=rets.index)
    start = rets.index[40]
    sm = oc.slice_metrics(rep["equity"], start)
    att = oc.pnl_attribution_replay(rep["rets"], rep["weights"], rep["sata"],
                                    start, sata_daily=oc.SATA_DAILY)
    dpl = oc.pnl_daily_replay(rep["rets"], rep["weights"], rep["sata"], start,
                              sata_daily=oc.SATA_DAILY)
    assert abs(att["total"] - sm["total_ret"]) < 1e-12
    assert abs(sum(att["per_key"].values()) + att["sata"]
               - sm["total_ret"]) < 1e-9
    assert abs(float(dpl["total"].sum()) - sm["total_ret"]) < 1e-9
    # no idle-cash coupon anywhere in a fully-invested book
    assert att["sata"] == 0.0


def test_returns_matrix_carries_underlying_moves_not_strategy_returns():
    # the whole point of the source: it earns the instrument's own move even
    # on days its signal was flat, unlike returns_matrix
    results, _ = _universe()
    bh = oc.bh_returns_matrix(results)
    strat = oc.returns_matrix(results)
    for res in results:
        px = pd.Series(np.asarray(res["r"]["bh"], float),
                       index=pd.DatetimeIndex(res["dates"]))
        expect = px.pct_change().dropna()
        got = bh[res["key"]].reindex(expect.index)
        assert np.allclose(got.to_numpy(), expect.to_numpy())
    # flat days differ — the strategy earns nothing there, buy & hold does not
    assert not np.allclose(
        np.nan_to_num(bh.reindex(strat.index).to_numpy()),
        np.nan_to_num(strat.to_numpy()))


def test_empty_universe_returns_none():
    assert oc.equal_weight_bh_replay([]) is None
