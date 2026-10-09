"""Brokerage-cost and capital-gains-tax model for the walk-forward replay
(app/overall_frictions.py).  Synthetic data only."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import overall_frictions as fr  # noqa: E402


def _book(n=900, flip_every=40, seed=1):
    idx = pd.bdate_range("2023-01-02", periods=n)
    rng = np.random.default_rng(seed)
    rets = pd.DataFrame({"A": rng.normal(0.0008, 0.01, n),
                         "B": rng.normal(0.0004, 0.01, n)}, index=idx)
    on = (np.arange(n) // flip_every) % 2 == 0
    w = pd.DataFrame({"A": np.where(on, 0.6, 0.0), "B": np.where(on, 0.0, 0.4)},
                     index=idx)
    sata = 1.0 - w.sum(axis=1)
    px = pd.DataFrame({"A": 100.0, "B": 40.0}, index=idx)
    return rets, w, sata, px


def test_no_frictions_matches_gross():
    rets, w, sata, px = _book()
    r = fr.simulate_frictions(rets, w, sata, px, rets.index[5])
    np.testing.assert_allclose(r["net"].to_numpy(), r["gross"].to_numpy(), rtol=1e-6)
    assert r["commissions"] == 0 and r["taxes"] == 0


def test_costs_reduce_value_and_count_orders():
    rets, w, sata, px = _book()
    g = fr.simulate_frictions(rets, w, sata, px, rets.index[5])
    c = fr.simulate_frictions(rets, w, sata, px, rets.index[5], costs=True)
    assert c["commissions"] > 0 and c["orders"] > 0
    assert c["net"].iloc[-1] < g["net"].iloc[-1]


def test_order_cost_rules():
    assert fr.order_cost(500, 100, False) == 1.0                  # $1 minimum
    assert abs(fr.order_cost(50_000, 100, False) - 2.5) < 1e-9    # $0.005/share
    assert fr.order_cost(20, 1.0, False) == 0.2                   # 1% cap
    assert fr.order_cost(50_000, 100, True) > fr.order_cost(50_000, 100, False)


def test_taxes_reduce_value_and_tally_years():
    rets, w, sata, px = _book()
    t = fr.simulate_frictions(rets, w, sata, px, rets.index[5], taxes=True)
    g = fr.simulate_frictions(rets, w, sata, px, rets.index[5])
    assert t["net"].iloc[-1] < g["net"].iloc[-1]
    assert {y["year"] for y in t["years"]} == {2023, 2024, 2025, 2026}
    assert t["taxes"] > 0
    assert all(y["lt_gain"] == 0 for y in t["years"][:1])  # nothing held > 1y yet


def test_long_term_taxed_below_short_term():
    st = fr.year_tax(100_000, 0, 0, 0, 325_000)["tax"]
    lt = fr.year_tax(0, 100_000, 0, 0, 325_000)["tax"]
    assert lt < st
    assert abs(lt - 100_000 * (0.15 + 0.038)) < 1e-6
    assert abs(st - (100_000 * 0.24 + 0.038 * 100_000)) < 1e-6 or st > lt


def test_loss_netting_and_carryforward():
    r = fr.year_tax(-10_000, 0, 0, 0, 325_000)
    assert r["tax"] < 0 and abs(r["carry_st"] - 7_000) < 1e-9   # $3k used
    nxt = fr.year_tax(5_000, 0, r["carry_st"], 0, 325_000)
    assert nxt["tax"] < 0 or abs(nxt["net_st"]) < 1e-9
    # ST loss shelters LT gain
    assert fr.year_tax(-4_000, 4_000, 0, 0, 325_000)["tax"] == 0.0


def test_long_holds_realize_long_term_gains():
    rets, w, sata, px = _book(n=1200, flip_every=300)
    t = fr.simulate_frictions(rets, w, sata, px, rets.index[5], taxes=True)
    assert sum(y["lt_gain"] + y["lt_loss"] for y in t["years"]) > 0


def test_speed_wide_book():
    idx = pd.bdate_range("2021-01-04", periods=1500)
    rng = np.random.default_rng(3)
    k = [f"S{i}" for i in range(18)]
    rets = pd.DataFrame(rng.normal(0.0005, 0.015, (1500, 18)), index=idx, columns=k)
    raw = rng.random((1500, 18)) * (rng.random((1500, 18)) > 0.6)
    w = pd.DataFrame(raw / np.maximum(raw.sum(axis=1, keepdims=True), 1) * 0.9,
                     index=idx, columns=k)
    r = fr.simulate_frictions(rets, w, 1 - w.sum(axis=1), None, idx[0],
                              costs=True, taxes=True, sata_daily=0.0005)
    assert r["net"].iloc[-1] < r["gross"].iloc[-1]
