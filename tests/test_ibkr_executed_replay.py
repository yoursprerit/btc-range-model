"""IBKR as-published record rebuilt from the shares the executed runs held."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

import account_pnl as ap  # noqa: E402
import overall_core as oc  # noqa: E402


def _rec(on, nl, positions):
    return {"executed_on": on, "as_of": on,
            "payload": {"net_liq": nl, "positions": positions}}


def _pos(sym, shares, px):
    return {"symbol": sym, "shares": shares, "market_price": px}


def _closes():
    idx = pd.bdate_range("2026-10-12", "2026-10-16")
    return pd.DataFrame({"XLE": [100.0, 102.0, 101.0, 99.0, 100.0],
                         "GRID": [50.0, 50.0, 48.0, 49.0, 50.0]}, index=idx)


def _hold(*recs_oldest_first):
    return ap.holdings_from_records(list(reversed(recs_oldest_first)))


def test_holdings_cash_is_net_liq_minus_positions():
    h = _hold(_rec("2026-10-12", 10_000, [_pos("XLE", 40, 100.0)]))
    assert h[0]["cash"] == pytest.approx(6_000)
    assert h[0]["shares"] == {"XLE": 40.0}


def test_replay_moves_with_the_shares_actually_held():
    # target was 40 XLE + 100 GRID but only the XLE leg filled
    h = _hold(_rec("2026-10-12", 10_000, [_pos("XLE", 40, 100.0)]))
    rep = oc.ibkr_executed_replay(h, _closes(), columns=["XLE", "GRID"])
    eq = rep["equity"]
    assert eq.iloc[0] == 1.0
    assert eq.loc["2026-10-13"] == pytest.approx(1 + 40 * 2.0 / 10_000)
    assert eq.loc["2026-10-14"] == pytest.approx(1 + 40 * 1.0 / 10_000)
    assert (rep["weights"]["GRID"] == 0).all()


def test_trade_day_costs_are_not_in_the_curve():
    # the 13th's run buys 100 GRID and ends $20 lower than the market move alone
    h = _hold(_rec("2026-10-12", 10_000, [_pos("XLE", 40, 100.0)]),
              _rec("2026-10-13", 10_080 - 20, [_pos("XLE", 40, 102.0),
                                                _pos("GRID", 20, 50.0)]))
    rep = oc.ibkr_executed_replay(h, _closes())
    gross = 10_000 + 40 * 2.0                     # holdings going INTO the 13th
    assert rep["equity"].loc["2026-10-13"] == pytest.approx(gross / 10_000)
    # …so portfolio value sits above reported net liq by exactly those costs
    pv = 10_000 * rep["equity"].loc["2026-10-13"]
    assert pv - 10_060 == pytest.approx(20)
    # next bar earns what the 13th's run left, over the account after trading
    v_post = rep["net_liq"].loc["2026-10-13"]
    d = (40 * (101 - 102) + 20 * (48 - 50)) / v_post
    assert rep["ret"].loc["2026-10-14"] == pytest.approx(
        rep["equity"].loc["2026-10-14"] / rep["equity"].loc["2026-10-13"] - 1)
    assert rep["ret"].loc["2026-10-14"] == pytest.approx(d)


def test_exited_sleeve_stops_earning():
    h = _hold(_rec("2026-10-12", 10_000, [_pos("GRID", 100, 50.0)]),
              _rec("2026-10-13", 10_000, []))
    rep = oc.ibkr_executed_replay(h, _closes(), columns=["XLE", "GRID"])
    assert rep["ret"].loc["2026-10-14"] == 0 and rep["ret"].loc["2026-10-15"] == 0
    assert rep["cash"].iloc[-1] == pytest.approx(1.0)


def test_sata_without_a_close_accrues_the_coupon():
    h = _hold(_rec("2026-10-12", 10_000, [_pos("SATA", 100, 100.0)]))
    rep = oc.ibkr_executed_replay(h, _closes(), columns=["XLE", "GRID"],
                                  sata_daily=0.001)
    assert rep["equity"].iloc[-1] == pytest.approx(1.001 ** 4)
    assert rep["sata"].iloc[-1] == pytest.approx(1.0)


def test_weights_sum_to_one_and_unknown_keys_are_dropped():
    h = _hold(_rec("2026-10-12", 10_000, [_pos("XLE", 40, 100.0),
                                           _pos("OIH", 5, 300.0)]))
    rep = oc.ibkr_executed_replay(h, _closes(), columns=["XLE", "GRID"])
    assert rep["dropped"] == ["OIH"]
    assert np.allclose(rep["weights"].sum(axis=1) + rep["sata"] + rep["cash"], 1.0)


def test_none_without_holdings_closes_or_two_bars():
    h = _hold(_rec("2026-10-16", 10_000, [_pos("XLE", 40, 100.0)]))
    assert oc.ibkr_executed_replay(None, _closes()) is None
    assert oc.ibkr_executed_replay(h, None) is None
    assert oc.ibkr_executed_replay(h, _closes()) is None      # one bar only


def test_deposits_do_not_move_the_curve():
    h = _hold(_rec("2026-10-12", 10_000, [_pos("XLE", 40, 100.0)]),
              _rec("2026-10-13", 15_080, [_pos("XLE", 40, 102.0)]))   # +$5,000 in
    rep = oc.ibkr_executed_replay(h, _closes())
    assert rep["equity"].loc["2026-10-13"] == pytest.approx(1.008)
