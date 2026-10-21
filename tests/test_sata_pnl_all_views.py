"""SATA's profit & loss — buys/sells AND dividends — in every P&L view.

The realized-P&L-by-asset ledger carries SATA as a line of its own, and both
executed-fills replays (C2, IBKR) add the dividends a held SATA collects on
top of its price move, so portfolio value and every metric off it follow.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

import account_pnl as ap  # noqa: E402
import overall_core as oc  # noqa: E402

IDX = pd.bdate_range("2026-10-12", periods=5)          # Mon … Fri


def _frame(**cols):
    return pd.DataFrame(cols, index=IDX)


# ── realized P&L by asset: SATA is a ledger line ──────────────────────────
def _book():
    w = _frame(AAA=[0.0, 0.4, 0.4, 0.2, 0.0])
    rets = _frame(AAA=[0.0, 0.10, 0.05, -0.02, 0.01])
    sata = pd.Series([1.0, 0.6, 0.6, 0.8, 1.0], index=IDX)
    return w, rets, sata


def test_sata_is_off_by_default_and_a_line_when_asked():
    w, rets, sata = _book()
    assert "SATA" not in oc.realized_pnl_by_asset(w, sata, IDX[0], rets)
    assert "SATA" in oc.realized_pnl_by_asset(w, sata, IDX[0], rets,
                                              include_sata=True)


@pytest.mark.parametrize("sd", [0.0, oc.SATA_DAILY, 0.002])
def test_sata_realized_plus_unrealized_equals_its_attribution(sd):
    w, rets, sata = _book()
    rlz = oc.realized_pnl_by_asset(w, sata, IDX[0], rets, sata_daily=sd,
                                   include_sata=True)
    att = oc.pnl_attribution_replay(rets, w, sata, IDX[0], sata_daily=sd)
    r = rlz["SATA"]
    assert np.isclose(r["total"], att["sata"])
    assert np.isclose(r["pnl"] + r["unrealized"], att["sata"])
    # the book trimmed SATA once (1.0 → 0.6) to fund AAA
    assert r["n_trims"] == 1 and r["n_sells"] == 0
    assert (r["total"] > 0) == (sd > 0)         # the yield is in the line


def test_sata_buy_sell_price_pnl_is_separate_from_dividends(tmp_path, monkeypatch):
    # a real series: SATA falls 1% on the 14th with no dividend, pays on the rest
    w, rets, sata = _book()
    px = pd.Series([100.0, 100.0, 99.0, 99.0, 99.0], index=IDX)
    csv = pd.DataFrame({"close": px, "dividend": [0.0, 0.05, 0.0, 0.05, 0.05]})
    path = tmp_path / "sata_daily.csv"
    csv.to_csv(path)
    monkeypatch.setattr(oc, "SATA_DATA_PATH", path)
    monkeypatch.setattr(oc, "load_sata_closes",
                        lambda p=path: oc.load_sata_closes.__wrapped__(p)
                        if hasattr(oc.load_sata_closes, "__wrapped__")
                        else px)
    sata_ret = oc.load_sata_returns(path)
    rlz = oc.realized_pnl_by_asset(w, sata, IDX[0], rets, sata_daily=sata_ret,
                                   include_sata=True)["SATA"]
    att = oc.pnl_attribution_replay(rets, w, sata, IDX[0], sata_daily=sata_ret)
    assert np.isclose(rlz["total"], att["sata"])
    assert np.isclose(rlz["pnl"], rlz["pnl_price"] + rlz["pnl_div"])
    assert np.isclose(rlz["unrealized"],
                      rlz["unrealized_price"] + rlz["unrealized_div"])
    # a price drop with a dividend that does not cover it: price leg negative
    assert rlz["pnl_price"] + rlz["unrealized_price"] < 0
    assert rlz["pnl_div"] + rlz["unrealized_div"] > 0


# ── executed replays: dividends on a held SATA ────────────────────────────
DIVS = pd.Series([0.05, 0.05, 0.05], index=pd.to_datetime(
    ["2026-10-13", "2026-10-14", "2026-10-15"]))


def _rec(on, nl, positions):
    return {"executed_on": on, "as_of": on,
            "payload": {"net_liq": nl, "positions": positions}}


def _pos(sym, shares, px):
    return {"symbol": sym, "shares": shares, "market_price": px}


def _closes():
    return pd.DataFrame({"XLE": [100.0] * 5, "SATA": [100.0] * 5}, index=IDX)


def test_ibkr_executed_replay_earns_the_dividends_on_held_sata():
    h = ap.holdings_from_records([_rec("2026-10-12", 10_000,
                                       [_pos("SATA", 100, 100.0)])])
    flat = oc.ibkr_executed_replay(h, _closes(), columns=["XLE"])
    paid = oc.ibkr_executed_replay(h, _closes(), columns=["XLE"], sata_div=DIVS)
    assert np.allclose(flat["equity"], 1.0)           # price alone: nothing
    # 100 shares × $0.05 on each of the 13th, 14th and 15th
    assert paid["equity"].loc["2026-10-13"] == pytest.approx(1 + 5 / 10_000)
    assert paid["equity"].loc["2026-10-15"] == pytest.approx(
        (10_000 + 15) / 10_000 * 1.0)
    assert paid["equity"].iloc[-1] == pytest.approx(paid["equity"].loc["2026-10-15"])
    assert np.allclose(paid["weights"].sum(axis=1) + paid["sata"] + paid["cash"], 1.0)


def test_ibkr_dividends_are_not_double_counted_across_runs():
    # a second run on the 14th restates cash including the dividends so far
    h = ap.holdings_from_records(list(reversed([
        _rec("2026-10-12", 10_000, [_pos("SATA", 100, 100.0)]),
        _rec("2026-10-14", 10_010, [_pos("SATA", 100, 100.0)])])))
    paid = oc.ibkr_executed_replay(h, _closes(), columns=["XLE"], sata_div=DIVS)
    assert paid["equity"].loc["2026-10-14"] == pytest.approx(1 + 10 / 10_000)
    assert paid["equity"].loc["2026-10-15"] == pytest.approx(
        (1 + 10 / 10_000) * (10_010 + 5) / 10_010)


def _snap(positions, closed=None):
    return {"schema": "c2-positions/v1",
            "fetched_at_utc": "2026-10-16T19:45:00+00:00",
            "model_account_value": 50_000.0, "starting_cash": 50_000.0,
            "positions": positions, "closed_trades": closed or []}


def test_c2_executed_replay_earns_the_dividends_on_held_sata():
    pos = {"key": "SATA", "symbol": "SATA", "shares": 100, "avg_cost": 100.0,
           "opened": "2026-10-12T19:30:00Z"}
    snap = _snap([pos])
    flat = oc.c2_executed_replay(snap, [snap], _closes())
    paid = oc.c2_executed_replay(snap, [snap], _closes(), sata_div=DIVS)
    assert np.allclose(flat["equity"], 1.0)
    assert (paid["equity"].iloc[-1] - 1) * 50_000 == pytest.approx(15.0)


def test_c2_closed_sata_trade_keeps_its_dividends_and_price_pnl():
    # bought on the 12th, sold on the 15th at 99: −$100 price, 3 × $0.05 × 100 shares
    sold = {"trade_id": 1, "key": "SATA", "symbol": "SATA", "side": "long",
            "quantity": 100, "entry_px": 100.0, "exit_px": 99.0,
            "opened": "2026-10-12T19:30:00Z", "closed": "2026-10-15T19:30:00Z",
            "pnl": -100.0, "commission": 0.0}
    snap = _snap([], closed=[sold])
    paid = oc.c2_executed_replay(snap, [snap], _closes(), sata_div=DIVS)
    assert (paid["equity"].iloc[-1] - 1) * 50_000 == pytest.approx(-100.0 + 15.0)
    # held through the 14th close: only two dividends so far, price at the close
    assert (paid["equity"].loc["2026-10-14"] - 1) * 50_000 == pytest.approx(10.0)
