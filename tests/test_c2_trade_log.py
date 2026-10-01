"""Collective2 fills → the as-published view's trade log and account figures."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import overall_core as oc  # noqa: E402


def _snap(fetched, positions, closed=None, value=50_000.0, start=50_000.0):
    return {"schema": "c2-positions/v1", "fetched_at_utc": fetched,
            "model_account_value": value, "starting_cash": start,
            "positions": positions, "closed_trades": closed or []}


def _pos(sym, shares, avg, opened="2026-09-24T14:01:58Z"):
    return {"key": sym, "symbol": sym, "shares": shares, "avg_cost": avg,
            "opened": opened}


FRI = _snap("2026-09-25T19:31:29+00:00",
            [_pos("WGMI", 178, 49.59), _pos("GRID", 82, 177.99)], value=49_986.0)
WGMI_CLOSED = {"trade_id": 9, "key": "WGMI", "symbol": "WGMI", "side": "long",
               "quantity": 178, "entry_px": 49.59, "exit_px": 46.05,
               "opened": "2026-09-24T14:01:58Z", "closed": "2026-09-28T19:31:02Z",
               "pnl": -631.12, "commission": 1.0}


def test_closed_rows_use_c2_fill_prices_and_new_york_dates():
    mon = _snap("2026-09-28T19:45:00+00:00", [_pos("GRID", 82, 177.99)],
                closed=[WGMI_CLOSED], value=48_803.0)
    rows = oc.c2_trade_log(mon, [FRI, mon], prices={"GRID": 180.0})
    closed = [r for r in rows if not r["open"]]
    assert len(closed) == 1 and not closed[0]["pending"]
    w = closed[0]
    assert (w["entry_date"], w["exit_date"]) == (pd.Timestamp("2026-09-24"),
                                                 pd.Timestamp("2026-09-28"))
    assert (w["entry_px"], w["exit_px"]) == (49.59, 46.05)
    assert np.isclose(w["pnl_usd"] * 50_000, -631.12)
    assert np.isclose(w["ret"], 46.05 / 49.59 - 1)
    g = rows[0]
    assert g["open"] and g["key"] == "GRID" and g["exit_px"] == 180.0
    assert np.isclose(g["pnl_usd"] * 50_000, 82 * (180.0 - 177.99))


def test_a_position_that_vanished_without_a_c2_trade_yet_is_pending():
    mon = _snap("2026-09-28T19:31:34+00:00", [_pos("GRID", 82, 177.99)],
                value=48_803.0)
    rows = oc.c2_trade_log(mon, [FRI, mon])
    pend = [r for r in rows if r["pending"]]
    assert [(r["key"], r["exit_date"]) for r in pend] == \
        [("WGMI", pd.Timestamp("2026-09-28"))]
    assert pend[0]["exit_px"] is None and pend[0]["entry_px"] == 49.59


def test_account_summary_is_per_dollar_of_starting_cash():
    mon = _snap("2026-09-28T19:45:00+00:00", [_pos("GRID", 82, 177.99)],
                closed=[WGMI_CLOSED], value=48_803.0)
    s = oc.c2_account_summary(mon, [FRI, mon], prices={"GRID": 180.0})
    assert np.isclose(s["value"] * 100_000, 97_606.0)       # $100k-calibrated
    assert np.isclose(s["total_ret"], 48_803 / 50_000 - 1)
    assert np.isclose(s["realized"] * 50_000, -631.12)
    assert s["n_closed"] == 1 and s["wins"] == 0 and s["n_pending"] == 0
    assert s["curve"].index[0] == pd.Timestamp("2026-09-23")
    assert s["curve"].iloc[0] == 1.0 and s["mdd"] < 0
    assert s["starting_cash_known"]


def test_missing_starting_cash_falls_back_to_the_default():
    snap = dict(_snap("2026-09-25T19:31:29+00:00", []), starting_cash=None)
    assert oc.c2_starting_cash(snap) == oc.C2_STARTING_CASH_DEFAULT
    assert not oc.c2_account_summary(snap)["starting_cash_known"]


def test_headline_return_is_c2s_own_figure_in_either_unit():
    base = _snap("2026-09-28T20:16:00+00:00", [], value=48_803.0)
    for raw in (-2.8, -0.028):                          # percent or fraction
        s = oc.c2_account_summary(dict(base, c2_return=raw))
        assert np.isclose(s["total_ret"], -0.028)
        assert np.isclose(s["value"] * 100_000, 97_200.0)
        assert np.isclose(s["value_ret"], 48_803 / 50_000 - 1)
    s = oc.c2_account_summary(base)                     # none yet: value-based
    assert s["c2_return"] is None and np.isclose(s["total_ret"], s["value_ret"])


def test_c2_replay_fills_a_weekend_book_at_the_next_session():
    # Thu book holds EQ; the Sunday book exits it. C2 sells at MONDAY's close,
    # so Monday's −4% is still earned; Tuesday is not.
    idx = pd.date_range("2026-09-24", "2026-09-29", freq="D")    # Thu..Tue
    rets = pd.DataFrame({"EQ": [0.0, 0.01, 0.0, 0.0, -0.04, 0.02]}, index=idx)
    books = [{"as_of": "2026-09-24", "weights": {"EQ": 1.0}, "cash_weight": 0.0},
             {"as_of": "2026-09-27", "weights": {}, "cash_weight": 1.0}]
    lag = oc.published_book_replay(rets, books, sata_daily=0.0, fill_on_sessions=True)
    assert np.allclose(lag["ret"].to_numpy(), [0, 0.01, 0, 0, -0.04, 0])
    # without the session rule the exit is booked at Friday's close
    old = oc.published_book_replay(rets, books, sata_daily=0.0)
    assert np.allclose(old["ret"].to_numpy(), [0, 0.01, 0, 0, 0, 0])


def test_c2_value_that_contradicts_its_own_fills_falls_back_to_the_ledger():
    # 2026-09-30: C2 reported $33,794 (−32.5%) for the same six holdings that
    # were worth $48.5k the day before — its mark dropped GRID.  Marked at
    # market, the fills give ≈ −3.4%, and the cash is what the fills left.
    held = [_pos("GRID", 82, 177.99), _pos("OIH", 22, 394.35545),
            _pos("GLDM", 99, 83.39576), _pos("XLE", 113, 62.56274),
            _pos("ERX", 47, 105.2), _pos("UGL", 101, 48.29)]
    px = {"GRID": 177.11, "OIH": 378.37, "GLDM": 82.18, "XLE": 61.5,
          "ERX": 100.42, "UGL": 45.73}
    wgmi = dict(WGMI_CLOSED, exit_px=46.36, pnl=-578.5)
    tue = dict(_snap("2026-09-29T21:43:56+00:00", held, [wgmi], value=48_517.0),
               c2_return=-3.05, cash=23_980.1)
    wed = dict(_snap("2026-09-30T21:44:31+00:00", held, [wgmi], value=33_794.0),
               c2_return=-32.5, cash=23_980.0)
    s = oc.c2_account_summary(wed, [FRI, tue, wed], prices=px)
    ledger = (-578.5 + sum(p["shares"] * (px[p["key"]] - p["avg_cost"])
                           for p in held)) / 50_000
    assert s["c2_mismatch"] and np.isclose(s["total_ret"], ledger)
    assert -0.04 < s["total_ret"] < -0.03
    assert np.isclose(s["value"] * 100_000, 100_000 * (1 + ledger))
    assert np.isclose(s["c2_return"], -0.325)              # still reported
    cost = sum(p["shares"] * p["avg_cost"] for p in held)
    assert np.isclose(s["cash"] * 50_000, 50_000 - 578.5 - cost)
    assert np.isclose(s["curve"].iloc[-1], 1 + ledger) and s["mdd"] > -0.05
    # a C2 figure in line with the fills is kept as reported
    ok = oc.c2_account_summary(tue, [FRI, tue],
                               prices={k: v + 0.5 for k, v in px.items()})
    assert not ok["c2_mismatch"] and np.isclose(ok["total_ret"], -0.0305)
