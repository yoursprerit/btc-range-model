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
