"""Unit tests for ``daily_trade_log`` (app/overall_core.py) — the P&L
section's day-by-day buy/sell log.

Synthetic weight matrices only (no network, no Streamlit): the buy/sell vs
add/trim (signal change vs daily tilt) classification, execution-close row
dating under the decided-at-previous-close convention, the published-book
turnover convention (½·(Σ|Δw| + |Δcash|)), the ``min_delta`` noise floor,
quiet-day omission, newest-first ordering, and that the SAME function runs
off both P&L sources (the as-published record's ``published_book_replay``
weights feed it unchanged).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import overall_core as oc  # noqa: E402


# Mon 2026-07-06 … Fri 2026-07-10 — five business days
IDX = pd.date_range("2026-07-06", periods=5, freq="D")


def _frame(**cols):
    return pd.DataFrame(cols, index=IDX)


def _sata(values):
    return pd.Series(values, index=IDX, dtype=float)


def test_buy_sell_are_signal_changes_and_resizes_are_tilts():
    w = _frame(AAA=[0.0, 0.4, 0.5, 0.5, 0.0],
               BBB=[0.6, 0.6, 0.6, 0.6, 0.6])
    log = oc.daily_trade_log(w, _sata([0.4, 0.0, 0.0, 0.0, 0.4]), IDX[0])
    by_date = {d["date"]: d for d in log}
    # 0 → 0.4: a buy (signal opened the sleeve)
    buy = by_date[IDX[0]]["actions"][0]
    assert (buy["key"], buy["action"], buy["signal_change"]) == ("AAA", "buy", True)
    assert np.isclose(buy["delta"], 0.4)
    # 0.4 → 0.5 while held: a tilt add, not a signal change
    add = by_date[IDX[1]]["actions"][0]
    assert (add["action"], add["signal_change"]) == ("add", False)
    # 0.5 → 0: a sell (signal closed the sleeve)
    sell = by_date[IDX[3]]["actions"][0]
    assert (sell["key"], sell["action"], sell["signal_change"]) == ("AAA", "sell", True)
    assert np.isclose(sell["w0"], 0.5)
    # BBB never moved — it must never appear as an action
    assert all(a["key"] != "BBB" for d in log for a in d["actions"])


def test_trim_while_held_is_a_tilt_adjustment():
    w = _frame(AAA=[0.5, 0.3, 0.3, 0.3, 0.3])
    log = oc.daily_trade_log(w, _sata([0.5, 0.7, 0.7, 0.7, 0.7]), IDX[0])
    assert len(log) == 1
    a = log[0]["actions"][0]
    assert (a["action"], a["signal_change"]) == ("trim", False)
    assert np.isclose(a["delta"], -0.2)


def test_rows_dated_at_the_execution_close():
    # the row-d book earns day d but was put on at close of d−1 (the
    # decided-at-previous-close convention) — the diff between rows d−1 and
    # d is the ticket executed at close of d−1, so that's the row's date
    w = _frame(AAA=[0.0, 0.0, 1.0, 1.0, 1.0])
    log = oc.daily_trade_log(w, _sata([1.0, 1.0, 0.0, 0.0, 0.0]), IDX[0])
    assert len(log) == 1
    assert log[0]["date"] == IDX[1]


def test_turnover_matches_the_published_book_convention():
    # ½ × (Σ|Δweight| + |Δcash|) — same maths as published_book_replay's trims
    w = _frame(AAA=[0.6, 0.4, 0.4, 0.4, 0.4],
               BBB=[0.4, 0.4, 0.4, 0.4, 0.4])
    log = oc.daily_trade_log(w, _sata([0.0, 0.2, 0.2, 0.2, 0.2]), IDX[0])
    assert len(log) == 1
    assert np.isclose(log[0]["turnover"], 0.5 * (0.2 + 0.2))
    assert np.isclose(log[0]["gross"], 0.2)
    assert np.isclose(log[0]["cash0"], 0.0)
    assert np.isclose(log[0]["cash1"], 0.2)


def test_quiet_days_omitted_and_noise_below_min_delta_ignored():
    w = _frame(AAA=[0.5, 0.5001, 0.5, 0.5, 0.5])          # sub-min_delta noise
    assert oc.daily_trade_log(w, _sata([0.5] * 5), IDX[0]) == []


def test_newest_first_and_actions_sorted_by_move_size():
    w = _frame(AAA=[0.0, 0.1, 0.1, 0.1, 0.5],
               BBB=[0.0, 0.4, 0.4, 0.4, 0.2])
    log = oc.daily_trade_log(w, _sata([1.0, 0.5, 0.5, 0.5, 0.3]), IDX[0])
    assert [d["date"] for d in log] == [IDX[3], IDX[0]]    # newest first
    # within a day the biggest |Δweight| leads
    assert [a["key"] for a in log[1]["actions"]] == ["BBB", "AAA"]
    assert [a["key"] for a in log[0]["actions"]] == ["AAA", "BBB"]


def test_per_day_counts():
    w = _frame(AAA=[0.0, 0.3, 0.3, 0.3, 0.3],
               BBB=[0.5, 0.0, 0.0, 0.0, 0.0],
               CCC=[0.2, 0.3, 0.3, 0.3, 0.3])
    log = oc.daily_trade_log(w, _sata([0.3, 0.4, 0.4, 0.4, 0.4]), IDX[0])
    d = log[0]
    assert (d["n_buys"], d["n_sells"], d["n_resize"]) == (1, 1, 1)


def test_start_filter_and_too_short_window():
    w = _frame(AAA=[0.0, 1.0, 1.0, 0.0, 0.0])
    sata = _sata([1.0, 0.0, 0.0, 1.0, 1.0])
    # anchored after the buy: only the later sell remains
    late = oc.daily_trade_log(w, sata, IDX[2])
    assert [a["action"] for d in late for a in d["actions"]] == ["sell"]
    # <2 rows on/after start → nothing to diff
    assert oc.daily_trade_log(w, sata, IDX[4]) == []


def test_trim_pnl_measured_from_the_entry_close():
    # bought at close of IDX0 (earning rows 1-2), trimmed at close of IDX2 —
    # the slice sold realized (1+r1)(1+r2)−1; entry_date is the buy close
    w = _frame(AAA=[0.0, 0.4, 0.4, 0.2, 0.2])
    rets = _frame(AAA=[0.03, 0.10, 0.05, -0.02, 0.01])
    log = oc.daily_trade_log(w, _sata([1.0, 0.6, 0.6, 0.8, 0.8]), IDX[0],
                             returns=rets)
    trim = next(a for d in log for a in d["actions"] if a["action"] == "trim")
    assert np.isclose(trim["pnl"], 1.10 * 1.05 - 1)
    assert trim["entry_date"] == IDX[0]
    # the buy itself carries no P&L
    buy = next(a for d in log for a in d["actions"] if a["action"] == "buy")
    assert buy["pnl"] is None and buy["entry_date"] is None


def test_full_sell_also_carries_pnl():
    w = _frame(AAA=[0.0, 0.4, 0.4, 0.0, 0.0])
    rets = _frame(AAA=[0.03, 0.10, 0.05, -0.02, 0.01])
    log = oc.daily_trade_log(w, _sata([1.0, 0.6, 0.6, 1.0, 1.0]), IDX[0],
                             returns=rets)
    sell = next(a for d in log for a in d["actions"] if a["action"] == "sell")
    assert np.isclose(sell["pnl"], 1.10 * 1.05 - 1)
    assert sell["entry_date"] == IDX[0]


def test_pnl_none_without_returns_or_return_column():
    w = _frame(AAA=[0.5, 0.5, 0.3, 0.3, 0.3])
    sata = _sata([0.5, 0.5, 0.7, 0.7, 0.7])
    a = oc.daily_trade_log(w, sata, IDX[0])[0]["actions"][0]
    assert a["pnl"] is None and a["entry_date"] is None
    rets = _frame(BBB=[0.0, 0.0, 0.0, 0.0, 0.0])   # no AAA column
    a = oc.daily_trade_log(w, sata, IDX[0], returns=rets)[0]["actions"][0]
    assert a["pnl"] is None


def test_held_from_matrix_start_measures_from_the_first_close():
    # already held at row 0 → the anchor bar is the cost basis: only r1
    # counts for a trim executed at close of IDX1 (r0 excluded)
    w = _frame(AAA=[0.5, 0.5, 0.3, 0.3, 0.3])
    rets = _frame(AAA=[0.20, 0.10, 0.05, 0.0, 0.0])
    log = oc.daily_trade_log(w, _sata([0.5, 0.5, 0.7, 0.7, 0.7]), IDX[0],
                             returns=rets)
    a = log[0]["actions"][0]
    assert np.isclose(a["pnl"], 0.10)
    assert a["entry_date"] == IDX[0]


def test_pnl_uses_the_true_entry_before_the_window_anchor():
    # bought at close of IDX0, window anchored at IDX2 — the trim executed
    # at close of IDX3 still measures from the real buy: (1+r1)(1+r2)(1+r3)
    w = _frame(AAA=[0.0, 0.4, 0.4, 0.4, 0.2])
    rets = _frame(AAA=[0.03, 0.10, 0.05, 0.02, -0.01])
    log = oc.daily_trade_log(w, _sata([1.0, 0.6, 0.6, 0.6, 0.8]), IDX[2],
                             returns=rets)
    assert len(log) == 1 and log[0]["date"] == IDX[3]
    a = log[0]["actions"][0]
    assert np.isclose(a["pnl"], 1.10 * 1.05 * 1.02 - 1)
    assert a["entry_date"] == IDX[0]


def test_pnl_restarts_at_reentry():
    # closed and re-opened — the later trim measures from the SECOND buy
    w = _frame(AAA=[0.4, 0.0, 0.4, 0.4, 0.2])
    rets = _frame(AAA=[0.03, 0.10, 0.05, 0.02, -0.01])
    log = oc.daily_trade_log(w, _sata([0.6, 1.0, 0.6, 0.6, 0.8]), IDX[0],
                             returns=rets)
    trim = next(a for d in log for a in d["actions"] if a["action"] == "trim")
    assert np.isclose(trim["pnl"], 1.05 * 1.02 - 1)   # rows 2-3 only
    assert trim["entry_date"] == IDX[1]


def test_runs_off_the_as_published_replay_unchanged():
    # the P&L source toggle swaps the weight matrix under the same call —
    # published_book_replay's weights/sata must feed it directly
    rets = pd.DataFrame({"AAA": 0.0, "BBB": 0.0}, index=IDX)
    books = [
        {"schema": "overall-target-book/v1", "as_of": "2026-07-06",
         "profile": "Balanced", "book_mode": "live",
         "weights": {"AAA": 0.6}, "cash_weight": 0.4, "exec_price": {}},
        {"schema": "overall-target-book/v1", "as_of": "2026-07-08",
         "profile": "Balanced", "book_mode": "live",
         "weights": {"AAA": 0.4, "BBB": 0.3}, "cash_weight": 0.3,
         "exec_price": {}},
    ]
    rep = oc.published_book_replay(rets, books, sata_daily=0.0)
    log = oc.daily_trade_log(rep["weights"], rep["sata"], IDX[0])
    assert len(log) == 1
    # book(07-08) starts earning 07-09 → executed at the 07-08 close
    assert log[0]["date"] == pd.Timestamp("2026-07-08")
    acts = {a["key"]: a for a in log[0]["actions"]}
    assert acts["BBB"]["action"] == "buy" and acts["BBB"]["signal_change"]
    assert acts["AAA"]["action"] == "trim" and not acts["AAA"]["signal_change"]
    # ½ × (|Δ AAA| + |Δ BBB| + |Δcash|) = ½ × (0.2 + 0.3 + 0.1)
    assert np.isclose(log[0]["turnover"], 0.3)
