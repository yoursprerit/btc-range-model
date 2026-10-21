"""Strategy P&L ``Portfolio value now`` → C2 ``Total portfolio value`` bridge."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import overall_core as oc  # noqa: E402

DAYS = pd.bdate_range("2026-10-05", periods=6)
EXEC = pd.Series([1.0, 1.02, 1.04, 1.03, 1.05, 1.08], index=DAYS)
BOOKS = pd.Series([1.0, 1.03, 1.06, 1.05, 1.08, 1.12], index=DAYS)


def _acct(value_ret, ledger_ret=None):
    return {"value_ret": value_ret, "ledger_ret": ledger_ret}


def _sum(rc):
    return rc["strategy_value"] + sum(s["delta"] for s in rc["steps"])


def test_steps_add_up_to_the_c2_value():
    pv = 100_000.0
    start, end = DAYS[2], DAYS[4]
    strategy = pv * EXEC[end] / EXEC[start] * 0.99           # a 1 % tax drag
    rc = oc.c2_reconciliation(_acct(0.075, 0.0795), EXEC, start, end, pv,
                              strategy, book_eq=BOOKS)
    assert np.isclose(_sum(rc), pv * 1.075)
    assert np.isclose(rc["c2_value"], pv * 1.075)
    labels = [s["label"] for s in rc["steps"]]
    assert len(labels) == 5 and "Taxes" in labels[0]
    d = {s["label"].split(" ")[0]: s["delta"] for s in rc["steps"]}
    assert np.isclose(d["Taxes"], pv * EXEC[end] / EXEC[start] * 0.01)
    # P&L before the start date: the re-based window omits the first 4 %
    assert np.isclose(d["P&L"], pv * (EXEC[end] - EXEC[end] / EXEC[start]))
    assert np.isclose(d["Bars"], pv * (EXEC.iloc[-1] - EXEC[end]))


def test_window_from_the_first_bar_has_no_start_or_end_step():
    pv = 50_000.0
    rc = oc.c2_reconciliation(_acct(0.08, 0.08), EXEC, DAYS[0], DAYS[-1], pv,
                              pv * EXEC.iloc[-1])
    deltas = {s["label"].split(" ")[0]: s["delta"] for s in rc["steps"]}
    assert np.isclose(deltas["Taxes"], 0) and np.isclose(deltas["P&L"], 0)
    assert np.isclose(deltas["Bars"], 0) and np.isclose(deltas["C2's"], 0)
    assert np.isclose(_sum(rc), rc["c2_value"])


def test_no_ledger_skips_the_marks_step_and_still_adds_up():
    pv = 100_000.0
    rc = oc.c2_reconciliation(_acct(-0.30), EXEC, DAYS[0], DAYS[-1], pv,
                              pv * EXEC.iloc[-1])
    assert len(rc["steps"]) == 4
    assert not any("marks" in s["label"] for s in rc["steps"])
    assert np.isclose(_sum(rc), pv * 0.70)
    assert rc["steps"][-1]["delta"] < 0          # C2's report is the outlier


def test_execution_shortfall_is_executed_minus_books_over_the_same_bars():
    pv = 100_000.0
    rc = oc.c2_reconciliation(_acct(0.08, 0.08), EXEC, DAYS[0], DAYS[-1], pv,
                              pv * EXEC.iloc[-1], book_eq=BOOKS)
    ex = rc["execution"]
    assert np.isclose(ex["book_ret"], 0.12) and np.isclose(ex["exec_ret"], 0.08)
    assert np.isclose(ex["shortfall"], -0.04)
    assert np.isclose(ex["shortfall_usd"], -4_000.0)


def test_books_are_aligned_to_the_executed_bars():
    longer = pd.concat([pd.Series([0.9], index=[DAYS[0] - pd.Timedelta(days=7)]),
                        BOOKS])                  # earlier history is ignored
    rc = oc.c2_reconciliation(_acct(0.08, 0.08), EXEC, DAYS[0], DAYS[-1], 1e5,
                              1e5 * EXEC.iloc[-1], book_eq=longer)
    assert np.isclose(rc["execution"]["book_ret"], 0.12)


def test_nothing_to_bridge_without_an_account_or_curve():
    assert oc.c2_reconciliation(None, EXEC, DAYS[0], DAYS[-1], 1e5, 1e5) is None
    assert oc.c2_reconciliation(_acct(0.0), None, DAYS[0], DAYS[-1], 1e5, 1e5) is None
    assert oc.c2_reconciliation(_acct(0.0), EXEC.iloc[:1], DAYS[0], DAYS[0],
                                1e5, 1e5) is None
