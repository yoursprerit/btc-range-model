"""Monthly-deposit dollar P&L for an IBKR account's as-published record."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

import account_pnl as ap  # noqa: E402
import executed_book as eb  # noqa: E402


def _bars(start="2026-10-07", end="2027-03-31"):
    return pd.bdate_range(start, end)


def _flat(idx):
    return pd.Series(100.0, index=idx)


def test_no_deposits_is_just_the_curve_scaled():
    idx = _bars()
    curve = pd.Series(np.linspace(100, 130, len(idx)), index=idx)
    sim = ap.simulate_account(curve, idx[0], 50_000, 0.0)
    assert sim["value"].iloc[-1] == pytest.approx(50_000 * 1.30)
    s = ap.window_summary(sim, idx[0])
    assert s["deposits"] == 0 and s["pnl"] == pytest.approx(15_000)


def test_flat_market_deposits_are_not_profit():
    """The whole point: money you add is not strategy P&L."""
    idx = _bars()
    sim = ap.simulate_account(_flat(idx), idx[0], 10_000, 1_000, day=1)
    s = ap.window_summary(sim, idx[0])
    assert s["n_deposits"] == 5            # Nov, Dec, Jan, Feb, Mar
    assert s["deposits"] == 5_000
    assert s["end_value"] == pytest.approx(15_000)
    assert s["pnl"] == pytest.approx(0.0)


def test_deposit_earns_only_from_the_next_bar():
    idx = pd.bdate_range("2026-10-28", "2026-11-06")
    curve = pd.Series([100, 100, 100, 100, 110, 110, 110, 110.0], index=idx[:8])
    # Nov 2026: the 2nd is a Monday bar; deposit credited at that close
    sim = ap.simulate_account(curve, idx[0], 1_000, 500, day=2)
    dep_bar = sim.index[sim["deposit"] > 0][0]
    assert dep_bar == pd.Timestamp("2026-11-02")
    # the +10% on Nov 2 happens BEFORE the deposit lands (curve jumps 100->110 on the 2nd? it is idx[4]=Nov 3)
    assert sim["value"].loc[dep_bar] == pytest.approx(1_000 + 500)
    nxt = sim.index[sim.index.get_loc(dep_bar) + 1]
    assert sim["value"].loc[nxt] == pytest.approx(1_500 * 1.10)


def test_deposit_day_rolls_forward_over_weekends():
    idx = _bars("2026-10-07", "2026-12-31")
    bars = ap.deposit_bars(idx, idx[0], day=1)
    assert pd.Timestamp("2026-11-02") in bars          # Nov 1 is a Sunday
    assert pd.Timestamp("2026-12-01") in bars
    assert all(b.weekday() < 5 for b in bars)


def test_first_month_has_no_deposit_and_future_months_are_not_due():
    idx = _bars("2026-10-07", "2026-11-15")
    bars = ap.deposit_bars(idx, idx[0], day=20)         # Nov 20 > last bar
    assert bars == []
    bars = ap.deposit_bars(idx, idx[0], day=10)
    assert bars == [pd.Timestamp("2026-11-10")]


def test_day_is_clamped_to_a_day_every_month_has():
    assert ap.clamp_day(31) == 28 and ap.clamp_day(0) == 1 and ap.clamp_day("x") == 1


def test_window_start_later_than_the_record_start():
    idx = _bars()
    sim = ap.simulate_account(_flat(idx), idx[0], 10_000, 1_000, day=1)
    s = ap.window_summary(sim, "2026-12-15")
    # from Dec 15 the value is 12,000 (Oct start + Nov, Dec deposits); Jan-Mar add 3,000
    assert s["start_value"] == pytest.approx(12_000)
    assert s["n_deposits"] == 3 and s["pnl"] == pytest.approx(0.0)


def test_pnl_matches_time_weighted_return_when_deposits_are_zero_and_positive_when_up():
    idx = _bars("2026-10-07", "2027-01-29")
    curve = pd.Series(np.linspace(100, 120, len(idx)), index=idx)
    with_dep = ap.window_summary(ap.simulate_account(curve, idx[0], 10_000, 500), idx[0])
    assert with_dep["pnl"] > 0
    assert with_dep["pnl"] < with_dep["end_value"] - 10_000     # deposits excluded


def test_mwr_is_close_to_twr_without_flows_and_unsolvable_windows_are_none():
    idx = _bars("2026-10-07", "2027-10-06")
    curve = pd.Series(np.linspace(100, 110, len(idx)), index=idx)
    s = ap.window_summary(ap.simulate_account(curve, idx[0], 10_000, 0.0), idx[0])
    assert s["mwr"] == pytest.approx(0.10, abs=0.01)
    short = ap.window_summary(ap.simulate_account(_flat(idx[:10]), idx[0], 1_000, 0.0), idx[0])
    assert short["mwr"] is None


def test_bad_inputs_give_empty_not_errors():
    idx = _bars()
    assert len(ap.simulate_account(_flat(idx), idx[0], 0.0, 100)) == 0
    assert ap.window_summary(ap.simulate_account(_flat(idx[:0]), idx[0], 100, 0), idx[0]) is None


def test_saved_deposit_defaults(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps({"_comment": "x", "U1": {"monthly": 500, "day": 15},
                             "U2": {"monthly": -5, "day": 40}, "U3": "junk"}))
    got = ap.load_deposit_defaults(p)
    assert got == {"U1": {"monthly": 500.0, "day": 15, "since": None},
                   "U2": {"monthly": 0.0, "day": 28, "since": None}}
    assert ap.load_deposit_defaults(tmp_path / "missing.json") == {}


def _report(tmp, acct, as_of, executed_utc, net_liq, mode="execute"):
    d = tmp / "accounts" / acct
    rep = d / "executed_book_live.json"
    payload = eb.build_payload(as_of=as_of, profile="Balanced", mode=mode,
                               account=acct, net_liq=net_liq, cash=0.0,
                               trades=[], positions=[], account_mode="live")
    payload["generated_at_utc"] = executed_utc
    d.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(payload))
    eb.archive_report(payload, rep, None)
    return rep


def test_record_info_needs_an_executed_run(tmp_path):
    rep = _report(tmp_path, "U1", "2026-10-09", "2026-10-12T19:30:00+00:00", 25_000.0)
    info = ap.record_info(rep)
    assert info["account"] == "U1" and info["first_net_liq"] == 25_000.0
    assert info["start"] == pd.Timestamp("2026-10-12") and info["n_runs"] == 1


def test_a_preview_only_account_has_no_record(tmp_path):
    rep = _report(tmp_path, "U2", "2026-10-09", "2026-10-12T19:30:00+00:00", 9_000.0,
                  mode="dry-run")
    assert ap.record_info(rep) is None            # nothing executed -> no radio option
    assert ap.record_info(tmp_path / "accounts" / "U9" / "executed_book_live.json") is None


def test_since_holds_deposits_back_until_the_effective_month():
    idx = pd.bdate_range("2026-08-03", "2027-01-29")
    bars = ap.deposit_bars(idx, idx[0], day=22, since="2026-10")
    assert [b.strftime("%Y-%m-%d") for b in bars] == [
        "2026-10-22", "2026-11-23", "2026-12-22", "2027-01-22"]   # Nov 22 = Sunday
    assert ap.deposit_bars(idx, idx[0], day=22)[0].month == 9     # default unchanged


def test_since_before_first_month_changes_nothing_and_saved_defaults_parse(tmp_path):
    idx = pd.bdate_range("2026-10-05", "2026-12-31")
    assert ap.deposit_bars(idx, idx[0], 22, since="2026-10")[0].month == 11
    p = tmp_path / "d.json"
    p.write_text('{"U1": {"monthly": 200, "day": 22, "from": "2026-10"}, "U2": {"monthly": 5}}')
    got = ap.load_deposit_defaults(p)
    assert got["U1"]["since"] == pd.Timestamp("2026-10-01") and got["U2"]["since"] is None
