"""Unit tests for the 🩺 strategy-health compute layer (app/health_core.py).

Covers the pure decay-monitor helpers with synthetic data — no network, no
Streamlit: drawdown/underwater stats, the rolling edge-vs-B&H window, the
bootstrap expectancy percentile, the published-book realized-return
reconstruction, breach persistence, and the snapshot's status logic.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import health_core as hc  # noqa: E402


def _days(n, start="2024-01-01"):
    return pd.date_range(start, periods=n, freq="D")


# ── drawdown / underwater ─────────────────────────────────────────────────
def test_drawdown_series_matches_hand_computation():
    eq = pd.Series([1.0, 1.2, 0.9, 1.2, 1.5], index=_days(5))
    dd = hc.drawdown_series(eq)
    assert np.isclose(dd.iloc[2], 0.9 / 1.2 - 1)
    assert dd.iloc[0] == 0 and dd.iloc[-1] == 0


def test_underwater_stats_current_and_max_spells():
    # highs on days 0-1, underwater days 2-4, recovery day 5, underwater 6-7
    dd = pd.Series([0, 0, -0.1, -0.2, -0.1, 0, -0.05, -0.02], index=_days(8))
    st = hc.underwater_stats(dd)
    assert st["max_days"] == 3          # day1 peak → day4 trough-side bar
    assert st["cur_days"] == 2          # day5 peak → day7 (still under)


def test_underwater_stats_at_high_is_zero():
    dd = pd.Series([0, -0.1, 0], index=_days(3))
    assert hc.underwater_stats(dd)["cur_days"] == 0


# ── rolling edge vs Buy & Hold ────────────────────────────────────────────
def test_rolling_edge_sign_and_magnitude():
    idx = _days(300)
    strat = pd.Series(0.002, index=idx)          # steady winner
    bh = pd.Series(0.001, index=idx)
    edge = hc.rolling_edge(strat, bh, window=126)
    assert len(edge) == 300 - 126 + 1
    expected = (1.002 ** 126 - 1.001 ** 126) * 100
    assert np.isclose(edge.iloc[-1], expected, rtol=1e-9)
    assert (edge > 0).all()


def test_rolling_edge_short_history_is_empty():
    idx = _days(50)
    edge = hc.rolling_edge(pd.Series(0.001, index=idx),
                           pd.Series(0.002, index=idx), window=126)
    assert edge.empty


# ── bootstrap percentile ──────────────────────────────────────────────────
def test_bootstrap_pctl_extremes_and_determinism():
    pool = list(np.linspace(-0.05, 0.05, 60))
    lo = hc.bootstrap_pctl(-1.0, pool, 20)
    hi = hc.bootstrap_pctl(+1.0, pool, 20)
    mid = hc.bootstrap_pctl(0.0, pool, 20)
    assert lo == 0.0 and hi == 100.0 and 30 < mid < 70
    assert mid == hc.bootstrap_pctl(0.0, pool, 20)   # fixed seed → reproducible


def test_bootstrap_pctl_insufficient_pool_returns_none():
    assert hc.bootstrap_pctl(0.0, [0.01] * 10, 20) is None


# ── published-book realized returns ───────────────────────────────────────
def test_realized_book_returns_uses_previous_book_and_sata():
    idx = pd.date_range("2026-07-13", periods=6, freq="D")   # Mon..Sat
    rets = pd.DataFrame({"AAA": [0.01] * 6, "BBB": [0.02] * 6}, index=idx)
    books = [dict(as_of="2026-07-14", weights={"AAA": 0.5}, cash_weight=0.5),
             dict(as_of="2026-07-16", weights={"BBB": 1.0}, cash_weight=0.0)]
    sata = 0.001
    live = hc.realized_book_returns(books, rets, sata_daily=sata)
    # first driven day is the bar AFTER the first book's as_of
    assert str(live.index[0].date()) == "2026-07-15"
    assert np.isclose(live.loc["2026-07-15"], 0.5 * 0.01 + 0.5 * sata)
    assert np.isclose(live.loc["2026-07-16"], 0.5 * 0.01 + 0.5 * sata)
    # the 07-16 book drives the NEXT day (decided-at-previous-close), and
    # Saturday 07-18 pays no SATA (cash weight is 0 anyway; weekday rule)
    assert np.isclose(live.loc["2026-07-17"], 0.02)
    assert np.isclose(live.loc["2026-07-18"], 0.02)


def test_realized_book_returns_empty_without_books():
    rets = pd.DataFrame({"AAA": [0.01]}, index=_days(1))
    assert hc.realized_book_returns([], rets, 0.001).empty


# ── persistence & status ──────────────────────────────────────────────────
def test_merge_first_breach_carries_and_clears():
    prev = {"SOXX:edge": "2026-07-01"}
    assert hc.merge_first_breach(prev, "SOXX:edge", True, "2026-07-30") == "2026-07-01"
    assert hc.merge_first_breach(prev, "SOXX:edge", False, "2026-07-30") is None
    assert hc.merge_first_breach(prev, "REMX:edge", True, "2026-07-30") == "2026-07-30"


def test_worst_status_ordering():
    assert hc.worst_status("green", "warming") == "warming"
    assert hc.worst_status("green", "yellow", "warming") == "yellow"
    assert hc.worst_status("yellow", "red") == "red"
    assert hc.worst_status() == "green"


def test_dd_health_red_on_deeper_than_reference():
    idx = _days(400)
    ret = pd.Series(0.001, index=idx)
    ret.iloc[100:110] = -0.02        # reference drawdown ≈ −18%
    ret.iloc[380:400] = -0.02        # live drawdown ≈ −33% → breach
    anchor = idx[300]
    out = hc.dd_health(ret, anchor, {}, "X", "2026-07-30")
    assert out["status"] == "red"
    assert out["value"] < out["ref_mdd"]


def test_dd_health_warming_without_reference_history():
    idx = _days(300)
    ret = pd.Series(0.001, index=idx)
    out = hc.dd_health(ret, idx[100], {}, "X", "2026-07-30")   # 101 ref bars < 250
    assert out["status"] == "warming"


def test_edge_health_yellow_then_red_after_persistence():
    idx = _days(300)
    strat = pd.Series(0.000, index=idx)
    bh = pd.Series(0.002, index=idx)                 # strategy loses to B&H
    fresh, _ = hc.edge_health(strat, bh, {}, "X", "2026-07-30")
    assert fresh["status"] == "yellow"               # first sighting
    aged, _ = hc.edge_health(strat, bh, {"X:edge": "2026-01-01"}, "X", "2026-07-30")
    assert aged["status"] == "red"                   # persisted > edge_red_days
    assert aged["first_breach_date"] == "2026-01-01"


def test_expectancy_health_warming_then_flags_cold_streak():
    warm = hc.expectancy_health([0.01] * 30, {}, "X", "2026-07-30")
    assert warm["status"] == "warming" and "10 more" in warm["need"]
    trades = [0.02] * 40 + [-0.05] * 20              # last 20 far below the pool
    cold = hc.expectancy_health(trades, {}, "X", "2026-07-30")
    assert cold["status"] == "yellow" and cold["pctl"] == 0.0
    aged = hc.expectancy_health(trades, {"X:expectancy": "2026-07-01"},
                                "X", "2026-07-30")
    assert aged["status"] == "red"


def test_tracking_health_gap_statuses():
    idx = _days(100)
    rep = pd.Series(0.001, index=idx)
    ok = hc.tracking_health(rep.copy(), rep, {}, "2026-07-30")
    assert ok["status"] == "green" and abs(ok["cum_gap"]) < 1e-9
    bad = hc.tracking_health(rep - 0.001, rep, {}, "2026-07-30")
    assert bad["status"] == "red"                    # 63-day gap ≈ −6.3%
    assert bad["first_breach_date"] == "2026-07-30"


def test_tracking_health_warming_until_min_days():
    idx = _days(10)                                  # < track_min_days
    rep = pd.Series(0.001, index=idx)
    early = hc.tracking_health(rep - 0.01, rep, {}, "2026-07-30")
    assert early["status"] == "warming"              # huge gap, but unarmed
    assert early["first_breach_date"] is None


def test_edge_health_deadband_ignores_hairline_negatives():
    idx = _days(300)
    strat = pd.Series(0.00100, index=idx)
    bh = pd.Series(0.00101, index=idx)               # edge ≈ −0.02pp: noise
    out, _ = hc.edge_health(strat, bh, {}, "X", "2026-07-30")
    assert out["status"] == "green" and out["first_breach_date"] is None


# ── snapshot round-trip on a synthetic universe ───────────────────────────
def _fake_result(key, n=700, drift=0.001, seed=1):
    rng = np.random.default_rng(seed)
    idx = _days(n, "2023-01-01")
    ret = pd.Series(rng.normal(drift, 0.01, n), index=idx)
    bh = (1 + pd.Series(rng.normal(0.0005, 0.01, n), index=idx)).cumprod()
    trades = [dict(ret=float(r), exit_date=str(d.date()))
              for d, r in zip(idx[::10], rng.normal(0.01, 0.03, n // 10))]
    return dict(key=key, name=key, parent=key, kind="core", emoji="🧪",
                as_of=idx[-1], n_trades=len(trades), ret=ret,
                dates=pd.Series(idx), r=dict(bh=bh.to_numpy(), trade_log=trades))


def test_build_snapshot_end_to_end_shape_and_history():
    results = [_fake_result("AAA", seed=1), _fake_result("BBB", seed=2)]
    idx = results[0]["ret"].index
    replays = {"Balanced": dict(ret=pd.Series(0.001, index=idx),
                                metrics=dict(total_ret=1.0, cagr=0.5,
                                             mdd=-0.1, sharpe=2.0, vol=0.1))}
    books = [dict(as_of=str(idx[-30].date()), weights={"AAA": 0.6},
                  cash_weight=0.4)]
    snap = hc.build_snapshot(results, replays, books, prev_snapshot=None,
                             sata_daily=0.00052)
    assert snap["schema"] == hc.SCHEMA
    assert {s["key"] for s in snap["sleeves"]} == {"AAA", "BBB"}
    assert snap["live_anchor"] == str(idx[-30].date())
    assert snap["portfolio"]["tracking"]["n_days"] == 29
    assert snap["verdict"]["status"] in ("green", "yellow", "red")
    hist = hc.history_rows(snap)
    assert set(hist["scope"]) == {"sleeve", "profile", "tracking"}
    # persistence round-trip: breach dates survive via prev_breach_map
    snap2 = hc.build_snapshot(results, replays, books, prev_snapshot=snap,
                              sata_daily=0.00052)
    for s2 in snap2["sleeves"]:
        s1 = next(s for s in snap["sleeves"] if s["key"] == s2["key"])
        for mon in ("dd", "edge", "expectancy"):
            if s1[mon]["first_breach_date"]:
                assert s2[mon]["first_breach_date"] == s1[mon]["first_breach_date"]
