"""Strategy Logic V2 — the 2026-10 loss-review rules and the version plumbing.

What V2 changed (``app/strategy_version.py`` has the history):

* trend engines gain an optional ENTRY GATE (WGMI: BTC above its SMA50), a
  HOLD condition folded into the long signal (REMX: SMA20 > SMA100) and a
  TRAILING stop; the gold dual-MA sleeves take the 25/100 cross only while
  the 100-day SMA is rising and trade 10%/12% trailing stops;
* a stop/trail hit at the latest close publishes a CLOSE (tone exit) instead
  of the V1 "ENTER" that left the position quietly held;
* the allocator is **adds-only**: a held sleeve is never trimmed by the daily
  tilt and is added to only when its target rises by ≥ 8 pp;
* every surface can show V1, V2 or the Combined (V1 before the cut-over, V2
  from it) view — ``version_for_date`` / ``combine_runs`` /
  ``published_book_replay(only_version=<set>)`` carry that.

These tests pin each rule on synthetic data so a regression in any of them
is caught without the live data feeds.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "app"))

import backtest_gldm as bg          # noqa: E402
import backtest_ticker as bt        # noqa: E402
import gldm_core as gc              # noqa: E402
import overall_core as oc           # noqa: E402
import strategy_version as sv       # noqa: E402
import ticker_config as tcfg        # noqa: E402


# ── version plumbing ───────────────────────────────────────────────────────
def test_current_version_is_v2_and_history_is_consistent():
    assert sv.STRATEGY_VERSION == "v2"
    assert sv.VERSION_HISTORY[-1] == ("v2", sv.STRATEGY_VERSION_START)
    assert oc.STRATEGY_VERSION == sv.STRATEGY_VERSION
    assert bt.active_version() == "v2" and bt.active_version("V1") == "v1"
    assert "STRATEGY LOGIC V2" in sv.badge_html()
    assert sv.version_label("v2") == "Strategy Logic V2"


def test_version_for_date_boundaries():
    assert sv.version_for_date("2026-07-16") == "pre-v1"
    assert sv.version_for_date("2026-07-30") == "pre-v1"
    assert sv.version_for_date("2026-07-31") == "v1"
    day_before = (pd.Timestamp(sv.STRATEGY_VERSION_START) - pd.Timedelta(days=1))
    assert sv.version_for_date(day_before) == "v1"
    assert sv.version_for_date(sv.STRATEGY_VERSION_START) == "v2"
    assert sv.version_for_date(pd.Timestamp("2030-01-01")) == "v2"


def test_views_map_to_versions():
    assert sv.versions_for_view(sv.VIEW_V1) == ("v1",)
    assert sv.versions_for_view("v2") == ("v2",)
    assert set(sv.versions_for_view(sv.VIEW_COMBINED)) == {"v1", "v2"}
    assert sv.cutover_dates() == ["2026-07-31", sv.STRATEGY_VERSION_START]
    assert set(sv.VIEW_TO_VERSION) == set(sv.VIEW_OPTIONS)


# ── trend engine rules ─────────────────────────────────────────────────────
def _cfg(**kw):
    base = dict(strategy_mode="dual_ma", ma_fast=2, ma_slow=4, ma_window=4,
                fixed_stop=1.0, stop_by_asset={}, oos_start="2020-01-01",
                v2_hold_ma_fast=0, v2_hold_ma_slow=0, v2_gate_col="",
                v2_gate_ma=0, v2_trail_stop=0.0, v2_trail_by_asset={},
                is_trend=True, macd_fast=12, macd_slow=26, macd_signal=9,
                vol_win=10, vol_med_win=20, vol_k=1.0)
    base.update(kw)
    ns = SimpleNamespace(**base)
    ns.trail_for = lambda col, version=None: (
        ns.v2_trail_by_asset.get(col, ns.v2_trail_stop) if str(version or "v2") == "v2" else 0.0)
    return ns


def test_hold_condition_only_applies_under_v2():
    # a rising series with one down bar: the 2/4 cross stays long through it,
    # the 1/2 hold condition (close vs SMA2) drops out on it under V2 only
    px = np.array([10, 11, 12, 13, 14, 15, 14.5, 16, 17, 18, 19.0])
    cfg = _cfg(v2_hold_ma_fast=1, v2_hold_ma_slow=2)
    v1 = bt.trend_long_array(cfg, px, version="v1")
    v2 = bt.trend_long_array(cfg, px, version="v2")
    assert v1[-1] and v2[-1]
    assert v1.sum() > v2.sum()                       # V2 is strictly a subset
    assert not (v2 & ~v1).any()
    assert np.array_equal(bt.trend_long_array(_cfg(), px, "v2"),
                          bt.trend_long_array(_cfg(), px, "v1"))   # no rule → identical


def test_entry_gate_array_needs_v2_and_the_column():
    daily = pd.DataFrame({"px_close": np.linspace(10, 20, 30),
                          "btc_close": np.r_[np.linspace(100, 90, 15), np.linspace(90, 130, 15)]},
                         index=pd.bdate_range("2024-01-01", periods=30))
    cfg = _cfg(v2_gate_col="btc_close", v2_gate_ma=5)
    assert bt.trend_entry_gate_array(cfg, daily, version="v1") is None
    assert bt.trend_entry_gate_array(_cfg(), daily, version="v2") is None
    g = bt.trend_entry_gate_array(cfg, daily, version="v2")
    assert g is not None and g.dtype == bool and len(g) == 30
    assert not g[:6].any() and g[-5:].all()        # falling BTC blocks, rising admits
    assert bt.trend_entry_gate_now(cfg, daily, "v2") is True
    assert bt.trend_entry_gate_now(cfg, daily.iloc[:10], "v2") is False
    assert bt.trend_entry_gate_array(cfg, daily.drop(columns="btc_close"), "v2") is None


def _preds(px, gate=None, long=None):
    idx = pd.bdate_range("2024-01-01", periods=len(px))
    df = pd.DataFrame({"px_close": px, "target_date": idx}, index=idx)
    for ver in ("v1", "v2"):
        df[f"trend_long_{ver}"] = (np.ones(len(px), bool) if long is None else long)
    df["trend_long"] = df["trend_long_v2"]
    if gate is not None:
        df["trend_gate_v2"] = gate
    return df


def test_simulate_regime_gate_blocks_entry_but_never_forces_an_exit():
    px = np.linspace(100, 120, 12)
    gate = np.array([False] * 4 + [True] * 2 + [False] * 6)
    cfg = _cfg(v2_gate_col="x", v2_gate_ma=3)
    r2 = bt.simulate_regime(cfg, _preds(px, gate), None, "px_close", version="v2")
    r1 = bt.simulate_regime(cfg, _preds(px, gate), None, "px_close", version="v1")
    assert r1["pos"][0] == 1                          # V1 enters on the first bar
    pos2 = list(r2["pos"])                            # pos[k] is bar k+1 (sim starts at bar 1)
    assert pos2[:4] == [0, 0, 0, 0]                   # gate False at bars 0-3 → no entry
    assert pos2[4] == 1 and all(p == 1 for p in pos2[4:])   # gate True at bar 4 → in at bar 5, held on
    assert r2["in_pos_now"] and r2["version"] == "v2"


def test_trailing_stop_exits_labels_and_blocks_same_bar_reentry():
    # rally to 130 then a 15% slide: a 10% trail exits at the first close
    # ≤ 117, the signal still says long so it re-enters on a LATER bar only
    px = np.array([100, 110, 120, 130, 125, 118, 116, 115, 120, 125.0])
    cfg = _cfg(v2_trail_stop=0.10)
    r = bt.simulate_regime(cfg, _preds(px), None, "px_close", version="v2")
    log = r["trade_log"]
    assert log and log[0]["reason"] == "trail −10%"
    p = _preds(px)
    assert pd.Timestamp(log[0]["exit_date"]) == p.index[6]        # 116 ≤ 130 × 0.9
    assert r["pos"][5] == 0 and r["pos"][6] == 1       # flat on the stop bar (bar 6), back on bar 7
    assert r["trail_px"] == pytest.approx(125.0 * 0.9)
    v1 = bt.simulate_regime(cfg, _preds(px), None, "px_close", version="v1")
    assert not v1["trade_log"]                        # V1 has no trailing stop


def test_stopped_last_bar_only_when_the_stop_closed_the_final_bar():
    px = np.array([100, 110, 120, 130, 125, 118, 116.0])     # trail hits on the last bar
    cfg = _cfg(v2_trail_stop=0.10)
    r = bt.simulate_regime(cfg, _preds(px), None, "px_close", version="v2")
    assert r["last_exit_reason"] == "trail −10%"
    assert oc._stopped_last_bar(r) == "trail −10%"
    px2 = np.append(px, 120.0)                                  # re-entered since
    r2 = bt.simulate_regime(cfg, _preds(px2), None, "px_close", version="v2")
    assert r2["in_pos_now"] and oc._stopped_last_bar(r2) is None
    sig_off = np.array([True] * 6 + [False])                    # signal exit, not a stop
    r3 = bt.simulate_regime(_cfg(), _preds(np.linspace(100, 110, 8), long=np.r_[sig_off, False]),
                            None, "px_close", version="v2")
    assert oc._stopped_last_bar(r3) is None


def test_combine_runs_splices_at_the_cutover():
    px = np.linspace(100, 150, 20)
    idx = pd.bdate_range("2026-09-21", periods=20)
    p = _preds(px); p.index = idx; p["target_date"] = idx
    r1 = bt.simulate_regime(_cfg(), p, None, "px_close", version="v1")
    r2 = bt.simulate_regime(_cfg(v2_trail_stop=0.10), p, None, "px_close", version="v2")
    # make the two runs differ: V2 flat throughout, V1 long
    r2 = dict(r2, strat=np.ones_like(r2["strat"]), pos=np.zeros_like(r2["pos"]),
              in_pos_now=False, trade_log=[], trades=np.array([]))
    cut = "2026-10-05"
    c = bt.combine_runs(r1, r2, cut)
    dates = pd.DatetimeIndex(pd.Series(c["dates"]))
    pre, post = dates < pd.Timestamp(cut), dates >= pd.Timestamp(cut)
    assert np.array_equal(c["pos"][pre], np.asarray(r1["pos"])[pre])
    assert np.array_equal(c["pos"][post], np.asarray(r2["pos"])[post])
    ret = np.r_[0.0, np.diff(c["strat"]) / c["strat"][:-1]]
    assert np.allclose(ret[post], 0.0)                 # V2 flat → no return after cut
    assert (c["version_series"][pre] == "v1").all() and (c["version_series"][post] == "v2").all()
    assert c["in_pos_now"] is False and c["version"] == "combined"


def test_run_strategy_combined_dispatch(monkeypatch):
    px = np.linspace(100, 150, 20)
    p = _preds(px)
    monkeypatch.setattr(bt, "version_cutover", lambda: str(p.index[10].date()))
    c = bt.run_strategy(_cfg(), p, None, "px_close", version="combined")
    assert c["version"] == "combined" and len(c["dates"]) == len(p) - 1


# ── real configs ──────────────────────────────────────────────────────────
def test_configs_carry_the_v2_rules():
    wgmi, remx, grid = (tcfg.get_config(k) for k in ("WGMI", "REMX", "GRID"))
    assert wgmi.has_v2_rules and wgmi.v2_gate_col == "btc_close" and wgmi.v2_gate_ma == 50
    assert remx.has_v2_rules and (remx.v2_hold_ma_fast, remx.v2_hold_ma_slow) == (20, 100)
    assert not grid.has_v2_rules
    assert "SMA20 > SMA100" in remx.rules_label("v2") and "SMA20" not in remx.rules_label("v1")
    assert "BTC > its SMA50" in wgmi.rules_label("v2")
    assert grid.rules_label("v1") == grid.rules_label("v2")


def test_gold_rules_switch_with_the_version():
    assert gc.stop_for("UGL", "v1") == pytest.approx(0.03)
    assert gc.stop_for("UGL", "v2") == 1.0 and gc.stop_for("GLDM", "v2") == 1.0
    assert gc.trail_for("GLDM", "v2") == pytest.approx(0.10)
    assert gc.trail_for("UGL", "v2") == pytest.approx(0.12)
    assert gc.trail_for("UGL", "v1") == 0.0
    assert gc.gate_rising_bars("v2") == 20 and gc.gate_rising_bars("v1") == 0
    assert gc.stop_for("GDX", "v2") == gc.stop_for("GDX", "v1")      # miners unchanged
    assert "trailing stop" in gc.rules_label("UGL", "v2")
    assert "−3% fixed stop" in gc.rules_label("UGL", "v1")


def test_dual_ma_gate_and_trail_on_synthetic_gold():
    n = 160
    idx = pd.bdate_range("2023-01-01", periods=n)
    # 100-day SMA falls for the first ~110 bars (declining prices), rises after
    px = np.r_[np.linspace(120, 80, 110), np.linspace(80, 130, 50)]
    preds = pd.DataFrame({"gldm_close": px, "ugl_close": px * 1.0, "target_date": idx}, index=idx)
    g = bg.dual_ma_gate_array(preds, rising_bars=20)
    assert not g[60:100].any() and g[-5:].all()
    assert bg.dual_ma_gate_array(preds, rising_bars=0).all()
    r_v1 = bg.simulate_dual(preds, "gldm_close", 1.0, oos_start="2023-01-01")
    r_gate = bg.simulate_dual(preds, "gldm_close", 1.0, oos_start="2023-01-01", gate_bars=20)
    first_v1 = int(np.argmax(np.asarray(r_v1["pos"]) > 0))
    first_gate = int(np.argmax(np.asarray(r_gate["pos"]) > 0))
    assert first_gate > first_v1                       # the gate delays the entry
    px2 = np.r_[np.linspace(80, 130, 60), np.linspace(130, 110, 10), np.linspace(110, 140, 30)]
    idx2 = pd.bdate_range("2023-01-01", periods=len(px2))
    preds2 = pd.DataFrame({"gldm_close": px2, "target_date": idx2}, index=idx2)
    r_tr = bg.simulate_dual(preds2, "gldm_close", 1.0, oos_start="2023-01-01", trail_pct=0.10)
    reasons = {t["reason"] for t in r_tr["trade_log"]}
    assert "trail −10%" in reasons


# ── decisions: stop-day CLOSE and gate WATCH ───────────────────────────────
def test_net_decision_stop_day_is_a_close_and_gate_blocks_entry():
    cfg = SimpleNamespace(is_trend=True)
    stopped = oc._net_decision(cfg, None, in_pos=False, last_close=1.0, ma_val=None,
                               long_now=True, stopped="trail −10%")
    assert stopped["state"] == "EXIT" and stopped["tone"] == "exit" and stopped.get("stopped")
    assert "CLOSE" in stopped["label"]
    gated = oc._net_decision(cfg, None, in_pos=False, last_close=1.0, ma_val=None,
                             long_now=True, gate_ok=False)
    assert gated["state"] == "WATCH" and gated["tone"] == "watch"
    plain = oc._net_decision(cfg, None, in_pos=False, last_close=1.0, ma_val=None,
                             long_now=True, gate_ok=True)
    assert plain["state"] == "ENTRY" and plain["tone"] == "buy"
    held = oc._net_decision(cfg, None, in_pos=True, last_close=1.0, ma_val=None,
                            long_now=True, stopped="stop −5%")
    assert held["state"] == "HOLD"                     # an open position is never "stopped"


def test_gold_engine_decision_mirrors_the_convention():
    import gldm_engine as ge
    d = ge._trend_decision(True, False, gate_ok=False)
    assert d["state"] == "WATCH"
    d = ge._trend_decision(True, False, stopped="trail −12%")
    assert d["state"] == "EXIT" and d["tone"] == "exit"
    assert ge._trend_decision(True, False) ["state"] == "ENTRY"


def _res(key, in_pos=False, tone="flat", state=None, label="FLAT", version="v2"):
    dec = dict(state=(state or ("HOLD" if in_pos else "FLAT")), label=label,
               ico="", tone=tone)
    return dict(key=key, parent=key, name=key, kind="core", emoji="", kemoji="",
                accent="#000", cap=0.30, last_close=100.0, dchg=0.0, ma_val=None,
                sentiment=50.0, decision=dec, alert="NEUTRAL", bull_regime=True,
                mom=0.05, win_rate=60.0, metrics=dict(sharpe=1.0),
                pos=dict(in_pos=in_pos, upnl=1.0), version=version)


_W = {"AAA": 0.3, "BBB": 0.3, "CCC": 0.3}


def test_gate_does_not_fund_a_stopped_sleeve():
    stopped = _res("AAA", tone="exit", state="EXIT", label="CLOSE — TRAIL HIT")
    held = _res("BBB", in_pos=True, tone="hold")
    gate = oc.signal_gated_allocation([stopped, held], _W, adds_only=0)
    assert gate["target"].get("AAA", 0.0) == 0.0
    assert gate["target"]["BBB"] > 0
    assert {a["key"]: a["action"] for a in gate["actions"]}["AAA"] == "STAND ASIDE"


# ── adds-only allocator ────────────────────────────────────────────────────
def test_adds_only_pins_held_sleeves_and_lets_entries_through():
    held_a = _res("AAA", in_pos=True, tone="hold")
    held_b = _res("BBB", in_pos=True, tone="hold")
    entry = _res("CCC", tone="buy", state="ENTRY", label="ENTER")
    v1 = oc.signal_gated_allocation([held_a, held_b, entry], _W, adds_only=0)
    prev = {"AAA": 0.30, "BBB": 0.05}                 # BBB was small; AAA at cap
    v2 = oc.signal_gated_allocation([held_a, held_b, entry], _W, prev_weights=prev)
    assert v2["adds_only"] == pytest.approx(oc.ADDS_ONLY_BAND)
    assert v2["target"]["AAA"] == pytest.approx(0.30)       # never trimmed below prev
    # BBB's tilted target is far above 0.05 (≥ band) → it IS added to
    assert v2["target"]["BBB"] > 0.05 + oc.ADDS_ONLY_BAND - 1e-9
    assert v2["target"]["CCC"] > 0                           # fresh entry funded
    # within-band wobble on a held sleeve is ignored
    prev2 = dict(prev, BBB=v1["target"]["BBB"] - 0.02)
    v3 = oc.signal_gated_allocation([held_a, held_b, entry], _W, prev_weights=prev2)
    assert v3["target"]["BBB"] == pytest.approx(prev2["BBB"])
    assert "BBB" in v3["held_pinned"]
    # explicit V1 behaviour: adds_only=0 reproduces the pure tilt
    v4 = oc.signal_gated_allocation([held_a, held_b, entry], _W, prev_weights=prev, adds_only=0)
    assert v4["target"] == v1["target"]


def test_adds_only_band_is_a_version_property():
    assert oc.adds_only_band("v1") == 0.0
    assert oc.adds_only_band("v2") == pytest.approx(0.08)
    assert oc.adds_only_band() == pytest.approx(0.08)
    r = [_res("AAA", version="v1")]
    assert oc.signal_gated_allocation(r, _W, prev_weights={"AAA": 0.1})["adds_only"] == 0.0


def _universe(n_days=120, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n_days)
    out = []
    for key in ("AAA", "BBB", "CCC", "DDD"):
        bh = 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.012, n_days)); bh[0] = 100.0
        # mostly in the market with occasional flat stretches, so the funded
        # set changes and the caps bind (4 × 30% > 100%) — the tilt re-sizes
        pos = (rng.random(n_days) > 0.08).astype(float)
        strat = 100.0 * np.cumprod(1 + np.r_[0.0, np.diff(bh) / bh[:-1]] * pos)
        ret = pd.Series(np.diff(strat) / strat[:-1], index=idx[1:]).rename(key)
        out.append(dict(key=key, name=key, kind="core", parent=key, accent="#000",
                        emoji="•", dates=idx, ret=ret, version="v2",
                        pos_series=pd.Series(pos, index=idx), pos=dict(in_pos=True),
                        r=dict(bh=bh, dates=list(idx), trade_log=[])))
    return out


def test_replay_adds_only_never_trims_a_held_sleeve():
    res = _universe()
    bw = {"AAA": 0.3, "BBB": 0.3, "CCC": 0.3, "DDD": 0.3}
    v1 = oc.replay_gated_allocation(res, base_weights=bw, adds_only=0)
    v2 = oc.replay_gated_allocation(res, base_weights=bw, adds_only=0.08)
    W1, W2 = v1["weights"], v2["weights"]
    P = oc.position_matrix(res, W2.index)
    held = (P.shift(1) > 0) & (P > 0)                 # in the market yesterday AND today
    # the rule: a held sleeve is scaled down ONLY to fund an add (≥ band) or a
    # fresh entry that overflows the book — never by the tilt on its own.  So
    # on every day with no add/entry, no held weight falls; and every add on
    # a held sleeve is at least the band wide.
    dW = W2.diff().iloc[1:]
    any_up = (dW > 1e-9).any(axis=1)
    held_down = (dW.where(held.iloc[1:]) < -1e-9).any(axis=1)
    assert not (held_down & ~any_up).any()
    # (an add is ≥ the band BEFORE the overflow rescale; after a rescale a
    # big add can net to less, so the band is pinned on the live gate instead)
    assert (W1.diff().where(held).iloc[1:].fillna(0.0) < -1e-9).any().any()   # V1 tilt trims
    # and the V2 book is quieter overall: far fewer resize days
    resize_v1 = (W1.diff().where(held).abs() > 1e-9).any(axis=1).sum()
    resize_v2 = (W2.diff().where(held).abs() > 1e-9).any(axis=1).sum()
    assert resize_v2 < resize_v1
    assert not np.allclose(W1.to_numpy(), W2.to_numpy())
    assert v2["turnover"]["mean"] <= v1["turnover"]["mean"]


def test_replay_adds_only_from_switches_mid_history():
    res = _universe()
    bw = {"AAA": 0.3, "BBB": 0.3, "CCC": 0.3, "DDD": 0.3}
    idx = res[0]["ret"].index
    cut = idx[60]
    v1 = oc.replay_gated_allocation(res, base_weights=bw, adds_only=0)
    mix = oc.replay_gated_allocation(res, base_weights=bw, adds_only=0.08, adds_only_from=cut)
    pre = mix["weights"].index < cut
    assert np.allclose(mix["weights"].to_numpy()[pre], v1["weights"].to_numpy()[pre])
    P = oc.position_matrix(res, mix["weights"].index)
    held = (P.shift(1) > 0) & (P > 0)
    dW = mix["weights"].diff().loc[cut + pd.Timedelta(days=1):]
    any_up = (dW > 1e-9).any(axis=1)
    held_down = (dW.where(held.loc[dW.index]) < -1e-9).any(axis=1)
    assert not (held_down & ~any_up).any()
    assert not np.allclose(mix["weights"].to_numpy()[~pre], v1["weights"].to_numpy()[~pre])


def test_walkforward_version_selects_the_adds_only_rule():
    res = _universe(n_days=320)
    wf_v1 = oc.walkforward_gated_replay(res, version="v1", n_samples=200, min_hist=40)
    wf_v2 = oc.walkforward_gated_replay(res, version="v2", n_samples=200, min_hist=40)
    assert wf_v1["adds_only"] == 0.0 and wf_v2["adds_only"] == pytest.approx(0.08)
    assert wf_v1["version"] == "v1" and wf_v2["version"] == "v2"
    assert wf_v2["turnover"]["mean"] <= wf_v1["turnover"]["mean"]


# ── published record: version views ──────────────────────────────────────
def _book(as_of, w, cash, ver):
    return dict(as_of=as_of, weights=w, cash_weight=cash, strategy_version=ver)


def test_published_replay_admits_a_set_of_versions():
    idx = pd.bdate_range("2026-07-01", periods=12)
    rets = pd.DataFrame({"AAA": 0.01}, index=idx)
    books = [_book("2026-07-02", {"AAA": 1.0}, 0.0, "v1"),
             _book("2026-07-08", {"AAA": 0.5}, 0.5, "v2")]
    both = oc.published_book_replay(rets, books, sata_daily=0.0, only_version={"v1", "v2"})
    v1 = oc.published_book_replay(rets, books, sata_daily=0.0, only_version="v1")
    v2 = oc.published_book_replay(rets, books, sata_daily=0.0, only_version=("v2",))
    assert len(both["books"]) == 2 and [s["version"] for s in both["version_spans"]] == ["v1", "v2"]
    assert len(v1["books"]) == 1 and v1["books"][0]["version"] == "v1"
    assert len(v2["books"]) == 1 and v2["books"][0]["version"] == "v2"


# ── execution alert: unexecuted CLOSE ──────────────────────────────────────
def test_unexecuted_closes_flags_held_or_unrun():
    prev = dict(as_of="2026-10-07", weights={"GRID": 0.3},
                actions=[dict(key="WGMI", action="CLOSE"), dict(key="GRID", action="HOLD")])
    executed = dict(as_of="2026-10-07", positions=[dict(key="WGMI", shares=10),
                                                   dict(key="GRID", shares=5)])
    out = oc.unexecuted_closes(prev, executed)
    assert [a["key"] for a in out] == ["WGMI"] and out[0]["account"] == "IBKR"
    done = dict(as_of="2026-10-07", positions=[dict(key="GRID", shares=5)])
    assert oc.unexecuted_closes(prev, done) == []
    stale = dict(as_of="2026-10-01", positions=[])
    out = oc.unexecuted_closes(prev, stale)
    assert out and "no IBKR execution" in out[0]["reason"]
    assert oc.unexecuted_closes(None, executed) == []
    assert oc.unexecuted_closes(dict(prev, actions=[]), executed) == []
    c2 = dict(book_as_of="2026-10-07", positions=[dict(symbol="WGMI", shares=3)])
    out = oc.unexecuted_closes(prev, done, c2)
    assert [(a["key"], a["account"]) for a in out] == [("WGMI", "Collective2")]


# ── health builder paths ──────────────────────────────────────────────────
def test_health_view_paths():
    sys.path.insert(0, str(REPO / "scripts"))
    import build_strategy_health as bsh
    out, hist = Path("x/strategy_health.json"), Path("x/health_history.csv")
    assert bsh.view_paths(out, hist, "combined") == (out, hist)
    assert bsh.view_paths(out, hist, "v1") == (Path("x/strategy_health_v1.json"),
                                               Path("x/health_history_v1.csv"))
    assert bsh.VIEWS == ["combined", "v1", "v2"]


# ── live-price mirrors of the V2 rules ────────────────────────────────────
def test_live_exit_flags_a_breached_trailing_stop_and_entry_respects_the_gate():
    cfg = tcfg.get_config("GRID")               # any trend config; the trail is on the result
    held = dict(key="UGL", parent="GLDM", mode="dual_ma", cfg=None, close_hist=None,
                decision=dict(tone="hold"), pos=dict(in_pos=True, trail_px=45.0),
                last_close=46.0, bar_close=46.0, version="v2")
    assert oc.live_exit_keys([held], {"GLDM": {"price": 44.5}}) == {"UGL"}
    assert oc.live_exit_keys([held], {"GLDM": {"price": 45.5}}) == set()
    flat = dict(key="WGMI", parent="WGMI", mode="ma_vol", cfg=cfg,
                close_hist=[10.0, 10.0, 10.0], decision=dict(tone="flat"),
                pos=dict(in_pos=False), last_close=10.0, bar_close=10.0,
                version="v2", gate_ok=False)
    assert oc.live_entry_keys([flat], {"WGMI": {"price": 50.0}}) == set()


# ── V2 post-stop cooldown (SOXX) ──────────────────────────────────────────
def test_stop_cooldown_keeps_the_sleeve_flat_then_readmits():
    # rally, a −5% stop hit, then the trend keeps saying long: V1 re-enters the
    # next bar, V2 (cooldown 3) stays flat for three bars first
    cfg = _cfg(fixed_stop=0.05)
    cfg.v2_stop_cooldown = 3
    cfg.stop_cooldown_for = lambda version=None: (3 if str(version or "v2") == "v2" else 0)
    # entry at 100 on bar 1; bar 4 closes at 94 (−6%) → the −5% stop fires
    px = np.array([100, 104, 108, 112, 94, 100, 104, 108, 112, 116, 118.0])
    p = _preds(px)
    v1 = bt.simulate_regime(cfg, p, None, "px_close", stop_pct=0.05, version="v1")
    v2 = bt.simulate_regime(cfg, p, None, "px_close", stop_pct=0.05, version="v2")
    assert v1["trade_log"][0]["reason"] == "stop −5%"
    # sim pos[k] is bar k+1: stop on bar 4 → pos[3]=0; V1 back in on bar 5 (pos[4]=1);
    # V2 stays flat through bar 7 (pos[4..6]=0) and re-enters on bar 8 (pos[7]=1)
    assert list(v1["pos"][3:5]) == [0, 1]
    assert list(v2["pos"][3:8]) == [0, 0, 0, 0, 1]
    assert v2["cooldown_left"] == 0 and v2["in_pos_now"]
    # cut the series right after the stop: cooldown still running → reported
    v2s = bt.simulate_regime(cfg, _preds(px[:6]), None, "px_close", stop_pct=0.05, version="v2")
    assert not v2s["in_pos_now"] and v2s["cooldown_left"] == 2
    # the live decision turns that into a WATCH, never an ENTER
    dec = oc._net_decision(SimpleNamespace(is_trend=True), None, in_pos=False, last_close=1.0,
                           ma_val=None, long_now=True, cooldown_left=2)
    assert dec["state"] == "WATCH" and "COOLDOWN" in dec["label"]
    flat = dict(key="SOXX", parent="SOXX", mode="dual_ma", cfg=tcfg.get_config("SOXX"),
                close_hist=[1.0, 1.0, 1.0], decision=dict(tone="watch"), pos=dict(in_pos=False),
                last_close=1.0, bar_close=1.0, version="v2", cooldown_left=2)
    assert oc.live_entry_keys([flat], {"SOXX": {"price": 50.0}}) == set()


def test_soxx_config_carries_the_cooldown_and_soxl_is_untouched():
    soxx = tcfg.get_config("SOXX")
    assert soxx.has_v2_rules and soxx.stop_cooldown_for("v2") == 5 and soxx.stop_cooldown_for("v1") == 0
    assert "5-bar cooldown" in soxx.rules_label("v2") and "cooldown" not in soxx.rules_label("v1")
    assert soxx.stop_for("soxl_close") >= 0.999            # SOXL is stop-less → cooldown never fires
