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


def _res(key, in_pos=False, tone="flat", state=None, label="FLAT", version="v2",
         parent=None):
    dec = dict(state=(state or ("HOLD" if in_pos else "FLAT")), label=label,
               ico="", tone=tone)
    return dict(key=key, parent=parent or key, name=key, kind="core", emoji="", kemoji="",
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


def _universe(n_days=120, seed=3, parents=None):
    parents = parents or {}
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
        out.append(dict(key=key, name=key, kind="core", parent=parents.get(key, key), accent="#000",
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
    assert wf_v1["cluster_cap"] == 0.0 and wf_v2["cluster_cap"] == pytest.approx(oc.CLUSTER_CAP)
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


# ── parent-cluster cap ────────────────────────────────────────────────────
_ENERGY_CAPS = {"XLE": 0.30, "OIH": 0.18, "ERX": 0.10, "GRID": 0.30}
_ENERGY_CL = {"XLE": "XLE", "OIH": "XLE", "ERX": "XLE", "GRID": "GRID"}


def test_cluster_cap_is_a_version_property():
    assert oc.cluster_cap_for("v1") == 0.0
    assert oc.cluster_cap_for("v2") == pytest.approx(0.30)
    assert oc.cluster_cap_for() == pytest.approx(oc.CLUSTER_CAP)
    assert oc.cluster_map([_res("OIH", parent="XLE"), _res("GRID")]) == {"OIH": "XLE", "GRID": "GRID"}


def test_apply_cluster_cap_binds_the_cluster_and_respreads_the_freed_weight():
    raw = {"XLE": 0.3, "OIH": 0.18, "ERX": 0.1, "GRID": 0.05}
    plain = oc._apply_cluster_cap(raw, _ENERGY_CAPS, _ENERGY_CL, 0.0)
    assert oc.cluster_weights(plain, _ENERGY_CL)["XLE"] > 0.30 + 1e-9
    capped = oc._apply_cluster_cap(raw, _ENERGY_CAPS, _ENERGY_CL, 0.30)
    cw = oc.cluster_weights(capped, _ENERGY_CL)
    assert cw["XLE"] == pytest.approx(0.30)
    assert capped["GRID"] == pytest.approx(0.30)          # freed weight re-spread up to GRID's cap
    assert sum(capped.values()) == pytest.approx(0.60)    # what nobody can absorb is left for SATA
    # the cluster keeps its pro-rata shape
    assert capped["XLE"] / capped["OIH"] == pytest.approx(plain["XLE"] / plain["OIH"])
    for k, v in capped.items():
        assert v <= _ENERGY_CAPS[k] + 1e-9


def test_bind_cluster_cap_scales_a_pinned_cluster_without_respreading():
    t = {"XLE": 0.3, "OIH": 0.18, "ERX": 0.1, "GRID": 0.2}
    out = oc._bind_cluster_cap(t, _ENERGY_CL, 0.30)
    assert oc.cluster_weights(out, _ENERGY_CL)["XLE"] == pytest.approx(0.30)
    assert out["XLE"] / out["ERX"] == pytest.approx(3.0)
    assert out["GRID"] == pytest.approx(0.20)             # untouched, no re-spread
    assert oc._bind_cluster_cap(t, _ENERGY_CL, 0.0) == t


def test_signal_gated_allocation_binds_parent_clusters():
    sleeves = [_res("XLE", in_pos=True, tone="hold"),
               _res("OIH", in_pos=True, tone="hold", parent="XLE"),
               _res("ERX", in_pos=True, tone="hold", parent="XLE"),
               _res("GRID", in_pos=True, tone="hold")]
    w = {"XLE": 0.3, "OIH": 0.3, "ERX": 0.3, "GRID": 0.1}
    free = oc.signal_gated_allocation(sleeves, w, caps=_ENERGY_CAPS, adds_only=0, cluster_cap=0)
    assert free["cluster_cap"] == 0.0 and free["clusters"]["XLE"] > 0.30 + 1e-9
    v2 = oc.signal_gated_allocation(sleeves, w, caps=_ENERGY_CAPS)
    assert v2["cluster_cap"] == pytest.approx(oc.CLUSTER_CAP)
    assert v2["clusters"]["XLE"] == pytest.approx(0.30)
    assert v2["clusters"]["GRID"] == pytest.approx(0.30)
    assert v2["tilt_target"] == v2["target"]              # no prev book → nothing pinned
    # a book the adds-only pin holds above the cap is scaled down to it (freed → SATA)
    prev = {"XLE": 0.30, "OIH": 0.18, "ERX": 0.10, "GRID": 0.30}
    v3 = oc.signal_gated_allocation(sleeves, w, caps=_ENERGY_CAPS, prev_weights=prev)
    assert v3["clusters"]["XLE"] == pytest.approx(0.30)
    assert v3["target"]["XLE"] / v3["target"]["OIH"] == pytest.approx(0.30 / 0.18)
    assert v3["target"]["GRID"] == pytest.approx(0.30)
    assert v3["sata"] == pytest.approx(0.40)
    # V1-stamped results carry no cluster cap
    v1 = oc.signal_gated_allocation([dict(r, version="v1") for r in sleeves], w, caps=_ENERGY_CAPS)
    assert v1["cluster_cap"] == 0.0 and v1["clusters"]["XLE"] > 0.30 + 1e-9


def test_historical_allocation_takes_the_cluster_cap():
    snap = dict(rows=[dict(key="XLE", parent="XLE", in_pos=True),
                      dict(key="OIH", parent="XLE", in_pos=True),
                      dict(key="ERX", parent="XLE", in_pos=True),
                      dict(key="GRID", parent="GRID", in_pos=False)])
    w = {"XLE": 0.3, "OIH": 0.3, "ERX": 0.3, "GRID": 0.1}
    free = oc.historical_allocation(snap, w, caps=_ENERGY_CAPS)
    assert sum(free["book"].values()) == pytest.approx(0.58)
    capped = oc.historical_allocation(snap, w, caps=_ENERGY_CAPS, cluster_cap=0.30)
    assert sum(capped["book"].values()) == pytest.approx(0.30)
    assert capped["sata"] == pytest.approx(0.70)


def test_replay_cluster_cap_binds_every_day_and_switches_on_from_a_date():
    res = _universe(parents={"BBB": "AAA", "CCC": "AAA"})
    bw = {k: 0.25 for k in ("AAA", "BBB", "CCC", "DDD")}
    free = oc.replay_gated_allocation(res, base_weights=bw, adds_only=0.08, cluster_cap=0)
    capped = oc.replay_gated_allocation(res, base_weights=bw, adds_only=0.08, cluster_cap=0.30)
    cl_free = free["weights"][["AAA", "BBB", "CCC"]].sum(axis=1)
    cl_cap = capped["weights"][["AAA", "BBB", "CCC"]].sum(axis=1)
    assert (cl_free > 0.30 + 1e-9).any()
    assert (cl_cap <= 0.30 + 1e-9).all()
    assert (capped["weights"]["DDD"] <= 0.30 + 1e-9).all()
    assert (capped["weights"].sum(axis=1) + capped["sata"]).round(9).eq(1.0).all()
    assert capped["cluster_cap"] == pytest.approx(0.30) and free["cluster_cap"] == 0.0
    cut = res[0]["dates"][60]
    mix = oc.replay_gated_allocation(res, base_weights=bw, adds_only=0.08,
                                     cluster_cap=0.30, cluster_cap_from=cut)
    cl_mix = mix["weights"][["AAA", "BBB", "CCC"]].sum(axis=1)
    pd.testing.assert_series_equal(cl_mix.loc[:cut - pd.Timedelta(days=1)],
                                   cl_free.loc[:cut - pd.Timedelta(days=1)])
    assert (cl_mix.loc[cut:] <= 0.30 + 1e-9).all()


# ── universe changes by version (OIH → XOP) ───────────────────────────────
def test_universe_membership_follows_the_version():
    assert oc.sleeve_in_universe("OIH", "v1") and not oc.sleeve_in_universe("XOP", "v1")
    assert oc.sleeve_in_universe("XOP", "v2") and not oc.sleeve_in_universe("OIH", "v2")
    assert oc.sleeve_in_universe("OIH", "combined") and oc.sleeve_in_universe("XOP", "combined")
    assert oc.sleeve_in_universe("XLE", "v1") and oc.sleeve_in_universe("XLE", "v2")
    v1, v2 = oc.universe_keys("v1"), oc.universe_keys("v2")
    assert "OIH" in v1 and "XOP" not in v1 and "XOP" in v2 and "OIH" not in v2
    assert len(v1) == len(v2) == 18
    assert oc.CAP_BY_KEY["XOP"] == oc.CAP_BY_KIND["beta"]
    assert "XOP" in dict(tcfg.get_config("XLE").traded_assets)
    assert "OIH" in dict(tcfg.get_config("XLE").traded_assets)   # the XLE app keeps the tab


def _sleeve(key, parent, idx, rng, in_pos=True):
    n = len(idx)
    px = 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.01, n)); px[0] = 100.0
    pos = np.ones(n)
    strat = 100.0 * np.cumprod(1 + np.r_[0.0, np.diff(px) / px[:-1]] * pos)
    ret = pd.Series(np.diff(strat) / strat[:-1], index=idx[1:]).rename(key)
    log = [dict(entry_date=idx[5], exit_date=idx[40], ret=0.05),
           dict(entry_date=idx[50], exit_date=idx[-5], ret=-0.02)]
    return dict(key=key, name=key, kind="beta", parent=parent, accent="#000", emoji="•",
                dates=idx, ret=ret, strat=strat, version="v2",
                pos_series=pd.Series(pos, index=idx), pos=dict(in_pos=in_pos, upnl=1.0),
                decision=dict(state="HOLD", label="LONG", ico="", tone="hold"),
                metrics=dict(sharpe=1.0), win_rate=50.0, n_trades=2,
                r=dict(bh=px, dates=list(idx), strat=strat, pos=pos, trade_log=log,
                       trades=np.array([0.05, -0.02]), in_pos_now=in_pos))


def test_combine_results_splices_a_universe_change_at_the_cutover():
    rng = np.random.default_rng(5)
    cut = pd.Timestamp(sv.STRATEGY_VERSION_START)
    idx = pd.bdate_range(cut - pd.Timedelta(days=120), periods=120)
    assert idx[0] < cut < idx[-1]
    xle1, xle2 = _sleeve("XLE", "XLE", idx, rng), _sleeve("XLE", "XLE", idx, rng)
    oih = _sleeve("OIH", "XLE", idx, rng)                 # V1 only
    xop = _sleeve("XOP", "XLE", idx, rng)                 # V2 only
    grid1, grid2 = _sleeve("GRID", "GRID", idx, rng), _sleeve("GRID", "GRID", idx, rng)
    out = oc.combine_results([xle1, oih, grid1], [xle2, xop, grid2])
    by = {r["key"]: r for r in out}
    assert list(by) == ["XLE", "XOP", "OIH", "GRID"]       # retired OIH sits after its parent group
    assert all(r["version"] == "combined" for r in out)
    # OIH: V1 stream before the cut-over, flat (no position, no return) from it
    assert (by["OIH"]["pos_series"].loc[:cut - pd.Timedelta(days=1)] == 1).all()
    assert (by["OIH"]["pos_series"].loc[cut:] == 0).all()
    assert (by["OIH"]["ret"].loc[cut:] == 0).all()
    assert by["OIH"]["ret"].loc[:cut - pd.Timedelta(days=1)].abs().sum() > 0
    assert by["OIH"]["pos"]["in_pos"] is False and by["OIH"]["decision"]["tone"] == "flat"
    assert all(pd.Timestamp(t["exit_date"]) < cut for t in by["OIH"]["r"]["trade_log"])
    # XOP: flat before the cut-over, V2 stream from it, live state V2's
    assert (by["XOP"]["pos_series"].loc[:cut - pd.Timedelta(days=1)] == 0).all()
    assert (by["XOP"]["pos_series"].loc[cut:] == 1).all()
    assert (by["XOP"]["ret"].loc[:cut - pd.Timedelta(days=1)] == 0).all()
    assert by["XOP"]["ret"].loc[cut:].abs().sum() > 0
    assert by["XOP"]["pos"]["in_pos"] is True
    # the replay sees both: OIH funded only before the cut-over, XOP only from it
    bw = {"XLE": 0.3, "OIH": 0.3, "XOP": 0.3, "GRID": 0.3}
    rep = oc.replay_gated_allocation(out, base_weights=bw, adds_only=0, cluster_cap=0)
    W = rep["weights"]
    assert (W["OIH"].loc[cut:] == 0).all() and W["OIH"].loc[:cut - pd.Timedelta(days=1)].max() > 0
    assert (W["XOP"].loc[:cut - pd.Timedelta(days=1)] == 0).all() and W["XOP"].loc[cut:].max() > 0


def test_priority_history_tolerates_a_sleeve_with_no_closed_trades():
    rng = np.random.default_rng(9)
    idx = pd.bdate_range("2024-01-01", periods=80)
    a = _sleeve("AAA", "AAA", idx, rng)
    b = _sleeve("BBB", "BBB", idx, rng)
    b["r"]["trade_log"] = []                      # no closed trade yet …
    b["r"]["trades"] = np.array([])               # … and an EMPTY ndarray of trade returns
    comp = oc.priority_component_history([a, b], idx)
    assert set(comp) == {"AAA", "BBB"}
    assert (comp["BBB"]["wr"] == 0.5).all()       # neutral win rate, no crash
    rep = oc.replay_gated_allocation([a, b], base_weights={"AAA": 0.5, "BBB": 0.5}, adds_only=0)
    assert rep["weights"]["BBB"].max() > 0


def test_walkforward_anchors_skip_a_sleeve_with_no_history_in_the_window():
    rng = np.random.default_rng(21)
    idx = pd.bdate_range("2023-01-02", periods=400)
    rets = pd.DataFrame({k: rng.normal(0.0005, 0.01, len(idx)) for k in ("AAA", "BBB", "CCC")}, index=idx)
    pos = pd.DataFrame(1.0, index=idx, columns=rets.columns)
    # DDD joins the universe only in the last quarter: zero before that
    joined = pd.Timestamp("2024-04-01")
    rets["DDD"] = np.where(idx >= joined, rng.normal(0.0005, 0.01, len(idx)), 0.0)
    pos["DDD"] = (idx >= joined).astype(float)
    with_ddd = oc.walkforward_anchors(rets, pos=pos, n_samples=300, min_hist=60, seed=3)
    without = oc.walkforward_anchors(rets[["AAA", "BBB", "CCC"]], pos=pos[["AAA", "BBB", "CCC"]],
                                     n_samples=300, min_hist=60, seed=3)
    assert len(with_ddd) == len(without) > 2
    ew = with_ddd[0][1]["DDD"]                       # the warm-up equal-weight constant
    # entry 0 is the warm-up constant (cap-normalised 1/n, so it depends on n);
    # every REFIT entry must match the universe-without-DDD fit exactly
    for (d1, w1), (d2, w2) in zip(with_ddd[1:], without[1:]):
        assert d1 == d2
        if d1 <= joined:                              # DDD had no history → others fitted as if absent
            assert w1["DDD"] == pytest.approx(ew)
            for k in ("AAA", "BBB", "CCC"):
                assert w1[k] == pytest.approx(w2[k])
    # once DDD trades inside the window, the refit is the plain four-sleeve fit
    last_d, last_w = with_ddd[-1]
    assert last_d > joined
    fit_r = rets.loc[:idx[idx < last_d][-1]]
    o = oc.optimize_weights(fit_r, caps=oc.CAP_BY_KEY, n_samples=300, seed=3, mdd_floor=-0.35,
                            pos=pos.loc[fit_r.index], sata_daily=oc.SATA_DAILY,
                            objective="balanced", fundamental=False)
    assert last_w == pytest.approx(o["optimal"]["weights"])


def test_as_published_pricing_basis_is_version_independent():
    """The as-published record compounds PRICE returns (``bh_returns_matrix``):
    identical across V1 / V2 / Combined, and untouched by a sleeve going flat
    or being retired in one generation (whose strategy stream would drop the
    move of a position the book still held)."""
    rng = np.random.default_rng(11)
    cut = pd.Timestamp(sv.STRATEGY_VERSION_START)
    idx = pd.bdate_range(cut - pd.Timedelta(days=60), periods=60)
    xle1, xle2 = _sleeve("XLE", "XLE", idx, rng), _sleeve("XLE", "XLE", idx, rng)
    oih = _sleeve("OIH", "XLE", idx, rng)
    comb = oc.combine_results([xle1, oih], [xle2])
    px_v1 = oc.bh_returns_matrix([xle1, oih])
    px_c = oc.bh_returns_matrix(comb)
    pd.testing.assert_frame_equal(px_v1[["XLE", "OIH"]].fillna(0), px_c[["XLE", "OIH"]].fillna(0))
    # the strategy stream of the retired sleeve IS flat after the cut-over …
    assert (oc.returns_matrix(comb)["OIH"].loc[cut:].fillna(0) == 0).all()
    # … but a book that held it keeps earning its price move on that basis
    books = [dict(as_of=str(idx[5].date()), weights={"OIH": 1.0}, cash_weight=0.0,
                  strategy_version="v1")]
    rep_px = oc.published_book_replay(px_c, books, sata_daily=0.0, only_version={"v1", "v2"})
    rep_strat = oc.published_book_replay(oc.returns_matrix(comb), books, sata_daily=0.0,
                                         only_version={"v1", "v2"})
    assert rep_px is not None and rep_strat is not None
    assert (rep_strat["ret"].loc[cut:] == 0).all()                 # strategy basis: nothing
    pd.testing.assert_series_equal(rep_px["ret"].loc[cut:], px_c["OIH"].loc[cut:],
                                   check_names=False)              # price basis: the real move
    assert (rep_px["weights"]["OIH"].loc[cut:] > 0).all()


def test_replay_start_and_init_weights_continue_a_held_book():
    res = _universe(n_days=100)
    bw = {k: 0.25 for k in ("AAA", "BBB", "CCC", "DDD")}
    full = oc.replay_gated_allocation(res, base_weights=bw, adds_only=0.08)
    cut = res[0]["dates"][60]
    held = {k: float(v) for k, v in full["weights"].loc[:cut - pd.Timedelta(days=1)].iloc[-1].items() if v > 0}
    part = oc.replay_gated_allocation(res, base_weights=bw, adds_only=0.08, start=cut, init_weights=held)
    assert part["weights"].index[0] == cut and len(part["ret"]) == int((full["ret"].index >= cut).sum())
    # continuing from the same held book reproduces the full replay from the cut
    pd.testing.assert_frame_equal(part["weights"], full["weights"].loc[cut:])
    pd.testing.assert_series_equal(part["ret"], full["ret"].loc[cut:])


def test_combined_walkforward_is_a_splice_of_the_two_generations():
    cut = pd.Timestamp(sv.STRATEGY_VERSION_START)
    rng = np.random.default_rng(4)
    idx = pd.bdate_range(cut - pd.Timedelta(days=400), periods=330)
    assert idx[0] < cut < idx[-1]
    def gen(keys, seed):
        r = np.random.default_rng(seed); out = []
        for k in keys:
            s_ = _sleeve(k, k, idx, r); out.append(s_)
        return out
    v1 = gen(("AAA", "BBB", "CCC", "OLD"), 1)      # 4 sleeves: 4 × 30 % cap ≥ 100 %
    v2 = gen(("AAA", "BBB", "CCC", "NEW"), 2)
    comb = oc.combine_results(v1, v2)
    kw = dict(n_samples=200, min_hist=40)
    w1 = oc.walkforward_gated_replay(v1, version="v1", **kw)
    w2 = oc.walkforward_gated_replay(v2, version="v2", **kw)
    wc = oc.walkforward_gated_replay(comb, version="combined", gens=(v1, v2), **kw)
    assert wc.get("spliced") and wc["version"] == "combined"
    pre = lambda s: s.loc[:cut - pd.Timedelta(days=1)]
    pd.testing.assert_series_equal(pre(wc["ret"]), pre(w1["ret"]))            # V1 exactly before
    assert (wc["weights"]["NEW"].loc[:cut - pd.Timedelta(days=1)] == 0).all()
    assert (wc["weights"]["OLD"].loc[cut:] == 0).all()                         # OLD retired from the cut
    assert wc["weights"]["NEW"].loc[cut:].max() > 0
    assert wc["adds_only"] == pytest.approx(oc.ADDS_ONLY_BAND) and wc["cluster_cap"] == pytest.approx(oc.CLUSTER_CAP)
    # V2's own anchors govern the post-cut book (its anchor in force at the cut-over)
    a2 = [w for d, w in w2["anchors"] if pd.Timestamp(d) <= cut][-1]
    ac = [w for d, w in wc["anchors"] if pd.Timestamp(d) == cut][0]
    assert ac == a2
    assert (wc["version_series"].loc[:cut - pd.Timedelta(days=1)] == "v1").all()
    assert (wc["version_series"].loc[cut:] == "v2").all()
