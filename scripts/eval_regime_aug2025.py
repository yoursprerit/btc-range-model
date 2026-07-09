"""Regime-divergence (trained THROUGH Aug-2025) vs the current MA filter vs B&H.

For each MA-mode ETF app (SOXX, VEGN, GRID, REMX, WGMI):

  1. Fit the daily H/L signal model on the pre-OOS window (exactly as the live
     apps do → the long backtest window stays genuinely out-of-sample for the
     forecast bands).
  2. Pick the OPTIMAL divergence thresholds (U1 × D2 × stop × D1-exit) using
     ONLY data up to 2025-08-01 — the "training data until August 2025" the
     brief asks for (no peeking past Aug-2025 when choosing the strategy).
     Selection rule = the repo's own: maximise Sharpe s.t. MDD <= Buy&Hold,
     tie-broken by return.
  3. With those FIXED thresholds, report metrics over:
       * the app's current reported OOS window (like-for-like vs today's config)
       * the true holdout Aug-2025 -> now (data the optimiser never saw)
     against the current MA config and Buy&Hold.
"""
from __future__ import annotations
import sys, itertools, json
from pathlib import Path
import pandas as pd, numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "app")); sys.path.insert(0, str(_REPO))
import ticker_core as tc, backtest_ticker as bt          # noqa
from ticker_config import get_config                      # noqa
from dataclasses import replace                           # noqa

ASSETS = ["SOXX", "VEGN", "GRID", "REMX", "WGMI"]
TRAIN_CUTOFF = "2025-08-01"
HOLDOUT_START = "2025-08-01"
U1S = [0.05, 0.08, 0.12, 0.16]
D2S = [-0.08, -0.10, -0.14, -0.18]


def _load(cfg):
    csv = tc.cache_paths(cfg)["daily"]
    df = pd.read_csv(csv, index_col=0, parse_dates=True)
    if "px_close" not in df.columns:
        pref = f"{cfg.primary_symbol.lower()}_"
        df = df.rename(columns={c: "px_" + c[len(pref):]
                                for c in df.columns if c.startswith(pref)})
    return df


def _metrics(r):
    return bt._metrics(r["strat"], r["dates"]), bt._metrics(r["bh"], r["dates"])


def _fmt(m):
    return f"ret {m['total_ret']*100:+7.1f}%  mdd {m['mdd']*100:6.1f}%  sharpe {m['sharpe']:5.2f}"


results = {}
for key in ASSETS:
    cfg = get_config(key)
    daily = _load(cfg)
    preds = bt.build_predictions(cfg, daily)          # H/L model fit on pre-OOS
    sig = bt.precompute_signals(cfg, preds)
    oos = cfg.oos_start

    # ---- 1. current MA config over the full OOS window ----
    r_ma_full = bt.simulate_regime(cfg, preds, sig, "px_close",
                                   ma_window=cfg.ma_window, stop_pct=cfg.fixed_stop,
                                   oos_start=oos)
    ma_full, bh_full = _metrics(r_ma_full)

    # ---- 2. optimal divergence thresholds trained on oos -> Aug-2025 ONLY ----
    best = None
    for U1, D2, stop, d1x in itertools.product(U1S, D2S, [cfg.fixed_stop, 1.0], [False, True]):
        rt = bt.simulate(cfg, preds, sig, "px_close", stop_pct=stop, U1=U1, D2=D2,
                         D1=cfg.d1_errlo_min, use_d1_exit=d1x,
                         oos_start=oos, end=TRAIN_CUTOFF)
        sm = bt._metrics(rt["strat"], rt["dates"])
        bm = bt._metrics(rt["bh"], rt["dates"])
        ok = sm["mdd"] >= bm["mdd"] - 1e-9          # MDD no worse than B&H (train)
        score = (ok, sm["sharpe"], sm["total_ret"])
        cand = dict(U1=U1, D2=D2, stop=stop, d1x=d1x, _score=score)
        if best is None or score > best["_score"]:
            best = cand

    # ---- 3. evaluate the FIXED optimal divergence over full OOS + holdout ----
    def _div(oos_start, end=None):
        return bt.simulate(cfg, preds, sig, "px_close", stop_pct=best["stop"],
                           U1=best["U1"], D2=best["D2"], D1=cfg.d1_errlo_min,
                           use_d1_exit=best["d1x"], oos_start=oos_start, end=end)

    r_div_full = _div(oos)
    div_full, _ = _metrics(r_div_full)

    r_ma_hold = bt.simulate_regime(cfg, preds, sig, "px_close", ma_window=cfg.ma_window,
                                   stop_pct=cfg.fixed_stop, oos_start=HOLDOUT_START)
    ma_hold, bh_hold = _metrics(r_ma_hold)
    r_div_hold = _div(HOLDOUT_START)
    div_hold, _ = _metrics(r_div_hold)

    tag = (f"U1={best['U1']} D2={best['D2']} "
           f"stop={'none' if best['stop']==1.0 else str(int(best['stop']*100))+'%'} "
           f"d1exit={best['d1x']}")

    print(f"\n===== {key}  (MA{cfg.ma_window}/{int(cfg.fixed_stop*100)}%) =====")
    print(f"  optimal divergence (train {oos}->{TRAIN_CUTOFF}): {tag}")
    print(f"  --- FULL OOS  {oos} -> now ---")
    print(f"    Buy&Hold      : {_fmt(bh_full)}")
    print(f"    MA (current)  : {_fmt(ma_full)}")
    print(f"    Divergence    : {_fmt(div_full)}   "
          f"div>MA sharpe? {div_full['sharpe']>ma_full['sharpe']}  "
          f"ret? {div_full['total_ret']>ma_full['total_ret']}  "
          f"div>B&H sharpe? {div_full['sharpe']>bh_full['sharpe']}")
    print(f"  --- HOLDOUT   {HOLDOUT_START} -> now (unseen) ---")
    print(f"    Buy&Hold      : {_fmt(bh_hold)}")
    print(f"    MA (current)  : {_fmt(ma_hold)}")
    print(f"    Divergence    : {_fmt(div_hold)}   "
          f"div>MA sharpe? {div_hold['sharpe']>ma_hold['sharpe']}  "
          f"ret? {div_hold['total_ret']>ma_hold['total_ret']}")

    results[key] = dict(tag=tag, ma_window=cfg.ma_window, stop=cfg.fixed_stop,
                        full=dict(bh=bh_full, ma=ma_full, div=div_full, oos=oos),
                        holdout=dict(bh=bh_hold, ma=ma_hold, div=div_hold))

Path(_REPO / "scratchpad_regime_eval.json").write_text(json.dumps(results, indent=2, default=str))
print("\nsaved scratchpad_regime_eval.json")
