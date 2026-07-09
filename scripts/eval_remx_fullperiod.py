"""Find REMX's best strategy over the FULL period (2015 -> now, the 🌍 Full
history bull+bear window), across every strategy family the engine supports.

Objective (the repo's own): maximise full-period total return subject to a max
drawdown no worse than Buy&Hold, tie-broken by Sharpe.  Also reports each
candidate's 2021->now OOS metrics for context.
"""
from __future__ import annotations
import sys, itertools
from pathlib import Path
from dataclasses import replace
import numpy as np, pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "app")); sys.path.insert(0, str(_REPO))
import ticker_core as tc, backtest_ticker as bt
from ticker_config import get_config

FULL = "2015-01-01"
OOS = "2021-01-01"
cfg0 = get_config("REMX")
daily = pd.read_csv(tc.cache_paths(cfg0)["daily"], index_col=0, parse_dates=True)
if "px_close" not in daily.columns:
    pref = cfg0.primary_symbol.lower() + "_"
    daily = daily.rename(columns={c: "px_" + c[len(pref):] for c in daily.columns if c.startswith(pref)})
preds = bt.build_predictions(cfg0, daily)
sig = bt.precompute_signals(cfg0, preds)


def metr(cfg, start):
    r = bt.run_strategy(cfg, preds, sig, "px_close", oos_start=start)
    return bt._metrics(r["strat"], r["dates"]), len(r["trades"]), r


bh_full, _, _ = metr(cfg0, FULL)
bh_full = bt._metrics(bt.run_strategy(cfg0, preds, sig, "px_close", oos_start=FULL)["bh"],
                      bt.run_strategy(cfg0, preds, sig, "px_close", oos_start=FULL)["dates"])

cands = []
# MA single
for w in (20, 30, 40, 50, 60, 80, 100, 120, 150, 180, 200, 250):
    for stop in (1.0, 0.05, 0.08, 0.10):
        cands.append(replace(cfg0, strategy_mode="ma", ma_window=w, fixed_stop=stop))
# dual-MA
for f, s in itertools.product((10, 20, 25, 30, 40, 50), (80, 100, 120, 150, 200)):
    if f >= s:
        continue
    for stop in (1.0, 0.05):
        cands.append(replace(cfg0, strategy_mode="dual_ma", ma_fast=f, ma_slow=s, fixed_stop=stop))
# MACD
for f, s, sg in itertools.product((8, 10, 12), (20, 26, 34), (9,)):
    for stop in (1.0, 0.05):
        cands.append(replace(cfg0, strategy_mode="macd", macd_fast=f, macd_slow=s, macd_signal=sg, fixed_stop=stop))
# MA + vol filter
for w, vw, mw, k in itertools.product((50, 100, 150), (10, 20), (189, 252), (0.95, 1.0, 1.1)):
    for stop in (1.0, 0.05):
        cands.append(replace(cfg0, strategy_mode="ma_vol", ma_window=w, vol_win=vw,
                             vol_med_win=mw, vol_k=k, fixed_stop=stop))
# divergence (current + a few thresholds)
for U1, D2, stop, d1x in itertools.product((0.05, 0.12, 0.16), (-0.10, -0.18), (1.0, 0.05), (False, True)):
    cands.append(replace(cfg0, strategy_mode="divergence", u1_errhi_min=U1, d2_errhi_max=D2,
                         fixed_stop=stop, use_d1_exit=d1x))


def label(c):
    if c.strategy_mode == "ma":
        return f"MA{c.ma_window}/{'—' if c.fixed_stop>=0.999 else str(int(c.fixed_stop*100))+'%'}"
    if c.strategy_mode == "dual_ma":
        return f"Dual {c.ma_fast}/{c.ma_slow}/{'—' if c.fixed_stop>=0.999 else str(int(c.fixed_stop*100))+'%'}"
    if c.strategy_mode == "macd":
        return f"MACD {c.macd_fast}/{c.macd_slow}/{c.macd_signal}/{'—' if c.fixed_stop>=0.999 else str(int(c.fixed_stop*100))+'%'}"
    if c.strategy_mode == "ma_vol":
        return f"MA{c.ma_window}+vol vw{c.vol_win} m{c.vol_med_win} k{c.vol_k}/{'—' if c.fixed_stop>=0.999 else str(int(c.fixed_stop*100))+'%'}"
    return f"Div U1{c.u1_errhi_min} D2{c.d2_errhi_max} d1x{int(c.use_d1_exit)}/{'—' if c.fixed_stop>=0.999 else str(int(c.fixed_stop*100))+'%'}"


rows = []
for c in cands:
    mf, ntr, _ = metr(c, FULL)
    mo, _, _ = metr(c, OOS)
    rows.append((c, mf, mo, ntr))

print(f"REMX Buy&Hold FULL {FULL}->now: ret {bh_full['total_ret']*100:+.1f}%  "
      f"mdd {bh_full['mdd']*100:.1f}%  sharpe {bh_full['sharpe']:.2f}\n")

feas = [r for r in rows if r[1]["mdd"] >= bh_full["mdd"] - 1e-9]
pool = feas if feas else rows
# maximise full-period return s.t. MDD <= B&H, tie-break Sharpe
pool.sort(key=lambda r: (r[1]["total_ret"], r[1]["sharpe"]), reverse=True)
print("=== TOP by full-period return (MDD <= B&H) ===")
print(f"{'strategy':40s} {'FULL ret/mdd/shrp':26s} {'OOS21 ret/mdd/shrp':24s} ntr")
for c, mf, mo, ntr in pool[:12]:
    print(f"{label(c):40s} {mf['total_ret']*100:+7.1f}% {mf['mdd']*100:6.1f}% {mf['sharpe']:5.2f}   "
          f"{mo['total_ret']*100:+7.1f}% {mo['mdd']*100:6.1f}% {mo['sharpe']:5.2f}  {ntr}")

# also best by full Sharpe
print("\n=== TOP by full-period Sharpe (MDD <= B&H) ===")
for c, mf, mo, ntr in sorted(pool, key=lambda r: r[1]["sharpe"], reverse=True)[:6]:
    print(f"{label(c):40s} {mf['total_ret']*100:+7.1f}% {mf['mdd']*100:6.1f}% {mf['sharpe']:5.2f}   "
          f"{mo['total_ret']*100:+7.1f}% {mo['mdd']*100:6.1f}% {mo['sharpe']:5.2f}  {ntr}")
