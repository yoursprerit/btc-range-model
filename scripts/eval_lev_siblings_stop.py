"""Should any leveraged sibling (MSTU · UGL · NUGT · ERX) also go stop-less?

Applies the SOXL stop-width sweep to the OTHER leveraged sleeves, each through
the SAME live engine the Overall app uses (so the result is actionable):

  MSTU (2× MSTR)  → btc_ct_engine  (STOP_PCT["MSTU"], with SL re-entry)  [live: −3%]
  UGL  (2× gold)  → gldm_engine    (STOP_BY_ASSET["UGL"])                [live: −3%]
  NUGT (2× miners)→ gldm_engine    (STOP_BY_ASSET fallback FIXED_STOP)    [live: −3%]
  ERX  (2× energy)→ daily run_asset (XLE divergence, stop_by_asset)       [live: −8%]

For each: sweep the stop none…−30% and report return / Sharpe / win-rate / MDD /
trades on the sleeve's full OOS window, plus the combined equal-weight impact
(current stop vs no stop) — the deterministic, trustworthy portfolio read.

    python scripts/eval_lev_siblings_stop.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "app"))
sys.path.insert(0, str(_REPO))

import backtest_ticker as bt        # noqa: E402
import overall_core as oc           # noqa: E402
import btc_ct_engine as bce         # noqa: E402
import gldm_engine as ge            # noqa: E402
import gldm_core as gc              # noqa: E402
from ticker_config import get_config  # noqa: E402


def _sleeve_row(sl, tag):
    m = sl["metrics"]
    return (f"  {tag:>6s} | {m['total_ret']*100:+10.1f}% {m['mdd']*100:+7.1f}% "
            f"{m['sharpe']:7.2f} {sl['win_rate']:5.0f}% {sl['n_trades']:7d}")


HDR = f"  {'Stop':>6s} | {'Return':>11s} {'MDD':>8s} {'Sharpe':>7s} {'Win%':>6s} {'Trades':>7s}"

# ── grids (include the shipped stop so the current point is visible) ──────
GRID_GOLD = [(1.0, "none"), (0.30, "−30%"), (0.20, "−20%"), (0.15, "−15%"),
             (0.10, "−10%"), (0.05, "−5%"), (0.03, "−3%")]
GRID_MSTU = [(None, "none"), (0.30, "−30%"), (0.20, "−20%"), (0.15, "−15%"),
             (0.10, "−10%"), (0.05, "−5%"), (0.03, "−3%")]
GRID_ERX = [(1.0, "none"), (0.30, "−30%"), (0.20, "−20%"), (0.15, "−15%"),
            (0.10, "−10%"), (0.08, "−8%"), (0.05, "−5%")]

results_cache = {}


# ════════════════════════════════════════════════════════════════════════
# MSTU — btc_ct_engine (patch STOP_PCT["MSTU"])
# ════════════════════════════════════════════════════════════════════════
print("=" * 92)
print("MSTU (2× MSTR) · btc_ct_engine · live stop = −3% · history from 2024")
print("=" * 92)
print(HDR + "\n  " + "-" * 58)
_orig = dict(bce.STOP_PCT)
mstu_sleeves = {}
for stop, tag in GRID_MSTU:
    bce.STOP_PCT["MSTU"] = stop
    grp = bce.run_btc_ct()
    sl = next((r for r in grp if r["key"] == "MSTU"), None)
    if sl:
        mstu_sleeves[tag] = sl
        print(_sleeve_row(sl, tag))
bce.STOP_PCT.update(_orig)


# ════════════════════════════════════════════════════════════════════════
# UGL & NUGT — gldm_engine (patch STOP_BY_ASSET)
# ════════════════════════════════════════════════════════════════════════
for KEY in ("UGL", "NUGT"):
    print("\n" + "=" * 92)
    lev = "2× Gold" if KEY == "UGL" else "2× Gold Miners"
    print(f"{KEY} ({lev}) · gldm_engine · live stop = −3% · OOS 2021→now")
    print("=" * 92)
    print(HDR + "\n  " + "-" * 58)
    _orig_sba = dict(gc.STOP_BY_ASSET)
    _orig_fs = gc.FIXED_STOP
    sleeves = {}
    for stop, tag in GRID_GOLD:
        gc.STOP_BY_ASSET[KEY] = stop          # explicit per-asset override
        grp = ge.run_gldm()
        sl = next((r for r in grp if r["key"] == KEY), None)
        if sl:
            sleeves[tag] = sl
            print(_sleeve_row(sl, tag))
    gc.STOP_BY_ASSET.clear(); gc.STOP_BY_ASSET.update(_orig_sba)
    gc.FIXED_STOP = _orig_fs
    results_cache[KEY] = sleeves


# ════════════════════════════════════════════════════════════════════════
# ERX — daily run_asset on the XLE divergence signal (live path, like SOXL)
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 92)
print("ERX (2× Energy) · XLE divergence daily engine · live stop = −8% · OOS 2021→now")
print("=" * 92)
xle = get_config("XLE")
daily = oc._load_daily(xle)
preds = bt.build_predictions(xle, daily)
sig = bt.precompute_signals(xle, preds)


def _erx_row(stop, tag, s, e=None):
    r = bt.run_strategy(xle, preds, sig, "erx_close", oos_start=s, end=e, stop_pct=stop)
    m = bt._metrics(r["strat"], r["dates"])
    wr = float((r["trades"] > 0).mean() * 100) if len(r["trades"]) else 0.0
    return (f"  {tag:>6s} | {m['total_ret']*100:+10.1f}% {m['mdd']*100:+7.1f}% "
            f"{m['sharpe']:7.2f} {wr:5.0f}% {len(r['trades']):7d}")


print("Full OOS (2021 → now)\n" + HDR + "\n  " + "-" * 58)
for stop, tag in GRID_ERX:
    print(_erx_row(stop, tag, xle.oos_start))
print("\nBear (2021–2022) — where a stop should matter most\n" + HDR + "\n  " + "-" * 58)
for stop, tag in GRID_ERX:
    print(_erx_row(stop, tag, "2021-01-01", "2022-12-31"))


# ════════════════════════════════════════════════════════════════════════
# Overall combined portfolio — equal-weight clean signal, current vs no-stop
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 92)
print("OVERALL — equal-weight blend (deterministic clean signal): current stop vs no-stop")
print("=" * 92)
base = oc.run_universe()


def _ew(results):
    rets = oc.returns_matrix(results)
    pos = oc.position_matrix(results, rets.index)
    w = np.full(rets.shape[1], 1.0 / rets.shape[1])
    return oc.curve_metrics(oc._equity(oc._combine(rets, w, pos, oc.SATA_DAILY)))


def _swap(results, key, sleeve):
    return [r for r in results if r["key"] != key] + ([sleeve] if sleeve else [])


ew_base = _ew(base)
print(f"  baseline (all live stops): ret {ew_base['total_ret']*100:+8.1f}%  "
      f"MDD {ew_base['mdd']*100:+6.1f}%  Sharpe {ew_base['sharpe']:.3f}")

# MSTU no-stop
bce.STOP_PCT["MSTU"] = None
mstu_ns = next((r for r in bce.run_btc_ct() if r["key"] == "MSTU"), None)
bce.STOP_PCT.update(_orig)
# UGL / NUGT no-stop (both at once and individually)
def _gldm_nostop(keys):
    o = dict(gc.STOP_BY_ASSET)
    for k in keys:
        gc.STOP_BY_ASSET[k] = 1.0
    grp = ge.run_gldm()
    gc.STOP_BY_ASSET.clear(); gc.STOP_BY_ASSET.update(o)
    return {k: next((r for r in grp if r["key"] == k), None) for k in keys}

gold_ns = _gldm_nostop(["UGL", "NUGT"])
# ERX no-stop via run_asset
xcfg = replace(xle, stop_by_asset={"erx_close": 1.0})
erx_ns = next((r for r in oc.run_asset(xcfg) if r["key"] == "ERX"), None)

for key, sleeve in (("MSTU", mstu_ns), ("UGL", gold_ns["UGL"]),
                    ("NUGT", gold_ns["NUGT"]), ("ERX", erx_ns)):
    if sleeve is None:
        print(f"  {key}: sleeve rebuild failed"); continue
    ew = _ew(_swap(base, key, sleeve))
    print(f"  {key} no-stop: ret {ew['total_ret']*100:+8.1f}% ({(ew['total_ret']-ew_base['total_ret'])*100:+6.1f}pp)  "
          f"MDD {ew['mdd']*100:+6.1f}%  Sharpe {ew['sharpe']:.3f} ({ew['sharpe']-ew_base['sharpe']:+.3f})")

print("\nDone.", flush=True)
