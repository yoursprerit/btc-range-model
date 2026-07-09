"""Does feeding the optimal-divergence signals into the Overall allocation mix
improve the combined portfolio?

Baseline = live universe with each app's CURRENT strategy.  Then re-run the
5 ETF sleeves (SOXX/VEGN/GRID/REMX/WGMI) in divergence mode with the thresholds
that were optimal on training-through-Aug-2025, rebuild the returns matrix and
re-optimise the cross-asset weights with the EXACT same optimiser call the live
app uses.  Scenario A swaps REMX only (the sole individual winner); Scenario B
swaps all five.
"""
from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import replace
import numpy as np, pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "app")); sys.path.insert(0, str(_REPO))
import overall_core as oc          # noqa
from ticker_config import get_config

OPT = {   # optimal divergence thresholds (trained oos->Aug-2025); stop 1.0 = none
    "SOXX": dict(u1_errhi_min=0.12, d2_errhi_max=-0.08, fixed_stop=1.0, use_d1_exit=True),
    "VEGN": dict(u1_errhi_min=0.05, d2_errhi_max=-0.18, fixed_stop=0.05, use_d1_exit=True),
    "GRID": dict(u1_errhi_min=0.05, d2_errhi_max=-0.10, fixed_stop=1.0, use_d1_exit=True),
    "REMX": dict(u1_errhi_min=0.16, d2_errhi_max=-0.18, fixed_stop=1.0, use_d1_exit=False),
    "WGMI": dict(u1_errhi_min=0.05, d2_errhi_max=-0.14, fixed_stop=1.0, use_d1_exit=False),
}


def optimise(results):
    rets = oc.returns_matrix(results)
    pos = oc.position_matrix(results, rets.index)
    opt = oc.optimize_weights(rets, pos=pos, sata_daily=oc.SATA_DAILY)
    return opt, rets


def show(tag, opt, keys=("REMX", "WGMI", "SOXX", "VEGN", "GRID")):
    o = opt["optimal"]
    w = {k: o["weights"].get(k, 0.0) for k in keys}
    ws = "  ".join(f"{k}={w[k]*100:.2f}%" for k in keys)
    print(f"  {tag:22s} ret {o['total_ret']*100:8.1f}%  CAGR {o['cagr']*100:5.1f}%  "
          f"MDD {o['mdd']*100:6.2f}%  Sharpe {o['sharpe']:.3f}  vol {o['vol']*100:.2f}%")
    print(f"    weights: {ws}")


print("Running baseline universe (live fetch)…", flush=True)
base = oc.run_universe()
print(f"  {len(base)} instruments: {[r['key'] for r in base]}")
if oc._LAST_ERRORS:
    print("  LOAD ERRORS:", oc._LAST_ERRORS)

opt_base, rets_base = optimise(base)
print(f"\nReturn matrix {rets_base.shape}  {rets_base.index[0].date()} -> {rets_base.index[-1].date()}")
print("\n=== BASELINE (current strategies) ===")
show("baseline", opt_base)


def rerun_app(app_key):
    cfg = replace(get_config(app_key), strategy_mode="divergence", **OPT[app_key])
    return oc.run_asset(cfg)


def swap(base_results, app_keys):
    swapped_keys = set(app_keys)
    new = [r for r in base_results if r["parent"] not in swapped_keys]
    for ak in app_keys:
        new.extend(rerun_app(ak))
    # preserve original column order
    order = [r["key"] for r in base_results]
    by_key = {r["key"]: r for r in new}
    return [by_key[k] for k in order if k in by_key]


print("\n=== SCENARIO A: REMX -> divergence (only individual winner) ===")
resA = swap(base, ["REMX"])
optA, _ = optimise(resA)
show("A: REMX=div", optA)

print("\n=== SCENARIO B: all 5 ETFs -> divergence ===")
resB = swap(base, ["SOXX", "VEGN", "GRID", "REMX", "WGMI"])
optB, _ = optimise(resB)
show("B: all5=div", optB)

print("\n=== DELTAS vs baseline (optimal blend, full 2021->now) ===")
for tag, opt in [("A REMX-only", optA), ("B all-5", optB)]:
    b, o = opt_base["optimal"], opt["optimal"]
    print(f"  {tag:12s} dRet {(o['total_ret']-b['total_ret'])*100:+6.1f}pp  "
          f"dCAGR {(o['cagr']-b['cagr'])*100:+5.2f}pp  "
          f"dMDD {(o['mdd']-b['mdd'])*100:+5.2f}pp  "
          f"dSharpe {o['sharpe']-b['sharpe']:+.3f}")

# ---- recent / holdout sub-windows: apply each scenario's OWN optimal weights
#      to its OWN return streams (the app sets weights once, then lives with them) ----
def sub_metrics(results, opt, start):
    rets = oc.returns_matrix(results); pos = oc.position_matrix(results, rets.index)
    w = np.array([opt["optimal"]["weights"].get(c, 0.0) for c in rets.columns])
    r = rets.loc[pd.Timestamp(start):]; p = pos.loc[pd.Timestamp(start):]
    eq = oc._equity(oc._combine(r, w, p, oc.SATA_DAILY))
    return oc.curve_metrics(eq)

print("\n=== RECENT-WINDOW blend (each scenario's own optimal weights) ===")
for start in ("2025-01-01", "2025-08-01"):
    print(f"  window {start} -> now:")
    for tag, res, opt in [("baseline", base, opt_base), ("A REMX=div", resA, optA),
                          ("B all5=div", resB, optB)]:
        m = sub_metrics(res, opt, start)
        print(f"    {tag:12s} ret {m['total_ret']*100:7.1f}%  MDD {m['mdd']*100:6.2f}%  "
              f"Sharpe {m['sharpe']:.3f}  vol {m['vol']*100:.2f}%")
