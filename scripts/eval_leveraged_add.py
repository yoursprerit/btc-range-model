"""Do leveraged siblings (SOXL, ERX, NUGT) improve the Overall combined back-test?

This reproduces the SOXL_ERX_ADDITION_EVAL.md analysis.  It builds each
leveraged sleeve *driven by an existing tuned parent signal* (the repo's sibling
architecture — SOXL off the SOXX 25/100 dual-MA, ERX and OIH off the XLE energy
divergence, NUGT off the gold divergence) and folds it into the live universe,
re-running the IDENTICAL optimize_weights() the app/build script use.

Every sleeve is built in ONE run so all deltas share the same baseline optimise.
SOXL and ERX are the two promoted into the repo (SOXX / XLE configs); NUGT is
reported as a strong optional third.

    python scripts/eval_leveraged_add.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np  # noqa: F401  (kept for parity with the other eval scripts)

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "app"))
sys.path.insert(0, str(_REPO))

import overall_core as oc   # noqa: E402


# (key, parent-signal, yahoo-symbol) — each sibling is traded off the PARENT signal
SIBLINGS = [("SOXL", "SOXX", "SOXL"),
            ("ERX",  "XLE",  "ERX"),
            ("NUGT", "GLDM", "NUGT")]


def sib_sleeve(parent_key: str, sym: str) -> list[dict]:
    """Build ONLY the leveraged sibling, driven by the parent's tuned signal."""
    pc = oc.overall_config(parent_key)      # handles BTC/GLDM specials + ETF configs
    cfg = replace(pc, extra_syms={**pc.extra_syms, sym.lower(): sym},
                  traded_assets=[(sym, f"{sym.lower()}_close")],
                  asset_labels={f"{sym.lower()}_close": sym})
    return [s for s in oc.run_asset(cfg) if s["key"] == sym]


def optimise(results):
    rets = oc.returns_matrix(results)
    pos = oc.position_matrix(results, rets.index)
    return {n: oc.optimize_weights(rets, caps=oc.caps_for(n), pos=pos,
                                   sata_daily=oc.SATA_DAILY, mdd_floor=p["mdd_floor"],
                                   objective=p["objective"])
            for n, p in oc.RISK_PROFILES.items()}


def line(o, b, keys):
    w = " ".join(f"{k} {o['weights'].get(k, 0) * 100:4.1f}%" for k in keys)
    return (f"    ret {b['total_ret']*100:7.1f}%→{o['total_ret']*100:7.1f}% "
            f"({(o['total_ret']-b['total_ret'])*100:+7.1f}pp)  "
            f"MDD {b['mdd']*100:6.2f}%→{o['mdd']*100:6.2f}%  "
            f"Sharpe {b['sharpe']:.3f}→{o['sharpe']:.3f} ({o['sharpe']-b['sharpe']:+.3f})  [{w}]")


print("Running baseline universe (live fetch, ~30-90s)…", flush=True)
base = oc.run_universe()
# strip the promoted siblings from the baseline so the comparison is "before":
BASE_ONLY = [r for r in base if r["key"] not in ("SOXL", "ERX")]
print(f"  baseline universe ({len(BASE_ONLY)} instruments): "
      f"{[r['key'] for r in BASE_ONLY]}", flush=True)
opt_base = optimise(BASE_ONLY)

print("Building leveraged sleeves…", flush=True)
sleeves = {}
for key, parent, sym in SIBLINGS:
    sl = sib_sleeve(parent, sym)
    if sl:
        m = sl[0]["metrics"]
        print(f"  {key:5s} own strat {m['total_ret']*100:6.0f}% "
              f"MDD {m['mdd']*100:5.0f}% Sharpe {m['sharpe']:.2f}", flush=True)
        sleeves[key] = sl

scenarios = {
    "+SOXL only":        BASE_ONLY + sleeves.get("SOXL", []),
    "+ERX only":         BASE_ONLY + sleeves.get("ERX", []),
    "+SOXL+ERX (pair)":  BASE_ONLY + sleeves.get("SOXL", []) + sleeves.get("ERX", []),
    "+NUGT only":        BASE_ONLY + sleeves.get("NUGT", []),
}

for name, res in scenarios.items():
    print("\n" + "=" * 92 + f"\n### {name}\n" + "=" * 92, flush=True)
    oc2 = optimise(res)
    keys = [k for k in ("SOXL", "ERX", "NUGT") if any(r["key"] == k for r in res)]
    for prof in ("Balanced", "Growth", "Aggressive"):
        print(f"  [{prof:10s} optimal]", flush=True)
        print(line(oc2[prof]["optimal"], opt_base[prof]["optimal"], keys), flush=True)
    print("  [equal-weight — deterministic, clean attributable signal]", flush=True)
    print(line(oc2["Aggressive"]["equal"], opt_base["Aggressive"]["equal"], keys), flush=True)

print("\nDone.", flush=True)
