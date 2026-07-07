"""Build & cache the Overall Trading combined back-test artifacts.

Runs every asset through the unified daily engine, computes the optimal
cross-asset allocation and the combined portfolio back-test, and writes a
compact JSON the app can load instantly.  Re-run to refresh.

    python scripts/build_overall.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "app"))
sys.path.insert(0, str(_REPO))

import overall_core as oc   # noqa: E402


def main():
    print("Running universe (live fetch, ~30-90s)…", flush=True)
    results = oc.run_universe()
    print(f"  ran {len(results)} instruments across {len(oc.PARENT_KEYS)} apps: "
          f"{[(r['key'], r['parent']) for r in results]}", flush=True)

    rets = oc.returns_matrix(results)
    pos = oc.position_matrix(results, rets.index)
    print("Return matrix:", rets.shape, rets.index[0].date(), "->", rets.index[-1].date())

    opt = oc.optimize_weights(rets, pos=pos, sata_daily=oc.SATA_DAILY)
    w = np.array([opt["optimal"]["weights"][c] for c in opt["cols"]])
    print("\nOptimal weights:")
    for c in opt["cols"]:
        print(f"  {c:5s} {opt['optimal']['weights'][c]*100:5.1f}%")
    print(f"  Optimal    ret {opt['optimal']['total_ret']*100:7.1f}%  "
          f"CAGR {opt['optimal']['cagr']*100:5.1f}%  MDD {opt['optimal']['mdd']*100:6.1f}%  "
          f"Sharpe {opt['optimal']['sharpe']:.2f}")
    print(f"  EqualWt    ret {opt['equal']['total_ret']*100:7.1f}%  "
          f"CAGR {opt['equal']['cagr']*100:5.1f}%  MDD {opt['equal']['mdd']*100:6.1f}%  "
          f"Sharpe {opt['equal']['sharpe']:.2f}")
    print(f"  RiskParity ret {opt['risk_parity']['total_ret']*100:7.1f}%  "
          f"CAGR {opt['risk_parity']['cagr']*100:5.1f}%  MDD {opt['risk_parity']['mdd']*100:6.1f}%  "
          f"Sharpe {opt['risk_parity']['sharpe']:.2f}")

    bm = oc.benchmarks(rets, results, pos=pos, sata_daily=oc.SATA_DAILY)
    print("\nBenchmarks:")
    print(f"  8× Buy&Hold (EW)   ret {bm['bh_equal']['total_ret']*100:7.1f}%  "
          f"MDD {bm['bh_equal']['mdd']*100:6.1f}%  Sharpe {bm['bh_equal']['sharpe']:.2f}")
    print(f"  8× Strategy (EW)   ret {bm['strat_equal']['total_ret']*100:7.1f}%  "
          f"MDD {bm['strat_equal']['mdd']*100:6.1f}%  Sharpe {bm['strat_equal']['sharpe']:.2f}")
    print(f"  Best single ({bm['best_single']['key']}) ret {bm['best_single']['total_ret']*100:7.1f}%  "
          f"MDD {bm['best_single']['mdd']*100:6.1f}%  Sharpe {bm['best_single']['sharpe']:.2f}")

    per = oc.period_breakdown(rets, w, oc.COMBINED_PERIODS, pos=pos, sata_daily=oc.SATA_DAILY)
    print("\nOptimal combined by period:")
    for row in per:
        print(f"  {row['label']:28s} ret {row['total_ret']*100:8.1f}%  "
              f"MDD {row['mdd']*100:6.1f}%  Sharpe {row['sharpe']:.2f}")

    gate = oc.signal_gated_allocation(results, opt["optimal"]["weights"])
    print(f"\nToday: {gate['n_active']} positions open, {gate['n_open']} new entries, "
          f"SATA {gate['sata']*100:.0f}%")
    print(f"  Priority rank: {gate['priority_rank']}")
    for a in gate["actions"]:
        pr = f"prio {a['priority']:.2f}" if a["priority"] is not None else ""
        print(f"  {a['action']:11s} {a['key']:5s} target {a['target']*100:5.1f}%  "
              f"{pr:9s} [{a['decision']}]")

    # persist compact JSON -------------------------------------------------
    outdir = _REPO / "data" / "overall"
    outdir.mkdir(parents=True, exist_ok=True)
    payload = dict(
        as_of=str(rets.index[-1].date()),
        assets=[r["key"] for r in results],
        optimal=opt["optimal"], equal=opt["equal"], risk_parity=opt["risk_parity"],
        cols=opt["cols"],
        benchmarks={k: {kk: vv for kk, vv in v.items() if kk != "equity"}
                    for k, v in bm.items()},
        periods=per,
    )
    (outdir / "overall_results.json").write_text(json.dumps(payload, indent=1))
    print(f"\nWrote {outdir/'overall_results.json'}")


if __name__ == "__main__":
    main()
