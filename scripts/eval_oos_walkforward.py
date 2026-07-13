"""Out-of-sample walk-forward evaluation of the Overall Trading strategy.

Read-only. Changes nothing. Reproduces the tables in OVERALL_OOS_WALKFORWARD_EVAL.md:
  * sanity check — app committed weights reproduce the artifact headline exactly;
  * decomposition ladder — how the app's 2022->now CAGR builds up from the honest
    out-of-sample baseline (in-sample weights -> +names -> +leverage -> +SATA);
  * final matched comparison — leverage + SATA + crypto sleeve on BOTH sides.

Method: portfolio weights are fit on 2021 only (through 2021-12-31), frozen, and
applied untouched to 2022-01-01 -> now. The signals are already out-of-sample (each
asset's H/L model is trained on its pre-2021 window). The app, by contrast, fits its
weights on the full 2021->now history (in-sample). BTC/MSTR/MSTU/WGMI have no 2021
data, so their weights (where used) are imported from the app (acknowledged hindsight)
and the other 13 assets are fit out-of-sample on 2021.

    python scripts/eval_oos_walkforward.py
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "app"))
sys.path.insert(0, str(_REPO))

import numpy as np
import pandas as pd
import overall_core as oc

WIN, END, SAT = "2022-01-01", "2026-12-31", oc.SATA_DAILY
EXCL = ["BTC", "MSTR", "MSTU", "WGMI"]          # no 2021 data -> weights imported from app


def _load():
    art = json.load(open(_REPO / "data" / "overall" / "overall_results.json"))
    results = oc.run_universe()
    by = {r["key"]: r for r in results}
    cols_all = [r["key"] for r in results]
    strat, pos = {}, {}
    for k in cols_all:
        strat[k] = by[k]["ret"]
        pos[k] = by[k]["pos_series"]
    STRAT = pd.DataFrame(strat).sort_index()
    POS = pd.DataFrame(pos).reindex(STRAT.index).fillna(0.0)
    return art, STRAT, POS, cols_all


def main():
    art, STRAT, POS, ALL = _load()
    LEV = {k for k, m in oc.ASSET_META.items() if m["kind"] == "lev"}
    UNLEV = [k for k in ALL if k not in LEV]

    def cm(dret):
        d = pd.Series(dret).dropna()
        return oc.curve_metrics((1 + d).cumprod()) if len(d) > 10 else None

    def combine(cols, w, use_pos, sata):
        return oc._combine(STRAT[cols], w, POS[cols] if use_pos else None, sata_daily=sata)

    def win(cols, w, sata):                       # metrics over 2022->now window
        return cm(combine(cols, w, True, sata).loc[WIN:END])

    def fit(cols, fit_end, prof, sata):
        cfg = oc.RISK_PROFILES[prof]
        capd = {c: oc.caps_for(prof)[c] for c in cols}
        o = oc.optimize_weights(STRAT[cols].loc[:fit_end], caps=capd, seed=7,
                                mdd_floor=cfg["mdd_floor"], pos=POS[cols].loc[:fit_end],
                                sata_daily=sata, objective=cfg["objective"], fundamental=False)
        return o["cols"], np.array([o["optimal"]["weights"][c] for c in o["cols"]])

    UNLEV_2021 = [k for k in UNLEV if STRAT[k].loc[:"2021-12-31"].dropna().shape[0] > 150]
    UNLEV_FULL = UNLEV
    PROFILES = ("Balanced", "Growth", "Aggressive")

    print("=== SANITY: app committed weights reproduce the artifact (full period) ===")
    for p in PROFILES:
        cols = art["cols"]; w = np.array([art["profiles"][p]["optimal"]["weights"][c] for c in cols])
        m = cm(combine(cols, w, True, SAT))
        a = art["profiles"][p]["optimal"]
        print(f"  {p:11s} computed CAGR {m['cagr']*100:5.1f}%  | artifact CAGR {a['cagr']*100:5.1f}%  "
              f"({'OK' if abs(m['cagr']-a['cagr'])<1e-3 else 'MISMATCH'})")

    hdr = f"{'':46s}{'total ret':>12s}{'CAGR':>8s}{'Sharpe':>8s}{'MaxDD':>8s}"
    print(f"\n=== DECOMPOSITION — how the app's {WIN}->now CAGR builds up ===")
    for p in PROFILES:
        print(f"\n[{p}]\n{hdr}")
        rungs = []
        c, w = fit(UNLEV_2021, "2021-12-31", p, 0.0); rungs.append(("1. TRUE OOS (unlev, 2021-fit, no SATA)", win(c, w, 0.0)))
        c, w = fit(UNLEV_2021, END, p, 0.0);          rungs.append(("2.  + in-sample weights (unlev-9)", win(c, w, 0.0)))
        c, w = fit(UNLEV_FULL, END, p, 0.0);          rungs.append(("3.  + add BTC/MSTR/WGMI (unlev-12)", win(c, w, 0.0)))
        c, w = fit(ALL, END, p, 0.0);                 rungs.append(("4.  + leverage (all 17)", win(c, w, 0.0)))
        rungs.append(("5.  + SATA idle-cash yield", win(c, w, SAT)))
        cols = art["cols"]; w6 = np.array([art["profiles"][p]["optimal"]["weights"][cc] for cc in cols])
        rungs.append(("6. APP AS-IS (committed wts + SATA)", win(cols, w6, SAT)))
        for name, m in rungs:
            print(f"   {name:46s}{m['total_ret']*100:>+11.1f}%{m['cagr']*100:>+7.1f}%{m['sharpe']:>8.2f}{m['mdd']*100:>+7.1f}%")

    print(f"\n=== FINAL MATCHED COMPARISON — leverage + SATA + crypto on BOTH sides ({WIN}->now) ===")
    for p in PROFILES:
        cols = art["cols"]; wd = art["profiles"][p]["optimal"]["weights"]
        F = [c for c in cols if c not in EXCL]
        W_E = sum(wd[c] for c in EXCL)
        cF, wF = fit(F, "2021-12-31", p, SAT)
        wFd = {cF[i]: wF[i] for i in range(len(cF))}
        combined = np.array([wd[c] if c in EXCL else wFd.get(c, 0.0) * (1 - W_E) for c in cols])
        oosm = win(cols, combined, SAT)
        appm = win(cols, np.array([wd[c] for c in cols]), SAT)
        cw = " ".join(f"{c} {wd[c]*100:.0f}%" for c in EXCL if wd[c] > 0.01)
        print(f"\n[{p}]  imported crypto wts: {cw}")
        print(f"   {'OOS+crypto (rest OOS-fit)':46s}{oosm['total_ret']*100:>+11.1f}%{oosm['cagr']*100:>+7.1f}%{oosm['sharpe']:>8.2f}{oosm['mdd']*100:>+7.1f}%")
        print(f"   {'APP as-is':46s}{appm['total_ret']*100:>+11.1f}%{appm['cagr']*100:>+7.1f}%{appm['sharpe']:>8.2f}{appm['mdd']*100:>+7.1f}%")
        print(f"   -> residual raw-return gap {appm['cagr']/oosm['cagr']:.2f}x (= non-crypto in-sample weighting)")


if __name__ == "__main__":
    main()
