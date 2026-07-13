"""Fractional-Kelly-as-risk-dial — evaluation vs the existing 3 risk profiles.

READ-ONLY EVALUATION.  No strategy, profile, cap, or weight is changed.

The idea under test: replace the three hand-tuned risk profiles (which vary
per-instrument CAPS + objective + drawdown budget) with ONE growth-optimal
(full-Kelly) blend that is simply DEPLOYED at fraction f of the bankroll, the
rest parked in SATA cash — f being the risk dial (Aggressive=1x, Growth=1/2x,
Balanced=1/4x).

Claim to verify: this "Pareto-improves risk-adjusted return" vs the existing
profiles.  The fair test is a MATCHED-RISK comparison: dial f down until the
fractional-Kelly drawdown (or vol) equals each existing profile's, then compare
returns / Sharpe.  Higher return at equal risk = genuine improvement; lower =
the claim fails.

All comparisons are fundamental-tilt OFF (isolate the allocation mechanism);
a tilt-ON parity line is printed for the default profile.

Reproduce:  python scripts/eval_kelly_dial.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "app"))
sys.path.insert(0, str(_REPO))

import overall_core as oc   # noqa: E402

CUR_OBJ = {"Balanced": "balanced", "Growth": "max_return", "Aggressive": "max_return"}


def calmar(m):
    return m["cagr"] / abs(m["mdd"]) if m["mdd"] else float("nan")


def _fmt(m):
    return (f"ret {m['total_ret']*100:8.1f}%  CAGR {m['cagr']*100:5.1f}%  "
            f"MDD {m['mdd']*100:6.2f}%  Sharpe {m['sharpe']:5.2f}  "
            f"vol {m['vol']*100:5.1f}%  Calmar {calmar(m):4.2f}")


def frac_curve(r_full, sata_daily, f):
    return oc._equity(f * r_full + (1.0 - f) * sata_daily)


def frac_metrics(r_full, sata_daily, f):
    return oc.curve_metrics(frac_curve(r_full, sata_daily, f))


def solve_f(r_full, sata_daily, key, target, lo=0.001, hi=1.0):
    """Bisection on f so frac-Kelly metric[key] matches target (monotone in f).
    For 'mdd' target is negative and metric increases (toward 0) as f falls."""
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        v = frac_metrics(r_full, sata_daily, mid)[key]
        if key == "mdd":               # mdd rises (less negative) as f falls
            if v < target:             # too deep -> reduce f
                hi = mid
            else:
                lo = mid
        else:                          # vol/total_ret increase with f
            if v > target:
                hi = mid
            else:
                lo = mid
    return 0.5 * (lo + hi)


def main():
    print("Running universe (live fetch)…", flush=True)
    results = oc.run_universe()
    rets = oc.returns_matrix(results)
    pos = oc.position_matrix(results, rets.index)
    sata = oc.SATA_DAILY
    cols = list(rets.columns)
    print(f"  {len(results)} instruments, matrix {rets.shape}  "
          f"{rets.index[0].date()} -> {rets.index[-1].date()}")

    # ── existing profiles (tilt OFF), keep their metrics AND curves ───────────
    existing = {}
    for name, prof in oc.RISK_PROFILES.items():
        o = oc.optimize_weights(rets, caps=oc.caps_for(name), pos=pos, sata_daily=sata,
                                mdd_floor=prof["mdd_floor"], objective=CUR_OBJ[name],
                                fundamental=False)
        existing[name] = o["optimal"]

    # ── ONE growth-optimal base blend (Kelly @ widest / Aggressive caps) ──────
    base = oc.optimize_weights(rets, caps=oc.caps_for("Aggressive"), pos=pos,
                               sata_daily=sata, mdd_floor=oc.RISK_PROFILES["Aggressive"]["mdd_floor"],
                               objective="kelly", fundamental=False)
    w_star = np.array([base["optimal"]["weights"][c] for c in cols])
    r_full = oc._combine(rets, w_star, pos, sata)
    print(f"\nGrowth-optimal base blend (Kelly, Aggressive caps): "
          f"{_fmt(base['optimal'])}")

    # ── A. literal proposal: 1x / 1/2x / 1/4x of the base blend ───────────────
    print("\n" + "=" * 88)
    print("A. LITERAL DIAL — full/half/quarter of ONE growth-optimal blend  vs  existing profiles")
    print("=" * 88)
    tiers = [("Aggressive", 1.00), ("Growth", 0.50), ("Balanced", 0.25)]
    for name, f in tiers:
        dm = frac_metrics(r_full, sata, f)
        em = existing[name]
        print(f"\n[{name} tier]")
        print(f"   existing  ({CUR_OBJ[name]:10s})  {_fmt(em)}")
        print(f"   {f:.2f}x-Kelly dial            {_fmt(dm)}")
        print(f"   Δ dial-existing: dCAGR {(dm['cagr']-em['cagr'])*100:+5.1f}pp  "
              f"dMDD {(dm['mdd']-em['mdd'])*100:+6.2f}pp  dSharpe {dm['sharpe']-em['sharpe']:+5.2f}  "
              f"dCalmar {calmar(dm)-calmar(em):+5.2f}")

    # ── B. MATCHED-DRAWDOWN — the honest Pareto test ──────────────────────────
    print("\n" + "=" * 88)
    print("B. MATCHED DRAWDOWN — dial f set so frac-Kelly MDD == each existing profile's MDD")
    print("=" * 88)
    for name in ("Balanced", "Growth", "Aggressive"):
        em = existing[name]
        f = solve_f(r_full, sata, "mdd", em["mdd"])
        dm = frac_metrics(r_full, sata, f)
        better = "dial WINS" if dm["cagr"] > em["cagr"] else "existing wins"
        print(f"\n[{name}]  target MDD {em['mdd']*100:.2f}%  ->  dial f = {f:.3f}")
        print(f"   existing   {_fmt(em)}")
        print(f"   dial(f={f:.2f}) {_fmt(dm)}")
        print(f"   at equal drawdown: dCAGR {(dm['cagr']-em['cagr'])*100:+5.1f}pp  "
              f"dSharpe {dm['sharpe']-em['sharpe']:+5.2f}  ->  {better} on return")

    # ── C. MATCHED-VOLATILITY — alternative risk match ────────────────────────
    print("\n" + "=" * 88)
    print("C. MATCHED VOLATILITY — dial f set so frac-Kelly vol == each existing profile's vol")
    print("=" * 88)
    for name in ("Balanced", "Growth", "Aggressive"):
        em = existing[name]
        f = solve_f(r_full, sata, "vol", em["vol"])
        dm = frac_metrics(r_full, sata, f)
        better = "dial WINS" if dm["cagr"] > em["cagr"] else "existing wins"
        print(f"\n[{name}]  target vol {em['vol']*100:.1f}%  ->  dial f = {f:.3f}")
        print(f"   existing   {_fmt(em)}")
        print(f"   dial(f={f:.2f}) {_fmt(dm)}")
        print(f"   at equal vol: dCAGR {(dm['cagr']-em['cagr'])*100:+5.1f}pp  "
              f"dSharpe {dm['sharpe']-em['sharpe']:+5.2f}  ->  {better} on return")

    # ── D. full f-frontier ────────────────────────────────────────────────────
    print("\n" + "=" * 88)
    print("D. FRACTIONAL-KELLY FRONTIER (base = growth-optimal blend)")
    print("=" * 88)
    print(f"   {'f':>5s}  {'CAGR':>7s} {'MDD':>8s} {'Sharpe':>7s} {'vol':>6s} {'Calmar':>7s}")
    for f in (1.0, 0.85, 0.70, 0.55, 0.45, 0.35, 0.25, 0.15, 0.10):
        m = frac_metrics(r_full, sata, f)
        print(f"   {f:5.2f}  {m['cagr']*100:6.1f}% {m['mdd']*100:7.2f}% "
              f"{m['sharpe']:7.2f} {m['vol']*100:5.1f}% {calmar(m):7.2f}")

    # ── E. period robustness at matched-DD points ─────────────────────────────
    print("\n" + "=" * 88)
    print("E. PERIOD ROBUSTNESS — matched-DD dial vs existing, per profile")
    print("=" * 88)
    for name in ("Balanced", "Growth", "Aggressive"):
        em = existing[name]
        f = solve_f(r_full, sata, "mdd", em["mdd"])
        w_ex = np.array([em["weights"][c] for c in cols])
        print(f"\n[{name}]  (dial f={f:.2f} matched to MDD {em['mdd']*100:.1f}%)")
        for lbl, s, e in oc.COMBINED_PERIODS:
            sub = rets.loc[pd.Timestamp(s):(pd.Timestamp(e) if e else None)]
            if len(sub) < 5:
                continue
            subpos = pos.loc[sub.index]
            ex_m = oc.curve_metrics(oc._equity(oc._combine(sub, w_ex, subpos, sata)))
            rf_sub = oc._combine(sub, w_star, subpos, sata)
            di_m = oc.curve_metrics(frac_curve(rf_sub, sata, f))
            print(f"   {lbl:26s}  existing ret {ex_m['total_ret']*100:8.1f}% MDD {ex_m['mdd']*100:6.1f}% Sh {ex_m['sharpe']:.2f}"
                  f"   |  dial ret {di_m['total_ret']*100:8.1f}% MDD {di_m['mdd']*100:6.1f}% Sh {di_m['sharpe']:.2f}")

    # ── F. tilt-ON parity note (default profile) ──────────────────────────────
    print("\n" + "=" * 88)
    print("F. TILT-ON CHECK (default Aggressive, as the app actually ships)")
    print("=" * 88)
    on = oc.optimize_weights(rets, caps=oc.caps_for("Aggressive"), pos=pos, sata_daily=sata,
                             mdd_floor=oc.RISK_PROFILES["Aggressive"]["mdd_floor"],
                             objective=CUR_OBJ["Aggressive"], fundamental=True)
    print(f"   app-as-ships (tilt ON)  {_fmt(on['optimal'])}")


if __name__ == "__main__":
    main()
