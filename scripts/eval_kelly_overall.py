"""Kelly-criterion allocation — evaluation vs the current Overall objectives.

READ-ONLY EVALUATION.  This does NOT change the live strategy, the default
profile, or the committed weights.  It adds a growth-optimal (``kelly``)
objective to ``optimize_weights`` (see ``app/overall_core.py``) and back-tests
it head-to-head against the objective each risk profile currently uses:

    Balanced   → "balanced"    (highest return among near-max-Sharpe blends)
    Growth     → "max_return"  (highest return inside a -22% DD budget)
    Aggressive → "max_return"  (highest return inside a -38% DD budget)  [default]

Kelly is compared on the IDENTICAL per-asset return streams, caps, drawdown
budgets and SATA idle-cash assumption, so the only thing that varies is the
allocation rule.  We report:

  1. Full-Kelly vs the current objective per profile (fundamental tilt OFF, so
     the pure allocation effect is isolated; then tilt ON = as the app delivers).
  2. Fractional Kelly (1x / 1/2x / 1/4x) — the standard estimation-error haircut,
     modelled as f of the bankroll in the Kelly blend and (1-f) parked in SATA.
  3. Per-asset Kelly fractions  f* = W - (1-W)/R  from each sleeve's own trade
     record (diagnostic: what "size each bet by its edge" would prescribe).

Reproduce:  python scripts/eval_kelly_overall.py
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


def _fmt(m: dict) -> str:
    return (f"ret {m['total_ret']*100:8.1f}%  CAGR {m['cagr']*100:5.1f}%  "
            f"MDD {m['mdd']*100:6.2f}%  Sharpe {m['sharpe']:5.2f}  "
            f"vol {m['vol']*100:5.1f}%  g {m.get('growth',float('nan'))*100:5.1f}%")


def frac_kelly_metrics(rets, w_k, pos, sata_daily, f):
    """Fractional Kelly: f of the bankroll deployed to the full-Kelly blend, the
    remaining (1-f) parked in SATA cash, rebalanced daily."""
    full = oc._combine(rets, np.asarray(w_k, float), pos, sata_daily)
    daily = f * full + (1.0 - f) * sata_daily
    return oc.curve_metrics(oc._equity(daily))


def _trade_returns(r) -> np.ndarray:
    """Per-trade return array for a sleeve, normalising the two engine formats:
    the ticker/GLDM engines return a float ndarray of returns; the BTC CT engine
    returns a list of trade-log dicts (entry_price/exit_price)."""
    t = r["r"].get("trades")
    if t is None or len(t) == 0:
        return np.array([])
    if isinstance(t[0], dict):
        out = []
        for d in t:
            ep, xp = d.get("entry_price"), d.get("exit_price")
            if ep and xp:
                out.append(xp / ep - 1.0)
        return np.asarray(out, float)
    return np.asarray(t, float)


def per_asset_kelly(results):
    """Classic discrete Kelly fraction per sleeve from its own trade log:
    f* = W - (1-W)/R,  W = win rate, R = avg win / |avg loss|.  Diagnostic."""
    rows = []
    for r in results:
        trades = _trade_returns(r)
        if len(trades) < 5:
            continue
        wins = trades[trades > 0]; losses = trades[trades < 0]
        W = len(wins) / len(trades)
        avg_w = wins.mean() if len(wins) else 0.0
        avg_l = -losses.mean() if len(losses) else np.nan
        R = (avg_w / avg_l) if (avg_l and avg_l == avg_l and avg_l > 0) else np.nan
        f_star = (W - (1 - W) / R) if (R == R and R > 0) else np.nan
        rows.append(dict(key=r["key"], kind=r["kind"], n=len(trades), W=W,
                         R=R, f_star=f_star))
    return rows


def main():
    print("Running universe (live fetch, ~30-90s)…", flush=True)
    results = oc.run_universe()
    keys = [r["key"] for r in results]
    print(f"  {len(results)} instruments: {keys}")
    if oc._LAST_ERRORS:
        print(f"  load errors: {oc._LAST_ERRORS}")

    rets = oc.returns_matrix(results)
    pos = oc.position_matrix(results, rets.index)
    sata = oc.SATA_DAILY
    print(f"  return matrix {rets.shape}  {rets.index[0].date()} -> {rets.index[-1].date()}")

    # objective each profile currently uses
    CUR_OBJ = {"Balanced": "balanced", "Growth": "max_return", "Aggressive": "max_return"}

    # ── 1. head-to-head, fundamental tilt OFF (pure allocation effect) ────────
    print("\n" + "=" * 78)
    print("1. CURRENT OBJECTIVE  vs  KELLY  (tilt OFF — pure allocation rule)")
    print("=" * 78)
    kelly_weights = {}
    for name, prof in oc.RISK_PROFILES.items():
        caps = oc.caps_for(name)
        cur = oc.optimize_weights(rets, caps=caps, pos=pos, sata_daily=sata,
                                  mdd_floor=prof["mdd_floor"], objective=CUR_OBJ[name],
                                  fundamental=False)
        kel = oc.optimize_weights(rets, caps=caps, pos=pos, sata_daily=sata,
                                  mdd_floor=prof["mdd_floor"], objective="kelly",
                                  fundamental=False)
        kelly_weights[name] = kel["optimal"]["weights"]
        cm, km = cur["optimal"], kel["optimal"]
        d = "AGGRESSIVE (default)" if name == oc.DEFAULT_PROFILE else name
        print(f"\n[{d}]  caps core/beta/lev = "
              f"{prof['caps']['core']:.2f}/{prof['caps']['beta']:.2f}/{prof['caps']['lev']:.2f}"
              f"  mdd_floor {prof['mdd_floor']:.2f}")
        print(f"   current ({CUR_OBJ[name]:10s})  {_fmt(cm)}")
        print(f"   kelly    (max-log-growth) {_fmt(km)}")
        print(f"   Δ kelly-current:  dRet {(km['total_ret']-cm['total_ret'])*100:+7.1f}pp  "
              f"dCAGR {(km['cagr']-cm['cagr'])*100:+5.1f}pp  dMDD {(km['mdd']-cm['mdd'])*100:+6.2f}pp  "
              f"dSharpe {km['sharpe']-cm['sharpe']:+5.2f}  dGrowth {(km['growth']-cm['growth'])*100:+5.1f}pp")

    # ── 2. fractional Kelly (estimation-error haircut) ────────────────────────
    print("\n" + "=" * 78)
    print("2. FRACTIONAL KELLY  (f of bankroll in Kelly blend, 1-f in SATA cash)")
    print("=" * 78)
    for name in oc.RISK_PROFILES:
        w_k = np.array([kelly_weights[name][c] for c in rets.columns])
        print(f"\n[{name}]")
        for f in (1.0, 0.5, 0.25):
            m = frac_kelly_metrics(rets, w_k, pos, sata, f)
            tag = {1.0: "full  (1x )", 0.5: "half  (1/2)", 0.25: "quarter(1/4)"}[f]
            print(f"   {tag}  {_fmt(m)}")

    # ── 3. as-delivered (fundamental tilt ON) for the default profile ─────────
    print("\n" + "=" * 78)
    print(f"3. AS-DELIVERED (tilt ON) — default profile {oc.DEFAULT_PROFILE}")
    print("=" * 78)
    caps = oc.caps_for(oc.DEFAULT_PROFILE)
    fl = oc.RISK_PROFILES[oc.DEFAULT_PROFILE]["mdd_floor"]
    cur = oc.optimize_weights(rets, caps=caps, pos=pos, sata_daily=sata, mdd_floor=fl,
                              objective=CUR_OBJ[oc.DEFAULT_PROFILE], fundamental=True)
    kel = oc.optimize_weights(rets, caps=caps, pos=pos, sata_daily=sata, mdd_floor=fl,
                              objective="kelly", fundamental=True)
    print(f"   current (as app ships)  {_fmt(cur['optimal'])}")
    print(f"   kelly   (tilt ON)       {_fmt(kel['optimal'])}")

    # ── 4. period breakdown, default profile, tilt OFF ────────────────────────
    print("\n" + "=" * 78)
    print(f"4. PERIOD BREAKDOWN — default {oc.DEFAULT_PROFILE}, tilt OFF")
    print("=" * 78)
    cur_w = np.array([oc.optimize_weights(rets, caps=caps, pos=pos, sata_daily=sata,
                                          mdd_floor=fl, objective=CUR_OBJ[oc.DEFAULT_PROFILE],
                                          fundamental=False)["optimal"]["weights"][c]
                      for c in rets.columns])
    kel_w = np.array([kelly_weights[oc.DEFAULT_PROFILE][c] for c in rets.columns])
    per_cur = oc.period_breakdown(rets, cur_w, oc.COMBINED_PERIODS, pos=pos, sata_daily=sata)
    per_kel = oc.period_breakdown(rets, kel_w, oc.COMBINED_PERIODS, pos=pos, sata_daily=sata)
    for pc, pk in zip(per_cur, per_kel):
        print(f"\n   {pc['label']}")
        print(f"      current  ret {pc['total_ret']*100:8.1f}%  MDD {pc['mdd']*100:6.1f}%  Sharpe {pc['sharpe']:.2f}")
        print(f"      kelly    ret {pk['total_ret']*100:8.1f}%  MDD {pk['mdd']*100:6.1f}%  Sharpe {pk['sharpe']:.2f}")

    # ── 5. weight diff (default profile, tilt OFF) ────────────────────────────
    print("\n" + "=" * 78)
    print(f"5. WEIGHT DIFFERENCE — default {oc.DEFAULT_PROFILE}, tilt OFF (%)")
    print("=" * 78)
    print(f"   {'asset':6s} {'current':>8s} {'kelly':>8s} {'Δ':>8s}")
    for c in rets.columns:
        cw = dict(zip(rets.columns, cur_w))[c] * 100
        kw = kelly_weights[oc.DEFAULT_PROFILE][c] * 100
        if cw > 0.05 or kw > 0.05:
            print(f"   {c:6s} {cw:7.1f}% {kw:7.1f}% {kw-cw:+7.1f}%")

    # ── 6. per-asset classic Kelly fractions (diagnostic) ─────────────────────
    print("\n" + "=" * 78)
    print("6. PER-ASSET KELLY FRACTION  f* = W - (1-W)/R  (from each sleeve's trades)")
    print("=" * 78)
    print(f"   {'asset':6s} {'kind':5s} {'trades':>6s} {'winrate':>8s} {'payoff R':>9s} {'f*':>8s}")
    for r in per_asset_kelly(results):
        fs = f"{r['f_star']*100:6.1f}%" if r["f_star"] == r["f_star"] else "   n/a"
        R = f"{r['R']:.2f}" if r["R"] == r["R"] else "n/a"
        print(f"   {r['key']:6s} {r['kind']:5s} {r['n']:6d} {r['W']*100:7.1f}% {R:>9s} {fs:>8s}")


if __name__ == "__main__":
    main()
