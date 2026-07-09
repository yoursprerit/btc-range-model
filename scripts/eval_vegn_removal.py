"""Does removing VEGN from the Overall combined strategy improve the back-test?

VEGN (US Vegan Climate ETF — ESG-screened, tech-tilted S&P 500 beta) is one of
the ten signal apps folded into the Overall Trading portfolio.  It already
carries the repo's lowest fundamental conviction (FUNDAMENTAL_VIEW["VEGN"]=0.60)
and earns only a sliver of the optimal blend.  This asks the direct question:
drop it from the traded universe entirely and let the SAME optimiser re-water-
fill the freed weight across the remaining instruments — does the combined
portfolio get better or worse?

Method (mirrors scripts/build_overall.py exactly):
  * run the live universe ONCE,
  * build the returns/position matrices with and without every VEGN sleeve,
  * re-optimise the cross-asset weights for each risk profile with the IDENTICAL
    optimize_weights() call the app/build script uses,
  * compare the optimal combined curve across all COMBINED_PERIODS.

Because VEGN's OOS start (2021) is not the binding one, dropping it does not move
the common back-test window — the two curves are directly comparable.

    python scripts/eval_vegn_removal.py
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

DROP = "VEGN"   # the app-key (parent) whose sleeves we remove


def optimise_profiles(results):
    """Return {profile: optimize_weights(...) dict} + the returns/pos matrices,
    using the exact same calls as scripts/build_overall.py."""
    rets = oc.returns_matrix(results)
    pos = oc.position_matrix(results, rets.index)
    profiles = {}
    for name, prof in oc.RISK_PROFILES.items():
        profiles[name] = oc.optimize_weights(
            rets, caps=oc.caps_for(name), pos=pos, sata_daily=oc.SATA_DAILY,
            mdd_floor=prof["mdd_floor"], objective=prof["objective"])
    return profiles, rets, pos


def combined_periods(rets, weights, pos):
    """Optimal combined curve metrics per COMBINED_PERIODS window."""
    w = np.array([weights.get(c, 0.0) for c in rets.columns])
    return oc.period_breakdown(rets, w, oc.COMBINED_PERIODS, pos=pos,
                               sata_daily=oc.SATA_DAILY)


def fmt(m):
    return (f"ret {m['total_ret']*100:8.1f}%  CAGR {m['cagr']*100:5.1f}%  "
            f"MDD {m['mdd']*100:6.2f}%  Sharpe {m['sharpe']:.3f}  "
            f"vol {m['vol']*100:5.2f}%")


# ── run the universe once ────────────────────────────────────────────────────
print("Running baseline universe (live fetch, ~30-90s)…", flush=True)
base = oc.run_universe()
print(f"  {len(base)} instruments: {[r['key'] for r in base]}")
if oc._LAST_ERRORS:
    print("  LOAD ERRORS:", oc._LAST_ERRORS)

# VEGN sleeves that will be removed (parent == 'VEGN')
vegn_keys = [r["key"] for r in base if r["parent"] == DROP]
no_vegn = [r for r in base if r["parent"] != DROP]
print(f"\nRemoving parent '{DROP}' → drops sleeves {vegn_keys}")
print(f"  universe: {len(base)} → {len(no_vegn)} instruments")

opt_base, rets_base, pos_base = optimise_profiles(base)
opt_nov, rets_nov, pos_nov = optimise_profiles(no_vegn)
print(f"\nReturn matrix WITH VEGN:    {rets_base.shape}  "
      f"{rets_base.index[0].date()} → {rets_base.index[-1].date()}")
print(f"Return matrix WITHOUT VEGN: {rets_nov.shape}  "
      f"{rets_nov.index[0].date()} → {rets_nov.index[-1].date()}")


# ── per-profile comparison (full OOS optimal blend) ──────────────────────────
print("\n" + "=" * 78)
print("OPTIMAL COMBINED BLEND — full OOS (2021 → now), by risk profile")
print("=" * 78)
for name in oc.RISK_PROFILES:
    b = opt_base[name]["optimal"]
    o = opt_nov[name]["optimal"]
    vw = b["weights"].get(DROP, 0.0)
    print(f"\n[{name}]  (VEGN weight WITH = {vw*100:.2f}%)")
    print(f"  WITH VEGN     {fmt(b)}")
    print(f"  WITHOUT VEGN  {fmt(o)}")
    print(f"  Δ (no-VEGN − with):  dRet {(o['total_ret']-b['total_ret'])*100:+6.2f}pp  "
          f"dCAGR {(o['cagr']-b['cagr'])*100:+5.2f}pp  "
          f"dMDD {(o['mdd']-b['mdd'])*100:+5.2f}pp  "
          f"dSharpe {o['sharpe']-b['sharpe']:+.3f}")


# ── period breakdown for the DEFAULT profile (what the app ships) ────────────
prof = oc.DEFAULT_PROFILE
print("\n" + "=" * 78)
print(f"PERIOD BREAKDOWN — default profile ({prof}), optimal blend")
print("=" * 78)
per_b = combined_periods(rets_base, opt_base[prof]["optimal"]["weights"], pos_base)
per_o = combined_periods(rets_nov, opt_nov[prof]["optimal"]["weights"], pos_nov)
by_o = {r["label"]: r for r in per_o}
for rb in per_b:
    ro = by_o.get(rb["label"])
    print(f"\n{rb['label']}")
    print(f"  WITH VEGN     {fmt(rb)}")
    if ro:
        print(f"  WITHOUT VEGN  {fmt(ro)}")
        print(f"  Δ  dRet {(ro['total_ret']-rb['total_ret'])*100:+6.2f}pp  "
              f"dMDD {(ro['mdd']-rb['mdd'])*100:+5.2f}pp  "
              f"dSharpe {ro['sharpe']-rb['sharpe']:+.3f}")


# ── profile-agnostic reference: equal-weight & risk-parity of the strategies ─
print("\n" + "=" * 78)
print("REFERENCE BLENDS (no optimiser tilt) — full OOS")
print("=" * 78)
for tag in ("equal", "risk_parity"):
    b = opt_base[prof][tag]
    o = opt_nov[prof][tag]
    print(f"\n[{tag}]")
    print(f"  WITH VEGN     {fmt(b)}")
    print(f"  WITHOUT VEGN  {fmt(o)}")
    print(f"  Δ  dRet {(o['total_ret']-b['total_ret'])*100:+6.2f}pp  "
          f"dMDD {(o['mdd']-b['mdd'])*100:+5.2f}pp  "
          f"dSharpe {o['sharpe']-b['sharpe']:+.3f}")
