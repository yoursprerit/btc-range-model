"""Would a portfolio-level emergency kill switch help the Overall book?

The per-sleeve engines already exit on their own signals and stops, and idle
weight parks in SATA — but nothing looks at the PORTFOLIO. This sweeps the
candidate circuit breakers that would, each applied as a strictly causal overlay
on the published walk-forward gated replay (decision taken at the close of bar
t-1, effective from bar t; while halted the book earns SATA on business days,
exactly as `replay_gated_allocation` treats undeployed weight):

  A  trailing-drawdown breaker   book DD from peak ≤ −X% → flat for N days
  B  same, held until the drawdown recovers (no peak reset)
  C  single-day crash breaker    one-day book return ≤ −X% → flat for N days
  D  parent-signal concentration cap   no single parent app above X% of NAV,
                                       the excess to SATA (a permanent control,
                                       not an event trigger — the comparison
                                       the kill-switch question really needs)
  E  volatility target           scale the book by target / realised 20d vol

Every variant is scored against the unmodified replay on the Balanced and
Growth profiles, plus a sub-period split, so a single lucky parameter can be
told from a real effect.

    python scripts/eval_kill_switch.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "app"))
sys.path.insert(0, str(_REPO))

import overall_core as oc           # noqa: E402

SATA_DAILY = oc.SATA_DAILY
PROFILES = ("Balanced", "Growth")
SUBPERIODS = [("2021-22", "2021-01-01", "2022-12-31"),
              ("2023-24", "2023-01-01", "2024-12-31"),
              ("2025-26", "2025-01-01", None)]


# ── scoring ──────────────────────────────────────────────────────────────
def metrics(ret: pd.Series) -> dict:
    eq = (1 + ret).cumprod()
    dd = eq / eq.cummax() - 1
    days = (ret.index[-1] - ret.index[0]).days
    return dict(total=eq.iloc[-1] - 1,
                cagr=eq.iloc[-1] ** (365.25 / days) - 1,
                mdd=dd.min(),
                sharpe=ret.mean() / ret.std() * np.sqrt(252) if ret.std() else 0.0,
                vol=ret.std() * np.sqrt(252))


def row(name: str, m: dict, base: dict | None = None, tail: str = "") -> str:
    d = ""
    if base:
        d = (f"  Δret {(m['total']-base['total'])*100:+7.0f}pp"
             f"  ΔMDD {(m['mdd']-base['mdd'])*100:+6.1f}pp"
             f"  ΔSharpe {m['sharpe']-base['sharpe']:+5.2f}")
    return (f"{name:<30s} ret {m['total']*100:7.0f}%  CAGR {m['cagr']*100:5.1f}%  "
            f"MDD {m['mdd']*100:6.1f}%  Sharpe {m['sharpe']:5.2f}  "
            f"vol {m['vol']*100:4.1f}%{d}{tail}")


# ── the breaker simulator ────────────────────────────────────────────────
def simulate(R: pd.Series, rule, cooldown: int = 5, size: float = 0.0,
             reset_peak: bool = True) -> tuple[pd.Series, dict]:
    """`rule(state) -> bool` is evaluated at the close of the current bar and
    halts the NEXT `cooldown` bars (exposure `size`, remainder in SATA).  State
    carries only information dated on/before that close, so the overlay is as
    look-ahead-free as the replay it sits on."""
    idx = R.index
    biz = (idx.dayofweek < 5).astype(float)
    r = R.to_numpy(float)
    out = np.zeros(len(idx))
    cur = np.zeros(len(idx))
    eq = peak = 1.0
    halted_until = -1
    halts = days_out = 0
    for t in range(len(idx)):
        mult = size if t <= halted_until else 1.0
        days_out += mult < 1.0
        out[t] = mult * r[t] + (1 - mult) * SATA_DAILY * biz[t]
        eq *= 1 + out[t]
        cur[t] = eq
        if t > halted_until:
            peak = max(peak, eq)
            if rule(dict(t=t, eq=eq, peak=peak, cur=cur, r=r, idx=idx)):
                halted_until = t + cooldown
                halts += 1
                if reset_peak:
                    peak = eq
    return pd.Series(out, index=idx), dict(halts=halts,
                                           pct_out=days_out / len(idx))


def dd_rule(th):
    return lambda st: st["eq"] / st["peak"] - 1 <= -th


def day_rule(th):
    return lambda st: st["r"][st["t"]] <= -th


# ── D: parent concentration cap (a weight overlay, not a trigger) ────────
def parent_capped(W: pd.DataFrame, rets: pd.DataFrame, parent: dict,
                  cap: float) -> pd.Series:
    biz = pd.Series((W.index.dayofweek < 5).astype(float), index=W.index)
    pcol = pd.Series({c: parent[c] for c in W.columns})
    pw = W.T.groupby(pcol).sum().T
    scale = (cap / pw.replace(0, np.nan)).clip(upper=1.0).fillna(1.0)
    Wx = pd.DataFrame({c: W[c] * scale[parent[c]] for c in W.columns},
                      index=W.index)
    idle = (1 - Wx.sum(axis=1)).clip(lower=0)
    ret = pd.Series((Wx.values * rets.values).sum(axis=1), index=W.index)
    return ret + idle * SATA_DAILY * biz, idle.mean()


def main():
    print("Running universe (live fetch, ~30-90s)…", flush=True)
    res = oc.run_universe()
    parent = {r["key"]: r["parent"] for r in res}
    rets_full = oc.returns_matrix(res)

    reps = {}
    for prof in PROFILES:
        p = oc.RISK_PROFILES[prof]
        print(f"Replaying {prof}…", flush=True)
        reps[prof] = oc.walkforward_gated_replay(
            res, caps=oc.caps_for(prof), mdd_floor=p["mdd_floor"],
            objective=p["objective"])

    # ── how much cash does the book already hold? ────────────────────────
    rep = reps["Balanced"]
    W, sata = rep["weights"], rep["sata"]
    pcol = pd.Series({c: parent[c] for c in W.columns})
    pw = W.T.groupby(pcol).sum().T
    print("\n" + "=" * 100)
    print("0. Does the book ever de-risk on its own?  (Balanced replay)")
    print("=" * 100)
    print(f"  SATA weight        mean {sata.mean()*100:4.1f}%   median "
          f"{sata.median()*100:4.1f}%   days above 50%: "
          f"{(sata > 0.5).sum():d} of {len(sata):d}")
    print(f"  deployed fraction  mean {W.sum(axis=1).mean():.2f}   min "
          f"{W.sum(axis=1).min():.2f}   days below 0.80: "
          f"{(W.sum(axis=1) < 0.80).sum():d}")
    print(f"  parents funded     mean {(pw > 1e-3).sum(axis=1).mean():.1f}   "
          f"min {(pw > 1e-3).sum(axis=1).min():d}")
    print(f"  largest parent     mean {pw.max(axis=1).mean()*100:4.1f}%   "
          f"max {pw.max(axis=1).max()*100:4.1f}%")

    for prof in PROFILES:
        R = reps[prof]["ret"]
        base = metrics(R)
        print("\n" + "=" * 100)
        print(f"{prof} — baseline: " + row("", base).strip())
        print("=" * 100)

        print("\nA. trailing-drawdown breaker (peak re-armed on re-entry)")
        for th in (0.08, 0.10, 0.12, 0.15, 0.20):
            for cd in (5, 10, 21):
                r, info = simulate(R, dd_rule(th), cooldown=cd)
                print("  " + row(f"DD ≤ −{th*100:.0f}%, flat {cd}d", metrics(r), base,
                                 f"  halts {info['halts']:3d}"
                                 f"  out {info['pct_out']*100:4.1f}%"))

        print("\nB. drawdown breaker held until the drawdown recovers")
        for th in (0.10, 0.15, 0.20):
            r, info = simulate(R, dd_rule(th), cooldown=10, reset_peak=False)
            print("  " + row(f"DD ≤ −{th*100:.0f}% until recovery", metrics(r), base,
                             f"  halts {info['halts']:3d}"
                             f"  out {info['pct_out']*100:4.1f}%"))

        print("\nC. single-day crash breaker")
        for th in (0.03, 0.04, 0.05, 0.06):
            for cd in (3, 5, 10, 21):
                r, info = simulate(R, day_rule(th), cooldown=cd)
                print("  " + row(f"day ≤ −{th*100:.0f}%, flat {cd}d", metrics(r), base,
                                 f"  halts {info['halts']:3d}"
                                 f"  out {info['pct_out']*100:4.1f}%"))

        # grid-wide robustness — a real effect survives most of its neighbourhood
        deltas = []
        for th in (0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25):
            for cd in (3, 5, 10, 15, 21, 42):
                deltas.append(metrics(simulate(R, dd_rule(th), cooldown=cd)[0]))
        for th in (0.03, 0.035, 0.04, 0.045, 0.05, 0.06, 0.07):
            for cd in (3, 5, 10, 15, 21, 42):
                deltas.append(metrics(simulate(R, day_rule(th), cooldown=cd)[0]))
        ds = np.array([m["sharpe"] - base["sharpe"] for m in deltas])
        dr = np.array([m["total"] - base["total"] for m in deltas])
        dm = np.array([m["mdd"] - base["mdd"] for m in deltas])
        print(f"\n  Across all {len(deltas)} breaker configs (A + C grids):")
        print(f"    Sharpe better in {(ds > 0).mean()*100:3.0f}%  "
              f"(median {np.median(ds):+.2f})")
        print(f"    Return better in {(dr > 0).mean()*100:3.0f}%  "
              f"(median {np.median(dr)*100:+.0f}pp)")
        print(f"    MDD    better in {(dm > 0).mean()*100:3.0f}%  "
              f"(median {np.median(dm)*100:+.1f}pp)")
        print(f"    Sharpe AND return better: {((ds > 0) & (dr > 0)).mean()*100:3.0f}%")

        Wp = reps[prof]["weights"]
        rmat = rets_full.reindex(Wp.index)[Wp.columns].fillna(0.0)
        print("\nD. parent-signal concentration cap (excess → SATA)")
        for cap in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
            r, idle = parent_capped(Wp, rmat, parent, cap)
            print("  " + row(f"parent cap {cap*100:.0f}%", metrics(r), base,
                             f"  mean cash {idle*100:4.1f}%"))

        print("\nE. volatility target (20d realised, lagged one bar)")
        rv = R.rolling(20).std().shift(1) * np.sqrt(252)
        biz = pd.Series((R.index.dayofweek < 5).astype(float), index=R.index)
        for tgt in (0.15, 0.20, 0.25, 0.30):
            m = (tgt / rv).clip(upper=1.0).fillna(1.0)
            r = m * R + (1 - m) * SATA_DAILY * biz
            print("  " + row(f"vol target {tgt*100:.0f}%", metrics(r), base,
                             f"  mean exposure {m.mean():.2f}"))

    # ── sub-period read on the two candidates worth a second look ────────
    print("\n" + "=" * 100)
    print("Sub-period stability — the only two overlays that beat the baseline")
    print("=" * 100)
    for prof in PROFILES:
        R = reps[prof]["ret"]
        Wp = reps[prof]["weights"]
        rmat = rets_full.reindex(Wp.index)[Wp.columns].fillna(0.0)
        cand = {"day ≤ −5%, flat 5d": simulate(R, day_rule(0.05), cooldown=5)[0],
                "parent cap 35%": parent_capped(Wp, rmat, parent, 0.35)[0]}
        for name, series in cand.items():
            print(f"\n  [{prof}] {name}")
            for lbl, s, e in SUBPERIODS:
                b, o = metrics(R.loc[s:e]), metrics(series.loc[s:e])
                print(f"    {lbl:8s} base ret {b['total']*100:7.0f}%  "
                      f"MDD {b['mdd']*100:6.1f}%  Sharpe {b['sharpe']:5.2f}"
                      f"   →   ret {o['total']*100:7.0f}%  "
                      f"MDD {o['mdd']*100:6.1f}%  Sharpe {o['sharpe']:5.2f}")

    # ── what the single-day breaker actually skipped ─────────────────────
    print("\n" + "=" * 100)
    print("Event log — every bar the −5% single-day breaker would have fired on")
    print("=" * 100)
    R = reps["Balanced"]["ret"]
    r, idx = R.to_numpy(float), R.index
    halted_until = -1
    skipped = []
    print(f"  {'date':<12s} {'trigger':>9s} {'next 5 bars':>13s} {'saved':>8s}")
    for t in range(len(idx)):
        if t > halted_until and r[t] <= -0.05:
            halted_until = t + 5
            fwd = float(np.prod(1 + r[t + 1:min(t + 6, len(idx))]) - 1)
            skipped.append(fwd)
            print(f"  {str(idx[t].date()):<12s} {r[t]*100:8.2f}% "
                  f"{fwd*100:12.2f}% {-fwd*100:+7.2f}pp")
    sk = np.array(skipped)
    t_stat = sk.mean() / (sk.std(ddof=1) / np.sqrt(len(sk))) if len(sk) > 1 else 0.0
    print(f"\n  n = {len(sk)} events in {(idx[-1]-idx[0]).days/365.25:.1f} years; "
          f"mean saved {-sk.mean()*100:+.2f}pp per event, "
          f"t = {-t_stat:+.2f} — below the 95% bar, and one event "
          f"({-sk.min()*100:.1f}pp) is {-sk.min()/-sk.sum()*100:.0f}% of the total")


if __name__ == "__main__":
    main()
