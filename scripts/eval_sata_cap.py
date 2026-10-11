"""Strategy Logic V2 — the SATA cap: uncapped vs capped replays.

Idle capital (what the signals leave undeployed) used to sit entirely in SATA —
Strive's variable-rate preferred, ~13 % coupon but a price that can fall. Under V2
it is parked in SATA only up to ``overall_core.SATA_CAP`` (30 %) of the book; the
excess stays in cash earning 0 %. SATA is the allocator's residual, so the cap moves
no sleeve weight: it only changes where the idle remainder sits and so how much of it
earns the coupon.

    python scripts/eval_sata_cap.py            # writes SATA_CAP_EVAL.md

Both profiles are replayed (walk-forward gated replay, V2, the production cluster
rules) two ways on the same universe, anchors and optimiser seeds:

  uncapped   sata_cap=None   — every idle dollar in SATA (what V2 did before)
  capped     sata_cap=0.30   — SATA ≤ 30 % of the book, the excess in cash (0 %)

and the cap-level sensitivity (50/40/30/20/10/0 %) is derived from the uncapped replay
(exact, since SATA is a residual; cross-checked against the real capped replay).
Windows: full replay, the pre-2025-07 selection window, the 2025-07 → hold-out, and
since 2026-03-01 (the P&L section's default window). Costs: 0 and 5 bp per unit of
weight moved (SATA leg included; cash is not traded). Three optimiser seeds (7 =
production). The back-test prices SATA at a steady coupon with no price risk, so it
CANNOT show what the cap protects against — the return cost is the premium.

Exit status 1 if an invariant fails (weights unchanged, weights + SATA + cash = 100 %,
the return gap equals the coupon on the cash leg, derived == real capped replay).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "app"))
sys.path.insert(0, str(_REPO))

import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
import overall_core as oc                # noqa: E402

PROFILES = ("Balanced", "Growth")
SEEDS = (7, 11, 23)
HOLD_START = pd.Timestamp("2025-07-01")
MAR1 = pd.Timestamp("2026-03-01")
WINDOWS = {
    "Full replay (2021 →)": (None, None),
    "Selection window (→ 2025-06-30)": (None, HOLD_START - pd.Timedelta(days=1)),
    "Hold-out (2025-07-01 →)": (HOLD_START, None),
    "Since 2026-03-01": (MAR1, None),
}
SENS_CAPS = (None, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0)
OUT_MD = _REPO / "SATA_CAP_EVAL.md"


def _metrics(r: pd.Series) -> np.ndarray:
    m = oc.curve_metrics((1 + r).cumprod())
    return np.array([m["total_ret"], m["mdd"], m["sharpe"]])


def _tv(w: pd.DataFrame, sata: pd.Series) -> pd.Series:
    return pd.concat([w, sata.rename("__s")], axis=1).diff().abs().sum(axis=1).fillna(0.0)


def _derive(unc: dict, cap):
    """Net-of-cash-yield return / SATA / cash series for ``cap`` derived from the
    uncapped replay (SATA is the residual, so this is exact)."""
    s = unc["sata"]
    sc = s if cap is None else s.clip(upper=cap)
    cash = s - sc
    biz = (unc["ret"].index.dayofweek < 5).astype(float)
    return unc["ret"] - cash * oc.SATA_DAILY * biz, sc, cash


def run() -> dict:
    print("Running the V2 universe (live data w/ CSV fallback)…")
    results = oc.run_universe("v2")
    if oc._LAST_ERRORS:
        for k, e in oc._LAST_ERRORS.items():
            print(f"  ⚠ {k}: {e}")
    cap = oc.sata_cap_for("v2")
    assert cap is not None, "the V2 SATA cap is not configured"
    rets = oc.returns_matrix(results)
    reps: dict[tuple, dict] = {}
    inv: list[str] = []
    for pname in PROFILES:
        prof = oc.RISK_PROFILES[pname]
        caps = oc.caps_for(pname)
        kw = dict(caps=caps, mdd_floor=prof["mdd_floor"], objective=prof["objective"],
                  version="v2", cluster_mix=oc.cluster_mix_for(pname, "v2"),
                  adds_only_cluster=oc.cluster_adds_band_for(pname, "v2"))
        for seed in SEEDS:
            print(f"  replay {pname} · seed {seed}…", flush=True)
            unc = oc.walkforward_gated_replay(
                results, seed=seed, sata_cap=None, caps=kw["caps"], mdd_floor=kw["mdd_floor"],
                objective=kw["objective"], version="v2", cluster_mix=kw["cluster_mix"],
                adds_only_cluster=kw["adds_only_cluster"])
            cpd = oc.walkforward_gated_replay(
                results, seed=seed, sata_cap=cap, caps=kw["caps"], mdd_floor=kw["mdd_floor"],
                objective=kw["objective"], version="v2", cluster_mix=kw["cluster_mix"],
                adds_only_cluster=kw["adds_only_cluster"])
            tag = f"{pname}/seed {seed}"
            if not unc["weights"].equals(cpd["weights"]):
                inv.append(f"{tag}: the cap moved a sleeve weight")
            tot = cpd["weights"].sum(axis=1) + cpd["sata"] + cpd["cash"]
            if not np.allclose(tot, 1.0):
                inv.append(f"{tag}: weights + SATA + cash != 100 %")
            if cpd["sata"].max() > cap + 1e-12:
                inv.append(f"{tag}: SATA above the cap")
            d_ret, d_sata, d_cash = _derive(unc, cap)
            if not np.allclose(d_ret.to_numpy(), cpd["ret"].to_numpy(), atol=1e-12):
                inv.append(f"{tag}: the real capped replay != the one derived from the uncapped replay")
            rl = {}
            for lab, rep in (("uncapped", unc), ("capped", cpd)):
                rl[lab] = oc.realized_pnl_by_asset(rep["weights"], rep["sata"], MAR1, rets,
                                                   active=rep.get("active"))
            reps[(pname, seed)] = dict(unc=dict(ret=unc["ret"], sata=unc["sata"], w=unc["weights"]),
                                       cpd=dict(ret=cpd["ret"], sata=cpd["sata"], cash=cpd["cash"],
                                                w=cpd["weights"]),
                                       turn=(unc["turnover"]["mean"], cpd["turnover"]["mean"]),
                                       crypto=rl)
    return dict(reps=reps, inv=inv, cap=cap,
                last_bar=max(v["unc"]["ret"].index[-1] for v in reps.values()))


def _table(reps, pname, cap, cost):
    out = {}
    for wname, (a, b) in WINDOWS.items():
        for lab in ("uncapped", "capped"):
            ms = []
            for s in SEEDS:
                v = reps[(pname, s)]
                d = v["unc"] if lab == "uncapped" else v["cpd"]
                tv = _tv(d["w"], d["sata"])
                ms.append(_metrics((d["ret"] - cost / 1e4 * tv).loc[a:b]))
            out[(wname, lab)] = np.mean(ms, axis=0)
    return out


def _sens(reps, pname):
    row = {}
    for c in SENS_CAPS:
        ms = []
        for s in SEEDS:
            d_ret, _, _ = _derive(reps[(pname, s)]["unc"], c)
            ms.append(_metrics(d_ret))
        row[c] = np.mean(ms, axis=0)
    return row


def _cell(m):
    return f"{m[0]*100:+,.0f}% / {m[1]*100:.1f}% / {m[2]:.2f}"


def write_md(out: dict) -> None:
    reps, cap = out["reps"], out["cap"]
    L = ["# SATA cap — uncapped vs capped replays (Strategy Logic V2)\n",
         f"*Generated by `scripts/eval_sata_cap.py` · data through **{out['last_bar'].date()}** · "
         f"walk-forward gated replay, V2 with the production cluster rules, 3 optimiser seeds "
         f"({', '.join(map(str, SEEDS))}; 7 = production) · cells are **total return / max drawdown / "
         f"Sharpe**.*\n",
         "## Question\n",
         "Whatever the signals leave undeployed used to sit entirely in **SATA** (Strive's "
         "variable-rate preferred: ~13 % coupon, but a preferred's price can fall) — up to 84 % "
         "of the book in the replay and 40 % in the published book at the time. What does it "
         f"cost to **limit SATA to {cap*100:.0f} % of the book and hold any excess in cash at 0 %**?\n",
         "## The rule\n",
         f"`overall_core.SATA_CAP = {cap}` (`sata_cap_for(version)`; V2 only, **both profiles**). The "
         "idle remainder splits (`split_idle`): SATA up to the cap, the excess in cash. SATA is the "
         "allocator's residual, so **no sleeve weight changes** — only where the idle capital sits, and so "
         "how much of it earns the coupon. It is applied everywhere the book is built or replayed: "
         "the live gate (`sata` / `cash` / `idle`), the closed-market freeze, the historical book, "
         "the walk-forward replay and Combined splice (`cash` series), the P&L section's attribution, "
         "trade log and realized-P&L maths (cash earns 0 %), the costs/taxes simulator (a separate 0 %-yield "
         "cash balance), the as-published replay (books are stamped with `sata_cap`; older books are "
         "unchanged), the health tracker and the published books: the **paper** book still holds all idle "
         "as cash, the **live** book parks `min(idle, cap)` in SATA and the excess in `cash_weight`. V1 has "
         "no cap; the Combined view applies it from the 2026-10-09 cut-over.\n",
         "## Results\n"]
    for pname in PROFILES:
        L.append(f"### {pname}\n")
        for cost in (0, 5):
            tab = _table(reps, pname, cap, cost)
            L.append(f"**{'No transaction costs' if cost == 0 else f'{cost} bp costs'}**\n")
            L.append("| Window | Uncapped (before) | **SATA ≤ 30 %** | Δ return |")
            L.append("|---|---|---|--:|")
            for wname in WINDOWS:
                u, c = tab[(wname, "uncapped")], tab[(wname, "capped")]
                L.append(f"| {wname} | {_cell(u)} | **{_cell(c)}** | {(c[0]-u[0])*100:+,.0f} pts |")
            L.append("")
        sens = _sens(reps, pname)
        L.append("Cap-level sensitivity (full replay, no costs, 3-seed mean): "
                 + " · ".join(f"{('no cap' if c is None else f'{int(c*100)}%')} {_cell(m)}"
                              for c, m in sens.items()) + "\n")
        v7 = reps[(pname, 7)]
        s0, s1, c1 = v7["unc"]["sata"], v7["cpd"]["sata"], v7["cpd"]["cash"]
        L.append(f"Production seed exposure: SATA averaged {s0.mean()*100:.1f} % of the book uncapped "
                 f"(max {s0.max()*100:.0f} %, above 30 % on {(s0 > cap).mean()*100:.0f} % of days); "
                 f"capped: SATA {s1.mean()*100:.1f} % · cash {c1.mean()*100:.1f} % on average "
                 f"(cash up to {c1.max()*100:.0f} % of the book; the average excess on the days it exists "
                 f"is {c1[c1 > 0].mean()*100:.1f} %).\n")
        L.append(f"Mean daily one-way turnover (3 seeds): uncapped "
                 f"{np.mean([reps[(pname, s)]['turn'][0] for s in SEEDS])*100:.2f} % · capped "
                 f"{np.mean([reps[(pname, s)]['turn'][1] for s in SEEDS])*100:.2f} % (cash is not traded).\n")
    L.append("## Reading it\n")
    L.append("* The cost is the lost coupon on the cash leg — about a 13 %/yr yield on an average 4 % of "
             "the book (24 % of days above the cap), concentrated in 2021–25 when the book was often "
             "mostly idle; recent windows lose only 1–2 points.\n"
             "* Max drawdown is unchanged on the full replay and 0.1–0.2 pt deeper in recent windows "
             "(losing the carry matters slightly more in a drawdown); Sharpe moves by 0.01–0.03.\n"
             "* Cost rises gradually as the cap tightens (sensitivity line above).\n"
             "* **What the back-test cannot show:** it prices SATA at a steady coupon with no price risk. "
             "The cap exists to bound that exposure, so the return cost above is the premium for an "
             "insurance whose payoff the replay does not model. Cash earns 0 % here by decision "
             "(a money-market yield would shrink the cost).\n"
             "* The optimiser that fits the quarterly anchors still assumes idle capital earns the SATA "
             "coupon (`_combine`), so anchors and the fixed-weight 'optimal blend' statistics ignore the "
             "cap; the published walk-forward replay, the live book and every P&L figure apply it.\n")
    L.append("## Invariants checked on every run\n")
    L.append("Sleeve weights identical capped vs uncapped · weights + SATA + cash = 100 % on every day · SATA never "
             "above the cap · the real capped replay equals the one derived from the uncapped replay "
             "(return gap = coupon on the cash leg, business days only). "
             + ("**PASS**.\n" if not out["inv"] else "**FAIL**:\n\n" + "\n".join(f"* {i}" for i in out["inv"]) + "\n"))
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {OUT_MD.name}")


def main() -> int:
    out = run()
    for pname in PROFILES:
        tab = _table(out["reps"], pname, out["cap"], 0)
        for wname in WINDOWS:
            u, c = tab[(wname, "uncapped")], tab[(wname, "capped")]
            print(f"{pname:9} {wname:<32} uncapped {_cell(u)}   capped {_cell(c)}")
    write_md(out)
    print("INVARIANTS:", "PASS" if not out["inv"] else "FAIL")
    for i in out["inv"]:
        print("  ✗", i)
    return 1 if out["inv"] else 0


if __name__ == "__main__":
    sys.exit(main())
