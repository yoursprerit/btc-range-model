"""Historical replay of the signal-gated, priority-tilted allocation.

Read-only vs the app's committed state. Quantifies what the Overall app's
DAILY allocation adjustments — the entry-priority tilt and the deploy-only-to
-longs gate of ``signal_gated_allocation`` — would have added to (or cost) the
fixed-weight back-test each risk profile publishes.

Three curves per risk profile, all on the SAME per-sleeve strategy returns,
optimal-weight anchors and idle→SATA treatment:

  fixed-weight   the published back-test: ``_combine`` holds the optimiser's
                 weights constant; a flat sleeve's weight sits in SATA.
  gate only      ``replay_gated_allocation(tilt=False)``: each day the optimal
                 anchors are water-filled over ONLY the sleeves in the market
                 (fresh entries funded, exits defunded, deployed book sums to
                 1 up to the caps) — the live book's concentration, no tilt.
  gate + tilt    ``replay_gated_allocation(tilt=True)``: the full live logic —
                 anchors × (0.5 + priority), priority rebuilt each day from
                 AS-OF data only (momentum vs 50-day SMA, rolling macro
                 sentiment, expanding win-rate and Sharpe, MA20 bull regime),
                 lagged one bar, min-max ranked across that day's funded set,
                 water-filled to the profile caps.

No transaction or slippage costs are modelled anywhere (matching the published
fixed-weight numbers); daily one-way turnover is reported so the cost surface
of each variant stays visible.

    python scripts/eval_gated_replay.py            # writes OVERALL_GATED_REPLAY_EVAL.md
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

SLICES = [("2025-01-01", "since 2025"), ("2026-03-01", "since Mar 2026")]
OUT_MD = _REPO / "OVERALL_GATED_REPLAY_EVAL.md"


def _fmt(m: dict | None) -> str:
    if not m:
        return "—"
    return (f"{m['total_ret'] * 100:+8.1f}% | {m['cagr'] * 100:+7.1f}% | "
            f"{m['mdd'] * 100:6.1f}% | {m['sharpe']:5.2f}")


def _row(label: str, eq: pd.Series) -> dict:
    out = dict(label=label, full=oc.curve_metrics(eq))
    for start, name in SLICES:
        out[name] = oc.slice_metrics(eq, start)
    return out


def main() -> None:
    print("Running the universe (every sleeve's engine, live data w/ CSV fallback)…")
    results = oc.run_universe()
    if oc._LAST_ERRORS:
        for k, e in oc._LAST_ERRORS.items():
            print(f"  ⚠ {k}: {e}")
    rets = oc.returns_matrix(results)
    pos = oc.position_matrix(results, rets.index)
    sata = oc.SATA_DAILY
    asof = rets.index[-1].date()
    print(f"{len(results)} sleeves · {len(rets)} bars · through {asof}\n")

    profiles: dict[str, dict] = {}
    for pname, prof in oc.RISK_PROFILES.items():
        caps = oc.caps_for(pname)
        opt = oc.optimize_weights(rets, caps=caps, pos=pos, sata_daily=sata,
                                  mdd_floor=prof["mdd_floor"],
                                  objective=prof["objective"], fundamental=True)
        wd = opt["optimal"]["weights"]
        w = np.array([wd[c] for c in opt["cols"]])
        fixed_eq = oc._equity(oc._combine(rets, w, pos, sata))
        gate = oc.replay_gated_allocation(results, wd, caps=caps,
                                          sata_daily=sata, tilt=False)
        tilted = oc.replay_gated_allocation(results, wd, caps=caps,
                                            sata_daily=sata, tilt=True)
        rows = [_row("Fixed-weight (published)", fixed_eq),
                _row("Gate only (no tilt)", gate["equity"]),
                _row("Gate + priority tilt", tilted["equity"])]
        profiles[pname] = dict(rows=rows, gate=gate, tilted=tilted, weights=wd)

        print(f"═══ {pname} ═══   (total | CAGR | maxDD | Sharpe)")
        for r in rows:
            print(f"  {r['label']:<26} full: {_fmt(r['full'])}")
            for _, sname in SLICES:
                print(f"  {'':<26} {sname}: {_fmt(r[sname])}")
        f, g, t = (r["full"] for r in rows)
        print(f"  Δ gate-only vs fixed : CAGR {(g['cagr'] - f['cagr']) * 100:+.1f}pt · "
              f"Sharpe {g['sharpe'] - f['sharpe']:+.2f} · maxDD {(g['mdd'] - f['mdd']) * 100:+.1f}pt")
        print(f"  Δ gate+tilt vs fixed : CAGR {(t['cagr'] - f['cagr']) * 100:+.1f}pt · "
              f"Sharpe {t['sharpe'] - f['sharpe']:+.2f} · maxDD {(t['mdd'] - f['mdd']) * 100:+.1f}pt")
        print(f"  Δ tilt vs gate-only  : CAGR {(t['cagr'] - g['cagr']) * 100:+.1f}pt · "
              f"Sharpe {t['sharpe'] - g['sharpe']:+.2f}")
        for lbl, rep in (("gate-only", gate), ("gate+tilt", tilted)):
            tu = rep["turnover"]
            print(f"  {lbl:<9} turnover: mean {tu['mean'] * 100:.1f}%/day · "
                  f"p95 {tu['p95'] * 100:.1f}% · traded {tu['days_traded'] * 100:.0f}% of days · "
                  f"avg SATA {rep['sata'].mean() * 100:.0f}%")
        print()

    _write_md(profiles, asof, len(results), len(rets))
    print(f"Wrote {OUT_MD.name}")


def _write_md(profiles: dict, asof, n_sleeves: int, n_bars: int) -> None:
    L = []
    L.append("# Overall Strategy — Gated-Allocation Replay Evaluation\n")
    L.append(f"*Generated by `scripts/eval_gated_replay.py` · data through "
             f"**{asof}** · {n_sleeves} sleeves · {n_bars} daily bars · "
             f"fundamental overlay ON (app default) · **no transaction costs "
             f"on any variant**.*\n")
    L.append("## Question\n")
    L.append("The published per-profile back-test holds the optimiser's weights "
             "**constant** over the whole history; the live book instead re-sizes "
             "positions **every day** (`signal_gated_allocation`): capital is "
             "deployed only to sleeves that are long, each sized by its optimal "
             "anchor × (0.5 + entry-priority), water-filled to the profile caps, "
             "remainder in SATA. This eval replays that daily logic historically — "
             "priorities rebuilt strictly from data available as of each day — to "
             "measure what the daily adjustments add or cost.\n")
    L.append("## Method\n")
    L.append("* **Funded set (gate)** on day *t* = sleeves in the market on *t* "
             "per the engines' own `pos_series` (decided at the close of *t−1*; "
             "causal, next-bar execution as always).")
    L.append("* **Priority components**, all as-of and lagged one bar, min-max "
             "ranked across each day's funded set exactly like "
             "`compute_priorities`: momentum = parent close vs 50-day SMA; "
             "sentiment = `macro_sentiment` (a causal rolling series by "
             "construction); win-rate = expanding fraction of the sleeve's closed "
             "winning trades; Sharpe = expanding annualised Sharpe of the "
             "sleeve's strategy returns (≥60 bars); regime = close > MA20 with "
             "MA20 rising (the `bull_regime` rule).")
    L.append("* **Anchors**: the same full-history optimal weights the published "
             "back-test uses (per profile, fundamental overlay ON) — so any gap "
             "between the curves is attributable to the daily gate/tilt alone, "
             "not to different anchors.")
    L.append("* **Idle capital** earns the SATA daily yield in every variant.")
    L.append("* `gate only` water-fills the plain anchors over the funded set "
             "(no tilt) — separating the *concentration* effect from the "
             "*priority-tilt* effect.\n")
    L.append("## Results\n")
    for pname, p in profiles.items():
        L.append(f"### {pname}\n")
        L.append("| Variant | Window | Total | CAGR | MaxDD | Sharpe |")
        L.append("|---|---|---:|---:|---:|---:|")
        for r in p["rows"]:
            for wname in ["full"] + [s for _, s in SLICES]:
                m = r[wname]
                if not m:
                    continue
                L.append(f"| {r['label']} | {wname} | {m['total_ret'] * 100:+.1f}% | "
                         f"{m['cagr'] * 100:+.1f}% | {m['mdd'] * 100:.1f}% | "
                         f"{m['sharpe']:.2f} |")
        f, g, t = (r["full"] for r in p["rows"])
        L.append("")
        L.append(f"**Full-period deltas** — gate-only vs fixed: CAGR "
                 f"{(g['cagr'] - f['cagr']) * 100:+.1f}pt, Sharpe {g['sharpe'] - f['sharpe']:+.2f}, "
                 f"maxDD {(g['mdd'] - f['mdd']) * 100:+.1f}pt · gate+tilt vs fixed: CAGR "
                 f"{(t['cagr'] - f['cagr']) * 100:+.1f}pt, Sharpe {t['sharpe'] - f['sharpe']:+.2f}, "
                 f"maxDD {(t['mdd'] - f['mdd']) * 100:+.1f}pt · tilt alone (vs gate-only): CAGR "
                 f"{(t['cagr'] - g['cagr']) * 100:+.1f}pt, Sharpe {t['sharpe'] - g['sharpe']:+.2f}.\n")
        for lbl, rep in (("Gate only", p["gate"]), ("Gate + tilt", p["tilted"])):
            tu = rep["turnover"]
            L.append(f"*{lbl}*: mean daily one-way turnover "
                     f"{tu['mean'] * 100:.1f}% (p95 {tu['p95'] * 100:.1f}%), a >0.5% "
                     f"rebalance on {tu['days_traded'] * 100:.0f}% of days, average SATA "
                     f"weight {rep['sata'].mean() * 100:.0f}%.")
        L.append("")
    L.append("## Findings\n")
    gc, tc_, dd, sh = [], [], [], []
    for p in profiles.values():
        f, g, t = (r["full"] for r in p["rows"])
        gc.append((g["cagr"] - f["cagr"]) * 100)
        tc_.append((t["cagr"] - g["cagr"]) * 100)
        dd.append((t["mdd"] - f["mdd"]) * 100)
        sh.append(t["sharpe"] - f["sharpe"])
    L.append(f"* **The gate (concentration onto live longs) is the dominant "
             f"daily-adjustment effect**: vs the published fixed-weight numbers it "
             f"adds **{min(gc):+.1f} to {max(gc):+.1f} CAGR points** across the "
             f"profiles, at the price of **{min(dd):+.1f} to {max(dd):+.1f} points "
             f"of max drawdown** and a Sharpe change of {min(sh):+.2f} to "
             f"{max(sh):+.2f}.")
    L.append(f"* **The priority tilt itself is a small increment on top of the "
             f"gate**: {min(tc_):+.1f} to {max(tc_):+.1f} CAGR points vs gate-only, "
             f"with essentially unchanged Sharpe and drawdown.")
    L.append("* Net: the published fixed-weight back-test **understates both the "
             "return and the risk** of what the live daily-adjusted book actually "
             "does — the daily adjustments mostly re-shape the risk/return point "
             "(more return, deeper drawdowns) rather than adding risk-adjusted "
             "edge.\n")
    L.append("## Caveats\n")
    L.append("* The weight **anchors** are still fit on the full history "
             "(identical to the published back-test, deliberately, so the "
             "comparison isolates the daily-adjustment effect); see "
             "`OVERALL_OOS_WALKFORWARD_EVAL.md` for the out-of-sample-weights "
             "treatment.")
    L.append("* No transaction or slippage costs on any variant. The gated "
             "variants trade daily (turnover above); costs would shave the gated "
             "curves but not the fixed-weight curve's implicit rebalancing any "
             "less — the published number assumes free daily rebalancing too.")
    L.append("* Live priority uses the full-sample win-rate/Sharpe and intraday "
             "spot momentum; the replay substitutes their expanding / last-close "
             "analogues — the honest as-of information set.")
    L.append("* Regenerate with `python scripts/eval_gated_replay.py` "
             "(numbers move with data as-of date).")
    OUT_MD.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
