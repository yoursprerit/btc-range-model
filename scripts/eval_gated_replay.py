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
  walk-forward   ``walkforward_gated_replay``: gate + tilt with the LAST
  (published)    look-ahead removed — anchor weights re-fit each Jan 1 on
                 prior data only (equal-weight warm-up, fundamental overlay
                 off).  THE NUMBERS THE APP PUBLISHES; verified prefix-
                 invariant by ``scripts/check_lookahead.py``.

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
        wfrep = oc.walkforward_gated_replay(results, caps=caps,
                                            mdd_floor=prof["mdd_floor"],
                                            objective=prof["objective"],
                                            sata_daily=sata, tilt=True)
        rows = [_row("Fixed-weight (in-sample anchors)", fixed_eq),
                _row("Gate only (no tilt)", gate["equity"]),
                _row("Gate + priority tilt", tilted["equity"]),
                _row("Walk-forward gate + tilt (published)", wfrep["equity"])]
        profiles[pname] = dict(rows=rows, gate=gate, tilted=tilted, wf=wfrep,
                               weights=wd)

        print(f"═══ {pname} ═══   (total | CAGR | maxDD | Sharpe)")
        for r in rows:
            print(f"  {r['label']:<26} full: {_fmt(r['full'])}")
            for _, sname in SLICES:
                print(f"  {'':<26} {sname}: {_fmt(r[sname])}")
        f, g, t, wfm = (r["full"] for r in rows)
        print(f"  Δ gate-only vs fixed : CAGR {(g['cagr'] - f['cagr']) * 100:+.1f}pt · "
              f"Sharpe {g['sharpe'] - f['sharpe']:+.2f} · maxDD {(g['mdd'] - f['mdd']) * 100:+.1f}pt")
        print(f"  Δ gate+tilt vs fixed : CAGR {(t['cagr'] - f['cagr']) * 100:+.1f}pt · "
              f"Sharpe {t['sharpe'] - f['sharpe']:+.2f} · maxDD {(t['mdd'] - f['mdd']) * 100:+.1f}pt")
        print(f"  Δ tilt vs gate-only  : CAGR {(t['cagr'] - g['cagr']) * 100:+.1f}pt · "
              f"Sharpe {t['sharpe'] - g['sharpe']:+.2f}")
        print(f"  Δ walk-forward anchors (removes the weight look-ahead): CAGR "
              f"{(wfm['cagr'] - t['cagr']) * 100:+.1f}pt vs in-sample-anchored gate+tilt")
        for lbl, rep in (("gate-only", gate), ("gate+tilt", tilted),
                         ("walk-fwd", wfrep)):
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
             f"fundamental overlay ON in the in-sample-anchored variants, OFF "
             f"in the walk-forward (published) variant · **no transaction "
             f"costs on any variant**.*\n")
    L.append("## Question\n")
    L.append("The app's per-profile back-test PREVIOUSLY held the optimiser's "
             "full-history weights **constant** over the whole history, while the "
             "live book re-sizes positions **every day** "
             "(`signal_gated_allocation`): capital deployed only to sleeves that "
             "are long, each sized by its anchor × (0.5 + entry-priority), "
             "water-filled to the profile caps, remainder in SATA. The app now "
             "**publishes the walk-forward gated replay** instead. This eval "
             "decomposes the difference layer by layer — gate, tilt, then "
             "anchor look-ahead removal.\n")
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
    L.append("* **Anchors**: the first three variants deliberately share the "
             "full-history optimal weights (per profile, fundamental overlay ON) "
             "so their gaps isolate the daily gate/tilt effects; the "
             "**walk-forward row re-fits anchors each Jan 1 on prior data only** "
             "(equal-weight warm-up, overlay OFF) — the app's published "
             "construction, prefix-invariance-verified by "
             "`scripts/check_lookahead.py`.")
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
        f, g, t, wfm = (r["full"] for r in p["rows"])
        L.append("")
        L.append(f"**Full-period deltas** — gate-only vs fixed: CAGR "
                 f"{(g['cagr'] - f['cagr']) * 100:+.1f}pt, Sharpe {g['sharpe'] - f['sharpe']:+.2f}, "
                 f"maxDD {(g['mdd'] - f['mdd']) * 100:+.1f}pt · gate+tilt vs fixed: CAGR "
                 f"{(t['cagr'] - f['cagr']) * 100:+.1f}pt, Sharpe {t['sharpe'] - f['sharpe']:+.2f}, "
                 f"maxDD {(t['mdd'] - f['mdd']) * 100:+.1f}pt · tilt alone (vs gate-only): CAGR "
                 f"{(t['cagr'] - g['cagr']) * 100:+.1f}pt, Sharpe {t['sharpe'] - g['sharpe']:+.2f} · "
                 f"walk-forward anchors vs in-sample anchors: CAGR "
                 f"{(wfm['cagr'] - t['cagr']) * 100:+.1f}pt — the size of the "
                 f"weight-level look-ahead the published numbers no longer carry.\n")
        for lbl, rep in (("Gate only", p["gate"]), ("Gate + tilt", p["tilted"]),
                         ("Walk-forward (published)", p["wf"])):
            tu = rep["turnover"]
            L.append(f"*{lbl}*: mean daily one-way turnover "
                     f"{tu['mean'] * 100:.1f}% (p95 {tu['p95'] * 100:.1f}%), a >0.5% "
                     f"rebalance on {tu['days_traded'] * 100:.0f}% of days, average SATA "
                     f"weight {rep['sata'].mean() * 100:.0f}%.")
        L.append("")
    L.append("## Findings\n")
    gc, tc_, dd, sh, wfd = [], [], [], [], []
    for p in profiles.values():
        f, g, t, w = (r["full"] for r in p["rows"])
        gc.append((g["cagr"] - f["cagr"]) * 100)
        tc_.append((t["cagr"] - g["cagr"]) * 100)
        dd.append((t["mdd"] - f["mdd"]) * 100)
        sh.append(t["sharpe"] - f["sharpe"])
        wfd.append((w["cagr"] - t["cagr"]) * 100)
    L.append(f"* **The gate (concentration onto live longs) is the dominant "
             f"daily-adjustment effect**: vs the old fixed-weight numbers it "
             f"adds **{min(gc):+.1f} to {max(gc):+.1f} CAGR points** across the "
             f"profiles, at the price of **{min(dd):+.1f} to {max(dd):+.1f} points "
             f"of max drawdown** and a Sharpe change of {min(sh):+.2f} to "
             f"{max(sh):+.2f}.")
    L.append(f"* **The priority tilt itself is a small increment on top of the "
             f"gate**: {min(tc_):+.1f} to {max(tc_):+.1f} CAGR points vs gate-only, "
             f"with essentially unchanged Sharpe and drawdown.")
    L.append(f"* **In-sample anchors flattered every variant**: re-fitting the "
             f"anchor weights walk-forward (prior data only, no fundamental "
             f"overlay) costs {min(wfd):+.1f} to {max(wfd):+.1f} CAGR points vs "
             f"the same gate+tilt logic on full-history anchors — that gap IS "
             f"the weight-level look-ahead. The app now publishes the "
             f"walk-forward numbers.")
    L.append("* Net: the old fixed-weight back-test mis-stated both the return "
             "and the risk of what the live daily-adjusted book actually does; "
             "the daily adjustments mostly re-shape the risk/return point "
             "(more return, deeper drawdowns) rather than adding risk-adjusted "
             "edge.\n")
    L.append("## Caveats\n")
    L.append("* The first three variants share full-history anchors "
             "DELIBERATELY, so their deltas isolate the gate and tilt effects; "
             "only the walk-forward row (and the app) removes the anchor "
             "look-ahead. Per-sleeve strategy parameters remain tuned on "
             "history in all variants; see `OVERALL_OOS_WALKFORWARD_EVAL.md` "
             "for the older frozen-2021-weights treatment.")
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
