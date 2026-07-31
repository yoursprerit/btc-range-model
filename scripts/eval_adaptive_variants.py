"""Adaptivity experiments for the Overall strategy — EVALUATION ONLY.

The published walk-forward gated replay adapts fast through its signals
(binary exits), moderately through the daily priority tilt, and slowly
through its anchors (expanding-window fits that never forget).  This harness
builds the four candidate upgrades discussed for making the strategy learn
and evolve when sleeves under/over-perform, and measures each against the
published baseline — WITHOUT touching the running strategy (every knob is an
opt-in parameter whose default reproduces the published behaviour, and the
app passes none of them):

  baseline        the published construction — QUARTERLY expanding-window
                  anchor refits (V3, adopted 2026-07 after this eval showed
                  it beat annual on both return and Sharpe), lifetime
                  (expanding) win-rate & Sharpe in the priority score, no
                  penalty box.
  V0 annual-refit the pre-V3 published construction (anchors re-fit each
                  Jan 1) — kept for reference.
  V1 rolling-anchors   anchors fit on only the trailing 504 bars (~2 years),
                  so the optimiser FORGETS old regimes.
  V2 rolling-priority  win-rate over the last 20 closed trades and Sharpe
                  over the last 252 bars — priority reacts to recent form.
  V4 penalty-box  a funded sleeve whose rolling 126-bar Sharpe (lagged one
                  bar) is negative has its raw weight halved before the
                  water-fill — automatic de-rating of cold sleeves.
  V5 combined     V1+V2+V4 on the quarterly baseline.

All variants remain strictly causal (rolling windows and lags only); the
script spot-checks that by prefix-invariance on the combined variant.

    python scripts/eval_adaptive_variants.py     # writes OVERALL_ADAPTIVE_EVAL.md
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "app"))
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO))

import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
import overall_core as ov                # noqa: E402
from check_lookahead import truncate_results   # noqa: E402

PROFILES = ("Balanced", "Growth")
SLICES = [("2025-01-01", "since 2025"), ("2026-01-01", "2026 YTD")]
OUT_MD = _REPO / "OVERALL_ADAPTIVE_EVAL.md"

VARIANTS = [
    ("Baseline (published, quarterly refits)", dict()),
    ("V0 Annual refits (pre-V3 published)", dict(refit="A")),
    ("V1 Rolling anchors (2y window)", dict(fit_window=504)),
    ("V2 Rolling priority (20 trades / 252 bars)",
     dict(wr_window=20, sharpe_window=252)),
    ("V4 Penalty box (126-bar Sharpe < 0 → ×0.5)",
     dict(penalty=dict(window=126, floor=0.0, mult=0.5))),
    ("V5 Combined (V1+V2+V4 on quarterly)",
     dict(fit_window=504, wr_window=20, sharpe_window=252,
          penalty=dict(window=126, floor=0.0, mult=0.5))),
]


def _metrics_row(wf) -> dict:
    eq = wf["equity"]
    out = dict(full=wf["metrics"], turnover=wf["turnover"]["mean"],
               sata=float(wf["sata"].mean()))
    for start, name in SLICES:
        out[name] = ov.slice_metrics(eq, start)
    return out


def _fmt(m: dict | None) -> str:
    if not m:
        return "—"
    return (f"{m['total_ret'] * 100:+9.1f}% | {m['cagr'] * 100:+7.1f}% | "
            f"{m['mdd'] * 100:6.1f}% | {m['sharpe']:5.2f}")


def main() -> None:
    print("Running the universe once…")
    results = ov.run_universe()
    print(f"{len(results)} sleeves\n")

    table: dict[str, dict[str, dict]] = {}
    for pname in PROFILES:
        prof = ov.RISK_PROFILES[pname]
        caps = ov.caps_for(pname)
        table[pname] = {}
        print(f"═══ {pname} ═══   (total | CAGR | maxDD | Sharpe)")
        for label, kw in VARIANTS:
            wf = ov.walkforward_gated_replay(
                results, caps=caps, mdd_floor=prof["mdd_floor"],
                objective=prof["objective"], sata_daily=ov.SATA_DAILY,
                tilt=True, **kw)
            row = _metrics_row(wf)
            table[pname][label] = row
            print(f"  {label:<44} full: {_fmt(row['full'])}")
            for _, sname in SLICES:
                print(f"  {'':<44} {sname}: {_fmt(row[sname])}")
            print(f"  {'':<44} turnover {row['turnover'] * 100:.1f}%/day · "
                  f"avg SATA {row['sata'] * 100:.0f}%")
        print()

    # ── causality spot-check on the most aggressive variant ────────────────
    print("Prefix-invariance spot-check (V5 combined, Balanced, cutoff 2026-01-15)…")
    kw = VARIANTS[-1][1]
    prof = ov.RISK_PROFILES["Balanced"]
    caps = ov.caps_for("Balanced")
    full = ov.walkforward_gated_replay(results, caps=caps,
                                       mdd_floor=prof["mdd_floor"],
                                       objective=prof["objective"],
                                       sata_daily=ov.SATA_DAILY, tilt=True, **kw)
    cut = pd.Timestamp("2026-01-15")
    orig_load = ov._load_daily
    ov._load_daily = lambda cfg, _c=cut: orig_load(cfg).loc[:_c]
    try:
        part = ov.walkforward_gated_replay(truncate_results(results, cut),
                                           caps=caps, mdd_floor=prof["mdd_floor"],
                                           objective=prof["objective"],
                                           sata_daily=ov.SATA_DAILY, tilt=True, **kw)
    finally:
        ov._load_daily = orig_load
    d = float((part["ret"] - full["ret"].loc[part["ret"].index]).abs().max())
    ok = d < 1e-12
    print(f"{'PASS' if ok else 'FAIL'}  Δret {d:.2e}\n")
    if not ok:
        sys.exit(1)

    _write_md(table, results, ok)
    print(f"Wrote {OUT_MD.name}")


def _write_md(table: dict, results: list, causal_ok: bool) -> None:
    asof = max(r["as_of"] for r in results).date()
    L = ["# Overall Strategy — Adaptivity Variants Evaluation\n",
         f"*Generated by `scripts/eval_adaptive_variants.py` · data through "
         f"**{asof}** · {len(results)} sleeves · EVALUATION ONLY — the running "
         f"strategy is unchanged (all knobs are opt-in parameters whose "
         f"defaults reproduce the published behaviour) · no transaction "
         f"costs.*\n",
         "## Question\n",
         "Can the strategy learn and evolve when sleeves under/over-perform? "
         "The published construction adapts through binary signals (fast), "
         "the priority tilt (moderate) and expanding-window anchor refits — "
         "**quarterly since 2026-07 (V3 of this eval, adopted after it beat "
         "annual refits on both return and Sharpe)**. The remaining candidate "
         "upgrades are tested against it, individually and combined:\n",
         "| Variant | What changes | Adaptation it adds |",
         "|---|---|---|",
         "| V0 Annual refits | the pre-V3 published cadence (each Jan 1) | "
         "reference: what adopting V3 bought |",
         "| V1 Rolling anchors | anchors fit on trailing 504 bars (~2y) "
         "instead of all history | the optimiser *forgets* old regimes |",
         "| V2 Rolling priority | win-rate over last 20 trades, Sharpe over "
         "last 252 bars | priority reacts to recent form, not lifetime "
         "averages |",
         "| V4 Penalty box | funded sleeve with rolling 126-bar Sharpe < 0 "
         "(lagged) gets raw weight ×0.5 | automatic de-rating of cold "
         "sleeves |",
         "| V5 Combined | V1+V2+V4 on the quarterly baseline | all of the "
         "above |\n",
         "All variants are strictly causal (rolling windows and one-bar lags "
         "only); the combined variant "
         f"{'PASSED' if causal_ok else 'FAILED'} the truncation "
         "prefix-invariance spot-check.\n",
         "## Results\n"]
    for pname, rows in table.items():
        L.append(f"### {pname}\n")
        L.append("| Variant | Window | Total | CAGR | MaxDD | Sharpe |")
        L.append("|---|---|---:|---:|---:|---:|")
        for label, row in rows.items():
            for wname in ["full"] + [s for _, s in SLICES]:
                m = row[wname]
                if not m:
                    continue
                L.append(f"| {label} | {wname} | {m['total_ret'] * 100:+.1f}% | "
                         f"{m['cagr'] * 100:+.1f}% | {m['mdd'] * 100:.1f}% | "
                         f"{m['sharpe']:.2f} |")
        base = rows["Baseline (published, quarterly refits)"]["full"]
        L.append("")
        for label, row in rows.items():
            if label.startswith("Baseline"):
                continue
            m = row["full"]
            L.append(f"* **{label}** vs baseline: CAGR "
                     f"{(m['cagr'] - base['cagr']) * 100:+.1f}pt · Sharpe "
                     f"{m['sharpe'] - base['sharpe']:+.2f} · maxDD "
                     f"{(m['mdd'] - base['mdd']) * 100:+.1f}pt · turnover "
                     f"{row['turnover'] * 100:.1f}%/day")
        L.append("")
    L.append("## Caveats\n")
    L.append("* One parameterisation per idea (504 bars, 20 trades, 252 bars, "
             "126-bar/0.0/×0.5) — none of these knobs were swept, so a "
             "variant that loses here might win at other settings (and "
             "sweeping them on this same history would be its own "
             "overfitting risk).")
    L.append("* Per-sleeve strategy parameters remain tuned on history in "
             "every variant; no transaction costs anywhere.")
    L.append("* Rolling anchor fits see at most ~2 years — with a 250-bar "
             "warm-up requirement their earliest refits behave close to the "
             "expanding fits, so differences concentrate in the later years.")
    L.append("* Regenerate with `python scripts/eval_adaptive_variants.py`.")
    OUT_MD.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
