# Threshold Optimization-Window Evaluation — Feb-28 vs May-31 cutoff

**Document type:** Read-only robustness audit of the BTC/MSTR/MSTU **signature
thresholds** (U1, D1, D2, D3, V-reversal, MA trend filter, per-asset stops).
**Nothing in the running app, config, weights, or model was changed.** This only
measures how the *documented OOS results* would change if the thresholds had been
optimized on a training window ending at the **model's own Feb-28-2026 training
cutoff** instead of the full period ending **May-31-2026**.

**Reproduce:** `python scripts/eval_threshold_opt_window.py`
(offline — cached vintage `data/backtest/` v3, pull 2026-07-11; CT artefact
`models/inference_assets_ct.joblib`; same feature-build and signal math as the UI
backtest in `scripts/compare_variants_full.py`.)

---

## The question

The live thresholds — `U1 err_hi_ma3 > +1.3%`, `D2 err_hi_ma3 < −1.3%`, D1/D3/
V-reversal, the above-MA30 trend gate, and per-asset stops (BTC none / MSTR −3% /
MSTU −3%) — were chosen to **maximize the full-period backtest (Jun 2024 →
May 31 2026).**

But the CT H/L model is **trained through Feb 28 2026**, so **Mar 2026 → present is
the model's genuine out-of-sample window** — and it is *inside* the threshold
optimization window. The thresholds were allowed to see Mar–May 2026, the very
period whose numbers the docs advertise as "OOS."

**Counterfactual:** refit the thresholds to maximize the full period ending
**Feb 28 2026** (the model boundary), making Mar → present out-of-sample for the
*thresholds too*, and compare the OOS backtest with the documented one.

## Method

* Same engine as the live UI backtest; only **U1, D2, and the two equity stops** are
  parametrized (the knobs the docs describe as re-tuned). D1 (0.5), D3, V-reversal
  (dn>0.8 / errlo>3%) and the MA30 gate are held at their structural live values.
* **Selection rule** = maximize the summed full-period return of BTC + MSTR + MSTU,
  shared U1/D2, per-asset stop — run on each training window (through Feb-28, through
  May-31). Grids: U1 ∈ {0.5…2.1}, D2 ∈ {−0.5…−2.1}, MSTR-stop ∈ {none,3,5,7%},
  MSTU-stop ∈ {none,3,5,7,10%} (1 260 configs).
* **OOS eval:** each selected config is backtested on **2026-03-01 → 2026-07-10**
  with fresh $100k at the OOS start — identical measurement for every config.

> **Reproduction check (validates the whole method):** optimizing on the full period
> through **May 31 recovers the live config *exactly*** — `U1 > +1.3 / D2 < −1.3`,
> MSTR −3%, MSTU −3%. So the grid + objective faithfully reverse-engineers how the
> live thresholds were set, and the Feb-28 run is an apples-to-apples counterfactual.

*(Absolute magnitudes differ from the headline numbers in `TRADING_STRATEGY.md` —
that doc repeatedly warns figures drift on every data re-pull and says to "treat
magnitudes as illustrative." All comparisons here are within one consistent engine
and one data vintage, so the **relative** Feb-vs-May gap is the finding, not the
absolute levels.)*

---

## 1. What each optimization window selects

| Optimized on | U1 | D2 | MSTR stop | MSTU stop |
|---|---|---|---|---|
| **Full → May 31 2026** (= live) | **> +1.3%** | **< −1.3%** | −3% | −3% |
| **Full → Feb 28 2026** (counterfactual) | > +1.3% | **< −2.1%** | −3% | −3% |

Only **one threshold moves: the D2 exit.** U1, both stops, and every structural
signal are identical. The Feb-cutoff optimizer loosens D2 from −1.3% to −2.1% (a much
more patient exit) because the Feb-truncated window is dominated by the huge 2024 →
early-2025 bull, where holding longer scores higher.

## 2. OOS backtest — documented vs honestly-out-of-sample

**OOS window 2026-03-01 → 2026-07-10, fresh $100k.** B&H over the window:
BTC −3.2% · MSTR −26.9% · MSTU −61.7%.

| Config | BTC | MSTR | MSTU |
|---|---|---|---|
| **LIVE / documented** (opt → May, U1+1.3 / D2−1.3) | **+1.5%** | **+37.8%** | **+79.2%** |
| **Counterfactual** (opt → Feb, U1+1.3 / **D2−2.1**) | **−17.8%** | **−1.7%** | **−12.8%** |
| **OOS delta (Feb − May)** | **−19.3 pp** | **−39.5 pp** | **−92.0 pp** |

Frozen at the model's own training boundary, the same optimization procedure would
have selected a threshold set whose OOS result **collapses from strongly beating a
falling Buy & Hold to roughly matching it** — MSTR from +37.8% to breakeven, MSTU
from +79% to −13%, BTC from a small gain to −18%.

On a clean Mar–May quarter the same ordering holds but is milder (MSTR +37.8→+23.6,
MSTU +79.2→+40.7); the gap widens through Jun–Jul as the patient D2−2.1 exit keeps
holding into further weakness.

## 3. Where the gap lives — the D2 exit sweep

Holding U1 = +1.3 and stops 3/3, sweeping only D2:

| D2 exit | Full→Feb train (sum) | **Genuine OOS (Mar–Jul, sum)** |
|---|---|---|
| −0.50 | 368 | +99.1 |
| −0.75 | 391 | +99.1 |
| −1.00 | 446 | +99.1 |
| **−1.30  ← live** | 594 | **+118.5** ← also the OOS-best |
| −1.50 | 594 | +118.5 |
| −1.80 | 670 | −4.8 |
| **−2.10  ← Feb argmax** | **678** | **−32.3** |

This is the crux. **Live D2 = −1.3 is genuinely the best exit out-of-sample**
(+118.5, the top of the OOS column). The problem is that on the **Feb-truncated
training window the objective keeps *rising* all the way to −2.1** — the Feb
optimizer has no reason to stop at −1.3 and every reason to loosen further. The only
thing that pulls the optimum back to −1.3 is **the Mar–May 2026 chop itself**, where
the loose −1.8/−2.1 exits get punished. That discriminating information exists in the
May-31 window and **does not exist yet** in the Feb-28 window.

Put differently: the live thresholds are OOS-good, but the *procedure* that produced
them only landed on them **because the optimization window overlapped the OOS
window.** The documented OOS outperformance is, to a large degree, a look-ahead
artifact of that overlap.

## 4. Robustness — is the −2.1 pick a knife-edge?

The single Feb argmax is unluckier than its neighbourhood, but the neighbourhood is
still materially worse than the documented result:

* **Top-10 Feb configs**: every one picks D2 −1.8 or −2.1; their three-asset OOS sums
  run −4.8 to −41.5, versus the live config's **+118.5** (1.5 + 37.8 + 79.2) on the
  same basis.
* **Top-decile (126 configs)** median OOS sum **+11.0%**, mean +48.5% — so a
  *robustness-aware* selection (not naive argmax) would have escaped the −2.1 trap and
  landed near breakeven-to-modestly-positive, still **far below** the documented
  numbers.
* The **live config ranks only 26th of 1 260 on the Feb objective** (feb_total 594 vs
  top 678) — confirming it was **not selectable** as the Feb optimum. A Feb-boundary
  optimizer would not have chosen it.

---

## Findings

1. **The documented OOS numbers are inflated by optimization-window look-ahead.**
   The thresholds were tuned on a window (through May 31) that contains the Mar–May
   2026 OOS period, so the "OOS" results are partly in-sample *for the thresholds.*
2. **Refit at the model's Feb-28 boundary, OOS performance collapses** — MSTR +37.8%
   → −1.7%, MSTU +79.2% → −12.8%, BTC +1.5% → −17.8% (Mar 2026 → Jul 2026). The naive
   Feb optimum barely differs from a losing Buy & Hold.
3. **The entire gap is the D2 exit threshold.** Live D2 −1.3 is coincidentally the
   true OOS-optimal exit, but a Feb-cutoff optimizer would pick −2.1 (best on the
   pre-Feb bull) and give most of it back in the post-Feb chop.
4. **Even robustness-aware Feb selection underperforms the documented headline** —
   top-decile median OOS ≈ +11% three-asset sum vs the documented +118.5% — so the
   conclusion does not depend on the single knife-edge argmax.

## Caveats

* Only U1/D2/stops were re-optimized; D1, D3, V-reversal and the MA gate were held
  fixed. Re-tuning those too would only *add* overfitting surface, widening (not
  narrowing) the Feb-vs-May gap.
* One data vintage, ~5–20 trades per asset per window — wide confidence intervals, as
  the strategy docs already stress. This is a **directional** robustness result, not a
  precise point estimate.
* MSTU uses the OLS-synthetic pre-inception series (same as the research scripts).
* All figures gross of costs/slippage; leverage tail risk understated at daily close.

## Bottom line

If the thresholds had been optimized to maximize the full period ending **Feb 28
2026** — the model's own training cutoff, which makes Mar 2026 → present genuinely
out-of-sample — the **OOS backtest would look substantially worse than the documented
one**: MSTR and MSTU fall from strong positive OOS returns to roughly breakeven-or-
negative, and BTC turns negative. The documented OOS edge is real *in-sample* but
rests on the D2 exit threshold having been chosen with visibility into the Mar–May
2026 window it is later reported on. A clean, model-aligned optimization boundary
removes most of that edge — a caution the existing docs' "heavy in-sample
optimization" and "small OOS sample" warnings already point toward, quantified here.
