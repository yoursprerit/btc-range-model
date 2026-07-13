# Threshold Optimization-Window Evaluation — sensitivity of the documented OOS

**Document type:** Read-only robustness audit of the BTC/MSTR/MSTU **signature
thresholds** (U1, D1, D2, D3, V-reversal, MA trend filter, per-asset stops).
**Nothing in the running app, config, weights, or model was changed.** This measures
how the *documented OOS results* would change if the thresholds had been optimized on
a training window ending **earlier** than the full period through **May 31 2026** —
specifically at the model's **Feb 28 2026** training cutoff and at **Aug 31 2025**.

**Reproduce:** `python scripts/eval_threshold_opt_window.py`
(offline — cached vintage `data/backtest/` v3, pull 2026-07-11; CT artefact
`models/inference_assets_ct.joblib`; same feature-build and signal math as the UI
backtest in `scripts/compare_variants_full.py`.)

---

## The question

The live thresholds — `U1 err_hi_ma3 > +1.3%`, `D2 err_hi_ma3 < −1.3%`, D1/D3/
V-reversal, the above-MA30 trend gate, per-asset stops (BTC none / MSTR −3% /
MSTU −3%) — were chosen to **maximize the full-period backtest (Jun 2024 →
May 31 2026).**

The CT H/L model is **trained through Feb 28 2026**, so **Mar 2026 → present is the
model's genuine out-of-sample window** — and it sits *inside* the threshold
optimization window. The thresholds were allowed to see Mar–May 2026, the very
period whose numbers the docs advertise as "OOS."

**Counterfactuals:** refit the thresholds to maximize the full period ending at an
**earlier** cutoff, so Mar 2026 → present is out-of-sample for the *thresholds too*,
and compare the OOS backtest with the documented one. Two cutoffs are tested:

* **Feb 28 2026** — the model's own training boundary.
* **Aug 31 2025** — six months earlier, i.e. before the Sep 2025 → 2026 downturn had
  even begun. This window is essentially the *pure 2024 → mid-2025 bull*.

## Method

* Same engine as the live UI backtest; only **U1, D2, and the two equity stops** are
  parametrized (the knobs the docs describe as re-tuned). D1 (0.5), D3, V-reversal
  (dn>0.8 / errlo>3%) and the MA30 gate are held at their structural live values.
* **Selection rule** = maximize the summed full-period return of BTC + MSTR + MSTU,
  shared U1/D2, per-asset stop, on each training window. Grids: U1 ∈ {0.5…2.1},
  D2 ∈ {−0.5…−2.1}, MSTR-stop ∈ {none,3,5,7%}, MSTU-stop ∈ {none,3,5,7,10%}
  (1 260 configs).
* **OOS eval window is held FIXED** at **2026-03-01 → 2026-07-10**, fresh $100k, for
  *every* cutoff — so the only thing that changes across rows is where the thresholds
  were allowed to "see," making it apples-to-apples.

> **Reproduction check (validates the method):** optimizing on the full period through
> **May 31 recovers the live config *exactly*** — `U1 > +1.3 / D2 < −1.3`, MSTR −3%,
> MSTU −3%. So the grid + objective faithfully reverse-engineers how the live
> thresholds were set, and the earlier-cutoff runs are true counterfactuals.

*(Absolute magnitudes differ from the headline numbers in `TRADING_STRATEGY.md` —
that doc repeatedly warns figures drift on every data re-pull and to "treat magnitudes
as illustrative." All comparisons here are within one engine and one data vintage, so
the **relative** cross-cutoff gap is the finding, not the absolute levels.)*

---

## 1. What each optimization window selects

| Optimized on | U1 | D2 | MSTR stop | MSTU stop |
|---|---|---|---|---|
| **Full → May 31 2026** (= live) | **> +1.3%** | **< −1.3%** | −3% | −3% |
| Full → Feb 28 2026 | > +1.3% | **< −2.1%** | −3% | −3% |
| Full → Aug 31 2025 | **> +1.1%** | **< −2.1%** | −3% | −3% |

Both earlier cutoffs loosen the **D2 exit** from −1.3% to the grid edge −2.1% (the Aug
window also nudges U1 down to +1.1%). Every training window that ends *before* the
Mar–May 2026 chop is dominated by the 2024 → mid-2025 bull, where a more patient exit
scores higher — so the optimizer keeps loosening D2 all the way out.

## 2. OOS backtest — documented vs honestly-out-of-sample

**Fixed OOS window 2026-03-01 → 2026-07-10, fresh $100k.** B&H over the window:
BTC −3.2% · MSTR −26.9% · MSTU −61.7%.

| Threshold set | BTC | MSTR | MSTU |
|---|---|---|---|
| **LIVE / documented** (opt → May 31, U1+1.3 / D2−1.3) | **+1.5%** | **+37.8%** | **+79.2%** |
| Counterfactual opt → **Feb 28 2026** (D2−2.1) | −17.8% | −1.7% | −12.8% |
| Counterfactual opt → **Aug 31 2025** (U1+1.1 / D2−2.1) | −16.3% | −1.8% | −12.9% |
| **Δ Feb-cutoff − live** | −19.3 pp | −39.5 pp | −92.0 pp |
| **Δ Aug-cutoff − live** | −17.8 pp | −39.6 pp | −92.2 pp |

**The Aug-31-2025 cutoff produces essentially the same OOS collapse as the Feb-28-2026
cutoff.** Both refits fall from strongly beating a falling Buy & Hold to roughly
matching it: MSTR from +37.8% to breakeven, MSTU from +79% to −13%, BTC from a small
gain to −16/−18%. Moving the cutoff earlier does **not** help — it slightly hurts.

## 3. Where the gap lives — the D2 exit sweep

Holding U1 = +1.3 and stops 3/3, sweeping only D2 (identical OOS column for both
earlier windows, because the OOS window is fixed):

| D2 exit | train sum → Aug 31 2025 | train sum → Feb 28 2026 | **Genuine OOS (Mar–Jul, sum)** |
|---|---|---|---|
| −0.50 | 451 | 368 | +99.1 |
| −1.00 | 492 | 446 | +99.1 |
| **−1.30  ← live** | 650 | 594 | **+118.5** ← also the OOS-best |
| −1.50 | 650 | 594 | +118.5 |
| −1.80 | 670 | 670 | −4.8 |
| **−2.10  ← both earlier argmaxes** | **678** | **678** | **−32.3** |

This is the crux, and it is identical for both earlier cutoffs. **Live D2 = −1.3 is
genuinely the best exit out-of-sample** (+118.5, the top of the OOS column). But on
*either* pre-chop training window the objective keeps *rising* to the −2.1 grid edge —
the optimizer has no reason to stop at −1.3. The only thing that pulls the optimum
back to −1.3 is **the Mar–May 2026 chop itself**, where the loose −1.8/−2.1 exits get
punished. That discriminating information exists only in the May-31 window and does
**not** exist yet at Feb-28 or Aug-31.

The live thresholds are OOS-good, but the *procedure* that produced them only landed on
them **because the optimization window overlapped the OOS window.** The documented OOS
outperformance is, to a large degree, a look-ahead artifact of that overlap.

### 3a. "But the Feb window contains the Oct-2025 crash and the Oct→Feb bear — why didn't that tighten D2?"

Because **that bear rewarded the *patient* exit, not the tight one.** Decomposing the
return by sub-period (fresh $100k each, U1=1.3, stops 3/3):

| Sub-period | BTC (−1.3 / −2.1) | MSTR (−1.3 / −2.1) | MSTU (−1.3 / −2.1) | Prefers |
|---|---|---|---|---|
| **Bull** Jun24–Sep25 | +21.4 / **+50.0** | 161.6 / 161.6 | 466.6 / 466.6 | −2.1 (BTC), tie |
| **Oct-bear** Oct25–Feb26 | **−5.7 / 0.0** | −5.4 / 0.0 | −6.2 / 0.0 | **−2.1 (all three)** |
| **Mar-chop** Mar26–Jul26 (OOS) | +1.5 / −17.8 | +37.8 / −1.7 | +79.2 / −12.8 | −1.3 (all three) |

Three reasons the Oct bear pushed *toward* −2.1:

1. **The strategy sat the Oct bear out in cash** — the U1 entry gate fired only **1 time
   in 151 bars** during Oct25→Feb26 (BTC below MA30, no hi-band breaks). You don't need a
   good exit for a bear you never enter; the bear is avoided by *not entering*, not by D2.
2. **The one trade it did take, the tight exit lost on.** Under D2 −1.3 there is exactly
   one Oct trade: enter Oct 1 → D2-exit Oct 11 at **−5.7%** (the Oct-10/11 crash). Under
   D2 −2.1 that trade never opens (a looser D2 fires less → `clean_7d` True more often →
   the `above_ma30 XOR clean_7d` rule *blocks* the marginal entry), so the window ends
   flat at 0.0%. The looser exit accidentally dodged the crash trade.
3. **The bull dwarfs it anyway.** Even the +5–6pp the Oct bear hands to −2.1 is a rounding
   error next to MSTU's +466% bull trades, which already prefer the patient exit.

So the optimizer was **not** blind to a bear — it saw one whose lesson was "hold longer."
The opposite lesson ("cut fast") is uniquely encoded in the **Mar–May 2026 chop**, a
distinct regime where BTC bounces enough to *trigger* U1 entries (fired 2× in the OOS
window) and then fades — the one setup where a tight exit saves you. That regime does not
appear anywhere before March 2026, which is why only the May-31 window recovers D2 −1.3.

## 4. Robustness — knife-edge, and does an earlier cutoff generalize worse?

| Optimized on | Top-decile OOS sum (median / mean) | LIVE-config rank on that objective |
|---|---|---|
| Feb 28 2026 | **+11.0% / +48.5%** | 26 of 1 260 |
| Aug 31 2025 | **−8.9% / +29.9%** | 21 of 1 260 |

* The single argmax (D2 −2.1) is unluckier than its neighbourhood, but the
  neighbourhood is still far below the documented result under both cutoffs.
* **The Aug-31-2025 window generalizes *worse* than Feb-28** (top-decile median OOS
  −8.9% vs +11.0%). It contains **zero downturn data** — it ends before the Sep 2025
  decline began — so its "good" training configs carry even less information about the
  bear/chop OOS regime.
* Under both cutoffs the **live config was not selectable** (ranks ~21–26 of 1 260) —
  a pre-chop optimizer would not have chosen it.

---

## Findings

1. **The documented OOS numbers are inflated by optimization-window look-ahead.** The
   thresholds were tuned on a window (through May 31) that contains the Mar–May 2026
   OOS period, so the "OOS" results are partly in-sample *for the thresholds.*
2. **Refit at any pre-chop boundary — Feb 28 2026 *or* Aug 31 2025 — the OOS
   performance collapses to the same place:** MSTR +37.8% → ≈−1.8%, MSTU +79.2% →
   ≈−13%, BTC +1.5% → ≈−17% (Mar → Jul 2026). Both barely differ from a losing B&H.
3. **The entire gap is the D2 exit threshold.** Live D2 −1.3 is coincidentally the
   true OOS-optimal exit, but any optimizer that cannot see the Mar–May 2026 chop
   picks −2.1 (best on the bull) and gives it all back afterwards.
4. **Moving the cutoff earlier does not help and slightly hurts.** Aug-31-2025 is a
   pure-bull window with no downturn signal; its good-training neighbourhood
   generalizes *worse* OOS (top-decile median −8.9% vs Feb's +11.0%).

## Caveats

* Only U1/D2/stops were re-optimized; D1, D3, V-reversal and the MA gate were held
  fixed. Re-tuning those would only *add* overfitting surface, widening the gap.
* One data vintage, ~5–20 trades per asset per window — wide confidence intervals, as
  the strategy docs already stress. Directional result, not a precise point estimate.
* MSTU uses the OLS-synthetic pre-inception series (same as the research scripts).
* All figures gross of costs/slippage; leverage tail risk understated at daily close.

## Bottom line

If the thresholds had been optimized to maximize the full period ending **Aug 31
2025**, the OOS backtest (Mar 2026 → present) would look **substantially worse than the
documented one — and essentially identical to the Feb-28-2026 counterfactual**: MSTR
and MSTU fall from strong positive OOS returns to roughly breakeven-or-negative, and
BTC turns negative. The reason is the same in both cases: the D2 exit threshold is
chosen with visibility into the Mar–May 2026 window it is later reported on. Any
optimization boundary that predates that chop — whether the model's own Feb-28 cutoff
or the earlier Aug-31-2025 pure-bull window — removes the look-ahead and most of the
documented edge with it. The earlier the pre-chop cutoff, the *less* downturn
information the thresholds carry, so Aug-2025 generalizes marginally worse, not better.
