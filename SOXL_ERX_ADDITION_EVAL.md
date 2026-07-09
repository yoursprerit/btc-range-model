# Adding leveraged siblings (SOXL · ERX) to the Overall strategy — evaluation

**Question.** The Overall Trading portfolio already trades leveraged / high-beta
siblings off a clean parent signal (MSTU=2× off BTC, UGL=2× off gold, OIH off
XLE). Its two **top-conviction signals had no leveraged sibling**: SOXX (semis,
1.40 conviction, a 25/100 dual-MA) and XLE (energy divergence). Does adding
**SOXL (3× semis)** and **ERX (2× energy)** — each traded off its parent's
already-tuned signal — improve the combined back-test when the goal is **higher
total return** (accepting a little more drawdown)?

**Implemented.** Unlike the other eval docs this one shipped: SOXL is now a
sibling in the SOXX config and ERX in the XLE config (`app/ticker_config.py`),
both registered `kind="lev"` in `app/overall_core.py`. Reproduce the analysis
with:

```bash
python scripts/eval_leveraged_add.py     # SOXL / ERX / NUGT, sleeve + combined deltas
python scripts/build_overall.py          # rebuild the shipped combined artifact
```

## Why leverage (and why on the parent signal)

Every earlier candidate eval showed the same thing: the existing book compounds
so fast (equal-weight +526% OOS) that any **1× diversifier dilutes total
return** — QQQ, DBA, DBMF all lowered return. To *raise* return you need a sleeve
whose own strategy out-returns the book, which — with no shorting and no new
signal risk — means **leverage on a signal that already sidesteps the bears**.
The long/flat trend rule is the natural home for a leveraged ETF: it holds the 3×
only while the parent trends up and stands aside in the chop where daily-rebalance
decay is worst. So SOXL is driven by SOXX's 25/100 dual-MA and ERX by the XLE
energy-divergence rule — never their own noisier signals — exactly as UGL/OIH are.

## Method

* Runs the live universe **once**, builds each leveraged sleeve *driven by its
  parent signal*, and folds it into the returns/position matrices.
* Re-optimises the cross-asset weights for **each risk profile** with the
  **identical** `optimize_weights()` call the app and `build_overall.py` use
  (same caps, SATA idle-cash yield, drawdown floor, objective, fundamental view).
* Leveraged sleeves carry the tightest cap (`CAP_BY_KIND["lev"]`: 0.10 Balanced →
  0.35 Aggressive), so the optimiser only leans on them when the extra beta pays.
* SOXL/ERX both inherit the parent's 2021 OOS start, so the common back-test
  window is unchanged (`1652 × 2021-01-05 → 2026-07-09`) and the curves compare
  directly.

Window: full OOS **2021-01-05 → 2026-07-09**.

---

## Result — SOXL is a large return lift; ERX is a rare win-win

### Sleeve-level (own strategy, driven by the parent signal), OOS

| Sleeve | Signal | Own return | Own MDD | Own Sharpe |
|---|---|---|---|---|
| **SOXL** 3× semis | SOXX 25/100 dual-MA | **+1829% → +2126%** | −67% | 1.07–1.09 |
| **ERX** 2× energy | XLE divergence | **+279%** | **−17%** | **1.33** |
| *(NUGT 2× gold miners)* | gold divergence | +497% | −31% | 1.21 |

### Combined blend — deltas vs the same baseline optimise (fundamental-neutral)

Baseline (Aggressive): **+1043%** return, **−16.75%** MDD, Sharpe **1.882**.

| Add | Aggressive Δreturn | Aggressive MDD | Aggressive ΔSharpe | Equal-wt Δreturn *(clean)* | wt |
|---|---|---|---|---|---|
| **+ SOXL** | **+843.6pp → +1887%** | −16.8% → −25.5% | −0.278 | **+171pp** | 24.0% |
| **+ ERX** | +40.9pp → +1084% | −16.8% → **−14.2%** | **+0.373** | +27pp (Sh +0.153) | 14.6% |
| + SOXL + ERX | +696.6pp → +1740% | −16.8% → −23.9% | −0.296 | +190pp | 28.6 / 3.6% |
| *(+ NUGT)* | +53.0pp → +1096% | −16.8% → **−15.8%** | **+0.421** | +45pp | 24.0% |

* **SOXL** is the raw-return engine: it nearly **doubles** the Aggressive
  return for a moderate drawdown increase that stays inside the −38% budget. The
  deterministic equal-weight blend (no optimiser/fundamental noise) confirms the
  lift is real — **+171pp** attributable return.
* **ERX** is the rare **win-win**: it lifts return **and Sharpe (+0.37) and trims
  drawdown**, because it is genuinely uncorrelated to the tech-heavy book (≈0 to
  SOXX/ARTY/BTC/gold; 0.99 only to XLE) and its XLE-signal strategy has a stellar
  1.33 own-Sharpe at just −17%.
* In the **pair**, SOXL's return-per-unit is so high it crowds ERX down to ~4%
  under the max-return objective, so the pair ≈ SOXL-alone with a small ERX
  drawdown cushion (−23.9% vs SOXL-alone −25.5%). Both still earn real weight, so
  these are trustworthy deltas, not the ~0.6%-weight search-noise seen when a
  redundant 1× (QQQ) was tested.

### Shipped combined result (with the fundamental view, SOXL 1.40 / ERX 1.00)

| Profile | Optimal return | MDD | Sharpe | SOXL wt | ERX wt |
|---|---|---|---|---|---|
| Balanced   | +725%  | **−4.98%** | 3.18 | ~0% | ~0% |
| Growth     | +1576% | −21.46% | 1.79 | ~12% | ~4% |
| **Aggressive** | **+2161%** | −34.16% | 1.40 | **30.4%** | 1.9% |

The leveraged caps do their job: **Balanced barely touches SOXL** (Sharpe held at
3.18, −5% drawdown ≈ unchanged), while **Aggressive roughly doubles total return
(~+1034% → +2161%)** by loading SOXL to 30%, pushing drawdown to −34% — deep, but
inside the profile's −38% budget by design.

---

## Verdict

**Yes — for a return-maximising mandate, adding SOXL and ERX is a real, sizeable
improvement, with the risk profiles ring-fencing the extra risk.**

* **SOXL** delivers the return the earlier 1× candidates could not: a genuine
  ~2× lift to the Aggressive book, confirmed on the noise-free equal-weight blend
  (+171pp). The cost is a deeper Aggressive drawdown (−17% → −34%) and a lower
  Aggressive Sharpe — the explicit return-for-drawdown trade. Because SOXL is a
  leveraged sleeve, **Balanced and Growth are largely insulated** (tight caps), so
  only users who dial up to Aggressive take the extra risk.
* **ERX** is a strict upgrade on every axis (return ↑, Sharpe ↑, drawdown ↓) and
  diversifies the tech-heavy book — the highest-quality single add.
* **NUGT** (2× gold miners) tested as a second win-win (+53pp return, −15.8% MDD,
  Sharpe +0.42) and is a strong optional third; it was left out of this ship to
  avoid over-stacking the already gold-leveraged sleeve (UGL + GDX).

**Honest caveats.** SOXL is 0.99-correlated to SOXX — it *amplifies* your largest
exposure, it does not diversify it, so its added drawdown is correlated with the
book's worst days, and 3× gap-risk in a sudden crash is worse than a daily
back-test captures. Leveraged-ETF decay **is** in these numbers (real SOXL/ERX
prices); the long/flat filter mitigates it by holding only in clean up-trends,
which is why the strategy Sharpe (~1.1) far exceeds a buy-&-hold 3×.
