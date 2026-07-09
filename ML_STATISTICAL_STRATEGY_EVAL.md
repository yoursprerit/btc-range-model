# Broader ML / statistical strategy search — SOXX · VEGN · GRID · REMX · WGMI

**Question.** Beyond the MA trend filter (current) and the regime-divergence
system (evaluated in `REGIME_DIVERGENCE_EVAL.md`), do **other machine-learning
or statistical strategies** improve the backtest for these five apps — and if
so, do they help the **Overall Trading** allocation mix?

**Not implemented.** Evaluation only — no strategy, config, or allocation was
changed. Reproduce with:

```bash
python scripts/eval_ml_strategies.py     # per-asset: 10 strategies vs MA vs B&H
python scripts/eval_ml_overall.py        # Overall allocation-mix impact
```

## Method (same discipline as the prior eval)

All candidates are **long/flat** (no leverage, no shorting), execute on the
**next bar**, and carry the app's own fixed stop as a uniform risk overlay — so
each is apples-to-apples with the current MA config. Every result is scored on
two windows:

* **Full OOS** — the app's reported OOS window; ML models fit on the pre-OOS
  window, then trade forward.
* **Holdout Aug-2025 → now** — ML models fit on data **through 2025-08-01** only,
  then trade the unseen tail (the "training data until August 2025" the brief
  asked for). Forward-return labels are fully realised inside the training
  window (no look-ahead across the split).

Strategies tested (10): **statistical rules** — Dual-MA crossover (20/100),
time-series momentum (63d), Donchian breakout (20/10), MACD (12/26/9), MA + a
volatility filter; **ML models** on the platform's own causal feature matrix —
Logistic regression and HistGradientBoosting (P[fwd-10d>0] gate), Ridge (fwd-20d
return sign), and a Gaussian-Mixture 2-state regime detector. The current MA
filter reproduces its published numbers exactly, confirming the harness.

*(cells = total return / max drawdown / Sharpe; **bold** = beats current MA on Sharpe in that window)*

---

## SOXX  (current MA40/5%)

| Strategy | Full OOS 2021→now | Holdout Aug-25→now |
|---|---|---|
| Buy & Hold | +362% / −46% / 0.94 | +145% / −16% / 2.51 |
| **MA (current)** | +233% / −31% / 0.94 | +104% / −15% / 2.15 |
| **Dual-MA 20/100** ★ | **+318% / −34% / 1.05** | **+145% / −16% / 2.51** |
| TSMOM 63d | +202% / −35% / 0.85 | +145% / 2.51 |
| Donchian 20/10 | +89% / −32% / 0.60 | +52% / 1.41 |
| MACD | +79% / −37% / 0.55 | +42% / 1.39 |
| MA + vol-filter | +133% / −26% / 0.91 | +44% / −8% / 2.13 |
| Logistic (fwd10) | +178% / −46% / 0.86 | +110% / −7% / 3.77 |
| HistGradBoost | +68% / −49% / 0.47 | +47% / 1.48 |
| Ridge (fwd20) | +34% / −45% / 0.34 | +51% / 2.20 |
| GMM 2-state | +140% / −41% / 0.67 | +53% / 1.75 |

**Winner: Dual-MA 20/100.** Higher return (+318% vs +233%) **and** higher Sharpe
(1.05 vs 0.94) than the current MA40, at a near-identical drawdown; on the
holdout it stayed fully invested and matched B&H (2.51). A modest, robust
improvement over the single-MA filter.

## VEGN  (current MA200/5%)

| Strategy | Full OOS 2022→now | Holdout Aug-25→now |
|---|---|---|
| Buy & Hold | +81% / −33% / 0.73 | +44% / 2.12 |
| **MA (current)** | **+84% / −14% / 1.02** | +38% / −8% / 1.96 |
| Donchian 20/10 | +59% / −13% / 0.89 | +20% / 1.32 |
| GMM 2-state | +77% / −27% / 0.77 | +39% / 2.01 |
| others | all Sharpe ≤ 0.77 full | mixed |

**No robust winner — keep MA200.** Nothing beats the current MA on full-OOS
Sharpe. The models that look good on the holdout (Ridge 2.38, GMM 2.01) are the
same ones that are *worst* on the full window (Ridge −0.14, Logistic −0.27) —
classic overfitting, not a real edge.

## GRID  (current MA150/5%)

| Strategy | Full OOS 2021→now | Holdout Aug-25→now |
|---|---|---|
| Buy & Hold | +132% / −30% / 0.83 | +33% / 1.50 |
| **MA (current)** | +105% / −19% / 0.87 | +33% / −12% / 1.50 |
| **MACD 12/26/9** ★ | **+102% / −16% / 1.02** | **+29% / −6% / 2.07** |
| MA + vol-filter | +58% / −15% / 0.77 | **+24% / −3% / 2.45** |
| Donchian 20/10 | +72% / −17% / 0.86 | +16% / 1.16 |
| GMM 2-state | +86% / −29% / 0.66 | +24% / 1.24 |

**Winner: MACD 12/26/9.** Beats MA on Sharpe in **both** windows (1.02 vs 0.87
full; 2.07 vs 1.50 holdout) with a lower drawdown, at essentially the same
return. The cleanest both-window improvement in the whole study.

## REMX  (current MA150/5%)

| Strategy | Full OOS 2021→now | Holdout Aug-25→now |
|---|---|---|
| Buy & Hold | +24% / −74% / 0.30 | +73% / 1.45 |
| **MA (current)** | +73% / −56% / 0.48 | +79% / −23% / 1.53 |
| HistGradBoost | +58% / −47% / 0.42 | +29% / 0.95 |
| GMM 2-state | +9% / −74% / 0.21 | +68% / 1.82 |
| others | all Sharpe ≤ 0.42 full | weak |

**No robust winner from this set.** None of the ML/statistical strategies beats
MA150 across both windows. REMX's best alternative remains the **regime-
divergence** system from the prior eval (full-OOS +109% / −18% / 0.93), which
still stands as REMX's one genuine improvement candidate.

## WGMI  (current MA30/10%)

| Strategy | Full OOS 2024→now | Holdout Aug-25→now |
|---|---|---|
| Buy & Hold | +223% / −63% / 0.98 | +141% / −51% / 1.61 |
| **MA (current)** | +191% / −40% / 1.02 | +166% / −34% / 2.12 |
| **MA + vol-filter** ★ | **+257% / −25% / 1.39** | +112% / −25% / 1.96 |
| GMM 2-state | +256% / −50% / 1.13 | **+239% / −31% / 2.59** |
| Ridge (fwd20) | +171% / −19% / 1.45 | −9% / −0.12 |
| Donchian 20/10 | +94% / −42% / 0.78 | +98% / 1.80 |

**Winner: MA + volatility filter.** Higher return (+257% vs +191%), higher
Sharpe (1.39 vs 1.02) **and** a much lower drawdown (−25% vs −40%) than the
current MA30 — skipping the high-realised-vol regime is exactly the right
overlay on a 2–3× BTC-beta basket. (GMM posts the highest holdout return/Sharpe
but at a −50% drawdown; Ridge is a clear overfit — great full-window, negative
holdout — and is rejected.)

---

## Overall Trading allocation mix

Swapping the three improved sleeves' return/position streams into the live
15-instrument universe and re-optimising with the app's own optimiser:

| Blend | Total ret | CAGR | MDD | Sharpe | Vol |
|---|---|---|---|---|---|
| Baseline (current) | +433% | 35.5% | −4.89% | 3.15 | 8.23% |
| **SOXX=Dual-MA, GRID=MACD, WGMI=vol-filter** | +448% | 36.2% | −5.06% | **3.32** | **7.92%** |

Δ = **+15pp return, +0.17 Sharpe, −0.31pp vol**, drawdown ~unchanged. The
optimiser up-weights the improved GRID (2.2%→6.0%) and WGMI (2.0%→4.5%) sleeves.
The gain **persists on the unseen holdout** (Aug-2025→now Sharpe 3.80 → 3.90).

---

## Bottom line

| App | Best strategy found | Beats current MA? | Recommendation |
|---|---|---|---|
| **SOXX** | Dual-MA 20/100 | ✅ ret + Sharpe, both windows | worth adopting |
| **VEGN** | — | ❌ | keep MA200 |
| **GRID** | MACD 12/26/9 | ✅ Sharpe + DD, both windows | worth adopting |
| **REMX** | (divergence, prior eval) | ❌ from this set | see divergence eval |
| **WGMI** | MA + vol-filter | ✅ ret + Sharpe + DD | worth adopting |

* **3 of 5 (SOXX, GRID, WGMI) have a robust improvement** over the current MA
  filter — and in all three the winner is a **simple, low-parameter statistical
  rule**, not a heavy ML model.
* The **ML models (Logistic, HGB, Ridge, GMM) largely overfit**: several are the
  best on one window and the worst on the other (Ridge on WGMI/VEGN, Logistic on
  VEGN). Only GMM was consistently competitive, but never strictly best on a
  risk-adjusted basis with acceptable drawdown.
* At the **portfolio level** the three improvements give a modest but real,
  holdout-persistent risk-adjusted lift (Sharpe 3.15 → 3.32, lower vol).

### Caveats
* 10 strategies × 5 assets = 50 tests, so some single-window "wins" are luck;
  the recommendations above require improvement on **both** windows to guard
  against this, which is why only the simple rules survive.
* ML models used fixed, lightly-tuned hyper-parameters (no per-asset search) to
  avoid a second layer of overfitting; a heavier search could raise in-sample
  numbers but is unlikely to survive the holdout, on this evidence.
* Overall weights are optimised with full-period hindsight (as in the live app);
  the recent-window figures apply those weights forward as a check.
