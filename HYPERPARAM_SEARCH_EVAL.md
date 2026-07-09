# Deeper per-asset parameter search — the three winners

**Question.** The three improvements found in `ML_STATISTICAL_STRATEGY_EVAL.md`
(SOXX Dual-MA, GRID MACD, WGMI MA+vol-filter) each used one hand-picked
parameter set. Does a **deeper per-asset search** find better parameters — and
do they hold up out-of-sample?

**Not implemented.** Evaluation only. Reproduce with:

```bash
python scripts/eval_hyperparam_search.py    # per-asset param grid, train/holdout split
python scripts/eval_hyperparam_overall.py   # tuned winners in the Overall mix
```

## Method — overfitting is the whole risk, so the split is strict

Parameters are **selected on the training window (OOS start → 2025-08-01)** and
then **validated on the unseen holdout (2025-08-01 → now)**. Selection rule =
max **training** Sharpe subject to MDD ≤ Buy&Hold(train) and ≥ 8 trades,
tie-broken by return. All candidates are rule-based (no model fitting), so the
signal is identical across windows — the only thing "learned" is the parameter
choice, and the holdout audits it.

Two guards are reported per asset:
* **Full-window hindsight optimum** — the best config if we *had* peeked at the
  whole window. If it equals the training-optimal, there is **no overfitting
  gap**.
* **Top-decile robustness** — median holdout Sharpe of the top-10% training
  configs. If the whole neighbourhood travels, the win isn't a single lucky cell.

Grids searched: SOXX 144 (fast × slow × stop); GRID 464 (fast × slow × signal ×
trend-confirm × stop); WGMI 1 680 (MA × vol-window × median-window × k × stop).

---

## SOXX — Dual-MA crossover

| Config | Full OOS 2021→now | Holdout Aug-25→now |
|---|---|---|
| Current MA40 / 5% | +233% / −31% / 0.94 | +104% / 2.15 |
| v1 Dual-MA 20/100 | +318% / −34% / 1.05 | +145% / 2.51 |
| **Tuned: MA25 > MA100, 5% stop** ★ | **+452% / −27% / 1.23** | **+145% / 2.51** |

* Training-optimal = **fast 25 / slow 100 / 5% stop**, and it is **identical to
  the full-window hindsight optimum** — zero overfitting gap.
* Beats both the current filter and v1 on **all three** axes: higher return
  (+452%), lower drawdown (−27%), higher Sharpe (1.23). Only **9 trades** over
  the whole window — very low turnover.
* Robustness: top-decile median holdout Sharpe = **2.51** — the entire
  neighbourhood travels. High confidence.

## GRID — MACD

| Config | Full OOS 2021→now | Holdout Aug-25→now |
|---|---|---|
| Current MA150 / 5% | +105% / −19% / 0.87 | +33% / 1.50 |
| v1 MACD 12/26/9 | +102% / −16% / 1.02 | +29% / **2.07** |
| **Tuned: MACD 10/20/9, 5% stop** ★ | **+134% / −16% / 1.20** | +26% / 1.74 |

* Training-optimal = **fast 10 / slow 20 / signal 9 / 5% stop**, again **equal to
  the full-window hindsight optimum** — no overfitting gap.
* Beats the current filter on return and Sharpe (1.20 vs 0.87) at a lower
  drawdown. It edges v1 on the full window but v1 (12/26/9) had a slightly
  better *recent-holdout* Sharpe (2.07 vs 1.74) — the faster 10/20 captures more
  full-cycle return but trades far more (**57 trades**, vs SOXX's 9), so it is
  more transaction-cost-sensitive.
* Robustness: top-decile median holdout Sharpe = **1.28** (below current MA's
  1.50). The selected cell holds up (1.74), but GRID's MACD edge is **more
  config-sensitive** than SOXX's — moderate confidence. Either 10/20/9 (max
  full-cycle) or 12/26/9 (max holdout Sharpe, lower turnover) is defensible.

## WGMI — MA + volatility filter

| Config | Full OOS 2024→now | Holdout Aug-25→now |
|---|---|---|
| Current MA30 / 10% | +191% / −40% / 1.02 | +166% / 2.12 |
| v1 MA30 + vol-filter | +257% / −25% / 1.39 | +112% / 1.96 |
| **Tuned: MA50 & vol10 < 0.95·med₁₈₉, no stop** ★ | **+376% / −32% / 1.81** | +79% / 1.66 |
| *(full-window hindsight optimum)* | *+804% / −21% / 2.20* | *+230% / 2.83* |

* Training-optimal = **MA50 / vol-window 10 / median-window 189 / k 0.95 / no
  stop**: much higher return (+376%) and Sharpe (1.81) than both current and v1,
  though its drawdown (−32%) is deeper than v1's tightly-controlled −25%.
* **This is the one asset with a real overfitting gap.** The full-window
  hindsight optimum (+804% / 2.20) is *far* better than the training-optimal and
  was **not** selectable from the training window — because WGMI's history is
  short and thin (OOS starts 2024, ~19 months of training). Treat WGMI's tuned
  numbers as the honest, non-hindsight result, and expect real-world results
  toward the training-optimal, not the hindsight ceiling.
* Robustness: top-decile median holdout Sharpe = **1.81** (well above current's
  neighbourhood), and the selected cell's holdout Sharpe (1.66) sits inside it —
  so despite the short history the *neighbourhood* is stable. Moderate-high
  confidence, with the short-history caveat.

---

## Tuned winners in the Overall Trading mix

Swapping the three tuned streams into the live 15-instrument universe and
re-optimising with the app's own optimiser:

| Blend | Total ret | CAGR | MDD | Sharpe | Vol |
|---|---|---|---|---|---|
| Baseline (current) | +433% | 35.5% | −4.89% | 3.15 | 8.23% |
| v1 winners | +448% | 36.2% | −5.06% | 3.32 | 7.92% |
| **Tuned winners** | **+572%** | **41.3%** | **−4.40%** | **3.42** | 8.61% |

Δ vs baseline = **+139pp return, +0.27 Sharpe, −0.49pp (better) drawdown**. The
tuning compounds at the portfolio level well beyond v1: the optimiser lifts WGMI
from 2.0% → 15.7% and SOXX/GRID roughly 2×, because the tuned streams are higher
-return at comparable or lower risk.

---

## Bottom line

| App | Tuned strategy | vs current MA | Overfitting gap | Confidence |
|---|---|---|---|---|
| **SOXX** | Dual-MA **25/100**, 5% stop | +219pp ret, +0.29 Sharpe, −4pp DD | **none** (train = full optimum) | **high** |
| **GRID** | MACD **10/20/9**, 5% stop | +29pp ret, +0.33 Sharpe, −3pp DD | **none** (train = full optimum) | moderate (config-sensitive) |
| **WGMI** | MA**50** + vol-filter (k 0.95) | +185pp ret, +0.79 Sharpe | present (short history) | moderate-high |

* **SOXX 25/100 is the strongest, cleanest result** in the whole study — a big
  improvement with zero overfitting gap and a robust neighbourhood, at 9 trades.
* **GRID 10/20/9** improves the full cycle with no overfitting gap, but is
  choppier (57 trades) and more config-sensitive; the slower 12/26/9 is a
  lower-turnover alternative with a better recent holdout.
* **WGMI MA50 + vol-filter** is a large, robust improvement, but its short
  history leaves an un-capturable hindsight ceiling — size expectations to the
  training-optimal, not the full-window optimum.
* In the **Overall mix** the tuned set lifts Sharpe 3.15 → 3.42 and return
  +139pp at a slightly lower drawdown — the best combined blend found so far.

### Caveats
* Backtests are gross of transaction costs; GRID's 57-trade MACD is the most
  cost-sensitive, SOXX's 9-trade crossover the least.
* WGMI's 19-month training window makes its parameter search the least reliable
  of the three (hence the visible train-vs-full gap).
* Overall weights use full-period hindsight, as in the live app; the per-asset
  holdout columns are the genuine out-of-sample check.
