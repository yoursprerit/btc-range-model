# Overall Strategy — Out-of-Sample Walk-Forward Evaluation

**Document type:** Read-only robustness audit of the **Overall Trading** cross-asset
strategy. **No strategy, config, weights, or model were changed** — this only measures
how the *currently-implemented* Overall strategy compares to a genuinely
out-of-sample version of itself.

**Reproduce:** `python scripts/eval_oos_walkforward.py`

**Windows:** portfolio weights fit on **2021 only** (through 2021-12-31), frozen, and
applied **untouched to 2022-01-01 → now** (~4.5-year out-of-sample holdout).

---

## Question

The Overall app advertises, per risk profile (committed artifact, full period 2021→now):

| Profile | Advertised CAGR | Advertised total return | Sharpe | MaxDD |
|---|---|---|---|---|
| Balanced | **51.0%** | +866.7% | 3.22 | −9.8% |
| Growth | **73.8%** | +1,997% | 2.14 | −17.9% |
| Aggressive | **80.9%** | +2,522% | 1.80 | −24.1% |

**How much of this is genuine out-of-sample strategy, and how much is leverage, the
SATA idle-cash assumption, and weights fit on the very period being reported?**

## Method

* **The app's weights are in-sample by construction.** `optimize_weights` (in
  `app/overall_core.py`) fits the cross-asset weights on the **entire** 2021→now
  history and reports that same period; there is no train/test split anywhere (the
  app discloses this: *"Optimal weights are fit on this same history (in-sample)"*).
* **The out-of-sample (OOS) version** here fits the weights on **2021 only**, freezes
  them, and applies them untouched to 2022→now. The *signals* are already OOS (each
  asset's H/L model is trained on its pre-2021 window), so this isolates the effect of
  the **weight look-ahead**.
* **Ingredients are added back one at a time** so the headline can be decomposed:
  in-sample weights → the added crypto/miner names → **leverage** (2×/3× siblings) →
  the **SATA** idle-cash yield.
* **Crypto sleeve (BTC/MSTR/MSTU/WGMI) has no 2021 data**, so its weights cannot be
  OOS-fit. Where included, they are **imported from the app** (acknowledged hindsight)
  and the other 13 assets are fit OOS on 2021.
* **Sanity check:** recomputing the app's committed weights over the full period
  reproduces the artifact headline **exactly** (51.0 / 73.8 / 80.9% CAGR), confirming
  the reproduction engine is faithful.
* All figures are **gross of transaction costs**; leverage sleeves use **real ETF
  prices** (daily-rebalance decay is captured; intraday 3× gap/tail risk is not).

---

## 1. Decomposition — where the app's CAGR comes from (2022 → now)

Building the app up from the honest OOS baseline, one factor at a time (CAGR):

| Step | Balanced | Growth | Aggressive | Nature |
|---|---|---|---|---|
| **1. TRUE OOS** (unlev, 2021-fit wts, no SATA) | **14.3%** | **17.4%** | **17.0%** | genuine, generalizable |
| 2.  + in-sample weights (unlev) | 21.3% | 24.9% | 29.0% | weight look-ahead |
| 3.  + add BTC/MSTR/WGMI | 27.2% | 35.9% | 38.3% | added crypto/miner names |
| 4.  + leverage (2×/3×) | 38.3% | 56.8% | 65.2% | amplification (adds tail risk), not alpha |
| 5.  + SATA idle-cash yield | 56.7% | 75.9% | 84.5% | counterfactual assumption |
| **6. APP AS-IS** (committed wts + SATA) | **54.6%** | **78.2%** | **84.3%** | — |

Reading the Aggressive column, the app's ~84% CAGR ≈ **17.0 base + 14.5 SATA +
24.7 leverage + 3.7 crypto sleeve + 24.4 non-crypto in-sample weighting.** Two of those
five components (**SATA + weight look-ahead, ≈39 pp**) are non-repeatable; one
(**leverage, ≈25 pp**) is amplification available to any book at the cost of tail risk;
only **≈17 pp** is genuine out-of-sample strategy.

## 2. Final matched comparison — leverage + SATA + crypto sleeve on BOTH sides

The fairest like-for-like: leverage, the idle-cash yield, **and** the crypto sleeve (at
the app's own weights) applied to both, so the only remaining difference is the
in-sample weighting of the 13 non-crypto assets.

| Profile | | Total return | **CAGR** | Sharpe | MaxDD | Imported crypto wts |
|---|---|---|---|---|---|---|
| **Balanced** | OOS + crypto (rest OOS-fit) | +444% | **45.5%** | 3.23 | **−8.9%** | BTC 9 / MSTR 8 / MSTU 10 / WGMI 9 |
| | App as-is | +615% | **54.6%** | 3.19 | −9.8% | |
| **Growth** | OOS + crypto | +664% | **56.9%** | **2.43** | **−13.3%** | BTC 9 / MSTR 16 / MSTU 9 / WGMI 12 |
| | App as-is | +1,259% | **78.2%** | 2.25 | −17.9% | |
| **Aggressive** | OOS + crypto | +732% | **59.9%** | **2.05** | **−17.4%** | BTC 6 / **MSTR 24** / MSTU 2 / WGMI 4 |
| | App as-is | +1,479% | **84.3%** | 1.87 | −24.1% | |

**Residual raw-return gap: ~1.20× (Balanced) / 1.38× (Growth) / 1.41× (Aggressive)** —
i.e. 9 / 21 / 24 pp of CAGR — and that residual is **purely the in-sample weighting of
the non-crypto assets** (look-ahead that will not persist forward).

On **risk-adjusted** terms the OOS strategy **beats the app in Growth and Aggressive**
(higher Sharpe) and ties in Balanced — with **shallower drawdowns in all three**.

## 3. Trajectory — the gap as each ingredient is matched (Aggressive CAGR)

| What's matched | OOS CAGR | App CAGR | Gap |
|---|---|---|---|
| nothing (unlev, no SATA) | 17.0% | 84.3% | **5.0×** |
| + SATA | 31.5% | 84.3% | 2.7× |
| + leverage | 56.2% | 84.3% | 1.5× |
| + crypto sleeve | 59.9% | 84.3% | **1.41×** |

Each ingredient the app uses, once applied to the honest strategy too, closes most of
the distance — leaving only the in-sample weight fitting.

---

## Findings

1. **The genuine, out-of-sample, unleveraged, no-idle-yield engine is ~14–17% CAGR** —
   a *good* risk-adjusted strategy (Sharpe ~1.0–1.7, drawdowns −6% to −11%), but a
   fraction of the advertised 51–84%.
2. **The advertised CAGR is manufactured mostly by leverage + the SATA assumption +
   in-sample weight fitting**, not by out-of-sample skill. For Aggressive, ≈4/5 of the
   headline CAGR comes from those three, only ≈1/5 from the honest engine.
3. **Once leverage, SATA, and the crypto sleeve are matched on both sides, the app's
   out-of-sample-defensible CAGR is ~46–60%, not 55–84%**, and the ~1.2–1.4× residual is
   pure weight look-ahead.
4. **The OOS strategy is equal-or-better risk-adjusted** than the app (higher Sharpe in
   the aggressive profiles, shallower drawdowns throughout) — the app's extra return is
   bought with more drawdown, not better quality per unit of risk.

## Caveats

* **All numbers (OOS and app) are measured over a favourable multi-asset bull cycle**
  (2022–2026, especially the 2024–26 rip); they are optimistic as forward estimates.
* **SATA** is credited as a flat ~13%/yr at $100 par across all history — a
  counterfactual (a variable-rate perpetual preferred whose price can fall from par).
* **Leverage sleeves** carry intraday gap/tail risk a daily-close backtest understates.
* **No transaction costs / slippage** are modelled.
* The **crypto-sleeve weights are imported from the app** (hindsight) — they cannot be
  out-of-sample-validated (no 2021 data), so the 46–60% "OOS + crypto" figure is an
  **upper bound** on the honest OOS number.

## Bottom line

The Overall app's headline (51–84% CAGR) is not fraudulent, but it is **in-sample by
construction and heavily levered**. Stripped to a genuine out-of-sample, like-for-like
basis, the strategy's defensible return is **~46–60% CAGR with leverage + the idle-cash
yield, ~30% with the idle-cash yield alone, and ~15–17% as a plain unleveraged
engine** — and on a risk-adjusted basis the honest, frozen-weight version is **at least
as good as** the currently-implemented one. The remaining edge the app shows over its
own out-of-sample twin is **in-sample weight fitting**, which does not persist forward.
