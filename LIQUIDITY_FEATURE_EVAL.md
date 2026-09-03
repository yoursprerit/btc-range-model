# Net global liquidity as a feature — evaluation for the BTC daily High/Low model

**Status: tested and rejected (2026-09).** Net liquidity does not improve the
daily High/Low model. On the production feature set and production learners it
*costs* ~2.3 bp of mean absolute error (p = 0.034) — statistically
indistinguishable from the cost of adding the same number of **random** columns
(1.4 bp). On eight years of history the verdict is sharper still: liquidity
costs 1.27 bp (p = 0.0005) where the random columns cost nothing measurable
(p = 0.37) — it is worse than noise. No production feature set, model or config
was changed.

**Question.** The daily High/Low model (`src/pipeline_ct.py`) currently sees BTC
price/volume, seven cross-asset macro series, eleven blockchain.info on-chain
series, Fear & Greed and the Coinbase premium. It does **not** see central-bank
liquidity in any form. Should it? Specifically: does *true net liquidity* —
Fed balance sheet minus the Treasury General Account minus overnight reverse
repo, optionally plus the ECB — improve next-bar high/low accuracy?

Reproduce with:

```bash
python src/research_liquidity_features.py                      # main ablation, 2024-2026
python src/research_liquidity_features.py --dataset extended   # 2018-2026 robustness
python src/research_liquidity_horizon.py                       # value by forecast horizon
```

## Method

**The series.** `src/liquidity_data.py` builds

```
net_liq    = WALCL − WTREGEN − RRPONTSYD        (Fed assets − TGA − overnight RRP)
global_liq = net_liq + ECBASSETSW × USD/EUR
```

from FRED, cached under `data/liquidity/`. The other major central banks are
monthly and land on FRED weeks late, so a *same-day-honest* global aggregate is
the Fed plus the ECB plus a stale constant — adding the BoJ and PBoC at their
real publication lag would contribute almost no timely variation.

**Point-in-time discipline.** Every FRED observation carries its *observation*
date, not its release date. H.4.1 for the week ending Wednesday is not published
until Thursday 16:30 ET, and the ON-RRP print for day D lands ~13:15 ET on D —
both after the 12:00-UTC bar the model scores. Each series is therefore shifted
to the first bar that could genuinely have seen it, plus a further safety day.
Skipping this manufactures an edge that does not survive live trading.

**Features.** Levels are non-stationary, so nothing uses a raw level: 1w/4w/13w
log changes, position in a 1-year range, an acceleration term, the volatility of
liquidity flows, and the two fast-moving components (TGA, RRP) that traders
actually watch — twelve columns in total.

**Evaluation.** Walk-forward rather than a single train/val/test cut: expanding
train window, refit every 21 days, always scoring days the model has never seen.
Arms are compared paired on identical rows with a Diebold-Mariano test
(Newey-West HAC, lag 5) plus a 21-day block bootstrap. The production
climatology blend (α) is replicated with α calibrated only on strictly earlier
folds, because that shrink is what actually ships and it damps whatever a new
feature block does.

**The control that matters.** Every comparison carries a `base+noise` arm — the
same *number* of persistence-matched random columns. Without it, "adding
liquidity costs 2 bp" is unreadable: you cannot tell a useless signal from the
generic cost of widening the feature matrix.

---

## 1. There is no univariate relationship worth the name

Spearman correlation against tomorrow's excursion, 2024-01 → 2026-08 (n = 946),
with Newey-West t-statistics:

| feature | ρ(range) | t | ρ(tilt) | t |
|---|---|---|---|---|
| `liq_d7` | 0.057 | 1.17 | −0.009 | −0.05 |
| `liq_d28` | 0.025 | 0.20 | −0.029 | −0.76 |
| `liq_d91` | −0.007 | −0.32 | −0.002 | 0.63 |
| `liq_z364` | −0.101 | −1.02 | 0.016 | 1.15 |
| `liq_vol28` | −0.123 | −1.86 | 0.004 | 0.25 |
| `liq_tga_d7` | 0.070 | 1.64 | 0.003 | −0.47 |
| `liq_walcl_d28` | **−0.182** | −3.72 | 0.009 | 1.05 |
| `liq_g_z364` | **−0.158** | −3.30 | −0.017 | −0.09 |

Two things stand out. **Directionally it is dead**: every correlation with the
high/low *tilt* is ≤ 0.034 in absolute value. Whatever liquidity does, it does
not tell you whether tomorrow breaks up or down. And the largest range
correlation (0.182) is barely clear of the 0.128 that *persistence-matched
random columns* reach on the same sample.

## 2. …and what relationship there is flips sign between regimes

Spearman(feature, next-day range) by calendar year on the 2018-2026 sample:

| feature | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | flips |
|---|---|---|---|---|---|---|---|---|---|---|
| `liq_d91` | 0.03 | −0.01 | 0.27 | 0.23 | −0.41 | 0.21 | 0.11 | −0.03 | 0.13 | 6 |
| `liq_z364` | 0.10 | −0.02 | 0.23 | −0.04 | −0.14 | 0.09 | 0.08 | −0.15 | −0.30 | 5 |
| `liq_walcl_d28` | −0.33 | −0.09 | 0.38 | 0.03 | 0.32 | 0.03 | −0.06 | −0.23 | 0.05 | 3 |
| `liq_g_z364` | 0.07 | −0.11 | 0.09 | 0.06 | −0.11 | 0.30 | 0.00 | −0.33 | 0.22 | 6 |
| **liquidity block, mean** | | | | | | | | | | **4.1 / 8** |
| `atr_14` *(for scale)* | 0.70 | 0.45 | 0.48 | 0.47 | 0.48 | 0.31 | 0.26 | 0.39 | 0.47 | **0** |
| `range_ma7` *(for scale)* | 0.72 | 0.54 | 0.52 | 0.50 | 0.49 | 0.34 | 0.29 | 0.44 | 0.46 | **0** |
| `vol_20` *(for scale)* | 0.72 | 0.39 | 0.52 | 0.43 | 0.38 | 0.30 | 0.17 | 0.35 | 0.42 | **0** |

This is the heart of it. The whole-sample correlation of ~0.2 that makes
liquidity look promising is an average over years that individually run from
−0.41 to +0.38. A real range predictor does not behave this way: `atr_14`,
`range_ma7` and `vol_20` never flip sign across nine years and never drop below
+0.17. A predictor whose *sign* is regime-dependent cannot be relied on out of
sample, because the model can only learn the sign the training window happened
to contain.

## 3. Walk-forward ablation — production learners, production feature set

2024-01 → 2026-08, 446 out-of-sample days over 21 refits:

| arm | MAE hi | MAE lo | MAPE_H | MAPE_L | dir hit | Δloss | DM t | p | boot win |
|---|---|---|---|---|---|---|---|---|---|
| **base** | 124.8 bp | 130.1 bp | 1.221% | 1.335% | 52.0% | — | — | — | — |
| base + liq | 130.6 bp | 127.8 bp | 1.280% | 1.311% | 50.4% | +1.71 bp | 0.95 | 0.342 | 22.2% |
| base + liq core | 131.5 bp | 128.6 bp | 1.289% | 1.320% | 49.8% | +2.56 bp | 1.96 | 0.050 | 4.0% |
| base + **noise** | 131.3 bp | 135.2 bp | 1.284% | 1.388% | 48.0% | +5.77 bp | 3.77 | 0.000 | 0.1% |
| liq only | 118.3 bp | 119.1 bp | 1.157% | 1.228% | 49.3% | −8.76 bp | −2.64 | 0.008 | 98.5% |
| climatology | 118.5 bp | 122.6 bp | — | — | — | — | — | — | — |

And the same comparison **after the production α-blend**, which is what actually
ships:

| arm | MAE hi | MAE lo | Δloss | DM t | p | boot win |
|---|---|---|---|---|---|---|
| **base** | 117.1 bp | 123.4 bp | — | — | — | — |
| base + liq | 122.1 bp | 123.1 bp | **+2.32 bp** | 2.11 | **0.034** | 4.7% |
| base + liq core | 121.5 bp | 122.7 bp | +1.86 bp | 2.01 | 0.045 | 1.9% |
| base + **noise** | 118.4 bp | 124.9 bp | +1.38 bp | 2.02 | 0.043 | 3.0% |
| liq only | 118.1 bp | 119.2 bp | −1.62 bp | −0.79 | 0.428 | 68.5% |

**Adding liquidity is significantly harmful (p = 0.034), and it hurts by about
as much as adding twelve random columns (2.32 bp vs 1.38 bp).** That equivalence
is the finding: the liquidity block is paying the capacity cost of twelve extra
features and returning nothing for it.

### The `liq_only` result is not a point in liquidity's favour

Read carelessly, "liquidity alone beats the full model by 8.8 bp (p = 0.008)"
looks like a discovery. It is the opposite. A model whose features carry no
usable signal cannot do anything but regress to the training mean — and
`liq_only` lands within a basis point of pure climatology (118.3/119.1 vs
118.5/122.6). What the row actually shows is that on this window the **base**
model underperforms its own climatology (+6.92 bp, p = 0.022), which is exactly
the condition the pipeline's α-blend exists to handle. Once α is applied,
`liq_only`'s apparent advantage disappears (p = 0.428). The liquidity features
are not predicting; they are failing to predict quietly.

## 4. Eight years of history gives the same answer, more sharply

The 2024-2026 window is thin for a weekly macro series (~145 independent
liquidity releases), so the study was repeated on 2018-06 → 2026-09 using
Coinbase 12:00-UTC bars and FRED macro equivalents — a sample that contains the
2020-21 QE surge and the 2022-23 QT drain, the regimes where liquidity's
relationship with BTC is supposed to be strongest. 2,027 out-of-sample days:

| arm | MAE hi | MAE lo | dir hit | clf acc | Δloss (α-blended) | DM t | p | boot win |
|---|---|---|---|---|---|---|---|---|
| **base** | 142.8 bp | 143.1 bp | 56.2% | 55.6% | — | — | — | — |
| base + liq | 145.5 bp | 143.1 bp | 55.7% | 53.7% | **+1.27 bp** | 3.51 | **0.0005** | 0.2% |
| base + liq core | 144.2 bp | 142.7 bp | 55.1% | 54.4% | +0.49 bp | 2.03 | 0.043 | 4.7% |
| base + **noise** | 142.6 bp | 144.1 bp | 55.9% | 54.3% | +0.29 bp | 0.90 | 0.367 | 21.6% |
| liq only | — | — | — | — | +25.14 bp | 14.61 | 0.000 | 0.0% |

This is the cleanest read in the study, and it is **worse for liquidity than the
short window**. With enough data that twelve extra random columns no longer cost
anything measurable (`base+noise`, p = 0.37), the liquidity block still does
significant damage (p = 0.0005). Liquidity is not merely inert here — it is
worse than noise. It also degrades the direction classifier, 55.6% → 53.7%.

Note too that the univariate correlations on this sample *are* individually
significant (`liq_walcl_d28` ρ = 0.208, t = 2.53) and still do not translate into
out-of-sample accuracy. That gap between "statistically detectable in-sample"
and "useful out-of-sample" is exactly what section 2 predicts.

### This also resolves the `liq_only` oddity

On the short window `liq_only` appeared to beat `base`, which needed the
explanation in section 3. The long sample settles it outright: here `base` beats
its own climatology by 27.5 bp (t = −14.43) and `liq_only` is 25.1 bp **worse**
than base. Given a sample large enough to fit on, the model built purely from
liquidity features is close to the worst thing you can build. The short-window
result was an artifact of that window, not a property of the features.

The full-sample importance ranking is also sane on this sample — `range_ma7`,
`atr_7`, `range_today`, `atr_14`, `range_ma30`, `vix_ret_1` take the top slots,
rather than a random column placing 2nd as it did on 946 rows. The liquidity
block still ranks below noise: median 53/95 vs 38/95, summed importance 0.085
vs 0.115.

## 5. It is not a horizon mismatch either

The obvious defence is that a weekly macro series has no business predicting one
day ahead. The repo also ships 7-day and 14-day cone models, so the sweep tests
h = 1…30 on both a range target and a return target:

| h | range: Δliq | p | | return: Δliq | p |
|---|---|---|---|---|---|
| 1 | −1.24% | 0.41 | | −2.10% | 0.13 |
| 5 | −2.18% | 0.29 | | +1.12% | 0.70 |
| 10 | −3.00% | 0.32 | | −3.72% | 0.20 |
| 20 | −5.57% | 0.08 | | −6.19% | 0.11 |
| 30 | **+12.45%** | **0.003** | | −3.66% | 0.30 |

Nothing reaches significance in liquidity's favour at any horizon. The one
significant result is at h = 30 and says liquidity makes the model **worse**.
The improvement that drifts in around h = 20 and then reverses at h = 30 is the
shape of noise, not of a horizon effect.

## 6. The model ranks liquidity below random noise

Fitting a GBM on the base + liquidity + noise matrix and ranking all 140 columns
by importance:

| block | median rank (of 140) | summed importance |
|---|---|---|
| liquidity (12 cols) | 88 | 0.062 |
| **random noise (12 cols)** | **52** | **0.102** |

The learner finds the random columns *more* useful than the liquidity columns.
(That `noise_6` places 2nd overall is itself a caution about reading full-sample
GBM importances too literally — which is why the walk-forward tests above, not
this table, carry the verdict.)

---

## Why this is the expected result

The target is the size of a **one-day** excursion. That is overwhelmingly a
volatility-clustering problem, and it is already well served by `atr_*`,
`range_ma*`, `vol_*` and the y-target EMAs — features with stable correlations
of 0.3-0.7. Net liquidity updates weekly, moves ~0.5% a week, and transmits to
risk assets over months through valuation and positioning. There is no mechanism
by which last Wednesday's H.4.1 print tells you whether tomorrow's BTC bar
travels 1.2% or 2.4%, and the data agrees.

The liquidity thesis is a **multi-month, directional, level-of-price** argument.
This model is a **one-day, magnitude** forecaster. The mismatch is structural,
not a matter of better feature engineering.

## What would change this answer

* A **positioning/leverage** dataset rather than a balance-sheet one —
  perpetual funding rates, open interest, aggregate liquidation levels, stablecoin
  net issuance. These update continuously and are mechanically tied to next-day
  range in a way central-bank reserves are not.
* Liquidity as a **regime label** feeding allocation (see `OVERALL_STRATEGY.md`)
  rather than as a feature inside the daily H/L regressor — a different question
  from the one tested here, and the only one this evaluation leaves open.
* A materially longer sample. Even eight years is only ~430 independent weekly
  liquidity observations; the sign-flip table suggests that would mostly buy more
  evidence of instability.
