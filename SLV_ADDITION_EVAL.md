# Adding SLV (silver) as a new app to the Overall strategy — evaluation

**Question.** Should **SLV** (iShares Silver Trust — the 1× spot-silver ETF) be
added as a new forecasting/trading app, with the "most suitable, robust, optimal
and profitable" strategy, and does folding it into the Overall combined portfolio
**improve returns and/or risk-adjusted performance**?

**Answer up front: no.** SLV is the weakest signal asset in the candidate
universe — no strategy beats its own buy-&-hold on return, and its best Sharpe
(~0.54) is below every instrument already traded. It is also ~0.78 correlated to
a precious-metals sleeve the book **already covers with four instruments**
(GLDM · GDX · UGL · NUGT). So it is **return-dilutive** on a max-return mandate
and **cannot lift Sharpe** on a risk-adjusted mandate (a higher-Sharpe substitute
for the same exposure is already in the book). **Not recommended — do not
implement.** This doc is analysis only; nothing here touches `ticker_config.py`
or `overall_core.py`.

Reproduce the sleeve sweep + redundancy numbers (fully offline, committed data):

```bash
python scripts/eval_slv_add.py
```

---

## Method

* SLV daily OHLC is already cached (it is a REMX / GLDM macro driver) in
  `data/remx/macro_daily.csv` and `data/gldm/gldm_macro_daily.csv` as `slv_*`
  columns, **2015-01-02 → 2026-07-06** — a full bull+bear silver cycle.
* The candidate strategy is chosen by the **same sweep the other ETF apps use**
  (`backtest_ticker.sweep_asset`'s tiered objective: beat B&H in the loss
  periods, drawdown ≤ B&H in every period, retain ≥55% of B&H upside, then
  maximise full-OOS return tie-broken by Sharpe), driven through the repo's real
  engine (`simulate_regime` / `trend_long_array` / `_metrics`).
* Search space extends the built-in sweep across the whole trend family:
  `ma` (10 windows), `dual_ma` (8 fast/slow pairs), `macd` (4 param sets),
  `ma_vol` (4 windows × 4 vol-k), each × {−5% stop, no stop}.
* **Scope caveat.** Only the **price-driven trend engines** are swept. The
  **`divergence` Pure-Regime** engine needs the trained daily High/Low model,
  which this offline eval does not build — see *Caveats*. The trend result is
  already decisive, and divergence is very unlikely to clear the bar (below).

Window for the sleeve: full OOS **2021-01-01 → 2026-07-06**, same as every app.

---

## Result 1 — SLV is a poor trend/momentum asset (weakest sleeve in the book)

Silver spent the OOS window in choppy, mean-reverting ranges — the regime that
punishes trend-following hardest. **No configuration beats SLV's own
buy-&-hold on return, and the best Sharpe of any config is ~0.54.**

**SLV sleeve, OOS 2021→now (repo engine):**

| SLV strategy | Return | Max DD | Sharpe | Win-rate | Trades |
|---|---:|---:|---:|---:|---:|
| **Buy & Hold** | **+121%** | −51% | 0.59 | — | — |
| Best *return* — Dual-MA 20/50, −5% stop | +77% | −45% | 0.50 | 24% | 25 |
| Dual-MA 50/150, −5% stop | +80% | −45% | 0.50 | 25% | 12 |
| MA-150, −5% stop | +64% | −42% | 0.45 | 26% | 27 |
| **Best *risk-adjusted* — MA-50 + vol-filter (k 0.95), −5% stop** | **+39%** | **−16%** | **0.54** | **57%** | 58 |
| MA-100 + vol-filter (k 0.95) | +40% | −19% | 0.53 | 59% | 59 |

Two honest readings, both bad for an "improve returns / performance" mandate:

* **Return lens:** the best trend strategy returns **+77% vs buy-&-hold +121%** —
  the filter *loses* to simply holding silver, at a still-ugly −45% drawdown and a
  24% win-rate. This is the opposite of every shipped app, where the tuned
  strategy beats or risk-improves on B&H.
* **Robustness lens:** the only genuinely *robust* config is the WGMI-style
  **MA-50 + volatility filter** — it does what it should (cuts drawdown −51% →
  −16%, win-rate 57%), but caps return at **+39%**, a third of B&H.

Either way, **SLV's best Sharpe (~0.54) is the lowest in the whole universe.**
For comparison, the sleeves already traded: WGMI **1.81**, ERX **1.40**, XLE
**1.37**, SOXX **1.23**, NUGT **1.21**, GRID **1.20**, ARTY **0.99**. Silver is
not close to the quality bar the book is built on.

*(Full-history 2015→now is no better on quality: SLV B&H +271%; the best trend
config, Dual-MA 20/50, +175% at −45% DD / Sharpe 0.48 — again under-returning B&H
with a middling Sharpe.)*

---

## Result 2 — SLV is redundant with the gold sleeve the book already carries

SLV's daily returns are **0.72–0.79 correlated to the book's existing
precious-metals instruments** (OOS 2021→now):

| SLV ↔ | GLDM (gold) | GDX (gold miners) | UGL (2× gold) | Gold futures | SPX | Copper | SOXX | BTC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| corr | **+0.78** | +0.76 | **+0.79** | +0.72 | +0.26 | +0.50 | +0.27 | +0.08 |

Silver is, statistically, **a noisier version of gold**. The one thing that could
make a metals sleeve valuable — being *uncorrelated* to the tech/crypto-heavy book
(SLV↔SOXX 0.27, SLV↔BTC 0.08) — is **already delivered by the gold sleeve**, which
is the single most-covered exposure in the portfolio: **four instruments**
(GLDM `core`, GDX `beta`, UGL `lev`, NUGT `lev`), with NUGT alone taking **18.4%**
of the Aggressive optimum. SLV would add a fifth, lower-Sharpe entrant to an
exposure that is already saturated — and provide *less* of the same diversification
than the incumbents do, at a worse risk-adjusted price.

---

## Result 3 — the blend impact: dilutive on return, ~zero weight on Sharpe

SLV is a **1× `core`** instrument (unleveraged, cap 30% Balanced/Growth, 35%
Aggressive). Against the current book (cached `data/overall/overall_results.json`,
OOS 2021→now):

| Baseline blend | Return | Sharpe | Max DD |
|---|---:|---:|---:|
| Equal-weight strategy book (clean, no optimiser) | **+843%** | **2.74** | −12% |
| Balanced optimum | +836% | 3.17 | −9% |
| Growth optimum | +1950% | 2.08 | −18% |
| Aggressive optimum | +2472% | 1.77 | −24% |
| Best single sleeve (NUGT) | +1183% | 1.35 | −28% |

Both objectives reject SLV on first principles — the same logic the repo's own
`SOXL_ERX_ADDITION_EVAL.md` established:

* **Max-return mandate (Growth / Aggressive) → dilutive.** The eval doc's rule:
  *"any 1× diversifier dilutes total return … to raise return you need a sleeve
  whose own strategy out-returns the book."* The book compounds at **+843%**
  equal-weight (**+2472%** Aggressive); SLV's best sleeve is **+39% to +77%**.
  Adding a sub-book, unleveraged return stream can only pull the blended return
  **down**. Silver has no leveraged sibling in the book, so — unlike SOXL/ERX —
  there is no return-amplifying route to offset it.

* **Risk-adjusted mandate (Balanced) → ~zero weight, no Sharpe gain.** A
  diversifier only lifts portfolio Sharpe if it has **high own-Sharpe** *or*
  **low correlation to the book**. SLV has **neither that the book lacks**: its
  own Sharpe (0.54) is the universe's lowest, and its diversifying (gold-like)
  exposure is already held via higher-Sharpe instruments (NUGT 1.21, UGL). Under
  the near-max-Sharpe objective the optimiser squeezes low-Sharpe cores to a
  sliver already — REMX **0.5%**, PBW **0.6%**, OIH **0.2%** of the Aggressive
  optimum — and SLV, being *both* low-Sharpe *and* redundant, would land in the
  same ~0% bucket. A blend weight of ~0 means **no measurable improvement to
  return, Sharpe or drawdown** — it is search noise, exactly the outcome flagged
  for the rejected 1× candidates (QQQ, DBA, DBMF) in the leveraged-add eval.

**Net:** SLV either **lowers** total return (if the caps force it any weight) or
earns **~0 weight** (if the optimiser is free) — in neither case does it improve
returns *or* performance.

---

## Verdict

**No — adding SLV does not improve returns or performance, on any profile.**

* **Profitability / signal quality:** SLV is the worst-behaved asset tested. Its
  best strategy under-returns its own buy-&-hold (+77% vs +121%), and its top
  Sharpe (~0.54) is below every instrument already in the book. There is no
  "suitable, robust, optimal and profitable" strategy to be found here — silver's
  mean-reverting chop simply doesn't carry a tradeable trend edge in this window.
* **Return contribution:** as an unleveraged 1× core returning a fraction of the
  book, it is strictly **return-dilutive** on the Growth/Aggressive mandates.
* **Risk-adjusted contribution:** it is **0.78-redundant** with an already
  four-instrument gold sleeve and has the lowest own-Sharpe in the universe, so a
  Sharpe-aware optimiser assigns it **~0 weight** — no Sharpe or drawdown benefit.
* **Best available strategy, for the record:** if silver *had* to be traded, the
  **MA-50 + volatility filter (k 0.95, −5% stop)** is the most robust config
  (−16% DD, 57% win-rate, Sharpe 0.54) — but it still returns only +39% and does
  not change the portfolio verdict.

**Recommendation: do not add SLV.** If exposure to the silver leg of the metals
complex is ever desired, it is better expressed through the **existing gold
sleeve** (higher-Sharpe, same diversification) than by bolting on a lower-quality
fifth precious-metals instrument.

---

## Caveats & the one unrun check

* **Divergence engine not swept.** The `divergence` Pure-Regime engine needs the
  trained daily High/Low model and was not built offline here. It is the engine
  that wins on **boom-bust *crash* assets** (PBW, ARTY, REMX) where buy-&-hold
  rode a multi-year bear all the way down. Silver's OOS buy-&-hold is **+121%
  (up, not a crash)** and its price action is mean-reverting chop, not the clean
  momentum-divergence pattern the Pure-Regime system exploits — so it is very
  unlikely to clear the ~1.2 Sharpe bar the trend family already fell far short
  of. **If a confirmation is wanted before closing this out**, the single check
  is: train the SLV H/L model (`src/tickers/train_ticker.py SLV` with an SLV
  config) and run `backtest_ticker.py SLV --sweep` — the divergence rows would
  have to beat Sharpe ~0.54 *and* clear ~1.2 to change anything, which nothing in
  the trend family came close to.
* **A leveraged silver sibling (AGQ 2×) would not rescue it either.** SOXL/ERX
  added return because they lever a **high-Sharpe parent signal** (SOXX 1.23, XLE
  divergence 1.37). SLV's parent signal is a ~0.50 Sharpe coin-flip; 2×-levering
  a weak edge amplifies the decay and drawdown, not the edge. AGQ-on-SLV is not
  analogous to SOXL-on-SOXX.
* **In-sample / single-asset caveats** are the usual ones (see `OVERALL_STRATEGY.md`
  §9). None of them flatter silver. Nothing here is investment advice.
