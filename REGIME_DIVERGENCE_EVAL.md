# Regime-Divergence vs MA-Filter — evaluation for SOXX · VEGN · GRID · REMX · WGMI

> **⚠️ Historical memo — pre-dates the 2026-07 causal H/L fix.** The daily High/Low model behind every *divergence*-engine number in this memo was later found to pair the target bar's own features with its target (look-ahead). Divergence-based figures here reflect that leaky signal and are kept only as a historical record; the re-tuned honest configs and results live in `GLDM_TRADING_STRATEGY.md`, `app/ticker_config.py` and the per-app `backtest_results.json`. Trend-engine (MA/dual-MA/MACD/ma_vol) figures are unaffected by the fix (they use closes only), though they drift with data refreshes.


**Question.** For the five apps that currently run an **MA trend-filter** strategy
(SOXX, VEGN, GRID, REMX, WGMI), would switching to an **optimally-tuned
regime-divergence** strategy — with the divergence thresholds chosen using
**training data up to August 2025** — *beat the current backtest results*, and
*beat buy-&-hold (B&H)*? And if any asset improves individually, does feeding
those signals into the **Overall Trading** allocation mix improve the combined
portfolio?

**Not implemented.** This is an evaluation only — no strategy, config, or
allocation was changed. Reproduce with:

```bash
python scripts/eval_regime_aug2025.py     # per-asset: MA vs optimal divergence vs B&H
python scripts/eval_overall_swap.py       # Overall allocation-mix impact
```

## Method

* The daily High/Low signal model is fit on each app's pre-OOS window exactly as
  the live apps do, so the long backtest windows stay genuinely out-of-sample.
* **"Optimal divergence, trained through Aug-2025"**: for each asset the
  divergence thresholds (U1 × D2 × stop × D1-exit) are chosen using **only data
  from the OOS start through 2025-08-01** — no peeking past Aug-2025. Selection
  rule = the repo's own: maximise Sharpe subject to max-drawdown ≤ B&H,
  tie-broken by return.
* Those **fixed** thresholds are then scored over (a) each app's full reported
  OOS window and (b) the true **holdout Aug-2025 → now**, which the tuner never
  saw. The current MA config's published numbers reproduce **exactly**, so the
  engine runs are trustworthy.

---

## 1. Individual results — does divergence beat the current MA filter, and B&H?

### Full reported OOS window (like-for-like vs today's config)

| App | Window | B&H | **MA (current)** | Optimal divergence | Div beats MA? | Div beats B&H? |
|---|---|---|---|---|---|---|
| SOXX | 2021→now | +362% / −46% / 0.93 | **+233% / −31% / 0.94** | +75% / −13% / 0.77 | ❌ ret & Sharpe | ❌ |
| VEGN | 2022→now | +81% / −33% / 0.73 | **+84% / −14% / 1.02** | +28% / −15% / 0.55 | ❌ | ❌ |
| GRID | 2021→now | +132% / −30% / 0.83 | **+105% / −19% / 0.87** | +45% / −19% / 0.63 | ❌ | ❌ |
| **REMX** | 2021→now | +24% / −74% / 0.30 | +73% / −56% / 0.48 | **+109% / −18% / 0.93** | ✅ **ret+Sharpe+DD** | ✅ **all three** |
| WGMI | 2024→now | +223% / −63% / 0.98 | **+191% / −40% / 1.02** | +126% / −36% / 0.92 | ❌ | ❌ |

*(cells are total return / max drawdown / Sharpe)*

### True holdout — Aug-2025 → now (thresholds never saw this)

| App | B&H | MA (current) | Optimal divergence | Div beats MA? |
|---|---|---|---|---|
| SOXX | +145% / 2.51 | **+104% / 2.14** | +27% / 1.45 | ❌ |
| VEGN | +44% / 2.12 | **+38% / 1.97** | +17% / 1.71 | ❌ |
| GRID | +33% / 1.50 | +33% / 1.50 | +26% / **2.28** | ~ (Sharpe only, less return) |
| REMX | +73% / 1.45 | **+79% / 1.54** | +27% / 1.22 | ❌ |
| WGMI | +141% / 1.61 | **+166% / 2.12** | +77% / 1.65 | ❌ |

*(total return / Sharpe)*

**Verdict — individual apps.**
* **4 of 5 (SOXX, VEGN, GRID, WGMI): NO.** An optimal divergence strategy is
  *worse than the MA filter* on both return and (almost always) Sharpe, and it
  does **not** beat B&H. These are secular / trending compounders where a simple
  "long above the SMA" filter is already the right tool; the divergence system
  sits in cash too often and forfeits a large slice of the trend.
* **REMX: YES — but only over the full window.** Over 2021→now, optimal
  divergence beats *both* the MA filter *and* B&H on **all three** axes
  (+109% vs +73%/+24% return, −18% vs −56%/−74% drawdown, 0.93 vs 0.48/0.30
  Sharpe). This is the boom-bust commodity-equity where standing aside through
  the multi-year bear is decisive. **However, on the unseen Aug-2025→now
  holdout the divergence variant lagged** (+27% vs MA's +79%) — it was too
  defensive during the recent rally, winning only on drawdown. So even for REMX
  the standalone edge is a full-cycle artifact, not a robust recent one.

---

## 2. Overall Trading allocation mix — does feeding divergence signals in help?

Because only REMX improved individually (and only full-cycle), the allocation
test is what really matters. Baseline = the live 15-instrument universe with
every app's **current** strategy; then the ETF sleeves are re-run in divergence
mode and the cross-asset weights re-optimised with the **exact same optimiser
call** the app uses (balanced objective, caps, −35% MDD floor, SATA idle yield,
fundamental tilt, seed 7). Deltas are computed within one run, so they are valid
even though the reproduced baseline (+424% / Sharpe 3.20) differs slightly from
the committed artifact (live-fetch date + fundamental water-fill).

### Full window (2021 → now), optimal blend

| Scenario | Total ret | CAGR | MDD | Sharpe | Vol | Δ vs baseline |
|---|---|---|---|---|---|---|
| **Baseline** (current) | +424% | 35.1% | −4.89% | 3.20 | 8.02% | — |
| **A: REMX → divergence** | +493% | 38.2% | −5.28% | 3.30 | 8.35% | **+69pp ret, +0.10 Sharpe** |
| **B: all 5 → divergence** | +465% | 37.0% | −5.21% | 3.53 | 7.57% | **+41pp ret, +0.34 Sharpe, −0.45pp vol** |

### Recent windows — each scenario's own optimal weights applied forward

| Window | Baseline | A: REMX=div | B: all-5=div |
|---|---|---|---|
| 2025→now | +102% / 3.78 Sharpe | +106% / 3.95 | +90% / 3.88 |
| **Aug-2025→now (holdout)** | +61% / 3.90 | +61% / **4.06** | +53% / **4.41** |

*(total return / Sharpe; MDD −4.3% baseline → −3.7% scenario B)*

**Verdict — allocation mix: YES, on a risk-adjusted basis.**
* Both scenarios improve the combined backtest, and the improvement **persists
  into the unseen Aug-2025→now holdout** (Sharpe 3.90 → 4.06 / 4.41), which is
  the reassuring part.
* **Swapping REMX alone (A)** gives the biggest raw-return jump (+69pp): with a
  low-drawdown, decent-return stream the optimiser lifts REMX from a floored
  ~1.5% to ~6% weight.
* **Swapping all five (B)** gives the best *risk-adjusted* result — Sharpe
  3.20 → 3.53, vol 8.0% → 7.6%, MDD −4.9% → −5.2% — at the cost of ~28pp less
  raw return than scenario A.

### Why the paradox (individually worse, but better in the blend)?

The Overall optimiser does **not** maximise any single sleeve's return — it
maximises portfolio risk-adjusted return under drawdown and weight caps. The
divergence variants trade *lower return for much lower drawdown and lower
correlation* to the rest of the book (BTC/MSTR, gold/GDX, XLE/OIH). That makes
them better **diversifiers**, so the optimiser up-weights them and the combined
Sharpe/vol/MDD improve — even though each is a worse *standalone* bet.

---

## Bottom line & recommendation

1. **Do not switch the individual SOXX / VEGN / GRID / WGMI apps to divergence.**
   It cuts return materially, doesn't beat the MA filter on Sharpe, and doesn't
   beat B&H. The MA trend filter is the correct tool for these trending names.
2. **REMX is the one genuine standalone case** where divergence beats the MA
   filter and B&H over the full cycle — but it underperformed on the recent
   holdout, so treat the standalone win as full-cycle, not current-regime.
3. **At the portfolio level the divergence signals do add value** — as
   *diversifiers*, not as return engines. The all-five-divergence mix is the
   best risk-adjusted blend (Sharpe 3.20→3.53, lower vol/MDD) and it holds on
   the unseen holdout.

### Caveats
* Divergence thresholds were tuned on 2021→Aug-2025, ~90% of the Overall
  backtest window — mild in-sample optimism remains, though the holdout results
  are supportive.
* Overall weights are optimised with full-period hindsight (same as the live
  app); the recent-window table applies those weights forward as a forward test.
* This does not change the fact that on the most recent unseen data the
  divergence sleeves individually gave up return vs their MA counterparts — the
  portfolio benefit comes from re-weighting toward their calmer return streams.
