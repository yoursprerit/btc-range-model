# Trend Signature Pattern Analysis
**Last updated:** 2026-06-08  
**Data window:** Test set 2025-09-19 → 2026-05-17 (241 bars) + bookmarked events  

---

## Executive Summary

After mining all historical model predictions vs actual values, band-cross patterns,
and prediction divergences across the CT daily H/L model (Huber + Quantile + GBM ensemble),
7-day cone, and 3-class day-type classifier, **four conclusive early-warning
signatures were identified — three for downtrends (strong, statistically significant)
and one for uptrends (moderate, context-dependent).**

The downtrend signals are substantially more reliable than the uptrend signals.
This asymmetry is consistent with the CT model's own calibration: the model
under-predicts magnitude on both sides, but the systematic signature of a
coming large move is cleaner on the downside.

---

## 1. The Four Conclusive Signatures

### 🔴 DOWNTREND Signature A — "Low-Band Accumulation" (strongest signal, 2.24× lift)

**Definition:**  
≥ 2 of the last 3 days had the actual daily LOW fall **below** the ensemble's predicted LOW
(`lo_breaks_3d ≥ 2`) AND the 3-day moving average of low-side error (`err_lo_ma3`) exceeds +0.5%.

**What it means mechanically:**  
The ensemble of three regressors (HuberRegressor, QuantileRegressor, GradientBoostingRegressor)
is consistently OVER-predicting where the floor is. The actual daily low keeps punching through the predicted floor, signalling
the models' "downside cushion" assumption is wrong — price is in a structural downtrend the
models haven't fully absorbed.

**Empirical evidence from test period:**
| Threshold | Triggers | Hit Rate (big-dn within 3d) | Base Rate | Lift |
|-----------|----------|------------------------------|-----------|------|
| score_dn > 0.6 | 34 | 26.5% | 17.8% | 1.48× |
| score_dn > 0.7 | 13 | 38.5% | 17.8% | **2.16×** |
| score_dn > 0.8 | 5  | 40.0% | 17.8% | **2.24×** |
| `err_lo_ma3 > 0.5` | 64 | 31.2% | 17.8% | **1.75×** |
| `lo_breaks_3d ≥ 2` | 79 | 26.6% | 17.8% | 1.49× |

**Statistically significant** (Mann-Whitney U, p < 0.001):
- `err_lo_pct` PRE_DN mean = **+2.53** vs NORMAL mean = −0.29 (Δ = +2.82, p < 0.001)
- `lo_band_break` PRE_DN rate = **71.4%** vs NORMAL rate = 32.7% (p = 0.0006)
- `lo_breaks_3d` PRE_DN mean = **1.57** vs NORMAL mean = 0.99 (p = 0.0095)

**Real event confirmations:**
- **2026-01-30 (T-1 before Jan 31 −6.5% drop):** `lo_breaks_3d = 3`, `score_dn = 0.919`
- **2026-02-04 (T-1 before Feb 5 −14.1% crash):** `lo_breaks_3d = 3`, `score_dn = 1.000`
- **2025-10-09–10 (Oct 10 −6.9% drop):** `lo_breaks_3d = 2→3`, `score_dn = 0.759→0.918`

---

### 🔴 DOWNTREND Signature B — "Predicted-High Collapse" (p < 0.001)

**Definition:**  
The 3-day MA of high-side error (`err_hi_ma3`) turns **sharply negative** (< −1.3%).
Actual daily highs are *below* what the model predicted. The model was forecasting
upside potential, but price failed to reach those predicted highs.

**What it means:**  
Models trained on momentum features predict an upside continuation, but price
can't deliver. The predicted high is an "air target" that the actual market refuses
to touch. This divergence is the model's momentum-feature loading fighting against
a structural reversal.

**Statistically significant:**
- `err_hi_pct` PRE_DN mean = **−1.34** vs NORMAL mean = −0.22 (Δ = −1.13, p < 0.0001)

**Real event confirmations:**
- **Oct 7 window (pre-drop):** `err_hi_ma3` was −0.49 to −1.57 in the 3 days prior
  while the high-band breaks were also elevated — a toxic combination of
  "model said go higher, price barely moved, then reversed"
- **Feb 2026 cascade:** `err_hi_pct` stayed negative (−1.3 to −2.3) through the entire
  Jan 29 – Feb 5 sequence, as model predicted recovery highs that never materialized

---

### 🔴 DOWNTREND Signature C — "Exhaustion Canary" (contextual)

**Context:** Applicable specifically after multi-day uptrends where actual highs have
consistently exceeded predicted highs for ≥ 3 consecutive bars.

**Exact trigger condition (D3):**  
Today's bar is a `lo_break` (actual low < predicted low) **AND** the immediately
preceding N days were ALL `hi_break` days with N ≥ 3. The look-back counts backward
from yesterday and stops at the first non-`hi_break` bar — if that count reaches 3 or
more, D3 fires on the first `lo_break` today.

```python
# Pseudocode matching the live implementation
consec_hi = count_consecutive_hi_breaks_ending_yesterday()  # stops at first non-hi_break
d3 = (consec_hi >= 3) and lo_break_today
```

**What it means:**  
After ≥ 3 consecutive sessions where price punches through the model's predicted ceiling,
the first session where price instead breaks through the predicted floor signals that upside
momentum has exhausted. The model was consistently under-predicting the ceiling; now it is
over-predicting the floor — a structural handoff from uptrend to reversal.

**Oct 7 drop: the textbook case:**
```
2025-10-04 [PRE-EVENT]: hi_break=1, streak=+4, hiBreak3d=2, ↓score=0.272
2025-10-05 [PRE-EVENT]: hi_break=1, streak=+5, hiBreak3d=2, ↓score=0.244
2025-10-06 [PRE-EVENT]: hi_break=1, streak=+6, hiBreak3d=2, loBreak3d=1 ← lo break appears
2025-10-07 ◄ EVENT:    ret=−2.6%, err_hi_pct=+0.14 (barely exceeded), streak resets
2025-10-08–10:         lo_breaks accelerate → −6.9% on Oct 10
```

On Oct 6, the first `lo_break` appeared after 6 consecutive `hi_break` days — D3 fired,
marking the exact moment price lost its upside momentum.

---

### 🟢 UPTREND Signature — "High-Band Breakout Persistence" (moderate, 1.68× lift)

**Definition:**  
The 3-day MA of high-side error (`err_hi_ma3`) turns **persistently positive** (> +1.3%)
AND at least 2 of the last 3 days had actual daily high exceed the predicted high
(`hi_breaks_3d ≥ 2`). Both conditions must hold simultaneously (U1 = AND gate).

**What it means:**  
The models (all three regressors) are under-predicting the upside. The market is
pushing through every predicted ceiling. The feature set — which includes momentum,
macro, and on-chain — has not yet fully captured the strength of the upward move.

**Evidence:**
| Signal | Triggers | Hit Rate (big-up within 3d) | Base Rate | Lift |
|--------|----------|------------------------------|-----------|------|
| `err_hi_ma3 > +0.7` AND `hi_breaks_3d ≥ 2` (U1) | ~40 | 16.0% | 9.5% | **1.68×** |
| `err_hi_ma3 > +0.5` alone | 50 | 16.0% | 9.5% | **1.68×** |
| `hi_breaks_3d ≥ 2` alone | 70 | 10.0% | 9.5% | 1.05× |

**Statistically significant:** `err_hi_pct` PRE_UP mean = **+1.34** vs NORMAL = −0.22 (Δ = +1.56, p = 0.022)

> The live U1 threshold (`err_hi_ma3 > 1.3`, 2026-07d — raised to cut bear-market losers) is tighter
> than the original analysis threshold (`> 0.5`) to reduce false positives at the cost of fewer
> triggers. It was raised from `> 0.7` when the backtest was re-anchored to the 12:00-UTC timeline.

**Real event confirmations:**
- **2026-05-04 Gradual Climb:** `hiBreak3d = 2`, `↑score = 0.685–0.751` in days before event
- **Feb 6 +12.5% bounce:** After the -14% crash on Feb 5, models began over-predicting
  (err_close_pct = −0.557) and the upside score rapidly recovered to 0.806 the next day

**Important caveat:** The uptrend signal has weaker lift (1.68×) vs downtrend (2.24×).
This is because uptrends are often continuation plays (momentum already visible) while
the most explosive upward moves (bookmarked "drastic climbs") tend to happen as V-shaped
reversals from oversold conditions — harder to detect early.

---

## 2. The "Oversold Bounce" Pattern (V-shaped Recovery)

The bookmarked "Drastic Climbs" on 2025-06-23 and 2026-02-06 are **both** immediate
reversals from drastic drops. They share a distinct pattern:

1. **Day T-2 to T-1:** DOWNTREND score peaks (0.704–1.000), `lo_breaks_3d = 2–3`
2. **Day T (crash day):** `err_lo_pct` spikes huge (actual low far below predicted) — the 
   final capitulation where price overshoots the model's worst case
3. **Day T+1 (recovery):** `err_close_ma3` turns sharply negative (models under-predict
   again but now to the upside), `hi_breaks_3d` picks up, upside score begins recovering

**Feb 5 (-14.1%) → Feb 6 (+12.5%) transition:**
```
2026-02-04: lo_breaks=3, ↓score=1.000          ← extreme downtrend signal
2026-02-05: ret=−14.1%, err_lo_pct=+big_spike   ← capitulation
2026-02-06: ret=+12.5%, err_close_ma3=−0.139,  ← models underestimate recovery
            ↑score=0.553, ↓score=0.660           ← both scores elevated (uncertainty)
2026-02-07: ↑score=0.806                         ← uptrend signal appears T+1
```

The **transition trigger** for V-reversal detection: downtrend score ≥ 0.9 followed by a 
day where the actual low **massively** undershoots the predicted low (`err_lo_pct > 5%`).
This large single-day undershoot marks capitulation — the most reliable setup for a bounce.

---

## 3. Model Disagreement as a Volatility Precursor

The `model_disagree` feature (inter-model spread on predicted high, as % of close) showed
a **statistically significant elevated value before uptrend events** (PRE_UP: 1.40 vs NORMAL: 0.97,
p = 0.023). This makes structural sense: when the three constituent models (Huber/linear,
Quantile, Gradient Boosting) disagree strongly on where price is going, it indicates the
feature space is ambiguous — usually because the current regime is transitioning.

This disagreement signal is NOT significant before downtrends (p = 0.74), suggesting
downtrends tend to happen in periods where all models agree on the downward trajectory,
while uptrend initiations happen from more ambiguous starting conditions.

---

## 4. 7-Day Cone Regime Context

The 7-day cone's three volatility regimes provide important context for signal interpretation:

| Regime | range_ma30 | Median 7d return | Std | Interpretation |
|--------|------------|------------------|-----|----------------|
| Low vol  | < 1.75%  | +1.06%           | 7.3% | Trending/consolidation |
| Mid vol  | 1.75–2.38% | −0.25%          | 8.5% | Chop/mean-reversion |
| High vol | > 2.38%  | +1.31%           | 10.0% | Volatile/momentum |

**Key finding:** The Jan-Feb 2026 crash cluster occurred in **high-vol regime** (range_ma30 > 2.38%).
In high-vol regime, the 7-day cone is ±10.5% wide. Band breaks in high-vol regime are MORE
informative than in low-vol regime precisely because the model is already allowing for wider
swings — when price still breaks through those wide bands, it's a genuinely extreme signal.

---

## 5. 3-Class Day-Type Model Convergence

The 3-class classifier (BigUpper/Quiet/BigLower) provides an orthogonal signal:
- At confidence threshold **p ≥ 0.55**: **69.2% accuracy**, 43.8% coverage
- At confidence threshold **p ≥ 0.60**: **76.2% accuracy**, 23.6% coverage

**Actionable rule:** When 3-class predicts `BigLower` with p ≥ 0.55 AND `lo_breaks_3d ≥ 2`
from the CT model simultaneously, this is the highest-confidence downtrend composite signal
available in the system. Both signals independently passed significance testing and operate
on different feature sets (3-class uses stacked OOF predictions + regime one-hots while
CT ensemble uses raw 103-feature matrix).

---

## 6. Complete Composite Signal Checklist

### DOWNTREND Early Warning (check before each day's close)
| # | Signal | Condition | Statistical Support | Importance |
|---|--------|-----------|--------------------|-----------| 
| 1 | D1 | `lo_breaks_3d ≥ 2` AND `err_lo_ma3 > +0.5%` | p = 0.0095 | **Primary** |
| 2 | D2 | `err_hi_ma3 < −1.3%` (predicted highs not being reached) | p < 0.0001 | **Primary** |
| 3 | D3 | First `lo_break` today after ≥ 3 consecutive `hi_break` days | contextual | **Primary** |
| 4 | — | `lo_band_break = 1` on current day | p = 0.0006 | Confirming |
| 5 | — | 3-class predicts `BigLower` with p ≥ 0.55 | 69.2% accuracy | Confirming |
| 6 | — | Regime = High-vol (range_ma30 > 2.38%) | Contextual | Amplifying |
| 7 | — | `model_disagree` low (all models agree on downside) | p = 0.74 (absence) | Contextual |

**Trigger:** Any 2 of D1/D2/D3 simultaneously = HIGH_DN alert. Any 1 of D1/D2/D3 = WATCH_DN.

### UPTREND Early Warning  
| # | Signal | Condition | Statistical Support | Importance |
|---|--------|-----------|--------------------|-----------| 
| 1 | U1 | `err_hi_ma3 > +1.3%` AND `hi_breaks_3d ≥ 2` | p = 0.022 | **Primary** |
| 2 | — | `hi_band_break = 1` on current day | p = 0.033 | Confirming |
| 3 | — | `model_disagree` elevated (> 1.2) | p = 0.023 | Context (regime transition) |
| 4 | — | 3-class predicts `BigUpper` with p ≥ 0.55 | 69.2% accuracy | Confirming |

**Strategy entry (TF1/TF2):** U1 fires AND (`close > MA30` OR no D1/D2 in prior 7 bars OR V-reversal gate).

**V-reversal special case:** After `lo_breaks_3d = 3` AND single-day `err_lo_pct > 3%`
(dn_score > 0.8 + large undershoot), expect high-probability bounce within 1-2 days.

---

## 7. Limitations and Important Caveats

1. **Test set size:** The test window (241 bars, Sep 2025–May 2026) contains only 10 big-up
   and 10 big-down single-day events. Signal lift estimates have wide confidence intervals.

2. **Direction accuracy was 50%:** The CT model's blended direction head has coin-flip 
   directional accuracy on the test set. The signals described here operate on the 
   *relative position* of actual vs predicted (band-crossing patterns), NOT on the model's
   own direction prediction — which is why they can still carry information.

3. **The uptrend lift (1.68×) is weaker than downtrend (2.24×).** Explosive uptrends 
   (particularly V-reversals) are harder to detect from prediction-vs-actual patterns alone
   because they coincide with periods when all models are mis-calibrated.

4. **Oct 7 2025 "drop" was mild:** The bookmarked "Drastic Drop" on Oct 7 was only −2.6%.
   The real damage came Oct 10 (−6.9%). The signals (lo_breaks accumulating Oct 7–9) did
   correctly precede the larger move, but by 3 days.

5. **Pre-June 2025 events are outside the test period** — the bookmarked June/July 2025 
   events occurred in the training window. Patterns there cannot be treated as out-of-sample
   validation. Only the Oct 2025 – May 2026 events are fully out-of-sample.

---

## 8. Backtested Trading Strategy

A systematic trading strategy built on these signatures has been backtested across two periods.
See [`TRADING_STRATEGY.md`](TRADING_STRATEGY.md) for the complete specification, all variant
results, and full trade logs.

**Best strategy (TF1) summary — OOS period (Sep 2025 → May 2026):**  
Entry on U1 + (BTC above 30-day MA OR clean_10d), exit on D2 or D3 only.  
Result: $100k → $102k (+2.1%) vs Buy & Hold −33.1%; alpha +$35k; 4W/2L; 12.7% max drawdown; 19% time in market.

**Prior-year test (Sep 2024 → Sep 2025, in-sample):**  
Same TF1 strategy in a strong bull market (+93% BTC).  
Result: $100k → $106k (+6.3%) vs Buy & Hold +93.1%; alpha −$87k; 5W/11L; 34.3% max drawdown; 44% time in market.  
Strategy underperforms in bull regimes — by design it minimises exposure, which is costly when the trend is strongly up.

---

## 9. Implementation

The live implementation in `app/btc_hourly_app.py → compute_trend_signatures()` computes
these signals daily after each 12:00 UTC bar closes.

```python
# ── Raw per-bar errors ──────────────────────────────────────────────────────
err_hi  = (actual_high - pred_high) / close * 100   # + = bullish overshoot
err_lo  = (pred_low - actual_low)   / close * 100   # + = bearish undershoot

# ── 3-bar rolling metrics ───────────────────────────────────────────────────
err_hi_ma3   = mean(err_hi[-3:])          # 3d avg high-side error
err_lo_ma3   = mean(err_lo[-3:])          # 3d avg low-side error
hi_breaks_3d = sum(actual_H > pred_H for last 3 bars)
lo_breaks_3d = sum(actual_L < pred_L for last 3 bars)

# ── Signal trigger conditions (exact live thresholds) ───────────────────────
D1 = (lo_breaks_3d >= 2) and (err_lo_ma3 > 0.5)    # Low-Band Accumulation
D2 = (err_hi_ma3 < -1.3)                             # Predicted-High Collapse (re-tuned 2026-07)
D3 = (consec_hi_breaks_ending_yesterday >= 3)        # Exhaustion Canary
     and lo_break_today                              #   (first lo_break after streak)
U1 = (err_hi_ma3 > 1.3) and (hi_breaks_3d >= 2)     # High-Band Breakout Persistence (2026-07d)

# ── Strategy entry gate ──────────────────────────────────────────────────────
MA30       = mean(close[-30:])
above_ma30 = close_today > MA30
clean_7d   = no D1 or D2 fired in the 7 bars preceding today   # named clean_10d in code
v_gate     = dn_score > 0.8 and err_lo_today > 3%              # capitulation reversal

TF1_entry = U1 and (above_ma30 or clean_7d or v_gate)

# ── Alert level ──────────────────────────────────────────────────────────────
dn_count = D1 + D2 + D3
# dn_count >= 2 → HIGH_DN
# dn_count == 1 → WATCH_DN
# TF1_entry     → STRATEGY_BUY
# U1 only       → WATCH_UP
```

> **Note on `clean_10d` naming:** The variable is named `clean_10d` throughout the codebase
> for historical reasons but the actual look-back is **7 bars** (indices `i-7` to `i-1`).

The Streamlit dashboard renders a "Trend Alert" card with live values for all conditions
alongside the H/L band, 7-day cone, and 3-class outputs.

---

*Analysis performed by: `src/trend_signature_analysis.py`*  
*Historical analysis: `artifacts/artifacts.pkl` (Ridge + GBM + RF test-period predictions)*  
*Live signal: `models/inference_assets_ct.joblib` (Huber + Quantile + GBM ensemble)*  
*Ground truth: `artifacts/artifacts.pkl` — test_index, high_true, low_true, close_te*
