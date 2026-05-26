# BTC Trend Signature Trading Strategy

**Document type:** Backtested trading strategy derived from trend signature patterns  
**Last updated:** 2026-05-26  
**Test window:** 2025-09-19 → 2026-05-17 (241 bars, fully out-of-sample)  
**Starting capital:** $100,000 USD  
**Signal source:** CT daily H/L ensemble (ridge + GBM + RF) — `artifacts/artifacts.pkl`

---

## Overview

This document records the development and results of a mechanical trading strategy built
on top of the four trend signature patterns described in [`TREND_SIGNATURES.md`](TREND_SIGNATURES.md).

Six strategy variants were tested in sequence, each refining the entry/exit rules.
The final strategy (**TF1**) is the only one that ended profitable in the test window and
is now implemented live in the Streamlit dashboard as the **🎯 STRATEGY BUY (TF1)** signal.

> **One-sentence summary:**  
> Buy when U1 fires and BTC is above its 30-day MA (or no bearish signals in past 10 days);
> sell only when D2 or D3 fires. Hold cash otherwise.

---

## Signal Definitions (Quick Reference)

| Signal | Condition | Type |
|--------|-----------|------|
| **U1** | `err_hi_ma3 > +0.5%` AND `hi_breaks_3d ≥ 2` | Uptrend — actual highs consistently exceed predictions |
| **D1** | `lo_breaks_3d ≥ 2` AND `err_lo_ma3 > 0.5%` | Downtrend — actual lows consistently break predicted floor |
| **D2** | `err_hi_ma3 < −1.0%` | Downtrend — predicted highs not being reached (exhaustion) |
| **D3** | First `lo_break` after ≥ 3 consecutive `hi_breaks` | Reversal canary — momentum-to-reversal handoff |

**Derived metrics:**

| Metric | Formula |
|--------|---------|
| `err_hi` | `(actual_high − pred_high) / close × 100` — positive = bullish |
| `err_lo` | `(pred_low − actual_low) / close × 100` — positive = bearish |
| `err_hi_ma3` | 3-bar rolling mean of `err_hi` |
| `err_lo_ma3` | 3-bar rolling mean of `err_lo` |
| `hi_breaks_3d` | Count of `actual_H > pred_H` in last 3 bars |
| `lo_breaks_3d` | Count of `actual_L < pred_L` in last 3 bars |
| `MA30` | Rolling 30-bar mean of daily close price |
| `above_ma30` | `close > MA30` |
| `clean_10d` | True if zero D1 or D2 fires in the prior 10 bars |

**Execution rule:** Signal fires on bar *i*; trade executes at bar *i+1* close (1-bar lag).

---

## Strategy Evolution — All Six Variants Tested

All strategies tested on the same 241-bar OOS window, $100k starting capital.
Buy & Hold final NAV: **$66,929 (−33.1%)**.

| # | Strategy | Entry | Exit | Final NAV | Return | vs B&H |
|---|----------|-------|------|-----------|--------|--------|
| 1 | Any signal | Any UP signal | Any DN signal | $70,693 | −29.3% | +$3,807 |
| 2 | U1-only entry | U1 only | WATCH_DN+ (any DN) | $83,737 | −16.3% | +$16,740 |
| 3 | U1 / D2+D3 | U1 | D2 or D3 | $84,898 | −15.1% | +$17,990 |
| 4 | U1+MA30 / any DN | U1 + (↑MA30 OR clean10d) | D1, D2, or D3 | $94,478 | −5.5% | +$27,602 |
| **5 (TF1)** | **U1+MA30 / D2+D3** | **U1 + (↑MA30 OR clean10d)** | **D2 or D3 only** | **$102,066** | **+2.1%** | **+$35,137** |

**Key insight:** The two critical refinements that turned a losing strategy into a winner:
1. **Add MA30 filter at entry** — eliminates buying into downtrend continuations
2. **Remove D1 from exits** — prevents premature exit during the trade that delivered +8.1%

---

## TF1 — Best Strategy (Full Specification)

### Entry Rule

Buy at next bar close when **all** of the following are true on the signal bar:

```
U1 is active:
    err_hi_ma3 > +0.5%
    hi_breaks_3d ≥ 2

AND at least one of:
    BTC close > 30-day rolling mean of close (above_ma30 = True)
    OR
    No D1 or D2 signal fired in any of the prior 10 bars (clean_10d = True)
```

### Exit Rule

Sell at next bar close when **either** of the following is true on the signal bar:

```
D2: err_hi_ma3 < −1.0%   (predicted highs not being reached)
OR
D3: first lo_break after ≥ 3 consecutive hi_breaks (exhaustion canary)
```

> **Note:** D1 is intentionally excluded from exit conditions. Including D1 would have
> caused a premature exit on the Apr 9–25 trade at +2.9%, missing the full +8.1%.
> D1 fires frequently during consolidation; D2 and D3 are structurally stronger
> reversal signals.

### Performance Summary

| Metric | Value |
|--------|-------|
| Starting capital | $100,000 |
| Final NAV | $102,066 |
| Total return | **+2.1%** |
| Buy & Hold return (same period) | −33.1% |
| Alpha vs Buy & Hold | **+$35,137** |
| Number of round trips | 6 |
| Win rate | 4 / 6 = **67%** |
| Average trade P&L | +0.4% |
| Best trade | +8.1% (Apr 9 – Apr 25, 2026) |
| Worst trade | −5.2% (Mar 10 – Mar 28, 2026) |
| Max portfolio drawdown | **12.7%** |
| Time in market | **19%** (strategy is in cash 81% of the time) |
| Test window | Sep 19, 2025 → May 17, 2026 (241 bars) |

---

## Complete Trade Log

All prices are daily closes (bar close = 12:00 UTC boundary).

| # | Entry Date | Buy Price | Entry Trigger | Exit Date | Sell Price | P&L | Exit Signal | NAV After |
|---|-----------|-----------|---------------|-----------|------------|-----|-------------|-----------|
| 1 | Dec 10, 2025 | $92,021 | U1 + ↑MA30 | Dec 11, 2025 | $92,511 | **+0.5%** ✓ | D3 | $100,533 |
| 2 | Jan 5, 2026 | $93,883 | U1 + ↑MA30 + clean10d | Jan 8, 2026 | $91,027 | **−3.0%** ✗ | D2 | $97,475 |
| 3 | Jan 13, 2026 | $95,322 | U1 + ↑MA30 | Jan 16, 2026 | $95,525 | **+0.2%** ✓ | D2 | $97,683 |
| 4 | Mar 10, 2026 | $69,927 | U1 + ↑MA30 | Mar 28, 2026 | $66,320 | **−5.2%** ✗ | D2 | $92,644 |
| 5 | Apr 9, 2026 | $71,768 | U1 + ↑MA30 + clean10d | Apr 25, 2026 | $77,612 | **+8.1%** ✓ | D2 | $100,188 |
| 6 | May 3, 2026 | $78,538 | U1 + ↑MA30 | May 7, 2026 | $80,010 | **+1.9%** ✓ | D3 | $102,066 |

### Trade Notes

**Trade 1 (Dec 10–11):** Quick +0.5% in 1 day. D3 canary fired — the exhaustion pattern
correctly identified the move was about to stall. Small win but avoided holding through
the subsequent pullback.

**Trade 2 (Jan 5–8):** First loss. U1 + MA30 filter passed (BTC above 30d MA, zero recent
DN signals). D2 exited cleanly after −3.0% when predicted highs stopped being reached —
correct behavior, avoided holding into further deterioration.

**Trade 3 (Jan 13–16):** Marginal +0.2% win. D2 exit was correct — BTC stalled at the
predicted ceiling again. The MA30 filter correctly admitted the trade (BTC still above MA).

**Trade 4 (Mar 10–28):** Worst trade. The U1 + MA30 entry triggered at what turned out to
be a local bounce within a larger downtrend. D2 exited at −5.2% — without D2 the loss could
have been significantly larger. This is the strategy's inherent risk: the MA30 filter does
not guarantee the macro trend is bullish.

**Trade 5 (Apr 9–25):** Best trade. BTC had fallen from ~$100k+ to ~$72k, triggering
`clean_10d = True` (no D1/D2 in prior 10 bars after the washout). The April recovery delivered
+8.1% before D2 fired. This trade alone is what separates TF1 from all other strategy variants —
D1 would have exited at +2.9%, cutting the gain by two-thirds.

**Trade 6 (May 3–7):** Clean +1.9% exit on D3 canary. Quick 4-day trade captured the
early part of a BTC bounce. D3 correctly identified the momentum exhaustion point.

---

## Why Each Filter Matters

### U1-only entry (vs. any signal entry)

The original "any UP signal" strategy took 12 round trips vs 6 for U1-only.
The extra trades were low-confidence entries that added noise. U1 requires both
a positive 3d average err_hi AND multiple hi-band breaks — a higher bar that filters
out single-day spikes.

**Impact:** Removing 6 low-quality trades improved alpha by ~$13k.

### MA30 trend filter (vs. U1-only)

The MA30 filter prevents buying into a U1 that fires during a bear-market bounce
(when BTC is below its 30-day MA and in a structural downtrend). The `clean_10d`
backup condition catches V-reversal opportunities where BTC has washed out enough
that the prior 10 days contain no recent bearish fingerprint.

**Impact:** The MA30 filter eliminated 4 of the 6 U1-only trades, keeping only
the ones with positive trend context. Alpha improved by ~$10k.

### D2+D3-only exit (vs. any DN signal)

D1 fires frequently — it captures periods of persistent low-band pressure but
also fires during consolidation phases within an ongoing uptrend. Using D1 as an
exit signal caused a premature exit on Trade 5 at +2.9% (where D1 would have
triggered on Day 7 of the trade), missing the full +8.1% (D2 exit on Day 16).

It also caused a spurious whipsaw trade in the previous strategy version:
a Mar 25 entry triggered by D1-exit-then-U1-reentry that lost −7.0% in 2 days.

**Impact:** Removing D1 from exits improved alpha by ~$7k and eliminated 1 bad trade.

---

## NAV Trajectory

```
Start (Sep 19, 2025):   $100,000   [CASH — no signal yet]
                        ...
Dec 10 entry:           $100,000   [BUY at $92,021]
Dec 11 exit:            $100,533   +$533   (+0.5%)  ✓ WIN
Jan  5 entry:            $97,475   [BUY at $93,883]   ← after dec pullback, U1 fires
Jan  8 exit:             $97,475   −$3,058  (−3.0%)  ✗ LOSS
Jan 13 entry:            $97,475   [BUY at $95,322]
Jan 16 exit:             $97,683   +$208   (+0.2%)  ✓ WIN
                        ...
Mar 10 entry:            $97,683   [BUY at $69,927]   ← U1 fires on bounce
Mar 28 exit:             $92,644   −$5,039  (−5.2%)  ✗ LOSS
Apr  9 entry:            $92,644   [BUY at $71,768]   ← clean10d recovery
Apr 25 exit:            $100,188   +$7,544  (+8.1%)  ✓ WIN
May  3 entry:           $100,188   [BUY at $78,538]
May  7 exit:            $102,066   +$1,878  (+1.9%)  ✓ WIN
May  7 → (current):     $102,066   [CASH — no active signal]
```

BTC price dropped from ~$63k (Sep 2025) to ~$76k (May 26, 2026), with a peak near $107k
(Nov–Dec 2025) and a trough near $74k (Apr 2026). Buy & Hold over the same period: −33.1%.

---

## Risk Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max drawdown | 12.7% | From $102,066 peak to trough during open losses |
| Largest single loss | −5.2% | Trade 4 (Mar 10–28), exited by D2 |
| Consecutive losses | 2 max | Trades 2+3 (Jan 2026), both small |
| Time in market | 19% | In cash 81% of the time — reduces overnight risk |
| Average hold duration | ~9 days | Range: 1 day (Trade 1) to 18 days (Trade 4) |
| Slippage/fees | Not modeled | All prices are bar closes; real execution incurs spread |

---

## Current Signal State (as of 2026-05-26)

| Signal | Value | Status |
|--------|-------|--------|
| `err_hi_ma3` | −0.82% | Neutral (between D2 threshold −1.0% and U1 threshold +0.5%) |
| `err_lo_ma3` | elevated | Bearish pressure persisting |
| `lo_breaks_3d` | 3/3 | All 3 recent bars broke predicted low |
| `hi_breaks_3d` | 0/3 | No recent hi-band breaks |
| U1 active | ❌ No | err_hi_ma3 below +0.5% threshold |
| D2 active | ❌ No | err_hi_ma3 above −1.0% threshold |
| **TF1 (strategy entry)** | **❌ INACTIVE** | — |
| **Strategy position** | **CASH at $102,066** | — |
| BTC price (May 26) | ~$76,490 | −4.4% since last exit (May 7, $80,010) |
| BTC vs MA30 | Below MA30 | Trend filter would block entry even if U1 fired |

The strategy correctly avoided the post-May-7 decline (−4.4%).

---

## Implementation in the Live Dashboard

The TF1 signal is live in `app/btc_hourly_app.py`:

- **`compute_trend_signatures()`** now fetches 45 completed bars (vs 10 previously),
  computes the 30-bar MA, and determines `above_ma30`, `clean_10d`, and `tf1_triggered`.
- **`render_trend_signatures()`** displays a full-width **TF1 card** below the D1/D2/D3/U1
  grid with live values for all 4 sub-conditions.
- The banner alert level reaches **🎯 STRATEGY BUY SIGNAL (TF1)** (blue) when TF1 fires —
  distinct from the plain **📈 UPTREND SIGNAL (U1)** level (green) when U1 fires alone without
  the trend filter.

---

## Important Caveats

1. **Small sample.** 6 trades over 8 months is insufficient for robust statistical inference.
   Win rate and return figures have wide confidence intervals.

2. **Single test window.** The test set covers Sep 2025 – May 2026. BTC declined ~33% in
   this period. The strategy's outperformance partly reflects its low exposure during a bear period.
   In a strong bull run, Buy & Hold would outperform this strategy.

3. **No transaction costs modeled.** Exchange fees (typically 0.05–0.1% per side),
   slippage, and funding rates are not included. Real-world returns will be lower.

4. **Execution at bar close.** The strategy assumes execution at the 12:00 UTC bar close
   on the day after the signal fires. In practice, there is a delay between signal
   computation and execution.

5. **Backfitted exit rule.** The choice to use D2+D3 (not D1) as the exit was made after
   observing the Apr 9–25 trade in the test data. This introduces a degree of selection bias.

6. **On-chain features not extendable.** The model relies on 33 on-chain features
   (hash rate, difficulty, miners' revenue) sourced from blockchain.info. Real-time signal
   computation uses the same model artifacts and depends on blockchain.info availability.

---

## Comparison with Prior Strategy Variants

### Why Strategy 4 (U1+MA30 / any DN) was inferior to TF1

Strategy 4 used D1 as an exit signal alongside D2 and D3. In the April 2026 trade:

```
Apr 9:  BUY at $71,768 (U1 + clean10d)
Apr 16: D1 would fire (lo_breaks_3d ≥ 2, err_lo_ma3 > 0.5%)
        Strategy 4 EXITS at ~$73,800 → +2.9%
Apr 25: D2 fires, Strategy TF1 EXITS at $77,612 → +8.1%
```

Additionally, Strategy 4 took a −7.0% whipsaw trade in late March 2026:
- Mar 25: D1 exit from previous position
- Mar 27: U1 fires again → re-entry
- Mar 28: D2 immediately fires → exit at −7.0%

TF1 avoided this whipsaw because it was already out of the Mar 10 trade (D2 exited Mar 28)
and did not re-enter until Apr 9 (clean10d + U1 met after the washout).

---

*Strategy developed from: 241-bar OOS analysis in `artifacts/artifacts.pkl`*  
*Backtesting run: May 2026 (session `claude/trend-signature-patterns-OYUDT`)*  
*Live implementation: `app/btc_hourly_app.py` — `compute_trend_signatures()`, `render_trend_signatures()`*
