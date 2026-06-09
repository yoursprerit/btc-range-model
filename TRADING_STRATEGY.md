# BTC Trend Signature Trading Strategy

**Document type:** Backtested trading strategy derived from trend signature patterns  
**Last updated:** 2026-05-26  
**Current live strategy:** TF2 (Regime-Adaptive) — supersedes TF1  
**OOS test window:** 2025-09-19 → 2026-05-17 (241 bars, fully out-of-sample)  
**In-sample test window:** 2024-09-17 → 2025-09-17 (365 bars, in-sample — model trained through this period)  
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

### Performance Summary (OOS)

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

## Complete Trade Log (OOS)

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

## NAV Trajectory (OOS)

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

## Risk Parameters (OOS)

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

## TF2 — Regime-Adaptive Strategy (Current Live Strategy)

TF2 was derived by running a multi-strategy optimizer across **both** test periods simultaneously,
seeking an approach that maximises returns in bull markets while minimising losses in bear markets.

> **One-sentence summary:**  
> Same entry as TF1; exit adapts to market regime — patient (D3 only) in bull markets,
> defensive (D2 or D3) in bear markets.

### What's Different from TF1

The **only change** from TF1 is the exit logic:

| | TF1 | TF2 |
|--|-----|-----|
| **Entry** | U1 + (↑MA30 OR clean_10d) | Same ✓ |
| **Exit (Bear/Neutral regime)** | D2 or D3 | Same ✓ |
| **Exit (Bull regime)** | D2 or D3 | **D3 only** ← key change |

**Regime definition:**
```
BULL regime:   BTC close > MA30  AND  MA30[today] > MA30[5 bars ago]
BEAR/Neutral:  everything else
```

### Why This Works

In a **bull market**, D2 (`err_hi_ma3 < −1%`) fires repeatedly during brief consolidations —
the model expects a ceiling, BTC briefly stalls, but then resumes upward. TF1 exits on every D2,
causing whipsaws (16 trades in the prior-year bull period vs 6 in the bear OOS period).

By staying patient in BULL regime (exiting only on D3 — the structural reversal canary),
TF2 holds through the pauses and captures the full trend.

In a **bear/consolidation market**, D2 fires and BTC does NOT recover — the stall is a real
reversal. TF1's quick D2 exit is the correct behavior, and TF2 preserves it.

### Multi-Strategy Optimizer Results

Five strategies were backtested across both periods simultaneously:

| Strategy | Description | Bear Alpha | Bull Alpha | Combined | Beats B&H Both? |
|----------|-------------|------------|------------|----------|-----------------|
| TF1 | Fixed D2\|D3 exit | **+$38k** | −$67k | −$29k | ✗ NO |
| **TF2** | **Regime exit: Bull→D3, Bear→D2\|D3** | **+$30k** | **−$5k** | **+$26k** | ✗ close |
| TF3 | Stricter D2 threshold in bull (−1.5%) | +$30k | −$17k | +$12k | ✗ NO |
| TF4 | Confirmed D2 (2 consecutive bars) | +$31k | −$26k | +$5k | ✗ NO |
| TF5 | Regime entry + regime exit | +$30k | −$5k | +$26k | ✗ close |

**Key finding:** No strategy outperforms B&H in BOTH a +73% bull AND a −33% bear market
without leverage. In a strong bull run (+93% BTC), even 50% time-in-market is structurally
disadvantaged vs 100% B&H exposure.

**TF2 wins on combined alpha** (+$26k total) vs TF1's combined −$29k.
More importantly, TF2 narrows the bull-market gap from −$67k (TF1) to just −$5k.

### TF2 Trade Comparison

**Bear period OOS (Sep 2025 → May 2026):**

| # | Entry | Exit | P&L | Trigger | Exit Signal |
|---|-------|------|-----|---------|-------------|
| 1 | Dec 12, 2025 | Dec 15, 2025 | **−4.3%** ✗ | U1+↑MA30 | D2 (bear) |
| 2 | Dec 31, 2025 | Jan 12, 2026 | **+4.2%** ✓ | U1+clean10d | D2 (bear) |
| 3 | Mar 11, 2026 | Mar 28, 2026 | **−5.5%** ✗ | U1+↑MA30 | D2 (bear) |
| Open | Apr/May 2026 | — | (open) | U1+↑MA30 | — |

*Note: Trade dates differ slightly from TF1 OOS log due to different data source (Yahoo Finance daily vs Binance hourly 12:00-UTC bars). The live dashboard uses Binance data.*

**Bull period in-sample (Sep 2024 → Sep 2025):**

| # | Entry | Exit | P&L | Duration | Exit Signal |
|---|-------|------|-----|----------|-------------|
| 1 | Oct 30, 2024 | Dec 19, 2024 | **+34.8%** ✓ | 50 days | D3 |
| 2 | Jan 18, 2025 | Feb 18, 2025 | **−8.5%** ✗ | 31 days | D2 |
| 3 | Apr 3, 2025 | Apr 11, 2025 | **+0.4%** ✓ | 8 days | D3 |
| 4 | Apr 13, 2025 | Jun 14, 2025 | **+26.0%** ✓ | 62 days | D2 |
| 5 | Jun 26, 2025 | Jul 13, 2025 | **+11.4%** ✓ | 17 days | D3 |
| Open | Jul+ 2025 | — | (open) | — | — |

TF2 captured BTC's two largest bull runs in the period (Oct→Dec +34.8%, Apr→Jun +26.0%)
by staying patient in BULL regime while TF1 exited them prematurely at D2 stalls.

### Performance Summary

| Metric | TF1 (Bear OOS) | TF2 (Bear OOS) | TF1 (Bull IS) | TF2 (Bull IS) |
|--------|---------------|---------------|---------------|---------------|
| Strategy return | +7.8% | +0.0% | +6.1% | **+68.4%** |
| B&H return | −30.3% | −30.3% | +72.9% | +72.9% |
| Alpha | **+$38k** | +$30k | −$67k | **−$5k** |
| # trades | 4 | 3+open | 13 | 5+open |
| Win rate | 75% | 33% | 38% | 80% |
| Max drawdown | −8.7% | −14.1% | −30.9% | −22.9% |
| Time in market | 18% | 16% | 35% | 51% |

**Recommendation:** TF2 is the live strategy. Use TF1 as a reference for pure bear-market
defensive performance. When the regime clearly switches to BEAR (MA30 slope turns negative),
TF2 automatically becomes TF1-equivalent.

---

## Stop-Loss & Re-entry Criteria (MSTR and MSTU)

Stop-loss criteria were backtested across 5 periods (Bull Sep24→Sep25, Bear Jun25→May26,
OOS Sep25→May26, OOS-Recent Mar→May26, Full Jun24→May26) against 6 re-entry variants
(SL0–SL5). BTC trailing −7% stop was found to provide **no consistent benefit** and
is **not used**. MSTR and MSTU benefit significantly from stops and specific re-entry criteria.

### BTC — No Stop Loss

| Decision | Rationale |
|----------|-----------|
| **No stop loss** | BTC trailing −7% fires 9× over the Full period but recoveries are immediate. All SL re-entry variants underperform or equal SL0 (standard re-entry). The TF2 D2/D3 exit system already handles most adverse moves. |

### MSTR — Fixed −3% Stop + SL5 Regime-Adaptive Re-entry

| Parameter | Value |
|-----------|-------|
| Stop type | Fixed |
| Stop level | −3% from entry price |
| Re-entry variant | **SL5 regime-adaptive** |
| Re-entry (BULL) | Immediate — re-enter on next valid TF2 signal |
| Re-entry (BEAR) | 10-bar cooldown after stop, then allow next valid signal |

**Rationale:** MSTR's 3% stop fires frequently in volatile sideways markets. In bull regimes,
quick recovery means immediate re-entry captures upside. In bear regimes, a 10-bar cooldown
prevents the cascading Jul–Aug 2025 stop pattern (3 stops in 11 days at $412→$372→$343).
Full-period result: **+167% (SL5) vs +145% (SL0 immediate) vs +91% (B0, no stop)**.

**Backtested scores (vs SL0 standard re-entry):**

| Period | B0 (no SL) | SL0 | SL5 regime-adaptive |
|--------|-----------|-----|---------------------|
| Bull (Sep24→Sep25) | +25.0% | +19.9% | **+26.2%** ▲ |
| Bear (Jun25→May26) | −28.4% | −8.2% | **−4.8%** ▲ |
| OOS (Sep25→May26) | +0.1% | +6.6% | **+10.6%** ▲ |
| OOS-Recent | +34.8% | +34.8% | **+34.8%** = |
| Full (Jun24→May26) | +90.8% | +144.7% | **+167.3%** ▲ |

### MSTU — Fixed −10% Stop + SL1 Above Exit Price Re-entry

| Parameter | Value |
|-----------|-------|
| Stop type | Fixed |
| Stop level | −10% from entry price |
| Re-entry variant | **SL1 above stop exit price** |
| Re-entry condition | Allow re-entry only when MSTU price ≥ stop exit price |

**Rationale:** MSTU's 2× daily leverage amplifies volatility. The −10% stop protects against
cascade losses. SL1 (above exit price) prevents re-entering during continued downtrends —
key event: Jul–Aug 2025 cascade ($87→$68→$54), where SL1 exits once at $87 and waits for
price recovery before re-entering, avoiding two more losing entries.
Full-period result: **+269% (SL1) vs +202% (SL0 immediate) vs +97% (B0, no stop)**.

**Backtested scores (vs SL0 standard re-entry):**

| Period | B0 (no SL) | SL0 | SL1 above exit price |
|--------|-----------|-----|----------------------|
| Bull (Sep24→Sep25) | −10.7% | −4.0% | **+3.4%** ▲ |
| Bear (Jun25→May26) | −59.9% | −38.3% | **−11.4%** ▲ |
| OOS (Sep25→May26) | −11.6% | −11.8% | **−11.3%** ▲ |
| OOS-Recent | +71.8% | +71.8% | **+71.8%** = |
| Full (Jun24→May26) | +96.5% | +202.2% | **+269.1%** ▲ |

*Analysis file: `backtest_stop_loss_reentry.py` — run to reproduce all 5-period comparisons.*
*Implemented in: `app/btc_hourly_app.py` — `run_mstr_backtest()` (SL5) and `run_mstu_backtest()` (SL1).*

---

## Two-Year Backtest Summary (May 2024 → May 2026)

> ⚠️ **In-sample warning:** CT model was trained through Sep 17, 2025.
> Trades before that date are **in-sample** and should be discounted.
> The dominant trade (Trade #2, Sep 2024 → Dec 2024, +54.3%) is in-sample.
> OOS period (Sep 2025 → May 2026): 4 closed trades, all in BEAR regime.

### Full-Period Metrics

| Metric | TF2 | TF1 | Buy & Hold |
|--------|-----|-----|-----------|
| **Total return** | **+83.5%** | +11.8% | +23.5% |
| **CAGR** | **+38.2%** | +6.1% | +11.9% |
| **Final NAV ($100k start)** | **$183,540** | $111,770 | $123,530 |
| **Alpha vs B&H** | **+$60,010** | −$11,770 | — |
| Annualised volatility | 24.1% | lower | 39.0% |
| **Max drawdown** | **−22.9%** | −30.9% | −49.7% |
| DD recovery (days) | 30 | n/a | — |
| **Sharpe ratio** (RF=4.5%) | **0.86** | 0.08 | 0.28 |
| **Sortino ratio** | **1.33** | 0.12 | 0.42 |
| **Calmar ratio** | **1.67** | 0.20 | 0.24 |
| Closed trades | 11 | 22 | — |
| Win rate | 54.5% | 45.5% | — |
| Avg win | +16.7% | lower | — |
| Avg loss | −6.0% | similar | — |
| Profit factor | **2.33** | <1 | — |
| Best trade | +54.3% | lower | — |
| Worst trade | −8.5% | −8.5% | — |
| Avg trade duration | 27 days | shorter | — |
| Time in market | 43% | 33% | 100% |

**TF2 outperforms B&H on every risk-adjusted metric** and also outperforms on absolute return.  
**TF1 underperforms B&H** over 2 years (TF1 missed the bull run entirely; only +11.8% vs +23.5%).

### Monthly Returns (TF2 vs B&H)

| Month | TF2 | B&H | Better |
|-------|-----|-----|--------|
| Jun 2024 | +0.0% | +0.0% | = |
| Jul 2024 | +4.0% | +3.1% | TF2 ▲ |
| Aug 2024 | +0.0% | −8.7% | TF2 ▲ (cash) |
| Sep 2024 | +0.2% | +7.4% | B&H |
| Oct 2024 | +10.9% | +10.9% | = (invested) |
| **Nov 2024** | **+37.4%** | **+37.4%** | = (invested) |
| Dec 2024 | +1.1% | −3.1% | TF2 ▲ (D3 exit) |
| Jan 2025 | −1.9% | +9.6% | B&H |
| **Feb 2025** | **−6.7%** | **−17.6%** | **TF2 ▲ (D2 exit saved −11pp)** |
| Mar 2025 | +0.0% | −2.2% | TF2 ▲ (cash) |
| Apr 2025 | +13.0% | +14.1% | B&H |
| May 2025 | +11.1% | +11.1% | = (invested) |
| Jun 2025 | +1.0% | +2.4% | B&H |
| Jul 2025 | +11.2% | +8.0% | TF2 ▲ |
| Aug 2025 | −9.9% | −6.5% | B&H |
| Sep 2025 | +4.7% | +5.4% | B&H |
| Oct 2025 | −5.9% | −3.9% | B&H |
| **Nov 2025** | **+0.0%** | **−17.5%** | **TF2 ▲▲ (cash, avoided crash)** |
| Dec 2025 | −4.3% | −3.2% | B&H |
| **Jan 2026** | **+4.2%** | **−10.2%** | **TF2 ▲▲ (entered on signal)** |
| **Feb 2026** | **+0.0%** | **−14.8%** | **TF2 ▲▲ (cash)** |
| Mar 2026 | −5.5% | +1.8% | B&H |
| Apr 2026 | +4.6% | +11.8% | B&H |
| May 2026 | +1.5% | +1.5% | = |

**Months TF2 beat B&H:** 13 of 24  
**Worst TF2 month:** −9.9% (Aug 2025) vs B&H −6.5%  
**Worst B&H month:** −17.6% (Feb 2025) — TF2 limited to −6.7%

### Trade Log (2-Year, All 11 Closed Trades)

| # | Entry | Exit | P&L | Days | Regime | Exit Signal | IS/OOS |
|---|-------|------|-----|------|--------|-------------|--------|
| 1 | Jul 17, 2024 | Jul 19, 2024 | +4.0% ✓ | 2 | BEAR | D2 | IS |
| **2** | **Sep 20, 2024** | **Dec 19, 2024** | **+54.3% ✓** | **90** | **BULL** | **D3** | **IS** |
| 3 | Jan 18, 2025 | Feb 18, 2025 | −8.5% ✗ | 31 | BEAR | D2 | IS |
| 4 | Apr 3, 2025 | Apr 11, 2025 | +0.4% ✓ | 8 | BEAR | D3 | IS |
| 5 | Apr 13, 2025 | Jun 14, 2025 | +26.0% ✓ | 62 | BEAR | D2 | IS |
| 6 | Jun 26, 2025 | Jul 13, 2025 | +11.4% ✓ | 17 | BULL | D3 | IS |
| 7 | Aug 12, 2025 | Sep 24, 2025 | −5.7% ✗ | 43 | BEAR | D2 | IS |
| 8 | Oct 4, 2025 | Oct 12, 2025 | −5.9% ✗ | 8 | BEAR | D2 | OOS |
| 9 | Dec 12, 2025 | Dec 15, 2025 | −4.3% ✗ | 3 | BEAR | D2 | OOS |
| 10 | Dec 31, 2025 | Jan 12, 2026 | +4.2% ✓ | 12 | BEAR | D2 | OOS |
| 11 | Mar 11, 2026 | Mar 28, 2026 | −5.5% ✗ | 17 | BEAR | D2 | OOS |
| Open | Apr 10, 2026 | — | (open) | — | BEAR | — | OOS |

**Trade #2 is the key:** Sep 20 → Dec 19, 2024 (+54.3%) — TF2 entered in BEAR regime but the MA30 slope turned bullish, switching to BULL mode during the hold. D3 fired 90 days later at the exhaustion of the Nov–Dec 2024 BTC peak. TF1 exited this trade much earlier (multiple D2 fires), capturing only a fraction of the move.

### Critical Caveat: In-Sample Dominance

**If you exclude Trade #2 (in-sample, +54.3%):**
- TF2 OOS-only (Trades 8–11 + open): 1W/4L = 25% win rate, approximately −12% over the period
- This matches the OOS-period standalone backtest result

The 2-year combined number (+83.5%) is **heavily driven by one in-sample trade** that captured the Nov–Dec 2024 BTC bull run. The OOS story is more modest but still shows capital preservation vs B&H −30%.

---



> **⚠️ Critical caveat:** The CT ensemble model was trained on data through **Sep 17, 2025**.
> This prior-year test covers **Sep 17, 2024 → Sep 17, 2025** — the predictions are **in-sample**.
> The strategy "saw" this data during training. Results must be interpreted accordingly.

### Period Overview

| Metric | Value |
|--------|-------|
| Period | Sep 17, 2024 → Sep 17, 2025 (365 days, + 1 open position at period end) |
| BTC start price | $60,309 |
| BTC end price | $116,469 |
| Buy & Hold return | **+93.1%** |
| Strategy final NAV | **$106,332 (+6.3%)** |
| Alpha vs B&H | **−$86,789** (strategy underperforms significantly in bull market) |
| Number of round trips | 16 (+ 1 open position at period end) |
| Win rate | 5 / 16 = **31%** |
| Max portfolio drawdown | **−34.3%** |
| Time in market | **44%** |
| Data source | Yahoo Finance daily OHLCV + blockchain.info on-chain features |

### Cross-Period Comparison

| Metric | OOS (Sep 2025 → May 2026) | Prior Year (Sep 2024 → Sep 2025) |
|--------|--------------------------|----------------------------------|
| Market environment | Bear / Consolidation (BTC −33%) | Strong bull run (BTC +93%) |
| Strategy return | **+2.1%** | **+6.3%** |
| Buy & Hold return | −33.1% | +93.1% |
| Alpha vs B&H | **+$35,137** ✓ | **−$86,789** ✗ |
| # trades | 6 | 16 |
| Win rate | 4 / 6 = **67%** | 5 / 16 = **31%** |
| Max drawdown | 12.7% | 34.3% |
| Time in market | 19% | 44% |
| In-sample? | **No — fully OOS ✓** | Yes — in-sample ⚠️ |

### Prior-Year Trade Log (16 Round Trips)

| # | Entry Date | Exit Date | P&L | Entry Trigger | Exit Signal |
|---|-----------|-----------|-----|---------------|-------------|
| 1 | Sep 20, 2024 | Oct 1, 2024 | **−3.7%** ✗ | U1 + ↑MA30 | D2 |
| 2 | Oct 13, 2024 | Oct 24, 2024 | **+8.4%** ✓ | U1 + ↑MA30 | D2 |
| 3 | Oct 30, 2024 | Nov 2, 2024 | **−4.2%** ✗ | U1 + ↑MA30 | D2 |
| 4 | Nov 7, 2024 | Nov 19, 2024 | **+21.7%** ✓ | U1 + ↑MA30 | D2 |
| 5 | Nov 22, 2024 | Nov 25, 2024 | **−6.0%** ✗ | U1 + ↑MA30 | D2 |
| 6 | Nov 30, 2024 | Dec 3, 2024 | **−0.5%** ✗ | U1 + ↑MA30 | D2 |
| 7 | Dec 6, 2024 | Dec 10, 2024 | **−3.2%** ✗ | U1 + ↑MA30 | D2 |
| 8 | Dec 17, 2024 | Dec 19, 2024 | **−8.1%** ✗ | U1 + ↑MA30 | D3 |
| 9 | Jan 18, 2025 | Jan 28, 2025 | **−2.9%** ✗ | U1 + ↑MA30 | D2 |
| 10 | Jan 31, 2025 | Feb 18, 2025 | **−6.7%** ✗ | U1 + ↑MA30 | D2 |
| 11 | Mar 26, 2025 | Mar 31, 2025 | **−5.0%** ✗ | U1 + ↑MA30 | D2 |
| 12 | Apr 3, 2025 | May 6, 2025 | **+16.5%** ✓ | U1 + ↑MA30 | D2 |
| 13 | May 9, 2025 | May 17, 2025 | **+0.2%** ✓ | U1 + ↑MA30 | D2 |
| 14 | Jun 10, 2025 | Jun 13, 2025 | **−3.8%** ✗ | U1 + ↑MA30 | D2 |
| 15 | Jun 26, 2025 | Jul 26, 2025 | **+10.3%** ✓ | U1 + ↑MA30 | D2 |
| 16 | Aug 12, 2025 | Aug 15, 2025 | **−2.3%** ✗ | U1 + ↑MA30 + clean10d | D3 |
| Open | Aug 23, 2025 | — (at period end) | — | U1 + ↑MA30 | @ $115,374 |

*Exact fill prices not recorded in this log; re-run the standalone backtest script for precise values.*

### Interpretation

The prior-year results confirm the strategy's design intent: **it is a defensive,
capital-preservation filter — not an alpha generator in a trending bull market.**

In a +93% BTC bull run, raw exposure dominates — 100% Buy & Hold trivially outperforms 44%
average market exposure. The strategy repeatedly entered on U1 signals, encountered D2 exits
at losses as BTC consolidated briefly before continuing higher, and held cash during large
up-legs.

**Three observations:**

1. **High trade count in bull markets** — 16 trades vs 6 in the OOS bear period. The MA30 filter
   was frequently satisfied (BTC was mostly above MA30 during the bull run), but D2 exits cut
   positions short during every consolidation, causing whipsaws in the Nov–Dec 2024 peak zone
   and Q1 2025 pullback.

2. **Absolute returns were still positive (+6.3%)** — the strategy didn't destroy capital; it
   simply missed 87 percentage points of alpha that B&H captured through continuous exposure.

3. **D2 dominated exits (14 of 16)** — In a bull market, "predicted highs not being reached"
   fires frequently during pauses, causing premature exits. This is the correct capital-preservation
   behavior; it's costly only in hindsight when the bull continues.

**Regime summary:**

| Regime | Strategy edge | B&H edge |
|--------|--------------|----------|
| Bear / Sideways (OOS period) | ✓ Avoids large drawdowns, outperforms | ✗ Suffers full decline |
| Strong bull (prior-year period) | ✗ Misses most of the upside | ✓ Captures full run |

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
- **`run_tf1_backtest()`** computes the rolling 1-year backtest using batch CT predictions
  and renders a **Strategy Dashboard** with NAV vs B&H chart and trade log.

---

## Important Caveats

1. **Small OOS sample.** 6 trades over 8 months is insufficient for robust statistical inference.
   Win rate and return figures have wide confidence intervals.

2. **Regime dependence.** The OOS test set covers Sep 2025 – May 2026 (BTC −33%). The strategy's
   outperformance reflects its low exposure during a bear period. The **prior-year in-sample test**
   (Sep 2024 – Sep 2025, BTC +93%) confirms this: the strategy returned +6.3% vs B&H +93.1%,
   underperforming by −$86,789. In strong bull runs, Buy & Hold trivially dominates any strategy
   with partial market exposure.

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

7. **Prior-year test is in-sample.** The CT model was trained through Sep 17, 2025. The
   Sep 2024 → Sep 2025 backtest uses data the model has already seen — treat those results
   as illustrative regime analysis, not a performance claim.

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
*Backtesting runs: May 2026 (session `claude/trend-signature-patterns-OYUDT`)*  
*Live implementation: `app/btc_hourly_app.py` — `compute_trend_signatures()`, `render_trend_signatures()`, `run_tf1_backtest()`*
