# BTC Trend Signature Trading Strategy

**Document type:** Backtested trading strategy derived from trend signature patterns  
**Last updated:** 2026-07-04  

---

## ⭐ 2026-07d — Raise U1 to +1.3% to cut bear losers (CURRENT LIVE)

All three assets share the **same signal config**: 🎯 **Pure Regime** entry,
**U1 > +1.3%**, **D2 < −1.3%**, regime-adaptive D2/D3 exit, SL5 re-entry. Only the
**fixed stop** differs per asset. This raises U1 from +0.9% (2026-07c) to **+1.3%** — a
higher entry bar that filters the marginal entries which become bear-market losers.
The effect is consistent across every asset: **one fewer losing bear trade**, higher
bear return and shallower bear drawdown, while **bull capture is nearly maintained**
(a few points given up) and full-period return is flat-to-higher (MSTR/BTC up, MSTU
slightly down). Config lineage: 07c raised MSTU to Pure Regime / −3% stop; 07d raises U1.

| Asset | Gate | U1 | D2 | Stop | 🐂 Bull | 🐻 Bear | 🌐 Full | 07c (U1>0.9) |
|-------|------|----|----|------|---------|---------|---------|-----------|
| **MSTR** | Pure Regime | **+1.3%** | −1.3% | −3% | +125% | **+35%** | **+213%** | 131 / 27 / 205 |
| **MSTU** | Pure Regime | **+1.3%** | −1.3% | **−3%** | +281% | **+71%** | **+504%** | 260 / 66 / 534 |
| **BTC** | Pure Regime | **+1.3%** | −1.3% | none | +66% | **+7%** | **+102%** | 91 / 3 / 96 |

Bear losing trades per asset dropped from 3/3/2 (MSTR/MSTU/BTC at U1>0.9) to **2/2/1**
at U1>1.3; bear max-drawdown improved (e.g. BTC −16% → −9%, MSTR −22% → −18%).

**Bull-period end set to 2025-05-31** (was 2025-08-16): the bull window now ends the day
before the bear window begins (2025-06-01), so the two periods are contiguous and
non-overlapping and the split matches the Historical Replay tab. Any trade still open at the
cutoff is marked to its 2025-05-31 close, which drops the one summer-2025 trade the extended
window had captured (bull trade count 3 → 2 per asset) — raising MSTU bull sharply (+232% →
+281%), leaving MSTR ~flat (+124% → +125%) and lowering BTC bull (+89% → +66%). Bull B&H over
this window: BTC +52%, MSTR +142%, MSTU +59%.

Full-period B&H (unchanged, ends 2026-05-31): MSTR −2%, MSTU −76%, BTC +6%. All three beat
Buy & Hold on the full period.

**Important caveats (this is heavy in-sample optimization):**
- These are the best of a ~200-config sweep on **~7 trades per period**; the bull window
  is **in-sample** for the CT model, and MSTU's +504% rides ~2 large rally trades. The
  gains are real in-sample but the out-of-sample confidence interval is wide.
- MSTU's −3% stop on a **2× ETF** is deliberately tight; an even tighter −2% backtests
  higher but is a hair-trigger (likely overfit / high real-world whipsaw) and was rejected.
- Earlier turns evaluated a *trading-day-correct* grid that disagreed with the live 7-day
  grid; the numbers above use the **live grid** so they match the app tabs.

---

## ⭐ 2026-07 Re-tune — U1 > +1.1% / D2 < −1.3% on the 12:00-UTC bars

The signal thresholds were re-fit to the corrected 12:00-UTC bar timeline (the
same bars the live model and dashboard use). The previous thresholds
(U1 > +0.7% / D2 < −0.75%) were fit to the **retired midnight-UTC dataset**; when
that dataset was re-anchored to 12:00-UTC (so backtest = live = historical,
byte-identical predictions), the old thresholds fired too many marginal entries
on the noisier bars. Re-tuning raised the U1 entry bar and loosened the D2 exit:

- **U1 entry:** `err_hi_ma3 > +1.1%` AND `hi_breaks_3d ≥ 2`  *(was +0.7%)*
- **D2 exit:**  `err_hi_ma3 < −1.3%`  *(was −0.75%)*
- Single source of truth in `app/btc_hourly_app.py`: `U1_ERRHI_MIN`, `D2_ERRHI_MAX`.

The higher entry bar removes ~5 marginal U1 entries per period that were whipsaw
stop-outs while retaining both large rally trades; the looser D2 holds confirmed
trends longer. Result: higher return, roughly half the drawdown, and a much
stronger (partly-OOS) bear regime.

**Authoritative four-period results (re-tuned, 12:00-UTC bars, `bull_regime` gate,
SL5 re-entry, live stops — BTC none / MSTR −3% / MSTU −7%):**

| Asset | 🐂 Bull | 🐻 Bear | 🌐 Full | Full B&H | Full Sharpe / MaxDD |
|-------|---------|---------|---------|----------|---------------------|
| BTC   | +38.2%  | +8.6%   | **+50.2%**  | +6.2%   | +0.99 / −6.5%  |
| MSTR  | +60.0%  | +44.2%  | **+130.6%** | +4.4%   | +1.09 / −13.7% |
| MSTU  | +123.4% | +94.8%  | **+335.3%** | −86.9%  | +1.15 / −26.2% |

> These supersede the numbers in the older sections below, which reflect the prior
> U1 > +0.7% / D2 < −0.75% parameterization and/or the retired midnight-UTC dataset
> (e.g. the "+339.9% MSTR / +973.9% MSTU" figures were a midnight-bar artifact — see
> `backtest_retune_12utc.py` for the full recoverable-vs-never-real accounting).
> Older tables are retained for historical reference.

---

**Current live strategy:** TF2 (Regime-Adaptive) — supersedes TF1  
**OOS test window:** 2025-09-19 → 2026-05-17 (241 bars, fully out-of-sample)  
**In-sample test window:** 2024-09-17 → 2025-09-17 (365 bars, in-sample — model trained through this period)  
**Starting capital:** $100,000 USD  
**Signal source (historical backtest):** CT daily H/L — `artifacts/artifacts.pkl` (Ridge + GBM + RF)  
**Signal source (live dashboard):** CT daily H/L — `models/inference_assets_ct.joblib` (Huber + Quantile + GBM)

---

## Overview

This document records the development and results of a mechanical trading strategy built
on top of the four trend signature patterns described in [`TREND_SIGNATURES.md`](TREND_SIGNATURES.md).

Six strategy variants were tested in sequence, each refining the entry/exit rules.
The final strategy (**TF1**) is the only one that ended profitable in the test window and
is now implemented live in the Streamlit dashboard as the **🎯 STRATEGY BUY (TF1)** signal.

> **One-sentence summary:**  
> Buy when U1 fires and BTC is above its 30-day MA (or no bearish signals in the past 7 bars);
> sell only when D2 or D3 fires. Hold cash otherwise.

---

## Signal Definitions (Quick Reference)

| Signal | Condition | Type |
|--------|-----------|------|
| **U1** | `err_hi_ma3 > +1.3%` AND `hi_breaks_3d ≥ 2` | Uptrend — actual highs consistently exceed predictions |
| **D1** | `lo_breaks_3d ≥ 2` AND `err_lo_ma3 > 0.5%` | Downtrend — actual lows consistently break predicted floor |
| **D2** | `err_hi_ma3 < −1.3%` | Downtrend — predicted highs not being reached (exhaustion) |
| **D3** | Today is a `lo_break` AND ≥ 3 consecutive `hi_break` days immediately precede it | Reversal canary — momentum-to-reversal handoff |

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
| `clean_10d` | True if zero D1 or D2 fires in the 7 bars preceding the current bar (named `clean_10d` historically) — **entry is blocked when both `above_ma30` AND `clean_10d` are True simultaneously** (see combined-filter rule below) |

**Execution rule:** Signal fires on bar *i*; trade executes at bar *i* close (same-bar execution).

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

Buy at same-bar close (after-hours execution) when **all** of the following are true on the signal bar:

```
U1 is active:
    err_hi_ma3 > +0.5%
    hi_breaks_3d ≥ 2

AND exactly one of the following (XOR — combined is blocked):
    BTC close > 30-day rolling mean of close (above_ma30 = True)
    OR
    No D1 or D2 signal fired in any of the prior 7 bars (clean_10d = True)
    — but NOT both simultaneously (see Combined-Filter Rule below)

OR the V-reversal gate fires (overrides the combined block):
    V-reversal within the last 3 bars (dn_score > 0.8 AND err_lo > 3%)
```

### Exit Rule

Sell at same-bar close (after-hours execution) when **either** of the following is true on the signal bar:

```
D2: err_hi_ma3 < −1.3%  (predicted highs not being reached)
OR
D3: today is a lo_break AND ≥ 3 consecutive hi_break days immediately preceded it
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
`clean_10d = True` (no D1/D2 in prior 7 bars after the washout). The April recovery delivered
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
that the prior 7 bars contain no recent bearish fingerprint.

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

### Combined MA30+clean7d filter (entry blocked when both gates fire simultaneously)

When U1 fires and **both** `above_ma30` AND `clean_10d` are True at the same time, entry is now blocked (unless V-reversal gate also fires).

**Rationale:** Backtesting across Bear, Bull, and Full periods (Jun 2024 – May 2026) showed that when both conditions are simultaneously active, BTC has already been running a clean multi-week uptrend with no bearish signals for 7+ bars AND is trading above its 30-day MA. This is a **late-cycle momentum entry** — not a fresh breakout. Analysis across MSTR and MSTU showed:

| Trigger | n trades | Win rate | Avg P&L |
|---------|---------|----------|---------|
| U1+↑MA30+clean7d (combined — OLD) | 10 | **0%** | −4.1% |
| U1+↑MA30 only | 17 | 35% | +21.6% |
| U1+clean7d only | 2 | 100% | +4.0% |
| U1+V-reversal | 4 | 50% | +7.9% |

Pooled statistical test: combined vs MA30-only win rate difference is significant at **p = 0.033** (one-sided proportion z-test). All 10 combined trades resulted in stop-loss exits or breakeven D2/D3 exits with zero positive P&L.

**Why the combined condition signals late-cycle risk:** When the market is above MA30 AND has had no D1/D2 for 7 bars, it has already extended through a clean window. Any new U1 at that point is likely chasing a move that has been running for 1–2 weeks. The entry has poor risk/reward: limited upside (trend is mature) vs full stop-loss downside.

By contrast, a pure `U1+↑MA30` entry can fire after a brief consolidation (recent D1/D2 cleared), and a pure `U1+clean7d` fires when BTC is below MA30 but showing a fresh thrust after a washout — both are earlier-cycle with better entry timing.

**Implementation:** `tf1_entry = U1 AND ((above_ma30 XOR clean_10d) OR v_recent)`

**Impact (MSTR/MSTU, Jun 2024 – May 2026):**

| Period | Before | After (combined blocked) | Δ |
|--------|--------|--------------------------|---|
| 🐻 Bear MSTR | +2.7% | **+9.8%** | +7.1pp |
| 🐻 Bear MSTU | −5.0% | **+11.9%** | +16.9pp |
| 🐂 Bull MSTR | +198.7% | **+232.0%** | +33.3pp |
| 🐂 Bull MSTU | +532.1% | +532.1% | neutral |
| 🌐 Full MSTR | +219.4% | **+276.4%** | +57.0pp |
| 🌐 Full MSTU | +532.5% | **+644.8%** | +112.3pp |

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
| **Entry** | U1 + (↑MA30 XOR clean_10d, or V-rev) | Same ✓ |
| **Exit (Bear/Neutral regime)** | D2 or D3 | Same ✓ |
| **Exit (Bull regime)** | D2 or D3 | **D3 only** ← key change |

**Regime definition:**
```
BULL regime:   BTC close > MA30  AND  MA30[today] > MA30[5 bars ago]
BEAR/Neutral:  everything else
```

### Why This Works

In a **bull market**, D2 (`err_hi_ma3 < −0.75%`) fires repeatedly during brief consolidations —
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

## Four-Period Backtest Results — BTC, MSTR, MSTU (Jun 2024 – Jun 2026)

> **Methodology:** Same-bar execution (signal on bar *i*, trade at bar *i* close — after-hours fill for MSTR/MSTU) · $100,000 starting capital · **Stop-losses triggered on daily close, filled at close price** · 60-day pre-period signal warmup · **Combined MA30+clean7d entry blocked (XOR filter)** · Run: 2026-06-12

> ⚠️ **In-sample warning:** CT model trained through Feb 28, 2026. OOS period (Mar 2026 → present) is fully blind.

**Locked backtest periods:**

| Period | Window | BTC return (B&H) |
|--------|--------|-----------------|
| 🐻 Bear Market | Jun 2025 – May 2026 | −31.9% |
| 🐂 Bull Market | Jun 2024 – Jun 2025 | +85.1% |
| 🌐 Full Market | Jun 2024 – May 2026 | +29.1% |
| 🔬 OOS Only ⭐ | Rolling to yesterday | — (see live UI) |

---

### BTC — TF2 + V-Gate

#### Baseline (no stop loss)

| Period | Return | Sharpe | Max DD | Alpha vs B&H |
|--------|--------|--------|--------|--------------|
| 🐻 Bear (Jun 2025–May 2026) | **−4.2%** | −0.43 | −19.1% | **+27.7pp** vs B&H −31.9% |
| 🐂 Bull (Jun 2024–Jun 2025) | **+75.1%** | +1.47 | −17.1% | −10.0pp vs B&H +85.1% |
| 🌐 Full (Jun 2024–May 2026) | **+67.7%** | +0.76 | −24.1% | **+38.6pp** vs B&H +29.1% |
| 🔬 OOS | — | — | — | — (rolling, see live UI) |

#### With Trail −7% stop loss (no longer live — for reference only)

| Period | Return | Sharpe | Max DD | vs Baseline |
|--------|--------|--------|--------|-------------|
| 🐻 Bear | **−2.0%** | −0.36 | −17.3% | ▲ +2.2pp |
| 🐂 Bull | **+54.1%** | +1.18 | −17.0% | ▼ −21.0pp (trailing stop whipsaws bull dips) |
| 🌐 Full | **+51.0%** | +0.61 | −24.4% | ▼ −16.7pp |

**Note:** BTC no longer uses a stop loss. TF2 D2/D3 exits handle all adverse moves.

### BTC — Trade Log (11 Closed Trades, Jun 2024 – Jun 2026)

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

**Trade #2 is the key:** Sep 20 → Dec 19, 2024 (+54.3%) — TF2 captured the full Nov–Dec 2024 BTC bull run by holding in BULL regime (D3-only exit), while TF1 exited early on multiple D2 fires.

---

### MSTR — TF2 + V-Gate (BTC signals → MSTR execution)

BTC trend signals drive MSTR stock entries and exits. MicroStrategy holds ~580,000 BTC, making it a leveraged BTC proxy with equity liquidity. Live stop: **Fixed −3%** (close-price triggered, filled at close price).

#### Baseline (no stop loss)

| Period | Return | Sharpe | Max DD | Alpha vs B&H |
|--------|--------|--------|--------|--------------|
| 🐻 Bear (Jun 2025–May 2026) | **−7.5%** | −0.24 | −34.6% | **+53.1pp** vs B&H −60.6% |
| 🐂 Bull (Jun 2024–Jun 2025) | **+210.5%** | +1.62 | −31.8% | **+16.4pp** vs B&H +194.0% |
| 🌐 Full (Jun 2024–May 2026) | **+187.2%** | +0.97 | −41.8% | **+165.0pp** vs B&H +22.2% |
| 🔬 OOS | — | — | — | — (rolling, see live UI) |

#### With Fixed −3% stop loss + SL5 regime-adaptive re-entry (live strategy — close-price trigger) ⭐

*Stop-losses triggered on daily close, filled at close price. MSTR's high daily ATR means intraday triggers fire on normal consolidation; close-price triggering avoids false exits.*

| Period | Return | Sharpe | Max DD | vs Baseline |
|--------|--------|--------|--------|-------------|
| 🐻 Bear | **+9.8%** | — | — | ▲ vs B&H −60.6% |
| 🐂 Bull | **+232.0%** | — | — | ▲ vs B&H +125.9% |
| 🌐 Full | **+276.4%** | — | — | ▲ vs B&H −6.1% |
| 🔬 OOS | — | — | — | — (rolling, see live UI) |

**MSTR Fixed −3% close-price stop + SL5 re-entry beats the no-stop baseline on ALL 3 locked periods.** Positive Bear return (+9.8%) while B&H lost −60.6%. *(Updated 2026-06-12: combined MA30+clean7d entry filter applied — see "Combined MA30+clean7d filter" above)*

---

### MSTU — TF2 + V-Gate (BTC signals → MSTU execution)

MSTU is the T-Rex 2× Long MSTR ETF (inception Sep 18, 2024). For pre-inception dates (Jun–Sep 2024), the Sep 18 opening price is backward-filled (same as research script). Live stop: **Fixed −7%** (close-price triggered, filled at close price).

#### Baseline (no stop loss)

| Period | Return | Sharpe | Max DD | Alpha vs B&H |
|--------|--------|--------|--------|--------------|
| 🐻 Bear (Jun 2025–May 2026) | **−28.8%** | −0.29 | −62.0% | **+64.0pp** vs B&H −92.8% |
| 🐂 Bull (Jun 2024–Jun 2025) | **+325.6%** | +1.43 | −64.0% | **+115.6pp** vs B&H +210.0% |
| 🌐 Full (Jun 2024–May 2026) | **+202.9%** | +0.82 | −79.3% | **+278.9pp** vs B&H −76.0% |
| 🔬 OOS (Mar 2026–Jun 2026) | **+14.2%** | +0.75 | −33.5% | **+44.5pp** vs B&H −30.3% |

#### With Fixed −7% stop loss + SL5 re-entry (live strategy — close-price trigger) ⭐

*Note: Research evaluated all variants with −7% close-price stop (same as live). SL5 regime-adaptive is the clear winner across all three locked periods.*

| Period | Return | Sharpe | Max DD | vs Baseline |
|--------|--------|--------|--------|-------------|
| 🐻 Bear | **+11.9%** | — | — | ▲ vs B&H −91.8% |
| 🐂 Bull | **+532.1%** | — | — | ▲ vs B&H +210.0% |
| 🌐 Full | **+644.8%** | — | — | ▲ vs B&H −76.0% |
| 🔬 OOS | — | — | — | — (rolling, see live UI) |

**MSTU Fixed −7% close-price stop + SL5 re-entry beats the no-stop baseline on ALL 3 locked periods.** In BULL regime (BTC above MA30 AND MA30 rising), re-enter immediately on next valid TF2 signal; in BEAR/Neutral regime, wait 10 bars. *(Updated 2026-06-12: combined MA30+clean7d entry filter applied)*

---

## Stop-Loss Evaluation — All Configurations

Evaluated using `backtest_stop_loss.py` · Intraday triggering and fill · 1-bar lag execution · $100k start · Run: 2026-06-09
*(Historical reference — live strategy now uses close-price triggers per `backtest_stop_loss_reentry.py` evaluation)*

**Live strategy configs: BTC → No stop · MSTR → Fixed −3% · MSTU → Fixed −7%**

### BTC — Total Return by Period

| Stop Config | 🐻 Bear | 🐂 Bull | 🌐 Full | 🔬 OOS |
|-------------|--------|--------|--------|-------|
| Baseline (no SL) | −19.2% | +42.5% | +20.4% | −3.8% |
| **Fixed −3%** ★ | **−13.9%** ▲ | **+54.1%** ▲ | **+36.8%** ▲ | **+5.6%** ▲ |
| Fixed −5% | −18.4% ▲ | +34.2% ▼ | +15.2% ▼ | +1.3% ▲ |
| Fixed −7% | −17.7% ▲ | +45.1% ▲ | +24.8% ▲ | −1.4% ▲ |
| Fixed −10% | −19.2% = | +33.6% ▼ | +12.8% ▼ | −3.8% = |
| Trail −5% | −22.6% ▼ | −5.3% ▼ | −22.9% ▼ | −4.5% ▼ |
| **Trail −7% (live)** | **−17.1%** ▲ | **−0.8%** ▼ | **−14.0%** ▼ | **+7.0%** ▲ |
| Trail −10% | −17.7% ▲ | +10.5% ▼ | −4.9% ▼ | +0.2% ▲ |

★ **Fixed −3% is the only BTC config that beats baseline on all 4 periods.** Trail −7% (live) helps only in Bear and OOS.

### MSTR — Total Return by Period

| Stop Config | 🐻 Bear | 🐂 Bull | 🌐 Full | 🔬 OOS |
|-------------|--------|--------|--------|-------|
| Baseline (no SL) | −29.9% | +122.6% | +59.4% | +12.0% |
| **Fixed −3% (live)** ★ | **−0.6%** ▲ | **+143.4%** ▲ | **+149.5%** ▲ | **+23.1%** ▲ |
| Fixed −5% | −15.0% ▲ | +116.9% ▼ | +94.0% ▲ | +21.7% ▲ |
| Fixed −7% | −20.6% ▲ | +156.3% ▲ | +107.9% ▲ | +16.6% ▲ |
| Fixed −10% | −30.4% ▼ | +130.0% ▲ | +63.6% ▲ | +9.2% ▼ |
| Trail −5% | −24.7% ▲ | +14.6% ▼ | −9.7% ▼ | −10.6% ▼ |
| Trail −7% | −21.3% ▲ | +32.9% ▼ | +6.9% ▼ | +3.7% ▼ |
| Trail −10% | −33.0% ▼ | +3.0% ▼ | −29.5% ▼ | −3.7% ▼ |

★ **MSTR Fixed −3% beats baseline on ALL 4 periods** — the strongest result of any configuration across all assets.

### MSTU — Total Return by Period

| Stop Config | 🐻 Bear | 🐂 Bull | 🌐 Full | 🔬 OOS |
|-------------|--------|--------|--------|-------|
| Baseline (no SL) | −61.7% | +222.5% | +29.4% | +14.2% |
| Fixed −3% | −38.6% ▲ | +245.1% ▲ | +118.6% ▲ | −19.2% ▼ |
| Fixed −5% | −51.2% ▲ | +350.1% ▲ | +131.1% ▲ | −26.5% ▼ |
| **Fixed −7%** ★ | **−16.9%** ▲ | **+275.8%** ▲ | **+235.9%** ▲ | **+38.2%** ▲ |
| Fixed −10% (prev) | −33.5% ▲ | +198.7% ▼ | +120.9% ▲ | +39.1% ▲ |
| Trail −5% | +14.7% ▲ | +43.9% ▼ | +70.6% ▲ | +12.1% ▼ |
| Trail −7% | −15.1% ▲ | +47.8% ▼ | +32.5% ▲ | −2.8% ▼ |
| Trail −10% | −45.7% ▲ | −34.2% ▼ | −61.0% ▼ | −22.7% ▼ |

★ **MSTU Fixed −7% beats baseline on all 4 periods** — now the live stop. Fixed −10% (previous live) missed Bull by 23.8pp. Fixed −3% and −5% hurt OOS significantly.

### Key Findings

| Asset | Live Stop | Beats Baseline? | Final Decision |
|-------|-----------|-----------------|----------------|
| BTC | Trail −7% | 2/4 periods | **No stop loss** — TF2 D2/D3 exits handle adverse moves; trail stop adds whipsaw risk |
| MSTR | Fixed −3% ✓ | 4/4 periods | **Fixed −3% retained** + SL5 regime-adaptive re-entry (see below) |
| MSTU | Fixed −7% ✓ | 3/3 periods | **Fixed −7% retained** + SL5 regime-adaptive re-entry (see below) |

**Regime summary:**

| Regime | Strategy edge | B&H edge |
|--------|--------------|----------|
| Bear / Sideways (Bear period) | ✓ Avoids large drawdowns, outperforms on all three assets | ✗ Suffers full decline |
| Strong bull (Bull period) | ▲ Fixed −3%/−7% stops + SL5 re-entry outperform B&H (MSTR +232.0% vs B&H +125.9% ▲+106pp; MSTU +532.1% vs B&H +210.0% ▲+322pp) | ✓ Captures full run |

---

## Stop-Loss Re-Entry Criteria

After a stop-loss exit, re-entering naively on the next valid TF2 signal often leads to re-entering
the same down-move. A re-entry gate selectively blocks early re-entry to avoid whipsaw.

Evaluated using `backtest_stop_loss_reentry.py` · Same-bar execution (after-hours) · $100k start · Run: 2026-06-10

**7 variants tested (B0 = baseline no stop, SL0–SL5 = stop + various re-entry gates)**

### BTC — No Stop Loss

BTC's D2/D3 exit signals already handle adverse price moves effectively. The trailing −7% stop
adds inconsistent value: it helps in Bear/OOS periods but suppresses returns in Bull runs.
**Decision: remove stop loss from BTC entirely.** TF2 signals manage all exits.

### MSTR — SL5 Regime-Adaptive Re-Entry

| Variant | 🐻 Bear | 🐂 Bull | 🌐 Full | Notes |
|---------|--------|--------|--------|-------|
| B0 (no stop) | −7.5% | +210.5% | +187.2% | Baseline |
| SL0 (immediate re-entry) | +0.3% | +235.2% | +236.2% | Re-enter next bar after stop |
| SL1 (above SL exit price) | −5.0% | +158.8% | +145.7% | Re-enter when price ≥ stop exit |
| SL2 (above original entry) | −5.0% | +158.8% | +145.7% | Re-enter when price ≥ original entry |
| SL3 (BTC +2% recovery) | −18.1% | +165.4% | +117.3% | Re-enter when BTC up +2% from SL |
| SL4 (5-bar cooldown) | +0.3% | +235.2% | +236.2% | Fixed 5-bar wait |
| **SL5 (regime-adaptive)** | **+2.7%** | **+202.9%** | **+221.3%** | **Recommended ★** |

★ **SL5 gives the best Full-period return (+221.3% vs +187.2% baseline) and Bull return (+202.9%)**
while keeping Bear return positive (+2.7% vs +0.3% SL0). In BULL regime (BTC > MA30 AND MA30 rising), re-enter
immediately on next valid signal — the uptrend is intact. In BEAR/Neutral regime, wait 10 bars
before re-entering — avoids dead-cat-bounce whipsaw.

*Note: SL0–SL4 rows above were computed with the prior 1-bar lag execution model; SL5 reflects the current same-bar (after-hours) execution.*

**Implementation:**
- After stop exit: set `from_sl = True`, `bars_since_sl = 0`
- Each CASH bar: `bars_since_sl += 1`
- Re-entry gate: `not from_sl OR bull_regime[i] OR bars_since_sl >= 10`
- On re-entry: reset `from_sl = False`, `bars_since_sl = 0`

### MSTU — SL5 Regime-Adaptive Re-Entry

*Research used −7% close-price stop (same as live). SL5 regime-adaptive is the clear winner across all three periods, matching MSTR's re-entry logic.*

| Variant | 🐻 Bear | 🐂 Bull | 🌐 Full | Notes |
|---------|--------|--------|--------|-------|
| B0 (no stop) | −28.8% | +325.6% | +202.9% | Baseline |
| SL0 (immediate re-entry) | −11.1% | +408.2% | +351.9% | Re-enter next bar after stop |
| SL1 (above SL exit price) | −11.4% | +283.0% | +283.0% | Re-enter when price ≥ stop exit |
| SL2 (above original entry) | −11.4% | +283.0% | +283.0% | Re-enter when price ≥ original entry |
| SL3 (BTC +2% recovery) | −35.2% | +265.9% | +137.2% | Re-enter when BTC up +2% from SL |
| SL4 (5-bar cooldown) | −11.1% | +408.2% | +351.9% | Fixed 5-bar wait |
| **SL5 (regime-adaptive)** | **−5.0%** | **+530.7%** | **+531.1%** | **Recommended ★** |

★ **SL5 gives the best return across ALL periods** — Full +531.1% (vs +202.9% baseline), Bull +530.7%.
After a stop exit, in BULL regime (BTC > MA30 AND MA30 rising) re-enter immediately on next valid
signal — the uptrend is intact. In BEAR/Neutral regime, wait 10 bars before re-entering — avoids
dead-cat-bounce re-entries on the 2× leveraged instrument where drawdowns compound rapidly.

*Note: SL0–SL4 rows above were computed with the prior 1-bar lag execution model; SL5 reflects the current same-bar (after-hours) execution.*

**Implementation:**
- After stop exit: set `from_sl = True`, `bars_since_sl = 0`
- Each CASH bar: `bars_since_sl += 1`
- Re-entry gate: `not from_sl OR bull_regime[i] OR bars_since_sl >= 10`
- On re-entry: reset `from_sl = False`, `bars_since_sl = 0`

---

## Implementation in the Live Dashboard

The TF2 + V-Gate strategy with per-asset stop losses and re-entry criteria is live in `app/btc_hourly_app.py`:

- **`compute_trend_signatures()`** fetches 45 completed bars, computes the 30-bar MA, and
  determines `above_ma30`, `clean_10d`, and `tf1_triggered`.
- **`render_trend_signatures()`** displays a full-width TF1/TF2 card with live signal state.
- **`run_full_period_backtest()`** — BTC backtest with **no stop loss**. TF2 D2/D3 signals
  manage all exits. The trailing −7% stop was removed as it suppresses bull-market returns
  without consistent benefit.
- **`run_mstr_backtest()`** — MSTR backtest with **Fixed −3% stop** (`stop_px = entry × 0.97`),
  triggered when `mstr_close[i] < stop_px` (daily close, not intraday), filled at close price.
  After a stop exit, **SL5 regime-adaptive re-entry**: in BULL regime (BTC above MA30 AND MA30
  rising), re-enter immediately on next valid TF2 signal; in BEAR/Neutral regime, wait 10 bars.
  Close-price triggering avoids false stops on intraday noise (MSTR's daily ATR is 5–10%).
- **`run_mstu_backtest()`** — MSTU backtest with **Fixed −7% stop** (`stop_px = entry × 0.93`),
  triggered when `mstu_close[i] <= stop_px` (daily close, not intraday), filled at close price.
  After a stop exit, **SL5 regime-adaptive re-entry**: in BULL regime (BTC above MA30 AND MA30
  rising), re-enter immediately on next valid TF2 signal; in BEAR/Neutral regime, wait 10 bars.
  For pre-inception dates (before Sep 18, 2024), the Sep 18 opening price ($25.52)
  is backward-filled — matching the research script price source.
- All three backtests use **same-bar execution** (signal on bar i, trade at bar i close) — BTC trades
  24/7; MSTR/MSTU are bought/sold in after-hours once the BTC daily signal is confirmed.

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

4. **Execution at bar close.** The strategy assumes execution at the daily close on the same bar the
   signal fires. BTC trades 24/7 so same-day execution is realistic. MSTR/MSTU are assumed to be
   bought/sold in after-hours once the BTC daily signal is confirmed. After-hours spreads are wider
   than regular-session spreads — include this in cost estimates.

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
*Live implementation: `app/btc_hourly_app.py` — `compute_trend_signatures()`, `render_trend_signatures()`, `run_tf1_backtest()`*
