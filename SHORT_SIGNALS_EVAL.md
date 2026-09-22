# Can the downtrend signatures be traded short?

**Question:** Can any of the bearish signals built across the repo be used to
short the assets for consistent, risk-controlled profit? The signals tested
are the CT D1/D2/D3 "downtrend signatures", the ETF and gold divergence
D1/D2/D3, and the trend-family "below trend" state.

**Answer: no, not reliably.** No signal family gives a short edge that holds
up after costs in both the in-sample and out-of-sample windows. Adding a short
leg to the live long/flat engines **lowers Sharpe in every instrument tested**,
and deepens the max drawdown in almost all of them. The signatures do carry
information, but it is about **volatility, not direction**. After a bearish
signal the next ten bars are more volatile, and a big **up** move is *more*
likely than a big down move. This is the V-reversal / short-squeeze risk that
TREND_SIGNATURES.md §2 already describes. The right way to use these signals is
the way the repo already uses them: **step aside (go flat)**, not short.

One narrow lead is worth paper-tracking and nothing more: **D3 "exhaustion
canary" on the gold miners (GDX/NUGT)**. See §5.

Reproduce (offline, about 15 s, deterministic):
```bash
python scripts/eval_short_signals.py            # tables → stdout, CSVs → artifacts/
python scripts/eval_short_signals.py --panels   # list the signal panels only
```
The outputs are `artifacts/short_signals_eval.{events,grid,overlay,vol}.csv`
on the 2026-09-21 data vintage.

---

## 1. What was tested

The signals are rebuilt with **each app's own engine code**. Nothing is
reimplemented, so they match the live apps bar for bar.

| Family | Instruments | Bearish reads tested | Out-of-sample window |
|---|---|---|---|
| CT divergence (`btc_ct_engine.compute_sigs_pure`) | BTC · ETH · MSTR · MSTU | D1, D2, D3, HIGH_DN (≥2 of D1/D2/D3), ANY_DN, EXIT_DN (the live exit), D_BEAR (any D and below MA30) | **2026-03-01 →** (the CT ensemble's `train_end` is 2026-02-28) |
| ETF divergence (`backtest_ticker`) | PBW · ARTY | same set | 2021-01-01 → (ridge fit strictly pre-2021) |
| Gold divergence (`backtest_gldm`) | GDX · NUGT | same set | 2021-01-01 → |
| Trend (`trend_long_array`) | SOXX · SOXL · GRID · REMX · WGMI · XLE · OIH · ERX · GLDM · UGL | TREND_OFF (filter off), TREND_OFF_CONFIRMED (off, below a falling SMA50), FRESH_CROSS_DN | 2021-01-01 → |

> ⚠️ **The CT in-sample window is truly in-sample.** The CT ensemble was
> trained through 2026-02-28, so every BTC/ETH/MSTR/MSTU signal before March 2026
> comes from a model that has seen those bars. Only the roughly 200 bars from
> March to September 2026 are out-of-sample. That makes the CT result below
> *stronger* evidence: the signals fail to predict direction **even in-sample**.

**Execution realism:**
- Fills use the engines' own no-look-ahead conventions: the divergence
  signals are lagged one bar, and equity fills for MSTR/MSTU land on the next
  session after the signal.
- Shorts are simulated as a **real, non-rebalanced short** on the entry
  notional.
- Costs are 10 bp per side, plus annual borrow: 0% for BTC/ETH perps (a
  neutral assumption), 0.3–0.5% for liquid 1× ETFs, 2–5% for thin ETFs and
  MSTR, 3% for the 2–3× ETFs, and 20% for MSTU.
- Stops are close-based buy-stops.

**Rule grid, with every cell reported (nothing picked in-sample):**
- Entry on each bearish read.
- Exit on a bullish read (U1 or trend back on), on a max hold of 5, 10 or 20
  bars, or while the bear regime persists ("state").
- Each variant run with no stop and with a +8% stop.
- That gives **1,312 backtests** (680 IS/OOS pairs) and **456 event-study cells**.

---

## 2. Event study: does price fall after a bearish read?

The **short edge** is the mean forward return over all bars minus the mean
forward return after a signal. The p-value comes from a circular-shift
permutation, which rotates the signal so its clustering is kept and gives a
fair null for a sticky signal.

5-bar horizon:

| Family | Window | Cells | Edge > 0 | **Significant (p<0.05)** | Median edge (pp) |
|---|---|---|---|---|---|
| CT divergence | IS | 28 | 64% | **0%** | +0.24 |
| CT divergence | OOS | 24 | 71% | **0%** | +1.17 |
| ETF divergence | IS | 14 | 29% | 7% | −0.08 |
| ETF divergence | OOS | 14 | 71% | 7% | +0.06 |
| Gold divergence | IS | 14 | 14% | 14% | −1.32 |
| Gold divergence | OOS | 14 | 14% | 0% | −0.26 |
| Trend | IS | 24 | 33% | 8% | −0.24 |
| Trend | OOS | 20 | 80% | 0% | +0.42 |

Across all 228 five-bar cells, 6 reach p < 0.05. Chance alone would produce
about 11. For the headline CT signals, **none of 52 cells** is significant.

Even in-sample, BTC, MSTR and MSTU rose on average in the five bars after
D1, D3 and HIGH_DN:

| Five bars after… (IS) | BTC | MSTR | MSTU |
|---|---|---|---|
| D1 | +0.29% | +1.69% | +2.68% |
| D3 | +0.20% | +5.66% | +12.7% |
| HIGH_DN | −0.12% | +2.91% | +5.10% |

The "2.24× lift" in TREND_SIGNATURES.md is a lift in the *hit rate of big-down
days*. It is not a negative expected return, because big-up days become more
frequent too (see §4).

---

## 3. Short-rule backtests (after costs)

| Family | Window | Cells | Profitable | Sharpe > 0.5 | Median Sharpe | Median max drawdown |
|---|---|---|---|---|---|---|
| CT divergence | IS | 216 | 27% | 12% | 0.08 | −52% |
| CT divergence | OOS | 216 | 49% | 38% | 0.02 | −21% |
| ETF divergence | IS | 112 | **0%** | 0% | −0.73 | −49% |
| ETF divergence | OOS | 112 | 14% | 0% | −0.23 | −56% |
| Gold divergence | IS | 112 | 11% | 7% | −0.88 | −66% |
| Gold divergence | OOS | 112 | 11% | 2% | −0.36 | −72% |
| Trend | IS | 216 | 14% | 4% | −0.31 | −43% |
| Trend | OOS | 216 | 13% | 0% | −0.20 | −28% |

**Only 8 of 680 rule/asset pairs reach Sharpe > 0.5 in both windows.** Four
of those are MSTU 20-bar holds whose in-sample NAV fell to between −79% and
−107%, which is a wipe-out and not an edge. The best out-of-sample CT cells
(MSTR/MSTU "state" shorts, Sharpe about 1.8) rest on **3 trades** each. The
median CT out-of-sample cell has about 5 trades, because 200 bars cannot
support a conclusion.

---

## 4. Why shorting fails: the signals forecast volatility, not direction

In the table below, **Vol ratio** is realised volatility over the next 10 bars
after a signal divided by the unconditional level. **Big up** and **Big down**
are the share of signals followed by a 10-bar move beyond ±1σ.

| Family | Vol ratio (median) | Cells where vol is significantly higher | Big up | Big down |
|---|---|---|---|---|
| CT divergence | 1.08× | 18% | **16.6%** | 8.7% |
| ETF divergence | 1.11× | 86% | **17.6%** | 15.3% |
| Gold divergence | 1.04× | 14% | **20.3%** | 11.5% |
| Trend | 1.19× | 59% | **22.4%** | 14.6% |

Examples:
- **BTC after D1:** 17.2% of signals see a big up move and 11.7% a big down move.
- **MSTU after HIGH_DN:** 16.9% up vs 4.6% down.

A bearish signature marks a **high-volatility, two-sided** moment. The
largest moves that follow are often the capitulation-reversal rallies.
TREND_SIGNATURES.md §2 identifies these rallies as the setup the long engine's
V-reversal gate is built to *buy*. A short held through them is squeezed. On
top of that, every asset here has positive long-run drift, and the short leg
pays borrow on it.

---

## 5. Adding a short leg to the live engines

The comparison is "long when the engine is long, short instead of flat when
it is bearish" against the current long/flat engines. The windows are the
full out-of-sample window for the trend sleeves, and IS/OOS for the CT
sleeves.

| Instrument | Current long/flat Sharpe · max drawdown | + short leg Sharpe · max drawdown |
|---|---|---|
| BTC (OOS) | **2.10** · −4.8% | 1.09 · −11.0% |
| MSTR (OOS) | **1.50** · −11.8% | 0.45 · −30.1% |
| MSTU (OOS) | **1.48** · −22.8% | 0.38 · −52.3% |
| ETH (OOS) | **0.87** · −7.0% | 0.37 · −18.9% |
| SOXX | **1.05** · −29.0% | 0.75 · −36.3% (confirmed-only: 0.61) |
| SOXL | **1.00** · −69.4% | 0.73 · −79.9% |
| GRID | **0.69** · −23.0% | 0.08 · −37.0% |
| WGMI | **0.84** · −32.2% | 0.10 · −92.5% |
| REMX | **0.48** · −41.0% | 0.44 · −53.7% |
| GLDM | **1.05** · −19.1% | 0.92 · −27.0% (confirmed-only: 0.99 · −22.7%, +9 pp return) |
| XLE / OIH / ERX | 0.88 / 0.58 / 0.88 | 0.82 / 0.54 / 0.82 |

The short leg lowers Sharpe **in every instrument tested**. The closest to a
break-even is the gold trend confirmed-short (GLDM/UGL). It adds a few points
of return for a lower Sharpe and a deeper drawdown, so it is not an
improvement on a risk-adjusted basis.

### The one lead: D3 on the gold miners (paper-track only)

D3 (the first low-band break after ≥3 high-band breaks) on the GLDM
divergence signal, shorting GDX/NUGT, is the only rule that is
**positive in both windows** and whose timing beats a random-entry null. The
null uses the same rule, trade count and cluster shape with the entry timing
rotated, 300 draws.

| Out-of-sample 2021→ | Trades | Win rate | CAGR | Sharpe | Max drawdown | Timing p |
|---|---|---|---|---|---|---|
| GDX, hold 10, +8% stop | 28 | 61% | 2.8% | 0.30 | −22.9% | 0.017 |
| NUGT, hold 5, +8% stop | 28 | 68% | 8.9% | 0.62 | −27.7% | 0.010 |
| NUGT, hold 10, +8% stop | 28 | 64% | 8.8% | 0.54 | −28.9% | 0.003 |

This is not a "high, reliable profit":
- The returns are small against the drawdowns.
- The sample is 28 trades over 5.7 years.
- It is 1 of about 20 signal families tested, so some survivor bias is expected.
- NUGT's real borrow can be well above the 3% assumed here.

The fact that the ARTY and PBW D3 read fails as a short (0% of cells
profitable) argues against it being a general property of D3. If anything is
pursued, it should be a **paper-tracked GDX/NUGT D3 short at small size**,
judged only on trades taken after today.

---

## 6. What *can* raise profits reliably

1. **Keep "bearish ⇒ flat", not "bearish ⇒ short".** Flat is where the current
   Sharpe advantage comes from. Idle cash already earns the SATA yield, about
   13%, which beats every short sleeve's CAGR above.
2. **Use the signatures as a *volatility* input.** They reliably flag higher
   forward volatility, which suits position sizing: trim leveraged siblings
   such as MSTU, SOXL and NUGT when D-signals cluster, or widen stops. Another
   option is an options overlay that is long volatility rather than short
   direction, such as buying puts or put spreads while a HIGH_DN is live. This
   would be the next study.
3. **If a hedge is wanted**, hedge the *portfolio* (for example, a small index
   put position while several sleeves show HIGH_DN), not individual names.
   Single-name shorts on MSTR/MSTU, the miners and the 3× ETFs carry the
   largest squeeze and borrow risk in the universe.
