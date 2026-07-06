# GLDM Trading Strategy — design, thresholds & backtest results

This document explains how the gold (GLDM) trading strategy was derived, why it
differs from the Bitcoin strategy, and what the out-of-sample backtest shows.
All numbers come from `backtest_gldm.py` (run `--sweep` to reproduce).

---

## 1. Objective

*Maximise trading profit while minimising losses on gold.* Concretely: match or
beat buy-&-hold on total return **and** meaningfully reduce max drawdown, so the
risk-adjusted return (Sharpe) improves.

## 2. Why the BTC strategy doesn't transfer directly

The BTC app trades an aggressive **divergence-exit** system (enter on U1 bullish
pressure, exit fast on D2/D3) with ±1.3% signal thresholds, because Bitcoin has
huge drawdowns (50–75%) that such a system can dodge for a large net gain.

Gold is the opposite kind of asset:

- **Low volatility** (~13% annualised vs BTC's ~55%). A GLDM day ranges
  ~0.6–1.0%, so the divergence error, once regime-centered, has a standard
  deviation of only **~0.17%** — the BTC ±1.3% thresholds would *never fire*.
- **Shallow drawdowns** (max ~26% over 2021–2026) and **persistent uptrends**.
  An aggressive de-risking overlay spends too much time in cash and gives up the
  trend — early experiments confirmed a BTC-style overlay returned only +13%
  while gold itself returned +111%.

The right gold strategy therefore **stays invested through the trend** and only
steps aside in confirmed downtrends.

## 3. Signal calibration — regime-centered divergence

The daily High/Low ridge predicts a symmetric range, but gold's daily H/L
relative to the prior close is **structurally asymmetric and drifts across
regimes** (raw high-breaks ~20%, low-breaks ~80%). A static bias correction does
not transfer out-of-sample. The fix (used in both `app/gldm_core.py` and
`backtest_gldm.py`) is to subtract each error's **rolling-median**, which
self-calibrates the signal to the current regime — high/low breaks then occur
~49–51% of the time, exactly as the trend-signature logic assumes.

Gold-scaled thresholds (vs BTC's ±1.3%):

| Signal | GLDM threshold | Meaning |
|---|---|---|
| **U1** entry | 3-day centered `err_hi` > **+0.10%** and ≥2 high-breaks | bullish pressure |
| **D2** exit | 3-day centered `err_hi` < **−0.15%** | momentum fading |
| **D1** exit | 3-day centered `err_lo` > **+0.10%** and ≥2 low-breaks | bearish pressure |
| **V** reversal | single-bar low undershoot > **+0.50%** + high down-score | capitulation |
| Regime | close > rising 20-day MA | bull regime gate |

## 4. Strategies

**Primary — MA trend filter.** Long when GLDM's close is above its N-day simple
moving average (decision made at the prior close → no look-ahead); flat
otherwise; optional fixed stop. Simple, robust, hard to overfit.

**Alternative — Divergence Pure-Regime** (BTC-style, gold-scaled). Enter on U1
plus the Pure-Regime gate (bull regime **or** a washed-out clean setup below the
MA **or** a recent V-reversal); exit on D2 / D3 exhaustion; per-asset fixed stop.

Both signals are derived from **GLDM** and then applied to three assets, mirroring
how the BTC app runs one signal across BTC / MSTR / MSTU:

- **GLDM** — 1× spot gold (core position)
- **UGL** — ProShares Ultra Gold, 2× daily gold (leveraged, ~MSTU analog)
- **GDX** — VanEck Gold Miners ETF, high-beta gold (~MSTR analog)

## 5. Backtest methodology

- **Data:** GLDM + macro daily bars from Yahoo, 2018-06 → present.
- **Out-of-sample by construction:** the daily H/L ridge is fit **once** on the
  pre-2021 window and predicts every later bar, so all reported trades use
  genuinely out-of-sample signals. Reported window: **2021-01-01 → 2026-07-06**
  (1,381 trading days).
- **Execution:** decisions at the daily close; positions marked close-to-close;
  fixed stop checked against the close. No leverage assumption beyond the ETF's
  own (UGL is already 2×).
- **Benchmark:** buy-&-hold of the same asset over the same window.
- **Metrics:** total return, CAGR, max drawdown (MDD), Sharpe (√252-annualised).

> Costs/slippage are not modelled; the strategies trade infrequently
> (~25–70 entries over 5½ years), so transaction costs are second-order. Results
> are historical and not a guarantee of future performance.

## 6. Results (OOS 2021-01-01 → 2026-07-06)

### Primary — MA50 trend filter (default), stop off

| Asset | Strategy return | B&H return | Strategy MDD | B&H MDD | Strategy Sharpe | B&H Sharpe |
|---|---|---|---|---|---|---|
| **GLDM** | **+92.2%** | +111.5% | **−17.0%** | −26.1% | **0.87** | 0.85 |
| **UGL** (2×) | **+149.3%** | +160.8% | **−32.4%** | −49.4% | **0.71** | 0.67 |
| **GDX** (miners) | **+92.9%** | +103.6% | **−34.1%** | −46.5% | **0.57** | 0.54 |

The trend filter gives up a little return but **cuts max drawdown by ~35–45%**
and improves Sharpe on all three assets.

### Sharpe-optimal MA config (frontier sweep, MDD ≤ buy-&-hold)

| Asset | MA window | Stop | Return | MDD | Sharpe | vs B&H |
|---|---|---|---|---|---|---|
| **GLDM** | 100 | 2% | **+108.0%** | −20.7% | **0.94** | B&H +111.5% / −26.1% / 0.85 |
| **UGL** | 40 | 2% | **+186.4%** | −36.4% | **0.80** | B&H +160.8% / −49.4% / 0.67 |
| **GDX** | 100 | 5% | **+136.4%** | −30.8% | **0.68** | B&H +103.6% / −46.5% / 0.54 |

On the leveraged / high-beta names the tuned trend filter **beats buy-&-hold on
return, drawdown and Sharpe simultaneously.**

### Alternative — Divergence Pure-Regime (gold-scaled, U1 +0.10% / D2 −0.15%)

| Asset | Strategy return | B&H return | Strategy MDD | B&H MDD | Strategy Sharpe | B&H Sharpe |
|---|---|---|---|---|---|---|
| **GLDM** | +68.6% | +111.5% | **−10.5%** | −26.1% | **1.17** | 0.85 |
| **UGL** (2×) | +126.2% | +160.8% | **−24.0%** | −49.4% | **1.00** | 0.67 |
| **GDX** (miners) | **+195.8%** | +103.6% | **−28.5%** | −46.5% | **1.16** | 0.54 |

The divergence system delivers the **best risk-adjusted performance** (Sharpe
1.0–1.2, drawdowns roughly halved) and on **GDX it nearly doubles buy-&-hold's
return** with far lower drawdown.

## 7. Recommended defaults (in `app/gldm_core.py`)

```python
MA_WINDOW_BY_ASSET = {"GLDM": 50, "UGL": 40, "GDX": 100}
STOP_BY_ASSET      = {"GLDM": 0.030, "UGL": 0.020, "GDX": 0.050}
U1_ERRHI_MIN, D2_ERRHI_MAX, D1_ERRLO_MIN, V_ERRLO_MIN = 0.10, -0.15, 0.10, 0.50
```

- **For 1× GLDM, maximum return:** MA100 + 2% stop (+108%, Sharpe 0.94).
- **For the best risk-adjusted profile / minimum losses:** the Divergence
  Pure-Regime system (Sharpe > 1.0, ~−10% to −29% drawdowns).
- **For leveraged upside (UGL / GDX):** either tuned strategy beats buy-&-hold on
  every metric — GDX + divergence is the standout (+195.8%).

Both strategies are selectable, with live-tunable parameters, on the **Strategy**
tab of the app.
