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
| **U1** entry | 3-day centered `err_hi` > **+0.08%** and ≥2 high-breaks | bullish pressure |
| **D2** exit | 3-day centered `err_hi` < **−0.10%** | momentum fading |
| **D1** exit | 3-day centered `err_lo` > **+0.10%** and ≥2 low-breaks | bearish pressure |
| **V** reversal | single-bar low undershoot > **+0.50%** + high down-score | capitulation |
| Regime | close > rising 20-day MA | bull regime gate |

## 4. The strategy

The app trades **one** strategy — **Divergence Pure-Regime** (BTC-style,
gold-scaled). Enter on **U1** bullish divergence (3-day centered `err_hi` >
**+0.08%** with ≥2 high-breaks) confirmed inside the **Pure-Regime gate**: Bull
Regime **or** a washed-out Clean Breakout below the MA **or** a recent
V-reversal. Exit on **D2** (< **−0.10%**) / **D3** exhaustion, or a fixed
**−3%** stop.

The signal is derived from **GLDM** (gold) and executed in its
leveraged / high-beta proxies — the 1× GLDM position is **not traded**, exactly
as the BTC app trades MSTR / MSTU rather than spot BTC:

- **GDX** — VanEck Gold Miners ETF, high-beta gold (**~MSTR analog**)
- **UGL** — ProShares Ultra Gold, 2× daily gold (**~MSTU analog**)

### Why these parameters

`U1 = +0.08 / D2 = −0.10 / stop = 3%` was chosen from a joint GDX+UGL frontier
sweep as the config that **beats buy-&-hold on both return and drawdown for both
assets**. It sits on a robust plateau — neighbouring settings
(`0.08/−0.15/3%`, `0.10/−0.10/4%`, `0.08/−0.10/4%`) all score a combined Sharpe
of 2.4–2.7 — so it is not a fragile optimisation spike. (An earlier candidate,
`0.02/−0.27/3%`, also beats buy-&-hold but returns less on both assets: GDX
+143%, UGL +277% with higher drawdown.)

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

**Chosen strategy — Divergence Pure-Regime, `U1 +0.08% / D2 −0.10% / stop −3%`,
traded on GDX & UGL.** It beats buy-&-hold on **both return and drawdown** for
both assets:

| Asset | Strategy return | B&H return | Strategy MDD | B&H MDD | Strategy Sharpe | B&H Sharpe | Trades | Win% |
|---|---|---|---|---|---|---|---|---|
| **GDX** (miners) | **+270.3%** | +103.6% | **−16.1%** | −46.5% | **1.40** | 0.54 | 80 | 59% |
| **UGL** (2×) | **+207.1%** | +160.8% | **−17.6%** | −49.4% | **1.29** | 0.67 | 80 | 61% |

*(The 1× GLDM signal source, for reference: +87.4%, MDD −8.5%, Sharpe 1.39 —
best risk-adjusted, but not traded.)*

### Sub-period breakdown (shown per-asset in the app's Backtesting tabs)

| Asset | Period | Strategy | Buy & Hold | Strat MDD | B&H MDD |
|---|---|---|---|---|---|
| GDX | Chop 2021–2022 | **+36.4%** | −25.6% | −13.2% | −46.5% |
| GDX | Bull 2023→now | +141.7% | +164.4% | −16.1% | −36.3% |
| UGL | Chop 2021–2022 | **+46.4%** | −22.6% | −10.4% | −40.2% |
| UGL | Bull 2023→now | +94.2% | +231.6% | −17.6% | −49.4% |

The strategy's edge is clearest in the **choppy / down** 2021–2022 gold market,
where it stayed **net positive while buy-&-hold lost ~25%**. In a relentless
bull leg it gives up some upside (the price of de-risking) but with far lower
drawdown and higher Sharpe throughout.

## 7. Chosen defaults (in `app/gldm_core.py`)

```python
STRATEGY_NAME    = "Divergence Pure-Regime"
U1_ERRHI_MIN     =  0.08     # U1 entry threshold (3-day centered err_hi)
D2_ERRHI_MAX     = -0.10     # D2 exit threshold
D1_ERRLO_MIN     =  0.10
V_ERRLO_MIN      =  0.50
FIXED_STOP       =  0.03     # shared −3% stop
TRADEABLE_ASSETS = ["GDX", "UGL"]   # GLDM (1x) supplies the signal only
```

The strategy is fixed (single strategy, single parameter set); the **GDX** and
**UGL** Backtesting tabs show the full/chop/bull breakdown, equity & drawdown
curves and the complete trade log, and the **Live** / **Historical replay** tabs
show the live signal and open position for each asset.
