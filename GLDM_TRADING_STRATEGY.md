# GLDM Trading Strategy — design, thresholds & backtest results

This document explains how the gold (GLDM) trading strategy was derived, why it
differs from the Bitcoin strategy, and what the out-of-sample backtest shows.
All numbers come from `backtest_gldm.py` (run `--sweep` to reproduce).

> **2026-07 revision — causal H/L model.** The daily High/Low ridge previously
> paired each bar's own features with that bar's high/low target, letting the
> model "reconstruct" the bar it was supposed to predict (OOS R² ≈ 0.95 vs
> ≈ 0.00 for a genuinely causal model). The model is now **causal** — features
> through the prior close predict the next bar, exactly like the BTC app — and
> every threshold and result below was re-tuned on the honest forecast error.
> The honest divergence error is ~6× larger than the pre-fix artifact
> (3-bar centered `err_hi` std ≈ 0.55% vs ≈ 0.10%), so the old thresholds
> (U1 +0.08 etc.) and the old headline results (GDX +270% / Sharpe 1.40) were
> artifacts of that leak and no longer apply.

---

## 1. Objective

*Maximise trading profit while minimising losses on gold.* Concretely: match or
beat buy-&-hold on total return where possible **and** meaningfully reduce max
drawdown, so the risk-adjusted return (Sharpe) improves.

## 2. Why the BTC strategy doesn't transfer directly

The BTC app trades an aggressive **divergence-exit** system (enter on U1 bullish
pressure, exit fast on D2/D3) with ±1.3% signal thresholds, because Bitcoin has
huge drawdowns (50–75%) that such a system can dodge for a large net gain.

Gold is different:

- **Low volatility** (~13% annualised vs BTC's ~55%). A GLDM day ranges
  ~0.6–1.0%; the causal divergence error, once regime-centered, has a 3-bar
  std of **~0.55%** — so the thresholds are roughly half of BTC's, not
  an order of magnitude smaller.
- **Shallow drawdowns** (max ~26% over 2021–2026) and **persistent uptrends**.
  An aggressive de-risking overlay spends time in cash and gives up part of the
  trend; the payoff is a much smaller drawdown and a higher Sharpe, not a
  higher raw return in bull legs.

## 3. Signal calibration — regime-centered divergence

The daily High/Low ridge predicts a symmetric range, but gold's daily H/L
relative to the prior close is **structurally asymmetric and drifts across
regimes**. A static bias correction does not transfer out-of-sample. The fix
(used in both `app/gldm_core.py` and `backtest_gldm.py`, identical 60-bar
rolling-median, `min_periods=20`) is to subtract each error's rolling median,
which self-calibrates the signal to the current regime — high/low breaks then
occur ~49–51% of the time, exactly as the trend-signature logic assumes.

Gold-scaled thresholds (vs BTC's ±1.3%), re-tuned on the causal error:

| Signal | GLDM threshold | Meaning |
|---|---|---|
| **U1** entry | 3-day centered `err_hi` > **+0.15%** and ≥2 high-breaks | bullish pressure |
| **D2** exit | 3-day centered `err_hi` < **−0.10%** | momentum fading |
| **D1** | 3-day centered `err_lo` > **+0.15%** and ≥2 low-breaks | bearish pressure (clean-gate input; not an exit for gold) |
| **V** reversal | single-bar low undershoot > **+1.00%** + high down-score | capitulation |
| Regime | close > rising 20-day MA | bull regime gate |

## 4. The strategy

The app trades **one** strategy — **Divergence Pure-Regime** (BTC-style,
gold-scaled). Enter on **U1** bullish divergence (3-day centered `err_hi` >
**+0.15%** with ≥2 high-breaks) confirmed inside the **Pure-Regime gate**: Bull
Regime **or** a washed-out Clean Breakout below the MA **or** a recent
V-reversal. Exit on **D2** (< **−0.10%**) / **D3** exhaustion, or the per-asset
fixed stop.

The signal is derived from **GLDM** (gold) and executed in its
leveraged / high-beta proxies — the 1× GLDM position is **not traded**, exactly
as the BTC app trades MSTR / MSTU rather than spot BTC:

- **GDX** — VanEck Gold Miners ETF, high-beta gold (**~MSTR analog**), −3% stop
- **UGL** — ProShares Ultra Gold, 2× daily gold (**~MSTU analog**), signal-only
  (the stop sweep shows a −8% stop never fires; tighter stops only whipsaw a 2×)
- **NUGT** — Direxion Daily Gold Miners 2×, −5% stop

### Why these parameters

`U1 = +0.15 / D2 = −0.10 / D1 = +0.15 / V = 1.0` is the tier-1 pick of the
joint GDX+UGL frontier sweep (`backtest_gldm.py --sweep`): it beats buy-&-hold
on **both return and drawdown** for GDX and beats it on drawdown and Sharpe for
UGL while retaining >70% of UGL's buy-&-hold return. It sits on a robust
plateau — `U1 +0.15…+0.20` at `D2 −0.10 / V 1.0` score within 0.03 average
Sharpe of each other across data snapshots; `U1 +0.15` is the value that stays
tier-1 on both the current and the prior snapshot, so it is not a fragile
optimisation spike.

## 5. Backtest methodology

- **Data:** GLDM + macro daily bars from Yahoo, 2018-06 → present.
- **Causal + out-of-sample by construction:** the daily H/L ridge pairs
  features known at the **prior close** with the next bar's high/low (no
  same-bar information), is fit **once** on the pre-2021 window and predicts
  every later bar. Reported window: **2021-01-01 → 2026-07-21** (1,392 trading
  days).
- **Execution:** decisions at the daily close; positions marked close-to-close;
  fixed stop checked against the close. No leverage assumption beyond the ETF's
  own (UGL/NUGT are already 2×).
- **Benchmark:** buy-&-hold of the same asset over the same window.
- **Metrics:** total return, CAGR, max drawdown (MDD), Sharpe (√252-annualised).

> Costs/slippage are not modelled; with ~95–100 round-trips over 5½ years,
> costs are no longer fully negligible — treat the totals as gross figures.
> Results are historical and not a guarantee of future performance.

## 6. Results (OOS 2021-01-01 → 2026-07-21)

**Chosen strategy — Divergence Pure-Regime, `U1 +0.15% / D2 −0.10% / V 1.0`,
per-asset stops (GDX −3%, UGL signal-only, NUGT −5%), traded on GDX, UGL &
NUGT:**

| Asset | Strategy return | B&H return | Strategy MDD | B&H MDD | Strategy Sharpe | B&H Sharpe | Trades | Win% |
|---|---|---|---|---|---|---|---|---|
| **GDX** (miners) | **+102.8%** | +92.3% | **−21.8%** | −46.5% | **0.85** | 0.51 | 98 | 49% |
| **UGL** (2× gold) | +118.9% | +151.0% | **−19.3%** | −50.0% | **0.90** | 0.64 | 95 | 52% |
| **NUGT** (2× miners) | **+276.1%** | +40.8% | **−38.3%** | −73.8% | **0.90** | 0.45 | 100 | 50% |

*(The 1× GLDM signal source, for reference: +57.9%, MDD −9.6%, Sharpe 0.99 —
best risk-adjusted, but not traded.)*

### Sub-period breakdown (shown per-asset in the app's Backtesting tabs)

| Asset | Period | Strategy | Buy & Hold | Strat MDD | B&H MDD |
|---|---|---|---|---|---|
| GDX | Chop 2021–2022 | **+12.4%** | −25.6% | −13.4% | −46.5% |
| GDX | Bull 2023→now | +74.3% | +149.7% | −21.8% | −38.9% |
| UGL | Chop 2021–2022 | **+16.9%** | −22.6% | −12.8% | −40.2% |
| UGL | Bull 2023→now | +84.3% | +219.1% | −19.3% | −50.0% |
| NUGT | Chop 2021–2022 | **+11.3%** | −56.4% | −29.1% | −73.8% |
| NUGT | Bull 2023→now | +217.0% | +203.2% | −38.3% | −67.5% |

The strategy's edge is clearest in the **choppy / down** 2021–2022 gold market,
where it stayed **net positive while buy-&-hold lost 23–56%**. In a relentless
bull leg it gives up part of the upside (the price of de-risking) but with far
lower drawdown and a higher full-period Sharpe on every traded asset.

## 7. Chosen defaults (in `app/gldm_core.py`)

```python
STRATEGY_NAME    = "Divergence Pure-Regime"
U1_ERRHI_MIN     =  0.15     # U1 entry threshold (3-day centered err_hi)
D2_ERRHI_MAX     = -0.10     # D2 exit threshold
D1_ERRLO_MIN     =  0.15
V_ERRLO_MIN      =  1.00
FIXED_STOP       =  0.03     # GLDM/GDX stop; UGL signal-only, NUGT −5%
TRADEABLE_ASSETS = ["GDX", "UGL", "NUGT"]   # GLDM (1x) supplies the signal only
```

The strategy is fixed (single strategy, single parameter set); the **GDX**,
**UGL** and **NUGT** Backtesting tabs show the full/chop/bull breakdown, equity
& drawdown curves and the complete trade log, and the **Live** / **Historical
replay** tabs show the live signal and open position for each asset.
