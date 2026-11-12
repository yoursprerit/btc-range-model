# GRID retune study (2026-11) — what was adopted, what was not

**Adopted (2026-11-12):** keep GRID's MACD 10/20/9 (−5% stop) and, **under Strategy
Logic V2 only**, take the signal only while the close is above its **100-day SMA**
(`TickerConfig.v2_price_gate_ma`). V1 stays the bare MACD.

**Why.** GRID's MACD whipsawed in the 2026 H2 down-trend: two good trades
(+12.5 %, +15.6 %) followed by six small losers (−0.6 % … −6.3 %) once the price sat
below its 200-day SMA, ending 2026 at ~+10 % while energy / semis carried the book.

## Result — Overall walk-forward replay (V2 allocator, 18 sleeves, GRID cap 30 %)
`scripts/eval_grid_portfolio.py`. Return / Sharpe / max drawdown, single optimiser
seed, coupon-rate idle yield, no costs.

| Balanced | old MACD | **MACD & close > SMA100** |
|---|---|---|
| Entire replay (2021→) | +1052 % · 1.83 · −14.1 % | +1033 % · 1.81 · −16.7 % |
| 2025→ | +259 % · 2.41 · −11.3 % | +264 % · 2.40 · −11.1 % |
| 2026 YTD | +66 % · 1.85 | **+80 % · 2.14** |
| Last 3 months | −7 % · −1.41 | **−2 % · −0.35** |

Growth: 2021→ +1296 % → **+1369 %**, 2026 +76 % → **+86 %**. Aggressive: 2021→ +1588 % →
+1538 %, 2026 +90 % → +95 %. Calendar years (Balanced, old → new): 2021 +42/+42,
2022 +3/+1, 2023 +42/+44, 2024 +54/+51, 2025 +116/+102, 2026 +66/+80.

Stand-alone GRID on the repo engine, V2 vs V1: 2025→ +21 % (Sharpe 0.90, DD −9 %) vs
+16 % (0.60, −20 %); 2021→ +76 % (0.96) vs +101 % (0.92); full history +173 % vs +258 %
at max DD −14 % vs −20 %. **The gate trades raw stand-alone return for Sharpe and
drawdown**; inside the portfolio the full-history result is a tie and the recent year is
better.

## What else was tested (and why it was not adopted)
* **SMA200 ±2 % band + 12 % trailing stop** (`ma_band` mode, kept in the engine):
  clearly the best *stand-alone* rule (full 2015→ +365 % vs +258 %, 2025→ +39 % vs +16 %,
  ~11 trades, plateau over SMA 100–250 × band 0–4 %, pre-2021 walk-forward picks the same
  family) — but **worse inside the portfolio** (Balanced 2021→ +859 % / 1.71 vs +1052 % /
  1.83): it is long ~78 % of days and correlated 0.64 with the rest of the book (old
  MACD: 49 %, 0.31), so it diversifies less. Script: `scripts/eval_grid_retune.py`.
* **Faster exits** (8 % trail, 50-day-SMA exit): cost 22–40 pts of 2025→ return, no help
  in the real Covid crash, far worse in chop.
* **SMA150 / SMA200 gate:** give up more 2021→ return (Balanced +907 % / +864 %).
* **Re-entry cool-down:** erratic (10 d and 20 d help, 15 d hurts) — path luck.
* **GRID cap 10 %:** neutral-to-mixed; **conviction tilt 1.0:** untestable (the replay
  ignores `FUNDAMENTAL_VIEW`).
* Re-tuned MACD 12/15/14: fine stand-alone, no better than the old MACD in the portfolio.

## Honest limits
Gains are small (a few % – 15 pts) and inside single-seed allocator noise; ~12 variants
were run on the same history, so some selection bias remains. The mechanism (no MACD
trading in a down-trend) is simple and the gate also helped on 2015-2020 data that never
informed the choice (stand-alone Sharpe 0.29 → 0.37, DD −35 % → −21 %). The cost is a
slightly deeper worst drawdown in Balanced (−14.1 % → −16.7 %) and ~14 pts of 2025's run.
Weight guidance: GRID adds little at any size — target ~3–5 %, cap ≤ 10 %.
