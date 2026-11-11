# GRID retune (2026-11) — MACD 10/20/9 → 200-day SMA ±2% band + 12% trail

**Why.** GRID's MACD 10/20/9 (−5% stop) had stopped working: 2025→now it returned
+16% vs Buy&Hold +34% (Sharpe 0.60 vs 0.77) and the 2026 pull-back cost it a −20%
drawdown, with ~60 round-trips since 2021 (whipsaw, fee-sensitive).

**What was searched** (`scripts/eval_grid_retune.py`, ~2,700 candidates on the full
2015→now history, repo fill convention, 10 bp/side): MACD (fast/slow/signal) × stops
× trailing stops, crash-shield (X% off N-day high / re-entry SMA), plain and **banded**
SMA (100–250d, 0–4% entry/exit band), dual-MA, MACD∧SMA / MACD∨SMA hybrids, MACD with
a SMA-cross hold rule.

**Result — real engine (`backtest_ticker.run_strategy`, V2, no fees, per-window restart)**

| window | old MACD 10/20/9 −5% | **new SMA200 ±2% + 12% trail** | Buy&Hold |
|---|---|---|---|
| Full 2015→now | +258% · Sh 0.90 · DD −19.8% | **+365% · 0.93 · −16.7%** | +394% · 0.76 · −40.6% |
| OOS 2021→now | +101% · 0.92 · −19.8% | +101% · 0.80 · −16.7% | +98% · 0.64 · −30.4% |
| Bull 2023→now | +57% · 0.94 · −19.8% | **+82% · 0.95 · −13.7%** | +83% · 0.85 · −23.9% |
| Recent 2025→now | +16% · 0.60 · −19.8% | **+39% · 0.99 · −13.7%** | +34% · 0.77 · −23.9% |
| Bear 2021–22 | **+28%** · 0.89 · −15.8% | +11% · 0.43 · −16.7% | +8% · 0.29 · −30.4% |
| Trades (full) | 117 | 11 | – |

Independent standalone study (with fees): full-history Sharpe 0.52 → 0.84, 2025→ Sharpe 0.44 → 0.98.

**Why this one, not the highest-scoring row**
* **Plateau, not a spike:** every SMA 100–250 × band 0–4% cell has a full-history
  Sharpe of 0.52–0.79 (MACD baseline 0.52); 200d ±2% sits in the middle of it.
* **Walk-forward:** choosing by Sharpe on pre-2021 data only (or pre-2023) selects
  the SMA200 band family (±3%, 12% trail) and it holds up on the unseen years
  (2021→ +82%, Sharpe 0.70; 2025→ +34%, Sharpe 0.90). Median unseen-window Sharpe of
  the band grid beats the MACD grid (0.64 vs 0.43 from 2021; 0.77 vs 0.52 from 2023).
* **Low turnover:** ~11 trades in 11 years vs ~117 — robust to costs/slippage.
* **Trailing stop instead of −5% fixed:** a fixed −5% cuts winners on a name that
  routinely pulls back 5%; the 12% trail (V2 `v2_trail_stop`) only caps true breaks.

**Honest trade-offs**
* **2021–22 bear:** the slow SMA is late into a bear tape — +11% vs MACD's +28%
  (still above B&H, same drawdown). Full-history and recent numbers more than pay
  for it, but a 2022-style regime would favour the old fast filter.
* **2021→now Sharpe** is lower (0.80 vs 0.92) at an identical +101% return.
* The SMA∨MACD hybrid gives more return (+416% full, +45% recent) but a −31%
  drawdown and 36 trades; rejected as it re-imports MACD's whipsaw and the DD.
* A re-tuned MACD 12/15/14 also fixes 2025→ (+28%, Sh 1.04) but needs ~117 trades
  and has only 0.59 full-history Sharpe.
* The recent window is one trend leg (two trades); treat the 2025→ uplift as
  supporting evidence, the plateau/walk-forward/full-history as the main case.

**Code.** New `ma_band` mode (`TickerConfig.band_enter/band_exit`,
`backtest_ticker._trend_long_base`); GRID config now `ma_window=200`, ±2%,
`fixed_stop=1.0`, `v2_trail_stop=0.12`. Under V1 the sleeve has no trailing stop.
