# ₿ Bitcoin (BTC) — four-model forecaster & divergence strategy

The original app in this repo: a **four-model Bitcoin forecasting stack** served
through the Streamlit dashboard, plus the **BTC Divergence Pure-Regime** trading
strategy that trades BTC and its siblings **MSTR**, **MSTU** and **ETH** off
the Bitcoin signal (spot ETH is executed live through the **ETHA** ETF, just as
BTC is executed through IBIT).

Select it from the **Application radio at the top of the sidebar** →
**₿ Bitcoin (BTC)**. For the platform overview see **[`README.md`](README.md)**;
for the full strategy spec see **[`TRADING_STRATEGY.md`](TRADING_STRATEGY.md)**
and **[`BTC_MSTR_MSTU_STRATEGY_EVAL.md`](BTC_MSTR_MSTU_STRATEGY_EVAL.md)**; for a
worked example of reading a live divergence signal — the 2026-08 thrust, its 26
historical analogs, and why the live gate stayed flat — see
**[`BTC_BULLRUN_ANALOG_EVAL.md`](BTC_BULLRUN_ANALOG_EVAL.md)**; the
ETH sleeve — and an important **look-ahead finding affecting MSTR/MSTU (fixed
2026-07-25: equity fills now land at the first exchange close after the
signal moment)** — are in
**[`ETH_BMNR_STRATEGY_EVAL.md`](ETH_BMNR_STRATEGY_EVAL.md)**.

Each traded instrument has its own **Backtesting tab** — ₿ BTC · 📊 MSTR ·
📈 MSTU · 🔹 ETH — plus the two options tabs, all driven by the same BTC
signal.

## The four forecasting models

| Model | What it predicts | Cadence | Horizon |
|---|---|---|---|
| **Daily H/L (7am-CT)** | Next 24-hour bar's `high` and `low` | Once per day at 12:00 UTC (= 7am CDT / 6am CST) | 1 bar (24 h) ahead |
| **Hourly close** | Next-hour BTC closing price | Every 60 s in the live tab | 1 hour ahead |
| **7-day close cone** | Median + ±9.7 % band for the close 7 days out | Once per day | 7 days ahead |
| **3-class day-type** | Next-day bar shape: `BigUpper` / `BigLower` / `Quiet` | Once per day | 1 bar ahead |

All four are linear-or-shallow learners trained on a mix of BTC price/volume
history, cross-market macro indicators, on-chain blockchain metrics, and a
sentiment index. The Streamlit UI surfaces their predictions side-by-side with
realised price action so accuracy is observable in real time.

> **Anti-leakage methodology** (May 2026 retrain). Every model uses a strict
> chronological **train / val / test** split with a per-horizon embargo gap; α,
> β, model winner, and σ are all selected on **val** with **test untouched**
> until final reporting. The Fear & Greed feed — which alternative.me updates
> intraday on the current day's record — is lagged 1 day for the hourly model.
> The 3-class day-type model uses **TimeSeriesSplit-5 out-of-fold H/L
> predictions** for rows ≤ the H/L training cutoff to remove stacked in-sample
> contamination. See [§Leakage audit & fixes](#leakage-audit--fixes).

---

## Model 1 — Daily H/L (7am-CT day boundary)

### Day-boundary contract

A "day" in this model is a **24-hour bar starting at 12:00 UTC** =
- **7:00 AM CDT** (US Central Daylight Time, March → November), or
- **6:00 AM CST** (US Central Standard Time, November → March).

Fixed UTC anchor (not DST-following) so every bar is exactly 24 hours — no
23/25 h edge cases. The model predicts the **next** such bar's high and low.

### Pipeline (`src/pipeline_ct.py`)

```
Binance hourly OHLCV  ─┐
                       ├─► rebucket into 12:00→12:00 UTC daily bars
Yahoo daily macro     ─┤   (open=first, high=max, low=min, close=last, vol=sum)
(SPX, NDX, VIX, GOLD, │
DXY, TNX, ETH)        │
                       │
blockchain.info       ─┤── join on calendar date D ─► engineered features
on-chain (11 series)  │
                       │
Fear & Greed Index    ─┘

Target (per bar D):
  y_hi = (next_high - close) / close   ≥ 0
  y_lo = (close - next_low ) / close   ≥ 0
  Reconstruct: pred_high = close·(1 + y_hi),  pred_low = close·(1 - y_lo)

Model:
  Ensemble of 3 regressors (all wrapped in StandardScaler):
    1. HuberRegressor                (linear, robust)
    2. QuantileRegressor   (q=0.70, linear quantile)
    3. GradientBoostingRegressor     (loss="quantile", q=0.70,
                                      1500 trees, depth 3, lr 0.01)
  Pure mean of the three predictions, then a shrinkage blend with the
  training-set climatological mean offset (μ_hi, μ_lo) using α.

Direction head:
  GradientBoostingClassifier on sign(y_hi − y_lo). The post-α ensemble is
  reparameterised into (half-range m, asymmetry d); d is blended with a
  classifier-driven d using an adaptive β_eff = β_base × (1 − r × trend_str),
  where trend_str = min(|ret_5|/0.05, 1).

Tuning protocol (anti-leak):
  TRAIN  → fit ensemble + direction classifier
  VAL    → pick α and (β_base, r) by validation MAPE
  TEST   → final, untouched evaluation (never seen during tuning)
  Embargo of 1 day between train↔val and val↔test (= forecast horizon)

Current split (artefact dated 2026-05-21):
  TRAIN  2019-04-01 → 2025-09-17   (n=2342)
  VAL    2025-09-18 → 2025-11-17   (n=61)
  TEST   2025-11-18 → 2026-05-18   (n=182)
  Selected: α = 1.00, β_base = 1.00, r = 0.30.
```

### Features — full inventory (103 total)

Grouped by family. Suffixes: `_n` = lag in days, `_dN` = log-difference over N days, `_zN` = N-day rolling z-score. Notation `c` = BTC close, `h` = high, `l` = low, `v` = volume.

#### BTC price action (38 features)

| # | Feature | Definition | Why it matters |
|---|---|---|---|
| 1–6 | `ret_1, ret_3, ret_5, ret_7, ret_14, ret_30` | Sum of daily log-returns over the last 1/3/5/7/14/30 bars | Momentum signal across multiple horizons; range autocorrelates with recent direction |
| 7–10 | `vol_5, vol_10, vol_20, vol_30` | Rolling std-dev of log-returns over 5/10/20/30 bars | Realized volatility regime. Range scales with vol. |
| 11–13 | `atr_7, atr_14, atr_30` | ATR (true range)/`c` averaged over 7/14/30 days | Volatility in price-units, normalized by current price. Range = ATR scaled. |
| 14 | `range_today` | `(h - l) / c` | Current bar's range — the single strongest predictor of tomorrow's range |
| 15 | `range_ma7` | 7-day average of `range_today` | Range mean-reversion baseline |
| 16 | `range_ma30` | 30-day average of `range_today` | Long-run baseline; comparison to `range_ma7` reveals regime shift |
| 17 | `range_std30` | 30-day std-dev of `range_today` | Is volatility of volatility rising? Spike before regime breaks. |
| 18 | `rsi_14` | RSI on 14 closes | Overbought/oversold; affects asymmetry of tomorrow's range |
| 19 | `macd` | (EMA12 − EMA26) / `c` | Trend strength |
| 20 | `macd_sig` | EMA9 of MACD, /`c` | Trend-direction signal line |
| 21 | `macd_hist` | MACD − MACD signal | Trend-acceleration |
| 22 | `bb_width` | 4·σ20 / MA20 (Bollinger band width) | Compression → expansion volatility cycle |
| 23 | `dist_hi_30` | `c / max(c, 30) − 1` | How far below the 30-day high (≤ 0). Sets ceiling pressure. |
| 24 | `dist_lo_30` | `c / min(c, 30) − 1` | How far above the 30-day low (≥ 0). Sets floor pressure. |
| 25 | `dist_hi_90` | `c / max(c, 90) − 1` | 90-day-extreme proximity; longer-horizon trend context |
| 26 | `vol_chg_1` | `log(v) − log(v.shift(1))` | 1-day volume change |
| 27 | `vol_z_20` | z-score of `log(v)` over 20 days | Standardised volume surprise |
| 28 | `vol_ma_ratio` | `v / MA20(v)` | Volume vs recent average |
| 29–34 | `dow_0…dow_5` | Day-of-week one-hots (Mon=0…Sat=5) | BTC has real weekend behavior; Sun is the reference category |

#### Cross-asset macro (28 features)

Each of 7 macro symbols contributes 4 features:

| Symbol | Source | Features |
|---|---|---|
| `spx` (S&P 500) | Yahoo `^GSPC` | `spx_ret_1`, `spx_ret_5`, `spx_ret_20`, `spx_vol_20` |
| `ndx` (Nasdaq Composite) | Yahoo `^IXIC` | `ndx_ret_1`, `ndx_ret_5`, `ndx_ret_20`, `ndx_vol_20` |
| `vix` (CBOE Volatility) | Yahoo `^VIX` | `vix_ret_1`, `vix_ret_5`, `vix_ret_20`, `vix_vol_20` |
| `gold` (Gold futures) | Yahoo `GC=F` | `gold_ret_1`, `gold_ret_5`, `gold_ret_20`, `gold_vol_20` |
| `dxy` (US Dollar Index) | Yahoo `DX-Y.NYB` | `dxy_ret_1`, `dxy_ret_5`, `dxy_ret_20`, `dxy_vol_20` |
| `tnx` (10-yr Treasury yield) | Yahoo `^TNX` | `tnx_ret_1`, `tnx_ret_5`, `tnx_ret_20`, `tnx_vol_20` |
| `eth` (Ether) | Yahoo `ETH-USD` | `eth_ret_1`, `eth_ret_5`, `eth_ret_20`, `eth_vol_20` |

Each `*_ret_k` is `log(close[t]) − log(close[t-k])` and each `*_vol_20` is the std-dev of the corresponding 1-day return series over 20 days. **Rationale:** crypto correlates positively with risk-on (SPX/NDX), negatively with USD strength (DXY) and yields (TNX), and shows regime-dependent links to VIX and Gold.

Plus 4 cross-asset rolling correlations: `btc_spx_corr_30`, `btc_ndx_corr_30`, `btc_gold_corr_30`, `btc_dxy_corr_30` (30-day Pearson corr of daily log-returns).

#### On-chain (33 features)

Pulled from `blockchain.info`'s public charts API. 11 series × 3 transforms each:

| Series | Plain meaning |
|---|---|
| `oc_hash_rate` | Network total hash rate (security / miner commitment) |
| `oc_difficulty` | Mining difficulty (lags hash rate by epoch length) |
| `oc_n_transactions` | Daily tx count |
| `oc_miners_revenue` | Total daily revenue to miners (block subsidy + fees) |
| `oc_n_unique_addresses` | Daily active addresses — adoption proxy |
| `oc_transaction_fees_usd` | Aggregate fees paid (mempool pressure) |
| `oc_mempool_size` | Pending tx queue size |
| `oc_estimated_transaction_volume_usd` | On-chain dollar throughput |
| `oc_market_cap` | Daily market cap |
| `oc_avg_block_size` | Mean block size |
| `oc_cost_per_transaction` | Network cost / tx — efficiency measure |

For each series `S` the three features are `_d1` (`log(S[t]) − log(S[t-1])`), `_d7` (week-over-week), and `_z30` (30-day z-score of `log(S)`).

#### Self-lag & regime features (14 features)

| Family | Feature | Definition |
|---|---|---|
| Smoothed lag | `y_hi_ema3`, `y_hi_ema7`, `y_lo_ema3`, `y_lo_ema7` | EMA (span 3, 7) of `y_hi.shift(1)` / `y_lo.shift(1)`. Smoother than a raw lag → less mean-reversion overshoot. |
| Breakout signals | `above_3d_high`, `below_3d_low`, `bo_strength_up`, `bo_strength_dn` | Binary + magnitude versions of `close vs max/min(high, 3) / min/max(low, 3)`. |
| "Surprise" | `y_hi_surprise`, `y_lo_surprise` | `y_hi.shift(1) − ema7(y_hi.shift(1))` — how much yesterday's excursion differed from the recent baseline. |
| Downside | `dn_vol_5`, `dn_vol_20`, `below_sma50`, `below_sma50_5d` | Downside-only realised vol (5/20-day) + flags for "close below SMA50" (today, and ≥5 of last 5 days). |

Price-action and self-lag families dominate permutation importance (`range_today`, `atr_7`, `dist_hi_30`, day-of-week effects, and `oc_transaction_fees_usd_z30` as the strongest on-chain contributor) — range is mostly autoregressive; macro and on-chain provide secondary refinement.

### Held-out test metrics  (honest — α/β tuned on **val**, test untouched)

Test window: **2025-11-18 → 2026-05-18** (182 days). Validation: 2025-09-18 → 2025-11-17 (61 days). Training: 2,342 days from 2019-04-01.

| Metric | HIGH | LOW |
|---|---:|---:|
| **MAPE** | **1.32 %** | **1.30 %** |
| MAE (USD) | $1,049 | $988 |
| RMSE (USD) | $1,291 | $1,475 |
| Hit ±1 % | 45.1 % | 52.2 % |
| Hit ±2 % | 79.7 % | 87.9 % |
| σ (residual, return space) | 0.0161 | 0.0186 |
| 95 % CI half-width | ±3.16 % | ±3.64 % |
| Direction hit-rate (sign of y_hi − y_lo) | | **50.0 %** |

> **Honest framing.** With α and β picked on a held-out val slice (not test),
> MAPE_H lands at **~1.32 %** and the direction-head's apparent edge **collapses
> to coin-flip on test (50.0 %)**. The model's real value is **price-magnitude
> bracketing**, not direction calling — use it as a volatility cone.

### Inference contract (used by `app/`)

`compute_daily_forecast(asof_date_iso)` in `app/btc_hourly_app.py` re-builds the
same features from the latest 12:00-UTC bars, truncates to bars ≤ `asof_date`,
runs the ensemble, and returns `(pred_high, pred_low, CI bands, target window)`.
Cached 24 h per as-of-date key (rolls over at each 12:00 UTC).

---

## Model 2 — Hourly close

Predicts the next-hour log-return of `BTC-USD` (Yahoo) close; at inference the
predicted log-return is multiplied against the **live Binance spot** (refreshed
every 30 s) to produce the "1-hour-from-now" price point.

### Pipeline (`src/train_hourly_model.py`)

```
Yahoo BTC-USD 1h bars  ─┐
+ ETH-USD, SPX, NDX,   │
  VIX, GOLD, DXY, TNX  ─┤ all resampled to a common hourly grid
+ alternative.me F&G   ─┘ shifted 1 day, then forward-filled

Target:  y = log(close[t+1] / close[t])     (one-hour log-return)

Models compared: RidgeCV vs GradientBoostingRegressor (winner picked on VAL
direction-accuracy; σ fit on VAL residuals). Embargo of 1 hour between splits.

Current split (artefact dated 2026-05-21): TRAIN n=6487, VAL n=139, TEST n=339.
Winner: gbm.

F&G causality fix: alternative.me re-computes the current-day record throughout
the day, so the series is lagged 1 day before joining (identical in the live app).
```

### Features (59 total)

29 BTC price-action (multi-horizon returns/vol/ATR, range, RSI, MACD, Bollinger,
distance-from-extreme), 21 cross-asset macro (7 symbols × `ret_1h`/`ret_24h`/`vol_24h`)
plus `btc_eth_corr_24`, 4 Fear & Greed sentiment, and 5 time-of-day/calendar
(cyclical hour + day-of-week, weekend, `us_open`).

**Top contributors (permutation importance on TEST):** `tnx_ret_1h` (1-hour
change in the 10-yr Treasury yield) is the top driver, then `vix_ret_24h`,
`ret_4h`, `vix_ret_1h`, `btc_eth_corr_24`. Cross-asset features take 10 of the
top 15 slots — the hourly model is best read as a **macro-vol regime classifier**
routed through Bitcoin returns.

### Held-out metrics  (val picks the winner, test unbiased)

| Metric | VAL (selection) | TEST (unbiased) |
|---|---:|---:|
| MAPE on next-hour close | 0.40 % | **0.29 %** |
| MAE (USD) | $271 | **$220** |
| Hit ±1 % | 91.4 % | 96.2 % |
| **Direction accuracy** | 51.1 % | **50.15 %** |
| 95 % CI half-width (from VAL σ) | | **±1.09 %** |

> **Honest framing.** Picking the winner on VAL gives a TEST direction accuracy
> of **50.15 %** — a coin-flip. What the hourly model *can* honestly offer is a
> **tight 1-hour volatility cone** (±1.09 %) and a slight magnitude advantage
> (0.29 % MAPE vs ~0.39 % for the zero-return baseline). Use it as a vol-cone, not
> a directional signal.

---

## Model 3 — 7-day close cone

Predicts the expected close 7 days out with a regime-conditioned uncertainty
band. `src/train_7d_close_cone.py` bins training days into terciles of
`range_ma30` (low/mid/high realised-vol); the forecast is
`close(t) · exp(regime_median)` with a fixed ±9.7 % band. Embargo 7 days.

**Current artefact (2026-05-21):** Train 2019-04-01 → 2025-09-14 (n=2339); Test
2025-09-21 → 2026-05-12 (n=234). Held-out coverage of the ±9.7 % band: **88.0 %**.

`compute_7d_close_cone_forecast(asof_date_iso)` in `app/btc_hourly_app.py`
classifies the as-of bar's regime, applies the median move and band, and returns
the last 7 weekly closes vs their prior-week cone forecasts for the replay tab.

---

## Model 4 — 3-class day-type classifier

Categorical label for the next 12:00-UTC bar's shape:

| Class | Definition |
|---|---|
| `BigUpper` | (y_hi + y_lo) ≥ TRAIN tercile threshold **and** y_hi > y_lo |
| `BigLower` | (y_hi + y_lo) ≥ TRAIN tercile threshold **and** y_lo > y_hi |
| `Quiet`    | (y_hi + y_lo) < TRAIN tercile threshold |

`src/train_3class_day_type.py` uses the 103 daily features + 5 H/L-model-derived
features (`pred_y_hi`, `pred_y_lo`, `pred_range`, `pred_skew`, `p_bull`) + 3
cone-regime one-hots, fed to a GradientBoostingClassifier. **Anti-leak:**
TimeSeriesSplit-5 out-of-fold H/L predictions for rows ≤ H/L train_end (so the
3-class model never sees stacked in-sample H/L preds); 1-day embargo.

### Held-out test metrics (2026-02-19 → 2026-05-18, n=89)

| Metric | Value |
|---|---:|
| **Unconditional accuracy** | **52.8 %** (vs 33 % majority baseline) |
| Balanced accuracy | 49.6 % |

Selective accuracy by top-class probability:

| p ≥ | Coverage | Accuracy |
|---:|---:|---:|
| 0.50 | 57.3 % | 58.8 % |
| **0.55** | **43.8 %** | **69.2 %** |
| 0.60 | 23.6 % | 76.2 % |

> **Reading.** Not useful when forced to label every day (52.8 %), but it earns
> most of its accuracy where it's confident: at `p ≥ 0.55` it covers ~44 % of days
> at ~69 % accuracy. Trust the probability bar, not the pill colour.

---

## UI architecture (`app/btc_hourly_app.py`)

The page has **two tabs** sharing the same render code:

| Tab | "Now" anchor | Hourly chart window | Daily H/L target |
|---|---|---|---|
| **🔴 Live** | wall-clock `datetime.now(UTC)` | rolling last 24 h | tomorrow's bar (7am-CT-anchored) |
| **🕒 Historical replay** | user-picked CT timestamp | fixed 24-h CT day | bar starting on picked_date + 1 |

Historical-replay extras: a 7-day pill strip, ◀/▶ date arrows, a calendar
picker, a 25-tick datetime slider, and a 🔖 bookmarks panel backed by
`runtime/bookmarks.json`.

**Minimum replay date.** `fetch_data()` pulls hourly BTC + macro from yfinance,
which hard-caps 1-hour bars at ~730 days. `data/macro_hourly_cache.csv` is a
committed one-time snapshot that extends `min_date` backward (live data always
wins on overlap, so it never alters current predictions). Refresh it with
`python scripts/fetch_macro_hourly_cache.py`.

### How the UI talks to the models

All model loading and feature engineering happens **inside the Streamlit
process** — the "API" is implicit: functions (`load_assets`,
`compute_daily_forecast`, `compute_daily_series`) consume joblib artefacts and
return DataFrames/dicts the chart code renders. No FastAPI/gRPC layer, because
the app is a single-user analytics dashboard, not a high-QPS service.

### Where each price on the chart comes from

| Surface | Source feed | Pair | Frequency |
|---|---|---|---|
| Live spot KPI + crimson dot | Binance public ticker | BTC/**USDT** | 30 s |
| Hourly "Actual close" line | Yahoo (yfinance) | BTC/**USD** | 60-min bars, 300 s cache |
| Daily H/L bars | Binance hourly → rebucketed | BTC/**USDT** | 24 h, 6 h cache |
| 1-hour rolling ⭐ forecast | Hourly model × Binance spot | — | every rerun |

The Yahoo BTC-USD and Binance BTC/USDT feeds differ by a few basis points (USDT
premium + latency), so the live red dot occasionally sits slightly off the Yahoo
hourly close line — that's the feed gap, not a bug.

---

## Limitations & known caveats

1. **Direction edge is coin-flip out-of-sample** for both the daily H/L and
   hourly models. Use them for magnitude/volatility bracketing, not direction.
2. **The 3-class model is most useful in selective mode** (`p_top ≥ 0.55` →
   ~69 % accuracy at ~44 % coverage).
3. **Yahoo hourly history is capped at ~2 years**, so the replay tab's earliest
   selectable date floats forward.
4. **Hourly macro features are daily-resolution** — constant within a UTC day.
5. **The 7am-CT boundary uses a fixed 12:00 UTC anchor**, so Nov–Mar the bar
   actually starts 6 AM CST. Deliberate trade for uniform 24-h bars.
6. **`live_spot` is BTC/USDT, not BTC/USD** (~5 bp basis).
7. **σ_lo > σ_hi on daily H/L** — BTC drawdowns are heavier-tailed than rallies
   in this window, so the LOW CI is slightly wider.
8. **In-sample replay banner.** Picking a historical date inside a model's
   training window shows a yellow warning — those predictions are memorisation,
   not honest forecasts.

---

## Leakage audit & fixes

Following a May 2026 audit, five classes of issue were hardened:

| Issue | Where | Fix |
|---|---|---|
| Hyperparameters (α, β, model winner, σ) tuned on TEST | `pipeline_ct.py`, `train_hourly_model.py` | New TRAIN/VAL/TEST split; tuning on VAL, TEST untouched. |
| No embargo at train/test boundary | all 4 training scripts | Embargo of `horizon` rows: 1 day (daily/3-class), 1 hour (hourly), 7 days (cone). |
| Stacked in-sample H/L preds used as 3-class features | `train_3class_day_type.py` | TimeSeriesSplit-5 OOF H/L predictions for in-train rows. |
| Replay tab silently shows in-sample fit | `btc_hourly_app.py` | Yellow warning banner naming each affected model and its `train_end`. |
| Fear & Greed index updated intraday | `train_hourly_model.py` + `app` | F&G lagged 1 day before joining. |

A **July 2026 follow-up audit (2026-07-25)** fixed three further classes:

| Issue | Where | Fix |
|---|---|---|
| Equity fills preceded the signal: MSTR/MSTU (stock + options) filled at the US close of the signal-bar *date*, ~15 h (weekends: up to 2.5 days) before the CT signal is knowable at 12:00 UTC the next day | `btc_hourly_app.py` (4 backtests), `btc_ct_engine.py` | Fills moved to the first exchange close at/after signal availability (`_fill_after_signal` / `_next_session_close`). MSTR +296%→+165%, MSTU +685%→+499% on the fix-date vintage; BTC/ETH unchanged. |
| Options backtests backfilled early bars with later realized volatility (`.bfill()`) | `btc_hourly_app.py` | Trailing-only HV with a fixed causal prior for pre-window bars. |
| "OOS Only — Fully Blind" labels: the window is blind only to model weights — strategy thresholds/stops/gates were tuned (2026-07) on a window that includes it | `btc_hourly_app.py` dashboards | Relabeled "Model-OOS ⚠️ Strategy Tuned In-Window". |

Effect on headline metrics: MAPE moved ≤ 0.2 pp (magnitude estimates were
honest); direction accuracy collapsed by ~3.8 pp (it was test-tuning inflation).

---

## Legacy / archived models

The `legacy/` directory holds the previous-generation UTC-midnight pipeline plus
two ancillary models (7-day window max/min; per-horizon k=1..7) that the current
dashboard does not use. See [`legacy/README.md`](legacy/README.md) for details
and why they were superseded. The CT boundary is preferred for semantics
(uniform 24-h bars), not predictive lift — honestly tuned, it performs roughly
the same as the legacy pipeline.
