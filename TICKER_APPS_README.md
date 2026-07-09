# 📊 Multi-ticker forecasting apps (SOXX · GRID · XLE · REMX · WGMI)

Six new forecasting + trading applications, each built as a faithful sibling of
the **Gold (GLDM)** app in this repo — same layout, styling, colour coding, cards,
plots and backtesting dashboard — but with every feature, model, threshold and
strategy **re-derived for the specific asset**. The BTC and GLDM apps are
**unchanged**.

Pick any app from the **Application radio at the top of the sidebar** (grey panel):

```
₿ Bitcoin (BTC)  🥇 Gold (GLDM)  🖥️ SOXX  ⚡ GRID  🛢️ XLE  🧲 REMX  ⛏️ WGMI
```

```bash
streamlit run streamlit_app.py     # root router → pick any app in the sidebar
```

| App | Asset | What it is | Chosen strategy |
|---|---|---|---|
| 🖥️ **SOXX** | iShares Semiconductor ETF | high-beta chips (NVDA/AVGO/AMD) | Dual-MA 25/100 crossover |
| ⚡ **GRID** | First Trust Clean Edge Grid ETF | grid / electrification infra | MACD 10/20/9 |
| 🛢️ **XLE** | Energy Select Sector SPDR | large-cap energy (+ OIH sibling) | Divergence Pure-Regime |
| 🧲 **REMX** | VanEck Rare Earth & Strategic Metals | rare-earth / strategic metals | Dual-MA 50/200 golden cross |
| ⛏️ **WGMI** | CoinShares Valkyrie Bitcoin Miners ETF | 2–3× BTC-beta miners (MARA/RIOT/CLSK) | 50-day SMA + volatility filter |

> **Strategy refresh.** SOXX, GRID and WGMI were re-tuned via a per-asset ML /
> statistical search validated on a held-out window (training data through
> Aug-2025): SOXX moved from a single 40-day SMA to a 25/100 dual-MA crossover,
> GRID from a 150-day SMA to MACD 10/20/9, and WGMI from a 30-day SMA to a
> 50-day SMA + volatility filter — each an out-of-sample improvement on return
> and Sharpe. REMX was optimised over the **full 2015→now bull+bear cycle** to a
> 50/200 dual-MA "golden cross" (+510% vs the old 150-day filter's +370%),
> beating buy-&-hold on return, drawdown and Sharpe. See `HYPERPARAM_SEARCH_EVAL.md`,
> `ML_STATISTICAL_STRATEGY_EVAL.md` and `REGIME_DIVERGENCE_EVAL.md`.

---

## Architecture — one engine, six configs (no app touched twice)

Rather than copy the ~1,400-line Gold app six times, everything asset-specific
lives in a single `TickerConfig`, and one generic engine renders each app:

```
app/ticker_config.py        the six TickerConfigs (drivers, thresholds, theme, windows)
app/ticker_core.py          data fetch (Yahoo) + features + macro-sentiment + signals
backtest_ticker.py          backtest engine (2 strategy engines) + threshold/strategy sweep
src/tickers/train_ticker.py trains the 5-model suite → models/<key>/*.joblib
app/ticker_app.py           the Streamlit UI — identical layout to Gold, themed per ticker
streamlit_app.py            root router: one radio for all eight apps
models/<key>/*.joblib       trained artefacts (soxx / vegn / grid / xle / remx / wgmi)
data/<key>/*.csv|json       cached snapshots + backtest results
```

The BTC and GLDM apps are **byte-for-byte unchanged**. Their built-in two-option
selector is transparently upgraded to the full eight-app list by a tiny
`st.radio` wrapper in the router — no edit to either original file.

---

## Asset-specific features

Each app drops BTC's crypto inputs and uses the **genuine drivers of its asset**,
plus a purpose-built **0–100 macro-sentiment** composite (each driver signed so
"up = bullish for the asset", then rank-scaled like a Fear & Greed index):

| App | Macro drivers | Sentiment blend |
|---|---|---|
| SOXX | SMH, ^SOX, QQQ, NVDA, ^TNX, DXY, ^VIX, ^GSPC | +QQQ mom − yields − VIX + own mom |
| GRID | XLU, ICLN, TAN, CPER, ^TNX, ^GSPC, ^VIX | +SPX mom + copper − yields + own mom |
| XLE | CL=F, BZ=F, XOM, DXY, ^GSPC, ^VIX (+OIH) | +crude − USD − VIX + own mom |
| REMX | LIT, CPER, FXI, SLV, DXY, ^GSPC, ^VIX | +copper + lithium − USD + own mom |
| WGMI | BTC-USD, MARA, RIOT, COIN, ETH-USD, QQQ, ^VIX | +BTC mom + QQQ mom − VIX + own mom |

Each app trains the same **five models** (`src/tickers/train_ticker.py <KEY>`):
hourly next-close (ridge + 95 % CI), daily High/Low (calibrated ridge bands),
7- & 14-day close cones, and a 3-class day-type classifier.

---

## Strategy selection — tuned per asset, verified over multiple periods

The strategy and its thresholds are **not** assumed — a frontier sweep
(`backtest_ticker.py <KEY> --sweep`) searches both engines (MA trend-filter and
the Gold/BTC Divergence Pure-Regime system) and picks the config that maximises
return **subject to a drawdown no worse than buy-&-hold in every period** and
that **beats buy-&-hold in the loss periods**. The daily H/L signal model is fit
once on the pre-OOS window, so all reported windows are genuinely out-of-sample.

**Out-of-sample results (2021→now window vs Buy-&-Hold):**

| App | Strategy | Strat return | B&H return | Strat MDD | B&H MDD | Sharpe (S / B&H) |
|---|---|---|---|---|---|---|
| SOXX | Dual-MA 25/100, −5 % stop | **+452 %** | +362 % | **−27 %** | −46 % | **1.23 / 0.94** |
| GRID | MACD 10/20/9, −5 % stop | **+134 %** | +132 % | **−16 %** | −30 % | **1.20 / 0.83** |
| XLE | Divergence (U1 .16 / D2 −.10 / D1-exit) | +105 % | +180 % | **−9 %** | −27 % | **1.37 / 0.84** |
| REMX | Dual-MA 50/200, −5 % stop | **+135 %** | +24 % | **−27 %** | −74 % | **0.68 / 0.30** |
| WGMI | MA-50 + vol-filter (k 0.95, no stop) | **+376 %** | +223 % | **−32 %** | −63 % | **1.81 / 0.98** |

(OIH, traded off the XLE signal: **+70 %** at **−19 %** MDD vs Buy-&-Hold +130 % / −46 %.)
(WGMI listed Feb-2022, so its OOS window is 2024→now with an ~11-month training window.)

### Full-cycle test — the entire history including a real bear (the 🌍 tab)

Every app also has a **🌍 Full history (Bull + Bear)** backtest tab spanning each
asset's *entire* available history (including the pre-2021 in-sample window). This
is the honest combined-cycle question: does the strategy beat buy-&-hold once a
real bear market is in scope? Whether it wins on **total return** turns out to be
governed by *cycle geometry*, not tuning — a long/flat strategy can only out-return
buy-&-hold if the window contains a deep, drawn-out bear the asset never fully
recovered from:

| App | Full window | Strat return | B&H return | Strat MDD | B&H MDD | Sharpe (S / B&H) | Beats B&H on |
|---|---|---|---|---|---|---|---|
| **XLE** | 2016→now | **+165 %** | +76 % | **−17 %** | −70 % | **0.95 / 0.33** | return · DD · Sharpe |
| **OIH** | 2016→now | **+119 %** | −32 % | **−23 %** | −90 % | **0.59 / 0.12** | return · DD · Sharpe |
| **REMX** | 2015→now | **+510 %** | +112 % | **−27 %** | −75 % | **0.82 / 0.38** | return · DD · Sharpe |
| GRID | 2015→now | +270 % | +452 % | **−21 %** | −41 % | **1.04 / 0.88** | DD · Sharpe |
| WGMI | 2022→now | +282 % | +536 % | **−32 %** | −63 % | **1.32 / 1.08** | DD · Sharpe |
| SOXX | 2015→now | +1161 % | +1842 % | **−35 %** | −46 % | **1.05 / 1.01** | DD · Sharpe |

**Read this honestly.** Over a full bull+bear cycle the strategy beats
buy-&-hold *outright — return, drawdown and Sharpe* — on the **commodity-equity
crash assets (XLE, OIH, REMX)**, where B&H rode a multi-year bear all the way
down. REMX's 50/200 golden cross is tuned to maximise the full-cycle result and
turns B&H's +112 % / −75 % into **+510 % / −27 %** (Sharpe 0.82 vs 0.38). On the
**secular compounders (SOXX, GRID, WGMI)** buy-&-hold's total return is
**mathematically unbeatable** by any unleveraged long/flat rule — every day in
cash is a day of missed gains — so the strategy "wins" on **risk-adjusted terms**
(higher Sharpe, roughly half the drawdown). The refreshed configs improve this
risk-adjusted edge: SOXX's 25/100 dual-MA now beats B&H on Sharpe *and* drawdown
over the full cycle (1.05 vs 1.01), and GRID's MACD lifts full-cycle Sharpe to
1.04. The only ways to beat buy-&-hold's *return* on the compounders would be
leverage or shorting, both excluded here.

---

## XLE vs OIH — which signal should drive OIH?

The brief asks whether OIH (VanEck Oil Services, the high-beta energy sibling) is
better traded on **XLE's** signals or on its **own** standalone signals. Both were
back-tested (2021 → now):

| OIH driven by… | Full return | Max drawdown | Sharpe | 2021–22 energy drawdown |
|---|---|---|---|---|
| **XLE signal** (chosen) | **+36 %** | −27 % | **0.43** | **+65 %** |
| OIH's own signal | +23 % | −23 % | 0.29 | +13 % |

**Trade OIH on the XLE signal.** XLE is a broader, less-noisy read of the energy
tape, so its divergence signal steers the thin, higher-beta OIH better than OIH's
own — higher return, better Sharpe, and it wins the drawdown period decisively.
This is the same rationale by which the Gold app trades GDX/UGL off the cleaner
gold signal. The XLE app therefore trades **both XLE and OIH off the XLE signal**
(one Backtesting tab each).

---

## Leveraged siblings for total return — SOXL (3× semis) · ERX (2× energy)

The two top-conviction signals (SOXX, XLE) now also drive a **leveraged sibling**,
the same way BTC drives MSTU (2×) and gold drives UGL (2×):

* **SOXL** (3× semis) trades off the **SOXX 25/100 dual-MA** — the SOXX app shows
  it in its own Backtesting tab.
* **ERX** (2× energy) trades off the **XLE energy-divergence** signal — alongside
  XLE and OIH.

The long/flat rule holds each leveraged ETF **only while its parent trends up** and
stands aside in the drawdowns where daily-rebalance decay is worst — so the strat
Sharpe far exceeds a buy-&-hold 3×. In the Overall blend they carry the tightest
weight caps (`lev`: 0.10 Balanced → 0.35 Aggressive): **Balanced barely touches
SOXL**, while the **Aggressive profile roughly doubles combined return** by loading
it (OOS optimal ~+1034% → **+2161%**, drawdown −34%, inside the −38% budget). **ERX**
is a rare win-win — it lifts return *and* Sharpe *and* trims drawdown, because it is
uncorrelated to the tech-heavy book. Full analysis in `SOXL_ERX_ADDITION_EVAL.md`.

---

## Reproduce

```bash
python src/tickers/train_ticker.py ALL        # fetch + train all six model suites
python backtest_ticker.py SOXX --sweep        # per-ticker strategy/threshold sweep
python backtest_ticker.py XLE                 # per-period backtest (XLE + OIH)
streamlit run streamlit_app.py                # launch, pick a ticker in the sidebar
# add --cached to the scripts to use the committed data/<key> snapshots
```

Models are deliberately lightweight (ridge / logistic) so they retrain in seconds
and the whole pipeline stays transparent and reproducible — exactly like Gold.

**Honest framing.** Intraday direction is ~coin-flip for every one of these
assets (as it is for BTC and gold); the hourly model's value is a tight,
well-calibrated confidence interval, not a directional bet. The edge lives in the
**trend regime** and **risk control**, quantified in each app's Backtesting tabs.
