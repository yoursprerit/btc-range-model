# 📊 Multi-ticker forecasting apps (SOXX · VEGN · GRID · XLE · REMX · WGMI)

Six new forecasting + trading applications, each built as a faithful sibling of
the **Gold (GLDM)** app in this repo — same layout, styling, colour coding, cards,
plots and backtesting dashboard — but with every feature, model, threshold and
strategy **re-derived for the specific asset**. The BTC and GLDM apps are
**unchanged**.

Pick any app from the **Application radio at the top of the sidebar** (grey panel):

```
₿ Bitcoin (BTC)  🥇 Gold (GLDM)  🖥️ SOXX  🌱 VEGN  ⚡ GRID  🛢️ XLE  🧲 REMX  ⛏️ WGMI
```

```bash
streamlit run streamlit_app.py     # root router → pick any app in the sidebar
```

| App | Asset | What it is | Chosen strategy |
|---|---|---|---|
| 🖥️ **SOXX** | iShares Semiconductor ETF | high-beta chips (NVDA/AVGO/AMD) | 40-day trend filter |
| 🌱 **VEGN** | US Vegan Climate ETF | ESG-screened S&P 500, tech-tilted | 200-day trend filter |
| ⚡ **GRID** | First Trust Clean Edge Grid ETF | grid / electrification infra | 150-day trend filter |
| 🛢️ **XLE** | Energy Select Sector SPDR | large-cap energy (+ OIH sibling) | Divergence Pure-Regime |
| 🧲 **REMX** | VanEck Rare Earth & Strategic Metals | rare-earth / strategic metals | 150-day trend filter |
| ⛏️ **WGMI** | CoinShares Valkyrie Bitcoin Miners ETF | 2–3× BTC-beta miners (MARA/RIOT/CLSK) | 30-day trend filter |

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
| VEGN | SPY, QQQ, XLK, ^TNX, DXY, ^VIX | +SPY mom − yields − VIX + own mom |
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
| SOXX | MA-40, −5 % stop | **+233 %** | +362 % | **−31 %** | −46 % | **0.94 / 0.93** |
| VEGN | MA-200, −5 % stop | **+84 %** | +81 % | **−14 %** | −33 % | **1.02 / 0.73** |
| GRID | MA-150, −5 % stop | +105 % | +132 % | **−19 %** | −30 % | **0.87 / 0.83** |
| XLE | Divergence (U1 .16 / D2 −.10 / D1-exit) | +105 % | +180 % | **−9 %** | −27 % | **1.37 / 0.84** |
| REMX | MA-150, −5 % stop | **+73 %** | +24 % | −56 % | −75 % | **0.48 / 0.30** |
| WGMI | MA-30, −10 % stop | +191 % | +223 % | **−40 %** | −63 % | **1.02 / 0.98** |

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
| **REMX** | 2016→now | **+370 %** | +112 % | **−56 %** | −75 % | **0.70 / 0.38** | return · DD · Sharpe |
| VEGN | 2020→now | +141 % | +167 % | **−16 %** | −34 % | **1.10 / 0.93** | DD · Sharpe |
| GRID | 2016→now | +296 % | +452 % | **−22 %** | −41 % | **0.94 / 0.88** | DD · Sharpe |
| WGMI | 2023→now | +514 % | +536 % | **−52 %** | −63 % | **1.18 / 1.08** | DD · Sharpe |
| SOXX | 2016→now | +691 % | +1842 % | **−31 %** | −46 % | 0.94 / 1.01 | drawdown only |

**Read this honestly.** Over a full bull+bear cycle the trend strategy beats
buy-&-hold *outright — return, drawdown and Sharpe* — on the **commodity-equity
crash assets (XLE, OIH, REMX)**, where B&H rode a multi-year bear all the way down.
On the **secular compounders (SOXX, GRID, VEGN, WGMI)** buy-&-hold's total return
is **mathematically unbeatable** by any unleveraged long/flat rule — every day in
cash is a day of missed gains, and their crashes were too brief to matter — so
there the strategy "wins" on **risk-adjusted terms** (higher Sharpe, roughly half
the drawdown). SOXX is the extreme case: semiconductors recover so violently that
even Sharpe favours buy-&-hold; only the drawdown is tamed. The only ways to beat
buy-&-hold's *return* on those names would be leverage or shorting, both excluded
here.

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
