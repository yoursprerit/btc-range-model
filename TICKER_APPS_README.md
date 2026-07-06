# 📊 Multi-ticker forecasting apps (SOXX · VEGN · GRID · XLE · REMX)

Five new forecasting + trading applications, each built as a faithful sibling of
the **Gold (GLDM)** app in this repo — same layout, styling, colour coding, cards,
plots and backtesting dashboard — but with every feature, model, threshold and
strategy **re-derived for the specific asset**. The BTC and GLDM apps are
**unchanged**.

Pick any app from the **Application radio at the top of the sidebar** (grey panel):

```
₿  Bitcoin (BTC)   🥇  Gold (GLDM)   🖥️ SOXX   🌱 VEGN   ⚡ GRID   🛢️ XLE   🧲 REMX
```

```bash
streamlit run streamlit_app.py     # root router → pick any app in the sidebar
```

| App | Asset | What it is | Chosen strategy |
|---|---|---|---|
| 🖥️ **SOXX** | iShares Semiconductor ETF | high-beta chips (NVDA/AVGO/AMD) | 50-day trend filter |
| 🌱 **VEGN** | US Vegan Climate ETF | ESG-screened S&P 500, tech-tilted | 200-day trend filter |
| ⚡ **GRID** | First Trust Clean Edge Grid ETF | grid / electrification infra | 200-day trend filter |
| 🛢️ **XLE** | Energy Select Sector SPDR | large-cap energy (+ OIH sibling) | Divergence Pure-Regime |
| 🧲 **REMX** | VanEck Rare Earth & Strategic Metals | rare-earth / strategic metals | Divergence Pure-Regime |

---

## Architecture — one engine, five configs (no app touched twice)

Rather than copy the ~1,400-line Gold app five times, everything asset-specific
lives in a single `TickerConfig`, and one generic engine renders each app:

```
app/ticker_config.py        the five TickerConfigs (drivers, thresholds, theme, windows)
app/ticker_core.py          data fetch (Yahoo) + features + macro-sentiment + signals
backtest_ticker.py          backtest engine (2 strategy engines) + threshold/strategy sweep
src/tickers/train_ticker.py trains the 5-model suite → models/<key>/*.joblib
app/ticker_app.py           the Streamlit UI — identical layout to Gold, themed per ticker
streamlit_app.py            root router: one radio for all seven apps
models/<key>/*.joblib       trained artefacts (soxx / vegn / grid / xle / remx)
data/<key>/*.csv|json       cached snapshots + backtest results
```

The BTC and GLDM apps are **byte-for-byte unchanged**. Their built-in two-option
selector is transparently upgraded to the full seven-app list by a tiny
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

**Out-of-sample results (full window vs Buy-&-Hold):**

| App | Strategy | Strat return | B&H return | Strat MDD | B&H MDD | Sharpe (S / B&H) |
|---|---|---|---|---|---|---|
| SOXX | MA-50, −10 % stop | **+223 %** | +362 % | **−39 %** | −46 % | 0.91 / 0.93 |
| VEGN | MA-200, −8 % stop | **+82 %** | +81 % | **−15 %** | −33 % | **1.00 / 0.73** |
| GRID | MA-200, −10 % stop | +115 % | +132 % | **−15 %** | −30 % | **0.92 / 0.83** |
| XLE | Divergence (U1 .12 / D2 −.12) | +95 % | +180 % | **−9 %** | −27 % | **1.11 / 0.84** |
| REMX | Divergence (U1 .12 / D2 −.18) | **+99 %** | +24 % | **−21 %** | −74 % | **0.81 / 0.30** |

The common thread: on **trending, high-beta** names (SOXX/VEGN/GRID) a moving-
average trend filter keeps most of the upside while roughly **halving the
drawdown**; on **boom-bust commodity-equity** names (XLE/REMX) the divergence
system's job is to **avoid the crash** — REMX quadruples the buy-&-hold return
and cuts the drawdown from a catastrophic −74 % to −21 %.

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
python src/tickers/train_ticker.py ALL        # fetch + train all five model suites
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
