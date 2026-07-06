# 🥇 GLDM — Gold ETF forecast & trend strategy

A full forecasting + trading application for **GLDM** (SPDR Gold MiniShares,
which tracks the spot gold price), built as the gold counterpart of the
Bitcoin app in this repo. It shares the repo's structure and philosophy but
every feature, model and threshold is **re-derived for gold** — nothing is
copied blindly from BTC.

Select the app from the **radio button at the top of the sidebar** (grey panel):
**₿ Bitcoin (BTC)** or **🥇 Gold (GLDM)**. The BTC application is unchanged.

```
streamlit run streamlit_app.py     # root router → pick BTC or GLDM in the sidebar
```

---

## Why gold is not Bitcoin (and what changed)

| | Bitcoin | GLDM (gold) |
|---|---|---|
| Realised vol (annualised) | ~55% | ~13% |
| Trading calendar | 24/7 | US market hours, no weekends |
| Primary macro drivers | risk sentiment, crypto flows | **US dollar, real yields** |
| Daily range | ~3–4% | ~0.6–1.0% |
| Typical drawdown | 50–75% | 15–30% |

Because gold's daily range is ~⅓ of BTC's and it trends smoothly, the divergence
signal thresholds are an **order of magnitude smaller**, the stops are **much
tighter**, and the recommended strategy is a **simple trend filter** rather than
BTC's aggressive divergence-exit complex.

---

## Gold-specific features

The crypto-only inputs (ETH, Coinbase premium, crypto Fear & Greed) are dropped
and replaced with gold's genuine drivers — see `app/gldm_core.py`:

- **US Dollar Index (DXY)** — the strongest *inverse* driver of gold
- **10-year Treasury yield (`^TNX`)** — real-yield proxy, *inverse* driver
- **Silver (SLV)** — precious-metals-complex co-movement + gold/silver ratio
- **Gold futures (`GC=F`)** — basis / lead relative to the ETF
- **VIX** — risk-off gold bid; **S&P 500** — stock-vs-gold rotation
- **Gold macro sentiment (0–100)** — a purpose-built composite replacing the
  crypto Fear & Greed index: dollar weakness − rising real yields + risk-off +
  gold's own momentum, rank-scaled to a self-calibrating 0–100 gauge
- The usual technical toolkit on GLDM itself (returns, vol, ATR, RSI, MACD,
  Bollinger width, distance from moving averages / extremes) plus **month-of-year
  seasonality**, which gold exhibits.

---

## The five models

All are trained by `python src/gldm/train_gldm.py` and written to
`models/gldm/*.joblib`. Metrics below are **out-of-sample** (final 20% of
history, untouched during fitting).

| # | Model | Type | Target | OOS result |
|---|---|---|---|---|
| 1 | Hourly next-close | Ridge on log-returns | next-hour close + 95% CI | MAPE 0.39%, dir-acc ~49%, CI ±0.88% |
| 2 | Daily High/Low | Ridge ratio-to-prior-close (calibrated) | next-day H & L | MAPE ~0.15% each |
| 3 | 7-day close cone | Ridge + empirical quantile bands | 7-trading-day close | central path + P5–P95 band |
| 4 | 14-day close cone | Ridge + empirical quantile bands | 14-trading-day close | central path + P5–P95 band |
| 5 | 3-class day type | Logistic (balanced) | Trend-Up / Chop / Trend-Down | directional gauge |

**Honest framing.** Intraday gold direction is essentially a coin-flip (~49–50%),
exactly as it is for BTC — the hourly model's value is a *tight, well-calibrated
CI*, not a directional bet. The real edge lives in the **trend regime** and
**risk control**, which the strategy backtest quantifies below.

---

## The strategy (short version)

**One strategy — the gold-scaled BTC-style Divergence Pure-Regime system.** Enter
when U1 bullish divergence (regime-centered `err_hi` > +0.08%) confirms inside the
Pure-Regime gate (Bull Regime *or* washed-out Clean Breakout *or* V-reversal);
exit on D2 (< −0.10%) / D3 exhaustion or a fixed −3% stop.

The signal comes from **GLDM (gold)** but is executed in its leveraged / high-beta
proxies — the 1× GLDM position is **not traded**, exactly as the BTC app trades
**MSTR** and **MSTU** rather than spot BTC:

- **GDX** — VanEck Gold Miners ETF, high-beta gold (**~MSTR analog**)
- **UGL** — ProShares Ultra Gold, 2× gold (**~MSTU analog**)

Out-of-sample (2021→now) it beats buy-&-hold on **both return and drawdown** for
both assets: **GDX +270%** (MDD −16% vs −47%, Sharpe 1.40) and **UGL +207%**
(MDD −18% vs −49%, Sharpe 1.29) — and it stayed net-positive through the
2021–2022 gold correction while buy-&-hold fell ~25%.

Full methodology, parameter choice and per-period results:
**[`GLDM_TRADING_STRATEGY.md`](GLDM_TRADING_STRATEGY.md)**.

---

## Files

```
app/gldm_core.py          data fetch (Yahoo chart API) + gold features + strategy logic
app/gldm_hourly_app.py    the Streamlit app (Live / H·L·Cones / Strategy / Signals / Explain)
src/gldm/train_gldm.py    trains all five models → models/gldm/*.joblib
backtest_gldm.py          out-of-sample backtest engine + threshold frontier sweep
models/gldm/*.joblib      trained artefacts
data/gldm/*.csv           cached GLDM + macro snapshots (offline fallback / reproducibility)
streamlit_app.py          root router: BTC ⇄ GLDM sidebar radio
```

## Reproduce

```bash
python src/gldm/train_gldm.py            # fetch fresh data + train all 5 models
python backtest_gldm.py --sweep          # run the backtest + per-asset MA/threshold sweep
streamlit run streamlit_app.py           # launch, pick GLDM in the sidebar
# add --cached to the two scripts to use the committed data/gldm snapshots
```

The models are deliberately lightweight (ridge / logistic) so they retrain in
seconds and the whole pipeline is transparent and reproducible.
