# ETH-native regime-divergence engine vs the BTC parent signal — evaluation

**Question.** The shipped ETH sleeve trades spot ETH (12:00-UTC bars, executed
live via the ETHA ETF) off the **BTC** regime-divergence engine — the CT H/L
ensemble's U1/D2/D3 signals with the Standard-MA gate and a −8% stop
(`app/btc_ct_engine.py`, evaluated in `ETH_BMNR_STRATEGY_EVAL.md`). Would it be
better to **train a new regime-divergence engine on Ethereum's own prices** —
an ETH High/Low forecast model plus ETH-side divergence signals, with
thresholds tuned the repo's own way — than to keep ETH on the BTC engine?

**Answer: NO — keep ETH on the BTC engine.** Every ETH-native variant (six
model × feature-set combinations, plus a walk-forward rerun of the best one)
looks spectacular on the tuning window and then **loses money on the unseen
Aug-2025 → now holdout**, where the shipped BTC-parent sleeve is the *only*
strategy in the comparison that stayed positive (+8.4%, MDD −6.8%). The
in-sample edge of the ETH-native engines is threshold overfit, not signal.

**Evaluation only.** No strategy, config, or allocation is changed by this
memo. Reproduce with:

```bash
python scripts/eval_eth_native_divergence.py      # ~3 min; needs network for
                                                  # the 2019→ Binance history
```

---

## 1. Method

Everything mirrors the repo's own conventions so the comparison is
like-for-like:

* **ETH H/L model = the CT recipe ported to ETH.** The exact feature blocks of
  `backtest_trailing_stop.build_preds_offline` computed on ETH OHLCV, and the
  same 3-model ensemble (Huber + linear q0.70 + GBM q0.70, `ctens`). The
  ticker-app template (`backtest_ticker.build_predictions`'s ridge) is swept
  alongside (`ridge`). Three feature sets:
  * **E** — ETH-only price/TA (≈50 features);
  * **EB** — + BTC-parent block (BTC returns/vol, ETH-BTC correlation,
    BTC-above-SMA30, ETH/BTC ratio z-score);
  * **EBM** — + the macro / Coinbase-premium / on-chain block from the
    committed feature CSV (only spans 2023-11 →, so this set trains on ~610
    rows instead of ~2 300).
* **Training depth matched to the BTC artifact.** ETH OHLCV comes from Binance
  on the **same 12:00-UTC bar anchor** as the BTC signal
  (`pull_backtest_data.fetch_12utc`), extended back to **2019-01** via the
  `data-api.binance.vision` mirror (2 741 bars; the committed CSV starts
  2023-11; the BTC CT artifact trained on 2019-03 → 2026-02). Overlap vs the
  committed CSV differs ≤ 0.64% (different Binance venue).
* **Signals + backtest loop are faithful ports** of
  `btc_ct_engine.compute_sigs_pure` / `_run_bt`: same U1/D1/D2/D3
  construction, MA30 regime, Pure-Regime and Standard-MA gates, V-reversal,
  post-stop re-entry override, same-bar fill (legitimate — ETH bars close at
  the signal moment, exactly as BTC's do).
* **Honest tuning protocol** (the `eval_regime_aug2025.py` convention): the
  model is fit strictly on pre-Aug-2025 rows (1-bar embargo); divergence
  thresholds (U1 × D2 × V × gate × regime-source × stop, grids scaled to the
  ETH signal's own error std per `backtest_ticker.divergence_scale`) are tuned
  on **2024-03-05 → 2025-08-01 only**, selection rule = maximise Sharpe s.t.
  MDD ≤ B&H, tie-broken by return. **Aug-2025 → now is a true holdout** for
  both the model and the thresholds.
* A **regime-source axis** lets the divergence stay ETH-native while the MA30
  regime gate reads **BTC** (the parent-trend hybrid), since the earlier memo
  showed parent trend beats own trend on ETH.
* **Walk-forward robustness:** the champion combo is refit on an expanding
  window every 63 bars (15 refits, 2024-02 → 2026-07), so every prediction is
  out-of-sample; the fixed chosen thresholds are then applied.

Windows: **full** = the shipped sleeve's window 2024-03-05 → 2026-07-24;
**holdout** = 2025-08-01 → 2026-07-24 (a brutal ETH bear: B&H −47%).
Incumbent baselines reproduce `ETH_BMNR_STRATEGY_EVAL.md` exactly
(+8.8% no-stop, +40.0% / −22.9% shipped).

---

## 2. Results

*(cells are total return / max drawdown / Sharpe)*

| Strategy | Full 2024-03-05 → now | Holdout Aug-2025 → now |
|---|---|---|
| ETH Buy & hold | −52.1% / −67.5% / −0.09 | −47.1% / −67.5% / −0.63 |
| **CT-on-BTC, MA gate, −8% stop (shipped)** | **+40.0% / −22.9% / 0.52** | **+8.4% / −6.8% / 0.47** |
| CT-on-BTC, MA gate, no stop | +8.8% / −39.5% / 0.23 | +8.4% / −6.8% / 0.47 |
| BTC>SMA30 → ETH (10 bps) | +11.7% / −48.1% / 0.27 | −37.9% / −48.1% / −1.10 |
| ETH own SMA30 (10 bps) | +2.0% / −39.6% / 0.19 | −4.7% / −35.9% / 0.04 |
| ETH-native div **E/ridge** | +38.4% / −32.0% / 0.47 | −24.6% / −24.6% / −1.33 |
| ETH-native div **E/ctens** | +57.8% / −27.9% / 0.61 | −18.7% / −27.9% / −1.00 |
| ETH-native div **EB/ridge** | +38.1% / −37.1% / 0.48 | −35.1% / −37.1% / −1.55 |
| ETH-native div **EB/ctens** | +10.0% / −47.6% / 0.24 | −40.9% / −47.6% / −1.49 |
| ETH-native div **EBM/ridge** ◀ champion | +140.9% / −33.3% / 0.92 | −2.6% / −33.3% / 0.10 |
| ETH-native div **EBM/ctens** | +52.1% / −31.8% / 0.62 | −22.7% / −31.8% / −0.89 |
| ETH-native champion, **walk-forward** refits | +27.1% / −36.1% / 0.37 | −10.0% / −34.3% / −0.05 |

On the tuning window itself (2024-03 → 2025-08, where ETH B&H was −9.6%),
every ETH-native combo returned **+65% to +147% at Sharpe 1.0–1.4** — and then
every one of them went **negative on the year the tuner never saw**, with
24–48% drawdowns against the shipped sleeve's −6.8%. The champion
(EBM/ridge — the *smallest* training set, 610 rows, so also the easiest to
flatter) is merely the least bad at −2.6% / Sharpe 0.10; refit walk-forward
its full-window return collapses from +141% to +27%, confirming the headline
number was fit, not forecast.

Supporting detail:

* **Forecast quality.** Holdout MAPE on ETH highs/lows is 1.6–3.0% across the
  combos vs ≈ 1.1–1.2% for the BTC model on BTC — next-bar ETH ranges are
  simply harder to pin down, so the divergence signal that feeds U1/D2/D3 is
  noisier at the source.
* **The sweep itself keeps voting for BTC.** 5 of 6 combos chose the **BTC**
  MA30 regime gate over ETH's own even though the divergence layer was
  ETH-native, and all six chose the −8% stop — independently rediscovering
  the shipped sleeve's two structural choices.
* **Porting the BTC engine's literal constants** (U1 +1.3 / D2 −1.3) onto ETH
  errors is roughly flat (−4.2% full, −2.1% holdout): ETH's error scale
  (σ(ehma3) ≈ 1.5–1.9 vs BTC's calibrated 1.3 thresholds) means the constants
  don't transfer either.
* **Diversification angle.** The ETH-native champion's return stream is
  0.36-correlated to the live BTC sleeve vs 0.59 for the shipped ETH sleeve —
  a real decorrelation, but of a stream with holdout Sharpe 0.10 and a −33%
  drawdown; it earns no weight the optimiser would want.

---

## 3. Bottom line & recommendation

1. **Do not build an ETH-native regime-divergence engine.** Six variants
   spanning both repo model templates, three feature sets, ETH- and
   BTC-regime gates, and a matched 2019→ training depth all fail the unseen
   year — uniformly worse than the shipped sleeve on return, drawdown *and*
   Sharpe. The walk-forward rerun shows the in-sample edge does not survive
   honest refitting.
2. **If ETH is traded at all, the BTC parent signal remains the best engine
   for it** — on the holdout it is the only positive strategy in the table,
   at a fraction of everyone else's drawdown. This extends the repo's
   established "trade the parent, not the child" result (MSTR/MSTU, and the
   parent-trend finding in `ETH_BMNR_STRATEGY_EVAL.md`) from trend filters to
   the full divergence engine.
3. **The live question stays the one the earlier memo raised** — not *which*
   engine drives ETH, but whether the ETH sleeve earns its place in the book
   at all (it was recommended for removal / monitored-only status there).
   Nothing in this evaluation strengthens the case for ETH; it only closes
   the "maybe a native engine would fix it" avenue.

---

## 4. Addendum — the shipped strategy backtested from 2021

Follow-up question: *what would the ETH backtest look like from 2021 with the
currently implemented strategy?* The shipped sleeve reports from 2024-03-05
only because the CT feature CSV starts 2023-11, so
`scripts/eval_eth_2021_backtest.py` rebuilds the same 116-feature matrix back
to 2020-06 (Binance-mirror BTC 12:00-UTC bars, Yahoo macro, blockchain.info
on-chain, Coinbase premium — the pull script's own recipe), splices the
committed CSV verbatim from 2023-11, and runs the deployed engine unchanged.
The validation row over the shipped window reproduces the published figures
exactly (+40.0% / −22.9% / Sharpe 0.52), so the extension is trustworthy on
the live era.

**2021-01-01 → 2026-07-24** *(return / maxDD / Sharpe; trades / win rate)*:

| Strategy | Result |
|---|---|
| CT-on-BTC, Standard-MA gate, −8% stop (shipped) | **+0.7% / −60.6% / 0.17** · 36 tr / 44% |
| CT-on-BTC, Standard-MA gate, no stop | +26.2% / −61.1% / 0.26 · 34 tr / 47% |
| ETH Buy & hold | +154.6% / −78.9% / 0.50 |

Per year (shipped config vs B&H, each year rebased):

| Year | Strategy | Buy & hold |
|---|---|---|
| 2021 | −7.5% / −36.8% | **+406.1%** / −58.9% |
| 2022 | −14.1% / −43.3% | −68.0% / −73.8% |
| 2023 | −18.8% / −31.3% | +89.2% / −27.0% |
| 2024 | +12.5% / −23.0% | +39.7% / −43.7% |
| 2025 | **+28.1%** / −18.7% | −14.0% / −60.1% |
| 2026 YTD | **+8.4%** / −6.8% | −39.0% / −53.8% |

Reading: over the full window the current strategy is roughly **flat (+0.7%)
with a −61% drawdown**, far behind B&H's +155% — it misses essentially all of
2021's +406% (14 choppy trades, five −8% stops) and bleeds through 2022-23.
It only starts adding value from 2024 on, and is *excellent* in 2025-26
(+28% and +8% against B&H's −14% and −39%). That inflection is exactly where
the thresholds/gate/stop were tuned — and the deploy-mode CT model is
in-sample over 2021→2026-02 anyway — so the 2021-23 weakness is a fair
warning that the sleeve's published edge is concentrated in the recent,
tuned-on regime, while the pre-2024 numbers cannot be read as honest OOS
either (they're the same model *with* look-ahead into its training window,
and it still lost). Additional reconstruction caveats: pre-2023-11 features
are re-fetched today (not a production vintage) and pre-2023-11 ETH prices
come from the global-Binance mirror (≤0.6% venue difference), so
reconstructed-era signals are approximate.

### Caveats

* The holdout is a single ~1-year regime (a deep ETH bear). A long-only
  engine tuned on a window that rewarded catching rallies was always going to
  struggle there — but the incumbent navigated the same year at +8.4%, so the
  comparison is fair, and the walk-forward full-window result (+27% vs the
  fitted +141%) says the overfit is real, not regime-specific.
* Six combos were searched; all six are reported and the champion was chosen
  on the training window only. Threshold grids follow the repo's
  error-std-scaled convention, and the walk-forward applies thresholds chosen
  through Aug-2025 (mildly optimistic before that date, honest after).
* Divergence backtests use the app-faithful loop (no per-switch cost, 5–23
  trades — same convention as the shipped sleeve's numbers); simple-rule rows
  charge 10 bps per switch (the prior memo's cost model).
* Long-history bars come from the `data-api.binance.vision` mirror
  (global-Binance prices); they differ from the committed `api.binance.us`
  bars by ≤ 0.64%. CT-engine figures drift with feature-CSV refreshes; this
  memo is the 2026-07-25 run on the 2026-07-24 data vintage. Backtested
  performance is not indicative of future results.
