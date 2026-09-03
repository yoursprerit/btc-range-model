# Net global liquidity as a feature — evaluation for the BTC daily High/Low model

**Question.** The daily High/Low model (`src/pipeline_ct.py`) currently sees BTC
price/volume, seven cross-asset macro series, eleven blockchain.info on-chain
series, Fear & Greed and the Coinbase premium. It does **not** see central-bank
liquidity in any form. Should it? Specifically: does *true net liquidity* —
Fed balance sheet minus the Treasury General Account minus overnight reverse
repo, optionally plus the ECB — improve next-bar high/low accuracy?

**Not implemented.** This is an evaluation only. No production feature set,
model or config was changed. Reproduce with:

```bash
python src/research_liquidity_features.py                      # main ablation, 2024-2026
python src/research_liquidity_features.py --dataset extended   # 2018-2026 robustness
python src/research_liquidity_horizon.py                       # value by forecast horizon
```

## Method

**The series.** `src/liquidity_data.py` builds

```
net_liq = WALCL − WTREGEN − RRPONTSYD          (Fed assets − TGA − overnight RRP)
global_liq = net_liq + ECBASSETSW × USD/EUR
```

from FRED, cached under `data/liquidity/`. The other major central banks are
monthly and land on FRED weeks late, so a *same-day-honest* global aggregate is
the Fed plus the ECB plus a stale constant — adding the BoJ and PBoC at their
real publication lag would contribute almost no timely variation.

**Point-in-time discipline.** Every FRED observation carries its *observation*
date, not its release date. H.4.1 for the week ending Wednesday is not published
until Thursday 16:30 ET, and the ON-RRP print for day D lands ~13:15 ET on D —
both after the 12:00-UTC bar the model scores. Each series is therefore shifted
to the first bar that could genuinely have seen it, plus a further safety day.
Skipping this manufactures an edge that does not survive live trading.

**Features.** Levels are non-stationary, so nothing uses a raw level: 1w/4w/13w
log changes, position in a 1-year range, an acceleration term, the volatility of
liquidity flows, and the two fast-moving components (TGA, RRP) that traders
actually watch — twelve columns in total.

**Evaluation.** Walk-forward rather than a single train/val/test cut: expanding
train window, refit every 21 days, always scoring days the model has never seen.
Arms are compared paired on identical rows with a Diebold-Mariano test
(Newey-West HAC, lag 5) plus a 21-day block bootstrap. The production
climatology blend (α) is replicated with α calibrated only on strictly earlier
folds, because that shrink is what actually ships and it damps whatever a new
feature block does.

**The control that matters.** Every comparison carries a `base+noise` arm — the
same *number* of persistence-matched random columns. Without it, "adding
liquidity costs 2bp" is unreadable: you cannot tell a useless signal from the
generic cost of widening the feature matrix.
