# ETH-native regime-divergence engine vs the BTC parent signal — evaluation

**Question.** The shipped ETH sleeve trades spot ETH (12:00-UTC bars, executed
live via the ETHA ETF) off the **BTC** regime-divergence engine — the CT H/L
ensemble's U1/D2/D3 signals with the Standard-MA gate and a −8% stop
(`app/btc_ct_engine.py`, evaluated in `ETH_BMNR_STRATEGY_EVAL.md`). Would it be
better to **train a new regime-divergence engine on Ethereum's own prices** —
an ETH High/Low forecast model plus ETH-side divergence signals, with
thresholds tuned the repo's own way — than to keep ETH on the BTC engine?

**Evaluation only.** No strategy, config, or allocation is changed by this
memo. Reproduce with:

```bash
python scripts/eval_eth_native_divergence.py
```

(TBD — results being filled in)
