# Entry vs exit signal significance — BTC · MSTR · MSTU

**Question:** for the BTC divergence strategy, is the edge in the **entry**
signal (U1 + Standard-MA XOR gate / V-reversal) or the **exit** signal
(D3 exhaustion, or D2 fade outside a bull regime)?

**Harness:** [`scripts/eval_entry_exit_significance.py`](scripts/eval_entry_exit_significance.py)
— runs offline on the committed `data/backtest/` dataset through
`btc_ct_engine` (the live engine). Its simulator reproduces the engine's
trades exactly on every sleeve. Window 2024-03-05 → 2026-10-01 (~940 daily
bars), 2,000 Monte Carlo runs, seed 0.

```bash
python scripts/eval_entry_exit_significance.py                       # BTC MSTR MSTU
python scripts/eval_entry_exit_significance.py --assets ETH --sims 5000
```

## Verdict

**The entries carry the edge; the exits do not.** The exits neither forecast
declines nor reduce drawdown compared with a random holding period of the
same length. On MSTR/MSTU the sleeve tends to keep rising after a signal exit.

| | BTC (no stop) | MSTR (−3%) | MSTU (−6%) |
|---|---|---|---|
| Engine result (strategy vs B&H) | +58% vs +29% | +245% vs +29% | +677% vs −89% |
| Trades (stops) | 8 (0) | 8 (2) | 8 (2) |
| **A.** Fresh entry signal, 5-bar excess return | +3.0% (p=0.03) | +8.2% (p=0.008) | +15.3% (p=0.009) |
| **A.** Fresh exit signal, 5-bar excess return (want < 0) | +0.1% (p=0.57) | +1.3% (p=0.82) | +2.5% (p=0.81) |
| **A.** Executed signal exits, 10-bar move afterwards | +3.0% | +4.9% | +10.7% |
| **B.** Entry skill: random entries + real exits, p | 0.16 | **0.024** | **0.017** |
| **B.** Exit skill: real entries + random holds, p | 0.32 | 0.15 | 0.17 |
| **C.** Max drawdown, real exits vs random holds (median) | −28.1% vs −26.3% (p=0.95) | −22.5% vs −28.6% (p=0.30) | −42.3% vs −51.8% (p=0.33) |

*p-values are one-sided. In A, the null comes from circular-shifting the signal
mask. In B and C, the null comes from Monte Carlo runs. In C, p is the share of
random-hold runs whose drawdown is at least as shallow as the real exits'.*

## Reading it

- **Entry.** The fresh entry signal is followed by above-average returns at
  3–20 bars on all three sleeves. On MSTR/MSTU the effect is significant at
  every horizon, and random entries combined with the real exit rule
  reproduce the result only about 2% of the time. On BTC the effect is weaker
  (p≈0.03 at 5–10 bars only, and B is not significant).
- **Exit.** The fresh exit signal is no better than a coin flip on every
  sleeve. A random holding period of the same length earns about as much as
  the real exits (B, p 0.15–0.32). It also produces about the same drawdown
  (C, p 0.30–0.95), and on BTC the real exits are slightly *worse*. The real
  downside protection on MSTR/MSTU comes from the fixed stop. The stop stays
  active in every null run, so the test credits none of it to the signal exit.
- **Implication.** The exit rule is where there is room to improve, for
  example by holding longer or replacing D2 with a trailing rule. Treat any
  exit retune as a new hypothesis, not as a fix this eval has confirmed.

## Caveats

- **Small sample:** ~8 trades per sleeve, with 7 executed entries
  (excluding the warm start).
- **One shared signal:** all three sleeves trade the same BTC signal, and
  MSTU is 2× MSTR. The MSTR/MSTU results are **not** independent
  confirmations of each other or of BTC.
- **Optimistic p-values:** the U1/D2 thresholds and the stops were tuned on
  this same window, and five horizons are tested. A correction for multiple
  tests would remove the weaker BTC entry results.
- **Override disabled in the nulls:** the null runs switch off the post-stop
  re-entry override, because it is itself a U1-based entry and would leak
  entry skill into the random-entry null.
- **BTC gate:** the BTC sleeve uses `GATE_BY_ASSET["BTC"]`, which is the
  Standard-MA (XOR) gate, even though the `compute_sigs_pure` docstring still
  says "Pure Regime — used for BTC". The harness follows `GATE_BY_ASSET`,
  matching `run_btc_ct`.
