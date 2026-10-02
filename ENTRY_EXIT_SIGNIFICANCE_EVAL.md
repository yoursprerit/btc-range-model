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

## Exit package (signal exit + fixed stop) and the strategy vs buy-and-hold

Sections D and E of the harness treat the fixed stop as **part of** the exit
rather than holding it constant. The null in D uses the same real entries,
with random holding periods drawn from *all* observed holds and **no** stop.

**D. Exit rule ablation** (real entries, only the exit rule changes):

| Exit rule | BTC | MSTR | MSTU |
|---|---|---|---|
| Signal + stop (live) | +58% · MDD −28% · Sharpe 0.97 | **+245% · −22.5% · 1.55** | **+677% · −42% · 1.48** |
| Signal only | (same, no stop) | +184% · −27.5% · 1.26 | +402% · −49% · 1.19 |
| Stop only | — | −23% · −65% · 0.18 | −50% · −88% · 0.31 |
| Random holds, no stop (null median) | +42% · −26% · 0.74 | +165% · −38% · 1.12 | +313% · −69% · 1.07 |
| Package p (return / MDD / Sharpe) | 0.28 / 0.95 / 0.25 | 0.22 / 0.17 / 0.10 | 0.16 / 0.15 / 0.11 |

On MSTR/MSTU the two parts work together: signal plus stop beats either one
alone on return, drawdown and Sharpe. The stop on its own is a disaster
because it lets trades run until they get stopped out. Compared with a random
holding period, the package is consistently better, but **not significant on
its own** (p 0.10–0.22).

**E. Full strategy (entries + signal exits + stop) vs buy-and-hold:**

| | BTC | MSTR | MSTU |
|---|---|---|---|
| Return vs B&H | +58% vs +29% | +245% vs +29% | +677% vs −89% |
| MDD vs B&H | −28% vs −53% | −22.5% vs −83% | −42% vs −99% |
| Sharpe vs B&H | 0.97 vs 0.44 | 1.55 vs 0.55 | 1.48 vs 0.35 |
| Block bootstrap P(excess return ≤ 0) | 0.37 | 0.22 | **0.04** |
| Block bootstrap P(Sharpe ≤ B&H) | 0.22 | 0.06 | **0.04** |
| Random-timing null, p(null ≥ strategy) | 0.10 | **0.02** | **0.01** |
| Rolling 6-month windows beating B&H | 58% | 73% | 83% |
| Log excess by year (2024 / 2025 / 2026) | −0.28 / +0.28 / +0.20 | −0.32 / +0.96 / +0.35 | +0.27 / +2.65 / +1.38 |

- **MSTU** beats buy-and-hold significantly on every test. Its benchmark,
  however, is a 2× product that decays in volatile markets (B&H −89%), which
  makes it easy to beat.
- **MSTR** has clear timing skill: a random timer of the same exposure almost
  never matches it (p=0.02). Its risk-adjusted edge over B&H is borderline
  (p=0.06). Its excess *return* over B&H is not significant (p=0.22),
  because a ~2.5-year sample is too short to rule out luck in a few big moves.
- **BTC** beats buy-and-hold on drawdown and Sharpe, but none of its
  return tests are significant.
- **Reliability:** the strategy lagged buy-and-hold in 2024 (a strong rally
  it was mostly out of) for BTC and MSTR, and beat it in 2025 and 2026. Its
  value is in avoiding drawdowns, not in out-earning a bull market.

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
