# Does the Overall book need an emergency kill switch?

**Question.** Should the strategy carry a portfolio-level circuit breaker — one
rule that liquidates every sleeve into cash/SATA when an extreme market event
hits — on top of the per-sleeve signal exits and stops it already trades?

**Short answer.** Not as a *back-test* rule: every drawdown-triggered breaker
tested loses more return than it saves, and roughly a third of them make the
maximum drawdown **worse**. But the question exposes two things that are real:
the book has **no portfolio-level risk control at all** (it is ~98% deployed
every single day, and one parent signal has carried up to **61% of NAV**), and
the live kill switch that does exist **freezes the executor rather than
liquidating** — pulling it in a crash also blocks the strategy's own exits.
Fix those two instead.

```bash
python scripts/eval_kill_switch.py
```

Every variant below is a strictly causal overlay on the published walk-forward
gated replay: the decision is taken at the close of bar *t−1* and applies from
bar *t*, and while halted the book earns SATA on business days — the same
treatment `replay_gated_allocation` already gives undeployed weight. Figures are
the full OOS window (2021-01-05 → 2026-09-02), Balanced profile unless noted.
Baseline: **+1,398% / −24.6% MDD / Sharpe 1.70 / vol 25.3%**.

---

## 1. The premise checks out: nothing currently de-risks the portfolio

The design note says idle capital parks in SATA. In practice it almost never
does, because the water-fill refills the survivors to their caps:

| Reading (Balanced replay, 1,707 bars) | Value |
|---|---|
| SATA weight, mean / median | **1.6% / 0.0%** |
| Days with cash above 50% | **0 of 1,707** |
| Deployed fraction, mean / min | 0.98 / 0.58 |
| Parent signals funded, mean / min | 4.3 / **1** |
| Largest single parent, mean / max | **38.6% / 61.1%** |

When sleeves exit, the book does not go to cash — it **concentrates**. In the
April 2025 drawdown it ran 58% XLE-family + 40% GLDM-family across just two
parent signals; in June–July 2026 it was 43% XLE + 40% SOXX. The per-instrument
caps are per *instrument*, so one parent's core + beta + lev sleeves stack to
30 + 18 + 10 = 58% of NAV on a single signal. That is the concentration the
kill-switch instinct is reacting to, and it is worth fixing — but a trigger is
the wrong instrument for it.

---

## 2. Trailing-drawdown breaker — the classic kill switch. It loses.

Book drawdown from peak ≤ −X% → flat for N days, peak re-armed on re-entry.

| Rule | Return | MDD | Sharpe | Δreturn | ΔMDD | ΔSharpe |
|---|---:|---:|---:|---:|---:|---:|
| DD ≤ −8%, flat 5d | +749% | −22.0% | 1.44 | −648pp | +2.6pp | −0.27 |
| DD ≤ −8%, flat 10d | +388% | **−31.1%** | 1.16 | −1,010pp | **−6.5pp** | −0.55 |
| DD ≤ −10%, flat 5d | +1,499% | −18.8% | 1.78 | +101pp | +5.8pp | +0.07 |
| DD ≤ −10%, flat 10d | +1,009% | −26.5% | 1.59 | −389pp | −1.9pp | −0.11 |
| DD ≤ −15%, flat 10d | +1,230% | −21.2% | 1.66 | −168pp | +3.4pp | −0.04 |
| DD ≤ −20%, flat 21d | +1,230% | −26.0% | 1.66 | −168pp | −1.4pp | −0.04 |

Note the non-monotonicity: −8%/10d makes the drawdown **6.5pp worse** while
−10%/5d makes it 5.8pp better. A control whose sign flips on a two-point
parameter change is fitting noise, not managing risk. The mechanism is
straightforward — this book's drawdowns are V-shaped (2025-04 bottomed in 5
days, 2026-06 in 2), so a breaker that trips near the low sells the bottom and
sits out the rebound, then re-arms into the next leg down.

Holding until the drawdown recovers is worse still, because the strategy's own
exits have already de-risked the sleeves by then:

| Rule | Return | MDD | Sharpe | Days out |
|---|---:|---:|---:|---:|
| DD ≤ −10% until recovery | +889% | −14.2% | 1.65 | 22.8% |
| DD ≤ −15% until recovery | +970% | −18.0% | 1.65 | 18.7% |
| DD ≤ −20% until recovery | +1,161% | −22.0% | 1.66 | 5.3% |

Every one of them buys drawdown with several hundred points of return and gives
back Sharpe.

**Across all 90 drawdown- and day-triggered configs** (both grids, Balanced):
Sharpe improved in **23%**, return in **11%**, both in **11%**; median ΔSharpe
−0.05, median Δreturn −250pp. On Growth it is 22% / 7% / 7%, median −0.08 and
−579pp. Drawdown improves in ~two-thirds of configs — but that is what *any*
de-risking does, and it is bought badly: the median breaker pays ~190pp of
return per point of drawdown removed, against ~33pp for the always-on control
in §5.

---

## 3. Single-day crash breaker — the one that looks good, and why it isn't enough

One-day book return ≤ −5% → flat 5 days is the only rule that beats the baseline
on both profiles, and its parameter neighbourhood is a smooth ridge rather than
a needle (ΔSharpe on Balanced: +0.06 / +0.15 / **+0.21** / +0.12 / +0.02 across
2/3/5/7/10-day cooldowns at −5%):

| Profile | Rule | Return | MDD | Sharpe | Δ vs baseline |
|---|---|---:|---:|---:|---|
| Balanced | day ≤ −5%, flat 5d | +1,878% | −18.7% | 1.91 | +480pp / +5.9pp / +0.21 |
| Growth | day ≤ −5%, flat 5d | +2,015% | −23.4% | 1.68 | +273pp / +6.2pp / +0.13 |

Then look at the sample it rests on — **9 events in 5.7 years**:

| Date | Trigger | Next 5 bars | Saved |
|---|---:|---:|---:|
| 2021-02-25 | −5.33% | −1.12% | +1.12pp |
| 2021-03-18 | −5.38% | −3.00% | +3.00pp |
| 2022-05-09 | −6.79% | +2.83% | **−2.83pp** |
| 2022-06-13 | −5.06% | −7.04% | +7.04pp |
| 2025-04-03 | −6.30% | −11.00% | **+11.00pp** |
| 2026-01-30 | −6.18% | −0.84% | +0.84pp |
| 2026-06-05 | −9.11% | +0.43% | **−0.43pp** |
| 2026-06-23 | −6.42% | −3.20% | +3.20pp |
| 2026-07-01 | −5.74% | −1.96% | +1.96pp |

Mean +2.77pp per event, **t = 2.02** — under the 95% bar, and the single April
2025 tariff crash is **44% of the whole benefit**. Two of nine events lost
money. On Growth it is *negative* in 2021–22 (Sharpe 0.95 → 0.92) and only
positive in the two later windows. That is a plausible effect on a nine-point
sample, not an established one.

**Verdict:** defensible as a *small* discretionary overlay if you want one, but
it should not be sold as a back-tested edge, and it is nowhere near a reason to
build a full liquidate-to-cash machine.

---

## 4. Volatility targeting — honest, and strictly dominated here

Scale the whole book by target ÷ trailing 20-day realised vol (lagged one bar):

| Target | Return | MDD | Sharpe | Mean exposure |
|---|---:|---:|---:|---:|
| 15% | +638% | −13.8% | 1.87 | 0.69 |
| 20% | +861% | −18.3% | 1.76 | 0.84 |
| 25% | +1,020% | −21.4% | 1.70 | 0.92 |
| 30% | +1,233% | −23.7% | 1.71 | 0.96 |

It works the way it is supposed to (drawdown falls monotonically with the
target), and it is far better behaved than any trigger. But it pays −760pp of
return for +10.8pp of drawdown at the 15% target — a worse exchange rate than
§5 on every rung.

---

## 5. What actually fixes the risk: cap the parent signal, not the calendar

No single parent app above X% of NAV, the excess to SATA. Not an event trigger —
a permanent structural control, on every day.

| Cap | Return | MDD | Sharpe | Δreturn | ΔMDD | ΔSharpe | Mean cash |
|---|---:|---:|---:|---:|---:|---:|---:|
| 25% | +1,194% | −18.5% | **2.00** | −204pp | +6.1pp | **+0.30** | 19.8% |
| 30% | +1,371% | −19.9% | 1.93 | −27pp | +4.7pp | +0.22 | 12.0% |
| **35%** | **+1,402%** | **−21.5%** | **1.85** | **+4pp** | **+3.1pp** | **+0.15** | 8.0% |
| 40% | +1,418% | −23.2% | 1.79 | +20pp | +1.4pp | +0.09 | 5.2% |
| 50% | +1,428% | −22.7% | 1.75 | +30pp | +1.9pp | +0.04 | 2.9% |

Monotone in the parameter, positive at every rung, and the same shape on Growth
(cap 30%: +0.20 Sharpe; cap 35%: +0.15). A **35% parent cap is free** — +4pp of
return, 3.1pp less drawdown, +0.15 Sharpe — and it holds in every sub-period:

| Window (Balanced) | Baseline | Parent cap 35% |
|---|---|---|
| 2021–22 | +63% / −24.6% / 0.98 | +60% / **−16.4%** / **1.04** |
| 2023–24 | +104% / −14.5% / 1.68 | +107% / −14.4% / **1.82** |
| 2025–26 | +351% / −22.7% / 2.44 | +355% / −21.5% / **2.61** |

Same on Growth (2021–22 MDD −29.6% → −16.7%). It also gives the book the
property the kill switch was supposed to provide: when signals fail together,
cash rises **automatically**, because the excess is no longer force-fed into
whatever survived.

---

## 6. The live gap that is real — and the opposite of what you'd want

`scripts/ibkr_execute_book.py` already has a kill switch
(`IBKR_TRADING_DISABLED` / `--kill-switch-file`), and `docs/LIVE_TRADING.md`
documents `touch STOP_LIVE`. It aborts **before any order is placed**:

```python
if os.environ.get("IBKR_TRADING_DISABLED") or (
        args.kill_switch_file and Path(args.kill_switch_file).exists()):
    print("ABORT: trading is DISABLED (kill switch active). No orders placed.")
    return 0
```

That is a **freeze, not a liquidation**. Pulling it mid-crash leaves the account
fully invested *and* stops the strategy's own signal exits and stops from ever
reaching the broker — the one action guaranteed to make a crash worse. It is the
right tool for "the model or the data feed is wrong, stop trading"; it is the
wrong tool for "the market is falling".

If a true panic button is wanted, it should be a **separate, explicit
liquidate-to-SATA mode** — e.g. `--flatten-all`, which publishes an all-SATA
book and lets the normal executor path work it — kept manual and off the
back-test, so it is an operator decision about the world rather than a rule the
strategy claims an edge from.

---

## 7. Recommendation

1. **Don't add a drawdown-triggered kill switch.** It is negative on the median
   config, unstable in sign, and its own back-test can't distinguish it from
   noise.
2. **Add a 35% parent-signal cap** to `signal_gated_allocation` /
   `replay_gated_allocation` (excess to SATA, no redistribution). Free return,
   less drawdown, more Sharpe, monotone, and it makes the book raise cash on its
   own when signals fail together — the actual thing a kill switch was meant to
   do. This is the change worth making.
3. **Split the live kill switch in two**: keep `STOP_LIVE` as the freeze (for
   bad data / bad model), and add an explicit `--flatten-all` liquidate mode so
   the panic button does what the name implies. Manual, unmodelled, documented.
4. Optionally, treat "one-day book loss ≤ −5% → skip 5 sessions" as a
   *discretionary* overlay. It is the only trigger that survives both profiles,
   but on n = 9 with 44% of the benefit from one day it is a judgement call, not
   an edge.

## Caveats

- The window contains no 2008/2020-style event: the deepest baseline drawdown is
  −24.6% and the worst day −9.1%. A breaker's insurance value against a tail
  this history never saw is, by construction, untestable here — this memo says
  the *tested* rules cost more than they save, not that tail risk is absent.
- No transaction costs are modelled. The breakers churn far more than the
  baseline (up to 32% of days out of the market), so their real-world results
  are worse than shown; the parent cap adds little turnover.
- Overlays are applied to the replay's realised return stream, so a halt is
  modelled as instantaneous at the next bar's close, with no slippage on the way
  out or back in — again generous to the breakers.
- SATA is assumed to pay its ~13% coupon throughout, as everywhere else in the
  Overall back-test.
