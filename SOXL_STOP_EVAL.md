# Adding a fixed stop to SOXL (3× semis) — evaluation

**Question.** SOXL trades the SOXX 25/100 dual-MA with **no fixed stop**
(`stop_by_asset={"soxl_close": 1.0}`); the inherited −5% stop was dropped as far
too tight for a 3× ETF (`SOXL_ERX_ADDITION_EVAL.md`). But −5% is not the only
option — **would adding *some* stop, sized for leverage (−10% … −30%), improve
the SOXL sleeve and the Overall blend?** Impacts on **return, Sharpe, win-rate**.

Reproduce with:

```bash
python scripts/eval_soxl_stop.py     # Part 1 (SOXL sleeve grid) + Part 2 (Overall)
```

SOXL trades the SOXX signal; the only thing changed is SOXL's fixed stop across
the grid **none · −30% · −25% · −20% · −15% · −10% · −5%**. SOXX and every other
sleeve are untouched, so all deltas are attributable to the SOXL-stop change.

Window: full OOS **2021 → now** (plus sub-periods).

---

## Part 1 — SOXL sleeve: win-rate strictly falls; a wide stop is at best break-even

### Full OOS (2021 → now)

| Stop | Return | Max DD | Sharpe | Win% | Trades |
|---|---|---|---|---|---|
| **none (shipped)** | +2626% | −69.1% | 1.14 | **80%** | 5 |
| −30% | +2687% | −68.2% | 1.14 | 67% | 6 |
| −25% | **+2922%** | **−65.6%** | **1.16** | 57% | 7 |
| −20% | +2714% | −67.8% | 1.14 | 57% | 7 |
| −15% | +2719% | −67.7% | 1.14 | 50% | 8 |
| −10% | +2431% | −64.4% | 1.12 | 33% | 12 |
| −5% | +1929% | −67.0% | 1.08 | **22%** | 18 |

* **Win-rate — strictly lowered by any stop**, monotonically (none 80% → −5% 22%).
  Each stop-out is a manufactured −X% losing round-trip, so a stop can only *cut*
  the win-rate. There is no win-rate improvement anywhere.
* **Return** — tight stops hurt badly (−5% +1929%, −10% +2431% vs no-stop +2626%);
  a **wide −25% stop marginally beats** no-stop (+2922%, ≈+11% relative), but
  −20/−15% are flat and this is a single-path artifact, not a repeatable edge.
* **Sharpe** — essentially flat for wide stops (−25% 1.16 vs 1.14), degrades as
  the stop tightens (−5% 1.08).
* **Drawdown** — barely dented even by a wide stop (−69%→−66% best): a
  close-based fixed stop can't catch 3× gap/intraday moves.

### The bear is the only place a wide stop earns its keep (2021–2022)

| Stop | Return | Max DD | Sharpe | Win% | Trades |
|---|---|---|---|---|---|
| none | −22.1% | −69.1% | 0.22 | 67% | 3 |
| −25% | **−13.5%** | −65.6% | **0.29** | 40% | 5 |
| −10% | −27.1% | −64.4% | 0.16 | 20% | 10 |
| −5% | −41.6% | −67.0% | 0.01 | 12% | 16 |

A **wide −25% stop cuts the 2021–22 loss to −13.5%** (from −22.1%) and lifts
Sharpe 0.22→0.29 — but still at a **lower win-rate** (40% vs 67%). **Tight stops
make the bear far worse** (−5% → −41.6%), the whipsaw that motivated dropping the
inherited −5% in the first place. (Bull 2023→ and Recent 2025→ hold 0–2 trades,
so every stop returns the identical +3517% / +886% — a stop is irrelevant when
the trend never retraces to it.)

---

## Part 2 — Overall combined portfolio: no meaningful gain

Same 17-instrument universe, the **identical** `optimize_weights()` the app and
`build_overall.py` use; only SOXL's stop changes.

**Aggressive optimal (fundamental view on):**

| SOXL stop | sleeve ret / Sh / win | Agg return | Agg MDD | Agg Sharpe | SOXL wt |
|---|---|---|---|---|---|
| none | +2626% / 1.14 / 80% | +2208% | **−26.7%** | **1.59** | 35.0% |
| −25% | +2922% / 1.16 / 57% | +2621% | −29.6% | 1.53 | 27.1% |
| −20% | +2714% / 1.14 / 57% | +2525% | −30.4% | 1.51 | 27.1% |
| −10% | +2431% / 1.12 / 33% | +2383% | −29.6% | 1.50 | 27.1% |
| −5% | +1929% / 1.08 / 22% | +2118% | −29.6% | 1.45 | 27.1% |

Adding a stop **cuts SOXL's optimal weight (35%→27%)** and buys a bit more raw
return by reshuffling into other sleeves, but at a **lower Sharpe (1.59→~1.51)
and deeper drawdown (−26.7%→~−30%)** — a worse risk-adjusted book, not a better
one.

**Equal-weight blend (deterministic — clean attributable signal):**

| SOXL stop | Return | Max DD | Sharpe |
|---|---|---|---|
| none | +770.2% | −7.1% | 2.775 |
| −20% | +772.5% | −7.1% | 2.778 |
| −10% | +764.9% | −7.1% | 2.775 |

The trustworthy deterministic signal is **≈0** (±2–5pp on a +770% base, Sharpe
±0.003). The optimised-profile swings are optimiser reshuffle noise, not a real
edge.

---

## Verdict

**No — adding a stop to SOXL does not improve performance.**

* **Win-rate:** strictly *lowered* by any stop (80% → 22% as it tightens) — the
  opposite of an improvement.
* **Return / Sharpe:** tight stops (−5%/−10%) clearly hurt both; only a very wide
  (~−25%) stop is roughly break-even (a hair more return, Sharpe +0.02) and even
  that is single-path noise. Its only genuine benefit is trimming the 2021–22
  bear loss (−22%→−14%), still at a lower win-rate.
* **Overall book:** nil on the clean equal-weight signal; mildly *negative*
  (lower Sharpe, deeper drawdown) in the optimised Aggressive profile.

The shipped **no-stop** SOXL config is the right call. If any stop were used it
would have to be very wide (≈−25%, never −5/−10%) and the payoff is marginal and
essentially bear-market-only, paid for with win-rate. **No config change is
warranted.** (See `SOXX_STOP_EVAL.md` for the mirror case: 1× SOXX *keeps* its
−5% stop because a −5% stop is right-sized for 1× and only too tight for 3×.)
