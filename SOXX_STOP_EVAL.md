# Removing SOXX's −5% stop (signal-only exits) — evaluation

**Question.** SOXX and its 3× sibling SOXL trade the *same* 25/100 dual-MA
signal, yet SOXX back-tests with a much lower, choppier **win-rate** (50% vs
SOXL's 80% OOS). SOXL got there by dropping the inherited −5% fixed stop
(`stop_by_asset={"soxl_close": 1.0}`, see `SOXL_ERX_ADDITION_EVAL.md`). **Would
removing SOXX's own −5% stop — signal-only exits, like SOXL — improve the SOXX
sleeve and the Overall combined strategy?**

Reproduce with:

```bash
python scripts/eval_soxx_stop.py     # Part 1 (SOXX sleeve) + Part 2 (Overall blend)
```

Both parts run the live universe once. SOXX trades its 25/100 dual-MA
(`app/ticker_config.py`); the only thing changed is the SOXX (`px_close`) fixed
stop: **−5% (shipped)** vs **1.0 (no stop, signal-only exits)**. SOXL is left on
its no-stop override throughout, so every delta is attributable to the SOXX-stop
change alone.

Window: full OOS **2021-01-04 → 2026-07-09** (plus the standard sub-periods).

---

## Part 1 — SOXX sleeve: the win-rate rises, but real performance does not

| Period | Stop | Return | Max DD | Sharpe | Win% | Trades |
|---|---|---|---|---|---|---|
| 🌐 Full OOS 2021→ | −5% | **+451.7%** | **−26.7%** | **1.22** | 50% | 8 |
| 🌐 Full OOS 2021→ | none | +431.1% | −29.4% | 1.20 | **80%** | 5 |
| 🌍 Full history | −5% | **+1161%** | **−35.4%** | **1.04** | 44% | 16 |
| 🌍 Full history | none | +1128% | −35.4% | 1.03 | 55% | 11 |
| 🐻 Bear 2021–22 | −5% | **+9.8%** | **−26.7%** | **0.31** | 33% | 6 |
| 🐻 Bear 2021–22 | none | +5.7% | −29.4% | 0.24 | 67% | 3 |
| 🐂 Bull 2023→ | either | +407.4% | −24.9% | 1.66 | 100% | 2 |
| 🔬 Recent 2025→ | either | +169.4% | −15.8% | 2.07 | — | 0 |

Removing the stop **raises the win-rate on every period** (OOS 50→80%, full
44→55%, bear 33→67%) — but on the metrics that matter it is **flat-to-slightly-worse**:
return falls (~−20pp OOS, −33pp full, −4pp bear), drawdown deepens
(−26.7%→−29.4%), and Sharpe is unchanged-to-lower (1.22→1.20). The win-rate lift
is cosmetic.

### Why — the trade log (default OOS)

```
−5% stop: 8 trades, 4 green, win-rate 50%, 3 stop-out exits
  2021-01-04 → 2021-05-28   +14.6%  WIN   MA cross-down
  2021-06-15 → 2022-02-09   +15.2%  WIN   MA cross-down
  2022-08-19 → 2022-08-26    −5.2%  LOSS  stop −5%
  2022-08-29 → 2022-09-02    −5.1%  LOSS  stop −5%
  2022-09-06 → 2022-09-13    −0.2%  LOSS  MA cross-down
  2022-12-08 → 2022-12-19    −5.4%  LOSS  stop −5%
  2022-12-20 → 2023-09-29   +33.3%  WIN   MA cross-down   ← re-entered ~5% lower
  2023-12-05 → 2024-08-15   +37.0%  WIN   MA cross-down

no stop: 5 trades, 4 green, win-rate 80%, 0 stop-out exits
  2021-01-04 → 2021-05-28   +14.6%  WIN   MA cross-down
  2021-06-15 → 2022-02-09   +15.2%  WIN   MA cross-down
  2022-08-19 → 2022-09-13   −13.0%  LOSS  MA cross-down   ← one big loss, not 3 clipped ones
  2022-12-08 → 2023-09-29   +25.3%  WIN   MA cross-down   ← rode the Dec dip down first
  2023-12-05 → 2024-08-15   +37.0%  WIN   MA cross-down
```

The mechanism is entirely a **denominator/accounting** effect, not an edge:

* **Every stop-out is, by construction, a realized −5% loss**, and it *splits one
  trend into several round-trips*. So the −5% stop manufactures extra losing
  trades (3 stop-outs here) — dropping the win-rate to 50% on 8 trades — even
  though the strategy's dollars are fine.
* But those stop-outs are **cheap and often helpful on a 1× ETF**: in Dec-2022
  the stop cut the position at −5.4% and **re-entered ~5% lower**, capturing
  +33.3% vs the no-stop trade's +25.3% (which sat through the dip first). Net,
  the stop *adds* return on 1× SOXX.
* No-stop trades less often (5 vs 8) and each round-trip is longer, so a larger
  share close green (80%) — but its single un-clipped 2022 loss (−13.0%) is why
  its drawdown is deeper and its return lower.

**This is the mirror image of SOXL.** On the 3× ETF a −5% stop was *too tight* —
routine 3× wobble tripped it into 18 trades at a 22% win-rate, so removing it
was a large win. On 1× SOXX the same −5% stop is **well-matched**: it clips the
2022 bear cheaply and never whipsaws, so removing it only trades away a little
return and drawdown protection for a prettier win-rate.

---

## Part 2 — Overall combined portfolio: negligible

Same universe (17 instruments), the **identical** `optimize_weights()` the app
and `build_overall.py` use, SOXX-stop the only change:

| Profile | Stop | Return | Max DD | Sharpe | SOXX wt | SOXL wt |
|---|---|---|---|---|---|---|
| Balanced | −5% | +736.6% | −5.2% | 3.31 | 2.2% | 2.9% |
| Balanced | none | +780.7% | −5.2% | 3.25 | 2.4% | 2.6% |
| Growth | −5% | +1875.8% | −18.6% | 2.02 | 2.2% | 18.0% |
| Growth | none | +1819.9% | −17.7% | 2.04 | 2.1% | 17.5% |
| Aggressive | −5% | +2353.5% | −26.6% | 1.58 | 2.6% | 31.3% |
| Aggressive | none | +2458.7% | −25.0% | 1.65 | 7.1% | 30.0% |

**Equal-weight blend (deterministic — clean attributable signal, no optimiser /
fundamental-tilt noise):**

| Stop | Return | Max DD | Sharpe |
|---|---|---|---|
| −5% | +771.0% | −7.1% | 2.783 |
| none | +768.4% | −7.1% | 2.779 |
| **Δ** | **−2.6pp** | **±0.0pp** | **−0.004** |

The clean equal-weight signal is **≈0** (−2.6pp on a +771% base; Sharpe −0.004).
The larger per-profile swings (+44pp Balanced, −56pp Growth, +105pp Aggressive)
are **optimiser weight-reshuffle noise, not a real edge**: SOXX earns only a
~2–7% weight in the blend because **SOXL already carries the semis beta**, so its
stop barely moves the book, and the Aggressive +105pp rides a +4.5pp SOXX
weight-jump that is inside search noise on a tiny 1× sleeve — the same
low-weight noise the earlier eval docs flag. The trustworthy, deterministic read
is **no meaningful change**.

---

## Verdict

**No — keep SOXX's −5% stop.** Removing it lifts the win-rate (50%→80% OOS) —
the cosmetic that made SOXX look "worse" than SOXL — but that is a pure
stop-accounting artifact, not a performance gain:

* **SOXX sleeve:** return *falls* (+451.7%→+431.1% OOS), drawdown *deepens*
  (−26.7%→−29.4%), Sharpe is flat-to-worse (1.22→1.20). The −5% stop is genuinely
  well-suited to 1× SOXX (it clips the 2022 bear cheaply and re-enters lower).
* **Overall blend:** deterministic impact ≈0 (−2.6pp return, −0.004 Sharpe),
  because SOXX is only a ~2% sleeve behind SOXL.

The SOXX↔SOXL win-rate gap is explained by **leverage, not signal quality**: a
−5% stop is right-sized for 1× and far too tight for 3×. That is exactly why the
repo already applies the no-stop override to SOXL *only* (`stop_by_asset`) and
leaves 1× SOXX on its tuned −5% stop. **No config change is warranted.**
