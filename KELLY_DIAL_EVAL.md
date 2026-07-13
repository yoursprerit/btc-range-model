# Fractional Kelly as the Risk-Profile Dial — Evaluation (and Refutation)

**Document type:** Read-only evaluation. **No strategy, profile, cap, weight, or model is
changed.** This tests one specific proposal and reports the result — including where it
fails.

**Reproduce:** `python scripts/eval_kelly_dial.py`
**Data:** live universe, 17 instruments, 2021-01-05 → 2026-07-10 (OOS), fundamental tilt
OFF unless noted. Sanity anchor: tilt-ON default reproduces the committed Aggressive
headline (80.9% CAGR / −24.12% MaxDD / Sharpe 1.80).

## The proposal under test

Replace the three hand-tuned risk profiles — which today vary **per-instrument caps +
objective + drawdown budget** — with **one growth-optimal (full-Kelly) blend** that is
simply *deployed at fraction f of the bankroll*, the rest parked in SATA cash. `f` becomes
the single risk dial: **Aggressive = 1× · Growth = ½× · Balanced = ¼×**. The earlier
`KELLY_CRITERION_EVAL.md` suggested this "Pareto-improves risk-adjusted return."

**This evaluation tests that claim properly — at matched risk — and it does NOT hold.**

The base growth-optimal blend (Kelly objective at the widest/Aggressive caps) is, as
established previously, identical to the current Aggressive profile:
**+2575% / 81.6% CAGR / −24.27% MaxDD / Sharpe 1.79 / Calmar 3.36.**

---

## 1. The decisive test — matched drawdown

The only fair way to ask "does the dial improve on the existing profile" is to dial `f` down
until the fractional-Kelly **drawdown equals** the existing profile's, then compare return
and Sharpe. (Comparing arbitrary f-tiers to arbitrary profiles — as the earlier note did —
just compares different risk points.)

| Tier | Target MaxDD | dial `f` | Existing CAGR / Sharpe | Dial CAGR / Sharpe | Winner |
|---|---:|---:|---|---|---|
| **Balanced** | −8.81% | 0.367 | **48.2% / 3.32** | 39.2% / 2.53 | **existing** (+9.0pp, +0.79 Sh) |
| **Growth** | −17.57% | 0.715 | **74.1% / 2.17** | 62.0% / 1.96 | **existing** (+12.1pp, +0.21 Sh) |
| **Aggressive** | −24.27% | 1.000 | 81.6% / 1.79 | 81.6% / 1.79 | tie (dial *is* the base) |

**At equal drawdown the existing profiles win at every tier** — more return *and* higher
Sharpe. Matched on **volatility** instead of drawdown gives the same verdict (Balanced:
48.2% vs 36.8%; Growth: 74.1% vs 63.7% — existing wins both).

**Why the dial loses:** the existing lower-tier profiles don't merely dilute the aggressive
book with cash — they **re-optimise composition under tighter caps**. Balanced in particular
runs the near-max-Sharpe objective with tight caps and achieves **Sharpe 3.32**, whereas the
Aggressive base blend is only Sharpe 1.79. Uniformly scaling a 1.79-Sharpe blend with cash
cannot reach a genuinely re-optimised 3.32-Sharpe blend at the same risk. The hand-tuned
profiles are doing something the single-blend dial structurally cannot.

## 2. The literal 1× / ½× / ¼× scheme

Comparing the proposal's own tiers to the matched existing profile:

| Tier | Existing (CAGR / MaxDD / Sharpe / Calmar) | Dial (CAGR / MaxDD / Sharpe / Calmar) | ΔCAGR |
|---|---|---|---:|
| Aggressive (1×) | 81.6% / −24.3% / 1.79 / 3.36 | 81.6% / −24.3% / 1.79 / 3.36 | +0.0pp |
| Growth (½×) | 74.1% / −17.6% / 2.17 / 4.22 | 47.7% / −12.2% / 2.22 / 3.90 | **−26.3pp** |
| Balanced (¼×) | 48.2% / −8.8% / 3.32 / 5.47 | 31.9% / −5.7% / 3.07 / 5.57 | **−16.3pp** |

The dial tiers land at **lower risk *and* much lower return** than the existing same-name
profiles. The ½× and ¼× dial points give up 16–26pp of CAGR; their marginally different
Sharpe/Calmar is just because they sit at a shallower risk point, not because they are more
efficient — §1 already showed that at *equal* risk they are strictly worse.

## 3. The fractional-Kelly frontier (for reference)

The dial does produce a clean, monotonic risk curve — Sharpe and Calmar rise smoothly as `f`
falls. That is its one genuine merit (a single, theory-grounded knob):

| f | CAGR | MaxDD | Sharpe | vol | Calmar |
|---:|---:|---:|---:|---:|---:|
| 1.00 | 81.6% | −24.3% | 1.79 | 30.7% | 3.36 |
| 0.70 | 61.0% | −17.2% | 1.97 | 21.5% | 3.54 |
| 0.55 | 51.0% | −13.5% | 2.14 | 16.9% | 3.78 |
| 0.35 | 38.1% | −8.4% | 2.58 | 10.7% | 4.55 |
| 0.25 | 31.9% | −5.7% | 3.07 | 7.7% | 5.57 |
| 0.15 | 25.8% | −3.1% | 4.21 | 4.6% | 8.25 |
| 0.10 | 22.8% | −1.9% | 5.64 | 3.1% | 11.86 |

**Caution:** the flattering low-`f` rows (Sharpe 4–6, Calmar 8–12) are **mostly the SATA
idle-cash assumption**, not strategy skill — at f = 0.10 the book is ~90% cash compounding
the counterfactual ~13% coupon. Those numbers are an artifact of that assumption, not a
reason to run a near-cash portfolio.

## 4. Period robustness — matched-DD dial vs existing

At drawdown-matched `f`, the existing profile out-returns the dial in **every** sub-period,
including the bear market — where the fixed aggressive composition actually draws down
*more* than its full-period match:

| | Balanced existing | Balanced dial (f=0.37) |
|---|---|---|
| 🌐 Full OOS | +772% / −8.8% / Sh 3.32 | +518% / −8.8% / Sh 2.53 |
| 🐻 Bear 21–22 | +93% / −5.1% / Sh 3.37 | +44% / **−8.5%** / Sh 1.53 |
| 🐂 Bull 23→ | +347% / −8.8% / Sh 3.28 | +328% / −8.8% / Sh 3.05 |
| 🔬 Recent 25→ | +150% / −4.7% / Sh 4.39 | +121% / −5.0% / Sh 3.62 |

Growth tier shows the same pattern (Full OOS existing +2018% vs dial +1325%). The dial is not
merely lower-return in aggregate — it is **less robust in the bear regime** at the tamer
tiers, because one static composition can't adapt the way per-tier re-optimisation does.

---

## Findings

1. **The "fractional-Kelly-as-dial Pareto-improves" claim is false.** At matched drawdown
   *and* matched volatility, the existing hand-tuned profiles deliver **more return and
   higher Sharpe** at every tier (Balanced +9pp CAGR / +0.79 Sharpe; Growth +12pp / +0.21).
2. **The earlier note compared unmatched risk points** ("half-Kelly beats full Aggressive on
   Sharpe & drawdown") — true but trivial (less risk ⇒ better ratios). Corrected here.
3. **The existing profiles' cap-varying design is the source of their edge** — especially
   Balanced's tight-cap, near-max-Sharpe composition (Sharpe 3.32), which a scaled copy of
   the Sharpe-1.79 aggressive blend cannot match.
4. **The dial's genuine merits are ergonomic, not performance:** one monotonic, theory-
   grounded knob producing smooth intermediate risk points. Its flattering low-`f` ratios are
   largely the SATA idle-cash assumption.

## Recommendation

**Do not replace the risk profiles with a fractional-Kelly dial.** It costs 9–26pp of CAGR at
matched risk and is less bear-robust. If a *simpler* profile mechanism is ever wanted, a
fractional dial is a defensible ergonomic choice — but it should be built on **per-tier
re-optimised blends** (dial the composition too), not a single scaled aggressive book, and it
must not be sold as a performance improvement. **Not implemented — evaluation only, per the
request.**

## Caveats

Favourable 2021–2026 bull cycle; gross of costs; SATA ~13% idle-cash and 2×/3× daily-close
leverage assumptions (which specifically inflate the low-`f` dial rows); Kelly estimates
μ/σ from a single regime. The matched-risk conclusion (existing > dial) is, if anything,
*strengthened* by removing the SATA tailwind, since low-`f` dial points lean hardest on it.

## Bottom line

Exposing fractional Kelly as the full/½/¼ risk dial is an **ergonomic simplification, not a
performance upgrade** — and on this data it is a measurable **downgrade** (−9 to −26pp CAGR
at matched drawdown, lower Sharpe, weaker bear behaviour). The existing profiles' practice of
re-optimising composition under tighter caps beats uniformly diluting one aggressive blend
with cash. The one worthwhile Kelly contribution remains what the first evaluation found — a
`kelly` objective that, harmlessly, reproduces the current `max_return` result — and nothing
here warrants changing the live strategy.
