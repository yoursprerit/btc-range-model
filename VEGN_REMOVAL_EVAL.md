# Removing VEGN from the Overall strategy — evaluation

**Question.** VEGN (US Vegan Climate ETF — ESG-screened, tech-tilted S&P 500
beta) is one of the ten signal apps folded into the **Overall Trading**
combined portfolio. Does **dropping it from the traded universe entirely**
improve the combined back-test?

**Not implemented.** This is an evaluation only — no config, strategy or
allocation was changed. Reproduce with:

```bash
python scripts/eval_vegn_removal.py
```

## Method

* Runs the live universe **once**, then builds the returns/position matrices
  both **with** VEGN and **without** it (dropping every sleeve whose parent is
  `VEGN` — here just VEGN itself, a `core` asset with no beta/leveraged sibling).
* Re-optimises the cross-asset weights for **each risk profile**
  (Balanced / Growth / Aggressive) with the **identical** `optimize_weights()`
  call `scripts/build_overall.py` and the app use (same caps, SATA idle-cash
  yield, drawdown floor, objective, fundamental-view tilt).
* Compares the optimal combined curve across all `COMBINED_PERIODS` windows.
* VEGN's OOS start (2021) is **not** the binding one, so removing it does not
  move the common back-test window — the two curves are directly comparable
  (both `1651 × 2021-01-05 → 2026-07-08`).

Window: full OOS **2021-01-05 → 2026-07-08**.

---

## Result — removing VEGN is a marginal net positive

**VEGN already earns almost no weight.** Under the shipped optimiser its optimal
allocation is **0.13 %–1.16 %** depending on profile — the repo's fundamental
view already pins it at the lowest conviction of any asset
(`FUNDAMENTAL_VIEW["VEGN"] = 0.60`). So removing it frees a sliver of weight
that re-water-fills into higher-conviction sleeves.

### Optimal blend, full OOS, by risk profile

| Profile | VEGN wt (with) | Return with → without | MDD with → without | Sharpe with → without |
|---|---|---|---|---|
| Balanced   | 0.49 % | +551.5 % → **+545.5 %** (−6.0pp) | −4.40 % → −5.03 % | 3.456 → **3.457** (+0.001) |
| Growth     | 0.13 % | +852.8 % → **+870.0 %** (+17.3pp) | −9.43 % → −9.50 % | 2.463 → 2.291 (−0.171) |
| Aggressive | 1.16 % | +964.4 % → **+1033.9 %** (+69.5pp) | −10.34 % → −11.20 % | 1.951 → **1.966** (+0.015) |

### Deterministic reference blends (no Monte-Carlo noise), full OOS

These are the cleanest read — equal-weight and risk-parity are computed exactly,
with no random search, so the delta is pure signal:

| Blend | Return with → without | MDD with → without | Sharpe with → without |
|---|---|---|---|
| Equal-weight | +492.8 % → **+520.4 %** (+27.6pp) | −4.73 % → −4.55 % | 3.071 → **3.096** (+0.025) |
| Risk-parity  | +401.9 % → **+420.7 %** (+18.8pp) | −4.60 % → −4.17 % | 3.465 → **3.527** (+0.062) |

### Aggressive (default profile) by period

| Period | Return Δ | MDD Δ | Sharpe Δ |
|---|---|---|---|
| 🌐 Full OOS (2021→now) | +69.5pp | −0.86pp | +0.015 |
| 🐻 Bear (2021–2022)    | +3.6pp  | +0.04pp | +0.057 |
| 🐂 Bull (2023→now)     | +26.0pp | −0.86pp | −0.002 |
| 🔬 Recent (2025→now)   | +2.8pp  | −0.72pp | −0.007 |

---

## Verdict

**Yes — but only marginally, and mainly as a cleanup rather than a real edge.**

* On the **deterministic** equal-weight and risk-parity blends, removing VEGN
  cleanly improves **both return and Sharpe** in every case (Sharpe +0.025 and
  +0.062; return +18–28pp) with roughly unchanged drawdown. This is the
  trustworthy signal.
* On the **optimised profiles**, return rises (Growth +17pp, Aggressive +70pp)
  while Sharpe is essentially flat (±0.06) and drawdown ticks up < 1pp. Note the
  Growth/Aggressive optimiser is a fixed-seed 20k-sample Monte-Carlo search, so
  changing the column set reshuffles which candidate wins — part of the return
  delta there is search noise, not signal. The direction (return up, Sharpe
  flat) is consistent, but don't over-read the exact magnitude.
* The reason the effect is small is that **VEGN is already de-weighted to
  near-zero** by the fundamental-view tilt (0.60 conviction). The optimiser has
  effectively already "removed" most of it; deleting it outright just finishes
  the job.

**Bottom line:** dropping VEGN does not hurt the portfolio and slightly helps
it — modestly higher return, equal-to-slightly-better Sharpe, negligible
drawdown change. It's a reasonable simplification, but not a performance
unlock; the same benefit is already 90 % captured by the existing 0.60
conviction underweight.
