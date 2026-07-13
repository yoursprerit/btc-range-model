# Kelly Criterion — Allocation Evaluation for the Overall Strategy

**Document type:** Read-only evaluation. **The live Overall strategy, its default
risk profile, the committed weights, and every model are UNCHANGED.** This adds a
growth-optimal (`kelly`) objective to the allocator so it can be *back-tested* head-to-head
against the objective the strategy currently uses, and measures a fractional-Kelly
bankroll overlay. Nothing here is wired into the app — `DEFAULT_PROFILE` is still
`Aggressive` → `max_return`.

**Reproduce:** `python scripts/eval_kelly_overall.py`

**Data:** live universe, 17 instruments, return matrix 2021-01-05 → 2026-07-11 (OOS start).
**Sanity anchor:** re-running the *current* objective with the fundamental tilt ON reproduces
the committed artifact's Aggressive headline **exactly** (80.9% CAGR / −24.12% MaxDD /
Sharpe 1.80), confirming the evaluation engine is faithful.

---

## What "Kelly" means here, and where it was wired in

The Kelly criterion sizes bets to maximise the **long-run geometric growth rate**
`E[log(1+r)]` of wealth. Two places in the Overall strategy can host it (see the prior
analysis): (1) the portfolio **allocation objective** in `optimize_weights`, and
(2) a **fractional-Kelly bankroll dial** deciding how much of the book to deploy vs park in
cash. Both are evaluated.

* **Implemented (`app/overall_core.py`):** a new `objective="kelly"` in `optimize_weights`
  that selects the long-only, capped blend maximising `E[log(1+r)]` inside the same
  drawdown budget. `curve_metrics` / `_metrics_batch` now also emit `log_growth` / `growth`.
  This is **available but unused by any profile** — purely for evaluation.
* **Modelled in the eval only:** fractional Kelly (deploy fraction *f* of the bankroll in
  the Kelly blend, park *1−f* in SATA cash) and the classic per-asset Kelly fraction
  `f* = W − (1−W)/R`.

Everything is compared on **identical** return streams, caps, drawdown budgets and the SATA
idle-cash assumption — the only thing that varies is the allocation rule.

---

## 1. Portfolio-level Kelly vs the current objective (fundamental tilt OFF)

Each profile is optimised with its *current* objective, then with `kelly`, on the same data.

| Profile (caps, DD budget) | Objective | Total ret | CAGR | MaxDD | Sharpe | Growth |
|---|---|---:|---:|---:|---:|---:|
| **Balanced** (0.30/0.18/0.10, −35%) | current `balanced` | +772.5% | 48.2% | **−8.8%** | **3.32** | 39.1% |
| | `kelly` | +1265.4% | 60.7% | −15.7% | 2.42 | 49.0% |
| **Growth** (0.30/0.25/0.18, −22%) | current `max_return` | +2019.1% | 74.0% | −17.6% | 2.17 | 59.3% |
| | `kelly` | **+2019.1%** | **74.0%** | **−17.6%** | **2.17** | **59.3%** |
| **Aggressive** (0.35/0.40/0.35, −38%) — *default* | current `max_return` | +2576.7% | 81.6% | −24.3% | 1.79 | 65.1% |
| | `kelly` | **+2576.7%** | **81.6%** | **−24.3%** | **1.79** | **65.1%** |

**Two results, both important:**

1. **For Growth and Aggressive (incl. the live default), Kelly picks the *exact same*
   blend as the current `max_return` objective** — identical to the last decimal, identical
   weights (see §4). Inside a binding drawdown budget on this strongly-trending universe, the
   return-maximising blend and the growth-maximising blend are the **same corner**. So a
   portfolio-Kelly objective adds **literally nothing** to 2 of the 3 profiles.
2. **For Balanced, full Kelly is strictly *more aggressive* than the current `balanced`
   objective** — +12.5pp CAGR but −0.90 Sharpe and nearly **2× the drawdown** (−15.7% vs
   −8.8%). That is textbook full-Kelly behaviour (growth-greedy, drawdown-heavy), and on a
   risk-adjusted basis it is **worse** than what Balanced already does.

**Verdict on the objective:** adopting full Kelly as a portfolio objective is either a
**no-op** (Growth/Aggressive) or a **risk-adjusted downgrade** (Balanced). It is not worth
switching to.

## 2. Fractional Kelly — the part that actually helps

Full Kelly is famously over-aggressive and fragile to estimation error (μ/σ here are
estimated on one bull cycle). The standard fix is **fractional Kelly**: deploy *f* of the
bankroll to the Kelly blend and hold *1−f* in cash (here SATA, ~13%/yr). Because idle cash
still earns the coupon, the trade-off is very favourable:

| Profile | Fraction | Total ret | CAGR | MaxDD | Sharpe |
|---|---|---:|---:|---:|---:|
| **Balanced** | full (1×) | +1265.4% | 60.7% | −15.7% | 2.42 |
| | half (½) | +481.7% | 37.6% | −7.6% | 3.18 |
| | quarter (¼) | +272.9% | 27.0% | **−3.3%** | **4.71** |
| **Growth** | full (1×) | +2019.1% | 74.0% | −17.6% | 2.17 |
| | half (½) | +638.0% | 43.7% | −8.6% | 2.75 |
| | quarter (¼) | +321.9% | 29.9% | −4.0% | 3.90 |
| **Aggressive** | full (1×) | +2576.7% | 81.6% | −24.3% | 1.79 |
| | half (½) | +759.0% | 47.7% | **−12.2%** | **2.22** |
| | quarter (¼) | +359.2% | 31.9% | −5.7% | 3.07 |

**This is the genuinely useful finding.** Each halving of *f* roughly **halves the
drawdown** while **raising Sharpe**, and — because parked capital earns SATA — still
compounds strongly. **Half-Kelly on Aggressive Pareto-beats full Aggressive**: higher Sharpe
(2.22 vs 1.79) *and* half the drawdown (−12.2% vs −24.3%), still +47.7% CAGR. Fractional
Kelly is a clean, theory-grounded **risk dial** — arguably a better basis for the three risk
profiles than the current hand-set drawdown budgets (full ≈ Aggressive, ½ ≈ Growth-ish,
¼ ≈ Balanced-ish).

## 3. Per-asset Kelly fractions (diagnostic)

Classic discrete Kelly per sleeve, from its own trade record (`f* = W − (1−W)/R`):

| Asset | Kind | Trades | Win rate | Payoff R | f* |
|---|---|---:|---:|---:|---:|
| SOXL | lev | 5 | 80.0% | 1.26 | 64.1% |
| MSTU | lev | 6 | 66.7% | 8.39 | 62.7% |
| MSTR | beta | 6 | 66.7% | 6.31 | 61.4% |
| WGMI | beta | 18 | 66.7% | 2.91 | 55.2% |
| BTC | core | 6 | 66.7% | 2.44 | 53.0% |
| ERX | lev | 67 | 67.2% | 1.56 | 46.1% |
| UGL | lev | 70 | 67.1% | 1.55 | 46.0% |
| XLE | core | 67 | 67.2% | 1.54 | 45.9% |
| SOXX | core | 8 | 50.0% | 6.30 | 42.1% |
| GLDM | core | 75 | 66.7% | 1.32 | 41.5% |
| NUGT | lev | 82 | 58.5% | 1.89 | 36.6% |
| GDX | beta | 80 | 58.8% | 1.76 | 35.3% |
| GRID | core | 57 | 54.4% | 2.19 | 33.5% |
| OIH | beta | 67 | 58.2% | 1.45 | 29.4% |
| ARTY | core | 91 | 53.8% | 1.65 | 25.9% |
| PBW | core | 85 | 50.6% | 1.36 | 14.2% |

**Reading it:** the ranking is sensible — high-edge / high-conviction sleeves (SOXL, the
MSTR/MSTU crypto proxies, WGMI, BTC) earn the largest fractions; weak sleeves (PBW, ARTY,
OIH) the smallest. **But the raw fractions sum to ≈ 6.5×** — wildly over-leveraged if taken
literally, and the thinnest edges (BTC/MSTR/MSTU/SOXL/SOXX) rest on **5–8 trades**, far too
few to trust a per-name Kelly bet. This is exactly why the strategy's **per-instrument caps
and a fractional multiplier are indispensable**; the per-asset Kelly numbers are useful as a
*conviction ranking*, not as literal position sizes.

## 4. Weight difference — default Aggressive, Kelly vs current (tilt OFF)

Every weight is identical (Δ = 0.0% on all 17 names), because Kelly and `max_return` select
the same candidate here:

```
BTC 6.0  MSTR 24.5  MSTU 1.3  GLDM 0.5  GDX 1.1  UGL 0.1  NUGT 18.9  SOXX 0.5
SOXL 26.6  GRID 5.6  XLE 3.4  OIH 0.7  ERX 0.5  REMX 0.7  WGMI 4.0  PBW 2.1  ARTY 3.6
```

Period breakdown (Full OOS / Bear 21–22 / Bull 23→ / Recent 25→) is likewise bit-identical
between Kelly and the current objective.

---

## Findings

1. **A portfolio-level Kelly objective does not improve the strategy.** For the default
   Aggressive profile and for Growth, it is mathematically **identical** to the current
   `max_return` objective (same weights, same curve). For Balanced it is **more aggressive
   and worse risk-adjusted** (−0.90 Sharpe, ~2× drawdown). The app's default is, in effect,
   **already sitting at the full-Kelly corner** inside its drawdown budget.
2. **Fractional Kelly is where the value is.** As a bankroll dial (deploy *f*, park *1−f* in
   SATA), it delivers materially better risk-adjusted profiles — half-Kelly on Aggressive
   beats full Aggressive on **both** Sharpe and drawdown while still compounding at ~48%.
3. **Per-asset Kelly validates the existing cap hierarchy conceptually** (edge-ranked
   sizing) but its raw fractions sum to ~6.5× and lean on tiny trade counts — unusable as
   literal sizes, confirming caps + fractional scaling are required.

## Recommendation

* **Do NOT adopt full Kelly as a portfolio objective** — it is a no-op or a downgrade.
* **If anything, expose fractional Kelly as the risk-profile mechanism** (e.g.
  Aggressive = full, Growth = ½, Balanced = ¼ of the growth-optimal blend, remainder in
  SATA). This is a theory-grounded replacement for the current hand-tuned drawdown budgets
  and Pareto-improves risk-adjusted return. **Not implemented here** — this document is
  evaluation only, per the request.

## Caveats

* All figures are over a **favourable 2021–2026 multi-asset bull cycle**, gross of costs,
  with the **SATA ~13% idle-cash** assumption and **leveraged (2×/3×) sleeves** priced on
  daily closes (intraday gap/tail risk understated). Kelly estimates μ/σ from this one
  regime, so full Kelly especially is optimistic forward.
* Kelly ≡ max_return is a **property of this dataset** (long trending sample, binding DD
  budget); on a different universe or a two-sided market they could diverge.
* The thin-trade sleeves (BTC/MSTR/MSTU/SOXL/SOXX: 5–8 trades) make both their per-asset
  Kelly fractions and their blend weights statistically fragile.

## Bottom line

Kelly's criterion, applied at the **portfolio-objective** level, tells the Overall strategy
nothing new — the default already maximises growth inside its drawdown budget, so a `kelly`
objective reproduces `max_return` exactly (and underperforms the `balanced` objective on
risk-adjusted terms). The one place Kelly **does** add value is as a **fractional bankroll
dial**: half-/quarter-Kelly overlays cut drawdowns roughly in proportion to *f* while lifting
Sharpe and still compounding strongly via the SATA yield. Recommended next step (not taken
here) is to trial fractional Kelly as the risk-profile mechanism, not full Kelly as an
allocation objective.
