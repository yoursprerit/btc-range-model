# Win-rate-gated instrument selection for the Overall strategy — evaluation

**Question.** Should the Overall universe only include instruments whose
trade **win rate is ≥ 55 %** — and if so, should the gate be measured on the
**full backtest period** (each instrument's own history, 2016→now for the
standard tickers) or on the **OOS windows** (2021→now, plus the Bear / Bull /
Recent sub-windows)? Does either criterion **maximize returns across all
three risk profiles** (Balanced / Growth / Aggressive)?

**Not implemented.** This is an evaluation only — no config, universe or
allocation was changed. Reproduce with:

```bash
python scripts/eval_winrate_selection.py
```

## Method

* Runs the live universe **once** (the same 18 streams the Overall engine
  trades), then computes each instrument's **trade-level win rate** over
  * its **full backtestable period** (a re-run of its exact engine from its
    first valid bar — e.g. 2016→now for XLE/SOXX/GRID/REMX, 2018→now for the
    gold stack; the BTC stack's CT features begin 2024, so its full period
    *is* its OOS period), and
  * the **OOS windows**: full OOS 2021→now (the `win_rate` the app itself
    surfaces) and the Bear 2021-22 / Bull 2023→ / Recent 2025→ sub-windows
    (closed trades, assigned by exit date).
* Builds four candidate universes — **ALL-18** (baseline), **A**
  full-period WR ≥ 55 %, **B** OOS WR ≥ 55 %, **C** WR ≥ 55 % in *every* OOS
  window with trades — and re-optimises each for **every risk profile** with
  the **identical** `optimize_weights()` call `scripts/build_overall.py`
  uses (same per-kind caps, SATA idle-cash yield, drawdown floor,
  objective). `fundamental=False` everywhere so the selection criterion, not
  the conviction tilt, drives the comparison; each result is checked across
  **three MC seeds** (7 / 11 / 23).

Window: full OOS **2021-01-04 → 2026-07-24** (staggered inception as in the
app). Combined-curve returns are total return over that window.

---

## 1. The win rates (as of 2026-07-24)

| Instrument | Kind | Full period | Full WR | (n) | OOS WR | (n) | Bear | Bull | Recent |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| BTC  | core | 2024→ (=OOS) | 62.5 | 8 | **62.5** | 8 | — | 62.5 | 80.0 |
| MSTR | beta | 2024→ (=OOS) | 75.0 | 8 | **75.0** | 8 | — | 75.0 | 80.0 |
| MSTU | lev  | 2024→ (=OOS) | 75.0 | 8 | **75.0** | 8 | — | 75.0 | 80.0 |
| ETH  | core | 2024→ (=OOS) | 62.5 | 8 | **62.5** | 8 | — | 62.5 | 80.0 |
| GLDM | core | 2018→ | **69.2** | 13 | **66.7** | 9 | 57.1 | 100 | 100 |
| UGL  | lev  | 2018→ | 52.9 | 17 | 54.5 | 11 | 50.0 | 66.7 | 100 |
| GDX  | beta | 2018→ | 54.3 | 138 | 53.1 | 96 | 40.0 | 57.7 | 55.9 |
| NUGT | lev  | 2018→ | 52.1 | 140 | 52.1 | 96 | 40.0 | 56.3 | 52.9 |
| SOXX | core | 2016→ | 46.7 | 15 | 50.0 | 8 | 33.3 | 100 | — |
| SOXL | lev  | 2016→ | **60.0** | 10 | **80.0** | 5 | 66.7 | 100 | — |
| GRID | core | 2016→ | 52.3 | 109 | 54.4 | 57 | 50.0 | 56.8 | 52.9 |
| XLE  | core | 2016→ | 53.1 | 49 | **80.0** | 10 | 80.0 | — | — |
| OIH  | beta | 2016→ | **57.1** | 49 | **80.0** | 10 | 80.0 | — | — |
| ERX  | lev  | 2016→ | 53.1 | 49 | **80.0** | 10 | 80.0 | — | — |
| REMX | core | 2016→ | 50.0 | 6 | 33.3 | 3 | 100 | 0.0 | — |
| WGMI | beta | 2022→ | 54.3 | 35 | **66.7** | 18 | — | 66.7 | 80.0 |
| PBW  | core | 2011→ | 51.8 | 197 | **57.0** | 79 | 48.1 | 61.5 | 68.0 |
| ARTY | core | 2018→ | **58.4** | 125 | **59.0** | 83 | 50.0 | 62.3 | 70.0 |

Bold = clears 55 %. "—" = no closed trades in the window. The two gates
disagree in both directions: **XLE/ERX/WGMI/PBW pass OOS but fail the full
period** (the XLE crash-shield's 2016-20 whipsaws drag its full WR to 53 %
on 49 trades, while its OOS record is 8-for-10), and nothing passes the full
period without also passing OOS except by a hair (OIH 57.1 full vs 80 OOS).
The resulting universes:

| Universe | n | Members |
|---|---:|---|
| ALL-18 (baseline) | 18 | everything |
| **A** full-period WR ≥ 55 % | 8 | BTC MSTR MSTU ETH GLDM SOXL OIH ARTY |
| **B** OOS WR ≥ 55 % | 12 | A + XLE ERX WGMI PBW |
| **C** every OOS window ≥ 55 % | 10 | B − PBW ARTY (bear-window WR < 55) |

(A ∩ B = A, so a "both periods" gate is identical to A.)

---

## 2. Result — no win-rate gate maximizes returns across all profiles

### Optimal blend by universe × profile (seed 7; total ret / MaxDD / Sharpe)

| Universe | Balanced | Growth | Aggressive |
|---|---|---|---|
| **ALL-18** | **+821 % / −9.3 % / 2.57** | +1,679 % / −18.8 % / 1.83 | +2,140 % / −30.9 % / 1.50 |
| A: full ≥ 55 % | +638 % / −13.1 % / 2.20 | +1,455 % / −21.3 % / 1.65 | +2,685 % / −35.3 % / 1.48 |
| B: OOS ≥ 55 % | +719 % / **−7.7 % / 2.62** | **+1,782 % / −20.8 % / 1.86** | +2,937 % / −35.2 % / 1.48 |
| C: all windows ≥ 55 % | +780 % / −15.3 % / 2.05 | +1,592 % / −21.2 % / 1.76 | **+3,476 % / −37.2 % / 1.47** |

The pattern holds across all three MC seeds (7 / 11 / 23):

* **Balanced** — the **full 18-name universe wins**. Its return is highest in
  2 of 3 seeds and its Sharpe (2.57–2.59) is never beaten except marginally
  by B (2.50–2.62, at −80…−100 pp of return in two seeds). A and C lose
  0.3–0.5 Sharpe *and* return here in every seed.
* **Growth** — **B wins on return in every seed** (+1,650…+1,782 % vs
  baseline +1,607…+1,679 %) at essentially the baseline's Sharpe.
* **Aggressive** — **any gate beats the baseline** (+2,644…+3,519 % vs
  +2,060…+2,268 %), with C best in 2 of 3 seeds and B best in the third —
  but always at a deeper drawdown (−32…−37 % vs the baseline's −28…−37 %),
  pressing right against the −38 % floor.

So the direct answers:

1. **Full period vs OOS: the OOS gate (B) is clearly the better of the two.**
   The full-period gate (A) is *dominated in every profile and every seed* —
   it keeps only 8 names, throws away XLE/ERX/WGMI/PBW whose tuned configs
   have genuinely strong recent records, and penalises instruments for the
   pre-2021 behaviour of engines that were tuned for the OOS era.
2. **No criterion maximizes returns across all risk profiles.** B is the only
   gate that never loses Sharpe to the baseline while adding Growth and
   Aggressive return, but on **Balanced** the ungated 18-name universe still
   delivers more return at the same-or-better Sharpe in most seeds. A
   criterion that wins Aggressive (C) is the *worst* Balanced choice.

### Why the answer flips with the profile

The names every gate removes — GDX, NUGT, GRID, SOXX, UGL, REMX — are
exactly the diversifiers the **Balanced** objective (near-max-Sharpe) leans
on: the baseline Balanced optimum allocates ~**18 % GDX, 15 % PBW, 8 % UGL,
6 % SOXX**. Removing them doesn't free "better" capacity for Balanced; it
removes decorrelation. The **max-return** profiles, by contrast, are capped
long before they run out of ideas — filtering the universe concentrates the
book into SOXL/MSTU/ERX (Aggressive-C: SOXL 33 %, MSTU 31 %, ERX 13 %),
which is precisely what a max-return objective wants and a Sharpe objective
doesn't.

The deterministic equal-weight ladder shows the same thing with zero
optimizer noise — tightening the gate is a **risk dial, not a free lunch**:

| Universe | Equal-weight | Risk-parity |
|---|---|---|
| ALL-18 | +751 % / −12.3 % / **2.21** | +528 % / −10.7 % / **2.47** |
| A (8) | +823 % / −15.6 % / 1.83 | +498 % / −11.9 % / 2.21 |
| B (12) | +938 % / −20.0 % / 1.90 | +564 % / −12.5 % / 2.24 |
| C (10) | **+1,097 %** / −28.0 % / 1.63 | **+616 %** / −19.6 % / 1.71 |

Raw return rises monotonically as the gate tightens; Sharpe and drawdown
deteriorate monotonically. A win-rate gate is therefore equivalent to
turning the risk knob — something the profile caps/objectives already do
explicitly, without deleting diversifiers.

---

## 3. Why a 55 % win-rate gate is a shaky selection rule here

* **Win rate is a poor proxy for return.** These are trend/divergence
  engines with asymmetric payoffs — SOXX (46.7 % full WR) fails both gates
  while **SOXL, traded off the *same* signal, is the single biggest return
  engine in the book** (+11,280 % full-period). REMX fails OOS (1-for-3
  trades) yet its full-period strategy return is +380 %. Cutting on WR cuts
  winners whose edge is payoff, not frequency.
* **The OOS samples are tiny.** XLE/ERX's 80 % OOS WR is 8-of-10 trades —
  Wilson 95 % CI [49 %, 94 %]; SOXL's 80 % is 4-of-5 [38 %, 96 %]; WGMI's
  66.7 % is 12-of-18 [44 %, 84 %]; REMX's 33 % is 1-of-3 [6 %, 79 %]. For
  most instruments the 55 % threshold cuts *inside* the confidence interval
  — the gate is largely classifying noise.
* **The BTC stack can't be gated on a full period at all** (CT features
  begin 2024), so a "full-period" rule silently grades a quarter of the
  universe on 8 trades of OOS data anyway.
* **Optimal weights remain in-sample** (as everywhere in the app): each
  universe's blend is fit on the same history it is scored on, so the
  Aggressive gains from B/C are best-case numbers, and the sub-window win
  rates use closed trades only.

---

## 4. Verdict

* **Between the two, gate on OOS — never on the full period.** The
  full-period ≥ 55 % gate (A) is dominated everywhere: it loses ~180 pp of
  Balanced return and 0.37 Sharpe, loses in Growth, and is the weakest of
  the three gates even in Aggressive.
* **But no win-rate criterion maximizes returns across all risk profiles.**
  The honest summary per profile: **Balanced → keep all 18** (the low-WR
  names are its diversifiers); **Growth → B (OOS ≥ 55 %) adds ~+100 pp at
  par Sharpe**; **Aggressive → B/C add +800–1,300 pp but at deeper
  drawdowns, against the −38 % floor, with in-sample weights**.
* **Recommendation: don't adopt a universe-level win-rate gate.** The
  per-kind caps + profile objectives already express the same
  return-vs-risk dial explicitly, keep the Balanced profile's
  diversification intact, and don't hinge membership on 5–10-trade win-rate
  samples. If a tilt toward high-WR names is wanted, it already exists in
  the right place: `compute_priorities()` weights `win_rate` at 0.20 in the
  live entry-priority score, scaling *sizing* continuously instead of
  deleting instruments on a noisy threshold.
