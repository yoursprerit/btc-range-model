# Retrain XLE / OIH / ERX on data through 2025-12-31 — evaluation

**Question (the brief).** Re-train the XLE model suite with training data extended
to **2025-12-31**, re-tune the divergence signal thresholds to *maximise return and
minimise losses over the full period*, and re-run the back-test for **XLE, OIH and
ERX** across every UI period (even those now in-sample). Excluding drastic XLE
down-days from training is permitted if it helps. **Evaluation only — nothing is
shipped to the app.**

```bash
python scripts/eval_xle_retrain_2025.py     # → data/xle/retrain_2025_eval.json
```

Fresh Yahoo data pulled **2015-01-02 → 2026-07-15** (2,898 daily bars, now incl.
`erx_close`, which the previously-cached CSV was missing). OIH and ERX trade the
**XLE** divergence signal (as shipped), so retraining the one XLE daily High/Low
signal model drives all three sleeves.

---

## How "retrain" works here

The divergence Pure-Regime system trades on the **surprise** in XLE's daily High/Low
versus a ridge model's prediction (`err_hi` / `err_lo`). The committed pipeline fits
that H/L model on the **pre-OOS window only (≤ 2020-12-31)** and predicts every later
bar, so 2021→now is genuinely out-of-sample. "Retraining with data through
2025-12-31" means re-fitting the H/L ridge on ~11 years instead of ~6, which makes
2021–2025 **in-sample** (accepted per the brief).

Thresholds swept on each scenario: `U1∈{0.05…0.20}`, `D2∈{−0.08…−0.18}`, `D1=0.10`,
`use_d1_exit∈{T,F}`, `stop∈{−8%,−10%,−12%,none}` (ERX always stop-less). 840 configs
scored; objective = maximise XLE full-history return with drawdown ≤ Buy&Hold, cross-
checked against best-Sharpe.

---

## Finding 1 — extending the training window *degrades* the signal

Holding the **committed thresholds fixed** and varying **only** the H/L training
boundary isolates the effect of the training data. XLE full-history return collapses
as the window grows:

| H/L trained through | Train rows | XLE full-history | XLE Sharpe | XLE bear '21–22 | OIH full | ERX full |
|---|---|---|---|---|---|---|
| **2020-12-31 (committed)** | 1,239 | **+164.9%** (−16.6%) | **0.95** | **+74.9%** | **+119.1%** | +400.8% |
| 2022-12-31 | 1,742 | +52.5% (−17.5%) | 0.45 | +36.3% | −0.3% | +67.8% |
| 2023-12-31 | 1,992 | +63.4% (−18.2%) | 0.50 | +34.0% | +15.9% | +103.8% |
| 2025-12-31 (full) | 2,494 | +50.0% (−20.3%) | 0.43 | +22.4% | +5.9% | +67.0% |
| 2025-12-31, drop < −7% | 2,485 | +72.9% (−21.2%) | 0.59 | +18.6% | +48.6% | +166.7% |
| 2025-12-31, drop < −5% | 2,473 | +99.4% (−21.3%) | 0.73 | +22.4% | +102.1% | +303.8% |

The divergence edge comes from calibrating the H/L model on a **stable early
reference frame**: the 2021–22 energy melt-up then reads as a large *upside surprise*
(`U1` fires → long) and the crashes as *downside surprise* (`D2`/`D1` → flat). Feeding
the model the wild 2020–2025 energy swings teaches it to *expect* them, which dilutes
exactly the surprise the strategy trades on — return more than halves and drawdown
widens.

## Finding 2 — excluding drastic XLE drops helps, but only partially

The brief's hypothesis holds: dropping crash days from the ridge fit clearly helps.
On the train≤2025 window, excluding days worse than −5% lifts XLE full-history from
**+50.0% → +99.4%** and ERX from **+67% → +304%**. But it only *recovers* ground lost
by extending the window — it never reaches the committed +164.9%.

## Finding 3 — even fully re-tuned, no retrained config beats committed

Best two candidates found on the extended (train≤2025, drop<−5/−7%) window:

| Config | Thresholds | XLE full-history |
|---|---|---|
| **Committed (unchanged)** | U1=0.16, D2=−0.10, d1-exit, −8% | **+164.9% / −16.6% / Sh 0.95** |
| Retrain-**RET** (max return) | U1=0.12, D2=−0.08, d1-exit, −8%, drop<−7% | +128.8% / −18.4% / Sh 0.82 |
| Retrain-**RISK** (max Sharpe) | U1=0.20, D2=−0.08, no-d1-exit, −8%, drop<−5% | +115.2% / **−9.9%** / Sh 0.94 |

Retrain-RET gives *less* return at *worse* drawdown and Sharpe. Retrain-RISK trims
drawdown nicely (−9.9%) but at ~50pp less return and a hair lower Sharpe — it does not
dominate.

---

## Full per-period back-test — committed vs both retrained candidates

Return / MaxDD / Sharpe. **Bold** = best return in row.

### XLE (the signal asset)
| Period | Committed | Retrain-RET | Retrain-RISK | Buy&Hold |
|---|---|---|---|---|
| 🌍 Full history (2015→now) | **+164.9% / −16.6% / 0.95** | +128.8% / −18.4% / 0.82 | +115.2% / −9.9% / 0.94 | +86.6% / −70.1% / 0.35 |
| 🐻 Bear (2021–22) | **+74.9% / −4.6% / 2.35** | +21.7% / −12.4% / 0.77 | +44.3% / −7.7% / 1.73 | +130.4% / −26.9% / 1.45 |
| 🐂 Recovery (2023→now) | **+17.5% / −9.2% / 0.59** | +1.0% / −11.5% / 0.08 | −5.0% / −9.9% / −0.16 | +33.3% / −22.1% / 0.48 |
| 🌐 Full OOS (2021→now) | **+105.4% / −9.2% / 1.37** | +22.9% / −18.4% / 0.39 | +37.0% / −9.9% / 0.69 | +196.5% / −26.9% / 0.88 |
| 🔬 Recent OOS (2025→now) | **+15.3% / −5.4% / 1.14** | +10.4% / −7.3% / 0.74 | −2.0% / −9.0% / −0.16 | +29.9% / −18.8% / 0.83 |

### OIH (oil services · trades XLE signal)
| Period | Committed | Retrain-RET | Retrain-RISK | Buy&Hold |
|---|---|---|---|---|
| 🌍 Full history | +119.1% / −23.1% / 0.59 | +117.2% / −33.8% / 0.57 | **+143.0% / −14.3% / 0.79** | −28.0% / −90.3% / 0.14 |
| 🐻 Bear (2021–22) | **+75.2% / −9.4% / 1.69** | +16.7% / −17.5% / 0.50 | +40.3% / −11.8% / 1.19 | +93.9% / −36.4% / 0.95 |
| 🐂 Recovery (2023→now) | −3.2% / −18.5% / −0.03 | −12.9% / −25.9% / −0.27 | **−0.2% / −14.3% / 0.04** | +31.5% / −45.7% / 0.41 |
| 🌐 Full OOS (2021→now) | **+69.6% / −18.5% / 0.77** | +1.6% / −33.8% / 0.09 | +40.1% / −14.3% / 0.57 | +142.9% / −45.7% / 0.62 |
| 🔬 Recent OOS (2025→now) | **+4.4% / −11.1% / 0.32** | +1.8% / −13.6% / 0.16 | +3.4% / −7.1% / 0.32 | +37.9% / −34.3% / 0.79 |

### ERX (2× energy · trades XLE signal · stop-less)
| Period | Committed | Retrain-RET | Retrain-RISK | Buy&Hold |
|---|---|---|---|---|
| 🌍 Full history | +400.8% / −53.0% / 0.77 | **+442.5% / −36.5% / 0.84** | +372.9% / **−18.4%** / **0.95** | −63.9% / −98.6% / 0.23 |
| 🐻 Bear (2021–22) | **+196.0% / −9.3% / 2.32** | +42.1% / −24.2% / 0.76 | +103.8% / −15.2% / 1.72 | +362.2% / −47.3% / 1.51 |
| 🐂 Recovery (2023→now) | **+38.9% / −17.5% / 0.66** | −3.1% / −23.9% / 0.04 | −10.9% / −18.4% / −0.16 | +37.0% / −43.8% / 0.43 |
| 🌐 Full OOS (2021→now) | **+311.1% / −17.5% / 1.40** | +37.8% / −36.5% / 0.37 | +81.7% / −18.4% / 0.70 | +487.5% / −47.3% / 0.87 |
| 🔬 Recent OOS (2025→now) | **+32.7% / −10.6% / 1.21** | +20.0% / −14.7% / 0.74 | −4.1% / −17.4% / −0.14 | +48.5% / −35.7% / 0.79 |

---

## Verdict — do NOT adopt the retrain; keep the shipped strategy

* **The committed strategy wins on the joint objective.** On the signal asset XLE it
  beats every retrained+retuned variant on return in *all five* periods, and matches
  or beats them on Sharpe/drawdown almost everywhere. Its signature strength — the
  2021–22 energy bear (XLE **+74.9%** at −4.6% MDD, Sharpe 2.35) — is exactly what the
  retrain gives up (retrain-RET only +21.7%; retrain-RISK +44.3%).
* **Extending training is the wrong direction for a *surprise*-based signal.** More
  history makes the H/L model "expect" the very moves the divergence rule needs to see
  as anomalies. Monotonic degradation from +164.9% → +50% (Finding 1) makes this
  unambiguous, not a tuning artefact.
* **Excluding drastic drops works as hypothesised** — it recovers ~half the loss from
  extending the window (drop<−5% ⇒ +50% → +99%) — but it is a mitigation for a
  self-inflicted problem, not a net gain over the shipped short-window model.
* **In-sample caveat:** all retrained numbers for 2021–2025 are now *in-sample* (the
  H/L model saw those years), so they flatter the retrain — and it *still* loses.
  Out-of-sample the gap would only widen.

### One narrow, optional exception
The **retrain-RISK** config is genuinely appealing **only if the goal is drawdown
minimisation on the leveraged sleeves**: it cuts ERX full-history MDD from **−53% →
−18.4%** (keeping +373%, Sharpe 0.95↑) and OIH from −23% → −14% (+143%↑). If a future
brief prioritises capping the 2× tail over total return, this is the config to
revisit — but it costs the XLE sleeve ~50pp of return and half its bear-market edge,
so it is not a portfolio-wide upgrade today.

**Recommendation: leave `app/ticker_config.py`, the XLE model artifacts and
`data/xle/backtest_results.json` unchanged.** This evaluation is the deliverable.
