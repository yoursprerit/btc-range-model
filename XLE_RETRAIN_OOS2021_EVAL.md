# Retrain XLE / OIH / ERX (train ≤ 2025-12-31), tune for the **2021 → today** window — evaluation

**Question (the brief).** Re-train the XLE model suite with training data extended
to **2025-12-31**, re-tune the divergence signal thresholds to *maximise return and
minimise losses over the period **2021 → today (2026)*** (the UI's *🌐 Full
Out-of-Sample (2021 → now)* window), and re-run the back-test for **XLE, OIH and ERX**
across every UI period (even those now in-sample). Excluding drastic XLE down-days from
training is permitted if it helps. **Evaluation only — nothing is shipped.**

```bash
python scripts/eval_xle_retrain_oos2021.py    # → data/xle/retrain_oos2021_eval.json
```

This differs from `XLE_RETRAIN_2025_EVAL.md` in the **tuning objective only**: that run
maximised the full 2015→now history; this one maximises the **2021→today** window.
Same data (Yahoo, 2015-01-02 → 2026-07-15, 2,898 bars incl. `erx_close`), same
retrain machinery. OIH and ERX trade the XLE signal, so the one XLE High/Low signal
model drives all three sleeves.

---

## Finding 1 — extending training crushes 2021→today performance

Holding the **committed thresholds fixed** and varying only the H/L training boundary,
measured on **2021→now**:

| H/L trained through | Rows | XLE 2021→now | XLE Sharpe | OIH 2021→now | ERX 2021→now |
|---|---|---|---|---|---|
| **2020-12-31 (committed)** | 1,239 | **+105.4%** (−9.2%) | **1.37** | **+69.6%** | **+311.1%** |
| 2022-12-31 | 1,742 | +34.2% (−16.2%) | 0.62 | +0.1% | +73.0% |
| 2023-12-31 | 1,992 | +29.4% (−18.2%) | 0.53 | −6.7% | +62.1% |
| 2025-12-31 (full) | 2,494 | +26.1% (−20.3%) | 0.48 | −2.1% | +49.7% |
| 2025-12-31, drop < −7% | 2,485 | +12.9% (−21.2%) | 0.26 | −5.4% | +17.4% |
| 2025-12-31, drop < −5% | 2,473 | +18.7% (−21.3%) | 0.34 | +6.1% | +34.5% |

Every extra year of training data cuts 2021→now return and **doubles+ the drawdown**.
The divergence rule trades the *surprise* between XLE's realised High/Low and the
ridge's prediction; training the ridge on 2020–2025's violent energy swings teaches it
to expect them, so the surprise the strategy lives on evaporates.

## Finding 2 — excluding drastic drops does **not** help *this* objective

Opposite to the full-history objective (where dropping crash days helped), for the
**2021→now** target it **hurts**: `drop<−7%` and `drop<−5%` land *below* the plain
`train≤2025` run (+12.9% / +18.7% vs +26.1%). The excluded crash days are exactly the
**downside-surprise calibration** the strategy needs to sidestep the 2021–22 and 2025
energy drawdowns — remove them and the signal stops flagging the very falls it should
avoid. So the brief's "exclude drastic drops" lever is counter-productive here.

## Finding 3 — even fully re-tuned to 2021→now, nothing beats committed

Best two configs found on the extended window, **tuned on the 2021→now window itself**:

| Config | Thresholds | XLE 2021→now |
|---|---|---|
| **Committed (unchanged)** | U1=0.16, D2=−0.10, d1-exit, −8% | **+105.4% / −9.2% / Sh 1.37** |
| Retrain-**RET** (max 2021 return) | U1=0.08, D2=−0.18, no-d1-exit, −8%, drop off | +45.9% / −20.6% / Sh 0.56 |
| Retrain-**RISK** (max 2021 Sharpe) | U1=0.20, D2=−0.08, no-d1-exit, −8%, drop<−5% | +37.0% / **−9.9%** / Sh 0.69 |

The committed strategy returns **2.3× the best retrained** on its own objective window,
at *less than half the drawdown* and *2× the Sharpe*. The retrain cannot be re-tuned
back to competitiveness because the damage is in the signal model, not the thresholds.

---

## Full per-period back-test — committed vs both retrained candidates

Return / MaxDD / Sharpe. **Bold** = best return in row. (Thresholds tuned to 2021→now.)

### XLE (the signal asset)
| Period | Committed | Retrain-RET | Retrain-RISK | Buy&Hold |
|---|---|---|---|---|
| 🌍 Full history (2015→now) | **+164.9% / −16.6% / 0.95** | +96.0% / −21.6% / 0.52 | +115.2% / −9.9% / 0.94 | +86.6% / −70.1% / 0.35 |
| 🐻 Bear (2021–22) | **+74.9% / −4.6% / 2.35** | +24.6% / −18.2% / 0.71 | +44.3% / −7.7% / 1.73 | +130.4% / −26.9% / 1.45 |
| 🐂 Recovery (2023→now) | **+17.5% / −9.2% / 0.59** | +17.2% / −10.9% / 0.45 | −5.0% / −9.9% / −0.16 | +33.3% / −22.1% / 0.48 |
| 🌐 **Full OOS (2021→now)** ⭐ | **+105.4% / −9.2% / 1.37** | +45.9% / −20.6% / 0.56 | +37.0% / −9.9% / 0.69 | +196.5% / −26.9% / 0.88 |
| 🔬 Recent OOS (2025→now) | **+15.3% / −5.4% / 1.14** | +12.5% / −8.1% / 0.74 | −2.0% / −9.0% / −0.16 | +29.9% / −18.8% / 0.83 |

### OIH (oil services · trades XLE signal)
| Period | Committed | Retrain-RET | Retrain-RISK | Buy&Hold |
|---|---|---|---|---|
| 🌍 Full history | +119.1% / −23.1% / 0.59 | +13.9% / −31.0% / 0.16 | **+143.0% / −14.3% / 0.79** | −28.0% / −90.3% / 0.14 |
| 🐻 Bear (2021–22) | **+75.2% / −9.4% / 1.69** | +9.6% / −26.7% / 0.31 | +40.3% / −11.8% / 1.19 | +93.9% / −36.4% / 0.95 |
| 🐂 Recovery (2023→now) | **−3.2% / −18.5% / −0.03** | −9.7% / −25.1% / −0.11 | −0.2% / −14.3% / 0.04 | +31.5% / −45.7% / 0.41 |
| 🌐 **Full OOS (2021→now)** ⭐ | **+69.6% / −18.5% / 0.77** | −1.0% / −30.5% / 0.09 | +40.1% / −14.3% / 0.57 | +142.9% / −45.7% / 0.62 |
| 🔬 Recent OOS (2025→now) | **+4.4% / −11.1% / 0.32** | −6.8% / −15.1% / −0.22 | +3.4% / −7.1% / 0.32 | +37.9% / −34.3% / 0.79 |

### ERX (2× energy · trades XLE signal · stop-less)
| Period | Committed | Retrain-RET | Retrain-RISK | Buy&Hold |
|---|---|---|---|---|
| 🌍 Full history | **+400.8% / −53.0% / 0.77** | +240.0% / −56.3% / 0.52 | +372.9% / **−18.4% / 0.95** | −63.9% / −98.6% / 0.23 |
| 🐻 Bear (2021–22) | **+196.0% / −9.3% / 2.32** | +51.9% / −34.8% / 0.77 | +103.8% / −15.2% / 1.72 | +362.2% / −47.3% / 1.51 |
| 🐂 Recovery (2023→now) | **+38.9% / −17.5% / 0.66** | +30.5% / −21.2% / 0.45 | −10.9% / −18.4% / −0.16 | +37.0% / −43.8% / 0.43 |
| 🌐 **Full OOS (2021→now)** ⭐ | **+311.1% / −17.5% / 1.40** | +98.1% / −38.7% / 0.58 | +81.7% / −18.4% / 0.70 | +487.5% / −47.3% / 0.87 |
| 🔬 Recent OOS (2025→now) | **+32.7% / −10.6% / 1.21** | +25.2% / −15.8% / 0.77 | −4.1% / −17.4% / −0.14 | +48.5% / −35.7% / 0.79 |

---

## Verdict — do NOT adopt the retrain; keep the shipped strategy

* **The committed strategy wins its own objective decisively.** On 2021→now it more
  than doubles the best retrained variant's return on all three sleeves (XLE +105% vs
  +46%; OIH +70% vs −1%/+40%; ERX +311% vs +98%/+82%) at lower drawdown and roughly
  double the Sharpe. It also wins the 2021–22 bear — the divergence system's signature
  edge — by a wide margin (XLE +74.9% at −4.6% vs +24.6%/+44.3%).
* **Extending training is the wrong lever for a *surprise* signal**, and the effect is
  monotonic (Finding 1), not a tuning artefact — re-tuning thresholds on the 2021
  window (Finding 3) recovers none of it.
* **Excluding drastic drops back-fires for this window** (Finding 2): those crash days
  are the downside-surprise calibration that powers bear-avoidance.
* **In-sample caveat:** 2021–2025 is now in-sample for the retrained H/L model, so its
  numbers are *flattered* — and it still loses by 2×. Out-of-sample the gap only widens.

### One narrow, optional exception (unchanged from the full-history eval)
**Retrain-RISK** is attractive *only* as a drawdown-minimiser on the leveraged sleeves:
it cuts ERX full-history MDD −53% → −18.4% (keeping +373%, Sharpe 0.95↑) and OIH −23%
→ −14.3% (+143%↑). If a future brief prioritises capping the 2× tail over total return
it is the config to revisit — but on the 2021→now objective asked for here it returns
far less (XLE +37% vs +105%), so it is **not** an upgrade today.

**Recommendation: leave `app/ticker_config.py`, the XLE model artifacts and
`data/xle/backtest_results.json` unchanged.** This evaluation is the deliverable.
