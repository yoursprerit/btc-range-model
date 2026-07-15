# Retrain XLE / OIH / ERX on the **2020→2025** window, tune for **2021 → today** — evaluation

**Question (the brief).** Re-train the XLE model suite with **new training data from
2020 to 2025** (a recent ~6-yr window — not the committed 2015→2020 early window, and
not the full 2015→2025 history), re-tune the divergence thresholds to *maximise return
and minimise losses over **2021 → today (2026)***, and re-run the back-test for **XLE,
OIH and ERX** across every UI period (even those now in-sample). Excluding drastic XLE
down-days from training is permitted if it helps. **Evaluation only — nothing shipped.**

```bash
python scripts/eval_xle_retrain_2020_2025.py   # → data/xle/retrain_2020_2025_eval.json
```

Third variant in the retrain study (see also `XLE_RETRAIN_2025_EVAL.md` = full-history
objective, `XLE_RETRAIN_OOS2021_EVAL.md` = 2021 objective on the 2015→2025 window).
Here the **training window itself** is `2020-01-01 → 2025-12-31`. Its left edge sits
inside the 2020 COVID / oil-war crash, so the "exclude drastic XLE drops" lever matters
most. Data: Yahoo 2015-01-02 → 2026-07-15 (2,898 bars incl. `erx_close`). OIH and ERX
trade the XLE signal, so the one XLE High/Low signal model drives all three sleeves.

---

## Finding 1 — the 2020→2025 window underperforms the committed window on 2021→now

Committed thresholds held **fixed**; only the H/L **training window** varies (metrics
on 2021→now):

| H/L training window | Rows | XLE 2021→now | Sharpe | OIH | ERX |
|---|---|---|---|---|---|
| **2015→2020 (committed)** | 1,239 | **+105.4%** (−9.2%) | **1.37** | **+69.6%** | **+311.1%** |
| 2015→2025 (full) | 2,494 | +26.1% (−20.3%) | 0.48 | −2.1% | +49.7% |
| 2020→2025 | 1,487 | +12.0% (−13.1%) | 0.26 | −9.1% | +24.0% |
| 2020→2025, drop < −7% | 1,478 | +7.1% (−22.2%) | 0.17 | +1.0% | +10.1% |
| 2020→2025, drop < −5% | 1,466 | +30.6% (−16.4%) | 0.51 | +19.7% | +63.3% |
| 2020→2025, drop < −4% | 1,450 | +20.3% (−17.9%) | 0.37 | +11.2% | +39.2% |

The recent 6-yr window is *worse* than the committed early window and even the full
history on 2021→now — the divergence rule wants a **stable early reference frame** for
its High/Low model, and 2020→2025 is nothing but regime change.

## Finding 2 — excluding drastic drops helps this window (the 2020 crash is inside it)

Because the training window now *contains* the 2020 crash, dropping those days from the
ridge fit clearly helps: `drop<−5%` lifts XLE 2021→now from **+12.0% → +30.6%** and ERX
from **+24% → +63%**. `drop<−4%` and `drop<−5%` are the best fixed-threshold points and
supply the winning configs in the sweep — so the brief's exclusion lever works here (it
back-fired only on the 2015→2025 window, whose left edge was already the calm 2015).

## Finding 3 — even fully re-tuned to 2021→now, it still loses to committed

Best two configs on the 2020→2025 window, tuned on the 2021→now window:

| Config | Thresholds | XLE 2021→now |
|---|---|---|
| **Committed (unchanged)** | U1=0.16, D2=−0.10, d1-exit, −8% | **+105.4% / −9.2% / Sh 1.37** |
| Retrain-**RET** (max 2021 return) | U1=0.10, D2=−0.15, no-d1-exit, −8%, drop<−4% | +62.9% / −21.1% / Sh 0.73 |
| Retrain-**RISK** (max 2021 Sharpe) | U1=0.10, D2=−0.12, no-d1-exit, −8%, drop<−4% | +60.0% / −19.4% / Sh 0.73 |

This is the **best of the three retrain variants** on the 2021 objective (+62.9% vs
+45.9% for the 2015→2025 window) — but still ~40pp short of committed on return, at
**2.3× the drawdown** and **half the Sharpe**.

---

## Full per-period back-test — committed vs both retrained candidates

Return / MaxDD / Sharpe. **Bold** = best return in row. (H/L trained 2020→2025;
thresholds tuned to 2021→now.)

### XLE (the signal asset)
| Period | Committed | Retrain-RET | Retrain-RISK | Buy&Hold |
|---|---|---|---|---|
| 🌍 Full history (2015→now) | **+164.9% / −16.6% / 0.95** | +111.1% / −21.1% / 0.63 | +96.7% / −19.4% / 0.59 | +86.6% / −70.1% / 0.35 |
| 🐻 Bear (2021–22) | **+74.9% / −4.6% / 2.35** | +30.3% / −19.3% / 0.89 | +34.9% / −17.6% / 1.02 | +130.4% / −26.9% / 1.45 |
| 🐂 Recovery (2023→now) | +17.5% / −9.2% / 0.59 | **+25.0% / −11.7% / 0.63** | +18.6% / −11.7% / 0.52 | +33.3% / −22.1% / 0.48 |
| 🌐 **Full OOS (2021→now)** ⭐ | **+105.4% / −9.2% / 1.37** | +62.9% / −21.1% / 0.73 | +60.0% / −19.4% / 0.73 | +196.5% / −26.9% / 0.88 |
| 🔬 Recent OOS (2025→now) | **+15.3% / −5.4% / 1.14** | +12.0% / −11.7% / 0.69 | +11.8% / −11.7% / 0.69 | +29.9% / −18.8% / 0.83 |

### OIH (oil services · trades XLE signal)
| Period | Committed | Retrain-RET | Retrain-RISK | Buy&Hold |
|---|---|---|---|---|
| 🌍 Full history | **+119.1% / −23.1% / 0.59** | +84.2% / −34.7% / 0.40 | +63.9% / −31.0% / 0.35 | −28.0% / −90.3% / 0.14 |
| 🐻 Bear (2021–22) | **+75.2% / −9.4% / 1.69** | +15.0% / −28.1% / 0.43 | +21.7% / −26.4% / 0.57 | +93.9% / −36.4% / 0.95 |
| 🐂 Recovery (2023→now) | −3.2% / −18.5% / −0.03 | **+34.0% / −15.8% / 0.61** | +13.7% / −15.8% / 0.33 | +31.5% / −45.7% / 0.41 |
| 🌐 **Full OOS (2021→now)** ⭐ | **+69.6% / −18.5% / 0.77** | +54.2% / −32.6% / 0.52 | +38.4% / −31.0% / 0.43 | +142.9% / −45.7% / 0.62 |
| 🔬 Recent OOS (2025→now) | +4.4% / −11.1% / 0.32 | +19.3% / −12.6% / 0.85 | **+20.1% / −12.6% / 0.89** | +37.9% / −34.3% / 0.79 |

### ERX (2× energy · trades XLE signal · stop-less)
| Period | Committed | Retrain-RET | Retrain-RISK | Buy&Hold |
|---|---|---|---|---|
| 🌍 Full history | **+400.8% / −53.0% / 0.77** | +333.6% / −38.8% / 0.64 | +282.4% / −36.4% / 0.61 | −63.9% / −98.6% / 0.23 |
| 🐻 Bear (2021–22) | **+196.0% / −9.3% / 2.32** | +62.9% / −35.6% / 0.91 | +75.0% / −33.0% / 1.04 | +362.2% / −47.3% / 1.51 |
| 🐂 Recovery (2023→now) | +38.9% / −17.5% / 0.66 | **+46.1% / −22.4% / 0.60** | +33.5% / −22.4% / 0.50 | +37.0% / −43.8% / 0.43 |
| 🌐 **Full OOS (2021→now)** ⭐ | **+311.1% / −17.5% / 1.40** | +138.0% / −38.8% / 0.73 | +133.8% / −36.4% / 0.74 | +487.5% / −47.3% / 0.87 |
| 🔬 Recent OOS (2025→now) | **+32.7% / −10.6% / 1.21** | +21.2% / −22.4% / 0.66 | +21.0% / −22.4% / 0.66 | +48.5% / −35.7% / 0.79 |

---

## Verdict — do NOT adopt the retrain; keep the shipped strategy

* **Committed wins its objective (2021→now) on all three sleeves** — XLE +105% vs +63%,
  OIH +70% vs +54%, ERX +311% vs +138% — at markedly lower drawdown and ~2× the Sharpe.
* **Committed dominates loss-control**, which the brief explicitly asks for: the 2021–22
  bear is XLE **+74.9% at −4.6% MDD (Sharpe 2.35)** vs the retrain's +30–35% at ~−18%.
  The recent-window model gives up exactly the bear-avoidance edge the divergence system
  exists for.
* **The 2020→2025 window does help in the recent regimes it trained on** — it edges
  committed on the 2023→now recovery (OIH +34% vs −3%; ERX +46% vs +39%) and the 2025
  sliver — because it is fit to that data. But that is in-sample flattery, and it is paid
  for with far worse bear behaviour and drawdowns.
* **Excluding drastic drops works as hypothesised for this window** (Finding 2) — the
  2020 crash lives inside 2020→2025, so removing it improves the fit — but it only lifts
  the retrain to +63%, still well below committed +105%.
* **In-sample caveat:** 2021–2025 is now in-sample for the retrained model, so its
  numbers are flattered and it *still* loses; out-of-sample the gap widens.

### Ranking across the three retrain studies (XLE 2021→now, best re-tuned config)
| Training data | Best XLE 2021→now | vs committed +105.4% |
|---|---|---|
| **2015→2020 (committed, unchanged)** | **+105.4% / −9.2% / 1.37** | — |
| 2020→2025 (this eval) | +62.9% / −21.1% / 0.73 | −42pp, 2.3× MDD |
| 2015→2025 (full history) | +45.9% / −20.6% / 0.56 | −60pp, 2.2× MDD |

**Recommendation: leave `app/ticker_config.py`, the XLE model artifacts and
`data/xle/backtest_results.json` unchanged.** Across all three training-window and
objective variants tried, no retrain beats the shipped strategy on the requested
objective. This evaluation is the deliverable.
