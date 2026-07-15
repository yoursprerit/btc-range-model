# Can any XLE / OIH / ERX strategy **beat Buy & Hold** over 2021 → today? — evaluation

**Question (the brief).** Find any new strategy — or update the current one — that
**beats Buy & Hold** for **2021 → today (2026)** on XLE, OIH and ERX, using new training
data from whichever period is suitable (and excluding drastic XLE drops if it helps).
Re-run the back-test across every UI period. **Evaluation only — nothing shipped.**

```bash
python scripts/eval_xle_beat_bh.py            # → data/xle/beat_bh_eval.json
```

The three prior studies (`XLE_RETRAIN_*_EVAL.md`) tuned to *maximise return / minimise
losses*. This one sets a harder, specific bar — **beat B&H total return** — and searches
**all strategy families**, not just the divergence retrain:

* **trend** — `ma` (10 windows), `dual_ma` (8 pairs), `macd` (3), `ma_vol` (9) — no
  fitted model, so no training window;
* **divergence** — U1 / D2 / d1-exit swept, H/L model on **three training windows**
  (2015→2020, 2015→2025, 2020→2025 with drop<−5%);
* **long-biased trailing stops** — buy-and-hold that only exits on a peak drawdown of
  12–30% and re-enters on an MA recovery (the class most likely to out-run B&H).

Each candidate is a single **XLE** signal applied to all three sleeves (as shipped);
stops sweep {none, −8%, −10%, −12%} for XLE/OIH, ERX always stop-less. ~350 configs.

---

## The starting point — the committed strategy does *not* beat B&H on return

| 2021 → now | Committed strategy | Buy & Hold | Strat beats B&H return? |
|---|---|---|---|
| **XLE** | +105.4% / −9.2% / **Sh 1.37** | +196.5% / −26.9% / 0.88 | ❌ (−91pp) |
| **OIH** | +69.6% / −18.5% / **Sh 0.77** | +142.9% / −45.7% / 0.62 | ❌ (−73pp) |
| **ERX** | +311.1% / −17.5% / **Sh 1.40** | +487.5% / −47.3% / 0.87 | ❌ (−176pp) |

The committed strategy already **beats B&H on Sharpe and drawdown** on all three sleeves
— it just gives up raw return by standing aside in parts of the bull.

## Finding — **no** strategy beats B&H total return on **any** sleeve over 2021→now

Across all ~350 configs (trend + divergence + trailing-stop + every training window),
**0 / 3 sleeves** beat B&H on total return. The best XLE result each family could muster:

| Strategy family (best XLE 2021→now) | XLE return | vs B&H +196.5% |
|---|---|---|
| Divergence (committed) | +105.4% (−9.2%, Sh 1.37) | −91pp |
| Dual-MA 25/100 | +81.9% (−34.3%, Sh 0.62) | −115pp |
| MA-100 | +41.4% (−51.2%, Sh 0.40) | −155pp |
| Long-biased trailing stop | +196.5% *(= B&H — only by never de-risking)* | tie |
| MACD 12/26/9 | +20.1% (−40.0%, Sh 0.27) | −176pp |

The trailing-stop class is the tell: the **only** way it matched XLE B&H was a 30% stop
that never triggers (i.e. it *is* B&H); every stop that actually de-risked lagged, and on
OIH/ERX even the best trailing config lost (OIH +127% vs +143%; ERX +258% vs +487%).

### Why this is structural, not a tuning failure
2021→today is a **strong secular energy bull** (XLE +196%, ERX +487%). A long/flat overlay
is in cash part of the time, so it can only **lag** B&H's raw return — it wins on
*risk-adjusted* return, never on total return. Beating B&H outright would require dodging a
drawdown *and* re-entering before the recovery; XLE's −27% dips were followed by rallies
B&H captured in full, so every de-risking rule that missed a fall also missed enough of the
rebound to net below B&H. No fitted model or threshold changes this — extending / windowing
the training data (Findings from the three retrain evals) only *reduced* return further.

## The one genuine return-beat — rotate up-beta (cross-instrument)

You can out-return the **energy sector** (XLE B&H) — just not each instrument's own B&H —
by trading the higher-beta sibling on the XLE signal:

| vs the XLE sector benchmark (B&H XLE +196.5% / −26.9%) | Return | MDD | Sharpe | Beats sector? |
|---|---|---|---|---|
| **ERX sleeve (2× energy, committed)** | **+311.1%** | **−17.5%** | **1.40** | ✅ higher return **and** lower drawdown |
| OIH sleeve | +69.6% | −18.5% | 0.77 | ❌ |
| XLE sleeve | +105.4% | −9.2% | 1.37 | ❌ |

The committed **ERX sleeve already out-returns the sector by +115pp at ~⅔ the drawdown** —
the up-beta lever is how you "beat B&H" on return without a crystal ball. It still trails
ERX's *own* B&H (+487%), because you cannot out-return a 2× ETF in a bull by de-risking —
and, importantly, **no tested strategy beat ERX's own B&H either** (best ERX strat = the
committed +311%).

---

## Full per-period back-test — committed vs B&H (all UI periods)

Return / MaxDD / Sharpe. The search's best config was 2021-identical to committed (it only
nudged the *pre-2021, in-sample* window via D2 −0.10→−0.12 + no stop: XLE full-history
+187% vs +165%), so it is **not** shown as a separate column — it changes nothing on the
2021→now objective.

### XLE
| Period | Committed strategy | Buy & Hold | Strat wins on |
|---|---|---|---|
| 🌍 Full history (2015→now) | +164.9% / −16.6% / 0.95 | +86.6% / −70.1% / 0.35 | **return, MDD, Sharpe** |
| 🐻 Bear (2021–22) | +74.9% / −4.6% / 2.35 | +130.4% / −26.9% / 1.45 | MDD, Sharpe |
| 🐂 Recovery (2023→now) | +17.5% / −9.2% / 0.59 | +33.3% / −22.1% / 0.48 | MDD, Sharpe |
| 🌐 **Full OOS (2021→now)** ⭐ | +105.4% / −9.2% / 1.37 | +196.5% / −26.9% / 0.88 | MDD, Sharpe |
| 🔬 Recent OOS (2025→now) | +15.3% / −5.4% / 1.14 | +29.9% / −18.8% / 0.83 | MDD, Sharpe |

### OIH
| Period | Committed strategy | Buy & Hold | Strat wins on |
|---|---|---|---|
| 🌍 Full history | +119.1% / −23.1% / 0.59 | −28.0% / −90.3% / 0.14 | **return, MDD, Sharpe** |
| 🐻 Bear (2021–22) | +75.2% / −9.4% / 1.69 | +93.9% / −36.4% / 0.95 | MDD, Sharpe |
| 🐂 Recovery (2023→now) | −3.2% / −18.5% / −0.03 | +31.5% / −45.7% / 0.41 | MDD only |
| 🌐 **Full OOS (2021→now)** ⭐ | +69.6% / −18.5% / 0.77 | +142.9% / −45.7% / 0.62 | MDD, Sharpe |
| 🔬 Recent OOS (2025→now) | +4.4% / −11.1% / 0.32 | +37.9% / −34.3% / 0.79 | MDD only |

### ERX (2× energy · stop-less)
| Period | Committed strategy | Buy & Hold | Strat wins on |
|---|---|---|---|
| 🌍 Full history | +400.8% / −53.0% / 0.77 | −63.9% / −98.6% / 0.23 | **return, MDD, Sharpe** |
| 🐻 Bear (2021–22) | +196.0% / −9.3% / 2.32 | +362.2% / −47.3% / 1.51 | MDD, Sharpe |
| 🐂 Recovery (2023→now) | +38.9% / −17.5% / 0.66 | +37.0% / −43.8% / 0.43 | **return, MDD, Sharpe** |
| 🌐 **Full OOS (2021→now)** ⭐ | +311.1% / −17.5% / 1.40 | +487.5% / −47.3% / 0.87 | MDD, Sharpe |
| 🔬 Recent OOS (2025→now) | +32.7% / −10.6% / 1.21 | +48.5% / −35.7% / 0.79 | MDD, Sharpe |

Note the **full-history** row: over a complete bull-and-bear cycle the committed strategy
beats B&H on **return, drawdown AND Sharpe** for all three — because it side-steps the
brutal energy bear that 2021→now-only omits. The 2021→now window starts *after* the 2020
crash and is bull-dominated, which is exactly why B&H's raw return is hard to beat there.

---

## Verdict

* **No strategy — trend, divergence, trailing-stop, or any retrain/training-window — beats
  Buy & Hold on total return over 2021→now for XLE, OIH or ERX.** It is structural: a
  de-risking overlay cannot out-return B&H in a strong bull; it can only win risk-adjusted.
* **The committed strategy already beats B&H the way a timing overlay can** — **Sharpe
  ~1.4–1.6× and roughly ⅓ the drawdown** on every sleeve, plus a clean **return + risk**
  win over the *full* cycle. That is the meaningful outperformance.
* **The only lever that out-returns the sector is up-beta**, and the book already pulls it:
  the **ERX sleeve returns +311% vs the XLE sector's +196% at lower drawdown**. Pushing
  raw return above each instrument's own B&H would require more leverage / a short sleeve /
  perfect drawdown timing — none achievable with the tested rules, and all at higher risk.

**Recommendation: leave `app/ticker_config.py`, the XLE model artifacts and
`data/xle/backtest_results.json` unchanged.** The committed strategy is already the better
risk-adjusted vehicle; B&H's 2021→now raw return is not beatable with a long/flat overlay.
This evaluation is the deliverable.
