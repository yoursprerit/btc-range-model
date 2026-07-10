# Should the other leveraged siblings go stop-less? (MSTU · UGL · NUGT · ERX)

**Question.** SOXL (3× semis) trades stop-less because a tight 1× stop whipsaws a
leveraged ETF. The book has four other leveraged siblings, each trading its
parent's tuned stop. Does the same SOXL logic apply — should any go stop-less?

Each was swept through the **same live engine the Overall app uses**, so the
result is actionable:

| Sibling | Engine | Live stop |
|---|---|---|
| MSTU (2× MSTR) | `btc_ct_engine` (`STOP_PCT`, SL re-entry) | −3% |
| UGL (2× gold) | `gldm_engine` (`STOP_BY_ASSET`) | −3% |
| NUGT (2× gold miners) | `gldm_engine` (`FIXED_STOP` fallback) | −3% |
| ERX (2× energy) | daily `run_asset` (XLE divergence) | −8% |

```bash
python scripts/eval_lev_siblings_stop.py
```

---

## Sleeve sweeps (full OOS window)

**ERX — remove it (clearest case).** Every stop ≥ −10% gives the *identical*
result: the XLE divergence signal never lets ERX fall past ~8–10% before exiting,
so a stop that wide never fires. The live −8% is the only tested width that
clips — and it costs return and Sharpe for **zero** drawdown or win-rate benefit.

| ERX stop | Return | MDD | Sharpe | Win% | Trades |
|---|---|---|---|---|---|
| **none / −30 … −10%** | **+311.1%** | −17.5% | **1.40** | 67% | 67 |
| −8% (live) | +278.6% | −17.5% | 1.33 | 67% | 67 |
| −5% | +272.6% | −17.5% | 1.32 | 68% | 69 |

In the 2021–22 bear the −8% stop **doesn't even trigger** (none and −8% both
+196% / −9.3% / Sharpe 2.32 / 79% win). Same trade count at every width → the
stop only worsens a few fills. **Pure downside; widen past −10% or drop it.**

**UGL — modest win-win.** Stop-less improves return, Sharpe AND win-rate at
essentially unchanged drawdown, over a robust 70+-trade sample.

| UGL stop | Return | MDD | Sharpe | Win% | Trades |
|---|---|---|---|---|---|
| **none** | **+247.1%** | −18.1% | **1.37** | **67%** | 70 |
| −5% | +227.5% | −18.0% | 1.35 | 64% | 76 |
| −3% (live) | +210.8% | −17.6% | 1.30 | 61% | 80 |

**NUGT — −3% is clearly too tight; widen to ~−5% (best) or drop.** The live −3%
is the *worst* point on return and win-rate. A −5% stop is the sweet spot — it
beats stop-less on return, Sharpe **and** drawdown.

| NUGT stop | Return | MDD | Sharpe | Win% | Trades |
|---|---|---|---|---|---|
| **−5%** | **+1183.3%** | **−28.0%** | **1.48** | 59% | 82 |
| none | +900.9% | −31.2% | 1.30 | 63% | 70 |
| −20% | +1067.6% | −29.2% | 1.39 | 64% | 72 |
| −3% (live) | +634.4% | −28.3% | 1.29 | 51% | 92 |

**MSTU — too tight, but too little history to act on.** Only 6–7 trades since the
2024 launch, driven by the (noisy) daily BTC signal — very low confidence.
Directionally the −3% hurts win-rate (43% vs 67%) as expected, but pure stop-less
deepens drawdown; a wide −20% is the best tested point.

| MSTU stop | Return | MDD | Sharpe | Win% | Trades |
|---|---|---|---|---|---|
| −20% | +439.5% | −52.0% | 1.07 | 67% | 6 |
| none | +390.6% | −54.1% | 1.01 | 67% | 6 |
| −3% (live) | +357.6% | −48.5% | 1.02 | 43% | 7 |

---

## Overall portfolio — equal-weight clean signal (current vs no-stop)

| Variant | Return | MDD | Sharpe |
|---|---|---|---|
| baseline (all live stops) | +779.2% | −7.1% | 2.755 |
| MSTU no-stop | +787.2% (+8.0pp) | −7.1% | 2.745 (−0.011) |
| UGL no-stop | +784.6% (+5.4pp) | −7.1% | 2.756 (+0.000) |
| NUGT no-stop | +798.5% (+19.2pp) | −7.5% | 2.724 (−0.031) |
| ERX no-stop | +783.4% (+4.2pp) | −7.1% | **2.760 (+0.005)** |

Each sleeve is ~1/17 of the book, so all effects are small: every variant nudges
return up; ERX and UGL are the only two that also improve (or hold) Sharpe, while
MSTU/NUGT add return at a hair more risk.

---

## Verdict

Unlike SOXX (where the 1× −5% stop was correctly kept), **the leveraged siblings
mostly want a looser or no stop — the SOXL lesson generalises**, but not
uniformly:

* **ERX (−8% → none): yes, remove it.** Dominant on every axis — +33pp return,
  +0.07 Sharpe, *identical* −17.5% drawdown and 67% win-rate; the stop provably
  never helps (doesn't fire in the bear). Ships exactly like SOXL — it runs on
  the daily engine, so `stop_by_asset={"erx_close": 1.0}` on the XLE config.
* **UGL (−3% → none): yes, a clean win-win.** +36pp return, +0.07 Sharpe, +6pp
  win-rate, flat drawdown, over 70+ trades. Set via `gldm_core.STOP_BY_ASSET`.
* **NUGT (−3% → ~−5%): change, but widen rather than drop.** The −3% is the worst
  case; a −5% stop beats stop-less on return, Sharpe AND drawdown (+1183% / 1.48
  / −28%). Widen, don't remove.
* **MSTU (−3%): directionally too tight, but only ~6 trades of history — do not
  act yet.** Revisit once the 2× has a real sample.

**Win-rate note (the SOXL parallel):** dropping the stop lifts win-rate for MSTU
(43→67%), UGL (61→67%) and NUGT (51→63%) — the same manufactured-loser effect —
but **not ERX** (67→67%), because ERX's −8% rarely fires, so it clips fills
without adding losing round-trips. Win-rate alone is, again, a poor guide; the
return/Sharpe/drawdown trade-off is what drives the recommendation.

## Shipped

Implemented (MSTU left at −3% pending more history):

* **ERX → stop-less** — `stop_by_asset={"erx_close": 1.0}` on the XLE config
  (`app/ticker_config.py`). Regenerated `data/xle/backtest_results.json` confirms
  **+311.1% / −17.5% / Sharpe 1.40** OOS (was +278.6% / 1.33 at −8%).
* **UGL → stop-less · NUGT → −5%** — `app/gldm_core.STOP_BY_ASSET`
  (`{"GLDM": −3%, "GDX": −3%, "UGL": none, "NUGT": −5%}`) with a `stop_for()`
  helper now used by both `gldm_engine` (Overall) and the Gold app.

The XLE, Gold and Overall apps surface each sibling's per-instrument exit
(strategy card, live position panel, description), and the XLE / Gold / Overall
back-test artifacts were regenerated. MSTU stays at −3% — only ~6 trades of 2×
history, too little to act on.
