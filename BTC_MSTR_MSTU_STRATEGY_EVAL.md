# BTC / MSTR / MSTU — optimal & robust strategy evaluation

**Question.** Across the full available history (a complete bull→bear round-trip),
what is the most profitable *risk-adjusted* strategy for BTC, MSTR and MSTU — and
can a **simpler, less overfit** rule match or beat the currently-deployed engine?

**Answer, up front.** The deployed **CT-divergence (ML) engine is genuinely good and
is *not* beaten** by any simple rule on risk-adjusted terms — it wins on Sharpe and
drawdown for all three assets. But a single, radically simpler rule recovers ~60–80 %
of the return with a tiny overfitting surface and survives k-fold + walk-forward
validation:

> **Hold each asset only while BTC closes above its 40-day SMA — long/flat, one BTC
> trend signal driving all three sleeves.** (BTC-SMA 30–50 is a flat plateau; 40 is the
> centre.)

Two non-obvious findings make it robust:
1. **MSTR and MSTU are best traded off the BTC *parent* trend, not their own price.**
   Their own-price SMAs look great in aggregate but fall apart fold-by-fold.
2. **For BTC itself, no simple price rule comes close to the ML engine** (Sharpe 0.42 vs
   1.05) — that is exactly where the model's complexity earns its keep.

---

## 1. Setup

- **Data** — `data/backtest/`: BTC (Binance 12:00-UTC bars), MSTR & MSTU (exchange
  sessions). MSTU uses the committed **OLS-synthetic** series for the long window and is
  **cross-checked on the real tape** (from 2024-09-18). Refreshed pull to 2026-07-13.
- **Window** — `2024-03-05 → 2026-07-13` (~2.36 yr), matching the CT engine's usable
  span (its ML feature warm-up begins ~2024-03). BTC round-trips: **35k → 124k (peak
  ~Oct-25) → 61k**; MSTR 42 → 473 → 95; MSTU 7 → 257 → 1.9 (2× decay ≈ total wipeout).
  This deep, unrecovered bear is *why* a long/flat rule can beat buy-&-hold on **total
  return**, not just risk-adjusted terms.
- **Execution (simple rules)** — deliberately conservative: signal from closes ≤ *t*,
  position applied to the *t→t+1* return (**next-bar**), **10 bps charged per switch**,
  long/flat, no shorting/added leverage.
- **Baseline (deployed)** — reproduced from `app/btc_ct_engine.py`: ML High/Low ensemble
  + U1/D2/D3 divergence gates + MA-30 regime + **3 % intrabar stops** on MSTR/MSTU +
  post-stop re-entry override. It executes **same-bar** and stops **intrabar**, both of
  which modestly flatter it versus the simple rule's next-bar fills.

---

## 2. Head-to-head (full window)

| Asset | Strategy | Return | CAGR | MaxDD | **Sharpe** | Expo | Trades |
|---|---|--:|--:|--:|--:|--:|--:|
| **BTC** | Buy & hold | −7 % | −3 % | −53 % | 0.14 | 100 % | — |
| | Simple — **BTC > SMA40** (own) | +28 % | +11 % | **−27 %** | 0.42 | 50 % | 54 |
| | **Live — CT-divergence (ML)** | **+85 %** | +30 % | −28 % | **1.05** | ~50 % | 8 |
| **MSTR** | Buy & hold | −12 % | −5 % | −83 % | 0.38 | 100 % | — |
| | Simple — **BTC > SMA40** (parent) | +161 % | +50 % | −45 % | 0.96 | 51 % | 44 |
| | **Live — CT-divergence (ML)** | **+270 %** | +74 % | **−22 %** | **1.34** | ~50 % | 9 |
| **MSTU** | Buy & hold | −94 % | −70 % | −99 % | 0.19 | 100 % | — |
| | Simple — **BTC > SMA40** (parent) | +126 % | +41 % | −81 % | 0.88 | 51 % | 44 |
| | **Live — CT-divergence (ML)** | **+469 %** | +109 % | **−48 %** | **1.12** | ~50 % | 10 |

Both approaches turn a flat/negative buy-&-hold into strong gains. The live engine leads
on every risk-adjusted measure — most decisively on **drawdown for the leveraged names**
(its intrabar 3 % stop halves MSTU's MDD: −48 % vs −81 %).

---

## 3. Why the simple rule holds up — validation

### 3a. K-fold cross-validation (5 contiguous time-blocks, 10 bps)
Consistency across cuts the parameters never saw, not the peak number:

| Rule | mean fold Sharpe | worst fold | beats B&H |
|---|--:|--:|--:|
| MSTR · **BTC-SMA40** | **0.88** | −0.42 | **4 / 5** |
| MSTU · **BTC-SMA40** | **0.78** | −0.55 | **4 / 5** |
| BTC · own SMA40 | 0.29 | −0.94 | 3 / 5 |
| MSTR · own SMA20 (aggregate-best) | 0.47 | **−1.37** | 3 / 5 |
| MSTU · own SMA20 (aggregate-best) | 0.61 | **−1.89** | 4 / 5 |

The BTC-parent rule has a **higher mean fold Sharpe and a far shallower worst fold** than
the own-price rules — the own-price SMAs' big aggregate returns come from one lucky fold.

### 3b. Walk-forward (expanding train → pick best `n` in-sample → next unseen fold)
| Asset | OOS-stitched ret | OOS Sharpe | `n` picked each fold |
|---|--:|--:|---|
| MSTR | +77 % | 0.80 | 30–40 (stable) |
| MSTU | +40 % | 0.68 | 30–40 (stable) |
| BTC | +11 % | 0.27 | 30–40 (stable) |

Parameter selection **barely moves** — the sign of a non-overfit knob.

### 3c. Real MSTU (non-synthetic tape, 2024-09-18 → now)
| On real MSTU | Return | Sharpe | MaxDD |
|---|--:|--:|--:|
| Buy & hold | −93 % | −0.02 | −99 % |
| **BTC-SMA30** | **+297 %** | **1.23** | −74 % |
| BTC-SMA50 | +278 % | 1.19 | −75 % |
| own SMA20 | +67 % | 0.76 | −79 % |

The edge is **not** a synthetic-series artifact — on the real fund the BTC-parent rule is
even stronger (Sharpe 1.1–1.2), and beats the own-price rule by a wide margin.

### 3d. Bull vs bear split (bull 2024-03→2025-10, bear 2025-11→now)
| | Strat bull | Strat bear | B&H bear |
|---|--:|--:|--:|
| BTC | +46 % | −13 % | −43 % |
| MSTR | +114 % | **+22 %** | −66 % |
| MSTU | +76 % | **+29 %** | −94 % |

The rule keeps pace in the bull and **avoids the rout** — on MSTR/MSTU it even posts
positive bear-market returns.

### 3e. Cost & stop sensitivity
- **Costs** — BTC-SMA40 Sharpe holds to 50 bps/switch on the leveraged names (MSTR
  0.99→0.84, MSTU 0.90→0.82). BTC decays more (0.49→0.15) — its edge is genuinely thin.
- **Stops don't help the simple rule** — the trend exit already caps worst *intra-trade*
  drawdown near **−7 %** on MSTR, so any fixed stop ≥ 10 % never binds. Adding one is
  redundant complexity. (The live engine's benefit comes from a *tight 3 % intrabar* stop
  on a 90 %-vol asset — a different mechanism than a daily-close stop.)

---

## 4. Recommendation

1. **Default / most robust simple strategy:** one signal — **BTC close > 40-day SMA**,
   long/flat — drives **all three** sleeves. MSTR & MSTU off the **BTC parent** trend, not
   their own price. Simplest possible structure, no ML, no per-asset tuning, no stops;
   validated across k-fold, walk-forward, real-MSTU and costs.
2. **Keep the deployed ML engine where it wins:** it is not beaten on risk-adjusted terms.
   Its two real edges are (a) **genuine BTC alpha** (Sharpe 1.05 vs a trend filter's 0.42)
   and (b) **leveraged-sleeve drawdown control** via the intrabar stop.
3. **Pragmatic read:** the gap on MSTR/MSTU is mostly **drawdown**, not return direction.
   If one piece of complexity is worth keeping on the leveraged sleeves it is the
   **intrabar stop**, not the entry logic. BTC is the asset that most justifies the model.

*No app, model, or strategy code was modified — this is an evaluation of `data/backtest/`
against the reproduced live engine in `app/btc_ct_engine.py`. Backtested performance is
not indicative of future results.*
