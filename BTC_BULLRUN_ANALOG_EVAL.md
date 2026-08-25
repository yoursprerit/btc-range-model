# Is the 2026-08 BTC thrust a real bull leg or a fake-out?

> **Question.** From the **regime-divergence** point of view, has the pattern seen in the
> last week (2026-08-19 → 08-23) happened before — and does the **currently implemented**
> Bitcoin strategy call it a confirmed bull run or a fake-out?
>
> **Data vintage:** committed pull `2026-08-24` (`data/backtest/manifest.json`), CT bars
> through **2026-08-23**. BTC/on-chain datasets pass QC; only the MSTU files are flagged
> (irrelevant here). Every number below drifts with each data refresh — with 8–27 events in
> the sample, sometimes materially.
>
> **Not implemented.** Evaluation only. No strategy, threshold, gate or config changed.
> Reproduce with:
>
> ```bash
> python scripts/eval_bullrun_analogs.py
> ```

## TL;DR

| | Verdict |
|---|---|
| **Regime-divergence (Pure-Regime) gate** | **Confirmed entry.** U1 fired 2026-08-20 @ $76,694, `U1 + ↑MA30`. Currently **LONG — HOLDING**, +2.3% unrealised. |
| **Currently implemented live BTC gate** (Standard MA / XOR) | **No confirmation.** The combined-filter veto blocked it. Live book (`data/overall/target_book.json`, as-of 2026-08-23): BTC · MSTR · MSTU · ETH all **STAND ASIDE — FLAT, 0% target**. |
| **Has the pattern occurred before?** | **Yes, 26 prior U1 onsets — but never this big.** The shock is rank 1 of 27 on all three magnitude axes. The *nearest* analogs (Feb-2024, Nov-2024) were real bull legs; the *structural* analogs (same gate configuration) are a coin-flip-plus. |
| **Statistically, real or fake?** | **Real thrust, unconfirmed follow-through.** The shock cohort is significantly bullish (p = 0.003–0.026), but the current event has already dropped into the **faded** bucket — the divergence reversed within two bars. |

---

## 1. What actually happened

BTC ran **+21.7% in five CT bars** ($64,490 → $78,456), on a 6–7× volume surge
(26.6k → 198.3k BTC/bar), then volume halved.

| CT bar | Close | 1-bar | `err_hi` | `err_hi_ma3` | `hi_breaks_3d` | U1 | Pure gate | Live MA gate |
|---|---|---|---|---|---|---|---|---|
| 2026-08-18 | 64,490 | +0.2% | −0.68 | −0.40 | 0 | — | — | — |
| 2026-08-19 | 71,916 | **+11.5%** | **+10.45** | +3.10 | 1 | — | — | — |
| 2026-08-20 | 76,694 | +6.6% | +7.33 | **+5.70** | 2 | ✅ | 🟢 **ENTER** | 🚫 blocked |
| 2026-08-21 | 77,130 | +0.6% | −0.50 | +5.76 | 2 | ✅ | hold | 🚫 blocked |
| 2026-08-22 | 77,314 | +0.2% | −1.46 | +1.79 | 1 | — | hold | — |
| 2026-08-23 | 78,456 | +1.5% | −0.36 | **−0.77** | 0 | — | hold | FLAT |

`err_hi = +10.45%` means the realised high beat the model's predicted high by 10.45% of
close — **the model was completely blindsided**. That is the literal definition of a regime
divergence, not a trend the forecaster was already tracking.

**Why the live gate is flat.** BTC trades the **Standard MA (XOR)** gate
(`BTC_STRATEGY_GATE = "above_ma30"`, `app/btc_hourly_app.py:127`):

```
entry = U1 AND ((above_ma30 XOR clean_7d) OR V-reversal)
```

On 08-20/21 **both** `above_ma30` and `clean_7d` were True, so the XOR was False, no
V-reversal was in the 5-bar window, and the combined-filter veto fired. That veto exists
precisely to refuse "everything looks perfect" late-cycle entries. The **Pure-Regime (OR)**
gate — `U1 AND (bull_regime OR (clean AND ¬above) OR V-reversal)` — has no such veto, so it
entered.

## 2. Has this pattern happened before?

27 fresh U1 onsets over the 937-bar CT window (2024-01-30 → 2026-08-23). The current one is
**the largest on every axis**:

| Metric | Now | Rank | Previous max |
|---|---|---|---|
| `err_hi` shock | **+10.45%** | 1 / 27 | +8.59% (2024-02-26) |
| `err_hi_ma3` | **+5.70%** | 1 / 27 | +3.69% (2025-03-02) |
| Prior 5-bar BTC return | **+21.7%** | 1 / 27 | +16.6% (2024-11-11) |

The +21.7% five-bar thrust is the **99.9th percentile** of all five-bar windows in the CT
history; only 2024-02-28 (+23.1%, the spot-ETF melt-up) was larger.

### Closest historical analogs, and how they resolved

| Analog | Context | +5d | +10d | +21d | +42d | +63d | +90d |
|---|---|---|---|---|---|---|---|
| **2024-02-26** (`err_hi` +8.6) | spot-ETF breakout | +9.3% | +19.8% | +11.6% | +25.1% | +8.2% | +21.1% |
| **2024-11-05** (`err_hi` +7.0) | US-election repricing | +10.7% | +22.8% | +25.8% | +41.5% | +28.1% | +33.2% |
| **2024-11-11** (`err_hi` +6.1) | same leg, later entry | +4.1% | +13.6% | +9.1% | +8.4% | +10.7% | +12.1% |
| **2025-03-02** (`err_hi` +7.4) | policy-headline spike | −7.2% | −10.4% | −5.2% | −8.4% | +1.4% | +12.3% |

Three of the four large-shock analogs were **real bull legs**; the fourth (2025-03-02) was a
textbook fake-out — a one-day headline spike **below** the MA30 that round-tripped inside a
week and cost the Pure-Regime sleeve −7.2%.

## 3. Statistical evidence

Bootstrap (20k draws, seed 7) of each cohort's median forward return against the
unconditional distribution of BTC bars in the same window. Unconditional baseline:
+0.38% / +0.33% / +0.21% / +1.16% median at +5/+10/+21/+42 bars (~51–54% hit rate).

| Cohort | n | +5d | +10d | +21d | +42d |
|---|---|---|---|---|---|
| **A** · same gate config *(above & clean & bull → Pure enters, live MA blocks)* | 10 | +2.39% (p=.114) | +3.44% (p=.116) | +5.08% (p=.104) | +5.76% (p=.160) |
| **B** · big divergence shock *(`err_hi` ≥ 5.5%)* | 8 | +4.20% (p=**.026**) | +8.58% (p=**.003**) | +10.35% (p=**.012**) | +15.29% (p=**.009**) |
| **C** · parabolic prior thrust *(5-bar ≥ +10%)* | 8 | −0.82% (p=.731) | +0.89% (p=.430) | +7.85% (p=**.045**) | +5.48% (p=.196) |

*(median forward return; one-sided bootstrap p)*

- **Cohort B is the bullish evidence, and it is the real one.** Every horizon clears
  p < 0.03, hit rates 62–75%, medians 8–15× the unconditional baseline. The current event
  is not merely in this cohort — it is 22% larger than its previous extreme.
- **Cohort A — the configuration the strategy is actually in — is not significant**
  (p ≈ 0.10–0.16). Directionally positive (medians +2.4% to +5.8%, 70–90% hit) but within
  bootstrap noise at n=10, and it contains the sample's worst outcome: **2026-01-14, −27.2%
  at 21 bars**.
- **Cohort C is the caution.** After a ≥ +10% five-bar thrust, the *near-term* edge is gone
  (median −0.8% at +5d, coin-flip at +10d). The edge only reappears at +21d. Buying the
  fifth day of a parabolic move has historically paid nothing for three weeks.

### The fade test — the one that currently reads negative

Split all past onsets by whether the divergence **persisted** in the three bars after onset
(mean `err_hi` > 0) or immediately **faded**:

| | n | +5d | +10d | +21d | +42d |
|---|---|---|---|---|---|
| **Persisted** | 9 | +4.39% (89% hit) | +7.84% (**100%** hit) | +6.51% (78%) | +8.37% (67%) |
| **Faded** | 17 | +0.96% (65%) | +1.55% (59%) | +2.82% (59%) | +2.59% (59%) |

**The current event is in the FADED bucket** (post-onset mean `err_hi` = **−0.77%**, the most
negative of any cohort-A analog). `err_hi_ma3` collapsed **+5.76 → −0.77 in two bars**,
`hi_breaks_3d` is back to **0**, and U1 stayed on for only **2 bars** (median run length: 1).
Price is still grinding up (+1.5% on 08-23) but it is now doing so **below** the model's
predicted highs — the surprise is spent.

## 4. So: fake-out or real?

**Neither label fits cleanly. The honest read is: a real, statistically significant thrust
whose follow-through the strategy has not yet confirmed — and which the live gate is
deliberately refusing to trade.**

**Points toward a real bull leg**
1. Largest divergence shock in the entire CT history, on 6–7× volume — a genuine regime
   break, not a marginal threshold trip.
2. Cohort B's edge is significant at every horizon (p = 0.003–0.026), and the two closest
   analogs by magnitude (Feb-2024, Nov-2024) were the start of multi-month bull legs.
3. Full regime support: `bull_regime` = True, close **+18.7% above the MA30** and the MA30
   itself rising — the structural trend filter is fully on.
4. The Pure-Regime engine is the better BTC configuration on this window: **+63.5% /
   Sharpe 0.98 / MDD −19.9%** vs Standard MA's **+59.0% / 0.88 / −28.1%** (B&H +15.1%),
   which matches `TRADING_STRATEGY.md`'s own note that Pure Regime wins BTC on risk-adjusted
   terms.

**Points toward a fake-out / at minimum, unconfirmed**
1. **The live strategy did not confirm it.** The XOR veto exists to reject exactly this
   configuration, and the published book is flat across all four BTC-signal sleeves.
2. **The divergence already reversed** — two bars after onset, `err_hi_ma3` is negative and
   `hi_breaks_3d` = 0. That drops the event into the faded bucket, whose forward stats are
   roughly a third of the persisted bucket's.
3. **Cohort A is not statistically significant** and contains a −27% tail (2026-01-14) with
   an almost identical gate signature.
4. **Cohort C says the next three weeks are the dangerous ones**: post-parabolic, the
   near-term median is negative.
5. **The sample is tiny** (n = 8–27) over a window the repo's own docs flag as heavily
   in-sample optimised, on ~5–8 trades per configuration. Two events moving would flip
   cohort A's sign.

**Practical reading of the divergence sleeve's open position.** It is long from $76,694
(+2.3%), BTC carries **no fixed stop**, and because `bull_regime` is True the regime-adaptive
exit means a **D2 alone will not exit** — only D3 (a low-break immediately after ≥3
consecutive high-breaks), or D2 once price loses the MA30. `err_hi_ma3` is −0.77 against a
D2 trip at −1.30, so D2 is one weak bar away, but with the MA30 18.7% below it would not
close the trade. The realistic risk is therefore **holding a long through a full round-trip
of the +21.7% thrust** before an exit fires — the 2026-01-14 (−27%) failure mode, and the
single biggest argument for the live gate's veto.

## Bottom line

- **From the regime-divergence point of view: yes, this is a confirmed entry, and yes, the
  pattern has precedent** — 26 prior U1 onsets, of which the closest-by-magnitude analogs
  (Feb-2024, Nov-2024) were genuine multi-month bull runs.
- **From the currently implemented BTC strategy's point of view: it is not confirmed.**
  The live Standard-MA gate blocked the entry and the book is flat. Nothing here says the
  move is fake — it says the strategy declines to pay up for the fifth day of a parabola
  when every trend box is already ticked.
- **The single most decision-relevant statistic is the fade**: the shock cohort's
  significant edge belongs mostly to events where the divergence *persisted*. This one did
  not. Until `err_hi` turns positive again — a fresh U1, which would also break the
  `clean_7d` leg of the XOR and could unblock the live gate — the correct label is
  **unconfirmed, not disproven**.

### Caveats

* n = 8–27 events; every cohort p-value rests on a handful of observations and the same
  ~2.6-year window the thresholds were tuned on. Treat these as directional, not decisive.
* Cohorts are defined *post hoc* from the current event's own features; the p-values are not
  corrected for that selection.
* Forward returns are raw BTC returns from the onset bar, not strategy P&L — the strategy's
  exits truncate both tails.
* Signals recompute from revisable Binance/macro/on-chain bars on every pull; with this few
  events, a refresh can move any of these tables.
