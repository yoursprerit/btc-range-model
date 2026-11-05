# Intraday rate-of-drop exits for the leveraged / high-drawdown sleeves

**Question.** Every engine exits on the *bar close*, and the live 2:00 PM CT check
only compares a *level* to a stop once a day. Would selling mid-bar on a fast drop —
without waiting for the close — improve SOXL · NUGT · WGMI (· MSTU) and the Overall blend?

```bash
python scripts/fetch_intraday_eval_cache.py   # snapshot hourly bars (yfinance caps at ~730 d)
python scripts/eval_intraday_drop.py          # --slip-bps 25 for a costlier fill
```

## Method

The shipped daily engines are untouched; a trigger is laid over the position they hold,
using hourly bars of the traded ETF. A day with no trigger keeps the engine's own return,
so every delta is the overlay's.

* **Trigger** at hourly bar *i*: price is more than THR below the highest close of the last
  *N* bars (never reaching back before the entry close). *N* ∈ {3, 7, 21} bars (½–3 sessions).
  THR is either **k·σ_hour·√N** (vol-normalised, k ∈ 2…5) or a **fixed %** (3…15 %) — 27 configs.
* **Fill** (primary): detect on the bar close, sell at the **next bar's open**, charge 10 bp.
  Alternative `touch`: detect on the bar low, fill at the level.
* **Re-entry** after a trigger: `flat` = stay out until the engine's next fresh entry (what
  every engine does after a stop); `rebuy` = buy back at the next close if the engine is still
  long (what the morning book would do on its own).
* **Window**: 2023-12 → 2026-11 (≈2.9 y — all the hourly history that exists). SOXL has only
  **3** baseline trades in it, NUGT 56, WGMI 15.
* **MSTU is not covered.** The CT engine runs a 7-day BTC-anchored calendar on a synthetic
  pre-listing MSTU series; its daily returns do not line up with MSTU's US-session bars
  (corr ≈ 0.04). It needs BTC 24/7 hourly bars — a follow-up.

## Result 1 — tight, "rate-of-drop-sensitive" triggers destroy value on every sleeve

Full window, `flat`, close-detect / next-open fill, 10 bp:

| Config | SOXL ret / MDD / Sharpe | NUGT | WGMI |
|---|---|---|---|
| baseline | +688% / −69% / 1.24 | +238% / −41% / 1.27 | +395% / −17% / 1.87 |
| 3 % · 3h | +14% / −3% / 0.79 | +78% / −22% / 0.86 | +16% / −13% / 0.45 |
| 5 % · 3h | +8% / −8% / 0.42 | +83% / −27% / 0.81 | +27% / −15% / 0.55 |
| 2σ · 3h | +15% / −3% / 0.79 | +64% / −27% / 0.73 | +37% / −15% / 0.72 |
| 12 % · 3h | +233% / −23% / 1.25 | +271% / −33% / 1.40 | no trigger |

On NUGT and WGMI most tight-trigger exits are whipsaws (NUGT 3 %·3h: 36 triggers, 12 saves,
24 whipsaws; WGMI 5 %·3h: 9 triggers, 2 saves, 7 whipsaws averaging +28 % to the engine's exit).
This is the same pattern as the stop-width memos (`SOXL_STOP_EVAL.md`, `LEV_SIBLINGS_STOP_EVAL.md`):
the engines are trend followers, and a fast drop inside an uptrend usually recovers.
Vol-normalising (k·σ) did **not** beat fixed % on this sample.

## Result 2 — only a crash-only trigger (≈ ≥12 % in ≤ 3 hours) is interesting, and only with `rebuy`

`12 % · 3h` is also the config the first half of the data picks (see Result 3).

| `rebuy`, 10 bp | Return | MDD | Sharpe | Triggers |
|---|---|---|---|---|
| SOXL baseline → overlay | +688 % → **+1086 %** | −69 % → −59 % | 1.24 → **1.42** | 22 |
| NUGT baseline → overlay | +238 % → +271 % | −41 % → −33 % | 1.27 → 1.40 | 3 |
| WGMI | unchanged — never reaches 12 % while held | | | 0 |

* **Why `flat` is worse on SOXL.** Its two triggers include 2025-11-07, a −12 % intraday drop that
  closed −3 % and then ran **+194 %** to the engine's exit. Staying flat forfeits that run
  (+688 % → +233 %). Re-entry policy matters more than the threshold.
* **SOXL `rebuy` is a noisy edge, not a clean one.** 22 triggers: 12 helped, 10 hurt; net Δlog
  +0.41 while the best three episodes alone are +0.82. It beats 90 % of random placements of the
  same number of triggers (placebo median −0.08), and its MDD is shallower than all 200 placebos —
  but a placebo percentile on 22 clustered events is not strong evidence.
* **Cost breakeven.** SOXL `rebuy` stays above baseline at 25 bp (+1010 %) and 50 bp (+894 %) per fill
  and is back to baseline (+696 %, Sharpe 1.26) at **100 bp** — a crash-time fill on a 3× ETF can be
  that wide.
* **`touch` detection is worse, not better.** Firing on intrabar lows triggers 35 times on SOXL and
  returns +560 % (< baseline): intrabar spikes that recover are exactly the whipsaws.
* **NUGT has 3 triggers** (2 slightly hurt, 1 — the 2026-01-30 gold crash, −13.7 % vs the engine's
  −25.2 % — helped). Too few to infer anything.

## Result 3 — train/test (config picked on the first half, judged on the second)

| Re-entry | Train pick | TEST ΔCalmar | TEST ΔSharpe | TEST ΔMDD | TEST ΔReturn | Configs that improved TEST Calmar |
|---|---|---|---|---|---|---|
| `flat` | 12 % · 3h | −0.11 | −0.05 | +17.9 pp | −127 pp | 4 % of 27 |
| `rebuy` | 12 % · 3h | **+0.36** | **+0.07** | +6.0 pp | +1.2 pp | 15 % of 27 |

Train and test rankings agree (Spearman +0.76 flat / +0.68 rebuy), but the 27 configs are not
independent, so that is a smoothness check, not 27 confirmations. The train half holds only three
SOXL triggers; most of the SOXL evidence (19 of 22) is in the test half, and the second-ranked
`rebuy` pick (15 % · 7h) loses 75 pp of return out of sample.

## Result 4 — Overall equal-weight blend (SOXL + NUGT + WGMI overlaid, 18 sleeves)

| | Return | MDD | Sharpe |
|---|---|---|---|
| baseline (from 2024-01-18) | +346 % | −12.2 % | 2.960 |
| `flat` overlay | +310 % | −8.2 % | 3.213 |
| `rebuy` overlay | +355 % | −10.6 % | 3.070 |
| test half, baseline → `rebuy` | +116.6 % → +116.1 % | −6.1 % → −5.7 % | 2.92 → 3.02 |

The blend improvement is small and mostly risk-side; each sleeve is ~1/18 of the book and this is the
equal-weight read, not the optimiser's.

## Verdict

1. **A rate-of-drop exit does not optimise the backtest.** Any trigger tight enough to fire often
   (≤ 8 % or ≤ 5σ) cuts return and Sharpe on all three sleeves. Whipsaw dominates.
2. **There is a narrow case for a crash circuit-breaker** — roughly "≥ 12 % below the last ~3 hours' high" —
   on SOXL and NUGT *with* `rebuy` (so a false alarm costs about one day, not the rest of the trend).
   Its demonstrated value is tail-risk (MDD −69 % → −59 % on SOXL, Overall MDD −12.2 % → −10.6 %) with
   a Sharpe nudge, and it does not survive ≥ 100 bp fills.
3. **The evidence is thin**: 2.9 years, SOXL has 3 baseline trades, NUGT 3 triggers, WGMI none, and
   the MSTU/BTC cluster was not tested.
4. **Not recommended to ship as-is.** If pursued, run it paper-only first: it needs hourly live
   quote checks (the current job is a single 2:00 PM CT snapshot), a re-entry rule that matches
   `rebuy` (the amended book is re-published each morning anyway), and fills modelled at ≥ 50 bp.

## Next

* MSTU / BTC / ETH cluster on 24/7 hourly BTC bars (≈ 58 % of the book in the worst case).
* A cool-off re-entry (e.g. re-enter after the next daily close that holds above the trigger fill)
  between `flat` and `rebuy`.
* Re-run `fetch_intraday_eval_cache.py` periodically — yfinance's hourly window rolls, and the cache
  is the only way the test stays reproducible.
