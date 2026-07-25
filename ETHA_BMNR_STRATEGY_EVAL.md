# ETHA / BMNR — strategy & portfolio-addition evaluation

**Status: evaluation only — nothing implemented.** No app, config or strategy
code was changed; reproduce every number with
`python scripts/eval_etha_bmnr.py`.

**Questions.**
1. Can trading **ETHA** (iShares Ethereum Trust, 1× spot-ETH ETF) and/or
   **BMNR** (Bitmine Immersion, the ETH-treasury equity) **on the BTC app's
   signals** beat buy-&-hold over the full available history?
2. Can a **new robust, optimal strategy built on ETHA's own price action** beat
   buy-&-hold — and is it better than the BTC-signal route?
3. Does adding ETHA, BMNR or both **improve the Overall portfolio** across all
   risk profiles — is either worth adding?

**Answers, up front.**
1. **ETHA — yes, decisively.** On the deployed CT-divergence engine (Standard-MA
   gate, signal-exit-only — the exact MSTR treatment) ETHA turns a **−46 %
   buy-&-hold into +264 %** at **−18 % MDD, Sharpe 1.61** — the strongest
   risk-adjusted sleeve the BTC signal has produced. **BMNR — no.** Its +104 %
   B&H is one unrepeatable two-week squeeze (Jul-2025, $7.75 → $135 → $16);
   every robust signal either misses it (CT: +13 %) or rides it into a −77 %
   drawdown, and **every simple rule is negative in the post-squeeze era**.
2. The best ETHA price-action rule — **ETHA > 30-day SMA, long/flat** — beats
   B&H (+95 %, Sharpe ~1.0, walk-forward-stable, confirmed on the full ETH-USD
   history), but is **strictly dominated by the BTC-signal route** on every
   metric and adds nothing at portfolio level. Trading ETHA off the **BTC
   parent signal beats every ETHA-price rule** — the MSTR/MSTU finding repeats.
3. **ETHA is worth adding** (as a `core` sleeve on the BTC CT signal): it
   improves Balanced and Growth **Sharpe and drawdown consistently across MC
   seeds** and is roughly return-neutral. **BMNR is not worth adding**: its
   apparent portfolio "benefit" is a cash-dilution artifact (in-market only 9 %
   of bars), and it drags the deterministic equal-weight book (−28 pp).

---

## 1. Setup

- **Data** — ETHA & BMNR daily closes from Yahoo (auto-adjusted), full listed
  history at evaluation time: **ETHA 2024-07-23 → 2026-07-24** (503 sessions;
  $26.24 → peak $36.59 on 2025-08-22 → $14.04) and **BMNR 2025-06-05 →
  2026-07-24** (285 sessions; $7.75 → $134.96 on 2025-07-03 → $15.79).
  ETH-USD (997 bars, round-tripped ~flat) for parent-signal and long-history
  checks. BTC signal from the committed CT feature CSV + trained model
  (`data/backtest/raw_features_daily.csv`, through 2026-07-23).
- **CT-engine runs** — byte-identical reuse of `app/btc_ct_engine.py`
  (`compute_sigs_pure` + `_run_bt`): same-bar fills, intrabar stops, both live
  gates (Pure-Regime OR / Standard-MA XOR), U1 +1.3 / D2 −1.3 thresholds, the
  post-stop re-entry override and 5-bar V-window — exactly how MSTR/MSTU are
  traded off the parent BTC signal.
- **Simple rules** — the prior-eval conservative convention: signal from closes
  ≤ *t* applied to the *t→t+1* return (next-bar), **10 bps per switch**,
  long/flat, no leverage. Cross-calendar signals (ETH closes at 00:00 UTC,
  after the 4 pm ET equity close) are charged an **extra day of lag**.
- **Portfolio runs** — the identical `overall_core.optimize_weights()` call the
  app and `scripts/build_overall.py` use (per-profile caps, SATA idle-cash
  yield, drawdown floor, objective), on the live 17-instrument universe,
  window 2021-01-05 → 2026-07-24.

---

## 2. Question 1 — ETHA / BMNR on the BTC signal (full period)

| Asset | Strategy (BTC signal) | Return | MaxDD | Sharpe | Trades | Win |
|---|---|--:|--:|--:|--:|--:|
| **ETHA** | Buy & hold | −46.5 % | −67.9 % | −0.08 | — | — |
| | **CT Standard-MA, no stop** *(MSTR treatment)* | **+264.1 %** | **−17.8 %** | **1.61** | 6 | 83 % |
| | CT Standard-MA, −6 % stop *(MSTU treatment)* | +222.1 % | −17.8 % | 1.49 | 8 | 62 % |
| | CT Pure-Regime, no stop *(BTC gate)* | +165.9 % | −31.2 % | 1.20 | 7 | 57 % |
| | Simple BTC > SMA30 (next-bar, 10 bps) | +130.7 % | −39.1 % | 1.12 | 43 | — |
| | Simple BTC > SMA40 | +68.8 % | −45.3 % | 0.78 | 39 | — |
| **BMNR** | Buy & hold | +103.8 % | −90.1 % | 0.90 | — | — |
| | CT Standard-MA, no stop | +13.1 % | −11.4 % | 0.54 | 2 | 50 % |
| | CT Pure-Regime, no stop | +27.7 % | −32.3 % | 0.57 | 4 | 50 % |
| | CT (any gate) with −3 %/−6 % stop | −22…+2 % | ≈−30 % | ≤0.14 | 3–5 | ≤33 % |
| | Simple BTC > SMA40 | +531.8 % | **−76.9 %** | 1.01 | 23 | — |

**ETHA.** The CT engine's ETHA sleeve is outstanding: 6 trades, worst trade
−2.9 %, exposure 28 %. One trade does the heavy lifting — long 2025-04-09 at
$12.42, D2-fade exit 2025-08-16 at $33.15 (**+167 %**), six days before the
all-time peak — after which it sat out the **entire −62 % ETH collapse**. Fixed
stops only hurt (a −8 % stop never fires — signal exits already cap intra-trade
pain), so the right treatment is MSTR's: **signal-exit-only, no stop**.

**BMNR.** The CT engine was flat through BMNR's July-2025 squeeze (its two
trades came later: −0.4 %, +13.6 %) — so it "only" made +13 %, but at −11 %
MDD. The simple BTC-trend rule caught the squeeze (+532 %) **and then rode the
collapse to a −77 % drawdown**. That return is not a repeatable edge — see §3.

---

## 3. Question 2 — a robust strategy on ETHA's own price action

Grid over SMA(5–200) / dual-MA / MACD / momentum, next-bar, 10 bps:

| Signal source → ETHA | Best rules | Return | MaxDD | Sharpe |
|---|---|--:|--:|--:|
| **ETHA own price** | **SMA30** | **+95.0 %** | −32.1 % | **0.98** |
| | SMA25 / MOM20 / DMA10-50 | +57…64 % | −37…−43 % | 0.73–0.77 |
| ETH-USD parent (honest 1-day lag) | SMA40 / MOM60 | +53…56 % | −33…−43 % | 0.71–0.72 |

**Validation of ETHA > SMA30 (and why it still loses):**

- **5-fold CV** (contiguous blocks): mean fold Sharpe **+0.58**, worst −2.02,
  beats B&H 4/5 — versus the **CT sleeve's +1.20 / −1.31 / 4-5** and simple
  BTC>SMA30's +0.90. The CT engine is more consistent fold-by-fold, not just
  in aggregate.
- **Walk-forward** (expanding train, pick best SMA *n* in-sample, apply to the
  next unseen fold): own-SMA picks are perfectly stable (30, 30, 30, 30) →
  OOS +45 %, Sharpe 0.76. The ETH-parent variant **falls apart** (unstable
  picks 100/30/40/80, OOS −30 %) — the parent-signal idea only works when the
  parent is BTC (BTC-SMA walk-forward: OOS +211 %, Sharpe 1.74).
- **Costs**: Sharpe holds 0.98 → 0.83 at 50 bps/switch (BTC>SMA30: 1.12 → 0.94).
- **Long-history check on ETH-USD itself** (2023-11 →, includes pre-ETHA):
  B&H +0.6 % / Sharpe 0.28; own-SMA30 **+124 % / 0.74**; BTC>SMA30 **+206 % /
  0.95**. The rule is real, not an ETHA-window artifact — and the BTC parent
  *still* wins on ETH's own longer tape.
- Caveat: the own-price Sharpe is moderately peaked at n=30 (25 → 0.77,
  35 → 0.75, 40 → 0.58) — a plateau, but a narrower one than the BTC eval's
  30–50 flat top.

**BMNR price action:** its own-price rules are unusable (best: SMA10, +7.5 % at
−75 % MDD). ETH/ETHA-parent trend rules show +440–550 % — but 5-fold CV puts
**all of it in fold 1** (the squeeze: +561…+654 %; every later fold ≤ +2 %),
and in the **post-squeeze era (2025-10 →)** every simple rule is negative
(BTC>SMA40 −22 %, ETH DMA25/100 −26 %, B&H −70 %); only the CT engine is
positive (+13 %) — by staying almost entirely flat. There is **no robust
tradable edge in BMNR yet** — only one founding event and its aftermath.

---

## 4. Question 3 — does adding them improve the Overall strategy?

Sleeves tested exactly as they would ship: **ETHA `core`** and **BMNR `beta`**
on the CT Standard-MA signal (no stop), folded into the live 17-instrument
universe and re-optimised with the identical `optimize_weights()` per profile
(quant-neutral figures; window 2021-01-05 → 2026-07-24, 1 667 bars).

| Scenario | Profile | Return | MaxDD | Sharpe | New-sleeve wt |
|---|---|--:|--:|--:|--:|
| Baseline | Balanced | +809 % | −12.7 % | 2.489 | — |
| | Growth | +1 761 % | −21.4 % | 1.694 | — |
| | Aggressive | +2 104 % | −29.4 % | 1.501 | — |
| **+ ETHA (CT)** | Balanced | +861 % (+53 pp) | **−9.8 %** | **2.585** | 16.2 % |
| | Growth | +1 678 % (−82 pp) | **−18.4 %** | **1.840** | 0.9 % |
| | Aggressive | +2 082 % (−22 pp) | −29.0 % | 1.515 | 12.7 % |
| + BMNR (CT) | Balanced | +742 % (−67 pp) | −13.2 % | 2.514 | 16.2 % |
| | Aggressive | +1 991 % (−112 pp) | −23.3 % | 1.661 | 5.0 % |
| + ETHA + BMNR | Balanced | +817 % (+8 pp) | −7.8 % | 2.548 | 18.6 / 3.9 % |
| | Aggressive | +2 306 % (+202 pp) | −29.3 % | 1.478 | 6.5 / 3.0 % |
| + ETHA (own-SMA30) | Balanced | +809 % (+1 pp) | −13.5 % | 2.465 | 5.0 % |

**Read these deltas against the MC noise band.** Re-running the *baseline*
optimiser with seeds 7/8/9 moves Growth +1 568…+1 829 % and Aggressive
+2 104…+2 338 % — so the return deltas above are mostly inside search noise.
What survives across **every seed**:

| Sharpe, baseline → +ETHA | seed 7 | seed 8 | seed 9 |
|---|---|---|---|
| Balanced | 2.489 → **2.585** | 2.490 → **2.547** | 2.506 → **2.578** |
| Growth | 1.694 → **1.840** | 1.731 → **1.800** | 1.764 → **1.843** |
| Aggressive | 1.501 → 1.515 | 1.445 → 1.399 | 1.465 → 1.466 |

- **ETHA lifts Balanced and Growth Sharpe in 6/6 seed-profile runs** and trims
  their MDD every time; Aggressive is neutral-within-noise. The deterministic
  **equal-weight check confirms it**: +16.9 pp return, MDD −15.5 → −13.9 %,
  Sharpe 2.169 → 2.205. In the recent era (2025 →) equal-weight goes +184.6 →
  +186.2 % with Sharpe 3.09 → 3.16.
- **ETHA earns real weight** (5–21 % across runs) and genuinely diversifies:
  its strategy stream is only 0.60-correlated to MSTR/MSTU, 0.21 to the BTC
  sleeve, ≈0–0.25 to everything else.
- **BMNR's apparent risk improvement is an artifact.** Its sleeve is in-market
  **9 % of bars** — inside `_combine`, a nearly-always-flat sleeve is mostly a
  SATA cash park, which is what shrinks Aggressive MDD (−29.4 → −23.3 %). The
  same de-risking is available by simply holding SATA. On the clean
  equal-weight test BMNR **subtracts** 28 pp with no Sharpe gain, and the
  recent-era slice drops +184.6 → +175.4 %.
- The **ETHA own-SMA30 sleeve adds nothing at book level** (Balanced +1 pp,
  Sharpe −0.02) — if ETHA is added, it should be on the CT signal.

---

## 5. Verdict

1. **Yes — trade ETHA on the BTC signal; it beats B&H by ~310 pp** at a
   fraction of the drawdown, and the CT-divergence engine beats every simple
   rule on it (Sharpe 1.61 vs 1.12 best-simple), fold-by-fold and in
   aggregate. The MSTR/MSTU architecture transfers cleanly: **parent signal,
   Standard-MA gate, signal-exit-only, no fixed stop**.
2. **The best ETHA price-action strategy (own SMA30, long/flat) is genuinely
   robust and beats B&H** (+95 % vs −46 %; walk-forward-stable; validated on
   the full ETH-USD history) — but it is **dominated by the BTC-signal route
   on return, drawdown, Sharpe and fold-consistency**, and adds nothing at
   portfolio level. Keep it as a benchmark, not a sleeve driver.
3. **Add ETHA (worth it), skip BMNR (not yet).** ETHA as an 18th instrument
   (`core`, capped 30–35 %) is a consistent Sharpe/MDD improvement at Balanced
   and Growth, return-neutral at Aggressive. BMNR has no robust edge on any
   tested signal, drags the equal-weight book, and its optimiser "benefit" is
   cash-dilution; revisit once it has meaningful post-squeeze history.

**If/when ETHA is implemented** (not done here): register it under the BTC
parent in the Overall universe (`kind="core"`, CT Standard-MA gate, no stop),
add a `FUNDAMENTAL_VIEW` conviction entry (crypto names currently carry 1.40 —
ETHA defaults to a neutral 1.0 in this eval), and extend the freshness/spot
plumbing to the new symbol. A 2× ETH sibling (e.g. ETHU) on the same signal
would be a natural follow-up eval, mirroring MSTU.

**Honest caveats.**
- ETHA's listed history is **~2 years** and its sleeve return is dominated by
  **one +167 % trade** (Apr → Aug 2025). K-fold consistency (4/5 folds beat
  B&H, worst fold −5 %) mitigates but cannot cure a short sample.
- Adding ETHA **concentrates more of the book on a single signal engine**: BTC,
  MSTR, MSTU and ETHA would all trade off the same CT model (which does take
  `eth_close` as a macro feature, but is trained to predict BTC bars). A
  regime where ETH decouples from BTC — e.g. an ETH-specific failure with BTC
  trending up — is unhedged and essentially untested (2024-26 ETH/BTC daily
  correlation stayed high throughout).
- BMNR is a treasury-mNAV equity with ongoing share-issuance dynamics; 13
  months of data, half of it one squeeze, supports no statistical claim beyond
  "nothing robust yet".
- Optimal weights are in-sample; MC deltas carry ±100–250 pp of search noise on
  the higher-octane profiles (quoted above). SATA is modelled, not
  market-tested. Backtested performance is not indicative of future results.

---

*Reproduce:* `python scripts/eval_etha_bmnr.py` (fetches ETHA/BMNR/ETH live,
builds CT predictions from the committed feature CSV + model, runs the live
universe, ~3–5 min). Numbers in this memo were produced 2026-07-25 against the
2026-07-23 feature vintage.*
