# ETH / BMNR — strategy & portfolio-addition evaluation

**Status: ETH is the shipped sleeve (2026-07f); BMNR rejected.** The Overall
universe's 18th instrument is **ETH (spot Ethereum)** on the BTC parent signal —
CT Standard-MA gate, signal-exit-only, no fixed stop — and it has its own
**🔹 ETH Backtesting** tab in the ₿ Bitcoin app. **Live execution routes to the
ETHA ETF**, exactly as the BTC sleeve executes via IBIT (`scripts/ibkr_symbols.py`).

> ### ⚠️ Read this before trusting any earlier ETHA figure
>
> This memo previously reported the sleeve on **ETHA ETF closes** and concluded it
> was clearly worth adding (+264 %, later +101 % on a refreshed vintage; Sharpe
> 1.6 → 1.0). **Re-measuring on spot ETH overturns that conclusion.** Two
> independent effects were inflating the ETHA numbers, and both are now gone:
>
> 1. **Fill timing (a real look-ahead).** The CT signal for bar *D* is derived
>    from bar *D*'s own realized high/low, so it is only knowable when bar *D*
>    closes at **D+1 12:00 UTC**. The engine filled equity/ETF sleeves at the
>    **US close on calendar date D** — roughly **16 hours before the signal
>    exists**. On ETHA that was worth **+33 pp** (+100.7 % → +67.5 % once the fill
>    moves past the signal). Spot ETH on BTC's own 12:00-UTC bars fills *at* the
>    signal moment, so the sleeve now has no timing advantage to remove.
> 2. **Window.** ETHA listed 2024-07-23, so its window silently skipped
>    2024-03 → 2024-07 — a stretch that costs the sleeve dearly. Same instrument,
>    same engine: **+65.3 %** from 2024-07-23 vs **+8.8 %** from 2024-03-05.
>
> **This look-ahead is not ETH-specific — it affects the deployed MSTR and MSTU
> sleeves too, and it is larger there.** See §5. It is pre-existing behaviour that
> this change neither introduced nor fixed.

Reproduce with `python scripts/eval_eth_bmnr.py`.

---

## 1. Setup

- **ETH prices** — `data/backtest/eth_usd_daily.csv`: ETH-USD rebucketed from
  Binance hourly to **BTC's own 12:00-UTC bar boundary** (996 bars,
  2023-11-01 → 2026-07-23). Because ETH trades 24/7, the sleeve's same-bar fill
  lands exactly at the close of the signal bar — an ETF price cannot do this.
  Full coverage of the CT window also means **no staggered start**: the ETH sleeve
  runs 2024-03-05 → now alongside BTC/MSTR/MSTU.
- **BMNR** — Bitmine Immersion (ETH-treasury equity), 2025-06-05 → now.
- **Signal** — reproduced from `app/btc_ct_engine.py`: the trained CT H/L
  ensemble, U1/D2/D3 divergence gates, MA-30 regime, both live entry gates.
- **Simple rules** — signal from closes ≤ *t* applied to the *t→t+1* return,
  **10 bps per switch**, long/flat.

---

## 2. Question 1 — ETH / BMNR on the BTC signal

Full window **2024-03-05 → 2026-07-23**:

| Asset | Strategy (BTC signal) | Return | MaxDD | Sharpe | Trades | Win |
|---|---|--:|--:|--:|--:|--:|
| **ETH** | Buy & hold | **−51.4 %** | −67.5 % | — | — | — |
| | **CT Standard-MA, no stop** *(shipped)* | **+8.8 %** | −39.5 % | **0.23** | 8 | 62 % |
| | CT Standard-MA, ETHA-matched window (2024-07-23 →) | +65.3 % | −22.9 % | 0.79 | 6 | — |
| | *(contrast: same window on ETHA ETF closes)* | *+100.7 %* | *−18.6 %* | *1.04* | *6* | — |
| **BMNR** | Buy & hold | +103.8 % | −90.1 % | 0.90 | — | — |
| | CT Standard-MA, no stop | +13.1 % | −11.4 % | 0.54 | 2 | 50 % |
| | Simple BTC > SMA40 | +531.8 % | **−76.9 %** | 1.01 | 23 | — |

**ETH beats buy-&-hold by ~60 pp at roughly half the drawdown — but that is close
to the only good thing to say about it.** Sharpe **0.23** is far below every other
CT sleeve (BTC 0.84 · MSTR 1.47 · MSTU 1.38), and the trade log is lumpy: a single
**−32.6 %** loss (2024-07-15 → 2024-08-04) swamps most of the winners.

The sleeve is also **highly sensitive to fill timing** — shifting the fill one bar
moves the result **±47 pp** (+8.8 % → +55.7 %). An edge that swings that much on
execution timing alone is not a robust edge.

**BMNR is unchanged and still rejected:** its +104 % B&H is one two-week July-2025
squeeze; 5-fold CV puts essentially all of it in fold 1, and every simple rule is
negative in the post-squeeze era (2025-10 →).

---

## 3. Question 2 — ETH's own price action

Grid over SMA(5–200) / dual-MA / MACD / momentum, next-bar, 10 bps, on the full
ETH-USD history (2023-11 →, longer than the CT window):

| Signal source → ETH | Return | MaxDD | Sharpe |
|---|--:|--:|--:|
| Buy & hold | +0.6 % | −67.6 % | 0.28 |
| **own SMA30** | **+124.4 %** | −36.8 % | **0.74** |
| own SMA40 | +111.9 % | −45.0 % | 0.71 |
| own MOM60 | +99.9 % | −39.1 % | 0.66 |
| **BTC > SMA30 (parent trend)** | **+206.0 %** | **−29.0 %** | **0.95** |

One finding survives from the original evaluation and one reverses:

- **Still true:** a simple **BTC-parent trend filter beats ETH's own-price
  rules** (Sharpe 0.95 vs 0.74) — the MSTR/MSTU "trade the parent, not the child"
  result replicates on ETH.
- **Reversed:** the **CT engine no longer beats the simple rules on ETH.** On the
  honest fill it manages Sharpe 0.23, well below a plain BTC-SMA30 filter's 0.95
  on the same asset. The ML engine's apparent edge on the ETH sleeve was an
  artifact of the ETF fill timing.

---

## 4. Question 3 — does ETH earn a place in the portfolio? **No.**

ETH folded in as a `core` sleeve and re-optimised with the identical
`optimize_weights()` (window 2021-01-05 → 2026-07-24, 1 667 bars). MC search noise
is large on the higher-octane profiles, so this reports **three seeds**:

| Sharpe: baseline → +ETH | seed 7 | seed 8 | seed 9 |
|---|---|---|---|
| **Balanced** | 2.527 → **2.453** | 2.558 → **2.515** | 2.518 → **2.510** |
| Growth | 1.734 → 1.868 | 1.880 → 1.764 | 1.797 → 1.843 |
| Aggressive | 1.522 → 1.687 | 1.457 → 1.468 | 1.476 → 1.476 |

- **Balanced Sharpe falls in 3 / 3 seeds.** Growth and Aggressive flip sign
  across seeds — that is noise, not signal.
- **The deterministic equal-weight test is unambiguous and negative:**
  return **754.3 % → 716.2 % (−38.1 pp)**, drawdown **−15.5 % → −19.2 %**,
  Sharpe **2.171 → 2.097**. Worse on all three axes at once.
- **ETH is 0.80-correlated to the BTC sleeve** — it largely *duplicates* exposure
  the book already carries rather than diversifying it. (ETHA's stream looked only
  0.21-correlated to BTC purely because its fill ran on a different clock — that
  "diversification" was a timing artifact.)
- The optimiser's own verdict: ETH earns just **0.6–3.8 %** at Balanced.

**Shipped artifact with ETH in** (`data/overall/overall_results.json`, 18
instruments, OOS 2021 → now):

| Profile | Return | MaxDD | Sharpe |
|---|--:|--:|--:|
| **Balanced** | +825.7 % | −16.08 % | **2.38** |
| Growth | +1 652.7 % | −20.14 % | 1.79 |
| Aggressive | +2 104.6 % | −31.97 % | 1.48 |

vs equal-weight buy-&-hold **+198.3 % / −39.0 % / 0.68**. By period (Balanced
optimum): 🌐 Full OOS **+825.7 % / −16.1 % / 2.38** · 🐻 Bear 2021-22
**+40.5 % / −8.0 % / 1.40** · 🐂 Bull 2023 → **+552.2 % / −16.1 % / 2.74** ·
🔬 Recent 2025 → **+213.2 % / −9.3 % / 3.28**.

For reference the ETHA-era artifact read Balanced +793 % / **−8.6 %** / **2.60** —
so on honest measurement the swap **costs ~0.2 Sharpe and roughly doubles Balanced
drawdown**.

### Recommendation

**Drop the ETH sleeve, or keep it only as a monitored 17+1.** It fails the one
test with no optimiser noise in it (equal-weight: worse return, worse drawdown,
worse Sharpe), it is 0.80-redundant with BTC, and its stand-alone Sharpe of 0.23
is the weakest in the universe. The per-kind cap keeps the live damage small
(~1–4 % at Balanced), which is why removing it is not urgent — but it is not an
improvement, and this memo should not be read as endorsing it. **That call is the
owner's; nothing here removes the sleeve.**

---

## 5. ⚠️ Separate, larger finding — the equity sleeves' fill precedes their signal

Isolating the ETH/ETHA gap surfaced this in the **deployed** engine, and it is not
ETH-specific:

| Sleeve (full CT window) | Fill @ close date *D* (as deployed) | Fill @ date *D+1* (post-signal) | Δ |
|---|--:|--:|--:|
| **MSTR** | **+337.8 %** · MDD −19.9 % · Sh **1.47** | **+183.7 %** · MDD −27.5 % · Sh **1.08** | **−154 pp**, −0.39 Sharpe |
| ETHA (retired) | +100.7 % · Sh 1.04 | +67.5 % · Sh 0.79 | −33 pp |
| ETH (12:00-UTC bars) | +8.8 % *(fill = signal moment)* | — | n/a |

The CT signal for bar *D* uses bar *D*'s realized high/low → knowable only at
**D+1 12:00 UTC**. BTC and ETH fill at exactly that instant (both are 12:00-UTC
bars). MSTR/MSTU fill at the **US equity close on date D**, ~16 h earlier. So
roughly **45 % of MSTR's headline return is unavailable to a live trader.**

`_relabel_result_to_visible()` already shifts *displayed* trade dates +1 day for
exactly this reason, so the convention is half-acknowledged in the code — the
**prices** were never shifted with them. The repo describes this as "same-bar
execution … modestly flatters it", which understates it.

**Not fixed here.** Correcting it would move every headline number in
`TRADING_STRATEGY.md`, the per-app artifacts and the live strategy — a decision for
the owner, not a side effect of an ETH swap. Suggested follow-up: re-tune MSTR/MSTU
against a post-signal fill and re-publish, or re-anchor the equity sleeves onto a
bar boundary that closes with the signal.

---

## 6. Caveats

- ETH's CT-sleeve window is ~2.4 years and 8 trades — thin either way.
- CT-engine numbers drift with feature-CSV refreshes (on-chain / Coinbase-premium
  series get revised between pulls); figures here are the 2026-07-25 pull.
- Optimal weights are in-sample; MC deltas carry ±100–300 pp of search noise on
  Growth/Aggressive, which is why the seed table and the deterministic
  equal-weight test carry the argument rather than any single run.
- SATA is modelled, not market-tested. Backtested performance is not indicative of
  future results.
