# 🩺 Strategy Health — the decay monitor

**How the platform detects that a strategy has stopped working.** This is the
*design + thresholds* doc for the 🩺 **Strategy Health** app; the compute lives
in `app/health_core.py` (unit-tested in `tests/test_health_core.py`), the
nightly build in `scripts/build_strategy_health.py`, and the read-only UI in
`app/health_app.py`.

---

## 1. Why monitor decay at all

Every sleeve's config was **selected** by sweeping candidates over the same
history it is reported on (stop sweeps, hyperparameter searches, the
2026-07-25 honest-fill re-sweep), so the live edge is expected to be smaller
than the back-tested edge, and simple technical rules (dual-MA, MACD,
divergence gates) are exactly the class of signal that dies when the regime
that made them work ends. Decay **starts at the sleeve level** and is masked
at the portfolio level by diversification — by the time the combined curve
shows it, a sleeve has been bleeding for months.

Two prior studies shape the design:

* **`OVERALL_OOS_WALKFORWARD_EVAL.md`** — the honest out-of-sample core is far
  smaller than the headline curve. Health is therefore judged against each
  curve's **own history**, never against the advertised numbers.
* **`OVERALL_ADAPTIVE_EVAL.md` (V4 penalty box)** — automatically de-rating
  cold sleeves was neutral-to-negative. So the monitor **never changes a
  weight**: an alarm triggers the *human* review protocol (re-run the sleeve's
  honest-fill re-sweep and decide — retune, de-rate, or retire).

## 2. The four monitors

| # | Monitor | Question | Data |
|---|---|---|---|
| **M1** | Book-vs-replay tracking | Is the *published* book earning what the model says it should? | `data/overall/book_archive/` + the walk-forward gated replay |
| **M2** | Drawdown tripwires | Has the curve gone deeper — or stayed underwater longer — than its reference history ever did? | each sleeve's strategy returns; each profile's replay returns |
| **M3** | Edge vs Buy & Hold | Is the *timing* alpha alive, or is absolute P&L just the asset rallying? | sleeve strategy vs B&H returns |
| **M4** | Trade expectancy | Are the last 20 trades consistent with the sleeve's own trade distribution? | per-sleeve closed-trade log |

**M1** reconstructs the daily return each archived target book earned
(decided-at-previous-close, close-to-close, SATA on the cash leg — the same
conventions as the replay) and compares it to the replay's same-day return.
Both legs are gross of costs, so the gap isolates *what was actually
published* (weights, timing, missed publishes); real fills layer in once the
executed-book history is long enough. **M2** is the classic falsification
test — it works even with few trades, which makes it the workhorse while
trade counts are small. **M3** exists because every sleeve's claim is timing
alpha over its own instrument. **M4** compares the rolling 20-trade mean
return against a bootstrap distribution of 20-trade samples from the sleeve's
*earlier* trades — "bad luck within the strategy's own variance" vs "behaviour
the strategy never produced".

## 3. States, thresholds, persistence

Four states per monitor: 🟢 healthy · 🟡 warning · 🔴 alarm · ⬜ **warming up**
(not enough data to judge — deliberately *not* green). Thresholds are fixed
**ex-ante** in `health_core.THRESHOLDS` (change them there and here together):

* **M1** — trailing **63-day** gap (live − replay) below **−1.5%** → 🟡, below
  **−4%** → 🔴; ⬜ until **21** live days exist.
* **M2** — reference = the curve **before the live anchor** (the first
  published book, 2026-07-16). Current drawdown deeper than the reference
  maximum, or a longer underwater spell, → 🔴; within **80%** of either limit
  → 🟡; needs **250** reference bars to arm.
* **M3** — rolling **126-bar** strategy-minus-B&H below **−0.5pp** (dead-band
  so hairline noise never flags) → 🟡; persisting **63 days** → 🔴.
* **M4** — last-**20**-trade mean below the **p10** of 4,000 bootstrap
  20-trade samples (fixed seed) → 🟡; below **p5** for **14+ days** → 🔴;
  ⬜ until 40 closed trades exist.

**Persistence lives in the committed artifact, not the UI**: each breach
carries a `first_breach_date` merged from the previous snapshot, so a
one-day flicker never escalates, "yellow for 41 days" is renderable, and every
flag's history is auditable in git. **Family-wise caveat:** with ~18 monitored
sleeves, about one 🟡 at any time is expected by chance — act on 🔴, or on a
🟡 that persists.

## 4. Pipeline & artifacts

```
publish-target-book.yml  (nightly, right after the book publishes; best-effort)
  └─ scripts/build_strategy_health.py
       ├─ overall_core.run_universe()                 (18 sleeves)
       ├─ walkforward_gated_replay() per UI profile   (Balanced, Growth)
       ├─ health_core.build_snapshot(...)
       ├─ data/overall/strategy_health.json           (snapshot + flags — the UI reads this)
       └─ data/overall/health_history.csv             (append-only daily series)
```

A health-build failure never blocks or unpublishes the day's book
(`continue-on-error` in the workflow). Seed or refresh by hand with
`python scripts/build_strategy_health.py`.

## 5. Where it surfaces

* **🩺 Strategy Health app** (sidebar) — verdict banner → portfolio strip
  (M1 chart + M2 profile tripwires) → sleeve board, worst first → per-sleeve
  drill-downs (drawdown vs reference max, rolling edge, expectancy band).
  Like 🕵️ Daily Audit it is deliberately light: it only reads the committed
  artifacts and never runs the engine.
* **🧭 Overall Trading** — a one-line health badge under the title, so a red
  flag finds the user on the page they open daily.
* **🕵️ Daily Audit** — section 5 shows the verdict (freshness answers
  *"did everything run?"*; health answers *"is it still working?"*).

## 6. What an alarm means (pre-committed responses)

* 🔴 **M2 on a sleeve/profile** — the back-test's risk model is falsified for
  that curve. Re-run the sleeve's honest-fill evaluation before the next
  quarterly refit; for a profile, de-risk first, analyse second.
* 🔴 **M3 / M4 on a sleeve** — the signal's edge is gone or its trades are
  off-distribution. Same protocol: re-sweep under the honest-fill rules and
  decide — retune, de-rate, or retire (the ETH/BMNR eval is the template).
* 🔴 **M1** — the published pipeline is bleeding vs the model even if every
  signal works: check publish timing, missed days, and weight drift between
  `signal_gated_allocation` and what was committed.

No automatic action is taken in any case — that is a deliberate design
decision backed by the V4 penalty-box result.
