# Data Consistency — why backtest numbers used to drift, and the fix

## The symptom

Running the app at different times of the same day produced drastically
different backtesting numbers with **zero code changes** — e.g. the risk-profile
total return showing ~970% in the morning and ~1204% in the evening, and the
walk-forward strategy P&L since 2026-03-01 showing 11% vs 38.6%.

## Root causes

Every equity/gold sleeve re-fetched its **full multi-year daily history live
from Yahoo's free chart API** each time the Streamlit cache expired (every
5–30 minutes), and every backtest — the per-sleeve simulations, the
walk-forward gated replay, the risk-profile metrics — was **recomputed from
whatever that fetch happened to return**. Yahoo's free feed is not a stable
historical source, especially on shared egress IPs (Streamlit Community
Cloud), so two page loads hours apart could legitimately see different input
data:

1. **Silently dropped feature columns.** `_merge()` fetches each symbol
   independently and simply skips symbols that return empty (rate-limit,
   transient error). A morning fetch that lost e.g. `vix`/`qqq` produced a
   *different feature set* — different sentiment gauge, different model
   inputs, different U1/D2/D3 signals **across the entire history** —
   therefore a completely different backtest, with no visible error.
2. **Silently swapped data sources.** When Yahoo withheld a session, the
   loaders repaired it from Yahoo's hourly feed or from Nasdaq's quote API
   (slightly different prints), and when a fetch failed entirely they fell
   back to a **stale committed CSV** ending days earlier. Each path yields a
   different history → different numbers.
3. **The in-progress bar.** During US market hours the newest "daily" bar is
   partial and keeps moving, nudging the tail of every simulated equity curve.
4. **Provider-side history revisions** (corrected prints, adjustment changes)
   were absorbed instantly and invisibly.

The BTC/MSTR/MSTU/ETH sleeve did **not** have this problem — it already runs
from a committed, checksummed, QC-guarded vintage (`data/backtest/` +
`manifest.json`, refreshed by `scripts/pull_backtest_data.py` and guarded by
`scripts/validate_refreshed_data.py`). The fix below extends that same
philosophy to every other sleeve.

## The fix: `app/data_gate.py`

All sleeve daily loads now go through a **quality gate with pinned
snapshots**:

```
                    ┌──────────────────────────────────────────────┐
 fetch (Yahoo) ───▶ │ 1 validate: columns · span · plausible moves │ ──pass──▶ pin as snapshot
                    │   index sanity · no regression vs snapshot   │           (CSV + manifest
                    │   per-day return agreement with snapshot     │            with SHA-256)
                    └──────────────────────────────────────────────┘
                                      │ fail (after 1 retry)
                                      ▼
                        serve the last KNOWN-GOOD snapshot
                        (rejection recorded in the audit trail)
```

* **Pinning** — once a validated fetch covers the most recent completed US
  session, its completed-bars history is persisted (`data/<sleeve>/
  macro_daily.csv` + `macro_daily_manifest.json`) and **reused byte-for-byte
  until the next session completes**. Morning and evening now compute the
  backtests from the *same* data; results roll forward only at the close.
* **Quality checks** (`run_quality_checks`) — non-empty, ≥260 rows, sorted
  unique session index, positive/finite closes, **every declared macro &
  sibling column present with data** (kills cause #1), no one-day traded move
  beyond ±75% (bad prints / large-ratio split splices; the bound clears the
  worst real 3× leveraged-ETF day on record — ERX −60% on 2020-03-09), no
  row-count or span
  regression versus the snapshot, **no completed session inside the last 150
  the snapshot already carries missing from the fetch** (see the dropped-session
  hole below), and per-day return agreement with the
  snapshot over the overlapping history (kills cause #4 silently rewriting
  the past).
* **Independent cross-check** — a session entering the snapshot for the
  first time has no prior reference for `history_agreement` to compare
  against, so its **traded closes are verified against Nasdaq's independent
  tape** (`app/market_fallback.py`) before the refresh is accepted: bases are
  aligned on the sessions just before the new ones (so a split can't
  masquerade as an error), and a disagreement beyond 0.5% rejects the fetch
  like any other failed check. Best-effort where Nasdaq can't serve the
  symbol or is unreachable — an absent second opinion never blocks a refresh,
  it just isn't recorded as verification. (Empirically the two tapes agree to
  fractions of a basis point — see the source-comparison study below.)
* **Fallback** — a rejected fetch never reaches the models; the last
  known-good snapshot is served and the failure is visible (kills causes
  #1–#2 as sources of silent drift).
* **Live vs history split** — the pinned history is completed-bars-only; the
  in-progress bar is overlaid from a tiny `range=5d` fetch for the live
  views and is never persisted (bounds cause #3 to the live row). In
  publisher mode (`OVERALL_COMPLETED_BARS_ONLY`) no partial bar is attached.
* **Audit trail** — every load appends `{decision, source, span, rows,
  SHA-256, failed checks, consumer app}` to `runtime/dataset_audit.json`;
  the 🕵️ **Daily Audit** app renders it (section *6 · Dataset audit trail*),
  including which apps consumed which dataset. Manifests double as the
  at-rest provenance record.
* **Fewer network calls** — the pinned fast path replaces the previous
  full-history refetch every 5–30 min per sleeve with one small `5d` fetch;
  the full fetch happens roughly once per session roll (or when validation
  demands a retry).

### Wired in

| Consumer | Loader | Dataset key |
|---|---|---|
| 🧭 Overall engine (all backtests, risk profiles, walk-forward replay) | `overall_core._load_daily` | each ticker key |
| Generic ticker apps (SOXX/GRID/XLE/REMX/WGMI/PBW/ARTY) | `ticker_app.get_daily` | ticker key |
| Gold + Gold Miners engines (GLDM·UGL·GDX·NUGT) | `gldm_engine._load_daily` | `GLDM` |
| BTC · MSTR · MSTU · ETH | already pinned via `data/backtest/` + manifest | (unchanged) |

Every entry point keeps its legacy path as a fallback if the gate itself is
unavailable, and behaviour is otherwise unchanged — same columns, same
engines, same signals; only the *stability and provenance* of the inputs
changed.

### The committed vintage (shared store)

`scripts/refresh_sleeve_snapshots.py` runs the same gate headlessly for every
sleeve and the daily `refresh-backtest-data` workflow commits the refreshed,
validated snapshots + manifests. The git repository is the "secure free
database": versioned, hash-verified (manifest SHA-256 is checked against the
CSV before a snapshot is trusted for pinning), with full history and
rollback. Every deployment boots from the same committed vintage instead of
re-trusting a live fetch.

### The dropped-session hole (ARTY phantom D3, 2026-08-15)

The row-count and span checks are both *aggregate*: a fetch that **back-fills
older gaps while dropping a recent session** grows the total and keeps the last
date, so it passed. `history_agreement` could not see it either — it compares
only dates present in **both** frames, and a hole simply isn't compared.

That is not cosmetic, because the signature engines read the last ~150 completed
bars **positionally**. On 2026-08-14 the ARTY refresh restored 07-21, 07-22 and
07-31 and dropped **2026-08-11** (2038 → 2041 rows, every check green). Removing
that bar re-links the high-break run: `consec_hi` 3 → 2, which is exactly the D3
exhaustion threshold. The ≈7:15 AM CT publish on 08-15 still saw a vintage
*with* 08-11 and booked ARTY `CLOSE` / `EXIT NEXT BAR — D3 exhaustion`; 37
minutes later the pinned snapshot was written *without* it, so every app read
`LONG — HOLDING` with no D3 anywhere — the action plan and the ARTY app
contradicted each other all day off one absent session.

`no_missing_recent_sessions` closes the hole: any session in the snapshot's last
150 that a later fetch fails to serve rejects the fetch, and the sleeve keeps
serving its known-good snapshot (visibly frozen in the daily audit) rather than
silently changing shape. The UI half of the fix is
`overall_core.book_phantom_actions` — see *Book vs engine* below.

### Book vs engine

A published book is frozen at the morning publish, so the action plan's pinned
Action / Signal cells can legitimately run *behind* the engine (a signal that
committed at a later close — flagged `_book_masked_exit` / `_book_masked_entry` /
`_book_masked_avoid`). `overall_core.book_phantom_actions` catches the opposite:
a book instruction the engine cannot derive from the **same as-of bar at all** —
a `CLOSE` on a position still held with no exit signal, or an `OPEN` on a sleeve
still flat. Neither state is reachable through the normal publish→execute
lifecycle, so it means the publish ran on a different data vintage. The plan
keeps the published pill (it is what the executor traded) and adds an amber ⚠️
flag with today's engine read, plus a callout above the table.

### Tests

`tests/test_data_gate.py` covers the checks (missing/dead columns, extreme
moves, regression, rewritten history, dropped recent sessions), the
pin/refetch/fallback flows, the completed-vs-in-progress split, snapshot tamper
detection, and the audit trail. `tests/test_book_phantom_actions.py` pins the
book-vs-engine mismatch flag — including that ordinary publish-lag and executed
book instructions stay unflagged.

## The BTC vintage freeze (the "+950% vs +1205%" incident, 2026-08-03)

After the equity gate shipped, the headline still jumped between reboots
(Balanced ~950% → ~1205%) with "all QC passed". Forensics (fully reproduced
offline) traced 100% of the movement to the **BTC/MSTR/MSTU/ETH sleeve's
dataset restating its own history on every daily pull**:

* `scripts/pull_backtest_data.py` fetched on-chain series with
  blockchain.info's `sampled=true` — a ~1,500-point sampling grid **anchored
  to "now"**, so the grid shifts every day. Measured between two consecutive
  vintages: `oc_mempool_size` changed on **all 1,005 rows** (median 15.5%,
  max ~5,960%) and `oc_market_cap` on all rows (median 0.7%); every other
  column was byte-stable. The CT model's signals recompute on the shifted
  features → historical gate decisions flip → sleeve totals swing (observed:
  ETH +182.5% → +40.0%, BTC +84.5% → +58.0%).
* `mstu_synthetic_daily.csv` (pre-inception MSTU, OLS on MSTR) was **re-fit
  on every pull** as the overlap window grew — 220 historical rows changed
  per day (MSTU sleeve +587.8% → +677.0%).
* `btc_ct_engine._ensure_fresh_features` fired this pull **in-process,
  unvalidated** (bypassing `validate_refreshed_data.py`) whenever a server's
  CSV looked stale — so a reboot could silently swap the vintage.

**Which number is correct?** Neither vintage was *wrong* — both are valid
samplings of the same underlying series on different grids (verified: both
match the unsampled `hash-rate` truth exactly; mempool at daily granularity
is inherently a sampling choice). The defect was that the sampling grid
moved daily, restating history. **Decision: the current vintage
(`raw_features` sha `6745c502`, through 2026-08-02 — the one producing
Balanced ≈ +1205%) is pinned as the baseline**, because it is the freshest
(includes the newest completed bar), its synthetic-MSTU fit uses the most
real overlap data (objectively the better estimate), and reverting to an
older sampling would itself be a restatement. From here, history is frozen.

The fix, mirroring the equity-side pin philosophy:

1. **Vintage freeze in the pull** (`_freeze_history`): committed rows keep
   their pinned values; only the newest 5 rows (correction window) and
   genuinely new rows take fresh values; rows a fetch loses are restored;
   the synthetic series is fully pinned (tail=0). MSTR/MSTU actual stay
   unfrozen (a split legitimately rescales adjusted prices). Measured
   effect: the daily diff dropped from 1,006 + 220 restated lines to ≤5.
2. **Validator enforcement** (`validate_refreshed_data.py`):
   `check_frozen_history` fails the workflow if any pinned value, row or
   column changed; `check_returns_agreement` (scale-invariant) guards the
   split-adjusted MSTR/MSTU files. A deliberate re-baseline requires BOTH
   `PULL_UNFROZEN=1` and `VALIDATE_ALLOW_RESTATEMENT=1` — always an
   explicit, visible, two-step decision.
3. **In-app self-pull disabled** by default (`BTC_ALLOW_SELF_PULL=1` opt-in
   for self-hosted boxes with no other refresh path): the app consumes the
   committed, validated vintage only.
4. **Data-vintage stamp** (`overall_core.data_vintage`, shown under the
   strategy-version badge in the Overall app): every backtest figure is a
   pure function of (strategy version, BTC-vintage hash, sleeve-snapshot
   hash) — a changed number now always arrives with a changed stamp, and an
   unchanged stamp guarantees unchanged numbers.

## The dropped BTC bar (host hourly-gap → stale sleeve, 2026-08-31)

**Symptom.** The 🧭 Overall app showed **STALE SIGNALS · BTC**, the daily
audit failed (`stale ['BTC']`) and the day's Target Book was withheld
(`data/overall/daily_audit.json`: `skip_reason: signal-freshness audit failed`)
— while the ₿ Bitcoin app looked perfectly fresh.

**Root cause.** `fetch_12utc` builds each 12:00-UTC daily bar from 24 hourly
Binance klines and emits the bar only when all 24 are present. The first host
tried, `api.binance.us`, is a *separate venue* — not a mirror — and it served
no kline at all for 04:00–12:00 UTC on Aug 31 (absent from its 1h **and** 1m
series; `data-api.binance.vision` had every one of them). Eight of those hours
belong to the bar starting Aug 30, so that completed bar was dropped: the pull
at 12:16 UTC — 16 minutes after the bar closed, inside the publish run —
committed a dataset still ending Aug 29, the BTC/MSTR/MSTU/ETH sleeve stayed a
bar behind, and the audit correctly refused to publish.

**Why the Bitcoin app hid it.** That page derived its "signals generated from
…" caption (and the `record_refresh` entry the 🕵️ Daily Audit tab reads) from
the newest **hourly** candle rather than the newest daily bar it actually held.
The hourly feed was current, so the page advertised the Aug 31 7:00 AM CT close
while every daily view on it still came from the Aug 29 bar — and because an
app's own record outranks the other sources in
`freshness.freshest_signal_record`, the Daily Audit tab inherited the same
false freshness.

**The fix.**

1. **Hourly-gap healing in the pull** (`_heal_hourly_gaps`): for a bar whose
   24-hour window has already CLOSED but whose hours are incomplete, the
   missing hours are taken from the next host that has them, with the donor's
   volume **rescaled onto the primary venue's scale** (median volume ratio over
   the hours both serve). Closes agree across venues to ~0.01%; raw volumes do
   not (~300×), and `vol_chg_1` / `vol_z_20` / `vol_ma_ratio` would read an
   unscaled splice as a volume shock lasting the whole 20-bar window. A donor
   whose prices disagree, or whose scale cannot be calibrated, is refused — the
   bar is then dropped as before and the audit correctly reports BTC stale.
   Healing is limited to the freeze-refreshable tail (5 days) and to bars the
   primary venue actually traded (≥12 of 24 hours), so it can never restate
   pinned history or import a whole bar from another venue.
2. **One host owns the series.** Paging used to fall through to the next host
   on any error, silently splicing two venues' volume scales into one column.
   The first host that answers now serves the whole series; changing the
   primary venue is a deliberate re-baseline (`PULL_UNFROZEN=1`).
3. **Truthful app freshness** (`app/btc_hourly_app.py`): the caption, the
   staleness warning and the Daily Audit record all come from the newest
   completed daily bar the page holds, so the Bitcoin app now reports exactly
   the staleness the Overall app does instead of masking it.

**The same gap, in the app (2026-08-31, later the same day).** The healing
above went into the puller only. The live ₿ Bitcoin app builds its OWN
12:00-UTC bars — `_fetch_binance_hourly` → `_rebucket_12utc` — and had none of
it, so the split simply moved: by 15:30 UTC `raw_features_daily.csv` carried a
healthy 08-30 bar while the app went on dropping it and showing *"daily signal
bar is 1 bar behind"*. Two things made the app worse off than the puller:

* its host list was `("api.binance.us", "api.binance.com")` — no
  `data-api.binance.vision`. `.com` answers 451 from Streamlit Community Cloud,
  so the page had exactly ONE usable host: the thin venue with the holes, and
  the donor the healing borrows from wasn't even reachable from it.
* `_binance_get` retried the next host **per page**, so a mid-series failure
  could splice two venues' volume scales into one column — the very thing the
  puller had just been fixed to stop doing.

What kept the page from being visibly wrong all afternoon was luck:
`_seed_daily_raw_from_versioned` back-fills `_fetch_daily_raw` from
`raw_features_daily.csv`, and that CSV happens to carry every column the daily
H/L model needs, so the healed 08-30 bar arrived via the committed vintage once
the refresh job had run. That made the app's live freshness a function of a
once-daily CI job — between a bar's 12:00-UTC close and that job's commit, the
page was a bar behind every time the venue had a hole.

**The fix.** Host discipline, hourly-gap healing and the 12:00-UTC rebucket now
live in one module, `binance_bars.py`, imported by both
`scripts/pull_backtest_data.py` and `app/btc_hourly_app.py`:

* one host owns the whole hourly series on both sides (`fetch_hourly`);
* both heal the gaps in already-closed bars before rebucketing
  (`heal_hourly_gaps` / `heal_hourly_frame`), with the donor's volume rescaled
  onto the primary venue's scale exactly as above — the app heals *before* the
  Yahoo-fallback merge, so the calibration never sees a non-Binance row;
* `BINANCE_API_HOSTS` in the app **is** `binance_bars.BINANCE_HOSTS`, so the
  app's live rows and the committed vintage they get spliced onto stay on one
  venue's volume scale by construction;
* the ETH 12:00-UTC builder (`_eth_daily_12utc_bars`) gets the same treatment.

Verified live at 19:00 UTC on 2026-08-31, with `api.binance.us` still missing
04:00–11:00 UTC: the app's path yields the 08-30 bar at close 78315.43 / volume
43.162611 — identical to the committed row.

The app's "daily signal bar is N bar(s) behind" warning stays, and now means
what it says: no host had the missing hours, or the donor was refused. It is no
longer raised merely because the daily refresh job hasn't run yet.

Note one latent trap left alone: `src/fetch_binance_hourly.py` (a training-time
script that writes `data/binance_hourly_btc.csv`, absent from the repo) pulls
from `api.binance.com` — a different venue from the app's primary. Committing a
CSV from it would splice volume scales into the app's hourly history; doing so
is a deliberate re-baseline, not a routine refresh.

Tests: `tests/test_btc_bar_freshness.py`.

## The stranded pending exit (SOXX/SOXL "exits next bar" forever, 2026-08-31)

**Symptom.** 🎯 Today's action plan kept flagging **SOXX** and **SOXL** red —
`⚠️ exits next bar` — on 2026-08-29, 08-30 and 08-31, days after the executor
had actually sold them (`data/overall/executed_archive`: both filled
2026-08-28, `reason: close (no longer in book)`) and after they had dropped out
of the published Target Book (`weights` carried only XLE/OIH/ERX from the
2026-08-27 book onward). Every published book from 08-27 to 08-30 recorded the
identical row: `action: CLOSE`, `in_pos: true`, `exits_next_bar: true`.

**Root cause — a punched price series, not a stale one.**
`backtest_ticker.build_predictions` drops any bar whose ridge **features** are
incomplete (`dropna(subset=feat_cols + …)`). That is right for the H/L model,
but the trend family's SMAs (`_rolling_mean`) count bars **positionally**: each
dropped row slides every rolling window one bar further back. SOXX rallied 14
straight sessions into 2026-04-20, so `rsi_14` — whose average loss was zero —
came out NaN and five April rows were dropped. From then on the simulation's
100-day slow SMA read **521.32** where the real one read **526.44**, so
`simulate_regime` still thought the 25/100 pair was crossed **up**:

| bar | fast 25 | slow 100 | engine (`trend_long_now`, ungapped) | simulation (gapped) |
|---|---|---|---|---|
| 2026-08-26 | 524.65 | 523.01 | long | long |
| 2026-08-27 | 523.62 | **524.83** | **flat** — death cross | long |
| 2026-08-28 | 522.88 | **526.44** | **flat** | long |

`_net_decision` takes `long_now` from the ungapped frame but `in_pos` from the
simulation. With the two disagreeing, the position never closed in the sim:
`in_pos_now` stayed `True` and the decision stayed the **in-position**
`EXIT NEXT BAR — BELOW TREND` for as long as the trend stayed broken. That is
a committed pending exit, so `exits_next_bar` rode into every publish and the
action plan shaded the row red every day — a pending exit that could never
execute.

This is the *inverse* of the ARTY vintage case above: there the book ran ahead
of the engine on a data vintage; here the engine's own two halves disagreed on
the same bar, so nothing flagged it.

**The fix.**

1. **One trend signal, computed on the contiguous history**
   (`backtest_ticker.build_predictions`): the long/flat array is derived once
   from the full input frame — the same `trend_long_array` that
   `trend_long_now`, the trend chart and `live_exit_keys` read — and carried on
   the surviving rows as a `trend_long` **column** (a column, not frame
   metadata, so it survives slicing and copying). `simulate_regime` uses it
   whenever it is present; an explicit `ma_window` (the sweep) still overrides.
   The simulated position and the displayed decision can no longer disagree,
   whatever the feature dropna removes. It also warms the SMAs up properly at
   the start of the prediction window — previously the first `ma_slow` rows ran
   on truncated partial windows, which is why XLE's `crash_dd` 52-week high was
   also mis-scaled (its OOS trade count drops 10 → 3, return 220.3% → 210.0%,
   as the drawdown gate now measures a true 52 weeks).
2. **RSI is defined without a down bar** (`ticker_core._rsi`, `gldm_core._rsi`):
   zero average loss is RSI **100**, not NaN (zero average gain is 0; a window
   that never moved is 50). This removes the drop at its source — SOXX and ARTY
   now lose no bars at all.

SOXX/SOXL read `FLAT — BELOW TREND` / `STAND ASIDE` on the 2026-08-28 close
after the fix, matching the executed book and the vacated Target Book.

Tests: `tests/test_trend_signal_alignment.py`.

## What can still move intraday (by design)

* The **live row** (in-progress bar), spot prices, and today's action plan —
  the live cockpit is supposed to tick.
* The BTC sleeve's *intraday* reads (Binance spot/hourly top-ups); its
  backtests run from the committed vintage.
* After the 4 PM ET close, the day's new session is validated and appended —
  numbers legitimately update once per trading day.
