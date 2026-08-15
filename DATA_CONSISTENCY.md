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

## What can still move intraday (by design)

* The **live row** (in-progress bar), spot prices, and today's action plan —
  the live cockpit is supposed to tick.
* The BTC sleeve's *intraday* reads (Binance spot/hourly top-ups); its
  backtests run from the committed vintage.
* After the 4 PM ET close, the day's new session is validated and appended —
  numbers legitimately update once per trading day.
