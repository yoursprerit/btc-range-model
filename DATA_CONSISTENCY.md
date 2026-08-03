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
  regression versus the snapshot, and per-day return agreement with the
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

### Tests

`tests/test_data_gate.py` covers the checks (missing/dead columns, extreme
moves, regression, rewritten history), the pin/refetch/fallback flows, the
completed-vs-in-progress split, snapshot tamper detection, and the audit
trail.

## What can still move intraday (by design)

* The **live row** (in-progress bar), spot prices, and today's action plan —
  the live cockpit is supposed to tick.
* The BTC sleeve's *intraday* reads (Binance spot/hourly top-ups); its
  backtests run from the committed vintage.
* After the 4 PM ET close, the day's new session is validated and appended —
  numbers legitimately update once per trading day.
