# Streamlit Community Cloud — memory analysis and remediation plan

**Symptom.** `https://btc-range-model.streamlit.app/` is being throttled by
Streamlit Community Cloud with *"Your app has gone over its resource limits …
It's using too much memory!"*  Community Cloud gives **one container with ~1 GB
of RAM** that serves **every viewer of the app from a single Python process**.

This document records what was measured, what the actual drivers are, and a
prioritised set of fixes ordered by (memory saved) ÷ (risk to behaviour).

---

## 1. Measured baseline

Measured on this repo's pinned stack (Python 3.12, numpy 2.2.6, pandas 2.2.3,
scipy 1.14.1, scikit-learn 1.8.0, streamlit 1.63), peak RSS after each step:

| Step | Δ RSS | Cumulative |
|---|---:|---:|
| interpreter start | — | 8 MiB |
| `import numpy` | +17 | 25 MiB |
| `import pandas` | +79 | 104 MiB |
| `import scipy.stats` | +59 | 163 MiB |
| `import sklearn.ensemble` | +24 | 187 MiB |
| `import streamlit` | +32 | 218 MiB |
| `import plotly.graph_objects` | +0 | 218 MiB |
| `import yfinance` | +16 | **235 MiB** |
| `joblib.load` — all 4 models in `models/` | +14 | 249 MiB |
| `read_csv data/macro_hourly_cache.csv` | +28 | 277 MiB |

**~235 MiB is gone before the app does anything** — about a quarter of the
budget, and essentially irreducible given the scientific stack.  Everything the
app itself does has to fit in the remaining ~750 MiB, shared by all viewers.

Two things measured *smaller* than expected and are therefore **not** the
problem:

* **Compiled sub-app code objects** (what `streamlit_app.py::_compiled_app`
  caches) retain **12.8 MB in total** for all ten sub-apps — `btc_hourly_app.py`
  alone is 6.3 MB. The router's `max_entries=32` is harmless in production,
  where mtimes never change.
* **`data/macro_hourly_cache.csv`** is 11 MB on disk but only a **5.9 MB**
  DataFrame (17 455 × 41, all `float64`).  Its 28 MiB footprint above is the CSV
  parser's *transient* buffers, not retained memory.

---

## 2. Root causes, in order of impact

### 2.1 `st.tabs` renders every tab body on every rerun ← **the dominant cost**

`app/btc_hourly_app.py` opens a **12-tab** top-level group (search for
`tab_live, tab_hist,`):

```
Live · Historical replay · BTC Backtesting · MSTR Backtesting · MSTU Backtesting
· MSTR-MSTU Plot · ETH Backtesting · MSTR Options · MSTU Options · Explainability
· Leading Indicators · Retrain Status
```

(**MSTR-MSTU Plot** is the cheapest of the twelve — one bounded
`st.cache_data(ttl=3600, max_entries=4)` frame of two daily close series and one
Plotly figure. It still pays the always-render tax below, like every other tab.)

Streamlit's `st.tabs` is **not lazy**: it is a layout container, and the `with
tab_x:` body of *every* tab executes on *every* script run.  Tab selection only
changes which pane the browser shows.  So each rerun of the BTC app:

* runs **six** backtest suites × three periods each (`_btc_bear/_bull/_full`,
  `_mstr_*`, `_mstu_*`, `_eth_*`, `_mstu_opts_*`, `_opts_*` — lines 17140–17500),
* builds the explainability and leading-indicator panels,
* materialises up to **37 Plotly figures** and 10 dataframes,

even for a viewer who only ever looks at "Live".  There are **15 `st.tabs`
groups** in that one file, several nested inside the backtest tabs.

The caches make this *fast*, but they make it *memory-expensive* — see 2.2.

### 2.2 Caches are unbounded, and `st.cache_data` stores a pickled copy

Verified against the installed Streamlit source:

* `st.cache_data` stores each value **pickled, as `bytes`**, in a
  `TTLCache(maxsize=max_entries, ttl=ttl)`.
* **`max_entries` defaults to `None` → `math.inf`.**  Across `app/` there are
  **91 cached functions and not one `max_entries`.**
* Every *call* returns a **freshly unpickled copy** — so a cached DataFrame
  costs its pickled bytes permanently *plus* a full new allocation per call
  site per rerun.
* Streamlit's `TTLCache` docstring is explicit: *"Expired entries are reported
  as absent on reads but are **only actually removed by a write**, by
  `expire()`, or by a length/size query."*  An entry whose key is never written
  again keeps its bytes in RAM after its TTL has passed.

**17 `@st.cache_data` functions have no `ttl` at all** (`btc_hourly_app.py`
lines 251, 3876, 4417, 4426, 4455, 4796, 4813, 5161, 5180, 5446, 5464, 5820,
5849, 6182, 6199, 6550; `executed_book_app.py:401`).  Twelve of them are the
backtests, which return dicts of multi-year Series plus full trade logs.  Those
entries live for the **entire process lifetime**.

### 2.3 Every backtest result is cached **twice**

Six pairs follow this shape:

```python
@st.cache_data(show_spinner="Running MSTR backtest …")          # copy #1
def run_mstr_backtest(...): ...

@st.cache_data(show_spinner="Loading fixed-period MSTR backtest …")   # copy #2
def _run_fixed_period_mstr_backtest(...):
    return run_mstr_backtest(...)          # ← already cached
```

The wrapper pickles and stores a second, identical copy of a payload the inner
function has already stored.  With 6 assets × 3 fixed periods that is **36
cached backtest payloads where 18 would do**.

### 2.4 One process accumulates the caches of *all sixteen* apps

`streamlit_app.py` routes OVERALL / BTC / GLDM / GDXM / 7 tickers / LEVERAGED /
DAILYAUDIT / HEALTH / TARGETBOOK / EXECUTEDBOOK / ASSISTANT through a single
container.
`st.cache_data` and `st.cache_resource` are **global to the process**, not
per-app and not per-session.  A viewer who tours the sidebar leaves behind the
resident cache of every app they touched, for the rest of the container's life.

### 2.5 The OVERALL app rebuilds itself every 45 s, forever

`app/overall_app.py:111` sets `_AUTOREFRESH_SECS = 45` and drives
`st_autorefresh`.  Every 45 seconds, **for every open browser tab**, Streamlit
re-executes all 4 886 lines of `overall_app.py`, unpickles the
`get_all_profiles` payload (returns matrix, position matrix, and a full
walk-forward gated replay **per risk profile**), and re-serialises 21 Plotly
figures.

Sustained allocate/free churn at that cadence is what turns a bounded working
set into a rising RSS: CPython's allocator and glibc's `malloc` rarely return
freed arenas to the OS, so the process **ratchets upward for hours and never
comes back down** — exactly the profile behind "reboot fixes it for a while".

### 2.6 `run_universe()` fans out 10 threads at once

`app/overall_core.py:661`:

```python
with ThreadPoolExecutor(max_workers=len(cfgs)) as ex:    # len(PARENT_KEYS) == 10
```

All ten instruments build their full daily datasets **concurrently**, so ten
`macro_daily.csv`-sized frames (1–2.6 MB on disk, several × that live, plus
`data_gate` validation copies) are in flight simultaneously.  It is the peak,
not the average, that gets a container killed.

### 2.7 Secondary: Plotly payloads and the ForwardMsg cache

Streamlit caches every delta message larger than `global.minCachedMessageSize`
(10 KB) for `global.maxCachedMessageAge` (2 script runs), per session.  Charts
built on the full hourly history (17 455 points) serialise to multi-MB JSON.
A display screen is ~1 000 px wide, so those points are not visible anyway.

---

## 3. Recommendations

### Tier 1 — no behavioural change, no numeric change

| # | Change | Where | Expected saving |
|---|---|---|---|
| 1 | Add `max_entries` to **every** `@st.cache_data`. Bucket-keyed functions → `max_entries=2`; parameterised backtests → `max_entries=4`–`8`. | all 91 sites in `app/` | Caps cache growth. Also forces the `TTLCache` write path that actually evicts expired entries. |
| 2 | Drop `@st.cache_data` from the six `_run_fixed_period_*` wrappers — the inner `run_*_backtest` is already cached on the same arguments. | `btc_hourly_app.py` 3876, 4796, 5161, 5446, 5820, 6182, 6550 | Halves ~36 backtest payloads to ~18. |
| 3 | Give the 17 no-TTL caches an explicit `ttl` (e.g. `ttl=6*3600` for backtests, which only change when the model or dataset does — both already in the key). | as listed in §2.2 | Lets long-idle payloads expire instead of living forever. |
| 4 | Cap the fan-out: `ThreadPoolExecutor(max_workers=4)`. These tasks are network-bound; 4 concurrent Yahoo fetches still saturate the link. | `overall_core.py:661` | Cuts the `run_universe` **peak**, which is what trips the limit. |
| 5 | ~~Downsample display-only series before handing them to Plotly.~~ **Withdrawn — the premise was wrong, see §3.1.** | — | — |
| 6 | Deployment config — see §4. | `.streamlit/config.toml` | Frees disconnected sessions in 30 s instead of 120 s; halves big-delta retention; drops the file-watcher threads. |
| 7 | Remove `nbformat` from `requirements.txt` — nothing under `app/` imports it. | `requirements.txt` | Smaller install; no runtime effect. |

### 3.1 Correction: the charts are not the problem

The original item 5 assumed the Plotly figures were built on the full 17 455-point
hourly history.  They are not, and it was withdrawn rather than implemented:

* the live hourly chart plots `LOOKBACK_HOURS = 24` — twenty-four points;
* the backtest equity curves are **daily**, ~840 points over the 2024-05-26 →
  today window;
* measured, a representative backtest figure (840 points × 4 traces) serialises
  to **150 KB** of JSON, and even a hypothetical full-hourly trace is only 189 KB.

Downsampling to "~2 000 points" would therefore have been a no-op on every chart
in the app, and anything more aggressive would have visibly degraded the equity
curves for no gain.  The real lever on chart payloads is how many *generations*
of them Streamlit retains, which is `global.maxCachedMessageAge` in §4 — that
one did ship.

---

## 3.2 What was actually implemented (Tier 1)

Verified: **655 tests pass, identical to the pre-change baseline.**

* **82 cache decorators given `max_entries`**, sized per function to its measured
  per-rerun argument cardinality rather than by a blanket rule.  This matters:
  `compute_day_type_forecast` is called once per calendar day in
  `compute_alltime_daytype_metrics`' loop — ~400 distinct keys and growing.  A
  blanket cap of 32 there was measured to cause **400 recomputes on every
  rerun**; at 1024 the second rerun recomputes **zero**.  Bulk data loaders and
  bucket-keyed functions get 2, backtests 8, per-date functions 256–1024.
* **6 duplicate cache layers removed** — the `_run_fixed_period_*` forwarders.
  The seventh (`_run_fixed_period_backtest`, line 3876) **kept** its decorator:
  it forwards to `run_full_period_backtest`, which is *not* cached, so it is the
  only cache layer there rather than a duplicate.
* **A 24-hour TTL added to the 12 backtest caches.**  Deviation from item 3 as
  written: the other no-TTL caches (`_training_cutoffs`,
  `_backtest_dataset_version`, `_backtest_dataset_mtime`,
  `executed_book_app._records`) are tiny values keyed on file mtimes that already
  self-invalidate, so a TTL there would buy no memory and only force needless
  recomputes.  They got `max_entries` alone.
* **`run_universe` fan-out capped** at 4 workers.
* **Router compiled-code cache** lowered from 32 entries to 12 (there are ten
  sub-apps).
* **`.streamlit/config.toml`** — the three settings in §4.
* **`nbformat` dropped** from `requirements.txt`, with a pointer comment: it is
  unused by the app and by CI, but `notebooks/builders/` still needs it.

### Still open — noticed while implementing

`run_full_period_backtest` is called **uncached** at `btc_hourly_app.py:13845`,
so a full backtest is recomputed on *every* rerun of the BTC app.  That is a
latency and peak-memory cost on every refresh.  It was left alone because adding
a cache there is a behaviour change, not a bound — but it is worth a look.

---

### Tier 2 — the real fix for §2.1, small UX change

**Make the heavy tab groups lazy.**  Because `st.tabs` always runs every body,
the only way to stop computing seven backtest suites for a viewer reading
"Live" is to *select* the section rather than *lay it out*:

```python
_SECTIONS = ["🔴 Live", "🕒 Historical replay", "₿ BTC Backtesting", ...]
_sel = st.segmented_control("", _SECTIONS, default=_SECTIONS[0],
                            key="btc_section", label_visibility="collapsed")
if _sel == "🔴 Live":
    render_dashboard(...)
elif _sel == "₿ BTC Backtesting":
    ...          # only this branch runs
```

* **Memory:** one section's working set per rerun instead of eleven.  This is
  the single largest win available.
* **Speed:** *faster*, not slower — the first paint stops waiting on 18
  backtests the viewer did not ask for.
* **Cost:** switching sections becomes a rerun rather than an instant
  client-side tab flip. With the caches warm that is sub-second, and each
  section keeps its own cached result.  `st.segmented_control` renders as a
  pill row, so it reads much like the tab strip it replaces.

A lighter-touch variant, if you would rather keep `st.tabs` exactly as it
looks: wrap each heavy tab body in `@st.fragment` and gate the compute behind a
"Load this backtest" button, so the expensive work happens on demand and
subsequent interaction inside the tab reruns only that fragment.

### Tier 3 — architectural, zero code risk, highest leverage

**Split the router into 2–3 separate Community Cloud apps from the same repo.**
Community Cloud allocates ~1 GB **per app**, not per account.  Three entry
points (e.g. `streamlit_app.py` for OVERALL + books, `btc_app.py` for
BTC/GLDM/GDXM, `tickers_app.py` for the generic tickers) give you ~3 GB of
total headroom, and each process only ever pays the baseline plus its own
caches.  The sidebar can carry links between them.  No app logic changes at
all — only which file each deployment points at.

### Not recommended

**Downcasting `float64` → `float32`** would halve the pickled size of the big
cached frames (`macro_hourly_cache` goes 5.9 MB → 3.0 MB, measured).  For a
**trading model** this is the wrong trade: float32 carries ~7 significant
digits, and compounded returns, drawdown paths and optimiser weights would shift
in the low-order digits versus every published backtest number in the repo's
`*_EVAL.md` documents.  If you want it, apply it **only** to display copies
handed to Plotly, never to anything feeding signal or portfolio maths.

---

## 4. `.streamlit/config.toml` additions

All defaults verified against the installed Streamlit; none change app logic.

```toml
[server]
# A deployed container never edits its own source — drop the watchdog
# threads and inotify handles. (default: "auto")
fileWatcherType = "none"

# Reclaim a closed browser tab's session state and its ForwardMsg
# references after 30s instead of 120s. (default: 120)
disconnectedSessionTTL = 30

[global]
# Hold large delta messages (the Plotly payloads) for one script run
# rather than two. Costs a little repeat bandwidth, halves retention.
# (default: 2)
maxCachedMessageAge = 1
```

`runner.postScriptGC` already defaults to `true`, so a `gc.collect()` after
every script run is in place.

---

## 5. Suggested order of work

1. ~~**Tier 1 items 1–4 and 6–7.**~~ **Done** — see §3.2. Redeploy and watch the
   memory graph in *Manage app*.
2. ~~**Tier 1 item 5**~~ — withdrawn, see §3.1.
3. **Tier 2** — lazy sections in `btc_hourly_app.py`. Largest single win, and
   now the largest remaining one.
4. **Tier 3** — split the deployment if the app keeps growing.

Reboot the app after deploying so the ratcheted RSS from §2.5 starts clean.

---

## 6. Appendix — how the measurements were taken

* RSS figures: `resource.getrusage(RUSAGE_SELF).ru_maxrss` between import steps.
* Retained code-object sizes: `tracemalloc.get_traced_memory()` around
  `compile(path.read_text(), path, "exec")` per sub-app.
* DataFrame sizes: `df.memory_usage(deep=True).sum()` and `len(pickle.dumps(df))`.
* Cache semantics: read directly from the installed
  `streamlit.runtime.caching.cache_utils`,
  `…caching.ttl_cache` and
  `…caching.storage.in_memory_cache_storage_wrapper`.
* Config defaults: `streamlit.config._config_options_template`.
