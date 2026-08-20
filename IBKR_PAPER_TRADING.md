# IBKR Paper Trading — Daily Rebalancer

Connect the **Overall Trading** strategy signals to an Interactive Brokers
**paper** account and rebalance it once per trading day. The paper account is
driven to the same allocation the Overall Streamlit app shows as **"Recommended
Live Possible Targetbook"** — no signal logic is re-implemented; everything
calls the same `overall_core` engine.

> **Paper only.** Every tool here refuses to run against any account whose id
> does not start with `DU` (IBKR's paper prefix) unless you explicitly pass
> `--allow-nonpaper`. `--dry-run` (no orders) is the default.

## Two topologies

| | **A — all-in-one** (`ibkr_rebalance.py`) | **C — publish/execute** (`publish_target_book.py` → `ibkr_execute_book.py`) |
|---|---|---|
| Where the model runs | On the trading host | In the cloud (GitHub Action / any host); **not** on the trading host |
| Trading host needs | full model stack + `ib_async` | only `ib_async` (+ pandas) — lightweight |
| Decision transport | none (same process) | a signed `target_book.json` artifact |
| Best when | one box does everything | you want the heavy model run off the trading box, auditable, and split from execution |

Both drive the account identically. **Option C** is documented in its own
section below; the setup (gateway, IBC, cron) is shared.

> **On a Windows laptop?** See **[`IBKR_OPTION_C_WINDOWS.md`](IBKR_OPTION_C_WINDOWS.md)**
> for the complete Windows walkthrough — Python 3.12, IB Gateway + IBC, the
> `scripts\ibkr_execute_daily.ps1` wrapper, and Task Scheduler automation.
>
> **On a free cloud VM (no laptop)?** See **[`docs/CLOUD_EXECUTOR.md`](docs/CLOUD_EXECUTOR.md)**
> — headless IB Gateway via Docker (`deploy/ibkr-gateway/`), the
> `scripts/ibkr_execute_daily.sh` wrapper, and cron on an Oracle Always-Free VM.
>
> **Going live (real money) alongside paper?** See **[`docs/LIVE_TRADING.md`](docs/LIVE_TRADING.md)**
> — `--account-mode live` with a pinned-account guard, exposure / per-order caps,
> a kill switch, and a second live gateway. Paper stays the default and runs in
> parallel.

---

## How it works

```
overall_core.run_universe()                       every strategy, live
      → optimize_weights(… Aggressive …)          app-default risk blend
      → fetch_spot() + apply_spot()               live prices overlaid
      → live_exit_keys(include_entries=True)      drop trend-broken names
      → signal_gated_allocation(force_exit=…)     today's target weights
```

- **Targets** are portfolio weights per signal key. The undeployed remainder is
  **held as cash** (no SATA leg in the live book).
- **BTC → IBIT.** IBKR has no spot-Bitcoin product, so the BTC signal sleeve is
  traded via the IBIT spot-Bitcoin ETF. Every other name is a US-listed
  ETF/equity traded under its own ticker. See `scripts/ibkr_symbols.py`.
- **Sizing** uses each traded instrument's own live quote (the BTC sleeve is
  sized on IBIT's price, not spot BTC).
- **Reconcile** against current IBKR positions with a **no-trade band**
  (default 1% of net-liq) to suppress churn, then place **market orders
  sells-first** so freed capital funds the buys.

The full universe: BTC(→IBIT), ETH(→ETHA), MSTR, MSTU, GLDM, GDX, UGL, NUGT, SOXX, SOXL,
GRID, XLE, OIH, ERX, REMX, WGMI, PBW, ARTY.

---

## Prerequisites — do these before ANY setup script

None of the setup tooling in this repo can create or configure an IBKR account:
the Windows script and the cloud compose file only *consume* credentials you
already hold. Work through this list first, on **every** topology (Windows
laptop, cloud VM, Option A or C). Client Portal menu labels move between IBKR's
UI revisions — the items matter, the exact paths may not match verbatim.

### 1. An IBKR paper account

1. **Create the paper account** — Client Portal → *Settings* → *Account
   Settings* → **Paper Trading Account**. IBKR generally requires an approved
   live account before it will issue a paper one.
2. **Record the paper username and set its password.** The paper login is a
   **separate username** from your live one with an independently set password.
   These are the credentials every setup path here asks for (`-IbUser` /
   `-IbPassword` on Windows, `TWS_USERID` / `TWS_PASSWORD` in the Docker `.env`).
3. **Confirm the account id starts with `DU`.** This is enforced, not cosmetic:
   `PAPER_ACCT_PREFIX = "DU"` in `scripts/ibkr_common.py` aborts the run on any
   other account, so a live login is refused rather than traded.

### 2. Disable 2FA on the paper login (required for automation)

IBKR's daily two-factor prompt is what breaks unattended login. A scheduled task
or timer cannot answer a phone tap, so it hangs and the rebalance silently
misses. Use a **standalone paper login with 2FA disabled** — ideally one with no
live trading permission attached, since IBC stores its password in plaintext
(see below). If you must keep 2FA, follow the IBC second-factor docs before
automating anything.

### 3. Share market data with the paper account (recommended)

Client Portal → *Settings* → **Market Data Subscriptions**. Without live quotes
the executor cannot price its marketable limits: each affected leg logs a
`WARN … falling back to MARKET` (`scripts/ibkr_common.py`) and trades as an
unprotected market order. It still trades — you just lose the price ceiling. If
you would rather not share data, widen the cap instead
(`--slippage-cap 0.015` / `IBKR_SLIPPAGE_CAP=0.015`).

### 4. The shared signing secret

`OVERALL_BOOK_SECRET` must be **the same value** in three places, and only one
of them signs:

| Role | Where it lives | Does |
|---|---|---|
| **Publisher** | GitHub **repo secret** (used by `publish-target-book.yml`) | **signs** |
| Streamlit app | `st.secrets`, else env | verifies |
| Executor host | env var (`setx` on Windows) | verifies |

A mismatch aborts the run at signature verification. Have the value in hand
before setup — for Option C the executor is useless without it. (Option A
computes signals locally and needs no secret.)

Two traps worth knowing before you start:

- **GitHub Actions secrets are write-only.** You cannot read the current value
  back out, so if it isn't recorded somewhere, rotation is the only path.
- **Changing the secret does not re-sign the committed book.** The signature is
  fixed at publish time, so any rotation must be followed by a fresh publish.
- **Setting the value in Streamlit changes nothing about signing** — Streamlit
  is a verifier. Windows walkthrough:
  [`IBKR_OPTION_C_WINDOWS.md` §4](IBKR_OPTION_C_WINDOWS.md#4-set-the-shared-secret).

### 5. Accept the plaintext-password trade-off

IBC stores the paper password unencrypted in its `config.ini` (Windows) or the
compose `.env` (cloud). That is IBC's design, not a choice this repo makes. It
is the main argument for a paper-only login that carries no live permissions.

> **Windows shortcut:** after setup, `setup_windows_option_c.ps1 -Preflight`
> mechanically checks every link in the chain (including whether your secret
> actually signs the book on disk), and `-Rehearse` runs the real scheduled
> task end to end without placing an order. Both exit non-zero on failure.

**Only once all five are done** should you run
`scripts\setup_windows_option_c.ps1` (Windows — see
[`IBKR_OPTION_C_WINDOWS.md`](IBKR_OPTION_C_WINDOWS.md)) or bring up the Docker
gateway (cloud — see [`docs/CLOUD_EXECUTOR.md`](docs/CLOUD_EXECUTOR.md)).

---

## One-time setup

### 1. Python environment (on the host that will run IB Gateway)

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt        # the strategy engine
.venv/bin/pip install -r requirements-ibkr.txt   # + ib_async broker layer
```

### 2. IB Gateway (paper)

1. Download **IB Gateway** (lighter than TWS):
   <https://www.interactivebrokers.com/en/trading/ib-api.php>
2. Log into your **paper** account (or a dedicated paper username).
3. **Configure → Settings → API → Settings**:
   - ☑ Enable ActiveX and Socket Clients
   - **Socket port `4002`** (IB Gateway paper default)
   - Add `127.0.0.1` to **Trusted IPs** (avoids the manual accept popup)
   - ☐ Read-Only API must be **unchecked** (orders need write access)

### 3. Unattended auto-login (IBC) — for the automated daily run

IB Gateway forces a daily restart and needs login each time; **IBC** automates
that so cron can rely on a live session.

1. Install IBC: <https://github.com/IbcAlpha/IBC>
2. Copy `deploy/ibc/config.ini.example` to your IBC config (e.g.
   `~/ibc/config.ini`) and fill in credentials + `TradingMode=paper`.
   **Do not commit the filled-in file** — it holds your password.
3. Launch IB Gateway under IBC at boot (and/or on a schedule shortly before the
   rebalance window). Set `AutoRestartTime` to the small hours so a restart
   never lands mid-rebalance.

> **2FA tip:** unattended login is simplest with a paper username that has 2FA
> disabled. If you must use IBKR Mobile 2FA, follow the IBC docs for
> second-factor handling.

---

## Running it

**Preview (default — places no orders).** Works even without a gateway; with one
connected it also diffs against your live positions:

```bash
.venv/bin/python scripts/ibkr_rebalance.py
```

**Execute against paper:**

```bash
.venv/bin/python scripts/ibkr_rebalance.py --execute
```

Useful flags:

| Flag | Default | Meaning |
|------|---------|---------|
| `--execute` | off (dry-run) | actually transmit orders |
| `--profile` | `Balanced` (app default) | `Balanced` / `Growth` / `Aggressive` |
| `--band` | `0.01` | no-trade band as a fraction of net-liq |
| `--fractional` | off | allow fractional shares (default: whole shares) |
| `--port` | `4002` | IB Gateway API port (paper) |
| `--fill-timeout` | `60` | seconds to wait for each order leg to fill |
| `--force` | off | ignore the weekend/holiday & stale-signal guards |
| `--allow-stale-bar` | off | trade a book whose bar is not the last completed session |
| `--allow-margin` | off | **danger** — permit buys beyond cash + realised sell proceeds |
| `--max-gross-frac` | `1.02` | abort a plan landing above this multiple of net-liq |
| `--max-turnover-frac` | `1.5` | abort a plan trading above this multiple of net-liq |
| `--max-price-drift` | `0.25` | abort when a name quotes this far from its sizing price |
| `--outside-rth` | off | allow the run outside 09:30–16:00 ET |
| `--allow-nonpaper` | off | **danger** — disable the paper-account guard |

The Option-A rebalancer carries the same limits as the Option-C executor, from
the same code — see [Pre-flight guards](#pre-flight-guards-what-the-executor-refuses-to-do).

### Recommended validation before trusting automation
1. `scripts/ibkr_rebalance.py` (dry-run) — eyeball the target book against the
   Overall app's "Recommended Live Possible Targetbook" panel; they should match.
2. Connect the gateway and dry-run again — check the order plan (deltas vs your
   current paper positions) looks sane.
3. `--execute` once by hand during US market hours; confirm fills and that
   resulting positions match the targets.
4. Only then enable the cron job.

---

## Automating the daily run

The market timing: signals are computed on completed daily bars and act on the
next bar, so the rebalance runs **2:30 PM US Central** (3:30 PM ET) on trading
days — inside regular trading hours, 30 minutes before the 3:00-PM-CT /
4:00-PM-ET close, so the fills land near the close the engine books against.

`scripts/ibkr_daily.sh` wraps a single `--execute` run with logging. Point cron
at it (the script and the Python guards both skip weekends/holidays):

```cron
CRON_TZ=America/Chicago
30 14 * * 1-5  /path/to/repo/scripts/ibkr_daily.sh >> /path/to/repo/logs/ibkr_cron.log 2>&1
```

`CRON_TZ=America/Chicago` keeps the slot at 2:30 PM Central through the CDT/CST
switch regardless of the host clock.

Override behaviour via env vars (see the script header): `IBKR_PYTHON`,
`IBKR_PROFILE`, `IBKR_BAND`, `IBKR_PORT`, `IBKR_EXTRA`.

Ensure IB Gateway (under IBC) is up **before** 2:30 PM CT and that its
`AutoRestartTime` sits well outside the rebalance window.

---

## Option C — publish in the cloud, execute locally

Split the decision from the execution: a **publisher** runs the model once (in
the cloud) and emits a small **signed JSON target book**; a lightweight
**executor** next to IB Gateway consumes it and trades. The trading host never
runs the model.

> **Architecture diagram:** [`docs/option_c_architecture.md`](docs/option_c_architecture.md)
> shows every step from the cloud weight engine to IBKR paper order placement
> (renders natively on GitHub; a styled standalone version is in
> [`docs/option_c_architecture.html`](docs/option_c_architecture.html)).

```
 publisher (cloud / GitHub Action)          executor (host with IB Gateway)
 scripts/publish_target_book.py             scripts/ibkr_execute_book.py
   run_universe → optimize → gate             load book → verify signature
   → target_book.json  ──── transport ────►   → validate freshness
   (HMAC-signed)        (git commit / URL)     → diff vs positions → trade paper
```

### The artifact (`data/overall/target_book.json`)

Schema `overall-target-book/v1` (see `app/target_book.py`): `as_of`,
`generated_at_utc`, `profile`, `weights`, `cash_weight`, `exec_price`,
trimmed `actions`, and an optional `signature` (HMAC-SHA256). It is
**self-contained** — the executor sizes orders from `weights` × net-liq using
the book's own `exec_price`, so it makes no market-data calls.

### Signing (recommended)

Because the transport may be public (a branch commit, a raw URL), set a shared
secret so the executor can prove the book is authentic before trading:

```bash
export OVERALL_BOOK_SECRET="a-long-random-string"   # same value on both sides
```

On the publisher it signs the artifact; on the executor it verifies (a mismatch
or a tampered book aborts the run). Pass `--require-signature` to the executor to
refuse an unsigned book outright.

> **If the secret is unset, verification is skipped, not failed.**
> `verify_signature()` returns success with the message
> `signed but no secret provided to verify` and the run continues. Check the
> executor's `Signature:` line — only `signature OK` means the book was actually
> verified.

### Publish

Run anywhere with the model stack installed (`requirements.txt`):

```bash
OVERALL_BOOK_SECRET=… python scripts/publish_target_book.py --profile Aggressive
# writes data/overall/target_book.json
```

Or let the **GitHub Action** do it: `.github/workflows/publish-target-book.yml`
runs the publisher and commits the artifact back to the branch. The scheduled
cron fires **once every day at ≈7:15 AM US Central** (≈15 min after the
7:00-AM-CT Bitcoin bar close), weekends and US market holidays included —
Bitcoin trades continuously, so the book is refreshed daily; only the
*executor* skips non-trading days. The daily audit runs **once**, before the
book is written, and the published book then stays **frozen until the next
morning's cycle**.

> **On-time delivery:** GitHub's `on: schedule` is best-effort — fires arrive
> minutes-to-hours late, and some days every slot is dropped (2026-08-03).
> The punctual 7:16-AM-CT fire therefore comes from an external cron-job.org
> job hitting the `workflow_dispatch` API with a fine-grained PAT — setup in
> [`docs/EXTERNAL_SCHEDULER.md`](docs/EXTERNAL_SCHEDULER.md). The workflow's
> own cron slots remain as same-day backup.

The **only** way to replace the frozen book intraday is to publish **on demand
from the app**: the 📋 Target Book page has a **🚀 Publish new target book**
button that dispatches the same GitHub Action via `workflow_dispatch`, so a
fresh signed book is computed and committed without leaving the UI (the
outgoing book is rotated to `target_book*_prev.json`). It needs a `GITHUB_TOKEN` in Streamlit secrets (or env) — a
fine-grained PAT with **Actions: read & write** on this repo. Optional secrets:
`GITHUB_REPO` (`owner/repo`, auto-detected otherwise) and `GITHUB_PUBLISH_REF`
(branch to run/commit on, default `main`).

> **Scheduling caveat:** GitHub runs `on: schedule` **only from the repository's
> default branch**. While this workflow lives only on the feature branch, trigger
> it **manually** (Actions tab → *Run workflow* → pick the branch) or via the
> API. The commented `schedule:` block activates only once the workflow is on the
> default branch. Add `OVERALL_BOOK_SECRET` as a repo/environment secret to sign.

### Execute

On the host running IB Gateway (needs only `requirements-ibkr.txt`):

```bash
# preview (default — no orders):
python scripts/ibkr_execute_book.py --file data/overall/target_book.json

# from the raw artifact URL on the branch:
python scripts/ibkr_execute_book.py --url https://raw.githubusercontent.com/<owner>/<repo>/<branch>/data/overall/target_book.json

# actually trade paper:
OVERALL_BOOK_SECRET=… python scripts/ibkr_execute_book.py --file data/overall/target_book.json --execute
```

If you use the git-commit transport, the executor host just does `git pull` (to
get the latest committed book) before running with `--file`.

Executor-specific flags: `--file` / `--url` / stdin (source), `--max-age-hours`
(reject a book generated too long ago, default 36 — spans the 7:15-AM-CT publish
anchor to the next day's 2:30-PM-CT executor slot), `--require-signature`. The
`--execute`, `--band`, `--fractional`, `--port`, `--allow-nonpaper`, `--force`
flags behave exactly as in the all-in-one rebalancer.

---

## Order routing

Both entry points place orders through the same `Broker.place()` in
`scripts/ibkr_common.py`, so paper and live route identically — only the safety
guards differ. Pick the order type with `--order-type` (env `IBKR_ORDER_TYPE`
via the wrapper scripts):

| `--order-type` | What is sent | When to use it |
|---|---|---|
| `marketable-limit` **(default)** | A LIMIT priced `--slippage-cap` **through** the touch — a BUY at 0.5% above the ask, a SELL at 0.5% below the bid | Everyday default. Fills like a market order in a normal book, but a stale quote, a flash dislocation or a bad print can't cost more than the cap |
| `moc` | Market-on-close — filled in the official 4:00 PM ET closing auction | Tightest tracking to the backtest, which books fills at the close (see below) |
| `market` | An unprotected market order (the pre-2026-08 behaviour) | Escape hatch; no price ceiling |

**Why a marketable limit rather than a plain limit.** A resting limit that
doesn't fill leaves the account holding the *wrong* exposure until tomorrow's
run — an open-ended tracking error, far worse than paying a spread. So the
limit is priced to cross immediately, and anything it still fails to fill
inside `--fill-timeout` is **cancelled and re-sent as a market order in the
same run** (partial fills escalate only the remainder). The ceiling therefore
costs nothing in fill certainty — it only ever caps the first attempt. If a
leg can't be priced at all (no market-data subscription, dead quote) it falls
back to a market order and says so in the log.

**Why MOC is worth considering.** Every backtest in this repo books fills at
the **close**; the executor trades at 2:30 PM CT, 30 minutes earlier, so live
results carry a structural 30-minute drift against those numbers. MOC removes
it, and the closing auction is the deepest, tightest liquidity of the day. The
trade-offs are real:

- Orders must be entered before the **15:50 ET** exchange cutoff. The
  2:30-PM-CT (3:30 PM ET) slot clears it by 20 minutes — this is exactly why
  MOC wasn't an option at the old 8:45-AM-CT slot. Past the cutoff the executor
  automatically falls back to a marketable limit rather than eat a rejection.
- **Nothing fills until 4:00 PM ET**, so the executor holds the connection for
  ~32 minutes to write a truthful execution report. Keep
  `TimeoutStartSec` in `ibkr-executor.service` above ~35 min (it ships at 3600).
- MOC can't be cancelled after the cutoff and accepts whole shares only
  (`--fractional` is ignored for MOC, with a warning).

```bash
# default — protected limits, 0.5% cap
python scripts/ibkr_execute_book.py --file data/overall/target_book.json --execute

# closing auction instead
python scripts/ibkr_execute_book.py --file data/overall/target_book.json --execute \
    --order-type moc

# wider cap for a host with delayed-only market data
python scripts/ibkr_execute_book.py --file … --execute --slippage-cap 0.015
```

The Executed Book tab shows what was actually sent per trade (`LMT $251.66`,
`MOC`, `MKT`), so an escalation or a market fallback is visible after the fact.

### Sequencing: sells settle, then buys are sized to the proceeds
Orders go out in two legs — **all sells first**, awaited (and escalated if a
marketable limit didn't fill), and only then the buys. The buy leg is then sized
against what the account can actually pay for: settled cash plus the proceeds
the sells *really* realised. A sell that half-fills therefore halves the
corresponding buying power instead of quietly drawing a margin loan, and the
trimmed names are reported rather than silently dropped. `--allow-margin`
restores the old unconditional behaviour.

MOC is the one exception: every leg prints in the same closing auction, so
sequencing would buy nothing but lost minutes against the 15:50 ET entry cutoff.
There the funding check uses the sells' expected proceeds, priced at the bid —
the floor on what they can realise, so the budget errs small.

**The budget is priced off live quotes, not the book.** The book carries
*yesterday's* close as its sizing price, so on a market that gapped up, shares
sized at the book price cost more than the book says and the account borrows the
difference. Each order's budget line is therefore taken from the quote at the
price it will actually pay — the ask plus the slippage cap for a buy, the bid
minus it for a sell, both the conservative side of the fill. A name that cannot
be quoted (no market-data subscription, dead feed) falls back to the book price.

Worked example — a book priced at $500, a market at $550, $50k of cash:

| | shares sent | worst-case spend | result |
|---|---|---|---|
| budgeted at the book price | 99 | $54,722 | **$4,722 of margin** |
| budgeted at the live quote | 89 | $49,195 | no margin |

### The historical record of past runs
Each run overwrites `data/overall/executed_book.json`, so the executor also
drops a dated copy of the report beside it, keyed by the signal bar it traded:

```
data/overall/executed_archive/<as_of>.json          # paper runs
data/overall/executed_archive/<as_of>_live.json     # live runs
```

### 💼 Current Positions — the account's own P&L, inside the Overall app
The execution report is the only artifact that knows what the account **really
paid**: the engines know an entry *bar* (a daily close they decided on), the
broker knows the *fill*. So the 🧭 Overall Trading app reads it directly, in a
**💼 Current Positions** collapsible section on both tabs, right under
*🛰️ Live signal & positions — by app*:

| Tab | Cost basis | Marked at |
|---|---|---|
| 🔴 **Live — Decision Cockpit** | IBKR average fill from `executed_book_live.json`, else `executed_book.json` | the same **live spot** every other price on that tab uses |
| 🕰️ **Historical View** | the run standing on the chosen date, from `executed_archive/` | each sleeve's **official close on the viewed bar** |

Each shows the position's **open** P&L (mark − cost basis on what is still held)
alongside the **realised** P&L that run banked, and their total.

**Realised P&L — what a trim actually banked.** The daily optimiser's tilt
often sells *part* of a name and keeps the rest. That trim is invisible in a
cost-basis view on its own: IBKR leaves the remaining shares' `avg_cost`
untouched, so the position's open P&L simply scales down with the share count
and the gain on the sold slice disappears. The report therefore carries
`realized_pnl` in two places — per position (what that name's trim booked) and
per account (the only figure that can include a name closed out **entirely**,
since IBKR drops a position the moment it hits zero shares). The section shows
both: a *Realised this run* metric with `open + realised = total`, a per-card
`💰 banked this run` line, and a footer that says how much came from names that
no longer have a card.

Worked example — 194 GDX at an average cost of \$96.94, marked at \$99.84:

| | Shares | Avg cost | Open P&L | Realised | Total |
|---|---|---|---|---|---|
| Full position | 194 | \$96.94 | **+\$563 (+2.99%)** | — | +\$563 |
| Tilt trims to 150 | 150 | \$96.94 | **+\$435 (+2.99%)** | +\$127.60 | **+\$563** |
| Tilt re-adds 44 @ \$105 | 194 | \$98.77 | **+\$208 (+1.09%)** | +\$127.60 | +\$336 |

The percentage is invariant to a trim (the basis per share does not move) and
open + realised reconciles back to the untrimmed figure. A *re-add* is what
genuinely moves the number: the basis blends to \$98.77, so the same 194 shares
now show +1.09% instead of +2.99% at an unchanged price.

Two properties worth knowing. The figure is per trading **session**, not since
inception — the executor reads it straight after its own rebalance, so in normal
operation it is that rebalance's result (a manual trade in the same session
lands in it too). And it is **nullable**: reports written before the field
existed omit it, and the UI renders "—" rather than \$0, because "nobody asked"
and "it banked nothing" are different facts. Archived records signed before the
field existed still verify — each signature covers the payload as it was
written, so no schema bump was needed.

Same card layout as the signal section above it, grouped by parent signal, so
the *engine's* position (entry bar → bar P&L) and the *account's* position
(average fill → live P&L) sit one under the other and can be read against each
other. They differ by exactly what the fill gave up against the signal close,
plus whatever a later top-up did to the average cost — which is the point of
showing both. Holdings with no quote fall back to the mark the report itself
carried and say so; keys the current universe no longer trades still render,
flagged as held-but-not-traded. Both sections are empty-safe: before the
executor has ever run they explain what will fill them in, and the Historical
one says when the chosen date precedes the archive.

Those records are what the Executed Book page's **🕰️ Historical** tab browses —
pick any past date and it replays that run in full: the trades placed, the
positions it ended with, and the drift against the target book *of that same
signal bar* (`data/overall/book_archive/<as_of>.json`), not today's. A re-run
for the same bar (a late `--refresh-report`, a manual after-hours top-up)
replaces that bar's record: last run wins, matching the account's end state.
The daily wrapper commits the archive alongside the report, so the cloud app
gets it on the same push.

To seed the archive from runs that happened before it existed, run it once from
an unshallowed clone — it rebuilds the records verbatim (signatures intact) from
git history:

```bash
git fetch --unshallow                 # only if the clone is shallow
python scripts/backfill_executed_archive.py --dry-run
python scripts/backfill_executed_archive.py
```

### Automating Option C
- **Publish**: manual `workflow_dispatch` on the Action (or on the default
  branch, the scheduled cron — daily, 7 days a week), or a cron on any host
  running the publisher.
- **Execute**: point `scripts/ibkr_daily.sh`-style cron at the executor instead
  of the rebalancer — e.g. `git pull && python scripts/ibkr_execute_book.py --file data/overall/target_book.json --execute`.
  The executor's freshness guard means a missing/late publish is a safe no-trade,
  not a stale trade — and its weekend/holiday guard means no orders are ever
  placed while the US market is closed, even though a fresh book is published
  on those days.

## Pre-flight guards (what the executor refuses to do)

Every one of these is checked before a single order is transmitted, and each
aborts the run rather than trading a plan it cannot explain. They were hardened
after the **2026-08-18 duplicate-execution incident**: three runs each read the
account as flat, each re-bought the whole book on top of what was already held,
and the paper account ended at 3.4× net liquidation on a $2.17M margin loan
(−8.1% NAV). Any single guard below would have stopped it.

| Guard | What it refuses | Override |
|---|---|---|
| **Verified positions read** | Trading on a positions read that misses value the account itself reports (`GrossPositionValue`). `ib_async` fills its position cache asynchronously, so a read taken too early comes back EMPTY — which the order planner cannot tell apart from a flat account. The read is now settled (`reqPositions` + re-read until stable) and cross-checked. | none — the next run re-reads |
| **Current-bar check** | A book whose signal bar is not the last completed session. The 36-hour freshness window could not catch this: a book published 7:15 AM CT is still "fresh" at 2:30 PM CT the *next* day. A withheld publish now means **no trade**. | `--allow-stale-bar` |
| **Duplicate-run lock** | A second execute run for a signal bar that already has an execution report (`executed_archive/<as_of>.json`). Dry-runs don't lock. | `--force-rerun` |
| **Account state** | Trading into an account that already carries a margin loan (negative cash) or is already geared past the cap. | `--allow-margin` |
| **Book math** | A book with a NaN/negative weight, weights summing past 100%, or a name carrying weight with no usable execution price. | none |
| **Projected exposure** | A plan that would leave gross exposure above `--max-gross-frac` (default 1.02×) of net liquidation. The book is unlevered by construction, so anything above ~1× means a bad read or a duplicate run. | raise `--max-gross-frac` |
| **Turnover** | A plan trading more than `--max-turnover-frac` (default 1.5×) of NAV — implausible for a daily rebalance. | raise the flag |
| **Price drift** | A name quoting further than `--max-price-drift` (default 25%) from the book's sizing price — a split between publish and execution would size every order wrong. | `--max-price-drift 0` |
| **Session hours** | Placing orders outside 09:30–16:00 ET (MOC excepted, which is priced into the close). | `--outside-rth` |
| **Funded buys** | Buys beyond settled cash **plus the proceeds the sells actually realised**, budgeted at the price each order can really pay — the live ask plus the slippage cap for a buy, the live bid minus it for a sell. Sells go first and are awaited; whatever they fail to realise shrinks the buys proportionally instead of being financed on margin. Names trimmed away appear in the report as `SKIPPED-FUNDING`. | `--allow-margin` |

After trading, the run re-reads the account and prints realised gross leverage
and the largest allocation drift — reported, never auto-corrected, because a
second corrective round is exactly the reflex that compounds a bad read.

**Both entry points run the same set.** The guards live in
`scripts/ibkr_common.py` (`Guards`, `preflight()`, `post_trade_check()`) and are
called by the Option-C executor *and* the Option-A rebalancer
(`scripts/ibkr_rebalance.py`) — they place orders through the same code, so they
refuse the same things. The Option-A path has no published book to lock against,
so it carries every guard except the duplicate-run lock.

The machine-checkable half also runs ahead of the session:
`python scripts/preflight_option_c.py` reports `book-current-bar` and
`book-already-executed` alongside the signature and freshness checks, so a
withheld publish or an already-traded bar shows up before the 2:30 PM CT slot
rather than as an ABORT in the log.

## Safety & limitations

- **Paper-account guard** blocks any non-`DU` account by default.
- **Dry-run is the default**; orders require `--execute`.
- **Freshness guard** aborts if the latest signal bar is > 4 days old (dead
  feed protection), and the current-bar check above narrows that to the one
  session the book is actually for.
- Leveraged sleeves (MSTU 2×, SOXL 3×, UGL/NUGT/ERX 2×) are real ETFs and trade
  normally, but they are the volatile part of the book — paper-test thoroughly.
- Market orders fill at the prevailing price; for illiquid names consider adding
  a limit-order variant before ever considering live capital.
- This tool is for **paper** validation of the strategy's live behaviour. Going
  to real money is a separate, deliberate decision beyond this repo's scope.
