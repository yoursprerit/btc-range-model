# Live trading, several IBKR accounts, market-on-close (Option C on one PC)

> **Setting up a PC?** Follow [`LIVE_SETUP_WINDOWS.md`](LIVE_SETUP_WINDOWS.md) - one command, no trial and error. This page is the reference.

> ⚠️ **Real money.** This trades the strategy in **funded** IBKR accounts. It was
> built from the paper trial, but **it has not been run against a live gateway
> yet** — the first live run is yours to supervise (see
> [Rollout](#rollout--do-this-in-order)). Read the whole page once before the
> first run.

One PC, **one IB Gateway live login**, as many accounts as that login manages.
Every account trades the **same signed live book** as **market-on-close (MOC)
orders only**, each sized to its own net liquidation and its own limits.

```
 CLOUD publisher ──► data/overall/target_book_live.json  (HMAC-signed; idle capital → SATA)
                              │ git pull
                              ▼
 scripts/ibkr_execute_live_accounts.ps1     (Task Scheduler, ~2:15 PM US Central)
   └─ scripts/ibkr_execute_accounts.py      (reads deploy/ibkr_accounts.json)
        ├─ checks the gateway login manages every configured account
        └─ starts ONE executor per account, TOGETHER ───────────────┐
             ibkr_execute_book.py --account U111… --strict-moc …    │ same IB Gateway,
             ibkr_execute_book.py --account U222… --strict-moc …    │ one clientId each
             …                                                      ▼
                                              IB Gateway (live) 127.0.0.1:4001
   └─ reports stay on this PC (git-ignored); publishing is an opt-in
```

**Why all accounts at once.** A MOC order must be *entered* before the exchange
cutoff (**15:50 ET**) but doesn't *fill* until the **16:00** auction, and each
executor holds its connection open through the auction (~30 min) to write a
truthful report. Run one after another, account 2 would start after the cutoff.
Run together, every order is in within a few minutes and one auction fills them
all. A failing or hung account cannot affect the others (separate processes).

---

## What "strict MOC" means

| Situation | Behaviour |
|---|---|
| Normal run, before 15:50 ET | Whole-share `MOC` / `DAY` orders, every one stamped with its account id. |
| Run starts **after** the cutoff (PC was off, gateway slow) | **Skip the session.** Nothing is sent, exit code 3, an alert is raised. No limit/market fallback and **no after-hours catch-up** — tomorrow's book supersedes today's. |
| Early-close day (day after Thanksgiving, Dec 24) | The cutoff moves to 12:50 ET, so the normal slot is past it → session skipped. |
| An account holds fractional shares (e.g. 22.8672) | MOC is whole shares only. Buys round to the nearest share; **sells round DOWN and never exceed what is held**. The leftover fraction (e.g. 0.87 share) can't be sold by an MOC order: the log says so (`note: … sell the fractional remainder by hand`) and it is left alone. |
| The broker rejects an order (IRA/product restriction, permissions) | Reported **within seconds** in the log and alert, with the broker's message, while there's still time before the cutoff to act in TWS. Nothing is substituted. |
| MOC didn't fully fill after the auction | Exit code 4 + alert; nothing is re-sent. Tomorrow's run re-plans from what the account really holds. |

Sells and buys all go in together (they print in the same auction). Buys are
funded from **settled cash + the sells' expected proceeds, priced at the bid**,
so an account can't accidentally borrow — unless you set `allow_margin` for a
margin account.

---

## Setup

### 1. The IB Gateway live login

1. One IBKR **username** that manages all the accounts (linked / multi-account
   under the same login). Confirm in TWS/Client Portal that the account selector
   lists every account you want to trade.
2. Install **IB Gateway** (stable) and **IBC** (unattended login). Copy
   `deploy/ibc/config.live.ini.example` → e.g. `C:\IBC\config-live.ini`, fill in
   the login. `TradingMode=live`, API port **4001**, `ReadOnlyApi=no`.
   *Never commit the filled-in file* (IBC stores the password in clear text).
3. In Gateway → *Configure → Settings → API → Settings*: **Enable ActiveX and
   Socket Clients**, **uncheck Read-Only API**, **Trusted IPs: 127.0.0.1**, socket
   port **4001**. Under *Precautions*, bypass order-size/price warnings for API
   orders if your account would otherwise show confirmation pop-ups (a modal
   dialog blocks an unattended order).
4. **2FA is mandatory on a funded login.** Use IBKR Mobile push. IBC re-sends it
   after a timeout, but a human has to approve it: expect to approve a push **about
   weekly** (IBKR periodically forces a full re-login — check IBC's docs for the
   cadence on your setup) and after any restart. If the gateway isn't logged in at run time the runner exits **2 and
   trades nothing** — fail-safe, but you'd miss the day.
5. Market data: subscribe the live feeds for the traded ETFs if you can (price
   sanity check + buy sizing use quotes). Without them set
   `"market_data_type": 3` (free delayed) in the accounts file; funding then
   falls back to the book's own prices.

### 2. The accounts file

```powershell
copy deploy\ibkr_accounts.example.json deploy\ibkr_accounts.json   # git-ignored
notepad deploy\ibkr_accounts.json
```

```jsonc
{
  "port": 4001,                     // live gateway
  "client_id_base": 40,             // account i uses 40+i (stable per list position)
  "stagger_s": 10,                  // seconds between starting each account (gateway is slow if hit at once)
  "book": "data/overall/target_book_live.json",
  "kill_switch_file": "STOP_LIVE",  // create this file to halt trading
  "defaults": {                     // every account inherits these…
    "band": 0.01,                   //   no-trade band (fraction of the account's NAV)
    "max_deploy_frac": 0.25,        //   TOTAL invested ≤ 25% of NAV across every asset incl. SATA (scaled proportionally); the other 75% stays cash
    "max_order_notional": 5000,     //   $ cap on any single order
    "max_gross_frac": 1.02, "max_turnover_frac": 1.5, "max_price_drift": 0.25
  },
  "accounts": [
    { "id": "U1111111", "label": "Taxable margin", "type": "margin" },
    { "id": "U2222222", "label": "Roth IRA", "type": "ira",
      "max_order_notional": 3000 },           // …and can override any of them
    { "id": "U3333333", "label": "Paused", "type": "ira", "enabled": false }
  ]
}
```

* `child_timeout_s` (seconds, default 3900): how long one account's executor may run before the
  runner kills it. It is never allowed below the 60-minute MOC wait ceiling plus 5 minutes — a
  smaller value (the old example carried 3000) is raised automatically, so the executor is not
  killed while it waits for the 4:00 PM ET auction.
* `expected_profile` (optional, per account, e.g. `"Balanced"`): the risk profile this account
  follows. A published book for any other profile **aborts that account's run** with an alert.
  Recommended for every live account — see "Is this the right book?" in `LIVE_TRADING.md`.
* `type`: `margin`, `ira`, or `cash`. `allow_margin: true` is accepted **only**
  on `margin`; an IRA/cash account can never be configured to borrow.
* The file is validated before anything runs: duplicate ids, paper (`D…`) ids,
  unknown keys and out-of-range limits are all rejected with the reason.
* **Start small:** a low `max_deploy_frac` and a per-order cap bound the damage
  of any bug. Raise them only after several clean days.
* `enabled: false` removes an account from the run without deleting its config.

### 3. Secret, kill switch, scheduler

```powershell
# the signing secret must be set for the user (same value as the publisher)
setx OVERALL_BOOK_SECRET "<the same secret the publisher signs with>"

# register the scheduled task (elevated PowerShell) - NOT part of -All on purpose
powershell -ExecutionPolicy Bypass -File scripts\setup_windows_option_c.ps1 -LiveTask -LiveTaskTime 14:15
```

* The live runner **requires** the secret and a valid signature; unlike paper, it
  will not trade an unsigned book.
* `-LiveTaskTime` is **local machine time**. 14:15 is 2:15 PM **US Central** =
  3:15 PM ET — 35 min before the cutoff. MOC fills at the close whenever it is
  entered, so earlier costs nothing and buys margin for a slow gateway or `git
  pull`. On another time zone use the local equivalent.
* The wrapper asks Windows not to sleep while it runs (the auction wait is ~30
  min). Also set Power Options so the PC isn't sleeping at 2:15 PM, and keep the
  user **logged on** (the task runs interactively).
* The legacy `ibkr_execute_daily.ps1` now **refuses** `IBKR_ACCOUNT_MODE=live`
  (set `IBKR_ALLOW_LEGACY_LIVE=1` to override) so a live run can't bypass strict
  MOC by accident.

### 4. Optional alerts

`setx IBKR_ALERT_WEBHOOK "https://hooks.slack.com/…"` (Slack/Discord/Mattermost
compatible). The runner alerts on: gateway not logged in, an account the login
doesn't manage, a session skipped for the cutoff, any account that failed, was
rejected, or didn't fully fill.

### 5. Dead-man's switch and the 4:15 PM reconcile

Alerts only fire from a run that started. If the PC is off or asleep, or Windows
restarted, nothing runs and nothing complains. A heartbeat fixes that: each run
pings an outside monitor, and the **monitor** alerts when a ping is missing.

1. Create two free checks at healthchecks.io (or Uptime Kuma push monitors):
   * **Run** - schedule `15 14 * * *`, timezone `America/Chicago`, grace 30-45 min.
   * **Reconcile** - schedule `15 16 * * *`, timezone `America/New_York`, grace 30-45 min.
   Both are *daily*: a weekend or holiday is a normal, successful "nothing to do" run that still pings.
2. `setx IBKR_HEARTBEAT_URL "https://hc-ping.com/<run-uuid>"` and
   `setx IBKR_RECONCILE_HEARTBEAT_URL "https://hc-ping.com/<reconcile-uuid>"`
   (https only; the URL is a secret). For Kuma add `setx IBKR_HEARTBEAT_STYLE kuma`.
3. Pings carry statuses and masked account ids (`...6213`) only - never balances or holdings.
   A monitoring outage never stops a run. Previews and `--check` never ping.

**Reconcile** (`ibkr_reconcile_accounts.py`, task from `setup_windows_option_c.ps1 -ReconcileTask`,
15:15 CT = 16:15 ET) is **read-only**: it never places or cancels orders. For each account that
sent orders today it reads the day's executions and working orders from IBKR and
* rebuilds a signed `executed_book_live.json` if the run died before writing it;
* writes late fills into an existing report (never overwrites a record with an empty one);
* alerts on any unfilled leg, orders still working after the close, or missing executions.
Exit codes: `0` all reconciled / `1` a problem was found / `2` gateway or config / `3` nothing sent today.
Use `--dry-run` to see what it would do and `--force` to run before 16:05 ET.

---

## Rollout — do this in order

1. **Pre-flight** (read-only):
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\setup_windows_option_c.ps1 -LivePreflight -LiveTaskTime 14:15
   ```
   Every line must be PASS. It validates the accounts file, that the secret
   verifies the live book, the MOC slot, the task, and that the gateway login
   manages **every** configured account.
2. **Preview** during market hours — connects, reads every account, runs all
   guards, prints each account's order plan, **sends nothing**:
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\setup_windows_option_c.ps1 -LiveRehearse
   ```
   Check each plan against what you expect for that account's size and limits.
3. **One account first.** Set all but one account `"enabled": false` (or run
   `python scripts\ibkr_execute_accounts.py --execute --only U1111111`) with a
   small `max_deploy_frac`, before 15:45 ET on a trading day. Watch the orders
   appear as `MOC` in TWS, then the 4 PM fills, then the report in
   `data\overall\accounts\U1111111\`.
4. **Add accounts one at a time**, then let the scheduled task run unattended.
5. Reconcile daily against the paper control and the Executed Book page
   (**🏦 IBKR Account #U…** broker radio) for the first few weeks; raise caps slowly.

Manual run any time: `python scripts\ibkr_execute_accounts.py` (preview) or
`--execute`; `--check` verifies config + gateway only; `--only <id>` restricts to
an account. Reports are **not** published to GitHub unless you set `IBKR_PUBLISH_REPORTS=1` (see *Privacy*).

---

## Guards (all kept from the paper trial, applied per account)

| Guard | Notes |
|---|---|
| Signature verified, **required** | live won't trade an unsigned book |
| Right book | live mode refuses a book whose `book_mode` isn't `live` |
| Current-bar / freshness / trading-day | a withheld publish = no trade |
| Account pinned | `--account` + `--expected-account`; the login's account list must contain it; orders carry `account=` explicitly, and fills/positions/cash are read per account |
| Verified positions read | an incomplete read aborts (the 2026-08-18 failure mode) |
| Duplicate-run lock | per account (execution archive) **plus** a send-lock written the instant orders are transmitted — a MOC report only exists after 4 PM, so the archive alone can't stop a 3:40 PM re-run |
| Working-orders check | refuses to send into an account that already has orders working (`--ignore-working-orders` to override) |
| Cash-funded buys | no margin unless a margin account opts in |
| Exposure / turnover / price-drift caps | `max_gross_frac`, `max_turnover_frac`, `max_price_drift` |
| Deploy cap & per-order cap | `max_deploy_frac` is the **total** the book may put to work, across every asset **including SATA** (all positions scaled by the same factor); the rest of the account is left as cash and never parked anywhere. It is a ceiling: a book that already deploys less is not scaled up. `max_order_notional` clamps any single order. |
| SATA cap | SATA never exceeds the book's own signed `sata_cap` (V2: 30 %). A book published before the cap has none, so the executor applies a 30 % default first (`--default-sata-cap`, `-1` turns it off) — a stale pre-cap book can still be the one that trades if the next publish is withheld. The deploy cap is applied after it. Example: book risk 60 % + SATA 40 % → SATA trimmed to 30 % → total 90 % → ×(25/90) → every position × 0.278, total 25 %, cash 75 %. |
| Per-order limits | Independent of the book: no SELL larger than the shares held; no single BUY above `--max-buy-frac` (40 %) of net liquidation; no more than `--max-orders` (25) orders in a run. |
| Clock | The PC clock must be within 30 s of IBKR's server clock (the MOC cutoff and trading-day tests read it). Fix: `w32tm /resync`. |
| Currency | The account's NetLiquidation must be in USD; any other base currency is refused. |
| Market calendar | The holiday / early-close tables end in 2027 (`CALENDAR_LAST_YEAR`). The runner warns 120 days before and **refuses to trade** once past them, so a holiday is never mistaken for a trading day. A test fails ~120 days ahead as a reminder to extend them. |
| Post-trade verdict | After the auction the account is re-read; gearing above the cap or an allocation more than 5 pp off target makes the account exit **5** and raises an alert (nothing is re-sent). |
| Gateway retry | A gateway that isn't answering is retried every 30 s for up to 10 min (stopping 6 min before the cutoff) before the session is given up. |
| Order tag | Every order carries `orderRef = btc-range:<book date>` so the bot's orders are recognisable in TWS and audits. |
| Atomic reports | Reports, archives and reset markers are written to a temp file and renamed, so a crash cannot leave half a file. |
| Report publishing | **Off by default** (reports hold account numbers, balances and holdings). With `IBKR_PUBLISH_REPORTS=1` it commits **only** the report files and refuses to push from a detached HEAD. |
| Logs | Per-account and wrapper logs rotate at 5 MB (3 kept); each run logs the Python / `ib_async` / `pandas` / `numpy` versions. |
| Least privilege | The scheduled tasks run with normal user rights, not as administrator. |
| Kill switch | `kill_switch_file` or `IBKR_TRADING_DISABLED=1` |
| Strict MOC | no fallback, no after-hours catch-up; early-close days handled |

Lessons from the paper trial baked in: the 2026-08-18 duplicate execution
(3.4× NAV) → the verified read, locks and caps above; the 2026-08-17 Task
Scheduler crash on a non-ASCII arrow → the wrapper forces UTF-8 and stays
ASCII; the secret-rotation mismatch → pre-flight checks the secret actually
*verifies* the book; the 18:48 ET after-hours run → no outside-hours orders for
live; push races → one report commit per run, rebased once if rejected.

---

## Strategy P&L per account, with monthly deposits

On the **Overall → 📈 Strategy P&L** section, choosing **As-published record** as
the *Performance source* shows an **Account** picker: **Collective2 model account**
and, once an account has **executed** the book, **IBKR #…** — one per account
(read from the local `accounts/` folder or the encrypted vault, like 💼 Current
Positions). An account that has no executed run yet (or only previews) gets no
option.

* The percentage figures are time-weighted, so cash you add never moves them.
* The dollar figures replay the account from the day it first traded the book,
  using **your monthly deposit and the day of the month it arrives** (fields
  under the radio; day 1–28, weekends/holidays roll to the next trading day).
  A deposit is credited at that day's close, earns from the next bar, and is
  **excluded from P&L** (`P&L = value now − start balance − deposits`).
* The starting balance defaults to the net liquidation IBKR reported after the
  account's first executed run; you can override it.
* To save a schedule across sessions, add it to `data/overall/account_deposits.json`:
  `{"U1234567": {"monthly": 500, "day": 15}}` — keep it **local** and do not commit real amounts (`git update-index --skip-worktree data/overall/account_deposits.json`), or just enter them on the page each session.
* A 🔎 *Check* line compares the replayed balance with the net liquidation IBKR
  reported at the last run; a gap means your real deposits differ from the
  schedule (or fills drifted from the published books).
* Costs/taxes toggles and the per-asset tables below the headline scale the
  account's balance at the start date (they don't model the deposits).

## Security checklist (things only you can do)

* **Apply the lower-privilege task.** Existing tasks keep the old (administrator)
  setting. Re-register them once: `setup_windows_option_c.ps1 -LiveTask`
  (and `-Task` if you still use the paper one); `-Force` replaces them.
* **Repo is private?** Execution reports hold account ids, balances and holdings:
  GitHub → Settings → General → Danger Zone must say *private*.
* **Protect `main`.** The wrapper runs whatever code is on `main` each morning, so
  require pull-request review and passing checks and block direct pushes
  (Settings → Branches). Turn on 2FA / a passkey for the GitHub account, give the
  cron-job.org token an expiry, and rotate `OVERALL_BOOK_SECRET` now and then.
* **IBC config holds the password in clear text.** Restrict it to your user:
  `icacls "C:\Users\<you>\Documents\IBC\config-live.ini" /inheritance:r /grant:r "$($env:USERNAME):(R)"`
* **IBKR login:** use a dedicated username with trading permission only (no
  funds transfer), turn on its IP restriction if you can, and keep API access to
  `127.0.0.1` (Gateway → API → Settings → trusted IP `127.0.0.1`; never open
  port 4001 in the firewall).
* **Windows:** BitLocker on; set Windows Update active hours so no restart lands
  between about 2 and 4:15 PM Central.
* **Lock the dependency versions** once a rehearsal looks right:
  `.\.venv\Scripts\python.exe -m pip freeze > requirements-ibkr.lock` and install
  from it (`pip install -r requirements-ibkr.lock`); occasionally run
  `pip install pip-audit; pip-audit` to check for known vulnerabilities. Each run
  logs the versions it used, so a behaviour change can be traced.

## Where things live

| Thing | Path |
|---|---|
| Accounts config (yours, git-ignored) | `deploy/ibkr_accounts.json` |
| Per-account report + history | `data/overall/accounts/<ACCT>/executed_book_live.json`, `…/executed_archive/` |
| Per-account send-lock (git-ignored) | `data/overall/accounts/<ACCT>/.send_locks/<as_of>.lock` |
| Logs | `logs/ibkr_live_executor.log` (whole run), `logs/accounts/<ACCT>.log` (per account) |
| Runner / wrapper | `scripts/ibkr_execute_accounts.py`, `scripts/ibkr_execute_live_accounts.ps1` (`.sh` twin) |

> **Privacy note.** The execution reports are committed to the repo and contain
> each account's id and holdings (as the paper reports always did). Keep the
> repository **private**.

## Privacy - what never goes to GitHub

* Account numbers, balances, holdings, fills and your IBKR login live **only on the PC**:
  `deploy/ibkr_accounts.json`, `data/overall/accounts/` (reports, archives, send-locks),
  `logs/`, `C:\IBC\config-live.ini` (password) are git-ignored, and a test fails the build if a
  real-looking account id is ever committed.
* Execution reports are **not** pushed unless you set `IBKR_PUBLISH_REPORTS=1`. The *Executed Book*
  and *Strategy P&L* pages then show live accounts only when Streamlit runs **on this PC**
  (`streamlit run app/...`), not on a cloud deployment.
* Alerts and heartbeats go to *your* Slack / healthchecks.io, not GitHub, and carry masked ids (`...4567`).
* Never paste account numbers, balances, logs or the signing secret into issues, PR descriptions or commits.

## Exit codes (runner and wrapper)

`0` all good · `1` an account failed / didn't fully fill / a configured account
isn't managed by the login · `2` gateway or config problem, **nothing traded** ·
`3` session skipped (past the MOC cutoff, holiday, kill switch). Per account:
`4` orders sent but not fully filled · `5` filled but the account did not land on target.

## Troubleshooting

| Symptom | Meaning / fix |
|---|---|
| `cannot open an API session on 127.0.0.1:4001` (exit 2) | Gateway not running or not logged in — approve the IBKR Mobile push / restart IBC. Nothing was traded. |
| `positions read explains $0 of the $… the account reports` / `account updates request timed out` | The gateway was slow to send that account's prices when several accounts connected at once. The executor now re-requests them (twice, each bounded to 8 s, so a silent gateway can never hang the run), then values the positions at **live quotes** as an independent cross-check, and only refuses if neither agrees with the account's own total. If it still happens, raise `stagger_s` in the accounts file (default 10 s) or restart the gateway. |
| `✗ U… is NOT managed by this login` | Wrong login, or the account isn't linked to it. Fix before the next run; that account is skipped, the others still trade. |
| `this login manages N accounts … pass --account` | Running the single-account executor by hand on a multi-account login — pass `--account U…` (the runner always does). |
| `SKIP SESSION … past the 15:50 ET MOC entry cutoff` | The run started too late. Move the task earlier; check the PC wasn't asleep / the gateway was up. |
| `account … already has N working order(s)` | A previous run's MOC orders are still live; sending again would double them. Wait or cancel in TWS. |
| `orders for signal bar … were already transmitted` | The send-lock. The earlier run sent orders; check TWS and `--refresh-report`. `--force-rerun` only if it placed nothing. |
| `!! MOC … was Inactive/Cancelled by the broker: …` | The broker refused it (e.g. IRA/product restriction, permissions). Act in TWS before 15:50 ET; nothing is substituted. |
| `MOC NOT FULLY FILLED` (exit 4) | Check TWS. The next session re-plans from actual holdings. |
| `ABORT: … book_mode=…, not 'live'` | `book` in the accounts file points at the paper book. |
| `could not push the reports … un-committing, files kept` | Only appears if you opted in to publishing and the PC has no git write credential (`IBKR_OPTION_C_WINDOWS.md` §8). Trading was unaffected. |

## What is and isn't verified

* **Verified** (unit/integration tests with a fake gateway, ~70 cases in
  `tests/test_multi_account.py`, `tests/test_executor_multi_account.py`): account
  selection and pinning, order stamping, per-account fills/locks/reports, strict
  MOC and early-close handling, config validation, parallel execution and
  failure isolation, exit codes.
* **Not verified here** (needs your PC): behaviour against a real IB Gateway
  login with several accounts, the PowerShell scripts (written ASCII-only but
  not executed in the build environment), IBC/2FA, and how IBKR treats MOC for
  each of your account types. That is exactly what the preview and
  one-account-first steps above are for.
