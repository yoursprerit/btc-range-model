# Live trading, several IBKR accounts, market-on-close (Option C on one PC)

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
   └─ one git commit + push of every account's report
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
| A name would need fractional shares | Whole shares only (MOC can't be fractional); the executor rounds. |
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
  "book": "data/overall/target_book_live.json",
  "kill_switch_file": "STOP_LIVE",  // create this file to halt trading
  "defaults": {                     // every account inherits these…
    "band": 0.01,                   //   no-trade band (fraction of the account's NAV)
    "max_deploy_frac": 0.25,        //   risk assets ≤ 25% of NAV; the rest parks in SATA (up to the book's 30 % SATA cap; cash beyond)
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
an account; `--no-push` skips publishing the reports.

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
| Deploy cap & per-order cap | `max_deploy_frac`, `max_order_notional` |
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

On the **Overall → 📈 Strategy P&L** section the *Performance source* radio has
**As-published record (C2)** and, once an account has **executed** the book,
**As-published record (IBKR #…)** — one per account. An account that has no
executed run yet (or only previews) gets no option.

* The percentage figures are time-weighted, so cash you add never moves them.
* The dollar figures replay the account from the day it first traded the book,
  using **your monthly deposit and the day of the month it arrives** (fields
  under the radio; day 1–28, weekends/holidays roll to the next trading day).
  A deposit is credited at that day's close, earns from the next bar, and is
  **excluded from P&L** (`P&L = value now − start balance − deposits`).
* The starting balance defaults to the net liquidation IBKR reported after the
  account's first executed run; you can override it.
* To save a schedule across sessions, add it to `data/overall/account_deposits.json`:
  `{"U1234567": {"monthly": 500, "day": 15}}` and commit it.
* A 🔎 *Check* line compares the replayed balance with the net liquidation IBKR
  reported at the last run; a gap means your real deposits differ from the
  schedule (or fills drifted from the published books).
* Costs/taxes toggles and the per-asset tables below the headline scale the
  account's balance at the start date (they don't model the deposits).

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

## Exit codes (runner and wrapper)

`0` all good · `1` an account failed / didn't fully fill / a configured account
isn't managed by the login · `2` gateway or config problem, **nothing traded** ·
`3` session skipped (past the MOC cutoff, holiday, kill switch).

## Troubleshooting

| Symptom | Meaning / fix |
|---|---|
| `cannot open an API session on 127.0.0.1:4001` (exit 2) | Gateway not running or not logged in — approve the IBKR Mobile push / restart IBC. Nothing was traded. |
| `✗ U… is NOT managed by this login` | Wrong login, or the account isn't linked to it. Fix before the next run; that account is skipped, the others still trade. |
| `this login manages N accounts … pass --account` | Running the single-account executor by hand on a multi-account login — pass `--account U…` (the runner always does). |
| `SKIP SESSION … past the 15:50 ET MOC entry cutoff` | The run started too late. Move the task earlier; check the PC wasn't asleep / the gateway was up. |
| `account … already has N working order(s)` | A previous run's MOC orders are still live; sending again would double them. Wait or cancel in TWS. |
| `orders for signal bar … were already transmitted` | The send-lock. The earlier run sent orders; check TWS and `--refresh-report`. `--force-rerun` only if it placed nothing. |
| `!! MOC … was Inactive/Cancelled by the broker: …` | The broker refused it (e.g. IRA/product restriction, permissions). Act in TWS before 15:50 ET; nothing is substituted. |
| `MOC NOT FULLY FILLED` (exit 4) | Check TWS. The next session re-plans from actual holdings. |
| `ABORT: … book_mode=…, not 'live'` | `book` in the accounts file points at the paper book. |
| `could not push the reports … un-committing, files kept` | No git write credential (see `IBKR_OPTION_C_WINDOWS.md` §8). Trading was unaffected. |

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
