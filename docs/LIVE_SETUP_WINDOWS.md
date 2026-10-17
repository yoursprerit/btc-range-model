# Live multi-account trading on Windows - setup guide

Goal: from a fresh (or half-configured) Windows PC to **live market-on-close trading
in several IBKR accounts under one login**, with as little hand work as possible.
Everything below was written from the problems hit the first time; section 9 lists
each one and what now prevents it.

> Real money. The first live day is yours to supervise (section 6).
> Reference for how the executor behaves: [`LIVE_MULTI_ACCOUNT_MOC.md`](LIVE_MULTI_ACCOUNT_MOC.md).

**Time needed:** about 30 minutes, plus waiting for your phone to approve 2FA.

---

## 0. Have these ready (5 minutes)

| Item | Where to get it |
|---|---|
| IBKR **username + password** of the one login that manages all your accounts | you |
| Your phone with **IBKR Mobile** (2FA push) | you |
| `OVERALL_BOOK_SECRET` - the **same value** as the GitHub repo secret of that name | GitHub secrets cannot be read back; use your password manager's copy |
| Which accounts to trade, and each one's type: `margin` (taxable), `ira` (Roth **and** Traditional IRA), or `cash` | you |
| *(optional)* a Slack/Discord webhook, and two healthchecks.io ping URLs | see section 7 |

Nothing about your accounts is ever sent to GitHub (section 8).

## 1. One-time Windows preparation (5 minutes)

1. **Windows account:** sign in as the normal user that will run the trading (not a different admin).
2. **Never sleep:** the wizard sets "never sleep on AC power". On a laptop keep it plugged in and the lid open (or set lid-close to *Do nothing*).
3. **Auto-login** so a reboot brings everything back: press `Win+R`, run `netplwiz`, untick *Users must enter a user name and password*, enter your Windows password.
4. **Install IB Gateway** (Stable, Windows) from interactivebrokers.com -> *Trading -> IB Gateway* into the default `C:\Jts`. Do not log in yet.

## 2. Get the code

Open **PowerShell** (regular, not "as administrator") and run:

```powershell
cd $HOME\Documents
git clone https://github.com/yoursprerit/btc-range-model.git TradingRepo\btc-range-model
cd TradingRepo\btc-range-model
```

(If the folder already exists: `cd` into it and run `git checkout main; git pull`.)

## 3. Run the setup wizard (15 minutes)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_live.ps1
```

It installs Git/Python if missing, creates `.venv`, installs the dependencies, and then
walks through these steps. Each is safe to re-run; answer the prompts.

| Step | What it does for you |
|---|---|
| `gateway` | Finds the installed IB Gateway and its **exact version** (e.g. 1051) and bundled Java; repairs a stale `inst_jre.cfg`. |
| `ibc` | Downloads IBC to `C:\IBC`, asks for your IBKR login, writes `C:\IBC\config-live.ini` (live, port 4001, API not read-only) readable only by you. |
| `startbat` | Builds `C:\IBC\StartGateway-live.bat` with the **right gateway version, config path and Java path** already filled in. |
| `secret` | Stores `OVERALL_BOOK_SECRET` (typed hidden) and the optional alert/heartbeat URLs for your user. |
| `accounts` | Asks you to start the gateway and approve 2FA, **discovers your accounts**, asks per account "trade it? / type / label", writes `deploy\ibkr_accounts.json` with conservative limits. |
| `tasks` | Registers the 2:20 PM run, the 4:15 PM reconcile, gateway-at-logon and a 5-minute watchdog; **disables the old paper task and paper gateway shortcut**. |
| `verify` | Runs the live pre-flight and prints PASS / WARN / FAIL per check. |

To run one step again: `scripts\setup_live.ps1 -Only accounts` (or `ibc`, `startbat`, ...).
Health report any time, changing nothing: `scripts\setup_live.ps1 -Check`.

If the `tasks` step complains about permissions, re-run `scripts\setup_live.ps1 -Only tasks`
from PowerShell opened with *Run as administrator* (the tasks still run without admin rights).

## 4. Check the gateway settings once (2 minutes)

Start the gateway (`C:\IBC\StartGateway-live.bat`), approve the phone push, then in the Gateway window:
**Configure -> Settings -> API -> Settings**

* *Enable ActiveX and Socket Clients* - ticked
* *Read-Only API* - **unticked**
* *Trusted IPs* - contains `127.0.0.1`
* *Socket port* - `4001`

And under **API -> Precautions**, tick the options that bypass order-size / price pop-ups for API orders
(a pop-up would block an unattended order).

## 5. Pre-flight and dress rehearsal (5 minutes)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows_option_c.ps1 -LivePreflight
powershell -ExecutionPolicy Bypass -File scripts\setup_windows_option_c.ps1 -LiveRehearse
```

* `-LivePreflight` checks the secret, signature, book freshness, tasks, monitors and that the gateway answers.
* `-LiveRehearse` runs the real scheduled job with the kill switch armed: the whole chain, **zero orders**.
* Expect on each account, in the log: `positions read OK`, `Deploy cap 25% of NAV ...`, and the planned orders.
  An "account updates request timed out ... valuing positions at live quotes" line is **normal** (section 9).

Then **sign out and back in (or reboot)** once: this proves the gateway starts by itself and that the scheduled
tasks see the environment variables. Approve the phone push when the gateway window asks.

## 6. First live day

1. Before 2:20 PM Central: gateway window logged in, phone nearby, `scripts\setup_live.ps1 -Check` all OK.
2. 2:20 PM CT: the task runs. Watch `logs\ibkr_live.log`. Each account prints its orders, then waits for the closing auction.
3. After 3:00 PM CT: in TWS check the fills. At 3:15 PM CT the reconcile task confirms them (and alerts if anything is unfilled).
4. To stop everything at any moment: create the file named in `kill_switch_file` (`STOP_LIVE` in the repo folder), or set `"enabled": false` for one account.
5. Start small. Keep `max_deploy_frac` at 0.25 and `max_order_notional` at 5000 until several days are clean.

## 7. Monitoring (recommended, 10 minutes)

If the PC is off, nothing can alert you - so a **dead-man's switch** watches from outside.
Create two daily checks on healthchecks.io (run: `15 14 * * *` America/Chicago, reconcile: `15 16 * * *`
America/New_York, grace 45 min), then re-run `scripts\setup_live.ps1 -Only secret` and paste the two URLs.
Details: [`LIVE_MULTI_ACCOUNT_MOC.md`](LIVE_MULTI_ACCOUNT_MOC.md) section 5.

## 8. Privacy - nothing about your accounts goes to GitHub

* Account numbers, the accounts file, balances, holdings, fills, logs and the IBKR login stay **on this PC**
  (`deploy\ibkr_accounts.json`, `data\overall\accounts\`, `logs\`, `C:\IBC\` are git-ignored, and a test fails if a real account id is ever committed).
* Execution reports are **not** pushed to GitHub. (Opt in with `setx IBKR_PUBLISH_REPORTS 1` only if you accept that.)
  The Executed Book / P&L pages then show the live accounts when Streamlit runs on this PC: `.venv\Scripts\streamlit run app\...`.
* The scheduled run only does `git pull`; the PC never needs write access to the repository.
* Alerts and heartbeats carry masked ids (`...4567`), never balances.
* Do not paste logs, account numbers or balances into GitHub issues, PRs or chats without masking them.

## 9. Every problem we hit, and the fix

| What you saw | Cause | Fix / now prevented by |
|---|---|---|
| `invalid JSON ... line 34 column 5` when loading the accounts file | trailing comma or comment in `ibkr_accounts.json` | the wizard writes the file itself; to edit by hand use `scripts\setup_live.ps1 -Check` which validates it |
| `IBC configuration file ... Documents\IBC\config-live.ini does not exist` | the stock bat looks in `Documents\IBC` | `startbat` sets `IBC_INI=C:\IBC\config-live.ini` |
| `TWS_MAJOR_VRSN` / "cannot find Gateway" after installing 10.51 | the bat still named an old version | `startbat` reads the installed version from `C:\Jts\ibgateway\<n>` |
| `Can't find suitable Java installation` | stale `.install4j\inst_jre.cfg` | `gateway` repairs it (backup kept) and `startbat` sets `JAVA_PATH` |
| `'wmic' is not recognized` | harmless noise from IBC on newer Windows | ignore |
| Pre-flight "Account updates request timed out" / `reqAccountUpdates` | the gateway does not serve that feed for some accounts | automatic: bounded wait, retry, then values positions at live quotes. Not an error |
| Pre-flight seemed to hang | a blocking gateway call with no timeout | every call is now time-limited (20 s) |
| "positions read abort explains $0" | account feed unserved | same fallback as above |
| Fractional share holding oversold (22.87 -> 23) | rounded sells up | sells round **down**; the fractional remainder is left alone |
| SATA 40% in the book | idle capital went to SATA | the book is capped at 30% (`sata_cap`); the rest stays cash |
| `max_deploy_frac 0.25` seemed ignored | it was applied per asset, not in total | it is now the **total** across every asset incl. SATA; the rest stays cash |
| Roth IRA account type? | - | use `"type": "ira"` for Roth **and** Traditional |
| "Is my signing secret right?" | cannot read GitHub secrets back | pre-flight check `secret-verifies` proves the secret signs the book on disk |
| Paper trading still running | paper task + paper gateway shortcut | `tasks` disables the paper task, removes the paper shortcut and warns if port 4002 is still open |
| Gateway window closed / PC rebooted | nothing restarted it | `IB Gateway LIVE (logon)` task + 5-minute watchdog |
| Gateway restarted, run found it logged out | 2FA needs your phone | watchdog relaunches it; you must still tap the push. The dead-man's switch tells you if the run did not happen |
| Run did not happen at all | PC off / asleep | dead-man's switch (section 7); "never sleep" set by the wizard |
| `ERRORLEVEL 1001/1006` from the bat | IBC could not start the gateway (see the lines above it in the window) | read the line above; usually one of the rows above |

## 10. Routine care

| When | What |
|---|---|
| Daily | nothing; glance at the monitor / Slack |
| About weekly (Sunday) | IBKR forces a fresh login: approve the phone push |
| After any Windows update / reboot | log in happens automatically; approve the push if shown |
| Adding or removing an account | `scripts\setup_live.ps1 -Only accounts` |
| IB Gateway upgrade | install it, then `scripts\setup_live.ps1 -Only gateway` and `-Only startbat` (new version number) |
| Changed IBKR password | `scripts\setup_live.ps1 -Only ibc` (answer yes to rewrite) |
| Stop trading for a while | create `STOP_LIVE` in the repo folder (delete it to resume) |
| Remove everything | `Unregister-ScheduledTask` the four `IBKR Option C LIVE ...` / `IB Gateway LIVE ...` tasks; delete `C:\IBC\config-live.ini` |
