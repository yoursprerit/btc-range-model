# Option C on a Windows Laptop — Publish in the Cloud, Execute Locally

End-to-end guide to running **Option C** with the **executor + IB Gateway on a
Windows laptop**. The heavy model run happens elsewhere (a GitHub Action or any
machine with the model stack); your laptop only consumes a small **signed target
book** and trades the IBKR **paper** account.

> This is the Windows-specific companion to `IBKR_PAPER_TRADING.md`. Read that
> for the cross-platform overview and the artifact schema; read this for the
> exact Windows install/run/automate steps.

---

## 0. How Option C fits together

```
 PUBLISHER  (cloud / GitHub Action — has the model)     EXECUTOR  (your Windows laptop)
 scripts/publish_target_book.py                         scripts/ibkr_execute_book.py
   run_universe → optimize → gate                          git pull  →  load target_book.json
   → data/overall/target_book.json  ──(git commit)──►      → verify HMAC signature
   (HMAC-signed with OVERALL_BOOK_SECRET)                  → validate freshness / trading day
                                                           → diff vs IBKR paper positions
                                                           → place market orders (sells→buys)
                                                                     │
                                                            IB Gateway (paper)  ── 127.0.0.1:4002
```

- **Your laptop never runs the model.** It needs only `ib_async` + pandas.
- The book is **self-contained and signed**: the executor sizes orders from the
  book's own weights and prices, verifies the signature, and only then trades.
- **Paper only**: the executor refuses any account that isn't a `DU…` paper
  account; `--dry-run` (no orders) is the default.

---

## ⚡ Quick start — automated setup (steps 2–7 in one script)

Most of the setup below is scripted in **`scripts\setup_windows_option_c.ps1`**.
It's idempotent (safe to re-run) and each phase is individually selectable.

From an **elevated** PowerShell (Run as Administrator), in the cloned repo:

```powershell
# full setup: Python 3.12 + Git, venv + deps, secret, IB Gateway installer,
# IBC (with paper creds), and the daily scheduled task
powershell -ExecutionPolicy Bypass -File scripts\setup_windows_option_c.ps1 -All -IbUser myPaperUser -IbPassword 's3cret'
```

What it automates vs what stays manual:

| Automated by the script | Stays manual (by design) |
|---|---|
| Install Python 3.12 + Git (winget) | IBKR **licence click-through** during install |
| Create `.venv`, install `requirements-ibkr.txt` | **First IB Gateway login / 2FA** |
| Set `OVERALL_BOOK_SECRET` (generates one if omitted) | Ticking the API settings (or let IBC enforce them) |
| Download + launch the IB Gateway installer | Deciding to `--execute` |
| Download IBC + template its `config.ini` (paper) | |
| Register the daily Task Scheduler job | |
| Verify the whole chain | |

Run a single phase instead of everything, e.g. just rebuild the venv and
re-register the task:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows_option_c.ps1 -Venv -Task
```

Phase flags: `-InstallPython -Venv -Secret -Gateway -IBC -Task -Verify` (or
`-All`). Re-open PowerShell after the run so the new `OVERALL_BOOK_SECRET` is
visible. Then jump to **§5** to publish/pull a book and do your first dry-run.

> Prefer to understand each step first, or not use winget? The manual walkthrough
> for steps 2–7 follows below and remains fully supported. The script simply
> automates it.

---

## 1. What you need on the Windows laptop

| Component | Purpose |
|---|---|
| **Python 3.12** | run the executor |
| **This repo** (cloned) | the executor scripts + the pulled target book |
| **IB Gateway** | the authenticated bridge to IBKR (paper) |
| **IBC** *(optional)* | auto-login IB Gateway for unattended runs |
| **`OVERALL_BOOK_SECRET`** | shared secret to verify the book's signature |

The laptop must be **on, awake, and online** at rebalance time. (Task Scheduler
can wake it — see §7.)

---

## 2. Install Python 3.12 and the executor

1. Install **Python 3.12** from <https://www.python.org/downloads/windows/>.
   Tick **"Add python.exe to PATH"**. Verify in PowerShell:
   ```powershell
   py -3.12 --version
   ```
2. Install **Git for Windows** from <https://git-scm.com/download/win>.
3. Clone the repo and create an isolated environment (only the *broker* deps —
   the model stack is NOT needed on this host):
   ```powershell
   cd C:\
   git clone https://github.com/yoursprerit/btc-range-model.git
   cd C:\btc-range-model
   git checkout claude/trading-signals-ibkr-paper-jwyvrc

   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements-ibkr.txt
   ```
   > If `Activate.ps1` is blocked, allow local scripts for your user once:
   > `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
4. Smoke-test the executor (no gateway needed — it just prints the book if one
   exists, or exits cleanly):
   ```powershell
   .\.venv\Scripts\python.exe scripts\ibkr_execute_book.py --help
   ```

---

## 3. Install and configure IB Gateway (paper)

1. Download **IB Gateway** (the lightweight standalone — not full TWS):
   <https://www.interactivebrokers.com/en/trading/ibgateway-stable.php>
2. Launch it and choose **Paper Trading**, then log in with your paper
   credentials (or a dedicated paper username).
3. In IB Gateway: **Configure → Settings → API → Settings**:
   - ☑ **Enable ActiveX and Socket Clients**
   - **Socket port** = **`4002`** (IB Gateway paper default)
   - **Trusted IPs**: add `127.0.0.1` (so the executor connects with no popup)
   - ☐ **Read-Only API** must be **unchecked** (orders need write access)
   - Apply / OK.
4. Leave IB Gateway **running and logged in**. It listens on `127.0.0.1:4002`;
   this is local-only, so Windows Firewall needs no inbound rule.

> **Daily restart:** IB Gateway forces a restart once a day and needs login
> again. For hands-off operation use **IBC** (§6). For a first manual test you
> can just log in by hand.

---

## 4. Set the shared secret

The book is HMAC-signed so the executor can prove it's authentic before trading.
Set the **same** `OVERALL_BOOK_SECRET` value that the publisher uses. Make it a
**user environment variable** so scheduled tasks inherit it:

```powershell
# one-time; opens a new value in the persistent user environment
setx OVERALL_BOOK_SECRET "your-long-random-shared-secret"
```

Close and reopen PowerShell after `setx` (it doesn't affect the current session).
Verify:

```powershell
$env:OVERALL_BOOK_SECRET
```

> If you skip signing, pass nothing and the executor will run but warn that the
> book is unverified. Signing is strongly recommended since the transport (a git
> branch) is readable by anyone with repo access.

---

## 5. Get a target book and run the executor by hand

### 5a. Publish a book (on the cloud/model side — once, to test)
Trigger the GitHub Action **Publish target book (IBKR Option C)** from the
Actions tab (*Run workflow* → pick the `claude/trading-signals-ibkr-paper-jwyvrc`
branch). It commits `data/overall/target_book.json` to that branch. (See
`IBKR_PAPER_TRADING.md` → *Option C* for running the publisher yourself instead.)

### 5b. Pull it onto the laptop
```powershell
cd C:\btc-range-model
git pull --ff-only origin claude/trading-signals-ibkr-paper-jwyvrc
```

### 5c. Dry-run first (NO orders)
With IB Gateway running, preview the plan — it reads the book, verifies the
signature, and diffs against your current paper positions:

```powershell
.\.venv\Scripts\python.exe scripts\ibkr_execute_book.py --file data\overall\target_book.json
```

You should see the published book, `Signature: signature OK`, a freshness line,
your `DU…` account, and the order plan. Compare it against the Overall app's
**"Recommended now (live-adjusted)"** panel — the target weights should match.

### 5d. Execute against paper
When the dry-run looks right, place the orders during US market hours:

```powershell
.\.venv\Scripts\python.exe scripts\ibkr_execute_book.py --file data\overall\target_book.json --execute
```

Confirm the fills in IB Gateway and that resulting positions match the targets.

Useful flags (same as the all-in-one rebalancer, plus book-source options):

| Flag | Meaning |
|---|---|
| `--file` / `--url` / *(stdin)* | where to read the book from |
| `--execute` | actually transmit (default is dry-run) |
| `--band 0.02` | widen the no-trade band (fraction of net-liq) |
| `--fractional` | allow fractional shares (default: whole shares) |
| `--port 4002` | IB Gateway API port (paper) |
| `--max-age-hours 12` | reject a book generated longer ago than this |
| `--require-signature` | refuse an unsigned book |
| `--force` | override the weekend/holiday & freshness guards |

---

## 6. Unattended auto-login with IBC (for automation)

To let a scheduled task trade without you logging in each day, run IB Gateway
under **IBC**:

1. Install IBC for Windows: <https://github.com/IbcAlpha/IBC/releases>
   (unzip to e.g. `C:\IBC`).
2. Copy `deploy\ibc\config.ini.example` (in this repo) to your IBC config and
   fill in credentials + `TradingMode=paper`. **Do not commit the filled-in
   file.**
3. Start IB Gateway via IBC's `StartGateway.bat` (point it at your config). Set
   `AutoRestartTime` in the config to the small hours so the daily restart never
   lands during your rebalance window.
4. Optionally add `StartGateway.bat` to a **logon** scheduled task or the Startup
   folder so the gateway comes up whenever the laptop boots.

> **2FA tip:** unattended login is simplest with a paper username that has 2FA
> disabled. If you must use IBKR Mobile 2FA, follow the IBC docs for
> second-factor handling.

---

## 7. Automate the daily rebalance with Task Scheduler

A PowerShell wrapper is included: `scripts\ibkr_execute_daily.ps1`. It pulls the
latest book from the branch, then runs the executor with `--execute`, logging to
`logs\ibkr_executor.log`.

**Create the scheduled task (PowerShell, one-time):**

```powershell
$ps  = "powershell.exe"
$arg = '-NoProfile -ExecutionPolicy Bypass -File "C:\btc-range-model\scripts\ibkr_execute_daily.ps1"'
$act = New-ScheduledTaskAction -Execute $ps -Argument $arg -WorkingDirectory "C:\btc-range-model"

# 09:45 America/New_York — a few minutes after the US open. Set your laptop's
# clock/zone accordingly, or adjust this local time to equal 09:45 ET.
$trg = New-ScheduledTaskTrigger -Daily -At 9:45AM

# Wake the laptop if asleep, and run whether or not you're logged in.
$set = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable `
        -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries

Register-ScheduledTask -TaskName "IBKR Option C executor" `
    -Action $act -Trigger $trg -Settings $set -RunLevel Highest -Description `
    "Pull the published target book and rebalance the IBKR paper account."
```

Notes:
- The task runs `ibkr_execute_daily.ps1`, which honors `IBKR_BRANCH`, `IBKR_BAND`,
  `IBKR_PORT`, `IBKR_BOOK`, `IBKR_PYTHON` env vars if you want to override
  defaults.
- `OVERALL_BOOK_SECRET` must exist as a **user/system** env var (set via `setx`
  in §4) so the task inherits it.
- **The weekday-only + holiday logic lives in the executor**, so a Saturday fire
  is a safe no-op — you don't need a weekday-only trigger, though you can add
  `-DaysOfWeek` to the trigger if you prefer.
- Ensure **IB Gateway (under IBC) is up before 09:45** and that Windows sleep
  settings allow `-WakeToRun` (Control Panel → Power Options).

**Test the task immediately:**
```powershell
Start-ScheduledTask -TaskName "IBKR Option C executor"
Get-Content C:\btc-range-model\logs\ibkr_executor.log -Tail 40
```

---

## 8. Daily operating picture

1. **Publisher** (cloud) emits a fresh signed `target_book.json` **every day** —
   weekends and holidays included, since Bitcoin keeps trading and the signals
   keep moving while the US market is closed.
2. **Laptop** wakes ~09:45 ET → `git pull` → executor verifies + trades paper.
   The executor only ever trades on US market days: its weekend/holiday guard
   makes any Saturday/holiday fire a safe no-op.
3. Guards keep it safe: weekend/holiday skip (executor side), stale-book
   refusal, paper-account check, HMAC verification, and the no-trade band
   suppressing tiny churn.
4. Review `logs\ibkr_executor.log` and the IBKR paper account periodically.

---

## 9. Troubleshooting (Windows)

| Symptom | Fix |
|---|---|
| `Could not connect to IB Gateway` | Gateway not running / not logged in, or API not enabled on port 4002. Re-check §3. |
| Connects but `not a paper account` abort | You're on a live login. Switch IB Gateway to **Paper Trading**. (Do **not** use `--allow-nonpaper` unless you truly intend live.) |
| `signature MISMATCH` / `no signature present…` | `OVERALL_BOOK_SECRET` on the laptop doesn't match the publisher's, or the book was edited. Re-set the secret (§4). |
| `book generated …h ago (> 12h) — stale` | The publisher didn't run today, or the pull failed. Re-publish, `git pull`, retry. Use `--max-age-hours` only if you understand the risk. |
| `Not trading: … weekend/holiday` | Working as intended. `--force` overrides. |
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once. |
| Scheduled task didn't run overnight | Laptop asleep without `-WakeToRun`, or on battery with battery guards. Re-check §7 power settings. |
| Orders don't fill | Outside US market hours, or the paper account lacks buying power / the symbol is halted. Market orders fill during RTH. |

---

## 10. Safety recap

- **Paper only** — non-`DU` accounts are refused by default.
- **Dry-run is the default** — orders require `--execute`.
- **Signed books only** (recommended) — set `--require-signature` to enforce.
- **Freshness guards** — stale bar or stale generation aborts the run.
- Leveraged sleeves (MSTU 2×, SOXL 3×, UGL/NUGT/ERX 2×) trade normally but are
  the volatile part of the book — paper-test thoroughly before trusting it.
- This is for **paper** validation. Real money is a separate, deliberate step
  beyond this repo's scope.
