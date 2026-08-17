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

## 0.5 Prerequisites — have these ready BEFORE running the setup script

The setup script cannot create or configure anything on IBKR's side; it only
writes credentials you already hold into IBC's config. Running it without these
in hand leaves you with a half-configured box.

**On IBKR (do these first — full detail in
[`IBKR_PAPER_TRADING.md` → *Prerequisites*](IBKR_PAPER_TRADING.md#prerequisites--do-these-before-any-setup-script)):**

| # | Prerequisite | Why it blocks setup |
|---|---|---|
| 1 | A **paper account**, created in Client Portal | Nothing to trade against otherwise |
| 2 | The **paper username + password** (separate from your live login) | These are `-IbUser` / `-IbPassword` |
| 3 | Account id starts with **`DU`** | The executor aborts on any non-`DU` account |
| 4 | **2FA disabled** on that paper login | The scheduled task can't answer a phone prompt |
| 5 | **Market data shared** to the paper account *(recommended)* | Without quotes, marketable limits fall back to unprotected market orders |
| 6 | The **`OVERALL_BOOK_SECRET`** value used by the publisher | A mismatch aborts at signature verification. You **cannot read this back out of GitHub** — see §4 |

**On the laptop:**

| Prerequisite | Notes |
|---|---|
| **Windows 10/11 with `winget`** | The `-InstallPython` phase needs it. Missing? Install *App Installer* from the Microsoft Store, or install Python 3.12 + Git by hand (§2). |
| **An elevated PowerShell** | Run as Administrator — required for the installer and scheduled-task phases. |
| **The repo cloned** | `git clone https://github.com/yoursprerit/btc-range-model.git C:\btc-range-model` |
| **Power settings that allow wake** | The daily task registers `-WakeToRun`, but Windows must permit it (§7). |
| **Your machine's timezone** | The task trigger fires in **local** time; the target is 2:30 PM US Central. Pass `-TaskTime` accordingly (§7). |
| **Git write credentials** *(optional)* | Only needed to publish the execution report back so the cloud app's *Executed Book* tab updates. Trading works without it (§8). |

Only once all of the above is true should you run the quick start below.

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
| Install Python 3.12 + Git (winget) | **Creating the IBKR paper account** + login (§0.5) |
| Create `.venv`, install `requirements-ibkr.txt` | IBKR **licence click-through** during install |
| Set `OVERALL_BOOK_SECRET` (generates one if omitted) | **First IB Gateway login / 2FA** |
| Download + launch the IB Gateway installer | Ticking the API settings (or let IBC enforce them) |
| Download IBC + template its `config.ini` (paper) | Sharing market data to the paper account (§0.5) |
| Register the daily Task Scheduler job | Deciding to `--execute` |
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
| **An IBKR paper account (`DU…`) + its login** | what gets traded — create it first (§0.5) |
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

   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements-ibkr.txt
   ```
   That file installs **`ib_async` + `pandas`/`numpy`** — the executor and the
   shared `target_book` / `ibkr_common` modules all import pandas, and
   `ib_async` does not pull it in. Do **not** install `requirements.txt` here:
   that is the model/Streamlit stack and this host does not run the model.
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
`OVERALL_BOOK_SECRET` must hold **one identical value in three places**, and only
one of them signs:

| Role | Where the value lives | What it does |
|---|---|---|
| **Publisher** | **GitHub repo secret** (`Settings → Secrets and variables → Actions`), consumed by `.github/workflows/publish-target-book.yml` | **Signs** the book |
| Streamlit app | `st.secrets` (Manage app → Secrets), or an env var | Verifies |
| **This laptop** | **user** environment variable, via `setx` | Verifies |

Only the GitHub Action signs. **The Streamlit app is a verifier, not the
publisher** — setting the value there does not change how any book is signed.

```powershell
# one-time; writes a new value into the persistent user environment
setx OVERALL_BOOK_SECRET "your-long-random-shared-secret"
```

Close and reopen PowerShell after `setx` — it does not affect the current
session. Verify:

```powershell
[Environment]::GetEnvironmentVariable('OVERALL_BOOK_SECRET','User')   # persisted value
$env:OVERALL_BOOK_SECRET                                              # visible to THIS shell
```

Both must be non-empty. The first is what Task Scheduler will inherit; the
second is what your interactive dry-run will use.

### ⚠️ The setup script mints a NEW secret if you don't give it one

`setup_windows_option_c.ps1 -Secret` (and therefore `-All`) uses `-BookSecret`
if supplied, keeps an existing user value if there is one, and **otherwise
generates a fresh random 40-character secret** and prints it in magenta:

```
[warn] generated a new secret - set the SAME value on the PUBLISHER
[warn] (GitHub repo secret OVERALL_BOOK_SECRET):
```

That warning is a required follow-up action, not a status message. Until you
copy the value into the **GitHub repo secret** and re-publish, the publisher is
still signing with the old key and both verifiers will reject every book.

Avoid this entirely by passing the publisher's existing value:

```powershell
... setup_windows_option_c.ps1 -All -BookSecret "the-publishers-existing-value" -IbUser … -IbPassword …
```

### Rotating the secret

**GitHub Actions secrets are write-only — you cannot read the old value back
out.** If it isn't written down somewhere, your only route is forward. All three
places must change, and a book must be re-signed:

1. **GitHub → repo Settings → Secrets and variables → Actions →
   `OVERALL_BOOK_SECRET` → Update** to the new value.
2. **Re-publish.** Changing the secret does *not* re-sign the book already
   committed — the signature is fixed at publish time. Run the **Publish target
   book (IBKR Option C)** Action, or press **🚀 Publish new target book** in the
   Target Book app.
3. **Streamlit → Manage app → Secrets** → set the new value, then **reboot the
   app**.
4. **Laptop:** `setx OVERALL_BOOK_SECRET "…"`, then open a new PowerShell.
5. `git pull` on the laptop and re-run the dry-run (§5c).

> **Streamlit precedence:** `app/target_book_app.py` checks `st.secrets`
> **first** and only then environment variables. A stale value in the Secrets
> panel silently wins over a correct env var.

> **Whitespace:** the comparison is over raw bytes. A trailing newline or space
> pasted into any of the three places produces a mismatch.

### ⚠️ No secret means the signature is NOT checked

If `OVERALL_BOOK_SECRET` is unset, `verify_signature()` **returns success** with
the message `signed but no secret provided to verify`, and the run continues.
A dry-run that reports that line has verified nothing — it is not the same as
`signature OK`. This bites most often when you run in a PowerShell window opened
*before* `setx`, since `setx` only affects new processes.

Pass `--require-signature` to refuse an unsigned book outright. Signing is
strongly recommended: the transport (a git branch) is readable by anyone with
repo access.

### Checking a book against a candidate secret

Stdlib only — works even before the venv exists:

```powershell
.\.venv\Scripts\python.exe -c "import os,json,hmac,hashlib; p=json.load(open('data/overall/target_book.json')); b={k:v for k,v in p.items() if k!='signature'}; c=json.dumps(b,sort_keys=True,separators=(',',':')).encode(); print('MATCH' if hmac.compare_digest(hmac.new(os.environ['OVERALL_BOOK_SECRET'].encode(),c,hashlib.sha256).hexdigest(), p['signature']['value']) else 'MISMATCH')"
```

---

## 5. Get a target book and run the executor by hand

### 5a. Publish a book (on the cloud/model side)
The publisher runs on `main` on its own daily schedule, so a fresh signed book is
normally already committed to `data/overall/target_book.json` — you usually just
pull it. To force one now, trigger the GitHub Action **Publish target book (IBKR
Option C)** from the Actions tab (*Run workflow* → `main`), or use the app's
**🚀 Publish new target book** button. (See `IBKR_PAPER_TRADING.md` → *Option C*
for running the publisher yourself instead.)

### 5b. Pull it onto the laptop
```powershell
cd C:\btc-range-model
git pull --ff-only origin main
```

### 5c. Dry-run first (NO orders)
With IB Gateway running, preview the plan — it reads the book, verifies the
signature, and diffs against your current paper positions:

```powershell
.\.venv\Scripts\python.exe scripts\ibkr_execute_book.py --file data\overall\target_book.json
```

You should see the published book, `Signature: signature OK`, a freshness line,
your `DU…` account, and the order plan. Compare it against the Overall app's
**"Recommended Live Possible Targetbook"** panel — the target weights should match.

> **Read the `Signature:` line, don't just check the exit code.** Only
> `signature OK` means the book was cryptographically verified.
> `signed but no secret provided to verify` means `OVERALL_BOOK_SECRET` was not
> visible to this shell and **nothing was checked** — see §4.

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
| `--max-age-hours 36` | reject a book generated longer ago than this (default 36 spans the 7:15-AM-CT publish anchor → next day's 2:30-PM-CT executor slot, so yesterday's book still trades if today's publish was withheld) |
| `--order-type` | `marketable-limit` (default), `moc`, or `market` — see IBKR_PAPER_TRADING.md § Order routing (env `IBKR_ORDER_TYPE`) |
| `--slippage-cap 0.005` | how far through the touch a marketable limit prices (env `IBKR_SLIPPAGE_CAP`) |
| `--require-signature` | refuse an unsigned book |
| `--force` | override the weekend/holiday & freshness guards |
| `--outside-rth` | **manual after-hours trading.** Stamps `outsideRth` on the orders and forces marketable-limit, because IBKR *rejects* MARKET and MOC orders outside regular hours. With no live quote it prices off the book's own `exec_price`; a leg it still cannot price is skipped rather than sent blind. Unfilled limits stay working — the market-order escalation does not exist outside RTH. **The scheduled 2:30 PM CT wrapper never passes this**, so automation is unaffected. |
| `--market-data-type 3` | request **delayed** quotes. Delayed data is free and needs no subscription, but IBKR only serves it when asked — use this if every leg logs error 10089 / `falling back to MARKET`. Pair with a wider `--slippage-cap` since the reference is 15 minutes stale. |

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
4. **Required for automation — make the gateway come back on its own.** Nothing
   in `setup_windows_option_c.ps1` starts IB Gateway; it only installs it and
   templates IBC's config. Task Scheduler will fire the executor at 2:30 PM
   whether or not the gateway is up, and the run fails if it isn't. Put
   `StartGateway.bat` in your Startup folder:
   ```powershell
   $s = (New-Object -ComObject WScript.Shell).CreateShortcut(
          "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup\StartGateway.lnk")
   $s.TargetPath = 'C:\IBC\StartGateway.bat'; $s.WorkingDirectory = 'C:\IBC'; $s.Save()
   ```
   This also covers IB Gateway's forced once-a-day restart: IBC re-logs-in only
   while it is the process supervising the session.

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

# 2:30 PM US Central (3:30 PM ET) — 30 minutes before the equity close. Task
# Scheduler triggers fire in the LAPTOP's local time, so this literal is right
# on a Central-time machine; on any other zone set the local equivalent
# (e.g. 3:30PM on Eastern, 12:30PM on Pacific).
$trg = New-ScheduledTaskTrigger -Daily -At 2:30PM

# Wake the laptop if asleep; keep running on battery; catch up on a missed fire.
$set = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable `
        -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries

Register-ScheduledTask -TaskName "IBKR Option C executor" `
    -Action $act -Trigger $trg -Settings $set -RunLevel Highest -Description `
    "Pull the published target book and rebalance the IBKR paper account."
```

Notes:
- The task runs `ibkr_execute_daily.ps1`, which pulls the book from **`main`** by
  default (where the publisher commits it). Override with `-Branch <name>` or the
  `IBKR_BRANCH` env var; `IBKR_BAND`, `IBKR_PORT`, `IBKR_BOOK`, `IBKR_PYTHON`
  override the other defaults.
- `OVERALL_BOOK_SECRET` must exist as a **user/system** env var (set via `setx`
  in §4) so the task inherits it. The task reads the environment fresh at each
  fire, so setting it after registering the task is fine.
- **The task runs only while you are logged on.** `Register-ScheduledTask`
  without an explicit `-Principal` registers the current user with an
  *interactive* logon type. A locked screen is fine; a signed-out or shut-down
  machine is not. To run signed-out, re-register with
  `-User "$env:USERNAME" -LogonType S4U`, but note that S4U breaks DPAPI — which
  breaks Git Credential Manager, so pair it with the SSH deploy key in §8.
- **Registering the task needs an elevated PowerShell** (`-RunLevel Highest`).
  If your `-All` run died before phase 7, the task does not exist; re-run
  `... setup_windows_option_c.ps1 -Task` as Administrator.
- **The weekday-only + holiday logic lives in the executor**, so a Saturday fire
  is a safe no-op — you don't need a weekday-only trigger, though you can add
  `-DaysOfWeek` to the trigger if you prefer.
- Ensure **IB Gateway (under IBC) is up before 2:30 PM CT** and that Windows
  sleep settings allow `-WakeToRun` (Control Panel → Power Options).

**Confirm it registered, and when it will fire:**
```powershell
Get-ScheduledTask -TaskName "IBKR Option C executor" |
    Get-ScheduledTaskInfo | Select-Object NextRunTime, LastRunTime, LastTaskResult
Get-TimeZone      # NextRunTime is LOCAL time — confirm it lands on 2:30 PM US Central
```

**Test the task immediately:**
```powershell
Start-ScheduledTask -TaskName "IBKR Option C executor"
Get-Content C:\btc-range-model\logs\ibkr_executor.log -Tail 40
```

> This is the only test that exercises the whole chain, but note the wrapper
> passes `--execute` — **it places real paper orders now**, not a dry-run. That
> is safe (paper-only guard) but fills at the current price rather than your
> 2:30 PM slot. When the scheduled fire comes around it will find you inside the
> no-trade band and no-op.

---

## 8. Git write credentials for the execution report (optional)

After trading, `ibkr_execute_daily.ps1` commits `data\overall\executed_book.json`
and pushes it to `main` so the cloud app's **Executed Book** tab shows the fills.
This is **cosmetic** — trades execute and the report is written locally either
way. Without credentials the wrapper logs
`WARN: could not push execution report … rolling back`, resets to `origin/main`,
and carries on.

The push runs from **Task Scheduler, detached — no console, no prompt**. The
credential must therefore be readable non-interactively; anything that would pop
a dialog just fails.

You do **not** need a git identity: the wrapper commits with
`git -c user.name="ibkr-executor" -c user.email="executor@localhost"`.

Check which transport you're on first:

```powershell
cd C:\btc-range-model
git remote -v
git branch --show-current      # should be `main` — the wrapper pushes HEAD:main
```

### Option A — deploy key (scoped to this repo, no expiry)

```powershell
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.ssh" | Out-Null
ssh-keygen -t ed25519 -f "$env:USERPROFILE\.ssh\id_btc_executor" -N '""' -C "btc-executor-laptop"
Get-Content "$env:USERPROFILE\.ssh\id_btc_executor.pub"
```

`ssh-keygen` will **not** create `.ssh` for you — the `New-Item` line is what
prevents `Saving key … failed: No such file or directory`.

Paste the printed public key at repo **Settings → Deploy keys → Add deploy key**
and tick **Allow write access**. Then:

```powershell
git remote set-url origin git@github.com:yoursprerit/btc-range-model.git
Add-Content "$env:USERPROFILE\.ssh\config" @"

Host github.com
  IdentityFile ~/.ssh/id_btc_executor
  IdentitiesOnly yes
"@
ssh -T git@github.com     # accept the host key ONCE, so the detached task never stalls on it
```

- Expect `Hi yoursprerit/btc-range-model! You've successfully authenticated…`.
- **If it prompts for a passphrase**, the `-N '""'` quoting didn't take. Clear it
  with `ssh-keygen -p -f "$env:USERPROFILE\.ssh\id_btc_executor"` (press Enter
  twice) — a passphrased key cannot work unattended.
- If OpenSSH complains the permissions are too open:
  `icacls "$env:USERPROFILE\.ssh\id_btc_executor" /inheritance:r /grant:r "$($env:USERNAME):(R)"`

### Option B — fine-grained PAT via Git Credential Manager

GCM ships with Git for Windows and stores into Windows Credential Manager, which
any process running as your user can read.

Create the token at **GitHub → Settings → Developer settings → Personal access
tokens → Fine-grained tokens**: repository access **only** `yoursprerit/btc-range-model`,
permission **Contents: Read and write**. That single permission is all a push
needs. Mind the expiry — when it lapses the push starts failing silently.

Store it without any UI:

```powershell
$pat = 'github_pat_xxxxxxxx'
@"
protocol=https
host=github.com
username=yoursprerit
password=$pat

"@ | git credential approve
```

The blank line inside the here-string is required — it terminates the input.

> Embedding the token in the remote URL (`https://<PAT>@github.com/…`) also
> works but writes it in cleartext into `.git/config` and leaks it into verbose
> git errors. Prefer A or B.

### Verify it works *headless*

An interactive push can succeed via a prompt the scheduler can never show, so
force git to fail instead of asking:

```powershell
$env:GIT_TERMINAL_PROMPT = 0
git push origin HEAD:main
```

`could not read Username for 'https://github.com'` means the credential is not
stored and the scheduled push will fail the same way.

### Or turn the push off

```powershell
setx IBKR_NO_PUSH_REPORT 1
```

A defensible choice — it keeps any write credential off the trading laptop.

---

## 9. Pre-flight — will it actually fire today?

A green dry-run proves the book and the gateway were fine *at that moment*. It
does not prove the scheduled run will work. Walk this list once after setup:

| # | Check | Command / where |
|---|---|---|
| 1 | Task exists and `NextRunTime` is today at your local equivalent of 2:30 PM CT | `Get-ScheduledTask -TaskName "IBKR Option C executor" \| Get-ScheduledTaskInfo` |
| 2 | Machine timezone is what you assumed when you set `-TaskTime` | `Get-TimeZone` |
| 3 | Secret is persisted **and** visible to a fresh shell | `[Environment]::GetEnvironmentVariable('OVERALL_BOOK_SECRET','User')` |
| 4 | Dry-run prints `Signature: signature OK` — *not* `no secret provided to verify` | §5c |
| 5 | IB Gateway is listening, and will still be at 2:30 | `Test-NetConnection 127.0.0.1 -Port 4002` → `TcpTestSucceeded : True` |
| 6 | Gateway restarts on its own after a reboot / IBKR's daily restart | §6 step 4 |
| 7 | Repo is on `main` and fast-forwardable | `git branch --show-current`; `git pull --ff-only origin main` |
| 8 | Push credential works headless *(only if you want the report)* | §8 |
| 9 | Laptop will be **on and logged on** at 2:30 (locked is fine) | §7 |
| 10 | Today's book published and audit passed | Streamlit Target Book tab, or `git log -1 --oneline origin/main -- data/overall/target_book.json` |

The single most common gap is **#5/#6** — nothing in the setup script starts IB
Gateway.

---

## 10. Daily operating picture

1. **Publisher** (cloud) emits a fresh signed `target_book.json` **every day** —
   weekends and holidays included, since Bitcoin keeps trading and the signals
   keep moving while the US market is closed.
2. **Laptop** wakes ~2:30 PM CT (3:30 PM ET) → `git pull` → executor verifies +
   trades paper.
   The executor only ever trades on US market days: its weekend/holiday guard
   makes any Saturday/holiday fire a safe no-op.
3. Guards keep it safe: weekend/holiday skip (executor side), stale-book
   refusal, paper-account check, HMAC verification, and the no-trade band
   suppressing tiny churn.
4. Review `logs\ibkr_executor.log` and the IBKR paper account periodically.

---

## 11. Troubleshooting (Windows)

| Symptom | Fix |
|---|---|
| `Could not connect to IB Gateway` | Gateway not running / not logged in, or API not enabled on port 4002. Re-check §3. |
| Connects but `not a paper account` abort | You're on a live login. Switch IB Gateway to **Paper Trading**. (Do **not** use `--allow-nonpaper` unless you truly intend live.) |
| `signature MISMATCH` / `no signature present…` | The laptop's (or Streamlit's) `OVERALL_BOOK_SECRET` doesn't match the **publisher's**, or the book was edited. Usually means the setup script minted a new secret and the GitHub repo secret was never updated — full rotation procedure in §4. Remember to **re-publish**: changing the secret does not re-sign an already-committed book. |
| `Signature: signed but no secret provided to verify` | Not an error, but **nothing was verified**. `OVERALL_BOOK_SECRET` isn't visible to that process — usually a shell opened before `setx`. Open a new PowerShell (§4). |
| `ModuleNotFoundError: No module named 'pandas'` | The venv is incomplete. `.\.venv\Scripts\python.exe -m pip install -r requirements-ibkr.txt`, or re-run `... setup_windows_option_c.ps1 -Venv`. |
| `The string is missing the terminator` / `Missing closing '}'` running a `.ps1` | The script file picked up non-ASCII characters. Windows PowerShell 5.1 reads BOM-less `.ps1` as ANSI, and a UTF-8 em dash decodes to a curly quote that opens a string. Keep these scripts pure ASCII, or save them UTF-8 **with** BOM. |
| `Saving key … failed: No such file or directory` from `ssh-keygen` | `%USERPROFILE%\.ssh` doesn't exist yet — create it first (§8). |
| `could not push execution report … rolling back` | No git write credential on this host. Trading still happened; only the cloud *Executed Book* tab is stale. Set one up (§8) or silence it with `setx IBKR_NO_PUSH_REPORT 1`. |
| `book generated …h ago (> 30h) — stale` | The publisher didn't run today (or yesterday), or the pull failed. Re-publish, `git pull`, retry. Use `--max-age-hours` only if you understand the risk. |
| `Not trading: … weekend/holiday` | Working as intended. `--force` overrides. |
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once. |
| Scheduled task didn't run overnight | Laptop asleep without `-WakeToRun`, on battery with battery guards, or **signed out** (the default principal only runs while logged on). Re-check §7. |
| Task fired at the wrong hour | `-TaskTime` is **local machine time**, not Central. `Get-TimeZone`, then re-register with the local equivalent (§7). |
| Task exists but never runs / `LastTaskResult` non-zero | Read `logs\ibkr_executor.log` — the wrapper logs every step. Work down the §9 pre-flight list. |
| Orders don't fill | Outside US market hours, or the paper account lacks buying power / the symbol is halted. Market orders fill during RTH. |
| Log stops right after `Published book …` with no `Signature:` line and no `done` | The classic detached-run failure, fixed in the wrapper as of 2026-08-17. Redirected stdout on Windows is encoded with the ANSI codepage, so the `→` in the book printout raised `UnicodeEncodeError`; `2>&1` then turned the traceback into a terminating error under `$ErrorActionPreference='Stop'`, killing the wrapper before it could log anything. `git pull` to get the fix. Note the same command works interactively — a console stdout takes the Unicode path, so this only ever shows up under Task Scheduler. |
| Log shows `ù` where `—` should be | Cosmetic mojibake from the same encoding mismatch (cp1252 out, cp437 in); fixed by the same change. |

---

## 12. Safety recap

- **Paper only** — non-`DU` accounts are refused by default.
- **Dry-run is the default** — orders require `--execute`.
- **Signed books only** (recommended) — set `--require-signature` to enforce.
- **Freshness guards** — stale bar or stale generation aborts the run.
- Leveraged sleeves (MSTU 2×, SOXL 3×, UGL/NUGT/ERX 2×) trade normally but are
  the volatile part of the book — paper-test thoroughly before trusting it.
- This is for **paper** validation. Real money is a separate, deliberate step
  beyond this repo's scope.
