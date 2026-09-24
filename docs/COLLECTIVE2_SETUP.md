# Collective2 setup runbook

Step-by-step setup for publishing the daily **paper** target book to a
Collective2 (C2) strategy, from zero to a verified daily run. How the
publisher works and its safety guards are in [`COLLECTIVE2.md`](COLLECTIVE2.md).

**End state:** every weekday at 2:30 PM America/Chicago, cron-job.org triggers
`.github/workflows/publish-c2.yml`, which sends the day's signed book to C2
with `scripts/publish_c2.py`. Nothing runs on your PC.

```
7:15 AM CT  publish-target-book.yml  → data/overall/target_book.json (signed)
2:30 PM CT  cron-job.org → publish-c2.yml → C2 SetDesiredPositions
2:30 PM CT  your IBKR executor (unchanged) trades the same book
```

| Part | What you get | Time |
|---|---|---|
| [1. Collective2](#1-collective2-strategy-and-api-key) | `C2_STRATEGY_ID`, `C2_API_KEY` | ~10 min |
| [2. GitHub repo](#2-github-repository-secrets) | Secrets the workflow reads | ~3 min |
| [3. cron-job.org](#3-cron-joborg-230-pm-ct-trigger) | On-time 2:30 PM CT trigger | ~5 min |
| [4. Testing](#4-testing) | A verified first publish | ~15 min, partly during market hours |

---

## 1. Collective2: strategy and API key

### 1.1 Create an account
1. Sign up at **collective2.com** and verify your email.
2. Complete your profile. Your manager name is shown next to the strategy.

### 1.2 Create the strategy
1. Open **Create a Strategy** (manager area of your account menu).
2. Fill in:

   | Field | Value |
   |---|---|
   | Name | Your strategy name (e.g. *Compass Rotation*). Check it is not already taken; C2 names are hard to change once you have subscribers. |
   | Instruments / security types | **Stocks** (US equities and ETFs). Nothing else is needed. |
   | Starting capital | **$100,000** recommended. The smallest weights (e.g. XLE 3.2%, OIH 5.2% at ~$395) must still size to whole shares. Do not go below ~$25k. |
   | Visibility | **Private** while testing; make it public once the record builds. |
   | Price | $0 / unset for now. |
   | Description | e.g. *Daily-rebalanced, regime-gated trend rotation across gold, energy, semis, crypto (via IBIT/ETHA spot ETFs) and AI/thematic ETFs, including 2–3× leveraged sleeves.* |
3. Save.

### 1.3 Get `C2_STRATEGY_ID`
Open the strategy's page. The number in the URL is the ID, e.g.
`collective2.com/details/`**`123456789`**. Digits only.

### 1.4 Get `C2_API_KEY`
1. Go to **collective2.com/apikey** (or **collective2.com/account-info**) and
   sign in.
2. Create a new API key:
   * Name: `github-publish`.
   * Role: one that can **manage and trade your own strategies**. A read-only
     role cannot set positions.
3. Copy the key immediately. Treat it like a password: anyone holding it can
   trade the strategy.

### 1.5 Verify both values (any time)
From any terminal (Windows PowerShell: use `curl.exe`):

```bash
curl -s -H "Authorization: Bearer YOUR_C2_API_KEY" \
  "https://api4-general.collective2.com/Strategies/GetStrategyDetails?StrategyId=YOUR_STRATEGY_ID"
```

| Result | Meaning |
|---|---|
| JSON with your strategy name and `"ModelAccountValue": 100000` | Key and ID both work. |
| `401` | Wrong key, or missing the `Bearer ` prefix. |
| `403` / empty `Results` | Key role is read-only, or the ID is not your strategy. |

---

## 2. GitHub repository secrets

1. Open **github.com/yoursprerit/btc-range-model → Settings → Secrets and
   variables → Actions → New repository secret**.
2. Add:

   | Name | Value |
   |---|---|
   | `C2_API_KEY` | The key from 1.4 |
   | `C2_STRATEGY_ID` | The number from 1.3 (digits only) |

3. Confirm `OVERALL_BOOK_SECRET` is already listed. The book publish uses it to
   sign the book, and the C2 workflow uses it to verify the signature
   (`--require-signature`); without it every run aborts.

Until `C2_API_KEY` and `C2_STRATEGY_ID` exist, the workflow finishes green with
a *Collective2 not configured* notice and sends nothing.

The workflow must be on `main`: GitHub only fires `on: schedule` from the
default branch, and the cron-job.org request below dispatches `main`.

---

## 3. cron-job.org: 2:30 PM CT trigger

GitHub's own cron slots in `publish-c2.yml` (every 15 minutes in the
afternoon) are best-effort and often fire 1–3 hours late, so with only 30
minutes between 2:30 PM CT and the close they rarely land in time. The
cron-job.org job calling GitHub's `workflow_dispatch` API is the real trigger.

### 3.1 GitHub token (skip if the 7:16 AM book job already works)
The existing book-publish job ([`EXTERNAL_SCHEDULER.md`](EXTERNAL_SCHEDULER.md))
already holds a suitable token; it is scoped to the repository, so it can
trigger this workflow too. Only if you need a new one:

1. GitHub → **Settings → Developer settings → Personal access tokens →
   Fine-grained tokens → Generate new token**.
2. Repository access: **Only select repositories → `btc-range-model`**.
3. Repository permissions: **Actions → Read and write** (nothing else).
4. Generate, copy it (shown once), and note the expiry date in your calendar.
   An expired token stops the 2:30 PM trigger with no other warning than
   cron-job.org's failure email.

### 3.2 Create the job

**Option A: clone the existing 7:16 AM job (quickest)**
1. cron-job.org → **Dashboard → Cronjobs** → on the job calling
   `publish-target-book.yml`, click **⋮ → Copy/Clone**.
2. Change only:
   * **Title:** `C2 publish 2:30 PM CT`
   * **URL:**
     ```
     https://api.github.com/repos/yoursprerit/btc-range-model/actions/workflows/publish-c2.yml/dispatches
     ```
   * **Schedule:** as in Option B step 2.
3. Save. Method, body, headers and token carry over.

**Option B: create it from scratch**
1. **Create cronjob** → *Common* tab:
   * **Title:** `C2 publish 2:30 PM CT`
   * **URL:** the `publish-c2.yml/dispatches` URL above.
2. **Execution schedule → Custom:**
   * Minutes `30`, Hours `14` (2:30 PM), every day of month, every month.
   * Days of week: **Mon–Fri** only.
   * **Time zone: America/Chicago** (tracks the CDT/CST switch automatically).
3. *Advanced* tab:
   * **Request method:** `POST`
   * **Request body:**
     ```
     {"ref":"main"}
     ```
   * **Headers:**

     | Key | Value |
     |---|---|
     | `Authorization` | `Bearer <YOUR_GITHUB_TOKEN>` |
     | `Accept` | `application/vnd.github+json` |
     | `X-GitHub-Api-Version` | `2022-11-28` |
4. **Notifications:** enable *notify on failure*.
5. **Create**.

A dispatch skips only the workflow's 2:30 PM guard. `publish_c2.py` still skips a
book already sent, weekends/holidays and anything outside regular hours, so a
holiday fire or an overlap with a GitHub backup slot is harmless.

---

## 4. Testing

Stages 1–3 are safe at any hour; stage 4 sends real (model-account) orders.

### 4.1 Credentials (any time)
Run the `curl` check in [1.5](#15-verify-both-values-any-time).

### 4.2 Local preview (any time, nothing sent)
From the repo folder, using the repo's virtual environment (the system
Python usually lacks pandas: `ModuleNotFoundError: No module named 'pandas'`).

Windows PowerShell:
```powershell
cd C:\path\to\btc-range-model
git pull
$env:C2_API_KEY="your-c2-api-key"
$env:C2_STRATEGY_ID="123456789"
.\.venv\Scripts\python.exe scripts\publish_c2.py
```

macOS / Linux:
```bash
git pull
C2_API_KEY=… C2_STRATEGY_ID=… .venv/bin/python scripts/publish_c2.py
```

* No `.venv`? `python -m pip install "numpy>=1.26,<2.3" "pandas>=2.1,<2.3"`.
* No key yet? Add `--capital 100000` instead of the two variables.
* `$env:` values last only for that PowerShell window.

Expected: a share table sized against C2's model-account value, ending with
**`DRY-RUN — nothing sent.`** Check that ~98–99% of capital is invested and no
important ticker is listed under `dropped:` (too small for one share → raise
the strategy's capital).

With no `OVERALL_BOOK_SECRET` set locally the signature line reads *signed but
no secret provided to verify*; that is fine for a preview.

### 4.3 cron-job.org → GitHub chain (any time)
1. cron-job.org → the job → **Test run**. Expect **HTTP 204**.
2. GitHub → **Actions** → a *Publish book to Collective2* run appears within
   seconds (event `workflow_dispatch`). Open it:

   | Output | Meaning |
   |---|---|
   | `SKIP: outside regular trading hours` or `… is a weekend` | Chain works end to end: secrets present, book signature and freshness verified. Nothing sent. |
   | *Collective2 not configured* notice | `C2_API_KEY` / `C2_STRATEGY_ID` secret missing or misnamed. |
   | `ABORT: book signature did not verify` | `OVERALL_BOOK_SECRET` does not match the secret that signed the book. |
   | `ABORT: book failed validation` | Today's book is stale or its publish-time audit failed; check the book publish workflow. |

### 4.4 First real publish (weekday, 8:30 AM–3:00 PM CT)
1. GitHub → **Actions → Publish book to Collective2 → Run workflow** (branch
   `main`), or **Test run** in cron-job.org.
2. Open the run's **summary**: one `new BUY …` line per ticker and **no
   `REJECTED` lines**.
3. A commit `chore(c2): published book as-of …` appears on `main` (the
   already-sent marker, `data/overall/c2_publish_state.json`).
4. On collective2.com, the strategy shows the orders filled and positions
   matching the summary.
5. Run it again: it must print `SKIP: book for … already sent to C2`, proving
   a book is never sent twice.

A rejected ticker (most likely a smaller ETF such as ARTY, WGMI or GRID) fails
the run and is not recorded as sent; exclude or remap it in the publisher and
re-run.

### 4.5 First scheduled day
The next weekday after 2:30 PM CT, confirm a run with event
`workflow_dispatch` (cron-job.org) in the Actions tab and a new
`chore(c2): published book …` commit. After a few clean days, rely on the
failure emails from cron-job.org and GitHub.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| cron-job.org `401 Bad credentials` | GitHub token pasted wrong, or missing `Bearer `. |
| cron-job.org `403` / `404` | Token lacks **Actions: Read and write**, is not scoped to `btc-range-model`, or the URL has a typo. |
| cron-job.org `422` | Body is not exactly `{"ref":"main"}`. |
| `ModuleNotFoundError: No module named 'pandas'` (local) | Run with `.venv\Scripts\python.exe`, or install numpy/pandas as in 4.2. |
| `ABORT: … C2_API_KEY and C2_STRATEGY_ID` (local) | Set both variables, or preview with `--capital 100000`. |
| C2 `HTTP 401` / `403` in the run | Wrong key, read-only role, or wrong strategy ID. Re-check with 1.5. |
| `ABORT: --require-signature but OVERALL_BOOK_SECRET is not set` (Actions) | Add the `OVERALL_BOOK_SECRET` repo secret (same value the book publish uses). |
| `ABORT: book holds positions but none sized to a whole share` | Model capital too small; raise it on C2. |
| `FAILED: C2 rejected part of the request` | See the `REJECTED` line in the summary for the ticker and reason. |
| No run at 2:30 PM | cron-job.org job disabled or token expired. GitHub's 15-minute backup slots only help if one lands before the 3:00 PM CT close; otherwise that day is skipped and C2 keeps the previous positions. |

## Later

* **Go public:** switch the strategy to public and set a monthly price once
  the verified record is a few months old.
* **Trades-Own-Strategy:** once the same book trades with real money at IBKR,
  apply on the strategy page for the certification (trust badge; revenue
  share 60% instead of 50%).
* **BTC/ETH signals** appear in the book as `BTC`/`ETH` but are published to
  C2 as **IBIT**/**ETHA** (sized on the ETF's price). The first time one fires,
  check that day's run summary shows a sensible IBIT/ETHA share count.
