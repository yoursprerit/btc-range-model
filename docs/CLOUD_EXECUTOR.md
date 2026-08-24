# Cloud executor — run IB Gateway + the Option C executor on a free VM

Host the **executor half** of Option C in the cloud so paper rebalancing runs
without a laptop. The signals are already published in the cloud (the GitHub
Action); this box only **consumes the signed target book and places paper
orders**, so it stays small and cheap — or free.

```
 GitHub Action (publisher)  ──commits target_book.json──►  this VM
                                                             ├── IB Gateway (paper, headless via Docker/IBC)
                                                             └── cron → ibkr_execute_daily.sh → ibkr_execute_book.py → paper orders
```

> Full pipeline: [`docs/option_c_architecture.md`](option_c_architecture.md).
> Windows-laptop variant: [`IBKR_OPTION_C_WINDOWS.md`](../IBKR_OPTION_C_WINDOWS.md).

---

## 1. Pick a host (free options)

| Host | Free? | Notes |
|------|-------|-------|
| **Oracle Cloud — Always Free (Arm Ampere A1)** | **Free forever** | Up to 4 cores / 24 GB; you need ~1–2 cores / 4–6 GB. **Recommended.** A1 capacity can be scarce — retry or pick a quieter region. |
| **Google Cloud — e2-micro Always Free** | Free forever | 1 GB RAM (us-west1/central1/east1). Tight for the Java gateway — add ~2 GB swap. |
| AWS / Azure free tier | 12 months only | `t3.micro` / `B1s`, then billed. Skip for "free forever". |
| **Hetzner CX22 (~€4/mo)** / $5 DO/Linode | Cheap, not free | Rock-solid fallback if free tiers frustrate you — worth it for anything touching a broker. |

Use **Ubuntu 22.04/24.04**. Only **outbound** access to IBKR is needed — no
inbound ports; keep the firewall closed and SSH key-only.

---

## 1b. Prerequisites — settle these BEFORE building the VM

Nothing here provisions anything on IBKR's side; the compose file only consumes
credentials you already hold. Full detail in
[`IBKR_PAPER_TRADING.md` → *Prerequisites*](../IBKR_PAPER_TRADING.md#prerequisites--do-these-before-any-setup-script).

| # | Prerequisite | Why it matters on a VM |
|---|---|---|
| 1 | An IBKR **paper account** and its **separate username/password** | Become `TWS_USERID` / `TWS_PASSWORD` in `deploy/ibkr-gateway/.env` |
| 2 | Account id starts with **`DU`** | Enforced by `PAPER_ACCT_PREFIX` in `scripts/ibkr_common.py`; also what the healthcheck asserts (§6c) |
| 3 | **2FA disabled** on that paper login | **The** thing that breaks a headless box — see below |
| 4 | **Market data shared** to the paper account | Without quotes, marketable limits degrade to unprotected market orders (§6) |
| 5 | The publisher's **`OVERALL_BOOK_SECRET`** | A mismatch aborts every run at signature verification |
| 6 | **Git write credentials** *(only for the Executed Book write-back)* | Deploy key or fine-grained token — see §6d |

> **The 2FA gotcha (read this).** IBKR's daily two-factor auth is what breaks
> unattended login. Use a **standalone paper-trading account with 2FA disabled**
> (register a separate paper login, not the paper user tied to a funded account).
> With 2FA off, IBC logs the gateway in on its own — the whole thing then runs
> untouched. On a headless VM there is no one to tap the prompt, so this is not
> optional in practice; the healthcheck in §6c exists precisely to catch a
> gateway that is up but silently *not logged in*.

---

## 2. Install prerequisites

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin git python3.12 python3.12-venv
sudo usermod -aG docker "$USER"   # log out/in so docker works without sudo
```

Clone the repo and build the executor venv (only the light broker deps — **no
model stack** on this host):

```bash
git clone https://github.com/yoursprerit/btc-range-model.git
cd btc-range-model
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-ibkr.txt
```

---

## 3. Start IB Gateway (headless, paper) with Docker

A ready compose file lives in `deploy/ibkr-gateway/` and uses the maintained
`gnzsnz/ib-gateway` image (IB Gateway + IBC auto-login + Xvfb bundled).

```bash
cd deploy/ibkr-gateway
cp .env.example .env
nano .env            # set TWS_USERID / TWS_PASSWORD (your paper login)
docker compose up -d
docker compose logs -f    # watch it log in; Ctrl-C when you see it's ready
```

- The container maps the **paper API port to host `4004`** (live → 4003), bound
  to `127.0.0.1` only — never internet-facing.
- It restarts nightly at 03:00 ET (away from the rebalance window) and logs back
  in automatically.

> To watch the GUI while troubleshooting, uncomment the `5900:5900` VNC line in
> the compose file and connect a VNC client over an SSH tunnel.

---

## 4. Set the shared secret

Same `OVERALL_BOOK_SECRET` value used by the publisher (GitHub) and the Streamlit
app, so the book's signature verifies before any order:

```bash
# add to the login shell so cron inherits it
echo 'export OVERALL_BOOK_SECRET="your-shared-secret"' >> ~/.profile
source ~/.profile
```

---

## 5. Test the executor by hand

Point it at the container's paper port (`4004`). Dry-run first — **no orders**:

```bash
git pull origin main    # get the latest published book
.venv/bin/python scripts/ibkr_execute_book.py \
    --file data/overall/target_book.json --port 4004
```

You should see the book, `Signature: signature OK`, a freshness line, your `DU…`
account, and the order plan. When it looks right, place the paper orders:

```bash
.venv/bin/python scripts/ibkr_execute_book.py \
    --file data/overall/target_book.json --port 4004 --execute
```

---

## 6. Automate with cron

`scripts/ibkr_execute_daily.sh` does `git pull` → executor `--execute` → logs.
Tell it the container's paper port via `IBKR_PORT=4004`:

```cron
CRON_TZ=America/Chicago
30 14 * * 1-5  IBKR_PORT=4004 /home/ubuntu/btc-range-model/scripts/ibkr_execute_daily.sh >> /home/ubuntu/btc-range-model/logs/ibkr_cron.log 2>&1
```

- Fires weekdays **2:30 PM US Central** (3:30 PM ET) — 30 minutes before the
  equity close and hours after the cloud publisher's morning commit, so the
  `git pull` gets the day's book. `CRON_TZ=America/Chicago` holds the slot
  across the CDT/CST switch.
- The executor's own guards make a stray, early or repeated run a safe no-op —
  nothing trades unless everything checks out: weekend/holiday, signature,
  paper-only, a verified positions read, the book's bar being *this* session's,
  a duplicate-run lock, exposure/turnover caps and cash-funded buys. The full
  table is in [`IBKR_PAPER_TRADING.md`](../IBKR_PAPER_TRADING.md#pre-flight-guards-what-the-executor-refuses-to-do).
- Env overrides (see the script header): `IBKR_PYTHON`, `IBKR_BRANCH`, `IBKR_BOOK`,
  `IBKR_BAND`, `IBKR_HOST`, `IBKR_PORT`, `IBKR_EXTRA`, `IBKR_NO_PULL`,
  `IBKR_ORDER_TYPE`, `IBKR_SLIPPAGE_CAP`.
- **Book source defaults to `main`** — where the publisher commits the daily
  book. Override with `IBKR_BRANCH` only if you are testing off a feature branch;
  pointing it at a stale branch means the freshness guard refuses to trade, which
  presents as a silent daily no-op.
- **Freshness:** the executor rejects a book generated more than
  `--max-age-hours` ago (**default 36**), *and* — since the 2026-08-18
  duplicate-execution incident — one whose **equity basis** is not the last
  completed session. The generation window alone could not catch a day-old book
  (a 7:15-AM-CT publish is still inside 36 h at the next day's 2:30-PM-CT slot),
  and re-trading one into an account that already holds it is how that incident
  started. **A withheld publish now means no trade**, which was always the
  intended fallback: a missing book is a missing decision, not a licence to
  re-run the last one. `--allow-stale-bar` overrides, for deliberate catch-up
  only.
  The basis is the book's signature-covered `signal_basis.equity_close`, not its
  `as_of`: the publisher runs 7 days a week for the 24/7 sleeves, so the
  Saturday, Sunday and Monday books all carry a weekend `as_of` over Friday's
  close. Gating on `as_of` refused those as "ahead of the last completed
  session", which cost a full session's trading every Monday and after every US
  market holiday (2026-08-24).
- **Repeat runs are no-ops.** Once a signal bar has an execution report in
  `data/overall/executed_archive/`, a second `--execute` for that bar aborts
  (exit 0). A scheduler that retries a failed run therefore cannot double-trade;
  `--refresh-report` re-states the account without placing orders.

### Order routing on this host

Orders default to **marketable limits**: a limit priced `IBKR_SLIPPAGE_CAP`
(0.5%) through the touch, with automatic market-order escalation for anything
unfilled inside the fill timeout. Full rationale in
[`IBKR_PAPER_TRADING.md` § Order routing](../IBKR_PAPER_TRADING.md#order-routing).
Two VM-specific consequences:

- **Delayed-only market data** (no subscription shared to the paper account)
  means legs that can't be priced log `WARN … falling back to MARKET` and go out
  unprotected. Either share the data (§1b) or widen the cap:
  `IBKR_SLIPPAGE_CAP=0.015`.
- **`IBKR_ORDER_TYPE=moc`** fills in the 4:00 PM ET closing auction — the price
  the backtest books against — but the executor then **holds the connection
  ~32 minutes** waiting for the print. Keep `TimeoutStartSec` in
  `ibkr-executor.service` comfortably above that; it ships at `3600`. MOC also
  requires entry before the **15:50 ET** cutoff (the 2:30-PM-CT slot clears it by
  20 min; past it the executor falls back to a marketable limit) and accepts
  **whole shares only** — `--fractional` / `IBKR_EXTRA=--fractional` is ignored
  for MOC, with a warning.

Check the log after the first scheduled run:

```bash
tail -n 60 ~/btc-range-model/logs/ibkr_cron.log
```

---

## 6b. Prefer systemd over cron (more robust)

systemd gives you dependency ordering (start after Docker), `journalctl` logs,
visible next-run times, and `OnFailure=` alerting. Units are in
`deploy/systemd/` — **edit the paths / `User=` in each file** (they assume
`ubuntu` and `/home/ubuntu/btc-range-model`), then:

```bash
# put the signing secret / port / webhook where systemd can read them
cp deploy/systemd/executor.env.example deploy/systemd/executor.env
nano deploy/systemd/executor.env         # OVERALL_BOOK_SECRET, IBKR_PORT=4004, IBKR_ALERT_WEBHOOK

sudo cp deploy/systemd/ibkr-executor.service \
        deploy/systemd/ibkr-executor.timer \
        deploy/systemd/ibkr-gateway-healthcheck.service \
        deploy/systemd/ibkr-gateway-healthcheck.timer \
        deploy/systemd/ibkr-alert@.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ibkr-executor.timer ibkr-gateway-healthcheck.timer

systemctl list-timers 'ibkr-*'                 # confirm next run times
journalctl -u ibkr-executor.service -f         # watch the next rebalance
```

- **`ibkr-executor.timer`** fires the executor weekdays 2:30 PM US Central
  (3:30 PM ET).
- **`ibkr-gateway-healthcheck.timer`** probes the gateway every 30 min, 08:00–16:30
  ET (skipping the nightly 03:00-ET restart window to avoid false alarms).
- Both units carry `OnFailure=ibkr-alert@%n.service`, so a failed run also fires
  an alert — on top of the healthcheck's own webhook post.

> The timezone suffix in the `.timer` files (`America/Chicago` on the executor,
> `America/New_York` on the healthcheck) needs **systemd v252+** (Ubuntu 24.04
> has it). On older systemd, set the host clock to the matching zone with
> `sudo timedatectl set-timezone …` and delete the suffix from the
> `OnCalendar=` lines — note the two units are anchored to different zones, so
> on an older box convert the executor's 14:30 Central into the host's zone
> before dropping the suffix.

## 6c. Gateway healthcheck & alerts

`scripts/ibkr_gateway_healthcheck.py` doesn't just ping the port — it opens an
IBKR API session and requires a **paper (`DU…`) managed account**, so it catches
the case where the gateway is running but **not logged in** (a failed login or a
surprise 2FA prompt).

Run it by hand:

```bash
.venv/bin/python scripts/ibkr_gateway_healthcheck.py --port 4004
# 🟢 healthy — 127.0.0.1:4004 logged in, paper account DU1234567
```

On failure it exits non-zero **and** POSTs to `IBKR_ALERT_WEBHOOK` if set. The
webhook payload includes both `text` and `content` fields, so a **Slack** or
**Discord** incoming-webhook URL works as-is:

```bash
# in deploy/systemd/executor.env
IBKR_ALERT_WEBHOOK=https://hooks.slack.com/services/XXX/YYY/ZZZ
```

Leave the webhook blank to log to `journalctl` only.

## 6d. Executed Book write-back (optional)

After each `--execute` run the executor writes
`data/overall/executed_book.json` (trades placed + resulting positions), and the
daily wrapper **commits and pushes it back** to the branch so the cloud app's
**✅ Executed Book** tab can show what actually happened.

This is the only step on the VM that **writes** to the repo, so the host needs
**git write credentials**:

- **SSH deploy key** (recommended): add a key to the repo with *write* access and
  clone via SSH (`git@github.com:…`), or
- **HTTPS + a fine-grained token**: set the remote to
  `https://<token>@github.com/yoursprerit/btc-range-model.git`.

If the host has read-only access, execution still works — the push just fails and
the wrapper **rolls back** so the branch never diverges (the Executed Book tab
simply won't update). Set `IBKR_NO_PUSH_REPORT=1` to disable the write-back
entirely.

> Signing: the report is signed with the same `OVERALL_BOOK_SECRET`, so the
> Executed Book tab shows "✅ signature verified" when the Streamlit secret matches.

---

## 7. Security checklist

- **Never expose the API port** (4004/4002) beyond `127.0.0.1`. The compose file
  already binds to localhost.
- SSH keys only; disable password login; keep the VM firewall closed to inbound.
- `.env` holds your paper password — it is gitignored; keep it `chmod 600`.
- It's a **paper** account, but treat the box as sensitive anyway.
- Keep Docker and the image updated: `docker compose pull && docker compose up -d`.

---

## 8. Costs & caveats

- **Oracle Always Free** is free indefinitely; a *running* gateway keeps the VM
  active so it isn't reclaimed as idle.
- **GCP e2-micro** works but is RAM-tight — add swap:
  `sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile` (persist in `/etc/fstab`).
- IBKR permits running the gateway on a VPS. If your paper login later starts
  demanding 2FA, switch to a standalone paper account with 2FA off, or follow the
  IBC second-factor docs.
