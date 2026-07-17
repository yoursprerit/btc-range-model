# Live trading — running real money alongside paper

> ⚠️ **Real money.** This runs the strategy against a **funded IBKR account**.
> The strategy was built and validated on paper; treat live as a monitored trial
> with small capital. Keep **paper running in parallel** as your control — don't
> replace it.

Live is **the same pipeline as paper** with a different gateway, a stricter
account guard, and hard exposure limits. One signed target book drives both; you
just run the executor twice.

```
Cloud publisher → target_book.json  (one signed book)
   ├── paper executor  → gateway :4004 (DU…)  → paper account   (control)
   └── live  executor  → gateway :4003 (U…)   → LIVE account    (real money, capped)
```

---

## What makes live safe (the guards you get)

| Guard | Flag / setting | Effect |
|-------|----------------|--------|
| **Account mode** | `--account-mode live` | Refuses to run unless the account is **non-paper**. |
| **Explicit confirm** | `--confirm-live` | Live never runs by accident — the wrapper only adds this when `IBKR_ACCOUNT_MODE=live`. |
| **Pinned account** | `--expected-account U1234567` | Aborts if the connected account isn't **exactly** yours — can't trade the wrong account. |
| **Exposure cap** | `--max-deploy-frac 0.25` | Deploys at most **25% of net-liq**; the rest stays cash. |
| **Per-order cap** | `--max-order-notional 5000` | Clamps any single order to a dollar ceiling (fat-finger / bug backstop). |
| **Kill switch** | `--kill-switch-file …/STOP_LIVE` or `IBKR_TRADING_DISABLED=1` | `touch` the file to halt live on the next run — no cron/systemd edits. |
| **Dry-run default** | (no `--execute`) | Orders require `--execute`; the wrapper adds it deliberately. |

All the paper guards still apply too: signature verification, freshness, weekend/
holiday skip, and the no-trade band.

---

## The one hard part: 2FA on live login

Paper automation is easy with a **standalone paper account (2FA off)**. **Live is
different** — IBKR *mandates* two-factor auth on funded-account logins, so a live
IB Gateway can't log in fully unattended the way paper does. Options:

- **IBKR Mobile soft-token + IBC** — IBC can handle IBKR Mobile 2FA in some
  configurations; follow the IBC second-factor docs. Most robust for headless.
- **Attended login** — log the live gateway in yourself (VNC into the container)
  and leave it running; IBC keeps the session alive through the daily restart.
- Accept that you may need to re-auth the live gateway periodically.

The paper side keeps running regardless — a live-login lapse only pauses live.

---

## Setup (adds to the cloud runbook)

Assumes you've done [`docs/CLOUD_EXECUTOR.md`](CLOUD_EXECUTOR.md) for paper.

### 1. Start the live gateway (separate login)

```bash
cd deploy/ibkr-gateway
nano .env                     # add TWS_USERID_LIVE / TWS_PASSWORD_LIVE
docker compose --profile live up -d ib-gateway-live
docker compose logs -f ib-gateway-live      # complete 2FA if prompted
```

Live API is on host **`4003`** (paper stays on `4004`).

### 2. Find your live account id and dry-run

```bash
# dry-run against LIVE — places NO orders, just shows the plan + guard result:
.venv/bin/python scripts/ibkr_execute_book.py \
    --file data/overall/target_book.json \
    --port 4003 --account-mode live --confirm-live \
    --max-deploy-frac 0.25 --max-order-notional 5000
```

Note the `U…` account id it prints. Then re-run with `--expected-account U…` and
confirm the plan is **capped at 25%** and per-order clamps look right.

### 3. First live execution (by hand, market hours)

```bash
.venv/bin/python scripts/ibkr_execute_book.py \
    --file data/overall/target_book.json \
    --port 4003 --account-mode live --confirm-live \
    --expected-account U1234567 \
    --max-deploy-frac 0.25 --max-order-notional 5000 --execute
```

Check fills in IB Gateway and the **✅ Executed Book → Live** tab (the report
commits back as `executed_book_live.json`). Watch the **vs Target Book** drift.

### 4. Automate live in parallel with paper

The daily wrapper reads the live switches from the env, so you point a **second**
systemd unit at `executor-live.env`:

```bash
cp deploy/systemd/executor-live.env.example deploy/systemd/executor-live.env
nano deploy/systemd/executor-live.env      # secret, IBKR_PORT=4003, IBKR_ACCOUNT_MODE=live,
                                           # IBKR_EXPECTED_ACCOUNT, caps, kill-switch path

sudo cp /etc/systemd/system/ibkr-executor.service /etc/systemd/system/ibkr-executor-live.service
sudo cp /etc/systemd/system/ibkr-executor.timer   /etc/systemd/system/ibkr-executor-live.timer
# edit ibkr-executor-live.service: EnvironmentFile → …/deploy/systemd/executor-live.env
#                                  Description → "IBKR LIVE executor"
sudo systemctl daemon-reload
sudo systemctl enable --now ibkr-executor-live.timer

systemctl list-timers 'ibkr-*'             # paper + live both scheduled
```

Now both fire each weekday morning: paper (`4004`, no caps) and live (`4003`,
25% cap). They write separate reports and appear under the Executed Book tab's
**Paper / Live** selector.

---

## Halting live fast

```bash
touch /home/ubuntu/btc-range-model/STOP_LIVE     # next live run aborts before any order
# …later:
rm /home/ubuntu/btc-range-model/STOP_LIVE         # resume
```

Paper is unaffected. To stop the schedule entirely:
`sudo systemctl disable --now ibkr-executor-live.timer`.

---

## Recommended rollout

1. **Weeks of paper+live in parallel** at a low `--max-deploy-frac` (start 25%),
   whole shares, tight per-order cap.
2. Reconcile the **Live** Executed Book against the **Paper** one daily — same
   book, so the fills should track (differences are account size, caps, and
   liquidity).
3. Only raise the deploy cap once you're satisfied the live fills match intent.
4. Keep the kill switch one `touch` away.

---

## Safety recap

- **Paper stays the control** — never remove it.
- Live requires **mode + confirm + (recommended) pinned account** — three
  independent things must all be set for a real order to go out.
- **Exposure and per-order caps** bound the blast radius of any bug.
- The strategy is validated on paper; live is your decision and your risk. This
  repo gives you the guardrails, not a guarantee.
