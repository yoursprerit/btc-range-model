# Live trading — running real money alongside paper

> ⚠️ **Real money.** This runs the strategy against a **funded IBKR account**.
> The strategy was built and validated on paper; treat live as a monitored trial
> with small capital. Keep **paper running in parallel** as your control — don't
> replace it.

Live is **the same pipeline as paper** with a different gateway, a stricter
account guard, and hard exposure limits. The publisher emits **two signed books**
from one engine run — `target_book.json` (paper: idle → **cash**) and
`target_book_live.json` (live: idle → a **SATA** position, ~13% yield). Each
executor picks the book matching its `--account-mode`.

```
Cloud publisher → target_book.json (paper) + target_book_live.json (live)
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
| **Exposure cap** | `--max-deploy-frac 0.25` | Caps **risk assets** at **25% of net-liq**; the freed weight goes to **SATA** on the live book (idle → yield park), or cash on paper. |
| **Per-order cap** | `--max-order-notional 5000` | Clamps any single order to a dollar ceiling (fat-finger / bug backstop). |
| **Kill switch** | `--kill-switch-file …/STOP_LIVE` or `IBKR_TRADING_DISABLED=1` | `touch` the file to halt live on the next run — no cron/systemd edits. |
| **Dry-run default** | (no `--execute`) | Orders require `--execute`; the wrapper adds it deliberately. |

All the paper guards still apply too: signature verification, freshness, weekend/
holiday skip, and the no-trade band.

### The guards added after the 2026-08-18 incident

On 2026-08-18 the **paper** account was bought four times over: three runs each
read the account as flat (IB Gateway had not delivered the position
subscription), each re-bought the whole book on top of what was already held,
and the account ended at **3.4× net liquidation on a $2.17M margin loan**, −8.1%
NAV in a day. On a funded account that is a margin call, not a bad afternoon.
Everything below now applies to live and paper alike — read
[`IBKR_PAPER_TRADING.md`](../IBKR_PAPER_TRADING.md#pre-flight-guards-what-the-executor-refuses-to-do)
for the full table.

| Guard | Default | Why it matters for real money |
|-------|---------|-------------------------------|
| **Verified positions read** | always on | A read that misses value the account itself reports aborts the run. This is the guard that stops "the account looks flat, buy everything again". |
| **Current-bar check** | always on (`--allow-stale-bar` to override) | Only the last completed session's book may trade. A withheld publish means **no trade**. |
| **Duplicate-run lock** | always on (`--force-rerun`) | A signal bar with an execution report already archived is never executed twice. |
| **Cash-funded buys** | always on (`--allow-margin`) | Buys are capped at settled cash + the proceeds the sells actually realised, budgeted at live quotes. **A live account cannot take on an unintended margin loan.** |
| **Account state** | always on | Refuses to trade an account that already carries a margin loan or is already geared past the cap. |
| **Projected exposure** | `--max-gross-frac 1.02` | Aborts any plan landing above ~1× NAV. Note this sits *on top of* `--max-deploy-frac`: that one caps the book's risk weight, this one catches a plan that would gear the account regardless of what the book said. |
| **Turnover** | `--max-turnover-frac 1.5` | Aborts an implausible churn — the signature of a plan built against the wrong picture of the account. |
| **Price drift** | `--max-price-drift 0.25` | Aborts when a name quotes far from its sizing price (a split between publish and execution would size every order wrong). |
| **Session hours** | always on (`--outside-rth`) | No orders outside 09:30–16:00 ET. The incident's first round went out at 18:48 ET. |
| **Post-trade verification** | always on | Re-reads the account and reports realised leverage and the largest drift. It never auto-corrects — a corrective second round is what compounds a bad read. |

> **Before funding the account**, run a live-mode dry-run (`--account-mode live`
> without `--execute`) and read the `Pre-flight:` block: every line must be a ✓.
> The same block prints on every real run, in the log the wrapper keeps.

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
    --file data/overall/target_book_live.json \
    --port 4003 --account-mode live --confirm-live \
    --max-deploy-frac 0.25 --max-order-notional 5000
```

Note the `U…` account id it prints. Then re-run with `--expected-account U…` and
confirm the plan is **capped at 25%** and per-order clamps look right.

### 3. First live execution (by hand, market hours)

```bash
.venv/bin/python scripts/ibkr_execute_book.py \
    --file data/overall/target_book_live.json \
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

Now both fire each weekday at **2:30 PM US Central** (3:30 PM ET), the slot in
`ibkr-executor.timer` the live copy inherits: paper (`4004`, no caps) and live
(`4003`, 25% cap). They write separate reports and appear under the Executed
Book tab's **Paper / Live** selector.

Both also route orders identically — **marketable limits** by default, priced
0.5% through the touch, with any unfilled remainder escalated to a market order
in the same run. Keep that on for live: it is the only thing standing between a
bad quote and an unbounded fill price, and it costs nothing when the book is
normal. `IBKR_ORDER_TYPE=moc` in `executor-live.env` switches to the 4:00 PM ET
closing auction (see IBKR_PAPER_TRADING.md § Order routing) — if you do, keep
`TimeoutStartSec` above ~35 min so the run isn't killed mid-auction.

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
