# Collective2 publisher

`scripts/publish_c2.py` mirrors the daily target book onto a
[Collective2](https://collective2.com) strategy, so C2 keeps an independently
verified, public track record of exactly what the book holds.

It reads the same signed `data/overall/target_book.json` the IBKR executor
trades, sizes each weight into whole shares against the C2 strategy's
model-account value, and sends the list to C2's
`POST /v2/Strategies/SetDesiredPositions` (API v4,
`https://api4-general.collective2.com`). C2 places the market orders that move
its model account onto those positions.

## Setup (runs in GitHub Actions — no PC needed)

**Step-by-step runbook with testing:** [`COLLECTIVE2_SETUP.md`](COLLECTIVE2_SETUP.md).

`.github/workflows/publish-c2.yml` sends the day's **paper** book to C2 at
**9:00 AM America/Chicago** on weekdays, 30 minutes into the session and
after the 7:15 AM CT book publish.

1. Create a strategy on C2 (US stocks enabled) and note its **StrategyId**.
2. Create an API key in your C2 account settings.
3. GitHub → repo **Settings → Secrets and variables → Actions** → add
   `C2_API_KEY` and `C2_STRATEGY_ID` (`OVERALL_BOOK_SECRET` already exists
   for the book publish). Until they are set the workflow exits green with a
   notice.
4. Merge to `main` — `on: schedule` only fires from the default branch.
5. For an on-time 9:00 fire, add the cron-job.org job in
   [`EXTERNAL_SCHEDULER.md` § C2](EXTERNAL_SCHEDULER.md#collective2-publish-900-am-ct).
   The workflow's hourly cron slots (to the close) are the backup.

Each run writes its output to the job summary. The last book sent is recorded
in `data/overall/c2_publish_state.json` (committed by the workflow), so the
dispatch, backup slots and manual runs never send the same book twice. If
today's book is late or withheld, runs skip until it lands; outside regular
hours they skip too.

**Order of fills:** C2 trades at ~9:00 AM CT, before the IBKR executor's 2:30
PM CT slot, so C2 subscribers trade the book before your own account does.

## Manual use

```bash
# preview against a notional capital — no key needed, nothing sent
python scripts/publish_c2.py --capital 100000

# preview against C2's model-account value
C2_API_KEY=… C2_STRATEGY_ID=… python scripts/publish_c2.py

# send
OVERALL_BOOK_SECRET=… C2_API_KEY=… C2_STRATEGY_ID=… python scripts/publish_c2.py --execute
```

## Guards

| Guard | Why |
|---|---|
| Dry-run by default | Sending requires `--execute`. |
| Signature + `target_book.validate` | Same checks as the executor: a tampered, stale or audit-failed book is never published. |
| No accidental liquidation | SetDesiredPositions closes every position left out of the request. A book that sizes to zero shares is refused; an all-cash book needs `--allow-flat`. |
| Weights ≤ 100%, floored shares, 1% cash buffer | The model account never sizes onto margin from the book's publish-time prices. |
| Trading day + regular hours | C2 cannot adjust positions while the market is shut (`--outside-rth` overrides the hours). |
| Once per book | `data/overall/c2_publish_state.json` records the last book sent; a re-run skips it (`--force` re-sends), so intraday price moves cannot churn the model account. |
| Rejected signals fail the run | A C2 rejection (e.g. an unsupported symbol) exits non-zero and is not recorded as sent. |

## Notes

* The BTC and ETH sleeves trade through **IBIT** / **ETHA**, as on IBKR
  (`scripts/ibkr_symbols.py`).
* Use the **paper** book (the default). The live book parks idle cash in SATA,
  which is not part of the published strategy.
* Set the C2 strategy's starting capital high enough that the smallest weights
  still size to at least one share; dropped names are listed in the output.
