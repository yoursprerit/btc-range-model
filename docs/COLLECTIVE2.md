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
**2:30 PM America/Chicago** on weekdays, 30 minutes before the close — the
same slot as the IBKR executor, so C2 fills near the close the backtest books
against.

1. Create a strategy on C2 (US stocks enabled) and note its **StrategyId**.
2. Create an API key in your C2 account settings.
3. GitHub → repo **Settings → Secrets and variables → Actions** → add
   `C2_API_KEY` and `C2_STRATEGY_ID` (`OVERALL_BOOK_SECRET` already exists
   for the book publish). Until they are set the workflow exits green with a
   notice.
4. Merge to `main` — `on: schedule` only fires from the default branch.
5. For an on-time 2:30 PM fire, add the cron-job.org job in
   [`EXTERNAL_SCHEDULER.md` § C2](EXTERNAL_SCHEDULER.md#collective2-publish-230-pm-ct).
   The workflow's 15-minute cron slots are a best-effort backup.

Each run writes its output to the job summary. The last book sent is recorded
in `data/overall/c2_publish_state.json` (committed by the workflow), so the
dispatch, backup slots and manual runs never send the same book twice. If
today's book is late or withheld, runs skip until it lands; outside regular
hours they skip too.

**Positions snapshot:** every run also reads what the C2 model account
actually holds (`GetStrategyOpenPositions` — shares and C2's average fill,
`AvgPx`) and saves it to `data/overall/c2_positions.json`. After sending a
book, the run waits a minute so the fills can land first. The file is rewritten
only when the holdings or the book change, or once a day, so the 15-minute
slots don't each add a commit. The 🧭 Overall app's **💼 Current Positions**
section reads this file by default, marked at the live spot. It keeps up to
date without any local executor. The IBKR report is still available there as
a second choice.

**Timing:** C2 trades at ~2:30 PM CT, alongside the IBKR executor, so the C2
record, the IBKR account and the close-based backtest stay aligned. With only
30 minutes to the close there is little room for delay: if the cron-job.org
fire fails and no GitHub backup slot lands before 3:00 PM CT, that day is not
sent and C2 keeps the previous positions until the next day's book.

## Manual use

```bash
# preview against a notional capital — no key needed, nothing sent
python scripts/publish_c2.py --capital 100000

# preview against C2's model-account value
C2_API_KEY=… C2_STRATEGY_ID=… python scripts/publish_c2.py

# send
OVERALL_BOOK_SECRET=… C2_API_KEY=… C2_STRATEGY_ID=… python scripts/publish_c2.py --execute

# read-only: save C2's open positions for the Overall app's Current Positions
C2_API_KEY=… C2_STRATEGY_ID=… python scripts/publish_c2.py --snapshot
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
| No-trade band (`--band`, default 1% of capital) | Same as the executor's `IBKR_BAND`: a name C2 already holds keeps its quantity when the resize would move less than 1% of capital, so an unchanged book places no orders. New positions and exits always go through; the band is dropped for a run if holding would exceed 100% of capital. Over the last 15 archived books it cut C2 orders from 63 to 29. |
| Rejected signals fail the run | A C2 rejection (e.g. an unsupported symbol) exits non-zero and is not recorded as sent. |

## Notes

* The BTC and ETH sleeves trade through **IBIT** / **ETHA**, as on IBKR
  (`scripts/ibkr_symbols.py`).
* Use the **paper** book (the default). The live book parks idle cash in SATA,
  which is not part of the published strategy.
* Set the C2 strategy's starting capital high enough that the smallest weights
  still size to at least one share; dropped names are listed in the output.
