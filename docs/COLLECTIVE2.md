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

## Setup

1. Create a strategy on C2 (US stocks enabled) and note its **StrategyId**.
2. Create an API key in your C2 account settings.
3. On the executor host, add to `deploy/systemd/executor.env` (or the
   environment of whatever runs `ibkr_execute_daily.sh` / `.ps1`):

   ```
   C2_PUBLISH=1
   C2_API_KEY=…
   C2_STRATEGY_ID=…
   ```

The daily executor script then runs the publisher **after** its own IBKR
orders (2:30 PM CT slot), so the public strategy never competes with the
account for fills. A C2 failure is logged and never fails the rebalance.

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
| Once per book | `logs/c2_publish_state.json` records the last book sent; a re-run skips it (`--force` re-sends), so intraday price moves cannot churn the model account. |
| Rejected signals fail the run | A C2 rejection (e.g. an unsupported symbol) exits non-zero and is not recorded as sent. |

## Notes

* The BTC and ETH sleeves trade through **IBIT** / **ETHA**, as on IBKR
  (`scripts/ibkr_symbols.py`).
* Use the **paper** book (the default). The live book parks idle cash in SATA,
  which is not part of the published strategy.
* Set the C2 strategy's starting capital high enough that the smallest weights
  still size to at least one share; dropped names are listed in the output.
* With `IBKR_ORDER_TYPE=moc`, or on a missed-slot catch-up, the executor
  finishes after the 4:00 PM ET close, so the regular-hours guard skips that
  day's C2 publish; C2 picks up again with the next day's book. Keep the
  default marketable-limit routing to publish every session.
