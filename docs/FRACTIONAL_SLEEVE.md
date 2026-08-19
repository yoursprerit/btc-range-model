# Trading a slice of the account — the sleeve

> Start the strategy on **part** of an account and let that part **compound on
> its own P&L**, while the rest of the money is never touched.

This is the mechanism for "I want to go live, but not with everything." It works
identically on paper and live — the paper account is where you rehearse it.

---

## 1. Pick your accounting boundary first

There are only two honest ways to run a strategy on part of your money.

| | How the accounting works | Use when |
|---|---|---|
| **A separate IBKR account** (or a sub-account) | The **broker** keeps the books. Fund it with the stake, point the executor at it with `--expected-account`, and everything downstream — NAV, cash, P&L, tax lots, statements — is already scoped to the sleeve. Zero extra code. | **Always, if you can open one.** This is the correct answer for real money. |
| **A sleeve ledger in one account** (this doc) | The executor keeps the books: a small signed JSON tracks the sleeve's cash, positions come from the broker, and orders are sized off the sleeve's NAV instead of the account's. | You are on paper, you are testing, or opening a second account is not worth it yet. |

The ledger carries **one assumption**: the sleeve is the *only* thing trading
the book's tickers in that account. The executor already closes anything held
but not in the book, so it behaves as the sole owner of those names. If you also
trade SOXX or GLDM by hand in the same account, the ledger will drift, the
reconcile guard will halt the run, and you should be on a separate account.

---

## 2. Why not just `--max-deploy-frac 0.10`?

Because it is a **constant fraction**, not a **stake**. It re-derives "10%" from
the whole account on every run, so the strategy never actually keeps its own
results:

| | `--max-deploy-frac 0.10` | Sleeve |
|---|---|---|
| Strategy earns 10% on the slice | Gain is re-spread over the whole account; next run's stake is 10% of a 1%-larger account. You keep **a tenth** of what you earned. | Stake goes from $10,000 → $11,000. You keep **all** of it. |
| Strategy loses 20% on the slice | Next run **tops the slice back up** from the untouched cash. | Stake is $8,000. The strategy trades its way back or does not. |
| You deposit $50k for unrelated reasons | The strategy silently gets 10% of it. | Nothing changes. |
| What the equity curve means | A fraction of the account's curve. | The strategy's **own** compounded curve — directly comparable to the backtest. |

Both still exist and compose: with a sleeve open, `--max-deploy-frac` becomes a
throttle **inside** the sleeve (`0.5` = keep half the sleeve in cash). They
multiply, so set one or reason about the product.

---

## 3. Open a sleeve

Everything is one flag on the existing executor. **Do it as a dry-run first** —
opening the ledger is the only part of a dry-run that writes.

```bash
# paper: hand the strategy 10% of the account, once
python scripts/ibkr_execute_book.py --file data/overall/target_book.json \
    --sleeve-init-frac 0.10 --fractional

# or an exact dollar stake
python scripts/ibkr_execute_book.py --file data/overall/target_book.json \
    --sleeve-init-usd 10000 --fractional
```

That writes `data/overall/sleeve.json` (`sleeve_live.json` in live mode) — a
signed ledger, same HMAC as the books:

```json
{ "schema": "sleeve-ledger/v1",
  "account": "DU1234567", "account_mode": "paper",
  "inception": { "usd": 10000.0, "account_net_liq": 100000.0,
                 "frac_of_account": 0.10 },
  "cash": 10000.0, "last_as_of": "2026-08-18",
  "entries": [ … ] }
```

Day one is 100% cash — the first rebalance is what deploys it.

**From then on, change nothing.** The scheduled wrapper finds the ledger beside
the execution report and keeps compounding it. `--sleeve-init-*` on an existing
ledger is refused, so a stray flag can never silently reset your history.

---

## 4. What the executor does differently

```
sleeve NAV = sleeve cash + Σ shares × price          (book prices, broker marks as fallback)
```

| | Sized against |
|---|---|
| Target dollar per name (`build_order_plan`) | **sleeve NAV** |
| No-trade band (`--band`) | **sleeve NAV** — a 1% band on a $100k account is a 10% band on a $10k sleeve, which would suppress every rebalance |
| Exposure cap (`--max-gross-frac`) | **sleeve NAV** — on net liq, a 10% sleeve could gear 10:1 before the 1.02× cap noticed |
| Turnover cap (`--max-turnover-frac`) | **sleeve NAV** |
| Margin-loan check, cash funding | **the account** — cash and a margin loan are real, account-wide facts |

Cash moves on **fills**, never on intentions: partial fills move the part that
filled, slippage lands in the ledger because the fill price is the fill price,
and a dry-run's `PLANNED` rows move nothing. A signal bar can only be booked
once, so `--force-rerun` cannot debit the sleeve twice.

---

## 5. Managing it

```bash
# where does the sleeve stand? (a preview places no orders)
python scripts/ibkr_execute_book.py --file data/overall/target_book.json

#   Sleeve: sleeve $11,240 of account $101,240 (11.1%)
#     stake $10,000.00 → NAV $11,240.31 (+1,240.31, +12.40% compounded)

# CAPITAL — add to the stake / take money out of it
python scripts/ibkr_execute_book.py … --sleeve-adjust  2500   # deposit
python scripts/ibkr_execute_book.py … --sleeve-adjust -1000   # withdraw

# P&L — a dividend, an interest credit, a fee the fills cannot see
python scripts/ibkr_execute_book.py … --sleeve-income  42.17
python scripts/ibkr_execute_book.py … --sleeve-income -30.00

# trade the whole account again for one run
python scripts/ibkr_execute_book.py … --no-sleeve

# start over at a new stake, discarding the compounded history
python scripts/ibkr_execute_book.py … --sleeve-reset --sleeve-init-frac 0.25
```

These two are the only money movements not driven by a fill, and **the
distinction is the whole point of the ledger**:

* `--sleeve-adjust` is **capital** — it moves the base `return_pct` is measured
  against, so topping the sleeve up never flatters its record.
* `--sleeve-income` is **P&L** — the sleeve earned it by holding what it holds,
  so it lifts the return exactly the way a price move does.

Book a GLDM or XLE dividend as capital and the cash arrives, the base rises with
it, and the strategy's measured return goes *down* on a day it made money. Every
use of either is stamped in `entries`. Keep both as one-off commands; in the
wrapper's env they would re-run every day.

A withdrawal is floored at (roughly) zero sleeve cash — the money in a
fully-invested sleeve is in its positions, and booking cash it does not have
would size the next book against a fiction. Cut exposure first.

The **Executed Book** page shows the sleeve block after any sleeve run — NAV,
capital in, compounded P&L — and measures the allocation-drift table against
sleeve NAV, so target vs actual still reads in the book's own weights.

---

## 6. When a run refuses to trade

The sleeve is a claim on part of a real account, and the executor checks it
still is one before every order.

| Message | What happened | Fix |
|---|---|---|
| `sleeve NAV … exceeds the account's …` | Ledger drifted above the account — a withdrawal, or fills that were never booked. | Reconcile with `--sleeve-adjust`, or `--sleeve-reset` at the honest number. |
| `sleeve claims $X cash but the account holds $Y` | The buy leg would borrow. | Fund the account, `--sleeve-adjust` down, or `--allow-margin` deliberately. |
| `sleeve cash is $-X — carrying a loan` | The sleeve's cash went materially negative — an unbooked fee run, or a withdrawal that should not have cleared. | `--sleeve-income` the missing debits, or reduce exposure. A few dollars of fees is tolerated and never trips this. |
| `ledger belongs to account … / is a paper sleeve` | Paper ledger pointed at the live account or vice versa. | Use `--sleeve-file`, or the mode's own default ledger. |
| `sleeve holds … with no usable price` | A held name has neither a book price nor a broker mark. Valuing it at zero would understate NAV and over-buy everything else. | Restore market data, or close the name. |

---

## 7. Practical notes for a small sleeve

* **Use `--fractional`.** Whole-share rounding is a rounding error against a
  $100k account and an allocation error against a $10k one — one $600 share is
  6% of it. The executor prints a NOTE when the priciest name is over 1% of the
  sleeve per share. (IBKR must have fractional trading enabled for the account.)
* **Commissions bite harder.** A fixed per-order commission is a fixed cost
  against a much smaller base; a daily rebalance on a tiny sleeve can spend a
  meaningful share of its edge. Widen `--band` if the plan churns.
* **Keep the ledger in git.** It commits and pushes with the execution report,
  so the cloud app and the next run see the same history. Losing it loses the
  compounded record, not the money.
* **Ramp with the stake, not the strategy.** Growing conviction means
  `--sleeve-adjust +N`, which is stamped and separable, rather than editing
  weights or caps.
