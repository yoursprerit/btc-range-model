---
description: Evaluate a new strategy/universe/parameter hypothesis through the repo's honest backtest harness and produce a house-style EVAL doc with a verdict
argument-hint: <hypothesis, e.g. "add a -10% trailing stop to XLE" or "candidate sleeve: URA on the REMX signal">
---

Evaluate this hypothesis through the repo's standard evaluation harness:
**$ARGUMENTS**

You are producing the same class of artifact as the repo's `*_EVAL.md` studies
(`ETH_BMNR_STRATEGY_EVAL.md`, `SOXL_ERX_ADDITION_EVAL.md`,
`VEGN_REMOVAL_EVAL.md` are the style references): a falsifiable question, the
honest method, the numbers, and a plain verdict — **including "no edge /
reject" when that is what the data says**. Most hypotheses fail; a clean
rejection is a successful outcome of this command.

## Method — the house rules (all mandatory)

1. **Reuse the harness.** Model your evaluation on the closest existing
   `scripts/eval_*.py` (read 2–3 first). New code goes in a new
   `scripts/eval_<slug>.py` — never modify engines, configs or `overall_core`
   to test a hypothesis. Use the same engines the apps trade
   (`backtest_ticker`, `backtest_gldm`, the BTC CT engine) wherever possible.

2. **Honest fills.** Signals are computed on completed bars only; equity fills
   land at the first exchange close AFTER the signal moment (the 2026-07-25
   MSTR/MSTU look-ahead fix is the cautionary tale — read
   `ETH_BMNR_STRATEGY_EVAL.md` §4 before writing any fill logic).

3. **No look-ahead.** The published standard is prefix-invariance
   (`scripts/check_lookahead.py`): recomputing on truncated history must not
   change the past. Any parameter you tune on the full window must be labelled
   in-sample in the doc.

4. **Standard OOS windows.** Report 🌐 Full OOS 2021→now, 🐻 Bear 2021–22,
   🐂 Bull 2023→now, 🔬 Recent 2025→now — total return, MDD, Sharpe, and trade
   count — vs the relevant baselines: buy-&-hold of the underlying AND the
   current config it would replace or join.

5. **Portfolio impact, not just sleeve merit.** If the hypothesis adds or
   changes a sleeve, measure the Overall effect: correlation to the existing
   sleeves and the Balanced/Growth replay deltas (the ETH lesson: a sleeve can
   be individually positive and still cost the portfolio Sharpe).

6. **Name the costs.** The house backtests charge no transaction costs — say
   what the hypothesis' trade frequency implies about that gap.

## Deliverables

- `scripts/eval_<slug>.py` — reproducible, runnable as
  `python scripts/eval_<slug>.py`;
- `<SLUG>_EVAL.md` at the repo root, house style, ending with an explicit
  **ADOPT / REJECT / NEEDS-MORE-DATA** verdict and what would change it;
- a short inline summary of the verdict with the headline numbers.

Commit on the current working branch. Do not wire an ADOPT verdict into any
live config — adoption is a separate, human-reviewed change.
