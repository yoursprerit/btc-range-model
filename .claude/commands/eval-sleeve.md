---
description: Re-run one sleeve's honest-fill evaluation and draft a retune / de-rate / retire / watch recommendation (the STRATEGY_HEALTH.md review protocol, on demand)
argument-hint: <SLEEVE_KEY e.g. MSTU, GDX, SOXL> [optional context, e.g. "M2 red since 2026-08-03"]
---

Run the Strategy Health review protocol for one sleeve: **$ARGUMENTS**

You are executing the human review protocol from `STRATEGY_HEALTH.md` for the
sleeve named above (the optional text after the sleeve key is context about why
this review was requested). The deliverable is an evidence-backed
recommendation — **you do not change any live config unless the user asks in a
follow-up**.

## Protocol

1. **Orient.** Read `STRATEGY_HEALTH.md` (monitors + thresholds + protocol) and
   `OVERALL_STRATEGY.md` §2 (which engine, parent signal, stop and instrument
   kind this sleeve trades). Find the sleeve's config origin story in the
   `*_EVAL.md` docs at the repo root (e.g. `LEV_SIBLINGS_STOP_EVAL.md`,
   `HYPERPARAM_SEARCH_EVAL.md`, `SOXL_STOP_EVAL.md` — grep for the ticker).

2. **Current health.** Read the sleeve's entry in
   `data/overall/strategy_health.json` and its rows in
   `data/overall/health_history.csv`. Characterise the trend: fresh shock vs
   slow bleed; which monitor(s) (M2 drawdown / M3 edge / M4 expectancy) are
   degraded and by how much vs their thresholds.

3. **Re-run the honest-fill evaluation** with the repo's existing harness —
   never invent a new methodology:
   - `scripts/eval_hyperparam_search.py` — the per-sleeve config sweep;
   - `scripts/eval_lev_siblings_stop.py` — the stop re-sweep, if the sleeve is
     a `beta`/`lev` sibling;
   - `scripts/eval_oos_walkforward.py` — the walk-forward OOS view.
   Honest-fill conventions are non-negotiable: equity fills land at the first
   exchange close AFTER the signal moment; no look-ahead (the repo's standard
   is prefix-invariance, `scripts/check_lookahead.py`). If a script doesn't
   parameterise by sleeve, run it as-is and extract this sleeve's rows.

4. **Judge** — is the current config still on (or near) the frontier of the
   fresh sweep, or has the frontier moved? Distinguish:
   - **DECAY** — the edge is gone or the regime that made it work ended;
   - **BAD LUCK** — the drawdown/expectancy is within the sleeve's own
     historical variance (M4's bootstrap frames this);
   - **ARTIFACT** — threshold/warm-up/data quirk, not a real signal.

5. **Report.** Write `reports/evals/<YYYY-MM-DD>-<SLEEVE>.md` in the house
   eval-doc style (question → method → numbers → verdict; tables for the sweep
   results; explicit caveats). End with ONE recommendation —
   **RETUNE** (name the exact new config and its sweep numbers) /
   **DE-RATE** (say what cap or weight change) / **RETIRE** /
   **WATCH** — plus a confidence level and what evidence would change it.

6. If working in a session on a branch: commit the report and (only if the
   user asked for a PR) open one. Otherwise just present the verdict inline
   with the report path.

## Guardrails

- Touch nothing outside `reports/evals/` — configs, engines, weights, caps and
  books stay as they are; the recommendation is text.
- Report honest numbers even when they argue for retiring a sleeve the repo
  seems attached to — the ETH sleeve's own doc is the house precedent for
  "added it, measured it, and said plainly it is weak".
