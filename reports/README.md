# reports/ — agent-authored analysis (never live config)

Markdown reports produced by the AI agents documented in
[`docs/AI_AGENTS.md`](../docs/AI_AGENTS.md). Everything here is **advisory
text** delivered via pull request — nothing in this directory is read by the
engines, the optimiser, the publisher or the executor.

- `health_triage/` — the 🤖 health-triage agent's investigation of 🩺 Strategy
  Health alarms, one report per alarm episode
  (`<as_of>-<fingerprint>.md`), each ending in a RETUNE / DE-RATE / RETIRE /
  WATCH recommendation for a human to act on.
- `evals/` — on-demand sleeve reviews from the `/eval-sleeve` command.
  (Full hypothesis studies keep landing as `*_EVAL.md` at the repo root,
  alongside the existing house evals.)
