# 🤖 AI agents — setup & operations runbook

Three Claude-powered agents attach to the **edges** of the platform. None of
them sits in the signal or allocation path — the daily book stays fully
deterministic, look-ahead-audited and HMAC-signed exactly as before. The agents
compress the *human* work around it: triaging health alarms, diagnosing
pipeline failures, and running evaluation studies.

| # | Agent | Where it runs | Trigger | Deliverable |
|---|---|---|---|---|
| 1 | **Health triage** | GitHub Actions (`.github/workflows/health-triage.yml`) | After each completed *Publish target book* run, gated on an actionable 🩺 alarm | A triage report + recommendation **PR** |
| 2 | **Ops babysitter** | GitHub Actions (`.github/workflows/ops-babysitter.yml`) | Any **failed** run of the publish / data-refresh workflows | Job-summary diagnosis; a fix **PR** when warranted |
| 3 | **Research fan-out** | Claude Code sessions (CLI or claude.ai/code), on demand | You — `/eval-sleeve`, `/eval-hypothesis` | Eval scripts + house-style `*_EVAL.md` verdicts |

```
publish workflow ──success──► health-triage:  gate (seconds) ──alarm──► agent ──► PR
      │
      └────failure──► ops-babysitter: collect logs ──► agent ──► summary or fix PR

you ──/eval-sleeve, /eval-hypothesis──► Claude Code session ──► eval doc on a branch
```

**Design rule (all three):** agents **propose, humans dispose**. Every write
path ends at a pull request you review; nothing an agent does can change what
gets traded without a merge. The health monitor's own principle
(`STRATEGY_HEALTH.md`: an alarm triggers a *human* decision) is preserved —
the agent only does the investigation.

---

## 1. One-time setup (~5 minutes)

1. **Anthropic API key** — create one at <https://console.anthropic.com>, then
   add it as a repository secret:
   *GitHub → Settings → Secrets and variables → Actions → New repository
   secret*, name **`ANTHROPIC_API_KEY`**.
2. **Install the Claude GitHub App** on this repository:
   <https://github.com/apps/claude> → Install → select `btc-range-model`.
   The workflows' `claude-code-action` step exchanges its OIDC token for the
   app's credentials (that is what `id-token: write` is for) so the agent can
   push branches and open PRs.
   *Alternative without the app:* pass `github_token: ${{ github.token }}` to
   the `anthropics/claude-code-action@v1` step in both workflows.
3. **Merge these workflows to `main`.** `workflow_run` triggers only fire for
   workflow files on the default branch — the agents are dormant until then.
4. Nothing else. Both workflows are already permission-scoped
   (`contents/pull-requests: write`, `actions: read`) and — deliberately —
   **never receive `OVERALL_BOOK_SECRET`** or any IBKR credential, so a
   misbehaving agent cannot sign a book or reach the broker.

> **Issues are disabled on this repo** (the publish workflow's failure
> reporter already falls back to job summaries because of the HTTP 410).
> That's why both agents deliver via PRs and job summaries, and why the
> babysitter watches `workflow_run` failures instead of tracking issues.

### Cost expectations

The expensive part (a Claude session) runs **only when something is wrong**:
the triage gate is a stdlib script that exits in seconds on green days, and
the babysitter only launches on a failed run. A quiet week costs ~$0. An
actual triage or diagnosis session typically lands in the low single-digit
dollars of API usage; both are capped by `--max-turns` and job timeouts.

---

## 2. Agent 1 — Health triage

**What it automates.** `STRATEGY_HEALTH.md` deliberately ends every alarm at a
human protocol: *"re-run the sleeve's honest-fill re-sweep and decide — retune,
de-rate, or retire."* This agent runs that first pass the same day the alarm
fires, instead of whenever you next find time.

**How it runs.**

1. Every completed *Publish target book* run (which also rebuilds
   `data/overall/strategy_health.json`) triggers the workflow.
2. **Gate** — `scripts/health_alarm_gate.py` (stdlib-only, seconds) parses the
   snapshot. Actionable = any 🔴 monitor, or a 🟡 with a `first_breach_date`
   ≥14 days old — the doc's "act on red, or a yellow that persists" rule. No
   alarm → the run ends here, costing nothing.
3. **Dedupe** — the gate fingerprints the alarm *episode* (keyed on each
   breach's `first_breach_date`, so the id is stable while the same breach
   persists). If any PR already carries `health-triage-<fingerprint>`, the run
   skips — one episode → one PR, not one per day.
4. **Agent** — a runner with the full engine stack installed investigates each
   alarm using the repo's own tools (`health_history.csv`, the sleeve's
   `*_EVAL.md` history, `scripts/eval_*.py` re-sweeps), writes
   `reports/health_triage/<as_of>-<fingerprint>.md`, and opens a PR whose body
   holds a **RETUNE / DE-RATE / RETIRE / WATCH** recommendation per sleeve.

**You then:** review the PR like any eval doc. Merging it archives the report;
acting on a recommendation (changing a stop, cap, or the universe) is a
separate, deliberate change — the agent never makes it for you.

**Run it manually:** *Actions → Health triage agent → Run workflow*. Inputs:
`persist_days` (default 14) tightens/loosens the persistent-yellow rule;
`skip_dedupe` forces a fresh triage of an already-triaged episode.

**Test the gate locally** (no API key needed):

```bash
python scripts/health_alarm_gate.py                # human summary + exit 0
python scripts/health_alarm_gate.py --out /tmp/alarms.json --persist-days 21
```

**Guardrails:** the agent may write only under `reports/health_triage/`; it
cannot publish books, and its PR is the end of its authority.

---

## 3. Agent 2 — Ops babysitter

**What it automates.** A missed or stale publish is direct P&L risk (the
executor keeps trading yesterday's verified book until its freshness guard
stops it). Today a failed run means reading Actions logs by hand; this agent
does the reading — and knows the pipeline's designed failure modes.

**How it runs.**

1. Any run of **Publish target book (IBKR Option C)** or **Refresh backtest
   dataset** that completes with `conclusion: failure` triggers the workflow.
2. A pre-step collects the evidence with `gh` (run metadata, failing-step
   logs, the last 20 runs of that workflow) so the agent starts informed.
3. The agent classifies the failure:
   - **WITHHELD** — the freshness audit failed and the book was deliberately
     not published (`publish_target_book.py` exit 2). Designed safe path.
   - **TRANSIENT** — feed lag / rate limit / network flake a catch-up slot
     will clear.
   - **REAL** — an actual engine, dependency or workflow bug.
4. First-occurrence WITHHELD/TRANSIENT gets a job-summary diagnosis and
   nothing else — no noise for failures the pipeline is designed to absorb.
   It **escalates** (≥3 failures the same day, or the same cause on ≥2
   consecutive days) or a REAL failure gets a reproduction attempt (the runner
   has the full stack: `pull_backtest_data.py`, `validate_refreshed_data.py`)
   and, if a fix is found, an `ops-fix/<date>-<slug>` branch and a PR marked
   `ops-babysitter-fix` (open-PR dedupe prevents duplicates).

**You then:** on quiet failures, read nothing — the summary is there if you
want it. On a fix PR, review and merge; the babysitter's scope is pipeline
code only (`scripts/` fetch/validate/publish, workflows, `market_fallback.py`,
`freshness.py`, `data_gate.py`), never strategy behaviour or `data/`
artifacts.

**Run it manually:** *Actions → Ops babysitter agent → Run workflow* — leave
`run_id` blank to diagnose the newest failed run, or paste a specific run id
(the number in the failed run's URL).

---

## 4. Agent 3 — Research fan-out

**What it automates.** The platform's real edge-generation loop is the
`scripts/eval_*.py` → `*_EVAL.md` studies. The two slash commands package the
house methodology (honest fills, prefix-invariance, standard OOS windows,
portfolio-impact measurement) so each study starts from the full ruleset
instead of from scratch — and so several can run in parallel.

**Where it runs:** any Claude Code session with this repo — the CLI on your
machine (`claude` in the repo root) or a cloud session at
<https://claude.ai/code>. The commands live in `.claude/commands/` and are
picked up automatically; the GitHub-Actions agents can also invoke the same
protocol.

**Usage.**

```
/eval-sleeve MSTU M2 tripwire red since 2026-08-03
/eval-sleeve GDX
/eval-hypothesis add a -10% trailing stop to the XLE crash-shield
/eval-hypothesis candidate sleeve: URA traded off the REMX signal
```

- **`/eval-sleeve <KEY> [context]`** — runs the Strategy Health review
  protocol for one sleeve on demand (same protocol the triage agent runs
  automatically): fresh honest-fill re-sweep, decay-vs-bad-luck judgement,
  report in `reports/evals/`, RETUNE/DE-RATE/RETIRE/WATCH verdict.
- **`/eval-hypothesis <text>`** — evaluates a new idea end-to-end through the
  harness: a new `scripts/eval_<slug>.py`, the standard OOS windows and
  baselines, portfolio-level impact, and a root-level `*_EVAL.md` ending in an
  explicit ADOPT / REJECT / NEEDS-MORE-DATA verdict. Rejections are expected
  and are a successful outcome.

**The fan-out pattern:** hypotheses are independent, so run them in parallel —
one cloud session (or terminal tab) per hypothesis, each on its own branch:

```bash
git checkout -b eval/xle-trailing-stop   # session 1: /eval-hypothesis ...
git checkout -b eval/ura-sleeve          # session 2: /eval-hypothesis ...
```

Each session ends with a branch you can PR and compare. Merging an eval doc
records the study; **adopting** a change into a live config is always a
separate, human-reviewed edit.

---

## 5. Security & guardrail summary

- **No agent touches strategy behaviour.** Triage writes only reports;
  the babysitter touches only pipeline code; research commands write eval
  scripts + docs. Live configs, engines, weights, caps and books are out of
  scope for all three — by prompt contract, and reviewable because every
  change arrives as a PR.
- **No agent can publish or trade.** `OVERALL_BOOK_SECRET` and IBKR
  credentials are not exposed to any agent job; the signed-publish and
  executor paths are untouched.
- **Everything is auditable in git**: gate decisions in the Actions log,
  reports and fixes as PRs, recommendations as committed markdown.
- Optional hardening: protect `main` so even a confused agent (or a
  compromised dependency in its runner) cannot push to it — both agents are
  instructed to work on `health-triage/*` and `ops-fix/*` branches only.
- The agents' Bash access in CI is allowlisted (`python/pip/git/gh/jq` and
  read-only file utilities) via `--allowedTools` in each workflow; widen or
  narrow it there.

## 6. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Triage/babysitter job fails at the Claude step immediately | `ANTHROPIC_API_KEY` secret missing or invalid |
| Agent ran but could not push / open a PR | Claude GitHub App not installed on the repo (§1.2) — or pass `github_token` explicitly |
| Workflows never trigger | They only fire from `main` — merge them first; `workflow_run` needs the *monitored* workflow names to match exactly |
| Triage skips even though the dashboard shows red | The episode already has a PR (fingerprint dedupe) — re-run manually with `skip_dedupe: true`, or a new breach date will re-arm it automatically |
| Triage never fires on a yellow you care about | Yellows arm only after `persist_days` (default 14) — dispatch manually with a smaller value |
| Babysitter is silent on a failure | It ran and classified it WITHHELD/TRANSIENT — the diagnosis is in the babysitter run's job summary, by design |
