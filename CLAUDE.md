# Working notes for Claude Code in this repository

## Strategy change log — log every change (standing instruction)

Every change to the strategy or the platform around it is logged in
`data/changelog/strategy_changelog.json` (shown on the **📜 Strategy Change Log**
page, `app/changelog_app.py`). **In the same commit/PR as the change**, add one
entry under the version it belongs to:

```json
{"version": "v2", "date": "YYYY-MM-DD", "kind": "allocator",
 "title": "Short name of the change",
 "description": "One or two sentences: what changed.",
 "impact": "Optional — why it helps / the evidence (numbers).",
 "profile": "Balanced",
 "links": [{"label": "PR #NNN", "url": "https://github.com/yoursprerit/btc-range-model/pull/NNN"}]}
```

* `date` is the day the change reaches `main` (the PR's merge date, UTC). Different
  changes inside one version have different dates.
* `kind` is one of `sleeve`, `allocator`, `execution`, `data`, `analytics`, `ui`.
* `version` is the strategy-logic generation in force — currently **v2**
  (`app/strategy_version.py`). Do not bump the version unless asked; a new
  generation also needs a row in `VERSION_HISTORY` and a new entry under
  `versions` in the log (`tests/test_strategy_changelog.py` keeps the two in step).
* `profile` is only for rules that apply to one risk profile.
* The PR number is only known after the PR is opened — add the link in a follow-up
  commit on the same branch before merging.
* Keep the log out of `app/*.py`: the P&L snapshot's code fingerprint hashes the
  engine modules, so a data file edit must not invalidate it.
