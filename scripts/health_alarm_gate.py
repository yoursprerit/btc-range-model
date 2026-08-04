#!/usr/bin/env python3
"""Health-alarm gate — decides whether the 🤖 health-triage agent should run.

Reads ``data/overall/strategy_health.json`` (the 🩺 Strategy Health snapshot,
schema ``strategy-health/v1``) and reports which monitors demand a human-grade
review, per the STRATEGY_HEALTH.md protocol: **act on 🔴, or on a 🟡 that
persists**.

Triage is needed when any of:
  * a sleeve monitor (M2 drawdown / M3 edge / M4 expectancy) is ``red``;
  * a profile drawdown tripwire or the M1 book-vs-replay tracker is ``red``;
  * any of the above is ``yellow`` with a ``first_breach_date`` at least
    ``--persist-days`` (default 14) before the snapshot's ``as_of``.

The gate is deliberately **stdlib-only** (no pandas/model stack) so the CI job
that runs it costs seconds, and it never uses the wall clock — persistence is
judged against the snapshot's own ``as_of``, so re-running the gate on old data
gives the old answer.

Outputs
  * a human summary on stdout;
  * ``--out FILE`` — the alarm list as pretty JSON (fed to the triage agent);
  * when ``$GITHUB_OUTPUT`` is set (or ``--github-output``): ``triage_needed``,
    ``n_alarms``, ``fingerprint`` and a one-line ``summary``.

The **fingerprint** is a stable id for the current alarm *set* (each alarm keyed
by scope/key/monitor/first_breach_date). While the same breaches persist the
fingerprint does not change day-to-day, so the triage workflow can dedupe: one
alarm episode → one triage PR, not one per day.

Exit codes: 0 = ran fine (whether or not triage is needed) · 2 = snapshot
missing/unreadable. The gate signals via its outputs, not its exit code.

Usage:
  python scripts/health_alarm_gate.py
  python scripts/health_alarm_gate.py --persist-days 21 --out /tmp/alarms.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import date
from pathlib import Path

DEFAULT_SNAPSHOT = Path("data/overall/strategy_health.json")
DEFAULT_PERSIST_DAYS = 14

# Sleeve monitor keys in the snapshot → the doc's monitor names.
SLEEVE_MONITORS = {
    "dd": "M2 drawdown tripwire",
    "edge": "M3 edge vs Buy & Hold",
    "expectancy": "M4 trade expectancy",
}

# Scalar context fields worth surfacing to the triage agent, per monitor node.
_DETAIL_KEYS = (
    "value", "value_pp", "pctl", "need", "ref_mdd", "cur_uw_days",
    "ref_max_uw_days", "gap_window", "mean_daily_gap_bps", "n_days",
    "n_trades", "first_breach_date",
)


def _parse_date(s: str | None) -> date | None:
    try:
        return date.fromisoformat(str(s)[:10])
    except (TypeError, ValueError):
        return None


def _detail(node: dict) -> dict:
    return {k: node[k] for k in _DETAIL_KEYS if node.get(k) is not None}


def collect_alarms(snap: dict, persist_days: int) -> list[dict]:
    as_of = _parse_date(snap.get("as_of"))
    alarms: list[dict] = []

    def check(scope: str, key: str, monitor: str, label: str,
              node: object, extra: dict | None = None) -> None:
        if not isinstance(node, dict):
            return
        status = node.get("status")
        breach = _parse_date(node.get("first_breach_date"))
        age = (as_of - breach).days if (as_of and breach) else None
        if status == "red":
            kind = "red"
        elif status == "yellow" and age is not None and age >= persist_days:
            kind = "persistent-yellow"
        else:
            return
        alarms.append(dict(
            scope=scope, key=key, monitor=monitor, label=label, status=status,
            kind=kind, first_breach_date=node.get("first_breach_date"),
            days_in_breach=age, detail=_detail(node), **(extra or {})))

    for s in snap.get("sleeves") or []:
        extra = dict(name=s.get("name"), parent=s.get("parent"),
                     instrument_kind=s.get("kind"), n_trades=s.get("n_trades"))
        for mon, label in SLEEVE_MONITORS.items():
            check("sleeve", s.get("key", "?"), mon, label, s.get(mon), extra)

    portfolio = snap.get("portfolio") or {}
    for name, p in (portfolio.get("profiles") or {}).items():
        check("profile", name, "dd", "M2 profile drawdown tripwire", p)
    check("portfolio", "book", "tracking", "M1 book-vs-replay tracking",
          portfolio.get("tracking"))
    return alarms


def fingerprint(alarms: list[dict]) -> str:
    ids = sorted(f"{a['scope']}:{a['key']}:{a['monitor']}:"
                 f"{a.get('first_breach_date') or '?'}" for a in alarms)
    return hashlib.sha256("|".join(ids).encode()).hexdigest()[:12]


def summarize(alarms: list[dict]) -> str:
    if not alarms:
        return "no actionable alarms"
    parts = []
    for a in alarms:
        tag = "RED" if a["kind"] == "red" else f"yellow {a['days_in_breach']}d"
        parts.append(f"{a['scope']}:{a['key']}/{a['monitor']}[{tag}]")
    return " · ".join(parts)


def _write_github_output(payload: dict) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as fh:
        for k, v in payload.items():
            fh.write(f"{k}={v}\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    ap.add_argument("--persist-days", type=int, default=DEFAULT_PERSIST_DAYS,
                    help="a yellow this many days old is treated as actionable "
                         f"(default {DEFAULT_PERSIST_DAYS})")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the alarm list (pretty JSON) here")
    ap.add_argument("--github-output", action="store_true",
                    help="force step outputs even without $GITHUB_OUTPUT "
                         "(printed KEY=VALUE lines)")
    args = ap.parse_args(argv)

    try:
        snap = json.loads(args.snapshot.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"ERROR: cannot read snapshot {args.snapshot}: {e}",
              file=sys.stderr)
        return 2

    alarms = collect_alarms(snap, args.persist_days)
    fp = fingerprint(alarms) if alarms else ""
    verdict = snap.get("verdict") or {}
    summary = summarize(alarms)

    print(f"Snapshot as-of {snap.get('as_of')} — verdict "
          f"{verdict.get('status', '?')} ({verdict.get('headline', '')}).")
    print(f"Actionable alarms ({len(alarms)}): {summary}")
    if alarms:
        print(f"Fingerprint: {fp}")
        for a in alarms:
            since = a.get("first_breach_date") or "unknown start"
            print(f"  - {a['scope']} {a['key']} · {a['label']} → "
                  f"{a['kind']} (since {since})")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(dict(
            as_of=snap.get("as_of"), generated_at_utc=snap.get("generated_at_utc"),
            persist_days=args.persist_days, fingerprint=fp,
            verdict=verdict, alarms=alarms), indent=1))
        print(f"Alarm payload written to {args.out}")

    outputs = dict(triage_needed=str(bool(alarms)).lower(),
                   n_alarms=len(alarms), fingerprint=fp, summary=summary)
    _write_github_output(outputs)
    if args.github_output and not os.environ.get("GITHUB_OUTPUT"):
        for k, v in outputs.items():
            print(f"{k}={v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
