"""One-time backfill of ``data/overall/executed_archive/`` from git history.

Every rebalance committed ``data/overall/executed_book.json`` (and, in live
mode, ``executed_book_live.json``), but those files only ever hold the LAST
run — each execution overwrote the previous one, so the record of what was
traded on earlier days survives only in the repo's history.  This walks every
commit that touched a report file, keeps the LAST run per signal bar
(``as_of``) per account mode, and writes each winner to
``executed_archive/<as_of>[_live].json`` — the dated records the Executed Book
page's **🕰️ Historical** tab browses.

The bytes are copied VERBATIM from git, so historical HMAC signatures remain
valid.  Paper and live runs stay separate (a paper fill is not a live fill).
Idempotent: re-running overwrites the archive with the same content.  Needs an
unshallowed clone — a shallow checkout silently truncates the record, so the
script refuses one.

Usage:  python scripts/backfill_executed_archive.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "app"))
import executed_book as eb  # noqa: E402

_REPORTS = {"": "data/overall/executed_book.json",
            "_live": "data/overall/executed_book_live.json"}


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(_REPO), *args], check=True,
                          capture_output=True, text=True).stdout


def _show(commit: str, path: str) -> str | None:
    try:
        return _git("show", f"{commit}:{path}")
    except subprocess.CalledProcessError:
        return None            # file absent at that commit


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be written without writing")
    ap.add_argument("--ignore-reset", action="store_true",
                    help="restore runs from before an account reset too (they "
                         "belong to a PREVIOUS account — normally you do not "
                         "want them back)")
    args = ap.parse_args()

    if _git("rev-parse", "--is-shallow-repository").strip() == "true":
        print("ERROR: shallow clone — run `git fetch --unshallow` first, or "
              "the archive would silently miss older executions.", file=sys.stderr)
        return 1

    commits = _git("log", "--format=%H", "--", *_REPORTS.values()).split()
    print(f"{len(commits)} commits touched an execution report.")

    # per (variant, as_of): (generated_at, verbatim_text) — last run wins, the
    # same rule the executor's own archiving applies
    best: dict[tuple[str, str], tuple[str, str]] = {}
    for c in commits:
        for variant, path in _REPORTS.items():
            text = _show(c, path)
            if not text:
                continue
            try:
                payload = json.loads(text)
            except Exception:
                continue
            if payload.get("schema") != eb.SCHEMA:
                continue
            as_of, gen = payload.get("as_of"), payload.get("generated_at_utc")
            if not as_of or not gen:
                continue
            cur = best.get((variant, as_of))
            if cur is None or gen > cur[0]:
                best[(variant, as_of)] = (gen, text)

    if not best:
        print("No execution reports found in history — nothing to do.")
        return 0

    skipped = 0
    for variant, as_of in sorted(best):
        gen, text = best[(variant, as_of)]
        anchor = _REPO / _REPORTS[variant]
        # a reset retired everything up to its cutoff — those runs are a
        # different account's, and restoring them is what the marker prevents
        reset = eb.read_reset(anchor)
        if reset and not args.ignore_reset and as_of <= reset.get("cutoff_as_of", ""):
            skipped += 1
            continue
        out = eb.archive_path(anchor, as_of)
        mode = "live" if variant else "paper"
        print(f"  {as_of} ← executed {gen} ({mode}) → {out.relative_to(_REPO)}")
        if not args.dry_run:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text)
    if skipped:
        print(f"  skipped {skipped} run(s) retired by an account reset "
              f"(--ignore-reset to restore them anyway)")
    print(f"{'Would write' if args.dry_run else 'Wrote'} {len(best) - skipped} "
          f"record(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
