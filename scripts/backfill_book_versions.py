"""Backfill strategy-logic provenance for archived books published BEFORE the
publisher stamped ``strategy_version`` / ``code_sha`` into the payload.

Writes ``data/overall/book_versions.json`` — a committed side-car mapping each
archived ``as_of`` to the commit that actually published it, labelled
``pre-v1``.  The as-published P&L view merges this with the inline stamps on
newer books (inline wins) to segment the record by strategy generation, so
old-logic and current-logic performance are never silently conflated.

Recovery: the archived books cannot be edited (they are HMAC-signed), and the
early ones were bulk-committed when the archive was introduced — but every
daily publish DID commit ``data/overall/target_book_live.json`` (falling back
to the paper ``target_book.json`` for days before the live book existed).
Walking that file's git history and parsing each revision's ``as_of``
recovers the publishing commit per signal day; the LAST commit per ``as_of``
wins, matching the archive's last-publish-wins semantics.

Idempotent — safe to re-run as the archive grows; archived books that already
carry an inline ``strategy_version`` are skipped (nothing to backfill).

    python scripts/backfill_book_versions.py            # writes the side-car
    python scripts/backfill_book_versions.py --dry-run  # print, don't write
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
SCHEMA = "book-versions/v1"
PRE_STAMP_VERSION = "pre-v1"      # every book published before inline stamping
ARCHIVE_DIR = _REPO / "data" / "overall" / "book_archive"
OUT_JSON = _REPO / "data" / "overall" / "book_versions.json"
BOOK_HISTORY_FILES = [            # daily-committed payloads, in priority order
    "data/overall/target_book_live.json",
    "data/overall/target_book.json",
]


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=_REPO, capture_output=True,
                          text=True, timeout=120).stdout


def publish_commits_by_as_of() -> dict[str, str]:
    """``as_of`` (YYYY-MM-DD) → short SHA of the commit that last published
    that signal day's book, recovered from the daily-committed book files."""
    out: dict[str, str] = {}
    for rel in BOOK_HISTORY_FILES:
        # oldest→newest so later publishes of the same as_of overwrite earlier
        shas = _git("log", "--reverse", "--format=%H", "--", rel).split()
        for sha in shas:
            try:
                payload = json.loads(_git("show", f"{sha}:{rel}"))
                as_of = str(payload.get("as_of") or "")
            except Exception:
                continue
            if as_of and (rel == BOOK_HISTORY_FILES[0] or as_of not in out):
                out[as_of] = sha[:12]         # live-book history takes priority
    return out


def build_sidecar() -> dict:
    commits = publish_commits_by_as_of()
    books: dict[str, dict] = {}
    for p in sorted(ARCHIVE_DIR.glob("*.json")):
        try:
            b = json.loads(p.read_text())
        except Exception:
            continue
        if b.get("strategy_version"):         # inline stamp — nothing to backfill
            continue
        as_of = str(b.get("as_of") or p.stem)
        books[as_of] = dict(
            strategy_version=PRE_STAMP_VERSION,
            code_sha=commits.get(as_of, "unknown"))
    return dict(
        schema=SCHEMA,
        generated_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        note=(f"Backfilled provenance for archived books published before "
              f"inline strategy_version stamping. '{PRE_STAMP_VERSION}' spans "
              "evolving logic (per-book code_sha is the publishing commit); "
              "books from stamping onward carry their version inline and are "
              "not listed here."),
        books=books)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="print the side-car instead of writing it")
    args = ap.parse_args()
    payload = build_sidecar()
    text = json.dumps(payload, indent=1)
    if args.dry_run:
        print(text)
        return 0
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(text)
    print(f"Wrote {OUT_JSON} — {len(payload['books'])} pre-stamp "
          f"book{'s' if len(payload['books']) != 1 else ''} labelled "
          f"'{PRE_STAMP_VERSION}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
