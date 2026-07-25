"""One-time backfill of ``data/overall/book_archive/`` from git history.

Every daily publish committed ``data/overall/target_book_live.json`` (and the
paper ``target_book.json``), so the full record of what was actually published
lives in the repo's history.  This walks every commit that touched a book
file, keeps the LAST publish per ``as_of`` day (an intraday re-publish
replaced that day's book — the executor traded the final one), and writes each
winner to ``book_archive/<as_of>.json``.

The bytes are copied VERBATIM from git, so historical HMAC signatures remain
valid.  LIVE books are preferred over paper ones for the same publish (same
risk weights; the live book is the one traded).  Idempotent: re-running
overwrites the archive with the same content.  Needs an unshallowed clone —
a shallow checkout silently truncates the record, so the script refuses one.

Usage:  python scripts/backfill_book_archive.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "app"))
import target_book as tb  # noqa: E402

_BOOK_PATHS = ["data/overall/target_book_live.json",
               "data/overall/target_book.json"]
_ARCHIVE_ANCHOR = _REPO / "data" / "overall" / "target_book_live.json"


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
    args = ap.parse_args()

    if _git("rev-parse", "--is-shallow-repository").strip() == "true":
        print("ERROR: shallow clone — run `git fetch --unshallow` first, or "
              "the archive would silently miss older publishes.", file=sys.stderr)
        return 1

    commits = _git("log", "--format=%H", "--", *_BOOK_PATHS).split()
    print(f"{len(commits)} commits touched a target-book file.")

    # per as_of: (generated_at, is_live, verbatim_text) — newest publish wins,
    # live preferred over paper at equal generation time
    best: dict[str, tuple[str, bool, str]] = {}
    for c in commits:
        for path in _BOOK_PATHS:
            text = _show(c, path)
            if not text:
                continue
            try:
                payload = json.loads(text)
            except Exception:
                continue
            as_of, gen = payload.get("as_of"), payload.get("generated_at_utc")
            if not as_of or not gen:
                continue
            is_live = payload.get("book_mode") == "live"
            cur = best.get(as_of)
            if cur is None or (gen, is_live) > (cur[0], cur[1]):
                best[as_of] = (gen, is_live, text)

    if not best:
        print("No publishable books found in history — nothing to do.")
        return 0

    for as_of in sorted(best):
        gen, is_live, text = best[as_of]
        out = tb.archive_path(_ARCHIVE_ANCHOR, as_of)
        mode = "live" if is_live else "paper"
        print(f"  {as_of} ← published {gen} ({mode}) → {out.relative_to(_REPO)}")
        if not args.dry_run:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text)
    print(f"{'Would write' if args.dry_run else 'Wrote'} {len(best)} record(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
