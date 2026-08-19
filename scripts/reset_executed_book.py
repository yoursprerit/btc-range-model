"""Reset the Executed Book after resetting the broker account.

A fresh paper account (or a wiped balance) makes every earlier execution report
a record of a DIFFERENT account: same schema, unrelated money. Showing them
alongside the new account's runs is worse than showing nothing — the net-liq
chart in the Historical tab would step across two unrelated accounts, and the
duplicate-run lock could refuse a bar the new account has never traded.

This removes the current report and the archived runs, and writes a reset marker
(``executed_archive/_reset.json``) recording the cutoff. The marker matters: the
old reports live on in git history, so without it
``scripts/backfill_executed_archive.py`` would restore exactly what was removed.

    python scripts/reset_executed_book.py --dry-run     # see what would go
    python scripts/reset_executed_book.py               # paper (default)
    python scripts/reset_executed_book.py --account-mode live

By default the cutoff is the newest archived run, so everything up to and
including it is retired and the next run starts the record fresh. Pass
``--cutoff`` to retire a specific bar instead.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "app"))
import executed_book as eb  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--account-mode", choices=("paper", "live"), default="paper",
                    help="which account's record to reset (default: paper)")
    ap.add_argument("--cutoff", help="retire runs up to and including this signal "
                                     "bar (default: the newest archived run)")
    ap.add_argument("--reason", default="account reset",
                    help="recorded in the marker for whoever reads it later")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be removed without removing it")
    args = ap.parse_args()

    report = _REPO / "data" / "overall" / (
        "executed_book_live.json" if args.account_mode == "live"
        else "executed_book.json")
    records = eb.archived_records(report)
    cutoff = args.cutoff or (records[0]["as_of"] if records else None)
    if cutoff is None and not report.exists():
        print(f"Nothing to reset for the {args.account_mode} account.")
        return 0

    doomed = [report] if report.exists() else []
    doomed += [r["path"] for r in records]
    for p in doomed:
        print(f"  {'would remove' if args.dry_run else 'removing'} "
              f"{p.relative_to(_REPO)}")
    if cutoff:
        print(f"  {'would record' if args.dry_run else 'recording'} reset marker "
              f"→ runs up to and including {cutoff} are retired")
    if args.dry_run:
        print(f"Would reset the {args.account_mode} Executed Book "
              f"({len(doomed)} file(s)).")
        return 0

    for p in doomed:
        p.unlink(missing_ok=True)
    if cutoff:
        marker = eb.write_reset(report, cutoff, args.reason)
        print(f"Wrote {marker.relative_to(_REPO)}")
    print(f"Reset the {args.account_mode} Executed Book ({len(doomed)} file(s) "
          f"removed). The next execution report starts the record fresh.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
