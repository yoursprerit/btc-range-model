"""Set which positions are dropped from the published target books.

The publisher (``scripts/publish_target_book.py``) and the 2:00-PM-CT intraday
stop check read ``data/overall/book_exclusions.json`` and leave those names out
of the matching book — their weight goes to SATA (up to the V2 cap) and cash,
not to the other names.  An exclusion is stamped with TODAY's CT date and applies
to today's publishes only; tomorrow's book is the strategy's, unamended, so the
exclusion must be repeated each day it is intended.  Commit the file for the
publish to pick it up.

    python scripts/set_book_exclusions.py --paper SOXX SOXL --live SOXX SOXL
    python scripts/set_book_exclusions.py --clear
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
PATH = _REPO / "data" / "overall" / "book_exclusions.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--paper", nargs="*", metavar="KEY", help="exclude from the PAPER book")
    ap.add_argument("--live", nargs="*", metavar="KEY", help="exclude from the LIVE book")
    ap.add_argument("--clear", action="store_true", help="remove every exclusion")
    a = ap.parse_args()
    today = datetime.now(ZoneInfo("America/Chicago")).date().isoformat()
    cur = {"paper": [], "live": []}
    try:
        old = json.loads(PATH.read_text())
        if old.get("date") == today and not a.clear:   # keep only today's list
            cur.update({m: old.get(m) or [] for m in cur})
    except Exception:
        pass
    for mode in ("paper", "live"):
        keys = getattr(a, mode)
        if keys is not None:
            cur[mode] = sorted({k.upper() for k in keys})
    cur["date"] = today
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(cur, indent=1) + "\n")
    print(f"{PATH}: paper={cur['paper']} live={cur['live']} (today only: {today})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
