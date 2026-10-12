"""Build the precomputed 📈 Overall-strategy-P&L snapshot.

    python scripts/build_pnl_snapshot.py            # skip if nothing changed
    python scripts/build_pnl_snapshot.py --force    # always rewrite

Writes data/overall/pnl_snapshot.pkl.xz by default (see app/pnl_snapshot.py); CI
points --out at a checkout of the data-snapshots branch.  Invoked daily
by .github/workflows/build-pnl-snapshot.yml.  Exit 0 and "unchanged" when the
engine code, data vintage and last bar match the committed snapshot and it is
still fresh, so repeated CI slots do not grow the repo.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "app"))
sys.path.insert(0, str(_REPO))

import pnl_snapshot as ps       # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the P&L snapshot")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", type=Path, default=ps.SNAPSHOT_PATH)
    a = ap.parse_args()

    t0 = time.time()
    print("Building P&L snapshot (all views × profiles)…", flush=True)
    snap = ps.build()
    key = ps.content_key(snap)

    if not a.force and a.out.exists():
        try:
            old = ps.read(a.out)
            if ps.content_key(old) == key:
                print(f"unchanged ({key}) — keeping the committed snapshot")
                return 0
        except Exception as exc:
            print(f"(could not compare with existing snapshot: {exc})")

    ps.save(snap, a.out)
    print(f"wrote {a.out} ({a.out.stat().st_size/1e6:.2f} MB) — last bar "
          f"{snap['last_bar']}, {len(snap['packs'])} packs, "
          f"{time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
