"""Build the 🩺 Strategy Health snapshot (nightly decay monitor).

Runs the full Overall engine, the walk-forward gated replay per monitored risk
profile, and the published-book realized-return reconstruction, then writes:

    data/overall/strategy_health.json   — current snapshot + flags (the UI reads this)
    data/overall/health_history.csv     — append-only daily metric series

Invoked by .github/workflows/publish-target-book.yml right after the target
book publishes (best-effort: a health failure never blocks the book), or by
hand:

    python scripts/build_strategy_health.py
    python scripts/build_strategy_health.py --profiles Balanced

Breach persistence (``first_breach_date``) is carried from the previous
committed snapshot, so flags are auditable in git history.  Methodology and
thresholds: STRATEGY_HEALTH.md.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "app"))
sys.path.insert(0, str(_REPO))

import overall_core as oc          # noqa: E402
import health_core as hc           # noqa: E402

DEFAULT_OUT = _REPO / "data" / "overall" / "strategy_health.json"
DEFAULT_HIST = _REPO / "data" / "overall" / "health_history.csv"
ARCHIVE_DIR = _REPO / "data" / "overall" / "book_archive"
# Monitor the profiles the UI offers (Aggressive is retired from the interface);
# the first entry doubles as the tracking monitor's replay benchmark and must be
# the profile the published book trades (overall_core.DEFAULT_PROFILE).
DEFAULT_PROFILES = [oc.DEFAULT_PROFILE] + \
    [p for p in oc.RISK_PROFILES if p not in (oc.DEFAULT_PROFILE, "Aggressive")]


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the strategy-health snapshot")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--history", default=str(DEFAULT_HIST))
    ap.add_argument("--profiles", nargs="*", default=DEFAULT_PROFILES,
                    choices=list(oc.RISK_PROFILES), metavar="PROFILE",
                    help=f"risk profiles to monitor (default: {' '.join(DEFAULT_PROFILES)})")
    args = ap.parse_args()

    print("Running the universe (all sleeves)…")
    results = oc.run_universe()
    if not results:
        raise RuntimeError("run_universe() returned no instruments — check data feeds")
    print(f"  {len(results)} instruments.")

    replays: dict[str, dict] = {}
    for name in args.profiles:
        prof = oc.RISK_PROFILES[name]
        print(f"Walk-forward gated replay — {name}…")
        try:
            replays[name] = oc.walkforward_gated_replay(
                results, caps=oc.caps_for(name), mdd_floor=prof["mdd_floor"],
                objective=prof["objective"], sata_daily=oc.SATA_DAILY, tilt=True)
        except Exception:
            # one profile failing must not sink the snapshot for the rest
            traceback.print_exc()
            replays[name] = None

    books = hc.load_books(ARCHIVE_DIR)
    print(f"Loaded {len(books)} archived books from {ARCHIVE_DIR}.")
    prev = hc.load_prev(Path(args.out))

    snap = hc.build_snapshot(results, replays, books, prev,
                             sata_daily=oc.SATA_DAILY,
                             rets=oc.returns_matrix(results))
    hc.write_artifacts(snap, Path(args.out), Path(args.history))
    v = snap["verdict"]
    print(f"Wrote {args.out} + {args.history}")
    print(f"Verdict: {hc.STATUS_EMOJI[v['status']]} {v['headline']} "
          f"({v['n_sleeves']} sleeves, {v['n_warming']} warming up)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
