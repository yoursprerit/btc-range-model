#!/usr/bin/env python3
"""Refresh, quality-check and PIN every sleeve's daily dataset snapshot.

Runs the same gate the apps use (``app/data_gate.py``) for each generic
ticker sleeve (SOXX / GRID / XLE / REMX / WGMI / PBW / ARTY, per
``ticker_config``) plus the shared gold dataset (GLDM · UGL · GDX · NUGT),
headlessly and in completed-bars-only mode — so what gets committed is the
exact validated vintage every deployment then reuses at boot instead of
re-trusting a live Yahoo fetch.

* a fetch that PASSES quality control replaces the snapshot CSV and writes a
  manifest (rows, span, columns, SHA-256, per-check results) next to it;
* a fetch that FAILS is discarded — the last known-good snapshot stays, and
  this script exits non-zero so the workflow surfaces the failure (the good
  sleeves are still committed by the workflow's idempotent commit step).

Run from the repo root:  python scripts/refresh_sleeve_snapshots.py
Exit 0 = every sleeve pinned/refreshed; 1 = at least one sleeve fell back.

The BTC/MSTR/MSTU/ETH vintage is refreshed separately by
``scripts/pull_backtest_data.py`` (guarded by validate_refreshed_data.py) —
this script covers the equity/gold sleeves that used to fetch live only.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "app"))

# Pin completed bars at the publish anchor — never an in-progress session.
os.environ.setdefault("OVERALL_COMPLETED_BARS_ONLY", "1")

import ticker_config                                    # noqa: E402
import ticker_core as tc                                # noqa: E402
import gldm_core as gc                                  # noqa: E402
import data_gate as dg                                  # noqa: E402

CONSUMER = "refresh-workflow"


def _gldm_spec() -> dg.GateSpec:
    """The shared gold dataset spec (kept in sync with gldm_engine)."""
    traded = [f"gldm_{s}" for s in ("open", "high", "low", "close")]
    traded += [f"{t.lower()}_close" for t in gc.LEVERAGED_SYMBOLS]
    if "nugt_close" not in traded:
        traded.append("nugt_close")
    syms = {"gldm_close": gc.PRIMARY_SYMBOL}
    syms.update({f"{t.lower()}_close": t for t in gc.LEVERAGED_SYMBOLS})
    syms.setdefault("nugt_close", "NUGT")
    return dg.GateSpec(key="GLDM", price_col="gldm_close",
                       traded_close_cols=traded,
                       macro_close_cols=[f"{n}_close" for n in gc.MACRO_SYMS],
                       snapshot_csv=gc.DAILY_CACHE_CSV,
                       symbol_by_col=syms)


def main() -> int:
    jobs = [(k, dg.ticker_spec(ticker_config.get_config(k)),
             (lambda c=ticker_config.get_config(k): tc.fetch_daily(c)))
            for k in ticker_config.APP_KEYS]
    # gldm_core.fetch_daily already includes UGL/GDX/NUGT (LEVERAGED_SYMBOLS)
    jobs.append(("GLDM", _gldm_spec(),
                 lambda: gc.fetch_daily(start="2018-06-26")))

    bad: list[str] = []
    print(f"{'sleeve':8} {'decision':20} {'span':26} {'rows':>6}  "
          f"{'sha256':16}  failed checks")
    print("-" * 96)
    for key, spec, fetch in jobs:
        try:
            df = dg.gated_daily(spec, fetch_full=fetch, consumer=CONSUMER)
            info = df.attrs.get("data_gate", {})
        except Exception as exc:                        # never kill the loop
            info = dict(decision="error", failed_checks=[str(exc)])
            df = None
        decision = info.get("decision", "error")
        span = (f"{df.index.min().date()} → {df.index.max().date()}"
                if df is not None and len(df) else "—")
        notes = list(info.get("failed_checks") or [])
        # sessions taken back from the pinned snapshot because the feed had
        # withdrawn or only reconstructed them — the repair should be visible in
        # the refresh log, not silent (see data_gate._restore_from_snapshot)
        if info.get("spliced_sessions"):
            notes.append("restored: " + ", ".join(info["spliced_sessions"]))
        print(f"{key:8} {decision:20} {span:26} "
              f"{(len(df) if df is not None else 0):>6}  "
              f"{info.get('sha256', '—'):16}  "
              f"{', '.join(notes) or '—'}")
        if decision not in ("pinned", "refreshed"):
            bad.append(key)

    if bad:
        print(f"\nFAILED sleeves (kept their last known-good snapshot): {bad}")
        return 1
    print("\nOK: every sleeve pinned/refreshed with a validated snapshot.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
