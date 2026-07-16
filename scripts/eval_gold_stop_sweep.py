"""Stop-loss width sweep for the three traded gold assets (GDX · UGL · NUGT).

Each is swept through the SAME live engine the Gold / Overall apps use
(``gldm_engine`` → ``STOP_BY_ASSET``), so the numbers are actionable. For every
stop width we report Return / MDD / Sharpe / Win% / Trades on the full OOS
window (2021 → now). The shipped live stop is flagged with ``*``.

    python scripts/eval_gold_stop_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "app"))
sys.path.insert(0, str(_REPO))

import gldm_engine as ge   # noqa: E402
import gldm_core as gc     # noqa: E402

# fine grid, none … −20%
GRID = [(1.0, "none"), (0.20, "−20%"), (0.15, "−15%"), (0.12, "−12%"),
        (0.10, "−10%"), (0.08, "−8%"), (0.07, "−7%"), (0.06, "−6%"),
        (0.05, "−5%"), (0.04, "−4%"), (0.03, "−3%"), (0.02, "−2%")]

LIVE = {"GDX": 0.03, "UGL": 1.0, "NUGT": 0.05}   # shipped stops
HDR = f"  {'Stop':>6s} | {'Return':>11s} {'MDD':>8s} {'Sharpe':>7s} {'Win%':>6s} {'Trades':>7s}"


def _row(tag, sl, live):
    m = sl["metrics"]
    flag = " *" if live else ""
    return (f"  {tag:>6s} | {m['total_ret']*100:+10.1f}% {m['mdd']*100:+7.1f}% "
            f"{m['sharpe']:7.2f} {sl['win_rate']:5.0f}% {sl['n_trades']:7d}{flag}")


for KEY in ("GDX", "UGL", "NUGT"):
    label = {"GDX": "VanEck Gold Miners (high-beta 1×)",
             "UGL": "ProShares Ultra Gold (2×)",
             "NUGT": "Direxion Gold Miners (2×)"}[KEY]
    print("=" * 84)
    print(f"{KEY} — {label}   ·   gldm_engine · OOS 2021→now   (* = shipped stop)")
    print("=" * 84)
    print(HDR + "\n  " + "-" * 58)
    _orig = dict(gc.STOP_BY_ASSET)
    for stop, tag in GRID:
        gc.STOP_BY_ASSET[KEY] = stop
        grp = ge.run_gldm()
        sl = next((r for r in grp if r["key"] == KEY), None)
        if sl:
            is_live = abs(stop - LIVE[KEY]) < 1e-9
            print(_row(tag, sl, is_live))
    gc.STOP_BY_ASSET.clear(); gc.STOP_BY_ASSET.update(_orig)
    print()

print("Done.", flush=True)
