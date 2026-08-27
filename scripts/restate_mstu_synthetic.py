#!/usr/bin/env python3
"""One-off restatement of ``data/backtest/mstu_synthetic_daily.csv`` onto MSTU's
current split-adjusted price scale.

WHY THIS EXISTS
---------------
That CSV carries two different things in one column (see
``pull_backtest_data.synthesise_mstu``): OLS-synthetic prices *before* MSTU's
2024-09-18 inception, and MSTU's **actual closes** from inception on.  The
pull script pins it with ``_freeze_history(..., tail=0)`` — every committed row
frozen forever — because re-fitting the OLS would rewrite pre-inception history.

That freeze is not split-safe.  ``mstu_daily.csv`` is deliberately left
unfrozen "for split safety" (it gets a scale-invariant returns check in
``validate_refreshed_data.py``), so when MSTU reverse-split ~9.9:1 in late
August 2026 that file was restated onto the new scale — while every pinned row
in the synthetic file kept the OLD scale.  Newly appended rows then arrived on
the NEW scale, splicing the two together and fabricating a **+917.81%
single-day gain** between 2026-08-21 and 2026-08-24.

Nothing about that day was real: MSTU actually moved +5.71% and MSTR +2.83%.
The engine prefers this file over ``mstu_daily.csv`` (``btc_ct_engine.
_load_prices``), so the fabricated bar reached the whole MSTU sleeve — most
visibly the Overall app's equal-weight buy & hold curve, which it inflated
from ~+20% to +80% since 2026-07-31.

WHAT IT DOES
------------
Restates the file to what an unfrozen rebuild would produce, without re-fitting
(so the vintage freeze's actual purpose — pinned pre-inception *history* — is
preserved):

  * rows **before** inception — multiplied by the scale factor K.  Returns are
    untouched to float precision; only the price level moves.  K is taken from
    the inception anchor (the OLS series is built so its last pre-inception
    value IS MSTU's first actual close), which makes the splice exact by
    construction, and is cross-checked against the median ``actual/pinned``
    ratio over the clean history.
  * rows **from** inception — replaced with ``mstu_daily.csv`` closes verbatim.
    That is this file's own contract, it is split-safe by design, and it also
    repairs the frozen August 2026 rows, which had drifted from the real closes
    by up to 13% before the splice.

Idempotent: once restated, K collapses to 1.0 and both halves already match, so
a second run is a no-op and says so.

    python scripts/restate_mstu_synthetic.py            # dry run, prints the diff
    python scripts/restate_mstu_synthetic.py --apply    # write the CSV
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SYN_CSV = REPO / "data" / "backtest" / "mstu_synthetic_daily.csv"
REAL_CSV = REPO / "data" / "backtest" / "mstu_daily.csv"

# MSTU's first trading day — the boundary between the OLS back-fill and the
# real closes (``pull_backtest_data.INCEPTION``).
INCEPTION = pd.Timestamp("2024-09-18")

# a rescale smaller than this is float noise, not a corporate action
SCALE_EPS = 1e-6
# the pinned/actual ratio must be this flat across the clean history before we
# believe it is a single global rescale rather than a corrupted series
RATIO_FLATNESS = 1e-5
# |Δ daily return| at or below this is CSV float round-trip, not a restatement
RETURN_NOISE_FLOOR = 1e-6


def _read(path: Path, col: str = "close") -> pd.Series:
    df = pd.read_csv(path, parse_dates=["Date"], float_precision="round_trip")
    return (df.set_index("Date").sort_index()[col].astype(float)
            .rename(path.stem))


def restate(syn: pd.Series, real: pd.Series) -> tuple[pd.Series, dict]:
    """Return the restated series plus a report of what moved and why."""
    pre = syn[syn.index < INCEPTION]
    post = syn[syn.index >= INCEPTION]
    if pre.empty or post.empty or real.empty:
        raise SystemExit("unexpected file shape — refusing to guess")

    # K from the inception anchor: the OLS back-fill is built so its last bar
    # equals MSTU's first actual close, so this makes the splice exact
    k_anchor = float(real.iloc[0] / post.iloc[0])

    # cross-check: the same constant should relate pinned to actual across the
    # whole clean history.  Anything the splice already corrupted is excluded
    # by taking the median and reporting the spread.
    ratio = (real / syn).dropna()
    flat = ratio[(ratio - k_anchor).abs() / k_anchor < RATIO_FLATNESS]
    k_median = float(np.median(flat)) if len(flat) else float("nan")
    if len(flat) < 100:
        raise SystemExit(
            f"only {len(flat)} rows agree with the anchor scale {k_anchor:.9f} "
            "— this is not a clean single rescale; investigate before restating")
    drift = abs(k_anchor - k_median) / k_median
    if drift > RATIO_FLATNESS:
        raise SystemExit(f"anchor scale {k_anchor:.9f} and median scale "
                         f"{k_median:.9f} disagree ({drift*1e6:.1f} ppm)")

    fixed_pre = pre * k_anchor
    # the file's contract: from inception this column IS mstu_daily.csv
    fixed_post = real.reindex(post.index.union(real.index)).dropna()
    fixed = pd.concat([fixed_pre, fixed_post]).sort_index()
    fixed = fixed[~fixed.index.duplicated(keep="last")]

    old_ch = syn.pct_change()
    new_ch = fixed.reindex(syn.index).pct_change()
    changed = (~np.isclose(fixed.reindex(syn.index), syn, rtol=1e-9)).sum()
    # A pure rescale leaves every return alone in exact arithmetic, but the CSV
    # stores ~9 significant digits, so round-tripping moves them by ~1e-8.  Only
    # a move far above that floor is a real restatement of what happened that
    # day — report the two separately rather than lumping noise in with signal.
    delta = (new_ch - old_ch).abs().dropna()
    real_moves = delta[delta > RETURN_NOISE_FLOOR]
    return fixed, {
        "k": k_anchor, "k_median": k_median, "n_flat": len(flat),
        "n_pre": len(pre), "n_post": len(post), "n_rows_rescaled": int(changed),
        "n_returns_changed": len(real_moves),
        "n_returns_noise": int(len(delta) - len(real_moves)),
        "returns_noise_max": float(delta[delta <= RETURN_NOISE_FLOOR].max())
                             if len(delta) > len(real_moves) else 0.0,
        "returns_changed_from": (str(real_moves.index[0].date())
                                 if len(real_moves) else None),
        "returns_changed_to": (str(real_moves.index[-1].date())
                               if len(real_moves) else None),
        "old_max_1d": float(old_ch.max()), "old_max_1d_on": str(old_ch.idxmax().date()),
        "new_max_1d": float(new_ch.max()), "new_max_1d_on": str(new_ch.idxmax().date()),
        "old_span": (float(syn.iloc[0]), float(syn.iloc[-1])),
        "new_span": (float(fixed.iloc[0]), float(fixed.iloc[-1])),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="write the restated CSV (default: dry run)")
    args = ap.parse_args()

    syn, real = _read(SYN_CSV), _read(REAL_CSV)
    print(f"synthetic : {syn.index[0].date()} → {syn.index[-1].date()}  n={len(syn)}")
    print(f"actual    : {real.index[0].date()} → {real.index[-1].date()}  n={len(real)}")

    fixed, rep = restate(syn, real)

    if abs(rep["k"] - 1.0) < SCALE_EPS and rep["n_rows_rescaled"] == 0:
        print("\nAlready on the current split-adjusted scale — nothing to do.")
        return 0

    print(f"\nscale factor K = {rep['k']:.9f}  (anchor)"
          f"  ·  {rep['k_median']:.9f} (median over {rep['n_flat']} clean rows)")
    print(f"  {rep['n_pre']} pre-inception rows rescaled ×K — returns unchanged")
    print(f"  {rep['n_post']} rows from inception replaced with mstu_daily.csv closes")
    print(f"  {rep['n_rows_rescaled']} of {len(syn)} rows change value")
    print(f"  {rep['n_returns_changed']} rows change what actually HAPPENED "
          f"that day"
          + (f" ({rep['returns_changed_from']} → {rep['returns_changed_to']})"
             if rep["returns_changed_from"] else "")
          + f"; the other {rep['n_returns_noise']} move by "
            f"≤{rep['returns_noise_max']:.1e} (CSV round-trip)")
    print(f"\n  largest 1-day move  before: {rep['old_max_1d']*100:+10.2f}% "
          f"on {rep['old_max_1d_on']}")
    print(f"  largest 1-day move  after : {rep['new_max_1d']*100:+10.2f}% "
          f"on {rep['new_max_1d_on']}")
    print(f"  price span          before: {rep['old_span'][0]:.4f} → {rep['old_span'][1]:.4f}")
    print(f"  price span          after : {rep['new_span'][0]:.4f} → {rep['new_span'][1]:.4f}")

    if not args.apply:
        print("\nDry run — pass --apply to write.")
        return 0
    out = fixed.rename("close").to_frame()
    out.index.name = "Date"
    out.to_csv(SYN_CSV)
    print(f"\nWrote {SYN_CSV.relative_to(REPO)}  ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
