"""No-look-ahead check for the walk-forward gated replay (the app's back-test).

A back-test is free of look-ahead iff its history never changes when the world
is truncated: the daily returns it reports for days ≤ T must be IDENTICAL
whether it is run on data through T or on data through today.  This script
enforces that *prefix-invariance* directly:

  1. run the full universe once and build the walk-forward gated replay per
     risk profile on the complete history;
  2. for several cutoff dates T, truncate EVERY input the replay consumes to
     ≤ T — each sleeve's return/position series, its ``bh`` price path, its
     closed-trade log, and (via a patched ``_load_daily``) the macro data the
     sentiment gauge reads — and rebuild the replay from scratch;
  3. assert the truncated run's daily returns, weight matrix and anchor
     schedule exactly match the full run's prefix (tolerance 1e-12, i.e.
     float-exact).

Any use of future information anywhere in the pipeline — anchor fitting,
priority components, gating, water-fill — would shift some pre-T value and
fail the assertion.  What this cannot test: the per-sleeve ENGINE series
(``pos_series``/returns) are inputs here, not recomputed; their causality
(decide at close t−1, fill at t) is the engines' own contract, audited
separately (see the 2026-07-25 look-ahead fix notes in the Methodology tab).

    python scripts/check_lookahead.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "app"))
sys.path.insert(0, str(_REPO))

import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
import overall_core as ov                # noqa: E402

CUTOFFS = ["2025-06-30", "2026-01-15", "2026-06-30"]
TOL = 1e-12


def truncate_results(results: list[dict], cutoff) -> list[dict]:
    """A copy of the universe as it existed at ``cutoff``'s close: every dated
    field the replay reads is sliced to ≤ cutoff; nothing else is touched."""
    cutoff = pd.Timestamp(cutoff)
    out = []
    for res in results:
        dates = pd.DatetimeIndex(pd.Series(res["dates"]))
        keep = dates <= cutoff
        n = int(keep.sum())
        r = dict(res["r"])
        r["bh"] = np.asarray(r["bh"], float)[:n]
        log = r.get("trade_log")
        if log:
            r["trade_log"] = [t for t in log if isinstance(t, dict)
                              and pd.Timestamp(t["exit_date"]) <= cutoff]
        tr = r.get("trades")
        if tr is not None and len(tr) and isinstance(tr[0], dict):
            r["trades"] = [t for t in tr if t.get("exit_date") is not None
                           and pd.Timestamp(t["exit_date"]) <= cutoff]
        new = dict(res)
        new["r"] = r
        new["dates"] = pd.Series(dates[keep])
        new["ret"] = res["ret"].loc[:cutoff]
        new["pos_series"] = res["pos_series"].loc[:cutoff]
        out.append(new)
    return out


def main() -> None:
    print("Running the universe once (full history)…")
    results = ov.run_universe()
    print(f"{len(results)} sleeves\n")

    orig_load_daily = ov._load_daily
    failures = 0
    for pname, prof in ov.RISK_PROFILES.items():
        caps = ov.caps_for(pname)
        full = ov.walkforward_gated_replay(
            results, caps=caps, mdd_floor=prof["mdd_floor"],
            objective=prof["objective"], sata_daily=ov.SATA_DAILY, tilt=True)
        for cut in CUTOFFS:
            cut_ts = pd.Timestamp(cut)
            trunc_res = truncate_results(results, cut_ts)
            # the sentiment gauge reads macro CSV/Yahoo data through _load_daily;
            # truncate that world too, so nothing post-cutoff exists anywhere
            ov._load_daily = lambda cfg, _c=cut_ts: orig_load_daily(cfg).loc[:_c]
            try:
                part = ov.walkforward_gated_replay(
                    trunc_res, caps=caps, mdd_floor=prof["mdd_floor"],
                    objective=prof["objective"], sata_daily=ov.SATA_DAILY,
                    tilt=True)
            finally:
                ov._load_daily = orig_load_daily

            idx = part["ret"].index
            d_ret = float((part["ret"] - full["ret"].loc[idx]).abs().max())
            d_w = float((part["weights"] - full["weights"].loc[idx]).abs().to_numpy().max())
            d_sata = float((part["sata"] - full["sata"].loc[idx]).abs().max())
            anch_full = [(d, w) for d, w in full["anchors"] if pd.Timestamp(d) <= cut_ts]
            anch_ok = len(anch_full) == len(part["anchors"]) and all(
                pd.Timestamp(d1) == pd.Timestamp(d2)
                and max(abs(w1.get(k, 0.0) - w2.get(k, 0.0))
                        for k in set(w1) | set(w2)) < TOL
                for (d1, w1), (d2, w2) in zip(anch_full, part["anchors"]))
            ok = d_ret < TOL and d_w < TOL and d_sata < TOL and anch_ok
            failures += (not ok)
            print(f"{'PASS' if ok else 'FAIL'}  {pname:<11} cutoff {cut}  "
                  f"({len(idx)} bars)  Δret {d_ret:.2e} · Δweights {d_w:.2e} · "
                  f"ΔSATA {d_sata:.2e} · anchors {'match' if anch_ok else 'DIFFER'}")

    print()
    if failures:
        print(f"❌ {failures} prefix-invariance violation(s) — look-ahead present.")
        sys.exit(1)
    print("✅ Prefix-invariant at every cutoff × profile — the replay's history "
          "never changes when the future is removed. No look-ahead in the "
          "allocation pipeline.")


if __name__ == "__main__":
    main()
