#!/usr/bin/env python3
"""Guard the automated backtest-data refresh: validate a freshly-pulled
``data/backtest/raw_features_daily.csv`` before the workflow commits it, so a
failed or partial data pull can never overwrite the known-good versioned CSV
(the file the live app falls back to when blockchain.info is unreachable).

Compares the working-tree CSV against the version committed at ``HEAD`` and
fails (non-zero exit) if the refresh would REGRESS the dataset:

  * a required column is missing — especially the 11 on-chain ``oc_*`` series
    and the 3 Coinbase-premium ``cb_*`` columns the daily H/L model needs;
  * the recent on-chain tail is entirely NaN — i.e. the pull did not actually
    obtain fresh on-chain data (the exact failure mode this refresh exists to
    fix), so committing it would be pointless;
  * fewer rows than HEAD, or a last date older than HEAD (a backward step);
  * ``btc_close`` has non-positive / non-finite values.

Exit 0 = safe to commit; non-zero = discard the pull and skip the commit.

Run from the repo root:  python scripts/validate_refreshed_data.py
"""

import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CSV = "data/backtest/raw_features_daily.csv"

OC_COLS = [
    "oc_hash_rate", "oc_difficulty", "oc_n_transactions", "oc_miners_revenue",
    "oc_n_unique_addresses", "oc_transaction_fees_usd", "oc_mempool_size",
    "oc_estimated_transaction_volume_usd", "oc_market_cap", "oc_avg_block_size",
    "oc_cost_per_transaction",
]
CB_COLS = ["cb_premium", "cb_premium_ma3", "cb_premium_z7"]
BASE_COLS = ["btc_close", "btc_high", "btc_low", "btc_volume"]
REQUIRED = BASE_COLS + OC_COLS + CB_COLS

TAIL = 10  # rows of the recent tail that must carry real on-chain values


def _load_head_version() -> pd.DataFrame | None:
    """The CSV as committed at HEAD, or None when it can't be read (first commit)."""
    try:
        out = subprocess.run(
            ["git", "show", f"HEAD:{CSV}"],
            capture_output=True, text=True, check=True,
        ).stdout
        return pd.read_csv(io.StringIO(out), index_col=0, parse_dates=True)
    except Exception:
        return None


def main() -> int:
    if not Path(CSV).exists():
        print(f"FAIL: {CSV} does not exist")
        return 1
    try:
        new = pd.read_csv(CSV, index_col=0, parse_dates=True)
    except Exception as exc:
        print(f"FAIL: cannot read {CSV}: {exc}")
        return 1

    errs: list[str] = []

    # 1. Required columns present.
    missing = [c for c in REQUIRED if c not in new.columns]
    if missing:
        errs.append(f"missing required columns: {missing}")

    # 2. btc_close sanity.
    if "btc_close" in new.columns:
        bc = new["btc_close"].dropna()
        if bc.empty or (bc <= 0).any() or not np.isfinite(bc.to_numpy()).all():
            errs.append("btc_close has empty / non-positive / non-finite values")

    # 3. Recent on-chain tail must carry real values (the pull actually worked).
    present_oc = [c for c in OC_COLS if c in new.columns]
    if present_oc:
        tail = new[present_oc].tail(TAIL)
        if int(tail.notna().to_numpy().sum()) == 0:
            errs.append(f"recent on-chain tail (last {TAIL} rows) is entirely NaN")

    # 4. No regression versus the committed version.
    old = _load_head_version()
    if old is not None:
        if len(new) < len(old):
            errs.append(f"row count regressed: {len(new)} < {len(old)} at HEAD")
        if new.index.max() < old.index.max():
            errs.append(
                f"last date regressed: {new.index.max().date()} "
                f"< {old.index.max().date()} at HEAD"
            )
    else:
        print("WARN: no HEAD version of the CSV to compare against")

    if errs:
        print("VALIDATION FAILED — refuse to commit:")
        for e in errs:
            print(f"  - {e}")
        return 1

    tail_nn = (new[present_oc].tail(TAIL).notna().to_numpy().mean() * 100
               if present_oc else 0.0)
    print(
        f"OK: {len(new)} rows  {new.index.min().date()} -> {new.index.max().date()}  "
        f"| {len(present_oc)}/{len(OC_COLS)} on-chain cols  "
        f"| recent on-chain non-null {tail_nn:.0f}%"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
