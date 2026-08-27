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
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CSV = "data/backtest/raw_features_daily.csv"

# Vintage-freeze contract (see scripts/pull_backtest_data.py::_freeze_history):
# committed history older than the newest FREEZE_TAIL rows must survive a
# refresh byte-identically — a pull that restates the past is corrupt (or a
# deliberate re-baseline, which must bypass BOTH the pull freeze and this
# validator explicitly).  MSTR/MSTU actual are yfinance-adjusted, where a
# split legitimately rescales all prices, so they get a scale-invariant
# RETURNS-agreement check instead of a frozen-values check.
FREEZE_TAIL = 5
FROZEN_CSVS = {  # path → frozen tail window
    "data/backtest/raw_features_daily.csv": FREEZE_TAIL,
    "data/backtest/btc_usd_daily.csv": FREEZE_TAIL,
    "data/backtest/eth_usd_daily.csv": FREEZE_TAIL,
    "data/backtest/mstu_synthetic_daily.csv": 0,
}
# Frozen files whose pinned history may be rescaled as a whole but never
# reshaped.  mstu_synthetic_daily.csv is half OLS back-fill (rightly pinned —
# a re-fit rewrites the past) and half MSTU's own split-adjusted closes, which
# a corporate action legitimately restates.  Checking levels there rejects a
# real split; checking them after dividing out one global factor still rejects
# every restatement that changes what happened on a day.  Not applying this
# is how the 2026-08 MSTU splice reached production.
SCALE_INVARIANT_FROZEN = {"data/backtest/mstu_synthetic_daily.csv"}
# A single session moving more than this in a daily close series is a data
# artifact, not a market: even 3x ETFs cannot get there without a corporate
# action, and a corporate action is applied to the whole history, not one bar.
MAX_1D_LOG_MOVE = np.log(1.8)
JUMP_CSVS = {  # path → close column checked for fabricated single-day jumps
    "data/backtest/btc_usd_daily.csv": "close",
    "data/backtest/eth_usd_daily.csv": "close",
    "data/backtest/mstr_daily.csv": "close",
    "data/backtest/mstu_daily.csv": "close",
    "data/backtest/mstu_synthetic_daily.csv": "close",
}
RETURNS_CSVS = {  # path → close column checked for return agreement
    "data/backtest/mstr_daily.csv": "close",
    "data/backtest/mstu_daily.csv": "close",
}
_REL_TOL = 1e-9          # frozen values: effectively exact (CSV float roundtrip)
_RET_TOL = 1e-3          # |Δ log-return| on adjusted series
_RET_MAX_BAD = 6         # tolerated corrected days out of the last 120

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


def _load_head_version(path: str = CSV) -> pd.DataFrame | None:
    """The CSV as committed at HEAD, or None when it can't be read (first commit)."""
    try:
        out = subprocess.run(
            ["git", "show", f"HEAD:{path}"],
            capture_output=True, text=True, check=True,
        ).stdout
        return pd.read_csv(io.StringIO(out), index_col=0, parse_dates=True,
                           float_precision="round_trip")
    except Exception:
        return None


def _global_scale(a: pd.Series, b: pd.Series, flatness: float = 1e-5) -> float:
    """``a/b`` when the two series differ by one flat factor (a split), else 1.0.

    Flatness is the whole test: a corporate action rescales every bar by the
    same number, so anything that only rescales SOME of them is a restatement
    and must keep failing the frozen check."""
    both = a.dropna().index.intersection(b.dropna().index)
    if len(both) < 100:
        return 1.0
    ratio = (a.loc[both] / b.loc[both]).replace([np.inf, -np.inf],
                                                np.nan).dropna()
    if len(ratio) < 100:
        return 1.0
    k = float(np.median(ratio))
    if not np.isfinite(k) or k <= 0:
        return 1.0
    return k if float(((ratio - k).abs() / k).max()) <= flatness else 1.0


def check_frozen_history(new: pd.DataFrame, old: pd.DataFrame,
                         tail: int, rel_tol: float = _REL_TOL,
                         scale_invariant: bool = False) -> list[str]:
    """Errors if any committed row older than ``old``'s last ``tail`` rows was
    dropped or had a shared-column value changed beyond ``rel_tol``.  NaN in
    both vintages agrees; NaN in exactly one is a restatement.

    With ``scale_invariant``, a pinned column that the fresh vintage rescales by
    one flat factor across its whole history — a split — is compared on that
    common scale instead of being rejected outright.  Any bar whose value moves
    RELATIVE to the rest still fails."""
    errs: list[str] = []
    if old is None or not len(old):
        return errs
    frozen = old.index[:-tail] if tail else old.index
    missing = frozen.difference(new.index)
    if len(missing):
        errs.append(f"{len(missing)} pinned row(s) dropped "
                    f"(e.g. {pd.Timestamp(missing[0]).date()})")
    shared = frozen.intersection(new.index)
    for c in old.columns:
        if c not in new.columns:
            errs.append(f"pinned column '{c}' dropped")
            continue
        a = pd.to_numeric(old.loc[shared, c], errors="coerce")
        b = pd.to_numeric(new.loc[shared, c], errors="coerce")
        if scale_invariant:
            k = _global_scale(b, a)
            if k != 1.0:
                a = a * k          # compare the pinned vintage on the new scale
        nan_flip = int((a.isna() != b.isna()).sum())
        both = a.notna() & b.notna()
        denom = a[both].abs().where(a[both].abs() > 0, 1.0)
        bad = int(((a[both] - b[both]).abs() / denom > rel_tol).sum())
        if nan_flip or bad:
            errs.append(f"'{c}': {bad} pinned value(s) restated"
                        + (f", {nan_flip} NaN flip(s)" if nan_flip else ""))
    return errs


def check_returns_agreement(new: pd.DataFrame, old: pd.DataFrame,
                            close_col: str, window: int = 120,
                            tol: float = _RET_TOL,
                            max_bad: int = _RET_MAX_BAD) -> list[str]:
    """Scale-invariant history check for split/dividend-adjusted series: the
    per-day log returns must agree on the recent shared history even when a
    corporate action rescales every price level."""
    if old is None or not len(old) or close_col not in old.columns:
        return []
    if close_col not in new.columns:
        return [f"close column '{close_col}' missing"]
    a = np.log(pd.to_numeric(new[close_col], errors="coerce")).diff()
    b = np.log(pd.to_numeric(old[close_col], errors="coerce")).diff()
    shared = a.dropna().index.intersection(b.dropna().index)[-window:]
    if len(shared) < 10:
        return []
    bad = int(((a.loc[shared] - b.loc[shared]).abs() > tol).sum())
    if bad > max_bad:
        return [f"{bad}/{len(shared)} recent daily returns disagree with HEAD "
                f"(tolerance {tol})"]
    return []


def check_no_new_jumps(new: pd.DataFrame, old: pd.DataFrame, close_col: str,
                       max_log: float = MAX_1D_LOG_MOVE) -> list[str]:
    """Errors on an implausible single-day move that is NEW to this refresh.

    Moves already committed at HEAD are left alone — they have been seen, and
    blocking them forever would wedge every future pull — but a bar that only
    appears now is the signature of a corrupted splice, and it is exactly what
    silently reached production in the MSTU case."""
    if close_col not in new.columns:
        return [f"close column '{close_col}' missing"]
    px = pd.to_numeric(new[close_col], errors="coerce")
    r = np.log(px / px.shift(1)).dropna()
    bad = r[r.abs() > max_log]
    if old is not None and close_col in old.columns:
        po = pd.to_numeric(old[close_col], errors="coerce")
        known = np.log(po / po.shift(1)).dropna()
        known = set(known[known.abs() > max_log].index)
        bad = bad[[d not in known for d in bad.index]]
    if len(bad):
        worst = bad.abs().idxmax()
        return [f"{len(bad)} new single-day move(s) over "
                f"{(np.exp(max_log)-1)*100:.0f}% (worst "
                f"{(np.exp(bad.loc[worst])-1)*100:+.1f}% on "
                f"{pd.Timestamp(worst).date()}) — corrupted splice?"]
    return []


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

    # 5. Vintage freeze — pinned history must survive the refresh untouched.
    override = os.environ.get("VALIDATE_ALLOW_RESTATEMENT", "")
    if override.strip().lower() in ("1", "true", "yes", "on"):
        print("WARN: VALIDATE_ALLOW_RESTATEMENT set — frozen-history checks "
              "SKIPPED (deliberate re-baseline)")
    else:
        for path, tail in FROZEN_CSVS.items():
            if not Path(path).exists():
                continue
            try:
                cur = pd.read_csv(path, index_col=0, parse_dates=True,
                                  float_precision="round_trip")
            except Exception as exc:
                errs.append(f"{path}: unreadable ({exc})")
                continue
            for e in check_frozen_history(
                    cur, _load_head_version(path), tail,
                    scale_invariant=path in SCALE_INVARIANT_FROZEN):
                errs.append(f"{Path(path).name}: {e}")
        for path, ccol in RETURNS_CSVS.items():
            if not Path(path).exists():
                continue
            try:
                cur = pd.read_csv(path, index_col=0, parse_dates=True,
                                  float_precision="round_trip")
            except Exception as exc:
                errs.append(f"{path}: unreadable ({exc})")
                continue
            for e in check_returns_agreement(cur, _load_head_version(path), ccol):
                errs.append(f"{Path(path).name}: {e}")

    # 6. No fabricated single-day jump (runs even under a deliberate
    #    re-baseline — a re-baseline is never a reason to accept a broken bar).
    for path, ccol in JUMP_CSVS.items():
        if not Path(path).exists():
            continue
        try:
            cur = pd.read_csv(path, index_col=0, parse_dates=True,
                              float_precision="round_trip")
        except Exception as exc:
            errs.append(f"{path}: unreadable ({exc})")
            continue
        for e in check_no_new_jumps(cur, _load_head_version(path), ccol):
            errs.append(f"{Path(path).name}: {e}")

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
