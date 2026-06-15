#!/usr/bin/env python3
"""
Pull fresh price data for all backtests (BTC, MSTR, MSTU), validate it,
synthesise MSTU pre-inception prices via OLS, and save versioned CSVs to
data/backtest/ with a manifest.json.

Run manually whenever you want to refresh the dataset:
    python scripts/pull_backtest_data.py

The version badge displayed in the UI reads from data/backtest/manifest.json.
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

REPO_ROOT   = Path(__file__).resolve().parent.parent
DATA_DIR    = REPO_ROOT / "data" / "backtest"
INCEPTION   = pd.Timestamp("2024-09-18")   # MSTU first trading day
FETCH_FROM  = "2023-11-01"                 # 200+ days before earliest backtest start

# ─── helpers ────────────────────────────────────────────────────────────────

def _norm_df(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns (yfinance quirk) and normalise index to UTC-naive dates."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    return df


def _download(ticker: str, start: str, retries: int = 3) -> pd.DataFrame:
    """Download from yfinance with retry."""
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
            df = _norm_df(df)
            if not df.empty:
                return df
        except Exception as exc:
            print(f"  [attempt {attempt}] {ticker}: {exc}")
    raise RuntimeError(f"Failed to download {ticker} after {retries} attempts")


def _checksum(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.to_csv().encode()).hexdigest()[:16]


# ─── quality checks ─────────────────────────────────────────────────────────

def validate(name: str, df: pd.DataFrame, close_col: str = "close",
             min_rows: int = 100) -> bool:
    issues = []

    if len(df) < min_rows:
        issues.append(f"only {len(df)} rows (expected ≥ {min_rows})")

    if close_col not in df.columns:
        issues.append(f"missing column '{close_col}'")
        print(f"  [QC FAIL] {name}: {'; '.join(issues)}")
        return False

    prices = df[close_col].dropna()

    if len(prices) < min_rows:
        issues.append(f"only {len(prices)} non-NaN prices")

    if (prices <= 0).any():
        bad = (prices <= 0).sum()
        issues.append(f"{bad} non-positive price(s)")

    # Flat run: >7 consecutive unchanged closes on trading days (weekends excluded)
    trading = prices.loc[prices.index.dayofweek < 5]
    flat_streak = (trading.diff().abs() < 1e-8).rolling(7).sum()
    if (flat_streak >= 7).any():
        issues.append("suspicious flat run of >7 trading days")

    # Large single-day move (>80%) — likely a data error
    log_r = np.log(prices / prices.shift(1)).dropna()
    if (log_r.abs() > np.log(1.8)).any():
        n_big = (log_r.abs() > np.log(1.8)).sum()
        issues.append(f"{n_big} daily move(s) >80% (possible split/error)")

    # Coverage check
    last = df.index[-1]
    today = pd.Timestamp.today().normalize()
    if (today - last).days > 5:
        issues.append(f"stale — last row is {last.date()} ({(today-last).days}d ago)")

    if issues:
        print(f"  [QC WARN] {name}: {'; '.join(issues)}")
        return False

    print(f"  [QC OK ] {name}: {len(df)} rows  "
          f"{df.index[0].date()} → {df.index[-1].date()}")
    return True


# ─── MSTU OLS synthesis ─────────────────────────────────────────────────────

def synthesise_mstu(mstr_df: pd.DataFrame, mstu_actual: pd.DataFrame) -> pd.Series:
    """
    Synthesise MSTU prices before inception via OLS on MSTR log-returns.

    Returns a pd.Series (index=DatetimeIndex, name='close') covering the full
    range of mstr_df, with synthetic prices before INCEPTION and actual
    post-inception prices after.
    """
    common = mstr_df.index.intersection(mstu_actual.index)
    common = common[common >= INCEPTION]

    beta, alpha = 2.0, -0.0002   # fallback (theoretical 2×)

    if len(common) >= 10:
        mstr_lr = np.log(mstr_df.loc[common, "close"] /
                         mstr_df.loc[common, "close"].shift(1)).dropna()
        mstu_lr = np.log(mstu_actual.loc[common, "close"] /
                         mstu_actual.loc[common, "close"].shift(1)).dropna()
        aligned = mstr_lr.index.intersection(mstu_lr.index)
        x = mstr_lr.loc[aligned].values
        y = mstu_lr.loc[aligned].values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() >= 10:
            coeffs = np.polyfit(x[mask], y[mask], 1)
            beta, alpha = float(coeffs[0]), float(coeffs[1])
            print(f"  OLS  β={beta:.4f}  α={alpha:.6f}  "
                  f"(n={mask.sum()} overlapping bars)")

    # Pre-inception index
    pre_idx = mstr_df.index[mstr_df.index < INCEPTION]
    mstr_pre = mstr_df.loc[pre_idx, "close"]
    mstr_lr_pre = np.log(mstr_pre / mstr_pre.shift(1)).fillna(0.0).values
    syn_lr = beta * mstr_lr_pre + alpha

    # Anchor to first actual MSTU close
    anchor_px = float(
        mstu_actual.loc[mstu_actual.index >= INCEPTION, "close"].iloc[0]
    )
    cum_lr = np.cumsum(syn_lr)
    syn_prices = anchor_px * np.exp(cum_lr - cum_lr[-1])

    syn_series    = pd.Series(syn_prices, index=pre_idx, name="close")
    actual_series = mstu_actual["close"].copy()
    actual_series.name = "close"

    combined = pd.concat([syn_series, actual_series]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined


# ─── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pull_ts = datetime.now(timezone.utc)
    pull_date = pull_ts.strftime("%Y-%m-%d")
    print(f"=== pull_backtest_data.py  ({pull_date} UTC) ===\n")

    # ── download ─────────────────────────────────────────────────────────────
    print("Downloading BTC-USD …")
    btc_raw = _download("BTC-USD", FETCH_FROM)
    btc = btc_raw[["Open", "High", "Low", "Close", "Volume"]].rename(
        columns=str.lower)
    btc = btc.ffill()

    print("Downloading MSTR …")
    mstr_raw = _download("MSTR", FETCH_FROM)
    mstr = mstr_raw[["Open", "High", "Low", "Close", "Volume"]].rename(
        columns=str.lower)
    mstr = mstr.ffill()

    print("Downloading MSTU …")
    mstu_raw = _download("MSTU", str(INCEPTION.date()))
    mstu = mstu_raw[["Open", "High", "Low", "Close", "Volume"]].rename(
        columns=str.lower)
    mstu = mstu.ffill()

    # ── MSTU OLS synthesis ───────────────────────────────────────────────────
    print("\nSynthesising MSTU pre-inception prices (OLS on MSTR) …")
    mstu_syn = synthesise_mstu(mstr, mstu).to_frame("close")

    # ── quality checks ───────────────────────────────────────────────────────
    print("\nQuality checks:")
    btc_ok    = validate("BTC-USD",           btc,      close_col="close")
    mstr_ok   = validate("MSTR",              mstr,     close_col="close")
    mstu_ok   = validate("MSTU (actual)",     mstu,     close_col="close", min_rows=50)
    msyn_ok   = validate("MSTU (synthetic)",  mstu_syn, close_col="close", min_rows=100)

    all_ok = all([btc_ok, mstr_ok, mstu_ok, msyn_ok])
    if not all_ok:
        print("\n⚠  One or more QC checks failed — data saved with warnings.")
    else:
        print("\n✅  All QC checks passed.")

    # ── save CSVs ────────────────────────────────────────────────────────────
    print(f"\nSaving CSVs to {DATA_DIR}/")
    btc.to_csv(      DATA_DIR / "btc_usd_daily.csv")
    mstr.to_csv(     DATA_DIR / "mstr_daily.csv")
    mstu.to_csv(     DATA_DIR / "mstu_daily.csv")
    mstu_syn.to_csv( DATA_DIR / "mstu_synthetic_daily.csv")

    # ── manifest ─────────────────────────────────────────────────────────────
    manifest = {
        "version":       "v1",
        "pull_date":     pull_date,
        "pull_ts_utc":   pull_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_by":  "scripts/pull_backtest_data.py",
        "qc_all_passed": all_ok,
        "datasets": {
            "btc_usd_daily": {
                "file":             "btc_usd_daily.csv",
                "ticker":           "BTC-USD",
                "rows":             len(btc),
                "date_from":        str(btc.index[0].date()),
                "date_to":          str(btc.index[-1].date()),
                "checksum_sha256":  _checksum(btc),
                "qc_passed":        btc_ok,
            },
            "mstr_daily": {
                "file":             "mstr_daily.csv",
                "ticker":           "MSTR",
                "rows":             len(mstr),
                "date_from":        str(mstr.index[0].date()),
                "date_to":          str(mstr.index[-1].date()),
                "checksum_sha256":  _checksum(mstr),
                "qc_passed":        mstr_ok,
            },
            "mstu_daily": {
                "file":             "mstu_daily.csv",
                "ticker":           "MSTU",
                "rows":             len(mstu),
                "date_from":        str(mstu.index[0].date()),
                "date_to":          str(mstu.index[-1].date()),
                "checksum_sha256":  _checksum(mstu),
                "qc_passed":        mstu_ok,
            },
            "mstu_synthetic_daily": {
                "file":             "mstu_synthetic_daily.csv",
                "ticker":           "MSTU (OLS-synthetic)",
                "rows":             len(mstu_syn),
                "date_from":        str(mstu_syn.index[0].date()),
                "date_to":          str(mstu_syn.index[-1].date()),
                "checksum_sha256":  _checksum(mstu_syn),
                "qc_passed":        msyn_ok,
            },
        },
    }

    with open(DATA_DIR / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nmanifest.json written  →  version={manifest['version']}  "
          f"pull_date={manifest['pull_date']}")
    print("\nDone.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
