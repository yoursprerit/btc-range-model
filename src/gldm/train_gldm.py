"""Train the full GLDM model suite.

Five model TYPES, mirroring the BTC application but built on GLDM (gold) data
with gold-appropriate features (see app/gldm_core.py):

  1. hourly    → next-hour close (ridge on log-returns) + 95% CI
  2. daily_hl  → next daily High & Low (ratio-to-prior-close) + residual bands
  3. 7d_cone   → 7-trading-day close cone (central path + quantile bands)
  4. 14d_cone  → 14-trading-day close cone
  5. 3class    → next-day type (Trend Up / Chop / Trend Down)

Every artefact is written to models/gldm/*.joblib with a schema the GLDM app
and backtest consume directly.

Run:
    python src/gldm/train_gldm.py            # fetch fresh, train all
    python src/gldm/train_gldm.py --cached   # use data/gldm/*.csv snapshots
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (mean_absolute_error, r2_score, accuracy_score,
                             confusion_matrix)

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "app"))
sys.path.insert(0, str(_ROOT))
import gldm_core as gc  # noqa: E402


# ── helpers ──────────────────────────────────────────────────────────────
def _clean_features(feat: pd.DataFrame, max_nan_frac: float = 0.35) -> list[str]:
    """Keep feature columns whose NaN fraction is below the threshold — this
    drops the sparse intraday macro series (DXY/VIX hourly) automatically while
    retaining GLDM's own reliably-present technicals."""
    frac = feat.isna().mean()
    return [c for c in feat.columns if frac[c] <= max_nan_frac]


def _temporal_split(idx: pd.DatetimeIndex, test_frac=0.20, val_frac=0.15):
    n = len(idx)
    test_start = idx[int(n * (1 - test_frac))]
    val_start = idx[int(n * (1 - test_frac - val_frac))]
    return val_start, test_start


def _ridge():
    return Pipeline([("sc", StandardScaler()),
                     ("m", RidgeCV(alphas=np.logspace(-3, 3, 13)))])


def _print_head(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


# ══════════════════════════════════════════════════════════════════════════
# 1. HOURLY next-close
# ══════════════════════════════════════════════════════════════════════════
def train_hourly(hourly: pd.DataFrame):
    _print_head("1/5  HOURLY next-close (ridge on log-returns)")
    feat = gc.build_hourly_features(hourly)
    c = hourly["gldm_close"]
    rt = np.log(c).diff()
    feat_cols = _clean_features(feat)
    data = feat[feat_cols].copy()
    data["y_ret"] = rt.shift(-1)
    data["close"] = c
    data["next_close"] = c.shift(-1)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    val_start, test_start = _temporal_split(data.index)
    train = data.loc[:val_start].iloc[:-1]
    val = data.loc[val_start:test_start].iloc[:-1]
    test = data.loc[test_start:]
    Xtr, ytr = train[feat_cols], train["y_ret"]
    Xva, yva = val[feat_cols], val["y_ret"]
    Xte, yte = test[feat_cols], test["y_ret"]

    m = _ridge().fit(Xtr, ytr)
    pv = m.predict(Xva); pt = m.predict(Xte)
    sigma = float(np.std(yva.values - pv))

    def _metrics(y, close, nxt, pred):
        pc = close * np.exp(pred)
        rel = np.abs(pc - nxt) / nxt
        return dict(MAPE_pct=float(rel.mean() * 100),
                    hit1_pct=float((rel <= 0.01).mean() * 100),
                    hit05_pct=float((rel <= 0.005).mean() * 100),
                    dir_acc_pct=float(np.mean(np.sign(pred) == np.sign(y)) * 100),
                    R2=float(r2_score(y, pred)))
    mt = _metrics(yte.values, test["close"].values, test["next_close"].values, pt)
    print(f"  n_train={len(train)} n_test={len(test)}  feats={len(feat_cols)}")
    print(f"  TEST  MAPE={mt['MAPE_pct']:.3f}%  dir_acc={mt['dir_acc_pct']:.1f}%  "
          f"R2={mt['R2']:.3f}  95%CI=±{1.96*sigma*100:.2f}%")

    joblib.dump(dict(
        model=m, sigma=sigma, feat_cols=feat_cols, best_name="ridge",
        train_start=str(train.index.min()), train_end=str(train.index.max()),
        test_start=str(test.index.min()), test_end=str(test.index.max()),
        metrics_test=mt, asset="GLDM", horizon="1h",
    ), gc.HOURLY_MODEL_GLDM)
    print(f"  saved {gc.HOURLY_MODEL_GLDM.name}")


# ══════════════════════════════════════════════════════════════════════════
# 2. DAILY High/Low
# ══════════════════════════════════════════════════════════════════════════
def train_daily_hl(daily: pd.DataFrame):
    _print_head("2/5  DAILY High/Low (ratio-to-prior-close + residual bands)")
    feat = gc.build_daily_features(daily)
    close = daily["gldm_close"]; high = daily["gldm_high"]; low = daily["gldm_low"]
    prev_c = close.shift(1)
    # targets: next bar's high/low as a ratio to the PRIOR close (as-of close)
    y_hi = (high / prev_c - 1.0)
    y_lo = (low / prev_c - 1.0)
    feat_cols = _clean_features(feat)
    # CAUSAL alignment: shift features one bar forward so the row for bar D
    # pairs feat(D−1) — info through the as-of close — with bar D's high/low.
    # This matches inference (predict_next_daily_hl applies the model to the
    # latest completed bar's features to forecast the NEXT bar). Unshifted,
    # the target bar's own range/return leak into the features.
    data = feat[feat_cols].shift(1).copy()
    data["y_hi"] = y_hi
    data["y_lo"] = y_lo
    data["close_asof"] = prev_c
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    val_start, test_start = _temporal_split(data.index)
    train = data.loc[:test_start].iloc[:-1]
    test = data.loc[test_start:]
    mh = _ridge().fit(train[feat_cols], train["y_hi"])
    ml = _ridge().fit(train[feat_cols], train["y_lo"])
    # residual std on train tail (for the +/- band around the point estimate)
    res_hi = train["y_hi"].values - mh.predict(train[feat_cols])
    res_lo = train["y_lo"].values - ml.predict(train[feat_cols])
    sig_hi = float(np.std(res_hi)); sig_lo = float(np.std(res_lo))
    # ── calibration: center the H/L bands so predicted High/Low straddle the
    #    realised range (raw ridge is biased upward). Stored as $-per-$ ratios
    #    applied on top of the point estimate at inference time.
    raw_ph_tr = train["close_asof"].values * (1 + mh.predict(train[feat_cols]))
    raw_pl_tr = train["close_asof"].values * (1 + ml.predict(train[feat_cols]))
    ah_tr = (train["y_hi"].values + 1) * train["close_asof"].values
    al_tr = (train["y_lo"].values + 1) * train["close_asof"].values
    bias_hi = float(np.mean((ah_tr - raw_ph_tr) / train["close_asof"].values))
    bias_lo = float(np.mean((al_tr - raw_pl_tr) / train["close_asof"].values))

    # test metrics: coverage & mean abs error in %
    ph = mh.predict(test[feat_cols]); pl = ml.predict(test[feat_cols])
    pred_high = test["close_asof"].values * (1 + ph) + bias_hi * test["close_asof"].values
    pred_low = test["close_asof"].values * (1 + pl) + bias_lo * test["close_asof"].values
    ah = (test["y_hi"].values + 1) * test["close_asof"].values
    al = (test["y_lo"].values + 1) * test["close_asof"].values
    mae_hi = float(np.mean(np.abs(pred_high - ah) / ah) * 100)
    mae_lo = float(np.mean(np.abs(pred_low - al) / al) * 100)
    # band coverage: fraction of days where actual H≤predH+band and actual L≥predL-band
    cov = float(np.mean((ah <= pred_high + sig_hi * test["close_asof"].values) &
                        (al >= pred_low - sig_lo * test["close_asof"].values)) * 100)
    print(f"  n_train={len(train)} n_test={len(test)}  feats={len(feat_cols)}")
    print(f"  TEST  MAPE_hi={mae_hi:.3f}%  MAPE_lo={mae_lo:.3f}%  band_cov={cov:.1f}%")

    joblib.dump(dict(
        model_high=mh, model_low=ml, sigma_high=sig_hi, sigma_low=sig_lo,
        bias_high=bias_hi, bias_low=bias_lo,
        feat_cols=feat_cols, asset="GLDM",
        calibration_meta=dict(train_end=str(train.index.max()),
                              test_start=str(test.index.min())),
        metrics_test=dict(mae_hi_pct=mae_hi, mae_lo_pct=mae_lo, band_cov_pct=cov),
    ), gc.DAILY_HL_GLDM)
    print(f"  saved {gc.DAILY_HL_GLDM.name}")


# ══════════════════════════════════════════════════════════════════════════
# 3 & 4. CLOSE CONES (7 / 14 trading days)
# ══════════════════════════════════════════════════════════════════════════
def train_cone(daily: pd.DataFrame, horizon: int, out_path: Path, tag: str):
    _print_head(f"{tag}  {horizon}-day close cone (central path + quantile bands)")
    feat = gc.build_daily_features(daily)
    close = daily["gldm_close"]
    logc = np.log(close)
    y = logc.shift(-horizon) - logc          # H-day forward log return
    feat_cols = _clean_features(feat)
    data = feat[feat_cols].copy()
    data["y"] = y
    data["close"] = close
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    val_start, test_start = _temporal_split(data.index)
    train = data.loc[:test_start].iloc[:-horizon]   # embargo = horizon
    test = data.loc[test_start:]
    m = _ridge().fit(train[feat_cols], train["y"])
    res = train["y"].values - m.predict(train[feat_cols])
    sigma = float(np.std(res))
    # quantile band multipliers (empirical, so bands aren't gaussian-forced)
    q = {int(p * 100): float(np.quantile(res, p)) for p in (0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95)}
    pt = m.predict(test[feat_cols])
    # coverage of the 5-95 band on test
    lo = pt + q[5]; hi = pt + q[95]
    cov = float(np.mean((test["y"].values >= lo) & (test["y"].values <= hi)) * 100)
    r2 = float(r2_score(test["y"].values, pt))
    print(f"  n_train={len(train)} n_test={len(test)}  feats={len(feat_cols)}")
    print(f"  TEST  R2={r2:.3f}  central_MAE={mean_absolute_error(test['y'].values, pt)*100:.2f}%  "
          f"5-95_band_cov={cov:.1f}%  sigma={sigma*100:.2f}%")

    joblib.dump(dict(
        model=m, sigma=sigma, quantiles=q, feat_cols=feat_cols,
        horizon=horizon, asset="GLDM",
        calibration_meta=dict(train_end=str(train.index.max()),
                              test_start=str(test.index.min())),
        metrics_test=dict(R2=r2, band_cov_pct=cov),
    ), out_path)
    print(f"  saved {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════
# 5. 3-CLASS day type
# ══════════════════════════════════════════════════════════════════════════
def train_day_type(daily: pd.DataFrame):
    _print_head("5/5  3-class day type (Trend Up / Chop / Trend Down)")
    feat = gc.build_daily_features(daily)
    close = daily["gldm_close"]
    fwd = np.log(close).shift(-1) - np.log(close)   # next-day return
    label = pd.Series(1, index=close.index)         # 1 = Chop
    label[fwd >= gc.DAY_UP_THRESH] = 2              # 2 = Trend Up
    label[fwd <= gc.DAY_DOWN_THRESH] = 0           # 0 = Trend Down
    feat_cols = _clean_features(feat)
    data = feat[feat_cols].copy()
    data["y"] = label
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data.iloc[:-1]                            # drop last (unknown fwd)
    val_start, test_start = _temporal_split(data.index)
    train = data.loc[:test_start].iloc[:-1]
    test = data.loc[test_start:]
    clf = Pipeline([("sc", StandardScaler()),
                    ("m", LogisticRegression(max_iter=2000, C=0.5,
                                             class_weight="balanced"))])
    clf.fit(train[feat_cols], train["y"].astype(int))
    pred = clf.predict(test[feat_cols])
    acc = float(accuracy_score(test["y"].astype(int), pred) * 100)
    base = float(test["y"].value_counts(normalize=True).max() * 100)
    cm = confusion_matrix(test["y"].astype(int), pred, labels=[0, 1, 2])
    print(f"  n_train={len(train)} n_test={len(test)}  feats={len(feat_cols)}")
    print(f"  TEST  accuracy={acc:.1f}%  (majority-class baseline={base:.1f}%)")
    print(f"  confusion [Down/Chop/Up]:\n{cm}")

    joblib.dump(dict(
        model=clf, classes={0: "Trend Down", 1: "Chop", 2: "Trend Up"},
        feat_cols=feat_cols, asset="GLDM",
        up_thresh=gc.DAY_UP_THRESH, down_thresh=gc.DAY_DOWN_THRESH,
        calibration_meta=dict(train_end=str(train.index.max()),
                              test_start=str(test.index.min())),
        metrics_test=dict(accuracy_pct=acc, baseline_pct=base),
    ), gc.DAY_TYPE_GLDM)
    print(f"  saved {gc.DAY_TYPE_GLDM.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cached", action="store_true",
                    help="use data/gldm/*.csv snapshots instead of fetching")
    args = ap.parse_args()

    if args.cached and gc.DAILY_CACHE_CSV.exists():
        daily = pd.read_csv(gc.DAILY_CACHE_CSV, index_col=0, parse_dates=True)
        hourly = pd.read_csv(gc.HOURLY_CACHE_CSV, index_col=0, parse_dates=True)
    else:
        print("Fetching GLDM + macro data from Yahoo ...")
        daily = gc.fetch_daily(start="2015-01-01")
        hourly = gc.fetch_hourly(range_="730d")
        daily.to_csv(gc.DAILY_CACHE_CSV)
        hourly.to_csv(gc.HOURLY_CACHE_CSV)
    print(f"daily rows={len(daily)}  hourly rows={len(hourly)}  "
          f"(built {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC})")

    train_hourly(hourly)
    train_daily_hl(daily)
    train_cone(daily, 7, gc.CONE_7D_GLDM, "3/5")
    train_cone(daily, 14, gc.CONE_14D_GLDM, "4/5")
    train_day_type(daily)
    print("\nAll GLDM models trained.")


if __name__ == "__main__":
    main()
