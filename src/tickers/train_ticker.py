"""Train the full model suite for a generic ticker app.

Config-driven port of ``src/gldm/train_gldm.py``.  Five model types, built on
the ticker's own data + its config-specific macro drivers (see
``app/ticker_core.py``):

  1. hourly    → next-hour close (ridge on log-returns) + 95% CI
  2. daily_hl  → next daily High & Low (ratio-to-prior-close) + residual bands
  3. 7d_cone   → 7-trading-day close cone (central path + quantile bands)
  4. 14d_cone  → 14-trading-day close cone
  5. 3class    → next-day type (Trend Up / Chop / Trend Down)

Artefacts → models/<key>/*.joblib with the schema the ticker app + backtest
consume directly.

Run:
    python src/tickers/train_ticker.py SOXX            # fetch fresh, train all
    python src/tickers/train_ticker.py SOXX --cached   # use cached CSVs
    python src/tickers/train_ticker.py ALL             # every configured ticker
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
import ticker_core as tc  # noqa: E402
from ticker_config import get_config, CONFIGS  # noqa: E402


def _clean_features(feat: pd.DataFrame, max_nan_frac: float = 0.35) -> list[str]:
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


def _head(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


# 1. HOURLY next-close ─────────────────────────────────────────────────────
def train_hourly(cfg, hourly, out_path):
    _head("1/5  HOURLY next-close (ridge on log-returns)")
    feat = tc.build_hourly_features(cfg, hourly)
    c = hourly["px_close"]
    rt = np.log(c).diff()
    feat_cols = _clean_features(feat)
    data = feat[feat_cols].copy()
    data["y_ret"] = rt.shift(-1); data["close"] = c; data["next_close"] = c.shift(-1)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    val_start, test_start = _temporal_split(data.index)
    train = data.loc[:val_start].iloc[:-1]
    val = data.loc[val_start:test_start].iloc[:-1]
    test = data.loc[test_start:]
    m = _ridge().fit(train[feat_cols], train["y_ret"])
    pv = m.predict(val[feat_cols]); pt = m.predict(test[feat_cols])
    sigma = float(np.std(val["y_ret"].values - pv))
    pc = test["close"].values * np.exp(pt)
    rel = np.abs(pc - test["next_close"].values) / test["next_close"].values
    mt = dict(MAPE_pct=float(rel.mean() * 100),
              dir_acc_pct=float(np.mean(np.sign(pt) == np.sign(test["y_ret"].values)) * 100),
              R2=float(r2_score(test["y_ret"].values, pt)))
    print(f"  n_train={len(train)} n_test={len(test)} feats={len(feat_cols)}  "
          f"MAPE={mt['MAPE_pct']:.3f}% dir={mt['dir_acc_pct']:.1f}% 95%CI=±{1.96*sigma*100:.2f}%")
    joblib.dump(dict(model=m, sigma=sigma, feat_cols=feat_cols, best_name="ridge",
                     train_start=str(train.index.min()), train_end=str(train.index.max()),
                     test_start=str(test.index.min()), test_end=str(test.index.max()),
                     metrics_test=mt, asset=cfg.key, horizon="1h"), out_path)
    print(f"  saved {out_path.name}")


# 2. DAILY High/Low ────────────────────────────────────────────────────────
def train_daily_hl(cfg, daily, out_path):
    _head("2/5  DAILY High/Low (ratio-to-prior-close + residual bands)")
    feat = tc.build_daily_features(cfg, daily)
    close = daily["px_close"]; high = daily["px_high"]; low = daily["px_low"]
    prev_c = close.shift(1)
    y_hi = (high / prev_c - 1.0); y_lo = (low / prev_c - 1.0)
    feat_cols = _clean_features(feat)
    # CAUSAL alignment: pair feat(D−1) with bar D's high/low so training matches
    # inference (latest features → NEXT bar's H/L). See train_gldm.train_daily_hl.
    data = feat[feat_cols].shift(1).copy()
    data["y_hi"] = y_hi; data["y_lo"] = y_lo; data["close_asof"] = prev_c
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    _, test_start = _temporal_split(data.index)
    train = data.loc[:test_start].iloc[:-1]; test = data.loc[test_start:]
    mh = _ridge().fit(train[feat_cols], train["y_hi"])
    ml = _ridge().fit(train[feat_cols], train["y_lo"])
    sig_hi = float(np.std(train["y_hi"].values - mh.predict(train[feat_cols])))
    sig_lo = float(np.std(train["y_lo"].values - ml.predict(train[feat_cols])))
    raw_ph_tr = train["close_asof"].values * (1 + mh.predict(train[feat_cols]))
    raw_pl_tr = train["close_asof"].values * (1 + ml.predict(train[feat_cols]))
    ah_tr = (train["y_hi"].values + 1) * train["close_asof"].values
    al_tr = (train["y_lo"].values + 1) * train["close_asof"].values
    bias_hi = float(np.mean((ah_tr - raw_ph_tr) / train["close_asof"].values))
    bias_lo = float(np.mean((al_tr - raw_pl_tr) / train["close_asof"].values))
    ph = mh.predict(test[feat_cols]); pl = ml.predict(test[feat_cols])
    pred_high = test["close_asof"].values * (1 + ph) + bias_hi * test["close_asof"].values
    pred_low = test["close_asof"].values * (1 + pl) + bias_lo * test["close_asof"].values
    ah = (test["y_hi"].values + 1) * test["close_asof"].values
    al = (test["y_lo"].values + 1) * test["close_asof"].values
    mae_hi = float(np.mean(np.abs(pred_high - ah) / ah) * 100)
    mae_lo = float(np.mean(np.abs(pred_low - al) / al) * 100)
    print(f"  n_train={len(train)} n_test={len(test)} feats={len(feat_cols)}  "
          f"MAPE_hi={mae_hi:.3f}% MAPE_lo={mae_lo:.3f}%")
    joblib.dump(dict(model_high=mh, model_low=ml, sigma_high=sig_hi, sigma_low=sig_lo,
                     bias_high=bias_hi, bias_low=bias_lo, feat_cols=feat_cols, asset=cfg.key,
                     calibration_meta=dict(train_end=str(train.index.max()),
                                           test_start=str(test.index.min())),
                     metrics_test=dict(mae_hi_pct=mae_hi, mae_lo_pct=mae_lo)), out_path)
    print(f"  saved {out_path.name}")


# 3 & 4. CLOSE CONES ───────────────────────────────────────────────────────
def train_cone(cfg, daily, horizon, out_path, tag):
    _head(f"{tag}  {horizon}-day close cone (central path + quantile bands)")
    feat = tc.build_daily_features(cfg, daily)
    close = daily["px_close"]; logc = np.log(close)
    y = logc.shift(-horizon) - logc
    feat_cols = _clean_features(feat)
    data = feat[feat_cols].copy(); data["y"] = y; data["close"] = close
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    _, test_start = _temporal_split(data.index)
    train = data.loc[:test_start].iloc[:-horizon]; test = data.loc[test_start:]
    m = _ridge().fit(train[feat_cols], train["y"])
    res = train["y"].values - m.predict(train[feat_cols])
    sigma = float(np.std(res))
    q = {int(p * 100): float(np.quantile(res, p)) for p in (0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95)}
    pt = m.predict(test[feat_cols])
    lo = pt + q[5]; hi = pt + q[95]
    cov = float(np.mean((test["y"].values >= lo) & (test["y"].values <= hi)) * 100)
    r2 = float(r2_score(test["y"].values, pt))
    print(f"  n_train={len(train)} n_test={len(test)}  R2={r2:.3f} band_cov={cov:.1f}% sigma={sigma*100:.2f}%")
    joblib.dump(dict(model=m, sigma=sigma, quantiles=q, feat_cols=feat_cols,
                     horizon=horizon, asset=cfg.key,
                     calibration_meta=dict(train_end=str(train.index.max()),
                                           test_start=str(test.index.min())),
                     metrics_test=dict(R2=r2, band_cov_pct=cov)), out_path)
    print(f"  saved {out_path.name}")


# 5. 3-CLASS day type ──────────────────────────────────────────────────────
def train_day_type(cfg, daily, out_path):
    _head("5/5  3-class day type (Trend Up / Chop / Trend Down)")
    feat = tc.build_daily_features(cfg, daily)
    close = daily["px_close"]
    fwd = np.log(close).shift(-1) - np.log(close)
    label = pd.Series(1, index=close.index)
    label[fwd >= cfg.day_up_thresh] = 2
    label[fwd <= cfg.day_down_thresh] = 0
    feat_cols = _clean_features(feat)
    data = feat[feat_cols].copy(); data["y"] = label
    data = data.replace([np.inf, -np.inf], np.nan).dropna().iloc[:-1]
    _, test_start = _temporal_split(data.index)
    train = data.loc[:test_start].iloc[:-1]; test = data.loc[test_start:]
    clf = Pipeline([("sc", StandardScaler()),
                    ("m", LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced"))])
    clf.fit(train[feat_cols], train["y"].astype(int))
    pred = clf.predict(test[feat_cols])
    acc = float(accuracy_score(test["y"].astype(int), pred) * 100)
    base = float(test["y"].value_counts(normalize=True).max() * 100)
    print(f"  n_train={len(train)} n_test={len(test)}  accuracy={acc:.1f}% (baseline={base:.1f}%)")
    joblib.dump(dict(model=clf, classes={0: "Trend Down", 1: "Chop", 2: "Trend Up"},
                     feat_cols=feat_cols, asset=cfg.key,
                     up_thresh=cfg.day_up_thresh, down_thresh=cfg.day_down_thresh,
                     calibration_meta=dict(train_end=str(train.index.max()),
                                           test_start=str(test.index.min())),
                     metrics_test=dict(accuracy_pct=acc, baseline_pct=base)), out_path)
    print(f"  saved {out_path.name}")


def train_one(key, cached):
    cfg = get_config(key)
    paths = tc.model_paths(cfg); cpaths = tc.cache_paths(cfg)
    print("\n" + "#" * 72 + f"\n#  Training {cfg.key} — {cfg.name}\n" + "#" * 72)
    if cached and cpaths["daily"].exists():
        daily = pd.read_csv(cpaths["daily"], index_col=0, parse_dates=True)
        hourly = (pd.read_csv(cpaths["hourly"], index_col=0, parse_dates=True)
                  if cpaths["hourly"].exists() else pd.DataFrame())
    else:
        print("Fetching data from Yahoo ...")
        daily = tc.fetch_daily(cfg); hourly = tc.fetch_hourly(cfg)
        daily.to_csv(cpaths["daily"])
        if not hourly.empty:
            hourly.to_csv(cpaths["hourly"])
    print(f"daily rows={len(daily)} hourly rows={len(hourly)}  "
          f"(built {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC})")
    if not hourly.empty and len(hourly) > 200:
        train_hourly(cfg, hourly, paths["hourly"])
    else:
        print("  (hourly history too short — skipping hourly model)")
    train_daily_hl(cfg, daily, paths["daily_hl"])
    train_cone(cfg, daily, 7, paths["cone_7d"], "3/5")
    train_cone(cfg, daily, 14, paths["cone_14d"], "4/5")
    train_day_type(cfg, daily, paths["day_type"])
    print(f"\nAll {cfg.key} models trained → {paths['daily_hl'].parent}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker", help="ticker key, or ALL")
    ap.add_argument("--cached", action="store_true")
    args = ap.parse_args()
    keys = list(CONFIGS) if args.ticker.upper() == "ALL" else [args.ticker.upper()]
    for k in keys:
        train_one(k, args.cached)


if __name__ == "__main__":
    main()
