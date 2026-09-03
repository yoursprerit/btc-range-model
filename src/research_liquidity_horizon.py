"""
At what forecast horizon (if any) does net liquidity start to pay?

The daily high/low model looks one bar ahead. Net liquidity moves on a weekly
release cycle and its economic story — reserves crowding into risk assets — is
a multi-week one, so a null result at 1 day says nothing about 7 or 14 days.
The repo already ships 7-day and 14-day cone models, so the question of which
horizon liquidity belongs to is a practical one.

This sweeps h = 1, 5, 10, 20, 30 bars and, at each h, compares walk-forward
out-of-sample skill with and without the liquidity block on two targets:

    range_h  = (max high over h) / close - (min low over h) / close   (cone width)
    ret_h    = log return over h bars                                 (direction)

A persistence-matched random block is carried through every horizon as the
control, so "liquidity helps at h=20" only counts if it beats noise at h=20.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from liquidity_data import liquidity_features, load_net_liquidity   # noqa: E402
from research_liquidity_features import (build_features, diebold_mariano,  # noqa: E402
                                         load_extended_dataset,
                                         load_repo_dataset)


def horizon_targets(raw: pd.DataFrame, h: int) -> pd.DataFrame:
    c, hi, lo = raw["btc_close"], raw["btc_high"], raw["btc_low"]
    fwd_hi = hi.shift(-1).rolling(h).max().shift(-(h - 1))
    fwd_lo = lo.shift(-1).rolling(h).min().shift(-(h - 1))
    return pd.DataFrame({
        "range_h": (fwd_hi - fwd_lo) / c,
        "ret_h": np.log(c.shift(-h) / c),
    }, index=raw.index)


def wf_predict(data, cols, target, *, min_train, step, embargo):
    """Expanding-window OOS predictions for one target column."""
    mdl = Pipeline([("sc", StandardScaler()),
                    ("m", GradientBoostingRegressor(
                        n_estimators=300, max_depth=3, learning_rate=0.04,
                        subsample=0.8, random_state=42))])
    X, y = data[cols], data[target]
    preds, n = [], len(data)
    for start in range(min_train, n, step):
        stop = min(start + step, n)
        tr = slice(0, max(start - embargo, 1))
        m = clone(mdl).fit(X.iloc[tr], y.iloc[tr])
        preds.append(pd.Series(m.predict(X.iloc[start:stop]),
                               index=X.index[start:stop]))
    return pd.concat(preds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["repo", "extended"], default="repo")
    ap.add_argument("--cache", default=str(ROOT / "data" / "liquidity"))
    ap.add_argument("--min-train", type=int, default=500)
    ap.add_argument("--step", type=int, default=21)
    args = ap.parse_args()
    cache = Path(args.cache)

    raw = (load_repo_dataset() if args.dataset == "repo"
           else load_extended_dataset(cache))
    liq = load_net_liquidity(start="2017-01-01", cache_dir=cache)
    feats = build_features(raw)
    lf = liquidity_features(liq, feats.index)
    liq_cols = list(lf.columns)

    rng = np.random.default_rng(7)
    noise = pd.DataFrame(rng.standard_normal((len(feats), len(liq_cols))),
                         index=feats.index,
                         columns=[f"noise_{i}" for i in range(len(liq_cols))]
                         ).ewm(span=20).mean()
    noise_cols = list(noise.columns)

    meta = ["y_hi", "y_lo", "close", "next_high", "next_low"]
    base_cols = [c for c in feats.columns if c not in meta]

    print("=" * 78)
    print(f"  LIQUIDITY VALUE BY FORECAST HORIZON  [{args.dataset}]")
    print("=" * 78)
    print(f"  {len(base_cols)} base + {len(liq_cols)} liquidity features; "
          f"walk-forward min_train={args.min_train} step={args.step}")

    for target in ("range_h", "ret_h"):
        print(f"\n  target = {target}")
        print(f"  {'h':>3} {'n':>5} {'MAE base':>10} {'+liq':>10} {'+noise':>10} "
              f"{'Δliq%':>8} {'DM t':>7} {'p':>7}")
        for h in (1, 5, 10, 20, 30):
            tg = horizon_targets(raw, h)
            data = feats.join(lf).join(noise).join(tg).dropna()
            if len(data) < args.min_train + 40:
                print(f"  {h:>3}  (too few rows: {len(data)})")
                continue
            out = {}
            for name, cols in (("base", base_cols),
                               ("liq", base_cols + liq_cols),
                               ("noise", base_cols + noise_cols)):
                p = wf_predict(data, cols, target, min_train=args.min_train,
                               step=args.step, embargo=h)
                out[name] = np.abs(p - data[target].reindex(p.index)).values
            dbar, t, pv = diebold_mariano(out["liq"], out["base"])
            d_pct = (out["liq"].mean() / out["base"].mean() - 1) * 100
            print(f"  {h:>3} {len(out['base']):>5} {out['base'].mean():>10.5f} "
                  f"{out['liq'].mean():>10.5f} {out['noise'].mean():>10.5f} "
                  f"{d_pct:>+7.2f}% {t:>7.2f} {pv:>7.3f}", flush=True)


if __name__ == "__main__":
    main()
