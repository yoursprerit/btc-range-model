"""
Does net global liquidity help the BTC daily high/low model?

The production daily model (``src/pipeline_ct.py``) predicts, from a 12:00-UTC
bar, how far the *next* bar reaches above and below the current close:

    y_hi = (next_high - close) / close      y_lo = (close - next_low) / close

Both targets are positive magnitudes, so the model is mostly forecasting
tomorrow's excursion size with a directional tilt on top. This script asks
whether Fed/ECB net-liquidity features (``src/liquidity_data.py``) add anything
to that, using the production feature matrix and the production learners.

Method
------
* Feature matrix rebuilt exactly as ``pipeline_ct.py`` builds it, from the
  git-tracked ``data/backtest/raw_features_daily.csv``.
* **Walk-forward** evaluation rather than one train/val/test cut: an expanding
  train window, refit every ``--step`` days, always predicting days the model
  has never seen. One 6-month test window on ~1000 rows cannot separate a real
  edge from luck; ~20 folds of paired predictions can.
* Arms are compared **paired on identical rows** and judged with a
  Diebold-Mariano test on the per-day loss differential (Newey-West HAC, so
  serial correlation in the errors does not inflate significance).
* A ``base+noise`` arm — the same number of *random* columns — is the control
  that separates "liquidity carries no signal" from "any 12 extra columns cost
  the model this much". Without it a small loss looks like evidence.

Usage
-----
    python src/research_liquidity_features.py                  # repo dataset
    python src/research_liquidity_features.py --dataset extended
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (GradientBoostingClassifier,
                              GradientBoostingRegressor)
from sklearn.linear_model import HuberRegressor, QuantileRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from liquidity_data import liquidity_features, load_net_liquidity  # noqa: E402

ALPHA_QUANT = 0.70          # matches pipeline_ct.py
MACRO_NAMES = ["spx", "ndx", "vix", "gold", "dxy", "tnx", "eth"]


# ══════════════════════════════════════════════════════════════════════════
# FEATURE MATRIX  (mirrors src/pipeline_ct.py section 4)
# ══════════════════════════════════════════════════════════════════════════
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    c, h, l = df["btc_close"], df["btc_high"], df["btc_low"]
    v = df["btc_volume"]

    nh, nl = h.shift(-1), l.shift(-1)
    y_hi = (nh - c) / c
    y_lo = (c - nl) / c

    ret = np.log(c).diff()
    for k in (1, 3, 5, 7, 14, 30):
        f[f"ret_{k}"] = ret.rolling(k).sum()
    for k in (5, 10, 20, 30):
        f[f"vol_{k}"] = ret.rolling(k).std()

    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()],
                   axis=1).max(axis=1)
    for k in (7, 14, 30):
        f[f"atr_{k}"] = tr.rolling(k).mean() / c
    f["range_today"] = (h - l) / c
    f["range_ma7"] = ((h - l) / c).rolling(7).mean()
    f["range_ma30"] = ((h - l) / c).rolling(30).mean()
    f["range_std30"] = ((h - l) / c).rolling(30).std()

    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    f["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    f["macd"] = macd / c
    f["macd_sig"] = macd.ewm(span=9, adjust=False).mean() / c
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c

    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_width"] = (4 * sd20) / ma20
    f["dist_hi_30"] = c / c.rolling(30).max() - 1
    f["dist_lo_30"] = c / c.rolling(30).min() - 1
    f["dist_hi_90"] = c / c.rolling(90).max() - 1

    f["vol_chg_1"] = np.log(v).diff()
    f["vol_z_20"] = ((np.log(v) - np.log(v).rolling(20).mean())
                     / np.log(v).rolling(20).std())
    f["vol_ma_ratio"] = v / v.rolling(20).mean()

    dow = df.index.dayofweek
    for i in range(6):
        f[f"dow_{i}"] = (dow == i).astype(float)

    for nm in MACRO_NAMES:
        col = f"{nm}_close"
        if col not in df.columns:
            continue
        s = df[col]
        for k in (1, 5, 20):
            f[f"{nm}_ret_{k}"] = np.log(s).diff(k)
        f[f"{nm}_vol_20"] = np.log(s).diff().rolling(20).std()

    for nm in ("spx", "ndx", "gold", "dxy"):
        col = f"{nm}_close"
        if col in df.columns:
            f[f"btc_{nm}_corr_30"] = ret.rolling(30).corr(np.log(df[col]).diff())

    for col in [x for x in df.columns if x.startswith("oc_")]:
        sl = np.log(df[col].astype(float).replace(0, np.nan))
        f[f"{col}_d1"] = sl.diff(1)
        f[f"{col}_d7"] = sl.diff(7)
        f[f"{col}_z30"] = (sl - sl.rolling(30).mean()) / sl.rolling(30).std()

    for col in ("cb_premium", "cb_premium_ma3", "cb_premium_z7"):
        if col in df.columns:
            f[col] = df[col]

    f["y_hi_ema3"] = y_hi.shift(1).ewm(span=3, adjust=False).mean()
    f["y_lo_ema3"] = y_lo.shift(1).ewm(span=3, adjust=False).mean()
    f["y_hi_ema7"] = y_hi.shift(1).ewm(span=7, adjust=False).mean()
    f["y_lo_ema7"] = y_lo.shift(1).ewm(span=7, adjust=False).mean()

    prev_3_hi = h.shift(1).rolling(3).max()
    prev_3_lo = l.shift(1).rolling(3).min()
    f["above_3d_high"] = (c > prev_3_hi).astype(float)
    f["below_3d_low"] = (c < prev_3_lo).astype(float)
    f["bo_strength_up"] = (c / prev_3_hi - 1).clip(lower=0)
    f["bo_strength_dn"] = (1 - c / prev_3_lo).clip(lower=0)
    _yh, _yl = y_hi.shift(1), y_lo.shift(1)
    f["y_hi_surprise"] = _yh - _yh.ewm(span=7, adjust=False).mean()
    f["y_lo_surprise"] = _yl - _yl.ewm(span=7, adjust=False).mean()

    neg_ret = ret.clip(upper=0)
    f["dn_vol_5"] = neg_ret.rolling(5).std()
    f["dn_vol_20"] = neg_ret.rolling(20).std()
    sma50 = c.rolling(50).mean()
    f["below_sma50"] = (c < sma50).astype(float)
    f["below_sma50_5d"] = f["below_sma50"].rolling(5).min().fillna(0)

    f["y_hi"], f["y_lo"] = y_hi, y_lo
    f["close"], f["next_high"], f["next_low"] = c, nh, nl
    return f.replace([np.inf, -np.inf], np.nan)


# ══════════════════════════════════════════════════════════════════════════
# DATASETS
# ══════════════════════════════════════════════════════════════════════════
def load_repo_dataset() -> pd.DataFrame:
    csv = ROOT / "data" / "backtest" / "raw_features_daily.csv"
    return pd.read_csv(csv, index_col=0, parse_dates=True).sort_index()


def fetch_coinbase_hourly(out: Path, start: str = "2018-06-01") -> None:
    """Pull BTC-USD hourly candles from Coinbase (300 per request).

    ~5 MB and fully regenerable, so it is not tracked in git; the extended arm
    calls this on a cache miss.
    """
    import datetime as dt
    import json
    import time
    import urllib.request

    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    cur = dt.datetime.fromisoformat(start).replace(tzinfo=dt.timezone.utc)
    end = dt.datetime.now(dt.timezone.utc)
    step = dt.timedelta(hours=300)
    rows = []
    print(f">>> Fetching Coinbase hourly {start} → today …")
    while cur < end:
        hi = min(cur + step, end)
        q = f"{url}?granularity=3600&start={cur.isoformat()}&end={hi.isoformat()}"
        for attempt in range(5):
            try:
                req = urllib.request.Request(q, headers={"User-Agent": "research/1.0"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    rows.extend(json.load(r))
                break
            except Exception:
                if attempt == 4:
                    print(f"   ! gave up on {cur.date()}")
                else:
                    time.sleep(2 ** attempt)
        cur = hi
        time.sleep(0.16)
    df = pd.DataFrame(rows, columns=["ts", "low", "high", "open", "close", "volume"])
    df = df.drop_duplicates(subset="ts").sort_values("ts")
    df["timestamp_utc"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.set_index("timestamp_utc")[["open", "high", "low", "close", "volume"]].to_csv(out)
    print(f">>> Saved {out}  ({len(df)} hourly bars)")


def load_extended_dataset(cache: Path) -> pd.DataFrame:
    """Longer history: Coinbase 12:00-UTC BTC bars + FRED daily macro.

    Yahoo is not reachable from this environment, so the cross-asset block is
    rebuilt from FRED equivalents. It is close to, but not identical with, the
    production macro set — this arm is a robustness check over a sample that
    includes the 2020-21 QE surge and the 2022-23 QT drain, not a restatement
    of the production model.
    """
    from liquidity_data import _fetch_fred

    hourly_csv = cache / "cb_hourly.csv"
    if not hourly_csv.exists():
        fetch_coinbase_hourly(hourly_csv)
    hr = pd.read_csv(hourly_csv, index_col=0, parse_dates=True)
    hr.index = pd.DatetimeIndex(hr.index).tz_convert("UTC").tz_localize(None)
    hr = hr.sort_index()
    # Same bucketing as pipeline_ct.rebucket_12utc: bar D holds the hours whose
    # timestamp minus 12h lands on calendar day D, so bar D spans D 12:00 →
    # D+1 12:00 and is *labelled with its start date*. Incomplete bars are
    # dropped, again matching production.
    g = hr.groupby((hr.index - pd.Timedelta(hours=12)).normalize())
    bars = g.agg(btc_high=("high", "max"), btc_low=("low", "min"),
                 btc_close=("close", "last"), btc_volume=("volume", "sum"),
                 n_hours=("close", "size"))
    bars = bars[bars["n_hours"] == 24].drop(columns="n_hours")

    # Calendar-date alignment, as in pipeline_ct section 3: bar D takes date D's
    # daily close, which prints ~21:00 UTC on D — inside bar D and therefore
    # known before the model scores bar D+1.
    fred_macro = {"spx": "SP500", "ndx": "NASDAQCOM", "vix": "VIXCLS",
                  "dxy": "DTWEXBGS", "tnx": "DGS10"}
    for nm, sid in fred_macro.items():
        ser = _fetch_fred(sid, cache)
        bars[f"{nm}_close"] = ser.reindex(bars.index.union(ser.index)).ffill(
        ).reindex(bars.index)
    return bars.ffill(limit=5).dropna(subset=["btc_close"])


# ══════════════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════════════
def make_regressors(fast: bool):
    n_est = 400 if fast else 1500
    lr = 0.0375 if fast else 0.01
    mk = lambda m: Pipeline([("sc", StandardScaler()), ("m", m)])
    return {
        "huber": mk(HuberRegressor(max_iter=500, alpha=0.001)),
        "quant_lin": mk(QuantileRegressor(quantile=ALPHA_QUANT, alpha=0.001,
                                          solver="highs")),
        "gbm_quant": mk(GradientBoostingRegressor(
            loss="quantile", alpha=ALPHA_QUANT, n_estimators=n_est,
            max_depth=3, learning_rate=lr, subsample=0.8, random_state=42)),
    }


def make_classifier(fast: bool):
    n_est = 250 if fast else 500
    lr = 0.04 if fast else 0.02
    return Pipeline([("sc", StandardScaler()),
                     ("m", GradientBoostingClassifier(
                         n_estimators=n_est, max_depth=3, learning_rate=lr,
                         subsample=0.8, random_state=42))])


def walk_forward(data: pd.DataFrame, feat_cols: list[str], *, min_train: int,
                 step: int, fast: bool, with_direction: bool = True
                 ) -> pd.DataFrame:
    """Expanding-window out-of-sample predictions, refit every ``step`` days.

    A 1-day embargo between train end and the first scored row keeps the
    shift(-1) target out of the training window.
    """
    X = data[feat_cols]
    yhi, ylo = data["y_hi"], data["y_lo"]
    out = []
    n = len(data)
    for fold, start in enumerate(range(min_train, n, step)):
        stop = min(start + step, n)
        tr = slice(0, start - 1)          # -1 = embargo for the 1-day target
        te = slice(start, stop)
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        if len(X_te) == 0:
            continue
        ph, pl = [], []
        for ctor in make_regressors(fast).values():
            mh = clone(ctor).fit(X_tr, yhi.iloc[tr])
            ml = clone(ctor).fit(X_tr, ylo.iloc[tr])
            ph.append(mh.predict(X_te))
            pl.append(ml.predict(X_te))
        rec = pd.DataFrame({
            "pred_hi": np.mean(ph, axis=0),
            "pred_lo": np.mean(pl, axis=0),
            "mu_hi": yhi.iloc[tr].mean(),
            "mu_lo": ylo.iloc[tr].mean(),
        }, index=X_te.index)
        rec["fold"] = fold
        if with_direction:
            d_tr = (yhi.iloc[tr] - ylo.iloc[tr]) / 2
            clf = make_classifier(fast).fit(X_tr, (d_tr > 0).astype(int))
            rec["p_bull"] = clf.predict_proba(X_te)[:, 1]
        out.append(rec)
    res = pd.concat(out)
    for col in ("y_hi", "y_lo", "close", "next_high", "next_low"):
        res[col] = data[col].reindex(res.index)
    return res


# ══════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════
def score(res: pd.DataFrame) -> dict:
    ph = np.clip(res["pred_hi"].values, 0, None)
    pl = np.clip(res["pred_lo"].values, 0, None)
    pred_h = res["close"].values * (1 + ph)
    pred_l = res["close"].values * (1 - pl)
    d_act = (res["y_hi"] - res["y_lo"]).values / 2
    d_pred = (res["pred_hi"] - res["pred_lo"]).values / 2
    m = {
        "MAE_hi_bp": np.abs(res["pred_hi"] - res["y_hi"]).mean() * 1e4,
        "MAE_lo_bp": np.abs(res["pred_lo"] - res["y_lo"]).mean() * 1e4,
        "MAPE_H": (np.abs(pred_h - res["next_high"]) / res["next_high"]).mean() * 100,
        "MAPE_L": (np.abs(pred_l - res["next_low"]) / res["next_low"]).mean() * 100,
        "dir_hit": (np.sign(d_pred) == np.sign(d_act)).mean() * 100,
        "n": len(res),
    }
    if "p_bull" in res:
        m["clf_acc"] = (((res["p_bull"] > 0.5).astype(int)
                         == (d_act > 0).astype(int)).mean() * 100)
    return m


def alpha_blend(res: pd.DataFrame, grid=np.linspace(0, 1, 21)) -> pd.DataFrame:
    """Apply the pipeline's climatology blend, calibrated out-of-sample.

    ``pipeline_ct.py`` shrinks the ensemble toward the training mean by a factor
    α picked on a validation split. Replicating that here matters: the shrink is
    what the production model actually ships, and it damps whatever a new
    feature block does — good or bad. α for fold *i* is tuned only on the
    realised out-of-sample rows of folds < i, so no fold sees its own answer.
    The first fold has nothing to tune on and passes through at α = 1.
    """
    out = res.copy()
    ph, pl = res["pred_hi"].copy(), res["pred_lo"].copy()
    for fold in sorted(res["fold"].unique()):
        prior = res[res["fold"] < fold]
        cur = res["fold"] == fold
        if len(prior) < 40:
            continue
        best, best_a = np.inf, 1.0
        for a in grid:
            bh = a * prior["pred_hi"] + (1 - a) * prior["mu_hi"]
            bl = a * prior["pred_lo"] + (1 - a) * prior["mu_lo"]
            err = (np.abs(bh - prior["y_hi"]).mean()
                   + np.abs(bl - prior["y_lo"]).mean())
            if err < best:
                best, best_a = err, a
        ph[cur] = best_a * res.loc[cur, "pred_hi"] + (1 - best_a) * res.loc[cur, "mu_hi"]
        pl[cur] = best_a * res.loc[cur, "pred_lo"] + (1 - best_a) * res.loc[cur, "mu_lo"]
    out["pred_hi"], out["pred_lo"] = ph, pl
    return out


def per_day_loss(res: pd.DataFrame) -> np.ndarray:
    """Mean absolute error on the two targets, per day (the DM loss)."""
    return (np.abs(res["pred_hi"] - res["y_hi"]).values
            + np.abs(res["pred_lo"] - res["y_lo"]).values) / 2


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, lag: int = 5):
    """DM statistic for loss_a - loss_b with a Newey-West HAC variance.

    Negative t = arm A has the lower loss. Returns (mean_diff, t, p).
    """
    d = loss_a - loss_b
    n = len(d)
    dbar = d.mean()
    dc = d - dbar
    gamma0 = (dc @ dc) / n
    var = gamma0
    for k in range(1, lag + 1):
        gk = (dc[k:] @ dc[:-k]) / n
        var += 2 * (1 - k / (lag + 1)) * gk
    var = max(var, 1e-24)
    t = dbar / np.sqrt(var / n)
    # normal approximation is fine at these sample sizes
    from math import erfc, sqrt
    p = erfc(abs(t) / sqrt(2))
    return dbar, t, p


def block_bootstrap_win_rate(loss_a, loss_b, block=21, n_boot=2000, seed=0):
    """Share of block-bootstrap resamples in which arm A has the lower loss."""
    rng = np.random.default_rng(seed)
    d = loss_a - loss_b
    n = len(d)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, max(n - block, 1), size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)
    idx = np.clip(idx[:, :n], 0, n - 1)
    return float((d[idx].mean(axis=1) < 0).mean())


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / den) if den > 0 else 0.0


def hac_t(x: np.ndarray, y: np.ndarray, lag: int = 10) -> float:
    """t-stat of the slope in y ~ x with Newey-West errors (both standardised)."""
    x = (x - x.mean()) / (x.std() or 1)
    y = (y - y.mean()) / (y.std() or 1)
    n = len(x)
    beta = (x @ y) / (x @ x)
    e = y - beta * x
    u = x * e
    uc = u - u.mean()
    var = (uc @ uc) / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * ((uc[k:] @ uc[:-k]) / n)
    se = np.sqrt(max(var, 1e-24) * n) / (x @ x)
    return float(beta / se) if se > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["repo", "extended"], default="repo")
    ap.add_argument("--cache", default=str(ROOT / "data" / "liquidity"))
    ap.add_argument("--min-train", type=int, default=500)
    ap.add_argument("--step", type=int, default=21)
    ap.add_argument("--fast", action="store_true",
                    help="smaller GBMs — for iterating, not for the headline")
    ap.add_argument("--safety-lag", type=int, default=1,
                    help="extra days of publication lag on every FRED series")
    args = ap.parse_args()
    cache = Path(args.cache)

    print("=" * 78)
    print(f"  NET LIQUIDITY ABLATION — BTC daily high/low  [{args.dataset}]")
    print("=" * 78)

    raw = (load_repo_dataset() if args.dataset == "repo"
           else load_extended_dataset(cache))
    print(f"\nRaw bars: {raw.shape}  {raw.index.min().date()} → {raw.index.max().date()}")

    liq = load_net_liquidity(start="2017-01-01", cache_dir=cache,
                             safety_lag_days=args.safety_lag)
    print(f"Liquidity panel: {liq.shape}  {liq.index.min().date()} → "
          f"{liq.index.max().date()}   net_liq now = "
          f"${liq['net_liq'].iloc[-1] / 1e6:.2f}T")

    feats = build_features(raw)
    lf = liquidity_features(liq, feats.index)
    liq_cols = list(lf.columns)
    data = feats.join(lf).dropna()
    meta = ["y_hi", "y_lo", "close", "next_high", "next_low"]
    base_cols = [c for c in feats.columns if c not in meta]
    print(f"Model matrix: {data.shape}  {data.index.min().date()} → "
          f"{data.index.max().date()}  ({len(base_cols)} base + "
          f"{len(liq_cols)} liquidity features)")

    # ── univariate diagnostics ────────────────────────────────────────────
    print("\n" + "-" * 78)
    print("  1. Does liquidity co-move with tomorrow's excursion at all?")
    print("-" * 78)
    rng_t = (data["y_hi"] + data["y_lo"]).values     # tomorrow's total range
    asym = (data["y_hi"] - data["y_lo"]).values      # tomorrow's tilt
    print(f"{'feature':<16} {'ρ(range)':>10} {'t_HAC':>8} {'ρ(tilt)':>10} {'t_HAC':>8}")
    for c in liq_cols:
        x = data[c].values
        print(f"{c:<16} {spearman(x, rng_t):>10.3f} {hac_t(x, rng_t):>8.2f} "
              f"{spearman(x, asym):>10.3f} {hac_t(x, asym):>8.2f}")

    # a same-length block of random columns is the yardstick for "how big does
    # a correlation get by chance in this sample"
    rng = np.random.default_rng(7)
    noise = pd.DataFrame(
        rng.standard_normal((len(data), len(liq_cols))),
        index=data.index, columns=[f"noise_{i}" for i in range(len(liq_cols))])
    # give the noise the same persistence as the liquidity features so the
    # comparison is not rigged by autocorrelation
    noise = noise.ewm(span=20).mean()
    noise_cols = list(noise.columns)
    data = data.join(noise)
    nz_r = [abs(spearman(data[c].values, rng_t)) for c in noise_cols]
    print(f"\n  |ρ| vs range — liquidity: max "
          f"{max(abs(spearman(data[c].values, rng_t)) for c in liq_cols):.3f}"
          f"   persistence-matched noise: max {max(nz_r):.3f}")

    # ── is the relationship even stable? ─────────────────────────────────
    print("\n" + "-" * 78)
    print("  1b. Year-by-year stability of the liquidity/range relationship")
    print("-" * 78)
    years = sorted(data.index.year.unique())
    keep = [y for y in years if (data.index.year == y).sum() >= 60]
    hdr = "feature".ljust(16) + "".join(f"{y:>7}" for y in keep) + "  flips"
    print(hdr)
    rng_s = pd.Series(rng_t, index=data.index)
    flips = []
    for c in liq_cols + ["atr_14", "range_ma7", "vol_20"]:
        row, signs = [], []
        for y in keep:
            m = data.index.year == y
            r = spearman(data.loc[m, c].values, rng_s[m].values)
            row.append(r)
            signs.append(np.sign(r))
        nf = sum(1 for a, b in zip(signs, signs[1:]) if a != b)
        if c in liq_cols:
            flips.append(nf)
        else:
            print("  (established range predictor, for scale)"
                  if c == "atr_14" else "", end="")
        print(c.ljust(16) + "".join(f"{v:>7.2f}" for v in row) + f"{nf:>7}")
    print(f"\n  liquidity block: {np.mean(flips):.1f} sign flips on average out of "
          f"{len(keep) - 1} year transitions — a predictor whose sign is not "
          f"stable\n  across regimes cannot be relied on out of sample.")

    # ── walk-forward arms ────────────────────────────────────────────────
    arms = {
        "base": base_cols,
        "base+liq": base_cols + liq_cols,
        "base+liq_core": base_cols + [c for c in liq_cols
                                      if c in ("liq_d7", "liq_d28", "liq_d91",
                                               "liq_z364")],
        "base+noise": base_cols + noise_cols,
        "liq_only": liq_cols,
    }
    print("\n" + "-" * 78)
    print(f"  2. Walk-forward ablation  (min_train={args.min_train}, "
          f"step={args.step}, {'FAST' if args.fast else 'production'} learners)")
    print("-" * 78)

    results, losses = {}, {}
    blended, bl_losses = {}, {}
    for name, cols in arms.items():
        res = walk_forward(data, cols, min_train=args.min_train, step=args.step,
                           fast=args.fast)
        results[name] = res
        losses[name] = per_day_loss(res)
        bres = alpha_blend(res)
        blended[name] = bres
        bl_losses[name] = per_day_loss(bres)
        s, bs = score(res), score(bres)
        print(f"  {name:<14} n={s['n']:<4} MAE_hi={s['MAE_hi_bp']:6.1f}bp  "
              f"MAE_lo={s['MAE_lo_bp']:6.1f}bp  MAPE_H={s['MAPE_H']:.3f}%  "
              f"MAPE_L={s['MAPE_L']:.3f}%  dir={s['dir_hit']:.1f}%  "
              f"clf={s.get('clf_acc', float('nan')):.1f}%", flush=True)
        print(f"  {'  └ α-blended':<14}       MAE_hi={bs['MAE_hi_bp']:6.1f}bp  "
              f"MAE_lo={bs['MAE_lo_bp']:6.1f}bp  MAPE_H={bs['MAPE_H']:.3f}%  "
              f"MAPE_L={bs['MAPE_L']:.3f}%", flush=True)

    # climatology reference — the train-mean forecast the pipeline blends toward
    ref = results["base"]
    clim_loss = (np.abs(ref["mu_hi"] - ref["y_hi"]).values
                 + np.abs(ref["mu_lo"] - ref["y_lo"]).values) / 2
    print(f"  {'climatology':<14} n={len(ref):<4} "
          f"MAE_hi={np.abs(ref['mu_hi'] - ref['y_hi']).mean() * 1e4:6.1f}bp  "
          f"MAE_lo={np.abs(ref['mu_lo'] - ref['y_lo']).mean() * 1e4:6.1f}bp")

    print("\n" + "-" * 78)
    print("  3. Paired tests vs `base`   (negative t = the arm beats base)")
    print("-" * 78)
    print(f"{'arm':<14} {'Δloss(bp)':>11} {'DM t':>8} {'p':>8} {'boot win%':>10}")
    for name in ("base+liq", "base+liq_core", "base+noise", "liq_only"):
        dbar, t, p = diebold_mariano(losses[name], losses["base"])
        w = block_bootstrap_win_rate(losses[name], losses["base"])
        print(f"{name:<14} {dbar * 1e4:>11.2f} {t:>8.2f} {p:>8.3f} {w * 100:>9.1f}%")
    dbar, t, p = diebold_mariano(losses["base"], clim_loss)
    print(f"{'base vs clim':<14} {dbar * 1e4:>11.2f} {t:>8.2f} {p:>8.3f}")

    print("\n  same tests after the production α-blend (what actually ships):")
    print(f"{'arm':<14} {'Δloss(bp)':>11} {'DM t':>8} {'p':>8} {'boot win%':>10}")
    for name in ("base+liq", "base+liq_core", "base+noise", "liq_only"):
        dbar, t, p = diebold_mariano(bl_losses[name], bl_losses["base"])
        w = block_bootstrap_win_rate(bl_losses[name], bl_losses["base"])
        print(f"{name:<14} {dbar * 1e4:>11.2f} {t:>8.2f} {p:>8.3f} {w * 100:>9.1f}%")

    # ── where liquidity ranks when the model is allowed to use it ────────
    print("\n" + "-" * 78)
    print("  4. GBM importance rank of the liquidity block (full-sample fit)")
    print("-" * 78)
    cols = base_cols + liq_cols + noise_cols
    gb = GradientBoostingRegressor(loss="quantile", alpha=ALPHA_QUANT,
                                   n_estimators=400, max_depth=3,
                                   learning_rate=0.0375, subsample=0.8,
                                   random_state=42)
    gb.fit(data[cols], data["y_hi"] + data["y_lo"])
    imp = pd.Series(gb.feature_importances_, index=cols).sort_values(ascending=False)
    ranks = {c: i + 1 for i, c in enumerate(imp.index)}
    print("  top 12 overall: " + ", ".join(imp.index[:12]))
    liq_ranks = sorted(ranks[c] for c in liq_cols)
    nz_ranks = sorted(ranks[c] for c in noise_cols)
    print(f"  liquidity ranks (of {len(cols)}): {liq_ranks}")
    print(f"  noise     ranks (of {len(cols)}): {nz_ranks}")
    print(f"  median rank — liquidity {np.median(liq_ranks):.0f}  "
          f"vs noise {np.median(nz_ranks):.0f}")
    print(f"  summed importance — liquidity {imp[liq_cols].sum():.4f}  "
          f"vs noise {imp[noise_cols].sum():.4f}")


if __name__ == "__main__":
    main()
