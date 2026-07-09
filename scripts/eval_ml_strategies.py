"""Search a spread of ML / statistical long-flat strategies for the five
MA-mode ETF apps (SOXX, VEGN, GRID, REMX, WGMI) and compare each to the
CURRENT MA filter and to Buy&Hold.

All strategies are long/flat (no leverage, no shorting), execute on the NEXT
bar (decision at close i → position for i+1), and carry the app's own fixed
stop as a uniform risk overlay — so every candidate is apples-to-apples with
the current MA config.  ML models are fit causally:

  * full-OOS window : fit on the pre-OOS window, trade oos_start → now
  * Aug-2025 holdout: fit on data through 2025-08-01, trade 2025-08-01 → now

with the forward-return label fully realised inside the training window (no
look-ahead across the split).
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "app")); sys.path.insert(0, str(_REPO))
import ticker_core as tc                        # noqa
from ticker_config import get_config            # noqa

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.mixture import GaussianMixture

ASSETS = ["SOXX", "VEGN", "GRID", "REMX", "WGMI"]
HOLDOUT = "2025-08-01"
TRADING = 252


def load(cfg):
    df = pd.read_csv(tc.cache_paths(cfg)["daily"], index_col=0, parse_dates=True)
    if "px_close" not in df.columns:
        pref = f"{cfg.primary_symbol.lower()}_"
        df = df.rename(columns={c: "px_" + c[len(pref):]
                                for c in df.columns if c.startswith(pref)})
    return df


# ── generic long/flat simulator (next-bar exec + fixed stop) ──────────────
def sim(price, long_at_close, i0, i1, stop):
    eq = 1.0; rets = []; in_pos = False; entry = np.nan; ntr = 0
    for i in range(i0, i1):
        r = 0.0
        if in_pos:
            r = price[i] / price[i - 1] - 1.0
        rets.append(r); eq *= (1 + r)
        if in_pos:
            if price[i] <= entry * (1 - stop) or not long_at_close[i - 1]:
                in_pos = False; entry = np.nan
        elif long_at_close[i - 1]:
            in_pos = True; entry = price[i]; ntr += 1
    return np.array(rets), ntr


def metrics(rets):
    if len(rets) < 2:
        return dict(ret=0, mdd=0, sharpe=0)
    eq = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(eq)
    mdd = float(np.min(eq / peak - 1))
    sh = float(np.mean(rets) / (np.std(rets) + 1e-12) * np.sqrt(TRADING))
    return dict(ret=float(eq[-1] - 1), mdd=mdd, sharpe=sh)


def bh_metrics(price, i0, i1):
    seg = price[i0:i1]
    rets = np.diff(seg) / seg[:-1]
    return metrics(rets)


# ── signal builders ───────────────────────────────────────────────────────
def rsi(c, n=14):
    d = c.diff(); up = d.clip(lower=0).rolling(n).mean(); dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))


def rule_signals(cfg, daily):
    c = daily["px_close"]
    out = {}
    # current MA filter (baseline)
    out["MA (current)"] = (c > c.rolling(cfg.ma_window).mean()).to_numpy()
    # dual-MA crossover 20/100
    out["Dual-MA 20/100"] = (c.rolling(20).mean() > c.rolling(100).mean()).to_numpy()
    # time-series momentum, 63-day
    out["TSMOM 63d"] = (c / c.shift(63) - 1 > 0).to_numpy()
    # Donchian 20/10 breakout (stateful long/flat)
    hi = c.rolling(20).max().shift(1); lo = c.rolling(10).min().shift(1)
    st = np.zeros(len(c), bool); cur = False
    cv = c.to_numpy(); hv = hi.to_numpy(); lv = lo.to_numpy()
    for i in range(len(cv)):
        if not np.isnan(hv[i]) and cv[i] >= hv[i]:
            cur = True
        elif not np.isnan(lv[i]) and cv[i] <= lv[i]:
            cur = False
        st[i] = cur
    out["Donchian 20/10"] = st
    # MACD histogram > 0
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    out["MACD 12/26/9"] = (macd - macd.ewm(span=9, adjust=False).mean() > 0).to_numpy()
    # MA filter + volatility filter (skip high-vol regime)
    rt = np.log(c).diff(); v20 = rt.rolling(20).std(); vmed = v20.rolling(252).median()
    out["MA + vol-filter"] = ((c > c.rolling(cfg.ma_window).mean()) & (v20 < vmed)).to_numpy()
    return out


def ml_signal(kind, feat, price_s, dates, train_cut, horizon=10):
    """Fit causally on rows whose forward label ends before train_cut; return a
    long_at_close boolean over ALL bars."""
    X = feat.replace([np.inf, -np.inf], np.nan)
    frac = X.isna().mean()
    cols = [c for c in X.columns if frac[c] <= 0.35]
    X = X[cols]
    c = price_s
    i_cut = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(train_cut))))

    if kind == "gmm":                       # 2-state regime on [ret20, vol20]
        rt = np.log(c).diff()
        F = pd.DataFrame({"r20": rt.rolling(20).sum(), "v20": rt.rolling(20).std()}).to_numpy()
        tr = F[:max(i_cut - horizon, 30)]
        tr = tr[~np.isnan(tr).any(axis=1)]
        gm = GaussianMixture(2, covariance_type="full", random_state=7).fit(tr)
        bull = int(np.argmax(gm.means_[:, 0]))     # higher-mean-return state
        Ffill = np.nan_to_num(F)
        lab = gm.predict(Ffill)
        return (lab == bull)

    fwd = c.shift(-horizon) / c - 1.0
    y = (fwd > 0).astype(int)
    tr_end = max(i_cut - horizon, 30)
    idx = np.arange(len(c))
    tr_mask = (idx < tr_end) & X.notna().all(axis=1).to_numpy() & fwd.notna().to_numpy()
    Xtr, ytr = X[tr_mask], y[tr_mask]
    if kind == "logit":
        m = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                      ("m", LogisticRegression(max_iter=1000, C=0.5))]).fit(Xtr, ytr)
        p = m.predict_proba(X)[:, 1]
        return p >= 0.5
    if kind == "hgb":
        m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                           max_iter=300, l2_regularization=1.0,
                                           random_state=7).fit(Xtr, ytr)
        p = m.predict_proba(X)[:, 1]
        return p >= 0.5
    if kind == "ridge_fwd":                 # regress forward 20d return, long if >0
        h = 20; fwd20 = c.shift(-h) / c - 1.0
        trm = (idx < max(i_cut - h, 30)) & X.notna().all(axis=1).to_numpy() & fwd20.notna().to_numpy()
        m = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                      ("m", RidgeCV(alphas=np.logspace(-3, 3, 13)))]).fit(X[trm], fwd20[trm])
        return m.predict(X) > 0
    raise ValueError(kind)


ML_KINDS = [("Logistic (fwd10 P>0)", "logit"), ("HistGradBoost (fwd10)", "hgb"),
            ("Ridge (fwd20 ret>0)", "ridge_fwd"), ("GMM 2-state regime", "gmm")]


def fmt(m):
    return f"{m['ret']*100:+8.1f}% / {m['mdd']*100:6.1f}% / {m['sharpe']:5.2f}"


for key in ASSETS:
    cfg = get_config(key)
    daily = load(cfg)
    feat = tc.build_daily_features(cfg, daily)
    price = daily["px_close"].to_numpy(float)
    dates = daily.index.values
    stop = cfg.fixed_stop
    n = len(price)
    i_oos = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(cfg.oos_start))))
    i_hold = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(HOLDOUT))))

    print(f"\n{'='*78}\n{key}  (current = MA{cfg.ma_window}/{int(stop*100)}%)   "
          f"OOS {cfg.oos_start}→now  |  holdout {HOLDOUT}→now")
    print(f"{'strategy':22s} | {'FULL OOS  ret/MDD/Sharpe':28s} | {'HOLDOUT  ret/MDD/Sharpe':28s}")
    print("-" * 88)

    # Buy&Hold
    print(f"{'Buy & Hold':22s} | {fmt(bh_metrics(price, i_oos, n)):28s} | {fmt(bh_metrics(price, i_hold, n)):28s}")

    rules = rule_signals(cfg, daily)
    for name, sigv in rules.items():
        mf = metrics(sim(price, sigv, i_oos, n, stop)[0])
        mh = metrics(sim(price, sigv, i_hold, n, stop)[0])
        print(f"{name:22s} | {fmt(mf):28s} | {fmt(mh):28s}")

    for name, kind in ML_KINDS:
        sig_full = ml_signal(kind, feat, daily["px_close"], dates, cfg.oos_start)
        sig_hold = ml_signal(kind, feat, daily["px_close"], dates, HOLDOUT)
        mf = metrics(sim(price, sig_full, i_oos, n, stop)[0])
        mh = metrics(sim(price, sig_hold, i_hold, n, stop)[0])
        print(f"{name:22s} | {fmt(mf):28s} | {fmt(mh):28s}")
