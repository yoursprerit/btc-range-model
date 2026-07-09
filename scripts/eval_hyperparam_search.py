"""Deeper per-asset parameter search on the three winners:

  SOXX  Dual-MA crossover     (fast, slow, stop)
  GRID  MACD                  (fast, slow, signal, trend-confirm, stop)
  WGMI  MA + volatility filter(ma, vol-window, median-window, k, stop)

Discipline against overfitting: parameters are SELECTED on the TRAINING window
(OOS start -> 2025-08-01) and then VALIDATED on the UNSEEN holdout
(2025-08-01 -> now).  Selection rule = max Sharpe s.t. MDD <= Buy&Hold(train)
and >= 8 trades, tie-broken by return.  We report the training-optimal config on
train / holdout / full windows, compare to the current app config and the v1
hand-picked winner, and show how robust the training-optimal neighbourhood is
(median holdout Sharpe of the top-decile training configs).
"""
from __future__ import annotations
import sys, itertools
from pathlib import Path
import numpy as np, pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "app")); sys.path.insert(0, str(_REPO))
import ticker_core as tc                        # noqa
from ticker_config import get_config            # noqa

HOLDOUT = "2025-08-01"
TRADING = 252


def load(cfg):
    df = pd.read_csv(tc.cache_paths(cfg)["daily"], index_col=0, parse_dates=True)
    if "px_close" not in df.columns:
        pref = f"{cfg.primary_symbol.lower()}_"
        df = df.rename(columns={c: "px_" + c[len(pref):]
                                for c in df.columns if c.startswith(pref)})
    return df


def sim(price, long_at_close, i0, i1, stop):
    eq = 1.0; rets = []; in_pos = False; entry = np.nan; ntr = 0
    for i in range(i0, i1):
        r = (price[i] / price[i - 1] - 1.0) if in_pos else 0.0
        rets.append(r); eq *= (1 + r)
        if in_pos:
            if price[i] <= entry * (1 - stop) or not long_at_close[i - 1]:
                in_pos = False; entry = np.nan
        elif long_at_close[i - 1]:
            in_pos = True; entry = price[i]; ntr += 1
    return np.array(rets), ntr


def metrics(rets):
    if len(rets) < 2:
        return dict(ret=0.0, mdd=0.0, sharpe=0.0)
    eq = np.cumprod(1 + rets); peak = np.maximum.accumulate(eq)
    return dict(ret=float(eq[-1] - 1), mdd=float(np.min(eq / peak - 1)),
                sharpe=float(np.mean(rets) / (np.std(rets) + 1e-12) * np.sqrt(TRADING)))


def bh(price, i0, i1):
    seg = price[i0:i1]; return metrics(np.diff(seg) / seg[:-1])


# ── candidate signal generators (rule-based → no fitting, no leakage) ──────
def gen_soxx(c):
    ma = {w: c.rolling(w).mean() if w > 1 else c for w in (1, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150, 200)}
    for fast in (1, 10, 15, 20, 25, 30, 40, 50):
        for slow in (60, 80, 100, 120, 150, 200):
            if fast >= slow:
                continue
            sig = (ma[fast] > ma[slow]).to_numpy()
            for stop in (1.0, 0.05, 0.08):
                yield dict(fast=fast, slow=slow, stop=stop), sig, stop


def gen_grid(c):
    emas = {s: c.ewm(span=s, adjust=False).mean() for s in (6, 8, 10, 12, 16, 20, 26, 34, 40, 50)}
    for fast in (6, 8, 10, 12, 16, 20):
        for slow in (20, 26, 34, 40, 50):
            if fast >= slow:
                continue
            macd = emas[fast] - emas[slow]
            for sigspan in (5, 9, 12, 16):
                hist = (macd - macd.ewm(span=sigspan, adjust=False).mean())
                for confirm in (False, True):
                    sig = ((hist > 0) & (macd > 0)) if confirm else (hist > 0)
                    sig = sig.to_numpy()
                    for stop in (1.0, 0.05):
                        yield dict(fast=fast, slow=slow, signal=sigspan,
                                   confirm=confirm, stop=stop), sig, stop


def gen_wgmi(c):
    rt = np.log(c).diff()
    mas = {w: c.rolling(w).mean() for w in (15, 20, 25, 30, 40, 50, 60)}
    vols = {w: rt.rolling(w).std() for w in (10, 15, 20, 30)}
    refs = {(vw, rw): vols[vw].rolling(rw).median() for vw in vols for rw in (126, 189, 252)}
    for ma in (15, 20, 25, 30, 40, 50, 60):
        above = c > mas[ma]
        for vw in (10, 15, 20, 30):
            for rw in (126, 189, 252):
                ref = refs[(vw, rw)]
                for k in (0.85, 0.95, 1.0, 1.1, 1.25):
                    sig = (above & (vols[vw] < k * ref)).to_numpy()
                    for stop in (1.0, 0.08, 0.10, 0.15):
                        yield dict(ma=ma, vol_win=vw, med_win=rw, k=k, stop=stop), sig, stop


GEN = {"SOXX": gen_soxx, "GRID": gen_grid, "WGMI": gen_wgmi}
CURRENT = {"SOXX": "MA40/5%", "GRID": "MA150/5%", "WGMI": "MA30/10%"}
V1 = {"SOXX": "Dual-MA 20/100", "GRID": "MACD 12/26/9", "WGMI": "MA30 + vol-filter (k=1.0,vw20,rw252)"}


def stops(s):
    return "none" if s == 1.0 else f"{int(s*100)}%"


for key in ("SOXX", "GRID", "WGMI"):
    cfg = get_config(key)
    daily = load(cfg); c = daily["px_close"]; price = c.to_numpy(float)
    dates = daily.index.values
    i_oos = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(cfg.oos_start))))
    i_hold = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(HOLDOUT))))
    n = len(price)
    bh_tr = bh(price, i_oos, i_hold)

    rows = []
    for params, sig, stop in GEN[key](c):
        rt_tr, ntr_tr = sim(price, sig, i_oos, i_hold, stop)
        m_tr = metrics(rt_tr)
        rt_ho, _ = sim(price, sig, i_hold, n, stop); m_ho = metrics(rt_ho)
        rt_fu, ntr_fu = sim(price, sig, i_oos, n, stop); m_fu = metrics(rt_fu)
        rows.append(dict(params=params, ntr_tr=ntr_tr, ntr_fu=ntr_fu,
                         tr=m_tr, ho=m_ho, fu=m_fu))

    # selection on TRAIN: max Sharpe s.t. MDD<=B&H(train) and >=8 trades
    feas = [r for r in rows if r["tr"]["mdd"] >= bh_tr["mdd"] - 1e-9 and r["ntr_tr"] >= 8]
    pool = feas if feas else rows
    best = max(pool, key=lambda r: (r["tr"]["sharpe"], r["tr"]["ret"]))
    # overfit ceiling: best on the FULL window (would-be hindsight optimum)
    best_full = max(rows, key=lambda r: (r["fu"]["sharpe"], r["fu"]["ret"]))
    # robustness: top-decile of train configs by Sharpe → their holdout Sharpe
    ranked = sorted(rows, key=lambda r: r["tr"]["sharpe"], reverse=True)
    topk = ranked[:max(5, len(ranked) // 10)]
    med_ho_top = float(np.median([r["ho"]["sharpe"] for r in topk]))

    def line(tag, r):
        return (f"  {tag:34s} train {r['tr']['ret']*100:+7.1f}%/{r['tr']['sharpe']:.2f}  "
                f"holdout {r['ho']['ret']*100:+7.1f}%/{r['ho']['sharpe']:.2f}  "
                f"full {r['fu']['ret']*100:+7.1f}%/{r['fu']['mdd']*100:.0f}%/{r['fu']['sharpe']:.2f}  "
                f"(ntr {r['ntr_fu']})")

    print(f"\n{'='*104}\n{key}  current={CURRENT[key]}  |  windows: train {cfg.oos_start}->{HOLDOUT}, "
          f"holdout {HOLDOUT}->now   [ret/Sharpe]")
    print(f"  B&H(train) MDD {bh_tr['mdd']*100:.0f}%   |   {len(rows)} configs searched, "
          f"{len(feas)} feasible")
    print(line(f"TRAIN-OPTIMAL {best['params']}", best))
    print(line("  (full-window hindsight optimum)", best_full))
    print(f"  robustness: median holdout Sharpe of top-decile train configs = {med_ho_top:.2f}")
