"""Feed the TUNED winners (from the deeper per-asset search) into the Overall
allocation mix and re-optimise, vs the current-strategy baseline.

  SOXX  MA25 > MA100, 5% stop
  GRID  MACD 10/20/9 (hist>0), 5% stop
  WGMI  price>MA50 & vol10 < 0.95*median(vol10,189), no stop
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "app")); sys.path.insert(0, str(_REPO))
import overall_core as oc
from ticker_config import get_config


def load(cfg):
    df = pd.read_csv(oc.tc.cache_paths(cfg)["daily"], index_col=0, parse_dates=True)
    if "px_close" not in df.columns:
        pref = f"{cfg.primary_symbol.lower()}_"
        df = df.rename(columns={c: "px_" + c[len(pref):]
                                for c in df.columns if c.startswith(pref)})
    return df


def macd_hist(c, f, s, sig):
    m = c.ewm(span=f, adjust=False).mean() - c.ewm(span=s, adjust=False).mean()
    return m - m.ewm(span=sig, adjust=False).mean()


TUNED = {   # (signal_fn, stop)
    "SOXX": (lambda d: d["px_close"].rolling(25).mean() > d["px_close"].rolling(100).mean(), 0.05),
    "GRID": (lambda d: macd_hist(d["px_close"], 10, 20, 9) > 0, 0.05),
    "WGMI": (lambda d: (d["px_close"] > d["px_close"].rolling(50).mean())
             & (np.log(d["px_close"]).diff().rolling(10).std()
                < 0.95 * np.log(d["px_close"]).diff().rolling(10).std().rolling(189).median()), 1.0),
}


def sim_stream(cfg, signal_fn, stop):
    daily = load(cfg); c = daily["px_close"]
    long_at_close = signal_fn(daily).to_numpy()
    price = c.to_numpy(float); dates = daily.index
    i0 = int(np.searchsorted(dates.values, np.datetime64(pd.Timestamp(cfg.oos_start))))
    n = len(price)
    rets = np.zeros(n); pos = np.zeros(n); in_pos = False; entry = np.nan
    for i in range(i0, n):
        rets[i] = (price[i] / price[i - 1] - 1.0) if in_pos else 0.0
        pos[i] = 1.0 if in_pos else 0.0
        if in_pos:
            if price[i] <= entry * (1 - stop) or not long_at_close[i - 1]:
                in_pos = False; entry = np.nan
        elif long_at_close[i - 1]:
            in_pos = True; entry = price[i]
    idx = dates[i0:n]
    return pd.Series(rets[i0:n], index=idx), pd.Series(pos[i0:n], index=idx)


def optimise(results):
    rets = oc.returns_matrix(results); pos = oc.position_matrix(results, rets.index)
    return oc.optimize_weights(rets, pos=pos, sata_daily=oc.SATA_DAILY)


def show(tag, opt):
    o = opt["optimal"]; ks = ("SOXX", "GRID", "WGMI")
    w = "  ".join(f"{k}={o['weights'].get(k,0)*100:.2f}%" for k in ks)
    print(f"  {tag:12s} ret {o['total_ret']*100:8.1f}%  CAGR {o['cagr']*100:5.1f}%  "
          f"MDD {o['mdd']*100:6.2f}%  Sharpe {o['sharpe']:.3f}  vol {o['vol']*100:.2f}%\n    {w}")


print("Running baseline universe (live fetch)…", flush=True)
base = oc.run_universe()
opt_base = optimise(base)
print("\n=== BASELINE (current strategies) ===")
show("baseline", opt_base)

streams = {k: sim_stream(get_config(k), fn, st) for k, (fn, st) in TUNED.items()}
sw = []
for r in base:
    if r["key"] in streams:
        r = dict(r); r["ret"], r["pos_series"] = streams[r["key"]]
    sw.append(r)
opt_sw = optimise(sw)
print("\n=== TUNED (SOXX 25/100, GRID MACD 10/20/9, WGMI MA50+vol k0.95) ===")
show("tuned", opt_sw)

b, o = opt_base["optimal"], opt_sw["optimal"]
print(f"\n  Δ vs baseline: dRet {(o['total_ret']-b['total_ret'])*100:+.1f}pp  "
      f"dMDD {(o['mdd']-b['mdd'])*100:+.2f}pp  dSharpe {o['sharpe']-b['sharpe']:+.3f}  "
      f"dVol {(o['vol']-b['vol'])*100:+.2f}pp")
