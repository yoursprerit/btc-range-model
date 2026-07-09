"""Feed the improved per-asset signals (SOXX=Dual-MA 20/100, GRID=MACD,
WGMI=MA+vol-filter) into the Overall allocation mix and re-optimise, to see if
the combined portfolio backtest improves.  Baseline = live universe with each
app's current strategy; only the return/position streams of the three improved
sleeves are swapped, then weights are re-optimised with the same optimiser call.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "app")); sys.path.insert(0, str(_REPO))
import overall_core as oc          # noqa
from ticker_config import get_config


def load(cfg):
    df = pd.read_csv(oc.tc.cache_paths(cfg)["daily"], index_col=0, parse_dates=True)
    if "px_close" not in df.columns:
        pref = f"{cfg.primary_symbol.lower()}_"
        df = df.rename(columns={c: "px_" + c[len(pref):]
                                for c in df.columns if c.startswith(pref)})
    return df


def sim_stream(cfg, signal_fn):
    """Daily strategy return + in-position Series over the app's OOS window."""
    daily = load(cfg)
    c = daily["px_close"]
    long_at_close = signal_fn(daily).to_numpy()
    price = c.to_numpy(float); dates = daily.index
    stop = cfg.fixed_stop
    i0 = int(np.searchsorted(dates.values, np.datetime64(pd.Timestamp(cfg.oos_start))))
    n = len(price)
    rets = np.zeros(n); pos = np.zeros(n); in_pos = False; entry = np.nan
    for i in range(i0, n):
        held = in_pos
        r = (price[i] / price[i - 1] - 1.0) if in_pos else 0.0
        rets[i] = r; pos[i] = 1.0 if held else 0.0
        if in_pos:
            if price[i] <= entry * (1 - stop) or not long_at_close[i - 1]:
                in_pos = False; entry = np.nan
        elif long_at_close[i - 1]:
            in_pos = True; entry = price[i]
    idx = dates[i0:n]
    return pd.Series(rets[i0:n], index=idx), pd.Series(pos[i0:n], index=idx)


SIGNALS = {
    "SOXX": lambda d: d["px_close"].rolling(20).mean() > d["px_close"].rolling(100).mean(),
    "GRID": lambda d: (d["px_close"].ewm(span=12, adjust=False).mean()
                       - d["px_close"].ewm(span=26, adjust=False).mean())
                      .pipe(lambda m: m - m.ewm(span=9, adjust=False).mean()) > 0,
    "WGMI": lambda d: (d["px_close"] > d["px_close"].rolling(get_config("WGMI").ma_window).mean())
                      & (np.log(d["px_close"]).diff().rolling(20).std()
                         < np.log(d["px_close"]).diff().rolling(20).std().rolling(252).median()),
}


def optimise(results):
    rets = oc.returns_matrix(results); pos = oc.position_matrix(results, rets.index)
    return oc.optimize_weights(rets, pos=pos, sata_daily=oc.SATA_DAILY), rets, pos


def show(tag, opt):
    o = opt["optimal"]; ks = ("SOXX", "GRID", "WGMI")
    w = "  ".join(f"{k}={o['weights'].get(k,0)*100:.2f}%" for k in ks)
    print(f"  {tag:16s} ret {o['total_ret']*100:8.1f}%  CAGR {o['cagr']*100:5.1f}%  "
          f"MDD {o['mdd']*100:6.2f}%  Sharpe {o['sharpe']:.3f}  vol {o['vol']*100:.2f}%")
    print(f"    weights: {w}")


print("Running baseline universe (live fetch)…", flush=True)
base = oc.run_universe()
print(f"  {len(base)} instruments")
opt_base, rets_base, pos_base = optimise(base)
print("\n=== BASELINE (current strategies) ===")
show("baseline", opt_base)

# swap the three improved sleeves' return + position streams
streams = {k: sim_stream(get_config(k), fn) for k, fn in SIGNALS.items()}
swapped = []
for r in base:
    if r["key"] in streams:
        r = dict(r)
        r["ret"], r["pos_series"] = streams[r["key"]]
    swapped.append(r)
opt_sw, rets_sw, pos_sw = optimise(swapped)
print("\n=== SWAP: SOXX=Dual-MA, GRID=MACD, WGMI=MA+vol-filter ===")
show("improved", opt_sw)

b, o = opt_base["optimal"], opt_sw["optimal"]
print(f"\n  Δ vs baseline: dRet {(o['total_ret']-b['total_ret'])*100:+.1f}pp  "
      f"dCAGR {(o['cagr']-b['cagr'])*100:+.2f}pp  dMDD {(o['mdd']-b['mdd'])*100:+.2f}pp  "
      f"dSharpe {o['sharpe']-b['sharpe']:+.3f}  dVol {(o['vol']-b['vol'])*100:+.2f}pp")


def sub(results, opt, start):
    rets = oc.returns_matrix(results); pos = oc.position_matrix(results, rets.index)
    w = np.array([opt["optimal"]["weights"].get(c, 0.0) for c in rets.columns])
    r = rets.loc[pd.Timestamp(start):]; p = pos.loc[pd.Timestamp(start):]
    return oc.curve_metrics(oc._equity(oc._combine(r, w, p, oc.SATA_DAILY)))


print("\n=== recent windows (each blend's own optimal weights) ===")
for s in ("2025-01-01", "2025-08-01"):
    mb = sub(base, opt_base, s); ms = sub(swapped, opt_sw, s)
    print(f"  {s}->now  baseline ret {mb['total_ret']*100:6.1f}% Sharpe {mb['sharpe']:.2f} | "
          f"improved ret {ms['total_ret']*100:6.1f}% Sharpe {ms['sharpe']:.2f}")
