"""Head-to-head: divergence (U1/D1/D2/D3) vs the MA trend filter vs Buy&Hold,
for the five MA-mode ETF apps (SOXX, VEGN, GRID, REMX, WGMI).

For each asset we build predictions/signals once, then:
  * run the current MA config,
  * sweep divergence thresholds (U1 × D2 × stop × D1-exit) and keep the best
    by Sharpe subject to MDD no worse than Buy&Hold,
over the out-of-sample window and the config's periods.
"""
from __future__ import annotations
import sys, itertools
from pathlib import Path
import pandas as pd, numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "app")); sys.path.insert(0, str(_REPO))
import ticker_core as tc, backtest_ticker as bt          # noqa
from ticker_config import get_config                      # noqa
from dataclasses import replace                           # noqa

ASSETS = ["SOXX", "VEGN", "GRID", "REMX", "WGMI"]
U1S = [0.05, 0.08, 0.12, 0.16]
D2S = [-0.08, -0.10, -0.14, -0.18]
STOPS = None   # filled per asset: (config stop, no stop)


def _load(cfg):
    csv = tc.cache_paths(cfg)["daily"]
    if Path(csv).exists():
        df = pd.read_csv(csv, index_col=0, parse_dates=True)
        if "px_close" not in df.columns:
            pref = f"{cfg.primary_symbol.lower()}_"
            df = df.rename(columns={c: "px_"+c[len(pref):] for c in df.columns if c.startswith(pref)})
        return df
    return tc.fetch_daily(cfg)


def _m(r):
    return bt._metrics(r["strat"], r["dates"]), bt._metrics(r["bh"], r["dates"])


for key in ASSETS:
    cfg = get_config(key)
    daily = _load(cfg)
    preds = bt.build_predictions(cfg, daily)
    sig = bt.precompute_signals(cfg, preds)
    oos = cfg.oos_start

    # current MA strategy
    ma_cfg = cfg
    r_ma = bt.simulate_regime(ma_cfg, preds, sig, "px_close",
                              ma_window=cfg.ma_window, stop_pct=cfg.fixed_stop, oos_start=oos)
    ma_m, bh_m = _m(r_ma)

    # divergence sweep
    best = None
    for U1, D2, stop, d1x in itertools.product(U1S, D2S, [cfg.fixed_stop, 1.0], [False, True]):
        dcfg = replace(cfg, strategy_mode="divergence",
                       u1_errhi_min=U1, d2_errhi_max=D2, fixed_stop=stop, use_d1_exit=d1x)
        r = bt.simulate(dcfg, preds, sig, "px_close", stop_pct=stop, U1=U1, D2=D2,
                        D1=cfg.d1_errlo_min, use_d1_exit=d1x, oos_start=oos)
        sm, _ = _m(r)
        cand = dict(U1=U1, D2=D2, stop=stop, d1x=d1x, sharpe=sm["sharpe"],
                    ret=sm["total_ret"], mdd=sm["mdd"], nt=len(r["trades"]))
        # keep best Sharpe with MDD no worse than B&H
        ok = sm["mdd"] >= bh_m["mdd"] - 1e-9
        score = (ok, sm["sharpe"])
        if best is None or score > (best["_ok"], best["sharpe"]):
            cand["_ok"] = ok; best = cand

    print(f"\n===== {key} (OOS {oos}→now) =====")
    print(f"  Buy&Hold   : ret {bh_m['total_ret']*100:7.1f}%  mdd {bh_m['mdd']*100:6.1f}%  sharpe {bh_m['sharpe']:.2f}")
    print(f"  MA{cfg.ma_window:<3d}/{cfg.fixed_stop*100:.0f}% : ret {ma_m['total_ret']*100:7.1f}%  mdd {ma_m['mdd']*100:6.1f}%  sharpe {ma_m['sharpe']:.2f}"
          f"   {'BEATS B&H(sharpe)' if ma_m['sharpe']>bh_m['sharpe'] else ''}")
    tag = f"U1={best['U1']} D2={best['D2']} stop={'none' if best['stop']==1.0 else str(int(best['stop']*100))+'%'} d1exit={best['d1x']}"
    print(f"  DIV best   : ret {best['ret']*100:7.1f}%  mdd {best['mdd']*100:6.1f}%  sharpe {best['sharpe']:.2f}   [{tag}], {best['nt']} trades")
    winner = max([("Buy&Hold", bh_m['sharpe']), ("MA", ma_m['sharpe']), ("Divergence", best['sharpe'])], key=lambda x: x[1])
    print(f"  → best Sharpe: {winner[0]}  |  div beats MA on Sharpe? {best['sharpe']>ma_m['sharpe']}  ret? {best['ret']>ma_m['total_ret']}")
