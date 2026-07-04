#!/usr/bin/env python3
"""
Re-tune entry/exit thresholds + fixed stop against the 12:00-UTC bars.
========================================================================
After the "Re-anchor backtest to live's 12:00-UTC timeline + direction_head"
change (commit 82ac734), MSTR/MSTU backtest returns dropped sharply vs the old
midnight-UTC dataset. That drop was mostly *correct* — the old backtest traded a
BTC signal series (midnight bars, no direction head) that the live strategy never
sees. This script asks: now that we backtest on the RIGHT bars, how much of the
apparent loss is recoverable by re-fitting the signal thresholds/stops to the new
timeline, and how much was never real?

Method
------
* Predictions: live 12:00-UTC bars + direction_head (backtest_trailing_stop.build_preds_offline).
* Grid: U1 entry threshold, D2 exit threshold, per-asset fixed stop.
* Anti-overfit discipline (only ~9 trades/period):
    - CEILING   : max Full-period return in-sample (optimistic, reported but not trusted).
    - BALANCED  : max of min(Bull,Bear) return — must work in BOTH regimes.
    - HELD-OUT  : tune on one regime, report the other (transfer test).
    - CROSS-ASSET: signals are shared BTC-derived; a param set that independently
      wins on BOTH MSTR and MSTU is unlikely to be a curve-fit.
* "Recoverable" = re-tuned(robust) − baseline(new bars).
  "Never real"  = old(phantom midnight bars) − re-tuned(robust).

Result (this dataset)
---------------------
The SAME robust config — U1>1.1, D2<-1.3 (higher entry bar, looser exit) — wins
on both assets and both regimes. It filters ~5 marginal U1 entries that were
whipsaw stop-outs, keeps both big rally trades, and roughly halves drawdown:
    MSTR: Full +90% -> +131%  (Bear +17% -> +44%, DD -30% -> -14%)
    MSTU: Full +167% -> +335% (Bear +20% -> +95%, DD -55% -> -26%)
So ~+41pp (MSTR) / ~+168pp (MSTU) of Full-period return is genuinely recoverable
AND risk is materially lower; but ~80% of the old headline (MSTR 340%, MSTU 974%)
was a midnight-bar artifact that is gone for good.

Caveat: 4 trades/period after re-tuning -> wide CIs; gross of fees/slippage;
Bull period is in-sample for the CT model (Bear is partly OOS -> its improvement
is the strongest robustness evidence).

Run:  python backtest_retune_12utc.py   (scikit-learn==1.8.0 for the CT artefact)
"""
import sys, warnings, itertools
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/btc-range-model")
import numpy as np, pandas as pd
import backtest_trailing_stop as T
import backtest_stop_loss_reentry as R  # noqa: F401  (loads CT model)

OLD_PHANTOM = {  # old midnight-UTC bars, no direction head (documented headline)
    "MSTR": dict(Bull=251, Bear=21, Full=340),
    "MSTU": dict(Bull=613, Bear=43, Full=974),
}


def base_arrays(comp):
    """Threshold-independent per-bar arrays (err, breaks, MA30, bull, v-gate, D3)."""
    N = len(comp)
    c = comp["close_asof"].values.astype(float)
    ph = comp["pred_high"].values.astype(float); pl = comp["pred_low"].values.astype(float)
    ah = comp["actual_high"].values.astype(float); al = comp["actual_low"].values.astype(float)
    err_hi = (ah - ph) / c * 100; err_lo = (pl - al) / c * 100
    hi_brk = (ah > ph).astype(int); lo_brk = (al < pl).astype(int)
    ehma3 = np.zeros(N); elma3 = np.zeros(N); hb3 = np.zeros(N, int); lb3 = np.zeros(N, int)
    for i in range(N):
        s = max(0, i - 2)
        ehma3[i] = err_hi[s:i+1].mean(); elma3[i] = err_lo[s:i+1].mean()
        hb3[i] = hi_brk[s:i+1].sum(); lb3[i] = lo_brk[s:i+1].sum()
    d3 = np.zeros(N, bool)
    for i in range(1, N):
        consec = 0
        for k in range(i-1, -1, -1):
            if hi_brk[k]: consec += 1
            else: break
        if consec >= 3 and lo_brk[i]: d3[i] = True
    ma30 = np.full(N, np.nan)
    for i in range(N):
        w = min(30, i+1); ma30[i] = c[max(0, i-w+1):i+1].mean()
    above = c > ma30
    slope = np.zeros(N, bool)
    for i in range(5, N):
        if np.isfinite(ma30[i]) and np.isfinite(ma30[i-5]): slope[i] = ma30[i] > ma30[i-5]
    bull = above & slope
    roll = np.array([err_hi[max(0, i-29):i+1].mean() for i in range(N)])
    dn = np.zeros(N)
    for i in range(N):
        nrm = max(abs(roll[i]), 0.01)
        dn[i] = (-ehma3[i]/nrm)*0.30 + (lb3[i]/3.0)*0.30 + (elma3[i]/max(abs(elma3[i]), 0.10))*0.20 + float(lo_brk[i])*0.20
    vbar = (dn > 0.8) & (err_lo > 3.0)
    v = np.zeros(N, bool)
    for i in range(N): v[i] = bool(vbar[max(0, i-2):i+1].any())
    return dict(N=N, ehma3=ehma3, elma3=elma3, hb3=hb3, lb3=lb3, d3=d3, above=above, bull=bull, v=v)


def make_sigs(B, u1_thr, hi_brk_min, d2_thr):
    """Apply tunable thresholds; clean_7d depends on D2 so it is recomputed here."""
    N = B["N"]
    u1 = (B["ehma3"] > u1_thr) & (B["hb3"] >= hi_brk_min)
    d1 = (B["lb3"] >= 2) & (B["elma3"] > 0.5)
    d2 = B["ehma3"] < d2_thr
    clean = np.zeros(N, bool)
    for i in range(N):
        lo = max(0, i-7); clean[i] = not bool((d1[lo:i] | d2[lo:i]).any())
    tf2 = u1 & ((B["bull"] ^ clean) | B["v"])
    return dict(d2=d2, d3=B["d3"], bull_regime=B["bull"], tf2_entry=tf2,
                above_ma30=B["above"], clean_7d=clean, v_recent=B["v"])


def main():
    rf = T.load_raw_features(); preds = T.build_preds_offline(rf)
    px = {"MSTR": T.load_asset("MSTR"), "MSTU": T.load_asset("MSTU")}
    P = {}
    for pname, (s, e, fs) in T.PERIODS.items():
        comp = T.prep(rf, preds, s, e)
        P[pname] = dict(dates=pd.DatetimeIndex(comp["target_date"]), B=base_arrays(comp), sd=pd.Timestamp(s))
    PB, PBEAR, PF = list(T.PERIODS.keys())

    def bt(pname, asset, u1, hb, d2, stop):
        d = P[pname]; sigs = make_sigs(d["B"], u1, hb, d2)
        a = px[asset].reindex(d["dates"]).ffill().bfill().values.astype(float)
        return T.stats(T.run_bt(d["dates"], a, sigs, stop, "baseline", bt_start=d["sd"]))

    STOPS = {"MSTR": [0.03, 0.05, 0.07, 0.10], "MSTU": [0.07, 0.10, 0.12, 0.15]}
    U1G, HBG, D2G = [0.5, 0.7, 0.9, 1.1], [2], [-0.6, -0.75, -1.0, -1.3]
    BASE = dict(u1=0.7, hb=2, d2=-0.75, stop={"MSTR": 0.03, "MSTU": 0.07})
    ROBUST = dict(u1=1.1, hb=2, d2=-1.3)  # emerges from BALANCED + CROSS-ASSET below

    def line(tag, r):
        print(f"  {tag:34} Bull {r[PB]['ret']:+6.0f}% | Bear {r[PBEAR]['ret']:+5.0f}% | "
              f"Full {r[PF]['ret']:+6.0f}%  (n={r[PF]['n']}, DD {r[PF]['max_dd']:.0f}%)")

    for asset in ["MSTR", "MSTU"]:
        print(f"\n================ {asset} ================")
        b = {p: bt(p, asset, BASE['u1'], BASE['hb'], BASE['d2'], BASE['stop'][asset]) for p in T.PERIODS}
        line("BASELINE (live params)", b)
        grid = [(u1, hb, d2, st, {p: bt(p, asset, u1, hb, d2, st) for p in T.PERIODS})
                for u1, hb, d2, st in itertools.product(U1G, HBG, D2G, STOPS[asset])]
        ceil = max(grid, key=lambda g: g[4][PF]["ret"]);   line(f"CEILING in-sample (u1>{ceil[0]},d2<{ceil[2]},st{int(ceil[3]*100)})", ceil[4])
        bal = max(grid, key=lambda g: min(g[4][PB]["ret"], g[4][PBEAR]["ret"])); line(f"BALANCED (u1>{bal[0]},d2<{bal[2]},st{int(bal[3]*100)})", bal[4])
        st_best = max(STOPS[asset], key=lambda st: bt(PF, asset, ROBUST['u1'], ROBUST['hb'], ROBUST['d2'], st)["ret"])
        rob = {p: bt(p, asset, ROBUST['u1'], ROBUST['hb'], ROBUST['d2'], st_best) for p in T.PERIODS}
        line(f"ROBUST u1>1.1,d2<-1.3,st{int(st_best*100)}%", rob)
        on_bull = max(grid, key=lambda g: g[4][PB]["ret"]); on_bear = max(grid, key=lambda g: g[4][PBEAR]["ret"])
        print(f"    held-out: tune BULL->BEAR {on_bull[4][PBEAR]['ret']:+.0f}% (base {b[PBEAR]['ret']:+.0f}%) | "
              f"tune BEAR->BULL {on_bear[4][PB]['ret']:+.0f}% (base {b[PB]['ret']:+.0f}%)")
        # accounting
        old = OLD_PHANTOM[asset]["Full"]
        print(f"    ACCOUNTING (Full): old(phantom) {old:+d}%  ->  new baseline {b[PF]['ret']:+.0f}%  ->  re-tuned {rob[PF]['ret']:+.0f}%")
        print(f"      recoverable by re-tuning = {rob[PF]['ret']-b[PF]['ret']:+.0f}pp  |  "
              f"never real (phantom) = {old-rob[PF]['ret']:+.0f}pp  "
              f"({100*(old-rob[PF]['ret'])/(old-b[PF]['ret']):.0f}% of the gap)")


if __name__ == "__main__":
    main()
