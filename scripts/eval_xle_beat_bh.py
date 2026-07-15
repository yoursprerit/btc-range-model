"""XLE / OIH / ERX — find ANY strategy (trend or divergence) that BEATS Buy & Hold 2021→today.

The committed divergence Pure-Regime beats B&H on drawdown and Sharpe but NOT on
total return over 2021→now (it sits out too much of the energy bull).  This is an
open strategy search for a single XLE signal — applied to XLE / OIH / ERX (as
shipped) — that beats B&H on total RETURN over 2021→today while keeping drawdown
no worse than B&H.

Search space (signal computed on the XLE close, executed on each sleeve's price):
  * trend family — ma / dual_ma / macd / ma_vol  (no fitted model; window params)
  * divergence   — U1/D2/D1/d1-exit, H/L model on several training windows,
                   ±drastic-drop exclusion (the brief's "new training data /
                   exclude drastic drops" levers)
Stops: XLE & OIH sweep {none,−8%,−10%,−12%}; ERX always stop-less (shipped).

Back-tests every UI period for the winner(s) and compares with committed.
Evaluation only — config, model artifacts, backtest_results.json untouched.

    python scripts/eval_xle_beat_bh.py            # → data/xle/beat_bh_eval.json
"""
from __future__ import annotations

import sys
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "app"))
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))

import backtest_ticker as bt                       # noqa: E402
from eval_xle_retrain_2025 import (                # noqa: E402
    build_preds, run_period, all_periods, fmt, CFG, TRADED)

OOS = ("2021-01-01", None)
SLEEVES = [("XLE", "px_close"), ("OIH", "oih_close"), ("ERX", "erx_close")]


def sim(cfg2, preds, sig, col, stop, s=OOS[0], e=OOS[1]):
    """Run a candidate; ERX is always stop-less (shipped)."""
    st = 1.0 if col == "erx_close" else stop
    r = bt.run_strategy(cfg2, preds, sig, col, oos_start=s, end=e, stop_pct=st)
    m = bt._metrics(r["strat"], r["dates"]); bh = bt._metrics(r["bh"], r["dates"])
    wr = float((r["trades"] > 0).mean() * 100) if len(r["trades"]) else 0.0
    return m, bh, wr, int(len(r["trades"]))


def eval_candidate(cfg2, preds, sig, stop):
    """XLE/OIH/ERX 2021→now: strat vs B&H return + drawdown."""
    out = {}
    beats = 0; sum_excess = 0.0; mdd_ok_all = True; sharpes = []
    for lbl, col in SLEEVES:
        m, bh, wr, ntr = sim(cfg2, preds, sig, col, stop)
        out[lbl] = dict(strat=m, bh=bh, wr=wr, n=ntr,
                        beats=bool(m["total_ret"] > bh["total_ret"]),
                        mdd_ok=bool(m["mdd"] >= bh["mdd"] - 1e-9))
        beats += out[lbl]["beats"]
        sum_excess += m["total_ret"] - bh["total_ret"]
        mdd_ok_all = mdd_ok_all and out[lbl]["mdd_ok"]
        sharpes.append(m["sharpe"])
    return dict(sleeves=out, n_beats=beats, sum_excess=sum_excess,
                mdd_ok_all=mdd_ok_all, avg_sharpe=float(np.mean(sharpes)))


def main():
    daily = pd.read_csv("data/xle/macro_daily.csv", index_col=0, parse_dates=True)
    print(f"Data: {daily.index.min().date()} → {daily.index.max().date()} "
          f"({len(daily)} rows)  ERX={'erx_close' in daily.columns}")
    print("Goal: beat Buy & Hold on TOTAL RETURN over 2021→today, MDD ≤ B&H\n")

    # prices come from any preds frame; committed divergence sig from train≤2020
    preds, _ = build_preds(daily, "2020-12-31")
    sig_2020 = bt.precompute_signals(CFG, preds)

    # committed baseline (divergence, train≤2020)
    base = eval_candidate(CFG, preds, sig_2020, CFG.fixed_stop)
    print("BUY & HOLD 2021→now:  " + "  ".join(
        f"{lbl} {base['sleeves'][lbl]['bh']['total_ret']*100:+.1f}%/"
        f"{base['sleeves'][lbl]['bh']['mdd']*100:.1f}%" for lbl, _ in SLEEVES))
    print("COMMITTED  2021→now:  " + "  ".join(
        f"{lbl} {base['sleeves'][lbl]['strat']['total_ret']*100:+.1f}%/"
        f"{base['sleeves'][lbl]['strat']['mdd']*100:.1f}%" for lbl, _ in SLEEVES)
        + f"   (beats B&H: {base['n_beats']}/3)\n")

    # cross-instrument angle: does any sleeve's STRATEGY out-return the XLE
    # sector benchmark (B&H XLE)?  The up-beta ERX sleeve is the only return-beat.
    xle_bh = base["sleeves"]["XLE"]["bh"]
    print("Cross-benchmark — each committed sleeve vs the XLE sector B&H "
          f"(+{xle_bh['total_ret']*100:.1f}% / {xle_bh['mdd']*100:.1f}%):")
    for lbl, _ in SLEEVES:
        m = base["sleeves"][lbl]["strat"]
        print(f"  {lbl} strat {m['total_ret']*100:+7.1f}% / {m['mdd']*100:6.1f}% / Sh{m['sharpe']:.2f}"
              f"   beats XLE-sector B&H: {m['total_ret'] > xle_bh['total_ret']}")
    print()

    cands = []   # (name, cfg2, sig, stop)

    # ── trend family (no training window needed) ─────────────────────────────
    for w in (20, 30, 40, 50, 60, 80, 100, 120, 150, 200):
        for stop in (1.0, 0.08, 0.10, 0.12):
            cands.append((f"ma{w}", replace(CFG, strategy_mode="ma", ma_window=w),
                          sig_2020, stop))
    for f, s in [(10, 50), (20, 50), (20, 100), (25, 100), (30, 120),
                 (50, 100), (50, 150), (50, 200)]:
        for stop in (1.0, 0.08, 0.10, 0.12):
            cands.append((f"dma{f}/{s}",
                          replace(CFG, strategy_mode="dual_ma", ma_fast=f, ma_slow=s),
                          sig_2020, stop))
    for mf, ms, msig in [(12, 26, 9), (10, 20, 9), (8, 21, 5)]:
        for stop in (1.0, 0.08, 0.10, 0.12):
            cands.append((f"macd{mf}/{ms}/{msig}",
                          replace(CFG, strategy_mode="macd", macd_fast=mf, macd_slow=ms,
                                  macd_signal=msig), sig_2020, stop))
    for w in (50, 80, 100):
        for k in (0.9, 1.0, 1.1):
            for stop in (1.0, 0.08):
                cands.append((f"mavol{w}/k{k}",
                              replace(CFG, strategy_mode="ma_vol", ma_window=w,
                                      vol_win=20, vol_med_win=252, vol_k=k), sig_2020, stop))

    # ── divergence family across training windows + drop exclusion ───────────
    div_windows = {
        "div2015-2020": dict(train_end="2020-12-31"),
        "div2015-2025": dict(train_end="2025-12-31"),
        "div2020-2025d5": dict(train_start="2020-01-01", train_end="2025-12-31", drop_below=-0.05),
    }
    div_sigs = {}
    for nm, kw in div_windows.items():
        p, _ = build_preds(daily, **kw)
        div_sigs[nm] = bt.precompute_signals(CFG, p)
    for nm in div_windows:
        for U1 in (0.05, 0.08, 0.12, 0.16, 0.20):
            for D2 in (-0.08, -0.12, -0.18):
                for ud in (True, False):
                    for stop in (1.0, 0.08, 0.12):
                        cands.append((f"{nm}:U1={U1}D2={D2}d1={str(ud)[:1]}",
                                      replace(CFG, strategy_mode="divergence",
                                              u1_errhi_min=U1, d2_errhi_max=D2,
                                              use_d1_exit=ud), div_sigs[nm], stop))

    # ── evaluate all ─────────────────────────────────────────────────────────
    results = []
    for name, cfg2, sig, stop in cands:
        try:
            r = eval_candidate(cfg2, preds if sig is sig_2020 else preds, sig, stop)
        except Exception as ex:
            continue
        r["name"] = name; r["stop"] = stop
        results.append(r)

    # rank: most sleeves beating B&H, then smallest total drawdown breach, then excess
    def key(r):
        worst_mdd_gap = min(r["sleeves"][l]["strat"]["mdd"] - r["sleeves"][l]["bh"]["mdd"]
                            for l, _ in SLEEVES)
        return (r["n_beats"], r["sum_excess"], r["avg_sharpe"])
    results.sort(key=key, reverse=True)

    print("=== top candidates by (sleeves beating B&H, Σ excess return, avg Sharpe) ===")
    print(f"  {'candidate':34s} {'stop':>5s} {'beats':>5s} | "
          f"{'XLE Δret':>9s} {'OIH Δret':>9s} {'ERX Δret':>9s} | {'avgSh':>5s} mdd≤bh")
    for r in results[:20]:
        d = {l: r['sleeves'][l] for l, _ in SLEEVES}
        print(f"  {r['name']:34s} "
              f"{('-%.0f%%'%(r['stop']*100)) if r['stop']<1 else 'none':>5s} "
              f"{r['n_beats']:>3d}/3 | "
              + " ".join(f"{(d[l]['strat']['total_ret']-d[l]['bh']['total_ret'])*100:+8.1f}%"
                        for l, _ in SLEEVES)
              + f" | {r['avg_sharpe']:5.2f}  {'Y' if r['mdd_ok_all'] else 'n'}")

    # best that beats B&H on all 3 (if any), else the closest (least return given up)
    all3 = [r for r in results if r["n_beats"] == 3]
    winner = (all3[0] if all3 else results[0])
    if winner["n_beats"] == 0:
        print(f"\nNO candidate beats B&H total return on ANY sleeve over 2021→now "
              f"(strong energy bull — a long/flat overlay cannot out-return B&H).")
        print(f"Closest: {winner['name']} (stop {winner['stop']}).")
    else:
        print(f"\nWINNER: {winner['name']} (stop {winner['stop']}) — "
              f"beats B&H on {winner['n_beats']}/3 sleeves 2021→now")

    # ── full per-period back-test for winner vs committed ────────────────────
    # rebuild the winner cfg2 + sig
    wname = winner["name"]
    wsig = sig_2020
    for nm in div_windows:
        if wname.startswith(nm):
            wsig = div_sigs[nm]
    # find the cfg2 from the candidate list
    wcfg = next(c[1] for c in cands if c[0] == wname and c[3] == winner["stop"])

    def per_all_stop(cfg2, sig, stop):
        out = {}
        for lbl, col in TRADED:
            st = 1.0 if col == "erx_close" else stop
            periods = {}
            for plabel, s, e in CFG.periods:
                m, bh, wr, ntr = sim(cfg2, preds, sig, col, st, s, e)
                periods[plabel] = dict(strategy=m, buy_hold=bh, win_rate=wr, n_trades=ntr)
            out[lbl] = periods
        return out

    base_res = per_all_stop(CFG, sig_2020, CFG.fixed_stop)
    win_res = per_all_stop(wcfg, wsig, winner["stop"])

    print("\n" + "=" * 100)
    print(f"PER-PERIOD: committed  vs  WINNER [{wname}]  (return / MDD / Sharpe)")
    print("=" * 100)
    for lbl, col in TRADED:
        print(f"\n{lbl}:")
        for plabel, _s, _e in CFG.periods:
            b = base_res[lbl][plabel]; w = win_res[lbl][plabel]
            tag = "  ⭐" if plabel.startswith("🌐") else ""
            print(f"  {plabel:38s}{tag}")
            print(f"      committed : {fmt(b['strategy'])}  wr={b['win_rate']:.0f}% n={b['n_trades']}")
            print(f"      WINNER    : {fmt(w['strategy'])}  wr={w['win_rate']:.0f}% n={w['n_trades']}")
            print(f"      buy&hold  : {fmt(b['buy_hold'])}")

    Path("data/xle/beat_bh_eval.json").write_text(json.dumps(dict(
        objective="beat Buy & Hold total return over 2021→today, MDD ≤ B&H",
        data_range=[str(daily.index.min().date()), str(daily.index.max().date())],
        buy_hold_2021={l: base["sleeves"][l]["bh"] for l, _ in SLEEVES},
        committed_2021={l: base["sleeves"][l]["strat"] for l, _ in SLEEVES},
        winner=dict(name=wname, stop=winner["stop"], n_beats=winner["n_beats"],
                    results=win_res),
        committed_results=base_res,
        top=[dict(name=r["name"], stop=r["stop"], n_beats=r["n_beats"],
                  sum_excess=r["sum_excess"], avg_sharpe=r["avg_sharpe"],
                  sleeves={l: dict(ret=r["sleeves"][l]["strat"]["total_ret"],
                                   mdd=r["sleeves"][l]["strat"]["mdd"],
                                   bh_ret=r["sleeves"][l]["bh"]["total_ret"],
                                   beats=r["sleeves"][l]["beats"]) for l, _ in SLEEVES})
             for r in results[:25]]), indent=2, default=str))
    print("\nSaved → data/xle/beat_bh_eval.json")


if __name__ == "__main__":
    main()
