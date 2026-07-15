"""XLE / OIH / ERX — retrain through 2025-12-31, tune thresholds for the 2021→TODAY window.

Companion to ``eval_xle_retrain_2025.py`` (which optimised the FULL 2015→now
history).  Same machinery — re-fit the XLE daily High/Low signal model on data
extended to 2025-12-31, optionally exclude drastic XLE down-days, sweep the
divergence thresholds — but the tuning OBJECTIVE is now:

    maximise return AND minimise losses over the period 2021-01-01 → today (2026)

which is the '🌐 Full Out-of-Sample (2021 → now)' window in the UI.  OIH & ERX
trade the XLE signal, so the one XLE signal model drives all three sleeves.

Back-tests XLE / OIH / ERX across EVERY UI period (some now in-sample) and
compares with the currently-implemented strategy.  Evaluation only — nothing is
shipped (config, model artifacts, backtest_results.json untouched).

    python scripts/eval_xle_retrain_oos2021.py   # → data/xle/retrain_oos2021_eval.json
"""
from __future__ import annotations

import sys
import json
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

OOS = ("2021-01-01", None)          # the tuning objective window (2021 → today)
BEAR = ("2021-01-01", "2022-12-31")


def score_oos(preds, sig, U1, D2, D1, ud, stop):
    """XLE metrics on 2021→now (objective) + the 2021-22 bear (loss control)."""
    m, bh, _, _ = run_period(preds, sig, "px_close", U1, D2, D1, ud, stop, *OOS)
    mb, bhb, _, _ = run_period(preds, sig, "px_close", U1, D2, D1, ud, stop, *BEAR)
    ntr = run_period(preds, sig, "px_close", U1, D2, D1, ud, stop, *OOS)[3]
    return m, bh, mb, bhb, ntr


def main():
    daily = pd.read_csv("data/xle/macro_daily.csv", index_col=0, parse_dates=True)
    print(f"Data: {daily.index.min().date()} → {daily.index.max().date()} "
          f"({len(daily)} rows)  ERX={'erx_close' in daily.columns}")
    print("Tuning objective: maximise return / minimise losses over 2021-01-01 → today\n")

    # committed baseline — train ≤ 2020-12-31, live thresholds
    base_p, base_n = build_preds(daily, "2020-12-31")
    base_s = bt.precompute_signals(CFG, base_p)
    LIVE = dict(U1=CFG.u1_errhi_min, D2=CFG.d2_errhi_max, D1=CFG.d1_errlo_min,
                use_d1=CFG.use_d1_exit, stop=CFG.fixed_stop)
    base_res = {lbl: all_periods(base_p, base_s, col, LIVE["U1"], LIVE["D2"],
                                 LIVE["D1"], LIVE["use_d1"], LIVE["stop"])
                for lbl, col in TRADED}
    bmx = base_res["XLE"]["🌐 Full Out-of-Sample (2021 → now)"]
    print(f"BASELINE (committed, train≤2020): XLE 2021→now = "
          f"{fmt(bmx['strategy'])}  n={bmx['n_trades']}")

    # ── Section A: fixed committed thresholds, vary H/L training boundary ─────
    print("\n" + "=" * 92)
    print("SECTION A — committed thresholds FIXED, vary H/L training boundary "
          "(metrics on 2021→now)")
    print("=" * 92)
    print(f"  {'train≤':>10s} {'drop':>6s} {'rows':>5s} | {'XLE 2021→now':>24s} | "
          f"{'OIH':>9s} | {'ERX':>9s}")
    for te, db in [("2020-12-31", None), ("2022-12-31", None), ("2023-12-31", None),
                   ("2025-12-31", None), ("2025-12-31", -0.07), ("2025-12-31", -0.05)]:
        p, n = build_preds(daily, te, drop_below=db)
        s = bt.precompute_signals(CFG, p)
        mx = run_period(p, s, "px_close", LIVE["U1"], LIVE["D2"], LIVE["D1"],
                        LIVE["use_d1"], LIVE["stop"], *OOS)[0]
        mo = run_period(p, s, "oih_close", LIVE["U1"], LIVE["D2"], LIVE["D1"],
                        LIVE["use_d1"], LIVE["stop"], *OOS)[0]
        me = run_period(p, s, "erx_close", LIVE["U1"], LIVE["D2"], LIVE["D1"],
                        LIVE["use_d1"], 1.0, *OOS)[0]
        print(f"  {te:>10s} {str(db):>6s} {n:5d} | "
              f"{mx['total_ret']*100:+7.1f}% MDD{mx['mdd']*100:6.1f}% Sh{mx['sharpe']:4.2f} | "
              f"{mo['total_ret']*100:+8.1f}% | {me['total_ret']*100:+8.1f}%")

    # ── retrain scenarios (train ≤ 2025-12-31) ───────────────────────────────
    scenarios = {"retrain_full": dict(train_end="2025-12-31", drop_below=None),
                 "retrain_no<-7%": dict(train_end="2025-12-31", drop_below=-0.07),
                 "retrain_no<-5%": dict(train_end="2025-12-31", drop_below=-0.05)}
    cache = {}
    for name, kw in scenarios.items():
        p, n = build_preds(daily, **kw)
        cache[name] = (p, bt.precompute_signals(CFG, p), n)
        print(f"  built {name}: train rows={n}")

    # ── threshold sweep tuned to 2021→now ────────────────────────────────────
    U1g = [0.05, 0.08, 0.10, 0.12, 0.14, 0.16, 0.20]
    D2g = [-0.08, -0.10, -0.12, -0.15, -0.18]
    D1g = [0.10]
    UDg = [True, False]
    STOPg = [0.08, 0.10, 0.12, 1.0]
    bh_oos_mdd = bmx["buy_hold"]["mdd"]     # B&H 2021→now MDD (drawdown ceiling)

    rows = []
    for scen, (p, s, n) in cache.items():
        for U1 in U1g:
            for D2 in D2g:
                for D1 in D1g:
                    for ud in UDg:
                        for stop in STOPg:
                            m, bh, mb, bhb, ntr = score_oos(p, s, U1, D2, D1, ud, stop)
                            rows.append(dict(scen=scen, U1=U1, D2=D2, D1=D1, d1_exit=ud,
                                             stop=stop, ret=m["total_ret"], mdd=m["mdd"],
                                             sharpe=m["sharpe"], trades=ntr,
                                             mdd_ok=bool(m["mdd"] >= bh_oos_mdd - 1e-9),
                                             bear_ok=bool(mb["total_ret"] >= bhb["total_ret"] - 1e-9
                                                          or mb["mdd"] >= bhb["mdd"])))

    valid = [r for r in rows if r["mdd_ok"]]
    pool = sorted(valid or rows, key=lambda r: (r["ret"], r["sharpe"]), reverse=True)
    by_sharpe = sorted(valid or rows, key=lambda r: r["sharpe"], reverse=True)

    print(f"\n=== 2021→now sweep: {len(rows)} configs, {len(valid)} with MDD≤B&H ===")
    print(f"  {'scen':16s} {'U1':>5s} {'D2':>6s} {'d1x':>4s} {'stop':>5s} "
          f"{'2021ret':>9s} {'MDD':>7s} {'Sh':>5s} {'trd':>4s} bear")
    for r in pool[:12]:
        print(f"  {r['scen']:16s} {r['U1']:5.2f} {r['D2']:6.2f} {str(r['d1_exit'])[:1]:>4s} "
              f"{('-%.0f%%'%(r['stop']*100)) if r['stop']<1 else 'none':>5s} "
              f"{r['ret']*100:+8.1f}% {r['mdd']*100:6.1f}% {r['sharpe']:5.2f} "
              f"{r['trades']:4d}  {'Y' if r['bear_ok'] else 'n'}")
    print("\nTop 5 by Sharpe (MDD≤B&H):")
    for r in by_sharpe[:5]:
        print(f"  {r['scen']:16s} U1={r['U1']:.2f} D2={r['D2']:.2f} d1x={str(r['d1_exit'])[:1]} "
              f"stop={('-%.0f%%'%(r['stop']*100)) if r['stop']<1 else 'none':>5s} "
              f"ret={r['ret']*100:+7.1f}% MDD={r['mdd']*100:6.1f}% Sh={r['sharpe']:.2f} trd={r['trades']}")

    ret_pick, risk_pick = pool[0], by_sharpe[0]
    print(f"\nRET  pick (max 2021→now return): {ret_pick}")
    print(f"RISK pick (max 2021→now Sharpe): {risk_pick}")

    def per_all(pick):
        p, s, _n = cache[pick["scen"]]
        return {lbl: all_periods(p, s, col, pick["U1"], pick["D2"], pick["D1"],
                                 pick["d1_exit"], pick["stop"]) for lbl, col in TRADED}

    ret_res, risk_res = per_all(ret_pick), per_all(risk_pick)

    print("\n" + "=" * 100)
    print("PER-PERIOD: committed  vs  retrain-RET  vs  retrain-RISK  (return / MDD / Sharpe)")
    print("  (thresholds tuned to the 2021→now window)")
    print("=" * 100)
    for lbl, col in TRADED:
        print(f"\n{lbl}:")
        for plabel, _s, _e in CFG.periods:
            b = base_res[lbl][plabel]; r = ret_res[lbl][plabel]; k = risk_res[lbl][plabel]
            print(f"  {plabel:38s}")
            print(f"      committed    : {fmt(b['strategy'])}  wr={b['win_rate']:.0f}% n={b['n_trades']}")
            print(f"      retrain-RET  : {fmt(r['strategy'])}  wr={r['win_rate']:.0f}% n={r['n_trades']}")
            print(f"      retrain-RISK : {fmt(k['strategy'])}  wr={k['win_rate']:.0f}% n={k['n_trades']}")
            print(f"      buy&hold     : {fmt(b['buy_hold'])}")

    def meta(pick, res):
        return dict(scenario=pick["scen"], train_end="2025-12-31",
                    thresholds=dict(U1=pick["U1"], D2=pick["D2"], D1=pick["D1"],
                                    use_d1_exit=pick["d1_exit"], stop=pick["stop"]),
                    oos_2021=dict(ret=pick["ret"], mdd=pick["mdd"],
                                  sharpe=pick["sharpe"], trades=pick["trades"]),
                    results=res)

    Path("data/xle/retrain_oos2021_eval.json").write_text(json.dumps(dict(
        objective="maximise return / minimise losses over 2021→today",
        data_range=[str(daily.index.min().date()), str(daily.index.max().date())],
        baseline=dict(train_end="2020-12-31", thresholds=LIVE, results=base_res),
        retrain_RET=meta(ret_pick, ret_res), retrain_RISK=meta(risk_pick, risk_res),
        sweep_top=pool[:20]), indent=2, default=str))
    print("\nSaved → data/xle/retrain_oos2021_eval.json")


if __name__ == "__main__":
    main()
