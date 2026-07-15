"""XLE / OIH / ERX — retrain on data through 2025-12-31 + re-tune divergence thresholds.

Evaluation only (does NOT touch the app, the committed config, or the model/artifact
files).  It answers the brief:

  * Re-train the XLE daily H/L signal model with training data extended to
    2025-12-31 (the current committed model trains only on the pre-OOS window,
    ≤ 2020-12-31).  OIH & ERX trade the XLE signal, so this one signal model
    drives all three.
  * Optionally EXCLUDE drastic XLE down-days from the H/L training fit (the brief
    permits this if it helps) — the huge 2020 oil-war / COVID drops distort the
    low-band ridge.
  * Re-tune the divergence signal thresholds (U1 / D2 / D1 / use_d1_exit / stop)
    to maximise return and control losses over the FULL period.
  * Back-test XLE / OIH / ERX across EVERY UI period (even those now in-sample,
    per the brief) and compare with the currently-implemented strategy.

    python scripts/eval_xle_retrain_2025.py

Writes a machine-readable dump to data/xle/retrain_2025_eval.json (git-ignored
scratch — the committed artifacts are left untouched).
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "app"))
sys.path.insert(0, str(_REPO))

import ticker_core as tc              # noqa: E402
import backtest_ticker as bt         # noqa: E402
from ticker_config import get_config  # noqa: E402

CFG = get_config("XLE")
TRADED = [("XLE", "px_close"), ("OIH", "oih_close"), ("ERX", "erx_close")]
# ERX (2x) always trades stop-less on the XLE signal (shipped decision); XLE/OIH
# carry the swept 1x stop.
LEV_STOPLESS = {"erx_close": 1.0}


# ── H/L prediction builder with a configurable train window + drop filter ──
def build_preds(daily, train_end, drop_below=None, train_start=None):
    """Fit the daily H/L ridge on data in [train_start, train_end], predict EVERY bar.

    ``train_start`` (e.g. "2020-01-01"): left edge of the training window.  None
    keeps the full history back to the first available bar (the committed default).
    ``drop_below`` (e.g. -0.07): exclude training rows whose realised XLE daily
    close-to-close return is below that level, so drastic energy crash days do
    not distort the ridge fit / residual bias.  Prediction is still produced for
    every bar (nothing is dropped from the back-test).
    """
    feat = tc.build_daily_features(CFG, daily)
    close = daily["px_close"]; high = daily["px_high"]; low = daily["px_low"]
    prev_c = close.shift(1)
    y_hi = high / prev_c - 1.0
    y_lo = low / prev_c - 1.0
    ret_cc = close / prev_c - 1.0
    frac = feat.isna().mean()
    feat_cols = [c for c in feat.columns if frac[c] <= 0.35]

    df = feat[feat_cols].copy()
    df["y_hi"] = y_hi; df["y_lo"] = y_lo; df["close_asof"] = prev_c
    df["actual_high"] = high; df["actual_low"] = low
    df["px_close"] = daily["px_close"]; df["ret_cc"] = ret_cc
    for _lbl, col in TRADED:
        if col in daily and col not in df:
            df[col] = daily[col]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=feat_cols + ["y_hi", "y_lo", "close_asof"])

    train = df.loc[:train_end]
    if train_start is not None:
        train = train.loc[train_start:]
    if drop_below is not None:
        keep = train["ret_cc"] >= drop_below
        train = train[keep]

    def _mk():
        return Pipeline([("sc", StandardScaler()),
                         ("m", RidgeCV(alphas=np.logspace(-3, 3, 13)))])
    mh = _mk().fit(train[feat_cols], train["y_hi"])
    ml = _mk().fit(train[feat_cols], train["y_lo"])
    raw_ph = df["close_asof"] * (1 + mh.predict(df[feat_cols]))
    raw_pl = df["close_asof"] * (1 + ml.predict(df[feat_cols]))
    # bias correction estimated on the (filtered) training window
    tr_idx = train.index
    bias_hi = float((df.loc[tr_idx, "actual_high"] - raw_ph.loc[tr_idx]).mean())
    bias_lo = float((df.loc[tr_idx, "actual_low"] - raw_pl.loc[tr_idx]).mean())
    df["pred_high"] = raw_ph + bias_hi
    df["pred_low"] = raw_pl + bias_lo
    df["target_date"] = df.index
    keep = ["close_asof", "pred_high", "pred_low", "actual_high", "actual_low",
            "target_date", "px_close"] + [c for _l, c in TRADED if c != "px_close"]
    return df[keep].copy(), int(len(train))


# ── metrics helpers ─────────────────────────────────────────────────────────
def run_period(preds, sig, col, U1, D2, D1, use_d1, stop, s, e):
    r = bt.simulate(CFG, preds, sig, col, stop_pct=stop, U1=U1, D2=D2, D1=D1,
                    use_d1_exit=use_d1, oos_start=s, end=e)
    m = bt._metrics(r["strat"], r["dates"])
    bh = bt._metrics(r["bh"], r["dates"])
    wr = float((r["trades"] > 0).mean() * 100) if len(r["trades"]) else 0.0
    return m, bh, wr, int(len(r["trades"]))


def all_periods(preds, sig, col, U1, D2, D1, use_d1, stop):
    out = {}
    for lbl, s, e in CFG.periods:
        st = LEV_STOPLESS.get(col, stop)
        m, bh, wr, ntr = run_period(preds, sig, col, U1, D2, D1, use_d1, st, s, e)
        out[lbl] = dict(strategy=m, buy_hold=bh, win_rate=wr, n_trades=ntr)
    return out


def fmt(m):
    return f"{m['total_ret']*100:+8.1f}% MDD={m['mdd']*100:6.1f}% Sh={m['sharpe']:5.2f}"


# ── candidate scoring on the FULL-history window (the brief's "full period") ──
def score_full(preds, sig, U1, D2, D1, use_d1, stop):
    """Return XLE full-history metrics + a bear-period check (2021-22)."""
    fs, fe = "2015-01-01", None
    m, bh, wr, ntr = run_period(preds, sig, "px_close", U1, D2, D1, use_d1, stop, fs, fe)
    mb, bhb, _, _ = run_period(preds, sig, "px_close", U1, D2, D1, use_d1, stop,
                               "2021-01-01", "2022-12-31")
    return m, bh, mb, bhb, ntr


def main():
    daily = pd.read_csv("data/xle/macro_daily.csv", index_col=0, parse_dates=True)
    print(f"Data: {daily.index.min().date()} → {daily.index.max().date()} "
          f"({len(daily)} rows)  ERX={'erx_close' in daily.columns}\n")

    # ── committed baseline: train ≤ 2020-12-31, live thresholds ──────────────
    base_preds, base_n = build_preds(daily, "2020-12-31")
    base_sig = bt.precompute_signals(CFG, base_preds)
    LIVE = dict(U1=CFG.u1_errhi_min, D2=CFG.d2_errhi_max, D1=CFG.d1_errlo_min,
                use_d1=CFG.use_d1_exit, stop=CFG.fixed_stop)
    print(f"BASELINE (committed): train≤2020-12-31 ({base_n} rows), "
          f"U1={LIVE['U1']} D2={LIVE['D2']} D1={LIVE['D1']} "
          f"d1_exit={LIVE['use_d1']} stop=-{LIVE['stop']*100:.0f}% (ERX stop-less)")
    base_res = {lbl: all_periods(base_preds, base_sig, col,
                                 LIVE["U1"], LIVE["D2"], LIVE["D1"],
                                 LIVE["use_d1"], LIVE["stop"]) for lbl, col in TRADED}

    # ── SECTION A: isolate the training-window effect ─────────────────────────
    # Hold the COMMITTED thresholds fixed; vary ONLY the H/L training boundary.
    # Shows whether extending training past 2020 helps or hurts the signal.
    print("\n" + "=" * 92)
    print("SECTION A — committed thresholds held FIXED, vary only H/L training boundary")
    print("  (XLE full-history 2015→now / bear 2021-22 / OIH & ERX full-history)")
    print("=" * 92)
    boundaries = [("2020-12-31", None), ("2022-12-31", None), ("2023-12-31", None),
                  ("2025-12-31", None), ("2025-12-31", -0.07), ("2025-12-31", -0.05)]
    print(f"  {'train≤':>10s} {'drop':>6s} {'rows':>5s} | "
          f"{'XLE full':>22s} | {'XLE bear':>10s} | {'OIH full':>10s} | {'ERX full':>10s}")
    for te, db in boundaries:
        p, n = build_preds(daily, te, drop_below=db)
        s = bt.precompute_signals(CFG, p)
        mx, bhx, mxb, _, _ = score_full(p, s, LIVE["U1"], LIVE["D2"], LIVE["D1"],
                                        LIVE["use_d1"], LIVE["stop"])
        mo = run_period(p, s, "oih_close", LIVE["U1"], LIVE["D2"], LIVE["D1"],
                        LIVE["use_d1"], LIVE["stop"], "2015-01-01", None)[0]
        me = run_period(p, s, "erx_close", LIVE["U1"], LIVE["D2"], LIVE["D1"],
                        LIVE["use_d1"], 1.0, "2015-01-01", None)[0]
        tag = f"{te[:7]}" + (f" <{db*100:.0f}%" if db else "")
        print(f"  {te:>10s} {str(db):>6s} {n:5d} | "
              f"{mx['total_ret']*100:+7.1f}% MDD{mx['mdd']*100:6.1f}% Sh{mx['sharpe']:4.2f} | "
              f"{mxb['total_ret']*100:+8.1f}% | "
              f"{mo['total_ret']*100:+8.1f}% | {me['total_ret']*100:+8.1f}%")

    # ── retrain scenarios: train ≤ 2025-12-31, ±drastic-drop exclusion ───────
    scenarios = {
        "retrain_full":      dict(train_end="2025-12-31", drop_below=None),
        "retrain_no<-7%":    dict(train_end="2025-12-31", drop_below=-0.07),
        "retrain_no<-5%":    dict(train_end="2025-12-31", drop_below=-0.05),
    }
    preds_cache = {}
    for name, kw in scenarios.items():
        p, n = build_preds(daily, **kw)
        s = bt.precompute_signals(CFG, p)
        preds_cache[name] = (p, s, n)
        print(f"  built {name}: train rows={n}")

    # ── threshold sweep (XLE signal) on each retrain scenario ────────────────
    U1_grid = [0.05, 0.08, 0.10, 0.12, 0.14, 0.16, 0.20]
    D2_grid = [-0.08, -0.10, -0.12, -0.15, -0.18]
    D1_grid = [0.10]
    D1EXIT = [True, False]
    STOP_grid = [0.08, 0.10, 0.12, 1.0]

    print("\n=== threshold sweep — maximise XLE full-history return, MDD-controlled ===")
    sweep_rows = []
    for scen, (p, s, n) in preds_cache.items():
        for U1 in U1_grid:
            for D2 in D2_grid:
                for D1 in D1_grid:
                    for ud in D1EXIT:
                        for stop in STOP_grid:
                            m, bh, mb, bhb, ntr = score_full(p, s, U1, D2, D1, ud, stop)
                            # constraints: MDD no worse than B&H full-history AND
                            # beat B&H in the 2021-22 bear (loss-control brief)
                            mdd_ok = m["mdd"] >= bh["mdd"] - 1e-9
                            bear_ok = mb["total_ret"] >= bhb["total_ret"] - 1e-9 or mb["mdd"] >= bhb["mdd"]
                            sweep_rows.append(dict(
                                scen=scen, U1=U1, D2=D2, D1=D1, d1_exit=ud, stop=stop,
                                ret=m["total_ret"], mdd=m["mdd"], sharpe=m["sharpe"],
                                trades=ntr, mdd_ok=bool(mdd_ok), bear_ok=bool(bear_ok)))

    # rank: prefer configs that keep MDD <= B&H; then max (return, sharpe)
    valid = [r for r in sweep_rows if r["mdd_ok"]]
    pool = valid if valid else sweep_rows
    pool.sort(key=lambda r: (r["ret"], r["sharpe"]), reverse=True)
    print(f"\nTop 12 configs (of {len(sweep_rows)} swept; {len(valid)} MDD≤B&H):")
    print(f"  {'scen':16s} {'U1':>5s} {'D2':>6s} {'d1x':>5s} {'stop':>5s} "
          f"{'ret':>9s} {'MDD':>7s} {'Sh':>5s} {'trd':>4s} bear")
    for r in pool[:12]:
        print(f"  {r['scen']:16s} {r['U1']:5.2f} {r['D2']:6.2f} {str(r['d1_exit'])[:1]:>5s} "
              f"{('-%.0f%%'%(r['stop']*100)) if r['stop']<1 else 'none':>5s} "
              f"{r['ret']*100:+8.1f}% {r['mdd']*100:6.1f}% {r['sharpe']:5.2f} "
              f"{r['trades']:4d}  {'Y' if r['bear_ok'] else 'n'}")

    # also show the best *Sharpe*-ranked and best *lowest-MDD-with-decent-return*
    by_sharpe = sorted(valid or sweep_rows, key=lambda r: r["sharpe"], reverse=True)[:5]
    print("\nTop 5 by Sharpe (MDD≤B&H):")
    for r in by_sharpe:
        print(f"  {r['scen']:16s} U1={r['U1']:.2f} D2={r['D2']:.2f} d1x={str(r['d1_exit'])[:1]} "
              f"stop={('-%.0f%%'%(r['stop']*100)) if r['stop']<1 else 'none':>5s} "
              f"ret={r['ret']*100:+7.1f}% MDD={r['mdd']*100:6.1f}% Sh={r['sharpe']:.2f} trd={r['trades']}")

    # Two representative retrained candidates per the brief's joint objective:
    #   RET  = best full-history return achievable on the extended window
    #   RISK = best Sharpe / lowest drawdown achievable ("minimise losses")
    ret_pick = pool[0]
    risk_pick = by_sharpe[0]
    print(f"\nRET  pick (max full-history return): {ret_pick}")
    print(f"RISK pick (max Sharpe, MDD≤B&H)   : {risk_pick}")

    def per_all(pick):
        p, s, _n = preds_cache[pick["scen"]]
        return {lbl: all_periods(p, s, col, pick["U1"], pick["D2"], pick["D1"],
                                 pick["d1_exit"], pick["stop"]) for lbl, col in TRADED}

    ret_res = per_all(ret_pick)
    risk_res = per_all(risk_pick)

    print("\n" + "=" * 100)
    print("PER-PERIOD: committed  vs  retrained-RET  vs  retrained-RISK  (return / MDD / Sharpe)")
    print("=" * 100)
    for lbl, col in TRADED:
        print(f"\n{lbl}:")
        for plabel, _s, _e in CFG.periods:
            b = base_res[lbl][plabel]; r = ret_res[lbl][plabel]; k = risk_res[lbl][plabel]
            print(f"  {plabel:38s}")
            print(f"      committed     : {fmt(b['strategy'])}  wr={b['win_rate']:.0f}% n={b['n_trades']}")
            print(f"      retrain-RET   : {fmt(r['strategy'])}  wr={r['win_rate']:.0f}% n={r['n_trades']}")
            print(f"      retrain-RISK  : {fmt(k['strategy'])}  wr={k['win_rate']:.0f}% n={k['n_trades']}")
            print(f"      buy&hold      : {fmt(b['buy_hold'])}")

    def pick_meta(pick, res):
        return dict(scenario=pick["scen"], train_end="2025-12-31",
                    thresholds=dict(U1=pick["U1"], D2=pick["D2"], D1=pick["D1"],
                                    use_d1_exit=pick["d1_exit"], stop=pick["stop"]),
                    full_hist=dict(ret=pick["ret"], mdd=pick["mdd"], sharpe=pick["sharpe"],
                                   trades=pick["trades"]),
                    results=res)

    dump = dict(
        data_range=[str(daily.index.min().date()), str(daily.index.max().date())],
        baseline=dict(train_end="2020-12-31", train_rows=base_n, thresholds=LIVE,
                      results=base_res),
        retrain_RET=pick_meta(ret_pick, ret_res),
        retrain_RISK=pick_meta(risk_pick, risk_res),
        sweep_top=pool[:20])
    Path("data/xle/retrain_2025_eval.json").write_text(json.dumps(dump, indent=2, default=str))
    print("\nSaved → data/xle/retrain_2025_eval.json")


if __name__ == "__main__":
    main()
