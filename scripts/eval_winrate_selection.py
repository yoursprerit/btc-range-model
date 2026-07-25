"""Win-rate-gated instrument selection for the Overall strategy — evaluation.

Read-only. Changes nothing. Reproduces the tables in
WINRATE_SELECTION_EVAL.md:

  * per-instrument trade win rates over the FULL backtest period (each
    instrument's own history, e.g. 2016->now for the standard tickers) and
    over the OOS windows (2021->now full OOS + Bear / Bull / Recent);
  * four candidate universes — ALL-18 baseline, A: full-period WR>=55%,
    B: OOS WR>=55%, C: WR>=55% in every OOS window with trades;
  * each universe re-optimised for every risk profile (Balanced / Growth /
    Aggressive) with the identical optimize_weights() call
    scripts/build_overall.py uses (same caps, SATA, drawdown floor,
    objective), fundamental=False so the criterion — not the conviction
    tilt — drives the comparison, across MC seeds 7 / 11 / 23.

    python scripts/eval_winrate_selection.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "app"))
sys.path.insert(0, str(_REPO))

import numpy as np
import pandas as pd

import overall_core as oc            # noqa: E402
import backtest_ticker as bt         # noqa: E402

THR = 55.0
SEEDS = (7, 11, 23)
# OOS sub-windows — match oc.COMBINED_PERIODS
WINDOWS = dict(bear=("2021-01-01", "2022-12-31"),
               bull=("2023-01-01", None),
               recent=("2025-01-01", None))


def _wr_from_log(trade_log, start, end):
    """Win rate over CLOSED trades whose exit falls in [start, end]."""
    s = pd.Timestamp(start) if start else pd.Timestamp("1900-01-01")
    e = pd.Timestamp(end) if end else pd.Timestamp("2100-01-01")
    sel = [t for t in (trade_log or []) if s <= pd.Timestamp(t["exit_date"]) <= e]
    if not sel:
        return None, 0
    return 100.0 * sum(float(t["ret"]) > 0 for t in sel) / len(sel), len(sel)


def _win_rates(results):
    """Full-period + OOS-window trade win rates for every instrument."""
    rows = {}
    for res in results:
        tl = (res.get("r") or {}).get("trade_log") or []
        row = dict(kind=res["kind"], oos_wr=res["win_rate"], oos_n=res["n_trades"])
        for wname, (ws, we) in WINDOWS.items():
            row[f"wr_{wname}"], row[f"n_{wname}"] = _wr_from_log(tl, ws, we)
        rows[res["key"]] = row

    # full-period reruns — ticker parents from each sibling's first valid bar
    for cfg in oc.all_configs():
        if cfg.key in ("BTC", "GLDM", "GDXM"):
            continue
        daily = oc._load_daily(cfg)
        if daily is None or daily.empty:
            continue
        preds = bt.build_predictions(cfg, daily)
        sig = bt.precompute_signals(cfg, preds)
        for label, col in cfg.traded_assets:
            if label not in rows or col not in preds.columns:
                continue
            valid = preds.loc[preds[col].notna(), "target_date"]
            r = bt.run_strategy(cfg, preds, sig, col,
                                oos_start=str(pd.Timestamp(valid.iloc[0]).date()))
            rows[label]["full_wr"] = (float((r["trades"] > 0).mean() * 100)
                                      if len(r["trades"]) else None)
            rows[label]["full_n"] = int(len(r["trades"]))
            rows[label]["full_start"] = str(pd.Timestamp(r["dates"][0]).date())

    # gold stack — same run_asset_sim dispatch the Gold app trades, full window
    import gldm_engine as ge
    import backtest_gldm as bg
    gd = ge._load_daily()
    gp = bg.build_predictions(gd)
    if "nugt_close" in gd.columns:
        gp = gp.copy()
        gp["nugt_close"] = gd["nugt_close"].reindex(gp.index)
    gsig = bg.precompute_signals(gp)
    for key in ("GLDM", "GDX", "UGL", "NUGT"):
        if key not in rows:
            continue
        r = bg.run_asset_sim(gp, gsig, key, oos_start="2015-01-01")
        rows[key]["full_wr"] = (float((r["trades"] > 0).mean() * 100)
                                if len(r["trades"]) else None)
        rows[key]["full_n"] = int(len(r["trades"]))
        rows[key]["full_start"] = str(pd.Timestamp(r["dates"][0]).date())

    # BTC stack: CT features begin 2024 -> full backtestable period == OOS
    for key in ("BTC", "MSTR", "MSTU", "ETH"):
        if key in rows:
            rows[key]["full_wr"] = rows[key]["oos_wr"]
            rows[key]["full_n"] = rows[key]["oos_n"]
            rows[key]["full_start"] = "(=OOS)"
    return rows


def main():
    print("Running universe (live fetch, ~30-90s)…", flush=True)
    results = oc.run_universe()
    keys = [r["key"] for r in results]
    rets = oc.returns_matrix(results)
    pos = oc.position_matrix(results, rets.index)
    print(f"  {len(keys)} instruments, {rets.index[0].date()} -> {rets.index[-1].date()}")

    rows = _win_rates(results)
    df = pd.DataFrame(rows).T
    print("\n=== TRADE WIN RATES (%; sub-windows = closed trades by exit date) ===")
    print(df[["kind", "full_start", "full_wr", "full_n", "oos_wr", "oos_n",
              "wr_bear", "n_bear", "wr_bull", "n_bull", "wr_recent", "n_recent"]]
          .to_string(float_format=lambda v: f"{v:.1f}"))

    def _sel(pred):
        return [k for k in keys if pred(rows[k])]

    universes = {
        "ALL-18 (baseline)": keys,
        "A: full-period WR>=55%": _sel(lambda r: (r.get("full_wr") or 0) >= THR),
        "B: OOS WR>=55%": _sel(lambda r: (r.get("oos_wr") or 0) >= THR),
        "C: every OOS window>=55%": _sel(
            lambda r: (r.get("oos_wr") or 0) >= THR and all(
                r.get(f"wr_{w}") is None or r[f"wr_{w}"] >= THR
                for w in WINDOWS)),
    }
    print("\n=== CANDIDATE UNIVERSES ===")
    for name, u in universes.items():
        print(f"  {name:26s} n={len(u):2d}  {u}")

    print("\n=== OPTIMAL BLEND BY UNIVERSE x PROFILE (fundamental=False) ===")
    for seed in SEEDS:
        print(f"\n--- MC seed {seed} ---")
        for uname, cols in universes.items():
            line = f"  {uname:26s}"
            for p in ("Balanced", "Growth", "Aggressive"):
                prof = oc.RISK_PROFILES[p]
                caps = {c: oc.caps_for(p)[c] for c in cols}
                o = oc.optimize_weights(rets[cols], caps=caps, seed=seed,
                                        pos=pos[cols], sata_daily=oc.SATA_DAILY,
                                        mdd_floor=prof["mdd_floor"],
                                        objective=prof["objective"],
                                        fundamental=False)
                m = o["optimal"]
                line += (f" | {p[:3]} {m['total_ret']*100:+8.1f}% "
                         f"Sh {m['sharpe']:.2f} DD {m['mdd']*100:5.1f}%")
            print(line, flush=True)

    print("\n=== DETERMINISTIC REFERENCE BLENDS (no MC noise) ===")
    for uname, cols in universes.items():
        caps = {c: oc.caps_for("Balanced")[c] for c in cols}
        o = oc.optimize_weights(rets[cols], caps=caps, seed=7, pos=pos[cols],
                                sata_daily=oc.SATA_DAILY,
                                objective="balanced", fundamental=False)
        e, rp = o["equal"], o["risk_parity"]
        print(f"  {uname:26s} EW {e['total_ret']*100:+8.1f}% DD {e['mdd']*100:6.1f}% "
              f"Sh {e['sharpe']:.2f} | RP {rp['total_ret']*100:+8.1f}% "
              f"DD {rp['mdd']*100:6.1f}% Sh {rp['sharpe']:.2f}")


if __name__ == "__main__":
    main()
