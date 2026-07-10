"""Does removing SOXX's −5% fixed stop (signal-only exits) improve performance?

Two questions, one script:

  PART 1 — SOXX standalone (own 25/100 dual-MA sleeve): compare the shipped
           −5% fixed stop vs no stop (signal-only exits) across every backtest
           period — return, MDD, Sharpe, win-rate, trades — plus the trade log
           so the mechanism (stop-outs = manufactured losing round-trips) is
           visible.

  PART 2 — Overall combined portfolio: swap the SOXX sleeve's −5% stop for no
           stop and re-run the IDENTICAL optimize_weights() the app/build use,
           for all three risk profiles.  SOXL (already no-stop) is untouched, so
           the delta is attributable to the SOXX-stop change alone.

    python scripts/eval_soxx_stop.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "app"))
sys.path.insert(0, str(_REPO))

import backtest_ticker as bt        # noqa: E402
import overall_core as oc           # noqa: E402
from ticker_config import get_config  # noqa: E402


def m(r):
    return bt._metrics(r["strat"], r["dates"])


def winrate(r):
    t = r["trades"]
    return float((t > 0).mean() * 100) if len(t) else 0.0


# ════════════════════════════════════════════════════════════════════════
# PART 1 — SOXX standalone: −5% stop vs no stop, every period
# ════════════════════════════════════════════════════════════════════════
print("Fetching SOXX + building signals…", flush=True)
cfg = get_config("SOXX")
daily = oc._load_daily(cfg)
preds = bt.build_predictions(cfg, daily)
sig = bt.precompute_signals(cfg, preds)
oos_end = str(preds["target_date"].iloc[-1].date())

print("\n" + "=" * 96)
print(f"PART 1 — SOXX (px_close) 25/100 dual-MA · own sleeve · OOS→{oos_end}")
print("=" * 96)
hdr = f"{'Period':34s} {'stop':>6s} | {'Return':>9s} {'MDD':>8s} {'Sharpe':>7s} {'Win%':>6s} {'Trades':>7s}"

for plabel, s, e in cfg.periods:
    print("\n" + plabel)
    print(hdr)
    print("  " + "-" * 90)
    bh_printed = False
    for stop, tag in ((0.05, "−5%"), (1.0, "none")):
        r = bt.run_strategy(cfg, preds, sig, "px_close", oos_start=s, end=e, stop_pct=stop)
        sm = m(r)
        print(f"{'':34s} {tag:>6s} | {sm['total_ret']*100:+8.1f}% {sm['mdd']*100:+7.1f}% "
              f"{sm['sharpe']:7.2f} {winrate(r):5.0f}% {len(r['trades']):7d}")
        if not bh_printed:
            bm = bt._metrics(r["bh"], r["dates"])
            print(f"{'  (buy & hold)':34s} {'':>6s} | {bm['total_ret']*100:+8.1f}% "
                  f"{bm['mdd']*100:+7.1f}% {bm['sharpe']:7.2f} {'':6s} {'':7s}")
            bh_printed = True

# Trade-log diff on the default OOS window — shows WHY win-rate moves
print("\n" + "-" * 96)
print("Trade log — default OOS window (2021→now): stop-outs are the manufactured losers")
print("-" * 96)
for stop, tag in ((0.05, "−5% stop"), (1.0, "no stop")):
    r = bt.run_strategy(cfg, preds, sig, "px_close", stop_pct=stop)
    log = r["trade_log"]
    wins = sum(1 for t in log if t["ret"] > 0)
    stops = sum(1 for t in log if "stop" in t["reason"])
    print(f"\n  {tag}: {len(log)} trades, {wins} green, win-rate {winrate(r):.0f}%, "
          f"{stops} stop-out exits")
    for t in log:
        d0 = str(np.datetime_as_string(np.datetime64(t['entry_date']), unit='D'))
        d1 = str(np.datetime_as_string(np.datetime64(t['exit_date']), unit='D'))
        flag = "WIN " if t["ret"] > 0 else "LOSS"
        print(f"      {d0} → {d1}  {t['ret']*100:+7.1f}%  [{flag}]  exit: {t['reason']}")


# ════════════════════════════════════════════════════════════════════════
# PART 2 — Overall combined portfolio impact
# ════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 96)
print("PART 2 — Overall combined portfolio: SOXX −5% stop  vs  SOXX no stop")
print("=" * 96)
print("Running full live universe…", flush=True)
base = oc.run_universe()
if oc._LAST_ERRORS:
    print("  load errors:", oc._LAST_ERRORS)
print(f"  {len(base)} instruments: {[r['key'] for r in base]}", flush=True)

# Build the no-stop SOXX sleeve (SOXL kept no-stop; only SOXX px_close changes).
cfg_ns = replace(cfg, stop_by_asset={"px_close": 1.0, "soxl_close": 1.0})
soxx_ns = [r for r in oc.run_asset(cfg_ns) if r["key"] == "SOXX"]
assert soxx_ns, "no-stop SOXX sleeve failed to build"
variant = [r for r in base if r["key"] != "SOXX"] + soxx_ns


def optimise(results):
    rets = oc.returns_matrix(results)
    pos = oc.position_matrix(results, rets.index)
    return {n: oc.optimize_weights(rets, caps=oc.caps_for(n), pos=pos,
                                   sata_daily=oc.SATA_DAILY, mdd_floor=p["mdd_floor"],
                                   objective=p["objective"])
            for n, p in oc.RISK_PROFILES.items()}


# SOXX sleeve own-metrics, both ways
def soxx_metrics(results):
    r = next(x for x in results if x["key"] == "SOXX")
    return r["metrics"], r["win_rate"], r["n_trades"]

b_met, b_wr, b_nt = soxx_metrics(base)
v_met, v_wr, v_nt = soxx_metrics(variant)
print("\n  SOXX sleeve inside the universe (own strategy stream):")
print(f"    −5% stop : ret {b_met['total_ret']*100:+7.1f}%  MDD {b_met['mdd']*100:+6.1f}%  "
      f"Sharpe {b_met['sharpe']:.2f}  win {b_wr:.0f}%  trades {b_nt}")
print(f"    no stop  : ret {v_met['total_ret']*100:+7.1f}%  MDD {v_met['mdd']*100:+6.1f}%  "
      f"Sharpe {v_met['sharpe']:.2f}  win {v_wr:.0f}%  trades {v_nt}")

opt_b = optimise(base)
opt_v = optimise(variant)

print("\n  Combined optimised portfolio (identical optimize_weights, fundamental view on):")
print(f"    {'Profile':11s} {'variant':9s} | {'Return':>9s} {'MDD':>8s} {'Sharpe':>7s} {'SOXX wt':>8s} {'SOXL wt':>8s}")
print("    " + "-" * 74)
for prof in ("Balanced", "Growth", "Aggressive"):
    for tag, opt in (("−5% stop", opt_b), ("no stop", opt_v)):
        o = opt[prof]["optimal"]
        sx = o["weights"].get("SOXX", 0.0) * 100
        sl = o["weights"].get("SOXL", 0.0) * 100
        print(f"    {prof:11s} {tag:9s} | {o['total_ret']*100:+8.1f}% {o['mdd']*100:+7.1f}% "
              f"{o['sharpe']:7.2f} {sx:7.1f}% {sl:7.1f}%")
    # delta line
    ob, ov = opt_b[prof]["optimal"], opt_v[prof]["optimal"]
    print(f"    {'':11s} {'Δ':9s} | {(ov['total_ret']-ob['total_ret'])*100:+8.1f}pp "
          f"{(ov['mdd']-ob['mdd'])*100:+7.1f}pp {ov['sharpe']-ob['sharpe']:+7.2f} "
          f"{(ov['weights'].get('SOXX',0)-ob['weights'].get('SOXX',0))*100:+7.1f}pp")

# Deterministic equal-weight — clean attribution, no optimiser/fundamental noise
print("\n  Equal-weight blend (deterministic, clean attributable signal):")
eb = opt_b["Aggressive"]["equal"]; ev = opt_v["Aggressive"]["equal"]
print(f"    −5% stop : ret {eb['total_ret']*100:+8.1f}%  MDD {eb['mdd']*100:+6.1f}%  Sharpe {eb['sharpe']:.3f}")
print(f"    no stop  : ret {ev['total_ret']*100:+8.1f}%  MDD {ev['mdd']*100:+6.1f}%  Sharpe {ev['sharpe']:.3f}")
print(f"    Δ        : ret {(ev['total_ret']-eb['total_ret'])*100:+8.1f}pp  "
      f"MDD {(ev['mdd']-eb['mdd'])*100:+6.1f}pp  Sharpe {ev['sharpe']-eb['sharpe']:+.3f}")

print("\nDone.", flush=True)
