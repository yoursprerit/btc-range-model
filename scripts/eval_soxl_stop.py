"""Does ADDING a fixed stop to SOXL improve performance? (sleeve + Overall)

SOXL (3x semis) currently trades the SOXX 25/100 dual-MA with NO fixed stop
(stop_by_asset={"soxl_close": 1.0}); the inherited -5% stop was dropped because
it was far too tight for a 3x ETF.  This sweeps a RANGE of stop widths — none,
-30%, -25%, -20%, -15%, -10%, -5% — to test whether any stop (not just -5%)
improves the sleeve and the Overall blend.  Reports return, Sharpe, win-rate.

  PART 1 — SOXL sleeve (soxl_close on the SOXX signal), stop grid x periods.
  PART 2 — Overall combined portfolio: swap the SOXL sleeve for each stop and
           re-run the IDENTICAL optimize_weights() the app/build use.  SOXX and
           every other sleeve are untouched, so deltas are attributable to the
           SOXL-stop change alone.

    python scripts/eval_soxl_stop.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "app"))
sys.path.insert(0, str(_REPO))

import backtest_ticker as bt        # noqa: E402
import overall_core as oc           # noqa: E402
from ticker_config import get_config  # noqa: E402

STOPS = [(1.0, "none"), (0.30, "−30%"), (0.25, "−25%"), (0.20, "−20%"),
         (0.15, "−15%"), (0.10, "−10%"), (0.05, "−5%")]


def m(r):
    return bt._metrics(r["strat"], r["dates"])


def winrate(r):
    t = r["trades"]
    return float((t > 0).mean() * 100) if len(t) else 0.0


# ════════════════════════════════════════════════════════════════════════
# PART 1 — SOXL sleeve, stop grid
# ════════════════════════════════════════════════════════════════════════
print("Fetching SOXX+SOXL + building signals…", flush=True)
cfg = get_config("SOXX")
daily = oc._load_daily(cfg)
preds = bt.build_predictions(cfg, daily)
sig = bt.precompute_signals(cfg, preds)
oos_end = str(preds["target_date"].iloc[-1].date())

print("\n" + "=" * 92)
print(f"PART 1 — SOXL (soxl_close) on SOXX 25/100 dual-MA · OOS→{oos_end}")
print("=" * 92)
hdr = f"{'Stop':>6s} | {'Return':>10s} {'MDD':>8s} {'Sharpe':>7s} {'Win%':>6s} {'Trades':>7s}"

for plabel, s, e in cfg.periods:
    print("\n" + plabel + "\n" + hdr)
    print("  " + "-" * 56)
    for stop, tag in STOPS:
        r = bt.run_strategy(cfg, preds, sig, "soxl_close", oos_start=s, end=e, stop_pct=stop)
        sm = m(r)
        print(f"{tag:>6s} | {sm['total_ret']*100:+9.1f}% {sm['mdd']*100:+7.1f}% "
              f"{sm['sharpe']:7.2f} {winrate(r):5.0f}% {len(r['trades']):7d}")


# ════════════════════════════════════════════════════════════════════════
# PART 2 — Overall combined portfolio impact
# ════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 92)
print("PART 2 — Overall combined portfolio: SOXL stop sweep")
print("=" * 92)
print("Running full live universe (baseline = SOXL no stop)…", flush=True)
base = oc.run_universe()
print(f"  {len(base)} instruments (errors: {oc._LAST_ERRORS})", flush=True)


def optimise(results):
    rets = oc.returns_matrix(results)
    pos = oc.position_matrix(results, rets.index)
    return {n: oc.optimize_weights(rets, caps=oc.caps_for(n), pos=pos,
                                   sata_daily=oc.SATA_DAILY, mdd_floor=p["mdd_floor"],
                                   objective=p["objective"])
            for n, p in oc.RISK_PROFILES.items()}


def soxl_sleeve(stop):
    """Rebuild ONLY the SOXL sleeve with a given stop; swap into the universe."""
    cfg_v = replace(cfg, stop_by_asset={"soxl_close": stop})
    sl = [r for r in oc.run_asset(cfg_v) if r["key"] == "SOXL"]
    return [r for r in base if r["key"] != "SOXL"] + sl


# baseline optimise (SOXL no stop, as shipped)
opt_base = optimise(base)

print("\n  SOXL sleeve inside the universe + optimised portfolio (fundamental view on):")
print(f"  {'Stop':>6s} | {'sleeve ret':>10s} {'Sh':>5s} {'win':>5s} || "
      f"{'Agg ret':>9s} {'Agg MDD':>8s} {'Agg Sh':>6s} {'SOXL wt':>8s}")
print("  " + "-" * 82)
for stop, tag in STOPS:
    res = base if stop == 1.0 else soxl_sleeve(stop)
    sleeve = next(x for x in res if x["key"] == "SOXL")
    optv = opt_base if stop == 1.0 else optimise(res)
    agg = optv["Aggressive"]["optimal"]
    print(f"  {tag:>6s} | {sleeve['metrics']['total_ret']*100:+9.1f}% "
          f"{sleeve['metrics']['sharpe']:5.2f} {sleeve['win_rate']:4.0f}% || "
          f"{agg['total_ret']*100:+8.1f}% {agg['mdd']*100:+7.1f}% {agg['sharpe']:6.2f} "
          f"{agg['weights'].get('SOXL',0)*100:7.1f}%")

# clean deterministic equal-weight signal, no-stop vs a representative -20% stop
print("\n  Equal-weight blend (deterministic clean signal): none vs −20% vs −10%")
for stop, tag in ((1.0, "none"), (0.20, "−20%"), (0.10, "−10%")):
    res = base if stop == 1.0 else soxl_sleeve(stop)
    ew = optimise(res)["Aggressive"]["equal"] if stop != 1.0 else opt_base["Aggressive"]["equal"]
    print(f"    {tag:>5s}: ret {ew['total_ret']*100:+8.1f}%  MDD {ew['mdd']*100:+6.1f}%  "
          f"Sharpe {ew['sharpe']:.3f}")

print("\nDone.", flush=True)
