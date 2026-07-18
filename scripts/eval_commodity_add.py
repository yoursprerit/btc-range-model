"""Does adding ANOTHER commodity sleeve (beyond gold GLDM+, energy XLE+ and
strategic-metals REMX) improve the Overall combined back-test?

Candidates — the liquid US-listed commodity / commodity-equity ETFs not already
in the universe:

    SLV   silver bullion              SIL   silver miners
    CPER  copper futures              COPX  copper miners
    URA   uranium miners              PPLT  platinum
    PALL  palladium                   DBA   agriculture (re-test)
    DBC   broad commodity basket      UNG   natural gas
    LIT   lithium & battery metals    XME   diversified metals & mining

Method (mirrors eval_leveraged_add.py / the REMX tuning convention):

1.  Each candidate gets its own signal, chosen from the repo's long/flat trend
    family (`ma` / `dual_ma` / `macd` / `ma_vol` × {−5% stop, no stop}) — the
    exact grid the existing ETF apps were tuned on — selected by FULL-PERIOD
    (2016→now) Sharpe, the same "tuned to maximise full-period results"
    convention REMX ships with.
2.  The winning sleeve's OOS (2021→now) long/flat return stream is folded into
    the live universe and the cross-asset weights are re-optimised with the
    IDENTICAL optimize_weights() call the app uses, for every risk profile.
3.  Deltas are reported vs the same-baseline optimise, plus the deterministic
    equal-weight blend (clean attributable signal, no optimiser noise).
4.  Candidates that individually help are then tested jointly.
5.  MARGINAL TEST (noise-free): each candidate is blended into the FROZEN
    baseline optimal book at fixed 2/5/10% weights — pure arithmetic, no
    Monte-Carlo re-search, so the delta is 100% attributable to the candidate.
    (Adding a column changes the optimiser's search dimension, which moves the
    MC optimum by ±hundreds of pp regardless of the candidate — the same
    search-noise the QQQ test in SOXL_ERX_ADDITION_EVAL.md documented.)
6.  GOLD-SIGNAL SIBLINGS: SLV / SIL / AGQ (2× silver) traded off the gold
    app's tuned divergence signal — the exact architecture that shipped
    GDX/UGL/NUGT — since the precious-metals complex is the one place a new
    commodity can ride an already-strong signal instead of a weak new one.

    python scripts/eval_commodity_add.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "app"))
sys.path.insert(0, str(_REPO))

import overall_core as oc      # noqa: E402
import ticker_core as tc       # noqa: E402
import backtest_ticker as bt   # noqa: E402
from ticker_config import TickerConfig  # noqa: E402

OOS_START = "2021-01-01"
FULL_START = "2016-01-01"      # selection window start (200d warmup from 2015)

CANDIDATES = [
    # key    yahoo   name
    ("SLV",  "SLV",  "Silver bullion"),
    ("SIL",  "SIL",  "Silver miners"),
    ("CPER", "CPER", "Copper futures"),
    ("COPX", "COPX", "Copper miners"),
    ("URA",  "URA",  "Uranium miners"),
    ("PPLT", "PPLT", "Platinum"),
    ("PALL", "PALL", "Palladium"),
    ("DBA",  "DBA",  "Agriculture"),
    ("DBC",  "DBC",  "Broad commodity"),
    ("UNG",  "UNG",  "Natural gas"),
    ("LIT",  "LIT",  "Lithium & battery"),
    ("XME",  "XME",  "Metals & mining"),
]

# the repo's standard trend-engine grid (every existing ETF app ships one of
# these): label -> config field overrides
ENGINES = {
    "dual_ma 50/200": dict(strategy_mode="dual_ma", ma_window=200, ma_fast=50,  ma_slow=200),
    "dual_ma 25/100": dict(strategy_mode="dual_ma", ma_window=100, ma_fast=25,  ma_slow=100),
    "ma 150":         dict(strategy_mode="ma",      ma_window=150),
    "macd 10/20/9":   dict(strategy_mode="macd",    ma_window=50, macd_fast=10, macd_slow=20, macd_signal=9),
    "ma_vol 50":      dict(strategy_mode="ma_vol",  ma_window=50, vol_win=20, vol_med_win=252, vol_k=1.0),
}
STOPS = [0.05, 1.0]            # −5% fixed stop · signal-exit only


def cand_cfg(key: str, sym: str, name: str, **over) -> TickerConfig:
    base = dict(strategy_mode="dual_ma", ma_window=200, ma_fast=50, ma_slow=200,
                fixed_stop=0.05)
    base.update(over)
    return TickerConfig(
        key=key, name=name, blurb=name, emoji="🧪",
        accent="#64748b", accent_dark="#334155", accent_bg="#f8fafc", accent_bg2="#f1f5f9",
        primary_symbol=sym, macro_syms={}, extra_syms={},
        sentiment=[("px_close", "mom", +1.0)], sentiment_label=f"{key} sentiment",
        traded_assets=[(key, "px_close")], asset_labels={"px_close": name},
        strategy_name=f"{key} trend", fetch_start="2015-01-01", oos_start=OOS_START,
        **base)


def sleeve_result(key: str, cfg: TickerConfig, r: dict) -> dict:
    """Minimal result dict — returns_matrix/position_matrix need key/ret/pos_series."""
    dates = pd.to_datetime(pd.Series(r["dates"]))
    strat = np.asarray(r["strat"], float)
    return dict(
        key=key, parent=key, kind="core",
        ret=pd.Series(np.diff(strat) / strat[:-1], index=dates.iloc[1:]).rename(key),
        pos_series=pd.Series(np.asarray(r["pos"], float), index=dates).rename(key),
        metrics=bt._metrics(strat, r["dates"]),
    )


def optimise(results, extra_kind: dict[str, str] | None = None):
    """The exact per-profile optimise the app / other evals run, with candidate
    keys (absent from ASSET_META) capped at their kind's per-profile cap."""
    rets = oc.returns_matrix(results)
    pos = oc.position_matrix(results, rets.index)
    out = {}
    for n, p in oc.RISK_PROFILES.items():
        caps = oc.caps_for(n)
        for k, kind in (extra_kind or {}).items():
            caps[k] = p["caps"][kind]
        out[n] = oc.optimize_weights(rets, caps=caps, pos=pos,
                                     sata_daily=oc.SATA_DAILY,
                                     mdd_floor=p["mdd_floor"],
                                     objective=p["objective"])
    return out


def line(o, b, keys):
    w = " ".join(f"{k} {o['weights'].get(k, 0) * 100:4.1f}%" for k in keys)
    return (f"    ret {b['total_ret']*100:7.1f}%→{o['total_ret']*100:7.1f}% "
            f"({(o['total_ret']-b['total_ret'])*100:+7.1f}pp)  "
            f"MDD {b['mdd']*100:6.2f}%→{o['mdd']*100:6.2f}%  "
            f"Sharpe {b['sharpe']:.3f}→{o['sharpe']:.3f} ({o['sharpe']-b['sharpe']:+.3f})  [{w}]")


# ── 1. candidate sleeves: fetch once, sweep the engine grid ───────────────
print("Sweeping candidate commodity sleeves…", flush=True)
sleeves: dict[str, dict] = {}     # key -> dict(cfg, result, full, oos, engine)
for key, sym, name in CANDIDATES:
    probe = cand_cfg(key, sym, name)
    try:
        daily = tc.fetch_daily(probe)
    except Exception as e:
        print(f"  {key:5s} FETCH FAILED: {e}", flush=True)
        continue
    if daily is None or daily.empty or len(daily) < 400:
        print(f"  {key:5s} insufficient history ({0 if daily is None else len(daily)} bars)", flush=True)
        continue
    preds = bt.build_predictions(probe, daily)
    sig = bt.precompute_signals(probe, preds)
    best = None
    for eng_label, over in ENGINES.items():
        for stop in STOPS:
            cfg = cand_cfg(key, sym, name, fixed_stop=stop, **over)
            try:
                rf = bt.run_strategy(cfg, preds, sig, "px_close", oos_start=FULL_START)
            except Exception:
                continue
            if len(rf.get("dates", [])) < 250:
                continue
            mf = bt._metrics(np.asarray(rf["strat"], float), rf["dates"])
            tag = f"{eng_label} · stop {'none' if stop >= 0.999 else f'-{stop:.0%}'}"
            if best is None or mf["sharpe"] > best["full"]["sharpe"]:
                best = dict(cfg=cfg, engine=tag, full=mf)
    if best is None:
        print(f"  {key:5s} no viable engine", flush=True)
        continue
    ro = bt.run_strategy(best["cfg"], preds, sig, "px_close", oos_start=OOS_START)
    mo = bt._metrics(np.asarray(ro["strat"], float), ro["dates"])
    bh = bt._metrics(np.asarray(ro["bh"], float), ro["dates"])
    sleeves[key] = dict(**best, oos=mo, bh=bh, result=sleeve_result(key, best["cfg"], ro))
    f, o = best["full"], mo
    print(f"  {key:5s} {best['engine']:28s} full {f['total_ret']*100:6.0f}% "
          f"Sh {f['sharpe']:5.2f} | OOS {o['total_ret']*100:6.0f}% "
          f"MDD {o['mdd']*100:4.0f}% Sh {o['sharpe']:5.2f} "
          f"(B&H OOS {bh['total_ret']*100:5.0f}%)", flush=True)

# ── 2. baseline universe + per-candidate re-optimise ─────────────────────
print("\nRunning baseline universe (live fetch, ~60-90s)…", flush=True)
base = oc.run_universe()
print(f"  {len(base)} instruments: {[r['key'] for r in base]}", flush=True)
if oc._LAST_ERRORS:
    print("  LOAD ERRORS:", oc._LAST_ERRORS, flush=True)
opt_base = optimise(base)

summary = {}
for key, sl in sleeves.items():
    print("\n" + "=" * 92 + f"\n### +{key} ({sl['engine']})\n" + "=" * 92, flush=True)
    o2 = optimise(base + [sl["result"]], extra_kind={key: "core"})
    for prof in ("Balanced", "Growth", "Aggressive"):
        print(f"  [{prof:10s} optimal]", flush=True)
        print(line(o2[prof]["optimal"], opt_base[prof]["optimal"], [key]), flush=True)
    print("  [equal-weight — deterministic, clean attributable signal]", flush=True)
    print(line(o2["Aggressive"]["equal"], opt_base["Aggressive"]["equal"], [key]), flush=True)
    a, b = o2["Aggressive"]["optimal"], opt_base["Aggressive"]["optimal"]
    g, bg = o2["Growth"]["optimal"], opt_base["Growth"]["optimal"]
    s, bs = o2["Balanced"]["optimal"], opt_base["Balanced"]["optimal"]
    summary[key] = dict(
        d_ret_aggr=a["total_ret"] - b["total_ret"],
        d_sh_aggr=a["sharpe"] - b["sharpe"],
        d_ret_growth=g["total_ret"] - bg["total_ret"],
        d_sh_bal=s["sharpe"] - bs["sharpe"],
        wt_aggr=a["weights"].get(key, 0.0))

# ── 3. joint test of the individual winners ──────────────────────────────
winners = [k for k, d in summary.items()
           if (d["d_ret_aggr"] > 0.02 or d["d_sh_bal"] > 0.02) and d["wt_aggr"] > 0.01]
if len(winners) > 1:
    print("\n" + "=" * 92 + f"\n### JOINT: +{' +'.join(winners)}\n" + "=" * 92, flush=True)
    res = base + [sleeves[k]["result"] for k in winners]
    o2 = optimise(res, extra_kind={k: "core" for k in winners})
    for prof in ("Balanced", "Growth", "Aggressive"):
        print(f"  [{prof:10s} optimal]", flush=True)
        print(line(o2[prof]["optimal"], opt_base[prof]["optimal"], winners), flush=True)
    print("  [equal-weight]", flush=True)
    print(line(o2["Aggressive"]["equal"], opt_base["Aggressive"]["equal"], winners), flush=True)

# ── 4. ranked summary ─────────────────────────────────────────────────────
print("\n" + "=" * 92 + "\n### SUMMARY (sorted by Aggressive Δreturn)\n" + "=" * 92, flush=True)
print(f"  {'cand':5s} {'engine':30s} {'ΔretAggr':>9s} {'ΔshAggr':>8s} "
      f"{'ΔretGrw':>8s} {'ΔshBal':>7s} {'wtAggr':>7s}", flush=True)
for k in sorted(summary, key=lambda k: -summary[k]["d_ret_aggr"]):
    d = summary[k]
    print(f"  {k:5s} {sleeves[k]['engine']:30s} {d['d_ret_aggr']*100:+8.1f}pp "
          f"{d['d_sh_aggr']:+8.3f} {d['d_ret_growth']*100:+7.1f}pp "
          f"{d['d_sh_bal']:+7.3f} {d['wt_aggr']*100:6.1f}%", flush=True)


# ── 5. deterministic marginal test (no Monte-Carlo noise) ─────────────────
def sleeve_stream(res: dict, index: pd.Index) -> pd.Series:
    """One sleeve's daily return WITH idle capital earning SATA — the same
    treatment _combine gives a deployed weight that is out of the market."""
    r = res["ret"].reindex(index)
    p = res["pos_series"].reindex(index).fillna(0.0)
    avail = r.notna()
    out = r.fillna(0.0) + oc.SATA_DAILY * (1.0 - p) * avail
    out[~avail] = np.nan
    return out


def marginal(P: pd.Series, C: pd.Series, w: float) -> dict:
    """(1−w)·baseline-book + w·candidate, candidate weight parked in SATA
    before the candidate's history begins."""
    c = C.fillna(oc.SATA_DAILY)
    return oc.curve_metrics(oc._equity((1.0 - w) * P + w * c))


rets_b = oc.returns_matrix(base)
pos_b = oc.position_matrix(base, rets_b.index)
print("\n" + "=" * 92, flush=True)
print("### MARGINAL TEST — candidate blended into the FROZEN baseline optimal book", flush=True)
print("### (pure arithmetic at fixed weights; no optimiser re-search)", flush=True)
print("=" * 92, flush=True)
for prof in ("Balanced", "Aggressive"):
    wts = opt_base[prof]["optimal"]["weights"]
    wv = np.array([wts.get(c, 0.0) for c in rets_b.columns])
    P = oc._combine(rets_b, wv, pos_b, oc.SATA_DAILY)
    mb = oc.curve_metrics(oc._equity(P))
    print(f"\n  [{prof}] baseline book: ret {mb['total_ret']*100:.1f}%  "
          f"MDD {mb['mdd']*100:.2f}%  Sharpe {mb['sharpe']:.3f}", flush=True)
    print(f"  {'cand':5s} " + " | ".join(f"{f'w={w:.0%}':>26s}" for w in (0.02, 0.05, 0.10)), flush=True)
    for k, sl in sleeves.items():
        C = sleeve_stream(sl["result"], rets_b.index)
        cells = []
        for w in (0.02, 0.05, 0.10):
            m = marginal(P, C, w)
            cells.append(f"{(m['total_ret']-mb['total_ret'])*100:+6.1f}pp "
                         f"Sh{m['sharpe']-mb['sharpe']:+.3f}")
        print(f"  {k:5s} " + " | ".join(f"{c:>26s}" for c in cells), flush=True)


# ── 6. precious-metal siblings off the tuned GOLD signal ──────────────────
# The GDX/UGL/NUGT architecture: trade the sibling off the parent's strong
# signal, never its own.  Stops mirror each sibling's shipped analogue:
# SLV −3% (like GLDM), SIL −5% (like NUGT), AGQ signal-exit-only (like UGL).
GOLD_SIBS = [("SLV", "SLV", "core", 0.03), ("SIL", "SIL", "beta", 0.05),
             ("AGQ", "AGQ", "lev", 1.0)]

print("\n" + "=" * 92, flush=True)
print("### GOLD-SIGNAL SIBLINGS — SLV / SIL / AGQ traded off the gold divergence signal", flush=True)
print("=" * 92, flush=True)
pc = oc.overall_config("GLDM")
gold_sleeves = {}
for key, sym, kind, stop in GOLD_SIBS:
    col = f"{sym.lower()}_close"
    cfg = replace(pc, extra_syms={**pc.extra_syms, sym.lower(): sym},
                  traded_assets=[(sym, col)], asset_labels={col: sym},
                  stop_by_asset={**getattr(pc, "stop_by_asset", {}), col: stop})
    try:
        sl = [s for s in oc.run_asset(cfg) if s["key"] == sym]
    except Exception as e:
        print(f"  {key:4s} FAILED: {e}", flush=True)
        continue
    if not sl:
        print(f"  {key:4s} no result", flush=True)
        continue
    m = sl[0]["metrics"]
    gold_sleeves[key] = dict(kind=kind, result=sl[0])
    print(f"  {key:4s} ({kind}, stop {'none' if stop >= 0.999 else f'-{stop:.0%}'}) own strat: "
          f"ret {m['total_ret']*100:6.0f}%  MDD {m['mdd']*100:5.0f}%  "
          f"Sharpe {m['sharpe']:.2f}", flush=True)

for key, g in gold_sleeves.items():
    print("\n" + "-" * 92 + f"\n  +{key} off gold signal — full re-optimise\n" + "-" * 92, flush=True)
    o2 = optimise(base + [g["result"]], extra_kind={key: g["kind"]})
    for prof in ("Balanced", "Growth", "Aggressive"):
        print(f"  [{prof:10s} optimal]", flush=True)
        print(line(o2[prof]["optimal"], opt_base[prof]["optimal"], [key]), flush=True)
    print("  [equal-weight — deterministic]", flush=True)
    print(line(o2["Aggressive"]["equal"], opt_base["Aggressive"]["equal"], [key]), flush=True)

print("\n  marginal test (frozen baseline optimal book):", flush=True)
for prof in ("Balanced", "Aggressive"):
    wts = opt_base[prof]["optimal"]["weights"]
    wv = np.array([wts.get(c, 0.0) for c in rets_b.columns])
    P = oc._combine(rets_b, wv, pos_b, oc.SATA_DAILY)
    mb = oc.curve_metrics(oc._equity(P))
    print(f"  [{prof}]", flush=True)
    for key, g in gold_sleeves.items():
        C = sleeve_stream(g["result"], rets_b.index)
        cells = []
        for w in (0.02, 0.05, 0.10):
            m = marginal(P, C, w)
            cells.append(f"w={w:.0%} {(m['total_ret']-mb['total_ret'])*100:+6.1f}pp "
                         f"Sh{m['sharpe']-mb['sharpe']:+.3f} "
                         f"MDD{(m['mdd']-mb['mdd'])*100:+5.2f}pp")
        print(f"    {key:4s} " + " | ".join(cells), flush=True)

print("\nDone.", flush=True)
