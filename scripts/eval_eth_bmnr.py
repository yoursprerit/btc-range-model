"""Does ETH (and/or BMNR) earn a place in the strategy?  Reproduces
ETH_BMNR_STRATEGY_EVAL.md.

  Part 1 — trade ETH (spot Ethereum, on BTC's own 12:00-UTC bars) and BMNR
           (Bitmine Immersion, the ETH-treasury equity) on the BTC app's
           signals: the CT-divergence engine (both live gates, per-asset stop
           grid, exactly as MSTR/MSTU are traded) and the simple BTC>SMA rule.
           Also quantifies the FILL-TIMING look-ahead that inflated the earlier
           ETHA-based figures (the CT signal for bar D is only knowable at
           D+1 12:00 UTC, but equity/ETF sleeves fill at date D's US close).
  Part 2 — search a robust strategy on ETH's own price action (SMA / dual-MA /
           MACD / momentum grids, also ETH-USD as the signal parent), then
           validate: 5-fold CV, expanding walk-forward, cost sensitivity, the
           long ETH-USD history, and BMNR's post-squeeze era.
  Part 3 — fold the best sleeves into the Overall universe and re-optimise with
           the IDENTICAL optimize_weights() per risk profile (plus MC-seed
           noise band, deterministic equal-weight check, 2025→now slice).

    python scripts/eval_eth_bmnr.py

Data is fetched live (Yahoo chart API via the proxy-friendly requests path);
the BTC signal comes from the committed CT feature CSV + trained model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
for _p in (str(_REPO), str(_REPO / "app"), str(_REPO / "scripts")):
    sys.path.insert(0, _p)

from pull_backtest_data import _yf_requests          # noqa: E402
import backtest_trailing_stop as T                   # noqa: E402  (loads CT model)
import btc_ct_engine as E                            # noqa: E402
import overall_core as oc                            # noqa: E402

TD = 252


# ── data ───────────────────────────────────────────────────────────────────
def px(ticker: str) -> pd.Series:
    s = _yf_requests(ticker, "2023-11-01")["Close"].dropna()
    s.name = ticker
    return s


def eth_12utc() -> pd.Series:
    """Spot ETH on BTC's 12:00-UTC bar boundary (the shipped sleeve's prices).
    Same anchor as the signal bar, so the same-bar fill is genuine."""
    s = pd.read_csv(_REPO / "data" / "backtest" / "eth_usd_daily.csv",
                    index_col=0, parse_dates=True)["close"].astype(float)
    s.name = "ETH"
    return s


# ── CT signal state (built once) ───────────────────────────────────────────
def ct_state(start="2024-01-01") -> dict:
    rf = T.load_raw_features()
    preds = T.build_preds_offline(rf)
    comp = T.prep(rf, preds, start, str(rf.index[-1].date()))
    return dict(comp=comp, sigs=E.compute_sigs_pure(comp),
                dates=pd.DatetimeIndex(comp["target_date"]))


def run_ct_sleeve(st: dict, asset: pd.Series, gate="ma", stop=None) -> dict:
    """An asset through the BTC app's CT engine (same-bar, intrabar stop),
    exactly as MSTR/MSTU are traded off the parent BTC signal."""
    sigs = dict(st["sigs"])
    sigs["tf2_entry"] = sigs["tf2_entry_ma"] if gate == "ma" else sigs["tf2_entry_pure"]
    arr = asset.reindex(st["dates"]).ffill().to_numpy(float)
    return E._run_bt(st["dates"], arr, sigs, stop, asset.index[0])


# ── simple-rule engine (prior-eval methodology) ────────────────────────────
def metrics(eq: pd.Series) -> dict:
    eq = eq.dropna()
    if len(eq) < 2:
        return dict(total=np.nan, cagr=np.nan, mdd=np.nan, sharpe=np.nan)
    r = eq.pct_change().dropna()
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    return dict(total=eq.iloc[-1] / eq.iloc[0] - 1,
                cagr=(eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1,
                mdd=float((eq / eq.cummax() - 1).min()),
                sharpe=float(r.mean() / (r.std() + 1e-12) * np.sqrt(TD)))


def simple_bt(asset: pd.Series, sig: pd.Series, cost=0.001, extra_lag=0) -> dict:
    """Long/flat: signal from closes ≤ t applies to the t→t+1 return (next-bar),
    `cost` charged per switch.  The signal is ffilled onto the asset's calendar;
    extra_lag adds days of delay (cross-calendar honesty sensitivity)."""
    s = (sig.astype(float).reindex(asset.index.union(sig.index)).ffill()
         .reindex(asset.index).fillna(0.0))
    pos = s.shift(1 + extra_lag).fillna(0.0)
    ret = asset.pct_change().fillna(0.0)
    sw = pos.diff().abs().fillna(0.0)
    strat = pos * ret - sw * cost
    m = metrics((1 + strat).cumprod())
    m.update(expo=float(pos.mean()), trades=int((sw > 0).sum()), strat=strat, pos=pos)
    return m


def rule_sma(c, n):       return c > c.rolling(n).mean()
def rule_dma(c, f, s):    return c.rolling(f).mean() > c.rolling(s).mean()
def rule_mom(c, n):       return c > c.shift(n)
def rule_macd(c, f, s, g):
    m = c.ewm(span=f, adjust=False).mean() - c.ewm(span=s, adjust=False).mean()
    return m > m.ewm(span=g, adjust=False).mean()


def grid(close):
    out = {}
    for n in (5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150, 200):
        out[f"SMA{n}"] = rule_sma(close, n)
    for f, s in ((10, 50), (20, 50), (20, 100), (25, 100), (50, 100), (10, 100), (50, 200)):
        out[f"DMA{f}/{s}"] = rule_dma(close, f, s)
    for p in ((10, 20, 9), (12, 26, 9)):
        out[f"MACD{p[0]}/{p[1]}/{p[2]}"] = rule_macd(close, *p)
    for n in (20, 60, 90):
        out[f"MOM{n}"] = rule_mom(close, n)
    return out


def kfold_line(name, strat, bh, k=5):
    # each stream is split on its own calendar (the CT sleeve runs on 7-day
    # crypto bars, B&H on exchange sessions) — folds are contiguous fifths
    fr = np.array_split(np.arange(len(strat)), k)
    fb = np.array_split(np.arange(len(bh)), k)
    sh, rets, beats = [], [], 0
    for f, g in zip(fr, fb):
        s, b = strat.iloc[f], bh.iloc[g]
        sh.append(float(s.mean() / (s.std() + 1e-12) * np.sqrt(TD)))
        rets.append(float((1 + s).prod() - 1))
        beats += rets[-1] >= float((1 + b).prod() - 1)
    print(f"  {name:32s} mean fold Sh {np.mean(sh):+5.2f}  worst {min(sh):+5.2f}  "
          f"beats B&H {beats}/{k}  fold rets: "
          + " ".join(f"{r*100:+6.1f}%" for r in rets))


def walkforward(asset, src, ns=(10, 15, 20, 25, 30, 40, 50, 60, 80, 100), k=5, lag=0):
    folds = np.array_split(np.arange(len(asset)), k)
    oos, picks = [], []
    for i in range(1, k):
        t_end = asset.index[folds[i - 1][-1]]
        best = max(ns, key=lambda n: np.nan_to_num(
            simple_bt(asset.loc[:t_end], rule_sma(src, n), extra_lag=lag)["sharpe"], nan=-9))
        picks.append(best)
        oos.append(simple_bt(asset, rule_sma(src, best), extra_lag=lag)["strat"].iloc[folds[i]])
    s = pd.concat(oos)
    return (float((1 + s).prod() - 1),
            float(s.mean() / (s.std() + 1e-12) * np.sqrt(TD)), picks)


def main() -> int:
    print("Loading ETH (12:00-UTC bars) · fetching BMNR / ETHA / ETH-USD …", flush=True)
    eth = eth_12utc()                 # the shipped sleeve
    bmnr = px("BMNR")
    etha = px("ETHA")                 # retired ETF sleeve — kept for the contrast row
    ethyf = px("ETH-USD")             # midnight-UTC ETH, for the own-price grid
    print("Building CT predictions (committed features + trained model)…", flush=True)
    st = ct_state()
    btc = pd.Series(st["comp"]["actual_close"].to_numpy(float), index=st["dates"])

    # ═══ Part 1 — ETHA / BMNR on the BTC signal ═══════════════════════════
    print("\n" + "═" * 90 + "\nPART 1 — ETHA / BMNR on BTC signals\n" + "═" * 90)
    for name, a in (("ETH", eth), ("BMNR", bmnr)):
        print(f"\n=== {name}  ({a.index[0].date()} → {a.index[-1].date()}, {len(a)} bars) ===")
        m = metrics(a)
        print(f"  {'Buy & hold':30s} tot {m['total']*100:+8.1f}%  MDD {m['mdd']*100:6.1f}%  "
              f"Sh {m['sharpe']:5.2f}")
        for gate in ("ma", "pure"):
            for stop in (None, 0.03, 0.06, 0.08):
                bt = run_ct_sleeve(st, a, gate, stop)
                nav = bt["nav"][bt["nav"].index >= a.index[0]]
                mm = metrics(nav)
                wr = 100 * np.mean([t["ret"] > 0 for t in bt["trades"]]) if bt["trades"] else 0
                lab = f"CT-{gate.upper()} stop={'none' if stop is None else f'-{stop*100:.0f}%'}"
                print(f"  {lab:30s} tot {mm['total']*100:+8.1f}%  MDD {mm['mdd']*100:6.1f}%  "
                      f"Sh {mm['sharpe']:5.2f}  tr {len(bt['trades']):2d}  wr {wr:3.0f}%")
        for n in (30, 40, 50):
            mm = simple_bt(a, rule_sma(btc, n))
            print(f"  {'BTC>SMA%d (next-bar, 10bps)' % n:30s} tot {mm['total']*100:+8.1f}%  "
                  f"MDD {mm['mdd']*100:6.1f}%  Sh {mm['sharpe']:5.2f}  tr {mm['trades']:3d}")

    ct_e = run_ct_sleeve(st, eth, "ma", None)
    print("\nETH CT-MA trade log:")
    for t in ct_e["trades"]:
        print(f"  {t['entry_date'].date()} → {t['exit_date'].date()}  "
              f"{t['entry_price']:6.2f} → {t['exit_price']:6.2f}  {t['ret']*100:+7.1f}%  {t['reason']}")

    # ═══ Part 1b — fill-timing look-ahead (see eval doc §5) ═══════════════
    print("\n" + "═" * 90)
    print("PART 1b — fill timing: the CT signal for bar D is knowable only at")
    print("          D+1 12:00 UTC, yet equity/ETF sleeves fill at date D's US close.")
    print("═" * 90)
    mstr = pd.read_csv(_REPO / "data" / "backtest" / "mstr_daily.csv",
                       index_col=0, parse_dates=True)["close"].astype(float)

    def _fill_row(label, series, start):
        bt = run_ct_sleeve(st, series, "ma", None)
        nav = bt["nav"][bt["nav"].index >= pd.Timestamp(start)]
        m = metrics(nav)
        print(f"  {label:52s} {m['total']*100:+8.1f}%  MDD {m['mdd']*100:6.1f}%  Sh {m['sharpe']:5.2f}")
        return m["total"]

    for nm, ser, start in (("MSTR", mstr, "2024-03-05"),
                           ("ETHA (retired ETF sleeve)", etha, "2024-07-23"),
                           ("ETH  (12:00-UTC bars)", eth, "2024-03-05")):
        a = _fill_row(f"{nm}: fill @ close date D    (as deployed)", ser, start)
        b = _fill_row(f"{nm}: fill @ close date D+1  (post-signal)",
                      ser.shift(-1).dropna(), start)
        print(f"    → timing contribution: {(a - b) * 100:+.1f}pp\n")

    # ═══ Part 2 — ETH price-action grid + validation ══════════════════════
    print("\n" + "═" * 90 + "\nPART 2 — ETHA price-action strategy search + validation\n" + "═" * 90)
    for asset, src, nm, lag, top in ((ethyf, ethyf, "ETH ← own price", 0, 10),
                                     (ethyf, btc, "ETH ← BTC parent trend", 0, 6),
                                     (bmnr, bmnr, "BMNR ← own price", 0, 5),
                                     (bmnr, ethyf, "BMNR ← ETH parent (1d lag)", 1, 5)):
        rows = sorted(((n, simple_bt(asset, sig, extra_lag=lag)) for n, sig in grid(src).items()),
                      key=lambda r: -np.nan_to_num(r[1]["sharpe"], nan=-9))[:top]
        print(f"\n--- {nm} — top by Sharpe ---")
        for n, m in rows:
            print(f"  {n:12s} tot {m['total']*100:+8.1f}%  MDD {m['mdd']*100:6.1f}%  "
                  f"Sh {m['sharpe']:5.2f}  expo {m['expo']*100:3.0f}%  tr {m['trades']:3d}")

    print("\n5-fold CV (contiguous blocks) — ETH:")
    bh_e = eth.pct_change().fillna(0.0)
    nav_e = ct_e["nav"][ct_e["nav"].index >= eth.index[0]]
    kfold_line("CT-MA (BTC signal)", nav_e.pct_change().fillna(0.0), bh_e)
    kfold_line("BTC>SMA30", simple_bt(eth, rule_sma(btc, 30))["strat"], bh_e)
    kfold_line("own SMA30", simple_bt(eth, rule_sma(eth, 30))["strat"], bh_e)
    print("5-fold CV — BMNR:")
    bh_b = bmnr.pct_change().fillna(0.0)
    ct_b = run_ct_sleeve(st, bmnr, "ma", None)
    kfold_line("CT-MA (BTC signal)",
               ct_b["nav"][ct_b["nav"].index >= bmnr.index[0]].pct_change().fillna(0.0), bh_b)
    kfold_line("BTC>SMA40", simple_bt(bmnr, rule_sma(btc, 40))["strat"], bh_b)
    kfold_line("ETH DMA25/100 (lag1)",
               simple_bt(bmnr, rule_dma(ethyf, 25, 100), extra_lag=1)["strat"], bh_b)

    print("\nWalk-forward (expanding, best SMA n in-sample → next fold):")
    for asset, src, nm, lag in ((eth, eth, "ETH own-SMA", 0),
                                (eth, btc, "ETH BTC-SMA", 0),
                                (ethyf, ethyf, "ETH-USD(mid) own-SMA", 0)):
        ret, sh, picks = walkforward(asset, src, lag=lag)
        print(f"  {nm:22s} OOS ret {ret*100:+7.1f}%  Sh {sh:+5.2f}  picks {picks}")

    print("\nCost sensitivity (Sharpe @ 0/10/25/50 bps):")
    for nm, a, sig, lag in (("ETH BTC>SMA30", eth, rule_sma(btc, 30), 0),
                            ("ETH own SMA30", eth, rule_sma(eth, 30), 0)):
        shs = [simple_bt(a, sig, cost=c, extra_lag=lag)["sharpe"] for c in (0, .001, .0025, .005)]
        print(f"  {nm:18s} " + "  ".join(f"{s:+.2f}" for s in shs))

    print("\nLong-history check on ETH-USD itself (2023-11 →):")
    for nm, sig in (("ETH B&H", None), ("own SMA30", rule_sma(ethyf, 30)),
                    ("BTC>SMA30", rule_sma(btc, 30))):
        m = metrics(ethyf) if sig is None else simple_bt(ethyf, sig)
        print(f"  {nm:12s} tot {m['total']*100:+7.1f}%  MDD {m['mdd']*100:6.1f}%  Sh {m['sharpe']:5.2f}")

    print("\nBMNR post-squeeze era (2025-10-01 →):")
    b2 = bmnr.loc["2025-10-01":]
    m = metrics(b2)
    print(f"  {'B&H':22s} tot {m['total']*100:+7.1f}%  MDD {m['mdd']*100:6.1f}%  Sh {m['sharpe']:5.2f}")
    for nm, sig, lag in (("BTC>SMA40", rule_sma(btc, 40), 0),
                         ("ETH DMA25/100 (lag1)", rule_dma(ethyf, 25, 100), 1)):
        mm = simple_bt(b2, sig, extra_lag=lag)
        print(f"  {nm:22s} tot {mm['total']*100:+7.1f}%  MDD {mm['mdd']*100:6.1f}%  Sh {mm['sharpe']:5.2f}")

    # ═══ Part 3 — Overall portfolio, per risk profile ═════════════════════
    print("\n" + "═" * 90 + "\nPART 3 — fold into the Overall universe (per risk profile)\n" + "═" * 90)

    def sleeve(key, a, kind, mode="ct"):
        if mode == "ct":
            bt = run_ct_sleeve(st, a, "ma", None)
            nav = bt["nav"][bt["nav"].index >= a.index[0]]
            return dict(key=key, kind=kind, ret=nav.pct_change().fillna(0.0).rename(key),
                        pos_series=bt["pos"][bt["pos"].index >= a.index[0]].rename(key))
        m = simple_bt(a, rule_sma(a, 30))
        return dict(key=key, kind=kind, ret=m["strat"].rename(key), pos_series=m["pos"].rename(key))

    ETH_S, BMNR_S = sleeve("ETH", eth, "core"), sleeve("BMNR", bmnr, "beta")
    ETH_SMA = sleeve("ETH", eth, "core", mode="sma")

    print("Running live universe (~1-2 min)…", flush=True)
    base = [r for r in oc.run_universe() if r["key"] != "ETH"]   # 17-name baseline
    if oc._LAST_ERRORS:
        print("LOAD ERRORS:", oc._LAST_ERRORS)
    print("baseline:", [r["key"] for r in base])

    def optimise(results, extra, seed=7):
        rets = oc.returns_matrix(results)
        pos = oc.position_matrix(results, rets.index)
        out = {}
        for n, p in oc.RISK_PROFILES.items():
            caps = oc.caps_for(n)
            caps.update({k: p["caps"][kk] for k, kk in extra.items()})
            out[n] = oc.optimize_weights(rets, caps=caps, pos=pos, sata_daily=oc.SATA_DAILY,
                                         mdd_floor=p["mdd_floor"], objective=p["objective"],
                                         fundamental=True, seed=seed)
        return out

    obase = optimise(base, {})
    scen = {"+ETH (CT)": (base + [ETH_S], {"ETH": "core"}, ["ETH"]),
            "+BMNR (CT)": (base + [BMNR_S], {"BMNR": "beta"}, ["BMNR"]),
            "+ETH+BMNR": (base + [ETH_S, BMNR_S], {"ETH": "core", "BMNR": "beta"},
                          ["ETH", "BMNR"]),
            "+ETH (own-SMA30)": (base + [ETH_SMA], {"ETH": "core"}, ["ETH"])}
    for name, (res, kinds, keys) in scen.items():
        print(f"\n### {name}")
        o = optimise(res, kinds)
        for prof in ("Balanced", "Growth", "Aggressive"):
            a, b = o[prof]["optimal_pretilt"], obase[prof]["optimal_pretilt"]
            w = " ".join(f"{k} {a['weights'].get(k, 0)*100:4.1f}%" for k in keys)
            print(f"  {prof:10s} ret {b['total_ret']*100:7.1f}%→{a['total_ret']*100:7.1f}% "
                  f"({(a['total_ret']-b['total_ret'])*100:+7.1f}pp)  "
                  f"MDD {b['mdd']*100:6.2f}%→{a['mdd']*100:6.2f}%  "
                  f"Sh {b['sharpe']:.3f}→{a['sharpe']:.3f} ({a['sharpe']-b['sharpe']:+.3f})  [{w}]")
        a, b = o["Aggressive"]["equal"], obase["Aggressive"]["equal"]
        print(f"  equal-wt   ret {b['total_ret']*100:7.1f}%→{a['total_ret']*100:7.1f}% "
              f"({(a['total_ret']-b['total_ret'])*100:+7.1f}pp)  "
              f"MDD {b['mdd']*100:6.2f}%→{a['mdd']*100:6.2f}%  "
              f"Sh {b['sharpe']:.3f}→{a['sharpe']:.3f}")

    print("\nMC-seed noise band (baseline vs +ETH, neutral optimal):")
    for seed in (7, 8, 9):
        ob = optimise(base, {}, seed=seed)
        oe = optimise(base + [ETH_S], {"ETH": "core"}, seed=seed)
        print(f"  seed {seed}: " + "  ".join(
            f"{p}: {ob[p]['optimal_pretilt']['sharpe']:.3f}→{oe[p]['optimal_pretilt']['sharpe']:.3f}"
            for p in ("Balanced", "Growth", "Aggressive")))

    print("\n2025→now equal-weight slice:")
    for nm, res in (("baseline", base), ("+ETH", base + [ETH_S]),
                    ("+BMNR", base + [BMNR_S]), ("+both", base + [ETH_S, BMNR_S])):
        rets = oc.returns_matrix(res)
        pos = oc.position_matrix(res, rets.index)
        w = np.full(len(rets.columns), 1.0 / len(rets.columns))
        m = oc.slice_metrics(oc._equity(oc._combine(rets, w, pos, oc.SATA_DAILY)), "2025-01-01")
        print(f"  {nm:9s} ret {m['total_ret']*100:+7.1f}%  MDD {m['mdd']*100:6.2f}%  Sh {m['sharpe']:.2f}")

    rets_all = oc.returns_matrix(base + [ETH_S, BMNR_S])
    print("\nStrategy-stream correlation of the new sleeves to the book:")
    print(rets_all.corr()[["ETH", "BMNR"]].drop(index=["ETH", "BMNR"]).round(2).T.to_string())
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
