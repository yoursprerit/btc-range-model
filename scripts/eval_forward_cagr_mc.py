"""Forward 5-year CAGR distribution for the Overall book (all sleeves, levered).

Read-only. Nothing about the strategy, its configs or its weights is changed —
this projects the CURRENTLY-PUBLISHED strategy forward and asks: what is the
distribution of 5-year CAGR outcomes, and how much of it clears a given target
(default 40%)?

The book projected is the real one: 18 sleeves including every high-beta and
2x/3x sibling, traded off their parent signals, sized by the walk-forward gated
replay (`overall_core.walkforward_gated_replay`) — the same construction the app
publishes and the same one `scripts/build_strategy_health.py` monitors.

Two layers of uncertainty, kept deliberately separate:

  PATH RISK   a stationary (Politis-Romano) block bootstrap of the replay's own
              daily return stream. Geometric block lengths (mean 21 bars) keep
              the momentum, vol-clustering and regime runs of the sample — the
              2022 crypto winter and the 2026 BTC 124k->61k round-trip included.
              This layer alone answers "if the edge is exactly what the
              back-test says, how much does luck move a 5-year outcome?"

  DRIFT RISK  the bootstrap treats the sample mean as truth, which is the single
              biggest error in any forward projection. So each simulated path is
              RE-CENTRED to zero and given a drawn forward drift:

                  m = phi * ln(1+cagr_backtest)   edge retention (the judgment)
                      + eps                       sampling error on the drift
                      - costs                     turnover x spread
                      - drag                      measured book-vs-replay gap

              `eps` is not a judgment call: with Sharpe S over T years the
              standard error of the annualised mean is sigma/sqrt(T), which for
              this book is ~11-17 log points a year. `costs` uses the replay's
              own measured turnover. `drag` uses the M1 tracking series in
              `data/overall/strategy_health.json` (mean and standard error of
              the daily live-minus-replay gap), truncated at zero.

              `phi` — how much of the back-tested edge survives contact with the
              future — is where the real uncertainty lives, so it is NOT hidden
              in a single number. Three worlds are simulated separately and then
              blended, with the blend weights exposed on the command line:

                W1 edge holds        phi ~ N(1.00, 0.05)
                W2 documented        phi ~ N(0.80, 0.08)  the repo's own OOS
                   haircut                                residual (0.77-0.86
                                                          log retention, see
                                                          OVERALL_OOS_WALKFORWARD_EVAL.md
                W3 selection-        phi ~ N(0.45, 0.12)  closer to the honest-
                   dominated                              OOS core; the 2021-26
                                                          bull does not repeat

  TAIL SHOCK  a daily-close bootstrap cannot see the intraday gap risk of the
              2x/3x sleeves (a caveat every eval in this repo carries). Poisson
              events (default 0.25/yr) apply a one-day gap loss of U(20%,60%) to
              the profile's leveraged-cap weight. `--no-shock` turns it off.

    python scripts/eval_forward_cagr_mc.py                 # writes the eval doc
    python scripts/eval_forward_cagr_mc.py --target 0.25   # different hurdle
    python scripts/eval_forward_cagr_mc.py --cache CACHE   # reuse replay streams
    python scripts/eval_forward_cagr_mc.py --benchmark     # + passive baskets
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

HEALTH = ROOT / "data" / "overall" / "strategy_health.json"
DOC = ROOT / "FORWARD_CAGR_DISTRIBUTION_EVAL.md"
PNG = ROOT / "forward_cagr_distribution.png"
JSON_OUT = ROOT / "data" / "overall" / "forward_cagr_mc.json"

PROFILES = ("Balanced", "Growth", "Aggressive")

# Buy & hold means BUY, then HOLD: no rebalancing, and no idle-cash yield —
# a passive basket is always fully invested, so there is no SATA leg anywhere
# in any of these (the two rebalanced variants are kept only to show what the
# rebalancing itself was worth).
BH_DAY1 = "B&H (day-1 basket)"
BH_DRIFT = "B&H (all 18, held)"
BH_CARRY = "EW rebalanced daily (carried)"
BH_REPO = "EW rebalanced daily (repo benchmark)"
BENCHMARKS = (BH_DRIFT, BH_DAY1)
BENCHMARKS_REBAL = (BH_CARRY, BH_REPO)
ALL_BENCHMARKS = BENCHMARKS + BENCHMARKS_REBAL

# phi = fraction of the back-tested log drift that survives forward.
WORLDS = {
    "W1 edge holds": dict(mu=1.00, sd=0.05, weight=0.15),
    "W2 documented haircut": dict(mu=0.80, sd=0.08, weight=0.40),
    "W3 selection-dominated": dict(mu=0.45, sd=0.12, weight=0.45),
}

PCTLS = (5, 10, 25, 50, 75, 90, 95)


# ════════════════════════════════════════════════════════════════════════
# INPUTS — the replay stream, its turnover, and the live tracking gap
# ════════════════════════════════════════════════════════════════════════
def buy_and_hold_streams(res: list[dict], index: pd.Index) -> tuple[dict, dict, dict]:
    """Passive comparison baskets built from the SAME instruments.

    Buy & hold is taken literally in the two default variants: bought once,
    held, NEVER rebalanced, and never credited an idle-cash yield (a passive
    basket is always fully invested, so there is no SATA leg — the strategy's
    cash park has no analogue here and none is invented).

    ``B&H (all 18, held)``   1/n into each instrument at its first bar, then
                             left alone. Later listings are funded pro-rata
                             from the existing holdings, which is the only
                             trade the basket ever makes. Winners compound,
                             losers shrink toward irrelevance.
    ``B&H (day-1 basket)``   what could actually have been bought on the first
                             bar: equal weight across the instruments already
                             trading then, never touched again. No cash is
                             reserved for later listings, so this basket simply
                             never owns them.

    Two daily-rebalanced variants are computed for reference (--benchmark-
    rebalanced), purely to show what the rebalancing itself was worth:

    ``EW rebalanced daily (carried)``  equal target weight over every started
                             instrument, held across non-trading days (a closed
                             market contributes 0 and keeps its weight) — the
                             same carry convention the strategy replay uses.
    ``EW rebalanced daily (repo benchmark)``  `equal_weight_bh_replay` exactly
                             as the app publishes it. It renormalises over the
                             sleeves that PRINTED A BAR, and on a weekend only
                             the crypto sleeves print — so it spends every
                             weekend 100% in crypto. A calendar artifact, not a
                             portfolio decision.
    """
    import overall_core as oc
    bh = oc.bh_returns_matrix(res).reindex(index)
    cols, turn, mets = {}, {}, {}

    rb = oc.equal_weight_bh_replay(res, index=index)
    cols[BH_REPO] = pd.Series(rb["ret"])
    W = rb["weights"]
    turn[BH_REPO] = float((0.5 * W.diff().abs().sum(axis=1)).iloc[1:].mean())
    mets[BH_REPO] = {k: float(v) for k, v in rb["metrics"].items()}

    r = bh.to_numpy(float)
    started = np.maximum.accumulate(~np.isnan(r), axis=0)     # inception onward
    r0 = np.nan_to_num(r)                                     # closed market: 0%
    n_live = started.sum(axis=1, keepdims=True).clip(min=1)
    tgt = started / n_live                                    # equal, carried
    ret_carry = (tgt * r0).sum(axis=1)
    # real rebalancing turnover: the drift back to target, each bar
    drift_w = tgt * (1.0 + r0) / (1.0 + ret_carry)[:, None]
    turn[BH_CARRY] = float(0.5 * np.abs(tgt[1:] - drift_w[:-1]).sum(axis=1).mean())
    cols[BH_CARRY] = pd.Series(ret_carry, index=index)
    mets[BH_CARRY] = {k: float(v) for k, v in
                      oc.curve_metrics(oc._equity(cols[BH_CARRY])).items()}

    # literal buy & hold: fund each name at its own first bar, never rebalance
    vals = np.zeros(r.shape[1])
    out = np.zeros(len(index))
    live = np.zeros(r.shape[1], dtype=bool)
    for t in range(len(index)):
        new = started[t] & ~live
        if new.any():
            tot = vals.sum()
            for k in np.flatnonzero(new):
                add = (tot if tot > 0 else 1.0) / int(started[t].sum())
                if tot > 0:
                    vals *= (tot - add) / tot     # fund pro-rata from holdings
                vals[k] = add
            live |= new
        prev = vals.sum()
        vals = vals * (1.0 + r0[t])
        out[t] = vals.sum() / prev - 1.0 if prev > 0 else 0.0
    cols[BH_DRIFT] = pd.Series(out, index=index)
    turn[BH_DRIFT] = 0.0
    mets[BH_DRIFT] = {k: float(v) for k, v in
                      oc.curve_metrics(oc._equity(cols[BH_DRIFT])).items()}

    # what you could actually have bought on day one: the names that already
    # traded at the first bar, equal weight, then never touched again. No cash
    # reserved for later listings (that would be an unpaid drag nobody runs),
    # so this basket simply never owns the sleeves that listed later.
    day1 = ~np.isnan(r[0])
    v1 = day1 / day1.sum()
    out1 = np.zeros(len(index))
    for t in range(len(index)):
        prev = v1.sum()
        v1 = v1 * (1.0 + r0[t])
        out1[t] = v1.sum() / prev - 1.0
    cols[BH_DAY1] = pd.Series(out1, index=index)
    turn[BH_DAY1] = 0.0
    mets[BH_DAY1] = {k: float(v) for k, v in
                     oc.curve_metrics(oc._equity(cols[BH_DAY1])).items()}

    lev = [c for c in bh.columns if oc.ASSET_META[c]["kind"] == "lev"]
    day1_names = [c for c, ok in zip(bh.columns, day1) if ok]
    return cols, turn, dict(
        metrics=mets, lev_share=len(lev) / bh.shape[1], lev_names=lev,
        day1_names=day1_names,
        day1_lev_share=len([c for c in day1_names
                            if oc.ASSET_META[c]["kind"] == "lev"]) / len(day1_names),
        late_names=[c for c in bh.columns if c not in day1_names])


def load_streams(cache: Path | None) -> tuple[pd.DataFrame, dict, dict, dict]:
    """Daily return streams (strategy profiles + buy & hold), turnover, metrics."""
    if cache and (cache / "streams.csv").exists():
        rets = pd.read_csv(cache / "streams.csv", index_col=0, parse_dates=True)
        meta = json.loads((cache / "streams_meta.json").read_text())
        return rets, meta["turnover"], meta["metrics"], meta["bh"]

    import overall_core as oc
    print("run_universe() …", flush=True)
    res = oc.run_universe()
    if len(res) < 18:
        print(f"  WARNING: only {len(res)} instruments — {oc._LAST_ERRORS}")
    print(f"  {len(res)} instruments.", flush=True)

    cols, turn, mets = {}, {}, {}
    for name in PROFILES:
        prof = oc.RISK_PROFILES[name]
        print(f"walk-forward gated replay — {name} …", flush=True)
        rep = oc.walkforward_gated_replay(
            res, caps=oc.caps_for(name), mdd_floor=prof["mdd_floor"],
            objective=prof["objective"], sata_daily=oc.SATA_DAILY, tilt=True)
        cols[name] = pd.Series(rep["ret"])
        turn[name] = float(rep["turnover"]["mean"])     # one-way, per bar
        mets[name] = {k: float(v) for k, v in rep["metrics"].items()}
    rets = pd.DataFrame(cols).sort_index()

    print("equal-weight buy & hold (3 variants) …", flush=True)
    bh_cols, bh_turn, bh_extra = buy_and_hold_streams(res, rets.index)
    for k, v in bh_cols.items():
        rets[k] = v
    turn |= bh_turn
    mets |= bh_extra["metrics"]

    if cache:
        cache.mkdir(parents=True, exist_ok=True)
        rets.to_csv(cache / "streams.csv")
        (cache / "streams_meta.json").write_text(json.dumps(
            {"turnover": turn, "metrics": mets,
             "bh": {k: v for k, v in bh_extra.items() if k != "metrics"}},
            indent=1))
    return rets, turn, mets, {k: v for k, v in bh_extra.items() if k != "metrics"}


def tracking_gap() -> tuple[float, float, int]:
    """Mean and standard error of the daily live-book-minus-replay gap (M1)."""
    h = json.loads(HEALTH.read_text())
    s = h["portfolio"]["tracking"]["series"]
    live, rep = np.asarray(s["live"], float), np.asarray(s["replay"], float)
    lr, rr = np.diff(live) / live[:-1], np.diff(rep) / rep[:-1]
    gap = lr - rr
    active = gap[(np.abs(lr) + np.abs(rr)) > 1e-9]     # skip frozen weekend bars
    n = len(active)
    return float(active.mean()), float(active.std(ddof=1) / np.sqrt(n)), n


# ════════════════════════════════════════════════════════════════════════
# THE SIMULATION
# ════════════════════════════════════════════════════════════════════════
def block_bootstrap_idx(n_src: int, n_bars: int, n_sims: int,
                        mean_block: int, rng) -> np.ndarray:
    """Stationary bootstrap indices: geometric blocks, wrapped."""
    p = 1.0 / mean_block
    idx = np.empty((n_sims, n_bars), dtype=np.int64)
    cur = rng.integers(0, n_src, size=n_sims)
    for t in range(n_bars):
        idx[:, t] = cur
        cont = rng.random(n_sims) >= p
        cur = np.where(cont, (cur + 1) % n_src, rng.integers(0, n_src, size=n_sims))
    return idx


def simulate(logret: np.ndarray, drift: np.ndarray, n_bars: int, bars_yr: float,
             mean_block: int, rng, lev_cap: float, shock_lambda: float,
             years: float, chunk: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Return (cagr, max_drawdown) arrays, one entry per simulated 5-year path."""
    centred = logret - logret.mean()          # path shape only; drift is drawn
    n_sims = len(drift)
    cagr = np.empty(n_sims)
    mdd = np.empty(n_sims)

    for lo in range(0, n_sims, chunk):
        hi = min(lo + chunk, n_sims)
        idx = block_bootstrap_idx(len(centred), n_bars, hi - lo, mean_block, rng)
        L = centred[idx] + (drift[lo:hi, None] / bars_yr)

        if shock_lambda > 0:
            n_ev = rng.poisson(shock_lambda * years, size=hi - lo)
            for k in np.flatnonzero(n_ev):
                days = rng.integers(0, n_bars, size=n_ev[k])
                gaps = rng.uniform(0.20, 0.60, size=n_ev[k])
                np.add.at(L[k], days, np.log1p(-lev_cap * gaps))

        cum = np.cumsum(L, axis=1)
        cagr[lo:hi] = np.expm1(cum[:, -1] / years)
        mdd[lo:hi] = np.expm1((cum - np.maximum.accumulate(cum, axis=1)).min(axis=1))
    return cagr, mdd


def draw_drift(world: dict | None, base_log: float, se_log: float,
               turn_yr: float, gap_mu: float, gap_se: float, bars_yr: float,
               n: int, rng) -> np.ndarray:
    """Annual forward log drift per simulation."""
    if world is None:                          # reference: back-test as truth
        return np.full(n, base_log)
    phi = rng.normal(world["mu"], world["sd"], n)
    eps = rng.normal(0.0, se_log, n)
    spread = rng.triangular(0.0002, 0.0005, 0.0015, n)     # per unit one-way
    cost = turn_yr * spread
    drag = np.clip(rng.normal(gap_mu, gap_se, n), 0, None) * bars_yr
    return phi * base_log + eps - cost - drag


def summarise(cagr: np.ndarray, mdd: np.ndarray, target: float) -> dict:
    q = np.percentile(cagr, PCTLS)
    return {
        "p{}".format(p): float(v) for p, v in zip(PCTLS, q)
    } | {
        "mean": float(cagr.mean()),
        "p_target": float((cagr >= target).mean()),
        "p_20": float((cagr >= 0.20).mean()),
        "p_0": float((cagr >= 0.0).mean()),
        "p_100": float((cagr >= 1.0).mean()),
        "mdd_p50": float(np.percentile(mdd, 50)),
        "mdd_p95": float(np.percentile(mdd, 5)),      # 5th pctl = worst 5%
        "p_mdd_50": float((mdd <= -0.50).mean()),
    }


def run_all(inputs: dict, worlds: dict, sims: int, years: float, block: int,
            shock: float, target: float, seed: int,
            reference: bool = False, verbose: bool = False) -> dict:
    """Every profile, every world, plus the weighted blend."""
    profiles = {}
    for prof, pi in inputs.items():
        n_bars = int(round(years * pi["bars_yr"]))
        rng = np.random.default_rng(seed)
        rec = {k: pi[k] for k in ("sigma_yr", "se_log", "turnover_daily",
                                  "turn_yr", "lev_cap", "backtest_mdd")}
        rec["backtest_cagr"] = float(np.expm1(pi["base_log"]))
        rec["worlds"] = {}

        def draw(world, n):
            return draw_drift(world, pi["base_log"], pi["se_log"], pi["turn_yr"],
                              pi["gap_mu"], pi["gap_se"], pi["bars_yr"], n, rng)

        if reference:      # no drift uncertainty, no costs, no shock: pure luck
            c, m = simulate(pi["logret"], draw(None, sims), n_bars, pi["bars_yr"],
                            block, rng, pi["lev_cap"], 0.0, years)
            rec["path_risk_only"] = summarise(c, m, target)

        pooled_c, pooled_m = [], []
        for name, w in worlds.items():
            c, m = simulate(pi["logret"], draw(w, sims), n_bars, pi["bars_yr"],
                            block, rng, pi["lev_cap"], shock, years)
            rec["worlds"][name] = summarise(c, m, target)
            pooled_c.append(c)
            pooled_m.append(m)

        # blend: resample the worlds at their weights
        pick = rng.choice(len(worlds), size=sims,
                          p=[w["weight"] for w in worlds.values()])
        take = rng.integers(0, sims, size=sims)
        bc = np.stack(pooled_c)[pick, take]
        bm = np.stack(pooled_m)[pick, take]
        rec["blended"] = summarise(bc, bm, target)
        rec["_blend_sample"] = bc
        profiles[prof] = rec

        if verbose:
            b = rec["blended"]
            print(f"{prof:11s} backtest {rec['backtest_cagr']:6.1%} → blended "
                  f"median {b['p50']:6.1%}  P(≥{target:.0%}) = {b['p_target']:5.1%}"
                  f"  P(≥0%) = {b['p_0']:5.1%}")
    return profiles


def sensitivity(inputs: dict, args, shock: float) -> list[dict]:
    """Re-run the blend under the assumptions most open to argument."""
    def worlds_with(w1, w2, w3):
        out = {k: dict(v) for k, v in WORLDS.items()}
        for k, w in zip(out, (w1, w2, w3)):
            out[k]["weight"] = w
        return out

    variants = [
        ("baseline (as published above)", dict()),
        ("longer bootstrap blocks (63)", dict(block=63)),
        ("longer bootstrap blocks (126)", dict(block=126)),
        ("no leveraged gap shock", dict(shock=0.0)),
        ("optimistic prior 40/40/20", dict(worlds=worlds_with(.4, .4, .2))),
        ("sceptical prior 5/30/65", dict(worlds=worlds_with(.05, .30, .65))),
    ]
    n = max(args.sims // 4, 25_000)
    rows = []
    for label, kw in variants:
        w = kw.get("worlds", WORLDS)
        tot = sum(x["weight"] for x in w.values())
        w = {k: dict(v, weight=v["weight"] / tot) for k, v in w.items()}
        print(f"sensitivity — {label} …", flush=True)
        got = run_all(inputs, w, n, args.years, kw.get("block", args.block),
                      kw.get("shock", shock), args.target, args.seed)
        rows.append({"label": label,
                     "p": {p: got[p]["blended"]["p_target"] for p in got},
                     "median": {p: got[p]["blended"]["p50"] for p in got}})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sims", type=int, default=200_000)
    ap.add_argument("--years", type=float, default=5.0)
    ap.add_argument("--target", type=float, default=0.40, help="CAGR hurdle")
    ap.add_argument("--block", type=int, default=21, help="mean bootstrap block")
    ap.add_argument("--shock-lambda", type=float, default=0.25,
                    help="leveraged-sleeve gap events per year (0 to disable)")
    ap.add_argument("--no-shock", action="store_true")
    ap.add_argument("--weights", type=float, nargs=3, metavar=("W1", "W2", "W3"),
                    help="override the world blend weights")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--benchmark", action="store_true",
                    help="also project buy & hold of the same instruments — "
                         "bought once, never rebalanced, no idle-cash yield")
    ap.add_argument("--benchmark-rebalanced", action="store_true",
                    help="additionally show the daily-rebalanced equal-weight "
                         "variants, to isolate what rebalancing was worth")
    ap.add_argument("--sensitivity", action="store_true",
                    help="also re-run the blend under the arguable assumptions")
    ap.add_argument("--no-doc", action="store_true")
    args = ap.parse_args()

    if args.weights:
        for w, k in zip(args.weights, WORLDS):
            WORLDS[k]["weight"] = w
    wsum = sum(w["weight"] for w in WORLDS.values())
    for w in WORLDS.values():
        w["weight"] /= wsum

    import overall_core as oc
    rets, turnover, metrics, bh_extra = load_streams(args.cache)
    years_hist = (rets.index.max() - rets.index.min()).days / 365.25
    bars_yr = len(rets) / years_hist
    gap_mu, gap_se, gap_n = tracking_gap()
    shock = 0.0 if args.no_shock else args.shock_lambda

    bench = list(BENCHMARKS) if (args.benchmark or args.benchmark_rebalanced) else []
    if args.benchmark_rebalanced:
        bench += list(BENCHMARKS_REBAL)
    streams = list(PROFILES) + bench
    inputs = {}
    for prof in streams:
        logret = np.log1p(rets[prof].dropna().to_numpy())
        sigma_yr = float(logret.std(ddof=1) * np.sqrt(bars_yr))
        # a passive basket has no publish pipeline, so no measured tracking
        # drag; its permanent 2x/3x holding is the whole leveraged share of
        # the universe, not a profile cap
        is_bh = prof in ALL_BENCHMARKS
        inputs[prof] = dict(
            logret=logret,
            # anchor the drift on the PUBLISHED metric (curve_metrics bases the
            # equity curve at the first bar's close, so it is not quite the raw
            # sum of the log returns); the stream supplies path shape only
            base_log=float(np.log1p(metrics[prof]["cagr"])),
            sigma_yr=sigma_yr,
            se_log=sigma_yr / np.sqrt(years_hist),      # SE of the annual mean
            turn_yr=turnover[prof] * bars_yr,
            turnover_daily=turnover[prof],
            lev_cap=(bh_extra["lev_share"] if is_bh
                     else oc.RISK_PROFILES[prof]["caps"]["lev"]),
            backtest_mdd=metrics[prof]["mdd"],
            gap_mu=0.0 if is_bh else gap_mu,
            gap_se=0.0 if is_bh else gap_se, bars_yr=bars_yr)

    print(f"\nreplay: {len(rets)} bars · {rets.index.min().date()} → "
          f"{rets.index.max().date()} · {years_hist:.2f}y · {bars_yr:.1f} bars/yr")
    print(f"live tracking gap: {gap_mu*1e4:+.2f} bps/bar ± {gap_se*1e4:.2f} "
          f"(n={gap_n}, t={gap_mu/gap_se:+.2f})")
    print(f"simulating {args.sims:,} paths × "
          f"{int(round(args.years * bars_yr))} bars, "
          f"block {args.block}, shock λ={shock}/yr\n")

    out = {"generated_for_target": args.target, "years": args.years,
           "sims": args.sims, "bars_per_year": bars_yr,
           "hist_years": years_hist, "block": args.block,
           "shock_lambda": shock, "gap_bps": gap_mu * 1e4, "gap_n": gap_n,
           "worlds": {k: dict(v) for k, v in WORLDS.items()},
           "profiles": run_all(inputs, WORLDS, args.sims, args.years,
                               args.block, shock, args.target, args.seed,
                               reference=True, verbose=True)}

    if args.sensitivity:
        out["sensitivity"] = sensitivity(inputs, args, shock)

    if not args.no_doc:
        write_doc(out, args)
        write_png(out, args)
    JSON_OUT.write_text(json.dumps(
        {k: (v if k != "profiles" else
             {p: {kk: vv for kk, vv in r.items() if not kk.startswith("_")}
              for p, r in v.items()})
         for k, v in out.items()}, indent=1, default=float))
    print(f"\nWrote {JSON_OUT.relative_to(ROOT)}")
    return 0


# ════════════════════════════════════════════════════════════════════════
# OUTPUT
# ════════════════════════════════════════════════════════════════════════
def _row(name: str, s: dict, target: float) -> str:
    return ("| {} | {:.0%} | {:.0%} | {:.0%} | **{:.0%}** | {:.0%} | {:.0%} | "
            "{:.0%} | **{:.0f}%** | {:.0f}% | {:.0f}% |").format(
        name, s["p5"], s["p10"], s["p25"], s["p50"], s["p75"], s["p90"],
        s["p95"], 100 * s["p_target"], 100 * s["p_20"], 100 * s["p_0"])


def write_doc(out: dict, args) -> None:
    t = out["generated_for_target"]
    L = []
    L.append("# Forward 5-Year CAGR Distribution — the book as traded "
             "(all sleeves, leverage included)\n")
    L.append(f"*Generated by `scripts/eval_forward_cagr_mc.py` · "
             f"{out['sims']:,} paths per world × {args.years:.0f} years · "
             f"replay through the committed data vintage · "
             f"{out['bars_per_year']:.0f} bars/yr.*\n")
    L.append("**Read-only projection. No strategy, config, weight or model was "
             "changed.** This is the *currently-published* walk-forward gated "
             "replay — 18 sleeves, every high-beta and 2x/3x sibling traded off "
             "its parent signal, priority-tilted and water-filled to the profile "
             "caps, idle capital in SATA — projected forward.\n")

    L.append("## Method in one paragraph\n")
    L.append("Each simulated 5-year path is a **stationary block bootstrap** "
             f"(geometric blocks, mean {args.block} bars) of the replay's own "
             "daily returns, so momentum, vol-clustering and regime runs survive "
             "— including the 2022 crypto winter and the 2026 BTC round-trip. "
             "The bootstrap would otherwise treat the back-tested mean as truth, "
             "so every path is **re-centred and given a drawn forward drift**: "
             "`phi × ln(1+CAGR_backtest) + sampling error − costs − tracking "
             "drag`. Sampling error is the standard error of the annualised mean "
             "(sigma/sqrt(T)); costs are the replay's own measured turnover times "
             "a drawn spread; drag is the measured live book-vs-replay gap. "
             "`phi` — the share of the back-tested edge that survives forward — "
             "is the judgment, so three worlds are reported separately before "
             "being blended.\n")

    L.append("| World | phi (edge retention) | Blend weight | Rationale |")
    L.append("|---|---|---:|---|")
    rat = {
        "W1 edge holds": "the replay drift IS the forward drift — only sampling "
                         "error, costs and the measured drag bite",
        "W2 documented haircut": "matches this repo's own OOS residual "
                                 "(0.77–0.86 log retention, "
                                 "`OVERALL_OOS_WALKFORWARD_EVAL.md` §2)",
        "W3 selection-dominated": "sleeve parameters were tuned on this history "
                                  "and the 2021–26 bull does not repeat; closer "
                                  "to the honest-OOS core",
    }
    for k, w in out["worlds"].items():
        L.append(f"| {k} | N({w['mu']:.2f}, {w['sd']:.2f}) | "
                 f"{w['weight']:.0%} | {rat[k]} |")
    L.append("")

    bh_rows = [p for p in out["profiles"] if p in ALL_BENCHMARKS]
    if bh_rows:
        L.append("## Strategy vs passive — the comparison in one table\n")
        L.append("Same instruments, same simulation machinery, same three "
                 "worlds. **Buy & hold is literal here: bought once, never "
                 "rebalanced, and never credited the SATA idle-cash yield** — "
                 "a passive basket is always fully invested, so it has no cash "
                 "leg to pay a coupon on. For the passive rows `phi` means "
                 "something different from the strategy's: there is no fitted "
                 "signal to decay, so it stands for **universe-selection "
                 "hindsight** (these tickers were assembled in 2026, knowing "
                 "which ones ripped) plus the same regime question. The passive "
                 "rows also carry **no tracking drag** (nothing to publish or "
                 "mis-execute), which if anything flatters them.\n")
        L.append(f"| Book | Historical CAGR | Blended median | P(≥{t:.0%}) | "
                 "P(≥20%) | P(≥0%) | Median worst DD | Turnover |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for p, rec in out["profiles"].items():
            b = rec["blended"]
            L.append(f"| {'**' + p + '**' if p not in ALL_BENCHMARKS else p} | "
                     f"{rec['backtest_cagr']:.1%} | **{b['p50']:.0%}** | "
                     f"{b['p_target']:.0%} | {b['p_20']:.0%} | {b['p_0']:.0%} | "
                     f"{b['mdd_p50']:.0%} | {rec['turn_yr']:.1f}x/yr |")
        L.append("")

    for prof, rec in out["profiles"].items():
        L.append(f"## {prof}\n")
        lbl = ("Historical (2021 → now)" if prof in ALL_BENCHMARKS
               else "Back-test (published replay)")
        L.append(f"{lbl}: **{rec['backtest_cagr']:.1%} "
                 f"CAGR**, max drawdown {rec['backtest_mdd']:.1%}, annualised "
                 f"vol {rec['sigma_yr']:.1%}. Standard error of that drift: "
                 f"**±{rec['se_log']:.1%}/yr** — before any question of edge "
                 f"decay. Measured turnover {rec['turnover_daily']:.1%}/bar "
                 f"one-way ({rec['turn_yr']:.1f}x/yr).\n")
        L.append("| Distribution | p5 | p10 | p25 | **median** | p75 | p90 | p95 "
                 f"| **P(≥{t:.0%})** | P(≥20%) | P(≥0%) |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        L.append(_row("*path risk only (back-test as truth)*",
                      rec["path_risk_only"], t))
        for k, s in rec["worlds"].items():
            L.append(_row(k, s, t))
        L.append(_row("**BLENDED**", rec["blended"], t))
        b = rec["blended"]
        L.append("")
        L.append(f"Drawdown along the way (blended): median worst drawdown "
                 f"**{b['mdd_p50']:.0%}**, 5% of paths worse than "
                 f"**{b['mdd_p95']:.0%}**, and **{b['p_mdd_50']:.0%}** of paths "
                 f"see a drawdown past −50%.\n")

    if out.get("sensitivity"):
        L.append(f"## Sensitivity — P(5-yr CAGR ≥ {t:.0%}) under the arguable "
                 "assumptions\n")
        L.append("| Variant | " + " | ".join(out["profiles"]) + " |")
        L.append("|---|" + "---:|" * len(out["profiles"]))
        for row in out["sensitivity"]:
            L.append("| {} | {} |".format(
                row["label"],
                " | ".join(f"{row['p'][p]:.0%}" for p in out["profiles"])))
        L.append("")
        L.append("Block length barely moves the answer — path shape is not what "
                 "drives it. The **prior on edge retention is the whole ball "
                 "game**, which is the honest shape of this question: it is not "
                 "a statistics problem, it is a judgment about how much of a "
                 "back-test survives.\n")

    L.append("## How to read this\n")
    L.append(f"* The **path-risk-only** row is the honest answer to *\"if the "
             f"back-test is exactly right, is 5 years long enough to be sure of "
             f"{t:.0%}?\"* — the spread there is pure luck, no scepticism at all.\n")
    L.append("* Every row below it adds a documented, measured or explicitly "
             "argued deduction. The gap between the top row and the blended row "
             "is the price of the strategy's own caveats.\n")
    L.append("* The blend weights are a judgment and the single biggest lever on "
             "the headline number. Re-run with `--weights` to see your own prior: "
             "`python scripts/eval_forward_cagr_mc.py --weights 0.4 0.4 0.2`.\n")

    L.append("## What is NOT in the distribution\n")
    L.append("* **Structural breaks** — a sleeve's signal dying outright, a "
             "leveraged ETF closing or being restructured, SATA cutting its "
             "coupon or trading away from par (the replay credits ~13%/yr at par, "
             "flat, forever), an exchange or custody failure on the crypto leg.\n")
    L.append("* **Capacity and execution reality beyond the measured gap** — "
             "the tracking series is short and its mean is not yet statistically "
             "distinguishable from zero; it is modelled as a drag, not a fact.\n")
    L.append("* **Position-level tail risk beyond the modelled gap events** — "
             "a daily-close bootstrap cannot see intraday 3x behaviour.\n")
    L.append("* **Your own behaviour.** Every path here is traded to completion "
             "without discretion. The blended distribution puts a material share "
             "of paths through a >50% drawdown; a strategy abandoned at the "
             "bottom of one realises none of the recovery above it.\n")
    DOC.write_text("\n".join(L) + "\n")
    print(f"Wrote {DOC.relative_to(ROOT)}")


def write_png(out: dict, args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = out["generated_for_target"]
    profs = list(out["profiles"])
    ncol = min(3, len(profs))
    nrow = int(np.ceil(len(profs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.6 * nrow),
                             sharey=True, squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(profs):]:
        ax.axis("off")
    for ax, prof in zip(axes, profs):
        rec = out["profiles"][prof]
        v = rec["_blend_sample"]
        s = v[(v >= -0.6) & (v <= 1.6)]          # trim, never clip: a clipped
        ax.hist(s, bins=120, color="#4c72b0",    # tail piles into one fake bar
                alpha=0.85, density=True)
        b = rec["blended"]
        ax.axvline(t, color="#c44e52", lw=2, ls="--",
                   label=f"{t:.0%} target — P = {b['p_target']:.0%}")
        ax.axvline(b["p50"], color="#dd8452", lw=2,
                   label=f"median {b['p50']:.0%}")
        ax.axvline(rec["backtest_cagr"], color="#55a868", lw=2, ls=":",
                   label=("historical " if prof in ALL_BENCHMARKS else "back-test ")
                         + f"{rec['backtest_cagr']:.0%}")
        ax.axvline(0, color="0.4", lw=1)
        ax.set_title(f"{prof}", fontsize=12, weight="bold")
        ax.set_xlabel("5-year CAGR")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("density")
    fig.suptitle("Forward 5-year CAGR — blended distribution "
                 "(same 18 instruments: strategy profiles vs equal-weight B&H)"
                 if any(p in ALL_BENCHMARKS for p in profs) else
                 "Forward 5-year CAGR — blended distribution "
                 "(walk-forward gated replay, all sleeves incl. leverage)",
                 fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(PNG, dpi=130)
    print(f"Wrote {PNG.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
