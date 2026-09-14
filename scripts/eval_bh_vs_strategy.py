"""Head-to-head: can buy & hold beat the strategy over the next 5 years?

Read-only. Nothing about the strategy, its configs or its weights is changed.

`eval_forward_cagr_mc.py` draws the strategy's future and buy & hold's future
INDEPENDENTLY, which is fine for "what will each do" and useless for "which one
wins" — two independent draws disagree for reasons the market never supplies.
This eval pairs them: every simulated 5-year path resamples the SAME DAYS for
both books, so they always live through the same market, and the difference in
their outcomes is only what the books themselves do differently.

Why the pairing matters here: the relationship is NOT a constant beta. Measured
monthly against the passive basket, Balanced returns +8.7% in the basket's up
months (the basket makes +7.2%) and only -2.6% in its down months (the basket
loses -6.8%) — it captures more than all of the upside and well under half of
the downside. A daily regression flattens that to beta 0.72; resampling real
days keeps it.

THE DRIFT LAYER, and the two dials that decide the answer:

    market      the passive basket's forward CAGR. Applied as a per-bar shift
                to the resampled market stream; the strategy inherits
                beta x that shift, beta being the regression slope — the
                first-order response of the strategy to a change in market
                drift.
    alpha       how much of the strategy's historical alpha (its drift above
    retention   beta x market, ~31-38%/yr depending on profile) survives
                forward. This is the fitted part, and the part that decays.
                At 0 the strategy keeps its market exposure profile and loses
                every bit of timing value.

Both dials are swept as a grid, because the honest answer is "it depends on
those two numbers and here is the whole surface", not a single probability.
A headline blend over the same three worlds as the main eval is also reported.

Sampling pools: `all` resamples any day; `bull` resamples only days when GOLD
and BITCOIN were both above their 200-day moving average (399 of 1717 bars,
2024-05 to 2026-08) — the closest thing in the data to the user's base case,
and it carries the strategy's behaviour in those regimes with it.

    python scripts/eval_bh_vs_strategy.py                 # writes the eval doc
    python scripts/eval_bh_vs_strategy.py --sims 100000
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
DOC = ROOT / "BH_VS_STRATEGY_EVAL.md"
PNG = ROOT / "bh_vs_strategy.png"
JSON_OUT = ROOT / "data" / "overall" / "bh_vs_strategy.json"

PROFILES = ("Balanced", "Growth", "Aggressive")
MARKET = "B&H (all 18, held)"          # the passive book both are measured on
ALT_MARKET = "B&H (day-1 basket)"

# market retention g, and the alpha retention that goes with it. Alpha is the
# fitted part, so it is assumed to decay faster than the asset drift does.
WORLDS = {
    "W1 edge holds": dict(g=(1.00, 0.05), a=(0.85, 0.10), weight=0.15),
    "W2 documented haircut": dict(g=(0.80, 0.08), a=(0.55, 0.15), weight=0.40),
    "W3 selection-dominated": dict(g=(0.45, 0.12), a=(0.25, 0.15), weight=0.45),
}

MARKET_GRID = (-0.10, 0.0, 0.10, 0.20, 0.30, 0.40, 0.60, 0.80)
ALPHA_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)
PCTLS = (5, 10, 25, 50, 75, 90, 95)
SPREAD = (0.0002, 0.0005, 0.0015)      # per unit one-way turnover, triangular


# ════════════════════════════════════════════════════════════════════════
# INPUTS
# ════════════════════════════════════════════════════════════════════════
def load(cache: Path) -> tuple[pd.DataFrame, dict, dict, np.ndarray]:
    """Streams + turnover/metrics + the gold-and-bitcoin bull-day mask."""
    streams = cache / "streams.csv"
    if not streams.exists():
        raise SystemExit(f"missing {streams} — run scripts/eval_forward_cagr_mc.py "
                         f"--benchmark --cache {cache} first")
    df = pd.read_csv(streams, index_col=0, parse_dates=True)
    meta = json.loads((cache / "streams_meta.json").read_text())

    closes = cache / "asset_closes.csv"
    if closes.exists():
        px = pd.read_csv(closes, index_col=0, parse_dates=True)
    else:
        import overall_core as oc
        res = oc.run_universe()
        px = pd.DataFrame({r["key"]: oc.asset_close_series(r) for r in res
                           if oc.asset_close_series(r) is not None})
        px = px.reindex(df.index).ffill()
        px.to_csv(closes)
    px = px.reindex(df.index).ffill()
    bull = np.ones(len(df), dtype=bool)
    for k in ("GLDM", "BTC"):
        s = px[k]
        bull &= (s > s.rolling(200, min_periods=50).mean()).fillna(False).to_numpy()
    return df, meta["turnover"], meta["metrics"], bull


def tracking_gap() -> tuple[float, float]:
    h = json.loads(HEALTH.read_text())
    s = h["portfolio"]["tracking"]["series"]
    live, rep = np.asarray(s["live"], float), np.asarray(s["replay"], float)
    lr, rr = np.diff(live) / live[:-1], np.diff(rep) / rep[:-1]
    gap = lr - rr
    a = gap[(np.abs(lr) + np.abs(rr)) > 1e-9]
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)))


# ════════════════════════════════════════════════════════════════════════
# PAIRED SIMULATION
# ════════════════════════════════════════════════════════════════════════
def paired_totals(log_s: np.ndarray, log_m: np.ndarray, n_bars: int, n_sims: int,
                  mean_block: int, rng, chunk: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """Total 5-year log return of BOTH books over the same resampled days.

    Both streams are re-centred to zero drift; the drift is added afterwards by
    the caller, which is what makes the whole (market x alpha) grid a matter of
    arithmetic on these two vectors instead of a fresh simulation per cell.
    """
    cs, cm = log_s - log_s.mean(), log_m - log_m.mean()
    n_src = len(cs)
    p = 1.0 / mean_block
    tot_s, tot_m = np.empty(n_sims), np.empty(n_sims)
    for lo in range(0, n_sims, chunk):
        hi = min(lo + chunk, n_sims)
        k = hi - lo
        cur = rng.integers(0, n_src, size=k)
        acc_s = np.zeros(k)
        acc_m = np.zeros(k)
        for _ in range(n_bars):
            acc_s += cs[cur]
            acc_m += cm[cur]                      # SAME day for both books
            cont = rng.random(k) >= p
            cur = np.where(cont, (cur + 1) % n_src, rng.integers(0, n_src, size=k))
        tot_s[lo:hi], tot_m[lo:hi] = acc_s, acc_m
    return tot_s, tot_m


def shock_logs(lev_w: float, lam: float, years: float, n: int, rng) -> np.ndarray:
    """Total log impact of leveraged-sleeve gap events over the horizon."""
    if lam <= 0 or lev_w <= 0:
        return np.zeros(n)
    n_ev = rng.poisson(lam * years, size=n)
    out = np.zeros(n)
    for k in np.flatnonzero(n_ev):
        gaps = rng.uniform(0.20, 0.60, size=n_ev[k])
        out[k] = np.log1p(-lev_w * gaps).sum()
    return out


def outcomes(base_s: np.ndarray, base_m: np.ndarray, pi: dict, years: float,
             m_cagr: np.ndarray, a_ret: np.ndarray, draws: dict) -> tuple:
    """Turn the re-centred path totals into a pair of 5-year CAGRs.

    m_cagr is the assumed forward CAGR of the passive basket (per simulation);
    a_ret the share of the strategy's historical alpha that survives.
    """
    m_log = np.log1p(m_cagr) + draws["eps_m"]                    # market drift
    s_log = pi["beta"] * m_log + a_ret * pi["alpha_log"] + draws["eps_a"]
    tot_m = base_m + years * m_log - years * draws["cost_m"] + draws["shock_m"]
    tot_s = (base_s + years * s_log - years * draws["cost_s"]
             - years * draws["drag_s"] + draws["shock_s"])
    return np.expm1(tot_s / years), np.expm1(tot_m / years)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sims", type=int, default=200_000)
    ap.add_argument("--years", type=float, default=5.0)
    ap.add_argument("--block", type=int, default=21)
    ap.add_argument("--shock-lambda", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--market", default=MARKET)
    ap.add_argument("--no-doc", action="store_true")
    args = ap.parse_args()

    import overall_core as oc
    df, turnover, metrics, bull = load(args.cache)
    years_hist = (df.index.max() - df.index.min()).days / 365.25
    bars_yr = len(df) / years_hist
    n_bars = int(round(args.years * bars_yr))
    gap_mu, gap_se = tracking_gap()
    lev_share = len([c for c in oc.ASSET_META if oc.ASSET_META[c]["kind"] == "lev"])
    lev_share /= len(oc.ASSET_META)

    log_m_all = np.log1p(df[args.market].to_numpy())
    L_m_hist = float(np.log1p(metrics[args.market]["cagr"]))
    sig_m = float(log_m_all.std(ddof=1) * np.sqrt(bars_yr))

    print(f"market book: {args.market} — historical {np.expm1(L_m_hist):.1%} CAGR")
    print(f"bull pool: {bull.sum()} of {len(df)} bars "
          f"({df.index[bull].min().date()} → {df.index[bull].max().date()})\n")

    out = {"market": args.market, "sims": args.sims, "years": args.years,
           "block": args.block, "bars_per_year": bars_yr,
           "bull_bars": int(bull.sum()), "n_bars_hist": len(df),
           "bull_from": str(df.index[bull].min().date()),
           "bull_to": str(df.index[bull].max().date()),
           "market_grid": list(MARKET_GRID), "alpha_grid": list(ALPHA_GRID),
           "worlds": {k: {kk: (list(vv) if isinstance(vv, tuple) else vv)
                          for kk, vv in v.items()} for k, v in WORLDS.items()},
           "profiles": {}}

    for prof in PROFILES:
        s = df[prof].to_numpy()
        m = df[args.market].to_numpy()
        beta = float(np.cov(s, m, ddof=1)[0, 1] / np.var(m, ddof=1))
        L_s_hist = float(np.log1p(metrics[prof]["cagr"]))
        se_m = sig_m / np.sqrt(years_hist)
        lev_cap = oc.RISK_PROFILES[prof]["caps"]["lev"]

        rec = {"beta": beta,
               "alpha_cagr": float(np.expm1(L_s_hist - beta * L_m_hist)),
               "se_m": se_m,
               "hist_cagr": float(np.expm1(L_s_hist)),
               "market_hist_cagr": float(np.expm1(L_m_hist)),
               "pools": {}}

        for pool_name, mask in (("all", np.ones(len(df), bool)), ("bull", bull)):
            # beta AND alpha are estimated inside the pool: in a bull regime the
            # gate behaves differently (it concentrates into what is trending),
            # and re-centring the resampled paths would otherwise throw that
            # away and leave the two pools reporting the same thing.
            ls, lm = np.log1p(s[mask]), np.log1p(m[mask])
            b_pool = float(np.cov(s[mask], m[mask], ddof=1)[0, 1]
                           / np.var(m[mask], ddof=1))
            L_s_pool = float(ls.mean() * bars_yr)
            L_m_pool = float(lm.mean() * bars_yr)
            alpha_log = L_s_pool - b_pool * L_m_pool
            resid = ls - b_pool * lm
            se_a = float(resid.std(ddof=1) * np.sqrt(bars_yr)
                         / np.sqrt(len(ls) / bars_yr))
            pi = dict(beta=b_pool, alpha_log=alpha_log)

            rng = np.random.default_rng(args.seed)
            base_s, base_m = paired_totals(ls, lm, n_bars, args.sims,
                                           args.block, rng)

            draws = dict(
                eps_m=rng.normal(0, se_m, args.sims),
                eps_a=rng.normal(0, se_a, args.sims),
                cost_s=turnover[prof] * bars_yr * rng.triangular(*SPREAD, args.sims),
                cost_m=turnover[args.market] * bars_yr
                * rng.triangular(*SPREAD, args.sims),
                drag_s=np.clip(rng.normal(gap_mu, gap_se, args.sims), 0, None) * bars_yr,
                shock_s=shock_logs(lev_cap, args.shock_lambda, args.years,
                                   args.sims, rng),
                shock_m=shock_logs(lev_share, args.shock_lambda, args.years,
                                   args.sims, rng))

            grid = []
            for mc in MARKET_GRID:
                row = []
                for ar in ALPHA_GRID:
                    cs, cm = outcomes(base_s, base_m, pi, args.years,
                                      np.full(args.sims, mc),
                                      np.full(args.sims, ar), draws)
                    row.append(float((cm > cs).mean()))
                grid.append(row)

            # headline: the three worlds, market drift and alpha drawn together
            pick = rng.choice(len(WORLDS), size=args.sims,
                              p=[w["weight"] for w in WORLDS.values()])
            g = np.empty(args.sims)
            a = np.empty(args.sims)
            for i, w in enumerate(WORLDS.values()):
                sel = pick == i
                g[sel] = rng.normal(*w["g"], sel.sum())
                a[sel] = np.clip(rng.normal(*w["a"], sel.sum()), 0, None)
            cs, cm = outcomes(base_s, base_m, pi, args.years,
                              np.expm1(g * L_m_hist), a, draws)
            diff = cm - cs
            rec["pools"][pool_name] = {
                "grid": grid,
                "beta": b_pool,
                "alpha_cagr": float(np.expm1(alpha_log)),
                "pool_market_cagr": float(np.expm1(L_m_pool)),
                "pool_strategy_cagr": float(np.expm1(L_s_pool)),
                "pool_bars": int(mask.sum()),
                "p_bh_wins": float((cm > cs).mean()),
                "strategy": {f"p{p}": float(np.percentile(cs, p)) for p in PCTLS},
                "bh": {f"p{p}": float(np.percentile(cm, p)) for p in PCTLS},
                "diff": {f"p{p}": float(np.percentile(diff, p)) for p in PCTLS},
                "strategy_median": float(np.median(cs)),
                "bh_median": float(np.median(cm)),
                "_diff_sample": diff,
            }
            print(f"{prof:11s} pool={pool_name:4s} beta {b_pool:.2f} "
                  f"alpha {np.expm1(alpha_log):.0%}/yr "
                  f"(pool: strat {np.expm1(L_s_pool):.0%} vs basket "
                  f"{np.expm1(L_m_pool):.0%}) → "
                  f"P(B&H beats strategy) = "
                  f"{rec['pools'][pool_name]['p_bh_wins']:.1%}")
        out["profiles"][prof] = rec

    if not args.no_doc:
        write_doc(out)
        write_png(out)
    JSON_OUT.write_text(json.dumps(
        {k: (v if k != "profiles" else
             {p: {kk: (vv if kk != "pools" else
                       {pn: {a: b for a, b in pv.items()
                             if not a.startswith("_")}
                        for pn, pv in vv.items()})
                  for kk, vv in r.items()}
              for p, r in v.items()})
         for k, v in out.items()}, indent=1, default=float))
    print(f"\nWrote {JSON_OUT.relative_to(ROOT)}")
    return 0


# ════════════════════════════════════════════════════════════════════════
# OUTPUT
# ════════════════════════════════════════════════════════════════════════
def write_doc(out: dict) -> None:
    L = []
    L.append("# Can buy & hold beat the strategy? — paired 5-year simulation\n")
    L.append(f"*Generated by `scripts/eval_bh_vs_strategy.py` · {out['sims']:,} "
             f"paired paths × {out['years']:.0f} years · market book = "
             f"**{out['market']}** (bought once, never rebalanced, no idle-cash "
             f"yield).*\n")
    L.append("**Read-only. No strategy, config, weight or model was changed.**\n")

    L.append("## Why this is not just the two distributions side by side\n")
    L.append("`FORWARD_CAGR_DISTRIBUTION_EVAL.md` draws each book's future "
             "independently, so subtracting its two distributions would count "
             "disagreement the market never supplied. Here **every path "
             "resamples the same days for both books**, so they always live "
             "through the same market and the gap between them is only what the "
             "books do differently.\n")
    L.append("That matters because the relationship is not a flat beta. "
             "Measured monthly against the passive basket, Balanced returns "
             "**+8.7% in the basket's up months** (basket +7.2%) and **−2.6% in "
             "its down months** (basket −6.8%): more than all of the upside, "
             "well under half of the downside. Resampling real days keeps that "
             "asymmetry; a beta would flatten it.\n")

    L.append("## The two dials\n")
    L.append("| Dial | What it is | Where it comes from |")
    L.append("|---|---|---|")
    L.append("| **Market** | the passive basket's forward CAGR | your call — "
             "swept across the grid below |")
    L.append("| **Alpha retention** | the share of the strategy's historical "
             "alpha that survives | the open question in every other eval |")
    L.append("")
    L.append("Per profile, measured on the actual streams:\n")
    L.append("| Profile | Beta vs the basket | Historical alpha | "
             "Historical CAGR |")
    L.append("|---|---:|---:|---:|")
    for p, rec in out["profiles"].items():
        L.append(f"| {p} | {rec['beta']:.2f} | {rec['alpha_cagr']:.0%}/yr | "
                 f"{rec['hist_cagr']:.1%} |")
    L.append(f"\n(The basket itself returned {list(out['profiles'].values())[0]['market_hist_cagr']:.1%} "
             f"a year over the same window.)\n")

    for p, rec in out["profiles"].items():
        L.append(f"## {p}\n")
        for pool, pv in rec["pools"].items():
            lbl = ("any market (all days resampled)" if pool == "all"
                   else f"gold-and-bitcoin bull only ({out['bull_bars']} of "
                        f"{out['n_bars_hist']} bars, {out['bull_from']} → "
                        f"{out['bull_to']})")
            L.append(f"### P(buy & hold beats {p}) — {lbl}\n")
            L.append(f"*In this pool: beta **{pv['beta']:.2f}**, alpha "
                     f"**{pv['alpha_cagr']:.0%}/yr** — measured on the pool's "
                     f"own days, where the strategy annualised "
                     f"{pv['pool_strategy_cagr']:.0%} against the basket's "
                     f"{pv['pool_market_cagr']:.0%}.*\n")
            L.append("| Basket does → | " + " | ".join(
                f"alpha keeps {a:.0%}" for a in out["alpha_grid"]) + " |")
            L.append("|---|" + "---:|" * len(out["alpha_grid"]))
            for mc, row in zip(out["market_grid"], pv["grid"]):
                L.append(f"| **{mc:+.0%}/yr** | " +
                         " | ".join(f"{v:.0%}" for v in row) + " |")
            L.append("")
            L.append(f"Blended over the three worlds: **P(buy & hold wins) = "
                     f"{pv['p_bh_wins']:.0%}** · median strategy "
                     f"{pv['strategy_median']:.0%}/yr vs median basket "
                     f"{pv['bh_median']:.0%}/yr · the gap (basket − strategy) "
                     f"runs p5 {pv['diff']['p5']:+.0%}, median "
                     f"{pv['diff']['p50']:+.0%}, p95 {pv['diff']['p95']:+.0%}.\n")

    L.append("## Reading the grid\n")
    L.append("* **A strong bull does not, by itself, hand it to buy & hold.** "
             "The strategy is a momentum concentrator: it funds whatever is "
             "trending up to the profile caps, so a rising market is where its "
             "gate earns most. In the real gold-and-bitcoin bull days in this "
             "sample the passive basket annualised ~30% while Balanced "
             "annualised ~134%.\n")
    L.append("* **What hands it to buy & hold is the alpha going away.** Read "
             "down the left column: with alpha retention at 0 the strategy is "
             "just a beta-weighted version of the basket, and since Balanced's "
             "beta is below 1, the basket wins nearly every path in any "
             "positive market. That — not the bull itself — is the scenario to "
             "worry about.\n")
    L.append("* **The two dials interact.** A strong market RAISES the alpha "
             "retention needed to stay ahead, because the return given up by "
             "being out of the market scales with the market. A weak or "
             "negative market flips it: the strategy's cash park and stops win "
             "on their own, with no alpha at all.\n")

    L.append("## What is not modelled\n")
    L.append("* The beta is the first-order response of the strategy to a shift "
             "in market drift. The real gate is non-linear, and the paired "
             "resampling preserves that only for the regimes present in the "
             "sample — a market far outside the 2021-26 range is extrapolation.\n")
    L.append("* Alpha retention is applied as a flat scale on the whole "
             "historical alpha. A signal that inverts (systematically out when "
             "the market rips) would be worse than retention 0, and is not in "
             "the grid.\n")
    L.append("* The bull pool is 399 bars from a single episode (2024-05 to "
             "2026-08). Treat it as one draw of what a bull looks like, not a "
             "law.\n")
    DOC.write_text("\n".join(L) + "\n")
    print(f"Wrote {DOC.relative_to(ROOT)}")


def write_png(out: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    profs = list(out["profiles"])
    fig, axes = plt.subplots(2, len(profs), figsize=(5.2 * len(profs), 8.6),
                             squeeze=False)
    for j, prof in enumerate(profs):
        rec = out["profiles"][prof]
        for i, pool in enumerate(("all", "bull")):
            ax = axes[i][j]
            g = np.array(rec["pools"][pool]["grid"])
            im = ax.imshow(g, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(out["alpha_grid"])))
            ax.set_xticklabels([f"{a:.0%}" for a in out["alpha_grid"]], fontsize=8)
            ax.set_yticks(range(len(out["market_grid"])))
            ax.set_yticklabels([f"{m:+.0%}" for m in out["market_grid"]], fontsize=8)
            for y in range(g.shape[0]):
                for x in range(g.shape[1]):
                    ax.text(x, y, f"{g[y, x]:.0%}", ha="center", va="center",
                            fontsize=7.5,
                            color="black" if 0.15 < g[y, x] < 0.85 else "white")
            ax.set_xlabel("alpha retained", fontsize=9)
            ax.set_ylabel("basket CAGR", fontsize=9)
            ax.set_title(f"{prof} — {'any market' if pool == 'all' else 'gold+BTC bull'}"
                         f"\nP(buy & hold wins)", fontsize=10, weight="bold")
    fig.suptitle("Probability buy & hold beats the strategy over 5 years "
                 "(paired paths, same market for both)",
                 fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(PNG, dpi=130)
    print(f"Wrote {PNG.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
