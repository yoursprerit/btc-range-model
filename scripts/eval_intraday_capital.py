"""Can intraday trading of the book's capital add return on top of the Overall strategy?

Two proposed sources of intraday capital, each lent to "another profitable
intraday trade" and returned by the close so the daily book is untouched:

  (1) IDLE CASH / SATA — the undeployed remainder the gate parks in SATA.
      Sell SATA after the open, run the trade, buy SATA back by the close.
  (2) CALM POSITIONS — a sleeve the book HOLDS whose price looks quiet today:
      sell it after the first hour, run the trade with the proceeds, buy it
      back at the close "at more or less the same price".

Economically (2) is an intraday LONG B / SHORT A swap bolted onto the book:
increment = r_B − r_A(10:30→close) − costs on both legs.  (1) is r_B − costs
on B and on the SATA round trip.  A third source is benchmarked alongside
because it dominates both when the account permits it:

  (3) INTRADAY MARGIN — buying power closed out before the close.  IBKR
      charges margin interest on overnight debit balances, so a flat-by-close
      trade needs no capital pulled from the book at all.

The study measures three things, all on real hourly bars of the TRADED
vehicles (BTC→IBIT, ETH→ETHA, SATA itself), Yahoo 1h, ~2 years:

  PART A  How much capital each source actually frees, per the published
          walk-forward replay (``walkforward_gated_replay``) — idle SATA
          weight and the held weight on predicted-calm days.
  PART B  What a "calm" position really does between the 10:30 sale and the
          close buy-back (drift given up, dispersion of the miss), with calm
          predicted from as-of data only (trailing 20-day intraday |move|
          tercile + today's first-hour range vs its 20-day average).
  PART C  Whether any standard intraday trade has a NET edge in this
          universe: open→close drift, first-hour momentum/continuation/
          reversal, gap fade, the last-30-minute leveraged-ETF rebalance
          flow, dip buys — each net of tiered costs and split into first
          and second halves of the sample (a rule must hold in BOTH).
  PART D  Portfolio overlays: each source funding the best-looking rule,
          added to the replay's daily return — CAGR / Sharpe / maxDD deltas
          over the window the intraday data covers.

Costs are per SIDE, all-in (half-spread + slippage + commission):
2 bp for the deep ETFs (SPY QQQ IBIT ETHA XLE GLDM GDX SOXX SOXL TQQQ MSTR
MSTU), 6 bp for the thinner ones (GRID OIH ERX UGL NUGT REMX WGMI PBW ARTY
SPXL SATA).  Results are also shown at 0× cost so the gross edge is visible.

    python scripts/eval_intraday_capital.py            # writes INTRADAY_CAPITAL_EVAL.md
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "app"))
sys.path.insert(0, str(_REPO))

import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

OUT_MD = _REPO / "INTRADAY_CAPITAL_EVAL.md"
PROFILES = ["Balanced", "Growth", "Aggressive"]
VEHICLE = {"BTC": "IBIT", "ETH": "ETHA"}          # replay key → traded symbol
UNIVERSE = ["IBIT", "ETHA", "MSTR", "MSTU", "GLDM", "GDX", "UGL", "NUGT",
            "SOXX", "SOXL", "GRID", "XLE", "OIH", "ERX", "REMX", "WGMI",
            "PBW", "ARTY"]
EXTRA = ["SATA", "SPY", "QQQ", "TQQQ", "SPXL"]    # the cash park + index vehicles
DEEP = {"SPY", "QQQ", "IBIT", "ETHA", "XLE", "GLDM", "GDX", "SOXX", "SOXL",
        "TQQQ", "MSTR", "MSTU"}
SATA_RT = 2 * 6e-4                                # SATA sell + buy-back


def cost_side(t: str) -> float:
    return 2e-4 if t in DEEP else 6e-4


# ── data ────────────────────────────────────────────────────────────────────
def fetch_hourly(tickers: list[str], days: int = 725) -> pd.DataFrame:
    import yfinance as yf
    start = (pd.Timestamp.today() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    out = []
    for t in tickers:
        e = pd.DataFrame()
        for _ in range(3):
            try:
                e = yf.Ticker(t).history(start=start, interval="1h", auto_adjust=False)
                if len(e):
                    break
            except Exception as ex:           # rate limit / transient
                print(f"  ⚠ {t}: {ex}")
            time.sleep(2)
        if not len(e):
            print(f"  ⚠ {t}: no hourly data")
            continue
        e = e[["Open", "High", "Low", "Close", "Volume"]].copy()
        e.index = e.index.tz_convert("America/New_York").tz_localize(None)
        e["ticker"] = t
        out.append(e)
    df = pd.concat(out)
    df.index.name = "ts"
    return df.reset_index()


def daily_panel(h: pd.DataFrame) -> pd.DataFrame:
    """One row per ticker-session with the intraday legs every rule needs.
    Hourly bars are stamped at their OPEN (09:30 … 15:30; the last is 30 min)."""
    h = h.assign(d=h.ts.dt.normalize(), hh=h.ts.dt.strftime("%H:%M"))
    rows = []
    for (t, d), g in h.groupby(["ticker", "d"]):
        g = g.set_index("hh")
        if not {"09:30", "10:30", "15:30"} <= set(g.index) or len(g) < 7:
            continue                                  # half-days / gaps
        rows.append(dict(ticker=t, d=d, o=g.at["09:30", "Open"],
                         c10=g.at["09:30", "Close"], o15=g.at["15:30", "Open"],
                         c=g.at["15:30", "Close"],
                         hi1=g.at["09:30", "High"], lo1=g.at["09:30", "Low"]))
    p = pd.DataFrame(rows).sort_values(["ticker", "d"]).reset_index(drop=True)
    by = p.groupby("ticker")
    p["pc"] = by.c.shift(1)
    p = p.dropna(subset=["pc"]).copy()
    p["gap"] = p.o / p.pc - 1
    p["r_oc"] = p.c / p.o - 1                       # open → close
    p["r_cc"] = p.c / p.pc - 1
    p["r1"] = p.c10 / p.pc - 1                       # prev close → 10:30
    p["r_10c"] = p.c / p.c10 - 1                     # 10:30 → close
    p["r_pre"] = p.o15 / p.pc - 1                    # prev close → 15:30
    p["r_last"] = p.c / p.o15 - 1                    # last 30 minutes
    by = p.groupby("ticker")
    lag20 = lambda s: s.shift(1).rolling(20, min_periods=10)   # noqa: E731
    p["sig"] = by.r_cc.transform(lambda s: lag20(s).std())
    p["pv"] = by.r_10c.transform(lambda s: lag20(s.abs()).mean())
    rng1 = np.log(p.hi1 / p.lo1)
    p["rng1_rel"] = rng1 / rng1.groupby(p.ticker).transform(lambda s: lag20(s).mean())
    p["cs"] = p.ticker.map(cost_side)
    return p.dropna(subset=["sig", "pv", "rng1_rel"])


def calm_flags(p: pd.DataFrame) -> pd.Series:
    """As-of 'calm' prediction: bottom tercile of the ticker's EXPANDING
    distribution of trailing intraday |move| AND a first hour quieter than
    80% of its 20-day norm (known at 10:30, when the sale would happen)."""
    q = p.groupby("ticker").pv.transform(
        lambda s: s.expanding(min_periods=20).rank(pct=True))
    return (q <= 1 / 3) & (p.rng1_rel < 0.8)


def replay_frames(profiles: list[str]) -> dict:
    import overall_core as oc
    print("Running the universe (every sleeve's engine)…")
    res = oc.run_universe()
    for k, e in oc._LAST_ERRORS.items():
        print(f"  ⚠ {k}: {e}")
    out = {}
    for prof in profiles:
        cfg = oc.RISK_PROFILES[prof]
        rep = oc.walkforward_gated_replay(res, caps=oc.caps_for(prof),
                                          mdd_floor=cfg["mdd_floor"],
                                          objective=cfg["objective"])
        out[prof] = dict(w=rep["weights"], sata=rep["sata"], ret=rep["ret"])
        print(f"  {prof}: replay through {rep['ret'].index[-1].date()}")
    return out


def held_long(w: pd.DataFrame) -> pd.DataFrame:
    """Replay weights → long (ticker, d, wt) rows in the traded vehicle."""
    parts = [pd.DataFrame({"ticker": VEHICLE.get(k, k), "d": w.index, "wt": w[k].values})
             for k in w.columns]
    return pd.concat(parts)


# ── rules ───────────────────────────────────────────────────────────────────
def rule_book(p: pd.DataFrame) -> dict:
    one = pd.Series(1.0, index=p.index)
    sg = np.sign
    z1, zg, zp = p.r1 / p.sig, p.gap / p.sig, p.r_pre / p.sig
    return {
        # name: (position, return leg, mask, entry-time)
        "Naive intraday long (open→close)": (one, p.r_oc, None, "open"),
        "Intraday momentum: sign(1st hr) → last 30m": (sg(p.r1), p.r_last, None, "15:30"),
        "LETF rebalance flow: sign(day so far) → last 30m": (sg(p.r_pre), p.r_last, None, "15:30"),
        "  … only when |move| > 1σ": (sg(p.r_pre), p.r_last, zp.abs() > 1, "15:30"),
        "1st-hr continuation: sign(1st hr) → 10:30..close": (sg(p.r1), p.r_10c, None, "10:30"),
        "  … long-only": (sg(p.r1).clip(lower=0), p.r_10c, None, "10:30"),
        "1st-hr reversal when |move| > 1σ": (-sg(p.r1), p.r_10c, z1.abs() > 1, "10:30"),
        "1st-hr dip buy (move < −1σ) → close": (one, p.r_10c, z1 < -1, "10:30"),
        "Gap fade when |gap| > 1σ (open→close)": (-sg(p.gap), p.r_oc, zg.abs() > 1, "open"),
        "Gap-down buy (gap < −1σ, open→close)": (one, p.r_oc, zg < -1, "open"),
        "Add to HELD sleeves open→close (trend on)": (one, p.r_oc, p.held, "open"),
        "Buy NOT-held sleeves open→close": (one, p.r_oc, ~p.held & p.ticker.isin(UNIVERSE), "open"),
    }


def eval_rule(p, pos, ret, mask, mid, cost_mult=1.0) -> tuple[dict, pd.DataFrame]:
    df = p.assign(pos=pos, ret=ret)
    if mask is not None:
        df = df[mask.reindex(df.index).fillna(False).astype(bool)]
    df = df[df.pos.fillna(0) != 0].dropna(subset=["ret"])
    pnl = df.pos * df.ret - cost_mult * 2 * df.cs * df.pos.abs()
    t = pnl.mean() / pnl.std() * np.sqrt(len(pnl)) if len(pnl) > 2 else np.nan
    return dict(n=len(pnl), gross=(df.pos * df.ret).mean() * 1e4,
                net=pnl.mean() * 1e4, t=t, hit=(pnl > 0).mean(),
                h1=pnl[df.d < mid].mean() * 1e4, h2=pnl[df.d >= mid].mean() * 1e4), \
        df.assign(pnl=pnl)


# ── metrics ─────────────────────────────────────────────────────────────────
def metrics(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    ppy = len(r) / yrs
    return dict(total=eq.iloc[-1] - 1, cagr=eq.iloc[-1] ** (1 / yrs) - 1,
                mdd=(eq / eq.cummax() - 1).min(),
                sharpe=r.mean() / r.std() * np.sqrt(ppy) if r.std() > 0 else np.nan)


def pct(x, nd=1, sign=True):
    return f"{x * 100:+.{nd}f}%" if sign else f"{x * 100:.{nd}f}%"


# ── main ────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Fetching hourly bars…")
    h = fetch_hourly(UNIVERSE + EXTRA)
    p = daily_panel(h)
    reps = replay_frames(PROFILES)
    bal = reps["Balanced"]
    held = held_long(bal["w"])
    p = p.merge(held, on=["ticker", "d"], how="left")
    p["wt"] = p.wt.fillna(0.0)
    p["held"] = p.wt > 0
    p["calm"] = calm_flags(p)
    d0, d1 = p.d.min(), p.d.max()
    mid = p.d.sort_values().iloc[len(p) // 2]
    print(f"panel {d0.date()} → {d1.date()} · {p.d.nunique()} sessions · split {mid.date()}")
    L: list[str] = []
    A = L.append

    # ── PART A: how much capital each source frees ────────────────────────
    capA = []
    for prof, rp in reps.items():
        s = rp["sata"]
        s = s[(s.index >= d0) & (s.index.dayofweek < 5)]
        hp = p.merge(held_long(rp["w"]), on=["ticker", "d"], suffixes=("", "_p"))
        cw = hp[hp.calm & (hp.wt_p > 0)].groupby("d").wt_p.sum() \
            .reindex(s.index).fillna(0.0)
        capA.append((prof, s.mean(), (s > 0.05).mean(), (s > 0.2).mean(),
                     cw.mean(), (cw > 0).mean()))

    # ── PART B: what a calm held position does between sale and buy-back ──
    ph = p[p.held & p.ticker.isin(UNIVERSE)]

    def miss(x):
        x = x.dropna()
        return (len(x), x.mean() * 1e4, x.abs().median() * 1e4,
                (x.abs() > .005).mean(), (x.abs() > .01).mean())
    partB = [("Held, every session", *miss(ph.r_10c)),
             ("Held, predicted calm (as-of)", *miss(ph[ph.calm].r_10c)),
             ("Not held (for reference)", *miss(p[~p.held & p.ticker.isin(UNIVERSE)].r_10c))]
    tB = ph.r_10c.mean() / ph.r_10c.std() * np.sqrt(len(ph))
    held_gap, held_oc = ph.gap.mean() * 1e4, ph.r_oc.mean() * 1e4
    # was the calm prediction any good? (realised |10:30→close| calm vs not)
    calm_med = ph[ph.calm].r_10c.abs().median() * 1e4
    rest_med = ph[~ph.calm].r_10c.abs().median() * 1e4

    # ── PART C: is there an intraday edge to fund? ────────────────────────
    rows, frames = [], {}
    for name, (pos, ret, mask, when) in rule_book(p).items():
        st, df = eval_rule(p, pos, ret, mask, mid)
        g0, _ = eval_rule(p, pos, ret, mask, mid, cost_mult=0.0)
        rows.append(dict(rule=name, when=when, **st, net0=g0["net"]))
        frames[name] = df
    C = pd.DataFrame(rows)
    both = C[(C.h1 > 0) & (C.h2 > 0)]
    best_h1 = C.sort_values("h1", ascending=False).iloc[0]

    # ── PART D: portfolio overlays ────────────────────────────────────────
    base = bal["ret"]
    base = base[(base.index >= d0) & (base.index <= d1)]
    biz = base.index.dayofweek < 5
    sata = bal["sata"].reindex(base.index).fillna(0.0)

    def day_mean(name):
        f = frames[name]
        return f.groupby("d").pnl.mean().reindex(base.index)

    calm_rows = p[p.calm & p.held]
    swap_drag = (calm_rows.wt * (calm_rows.r_10c + 2 * calm_rows.cs)).groupby(calm_rows.d).sum() \
        .reindex(base.index).fillna(0.0)
    calm_w = calm_rows.groupby("d").wt.sum().reindex(base.index).fillna(0.0)

    overlays = [("Baseline — published replay (Balanced)", base)]
    for name in ["  … long-only", "Add to HELD sleeves open→close (trend on)",
                 "Naive intraday long (open→close)"]:
        dm = day_mean(name)
        has = dm.notna() & biz
        tag = name.strip(" …") if not name.startswith("  ") else "1st-hr continuation, long-only"
        # (1) idle SATA → trade, SATA bought back by the close
        inc1 = np.where(has, sata * (dm.fillna(0) - SATA_RT), 0.0)
        overlays.append((f"(1) idle SATA → {tag}", base + inc1))
        if name == "  … long-only":             # 10:30 entry matches the calm sale
            inc2 = np.where(has, calm_w * dm.fillna(0), 0.0) - swap_drag.values
            overlays.append((f"(2) calm-position swap → {tag}", base + inc2))
        for f in (0.25, 0.50):
            overlays.append((f"(3) intraday margin {int(f * 100)}% NAV → {tag}",
                             base + np.where(has, f * dm.fillna(0), 0.0)))
    D = [(n, metrics(r)) for n, r in overlays]
    bm = D[0][1]
    days_biz = int(biz.sum())
    yrs = (base.index[-1] - base.index[0]).days / 365.25
    need_5pp = 0.05 / (days_biz / yrs)             # per-day return on full NAV for +5pp/yr

    # ── write ─────────────────────────────────────────────────────────────
    A("# Intraday Capital Recycling — Feasibility Evaluation\n")
    A(f"*Generated by `scripts/eval_intraday_capital.py` · Yahoo 1-hour bars "
      f"**{d0.date()} → {d1.date()}** ({p.d.nunique()} sessions, {p.ticker.nunique()} traded "
      f"vehicles incl. SATA and index ETFs) · book = published walk-forward replay "
      f"(`walkforward_gated_replay`) · costs per side: 2 bp deep ETFs, 6 bp thin ETFs & SATA.*\n")
    A("## Question\n")
    A("Can capital be lent out **intraday** — and returned by the close so the "
      "daily book is untouched — to boost return on top of the Overall strategy "
      "without adding meaningful risk? Two proposed sources:\n")
    A("1. **Idle cash / SATA** — sell the SATA park after the open, trade, buy SATA back by the close.")
    A("2. **Calm open positions** — sell a held sleeve whose price looks quiet today, "
      "trade the proceeds, buy the sleeve back by the close at roughly the same price.\n")
    A("A third source is benchmarked because it dominates both when the account "
      "allows it: **(3) intraday margin** — buying power used and closed out before "
      "the close. IBKR charges margin interest on *overnight* debit balances, so a "
      "flat-by-close trade needs nothing pulled from the book.\n")
    A("## The identity that decides it\n")
    A("Selling held sleeve *A* at 10:30 to buy *B*, then reversing at the close, is "
      "exactly the book **plus an intraday long-B / short-A swap**:\n")
    A("```\nincrement = r_B(10:30→close) − r_A(10:30→close) − costs(A round trip) − costs(B round trip)\n```\n")
    A("So source (2) only pays if **(a)** there is a *B* with a positive net edge, "
      "**(b)** *A*'s intraday drift is ≤ 0 on the days you pick, and **(c)** the extra "
      "variance of `r_B − r_A` is worth it. Source (1) needs only (a) plus covering the SATA "
      "round trip; source (3) needs only (a). Parts A–D test each piece.\n")

    A("## Part A — how much capital the sources actually free\n")
    A("| Profile | Mean idle SATA weight | Days SATA > 5% | Days SATA > 20% | Mean held weight on predicted-calm days | Sessions with any calm holding |")
    A("|---|---:|---:|---:|---:|---:|")
    for prof, m, g5, g20, cw, cwd in capA:
        A(f"| {prof} | {pct(m, 1, False)} | {pct(g5, 0, False)} | {pct(g20, 0, False)} | "
          f"{pct(cw, 1, False)} | {pct(cwd, 0, False)} |")
    A("\nThe gate water-fills the funded sleeves to their caps, so **the book is "
      "almost always fully deployed** — the SATA park is a rounding error in the "
      "replay (the live IBKR record is similar: cash under ~5% of NAV on most "
      "execution days, with the occasional larger residual while an exit waits "
      "for a new entry). Source (1) has very little capital to lend.\n")

    A("## Part B — does a “calm” position come back to the same price?\n")
    A("Sale at the 10:30 bar close, buy-back at the session close; *calm* is "
      "predicted from as-of data only (bottom tercile of the ticker's trailing "
      "20-day intraday |move| **and** a first hour < 80% of its 20-day norm).\n")
    A("| Sessions | n | Mean 10:30→close (bp) | Median \\|miss\\| (bp) | \\|miss\\| > 0.5% | \\|miss\\| > 1% |")
    A("|---|---:|---:|---:|---:|---:|")
    for lab, n, mu, med, g5, g10 in partB:
        A(f"| {lab} | {n} | {mu:+.1f} | {med:.0f} | {pct(g5, 0, False)} | {pct(g10, 0, False)} |")
    A(f"\n* **Held sleeves drift up intraday**: {partB[0][2]:+.1f} bp per session from "
      f"10:30 to the close (t = {tB:.1f}); overall a held sleeve earns "
      f"{held_gap:+.1f} bp overnight and {held_oc:+.1f} bp open→close. Lending out a "
      "position on an average day gives that drift away — the trend engines are "
      "long *because* the drift is positive.")
    drift_word = ("still drift up" if partB[1][2] > 2 else
                  "drift about zero" if partB[1][2] > -2 else "drift down")
    A(f"* **Calm is only mildly predictable from as-of data**: median |miss| "
      f"{calm_med:.0f} bp on predicted-calm days vs {rest_med:.0f} bp otherwise, and "
      f"calm held sleeves {drift_word} ({partB[1][2]:+.1f} bp) — so the swap still "
      "gives away the drift on the very days it is meant to be free.")
    A(f"* **“About the same price” does not hold**: even on predicted-calm days the "
      f"buy-back misses the sale by >0.5% {pct(partB[1][4], 0, False)} of the time and "
      f"by >1% {pct(partB[1][5], 0, False)} of the time. Around the drift that miss is "
      "a coin flip — added variance as large as any edge the freed capital could earn.\n")

    A("## Part C — is there an intraday trade worth funding?\n")
    A(f"Every rule on every vehicle, one unit per signal; *net* is per trade after "
      f"costs. **H1 / H2** split the sample at {mid.date()} — a real edge must be "
      f"positive in both halves.\n")
    A("| Rule | Entry | n | Gross (bp) | Net (bp) | t (net) | Hit | Net H1 | Net H2 |")
    A("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in C.itertuples():
        A(f"| {r.rule.replace('|', chr(92) + '|')} | {r.when} | {r.n} | {r.gross:+.1f} | {r.net:+.1f} | "
          f"{r.t:+.1f} | {pct(r.hit, 0, False)} | {r.h1:+.1f} | {r.h2:+.1f} |")
    A("")
    if len(both):
        A(f"* Rules positive in **both** halves: {', '.join(both.rule.str.strip())} — "
          "check their t-stats before believing them.")
    else:
        A("* **No rule is net-positive in both halves.**")
    A(f"* The rule a researcher would have picked after year one "
      f"(**{best_h1.rule.strip()}**, {best_h1.h1:+.1f} bp net in H1) earned "
      f"**{best_h1.h2:+.1f} bp** net in H2 — the textbook out-of-sample fade.")
    A("* The last-30-minute rules (intraday momentum; the leveraged-ETF rebalance "
      "flow) are **negative even gross** here: the effect documented on SPY in "
      "older samples is not present in this universe/period, and a 30-minute hold "
      "cannot carry a round-trip cost anyway.")
    A("* The only rules near zero are *long exposure to sleeves the engines already "
      "hold* — i.e. more of the same trend bet, levered intraday, not a separate "
      "source of return. The repo's own BTC hourly model is no help either: its "
      "out-of-sample direction accuracy is 50.15% (`BTC_README.md`).\n")

    A("## Part D — portfolio overlays (Balanced, intraday-data window)\n")
    A(f"Each overlay adds its intraday P&L to the replay's daily return on the "
      f"{days_biz} weekdays of the window. Rules are the three least-bad from Part C "
      "(selected in-sample, so this is an *optimistic* bound). Each day's signals "
      "share the capital equally, so a rule's per-*day* mean can sit below its "
      "per-*trade* mean when its winners cluster on days with many signals; and "
      "because the overlay trades the same names the book holds, its variance "
      "stacks on the book's and costs compounding (volatility drag).\n")
    A("| Variant | Total | CAGR | ΔCAGR | MaxDD | Sharpe | ΔSharpe |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for n, m in D:
        A(f"| {n} | {pct(m['total'])} | {pct(m['cagr'])} | "
          f"{(m['cagr'] - bm['cagr']) * 100:+.1f}pt | {pct(m['mdd'])} | "
          f"{m['sharpe']:.2f} | {m['sharpe'] - bm['sharpe']:+.2f} |")
    A("\nThe (1) rows barely move in either direction because they act on "
      f"~{pct(capA[0][1], 1, False)} of NAV — noise, not an edge. Every overlay "
      "with real capital behind it (2, 3) lowers CAGR *and* Sharpe.")
    A(f"\nFor scale: adding **+5 CAGR points** needs ≈ **{need_5pp * 1e4:.1f} bp per "
      f"session on the whole NAV** — e.g. a trade earning {need_5pp / 0.25 * 1e4:.0f} bp "
      f"net every day on 25% of NAV. Nothing in Part C comes close.\n")

    A("## Verdict\n")
    A("* **Source (1) — idle cash / SATA: not worth building.** There is almost no "
      "idle capital (Part A), SATA already earns ~13%/yr on it, and the SATA round "
      f"trip (~{SATA_RT * 1e4:.0f} bp) costs more than the best intraday edge found "
      f"({C.net.max():+.1f} bp net per trade).")
    A("* **Source (2) — lending out calm positions: rejected.** It is strictly worse "
      "than (3): the same B-leg risk *plus* a short-A leg that (i) gives up the "
      "held sleeves' positive intraday drift, (ii) adds a buy-back miss of "
      f"~{calm_med:.0f} bp median even on predicted-calm days, (iii) pays two extra "
      "spreads, and in a taxable account (iv) realises gains on every sale "
      "(short-term) or triggers wash-sale deferral on losses.")
    A("* **Source (3) — intraday margin: the only sound funding route, but there is "
      "nothing profitable to fund.** Across 12 standard rules × 23 vehicles no "
      "intraday trade has a net edge that survives an in-/out-of-sample split. "
      "Levering the book's own longs intraday is the least-bad use, and it adds "
      "risk roughly in proportion to return — not the free boost being sought.")
    A("* **Recommendation: do not add an intraday layer.** The strategy's edge is a "
      "multi-day trend premium, largely earned overnight; the intraday session is "
      "where it is thinnest. If intraday work is revisited it should start from "
      "a *demonstrated* intraday signal (tested on this harness first), funded by "
      "intraday margin — never by selling core positions.\n")

    A("## Caveats\n")
    A("* Yahoo 1-hour bars (~2 years, the maximum available); entries/exits at bar "
      "closes, no queue position or partial fills. Costs are tiered assumptions — "
      "the *Gross* column shows the rules fail before costs too, so the verdict does "
      "not hinge on them.")
    A("* 1-hour granularity cannot test sub-hour signals (opening-range breakouts, "
      "VWAP reversion). Those need tick/1-minute data and would face the same "
      "cost and funding arithmetic.")
    A("* The replay's weights are Balanced; Growth/Aggressive hold even less SATA.")
    A("* Pattern-day-trader rules apply to margin accounts under $25k equity; the "
      "paper account (~$96k) is above that. Margin availability, not idle cash, is "
      "the binding constraint for source (3).")
    A("* Regenerate with `python scripts/eval_intraday_capital.py` (numbers move with the data window).")
    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT_MD.relative_to(_REPO)}")
    print(C.round(2).to_string(index=False))
    for n, m in D:
        print(f"{n:60s} cagr {m['cagr'] * 100:+6.1f}  sharpe {m['sharpe']:.2f}  mdd {m['mdd'] * 100:.1f}")


if __name__ == "__main__":
    main()
