"""Are the BTC strategy's ENTRY signals or its EXIT signals the real edge?

Splits the BTC divergence strategy (btc_ct_engine — the same engine the Overall
app runs) into its two halves and tests each for statistical significance, per
sleeve (BTC · MSTR · MSTU by default; ETH via --assets):

  A. Event study — the sleeve's forward log-return h bars after a FRESH signal
     (rising edge) minus the unconditional mean.  p-values from a circular-shift
     permutation of the signal mask (keeps the signals' clustering), one-sided
     in the signal's expected direction (entry ↑, exit ↓).  Run both on the raw
     signal and on the trades the backtest actually executed.

  B. Trade-level decomposition (Monte Carlo):
       ENTRY skill — random entries (same per-bar rate as the live gate) + the
                     real exit rule / stop / SL re-entry  → p(null ≥ observed).
       EXIT skill  — the real entries + random holding periods drawn from the
                     observed signal-exit durations (the fixed stop stays live,
                     so only the SIGNAL exit is replaced) → p(null ≥ observed).

  C. Exit drawdown test — same real entries / random holds as B, but scored on
     risk instead of return: the strategy's max drawdown and its worst in-trade
     drawdown (peak-to-trough while long).  p = share of random-hold runs whose
     drawdown is at least as shallow as the real exits'.  Low p ⇒ the exits
     genuinely cut drawdown beyond what an equal-length hold would.

  D. Exit PACKAGE (signal exit + fixed stop scored together) — real entries
     under live / signal-only / stop-only exits, against random holds drawn
     from all observed holds with NO stop (so the stop is credited to the
     package, unlike B/C) → p for return, MDD and Sharpe.

  E. Full strategy vs buy-and-hold — circular block bootstrap of the paired
     daily returns (P(excess ≤ 0), P(Sharpe ≤ B&H)), a random-timing null
     (random entries at the live rate + random holds), and reliability: share
     of rolling 126-bar windows beating B&H plus per-calendar-year excess.

The simulator in B/C mirrors btc_ct_engine._run_bt (signal exits, fixed stop,
SL5 regime-adaptive re-entry, post-stop override); the "observed" line is the
simulator under the real rules and is printed next to the engine's own result
so any drift is visible.  The null runs disable the post-stop override (it is
itself a U1-based entry, which would leak entry skill into the random-entry
null).

Caveats: ~2.5 years of daily bars, ~10 trades per sleeve, several horizons
tested, and the U1/D2 thresholds were tuned on this same window — every p-value
here is optimistic.  The three sleeves share one BTC signal, so they are NOT
independent confirmations of each other.

    python scripts/eval_entry_exit_significance.py
    python scripts/eval_entry_exit_significance.py --assets BTC MSTR MSTU ETH --sims 5000
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("BTC_ALLOW_SELF_PULL", "0")   # offline: use committed data

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "app"))
sys.path.insert(0, str(_REPO))

import btc_ct_engine as E   # noqa: E402

T = E.T
START = "2024-01-01"
HORIZONS = (1, 3, 5, 10, 20)
EXEC_HORIZONS = (3, 5, 10, 20)
SL_COOLDOWN = 10            # _run_bt: re-entry after a stop waits 10 bars unless bull
BARS_PER_YEAR = 365         # CT bars are calendar-daily (equity sleeves carry fills over weekends)
BLOCK = 20                  # block-bootstrap block length (~1 month)
ROLL = 126                  # rolling reliability window (~6 months of bars)


# ── data ─────────────────────────────────────────────────────────────────────
def load():
    rf = T.load_raw_features()
    preds = T.build_preds_offline(rf)
    comp = T.prep(rf, preds, START, str(rf.index[-1].date()))
    dates = pd.DatetimeIndex(comp["target_date"])
    sigs = E.compute_sigs_pure(comp)
    px = E._load_prices(dates, comp)
    return dates, sigs, px


def sleeve_sigs(sigs: dict, key: str) -> dict:
    s = dict(sigs)
    s["tf2_entry"] = (sigs["tf2_entry_ma"] if E.GATE_BY_ASSET[key] == "above_ma30"
                      else sigs["tf2_entry_pure"])
    return s


# ── simulator mirroring btc_ct_engine._run_bt ────────────────────────────────
def simulate(lp, b0, entry, exit_sig, stop, bull, override_ok, hold_fn=None):
    """Return (trades [(i, j, was_sl)], pos[N]) — pos[t]=1 when long over t-1→t.

    entry / exit_sig / bull / override_ok are bool arrays; hold_fn(i) → bars to
    hold replaces the signal exit (stop still applies) when given."""
    N = len(lp)
    stop_lvl = -np.inf if stop is None else np.log(1.0 - stop)
    trades, pos = [], np.zeros(N)
    from_sl, since_sl, i = False, 0, b0
    while i < N - 1:
        if from_sl:
            since_sl += 1
        exit_now = exit_sig[i]
        reentry_ok = (not from_sl) or bull[i] or since_sl >= SL_COOLDOWN
        override = from_sl and since_sl <= E.REENTRY_OVERRIDE_BARS and override_ok[i] and not exit_now
        if not ((entry[i] and reentry_ok and (from_sl or not exit_now)) or override):
            i += 1
            continue
        target = N - 1 if hold_fn is None else min(i + max(1, hold_fn(i)), N - 1)
        j, was_sl = N - 1, False
        for k in range(i + 1, N):
            if lp[k] - lp[i] <= stop_lvl:
                j, was_sl = k, True
                break
            if (hold_fn is None and exit_sig[k]) or (hold_fn is not None and k >= target):
                j = k
                break
        pos[i + 1:j + 1] = 1.0
        trades.append((i, j, was_sl))
        from_sl, since_sl = was_sl, 0
        i = j + 1                      # the exit bar itself is never a re-entry bar
    return trades, pos


def score(lp, trades, pos):
    """Total log-return, max drawdown of the NAV path, worst in-trade drawdown."""
    r = np.r_[0.0, np.diff(lp)] * pos
    nav = np.cumsum(r)
    mdd = float((nav - np.maximum.accumulate(nav)).min())
    worst = 0.0
    for i, j, _ in trades:
        seg = lp[i:j + 1] - lp[i]
        worst = min(worst, float((seg - np.maximum.accumulate(seg)).min()))
    tot = float(sum(lp[j] - lp[i] for i, j, _ in trades))
    return tot, mdd, worst


# ── A. event study ───────────────────────────────────────────────────────────
def fresh(mask, b0):
    m = mask.copy()
    m[:b0] = False
    return m & ~np.r_[False, m[:-1]]


def circ_p(lp, b0, mask, h, sign, rng, k=5000):
    N = len(lp)
    r = np.full(N, np.nan)
    r[:N - h] = lp[h:] - lp[:N - h]
    valid = np.isfinite(r)
    valid[:b0] = False
    idx = np.flatnonzero(valid)
    m, rv = mask[idx], r[idx]
    if not m.any():
        return 0, float("nan"), float("nan")
    obs = rv[m].mean() - rv.mean()
    null = np.array([rv[np.roll(m, rng.integers(1, len(m)))].mean() - rv.mean()
                     for _ in range(k)])
    return int(m.sum()), obs * 100, float(np.mean(sign * null >= sign * obs))


# ── per-sleeve report ────────────────────────────────────────────────────────
def run_sleeve(key, dates, sigs, px_all, n_sims, seed):
    rng = np.random.default_rng(seed)
    px = px_all[key].astype(float)
    fin = np.isfinite(px) & (px > 0)
    n = int(np.flatnonzero(fin)[-1]) + 1                 # drop pending-fill NaN tail
    dates, px = dates[:n], px[:n]
    s = {k: (v[:n] if isinstance(v, np.ndarray) else v) for k, v in sleeve_sigs(sigs, key).items()}
    lp = np.log(px)
    b0 = max(T.WARMUP, int(dates.searchsorted(pd.Timestamp(START))))
    stop = E.STOP_PCT[key]
    entry = s["tf2_entry"].astype(bool)
    bull = s["bull_regime"].astype(bool)
    exit_sig = s["d3"] | (s["d2"] & ~bull)
    override_ok = s["u1"] & s["above_ma30"]
    no_ovr = np.zeros(n, bool)

    eng = E._run_bt(dates, px, s, stop, pd.Timestamp(START))
    tr_obs, pos_obs = simulate(lp, b0, entry, exit_sig, stop, bull, override_ok)
    tot, mdd, worst = score(lp, tr_obs, pos_obs)
    bh_lp = lp[b0:] - lp[b0]
    bh_mdd = float((bh_lp - np.maximum.accumulate(bh_lp)).min())

    print("=" * 100)
    print(f"{key}  ·  gate={'Standard MA (XOR)' if E.GATE_BY_ASSET[key] == 'above_ma30' else 'Pure Regime'}"
          f"  ·  stop={'none' if stop is None else f'−{stop:.0%}'}  ·  "
          f"{dates[b0].date()} → {dates[-1].date()} ({n - b0} bars)")
    eng_tr = eng["trades"]
    print(f"  engine : {len(eng_tr)} trades, win {np.mean([t['ret'] > 0 for t in eng_tr]):.0%}, "
          f"strategy {eng['nav'].iloc[-1] / eng['nav'].iloc[0] - 1:+.1%} vs B&H "
          f"{eng['bh'].iloc[-1] / eng['bh'].iloc[0] - 1:+.1%}")
    print(f"  sim    : {len(tr_obs)} trades ({sum(t[2] for t in tr_obs)} stops), "
          f"strategy {np.expm1(tot):+.1%}, MDD {np.expm1(mdd):+.1%} (B&H MDD {np.expm1(bh_mdd):+.1%}), "
          f"worst in-trade DD {np.expm1(worst):+.1%}")

    print("\n  A. Event study — fwd log-return after signal minus unconditional (one-sided circular-shift p)")
    ei, xi = np.zeros(n, bool), np.zeros(n, bool)
    for t in eng_tr:
        ei[dates.get_loc(pd.Timestamp(t["entry_date"]))] = True
        if not t["was_sl"]:
            xi[dates.get_loc(pd.Timestamp(t["exit_date"]))] = True
    ei[:b0 + 1] = False                  # the first bar is a forced warm start, not a signal
    rows = [("entry signal (fresh)", fresh(entry, b0), +1, HORIZONS),
            ("exit signal (fresh)", fresh(exit_sig, b0), -1, HORIZONS),
            ("executed entries", ei, +1, EXEC_HORIZONS),
            ("executed signal exits", xi, -1, EXEC_HORIZONS)]
    for name, mask, sign, hs in rows:
        cells = []
        for h in hs:
            cnt, ex, p = circ_p(lp, b0, mask, h, sign, rng)
            star = "*" if p < 0.05 else " "
            cells.append(f"h{h:<2d} {ex:+6.2f}% p={p:.3f}{star}")
        print(f"    {name:24s} n={cnt:3d} | " + " | ".join(cells))

    # B/C Monte Carlo
    sig_durs = np.array([j - i for i, j, sl in tr_obs if not sl] or [1])
    p_ent = entry[b0:].mean()
    ent_null, ext_null = [], []
    for _ in range(n_sims):
        m = rng.random(n) < p_ent
        tr, pos = simulate(lp, b0, m, exit_sig, stop, bull, no_ovr)
        ent_null.append(score(lp, tr, pos))
        tr, pos = simulate(lp, b0, entry, exit_sig, stop, bull, no_ovr,
                           hold_fn=lambda i: int(rng.choice(sig_durs)))
        ext_null.append(score(lp, tr, pos))
    ent_null, ext_null = np.array(ent_null), np.array(ext_null)

    print(f"\n  B. Trade-level decomposition ({n_sims} sims) — total return")
    print(f"    ENTRY skill (random entries, real exits)   null median {np.expm1(np.median(ent_null[:, 0])):+7.1%}"
          f"  observed {np.expm1(tot):+7.1%}  p={np.mean(ent_null[:, 0] >= tot):.3f}")
    print(f"    EXIT  skill (real entries, random holds)    null median {np.expm1(np.median(ext_null[:, 0])):+7.1%}"
          f"  observed {np.expm1(tot):+7.1%}  p={np.mean(ext_null[:, 0] >= tot):.3f}")

    print(f"\n  C. Exit drawdown test (real entries, random holds; p = share of nulls at least as shallow)")
    print(f"    max drawdown          null median {np.expm1(np.median(ext_null[:, 1])):+7.1%}"
          f"  observed {np.expm1(mdd):+7.1%}  p={np.mean(ext_null[:, 1] >= mdd):.3f}")
    print(f"    worst in-trade DD     null median {np.expm1(np.median(ext_null[:, 2])):+7.1%}"
          f"  observed {np.expm1(worst):+7.1%}  p={np.mean(ext_null[:, 2] >= worst):.3f}")

    exit_package(lp, b0, entry, exit_sig, stop, bull, override_ok, tr_obs, pos_obs, n_sims, rng)
    vs_buy_hold(lp, b0, dates, entry, tr_obs, pos_obs, n_sims, rng)
    print()


def sharpe(lp, pos, b0):
    r = (np.expm1(np.r_[0.0, np.diff(lp)]) * pos)[b0 + 1:]
    return float(r.mean() / r.std() * np.sqrt(BARS_PER_YEAR)) if r.std() > 0 else 0.0


def exit_package(lp, b0, entry, exit_sig, stop, bull, override_ok, tr_obs, pos_obs, n_sims, rng):
    """D. Exit = signal exit + fixed stop, scored as ONE package.

    Same real entries throughout; only the exit rule changes.  The null holds
    each trade for a random length drawn from ALL observed holds (stopped ones
    included) with NO stop, so the stop's contribution is credited to the
    package instead of being kept in the null as in B/C."""
    n = len(lp)
    no_exit = np.zeros(n, bool)
    variants = [("signal + stop (live)", exit_sig, stop, override_ok),
                ("signal only", exit_sig, None, override_ok),
                ("stop only", no_exit, stop, override_ok)]
    print(f"\n  D. Exit package — real entries, exit rule varied "
          f"(stop {'none — package = signal only' if stop is None else f'−{stop:.0%}'})")
    print(f"    {'exit rule':24s} {'trades':>6s} {'return':>9s} {'MDD':>8s} {'Sharpe':>7s}")
    for name, ex, st, ovr in variants:
        if st is None and ex is no_exit:
            continue
        tr, pos = simulate(lp, b0, entry, ex, st, bull, ovr)
        tot, mdd, _ = score(lp, tr, pos)
        print(f"    {name:24s} {len(tr):6d} {np.expm1(tot):+9.1%} {np.expm1(mdd):+8.1%} {sharpe(lp, pos, b0):7.2f}")
    tot, mdd, _ = score(lp, tr_obs, pos_obs)
    durs = np.array([j - i for i, j, _ in tr_obs] or [1])
    null = []
    for _ in range(n_sims):
        tr, pos = simulate(lp, b0, entry, no_exit, None, bull, np.zeros(n, bool),
                           hold_fn=lambda i: int(rng.choice(durs)))
        null.append((*score(lp, tr, pos)[:2], sharpe(lp, pos, b0)))
    null = np.array(null)
    sh = sharpe(lp, pos_obs, b0)
    print(f"    {'random holds, no stop':24s} {'':6s} {np.expm1(np.median(null[:, 0])):+9.1%} "
          f"{np.expm1(np.median(null[:, 1])):+8.1%} {np.median(null[:, 2]):7.2f}   ← null median")
    print(f"    package p-values vs null:  return p={np.mean(null[:, 0] >= tot):.3f}  "
          f"MDD p={np.mean(null[:, 1] >= mdd):.3f}  Sharpe p={np.mean(null[:, 2] >= sh):.3f}")


def vs_buy_hold(lp, b0, dates, entry, tr_obs, pos_obs, n_sims, rng):
    """E. Does the full strategy (entries + signal exits + stop) beat buy-and-hold,
    and how reliably?"""
    n = len(lp)
    r_bh = np.expm1(np.r_[0.0, np.diff(lp)])[b0 + 1:]
    r_st = r_bh * pos_obs[b0 + 1:]
    tot_st, tot_bh = np.log1p(r_st).sum(), np.log1p(r_bh).sum()
    obs_ex = tot_st - tot_bh
    sh_st, sh_bh = sharpe(lp, pos_obs, b0), sharpe(lp, np.ones(n), b0)
    bh_lp = lp[b0:] - lp[b0]
    bh_mdd = float((bh_lp - np.maximum.accumulate(bh_lp)).min())
    st_mdd = score(lp, tr_obs, pos_obs)[1]

    # circular block bootstrap of the PAIRED daily returns (keeps vol clustering
    # and the strategy/B&H dependence) → distribution of log excess & Sharpe gap
    m = len(r_bh)
    nb = int(np.ceil(m / BLOCK))
    ex_bs, dsh_bs = [], []
    for _ in range(n_sims):
        idx = ((rng.integers(0, m, nb)[:, None] + np.arange(BLOCK)) % m).ravel()[:m]
        a, b = r_st[idx], r_bh[idx]
        ex_bs.append(np.log1p(a).sum() - np.log1p(b).sum())
        sa = a.mean() / a.std() if a.std() > 0 else 0.0
        dsh_bs.append((sa - b.mean() / b.std()) * np.sqrt(BARS_PER_YEAR))
    ex_bs, dsh_bs = np.array(ex_bs), np.array(dsh_bs)
    lo, hi = np.percentile(ex_bs, [5, 95])

    # random-timing null: random entries at the live rate, random holds, no stop
    p_ent = entry[b0:].mean()
    durs = np.array([j - i for i, j, _ in tr_obs] or [1])
    zero = np.zeros(n, bool)
    rt = []
    for _ in range(n_sims):
        tr, pos = simulate(lp, b0, rng.random(n) < p_ent, zero, None, zero, zero,
                           hold_fn=lambda i: int(rng.choice(durs)))
        rt.append(score(lp, tr, pos)[0])
    rt = np.array(rt)

    # reliability: rolling windows + calendar years
    cs_st = np.r_[0.0, np.cumsum(np.log1p(r_st))]
    cs_bh = np.r_[0.0, np.cumsum(np.log1p(r_bh))]
    wins = [(cs_st[k + ROLL] - cs_st[k]) > (cs_bh[k + ROLL] - cs_bh[k])
            for k in range(0, m - ROLL + 1)]
    yrs = pd.Series(np.log1p(r_st) - np.log1p(r_bh), index=dates[b0 + 1:]).groupby(
        dates[b0 + 1:].year).sum()

    print(f"\n  E. Full strategy (entries + signal exits + stop) vs buy-and-hold")
    print(f"    return {np.expm1(tot_st):+.1%} vs B&H {np.expm1(tot_bh):+.1%}  ·  "
          f"MDD {np.expm1(st_mdd):+.1%} vs {np.expm1(bh_mdd):+.1%}  ·  "
          f"Sharpe {sh_st:.2f} vs {sh_bh:.2f}")
    print(f"    block bootstrap ({BLOCK}-bar blocks): log excess {obs_ex:+.2f}  "
          f"90% CI [{lo:+.2f}, {hi:+.2f}]  P(excess ≤ 0)={np.mean(ex_bs <= 0):.3f}  "
          f"P(Sharpe ≤ B&H)={np.mean(dsh_bs <= 0):.3f}")
    print(f"    random-timing null (same entry rate & holds): median {np.expm1(np.median(rt)):+.1%}  "
          f"p(null ≥ strategy)={np.mean(rt >= tot_st):.3f}")
    print(f"    rolling {ROLL}-bar windows beating B&H: {np.mean(wins):.0%} of {len(wins)}  ·  "
          "by year (log excess): " + "  ".join(f"{y} {v:+.2f}" for y, v in yrs.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--assets", nargs="+", default=["BTC", "MSTR", "MSTU"],
                    choices=["BTC", "MSTR", "MSTU", "ETH"])
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dates, sigs, px = load()
    for key in a.assets:
        if key in px:
            run_sleeve(key, dates, sigs, px, a.sims, a.seed)


if __name__ == "__main__":
    main()
