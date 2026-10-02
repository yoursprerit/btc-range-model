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
    print()


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
