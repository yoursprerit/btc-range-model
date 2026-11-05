"""Intraday rate-of-drop exits — would selling mid-bar instead of waiting for the
close help the leveraged / high-drawdown sleeves?  (SOXL · NUGT · WGMI · MSTU)

The backtest engines are bar-close: a sleeve's exit (signal, fixed stop or trail)
is read off the daily close and filled at it, and the live 2:00 PM CT check only
looks at a *level* breach once a day.  This evaluates an additional **intraday
drop trigger** laid over the shipped daily engines, using hourly bars of the
traded ETF itself:

    trigger at hourly bar i  ⇔  price has fallen more than THR from the highest
    close of the last N bars (never reaching back before the entry close)

    THR = k · σ_hour · √N        (vol-normalised: one number per cluster)
    THR = p %                    (fixed %: the family the earlier stop memos swept)

``σ_hour`` is the trailing 20-session std of intra-session hourly returns, known
before the bar.  Two fill models bracket reality:

    close  (conservative, PRIMARY) detect on the bar CLOSE, fill at the NEXT bar's open
    touch  (instant-fill bound)    detect on the bar LOW touching the level, fill at
                                   the level (or the open if it gapped through).  Best
                                   possible fill price, but it also fires on intrabar
                                   spikes that recover by the bar's close, so it
                                   whipsaws MORE — it is not simply "better"

and two re-entry rules after a trigger exit:

    flat   stay flat until the daily engine's next fresh entry (what the engines
           do after any stop)
    rebuy  buy back at the next close if the daily engine is still long (what the
           morning book would do — i.e. the whipsaw-churn case)

Everything not touched by a trigger keeps the engine's own daily return, so every
delta is attributable to the overlay.  Costs: ``--slip-bps`` (default 10) on every
extra fill.

Hourly history is only ~2.9 years (yfinance 730-day cap, snapshotted in
``data/intraday_eval/`` by ``scripts/fetch_intraday_eval_cache.py``) — a handful of
real sell-offs — so Part 2 is a train/test split with a rank-stability check, not a
sweep to a single best number.

MSTU is NOT covered in this first cut: the CT engine runs a 7-day BTC-anchored
calendar on a synthetic pre-listing MSTU series (its daily returns do not line up
with MSTU's US-session bars — corr ≈ 0.04), so a clean MSTU test needs BTC 24/7
hourly bars (a follow-up).

    python scripts/eval_intraday_drop.py
    python scripts/eval_intraday_drop.py --slip-bps 25
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "app"))
sys.path.insert(0, str(_REPO))

import backtest_ticker as bt        # noqa: E402
import overall_core as oc           # noqa: E402

SLEEVES = ["SOXL", "NUGT", "WGMI"]            # MSTU: see module docstring
NS = [3, 7, 21]                               # lookback in hourly bars (~½ day, 1 day, 3 days)
VOL_K = [2.0, 3.0, 4.0, 5.0]
PCT_P = [0.03, 0.05, 0.08, 0.12, 0.15]
PLACEBO_DRAWS = 200
WARMUP_SESSIONS = 10                          # window starts after σ has history


# ── data ─────────────────────────────────────────────────────────────────
class Hourly:
    """Hourly OHLC of one ETF + session bookkeeping."""

    def __init__(self, ticker: str):
        h = pd.read_csv(_REPO / "data" / "intraday_eval" / f"hourly_{ticker}.csv",
                        index_col=0, parse_dates=True).sort_index()
        h = h[~h.index.duplicated(keep="last")]
        self.o, self.hi, self.lo, self.c = (h[k].to_numpy(float) for k in
                                            ("open", "high", "low", "close"))
        self.day = h.index.normalize()
        self.sess = {}                          # date -> (first_idx, last_idx)
        for d, g in pd.Series(np.arange(len(h)), index=self.day).groupby(level=0):
            self.sess[d] = (int(g.iloc[0]), int(g.iloc[-1]))
        self.dates = sorted(self.sess)
        r = np.full(len(h), np.nan)
        same = self.day[1:] == self.day[:-1]
        r[1:] = np.where(same, self.c[1:] / self.c[:-1] - 1.0, np.nan)
        # σ known BEFORE bar i: 20 sessions ≈ 140 bars of intra-session returns
        self.sigma = pd.Series(r).rolling(140, min_periods=40).std().shift(1).to_numpy()

    def prev_close(self, d):
        i0, _ = self.sess[d]
        return self.c[i0 - 1] if i0 > 0 else np.nan


# ── the trigger ──────────────────────────────────────────────────────────
def scan_day(H: Hourly, d, floor, N, fam, par, fill):
    """First trigger on session ``d`` → (fill_px, bar_idx, drop) or None."""
    i0, i1 = H.sess[d]
    for i in range(i0, i1 + 1):
        s = H.sigma[i]
        if fam == "vol":
            if not np.isfinite(s):
                continue
            thr = min(par * s * np.sqrt(N), 0.95)
        else:
            thr = par
        ref = H.c[max(i - N, floor):i].max() if i > floor else np.nan
        if not np.isfinite(ref):
            continue
        level = ref * (1.0 - thr)
        if fill == "touch":
            if H.lo[i] <= level:
                px = H.o[i] if H.o[i] <= level else level
                return px, i, 1.0 - px / ref
        else:                                    # close-detect, next-open fill
            if H.c[i] <= level:
                px = H.o[i + 1] if i < i1 else H.c[i]
                return px, i, 1.0 - H.c[i] / ref
    return None


def overlay(ret: pd.Series, pos: pd.Series, H: Hourly, N, fam, par,
            fill="close", mode="flat", slip=0.001, force=None):
    """Lay the drop trigger over a sleeve's daily (ret, pos).  Returns
    (ret', pos', events, effective-hold flags)."""
    idx = ret.index
    r = ret.to_numpy(float).copy()
    p_end = pos.reindex(idx).fillna(0).to_numpy(float)
    p_new = p_end.copy()
    p_prev = np.r_[0.0, p_end[:-1]]
    hold_eff = (p_prev == 1.0)
    events = []
    blocked = False
    blocked_day = -1
    floor = 0
    # entry-close bar of the position in force on each day (floor for the lookback)
    entry_bar = np.zeros(len(idx), int)
    last_entry = 0
    for t in range(len(idx)):
        if t > 0 and p_end[t] == 1.0 and p_end[t - 1] == 0.0:
            d_e = idx[t]
            last_entry = H.sess[d_e][1] if d_e in H.sess else 0
        entry_bar[t] = last_entry
    for t in range(1, len(idx)):
        d = idx[t]
        if p_prev[t] != 1.0:                     # baseline flat into today
            blocked = False
            continue
        if d not in H.sess:
            continue
        if blocked:
            hold_eff[t] = False
            r[t] = 0.0
            if mode == "rebuy" and t == blocked_day + 1:
                p_new[t] = p_end[t]              # re-buy at today's close if still long
                if p_end[t] == 1.0:
                    r[t] = -slip                 # cost of the extra fill
                blocked = False
                floor_override = H.sess[d][1]
                entry_bar[t + 1:] = np.where(entry_bar[t + 1:] <= floor_override,
                                             floor_override, entry_bar[t + 1:])
            else:
                p_new[t] = 0.0
                if p_end[t] == 0.0:
                    blocked = False
            continue
        floor = entry_bar[t - 1] if t > 0 else 0
        if force is not None:                    # placebo: trigger at a given bar
            i_f = force.get(d)
            i0_, i1_ = H.sess[d]
            hit = None if i_f is None else (
                (H.o[i_f + 1] if i_f < i1_ else H.c[i_f]), i_f, 0.0)
        else:
            hit = scan_day(H, d, floor, N, fam, par, fill)
        if hit is None:
            continue
        px, bar, drop = hit
        pp = H.prev_close(d)
        if not np.isfinite(pp):
            continue
        r[t] = px * (1.0 - slip) / pp - 1.0
        p_new[t] = 0.0
        blocked, blocked_day = True, t
        if p_end[t] == 0.0:                      # engine exits at this close anyway
            blocked = False
        # counterfactual for the event study: sold here vs held to the engine's exit
        j = t
        while j < len(idx) and p_end[j] == 1.0:
            j += 1
        j = min(j, len(idx) - 1)
        dj = idx[j]
        exit_px = H.c[H.sess[dj][1]] if dj in H.sess else np.nan
        events.append(dict(date=d, bar=H.sess[d][0] != bar and f"+{bar - H.sess[d][0]}h" or "open",
                           drop=drop, fill=px, vs_exit=exit_px / px - 1.0 if np.isfinite(exit_px) else np.nan,
                           day_ret=r[t], base_day_ret=float(ret.iloc[t])))
    return (pd.Series(r, idx), pd.Series(p_new, idx), events, hold_eff)


# ── metrics ──────────────────────────────────────────────────────────────
def trades_from(r: np.ndarray, hold: np.ndarray) -> np.ndarray:
    out, cur = [], None
    for x, h in zip(r, hold):
        if h:
            cur = (1 + x) * (cur if cur is not None else 1.0)
        elif cur is not None:
            out.append(cur - 1); cur = None
    if cur is not None:
        out.append(cur - 1)
    return np.array(out)


def stats(r: pd.Series, hold: np.ndarray, lo, hi=None) -> dict:
    sl = r.loc[lo:hi]
    hd = pd.Series(hold, r.index).loc[lo:hi].to_numpy()
    eq = np.cumprod(1 + sl.to_numpy())
    m = bt._metrics(eq, sl.index.to_numpy())
    tr = trades_from(sl.to_numpy(), hd)
    m["win"] = float((tr > 0).mean() * 100) if len(tr) else 0.0
    m["trades"] = len(tr)
    m["calmar"] = m["cagr"] / abs(m["mdd"]) if m["mdd"] < 0 else np.nan
    return m


# ── driver ───────────────────────────────────────────────────────────────
def main(slip_bps: float):
    slip = slip_bps / 1e4
    print("Running the live universe (shipped daily engines)…", flush=True)
    base = oc.run_universe()
    by = {s["key"]: s for s in base}
    H = {k: Hourly(k) for k in SLEEVES}
    configs = ([("vol", k, N) for k in VOL_K for N in NS] +
               [("pct", p, N) for p in PCT_P for N in NS])

    def label(c):
        fam, par, N = c
        return (f"{par:g}σ·{N}h" if fam == "vol" else f"{par*100:g}%·{N}h")

    win, split, bh = {}, {}, {}
    for k in SLEEVES:
        ds = [d for d in H[k].dates if d in by[k]["ret"].index]
        lo = ds[WARMUP_SESSIONS]
        win[k] = (lo, ds[-1])
        split[k] = ds[WARMUP_SESSIONS + (len(ds) - WARMUP_SESSIONS) // 2]
        bpos = by[k]["pos_series"].reindex(by[k]["ret"].index).fillna(0).to_numpy()
        bh[k] = np.r_[0.0, bpos[:-1]] == 1.0
        n_tr = len(trades_from(by[k]["ret"].loc[lo:].to_numpy(),
                               pd.Series(bh[k], by[k]["ret"].index).loc[lo:].to_numpy()))
        print(f"  {k}: window {lo.date()} → {ds[-1].date()} ({len(ds) - WARMUP_SESSIONS} "
              f"sessions, {n_tr} baseline trades), train/test split {split[k].date()}")

    def run(k, c, fill="close", mode="flat", sl=slip, force=None):
        s = by[k]
        fam, par, N = c
        return overlay(s["ret"], s["pos_series"], H[k], N, fam, par, fill, mode, sl, force)

    def base_stats(k, a, b):
        return stats(by[k]["ret"], bh[k], a, b)

    cache = {}
    for mode in ("flat", "rebuy"):
        print("\n" + "#" * 100)
        print(f"#  RE-ENTRY = {mode.upper()}   ({'stay flat until the engine\'s next fresh entry' if mode == 'flat' else 'buy back at the next close if the engine is still long'})"
              f" · fill = close-detect / next-open · slip {slip_bps:g} bp")
        print("#" * 100)

        # ── Part 1 ───────────────────────────────────────────────────────
        print("\nPART 1 — per-sleeve grid (full window)")
        for k in SLEEVES:
            lo, hi = win[k]
            b = base_stats(k, lo, hi)
            print(f"\n{k} — {lo.date()} → {hi.date()}   (mode {by[k].get('mode')}, stop {by[k].get('stop')})")
            print(f"  {'config':>9s} | {'Return':>9s} {'MDD':>8s} {'Sharpe':>7s} {'Win%':>5s} "
                  f"{'Trd':>4s} | {'Trig':>4s} {'Save':>4s} {'Whip':>4s} {'avg save/whip':>14s}")
            print("  " + "-" * 82)
            print(f"  {'baseline':>9s} | {b['total_ret']*100:+8.1f}% {b['mdd']*100:+7.1f}% "
                  f"{b['sharpe']:7.2f} {b['win']:4.0f}% {b['trades']:4d} |")
            for c in configs:
                r2, p2, ev, he = run(k, c, "close", mode)
                cache[(mode, k, c)] = (r2, p2, ev, he)
                m = stats(r2, he, lo, hi)
                evw = [e for e in ev if lo <= e["date"] <= hi]
                sv = [e["vs_exit"] for e in evw if e["vs_exit"] < 0]
                wh = [e["vs_exit"] for e in evw if e["vs_exit"] >= 0]
                print(f"  {label(c):>9s} | {m['total_ret']*100:+8.1f}% {m['mdd']*100:+7.1f}% "
                      f"{m['sharpe']:7.2f} {m['win']:4.0f}% {m['trades']:4d} | "
                      f"{len(evw):4d} {len(sv):4d} {len(wh):4d} "
                      f"{(np.mean(sv)*100 if sv else 0):+6.1f}%/{(np.mean(wh)*100 if wh else 0):+5.1f}%")

        # ── Part 2: walk-forward selection ───────────────────────────────
        print("\nPART 2 — train/test: pick ONE config on the first half (pooled over sleeves), judge on the second")
        rows = []
        for c in configs:
            d_tr, d_te = [], []
            for k in SLEEVES:
                lo, hi = win[k]
                sp = split[k]
                r2, _, _, he = cache[(mode, k, c)]
                for (a, b_), acc in (((lo, sp - pd.Timedelta(days=1)), d_tr), ((sp, hi), d_te)):
                    mb, mo = base_stats(k, a, b_), stats(r2, he, a, b_)
                    acc.append((mo["calmar"] - mb["calmar"], mo["sharpe"] - mb["sharpe"],
                                (mo["mdd"] - mb["mdd"]) * 100, (mo["total_ret"] - mb["total_ret"]) * 100))
            rows.append(dict(cfg=c, tr=np.nanmean([x[0] for x in d_tr]), tr_sh=np.mean([x[1] for x in d_tr]),
                             te=np.nanmean([x[0] for x in d_te]), te_sh=np.mean([x[1] for x in d_te]),
                             te_mdd=np.mean([x[2] for x in d_te]), te_ret=np.mean([x[3] for x in d_te])))
        df = pd.DataFrame(rows)
        rho = df["tr"].rank().corr(df["te"].rank())
        print(f"Rank correlation of ΔCalmar, train vs test, across {len(df)} configs: {rho:+.2f} "
              f"(≈0 ⇒ the train ranking says nothing about the future)")
        ok = df[df["tr_sh"] >= 0].sort_values("tr", ascending=False)
        print(f"\n{'train-ranked top 5':>22s} | train ΔCalmar ΔSharpe | TEST ΔCalmar ΔSharpe ΔMDD(pp) ΔRet(pp)")
        for _, r in ok.head(5).iterrows():
            print(f"{label(r['cfg']):>22s} | {r['tr']:+13.2f} {r['tr_sh']:+7.2f} | "
                  f"{r['te']:+12.2f} {r['te_sh']:+7.2f} {r['te_mdd']:+8.1f} {r['te_ret']:+8.1f}")
        print(f"\nAll {len(df)} configs, TEST half: improved Calmar {(df['te'] > 0).mean()*100:.0f}% · "
              f"improved Sharpe {(df['te_sh'] > 0).mean()*100:.0f}% · reduced |MDD| {(df['te_mdd'] > 0).mean()*100:.0f}% · "
              f"median ΔMDD {df['te_mdd'].median():+.1f}pp · median ΔRet {df['te_ret'].median():+.1f}pp")
        best = ok.iloc[0]["cfg"] if len(ok) else configs[0]

        # ── Part 3: fill / cost sensitivity ──────────────────────────────
        print(f"\nPART 3 — sensitivity of the train-selected config {label(best)} (full window)")
        print(f"{'sleeve':>6s} {'variant':>22s} | {'Return':>9s} {'MDD':>8s} {'Sharpe':>7s} {'Trig':>4s}")
        for k in SLEEVES:
            lo, hi = win[k]
            b = base_stats(k, lo, hi)
            print(f"{k:>6s} {'baseline':>22s} | {b['total_ret']*100:+8.1f}% {b['mdd']*100:+7.1f}% {b['sharpe']:7.2f}")
            for fill, sb in (("close", slip_bps), ("touch", slip_bps), ("close", 0.0),
                             ("close", 25.0), ("close", 50.0), ("close", 100.0)):
                r2, _, ev, he = run(k, best, fill, mode, sb / 1e4)
                m = stats(r2, he, lo, hi)
                evw = [e for e in ev if lo <= e["date"] <= hi]
                print(f"{'':>6s} {f'{fill}/{sb:g}bp':>22s} | {m['total_ret']*100:+8.1f}% "
                      f"{m['mdd']*100:+7.1f}% {m['sharpe']:7.2f} {len(evw):4d}")

        # ── Part 4: event study + concentration ──────────────────────────
        print(f"\nPART 4 — every trigger of {label(best)}: Δlog-return of that episode (2 days: trigger day + next)")
        for k in SLEEVES:
            r2, _, ev, _ = cache[(mode, k, best)]
            lo, hi = win[k]
            ev = [e for e in ev if lo <= e["date"] <= hi]
            ret = by[k]["ret"]
            dl = []
            for e in ev:
                i = ret.index.get_loc(e["date"])
                dl.append(float(np.log1p(r2.iloc[i:i + 2]).sum() - np.log1p(ret.iloc[i:i + 2]).sum())
                          if mode == "rebuy" else
                          float(np.log1p(r2.iloc[i]) - np.log1p(ret.iloc[i]) - np.log1p(e["vs_exit"])))
            dl = np.array(dl)
            if len(dl) == 0:
                print(f"\n{k}: 0 triggers"); continue
            top3 = np.sort(dl)[::-1][:3].sum()
            print(f"\n{k}: {len(ev)} triggers · {int((dl > 0).sum())} helped / {int((dl < 0).sum())} hurt · "
                  f"net Δlog {dl.sum():+.2f} · best 3 episodes {top3:+.2f}")
            if len(ev) <= 12:
                for e, x in zip(ev, dl):
                    print(f"    {e['date'].date()} {e['bar']:>5s} drop {e['drop']*100:5.1f}%  day {e['day_ret']*100:+6.1f}% "
                          f"(engine {e['base_day_ret']*100:+6.1f}%)  episode Δlog {x:+.3f}  "
                          f"[sold-here vs engine-exit {e['vs_exit']*100:+.1f}%]")

        # ── Part 5: placebo ──────────────────────────────────────────────
        print(f"\nPART 5 — placebo: same NUMBER of triggers, random held days/bars ({PLACEBO_DRAWS} draws) vs the real {label(best)}")
        rng = np.random.default_rng(7)
        for k in SLEEVES:
            lo, hi = win[k]
            r2, _, ev, he = cache[(mode, k, best)]
            evw = [e for e in ev if lo <= e["date"] <= hi]
            n = len(evw)
            if n == 0:
                print(f"  {k}: 0 triggers — nothing to test"); continue
            held = [d for d, h in zip(by[k]["ret"].index, bh[k]) if h and lo <= d <= hi and d in H[k].sess]
            real = np.log1p(r2.loc[lo:hi]).sum() - np.log1p(by[k]["ret"].loc[lo:hi]).sum()
            rmdd = stats(r2, he, lo, hi)["mdd"]
            sims, mdds = [], []
            for _ in range(PLACEBO_DRAWS):
                days = rng.choice(len(held), size=min(n, len(held)), replace=False)
                f = {held[j]: int(rng.integers(H[k].sess[held[j]][0], H[k].sess[held[j]][1] + 1)) for j in days}
                rp, _, _, hp = run(k, best, "close", mode, slip, force=f)
                sims.append(np.log1p(rp.loc[lo:hi]).sum() - np.log1p(by[k]["ret"].loc[lo:hi]).sum())
                mdds.append(stats(rp, hp, lo, hi)["mdd"])
            sims, mdds = np.array(sims), np.array(mdds)
            print(f"  {k}: real Δlog {real:+.2f} → beats {(sims < real).mean()*100:.0f}% of random placements "
                  f"(placebo median {np.median(sims):+.2f}, 5–95% {np.percentile(sims, 5):+.2f}…{np.percentile(sims, 95):+.2f}) · "
                  f"MDD real {rmdd*100:+.1f}% vs placebo median {np.median(mdds)*100:+.1f}% "
                  f"(real is shallower than {(mdds < rmdd).mean()*100:.0f}% of placebos)")

        # ── Part 6: Overall equal-weight impact ──────────────────────────
        print(f"\nPART 6 — Overall equal-weight blend with {label(best)} on SOXL+NUGT+WGMI together")

        def ew(results):
            rets = oc.returns_matrix(results)
            pos = oc.position_matrix(results, rets.index)
            w = np.full(rets.shape[1], 1.0 / rets.shape[1])
            return oc._equity(oc._combine(rets, w, pos, oc.SATA_DAILY))

        swapped = []
        for s in base:
            if s["key"] in SLEEVES:
                r2, p2, _, _ = cache[(mode, s["key"], best)]
                s = {**s, "ret": r2, "pos_series": p2}
            swapped.append(s)
        e0, e1 = ew(base), ew(swapped)
        for tag, lo_ in (("common window", max(win[k][0] for k in SLEEVES)),
                         ("test half   ", max(split[k] for k in SLEEVES))):
            m0, m1 = oc.slice_metrics(e0, lo_), oc.slice_metrics(e1, lo_)
            print(f"  {tag} from {pd.Timestamp(lo_).date()}: baseline ret {m0['total_ret']*100:+7.1f}% "
                  f"MDD {m0['mdd']*100:+6.1f}% Sharpe {m0['sharpe']:.3f}  →  overlay ret "
                  f"{m1['total_ret']*100:+7.1f}% MDD {m1['mdd']*100:+6.1f}% Sharpe {m1['sharpe']:.3f}")
    print("\nDone.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slip-bps", type=float, default=10.0)
    main(ap.parse_args().slip_bps)
