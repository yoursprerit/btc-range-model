"""Historical analogs for the 2026-08 BTC thrust — is it a real bull leg or a fake-out?

Asks two questions of the *live* BTC signal engine (``app/btc_ct_engine.py``,
which byte-identically reproduces ``app/btc_hourly_app.py``'s H/L predictions
and Pure-Regime / Standard-MA gates):

  1. **Has this pattern happened before?**  Enumerate every *fresh U1 onset*
     (the regime-divergence entry trigger: ``err_hi_ma3 > +1.3%`` AND
     ``hi_breaks_3d >= 2`` on a bar where U1 was off the bar before) over the
     whole CT window, tag each with its gate configuration
     (``above_ma30`` / ``clean_7d`` / ``bull_regime`` -> Pure-Regime vs
     Standard-MA verdict) and its forward BTC return at +5/+10/+21/+42 bars.

  2. **Real or fake-out?**  Score the current event against three cohorts of
     past analogs (same gate configuration · big divergence shock · parabolic
     prior thrust) and bootstrap each cohort's median forward return against
     the unconditional distribution of BTC bars in the same window.  Also runs
     the "fade test": split events by whether the divergence *persisted* in the
     three bars after onset (mean ``err_hi`` > 0) or immediately faded.

Both gates are then backtested end-to-end so the standalone question — which
gate is actually live, and what it says today — is answered from the engine
rather than from a doc.

    python scripts/eval_bullrun_analogs.py

Everything is computed from the committed data vintage
(``data/backtest/raw_features_daily.csv``); numbers drift with each refresh.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "app"))

import btc_ct_engine as E                                   # noqa: E402
import backtest_trailing_stop as T                          # noqa: E402

CT_START = "2023-11-01"
BT_START = pd.Timestamp("2024-06-01")
HORIZONS = (5, 10, 21, 42)
SHOCK_MIN = 5.5          # err_hi (%) that qualifies a bar as a divergence shock
THRUST_MIN = 10.0        # prior 5-bar BTC return (%) that qualifies as parabolic
BOOT_N = 20_000
SEED = 7


# ── signal frame ────────────────────────────────────────────────────────────
def build_signals() -> pd.DataFrame:
    rf = T.load_raw_features()
    preds = T.build_preds_offline(rf)
    comp = T.prep(rf, preds, CT_START, str(rf.index[-1].date()))
    sigs = E.compute_sigs_pure(comp)
    cols = ("err_hi", "err_lo", "ehma3", "elma3", "hb3", "lb3", "u1", "d1",
            "d2", "d3", "above_ma30", "clean_7d", "bull_regime", "v_recent",
            "tf2_entry_pure", "tf2_entry_ma")
    df = pd.DataFrame({k: sigs[k] for k in cols},
                      index=pd.DatetimeIndex(comp["target_date"]))
    df["close"] = comp["actual_close"].to_numpy(float)
    return df


# ── event table ─────────────────────────────────────────────────────────────
def onsets(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"].to_numpy(float)
    n = len(c)
    u1 = df["u1"].to_numpy(bool)
    err = df["err_hi"].to_numpy(float)
    d2, d3 = df["d2"].to_numpy(bool), df["d3"].to_numpy(bool)
    bull = df["bull_regime"].to_numpy(bool)

    def fwd(i, k):
        return (c[i + k] / c[i] - 1) * 100 if i + k < n else np.nan

    def dd(i, k):
        seg = c[i:min(n, i + k + 1)]
        return float(np.min(seg / np.maximum.accumulate(seg) - 1) * 100) if len(seg) > 1 else np.nan

    def bars_to_exit(i):
        for j in range(i + 1, n):
            if d3[j] or (d2[j] and not bull[j]):
                return j - i
        return np.nan

    rows = []
    for i in range(1, n):
        if not (u1[i] and not u1[i - 1]):
            continue
        post = err[i + 1:i + 4]
        rows.append(dict(
            date=df.index[i], close=c[i],
            ehma3=float(df["ehma3"].iloc[i]),
            errhi_max3=float(np.max(err[max(0, i - 2):i + 1])),
            ret5_prior=(c[i] / c[max(0, i - 5)] - 1) * 100,
            above=bool(df["above_ma30"].iloc[i]), clean=bool(df["clean_7d"].iloc[i]),
            bull=bool(bull[i]),
            pure=bool(df["tf2_entry_pure"].iloc[i]), ma=bool(df["tf2_entry_ma"].iloc[i]),
            post_errhi3=float(post.mean()) if len(post) else np.nan,
            dd21=dd(i, 21), exit_in=bars_to_exit(i),
            **{f"f{k}": fwd(i, k) for k in HORIZONS}))
    return pd.DataFrame(rows)


# ── stats ───────────────────────────────────────────────────────────────────
def unconditional(df: pd.DataFrame, k: int) -> np.ndarray:
    c = df["close"].to_numpy(float)
    return np.array([(c[i + k] / c[i] - 1) * 100 for i in range(len(c) - k)])


def boot_p(df: pd.DataFrame, sample: np.ndarray, k: int, rng) -> float:
    """One-sided P(median of a random same-size draw >= the cohort's median)."""
    base = unconditional(df, k)
    obs = np.median(sample)
    draws = np.median(rng.choice(base, size=(BOOT_N, len(sample))), axis=1)
    return float((draws >= obs).mean())


def describe(name: str, co: pd.DataFrame, df: pd.DataFrame, rng) -> None:
    hist = co.dropna(subset=[f"f{HORIZONS[-1]}"])
    if hist.empty:
        print(f"  {name}: no completed history"); return
    parts = []
    for k in HORIZONS:
        s = hist[f"f{k}"].dropna().to_numpy()
        parts.append("+%2dd med %+6.2f%% hit %3.0f%% p=%.3f"
                     % (k, np.median(s), 100 * (s > 0).mean(), boot_p(df, s, k, rng)))
    print(f"  {name} (n={len(hist)})")
    print("     " + " | ".join(parts))


# ── gate backtests ──────────────────────────────────────────────────────────
def gate_runs(df: pd.DataFrame) -> None:
    rf = T.load_raw_features()
    preds = T.build_preds_offline(rf)
    comp = T.prep(rf, preds, CT_START, str(rf.index[-1].date()))
    sigs = E.compute_sigs_pure(comp)
    dates = pd.DatetimeIndex(comp["target_date"])
    px = comp["actual_close"].to_numpy(float)
    for label, key in (("Standard MA  [LIVE BTC gate]", "tf2_entry_ma"),
                       ("Pure Regime  [divergence]", "tf2_entry_pure")):
        s = dict(sigs); s["tf2_entry"] = sigs[key]
        bt = E._run_bt(dates, px, s, E.STOP_PCT["BTC"], BT_START)
        m, bh = E._curve_metrics(bt["nav"]), E._curve_metrics(bt["bh"])
        dec = E._decision(s, len(dates) - 1, bt["open_pos"])
        print(f"\n  {label}")
        print("    %d trades | total %+.1f%% | Sharpe %.2f | MDD %.1f%% | B&H %+.1f%%"
              % (len(bt["trades"]), m["total_ret"] * 100, m["sharpe"],
                 m["mdd"] * 100, bh["total_ret"] * 100))
        print("    today: %s" % dec["label"])
        if bt["open_entry"]:
            oe = bt["open_entry"]
            print("    open:  %s @ %.2f (%s) -> %+.2f%% unrealised"
                  % (oe["date"].date(), oe["price"], oe["trigger"],
                     (px[-1] / oe["price"] - 1) * 100))


def main() -> None:
    rng = np.random.default_rng(SEED)
    df = build_signals()
    ev = onsets(df)
    cur = ev.iloc[-1]
    pd.set_option("display.width", 300)

    print("=" * 96)
    print("BTC bull-run analogs — CT window %s .. %s (%d bars)"
          % (df.index[0].date(), df.index[-1].date(), len(df)))
    print("=" * 96)

    print("\n[1] Every fresh U1 onset (regime-divergence trigger)\n")
    show = ev.copy(); show["date"] = show["date"].dt.date
    print(show.round(2).to_string(index=False))

    print("\n[2] Where the current event sits in the distribution")
    for col, lbl in (("errhi_max3", "err_hi shock (%)"), ("ehma3", "err_hi_ma3 (%)"),
                     ("ret5_prior", "prior 5-bar BTC return (%)")):
        rank = int((ev[col] > cur[col]).sum()) + 1
        print("    %-28s %6.2f  -> rank %d of %d onsets (previous max %.2f)"
              % (lbl, cur[col], rank, len(ev), ev[col].iloc[:-1].max()))
    print("    gate config: above_ma30=%s clean_7d=%s bull_regime=%s -> Pure=%s / StdMA=%s"
          % (cur.above, cur.clean, cur.bull, cur.pure, cur.ma))

    print("\n[3] Cohort forward returns vs unconditional BTC (bootstrap, one-sided)\n")
    base = "  unconditional: " + " | ".join(
        "+%2dd med %+6.2f%% hit %3.0f%%"
        % (k, np.median(unconditional(df, k)), 100 * (unconditional(df, k) > 0).mean())
        for k in HORIZONS)
    print(base + "\n")
    hist = ev.iloc[:-1]
    describe("A · same gate config (above & clean & bull -> Pure ENTER / StdMA BLOCKED)",
             hist[hist.above & hist.clean & hist.bull], df, rng)
    describe("B · big divergence shock (err_hi >= %.1f%%)" % SHOCK_MIN,
             hist[hist.errhi_max3 >= SHOCK_MIN], df, rng)
    describe("C · parabolic prior thrust (5-bar return >= %.0f%%)" % THRUST_MIN,
             hist[hist.ret5_prior >= THRUST_MIN], df, rng)

    print("\n[4] Fade test — did the divergence persist after onset?\n")
    h = hist.dropna(subset=["post_errhi3", f"f{HORIZONS[-1]}"])
    for lbl, sub in (("persisted (mean err_hi > 0 over next 3 bars)", h[h.post_errhi3 > 0]),
                     ("faded     (mean err_hi <= 0)", h[h.post_errhi3 <= 0])):
        print("  %-46s n=%2d  " % (lbl, len(sub))
              + " | ".join("+%2dd med %+6.2f%% hit %3.0f%%"
                           % (k, sub[f"f{k}"].median(), 100 * (sub[f"f{k}"] > 0).mean())
                           for k in HORIZONS))
    print("\n  current event post-onset mean err_hi = %+.2f%%  -> %s bucket"
          % (cur.post_errhi3, "persisted" if cur.post_errhi3 > 0 else "FADED"))

    print("\n[5] What each gate actually did / says today")
    gate_runs(df)
    print()


if __name__ == "__main__":
    main()
