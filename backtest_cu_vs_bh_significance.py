#!/usr/bin/env python3
"""
Statistical significance: Confirmed Uptrend TF2 + V-Gate Reversal  vs  Buy & Hold
=================================================================================

The existing `backtest_stat_significance.py` tests strategy VARIANTS against each
other (Pure Regime vs Confirmed Uptrend vs Standard). It does NOT test any
variant against Buy & Hold. This script fills that gap.

"Confirmed Uptrend" is Option B in backtest_option_c.py:
    tf_option_b = u1 & ((bull_regime ^ clean_7d) | v_recent)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^
                       confirmed-uptrend entry      V-gate reversal

Comparing a partial-exposure timing strategy against Buy & Hold over a SINGLE
realised price path is fundamentally different from a two-sample trade test:
B&H has no "trades". We therefore work on the daily NAV return series and use:

  1. Newey-West (HAC) t-test on the daily alpha series d = r_strat - r_bh
     (H0: mean daily alpha <= 0). HAC corrects for autocorrelation/overlap.
  2. Circular block bootstrap (joint, block=20d) resampling the (r_strat, r_bh)
     PAIRS in blocks -> distribution of terminal-wealth gap and Sharpe gap, and
     the bootstrap estimate of P(strategy terminal wealth > B&H).

Runs fully OFFLINE on the locked CSVs in data/backtest/ (manifest pull 2026-06-15).

Usage:
    python backtest_cu_vs_bh_significance.py
"""

import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

try:
    import sklearn._loss._loss as _sklearn_loss_ext
    sys.modules.setdefault("_loss", _sklearn_loss_ext)
    import sklearn._loss.loss
    import sklearn._loss.link
except Exception:
    pass

import numpy as np
import pandas as pd
from scipy import stats

# backtest_option_c imports yfinance/requests at module load for its live data
# fetch path. We run fully offline from CSVs and never call those fetchers, so
# stub the network deps to allow the import.
import types as _types
for _m in ("yfinance", "requests"):
    if _m not in sys.modules:
        sys.modules[_m] = _types.ModuleType(_m)

from backtest_option_c import (
    build_ct_preds, _prep_comp, compute_signals, run_backtest,
    WARMUP, INITIAL_CAPITAL, ASSET_CONFIG,
)

DATA = _ROOT / "data" / "backtest"
PERIODS = {
    "Bull (Jun24->May25)": ("2024-06-01", "2025-05-31", "2024-03-01"),
    "Bear (Jun25->May26)": ("2025-06-01", "2026-05-31", "2025-03-01"),
    "Full (Jun24->May26)": ("2024-06-01", "2026-05-31", "2024-03-01"),
}
# Confirmed Uptrend uses no stop on BTC; MSTR -3%, MSTU -7% (live config).
ASSET_SL = {"BTC": 0.0, "MSTR": 0.03, "MSTU": 0.07}
N_BOOT = 10_000
BLOCK = 20
RNG = np.random.default_rng(42)
W = 100


def hr(ch="-"): print(ch * W)


# ── Offline data loading ────────────────────────────────────────────────────

def load_raw_features() -> pd.DataFrame:
    """raw_features_daily.csv -> df_raw in the shape build_ct_preds expects."""
    df = pd.read_csv(DATA / "raw_features_daily.csv")
    df["date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    df = df.drop(columns=["Date"]).set_index("date").sort_index()
    return df


def load_asset(name: str) -> pd.Series:
    """Daily close series for BTC/MSTR/MSTU from the locked CSVs."""
    fn = {"BTC": "btc_usd_daily.csv", "MSTR": "mstr_daily.csv",
          "MSTU": "mstu_daily.csv"}[name]
    d = pd.read_csv(DATA / fn)
    d["date"] = pd.to_datetime(d["Date"]).dt.tz_localize(None).dt.normalize()
    px = d.set_index("date")["close"].sort_index()
    all_days = pd.date_range(px.index[0], px.index[-1], freq="D")
    return px.reindex(all_days).ffill()


# ── Statistics ──────────────────────────────────────────────────────────────

def newey_west_tstat(d: np.ndarray, lags: int = 10) -> tuple[float, float]:
    """One-sample HAC (Newey-West) t-test that mean(d) > 0. Returns (t, p_one_sided)."""
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 5:
        return float("nan"), float("nan")
    dm = d - d.mean()
    gamma0 = np.dot(dm, dm) / n
    s = gamma0
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        cov = np.dot(dm[k:], dm[:-k]) / n
        s += 2.0 * w * cov
    se = np.sqrt(s / n)
    if se <= 0:
        return float("nan"), float("nan")
    t = d.mean() / se
    p_one = 1.0 - stats.t.cdf(t, df=n - 1)   # H1: mean > 0
    return float(t), float(p_one)


def circular_block_pairs(r_s: np.ndarray, r_b: np.ndarray,
                         n: int = N_BOOT, block: int = BLOCK):
    """
    Circular block bootstrap on the JOINT daily-return pairs.
    Returns arrays of: terminal-wealth gap (strat-bh, %), Sharpe gap (strat-bh).
    Also returns P(strat terminal > bh terminal).
    """
    T = len(r_s)
    gap_term = np.empty(n)
    gap_sharpe = np.empty(n)
    wins = 0
    n_blocks = int(np.ceil(T / block))
    for i in range(n):
        starts = RNG.integers(0, T, size=n_blocks)
        idx = ((starts[:, None] + np.arange(block)[None, :]) % T).ravel()[:T]
        ss, bb = r_s[idx], r_b[idx]
        term_s = np.prod(1 + ss) - 1
        term_b = np.prod(1 + bb) - 1
        gap_term[i] = (term_s - term_b) * 100
        sh_s = ss.mean() / ss.std() * np.sqrt(252) if ss.std() > 0 else 0.0
        sh_b = bb.mean() / bb.std() * np.sqrt(252) if bb.std() > 0 else 0.0
        gap_sharpe[i] = sh_s - sh_b
        if term_s > term_b:
            wins += 1
    return gap_term, gap_sharpe, wins / n


def ci95(a: np.ndarray) -> tuple[float, float]:
    a = a[np.isfinite(a)]
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


# ── Driver ──────────────────────────────────────────────────────────────────

def bh_nav(asset_px: np.ndarray, bt0: int, cap: float) -> pd.Series:
    out = np.full(len(asset_px), np.nan)
    base = asset_px[bt0]
    for i in range(bt0, len(asset_px)):
        out[i] = cap * asset_px[i] / base
    return out


def main():
    print("=" * W)
    print("  CONFIRMED UPTREND (TF2 + V-Gate) vs BUY & HOLD  -  statistical significance")
    print("  Offline run on locked CSVs (data/backtest, pull 2026-06-15)")
    print("  Tests: Newey-West HAC t-test on daily alpha | circular block bootstrap (joint pairs)")
    print("=" * W)

    df_raw = load_raw_features()
    print(f"\n  raw_features_daily: {len(df_raw)} rows, {df_raw.index[0].date()} -> {df_raw.index[-1].date()}")
    print("  Building CT predictions offline ...")
    preds = build_ct_preds(df_raw)
    print(f"  preds: {len(preds)} bars\n")

    asset_px_cache = {a: load_asset(a) for a in ["BTC", "MSTR", "MSTU"]}

    summary_rows = []

    for pname, (start, end, _fs) in PERIODS.items():
        comp = _prep_comp(df_raw, preds, start, end)
        if len(comp) < WARMUP + 3:
            print(f"  [SKIP] {pname}: insufficient bars"); continue
        dates = pd.DatetimeIndex(comp["target_date"])
        sigs = compute_signals(comp)
        sd = pd.Timestamp(start)
        bt0 = max(WARMUP, int(dates.searchsorted(sd)))

        for asset in ["BTC", "MSTR", "MSTU"]:
            px = asset_px_cache[asset].reindex(dates).ffill().bfill().values.astype(float)
            if np.sum(np.isfinite(px[bt0:]) & (px[bt0:] > 0)) < 10:
                print(f"  [SKIP] {asset} {pname}: insufficient prices"); continue
            sl = ASSET_SL[asset]

            # Confirmed Uptrend = Option B entry gate
            trades, nav = run_backtest(dates, px, sigs, sigs["tf_option_b"],
                                       sl, INITIAL_CAPITAL, sd)
            nav = nav.values.astype(float)
            bh = bh_nav(px, bt0, INITIAL_CAPITAL)

            s_nav = nav[bt0:]
            b_nav = bh[bt0:]
            r_s = np.diff(s_nav) / s_nav[:-1]
            r_b = np.diff(b_nav) / b_nav[:-1]
            mask = np.isfinite(r_s) & np.isfinite(r_b)
            r_s, r_b = r_s[mask], r_b[mask]

            strat_ret = (s_nav[-1] / INITIAL_CAPITAL - 1) * 100
            bh_ret = (b_nav[-1] / INITIAL_CAPITAL - 1) * 100
            alpha = strat_ret - bh_ret

            d = r_s - r_b
            t_nw, p_nw = newey_west_tstat(d, lags=10)
            gap_t, gap_sh, p_win = circular_block_pairs(r_s, r_b)
            lo_t, hi_t = ci95(gap_t)
            lo_s, hi_s = ci95(gap_sh)

            sh_s = r_s.mean() / r_s.std() * np.sqrt(252) if r_s.std() > 0 else 0.0
            sh_b = r_b.mean() / r_b.std() * np.sqrt(252) if r_b.std() > 0 else 0.0

            sig = lambda p: ("***" if p < 0.01 else "**" if p < 0.05
                             else "*" if p < 0.10 else "ns")

            hr("=")
            print(f"  {pname}  |  {asset}  |  {len(trades)} trades  |  days={len(r_s)}")
            hr()
            print(f"  Strategy return : {strat_ret:>+10.1f}%     Sharpe {sh_s:>+5.2f}")
            print(f"  Buy & Hold      : {bh_ret:>+10.1f}%     Sharpe {sh_b:>+5.2f}")
            print(f"  Alpha (realised): {alpha:>+10.1f} pp")
            print()
            print(f"  Newey-West HAC t-test (H0: daily alpha<=0): t={t_nw:>+6.2f}  p={p_nw:.4f} {sig(p_nw)}")
            print(f"  Block-bootstrap terminal-gap 95% CI: [{lo_t:>+8.1f}pp, {hi_t:>+8.1f}pp]  "
                  f"{'(excludes 0)' if lo_t > 0 or hi_t < 0 else '(straddles 0)'}")
            print(f"  Block-bootstrap Sharpe-gap   95% CI: [{lo_s:>+6.2f}, {hi_s:>+6.2f}]  "
                  f"{'(excludes 0)' if lo_s > 0 or hi_s < 0 else '(straddles 0)'}")
            print(f"  P(strategy beats B&H) [bootstrap, within-period]: {p_win*100:>5.1f}%")
            print()

            summary_rows.append(dict(period=pname, asset=asset, n_trades=len(trades),
                                     strat=strat_ret, bh=bh_ret, alpha=alpha,
                                     p_nw=p_nw, p_win=p_win, ci_lo=lo_t, ci_hi=hi_t))

    # ── Summary table ────────────────────────────────────────────────────────
    hr("=")
    print("  SUMMARY")
    hr("=")
    print(f"  {'Period':22} {'Asset':5} {'N':>3} {'Strat%':>9} {'B&H%':>9} {'Alpha':>9} "
          f"{'p(NW)':>7} {'P(win)':>7} {'TermGap 95% CI':>22}")
    for r in summary_rows:
        print(f"  {r['period']:22} {r['asset']:5} {r['n_trades']:>3d} "
              f"{r['strat']:>+8.1f} {r['bh']:>+8.1f} {r['alpha']:>+8.1f} "
              f"{r['p_nw']:>7.3f} {r['p_win']*100:>6.1f}% "
              f"[{r['ci_lo']:>+7.0f},{r['ci_hi']:>+7.0f}]")

    print("""
  READING THIS
  ------------
  * p(NW)  : Newey-West HAC one-sided p-value that mean daily alpha > 0.
             p<0.05 => realised daily out-performance unlikely under H0 of zero edge.
  * P(win) : bootstrap probability the strategy out-earns B&H, resampling blocks of
             the OBSERVED return pairs. This is a WITHIN-PERIOD figure: it quantifies
             path/sampling noise for this regime, NOT future regimes.
  * Term-gap CI excluding 0 => the out/under-performance is robust to block resampling.

  HARD LIMITATION (why "proof" is not possible here):
  ---------------------------------------------------
  There is ONE realised 2-year path = one bull regime + one bear regime, ~3-11 trades
  per cell. Bootstraps measure sampling noise WITHIN that path; they cannot manufacture
  independent regimes. The strategy structurally WINS in bear (low exposure) and LOSES
  in bull (partial exposure vs 100% B&H). So "likelihood of beating B&H" is regime-
  conditional, not a single universal number. Treat results as evidence, not proof.
""")
    hr("=")


if __name__ == "__main__":
    main()
