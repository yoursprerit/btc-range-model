#!/usr/bin/env python3
"""
Statistical significance testing for entry gate variants.
Tests whether Pure Regime (PR) outperforms Confirmed Uptrend (CU) and Standard
in a statistically meaningful way using:

  1. Per-trade return tests: Welch t-test and Mann-Whitney U
  2. Bootstrap permutation on total compounded return
  3. Block bootstrap on daily NAV returns (Sharpe ratio CI)

Focuses on MSTR (no inception/bfill complications) across all periods.

Usage:
    python backtest_stat_significance.py
"""

import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

try:
    import sklearn._loss._loss as _sklearn_loss_ext
    if "_loss" not in sys.modules:
        sys.modules["_loss"] = _sklearn_loss_ext
    import sklearn._loss.loss
    import sklearn._loss.link
except Exception:
    pass

import joblib
import numpy as np
import pandas as pd
from scipy import stats

# ── Reuse data-fetching and signal code from backtest_option_c ───────────────
from backtest_option_c import (
    fetch_data, fetch_asset_prices, build_ct_preds, _prep_comp,
    compute_signals, run_backtest, WARMUP, INITIAL_CAPITAL,
)

PERIODS = {
    "Bull  (Jun24→May25)": ("2024-06-01", "2025-05-31", "2024-03-01"),
    "Bear  (Jun25→May26)": ("2025-06-01", "2026-05-31", "2025-03-01"),
    "Full  (Jun24→May26)": ("2024-06-01", "2026-05-31", "2024-03-01"),
}
MSTR_SL = 0.03
N_BOOTSTRAP = 10_000
BLOCK_SIZE  = 20   # ~1 month blocks for block bootstrap
RNG = np.random.default_rng(42)

W = 100

def hr(ch="─"): print(ch * W)


# ── Statistical tests ─────────────────────────────────────────────────────────

def welch_ttest(a: list, b: list) -> tuple[float, float]:
    """Two-sided Welch t-test. Returns (t-stat, p-value)."""
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    return stats.ttest_ind(a, b, equal_var=False)


def mann_whitney(a: list, b: list) -> tuple[float, float]:
    """Two-sided Mann-Whitney U. Returns (U-stat, p-value)."""
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    return stats.mannwhitneyu(a, b, alternative="two-sided")


def bootstrap_total_return(trade_pnls: list, n: int = N_BOOTSTRAP) -> np.ndarray:
    """
    Bootstrap distribution of compounded total return.
    Resamples trades (with replacement) and compounds them.
    Returns array of simulated total-return values.
    """
    arr = np.array(trade_pnls) / 100 + 1  # convert % to multipliers
    results = np.empty(n)
    for i in range(n):
        sample = RNG.choice(arr, size=len(arr), replace=True)
        results[i] = (np.prod(sample) - 1) * 100
    return results


def block_bootstrap_sharpe(daily_rets: pd.Series, n: int = N_BOOTSTRAP,
                             block: int = BLOCK_SIZE) -> np.ndarray:
    """
    Circular block bootstrap on daily returns → distribution of annualised Sharpe.
    """
    r = daily_rets.dropna().values
    T = len(r)
    if T < block * 2:
        return np.array([float("nan")])
    sharpes = np.empty(n)
    for i in range(n):
        indices = []
        while len(indices) < T:
            start = RNG.integers(0, T)
            blk = np.arange(start, start + block) % T
            indices.extend(blk.tolist())
        sample = r[np.array(indices[:T])]
        mu, sd = sample.mean(), sample.std()
        sharpes[i] = (mu / sd * np.sqrt(252)) if sd > 0 else 0.0
    return sharpes


def p_value_greater(dist_a: np.ndarray, dist_b: np.ndarray) -> float:
    """
    Bootstrap p-value: P(A > B) assuming H0: A <= B.
    Computed as fraction of bootstrap samples where A <= B (one-sided).
    Returns two-sided p: 2 * min(P(A>B), P(A<B))
    """
    p_greater = float(np.mean(dist_a > dist_b))
    return 2 * min(p_greater, 1 - p_greater)


def ci95(arr: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


# ── Per-trade analysis ────────────────────────────────────────────────────────

def analyse_trades(std_t, cu_t, pr_t, label: str):
    print(f"\n{'─'*W}")
    print(f"  PER-TRADE ANALYSIS  [{label}]")
    print(f"{'─'*W}")
    std_pnls = [t["pnl_pct"] for t in std_t]
    cu_pnls  = [t["pnl_pct"] for t in cu_t]
    pr_pnls  = [t["pnl_pct"] for t in pr_t]

    print(f"  {'':30s} {'Standard':>12} {'CU':>12} {'PR':>12}")
    print(f"  {'N trades':30s} {len(std_pnls):>12d} {len(cu_pnls):>12d} {len(pr_pnls):>12d}")
    if std_pnls:
        print(f"  {'Mean trade %':30s} {np.mean(std_pnls):>+11.2f}% {np.mean(cu_pnls) if cu_pnls else float('nan'):>+11.2f}% {np.mean(pr_pnls) if pr_pnls else float('nan'):>+11.2f}%")
        print(f"  {'Median trade %':30s} {np.median(std_pnls):>+11.2f}% {np.median(cu_pnls) if cu_pnls else float('nan'):>+11.2f}% {np.median(pr_pnls) if pr_pnls else float('nan'):>+11.2f}%")
        print(f"  {'Std dev of trades':30s} {np.std(std_pnls):>12.2f}  {np.std(cu_pnls) if cu_pnls else float('nan'):>12.2f}  {np.std(pr_pnls) if pr_pnls else float('nan'):>12.2f}")

    print(f"\n  Hypothesis tests (H0: no difference between variants):")
    print(f"  {'Test':30s} {'PR vs Standard':>20} {'PR vs CU':>20}")

    # Welch t-test
    if pr_pnls and std_pnls:
        t1, p1 = welch_ttest(pr_pnls, std_pnls)
        t2, p2 = welch_ttest(pr_pnls, cu_pnls) if cu_pnls else (float("nan"), float("nan"))
        sig1 = "***" if p1 < 0.01 else ("**" if p1 < 0.05 else ("*" if p1 < 0.10 else "ns"))
        sig2 = "***" if p2 < 0.01 else ("**" if p2 < 0.05 else ("*" if p2 < 0.10 else "ns"))
        print(f"  {'Welch t-test (p-value)':30s} {p1:>14.4f} {sig1:>5}  {p2:>14.4f} {sig2:>5}")

    # Mann-Whitney U
    if pr_pnls and std_pnls:
        _, p1 = mann_whitney(pr_pnls, std_pnls)
        _, p2 = mann_whitney(pr_pnls, cu_pnls) if cu_pnls else (float("nan"), float("nan"))
        sig1 = "***" if p1 < 0.01 else ("**" if p1 < 0.05 else ("*" if p1 < 0.10 else "ns"))
        sig2 = "***" if p2 < 0.01 else ("**" if p2 < 0.05 else ("*" if p2 < 0.10 else "ns"))
        print(f"  {'Mann-Whitney U (p-value)':30s} {p1:>14.4f} {sig1:>5}  {p2:>14.4f} {sig2:>5}")

    print(f"\n  Significance: *** p<0.01  ** p<0.05  * p<0.10  ns = not significant")


# ── Bootstrap analysis ────────────────────────────────────────────────────────

def analyse_bootstrap(std_t, cu_t, pr_t, std_nav, cu_nav, pr_nav, label: str):
    print(f"\n{'─'*W}")
    print(f"  BOOTSTRAP ANALYSIS  [{label}]  (N={N_BOOTSTRAP:,} resamples)")
    print(f"{'─'*W}")

    # ── Total return bootstrap ────────────────────────────────────────────────
    std_pnls = [t["pnl_pct"] for t in std_t]
    cu_pnls  = [t["pnl_pct"] for t in cu_t]
    pr_pnls  = [t["pnl_pct"] for t in pr_t]

    print(f"\n  A) Trade-resample bootstrap — compounded total return (95% CI):")
    print(f"  {'Variant':20s} {'Observed':>12} {'CI 2.5%':>12} {'CI 97.5%':>12}")

    boot_std = bootstrap_total_return(std_pnls) if std_pnls else np.array([0.0])
    boot_cu  = bootstrap_total_return(cu_pnls)  if cu_pnls  else np.array([0.0])
    boot_pr  = bootstrap_total_return(pr_pnls)  if pr_pnls  else np.array([0.0])

    obs_std = (np.prod(np.array(std_pnls)/100+1)-1)*100 if std_pnls else 0.0
    obs_cu  = (np.prod(np.array(cu_pnls)/100+1)-1)*100  if cu_pnls  else 0.0
    obs_pr  = (np.prod(np.array(pr_pnls)/100+1)-1)*100  if pr_pnls  else 0.0

    for name, obs, boot in [("Standard", obs_std, boot_std),
                              ("CU (Confirmed Uptrend)", obs_cu, boot_cu),
                              ("PR (Pure Regime)", obs_pr, boot_pr)]:
        lo, hi = ci95(boot)
        print(f"  {name:20s} {obs:>+11.1f}%  [{lo:>+10.1f}%, {hi:>+10.1f}%]")

    p_pr_vs_std = p_value_greater(boot_pr, boot_std)
    p_pr_vs_cu  = p_value_greater(boot_pr, boot_cu)
    sig_vs_std = "***" if p_pr_vs_std < 0.01 else ("**" if p_pr_vs_std < 0.05 else ("*" if p_pr_vs_std < 0.10 else "ns"))
    sig_vs_cu  = "***" if p_pr_vs_cu < 0.01 else ("**" if p_pr_vs_cu < 0.05 else ("*" if p_pr_vs_cu < 0.10 else "ns"))
    print(f"\n  Bootstrap p-value (PR vs Standard): {p_pr_vs_std:.4f} {sig_vs_std}")
    print(f"  Bootstrap p-value (PR vs CU):       {p_pr_vs_cu:.4f} {sig_vs_cu}")

    # ── Block bootstrap Sharpe ────────────────────────────────────────────────
    print(f"\n  B) Block bootstrap Sharpe ratio (block={BLOCK_SIZE}d, 95% CI):")
    print(f"  {'Variant':20s} {'Observed':>12} {'CI 2.5%':>12} {'CI 97.5%':>12}")

    def _daily_ret(nav: pd.Series) -> pd.Series:
        return nav.dropna().pct_change().dropna()

    def _sharpe(rets: pd.Series) -> float:
        r = rets.dropna()
        return float(r.mean() / r.std() * np.sqrt(252)) if len(r) > 1 and r.std() > 0 else 0.0

    std_dr = _daily_ret(std_nav); cu_dr = _daily_ret(cu_nav); pr_dr = _daily_ret(pr_nav)
    sh_std = _sharpe(std_dr); sh_cu = _sharpe(cu_dr); sh_pr = _sharpe(pr_dr)

    bb_std = block_bootstrap_sharpe(std_dr)
    bb_cu  = block_bootstrap_sharpe(cu_dr)
    bb_pr  = block_bootstrap_sharpe(pr_dr)

    for name, obs, bb in [("Standard", sh_std, bb_std),
                            ("CU (Confirmed Uptrend)", sh_cu, bb_cu),
                            ("PR (Pure Regime)", sh_pr, bb_pr)]:
        lo, hi = ci95(bb[np.isfinite(bb)])
        print(f"  {name:20s} {obs:>12.3f}  [{lo:>10.3f}, {hi:>10.3f}]")

    p_sh_vs_std = p_value_greater(bb_pr[np.isfinite(bb_pr)], bb_std[np.isfinite(bb_std)])
    p_sh_vs_cu  = p_value_greater(bb_pr[np.isfinite(bb_pr)], bb_cu[np.isfinite(bb_cu)])
    sig_sh_std = "***" if p_sh_vs_std < 0.01 else ("**" if p_sh_vs_std < 0.05 else ("*" if p_sh_vs_std < 0.10 else "ns"))
    sig_sh_cu  = "***" if p_sh_vs_cu < 0.01 else ("**" if p_sh_vs_cu < 0.05 else ("*" if p_sh_vs_cu < 0.10 else "ns"))
    print(f"\n  Bootstrap p-value Sharpe (PR vs Standard): {p_sh_vs_std:.4f} {sig_sh_std}")
    print(f"  Bootstrap p-value Sharpe (PR vs CU):       {p_sh_vs_cu:.4f} {sig_sh_cu}")
    print(f"\n  Significance: *** p<0.01  ** p<0.05  * p<0.10  ns = not significant")


# ── Effect size ───────────────────────────────────────────────────────────────

def cohens_d(a: list, b: list) -> float:
    """Cohen's d: effect size for mean difference."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_sd = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    return (np.mean(a) - np.mean(b)) / pooled_sd if pooled_sd > 0 else float("nan")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    hr("═")
    print("  STATISTICAL SIGNIFICANCE: Pure Regime vs CU vs Standard  (MSTR, SL5)")
    print("  Tests: Welch t-test · Mann-Whitney U · Bootstrap total-return · Block-bootstrap Sharpe")
    print("  Caveat: short backtest window (~2 yrs) means low statistical power — interpret carefully")
    hr("═")

    cache: dict = {}

    for pname, (start, end, fs) in PERIODS.items():
        fetch_end = (pd.Timestamp(end) + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
        ck = (fs, fetch_end)
        if ck not in cache:
            print(f"\n  Fetching data for {pname}  ({fs} → {fetch_end}) …")
            df_raw = fetch_data(fs, fetch_end)
            preds  = build_ct_preds(df_raw)
            mstr_px = fetch_asset_prices("MSTR", fs, fetch_end)
            cache[ck] = dict(df_raw=df_raw, preds=preds, mstr_px=mstr_px)
        else:
            print(f"\n  [cache hit] {pname}")

        c = cache[ck]
        comp = _prep_comp(c["df_raw"], c["preds"], start, end)
        if len(comp) < WARMUP + 3:
            print(f"  [SKIP] {pname}: insufficient bars"); continue

        dates = pd.DatetimeIndex(comp["target_date"])
        sigs  = compute_signals(comp)
        sd    = pd.Timestamp(start)
        _bt0  = max(WARMUP, int(dates.searchsorted(sd)))

        px_raw = c["mstr_px"]
        a = px_raw.reindex(dates).ffill().bfill().values.astype(float)
        if np.sum(np.isfinite(a[_bt0:]) & (a[_bt0:] > 0)) < 5:
            print(f"  [SKIP] MSTR: insufficient prices"); continue

        std_t, std_nav = run_backtest(dates, a, sigs, sigs["tf_current"],  MSTR_SL, INITIAL_CAPITAL, sd)
        cu_t,  cu_nav  = run_backtest(dates, a, sigs, sigs["tf_option_b"], MSTR_SL, INITIAL_CAPITAL, sd)
        pr_t,  pr_nav  = run_backtest(dates, a, sigs, sigs["tf_option_c"], MSTR_SL, INITIAL_CAPITAL, sd)

        # Trim NAV to backtest window
        std_nav = std_nav.loc[dates[_bt0]:].dropna()
        cu_nav  = cu_nav.loc[dates[_bt0]:].dropna()
        pr_nav  = pr_nav.loc[dates[_bt0]:].dropna()

        # Summary headline
        print(f"\n{'═'*W}")
        print(f"  PERIOD: {pname}  |  ASSET: MSTR")
        std_ret = (std_nav.iloc[-1]/INITIAL_CAPITAL-1)*100 if not std_nav.empty else 0
        cu_ret  = (cu_nav.iloc[-1]/INITIAL_CAPITAL-1)*100  if not cu_nav.empty  else 0
        pr_ret  = (pr_nav.iloc[-1]/INITIAL_CAPITAL-1)*100  if not pr_nav.empty  else 0
        print(f"  Total return  —  Standard: {std_ret:+.1f}%  |  CU: {cu_ret:+.1f}%  |  PR: {pr_ret:+.1f}%")
        print(f"  Trade counts  —  Standard: {len(std_t)}      |  CU: {len(cu_t)}      |  PR: {len(pr_t)}")
        print(f"{'═'*W}")

        label = f"MSTR {pname}"
        analyse_trades(std_t, cu_t, pr_t, label)
        analyse_bootstrap(std_t, cu_t, pr_t, std_nav, cu_nav, pr_nav, label)

        # Effect sizes
        std_pnls = [t["pnl_pct"] for t in std_t]
        cu_pnls  = [t["pnl_pct"] for t in cu_t]
        pr_pnls  = [t["pnl_pct"] for t in pr_t]
        d_vs_std = cohens_d(pr_pnls, std_pnls) if pr_pnls and std_pnls else float("nan")
        d_vs_cu  = cohens_d(pr_pnls, cu_pnls)  if pr_pnls and cu_pnls  else float("nan")
        print(f"\n  Cohen's d effect size (PR vs Standard): {d_vs_std:>+.3f}  "
              f"({'large' if abs(d_vs_std)>0.8 else 'medium' if abs(d_vs_std)>0.5 else 'small' if abs(d_vs_std)>0.2 else 'negligible'})")
        print(f"  Cohen's d effect size (PR vs CU):       {d_vs_cu:>+.3f}  "
              f"({'large' if abs(d_vs_cu)>0.8 else 'medium' if abs(d_vs_cu)>0.5 else 'small' if abs(d_vs_cu)>0.2 else 'negligible'})")

    hr("═")
    print("""
  INTERPRETATION GUIDE
  ─────────────────────
  Statistical power is severely limited by the short backtest window (~2 years,
  ~15–40 trades per variant). Even a genuinely superior strategy may fail to
  achieve p<0.05 due to small sample size.

  Key things to look for:
  • p < 0.05 on BOTH t-test AND Mann-Whitney → stronger evidence
  • Bootstrap CI for PR that does not overlap with Standard or CU → meaningful gap
  • Cohen's d > 0.5 (medium) even if p > 0.05 → economically significant
  • Consistent direction across all three periods (Bull, Bear, Full)
  • If p > 0.10 everywhere → insufficient evidence, cannot claim significance

  Bottom line: backtests over 2 years are almost never statistically significant
  at conventional levels. Use effect sizes and CI overlap as the practical guide.
    """)
    hr("═")


if __name__ == "__main__":
    main()
