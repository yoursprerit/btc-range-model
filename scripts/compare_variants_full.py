#!/usr/bin/env python3
"""
Full-period variant comparison matching the UI backtest logic exactly.

Logic:
  BTC  — NO stop-loss; exits only on D3 (bull) or D2/D3 (bear)   [same as UI tab]
  MSTR — −3 % stop-loss, SL5 regime-adaptive re-entry             [same as UI tab]
  MSTU — −7 % stop-loss, SL5 regime-adaptive re-entry             [same as UI tab]

Prices: versioned data/backtest/ CSVs (v1 · 2026-06-15)
Signals: CT model predictions (live)

Full period: Jun 2024 → May 2026
"""

import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

try:
    import sklearn._loss._loss as _e
    if "_loss" not in sys.modules: sys.modules["_loss"] = _e
    import sklearn._loss.loss, sklearn._loss.link
except Exception: pass

import numpy as np
import pandas as pd

# ── reuse signal generation from backtest_option_c ───────────────────────────
from backtest_option_c import (
    fetch_data, build_ct_preds, _prep_comp, compute_signals,
    WARMUP, INITIAL_CAPITAL,
)

DATA_DIR = _ROOT / "data" / "backtest"
CAP      = INITIAL_CAPITAL

FULL_START = "2024-06-01"
FULL_END   = "2026-05-31"
PRE_START  = "2024-03-01"   # warm-up fetch start

VARIANTS = {
    "Standard (above_ma30)":   "tf_current",
    "Confirmed Uptrend (CU)":  "tf_option_b",
    "Pure Regime (PR)":        "tf_option_c",
}

# ─── load versioned CSVs ─────────────────────────────────────────────────────

def _load_csv(fname):
    df = pd.read_csv(DATA_DIR / fname, index_col=0, parse_dates=True)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    df.columns = [c.lower() for c in df.columns]
    return df

btc_csv  = _load_csv("btc_usd_daily.csv")
mstr_csv = _load_csv("mstr_daily.csv")
mstu_csv = _load_csv("mstu_synthetic_daily.csv")   # full history (OLS pre-inception)

# ─── BTC backtest — NO stop-loss ─────────────────────────────────────────────

def run_btc_backtest(dates, btc_px, sigs, entry_key, bt_start):
    N      = len(dates)
    d2     = sigs["d2"];  d3 = sigs["d3"];  bull = sigs["bull_regime"]
    tf     = sigs[entry_key]
    _bt0   = max(WARMUP, int(pd.DatetimeIndex(dates).searchsorted(bt_start)))
    nav    = CAP; pos = "CASH"; qty = 0.0
    trades = []; nav_arr = np.full(N, np.nan)

    for i in range(N):
        price = btc_px[i]
        if i < _bt0:
            nav_arr[i] = CAP; continue
        if not np.isfinite(price) or price <= 0:
            nav_arr[i] = qty * btc_px[i-1] if pos == "LONG" else nav; continue

        if pos == "LONG":
            should_exit = bool(d3[i] or (d2[i] and not bull[i]))
            if should_exit:
                nav = qty * price
                trades.append(dict(
                    pnl_pct=(price/e_price-1)*100,
                    duration=(dates[i]-e_date).days,
                    stop=False,
                ))
                pos = "CASH"; qty = 0.0
            else:
                nav = qty * price
        else:
            _exit_at_i = d3[i] or (d2[i] and not bull[i])
            if tf[i] and not _exit_at_i:
                qty = nav / price; e_price = price; e_date = dates[i]
                pos = "LONG"

        nav_arr[i] = qty * price if pos == "LONG" else nav

    if pos == "LONG":
        nav = qty * btc_px[N-1]

    return trades, pd.Series(nav_arr, index=pd.DatetimeIndex(dates))

# ─── MSTR/MSTU backtest — with SL5 regime-adaptive re-entry ─────────────────

def run_sl_backtest(dates, asset_px, sigs, entry_key, sl_pct, bt_start):
    N    = len(dates)
    d2   = sigs["d2"];  d3 = sigs["d3"];  bull = sigs["bull_regime"]
    tf   = sigs[entry_key]
    _bt0 = max(WARMUP, int(pd.DatetimeIndex(dates).searchsorted(bt_start)))
    nav  = CAP; pos = "CASH"; qty = 0.0; stop_px = 0.0
    from_sl = False; bars_since_sl = 0
    e_price = e_date = None
    trades = []; nav_arr = np.full(N, np.nan)

    for i in range(N):
        price = asset_px[i]
        if i < _bt0:
            nav_arr[i] = CAP; continue
        if not np.isfinite(price) or price <= 0:
            nav_arr[i] = qty * asset_px[i-1] if pos == "LONG" else nav; continue

        if pos == "LONG":
            if price <= stop_px:
                nav = qty * price
                trades.append(dict(
                    pnl_pct=(price/e_price-1)*100,
                    duration=(dates[i]-e_date).days,
                    stop=True,
                ))
                pos = "CASH"; qty = 0.0; stop_px = 0.0
                from_sl = True; bars_since_sl = 0
            else:
                should_exit = bool(d3[i] or (d2[i] and not bull[i]))
                if should_exit:
                    nav = qty * price
                    trades.append(dict(
                        pnl_pct=(price/e_price-1)*100,
                        duration=(dates[i]-e_date).days,
                        stop=False,
                    ))
                    pos = "CASH"; qty = 0.0; stop_px = 0.0; from_sl = False
                else:
                    nav = qty * price
        else:
            if from_sl:
                bars_since_sl += 1
            _exit_at_i = d3[i] or (d2[i] and not bull[i])
            _sl_ok = (not from_sl or bool(bull[i]) or bars_since_sl >= 10)
            if tf[i] and _sl_ok and (from_sl or not _exit_at_i):
                qty = nav / price; e_price = price; e_date = dates[i]
                pos = "LONG"; stop_px = price * (1 - sl_pct)
                from_sl = False; bars_since_sl = 0

        nav_arr[i] = qty * price if pos == "LONG" else nav

    if pos == "LONG":
        nav = qty * asset_px[N-1]

    return trades, pd.Series(nav_arr, index=pd.DatetimeIndex(dates))

# ─── helpers ─────────────────────────────────────────────────────────────────

def stats(trades, nav_s):
    nav_clean = nav_s.dropna()
    final_nav = float(nav_clean.iloc[-1])
    ret = (final_nav / CAP - 1) * 100
    wins   = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    wr     = 100 * len(wins) / max(1, len(trades))
    avg_p  = float(np.mean([t["pnl_pct"] for t in trades])) if trades else 0.0
    best_t = max((t["pnl_pct"] for t in trades), default=0.0)
    worst_t= min((t["pnl_pct"] for t in trades), default=0.0)
    rm     = nav_clean.cummax()
    max_dd = float(((nav_clean - rm) / rm * 100).min())
    dr     = nav_clean.pct_change().fillna(0)
    rf_d   = (1.045) ** (1/252) - 1
    exc    = dr - rf_d
    sharpe = float(exc.mean() / exc.std() * np.sqrt(252)) if exc.std() > 0 else 0.0
    n_sl   = sum(1 for t in trades if t.get("stop"))
    return dict(
        ret=ret, final_nav=final_nav, n_trades=len(trades),
        n_wins=len(wins), n_losses=len(losses), n_sl=n_sl,
        win_rate=wr, avg_pnl=avg_p, best=best_t, worst=worst_t,
        max_dd=max_dd, sharpe=sharpe,
    )

# ─── main ────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═"*90)
    print("  VARIANT COMPARISON — FULL PERIOD  Jun 2024 → May 2026")
    print("  Using versioned dataset v1 (data/backtest/) · 2026-06-15")
    print("  BTC: no stop-loss  |  MSTR: −3% SL  |  MSTU: −7% SL (UI-exact logic)")
    print("═"*90)

    # ── fetch signals ─────────────────────────────────────────────────────────
    fetch_start = (pd.Timestamp(PRE_START) - pd.Timedelta(days=200)).strftime("%Y-%m-%d")
    fetch_end   = (pd.Timestamp(FULL_END)  + pd.Timedelta(days=3)).strftime("%Y-%m-%d")

    print(f"\nFetching CT model predictions ({fetch_start} → {fetch_end}) …")
    raw_df = fetch_data(fetch_start, fetch_end)

    # Override BTC actuals with versioned CSV
    raw_df["btc_close"] = btc_csv["close"].reindex(raw_df.index).ffill().bfill()
    raw_df["btc_high"]  = btc_csv["high"].reindex(raw_df.index).ffill().bfill()
    raw_df["btc_low"]   = btc_csv["low"].reindex(raw_df.index).ffill().bfill()

    preds = build_ct_preds(raw_df)
    comp  = _prep_comp(raw_df, preds, PRE_START, FULL_END)
    sigs  = compute_signals(comp)
    dates = pd.DatetimeIndex(comp["target_date"])
    bt_start = pd.Timestamp(FULL_START)
    N = len(dates)

    # ── BTC execution prices from versioned CSV ───────────────────────────────
    btc_px = btc_csv["close"].reindex(dates).ffill().bfill().values.astype(float)

    # BTC buy-and-hold
    _b0      = max(WARMUP, int(dates.searchsorted(bt_start)))
    bh_btc   = pd.Series(
        CAP * btc_px[_b0:] / btc_px[_b0], index=dates[_b0:]
    )
    bh_btc_ret = (bh_btc.iloc[-1] / CAP - 1) * 100

    # ── MSTR execution prices from versioned CSV ──────────────────────────────
    mstr_all = mstr_csv["close"].reindex(
        pd.date_range(mstr_csv.index[0], mstr_csv.index[-1], freq="D")
    ).ffill()
    mstr_px = mstr_all.reindex(dates).ffill().bfill().values.astype(float)

    bh_mstr_ret = (mstr_px[_b0:].sum() / mstr_px[_b0] * 0 +   # avoid confusion
                   (mstr_px[_b0+len(mstr_px[_b0:])-1] / mstr_px[_b0] - 1) * 100)
    bh_mstr_last = mstr_px[min(_b0 + len(dates) - _b0 - 1, N-1)]
    bh_mstr_ret = (bh_mstr_last / mstr_px[_b0] - 1) * 100

    # ── MSTU execution prices from versioned synthetic CSV ────────────────────
    mstu_all = mstu_csv["close"].reindex(
        pd.date_range(mstu_csv.index[0], mstu_csv.index[-1], freq="D")
    ).ffill()
    mstu_px = mstu_all.reindex(dates).ffill().bfill().values.astype(float)
    bh_mstu_ret = (mstu_px[N-1] / mstu_px[_b0] - 1) * 100

    # ── run all variants ──────────────────────────────────────────────────────
    results = {"BTC": {}, "MSTR": {}, "MSTU": {}}
    bh_rets  = {"BTC": bh_btc_ret, "MSTR": bh_mstr_ret, "MSTU": bh_mstu_ret}

    for vname, vkey in VARIANTS.items():
        t_btc,  n_btc  = run_btc_backtest(dates, btc_px,  sigs, vkey, bt_start)
        t_mstr, n_mstr = run_sl_backtest(dates, mstr_px, sigs, vkey, 0.03, bt_start)
        t_mstu, n_mstu = run_sl_backtest(dates, mstu_px, sigs, vkey, 0.07, bt_start)

        results["BTC"][vname]  = stats(t_btc,  n_btc)
        results["MSTR"][vname] = stats(t_mstr, n_mstr)
        results["MSTU"][vname] = stats(t_mstu, n_mstu)

    # ── print results ─────────────────────────────────────────────────────────
    for asset in ["BTC", "MSTR", "MSTU"]:
        sl_note = "no SL" if asset == "BTC" else ("−3% SL" if asset == "MSTR" else "−7% SL")
        print(f"\n{'─'*90}")
        print(f"  {asset} — Full Period Jun 2024 → May 2026  [{sl_note}]")
        print(f"  Buy-and-Hold: {bh_rets[asset]:+.1f}%")
        print(f"{'─'*90}")
        print(f"  {'Variant':<32} {'Return':>9} {'vs B&H':>8} {'Trades':>7} {'Win%':>7} "
              f"{'AvgPnL':>8} {'Best':>8} {'Worst':>8} {'MaxDD':>8} {'Sharpe':>7}")
        print(f"  {'─'*32} {'─'*9} {'─'*8} {'─'*7} {'─'*7} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*7}")

        best_ret = -999
        best_var = ""
        for vname, s in results[asset].items():
            alpha = s["ret"] - bh_rets[asset]
            short = vname.split("(")[0].strip()
            marker = " ◀ BEST" if s["ret"] == max(r["ret"] for r in results[asset].values()) else ""
            print(f"  {short:<32} {s['ret']:>+8.1f}% {alpha:>+7.1f}pp {s['n_trades']:>7d} "
                  f"{s['win_rate']:>6.0f}% {s['avg_pnl']:>+7.1f}% {s['best']:>+7.1f}% "
                  f"{s['worst']:>+7.1f}% {s['max_dd']:>+7.1f}% {s['sharpe']:>7.2f}{marker}")
            if s["ret"] > best_ret:
                best_ret = s["ret"]; best_var = vname

        print(f"\n  ★ Winner: {best_var} → {best_ret:+.1f}%")

    # ── summary table ─────────────────────────────────────────────────────────
    print(f"\n{'═'*90}")
    print(f"  WINNER SUMMARY — Full Period (Jun 2024 → May 2026)")
    print(f"{'─'*90}")
    for asset in ["BTC", "MSTR", "MSTU"]:
        sl_note = "no SL" if asset == "BTC" else ("−3% SL" if asset == "MSTR" else "−7% SL")
        best_v  = max(results[asset], key=lambda v: results[asset][v]["ret"])
        best_r  = results[asset][best_v]["ret"]
        bh_r    = bh_rets[asset]
        short   = best_v.split("(")[0].strip()
        print(f"  {asset:<6} [{sl_note:<7}]  ★ {short:<26} {best_r:+.1f}%  "
              f"(B&H {bh_r:+.1f}%,  α={best_r-bh_r:+.1f}pp)")
    print("═"*90 + "\n")


if __name__ == "__main__":
    main()
