"""Stop-loss threshold sweep for the MSTR and MSTU Backtesting tabs.

Reuses the exact MSTU/MSTR tab methodology (same versioned data/backtest/
dataset, same CT prediction pipeline, same TF2+V-Gate signals, same SL5
re-entry logic, same statistics).  Signals are built once per period; only
the stop-loss threshold and the executed instrument change between runs.

Current production thresholds:  MSTR = 3% (price*0.97),  MSTU = 7% (price*0.93).

Run:  python backtest_stop_threshold_sweep.py [entry_gate]
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

from backtest_mstu_mstr_stop import (
    build_backtest_preds, _load_mstr_prices, _load_mstu_synthetic,
)


# ──────────────────────────────────────────────────────────────────────────
# Build all signal + price arrays for a period (once)
# ──────────────────────────────────────────────────────────────────────────
def build_signals(end_date_iso: str, backtest_start_iso: str, entry_gate: str):
    WARMUP = 35
    end_dt = pd.Timestamp(end_date_iso); start_dt = pd.Timestamp(backtest_start_iso)
    pre_dt = start_dt - pd.Timedelta(days=60)
    fetch_start = (start_dt - pd.Timedelta(days=200)).strftime("%Y-%m-%d")
    fetch_end   = (end_dt + pd.Timedelta(days=3)).strftime("%Y-%m-%d")

    preds, raw_df = build_backtest_preds(fetch_start, fetch_end)
    btc_h = raw_df["btc_high"]; btc_l = raw_df["btc_low"]; btc_c = raw_df["btc_close"]
    preds = preds.loc[(preds.index >= pre_dt) & (preds.index <= end_dt)].copy()
    preds["actual_high"]  = btc_h.reindex(preds.index).values
    preds["actual_low"]   = btc_l.reindex(preds.index).values
    preds["actual_close"] = btc_c.reindex(preds.index).values
    comp = preds.dropna(subset=["actual_high", "actual_low", "actual_close"]).reset_index()
    N = len(comp)
    dates = pd.DatetimeIndex(comp["target_date"])
    _bt0 = max(WARMUP, int(dates.searchsorted(start_dt)))

    mstu_px = _load_mstu_synthetic().reindex(dates).ffill().bfill().values.astype(float)
    _mstr_close = _load_mstr_prices()["close"].sort_index()
    _mstr_all = _mstr_close.reindex(
        pd.date_range(_mstr_close.index[0], max(_mstr_close.index[-1], end_dt), freq="D"))
    mstr_px = _mstr_all.reindex(dates).ffill().bfill().values.astype(float)

    c_asof  = comp["close_asof"].values.astype(float)
    pred_hi = comp["pred_high"].values.astype(float)
    pred_lo = comp["pred_low"].values.astype(float)
    act_hi  = comp["actual_high"].values.astype(float)
    act_lo  = comp["actual_low"].values.astype(float)
    err_hi = (act_hi - pred_hi) / c_asof * 100
    err_lo = (pred_lo - act_lo) / c_asof * 100
    hi_brk = (act_hi > pred_hi).astype(int)
    lo_brk = (act_lo < pred_lo).astype(int)

    ehma3 = np.zeros(N); elma3 = np.zeros(N); hb3 = np.zeros(N, dtype=int); lb3 = np.zeros(N, dtype=int)
    for i in range(N):
        s = max(0, i - 2)
        ehma3[i] = np.mean(err_hi[s:i + 1]); elma3[i] = np.mean(err_lo[s:i + 1])
        hb3[i] = int(np.sum(hi_brk[s:i + 1])); lb3[i] = int(np.sum(lo_brk[s:i + 1]))
    u1 = (ehma3 > 0.7) & (hb3 >= 2)
    d1 = (lb3 >= 2) & (elma3 > 0.5)
    d2 = ehma3 < -0.75
    d3 = np.zeros(N, dtype=bool)
    for i in range(1, N):
        consec = 0
        for k in range(i - 1, -1, -1):
            if hi_brk[k]: consec += 1
            else: break
        if consec >= 3 and lo_brk[i]: d3[i] = True
    ma30 = np.full(N, np.nan)
    for i in range(N):
        w = min(30, i + 1); ma30[i] = np.mean(c_asof[max(0, i - w + 1):i + 1])
    above_ma30 = c_asof > ma30
    ma30_slope_pos = np.zeros(N, dtype=bool)
    for i in range(N):
        if i >= 5 and np.isfinite(ma30[i]) and np.isfinite(ma30[i - 5]):
            ma30_slope_pos[i] = ma30[i] > ma30[i - 5]
    bull_regime = above_ma30 & ma30_slope_pos
    clean_10d = np.zeros(N, dtype=bool)
    for i in range(N):
        clean_10d[i] = not bool(np.any(d1[max(0, i - 7):i] | d2[max(0, i - 7):i]))
    roll_ehi_norm = np.array([float(np.mean(err_hi[max(0, i - 29):i + 1])) for i in range(N)])
    dn_score = np.zeros(N)
    for i in range(N):
        norm = max(abs(roll_ehi_norm[i]), 0.01)
        dn_score[i] = ((-ehma3[i] / norm) * 0.30 + (lb3[i] / 3.0) * 0.30 +
                       (elma3[i] / max(abs(elma3[i]), 0.10)) * 0.20 + float(lo_brk[i]) * 0.20)
    v_rev_bar = (dn_score > 0.8) & (err_lo > 3.0)
    v_recent = np.zeros(N, dtype=bool)
    for i in range(N):
        v_recent[i] = bool(np.any(v_rev_bar[max(0, i - 2):i + 1]))

    if entry_gate == "above_ma30":
        tf1_entry = u1 & ((above_ma30 ^ clean_10d) | v_recent)
    elif entry_gate == "bull_regime":
        tf1_entry = u1 & ((bull_regime ^ clean_10d) | v_recent)
    else:
        tf1_entry = u1 & (bull_regime | (clean_10d & ~above_ma30) | v_recent)

    return dict(N=N, _bt0=_bt0, dates=dates, mstu_px=mstu_px, mstr_px=mstr_px,
                d2=d2, d3=d3, bull_regime=bull_regime, above_ma30=above_ma30,
                v_recent=v_recent, tf1_entry=tf1_entry)


# ──────────────────────────────────────────────────────────────────────────
# Generic backtest loop (identical to run_mstr/mstu_backtest); stop_pct varies.
# stop_pct=None disables the stop entirely (signal-only exits).
# ──────────────────────────────────────────────────────────────────────────
def run_loop(sig: dict, instrument: str, stop_pct: float | None,
             initial_capital: float = 100_000.0):
    px = sig["mstr_px"] if instrument == "MSTR" else sig["mstu_px"]
    N = sig["N"]; _bt0 = sig["_bt0"]; dates = sig["dates"]
    d2 = sig["d2"]; d3 = sig["d3"]; bull_regime = sig["bull_regime"]
    above_ma30 = sig["above_ma30"]; v_recent = sig["v_recent"]; tf1_entry = sig["tf1_entry"]
    stop_mult = (1.0 - stop_pct) if stop_pct is not None else None

    nav = initial_capital; pos = "CASH"; qty = 0.0
    e_price = e_nav = e_date = e_trigger = None
    e_reentry = False; stop_px = 0.0
    from_sl = False; bars_since_sl = 0
    trades = []; nav_arr = np.full(N, np.nan)

    for i in range(N):
        price = px[i]
        if i < _bt0:
            nav_arr[i] = initial_capital; continue
        if not np.isfinite(price) or price <= 0:
            nav_arr[i] = qty * px[i - 1] if pos == "LONG" and i > 0 else nav
            continue
        if pos == "LONG":
            cur = qty * price
            if stop_mult is not None and price <= stop_px:
                nav = qty * price
                trades.append(dict(entry_price=e_price, entry_nav=e_nav, exit_date=dates[i],
                                   pnl_pct=(price / e_price - 1) * 100, pnl_abs=nav - e_nav,
                                   duration_days=(dates[i] - e_date).days, stop_triggered=True,
                                   was_reentry=e_reentry))
                pos = "CASH"; qty = 0.0; stop_px = 0.0; e_reentry = False
                from_sl = True; bars_since_sl = 0
            else:
                should_exit = bool(d3[i] or (d2[i] and not bull_regime[i]))
                if should_exit:
                    nav = cur
                    trades.append(dict(entry_price=e_price, entry_nav=e_nav, exit_date=dates[i],
                                       pnl_pct=(price / e_price - 1) * 100, pnl_abs=nav - e_nav,
                                       duration_days=(dates[i] - e_date).days, stop_triggered=False,
                                       was_reentry=e_reentry))
                    pos = "CASH"; qty = 0.0; stop_px = 0.0; e_reentry = False; from_sl = False
                else:
                    nav = cur
        else:
            if from_sl:
                bars_since_sl += 1
            _exit_at_i = d3[i] or (d2[i] and not bull_regime[i])
            _sl_reentry_ok = (not from_sl or bool(bull_regime[i]) or bars_since_sl >= 10)
            if tf1_entry[i] and _sl_reentry_ok and (from_sl or not _exit_at_i):
                e_reentry = bool(from_sl)
                qty = nav / price; e_price = price; e_date = dates[i]; e_nav = nav; pos = "LONG"
                stop_px = price * stop_mult if stop_mult is not None else 0.0
                from_sl = False; bars_since_sl = 0
        nav_arr[i] = qty * price if pos == "LONG" else nav

    if pos == "LONG" and np.isfinite(px[N - 1]) and px[N - 1] > 0:
        nav_arr[N - 1] = qty * px[N - 1]

    nav_series = pd.Series(nav_arr[_bt0:], index=dates[_bt0:]).ffill()
    bh_series = pd.Series(initial_capital * px[_bt0:] / px[_bt0], index=dates[_bt0:])
    final_nav = float(nav_series.iloc[-1])
    strat_ret = (final_nav / initial_capital - 1) * 100
    bh_ret = (float(bh_series.iloc[-1]) / initial_capital - 1) * 100
    wins = [t for t in trades if t["pnl_pct"] > 0]
    win_rate = 100 * len(wins) / len(trades) if trades else 0.0
    rm = nav_series.cummax(); max_dd = float(((nav_series - rm) / rm * 100).min())
    days_in = sum(t["duration_days"] for t in trades)
    tot_days = max(1, (dates[N - 1] - dates[_bt0]).days)
    rf_daily = (1.045) ** (1 / 252) - 1
    dr_s = nav_series.pct_change().fillna(0); exc_s = dr_s - rf_daily
    sharpe = float(exc_s.mean() / exc_s.std() * np.sqrt(252)) if exc_s.std() > 0 else 0.0
    TAX = 0.35; _net: dict = {}
    for t in trades:
        yr = pd.Timestamp(t["exit_date"]).year
        _net[yr] = _net.get(yr, 0.0) + t["pnl_abs"]
    after_tax_ret = ((final_nav - sum(TAX * max(0.0, n) for n in _net.values())) /
                     initial_capital - 1) * 100
    return dict(strat_ret=strat_ret, bh_ret=bh_ret, max_drawdown=max_dd, sharpe=sharpe,
                n_trades=len(trades), n_stops=sum(1 for t in trades if t["stop_triggered"]),
                win_rate=win_rate, after_tax_ret=after_tax_ret, alpha=final_nav - float(bh_series.iloc[-1]))


PERIODS = [
    ("Bear (Jun25-May26)", "2026-05-31", "2025-06-04"),
    ("Bull (Jun24-Jun25)", "2025-06-14", "2024-06-01"),
    ("Full (Jun24-May26)", "2026-05-31", "2024-06-01"),
]
MSTR_GRID = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, None]
MSTU_GRID = [0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15, None]
CURRENT = {"MSTR": 0.03, "MSTU": 0.07}


def _fmt_pct(p):
    return "no-stop" if p is None else f"{p*100:.0f}%"


def sweep_instrument(instrument: str, grid, signals_by_period: dict):
    print(f"\n{'='*112}\n{instrument} stop-loss threshold sweep   (current = {_fmt_pct(CURRENT[instrument])})\n{'='*112}")
    hdr = (f"{'Period':<20}{'Stop':>8}{'Strat%':>11}{'B&H%':>9}{'Alpha$':>12}"
           f"{'MaxDD%':>9}{'Sharpe':>8}{'Trades':>7}{'Stops':>6}{'Win%':>7}{'AfterTax%':>10}")
    for label, end_iso, start_iso in PERIODS:
        print(hdr); print("-" * len(hdr))
        sig = signals_by_period[label]
        for sp in grid:
            r = run_loop(sig, instrument, sp)
            mark = "  <- current" if sp == CURRENT[instrument] else ""
            print(f"{label:<20}{_fmt_pct(sp):>8}{r['strat_ret']:>11.1f}{r['bh_ret']:>9.1f}"
                  f"{r['alpha']:>12,.0f}{r['max_drawdown']:>9.1f}{r['sharpe']:>8.2f}"
                  f"{r['n_trades']:>7d}{r['n_stops']:>6d}{r['win_rate']:>7.1f}"
                  f"{r['after_tax_ret']:>10.1f}{mark}")
        print()


def main():
    gate = sys.argv[1] if len(sys.argv) > 1 else "bull_regime"
    print(f"\nStop-loss threshold sweep · entry_gate={gate} · dataset v2 (2026-06-15)")
    signals_by_period = {lbl: build_signals(e, s, gate) for lbl, e, s in PERIODS}
    sweep_instrument("MSTR", MSTR_GRID, signals_by_period)
    sweep_instrument("MSTU", MSTU_GRID, signals_by_period)


if __name__ == "__main__":
    main()
