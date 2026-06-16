"""Evaluate: does triggering the MSTU stop-loss exit off the MSTR 3% stop
(instead of MSTU's own 7% close stop) improve MSTU backtest performance?

This is a self-contained replica of the MSTU Backtesting tab's methodology
(``run_mstu_backtest`` in app/btc_hourly_app.py).  It reuses the exact same
versioned dataset (data/backtest/), the same CT prediction pipeline
(``_build_backtest_preds`` CSV path), the same TF2+V-Gate signal construction,
and the same backtest loop / statistics.  The ONLY thing that changes between
runs is the stop-loss *trigger*:

  * baseline   — MSTU close <= mstu_entry * 0.93   (current, 7% MSTU stop)
  * mstr_stop  — MSTR close <= mstr_entry * 0.97   (3% MSTR stop drives MSTU exit)
  * combined   — whichever of the two triggers first

Exit fill in every case is the MSTU close on the trigger bar (matching the app).
All other behaviour — D2/D3 regime exits, SL5 regime-adaptive re-entry, entry
gate — is identical and shared across variants.

Run:  python backtest_mstu_mstr_stop.py
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
import joblib

from paths import DAILY_MODEL_CT, DATA_DIR

_BT_DATA = DATA_DIR / "backtest"


# ──────────────────────────────────────────────────────────────────────────
# Data loaders (mirror app/btc_hourly_app.py exactly)
# ──────────────────────────────────────────────────────────────────────────
def _read_price_csv(filename: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(_BT_DATA / filename, index_col=0, parse_dates=True)
        df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception:
        return None


def _load_raw_features() -> pd.DataFrame | None:
    df = pd.read_csv(_BT_DATA / "raw_features_daily.csv", index_col=0, parse_dates=True)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    return df


def _load_mstr_prices() -> pd.DataFrame | None:
    return _read_price_csv("mstr_daily.csv")


def _load_mstu_prices() -> pd.DataFrame | None:
    return _read_price_csv("mstu_daily.csv")


def _load_mstu_synthetic() -> pd.Series | None:
    df = _read_price_csv("mstu_synthetic_daily.csv")
    return df["close"] if df is not None and "close" in df.columns else None


# ──────────────────────────────────────────────────────────────────────────
# CT prediction pipeline — CSV path of _build_backtest_preds (verbatim)
# ──────────────────────────────────────────────────────────────────────────
def build_backtest_preds(fetch_start_iso: str, fetch_end_iso: str):
    AD_bt = joblib.load(str(DAILY_MODEL_CT))
    _fs = pd.Timestamp(fetch_start_iso); _fe = pd.Timestamp(fetch_end_iso)
    _rf = _load_raw_features()
    _rf_slice = _rf.loc[(_rf.index >= _fs) & (_rf.index <= _fe)]
    if len(_rf_slice) < 35:
        return None
    df = _rf_slice.copy()
    raw_df = df[["btc_close", "btc_high", "btc_low", "btc_volume"]].copy()

    c = df["btc_close"]; h = df["btc_high"]
    l_ = df["btc_low"];  v = df["btc_volume"]
    ret = np.log(c).diff()
    feat = pd.DataFrame(index=df.index)
    for k in [1, 3, 5, 7, 14, 30]: feat[f"ret_{k}"] = ret.rolling(k).sum()
    for k in [5, 10, 20, 30]:      feat[f"vol_{k}"] = ret.rolling(k).std()
    prev_c = c.shift(1)
    tr = pd.concat([(h - l_), (h - prev_c).abs(), (l_ - prev_c).abs()], axis=1).max(axis=1)
    for k in [7, 14, 30]: feat[f"atr_{k}"] = tr.rolling(k).mean() / c
    feat["range_today"] = (h - l_) / c
    feat["range_ma7"]   = ((h - l_) / c).rolling(7).mean()
    feat["range_ma30"]  = ((h - l_) / c).rolling(30).mean()
    feat["range_std30"] = ((h - l_) / c).rolling(30).std()
    gain = c.diff().clip(lower=0).rolling(14).mean()
    loss = (-c.diff().clip(upper=0)).rolling(14).mean()
    feat["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    e12 = c.ewm(span=12, adjust=False).mean(); e26 = c.ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    feat["macd"]      = macd / c
    feat["macd_sig"]  = macd.ewm(span=9, adjust=False).mean() / c
    feat["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    ma20 = c.rolling(20).mean(); sd20 = c.rolling(20).std()
    feat["bb_width"]   = (4 * sd20) / ma20
    feat["dist_hi_30"] = c / c.rolling(30).max() - 1
    feat["dist_lo_30"] = c / c.rolling(30).min() - 1
    feat["dist_hi_90"] = c / c.rolling(90).max() - 1
    feat["vol_chg_1"]    = np.log(v).diff()
    feat["vol_z_20"]     = (np.log(v) - np.log(v).rolling(20).mean()) / np.log(v).rolling(20).std()
    feat["vol_ma_ratio"] = v / v.rolling(20).mean()
    dow = df.index.dayofweek
    for i in range(6): feat[f"dow_{i}"] = (dow == i).astype(float)
    for nm in ["spx", "ndx", "vix", "gold", "dxy", "tnx", "eth"]:
        col = f"{nm}_close"
        if col not in df.columns: continue
        s = df[col]; lr = np.log(s).diff()
        for k in [1, 5, 20]: feat[f"{nm}_ret_{k}"] = lr.rolling(k).sum()
        feat[f"{nm}_vol_20"] = lr.rolling(20).std()
    for corr_nm, corr_col in [("spx", "spx_close"), ("ndx", "ndx_close"),
                              ("gold", "gold_close"), ("dxy", "dxy_close")]:
        if corr_col in df.columns:
            feat[f"btc_{corr_nm}_corr_30"] = ret.rolling(30).corr(np.log(df[corr_col]).diff())
    for col in [x for x in df.columns if x.startswith("oc_")]:
        s = df[col].astype(float); sl = np.log(s.replace(0, np.nan))
        feat[f"{col}_d1"]  = sl.diff(1); feat[f"{col}_d7"] = sl.diff(7)
        feat[f"{col}_z30"] = (sl - sl.rolling(30).mean()) / sl.rolling(30).std()
    nh = h.shift(-1); nl = l_.shift(-1)
    y_hi = (nh - c) / c; y_lo = (c - nl) / c
    feat["y_hi_ema3"] = y_hi.shift(1).ewm(span=3, adjust=False).mean()
    feat["y_lo_ema3"] = y_lo.shift(1).ewm(span=3, adjust=False).mean()
    feat["y_hi_ema7"] = y_hi.shift(1).ewm(span=7, adjust=False).mean()
    feat["y_lo_ema7"] = y_lo.shift(1).ewm(span=7, adjust=False).mean()
    p3h = h.shift(1).rolling(3).max(); p3l = l_.shift(1).rolling(3).min()
    feat["above_3d_high"]  = (c > p3h).astype(float)
    feat["below_3d_low"]   = (c < p3l).astype(float)
    feat["bo_strength_up"] = (c / p3h - 1).clip(lower=0)
    feat["bo_strength_dn"] = (1 - c / p3l).clip(lower=0)
    yhl = y_hi.shift(1); yll = y_lo.shift(1)
    feat["y_hi_surprise"] = yhl - yhl.ewm(span=7, adjust=False).mean()
    feat["y_lo_surprise"] = yll - yll.ewm(span=7, adjust=False).mean()
    nr = ret.clip(upper=0)
    feat["dn_vol_5"]  = nr.rolling(5).std()
    feat["dn_vol_20"] = nr.rolling(20).std()
    sma50 = c.rolling(50).mean()
    feat["below_sma50"]    = (c < sma50).astype(float)
    feat["below_sma50_5d"] = feat["below_sma50"].rolling(5).min().fillna(0)
    for _cb_col in ["cb_premium", "cb_premium_ma3", "cb_premium_z7"]:
        feat[_cb_col] = df[_cb_col].fillna(0.0)

    fc = AD_bt["feat_cols"]
    feat = feat.replace([np.inf, -np.inf], np.nan)
    for col in fc:
        if col not in feat.columns: feat[col] = np.nan
    F = feat[fc].dropna()
    if F.empty:
        return None

    if AD_bt.get("ensemble") and AD_bt.get("constituents"):
        yhi = np.mean([con["m_hi"].predict(F) for con in AD_bt["constituents"]], axis=0)
        ylo = np.mean([con["m_lo"].predict(F) for con in AD_bt["constituents"]], axis=0)
        if AD_bt.get("blended") and float(AD_bt.get("alpha", 1.0)) < 1.0:
            a = float(AD_bt["alpha"])
            yhi = a * yhi + (1 - a) * float(AD_bt.get("mu_hi", 0))
            ylo = a * ylo + (1 - a) * float(AD_bt.get("mu_lo", 0))
    else:
        yhi = AD_bt["hi_model"].predict(F)
        ylo = AD_bt["lo_model"].predict(F)

    c_vals = c.reindex(F.index).values
    ph = c_vals * (1 + np.clip(yhi, 0, None))
    pl = c_vals * (1 - np.clip(ylo, 0, None))
    idx_arr = np.asarray(F.index, dtype="datetime64[ns]")
    nd = np.empty(len(F), dtype="datetime64[ns]")
    nd[:-1] = idx_arr[1:]; nd[-1] = idx_arr[-1] + np.timedelta64(1, "D")

    preds_df = pd.DataFrame(
        {"close_asof": c_vals, "pred_high": ph, "pred_low": pl},
        index=pd.DatetimeIndex(nd, name="target_date"),
    )
    preds_df = preds_df[~preds_df.index.duplicated(keep="last")]
    return preds_df, raw_df


# ──────────────────────────────────────────────────────────────────────────
# Signal construction + MSTU loop (replica of run_mstu_backtest)
# stop_mode ∈ {"baseline", "mstr_stop", "combined"}
# ──────────────────────────────────────────────────────────────────────────
def run_mstu_backtest(end_date_iso: str,
                      backtest_start_iso: str,
                      initial_capital: float = 100_000.0,
                      entry_gate: str = "bull_regime",
                      stop_mode: str = "baseline"):
    WARMUP = 35
    end_dt   = pd.Timestamp(end_date_iso)
    start_dt = pd.Timestamp(backtest_start_iso)
    pre_dt   = start_dt - pd.Timedelta(days=60)
    fetch_start = (start_dt - pd.Timedelta(days=200)).strftime("%Y-%m-%d")
    fetch_end   = (end_dt + pd.Timedelta(days=3)).strftime("%Y-%m-%d")

    ext = build_backtest_preds(fetch_start, fetch_end)
    if ext is None:
        return None
    preds, raw_df = ext

    btc_closes = raw_df["btc_close"]; btc_highs = raw_df["btc_high"]; btc_lows = raw_df["btc_low"]
    preds = preds.loc[(preds.index >= pre_dt) & (preds.index <= end_dt)].copy()
    if len(preds) < WARMUP + 3:
        return None
    preds["actual_high"]  = btc_highs.reindex(preds.index).values
    preds["actual_low"]   = btc_lows.reindex(preds.index).values
    preds["actual_close"] = btc_closes.reindex(preds.index).values
    comp = preds.dropna(subset=["actual_high", "actual_low", "actual_close"]).reset_index()
    N = len(comp)
    if N < WARMUP + 3:
        return None
    dates = pd.DatetimeIndex(comp["target_date"])
    _bt0  = max(WARMUP, int(dates.searchsorted(start_dt)))
    if N - _bt0 < 3:
        return None

    # MSTU execution prices (synthetic CSV — covers full range incl. pre-inception)
    _mstu_syn_csv = _load_mstu_synthetic()
    mstu_px = _mstu_syn_csv.reindex(dates).ffill().bfill().values.astype(float)

    # MSTR prices for the correlation-driven stop (same source/alignment as run_mstr_backtest)
    _mstr_csv = _load_mstr_prices()
    _mstr_close = _mstr_csv["close"].sort_index()
    _mstr_all = _mstr_close.reindex(
        pd.date_range(_mstr_close.index[0], max(_mstr_close.index[-1], end_dt), freq="D")
    )
    mstr_px = _mstr_all.reindex(dates).ffill().bfill().values.astype(float)

    # ── BTC signal arrays (identical to run_mstu_backtest) ──
    c_asof  = comp["close_asof"].values.astype(float)
    pred_hi = comp["pred_high"].values.astype(float)
    pred_lo = comp["pred_low"].values.astype(float)
    act_hi  = comp["actual_high"].values.astype(float)
    act_lo  = comp["actual_low"].values.astype(float)
    err_hi = (act_hi - pred_hi) / c_asof * 100
    err_lo = (pred_lo - act_lo) / c_asof * 100
    hi_brk = (act_hi > pred_hi).astype(int)
    lo_brk = (act_lo < pred_lo).astype(int)

    ehma3 = np.zeros(N); elma3 = np.zeros(N)
    hb3   = np.zeros(N, dtype=int); lb3 = np.zeros(N, dtype=int)
    for i in range(N):
        s = max(0, i - 2)
        ehma3[i] = np.mean(err_hi[s:i + 1]); elma3[i] = np.mean(err_lo[s:i + 1])
        hb3[i]   = int(np.sum(hi_brk[s:i + 1])); lb3[i] = int(np.sum(lo_brk[s:i + 1]))

    u1 = (ehma3 > 0.7) & (hb3 >= 2)
    d1 = (lb3 >= 2) & (elma3 > 0.5)
    d2 = ehma3 < -0.75
    d3 = np.zeros(N, dtype=bool)
    for i in range(1, N):
        consec = 0
        for k in range(i - 1, -1, -1):
            if hi_brk[k]: consec += 1
            else: break
        if consec >= 3 and lo_brk[i]:
            d3[i] = True

    ma30 = np.full(N, np.nan)
    for i in range(N):
        w = min(30, i + 1)
        ma30[i] = np.mean(c_asof[max(0, i - w + 1):i + 1])
    above_ma30     = c_asof > ma30
    ma30_slope_pos = np.zeros(N, dtype=bool)
    for i in range(N):
        if i >= 5 and np.isfinite(ma30[i]) and np.isfinite(ma30[i - 5]):
            ma30_slope_pos[i] = ma30[i] > ma30[i - 5]
    bull_regime = above_ma30 & ma30_slope_pos

    clean_10d = np.zeros(N, dtype=bool)
    for i in range(N):
        lo_i = max(0, i - 7)
        clean_10d[i] = not bool(np.any(d1[lo_i:i] | d2[lo_i:i]))

    _DN_NORM_W = 30
    roll_ehi_norm = np.array([
        float(np.mean(err_hi[max(0, i - _DN_NORM_W + 1):i + 1])) for i in range(N)
    ])
    dn_score_arr = np.zeros(N)
    for i in range(N):
        norm = max(abs(roll_ehi_norm[i]), 0.01)
        dn_score_arr[i] = (
            (-ehma3[i] / norm) * 0.30 +
            (lb3[i] / 3.0) * 0.30 +
            (elma3[i] / max(abs(elma3[i]), 0.10)) * 0.20 +
            float(lo_brk[i]) * 0.20
        )
    v_rev_bar = (dn_score_arr > 0.8) & (err_lo > 3.0)
    v_recent  = np.zeros(N, dtype=bool)
    for i in range(N):
        v_recent[i] = bool(np.any(v_rev_bar[max(0, i - 2):i + 1]))

    if entry_gate == "above_ma30":
        tf1_entry = u1 & ((above_ma30 ^ clean_10d) | v_recent)
    elif entry_gate == "bull_regime":
        tf1_entry = u1 & ((bull_regime ^ clean_10d) | v_recent)
    else:  # pure_regime
        tf1_entry = u1 & (bull_regime | (clean_10d & ~above_ma30) | v_recent)

    # ── Backtest loop ──
    nav = initial_capital; pos = "CASH"; mstu_qty = 0.0
    e_price = e_nav = e_date = e_trigger = None
    e_reentry = False
    stop_px = 0.0          # MSTU 7% stop level
    mstr_stop_px = 0.0     # MSTR 3% stop level (in MSTR price terms)
    from_sl = False; bars_since_sl = 0
    trades = []; nav_arr = np.full(N, np.nan)

    for i in range(N):
        price = mstu_px[i]
        if i < _bt0:
            nav_arr[i] = initial_capital; continue
        if not np.isfinite(price) or price <= 0:
            nav_arr[i] = mstu_qty * mstu_px[i - 1] if pos == "LONG" and i > 0 else nav
            continue
        if pos == "LONG":
            cur = mstu_qty * price
            # ── stop trigger depends on stop_mode ──
            mstu_hit = price <= stop_px
            mstr_hit = np.isfinite(mstr_px[i]) and mstr_px[i] <= mstr_stop_px
            if stop_mode == "baseline":
                stop_hit = mstu_hit; stop_lbl = "SL-fixed-7%"
            elif stop_mode == "mstr_stop":
                stop_hit = mstr_hit; stop_lbl = "SL-MSTR-3%"
            else:  # combined
                stop_hit = mstu_hit or mstr_hit
                stop_lbl = "SL-MSTR-3%" if (mstr_hit and not mstu_hit) else "SL-fixed-7%"
            if stop_hit:
                exit_px = price; nav = mstu_qty * exit_px
                trades.append(dict(
                    entry_date=e_date, entry_price=e_price, entry_nav=e_nav,
                    entry_trigger=e_trigger, exit_date=dates[i], exit_price=exit_px,
                    exit_nav=nav, pnl_pct=(exit_px / e_price - 1) * 100,
                    pnl_abs=nav - e_nav, exit_signal=stop_lbl,
                    duration_days=(dates[i] - e_date).days, stop_triggered=True,
                    was_reentry=e_reentry,
                ))
                pos = "CASH"; mstu_qty = 0.0; stop_px = 0.0; mstr_stop_px = 0.0
                e_reentry = False; from_sl = True; bars_since_sl = 0
            else:
                should_exit = bool(d3[i] or (d2[i] and not bull_regime[i]))
                exit_lbl = "D3" if d3[i] else "D2 (bear)"
                if should_exit:
                    nav = cur
                    trades.append(dict(
                        entry_date=e_date, entry_price=e_price, entry_nav=e_nav,
                        entry_trigger=e_trigger, exit_date=dates[i], exit_price=price,
                        exit_nav=nav, pnl_pct=(price / e_price - 1) * 100,
                        pnl_abs=nav - e_nav, exit_signal=exit_lbl,
                        duration_days=(dates[i] - e_date).days, stop_triggered=False,
                        was_reentry=e_reentry,
                    ))
                    pos = "CASH"; mstu_qty = 0.0; stop_px = 0.0; mstr_stop_px = 0.0
                    e_reentry = False; from_sl = False
                else:
                    nav = cur
        else:
            if from_sl:
                bars_since_sl += 1
            _exit_at_i = d3[i] or (d2[i] and not bull_regime[i])
            _sl_reentry_ok = (not from_sl or bool(bull_regime[i]) or bars_since_sl >= 10)
            if tf1_entry[i] and _sl_reentry_ok and (from_sl or not _exit_at_i):
                e_reentry = bool(from_sl)
                mstu_qty = nav / price; e_price = price; e_date = dates[i]
                e_nav = nav; pos = "LONG"
                stop_px = price * 0.93
                mstr_stop_px = mstr_px[i] * 0.97
                from_sl = False; bars_since_sl = 0
                if v_recent[i]:        e_trigger = "U1 + V-reversal"
                elif above_ma30[i]:    e_trigger = "U1 + ↑MA30"
                else:                  e_trigger = "U1 + Clean 7d"
        nav_arr[i] = mstu_qty * price if pos == "LONG" else nav

    if pos == "LONG" and np.isfinite(mstu_px[N - 1]) and mstu_px[N - 1] > 0:
        nav_arr[N - 1] = mstu_qty * mstu_px[N - 1]

    nav_series = pd.Series(nav_arr[_bt0:], index=dates[_bt0:]).ffill()
    bh_px0 = mstu_px[_bt0]
    bh_series = pd.Series(initial_capital * mstu_px[_bt0:] / bh_px0, index=dates[_bt0:])

    final_nav = float(nav_series.iloc[-1]); final_bh = float(bh_series.iloc[-1])
    strat_ret = (final_nav / initial_capital - 1) * 100
    bh_ret    = (final_bh / initial_capital - 1) * 100
    wins   = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    win_rate = 100 * len(wins) / len(trades) if trades else 0.0
    avg_pnl  = float(np.mean([t["pnl_pct"] for t in trades])) if trades else 0.0
    best_t   = float(max([t["pnl_pct"] for t in trades])) if trades else 0.0
    worst_t  = float(min([t["pnl_pct"] for t in trades])) if trades else 0.0
    rm = nav_series.cummax(); max_dd = float(((nav_series - rm) / rm * 100).min())
    rm_bh = bh_series.cummax(); bh_max_dd = float(((bh_series - rm_bh) / rm_bh * 100).min())
    days_in = sum(t["duration_days"] for t in trades)
    tot_days = max(1, (dates[N - 1] - dates[_bt0]).days)
    rf_daily = (1.045) ** (1 / 252) - 1
    dr_s = nav_series.pct_change().fillna(0); dr_b = bh_series.pct_change().fillna(0)
    exc_s = dr_s - rf_daily; exc_b = dr_b - rf_daily
    sharpe = float(exc_s.mean() / exc_s.std() * np.sqrt(252)) if exc_s.std() > 0 else 0.0
    bh_sharpe = float(exc_b.mean() / exc_b.std() * np.sqrt(252)) if exc_b.std() > 0 else 0.0
    TAX_RATE = 0.35
    _annual_net: dict = {}
    for t in trades:
        yr = pd.Timestamp(t["exit_date"]).year
        _annual_net[yr] = _annual_net.get(yr, 0.0) + t["pnl_abs"]
    total_tax_paid = sum(TAX_RATE * max(0.0, net) for net in _annual_net.values())
    after_tax_nav = final_nav - total_tax_paid
    after_tax_ret = (after_tax_nav / initial_capital - 1) * 100

    return dict(
        stop_mode=stop_mode, entry_gate=entry_gate, trades=trades,
        final_nav=final_nav, final_bh=final_bh, strat_ret=strat_ret, bh_ret=bh_ret,
        alpha_abs=final_nav - final_bh, n_trades=len(trades),
        n_stop_exits=sum(1 for t in trades if t.get("stop_triggered")),
        n_reentries=sum(1 for t in trades if t.get("was_reentry")),
        n_wins=len(wins), n_losses=len(losses), win_rate=win_rate, avg_pnl=avg_pnl,
        best_trade=best_t, worst_trade=worst_t, max_drawdown=max_dd, bh_max_dd=bh_max_dd,
        sharpe=sharpe, bh_sharpe=bh_sharpe, time_in_mkt=100 * days_in / tot_days,
        after_tax_ret=after_tax_ret, start_date=dates[_bt0], end_date=dates[N - 1],
    )


# ──────────────────────────────────────────────────────────────────────────
PERIODS = [
    ("Bear  (Jun25–May26)", "2026-05-31", "2025-06-04"),
    ("Bull  (Jun24–Jun25)", "2025-06-14", "2024-06-01"),
    ("Full  (Jun24–May26)", "2026-05-31", "2024-06-01"),
]
MODES = ["baseline", "mstr_stop", "combined"]
MODE_LABEL = {"baseline": "MSTU 7% (current)",
              "mstr_stop": "MSTR 3% trigger",
              "combined": "first of either"}


def main():
    gate = sys.argv[1] if len(sys.argv) > 1 else "bull_regime"
    print(f"\nMSTU stop-loss source comparison · entry_gate={gate}")
    print("Dataset: data/backtest/ v2 (2026-06-15)  ·  exit fill = MSTU close\n")
    hdr = (f"{'Period':<22}{'Stop source':<20}{'Strat%':>9}{'B&H%':>9}"
           f"{'Alpha$':>11}{'MaxDD%':>9}{'Sharpe':>8}{'Trades':>7}{'Stops':>6}"
           f"{'Win%':>7}{'AfterTax%':>10}")
    print(hdr); print("-" * len(hdr))
    for label, end_iso, start_iso in PERIODS:
        rows = {}
        for mode in MODES:
            r = run_mstu_backtest(end_iso, start_iso, entry_gate=gate, stop_mode=mode)
            rows[mode] = r
            if r is None:
                print(f"{label:<22}{MODE_LABEL[mode]:<20}  (no result)")
                continue
            print(f"{label:<22}{MODE_LABEL[mode]:<20}"
                  f"{r['strat_ret']:>9.1f}{r['bh_ret']:>9.1f}{r['alpha_abs']:>11,.0f}"
                  f"{r['max_drawdown']:>9.1f}{r['sharpe']:>8.2f}{r['n_trades']:>7d}"
                  f"{r['n_stop_exits']:>6d}{r['win_rate']:>7.1f}{r['after_tax_ret']:>10.1f}")
        print("-" * len(hdr))


if __name__ == "__main__":
    main()
