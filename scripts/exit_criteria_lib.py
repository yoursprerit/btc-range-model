"""Standalone replication of the UI's TF2+V-Gate backtest pipeline.

This module reproduces — bit for bit — the prediction build, signal computation,
and per-asset backtest loops from ``app/btc_hourly_app.py`` (run_btc_backtest,
run_mstr_backtest, run_mstu_backtest), but factors the EXIT decision out into a
pluggable callable so alternative exit criteria can be swept and compared on
identical data and methodology.

Data source: data/backtest/ versioned CSVs (the same files the UI loads when a
snapshot is present, i.e. data_mtime > 0).  Model: models/inference_assets_ct.joblib.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from paths import DAILY_MODEL_CT  # noqa: E402

_BT = ROOT / "data" / "backtest"

SATA_ANNUAL_RATE   = 0.13
SATA_BUSINESS_DAYS = 250
SATA_DAILY_FACTOR  = 1.0 + SATA_ANNUAL_RATE / SATA_BUSINESS_DAYS

WARMUP = 35


# ──────────────────────────────────────────────────────────────────────────
# Data loaders (mirror app helpers)
# ──────────────────────────────────────────────────────────────────────────
def _read_price_csv(filename: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(_BT / filename, index_col=0, parse_dates=True)
        df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception:
        return None


def _load_raw_features() -> pd.DataFrame:
    df = pd.read_csv(_BT / "raw_features_daily.csv", index_col=0, parse_dates=True)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    return df


# ──────────────────────────────────────────────────────────────────────────
# Prediction build (CSV path of _build_backtest_preds) — feature engineering
# is copied verbatim from app/btc_hourly_app.py.
# ──────────────────────────────────────────────────────────────────────────
_AD = joblib.load(str(DAILY_MODEL_CT))


def build_backtest_preds(fetch_start_iso: str, fetch_end_iso: str):
    AD_bt = _AD
    _fs = pd.Timestamp(fetch_start_iso); _fe = pd.Timestamp(fetch_end_iso)
    _rf = _load_raw_features()
    _rf_slice = _rf.loc[(_rf.index >= _fs) & (_rf.index <= _fe)]
    if len(_rf_slice) < 35:
        return None
    df = _rf_slice.copy()
    raw_df = df[["btc_close", "btc_high", "btc_low", "btc_volume"]].copy()

    c   = df["btc_close"]; h = df["btc_high"]
    l_  = df["btc_low"];   v = df["btc_volume"]
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
        if _cb_col in df.columns:
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
# Signal computation (verbatim from the app backtest loops)
# ──────────────────────────────────────────────────────────────────────────
def compute_signals(comp: pd.DataFrame):
    N = len(comp)
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
    hb3 = np.zeros(N, dtype=int); lb3 = np.zeros(N, dtype=int)
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
        if consec >= 3 and lo_brk[i]:
            d3[i] = True

    ma30 = np.full(N, np.nan)
    for i in range(N):
        w = min(30, i + 1)
        ma30[i] = np.mean(c_asof[max(0, i - w + 1):i + 1])
    above_ma30 = c_asof > ma30
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
    v_recent = np.zeros(N, dtype=bool)
    for i in range(N):
        v_recent[i] = bool(np.any(v_rev_bar[max(0, i - 2):i + 1]))

    return dict(
        N=N, err_hi=err_hi, err_lo=err_lo, hi_brk=hi_brk, lo_brk=lo_brk,
        ehma3=ehma3, elma3=elma3, hb3=hb3, lb3=lb3,
        u1=u1, d1=d1, d2=d2, d3=d3, ma30=ma30,
        above_ma30=above_ma30, ma30_slope_pos=ma30_slope_pos,
        bull_regime=bull_regime, clean_10d=clean_10d, v_recent=v_recent,
    )


def entry_signal(sig, entry_gate="bull_regime"):
    u1 = sig["u1"]; above_ma30 = sig["above_ma30"]; clean_10d = sig["clean_10d"]
    bull_regime = sig["bull_regime"]; v_recent = sig["v_recent"]
    if entry_gate == "above_ma30":
        return u1 & ((above_ma30 ^ clean_10d) | v_recent)
    elif entry_gate == "bull_regime":
        return u1 & ((bull_regime ^ clean_10d) | v_recent)
    else:  # pure_regime
        return u1 & (bull_regime | (clean_10d & ~above_ma30) | v_recent)


# ──────────────────────────────────────────────────────────────────────────
# Generic backtest loop with pluggable exit logic
# ──────────────────────────────────────────────────────────────────────────
def load_asset_prices(asset, dates, raw_df, comp):
    """Return (exec_px, intraday_low) arrays aligned to `dates`."""
    if asset == "BTC":
        _btc = _read_price_csv("btc_usd_daily.csv")
        closes = _btc["close"]; lows = _btc["low"]
        idx = pd.DatetimeIndex(dates)
        px = closes.reindex(idx).values.astype(float)
        lo = lows.reindex(idx).values.astype(float)
        return px, lo
    if asset == "MSTR":
        _m = _read_price_csv("mstr_daily.csv")
        close = _m["close"].sort_index(); low = _m["low"].sort_index()
        _all = close.reindex(pd.date_range(close.index[0], max(close.index[-1], dates[-1]), freq="D")).ffill()
        _lo_all = low.reindex(pd.date_range(low.index[0], max(low.index[-1], dates[-1]), freq="D")).ffill()
        px = _all.reindex(dates).ffill().bfill().values.astype(float)
        lo = _lo_all.reindex(dates).ffill().bfill().values.astype(float)
        return px, lo
    if asset == "MSTU":
        _syn = _read_price_csv("mstu_synthetic_daily.csv")["close"]
        px = _syn.reindex(dates).ffill().bfill().values.astype(float)
        _act = _read_price_csv("mstu_daily.csv")
        if _act is not None and "low" in _act.columns:
            _lo_raw = _act["low"].sort_index()
            _lo_all = _lo_raw.reindex(pd.date_range(_lo_raw.index[0], max(_lo_raw.index[-1], dates[-1]), freq="D")).ffill()
            lo = _lo_all.reindex(dates).ffill().bfill().values.astype(float)
        else:
            lo = px.copy()
        return px, lo
    raise ValueError(asset)


def run_backtest(asset, start_iso, end_iso, exit_fn, *,
                 entry_gate="bull_regime", stop_kind=None, stop_pct=None,
                 trail_pct=None, sl5_reentry=True, initial_capital=100_000.0):
    """Generic backtest.

    exit_fn(ctx, i) -> (should_exit: bool, label: str)  — signal-based exit.
    stop_kind: None | 'fixed' | 'trail'. stop_pct / trail_pct are fractions (e.g. 0.03).
    Stops trigger & fill on close (matching the live methodology).
    sl5_reentry: apply SL5 regime-adaptive re-entry gate after a stop exit.
    """
    end_dt = pd.Timestamp(end_iso); start_dt = pd.Timestamp(start_iso)
    pre_dt = start_dt - pd.Timedelta(days=60)
    fetch_start = (start_dt - pd.Timedelta(days=200)).strftime("%Y-%m-%d")
    fetch_end = (end_dt + pd.Timedelta(days=3)).strftime("%Y-%m-%d")

    ext = build_backtest_preds(fetch_start, fetch_end)
    if ext is None: return None
    preds, raw_df = ext
    btc_closes = raw_df["btc_close"]; btc_highs = raw_df["btc_high"]; btc_lows = raw_df["btc_low"]
    preds = preds.loc[(preds.index >= pre_dt) & (preds.index <= end_dt)].copy()
    preds["actual_high"] = btc_highs.reindex(preds.index).values
    preds["actual_low"]  = btc_lows.reindex(preds.index).values
    preds["actual_close"] = btc_closes.reindex(preds.index).values
    comp = preds.dropna(subset=["actual_high", "actual_low", "actual_close"]).reset_index()
    N = len(comp)
    if N < WARMUP + 3: return None
    dates = pd.DatetimeIndex(comp["target_date"])
    _bt0 = max(WARMUP, int(dates.searchsorted(start_dt)))
    if N - _bt0 < 3: return None

    sig = compute_signals(comp)
    tf1_entry = entry_signal(sig, entry_gate)
    bull_regime = sig["bull_regime"]; above_ma30 = sig["above_ma30"]; v_recent = sig["v_recent"]
    d2 = sig["d2"]; d3 = sig["d3"]

    if asset == "BTC":
        # BTC uses versioned actual_close as execution price (matches run_btc_backtest)
        px = comp["actual_close"].values.astype(float)
        _btc = _read_price_csv("btc_usd_daily.csv")
        if _btc is not None and "close" in _btc.columns:
            px = _btc["close"].reindex(dates).values.astype(float)
            lo = _btc["low"].reindex(dates).values.astype(float) if "low" in _btc.columns else btc_lows.reindex(dates).values.astype(float)
        else:
            lo = btc_lows.reindex(dates).values.astype(float)
    else:
        px, lo = load_asset_prices(asset, dates, raw_df, comp)

    ctx = dict(d2=d2, d3=d3, bull_regime=bull_regime, above_ma30=above_ma30,
               v_recent=v_recent, sig=sig, px=px, lo=lo, dates=dates)

    nav = initial_capital; pos = "CASH"; qty = 0.0
    e_price = e_nav = e_date = None
    stop_px = 0.0; peak = 0.0
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
            stopped = False
            if stop_kind == "trail":
                peak = max(peak, price)
                eff_stop = peak * (1 - trail_pct)
                if price <= eff_stop: stopped = True
            elif stop_kind == "fixed":
                if price <= stop_px: stopped = True
            if stopped:
                nav = qty * price
                trades.append(dict(entry_date=e_date, entry_price=e_price, exit_date=dates[i],
                                   exit_price=price, pnl_pct=(price / e_price - 1) * 100,
                                   pnl_abs=nav - e_nav, exit_signal="STOP",
                                   duration_days=(dates[i] - e_date).days, stop_triggered=True))
                pos = "CASH"; qty = 0.0; stop_px = 0.0; peak = 0.0
                from_sl = True; bars_since_sl = 0
            else:
                should_exit, lbl = exit_fn(ctx, i)
                if should_exit:
                    nav = cur
                    trades.append(dict(entry_date=e_date, entry_price=e_price, exit_date=dates[i],
                                       exit_price=price, pnl_pct=(price / e_price - 1) * 100,
                                       pnl_abs=nav - e_nav, exit_signal=lbl,
                                       duration_days=(dates[i] - e_date).days, stop_triggered=False))
                    pos = "CASH"; qty = 0.0; stop_px = 0.0; peak = 0.0
                    from_sl = False
                else:
                    nav = cur
        else:
            if from_sl: bars_since_sl += 1
            should_exit_i, _ = exit_fn(ctx, i)
            _sl_ok = (not from_sl or bool(bull_regime[i]) or bars_since_sl >= 10) if sl5_reentry else True
            if tf1_entry[i] and _sl_ok and (from_sl or not should_exit_i):
                qty = nav / price; e_price = price; e_date = dates[i]; e_nav = nav; pos = "LONG"
                peak = price
                if stop_kind == "fixed": stop_px = price * (1 - stop_pct)
                from_sl = False; bars_since_sl = 0
        nav_arr[i] = qty * price if pos == "LONG" else nav

    if pos == "LONG" and np.isfinite(px[N - 1]) and px[N - 1] > 0:
        nav_arr[N - 1] = qty * px[N - 1]

    nav_series = pd.Series(nav_arr[_bt0:], index=dates[_bt0:]).ffill()
    bh_series = pd.Series(initial_capital * px[_bt0:] / px[_bt0], index=dates[_bt0:])
    final_nav = float(nav_series.iloc[-1]); final_bh = float(bh_series.iloc[-1])
    rm = nav_series.cummax(); max_dd = float(((nav_series - rm) / rm * 100).min())
    rf_daily = (1.045) ** (1 / 252) - 1
    dr_s = nav_series.pct_change().fillna(0); exc_s = dr_s - rf_daily
    sharpe = float(exc_s.mean() / exc_s.std() * np.sqrt(252)) if exc_s.std() > 0 else 0.0
    wins = [t for t in trades if t["pnl_pct"] > 0]
    days_in = sum(t["duration_days"] for t in trades)
    tot_days = max(1, (dates[N - 1] - dates[_bt0]).days)
    return dict(
        final_nav=final_nav, strat_ret=(final_nav / initial_capital - 1) * 100,
        bh_ret=(final_bh / initial_capital - 1) * 100, alpha_abs=final_nav - final_bh,
        max_dd=max_dd, sharpe=sharpe, n_trades=len(trades),
        win_rate=100 * len(wins) / len(trades) if trades else 0.0,
        time_in_mkt=100 * days_in / tot_days, trades=trades,
        nav_series=nav_series, bh_series=bh_series,
    )


# ── Exit function library ─────────────────────────────────────────────────
def exit_tf2(ctx, i):
    """Live exit: D3 always, D2 only in bear/neutral regime."""
    if ctx["d3"][i]:
        return True, "D3"
    if ctx["d2"][i] and not ctx["bull_regime"][i]:
        return True, "D2 (bear)"
    return False, ""


def exit_tf1(ctx, i):
    """Fixed D2|D3 exit (no regime patience)."""
    if ctx["d3"][i]: return True, "D3"
    if ctx["d2"][i]: return True, "D2"
    return False, ""


def exit_d3_only(ctx, i):
    """Patient: exit only on D3 structural reversal."""
    if ctx["d3"][i]: return True, "D3"
    return False, ""


def exit_with_d1(ctx, i):
    """TF2 plus D1 as an additional bear-regime exit."""
    if ctx["d3"][i]: return True, "D3"
    if not ctx["bull_regime"][i] and (ctx["d2"][i] or ctx["sig"]["d1"][i]):
        return True, "D1/D2 (bear)"
    return False, ""


def make_exit_d2_confirmed():
    """TF2 but require D2 on two consecutive bars (bear) before exiting."""
    def _f(ctx, i):
        if ctx["d3"][i]: return True, "D3"
        d2 = ctx["d2"]; br = ctx["bull_regime"]
        if i >= 1 and d2[i] and d2[i - 1] and not br[i]:
            return True, "D2x2 (bear)"
        return False, ""
    return _f
