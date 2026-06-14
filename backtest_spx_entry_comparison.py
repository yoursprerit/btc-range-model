"""
SPX-Enhanced Entry Signal Comparison Backtest
==============================================
Compares current TF2+V-Gate entry signal vs SPX-filtered entry for MSTR/MSTU.

  Current:  tf1_entry = u1 & ((above_ma30 ^ clean_10d) | v_recent)
  New:      mstr_entry = tf1_entry & (spx_slope_pos | v_recent)
            where:
              spx_ma20      = pd.Series(spx_close).rolling(20).mean()
              spx_slope_pos = spx_ma20 > spx_ma20.shift(5)

Run:  python backtest_spx_entry_comparison.py
"""
import sys, os, warnings, time
warnings.filterwarnings("ignore")
from pathlib import Path
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

try:
    import sklearn._loss._loss as _sklearn_loss_ext
    if "_loss" not in sys.modules:
        sys.modules["_loss"] = _sklearn_loss_ext
    import sklearn._loss.loss    # noqa: F401
    import sklearn._loss.link    # noqa: F401
except Exception:
    pass

import numpy as np
import pandas as pd
import joblib
import requests
import yfinance as yf

from paths import DAILY_MODEL_CT

# ── Config ────────────────────────────────────────────────────────────────────

INITIAL_CAPITAL = 100_000.0
WARMUP          = 35
TODAY           = pd.Timestamp.now().normalize().strftime("%Y-%m-%d")

PERIODS = {
    "Bear Market (Jun 2025–May 2026)": ("2025-06-01", "2026-05-31"),
    "Bull Market (Jun 2024–Jun 2025)": ("2024-06-05", "2025-06-14"),
    "Full Market (Jun 2024–May 2026)": ("2024-06-01", "2026-05-31"),
    f"OOS Only   (Mar 2026–{TODAY})":  ("2026-03-01", TODAY),
}

_MACRO = {
    "eth":  "ETH-USD",
    "spx":  "^GSPC",
    "ndx":  "^IXIC",
    "vix":  "^VIX",
    "gold": "GC=F",
    "dxy":  "DX-Y.NYB",
    "tnx":  "^TNX",
}

_ONCHAIN = [
    "hash-rate", "difficulty", "n-transactions", "miners-revenue",
    "n-unique-addresses", "transaction-fees-usd", "mempool-size",
    "estimated-transaction-volume-usd", "market-cap",
    "avg-block-size", "cost-per-transaction",
]

# ── Data fetch & prediction ───────────────────────────────────────────────────

def _yf_dl(sym, start, end):
    d = yf.download(sym, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] for c in d.columns]
    d.index = pd.DatetimeIndex(d.index).tz_localize(None).normalize()
    return d


def build_preds_with_spx(fetch_start: str, fetch_end: str):
    """Build CT predictions + SPX series (yfinance daily data, matching UI)."""
    print(f"  Loading model …", end=" ", flush=True)
    try:
        AD = joblib.load(str(DAILY_MODEL_CT))
    except Exception as e:
        print(f"FAILED: {e}"); return None
    print("OK")

    print(f"  BTC-USD {fetch_start}→{fetch_end} …", end=" ", flush=True)
    try:
        d_btc = _yf_dl("BTC-USD", fetch_start, fetch_end)
    except Exception as e:
        print(f"FAILED: {e}"); return None
    df = pd.DataFrame({
        "btc_close":  d_btc["Close"],
        "btc_high":   d_btc["High"],
        "btc_low":    d_btc["Low"],
        "btc_volume": d_btc.get("Volume", pd.Series(dtype=float)),
    })
    raw_df = df[["btc_close","btc_high","btc_low","btc_volume"]].copy()
    print(f"{len(df)} bars")

    print("  Macro …", end=" ", flush=True)
    for nm, sym in _MACRO.items():
        try:
            _d = _yf_dl(sym, fetch_start, fetch_end)
            df[f"{nm}_close"] = _d["Close"].reindex(df.index).ffill(limit=7)
        except Exception:
            pass
    print("OK")

    print("  On-chain …", end=" ", flush=True)
    ok = 0
    for m in _ONCHAIN:
        col = f"oc_{m.replace('-','_')}"
        try:
            r = requests.get(
                f"https://api.blockchain.info/charts/{m}",
                params={"timespan":"3years","format":"json","sampled":"true"},
                timeout=20)
            vals = r.json().get("values", [])
            s = pd.Series(
                {pd.Timestamp(v["x"], unit="s").normalize(): v["y"] for v in vals},
                name=col, dtype=float)
            s = s[~s.index.duplicated(keep="last")].sort_index()
            s.index = pd.DatetimeIndex(s.index).tz_localize(None)
            df[col] = s.reindex(df.index).ffill(limit=7)
            ok += 1
        except Exception:
            pass
    print(f"{ok}/{len(_ONCHAIN)} OK")

    # Coinbase premium (optional; zero-fill on failure to keep features consistent)
    df["cb_premium"] = df["cb_premium_ma3"] = df["cb_premium_z7"] = 0.0
    _CB_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    _cb_rows: list = []
    _cb_cur = pd.Timestamp(fetch_start); _cb_end = pd.Timestamp(fetch_end)
    while _cb_cur <= _cb_end:
        _cc = min(_cb_cur + pd.Timedelta(days=299), _cb_end)
        try:
            _r2 = requests.get(_CB_URL, params={
                "granularity": 86400,
                "start": _cb_cur.strftime("%Y-%m-%dT00:00:00Z"),
                "end": (_cc + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z"),
            }, timeout=30)
            if _r2.status_code == 200:
                _cb_rows.extend(_r2.json())
        except Exception:
            pass
        _cb_cur = _cc + pd.Timedelta(days=1)
        time.sleep(0.05)
    if _cb_rows:
        _cb_df = pd.DataFrame(_cb_rows, columns=["ts","low","high","open","close","volume"])
        _cb_df["date"] = pd.to_datetime(_cb_df["ts"], unit="s").dt.normalize()
        _cb_df = _cb_df.drop_duplicates("date").set_index("date").sort_index()
        _cb_close = _cb_df["close"].reindex(df.index).astype(float)
        c_ref = df["btc_close"]
        _prem = (_cb_close - c_ref) / c_ref * 100
        df["cb_premium"]     = _prem
        df["cb_premium_ma3"] = _prem.rolling(3).mean()
        df["cb_premium_z7"]  = (_prem - _prem.rolling(7).mean()) / _prem.rolling(7).std()

    # Feature engineering (identical to app's _build_backtest_preds)
    c = df["btc_close"]; h = df["btc_high"]
    l_ = df["btc_low"];  v = df["btc_volume"]
    ret = np.log(c).diff()
    feat = pd.DataFrame(index=df.index)

    for k in [1,3,5,7,14,30]: feat[f"ret_{k}"] = ret.rolling(k).sum()
    for k in [5,10,20,30]:    feat[f"vol_{k}"] = ret.rolling(k).std()
    prev_c = c.shift(1)
    tr = pd.concat([(h-l_),(h-prev_c).abs(),(l_-prev_c).abs()], axis=1).max(axis=1)
    for k in [7,14,30]: feat[f"atr_{k}"] = tr.rolling(k).mean() / c
    feat["range_today"] = (h-l_)/c
    feat["range_ma7"]   = ((h-l_)/c).rolling(7).mean()
    feat["range_ma30"]  = ((h-l_)/c).rolling(30).mean()
    feat["range_std30"] = ((h-l_)/c).rolling(30).std()
    gain = c.diff().clip(lower=0).rolling(14).mean()
    loss = (-c.diff().clip(upper=0)).rolling(14).mean()
    feat["rsi_14"] = 100 - 100/(1 + gain/loss.replace(0, np.nan))
    e12 = c.ewm(span=12,adjust=False).mean(); e26 = c.ewm(span=26,adjust=False).mean()
    macd = e12 - e26
    feat["macd"]      = macd/c
    feat["macd_sig"]  = macd.ewm(span=9,adjust=False).mean()/c
    feat["macd_hist"] = (macd - macd.ewm(span=9,adjust=False).mean())/c
    ma20 = c.rolling(20).mean(); sd20 = c.rolling(20).std()
    feat["bb_width"]   = (4*sd20)/ma20
    feat["dist_hi_30"] = c/c.rolling(30).max() - 1
    feat["dist_lo_30"] = c/c.rolling(30).min() - 1
    feat["dist_hi_90"] = c/c.rolling(90).max() - 1
    feat["vol_chg_1"]    = np.log(v).diff()
    feat["vol_z_20"]     = (np.log(v)-np.log(v).rolling(20).mean())/np.log(v).rolling(20).std()
    feat["vol_ma_ratio"] = v/v.rolling(20).mean()
    dow = df.index.dayofweek
    for i in range(6): feat[f"dow_{i}"] = (dow==i).astype(float)
    for nm in ["spx","ndx","vix","gold","dxy","tnx","eth"]:
        col = f"{nm}_close"
        if col not in df.columns: continue
        s = df[col]; lr = np.log(s).diff()
        for k in [1,5,20]: feat[f"{nm}_ret_{k}"] = lr.rolling(k).sum()
        feat[f"{nm}_vol_20"] = lr.rolling(20).std()
    for corr_nm, corr_col in [("spx","spx_close"),("ndx","ndx_close"),
                               ("gold","gold_close"),("dxy","dxy_close")]:
        if corr_col in df.columns:
            feat[f"btc_{corr_nm}_corr_30"] = ret.rolling(30).corr(np.log(df[corr_col]).diff())
    for col in [x for x in df.columns if x.startswith("oc_")]:
        s = df[col].astype(float); sl = np.log(s.replace(0, np.nan))
        feat[f"{col}_d1"]  = sl.diff(1)
        feat[f"{col}_d7"]  = sl.diff(7)
        feat[f"{col}_z30"] = (sl - sl.rolling(30).mean())/sl.rolling(30).std()
    nh = h.shift(-1); nl = l_.shift(-1)
    y_hi = (nh-c)/c; y_lo = (c-nl)/c
    feat["y_hi_ema3"] = y_hi.shift(1).ewm(span=3,adjust=False).mean()
    feat["y_lo_ema3"] = y_lo.shift(1).ewm(span=3,adjust=False).mean()
    feat["y_hi_ema7"] = y_hi.shift(1).ewm(span=7,adjust=False).mean()
    feat["y_lo_ema7"] = y_lo.shift(1).ewm(span=7,adjust=False).mean()
    p3h = h.shift(1).rolling(3).max(); p3l = l_.shift(1).rolling(3).min()
    feat["above_3d_high"]  = (c > p3h).astype(float)
    feat["below_3d_low"]   = (c < p3l).astype(float)
    feat["bo_strength_up"] = (c/p3h - 1).clip(lower=0)
    feat["bo_strength_dn"] = (1 - c/p3l).clip(lower=0)
    yhl = y_hi.shift(1); yll = y_lo.shift(1)
    feat["y_hi_surprise"] = yhl - yhl.ewm(span=7,adjust=False).mean()
    feat["y_lo_surprise"] = yll - yll.ewm(span=7,adjust=False).mean()
    nr = ret.clip(upper=0)
    feat["dn_vol_5"]  = nr.rolling(5).std()
    feat["dn_vol_20"] = nr.rolling(20).std()
    sma50 = c.rolling(50).mean()
    feat["below_sma50"]    = (c < sma50).astype(float)
    feat["below_sma50_5d"] = feat["below_sma50"].rolling(5).min().fillna(0)
    for _cb_col in ["cb_premium","cb_premium_ma3","cb_premium_z7"]:
        feat[_cb_col] = df[_cb_col].fillna(0.0)

    # Predictions
    print("  Building predictions …", end=" ", flush=True)
    fc = AD["feat_cols"]
    feat = feat.replace([np.inf, -np.inf], np.nan)
    for col in fc:
        if col not in feat.columns: feat[col] = np.nan
    F = feat[fc].dropna()
    if F.empty:
        print("FAILED — no valid feature rows"); return None

    if AD.get("ensemble") and AD.get("constituents"):
        yhi = np.mean([con["m_hi"].predict(F) for con in AD["constituents"]], axis=0)
        ylo = np.mean([con["m_lo"].predict(F) for con in AD["constituents"]], axis=0)
        if AD.get("blended") and float(AD.get("alpha", 1.0)) < 1.0:
            a = float(AD["alpha"])
            yhi = a*yhi + (1-a)*float(AD.get("mu_hi", 0))
            ylo = a*ylo + (1-a)*float(AD.get("mu_lo", 0))
    else:
        yhi = AD["hi_model"].predict(F)
        ylo = AD["lo_model"].predict(F)

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
    print(f"OK ({len(preds_df)} rows)")

    spx_close = df["spx_close"].ffill() if "spx_close" in df.columns else pd.Series(dtype=float, index=df.index)
    return preds_df, raw_df, spx_close


# ── Signal computation (shared between both entry variants) ───────────────────

def compute_signals(comp, spx_series):
    """Return dict of numpy arrays for all signals + SPX slope."""
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
    hb3   = np.zeros(N, dtype=int); lb3 = np.zeros(N, dtype=int)
    for i in range(N):
        s = max(0, i-2)
        ehma3[i] = np.mean(err_hi[s:i+1]); elma3[i] = np.mean(err_lo[s:i+1])
        hb3[i]   = int(np.sum(hi_brk[s:i+1])); lb3[i] = int(np.sum(lo_brk[s:i+1]))

    u1 = (ehma3 > 0.7) & (hb3 >= 2)
    d1 = (lb3 >= 2)    & (elma3 > 0.5)
    d2 = ehma3 < -0.75
    d3 = np.zeros(N, dtype=bool)
    for i in range(1, N):
        consec = 0
        for k in range(i-1, -1, -1):
            if hi_brk[k]: consec += 1
            else: break
        if consec >= 3 and lo_brk[i]:
            d3[i] = True

    ma30 = np.full(N, np.nan)
    for i in range(N):
        w = min(30, i+1)
        ma30[i] = np.mean(c_asof[max(0, i-w+1):i+1])
    above_ma30     = c_asof > ma30
    ma30_slope_pos = np.zeros(N, dtype=bool)
    for i in range(N):
        if i >= 5 and np.isfinite(ma30[i]) and np.isfinite(ma30[i-5]):
            ma30_slope_pos[i] = ma30[i] > ma30[i-5]
    bull_regime = above_ma30 & ma30_slope_pos

    clean_10d = np.zeros(N, dtype=bool)
    for i in range(N):
        lo_i = max(0, i-7)
        clean_10d[i] = not bool(np.any(d1[lo_i:i] | d2[lo_i:i]))

    _DN_NORM_W    = 30
    roll_ehi_norm = np.array([
        float(np.mean(err_hi[max(0, i-_DN_NORM_W+1):i+1])) for i in range(N)
    ])
    dn_score_arr = np.zeros(N)
    for i in range(N):
        norm = max(abs(roll_ehi_norm[i]), 0.01)
        dn_score_arr[i] = (
            (-ehma3[i] / norm)                           * 0.30 +
            (lb3[i]    / 3.0)                            * 0.30 +
            (elma3[i]  / max(abs(elma3[i]), 0.10))       * 0.20 +
            float(lo_brk[i])                             * 0.20
        )
    v_rev_bar = (dn_score_arr > 0.8) & (err_lo > 3.0)
    v_recent  = np.zeros(N, dtype=bool)
    for i in range(N):
        v_recent[i] = bool(np.any(v_rev_bar[max(0, i-2):i+1]))

    # Current entry (XOR gate: exactly one of MA30-up or clean_10d, or V-rev)
    tf1_entry = u1 & ((above_ma30 ^ clean_10d) | v_recent)

    # SPX MA20 slope filter
    dates = pd.DatetimeIndex(comp["target_date"])
    if len(spx_series) > 0:
        spx_aligned = spx_series.reindex(dates).ffill().bfill()
        spx_ma20    = spx_aligned.rolling(20).mean()
        spx_slope   = (spx_ma20 > spx_ma20.shift(5)).fillna(False)
        spx_slope_pos = spx_slope.values.astype(bool)
    else:
        spx_slope_pos = np.ones(N, dtype=bool)  # no filter if SPX unavailable

    # New entry: tf1_entry filtered by SPX slope (V-reversal bypasses SPX filter)
    new_entry = tf1_entry & (spx_slope_pos | v_recent)

    return dict(
        u1=u1, d1=d1, d2=d2, d3=d3,
        above_ma30=above_ma30, bull_regime=bull_regime,
        clean_10d=clean_10d, v_recent=v_recent,
        tf1_entry=tf1_entry, new_entry=new_entry,
        spx_slope_pos=spx_slope_pos,
        lo_brk=lo_brk,
    )


# ── Trade log / stats helpers ─────────────────────────────────────────────────

def compute_stats(trades, nav_series, bh_series, initial_capital, start_dt, dates, _bt0):
    wins   = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    win_rate = 100*len(wins)/len(trades) if trades else 0.0
    avg_pnl  = float(np.mean([t["pnl_pct"] for t in trades])) if trades else 0.0
    best_t   = float(max([t["pnl_pct"] for t in trades])) if trades else 0.0
    worst_t  = float(min([t["pnl_pct"] for t in trades])) if trades else 0.0
    rm       = nav_series.cummax()
    max_dd   = float(((nav_series - rm)/rm*100).min())
    rm_bh    = bh_series.cummax()
    bh_max_dd= float(((bh_series - rm_bh)/rm_bh*100).min())
    days_in  = sum(t["duration_days"] for t in trades)
    tot_days = max(1, (dates[-1] - dates[_bt0]).days)
    rf_daily = (1.045)**(1/252) - 1
    dr_s     = nav_series.pct_change().fillna(0)
    dr_b     = bh_series.pct_change().fillna(0)
    exc_s    = dr_s - rf_daily; exc_b = dr_b - rf_daily
    sharpe   = float(exc_s.mean()/exc_s.std()*np.sqrt(252)) if exc_s.std()>0 else 0.0
    bh_sharpe= float(exc_b.mean()/exc_b.std()*np.sqrt(252)) if exc_b.std()>0 else 0.0
    final_nav= float(nav_series.iloc[-1]); final_bh = float(bh_series.iloc[-1])
    strat_ret= (final_nav/initial_capital - 1)*100
    bh_ret   = (final_bh/initial_capital  - 1)*100
    TAX_RATE = 0.35
    _annual_net: dict = {}
    for t in trades:
        yr = pd.Timestamp(t["exit_date"]).year
        _annual_net[yr] = _annual_net.get(yr, 0.0) + t["pnl_abs"]
    total_tax_paid = sum(TAX_RATE * max(0.0, net) for net in _annual_net.values())
    after_tax_nav  = final_nav - total_tax_paid
    after_tax_ret  = (after_tax_nav/initial_capital - 1)*100
    n_sl  = sum(1 for t in trades if t.get("stop_triggered"))
    profit_factor = (
        sum(t["pnl_abs"] for t in wins) / max(1e-9, abs(sum(t["pnl_abs"] for t in losses)))
        if (wins and losses) else (float("inf") if wins else 0.0)
    )
    return dict(
        final_nav=final_nav, final_bh=final_bh,
        strat_ret=strat_ret, bh_ret=bh_ret,
        alpha_abs=final_nav-final_bh,
        n_trades=len(trades), n_wins=len(wins), n_losses=len(losses),
        n_sl=n_sl, win_rate=win_rate, avg_pnl=avg_pnl,
        best_trade=best_t, worst_trade=worst_t,
        max_drawdown=max_dd, bh_max_dd=bh_max_dd,
        sharpe=sharpe, bh_sharpe=bh_sharpe,
        time_in_mkt=100*days_in/tot_days,
        after_tax_nav=after_tax_nav, after_tax_ret=after_tax_ret,
        total_tax_paid=total_tax_paid,
        profit_factor=profit_factor,
    )


# ── Backtest loop ─────────────────────────────────────────────────────────────

def run_backtest(ticker, px_arr, stop_pct, entry_arr, sig, dates, _bt0,
                 initial_capital, label):
    """Generic backtest loop for any ticker price array and entry signal array."""
    N     = len(px_arr)
    d3    = sig["d3"]; d2 = sig["d2"]
    bull  = sig["bull_regime"]
    v_rec = sig["v_recent"]
    above = sig["above_ma30"]
    c10d  = sig["clean_10d"]

    nav     = initial_capital; pos = "CASH"; qty = 0.0
    e_price = e_nav = e_date = e_trigger = None
    e_reentry = False; stop_px = 0.0
    from_sl = False; bars_since_sl = 0
    trades  = []; nav_arr = np.full(N, np.nan)

    for i in range(N):
        price = px_arr[i]
        if i < _bt0:
            nav_arr[i] = initial_capital; continue
        if not np.isfinite(price) or price <= 0:
            nav_arr[i] = qty * px_arr[i-1] if pos == "LONG" and i > 0 else nav
            continue

        if pos == "LONG":
            cur = qty * price
            if price <= stop_px:
                exit_nav = qty * price; nav = exit_nav
                trades.append(dict(
                    entry_date=e_date, entry_price=e_price, entry_nav=e_nav,
                    entry_trigger=e_trigger, exit_date=dates[i], exit_price=price,
                    exit_nav=nav, pnl_pct=(price/e_price-1)*100,
                    pnl_abs=nav-e_nav,
                    exit_signal=f"SL-fixed-{round(stop_pct*100)}%",
                    duration_days=(dates[i]-e_date).days, stop_triggered=True,
                    was_reentry=e_reentry,
                ))
                pos = "CASH"; qty = 0.0; stop_px = 0.0; e_reentry = False
                from_sl = True; bars_since_sl = 0
            else:
                should_exit = bool(d3[i] or (d2[i] and not bull[i]))
                exit_lbl    = "D3" if d3[i] else "D2 (bear)"
                if should_exit:
                    nav = cur
                    trades.append(dict(
                        entry_date=e_date, entry_price=e_price, entry_nav=e_nav,
                        entry_trigger=e_trigger, exit_date=dates[i], exit_price=price,
                        exit_nav=nav, pnl_pct=(price/e_price-1)*100,
                        pnl_abs=nav-e_nav, exit_signal=exit_lbl,
                        duration_days=(dates[i]-e_date).days, stop_triggered=False,
                        was_reentry=e_reentry,
                    ))
                    pos = "CASH"; qty = 0.0; stop_px = 0.0; e_reentry = False
                    from_sl = False
                else:
                    nav = cur
        else:
            if from_sl: bars_since_sl += 1
            _exit_at_i    = d3[i] or (d2[i] and not bull[i])
            _sl_reentry_ok = (not from_sl or bool(bull[i]) or bars_since_sl >= 10)
            if entry_arr[i] and _sl_reentry_ok and (from_sl or not _exit_at_i):
                e_reentry = bool(from_sl)
                qty = nav / price; e_price = price; e_date = dates[i]
                e_nav = nav; pos = "LONG"; stop_px = price * (1 - stop_pct)
                from_sl = False; bars_since_sl = 0
                if v_rec[i]:          e_trigger = "U1 + V-reversal"
                elif above[i]:        e_trigger = "U1 + ↑MA30"
                else:                 e_trigger = "U1 + Clean 7d"
        nav_arr[i] = qty * price if pos == "LONG" else nav

    if pos == "LONG" and np.isfinite(px_arr[N-1]) and px_arr[N-1] > 0:
        nav_arr[N-1] = qty * px_arr[N-1]

    nav_series = pd.Series(nav_arr[_bt0:], index=dates[_bt0:]).ffill()
    bh_series  = pd.Series(initial_capital * px_arr[_bt0:] / px_arr[_bt0], index=dates[_bt0:])
    stats = compute_stats(trades, nav_series, bh_series, initial_capital,
                          dates[_bt0], dates, _bt0)
    return dict(label=label, trades=trades, nav_series=nav_series,
                bh_series=bh_series, stats=stats, open_pos=pos=="LONG")


# ── Equity price fetch ────────────────────────────────────────────────────────

def fetch_equity(ticker, pre_dt, end_dt, dates):
    try:
        d = yf.download(ticker, start=pre_dt.strftime("%Y-%m-%d"),
                        end=(end_dt + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
                        progress=False, auto_adjust=True)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = [c[0] for c in d.columns]
        d.index = pd.DatetimeIndex(d.index).tz_localize(None).normalize()
        px_raw = d["Close"].sort_index()
        # Fill across weekends; bfill for pre-inception (MSTU)
        px_all = px_raw.reindex(
            pd.date_range(px_raw.index[0], max(px_raw.index[-1], end_dt), freq="D")
        ).ffill()
        return px_all.reindex(dates).ffill().bfill().values.astype(float)
    except Exception as e:
        print(f"    WARNING: could not fetch {ticker}: {e}")
        return None


# ── Formatting helpers ────────────────────────────────────────────────────────

W = 14  # column width

def _hdr(cols):
    return " | ".join(f"{c:>{W}}" for c in cols)

def _row(vals):
    return " | ".join(f"{v:>{W}}" for v in vals)

def _sep(n_cols):
    return "-+-".join(["-"*W]*n_cols)

def _fmt_pct(v): return f"{v:+.1f}%"
def _fmt_nav(v): return f"${v:,.0f}"
def _fmt_dd(v):  return f"{v:.1f}%"
def _fmt_sh(v):  return f"{v:.2f}"
def _fmt_pf(v):  return f"{v:.2f}x" if v < 100 else "∞"

def _delta(new, cur, is_pct=False, higher_better=True):
    d = new - cur
    sign = "▲" if d > 0 else ("▼" if d < 0 else "·")
    better = (d > 0 and higher_better) or (d < 0 and not higher_better)
    prefix = "+" if better else ""
    if is_pct:
        return f"{sign}{prefix}{d:+.1f}pp"
    return f"{sign}{d:+.1f}%"


# ── Per-period runner ─────────────────────────────────────────────────────────

def run_period(period_name, start_iso, end_iso, preds_df, raw_df, spx_series):
    print(f"\n  Running period: {period_name}")
    start_dt = pd.Timestamp(start_iso); end_dt = pd.Timestamp(end_iso)
    pre_dt   = start_dt - pd.Timedelta(days=60)
    btc_closes = raw_df["btc_close"]
    btc_highs  = raw_df["btc_high"]
    btc_lows   = raw_df["btc_low"]

    p = preds_df.loc[(preds_df.index >= pre_dt) & (preds_df.index <= end_dt)].copy()
    if len(p) < WARMUP + 3:
        print("    SKIP — insufficient prediction rows"); return None

    p["actual_high"]  = btc_highs.reindex(p.index).values
    p["actual_low"]   = btc_lows.reindex(p.index).values
    p["actual_close"] = btc_closes.reindex(p.index).values
    comp = p.dropna(subset=["actual_high","actual_low","actual_close"]).reset_index()
    N = len(comp)
    if N < WARMUP + 3:
        print("    SKIP — insufficient rows after dropna"); return None

    dates = pd.DatetimeIndex(comp["target_date"])
    _bt0  = max(WARMUP, int(dates.searchsorted(start_dt)))
    if N - _bt0 < 3:
        print("    SKIP — not enough bars after warmup"); return None

    sig = compute_signals(comp, spx_series)

    results = {}
    for ticker, stop_pct in [("MSTR", 0.03), ("MSTU", 0.07)]:
        print(f"    {ticker} …", end=" ", flush=True)
        px = fetch_equity(ticker, pre_dt, end_dt, dates)
        if px is None or not np.isfinite(px[_bt0]):
            print("SKIP (no data)"); continue

        r_cur = run_backtest(ticker, px, stop_pct, sig["tf1_entry"], sig,
                             dates, _bt0, INITIAL_CAPITAL,
                             f"{ticker} — Current (TF1+V-Gate)")
        r_new = run_backtest(ticker, px, stop_pct, sig["new_entry"], sig,
                             dates, _bt0, INITIAL_CAPITAL,
                             f"{ticker} — New (TF1+V-Gate+SPX slope)")

        # Count SPX-filtered entries (entries allowed in current but blocked in new)
        n_blocked = int(np.sum(sig["tf1_entry"][_bt0:] & ~sig["new_entry"][_bt0:]))
        n_spx_pass = int(np.sum(sig["new_entry"][_bt0:]))
        print(f"OK — current:{r_cur['stats']['n_trades']} trades, "
              f"new:{r_new['stats']['n_trades']} trades, "
              f"SPX-blocked:{n_blocked} signals")

        results[ticker] = dict(current=r_cur, new=r_new,
                               n_blocked=n_blocked, n_spx_pass=n_spx_pass)

    return results


# ── Pretty-print comparison table ────────────────────────────────────────────

def print_summary_table(all_results):
    print("\n" + "="*120)
    print("  COMPARISON: Current TF2+V-Gate  vs  New SPX-Slope-Filtered Entry")
    print("  (Δ = New minus Current;  ▲=improvement  ▼=degradation)")
    print("="*120)

    for ticker in ["MSTR", "MSTU"]:
        print(f"\n{'─'*120}")
        print(f"  {ticker}")
        print(f"{'─'*120}")
        hdr_cols = ["Period", "Variant", "Total Ret%", "After-Tax%", "B&H Ret%",
                    "Max DD%", "Sharpe", "Profit F.", "#Trades", "Win Rate%", "Time Mkt%"]
        print("  " + _hdr(hdr_cols))
        print("  " + _sep(len(hdr_cols)))

        for period_name, period_data in all_results.items():
            if period_data is None or ticker not in period_data:
                for lbl in ["Current", "New", "Δ (New−Cur)"]:
                    print("  " + _row([period_name if lbl=="Current" else "", lbl,
                                       *["n/a"]*(len(hdr_cols)-2)]))
                continue

            pd_t = period_data[ticker]
            for variant_key, label in [("current","Current"), ("new","New")]:
                s = pd_t[variant_key]["stats"]
                cols = [
                    period_name if label=="Current" else "",
                    label,
                    _fmt_pct(s["strat_ret"]),
                    _fmt_pct(s["after_tax_ret"]),
                    _fmt_pct(s["bh_ret"]),
                    _fmt_dd(s["max_drawdown"]),
                    _fmt_sh(s["sharpe"]),
                    _fmt_pf(s["profit_factor"]),
                    str(s["n_trades"]),
                    f"{s['win_rate']:.1f}%",
                    f"{s['time_in_mkt']:.0f}%",
                ]
                print("  " + _row(cols))

            # Delta row
            sc = pd_t["current"]["stats"]; sn = pd_t["new"]["stats"]
            nb = pd_t["n_blocked"]
            delta_cols = [
                "", f"Δ [SPX blocked:{nb}]",
                _delta(sn["strat_ret"],    sc["strat_ret"],   higher_better=True),
                _delta(sn["after_tax_ret"],sc["after_tax_ret"],higher_better=True),
                "—",
                _delta(sn["max_drawdown"], sc["max_drawdown"], higher_better=False),
                _delta(sn["sharpe"],       sc["sharpe"],       higher_better=True),
                "—", "—",
                _delta(sn["win_rate"],     sc["win_rate"],     higher_better=True),
                "—",
            ]
            print("  " + _row(delta_cols))
            print("  " + "·"*(_sep(len(hdr_cols)).__len__()))

        print()


def print_trade_logs(all_results):
    print("\n" + "="*120)
    print("  TRADE LOGS — MSTR & MSTU  (both Current and New signals)")
    print("="*120)

    for ticker in ["MSTR", "MSTU"]:
        for period_name, period_data in all_results.items():
            if period_data is None or ticker not in period_data:
                continue
            pd_t = period_data[ticker]
            for variant_key, vlabel in [("current", "Current"), ("new", "SPX-Filtered New")]:
                bt = pd_t[variant_key]
                trades = bt["trades"]
                s = bt["stats"]
                print(f"\n  ── {ticker} | {period_name} | {vlabel} ──")
                print(f"     Trades: {s['n_trades']}  |  Win Rate: {s['win_rate']:.1f}%  |  "
                      f"Total Return: {s['strat_ret']:+.1f}%  |  Max DD: {s['max_drawdown']:.1f}%  |  "
                      f"Sharpe: {s['sharpe']:.2f}  |  Profit Factor: {_fmt_pf(s['profit_factor'])}")
                if not trades:
                    print("     (no trades)")
                    continue
                tlog_cols = ["#", "Entry Date", "Entry$", "Exit Date", "Exit$",
                             "P&L%", "P&L$", "Days", "Exit Signal", "Trigger", "NAV After"]
                TW = 13
                print("  " + " | ".join(f"{c:>{TW}}" for c in tlog_cols))
                print("  " + "-+-".join(["-"*TW]*len(tlog_cols)))
                for j, t in enumerate(trades, 1):
                    result = "WIN" if t["pnl_pct"] > 0 else "LOSS"
                    star   = "✓" if t["pnl_pct"] > 0 else "✗"
                    row = [
                        str(j),
                        str(pd.Timestamp(t["entry_date"]).date()),
                        f"${t['entry_price']:.2f}",
                        str(pd.Timestamp(t["exit_date"]).date()),
                        f"${t['exit_price']:.2f}",
                        f"{t['pnl_pct']:+.2f}%",
                        f"${t['pnl_abs']:+,.0f}",
                        str(t["duration_days"]),
                        t["exit_signal"],
                        t.get("entry_trigger", "—"),
                        f"${t['exit_nav']:,.0f}",
                    ]
                    print("  " + " | ".join(f"{v:>{TW}}" for v in row))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*70)
    print(" BTC TF2+V-Gate vs SPX-Slope-Filtered Entry — Comparison Backtest")
    print("="*70)

    # Determine the widest fetch window needed (all periods)
    all_starts = [pd.Timestamp(s) for s, _ in PERIODS.values()]
    all_ends   = [pd.Timestamp(e) for _, e in PERIODS.values()]
    fetch_start = (min(all_starts) - pd.Timedelta(days=210)).strftime("%Y-%m-%d")
    fetch_end   = (max(all_ends)   + pd.Timedelta(days=3)).strftime("%Y-%m-%d")

    print(f"\n[1/2] Building predictions & fetching data ({fetch_start} → {fetch_end})")
    result = build_preds_with_spx(fetch_start, fetch_end)
    if result is None:
        print("ERROR: Could not build predictions — aborting."); return

    preds_df, raw_df, spx_series = result
    print(f"  SPX close: {int(spx_series.notna().sum())} bars")

    print("\n[2/2] Running backtests for all periods × tickers × variants …")
    all_results = {}
    for period_name, (start_iso, end_iso) in PERIODS.items():
        all_results[period_name] = run_period(
            period_name, start_iso, end_iso, preds_df, raw_df, spx_series
        )

    print_summary_table(all_results)
    print_trade_logs(all_results)

    # SPX filter impact summary
    print("\n" + "="*120)
    print("  SPX-SLOPE FILTER IMPACT SUMMARY")
    print("="*120)
    print(f"  {'Period':<45}  {'Ticker':<6}  {'Cur Trades':>10}  "
          f"{'New Trades':>10}  {'Blocked':>8}  {'Cur Ret%':>9}  {'New Ret%':>9}  "
          f"{'Δ Ret':>8}  {'Cur MaxDD':>10}  {'New MaxDD':>10}  {'Δ DD':>8}")
    print(f"  {'-'*45}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*9}  {'-'*9}  "
          f"{'-'*8}  {'-'*10}  {'-'*10}  {'-'*8}")
    for period_name, period_data in all_results.items():
        if period_data is None:
            continue
        for ticker in ["MSTR", "MSTU"]:
            if ticker not in period_data:
                continue
            pd_t   = period_data[ticker]
            sc, sn = pd_t["current"]["stats"], pd_t["new"]["stats"]
            nb     = pd_t["n_blocked"]
            d_ret  = sn["strat_ret"] - sc["strat_ret"]
            d_dd   = sn["max_drawdown"] - sc["max_drawdown"]
            ret_flag = "▲" if d_ret > 0 else ("▼" if d_ret < 0 else "·")
            dd_flag  = "▲" if d_dd > 0 else ("▼" if d_dd < 0 else "·")  # ▲ DD is worse
            dd_flag  = "▲" if d_dd < 0 else ("▼" if d_dd > 0 else "·")  # less negative = improvement
            print(f"  {period_name:<45}  {ticker:<6}  "
                  f"{sc['n_trades']:>10}  {sn['n_trades']:>10}  {nb:>8}  "
                  f"{sc['strat_ret']:>+8.1f}%  {sn['strat_ret']:>+8.1f}%  "
                  f"{ret_flag}{d_ret:>+6.1f}pp  "
                  f"{sc['max_drawdown']:>+9.1f}%  {sn['max_drawdown']:>+9.1f}%  "
                  f"{dd_flag}{d_dd:>+6.1f}pp")
    print()


if __name__ == "__main__":
    main()
