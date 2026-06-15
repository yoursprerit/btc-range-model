#!/usr/bin/env python3
"""
Full-period variant comparison matching the UI backtest logic exactly.

Logic:
  BTC  — NO stop-loss; exits only on D3 (bull) or D2/D3 (bear)   [same as UI tab]
  MSTR — −3 % stop-loss, SL5 regime-adaptive re-entry             [same as UI tab]
  MSTU — −7 % stop-loss, SL5 regime-adaptive re-entry             [same as UI tab]

Data (v2 · 2026-06-15):
  Signals and features: data/backtest/raw_features_daily.csv      (same path as UI)
  Prices:               data/backtest/{btc,mstr,mstu}_*.csv

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

import json
import joblib
import numpy as np
import pandas as pd

DATA_DIR = _ROOT / "data" / "backtest"
MODEL_PATH = _ROOT / "models" / "inference_assets_ct.joblib"

FULL_START = "2024-06-01"
FULL_END   = "2026-05-31"

WARMUP          = 35
INITIAL_CAPITAL = 100_000.0
CAP             = INITIAL_CAPITAL

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


btc_csv      = _load_csv("btc_usd_daily.csv")
mstr_csv     = _load_csv("mstr_daily.csv")
mstu_csv     = _load_csv("mstu_synthetic_daily.csv")
raw_feat_csv = _load_csv("raw_features_daily.csv")

manifest = {}
try:
    manifest = json.loads((DATA_DIR / "manifest.json").read_text())
except Exception:
    pass

# ─── build CT predictions from versioned raw_features CSV ────────────────────
# Matches the UI's _build_backtest_preds path when data_mtime > 0 (_skip_fetch=True)

def _build_preds(start_dt: pd.Timestamp, end_dt: pd.Timestamp, AD: dict):
    """Feature-engineer → predict using the versioned raw_features CSV."""
    pre_dt  = start_dt - pd.Timedelta(days=60)
    fetch_s = start_dt - pd.Timedelta(days=200)
    fetch_e = end_dt   + pd.Timedelta(days=3)

    df = raw_feat_csv.loc[(raw_feat_csv.index >= fetch_s) &
                          (raw_feat_csv.index <= fetch_e)].copy()

    c = df["btc_close"]; h = df["btc_high"]
    l_ = df["btc_low"];  v = df["btc_volume"]
    ret  = np.log(c).diff()
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
    feat["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    e12  = c.ewm(span=12, adjust=False).mean()
    e26  = c.ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    feat["macd"]      = macd / c
    feat["macd_sig"]  = macd.ewm(span=9, adjust=False).mean() / c
    feat["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    ma20 = c.rolling(20).mean(); sd20 = c.rolling(20).std()
    feat["bb_width"]   = (4*sd20) / ma20
    feat["dist_hi_30"] = c / c.rolling(30).max() - 1
    feat["dist_lo_30"] = c / c.rolling(30).min() - 1
    feat["dist_hi_90"] = c / c.rolling(90).max() - 1
    feat["vol_chg_1"]    = np.log(v).diff()
    feat["vol_z_20"]     = (np.log(v) - np.log(v).rolling(20).mean()) / np.log(v).rolling(20).std()
    feat["vol_ma_ratio"] = v / v.rolling(20).mean()
    dow = df.index.dayofweek
    for i in range(6): feat[f"dow_{i}"] = (dow == i).astype(float)
    for nm in ["spx","ndx","vix","gold","dxy","tnx","eth"]:
        col = f"{nm}_close"
        if col not in df.columns: continue
        s = df[col]; lr = np.log(s).diff()
        for k in [1,5,20]: feat[f"{nm}_ret_{k}"] = lr.rolling(k).sum()
        feat[f"{nm}_vol_20"] = lr.rolling(20).std()
    for cnm, ccol in [("spx","spx_close"),("ndx","ndx_close"),
                      ("gold","gold_close"),("dxy","dxy_close")]:
        if ccol in df.columns:
            feat[f"btc_{cnm}_corr_30"] = ret.rolling(30).corr(np.log(df[ccol]).diff())
    for col in [x for x in df.columns if x.startswith("oc_")]:
        s = df[col].astype(float); sl = np.log(s.replace(0, np.nan))
        feat[f"{col}_d1"]  = sl.diff(1)
        feat[f"{col}_d7"]  = sl.diff(7)
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
    for cb in ["cb_premium", "cb_premium_ma3", "cb_premium_z7"]:
        feat[cb] = df[cb].fillna(0.0)

    fc   = AD["feat_cols"]
    feat = feat.replace([np.inf, -np.inf], np.nan)
    for col in fc:
        if col not in feat.columns: feat[col] = np.nan
    F = feat[fc].dropna()

    if AD.get("ensemble") and AD.get("constituents"):
        yhi = np.mean([con["m_hi"].predict(F) for con in AD["constituents"]], axis=0)
        ylo = np.mean([con["m_lo"].predict(F) for con in AD["constituents"]], axis=0)
        if AD.get("blended") and float(AD.get("alpha", 1.0)) < 1.0:
            a   = float(AD["alpha"])
            yhi = a * yhi + (1-a) * float(AD.get("mu_hi", 0))
            ylo = a * ylo + (1-a) * float(AD.get("mu_lo", 0))
    else:
        yhi = AD["hi_model"].predict(F)
        ylo = AD["lo_model"].predict(F)

    c_vals = c.reindex(F.index).values
    ph = c_vals * (1 + np.clip(yhi, 0, None))
    pl = c_vals * (1 - np.clip(ylo, 0, None))
    idx = np.asarray(F.index, dtype="datetime64[ns]")
    nd  = np.empty(len(F), dtype="datetime64[ns]")
    nd[:-1] = idx[1:]; nd[-1] = idx[-1] + np.timedelta64(1, "D")
    preds = pd.DataFrame({"close_asof": c_vals, "pred_high": ph, "pred_low": pl},
                         index=pd.DatetimeIndex(nd, name="target_date"))
    preds = preds[~preds.index.duplicated(keep="last")]

    preds2 = preds.loc[(preds.index >= pre_dt) & (preds.index <= end_dt)].copy()
    preds2["actual_high"]  = h.reindex(preds2.index).values
    preds2["actual_low"]   = l_.reindex(preds2.index).values
    preds2["actual_close"] = c.reindex(preds2.index).values
    comp = preds2.dropna(subset=["actual_high","actual_low","actual_close"]).reset_index()
    return comp


def _compute_signals(comp: pd.DataFrame) -> dict:
    """Compute all trading signals from comp (identical to UI run_mstr_backtest)."""
    N      = len(comp)
    c_asof = comp["close_asof"].values.astype(float)
    ph     = comp["pred_high"].values.astype(float)
    pl     = comp["pred_low"].values.astype(float)
    ah     = comp["actual_high"].values.astype(float)
    al     = comp["actual_low"].values.astype(float)

    err_hi = (ah - ph) / c_asof * 100
    err_lo = (pl - al) / c_asof * 100
    hi_brk = (ah > ph).astype(int)
    lo_brk = (al < pl).astype(int)

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
        c2 = 0
        for k in range(i-1, -1, -1):
            if hi_brk[k]: c2 += 1
            else: break
        if c2 >= 3 and lo_brk[i]: d3[i] = True

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

    clean_7d = np.zeros(N, dtype=bool)
    for i in range(N):
        lo_i = max(0, i-7)
        clean_7d[i] = not bool(np.any(d1[lo_i:i] | d2[lo_i:i]))

    roll_norm = np.array([float(np.mean(err_hi[max(0,i-29):i+1])) for i in range(N)])
    dn_score  = np.zeros(N)
    for i in range(N):
        norm = max(abs(roll_norm[i]), 0.01)
        dn_score[i] = (
            (-ehma3[i] / norm)                            * 0.30 +
            (lb3[i]    / 3.0)                             * 0.30 +
            (elma3[i]  / max(abs(elma3[i]), 0.10))        * 0.20 +
            float(lo_brk[i])                              * 0.20
        )
    v_rev_bar = (dn_score > 0.8) & (err_lo > 3.0)
    v_recent  = np.zeros(N, dtype=bool)
    for i in range(N):
        v_recent[i] = bool(np.any(v_rev_bar[max(0, i-2):i+1]))

    tf_current  = u1 & ((above_ma30 ^ clean_7d) | v_recent)
    tf_option_b = u1 & ((bull_regime ^ clean_7d) | v_recent)
    tf_option_c = u1 & (bull_regime | (clean_7d & ~above_ma30) | v_recent)

    return dict(
        d2=d2, d3=d3, bull_regime=bull_regime, above_ma30=above_ma30,
        clean_7d=clean_7d, v_recent=v_recent,
        tf_current=tf_current, tf_option_b=tf_option_b, tf_option_c=tf_option_c,
    )


# ─── BTC backtest — NO stop-loss ─────────────────────────────────────────────

def run_btc_backtest(dates, btc_px, sigs, entry_key, bt_start):
    N   = len(dates)
    d2  = sigs["d2"];  d3 = sigs["d3"];  bull = sigs["bull_regime"]
    tf  = sigs[entry_key]
    _bt0 = max(WARMUP, int(pd.DatetimeIndex(dates).searchsorted(bt_start)))
    nav  = CAP; pos = "CASH"; qty = 0.0
    e_price = e_date = None
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
    ret    = (final_nav / CAP - 1) * 100
    wins   = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    wr     = 100 * len(wins) / max(1, len(trades))
    avg_p  = float(np.mean([t["pnl_pct"] for t in trades])) if trades else 0.0
    best_t = max((t["pnl_pct"] for t in trades), default=0.0)
    worst_t = min((t["pnl_pct"] for t in trades), default=0.0)
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
    ds_ver  = manifest.get("version", "?")
    ds_date = manifest.get("pull_date", "?")

    print("\n" + "═"*90)
    print("  VARIANT COMPARISON — FULL PERIOD  Jun 2024 → May 2026")
    print(f"  Dataset {ds_ver} · {ds_date}  (data/backtest/raw_features_daily.csv — same as UI)")
    print("  BTC: no stop-loss  |  MSTR: −3% SL  |  MSTU: −7% SL (UI-exact logic)")
    print("═"*90)

    print(f"\nLoading CT model …")
    AD = joblib.load(str(MODEL_PATH))

    start_dt = pd.Timestamp(FULL_START)
    end_dt   = pd.Timestamp(FULL_END)
    bt_start = start_dt

    print(f"Building predictions from versioned raw_features_daily.csv …")
    comp = _build_preds(start_dt, end_dt, AD)
    sigs = _compute_signals(comp)
    dates = pd.DatetimeIndex(comp["target_date"])
    N = len(comp)
    _bt0 = max(WARMUP, int(dates.searchsorted(bt_start)))
    print(f"  comp: {N} rows  {dates[0].date()} → {dates[-1].date()}")
    print(f"  backtest starts at index {_bt0} = {dates[_bt0].date()}")

    # ── BTC prices ────────────────────────────────────────────────────────────
    btc_px = btc_csv["close"].reindex(dates).ffill().bfill().values.astype(float)
    bh_btc = (btc_px[N-1] / btc_px[_bt0] - 1) * 100

    # ── MSTR prices ───────────────────────────────────────────────────────────
    _mstr_all = mstr_csv["close"].sort_index().reindex(
        pd.date_range(mstr_csv.index[0], mstr_csv.index[-1], freq="D")
    ).ffill()
    mstr_px = _mstr_all.reindex(dates).ffill().bfill().values.astype(float)
    bh_mstr = (mstr_px[N-1] / mstr_px[_bt0] - 1) * 100

    # ── MSTU prices ───────────────────────────────────────────────────────────
    _mstu_all = mstu_csv["close"].sort_index().reindex(
        pd.date_range(mstu_csv.index[0], mstu_csv.index[-1], freq="D")
    ).ffill()
    mstu_px = _mstu_all.reindex(dates).ffill().bfill().values.astype(float)
    bh_mstu = (mstu_px[N-1] / mstu_px[_bt0] - 1) * 100

    # ── run all variants ──────────────────────────────────────────────────────
    results = {"BTC": {}, "MSTR": {}, "MSTU": {}}
    bh_rets = {"BTC": bh_btc, "MSTR": bh_mstr, "MSTU": bh_mstu}

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
        for vname, s in results[asset].items():
            alpha  = s["ret"] - bh_rets[asset]
            short  = vname.split("(")[0].strip()
            marker = " ◀ BEST" if s["ret"] == max(r["ret"] for r in results[asset].values()) else ""
            print(f"  {short:<32} {s['ret']:>+8.1f}% {alpha:>+7.1f}pp {s['n_trades']:>7d} "
                  f"{s['win_rate']:>6.0f}% {s['avg_pnl']:>+7.1f}% {s['best']:>+7.1f}% "
                  f"{s['worst']:>+7.1f}% {s['max_dd']:>+7.1f}% {s['sharpe']:>7.2f}{marker}")
        best_v = max(results[asset], key=lambda v: results[asset][v]["ret"])
        print(f"\n  ★ Winner: {best_v} → {results[asset][best_v]['ret']:+.1f}%")

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
