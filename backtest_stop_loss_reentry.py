#!/usr/bin/env python3
"""
Stop-Loss Re-Entry Criteria Analysis
=====================================
Exact stop-loss levels tested:
  BTC  : trailing stop  −7%   (fires when close drops 7% below the running peak)
  MSTR : fixed stop     −3%   (fires when MSTR close drops 3% below entry price)
  MSTU : fixed stop     −10%  (fires when MSTU close drops 10% below entry price)

IMPLEMENTED RECOMMENDATIONS (in app/btc_hourly_app.py and TRADING_STRATEGY.md):
  BTC  : NO STOP LOSS — trailing −7% provides no consistent benefit; TF2 D2/D3
         exits already handle adverse moves; all SL variants underperform vs B0.
  MSTR : Fixed −3% stop + SL5 regime-adaptive re-entry
         (bull: immediate re-entry on next signal; bear: 10-bar cooldown)
  MSTU : Fixed −10% stop + SL1 above-exit-price re-entry
         (re-enter only when MSTU recovers above the stop exit price)

These tight stops cause premature exits in bull markets — the position is stopped
out during a normal consolidation, then BTC continues higher without us.

Question: after a stop fires, what re-entry criterion best recovers the missed
upside while avoiding re-entering into continued downside?

Baseline:
  B0 — TF2+V-Gate, NO stop loss (signal-only exits)

Stop-loss + re-entry variants (each variant only changes what happens AFTER a SL):
  SL0 — standard re-entry     : wait for normal U1+MA30/clean7d/V-gate, no price gate
  SL1 — above stop level      : re-enter only when asset price recovers above the SL
                                 exit price (most natural "wait for recovery" gate)
  SL2 — above entry price     : re-enter only when asset price recovers above the
                                 original entry price (full pullback recovery)
  SL3 — BTC +2% recovery      : re-enter when BTC close > BTC-at-SL-exit × 1.02
                                 (signal asset confirms recovery, not execution asset)
  SL4 — 5-bar cooldown        : block re-entry for 5 bars after SL, then standard
  SL5 — regime-adaptive       : bull regime → standard re-entry; bear → 10-bar cooldown
                                 ★ RECOMMENDED for MSTR

Periods:
  Bull  Sep 2024 → Sep 2025   BTC +93%   [in-sample for CT model]
  Bear  Jun 2025 → May 2026   BTC −35%   [partly OOS]
  OOS   Sep 2025 → May 2026   BTC −33%   [fully OOS]
  OOS-Recent  Mar → May 2026  [fully OOS, recent]
  Full  Jun 2024 → May 2026   2-year combined

Assets: BTC (stop on BTC price), MSTR (stop on MSTR price), MSTU (stop on MSTU price).
All entry/exit SIGNALS are BTC-derived (TF2+V-Gate).
Stop loss fires on EXECUTION ASSET price.
MSTU (T-Rex 2X Long MSTR) launched ≈ Sep 18 2024; unavailable for earlier periods.
"""

import sys, warnings, joblib, requests, time as _time
warnings.filterwarnings("ignore")
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import yfinance as yf

# ─────────────────────────────────────────────────────────────────────────────
# Configuration — exact stop levels from the live strategy
# ─────────────────────────────────────────────────────────────────────────────
ASSET_CONFIG = {
    "BTC": {
        "ticker":    "BTC-USD",
        "sl_type":   "trailing",   # trailing from running high-watermark
        "sl_pct":    0.07,         # fire when price < watermark × (1 − 0.07)
        "name":      "Bitcoin",
    },
    "MSTR": {
        "ticker":    "MSTR",
        "sl_type":   "fixed",      # fixed from entry price
        "sl_pct":    0.03,         # fire when MSTR < entry × 0.97
        "name":      "MicroStrategy",
    },
    "MSTU": {
        "ticker":    "MSTU",
        "sl_type":   "fixed",      # fixed from entry price
        "sl_pct":    0.10,         # fire when MSTU < entry × 0.90
        "name":      "T-Rex 2X Long MSTR (2×)",
    },
}

REENTRY_LABELS = {
    "B0":  "TF2+VGate (no SL)  [baseline]",
    "SL0": "SL + standard re-entry",
    "SL1": "SL + above SL exit price",       # asset recovers past stop level
    "SL2": "SL + above original entry",      # asset recovers above entry price
    "SL3": "SL + BTC +2% recovery",          # BTC signal asset confirms
    "SL4": "SL + 5-bar cooldown",
    "SL5": "SL + regime-adaptive",
}

PERIODS = {
    "Bull (Sep24→Sep25)":     ("2024-09-17", "2025-09-17", "2024-06-01"),
    "Bear (Jun25→May26)":     ("2025-06-01", "2026-05-26", "2025-03-01"),
    "OOS (Sep25→May26)":      ("2025-09-19", "2026-05-26", "2025-06-01"),
    "OOS-Recent (Mar→May26)": ("2026-03-01", "2026-05-26", "2025-12-01"),
    "Full (Jun24→May26)":     ("2024-06-01", "2026-05-26", "2024-03-01"),
}

INITIAL_CAPITAL = 100_000.0
WARMUP = 35

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
_MODEL_PATH = _ROOT / "models" / "inference_assets_ct.joblib"
if not _MODEL_PATH.exists():
    _MODEL_PATH = _ROOT / "artifacts" / "artifacts.pkl"

print(f"Loading CT model from: {_MODEL_PATH}")
AD = joblib.load(str(_MODEL_PATH))
print(f"  keys: {list(AD.keys())[:6]}")

# ─────────────────────────────────────────────────────────────────────────────
# Data fetching
# ─────────────────────────────────────────────────────────────────────────────
_MACRO_SYMS = {
    "eth": "ETH-USD", "spx": "^GSPC", "ndx": "^IXIC",
    "vix": "^VIX",   "gold": "GC=F",  "dxy": "DX-Y.NYB", "tnx": "^TNX",
}
_ONCHAIN = [
    "hash-rate", "difficulty", "n-transactions", "miners-revenue",
    "n-unique-addresses", "transaction-fees-usd", "mempool-size",
    "estimated-transaction-volume-usd", "market-cap",
    "avg-block-size", "cost-per-transaction",
]


def _yf_dl(ticker: str, start: str, end: str) -> pd.DataFrame:
    d = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] for c in d.columns]
    d.index = pd.DatetimeIndex(d.index).tz_localize(None).normalize()
    return d


def fetch_data(fetch_start: str, fetch_end: str) -> pd.DataFrame:
    print(f"  Fetching {fetch_start} → {fetch_end} …")
    d = _yf_dl("BTC-USD", fetch_start, fetch_end)
    frames = {
        "btc_close": d["Close"], "btc_high": d["High"],
        "btc_low":   d["Low"],   "btc_volume": d["Volume"],
    }
    for nm, sym in _MACRO_SYMS.items():
        try:
            frames[f"{nm}_close"] = _yf_dl(sym, fetch_start, fetch_end)["Close"]
        except Exception:
            pass
    df = pd.DataFrame(frames).sort_index().ffill(limit=5)
    df.index.name = "date"
    print("    On-chain …", end=" ", flush=True)
    ok = 0
    for m in _ONCHAIN:
        try:
            r = requests.get(
                f"https://api.blockchain.info/charts/{m}",
                params={"timespan": "3years", "format": "json", "sampled": "true"},
                timeout=20)
            vals = r.json().get("values", [])
            s = pd.Series(
                {pd.Timestamp(v["x"], unit="s").normalize(): v["y"] for v in vals},
                name=f"oc_{m.replace('-','_')}", dtype=float)
            s = s[~s.index.duplicated(keep="last")].sort_index()
            s.index = pd.DatetimeIndex(s.index).tz_localize(None)
            df[s.name] = s.reindex(df.index).ffill(limit=7)
            ok += 1
        except Exception:
            pass
    print(f"{ok}/{len(_ONCHAIN)} OK")
    return df


def fetch_asset_prices(ticker: str, fetch_start: str, fetch_end: str) -> pd.Series:
    try:
        d = _yf_dl(ticker, fetch_start, fetch_end)
        px = d["Close"].sort_index()
        all_days = pd.date_range(px.index[0], max(px.index[-1], pd.Timestamp(fetch_end)), freq="D")
        return px.reindex(all_days).ffill()
    except Exception as e:
        print(f"    [WARN] {ticker}: {e}")
        return pd.Series(dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# CT Predictions (includes Coinbase premium)
# ─────────────────────────────────────────────────────────────────────────────
def build_ct_preds(df: pd.DataFrame,
                   fetch_start: str = "2020-01-01",
                   fetch_end: str   = "2030-01-01") -> pd.DataFrame:
    c, h, l_, v = df["btc_close"], df["btc_high"], df["btc_low"], df["btc_volume"]
    ret = np.log(c).diff()
    f = pd.DataFrame(index=df.index)
    for k in [1, 3, 5, 7, 14, 30]: f[f"ret_{k}"] = ret.rolling(k).sum()
    for k in [5, 10, 20, 30]:       f[f"vol_{k}"] = ret.rolling(k).std()
    prev_c = c.shift(1)
    tr = pd.concat([(h-l_), (h-prev_c).abs(), (l_-prev_c).abs()], axis=1).max(axis=1)
    for k in [7, 14, 30]: f[f"atr_{k}"] = tr.rolling(k).mean() / c
    f["range_today"]  = (h - l_) / c
    f["range_ma7"]    = ((h - l_) / c).rolling(7).mean()
    f["range_ma30"]   = ((h - l_) / c).rolling(30).mean()
    f["range_std30"]  = ((h - l_) / c).rolling(30).std()
    g  = c.diff().clip(lower=0).rolling(14).mean()
    ls = (-c.diff().clip(upper=0)).rolling(14).mean()
    f["rsi_14"] = 100 - 100 / (1 + g / ls.replace(0, np.nan))
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    f["macd"]      = macd / c
    f["macd_sig"]  = macd.ewm(span=9, adjust=False).mean() / c
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    ma20 = c.rolling(20).mean(); sd20 = c.rolling(20).std()
    f["bb_width"]   = (4 * sd20) / ma20
    f["dist_hi_30"] = c / c.rolling(30).max() - 1
    f["dist_lo_30"] = c / c.rolling(30).min() - 1
    f["dist_hi_90"] = c / c.rolling(90).max() - 1
    f["vol_chg_1"]    = np.log(v).diff()
    f["vol_z_20"]     = (np.log(v)-np.log(v).rolling(20).mean()) / np.log(v).rolling(20).std()
    f["vol_ma_ratio"] = v / v.rolling(20).mean()
    dow = df.index.dayofweek
    for i in range(6): f[f"dow_{i}"] = (dow == i).astype(float)
    for nm in ["spx","ndx","vix","gold","dxy","tnx","eth"]:
        col = f"{nm}_close"
        if col not in df.columns: continue
        s = df[col]; lr = np.log(s).diff()
        for k in [1, 5, 20]: f[f"{nm}_ret_{k}"] = lr.rolling(k).sum()
        f[f"{nm}_vol_20"] = lr.rolling(20).std()
    for corr_nm, corr_col in [("spx","spx_close"),("ndx","ndx_close"),
                                ("gold","gold_close"),("dxy","dxy_close")]:
        if corr_col in df.columns:
            f[f"btc_{corr_nm}_corr_30"] = ret.rolling(30).corr(np.log(df[corr_col]).diff())
    for col in [x for x in df.columns if x.startswith("oc_")]:
        s = df[col].astype(float); sl = np.log(s.replace(0, np.nan))
        f[f"{col}_d1"]  = sl.diff(1); f[f"{col}_d7"] = sl.diff(7)
        f[f"{col}_z30"] = (sl - sl.rolling(30).mean()) / sl.rolling(30).std()
    nh, nl_ = h.shift(-1), l_.shift(-1)
    y_hi = (nh - c) / c; y_lo = (c - nl_) / c
    f["y_hi_ema3"] = y_hi.shift(1).ewm(span=3, adjust=False).mean()
    f["y_lo_ema3"] = y_lo.shift(1).ewm(span=3, adjust=False).mean()
    f["y_hi_ema7"] = y_hi.shift(1).ewm(span=7, adjust=False).mean()
    f["y_lo_ema7"] = y_lo.shift(1).ewm(span=7, adjust=False).mean()
    p3h = h.shift(1).rolling(3).max(); p3l = l_.shift(1).rolling(3).min()
    f["above_3d_high"]  = (c > p3h).astype(float)
    f["below_3d_low"]   = (c < p3l).astype(float)
    f["bo_strength_up"] = (c / p3h - 1).clip(lower=0)
    f["bo_strength_dn"] = (1 - c / p3l).clip(lower=0)
    ya = y_hi.shift(1); yb = y_lo.shift(1)
    f["y_hi_surprise"] = ya - ya.ewm(span=7, adjust=False).mean()
    f["y_lo_surprise"] = yb - yb.ewm(span=7, adjust=False).mean()
    neg_ret = ret.clip(upper=0)
    f["dn_vol_5"]  = neg_ret.rolling(5).std()
    f["dn_vol_20"] = neg_ret.rolling(20).std()
    sma50 = c.rolling(50).mean()
    f["below_sma50"]    = (c < sma50).astype(float)
    f["below_sma50_5d"] = f["below_sma50"].rolling(5).min().fillna(0)

    # ── Coinbase premium (public API; set to 0 if unavailable) ───────────
    _CB_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    _cb_rows: list = []
    _cb_cur  = pd.Timestamp(fetch_start)
    _cb_end  = pd.Timestamp(fetch_end)
    while _cb_cur <= _cb_end:
        _cb_chunk = min(_cb_cur + pd.Timedelta(days=299), _cb_end)
        try:
            _r = requests.get(_CB_URL, params={
                "granularity": 86400,
                "start": _cb_cur.strftime("%Y-%m-%dT00:00:00Z"),
                "end":   (_cb_chunk + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z"),
            }, timeout=30)
            if _r.status_code == 200:
                _cb_rows.extend(_r.json())
        except Exception:
            pass
        _cb_cur = _cb_chunk + pd.Timedelta(days=1)
        _time.sleep(0.15)
    if _cb_rows:
        _cb_df = pd.DataFrame(_cb_rows, columns=["ts","low","high","open","close","volume"])
        _cb_df["date"] = pd.to_datetime(_cb_df["ts"], unit="s").dt.normalize()
        _cb_df = _cb_df.drop_duplicates("date").set_index("date").sort_index()
        _cb_close = _cb_df["close"].reindex(df.index).astype(float)
        _prem = (_cb_close - c) / c * 100
        f["cb_premium"]     = _prem
        f["cb_premium_ma3"] = _prem.rolling(3).mean()
        f["cb_premium_z7"]  = (_prem - _prem.rolling(7).mean()) / _prem.rolling(7).std()
        print(f"    Coinbase premium: {int(_prem.notna().sum())} days")
    else:
        f["cb_premium"] = f["cb_premium_ma3"] = f["cb_premium_z7"] = 0.0
        print("    Coinbase premium: unavailable (set to 0)")

    fc = AD["feat_cols"]
    f  = f.replace([np.inf, -np.inf], np.nan)
    for col in fc:
        if col not in f.columns: f[col] = np.nan
    for _cb_col in ["cb_premium", "cb_premium_ma3", "cb_premium_z7"]:
        if _cb_col in f.columns:
            f[_cb_col] = f[_cb_col].fillna(0.0)
    F = f[fc].dropna()
    if F.empty:
        raise RuntimeError("Feature matrix empty — check data")
    print(f"    CT predictions: {len(F)} rows")

    if AD.get("ensemble") and AD.get("constituents"):
        yhi = np.mean([con["m_hi"].predict(F) for con in AD["constituents"]], axis=0)
        ylo = np.mean([con["m_lo"].predict(F) for con in AD["constituents"]], axis=0)
        if AD.get("blended") and float(AD.get("alpha", 1.0)) < 1.0:
            a = float(AD["alpha"])
            yhi = a * yhi + (1 - a) * float(AD.get("mu_hi", 0))
            ylo = a * ylo + (1 - a) * float(AD.get("mu_lo", 0))
    else:
        yhi = AD["hi_model"].predict(F)
        ylo = AD["lo_model"].predict(F)

    c_vals = c.reindex(F.index).values
    ph = c_vals * (1 + np.clip(yhi, 0, None))
    pl = c_vals * (1 - np.clip(ylo, 0, None))
    idx = np.asarray(F.index, dtype="datetime64[ns]")
    nd = np.empty(len(F), dtype="datetime64[ns]")
    nd[:-1] = idx[1:]; nd[-1] = idx[-1] + np.timedelta64(1, "D")
    res = pd.DataFrame(
        {"close_asof": c_vals, "pred_high": ph, "pred_low": pl},
        index=pd.DatetimeIndex(nd, name="target_date"))
    return res[~res.index.duplicated(keep="last")]


# ─────────────────────────────────────────────────────────────────────────────
# Signal computation (BTC-derived; drives entry/exit for all assets)
# ─────────────────────────────────────────────────────────────────────────────
def compute_signals(comp: pd.DataFrame) -> dict:
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
        s = max(0, i - 2)
        ehma3[i] = np.mean(err_hi[s:i+1]); elma3[i] = np.mean(err_lo[s:i+1])
        hb3[i]   = int(np.sum(hi_brk[s:i+1])); lb3[i] = int(np.sum(lo_brk[s:i+1]))

    u1 = (ehma3 > 0.7) & (hb3 >= 2)
    d1 = (lb3 >= 2)    & (elma3 > 0.5)
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
    ma30_slope = np.zeros(N, dtype=bool)
    for i in range(5, N):
        if np.isfinite(ma30[i]) and np.isfinite(ma30[i - 5]):
            ma30_slope[i] = ma30[i] > ma30[i - 5]
    bull_regime = above_ma30 & ma30_slope

    clean_7d = np.zeros(N, dtype=bool)
    for i in range(N):
        lo_i = max(0, i - 7)
        clean_7d[i] = not bool(np.any(d1[lo_i:i] | d2[lo_i:i]))

    roll_norm = np.array([float(np.mean(err_hi[max(0, i-29):i+1])) for i in range(N)])
    dn_score  = np.zeros(N)
    for i in range(N):
        norm = max(abs(roll_norm[i]), 0.01)
        dn_score[i] = (
            (-ehma3[i] / norm)                        * 0.30 +
            (lb3[i] / 3.0)                            * 0.30 +
            (elma3[i] / max(abs(elma3[i]), 0.10))     * 0.20 +
            float(lo_brk[i])                          * 0.20
        )
    v_rev_bar = (dn_score > 0.8) & (err_lo > 3.0)
    v_recent  = np.zeros(N, dtype=bool)
    for i in range(N):
        v_recent[i] = bool(np.any(v_rev_bar[max(0, i - 2):i + 1]))

    # EMA10 of BTC close (for SL3 variant)
    ema10 = pd.Series(c_asof).ewm(span=10, adjust=False).mean().values

    tf2_entry = u1 & (above_ma30 | clean_7d | v_recent)

    return dict(
        N=N, c_asof=c_asof,
        u1=u1, d1=d1, d2=d2, d3=d3,
        above_ma30=above_ma30, bull_regime=bull_regime,
        clean_7d=clean_7d, v_recent=v_recent,
        ema10=ema10, tf2_entry=tf2_entry,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Core backtest engine
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(
    dates:      pd.DatetimeIndex,
    asset_px:   np.ndarray,   # execution prices (BTC / MSTR / MSTU)
    btc_px:     np.ndarray,   # BTC close — always used for signal checks
    sigs:       dict,
    variant:    str   = "B0",
    sl_type:    str   = "fixed",  # "trailing" | "fixed"
    sl_pct:     float = 0.10,
    cap:        float = 100_000.0,
) -> dict:
    """
    Execute one backtest variant.

    Signal decisions (entry gate + TF2 exit) always use BTC-derived sigs.
    Stop-loss and execution P&L use asset_px (BTC / MSTR / MSTU).
    Re-entry gate depends on variant:
      B0  — no SL, pure signal-based exits
      SL0 — SL fires, immediate standard re-entry
      SL1 — SL fires, re-enter only when asset_px > sl_exit_asset_price
      SL2 — SL fires, re-enter only when asset_px > original entry_price
      SL3 — SL fires, re-enter only when btc_px > btc_at_sl_exit × 1.02
      SL4 — SL fires, 5-bar cooldown then standard entry
      SL5 — SL fires, bull: standard; bear: 10-bar cooldown
    """
    N         = len(dates)
    d2        = sigs["d2"]; d3 = sigs["d3"]
    bull      = sigs["bull_regime"]
    tf2_entry = sigs["tf2_entry"]

    nav   = cap; pos = "CASH"; qty = 0.0
    e_asset_price = None   # asset entry price
    e_btc_price   = None   # BTC price at entry
    e_date        = None
    e_nav         = None
    e_trigger     = None
    hwm           = None   # high-water mark for trailing stop

    # State after a stop-loss exit
    from_sl          = False
    sl_exit_asset_px = None   # asset price at stop exit (= stop level)
    sl_exit_btc_px   = None   # BTC price at stop exit
    bars_since_sl    = 0

    trades  = []
    nav_arr = np.full(N, np.nan)

    for i in range(N):
        si    = i - 1
        price = asset_px[i]   # today's asset close

        if i < WARMUP:
            nav_arr[i] = cap
            continue

        if pos == "CASH" and from_sl:
            bars_since_sl += 1

        if pos == "LONG":
            cur = qty * price

            # Update high-water mark (trailing stop only)
            if sl_type == "trailing":
                hwm = max(hwm, price)
                stop_level = hwm * (1.0 - sl_pct)
            else:
                stop_level = e_asset_price * (1.0 - sl_pct)

            # ── Stop-loss check ────────────────────────────────────────────
            sl_fired = (variant != "B0") and (price <= stop_level)

            if sl_fired:
                nav = cur
                trades.append(dict(
                    entry_date    = e_date,
                    entry_price   = e_asset_price,
                    entry_nav     = e_nav,
                    entry_trigger = e_trigger,
                    exit_date     = dates[i],
                    exit_price    = price,
                    exit_nav      = nav,
                    pnl_pct       = (price / e_asset_price - 1) * 100,
                    pnl_abs       = nav - e_nav,
                    exit_signal   = (f"SL-trail-{sl_pct*100:.0f}%"
                                     if sl_type == "trailing"
                                     else f"SL-fixed-{sl_pct*100:.0f}%"),
                    duration_days = (dates[i] - e_date).days,
                    was_sl        = True,
                    hwm           = hwm if sl_type == "trailing" else e_asset_price,
                ))
                pos = "CASH"; qty = 0.0
                from_sl          = True
                sl_exit_asset_px = price
                sl_exit_btc_px   = btc_px[i]
                bars_since_sl    = 0
                hwm              = None

            elif si >= 0:
                # ── TF2 signal exit ────────────────────────────────────────
                should_exit = bool(d3[si] or (d2[si] and not bull[si]))
                xl = "D3" if d3[si] else "D2"
                if should_exit:
                    nav = cur
                    trades.append(dict(
                        entry_date    = e_date,
                        entry_price   = e_asset_price,
                        entry_nav     = e_nav,
                        entry_trigger = e_trigger,
                        exit_date     = dates[i],
                        exit_price    = price,
                        exit_nav      = nav,
                        pnl_pct       = (price / e_asset_price - 1) * 100,
                        pnl_abs       = nav - e_nav,
                        exit_signal   = xl,
                        duration_days = (dates[i] - e_date).days,
                        was_sl        = False,
                        hwm           = hwm,
                    ))
                    pos = "CASH"; qty = 0.0
                    from_sl = False; hwm = None
                else:
                    nav = cur
            else:
                nav = cur

        else:  # CASH
            if si >= 0 and tf2_entry[si]:
                allow = True
                if from_sl and variant != "B0":
                    a_now   = asset_px[si]   # signal bar asset price (1-bar lag)
                    b_now   = btc_px[si]
                    if variant == "SL0":
                        allow = True

                    elif variant == "SL1":
                        # Re-enter only when asset has recovered above stop exit price
                        allow = (sl_exit_asset_px is not None and
                                 a_now >= sl_exit_asset_px)

                    elif variant == "SL2":
                        # Re-enter only when asset has fully recovered above original entry
                        allow = (e_asset_price is not None and
                                 a_now >= e_asset_price)

                    elif variant == "SL3":
                        # BTC must be ≥2% above BTC price at stop exit
                        allow = (sl_exit_btc_px is not None and
                                 b_now >= sl_exit_btc_px * 1.02)

                    elif variant == "SL4":
                        # 5-bar cooldown
                        allow = (bars_since_sl >= 5)

                    elif variant == "SL5":
                        # Regime-adaptive: bull → immediate; bear → 10-bar cooldown
                        allow = bool(bull[si]) or (bars_since_sl >= 10)

                if allow:
                    if not np.isfinite(price) or price <= 0:
                        nav_arr[i] = nav
                        continue
                    qty = nav / price
                    e_asset_price = price
                    e_btc_price   = btc_px[i]
                    e_date        = dates[i]
                    e_nav         = nav
                    pos           = "LONG"
                    hwm           = price   # initialise trailing HWM at entry
                    from_sl       = False
                    bars_since_sl = 0
                    abv = sigs["above_ma30"]; c7d = sigs["clean_7d"]; vr = sigs["v_recent"]
                    if vr[si] and not abv[si] and not c7d[si]:
                        e_trigger = "U1+V-reversal"
                    elif abv[si] and c7d[si]:
                        e_trigger = "U1+↑MA30+clean7d"
                    elif abv[si]:
                        e_trigger = "U1+↑MA30"
                    else:
                        e_trigger = "U1+clean7d"

        nav_arr[i] = qty * price if pos == "LONG" else nav

    if pos == "LONG" and np.isfinite(asset_px[N - 1]):
        nav_arr[N - 1] = qty * asset_px[N - 1]

    nav_s = pd.Series(nav_arr[WARMUP:], index=dates[WARMUP:]).ffill()
    bh_s  = pd.Series(
        cap * asset_px[WARMUP:] / asset_px[WARMUP], index=dates[WARMUP:])
    return dict(trades=trades, nav=nav_s, bh=bh_s,
                open_pos=(pos == "LONG"),
                open_entry=dict(price=e_asset_price, date=e_date) if pos == "LONG" else None)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def metrics(res: dict, cap: float = INITIAL_CAPITAL) -> dict:
    nav = res["nav"]; bh = res["bh"]; trades = res["trades"]
    fn  = float(nav.iloc[-1]); fb = float(bh.iloc[-1])
    n_y = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = ((fn / cap) ** (1 / n_y) - 1) * 100 if n_y > 0 else 0
    dr  = nav.pct_change().fillna(0)
    rfd = 1.045 ** (1 / 252) - 1
    exc = dr - rfd
    sharpe = float(exc.mean() / exc.std() * np.sqrt(252)) if exc.std() > 0 else 0
    rm     = nav.cummax(); dd = (nav - rm) / rm * 100
    max_dd = float(dd.min())
    rm_bh  = bh.cummax()
    bh_maxdd = float(((bh - rm_bh) / rm_bh * 100).min())
    wins   = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    sl_ex  = [t for t in trades if t.get("was_sl")]
    win_rate = 100 * len(wins) / len(trades) if trades else 0
    gp = sum(t["pnl_abs"] for t in wins)  if wins   else 0
    gl = abs(sum(t["pnl_abs"] for t in losses)) if losses else 1e-9
    pf = gp / gl
    avg_win  = float(np.mean([t["pnl_pct"] for t in wins]))   if wins   else 0
    avg_loss = float(np.mean([t["pnl_pct"] for t in losses])) if losses else 0
    days_in  = sum(t["duration_days"] for t in trades)
    tot_days = max(1, (nav.index[-1] - nav.index[0]).days)
    return dict(
        final_nav=fn,  bh_nav=fb,
        strat_ret=(fn/cap-1)*100, bh_ret=(fb/cap-1)*100,
        alpha_abs=fn-fb, cagr=cagr,
        sharpe=sharpe, max_dd=max_dd, bh_maxdd=bh_maxdd,
        n_trades=len(trades), n_sl=len(sl_ex),
        n_wins=len(wins), n_losses=len(losses),
        win_rate=win_rate, avg_win=avg_win, avg_loss=avg_loss,
        profit_factor=pf,
        time_in=100*days_in/tot_days,
        trades=trades, open_pos=res.get("open_pos"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Period runner
# ─────────────────────────────────────────────────────────────────────────────
def _prep_comp(df_raw, preds, start_iso, end_iso):
    sd, ed = pd.Timestamp(start_iso), pd.Timestamp(end_iso)
    p = preds.loc[
        (preds.index >= sd - pd.Timedelta(days=60)) & (preds.index <= ed)
    ].copy()
    p["actual_high"]  = df_raw["btc_high"].reindex(p.index).values
    p["actual_low"]   = df_raw["btc_low"].reindex(p.index).values
    p["actual_close"] = df_raw["btc_close"].reindex(p.index).values
    comp = p.dropna(subset=["actual_high","actual_low","actual_close"]).reset_index()
    comp = comp[comp["target_date"] >= sd].reset_index(drop=True)
    return comp


def run_period(period_name, start_iso, end_iso, fetch_start, df_raw_cache):
    fetch_end = (pd.Timestamp(end_iso) + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    cache_key = (fetch_start, fetch_end)
    if cache_key not in df_raw_cache:
        df_raw      = fetch_data(fetch_start, fetch_end)
        preds       = build_ct_preds(df_raw, fetch_start=fetch_start, fetch_end=fetch_end)
        mstr_px_raw = fetch_asset_prices("MSTR", fetch_start, fetch_end)
        mstu_px_raw = fetch_asset_prices("MSTU", fetch_start, fetch_end)
        df_raw_cache[cache_key] = dict(
            df_raw=df_raw, preds=preds,
            mstr_px=mstr_px_raw, mstu_px=mstu_px_raw)
    cached      = df_raw_cache[cache_key]
    df_raw      = cached["df_raw"]
    preds       = cached["preds"]
    mstr_px_raw = cached["mstr_px"]
    mstu_px_raw = cached["mstu_px"]

    comp = _prep_comp(df_raw, preds, start_iso, end_iso)
    if len(comp) < WARMUP + 3:
        print(f"  [SKIP] {period_name}: insufficient bars ({len(comp)})")
        return {}
    dates = pd.DatetimeIndex(comp["target_date"])
    sigs  = compute_signals(comp)
    N     = sigs["N"]
    btc_px = comp["actual_close"].values.astype(float)

    def align_asset(px_series):
        if px_series.empty: return None
        a = px_series.reindex(dates).ffill().bfill().values.astype(float)
        if np.sum(np.isfinite(a) & (a > 0)) < WARMUP + 3: return None
        return a

    asset_arrays = {
        "BTC":  btc_px,
        "MSTR": align_asset(mstr_px_raw),
        "MSTU": align_asset(mstu_px_raw),
    }

    variants = list(REENTRY_LABELS.keys())
    period_results = {}

    for asset_name, asset_px in asset_arrays.items():
        if asset_px is None:
            print(f"  [SKIP] {asset_name}: no price data for {period_name}")
            continue
        if asset_name == "MSTU":
            first_valid = int(np.argmax(np.isfinite(asset_px) & (asset_px > 0)))
            if first_valid > WARMUP + 10:
                print(f"  [SKIP] MSTU: data starts too late (bar {first_valid}/{N})")
                continue

        cfg    = ASSET_CONFIG[asset_name]
        sl_t   = cfg["sl_type"]
        sl_pct = cfg["sl_pct"]
        period_results[asset_name] = {}

        for variant in variants:
            try:
                res = run_backtest(dates, asset_px, btc_px, sigs,
                                   variant=variant, sl_type=sl_t, sl_pct=sl_pct,
                                   cap=INITIAL_CAPITAL)
                period_results[asset_name][variant] = metrics(res, INITIAL_CAPITAL)
            except Exception as e:
                print(f"  [ERR] {asset_name}/{variant}: {e}")

    return period_results


# ─────────────────────────────────────────────────────────────────────────────
# Printing helpers
# ─────────────────────────────────────────────────────────────────────────────
_HDR = ("Ret%", "B&H%", "Alpha$k", "MaxDD%", "Sharpe", "Trades", "#SL", "WR%", "TiM%")
_COL = 9

def _fmt_row(variant_label, m, baseline_nav=None):
    if m is None:
        return f"  {variant_label:<30}  — data unavailable"
    marker = "▲" if (baseline_nav and m["final_nav"] > baseline_nav) else " "
    return (
        f"  {variant_label:<30}"
        f"{m['strat_ret']:>{_COL-1}.1f}%"
        f"{m['bh_ret']:>{_COL-1}.1f}%"
        f"${m['alpha_abs']/1e3:>{_COL-2}.1f}k"
        f"{m['max_dd']:>{_COL-1}.1f}%"
        f"{m['sharpe']:>{_COL}.2f}"
        f"{m['n_trades']:>{_COL}}"
        f"{m['n_sl']:>{_COL}}"
        f"{m['win_rate']:>{_COL-1}.0f}%"
        f"{m['time_in']:>{_COL-1}.0f}%"
        f"  {marker}"
    )


def _sep():
    return "  " + "─" * (30 + _COL * 9 + 4)


def print_period_asset(period_name, asset_name, results):
    cfg = ASSET_CONFIG[asset_name]
    sl_desc = (f"trailing −{cfg['sl_pct']*100:.0f}%"
               if cfg["sl_type"] == "trailing"
               else f"fixed −{cfg['sl_pct']*100:.0f}% from entry")
    print(f"\n  {asset_name} ({cfg['name']})  — stop: {sl_desc}")
    print(_sep())
    hdr = f"  {'Variant':<30}" + "".join(f"{h:>{_COL}}" for h in _HDR)
    print(hdr)
    print(_sep())

    b0_nav = results.get("B0", {}).get("final_nav")
    sl0_nav = results.get("SL0", {}).get("final_nav")
    for v, lbl in REENTRY_LABELS.items():
        label = f"{'★' if v=='B0' else ' '} [{v}] {lbl}"[:30]
        m = results.get(v)
        ref = sl0_nav if v not in ("B0",) else None
        print(_fmt_row(label, m, ref))
    print(_sep())


def print_full_period(period_name, period_results):
    W = 78
    print(f"\n{'═'*W}")
    print(f"  PERIOD: {period_name}")
    print(f"{'═'*W}")
    for asset in ["BTC", "MSTR", "MSTU"]:
        if asset in period_results:
            print_period_asset(period_name, asset, period_results[asset])
        else:
            print(f"\n  {asset}: no data")


def print_cross_period_summary(all_results):
    W = 88
    print(f"\n{'═'*W}")
    print("  CROSS-PERIOD SUMMARY — Strategy Return % (and Δ vs SL0)")
    print(f"{'═'*W}")
    periods  = list(all_results.keys())
    variants = list(REENTRY_LABELS.keys())
    P_W, V_W = 24, 17

    for asset in ["BTC", "MSTR", "MSTU"]:
        cfg = ASSET_CONFIG[asset]
        sl_desc = (f"trail −{cfg['sl_pct']*100:.0f}%" if cfg["sl_type"]=="trailing"
                   else f"fixed −{cfg['sl_pct']*100:.0f}%")
        print(f"\n  {'─'*85}")
        print(f"  {asset} ({cfg['name']})  — stop: {sl_desc}")
        print(f"  {'─'*85}")
        pnames_short = [p.split("(")[0].strip()[:15] for p in periods]
        print(f"  {'Variant':<{P_W}}" + "".join(f"{p:>{V_W}}" for p in pnames_short))
        print(f"  {'─'*(P_W + V_W*len(periods))}")

        for v in variants:
            row_vals = []
            for pname in periods:
                pr = all_results.get(pname, {}).get(asset, {})
                m  = pr.get(v)
                sl0 = pr.get("SL0")
                if m is None:
                    row_vals.append("    n/a")
                elif v == "B0" or sl0 is None:
                    row_vals.append(f"{m['strat_ret']:>+8.1f}%")
                else:
                    delta = m["strat_ret"] - sl0["strat_ret"]
                    sign  = "+" if delta >= 0 else ""
                    row_vals.append(f"{m['strat_ret']:>+6.1f}%({sign}{delta:.1f})")
            label = (f"{'★' if v=='B0' else ' '} {REENTRY_LABELS[v]}")[:P_W]
            print(f"  {label:<{P_W}}" + "".join(f"{v:>{V_W}}" for v in row_vals))

        # B&H row
        bh_vals = []
        for pname in periods:
            pr = all_results.get(pname, {}).get(asset, {})
            m  = pr.get("B0")
            bh_vals.append(f"{m['bh_ret']:>+8.1f}%" if m else "    n/a")
        print(f"  {'─'*(P_W + V_W*len(periods))}")
        print(f"  {'  Buy & Hold':<{P_W}}" + "".join(f"{v:>{V_W}}" for v in bh_vals))

    print(f"\n{'═'*W}")


def print_sl_trade_detail(period_results, period_name, asset):
    """Show post-SL re-entry lag and outcome per variant for each SL exit."""
    if asset not in period_results: return
    res  = period_results[asset]
    sl0  = res.get("SL0")
    if not sl0: return
    sl_exits = [t for t in sl0["trades"] if t.get("was_sl")]
    if not sl_exits:
        print(f"\n  ✓ {asset} — no stops triggered in {period_name}")
        return

    cfg = ASSET_CONFIG[asset]
    sl_desc = (f"trail −{cfg['sl_pct']*100:.0f}%" if cfg["sl_type"]=="trailing"
               else f"fixed −{cfg['sl_pct']*100:.0f}%")
    print(f"\n  Stop-Loss Events — {asset} ({sl_desc}), {period_name}")
    print(f"  {len(sl_exits)} stop(s) fired in SL0 baseline")

    v_list = ["SL0","SL1","SL2","SL3","SL4","SL5"]
    header = (f"  {'Exit Date':>12}  {'Exit $':>10}  {'SL P&L':>8}  {'HWM $':>10}  "
              + "  ".join(f"[{v}]" for v in v_list))
    print(f"\n{header}")
    print(f"  {'─'*105}")

    for sl_t in sl_exits:
        exit_dt  = pd.Timestamp(sl_t["exit_date"])
        hwm_str  = (f"${sl_t['hwm']:>9,.0f}" if sl_t.get("hwm") else "     n/a   ")
        row = (f"  {exit_dt.strftime('%b %d %Y'):>12}  "
               f"${sl_t['exit_price']:>9,.1f}  "
               f"{sl_t['pnl_pct']:>+7.1f}%  "
               f"{hwm_str}  ")
        for v in v_list:
            vres = res.get(v)
            if vres is None:
                row += f"  {'n/a':>14}"; continue
            # Find the first re-entry trade AFTER this SL exit
            nxt = next(
                (t for t in vres["trades"]
                 if pd.Timestamp(t["entry_date"]) > exit_dt
                 and not t.get("was_sl")),
                None
            )
            if nxt:
                lag = (pd.Timestamp(nxt["entry_date"]) - exit_dt).days
                row += f"  {nxt['pnl_pct']:>+7.1f}%(+{lag:2d}d)"
            else:
                row += f"  {'stayed cash':>14}"
        print(row)
    print()


def print_recommendation(all_results):
    W = 88
    print(f"\n{'═'*W}")
    print("  RECOMMENDATION SUMMARY")
    print(f"{'═'*W}")
    periods  = list(all_results.keys())
    sl_vars  = [v for v in REENTRY_LABELS if v != "B0"]

    for asset in ["BTC", "MSTR", "MSTU"]:
        cfg = ASSET_CONFIG[asset]
        sl_desc = (f"trail −{cfg['sl_pct']*100:.0f}%" if cfg["sl_type"]=="trailing"
                   else f"fixed −{cfg['sl_pct']*100:.0f}%")
        print(f"\n  {'─'*65}")
        print(f"  {asset} ({cfg['name']})  — stop: {sl_desc}")
        print(f"  {'─'*65}")

        # Count how many SL exits occurred across all periods
        total_sl = 0
        for pname in periods:
            sl0 = all_results.get(pname, {}).get(asset, {}).get("SL0")
            if sl0:
                total_sl += sl0["n_sl"]
        print(f"  Total stop-loss fires (SL0 across all periods): {total_sl}")

        # Scoring: vs SL0 on NAV (+1) and max_dd (+0.5)
        scores = {v: [] for v in sl_vars}
        for pname in periods:
            pr  = all_results.get(pname, {}).get(asset, {})
            sl0 = pr.get("SL0")
            if sl0 is None: continue
            for v in sl_vars:
                m = pr.get(v)
                if m is None: continue
                nav_win = m["final_nav"] > sl0["final_nav"]
                dd_win  = m["max_dd"]   > sl0["max_dd"]   # less negative = better
                scores[v].append(nav_win + 0.5 * dd_win)

        print(f"\n  {'Variant':<32}  {'Score/1.5':>9}  {'Win%vsS0':>9}  "
              f"{'Periods w/data':>14}")
        for v in sl_vars:
            sc = scores[v]
            if not sc:
                print(f"  {REENTRY_LABELS[v]:<32}  {'n/a':>9}")
                continue
            avg = float(np.mean(sc))
            wp  = 100 * sum(s > 0 for s in sc) / len(sc)
            best_mark = "◄ BEST" if (avg == max(
                float(np.mean(scores[vv])) for vv in sl_vars if scores[vv]
            )) else ""
            print(f"  {REENTRY_LABELS[v]:<32}  {avg:>9.2f}  {wp:>8.0f}%  "
                  f"{len(sc):>14}  {best_mark}")

        # Verdict
        scored = [(v, float(np.mean(scores[v]))) for v in sl_vars if scores[v]]
        if not scored:
            print(f"\n  [No data — cannot recommend]")
            continue
        best_v, best_s = max(scored, key=lambda x: x[1])
        print(f"\n  Verdict for {asset}:")
        if total_sl == 0:
            print(f"    ✓ Stop at {sl_desc} NEVER fires in any tested period.")
            print(f"      The TF2+V-Gate signal exits already provide equivalent protection.")
            print(f"      No re-entry criterion is needed — position is never stopped out.")
        elif best_s <= 0.6:
            print(f"    ⚠ No variant consistently beats SL0 (best avg score: {best_s:.2f}/1.5).")
            print(f"      Standard immediate re-entry (SL0) is the simplest acceptable default.")
        else:
            print(f"    ✓ [{best_v}] {REENTRY_LABELS[best_v]}")
            print(f"      Avg score vs SL0: {best_s:.2f}/1.5 across {len(scores[best_v])} periods.")

    print(f"\n{'═'*W}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═" * 88)
    print("  STOP-LOSS RE-ENTRY CRITERIA ANALYSIS")
    print("  BTC (trail −7%) | MSTR (fixed −3%) | MSTU (fixed −10%)")
    print("  TF2+V-Gate entry/exit signals; stop on execution-asset price")
    print("═" * 88)
    print("\nStop-loss levels in current strategy:")
    for asset, cfg in ASSET_CONFIG.items():
        sl_desc = (f"trailing −{cfg['sl_pct']*100:.0f}% from running peak"
                   if cfg["sl_type"] == "trailing"
                   else f"fixed −{cfg['sl_pct']*100:.0f}% from entry price")
        print(f"  {asset:<6}: {sl_desc}")
    print("\nRe-entry variants tested (applied ONLY after a stop-loss exit):")
    for v, lbl in REENTRY_LABELS.items():
        print(f"  {v}: {lbl}")
    print()

    df_raw_cache: dict = {}
    all_results: dict  = {}

    for pname, (start, end, fs) in PERIODS.items():
        print(f"\n{'─'*70}")
        print(f"  PERIOD: {pname}  ({start} → {end})")
        print(f"{'─'*70}")
        pr = run_period(pname, start, end, fs, df_raw_cache)
        all_results[pname] = pr
        print_full_period(pname, pr)

    # ── Cross-period summary ───────────────────────────────────────────────
    print_cross_period_summary(all_results)

    # ── Per-stop-loss-event drill-down ─────────────────────────────────────
    print(f"\n{'═'*88}")
    print("  STOP-LOSS EVENT DETAIL — what happens after each stop fires")
    print(f"{'═'*88}")
    for pname in ["Bull (Sep24→Sep25)", "Bear (Jun25→May26)",
                  "OOS (Sep25→May26)", "Full (Jun24→May26)"]:
        if pname not in all_results: continue
        for asset in ["BTC", "MSTR", "MSTU"]:
            if asset in all_results[pname]:
                print_sl_trade_detail(all_results[pname], pname, asset)

    # ── Recommendations ───────────────────────────────────────────────────
    print_recommendation(all_results)
    print("\n✓ Analysis complete.")
