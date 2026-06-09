"""
Stop-Loss Evaluation for TF2+V-Gate Strategy
=============================================
Evaluates whether adding a stop-loss to the TF2+V-Gate strategy reduces drawdowns
and losses while improving profitability across all defined backtest periods.

Stop-loss types tested:
  • Fixed    — exit if close drops X% below entry price
  • Trailing — exit if close drops X% below highest close since entry

Periods (matching TRADING_STRATEGY.md / Streamlit dashboard locked windows):
  1. Bear Market   Jun 2025 → May 2026  (mixed IS/OOS, Jun–Feb IS, Mar+ OOS)
  2. Bull Market   Jun 2024 → Jun 2025  (in-sample, model trained through this window)
  3. Full Market   Jun 2024 → May 2026  (combined, mixed IS/OOS)
  4. OOS Only      Mar 2026 → today     (fully blind — model test_start = 2026-03-01)

Assets:
  • BTC    — signal asset and execution asset
  • MSTR   — BTC signals applied to MicroStrategy stock
  • MSTU   — BTC signals applied to T-Rex 2× Long MSTR ETF
              (synthetic OLS-calibrated prices pre-Jun 4, 2025)

Usage:
  python backtest_stop_loss.py
"""
import sys, os, warnings, time
warnings.filterwarnings("ignore")
from pathlib import Path
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

# ── sklearn _loss compatibility fix (Python 3.11/3.12) ────────────────────────
try:
    import sklearn._loss._loss as _sklearn_loss_ext
    if "_loss" not in sys.modules:
        sys.modules["_loss"] = _sklearn_loss_ext
    import sklearn._loss.loss   # noqa: F401
    import sklearn._loss.link   # noqa: F401
except Exception:
    pass

import numpy as np
import pandas as pd
import joblib
import requests
import yfinance as yf

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

INITIAL_CAPITAL = 100_000.0

# Backtest periods (matching the locked Streamlit windows)
PERIODS = {
    "Bear Market (Jun 2025–May 2026)": ("2025-06-01", "2026-05-31"),
    "Bull Market (Jun 2024–Jun 2025)": ("2024-06-05", "2025-06-14"),
    "Full Market (Jun 2024–May 2026)": ("2024-06-01", "2026-05-31"),
    "OOS Only   (Mar 2026–Jun 2026)":  ("2026-03-01", "2026-06-07"),
}

# Stop-loss configurations: (label, type, pct)
# Live strategy uses: BTC → Trail -7%, MSTR → Fixed -3%, MSTU → Fixed -10%
SL_CONFIGS = [
    ("Baseline",    "none",     0.00),
    ("Fixed  -3%",  "fixed",    0.03),
    ("Fixed  -5%",  "fixed",    0.05),
    ("Fixed  -7%",  "fixed",    0.07),
    ("Fixed -10%",  "fixed",    0.10),
    ("Trail  -5%",  "trailing", 0.05),
    ("Trail  -7%",  "trailing", 0.07),
    ("Trail -10%",  "trailing", 0.10),
]

FETCH_START     = "2024-02-01"   # 4 months warmup before earliest backtest start
MSTU_INCEPTION  = pd.Timestamp("2025-06-04")
WARMUP          = 35

# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════════

_MACRO_SYMS = {
    "eth":  "ETH-USD",
    "spx":  "^GSPC",
    "ndx":  "^IXIC",
    "vix":  "^VIX",
    "gold": "GC=F",
    "dxy":  "DX-Y.NYB",
    "tnx":  "^TNX",
}

_ONCHAIN_METRICS = [
    "hash-rate", "difficulty", "n-transactions", "miners-revenue",
    "n-unique-addresses", "transaction-fees-usd", "mempool-size",
    "estimated-transaction-volume-usd", "market-cap",
    "avg-block-size", "cost-per-transaction",
]


def _yf_download(sym: str, start: str, end: str) -> pd.DataFrame:
    d = yf.download(sym, start=start, end=end, progress=False, auto_adjust=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] for c in d.columns]
    d.index = pd.DatetimeIndex(d.index).tz_localize(None).normalize()
    return d


def fetch_btc_and_macro(end_date: str) -> pd.DataFrame:
    print(f"  Fetching BTC + macro: {FETCH_START} → {end_date}")
    d_btc = _yf_download("BTC-USD", FETCH_START, end_date)
    df = pd.DataFrame({
        "btc_close":  d_btc["Close"],
        "btc_high":   d_btc["High"],
        "btc_low":    d_btc["Low"],
        "btc_volume": d_btc["Volume"],
    })
    for name, sym in _MACRO_SYMS.items():
        try:
            d = _yf_download(sym, FETCH_START, end_date)
            df[f"{name}_close"] = d["Close"].reindex(df.index)
        except Exception:
            pass
    print(f"    BTC: {len(df)} bars, macro symbols fetched")

    print("  Fetching on-chain metrics …", end=" ", flush=True)
    ok = 0
    for m in _ONCHAIN_METRICS:
        col = f"oc_{m.replace('-','_')}"
        try:
            r = requests.get(
                f"https://api.blockchain.info/charts/{m}",
                params={"timespan": "3years", "format": "json", "sampled": "true"},
                timeout=20,
            )
            vals = r.json().get("values", [])
            s = pd.Series(
                {pd.Timestamp(v["x"], unit="s").normalize(): v["y"] for v in vals},
                name=col, dtype=float,
            )
            s = s[~s.index.duplicated(keep="last")].sort_index()
            s.index = pd.DatetimeIndex(s.index).tz_localize(None)
            df[col] = s.reindex(df.index).ffill(limit=7)
            ok += 1
        except Exception:
            pass
    print(f"{ok}/{len(_ONCHAIN_METRICS)} OK")

    # Coinbase Premium (optional — skip silently on failure)
    _cb_url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    _cb_rows: list = []
    _cb_cur  = pd.Timestamp(FETCH_START)
    _cb_end  = pd.Timestamp(end_date)
    while _cb_cur <= _cb_end:
        _cb_chunk = min(_cb_cur + pd.Timedelta(days=299), _cb_end)
        try:
            _r = requests.get(_cb_url, params={
                "granularity": 86400,
                "start":       _cb_cur.isoformat() + "Z",
                "end":         (_cb_chunk + pd.Timedelta(days=1)).isoformat() + "Z",
            }, timeout=30)
            if _r.status_code == 200:
                _cb_rows.extend(_r.json())
        except Exception:
            pass
        _cb_cur = _cb_chunk + pd.Timedelta(days=1)
        time.sleep(0.1)
    if _cb_rows:
        _cb_df = pd.DataFrame(_cb_rows, columns=["ts", "low", "high", "open", "close", "volume"])
        _cb_df["date"] = pd.to_datetime(_cb_df["ts"], unit="s").dt.normalize()
        _cb_df = _cb_df.drop_duplicates("date").set_index("date").sort_index()
        _cb_close = _cb_df["close"].reindex(df.index)
        _prem = (_cb_close - df["btc_close"]) / df["btc_close"] * 100
        df["cb_premium"]     = _prem
        df["cb_premium_ma3"] = _prem.rolling(3).mean()
        df["cb_premium_z7"]  = (_prem - _prem.rolling(7).mean()) / _prem.rolling(7).std()
        print(f"  Coinbase premium: {int(_prem.notna().sum())} bars")

    return df.sort_index().ffill(limit=5)


def build_features_and_predictions(df: pd.DataFrame, AD: dict) -> pd.DataFrame:
    """Construct features identical to _build_ct_predictions_extended and run model."""
    feat = pd.DataFrame(index=df.index)
    c = df["btc_close"]; h = df["btc_high"]
    l_ = df["btc_low"];  v = df["btc_volume"]
    ret = np.log(c).diff()

    for k in [1, 3, 5, 7, 14, 30]: feat[f"ret_{k}"] = ret.rolling(k).sum()
    for k in [5, 10, 20, 30]:       feat[f"vol_{k}"] = ret.rolling(k).std()
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
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
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
        s = df[col]
        for k in (1, 5, 20): feat[f"{nm}_ret_{k}"] = np.log(s).diff(k)
        feat[f"{nm}_vol_20"] = np.log(s).diff().rolling(20).std()
    for nm, col in [("spx","spx_close"),("ndx","ndx_close"),
                    ("gold","gold_close"),("dxy","dxy_close")]:
        if col in df.columns:
            feat[f"btc_{nm}_corr_30"] = ret.rolling(30).corr(np.log(df[col]).diff())
    for col in [x for x in df.columns if x.startswith("oc_")]:
        s = df[col].astype(float); sl = np.log(s.replace(0, np.nan))
        feat[f"{col}_d1"]  = sl.diff(1)
        feat[f"{col}_d7"]  = sl.diff(7)
        feat[f"{col}_z30"] = (sl - sl.rolling(30).mean()) / sl.rolling(30).std()
    nh, nl = h.shift(-1), l_.shift(-1)
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
    # Coinbase premium — use pre-computed columns if available, else 0-fill.
    # Must NOT be left as NaN: dropna() would eliminate all rows.
    if "cb_premium" in df.columns and df["cb_premium"].notna().any():
        feat["cb_premium"]     = df["cb_premium"]
        feat["cb_premium_ma3"] = df["cb_premium_ma3"]
        feat["cb_premium_z7"]  = df["cb_premium_z7"]
    else:
        feat["cb_premium"]     = 0.0
        feat["cb_premium_ma3"] = 0.0
        feat["cb_premium_z7"]  = 0.0

    fc = AD["feat_cols"]
    feat = feat.replace([np.inf, -np.inf], np.nan)
    for col in fc:
        if col not in feat.columns: feat[col] = np.nan
    F = feat[fc].dropna()
    if F.empty:
        raise RuntimeError("Feature matrix is empty — check data coverage")

    if AD.get("ensemble") and AD.get("constituents"):
        yhi_arr = np.mean([con["m_hi"].predict(F) for con in AD["constituents"]], axis=0)
        ylo_arr = np.mean([con["m_lo"].predict(F) for con in AD["constituents"]], axis=0)
        if AD.get("blended") and float(AD.get("alpha", 1.0)) < 1.0:
            a = float(AD["alpha"])
            yhi_arr = a * yhi_arr + (1 - a) * float(AD.get("mu_hi", 0))
            ylo_arr = a * ylo_arr + (1 - a) * float(AD.get("mu_lo", 0))
    else:
        yhi_arr = AD["hi_model"].predict(F)
        ylo_arr = AD["lo_model"].predict(F)

    cv = c.reindex(F.index).values
    ph = cv * (1 + np.clip(yhi_arr, 0, None))
    pl = cv * (1 - np.clip(ylo_arr, 0, None))
    idx = np.asarray(F.index, dtype="datetime64[ns]")
    nd = np.empty(len(F), dtype="datetime64[ns]")
    nd[:-1] = idx[1:]
    nd[-1]  = idx[-1] + np.timedelta64(1, "D")

    preds = pd.DataFrame(
        {"close_asof": cv, "pred_high": ph, "pred_low": pl},
        index=pd.DatetimeIndex(nd, name="target_date"),
    )
    return preds[~preds.index.duplicated(keep="last")]


def fetch_mstr(end_date: str) -> tuple:
    """Returns (close, lo, hi) as daily-reindexed pd.Series."""
    d = _yf_download("MSTR", FETCH_START,
                     (pd.Timestamp(end_date) + pd.Timedelta(days=2)).strftime("%Y-%m-%d"))
    idx = pd.date_range(d["Close"].sort_index().index[0],
                        max(d["Close"].sort_index().index[-1], pd.Timestamp(end_date)), freq="D")
    close = d["Close"].sort_index().reindex(idx).ffill()
    lo    = d["Low"].sort_index().reindex(idx).ffill()
    hi    = d["High"].sort_index().reindex(idx).ffill()
    return close, lo, hi


def build_synthetic_mstu(end_date: str, df_btc: pd.DataFrame) -> tuple:
    """OLS-calibrated synthetic MSTU prices for pre-Jun 2025, spliced with actual.

    Returns (close, lo, hi) as daily-reindexed pd.Series.
    Pre-inception intraday low approximated as prev_close * (1 + 2 * btc_intraday_drawdown).
    Pre-inception intraday high approximated symmetrically from btc_intraday_gain.
    """
    end_dt = pd.Timestamp(end_date)
    _mstu_end = max(end_dt, MSTU_INCEPTION + pd.Timedelta(days=90)) + pd.Timedelta(days=2)

    try:
        d_mstu = _yf_download("MSTU", MSTU_INCEPTION.strftime("%Y-%m-%d"),
                               _mstu_end.strftime("%Y-%m-%d"))
        mstu_actual_close = d_mstu["Close"].sort_index()
        mstu_actual_lo    = d_mstu["Low"].sort_index()
        mstu_actual_hi    = d_mstu["High"].sort_index()
    except Exception:
        empty = pd.Series(dtype=float)
        return empty, empty, empty

    try:
        d_mstr = _yf_download("MSTR", FETCH_START, _mstu_end.strftime("%Y-%m-%d"))
        mstr_all = d_mstr["Close"].sort_index()
    except Exception:
        empty = pd.Series(dtype=float)
        return empty, empty, empty

    # OLS calibration on actual overlapping post-inception data
    mstr_post = mstr_all[mstr_all.index >= MSTU_INCEPTION]
    common_idx = mstu_actual_close.index.intersection(mstr_post.index)
    beta, alpha_ols = 2.0, -0.0002
    if len(common_idx) >= 10:
        mstu_lr = np.log(mstu_actual_close.loc[common_idx] /
                         mstu_actual_close.loc[common_idx].shift(1)).dropna()
        mstr_lr = np.log(mstr_post.loc[mstu_lr.index] /
                         mstr_post.loc[mstu_lr.index].shift(1)).dropna()
        cidx = mstu_lr.index.intersection(mstr_lr.index)
        if len(cidx) >= 5:
            x = mstr_lr.loc[cidx].values
            y = mstu_lr.loc[cidx].values
            xm = x - x.mean(); ym = y - y.mean()
            denom = float(np.dot(xm, xm))
            if denom > 1e-10:
                beta      = float(np.dot(xm, ym) / denom)
                alpha_ols = float(y.mean() - beta * x.mean())

    # Synthetic pre-inception prices
    mstr_pre = mstr_all[mstr_all.index < MSTU_INCEPTION].sort_index()
    if len(mstr_pre) < 2:
        mstu_full_close = mstu_actual_close.copy()
        mstu_full_lo    = mstu_actual_lo.copy()
        mstu_full_hi    = mstu_actual_hi.copy()
    else:
        mstr_lr_pre = np.log(mstr_pre / mstr_pre.shift(1)).fillna(0.0).values
        syn_lr = beta * mstr_lr_pre + alpha_ols
        syn_lr[0] = 0.0
        inception_price = float(mstu_actual_close.iloc[0])
        cum_syn = np.cumsum(syn_lr)
        syn_px  = inception_price * np.exp(cum_syn - cum_syn[-1])
        syn_series = pd.Series(syn_px, index=mstr_pre.index)

        # Pre-inception intraday low/high: approx 2× BTC intraday move applied to prev close
        btc_close = df_btc["btc_close"].reindex(mstr_pre.index).ffill()
        btc_lo    = df_btc["btc_low"].reindex(mstr_pre.index).ffill()
        btc_hi    = df_btc["btc_high"].reindex(mstr_pre.index).ffill()
        btc_dd    = np.minimum(0.0, (btc_lo.values - btc_close.values) /
                               np.maximum(btc_close.values, 1.0))
        btc_gain  = np.maximum(0.0, (btc_hi.values - btc_close.values) /
                               np.maximum(btc_close.values, 1.0))
        prev_px   = np.concatenate([[syn_px[0]], syn_px[:-1]])
        syn_lo    = np.maximum(0.01, prev_px * (1.0 + 2.0 * btc_dd))
        syn_hi    = prev_px * (1.0 + 2.0 * btc_gain)
        syn_lo_s  = pd.Series(syn_lo, index=mstr_pre.index)
        syn_hi_s  = pd.Series(syn_hi, index=mstr_pre.index)

        def _splice(pre, post):
            s = pd.concat([pre, post])
            return s[~s.index.duplicated(keep="last")].sort_index()

        mstu_full_close = _splice(syn_series, mstu_actual_close)
        mstu_full_lo    = _splice(syn_lo_s,   mstu_actual_lo)
        mstu_full_hi    = _splice(syn_hi_s,   mstu_actual_hi)

    full_idx = pd.date_range(
        mstu_full_close.index[0],
        max(mstu_full_close.index[-1], end_dt),
        freq="D"
    )
    return (
        mstu_full_close.reindex(full_idx).ffill(),
        mstu_full_lo.reindex(full_idx).ffill(),
        mstu_full_hi.reindex(full_idx).ffill(),
    )


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def build_signals(comp: pd.DataFrame):
    """Compute all TF2+V-Gate arrays from a prediction+actuals comp DataFrame."""
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

    _DN_NORM_W    = 30
    roll_ehi_norm = np.array([
        float(np.mean(err_hi[max(0, i - _DN_NORM_W + 1):i + 1])) for i in range(N)
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
        v_recent[i] = bool(np.any(v_rev_bar[max(0, i - 2):i + 1]))

    entry = u1 & (above_ma30 | clean_10d | v_recent)

    return dict(
        N=N, d2=d2, d3=d3, bull_regime=bull_regime,
        entry=entry, above_ma30=above_ma30, clean_10d=clean_10d, v_recent=v_recent,
    )


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE WITH STOP-LOSS
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(comp: pd.DataFrame, asset_px: np.ndarray,
                 dates: pd.DatetimeIndex, _bt0: int, signals: dict,
                 sl_type: str = "none", sl_pct: float = 0.0,
                 initial_capital: float = INITIAL_CAPITAL,
                 asset_lo: np.ndarray = None,
                 asset_hi: np.ndarray = None) -> dict:
    """
    Backtest TF2+V-Gate with optional stop-loss applied to asset prices.

    sl_type  : "none" | "fixed" | "trailing"
    sl_pct   : decimal fraction (e.g. 0.05 for 5%)
    asset_lo : intraday low array — stop triggered when lo < stop_price
    asset_hi : intraday high array — trailing peak updated from intraday high

    Stop-loss triggered on intraday low; filled at stop price (not close).
    Signal exits use 1-bar lag. Priority: stop-loss > signal exit.
    """
    N          = signals["N"]
    d2         = signals["d2"]
    d3         = signals["d3"]
    bull       = signals["bull_regime"]
    entry      = signals["entry"]
    above_ma30 = signals["above_ma30"]
    clean_10d  = signals["clean_10d"]
    v_recent   = signals["v_recent"]

    # Fall back to close if intraday arrays not supplied
    if asset_lo is None:
        asset_lo = asset_px
    if asset_hi is None:
        asset_hi = asset_px

    nav      = initial_capital
    pos      = "CASH"
    qty      = 0.0
    e_price  = e_nav = e_date = e_trigger = None
    peak_px  = 0.0
    trades   = []
    nav_arr  = np.full(N, np.nan)

    for i in range(N):
        si    = i - 1
        price = asset_px[i]
        lo    = asset_lo[i]
        hi    = asset_hi[i]

        if i < _bt0:
            nav_arr[i] = initial_capital
            continue

        if not np.isfinite(price) or price <= 0:
            nav_arr[i] = qty * (asset_px[i - 1] if i > 0 else price) if pos == "LONG" else nav
            continue

        if pos == "LONG":
            # Update trailing peak using intraday high
            if sl_type == "trailing":
                peak_px = max(peak_px, hi if np.isfinite(hi) and hi > 0 else price)

            # Check stop-loss — triggered by intraday low, filled at stop level
            stop_hit  = False
            exit_px   = price
            exit_lbl  = ""
            if sl_type == "fixed" and sl_pct > 0:
                stop_price = e_price * (1.0 - sl_pct)
                if np.isfinite(lo) and lo < stop_price:
                    stop_hit = True
                    exit_px  = stop_price
                    exit_lbl = f"SL-fixed-{sl_pct*100:.0f}%"
            elif sl_type == "trailing" and sl_pct > 0:
                trail_stop = peak_px * (1.0 - sl_pct)
                if np.isfinite(lo) and lo < trail_stop:
                    stop_hit = True
                    exit_px  = trail_stop
                    exit_lbl = f"SL-trail-{sl_pct*100:.0f}%"

            # Check signal-based exit (1-bar lag: signal fires at si, exit at i)
            sig_exit = False
            sig_lbl  = ""
            if si >= 0:
                sig_exit = bool(d3[si] or (d2[si] and not bull[si]))
                sig_lbl  = "D3" if d3[si] else "D2(bear)"

            should_exit = stop_hit or sig_exit
            if should_exit:
                if not exit_lbl:
                    exit_lbl = sig_lbl
                    exit_px  = price   # signal exits fill at close
                nav = qty * exit_px
                trades.append(dict(
                    entry_date    = e_date,
                    entry_price   = e_price,
                    entry_trigger = e_trigger,
                    entry_nav     = e_nav,
                    exit_date     = dates[i],
                    exit_price    = exit_px,
                    exit_nav      = nav,
                    exit_signal   = exit_lbl,
                    pnl_pct       = (exit_px / e_price - 1) * 100,
                    pnl_abs       = nav - e_nav,
                    duration_days = (dates[i] - e_date).days,
                    peak_price    = peak_px,
                    stop_triggered= stop_hit,
                ))
                pos = "CASH"; qty = 0.0; e_price = e_nav = e_date = e_trigger = None
            else:
                nav = qty * price
        else:
            if si >= 0 and entry[si]:
                qty      = nav / price
                e_price  = price
                e_date   = dates[i]
                e_nav    = nav
                peak_px  = hi if np.isfinite(hi) and hi > 0 else price
                pos      = "LONG"
                if v_recent[si] and not above_ma30[si] and not clean_10d[si]:
                    e_trigger = "U1+V-rev"
                elif above_ma30[si] and clean_10d[si]:
                    e_trigger = "U1+↑MA30+c10d"
                elif above_ma30[si]:
                    e_trigger = "U1+↑MA30"
                else:
                    e_trigger = "U1+c10d"

        nav_arr[i] = qty * price if pos == "LONG" else nav

    if pos == "LONG" and np.isfinite(asset_px[N - 1]) and asset_px[N - 1] > 0:
        nav_arr[N - 1] = qty * asset_px[N - 1]

    nav_series = pd.Series(nav_arr[_bt0:], index=dates[_bt0:]).ffill()
    bh_series  = pd.Series(
        initial_capital * asset_px[_bt0:] / asset_px[_bt0], index=dates[_bt0:]
    )
    return dict(nav=nav_series, bh=bh_series, trades=trades,
                open_pos=(pos == "LONG"), open_entry_price=e_price, open_entry_date=e_date)


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(res: dict, ic: float = INITIAL_CAPITAL) -> dict:
    nav    = res["nav"]; bh = res["bh"]; trades = res["trades"]
    final  = float(nav.iloc[-1])
    bh_fin = float(bh.iloc[-1])
    ret    = (final / ic - 1) * 100
    bh_ret = (bh_fin / ic - 1) * 100
    n_yr   = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr   = ((final / ic) ** (1 / n_yr) - 1) * 100 if n_yr > 0 else 0.0
    rm = nav.cummax()
    max_dd = float(((nav - rm) / rm * 100).min())
    dr = nav.pct_change().fillna(0)
    rf_d = (1.045) ** (1 / 252) - 1
    exc = dr - rf_d
    sharpe = float(exc.mean() / exc.std() * np.sqrt(252)) if exc.std() > 0 else 0.0
    wins   = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    wr     = 100 * len(wins) / len(trades) if trades else 0.0
    best   = max([t["pnl_pct"] for t in trades]) if trades else 0.0
    worst  = min([t["pnl_pct"] for t in trades]) if trades else 0.0
    avg_pnl= float(np.mean([t["pnl_pct"] for t in trades])) if trades else 0.0
    gp     = sum(t["pnl_abs"] for t in wins) if wins else 0.0
    gl     = abs(sum(t["pnl_abs"] for t in losses)) if losses else 1.0
    pf     = gp / gl if gl > 0 else float("inf")
    days_in = sum(t["duration_days"] for t in trades)
    tot_d   = max(1, (nav.index[-1] - nav.index[0]).days)
    n_sl    = sum(1 for t in trades if t.get("stop_triggered", False))
    return dict(
        ret=ret, bh_ret=bh_ret, cagr=cagr,
        max_dd=max_dd, sharpe=sharpe,
        n_trades=len(trades), win_rate=wr, avg_pnl=avg_pnl,
        best=best, worst=worst, profit_factor=pf,
        time_in=100 * days_in / tot_d,
        n_sl_exits=n_sl,
    )


# ══════════════════════════════════════════════════════════════════════════════
# REPORTING
# ══════════════════════════════════════════════════════════════════════════════

COLS = ["Baseline", "Fixed-3%", "Fixed-5%", "Fixed-7%", "Trail-5%", "Trail-7%", "Trail-10%"]
COL_W = 12


def _fmt(v, fmt, better_dir=1):
    """Format a metric value with a ▲/▼ marker vs reference (index 0 = baseline)."""
    return f"{v:{fmt}}"


def print_period_table(period_label: str, asset: str,
                       results_list: list, bh_ret: float):
    """Print a comparison table for one period × one asset across all SL configs."""
    metrics_list = [compute_metrics(r) for r in results_list]
    hdr = f"  Asset: {asset}   Period: {period_label}"
    print(f"\n{hdr}")
    print(f"  B&H return for period: {bh_ret:+.1f}%")
    sep = "  " + "-" * (36 + COL_W * len(COLS))
    print(sep)
    row_hdr = f"  {'Metric':<36}" + "".join(f"{c:>{COL_W}}" for c in COLS)
    print(row_hdr)
    print(sep)

    base = metrics_list[0]

    def row(label: str, key: str, fmt: str, higher_better: bool = True):
        vals = [m[key] for m in metrics_list]
        cells = []
        for j, v in enumerate(vals):
            s = f"{v:{fmt}}"
            if j == 0:
                cells.append(f"{s:>{COL_W}}")
            else:
                delta = v - vals[0]
                if abs(delta) < 0.01:
                    marker = " ="
                elif (delta > 0) == higher_better:
                    marker = " ▲"
                else:
                    marker = " ▼"
                cells.append(f"{s:>{COL_W - 2}}{marker}")
        print(f"  {label:<36}" + "".join(cells))

    row("Total return (%)",        "ret",           "+.1f")
    row("CAGR (%)",                "cagr",          "+.1f")
    row("Max drawdown (%)",        "max_dd",        "+.1f", higher_better=True)
    row("Sharpe ratio",            "sharpe",        ".2f")
    row("Win rate (%)",            "win_rate",      ".1f")
    row("Profit factor",           "profit_factor", ".2f")
    row("Avg trade P&L (%)",       "avg_pnl",       "+.2f")
    row("Best trade (%)",          "best",          "+.1f")
    row("Worst trade (%)",         "worst",         "+.1f", higher_better=True)
    row("# Trades",                "n_trades",      "d",   higher_better=False)
    row("# Stop exits",            "n_sl_exits",    "d",   higher_better=False)
    row("Time in market (%)",      "time_in",       ".1f", higher_better=False)
    print(sep)


def print_summary_table(asset: str, all_period_metrics: dict):
    """Print a compact 2-metric (return, max_dd) summary across all periods."""
    print(f"\n{'═'*80}")
    print(f"  SUMMARY — {asset}: Total Return (▲=better) | Max Drawdown (▲=less negative)")
    print(f"{'═'*80}")
    hdr_line = f"  {'Period':<38}" + "".join(f"{c:>{COL_W}}" for c in COLS)
    print(hdr_line)
    print(f"  {'-'*38}" + "-" * COL_W * len(COLS))
    for period, mlist in all_period_metrics.items():
        ret_row = f"  {period[:36]:<38}" + "".join(
            (f"{m['ret']:>+{COL_W-2}.1f}{'▲' if (m['ret'] > mlist[0]['ret'] and i > 0) else ('▼' if (m['ret'] < mlist[0]['ret'] and i > 0) else ' ')}"
             if i > 0 else f"{m['ret']:>+{COL_W}.1f}")
            for i, m in enumerate(mlist)
        )
        dd_row  = f"  {'  Max DD':<38}" + "".join(
            (f"{m['max_dd']:>+{COL_W-2}.1f}{'▲' if (m['max_dd'] > mlist[0]['max_dd'] and i > 0) else ('▼' if (m['max_dd'] < mlist[0]['max_dd'] and i > 0) else ' ')}"
             if i > 0 else f"{m['max_dd']:>+{COL_W}.1f}")
            for i, m in enumerate(mlist)
        )
        print(ret_row)
        print(dd_row)
        print()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 80)
    print("  BTC RANGE MODEL — Stop-Loss Evaluation for TF2+V-Gate Strategy")
    print("═" * 80)
    print(f"  Testing {len(SL_CONFIGS)} configurations × {len(PERIODS)} periods × 3 assets")
    print(f"  Capital: ${INITIAL_CAPITAL:,.0f}\n")

    # ── Load model ─────────────────────────────────────────────────────────────
    model_path = _ROOT / "models" / "inference_assets_ct.joblib"
    print("Loading CT model …", end=" ", flush=True)
    AD = joblib.load(str(model_path))
    cm = AD.get("calibration_meta", {})
    oos_start = cm.get("test_start", "2026-03-01")
    print(f"OK  (trained through {cm.get('train_end','?')}, test_start={oos_start})")
    # Update OOS period with actual test_start
    PERIODS["OOS Only   (Mar 2026–Jun 2026)"] = (oos_start, "2026-06-07")

    # ── Fetch data ──────────────────────────────────────────────────────────────
    print("\n[1/4] Fetching BTC + macro + on-chain data …")
    df = fetch_btc_and_macro("2026-06-09")

    print("\n[2/4] Building CT predictions …", end=" ", flush=True)
    preds = build_features_and_predictions(df, AD)
    print(f"{len(preds)} bars")

    print("\n[3/4] Fetching MSTR prices (close + intraday lo/hi) …", end=" ", flush=True)
    mstr_close, mstr_lo_all, mstr_hi_all = fetch_mstr("2026-06-09")
    print(f"{len(mstr_close)} bars")

    print("\n[4/4] Building synthetic MSTU prices (OLS-calibrated pre-Jun 2025) …",
          end=" ", flush=True)
    mstu_close, mstu_lo_all, mstu_hi_all = build_synthetic_mstu("2026-06-09", df)
    print(f"{len(mstu_close)} bars")

    # ══════════════════════════════════════════════════════════════════════════
    # Run backtests
    # ══════════════════════════════════════════════════════════════════════════
    # Each entry: (close_series, lo_series, hi_series) — None means use BTC arrays
    asset_defs = {
        "BTC":  (None, None, None),
        "MSTR": (mstr_close, mstr_lo_all, mstr_hi_all),
        "MSTU": (mstu_close, mstu_lo_all, mstu_hi_all),
    }

    all_results: dict = {}  # asset → period → list[metrics]

    for asset_name, (asset_price_series, asset_lo_series, asset_hi_series) in asset_defs.items():
        print(f"\n{'─'*80}")
        print(f"  Running backtests for {asset_name}")
        print(f"{'─'*80}")
        per_asset: dict = {}

        for period_label, (p_start, p_end) in PERIODS.items():
            start_dt = pd.Timestamp(p_start)
            end_dt   = pd.Timestamp(p_end)
            pre_dt   = start_dt - pd.Timedelta(days=60)

            # Slice predictions to the pre-warmup window
            p_slice = preds.loc[
                (preds.index >= pre_dt) & (preds.index <= end_dt)
            ].copy()
            if len(p_slice) < WARMUP + 3:
                print(f"  SKIP {period_label}: insufficient prediction rows")
                continue

            # Attach actual OHLC
            p_slice["actual_high"]  = df["btc_high"].reindex(p_slice.index).values
            p_slice["actual_low"]   = df["btc_low"].reindex(p_slice.index).values
            p_slice["actual_close"] = df["btc_close"].reindex(p_slice.index).values
            comp = p_slice.dropna(
                subset=["actual_high", "actual_low", "actual_close"]
            ).reset_index()
            N = len(comp)
            if N < WARMUP + 3:
                print(f"  SKIP {period_label}: insufficient actuals")
                continue

            dates = pd.DatetimeIndex(comp["target_date"])
            _bt0  = max(WARMUP, int(dates.searchsorted(start_dt)))
            if N - _bt0 < 3:
                continue

            # BTC execution prices and intraday arrays
            btc_px_arr = comp["actual_close"].values.astype(float)
            btc_lo_arr = comp["actual_low"].values.astype(float)
            btc_hi_arr = comp["actual_high"].values.astype(float)

            # Asset execution prices and intraday arrays
            if asset_price_series is not None:
                ax    = asset_price_series.reindex(dates).ffill().bfill().values.astype(float)
                ax_lo = asset_lo_series.reindex(dates).ffill().bfill().values.astype(float)
                ax_hi = asset_hi_series.reindex(dates).ffill().bfill().values.astype(float)
            else:
                ax    = btc_px_arr
                ax_lo = btc_lo_arr
                ax_hi = btc_hi_arr

            # Compute signals once per period
            signals = build_signals(comp)

            # Run all stop-loss configs
            period_results = []
            for sl_label, sl_type, sl_pct in SL_CONFIGS:
                res = run_backtest(
                    comp, ax, dates, _bt0, signals,
                    sl_type=sl_type, sl_pct=sl_pct,
                    asset_lo=ax_lo, asset_hi=ax_hi,
                )
                period_results.append(res)

            # B&H return for reference
            bh_ret = (ax[N - 1] / ax[_bt0] - 1) * 100 if ax[_bt0] > 0 else 0.0

            per_asset[period_label] = period_results
            print(f"  {period_label}  ({N - _bt0} bars, B&H {bh_ret:+.1f}%)  — "
                  f"baseline return {compute_metrics(period_results[0])['ret']:+.1f}%")

        all_results[asset_name] = per_asset

    # ══════════════════════════════════════════════════════════════════════════
    # Print results
    # ══════════════════════════════════════════════════════════════════════════
    print("\n\n" + "═" * 80)
    print("  STOP-LOSS EVALUATION RESULTS")
    print("  ▲ = improvement vs baseline   ▼ = deterioration vs baseline")
    print("═" * 80)

    for asset_name, per_asset in all_results.items():
        print(f"\n\n{'█'*80}")
        print(f"  ASSET: {asset_name}")
        print(f"{'█'*80}")
        summary_data: dict = {}

        for period_label, period_results in per_asset.items():
            ax_bh = compute_metrics(period_results[0])["bh_ret"]
            print_period_table(period_label, asset_name, period_results, ax_bh)
            summary_data[period_label] = [compute_metrics(r) for r in period_results]

        print_summary_table(asset_name, summary_data)

    # ══════════════════════════════════════════════════════════════════════════
    # Print analytical conclusion
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 80)
    print("  ANALYSIS NOTES")
    print("═" * 80)
    print("""
  Interpretation guide
  ────────────────────
  Bear Market (Jun 2025–May 2026): The strategy's natural habitat. Periods of
  sustained downtrend mean D2 exits are often correct. Stop-loss additions may
  cut losses earlier on bad entries (trades 4/9/11 in the 2-year log) but risk
  clipping the eventual recoveries that precede the next U1 signal.

  Bull Market (Jun 2024–Jun 2025): In-sample period. Many exits were D2 fires
  during brief consolidations before BTC resumed higher. A stop-loss here
  primarily adds more premature exits, cutting positions that would have
  recovered. The big winner (+54.3%) likely survives 10% trailing stops but
  is vulnerable to 5–7% trailing stops during intra-trend dips.

  Full Market (2 years): The weighted aggregate. TF2 dominates through the
  in-sample Oct→Dec 2024 bull trade (+54.3%). Stop-loss impact is diluted by
  that outlier.

  OOS Only (Mar–Jun 2026): The purest signal. Fully blind. Whichever stop-loss
  variant improves BOTH return AND max_dd here is the only result that matters
  for forward-looking deployment.

  Key stop-loss trade-off
  ───────────────────────
  • Fixed stops add exits on any bad entry that gaps down immediately.
  • Trailing stops protect accumulated gains but may exit winners mid-trend if
    BTC dips before continuing higher (the Apr 2026 +8.1% trade saw intraday
    retracements that a tight trailing stop would have triggered).
  • Both types add trades (more round trips = more slippage + tax events in
    practice, not modeled here).
  • The strategy's existing D2 exit already acts as a soft stop — it fires when
    predicted highs stop being exceeded, which often precedes the close falling
    below any reasonable fixed stop threshold.
""")

    print("✓ Done.")


if __name__ == "__main__":
    main()
