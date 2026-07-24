"""Generic ticker forecasting core — the config-driven engine.

One implementation of everything that the Gold app's ``gldm_core.py`` does —
data fetch, feature engineering, the macro-sentiment gauge and the trend-
signature signal logic — but parameterised by a :class:`ticker_config.TickerConfig`
so the new apps (SOXX / GRID / XLE / REMX) share a single, tested
codebase and stay visually and behaviourally consistent with Gold.

The primary (signal) asset is always column-prefixed ``px_`` in the merged
frame; macro drivers and traded siblings keep the names given in the config.

Imported by:
  * app/ticker_app.py          the Streamlit UI
  * src/tickers/train_ticker.py model training
  * backtest_ticker.py         strategy backtest + threshold tuning

The BTC and GLDM modules are never imported or modified here.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ticker_config import TickerConfig, get_config  # noqa: F401

warnings.filterwarnings("ignore")

_APP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _APP_DIR.parent


# ── per-ticker paths ─────────────────────────────────────────────────────
def models_dir(cfg: TickerConfig) -> Path:
    p = _REPO_ROOT / "models" / cfg.key.lower()
    p.mkdir(parents=True, exist_ok=True)
    return p


def data_dir(cfg: TickerConfig) -> Path:
    p = _REPO_ROOT / "data" / cfg.key.lower()
    p.mkdir(parents=True, exist_ok=True)
    return p


def model_paths(cfg: TickerConfig) -> dict:
    m = models_dir(cfg)
    return dict(
        hourly=m / "inference_assets_hourly.joblib",
        daily_hl=m / "inference_assets_daily_hl.joblib",
        cone_7d=m / "inference_assets_7d_cone.joblib",
        cone_14d=m / "inference_assets_14d_cone.joblib",
        day_type=m / "inference_assets_3class.joblib",
    )


def cache_paths(cfg: TickerConfig) -> dict:
    d = data_dir(cfg)
    return dict(daily=d / "macro_daily.csv", hourly=d / "macro_hourly.csv",
                backtest=d / "backtest_results.json", sweep=d / "sweep_results.json")


# ════════════════════════════════════════════════════════════════════════
# DATA FETCH — Yahoo chart API (robust behind proxies), same as the Gold app.
# ════════════════════════════════════════════════════════════════════════
_UA = {"User-Agent": "Mozilla/5.0 (compatible; ticker-forecaster/1.0)"}
_YH_HOSTS = ("https://query2.finance.yahoo.com", "https://query1.finance.yahoo.com")


def _chart(symbol: str, interval: str, start: str | None = None,
           range_: str | None = None, timeout: int = 30) -> pd.DataFrame:
    params: dict = {"interval": interval}
    if range_:
        params["range"] = range_
    else:
        p1 = int(pd.Timestamp(start or "2010-01-01").timestamp())
        p2 = int(pd.Timestamp.now(tz="UTC").timestamp())
        params["period1"], params["period2"] = p1, p2
    for host in _YH_HOSTS:
        try:
            r = requests.get(f"{host}/v8/finance/chart/{symbol}",
                             params=params, headers=_UA, timeout=timeout)
            if r.status_code != 200:
                continue
            res = r.json()["chart"]["result"][0]
            ts = res.get("timestamp")
            if not ts:
                continue
            q = res["indicators"]["quote"][0]
            idx = pd.to_datetime(ts, unit="s")
            df = pd.DataFrame({
                "open": q.get("open"), "high": q.get("high"),
                "low": q.get("low"), "close": q.get("close"),
                "volume": q.get("volume"),
            }, index=idx)
            df = df[~df.index.duplicated(keep="last")].sort_index()
            return df.dropna(subset=["close"])
        except Exception:
            continue
    return pd.DataFrame()


def _merge(symbol_map: dict, interval: str, start: str | None = None,
           range_: str | None = None) -> pd.DataFrame:
    """Fetch every symbol concurrently (network-bound) and column-merge them.

    Yahoo chart requests are independent, so a small thread pool cuts a
    multi-symbol fetch from sum-of-latencies to ~one round-trip.  Insertion
    order is preserved so the merged columns are deterministic.
    """
    from concurrent.futures import ThreadPoolExecutor

    def _one(item):
        name, sym = item
        d = _chart(sym, interval, start=start, range_=range_)
        if d.empty:
            return None
        if interval == "1d":
            d.index = pd.to_datetime(d.index).normalize()
            d = d[~d.index.duplicated(keep="last")]
        return d.add_prefix(f"{name}_")

    items = list(symbol_map.items())
    if not items:
        return pd.DataFrame()
    with ThreadPoolExecutor(max_workers=min(8, len(items))) as ex:
        frames = [f for f in ex.map(_one, items) if f is not None]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()


def _primary_map(cfg: TickerConfig) -> dict:
    """name->symbol map for the primary + traded siblings (px_ + extras)."""
    m = {"px": cfg.primary_symbol}
    m.update(cfg.extra_syms)
    return m


def fetch_daily(cfg: TickerConfig, start: str | None = None) -> pd.DataFrame:
    start = start or cfg.fetch_start
    syms = _primary_map(cfg)
    syms.update(cfg.macro_syms)
    df = _merge(syms, "1d", start=start)
    if df.empty:
        return df
    df.index = pd.to_datetime(df.index).normalize()
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna(subset=["px_close"])
    macro_cols = [c for c in df.columns if not c.startswith("px_")]
    df[macro_cols] = df[macro_cols].ffill(limit=5)
    # publisher mode: a published Target Book must be computed from COMPLETED
    # market closes only — trim the in-progress *today* bar Yahoo includes
    # during US market hours (the live apps keep it; only the publisher sets
    # the flag).
    import freshness as _fr
    if _fr.completed_bars_only():
        df = _fr.drop_in_progress_us_bar(df)
    return df


def fetch_hourly(cfg: TickerConfig, range_: str = "730d") -> pd.DataFrame:
    syms = {"px": cfg.primary_symbol}
    syms.update(cfg.macro_syms)
    df = _merge(syms, "1h", range_=range_)
    if df.empty:
        return df
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    df.index = idx.floor("h")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["px_close"])
    macro_cols = [c for c in df.columns if not c.startswith("px_")]
    df[macro_cols] = df[macro_cols].ffill(limit=48)
    return df


# ════════════════════════════════════════════════════════════════════════
# SENTIMENT — generic macro composite (replaces the crypto Fear & Greed idx)
# ════════════════════════════════════════════════════════════════════════
def macro_sentiment(cfg: TickerConfig, df: pd.DataFrame, window: int = 252) -> pd.Series:
    """0-100 sentiment gauge; high = bullish backdrop for this asset.

    Blends the drivers listed in ``cfg.sentiment`` (each already signed so that
    "up = bullish for the asset"), z-scores them, averages, then maps to a
    rolling percentile rank — self-calibrating exactly like the Gold app's gauge.
    """
    def _z(s):
        s = s.astype(float)
        return (s - s.rolling(window, min_periods=20).mean()) / \
               s.rolling(window, min_periods=20).std()

    comp = pd.Series(0.0, index=df.index)
    n = 0
    for col, kind, sign in cfg.sentiment:
        if col not in df or not df[col].notna().any():
            continue
        s = df[col].astype(float)
        if kind == "mom":
            raw = np.log(s.replace(0, np.nan)).diff(20)
        elif kind == "chg":
            raw = s.diff(20)
        else:  # "lvl"
            raw = s
        comp = comp.add(_z(sign * raw), fill_value=0.0)
        n += 1
    if n:
        comp = comp / n
    rank = comp.rolling(window, min_periods=20).apply(
        lambda a: (a[-1] > a).mean() * 100 if np.isfinite(a[-1]) else np.nan, raw=True)
    return rank.rename("sentiment")


# ════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════════════
def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


# macro column names that are yields / levels (diff, not log-return)
_LEVEL_LIKE = {"tnx", "vix"}


def build_daily_features(cfg: TickerConfig, df: pd.DataFrame) -> pd.DataFrame:
    """Daily feature matrix for the H/L, cone and day-type models. All features
    use only information available at the bar close (causal for next bar)."""
    f = pd.DataFrame(index=df.index)
    c = df["px_close"]; h = df["px_high"]; l_ = df["px_low"]
    o = df["px_open"]; v = df.get("px_volume")

    rt = np.log(c).diff()
    for k in (1, 2, 3, 5, 10, 20):
        f[f"ret_{k}d"] = rt.rolling(k).sum()
    for k in (5, 10, 20):
        f[f"vol_{k}d"] = rt.rolling(k).std()
    prev_c = c.shift(1)
    tr = pd.concat([(h - l_), (h - prev_c).abs(), (l_ - prev_c).abs()], axis=1).max(axis=1)
    for k in (5, 14, 20):
        f[f"atr_{k}d"] = tr.rolling(k).mean() / c
    f["range_now"] = (h - l_) / c
    f["range_ma5"] = f["range_now"].rolling(5).mean()
    f["range_ma20"] = f["range_now"].rolling(20).mean()
    f["gap"] = (o - prev_c) / prev_c
    ma20 = c.rolling(20).mean(); ma50 = c.rolling(50).mean()
    f["dist_ma20"] = c / ma20 - 1
    f["dist_ma50"] = c / ma50 - 1
    f["ma20_slope"] = ma20 / ma20.shift(5) - 1
    f["dist_hi_20"] = c / c.rolling(20).max() - 1
    f["dist_lo_20"] = c / c.rolling(20).min() - 1
    f["dist_hi_52w"] = c / c.rolling(252).max() - 1
    f["rsi_14"] = _rsi(c, 14)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    f["macd"] = macd / c
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    sd20 = c.rolling(20).std()
    f["bb_width"] = (4 * sd20) / ma20
    if v is not None:
        lv = np.log(v.replace(0, np.nan))
        f["vol_chg"] = lv.diff()
        f["vol_z20"] = (lv - lv.rolling(20).mean()) / lv.rolling(20).std()

    # ── macro drivers (correct sign is learned by the model) ─────────────
    for nm in cfg.macro_syms:
        col = f"{nm}_close"
        if col not in df or not df[col].notna().any():
            continue
        s = df[col].astype(float)
        if nm in _LEVEL_LIKE:
            f[f"{nm}_chg_1"] = s.diff()
            f[f"{nm}_chg_5"] = s.diff(5)
            f[f"{nm}_chg_20"] = s.diff(20)
            f[f"{nm}_level"] = s
        else:
            ls = np.log(s.replace(0, np.nan))
            f[f"{nm}_ret_1"] = ls.diff()
            f[f"{nm}_ret_5"] = ls.diff(5)
            f[f"{nm}_ret_20"] = ls.diff(20)
            # 20d rolling correlation of the asset vs this driver (regime read)
            f[f"{nm}_corr20"] = rt.rolling(20).corr(ls.diff())

    f["sentiment"] = macro_sentiment(cfg, df)
    f["sent_chg_5"] = f["sentiment"].diff(5)

    mth = df.index.month
    f["month_sin"] = np.sin(2 * np.pi * mth / 12)
    f["month_cos"] = np.cos(2 * np.pi * mth / 12)
    dow = df.index.dayofweek
    f["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    f["dow_cos"] = np.cos(2 * np.pi * dow / 5)
    return f.replace([np.inf, -np.inf], np.nan)


def build_hourly_features(cfg: TickerConfig, df: pd.DataFrame) -> pd.DataFrame:
    """Hourly feature matrix for the next-close model."""
    f = pd.DataFrame(index=df.index)
    c = df["px_close"]; h = df["px_high"]; l_ = df["px_low"]
    v = df.get("px_volume")
    rt = np.log(c).diff()
    for k in (1, 2, 4, 8, 12, 24, 48):
        f[f"ret_{k}h"] = rt.rolling(k).sum()
    for k in (4, 8, 24, 48):
        f[f"vol_{k}h"] = rt.rolling(k).std()
    prev_c = c.shift(1)
    tr = pd.concat([(h - l_), (h - prev_c).abs(), (l_ - prev_c).abs()], axis=1).max(axis=1)
    for k in (4, 12, 24):
        f[f"atr_{k}h"] = tr.rolling(k).mean() / c
    f["range_now"] = (h - l_) / c
    f["range_ma24"] = f["range_now"].rolling(24).mean()
    f["rsi_14"] = _rsi(c, 14)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    f["macd"] = macd / c
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    f["dist_hi_24"] = c / c.rolling(24).max() - 1
    f["dist_lo_24"] = c / c.rolling(24).min() - 1
    if v is not None:
        lv = np.log(v.replace(0, np.nan))
        f["vol_chg_1"] = lv.diff()
        f["vol_z_24"] = (lv - lv.rolling(24).mean()) / lv.rolling(24).std()
    for nm in cfg.macro_syms:
        col = f"{nm}_close"
        if col in df and df[col].notna().any():
            s = df[col]
            f[f"{nm}_ret_1h"] = np.log(s.replace(0, np.nan)).diff()
            f[f"{nm}_ret_24h"] = np.log(s.replace(0, np.nan)).diff(24)
    f["sentiment"] = macro_sentiment(cfg, df, window=24 * 30)
    hr = df.index.hour
    f["hr_sin"] = np.sin(2 * np.pi * hr / 24)
    f["hr_cos"] = np.cos(2 * np.pi * hr / 24)
    dow = df.index.dayofweek
    f["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    f["dow_cos"] = np.cos(2 * np.pi * dow / 5)
    return f.replace([np.inf, -np.inf], np.nan)


# ════════════════════════════════════════════════════════════════════════
# TREND-SIGNATURE SIGNAL LOGIC  (identical structure to the Gold app; the
# thresholds come from the config so each asset self-scales)
# ════════════════════════════════════════════════════════════════════════
def compute_trend_signatures(cfg: TickerConfig, completed: pd.DataFrame) -> dict | None:
    """Trend-signature signals from a frame of completed bars (oldest first).

    ``completed`` columns: close_asof, pred_high, pred_low, actual_high,
    actual_low, target_date. Returns a dict of signal values / trigger flags, or
    None if fewer than 3 completed bars are supplied.
    """
    if completed is None or len(completed) < 3:
        return None
    c = completed["close_asof"].to_numpy(float)
    phi = completed["pred_high"].to_numpy(float)
    plo = completed["pred_low"].to_numpy(float)
    ah = completed["actual_high"].to_numpy(float)
    al = completed["actual_low"].to_numpy(float)
    n = len(c)

    err_hi = (ah - phi) / c * 100
    err_lo = (plo - al) / c * 100

    # Regime-adaptive centering — MUST match backtest_ticker.precompute_signals
    # exactly (60-bar rolling median, min_periods=20): the tuned U1/D2/D1
    # thresholds assume this centering, so a different window here would fire
    # the live signals on different bars than the backtest. Callers must supply
    # enough completed bars (~150) for the tail bars to carry full medians.
    def _center(x):
        s = pd.Series(x)
        med = s.rolling(60, min_periods=20).median()
        med = med.fillna(s.expanding(min_periods=1).median())
        return (s - med).to_numpy()
    err_hi = _center(err_hi)
    err_lo = _center(err_lo)
    hi_break = (err_hi > 0).astype(int)
    lo_break = (err_lo > 0).astype(int)

    w3 = min(3, n)
    err_hi_ma3 = float(np.mean(err_hi[-w3:]))
    err_lo_ma3 = float(np.mean(err_lo[-w3:]))
    hi_breaks_3d = int(np.sum(hi_break[-w3:]))
    lo_breaks_3d = int(np.sum(lo_break[-w3:]))
    ehma3 = np.array([np.mean(err_hi[max(0, i - 2):i + 1]) for i in range(n)])
    elma3 = np.array([np.mean(err_lo[max(0, i - 2):i + 1]) for i in range(n)])

    streak = 0
    if n >= 2:
        for k in range(n - 1, 0, -1):
            d = int(np.sign(c[k] - c[k - 1]))
            if k == n - 1:
                streak = d
            elif d == int(np.sign(streak)) and streak != 0:
                streak += d
            else:
                break

    consec_hi = 0
    exhaustion_active = False
    if n >= 4:
        for k in range(n - 2, -1, -1):
            if hi_break[k]:
                consec_hi += 1
            else:
                break
        exhaustion_active = consec_hi >= 3 and bool(lo_break[-1])

    dn_norm_per = np.array([np.mean(err_hi[max(0, j - 29):j + 1]) for j in range(n)])
    v_rev = np.zeros(n, dtype=bool)
    for j in range(n):
        s3 = max(0, j - 2)
        eh = np.mean(err_hi[s3:j + 1]); el = np.mean(err_lo[s3:j + 1])
        lb3 = int(np.sum(lo_break[s3:j + 1]))
        nrm = max(abs(dn_norm_per[j]), 0.01)
        dns = ((-eh / nrm) * 0.30 + (lb3 / 3.0) * 0.30 +
               (el / max(abs(el), 0.1)) * 0.20 + float(lo_break[j]) * 0.20)
        v_rev[j] = (dns > 0.8) and (float(err_lo[j]) > cfg.v_errlo_min)
    v_recent_gate = bool(np.any(v_rev[-min(3, n):]))

    dn_score_raw = ((-err_hi_ma3 / max(abs(float(np.mean(err_hi[-min(30, n):]))), 0.01)) * 0.30
                    + lo_breaks_3d / 3 * 0.30
                    + err_lo_ma3 / max(abs(err_lo_ma3), 0.1) * 0.20
                    + float(lo_break[-1]) * 0.20)
    last_lo_err = float(err_lo[-1]); last_hi_err = float(err_hi[-1])
    capitulation_signal = dn_score_raw > 0.7 and last_lo_err > cfg.v_errlo_min
    v_reversal_likely = dn_score_raw > 0.8 and last_lo_err > cfg.v_errlo_min

    d1_hist = np.zeros(n, dtype=bool); d2_hist = np.zeros(n, dtype=bool)
    d3_hist = np.zeros(n, dtype=bool); u1_hist = np.zeros(n, dtype=bool)
    hi_run = 0   # consecutive hi_breaks ending at the previous bar
    for i in range(n):
        s = max(0, i - 2)
        eh = np.mean(err_hi[s:i + 1]); el = np.mean(err_lo[s:i + 1])
        hb3 = int(np.sum(hi_break[s:i + 1]))
        lb3 = int(np.sum(lo_break[s:i + 1]))
        d1_hist[i] = (lb3 >= 2) and (el > cfg.d1_errlo_min)
        d2_hist[i] = eh < cfg.d2_errhi_max
        d3_hist[i] = (hi_run >= 3) and bool(lo_break[i])
        u1_hist[i] = (eh > cfg.u1_errhi_min) and (hb3 >= 2)
        hi_run = hi_run + 1 if hi_break[i] else 0
    ma_w = min(20, n)
    ma20_value = float(np.mean(c[-ma_w:]))
    ma20_5d_ago = float(np.mean(c[-(ma_w + 5):-5])) if n >= ma_w + 5 else ma20_value
    above_ma20 = c[-1] > ma20_value
    ma20_slope_pos = ma20_value > ma20_5d_ago
    bull_regime = above_ma20 and ma20_slope_pos
    if n >= 8:
        clean_10d = not bool(np.any(d1_hist[-8:-1] | d2_hist[-8:-1]))
    elif n >= 2:
        clean_10d = not bool(np.any(d1_hist[:-1] | d2_hist[:-1]))
    else:
        clean_10d = False

    d1_triggered = (lo_breaks_3d >= 2) and (err_lo_ma3 > cfg.d1_errlo_min)
    d2_triggered = err_hi_ma3 < cfg.d2_errhi_max
    d3_triggered = exhaustion_active
    u1_triggered = (err_hi_ma3 > cfg.u1_errhi_min) and (hi_breaks_3d >= 2)
    entry_triggered = u1_triggered and (bull_regime or (clean_10d and not above_ma20) or v_recent_gate)

    dn_count = int(d1_triggered) + int(d2_triggered) + int(d3_triggered)
    up_count = int(u1_triggered)
    if dn_count >= 3 or (dn_count >= 2 and v_reversal_likely):
        alert_level = "HIGH_DN"
    elif dn_count == 2:
        alert_level = "ELEVATED_DN"
    elif dn_count == 1 and not u1_triggered:
        alert_level = "WATCH_DN"
    elif entry_triggered:
        alert_level = "STRATEGY_BUY"
    elif up_count >= 1 and dn_count == 0:
        alert_level = "WATCH_UP"
    else:
        alert_level = "NEUTRAL"

    detail_rows = []
    for i in range(max(0, n - 5), n):
        detail_rows.append(dict(
            date=completed["target_date"].iloc[i], close=float(c[i]),
            pred_hi=float(phi[i]), pred_lo=float(plo[i]),
            actual_hi=float(ah[i]), actual_lo=float(al[i]),
            err_hi_pct=float(err_hi[i]), err_hi_ma3=float(ehma3[i]),
            err_lo_pct=float(err_lo[i]), err_lo_ma3=float(elma3[i]),
            hi_break=bool(hi_break[i]), lo_break=bool(lo_break[i]),
        ))

    return dict(
        err_hi_ma3=err_hi_ma3, err_lo_ma3=err_lo_ma3,
        hi_breaks_3d=hi_breaks_3d, lo_breaks_3d=lo_breaks_3d,
        streak=streak, last_lo_err=last_lo_err, last_hi_err=last_hi_err,
        dn_score_raw=dn_score_raw, ma20_value=ma20_value,
        above_ma20=above_ma20, ma20_slope_pos=ma20_slope_pos,
        bull_regime=bull_regime, clean_10d=clean_10d,
        d1_triggered=d1_triggered, d2_triggered=d2_triggered,
        d3_triggered=d3_triggered, u1_triggered=u1_triggered,
        entry_triggered=entry_triggered,
        capitulation_signal=capitulation_signal, v_reversal_likely=v_reversal_likely,
        v_recent_gate=v_recent_gate, exhaustion_active=exhaustion_active,
        consec_hi=consec_hi, alert_level=alert_level,
        dn_count=dn_count, up_count=up_count, detail_rows=detail_rows,
        n_bars=n, as_of_date=completed["target_date"].iloc[-1],
        # Per-bar D1/D2/D3/U1 history (aligned to sig_dates). D1/D2 are the
        # exact arrays the Clean-Breakout gate scans (clean_10d = no D1/D2 in
        # the prior 7 bars); D3/U1 use the same per-bar math as the current-bar
        # triggers. Exposed so the daily-H/L chart can mark where each fired.
        d1_hist=d1_hist, d2_hist=d2_hist, d3_hist=d3_hist, u1_hist=u1_hist,
        sig_dates=pd.to_datetime(completed["target_date"]).to_numpy(),
    )
