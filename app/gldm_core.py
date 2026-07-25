"""GLDM (SPDR Gold MiniShares) forecasting — shared core.

Single source of truth for the GLDM application: data fetch, feature
engineering, the gold-appropriate strategy constants, and the trend-signature
signal logic.  Imported by:

  * app/gldm_hourly_app.py   (the Streamlit UI)
  * src/gldm/train_gldm.py   (model training)
  * backtest_gldm.py         (strategy backtest + threshold tuning)

Design notes — why GLDM differs from BTC
-----------------------------------------
GLDM tracks the spot gold price (~1/100 oz per share, 0.10% expense ratio).
Gold's realised volatility is roughly a QUARTER of Bitcoin's (~13% annualised
vs ~55%), and it trades on the US equity calendar (no weekends, ~6.5h/day),
not 24/7.  The right feature set and the right signal thresholds are therefore
completely different from BTC's:

  * Drop crypto-only inputs (ETH, Coinbase premium, crypto Fear & Greed).
  * Add gold's genuine macro drivers, all with the correct sign:
      - DXY  (US Dollar Index)      → strong INVERSE driver of gold
      - ^TNX (10Y Treasury yield)   → real-yield proxy, INVERSE driver
      - SLV  (silver)               → precious-metals complex co-movement
      - GC=F (gold futures)         → basis / lead relative to the ETF
      - ^VIX (equity vol)           → risk-off gold bid
      - ^GSPC(S&P 500)              → risk sentiment / stock-vs-gold rotation
  * Replace the crypto Fear & Greed index with a purpose-built
    ``gold_macro_sentiment`` composite (dollar weakness + falling real yields
    + risk-off + gold momentum), rank-scaled to 0-100.
  * Scale the trend-signature thresholds and stops to gold's volatility
    (see GLDM_STRATEGY below).
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────
_APP_DIR   = Path(__file__).resolve().parent
_REPO_ROOT = _APP_DIR.parent
GLDM_MODELS_DIR = _REPO_ROOT / "models" / "gldm"
GLDM_DATA_DIR   = _REPO_ROOT / "data" / "gldm"
GLDM_MODELS_DIR.mkdir(parents=True, exist_ok=True)
GLDM_DATA_DIR.mkdir(parents=True, exist_ok=True)

HOURLY_MODEL_GLDM = GLDM_MODELS_DIR / "inference_assets_hourly.joblib"
DAILY_HL_GLDM     = GLDM_MODELS_DIR / "inference_assets_daily_hl.joblib"
CONE_7D_GLDM      = GLDM_MODELS_DIR / "inference_assets_7d_cone.joblib"
CONE_14D_GLDM     = GLDM_MODELS_DIR / "inference_assets_14d_cone.joblib"
DAY_TYPE_GLDM     = GLDM_MODELS_DIR / "inference_assets_3class.joblib"

DAILY_CACHE_CSV   = GLDM_DATA_DIR / "gldm_macro_daily.csv"
HOURLY_CACHE_CSV  = GLDM_DATA_DIR / "gldm_macro_hourly.csv"

# ── Symbols ──────────────────────────────────────────────────────────────
# The traded asset plus its leveraged / high-beta analogs (mirrors how the
# BTC app applies one shared signal to BTC, MSTR and MSTU):
#   GLDM → the core 1x gold position
#   UGL  → ProShares Ultra Gold, 2x daily gold      (analog of MSTU / 2x BTC)
#   GDX  → VanEck Gold Miners ETF, high-beta gold    (analog of MSTR)
#   NUGT → Direxion Daily Gold Miners 2x, leveraged miners (2× the GDX beta)
PRIMARY_SYMBOL = "GLDM"
LEVERAGED_SYMBOLS = {"UGL": "ProShares Ultra Gold (2x)",
                     "GDX": "VanEck Gold Miners ETF",
                     "NUGT": "Direxion Daily Gold Miners 2x"}

# Macro drivers merged into the feature frame (name → Yahoo symbol).
MACRO_SYMS = {
    "gc":  "GC=F",       # gold futures (basis / lead)
    "slv": "SLV",        # silver — precious-metals complex
    "dxy": "DX-Y.NYB",   # US Dollar Index — inverse driver
    "tnx": "^TNX",       # 10Y Treasury yield — real-yield proxy, inverse
    "vix": "^VIX",       # equity vol — risk-off gold bid
    "spx": "^GSPC",      # S&P 500 — risk sentiment
}

# ════════════════════════════════════════════════════════════════════════
# STRATEGY CONSTANTS — tuned for gold's volatility (see backtest_gldm.py)
# ════════════════════════════════════════════════════════════════════════
# ── THE strategy ─────────────────────────────────────────────────────────
# The gold app trades ONE strategy — the BTC-style Divergence Pure-Regime
# system, gold-scaled — applied to two assets (mirroring how BTC trades MSTR &
# MSTU rather than spot):
#     GDX  = VanEck Gold Miners ETF   (high-beta gold — the ~MSTR analog)
#     UGL  = ProShares Ultra Gold 2x  (leveraged gold — the ~MSTU analog)
# GLDM (1x) both supplies the signal AND is traded as the core low-beta sleeve
# (best risk-adjusted expression of the signal: OOS Sharpe ~0.99 at ~-10% MDD).
#
# BTC uses U1/D2 divergence thresholds of ±1.3% of close because a BTC day
# routinely ranges 3-4%.  The GLDM daily H/L model is CAUSAL (features through
# the prior close predict the next bar — see backtest_gldm.build_predictions),
# so its regime-centered divergence error has a genuine forecast scale: 3-bar
# err_hi std ≈ 0.55% OOS.  The config below is the tier-0 pick of the joint
# GDX+UGL frontier sweep (backtest_gldm.py --sweep) re-run 2026-07-25 on the
# lagged (bias-free) divergence engine: GDX +110%/−24% MDD/Sharpe 0.84 vs B&H
# +95%/−47%/0.51, NUGT +217%/−46%/0.77 vs B&H +45%/−74%/0.46.
# See GLDM_TRADING_STRATEGY.md.
STRATEGY_NAME = "Hybrid: Dual-MA Trend + Divergence Pure-Regime"
# ── MIDDLE-PATH ENGINE SPLIT (2026-07) ───────────────────────────────────
# The two engines are regime-complementary, so each sleeve gets the engine
# that suits its character (see GLDM_TRADING_STRATEGY.md §Middle path):
#   • GLDM & UGL (smooth gold trenders) → SOXX-style dual-MA 25/100 crossover
#     on the GLDM parent close, −3% stop.  OOS 2021→now: GLDM +137%/−19%/1.08,
#     UGL +302%/−38%/0.96 — beats both B&H and the divergence variant on
#     return AND Sharpe for these two sleeves.
#   • GDX & NUGT (miners) → Divergence Pure-Regime (below) — clearly better
#     Sharpe/drawdown than dual-MA on the miners, and it carries the chop
#     protection.  Post-2026-07-25 (bias-free engine + re-sweep):
#     GDX +110%/−24%/0.84, NUGT +217%/−46%/0.77.
# Combined equal-weight gold stack OOS: +203% / −29% MDD / Sharpe 1.07
# (pre-fix +214%/−19.8%/1.15 figures are void — same-bar fill).
ENGINE_BY_ASSET = {"GLDM": "dual_ma", "UGL": "dual_ma",
                   "GDX": "divergence", "NUGT": "divergence"}
DUAL_MA_FAST = 25      # fast SMA of the GLDM parent close (SOXX-consistent)
DUAL_MA_SLOW = 100     # slow SMA — the /100 plateau (20-50/100 all similar)


def engine_for(asset: str) -> str:
    return ENGINE_BY_ASSET.get(asset, "divergence")


# ── Divergence engine thresholds (GDX & NUGT; also the signal cards) ─────
# 2026-07-25 post-look-ahead-fix retune: the divergence engine now decides at
# close i−1 and fills at close i (see backtest_gldm.lag_signals), and the
# joint GDX+UGL frontier sweep re-run on that lagged engine + fresh data picks
# a looser D2 exit and stricter D1: U1 0.15→0.10, D2 −0.10→−0.20,
# D1 0.15→0.45 (V unchanged at 1.0).
U1_ERRHI_MIN =  0.10   # U1 entry:  3d-avg centered err_hi > +0.10%  (AND hi_breaks_3d ≥ 2)
D2_ERRHI_MAX = -0.20   # D2 exit:   3d-avg centered err_hi < −0.20%
D1_ERRLO_MIN =  0.45   # D1 exit:   3d-avg centered err_lo > +0.45%  (AND lo_breaks_3d ≥ 2)
V_ERRLO_MIN  =  1.00   # V-reversal capitulation: single-bar low undershoot > 1.00%
FIXED_STOP   =  0.03   # shared fixed stop (−3%) baseline (per-asset overrides below)
TRADEABLE_ASSETS = ["GLDM", "GDX", "UGL", "NUGT"]   # GLDM (1x) = core traded sleeve

# ── REFERENCE trend variant — price-vs-MA filter (not traded) ────────────
# The backtest (backtest_gldm.py, OOS 2021→now) keeps a simple "long above the
# N-day SMA" filter as a labelled reference: it captures ~85-99% of buy &
# hold's return at roughly half the max drawdown with a better Sharpe on
# GLDM, UGL and GDX.  These per-asset MA windows / stops are the
# Sharpe-optimal configs from the frontier sweep (MDD ≤ buy & hold):
MA_WINDOW = 50                                   # default trend-filter window
MA_WINDOW_BY_ASSET = {"GLDM": 50, "UGL": 40, "GDX": 100}

# Fixed stop per traded asset (per its middle-path engine), re-swept
# 2026-07-25 on the lagged divergence engine:
#   • GLDM (dual-MA) → −3% (the −2/−3/−5/none plateau is flat; keep the tuned −3%).
#   • UGL  (dual-MA, 2× gold) → −3%: still the frontier best (+302%/0.96 vs
#     +264%/0.90 stop-less).
#   • GDX  (divergence) → −5%: frontier best on the lagged engine
#     (+110%/0.84 vs −3%'s +105%/0.83) — the plateau is flat, −5% whipsaws less.
#   • NUGT (divergence, 2× miners) → −8%: the lagged-engine frontier prefers
#     loose stops (none +226%/0.79 > −8% +217%/0.77 > −5% +207%/0.76); −8% is
#     kept instead of stop-less to truncate the 2×-fund crash tail (same
#     tail-protection convention as MSTU) at a cost of ~9 pp / 0.02 Sharpe.
STOP_BY_ASSET = {"GLDM": FIXED_STOP, "GDX": 0.05, "UGL": FIXED_STOP, "NUGT": 0.08}


def stop_for(asset: str) -> float:
    """Per-asset fixed stop (fraction; 1.0 = no fixed stop, signal-only exits).
    Single source of truth for the Gold app and the Gold engine so a leveraged
    sibling's looser/absent stop is never rendered or back-tested as the −3%."""
    return STOP_BY_ASSET.get(asset, FIXED_STOP)


def has_stop_for(asset: str) -> bool:
    return stop_for(asset) < 0.999

# Day-type classifier return bands (gold-scaled): next-day close-to-close.
DAY_UP_THRESH   =  0.004    # ≥ +0.4% → Trend Up
DAY_DOWN_THRESH = -0.004    # ≤ −0.4% → Trend Down (else Chop)

# ════════════════════════════════════════════════════════════════════════
# DATA FETCH — Yahoo chart API (robust behind proxies where yfinance's
# curl backend gets connection-reset).  yfinance is used only as a fallback.
# ════════════════════════════════════════════════════════════════════════
_UA = {"User-Agent": "Mozilla/5.0 (compatible; GLDM-forecaster/1.0)"}
_YH_HOSTS = ("https://query2.finance.yahoo.com", "https://query1.finance.yahoo.com")


def _chart(symbol: str, interval: str, start: str | None = None,
           range_: str | None = None, timeout: int = 30) -> pd.DataFrame:
    """Fetch OHLCV for one symbol from Yahoo's chart endpoint.

    Returns a tz-naive UTC-indexed OHLCV frame, or empty on failure.
    """
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
    """Fetch every symbol in ``symbol_map`` (name→ticker) concurrently and
    column-prefix.  The requests are independent, so a thread pool turns a
    sum-of-latencies fetch into ~one round-trip (order preserved)."""
    from concurrent.futures import ThreadPoolExecutor

    def _one(item):
        name, sym = item
        d = _chart(sym, interval, start=start, range_=range_)
        if d.empty:
            return None
        # Daily bars from different exchanges/indices carry different intraday
        # open timestamps (e.g. NYSE 13:30 UTC vs an index at 00:00), which
        # would misalign on a raw concat and produce mostly-NaN columns.
        # Collapse each daily frame to one row per calendar date first.
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


def fetch_daily(start: str = "2015-01-01") -> pd.DataFrame:
    """Daily OHLCV for GLDM + leveraged analogs + macro, aligned on a daily
    grid, macro forward-filled across gaps. Index = tz-naive dates (UTC)."""
    syms = {"gldm": PRIMARY_SYMBOL}
    syms.update({t.lower(): t for t in LEVERAGED_SYMBOLS})   # ticker→ticker
    syms.update(MACRO_SYMS)
    df = _merge(syms, "1d", start=start)
    if df.empty:
        return df
    df.index = pd.to_datetime(df.index).normalize()
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna(subset=["gldm_close"])
    # forward-fill macro / analog columns across holidays (limit ~5 days)
    macro_cols = [c for c in df.columns if not c.startswith("gldm_")]
    df[macro_cols] = df[macro_cols].ffill(limit=5)
    # publisher mode: trim every US bar after the publish day's 7:15-AM-CT
    # anchor basis session, so a published Target Book only ever sees the
    # pre-anchor market close (see app/freshness.py).
    import freshness as _fr
    # See ticker_core.fetch_daily: Yahoo's daily feed can sit a session behind its
    # hourly feed, stranding the gold apps one bar back and tripping the Overall
    # STALE-SIGNALS alert. Rebuild the missing completed session from hourly.
    try:
        if _fr.expected_equity_asof() > pd.DatetimeIndex(df.index).max():
            _h = _merge(syms, "1h", range_="1mo")
            df = _fr.backfill_sessions_from_hourly(df, _h, "gldm_close")
            macro_cols = [c for c in df.columns if not c.startswith("gldm_")]
            df[macro_cols] = df[macro_cols].ffill(limit=5)
        # Still behind? Yahoo is withholding the session on both its feeds, so
        # fall back to an INDEPENDENT provider (Nasdaq's keyless quote API) for
        # the listed tickers. Indices/futures aren't served there and stay
        # ffilled. See app/market_fallback.py for why Yahoo remains primary.
        if _fr.expected_equity_asof() > pd.DatetimeIndex(df.index).max():
            import market_fallback as _mf
            _alt = _mf.merge_frame(syms, start=str(pd.Timestamp(
                pd.DatetimeIndex(df.index).max()).date()))
            df = _fr.merge_missing_sessions(df, _alt, "gldm_close")
            macro_cols = [c for c in df.columns if not c.startswith("gldm_")]
            df[macro_cols] = df[macro_cols].ffill(limit=5)
    except Exception:
        pass                    # best-effort; the audit still flags real staleness
    if _fr.completed_bars_only():
        df = _fr.drop_in_progress_us_bar(df, _fr.publish_anchor_ct())
    return df


def fetch_hourly(range_: str = "730d") -> pd.DataFrame:
    """Hourly OHLCV for GLDM + macro on an hourly grid (US market hours).
    Macro forward-filled with generous headroom for the overnight gap."""
    syms = {"gldm": PRIMARY_SYMBOL}
    syms.update(MACRO_SYMS)
    df = _merge(syms, "1h", range_=range_)
    if df.empty:
        return df
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    df.index = idx.floor("h")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["gldm_close"])
    macro_cols = [c for c in df.columns if not c.startswith("gldm_")]
    df[macro_cols] = df[macro_cols].ffill(limit=48)
    return df


# ════════════════════════════════════════════════════════════════════════
# SENTIMENT — gold macro composite (replaces the crypto Fear & Greed index)
# ════════════════════════════════════════════════════════════════════════
def gold_macro_sentiment(df: pd.DataFrame, window: int = 252) -> pd.Series:
    """0-100 gold sentiment gauge; high = bullish backdrop for gold.

    Blends four genuine gold drivers, each pointed so that "up = bullish gold":
      + dollar weakness            (−DXY 20-period momentum)
      + falling real yields        (−10Y yield 20-period change)
      + risk-off                   (VIX level, standardised)
      + gold's own trend           (GLDM 20-period momentum)
    The blended z-score is converted to a rolling percentile rank (0-100), so
    the reading is self-calibrating like a Fear & Greed index.
    """
    def _z(s):
        s = s.astype(float)
        return (s - s.rolling(window, min_periods=20).mean()) / \
               s.rolling(window, min_periods=20).std()

    comp = pd.Series(0.0, index=df.index)
    n = 0
    if "dxy_close" in df:
        comp = comp.add(_z(-np.log(df["dxy_close"]).diff(20)), fill_value=0.0); n += 1
    if "tnx_close" in df:
        comp = comp.add(_z(-df["tnx_close"].diff(20)), fill_value=0.0); n += 1
    if "vix_close" in df:
        comp = comp.add(_z(df["vix_close"]), fill_value=0.0); n += 1
    if "gldm_close" in df:
        comp = comp.add(_z(np.log(df["gldm_close"]).diff(20)), fill_value=0.0); n += 1
    if n:
        comp = comp / n
    # rolling percentile rank → 0..100
    rank = comp.rolling(window, min_periods=20).apply(
        lambda a: (a[-1] > a).mean() * 100 if np.isfinite(a[-1]) else np.nan, raw=True)
    return rank.rename("gold_sentiment")


# ════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════════════
def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """Daily feature matrix for the H/L, cone and day-type models.

    All features use only information available *at the close of the bar*, so
    predictions for the next bar are causal.
    """
    f = pd.DataFrame(index=df.index)
    c = df["gldm_close"]; h = df["gldm_high"]; l_ = df["gldm_low"]
    o = df["gldm_open"];  v = df.get("gldm_volume")

    rt = np.log(c).diff()
    # multi-horizon returns
    for k in (1, 2, 3, 5, 10, 20):
        f[f"ret_{k}d"] = rt.rolling(k).sum()
    # realised vol
    for k in (5, 10, 20):
        f[f"vol_{k}d"] = rt.rolling(k).std()
    # true range / ATR (fraction of close)
    prev_c = c.shift(1)
    tr = pd.concat([(h - l_), (h - prev_c).abs(), (l_ - prev_c).abs()], axis=1).max(axis=1)
    for k in (5, 14, 20):
        f[f"atr_{k}d"] = tr.rolling(k).mean() / c
    # intraday range
    f["range_now"]  = (h - l_) / c
    f["range_ma5"]  = f["range_now"].rolling(5).mean()
    f["range_ma20"] = f["range_now"].rolling(20).mean()
    # gap (overnight)
    f["gap"] = (o - prev_c) / prev_c
    # momentum / trend structure
    ma20 = c.rolling(20).mean(); ma50 = c.rolling(50).mean()
    f["dist_ma20"] = c / ma20 - 1
    f["dist_ma50"] = c / ma50 - 1
    f["ma20_slope"] = ma20 / ma20.shift(5) - 1
    f["dist_hi_20"] = c / c.rolling(20).max() - 1
    f["dist_lo_20"] = c / c.rolling(20).min() - 1
    f["dist_hi_52w"] = c / c.rolling(252).max() - 1
    # oscillators
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

    # ── Macro drivers (correct sign is learned by the model) ─────────────
    if "dxy_close" in df:
        d = df["dxy_close"]
        f["dxy_ret_1"]  = np.log(d).diff()
        f["dxy_ret_5"]  = np.log(d).diff(5)
        f["dxy_ret_20"] = np.log(d).diff(20)
    if "tnx_close" in df:
        t = df["tnx_close"]
        f["tnx_chg_1"]  = t.diff()
        f["tnx_chg_5"]  = t.diff(5)
        f["tnx_chg_20"] = t.diff(20)
        f["tnx_level"]  = t
    if "slv_close" in df:
        s = df["slv_close"]
        f["slv_ret_1"] = np.log(s).diff()
        f["slv_ret_5"] = np.log(s).diff(5)
        # gold/silver ratio momentum
        gsr = c / s
        f["gsr_chg_5"] = gsr / gsr.shift(5) - 1
    if "gc_close" in df:
        gc = df["gc_close"]
        f["gc_ret_1"] = np.log(gc).diff()
        # futures basis vs ETF (scaled): both track spot gold, divergence = flow signal
        f["gc_basis"] = (np.log(gc).diff() - rt)
    if "vix_close" in df:
        f["vix_level"] = df["vix_close"]
        f["vix_chg_5"] = df["vix_close"].diff(5)
    if "spx_close" in df:
        sp = df["spx_close"]
        f["spx_ret_1"] = np.log(sp).diff()
        f["spx_ret_5"] = np.log(sp).diff(5)
        # 20d correlation of gold vs stocks (regime: hedge vs risk-on)
        f["gold_spx_corr20"] = rt.rolling(20).corr(np.log(sp).diff())

    # sentiment composite
    f["gold_sentiment"] = gold_macro_sentiment(df)
    f["sent_chg_5"] = f["gold_sentiment"].diff(5)

    # calendar / seasonality (gold has month-of-year seasonality)
    mth = df.index.month
    f["month_sin"] = np.sin(2 * np.pi * mth / 12)
    f["month_cos"] = np.cos(2 * np.pi * mth / 12)
    dow = df.index.dayofweek
    f["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    f["dow_cos"] = np.cos(2 * np.pi * dow / 5)

    return f.replace([np.inf, -np.inf], np.nan)


def build_hourly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hourly feature matrix for the next-close model."""
    f = pd.DataFrame(index=df.index)
    c = df["gldm_close"]; h = df["gldm_high"]; l_ = df["gldm_low"]
    v = df.get("gldm_volume")
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
    for nm in ("gc", "slv", "dxy", "tnx", "vix", "spx"):
        col = f"{nm}_close"
        if col in df and df[col].notna().any():
            s = df[col]
            f[f"{nm}_ret_1h"]  = np.log(s.replace(0, np.nan)).diff()
            f[f"{nm}_ret_24h"] = np.log(s.replace(0, np.nan)).diff(24)
    f["gold_sentiment"] = gold_macro_sentiment(df, window=24 * 30)
    hr = df.index.hour
    f["hr_sin"] = np.sin(2 * np.pi * hr / 24)
    f["hr_cos"] = np.cos(2 * np.pi * hr / 24)
    dow = df.index.dayofweek
    f["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    f["dow_cos"] = np.cos(2 * np.pi * dow / 5)
    return f.replace([np.inf, -np.inf], np.nan)


# ════════════════════════════════════════════════════════════════════════
# TREND-SIGNATURE SIGNAL LOGIC
# ════════════════════════════════════════════════════════════════════════
def compute_trend_signatures(completed: pd.DataFrame) -> dict | None:
    """Compute the GLDM trend-signature signals from a frame of completed bars.

    ``completed`` must have columns: close_asof, pred_high, pred_low,
    actual_high, actual_low, target_date — one row per completed daily bar,
    oldest first.  Returns a dict of signal values / trigger flags, or None if
    fewer than 3 completed bars are supplied.

    Signals (gold-scaled thresholds from GLDM_STRATEGY):
      U1  err_hi_ma3 > U1_ERRHI_MIN AND hi_breaks_3d ≥ 2   → bullish pressure
      D1  lo_breaks_3d ≥ 2 AND err_lo_ma3 > D1_ERRLO_MIN   → bearish pressure
      D2  err_hi_ma3 < D2_ERRHI_MAX                        → momentum fading
      D3  first lo_break after ≥3 hi_break streak         → exhaustion canary
      V   dn_score > 0.8 AND single-bar low undershoot > V → capitulation
    Entry gate (Pure Regime): U1 AND (bull_regime OR (clean & below MA20) OR V).
    """
    if completed is None or len(completed) < 3:
        return None
    c   = completed["close_asof"].to_numpy(float)
    phi = completed["pred_high"].to_numpy(float)
    plo = completed["pred_low"].to_numpy(float)
    ah  = completed["actual_high"].to_numpy(float)
    al  = completed["actual_low"].to_numpy(float)
    n = len(c)

    err_hi = (ah - phi) / c * 100          # +ve = high ran hotter than predicted
    err_lo = (plo - al) / c * 100          # +ve = low undershot prediction
    # Regime-adaptive centering — MUST match backtest_gldm.precompute_signals
    # exactly (60-bar rolling median, min_periods=20): the ridge H/L band is
    # structurally asymmetric and drifts, so subtract each error's rolling
    # median before thresholding. Breaks are defined off the centered error so
    # the whole signal system is self-consistent (~50% base). The U1/D2/D1
    # thresholds were tuned in the backtest against THIS centering; a different
    # window here would fire the live signals on different bars than the
    # strategy was tuned on. Callers must supply enough completed-bar history
    # for the tail bars to carry a full 60-bar median (see signatures_asof).
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

    # consecutive close streak
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

    # D3 exhaustion: first lo_break after ≥3 consecutive hi_breaks
    consec_hi = 0
    exhaustion_active = False
    if n >= 4:
        for k in range(n - 2, -1, -1):
            if hi_break[k]:
                consec_hi += 1
            else:
                break
        exhaustion_active = consec_hi >= 3 and bool(lo_break[-1])

    # V-reversal capitulation (per-bar, last 3 bars)
    dn_norm_per = np.array([np.mean(err_hi[max(0, j - 29):j + 1]) for j in range(n)])
    v_rev = np.zeros(n, dtype=bool)
    for j in range(n):
        s3 = max(0, j - 2)
        eh = np.mean(err_hi[s3:j + 1]); el = np.mean(err_lo[s3:j + 1])
        lb3 = int(np.sum(lo_break[s3:j + 1]))
        nrm = max(abs(dn_norm_per[j]), 0.01)
        dns = ((-eh / nrm) * 0.30 + (lb3 / 3.0) * 0.30 +
               (el / max(abs(el), 0.1)) * 0.20 + float(lo_break[j]) * 0.20)
        v_rev[j] = (dns > 0.8) and (float(err_lo[j]) > V_ERRLO_MIN)
    v_recent_gate = bool(np.any(v_rev[-min(3, n):]))

    dn_score_raw = ((-err_hi_ma3 / max(abs(float(np.mean(err_hi[-min(30, n):]))), 0.01)) * 0.30
                    + lo_breaks_3d / 3 * 0.30
                    + err_lo_ma3 / max(abs(err_lo_ma3), 0.1) * 0.20
                    + float(lo_break[-1]) * 0.20)
    last_lo_err = float(err_lo[-1]); last_hi_err = float(err_hi[-1])
    capitulation_signal = dn_score_raw > 0.7 and last_lo_err > V_ERRLO_MIN
    v_reversal_likely   = dn_score_raw > 0.8 and last_lo_err > V_ERRLO_MIN

    # MA20 regime filter (gold trends smoothly → 20-bar is the right horizon)
    d1_hist = np.zeros(n, dtype=bool); d2_hist = np.zeros(n, dtype=bool)
    d3_hist = np.zeros(n, dtype=bool); u1_hist = np.zeros(n, dtype=bool)
    hi_run = 0   # consecutive hi_breaks ending at the previous bar
    for i in range(n):
        s = max(0, i - 2)
        eh = np.mean(err_hi[s:i + 1]); el = np.mean(err_lo[s:i + 1])
        hb3 = int(np.sum(hi_break[s:i + 1]))
        lb3 = int(np.sum(lo_break[s:i + 1]))
        d1_hist[i] = (lb3 >= 2) and (el > D1_ERRLO_MIN)
        d2_hist[i] = eh < D2_ERRHI_MAX
        d3_hist[i] = (hi_run >= 3) and bool(lo_break[i])
        u1_hist[i] = (eh > U1_ERRHI_MIN) and (hb3 >= 2)
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

    d1_triggered = (lo_breaks_3d >= 2) and (err_lo_ma3 > D1_ERRLO_MIN)
    d2_triggered = err_hi_ma3 < D2_ERRHI_MAX
    d3_triggered = exhaustion_active
    u1_triggered = (err_hi_ma3 > U1_ERRHI_MIN) and (hi_breaks_3d >= 2)
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
