"""Streamlit app for live hourly BTC next-close prediction.

Run:
    streamlit run btc_hourly_app.py

The app:
  * pulls the latest hourly BTC + macro + Fear&Greed data
  * builds the same features used at training time
  * applies the saved model and emits a next-hour forecast with 95% CI
  * plots the last N hours of actuals plus the forecast
  * auto-refreshes every REFRESH_SECONDS (default 600 = 10 min)
"""
import os, sys, time, warnings, joblib, requests, json
warnings.filterwarnings("ignore")
from datetime import datetime, timezone, timedelta, date as _date
from pathlib import Path

# Make the repo root importable so `from paths import …` works regardless
# of the cwd from which Streamlit is launched.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
try:
    from paths import (
        HOURLY_MODEL, DAILY_MODEL_CT, CONE_7D_MODEL, CONE_14D_MODEL, DAY_TYPE_MODEL,
        BINANCE_HOURLY_CSV,
        BOOKMARKS_FILE as _BOOKMARKS_PATH, RUNTIME_DIR,
    )
except ImportError:
    # Fallback for deployments where paths.py predates CONE_14D_MODEL
    from paths import (
        HOURLY_MODEL, DAILY_MODEL_CT, CONE_7D_MODEL, DAY_TYPE_MODEL,
        BINANCE_HOURLY_CSV,
        BOOKMARKS_FILE as _BOOKMARKS_PATH, RUNTIME_DIR,
    )
    CONE_14D_MODEL = Path(str(CONE_7D_MODEL)).parent / "inference_assets_14d_cone.joblib"

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go

# ════════════════════════════════════════════════════════════════════════
# CONFIG
ASSETS_PATH      = str(HOURLY_MODEL)
REFRESH_SECONDS  = 60           # auto-refresh interval (1 min — rolling forecast)
LOOKBACK_HOURS   = 24           # how many past hours to show
CACHE_TTL        = 300          # data cache lifetime (seconds)
BAND_PCT         = 0.005        # ±0.5% forecast band (around prediction)
# ════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="BTC Hourly Forecaster", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")
st.title("📈 Bitcoin — Live hourly next-close forecast")
st.caption(
    "Live feed: BTC + ETH + macro (Yahoo) + Fear & Greed (alternative.me). "
    "Model: ridge regression on log-returns. "
    "**Honest framing:** ±3 % accuracy on hourly close is trivial (hit-rate "
    "~100 %); the model's real signal is in **direction accuracy ~54 %** "
    "and **tight CI**."
)

# ──────────────────────────── load model ──────────────────────────────
@st.cache_resource
def load_assets():
    if not os.path.exists(ASSETS_PATH):
        st.error(f"Model artefacts not found at {ASSETS_PATH}.\n"
                 "Run `python train_hourly_model.py` first.")
        st.stop()
    return joblib.load(ASSETS_PATH)

A = load_assets()
model     = A["model"]
sigma     = A["sigma"]
feat_cols = A["feat_cols"]
best_name = A.get("best_name","ridge")


@st.cache_resource
def _training_cutoffs():
    """Return {model_label: train_end_date_or_None} for the 4 artefacts.

    Used by the historical-replay banner to warn the user when their
    picked date falls inside any model's training window — predictions
    for those dates are in-sample fit, not honest out-of-sample forecasts.
    """
    out = {}
    # Hourly: stored as ISO datetime on newer artefacts; fall back to test_start.
    he = A.get("train_end") or A.get("test_start")
    out["hourly close"] = pd.Timestamp(he).normalize() if he else None
    # Daily H/L
    if os.path.exists(str(DAILY_MODEL_CT)):
        try:
            meta = joblib.load(DAILY_MODEL_CT).get("calibration_meta", {})
            out["daily H/L"] = pd.Timestamp(meta.get("train_end")) if meta.get("train_end") else None
        except Exception:
            out["daily H/L"] = None
    # 7-day cone
    if os.path.exists(str(CONE_7D_MODEL)):
        try:
            meta = joblib.load(CONE_7D_MODEL).get("calibration_meta", {})
            out["7-day cone"] = pd.Timestamp(meta.get("train_end")) if meta.get("train_end") else None
        except Exception:
            out["7-day cone"] = None
    # 14-day cone
    if os.path.exists(str(CONE_14D_MODEL)):
        try:
            meta = joblib.load(CONE_14D_MODEL).get("calibration_meta", {})
            out["14-day cone"] = pd.Timestamp(meta.get("train_end")) if meta.get("train_end") else None
        except Exception:
            out["14-day cone"] = None
    # 3-class
    if os.path.exists(str(DAY_TYPE_MODEL)):
        try:
            meta = joblib.load(DAY_TYPE_MODEL).get("calibration_meta", {})
            out["3-class day type"] = pd.Timestamp(meta.get("train_end")) if meta.get("train_end") else None
        except Exception:
            out["3-class day type"] = None
    return out


def render_replay_in_sample_warning(target_date):
    """If `target_date` falls inside any model's training window, show a
    yellow warning explaining the predictions on this date are in-sample
    fit (memorisation), not honest out-of-sample forecasts."""
    if target_date is None:
        return
    td = pd.Timestamp(target_date).normalize()
    affected = []
    for name, end in _training_cutoffs().items():
        if end is not None and td <= end:
            affected.append(f"**{name}** (train ≤ {end.date()})")
    if affected:
        st.warning(
            "⚠️  **In-sample replay.**  You picked **"
            f"{td.date()}** — this date falls inside the training window of: "
            + ", ".join(affected) + ". The predictions and "
            "look-back metrics shown for in-sample dates are MEMORISATION, "
            "not honest forecasts; they will look unrealistically accurate. "
            "Pick a date AFTER each model's `train_end` to see genuine "
            "out-of-sample behaviour."
        )

with st.sidebar:
    st.markdown(
        "**Auto-refresh:** every "
        f"{REFRESH_SECONDS // 60} min. Click **Refresh now** to force.")
    if st.button("Refresh now", use_container_width=True):
        # Clear BOTH caches so retrained joblibs are picked up too — without
        # this, `cache_resource`-decorated loaders (`load_assets`,
        # `_load_cone_7d`, `_load_day_type`) hold onto the previous artefact
        # for the lifetime of the session.
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    st.markdown(
        "_Hourly BTC bars update every hour; macro and F&G update less often._"
    )
    # Model freshness — train_end of each artefact, so users can tell at a
    # glance which version of each model is live in the UI.
    st.markdown("---")
    st.caption("**Model freshness** (`train_end`)")
    for label, end in _training_cutoffs().items():
        end_str = f"`{end.date()}`" if end is not None else "_unknown_"
        st.caption(f"&bull; {label}: {end_str}", unsafe_allow_html=True)

# ───────────────────────── fetch helpers ──────────────────────────────
def _flat(df, name):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df[["Open","High","Low","Close","Volume"]].copy()
    df.columns = [f"{name}_{c.lower()}" for c in df.columns]
    idx = pd.to_datetime(df.index)
    if idx.tz is not None: idx = idx.tz_convert("UTC").tz_localize(None)
    df.index = idx
    return df[~df.index.duplicated(keep="last")].sort_index()

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_data():
    SYMS = {"btc":"BTC-USD","eth":"ETH-USD","spx":"^GSPC","ndx":"^IXIC",
            "vix":"^VIX","gold":"GC=F","dxy":"DX-Y.NYB","tnx":"^TNX"}
    parts = {}
    for name, sym in SYMS.items():
        # yfinance caps 1-hour bars at 730 days; "2y" sits right at that edge
        # and returns empty on Streamlit Cloud.  Try progressively shorter
        # windows until we get data.
        raw = pd.DataFrame()
        for period in ("729d", "180d", "59d"):
            try:
                raw = yf.download(sym, period=period, interval="60m",
                                  progress=False, auto_adjust=False)
                if not raw.empty:
                    break
            except Exception:
                continue
        parts[name] = _flat(raw, name)
    btc = parts["btc"]
    if btc.empty:
        st.error(
            "⚠️ Could not fetch live BTC/market data from Yahoo Finance. "
            "Please click **Refresh now** in the sidebar to retry."
        )
        st.stop()
    grid = pd.date_range(btc.index.min().floor("h"), btc.index.max().floor("h"),
                         freq="h")
    df = pd.DataFrame(index=grid)
    for name, d in parts.items():
        agg = {f"{name}_open":"first", f"{name}_high":"max", f"{name}_low":"min",
               f"{name}_close":"last", f"{name}_volume":"last"}
        d = d.resample("h").agg(agg)
        d[f"{name}_volume"] = d[f"{name}_volume"].replace(0, np.nan)
        d = d.reindex(grid).ffill(limit=168 if name not in ("btc","eth") else 4)
        df = df.join(d)
    df = df.dropna(subset=["btc_close"])

    # Fear & Greed daily, forward-filled to hourly.
    #
    # IMPORTANT (causality): alternative.me's current-day F&G record is
    # re-computed throughout the UTC day. Using the value stamped at
    # 2026-05-21 00:00 UTC at hour 14:00 UTC would be look-ahead, because
    # the value at fetch time reflects information from past 14:00 UTC.
    # We lag by 1 day so hours of date D use D-1's finalised value (this
    # mirrors src/train_hourly_model.py).
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=0", timeout=20).json()
        fng = pd.DataFrame(r["data"])
        fng["dt"]    = pd.to_datetime(fng["timestamp"].astype(int), unit="s").dt.normalize()
        fng["value"] = fng["value"].astype(int)
        fng = fng[["dt","value"]].sort_values("dt").drop_duplicates("dt").set_index("dt")
        fng = fng.shift(1, freq="D")   # ← 1-day causal lag (anti-leak)
        fng_h = fng.reindex(df.index.normalize()).ffill()
        df["fng"] = fng_h["value"].values
    except Exception:
        df["fng"] = 50  # neutral fallback
    df["fng_d24"] = pd.Series(df["fng"].values, index=df.index).diff(24).values
    df["fng_d7"]  = pd.Series(df["fng"].values, index=df.index).diff(24*7).values
    return df

@st.cache_data(ttl=30, show_spinner=False)
def fetch_live_spot():
    """Binance public ticker — true real-time BTC/USDT price (no API key)."""
    try:
        r = requests.get("https://api.binance.com/api/v3/ticker/price",
                         params={"symbol":"BTCUSDT"}, timeout=10)
        return float(r.json()["price"]), datetime.now(timezone.utc)
    except Exception:
        return None, None


@st.cache_data(ttl=60, show_spinner=False)
def fetch_btc_1m():
    """Fetch the last ~25 hours of 1-minute BTC/USDT klines from Binance.

    Makes two paginated requests (1 000 bars each) to cover 25 hours
    (1 500 minutes).  Returns a DataFrame with a UTC tz-naive
    DatetimeIndex and a single 'close' column.  Returns an empty
    DataFrame on any network or parsing error.
    """
    try:
        url = "https://api.binance.com/api/v3/klines"
        # First call: most recent 1 000 bars
        r1 = requests.get(url, params={"symbol": "BTCUSDT",
                                        "interval": "1m", "limit": 1000},
                          timeout=8)
        r1.raise_for_status()
        data = r1.json()
        if not data:
            return pd.DataFrame()
        # Second call: 500 bars ending just before the first batch
        earliest_ms = data[0][0]
        r2 = requests.get(url, params={"symbol": "BTCUSDT",
                                        "interval": "1m", "limit": 500,
                                        "endTime": earliest_ms - 1},
                          timeout=8)
        r2.raise_for_status()
        data = r2.json() + data          # oldest first
        idx = pd.to_datetime([row[0] for row in data],
                             unit="ms", utc=True).tz_localize(None)
        closes = [float(row[4]) for row in data]
        return pd.DataFrame({"close": closes}, index=idx)
    except Exception:
        return pd.DataFrame()

# ─────────────────────── Date bookmarks ────────────────────────────────
RUNTIME_DIR.mkdir(exist_ok=True)
BOOKMARKS_FILE = str(_BOOKMARKS_PATH)


def load_bookmarks():
    """Return dict of {category: [{"date": "YYYY-MM-DD", "label": ""}, ...]}."""
    if not os.path.exists(BOOKMARKS_FILE):
        return {}
    try:
        with open(BOOKMARKS_FILE) as f:
            data = json.load(f)
        # Normalize legacy formats (list of strings) → list of dicts
        for cat, entries in list(data.items()):
            data[cat] = [(e if isinstance(e, dict) else {"date": e, "label": ""})
                         for e in entries]
        return data
    except Exception:
        return {}


def save_bookmarks(data):
    with open(BOOKMARKS_FILE, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def add_bookmark(category, d, label=""):
    data = load_bookmarks()
    cat = data.setdefault(category, [])
    iso = d.isoformat() if isinstance(d, _date) else str(d)
    # Don't duplicate (same date in same category)
    if not any(e["date"] == iso for e in cat):
        cat.append({"date": iso, "label": label or ""})
        cat.sort(key=lambda e: e["date"])
        save_bookmarks(data)
    return data


def delete_bookmark(category, iso_date):
    data = load_bookmarks()
    if category in data:
        data[category] = [e for e in data[category] if e["date"] != iso_date]
        if not data[category]:
            del data[category]
        save_bookmarks(data)
    return data


# ──────── DAILY H/L forecast (7am-CT day boundary = 12:00 UTC) ────────
# Bar D covers [D 12:00 UTC, D+1 12:00 UTC). Indexed by start date D.
ANCHOR_HOUR_UTC = 12  # 7am CDT (summer) / 6am CST (winter)


def _rebucket_12utc(hourly):
    """Group hourly OHLCV into 24h bars starting at ANCHOR_HOUR_UTC.

    Returns bars indexed by start date D. Drops incomplete bars
    (anything other than 24 hours)."""
    if hourly.empty:
        # Return an empty frame with the correct columns and a DatetimeIndex
        # so callers that check .empty or iterate over .index don't crash.
        g = pd.DataFrame(columns=["open","high","low","close","volume"])
        g.index = pd.DatetimeIndex([], name="bar_start")
        return g
    h = hourly.copy()
    h["bucket"] = (h.index - pd.Timedelta(hours=ANCHOR_HOUR_UTC)).normalize()
    g = h.groupby("bucket").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"), n_hours=("close", "size"),
    )
    g = g[g["n_hours"] == 24].drop(columns="n_hours")
    g.index.name = "bar_start"
    return g


def _fetch_yfinance_hourly_fallback():
    """Fetch BTC 1-hour OHLCV from Yahoo Finance (BTC-USD).

    Used as an automatic fallback when the Binance public API is
    unavailable or rate-limiting the Streamlit Cloud server.
    yfinance covers ~730 days of 1-hour bars — enough for all model
    inference lookbacks (max 90 bars).

    Returns a DataFrame with columns [open, high, low, close, volume]
    and a UTC-naive DatetimeIndex named 'ts', or an empty frame on failure.
    """
    try:
        raw = yf.download("BTC-USD", period="730d", interval="60m",
                          progress=False, auto_adjust=False)
        if raw.empty:
            raise ValueError("yfinance returned empty data for BTC-USD")
        # Flatten MultiIndex columns (yfinance sometimes returns them)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] for c in raw.columns]
        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        df.index.name = "ts"
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df
    except Exception:
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        empty.index = pd.DatetimeIndex([], name="ts")
        return empty


@st.cache_data(ttl=600, show_spinner="Fetching BTC hourly from Binance …")
def _fetch_binance_hourly(days_back=None):
    """Return BTC hourly OHLCV with full history.

    Strategy: load `binance_hourly_btc.csv` (saved during training; covers
    2017-08 → save-time), then top-up from the Binance public API for any
    hours after the CSV's last row. This guarantees the daily-forecast
    pipeline can build features for ANY date the historical-tab picker
    allows, while staying fast on the common case (top-up = a few API calls).

    `days_back` is kept for backward compatibility — when set, only the last
    `days_back` days are returned (used by the live `_fetch_daily_raw` if
    the CSV is missing)."""
    CSV_PATH = str(BINANCE_HOURLY_CSV)
    parts = []

    if os.path.exists(CSV_PATH):
        csv_df = pd.read_csv(CSV_PATH, index_col="timestamp_utc", parse_dates=True)
        csv_df.index = csv_df.index.tz_localize(None)
        parts.append(csv_df)
        # Top up: API from the hour AFTER csv's last row → now
        start_ms = int((csv_df.index.max() + pd.Timedelta(hours=1)).timestamp() * 1000)
    else:
        # Fallback: pull ~days_back days from API (covers cold-start cases)
        d = days_back if days_back is not None else 400
        start_ms = int(datetime.now(timezone.utc).timestamp() * 1000) - d * 86400_000

    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    cursor = start_ms
    rows = []
    while cursor < end_ms:
        params = dict(symbol="BTCUSDT", interval="1h",
                      startTime=cursor, limit=1000)
        try:
            r = requests.get("https://api.binance.com/api/v3/klines",
                             params=params, timeout=30)
            batch = r.json()
        except Exception:
            break
        # Guard: Binance returns an error dict (e.g. rate-limit) instead of a
        # list of klines — a non-empty dict would pass `if not batch` but then
        # fail on `batch[-1][0]`.  Also coerce the open-time to int to handle
        # API versions that return timestamps as strings.
        if not batch or not isinstance(batch, list) or not isinstance(batch[-1], (list, tuple)):
            break
        rows.extend(batch)
        cursor = int(batch[-1][0]) + 3600_000
        time.sleep(0.1)
    if rows:
        cols = ["open_time","open","high","low","close","volume",
                "close_time","qv","n","tb","tq","ig"]
        new_df = pd.DataFrame(rows, columns=cols)
        new_df["ts"] = pd.to_datetime(new_df["open_time"], unit="ms",
                                      utc=True).dt.tz_convert(None)
        for c in ["open","high","low","close","volume"]:
            new_df[c] = new_df[c].astype(float)
        new_df = new_df.set_index("ts")[["open","high","low","close","volume"]]
        parts.append(new_df)

    if not parts:
        # Binance unavailable (blocked, rate-limited, or no CSV on this host).
        # Fall back to Yahoo Finance — already a project dependency and works
        # reliably on Streamlit Community Cloud.
        return _fetch_yfinance_hourly_fallback()
    df = pd.concat(parts)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if days_back is not None and len(df):
        cutoff = df.index.max() - pd.Timedelta(days=days_back)
        df = df.loc[df.index >= cutoff]
    return df


@st.cache_data(ttl=3600*6, show_spinner="Fetching daily macro + on-chain …")
def _fetch_daily_raw():
    """Build the daily-bar DataFrame anchored at 12:00 UTC.

      - BTC OHLCV: rebucketed from Binance hourly into 12:00→12:00 UTC bars.
      - Macro (Yahoo daily, indexed by calendar date): SPX/NDX/VIX/Gold/DXY/TNX/ETH.
        Joined to bar D using calendar date D — macro closes for D are
        published by ~21:00 UTC on day D, well before bar D ends (D+1 12:00).
      - On-chain (blockchain.info, daily UTC): same calendar-date join.
    Cached 6 h."""
    # 1. BTC 12:00-UTC daily bars (full history → any picked date is fully featured)
    btc_hourly = _fetch_binance_hourly()
    if btc_hourly.empty:
        st.error(
            "⚠️ Could not fetch BTC hourly data from Binance **or** Yahoo Finance. "
            "Both data sources appear to be unavailable right now. "
            "Please wait a minute and click **Refresh now** in the sidebar."
        )
        st.stop()
    btc_daily = _rebucket_12utc(btc_hourly).add_prefix("btc_")

    # 2. Macro daily from Yahoo (calendar-date indexed)
    START = (btc_daily.index.min() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    END = (datetime.now(timezone.utc).date()
           + pd.Timedelta(days=1).to_pytimedelta()).strftime("%Y-%m-%d")
    # Extended ticker list — covers both 7d and 14d GBM feature sets:
    #   Base:    ETH, SPX, NDX, VIX, GOLD, DXY, TNX
    #   +7d/14d: IRX (yield spread), HG (copper), XLI, XLF (econ activity),
    #            CL, NG, XLE (energy)
    SYMS = {"eth":"ETH-USD","spx":"^GSPC","ndx":"^IXIC",
            "vix":"^VIX","gold":"GC=F","dxy":"DX-Y.NYB","tnx":"^TNX",
            "irx":"^IRX","hg":"HG=F","xli":"XLI","xlf":"XLF",
            "cl":"CL=F","ng":"NG=F","xle":"XLE"}
    parts = []
    for name, sym in SYMS.items():
        d = yf.download(sym, start=START, end=END, progress=False,
                        auto_adjust=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = [c[0] for c in d.columns]
        d = d[["Open","High","Low","Close","Volume"]].copy()
        d.columns = [f"{name}_{c.lower()}" for c in d.columns]
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
        parts.append(d)
    mkt = parts[0]
    for p in parts[1:]: mkt = mkt.join(p, how="outer")

    # 3. On-chain (blockchain.info)
    ONCHAIN = ["hash-rate","difficulty","n-transactions","miners-revenue",
               "n-unique-addresses","transaction-fees-usd","mempool-size",
               "estimated-transaction-volume-usd","market-cap","avg-block-size",
               "cost-per-transaction"]
    oc_parts = []
    for s in ONCHAIN:
        try:
            j = requests.get(f"https://api.blockchain.info/charts/{s}?timespan=all&format=json&sampled=false",
                             timeout=30).json()
            idx = pd.to_datetime([x["x"] for x in j["values"]], unit="s").normalize()
            ser = pd.Series([x["y"] for x in j["values"]], index=idx,
                            name=f"oc_{s.replace('-','_')}")
            ser = ser[~ser.index.duplicated(keep="last")]
            oc_parts.append(ser)
        except Exception:
            pass
    oc = pd.concat(oc_parts, axis=1) if oc_parts else pd.DataFrame()

    # 4. Join — bar D's aux data comes from calendar date D
    df = btc_daily.join(mkt, how="left").join(oc, how="left").sort_index()
    df = df.loc[df["btc_close"].notna()].ffill(limit=5)
    return df


@st.cache_data(ttl=86400, show_spinner="Computing daily H/L forecast …")
def compute_daily_forecast(target_date_iso):
    """Apply the 12:00-UTC (7am-CT) daily model as it was trained.

    `target_date_iso` = ISO date of the bar BEING PREDICTED. That bar covers
    [target 12:00 UTC, target+1 12:00 UTC) = [target 7am CT, target+1 7am CT).
    Only data completed before the target bar starts is used — i.e. bars with
    start date ≤ target − 1 day (the latest such bar closes at target 7am CT,
    which is the target bar's open). The cache key is the target date, so the
    prediction recomputes once per 12:00-UTC rollover automatically."""
    path = str(DAILY_MODEL_CT)
    if not os.path.exists(path):
        return None
    AD = joblib.load(path)
    mh, ml = AD["hi_model"], AD["lo_model"]
    sh, sl = AD["sigma_hi"], AD["sigma_lo"]
    fc = AD["feat_cols"]

    df = _fetch_daily_raw().copy()

    # Truncate to bars that close at or before the target bar's open
    # (target 12:00 UTC = target 7am CT). Those are bars with start ≤ target−1.
    target_date = pd.Timestamp(target_date_iso)
    asof_cutoff = target_date - pd.Timedelta(days=1)
    df = df.loc[df.index <= asof_cutoff]
    if df.empty:
        return None
    df = df.sort_index().ffill(limit=5)

    # 3. Daily features (same as the daily training notebook)
    f = pd.DataFrame(index=df.index)
    c = df["btc_close"]; h = df["btc_high"]; l_ = df["btc_low"]; v = df["btc_volume"]
    ret = np.log(c).diff()
    for k in [1,3,5,7,14,30]: f[f"ret_{k}"] = ret.rolling(k).sum()
    for k in [5,10,20,30]:    f[f"vol_{k}"] = ret.rolling(k).std()
    prev_c = c.shift(1)
    tr = pd.concat([(h-l_),(h-prev_c).abs(),(l_-prev_c).abs()],axis=1).max(axis=1)
    for k in [7,14,30]: f[f"atr_{k}"] = tr.rolling(k).mean()/c
    f["range_today"] = (h-l_)/c
    f["range_ma7"]   = ((h-l_)/c).rolling(7).mean()
    f["range_ma30"]  = ((h-l_)/c).rolling(30).mean()
    f["range_std30"] = ((h-l_)/c).rolling(30).std()
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain/loss.replace(0,np.nan)
    f["rsi_14"] = 100 - 100/(1+rs)
    e12 = c.ewm(span=12,adjust=False).mean(); e26 = c.ewm(span=26,adjust=False).mean()
    macd = e12 - e26
    f["macd"]      = macd/c
    f["macd_sig"]  = macd.ewm(span=9,adjust=False).mean()/c
    f["macd_hist"] = (macd-macd.ewm(span=9,adjust=False).mean())/c
    ma20=c.rolling(20).mean(); sd20=c.rolling(20).std()
    f["bb_width"]   = (4*sd20)/ma20
    f["dist_hi_30"] = c/c.rolling(30).max()-1
    f["dist_lo_30"] = c/c.rolling(30).min()-1
    f["dist_hi_90"] = c/c.rolling(90).max()-1
    f["vol_chg_1"]    = np.log(v).diff()
    f["vol_z_20"]     = (np.log(v)-np.log(v).rolling(20).mean())/np.log(v).rolling(20).std()
    f["vol_ma_ratio"] = v/v.rolling(20).mean()
    dow = df.index.dayofweek
    for i in range(6): f[f"dow_{i}"] = (dow==i).astype(float)
    def mret(name, ks=(1,5,20)):
        s = df[f"{name}_close"]
        for k in ks: f[f"{name}_ret_{k}"] = np.log(s).diff(k)
        f[f"{name}_vol_20"] = np.log(s).diff().rolling(20).std()
    for nm in ["spx","ndx","vix","gold","dxy","tnx","eth"]: mret(nm)
    f["btc_spx_corr_30"]  = ret.rolling(30).corr(np.log(df["spx_close"]).diff())
    f["btc_ndx_corr_30"]  = ret.rolling(30).corr(np.log(df["ndx_close"]).diff())
    f["btc_gold_corr_30"] = ret.rolling(30).corr(np.log(df["gold_close"]).diff())
    f["btc_dxy_corr_30"]  = ret.rolling(30).corr(np.log(df["dxy_close"]).diff())
    for col in [x for x in df.columns if x.startswith("oc_")]:
        # Local `s_log` instead of `sl` — `sl` is bound earlier to AD["sigma_lo"]
        # and needed downstream for the LOW CI bands.
        s = df[col].astype(float); s_log = np.log(s.replace(0,np.nan))
        f[f"{col}_d1"]  = s_log.diff(1)
        f[f"{col}_d7"]  = s_log.diff(7)
        f[f"{col}_z30"] = (s_log - s_log.rolling(30).mean())/s_log.rolling(30).std()
    nh, nl = h.shift(-1), l_.shift(-1)
    y_hi = (nh-c)/c; y_lo = (c-nl)/c
    # Smoothed lag features (must match `src/pipeline_ct.py` exactly).
    f["y_hi_ema3"] = y_hi.shift(1).ewm(span=3, adjust=False).mean()
    f["y_lo_ema3"] = y_lo.shift(1).ewm(span=3, adjust=False).mean()
    f["y_hi_ema7"] = y_hi.shift(1).ewm(span=7, adjust=False).mean()
    f["y_lo_ema7"] = y_lo.shift(1).ewm(span=7, adjust=False).mean()
    # Anti-mean-reversion features
    prev_3_hi = h.shift(1).rolling(3).max()
    prev_3_lo = l_.shift(1).rolling(3).min()
    f["above_3d_high"] = (c > prev_3_hi).astype(float)
    f["below_3d_low"]  = (c < prev_3_lo).astype(float)
    f["bo_strength_up"] = (c / prev_3_hi - 1).clip(lower=0)
    f["bo_strength_dn"] = (1 - c / prev_3_lo).clip(lower=0)
    _y_hi_lag = y_hi.shift(1)
    _y_lo_lag = y_lo.shift(1)
    f["y_hi_surprise"] = _y_hi_lag - _y_hi_lag.ewm(span=7, adjust=False).mean()
    f["y_lo_surprise"] = _y_lo_lag - _y_lo_lag.ewm(span=7, adjust=False).mean()
    # LOW-specific / downside regime features
    neg_ret = ret.clip(upper=0)
    f["dn_vol_5"]  = neg_ret.rolling(5).std()
    f["dn_vol_20"] = neg_ret.rolling(20).std()
    sma50 = c.rolling(50).mean()
    f["below_sma50"] = (c < sma50).astype(float)
    f["below_sma50_5d"] = f["below_sma50"].rolling(5).min().fillna(0)

    f = f.replace([np.inf,-np.inf], np.nan)
    F = f[fc].dropna()
    if F.empty:
        return None
    # df was already truncated to bars with start ≤ target − 1 day, so the
    # last row is the as-of bar — the one whose close coincides with the
    # target bar's open (target 12:00 UTC = target 7am CT).
    asof = F.index[-1]
    close_asof = float(c.loc[asof])
    # The target bar covers [target 12:00 UTC, target+1 12:00 UTC).
    target_window_start = target_date + pd.Timedelta(hours=ANCHOR_HOUR_UTC)
    target_window_end   = target_date + pd.Timedelta(days=1, hours=ANCHOR_HOUR_UTC)

    # Robust scalar coercion (assets can come back as numpy 0-d arrays,
    # Series, or pure floats depending on how joblib reloaded them).
    def _scalar(x):
        if hasattr(x, "item"):
            try: return float(x.item())
            except Exception: pass
        if hasattr(x, "iloc"):
            return float(x.iloc[0])
        return float(np.asarray(x).ravel()[0])

    # Predict — either single model or ensemble (mean of constituents),
    # optionally blended with climatological mean offset.
    row = F.loc[[asof]]
    if AD.get("ensemble") and AD.get("constituents"):
        pred_hi_list = [_scalar(c["m_hi"].predict(row)[0]) for c in AD["constituents"]]
        pred_lo_list = [_scalar(c["m_lo"].predict(row)[0]) for c in AD["constituents"]]
        yhi = float(np.mean(pred_hi_list))
        ylo = float(np.mean(pred_lo_list))
        if AD.get("blended") and float(AD.get("alpha", 1.0)) < 1.0:
            a = float(AD["alpha"])
            yhi = a * yhi + (1 - a) * float(AD.get("mu_hi", yhi))
            ylo = a * ylo + (1 - a) * float(AD.get("mu_lo", ylo))
    else:
        yhi = _scalar(mh.predict(row)[0])
        ylo = _scalar(ml.predict(row)[0])

    # Direction head (optional, present in newer artefacts). Reparameterise
    # into (half-range m, asymmetry d), replace d with a blend of ensemble's
    # d and a classifier-driven d, then reconstruct yhi/ylo.
    # β is adaptive: trend_str = min(|ret_5|/trend_sat, 1); β_eff =
    # β_base × (1 − reduction × trend_str). On trending days β shrinks
    # (favour direction head); in chop β stays near β_base.
    dh = AD.get("direction_head")
    p_bull = None
    beta_eff = None
    if dh is not None and dh.get("classifier") is not None:
        try:
            clf = dh["classifier"]
            beta_base    = float(dh.get("beta", 1.0))
            reduction    = float(dh.get("beta_trend_reduction", 0.0))
            trend_sat    = float(dh.get("trend_saturation", 0.05))
            trend_feat   = dh.get("trend_feature", "ret_5")
            d_bull_mean  = float(dh.get("d_bull_mean", 0.0))
            d_bear_mean  = float(dh.get("d_bear_mean", 0.0))
            p_bull = float(clf.predict_proba(row)[0, 1])
            try:
                ret_val = float(row.iloc[0][trend_feat])
                trend_str = min(abs(ret_val) / trend_sat, 1.0)
            except Exception:
                trend_str = 0.0
            beta_eff = max(0.0, min(1.0, beta_base * (1.0 - reduction * trend_str)))
            m_pred = (yhi + ylo) / 2.0
            d_pred = (yhi - ylo) / 2.0
            d_dir  = p_bull * d_bull_mean + (1.0 - p_bull) * d_bear_mean
            d_blend = beta_eff * d_pred + (1.0 - beta_eff) * d_dir
            yhi = m_pred + d_blend
            ylo = m_pred - d_blend
        except Exception:
            # If the classifier fails, fall back to ensemble-only prediction
            p_bull = None
            beta_eff = None

    sh  = _scalar(sh); sl = _scalar(sl)
    clip0 = lambda x: float(max(float(x), 0.0))
    pred_high = close_asof * (1 + clip0(yhi))
    pred_low  = close_asof * (1 - clip0(ylo))
    band_hi_up = close_asof * (1 + yhi + 1.96*sh)
    band_hi_dn = close_asof * (1 + clip0(yhi - 1.96*sh))
    band_lo_up = close_asof * (1 - clip0(ylo - 1.96*sl))
    band_lo_dn = close_asof * (1 - clip0(ylo + 1.96*sl))

    return dict(
        as_of_date=asof, close_asof=close_asof,
        target_date=target_date,
        target_window_start=target_window_start,
        target_window_end=target_window_end,
        pred_high=pred_high, high_ci_lo=band_hi_dn, high_ci_hi=band_hi_up,
        pred_low =pred_low,  low_ci_lo =band_lo_dn, low_ci_hi =band_lo_up,
        p_bull=p_bull, beta_eff=beta_eff,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def compute_daily_series(end_target_date_iso, days_back=7):
    """Build a series of (pred_high, pred_low, actual_high, actual_low) for the
    last `days_back`+1 target days ending at `end_target_date`. Each prediction
    is generated by `compute_daily_forecast(target_date)`, which internally
    uses data through target − 1 day (i.e. through target 7am CT).

    The cache TTL is 1 hour so the series refreshes when underlying data does."""
    end_target = pd.Timestamp(end_target_date_iso)
    daily_df = _fetch_daily_raw()
    rows = []
    for i in range(days_back, -1, -1):
        target_date = end_target - pd.Timedelta(days=i)
        pred = compute_daily_forecast(target_date.strftime("%Y-%m-%d"))
        if pred is None:
            continue
        ts = pd.Timestamp(target_date)
        actual_h = (float(daily_df.loc[ts, "btc_high"])
                    if ts in daily_df.index and pd.notna(daily_df.loc[ts, "btc_high"])
                    else np.nan)
        actual_l = (float(daily_df.loc[ts, "btc_low"])
                    if ts in daily_df.index and pd.notna(daily_df.loc[ts, "btc_low"])
                    else np.nan)
        rows.append(dict(
            target_date=ts,
            as_of_date=pred["as_of_date"],
            close_asof=float(pred["close_asof"]),
            pred_high=float(pred["pred_high"]),
            pred_low =float(pred["pred_low"]),
            actual_high=actual_h,
            actual_low =actual_l,
        ))
    return pd.DataFrame(rows)


@st.cache_resource
def _load_cone_7d():
    """Load the 7-day close-price regime-cone artefact (or None if absent)."""
    p = str(CONE_7D_MODEL)
    if not os.path.exists(p):
        return None
    return joblib.load(p)


@st.cache_resource
def _load_cone_14d():
    """Load the 14-day close-price GBM+cone artefact (or None if absent)."""
    p = str(CONE_14D_MODEL)
    if not os.path.exists(p):
        return None
    return joblib.load(p)


@st.cache_data(ttl=3600 * 6, show_spinner=False)
def _build_cone_feature_matrix():
    """Build the shared daily feature matrix for the 7-day and 14-day GBM cone models.

    Mirrors the feature engineering in ``src/train_7d_close_cone.py`` and
    ``src/train_14d_close_cone.py`` exactly, using daily data from
    ``_fetch_daily_raw()`` (which now includes IRX, HG, XLI, XLF, CL, NG, XLE).

    Returns a DataFrame indexed by the same daily dates as ``_fetch_daily_raw()``.
    Rows with NaN are forward-filled (limit 5) to match training-time behaviour.
    Result is cached for 6 h — same TTL as ``_fetch_daily_raw()``.
    """
    df = _fetch_daily_raw()
    if df.empty:
        return pd.DataFrame()

    c   = df["btc_close"]
    h   = df.get("btc_high",   pd.Series(c, name="btc_high"))
    l_  = df.get("btc_low",    pd.Series(c, name="btc_low"))
    vol = df.get("btc_volume", pd.Series(1e6, index=c.index, name="btc_volume"))

    feats = {}

    # ── BTC own ──────────────────────────────────────────────────────────────
    ret = np.log(c).diff()
    for k in [1, 3, 5, 7, 10, 14, 21]:
        feats[f"btc_ret_{k}"] = ret.rolling(k).sum()
    for k in [7, 14, 20, 30]:
        feats[f"btc_vol_{k}"] = ret.rolling(k).std()
    _delta = c.diff()
    _gain  = _delta.clip(lower=0).rolling(14).mean()
    _loss  = (-_delta.clip(upper=0)).rolling(14).mean()
    feats["btc_rsi_14"] = 100 - 100 / (1 + _gain / _loss.replace(0, np.nan))
    feats["range_ma30"] = ((h - l_) / c).rolling(30).mean()

    # ── Macro helper ─────────────────────────────────────────────────────────
    def _add_macro(src_col, prefix, lookbacks=(1, 5, 10, 14, 20)):
        if src_col not in df.columns:
            return
        s = df[src_col]
        r = np.log(s.clip(lower=1e-9)).diff()
        for k in lookbacks:
            feats[f"{prefix}_ret_{k}"] = r.rolling(k).sum()
        feats[f"{prefix}_vol_20"] = r.rolling(20).std()

    # Baseline macro (both models)
    for _src, _pfx in [("spx_close","spx"),("ndx_close","ndx"),("vix_close","vix"),
                       ("gold_close","gold"),("dxy_close","dxy"),("eth_close","eth")]:
        _add_macro(_src, _pfx)

    # TNX (10-year yield) — returns + vol
    if "tnx_close" in df.columns:
        _tnx  = df["tnx_close"]
        _rtnx = np.log(_tnx.clip(lower=1e-9)).diff()
        for k in [1, 5, 10, 14, 20]:
            feats[f"tnx_ret_{k}"] = _rtnx.rolling(k).sum()
        feats["tnx_vol_20"] = _rtnx.rolling(20).std()

    # ── Yield curve spread (10Y − 3M) ────────────────────────────────────────
    if "tnx_close" in df.columns and "irx_close" in df.columns:
        _spread = df["tnx_close"] - df["irx_close"]
        feats["spread_10y_3m"]        = _spread
        feats["spread_10y_3m_chg_5"]  = _spread.diff(5)
        feats["spread_10y_3m_chg_20"] = _spread.diff(20)
        feats["curve_inverted"]       = (_spread < 0).astype(float)
        feats["curve_inverted_20d"]   = feats["curve_inverted"].rolling(20).mean()

    # ── Economic activity ─────────────────────────────────────────────────────
    _add_macro("xli_close", "xli")
    _add_macro("xlf_close", "xlf")

    if "hg_close" in df.columns:
        _hg   = df["hg_close"]
        _rhg  = np.log(_hg.clip(lower=1e-9)).diff()
        for k in [1, 5, 10, 14, 20]:
            feats[f"hg_ret_{k}"] = _rhg.rolling(k).sum()
        feats["hg_vol_20"] = _rhg.rolling(20).std()
        if "gold_close" in df.columns:
            _cg = np.log(_hg / df["gold_close"].clip(lower=1e-9))
            feats["copper_gold_ratio"]    = _cg
            feats["copper_gold_chg_14"]   = _cg.diff(14)
            feats["copper_gold_chg_20"]   = _cg.diff(20)

    # ── Energy ───────────────────────────────────────────────────────────────
    _add_macro("cl_close",  "cl")
    _add_macro("xle_close", "xle")

    if "ng_close" in df.columns:
        _ng  = df["ng_close"]
        _rng = np.log(_ng.clip(lower=1e-9)).diff()
        for k in [1, 5, 10, 14, 20]:
            feats[f"ng_ret_{k}"] = _rng.rolling(k).sum()
        feats["ng_vol_20"] = _rng.rolling(20).std()

    # ── ETH/BTC ratio ─────────────────────────────────────────────────────────
    if "eth_close" in df.columns:
        feats["eth_btc_ratio"]        = np.log(df["eth_close"] / c.clip(lower=1e-9))
        feats["eth_btc_ratio_chg_14"] = feats["eth_btc_ratio"].diff(14)

    # ── TI-B Momentum (7d model: RSI, Stochastic, Williams-R, CCI, ROC) ──────
    def _rsi(close, p):
        _d  = close.diff()
        _g  = _d.clip(lower=0).rolling(p).mean()
        _lo = (-_d.clip(upper=0)).rolling(p).mean()
        return 100 - 100 / (1 + _g / _lo.replace(0, np.nan))

    feats["rsi_7"]  = _rsi(c, 7)
    feats["rsi_21"] = _rsi(c, 21)
    feats["rsi_30"] = _rsi(c, 30)

    _lo14 = l_.rolling(14).min()
    _hi14 = h.rolling(14).max()
    _sk   = 100 * (c - _lo14) / (_hi14 - _lo14 + 1e-9)
    _sd   = _sk.rolling(3).mean()
    feats["stoch_k"]       = _sk
    feats["stoch_d"]       = _sd
    feats["stoch_kd_diff"] = _sk - _sd
    feats["williams_r"]    = -100 * (_hi14 - c) / (_hi14 - _lo14 + 1e-9)

    _tp       = (h + l_ + c) / 3
    _sma_tp20 = _tp.rolling(20).mean()
    _mad_tp20 = _tp.rolling(20).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    feats["cci_20"] = (_tp - _sma_tp20) / (0.015 * _mad_tp20 + 1e-9)
    feats["roc_7"]  = c / c.shift(7)  - 1
    feats["roc_14"] = c / c.shift(14) - 1

    # ── TI-D Volume (14d model: CMF, OBV, vol ratios, PV momentum) ───────────
    _mfm = ((c - l_) - (h - c)) / (h - l_ + 1e-9)
    _mfv = _mfm * vol
    feats["cmf_20"] = _mfv.rolling(20).sum() / (vol.rolling(20).sum() + 1e-9)

    _direction       = np.sign(c.diff()).fillna(0)
    _obv             = (_direction * vol).cumsum()
    _obv_ma20        = _obv.rolling(20).mean()
    feats["obv_vs_ma20"]  = _obv / (_obv_ma20.abs() + 1e-9) - 1
    feats["vol_ratio_20"] = vol / (vol.rolling(20).mean() + 1e-9) - 1
    feats["vol_ratio_5"]  = vol / (vol.rolling(5).mean()  + 1e-9) - 1

    _pv = ret * vol
    feats["pv_mom_5"]     = _pv.rolling(5).sum()  / (vol.rolling(5).sum()  + 1e-9)
    feats["pv_mom_20"]    = _pv.rolling(20).sum() / (vol.rolling(20).sum() + 1e-9)
    feats["vol_slope_20"] = np.log(vol.rolling(20).mean() + 1e-9).diff(20)

    # ── OC-B Miner/flow (7d model: tx fees, miners rev, mempool, tx vol) ─────
    _OC_MAP = {
        "oc_transaction_fees_usd":            "oc_tx_fees_usd",
        "oc_miners_revenue":                  "oc_miners_rev",
        "oc_mempool_size":                    "oc_mempool_sz",
        "oc_estimated_transaction_volume_usd":"oc_tx_vol_usd",
    }
    for _app_col, _feat_pfx in _OC_MAP.items():
        if _app_col not in df.columns:
            continue
        _s  = df[_app_col].astype(float)
        _ld = np.log(_s.clip(lower=1e-9)).diff()
        feats[f"{_feat_pfx}_ret_7"]  = _ld.rolling(7).sum()
        feats[f"{_feat_pfx}_ret_30"] = _ld.rolling(30).sum()
        feats[f"{_feat_pfx}_vol_20"] = _ld.rolling(20).std()

    if "oc_estimated_transaction_volume_usd" in df.columns:
        _tv  = df["oc_estimated_transaction_volume_usd"].astype(float)
        _nvt = np.log(c * 19_500_000 / _tv.clip(lower=1e-9))
        feats["oc_nvt_proxy_chg_30"] = _nvt.diff(30)

    if ("oc_miners_revenue" in df.columns and
            "oc_transaction_fees_usd" in df.columns):
        _mr = df["oc_miners_revenue"].astype(float)
        _tf = df["oc_transaction_fees_usd"].astype(float)
        _ms = np.log(_mr / _tf.clip(lower=1e-9))
        feats["oc_miner_stress_chg_30"] = _ms.diff(30)

    fdf = pd.DataFrame(feats, index=df.index)
    return fdf.replace([np.inf, -np.inf], np.nan).ffill(limit=5)


def _cone_predict_batch(art, feat_df, anchor_dates):
    """Apply the cone artefact's GBM (if use_ml=True) to a batch of anchor dates.

    Returns a dict {anchor_date: predicted_log_return}.  Falls back to
    an empty dict when the model is unavailable or feature columns are missing,
    allowing callers to use the regime-median fallback.
    """
    if not (art.get("use_ml") and art.get("ml_point_model") is not None):
        return {}
    feat_cols_model = art.get("ml_feature_cols", [])
    if not feat_cols_model or feat_df.empty:
        return {}
    avail = [a for a in anchor_dates if a in feat_df.index]
    if not avail:
        return {}
    try:
        X = pd.DataFrame(0.0, index=avail, columns=feat_cols_model)
        common = [c for c in feat_cols_model if c in feat_df.columns]
        X[common] = feat_df.loc[avail, common].values
        X = X.fillna(0)
        preds = art["ml_point_model"].predict(X)
        return dict(zip(avail, preds.tolist()))
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def compute_7d_close_cone_forecast(asof_date_iso):
    """Forecast BTC close 7 days after `asof_date_iso` using the regime cone.

    The cone is parameter-free at inference: classify the as-of bar's
    ``range_ma30`` into one of three training-set terciles, take the
    regime's median forward 7-day log-return, multiply against the
    as-of close, and apply a fixed ±9.7 % band (the headline empirical
    band-width from notebooks/btc_7d_close_research.ipynb).

    Returns ``None`` if the cone artefact is missing or the daily data
    can't reach back 49 days. Otherwise returns a dict with:

      history     – DataFrame of 7 weekly close observations spaced 7
                    days apart ending at the as-of bar
      pred_date   – target Timestamp = as_of + 7 days
      pred_close  – predicted USD close at pred_date
      lower / upper – ±9.7 % band on pred_close
      regime, regime_label – tercile index and short label
      band_pct    – the fixed band width (0.097)
      asof_close  – the as-of-bar close used as the anchor
    """
    cone = _load_cone_7d()
    if cone is None:
        return None
    daily = _fetch_daily_raw().copy()
    if daily.empty:
        return None

    asof = pd.Timestamp(asof_date_iso)
    # Snap to the latest available bar at or before asof
    bars_avail = daily.loc[daily.index <= asof]
    if bars_avail.empty:
        return None
    asof_t = bars_avail.index[-1]

    # range_ma30 needs 30 prior daily bars; weekly history needs 49 days back
    c = bars_avail["btc_close"]; h = bars_avail["btc_high"]; l_ = bars_avail["btc_low"]
    range_today = (h - l_) / c
    range_ma30  = range_today.rolling(30).mean()
    rm30 = range_ma30.loc[asof_t]
    if not np.isfinite(rm30):
        return None

    edges = np.asarray(cone["regime_edges"], dtype=float)
    regime = int(np.searchsorted(edges, float(rm30), side="right"))
    regime = max(0, min(len(edges), regime))
    regime_label = ["low vol","mid vol","high vol"][regime] if regime < 3 else f"r{regime}"

    stats = cone["regime_stats"][regime]
    # Median forward 7-day log-return for this regime (cone fallback)
    med_logret = float(stats[0.50] if 0.50 in stats else stats["0.5"])
    asof_close = float(c.loc[asof_t])
    band_pct = float(cone.get("band_pct", 0.097))
    # GBM point prediction (v2 artefact) when available
    gbm_preds = _cone_predict_batch(cone, _build_cone_feature_matrix(), [asof_t])
    if asof_t in gbm_preds:
        pred_logret = gbm_preds[asof_t]
    else:
        pred_logret = med_logret
    pred_close = asof_close * float(np.exp(pred_logret))
    lower = pred_close * (1 - band_pct)
    upper = pred_close * (1 + band_pct)
    pred_date = asof_t + pd.Timedelta(days=7)

    # Helper: snap a target date to the latest bar at or before it.
    def _snap(d):
        if d in c.index and pd.notna(c.loc[d]):
            return d
        prior = c.loc[c.index <= d]
        return prior.index[-1] if not prior.empty else None

    # 7 weekly target dates ending at asof_t (oldest → newest).
    hist_dates, hist_closes = [], []
    for k in range(6, -1, -1):
        snapped = _snap(asof_t - pd.Timedelta(days=7 * k))
        if snapped is not None:
            hist_dates.append(snapped); hist_closes.append(float(c.loc[snapped]))
    history = pd.DataFrame({"close": hist_closes}, index=pd.DatetimeIndex(hist_dates))

    # Historical predictions: for each target date d in `history.index`,
    # find the prediction-anchor date 7 days before, classify its regime
    # via range_ma30 at that anchor, and apply the regime median return.
    # The "actual" at d is just history.close[d].
    rows = []
    for d in history.index:
        a = _snap(d - pd.Timedelta(days=7))
        if a is None or not np.isfinite(range_ma30.loc[a]):
            continue
        a_close = float(c.loc[a])
        rm30_a  = float(range_ma30.loc[a])
        r       = int(np.searchsorted(edges, rm30_a, side="right"))
        r       = max(0, min(len(edges), r))
        m       = float(cone["regime_stats"][r][0.50])
        p_close = a_close * float(np.exp(m))
        rows.append(dict(
            anchor_date   = a,
            target_date   = d,
            anchor_close  = a_close,
            pred_close    = p_close,
            lower         = p_close * (1 - band_pct),
            upper         = p_close * (1 + band_pct),
            actual_close  = float(c.loc[d]),
            regime        = r,
        ))
    hist_preds = pd.DataFrame(rows)

    # In historical-replay mode the +7d target may already be in the past;
    # surface the realized close if data is available for that bar so the
    # caller can plot the actual alongside the forecast star.
    actual_pred_close, actual_pred_date = None, None
    last_avail = c.index[-1]
    if pred_date <= last_avail:
        snapped_pred = _snap(pred_date)
        if snapped_pred is not None and pd.notna(c.loc[snapped_pred]):
            actual_pred_close = float(c.loc[snapped_pred])
            actual_pred_date  = snapped_pred

    return dict(
        history          = history,
        hist_preds       = hist_preds,
        pred_date        = pred_date,
        pred_close       = pred_close,
        lower            = lower,
        upper            = upper,
        regime           = regime,
        regime_label     = regime_label,
        band_pct         = band_pct,
        asof_close       = asof_close,
        asof_date        = asof_t,
        regime_median_logret = med_logret,
        actual_pred_close = actual_pred_close,
        actual_pred_date  = actual_pred_date,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def compute_rolling_7d_series(end_date_iso, days_back=21):
    """Generate rolling daily 7-day forward close predictions.

    For each anchor day D from (end_date - days_back) to end_date,
    compute the 7-day forward cone prediction and check whether the
    target date (anchor + 7 calendar days) has a realized close.

    Returns a DataFrame with columns:
      anchor_date    – the anchor day (prediction anchored here)
      target_date    – anchor + 7 calendar days (prediction target)
      anchor_close   – realized BTC close at anchor date
      pred_close     – predicted BTC close at target_date
      lower / upper  – ±band_pct % bounds around pred_close
      actual_close   – realized close at target_date (NaN = still future)
      regime         – tercile index (0 = low vol, 1 = mid, 2 = high vol)
      regime_label   – human-readable regime label
    """
    cone = _load_cone_7d()
    if cone is None:
        return pd.DataFrame()
    daily = _fetch_daily_raw().copy()
    if daily.empty:
        return pd.DataFrame()

    end      = pd.Timestamp(end_date_iso)
    c        = daily["btc_close"]
    h        = daily["btc_high"]
    l_       = daily["btc_low"]
    range_today = (h - l_) / c
    range_ma30  = range_today.rolling(30).mean()
    edges    = np.asarray(cone["regime_edges"], dtype=float)
    band_pct = float(cone.get("band_pct", 0.097))

    def _snap(d):
        if d in c.index and pd.notna(c.loc[d]):
            return d
        prior = c.loc[c.index <= d]
        return prior.index[-1] if not prior.empty else None

    # Collect all valid anchor dates first so we can batch-predict with GBM
    candidate_anchors = []
    for i in range(days_back, -1, -1):
        a = _snap(end - pd.Timedelta(days=i))
        if a is not None and a in range_ma30.index and np.isfinite(range_ma30.loc[a]):
            candidate_anchors.append(a)

    # Batch GBM predictions (empty dict → fall back to regime median per anchor)
    gbm_preds_7d = _cone_predict_batch(cone, _build_cone_feature_matrix(),
                                       candidate_anchors)

    rows = []
    for anchor in candidate_anchors:
        rm30 = range_ma30.loc[anchor]
        a_close = float(c.loc[anchor])
        regime  = int(np.searchsorted(edges, float(rm30), side="right"))
        regime  = max(0, min(len(edges), regime))
        regime_label = (["low vol", "mid vol", "high vol"][regime]
                        if regime < 3 else f"r{regime}")
        stats = cone["regime_stats"][regime]
        med_logret = float(stats[0.50] if 0.50 in stats else stats["0.5"])
        pred_logret = gbm_preds_7d.get(anchor, med_logret)
        p_close = a_close * float(np.exp(pred_logret))
        lower   = p_close * (1 - band_pct)
        upper   = p_close * (1 + band_pct)
        target  = anchor + pd.Timedelta(days=7)
        # Only mark as realized if the target date itself is within
        # available data. _snap() can silently return the most-recent
        # past bar for any future date, so we must guard on `target`
        # (not on the snapped result) to avoid showing future "actuals".
        actual_close = np.nan
        if target <= c.index[-1]:
            target_snapped = _snap(target)
            if target_snapped is not None and pd.notna(c.loc[target_snapped]):
                actual_close = float(c.loc[target_snapped])
        rows.append(dict(
            anchor_date  = anchor,
            target_date  = target,
            anchor_close = a_close,
            pred_close   = p_close,
            lower        = lower,
            upper        = upper,
            actual_close = actual_close,
            regime       = regime,
            regime_label = regime_label,
        ))
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 14-day close-price cone (GBM point prediction + empirical ±band)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def compute_14d_close_cone_forecast(asof_date_iso):
    """Forecast BTC close 14 days after ``asof_date_iso`` using the GBM+cone model.

    Mirrors ``compute_7d_close_cone_forecast`` but with a 14-day horizon.
    Uses the artefact's GBM (``ml_point_model``) when ``use_ml=True``, falling
    back to the regime-median log-return when the model or features are absent.

    Returns a dict with the same schema as the 7-day equivalent (horizon_days=14),
    or ``None`` if the artefact or data is unavailable.
    """
    cone = _load_cone_14d()
    if cone is None:
        return None
    daily = _fetch_daily_raw().copy()
    if daily.empty:
        return None

    asof = pd.Timestamp(asof_date_iso)
    bars_avail = daily.loc[daily.index <= asof]
    if bars_avail.empty:
        return None
    asof_t = bars_avail.index[-1]

    c = bars_avail["btc_close"]
    h = bars_avail.get("btc_high",  c.copy())
    l_= bars_avail.get("btc_low",   c.copy())
    range_today = (h - l_) / c
    range_ma30  = range_today.rolling(30).mean()
    rm30 = range_ma30.loc[asof_t]
    if not np.isfinite(rm30):
        return None

    edges = np.asarray(cone["regime_edges"], dtype=float)
    regime = int(np.searchsorted(edges, float(rm30), side="right"))
    regime = max(0, min(len(edges), regime))
    regime_label = (["low vol", "mid vol", "high vol"][regime]
                    if regime < 3 else f"r{regime}")

    stats = cone["regime_stats"][regime]
    med_logret = float(stats[0.50] if 0.50 in stats else stats["0.5"])
    asof_close = float(c.loc[asof_t])
    band_pct = float(cone.get("band_pct", 0.16))   # ≈±16 % for 14-day horizon

    # GBM point prediction when available
    gbm_preds = _cone_predict_batch(cone, _build_cone_feature_matrix(), [asof_t])
    pred_logret = gbm_preds.get(asof_t, med_logret)
    pred_close = asof_close * float(np.exp(pred_logret))
    lower = pred_close * (1 - band_pct)
    upper = pred_close * (1 + band_pct)
    pred_date = asof_t + pd.Timedelta(days=14)

    def _snap(d):
        if d in c.index and pd.notna(c.loc[d]):
            return d
        prior = c.loc[c.index <= d]
        return prior.index[-1] if not prior.empty else None

    # Check if predicted target has already passed (historical replay)
    actual_pred_close, actual_pred_date = None, None
    last_avail = c.index[-1]
    if pred_date <= last_avail:
        snapped_pred = _snap(pred_date)
        if snapped_pred is not None and pd.notna(c.loc[snapped_pred]):
            actual_pred_close = float(c.loc[snapped_pred])
            actual_pred_date  = snapped_pred

    return dict(
        pred_date        = pred_date,
        pred_close       = pred_close,
        lower            = lower,
        upper            = upper,
        regime           = regime,
        regime_label     = regime_label,
        band_pct         = band_pct,
        asof_close       = asof_close,
        asof_date        = asof_t,
        regime_median_logret = med_logret,
        use_ml           = bool(asof_t in gbm_preds),
        actual_pred_close = actual_pred_close,
        actual_pred_date  = actual_pred_date,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def compute_rolling_14d_series(end_date_iso, days_back=30):
    """Generate rolling daily 14-day forward close predictions.

    For each anchor day D from (end_date − days_back) to end_date, predicts
    the close 14 calendar days forward.  With days_back=30 the resolved subset
    covers ~16 observations and the future wing shows ~14 active forecasts.

    Returns a DataFrame with columns identical to ``compute_rolling_7d_series``.
    """
    cone = _load_cone_14d()
    if cone is None:
        return pd.DataFrame()
    daily = _fetch_daily_raw().copy()
    if daily.empty:
        return pd.DataFrame()

    end = pd.Timestamp(end_date_iso)
    c   = daily["btc_close"]
    h   = daily.get("btc_high",  c.copy())
    l_  = daily.get("btc_low",   c.copy())
    range_today = (h - l_) / c
    range_ma30  = range_today.rolling(30).mean()
    edges    = np.asarray(cone["regime_edges"], dtype=float)
    band_pct = float(cone.get("band_pct", 0.16))

    def _snap(d):
        if d in c.index and pd.notna(c.loc[d]):
            return d
        prior = c.loc[c.index <= d]
        return prior.index[-1] if not prior.empty else None

    # Collect valid anchors for batch GBM prediction
    candidate_anchors = []
    for i in range(days_back, -1, -1):
        a = _snap(end - pd.Timedelta(days=i))
        if a is not None and a in range_ma30.index and np.isfinite(range_ma30.loc[a]):
            candidate_anchors.append(a)

    gbm_preds_14d = _cone_predict_batch(cone, _build_cone_feature_matrix(),
                                        candidate_anchors)

    rows = []
    for anchor in candidate_anchors:
        rm30 = range_ma30.loc[anchor]
        a_close = float(c.loc[anchor])
        regime  = int(np.searchsorted(edges, float(rm30), side="right"))
        regime  = max(0, min(len(edges), regime))
        regime_label = (["low vol", "mid vol", "high vol"][regime]
                        if regime < 3 else f"r{regime}")
        stats = cone["regime_stats"][regime]
        med_logret  = float(stats[0.50] if 0.50 in stats else stats["0.5"])
        pred_logret = gbm_preds_14d.get(anchor, med_logret)
        p_close = a_close * float(np.exp(pred_logret))
        lower   = p_close * (1 - band_pct)
        upper   = p_close * (1 + band_pct)
        target  = anchor + pd.Timedelta(days=14)
        actual_close = np.nan
        if target <= c.index[-1]:
            ts = _snap(target)
            if ts is not None and pd.notna(c.loc[ts]):
                actual_close = float(c.loc[ts])
        rows.append(dict(
            anchor_date  = anchor,
            target_date  = target,
            anchor_close = a_close,
            pred_close   = p_close,
            lower        = lower,
            upper        = upper,
            actual_close = actual_close,
            regime       = regime,
            regime_label = regime_label,
        ))
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600, show_spinner=False)
def compute_30d_cone_14d_metrics(end_date_iso):
    """30-day look-back metrics for the 14-day close-cone model.

    Requests 44 days of rolling anchors so the resolved subset
    (target_date already in the past) covers roughly 30 observations.
    Returns None if fewer than 5 resolved predictions are available.
    """
    rolling = compute_rolling_14d_series(end_date_iso, days_back=44)
    if rolling is None or rolling.empty:
        return None
    resolved = rolling[rolling["actual_close"].notna()].copy()
    if len(resolved) < 5:
        return None
    mape = ((resolved["actual_close"] - resolved["pred_close"]).abs()
            / resolved["pred_close"]).mean() * 100
    band_pct = float(resolved.iloc[0]["upper"] / resolved.iloc[0]["pred_close"] - 1)
    within = (((resolved["actual_close"] >= resolved["lower"])
               & (resolved["actual_close"] <= resolved["upper"])).mean() * 100)
    return dict(n=len(resolved), mape=float(mape),
                within_pct=float(within), band_pct=band_pct)


@st.cache_data(ttl=3600, show_spinner=False)
def compute_alltime_cone_14d_metrics(end_date_iso):
    """MAPE + band coverage over the full held-out test period for the 14-day cone.

    Anchors ``compute_rolling_14d_series`` at ``test_start`` so only genuinely
    out-of-sample predictions are included.  Returns None when fewer than 5
    resolved predictions are available.
    """
    cone = _load_cone_14d()
    if cone is None:
        return None
    meta = cone.get("calibration_meta", {})
    test_start_str = meta.get("test_start", "2025-09-25")
    test_start_ts  = pd.Timestamp(test_start_str)
    end_ts = pd.Timestamp(end_date_iso)
    days_back = int((end_ts - test_start_ts).days) + 28  # +14d lag + buffer
    rolling = compute_rolling_14d_series(end_date_iso, days_back=days_back)
    if rolling is None or rolling.empty:
        return None
    rolling  = rolling[rolling["anchor_date"] >= test_start_ts].copy()
    resolved = rolling[rolling["actual_close"].notna()]
    if len(resolved) < 5:
        return None
    mape = ((resolved["actual_close"] - resolved["pred_close"]).abs()
            / resolved["pred_close"]).mean() * 100
    band_pct = float(cone.get("band_pct", 0.16))
    within = (((resolved["actual_close"] >= resolved["lower"])
               & (resolved["actual_close"] <= resolved["upper"])).mean() * 100)
    regime_pct = resolved["regime"].value_counts(normalize=True).mul(100).to_dict()
    return dict(
        n=len(resolved), mape=float(mape), within_pct=float(within),
        band_pct=band_pct, test_start=test_start_str,
        regime_pct={int(k): float(v) for k, v in regime_pct.items()},
        stored_coverage=float(meta.get("held_out_band_coverage_pct", 0)),
    )


@st.cache_resource
def _load_daily_hl():
    """Load and cache the daily H/L ensemble artefact (or {} if absent)."""
    p = str(DAILY_MODEL_CT)
    if not os.path.exists(p):
        return {}
    return joblib.load(p)


@st.cache_resource
def _load_day_type():
    """Load the 3-class day-type GBM artefact (or None if missing)."""
    p = str(DAY_TYPE_MODEL)
    if not os.path.exists(p):
        return None
    return joblib.load(p)


@st.cache_data(ttl=86400, show_spinner="Classifying day-type …")
def compute_day_type_forecast(target_date_iso):
    """Classify the next 12:00-UTC bar as BigUpper / BigLower / Quiet.

    Uses the H/L model's predictions + the cone regime + a handful of
    raw daily features. Returns:
      predicted_class    – str
      probability        – top-class probability (0-1)
      proba_by_class     – {class: probability}
      target_date        – ISO date of the bar being classified
      as_of_date         – the latest completed bar used
      band_realized      – the test-set selective-accuracy table from the
                           artefact (for the caption)
    Returns None if the artefact is missing or the model can't load.
    """
    art = _load_day_type()
    if art is None:
        return None
    gbm   = art["model"]
    FEATS = art["feature_columns"]
    cone  = _load_cone_7d()

    # Recompute the same features the training script used at the cutoff
    df = _fetch_daily_raw().copy()
    target_date = pd.Timestamp(target_date_iso)
    asof_cutoff = target_date - pd.Timedelta(days=1)
    df = df.loc[df.index <= asof_cutoff].sort_index().ffill(limit=5)
    if df.empty:
        return None
    c = df["btc_close"]; h = df["btc_high"]; l_ = df["btc_low"]; v = df["btc_volume"]
    f = pd.DataFrame(index=df.index)
    ret = np.log(c).diff()
    for k in [3, 7, 14]:           f[f"ret_{k}"] = ret.rolling(k).sum()
    for k in [10, 20, 30]:         f[f"vol_{k}"] = ret.rolling(k).std()
    prev_c = c.shift(1)
    tr_ = pd.concat([(h - l_), (h - prev_c).abs(), (l_ - prev_c).abs()], axis=1).max(axis=1)
    for k in [7, 14, 30]:          f[f"atr_{k}"] = tr_.rolling(k).mean() / c
    f["range_today"] = (h - l_) / c
    f["range_ma7"]   = ((h - l_) / c).rolling(7).mean()
    f["range_ma30"]  = ((h - l_) / c).rolling(30).mean()
    f["range_std30"] = ((h - l_) / c).rolling(30).std()
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    f["macd"]      = macd / c
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    ma20 = c.rolling(20).mean(); sd20 = c.rolling(20).std()
    f["bb_width"]   = (4 * sd20) / ma20
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    f["rsi_14"] = 100 - 100 / (1 + rs)
    dow = df.index.dayofweek
    for i in range(6):
        f[f"dow_{i}"] = (dow == i).astype(float)

    # Cone regime one-hots
    edges = np.asarray(art.get("regime_edges") or cone["regime_edges"])
    reg   = int(np.searchsorted(edges, float(f["range_ma30"].iloc[-1]), side="right").clip(0, 2))
    for r in (0, 1, 2):
        f[f"regime_{r}"] = float(r == reg)

    # H/L model predictions and direction-head probability (use only the
    # as-of row to keep this cheap)
    fc = A["feat_cols"]
    daily_forecast = compute_daily_forecast(target_date_iso)
    if daily_forecast is None:
        return None
    close_asof = daily_forecast["close_asof"]
    pred_high  = daily_forecast["pred_high"]
    pred_low   = daily_forecast["pred_low"]
    p_bull     = daily_forecast.get("p_bull", 0.5)
    f["pred_y_hi"]  = (pred_high - close_asof) / close_asof
    f["pred_y_lo"]  = (close_asof - pred_low)  / close_asof
    f["pred_range"] = f["pred_y_hi"] + f["pred_y_lo"]
    f["pred_skew"]  = f["pred_y_hi"] - f["pred_y_lo"]
    f["p_bull"]     = float(p_bull)

    asof_t = df.index[-1]
    x_row = f.loc[[asof_t], FEATS]
    if x_row.isna().any().any():
        return None
    proba = gbm.predict_proba(x_row)[0]
    cls   = list(gbm.classes_)
    proba_by_class = {c: float(p) for c, p in zip(cls, proba)}
    top_idx = int(np.argmax(proba))
    return dict(
        predicted_class = cls[top_idx],
        probability     = float(proba[top_idx]),
        proba_by_class  = proba_by_class,
        target_date     = target_date,
        as_of_date      = asof_t,
        calibration     = art.get("calibration_meta", {}),
    )


def build_features(df):
    f = pd.DataFrame(index=df.index)
    c, h, l_, v = df["btc_close"], df["btc_high"], df["btc_low"], df["btc_volume"]
    rt = np.log(c).diff()
    for k in [1,2,4,8,12,24,48,72]: f[f"ret_{k}h"] = rt.rolling(k).sum()
    for k in [4,8,24,48]:           f[f"vol_{k}h"] = rt.rolling(k).std()
    prev_c = c.shift(1)
    tr = pd.concat([(h-l_),(h-prev_c).abs(),(l_-prev_c).abs()],axis=1).max(axis=1)
    for k in [4,12,24]: f[f"atr_{k}h"] = tr.rolling(k).mean()/c
    f["range_now"]   = (h-l_)/c
    f["range_ma24"]  = f["range_now"].rolling(24).mean()
    f["range_ma72"]  = f["range_now"].rolling(72).mean()
    f["vol_chg_1"]   = np.log(v).diff()
    f["vol_z_24"]    = (np.log(v)-np.log(v).rolling(24).mean())/np.log(v).rolling(24).std()
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain/loss.replace(0,np.nan)
    f["rsi_14"] = 100 - 100/(1+rs)
    ema12=c.ewm(span=12,adjust=False).mean(); ema26=c.ewm(span=26,adjust=False).mean()
    macd=ema12-ema26
    f["macd"]=macd/c; f["macd_hist"]=(macd-macd.ewm(span=9,adjust=False).mean())/c
    ma24=c.rolling(24).mean(); sd24=c.rolling(24).std()
    f["bb24_width"]=(4*sd24)/ma24
    f["dist_hi_24"]  = c/c.rolling(24).max() - 1
    f["dist_lo_24"]  = c/c.rolling(24).min() - 1
    f["dist_hi_168"] = c/c.rolling(168).max() - 1
    for nm in ["eth","spx","ndx","vix","gold","dxy","tnx"]:
        s = df[f"{nm}_close"]
        f[f"{nm}_ret_1h"]  = np.log(s).diff()
        f[f"{nm}_ret_24h"] = np.log(s).diff(24)
        f[f"{nm}_vol_24h"] = np.log(s).diff().rolling(24).std()
    f["btc_eth_corr_24"] = rt.rolling(24).corr(np.log(df["eth_close"]).diff())
    f["fng"]   = df["fng"]
    f["fng_d1"]= df["fng"].diff()
    f["fng_d7"]= df["fng_d7"]
    f["fng_d24"]=df["fng_d24"]
    hr = df.index.hour; dow = df.index.dayofweek
    f["hr_sin"]=np.sin(2*np.pi*hr/24);  f["hr_cos"]=np.cos(2*np.pi*hr/24)
    f["dow_sin"]=np.sin(2*np.pi*dow/7); f["dow_cos"]=np.cos(2*np.pi*dow/7)
    f["weekend"]=(dow>=5).astype(int)
    f["us_open"]=((hr>=13)&(hr<=20)&(dow<5)).astype(int)
    return f

# ═══════════════════ 30-day look-back metric helpers ════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def compute_30d_daily_hl_metrics(end_date_iso):
    """30-day look-back MAPE, hit-rate and direction accuracy for the daily H/L model.

    Uses ``compute_daily_series`` (already cached) to pull the last 30 target
    bars with realised actuals, then computes per-level metrics.
    Returns None if fewer than 5 bars are available.
    """
    series = compute_daily_series(end_date_iso, days_back=30)
    if series is None or series.empty:
        return None
    have = series["actual_high"].notna() & series["actual_low"].notna()
    s = series[have].copy()
    if len(s) < 5:
        return None
    mape_h = ((s["actual_high"] - s["pred_high"]).abs() / s["actual_high"]).mean() * 100
    mape_l = ((s["actual_low"]  - s["pred_low"] ).abs() / s["actual_low"] ).mean() * 100
    hit_h  = ((s["actual_high"] - s["pred_high"]).abs() / s["actual_high"] <= 0.015).mean() * 100
    hit_l  = ((s["actual_low"]  - s["pred_low"] ).abs() / s["actual_low"]  <= 0.015).mean() * 100
    # Day-to-day direction accuracy (did pred line trend the same way as actuals?)
    d_ah = s["actual_high"].diff(); d_ph = s["pred_high"].diff()
    d_al = s["actual_low" ].diff(); d_pl = s["pred_low" ].diff()
    valid = s.index[1:]
    dir_h = ((np.sign(d_ah[valid]) == np.sign(d_ph[valid])) & (d_ah[valid] != 0)).mean() * 100
    dir_l = ((np.sign(d_al[valid]) == np.sign(d_pl[valid])) & (d_al[valid] != 0)).mean() * 100
    return dict(n=len(s), mape_h=float(mape_h), mape_l=float(mape_l),
                hit_h=float(hit_h), hit_l=float(hit_l),
                dir_h=float(dir_h), dir_l=float(dir_l))


@st.cache_data(ttl=3600, show_spinner=False)
def compute_30d_cone_metrics(end_date_iso):
    """30-day look-back metrics for the 7-day close-cone model.

    Requests 37 days of rolling anchors so that the resolved subset
    (target_date already in the past) covers roughly 30 observations.
    Returns None if fewer than 5 resolved predictions are available.
    """
    rolling = compute_rolling_7d_series(end_date_iso, days_back=37)
    if rolling is None or rolling.empty:
        return None
    resolved = rolling[rolling["actual_close"].notna()].copy()
    if len(resolved) < 5:
        return None
    mape   = ((resolved["actual_close"] - resolved["pred_close"]).abs()
              / resolved["pred_close"]).mean() * 100
    # Recover band_pct from the first row (upper/pred_close - 1)
    band_pct = float(resolved.iloc[0]["upper"] / resolved.iloc[0]["pred_close"] - 1)
    within = (((resolved["actual_close"] >= resolved["lower"])
               & (resolved["actual_close"] <= resolved["upper"])).mean() * 100)
    return dict(n=len(resolved), mape=float(mape),
                within_pct=float(within), band_pct=band_pct)


@st.cache_data(ttl=3600 * 6, show_spinner="Computing 30-day day-type metrics …")
def compute_30d_daytype_metrics(end_date_iso):
    """30-day accuracy for the 3-class day-type (BigUpper / BigLower / Quiet) model.

    For each of the 30 completed target-days ending at ``end_date_iso``:
      1. Retrieve the model's predicted class via ``compute_day_type_forecast``
         (already cached, so only recomputed once per day).
      2. Derive the *actual* label from the realised H/L and the artefact's
         stored ``quiet_threshold`` — matching the training label function
         exactly.  Requires the target bar and the preceding close.
    Returns None if fewer than 5 days have both a prediction and an actual.
    """
    art = _load_day_type()
    if art is None:
        return None
    qthr  = float(art.get("quiet_threshold", 0.03))
    end   = pd.Timestamp(end_date_iso)
    daily = _fetch_daily_raw().copy()
    rows  = []
    for i in range(30, 0, -1):
        target     = end - pd.Timedelta(days=i)
        target_iso = target.strftime("%Y-%m-%d")
        if target not in daily.index:
            continue
        # As-of close: the latest daily bar that closed BEFORE the target bar
        asof_bars = daily.loc[daily.index < target]
        if asof_bars.empty:
            continue
        asof_t      = asof_bars.index[-1]
        close_asof  = float(daily.loc[asof_t, "btc_close"])
        ah          = daily.loc[target, "btc_high"]
        al          = daily.loc[target, "btc_low"]
        if not (pd.notna(ah) and pd.notna(al)):
            continue
        y_hi = (float(ah) - close_asof) / close_asof
        y_lo = (close_asof - float(al))  / close_asof
        rng  = y_hi + y_lo
        actual = ("Quiet" if rng < qthr
                  else ("BigUpper" if y_hi > y_lo else "BigLower"))
        pred_r = compute_day_type_forecast(target_iso)
        if pred_r is None:
            continue
        rows.append(dict(
            target_date    = target,
            predicted      = pred_r["predicted_class"],
            actual         = actual,
            probability    = float(pred_r["probability"]),
            correct        = (pred_r["predicted_class"] == actual),
        ))
    if not rows:
        return None
    df_r = pd.DataFrame(rows)
    acc  = float(df_r["correct"].mean() * 100)
    by_class = {}
    for cls in ["BigUpper", "BigLower", "Quiet"]:
        pred_mask   = df_r["predicted"] == cls
        actual_mask = df_r["actual"]    == cls
        correct_n   = int((pred_mask & actual_mask).sum())
        by_class[cls] = dict(
            pred_n   = int(pred_mask.sum()),
            actual_n = int(actual_mask.sum()),
            correct_n= correct_n,
        )
    return dict(n=len(df_r), accuracy=acc, by_class=by_class)


@st.cache_data(ttl=3600 * 4, show_spinner=False)
def compute_alltime_cone_metrics(end_date_iso):
    """MAPE + band coverage over the full held-out test period for the 7-day cone.

    Anchors ``compute_rolling_7d_series`` at ``test_start`` so only
    genuinely out-of-sample predictions are included.  Returns None when
    fewer than 5 resolved predictions are available.
    """
    cone = _load_cone_7d()
    if cone is None:
        return None
    meta = cone.get("calibration_meta", {})
    test_start_str = meta.get("test_start", "2025-09-21")
    test_start_ts  = pd.Timestamp(test_start_str)
    end_ts = pd.Timestamp(end_date_iso)
    days_back = int((end_ts - test_start_ts).days) + 14  # +7d lag + buffer
    rolling = compute_rolling_7d_series(end_date_iso, days_back=days_back)
    if rolling is None or rolling.empty:
        return None
    rolling = rolling[rolling["anchor_date"] >= test_start_ts].copy()
    resolved = rolling[rolling["actual_close"].notna()]
    if len(resolved) < 5:
        return None
    mape = ((resolved["actual_close"] - resolved["pred_close"]).abs()
            / resolved["pred_close"]).mean() * 100
    band_pct = float(cone.get("band_pct", 0.097))
    within = (((resolved["actual_close"] >= resolved["lower"])
               & (resolved["actual_close"] <= resolved["upper"])).mean() * 100)
    regime_pct = resolved["regime"].value_counts(normalize=True).mul(100).to_dict()
    return dict(
        n=len(resolved), mape=float(mape), within_pct=float(within),
        band_pct=band_pct, test_start=test_start_str,
        regime_pct={int(k): float(v) for k, v in regime_pct.items()},
        stored_coverage=float(meta.get("held_out_band_coverage_pct", 0)),
    )


@st.cache_data(ttl=3600 * 12, show_spinner=False)
def compute_alltime_daytype_metrics(end_date_iso):
    """Per-class precision / recall over the full test period for the day-type model.

    Iterates every calendar day from ``test_start`` to ``end_date_iso``,
    fetches the cached forecast, and derives the actual label from the
    artefact's ``quiet_threshold`` — the same logic as training.
    Returns None when fewer than 5 labelled days exist.
    """
    art = _load_day_type()
    if art is None:
        return None
    meta = art.get("calibration_meta", {})
    test_start_str = meta.get("test_start")
    if not test_start_str:
        return None
    test_start_ts = pd.Timestamp(test_start_str)
    end_ts = pd.Timestamp(end_date_iso)
    qthr  = float(art.get("quiet_threshold", 0.03))
    daily = _fetch_daily_raw().copy()
    rows  = []
    cur   = test_start_ts
    while cur <= end_ts:
        if cur in daily.index:
            asof_bars = daily.loc[daily.index < cur]
            if not asof_bars.empty:
                asof_t     = asof_bars.index[-1]
                close_asof = float(daily.loc[asof_t, "btc_close"])
                ah = daily.loc[cur, "btc_high"]
                al = daily.loc[cur, "btc_low"]
                if pd.notna(ah) and pd.notna(al):
                    y_hi = (float(ah) - close_asof) / close_asof
                    y_lo = (close_asof - float(al))  / close_asof
                    actual = ("Quiet" if (y_hi + y_lo) < qthr
                              else ("BigUpper" if y_hi > y_lo else "BigLower"))
                    pred_r = compute_day_type_forecast(cur.strftime("%Y-%m-%d"))
                    if pred_r is not None:
                        rows.append(dict(
                            predicted=pred_r["predicted_class"],
                            actual=actual,
                            probability=float(pred_r["probability"]),
                            correct=(pred_r["predicted_class"] == actual),
                        ))
        cur += pd.Timedelta(days=1)
    if not rows:
        return None
    df_r = pd.DataFrame(rows)
    acc  = float(df_r["correct"].mean() * 100)
    by_class = {}
    for cls in ["BigUpper", "BigLower", "Quiet"]:
        pm = df_r["predicted"] == cls
        am = df_r["actual"]    == cls
        cn = int((pm & am).sum())
        by_class[cls] = dict(pred_n=int(pm.sum()), actual_n=int(am.sum()), correct_n=cn)
    return dict(n=len(df_r), accuracy=acc, by_class=by_class,
                test_start=test_start_str)


# ─────────────────────────── fetch + predict ──────────────────────────
with st.spinner("Fetching live market data ..."):
    df = fetch_data()
    F  = build_features(df).replace([np.inf,-np.inf], np.nan)
    F  = F[feat_cols]
    # Forward-fill any stale macro features so the latest BTC hour is always
    # usable for inference even when SPX/VIX/TNX haven't ticked recently
    # (weekends, off-hours, holidays). The risk is using slightly stale
    # macro values, which is the right trade-off for a live system.
    F_filled = F.ffill()

# Use the most recent BTC bar where we have at least the core BTC features
valid_mask = F_filled.notna().all(axis=1)
if not valid_mask.any():
    st.error("Not enough recent data to compute features. Try again later.")
    st.stop()
latest_t_global = F_filled.index[valid_mask][-1]
live_spot, live_spot_ts = fetch_live_spot()


# ════════════════════════════════════════════════════════════════════════
# Dashboard renderer — used by both Live and Historical tabs
# ════════════════════════════════════════════════════════════════════════
def render_dashboard(as_of_t, *, is_live, live_spot=None, live_spot_ts=None,
                     hist_picker=None):
    """Render the full dashboard (KPIs + chart + look-back metrics)
    as-of `as_of_t`.  In live mode, `now_utc` is wall-clock and we anchor
    the prediction at the Binance live spot.  In historical mode, `now_utc`
    is the picked timestamp and the anchor is the hourly close at that time."""
    latest_t = as_of_t
    latest_close = float(df.loc[latest_t, "btc_close"])
    next_t = latest_t + pd.Timedelta(hours=1)

    # Daily H/L forecast — labelled by the TARGET date (the bar being predicted).
    # The target bar covers [target 7am CT, target+1 7am CT). Only data through
    # target 7am CT is used (bars with start ≤ target − 1 day).
    #   Live mode   → target = "today_CT" (the date for which 7am CT has most
    #                 recently passed; before 7am CT it stays on the previous day).
    #   Historical  → target = picked_date (the date the user selected).
    if is_live:
        ref_t = datetime.now(timezone.utc)
        target_date = pd.Timestamp((ref_t - timedelta(hours=ANCHOR_HOUR_UTC)).date())
    else:
        picked_date_ct = st.session_state.get("hist_date")
        if picked_date_ct is None:
            ref_t = as_of_t.replace(tzinfo=timezone.utc) if as_of_t.tzinfo is None else as_of_t
            target_date = pd.Timestamp((ref_t - timedelta(hours=ANCHOR_HOUR_UTC)).date())
        else:
            target_date = pd.Timestamp(picked_date_ct)
        # Warn if the picked date is in any model's training window.
        render_replay_in_sample_warning(target_date)
    daily = compute_daily_forecast(target_date.strftime("%Y-%m-%d"))

    # Rolling forecast target (now+1h in live, as_of+1h in historical)
    if is_live:
        now_utc = pd.Timestamp(datetime.now(timezone.utc)).tz_convert(None)
    else:
        now_utc = latest_t
    forecast_target = now_utc + pd.Timedelta(hours=1)

    x_now = F_filled.loc[[latest_t]]
    y_pred = float(model.predict(x_now)[0])

    # Anchor: live spot in live mode, hourly close in historical mode
    if is_live and live_spot is not None:
        anchor_price = live_spot
    else:
        anchor_price = latest_close
    pred_close   = anchor_price * np.exp(y_pred)
    pred_close_up = pred_close * (1 + BAND_PCT)
    pred_close_dn = pred_close * (1 - BAND_PCT)
    expected_ret_pct = (np.exp(y_pred) - 1) * 100
    fng_now = int(df.loc[latest_t, "fng"]) if pd.notna(df.loc[latest_t, "fng"]) else None

    # ─────────────────────────── headline KPIs ────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    if live_spot is not None:
        c1.metric("Live BTC spot (Binance)",
                  f"${live_spot:,.0f}",
                  delta=f"as of {live_spot_ts.strftime('%H:%M:%S')} UTC")
    else:
        c1.metric("Latest BTC close (Yahoo)",
                  f"${latest_close:,.0f}",
                  delta=f"as of {latest_t.strftime('%H:%M')} UTC")
    c2.metric(f"Forecast 1 h from now ({forecast_target.strftime('%H:%M:%S')} UTC)",
              f"${pred_close:,.0f}",
              delta=f"{expected_ret_pct:+.3f}% vs spot")
    c3.metric("Forecast band ±0.5 %",
              f"${pred_close_dn:,.0f} – ${pred_close_up:,.0f}",
              delta=f"width = {2*BAND_PCT*100:.1f} %")
    c4.metric("Fear & Greed (latest daily)",
              f"{fng_now if fng_now is not None else 'n/a'}",
              delta=(f"{df['fng'].diff().iloc[-1]:+.0f} d/d"
                     if fng_now is not None else None))

    # ---------- Daily H/L forecast KPIs (12:00-UTC = 7am-CT bars) ----------
    if daily is not None:
        ws = daily["target_window_start"]; we = daily["target_window_end"]
        td = pd.Timestamp(daily["target_date"])
        st.markdown(
            f"#### 🗓️ Daily H/L forecast for **{td.strftime('%Y-%m-%d')}** — "
            f"window **{ws.strftime('%Y-%m-%d %H:%M')} → {we.strftime('%Y-%m-%d %H:%M')} UTC**  "
            f"(= {td.strftime('%Y-%m-%d')} 7am CT → {(td + pd.Timedelta(days=1)).strftime('%Y-%m-%d')} 7am CT)  "
            f"<small>(uses data through {td.strftime('%Y-%m-%d')} 7am CT — i.e. "
            f"the bar starting {daily['as_of_date'].strftime('%Y-%m-%d')} 12:00 UTC, close "
            f"${daily['close_asof']:,.0f}. Refreshes at 12:00 UTC (7am CT) each day. "
            f"Model: ensemble (Huber+Bayes+GBM-MAE). "
            f"Backtest MAPE H=1.12%, L=1.31%; hit ±1% on 54–60% of test days.)</small>",
            unsafe_allow_html=True,
        )
        d1, d2 = st.columns(2)
        d1.metric("Predicted DAILY HIGH",
                  f"${daily['pred_high']:,.0f}",
                  delta=(f"+{(daily['pred_high']/daily['close_asof']-1)*100:.2f}% vs close"))
        d1.caption(
            f"±1.5 % band ${daily['pred_high']*0.985:,.0f} – ${daily['pred_high']*1.015:,.0f}"
        )
        d2.metric("Predicted DAILY LOW",
                  f"${daily['pred_low']:,.0f}",
                  delta=(f"{(daily['pred_low']/daily['close_asof']-1)*100:.2f}% vs close"))
        d2.caption(
            f"±1.5 % band ${daily['pred_low']*0.985:,.0f} – ${daily['pred_low']*1.015:,.0f}"
        )
        # Direction-head bias (if model artefact has one)
        pb = daily.get("p_bull")
        if pb is not None:
            pred_asym = (daily["pred_high"] - daily["close_asof"]) \
                        - (daily["close_asof"] - daily["pred_low"])
            bias_word = "bullish" if pred_asym > 0 else ("bearish" if pred_asym < 0 else "neutral")
            bias_color = "#1a7f37" if pred_asym > 0 else ("#b91c1c" if pred_asym < 0 else "#555")
            be = daily.get("beta_eff")
            be_str = (f" · β_eff = <b>{be:.2f}</b>" if be is not None else "")
            st.markdown(
                f"<small>🧭 <b>Direction head</b>: "
                f"P(bullish bar) = <b>{pb*100:.1f}%</b> · "
                f"resulting bias = <span style='color:{bias_color}'><b>{bias_word}</b></span> "
                f"(predicted up-tail − down-tail = ${pred_asym:+,.0f})"
                f"{be_str}</small>",
                unsafe_allow_html=True,
            )

    # ────── 3-class day-type classifier (Big Upper / Big Lower / Quiet) ──
    day_type = compute_day_type_forecast(target_date.strftime("%Y-%m-%d"))
    if day_type is not None:
        DT_COLORS  = {"BigUpper": "#16a34a", "BigLower": "#dc2626", "Quiet": "#475569"}
        DT_EMOJI   = {"BigUpper": "🔼",      "BigLower": "🔽",      "Quiet": "▫️"}
        DT_LABEL   = {"BigUpper": "Big Upper Movement",
                      "BigLower": "Big Lower Movement",
                      "Quiet":    "Quiet Movement"}
        pc   = day_type["predicted_class"]
        prob = day_type["probability"]
        probs = day_type["proba_by_class"]
        cal   = day_type["calibration"]
        td    = pd.Timestamp(day_type["target_date"])
        # Confidence-gated guidance — match the artefact's stored selective
        # table to a coarse band so the caption stays honest.
        if   prob >= 0.65: gating = "high confidence"
        elif prob >= 0.55: gating = "moderate confidence"
        elif prob >= 0.45: gating = "low confidence"
        else:              gating = "very low confidence"
        st.markdown(
            f"#### {DT_EMOJI[pc]} Day-type forecast — "
            f"<span style='color:{DT_COLORS[pc]}'>"
            f"<b>{DT_LABEL[pc]}</b></span> "
            f"<small>(confidence <b>{prob*100:.0f}%</b> · {gating})</small>",
            unsafe_allow_html=True,
        )
        # Probability bar: three horizontal segments with widths ∝ probability
        order = ["BigUpper", "BigLower", "Quiet"]
        bar_html = "<div style='display:flex; width:100%; height:24px; border-radius:6px; overflow:hidden; border:1px solid #ddd; font-size:11px; font-weight:600; color:white;'>"
        for k in order:
            pct = probs.get(k, 0.0) * 100
            border = ("3px solid #111" if k == pc else "0")
            bar_html += (
                f"<div title='{DT_LABEL[k]}: {pct:.1f}%' "
                f"style='flex: {max(probs.get(k,0.001), 0.001)}; "
                f"background:{DT_COLORS[k]}; "
                f"display:flex; align-items:center; justify-content:center; "
                f"border-right:{border};'>"
                f"{DT_EMOJI[k]} {pct:.0f}%"
                f"</div>"
            )
        bar_html += "</div>"
        st.markdown(bar_html, unsafe_allow_html=True)
        # Honest caption: test acc + selective accuracy at the relevant band
        test_acc = cal.get("test_accuracy_pct")
        test_n   = cal.get("test_n")
        sel      = cal.get("selective", [])
        # Pick the largest threshold the current prob satisfies
        matched  = max((s for s in sel if prob >= s["thr"]),
                       key=lambda s: s["thr"], default=None)
        sel_note = ""
        if matched is not None:
            sel_note = (f" When the model is at least {matched['thr']:.0%} "
                        f"confident it covers ~{matched['coverage_pct']:.0f}% of "
                        f"days at {matched['accuracy_pct']:.0f}% accuracy on the held-out tail.")
        st.caption(
            f"3-class day-type classifier (GBM, 31 features). Predicts the "
            f"realised next-day H/L bar shape for **{td.strftime('%Y-%m-%d')}**. "
            f"Hold-out unconditional accuracy = {test_acc:.0f}% on n={test_n} days "
            f"(majority baseline ≈ 33% on three balanced classes)."
            + sel_note
        )

    # ────── Historical picker (date strip, calendar, hour slider,
    # bookmarks) rendered RIGHT ABOVE the plots so the user can navigate
    # to a different day without scrolling back up.
    if hist_picker is not None:
        hist_picker()

    # ─────────────────────────── walk-forward look-back ───────────────────
    # Live mode  → last LOOKBACK_HOURS hours up to now.
    # Historical → fixed 24h bar [picked_date 12:00 UTC, +24h), matching the
    # daily model's anchor exactly. That is 7am CDT (summer) / 6am CST (winter)
    # — labelled loosely as "7am CT" per README §"Day-boundary contract".
    # Anchoring at 12:00 UTC (not DST-following local 7am) ensures the hourly
    # chart's 24h window covers the same bar the daily prediction is for.
    if is_live:
        look_idx = F_filled.index[(F_filled.index <= latest_t) & valid_mask][-LOOKBACK_HOURS:]
        win_start_utc = None  # signals to chart code: use look_idx-derived range
        win_end_utc   = None
    else:
        _CT = "America/Chicago"
        picked_date_ct = st.session_state.get("hist_date")
        if picked_date_ct is None:
            picked_date_ct = (latest_t.tz_localize("UTC")
                                       .tz_convert(_CT).tz_localize(None).date())
        # Fixed 12:00 UTC anchor — same boundary as _rebucket_12utc / daily model.
        win_start_utc = pd.Timestamp(picked_date_ct) + pd.Timedelta(hours=ANCHOR_HOUR_UTC)
        win_end_utc   = win_start_utc + pd.Timedelta(days=1)
        # CT-time edges for the chart's x-axis range (DST-correct labels).
        day_start_ct = (win_start_utc.tz_localize("UTC").tz_convert(_CT)
                                     .tz_localize(None))
        day_end_ct   = (win_end_utc.tz_localize("UTC").tz_convert(_CT)
                                   .tz_localize(None))
        look_idx = F_filled.index[(F_filled.index >= win_start_utc) &
                                  (F_filled.index <  win_end_utc) &
                                  valid_mask]
    y_lb = model.predict(F_filled.loc[look_idx])
    close_lb     = df.loc[look_idx, "btc_close"].values
    pred_close_lb = close_lb * np.exp(y_lb)
    pred_up_lb    = close_lb * np.exp(y_lb + 1.96*sigma)
    pred_dn_lb    = close_lb * np.exp(y_lb - 1.96*sigma)
    target_dates_lb = [d + pd.Timedelta(hours=1) for d in look_idx]
    actual_lb = []
    for d in target_dates_lb:
        actual_lb.append(float(df.loc[d, "btc_close"]) if d in df.index else np.nan)
    actual_lb = np.array(actual_lb)

    # back-test metrics on the look-back window
    mask = ~np.isnan(actual_lb)
    if mask.sum() > 5:
        rel = np.abs(pred_close_lb[mask] - actual_lb[mask]) / actual_lb[mask]
        pred_ret_lb = y_lb[mask]
        actual_ret_lb = np.log(actual_lb[mask] / close_lb[mask])
        lb_metrics = {
            "MAPE": rel.mean()*100,
            "hit3":  (rel<=0.03).mean()*100,
            "hit1":  (rel<=0.01).mean()*100,
            "hit0.5":(rel<=0.005).mean()*100,
            "dir_acc": np.mean(np.sign(pred_ret_lb)==np.sign(actual_ret_lb))*100,
        }
    else:
        lb_metrics = None

    # ─────────────────────────── chart ────────────────────────────────────
    fig = go.Figure()
    xt = pd.to_datetime(target_dates_lb)

    # All plot x-axes display US Central time (auto-handles CDT/CST via DST).
    # Source variables stay UTC; we convert only at the plot layer.
    CT_TZ = "America/Chicago"
    def _ct(ts):
        return ts.tz_localize("UTC").tz_convert(CT_TZ).tz_localize(None)
    look_idx_ct        = _ct(look_idx)
    xt_ct              = _ct(xt)
    now_ct             = _ct(pd.DatetimeIndex([now_utc]))[0]
    forecast_target_ct = _ct(pd.DatetimeIndex([forecast_target]))[0]

    # --- ±0.5 % band around PAST PREDICTIONS (not around the actuals) ---
    # Plotted at the target time of each prediction (= look_idx + 1h = xt).
    pred_band_up = pred_close_lb * (1 + BAND_PCT)
    pred_band_dn = pred_close_lb * (1 - BAND_PCT)
    fig.add_trace(go.Scatter(
        x=xt_ct, y=pred_band_up, mode="lines",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=xt_ct, y=pred_band_dn, mode="lines",
        line=dict(color="rgba(0,0,0,0)"), fill="tonexty",
        fillcolor="rgba(65,105,225,0.18)",
        name=f"Pred ±{BAND_PCT*100:.1f}% band",
        hoverinfo="skip",
    ))

    # --- Past actuals — 1-minute rolling price (live) / hourly (historical) ---
    # In live mode fetch Binance 1m klines so the black line updates every
    # minute and shows intraday detail.  Fall back to hourly bars in
    # historical replay or when the 1m fetch fails.
    # _1m_x / _1m_y are kept as shared variables so the coloured hourly
    # markers (added after marker_colors is computed below) can reuse them.
    _btc_1m = fetch_btc_1m() if is_live else pd.DataFrame()
    _1m_x = pd.DatetimeIndex([])
    _1m_y = np.array([], dtype=float)

    if is_live and not _btc_1m.empty:
        _1m_idx_ct = _ct(_btc_1m.index)          # UTC tz-naive → CT tz-naive
        _1m_cls    = _btc_1m["close"].values
        _win_start = (look_idx_ct[0] if len(look_idx_ct)
                      else now_ct - pd.Timedelta(hours=LOOKBACK_HOURS))
        _mask_1m   = (_1m_idx_ct >= _win_start) & (_1m_idx_ct <= now_ct)
        _1m_x = _1m_idx_ct[_mask_1m]
        _1m_y = _1m_cls[_mask_1m]

    if is_live and len(_1m_x) > 0:
        fig.add_trace(go.Scatter(
            x=list(_1m_x), y=list(_1m_y),
            mode="lines",
            line=dict(color="black", width=1.5),
            name="Actual price (1m, updates every min)",
            hovertemplate="%{x|%Y-%m-%d %H:%M} CT<br>$%{y:,.0f}<extra></extra>",
        ))
    else:
        # Historical mode or 1m fetch unavailable — hourly bars with flat
        # extension to the Now vline so there is no gap before the marker.
        _x_actual = list(look_idx_ct)
        _y_actual = list(close_lb)
        if is_live and _y_actual:
            _x_actual = _x_actual + [now_ct]
            _y_actual = _y_actual + [_y_actual[-1]]
        fig.add_trace(go.Scatter(
            x=_x_actual, y=_y_actual, mode="lines",
            line=dict(color="black", width=2),
            name="Actual close",
            hovertemplate="%{x|%Y-%m-%d %H:%M} CT<br>$%{y:,.0f}<extra></extra>",
        ))

    # --- Past predictions as discrete markers (no lagging line) ---
    # Coloured by directional correctness: green = predicted direction matched
    # the realised direction; red = miscalled; grey = no actual yet.
    mask_r = ~np.isnan(actual_lb)
    pred_dir = np.sign(y_lb)
    actual_ret_lb_full = np.where(mask_r,
                                  np.log(np.where(mask_r, actual_lb, 1) / close_lb),
                                  np.nan)
    actual_dir = np.where(mask_r, np.sign(actual_ret_lb_full), np.nan)
    correct = (pred_dir == actual_dir) & mask_r
    marker_colors = np.where(~mask_r, "lightgrey",
                    np.where(correct, "seagreen", "indianred"))

    fig.add_trace(go.Scatter(
        x=xt_ct, y=pred_close_lb, mode="markers",
        marker=dict(color=marker_colors, size=8,
                    line=dict(width=1, color="white")),
        name="Past hourly predictions (green = correct dir.)",
        customdata=np.column_stack([y_lb*100, actual_lb]),
        hovertemplate=("Past pred for %{x|%Y-%m-%d %H:%M} CT<br>"
                       "Pred close: $%{y:,.0f}<br>"
                       "Pred return: %{customdata[0]:+.3f}%<br>"
                       "Actual close: $%{customdata[1]:,.0f}<extra></extra>"),
    ))

    # Coloured hourly close markers on the 1m line.
    # Use actual_lb directly (the same Yahoo hourly closes used to evaluate
    # prediction correctness) — no need to search the 1m data.
    # Only plot where a realized close exists (actual_lb is not NaN).
    if len(_1m_x) > 0 or not is_live:
        _hm_x, _hm_y, _hm_col = [], [], []
        for _j, (_xt_ts, _act) in enumerate(zip(xt_ct, actual_lb)):
            if not np.isnan(_act):
                _hm_x.append(_xt_ts)
                _hm_y.append(float(_act))
                _hm_col.append(
                    str(marker_colors[_j]) if _j < len(marker_colors)
                    else "lightgrey"
                )
        if _hm_x:
            fig.add_trace(go.Scatter(
                x=_hm_x, y=_hm_y, mode="markers",
                marker=dict(color=_hm_col, size=9, symbol="circle",
                            line=dict(color="white", width=1.5)),
                name="Hourly realized close (green=correct dir.)",
                hovertemplate=(
                    "Realized %{x|%Y-%m-%d %H:%M} CT<br>"
                    "$%{y:,.0f}<extra></extra>"
                ),
            ))

    # --- LIVE FORECAST: prominent zone (rolling 1h-from-now) ---
    fig.add_vrect(
        x0=now_ct, x1=forecast_target_ct + pd.Timedelta(minutes=5),
        fillcolor="khaki", opacity=0.30, line_width=0, layer="below",
    )

    # Connector segment from anchor (live spot or latest close) to the forecast
    fig.add_trace(go.Scatter(
        x=[now_ct, forecast_target_ct],
        y=[anchor_price, pred_close],
        mode="lines",
        line=dict(color="darkorange", width=2),
        showlegend=False, hoverinfo="skip",
    ))

    # Forecast marker: STAR with ±0.5 % error bars at the ROLLING target time
    fig.add_trace(go.Scatter(
        x=[forecast_target_ct], y=[pred_close],
        mode="markers",
        marker=dict(symbol="star", size=14, color="darkorange",
                    line=dict(width=1.5, color="black")),
        error_y=dict(
            type="data", symmetric=False,
            array=[pred_close_up - pred_close],
            arrayminus=[pred_close - pred_close_dn],
            thickness=2.5, width=10, color="darkorange",
        ),
        name=f"🎯 Live rolling forecast → {forecast_target_ct.strftime('%H:%M:%S')} CT",
        hovertemplate=(f"<b>Live rolling 1 h forecast</b><br>"
                       f"For: %{{x|%Y-%m-%d %H:%M:%S}} CT<br>"
                       f"Pred: $%{{y:,.0f}}<br>"
                       f"Band (±{BAND_PCT*100:.1f}%%): ${pred_close_dn:,.0f} – ${pred_close_up:,.0f}"
                       f"<extra></extra>"),
    ))

    # --- Current wall-clock "Now" line ---
    fig.add_vline(x=now_ct, line=dict(color="crimson", width=2, dash="dash"))
    fig.add_annotation(
        x=now_ct, y=1.0, xref="x", yref="paper",
        text=f"<b>Now</b> {now_ct.strftime('%H:%M')} CT",
        showarrow=False, yanchor="bottom", xanchor="center",
        bgcolor="rgba(255,255,255,0.92)", bordercolor="crimson", borderwidth=1,
        font=dict(color="crimson", size=11),
    )

    # --- Daily H/L forecast: full-width flat threshold lines + CI ---
    actual_hi_now = None
    actual_lo_now = None
    if daily is not None:
        wstart = pd.Timestamp(daily["target_window_start"])
        wend   = pd.Timestamp(daily["target_window_end"])

        # Threshold lines: span the FULL x-axis
        fig.add_hline(
            y=daily["pred_high"],
            line=dict(color="green", width=2.5, dash="dot"),
            annotation_text=f"Daily Pred HIGH ${daily['pred_high']:,.0f}",
            annotation_position="top right",
            annotation_font=dict(color="green", size=12),
            annotation_bgcolor="rgba(255,255,255,0.92)",
            annotation_bordercolor="green",
            annotation_borderwidth=1,
        )
        fig.add_hline(
            y=daily["pred_low"],
            line=dict(color="red", width=2.5, dash="dot"),
            annotation_text=f"Daily Pred LOW ${daily['pred_low']:,.0f}",
            annotation_position="bottom right",
            annotation_font=dict(color="red", size=12),
            annotation_bgcolor="rgba(255,255,255,0.92)",
            annotation_bordercolor="red",
            annotation_borderwidth=1,
        )
        # ±2.5% bands around HIGH (green) and LOW (red).  When the ±2.5%
        # zones would overlap, we CLIP each at the midpoint between the two
        # predictions so the green and red never blend into yellow.
        DAILY_BAND_PCT = 0.015
        mid = (daily["pred_high"] + daily["pred_low"]) / 2
        hi_raw_dn = daily["pred_high"] * (1 - DAILY_BAND_PCT)
        hi_raw_up = daily["pred_high"] * (1 + DAILY_BAND_PCT)
        lo_raw_dn = daily["pred_low"]  * (1 - DAILY_BAND_PCT)
        lo_raw_up = daily["pred_low"]  * (1 + DAILY_BAND_PCT)
        hi_band_dn = max(hi_raw_dn, mid)         # clip green at mid
        hi_band_up = hi_raw_up
        lo_band_dn = lo_raw_dn
        lo_band_up = min(lo_raw_up, mid)         # clip red at mid
        if hi_band_up > hi_band_dn:
            fig.add_hrect(
                y0=hi_band_dn, y1=hi_band_up,
                fillcolor="rgba(0,170,0,0.16)", line_width=0, layer="below",
                annotation_text=f"±{DAILY_BAND_PCT*100:.1f}% around HIGH",
                annotation_position="top left",
                annotation_font=dict(color="green", size=10),
            )
        if lo_band_up > lo_band_dn:
            fig.add_hrect(
                y0=lo_band_dn, y1=lo_band_up,
                fillcolor="rgba(220,30,30,0.16)", line_width=0, layer="below",
                annotation_text=f"±{DAILY_BAND_PCT*100:.1f}% around LOW",
                annotation_position="bottom left",
                annotation_font=dict(color="red", size=10),
            )

        # --- Realised daily HIGH / LOW for the displayed bar (historical
        # mode only; the live target bar may still be in progress). Solid
        # lines distinguish realised from the dotted predicted thresholds.
        if not is_live:
            try:
                _daily_raw = _fetch_daily_raw()
                _tgt = pd.Timestamp(daily["target_date"]).normalize()
                if _tgt in _daily_raw.index:
                    _ah = _daily_raw.loc[_tgt, "btc_high"]
                    _al = _daily_raw.loc[_tgt, "btc_low"]
                    if pd.notna(_ah) and pd.notna(_al):
                        actual_hi_now = float(_ah)
                        actual_lo_now = float(_al)
            except Exception:
                pass
        if actual_hi_now is not None:
            fig.add_hline(
                y=actual_hi_now,
                line=dict(color="darkgreen", width=2, dash="solid"),
                annotation_text=f"Actual HIGH ${actual_hi_now:,.0f}",
                annotation_position="top left",
                annotation_font=dict(color="darkgreen", size=11),
                annotation_bgcolor="rgba(255,255,255,0.92)",
                annotation_bordercolor="darkgreen",
                annotation_borderwidth=1,
            )
            fig.add_hline(
                y=actual_lo_now,
                line=dict(color="darkred", width=2, dash="solid"),
                annotation_text=f"Actual LOW ${actual_lo_now:,.0f}",
                annotation_position="bottom left",
                annotation_font=dict(color="darkred", size=11),
                annotation_bgcolor="rgba(255,255,255,0.92)",
                annotation_bordercolor="darkred",
                annotation_borderwidth=1,
            )

    # --- Live spot price marker (Binance, current second) ---
    if live_spot is not None:
        fig.add_trace(go.Scatter(
            x=[now_ct], y=[live_spot], mode="markers",
            marker=dict(symbol="circle", size=11, color="crimson",
                        line=dict(width=1.5, color="white")),
            name=f"Live spot (Binance)",
            hovertemplate=(f"<b>Live BTC spot</b><br>"
                           f"%{{x|%Y-%m-%d %H:%M:%S}} CT<br>"
                           f"$%{{y:,.0f}}<extra></extra>"),
        ))
    # Label the forecast time at the bottom of the khaki zone (rolling target)
    fig.add_annotation(
        x=forecast_target_ct, y=0, xref="x", yref="paper",
        text=f"forecast: <b>{forecast_target_ct.strftime('%H:%M:%S')} CT</b>",
        showarrow=False, yanchor="top", xanchor="center", yshift=-25,
        bgcolor="rgba(255,255,255,0.92)", bordercolor="darkorange", borderwidth=1,
        font=dict(color="darkorange", size=11),
    )

    # x-axis range:
    #   live mode  → last look-back window, padded right for the rolling ⭐
    #   historical → fixed 7am-CT day [day_start_ct, day_end_ct]
    next_t_ct = _ct(pd.DatetimeIndex([next_t]))[0]
    if is_live:
        right_edge = max(next_t_ct, forecast_target_ct) + pd.Timedelta(minutes=30)
        left_edge  = (look_idx_ct[0] - pd.Timedelta(hours=1)) if len(look_idx_ct) else right_edge - pd.Timedelta(hours=LOOKBACK_HOURS)
    else:
        left_edge  = day_start_ct
        right_edge = day_end_ct
    fig.update_xaxes(
        tickformat="%d-%b %H:%M",
        title_text="Time (US Central)",
        title_standoff=12,
        range=[left_edge, right_edge],
    )
    # Bound y-axis tightly to actual data + key reference levels.
    # The daily 95 % CI hrects span ~5-10 % which would otherwise distort the plot.
    y_pts = list(close_lb)
    if len(_1m_y) > 0:
        y_pts.extend(_1m_y.tolist())
    y_pts.extend([pred_close, pred_close_up, pred_close_dn])
    if live_spot is not None: y_pts.append(live_spot)
    if daily is not None:
        y_pts.extend([daily["pred_high"], daily["pred_low"]])
    if actual_hi_now is not None:
        y_pts.extend([actual_hi_now, actual_lo_now])
    y_min, y_max = min(y_pts), max(y_pts)
    y_pad = max((y_max - y_min) * 0.10, y_max * 0.003)  # ≥0.3% breathing room
    fig.update_yaxes(range=[y_min - y_pad, y_max + y_pad])
    fig.update_layout(
        template="plotly_white", height=600, hovermode="x unified",
        title=dict(
            text=(f"<b>BTC live ROLLING 1 h forecast → "
                  f"{forecast_target_ct.strftime('%H:%M:%S')} CT</b>"
                  "<br><span style='font-size:13px;color:#555'>"
                  f"Refreshes every {REFRESH_SECONDS}s; target slides forward each minute.  "
                  f"Last {LOOKBACK_HOURS}h actuals (black) ±0.5 % shaded.  "
                  f"Past-hour dots: <span style='color:seagreen'>green=correct dir.</span>/"
                  f"<span style='color:indianred'>red=miscalled</span>.  "
                  f"⭐ rolling forecast, anchored at live spot, ±0.5 % band.  "
                  f"Dotted lines = daily H/L threshold (refreshes at 12:00 UTC = 7am CT)."
                  "</span>"),
            x=0.01, xanchor="left", y=0.98, yanchor="top",
        ),
        yaxis_title="BTC / USD",
        margin=dict(t=115, r=210, b=80, l=70),
        legend=dict(orientation="v", x=1.02, xanchor="left", y=1.0, yanchor="top",
                    bgcolor="rgba(255,255,255,0.95)", bordercolor="#ccc",
                    borderwidth=1, font=dict(size=11)),
    )
    st.plotly_chart(fig, use_container_width=True,
                    key=f"chart_hourly_{'live' if is_live else 'hist'}")

    # ═══════ NEW PLOT — Daily H/L: last 7 days predictions + actuals ═══════
    # end_target = the rightmost target date — the bar STARTING on this date
    # is the one highlighted as the current forecast:
    #   Live      → today_CT's bar (starts today 7am CT, ends tomorrow 7am CT).
    #   Historical → bar starting on picked_date (matches the KPI card).
    if is_live:
        end_target = pd.Timestamp(
            (datetime.now(timezone.utc) - timedelta(hours=ANCHOR_HOUR_UTC)).date()
        )
    else:
        picked_date_ct = st.session_state.get("hist_date")
        if picked_date_ct is not None:
            end_target = pd.Timestamp(picked_date_ct)
        else:
            ref = as_of_t.replace(tzinfo=timezone.utc) if as_of_t.tzinfo is None else as_of_t
            end_target = pd.Timestamp((ref - timedelta(hours=ANCHOR_HOUR_UTC)).date())
    series = compute_daily_series(end_target.strftime("%Y-%m-%d"), days_back=7)

    if len(series) > 0:
        st.markdown(
            f"#### 📈 Daily H/L — predictions vs actuals "
            f"(last 7 bars + current target; each bar opens 12:00 UTC = 7am CT, "
            f"highlighted target = **{end_target.strftime('%Y-%m-%d')}**)"
        )
        fig2 = go.Figure()
        # ±2 % uncertainty bands around each predicted line — added first so
        # they render behind the prediction & actual markers.
        DAILY_BAND_PCT = 0.02
        # HIGH ±2% band (green tint)
        fig2.add_trace(go.Scatter(
            x=series["target_date"], y=series["pred_high"] * (1 + DAILY_BAND_PCT),
            mode="lines", line=dict(color="rgba(34,139,34,0)"),
            hoverinfo="skip", showlegend=False,
        ))
        fig2.add_trace(go.Scatter(
            x=series["target_date"], y=series["pred_high"] * (1 - DAILY_BAND_PCT),
            mode="lines", line=dict(color="rgba(34,139,34,0)"),
            fill="tonexty", fillcolor="rgba(34,139,34,0.13)",
            name=f"HIGH ±{DAILY_BAND_PCT*100:.0f}% band", hoverinfo="skip",
        ))
        # LOW ±2% band (red tint)
        fig2.add_trace(go.Scatter(
            x=series["target_date"], y=series["pred_low"] * (1 + DAILY_BAND_PCT),
            mode="lines", line=dict(color="rgba(220,20,60,0)"),
            hoverinfo="skip", showlegend=False,
        ))
        fig2.add_trace(go.Scatter(
            x=series["target_date"], y=series["pred_low"] * (1 - DAILY_BAND_PCT),
            mode="lines", line=dict(color="rgba(220,20,60,0)"),
            fill="tonexty", fillcolor="rgba(220,20,60,0.13)",
            name=f"LOW ±{DAILY_BAND_PCT*100:.0f}% band", hoverinfo="skip",
        ))
        # Predicted HIGH line
        fig2.add_trace(go.Scatter(
            x=series["target_date"], y=series["pred_high"],
            mode="lines+markers",
            line=dict(color="green", width=2.2, dash="dot"),
            marker=dict(size=9, symbol="circle"),
            name="Predicted HIGH",
            hovertemplate=("Bar starts %{x|%Y-%m-%d} 7am CT<br>"
                           "Pred HIGH $%{y:,.0f}<extra></extra>"),
        ))
        # Predicted LOW line
        fig2.add_trace(go.Scatter(
            x=series["target_date"], y=series["pred_low"],
            mode="lines+markers",
            line=dict(color="red", width=2.2, dash="dot"),
            marker=dict(size=9, symbol="circle"),
            name="Predicted LOW",
            hovertemplate=("Bar starts %{x|%Y-%m-%d} 7am CT<br>"
                           "Pred LOW $%{y:,.0f}<extra></extra>"),
        ))
        # Actual HIGH/LOW where realised
        have = series["actual_high"].notna()
        if have.any():
            fig2.add_trace(go.Scatter(
                x=series.loc[have,"target_date"], y=series.loc[have,"actual_high"],
                mode="markers",
                marker=dict(symbol="x-thin", size=13,
                            line=dict(width=3, color="darkgreen")),
                name="Actual HIGH",
                hovertemplate=("Bar %{x|%Y-%m-%d} (7am CT → 7am CT next day)<br>"
                               "Actual HIGH $%{y:,.0f}<extra></extra>"),
            ))
            fig2.add_trace(go.Scatter(
                x=series.loc[have,"target_date"], y=series.loc[have,"actual_low"],
                mode="markers",
                marker=dict(symbol="x-thin", size=13,
                            line=dict(width=3, color="darkred")),
                name="Actual LOW",
                hovertemplate=("Bar %{x|%Y-%m-%d} (7am CT → 7am CT next day)<br>"
                               "Actual LOW $%{y:,.0f}<extra></extra>"),
            ))

        # Direction-correctness per day, computed SEPARATELY for HIGH and LOW.
        # For each day D, compare the day-over-day MOVE of the plotted lines:
        #   HIGH direction realised  = sign(actual_high[D] − actual_high[D-1])
        #   HIGH direction predicted = sign(pred_high[D]   − pred_high[D-1])
        #   LOW  direction realised  = sign(actual_low[D]  − actual_low[D-1])
        #   LOW  direction predicted = sign(pred_low[D]    − pred_low[D-1])
        # Match iff signs agree. This reflects what's visible in the chart:
        # did the predicted line trend the same way as the realised line?
        dir_hit_str = ""
        yest_act_hi  = series["actual_high"].shift(1)
        yest_act_lo  = series["actual_low"].shift(1)
        yest_pred_hi = series["pred_high"].shift(1)
        yest_pred_lo = series["pred_low"].shift(1)
        d_hi_act  = series["actual_high"] - yest_act_hi
        d_lo_act  = series["actual_low"]  - yest_act_lo
        d_hi_pred = series["pred_high"]   - yest_pred_hi
        d_lo_pred = series["pred_low"]    - yest_pred_lo
        have_hi = (yest_act_hi.notna() & series["actual_high"].notna()
                   & yest_pred_hi.notna() & series["pred_high"].notna())
        have_lo = (yest_act_lo.notna() & series["actual_low"].notna()
                   & yest_pred_lo.notna() & series["pred_low"].notna())
        correct_hi = (np.sign(d_hi_act) == np.sign(d_hi_pred)) & have_hi & (np.sign(d_hi_act) != 0)
        correct_lo = (np.sign(d_lo_act) == np.sign(d_lo_pred)) & have_lo & (np.sign(d_lo_act) != 0)
        if have_hi.any() or have_lo.any():
            hits_hi = int(correct_hi[have_hi].sum())
            hits_lo = int(correct_lo[have_lo].sum())
            n_hi = int(have_hi.sum())
            n_lo = int(have_lo.sum())
            dir_hit_str = (
                f"  <span style='color:#1a7f37'><b>HIGH</b> line-trend hit-rate: "
                f"{hits_hi}/{n_hi} = {hits_hi/n_hi*100:.0f}%</span> · "
                f"<span style='color:#b91c1c'><b>LOW</b> line-trend hit-rate: "
                f"{hits_lo}/{n_lo} = {hits_lo/n_lo*100:.0f}%</span>"
                if n_hi and n_lo else dir_hit_str
            )
            # HIGH row (top, green) and LOW row (slightly below, red)
            for i in range(len(series)):
                td_label = series["target_date"].iloc[i].strftime('%Y-%m-%d')
                if bool(have_hi.iloc[i]):
                    ok_h = bool(correct_hi.iloc[i])
                    fig2.add_annotation(
                        x=series["target_date"].iloc[i], y=1.10,
                        xref="x", yref="paper",
                        text=("✓" if ok_h else "✗"),
                        showarrow=False, yanchor="bottom", xanchor="center",
                        font=dict(color="seagreen", size=18, family="monospace"),
                        hovertext=(
                            f"{td_label} — HIGH day-to-day direction<br>"
                            f"Pred line: {'up' if d_hi_pred.iloc[i]>0 else 'down' if d_hi_pred.iloc[i]<0 else 'flat'} "
                            f"(Δ pred_high vs yest pred_high = ${d_hi_pred.iloc[i]:+,.0f})<br>"
                            f"Actual line: {'up' if d_hi_act.iloc[i]>0 else 'down' if d_hi_act.iloc[i]<0 else 'flat'} "
                            f"(Δ actual_high vs yest actual_high = ${d_hi_act.iloc[i]:+,.0f})<br>"
                            f"{'Matched' if ok_h else 'Missed'}"
                        ),
                    )
                if bool(have_lo.iloc[i]):
                    ok_l = bool(correct_lo.iloc[i])
                    fig2.add_annotation(
                        x=series["target_date"].iloc[i], y=1.02,
                        xref="x", yref="paper",
                        text=("✓" if ok_l else "✗"),
                        showarrow=False, yanchor="bottom", xanchor="center",
                        font=dict(color="indianred", size=18, family="monospace"),
                        hovertext=(
                            f"{td_label} — LOW day-to-day direction<br>"
                            f"Pred line: {'up' if d_lo_pred.iloc[i]>0 else 'down' if d_lo_pred.iloc[i]<0 else 'flat'} "
                            f"(Δ pred_low vs yest pred_low = ${d_lo_pred.iloc[i]:+,.0f})<br>"
                            f"Actual line: {'up' if d_lo_act.iloc[i]>0 else 'down' if d_lo_act.iloc[i]<0 else 'flat'} "
                            f"(Δ actual_low vs yest actual_low = ${d_lo_act.iloc[i]:+,.0f})<br>"
                            f"{'Matched' if ok_l else 'Missed'}"
                        ),
                    )

        # Highlight the right-most point (the target forecast)
        last_t = series["target_date"].iloc[-1]
        fig2.add_vrect(x0=last_t - pd.Timedelta(hours=12),
                       x1=last_t + pd.Timedelta(hours=12),
                       fillcolor="khaki", opacity=0.30, line_width=0,
                       layer="below")
        fig2.add_annotation(
            x=last_t, y=1.18, xref="x", yref="paper",
            text=f"<b>target forecast</b><br>{last_t.strftime('%Y-%m-%d')}",
            showarrow=False, yanchor="bottom", xanchor="center",
            bgcolor="rgba(255,255,255,0.92)", bordercolor="goldenrod",
            borderwidth=1, font=dict(color="goldenrod", size=10),
        )

        fig2.update_layout(
            template="plotly_white", height=500, hovermode="x unified",
            title=dict(
                text=("<b>Daily H/L — predictions (dotted) vs actuals (X markers)</b>"
                      "<br><span style='font-size:12px;color:#555'>"
                      "8 target days: last 7 with realised values + target forecast highlighted. "
                      "Top rows: <b style='color:#1a7f37'>green</b> = HIGH line "
                      "day-over-day direction (pred-trend vs actual-trend), "
                      "<b style='color:#b91c1c'>red</b> = LOW line "
                      "day-over-day direction. ✓ both lines moved the same way / ✗ opposed."
                      f"{dir_hit_str}"
                      "</span>"),
                x=0.01, xanchor="left", y=0.97, yanchor="top",
            ),
            yaxis_title="BTC / USD",
            xaxis_title="Target bar start date (US Central, bar opens 7am CT)",
            margin=dict(t=140, r=200, b=60, l=70),
            legend=dict(orientation="v", x=1.02, xanchor="left", y=1.0,
                        yanchor="top", bgcolor="rgba(255,255,255,0.95)",
                        bordercolor="#ccc", borderwidth=1, font=dict(size=11)),
        )
        fig2.update_xaxes(tickformat="%a %d-%b")
        st.plotly_chart(fig2, use_container_width=True,
                        key=f"chart_daily_{'live' if is_live else 'hist'}")

    # ─────────── 7-day close cone — rolling daily predictions chart ──────────
    # For each anchor day in the last 21 days the model predicts the close
    # 7 days out. Realized actuals are overlaid once that date passes,
    # coloured green (within ±9.7 % band) or red (outside).
    # The headline KPI line below still uses compute_7d_close_cone_forecast
    # for the current-anchor regime/return metadata.
    cone7 = compute_7d_close_cone_forecast(target_date.strftime("%Y-%m-%d"))
    if cone7 is not None:
        ret_pct = (np.exp(cone7["regime_median_logret"]) - 1) * 100
        st.markdown(
            f"#### 📅 7-day close-price cone — regime: **{cone7['regime_label']}**  "
            f"<small>(as-of {pd.Timestamp(cone7['asof_date']).strftime('%Y-%m-%d')} "
            f"close ${cone7['asof_close']:,.0f} → "
            f"forecast {cone7['pred_date'].strftime('%Y-%m-%d')} "
            f"${cone7['pred_close']:,.0f} "
            f"({ret_pct:+.2f}% regime median return), "
            f"band ±{cone7['band_pct']*100:.1f}%)</small>",
            unsafe_allow_html=True,
        )
        band_pct = cone7["band_pct"]
        # ── Rolling daily 7-day predictions (replaces weekly-spaced chart) ──
        # For each anchor day in the last 21 days, predict the close 7 days
        # out and compare against the realized close once that date passes.
        rolling7 = compute_rolling_7d_series(
            target_date.strftime("%Y-%m-%d"), days_back=21
        )

        if len(rolling7) > 0:
            resolved = rolling7[rolling7["actual_close"].notna()].copy()
            future   = rolling7[rolling7["actual_close"].isna()].copy()

            fig3 = go.Figure()

            # 1. ±band envelope fill across all target dates (background)
            all_td = rolling7.sort_values("target_date")
            fig3.add_trace(go.Scatter(
                x=list(all_td["target_date"]), y=list(all_td["upper"]),
                mode="lines", line=dict(color="rgba(37,99,235,0)"),
                hoverinfo="skip", showlegend=False,
            ))
            fig3.add_trace(go.Scatter(
                x=list(all_td["target_date"]), y=list(all_td["lower"]),
                mode="lines", fill="tonexty",
                fillcolor="rgba(147,197,253,0.22)",
                line=dict(color="rgba(37,99,235,0)"),
                name=f"±{band_pct*100:.1f}% prediction band",
                hoverinfo="skip",
            ))

            # 2. Prediction line + diamond markers at RESOLVED target dates
            if len(resolved) > 0:
                fig3.add_trace(go.Scatter(
                    x=list(resolved["target_date"]),
                    y=list(resolved["pred_close"]),
                    mode="lines+markers",
                    name="7d prediction (resolved target)",
                    line=dict(color="#2563eb", width=1.6, dash="dot"),
                    marker=dict(size=9, color="#2563eb", symbol="diamond",
                                line=dict(color="white", width=1)),
                    customdata=list(
                        resolved["anchor_date"].dt.strftime("%Y-%m-%d")
                    ),
                    hovertemplate=(
                        "Target %{x|%Y-%m-%d} — anchored %{customdata}<br>"
                        "Predicted $%{y:,.0f}"
                        "<extra></extra>"
                    ),
                ))

            # 3. Realized close LINE through resolved target dates (black)
            if len(resolved) > 0:
                fig3.add_trace(go.Scatter(
                    x=list(resolved["target_date"]),
                    y=list(resolved["actual_close"]),
                    mode="lines",
                    name="Realized close",
                    line=dict(color="#1f2937", width=2.2),
                    hovertemplate=(
                        "Realized %{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>"
                    ),
                ))
                # Accuracy markers: green circle = within band, red ✕ = outside
                within_mask = (
                    (resolved["actual_close"] >= resolved["lower"])
                    & (resolved["actual_close"] <= resolved["upper"])
                )
                within_df  = resolved[within_mask]
                outside_df = resolved[~within_mask]
                if len(within_df) > 0:
                    fig3.add_trace(go.Scatter(
                        x=list(within_df["target_date"]),
                        y=list(within_df["actual_close"]),
                        mode="markers",
                        name=f"Actual ✅ within ±{band_pct*100:.1f}% band",
                        marker=dict(size=11, color="#16a34a", symbol="circle",
                                    line=dict(color="white", width=1.5)),
                        customdata=[
                            f"{(r.actual_close - r.pred_close) / r.pred_close * 100:+.1f}%"
                            for r in within_df.itertuples()
                        ],
                        hovertemplate=(
                            "Actual %{x|%Y-%m-%d}<br>$%{y:,.0f}<br>"
                            "Error vs prediction: %{customdata}<br>"
                            "✅ Within band<extra></extra>"
                        ),
                    ))
                if len(outside_df) > 0:
                    fig3.add_trace(go.Scatter(
                        x=list(outside_df["target_date"]),
                        y=list(outside_df["actual_close"]),
                        mode="markers",
                        name=f"Actual ❌ outside ±{band_pct*100:.1f}% band",
                        marker=dict(size=12, color="#dc2626", symbol="x-thin",
                                    line=dict(color="#dc2626", width=3)),
                        customdata=[
                            f"{(r.actual_close - r.pred_close) / r.pred_close * 100:+.1f}%"
                            for r in outside_df.itertuples()
                        ],
                        hovertemplate=(
                            "Actual %{x|%Y-%m-%d}<br>$%{y:,.0f}<br>"
                            "Error vs prediction: %{customdata}<br>"
                            "❌ Outside band<extra></extra>"
                        ),
                    ))

            # 4. FUTURE predictions (target_date > latest realized data)
            if len(future) > 0:
                # Dashed connector from last resolved pred → future preds
                if len(resolved) > 0:
                    bridge_x = (
                        [resolved["target_date"].iloc[-1]]
                        + list(future["target_date"])
                    )
                    bridge_y = (
                        [resolved["pred_close"].iloc[-1]]
                        + list(future["pred_close"])
                    )
                else:
                    bridge_x = list(future["target_date"])
                    bridge_y = list(future["pred_close"])
                fig3.add_trace(go.Scatter(
                    x=bridge_x, y=bridge_y,
                    mode="lines",
                    line=dict(color="#2563eb", width=1.6, dash="dot"),
                    hoverinfo="skip", showlegend=False,
                ))
                # Near-term future points (all but the latest anchor)
                if len(future) > 1:
                    near = future.iloc[:-1]
                    fig3.add_trace(go.Scatter(
                        x=list(near["target_date"]),
                        y=list(near["pred_close"]),
                        mode="markers",
                        name="Near-term forecasts",
                        marker=dict(size=10, color="#93c5fd", symbol="star",
                                    line=dict(color="#2563eb", width=1)),
                        customdata=list(
                            near["anchor_date"].dt.strftime("%Y-%m-%d")
                        ),
                        hovertemplate=(
                            "Target %{x|%Y-%m-%d} — anchored %{customdata}<br>"
                            "Predicted $%{y:,.0f}<extra></extra>"
                        ),
                    ))
                # Latest anchor's prediction = the primary +7d star
                cur = future.iloc[-1]
                fig3.add_trace(go.Scatter(
                    x=[cur["target_date"]], y=[cur["pred_close"]],
                    mode="markers+text",
                    name=(f"+7d forecast "
                          f"(from {cur['anchor_date'].strftime('%Y-%m-%d')})"),
                    marker=dict(size=15, color="#2563eb", symbol="star",
                                line=dict(color="white", width=1.5)),
                    text=[f"${cur['pred_close']:,.0f}"],
                    textposition="top center",
                    error_y=dict(
                        type="data",
                        array=[cur["upper"] - cur["pred_close"]],
                        arrayminus=[cur["pred_close"] - cur["lower"]],
                        color="#60a5fa", thickness=2.5, width=7,
                    ),
                    hovertemplate=(
                        f"Target %{{x|%Y-%m-%d}}<br>"
                        f"Anchored {cur['anchor_date'].strftime('%Y-%m-%d')}<br>"
                        f"Predicted ${cur['pred_close']:,.0f}<br>"
                        f"Band ${cur['lower']:,.0f} – ${cur['upper']:,.0f}"
                        "<extra></extra>"
                    ),
                ))

            # Accuracy summary (resolved predictions only)
            accuracy_note = ""
            if len(resolved) > 0:
                mape = (
                    (resolved["actual_close"] - resolved["pred_close"]).abs()
                    / resolved["pred_close"] * 100
                ).mean()
                within_pct = (
                    ((resolved["actual_close"] >= resolved["lower"])
                     & (resolved["actual_close"] <= resolved["upper"]))
                    .mean() * 100
                )
                accuracy_note = (
                    f"  Rolling {len(resolved)} resolved predictions: "
                    f"MAPE = {mape:.1f}%; "
                    f"{within_pct:.0f}% within ±{band_pct*100:.1f}% band."
                )

            n_future   = len(future)
            n_resolved = len(resolved)
            title_detail = (
                "Blue ◆ = predicted close anchored 7 days before each target "
                "(rolling daily). "
                "Black line = realized close at target dates. "
                f"Green ● = actual within / Red ✕ = actual outside "
                f"±{band_pct*100:.1f}% band. "
                f"⭐ = {n_future} active forecast"
                f"{'s' if n_future != 1 else ''} (target still in future)."
            )
            fig3.update_layout(
                height=430, template="plotly_white",
                title=dict(
                    text=(
                        "<b>7-day close prediction — rolling daily anchors</b>"
                        "<br><span style='font-size:12px;color:#555'>"
                        + title_detail
                        + (f"  {accuracy_note}" if accuracy_note else "")
                        + "</span>"
                    ),
                    x=0.01, xanchor="left", y=0.97, yanchor="top",
                ),
                yaxis_title="BTC / USD",
                xaxis_title="Target date (close price predicted for this date, 7 days after anchor)",
                margin=dict(t=115, r=30, b=50, l=70),
                legend=dict(orientation="h", x=0, xanchor="left", y=1.08,
                            yanchor="bottom", bgcolor="rgba(255,255,255,0.95)",
                            bordercolor="#ccc", borderwidth=1, font=dict(size=11)),
            )
            fig3.update_xaxes(tickformat="%d-%b")
            st.plotly_chart(fig3, use_container_width=True,
                            key=f"chart_7d_rolling_{'live' if is_live else 'hist'}")
            st.caption(
                f"Rolling daily 7-day prediction chart: each anchor day generates "
                f"a ±{band_pct*100:.1f}% cone forecast for the close 7 days later. "
                "Blue diamonds = what was predicted; black line = what happened. "
                "Green circles = actual fell inside the cone (hit); "
                "red ✕ = actual outside (miss). "
                "Stars = active forecasts whose target date hasn't arrived yet. "
                + accuracy_note
                + "  The fixed ±9.7% interval corresponds to ≈88% empirical "
                "coverage on the held-out 8-month tail "
                "(see `notebooks/btc_7d_close_research.ipynb`)."
            )
        else:
            # Fallback when rolling data is unavailable (insufficient history)
            st.info(
                "Rolling 7-day prediction series unavailable "
                "(not enough historical data to compute regime features). "
                f"Current +7d forecast: "
                f"${cone7['pred_close']:,.0f} "
                f"(regime: {cone7['regime_label']}, "
                f"band ±{band_pct*100:.1f}%)."
            )

    # ─────────── 14-day close cone — rolling daily predictions chart ─────────
    cone14 = compute_14d_close_cone_forecast(target_date.strftime("%Y-%m-%d"))
    if cone14 is not None:
        ret_pct_14 = (np.exp(cone14["regime_median_logret"]) - 1) * 100
        ml_tag = " · GBM" if cone14.get("use_ml") else " · regime median"
        st.markdown(
            f"#### 📆 14-day close-price cone — regime: **{cone14['regime_label']}**  "
            f"<small>(as-of {pd.Timestamp(cone14['asof_date']).strftime('%Y-%m-%d')} "
            f"close ${cone14['asof_close']:,.0f} → "
            f"forecast {cone14['pred_date'].strftime('%Y-%m-%d')} "
            f"${cone14['pred_close']:,.0f} "
            f"(regime median {ret_pct_14:+.2f}%), "
            f"band ±{cone14['band_pct']*100:.1f}%{ml_tag})</small>",
            unsafe_allow_html=True,
        )
        band_pct_14 = cone14["band_pct"]
        rolling14 = compute_rolling_14d_series(
            target_date.strftime("%Y-%m-%d"), days_back=30
        )

        if len(rolling14) > 0:
            resolved14 = rolling14[rolling14["actual_close"].notna()].copy()
            future14   = rolling14[rolling14["actual_close"].isna()].copy()

            fig14 = go.Figure()

            # 1. ±band envelope fill across all target dates
            all_td14 = rolling14.sort_values("target_date")
            fig14.add_trace(go.Scatter(
                x=list(all_td14["target_date"]), y=list(all_td14["upper"]),
                mode="lines", line=dict(color="rgba(124,58,237,0)"),
                hoverinfo="skip", showlegend=False,
            ))
            fig14.add_trace(go.Scatter(
                x=list(all_td14["target_date"]), y=list(all_td14["lower"]),
                mode="lines", fill="tonexty",
                fillcolor="rgba(196,181,253,0.22)",
                line=dict(color="rgba(124,58,237,0)"),
                name=f"±{band_pct_14*100:.1f}% prediction band",
                hoverinfo="skip",
            ))

            # 2. Prediction line at RESOLVED target dates (purple)
            if len(resolved14) > 0:
                fig14.add_trace(go.Scatter(
                    x=list(resolved14["target_date"]),
                    y=list(resolved14["pred_close"]),
                    mode="lines+markers",
                    name="14d prediction (resolved target)",
                    line=dict(color="#7c3aed", width=1.6, dash="dot"),
                    marker=dict(size=9, color="#7c3aed", symbol="diamond",
                                line=dict(color="white", width=1)),
                    customdata=list(
                        resolved14["anchor_date"].dt.strftime("%Y-%m-%d")
                    ),
                    hovertemplate=(
                        "Target %{x|%Y-%m-%d} — anchored %{customdata}<br>"
                        "Predicted $%{y:,.0f}"
                        "<extra></extra>"
                    ),
                ))

            # 3. Realized close line through resolved dates (black)
            if len(resolved14) > 0:
                fig14.add_trace(go.Scatter(
                    x=list(resolved14["target_date"]),
                    y=list(resolved14["actual_close"]),
                    mode="lines",
                    name="Realized close",
                    line=dict(color="#1f2937", width=2.2),
                    hovertemplate=(
                        "Realized %{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>"
                    ),
                ))
                within_mask14 = (
                    (resolved14["actual_close"] >= resolved14["lower"])
                    & (resolved14["actual_close"] <= resolved14["upper"])
                )
                within14_df  = resolved14[within_mask14]
                outside14_df = resolved14[~within_mask14]
                if len(within14_df) > 0:
                    fig14.add_trace(go.Scatter(
                        x=list(within14_df["target_date"]),
                        y=list(within14_df["actual_close"]),
                        mode="markers",
                        name=f"Actual ✅ within ±{band_pct_14*100:.1f}% band",
                        marker=dict(size=11, color="#16a34a", symbol="circle",
                                    line=dict(color="white", width=1.5)),
                        customdata=[
                            f"{(r.actual_close - r.pred_close) / r.pred_close * 100:+.1f}%"
                            for r in within14_df.itertuples()
                        ],
                        hovertemplate=(
                            "Actual %{x|%Y-%m-%d}<br>$%{y:,.0f}<br>"
                            "Error vs prediction: %{customdata}<br>"
                            "✅ Within band<extra></extra>"
                        ),
                    ))
                if len(outside14_df) > 0:
                    fig14.add_trace(go.Scatter(
                        x=list(outside14_df["target_date"]),
                        y=list(outside14_df["actual_close"]),
                        mode="markers",
                        name=f"Actual ❌ outside ±{band_pct_14*100:.1f}% band",
                        marker=dict(size=12, color="#dc2626", symbol="x-thin",
                                    line=dict(color="#dc2626", width=3)),
                        customdata=[
                            f"{(r.actual_close - r.pred_close) / r.pred_close * 100:+.1f}%"
                            for r in outside14_df.itertuples()
                        ],
                        hovertemplate=(
                            "Actual %{x|%Y-%m-%d}<br>$%{y:,.0f}<br>"
                            "Error vs prediction: %{customdata}<br>"
                            "❌ Outside band<extra></extra>"
                        ),
                    ))

            # 4. Future predictions (target > latest data)
            if len(future14) > 0:
                if len(resolved14) > 0:
                    bridge_x = (
                        [resolved14["target_date"].iloc[-1]]
                        + list(future14["target_date"])
                    )
                    bridge_y = (
                        [resolved14["pred_close"].iloc[-1]]
                        + list(future14["pred_close"])
                    )
                else:
                    bridge_x = list(future14["target_date"])
                    bridge_y = list(future14["pred_close"])
                fig14.add_trace(go.Scatter(
                    x=bridge_x, y=bridge_y,
                    mode="lines",
                    line=dict(color="#7c3aed", width=1.6, dash="dot"),
                    hoverinfo="skip", showlegend=False,
                ))
                if len(future14) > 1:
                    near14 = future14.iloc[:-1]
                    fig14.add_trace(go.Scatter(
                        x=list(near14["target_date"]),
                        y=list(near14["pred_close"]),
                        mode="markers",
                        name="Near-term forecasts",
                        marker=dict(size=10, color="#c4b5fd", symbol="star",
                                    line=dict(color="#7c3aed", width=1)),
                        customdata=list(
                            near14["anchor_date"].dt.strftime("%Y-%m-%d")
                        ),
                        hovertemplate=(
                            "Target %{x|%Y-%m-%d} — anchored %{customdata}<br>"
                            "Predicted $%{y:,.0f}<extra></extra>"
                        ),
                    ))
                # Latest anchor = primary +14d star
                cur14 = future14.iloc[-1]
                fig14.add_trace(go.Scatter(
                    x=[cur14["target_date"]], y=[cur14["pred_close"]],
                    mode="markers+text",
                    name=(f"+14d forecast "
                          f"(from {cur14['anchor_date'].strftime('%Y-%m-%d')})"),
                    marker=dict(size=15, color="#7c3aed", symbol="star",
                                line=dict(color="white", width=1.5)),
                    text=[f"${cur14['pred_close']:,.0f}"],
                    textposition="top center",
                    error_y=dict(
                        type="data",
                        array=[cur14["upper"] - cur14["pred_close"]],
                        arrayminus=[cur14["pred_close"] - cur14["lower"]],
                        color="#a78bfa", thickness=2.5, width=7,
                    ),
                    hovertemplate=(
                        f"Target %{{x|%Y-%m-%d}}<br>"
                        f"Anchored {cur14['anchor_date'].strftime('%Y-%m-%d')}<br>"
                        f"Predicted ${cur14['pred_close']:,.0f}<br>"
                        f"Band ${cur14['lower']:,.0f} – ${cur14['upper']:,.0f}"
                        "<extra></extra>"
                    ),
                ))

            # Accuracy summary
            accuracy_note_14 = ""
            if len(resolved14) > 0:
                mape14 = (
                    (resolved14["actual_close"] - resolved14["pred_close"]).abs()
                    / resolved14["pred_close"] * 100
                ).mean()
                within_pct14 = (
                    ((resolved14["actual_close"] >= resolved14["lower"])
                     & (resolved14["actual_close"] <= resolved14["upper"]))
                    .mean() * 100
                )
                accuracy_note_14 = (
                    f"  Rolling {len(resolved14)} resolved predictions: "
                    f"MAPE = {mape14:.1f}%; "
                    f"{within_pct14:.0f}% within ±{band_pct_14*100:.1f}% band."
                )

            n_future14   = len(future14)
            title14_detail = (
                "Purple ◆ = predicted close anchored 14 days before each target "
                "(rolling daily). "
                "Black line = realized close at target dates. "
                f"Green ● = actual within / Red ✕ = actual outside "
                f"±{band_pct_14*100:.1f}% band. "
                f"⭐ = {n_future14} active forecast"
                f"{'s' if n_future14 != 1 else ''} (target still in future)."
            )
            fig14.update_layout(
                height=430, template="plotly_white",
                title=dict(
                    text=(
                        "<b>14-day close prediction — rolling daily anchors</b>"
                        "<br><span style='font-size:12px;color:#555'>"
                        + title14_detail
                        + (f"  {accuracy_note_14}" if accuracy_note_14 else "")
                        + "</span>"
                    ),
                    x=0.01, xanchor="left", y=0.97, yanchor="top",
                ),
                yaxis_title="BTC / USD",
                xaxis_title="Target date (close price predicted for this date, 14 days after anchor)",
                margin=dict(t=115, r=30, b=50, l=70),
                legend=dict(orientation="h", x=0, xanchor="left", y=1.08,
                            yanchor="bottom", bgcolor="rgba(255,255,255,0.95)",
                            bordercolor="#ccc", borderwidth=1, font=dict(size=11)),
            )
            fig14.update_xaxes(tickformat="%d-%b")
            st.plotly_chart(fig14, use_container_width=True,
                            key=f"chart_14d_rolling_{'live' if is_live else 'hist'}")
            st.caption(
                f"Rolling daily 14-day prediction chart: each anchor day generates "
                f"a ±{band_pct_14*100:.1f}% cone forecast for the close 14 days later. "
                "Purple diamonds = what was predicted; black line = what happened. "
                "Green circles = actual fell inside the cone (hit); "
                "red ✕ = actual outside (miss). "
                "Stars = active forecasts whose target date hasn't arrived yet. "
                + accuracy_note_14
                + "  Model: GBM point prediction (106 features: BTC momentum, "
                "macro, yield spread, econ activity, energy, ETH/BTC ratio, "
                "Chaikin Money Flow) + empirical ±band from return distribution. "
                "Trained on data through Sep 2025; OOS baseline: "
                "MAPE ≈8.7%, direction accuracy ≈58.5%, coverage ≈89%."
            )
        else:
            st.info(
                "Rolling 14-day prediction series unavailable "
                "(not enough historical data to compute regime features). "
                f"Current +14d forecast: "
                f"${cone14['pred_close']:,.0f} "
                f"(regime: {cone14['regime_label']}, "
                f"band ±{band_pct_14*100:.1f}%)."
            )

    # ══════════════════════════════════════════════════════════════════════
    # Accuracy panel — 30-day rolling  vs  full held-out test set
    # ══════════════════════════════════════════════════════════════════════

    def _badge(v30, vtest, lo_better=True, tol=0.20):
        """Drift badge: 🟢 ≤tol · 🟡 tol–2×tol · 🔴 >2×tol relative degradation.

        ``lo_better=True``  → a *rise* in the metric is bad (MAPE, MAE).
        ``lo_better=False`` → a *fall*  in the metric is bad (accuracy, hit-rate).
        Returns ⚪ when the test baseline is zero or None.
        """
        if vtest is None or vtest == 0:
            return "⚪"
        rel = (v30 - vtest) / abs(vtest)
        bad = rel if lo_better else -rel   # positive value = worse
        if bad <= tol:       return "🟢"
        if bad <= tol * 2.0: return "🟡"
        return "🔴"

    st.markdown("---")
    st.subheader("📊 Model accuracy — 30-day rolling vs full test set")
    st.caption(
        "Each metric shows the **30-day rolling value** with Δ vs the **full "
        "held-out test-set baseline** stored at training time.  "
        "Drift badge: 🟢 ≤ 20 % degradation · 🟡 20–40 % · 🔴 > 40 %."
    )

    # ── Compute 30-day hourly metrics (used across both accuracy & drift) ─
    lb30_start = latest_t - pd.Timedelta(days=30)
    lb30_idx   = F_filled.index[
        (F_filled.index > lb30_start) & (F_filled.index <= latest_t) & valid_mask
    ]
    hourly_30d = None   # dict with 30d metrics + y_mean/y_std for drift
    _y30_preds = None   # raw predictions array (kept for drift section)
    if len(lb30_idx) > 0:
        _y30_preds = model.predict(F_filled.loc[lb30_idx])
        _c30       = df.loc[lb30_idx, "btc_close"].values
        _pc30      = _c30 * np.exp(_y30_preds)
        _tgt30     = [d + pd.Timedelta(hours=1) for d in lb30_idx]
        _act30     = np.array([
            float(df.loc[d, "btc_close"]) if d in df.index else np.nan
            for d in _tgt30
        ])
        _m30 = ~np.isnan(_act30)
        if _m30.sum() > 5:
            _rel30 = np.abs(_pc30[_m30] - _act30[_m30]) / _act30[_m30]
            _pr30  = _y30_preds[_m30]
            _ar30  = np.log(_act30[_m30] / _c30[_m30])
            hourly_30d = dict(
                n       = int(_m30.sum()),
                mape    = float(_rel30.mean() * 100),
                hit3    = float((_rel30 <= 0.03).mean()  * 100),
                hit1    = float((_rel30 <= 0.01).mean()  * 100),
                hit05   = float((_rel30 <= 0.005).mean() * 100),
                dir_acc = float(np.mean(np.sign(_pr30) == np.sign(_ar30)) * 100),
                y_mean  = float(_pr30.mean() * 100),   # pct log-return
                y_std   = float(_pr30.std()  * 100),
            )

    # Pre-fetch other model results (cached, so instant on subsequent renders)
    _end_iso   = target_date.strftime("%Y-%m-%d")
    hl30       = compute_30d_daily_hl_metrics(_end_iso)
    cone30     = compute_30d_cone_metrics(_end_iso)
    cone14_30  = compute_30d_cone_14d_metrics(_end_iso)
    dt30       = compute_30d_daytype_metrics(_end_iso)
    cone_at    = compute_alltime_cone_metrics(_end_iso)
    cone14_at  = compute_alltime_cone_14d_metrics(_end_iso)
    dt_at      = compute_alltime_daytype_metrics(_end_iso)
    _hl_art    = _load_daily_hl()
    _hl_meta   = _hl_art.get("calibration_meta", {})
    _hl_mtest  = _hl_meta.get("metrics", {})
    _dt_art    = _load_day_type()
    _dt_meta   = (_dt_art.get("calibration_meta", {}) if _dt_art else {})
    _cone_art  = _load_cone_7d()
    _cone_meta = (_cone_art.get("calibration_meta", {}) if _cone_art else {})
    _cone14_art  = _load_cone_14d()
    _cone14_meta = (_cone14_art.get("calibration_meta", {}) if _cone14_art else {})
    _mt_h     = A.get("metrics_test", {})

    # ── 1. Hourly close model ────────────────────────────────────────────
    st.markdown("#### ⏱️ Hourly close model (GBM)")
    _ts_h = A.get("test_start"); _te_h = A.get("test_end")
    if _ts_h and _te_h:
        st.caption(
            f"Test period: **{pd.Timestamp(_ts_h).date()} → "
            f"{pd.Timestamp(_te_h).date()}** — metrics frozen from training run."
        )
    if hourly_30d and _mt_h:
        _defs_h = [
            ("MAPE",       "mape",    "MAPE_pct",    True,  "%.2f %%"),
            ("Hit ±1 %",   "hit1",    "hit1_pct",    False, "%.1f %%"),
            ("Hit ±0.5 %", "hit05",   "hit05_pct",   False, "%.1f %%"),
            ("Dir acc",    "dir_acc", "dir_acc_pct", False, "%.1f %%"),
        ]
        _hc = st.columns(len(_defs_h))
        for _ci, (_lbl, _k30, _ktest, _lo, _fmt) in enumerate(_defs_h):
            _v30   = hourly_30d[_k30]
            _vtest = _mt_h.get(_ktest, 0)
            _em    = _badge(_v30, _vtest, _lo)
            _hc[_ci].metric(
                f"{_em} {_lbl}",
                _fmt % _v30,
                delta=f"{_v30 - _vtest:+.2f} pp  (test: {_fmt % _vtest})",
                delta_color="inverse" if _lo else "normal",
                help=(f"30-day n = {hourly_30d['n']} · "
                      f"test baseline {_fmt % _vtest}  "
                      f"({pd.Timestamp(_ts_h).date() if _ts_h else '?'} → "
                      f"{pd.Timestamp(_te_h).date() if _te_h else '?'})"),
            )
    elif hourly_30d:
        _hc2 = st.columns(4)
        _hc2[0].metric("MAPE",       f"{hourly_30d['mape']:.2f} %")
        _hc2[1].metric("Hit ±1 %",   f"{hourly_30d['hit1']:.1f} %")
        _hc2[2].metric("Hit ±0.5 %", f"{hourly_30d['hit05']:.1f} %")
        _hc2[3].metric("Dir acc",    f"{hourly_30d['dir_acc']:.1f} %")
    else:
        st.info("Insufficient data for 30-day hourly metrics.")

    # ── 2. Daily H/L model ───────────────────────────────────────────────
    st.markdown("#### 📅 Daily H/L model (ensemble — Huber + Bayes + GBM-MAE)")
    if _hl_meta:
        st.caption(
            f"Test period: **{_hl_meta.get('test_start','?')} → "
            f"{_hl_meta.get('test_end','?')}** · n = {_hl_meta.get('test_n','?')} bars"
        )
    if hl30 and _hl_mtest:
        _defs_hl = [
            ("MAPE HIGH",        "mape_h", "MAPE_H",             True,  "%.2f %%"),
            ("MAPE LOW",         "mape_l", "MAPE_L",             True,  "%.2f %%"),
            ("Hit ±1.5 % HIGH",  "hit_h",  "hit2_H",             False, "%.1f %%"),
            ("Hit ±1.5 % LOW",   "hit_l",  "hit2_L",             False, "%.1f %%"),
            ("Dir acc HIGH",     "dir_h",  "direction_hit_rate",  False, "%.1f %%"),
            ("Dir acc LOW",      "dir_l",  "direction_hit_rate",  False, "%.1f %%"),
        ]
        _hlc = st.columns(len(_defs_hl))
        for _ci, (_lbl, _k30, _ktest, _lo, _fmt) in enumerate(_defs_hl):
            _v30   = hl30[_k30]
            _vtest = _hl_mtest.get(_ktest, 0)
            _em    = _badge(_v30, _vtest, _lo)
            _hlc[_ci].metric(
                f"{_em} {_lbl}",
                _fmt % _v30,
                delta=f"{_v30 - _vtest:+.2f} pp  (test: {_fmt % _vtest})",
                delta_color="inverse" if _lo else "normal",
                help=f"30-day n = {hl30['n']} · test baseline {_fmt % _vtest}",
            )
    elif hl30:
        _hlc2 = st.columns(6)
        _hlc2[0].metric("MAPE HIGH",       f"{hl30['mape_h']:.2f} %")
        _hlc2[1].metric("MAPE LOW",        f"{hl30['mape_l']:.2f} %")
        _hlc2[2].metric("Hit ±1.5% HIGH",  f"{hl30['hit_h']:.1f} %")
        _hlc2[3].metric("Hit ±1.5% LOW",   f"{hl30['hit_l']:.1f} %")
        _hlc2[4].metric("Dir acc HIGH",    f"{hl30['dir_h']:.1f} %")
        _hlc2[5].metric("Dir acc LOW",     f"{hl30['dir_l']:.1f} %")
    else:
        st.info("Insufficient data for 30-day daily H/L metrics.")

    # ── 3. 7-day close cone ──────────────────────────────────────────────
    st.markdown("#### 📐 7-day close cone (rolling daily anchors)")
    if _cone_meta:
        st.caption(
            f"Test period: **{_cone_meta.get('test_start','?')} → "
            f"{_cone_meta.get('test_end','?')}** · "
            f"stored coverage: "
            f"{_cone_meta.get('held_out_band_coverage_pct', 0):.1f} %"
            + (f"  ·  computed MAPE: {cone_at['mape']:.2f} %  (n={cone_at['n']})"
               if cone_at else "")
        )
    if cone30:
        _vtest_cov  = cone_at["within_pct"] if cone_at else _cone_meta.get("held_out_band_coverage_pct", 0)
        _vtest_mape = cone_at["mape"]        if cone_at else None
        _cc = st.columns(3)
        _em_cov = _badge(cone30["within_pct"], _vtest_cov, lo_better=False)
        _cc[0].metric(
            f"{_em_cov} Within ±{cone30['band_pct']*100:.1f} % band",
            f"{cone30['within_pct']:.1f} %",
            delta=(f"{cone30['within_pct'] - _vtest_cov:+.1f} pp  "
                   f"(test: {_vtest_cov:.1f} %)"),
            delta_color="normal",
            help=f"30-day n = {cone30['n']} · test coverage {_vtest_cov:.1f} %",
        )
        if _vtest_mape:
            _em_mape = _badge(cone30["mape"], _vtest_mape, lo_better=True)
            _cc[1].metric(
                f"{_em_mape} MAPE",
                f"{cone30['mape']:.2f} %",
                delta=(f"{cone30['mape'] - _vtest_mape:+.2f} pp  "
                       f"(test: {_vtest_mape:.2f} %)"),
                delta_color="inverse",
            )
        else:
            _cc[1].metric("MAPE", f"{cone30['mape']:.2f} %")
        _cc[2].metric("Resolved (30 d)", str(cone30["n"]),
                      help="Number of 7-day forecasts whose target date has passed")
    else:
        st.info("Insufficient data for 30-day cone metrics.")

    # ── 4. 14-day close cone ─────────────────────────────────────────────
    st.markdown("#### 📆 14-day close cone (GBM + empirical band)")
    if _cone14_meta:
        st.caption(
            f"Test period: **{_cone14_meta.get('test_start','?')} → "
            f"{_cone14_meta.get('test_end','?')}** · "
            f"stored coverage: "
            f"{_cone14_meta.get('held_out_band_coverage_pct', 0):.1f} %"
            + (f"  ·  computed MAPE: {cone14_at['mape']:.2f} %  (n={cone14_at['n']})"
               if cone14_at else "")
        )
    if cone14_30:
        _vtest_cov14  = cone14_at["within_pct"] if cone14_at else _cone14_meta.get("held_out_band_coverage_pct", 0)
        _vtest_mape14 = cone14_at["mape"]        if cone14_at else None
        _cc14 = st.columns(3)
        _em_cov14 = _badge(cone14_30["within_pct"], _vtest_cov14, lo_better=False)
        _cc14[0].metric(
            f"{_em_cov14} Within ±{cone14_30['band_pct']*100:.1f} % band",
            f"{cone14_30['within_pct']:.1f} %",
            delta=(f"{cone14_30['within_pct'] - _vtest_cov14:+.1f} pp  "
                   f"(test: {_vtest_cov14:.1f} %)"),
            delta_color="normal",
            help=f"30-day n = {cone14_30['n']} · test coverage {_vtest_cov14:.1f} %",
        )
        if _vtest_mape14:
            _em_mape14 = _badge(cone14_30["mape"], _vtest_mape14, lo_better=True)
            _cc14[1].metric(
                f"{_em_mape14} MAPE",
                f"{cone14_30['mape']:.2f} %",
                delta=(f"{cone14_30['mape'] - _vtest_mape14:+.2f} pp  "
                       f"(test: {_vtest_mape14:.2f} %)"),
                delta_color="inverse",
            )
        else:
            _cc14[1].metric("MAPE", f"{cone14_30['mape']:.2f} %")
        _cc14[2].metric("Resolved (30 d)", str(cone14_30["n"]),
                        help="Number of 14-day forecasts whose target date has passed")
    else:
        st.info("Insufficient data for 30-day 14d cone metrics.")

    # ── 5. 3-class day-type model ────────────────────────────────────────
    DT_ICON = {"BigUpper": "🟢", "BigLower": "🔴", "Quiet": "⚪"}
    st.markdown("#### 🏷️ 3-class day-type model (GBM: BigUpper / BigLower / Quiet)")
    if _dt_meta:
        _stored_acc = _dt_meta.get("test_accuracy_pct", 0)
        st.caption(
            f"Test period: **{_dt_meta.get('test_start','?')} → "
            f"{_dt_meta.get('test_end','?')}** · "
            f"n = {_dt_meta.get('test_n','?')} days · "
            f"stored accuracy: {_stored_acc:.1f} %"
            + (f"  ·  computed accuracy: {dt_at['accuracy']:.1f} %  (n={dt_at['n']})"
               if dt_at else "")
        )
    if dt30:
        _vtest_acc = dt_at["accuracy"] if dt_at else _dt_meta.get("test_accuracy_pct", 50.0)
        _em_acc = _badge(dt30["accuracy"], _vtest_acc, lo_better=False)
        _dtc = st.columns(4)
        _dtc[0].metric(
            f"{_em_acc} Overall accuracy",
            f"{dt30['accuracy']:.1f} %",
            delta=(f"{dt30['accuracy'] - _vtest_acc:+.1f} pp  "
                   f"(test: {_vtest_acc:.1f} %)"),
            delta_color="normal",
            help=f"n = {dt30['n']} · test baseline {_vtest_acc:.1f} %",
        )
        for _ci, _cls in enumerate(["BigUpper", "BigLower", "Quiet"]):
            _bc     = dt30["by_class"][_cls]
            _prec30 = _bc["correct_n"] / _bc["pred_n"] * 100 if _bc["pred_n"] else 0.0
            if dt_at:
                _rbc     = dt_at["by_class"][_cls]
                _prec_at = _rbc["correct_n"] / _rbc["pred_n"] * 100 if _rbc["pred_n"] else 0.0
                _em_p    = _badge(_prec30, _prec_at, lo_better=False)
                _dtc[_ci + 1].metric(
                    f"{_em_p} {DT_ICON[_cls]} {_cls}",
                    f"{_prec30:.0f} % prec",
                    delta=(f"{_prec30 - _prec_at:+.0f} pp  "
                           f"(test: {_prec_at:.0f} %)"),
                    delta_color="normal",
                    help=(f"30d: pred {_bc['pred_n']}× actual {_bc['actual_n']}×  "
                          f"correct {_bc['correct_n']}.  "
                          f"Test baseline precision {_prec_at:.0f} % "
                          f"(pred {_rbc['pred_n']}× actual {_rbc['actual_n']}×)."),
                )
            else:
                _recall30 = _bc["correct_n"] / _bc["actual_n"] * 100 if _bc["actual_n"] else 0.0
                _dtc[_ci + 1].metric(
                    f"{DT_ICON[_cls]} {_cls}",
                    f"{_prec30:.0f} % prec",
                    delta=f"pred {_bc['pred_n']}× · actual {_bc['actual_n']}×",
                    help=(f"Precision {_prec30:.0f} %  Recall {_recall30:.0f} %  "
                          f"(n = {_bc['pred_n']} predicted as {_cls})"),
                )
    else:
        st.info("Insufficient data for 30-day day-type metrics.")

    # ══════════════════════════════════════════════════════════════════════
    # Drift monitoring
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("🔍 Drift monitoring")
    st.caption(
        "**Performance drift** = 30-day metric vs test-set baseline (above).  "
        "**Prediction shift** = mean of recent predictions vs reference period.  "
        "**Input drift** = z-score of recent feature means vs test-period reference (hourly).  "
        "**Regime drift** = volatility regime distribution (7-day cone)."
    )

    # ── Status summary ────────────────────────────────────────────────────
    _drift_rows = []
    if hourly_30d and _mt_h:
        _pm_h  = _badge(hourly_30d["mape"],    _mt_h.get("MAPE_pct",    0), True)
        _da_h  = _badge(hourly_30d["dir_acc"], _mt_h.get("dir_acc_pct", 0), False)
        _ov_h  = "🔴" if "🔴" in (_pm_h, _da_h) else ("🟡" if "🟡" in (_pm_h, _da_h) else "🟢")
        _drift_rows.append({
            "Model":           "⏱️ Hourly close",
            "Perf drift":      (f"{_pm_h}  MAPE {hourly_30d['mape']:.2f}% "
                                f"vs {_mt_h.get('MAPE_pct',0):.2f}%"),
            "Direction drift": (f"{_da_h}  {hourly_30d['dir_acc']:.1f}% "
                                f"vs {_mt_h.get('dir_acc_pct',0):.1f}%"),
            "Overall":         _ov_h,
        })
    if hl30 and _hl_mtest:
        _pm_hl  = _badge(hl30["mape_h"], _hl_mtest.get("MAPE_H", 0), True)
        _da_hl  = _badge(hl30["dir_h"],  _hl_mtest.get("direction_hit_rate", 50), False)
        _ov_hl  = "🔴" if "🔴" in (_pm_hl, _da_hl) else ("🟡" if "🟡" in (_pm_hl, _da_hl) else "🟢")
        _drift_rows.append({
            "Model":           "📅 Daily H/L",
            "Perf drift":      (f"{_pm_hl}  MAPE-H {hl30['mape_h']:.2f}% "
                                f"vs {_hl_mtest.get('MAPE_H',0):.2f}%"),
            "Direction drift": (f"{_da_hl}  dir-H {hl30['dir_h']:.1f}% "
                                f"vs {_hl_mtest.get('direction_hit_rate',50):.1f}%"),
            "Overall":         _ov_hl,
        })
    if cone30:
        _vt_cov2 = cone_at["within_pct"] if cone_at else _cone_meta.get("held_out_band_coverage_pct", 88)
        _pm_c   = _badge(cone30["within_pct"], _vt_cov2, False)
        _drift_rows.append({
            "Model":           "📐 7-day cone",
            "Perf drift":      (f"{_pm_c}  coverage {cone30['within_pct']:.1f}% "
                                f"vs {_vt_cov2:.1f}%"),
            "Direction drift": "—",
            "Overall":         _pm_c,
        })
    if cone14_30:
        _vt_cov14_2 = cone14_at["within_pct"] if cone14_at else _cone14_meta.get("held_out_band_coverage_pct", 89)
        _pm_c14     = _badge(cone14_30["within_pct"], _vt_cov14_2, False)
        _drift_rows.append({
            "Model":           "📆 14-day cone",
            "Perf drift":      (f"{_pm_c14}  coverage {cone14_30['within_pct']:.1f}% "
                                f"vs {_vt_cov14_2:.1f}%"),
            "Direction drift": "—",
            "Overall":         _pm_c14,
        })
    if dt30:
        _vt_acc2 = dt_at["accuracy"] if dt_at else _dt_meta.get("test_accuracy_pct", 52.8)
        _pm_dt  = _badge(dt30["accuracy"], _vt_acc2, False)
        _drift_rows.append({
            "Model":           "🏷️ Day-type",
            "Perf drift":      (f"{_pm_dt}  accuracy {dt30['accuracy']:.1f}% "
                                f"vs {_vt_acc2:.1f}%"),
            "Direction drift": "—",
            "Overall":         _pm_dt,
        })
    if _drift_rows:
        st.dataframe(
            pd.DataFrame(_drift_rows),
            hide_index=True,
            use_container_width=True,
        )

    # ── Hourly: feature input drift ──────────────────────────────────────
    with st.expander("🌊 Input feature drift — hourly model (top-15 by importance)", expanded=False):
        _imp = A.get("importance")
        if _imp is not None and _ts_h:
            _ts_h_ts = pd.Timestamp(_ts_h)
            _ref_idx = F_filled.index[
                (F_filled.index >= _ts_h_ts) &
                (F_filled.index <  lb30_start) &
                valid_mask
            ]
            _rec_idx = F_filled.index[
                (F_filled.index >= lb30_start) &
                (F_filled.index <= latest_t) &
                valid_mask
            ]
            if len(_ref_idx) > 100 and len(_rec_idx) > 24:
                _top_f     = _imp.sort_values(ascending=False).head(15).index.tolist()
                _ref_means = F_filled.loc[_ref_idx, _top_f].mean()
                _ref_stds  = F_filled.loc[_ref_idx, _top_f].std()
                _rec_means = F_filled.loc[_rec_idx, _top_f].mean()
                _z_feats   = (_rec_means - _ref_means) / (_ref_stds + 1e-10)
                _z_sorted  = _z_feats.sort_values()
                _colors_f  = [
                    "#dc2626" if abs(z) > 2.0 else
                    "#f59e0b" if abs(z) > 1.0 else
                    "#16a34a"
                    for z in _z_sorted.values
                ]
                _fig_fd = go.Figure(go.Bar(
                    y=list(_z_sorted.index), x=list(_z_sorted.values),
                    orientation="h",
                    marker_color=_colors_f,
                    text=[f"{z:+.2f}" for z in _z_sorted.values],
                    textposition="outside",
                ))
                _fig_fd.add_vline(x= 0, line=dict(color="#111", width=1))
                _fig_fd.add_vline(x= 2, line=dict(color="#dc2626", width=1.2, dash="dash"))
                _fig_fd.add_vline(x=-2, line=dict(color="#dc2626", width=1.2, dash="dash"))
                _fig_fd.add_vline(x= 1, line=dict(color="#f59e0b", width=1,   dash="dot"))
                _fig_fd.add_vline(x=-1, line=dict(color="#f59e0b", width=1,   dash="dot"))
                _fig_fd.update_layout(
                    height=430, template="plotly_white",
                    title=dict(
                        text=("<b>Feature z-scores: last 30 days vs test-period reference</b>"
                              "<br><span style='font-size:11px;color:#555'>"
                              "z = (recent mean − ref mean) / ref std.  "
                              "🔴 |z|>2 = significant drift · 🟡 |z|>1 = mild · 🟢 stable."
                              "</span>"),
                        x=0.0, xanchor="left",
                    ),
                    xaxis_title="z-score (std deviations)",
                    yaxis_title=None,
                    margin=dict(t=80, r=90, b=40, l=170),
                )
                st.plotly_chart(_fig_fd, use_container_width=True,
                                key=f"drift_feat_{'live' if is_live else 'hist'}")
                _n_red = int((abs(_z_feats) > 2).sum())
                _n_yel = int(((abs(_z_feats) > 1) & (abs(_z_feats) <= 2)).sum())
                _n_grn = int((abs(_z_feats) <= 1).sum())
                st.caption(
                    f"Reference: {len(_ref_idx):,} hours from "
                    f"{_ts_h_ts.date()} to {lb30_start.date()}.  "
                    f"🔴 {_n_red} features significantly drifted (|z|>2) · "
                    f"🟡 {_n_yel} mildly drifted (1<|z|≤2) · "
                    f"🟢 {_n_grn} stable (|z|≤1)."
                )
            else:
                st.info("Insufficient reference-period data for feature drift analysis "
                        "(need test_start + at least 100 ref hours and 24 recent hours).")
        else:
            st.info("Feature importance not available in this artefact version.")

    # ── Hourly: prediction distribution shift ────────────────────────────
    with st.expander("📈 Prediction distribution shift — hourly model", expanded=False):
        if _y30_preds is not None and _ts_h:
            _ts_h_ts2 = pd.Timestamp(_ts_h)
            _ref_idx2 = F_filled.index[
                (F_filled.index >= _ts_h_ts2) &
                (F_filled.index <  lb30_start) &
                valid_mask
            ]
            if len(_ref_idx2) > 50:
                _y_ref2 = model.predict(F_filled.loc[_ref_idx2])
                _pc1, _pc2, _pc3 = st.columns(3)
                _mean_diff  = hourly_30d["y_mean"] - float(_y_ref2.mean() * 100)
                _std_diff   = hourly_30d["y_std"]  - float(_y_ref2.std()  * 100)
                _z_mean_ps  = float(
                    (_y30_preds.mean() - _y_ref2.mean())
                    / (_y_ref2.std() / np.sqrt(len(_y30_preds)) + 1e-12)
                )
                _z_em_ps = "🟢" if abs(_z_mean_ps) < 1.5 else ("🟡" if abs(_z_mean_ps) < 2.5 else "🔴")
                _pc1.metric(
                    "Pred mean log-ret (30d)",
                    f"{hourly_30d['y_mean']:+.4f} %",
                    delta=f"ref: {_y_ref2.mean()*100:+.4f} %  (Δ {_mean_diff:+.4f} pp)",
                    delta_color="off",
                    help="Systematic upward/downward bias in recent predictions vs test-period reference",
                )
                _pc2.metric(
                    "Pred std (30d)",
                    f"{hourly_30d['y_std']:.4f} %",
                    delta=f"ref: {_y_ref2.std()*100:.4f} %  (Δ {_std_diff:+.4f} pp)",
                    delta_color="off",
                    help="Spread of predicted log-returns — higher = model sees more uncertainty",
                )
                _pc3.metric(
                    f"{_z_em_ps} Mean-shift z-score",
                    f"{_z_mean_ps:.2f}",
                    help=(f"z = (recent mean − ref mean) / (ref std / √n).  "
                          f"|z| < 1.5 🟢 · 1.5–2.5 🟡 · >2.5 🔴.  "
                          f"Reference: {len(_ref_idx2):,} bars from "
                          f"{_ts_h_ts2.date()} → {lb30_start.date()}."),
                )
            else:
                st.info("Insufficient reference-period data for prediction shift analysis.")
        else:
            st.info("Prediction shift unavailable (no test_start in artefact).")

    # ── Daily H/L: prediction mean-offset shift ───────────────────────────
    with st.expander("📅 Daily H/L prediction shift", expanded=False):
        _mu_hi_ref = float(_hl_art.get("mu_hi", 0))
        _mu_lo_ref = float(_hl_art.get("mu_lo", 0))
        if hl30 and _mu_hi_ref and _mu_lo_ref:
            _ds30 = compute_daily_series(_end_iso, days_back=30)
            if not _ds30.empty and "close_asof" in _ds30.columns:
                _have_ds = _ds30["actual_high"].notna()
                _s30_hl  = _ds30[_have_ds]
                if not _s30_hl.empty:
                    _mu_hi_30 = ((_s30_hl["pred_high"] - _s30_hl["close_asof"])
                                 / _s30_hl["close_asof"]).mean()
                    _mu_lo_30 = ((_s30_hl["close_asof"] - _s30_hl["pred_low"])
                                 / _s30_hl["close_asof"]).mean()
                    _hps1, _hps2 = st.columns(2)
                    _hps1.metric(
                        "Mean pred upside offset (30d)",
                        f"{_mu_hi_30 * 100:+.2f} %",
                        delta=(f"vs training μ_hi {_mu_hi_ref * 100:+.2f} %  "
                               f"(Δ {(_mu_hi_30 - _mu_hi_ref)*100:+.2f} pp)"),
                        delta_color="off",
                        help=(f"Mean of (pred_high − close_asof) / close_asof over last 30 days "
                              f"vs climatological mean stored at training time."),
                    )
                    _hps2.metric(
                        "Mean pred downside offset (30d)",
                        f"{_mu_lo_30 * 100:+.2f} %",
                        delta=(f"vs training μ_lo {_mu_lo_ref * 100:+.2f} %  "
                               f"(Δ {(_mu_lo_30 - _mu_lo_ref)*100:+.2f} pp)"),
                        delta_color="off",
                    )
                else:
                    st.info("No completed daily bars in the last 30 days.")
            else:
                st.info("Daily series unavailable for prediction shift analysis.")
        else:
            st.info("Daily H/L prediction shift unavailable (mu_hi/mu_lo not in artefact).")

    # ── 7-day cone: regime distribution drift ────────────────────────────
    with st.expander("📐 7-day cone regime distribution drift", expanded=False):
        if cone_at and cone30:
            _rolling30_r = compute_rolling_7d_series(_end_iso, days_back=30)
            if _rolling30_r is not None and not _rolling30_r.empty:
                _rl = {0: "low vol", 1: "mid vol", 2: "high vol"}
                _reg30_pct = _rolling30_r["regime"].value_counts(normalize=True).mul(100)
                _reg_at_pct = cone_at["regime_pct"]
                _rfig = go.Figure()
                for _r, _rlbl in _rl.items():
                    _rfig.add_trace(go.Bar(
                        name=_rlbl,
                        x=["Last 30 days", "Full test period"],
                        y=[float(_reg30_pct.get(_r, 0)),
                           float(_reg_at_pct.get(_r, 0))],
                    ))
                _rfig.update_layout(
                    height=290, template="plotly_white", barmode="stack",
                    title="Regime distribution: last 30 days vs full test period",
                    yaxis_title="% of anchor days", yaxis_range=[0, 100],
                    margin=dict(t=50, r=30, b=40, l=50),
                    legend=dict(orientation="h", x=0, y=1.15),
                )
                st.plotly_chart(_rfig, use_container_width=True,
                                key=f"drift_regime_{'live' if is_live else 'hist'}")
                st.caption(
                    "Regime = range_ma30 tercile (low/mid/high volatility).  "
                    "A shift toward high-vol anchors means the model's median "
                    "return estimate will be less reliable (higher-variance regime)."
                )
            else:
                st.info("Insufficient data for regime drift analysis.")
        else:
            st.info("Regime drift requires full-test cone metrics (insufficient historical data).")

    # ─────────────────────── extra context: features now ──────────────────
    with st.expander("🔍 Latest feature snapshot (top contributors)"):
        imp = A.get("importance", None)
        if imp is not None:
            topf = imp.sort_values(ascending=False).head(15).index.tolist()
            snap = F.loc[latest_t, topf].rename("value").to_frame()
            snap["importance"] = imp.loc[topf].round(5)
            st.dataframe(snap.round(5), use_container_width=True)

    st.markdown("---")
    # Surface data freshness so user can verify the chart is current
    lag_min = (now_utc - latest_t).total_seconds() / 60
    st.caption(
        f"Rolling forecast target: **now + 1 h** → "
        f"**{forecast_target.strftime('%Y-%m-%d %H:%M:%S UTC')}** (slides each refresh).  "
        f"Hourly Yahoo bar latest = {latest_t.strftime('%H:%M UTC')} ({lag_min:.0f} min behind).  "
        f"Live Binance spot = "
        + (f"{live_spot_ts.strftime('%H:%M:%S UTC')}"
           if live_spot_ts else "_unavailable_") + ".  "
        f"Page auto-refresh every **{REFRESH_SECONDS}s**.  "
        f"Last run: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}."
    )



# ════════════════════════════════════════════════════════════════════════
# Tabs: Live | Historical
# ════════════════════════════════════════════════════════════════════════
tab_live, tab_hist = st.tabs(["🔴 Live (rolling now+1h)", "🕒 Historical replay"])

with tab_live:
    render_dashboard(latest_t_global, is_live=True,
                     live_spot=live_spot, live_spot_ts=live_spot_ts)

with tab_hist:
    valid_times = F_filled.index[valid_mask]
    if len(valid_times) < 30:
        st.error("Not enough historical data available yet.")
    else:
        CT_TZ = "America/Chicago"
        min_t = valid_times.min(); max_t = valid_times.max()
        min_t_ct = min_t.tz_localize("UTC").tz_convert(CT_TZ).tz_localize(None)
        max_t_ct = max_t.tz_localize("UTC").tz_convert(CT_TZ).tz_localize(None)
        min_date = min_t_ct.date()
        data_max_date = (max_t_ct - pd.Timedelta(hours=1)).date()
        # Historical replay is for PAST dates only — today's bar is in progress
        # (it hasn't closed at 7am CT next day yet) so we exclude it.
        today_ct = pd.Timestamp.now(tz=CT_TZ).date()
        max_date = min(data_max_date, today_ct - timedelta(days=1))
        if "hist_date" not in st.session_state:
            st.session_state["hist_date"] = max_date

        # Compute slider bounds from the current picked_date (DST-correct).
        # Summer: 7am→7am CDT. Winter: 6am→6am CST. Matches the daily model's
        # 24h bar anchored at 12:00 UTC.
        picked_date = st.session_state["hist_date"]
        import datetime as _dt
        slider_win_start_utc = pd.Timestamp(picked_date) + pd.Timedelta(hours=ANCHOR_HOUR_UTC)
        slider_win_end_utc   = slider_win_start_utc + pd.Timedelta(days=1)
        slider_min = (slider_win_start_utc.tz_localize("UTC")
                      .tz_convert(CT_TZ).tz_localize(None).to_pydatetime())
        slider_max = (slider_win_end_utc.tz_localize("UTC")
                      .tz_convert(CT_TZ).tz_localize(None).to_pydatetime())
        # Sanitize stored hour value if the date changed and it's now out of range.
        prior = st.session_state.get("hist_hour_ts")
        if prior is None or not (slider_min <= prior <= slider_max):
            st.session_state["hist_hour_ts"] = slider_max

        # Pre-compute target_t / actual_t from session state so render_dashboard
        # can render KPIs FIRST, then the picker, then the plots. (The picker's
        # widgets read/write the same session_state keys; on user interaction
        # Streamlit reruns and these recompute consistently for the next pass.)
        picked_t_ct = pd.Timestamp(st.session_state["hist_hour_ts"])
        target_t = (picked_t_ct
                    .tz_localize(CT_TZ, ambiguous=True, nonexistent="shift_forward")
                    .tz_convert("UTC").tz_localize(None))
        avail = valid_times[valid_times <= target_t]
        actual_t = avail[-1] if len(avail) else None
        actual_t_ct = (actual_t.tz_localize("UTC").tz_convert(CT_TZ).tz_localize(None)
                       if actual_t is not None else None)

        # Callbacks (closures over min_date/max_date)
        def _shift_date(delta_days):
            cur = st.session_state.get("hist_date", max_date)
            new = cur + timedelta(days=delta_days)
            st.session_state["hist_date"] = max(min_date, min(max_date, new))

        def _select_date(d):
            st.session_state["hist_date"] = max(min_date, min(max_date, d))

        def _go_to_bookmark(iso_date_str):
            try:
                d = _date.fromisoformat(iso_date_str)
                st.session_state["hist_date"] = max(min_date, min(max_date, d))
            except Exception:
                pass

        bookmarks = load_bookmarks()

        def _hist_picker():
            """Renders date strip, calendar, hour slider, and bookmarks expander.
            Called from inside render_dashboard so it sits right above the plots."""
            st.markdown("#### 🕒 Pick a historical date / hour to replay")

            # ── Day strip: ◀  [-3] [-2] [-1] [SEL] [+1] [+2] [+3]  ▶ ──
            cur_picked = st.session_state.get("hist_date", max_date)
            strip_cols = st.columns([0.4, 1, 1, 1, 1, 1, 1, 1, 0.4])
            with strip_cols[0]:
                st.button("◀", key="hist_prev_day", help="Previous day",
                          on_click=_shift_date, args=(-1,),
                          disabled=(cur_picked <= min_date),
                          use_container_width=True)
            for i, offset in enumerate(range(-3, 4)):
                d = cur_picked + timedelta(days=offset)
                label = d.strftime("%a\n%b %-d")
                in_range = (min_date <= d <= max_date)
                is_selected = (d == cur_picked)
                with strip_cols[i + 1]:
                    st.button(
                        label, key=f"hist_pill_{offset}",
                        help=d.strftime("%Y-%m-%d (US Central)"),
                        on_click=_select_date, args=(d,),
                        disabled=(not in_range),
                        type=("primary" if is_selected else "secondary"),
                        use_container_width=True,
                    )
            with strip_cols[8]:
                st.button("▶", key="hist_next_day", help="Next day",
                          on_click=_shift_date, args=(1,),
                          disabled=(cur_picked >= max_date),
                          use_container_width=True)

            # ── Calendar + hour slider side-by-side ──
            cal_col, slider_col = st.columns([1, 2])
            with cal_col:
                st.date_input(
                    "Or pick from calendar (CT)",
                    min_value=min_date, max_value=max_date,
                    key="hist_date",
                )
            with slider_col:
                st.slider(
                    "Hour (US Central) — 7am picked → 7am next day",
                    min_value=slider_min, max_value=slider_max,
                    step=_dt.timedelta(hours=1),
                    format="MMM D, HH:mm",
                    key="hist_hour_ts",
                )

            # As-of caption (snap if needed)
            if actual_t is not None:
                if actual_t != target_t:
                    st.caption(f"⚠️ Snapped to **{actual_t_ct} CT** ({actual_t} UTC) "
                               f"— picked {picked_t_ct} CT")
                else:
                    st.caption(f"As-of: **{actual_t_ct} CT** ({actual_t} UTC)")

            # ── Bookmarks (collapsed by default since panel is mid-page) ──
            with st.expander(
                f"🔖 Bookmarks  ({sum(len(v) for v in bookmarks.values())} saved "
                f"across {len(bookmarks)} categor{'y' if len(bookmarks)==1 else 'ies'})",
                expanded=False,
            ):
                bk_browse, bk_save = st.columns([1.2, 1])
                with bk_browse:
                    st.markdown("**Jump to a bookmarked date**")
                    if not bookmarks:
                        st.caption("_No bookmarks yet. Save the current date on the right._")
                    else:
                        cat = st.selectbox("Category", sorted(bookmarks.keys()),
                                           key="bk_cat_pick")
                        entries = bookmarks.get(cat, [])
                        if entries:
                            opt_labels = [
                                f"{e['date']}" + (f" — {e['label']}" if e.get("label") else "")
                                for e in entries
                            ]
                            idx = st.selectbox(
                                "Date", range(len(entries)),
                                format_func=lambda i: opt_labels[i],
                                key="bk_date_pick",
                            )
                            sel = entries[idx]
                            b1, b2 = st.columns([1, 1])
                            with b1:
                                st.button("Go to this date", key="bk_go",
                                          on_click=_go_to_bookmark, args=(sel["date"],),
                                          type="primary", use_container_width=True)
                            with b2:
                                st.button("🗑 Delete", key="bk_del",
                                          on_click=lambda c=cat, d=sel["date"]: delete_bookmark(c, d),
                                          use_container_width=True)
                with bk_save:
                    st.markdown("**Save current selection**")
                    cur_d = st.session_state.get("hist_date", max_date)
                    st.caption(f"Current date: **{cur_d.isoformat()}** (CT)")
                    cats = sorted(bookmarks.keys())
                    cat_choice = st.selectbox(
                        "Category", options=cats + ["➕ New category…"],
                        key="bk_save_cat",
                    )
                    if cat_choice == "➕ New category…":
                        new_cat = st.text_input("New category name", key="bk_new_cat",
                                                placeholder="e.g. Macro events")
                        final_cat = (new_cat or "").strip()
                    else:
                        final_cat = cat_choice
                    bk_label = st.text_input("Optional label", key="bk_label",
                                             placeholder="e.g. FOMC meeting")
                    if st.button("💾 Save bookmark", key="bk_save_btn",
                                 disabled=not final_cat, use_container_width=True):
                        add_bookmark(final_cat, cur_d, bk_label.strip())
                        st.success(f"Saved **{cur_d.isoformat()}** to *{final_cat}*")
                        st.rerun()

        if actual_t is None:
            st.error(f"No data available at or before {picked_t_ct} CT "
                     f"(= {target_t} UTC). Pick a different date/hour below.")
            _hist_picker()
        else:
            render_dashboard(actual_t, is_live=False,
                             live_spot=None, live_spot_ts=None,
                             hist_picker=_hist_picker)
# ─────────────────────── timer-driven re-run ──────────────────────────
time.sleep(REFRESH_SECONDS)
st.rerun()
