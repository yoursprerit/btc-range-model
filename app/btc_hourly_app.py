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
    Also returns "daily H/L test_start" for the OOS window boundary.
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
            out["daily H/L test_start"] = (pd.Timestamp(meta.get("test_start"))
                                           if meta.get("test_start") else None)
        except Exception:
            out["daily H/L"] = None
            out["daily H/L test_start"] = None
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
def compute_daily_forecast(target_date_iso, data_end=None):
    """Apply the 12:00-UTC (7am-CT) daily model as it was trained.

    `target_date_iso` = ISO date of the bar BEING PREDICTED. That bar covers
    [target 12:00 UTC, target+1 12:00 UTC) = [target 7am CT, target+1 7am CT).
    Only data completed before the target bar starts is used — i.e. bars with
    start date ≤ target − 1 day (the latest such bar closes at target 7am CT,
    which is the target bar's open).

    `data_end` = ISO date of the latest bar in _fetch_daily_raw() at call time.
    It is NOT used inside the function body — it exists solely to bust the cache
    when _fetch_daily_raw() gains a new bar.  Without this, a prediction made
    while the as-of bar was missing from the 6-hour _fetch_daily_raw() cache
    would remain stale for the full 24-hour TTL even after the raw data refreshed,
    causing today's and yesterday's predictions to share the same as-of bar and
    produce identical (flat) output."""
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


@st.cache_data(ttl=3600 * 6, show_spinner=False)
def compute_daily_series(end_target_date_iso, days_back=7):
    """Build a series of (pred_high, pred_low, actual_high, actual_low) for the
    last `days_back`+1 target days ending at `end_target_date`. Each prediction
    is generated by `compute_daily_forecast(target_date)`, which internally
    uses data through target − 1 day (i.e. through target 7am CT).

    Cache TTL: 6 hours — matches _fetch_daily_raw() so actual H/L values
    (which come from that layer) are never staler than the raw data itself.
    A new completed bar (closes at 12:00 UTC daily) becomes visible within
    6 hours of bar close, matching how quickly _fetch_daily_raw picks it up."""
    end_target = pd.Timestamp(end_target_date_iso)
    daily_df = _fetch_daily_raw()
    _data_end = daily_df.index.max().strftime("%Y-%m-%d") if not daily_df.empty else None
    rows = []
    for i in range(days_back, -1, -1):
        target_date = end_target - pd.Timedelta(days=i)
        pred = compute_daily_forecast(target_date.strftime("%Y-%m-%d"), data_end=_data_end)
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
    dir_acc = float(np.mean(
        np.sign(resolved["pred_close"] - resolved["anchor_close"])
        == np.sign(resolved["actual_close"] - resolved["anchor_close"])
    ) * 100)
    return dict(n=len(resolved), mape=float(mape),
                within_pct=float(within), band_pct=band_pct, dir_acc=dir_acc)


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
    test_start_str = meta.get("test_start", "2026-03-01")
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
    dir_acc = float(np.mean(
        np.sign(resolved["pred_close"] - resolved["anchor_close"])
        == np.sign(resolved["actual_close"] - resolved["anchor_close"])
    ) * 100)
    regime_pct = resolved["regime"].value_counts(normalize=True).mul(100).to_dict()
    return dict(
        n=len(resolved), mape=float(mape), within_pct=float(within),
        dir_acc=dir_acc, band_pct=band_pct, test_start=test_start_str,
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
def compute_day_type_forecast(target_date_iso, data_end=None):
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

    `data_end` busts the cache when _fetch_daily_raw() gains a new bar.
    See compute_daily_forecast() for details on why this matters.
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
    daily_forecast = compute_daily_forecast(target_date_iso, data_end=data_end)
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
    dir_acc = float(np.mean(
        np.sign(resolved["pred_close"] - resolved["anchor_close"])
        == np.sign(resolved["actual_close"] - resolved["anchor_close"])
    ) * 100)
    return dict(n=len(resolved), mape=float(mape),
                within_pct=float(within), band_pct=band_pct, dir_acc=dir_acc)


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
    _data_end = daily.index.max().strftime("%Y-%m-%d") if not daily.empty else None
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
        pred_r = compute_day_type_forecast(target_iso, data_end=_data_end)
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
    dir_acc = float(np.mean(
        np.sign(resolved["pred_close"] - resolved["anchor_close"])
        == np.sign(resolved["actual_close"] - resolved["anchor_close"])
    ) * 100)
    regime_pct = resolved["regime"].value_counts(normalize=True).mul(100).to_dict()
    return dict(
        n=len(resolved), mape=float(mape), within_pct=float(within),
        dir_acc=dir_acc, band_pct=band_pct, test_start=test_start_str,
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
    _data_end = daily.index.max().strftime("%Y-%m-%d") if not daily.empty else None
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
                    pred_r = compute_day_type_forecast(cur.strftime("%Y-%m-%d"),
                                                       data_end=_data_end)
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
# Trend Signature Alert System
# Mines last N completed daily bars for early-warning signatures of
# significant uptrends / downtrends, based on prediction-vs-actual patterns.
# ════════════════════════════════════════════════════════════════════════

def _fetch_current_bar_intraday():
    """Return a dict describing the **current in-progress** 12 UTC daily bar.

    Not cached independently — relies on _fetch_binance_hourly()'s 10-minute
    cache for freshness. This means intraday H/L updates every ~10 minutes,
    matching the app's live auto-refresh cycle, without being frozen by the
    6-hour TTL of compute_trend_signatures().

    Returns None if the current bar has no hourly data yet.
    """
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    # Bar boundary: today 12:00 UTC if hour >= 12, else yesterday 12:00 UTC
    if now_utc.hour >= ANCHOR_HOUR_UTC:
        bar_start = now_utc.replace(hour=ANCHOR_HOUR_UTC, minute=0, second=0, microsecond=0)
    else:
        bar_start = (now_utc - timedelta(days=1)).replace(
            hour=ANCHOR_HOUR_UTC, minute=0, second=0, microsecond=0)
    bar_date_iso = bar_start.date().isoformat()

    # Fetch hourly data for the last ~2 days (covers all hours since bar_start)
    try:
        hourly = _fetch_binance_hourly(days_back=3)
    except Exception:
        return None
    if hourly is None or hourly.empty:
        return None

    # Filter to candles within the current bar: bar_start ≤ ts < bar_start + 24h
    bar_end = bar_start + timedelta(hours=24)
    # The hourly index is the candle open time
    mask = (hourly.index >= bar_start) & (hourly.index < bar_end)
    bar_data = hourly.loc[mask]
    if bar_data.empty:
        return None

    hours_elapsed = len(bar_data)
    running_high  = float(bar_data["high"].max())
    running_low   = float(bar_data["low"].min())
    current_close = float(bar_data["close"].iloc[-1])

    return dict(
        bar_date_iso   = bar_date_iso,
        bar_start      = bar_start,
        running_high   = running_high,
        running_low    = running_low,
        current_close  = current_close,
        hours_elapsed  = hours_elapsed,
        pct_through    = hours_elapsed / 24.0,
    )


def _compute_intraday_signal(intraday: dict, daily_fc: dict) -> dict:
    """Compare the current bar's running H/L against today's frozen daily forecast.

    Returns a dict describing whether the bar is already breaking the
    predicted high/low bands — a real-time early warning that doesn't
    require waiting for bar close.

    Args:
        intraday: output of _fetch_current_bar_intraday()
        daily_fc: output of compute_daily_forecast() for today's bar date

    Returns:
        dict with intraday signal values and trigger flags, or None.
    """
    if intraday is None or daily_fc is None:
        return None

    pred_hi   = float(daily_fc["pred_high"])
    pred_lo   = float(daily_fc["pred_low"])
    close_ref = float(daily_fc["close_asof"])   # previous bar's close (model's reference)

    run_hi   = float(intraday["running_high"])
    run_lo   = float(intraday["running_low"])
    cur_close= float(intraday["current_close"])
    hrs      = int(intraday["hours_elapsed"])
    pct      = float(intraday["pct_through"])

    # Error vs daily prediction (same sign convention as bar-close signals)
    # Positive err_hi → intraday high already exceeded predicted high (bullish)
    # Positive err_lo → intraday low already undershot predicted low (bearish)
    err_hi_intra = (run_hi  - pred_hi) / close_ref * 100.0
    err_lo_intra = (pred_lo - run_lo)  / close_ref * 100.0

    hi_break_intra = run_hi  > pred_hi
    lo_break_intra = run_lo  < pred_lo

    # How far in % terms has price moved from the reference close
    move_pct = (cur_close - close_ref) / close_ref * 100.0

    # Severity buckets for the lo break (bearish intraday pressure)
    # We scale expectations by how many hours have elapsed:
    # - early in the bar (< 8h): even a small break is notable
    # - late in the bar (> 18h): already confirming what will show in bar close
    if lo_break_intra:
        if err_lo_intra > 2.0:
            lo_severity = "HIGH"
        elif err_lo_intra > 0.8:
            lo_severity = "MODERATE"
        else:
            lo_severity = "WATCH"
    else:
        lo_severity = "NONE"

    if hi_break_intra:
        if err_hi_intra > 2.0:
            hi_severity = "HIGH"
        elif err_hi_intra > 0.8:
            hi_severity = "MODERATE"
        else:
            hi_severity = "WATCH"
    else:
        hi_severity = "NONE"

    # Composite intraday alert
    if lo_break_intra and lo_severity == "HIGH":
        intra_alert = "INTRA_BEARISH_STRONG"
    elif lo_break_intra and lo_severity == "MODERATE":
        intra_alert = "INTRA_BEARISH"
    elif hi_break_intra and hi_severity == "HIGH":
        intra_alert = "INTRA_BULLISH_STRONG"
    elif hi_break_intra and hi_severity == "MODERATE":
        intra_alert = "INTRA_BULLISH"
    elif lo_break_intra or hi_break_intra:
        intra_alert = "INTRA_WATCH"
    else:
        intra_alert = "INTRA_NEUTRAL"

    return dict(
        bar_date_iso    = intraday["bar_date_iso"],
        hours_elapsed   = hrs,
        pct_through     = pct,
        pred_hi         = pred_hi,
        pred_lo         = pred_lo,
        close_ref       = close_ref,
        running_high    = run_hi,
        running_low     = run_lo,
        current_close   = cur_close,
        move_pct        = move_pct,
        err_hi_intra    = err_hi_intra,
        err_lo_intra    = err_lo_intra,
        hi_break_intra  = hi_break_intra,
        lo_break_intra  = lo_break_intra,
        hi_severity     = hi_severity,
        lo_severity     = lo_severity,
        intra_alert     = intra_alert,
    )


@st.cache_data(ttl=3600 * 6, show_spinner=False)
def compute_trend_signatures(target_date_iso: str):
    """Compute trend signature signals for the dashboard alert card.

    Cache TTL: 6 hours — matches _fetch_daily_raw() so signals refresh as
    soon as the underlying bar data refreshes from Yahoo/Binance.  The cache
    key includes target_date_iso, which rolls over automatically at 12:00 UTC
    (the daily bar boundary), so a new bar's data is picked up within 6 hours
    of bar close rather than being held for up to 24 hours.

    Uses the last 10 completed daily bars (bars with realized actual H/L)
    to compute four prediction-vs-actual divergence signals:

    DOWNTREND signals (statistically significant, p < 0.001):
      D1  lo_breaks_3d  — actual low broke predicted low ≥ 2 of last 3 days
      D2  err_hi_ma3    — 3d avg (actual_high − pred_high)/close < −1%
      D3  exhaustion    — first lo_break after streak of ≥ 3 hi_breaks (reversal canary)

    UPTREND signals (p = 0.022):
      U1  err_hi_ma3    — 3d avg (actual_high − pred_high)/close > +0.5%

    V-Reversal special signal:
      V   capitulation  — downtrend score ≥ 0.9 followed by lo_err > 5% today

    Returns a dict with all computed values and trigger flags.
    Returns None if fewer than 5 completed bars are available.
    """
    # Pull last 45 completed daily bars — 30 needed for MA30, 10 for clean_10d,
    # plus buffer; core rolling metrics still use only last 3/5 bars.
    series = compute_daily_series(target_date_iso, days_back=45)
    if series is None or series.empty:
        return None
    completed = series[series["actual_high"].notna() & series["actual_low"].notna()].copy()
    if len(completed) < 3:
        return None

    c   = completed["close_asof"].values.astype(float)
    phi = completed["pred_high"].values.astype(float)
    plo = completed["pred_low"].values.astype(float)
    ah  = completed["actual_high"].values.astype(float)
    al  = completed["actual_low"].values.astype(float)
    n   = len(completed)

    # ── Raw single-bar signals ──────────────────────────────────────────
    # Positive = actual HIGH exceeded prediction (bullish pressure)
    err_hi = (ah - phi) / c * 100
    # Positive = actual LOW undershot prediction (bearish pressure)
    err_lo = (plo - al) / c * 100
    # Band-break flags
    hi_break = (ah > phi).astype(int)
    lo_break = (al < plo).astype(int)

    # ── 3-day rolling averages (last 3 bars) ───────────────────────────
    window3 = min(3, n)
    err_hi_ma3 = float(np.mean(err_hi[-window3:]))
    err_lo_ma3 = float(np.mean(err_lo[-window3:]))
    hi_breaks_3d = int(np.sum(hi_break[-window3:]))
    lo_breaks_3d = int(np.sum(lo_break[-window3:]))

    # ── 5-day rolling ──────────────────────────────────────────────────
    window5 = min(5, n)
    hi_breaks_5d = int(np.sum(hi_break[-window5:]))
    lo_breaks_5d = int(np.sum(lo_break[-window5:]))

    # ── Consecutive return streak (using close_asof as daily close proxy) ─
    closes = completed["close_asof"].values.astype(float)
    streak = 0
    if len(closes) >= 2:
        for k in range(len(closes) - 1, 0, -1):
            d = np.sign(closes[k] - closes[k-1])
            if k == len(closes) - 1:
                streak = int(d)
            elif int(d) == np.sign(streak) and streak != 0:
                streak += int(d)
            else:
                break

    # ── D3 Exhaustion canary: first lo_break after hi_break streak ─────
    # Look for ≥ 3 consecutive hi_break days ending at some point,
    # followed by the first lo_break.
    exhaustion_active = False
    if n >= 4:
        consec_hi = 0
        for k in range(n - 2, -1, -1):
            if hi_break[k]:
                consec_hi += 1
            else:
                break
        if consec_hi >= 3 and lo_break[-1]:
            exhaustion_active = True

    # ── V-reversal capitulation: lo_err spike today ────────────────────
    # Last bar's single-day lo error > 5% (massive undershoot = capitulation)
    last_lo_err = float(err_lo[-1])
    last_hi_err = float(err_hi[-1])
    # Downtrend composite score — first term normalised by rolling 30-bar mean
    # (same window as backtest) so the 0.8 threshold has consistent meaning
    # regardless of total history available (live function loads up to 45 bars).
    _dn_norm = float(np.mean(err_hi[-min(30, n):]))
    dn_score_raw = (
        (-float(np.mean(err_hi[-window3:])) / max(abs(_dn_norm), 0.01)) * 0.30 +
        float(lo_breaks_3d) / 3 * 0.30 +
        float(err_lo_ma3)   / max(abs(err_lo_ma3), 0.1) * 0.20 +
        float(lo_break[-1]) * 0.20
    )
    capitulation_signal = (dn_score_raw > 0.7 and last_lo_err > 5.0)
    v_reversal_likely   = (dn_score_raw > 0.8 and last_lo_err > 5.0)

    # ── MA30 / Trend Filter (U1+MA30 strategy entry) ────────────────────
    # Re-compute D1/D2 signal for every historical bar so we can check
    # whether any fired in the 10 bars preceding the current bar (clean_10d).
    _d1_hist = np.zeros(n, dtype=bool)
    _d2_hist = np.zeros(n, dtype=bool)
    for _i in range(n):
        _s = max(0, _i - 2)
        _ehma_i = float(np.mean(err_hi[_s: _i + 1]))
        _elma_i = float(np.mean(err_lo[_s: _i + 1]))
        _hb3_i  = int(np.sum(hi_break[_s: _i + 1]))
        _lb3_i  = int(np.sum(lo_break[_s: _i + 1]))
        _d1_hist[_i] = (_lb3_i >= 2) and (_elma_i > 0.5)
        _d2_hist[_i] = (_ehma_i < -1.0)
    # 30-bar rolling mean of close prices (close_asof = the daily close proxy)
    ma30_window = min(30, n)
    ma30_value  = float(np.mean(c[-ma30_window:]))
    current_close_sig = float(c[-1])
    above_ma30 = current_close_sig > ma30_value
    # MA30 slope: compare current MA30 vs MA30 computed 5 bars ago
    # Bull regime = price above MA30 AND MA30 is rising (slope > 0 over last 5 bars)
    if n >= 35:
        ma30_5d_ago = float(np.mean(c[-35:-5]))   # 30-bar mean ending 5 bars ago
    else:
        ma30_5d_ago = ma30_value                   # insufficient data → assume flat
    ma30_slope_pos = ma30_value > ma30_5d_ago
    bull_regime = above_ma30 and ma30_slope_pos
    # clean_10d: zero D1 or D2 fires in the 10 bars *before* the current bar
    if n >= 11:
        clean_10d = not bool(np.any(_d1_hist[-11:-1] | _d2_hist[-11:-1]))
    elif n >= 2:
        clean_10d = not bool(np.any(_d1_hist[:-1] | _d2_hist[:-1]))
    else:
        clean_10d = False

    # ── Signature trigger flags ─────────────────────────────────────────
    d1_triggered  = (lo_breaks_3d >= 2) and (err_lo_ma3 > 0.5)
    d2_triggered  = (err_hi_ma3 < -1.0)
    d3_triggered  = exhaustion_active
    u1_triggered  = (err_hi_ma3 > 0.5) and (hi_breaks_3d >= 2)
    # TF1/TF2 entry signal (same for both): U1 confirmed by trend context.
    # TF2 adds regime-adaptive exits — BULL regime exits D3 only (patient),
    # BEAR/NEUTRAL exits D2 or D3 (defensive). Entry is identical.
    # V-gate: V-reversal (dn_score>0.8 & err_lo>5%) within last 3 bars also satisfies trend gate.
    v_gate_ok     = v_reversal_likely or capitulation_signal
    tf1_triggered = u1_triggered and (above_ma30 or clean_10d or v_gate_ok)

    # ── Composite alert level ───────────────────────────────────────────
    dn_count = int(d1_triggered) + int(d2_triggered) + int(d3_triggered)
    up_count = int(u1_triggered)

    if dn_count >= 3 or (dn_count >= 2 and v_reversal_likely):
        alert_level = "HIGH_DN"
    elif dn_count == 2:
        alert_level = "ELEVATED_DN"
    elif dn_count == 1 and not u1_triggered:
        alert_level = "WATCH_DN"
    elif tf1_triggered:
        alert_level = "STRATEGY_BUY"
    elif up_count >= 1 and dn_count == 0:
        alert_level = "WATCH_UP"
    else:
        alert_level = "NEUTRAL"

    # ── Per-bar detail (last 5 bars for sparkline display) ────────────
    detail_rows = []
    for i in range(max(0, n - 5), n):
        detail_rows.append(dict(
            date        = completed["target_date"].iloc[i],
            close       = float(closes[i]),
            pred_hi     = float(phi[i]),
            pred_lo     = float(plo[i]),
            actual_hi   = float(ah[i]),
            actual_lo   = float(al[i]),
            err_hi_pct  = float(err_hi[i]),
            err_lo_pct  = float(err_lo[i]),
            hi_break    = bool(hi_break[i]),
            lo_break    = bool(lo_break[i]),
        ))

    return dict(
        # Signal values
        err_hi_ma3        = err_hi_ma3,
        err_lo_ma3        = err_lo_ma3,
        hi_breaks_3d      = hi_breaks_3d,
        lo_breaks_3d      = lo_breaks_3d,
        hi_breaks_5d      = hi_breaks_5d,
        lo_breaks_5d      = lo_breaks_5d,
        streak            = streak,
        last_lo_err       = last_lo_err,
        last_hi_err       = last_hi_err,
        dn_score_raw      = dn_score_raw,
        # Trend-filter values (TF1/TF2 strategy entry + regime detection)
        ma30_value        = ma30_value,
        ma30_5d_ago       = ma30_5d_ago,
        current_close_sig = current_close_sig,
        above_ma30        = above_ma30,
        ma30_slope_pos    = ma30_slope_pos,
        bull_regime       = bull_regime,
        clean_10d         = clean_10d,
        # Trigger flags
        d1_triggered      = d1_triggered,
        d2_triggered      = d2_triggered,
        d3_triggered      = d3_triggered,
        u1_triggered      = u1_triggered,
        tf1_triggered     = tf1_triggered,
        capitulation_signal = capitulation_signal,
        v_reversal_likely   = v_reversal_likely,
        exhaustion_active   = exhaustion_active,
        # Composite alert
        alert_level  = alert_level,
        dn_count     = dn_count,
        up_count     = up_count,
        # Bar detail for mini-table
        detail_rows  = detail_rows,
        n_bars       = n,
        as_of_date   = completed["target_date"].iloc[-1],
    )


# ════════════════════════════════════════════════════════════════════════
# TF1 Trading Strategy — Batch Backtest Engine
# ════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600 * 6, show_spinner=False)
def _build_ct_batch_predictions():
    """Build CT daily H/L predictions for ALL available bars in one pass.

    Mirrors compute_daily_forecast() but builds the feature matrix ONCE on the
    full _fetch_daily_raw() dataset and runs vectorised batch prediction instead
    of rebuilding features for every individual target date.

    Returns a DataFrame indexed by TARGET date (the bar being predicted) with
    columns: close_asof, pred_high, pred_low.
    Returns None if the CT model artefact is missing.
    Cache TTL: 6 h — same as _fetch_daily_raw().
    """
    path = str(DAILY_MODEL_CT)
    if not os.path.exists(path):
        return None
    try:
        AD = joblib.load(path)
    except Exception:
        return None

    df = _fetch_daily_raw().copy()
    if df.empty:
        return None
    df = df.sort_index().ffill(limit=5)

    # ── Feature engineering (mirrors compute_daily_forecast exactly) ───
    f   = pd.DataFrame(index=df.index)
    c   = df["btc_close"]
    h   = df["btc_high"]
    l_  = df["btc_low"]
    v   = df["btc_volume"]
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
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    f["rsi_14"] = 100 - 100/(1+rs)
    e12 = c.ewm(span=12,adjust=False).mean()
    e26 = c.ewm(span=26,adjust=False).mean()
    macd = e12 - e26
    f["macd"]      = macd/c
    f["macd_sig"]  = macd.ewm(span=9,adjust=False).mean()/c
    f["macd_hist"] = (macd - macd.ewm(span=9,adjust=False).mean())/c
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
        s = df[col].astype(float); s_log = np.log(s.replace(0,np.nan))
        f[f"{col}_d1"]  = s_log.diff(1)
        f[f"{col}_d7"]  = s_log.diff(7)
        f[f"{col}_z30"] = (s_log - s_log.rolling(30).mean())/s_log.rolling(30).std()
    nh, nl = h.shift(-1), l_.shift(-1)
    y_hi = (nh-c)/c; y_lo = (c-nl)/c
    f["y_hi_ema3"] = y_hi.shift(1).ewm(span=3, adjust=False).mean()
    f["y_lo_ema3"] = y_lo.shift(1).ewm(span=3, adjust=False).mean()
    f["y_hi_ema7"] = y_hi.shift(1).ewm(span=7, adjust=False).mean()
    f["y_lo_ema7"] = y_lo.shift(1).ewm(span=7, adjust=False).mean()
    prev_3_hi = h.shift(1).rolling(3).max()
    prev_3_lo = l_.shift(1).rolling(3).min()
    f["above_3d_high"]  = (c > prev_3_hi).astype(float)
    f["below_3d_low"]   = (c < prev_3_lo).astype(float)
    f["bo_strength_up"] = (c / prev_3_hi - 1).clip(lower=0)
    f["bo_strength_dn"] = (1 - c / prev_3_lo).clip(lower=0)
    _y_hi_lag = y_hi.shift(1)
    _y_lo_lag = y_lo.shift(1)
    f["y_hi_surprise"] = _y_hi_lag - _y_hi_lag.ewm(span=7, adjust=False).mean()
    f["y_lo_surprise"] = _y_lo_lag - _y_lo_lag.ewm(span=7, adjust=False).mean()
    neg_ret = ret.clip(upper=0)
    f["dn_vol_5"]  = neg_ret.rolling(5).std()
    f["dn_vol_20"] = neg_ret.rolling(20).std()
    sma50 = c.rolling(50).mean()
    f["below_sma50"]    = (c < sma50).astype(float)
    f["below_sma50_5d"] = f["below_sma50"].rolling(5).min().fillna(0)

    fc = AD["feat_cols"]
    f  = f.replace([np.inf, -np.inf], np.nan)
    F  = f[fc].dropna()
    if F.empty:
        return None

    # ── Batch prediction ────────────────────────────────────────────────
    def _scalar(x):
        if hasattr(x, "item"):
            try: return float(x.item())
            except Exception: pass
        return float(np.asarray(x).ravel()[0])

    if AD.get("ensemble") and AD.get("constituents"):
        yhi = np.mean([con["m_hi"].predict(F) for con in AD["constituents"]], axis=0)
        ylo = np.mean([con["m_lo"].predict(F) for con in AD["constituents"]], axis=0)
        if AD.get("blended") and float(AD.get("alpha", 1.0)) < 1.0:
            a = float(AD["alpha"])
            yhi = a * yhi + (1-a) * float(AD.get("mu_hi", 0))
            ylo = a * ylo + (1-a) * float(AD.get("mu_lo", 0))
    else:
        yhi = AD["hi_model"].predict(F)
        ylo = AD["lo_model"].predict(F)

    # Direction head — vectorised predict_proba
    dh = AD.get("direction_head")
    if dh is not None and dh.get("classifier") is not None:
        try:
            clf         = dh["classifier"]
            beta_base   = float(dh.get("beta", 1.0))
            reduction   = float(dh.get("beta_trend_reduction", 0.0))
            trend_sat   = float(dh.get("trend_saturation", 0.05))
            trend_feat  = dh.get("trend_feature", "ret_5")
            d_bull_mean = float(dh.get("d_bull_mean", 0.0))
            d_bear_mean = float(dh.get("d_bear_mean", 0.0))
            p_bull      = clf.predict_proba(F)[:, 1]
            t_vals      = F[trend_feat].fillna(0).values if trend_feat in F.columns else np.zeros(len(F))
            t_str       = np.minimum(np.abs(t_vals) / trend_sat, 1.0)
            b_eff       = np.clip(beta_base * (1.0 - reduction * t_str), 0.0, 1.0)
            m_pred      = (yhi + ylo) / 2.0
            d_pred      = (yhi - ylo) / 2.0
            d_dir       = p_bull * d_bull_mean + (1 - p_bull) * d_bear_mean
            d_blnd      = b_eff * d_pred + (1 - b_eff) * d_dir
            yhi         = m_pred + d_blnd
            ylo         = m_pred - d_blnd
        except Exception:
            pass

    c_vals       = c.reindex(F.index).values
    pred_hi_vals = c_vals * (1 + np.clip(yhi, 0, None))
    pred_lo_vals = c_vals * (1 - np.clip(ylo, 0, None))

    # Shift index: feature row at date D predicts the NEXT bar (target = D+1).
    # Use the actual next index element rather than D + 1 calendar day, so we
    # handle non-consecutive dates (e.g. missing weekends) correctly.
    idx_arr    = np.asarray(F.index, dtype="datetime64[ns]")
    next_dates = np.empty(len(F), dtype="datetime64[ns]")
    next_dates[:-1] = idx_arr[1:]
    next_dates[-1]  = idx_arr[-1] + np.timedelta64(1, "D")

    result = pd.DataFrame(
        {"close_asof": c_vals, "pred_high": pred_hi_vals, "pred_low": pred_lo_vals},
        index=pd.DatetimeIndex(next_dates, name="target_date"),
    )
    return result[~result.index.duplicated(keep="last")]


@st.cache_data(ttl=3600 * 6, show_spinner="Building extended 2-year predictions …")
def _build_ct_predictions_extended(model_mtime: float = 0.0):
    """Build CT predictions using yfinance daily data fetched from 2024-02-01.

    Provides 115+ days of feature warmup before the May 26, 2024 backtest start
    so all 90-day rolling features (dist_hi_90, corr_30, etc.) are fully
    initialised.  Uses yfinance midnight-UTC daily closes — same source as
    backtest_2year.py — so results match the documented +83.5% two-year figure.

    Returns (preds_df, raw_df) or None on failure.
      preds_df : target_date index with close_asof / pred_high / pred_low
      raw_df   : date index with btc_close / btc_high / btc_low / btc_volume
    """
    path = str(DAILY_MODEL_CT)
    if not os.path.exists(path):
        return None
    try:
        AD = joblib.load(path)
    except Exception:
        return None

    FETCH_START = "2024-02-01"
    FETCH_END   = (datetime.now(timezone.utc).date()
                   + pd.Timedelta(days=1).to_pytimedelta()).strftime("%Y-%m-%d")

    # ── BTC daily (yfinance midnight-UTC close) ─────────────────────
    try:
        d_btc = yf.download("BTC-USD", start=FETCH_START, end=FETCH_END,
                            progress=False, auto_adjust=False)
        if isinstance(d_btc.columns, pd.MultiIndex):
            d_btc.columns = [c[0] for c in d_btc.columns]
        d_btc.index = pd.DatetimeIndex(d_btc.index).tz_localize(None).normalize()
    except Exception:
        return None

    raw_df = pd.DataFrame({
        "btc_close":  d_btc["Close"],
        "btc_high":   d_btc["High"],
        "btc_low":    d_btc["Low"],
        "btc_volume": d_btc["Volume"],
    })

    # ── Macro (yfinance daily) ──────────────────────────────────────
    SYMS = {"eth":"ETH-USD","spx":"^GSPC","ndx":"^IXIC",
            "vix":"^VIX","gold":"GC=F","dxy":"DX-Y.NYB","tnx":"^TNX"}
    df = raw_df.copy()
    for name, sym in SYMS.items():
        try:
            d = yf.download(sym, start=FETCH_START, end=FETCH_END,
                            progress=False, auto_adjust=False)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = [c[0] for c in d.columns]
            d.index = pd.DatetimeIndex(d.index).tz_localize(None).normalize()
            df[f"{name}_close"] = d["Close"].reindex(df.index)
        except Exception:
            pass

    # ── On-chain (blockchain.info) ──────────────────────────────────
    ONCHAIN = ["hash-rate","difficulty","n-transactions","miners-revenue",
               "n-unique-addresses","transaction-fees-usd","mempool-size",
               "estimated-transaction-volume-usd","market-cap",
               "avg-block-size","cost-per-transaction"]
    for m in ONCHAIN:
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
        except Exception:
            pass

    df = df.sort_index().ffill(limit=5)

    # ── Feature engineering (identical to _build_ct_batch_predictions) ─
    feat  = pd.DataFrame(index=df.index)
    c     = df["btc_close"]; h = df["btc_high"]
    l_    = df["btc_low"];   v = df["btc_volume"]
    ret   = np.log(c).diff()
    for k in [1,3,5,7,14,30]: feat[f"ret_{k}"] = ret.rolling(k).sum()
    for k in [5,10,20,30]:    feat[f"vol_{k}"] = ret.rolling(k).std()
    prev_c = c.shift(1)
    tr = pd.concat([(h-l_),(h-prev_c).abs(),(l_-prev_c).abs()], axis=1).max(axis=1)
    for k in [7,14,30]: feat[f"atr_{k}"] = tr.rolling(k).mean()/c
    feat["range_today"] = (h-l_)/c
    feat["range_ma7"]   = ((h-l_)/c).rolling(7).mean()
    feat["range_ma30"]  = ((h-l_)/c).rolling(30).mean()
    feat["range_std30"] = ((h-l_)/c).rolling(30).std()
    gain  = c.diff().clip(lower=0).rolling(14).mean()
    loss  = (-c.diff().clip(upper=0)).rolling(14).mean()
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
        s = df[col]
        for k in (1,5,20): feat[f"{nm}_ret_{k}"] = np.log(s).diff(k)
        feat[f"{nm}_vol_20"] = np.log(s).diff().rolling(20).std()
    for nm, col in [("spx","spx_close"),("ndx","ndx_close"),
                    ("gold","gold_close"),("dxy","dxy_close")]:
        if col in df.columns:
            feat[f"btc_{nm}_corr_30"] = ret.rolling(30).corr(np.log(df[col]).diff())
    for col in [x for x in df.columns if x.startswith("oc_")]:
        s = df[col].astype(float); sl = np.log(s.replace(0, np.nan))
        feat[f"{col}_d1"]  = sl.diff(1)
        feat[f"{col}_d7"]  = sl.diff(7)
        feat[f"{col}_z30"] = (sl - sl.rolling(30).mean())/sl.rolling(30).std()
    nh, nl = h.shift(-1), l_.shift(-1)
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

    fc = AD["feat_cols"]
    feat = feat.replace([np.inf, -np.inf], np.nan)
    for col in fc:
        if col not in feat.columns: feat[col] = np.nan
    F = feat[fc].dropna()
    if F.empty:
        return None

    # ── Batch prediction (same as _build_ct_batch_predictions) ────────
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
    dh = AD.get("direction_head")
    if dh is not None and dh.get("classifier") is not None:
        try:
            clf = dh["classifier"]
            beta_base   = float(dh.get("beta", 1.0))
            reduction   = float(dh.get("beta_trend_reduction", 0.0))
            trend_sat   = float(dh.get("trend_saturation", 0.05))
            trend_feat  = dh.get("trend_feature", "ret_5")
            d_bull_mean = float(dh.get("d_bull_mean", 0.0))
            d_bear_mean = float(dh.get("d_bear_mean", 0.0))
            p_bull   = clf.predict_proba(F)[:, 1]
            t_vals   = F[trend_feat].fillna(0).values if trend_feat in F.columns else np.zeros(len(F))
            t_str    = np.minimum(np.abs(t_vals)/trend_sat, 1.0)
            b_eff    = np.clip(beta_base*(1.0 - reduction*t_str), 0.0, 1.0)
            m_pred   = (yhi + ylo)/2.0; d_pred = (yhi - ylo)/2.0
            d_dir    = p_bull*d_bull_mean + (1-p_bull)*d_bear_mean
            d_blnd   = b_eff*d_pred + (1-b_eff)*d_dir
            yhi = m_pred + d_blnd; ylo = m_pred - d_blnd
        except Exception:
            pass

    c_vals       = c.reindex(F.index).values
    pred_hi_vals = c_vals * (1 + np.clip(yhi, 0, None))
    pred_lo_vals = c_vals * (1 - np.clip(ylo, 0, None))
    idx_arr    = np.asarray(F.index, dtype="datetime64[ns]")
    next_dates = np.empty(len(F), dtype="datetime64[ns]")
    next_dates[:-1] = idx_arr[1:]; next_dates[-1] = idx_arr[-1] + np.timedelta64(1,"D")
    preds_df = pd.DataFrame(
        {"close_asof": c_vals, "pred_high": pred_hi_vals, "pred_low": pred_lo_vals},
        index=pd.DatetimeIndex(next_dates, name="target_date"),
    )
    preds_df = preds_df[~preds_df.index.duplicated(keep="last")]
    return preds_df, raw_df


# Fixed backtest period end dates — locked to the dates established on 2026-05-28.
# Only the OOS period rolls daily; these never change until the model is retrained.
_BT_BEAR_END_LOCKED = "2026-05-28"
_BT_FULL_END_LOCKED = "2026-05-28"


def run_full_period_backtest(end_date_iso: str,
                             backtest_start_iso: str = "2024-05-26",
                             initial_capital: float = 100_000.0,
                             model_mtime: float = 0.0):
    """Full-period TF2 backtest using extended yfinance daily data.

    Fetches from 2024-02-01 to ensure 90-day feature warmup before the
    May 26, 2024 backtest start — reproduces the backtest_2year.py +83.5% figure.
    Returns the same dict structure as run_tf1_backtest.
    """
    WARMUP = 35

    ext = _build_ct_predictions_extended(model_mtime=model_mtime)
    if ext is None:
        return None
    preds, raw_df = ext

    closes = raw_df["btc_close"]
    highs  = raw_df["btc_high"]
    lows   = raw_df["btc_low"]

    end_dt   = pd.Timestamp(end_date_iso)
    start_dt = pd.Timestamp(backtest_start_iso)
    # Load 60 extra calendar days before start so MA30/clean10d/dn_score are
    # fully warm by the time the backtest window actually begins.
    pre_dt   = start_dt - pd.Timedelta(days=60)
    preds    = preds.loc[(preds.index >= pre_dt) & (preds.index <= end_dt)].copy()
    if len(preds) < WARMUP + 3:
        return None

    preds["actual_high"]  = highs.reindex(preds.index).values
    preds["actual_low"]   = lows.reindex(preds.index).values
    preds["actual_close"] = closes.reindex(preds.index).values

    comp = (preds.dropna(subset=["actual_high","actual_low","actual_close"])
                 .reset_index())
    N = len(comp)
    if N < WARMUP + 3:
        return None

    dates   = pd.DatetimeIndex(comp["target_date"])
    # _bt0: first bar at/after the requested start, and always >= WARMUP so
    # all rolling indicators (MA30, clean_10d, dn_score) are fully populated.
    _bt0 = max(WARMUP, int(dates.searchsorted(start_dt)))
    if N - _bt0 < 3:
        return None
    c_asof  = comp["close_asof"].values.astype(float)
    exec_px = comp["actual_close"].values.astype(float)
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

    u1 = (ehma3 > 0.5) & (hb3 >= 2)
    d1 = (lb3 >= 2)    & (elma3 > 0.5)
    d2 = ehma3 < -1.0
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
        lo_i = max(0, i-10)
        clean_10d[i] = not bool(np.any(d1[lo_i:i] | d2[lo_i:i]))

    # ── V-Gate 3-bar: V-reversal capitulation as third entry gate ────────────
    # dn_score replicates the live signal formula; v_rev_bar fires when a
    # single bar shows capitulation (dn_score > 0.8 AND err_lo > 5%).
    # v_recent[i] = True if v_rev_bar fired within the last 3 bars — bridges
    # the 1-2 bar lag before U1 co-fires after the bounce starts.
    # 30-bar rolling mean keeps the normaliser stable across regimes.
    _DN_NORM_W    = 30
    roll_ehi_norm = np.array([
        float(np.mean(err_hi[max(0, i - _DN_NORM_W + 1):i + 1]))
        for i in range(N)
    ])
    dn_score_arr = np.zeros(N)
    for i in range(N):
        norm = max(abs(roll_ehi_norm[i]), 0.01)
        dn_score_arr[i] = (
            (-ehma3[i] / norm)                               * 0.30 +
            (lb3[i]    / 3.0)                                * 0.30 +
            (elma3[i]  / max(abs(elma3[i]), 0.10))           * 0.20 +
            float(lo_brk[i])                                 * 0.20
        )
    v_rev_bar = (dn_score_arr > 0.8) & (err_lo > 5.0)
    v_recent  = np.zeros(N, dtype=bool)
    for i in range(N):
        v_recent[i] = bool(np.any(v_rev_bar[max(0, i-2):i+1]))

    tf1_entry = u1 & (above_ma30 | clean_10d | v_recent)

    nav     = initial_capital; pos = "CASH"; btc_qty = 0.0
    e_price = e_nav = e_date = e_trigger = None
    trades  = []; nav_arr = np.full(N, np.nan)

    for i in range(N):
        si    = i - 1; price = exec_px[i]
        if i < _bt0:
            nav_arr[i] = initial_capital; continue
        if pos == "LONG":
            cur = btc_qty * price
            if si >= 0:
                should_exit = bool(d3[si] or (d2[si] and not bull_regime[si]))
                exit_lbl    = "D3" if d3[si] else "D2 (bear)"
            else:
                should_exit = False; exit_lbl = "?"
            if should_exit:
                nav = cur
                trades.append(dict(
                    entry_date=e_date, entry_price=e_price, entry_nav=e_nav,
                    entry_trigger=e_trigger, exit_date=dates[i], exit_price=price,
                    exit_nav=nav, pnl_pct=(price/e_price-1)*100,
                    pnl_abs=nav-e_nav, exit_signal=exit_lbl,
                    duration_days=(dates[i]-e_date).days,
                ))
                pos = "CASH"; btc_qty = 0.0
            else:
                nav = cur
        else:
            if si >= 0 and tf1_entry[si]:
                btc_qty = nav/price; e_price = price; e_date = dates[i]
                e_nav = nav; pos = "LONG"
                if v_recent[si] and not above_ma30[si] and not clean_10d[si]:
                    e_trigger = "U1 + V-reversal"
                elif above_ma30[si] and clean_10d[si]:
                    e_trigger = "U1 + ↑MA30 + clean10d"
                elif above_ma30[si]:
                    e_trigger = "U1 + ↑MA30"
                else:
                    e_trigger = "U1 + clean10d"
        nav_arr[i] = btc_qty * price if pos == "LONG" else nav

    if pos == "LONG":
        nav_arr[N-1] = btc_qty * exec_px[N-1]

    nav_series = pd.Series(nav_arr[_bt0:], index=dates[_bt0:]).ffill()
    bh_series  = pd.Series(
        initial_capital * exec_px[_bt0:] / exec_px[_bt0], index=dates[_bt0:])

    # ── Statistics (identical to run_tf1_backtest) ───────────────────
    final_nav = float(nav_series.iloc[-1]); final_bh = float(bh_series.iloc[-1])
    strat_ret = (final_nav/initial_capital - 1)*100
    bh_ret    = (final_bh/initial_capital  - 1)*100
    wins      = [t for t in trades if t["pnl_pct"] > 0]
    losses    = [t for t in trades if t["pnl_pct"] <= 0]
    win_rate  = 100*len(wins)/len(trades) if trades else 0.0
    avg_pnl   = float(np.mean([t["pnl_pct"] for t in trades])) if trades else 0.0
    best_t    = float(max([t["pnl_pct"] for t in trades])) if trades else 0.0
    worst_t   = float(min([t["pnl_pct"] for t in trades])) if trades else 0.0
    rm        = nav_series.cummax()
    max_dd    = float(((nav_series - rm)/rm*100).min())
    rm_bh     = bh_series.cummax()
    bh_max_dd = float(((bh_series - rm_bh)/rm_bh*100).min())
    days_in   = sum(t["duration_days"] for t in trades)
    tot_days  = max(1, (dates[N-1] - dates[_bt0]).days)
    rf_daily  = (1.045)**(1/252) - 1
    dr_s      = nav_series.pct_change().fillna(0)
    dr_b      = bh_series.pct_change().fillna(0)
    exc_s     = dr_s - rf_daily; exc_b = dr_b - rf_daily
    sharpe    = float(exc_s.mean()/exc_s.std()*np.sqrt(252)) if exc_s.std()>0 else 0.0
    bh_sharpe = float(exc_b.mean()/exc_b.std()*np.sqrt(252)) if exc_b.std()>0 else 0.0
    # Tax: net gains/losses per calendar year; 35% rate on annual net gains only
    TAX_RATE = 0.35
    _annual_net: dict = {}
    for t in trades:
        yr = pd.Timestamp(t["exit_date"]).year
        _annual_net[yr] = _annual_net.get(yr, 0.0) + t["pnl_abs"]
    total_tax_paid = sum(TAX_RATE * max(0.0, net) for net in _annual_net.values())
    after_tax_nav  = final_nav - total_tax_paid
    after_tax_ret  = (after_tax_nav/initial_capital - 1)*100

    bull_regime_series = pd.Series(bull_regime[_bt0:].astype(bool), index=dates[_bt0:])

    return dict(
        strategy   = "TF2",
        trades     = trades,
        nav_series = nav_series,
        bh_series  = bh_series,
        open_pos   = pos == "LONG",
        open_entry = (dict(price=e_price, date=e_date, nav=e_nav) if pos == "LONG" else None),
        bull_regime_series = bull_regime_series,
        stats = dict(
            strategy        = "TF2",
            initial_capital = initial_capital,
            final_nav       = final_nav,      final_bh      = final_bh,
            strat_ret       = strat_ret,      bh_ret        = bh_ret,
            alpha_abs       = final_nav - final_bh,
            n_trades        = len(trades),
            n_wins          = len(wins),      n_losses      = len(losses),
            win_rate        = win_rate,       avg_pnl       = avg_pnl,
            best_trade      = best_t,         worst_trade   = worst_t,
            max_drawdown    = max_dd,         bh_max_dd     = bh_max_dd,
            sharpe          = sharpe,         bh_sharpe     = bh_sharpe,
            time_in_mkt     = 100*days_in/tot_days,
            after_tax_nav   = after_tax_nav,  after_tax_ret = after_tax_ret,
            total_tax_paid  = total_tax_paid,
            start_date      = start_dt,      # requested window start (for label display)
            end_date        = dates[N-1],
        ),
    )


@st.cache_data(show_spinner="Loading fixed-period backtest …")
def _run_fixed_period_backtest(end_date_iso: str, backtest_start_iso: str,
                                model_mtime: float = 0.0):
    """Cached wrapper for fixed-period backtests (Bear / Bull / Full Market).

    No TTL — results persist for the app process lifetime and only invalidate
    when the model file changes (via model_mtime).  The OOS period is NOT routed
    through here; it calls run_full_period_backtest directly so it rolls daily.
    """
    return run_full_period_backtest(end_date_iso, backtest_start_iso,
                                    model_mtime=model_mtime)


@st.cache_data(show_spinner="Running MSTR backtest …")
def run_mstr_backtest(end_date_iso: str,
                      backtest_start_iso: str = "2024-05-26",
                      initial_capital: float = 100_000.0,
                      model_mtime: float = 0.0):
    """MSTR backtest driven by BTC TF2+V-Gate signals.

    Computes all entry/exit signals from the BTC CT model (identical logic to
    run_full_period_backtest) but executes trades in MSTR stock instead of BTC.
    MSTR daily closes (split-adjusted) are fetched from yfinance and forward-filled
    across weekends/holidays so each BTC signal date has a valid execution price.
    """
    WARMUP = 35

    ext = _build_ct_predictions_extended(model_mtime=model_mtime)
    if ext is None:
        return None
    preds, raw_df = ext

    btc_closes = raw_df["btc_close"]
    btc_highs  = raw_df["btc_high"]
    btc_lows   = raw_df["btc_low"]

    end_dt   = pd.Timestamp(end_date_iso)
    start_dt = pd.Timestamp(backtest_start_iso)
    pre_dt   = start_dt - pd.Timedelta(days=60)
    preds    = preds.loc[(preds.index >= pre_dt) & (preds.index <= end_dt)].copy()
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

    # ── Fetch MSTR prices (split-adjusted so Aug-2024 10-for-1 split is seamless) ──
    try:
        d_mstr = yf.download(
            "MSTR",
            start=pre_dt.strftime("%Y-%m-%d"),
            end=(end_dt + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
            progress=False, auto_adjust=True,
        )
        if isinstance(d_mstr.columns, pd.MultiIndex):
            d_mstr.columns = [c[0] for c in d_mstr.columns]
        d_mstr.index = pd.DatetimeIndex(d_mstr.index).tz_localize(None).normalize()
    except Exception:
        return None

    if d_mstr.empty or "Close" not in d_mstr.columns:
        return None

    # Forward-fill MSTR to all calendar days (weekends use last trading close)
    mstr_raw = d_mstr["Close"].sort_index()
    mstr_all = mstr_raw.reindex(
        pd.date_range(mstr_raw.index[0],
                      max(mstr_raw.index[-1], end_dt), freq="D")
    ).ffill()
    mstr_px = mstr_all.reindex(dates).ffill().bfill().values.astype(float)

    # ── BTC signal arrays (identical to run_full_period_backtest) ────────────
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

    u1 = (ehma3 > 0.5) & (hb3 >= 2)
    d1 = (lb3 >= 2)    & (elma3 > 0.5)
    d2 = ehma3 < -1.0
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
        lo_i = max(0, i-10)
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
    v_rev_bar = (dn_score_arr > 0.8) & (err_lo > 5.0)
    v_recent  = np.zeros(N, dtype=bool)
    for i in range(N):
        v_recent[i] = bool(np.any(v_rev_bar[max(0, i-2):i+1]))

    tf1_entry = u1 & (above_ma30 | clean_10d | v_recent)

    # ── Backtest loop — execute in MSTR ──────────────────────────────────────
    nav      = initial_capital; pos = "CASH"; mstr_qty = 0.0
    e_price = e_nav = e_date = e_trigger = None
    trades  = []; nav_arr = np.full(N, np.nan)

    for i in range(N):
        si    = i - 1
        price = mstr_px[i]
        if i < _bt0:
            nav_arr[i] = initial_capital; continue
        if not np.isfinite(price) or price <= 0:
            nav_arr[i] = mstr_qty * mstr_px[i-1] if pos == "LONG" and i > 0 else nav
            continue
        if pos == "LONG":
            cur = mstr_qty * price
            if si >= 0:
                should_exit = bool(d3[si] or (d2[si] and not bull_regime[si]))
                exit_lbl    = "D3" if d3[si] else "D2 (bear)"
            else:
                should_exit = False; exit_lbl = "?"
            if should_exit:
                nav = cur
                trades.append(dict(
                    entry_date=e_date,    entry_price=e_price, entry_nav=e_nav,
                    entry_trigger=e_trigger, exit_date=dates[i], exit_price=price,
                    exit_nav=nav, pnl_pct=(price/e_price-1)*100,
                    pnl_abs=nav-e_nav,   exit_signal=exit_lbl,
                    duration_days=(dates[i]-e_date).days,
                ))
                pos = "CASH"; mstr_qty = 0.0
            else:
                nav = cur
        else:
            if si >= 0 and tf1_entry[si]:
                mstr_qty = nav / price; e_price = price; e_date = dates[i]
                e_nav = nav; pos = "LONG"
                if v_recent[si] and not above_ma30[si] and not clean_10d[si]:
                    e_trigger = "U1 + V-reversal"
                elif above_ma30[si] and clean_10d[si]:
                    e_trigger = "U1 + ↑MA30 + clean10d"
                elif above_ma30[si]:
                    e_trigger = "U1 + ↑MA30"
                else:
                    e_trigger = "U1 + clean10d"
        nav_arr[i] = mstr_qty * price if pos == "LONG" else nav

    if pos == "LONG" and np.isfinite(mstr_px[N-1]) and mstr_px[N-1] > 0:
        nav_arr[N-1] = mstr_qty * mstr_px[N-1]

    nav_series = pd.Series(nav_arr[_bt0:], index=dates[_bt0:]).ffill()
    bh_px0     = mstr_px[_bt0]
    bh_series  = pd.Series(
        initial_capital * mstr_px[_bt0:] / bh_px0, index=dates[_bt0:])

    # ── Statistics ────────────────────────────────────────────────────────────
    final_nav = float(nav_series.iloc[-1]); final_bh = float(bh_series.iloc[-1])
    strat_ret = (final_nav/initial_capital - 1)*100
    bh_ret    = (final_bh/initial_capital  - 1)*100
    wins      = [t for t in trades if t["pnl_pct"] > 0]
    losses    = [t for t in trades if t["pnl_pct"] <= 0]
    win_rate  = 100*len(wins)/len(trades) if trades else 0.0
    avg_pnl   = float(np.mean([t["pnl_pct"] for t in trades])) if trades else 0.0
    best_t    = float(max([t["pnl_pct"] for t in trades])) if trades else 0.0
    worst_t   = float(min([t["pnl_pct"] for t in trades])) if trades else 0.0
    rm        = nav_series.cummax()
    max_dd    = float(((nav_series - rm)/rm*100).min())
    rm_bh     = bh_series.cummax()
    bh_max_dd = float(((bh_series - rm_bh)/rm_bh*100).min())
    days_in   = sum(t["duration_days"] for t in trades)
    tot_days  = max(1, (dates[N-1] - dates[_bt0]).days)
    rf_daily  = (1.045)**(1/252) - 1
    dr_s      = nav_series.pct_change().fillna(0)
    dr_b      = bh_series.pct_change().fillna(0)
    exc_s     = dr_s - rf_daily; exc_b = dr_b - rf_daily
    sharpe    = float(exc_s.mean()/exc_s.std()*np.sqrt(252)) if exc_s.std()>0 else 0.0
    bh_sharpe = float(exc_b.mean()/exc_b.std()*np.sqrt(252)) if exc_b.std()>0 else 0.0
    TAX_RATE = 0.35
    _annual_net: dict = {}
    for t in trades:
        yr = pd.Timestamp(t["exit_date"]).year
        _annual_net[yr] = _annual_net.get(yr, 0.0) + t["pnl_abs"]
    total_tax_paid = sum(TAX_RATE * max(0.0, net) for net in _annual_net.values())
    after_tax_nav  = final_nav - total_tax_paid
    after_tax_ret  = (after_tax_nav/initial_capital - 1)*100

    bull_regime_series = pd.Series(bull_regime[_bt0:].astype(bool), index=dates[_bt0:])

    return dict(
        strategy   = "TF2+V-Gate (MSTR)",
        trades     = trades,
        nav_series = nav_series,
        bh_series  = bh_series,
        open_pos   = pos == "LONG",
        open_entry = (dict(price=e_price, date=e_date, nav=e_nav) if pos == "LONG" else None),
        bull_regime_series = bull_regime_series,
        stats = dict(
            strategy        = "TF2+V-Gate (MSTR)",
            initial_capital = initial_capital,
            final_nav       = final_nav,      final_bh      = final_bh,
            strat_ret       = strat_ret,      bh_ret        = bh_ret,
            alpha_abs       = final_nav - final_bh,
            n_trades        = len(trades),
            n_wins          = len(wins),      n_losses      = len(losses),
            win_rate        = win_rate,       avg_pnl       = avg_pnl,
            best_trade      = best_t,         worst_trade   = worst_t,
            max_drawdown    = max_dd,         bh_max_dd     = bh_max_dd,
            sharpe          = sharpe,         bh_sharpe     = bh_sharpe,
            time_in_mkt     = 100*days_in/tot_days,
            after_tax_nav   = after_tax_nav,  after_tax_ret = after_tax_ret,
            total_tax_paid  = total_tax_paid,
            start_date      = start_dt,
            end_date        = dates[N-1],
        ),
    )


@st.cache_data(show_spinner="Loading fixed-period MSTR backtest …")
def _run_fixed_period_mstr_backtest(end_date_iso: str, backtest_start_iso: str,
                                    model_mtime: float = 0.0):
    """Cached wrapper for fixed-period MSTR backtests (no TTL, invalidates on model change)."""
    return run_mstr_backtest(end_date_iso, backtest_start_iso,
                             model_mtime=model_mtime)


@st.cache_data(ttl=3600 * 6, show_spinner="Running strategy backtest …")
def run_tf1_backtest(end_date_iso: str, initial_capital: float = 100_000.0,
                     strategy: str = "TF2", start_date_iso: str = None):
    """Run a trading strategy backtest on a configurable window.

    Entry (both TF1 and TF2):
        U1 active (err_hi_ma3 > +0.5% AND hi_breaks_3d ≥ 2)
        AND (BTC > MA30  OR  clean_10d)

    Exit — TF1 (conservative, bear-optimised):
        D2 (err_hi_ma3 < −1%)  OR  D3 (exhaustion canary)

    Exit — TF2 (regime-adaptive, default):
        BULL regime (above_ma30 AND MA30 slope > 0) → exit D3 only (patient)
        BEAR / NEUTRAL regime                       → exit D2 OR D3 (defensive)

    Args:
        start_date_iso: If provided, backtest starts from this date.
                        If None, defaults to ~1-year rolling (end - 400 days).

    Execution: 1-bar lag — signal fires bar i, trade executes at bar i+1 close.
    Uses _build_ct_batch_predictions() for O(1) lookups. Cached 6 h.

    Returns a dict with trades, nav_series, bh_series, stats, open_pos.
    Returns None if insufficient data or the CT model is unavailable.
    """
    WARMUP = 35  # bars before backtest begins (MA30 needs 30 + 5 buffer)

    preds = _build_ct_batch_predictions()
    if preds is None or preds.empty:
        return None

    raw    = _fetch_daily_raw()
    closes = raw["btc_close"]
    highs  = raw["btc_high"]
    lows   = raw["btc_low"]

    # Determine window: explicit start or rolling ~1-year default
    end_dt   = pd.Timestamp(end_date_iso)
    start_dt = pd.Timestamp(start_date_iso) if start_date_iso else end_dt - pd.Timedelta(days=400)
    preds    = preds.loc[(preds.index >= start_dt) & (preds.index <= end_dt)].copy()
    if len(preds) < WARMUP + 3:
        return None

    # Attach actual H/L/close for each target date (vectorised join)
    preds["actual_high"]  = highs.reindex(preds.index).values
    preds["actual_low"]   = lows.reindex(preds.index).values
    preds["actual_close"] = closes.reindex(preds.index).values

    comp = (preds.dropna(subset=["actual_high","actual_low","actual_close"])
                 .reset_index())          # target_date becomes a column
    N = len(comp)
    if N < WARMUP + 3:
        return None

    dates   = pd.DatetimeIndex(comp["target_date"])
    c_asof  = comp["close_asof"].values.astype(float)
    exec_px = comp["actual_close"].values.astype(float)
    pred_hi = comp["pred_high"].values.astype(float)
    pred_lo = comp["pred_low"].values.astype(float)
    act_hi  = comp["actual_high"].values.astype(float)
    act_lo  = comp["actual_low"].values.astype(float)

    # ── Signal arrays ────────────────────────────────────────────────────
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

    u1 = (ehma3 > 0.5) & (hb3 >= 2)
    d1 = (lb3 >= 2)    & (elma3 > 0.5)
    d2 = ehma3 < -1.0
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
        ma30[i] = np.mean(c_asof[max(0, i-w+1): i+1])
    above_ma30 = c_asof > ma30

    # MA30 slope (5-bar): used for regime detection in TF2
    ma30_slope_pos = np.zeros(N, dtype=bool)
    for i in range(N):
        if i >= 5 and np.isfinite(ma30[i]) and np.isfinite(ma30[i-5]):
            ma30_slope_pos[i] = ma30[i] > ma30[i-5]
    # BULL regime: price above MA30 AND MA30 is rising
    bull_regime = above_ma30 & ma30_slope_pos

    clean_10d = np.zeros(N, dtype=bool)
    for i in range(N):
        lo_i = max(0, i-10)
        clean_10d[i] = not bool(np.any(d1[lo_i:i] | d2[lo_i:i]))

    # ── V-Gate 3-bar: V-reversal capitulation as third entry gate ────────────
    # 30-bar rolling mean keeps the normaliser stable across regimes.
    _DN_NORM_W    = 30
    roll_ehi_norm = np.array([
        float(np.mean(err_hi[max(0, i - _DN_NORM_W + 1):i + 1]))
        for i in range(N)
    ])
    dn_score_arr = np.zeros(N)
    for i in range(N):
        norm = max(abs(roll_ehi_norm[i]), 0.01)
        dn_score_arr[i] = (
            (-ehma3[i] / norm)                               * 0.30 +
            (lb3[i]    / 3.0)                                * 0.30 +
            (elma3[i]  / max(abs(elma3[i]), 0.10))           * 0.20 +
            float(lo_brk[i])                                 * 0.20
        )
    v_rev_bar = (dn_score_arr > 0.8) & (err_lo > 5.0)
    v_recent  = np.zeros(N, dtype=bool)
    for i in range(N):
        v_recent[i] = bool(np.any(v_rev_bar[max(0, i-2):i+1]))

    tf1_entry = u1 & (above_ma30 | clean_10d | v_recent)
    tf1_exit  = d2 | d3   # TF1 always exits D2|D3

    # ── Backtest loop (1-bar lag) ─────────────────────────────────────────
    nav      = initial_capital
    pos      = "CASH"
    btc_qty  = 0.0
    e_price = e_nav = e_date = e_trigger = None
    trades  = []
    nav_arr = np.full(N, np.nan)

    for i in range(N):
        si    = i - 1        # signal bar index
        price = exec_px[i]  # execution price = bar i close

        if i < WARMUP:
            nav_arr[i] = initial_capital
            continue

        if pos == "LONG":
            cur = btc_qty * price
            # Regime-adaptive exit for TF2; fixed D2|D3 for TF1
            if si >= 0:
                if strategy == "TF2":
                    # Bull regime: patience — wait for structural D3 reversal
                    # Bear/neutral regime: defensive — exit on first D2 or D3
                    should_exit = bool(d3[si] or (d2[si] and not bull_regime[si]))
                    exit_lbl    = "D3" if d3[si] else "D2 (bear)"
                else:   # TF1
                    should_exit = bool(tf1_exit[si])
                    exit_lbl    = "D3" if d3[si] else "D2"
            else:
                should_exit = False; exit_lbl = "?"
            if should_exit:
                nav = cur
                trades.append(dict(
                    entry_date    = e_date,    entry_price = e_price,
                    entry_nav     = e_nav,     entry_trigger = e_trigger,
                    exit_date     = dates[i],  exit_price  = price,
                    exit_nav      = nav,
                    pnl_pct       = (price / e_price - 1) * 100,
                    pnl_abs       = nav - e_nav,
                    exit_signal   = exit_lbl,
                    duration_days = (dates[i] - e_date).days,
                ))
                pos = "CASH"; btc_qty = 0.0
            else:
                nav = cur
        else:  # CASH
            if si >= 0 and tf1_entry[si]:
                btc_qty  = nav / price
                e_price  = price; e_date = dates[i]
                e_nav    = nav;   pos    = "LONG"
                if v_recent[si] and not above_ma30[si] and not clean_10d[si]:
                    e_trigger = "U1 + V-reversal"
                elif above_ma30[si] and clean_10d[si]:
                    e_trigger = "U1 + ↑MA30 + clean10d"
                elif above_ma30[si]:
                    e_trigger = "U1 + ↑MA30"
                else:
                    e_trigger = "U1 + clean10d"

        nav_arr[i] = btc_qty * price if pos == "LONG" else nav

    if pos == "LONG":
        nav_arr[N-1] = btc_qty * exec_px[N-1]

    nav_series = pd.Series(nav_arr[WARMUP:], index=dates[WARMUP:]).ffill()
    bh_series  = pd.Series(
        initial_capital * exec_px[WARMUP:] / exec_px[WARMUP],
        index=dates[WARMUP:],
    )

    # ── Statistics ───────────────────────────────────────────────────────
    final_nav = float(nav_series.iloc[-1])
    final_bh  = float(bh_series.iloc[-1])
    strat_ret = (final_nav / initial_capital - 1) * 100
    bh_ret    = (final_bh  / initial_capital - 1) * 100
    wins      = [t for t in trades if t["pnl_pct"] > 0]
    losses    = [t for t in trades if t["pnl_pct"] <= 0]
    win_rate  = 100 * len(wins) / len(trades) if trades else 0.0
    avg_pnl   = float(np.mean([t["pnl_pct"] for t in trades])) if trades else 0.0
    best_t    = float(max([t["pnl_pct"] for t in trades])) if trades else 0.0
    worst_t   = float(min([t["pnl_pct"] for t in trades])) if trades else 0.0
    rm        = nav_series.cummax()
    max_dd    = float(((nav_series - rm) / rm * 100).min())
    days_in   = sum(t["duration_days"] for t in trades)
    tot_days  = max(1, (dates[N-1] - dates[WARMUP]).days)

    # Sharpe ratio (annualised, RF = 4.5%, daily × √252)
    rf_daily   = (1.045) ** (1 / 252) - 1
    dr_strat   = nav_series.pct_change().fillna(0)
    dr_bh      = bh_series.pct_change().fillna(0)
    exc_strat  = dr_strat - rf_daily
    exc_bh     = dr_bh    - rf_daily
    sharpe     = float(exc_strat.mean() / exc_strat.std() * np.sqrt(252)) if exc_strat.std() > 0 else 0.0
    bh_sharpe  = float(exc_bh.mean()    / exc_bh.std()    * np.sqrt(252)) if exc_bh.std()    > 0 else 0.0

    # B&H max drawdown
    rm_bh      = bh_series.cummax()
    bh_max_dd  = float(((bh_series - rm_bh) / rm_bh * 100).min())

    # After-tax NAV — net gains/losses per calendar year; 35% rate on annual net gains only.
    # B&H pays 0% (gains unrealised, no tax event triggered)
    TAX_RATE = 0.35
    _annual_net: dict = {}
    for t in trades:
        yr = pd.Timestamp(t["exit_date"]).year
        _annual_net[yr] = _annual_net.get(yr, 0.0) + t["pnl_abs"]
    total_tax_paid  = sum(TAX_RATE * max(0.0, net) for net in _annual_net.values())
    after_tax_nav   = final_nav - total_tax_paid
    after_tax_ret   = (after_tax_nav / initial_capital - 1) * 100

    bull_regime_series = pd.Series(bull_regime[WARMUP:].astype(bool), index=dates[WARMUP:])

    return dict(
        strategy   = strategy,
        trades     = trades,
        nav_series = nav_series,
        bh_series  = bh_series,
        open_pos   = pos == "LONG",
        open_entry = (dict(price=e_price, date=e_date, nav=e_nav) if pos == "LONG" else None),
        bull_regime_series = bull_regime_series,
        stats = dict(
            strategy        = strategy,
            initial_capital = initial_capital,
            final_nav       = final_nav,      final_bh      = final_bh,
            strat_ret       = strat_ret,      bh_ret        = bh_ret,
            alpha_abs       = final_nav - final_bh,
            n_trades        = len(trades),
            n_wins          = len(wins),      n_losses      = len(losses),
            win_rate        = win_rate,       avg_pnl       = avg_pnl,
            best_trade      = best_t,         worst_trade   = worst_t,
            max_drawdown    = max_dd,         bh_max_dd     = bh_max_dd,
            sharpe          = sharpe,         bh_sharpe     = bh_sharpe,
            time_in_mkt     = 100 * days_in / tot_days,
            after_tax_nav   = after_tax_nav,  after_tax_ret = after_tax_ret,
            total_tax_paid  = total_tax_paid,
            start_date      = dates[WARMUP],
            end_date        = dates[N-1],
        ),
    )


def render_trading_strategy_dashboard(bt_bear, bt_bull, bt_full_oos=None,
                                       bt_full=None, key_suffix: str = "") -> None:
    """Render the TF2 trading strategy summary + backtest dashboard.

    Shows:
      • Strategy rules summary card (entry, exit, regime logic)
      • Unified comparison table: four periods × three columns (TF2+V-Gate | TF2+V-Gate-tax | B&H)
      • Equity-curve charts — one per period — in tabs
      • Expandable trade log for each period

    bt_bear:     Fixed bear window  (today-365d → today)
    bt_bull:     Fixed bull window  (Jun 2024 → Jun 2025)
    bt_full_oos: Full-period result used only for OOS stats extraction
    bt_full:     Full market period  (Jun 2024 → today)
    key_suffix:  appended to plotly_chart keys to avoid DuplicateElementKey.
    """
    st.markdown("---")
    st.subheader("🎯 TF2 + V-Gate Trading Strategy — Regime-Adaptive Backtest")

    if bt_bear is None and bt_bull is None:
        st.info("⚙️ Backtest computing … CT model batch predictions being built. "
                "Refresh in a moment.")
        return

    # Use whichever is available for the rules card period label
    _bt_ref = bt_bear if bt_bear is not None else bt_bull
    s_ref   = _bt_ref["stats"]

    # ── Strategy rules card ───────────────────────────────────────────
    st.markdown("""
<div style='background:#eff6ff; border:2px solid #2563eb; border-radius:12px;
     padding:16px 20px; margin:4px 0 16px 0; font-family:sans-serif;'>

  <div style='font-size:14px; font-weight:700; color:#1e3a8a; margin-bottom:14px;
       letter-spacing:0.3px;'>
    🎯 TF2 + V-Gate &nbsp;—&nbsp; Strategy Rules at a Glance
  </div>

  <!-- ENTRY -->
  <div style='margin-bottom:12px;'>
    <div style='font-size:11px; font-weight:700; color:#1e40af; text-transform:uppercase;
         letter-spacing:0.8px; margin-bottom:6px;'>📥 Entry — two conditions, both required</div>
    <table style='width:100%; border-collapse:collapse; font-size:12px; color:#1e3a8a;'>
      <tr>
        <td style='width:36px; vertical-align:top; padding:3px 6px 3px 0; font-weight:700;
             color:#2563eb;'>①</td>
        <td style='vertical-align:top; padding:3px 0;'>
          <b>U1 signal active</b> — model's predicted highs are being beaten consistently<br>
          <span style='color:#3b82f6; font-size:11px;'>
            err_hi_ma3 &gt; +0.5% &nbsp;&amp;&amp;&nbsp; hi_breaks_3d ≥ 2
          </span>
        </td>
      </tr>
      <tr>
        <td style='width:36px; vertical-align:top; padding:3px 6px 3px 0; font-weight:700;
             color:#2563eb;'>②</td>
        <td style='vertical-align:top; padding:3px 0;'>
          <b>At least one trend gate passes</b> (any of the three):
          <div style='margin:5px 0 0 4px; line-height:2.1;'>
            <span style='background:#dbeafe; border-radius:4px; padding:1px 7px;'>↑ MA30</span>
            &nbsp; BTC close above its 30-day moving average
            <br>
            <span style='background:#dbeafe; border-radius:4px; padding:1px 7px;'>Clean 10d</span>
            &nbsp; No D1 or D2 signal in the prior 10 bars (no recent bearish fingerprint)
            <br>
            <span style='background:#ede9fe; border-radius:4px; padding:1px 7px;'>⚡ V-reversal</span>
            &nbsp; Capitulation spike detected within the last 3 bars
            <span style='color:#7c3aed; font-size:11px;'>
              (dn_score &gt; 0.8 &amp;&amp; err_lo &gt; 5%)
            </span>
          </div>
        </td>
      </tr>
    </table>
  </div>

  <div style='border-top:1px solid #bfdbfe; margin:10px 0;'></div>

  <!-- EXIT -->
  <div style='margin-bottom:12px;'>
    <div style='font-size:11px; font-weight:700; color:#1e40af; text-transform:uppercase;
         letter-spacing:0.8px; margin-bottom:6px;'>📤 Exit — depends on current regime</div>
    <table style='width:100%; border-collapse:collapse; font-size:12px; color:#1e3a8a;'>
      <tr>
        <td style='width:110px; vertical-align:top; padding:3px 10px 3px 0;'>
          <span style='background:#dcfce7; color:#166534; font-weight:700; border-radius:5px;
               padding:2px 8px; font-size:11px;'>🐂 BULL regime</span>
        </td>
        <td style='vertical-align:top; padding:3px 0;'>
          Price above MA30 <b>AND</b> MA30 is rising (5-bar slope &gt; 0)<br>
          <span style='color:#166534;'>→ Exit on <b>D3 only</b></span>
          &nbsp;— wait for structural exhaustion, let winners run
        </td>
      </tr>
      <tr><td colspan='2' style='padding:4px 0;'></td></tr>
      <tr>
        <td style='width:110px; vertical-align:top; padding:3px 10px 3px 0;'>
          <span style='background:#fee2e2; color:#991b1b; font-weight:700; border-radius:5px;
               padding:2px 8px; font-size:11px;'>🐻 BEAR / Neutral</span>
        </td>
        <td style='vertical-align:top; padding:3px 0;'>
          Everything else (below MA30, or MA30 flat/falling)<br>
          <span style='color:#991b1b;'>→ Exit on <b>D2 or D3</b></span>
          &nbsp;— cut quickly, stalls become reversals
        </td>
      </tr>
    </table>
  </div>

  <div style='border-top:1px solid #bfdbfe; margin:10px 0;'></div>

  <!-- MECHANICS -->
  <div style='font-size:11px; color:#3b5280; line-height:1.8;'>
    ⏱ <b>Execution:</b> 1-day lag — signal on bar <i>i</i>, trade fills at bar <i>i+1</i> close
    &nbsp;·&nbsp;
    💰 <b>Capital:</b> $100,000 initial
    &nbsp;·&nbsp;
    ⚠️ Data before Mar 1, 2026 is <b>in-sample</b> (CT model training + validation period)
  </div>

</div>""", unsafe_allow_html=True)


    # ── Helpers ───────────────────────────────────────────────────────
    def _cell(val, ref, fmt_fn, higher_is_better=True, na=False):
        """Coloured table cell vs a reference value."""
        if na:
            return "<td style='text-align:center; color:#94a3b8;'>—</td>"
        fv = fmt_fn(val)
        if higher_is_better:
            bg = "#dcfce7" if val > ref else ("#fee2e2" if val < ref else "#f8fafc")
        else:
            bg = "#dcfce7" if val < ref else ("#fee2e2" if val > ref else "#f8fafc")
        return (f"<td style='text-align:center; background:{bg}; "
                f"font-weight:600; padding:7px 10px;'>{fv}</td>")

    def _plain(val, fmt_fn):
        return f"<td style='text-align:center; padding:7px 10px;'>{fmt_fn(val)}</td>"

    def _tag(txt, bg, fg="#1e293b"):
        return (f"<td style='text-align:center; background:{bg}; "
                f"color:{fg}; font-weight:600; padding:7px 10px;'>{txt}</td>")

    fmt_nav   = lambda v: f"${v:,.0f}"
    fmt_pct   = lambda v: f"{v:+.1f}%"
    fmt_dd    = lambda v: f"{v:.1f}%"
    fmt_ratio = lambda v: f"{v:.2f}"
    fmt_time  = lambda v: f"{v:.0f}%"
    fmt_tax   = lambda v: f"${v:,.0f}"

    def _period_cells(s):
        """4-cell group: TF2+V-Gate | TF2+V-Gate (35% STCG) | B&H (0%) | B&H (15% LTCG)."""
        if s is None:
            return "".join(["<td style='text-align:center; color:#94a3b8;'>n/a</td>"] * 4)
        tf2  = s["final_nav"];       tax = s["after_tax_nav"];  bh  = s["final_bh"]
        r2   = s["strat_ret"];       rt  = s["after_tax_ret"];  rb  = s["bh_ret"]
        dd2  = s["max_drawdown"];    ddb = s["bh_max_dd"]
        sh2  = s["sharpe"];          shb = s["bh_sharpe"]
        tim  = s["time_in_mkt"];     txp = s["total_tax_paid"]
        wr   = s["win_rate"];        nw  = s["n_wins"];          nl  = s["n_losses"]
        ic   = s["initial_capital"]
        bh_ltcg_tax = 0.15 * max(0.0, bh - ic)
        bh_ltcg     = bh - bh_ltcg_tax
        bh_ltcg_ret = (bh_ltcg / ic - 1) * 100
        return {
            "nav":  (_cell(tf2, bh, fmt_nav) +
                     _cell(tax, bh_ltcg, fmt_nav) +
                     _plain(bh,  fmt_nav) +
                     _plain(bh_ltcg, fmt_nav)),
            "ret":  (_cell(r2, rb, fmt_pct) +
                     _cell(rt, bh_ltcg_ret, fmt_pct) +
                     _plain(rb, fmt_pct) +
                     _plain(bh_ltcg_ret, fmt_pct)),
            "dd":   (_cell(dd2, ddb, fmt_dd, higher_is_better=False) +
                     _cell(na=True, val=0, ref=0, fmt_fn=fmt_dd) +
                     _plain(ddb, fmt_dd) +
                     _cell(na=True, val=0, ref=0, fmt_fn=fmt_dd)),
            "sh":   (_cell(sh2, shb, fmt_ratio) +
                     _cell(na=True, val=0, ref=0, fmt_fn=fmt_ratio) +
                     _plain(shb, fmt_ratio) +
                     _cell(na=True, val=0, ref=0, fmt_fn=fmt_ratio)),
            "tim":  (f"<td style='text-align:center; font-weight:600; padding:7px 10px;'>"
                     f"{fmt_time(tim)}</td>" +
                     _cell(na=True, val=0, ref=0, fmt_fn=fmt_time) +
                     "<td style='text-align:center; padding:7px 10px;'>100%</td>" +
                     "<td style='text-align:center; padding:7px 10px;'>100%</td>"),
            "tax":  (_tag("0% pre-tax", "#f1f5f9") +
                     _tag("35% STCG/yr", "#fef3c7") +
                     _tag("0% unrealised", "#dcfce7") +
                     _tag("15% LTCG exit", "#ecfdf5", fg="#065f46")),
            "taxpd": (_cell(na=True, val=0, ref=0, fmt_fn=fmt_tax) +
                      _tag(f"${txp:,.0f}", "#fef3c7") +
                      _tag("$0", "#dcfce7") +
                      _tag(f"${bh_ltcg_tax:,.0f}", "#ecfdf5", fg="#065f46")),
            "wr":   (f"<td style='text-align:center; font-weight:600; padding:7px 10px;'>"
                     f"{wr:.0f}% ({nw}W/{nl}L)</td>" +
                     _cell(na=True, val=0, ref=0, fmt_fn=fmt_pct) +
                     _cell(na=True, val=0, ref=0, fmt_fn=fmt_pct) +
                     _cell(na=True, val=0, ref=0, fmt_fn=fmt_pct)),
        }

    s_r = bt_bear["stats"] if bt_bear else None
    s_f = bt_bull["stats"] if bt_bull else None

    # ── OOS stats dict (same shape as s_r / s_f so _period_cells reuses it) ──
    cutoffs    = _training_cutoffs()
    ct_cutoff  = cutoffs.get("daily H/L")
    ct_str     = ct_cutoff.strftime("%b %d, %Y") if ct_cutoff else "Feb 27, 2026"
    # Use the model's actual test_start so the OOS column covers all genuine OOS bars.
    _oos_ts    = cutoffs.get("daily H/L test_start")
    OOS_START  = _oos_ts if _oos_ts is not None else pd.Timestamp("2026-03-01")
    s_oos      = None
    bt_oos     = None
    if bt_full_oos is not None:
        _fnav = bt_full_oos["nav_series"];  _fbh = bt_full_oos["bh_series"]
        _ftr  = bt_full_oos["trades"];      ic   = bt_full_oos["stats"]["initial_capital"]
        _onav_raw = _fnav[_fnav.index >= OOS_START]
        _obh_raw  = _fbh[_fbh.index  >= OOS_START]
        if len(_onav_raw) > 1:
            _osn  = float(_fnav.asof(OOS_START)) or ic
            _osb  = float(_fbh.asof(OOS_START))  or ic
            _onav = _onav_raw / _osn * ic
            _obh  = _obh_raw  / _osb  * ic
            _of   = float(_onav.iloc[-1]);  _obf = float(_obh.iloc[-1])
            _otr  = [t for t in _ftr if pd.Timestamp(t["entry_date"]) >= OOS_START]
            _ow   = [t for t in _otr if t["pnl_pct"] > 0]
            _ol   = [t for t in _otr if t["pnl_pct"] <= 0]
            _rf   = (1.045)**(1/252) - 1
            _odr  = _onav.pct_change().fillna(0)
            _osh  = (float((_odr-_rf).mean()/(_odr-_rf).std()*np.sqrt(252))
                     if (_odr-_rf).std() > 0 else 0.0)
            _bdr  = _obh.pct_change().fillna(0)
            _bsh  = (float((_bdr-_rf).mean()/(_bdr-_rf).std()*np.sqrt(252))
                     if (_bdr-_rf).std() > 0 else 0.0)
            _opk  = _onav.cummax()
            _odd  = float(((_onav-_opk)/_opk*100).min())
            _bpk  = _obh.cummax()
            _bdd  = float(((_obh-_bpk)/_bpk*100).min())
            # _norm rescales all dollar amounts from the full-run capital base to
            # the $100k OOS baseline.  Must be applied to pnl_abs before computing
            # tax so that tax and final_nav are in the same (normalised) scale.
            _norm = ic / _osn
            _oys: dict = {}
            for t in _otr:
                yr = pd.Timestamp(t["exit_date"]).year
                _oys[yr] = _oys.get(yr, 0.0) + t["pnl_abs"] * _norm
            _otax = sum(0.35*max(0.0, v) for v in _oys.values())
            _oat  = _of - _otax
            _oday = (int(_onav.index[-1].value) - int(OOS_START.value))//86_400_000_000_000
            _odin = sum(t["duration_days"] for t in _otr)
            s_oos = dict(
                initial_capital = ic,
                final_nav       = _of,        final_bh    = _obf,
                strat_ret       = (_of/ic-1)*100,  bh_ret = (_obf/ic-1)*100,
                after_tax_nav   = _oat,       after_tax_ret = (_oat/ic-1)*100,
                alpha_abs       = _of - _obf,
                max_drawdown    = _odd,        bh_max_dd   = _bdd,
                sharpe          = _osh,        bh_sharpe   = _bsh,
                time_in_mkt     = 100*_odin/max(_oday,1),
                total_tax_paid  = _otax,
                win_rate        = 100*len(_ow)/len(_otr) if _otr else 0.0,
                n_wins          = len(_ow),   n_losses    = len(_ol),
                start_date      = OOS_START,  end_date    = _onav.index[-1],
            )
            _otr_scaled = [dict(t, exit_nav=t["exit_nav"] * _norm) for t in _otr]
            # Open position in OOS window (if any)
            _oos_open = False
            _oos_oe   = None
            if bt_full_oos.get("open_pos") and bt_full_oos.get("open_entry"):
                _oe = bt_full_oos["open_entry"]
                if pd.Timestamp(_oe["date"]) >= OOS_START:
                    _oos_open = True
                    _oos_oe   = dict(price=_oe["price"], date=_oe["date"],
                                     nav=_oe["nav"] * _norm)
            # Slice regime shading to OOS window
            _full_reg = bt_full_oos.get("bull_regime_series")
            _oos_reg  = (_full_reg[_full_reg.index >= OOS_START]
                         if _full_reg is not None else None)
            bt_oos = dict(
                nav_series         = _onav,
                bh_series          = _obh,
                open_pos           = _oos_open,
                open_entry         = _oos_oe,
                trades             = _otr_scaled,
                stats              = s_oos,
                bull_regime_series = _oos_reg,
            )

    s_full = bt_full["stats"] if bt_full else None

    pr    = _period_cells(s_r)
    pf    = _period_cells(s_f)
    p_oos = _period_cells(s_oos)
    p_full = _period_cells(s_full)

    # Period header labels
    if s_r:
        lbl_r = (f"🐻 Bear Market Performance (last 12 months) ⚠️ IS<br>"
                 f"<span style='font-size:10px; font-weight:400; opacity:0.85;'>"
                 f"{pd.Timestamp(s_r['start_date']).strftime('%b %d, %Y')} → "
                 f"{pd.Timestamp(s_r['end_date']).strftime('%b %d, %Y')}</span>")
    else:
        lbl_r = "🐻 Bear Market Performance (last 12 months) ⚠️ IS"

    if s_f:
        lbl_f = (f"🐂 Bull Market Performance ⚠️ IS<br>"
                 f"<span style='font-size:10px; font-weight:400; opacity:0.85;'>"
                 f"{pd.Timestamp(s_f['start_date']).strftime('%b %d, %Y')} → "
                 f"{pd.Timestamp(s_f['end_date']).strftime('%b %d, %Y')}</span>")
    else:
        lbl_f = "🐂 Bull Market Performance ⚠️ IS"

    if s_oos:
        lbl_oos = (f"🔬 OOS Only — Fully Blind<br>"
                   f"<span style='font-size:10px; font-weight:400; opacity:0.85;'>"
                   f"{OOS_START.strftime('%b %d, %Y')} → "
                   f"{pd.Timestamp(s_oos['end_date']).strftime('%b %d, %Y')}"
                   f"</span><br>"
                   f"<span style='font-size:9px; font-weight:400; opacity:0.75;'>"
                   f"CT model trained to {ct_str}</span>")
    else:
        lbl_oos = "🔬 OOS Only"

    if s_full:
        lbl_full = (f"🌐 Full Market Performance ⚠️ Mixed IS/OOS<br>"
                    f"<span style='font-size:10px; font-weight:400; opacity:0.85;'>"
                    f"{pd.Timestamp(s_full['start_date']).strftime('%b %d, %Y')} → "
                    f"{pd.Timestamp(s_full['end_date']).strftime('%b %d, %Y')}</span>")
    else:
        lbl_full = "🌐 Full Market Performance"

    _sub4 = ("<th style='padding:5px 8px; text-align:center;'>📊 TF2+V-Gate</th>"
             "<th style='padding:5px 8px; text-align:center;'>🧾 TF2+V-Gate (35% STCG)</th>"
             "<th style='padding:5px 8px; text-align:center;'>🏦 B&amp;H (0% unrealised)</th>"
             "<th style='padding:5px 8px; text-align:center;'>💼 B&amp;H (15% LTCG)</th>")
    sub_hdr = (
        "<tr style='background:#334155; color:white; font-size:11px;'>"
        "<th style='padding:5px 12px;'></th>"
        + _sub4 +
        "<th style='width:6px; padding:0;'></th>"
        + _sub4 +
        "<th style='width:6px; padding:0;'></th>"
        + _sub4 +
        "<th style='width:6px; padding:0;'></th>"
        + _sub4 +
        "</tr>"
    )

    metric_rows = [
        ("Final NAV ($)",          "nav"),
        ("Total Return",           "ret"),
        ("Max Drawdown",           "dd"),
        ("Sharpe Ratio (RF 4.5%)", "sh"),
        ("Time in Market",         "tim"),
        ("Tax Treatment",          "tax"),
        ("Tax Paid",               "taxpd"),
        ("Win Rate",               "wr"),
    ]

    tbody = ""
    _sep = "<td style='width:6px; padding:0; border-left:3px solid #cbd5e1;'></td>"
    for i, (lbl, key) in enumerate(metric_rows):
        bg = "#f8fafc" if i % 2 == 0 else "#ffffff"
        tbody += (
            f"<tr style='background:{bg};'>"
            f"<td style='padding:7px 12px; font-weight:500; white-space:nowrap; "
            f"color:#334155;'>{lbl}</td>"
            f"{pr[key]}{_sep}{pf[key]}{_sep}{p_oos[key]}{_sep}{p_full[key]}"
            f"</tr>"
        )

    st.markdown(
        f"""
        <div style="overflow-x:auto; margin:10px 0 6px 0;">
        <table style="width:100%; border-collapse:collapse; font-size:13px;
                      border:1px solid #e2e8f0; border-radius:8px; overflow:hidden;">
          <thead>
            <tr style="background:#1e3a8a; color:white;">
              <th style="padding:10px 12px; text-align:left; font-weight:600; min-width:160px;">
                Metric</th>
              <th colspan="4" style="padding:10px 8px; text-align:center; font-weight:600;
                  border-right:3px solid #4c72b5;">
                {lbl_r}</th>
              <th style="width:6px; padding:0; background:#1e3a8a;"></th>
              <th colspan="4" style="padding:10px 8px; text-align:center; font-weight:600;
                  border-right:3px solid #4c72b5;">
                {lbl_f}</th>
              <th style="width:6px; padding:0; background:#1e3a8a;"></th>
              <th colspan="4" style="padding:10px 8px; text-align:center; font-weight:600;
                  background:#14532d; border-right:3px solid #166534;">
                {lbl_oos}</th>
              <th style="width:6px; padding:0; background:#1e3a8a;"></th>
              <th colspan="4" style="padding:10px 8px; text-align:center; font-weight:600;
                  background:#581c87;">
                {lbl_full}</th>
            </tr>
            {sub_hdr}
          </thead>
          <tbody>
            {tbody}
          </tbody>
        </table>
        </div>
        <p style="font-size:11px; color:#64748b; margin:2px 0 14px 0;">
        🟢 Green = better for investor vs B&amp;H benchmark &nbsp;|&nbsp;
        🔴 Red = worse &nbsp;|&nbsp;
        💡 Col 1 vs B&amp;H (0%); Col 2 (35% STCG/yr) vs B&amp;H (15% LTCG) — fair after-tax comparison.
        💼 B&amp;H 15% LTCG: single tax event at period end on total gain.
        ⚠️ Pre-Sep 2025 dates are <b>in-sample</b>.
        🔬 OOS: NAV normalised to $100k at {OOS_START.strftime("%b %d, %Y")};
        CT model last trained {ct_str}.
        🌐 Full Market: Jun 2024–today (mixed IS + OOS).
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ── Open-position badge (from bear period — most recent) ─────────────────────
    if bt_bear and bt_bear["open_pos"] and bt_bear["open_entry"]:
        oe  = bt_bear["open_entry"]
        unr = (float(bt_bear["nav_series"].iloc[-1]) / float(oe["nav"]) - 1) * 100
        oe_col = "#16a34a" if unr >= 0 else "#dc2626"
        st.markdown(
            f"<div style='background:#f0fdf4; border:1px solid #16a34a; border-radius:8px; "
            f"padding:8px 14px; margin:0 0 10px 0; font-size:13px; color:#14532d;'>"
            f"📍 <b>Open position</b> — bought "
            f"{pd.Timestamp(oe['date']).strftime('%b %d, %Y')} "
            f"@ ${oe['price']:,.0f} · "
            f"unrealized: <span style='color:{oe_col}; font-weight:700;'>"
            f"{unr:+.1f}%</span></div>",
            unsafe_allow_html=True,
        )

    # ── Charts in tabs ────────────────────────────────────────────────
    def _make_chart(bt, title):
        """Return a Plotly figure for one backtest result."""
        if bt is None:
            return None
        s   = bt["stats"]
        nav_s = bt["nav_series"]
        bh_s  = bt["bh_series"]
        has_p = bt["open_pos"]

        fig = go.Figure()

        # ── Regime background shading (bull=green, bear/neutral=red) ──────────
        regime = bt.get("bull_regime_series")
        if regime is not None and len(regime) > 0:
            arr     = regime.values.astype(bool)
            idx     = regime.index
            changes = np.where(np.diff(arr.astype(int)) != 0)[0] + 1
            starts  = np.concatenate([[0], changes])
            ends    = np.concatenate([changes, [len(arr)]])
            bull_leg = bear_leg = False
            for s_i, e_i in zip(starts, ends):
                is_bull  = bool(arr[s_i])
                color    = "rgba(34,197,94,0.10)" if is_bull else "rgba(239,68,68,0.07)"
                lname    = "🐂 Bull Regime" if is_bull else "🐻 Bear/Neutral"
                show_leg = (is_bull and not bull_leg) or (not is_bull and not bear_leg)
                e_idx    = min(e_i, len(idx) - 1)
                fig.add_vrect(
                    x0=idx[s_i], x1=idx[e_idx],
                    fillcolor=color, line_width=0,
                    name=lname, showlegend=show_leg,
                    legendgrouptitle_text=None,
                )
                if is_bull: bull_leg = True
                else:       bear_leg = True

        fig.add_trace(go.Scatter(
            x=bh_s.index, y=bh_s.values, name="Buy & Hold",
            line=dict(color="#94a3b8", width=1.5, dash="dot"),
            hovertemplate="%{x|%b %d, %Y}: $%{y:,.0f}<extra>Buy & Hold</extra>",
        ))
        fig.add_trace(go.Scatter(
            x=nav_s.index, y=nav_s.values, name="TF2+V-Gate Strategy",
            line=dict(color="#2563eb", width=2.5),
            hovertemplate="%{x|%b %d, %Y}: $%{y:,.0f}<extra>TF2+V-Gate Strategy</extra>",
        ))
        fig.add_hline(
            y=s["initial_capital"], line_dash="dash",
            line_color="#64748b", line_width=1, opacity=0.4,
            annotation_text="  $100k start", annotation_position="bottom right",
        )
        for t in bt["trades"]:
            entry_y = float(nav_s.asof(t["entry_date"])) if t["entry_date"] >= nav_s.index[0] else s["initial_capital"]
            exit_y  = float(t["exit_nav"])
            win_col = "#16a34a" if t["pnl_pct"] > 0 else "#dc2626"
            fig.add_trace(go.Scatter(
                x=[t["entry_date"]], y=[entry_y], mode="markers",
                marker=dict(symbol="triangle-up", size=13, color="#16a34a",
                            line=dict(width=1.5, color="white")),
                showlegend=False,
                hovertemplate=(
                    f"<b>BUY</b> {pd.Timestamp(t['entry_date']).strftime('%b %d')}"
                    f" @ ${t['entry_price']:,.0f}<br>{t['entry_trigger']}<extra></extra>"
                ),
            ))
            fig.add_trace(go.Scatter(
                x=[t["exit_date"]], y=[exit_y], mode="markers",
                marker=dict(symbol="triangle-down", size=13, color=win_col,
                            line=dict(width=1.5, color="white")),
                showlegend=False,
                hovertemplate=(
                    f"<b>SELL</b> {pd.Timestamp(t['exit_date']).strftime('%b %d')}"
                    f" @ ${t['exit_price']:,.0f} "
                    f"({t['pnl_pct']:+.1f}%) — {t['exit_signal']}<extra></extra>"
                ),
            ))
        if has_p and bt["open_entry"]:
            oe   = bt["open_entry"]
            oe_y = float(nav_s.asof(oe["date"])) if oe["date"] >= nav_s.index[0] else s["initial_capital"]
            fig.add_trace(go.Scatter(
                x=[oe["date"]], y=[oe_y], mode="markers",
                marker=dict(symbol="triangle-up", size=13, color="#f59e0b",
                            line=dict(width=1.5, color="white")),
                showlegend=False,
                hovertemplate=(
                    f"<b>BUY (OPEN)</b> {pd.Timestamp(oe['date']).strftime('%b %d')}"
                    f" @ ${oe['price']:,.0f}<extra></extra>"
                ),
            ))
        fig.update_layout(
            title=dict(text=title, font=dict(size=13), x=0, xanchor="left"),
            xaxis_title=None,
            yaxis_title="Portfolio Value ($)",
            yaxis_tickprefix="$", yaxis_tickformat=",.0f",
            height=360,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=36, b=0),
            hovermode="x unified",
            plot_bgcolor="#f8fafc", paper_bgcolor="#ffffff",
            xaxis=dict(showgrid=True, gridcolor="#e2e8f0", zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="#e2e8f0", zeroline=False),
        )
        return fig

    tab_labels = []
    if bt_bear:
        sr = bt_bear["stats"]
        tab_labels.append(
            f"🐻 Bear Market  "
            f"({pd.Timestamp(sr['start_date']).strftime('%b %Y')} → "
            f"{pd.Timestamp(sr['end_date']).strftime('%b %Y')})"
        )
    if bt_bull:
        sf = bt_bull["stats"]
        tab_labels.append(
            f"🐂 Bull Market  "
            f"({pd.Timestamp(sf['start_date']).strftime('%b %Y')} → "
            f"{pd.Timestamp(sf['end_date']).strftime('%b %Y')})"
        )
    if bt_oos:
        so = bt_oos["stats"]
        tab_labels.append(
            f"🔬 OOS Only  "
            f"({pd.Timestamp(so['start_date']).strftime('%b %Y')} → "
            f"{pd.Timestamp(so['end_date']).strftime('%b %Y')})"
        )
    if bt_full:
        sfl = bt_full["stats"]
        tab_labels.append(
            f"🌐 Full Market  "
            f"({pd.Timestamp(sfl['start_date']).strftime('%b %Y')} → "
            f"{pd.Timestamp(sfl['end_date']).strftime('%b %Y')})"
        )

    if tab_labels:
        tabs = st.tabs(tab_labels)
        tab_idx = 0
        if bt_bear:
            with tabs[tab_idx]:
                sr = bt_bear["stats"]
                fig_r = _make_chart(
                    bt_bear,
                    f"TF2+V-Gate vs B&H — Bear Market Performance  "
                    f"({pd.Timestamp(sr['start_date']).strftime('%b %d, %Y')} → "
                    f"{pd.Timestamp(sr['end_date']).strftime('%b %d, %Y')})"
                )
                if fig_r:
                    st.plotly_chart(fig_r, use_container_width=True, key=f"chart_bear_{key_suffix}")
            tab_idx += 1
        if bt_bull:
            with tabs[tab_idx]:
                sf = bt_bull["stats"]
                fig_f = _make_chart(
                    bt_bull,
                    f"TF2+V-Gate vs B&H — Bull Market Performance  "
                    f"({pd.Timestamp(sf['start_date']).strftime('%b %d, %Y')} → "
                    f"{pd.Timestamp(sf['end_date']).strftime('%b %d, %Y')})"
                )
                if fig_f:
                    st.plotly_chart(fig_f, use_container_width=True, key=f"chart_bull_{key_suffix}")
            tab_idx += 1
        if bt_oos:
            with tabs[tab_idx]:
                so = bt_oos["stats"]
                fig_o = _make_chart(
                    bt_oos,
                    f"TF2+V-Gate vs B&H — OOS Period (Fully Blind)  "
                    f"({pd.Timestamp(so['start_date']).strftime('%b %d, %Y')} → "
                    f"{pd.Timestamp(so['end_date']).strftime('%b %d, %Y')})"
                )
                if fig_o:
                    st.plotly_chart(fig_o, use_container_width=True, key=f"chart_oos_{key_suffix}")
            tab_idx += 1
        if bt_full:
            with tabs[tab_idx]:
                sfl = bt_full["stats"]
                fig_fl = _make_chart(
                    bt_full,
                    f"TF2+V-Gate vs B&H — Full Market Performance  "
                    f"({pd.Timestamp(sfl['start_date']).strftime('%b %d, %Y')} → "
                    f"{pd.Timestamp(sfl['end_date']).strftime('%b %d, %Y')})"
                )
                if fig_fl:
                    st.plotly_chart(fig_fl, use_container_width=True, key=f"chart_full_{key_suffix}")

    # ── Trade log ─────────────────────────────────────────────────────
    def _trade_log_rows(bt):
        if bt is None:
            return [], False, None
        s     = bt["stats"]
        has_p = bt["open_pos"]
        rows  = []
        for i, t in enumerate(bt["trades"], 1):
            rows.append({
                "#":           i,
                "Entry":       pd.Timestamp(t["entry_date"]).strftime("%b %d, %Y"),
                "Buy @":       f"${t['entry_price']:,.0f}",
                "Trigger":     t["entry_trigger"],
                "Exit":        pd.Timestamp(t["exit_date"]).strftime("%b %d, %Y"),
                "Sell @":      f"${t['exit_price']:,.0f}",
                "P&L":         f"{t['pnl_pct']:+.1f}%",
                "Result":      "✓ WIN" if t["pnl_pct"] > 0 else "✗ LOSS",
                "Days":        t["duration_days"],
                "Exit Signal": t["exit_signal"],
                "NAV After":   f"${t['exit_nav']:,.0f}",
            })
        if has_p and bt["open_entry"]:
            oe = bt["open_entry"]
            unr_pct = (float(bt["nav_series"].iloc[-1]) / float(oe["nav"]) - 1) * 100
            rows.append({
                "#":           len(bt["trades"]) + 1,
                "Entry":       pd.Timestamp(oe["date"]).strftime("%b %d, %Y"),
                "Buy @":       f"${oe['price']:,.0f}",
                "Trigger":     "—",
                "Exit":        "⏳ OPEN",
                "Sell @":      "—",
                "P&L":         f"{unr_pct:+.1f}% (unrlzd)",
                "Result":      "🟡 OPEN",
                "Days":        (pd.Timestamp.now() - pd.Timestamp(oe["date"])).days,
                "Exit Signal": "—",
                "NAV After":   "—",
            })
        return rows, has_p, s

    log_tabs = []
    if bt_bear: log_tabs.append("📋 Bear Market Trade Log")
    if bt_bull: log_tabs.append("📋 Bull Market Trade Log")
    if bt_oos:  log_tabs.append("📋 OOS Trade Log")
    if bt_full: log_tabs.append("📋 Full Market Trade Log")

    if log_tabs:
        ltabs = st.tabs(log_tabs)
        lt_idx = 0
        for bt_src, lbl in [(bt_bear, "bear"), (bt_bull, "bull"),
                             (bt_oos, "oos"), (bt_full, "full")]:
            if bt_src is None:
                continue
            rows, has_p, s = _trade_log_rows(bt_src)
            n_closed = len(bt_src["trades"])
            header   = f"{n_closed} closed trades" + (" + 1 open" if has_p else "")
            with ltabs[lt_idx]:
                if rows:
                    st.dataframe(pd.DataFrame(rows), hide_index=True,
                                 use_container_width=True)
                else:
                    st.info("No trades in this period — strategy was in cash throughout.")
                caption = (
                    "💡 P&L at execution prices (1-day lag from signal). "
                    "B&H normalised to $100k at period start. "
                )
                if lbl == "oos":
                    caption += "✅ Fully OOS — model never saw this data."
                elif lbl == "full":
                    caption += "⚠️ Mixed IS/OOS — pre-Sep 2025 trades are in-sample."
                else:
                    caption += "⚠️ pre-Sep 2025 = in-sample."
                st.caption(caption)
            lt_idx += 1


def render_mstr_trading_strategy_dashboard(bt_bear, bt_bull, bt_full_oos=None,
                                           bt_full=None, key_suffix: str = "") -> None:
    """Render the MSTR backtesting dashboard (BTC TF2+V-Gate signals → MSTR execution).

    Same four-period layout as render_trading_strategy_dashboard but with:
      • Purple color scheme to distinguish from BTC results
      • MSTR stock prices in trade log (split-adjusted)
      • B&H comparison uses MSTR rather than BTC
    """
    st.markdown("---")
    st.subheader("📊 MSTR — BTC Signal-Driven Backtest (TF2 + V-Gate)")

    if bt_bear is None and bt_bull is None:
        st.info("⚙️ MSTR backtest computing … fetching MSTR data and building signals. "
                "Refresh in a moment.")
        return

    # ── Strategy rules card ───────────────────────────────────────────────────
    st.markdown("""
<div style='background:#f5f3ff; border:2px solid #7c3aed; border-radius:12px;
     padding:16px 20px; margin:4px 0 16px 0; font-family:sans-serif;'>

  <div style='font-size:14px; font-weight:700; color:#4c1d95; margin-bottom:14px;
       letter-spacing:0.3px;'>
    📊 TF2 + V-Gate on MSTR &nbsp;—&nbsp; BTC Signals · MSTR Execution
  </div>

  <div style='background:#ede9fe; border-radius:8px; padding:10px 14px;
       margin-bottom:12px; font-size:12px; color:#4c1d95; font-weight:600;'>
    🔁 <b>Core idea:</b> Use Bitcoin's trend signatures (U1, D2, D3, V-Gate) as the signal
    engine, but buy and sell <b>MSTR stock</b> instead of BTC. MicroStrategy holds
    ~580,000 BTC on its balance sheet, making it a leveraged Bitcoin proxy with
    equity-market liquidity and tax treatment as a security.
  </div>

  <!-- ENTRY -->
  <div style='margin-bottom:12px;'>
    <div style='font-size:11px; font-weight:700; color:#5b21b6; text-transform:uppercase;
         letter-spacing:0.8px; margin-bottom:6px;'>📥 Entry — BTC signals trigger MSTR buy</div>
    <table style='width:100%; border-collapse:collapse; font-size:12px; color:#4c1d95;'>
      <tr>
        <td style='width:36px; vertical-align:top; padding:3px 6px 3px 0; font-weight:700;
             color:#7c3aed;'>①</td>
        <td style='vertical-align:top; padding:3px 0;'>
          <b>U1 signal active on BTC</b> — BTC's predicted highs consistently beaten<br>
          <span style='color:#7c3aed; font-size:11px;'>
            err_hi_ma3 &gt; +0.5% &nbsp;&amp;&amp;&nbsp; hi_breaks_3d ≥ 2
          </span>
        </td>
      </tr>
      <tr>
        <td style='width:36px; vertical-align:top; padding:3px 6px 3px 0; font-weight:700;
             color:#7c3aed;'>②</td>
        <td style='vertical-align:top; padding:3px 0;'>
          <b>At least one BTC trend gate passes</b>:
          <div style='margin:5px 0 0 4px; line-height:2.1;'>
            <span style='background:#ede9fe; border-radius:4px; padding:1px 7px;'>↑ MA30</span>
            &nbsp; BTC close above its 30-day moving average
            <br>
            <span style='background:#ede9fe; border-radius:4px; padding:1px 7px;'>Clean 10d</span>
            &nbsp; No D1 or D2 on BTC in prior 10 bars
            <br>
            <span style='background:#ddd6fe; border-radius:4px; padding:1px 7px;'>⚡ V-reversal</span>
            &nbsp; BTC capitulation spike within last 3 bars
            <span style='color:#7c3aed; font-size:11px;'>(dn_score &gt; 0.8 &amp;&amp; err_lo &gt; 5%)</span>
          </div>
        </td>
      </tr>
    </table>
  </div>

  <div style='border-top:1px solid #c4b5fd; margin:10px 0;'></div>

  <!-- EXIT -->
  <div style='margin-bottom:12px;'>
    <div style='font-size:11px; font-weight:700; color:#5b21b6; text-transform:uppercase;
         letter-spacing:0.8px; margin-bottom:6px;'>📤 Exit — BTC signals trigger MSTR sell</div>
    <table style='width:100%; border-collapse:collapse; font-size:12px; color:#4c1d95;'>
      <tr>
        <td style='width:110px; vertical-align:top; padding:3px 10px 3px 0;'>
          <span style='background:#dcfce7; color:#166534; font-weight:700; border-radius:5px;
               padding:2px 8px; font-size:11px;'>🐂 BULL regime</span>
        </td>
        <td style='vertical-align:top; padding:3px 0;'>
          BTC above MA30 AND MA30 rising → exit MSTR on <b>D3 only</b> (patient)
        </td>
      </tr>
      <tr><td colspan='2' style='padding:4px 0;'></td></tr>
      <tr>
        <td style='width:110px; vertical-align:top; padding:3px 10px 3px 0;'>
          <span style='background:#fee2e2; color:#991b1b; font-weight:700; border-radius:5px;
               padding:2px 8px; font-size:11px;'>🐻 BEAR / Neutral</span>
        </td>
        <td style='vertical-align:top; padding:3px 0;'>
          BTC below MA30 or MA30 flat → exit MSTR on <b>D2 or D3</b> (defensive)
        </td>
      </tr>
    </table>
  </div>

  <div style='border-top:1px solid #c4b5fd; margin:10px 0;'></div>

  <div style='font-size:11px; color:#5b21b6; line-height:1.8;'>
    ⏱ <b>Execution:</b> 1-day lag — BTC signal on bar <i>i</i>, MSTR trade at bar <i>i+1</i> close
    &nbsp;·&nbsp;
    💰 <b>Capital:</b> $100,000 initial
    &nbsp;·&nbsp;
    📅 <b>MSTR prices:</b> split-adjusted (10-for-1 split Aug 2024)
    &nbsp;·&nbsp;
    ⚠️ Pre-Mar 2026 data is <b>in-sample</b> for the BTC CT model
  </div>

</div>""", unsafe_allow_html=True)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _cell(val, ref, fmt_fn, higher_is_better=True, na=False):
        if na:
            return "<td style='text-align:center; color:#94a3b8;'>—</td>"
        fv = fmt_fn(val)
        if higher_is_better:
            bg = "#dcfce7" if val > ref else ("#fee2e2" if val < ref else "#f8fafc")
        else:
            bg = "#dcfce7" if val < ref else ("#fee2e2" if val > ref else "#f8fafc")
        return (f"<td style='text-align:center; background:{bg}; "
                f"font-weight:600; padding:7px 10px;'>{fv}</td>")

    def _plain(val, fmt_fn):
        return f"<td style='text-align:center; padding:7px 10px;'>{fmt_fn(val)}</td>"

    def _tag(txt, bg, fg="#1e293b"):
        return (f"<td style='text-align:center; background:{bg}; "
                f"color:{fg}; font-weight:600; padding:7px 10px;'>{txt}</td>")

    fmt_nav   = lambda v: f"${v:,.0f}"
    fmt_pct   = lambda v: f"{v:+.1f}%"
    fmt_dd    = lambda v: f"{v:.1f}%"
    fmt_ratio = lambda v: f"{v:.2f}"
    fmt_time  = lambda v: f"{v:.0f}%"
    fmt_tax   = lambda v: f"${v:,.0f}"

    def _period_cells(s):
        """4-cell group: TF2+V-Gate | TF2+V-Gate (35% STCG) | B&H (0%) | B&H (15% LTCG)."""
        if s is None:
            return "".join(["<td style='text-align:center; color:#94a3b8;'>n/a</td>"] * 4)
        tf2  = s["final_nav"];       tax = s["after_tax_nav"];  bh  = s["final_bh"]
        r2   = s["strat_ret"];       rt  = s["after_tax_ret"];  rb  = s["bh_ret"]
        dd2  = s["max_drawdown"];    ddb = s["bh_max_dd"]
        sh2  = s["sharpe"];          shb = s["bh_sharpe"]
        tim  = s["time_in_mkt"];     txp = s["total_tax_paid"]
        wr   = s["win_rate"];        nw  = s["n_wins"];          nl  = s["n_losses"]
        ic   = s["initial_capital"]
        bh_ltcg_tax = 0.15 * max(0.0, bh - ic)
        bh_ltcg     = bh - bh_ltcg_tax
        bh_ltcg_ret = (bh_ltcg / ic - 1) * 100
        return {
            "nav":  (_cell(tf2, bh, fmt_nav) +
                     _cell(tax, bh_ltcg, fmt_nav) +
                     _plain(bh,  fmt_nav) +
                     _plain(bh_ltcg, fmt_nav)),
            "ret":  (_cell(r2, rb, fmt_pct) +
                     _cell(rt, bh_ltcg_ret, fmt_pct) +
                     _plain(rb, fmt_pct) +
                     _plain(bh_ltcg_ret, fmt_pct)),
            "dd":   (_cell(dd2, ddb, fmt_dd, higher_is_better=False) +
                     _cell(na=True, val=0, ref=0, fmt_fn=fmt_dd) +
                     _plain(ddb, fmt_dd) +
                     _cell(na=True, val=0, ref=0, fmt_fn=fmt_dd)),
            "sh":   (_cell(sh2, shb, fmt_ratio) +
                     _cell(na=True, val=0, ref=0, fmt_fn=fmt_ratio) +
                     _plain(shb, fmt_ratio) +
                     _cell(na=True, val=0, ref=0, fmt_fn=fmt_ratio)),
            "tim":  (f"<td style='text-align:center; font-weight:600; padding:7px 10px;'>"
                     f"{fmt_time(tim)}</td>" +
                     _cell(na=True, val=0, ref=0, fmt_fn=fmt_time) +
                     "<td style='text-align:center; padding:7px 10px;'>100%</td>" +
                     "<td style='text-align:center; padding:7px 10px;'>100%</td>"),
            "tax":  (_tag("0% pre-tax", "#f1f5f9") +
                     _tag("35% STCG/yr", "#fef3c7") +
                     _tag("0% unrealised", "#dcfce7") +
                     _tag("15% LTCG exit", "#ecfdf5", fg="#065f46")),
            "taxpd": (_cell(na=True, val=0, ref=0, fmt_fn=fmt_tax) +
                      _tag(f"${txp:,.0f}", "#fef3c7") +
                      _tag("$0", "#dcfce7") +
                      _tag(f"${bh_ltcg_tax:,.0f}", "#ecfdf5", fg="#065f46")),
            "wr":   (f"<td style='text-align:center; font-weight:600; padding:7px 10px;'>"
                     f"{wr:.0f}% ({nw}W/{nl}L)</td>" +
                     _cell(na=True, val=0, ref=0, fmt_fn=fmt_pct) +
                     _cell(na=True, val=0, ref=0, fmt_fn=fmt_pct) +
                     _cell(na=True, val=0, ref=0, fmt_fn=fmt_pct)),
        }

    s_r = bt_bear["stats"] if bt_bear else None
    s_f = bt_bull["stats"] if bt_bull else None

    cutoffs   = _training_cutoffs()
    ct_cutoff = cutoffs.get("daily H/L")
    ct_str    = ct_cutoff.strftime("%b %d, %Y") if ct_cutoff else "Feb 27, 2026"
    _oos_ts   = cutoffs.get("daily H/L test_start")
    OOS_START = _oos_ts if _oos_ts is not None else pd.Timestamp("2026-03-01")
    s_oos = None; bt_oos = None

    if bt_full_oos is not None:
        _fnav = bt_full_oos["nav_series"]; _fbh = bt_full_oos["bh_series"]
        _ftr  = bt_full_oos["trades"];     ic   = bt_full_oos["stats"]["initial_capital"]
        _onav_raw = _fnav[_fnav.index >= OOS_START]
        _obh_raw  = _fbh[_fbh.index  >= OOS_START]
        if len(_onav_raw) > 1:
            _osn  = float(_fnav.asof(OOS_START)) or ic
            _osb  = float(_fbh.asof(OOS_START))  or ic
            _onav = _onav_raw / _osn * ic
            _obh  = _obh_raw  / _osb  * ic
            _of   = float(_onav.iloc[-1]);  _obf = float(_obh.iloc[-1])
            _otr  = [t for t in _ftr if pd.Timestamp(t["entry_date"]) >= OOS_START]
            _ow   = [t for t in _otr if t["pnl_pct"] > 0]
            _ol   = [t for t in _otr if t["pnl_pct"] <= 0]
            _rf   = (1.045)**(1/252) - 1
            _odr  = _onav.pct_change().fillna(0)
            _osh  = (float((_odr-_rf).mean()/(_odr-_rf).std()*np.sqrt(252))
                     if (_odr-_rf).std() > 0 else 0.0)
            _bdr  = _obh.pct_change().fillna(0)
            _bsh  = (float((_bdr-_rf).mean()/(_bdr-_rf).std()*np.sqrt(252))
                     if (_bdr-_rf).std() > 0 else 0.0)
            _opk  = _onav.cummax()
            _odd  = float(((_onav-_opk)/_opk*100).min())
            _bpk  = _obh.cummax()
            _bdd  = float(((_obh-_bpk)/_bpk*100).min())
            # _norm rescales all dollar amounts from the full-run capital base to
            # the $100k OOS baseline.  Must be applied to pnl_abs before computing
            # tax so that tax and final_nav are in the same (normalised) scale.
            _norm = ic / _osn
            _oys: dict = {}
            for t in _otr:
                yr = pd.Timestamp(t["exit_date"]).year
                _oys[yr] = _oys.get(yr, 0.0) + t["pnl_abs"] * _norm
            _otax = sum(0.35*max(0.0, v) for v in _oys.values())
            _oat  = _of - _otax
            _oday = (int(_onav.index[-1].value) - int(OOS_START.value))//86_400_000_000_000
            _odin = sum(t["duration_days"] for t in _otr)
            s_oos = dict(
                initial_capital = ic,
                final_nav=_of,   final_bh=_obf,
                strat_ret=(_of/ic-1)*100,   bh_ret=(_obf/ic-1)*100,
                after_tax_nav=_oat, after_tax_ret=(_oat/ic-1)*100,
                alpha_abs=_of - _obf,
                max_drawdown=_odd, bh_max_dd=_bdd,
                sharpe=_osh, bh_sharpe=_bsh,
                time_in_mkt=100*_odin/max(_oday, 1),
                total_tax_paid=_otax,
                win_rate=100*len(_ow)/len(_otr) if _otr else 0.0,
                n_wins=len(_ow), n_losses=len(_ol),
                start_date=OOS_START, end_date=_onav.index[-1],
            )
            _otr_scaled = [dict(t, exit_nav=t["exit_nav"]*_norm) for t in _otr]
            _oos_open = False; _oos_oe = None
            if bt_full_oos.get("open_pos") and bt_full_oos.get("open_entry"):
                _oe = bt_full_oos["open_entry"]
                if pd.Timestamp(_oe["date"]) >= OOS_START:
                    _oos_open = True
                    _oos_oe   = dict(price=_oe["price"], date=_oe["date"],
                                     nav=_oe["nav"] * _norm)
            _full_reg = bt_full_oos.get("bull_regime_series")
            _oos_reg  = (_full_reg[_full_reg.index >= OOS_START]
                         if _full_reg is not None else None)
            bt_oos = dict(
                nav_series=_onav, bh_series=_obh,
                open_pos=_oos_open, open_entry=_oos_oe,
                trades=_otr_scaled, stats=s_oos,
                bull_regime_series=_oos_reg,
            )

    s_full = bt_full["stats"] if bt_full else None

    pr     = _period_cells(s_r)
    pf     = _period_cells(s_f)
    p_oos  = _period_cells(s_oos)
    p_full = _period_cells(s_full)

    if s_r:
        lbl_r = (f"🐻 Bear Market (last 12 months) ⚠️ IS<br>"
                 f"<span style='font-size:10px; font-weight:400; opacity:0.85;'>"
                 f"{pd.Timestamp(s_r['start_date']).strftime('%b %d, %Y')} → "
                 f"{pd.Timestamp(s_r['end_date']).strftime('%b %d, %Y')}</span>")
    else:
        lbl_r = "🐻 Bear Market (last 12 months) ⚠️ IS"

    if s_f:
        lbl_f = (f"🐂 Bull Market ⚠️ IS<br>"
                 f"<span style='font-size:10px; font-weight:400; opacity:0.85;'>"
                 f"{pd.Timestamp(s_f['start_date']).strftime('%b %d, %Y')} → "
                 f"{pd.Timestamp(s_f['end_date']).strftime('%b %d, %Y')}</span>")
    else:
        lbl_f = "🐂 Bull Market ⚠️ IS"

    if s_oos:
        lbl_oos = (f"🔬 OOS Only — Fully Blind<br>"
                   f"<span style='font-size:10px; font-weight:400; opacity:0.85;'>"
                   f"{OOS_START.strftime('%b %d, %Y')} → "
                   f"{pd.Timestamp(s_oos['end_date']).strftime('%b %d, %Y')}"
                   f"</span><br>"
                   f"<span style='font-size:9px; font-weight:400; opacity:0.75;'>"
                   f"CT model trained to {ct_str}</span>")
    else:
        lbl_oos = "🔬 OOS Only"

    if s_full:
        lbl_full = (f"🌐 Full Market ⚠️ Mixed IS/OOS<br>"
                    f"<span style='font-size:10px; font-weight:400; opacity:0.85;'>"
                    f"{pd.Timestamp(s_full['start_date']).strftime('%b %d, %Y')} → "
                    f"{pd.Timestamp(s_full['end_date']).strftime('%b %d, %Y')}</span>")
    else:
        lbl_full = "🌐 Full Market"

    _sub4 = ("<th style='padding:5px 8px; text-align:center;'>📊 TF2+V-Gate (MSTR)</th>"
             "<th style='padding:5px 8px; text-align:center;'>🧾 TF2+V-Gate (35% STCG)</th>"
             "<th style='padding:5px 8px; text-align:center;'>🏦 B&amp;H MSTR (0% unrealised)</th>"
             "<th style='padding:5px 8px; text-align:center;'>💼 B&amp;H MSTR (15% LTCG)</th>")
    sub_hdr = (
        "<tr style='background:#334155; color:white; font-size:11px;'>"
        "<th style='padding:5px 12px;'></th>"
        + _sub4
        + "<th style='width:6px; padding:0;'></th>"
        + _sub4
        + "<th style='width:6px; padding:0;'></th>"
        + _sub4
        + "<th style='width:6px; padding:0;'></th>"
        + _sub4
        + "</tr>"
    )

    metric_rows = [
        ("Final NAV ($)",          "nav"),
        ("Total Return",           "ret"),
        ("Max Drawdown",           "dd"),
        ("Sharpe Ratio (RF 4.5%)", "sh"),
        ("Time in Market",         "tim"),
        ("Tax Treatment",          "tax"),
        ("Tax Paid",               "taxpd"),
        ("Win Rate",               "wr"),
    ]

    tbody = ""
    _sep = "<td style='width:6px; padding:0; border-left:3px solid #cbd5e1;'></td>"
    for i, (lbl, key) in enumerate(metric_rows):
        bg = "#f8fafc" if i % 2 == 0 else "#ffffff"
        tbody += (
            f"<tr style='background:{bg};'>"
            f"<td style='padding:7px 12px; font-weight:500; white-space:nowrap; "
            f"color:#334155;'>{lbl}</td>"
            f"{pr[key]}{_sep}{pf[key]}{_sep}{p_oos[key]}{_sep}{p_full[key]}"
            f"</tr>"
        )

    st.markdown(
        f"""
        <div style="overflow-x:auto; margin:10px 0 6px 0;">
        <table style="width:100%; border-collapse:collapse; font-size:13px;
                      border:1px solid #e2e8f0; border-radius:8px; overflow:hidden;">
          <thead>
            <tr style="background:#4c1d95; color:white;">
              <th style="padding:10px 12px; text-align:left; font-weight:600; min-width:160px;">
                Metric</th>
              <th colspan="4" style="padding:10px 8px; text-align:center; font-weight:600;
                  border-right:3px solid #7c3aed;">
                {lbl_r}</th>
              <th style="width:6px; padding:0; background:#4c1d95;"></th>
              <th colspan="4" style="padding:10px 8px; text-align:center; font-weight:600;
                  border-right:3px solid #7c3aed;">
                {lbl_f}</th>
              <th style="width:6px; padding:0; background:#4c1d95;"></th>
              <th colspan="4" style="padding:10px 8px; text-align:center; font-weight:600;
                  background:#14532d; border-right:3px solid #166534;">
                {lbl_oos}</th>
              <th style="width:6px; padding:0; background:#4c1d95;"></th>
              <th colspan="4" style="padding:10px 8px; text-align:center; font-weight:600;
                  background:#581c87;">
                {lbl_full}</th>
            </tr>
            {sub_hdr}
          </thead>
          <tbody>
            {tbody}
          </tbody>
        </table>
        </div>
        <p style="font-size:11px; color:#64748b; margin:2px 0 14px 0;">
        🟢 Green = better for investor vs B&amp;H benchmark &nbsp;|&nbsp;
        🔴 Red = worse &nbsp;|&nbsp;
        💡 Col 1 vs B&amp;H (0%); Col 2 (35% STCG/yr) vs B&amp;H (15% LTCG) — fair after-tax comparison.
        💼 B&amp;H 15% LTCG: single tax event at period end on total gain.
        ⚠️ Pre-Mar 2026 dates are <b>in-sample</b> for the BTC CT model.
        🔬 OOS: NAV normalised to $100k at {OOS_START.strftime("%b %d, %Y")};
        CT model last trained {ct_str}.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ── Open-position badge ───────────────────────────────────────────────────
    if bt_bear and bt_bear["open_pos"] and bt_bear["open_entry"]:
        oe  = bt_bear["open_entry"]
        unr = (float(bt_bear["nav_series"].iloc[-1]) / float(oe["nav"]) - 1) * 100
        oe_col = "#16a34a" if unr >= 0 else "#dc2626"
        st.markdown(
            f"<div style='background:#f5f3ff; border:1px solid #7c3aed; border-radius:8px; "
            f"padding:8px 14px; margin:0 0 10px 0; font-size:13px; color:#4c1d95;'>"
            f"📍 <b>Open MSTR position</b> — bought "
            f"{pd.Timestamp(oe['date']).strftime('%b %d, %Y')} "
            f"@ ${oe['price']:,.2f} · "
            f"unrealized: <span style='color:{oe_col}; font-weight:700;'>"
            f"{unr:+.1f}%</span></div>",
            unsafe_allow_html=True,
        )

    # ── Charts ────────────────────────────────────────────────────────────────
    def _make_mstr_chart(bt, title):
        if bt is None:
            return None
        s     = bt["stats"]
        nav_s = bt["nav_series"]
        bh_s  = bt["bh_series"]
        has_p = bt["open_pos"]

        fig = go.Figure()

        regime = bt.get("bull_regime_series")
        if regime is not None and len(regime) > 0:
            arr     = regime.values.astype(bool)
            idx     = regime.index
            changes = np.where(np.diff(arr.astype(int)) != 0)[0] + 1
            starts  = np.concatenate([[0], changes])
            ends    = np.concatenate([changes, [len(arr)]])
            bull_leg = bear_leg = False
            for s_i, e_i in zip(starts, ends):
                is_bull  = bool(arr[s_i])
                color    = "rgba(34,197,94,0.10)" if is_bull else "rgba(239,68,68,0.07)"
                lname    = "🐂 Bull Regime (BTC)" if is_bull else "🐻 Bear/Neutral (BTC)"
                show_leg = (is_bull and not bull_leg) or (not is_bull and not bear_leg)
                e_idx    = min(e_i, len(idx) - 1)
                fig.add_vrect(
                    x0=idx[s_i], x1=idx[e_idx],
                    fillcolor=color, line_width=0,
                    name=lname, showlegend=show_leg,
                )
                if is_bull: bull_leg = True
                else:       bear_leg = True

        fig.add_trace(go.Scatter(
            x=bh_s.index, y=bh_s.values, name="Buy & Hold MSTR",
            line=dict(color="#94a3b8", width=1.5, dash="dot"),
            hovertemplate="%{x|%b %d, %Y}: $%{y:,.0f}<extra>Buy & Hold MSTR</extra>",
        ))
        fig.add_trace(go.Scatter(
            x=nav_s.index, y=nav_s.values, name="TF2+V-Gate (MSTR)",
            line=dict(color="#7c3aed", width=2.5),
            hovertemplate="%{x|%b %d, %Y}: $%{y:,.0f}<extra>TF2+V-Gate (MSTR)</extra>",
        ))
        fig.add_hline(
            y=s["initial_capital"], line_dash="dash",
            line_color="#64748b", line_width=1, opacity=0.4,
            annotation_text="  $100k start", annotation_position="bottom right",
        )
        for t in bt["trades"]:
            entry_y = float(nav_s.asof(t["entry_date"])) if t["entry_date"] >= nav_s.index[0] else s["initial_capital"]
            exit_y  = float(t["exit_nav"])
            win_col = "#16a34a" if t["pnl_pct"] > 0 else "#dc2626"
            fig.add_trace(go.Scatter(
                x=[t["entry_date"]], y=[entry_y], mode="markers",
                marker=dict(symbol="triangle-up", size=13, color="#16a34a",
                            line=dict(width=1.5, color="white")),
                showlegend=False,
                hovertemplate=(
                    f"<b>BUY MSTR</b> {pd.Timestamp(t['entry_date']).strftime('%b %d')}"
                    f" @ ${t['entry_price']:,.2f}<br>{t['entry_trigger']}<extra></extra>"
                ),
            ))
            fig.add_trace(go.Scatter(
                x=[t["exit_date"]], y=[exit_y], mode="markers",
                marker=dict(symbol="triangle-down", size=13, color=win_col,
                            line=dict(width=1.5, color="white")),
                showlegend=False,
                hovertemplate=(
                    f"<b>SELL MSTR</b> {pd.Timestamp(t['exit_date']).strftime('%b %d')}"
                    f" @ ${t['exit_price']:,.2f} "
                    f"({t['pnl_pct']:+.1f}%) — {t['exit_signal']}<extra></extra>"
                ),
            ))
        if has_p and bt["open_entry"]:
            oe   = bt["open_entry"]
            oe_y = float(nav_s.asof(oe["date"])) if oe["date"] >= nav_s.index[0] else s["initial_capital"]
            fig.add_trace(go.Scatter(
                x=[oe["date"]], y=[oe_y], mode="markers",
                marker=dict(symbol="triangle-up", size=13, color="#f59e0b",
                            line=dict(width=1.5, color="white")),
                showlegend=False,
                hovertemplate=(
                    f"<b>BUY MSTR (OPEN)</b> {pd.Timestamp(oe['date']).strftime('%b %d')}"
                    f" @ ${oe['price']:,.2f}<extra></extra>"
                ),
            ))
        fig.update_layout(
            title=dict(text=title, font=dict(size=13), x=0, xanchor="left"),
            xaxis_title=None,
            yaxis_title="Portfolio Value ($)",
            yaxis_tickprefix="$", yaxis_tickformat=",.0f",
            height=360,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=36, b=0),
            hovermode="x unified",
            plot_bgcolor="#f8fafc", paper_bgcolor="#ffffff",
            xaxis=dict(showgrid=True, gridcolor="#e2e8f0", zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="#e2e8f0", zeroline=False),
        )
        return fig

    tab_labels = []
    if bt_bear:
        sr = bt_bear["stats"]
        tab_labels.append(
            f"🐻 Bear Market  "
            f"({pd.Timestamp(sr['start_date']).strftime('%b %Y')} → "
            f"{pd.Timestamp(sr['end_date']).strftime('%b %Y')})"
        )
    if bt_bull:
        sf = bt_bull["stats"]
        tab_labels.append(
            f"🐂 Bull Market  "
            f"({pd.Timestamp(sf['start_date']).strftime('%b %Y')} → "
            f"{pd.Timestamp(sf['end_date']).strftime('%b %Y')})"
        )
    if bt_oos:
        so = bt_oos["stats"]
        tab_labels.append(
            f"🔬 OOS Only  "
            f"({pd.Timestamp(so['start_date']).strftime('%b %Y')} → "
            f"{pd.Timestamp(so['end_date']).strftime('%b %Y')})"
        )
    if bt_full:
        sfl = bt_full["stats"]
        tab_labels.append(
            f"🌐 Full Market  "
            f"({pd.Timestamp(sfl['start_date']).strftime('%b %Y')} → "
            f"{pd.Timestamp(sfl['end_date']).strftime('%b %Y')})"
        )

    if tab_labels:
        tabs = st.tabs(tab_labels)
        tab_idx = 0
        if bt_bear:
            with tabs[tab_idx]:
                sr = bt_bear["stats"]
                fig_r = _make_mstr_chart(
                    bt_bear,
                    f"TF2+V-Gate (MSTR) vs B&H MSTR — Bear Market  "
                    f"({pd.Timestamp(sr['start_date']).strftime('%b %d, %Y')} → "
                    f"{pd.Timestamp(sr['end_date']).strftime('%b %d, %Y')})",
                )
                if fig_r:
                    st.plotly_chart(fig_r, use_container_width=True,
                                    key=f"mstr_chart_bear_{key_suffix}")
            tab_idx += 1
        if bt_bull:
            with tabs[tab_idx]:
                sf = bt_bull["stats"]
                fig_f = _make_mstr_chart(
                    bt_bull,
                    f"TF2+V-Gate (MSTR) vs B&H MSTR — Bull Market  "
                    f"({pd.Timestamp(sf['start_date']).strftime('%b %d, %Y')} → "
                    f"{pd.Timestamp(sf['end_date']).strftime('%b %d, %Y')})",
                )
                if fig_f:
                    st.plotly_chart(fig_f, use_container_width=True,
                                    key=f"mstr_chart_bull_{key_suffix}")
            tab_idx += 1
        if bt_oos:
            with tabs[tab_idx]:
                so = bt_oos["stats"]
                fig_o = _make_mstr_chart(
                    bt_oos,
                    f"TF2+V-Gate (MSTR) vs B&H MSTR — OOS Period (Fully Blind)  "
                    f"({pd.Timestamp(so['start_date']).strftime('%b %d, %Y')} → "
                    f"{pd.Timestamp(so['end_date']).strftime('%b %d, %Y')})",
                )
                if fig_o:
                    st.plotly_chart(fig_o, use_container_width=True,
                                    key=f"mstr_chart_oos_{key_suffix}")
            tab_idx += 1
        if bt_full:
            with tabs[tab_idx]:
                sfl = bt_full["stats"]
                fig_fl = _make_mstr_chart(
                    bt_full,
                    f"TF2+V-Gate (MSTR) vs B&H MSTR — Full Market  "
                    f"({pd.Timestamp(sfl['start_date']).strftime('%b %d, %Y')} → "
                    f"{pd.Timestamp(sfl['end_date']).strftime('%b %d, %Y')})",
                )
                if fig_fl:
                    st.plotly_chart(fig_fl, use_container_width=True,
                                    key=f"mstr_chart_full_{key_suffix}")

    # ── Trade log ─────────────────────────────────────────────────────────────
    def _mstr_trade_log_rows(bt):
        if bt is None:
            return [], False, None
        s     = bt["stats"]
        has_p = bt["open_pos"]
        rows  = []
        for i, t in enumerate(bt["trades"], 1):
            rows.append({
                "#":           i,
                "Entry":       pd.Timestamp(t["entry_date"]).strftime("%b %d, %Y"),
                "Buy MSTR @":  f"${t['entry_price']:,.2f}",
                "BTC Trigger": t["entry_trigger"],
                "Exit":        pd.Timestamp(t["exit_date"]).strftime("%b %d, %Y"),
                "Sell MSTR @": f"${t['exit_price']:,.2f}",
                "P&L":         f"{t['pnl_pct']:+.1f}%",
                "Result":      "✓ WIN" if t["pnl_pct"] > 0 else "✗ LOSS",
                "Days":        t["duration_days"],
                "BTC Signal":  t["exit_signal"],
                "NAV After":   f"${t['exit_nav']:,.0f}",
            })
        if has_p and bt["open_entry"]:
            oe = bt["open_entry"]
            unr_pct = (float(bt["nav_series"].iloc[-1]) / float(oe["nav"]) - 1) * 100
            rows.append({
                "#":           len(bt["trades"]) + 1,
                "Entry":       pd.Timestamp(oe["date"]).strftime("%b %d, %Y"),
                "Buy MSTR @":  f"${oe['price']:,.2f}",
                "BTC Trigger": "—",
                "Exit":        "⏳ OPEN",
                "Sell MSTR @": "—",
                "P&L":         f"{unr_pct:+.1f}% (unrlzd)",
                "Result":      "🟡 OPEN",
                "Days":        (pd.Timestamp.now() - pd.Timestamp(oe["date"])).days,
                "BTC Signal":  "—",
                "NAV After":   "—",
            })
        return rows, has_p, s

    log_tabs = []
    if bt_bear: log_tabs.append("📋 Bear Market Trade Log")
    if bt_bull: log_tabs.append("📋 Bull Market Trade Log")
    if bt_oos:  log_tabs.append("📋 OOS Trade Log")
    if bt_full: log_tabs.append("📋 Full Market Trade Log")

    if log_tabs:
        ltabs = st.tabs(log_tabs)
        lt_idx = 0
        for bt_src, lbl in [(bt_bear, "bear"), (bt_bull, "bull"),
                             (bt_oos, "oos"), (bt_full, "full")]:
            if bt_src is None:
                continue
            rows, has_p, s = _mstr_trade_log_rows(bt_src)
            with ltabs[lt_idx]:
                if rows:
                    st.dataframe(pd.DataFrame(rows), hide_index=True,
                                 use_container_width=True)
                else:
                    st.info("No trades in this period — strategy was in cash throughout.")
                caption = (
                    "💡 P&L at MSTR execution prices (1-day lag from BTC signal). "
                    "Entry/exit triggered by BTC TF2+V-Gate. MSTR prices split-adjusted. "
                )
                if lbl == "oos":
                    caption += "✅ Fully OOS — BTC CT model never saw this data."
                elif lbl == "full":
                    caption += "⚠️ Mixed IS/OOS — pre-Mar 2026 trades are in-sample."
                else:
                    caption += "⚠️ pre-Mar 2026 = in-sample for BTC CT model."
                st.caption(caption)
            lt_idx += 1


def render_trend_signatures(sigs: dict, *, intraday: dict = None):
    """Render the Trend Signature Alert Dashboard in the Streamlit UI.

    Organized as:
      • Composite alert banner (color-coded overall level)
      • Four signature cards in 2×2 grid (D1, D2, D3, U1)
      • TF1 full-width card (U1 + MA30 trend-filter — best backtest strategy)
      • V-reversal special signal
      • LIVE INTRADAY strip (current bar vs predictions, updates every ~10 min)
      • Last-5-bars mini-table with signal sparklines

    Args:
        sigs:     Output of compute_trend_signatures() — historical bar signals.
        intraday: Output of _compute_intraday_signal() — current bar status.
                  None in historical mode or when data unavailable.
    """
    al   = sigs["alert_level"]
    dnc  = sigs["dn_count"]
    upc  = sigs["up_count"]

    # ── Color scheme ───────────────────────────────────────────────────
    ALERT_CFG = {
        "HIGH_DN":     {"bg": "#fee2e2", "border": "#dc2626", "badge_bg": "#dc2626",
                        "badge_txt": "⚠️ HIGH DOWNTREND ALERT",       "txt_col": "#7f1d1d"},
        "ELEVATED_DN": {"bg": "#fef3c7", "border": "#d97706", "badge_bg": "#d97706",
                        "badge_txt": "🟠 ELEVATED DOWNTREND RISK",     "txt_col": "#78350f"},
        "WATCH_DN":    {"bg": "#fffbeb", "border": "#f59e0b", "badge_bg": "#f59e0b",
                        "badge_txt": "👁 DOWNTREND WATCH",              "txt_col": "#92400e"},
        "STRATEGY_BUY":{"bg": "#eff6ff", "border": "#2563eb", "badge_bg": "#2563eb",
                        "badge_txt": "🎯 STRATEGY BUY SIGNAL (TF1)",   "txt_col": "#1e3a8a"},
        "WATCH_UP":    {"bg": "#f0fdf4", "border": "#16a34a", "badge_bg": "#16a34a",
                        "badge_txt": "📈 UPTREND SIGNAL (U1)",          "txt_col": "#14532d"},
        "NEUTRAL":     {"bg": "#f8fafc", "border": "#94a3b8", "badge_bg": "#64748b",
                        "badge_txt": "⬜ NO ACTIVE SIGNAL",             "txt_col": "#334155"},
    }
    cfg = ALERT_CFG[al]

    # ── Banner ─────────────────────────────────────────────────────────
    as_of_str = pd.Timestamp(sigs["as_of_date"]).strftime("%Y-%m-%d")
    st.markdown(
        f"""
        <div style="
            background:{cfg['bg']}; border:2px solid {cfg['border']};
            border-radius:10px; padding:14px 18px; margin:12px 0 6px 0;">
            <div style="display:flex; align-items:center; gap:12px; flex-wrap:wrap;">
                <span style="
                    background:{cfg['badge_bg']}; color:white; font-weight:700;
                    font-size:14px; padding:5px 14px; border-radius:20px;
                    white-space:nowrap;">
                    {cfg['badge_txt']}
                </span>
                <span style="color:{cfg['txt_col']}; font-size:13px;">
                    <b>{dnc}/3</b> DN · <b>{upc}/1</b> UP ·
                    TF2 entry: <b>{'✅ ACTIVE' if sigs['tf1_triggered'] else '○ inactive'}</b>
                    (U1={'✓' if sigs['u1_triggered'] else '✗'}
                    MA30={'↑' if sigs['above_ma30'] else '↓'}
                    slope={'↑' if sigs.get('ma30_slope_pos') else '↓'}
                    regime={'🐂BULL' if sigs.get('bull_regime') else '🐻BEAR'}
                    clean10d={'✓' if sigs['clean_10d'] else '✗'})
                    · as-of: <b>{as_of_str}</b>
                    · <b>{sigs['n_bars']}</b> bars
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── ACTION SIGNAL banner — synthesised top-level indicator ────────────
    _bull_regime  = sigs.get("bull_regime", False)
    _regime_label = "🐂 BULL" if _bull_regime else "🐻 BEAR / NEUTRAL"
    _v_gate_ok    = sigs.get("v_reversal_likely", False) or sigs.get("capitulation_signal", False)
    _trend_ok     = sigs["above_ma30"] or sigs["clean_10d"] or _v_gate_ok
    _entry_signal = sigs["u1_triggered"] and _trend_ok
    _exit_d3      = sigs.get("exhaustion_active", False)
    _exit_d2      = (sigs.get("err_hi_ma3", 0) < -1.0) and not _bull_regime
    _exit_signal  = _exit_d3 or _exit_d2

    if _exit_signal and _entry_signal:
        _action_bg    = "#fef2f2"; _action_brd = "#dc2626"; _action_emoji = "🔴"
        _action_label = "EXIT SIGNAL ACTIVE"
        _action_sub   = (
            f"D3={'FIRED' if _exit_d3 else 'clear'}, "
            f"D2 (bear-only)={'FIRED' if _exit_d2 else 'clear'}  |  "
            f"Entry conditions also met — EXIT takes priority if in position"
        )
    elif _exit_signal:
        _action_bg    = "#fef2f2"; _action_brd = "#dc2626"; _action_emoji = "🔴"
        _action_label = "EXIT SIGNAL ACTIVE"
        _exit_parts   = []
        if _exit_d3: _exit_parts.append("D3 exhaustion fired (exit in all regimes)")
        if _exit_d2: _exit_parts.append(f"D2 hi-band collapse in {_regime_label} regime → defensive exit")
        _action_sub   = " · ".join(_exit_parts)
    elif _entry_signal:
        _action_bg    = "#f0fdf4"; _action_brd = "#16a34a"; _action_emoji = "🟢"
        _action_label = "ENTRY SIGNAL ACTIVE"
        _entry_gates  = []
        if sigs["above_ma30"]: _entry_gates.append("↑MA30")
        if sigs["clean_10d"]:  _entry_gates.append("Clean 10d")
        if _v_gate_ok:         _entry_gates.append("⚡V-reversal")
        _action_sub   = (
            f"U1 confirmed · trend gate: {' + '.join(_entry_gates)}  |  "
            f"Regime: {_regime_label}"
        )
    elif sigs["u1_triggered"] and not _trend_ok:
        _action_bg    = "#fefce8"; _action_brd = "#ca8a04"; _action_emoji = "🟡"
        _action_label = "U1 ACTIVE — TREND GATE BLOCKED"
        _action_sub   = (
            "U1 fired but entry requires ↑MA30 OR clean10d OR ⚡V-reversal. "
            f"BTC {'above' if sigs['above_ma30'] else 'below'} MA30, "
            f"clean10d={'YES' if sigs['clean_10d'] else 'NO'}, "
            f"V-rev={'YES' if _v_gate_ok else 'NO'} — watch for reversal"
        )
    else:
        _action_bg    = "#f8fafc"; _action_brd = "#94a3b8"; _action_emoji = "⬜"
        _action_label = "NO ACTIVE SIGNAL — NEUTRAL / WATCH"
        _action_sub   = (
            f"U1={'active' if sigs['u1_triggered'] else 'inactive'}, "
            f"Trend={'pass' if _trend_ok else 'blocked'}, "
            f"Exit={'none' if not _exit_signal else 'active'}  |  "
            f"Regime: {_regime_label}"
        )

    st.markdown(
        f"""
        <div style="background:{_action_bg}; border:2.5px solid {_action_brd};
            border-radius:12px; padding:14px 20px; margin:8px 0 4px 0;
            display:flex; align-items:center; gap:14px;">
          <div style="font-size:28px; line-height:1;">{_action_emoji}</div>
          <div>
            <div style="font-size:16px; font-weight:800; color:{_action_brd};
                letter-spacing:0.5px;">{_action_label}</div>
            <div style="font-size:12px; color:#334155; margin-top:3px;
                line-height:1.5;">{_action_sub}</div>
          </div>
          <div style="margin-left:auto; font-size:11px; color:#64748b;
              text-align:right; line-height:1.6;">
            <b>TF2 + V-Gate Strategy</b><br>
            Signal as of today's close
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Intraday V-gate watch note — advisory only, no premature action possible
    if (intraday is not None
            and intraday.get("lo_break_intra", False)
            and intraday.get("err_lo_intra", 0) > 5.0
            and intraday.get("pct_through", 0) >= 0.40
            and not sigs.get("v_reversal_likely", False)
            and not sigs.get("capitulation_signal", False)):
        _ia_pct = intraday["pct_through"] * 100
        _ia_hrs = intraday["hours_elapsed"]
        _ia_elo = intraday["err_lo_intra"]
        _ia_lo  = intraday["running_low"]
        _ia_plo = intraday["pred_lo"]
        st.markdown(
            f"""
            <div style="background:#faf5ff; border:1.5px dashed #7c3aed;
                border-radius:10px; padding:11px 16px; margin:2px 0 6px 0;
                display:flex; align-items:flex-start; gap:12px;">
              <div style="font-size:20px; line-height:1.3;">📍</div>
              <div style="flex:1;">
                <div style="font-size:13px; font-weight:700; color:#5b21b6;
                    margin-bottom:3px;">
                  INTRADAY WATCH — V-Reversal Threshold Hit
                </div>
                <div style="font-size:12px; color:#334155; line-height:1.6;">
                  Today's low <b>${_ia_lo:,.0f}</b> is already
                  <b>{_ia_elo:.1f}% below</b> the predicted floor
                  (${_ia_plo:,.0f}) with <b>{_ia_hrs}h elapsed
                  ({_ia_pct:.0f}% of bar complete)</b>.
                  If the bar closes here, the <b>⚡ V-gate will open</b> for
                  the next 3 daily closes.<br>
                  <span style="color:#7c3aed; font-weight:600;">
                  ⚠️ No action yet — entry also requires U1 to fire at bar
                  close. Watch tonight's close; ACTION SIGNAL above will
                  update within 6 hours of bar close.</span>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── 4 signature cards in 2×2 grid ─────────────────────────────────
    def _sig_card(title, icon, color, triggered, signal_rows, interpretation,
                  timing, prob_txt, conf_txt):
        """Render one signature card as styled HTML."""
        border_col  = color if triggered else "#cbd5e1"
        bg_col      = ("#fff7ed" if (triggered and color == "#dc2626")
                       else ("#f0fdf4" if (triggered and color == "#16a34a")
                             else "#f8fafc"))
        status_txt  = "● ACTIVE" if triggered else "○ CLEAR"
        status_badge = (
            "<span style='background:" + color + "; color:white; border-radius:12px; "
            "padding:2px 10px; font-size:11px; font-weight:700; "
            "margin-left:8px;'>" + status_txt + "</span>"
        )
        rows_html = ""
        for label, val, thr, fired in signal_rows:
            val_col = color if fired else "#64748b"
            if fired:
                fired_span = ("<span style='color:" + color
                              + "; font-weight:700; margin-left:auto;'>✓ TRIGGERED</span>")
            else:
                fired_span = "<span style='color:#94a3b8; margin-left:auto;'>○</span>"
            rows_html += (
                "<div style='display:flex; align-items:center; gap:8px; "
                "padding:4px 0; border-bottom:1px solid #e2e8f0;'>"
                "<span style='font-size:12px; color:#64748b; width:130px; "
                "flex-shrink:0;'>" + label + "</span>"
                "<span style='font-weight:700; color:" + val_col + "; font-size:13px; "
                "min-width:70px;'>" + val + "</span>"
                "<span style='font-size:11px; color:#94a3b8;'>threshold: " + thr + "</span>"
                + fired_span +
                "</div>"
            )
        title_col = color if triggered else "#475569"
        return (
            "<div style='background:" + bg_col + "; border:2px solid " + border_col + "; "
            "border-radius:10px; padding:14px; height:100%; box-sizing:border-box;'>"
            # Title row
            "<div style='font-size:14px; font-weight:700; color:" + title_col + "; "
            "margin-bottom:10px; line-height:1.3;'>"
            + icon + " " + title + status_badge + "</div>"
            # Signal values
            + "<div style='margin-bottom:10px;'>" + rows_html + "</div>"
            # Interpretation
            + "<div style='font-size:12px; color:#1e293b; background:rgba(0,0,0,0.04); "
            "border-radius:6px; padding:8px; margin-bottom:8px; line-height:1.5;'>"
            "<b>📊 What it predicts:</b><br>" + interpretation + "</div>"
            # Timing + Probability + Confidence
            + "<div style='font-size:11px; color:#475569; line-height:1.6;'>"
            "⏱ <b>Timing:</b> " + timing + "<br>"
            "📉 <b>Probability:</b> " + prob_txt + "<br>"
            "🔬 <b>Evidence:</b> " + conf_txt
            + "</div></div>"
        )

    col1, col2 = st.columns(2)

    # Card D1
    d1_rows = [
        ("Low-band breaks (3d)", f"{sigs['lo_breaks_3d']}/3",  "≥ 2",    sigs['lo_breaks_3d'] >= 2),
        ("err_lo 3d avg",        f"{sigs['err_lo_ma3']:+.2f}%","> +0.5%", sigs['err_lo_ma3'] > 0.5),
    ]
    with col1:
        st.markdown(
            _sig_card(
                title="D1 — Low-Band Accumulation",
                icon="🔴",
                color="#dc2626",
                triggered=sigs["d1_triggered"],
                signal_rows=d1_rows,
                interpretation=(
                    "The daily LOW keeps punching <b>below</b> the ensemble's predicted floor "
                    "on 2+ of the last 3 days. All three regressors (Huber, Quantile-GBM, RF) "
                    "have their downside cushion wrong — price is dropping faster than the "
                    "feature set has absorbed. A significant further drop within 1–3 days "
                    "is the most common outcome."
                ),
                timing="Typically precedes the next big drop by <b>1–3 days</b>. "
                       "Jan 30 2026 fired (T-1 before −6.5%); Feb 4 2026 fired (T-1 before −14.1%).",
                prob_txt="<b>26.6% hit rate</b> for >4% drop in next 3 days (base: 17.8%) → "
                         "<b>1.49× lift</b>. At score >0.7: <b>38.5% hit rate → 2.16× lift</b>.",
                conf_txt="p &lt; 0.001 (Mann-Whitney). Both lo_break rate and err_lo_pct "
                         "are highly significant pre-downtrend vs normal days.",
            ),
            unsafe_allow_html=True,
        )

    # Card D2
    d2_rows = [
        ("err_hi 3d avg",     f"{sigs['err_hi_ma3']:+.2f}%", "< −1.0%", sigs['err_hi_ma3'] < -1.0),
        ("Hi-band breaks (3d)",f"{sigs['hi_breaks_3d']}/3",  "< 1",     sigs['hi_breaks_3d'] < 1),
    ]
    with col2:
        st.markdown(
            _sig_card(
                title="D2 — Predicted-High Collapse",
                icon="🔴",
                color="#dc2626",
                triggered=sigs["d2_triggered"],
                signal_rows=d2_rows,
                interpretation=(
                    "The models predicted an upside the market can't deliver. Actual daily "
                    "highs are falling <b>below</b> what the ensemble forecast. The feature "
                    "set (momentum, macro, on-chain) still 'smells' uptrend, but price "
                    "refuses to reach the predicted ceiling — a classic <b>exhaustion fingerprint</b>. "
                    "The model's momentum loading is fighting a structural reversal."
                ),
                timing="Present throughout the <b>Jan 29 – Feb 5 2026 cascade</b>. "
                       "Each predicted high was an 'air target' the market rejected. "
                       "Oct 2025 pre-drop window: err_hi_ma3 ran at −0.5% to −1.6%.",
                prob_txt="Most statistically significant signal found. PRE_DN mean "
                         "<b>err_hi_pct = −1.34%</b> vs NORMAL −0.22% → <b>Δ = −1.13%</b>.",
                conf_txt="<b>p &lt; 0.0001</b> (Mann-Whitney). Strongest single discriminator "
                         "between pre-downtrend and normal days in 241-bar OOS test.",
            ),
            unsafe_allow_html=True,
        )

    col3, col4 = st.columns(2)

    # Card D3
    d3_streak_str = f"streak = {sigs['streak']:+d}" if sigs["streak"] != 0 else "streak = 0"
    d3_rows = [
        ("Consecutive hi-breaks before today", f"{sigs['hi_breaks_3d']}/3", "≥ 3 prior", False),
        ("Lo-break today",                     "YES" if sigs["detail_rows"] and sigs["detail_rows"][-1]["lo_break"] else "NO",
                                               "= 1",
                                               bool(sigs["detail_rows"] and sigs["detail_rows"][-1]["lo_break"])),
        ("Current streak",                     d3_streak_str, "≥ +4 prior", sigs["streak"] >= 4),
    ]
    with col3:
        st.markdown(
            _sig_card(
                title="D3 — Exhaustion Canary",
                icon="🟡",
                color="#d97706",
                triggered=sigs["d3_triggered"],
                signal_rows=d3_rows,
                interpretation=(
                    "After ≥ 3 consecutive days where actual highs <b>exceeded</b> the "
                    "predicted high (strong uptrend momentum), the <b>first appearance of "
                    "a low-band break</b> marks exhaustion. Price was pushing through every "
                    "predicted ceiling, then suddenly can't hold and takes out the predicted floor. "
                    "This is the exact inflection point — the momentum-to-reversal handoff. "
                    "Context: Oct 6 2025 fired; Oct 7–10 dropped −9.5% over 3 days."
                ),
                timing="Fires at the <b>exact top</b> of a momentum move — usually 1 day "
                       "before the reversal accelerates. Most useful after streak ≥ +4 days.",
                prob_txt="Contextual signal — most reliable when streak ≥ +4 AND this is "
                         "the <b>first</b> lo_break in the series. Oct 2025 textbook case.",
                conf_txt="Qualitative pattern (small sample). Use as confirming signal for "
                         "D1 or D2, not as standalone alert.",
            ),
            unsafe_allow_html=True,
        )

    # Card U1
    u1_rows = [
        ("err_hi 3d avg",      f"{sigs['err_hi_ma3']:+.2f}%","  > +0.5%", sigs['err_hi_ma3'] > 0.5),
        ("Hi-band breaks (3d)", f"{sigs['hi_breaks_3d']}/3", "≥ 2",       sigs['hi_breaks_3d'] >= 2),
        ("Current streak",      d3_streak_str,                "> 0",       sigs['streak'] > 0),
    ]
    with col4:
        st.markdown(
            _sig_card(
                title="U1 — High-Band Breakout Persistence",
                icon="🟢",
                color="#16a34a",
                triggered=sigs["u1_triggered"],
                signal_rows=u1_rows,
                interpretation=(
                    "Actual daily highs consistently <b>exceed</b> what the ensemble predicted "
                    "for 3+ days. All three regressors are systematically under-estimating "
                    "the upside. The feature set hasn't fully priced in the momentum — the "
                    "market is pushing through every predicted ceiling, a sign that the "
                    "uptrend has more runway. May 2026 gradual climb: fired 2 days before."
                ),
                timing="Precedes continued upward momentum by <b>1–3 days</b>. "
                       "Strongest when hi_breaks persist 3+ days with a positive streak. "
                       "V-reversal: also fires 1 day after a capitulation crash (see below).",
                prob_txt="<b>16.0% hit rate</b> for >4% up move in next 3 days "
                         "(base: 9.5%) → <b>1.68× lift</b>. Weaker than downtrend signals — "
                         "explosive uptrends are harder to detect from band patterns alone.",
                conf_txt="p = 0.022 (Mann-Whitney). Significant but lower lift than "
                         "downtrend signals. Use alongside 3-class BigUpper prediction.",
            ),
            unsafe_allow_html=True,
        )

    # ── TF2 full-width card (U1 + MA30 + regime-adaptive exit) ───────────
    # _bull_regime / _regime_label already computed above for ACTION SIGNAL
    _ma30_pct  = (sigs["current_close_sig"] / sigs["ma30_value"] - 1) * 100
    _slope_pct = (sigs["ma30_value"] / sigs["ma30_5d_ago"] - 1) * 100 if sigs.get("ma30_5d_ago") else 0.0
    _exit_mode = "D3 only (patient — hold the trend)" if _bull_regime else "D2 OR D3 (defensive exit)"

    tf1_rows = [
        ("U1 Signal",
         "✅ ACTIVE" if sigs["u1_triggered"] else "○ inactive",
         "= ACTIVE",
         sigs["u1_triggered"]),
        ("BTC vs 30-day MA",
         f"${sigs['current_close_sig']:,.0f}  vs  MA=${sigs['ma30_value']:,.0f} ({_ma30_pct:+.1f}%)",
         "> MA30",
         sigs["above_ma30"]),
        ("MA30 slope (5-bar)",
         f"MA30 = ${sigs['ma30_value']:,.0f}  vs  5d ago = ${sigs.get('ma30_5d_ago', sigs['ma30_value']):,.0f} "
         f"({_slope_pct:+.2f}%)",
         "> 0 (rising)",
         sigs.get("ma30_slope_pos", False)),
        ("Regime",
         f"{_regime_label}  →  Exit mode: {_exit_mode}",
         "≠ BEAR",
         _bull_regime),
        ("Clean 10d (no D1/D2)",
         "YES — zero D1/D2 fires in prior 10 bars" if sigs["clean_10d"] else "NO — recent D1 or D2 fired",
         "= YES",
         sigs["clean_10d"]),
        ("⚡ V-reversal (3-bar gate)",
         ("ACTIVE — capitulation within 3 bars, dn_score={:.2f}, err_lo={:+.1f}%"
          .format(sigs.get("dn_score_raw", 0), sigs.get("last_lo_err", 0)))
         if sigs.get("v_reversal_likely") or sigs.get("capitulation_signal")
         else "○ not active — no recent capitulation spike (err_lo < 5% or dn_score < 0.8)",
         "= ACTIVE",
         sigs.get("v_reversal_likely", False) or sigs.get("capitulation_signal", False)),
        ("Entry filter (↑MA30 OR clean10d OR V-rev)",
         "PASS ✅" if (sigs["above_ma30"] or sigs["clean_10d"]
                       or sigs.get("v_reversal_likely") or sigs.get("capitulation_signal"))
         else "FAIL ✗",
         "= PASS",
         sigs["above_ma30"] or sigs["clean_10d"]
         or sigs.get("v_reversal_likely", False) or sigs.get("capitulation_signal", False)),
    ]
    st.markdown(
        _sig_card(
            title="TF2 + V-Gate — Regime-Adaptive Strategy: U1 + Trend + V-reversal + Regime Exit",
            icon="🎯",
            color="#2563eb",
            triggered=sigs["tf1_triggered"],
            signal_rows=tf1_rows,
            interpretation=(
                "The <b>optimised regime-adaptive strategy</b> with V-reversal gate improvement. "
                "Entry: U1 (hi-band persistence) AND at least one of: BTC &gt; MA30, clean 10-day window, "
                "<b>or ⚡ V-reversal capitulation within 3 bars</b> (catches post-crash recoveries "
                "before price has crossed MA30). "
                "Exits adapt to regime: <b>BULL</b> (BTC &gt; MA30 &amp; MA30 rising) → D3 only "
                "(patient); <b>BEAR/Neutral</b> → D2 or D3 (defensive). "
                "V-gate improvement: +$4,685 NAV, Sharpe 0.86→0.90, Max DD −26.9%→−23.6% "
                "over the 2-year backtest period."
            ),
            timing=(
                "Entry fires 1–2 bars before momentum accelerates. "
                "V-gate fires 1–3 bars after capitulation (dn_score &gt; 0.8, err_lo &gt; 5%) "
                "— bridges the gap before U1 co-fires on the recovery bounce. "
                "In BULL regime: hold through D2 dips. In BEAR/neutral: exit quickly on D2."
            ),
            prob_txt=(
                "2-year backtest (May 2024–May 2026): <b>+87.9% return</b>, Sharpe 0.90, MaxDD −23.6%. "
                "B&amp;H: +23.5%. Alpha: <b>+$64,594</b>. "
                "OOS bear period (Sep 25–May 26): +$30k alpha. "
                "Bull period (in-sample ⚠️): +68% vs B&amp;H +73% — 87% of bull captured."
            ),
            conf_txt=(
                "⚠️ Bull-period test is in-sample (CT model trained through Sep 2025). "
                "Bear-period test is fully OOS. V-gate adds 2 trades over 2 years — "
                "both confirmed by U1 co-firing within 3 bars of capitulation. "
                "No strategy outperforms B&amp;H in both a +93% bull AND a −33% bear without leverage."
            ),
        ),
        unsafe_allow_html=True,
    )

    # ── V-Reversal signal — always show; highlight when active ────────────
    _v_likely = sigs.get("v_reversal_likely", False)
    _v_cap    = sigs.get("capitulation_signal", False)
    _v_active = _v_likely or _v_cap
    if _v_active:
        v_bg    = "#ede9fe" if _v_likely else "#fdf4ff"
        v_brd   = "#7c3aed" if _v_likely else "#a855f7"
        v_label = "⚡ V-REVERSAL ACTIVE — Entry gate open" if _v_likely else "💜 Capitulation Detected — monitoring for U1"
        v_body  = (
            f"Today's actual low <b>massively overshot</b> the predicted low "
            f"(<b>err_lo = {sigs['last_lo_err']:+.2f}%</b>, dn_score = <b>{sigs['dn_score_raw']:.2f}</b>). "
            f"This is the <b>V-reversal capitulation pattern</b>. "
            f"<b>The V-gate is open for the next 3 bars</b> — if U1 fires (err_hi_ma3 &gt; +0.5% AND "
            f"hi_breaks_3d ≥ 2) the strategy will enter even below MA30.<br><br>"
            f"Historical precedent: Feb 5 2026 (−14.1%, dn_score=4.59) → Feb 6 bounced +12.5%."
        )
    else:
        v_bg    = "#f8fafc"
        v_brd   = "#cbd5e1"
        v_label = "⚡ V-Reversal Gate — not active"
        v_body  = (
            f"No capitulation signal today (dn_score = <b>{sigs.get('dn_score_raw', 0):.2f}</b>, "
            f"err_lo = <b>{sigs.get('last_lo_err', 0):+.2f}%</b>). "
            f"Gate requires dn_score &gt; 0.8 <b>AND</b> err_lo &gt; 5.0% on the same bar. "
            f"<br><br>"
            f"<b>What the V-gate does:</b> When a capitulation spike fires, the strategy can enter "
            f"on U1 within the next 3 bars even if BTC is below MA30 and clean_10d fails — "
            f"catching post-crash recoveries before price has recrossed the trend filter. "
            f"Backtest impact: +$4,685 NAV, Sharpe 0.86→0.90, Max DD −26.9%→−23.6% (2-year period)."
        )
    st.markdown(
        f"""
        <div style="background:{v_bg}; border:2px solid {v_brd};
            border-radius:10px; padding:14px 18px; margin:8px 0;">
            <div style="font-weight:700; color:{v_brd}; font-size:13px;
                margin-bottom:6px;">{v_label}</div>
            <div style="font-size:12px; color:#1e293b; line-height:1.7;">
                {v_body}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Live Intraday Bar Panel ────────────────────────────────────────
    # Shows the current in-progress bar's running H/L vs today's frozen
    # daily predictions. Updates every ~10 minutes via _fetch_binance_hourly.
    # ── Last 5 bars mini-table ─────────────────────────────────────────
    if sigs["detail_rows"]:
        with st.expander("📋 Last 5 bars — signal detail", expanded=False):
            st.caption(
                "Each row is a **completed** daily bar (actual H/L known). "
                "err_hi = (actual_high − pred_high) / close × 100 — positive = actual exceeded prediction. "
                "err_lo = (pred_low − actual_low) / close × 100 — positive = actual undershot (more bearish). "
                "Hi/Lo Break = actual H > pred_H or actual L < pred_L."
            )
            rows_display = []
            for r in sigs["detail_rows"]:
                rows_display.append({
                    "Date":        pd.Timestamp(r["date"]).strftime("%Y-%m-%d"),
                    "Close ($)":   f"${r['close']:,.0f}",
                    "Pred H ($)":  f"${r['pred_hi']:,.0f}",
                    "Actual H ($)":f"${r['actual_hi']:,.0f}",
                    "err_hi (%)":  f"{r['err_hi_pct']:+.2f}%",
                    "Hi Break":    "✓" if r["hi_break"] else "–",
                    "Pred L ($)":  f"${r['pred_lo']:,.0f}",
                    "Actual L ($)":f"${r['actual_lo']:,.0f}",
                    "err_lo (%)":  f"{r['err_lo_pct']:+.2f}%",
                    "Lo Break":    "✓" if r["lo_break"] else "–",
                })
            st.dataframe(
                pd.DataFrame(rows_display),
                hide_index=True,
                use_container_width=True,
            )

        # ── Legend / interpretation guide ──────────────────────────────
        with st.expander("ℹ️ How to read the Trend Alert Dashboard", expanded=False):
            st.markdown("""
**Signal definitions** (all computed from the last 3–45 completed daily bars):

| Signal | Formula | Meaning when positive/high |
|--------|---------|---------------------------|
| `err_hi_ma3` | 3d avg of (actual_high − pred_high) / close × 100 | Actual highs **exceed** predictions → bullish pressure |
| `err_lo_ma3` | 3d avg of (pred_low − actual_low) / close × 100 | Actual lows **undershot** predictions → bearish pressure |
| `hi_breaks_3d` | Count of days in last 3 where actual H > pred H | Repeated upside breakouts → momentum building |
| `lo_breaks_3d` | Count of days in last 3 where actual L < pred L | Repeated downside breaks → trend deteriorating |
| `MA30` | Rolling 30-bar mean of daily close prices | Trend direction proxy (above = bullish context) |
| `clean_10d` | True if zero D1 or D2 fires in prior 10 bars | No recent bearish regime fingerprint |

**Alert levels:**
- 🔴 **HIGH DOWNTREND**: 3/3 or 2/3 + V-reversal → Highest-confidence downtrend signal (2.24× lift)
- 🟠 **ELEVATED DOWNTREND**: 2/3 conditions active → Strong signal (1.75× lift on err_lo_ma3)
- 🟡 **WATCH DOWNTREND**: 1 condition only → Monitor, not high-confidence alone
- 🎯 **STRATEGY BUY (TF2 + V-Gate)**: U1 + (↑MA30 OR clean10d OR ⚡V-reversal) → Regime-adaptive entry (see TF2 card below)
- 🟢 **UPTREND SIGNAL (U1)**: U1 active but entry gate not yet met → Upside momentum (1.68× lift)
- ⬜ **NEUTRAL**: No conditions active → Normal market, no strong directional signal

**TF2 + V-Gate — Regime-Adaptive Strategy (optimised across bull + bear regimes):**
- **Entry**: U1 fires AND (BTC > 30-day MA **OR** no D1/D2 in prior 10 days **OR** ⚡ V-reversal within 3 bars)
- **Exit** in BULL regime (price > MA30 AND MA30 rising): D3 only (patient — hold the trend)
- **Exit** in BEAR/Neutral regime: D2 (err_hi_ma3 < −1%) or D3 (defensive — cut quickly)
- **2-year backtest** (May 24–May 26): +87.9% return · Sharpe 0.90 · Max DD −23.6% · Alpha +$64,594 vs B&H
- **Bear period OOS** (Sep 25–May 26): +$30k alpha vs B&H · V-gate adds post-crash entries before MA30 cross

**Probability context:**
The hit rates above are from a 241-bar out-of-sample test window (Sep 2025 – May 2026).
They represent the fraction of times a ≥4% single-day move in that direction occurred
within the next 3 days after the signal fired. Lift = hit rate ÷ base rate.
These are informational only — the daily direction accuracy of the underlying models is ~50%.
            """)

    st.markdown("<div style='margin-bottom:6px;'></div>", unsafe_allow_html=True)


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
    _raw_latest = _fetch_daily_raw()
    _data_end   = _raw_latest.index.max().strftime("%Y-%m-%d") if not _raw_latest.empty else None
    daily = compute_daily_forecast(target_date.strftime("%Y-%m-%d"), data_end=_data_end)

    # Pre-compute cached results (no latency hit) for display at top of page.
    _bt_end      = target_date.strftime("%Y-%m-%d")
    # Pass model mtime so cache auto-invalidates when inference_assets_ct.joblib changes.
    _model_mtime = float(os.path.getmtime(str(DAILY_MODEL_CT))) if os.path.exists(str(DAILY_MODEL_CT)) else 0.0
    # Bear / Bull / Full Market use locked end dates (established 2026-05-28) and a
    # no-TTL cache so results are computed once per model version, not on every load.
    # Only the OOS period rolls daily; it calls run_full_period_backtest directly.
    _bt_bear     = _run_fixed_period_backtest(_BT_BEAR_END_LOCKED, "2025-06-01",  _model_mtime)
    _bt_bull     = _run_fixed_period_backtest("2025-06-14",         "2024-06-05", _model_mtime)
    _bt_full_oos = run_full_period_backtest(_bt_end, model_mtime=_model_mtime)  # rolls daily
    _bt_full     = _run_fixed_period_backtest(_BT_FULL_END_LOCKED,  "2024-06-05", _model_mtime)
    _chart_key   = "live" if is_live else "hist"
    sigs         = compute_trend_signatures(_bt_end)

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
    c1, c2, c3, c4, c5 = st.columns(5)
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
    _regime_bull = sigs.get("bull_regime", False) if sigs else None
    if _regime_bull is not None:
        _regime_val   = "🐂 BULL"          if _regime_bull else "🐻 BEAR / NEUTRAL"
        _regime_delta = "↑MA30 + rising"   if _regime_bull else "below MA30 or flat"
        c5.metric("Market Regime", _regime_val, delta=_regime_delta,
                  delta_color="normal" if _regime_bull else "inverse")
    else:
        c5.metric("Market Regime", "—")

    # ─────────────── TF2 + V-Gate Signal Watch Dashboard ─────────────────
    if sigs is not None:
        _intra_raw = _fetch_current_bar_intraday() if is_live else None
        _intra_sig = _compute_intraday_signal(_intra_raw, daily) if _intra_raw else None
        render_trend_signatures(sigs, intraday=_intra_sig)

    # ─────────────── TF2 + V-Gate Strategy Backtest Dashboard ────────────
    render_trading_strategy_dashboard(_bt_bear, _bt_bull, bt_full_oos=_bt_full_oos,
                                       bt_full=_bt_full, key_suffix=_chart_key)

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
    day_type = compute_day_type_forecast(target_date.strftime("%Y-%m-%d"), data_end=_data_end)
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
        _vtest_cov   = cone_at["within_pct"] if cone_at else _cone_meta.get("held_out_band_coverage_pct", 0)
        _vtest_mape  = cone_at["mape"]        if cone_at else None
        _vtest_dacc  = cone_at["dir_acc"]     if cone_at else None
        _cc = st.columns(4)
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
        if _vtest_dacc:
            _em_dacc = _badge(cone30["dir_acc"], _vtest_dacc, lo_better=False)
            _cc[2].metric(
                f"{_em_dacc} Dir accuracy",
                f"{cone30['dir_acc']:.1f} %",
                delta=(f"{cone30['dir_acc'] - _vtest_dacc:+.1f} pp  "
                       f"(test: {_vtest_dacc:.1f} %)"),
                delta_color="normal",
                help="% of forecasts where predicted direction (up/down vs anchor) matched actual",
            )
        else:
            _cc[2].metric("Dir accuracy", f"{cone30['dir_acc']:.1f} %",
                          help="% of forecasts where predicted direction (up/down vs anchor) matched actual")
        _cc[3].metric("Resolved (30 d)", str(cone30["n"]),
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
        _vtest_dacc14 = cone14_at["dir_acc"]     if cone14_at else None
        _cc14 = st.columns(4)
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
        if _vtest_dacc14:
            _em_dacc14 = _badge(cone14_30["dir_acc"], _vtest_dacc14, lo_better=False)
            _cc14[2].metric(
                f"{_em_dacc14} Dir accuracy",
                f"{cone14_30['dir_acc']:.1f} %",
                delta=(f"{cone14_30['dir_acc'] - _vtest_dacc14:+.1f} pp  "
                       f"(test: {_vtest_dacc14:.1f} %)"),
                delta_color="normal",
                help="% of forecasts where predicted direction (up/down vs anchor) matched actual",
            )
        else:
            _cc14[2].metric("Dir accuracy", f"{cone14_30['dir_acc']:.1f} %",
                            help="% of forecasts where predicted direction (up/down vs anchor) matched actual")
        _cc14[3].metric("Resolved (30 d)", str(cone14_30["n"]),
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
        _vt_cov2  = cone_at["within_pct"] if cone_at else _cone_meta.get("held_out_band_coverage_pct", 88)
        _vt_da7   = cone_at["dir_acc"]    if cone_at else None
        _pm_c     = _badge(cone30["within_pct"], _vt_cov2, False)
        _da_c     = _badge(cone30["dir_acc"], _vt_da7, False) if _vt_da7 else "—"
        _ov_c7    = "🔴" if "🔴" in (_pm_c, _da_c) else ("🟡" if "🟡" in (_pm_c, _da_c) else "🟢")
        _drift_rows.append({
            "Model":           "📐 7-day cone",
            "Perf drift":      (f"{_pm_c}  coverage {cone30['within_pct']:.1f}% "
                                f"vs {_vt_cov2:.1f}%"),
            "Direction drift": (f"{_da_c}  dir {cone30['dir_acc']:.1f}% "
                                f"vs {_vt_da7:.1f}%") if _vt_da7 else "—",
            "Overall":         _ov_c7,
        })
    if cone14_30:
        _vt_cov14_2 = cone14_at["within_pct"] if cone14_at else _cone14_meta.get("held_out_band_coverage_pct", 89)
        _vt_da14    = cone14_at["dir_acc"]     if cone14_at else None
        _pm_c14     = _badge(cone14_30["within_pct"], _vt_cov14_2, False)
        _da_c14     = _badge(cone14_30["dir_acc"], _vt_da14, False) if _vt_da14 else "—"
        _ov_c14     = "🔴" if "🔴" in (_pm_c14, _da_c14) else ("🟡" if "🟡" in (_pm_c14, _da_c14) else "🟢")
        _drift_rows.append({
            "Model":           "📆 14-day cone",
            "Perf drift":      (f"{_pm_c14}  coverage {cone14_30['within_pct']:.1f}% "
                                f"vs {_vt_cov14_2:.1f}%"),
            "Direction drift": (f"{_da_c14}  dir {cone14_30['dir_acc']:.1f}% "
                                f"vs {_vt_da14:.1f}%") if _vt_da14 else "—",
            "Overall":         _ov_c14,
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
# Retrain Status Dashboard  (TF2 strategy models only)
# ════════════════════════════════════════════════════════════════════════
def render_retrain_dashboard():
    """Re-training status for the three models that power TF2 trading signals."""
    today_utc = pd.Timestamp(datetime.now(timezone.utc)).normalize().tz_convert(None)
    _end_iso  = today_utc.strftime("%Y-%m-%d")

    # ── Load artefact metadata (cache_resource — instant) ────────────────────
    _hl_art   = _load_daily_hl()
    _hl_meta  = _hl_art.get("calibration_meta", {}) if _hl_art else {}
    _hl_mtest = _hl_meta.get("metrics", {})
    _cone_art  = _load_cone_7d()
    _cone_meta = _cone_art.get("calibration_meta", {}) if _cone_art else {}
    _dt_art   = _load_day_type()
    _dt_meta  = _dt_art.get("calibration_meta", {}) if _dt_art else {}
    cutoffs   = _training_cutoffs()

    # ── 30-day & all-time performance (cache_data — shared with Live tab) ─────
    hl30    = compute_30d_daily_hl_metrics(_end_iso)
    cone30  = compute_30d_cone_metrics(_end_iso)
    cone_at = compute_alltime_cone_metrics(_end_iso)
    dt30    = compute_30d_daytype_metrics(_end_iso)
    dt_at   = compute_alltime_daytype_metrics(_end_iso)
    sigs    = compute_trend_signatures(_end_iso) or {}

    # ── Age thresholds (days since train_end) ─────────────────────────────────
    WARN_D    = {"daily H/L": 42, "7-day cone": 84, "3-class day type": 42}
    OVERDUE_D = {"daily H/L": 56, "7-day cone": 90, "3-class day type": 56}

    # ── Drift badge helper ────────────────────────────────────────────────────
    def _drift(v, ref, lo_better=True, tol=0.20):
        if ref is None or ref == 0:
            return "⚪"
        rel = (v - ref) / abs(ref) * (1 if lo_better else -1)
        return "🟢" if rel <= tol else ("🟡" if rel <= tol * 2 else "🔴")

    # ── Per-model age & performance status ───────────────────────────────────
    RANK = {"🔴": 3, "🟡": 2, "🟢": 1, "⚪": 0}

    def _worst(*icons):
        return max(icons, key=lambda x: RANK.get(x[:2], 0))

    def _age_status(key):
        te   = cutoffs.get(key)
        warn = WARN_D.get(key, 56)
        over = OVERDUE_D.get(key, 56)
        if te is None:
            return None, "⚪", "unknown age"
        age = (today_utc - te).days
        if age >= over:
            return age, "🔴", f"{age}d  (limit: {over}d)"
        if age >= warn:
            return age, "🟡", f"{age}d  (warn: {warn}d)"
        return age, "🟢", f"{age}d  (limit: {over}d)"

    def _perf_hl():
        if not hl30:
            return "⚪", "no data (< 5 bars)"
        m = hl30["mape_h"]
        if m > 2.5: return "🔴", f"MAPE-H = {m:.2f}%  (retrain > 2.5%)"
        if m > 2.0: return "🟡", f"MAPE-H = {m:.2f}%  (warn > 2.0%)"
        return "🟢", f"MAPE-H = {m:.2f}%"

    def _perf_cone():
        if not cone30:
            return "⚪", "no data (< 5 obs.)"
        cov = cone30["within_pct"]
        if cov < 78: return "🔴", f"coverage = {cov:.1f}%  (retrain < 78%)"
        if cov < 82: return "🟡", f"coverage = {cov:.1f}%  (warn < 82%)"
        return "🟢", f"coverage = {cov:.1f}%"

    def _perf_dt():
        if not dt30:
            return "⚪", "no data (< 5 bars)"
        acc = dt30["accuracy"]
        if acc < 50: return "🔴", f"accuracy = {acc:.1f}%  (retrain < 50%)"
        if acc < 55: return "🟡", f"accuracy = {acc:.1f}%  (warn < 55%)"
        return "🟢", f"accuracy = {acc:.1f}%"

    hl_age_d,   hl_age_ic,   hl_age_lbl  = _age_status("daily H/L")
    con_age_d,  con_age_ic,  con_age_lbl = _age_status("7-day cone")
    dt_age_d,   dt_age_ic,   dt_age_lbl  = _age_status("3-class day type")
    hl_perf_ic,  hl_perf_lbl  = _perf_hl()
    con_perf_ic, con_perf_lbl = _perf_cone()
    dt_perf_ic,  dt_perf_lbl  = _perf_dt()

    hl_comp  = _worst(hl_age_ic,  hl_perf_ic)
    con_comp = _worst(con_age_ic, con_perf_ic)
    dt_comp  = _worst(dt_age_ic,  dt_perf_ic)
    overall  = _worst(hl_comp, con_comp, dt_comp)

    # ── 1. Top-level banner ───────────────────────────────────────────────────
    if overall[:2] == "🔴":
        st.error(
            "🔴  **RETRAIN REQUIRED** — One or more TF2 strategy models are past their "
            "maintenance threshold.  U1 / D2 / D3 signal accuracy may be degraded.  "
            "See action plan below."
        )
    elif overall[:2] == "🟡":
        st.warning(
            "🟡  **PLAN RETRAIN** — Models are approaching their maintenance window.  "
            "Schedule a retrain within the next 1–2 weeks."
        )
    else:
        st.success(
            "🟢  **Models are current** — All TF2 strategy models are within their maintenance "
            "window and performance metrics are on target."
        )

    st.markdown("---")

    # ── 2. Model cards (3 columns) ────────────────────────────────────────────
    st.markdown("#### Model Freshness & Performance")
    st.caption(
        "Only the three models used by the TF2 strategy are shown.  "
        "Hourly close model and 14-day cone are **not** included — they are not used for "
        "trading signals.  Age limit = mandatory retrain by.  Warn = start planning."
    )

    BG  = {"🔴": "#fef2f2", "🟡": "#fffbeb", "🟢": "#f0fdf4", "⚪": "#f8fafc"}
    BRD = {"🔴": "#fca5a5", "🟡": "#fcd34d", "🟢": "#86efac", "⚪": "#e2e8f0"}
    TXT = {"🔴": "#991b1b", "🟡": "#92400e", "🟢": "#14532d", "⚪": "#374151"}
    ACTION = {
        "🔴": "RETRAIN NOW",
        "🟡": "DUE SOON — plan retrain",
        "🟢": "NO ACTION NEEDED",
        "⚪": "STATUS UNKNOWN",
    }

    def _card(icon, title, role, comp, age_lbl, train_end_ts, perf_ic, perf_lbl, extra=None):
        bg  = BG.get(comp[:2],  "#f8fafc")
        brd = BRD.get(comp[:2], "#e2e8f0")
        txt = TXT.get(comp[:2], "#374151")
        lbl = ACTION.get(comp[:2], comp)
        te  = str(pd.Timestamp(train_end_ts).date()) if train_end_ts else "unknown"
        rows = ""
        if extra:
            for line in extra:
                rows += f'<p style="margin:2px 0;font-size:11px;color:#6b7280;">{line}</p>'
        return (
            f'<div style="border:2px solid {brd};border-radius:10px;padding:16px;'
            f'background:{bg};min-height:230px;">'
            f'<div style="font-size:15px;font-weight:700;margin-bottom:2px;">{icon} {title}</div>'
            f'<div style="font-size:11px;font-weight:600;color:#6b7280;text-transform:uppercase;'
            f'letter-spacing:.05em;margin-bottom:10px;">{role}</div>'
            f'<div style="font-size:19px;font-weight:800;color:{txt};margin-bottom:10px;">'
            f'{comp[:2]} {lbl}</div>'
            f'<p style="margin:3px 0;font-size:13px;"><strong>Last trained:</strong> {te}</p>'
            f'<p style="margin:3px 0;font-size:13px;"><strong>Age:</strong> {age_lbl}</p>'
            f'{rows}'
            f'<hr style="border-color:{brd};margin:10px 0 8px;">'
            f'<p style="margin:0;font-size:12px;color:#475569;">'
            f'<strong>Perf (30d):</strong> {perf_ic} {perf_lbl}</p>'
            f'</div>'
        )

    hl_te  = cutoffs.get("daily H/L")
    con_te = cutoffs.get("7-day cone")
    dt_te  = cutoffs.get("3-class day type")

    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        st.markdown(
            _card("📅", "Daily H/L Model",
                  "PRIMARY · drives U1 / D2 / D3 signals",
                  hl_comp, hl_age_lbl, hl_te, hl_perf_ic, hl_perf_lbl,
                  extra=[f"Warn at {WARN_D['daily H/L']}d  ·  limit {OVERDUE_D['daily H/L']}d"]),
            unsafe_allow_html=True,
        )
    with cc2:
        st.markdown(
            _card("📐", "7-day Close Cone",
                  "SUPPORTING · regime classification",
                  con_comp, con_age_lbl, con_te, con_perf_ic, con_perf_lbl,
                  extra=[f"Warn at {WARN_D['7-day cone']}d  ·  limit {OVERDUE_D['7-day cone']}d"]),
            unsafe_allow_html=True,
        )
    with cc3:
        st.markdown(
            _card("🏷️", "3-class Day Type",
                  "CONTEXT · BigUpper / BigLower / Quiet",
                  dt_comp, dt_age_lbl, dt_te, dt_perf_ic, dt_perf_lbl,
                  extra=[
                      f"Warn at {WARN_D['3-class day type']}d  ·  limit {OVERDUE_D['3-class day type']}d",
                      "⚠️ Must retrain <em>after</em> Daily H/L",
                  ]),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # ── 3. Performance metrics ────────────────────────────────────────────────
    st.markdown("#### 30-day Performance vs Test Baseline")
    st.caption(
        "Drift badges: 🟢 ≤20% degradation vs held-out test · 🟡 20–40% · 🔴 >40%.  "
        "Hard retrain triggers: **MAPE-H >2.5%** · **coverage <78%** · **accuracy <50%**."
    )

    # 3a. Daily H/L — primary signal driver
    with st.expander("📅 Daily H/L — prediction accuracy (drives U1 / D2 / D3 error terms)",
                     expanded=True):
        if hl30 and _hl_mtest:
            mh30, ml30 = hl30["mape_h"], hl30["mape_l"]
            hh30, hl_30 = hl30["hit_h"], hl30["hit_l"]
            dh30 = hl30["dir_h"]
            mh_t = _hl_mtest.get("MAPE_H", 1.32)
            ml_t = _hl_mtest.get("MAPE_L", 1.30)
            hh_t = _hl_mtest.get("hit2_H", 79.7)
            hl_t = _hl_mtest.get("hit2_L", 87.9)
            dh_t = _hl_mtest.get("direction_hit_rate", 50.0)
            hl_c = st.columns(5)
            hl_c[0].metric(
                f"{_drift(mh30, mh_t)} MAPE-H (30d)",
                f"{mh30:.2f}%",
                delta=f"{mh30 - mh_t:+.2f} pp  (test: {mh_t:.2f}%)",
                delta_color="inverse",
                help=(
                    "**Primary signal driver.**  "
                    "`err_hi = (actual_high − pred_high) / close × 100`.  "
                    "**U1** fires when `err_hi_ma3 > +0.5%`.  "
                    "**D2** fires when `err_hi_ma3 < −1.0%`.  "
                    "Higher MAPE → more noise in these error terms → false signals."
                ),
            )
            hl_c[1].metric(
                f"{_drift(ml30, ml_t)} MAPE-L (30d)",
                f"{ml30:.2f}%",
                delta=f"{ml30 - ml_t:+.2f} pp  (test: {ml_t:.2f}%)",
                delta_color="inverse",
                help=(
                    "`err_lo = (pred_low − actual_low) / close × 100`.  "
                    "**D1** fires when `err_lo_ma3 > +0.5%` & `lo_breaks_3d ≥ 2`."
                ),
            )
            hl_c[2].metric(
                f"{_drift(hh30, hh_t, False)} Hit ±1.5% HIGH (30d)",
                f"{hh30:.1f}%",
                delta=f"{hh30 - hh_t:+.1f} pp  (test: {hh_t:.1f}%)",
                delta_color="normal",
            )
            hl_c[3].metric(
                f"{_drift(hl_30, hl_t, False)} Hit ±1.5% LOW (30d)",
                f"{hl_30:.1f}%",
                delta=f"{hl_30 - hl_t:+.1f} pp  (test: {hl_t:.1f}%)",
                delta_color="normal",
            )
            hl_c[4].metric(
                f"{_drift(dh30, dh_t, False)} Direction acc (30d)",
                f"{dh30:.1f}%",
                delta=f"{dh30 - dh_t:+.1f} pp  (test: {dh_t:.1f}%)",
                delta_color="normal",
                help="Day-over-day: did pred_high trend the same direction as actual_high?",
            )
        else:
            st.info("Insufficient data — need ≥5 completed daily bars in the last 30 days.")

    # 3b. 7-day cone — regime coverage
    with st.expander("📐 7-day Cone — regime band coverage", expanded=True):
        if cone30:
            cov30  = cone30["within_pct"]
            bp30   = cone30["band_pct"]
            da30   = cone30["dir_acc"]
            cov_r  = cone_at["within_pct"] if cone_at else _cone_meta.get("held_out_band_coverage_pct", 88.0)
            da_r   = cone_at["dir_acc"]    if cone_at else None
            _cic   = "🔴" if cov30 < 78 else ("🟡" if cov30 < 82 else "🟢")
            _daic  = ("⚪" if not da_r
                      else "🔴" if da30 < da_r * 0.80
                      else "🟡" if da30 < da_r * 0.90 else "🟢")
            cone_c = st.columns(4)
            cone_c[0].metric(
                f"{_cic} Coverage ±{bp30*100:.1f}% band (30d)",
                f"{cov30:.1f}%",
                delta=f"{cov30 - cov_r:+.1f} pp  (ref: {cov_r:.1f}%)",
                delta_color="normal",
                help=(
                    "Regime quality signal.  Fraction of 7-day forecasts where actual close "
                    "fell within the ±band.  Regime tercile thresholds become unreliable when "
                    "this drops below 80% — trigger to retrain the cone."
                ),
            )
            _da_delta = (f"{da30 - da_r:+.1f} pp  (ref: {da_r:.1f}%)" if da_r
                         else "no ref available")
            cone_c[1].metric(
                f"{_daic} Direction acc (30d)",
                f"{da30:.1f}%",
                delta=_da_delta,
                delta_color="normal",
            )
            cone_c[2].metric(
                "Resolved forecasts (30d)",
                str(cone30["n"]),
                help="Number of 7-day forecasts whose target date has passed in the last 30 days.",
            )
            # Regime distribution from test period
            if cone_at and cone_at.get("regime_pct"):
                rp = cone_at["regime_pct"]
                cone_c[3].metric(
                    "Regime mix (test period)",
                    (f"L:{float(rp.get(0,0)):.0f}%  "
                     f"M:{float(rp.get(1,0)):.0f}%  "
                     f"H:{float(rp.get(2,0)):.0f}%"),
                    help=(
                        "L=low vol (range_ma30 < 3.79%)  "
                        "M=mid (3.79–5.06%)  "
                        "H=high (>5.06%).  "
                        "A sustained shift toward H means the model's "
                        "median-return estimates are less reliable."
                    ),
                )
        else:
            st.info("Insufficient data — need ≥5 resolved 7-day forecasts in the last 30 days.")

    # 3c. 3-class day type — BigUpper / BigLower / Quiet
    with st.expander("🏷️ 3-class Day Type — BigUpper / BigLower / Quiet accuracy",
                     expanded=True):
        if dt30:
            acc30 = dt30["accuracy"]
            acc_r = (dt_at["accuracy"] if dt_at
                     else _dt_meta.get("test_accuracy_pct", 52.8))
            _aic  = "🔴" if acc30 < 50 else ("🟡" if acc30 < 55 else "🟢")
            _DI   = {"BigUpper": "🟢", "BigLower": "🔴", "Quiet": "⚪"}
            dt_c  = st.columns(4)
            dt_c[0].metric(
                f"{_aic} Overall accuracy (30d)",
                f"{acc30:.1f}%",
                delta=f"{acc30 - acc_r:+.1f} pp  (test: {acc_r:.1f}%)",
                delta_color="normal",
                help=(
                    "Retrain threshold: <50%.  "
                    "This model uses OOF predictions from Daily H/L as input features — "
                    "always retrain it immediately after retraining Daily H/L."
                ),
            )
            for ci, cls in enumerate(["BigUpper", "BigLower", "Quiet"]):
                bc   = dt30["by_class"][cls]
                prec = bc["correct_n"] / bc["pred_n"] * 100 if bc["pred_n"] else 0.0
                if dt_at:
                    bca    = dt_at["by_class"][cls]
                    prec_r = bca["correct_n"] / bca["pred_n"] * 100 if bca["pred_n"] else 0.0
                    _pi    = "🔴" if prec < 40 else ("🟡" if prec < 55 else "🟢")
                    dt_c[ci + 1].metric(
                        f"{_DI[cls]} {cls} precision",
                        f"{prec:.0f}%",
                        delta=f"{prec - prec_r:+.0f} pp  (test: {prec_r:.0f}%)",
                        delta_color="normal",
                        help=f"Of the last 30 days predicted as {cls}, this fraction was correct.",
                    )
                else:
                    dt_c[ci + 1].metric(f"{_DI[cls]} {cls}", f"{prec:.0f}% prec")
        else:
            st.info("Insufficient data — need ≥5 completed daily bars in the last 30 days.")

    st.markdown("---")

    # ── 4. Live signal state ──────────────────────────────────────────────────
    st.markdown("#### TF2 Signal State — Current Values vs Thresholds")
    st.caption(
        "Computed from the last 3–5 completed daily H/L bars.  "
        "Signal reliability degrades when MAPE-H exceeds ~2.5%."
    )

    _ehi  = sigs.get("err_hi_ma3")
    _elo  = sigs.get("err_lo_ma3")
    _hb3  = sigs.get("hi_breaks_3d")
    _lb3  = sigs.get("lo_breaks_3d")
    _dns  = sigs.get("dn_score_raw")
    _lle  = sigs.get("last_lo_err")
    _bull = sigs.get("bull_regime")
    _ma30 = sigs.get("ma30_value")
    _cur  = sigs.get("current_close_sig")

    def _sig_val(v, fmt="+.2f"):
        return (f"{v:{fmt}}%" if v is not None else "—")

    sc = st.columns(5)
    sc[0].metric(
        "🟢 U1 — Entry",
        "🔥 ACTIVE" if sigs.get("u1_triggered") else "💤 inactive",
        delta=(f"err_hi_ma3 {_sig_val(_ehi)}  |  {_hb3}/3 hi-breaks"
               if _ehi is not None else "threshold: err_hi_ma3 > +0.5%"),
        delta_color="off",
        help=(
            "**Threshold:** `err_hi_ma3 > +0.5%` AND `hi_breaks_3d ≥ 2`.  "
            "**Entry gate:** U1 confirmed by above-MA30, clean 10d, or V-reversal.  "
            "Statistical lift: 1.68× over base rate."
        ),
    )
    sc[1].metric(
        "🔴 D2 — Exit (strongest)",
        "🔥 ACTIVE" if sigs.get("d2_triggered") else "💤 inactive",
        delta=(f"err_hi_ma3 {_sig_val(_ehi)}"
               if _ehi is not None else "threshold: err_hi_ma3 < −1.0%"),
        delta_color="off",
        help=(
            "**Threshold:** `err_hi_ma3 < −1.0%`.  "
            "2.24× lift — strongest exit signal.  "
            "Active in BEAR/NEUTRAL regime; in BULL regime only D3 triggers exit."
        ),
    )
    sc[2].metric(
        "🔴 D3 — Exhaustion",
        "🔥 ACTIVE" if sigs.get("d3_triggered") else "💤 inactive",
        delta=("exhaustion fired" if sigs.get("d3_triggered") else "no hi-break streak → lo-break"),
        delta_color="off",
        help=(
            "First `lo_break` after ≥3 consecutive `hi_breaks`.  "
            "Trend-exhaustion canary.  Active in ALL regimes (BULL + BEAR)."
        ),
    )
    sc[3].metric(
        "V-Gate — Reversal",
        "🔥 ACTIVE" if sigs.get("v_reversal_likely") else "💤 inactive",
        delta=(
            (f"dn_score {_dns:.2f}  |  lo_err {_lle:+.1f}%"
             if _dns is not None and _lle is not None else "threshold: dn_score >0.8 & lo_err >5%")
        ),
        delta_color="off",
        help=(
            "**Threshold:** `dn_score > 0.8` AND `lo_err > +5%`.  "
            "Capitulation-reversal gate — allows U1 entry without MA30 confirmation."
        ),
    )
    _regime_str = "🐂 BULL" if _bull else ("🐻 BEAR/NEUTRAL" if _bull is not None else "—")
    _regime_delta = (
        f"close {_cur:,.0f}  /  MA30 {_ma30:,.0f}"
        if _ma30 is not None and _cur is not None
        else "price > MA30 & slope rising"
    )
    sc[4].metric(
        "Regime Filter",
        _regime_str,
        delta=_regime_delta,
        delta_color="off",
        help=(
            "**BULL** = price > 30-day MA AND MA30[now] > MA30[5d ago] (rising slope).  "
            "BULL: exit on D3 only.  BEAR/NEUTRAL: exit on D2 or D3."
        ),
    )

    st.markdown("---")

    # ── 5. Cone regime definitions ────────────────────────────────────────────
    st.markdown("#### 7-day Cone — Regime Definitions & Band")
    st.caption(
        "Tercile boundaries are fit on the training set.  "
        "If current `range_ma30` persistently sits outside these bounds, "
        "regime classification becomes unreliable — an additional trigger to retrain the cone."
    )

    _edges    = (_cone_art.get("regime_edges") if _cone_art else None) or [0.0379, 0.0506]
    _lo_edge  = float(_edges[0]) if len(_edges) >= 1 else 0.0379
    _hi_edge  = float(_edges[1]) if len(_edges) >= 2 else 0.0506
    _cone_bnd = float(_cone_art.get("band_pct", 0.097)) if _cone_art else 0.097
    _cov_oos  = _cone_meta.get("held_out_band_coverage_pct", 88.0)

    rd = st.columns(4)
    rd[0].metric(
        "Low vol regime",
        f"range_ma30 < {_lo_edge*100:.2f}%",
        delta="Train median 7d return: +1.06%",
        delta_color="off",
        help="Lowest volatility tercile. Daily H/L range averaged below 3.79% of close.",
    )
    rd[1].metric(
        "Mid vol regime",
        f"{_lo_edge*100:.2f}% – {_hi_edge*100:.2f}%",
        delta="Train median 7d return: −0.25%",
        delta_color="off",
    )
    rd[2].metric(
        "High vol regime",
        f"range_ma30 > {_hi_edge*100:.2f}%",
        delta="Train median 7d return: +1.31%",
        delta_color="off",
        help="Highest uncertainty. Cone is least reliable here.",
    )
    rd[3].metric(
        "Cone band width",
        f"±{_cone_bnd*100:.1f}%",
        delta=f"OOS coverage: {_cov_oos:.1f}%  (retrain if 30d < 80%)",
        delta_color="off",
        help="Fixed symmetric band derived from empirical return distribution.  "
             "Retraining recalibrates this width.",
    )

    st.markdown("---")

    # ── 6. Retrain action plan ────────────────────────────────────────────────
    st.markdown("#### Retrain Action Plan")
    urgent = [(ic, n) for ic, n in [(hl_comp, "Daily H/L"),
                                     (dt_comp, "3-class Day Type"),
                                     (con_comp, "7-day Cone")] if ic[:2] == "🔴"]
    plan   = [(ic, n) for ic, n in [(hl_comp, "Daily H/L"),
                                     (dt_comp, "3-class Day Type"),
                                     (con_comp, "7-day Cone")] if ic[:2] == "🟡"]
    if not urgent and not plan:
        st.success("✅  No retrain action required at this time.")
    else:
        if urgent:
            st.error("🔴  **Retrain immediately:** " + ", ".join(n for _, n in urgent))
        if plan:
            st.warning("🟡  **Plan retrain within 1–2 weeks:** " + ", ".join(n for _, n in plan))
        if any(n == "3-class Day Type" for _, n in urgent + plan):
            st.info(
                "ℹ️  3-class Day Type uses OOF predictions from Daily H/L as stacked input features.  "
                "Always run `pipeline_ct.py` first, then `train_3class_day_type.py`."
            )

    with st.expander(
        "📋 Retrain commands (run from repo root)",
        expanded=bool(urgent or plan),
    ):
        st.code(
            "# Step 1 — Daily H/L ensemble  (PRIMARY — always run first)\n"
            "python src/pipeline_ct.py\n\n"
            "# Step 2 — 3-class day-type classifier  (dependency on Step 1)\n"
            "python src/train_3class_day_type.py\n\n"
            "# Step 3 — 7-day close cone  (independent; run quarterly or when coverage < 80%)\n"
            "python src/train_7d_close_cone.py\n\n"
            "# After retraining: click 'Refresh now' in the sidebar to reload model artefacts.",
            language="bash",
        )

    st.caption(f"Dashboard status computed as of **{today_utc.date()} UTC**.")


# ════════════════════════════════════════════════════════════════════════
# EXPLAINABILITY DASHBOARD
# ════════════════════════════════════════════════════════════════════════

# Feature metadata: (category, sub-category, display label)
_EXPL_FEAT_META: "dict[str, tuple[str, str, str]]" = {
    # Price momentum / returns
    "ret_1h":   ("Price Momentum", "Returns", "1h BTC return"),
    "ret_2h":   ("Price Momentum", "Returns", "2h BTC return"),
    "ret_4h":   ("Price Momentum", "Returns", "4h BTC return"),
    "ret_8h":   ("Price Momentum", "Returns", "8h BTC return"),
    "ret_12h":  ("Price Momentum", "Returns", "12h BTC return"),
    "ret_24h":  ("Price Momentum", "Returns", "24h BTC return"),
    "ret_48h":  ("Price Momentum", "Returns", "48h BTC return"),
    "ret_72h":  ("Price Momentum", "Returns", "72h BTC return"),
    # Volatility
    "vol_4h":      ("Technical", "Volatility", "Realized vol 4h"),
    "vol_8h":      ("Technical", "Volatility", "Realized vol 8h"),
    "vol_24h":     ("Technical", "Volatility", "Realized vol 24h"),
    "vol_48h":     ("Technical", "Volatility", "Realized vol 48h"),
    "atr_4h":      ("Technical", "Volatility", "ATR 4h"),
    "atr_12h":     ("Technical", "Volatility", "ATR 12h"),
    "atr_24h":     ("Technical", "Volatility", "ATR 24h"),
    "bb24_width":  ("Technical", "Volatility", "Bollinger width 24h"),
    # Price level / range
    "range_now":   ("Technical", "Price Level", "Current bar range"),
    "range_ma24":  ("Technical", "Price Level", "Range MA 24h"),
    "range_ma72":  ("Technical", "Price Level", "Range MA 72h"),
    "dist_hi_24":  ("Technical", "Price Level", "Dist from 24h high"),
    "dist_lo_24":  ("Technical", "Price Level", "Dist from 24h low"),
    "dist_hi_168": ("Technical", "Price Level", "Dist from 7d high"),
    # Volume
    "vol_chg_1": ("Technical", "Volume", "Volume change 1h"),
    "vol_z_24":  ("Technical", "Volume", "Volume z-score 24h"),
    # Oscillators
    "rsi_14":    ("Technical", "Oscillators", "RSI(14)"),
    "macd":      ("Technical", "Oscillators", "MACD"),
    "macd_hist": ("Technical", "Oscillators", "MACD histogram"),
    # ETH / crypto ecosystem
    "eth_ret_1h":      ("Crypto Market", "Ethereum", "ETH 1h return"),
    "eth_ret_24h":     ("Crypto Market", "Ethereum", "ETH 24h return"),
    "eth_vol_24h":     ("Crypto Market", "Ethereum", "ETH vol 24h"),
    "btc_eth_corr_24": ("Crypto Market", "Ethereum", "BTC-ETH corr 24h"),
    # Equities
    "spx_ret_1h":  ("Macro/Market", "Equities", "S&P 500 1h return"),
    "spx_ret_24h": ("Macro/Market", "Equities", "S&P 500 24h return"),
    "spx_vol_24h": ("Macro/Market", "Equities", "S&P 500 vol 24h"),
    "ndx_ret_1h":  ("Macro/Market", "Equities", "Nasdaq 1h return"),
    "ndx_ret_24h": ("Macro/Market", "Equities", "Nasdaq 24h return"),
    "ndx_vol_24h": ("Macro/Market", "Equities", "Nasdaq vol 24h"),
    # Risk sentiment / VIX
    "vix_ret_1h":  ("Macro/Market", "Risk Sentiment", "VIX 1h change"),
    "vix_ret_24h": ("Macro/Market", "Risk Sentiment", "VIX 24h change"),
    "vix_vol_24h": ("Macro/Market", "Risk Sentiment", "VIX vol 24h"),
    # Commodities / safe havens
    "gold_ret_1h":  ("Macro/Market", "Commodities", "Gold 1h return"),
    "gold_ret_24h": ("Macro/Market", "Commodities", "Gold 24h return"),
    "gold_vol_24h": ("Macro/Market", "Commodities", "Gold vol 24h"),
    # USD / dollar index
    "dxy_ret_1h":  ("Macro/Market", "USD Index", "DXY 1h change"),
    "dxy_ret_24h": ("Macro/Market", "USD Index", "DXY 24h change"),
    "dxy_vol_24h": ("Macro/Market", "USD Index", "DXY vol 24h"),
    # Interest rates
    "tnx_ret_1h":  ("Macro/Market", "Rates", "10Y yield 1h change"),
    "tnx_ret_24h": ("Macro/Market", "Rates", "10Y yield 24h change"),
    "tnx_vol_24h": ("Macro/Market", "Rates", "10Y yield vol 24h"),
    # Sentiment
    "fng":     ("Sentiment", "Fear & Greed", "F&G index level"),
    "fng_d1":  ("Sentiment", "Fear & Greed", "F&G 1d change"),
    "fng_d7":  ("Sentiment", "Fear & Greed", "F&G 7d change"),
    "fng_d24": ("Sentiment", "Fear & Greed", "F&G 24h change"),
    # Calendar / seasonality
    "hr_sin":  ("Seasonality", "Time", "Hour-of-day (sin)"),
    "hr_cos":  ("Seasonality", "Time", "Hour-of-day (cos)"),
    "dow_sin": ("Seasonality", "Time", "Day-of-week (sin)"),
    "dow_cos": ("Seasonality", "Time", "Day-of-week (cos)"),
    "weekend": ("Seasonality", "Time", "Weekend flag"),
    "us_open": ("Seasonality", "Time", "US market hours"),
}

# Category display config: color, emoji, description
_EXPL_CAT_META = {
    "Price Momentum": {
        "color": "#3b82f6", "emoji": "📈",
        "desc": "BTC log-returns across multiple timeframes (1h → 72h)",
    },
    "Technical": {
        "color": "#8b5cf6", "emoji": "📐",
        "desc": "Realized volatility, ATR, range, volume, RSI, MACD, Bollinger",
    },
    "Crypto Market": {
        "color": "#f59e0b", "emoji": "⚡",
        "desc": "Ethereum price action and BTC-ETH correlation",
    },
    "Macro/Market": {
        "color": "#ef4444", "emoji": "🌍",
        "desc": "Equities (SPX/NDX), fear index (VIX), gold, USD (DXY), 10Y rates",
    },
    "Sentiment": {
        "color": "#10b981", "emoji": "😨",
        "desc": "Crypto Fear & Greed index — level and short-term changes",
    },
    "Seasonality": {
        "color": "#6b7280", "emoji": "🕐",
        "desc": "Hour-of-day and day-of-week cyclical patterns",
    },
}


@st.cache_data(ttl=300, show_spinner=False)
def _compute_expl_data(latest_t_iso: str) -> "dict | None":
    """Compute signed feature contributions for the current bar.

    For Ridge: contribution_i = coef_i * z_i  (exact linear decomposition)
    For GBM:  contribution_i = pred(x) − pred(x | x_i = μ_i)  (perturbation)

    Both methods use the StandardScaler's training-time mean as the 'neutral'
    baseline so contributions represent deviation from historical norms.
    """
    t = pd.Timestamp(latest_t_iso)
    if t not in F_filled.index:
        return None
    x_row = F_filled.loc[t, feat_cols]
    if x_row.isna().any():
        return None
    x_arr = x_row.values.reshape(1, -1)

    scaler = model.named_steps["sc"]
    inner  = model.named_steps["m"]
    mu_train    = pd.Series(scaler.mean_,  index=feat_cols)
    sigma_train = pd.Series(scaler.scale_, index=feat_cols)
    z_scores = (x_row - mu_train) / sigma_train.replace(0, 1e-10)

    pred_now = float(model.predict(x_arr)[0])

    if best_name == "ridge":
        coef = pd.Series(inner.coef_, index=feat_cols)
        contributions = coef * z_scores
    else:
        # Perturbation: zero out each feature's deviation from training mean
        x_flat = x_row.values.copy().astype(float)
        contribs_dict: "dict[str, float]" = {}
        for i, feat in enumerate(feat_cols):
            x_pert = x_flat.copy()
            x_pert[i] = float(mu_train.iloc[i])
            pred_pert = float(model.predict(x_pert.reshape(1, -1))[0])
            contribs_dict[feat] = pred_now - pred_pert
        contributions = pd.Series(contribs_dict)

    # Regime context: last 30 days of BTC prices
    recent_idx = F_filled.index[
        valid_mask & (F_filled.index <= t) &
        (F_filled.index >= t - pd.Timedelta(days=30))
    ]
    if len(recent_idx) >= 24:
        btc_recent = df["btc_close"].reindex(recent_idx).dropna()
        cur_price  = float(btc_recent.iloc[-1])
        ma30_price = float(btc_recent.mean())
        pct_vs_ma  = (cur_price - ma30_price) / ma30_price * 100
        ret_30d    = float((btc_recent.iloc[-1] / btc_recent.iloc[0] - 1) * 100)
        n7 = min(7 * 24, len(btc_recent))
        ret_7d = float((btc_recent.iloc[-1] / btc_recent.iloc[-n7] - 1) * 100)
        btc_hist_idx = list(btc_recent.index)
        btc_hist_vals = list(btc_recent.values)
        ma_hist_vals = list(
            btc_recent.rolling(min(len(btc_recent), 168), min_periods=24).mean().values
        )
    else:
        cur_price = ma30_price = pct_vs_ma = ret_30d = ret_7d = None
        btc_hist_idx = btc_hist_vals = ma_hist_vals = []

    fng_now = float(x_row["fng"]) if "fng" in feat_cols else 50.0

    # Regime score
    score = 0
    if pct_vs_ma is not None:
        if   pct_vs_ma >  10: score += 3
        elif pct_vs_ma >   5: score += 2
        elif pct_vs_ma >   0: score += 1
        elif pct_vs_ma >  -5: score -= 1
        elif pct_vs_ma > -10: score -= 2
        else:                  score -= 3
    if ret_7d is not None:
        if   ret_7d >  15: score += 2
        elif ret_7d >   5: score += 1
        elif ret_7d <  -5: score -= 1
        elif ret_7d < -15: score -= 2
    if fng_now >= 75: score += 1
    elif fng_now <= 25: score -= 1

    if   score >= 3: regime = "Strong Bull"
    elif score >= 1: regime = "Bull"
    elif score == 0: regime = "Neutral"
    elif score >= -2: regime = "Bear"
    else:             regime = "Strong Bear"

    return dict(
        contributions  = contributions,
        z_scores       = z_scores,
        pred_return    = pred_now,
        regime         = regime,
        regime_score   = score,
        cur_price      = cur_price,
        ma30_price     = ma30_price,
        pct_vs_ma      = pct_vs_ma,
        ret_30d        = ret_30d,
        ret_7d         = ret_7d,
        fng_now        = fng_now,
        btc_hist_idx   = btc_hist_idx,
        btc_hist_vals  = btc_hist_vals,
        ma_hist_vals   = ma_hist_vals,
        as_of          = t,
    )


def render_explainability_dashboard():
    """Render the Explainability tab.

    Shows what macro, on-chain, technical, and sentiment factors are currently
    driving Bitcoin up or down, how strongly, and in what regime context.
    """
    st.markdown("## 🧠 What's Driving Bitcoin Right Now?")
    st.caption(
        "Feature contributions decompose the model's next-hour forecast into "
        "per-factor shares. **Green** = pushing the forecast higher · "
        "**Red** = pushing the forecast lower. "
        f"Model in use: **{best_name.upper()}** (hourly close predictor). "
        "Contributions are exact for Ridge (coeff × z-score); perturbation-based for GBM."
    )

    expl = _compute_expl_data(str(latest_t_global))
    if expl is None:
        st.warning("Insufficient data to compute feature contributions. "
                   "Click **Refresh now** in the sidebar to retry.")
        return

    contributions: pd.Series = expl["contributions"]
    z_scores: pd.Series      = expl["z_scores"]
    pred_ret: float          = expl["pred_return"]
    regime: str              = expl["regime"]

    # ── Regime banner ────────────────────────────────────────────────────
    _REGIME_STYLES = {
        "Strong Bull": ("#14532d", "#dcfce7", "#166534"),
        "Bull":        ("#14532d", "#f0fdf4", "#16a34a"),
        "Neutral":     ("#713f12", "#fefce8", "#ca8a04"),
        "Bear":        ("#7f1d1d", "#fef2f2", "#dc2626"),
        "Strong Bear": ("#450a0a", "#fee2e2", "#991b1b"),
    }
    _REGIME_EMOJI = {
        "Strong Bull": "🐂🐂", "Bull": "🐂",
        "Neutral": "〰️",
        "Bear": "🐻", "Strong Bear": "🐻🐻",
    }
    text_col, bg_col, border_col = _REGIME_STYLES.get(regime, ("#111", "#f9fafb", "#6b7280"))
    r_emoji = _REGIME_EMOJI.get(regime, "")
    fc_color = "#16a34a" if pred_ret > 0 else "#dc2626"
    fc_arrow = "▲" if pred_ret > 0 else "▼"

    st.markdown(
        f"""<div style="background:{bg_col};border-radius:10px;padding:14px 22px;
            border-left:5px solid {border_col};margin-bottom:12px;">
          <span style="font-size:20px;font-weight:700;color:{text_col};">
            {r_emoji}&nbsp;{regime} Regime
          </span>
          <span style="font-size:13px;color:{text_col};margin-left:18px;">
            1-hour model forecast:&nbsp;
            <b style="color:{fc_color};">{fc_arrow} {pred_ret*100:+.3f}%</b>
            &nbsp;log-return
          </span>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── KPI metrics row ──────────────────────────────────────────────────
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    with mc1:
        cp = expl.get("cur_price")
        r7 = expl.get("ret_7d")
        st.metric("BTC Price",
                  f"${cp:,.0f}" if cp else "—",
                  delta=f"{r7:+.1f}% 7d" if r7 is not None else None)
    with mc2:
        pvm = expl.get("pct_vs_ma")
        r30 = expl.get("ret_30d")
        st.metric("vs 30-day MA",
                  f"{pvm:+.1f}%" if pvm is not None else "—",
                  delta=f"{r30:+.1f}% 30d" if r30 is not None else None)
    with mc3:
        fng = expl.get("fng_now", 50)
        fng_label = (
            "Extreme Greed" if fng >= 75 else "Greed"    if fng >= 55 else
            "Neutral"       if fng >= 45 else "Fear"     if fng >= 25 else "Extreme Fear"
        )
        st.metric("Fear & Greed", f"{fng:.0f}", delta=fng_label,
                  delta_color="off")
    with mc4:
        st.metric("Hourly Forecast",
                  f"{fc_arrow} {abs(pred_ret*100):.3f}%",
                  delta="bullish" if pred_ret > 0 else "bearish",
                  delta_color="normal" if pred_ret > 0 else "inverse")
    with mc5:
        bull_c = contributions[contributions > 0]
        bear_c = contributions[contributions < 0]
        top_b  = bull_c.idxmax() if len(bull_c) else None
        top_be = bear_c.idxmin() if len(bear_c) else None
        top_b_label  = _EXPL_FEAT_META.get(top_b,  ("", "", str(top_b)))[2]  if top_b  else "—"
        top_be_label = _EXPL_FEAT_META.get(top_be, ("", "", str(top_be)))[2] if top_be else "—"
        st.markdown(
            f"<div style='font-size:12px;line-height:1.8'>"
            f"🟢 <b>Top bull:</b> {top_b_label}<br>"
            f"🔴 <b>Top bear:</b> {top_be_label}"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Main feature contribution chart ──────────────────────────────────
    st.markdown("### 📊 Feature Drivers — Current Bar")
    st.caption(
        "Top features sorted by absolute contribution.  "
        "Width = strength of influence.  "
        "Hover for current value, z-score and exact contribution."
    )

    N_SHOW = min(30, len(contributions))
    top_idx = contributions.abs().nlargest(N_SHOW).index
    c_show  = contributions[top_idx].sort_values()

    bar_colors, hover_texts, y_labels = [], [], []
    for feat in c_show.index:
        meta     = _EXPL_FEAT_META.get(feat, ("Other", feat, feat))
        cat_meta = _EXPL_CAT_META.get(meta[0], {"color": "#6b7280"})
        bar_colors.append("#16a34a" if c_show[feat] >= 0 else "#dc2626")
        z_val  = float(z_scores.get(feat, 0))
        raw_val = (float(F_filled.loc[latest_t_global, feat])
                   if feat in F_filled.columns else float("nan"))
        contrib_bps = c_show[feat] * 100
        hover_texts.append(
            f"<b>{meta[2]}</b><br>"
            f"Category: {meta[0]} › {meta[1]}<br>"
            f"Current value: {raw_val:.5g}<br>"
            f"Z-score vs training mean: {z_val:+.2f}σ<br>"
            f"Contribution: {contrib_bps:+.5f} bps"
        )
        y_labels.append(meta[2])

    fig_main = go.Figure(go.Bar(
        y=y_labels,
        x=list(c_show.values * 100),
        orientation="h",
        marker_color=bar_colors,
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_texts,
        text=[f"{v*100:+.4f}" for v in c_show.values],
        textposition="outside",
    ))
    fig_main.add_vline(x=0, line=dict(color="#111827", width=1.5))
    fig_main.update_layout(
        height=max(420, N_SHOW * 22 + 100),
        template="plotly_white",
        title=dict(
            text=(
                f"<b>Top {N_SHOW} drivers  ·  {latest_t_global.strftime('%Y-%m-%d %H:%M')} UTC</b>"
                "<br><span style='font-size:11px;color:#555'>"
                "Green = bullish contribution  ·  Red = bearish contribution  ·  "
                f"Total predicted log-return: <b>{pred_ret*100:+.4f}%</b>"
                "</span>"
            ),
            x=0.0, xanchor="left",
        ),
        xaxis_title="Contribution (predicted log-return ×100, basis-point scale)",
        yaxis_title=None,
        margin=dict(t=80, r=130, b=50, l=210),
        yaxis=dict(tickfont=dict(size=11)),
        xaxis=dict(zeroline=False),
    )
    st.plotly_chart(fig_main, use_container_width=True, key="expl_main_chart")

    # ── Category breakdown + regime context ──────────────────────────────
    st.markdown("### 📂 Factor Category Breakdown")
    col_cats, col_regime = st.columns([3, 2])

    # Aggregate by top-level category
    cat_pos: "dict[str, float]" = {}
    cat_neg: "dict[str, float]" = {}
    cat_net: "dict[str, float]" = {}
    for feat, val in contributions.items():
        cat = _EXPL_FEAT_META.get(str(feat), ("Other", feat, feat))[0]
        bps = float(val) * 100
        cat_pos[cat] = cat_pos.get(cat, 0.0) + max(0.0, bps)
        cat_neg[cat] = cat_neg.get(cat, 0.0) + min(0.0, bps)
        cat_net[cat] = cat_net.get(cat, 0.0) + bps

    cats_sorted = sorted(cat_net, key=lambda c: abs(cat_net[c]), reverse=True)

    with col_cats:
        c_emojis  = [f"{_EXPL_CAT_META.get(c, {}).get('emoji', '')} {c}" for c in cats_sorted]
        pos_vals  = [cat_pos.get(c, 0.0) for c in cats_sorted]
        neg_vals  = [cat_neg.get(c, 0.0) for c in cats_sorted]

        fig_cats = go.Figure()
        fig_cats.add_trace(go.Bar(
            name="Bullish", y=c_emojis, x=pos_vals, orientation="h",
            marker_color="#16a34a",
            hovertemplate="%{y}: +%{x:.4f} bps<extra>Bullish</extra>",
        ))
        fig_cats.add_trace(go.Bar(
            name="Bearish", y=c_emojis, x=neg_vals, orientation="h",
            marker_color="#dc2626",
            hovertemplate="%{y}: %{x:.4f} bps<extra>Bearish</extra>",
        ))
        fig_cats.add_vline(x=0, line=dict(color="#111827", width=1))
        fig_cats.update_layout(
            barmode="relative",
            height=320,
            template="plotly_white",
            title=dict(
                text="<b>Net bullish vs bearish by category</b>",
                x=0, xanchor="left",
            ),
            xaxis_title="Contribution (bps)",
            yaxis_title=None,
            margin=dict(t=50, r=40, b=50, l=170),
            legend=dict(orientation="h", y=-0.18),
            yaxis=dict(tickfont=dict(size=11)),
        )
        st.plotly_chart(fig_cats, use_container_width=True, key="expl_cats_chart")

    with col_regime:
        btc_idx  = expl.get("btc_hist_idx", [])
        btc_vals = expl.get("btc_hist_vals", [])
        ma_vals  = expl.get("ma_hist_vals", [])
        if btc_idx and len(btc_idx) >= 24:
            fig_reg = go.Figure()
            # Conditionally coloured fill: green where price > MA, red where below.
            # Strategy: two zero-height-when-inactive traces using clamped arrays —
            # no need to find exact crossover points.
            _btc_arr = np.array(btc_vals, dtype=float)
            _ma_arr  = np.array(ma_vals,  dtype=float)
            # Where MA is NaN (rolling warm-up), treat as equal to price → zero fill
            _ma_safe = np.where(np.isnan(_ma_arr), _btc_arr, _ma_arr)
            _top_grn = np.where(_btc_arr > _ma_safe, _btc_arr, _ma_safe)
            _bot_red = np.where(_btc_arr < _ma_safe, _btc_arr, _ma_safe)
            _x_rev   = btc_idx[::-1]
            # Green fill (above MA)
            fig_reg.add_trace(go.Scatter(
                x=btc_idx + _x_rev,
                y=list(_top_grn) + list(reversed(_ma_safe)),
                fill="toself", fillcolor="rgba(22,163,74,0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False, hoverinfo="skip",
            ))
            # Red fill (below MA)
            fig_reg.add_trace(go.Scatter(
                x=btc_idx + _x_rev,
                y=list(_ma_safe) + list(reversed(_bot_red)),
                fill="toself", fillcolor="rgba(220,38,38,0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False, hoverinfo="skip",
            ))
            fig_reg.add_trace(go.Scatter(
                x=btc_idx, y=ma_vals,
                name="30d MA", line=dict(color="#6b7280", width=1.5, dash="dash"),
                hovertemplate="MA: $%{y:,.0f}<extra></extra>",
            ))
            fig_reg.add_trace(go.Scatter(
                x=btc_idx, y=btc_vals,
                name="BTC Price", line=dict(color="#F7931A", width=2),
                hovertemplate="BTC: $%{y:,.0f}<extra></extra>",
            ))
            fig_reg.update_layout(
                height=320,
                template="plotly_white",
                title=dict(
                    text="<b>BTC vs 30-day MA (regime context)</b>",
                    x=0, xanchor="left",
                ),
                xaxis_title=None,
                yaxis_title="Price (USD)",
                yaxis_tickformat="$,.0f",
                margin=dict(t=50, r=20, b=40, l=90),
                legend=dict(orientation="h", y=-0.18),
            )
            st.plotly_chart(fig_reg, use_container_width=True, key="expl_regime_chart")
        else:
            st.info("Insufficient price history for regime chart.")

    # ── Category narrative ────────────────────────────────────────────────
    st.markdown("### 🗒️ Category Narratives")
    ncols = 3
    cat_chunks = [cats_sorted[i:i+ncols] for i in range(0, len(cats_sorted), ncols)]
    for chunk in cat_chunks:
        row_cols = st.columns(ncols)
        for col, cat in zip(row_cols, chunk):
            cm = _EXPL_CAT_META.get(cat, {"emoji": "", "color": "#6b7280", "desc": ""})
            net = cat_net.get(cat, 0.0)
            direction = "🟢 Bullish" if net > 0.001 else ("🔴 Bearish" if net < -0.001 else "⚪ Neutral")
            cat_feats = sorted(
                [(f, float(contributions[f])) for f in contributions.index
                 if _EXPL_FEAT_META.get(str(f), ("",))[0] == cat],
                key=lambda x: abs(x[1]), reverse=True,
            )[:3]
            top_lines = "  \n".join(
                f"{'▲' if v > 0 else '▼'} {_EXPL_FEAT_META.get(f, ('','',f))[2]}: "
                f"{v*100:+.4f} bps"
                for f, v in cat_feats
            ) or "_No features_"
            with col:
                st.markdown(
                    f"**{cm['emoji']} {cat}**  \n"
                    f"{direction} · **{net:+.4f} bps**  \n"
                    f"_{cm['desc']}_  \n\n"
                    f"{top_lines}"
                )

    st.markdown("---")

    # ── Z-score heatmap (current feature deviations from training norms) ─
    with st.expander("📡 Feature Z-Scores vs Training Baseline", expanded=False):
        st.caption(
            "How far each feature's current value is from its training-time mean.  "
            "🔴 |z| > 2 = unusual · 🟡 |z| > 1 = elevated · 🟢 |z| ≤ 1 = normal."
        )
        z_top = z_scores.abs().nlargest(min(30, len(z_scores))).index
        z_show = z_scores[z_top].sort_values()
        z_labels = [_EXPL_FEAT_META.get(f, ("","",f))[2] for f in z_show.index]
        z_colors = [
            "#dc2626" if abs(v) > 2 else "#f59e0b" if abs(v) > 1 else "#16a34a"
            for v in z_show.values
        ]
        fig_z = go.Figure(go.Bar(
            y=z_labels, x=list(z_show.values),
            orientation="h",
            marker_color=z_colors,
            text=[f"{v:+.2f}σ" for v in z_show.values],
            textposition="outside",
            hovertemplate="%{y}: %{x:+.3f}σ<extra></extra>",
        ))
        fig_z.add_vline(x=0,  line=dict(color="#111827", width=1))
        fig_z.add_vline(x= 2, line=dict(color="#dc2626", width=1.2, dash="dash"))
        fig_z.add_vline(x=-2, line=dict(color="#dc2626", width=1.2, dash="dash"))
        fig_z.add_vline(x= 1, line=dict(color="#f59e0b", width=1,   dash="dot"))
        fig_z.add_vline(x=-1, line=dict(color="#f59e0b", width=1,   dash="dot"))
        fig_z.update_layout(
            height=max(380, len(z_show) * 22 + 100),
            template="plotly_white",
            title=dict(
                text=(
                    "<b>Feature z-scores vs training-time baseline</b>"
                    "<br><span style='font-size:11px;color:#555'>"
                    "z = (current − training mean) / training std  ·  "
                    "🔴 |z|>2 unusual · 🟡 |z|>1 elevated · 🟢 normal"
                    "</span>"
                ),
                x=0, xanchor="left",
            ),
            xaxis_title="z-score (standard deviations from training mean)",
            yaxis_title=None,
            margin=dict(t=80, r=100, b=50, l=210),
            yaxis=dict(tickfont=dict(size=11)),
        )
        n_red = int((z_scores.abs() > 2).sum())
        n_yel = int(((z_scores.abs() > 1) & (z_scores.abs() <= 2)).sum())
        n_grn = int((z_scores.abs() <= 1).sum())
        st.plotly_chart(fig_z, use_container_width=True, key="expl_z_chart")
        st.caption(
            f"🔴 {n_red} features unusually deviated (|z|>2)  ·  "
            f"🟡 {n_yel} elevated (1<|z|≤2)  ·  "
            f"🟢 {n_grn} within normal range (|z|≤1)"
        )

    # ── Full feature table ───────────────────────────────────────────────
    with st.expander("🔍 Full feature contribution table", expanded=False):
        rows = []
        for feat in feat_cols:
            meta     = _EXPL_FEAT_META.get(feat, ("Other", feat, feat))
            cat_meta = _EXPL_CAT_META.get(meta[0], {"emoji": ""})
            contrib  = float(contributions.get(feat, 0.0))
            z_val    = float(z_scores.get(feat, 0.0))
            raw_val  = (float(F_filled.loc[latest_t_global, feat])
                        if feat in F_filled.columns else float("nan"))
            rows.append({
                "Category":            f"{cat_meta.get('emoji','')} {meta[0]}",
                "Sub-category":        meta[1],
                "Feature":             meta[2],
                "Current Value":       round(raw_val, 6),
                "Z-Score (σ)":         round(z_val, 3),
                "Contribution (bps)":  round(contrib * 100, 6),
                "Direction":           "▲ Bull" if contrib > 1e-9 else ("▼ Bear" if contrib < -1e-9 else "—"),
            })
        df_expl_table = pd.DataFrame(rows)
        df_expl_table = df_expl_table.sort_values(
            "Contribution (bps)", key=abs, ascending=False
        ).reset_index(drop=True)
        st.dataframe(
            df_expl_table,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Current Value":      st.column_config.NumberColumn(format="%.6g"),
                "Z-Score (σ)":        st.column_config.NumberColumn(format="%.3f"),
                "Contribution (bps)": st.column_config.NumberColumn(format="%.6f"),
            },
        )
        st.caption(
            f"Model: **{best_name.upper()}** · "
            f"As of: **{latest_t_global.strftime('%Y-%m-%d %H:%M')} UTC** · "
            f"Total forecast: **{pred_ret*100:+.5f}%** log-return · "
            f"{len(feat_cols)} features in model"
        )


# ════════════════════════════════════════════════════════════════════════
# Tabs: Live | Historical | Retrain Status | MSTR Backtesting | Explainability
# ════════════════════════════════════════════════════════════════════════
tab_live, tab_hist, tab_retrain, tab_mstr, tab_explain = st.tabs([
    "🔴 Live (rolling now+1h)",
    "🕒 Historical replay",
    "🔄 Retrain Status",
    "📊 MSTR Backtesting",
    "🧠 Explainability",
])

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

with tab_retrain:
    render_retrain_dashboard()

with tab_mstr:
    st.markdown("## 📊 MSTR — BTC Signal-Driven Backtesting")
    st.markdown(
        "This tab runs the **same TF2 + V-Gate strategy** used for Bitcoin but executes "
        "trades in **MSTR (MicroStrategy) stock** instead of BTC. All entry/exit signals "
        "are computed from the BTC CT model predictions — only the traded asset changes. "
        "MSTR holds ~580,000 BTC on its balance sheet and behaves as a leveraged Bitcoin "
        "proxy with equity-market tax treatment."
    )
    _mstr_model_mtime = (float(os.path.getmtime(str(DAILY_MODEL_CT)))
                         if os.path.exists(str(DAILY_MODEL_CT)) else 0.0)
    _mstr_today_iso   = pd.Timestamp.now(tz="UTC").normalize().strftime("%Y-%m-%d")
    _mstr_bear     = _run_fixed_period_mstr_backtest(
        _BT_BEAR_END_LOCKED, "2025-06-01", _mstr_model_mtime)
    _mstr_bull     = _run_fixed_period_mstr_backtest(
        "2025-06-14", "2024-06-05", _mstr_model_mtime)
    _mstr_full_oos = run_mstr_backtest(
        _mstr_today_iso, model_mtime=_mstr_model_mtime)   # rolls daily
    _mstr_full     = _run_fixed_period_mstr_backtest(
        _BT_FULL_END_LOCKED, "2024-06-05", _mstr_model_mtime)
    render_mstr_trading_strategy_dashboard(
        _mstr_bear, _mstr_bull,
        bt_full_oos=_mstr_full_oos,
        bt_full=_mstr_full,
        key_suffix="mstr_tab",
    )

with tab_explain:
    render_explainability_dashboard()

# ─────────────────────── timer-driven re-run ──────────────────────────
time.sleep(REFRESH_SECONDS)
st.rerun()
