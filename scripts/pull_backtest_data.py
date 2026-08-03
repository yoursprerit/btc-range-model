#!/usr/bin/env python3
"""
Pull ALL data used by the CT model for backtesting:
  - BTC-USD OHLCV            (yfinance)
  - 7 macro series           (yfinance: ETH, SPX, NDX, VIX, Gold, DXY, TNX)
  - 11 on-chain metrics      (blockchain.info)
  - Coinbase BTC premium     (Coinbase Exchange API)
  - MSTR OHLCV               (yfinance)
  - MSTU OHLCV               (yfinance, post-inception Sep 18 2024)
  - MSTU synthetic prices    (OLS on MSTR log-returns)
  - ETH-USD OHLCV            (Binance, 12:00-UTC bars — the spot-ETH sleeve
                              traded off the BTC CT signal; executed as ETHA)

Saves versioned CSVs to data/backtest/ and updates manifest.json.

Run manually to refresh the dataset:
    python scripts/pull_backtest_data.py

The UI reads manifest.json to display the dataset version badge and uses
raw_features_daily.csv to build CT model predictions without live API calls.
"""

import hashlib, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

REPO_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR   = REPO_ROOT / "data" / "backtest"
INCEPTION  = pd.Timestamp("2024-09-18")   # MSTU first trading day
FETCH_FROM = "2023-11-01"                 # 200+ days before earliest backtest start

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
_CB_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"

# Binance public hosts for hourly klines. `api.binance.com` returns HTTP 451 in
# some regions; `api.binance.us` and the `data-api.binance.vision` mirror serve
# the same public klines and are tried first.
_BINANCE_HOSTS = ("https://api.binance.us", "https://data-api.binance.vision",
                  "https://api.binance.com")
ANCHOR_HOUR_UTC = 12  # 7am CDT / 6am CST — the daily model's bar boundary

# ─── helpers ────────────────────────────────────────────────────────────────

def _binance_klines(start_ms: int, limit: int = 1000, timeout: int = 30,
                    symbol: str = "BTCUSDT"):
    """GET /api/v3/klines from the first Binance host that answers 200."""
    for host in _BINANCE_HOSTS:
        try:
            r = requests.get(host + "/api/v3/klines",
                             params=dict(symbol=symbol, interval="1h",
                                         startTime=start_ms, limit=limit),
                             timeout=timeout)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list):
                    return data
        except Exception:
            continue
    return None


def fetch_12utc(start_iso: str, symbol: str = "BTCUSDT") -> pd.DataFrame:
    """Daily OHLCV anchored at 12:00 UTC (7am CT), rebucketed from Binance
    hourly klines — the SAME bar boundary as the live app (_rebucket_12utc) and
    the daily H/L model. Replaces yfinance's midnight-UTC daily bars so the
    versioned dataset is consistent with the Live and Historical views.

    Used for BTC (the signal asset) and ETH (the spot-ETH sleeve): trading the
    ETH sleeve off a 12:00-UTC bar means its fill happens at the very moment the
    BTC signal bar closes — a genuine same-bar fill, which a US-hours ETF price
    could only approximate.

    Volume is the summed Binance base-asset volume; every volume feature
    downstream (log-diff, z-score, MA ratio) is scale-invariant.
    """
    start_ms = int(pd.Timestamp(start_iso, tz="UTC").timestamp() * 1000)
    end_ms   = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    cursor, rows = start_ms, []
    while cursor < end_ms:
        batch = _binance_klines(cursor, symbol=symbol)
        if not batch or not isinstance(batch[-1], (list, tuple)):
            break
        rows.extend(batch)
        cursor = int(batch[-1][0]) + 3600_000
        time.sleep(0.05)
    if not rows:
        raise RuntimeError(f"Binance hourly klines unavailable for {symbol} (all hosts failed)")
    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "qv", "n", "tb", "tq", "ig"]
    h = pd.DataFrame(rows, columns=cols)
    h["ts"] = pd.to_datetime(h["open_time"], unit="ms", utc=True).dt.tz_convert(None)
    for c in ["open", "high", "low", "close", "volume"]:
        h[c] = h[c].astype(float)
    h = h.set_index("ts")[["open", "high", "low", "close", "volume"]]
    h = h[~h.index.duplicated(keep="last")].sort_index()
    # Rebucket into 24h bars starting at ANCHOR_HOUR_UTC (identical to the app).
    h["bucket"] = (h.index - pd.Timedelta(hours=ANCHOR_HOUR_UTC)).normalize()
    g = h.groupby("bucket").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"), n_hours=("close", "size"))
    g = g[g["n_hours"] == 24].drop(columns="n_hours")
    g.index.name = "Date"
    return g


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    df.index.name = "Date"          # so mstr_daily.csv etc. keep a 'Date' header
    return df


_YH_HOSTS = ("https://query1.finance.yahoo.com", "https://query2.finance.yahoo.com")
_YH_UA = {"User-Agent": "Mozilla/5.0"}


def _yf_requests(ticker: str, start: str) -> pd.DataFrame:
    """Daily OHLCV from Yahoo's chart API via plain ``requests``, auto-adjusted
    for splits/dividends (matching ``yfinance`` ``auto_adjust=True``).  The
    yfinance client's own HTTP stack fails behind the agent proxy, whereas this
    is the same request path the live app already uses successfully."""
    p1 = int(pd.Timestamp(start).timestamp())
    p2 = int(pd.Timestamp.now(tz="UTC").timestamp())
    params = {"interval": "1d", "period1": p1, "period2": p2, "events": "div,splits"}
    for host in _YH_HOSTS:
        try:
            r = requests.get(f"{host}/v8/finance/chart/{ticker}",
                             params=params, headers=_YH_UA, timeout=30)
            if r.status_code != 200:
                continue
            res = r.json()["chart"]["result"][0]
            ts = res.get("timestamp")
            if not ts:
                continue
            q = res["indicators"]["quote"][0]
            close = np.array(q.get("close"), dtype=float)
            df = pd.DataFrame({
                "Open":  np.array(q.get("open"),   dtype=float),
                "High":  np.array(q.get("high"),   dtype=float),
                "Low":   np.array(q.get("low"),    dtype=float),
                "Close": close,
                "Volume": np.array(q.get("volume"), dtype=float),
            }, index=pd.to_datetime(ts, unit="s"))
            # auto-adjust OHLC by the adjusted-close ratio (splits + dividends),
            # so e.g. MSTR's 2024 split matches the yfinance-built history.
            adj = (res.get("indicators", {}).get("adjclose", [{}]) or [{}])[0].get("adjclose")
            if adj is not None:
                adjc = np.array(adj, dtype=float)
                ratio = np.where(close > 0, adjc / close, 1.0)
                for c in ("Open", "High", "Low", "Close"):
                    df[c] = df[c].values * ratio
            df = _norm(df)
            df = df[~df.index.duplicated(keep="last")].sort_index().dropna(subset=["Close"])
            if not df.empty:
                return df
        except Exception as exc:
            print(f"  [yf-requests {ticker}] {exc}")
    return pd.DataFrame()


def _yf(ticker: str, start: str, retries: int = 3) -> pd.DataFrame:
    # Prefer the proxy-friendly requests path; fall back to the yfinance client.
    df = _yf_requests(ticker, start)
    if not df.empty:
        return df
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
            df = _norm(df)
            if not df.empty:
                return df
        except Exception as exc:
            print(f"  [yf attempt {attempt}] {ticker}: {exc}")
            if attempt < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to download {ticker} after {retries} attempts")


def _checksum(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.to_csv().encode()).hexdigest()[:16]


# ─── vintage freeze ──────────────────────────────────────────────────────────
# Historical bars are PINNED: once a row has been committed, later pulls keep
# its values verbatim and may only (a) revise the newest FREEZE_TAIL rows (the
# legitimate correction window for late prints / ffill catch-ups) and
# (b) append genuinely new rows.  Without this, restate-prone sources rewrote
# the entire history on every pull — blockchain.info's `sampled=true` grid is
# anchored to *now*, so oc_mempool_size / oc_market_cap changed on ALL ~1,005
# rows daily (observed 2026-08-03: median 15.5% / 0.7% restatements), flipping
# historical CT gate decisions and swinging the Overall headline backtest by
# hundreds of points between vintages (the "+950% vs +1205%" incident).
#
# NOT applied to the yfinance-adjusted equity CSVs (MSTR / MSTU actual): a
# split legitimately rescales their whole history, which a freeze would
# corrupt.  Those are guarded by a scale-invariant returns-agreement check in
# scripts/validate_refreshed_data.py instead.
#
# Set PULL_UNFROZEN=1 for a deliberate full re-baseline (documented in
# DATA_CONSISTENCY.md) — the validator will reject the restatement unless it
# is bypassed too, so a re-baseline is always an explicit two-step decision.
FREEZE_TAIL = 5


def _freeze_enabled() -> bool:
    return os.environ.get("PULL_UNFROZEN", "").strip().lower() not in (
        "1", "true", "yes", "on")


def _freeze_history(name: str, new_df: pd.DataFrame, csv_path: Path,
                    tail: int = FREEZE_TAIL) -> pd.DataFrame:
    """Merge a fresh pull onto the pinned on-disk vintage: committed rows older
    than the last ``tail`` keep their pinned values; the tail and new rows take
    the fresh values; rows the fresh pull lost entirely are restored."""
    if not _freeze_enabled():
        print(f"  [freeze] {name}: DISABLED via PULL_UNFROZEN — full restatement")
        return new_df
    try:
        prev = pd.read_csv(csv_path, index_col=0, parse_dates=True,
                           float_precision="round_trip")
    except Exception:
        return new_df                              # first pull — nothing pinned
    prev = prev[~prev.index.duplicated(keep="last")].sort_index()
    if prev.empty:
        return new_df
    out = new_df.copy()
    frozen_idx = (prev.index[:-tail] if tail else prev.index).intersection(out.index)
    cols = [c for c in prev.columns if c in out.columns]
    if len(frozen_idx) and cols:
        out.loc[frozen_idx, cols] = prev.loc[frozen_idx, cols].to_numpy()
    lost = prev.index.difference(out.index)
    if len(lost):
        out = pd.concat([out, prev.loc[lost, cols]]).sort_index()
    n_new = len(out.index.difference(prev.index))
    print(f"  [freeze] {name}: {len(frozen_idx)} historical rows pinned, "
          f"{min(tail, len(prev))} tail rows refreshable, {n_new} appended"
          + (f", {len(lost)} lost rows restored" if len(lost) else ""))
    return out


# ─── quality checks ─────────────────────────────────────────────────────────

def validate(name: str, df: pd.DataFrame, close_col: str = "close",
             min_rows: int = 50) -> bool:
    issues = []
    if len(df) < min_rows:
        issues.append(f"only {len(df)} rows (expected ≥ {min_rows})")
    if close_col not in df.columns:
        issues.append(f"missing column '{close_col}'")
        print(f"  [QC FAIL] {name}: {'; '.join(issues)}")
        return False
    prices = df[close_col].dropna()
    if len(prices) < min_rows:
        issues.append(f"only {len(prices)} non-NaN prices")
    if (prices <= 0).any():
        issues.append(f"{(prices<=0).sum()} non-positive price(s)")
    trading = prices.loc[prices.index.dayofweek < 5]
    flat = (trading.diff().abs() < 1e-8).rolling(7).sum()
    if (flat >= 7).any():
        issues.append("suspicious flat run >7 trading days")
    log_r = np.log(prices / prices.shift(1)).dropna()
    if (log_r.abs() > np.log(1.8)).any():
        issues.append(f"{(log_r.abs()>np.log(1.8)).sum()} move(s) >80% in one day")
    today = pd.Timestamp.today().normalize()
    if (today - df.index[-1]).days > 5:
        issues.append(f"stale — last row {df.index[-1].date()} ({(today-df.index[-1]).days}d ago)")
    if issues:
        print(f"  [QC WARN] {name}: {'; '.join(issues)}")
        return False
    print(f"  [QC OK ] {name}: {len(df)} rows  {df.index[0].date()} → {df.index[-1].date()}")
    return True


def validate_raw(name: str, df: pd.DataFrame, min_rows: int = 100) -> bool:
    """Validate a raw features dataframe (multi-column, partial NaN OK)."""
    if len(df) < min_rows:
        print(f"  [QC WARN] {name}: only {len(df)} rows")
        return False
    non_null_pct = (1 - df.isnull().mean()).mean() * 100
    today = pd.Timestamp.today().normalize()
    stale = (today - df.index[-1]).days > 5
    if stale:
        print(f"  [QC WARN] {name}: stale — last row {df.index[-1].date()}")
        return False
    print(f"  [QC OK ] {name}: {len(df)} rows  {df.index[0].date()} → {df.index[-1].date()}  "
          f"avg non-null {non_null_pct:.0f}%  ({len(df.columns)} cols)")
    return True


# ─── MSTU OLS synthesis ─────────────────────────────────────────────────────

def synthesise_mstu(mstr_df: pd.DataFrame, mstu_actual: pd.DataFrame) -> pd.Series:
    common = mstr_df.index.intersection(mstu_actual.index)
    common = common[common >= INCEPTION]
    beta, alpha = 2.0, -0.0002
    if len(common) >= 10:
        ml = np.log(mstr_df.loc[common, "close"] /
                    mstr_df.loc[common, "close"].shift(1)).dropna()
        sl = np.log(mstu_actual.loc[common, "close"] /
                    mstu_actual.loc[common, "close"].shift(1)).dropna()
        al = ml.index.intersection(sl.index)
        x, y = ml.loc[al].values, sl.loc[al].values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() >= 10:
            c = np.polyfit(x[mask], y[mask], 1)
            beta, alpha = float(c[0]), float(c[1])
            print(f"  OLS  β={beta:.4f}  α={alpha:.6f}  (n={mask.sum()} bars)")
    pre_idx = mstr_df.index[mstr_df.index < INCEPTION]
    mstr_pre = mstr_df.loc[pre_idx, "close"]
    lr_pre = np.log(mstr_pre / mstr_pre.shift(1)).fillna(0.0).values
    syn_lr = beta * lr_pre + alpha
    anchor = float(mstu_actual.loc[mstu_actual.index >= INCEPTION, "close"].iloc[0])
    cum = np.cumsum(syn_lr)
    syn_px = anchor * np.exp(cum - cum[-1])
    combined = pd.concat([
        pd.Series(syn_px, index=pre_idx, name="close"),
        mstu_actual["close"].copy().rename("close"),
    ]).sort_index()
    return combined[~combined.index.duplicated(keep="last")]


# ─── main ───────────────────────────────────────────────────────────────────

def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pull_ts   = datetime.now(timezone.utc)
    pull_date = pull_ts.strftime("%Y-%m-%d")
    print(f"=== pull_backtest_data.py  ({pull_date} UTC) ===\n")

    fetch_end = (pd.Timestamp.today() + pd.Timedelta(days=3)).strftime("%Y-%m-%d")

    # ── 1. BTC-USD OHLCV (12:00-UTC bars from Binance hourly) ─────────────────
    print("Downloading BTC-USD (Binance hourly → 12:00-UTC daily bars) …")
    btc = fetch_12utc(FETCH_FROM)[["open","high","low","close","volume"]].ffill()

    # ── 2. Macro series ───────────────────────────────────────────────────────
    print("Downloading macro series …")
    macro_frames: dict = {}
    for nm, sym in _MACRO_SYMS.items():
        try:
            d = _yf(sym, FETCH_FROM)
            macro_frames[f"{nm}_close"] = d["Close"]
            print(f"  {sym:12s}: {len(d)} rows")
        except Exception as exc:
            print(f"  [WARN] {sym}: {exc}")

    # ── 3. On-chain (blockchain.info) ─────────────────────────────────────────
    print("Downloading on-chain metrics …")
    onchain_frames: dict = {}
    ok_count = 0
    for m in _ONCHAIN:
        col = f"oc_{m.replace('-','_')}"
        try:
            r = requests.get(
                f"https://api.blockchain.info/charts/{m}",
                params={"timespan": "3years", "format": "json", "sampled": "true"},
                timeout=25,
            )
            vals = r.json().get("values", [])
            s = pd.Series(
                {pd.Timestamp(v["x"], unit="s").normalize(): v["y"] for v in vals},
                name=col, dtype=float,
            )
            s = s[~s.index.duplicated(keep="last")].sort_index()
            s.index = pd.DatetimeIndex(s.index).tz_localize(None)
            onchain_frames[col] = s
            ok_count += 1
        except Exception as exc:
            print(f"  [WARN] {m}: {exc}")
    print(f"  On-chain: {ok_count}/{len(_ONCHAIN)} OK")

    # ── 4. Coinbase premium ───────────────────────────────────────────────────
    print("Downloading Coinbase BTC premium …")
    cb_rows: list = []
    cur = pd.Timestamp(FETCH_FROM)
    end = pd.Timestamp(fetch_end)
    while cur <= end:
        chunk = min(cur + pd.Timedelta(days=299), end)
        try:
            r2 = requests.get(_CB_URL, params={
                "granularity": 86400,
                "start": cur.strftime("%Y-%m-%dT00:00:00Z"),
                "end":   (chunk + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z"),
            }, timeout=30)
            if r2.status_code == 200:
                cb_rows.extend(r2.json())
        except Exception as exc:
            print(f"  [WARN] Coinbase chunk {cur.date()}: {exc}")
        cur = chunk + pd.Timedelta(days=1)
        time.sleep(0.15)
    print(f"  Coinbase candles: {len(cb_rows)} rows received")

    # ── 5. Build merged raw_df ────────────────────────────────────────────────
    print("\nBuilding merged raw_features dataframe …")
    df = btc[["close","high","low","volume"]].copy()
    df.columns = ["btc_close","btc_high","btc_low","btc_volume"]

    for col, s in macro_frames.items():
        df[col] = s.reindex(df.index).ffill(limit=7)

    for col, s in onchain_frames.items():
        df[col] = s.reindex(df.index).ffill(limit=7)

    if cb_rows:
        cb_df = pd.DataFrame(cb_rows, columns=["ts","low","high","open","close","volume"])
        cb_df["date"] = pd.to_datetime(cb_df["ts"], unit="s").dt.normalize()
        cb_df = cb_df.drop_duplicates("date").set_index("date").sort_index()
        cb_df.index = pd.DatetimeIndex(cb_df.index).tz_localize(None)
        cb_close = cb_df["close"].reindex(df.index).astype(float)
        prem = (cb_close - df["btc_close"]) / df["btc_close"] * 100
        df["cb_premium"]     = prem
        df["cb_premium_ma3"] = prem.rolling(3).mean()
        df["cb_premium_z7"]  = (prem - prem.rolling(7).mean()) / prem.rolling(7).std()
    else:
        df["cb_premium"] = df["cb_premium_ma3"] = df["cb_premium_z7"] = 0.0

    # ── 6. MSTR, MSTU, synthetic ──────────────────────────────────────────────
    print("Downloading MSTR …")
    mstr_raw = _yf("MSTR", FETCH_FROM)
    mstr = mstr_raw[["Open","High","Low","Close","Volume"]].rename(columns=str.lower).ffill()

    print("Downloading MSTU …")
    mstu_raw = _yf("MSTU", str(INCEPTION.date()))
    mstu = mstu_raw[["Open","High","Low","Close","Volume"]].rename(columns=str.lower).ffill()

    print("Synthesising MSTU pre-inception (OLS) …")
    mstu_syn = synthesise_mstu(mstr, mstu).to_frame("close")

    # ETH spot on the SAME 12:00-UTC bar boundary as BTC, so the ETH sleeve's
    # fill lands exactly when the BTC signal bar closes.  ETH is the traded
    # *signal* asset; live execution goes through the ETHA ETF (see
    # scripts/ibkr_symbols.py), exactly as the BTC sleeve executes via IBIT.
    print("Downloading ETH-USD (Binance hourly → 12:00-UTC daily bars) …")
    eth = fetch_12utc(FETCH_FROM, symbol="ETHUSDT")[["open","high","low","close","volume"]].ffill()

    # ── 6b. Vintage freeze — pinned history, append-only + tail corrections ──
    # (see _freeze_history; MSTR/MSTU actual stay unfrozen for split safety)
    print("\nApplying vintage freeze:")
    df       = _freeze_history("raw_features",   df,       DATA_DIR / "raw_features_daily.csv")
    btc      = _freeze_history("btc_usd_daily",  btc,      DATA_DIR / "btc_usd_daily.csv")
    eth      = _freeze_history("eth_usd_daily",  eth,      DATA_DIR / "eth_usd_daily.csv")
    # the OLS synthetic is pre-inception history — a re-fit rewrites the past
    # by definition, so it is fully pinned (tail=0) once committed
    mstu_syn = _freeze_history("mstu_synthetic", mstu_syn, DATA_DIR / "mstu_synthetic_daily.csv",
                               tail=0)

    # ── 7. Quality checks ─────────────────────────────────────────────────────
    print("\nQuality checks:")
    rf_ok   = validate_raw("raw_features", df)
    btc_ok  = validate("BTC-USD",           btc,      min_rows=300)
    mstr_ok = validate("MSTR",              mstr,     min_rows=200)
    mstu_ok = validate("MSTU (actual)",     mstu,     min_rows=50)
    msyn_ok = validate("MSTU (synthetic)",  mstu_syn, min_rows=100)
    eth_ok  = validate("ETH-USD",           eth,      min_rows=300)

    all_ok = all([rf_ok, btc_ok, mstr_ok, mstu_ok, msyn_ok, eth_ok])
    if not all_ok:
        print("\n⚠  One or more QC checks failed — data saved with warnings.")
    else:
        print("\n✅  All QC checks passed.")

    # ── 8. Save CSVs ─────────────────────────────────────────────────────────
    print(f"\nSaving CSVs to {DATA_DIR}/")
    df.to_csv(       DATA_DIR / "raw_features_daily.csv")
    btc.to_csv(      DATA_DIR / "btc_usd_daily.csv")
    mstr.to_csv(     DATA_DIR / "mstr_daily.csv")
    mstu.to_csv(     DATA_DIR / "mstu_daily.csv")
    mstu_syn.to_csv( DATA_DIR / "mstu_synthetic_daily.csv")
    eth.to_csv(      DATA_DIR / "eth_usd_daily.csv")

    # ── 9. Manifest ───────────────────────────────────────────────────────────
    manifest = {
        "version":       "v3",
        "pull_date":     pull_date,
        "pull_ts_utc":   pull_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_by":  "scripts/pull_backtest_data.py",
        "qc_all_passed": all_ok,
        "btc_bar_anchor": ("12:00 UTC (7am CT) — BTC OHLCV rebucketed from Binance "
                           "hourly to match the daily model and live view. Macro/on-chain "
                           "remain calendar-date joined. Equities (MSTR/MSTU) remain "
                           "exchange-session daily."),
        "datasets": {
            "raw_features_daily": {
                "file":            "raw_features_daily.csv",
                "description":     "Full merged raw data: BTC OHLCV + 7 macro + 11 on-chain + Coinbase premium",
                "columns":         list(df.columns),
                "rows":            len(df),
                "date_from":       str(df.index[0].date()),
                "date_to":         str(df.index[-1].date()),
                "checksum_sha256": _checksum(df),
                "qc_passed":       rf_ok,
            },
            "btc_usd_daily": {
                "file":            "btc_usd_daily.csv",
                "ticker":          "BTC-USD (Binance BTCUSDT, 12:00-UTC bars)",
                "note":            "volume is in BTC (base asset); all volume features are scale-invariant",
                "rows":            len(btc),
                "date_from":       str(btc.index[0].date()),
                "date_to":         str(btc.index[-1].date()),
                "checksum_sha256": _checksum(btc),
                "qc_passed":       btc_ok,
            },
            "mstr_daily": {
                "file":            "mstr_daily.csv",
                "ticker":          "MSTR",
                "rows":            len(mstr),
                "date_from":       str(mstr.index[0].date()),
                "date_to":         str(mstr.index[-1].date()),
                "checksum_sha256": _checksum(mstr),
                "qc_passed":       mstr_ok,
            },
            "mstu_daily": {
                "file":            "mstu_daily.csv",
                "ticker":          "MSTU",
                "rows":            len(mstu),
                "date_from":       str(mstu.index[0].date()),
                "date_to":         str(mstu.index[-1].date()),
                "checksum_sha256": _checksum(mstu),
                "qc_passed":       mstu_ok,
            },
            "mstu_synthetic_daily": {
                "file":            "mstu_synthetic_daily.csv",
                "ticker":          "MSTU (OLS-synthetic)",
                "rows":            len(mstu_syn),
                "date_from":       str(mstu_syn.index[0].date()),
                "date_to":         str(mstu_syn.index[-1].date()),
                "checksum_sha256": _checksum(mstu_syn),
                "qc_passed":       msyn_ok,
            },
            "eth_usd_daily": {
                "file":            "eth_usd_daily.csv",
                "ticker":          "ETH-USD (Binance ETHUSDT, 12:00-UTC bars)",
                "note":            ("spot-ETH sleeve traded off the BTC CT signal; same bar "
                                    "anchor as BTC so the fill lands on the signal bar's close. "
                                    "Live execution is via the ETHA ETF (cf. BTC→IBIT). "
                                    "Volume is in ETH (base asset); volume features are scale-invariant."),
                "rows":            len(eth),
                "date_from":       str(eth.index[0].date()),
                "date_to":         str(eth.index[-1].date()),
                "checksum_sha256": _checksum(eth),
                "qc_passed":       eth_ok,
            },
        },
    }
    with open(DATA_DIR / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nmanifest.json  version={manifest['version']}  pull_date={manifest['pull_date']}")
    print(f"raw_features_daily.csv: {len(df)} rows × {len(df.columns)} cols")
    print("\nDone.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
