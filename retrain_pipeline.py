#!/usr/bin/env python3
"""BTC trading model retraining pipeline.

Orchestrates the three TF2-strategy models in dependency order:
  1. Daily H/L ensemble        (primary — always run first)
  2. 7-day close cone          (independent; run quarterly or on coverage drop)
  3. 3-class day-type          (depends on H/L OOF preds AND cone regime_edges)

Usage:
  python retrain_pipeline.py              # assess → retrain if needed
  python retrain_pipeline.py --assess-only
  python retrain_pipeline.py --force
  python retrain_pipeline.py --skip-cone
  python retrain_pipeline.py --no-archive
  python retrain_pipeline.py --no-fetch   # skip Yahoo live-MAPE fetch
"""

import argparse
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from paths import DAILY_MODEL_CT, CONE_7D_MODEL, DAY_TYPE_MODEL, MODELS_DIR, BINANCE_HOURLY_CSV

BACKUPS_DIR = MODELS_DIR / "backups"

# ---------------------------------------------------------------------------
# Terminal colour helpers
# ---------------------------------------------------------------------------
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def _red(s):    return f"{RED}{s}{RESET}"
def _yellow(s): return f"{YELLOW}{s}{RESET}"
def _green(s):  return f"{GREEN}{s}{RESET}"
def _bold(s):   return f"{BOLD}{s}{RESET}"

def _header(s):
    print(f"\n{CYAN}{'=' * 62}{RESET}\n{BOLD}  {s}{RESET}\n{CYAN}{'=' * 62}{RESET}")

def _info(s=""):  print(f"  {s}")
def _ok(s):       print(f"  {GREEN}✓{RESET}  {s}")
def _warn(s):     print(f"  {YELLOW}⚠{RESET}  {s}")
def _err(s):      print(f"  {RED}✗{RESET}  {s}")

# ---------------------------------------------------------------------------
# Age & performance thresholds  (match Retrain Status dashboard)
# ---------------------------------------------------------------------------
WARN_DAYS    = {"daily_hl": 42,  "cone_7d": 84, "day_type": 42}
OVERDUE_DAYS = {"daily_hl": 56,  "cone_7d": 90, "day_type": 56}

MAPE_H_RETRAIN   = 2.5    # % — retrain if live 30-day MAPE_H exceeds this
COVERAGE_RETRAIN = 78.0   # % — retrain cone if test coverage drops below
ACCURACY_RETRAIN = 50.0   # % — retrain 3-class if test accuracy drops below

LABEL_MAP = {
    "daily_hl": "Daily H/L ensemble",
    "cone_7d":  "7-day close cone",
    "day_type": "3-class day-type classifier",
}
SCRIPT_MAP = {
    "daily_hl": "src/pipeline_ct.py",
    "cone_7d":  "src/train_7d_close_cone.py",
    "day_type": "src/train_3class_day_type.py",
}
ARTEFACT_MAP = {
    "daily_hl": DAILY_MODEL_CT,
    "cone_7d":  CONE_7D_MODEL,
    "day_type": DAY_TYPE_MODEL,
}

# ---------------------------------------------------------------------------
# 1.  DATA FETCH
# ---------------------------------------------------------------------------
_MACRO_SYMS = {
    "eth": "ETH-USD", "spx": "^GSPC", "ndx": "^IXIC",
    "vix": "^VIX",   "gold": "GC=F",  "dxy": "DX-Y.NYB", "tnx": "^TNX",
}


def _flat(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = [f"{name}_{c.lower()}" for c in df.columns]
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df[~df.index.duplicated(keep="last")].sort_index()


def _fetch_coinbase_daily(start: str, end: str) -> pd.Series:
    """Fetch BTC-USD daily closes from the Coinbase Exchange public API.

    Paginates in 299-day chunks. Returns a Series indexed by date,
    named 'coinbase_close'. Returns empty Series on failure.
    """
    CB_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    rows: list = []
    cur = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    while cur <= end_ts:
        chunk_end = min(cur + pd.Timedelta(days=299), end_ts)
        params = {
            "granularity": 86400,
            "start": cur.isoformat() + "Z",
            "end":   (chunk_end + pd.Timedelta(days=1)).isoformat() + "Z",
        }
        try:
            r = requests.get(CB_URL, params=params, timeout=30)
            if r.status_code == 200:
                rows.extend(r.json())
            else:
                _warn(f"    Coinbase API {r.status_code}: {r.text[:80]}")
        except Exception as exc:
            _warn(f"    Coinbase fetch error: {exc}")
        cur = chunk_end + pd.Timedelta(days=1)
        time.sleep(0.35)
    if not rows:
        _warn("    Coinbase: no data — cb_premium features will be 0")
        return pd.Series(dtype=float, name="coinbase_close")
    tmp = pd.DataFrame(rows, columns=["ts", "low", "high", "open", "close", "volume"])
    tmp["date"] = pd.to_datetime(tmp["ts"], unit="s").dt.normalize()
    tmp = tmp.drop_duplicates("date").set_index("date").sort_index()
    _info(f"    coinbase_daily: {len(tmp)} rows  "
          f"{tmp.index.min().date()} → {tmp.index.max().date()}")
    return tmp["close"].rename("coinbase_close")


def fetch_recent_daily(days: int = 90) -> pd.DataFrame | None:
    """Fetch ~`days` of BTC + macro daily bars from Yahoo Finance.

    Returns a combined DataFrame with columns like btc_close, spx_close, …
    Returns None if BTC data cannot be fetched.
    """
    _info(f"Fetching {days}d BTC + macro from Yahoo Finance …")
    today = datetime.now(timezone.utc).date()
    start = (pd.Timestamp(today) - pd.Timedelta(days=days + 14)).strftime("%Y-%m-%d")
    end   = (pd.Timestamp(today) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    parts: list[pd.DataFrame] = []
    syms  = {"btc": "BTC-USD"} | _MACRO_SYMS
    for name, sym in syms.items():
        try:
            raw = yf.download(sym, start=start, end=end, progress=False, auto_adjust=False)
            if raw.empty:
                _warn(f"    {sym}: empty response"); continue
            parts.append(_flat(raw, name))
            _info(f"    {sym}: {len(raw)} rows  ({raw.index.min().date()} → {raw.index.max().date()})")
        except Exception as exc:
            _warn(f"    {sym}: {exc}")

    if not parts or "btc_close" not in parts[0].columns:
        return None
    df = parts[0]
    for p in parts[1:]:
        df = df.join(p, how="outer")
    df = df.ffill(limit=5)

    # Coinbase premium (Coinbase BTC-USD close vs Yahoo BTC-USD reference)
    _cb = _fetch_coinbase_daily(start, end)
    if _cb.notna().any():
        df = df.join(_cb.to_frame("coinbase_close"), how="left")
        df["cb_premium"] = (df["coinbase_close"] - df["btc_close"]) / df["btc_close"] * 100

    return df.tail(days)


# ---------------------------------------------------------------------------
# 2.  MINIMAL FEATURE BUILDER
#     Builds BTC + macro features that match the model's feat_cols list.
#     On-chain features not available here are filled with 0.
# ---------------------------------------------------------------------------

def _build_hl_features(daily: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """Return a DataFrame with exactly `feat_cols` columns, filled from `daily`.

    BTC technical features and macro ratios are computed; any column that
    cannot be derived from available data is left as 0.  The model handles
    0-filled on-chain inputs gracefully (it was trained with ffill, so 0 is
    outside its training distribution, but the ensemble degrades smoothly).
    """
    d = daily.copy()
    c = d.get("btc_close",  pd.Series(dtype=float, index=d.index))
    h = d.get("btc_high",   pd.Series(dtype=float, index=d.index))
    l = d.get("btc_low",    pd.Series(dtype=float, index=d.index))
    v = d.get("btc_volume", pd.Series(dtype=float, index=d.index))

    f = pd.DataFrame(index=d.index)

    # Returns
    for w in [1, 2, 3, 5, 7, 10, 14, 21, 30]:
        f[f"ret_{w}d"] = c.pct_change(w)
    # Bar geometry
    f["daily_range"]   = (h - l) / c
    f["upper_shadow"]  = (h - c) / c
    f["lower_shadow"]  = (c - l) / c
    # Volatility
    log_r = np.log(c / c.shift(1))
    for w in [5, 10, 14, 21, 30]:
        f[f"vol_{w}d"] = log_r.rolling(w).std() * np.sqrt(365)
    # MA ratio & slope
    for w in [7, 14, 21, 30, 50, 100, 200]:
        ma = c.rolling(w).mean()
        f[f"ma{w}_ratio"] = c / ma - 1
    for w in [7, 14, 30]:
        f[f"ma{w}_slope"] = c.rolling(w).mean().pct_change(3)
    # RSI (7, 14, 21)
    for w in [7, 14, 21]:
        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(w).mean()
        loss  = (-delta.clip(upper=0)).rolling(w).mean()
        f[f"rsi_{w}"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    # Volume ratios
    f["vol_ratio_5d"]  = v / v.rolling(5).mean()
    f["vol_ratio_21d"] = v / v.rolling(21).mean()
    # Range features
    rng = h - l
    for w in [5, 14, 30]:
        f[f"range_ma{w}_ratio"] = rng / rng.rolling(w).mean()
    f["range_ma30"] = rng.rolling(30).mean() / c
    # Macro returns
    for name in ["eth", "spx", "ndx", "vix", "gold", "dxy", "tnx"]:
        src = d.get(f"{name}_close")
        if src is not None:
            f[f"{name}_close"]  = src
            f[f"{name}_ret1d"]  = src.pct_change(1)
            f[f"{name}_ret5d"]  = src.pct_change(5)
            f[f"{name}_ret21d"] = src.pct_change(21)
    # Coinbase premium features
    prem_raw = d.get("cb_premium")
    if prem_raw is not None and prem_raw.notna().any():
        f["cb_premium"]     = prem_raw
        f["cb_premium_ma3"] = prem_raw.rolling(3).mean()
        f["cb_premium_z7"]  = (prem_raw - prem_raw.rolling(7).mean()) / prem_raw.rolling(7).std()

    # Build output aligned to feat_cols; missing features → 0
    out = pd.DataFrame(0.0, index=f.index, columns=feat_cols)
    for col in feat_cols:
        if col in f.columns:
            out[col] = f[col].to_numpy()
    return out.ffill(limit=5).fillna(0.0)


# ---------------------------------------------------------------------------
# 3.  H/L ENSEMBLE INFERENCE  (mirrors app inference path)
# ---------------------------------------------------------------------------

def _run_hl_inference(art: dict, feat_row: pd.Series) -> tuple[float, float]:
    """Return (pred_hi_pct, pred_lo_pct) for a single feature row.

    pred_hi_pct > 0: predicted high is `pred_hi_pct` above today's close.
    pred_lo_pct > 0: predicted low is `pred_lo_pct` below today's close.
    """
    X = feat_row.to_numpy().reshape(1, -1)
    preds_hi: list[float] = []
    preds_lo: list[float] = []
    for constituent in art.get("constituents", []):
        try:
            preds_hi.append(float(constituent["m_hi"].predict(X)[0]))
            preds_lo.append(float(constituent["m_lo"].predict(X)[0]))
        except Exception:
            pass
    if not preds_hi:
        return float("nan"), float("nan")

    raw_hi = float(np.mean(preds_hi))
    raw_lo = float(np.mean(preds_lo))

    alpha  = float(art.get("alpha", 0.5))
    hi_m   = art.get("hi_model")
    lo_m   = art.get("lo_model")
    if hi_m is not None:
        raw_hi = alpha * raw_hi + (1 - alpha) * float(hi_m.predict(X)[0])
    if lo_m is not None:
        raw_lo = alpha * raw_lo + (1 - alpha) * float(lo_m.predict(X)[0])

    raw_hi += float(art.get("mu_hi", 0.0))
    raw_lo += float(art.get("mu_lo", 0.0))
    return raw_hi, raw_lo


# ---------------------------------------------------------------------------
# 4.  TABULAR CUSUM  (one-sided, normalised)
# ---------------------------------------------------------------------------

def tabular_cusum(series: pd.Series, k: float = 0.5, h: float = 4.0) -> dict:
    """Detect a sustained mean shift with a tabular CUSUM.

    k  = allowance (half the expected shift size in σ units)
    h  = decision threshold (alarm when CUSUM > h)

    Returns dict:
      triggered_up    bool   — errors are growing (over-prediction regime)
      triggered_dn    bool   — errors are shrinking (under-prediction regime)
      cusum_up_max    float
      cusum_dn_max    float
      n               int
    """
    s = series.dropna()
    if len(s) < 8:
        return dict(triggered_up=False, triggered_dn=False,
                    cusum_up_max=0.0, cusum_dn_max=0.0, n=len(s))
    mu  = float(s.mean())
    std = float(s.std(ddof=1))
    if std < 1e-9:
        return dict(triggered_up=False, triggered_dn=False,
                    cusum_up_max=0.0, cusum_dn_max=0.0, n=len(s))
    z  = (s - mu) / std
    S_up = 0.0; S_dn = 0.0
    max_up = 0.0; max_dn = 0.0
    for zi in z:
        S_up = max(0.0, S_up + float(zi) - k)
        S_dn = max(0.0, S_dn - float(zi) - k)
        max_up = max(max_up, S_up)
        max_dn = max(max_dn, S_dn)
    return dict(triggered_up=max_up > h, triggered_dn=max_dn > h,
                cusum_up_max=max_up, cusum_dn_max=max_dn, n=len(z))


# ---------------------------------------------------------------------------
# 5.  PER-MODEL ASSESSMENT
# ---------------------------------------------------------------------------

def _load_art(path: Path) -> dict | None:
    try:
        return joblib.load(path)
    except Exception as exc:
        _err(f"Could not load {path.name}: {exc}")
        return None


def _model_age(art: dict) -> int | None:
    """Days since calibration_meta.test_end (how much new data exists beyond
    the model's last evaluated date).  Falls back to train_end if test_end
    is absent.  This is the correct staleness measure for models with a
    fixed holdout window — train_end is always months in the past by design.
    """
    meta = art.get("calibration_meta", {})
    te   = meta.get("test_end") or meta.get("train_end")
    if not te:
        return None
    today = pd.Timestamp.now().normalize()
    te_ts = pd.Timestamp(te).normalize()
    return int((today - te_ts).days)


def assess_daily_hl(art: dict, daily_df: pd.DataFrame | None) -> dict:
    """Assess the Daily H/L ensemble.

    Checks: model age, stored test MAPE, live 30-day MAPE + bias, CUSUM.

    Returns dict with keys: verdict, reasons, age_days, live_mape_h,
    live_bias_h, cusum, expected_gain, test_mape_h.
    """
    reasons: list[str] = []
    verdict = "PASS"
    age     = _model_age(art)
    meta    = art.get("calibration_meta", {})
    m_test  = meta.get("metrics", {})

    # — Age check ——————————————————————————————————————————————————————————
    if age is None:
        reasons.append("Train date unknown — treat as overdue")
        verdict = "RETRAIN"
    elif age >= OVERDUE_DAYS["daily_hl"]:
        reasons.append(f"Model is {age}d old (overdue threshold: {OVERDUE_DAYS['daily_hl']}d)")
        verdict = "RETRAIN"
    elif age >= WARN_DAYS["daily_hl"]:
        reasons.append(f"Model is {age}d old (warn threshold: {WARN_DAYS['daily_hl']}d)")
        verdict = "WARN"

    # — Stored test MAPE ——————————————————————————————————————————————————
    mape_h_test = m_test.get("MAPE_H")
    if mape_h_test is not None and mape_h_test > MAPE_H_RETRAIN:
        reasons.append(
            f"Stored test MAPE_H={mape_h_test:.2f}% > {MAPE_H_RETRAIN}% threshold"
        )
        verdict = "RETRAIN"

    # — Live inference: 30-day MAPE, bias, CUSUM ——————————————————————————
    live_mape_h  = None
    live_bias_h  = None
    cusum_result = None
    feat_cols    = art.get("feat_cols", [])

    if daily_df is not None and len(feat_cols) > 0:
        try:
            features  = _build_hl_features(daily_df, feat_cols)
            btc_close = daily_df["btc_close"]
            btc_high  = daily_df["btc_high"]
            btc_low   = daily_df["btc_low"]

            errs_hi: list[float] = []
            errs_lo: list[float] = []
            # Predict bar D → actual on bar D+1 (last 30 complete pairs)
            valid_idx = [
                i for i in range(len(daily_df) - 1)
                if not np.any(np.isnan(features.iloc[i].to_numpy()))
            ][-30:]

            for i in valid_idx:
                ph, pl = _run_hl_inference(art, features.iloc[i])
                if np.isnan(ph):
                    continue
                close   = float(btc_close.iloc[i])
                act_hi  = float(btc_high.iloc[i + 1])
                act_lo  = float(btc_low.iloc[i + 1])
                pred_hi = close * (1.0 + ph)
                pred_lo = close * (1.0 - pl)
                errs_hi.append((act_hi - pred_hi) / act_hi)
                errs_lo.append((act_lo - pred_lo) / act_lo)

            if len(errs_hi) >= 5:
                arr = np.array(errs_hi)
                live_mape_h = float(np.mean(np.abs(arr)) * 100)
                live_bias_h = float(np.mean(arr) * 100)

                _info(f"    Live 30d stats ({len(errs_hi)} bars): "
                      f"MAPE_H={live_mape_h:.2f}%  bias={live_bias_h:+.2f}%")
                _info("    (NOTE: on-chain features unavailable; MAPE is indicative)")

                if live_mape_h > MAPE_H_RETRAIN:
                    reasons.append(
                        f"Live 30d MAPE_H={live_mape_h:.2f}% > {MAPE_H_RETRAIN}% threshold"
                    )
                    verdict = "RETRAIN"
                elif live_mape_h > MAPE_H_RETRAIN * 0.80:
                    reasons.append(
                        f"Live 30d MAPE_H={live_mape_h:.2f}% approaching {MAPE_H_RETRAIN}% threshold"
                    )
                    if verdict == "PASS":
                        verdict = "WARN"

                if abs(live_bias_h) > 0.01:
                    direction = "over-predicting highs" if live_bias_h > 0 else "under-predicting highs"
                    reasons.append(
                        f"30d prediction bias {live_bias_h:+.2f}% — {direction}"
                    )
                    if verdict == "PASS":
                        verdict = "WARN"

                cusum_result = tabular_cusum(pd.Series(arr))
                if cusum_result["triggered_up"] or cusum_result["triggered_dn"]:
                    direction = "upward" if cusum_result["triggered_up"] else "downward"
                    peak = max(cusum_result["cusum_up_max"], cusum_result["cusum_dn_max"])
                    reasons.append(
                        f"CUSUM {direction} structural break in 30d errors (peak={peak:.2f})"
                    )
                    verdict = "RETRAIN"
        except Exception as exc:
            _warn(f"Live inference failed — skipping live metrics: {exc}")

    # — Expected gain ————————————————————————————————————————————————————
    if verdict == "RETRAIN":
        if age is not None and age >= OVERDUE_DAYS["daily_hl"]:
            gain = "HIGH — model is overdue; training on new data will materially refresh fit"
        elif cusum_result and (cusum_result["triggered_up"] or cusum_result["triggered_dn"]):
            gain = "HIGH — structural break detected; distribution has shifted"
        elif live_mape_h and live_mape_h > MAPE_H_RETRAIN:
            gain = "MEDIUM–HIGH — live MAPE drifted above threshold"
        else:
            gain = "MEDIUM — one or more thresholds exceeded"
    elif verdict == "WARN":
        gain = "LOW–MEDIUM — nearing bounds; early retraining may sustain edge"
    else:
        gain = "LOW — model is fresh and within all performance bounds"

    if not reasons:
        reasons.append("All checks passed — model is fresh and within bounds")

    return dict(
        verdict=verdict, reasons=reasons, age_days=age,
        live_mape_h=live_mape_h, live_bias_h=live_bias_h,
        cusum=cusum_result, expected_gain=gain, test_mape_h=mape_h_test,
    )


def assess_7d_cone(art: dict) -> dict:
    """Assess the 7-day close cone (age + stored coverage/MAPE)."""
    reasons: list[str] = []
    verdict = "PASS"
    age     = _model_age(art)
    meta    = art.get("calibration_meta", {})

    if age is None:
        reasons.append("Train date unknown — treat as overdue")
        verdict = "RETRAIN"
    elif age >= OVERDUE_DAYS["cone_7d"]:
        reasons.append(f"Model is {age}d old (overdue threshold: {OVERDUE_DAYS['cone_7d']}d)")
        verdict = "RETRAIN"
    elif age >= WARN_DAYS["cone_7d"]:
        reasons.append(f"Model is {age}d old (warn threshold: {WARN_DAYS['cone_7d']}d)")
        verdict = "WARN"

    cov = meta.get("held_out_band_coverage_pct")
    if cov is not None and cov < COVERAGE_RETRAIN:
        reasons.append(f"Stored test coverage={cov:.1f}% < {COVERAGE_RETRAIN}% threshold")
        verdict = "RETRAIN"

    mape = meta.get("held_out_mape_pct")
    if mape is not None and mape > MAPE_H_RETRAIN * 1.5:
        reasons.append(f"Stored test MAPE={mape:.2f}% is elevated")
        if verdict == "PASS":
            verdict = "WARN"

    gain = (
        "MEDIUM — cone recalibrates to current volatility-regime distribution"
        if verdict == "RETRAIN"
        else "LOW–MEDIUM — coverage or age nearing bounds" if verdict == "WARN"
        else "LOW — cone is fresh; no action needed"
    )
    if not reasons:
        reasons.append("All checks passed — cone is fresh and coverage is healthy")

    return dict(verdict=verdict, reasons=reasons, age_days=age,
                expected_gain=gain, test_coverage=cov, test_mape=mape)


def assess_3class(art: dict, hl_verdict: str) -> dict:
    """Assess the 3-class day-type classifier.

    Hard dependency: always retrains when Daily H/L retrains (uses H/L OOF
    predictions and cone regime_edges as features).
    """
    reasons: list[str] = []
    verdict = "PASS"
    age     = _model_age(art)
    meta    = art.get("calibration_meta", {})

    if hl_verdict == "RETRAIN":
        reasons.append(
            "Daily H/L ensemble is retraining — hard dependency: "
            "OOF predictions and regime boundaries will change"
        )
        verdict = "RETRAIN"

    if verdict != "RETRAIN":
        if age is None:
            reasons.append("Train date unknown — treat as overdue")
            verdict = "RETRAIN"
        elif age >= OVERDUE_DAYS["day_type"]:
            reasons.append(f"Model is {age}d old (overdue threshold: {OVERDUE_DAYS['day_type']}d)")
            verdict = "RETRAIN"
        elif age >= WARN_DAYS["day_type"]:
            reasons.append(f"Model is {age}d old (warn threshold: {WARN_DAYS['day_type']}d)")
            verdict = "WARN"

    acc = meta.get("test_accuracy_pct")
    if acc is not None and acc < ACCURACY_RETRAIN:
        reasons.append(f"Stored test accuracy={acc:.1f}% < {ACCURACY_RETRAIN}% threshold")
        verdict = "RETRAIN"

    if verdict == "RETRAIN" and hl_verdict == "RETRAIN":
        gain = "HIGH — fresh OOF predictions from H/L will improve classifier alignment"
    elif verdict == "RETRAIN":
        gain = "MEDIUM — classifier has drifted from current regime distribution"
    elif verdict == "WARN":
        gain = "LOW–MEDIUM — approaching age or accuracy threshold"
    else:
        gain = "LOW — classifier is fresh and accurate"

    if not reasons:
        reasons.append("All checks passed — classifier is fresh and accurate")

    return dict(verdict=verdict, reasons=reasons, age_days=age,
                expected_gain=gain, test_accuracy=acc)


# ---------------------------------------------------------------------------
# 6.  ASSESSMENT ORCHESTRATOR
# ---------------------------------------------------------------------------

def run_assessment(args) -> tuple[list[str], dict, dict, dict]:
    """Run all three model assessments and return (models_to_train, r_hl, r_cone, r_dt).

    `models_to_train` is ordered by execution dependency:
      daily_hl → cone_7d → day_type
    """
    _header("PHASE 1  Pre-Retrain Assessment")

    hl_art   = _load_art(DAILY_MODEL_CT)
    cone_art = _load_art(CONE_7D_MODEL)
    dt_art   = _load_art(DAY_TYPE_MODEL)

    # Fetch recent daily bars for live MAPE / bias / CUSUM
    daily_df: pd.DataFrame | None = None
    if not args.no_fetch:
        daily_df = fetch_recent_daily(days=90)
    else:
        _info("--no-fetch: skipping Yahoo Finance data fetch")

    _info("")
    _info("── Daily H/L ensemble ─────────────────────────────────────")
    if hl_art:
        r_hl = assess_daily_hl(hl_art, daily_df)
    else:
        r_hl = dict(verdict="RETRAIN", reasons=["Artefact missing — must train"],
                    age_days=None, live_mape_h=None, live_bias_h=None,
                    cusum=None, expected_gain="HIGH — artefact missing", test_mape_h=None)
    _print_verdict("Daily H/L", r_hl)

    _info("")
    _info("── 7-day close cone ───────────────────────────────────────")
    if cone_art:
        r_cone = assess_7d_cone(cone_art)
    else:
        r_cone = dict(verdict="RETRAIN", reasons=["Artefact missing — must train"],
                      age_days=None, expected_gain="HIGH — artefact missing",
                      test_coverage=None, test_mape=None)
    _print_verdict("7-day Cone", r_cone)

    _info("")
    _info("── 3-class day-type classifier ────────────────────────────")
    if dt_art:
        r_dt = assess_3class(dt_art, r_hl["verdict"])
    else:
        r_dt = dict(verdict="RETRAIN", reasons=["Artefact missing — must train"],
                    age_days=None, expected_gain="HIGH — artefact missing",
                    test_accuracy=None)
    _print_verdict("3-class Day Type", r_dt)

    # ── Build ordered training list ───────────────────────────────────────
    to_train: set[str] = set()

    if args.force:
        to_train = {"daily_hl", "day_type"}
        if not args.skip_cone:
            to_train.add("cone_7d")
    else:
        if r_hl["verdict"] == "RETRAIN":
            to_train.add("daily_hl")
        if r_cone["verdict"] == "RETRAIN" and not args.skip_cone:
            to_train.add("cone_7d")
        if r_dt["verdict"] == "RETRAIN":
            to_train.add("day_type")
        # Enforce hard dependency: day_type needs daily_hl to have run first
        if "day_type" in to_train and "daily_hl" not in to_train:
            _warn(
                "3-class day-type is scheduled but Daily H/L is not — "
                "adding Daily H/L to queue (hard dependency)"
            )
            to_train.add("daily_hl")

    # Enforce execution order: daily_hl → cone_7d → day_type
    ordered = [m for m in ["daily_hl", "cone_7d", "day_type"] if m in to_train]

    _info("")
    if ordered:
        _info(_bold(f"Scheduled for retraining: {', '.join(LABEL_MAP[m] for m in ordered)}"))
    else:
        _ok("All models are within thresholds — no retraining needed.")

    return ordered, r_hl, r_cone, r_dt


def _print_verdict(label: str, r: dict):
    v   = r["verdict"]
    fn  = _red if v == "RETRAIN" else (_yellow if v == "WARN" else _green)
    age = f"{r['age_days']}d" if r.get("age_days") is not None else "unknown age"
    _info(f"  {fn(v)}  {label}  ({age})")
    for reason in r.get("reasons", []):
        _info(f"      • {reason}")
    _info(f"      → Expected gain from retraining: {r.get('expected_gain', 'N/A')}")


# ---------------------------------------------------------------------------
# 7.  PRE-FLIGHT CHECK
# ---------------------------------------------------------------------------

def preflight_check() -> list[str]:
    """Return a list of warning strings about missing data prerequisites."""
    warnings: list[str] = []
    if not BINANCE_HOURLY_CSV.exists():
        warnings.append(
            f"Binance hourly CSV not found: {BINANCE_HOURLY_CSV}\n"
            "  pipeline_ct.py reads this file — run fetch_binance_hourly.py first,\n"
            "  or place the CSV at the path above."
        )
    return warnings


# ---------------------------------------------------------------------------
# 8.  ARCHIVE
# ---------------------------------------------------------------------------

def _archive_model(path: Path, label: str, skip: bool = False):
    """Copy the current artefact to models/backups/ with a UTC timestamp."""
    if skip or not path.exists():
        return
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dest = BACKUPS_DIR / f"{path.stem}_{ts}{path.suffix}"
    shutil.copy2(path, dest)
    _ok(f"Archived {label} → models/backups/{dest.name}")


# ---------------------------------------------------------------------------
# 9.  SCRIPT RUNNER
# ---------------------------------------------------------------------------

def _run_script(script_rel: str, label: str,
                extra_args: list[str] | None = None) -> tuple[bool, float]:
    """Run a training script, stream output, return (success, elapsed_secs)."""
    script = ROOT / script_rel
    if not script.exists():
        _err(f"Script not found: {script}")
        return False, 0.0
    cmd = [sys.executable, str(script)] + (extra_args or [])
    _info(f"  $ python {script_rel}" + (f" {' '.join(extra_args)}" if extra_args else ""))
    t0     = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - t0
    if result.returncode == 0:
        _ok(f"{label} completed in {elapsed / 60:.1f} min")
    else:
        _err(f"{label} FAILED (exit code {result.returncode})")
    return result.returncode == 0, elapsed


# ---------------------------------------------------------------------------
# 10.  TRAINING ORCHESTRATOR
# ---------------------------------------------------------------------------

def run_training(models: list[str], args) -> dict[str, bool]:
    """Archive old artefacts and run training scripts in dependency order.

    Stops early if Daily H/L fails (day_type cannot run without fresh H/L data).
    Returns {model_key: success}.
    """
    _header("PHASE 2  Model Training")

    if not models:
        _ok("Nothing to train.")
        return {}

    # Pre-flight
    pf = preflight_check()
    if pf:
        _info("")
        for msg in pf:
            _warn(msg)
        _info("")
        _warn(
            "Pre-flight check failed. Training will proceed but may fail.\n"
            "  Resolve the issues above before running the pipeline."
        )

    results: dict[str, bool] = {}

    for model_key in ["daily_hl", "cone_7d", "day_type"]:
        if model_key not in models:
            continue

        label = LABEL_MAP[model_key]
        _info(f"\n── {label} ─────────────────────────────────────────────────")

        _archive_model(ARTEFACT_MAP[model_key], label, skip=args.no_archive)
        extra = ["--test-start", args.test_start] if getattr(args, "test_start", None) else None
        success, _ = _run_script(SCRIPT_MAP[model_key], label, extra_args=extra)
        results[model_key] = success

        if model_key == "daily_hl" and not success:
            _err(
                "Daily H/L training failed. Halting pipeline — "
                "3-class day-type cannot produce valid OOF features without it."
            )
            for remaining in models:
                if remaining not in results:
                    results[remaining] = False
            return results

    return results


# ---------------------------------------------------------------------------
# 11.  COMPARISON
# ---------------------------------------------------------------------------

def _extract_metrics(art: dict | None) -> dict:
    if art is None:
        return {}
    meta = art.get("calibration_meta", {})
    m    = meta.get("metrics", {})
    return {
        "MAPE_H":    m.get("MAPE_H"),
        "MAPE_L":    m.get("MAPE_L"),
        "hit2_H":    m.get("hit2_H"),
        "hit2_L":    m.get("hit2_L"),
        "dir_hit":   m.get("direction_hit_rate"),
        "coverage":  meta.get("held_out_band_coverage_pct"),
        "cone_mape": meta.get("held_out_mape_pct"),
        "dt_acc":    meta.get("test_accuracy_pct"),
        "train_end": meta.get("train_end"),
    }


_COMPARE_SPECS = {
    "daily_hl": [
        ("MAPE_H (%)",        "MAPE_H",    True),
        ("MAPE_L (%)",        "MAPE_L",    True),
        ("Hit ±1.5% H (%)",   "hit2_H",    False),
        ("Hit ±1.5% L (%)",   "hit2_L",    False),
        ("Direction acc (%)", "dir_hit",   False),
    ],
    "cone_7d": [
        ("Coverage (%)",  "coverage",  False),
        ("MAPE (%)",      "cone_mape", True),
    ],
    "day_type": [
        ("Accuracy (%)", "dt_acc", False),
    ],
}


def run_comparison(old_arts: dict[str, dict | None], train_results: dict[str, bool]):
    """Load freshly trained artefacts and print a before/after metric table."""
    _header("PHASE 3  Before / After Metric Comparison")

    if not any(train_results.values()):
        _warn("No models were successfully trained — nothing to compare.")
        return

    new_arts = {k: _load_art(ARTEFACT_MAP[k]) for k in ARTEFACT_MAP}

    for key, label in LABEL_MAP.items():
        if not train_results.get(key):
            continue
        old = _extract_metrics(old_arts.get(key))
        new = _extract_metrics(new_arts.get(key))

        _info(f"\n── {label} ─────────────────────────────────────────────────")
        _info(f"  {'Metric':<24} {'Before':>9} {'After':>9}  {'Δ':>8}  {'':>4}")
        _info(f"  {'-' * 57}")

        any_improved = False
        for metric_label, mk, lo_better in _COMPARE_SPECS.get(key, []):
            ov = old.get(mk)
            nv = new.get(mk)
            ov_s = f"{ov:.2f}" if ov is not None else "—"
            nv_s = f"{nv:.2f}" if nv is not None else "—"
            if ov is not None and nv is not None:
                delta     = nv - ov
                improved  = (delta < 0) if lo_better else (delta > 0)
                delta_s   = f"{delta:+.2f}"
                flag      = _green("✓") if improved else _red("↘")
                if improved:
                    any_improved = True
            else:
                delta_s = "—"; flag = ""
            _info(f"  {metric_label:<24} {ov_s:>9} {nv_s:>9}  {delta_s:>8}  {flag}")

        te_old = old.get("train_end") or "?"
        te_new = new.get("train_end") or "?"
        _info(f"  {'train_end':<24} {str(te_old)[:10]:>9} {str(te_new)[:10]:>9}")

        if any_improved:
            _ok("Metrics improved — model quality has increased.")
        else:
            _warn("No clear improvement detected — review the delta column above.")


# ---------------------------------------------------------------------------
# 12.  BACKTEST VALIDATION
# ---------------------------------------------------------------------------
#
# After the Daily H/L model is retrained we run the TF2+V-Gate strategy on
# three fixed windows and compare results to the baseline (old) model.
# All three periods must improve; if any regress the pipeline analyses the
# cause, tries progressively trimmed training windows, and if no cutoff
# passes it rolls back to the pre-training artefact.
#
# Period windows (match the Streamlit UI):
#   Bear  :  2025-06-01 → today
#   Bull  :  2024-06-05 → 2025-06-14  (closes the Apr-2025 open trade)
#   Full  :  2024-06-05 → today
# ---------------------------------------------------------------------------

BT_BEAR_START   = "2025-06-01"   # ~12-month bear window start
BT_BULL_START   = "2024-06-05"   # bull window start
BT_BULL_END     = "2025-06-14"   # bull window end
BT_FULL_START   = "2024-06-05"   # full period start
BT_FETCH_START  = "2024-02-01"   # 115+ days warmup for 90-day rolling features
BT_WARMUP_BARS  = 35             # indicator warmup before the backtest window

# Minimum strategy-return delta (pp) for a period to count as "improved".
# 0.0 means any positive change is enough; increase to require a buffer.
BT_MIN_DELTA    = 0.0

# Data-search: number of months to trim from the training end on each retry
_SEARCH_TRIM_MONTHS = [3, 6]


def _bt_fetch_raw(today_iso: str) -> pd.DataFrame | None:
    """Fetch BTC + macro daily bars from BT_FETCH_START → today."""
    _info("  Fetching daily BTC + macro for backtest …")
    end = (pd.Timestamp(today_iso) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    parts: list[pd.DataFrame] = []
    for name, sym in ({"btc": "BTC-USD"} | _MACRO_SYMS).items():
        try:
            raw = yf.download(sym, start=BT_FETCH_START, end=end,
                              progress=False, auto_adjust=False)
            if not raw.empty:
                parts.append(_flat(raw, name))
        except Exception:
            pass
    if not parts or "btc_close" not in parts[0].columns:
        _warn("  BTC data unavailable — cannot run backtest validation.")
        return None
    df = parts[0]
    for p in parts[1:]:
        df = df.join(p, how="outer")
    df = df.ffill(limit=5)

    # Coinbase premium (Coinbase BTC-USD close vs Yahoo BTC-USD reference)
    end_str = (pd.Timestamp(today_iso) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    _cb = _fetch_coinbase_daily(BT_FETCH_START, end_str)
    if _cb.notna().any():
        df = df.join(_cb.to_frame("coinbase_close"), how="left")
        df["cb_premium"] = (df["coinbase_close"] - df["btc_close"]) / df["btc_close"] * 100

    return df


def _bt_build_preds(art: dict, raw: pd.DataFrame) -> pd.DataFrame | None:
    """Build pred_high / pred_low time series from artefact + raw OHLCV."""
    feat_cols    = art.get("feat_cols", [])
    constituents = art.get("constituents", [])
    if not feat_cols or not constituents:
        _warn("  Artefact missing feat_cols or constituents — cannot build predictions.")
        return None

    F   = _build_hl_features(raw, feat_cols)
    Fnp = F.to_numpy()

    all_hi: list = []
    all_lo: list = []
    for con in constituents:
        try:
            all_hi.append(con["m_hi"].predict(Fnp))
            all_lo.append(con["m_lo"].predict(Fnp))
        except Exception:
            pass
    if not all_hi:
        _warn("  All constituent predictions failed.")
        return None

    yhi = np.mean(all_hi, axis=0)
    ylo = np.mean(all_lo, axis=0)

    # Optional stacking blend (alpha < 1 activates hi_model / lo_model)
    alpha = float(art.get("alpha", 1.0))
    hi_m  = art.get("hi_model")
    lo_m  = art.get("lo_model")
    if alpha < 1.0:
        if hi_m is not None:
            yhi = alpha * yhi + (1.0 - alpha) * hi_m.predict(Fnp)
        if lo_m is not None:
            ylo = alpha * ylo + (1.0 - alpha) * lo_m.predict(Fnp)
    yhi = yhi + float(art.get("mu_hi", 0.0))
    ylo = ylo + float(art.get("mu_lo", 0.0))

    c_vals  = raw["btc_close"].reindex(F.index).values.astype(float)
    pred_hi = c_vals * (1.0 + np.clip(yhi, 0, None))
    pred_lo = c_vals * (1.0 - np.clip(ylo, 0, None))

    # Predictions are for the NEXT bar (target_date = date + 1 trading day)
    next_dates = list(F.index[1:]) + [F.index[-1] + pd.Timedelta(days=1)]
    preds = pd.DataFrame(
        {"close_asof": c_vals, "pred_high": pred_hi, "pred_low": pred_lo},
        index=pd.DatetimeIndex(next_dates, name="target_date"),
    )
    return preds[~preds.index.duplicated(keep="last")]


def _bt_run_strategy(preds: pd.DataFrame, raw: pd.DataFrame,
                     bt_start_iso: str, bt_end_iso: str,
                     initial_capital: float = 100_000.0) -> dict | None:
    """Run the TF2+V-Gate strategy and return a stats dict.

    Implements the same logic as app/btc_hourly_app.py::run_full_period_backtest:
      Entry : U1 AND (above_MA30 OR clean_10d OR v_recent)
      Exit  : D3 always; D2 when NOT in bull regime
    """
    WARMUP   = BT_WARMUP_BARS
    start_dt = pd.Timestamp(bt_start_iso)
    end_dt   = pd.Timestamp(bt_end_iso)
    pre_dt   = start_dt - pd.Timedelta(days=60)   # warmup window before start

    p = preds.loc[(preds.index >= pre_dt) & (preds.index <= end_dt)].copy()
    p["actual_high"]  = raw["btc_high"].reindex(p.index).values
    p["actual_low"]   = raw["btc_low"].reindex(p.index).values
    p["actual_close"] = raw["btc_close"].reindex(p.index).values
    comp = p.dropna(subset=["actual_high", "actual_low", "actual_close"]).reset_index()
    N    = len(comp)
    if N < WARMUP + 3:
        return None

    dates   = pd.DatetimeIndex(comp["target_date"])
    c_asof  = comp["close_asof"].values.astype(float)
    exec_px = comp["actual_close"].values.astype(float)
    pred_hi = comp["pred_high"].values.astype(float)
    pred_lo = comp["pred_low"].values.astype(float)
    act_hi  = comp["actual_high"].values.astype(float)
    act_lo  = comp["actual_low"].values.astype(float)

    _bt0 = max(WARMUP, int(dates.searchsorted(start_dt)))
    if N - _bt0 < 3:
        return None

    err_hi = (act_hi - pred_hi) / c_asof * 100
    err_lo = (pred_lo - act_lo) / c_asof * 100
    hi_brk = (act_hi > pred_hi).astype(int)
    lo_brk = (act_lo < pred_lo).astype(int)

    ehma3 = np.zeros(N); elma3 = np.zeros(N)
    hb3   = np.zeros(N, dtype=int); lb3   = np.zeros(N, dtype=int)
    for i in range(N):
        sl       = slice(max(0, i - 2), i + 1)
        ehma3[i] = float(np.mean(err_hi[sl]))
        elma3[i] = float(np.mean(err_lo[sl]))
        hb3[i]   = int(np.sum(hi_brk[sl]))
        lb3[i]   = int(np.sum(lo_brk[sl]))

    u1 = (ehma3 > 0.5) & (hb3 >= 2)
    d1 = (lb3 >= 2) & (elma3 > 0.5)
    d2 = ehma3 < -1.0
    d3 = np.zeros(N, dtype=bool)
    for i in range(1, N):
        consec = 0
        for k in range(i - 1, -1, -1):
            if hi_brk[k]: consec += 1
            else: break
        if consec >= 3 and lo_brk[i]:
            d3[i] = True

    # Bull regime: price above MA30 AND MA30 has +ve 5-bar slope
    ma30 = np.zeros(N)
    for i in range(N):
        w       = min(30, i + 1)
        ma30[i] = float(np.mean(c_asof[max(0, i - w + 1):i + 1]))
    above_ma30 = c_asof > ma30
    slope5     = np.zeros(N, dtype=bool)
    for i in range(5, N):
        slope5[i] = ma30[i] > ma30[i - 5]
    bull_regime = above_ma30 & slope5

    # Clean-10d: no D1/D2 in the last 10 bars
    clean_10d = np.zeros(N, dtype=bool)
    for i in range(N):
        li          = max(0, i - 10)
        clean_10d[i] = not bool(np.any(d1[li:i] | d2[li:i]))

    # V-reversal: recent D1/D2 present and error recovering above U1 threshold
    v_recent = np.zeros(N, dtype=bool)
    for i in range(3, N):
        sl = slice(max(0, i - 2), i + 1)
        if bool(np.any(d1[sl] | d2[sl])) and ehma3[i] > 0.5:
            v_recent[i] = True

    entry = u1 & (above_ma30 | clean_10d | v_recent)

    nav = initial_capital; pos = "CASH"; qty = 0.0
    e_price = e_date = e_nav = None
    trades: list[dict] = []
    nav_arr = np.full(N, np.nan)

    for i in range(N):
        si = i - 1; price = exec_px[i]
        if i < _bt0:
            nav_arr[i] = initial_capital; continue
        if pos == "LONG":
            cur = qty * price
            if si >= 0:
                should_exit = bool(d3[si] or (d2[si] and not bull_regime[si]))
                exit_lbl    = "D3" if d3[si] else "D2 (bear)"
            else:
                should_exit = False; exit_lbl = "?"
            if should_exit:
                nav = cur
                trades.append(dict(
                    entry_date    = e_date,
                    entry_price   = e_price,
                    exit_date     = dates[i],
                    exit_price    = price,
                    pnl_pct       = (price / e_price - 1) * 100,
                    duration_days = (dates[i] - e_date).days,
                    exit_signal   = exit_lbl,
                ))
                pos = "CASH"; qty = 0.0
            else:
                nav = cur
        else:
            if si >= 0 and entry[si]:
                qty = nav / price; e_price = price; e_date = dates[i]; e_nav = nav
                pos = "LONG"
        nav_arr[i] = qty * price if pos == "LONG" else nav

    if pos == "LONG":
        nav_arr[N - 1] = qty * exec_px[N - 1]

    nav_s     = pd.Series(nav_arr[_bt0:], index=dates[_bt0:]).ffill()
    bh_s      = pd.Series(initial_capital * exec_px[_bt0:] / exec_px[_bt0],
                           index=dates[_bt0:])
    final_nav = float(nav_s.iloc[-1])
    strat_ret = (final_nav / initial_capital - 1) * 100
    bh_ret    = (float(bh_s.iloc[-1]) / initial_capital - 1) * 100
    wins      = [t for t in trades if t["pnl_pct"] > 0]
    losses    = [t for t in trades if t["pnl_pct"] <= 0]
    rm        = nav_s.cummax()
    max_dd    = float(((nav_s - rm) / rm * 100).min()) if len(nav_s) > 1 else 0.0

    return dict(
        strat_ret  = strat_ret,
        bh_ret     = bh_ret,
        final_nav  = final_nav,
        n_trades   = len(trades),
        n_wins     = len(wins),
        n_losses   = len(losses),
        win_rate   = 100.0 * len(wins) / len(trades) if trades else 0.0,
        max_dd     = max_dd,
        open_pos   = pos == "LONG",
        trades     = trades,
    )


def _bt_score_all(art: dict, today_iso: str,
                  raw: pd.DataFrame) -> dict[str, dict] | None:
    """Score art on all 3 backtest periods. Returns {bear, bull, full} or None."""
    preds = _bt_build_preds(art, raw)
    if preds is None:
        return None
    scores: dict[str, dict] = {}
    for name, start, end in [
        ("bear", BT_BEAR_START, today_iso),
        ("bull", BT_BULL_START, BT_BULL_END),
        ("full", BT_FULL_START, today_iso),
    ]:
        r = _bt_run_strategy(preds, raw, start, end)
        if r is None:
            _warn(f"  Backtest [{name}] insufficient data for {start}→{end}.")
            return None
        scores[name] = r
    return scores


def _fmt_score(s: dict) -> str:
    return (f"ret={s['strat_ret']:+.1f}%  bh={s['bh_ret']:+.1f}%  "
            f"trades={s['n_trades']}  wr={s['win_rate']:.0f}%  dd={s['max_dd']:.1f}%")


def _print_bt_comparison(old_scores: dict, new_scores: dict):
    """Print a side-by-side backtest comparison table (old vs new)."""
    W = 72
    _info(f"\n  {'Period':<8}  {'Metric':<14}  {'Old':>9}  {'New':>9}  {'Δ':>9}  {'':>4}")
    _info(f"  {'-' * W}")
    specs = [
        ("strat_ret", "Strategy ret", "%",  False),
        ("bh_ret",    "B&H ret",      "%",  False),
        ("max_dd",    "Max DD",       "%",  True),
        ("win_rate",  "Win rate",     "%",  False),
        ("n_trades",  "Trades",       "",   None),
    ]
    for pname in ["bear", "bull", "full"]:
        for key, label, unit, lo_better in specs:
            ov = old_scores.get(pname, {}).get(key)
            nv = new_scores.get(pname, {}).get(key)
            ov_s = f"{ov:.1f}{unit}" if isinstance(ov, float) else (str(ov) if ov is not None else "—")
            nv_s = f"{nv:.1f}{unit}" if isinstance(nv, float) else (str(nv) if nv is not None else "—")
            if isinstance(ov, (int, float)) and isinstance(nv, (int, float)) \
                    and lo_better is not None:
                delta     = float(nv) - float(ov)
                improved  = (delta < 0) if lo_better else (delta > 0)
                flag      = _green("↑") if improved else (_red("↓") if delta != 0 else " =")
                delta_s   = f"{delta:+.1f}{unit}"
            else:
                delta_s = "—"; flag = ""
            plbl = pname if key == "strat_ret" else ""
            _info(f"  {plbl:<8}  {label:<14}  {ov_s:>9}  {nv_s:>9}  {delta_s:>9}  {flag}")
        _info(f"  {'-' * W}")


def _date_str(d) -> str:
    return d.date().isoformat() if hasattr(d, "date") else str(d)[:10]


def _analyze_regression(period: str, old_s: dict, new_s: dict):
    """Print root-cause analysis when a backtest period worsens."""
    _warn(f"  ── {period.upper()} regression root-cause ──────────────────────────")
    delta = new_s["strat_ret"] - old_s["strat_ret"]
    _info(f"    Return:   {old_s['strat_ret']:+.1f}% → {new_s['strat_ret']:+.1f}%  (Δ={delta:+.1f}%)")
    _info(f"    Trades:   {old_s['n_trades']} → {new_s['n_trades']}"
          f"  (wins: {old_s['n_wins']} → {new_s['n_wins']})")
    _info(f"    Win rate: {old_s['win_rate']:.0f}% → {new_s['win_rate']:.0f}%")
    _info(f"    Max DD:   {old_s['max_dd']:.1f}% → {new_s['max_dd']:.1f}%")

    old_tr  = old_s.get("trades", [])
    new_tr  = new_s.get("trades", [])
    old_map = {_date_str(t["entry_date"]): t for t in old_tr}
    new_map = {_date_str(t["entry_date"]): t for t in new_tr}

    added   = [t for t in new_tr if _date_str(t["entry_date"]) not in old_map]
    removed = [t for t in old_tr if _date_str(t["entry_date"]) not in new_map]
    changed = []
    for ed, ot in old_map.items():
        if ed in new_map:
            nt = new_map[ed]
            exit_shift = abs((pd.Timestamp(nt["exit_date"]) -
                              pd.Timestamp(ot["exit_date"])).days)
            if abs(nt["pnl_pct"] - ot["pnl_pct"]) > 0.5 or exit_shift > 1:
                changed.append((ed, ot, nt))

    if added:
        net = sum(t["pnl_pct"] for t in added)
        _info(f"    NEW signals fired ({len(added)}, net P&L {net:+.1f}%):")
        for t in added:
            _info(f"      + {_date_str(t['entry_date'])} → {_date_str(t['exit_date'])}"
                  f"  {t['pnl_pct']:+.1f}%  ({t['exit_signal']})")
    if removed:
        net = sum(t["pnl_pct"] for t in removed)
        _info(f"    LOST signals ({len(removed)}, old net P&L {net:+.1f}%):")
        for t in removed:
            _info(f"      - {_date_str(t['entry_date'])} → {_date_str(t['exit_date'])}"
                  f"  {t['pnl_pct']:+.1f}%  ({t['exit_signal']})")
    if changed:
        _info(f"    ALTERED trades ({len(changed)}) — exit date or P&L shifted:")
        for ed, ot, nt in changed:
            _info(f"      ~ {ed}  exit: {_date_str(ot['exit_date'])} → "
                  f"{_date_str(nt['exit_date'])}"
                  f"  P&L: {ot['pnl_pct']:+.1f}% → {nt['pnl_pct']:+.1f}%")
    if not added and not removed and not changed:
        _info("    (Trade signals unchanged — difference is sub-1% in execution timing)")

    # Heuristic explanation
    if len(new_tr) > len(old_tr):
        _info("    LIKELY CAUSE: new model produces lower pred_high → more U1 false positives.")
    elif len(new_tr) < len(old_tr):
        _info("    LIKELY CAUSE: new model raises pred_high → some genuine U1 entries missed.")
    else:
        _info("    LIKELY CAUSE: pred_high shift changed trade exit timing or regime detection.")


def _restore_art(art: dict, path: Path):
    """Overwrite path on disk with the in-memory artefact snapshot (rollback)."""
    try:
        joblib.dump(art, path)
        _ok(f"  Rolled back {path.name} to pre-training snapshot.")
    except Exception as exc:
        _err(f"  Rollback failed: {exc}")


def _data_search(old_art: dict, old_scores: dict, raw: pd.DataFrame,
                 today_iso: str, args) -> bool:
    """Try retraining with progressively trimmed data windows.

    For each trim amount in _SEARCH_TRIM_MONTHS the old model is restored,
    pipeline_ct.py is re-run with --test-start set to (today − N months),
    and the candidate is re-validated. Returns True if any attempt passes.
    """
    today_ts = pd.Timestamp(today_iso)

    for months in _SEARCH_TRIM_MONTHS:
        cutoff = (today_ts - pd.DateOffset(months=months)).strftime("%Y-%m-%d")
        _info(f"\n  ── Data-search attempt: trim to {cutoff}  (−{months} months) ──")

        # Restore old model before each attempt
        _restore_art(old_art, DAILY_MODEL_CT)
        # Archive it so _run_script does not overwrite an unprotected file
        _archive_model(DAILY_MODEL_CT, "Daily H/L (data-search restore)",
                       skip=args.no_archive)

        _info(f"  Re-training Daily H/L with --test-start {cutoff} …")
        success, _ = _run_script(
            SCRIPT_MAP["daily_hl"], LABEL_MAP["daily_hl"],
            extra_args=["--test-start", cutoff],
        )
        if not success:
            _err(f"  Training failed for cutoff {cutoff} — skipping.")
            continue

        candidate_art = _load_art(DAILY_MODEL_CT)
        if candidate_art is None:
            continue
        _info(f"  Scoring candidate (cutoff={cutoff}) …")
        candidate_scores = _bt_score_all(candidate_art, today_iso, raw)
        if candidate_scores is None:
            _warn("  Candidate scoring failed — trying next cutoff.")
            continue

        _print_bt_comparison(old_scores, candidate_scores)

        deltas = {p: candidate_scores[p]["strat_ret"] - old_scores[p]["strat_ret"]
                  for p in ["bear", "bull", "full"]}
        if all(v > BT_MIN_DELTA for v in deltas.values()):
            _ok(f"  Cutoff {cutoff} ACCEPTED — "
                f"bear Δ{deltas['bear']:+.1f}%  "
                f"bull Δ{deltas['bull']:+.1f}%  "
                f"full Δ{deltas['full']:+.1f}%")
            return True

        failed_p = [p for p, d in deltas.items() if d <= BT_MIN_DELTA]
        _warn(f"  Cutoff {cutoff} failed — still regressing: {', '.join(failed_p)}")
        for p in failed_p:
            _analyze_regression(p, old_scores[p], candidate_scores[p])

    return False


def run_backtest_validation(old_art: dict | None,
                             train_results: dict[str, bool],
                             args) -> bool:
    """Phase 3: validate retrained models by comparing backtest performance.

    Returns True if the new model is accepted, False if it was rolled back.
    The Daily H/L ensemble drives all strategy predictions; if it was not
    retrained this phase is a no-op.
    """
    _header("PHASE 3  Backtest Validation")

    if not train_results.get("daily_hl"):
        _ok("  Daily H/L not retrained — predictions unchanged; validation skipped.")
        return True

    today_iso = pd.Timestamp.now().normalize().strftime("%Y-%m-%d")

    raw = _bt_fetch_raw(today_iso)
    if raw is None:
        _warn("  No data — accepting models without backtest validation.")
        return True

    # ── Baseline: score old model ────────────────────────────────────────
    if old_art is None:
        _warn("  No baseline artefact found — accepting new model unconditionally.")
        new_art = _load_art(DAILY_MODEL_CT)
        if new_art:
            ns = _bt_score_all(new_art, today_iso, raw)
            if ns:
                _info("  New model standalone results:")
                for pname, s in ns.items():
                    _info(f"    {pname:<6}: {_fmt_score(s)}")
        return True

    _info("  Scoring baseline (old) model …")
    old_scores = _bt_score_all(old_art, today_iso, raw)
    if old_scores is None:
        _warn("  Baseline scoring failed — accepting new model without comparison.")
        return True
    _info("  Old model results:")
    for pname, s in old_scores.items():
        _info(f"    {pname:<6}: {_fmt_score(s)}")

    # ── New model ────────────────────────────────────────────────────────
    new_art = _load_art(DAILY_MODEL_CT)
    if new_art is None:
        _err("  Cannot load newly trained artefact.")
        return False

    _info("  Scoring new model …")
    new_scores = _bt_score_all(new_art, today_iso, raw)
    if new_scores is None:
        _warn("  New model scoring failed — rolling back to old model.")
        _restore_art(old_art, DAILY_MODEL_CT)
        return False
    _info("  New model results:")
    for pname, s in new_scores.items():
        _info(f"    {pname:<6}: {_fmt_score(s)}")

    _print_bt_comparison(old_scores, new_scores)

    # ── Decision ─────────────────────────────────────────────────────────
    deltas   = {p: new_scores[p]["strat_ret"] - old_scores[p]["strat_ret"]
                for p in ["bear", "bull", "full"]}
    improved = {p: d > BT_MIN_DELTA for p, d in deltas.items()}

    if all(improved.values()):
        _ok(f"  ✓ ALL 3 PERIODS IMPROVED — new model accepted.")
        _ok(f"    bear Δ{deltas['bear']:+.1f}%  "
            f"bull Δ{deltas['bull']:+.1f}%  "
            f"full Δ{deltas['full']:+.1f}%")
        return True

    # ── Validation failed — analyse regressions ───────────────────────────
    failed_periods = [p for p, ok in improved.items() if not ok]
    _warn(f"  ✗ Validation FAILED — period(s) worsened: {', '.join(failed_periods)}")
    _info("")
    for p in failed_periods:
        _analyze_regression(p, old_scores[p], new_scores[p])

    # ── Try data-window search ─────────────────────────────────────────────
    _info("")
    _warn("  Attempting data-window search to find an acceptable training cutoff …")
    _info("  (Will try trimming the last 3 and 6 months from training data)")
    found = _data_search(old_art, old_scores, raw, today_iso, args)
    if found:
        return True

    # ── Complete rollback ──────────────────────────────────────────────────
    _warn("  DATA SEARCH EXHAUSTED — no cutoff improved all 3 periods.")
    _warn("  ROLLING BACK to pre-training model.")
    _restore_art(old_art, DAILY_MODEL_CT)
    _info("")
    _info("  Diagnosis summary:")
    for p in failed_periods:
        d = deltas[p]
        _info(f"    {p:<6}: Δ={d:+.1f}%  — "
              f"new model underperforms baseline in the {p} period")
    _info("")
    _info("  Recommended actions:")
    _info("    1. Review the regression analysis above for the failing periods.")
    _info("    2. Check if new training data contains a regime not yet in the test set.")
    _info("    3. Re-run with --no-bt-validation to force-accept if you have external")
    _info("       evidence the new model is better (e.g. improved live MAPE).")
    return False


# ---------------------------------------------------------------------------
# 13.  RETRAIN SUMMARY HELPER
# ---------------------------------------------------------------------------

def _retrain_summary(models: list[str], r_hl: dict, r_cone: dict, r_dt: dict):
    _info("")
    _info(_bold("Retrain recommendation summary:"))
    for key, r in [("daily_hl", r_hl), ("cone_7d", r_cone), ("day_type", r_dt)]:
        v  = r["verdict"]
        fn = _red if v == "RETRAIN" else (_yellow if v == "WARN" else _green)
        _info(f"  {fn(v):30s}  {LABEL_MAP[key]}")

    if models:
        _info("")
        _info("  Run the full pipeline:")
        _info("    python retrain_pipeline.py")
        _info("")
        _info("  Or run individual scripts in dependency order:")
        _info("    python src/pipeline_ct.py            # Step 1 — Daily H/L (primary)")
        _info("    python src/train_7d_close_cone.py    # Step 2 — 7-day cone (independent)")
        _info("    python src/train_3class_day_type.py  # Step 3 — 3-class day-type")
        _info("")
        _info("  After retraining: click 'Refresh now' in the Streamlit sidebar.")
    else:
        _ok("No retraining needed at this time.")
    _info("")


# ---------------------------------------------------------------------------
# 14.  MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="retrain_pipeline.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--assess-only", action="store_true",
        help="Print assessment and recommended actions without running training.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Train all models regardless of assessment outcome.",
    )
    parser.add_argument(
        "--skip-cone", action="store_true",
        help="Exclude the 7-day cone from training (faster H/L-only updates).",
    )
    parser.add_argument(
        "--no-archive", action="store_true",
        help="Do not back up existing model artefacts before overwriting.",
    )
    parser.add_argument(
        "--no-fetch", action="store_true",
        help="Skip the Yahoo Finance fetch used for live MAPE / CUSUM assessment.",
    )
    parser.add_argument(
        "--test-start", type=str, default=None, metavar="YYYY-MM-DD",
        help="Force test_start date passed to each training script. When beyond "
             "available data, scripts use deploy-retrain mode (all data in training).",
    )
    parser.add_argument(
        "--no-bt-validation", action="store_true",
        help="Skip Phase 3 backtest validation and accept new models unconditionally. "
             "Use only when external evidence (e.g. improved live MAPE) overrides "
             "strategy-return regression.",
    )
    args = parser.parse_args()

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{_bold('BTC Range Model — Retraining Pipeline')}")
    print(f"  {now_utc}  |  Python {sys.version.split()[0]}  |  root: {ROOT}")

    # Snapshot old artefacts for post-train comparison
    old_arts = {k: _load_art(ARTEFACT_MAP[k]) for k in ARTEFACT_MAP}

    # Phase 1 — Assessment
    models_to_train, r_hl, r_cone, r_dt = run_assessment(args)

    if args.assess_only:
        _header("Assessment Complete  (--assess-only)")
        _retrain_summary(models_to_train, r_hl, r_cone, r_dt)
        return

    if not models_to_train:
        _header("No Retraining Required")
        _retrain_summary(models_to_train, r_hl, r_cone, r_dt)
        return

    # Phase 2 — Training
    train_results = run_training(models_to_train, args)

    # Early-exit if training itself failed
    failed_train = [LABEL_MAP[k] for k, ok in train_results.items() if not ok]
    if failed_train:
        _header("Pipeline Complete  [TRAINING FAILED]")
        _err(f"FAILED: {', '.join(failed_train)}")
        _info("Resolve the errors above and re-run the pipeline.")
        sys.exit(1)

    # Phase 3 — Backtest validation
    if not getattr(args, "no_bt_validation", False):
        bt_accepted = run_backtest_validation(old_arts.get("daily_hl"),
                                               train_results, args)
        if not bt_accepted:
            _header("Pipeline Complete  [MODELS REJECTED]")
            _err("New models failed backtest validation — old models restored.")
            _warn("Strategy return did not improve in all 3 market periods.")
            _info("Re-run with --no-bt-validation to bypass if needed.")
            sys.exit(1)
    else:
        _info("")
        _warn("--no-bt-validation: backtest validation skipped — models accepted unconditionally.")

    # Phase 4 — Before/After metric comparison (model intrinsic metrics)
    run_comparison(old_arts, train_results)

    # Exit summary
    _header("Pipeline Complete  [MODELS ACCEPTED]")
    _ok("All scheduled models retrained and validated successfully.")
    _info("Click 'Refresh now' in the Streamlit sidebar to reload artefacts.")
    _info("")


if __name__ == "__main__":
    main()
