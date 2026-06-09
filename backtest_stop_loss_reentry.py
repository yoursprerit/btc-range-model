#!/usr/bin/env python3
"""
Stop-Loss Re-Entry Criteria Analysis
=====================================
Evaluates smart re-entry criteria after stop-loss exits for BTC, MSTR, and MSTU.

Why this analysis:
  Adding a hard stop loss to TF2+V-Gate "cuts" positions during bull-market
  consolidations — the downside protection comes at the cost of missed upside
  when the rally resumes.  This script quantifies how much upside is recovered
  by different re-entry strategies after a stop fires.

Baseline:
  B0 — TF2+V-Gate, NO stop loss (current live strategy)

Stop-loss + re-entry variants (tested for each asset):
  SL0 — stop loss + standard re-entry  (U1 + MA30/clean7d, same as base entry)
  SL1 — stop loss + price-recovery gate (BTC must recover ≥3% above SL-exit BTC price)
  SL2 — stop loss + EMA10 reclaim       (BTC close > EMA10 on re-entry signal bar)
  SL3 — stop loss + 5-bar cooldown      (wait 5 bars after SL then normal entry)
  SL4 — stop loss + regime-adaptive     (bull regime: immediate; bear regime: 10-bar cooldown)

Stop-loss percentages (calibrated to each asset's typical volatility):
  BTC:  −10%  (about 2× daily ATR-14, catches deep corrections not minor noise)
  MSTR: −20%  (≈2× BTC SL; MSTR beta vs BTC ≈ 1.5–2.5×)
  MSTU: −25%  (≈2.5× BTC SL; MSTU = 2× daily leveraged MSTR, ≈3–4× BTC)

Backtesting periods:
  1. Bull  (Sep 2024 → Sep 2025)  BTC +93%  [in-sample for CT model]
  2. Bear  (Jun 2025 → May 2026)  BTC −35%  [partly OOS]
  3. OOS   (Sep 2025 → May 2026)  BTC −33%  [fully OOS]
  4. OOS-Recent (Mar → May 2026)  BTC (recent)  [fully OOS]
  5. Full  (Jun 2024 → May 2026)  2-year combined

MSTU (T-Rex 2X Long MSTR) launched ≈ Sep 18 2024 — only available for periods
starting on/after that date.

All executions: 1-bar lag (signal on bar i → trade at bar i+1 close).
"""

import sys, warnings, joblib, requests
warnings.filterwarnings("ignore")
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import yfinance as yf

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
ASSET_CONFIG = {
    "BTC":  {"ticker": "BTC-USD", "sl_pct": 0.10, "name": "Bitcoin"},
    "MSTR": {"ticker": "MSTR",    "sl_pct": 0.20, "name": "MicroStrategy"},
    "MSTU": {"ticker": "MSTU",    "sl_pct": 0.25, "name": "T-Rex 2X Long MSTR (2×)"},
}

REENTRY_LABELS = {
    "B0":  "TF2+VGate (no SL)  ← baseline",
    "SL0": "SL + standard re-entry",
    "SL1": "SL + price-recovery +3%",
    "SL2": "SL + EMA10 reclaim",
    "SL3": "SL + 5-bar cooldown",
    "SL4": "SL + regime-adaptive",
}

PERIODS = {
    "Bull (Sep24→Sep25)":     ("2024-09-17", "2025-09-17", "2024-06-01"),
    "Bear (Jun25→May26)":     ("2025-06-01", "2026-05-26", "2025-03-01"),
    "OOS (Sep25→May26)":      ("2025-09-19", "2026-05-26", "2025-06-01"),
    "OOS-Recent (Mar→May26)": ("2026-03-01", "2026-05-26", "2025-12-01"),
    "Full (Jun24→May26)":     ("2024-06-01", "2026-05-26", "2024-03-01"),
}

INITIAL_CAPITAL = 100_000.0
WARMUP          = 35  # bars before backtest window starts

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
    """Fetch BTC OHLCV + macro + on-chain into one aligned daily DataFrame."""
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
                timeout=20,
            )
            vals = r.json().get("values", [])
            s = pd.Series(
                {pd.Timestamp(v["x"], unit="s").normalize(): v["y"] for v in vals},
                name=f"oc_{m.replace('-','_')}", dtype=float,
            )
            s = s[~s.index.duplicated(keep="last")].sort_index()
            s.index = pd.DatetimeIndex(s.index).tz_localize(None)
            df[s.name] = s.reindex(df.index).ffill(limit=7)
            ok += 1
        except Exception:
            pass
    print(f"{ok}/{len(_ONCHAIN)} OK")
    return df


def fetch_asset_prices(ticker: str, fetch_start: str, fetch_end: str) -> pd.Series:
    """Fetch split-adjusted closes for MSTR / MSTU, forward-filled daily."""
    try:
        d = _yf_dl(ticker, fetch_start, fetch_end)
        px = d["Close"].sort_index()
        all_days = pd.date_range(px.index[0], max(px.index[-1], pd.Timestamp(fetch_end)), freq="D")
        return px.reindex(all_days).ffill()
    except Exception as e:
        print(f"    [WARN] {ticker}: {e}")
        return pd.Series(dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# CT Predictions
# ─────────────────────────────────────────────────────────────────────────────
def build_ct_preds(df: pd.DataFrame, fetch_start: str = "2020-01-01", fetch_end: str = "2030-01-01") -> pd.DataFrame:
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
    g = c.diff().clip(lower=0).rolling(14).mean()
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
    f["vol_chg_1"]  = np.log(v).diff()
    f["vol_z_20"]   = (np.log(v) - np.log(v).rolling(20).mean()) / np.log(v).rolling(20).std()
    f["vol_ma_ratio"] = v / v.rolling(20).mean()
    dow = df.index.dayofweek
    for i in range(6): f[f"dow_{i}"] = (dow == i).astype(float)
    for nm in ["spx", "ndx", "vix", "gold", "dxy", "tnx", "eth"]:
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

    # ── Coinbase premium (fetch daily candles, set to 0 if unavailable) ──
    import time as _time
    _CB_URL  = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
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
        # Fill with 0 (neutral: no Coinbase/reference exchange premium)
        f["cb_premium"]     = 0.0
        f["cb_premium_ma3"] = 0.0
        f["cb_premium_z7"]  = 0.0
        print("    Coinbase premium: not available (set to 0)")

    fc = AD["feat_cols"]
    f  = f.replace([np.inf, -np.inf], np.nan)
    for col in fc:
        if col not in f.columns: f[col] = np.nan
    # Fill remaining NaN in cb_premium cols with 0 to avoid losing rows to dropna
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
        index=pd.DatetimeIndex(nd, name="target_date"),
    )
    return res[~res.index.duplicated(keep="last")]


# ─────────────────────────────────────────────────────────────────────────────
# Signal computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_signals(comp: pd.DataFrame) -> dict:
    """Compute all BTC signal arrays from a joined prediction+actual DataFrame."""
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

    # MA30 + regime
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

    # Clean-7d
    clean_7d = np.zeros(N, dtype=bool)
    for i in range(N):
        lo_i = max(0, i - 7)
        clean_7d[i] = not bool(np.any(d1[lo_i:i] | d2[lo_i:i]))

    # V-reversal (same as live app: dn_score > 0.8 & err_lo > 3%)
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

    # EMA10 of BTC close (used for SL2 re-entry gate)
    ema10_series = pd.Series(c_asof).ewm(span=10, adjust=False).mean().values

    # Full entry condition (TF2 + V-Gate)
    tf2_entry = u1 & (above_ma30 | clean_7d | v_recent)

    return dict(
        N=N, c_asof=c_asof,
        err_hi=err_hi, err_lo=err_lo, hi_brk=hi_brk, lo_brk=lo_brk,
        ehma3=ehma3, elma3=elma3, hb3=hb3, lb3=lb3,
        u1=u1, d1=d1, d2=d2, d3=d3,
        ma30=ma30, above_ma30=above_ma30, bull_regime=bull_regime,
        clean_7d=clean_7d, v_recent=v_recent, ema10=ema10_series,
        tf2_entry=tf2_entry,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Backtest engine
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(
    dates:      pd.DatetimeIndex,
    asset_px:   np.ndarray,   # execution prices for BTC / MSTR / MSTU
    btc_px:     np.ndarray,   # BTC close (always BTC for signal checks)
    sigs:       dict,
    variant:    str   = "B0", # "B0" = no SL; "SL0"–"SL4" = with SL
    sl_pct:     float = 0.10, # stop loss fraction (e.g. 0.10 = −10%)
    cap:        float = 100_000.0,
) -> dict:
    """
    Core backtest loop.
    Signal decisions (entry/exit) always use BTC-derived signals.
    P&L and stop-loss are computed on asset_px (BTC, MSTR, or MSTU prices).
    """
    N    = len(dates)
    u1   = sigs["u1"]; d2 = sigs["d2"]; d3 = sigs["d3"]
    abv  = sigs["above_ma30"]; bull = sigs["bull_regime"]
    c7d  = sigs["clean_7d"];   vr   = sigs["v_recent"]
    ema10 = sigs["ema10"]
    tf2_entry = sigs["tf2_entry"]

    nav   = cap; pos = "CASH"; qty = 0.0
    e_price_asset = None   # asset entry price (for SL calculation)
    e_price_btc   = None   # BTC price at entry (for SL1 gate check)
    e_date        = None
    e_nav         = None
    e_trigger     = None

    # Re-entry state (only matters when variant != "B0")
    from_sl          = False   # last exit was a stop loss
    sl_btc_exit_px   = None   # BTC price at the SL exit bar
    bars_since_sl    = 0       # bars elapsed since last SL exit

    trades    = []
    nav_arr   = np.full(N, np.nan)

    for i in range(N):
        si    = i - 1
        price = asset_px[i]   # today's asset close

        if i < WARMUP:
            nav_arr[i] = cap
            continue

        # ── COUNT bars since last stop loss ───────────────────────────────
        if pos == "CASH" and from_sl:
            bars_since_sl += 1

        if pos == "LONG":
            cur = qty * price

            # ── Stop-loss check (highest priority) ────────────────────────
            sl_fired = False
            if variant != "B0":
                sl_level = e_price_asset * (1.0 - sl_pct)
                if price <= sl_level:
                    sl_fired = True

            if sl_fired:
                nav = cur
                trades.append(dict(
                    entry_date    = e_date,
                    entry_price   = e_price_asset,
                    entry_nav     = e_nav,
                    entry_trigger = e_trigger,
                    exit_date     = dates[i],
                    exit_price    = price,
                    exit_nav      = nav,
                    pnl_pct       = (price / e_price_asset - 1) * 100,
                    pnl_abs       = nav - e_nav,
                    exit_signal   = f"SL-{sl_pct*100:.0f}%",
                    duration_days = (dates[i] - e_date).days,
                    was_sl        = True,
                ))
                pos = "CASH"; qty = 0.0
                from_sl       = True
                sl_btc_exit_px = btc_px[i]
                bars_since_sl  = 0

            elif si >= 0:
                # ── TF2 signal exit ──────────────────────────────────────
                should_exit = bool(d3[si] or (d2[si] and not bull[si]))
                xl = "D3" if d3[si] else "D2"
                if should_exit:
                    nav = cur
                    trades.append(dict(
                        entry_date    = e_date,
                        entry_price   = e_price_asset,
                        entry_nav     = e_nav,
                        entry_trigger = e_trigger,
                        exit_date     = dates[i],
                        exit_price    = price,
                        exit_nav      = nav,
                        pnl_pct       = (price / e_price_asset - 1) * 100,
                        pnl_abs       = nav - e_nav,
                        exit_signal   = xl,
                        duration_days = (dates[i] - e_date).days,
                        was_sl        = False,
                    ))
                    pos = "CASH"; qty = 0.0
                    from_sl = False
                else:
                    nav = cur
            else:
                nav = cur

        else:  # CASH
            if si >= 0 and tf2_entry[si]:
                # Decide if re-entry is allowed (only matters after SL)
                allow = True
                if from_sl and variant != "B0":
                    btc_now = btc_px[si]
                    if variant == "SL0":
                        allow = True   # no additional gate
                    elif variant == "SL1":
                        # BTC must recover ≥3% above the BTC price at SL exit
                        allow = (sl_btc_exit_px is not None and
                                 btc_now >= sl_btc_exit_px * 1.03)
                    elif variant == "SL2":
                        # BTC close must be above its own EMA10
                        allow = (btc_now > ema10[si])
                    elif variant == "SL3":
                        # 5-bar cooldown
                        allow = (bars_since_sl >= 5)
                    elif variant == "SL4":
                        # Regime-adaptive: bull → immediate; bear → 10-bar cooldown
                        allow = bool(bull[si]) or (bars_since_sl >= 10)

                if allow:
                    if not np.isfinite(price) or price <= 0:
                        nav_arr[i] = nav
                        continue
                    qty = nav / price
                    e_price_asset = price
                    e_price_btc   = btc_px[i]
                    e_date        = dates[i]
                    e_nav         = nav
                    pos           = "LONG"
                    from_sl       = False
                    bars_since_sl = 0
                    # Build trigger label
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
        cap * asset_px[WARMUP:] / asset_px[WARMUP], index=dates[WARMUP:]
    )
    return dict(trades=trades, nav=nav_s, bh=bh_s,
                open_pos=(pos == "LONG"),
                open_entry=dict(price=e_price_asset, date=e_date) if pos == "LONG" else None)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def metrics(res: dict, cap: float = INITIAL_CAPITAL) -> dict:
    nav = res["nav"]; bh = res["bh"]; trades = res["trades"]
    fn  = float(nav.iloc[-1]); fb = float(bh.iloc[-1])
    n_y = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = ((fn / cap) ** (1 / n_y) - 1) * 100 if n_y > 0 else 0

    dr  = nav.pct_change().fillna(0)
    rfd = (1.045) ** (1 / 252) - 1
    exc = dr - rfd
    sharpe  = float(exc.mean() / exc.std() * np.sqrt(252)) if exc.std() > 0 else 0
    rm      = nav.cummax(); dd = (nav - rm) / rm * 100
    max_dd  = float(dd.min())
    rm_bh   = bh.cummax(); dd_bh = (bh - rm_bh) / rm_bh * 100
    bh_maxdd = float(dd_bh.min())

    wins    = [t for t in trades if t["pnl_pct"] > 0]
    losses  = [t for t in trades if t["pnl_pct"] <= 0]
    sl_exits = [t for t in trades if t.get("was_sl")]
    win_rate = 100 * len(wins) / len(trades) if trades else 0

    gp = sum(t["pnl_abs"] for t in wins)  if wins   else 0
    gl = abs(sum(t["pnl_abs"] for t in losses)) if losses else 1e-9
    pf = gp / gl

    avg_win  = float(np.mean([t["pnl_pct"] for t in wins]))   if wins   else 0
    avg_loss = float(np.mean([t["pnl_pct"] for t in losses])) if losses else 0

    days_in  = sum(t["duration_days"] for t in trades)
    tot_days = max(1, (nav.index[-1] - nav.index[0]).days)

    return dict(
        final_nav  = fn,    bh_nav     = fb,
        strat_ret  = (fn / cap - 1) * 100,
        bh_ret     = (fb / cap - 1) * 100,
        alpha_abs  = fn - fb,
        cagr       = cagr,
        sharpe     = sharpe,
        max_dd     = max_dd,   bh_maxdd = bh_maxdd,
        n_trades   = len(trades),
        n_sl       = len(sl_exits),
        n_wins     = len(wins),   n_losses = len(losses),
        win_rate   = win_rate,
        avg_win    = avg_win,  avg_loss = avg_loss,
        profit_factor = pf,
        time_in    = 100 * days_in / tot_days,
        trades     = trades,
        open_pos   = res.get("open_pos"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Period-level runner: builds signals once, runs all variants
# ─────────────────────────────────────────────────────────────────────────────
def _prep_comp(df_raw, preds, start_iso, end_iso):
    """Join preds with actuals for [start, end]."""
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


def run_period(
    period_name: str, start_iso: str, end_iso: str, fetch_start: str,
    df_raw_cache: dict,
) -> dict:
    """Run all variants across BTC / MSTR / MSTU for one period."""
    fetch_end = (pd.Timestamp(end_iso) + pd.Timedelta(days=3)).strftime("%Y-%m-%d")

    # ── Fetch + predictions (cached by fetch_start) ───────────────────────
    cache_key = (fetch_start, fetch_end)
    if cache_key not in df_raw_cache:
        df_raw   = fetch_data(fetch_start, fetch_end)
        preds    = build_ct_preds(df_raw, fetch_start=fetch_start, fetch_end=fetch_end)
        mstr_px_raw = fetch_asset_prices("MSTR", fetch_start, fetch_end)
        mstu_px_raw = fetch_asset_prices("MSTU", fetch_start, fetch_end)
        df_raw_cache[cache_key] = dict(
            df_raw=df_raw, preds=preds,
            mstr_px=mstr_px_raw, mstu_px=mstu_px_raw,
        )
    cached = df_raw_cache[cache_key]
    df_raw      = cached["df_raw"]
    preds       = cached["preds"]
    mstr_px_raw = cached["mstr_px"]
    mstu_px_raw = cached["mstu_px"]

    # ── Build aligned comparison DataFrame ───────────────────────────────
    comp = _prep_comp(df_raw, preds, start_iso, end_iso)
    if len(comp) < WARMUP + 3:
        print(f"  [SKIP] {period_name}: insufficient bars ({len(comp)})")
        return {}
    dates = pd.DatetimeIndex(comp["target_date"])

    # ── Signal arrays (computed once) ─────────────────────────────────────
    sigs = compute_signals(comp)
    N    = sigs["N"]

    # ── Asset price arrays aligned to dates ───────────────────────────────
    btc_px = comp["actual_close"].values.astype(float)

    def align_asset(px_series: pd.Series) -> np.ndarray:
        """Align an asset price Series to dates; returns array or None if unavailable."""
        if px_series.empty:
            return None
        aligned = px_series.reindex(dates).ffill().bfill().values.astype(float)
        valid   = np.sum(np.isfinite(aligned) & (aligned > 0))
        if valid < WARMUP + 3:
            return None
        return aligned

    asset_arrays = {
        "BTC":  btc_px,
        "MSTR": align_asset(mstr_px_raw),
        "MSTU": align_asset(mstu_px_raw),
    }

    period_results = {}
    variants = list(REENTRY_LABELS.keys())

    for asset_name, asset_px in asset_arrays.items():
        if asset_px is None:
            print(f"  [SKIP] {asset_name}: no data for {period_name}")
            continue

        # Check earliest valid MSTU data
        if asset_name == "MSTU":
            first_valid = np.argmax(np.isfinite(asset_px) & (asset_px > 0))
            if first_valid > WARMUP + 10:
                print(f"  [SKIP] MSTU: first valid bar too late "
                      f"(bar {first_valid}/{N})")
                continue

        sl_pct = ASSET_CONFIG[asset_name]["sl_pct"]
        period_results[asset_name] = {}

        for variant in variants:
            try:
                res = run_backtest(dates, asset_px, btc_px, sigs,
                                   variant=variant, sl_pct=sl_pct,
                                   cap=INITIAL_CAPITAL)
                m = metrics(res, INITIAL_CAPITAL)
                period_results[asset_name][variant] = m
            except Exception as e:
                print(f"  [ERR] {asset_name}/{variant}: {e}")

    return period_results


# ─────────────────────────────────────────────────────────────────────────────
# Printing helpers
# ─────────────────────────────────────────────────────────────────────────────
def print_asset_period_table(period_name: str, asset: str, results: dict):
    """Print comparison table for one asset × one period."""
    variants = list(REENTRY_LABELS.keys())
    W = 16
    header_cols = ["Variant", "Ret%", "B&H%", "Alpha$k", "MaxDD%",
                   "Sharpe", "Trades", "#SL", "WinRate", "TiM%"]
    W2 = 9
    sep = "─" * (22 + W2 * (len(header_cols) - 1))

    asset_cfg = ASSET_CONFIG[asset]
    sl_pct    = asset_cfg["sl_pct"]

    print(f"\n  {asset} ({asset_cfg['name']})  ── SL level: −{sl_pct*100:.0f}%")
    print(f"  {sep}")
    print(f"  {'Variant':<22}" + "".join(f"{h:>{W2}}" for h in header_cols[1:]))
    print(f"  {sep}")

    for v in variants:
        if v not in results:
            print(f"  {REENTRY_LABELS[v]:<22}  — data unavailable")
            continue
        m = results[v]
        tag = "★" if v == "B0" else " "
        label = f"{tag} [{v}]"
        line = (
            f"  {label:<22}"
            f"{m['strat_ret']:>+{W2}.1f}%"[:-1] + "%"
            if False else
            f"  {label:<22}"
            f"{m['strat_ret']:>{W2-1}.1f}%"
            f"{m['bh_ret']:>{W2-1}.1f}%"
            f"${m['alpha_abs']/1e3:>{W2-2}.1f}k"
            f"{m['max_dd']:>{W2-1}.1f}%"
            f"{m['sharpe']:>{W2}.2f}"
            f"{m['n_trades']:>{W2}}"
            f"{m['n_sl']:>{W2}}"
            f"{m['win_rate']:>{W2-1}.0f}%"
            f"{m['time_in']:>{W2-1}.0f}%"
        )
        print(line)

    print(f"  {sep}")


def print_full_table(period_name: str, period_results: dict):
    """Print a compact multi-asset table for one period."""
    W  = 72
    print(f"\n{'═' * W}")
    print(f"  PERIOD: {period_name}")
    print(f"{'═' * W}")

    for asset in ["BTC", "MSTR", "MSTU"]:
        if asset not in period_results:
            print(f"  {asset}: no data")
            continue
        print_asset_period_table(period_name, asset, period_results[asset])


def print_cross_period_summary(all_results: dict):
    """
    Cross-period view: for each asset, show how each re-entry variant
    performs across all periods.
    """
    print(f"\n{'═' * 90}")
    print("  CROSS-PERIOD SUMMARY — Strategy Return % by Variant × Period")
    print(f"{'═' * 90}")

    periods = list(all_results.keys())
    variants = list(REENTRY_LABELS.keys())

    for asset in ["BTC", "MSTR", "MSTU"]:
        print(f"\n  {'─' * 85}")
        print(f"  Asset: {asset} ({ASSET_CONFIG[asset]['name']})"
              f"  SL: −{ASSET_CONFIG[asset]['sl_pct']*100:.0f}%")
        print(f"  {'─' * 85}")

        # Print header
        P_W = 22
        V_W = 12
        pnames_short = [p.split("(")[0].strip() for p in periods]
        print(f"  {'Variant':<{P_W}}" + "".join(f"{p:>{V_W}}" for p in pnames_short))
        print(f"  {'─' * (P_W + V_W * len(periods))}")

        for v in variants:
            row_vals = []
            for pname in periods:
                pr = all_results.get(pname, {})
                ar = pr.get(asset, {})
                m  = ar.get(v)
                if m is None:
                    row_vals.append("   n/a")
                else:
                    row_vals.append(f"{m['strat_ret']:>+.1f}%")

            label = REENTRY_LABELS[v][:P_W]
            print(f"  {label:<{P_W}}" + "".join(f"{v:>{V_W}}" for v in row_vals))

        # Show B&H row for reference
        bh_vals = []
        for pname in periods:
            pr = all_results.get(pname, {})
            ar = pr.get(asset, {})
            m  = ar.get("B0")
            if m is None:
                bh_vals.append("   n/a")
            else:
                bh_vals.append(f"{m['bh_ret']:>+.1f}%")
        print(f"  {'─' * (P_W + V_W * len(periods))}")
        print(f"  {'Buy & Hold':<{P_W}}" + "".join(f"{v:>{V_W}}" for v in bh_vals))

    print(f"\n{'═' * 90}")


def print_recommendation(all_results: dict):
    """Identify and print the recommended re-entry criterion per asset."""
    print(f"\n{'═' * 90}")
    print("  RECOMMENDATION SUMMARY")
    print(f"{'═' * 90}")

    periods     = list(all_results.keys())
    sl_variants = [v for v in REENTRY_LABELS if v != "B0"]

    for asset in ["BTC", "MSTR", "MSTU"]:
        print(f"\n  {'─' * 60}")
        print(f"  {asset} ({ASSET_CONFIG[asset]['name']})")
        print(f"  {'─' * 60}")

        # Score each variant: for each period, is it better than SL0?
        # Higher NAV than SL0 = positive point; also check max_dd vs SL0
        scores = {v: [] for v in sl_variants}
        b0_navs = {}

        for pname in periods:
            pr = all_results.get(pname, {}).get(asset, {})
            sl0 = pr.get("SL0")
            b0  = pr.get("B0")
            if sl0 is None:
                continue
            b0_navs[pname] = b0["final_nav"] if b0 else None

            for v in sl_variants:
                m = pr.get(v)
                if m is None:
                    continue
                # +1 for each period where variant beats SL0 on NAV
                # +0.5 for each period where max_dd is better than SL0
                beats_nav = m["final_nav"] > sl0["final_nav"]
                beats_dd  = m["max_dd"]    > sl0["max_dd"]   # less negative = better
                scores[v].append(beats_nav + 0.5 * beats_dd)

        # Print score table
        print(f"  {'Variant':<26}  {'Avg score':>10}  {'Win rate vs SL0':>16}  Notes")
        for v in sl_variants:
            sc = scores[v]
            if not sc:
                print(f"  {REENTRY_LABELS[v]:<26}  {'n/a':>10}")
                continue
            avg_sc  = float(np.mean(sc))
            win_pct = 100 * sum(s >= 0.5 for s in sc) / len(sc)
            marker  = "◄ BEST" if avg_sc == max(np.mean(s) for s in scores.values() if s) else ""
            print(f"  {REENTRY_LABELS[v]:<26}  {avg_sc:>10.2f}  {win_pct:>14.0f}%  {marker}")

        # Print verdict
        best_v = max(sl_variants, key=lambda v: float(np.mean(scores[v])) if scores[v] else -999)
        b0_text = ""
        baseline_variants_with_data = [v for v in sl_variants if scores[v]]
        if not baseline_variants_with_data:
            print("  [No data available for recommendation]")
            continue
        best_score = float(np.mean(scores[best_v]))

        print(f"\n  Recommendation for {asset}:")
        if best_score <= 0.6:
            print(f"    ⚠ No re-entry criterion consistently outperforms SL0 "
                  f"(best avg score: {best_score:.2f}/1.5). Standard re-entry "
                  f"(SL0) is the simplest and acceptable default.")
        else:
            print(f"    ✓ [{best_v}] {REENTRY_LABELS[best_v]}")
            print(f"      Avg score vs SL0: {best_score:.2f}/1.5 across periods.")

    print(f"\n{'═' * 90}")


def print_trade_log_after_sl(period_results: dict, period_name: str, asset: str):
    """Show how different variants handle post-SL re-entries."""
    if asset not in period_results:
        return
    res = period_results[asset]
    b0  = res.get("B0"); sl0 = res.get("SL0")
    if not b0 or not sl0:
        return

    sl_exits = [t for t in sl0["trades"] if t.get("was_sl")]
    if not sl_exits:
        print(f"\n  No stop-loss exits in {period_name} for {asset} — stop level "
              f"not triggered in this period.")
        return

    print(f"\n  Post-SL Re-Entry Comparison — {asset}, {period_name}")
    print(f"  SL level: −{ASSET_CONFIG[asset]['sl_pct']*100:.0f}%  "
          f"| {len(sl_exits)} stop loss exits in SL0")
    print(f"  {'SL Exit':>12}  {'SL Exit $':>10}  {'SL PnL%':>8}  "
          + "  ".join(f"{v:>8}" for v in ["SL0","SL1","SL2","SL3","SL4"]))
    print(f"  {'─' * 75}")

    for sl_t in sl_exits:
        exit_dt = pd.Timestamp(sl_t["exit_date"])
        row = f"  {exit_dt.strftime('%b %d %Y'):>12}  ${sl_t['exit_price']:>9,.1f}  {sl_t['pnl_pct']:>+7.1f}%"
        for v in ["SL0", "SL1", "SL2", "SL3", "SL4"]:
            vres = res.get(v)
            if vres is None:
                row += f"  {'n/a':>8}"; continue
            # Find next trade after this SL exit
            nxt = next(
                (t for t in vres["trades"]
                 if pd.Timestamp(t["entry_date"]) >= exit_dt),
                None
            )
            if nxt:
                lag_days = (pd.Timestamp(nxt["entry_date"]) - exit_dt).days
                row += f"  {nxt['pnl_pct']:>+6.1f}%(+{lag_days}d)"
            else:
                row += f"  {'no re-entry':>8}"
        print(row)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═" * 90)
    print("  STOP-LOSS RE-ENTRY CRITERIA ANALYSIS")
    print("  BTC | MSTR | MSTU  ×  TF2+V-Gate with Hard Stop + Smart Re-Entry")
    print("═" * 90)
    print("\nStop-loss levels:")
    for asset, cfg in ASSET_CONFIG.items():
        print(f"  {asset:<6}: −{cfg['sl_pct']*100:.0f}%  ({cfg['name']})")
    print("\nRe-entry variants:")
    for v, lbl in REENTRY_LABELS.items():
        print(f"  {v}: {lbl}")
    print()

    # Cache raw data to avoid redundant fetches across periods with same window
    df_raw_cache: dict = {}

    all_results: dict = {}
    for pname, (start, end, fs) in PERIODS.items():
        print(f"\n{'─' * 70}")
        print(f"  PERIOD: {pname}  ({start} → {end})")
        print(f"{'─' * 70}")
        pr = run_period(pname, start, end, fs, df_raw_cache)
        all_results[pname] = pr
        print_full_table(pname, pr)

    # ── Cross-period summary ───────────────────────────────────────────────
    print_cross_period_summary(all_results)

    # ── Detailed post-SL trade analysis for key periods ───────────────────
    for pname in ["Bull (Sep24→Sep25)", "Bear (Jun25→May26)", "OOS (Sep25→May26)"]:
        if pname not in all_results:
            continue
        for asset in ["BTC", "MSTR", "MSTU"]:
            if asset in all_results[pname]:
                print_trade_log_after_sl(all_results[pname], pname, asset)

    # ── Recommendations ───────────────────────────────────────────────────
    print_recommendation(all_results)

    print("\n✓ Analysis complete.")
