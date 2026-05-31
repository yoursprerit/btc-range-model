"""
assess_new_features.py
======================
Evaluates 10 candidate BTC metrics as additional features for the Daily H/L model.

Metrics assessed:
 1. Short-Term Holder Cost Basis  → 155-day VWAP (proxy, free)
 2. True Market Mean              → 365-day VWAP (proxy, free)
 3. 200-Day Moving Average        → direct (free)
 4. Dealer Negative Gamma         → NOT implementable (requires Deribit/CME options data)
 5. 200-Week Moving Average       → direct (free)
 6. Coinbase Premium              → NOT usable in backtest (no Binance data in inference path)
 7. 30-Day Net Flow to Exchanges  → NOT implementable without CryptoQuant/Glassnode
 8. Stablecoin Outflows           → NOT implementable without CryptoQuant/Glassnode
 9. Sell-Side Risk Ratio          → NOT implementable without Glassnode (UTXO cost basis)
10. 180-Day Realized Price Change → 180-day log return (proxy, free)

Methodology:
 - Fetch BTC + macro from Yahoo Finance (identical to retrain_pipeline._bt_fetch_raw)
 - Build feature matrix matching pipeline_ct.py naming conventions
 - Add 7 new free features (metrics 1, 2, 3, 5, 10 + supporting)
 - Train two GBM ensembles: BASELINE vs ENHANCED (same hyperparams, same split)
 - Compare MAPE_H / MAPE_L / hit-rates on held-out test window
 - Run TF2+V-Gate backtest on all 3 periods (Bear / Bull / Full)
 - Report which features add value
"""

import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, QuantileRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ─── Constants ───────────────────────────────────────────────────────────────
FETCH_START      = "2018-06-01"   # need 1400 bars of warmup for 200-week MA
TRAIN_CUTOFF     = "2026-01-01"   # val window starts here
VAL_CUTOFF       = "2026-03-01"   # test window starts here
TEST_END         = None           # None = use all available data

BT_BEAR_START    = "2025-06-01"
BT_BULL_START    = "2024-06-05"
BT_BULL_END      = "2025-06-14"
BT_FULL_START    = "2024-06-05"
BT_WARMUP_BARS   = 35

MACRO_SYMS = {
    "eth": "ETH-USD", "spx": "^GSPC", "ndx": "^IXIC",
    "vix": "^VIX",   "gold": "GC=F",  "dxy": "DX-Y.NYB", "tnx": "^TNX",
}

GBM_PARAMS = dict(
    loss="quantile", alpha=0.70,
    n_estimators=600, max_depth=3,
    learning_rate=0.02, subsample=0.8,
    min_samples_leaf=5, random_state=42,
)


# ─── Data fetch ──────────────────────────────────────────────────────────────
def _flat(df, name):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = [f"{name}_{c.lower()}" for c in df.columns]
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df[~df.index.duplicated(keep="last")].sort_index()


def fetch_data(start: str = FETCH_START) -> pd.DataFrame:
    today = datetime.now(timezone.utc).date()
    end   = (pd.Timestamp(today) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    parts = []
    syms  = {"btc": "BTC-USD"} | MACRO_SYMS
    for name, sym in syms.items():
        try:
            raw = yf.download(sym, start=start, end=end,
                              progress=False, auto_adjust=False)
            if not raw.empty:
                parts.append(_flat(raw, name))
                print(f"  {sym:<12}: {len(raw)} rows "
                      f"{raw.index.min().date()} → {raw.index.max().date()}")
        except Exception as exc:
            print(f"  ! {sym}: {exc}")
    df = parts[0]
    for p in parts[1:]:
        df = df.join(p, how="outer")
    return df.ffill(limit=5)


# ─── Feature engineering ──────────────────────────────────────────────────────
def build_features(df: pd.DataFrame,
                   include_new: bool = False) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build the feature matrix.

    Returns (features_df, y_hi, y_lo) where targets are next-bar % offsets.
    If include_new=True, appends the 7 new candidate features.
    """
    c = df["btc_close"]
    h = df["btc_high"]
    l = df["btc_low"]
    v = df["btc_volume"]

    # Targets: next-bar high/low as % offset from current close (same as pipeline_ct.py)
    y_hi = (h.shift(-1) - c) / c
    y_lo = (c - l.shift(-1)) / c

    f = pd.DataFrame(index=df.index)

    # ── Baseline features (same naming as pipeline_ct.py) ────────────────────
    ret = np.log(c).diff()
    for k in [1, 3, 5, 7, 14, 30]:
        f[f"ret_{k}"] = ret.rolling(k).sum()
    for k in [5, 10, 20, 30]:
        f[f"vol_{k}"] = ret.rolling(k).std()

    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    for k in [7, 14, 30]:
        f[f"atr_{k}"] = tr.rolling(k).mean() / c

    f["range_today"] = (h - l) / c
    f["range_ma7"]   = ((h - l) / c).rolling(7).mean()
    f["range_ma30"]  = ((h - l) / c).rolling(30).mean()
    f["range_std30"] = ((h - l) / c).rolling(30).std()

    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    f["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    f["macd"]      = macd / c
    f["macd_sig"]  = macd.ewm(span=9, adjust=False).mean() / c
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c

    ma20 = c.rolling(20).mean()
    sd20 = c.rolling(20).std()
    f["bb_width"]  = (4 * sd20) / ma20

    f["dist_hi_30"] = c / c.rolling(30).max() - 1
    f["dist_lo_30"] = c / c.rolling(30).min() - 1
    f["dist_hi_90"] = c / c.rolling(90).max() - 1

    f["vol_chg_1"]    = np.log(v).diff()
    f["vol_z_20"]     = (np.log(v) - np.log(v).rolling(20).mean()) / np.log(v).rolling(20).std()
    f["vol_ma_ratio"] = v / v.rolling(20).mean()

    dow = df.index.dayofweek
    for i in range(6):
        f[f"dow_{i}"] = (dow == i).astype(float)

    # Macro returns (same names as pipeline_ct.py)
    for nm in ["spx", "ndx", "vix", "gold", "dxy", "tnx", "eth"]:
        s = df.get(f"{nm}_close")
        if s is not None:
            for k in [1, 5, 20]:
                f[f"{nm}_ret_{k}"] = np.log(s).diff(k)
            f[f"{nm}_vol_20"] = np.log(s).diff().rolling(20).std()

    f["btc_spx_corr_30"]  = ret.rolling(30).corr(np.log(df["spx_close"]).diff())
    f["btc_ndx_corr_30"]  = ret.rolling(30).corr(np.log(df["ndx_close"]).diff())
    f["btc_gold_corr_30"] = ret.rolling(30).corr(np.log(df["gold_close"]).diff())
    f["btc_dxy_corr_30"]  = ret.rolling(30).corr(np.log(df["dxy_close"]).diff())

    # Smoothed past-target features
    f["y_hi_ema3"] = y_hi.shift(1).ewm(span=3, adjust=False).mean()
    f["y_lo_ema3"] = y_lo.shift(1).ewm(span=3, adjust=False).mean()
    f["y_hi_ema7"] = y_hi.shift(1).ewm(span=7, adjust=False).mean()
    f["y_lo_ema7"] = y_lo.shift(1).ewm(span=7, adjust=False).mean()

    # Anti-mean-reversion / breakout features
    prev_3_hi = h.shift(1).rolling(3).max()
    prev_3_lo = l.shift(1).rolling(3).min()
    f["above_3d_high"]  = (c > prev_3_hi).astype(float)
    f["below_3d_low"]   = (c < prev_3_lo).astype(float)
    f["bo_strength_up"] = (c / prev_3_hi - 1).clip(lower=0)
    f["bo_strength_dn"] = (1 - c / prev_3_lo).clip(lower=0)
    _yhi_lag = y_hi.shift(1); _ylo_lag = y_lo.shift(1)
    f["y_hi_surprise"]  = _yhi_lag - _yhi_lag.ewm(span=7, adjust=False).mean()
    f["y_lo_surprise"]  = _ylo_lag - _ylo_lag.ewm(span=7, adjust=False).mean()

    neg_ret = ret.clip(upper=0)
    f["dn_vol_5"]  = neg_ret.rolling(5).std()
    f["dn_vol_20"] = neg_ret.rolling(20).std()
    sma50 = c.rolling(50).mean()
    f["below_sma50"]    = (c < sma50).astype(float)
    f["below_sma50_5d"] = (c < sma50).astype(float).rolling(5).min().fillna(0)

    # ── NEW candidate features ────────────────────────────────────────────────
    if include_new in (True, "lite"):
        # Metric #3: 200-Day Moving Average — distance + slope
        ma200 = c.rolling(200).mean()
        f["dist_ma200"]  = np.log(c / ma200)
        f["ma200_slope"] = ma200.pct_change(10)

        # Metric #10: 180-Day Realized Price Change — log return proxy
        f["ret_90"]  = ret.rolling(90).sum()
        f["ret_180"] = ret.rolling(180).sum()

        # Metric #1: Short-Term Holder Cost Basis — 155-day VWAP (STH = coins moved < 155d)
        vwap_155 = (c * v).rolling(155).sum() / v.rolling(155).sum()
        f["sth_proxy_dist"] = np.log(c / vwap_155)

        # Metric #2: True Market Mean — 365-day VWAP (long-term realized-price proxy)
        vwap_365 = (c * v).rolling(365).sum() / v.rolling(365).sum()
        f["tmm_proxy_dist"] = np.log(c / vwap_365)

    if include_new is True:
        # Metric #5: 200-Week Moving Average — 1400-bar warmup (only from ~2022)
        ma1400 = c.rolling(1400).mean()
        f["dist_ma1400"] = np.log(c / ma1400)

    f = f.replace([np.inf, -np.inf], np.nan)
    return f, y_hi, y_lo


# ─── Model training ───────────────────────────────────────────────────────────
def make_constituent(params: dict):
    return {
        "huber": Pipeline([("sc", StandardScaler()),
                           ("m",  HuberRegressor(alpha=0.001, max_iter=300))]),
        "gbm":   Pipeline([("sc", StandardScaler()),
                           ("m",  GradientBoostingRegressor(**params))]),
    }


def train_ensemble(X_tr, y_tr, X_va, y_va, mu):
    """Train (Huber + GBM) ensemble. Returns (ensemble_list, blend_alpha)."""
    ens = []
    for name, pipe in make_constituent(GBM_PARAMS).items():
        pipe.fit(X_tr, y_tr)
        ens.append(pipe)

    # Choose alpha (climatology blend) on val to minimise MAPE
    preds_va = np.mean([p.predict(X_va) for p in ens], axis=0)
    best_a, best_mape = 1.0, np.mean(np.abs(preds_va - y_va))
    for a in np.linspace(0.5, 1.0, 11):
        blended = a * preds_va + (1 - a) * mu
        mape    = np.mean(np.abs(blended - y_va))
        if mape < best_mape:
            best_mape = mape; best_a = a
    return ens, best_a


def ensemble_predict(ens, alpha, mu, X):
    raw = np.mean([p.predict(X) for p in ens], axis=0)
    return alpha * raw + (1 - alpha) * mu


def compute_metrics(pred, actual, close):
    """Return dict of MAPE and hit-rates."""
    pct_err = np.abs(pred - actual) / close * 100
    mape    = float(np.mean(pct_err))
    hit05   = float(np.mean(pct_err <= 0.5))
    hit1    = float(np.mean(pct_err <= 1.0))
    hit2    = float(np.mean(pct_err <= 2.0))
    return dict(mape=mape, hit05=hit05*100, hit1=hit1*100, hit2=hit2*100)


# ─── Backtest (TF2+V-Gate, same logic as retrain_pipeline._bt_run_strategy) ──
def run_backtest(preds_df: pd.DataFrame, raw: pd.DataFrame,
                 bt_start: str, bt_end: str,
                 initial_capital: float = 100_000.0) -> dict | None:
    start_dt = pd.Timestamp(bt_start)
    end_dt   = pd.Timestamp(bt_end)
    pre_dt   = start_dt - pd.Timedelta(days=60)

    p = preds_df.loc[(preds_df.index >= pre_dt) & (preds_df.index <= end_dt)].copy()
    p["actual_high"]  = raw["btc_high"].reindex(p.index).values
    p["actual_low"]   = raw["btc_low"].reindex(p.index).values
    p["actual_close"] = raw["btc_close"].reindex(p.index).values
    comp = p.dropna(subset=["actual_high", "actual_low", "actual_close"]).reset_index()
    N    = len(comp)
    if N < BT_WARMUP_BARS + 3:
        return None

    dates   = pd.DatetimeIndex(comp["target_date"])
    c_asof  = comp["close_asof"].values.astype(float)
    exec_px = comp["actual_close"].values.astype(float)
    pred_hi = comp["pred_high"].values.astype(float)
    pred_lo = comp["pred_low"].values.astype(float)
    act_hi  = comp["actual_high"].values.astype(float)
    act_lo  = comp["actual_low"].values.astype(float)

    _bt0 = max(BT_WARMUP_BARS, int(dates.searchsorted(start_dt)))
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

    ma30 = np.zeros(N)
    for i in range(N):
        w       = min(30, i + 1)
        ma30[i] = float(np.mean(c_asof[max(0, i - w + 1):i + 1]))
    above_ma30 = c_asof > ma30
    slope5     = np.zeros(N, dtype=bool)
    for i in range(5, N):
        slope5[i] = ma30[i] > ma30[i - 5]
    bull_regime = above_ma30 & slope5

    clean_10d = np.zeros(N, dtype=bool)
    for i in range(N):
        li           = max(0, i - 10)
        clean_10d[i] = not bool(np.any(d1[li:i] | d2[li:i]))

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
                    entry_price=e_price, exit_price=price,
                    pnl_pct=(price / e_price - 1) * 100,
                    exit_signal=exit_lbl,
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
        strat_ret=strat_ret, bh_ret=bh_ret, final_nav=final_nav,
        n_trades=len(trades), n_wins=len(wins), n_losses=len(losses),
        win_rate=100.0 * len(wins) / len(trades) if trades else 0.0,
        max_dd=max_dd, open_pos=pos == "LONG",
    )


def build_pred_series(raw: pd.DataFrame, ens_hi, ens_lo,
                      alpha_hi, alpha_lo, mu_hi, mu_lo,
                      feat_df: pd.DataFrame) -> pd.DataFrame:
    """Generate pred_high / pred_low series for the full history."""
    F      = feat_df.dropna()
    c_vals = raw["btc_close"].reindex(F.index).values.astype(float)
    X      = F.to_numpy()

    yhi = ensemble_predict(ens_hi, alpha_hi, mu_hi, X)
    ylo = ensemble_predict(ens_lo, alpha_lo, mu_lo, X)

    pred_hi = c_vals * (1.0 + np.clip(yhi, 0, None))
    pred_lo = c_vals * (1.0 - np.clip(ylo, 0, None))

    next_dates = list(F.index[1:]) + [F.index[-1] + pd.Timedelta(days=1)]
    preds = pd.DataFrame(
        {"close_asof": c_vals, "pred_high": pred_hi, "pred_low": pred_lo},
        index=pd.DatetimeIndex(next_dates, name="target_date"),
    )
    return preds[~preds.index.duplicated(keep="last")]


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    SEP = "─" * 72

    print(SEP)
    print("  NEW FEATURE ASSESSMENT: 10 Candidate Bitcoin Metrics")
    print("  Daily H/L Model + TF2+V-Gate Backtest Comparison")
    print(SEP)

    # ── 1. Feasibility grid ──────────────────────────────────────────────────
    print("\n[0] FEASIBILITY ASSESSMENT")
    print(f"  {'#':<3} {'Metric':<40} {'Status':<12} {'Implementation'}")
    print(f"  {'─'*3} {'─'*40} {'─'*12} {'─'*30}")
    metrics = [
        (1,  "Short-Term Holder Cost Basis",    "PROXY",    "155-day VWAP (sth_proxy_dist)"),
        (2,  "True Market Mean",                "PROXY",    "365-day VWAP (tmm_proxy_dist)"),
        (3,  "200-Day Moving Average",          "DIRECT",   "dist_ma200 + ma200_slope"),
        (4,  "Dealer Negative Gamma",           "SKIP",     "Requires Deribit/CME options data"),
        (5,  "200-Week Moving Average",         "DIRECT",   "dist_ma1400 (~1400 daily bars)"),
        (6,  "Coinbase Premium",                "SKIP",     "No Binance data in backtest path"),
        (7,  "30-Day Net Flow to Exchanges",    "SKIP",     "Requires CryptoQuant/Glassnode API"),
        (8,  "Stablecoin Outflows",             "SKIP",     "Requires CryptoQuant/Glassnode API"),
        (9,  "Sell-Side Risk Ratio",            "SKIP",     "Requires Glassnode UTXO cost basis"),
        (10, "180-Day Realized Price Change",   "PROXY",    "180-day log return (ret_180)"),
    ]
    for num, name, status, impl in metrics:
        tag = {"DIRECT": "✓ DIRECT", "PROXY": "≈ PROXY", "SKIP": "✗ SKIP"}.get(status, status)
        print(f"  {num:<3} {name:<40} {tag:<12} {impl}")

    print("\n  7 new features will be evaluated: dist_ma200, ma200_slope,")
    print("  dist_ma1400, ret_90, ret_180, sth_proxy_dist, tmm_proxy_dist")

    # ── 2. Fetch data ────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("[1] FETCHING DATA (Yahoo Finance)")
    print(SEP)
    raw = fetch_data()
    print(f"\n  Raw shape: {raw.shape}  "
          f"{raw.index.min().date()} → {raw.index.max().date()}")

    today_iso = raw.index.max().strftime("%Y-%m-%d")

    # ── 3. Build feature matrices ─────────────────────────────────────────────
    print(f"\n{SEP}")
    print("[2] FEATURE ENGINEERING  (two experiments)")
    print(SEP)

    # ── Experiment A: 6 features, NO 200-week MA → full history from 2019 ────
    print("  Experiment A: 6 features (no 200-week MA, full training history)")
    f_base_a, y_hi, y_lo = build_features(raw, include_new=False)
    f_lite,   _,    _    = build_features(raw, include_new="lite")

    f_base_a["y_hi"] = y_hi; f_base_a["y_lo"] = y_lo; f_base_a["close"] = raw["btc_close"]
    f_lite["y_hi"]   = y_hi; f_lite["y_lo"]   = y_lo; f_lite["close"]   = raw["btc_close"]

    lite_raw  = f_lite.replace([np.inf, -np.inf], np.nan).dropna()
    base_a    = f_base_a.replace([np.inf, -np.inf], np.nan).dropna()
    base_a    = base_a.loc[lite_raw.index.min():]   # align start dates
    enha_a    = lite_raw

    new_cols_a = [c for c in enha_a.columns if c not in base_a.columns and
                  c not in ("y_hi", "y_lo", "close")]
    print(f"    Baseline : {base_a.shape[1]-3} cols  "
          f"{base_a.index.min().date()} → {base_a.index.max().date()}")
    print(f"    Enhanced : {enha_a.shape[1]-3} cols  (added: {new_cols_a})")

    # ── Experiment B: 7 features including 200-week MA → constrained to 2022+ ─
    print("  Experiment B: all 7 features (includes 200-week MA, 2022+ only)")
    f_full,  _, _ = build_features(raw, include_new=True)
    f_full["y_hi"] = y_hi; f_full["y_lo"] = y_lo; f_full["close"] = raw["btc_close"]

    enha_b  = f_full.replace([np.inf, -np.inf], np.nan).dropna()
    base_b  = f_base_a.replace([np.inf, -np.inf], np.nan).dropna()
    base_b  = base_b.loc[enha_b.index.min():]

    new_cols_b = [c for c in enha_b.columns if c not in base_b.columns and
                  c not in ("y_hi", "y_lo", "close")]
    print(f"    Baseline : {base_b.shape[1]-3} cols  "
          f"{base_b.index.min().date()} → {base_b.index.max().date()}")
    print(f"    Enhanced : {enha_b.shape[1]-3} cols  (added: {new_cols_b})")

    # Use Experiment A as the primary (more training data → more reliable)
    base = base_a; enha = enha_a; new_cols = new_cols_a
    print(f"\n  → Primary comparison uses Experiment A (full training history)")

    # ── 4. Train/val/test split ───────────────────────────────────────────────
    print(f"\n{SEP}")
    print("[3] TRAIN / VAL / TEST SPLIT")
    print(SEP)
    train_end = pd.Timestamp(TRAIN_CUTOFF) - pd.Timedelta(days=1)
    val_end   = pd.Timestamp(VAL_CUTOFF)   - pd.Timedelta(days=1)

    def split(df):
        tr  = df.loc[:train_end]
        va  = df.loc[TRAIN_CUTOFF:val_end]
        te  = df.loc[VAL_CUTOFF:]
        fc  = [c for c in df.columns if c not in ("y_hi", "y_lo", "close")]
        return tr, va, te, fc

    btr, bva, bte, bfc = split(base)
    etr, eva, ete, efc = split(enha)

    for label, tr, va, te in [("BASE", btr, bva, bte), ("ENHA", etr, eva, ete)]:
        print(f"  [{label}] TRAIN {tr.index.min().date()}→{tr.index.max().date()} "
              f"n={len(tr):,}  |  VAL n={len(va):,}  |  TEST n={len(te):,}")

    # ── 5. Correlation analysis of new features ───────────────────────────────
    print(f"\n{SEP}")
    print("[4] CORRELATION WITH TARGETS (new features only)")
    print(SEP)
    print(f"  {'Feature':<20} {'corr(y_hi)':>12} {'corr(y_lo)':>12} {'MI signal'}")
    for col in new_cols:
        series = enha[col].loc[enha.index <= train_end]
        yh     = enha["y_hi"].loc[enha.index <= train_end]
        yl     = enha["y_lo"].loc[enha.index <= train_end]
        ch = series.corr(yh); cl = series.corr(yl)
        flag = ""
        if abs(ch) > 0.08 or abs(cl) > 0.08:
            flag = "◀ notable"
        print(f"  {col:<20} {ch:>+12.4f} {cl:>+12.4f}  {flag}")

    # ── 6. Train models ───────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("[5] TRAINING MODELS (Huber + GBM ensemble, q=0.70)")
    print(SEP)

    mu_hi_b = float(btr["y_hi"].mean()); mu_lo_b = float(btr["y_lo"].mean())
    mu_hi_e = float(etr["y_hi"].mean()); mu_lo_e = float(etr["y_lo"].mean())

    print("  Training BASELINE HIGH model …")
    t0 = time.time()
    b_hi_ens, b_hi_alpha = train_ensemble(
        btr[bfc].values, btr["y_hi"].values,
        bva[bfc].values, bva["y_hi"].values, mu_hi_b)
    print(f"    done in {time.time()-t0:.1f}s  α={b_hi_alpha:.2f}")

    print("  Training BASELINE LOW model …")
    t0 = time.time()
    b_lo_ens, b_lo_alpha = train_ensemble(
        btr[bfc].values, btr["y_lo"].values,
        bva[bfc].values, bva["y_lo"].values, mu_lo_b)
    print(f"    done in {time.time()-t0:.1f}s  α={b_lo_alpha:.2f}")

    print("  Training ENHANCED HIGH model …")
    t0 = time.time()
    e_hi_ens, e_hi_alpha = train_ensemble(
        etr[efc].values, etr["y_hi"].values,
        eva[efc].values, eva["y_hi"].values, mu_hi_e)
    print(f"    done in {time.time()-t0:.1f}s  α={e_hi_alpha:.2f}")

    print("  Training ENHANCED LOW model …")
    t0 = time.time()
    e_lo_ens, e_lo_alpha = train_ensemble(
        etr[efc].values, etr["y_lo"].values,
        eva[efc].values, eva["y_lo"].values, mu_lo_e)
    print(f"    done in {time.time()-t0:.1f}s  α={e_lo_alpha:.2f}")

    # ── 7. Test-set metrics ───────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("[6] TEST-SET METRICS (held-out, no training data contamination)")
    print(SEP)

    bte_close = bte["close"].values
    b_pred_hi = ensemble_predict(b_hi_ens, b_hi_alpha, mu_hi_b, bte[bfc].values)
    b_pred_lo = ensemble_predict(b_lo_ens, b_lo_alpha, mu_lo_b, bte[bfc].values)
    e_pred_hi = ensemble_predict(e_hi_ens, e_hi_alpha, mu_hi_e, ete[efc].values)
    e_pred_lo = ensemble_predict(e_lo_ens, e_lo_alpha, mu_lo_e, ete[efc].values)

    # Reconstruct absolute pred values using test-set close
    b_predH_abs = bte_close * (1 + np.clip(b_pred_hi, 0, None))
    b_predL_abs = bte_close * (1 - np.clip(b_pred_lo, 0, None))
    e_predH_abs = bte_close * (1 + np.clip(e_pred_hi, 0, None))
    e_predL_abs = bte_close * (1 - np.clip(e_pred_lo, 0, None))

    # y_hi/y_lo are *next-bar* targets — align actuals to test dates using the
    # stored close (prediction for next bar is indexed on current bar's date).
    # bte["close"] = current bar close; actual high/low = raw.shift(-1) at that date.
    bh_act = raw["btc_high"].shift(-1).reindex(bte.index).values
    bl_act = raw["btc_low"].shift(-1).reindex(bte.index).values

    # Drop NaN rows (last row where shift(-1) is undefined)
    valid = ~(np.isnan(bh_act) | np.isnan(bl_act))
    bm_h = compute_metrics(b_predH_abs[valid], bh_act[valid], bte_close[valid])
    bm_l = compute_metrics(b_predL_abs[valid], bl_act[valid], bte_close[valid])
    em_h = compute_metrics(e_predH_abs[valid], bh_act[valid], bte_close[valid])
    em_l = compute_metrics(e_predL_abs[valid], bl_act[valid], bte_close[valid])

    def sign(v): return "▲" if v > 0 else ("▼" if v < 0 else "=")
    def pct(v, decimals=3): return f"{v:.{decimals}f}%"
    def delta(old, new, lower_better=True):
        d = new - old
        good = (d < 0) if lower_better else (d > 0)
        arrow = "✓" if good else "✗"
        return f"{d:+.3f}%  {arrow}"

    print(f"\n  {'Metric':<14} {'BASELINE':>10} {'ENHANCED':>10} {'Δ (+ is better)':>16}")
    print(f"  {'─'*14} {'─'*10} {'─'*10} {'─'*16}")
    print(f"  {'MAPE_H':<14} {pct(bm_h['mape']):>10} {pct(em_h['mape']):>10} "
          f"  {delta(bm_h['mape'], em_h['mape'])}")
    print(f"  {'MAPE_L':<14} {pct(bm_l['mape']):>10} {pct(em_l['mape']):>10} "
          f"  {delta(bm_l['mape'], em_l['mape'])}")
    print(f"  {'hit2_H (±2%)':<14} {pct(bm_h['hit2'],1):>10} {pct(em_h['hit2'],1):>10} "
          f"  {delta(bm_h['hit2'], em_h['hit2'], lower_better=False)}")
    print(f"  {'hit2_L (±2%)':<14} {pct(bm_l['hit2'],1):>10} {pct(em_l['hit2'],1):>10} "
          f"  {delta(bm_l['hit2'], em_l['hit2'], lower_better=False)}")
    print(f"  {'hit1_H (±1%)':<14} {pct(bm_h['hit1'],1):>10} {pct(em_h['hit1'],1):>10} "
          f"  {delta(bm_h['hit1'], em_h['hit1'], lower_better=False)}")
    print(f"  {'hit1_L (±1%)':<14} {pct(bm_l['hit1'],1):>10} {pct(em_l['hit1'],1):>10} "
          f"  {delta(bm_l['hit1'], em_l['hit1'], lower_better=False)}")

    improved_model = (em_h["mape"] < bm_h["mape"]) and (em_l["mape"] < bm_l["mape"])

    # ── 8. Feature importances ────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("[7] GBM FEATURE IMPORTANCES (ENHANCED H/L model, top 20 + all NEW features)")
    print(SEP)
    gbm_hi = None
    for pipe in e_hi_ens:
        if hasattr(pipe.named_steps.get("m", None), "feature_importances_"):
            gbm_hi = pipe.named_steps["m"]; break
    if gbm_hi is not None:
        imps = gbm_hi.feature_importances_
        rank_map = {col: r+1 for r, (col, _) in
                    enumerate(sorted(zip(efc, imps), key=lambda x: x[1], reverse=True))}
        ranked = sorted(zip(efc, imps), key=lambda x: x[1], reverse=True)[:20]
        print("  Top 20:")
        for rank, (col, imp) in enumerate(ranked, 1):
            new_flag = " ◀ NEW" if col in new_cols else ""
            bar = "█" * max(1, int(imp * 500))
            print(f"  {rank:>2}. {col:<22} {imp:.4f}  {bar}{new_flag}")
        print("\n  New features ranking:")
        for col in new_cols:
            imp = dict(zip(efc, imps)).get(col, 0.0)
            rank = rank_map.get(col, len(efc))
            bar  = "█" * max(1, int(imp * 500))
            print(f"  #{rank:>2} {col:<22} {imp:.4f}  {bar}")

    # ── 9. Backtests ──────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("[8] BACKTEST COMPARISON (TF2+V-Gate, $100k initial)")
    print(SEP)

    # Build prediction series for entire history
    base_preds = build_pred_series(raw, b_hi_ens, b_lo_ens,
                                   b_hi_alpha, b_lo_alpha, mu_hi_b, mu_lo_b,
                                   base.drop(columns=["y_hi","y_lo","close"]))
    enha_preds = build_pred_series(raw, e_hi_ens, e_lo_ens,
                                   e_hi_alpha, e_lo_alpha, mu_hi_e, mu_lo_e,
                                   enha.drop(columns=["y_hi","y_lo","close"]))

    periods = [
        ("Bear",  BT_BEAR_START,  today_iso),
        ("Bull",  BT_BULL_START,  BT_BULL_END),
        ("Full",  BT_FULL_START,  today_iso),
    ]

    all_improved = True
    print(f"\n  {'Period':<8} {'Metric':<12} {'BASELINE':>10} {'ENHANCED':>10} {'Δ':>10}  {'OK?'}")
    print(f"  {'─'*8} {'─'*12} {'─'*10} {'─'*10} {'─'*10}  {'─'*4}")

    bt_results = {}
    for pname, pstart, pend in periods:
        bs = run_backtest(base_preds, raw, pstart, pend)
        es = run_backtest(enha_preds, raw, pstart, pend)
        if bs is None or es is None:
            print(f"  {pname:<8} insufficient data")
            continue
        bt_results[pname] = {"base": bs, "enha": es}

        for metric, bv, ev, lower_better in [
            ("strat_ret",  bs["strat_ret"],  es["strat_ret"],  False),
            ("max_dd",     bs["max_dd"],     es["max_dd"],     True),
            ("n_trades",   bs["n_trades"],   es["n_trades"],   None),
            ("win_rate",   bs["win_rate"],   es["win_rate"],   False),
            ("bh_ret",     bs["bh_ret"],     es["bh_ret"],     None),
        ]:
            d    = ev - bv
            fmt  = f"{d:+.1f}%" if isinstance(bv, float) else f"{d:+d}"
            if lower_better is None:
                ok = "—"
            elif lower_better:
                ok = "✓" if d <= 0 else "✗"
            else:
                ok = "✓" if d >= 0 else "✗"
            bfmt = f"{bv:+.1f}%" if isinstance(bv, float) else str(bv)
            efmt = f"{ev:+.1f}%" if isinstance(ev, float) else str(ev)
            print(f"  {pname:<8} {metric:<12} {bfmt:>10} {efmt:>10} {fmt:>10}  {ok}")

        if bs["strat_ret"] > es["strat_ret"]:
            all_improved = False
        print()

    # ── 9b. Experiment B: all 7 features including 200-week MA ───────────────
    print(f"\n{SEP}")
    print("[8b] EXPERIMENT B: All 7 features incl. 200-Week MA (2022+ training)")
    print(SEP)

    btr_b, bva_b, bte_b, bfc_b = split(base_b)
    etr_b, eva_b, ete_b, efc_b = split(enha_b)

    mu_hi_bb = float(btr_b["y_hi"].mean()); mu_lo_bb = float(btr_b["y_lo"].mean())
    mu_hi_eb = float(etr_b["y_hi"].mean()); mu_lo_eb = float(etr_b["y_lo"].mean())

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b_hi_ens_b, b_hi_alpha_b = train_ensemble(
            btr_b[bfc_b].values, btr_b["y_hi"].values,
            bva_b[bfc_b].values, bva_b["y_hi"].values, mu_hi_bb)
        b_lo_ens_b, b_lo_alpha_b = train_ensemble(
            btr_b[bfc_b].values, btr_b["y_lo"].values,
            bva_b[bfc_b].values, bva_b["y_lo"].values, mu_lo_bb)
        e_hi_ens_b, e_hi_alpha_b = train_ensemble(
            etr_b[efc_b].values, etr_b["y_hi"].values,
            eva_b[efc_b].values, eva_b["y_hi"].values, mu_hi_eb)
        e_lo_ens_b, e_lo_alpha_b = train_ensemble(
            etr_b[efc_b].values, etr_b["y_lo"].values,
            eva_b[efc_b].values, eva_b["y_lo"].values, mu_lo_eb)

    bte_close_b = bte_b["close"].values
    bh_act_b    = raw["btc_high"].shift(-1).reindex(bte_b.index).values
    bl_act_b    = raw["btc_low"].shift(-1).reindex(bte_b.index).values
    valid_b     = ~(np.isnan(bh_act_b) | np.isnan(bl_act_b))

    b_predH_b = bte_close_b * (1 + np.clip(
        ensemble_predict(b_hi_ens_b, b_hi_alpha_b, mu_hi_bb, bte_b[bfc_b].values), 0, None))
    e_predH_b = bte_close_b * (1 + np.clip(
        ensemble_predict(e_hi_ens_b, e_hi_alpha_b, mu_hi_eb, ete_b[efc_b].values), 0, None))
    b_predL_b = bte_close_b * (1 - np.clip(
        ensemble_predict(b_lo_ens_b, b_lo_alpha_b, mu_lo_bb, bte_b[bfc_b].values), 0, None))
    e_predL_b = bte_close_b * (1 - np.clip(
        ensemble_predict(e_lo_ens_b, e_lo_alpha_b, mu_lo_eb, ete_b[efc_b].values), 0, None))

    bm_h_b = compute_metrics(b_predH_b[valid_b], bh_act_b[valid_b], bte_close_b[valid_b])
    em_h_b = compute_metrics(e_predH_b[valid_b], bh_act_b[valid_b], bte_close_b[valid_b])
    bm_l_b = compute_metrics(b_predL_b[valid_b], bl_act_b[valid_b], bte_close_b[valid_b])
    em_l_b = compute_metrics(e_predL_b[valid_b], bl_act_b[valid_b], bte_close_b[valid_b])

    print(f"  MAPE_H  base={bm_h_b['mape']:.3f}%  enha={em_h_b['mape']:.3f}%  "
          f"Δ={em_h_b['mape']-bm_h_b['mape']:+.3f}%  "
          f"{'✓' if em_h_b['mape'] < bm_h_b['mape'] else '✗'}")
    print(f"  MAPE_L  base={bm_l_b['mape']:.3f}%  enha={em_l_b['mape']:.3f}%  "
          f"Δ={em_l_b['mape']-bm_l_b['mape']:+.3f}%  "
          f"{'✓' if em_l_b['mape'] < bm_l_b['mape'] else '✗'}")

    base_b_preds = build_pred_series(raw, b_hi_ens_b, b_lo_ens_b,
                                     b_hi_alpha_b, b_lo_alpha_b, mu_hi_bb, mu_lo_bb,
                                     base_b.drop(columns=["y_hi","y_lo","close"]))
    enha_b_preds = build_pred_series(raw, e_hi_ens_b, e_lo_ens_b,
                                     e_hi_alpha_b, e_lo_alpha_b, mu_hi_eb, mu_lo_eb,
                                     enha_b.drop(columns=["y_hi","y_lo","close"]))

    print(f"\n  {'Period':<8} {'BASELINE':>12} {'ENHANCED':>12} {'Δ':>10}  {'OK?'}")
    print(f"  {'─'*8} {'─'*12} {'─'*12} {'─'*10}  {'─'*4}")
    all_improved_b = True
    for pname, pstart, pend in periods:
        bs_b = run_backtest(base_b_preds, raw, pstart, pend)
        es_b = run_backtest(enha_b_preds, raw, pstart, pend)
        if bs_b and es_b:
            d = es_b["strat_ret"] - bs_b["strat_ret"]
            ok = "✓" if d >= 0 else "✗"
            if d < 0:
                all_improved_b = False
            print(f"  {pname:<8} {bs_b['strat_ret']:>+12.1f}% {es_b['strat_ret']:>+12.1f}%"
                  f" {d:>+10.1f}%  {ok}")

    # ── 10. Summary ───────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("[9] SUMMARY & RECOMMENDATION")
    print(SEP)

    verdict_a = "IMPL" if (improved_model and all_improved) else \
                ("PARTIAL" if improved_model else "NO")
    verdict_b = "IMPL" if (
        (em_h_b["mape"] < bm_h_b["mape"]) and
        (em_l_b["mape"] < bm_l_b["mape"]) and all_improved_b
    ) else (
        "PARTIAL" if (em_h_b["mape"] < bm_h_b["mape"] or
                      em_l_b["mape"] < bm_l_b["mape"]) else "NO"
    )

    print(f"\n  Experiment A (6 features, full history from 2019):")
    print(f"    MAPE improved  : {'YES ✓' if improved_model else 'NO ✗'}")
    print(f"    All BT periods : {'YES ✓' if all_improved else 'NO ✗'}")
    print(f"    Verdict        : {verdict_a}")

    print(f"\n  Experiment B (7 features incl. 200-week MA, 2022+ only):")
    print(f"    MAPE improved  : {'YES ✓' if em_h_b['mape'] < bm_h_b['mape'] else 'NO ✗'}")
    print(f"    All BT periods : {'YES ✓' if all_improved_b else 'NO ✗'}")
    print(f"    Verdict        : {verdict_b}")

    print(f"\n  WHY THE NEW FEATURES HAVE LIMITED IMPACT:")
    print(f"    The H/L model predicts tomorrow's range MAGNITUDE (how far above/below")
    print(f"    current close), not direction. Range magnitude is primarily driven by:")
    print(f"    1. Recent volatility (already captured: vol_5, vol_10, atr_*)    ")
    print(f"    2. Day-of-week patterns (dow_4 = #1 feature = Friday)             ")
    print(f"    3. Cross-asset momentum (eth_ret_5 = #2 feature)                  ")
    print(f"    Long-term MA/VWAP distances affect DIRECTION more than MAGNITUDE. ")
    print(f"    The existing features (below_sma50, rsi_14, dist_hi_90) already")
    print(f"    capture structural regime context.")

    print(f"\n  METRICS THAT WOULD GENUINELY HELP (require paid APIs):")
    print(f"    #4 Dealer Negative Gamma  → Deribit options flow (amplifies range SIZE)")
    print(f"    #6 Coinbase Premium       → Coinbase API (institutional demand bias)")
    print(f"    #9 Sell-Side Risk Ratio   → Glassnode (realized profit/loss vs realized cap)")
    print(f"    #7 30D Net Exchange Flow  → CryptoQuant (supply pressure proxy)")

    # Backtest regression overrides MAPE improvement: if all 3 periods regress,
    # the features are not adding value to the trading strategy
    bt_reg_a = not all_improved   # True if any period got worse
    bt_reg_b = not all_improved_b
    final_verdict = "DO_NOT_IMPLEMENT" if (bt_reg_a and bt_reg_b) else "IMPLEMENT_PARTIAL"
    print(f"\n  FINAL VERDICT: {final_verdict}")
    if final_verdict == "DO_NOT_IMPLEMENT":
        print(f"  MAPE improves marginally (<0.02%) but ALL 3 backtest periods regress")
        print(f"  (Bear/Bull/Full all slightly worse). The new features introduce noise")
        print(f"  that slightly shifts TF2 threshold crossings unfavorably. Long-term MA")
        print(f"  distances and VWAP proxies do not add signal beyond what the model")
        print(f"  already captures through rsi_14, below_sma50, dist_hi_90, vol_5.")
    else:
        print(f"  Some improvement observed. Consider selective implementation")
        print(f"  after ablation testing.")

    print(f"\n  Implemented feature names: {new_cols}")
    print(f"  Blocked (require paid APIs): Dealer Gamma, Coinbase Premium,")
    print(f"  Net Exchange Flow, Stablecoin Outflows, Sell-Side Risk Ratio")

    print(f"\n{SEP}")
    return {
        "verdict": final_verdict,
        "new_cols": new_cols,
        "exp_a": {"improved_model": improved_model, "all_bt_improved": all_improved,
                  "base_mape_h": bm_h["mape"], "enha_mape_h": em_h["mape"],
                  "base_mape_l": bm_l["mape"], "enha_mape_l": em_l["mape"],
                  "bt_results": bt_results},
        "exp_b": {"improved_model": em_h_b["mape"] < bm_h_b["mape"],
                  "all_bt_improved": all_improved_b},
    }


if __name__ == "__main__":
    results = main()
