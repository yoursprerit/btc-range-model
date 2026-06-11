"""
D3 Lo-Break Early Intraday Trigger — Evaluation
================================================
Question: Would triggering D3 exit early (when today's intraday bar has already
breached pred_lo) improve strategy returns vs the current implementation (which
evaluates after the bar closes)?

Methodology
-----------
Current behaviour:  D3 fires at bar i when lo_brk[i]=1 (act_lo < pred_lo).
                    Exit executes at act_close[i].

Early trigger sim:  Same signal condition, but exit executes at pred_lo[i]
                    (the BTC band level at which the intraday breach first occurs).
                    For MSTR/MSTU, exit price is scaled proportionally:
                        mstr_exit_early[i] = mstr_px[i] * (pred_lo_btc[i] / act_close_btc[i])
                    This assumes MSTR/MSTU tracks BTC's intraday move to the breach level.

Trade-off
---------
  Recovery bars (act_close > pred_lo): early exit < close → WORSE
  Continuation bars (act_close < pred_lo): early exit > close → BETTER

Assets & periods (matching UI backtesting section)
---------------------------------------------------
  BTC   full 2024-05-26 → today
        bull 2024-06-01 → 2025-06-01
        bear 2025-06-01 → today
  MSTR  same date split
  MSTU  bear 2025-06-04 → today (inception-limited)
"""

import sys, os, warnings, joblib, requests
warnings.filterwarnings("ignore")
from pathlib import Path
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import yfinance as yf

# ─── Load model ───────────────────────────────────────────────────────────────
MODEL_PATH = str(_ROOT / "models" / "inference_assets_ct.joblib")
AD = joblib.load(MODEL_PATH)
print(f"Model loaded. Feature count: {len(AD['feat_cols'])}")

# ─── Macro / on-chain symbols ─────────────────────────────────────────────────
_MACRO_SYMS = {"eth": "ETH-USD", "spx": "^GSPC", "ndx": "^IXIC",
               "vix": "^VIX", "gold": "GC=F", "dxy": "DX-Y.NYB", "tnx": "^TNX"}
_ONCHAIN = ["hash-rate", "difficulty", "n-transactions", "miners-revenue",
            "n-unique-addresses", "transaction-fees-usd", "mempool-size",
            "estimated-transaction-volume-usd", "market-cap",
            "avg-block-size", "cost-per-transaction"]


def _yf_single(sym, start, end):
    d = yf.download(sym, start=start, end=end, progress=False, auto_adjust=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] for c in d.columns]
    d.index = pd.DatetimeIndex(d.index).tz_localize(None).normalize()
    return d


def fetch_data(fetch_start, fetch_end):
    frames = {}
    d = _yf_single("BTC-USD", fetch_start, fetch_end)
    frames.update({"btc_close": d["Close"], "btc_high": d["High"],
                   "btc_low": d["Low"], "btc_volume": d["Volume"]})
    for name, sym in _MACRO_SYMS.items():
        try:
            frames[f"{name}_close"] = _yf_single(sym, fetch_start, fetch_end)["Close"]
        except Exception:
            pass
    df = pd.DataFrame(frames).ffill(limit=5)
    df.index.name = "date"
    ok = 0
    for m in _ONCHAIN:
        col = f"oc_{m.replace('-', '_')}"
        try:
            r = requests.get(
                f"https://api.blockchain.info/charts/{m}",
                params={"timespan": "3years", "format": "json", "sampled": "true"},
                timeout=20)
            vals = r.json().get("values", [])
            s = pd.Series(
                {pd.Timestamp(v["x"], unit="s").normalize(): v["y"] for v in vals},
                name=col, dtype=float)
            s = s[~s.index.duplicated(keep="last")].sort_index()
            s.index = pd.DatetimeIndex(s.index).tz_localize(None)
            df[col] = s.reindex(df.index).ffill(limit=7)
            ok += 1
        except Exception:
            pass
    print(f"  On-chain: {ok}/{len(_ONCHAIN)} OK")
    return df


def build_preds(df, fetch_start="2020-01-01", fetch_end="2030-01-01"):
    import time as _time
    c, h, l_, v = df["btc_close"], df["btc_high"], df["btc_low"], df["btc_volume"]
    ret = np.log(c).diff()
    f = pd.DataFrame(index=df.index)
    for k in [1, 3, 5, 7, 14, 30]:
        f[f"ret_{k}"] = ret.rolling(k).sum()
    for k in [5, 10, 20, 30]:
        f[f"vol_{k}"] = ret.rolling(k).std()
    prev_c = c.shift(1)
    tr = pd.concat([(h-l_), (h-prev_c).abs(), (l_-prev_c).abs()], axis=1).max(axis=1)
    for k in [7, 14, 30]:
        f[f"atr_{k}"] = tr.rolling(k).mean() / c
    f["range_today"] = (h-l_) / c
    f["range_ma7"]   = ((h-l_)/c).rolling(7).mean()
    f["range_ma30"]  = ((h-l_)/c).rolling(30).mean()
    f["range_std30"] = ((h-l_)/c).rolling(30).std()
    g = c.diff().clip(lower=0).rolling(14).mean()
    ls = (-c.diff().clip(upper=0)).rolling(14).mean()
    f["rsi_14"] = 100 - 100 / (1 + g / ls.replace(0, np.nan))
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    f["macd"]      = macd / c
    f["macd_sig"]  = macd.ewm(span=9, adjust=False).mean() / c
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    ma20 = c.rolling(20).mean()
    sd20 = c.rolling(20).std()
    f["bb_width"]   = (4*sd20) / ma20
    f["dist_hi_30"] = c / c.rolling(30).max() - 1
    f["dist_lo_30"] = c / c.rolling(30).min() - 1
    f["dist_hi_90"] = c / c.rolling(90).max() - 1
    f["vol_chg_1"]  = np.log(v).diff()
    f["vol_z_20"]   = (np.log(v) - np.log(v).rolling(20).mean()) / np.log(v).rolling(20).std()
    f["vol_ma_ratio"] = v / v.rolling(20).mean()
    dow = df.index.dayofweek
    for i in range(6):
        f[f"dow_{i}"] = (dow == i).astype(float)
    for nm, col in [("spx", "spx_close"), ("ndx", "ndx_close"), ("vix", "vix_close"),
                    ("gold", "gold_close"), ("dxy", "dxy_close"), ("tnx", "tnx_close"),
                    ("eth", "eth_close")]:
        if col not in df.columns:
            continue
        s = df[col]
        lr = np.log(s).diff()
        for k in [1, 5, 20]:
            f[f"{nm}_ret_{k}"] = lr.rolling(k).sum()
        f[f"{nm}_vol_20"] = lr.rolling(20).std()
    for cn, cc in [("spx", "spx_close"), ("ndx", "ndx_close"),
                   ("gold", "gold_close"), ("dxy", "dxy_close")]:
        if cc in df.columns:
            f[f"btc_{cn}_corr_30"] = ret.rolling(30).corr(np.log(df[cc]).diff())
    for col in [x for x in df.columns if x.startswith("oc_")]:
        s = df[col].astype(float)
        sl = np.log(s.replace(0, np.nan))
        f[f"{col}_d1"] = sl.diff(1)
        f[f"{col}_d7"] = sl.diff(7)
        f[f"{col}_z30"] = (sl - sl.rolling(30).mean()) / sl.rolling(30).std()
    nh = h.shift(-1)
    nl = l_.shift(-1)
    yh = (nh - c) / c
    yl = (c - nl) / c
    f["y_hi_ema3"] = yh.shift(1).ewm(span=3, adjust=False).mean()
    f["y_lo_ema3"] = yl.shift(1).ewm(span=3, adjust=False).mean()
    f["y_hi_ema7"] = yh.shift(1).ewm(span=7, adjust=False).mean()
    f["y_lo_ema7"] = yl.shift(1).ewm(span=7, adjust=False).mean()
    p3h = h.shift(1).rolling(3).max()
    p3l = l_.shift(1).rolling(3).min()
    f["above_3d_high"] = (c > p3h).astype(float)
    f["below_3d_low"]  = (c < p3l).astype(float)
    f["bo_strength_up"] = (c / p3h - 1).clip(lower=0)
    f["bo_strength_dn"] = (1 - c / p3l).clip(lower=0)
    ya = yh.shift(1)
    yb = yl.shift(1)
    f["y_hi_surprise"] = ya - ya.ewm(span=7, adjust=False).mean()
    f["y_lo_surprise"] = yb - yb.ewm(span=7, adjust=False).mean()
    nr = ret.clip(upper=0)
    f["dn_vol_5"]  = nr.rolling(5).std()
    f["dn_vol_20"] = nr.rolling(20).std()
    sma50 = c.rolling(50).mean()
    f["below_sma50"]    = (c < sma50).astype(float)
    f["below_sma50_5d"] = f["below_sma50"].rolling(5).min().fillna(0)

    # Coinbase premium (fallback 0 if API unavailable — mirrors app behaviour)
    _CB_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    _cb_rows = []
    _cb_cur = pd.Timestamp(fetch_start)
    _cb_end = pd.Timestamp(fetch_end)
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
        _time.sleep(0.1)
    if _cb_rows:
        _cb_df = pd.DataFrame(_cb_rows, columns=["ts", "low", "high", "open", "close", "volume"])
        _cb_df["date"] = pd.to_datetime(_cb_df["ts"], unit="s").dt.normalize()
        _cb_df = _cb_df.drop_duplicates("date").set_index("date").sort_index()
        _cb_close = _cb_df["close"].reindex(df.index).astype(float)
        _prem = (_cb_close - c) / c * 100
        f["cb_premium"]     = _prem
        f["cb_premium_ma3"] = _prem.rolling(3).mean()
        f["cb_premium_z7"]  = (_prem - _prem.rolling(7).mean()) / _prem.rolling(7).std()
        print(f"  Coinbase premium: {int(_prem.notna().sum())} days OK")
    else:
        f["cb_premium"] = f["cb_premium_ma3"] = f["cb_premium_z7"] = 0.0
        print("  Coinbase premium: unavailable (set to 0)")

    fc = AD["feat_cols"]
    f = f.replace([np.inf, -np.inf], np.nan)
    for col in fc:
        if col not in f.columns:
            f[col] = np.nan
    for _cb_col in ["cb_premium", "cb_premium_ma3", "cb_premium_z7"]:
        if _cb_col in f.columns:
            f[_cb_col] = f[_cb_col].fillna(0.0)
    F = f[fc].dropna()
    if F.empty:
        raise RuntimeError("Feature matrix empty")
    print(f"  CT predictions: {len(F)} rows")
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
    cv = c.reindex(F.index).values
    ph = cv * (1 + np.clip(yhi, 0, None))
    pl = cv * (1 - np.clip(ylo, 0, None))
    idx = np.asarray(F.index, dtype="datetime64[ns]")
    nd = np.empty(len(F), dtype="datetime64[ns]")
    nd[:-1] = idx[1:]
    nd[-1]  = idx[-1] + np.timedelta64(1, "D")
    res = pd.DataFrame({"close_asof": cv, "pred_high": ph, "pred_low": pl},
                       index=pd.DatetimeIndex(nd, name="target_date"))
    return res[~res.index.duplicated(keep="last")]


# ─── Signal computation ────────────────────────────────────────────────────────
def compute_signals(comp):
    """Return arrays of signal flags for a reset-indexed comp DataFrame."""
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
        s = max(0, i-2)
        ehma3[i] = np.mean(err_hi[s:i+1])
        elma3[i] = np.mean(err_lo[s:i+1])
        hb3[i]   = int(np.sum(hi_brk[s:i+1]))
        lb3[i]   = int(np.sum(lo_brk[s:i+1]))

    u1 = (ehma3 > 0.7) & (hb3 >= 2)
    d1 = (lb3 >= 2)    & (elma3 > 0.5)
    d2 = ehma3 < -0.75
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
        lo_i = max(0, i-7)
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
    v_rev_bar = (dn_score_arr > 0.8) & (err_lo > 3.0)
    v_recent  = np.zeros(N, dtype=bool)
    for i in range(N):
        v_recent[i] = bool(np.any(v_rev_bar[max(0, i-2):i+1]))

    tf1_entry = u1 & (above_ma30 | clean_10d | v_recent)

    return dict(
        pred_lo=pred_lo, pred_hi=pred_hi,
        act_lo=act_lo, act_hi=act_hi,
        err_hi=err_hi, err_lo=err_lo,
        hi_brk=hi_brk, lo_brk=lo_brk,
        d1=d1, d2=d2, d3=d3,
        bull_regime=bull_regime,
        above_ma30=above_ma30,
        clean_10d=clean_10d,
        v_recent=v_recent,
        tf1_entry=tf1_entry,
    )


# ─── Generic backtest loop ─────────────────────────────────────────────────────
def run_backtest(dates, exec_px, sig, start_dt, initial_capital=100_000.0,
                 early_trigger=False, btc_pred_lo=None, btc_act_close=None):
    """
    Parameters
    ----------
    exec_px      : execution prices for the asset (BTC close or MSTR/MSTU close)
    sig          : dict of signal arrays from compute_signals (BTC-based)
    early_trigger: if True, D3 exits at pred_lo-scaled price instead of close
    btc_pred_lo  : BTC pred_lo array (needed for early_trigger scaling)
    btc_act_close: BTC actual_close array (needed for early_trigger scaling)
    """
    WARMUP = 35
    N = len(dates)
    d3         = sig["d3"]
    d2         = sig["d2"]
    bull_regime= sig["bull_regime"]
    tf1_entry  = sig["tf1_entry"]
    pred_lo    = sig["pred_lo"]     # BTC pred_lo (used as early-trigger threshold)
    act_close  = btc_act_close if btc_act_close is not None else exec_px

    _bt0 = max(WARMUP, int(np.searchsorted(np.array(dates, dtype="datetime64[ns]"),
                                            np.datetime64(start_dt, "ns"))))
    if N - _bt0 < 3:
        return None

    nav = initial_capital; pos = "CASH"; qty = 0.0
    e_price = e_nav = e_date = e_trigger = None
    trades = []; nav_arr = np.full(N, np.nan)

    for i in range(N):
        price = exec_px[i]
        if i < _bt0:
            nav_arr[i] = initial_capital
            continue

        if pos == "LONG":
            cur = qty * price
            should_exit = bool(d3[i] or (d2[i] and not bull_regime[i]))
            exit_lbl    = "D3" if d3[i] else "D2 (bear)"
            if should_exit:
                if early_trigger and d3[i]:
                    # Scale execution price: exit at the BTC pred_lo level rather than close
                    # For BTC itself: exec at pred_lo[i]
                    # For MSTR/MSTU: exec at price * (pred_lo_btc / act_close_btc)
                    if act_close is exec_px:
                        exit_px = pred_lo[i]          # BTC: exit at exact breach level
                    else:
                        # MSTR/MSTU: proportional to BTC's position relative to close
                        btc_ratio = btc_pred_lo[i] / btc_act_close[i]
                        exit_px = price * btc_ratio   # scale MSTR/MSTU proportionally
                    exit_px = max(exit_px, price * 0.5)  # sanity floor
                else:
                    exit_px = price
                nav = qty * exit_px
                trades.append(dict(
                    entry_date=e_date, entry_price=e_price, entry_nav=e_nav,
                    entry_trigger=e_trigger, exit_date=dates[i], exit_price=exit_px,
                    exit_nav=nav, pnl_pct=(exit_px/e_price - 1)*100,
                    pnl_abs=nav - e_nav, exit_signal=exit_lbl,
                    duration_days=(dates[i] - e_date).days,
                    # For analysis: capture close and pred_lo at exit bar
                    close_at_exit=price,
                    pred_lo_at_exit=pred_lo[i],
                    btc_pred_lo=btc_pred_lo[i] if btc_pred_lo is not None else pred_lo[i],
                    btc_act_close=btc_act_close[i] if btc_act_close is not None else price,
                ))
                pos = "CASH"; qty = 0.0
            else:
                nav = cur
        else:
            _exit_at_i = d3[i] or (d2[i] and not bull_regime[i])
            if tf1_entry[i] and not _exit_at_i:
                qty = nav / price; e_price = price; e_date = dates[i]
                e_nav = nav; pos = "LONG"
                e_trigger = "U1"
        nav_arr[i] = qty * price if pos == "LONG" else nav

    if pos == "LONG":
        nav_arr[N-1] = qty * exec_px[N-1]

    nav_series = pd.Series(nav_arr[_bt0:], index=dates[_bt0:]).ffill()
    bh_series  = pd.Series(
        initial_capital * exec_px[_bt0:] / exec_px[_bt0], index=dates[_bt0:])

    return dict(nav=nav_series, bh=bh_series, trades=trades)


# ─── Metrics ──────────────────────────────────────────────────────────────────
def metrics(res, label):
    nav = res["nav"]; bh = res["bh"]; trades = res["trades"]; cap = 100_000.0
    rf_daily = (1.045)**(1/252) - 1
    total_ret  = (nav.iloc[-1]/cap - 1)*100
    bh_ret     = (bh.iloc[-1]/cap  - 1)*100
    n_years    = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr       = ((nav.iloc[-1]/cap)**(1/max(n_years, 0.01)) - 1)*100
    dr         = nav.pct_change().fillna(0)
    bh_dr      = bh.pct_change().fillna(0)
    exc        = dr - rf_daily
    sharpe     = float(exc.mean()/exc.std()*np.sqrt(252)) if exc.std() > 0 else 0
    bh_exc     = bh_dr - rf_daily
    bh_sharpe  = float(bh_exc.mean()/bh_exc.std()*np.sqrt(252)) if bh_exc.std() > 0 else 0
    rm         = nav.cummax()
    max_dd     = float(((nav - rm)/rm*100).min())
    rm_bh      = bh.cummax()
    bh_max_dd  = float(((bh - rm_bh)/rm_bh*100).min())
    d3_exits   = [t for t in trades if t["exit_signal"] == "D3"]
    win_rate   = 100*len([t for t in trades if t["pnl_pct"] > 0]) / max(1, len(trades))
    return dict(
        label=label, total_ret=total_ret, bh_ret=bh_ret, cagr=cagr,
        sharpe=sharpe, bh_sharpe=bh_sharpe,
        max_dd=max_dd, bh_max_dd=bh_max_dd,
        n_trades=len(trades), win_rate=win_rate,
        n_d3_exits=len(d3_exits),
        final_nav=float(nav.iloc[-1]),
    )


# ─── D3 exit analysis helper ──────────────────────────────────────────────────
def analyse_d3_exits(base_trades, asset_label):
    """Print trade-level analysis of D3 exits.

    For each D3 exit: compares BTC close vs BTC pred_lo to determine whether
    early trigger (at pred_lo) would have been better or worse than close-of-day.
    Also shows the simulated early-trigger execution price for the traded asset.
    """
    d3t = [t for t in base_trades if t["exit_signal"] == "D3"]
    if not d3t:
        print(f"  {asset_label}: no D3 exits found")
        return
    rows = []
    for t in d3t:
        asset_close = t["close_at_exit"]       # asset (BTC/MSTR/MSTU) close
        btc_close   = t["btc_act_close"]       # BTC close (signal dimension)
        btc_plo     = t["btc_pred_lo"]         # BTC pred_lo (trigger threshold)
        # BTC recovery: did BTC close recover above pred_lo?
        btc_recovered = btc_close > btc_plo
        # BTC price delta at trigger vs close (pct)
        btc_delta_pct = (btc_plo / btc_close - 1) * 100 if btc_close > 0 else 0
        # Asset early-trigger exit = asset_close scaled by BTC breach ratio
        btc_ratio = btc_plo / btc_close if btc_close > 0 else 1.0
        asset_early_exit = asset_close * btc_ratio
        asset_delta_pct  = (btc_ratio - 1) * 100   # same ratio, asset-independent
        rows.append({
            "exit_date":        t["exit_date"].strftime("%Y-%m-%d") if hasattr(t["exit_date"], "strftime") else str(t["exit_date"]),
            "btc_close":        round(btc_close, 0),
            "btc_pred_lo":      round(btc_plo, 0),
            "btc_recovered":    btc_recovered,
            "btc_delta_pct":    round(btc_delta_pct, 3),
            "asset_close":      round(asset_close, 2),
            "asset_early_px":   round(asset_early_exit, 2),
            "asset_delta_pct":  round(asset_delta_pct, 3),
        })
    df = pd.DataFrame(rows)
    n     = len(df)
    n_rec = int(df["btc_recovered"].sum())
    n_cont = n - n_rec
    print(f"\n  {asset_label} — {n} D3 exits")
    print(f"    BTC recovered above pred_lo (early exit WORSE):  {n_rec}/{n}  ({100*n_rec/n:.0f}%)")
    print(f"    BTC continued below pred_lo (early exit BETTER): {n_cont}/{n}  ({100*n_cont/n:.0f}%)")
    print(f"    Avg BTC delta (pred_lo vs close): {df['btc_delta_pct'].mean():+.3f}%")
    cols = ["exit_date", "btc_close", "btc_pred_lo", "btc_recovered",
            "btc_delta_pct", "asset_close", "asset_early_px", "asset_delta_pct"]
    print(df[cols].to_string(index=False))
    return df


# ─── Compare two backtest results ─────────────────────────────────────────────
def compare(base, early, label):
    bm = metrics(base, "Base")
    em = metrics(early, "Early")
    delta_ret    = em["total_ret"] - bm["total_ret"]
    delta_sharpe = em["sharpe"]    - bm["sharpe"]
    delta_dd     = em["max_dd"]    - bm["max_dd"]  # negative delta = more drawdown
    nav_diff_abs = em["final_nav"] - bm["final_nav"]
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  {'Metric':<28} {'Base':>10} {'Early':>10} {'Delta':>10}")
    print(f"  {'-'*58}")
    print(f"  {'Total return (%)':28} {bm['total_ret']:>10.1f} {em['total_ret']:>10.1f} {delta_ret:>+10.1f}")
    print(f"  {'CAGR (%)':28} {bm['cagr']:>10.1f} {em['cagr']:>10.1f} {em['cagr']-bm['cagr']:>+10.1f}")
    print(f"  {'Sharpe':28} {bm['sharpe']:>10.2f} {em['sharpe']:>10.2f} {delta_sharpe:>+10.2f}")
    print(f"  {'Max drawdown (%)':28} {bm['max_dd']:>10.1f} {em['max_dd']:>10.1f} {delta_dd:>+10.1f}")
    print(f"  {'Win rate (%)':28} {bm['win_rate']:>10.1f} {em['win_rate']:>10.1f} {em['win_rate']-bm['win_rate']:>+10.1f}")
    print(f"  {'Trades':28} {bm['n_trades']:>10d} {em['n_trades']:>10d} {em['n_trades']-bm['n_trades']:>+10d}")
    print(f"  {'D3 exits':28} {bm['n_d3_exits']:>10d} {em['n_d3_exits']:>10d} {em['n_d3_exits']-bm['n_d3_exits']:>+10d}")
    print(f"  {'Final NAV ($)':28} {bm['final_nav']:>10,.0f} {em['final_nav']:>10,.0f} {nav_diff_abs:>+10,.0f}")
    print(f"  {'B&H return (%)':28} {bm['bh_ret']:>10.1f}")
    verdict = "BETTER" if delta_ret > 0 else "WORSE"
    print(f"\n  Verdict: Early trigger is {verdict} by {abs(delta_ret):.1f}pp total return, {delta_sharpe:+.2f} Sharpe")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    TODAY = pd.Timestamp.today().normalize()
    PERIODS = {
        "btc_full":  ("2024-05-26", TODAY.strftime("%Y-%m-%d")),
        "btc_bull":  ("2024-06-01", "2025-06-01"),
        "btc_bear":  ("2025-06-01", TODAY.strftime("%Y-%m-%d")),
        "mstr_full": ("2024-05-26", TODAY.strftime("%Y-%m-%d")),
        "mstr_bull": ("2024-06-01", "2025-06-01"),
        "mstr_bear": ("2025-06-01", TODAY.strftime("%Y-%m-%d")),
        "mstu_bear": ("2025-06-04", TODAY.strftime("%Y-%m-%d")),
    }

    # Fetch bounds
    all_starts = [pd.Timestamp(v[0]) for v in PERIODS.values()]
    all_ends   = [pd.Timestamp(v[1]) for v in PERIODS.values()]
    fetch_start = (min(all_starts) - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    fetch_end   = (max(all_ends)   + pd.Timedelta(days=3)).strftime("%Y-%m-%d")

    print(f"\nFetching data {fetch_start} → {fetch_end} …")
    df_raw = fetch_data(fetch_start, fetch_end)

    print("Building BTC predictions …")
    preds = build_preds(df_raw, fetch_start=fetch_start, fetch_end=fetch_end)

    # Merge actuals into predictions
    preds["actual_high"]  = df_raw["btc_high"].reindex(preds.index)
    preds["actual_low"]   = df_raw["btc_low"].reindex(preds.index)
    preds["actual_close"] = df_raw["btc_close"].reindex(preds.index)
    comp_full = preds.dropna(subset=["actual_high", "actual_low", "actual_close"]).reset_index()

    # Fetch MSTR & MSTU prices
    print("Fetching MSTR / MSTU prices …")
    start_mstr = (min(all_starts) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    d_mstr = _yf_single("MSTR", start_mstr, fetch_end)
    d_mstu = _yf_single("MSTU", start_mstr, fetch_end)

    mstr_close_s = d_mstr["Close"].sort_index() if "Close" in d_mstr.columns else None
    mstu_close_s = d_mstu["Close"].sort_index() if "Close" in d_mstu.columns else None

    if mstr_close_s is None:
        print("WARNING: MSTR data unavailable, skipping MSTR/MSTU")

    # Reindex MSTR/MSTU to all calendar days then map to comp_full dates
    def _extend_px(raw_series, target_dates):
        if raw_series is None:
            return None
        full_idx = pd.date_range(raw_series.index[0],
                                  max(raw_series.index[-1], target_dates[-1]), freq="D")
        return raw_series.reindex(full_idx).ffill().reindex(target_dates).ffill().bfill().values.astype(float)

    dates_full = pd.DatetimeIndex(comp_full["target_date"])
    mstr_px = _extend_px(mstr_close_s, dates_full)
    mstu_px = _extend_px(mstu_close_s, dates_full)

    # Pre-compute signal arrays for the full history (once, for all periods)
    sig_full = compute_signals(comp_full)
    btc_pred_lo_full  = comp_full["pred_low"].values.astype(float)
    btc_act_close_full = comp_full["actual_close"].values.astype(float)

    print("\n" + "="*60)
    print("  D3 EXIT ANALYSIS (base backtest — all D3 exits full history)")
    print("="*60)

    # ── Run full-period base backtest to collect D3 exit details ──────────────
    res_btc_base_full = run_backtest(
        dates_full, btc_act_close_full, sig_full,
        pd.Timestamp("2024-05-26"),
        early_trigger=False,
        btc_pred_lo=btc_pred_lo_full,
        btc_act_close=btc_act_close_full,
    )
    if res_btc_base_full:
        analyse_d3_exits(res_btc_base_full["trades"], "BTC (full period)")

    if mstr_px is not None:
        res_mstr_base_full = run_backtest(
            dates_full, mstr_px, sig_full,
            pd.Timestamp("2024-05-26"),
            early_trigger=False,
            btc_pred_lo=btc_pred_lo_full,
            btc_act_close=btc_act_close_full,
        )
        if res_mstr_base_full:
            analyse_d3_exits(res_mstr_base_full["trades"], "MSTR (full period)")

    if mstu_px is not None:
        res_mstu_base_full = run_backtest(
            dates_full, mstu_px, sig_full,
            pd.Timestamp("2025-06-04"),
            early_trigger=False,
            btc_pred_lo=btc_pred_lo_full,
            btc_act_close=btc_act_close_full,
        )
        if res_mstu_base_full:
            analyse_d3_exits(res_mstu_base_full["trades"], "MSTU (bear/inception)")

    # ── Period-by-period comparisons ──────────────────────────────────────────
    asset_configs = [
        ("BTC",  btc_act_close_full, btc_pred_lo_full, btc_act_close_full),
        ("MSTR", mstr_px,            btc_pred_lo_full, btc_act_close_full),
        ("MSTU", mstu_px,            btc_pred_lo_full, btc_act_close_full),
    ]

    period_map = {
        "BTC":  [("Full",  "2024-05-26", TODAY.strftime("%Y-%m-%d")),
                 ("Bull",  "2024-06-01", "2025-06-01"),
                 ("Bear",  "2025-06-01", TODAY.strftime("%Y-%m-%d"))],
        "MSTR": [("Full",  "2024-05-26", TODAY.strftime("%Y-%m-%d")),
                 ("Bull",  "2024-06-01", "2025-06-01"),
                 ("Bear",  "2025-06-01", TODAY.strftime("%Y-%m-%d"))],
        "MSTU": [("Bear",  "2025-06-04", TODAY.strftime("%Y-%m-%d"))],
    }

    for asset_name, exec_px, bplo, bclose in asset_configs:
        if exec_px is None:
            print(f"\nSkipping {asset_name} — price data unavailable")
            continue
        for period_name, pstart, pend in period_map[asset_name]:
            start_ts = pd.Timestamp(pstart)
            end_ts   = pd.Timestamp(pend)
            # Restrict comp_full to data <= pend
            mask = pd.DatetimeIndex(comp_full["target_date"]) <= end_ts
            if mask.sum() < 40:
                print(f"  Skipping {asset_name} {period_name} — insufficient data")
                continue

            comp_sub   = comp_full[mask].reset_index(drop=True)
            sig_sub    = compute_signals(comp_sub)
            dates_sub  = pd.DatetimeIndex(comp_sub["target_date"])
            exec_sub   = exec_px[:mask.sum()]
            bplo_sub   = bplo[:mask.sum()]
            bclose_sub = bclose[:mask.sum()]

            base = run_backtest(dates_sub, exec_sub, sig_sub, start_ts,
                                early_trigger=False,
                                btc_pred_lo=bplo_sub, btc_act_close=bclose_sub)
            early = run_backtest(dates_sub, exec_sub, sig_sub, start_ts,
                                 early_trigger=True,
                                 btc_pred_lo=bplo_sub, btc_act_close=bclose_sub)

            if base and early:
                compare(base, early, f"{asset_name} — {period_name} ({pstart} → {pend})")
            else:
                print(f"\n  {asset_name} {period_name}: insufficient data to run backtest")

    print("\nDone.")


if __name__ == "__main__":
    main()
