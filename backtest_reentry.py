"""
Re-Entry Criteria Evaluation after Stop-Loss Exits
====================================================
After a stop-loss is triggered (Trail-7% baseline), evaluates smart re-entry
criteria that let the position recapture upside without repeated whipsaws.

Re-entry criteria are ONLY applied after stop-loss exits, not after normal
D2/D3 signal exits (which are clean exits and can re-enter normally).

Base stop-loss: Trail-7% (trailing 7% from peak close since entry)

Re-entry Criteria Tested:
  RE0 — Immediate      : Standard U1 re-entry immediately after SL (current)
  RE1 — Cooldown-3d    : Block re-entry for 3 bars after SL exit
  RE2 — Cooldown-5d    : Block re-entry for 5 bars after SL exit
  RE3 — Recovery+2%    : Only re-enter when asset price > sl_exit × 1.02
  RE4 — Recovery+5%    : Only re-enter when asset price > sl_exit × 1.05
  RE5 — MA30-Mandatory : Only re-enter after SL if BTC price is above MA30
  RE6 — Strong-U1      : After SL, require err_hi_ma3 > 1.2 AND hi_breaks >= 3
  RE7 — 3d+MA30        : 3-bar cooldown AND above MA30 (combines RE1+RE5)
  RE8 — D2Reset+MA30   : After SL, wait for D2 to fire once (momentum cleared),
                         then re-enter on next U1 above MA30

Also included for reference:
  Baseline (no SL)     : Strategy without any stop-loss applied

Assets: BTC, MSTR, MSTU
Periods: Bear (Jun 2025–May 2026), Bull (Jun 2024–Jun 2025),
         Full (Jun 2024–May 2026), OOS (Mar 2026–today)

Usage:
  python backtest_reentry.py
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

PERIODS = {
    "Bear Market (Jun 2025–May 2026)": ("2025-06-01", "2026-05-31"),
    "Bull Market (Jun 2024–Jun 2025)": ("2024-06-05", "2025-06-14"),
    "Full Market (Jun 2024–May 2026)": ("2024-06-01", "2026-05-31"),
    "OOS Only   (Mar 2026–Jun 2026)":  ("2026-03-01", "2026-06-07"),
}

FETCH_START    = "2024-02-01"
MSTU_INCEPTION = pd.Timestamp("2025-06-04")
WARMUP         = 35

# Base stop-loss used for all re-entry criteria tests
SL_TYPE = "trailing"
SL_PCT  = 0.07

# Re-entry criteria: (label, mode, param)
# Modes: "NO_SL" | "none" | "cooldown" | "recovery" | "ma30" | "strong_u1"
#        | "composite" | "d2reset_ma30"
# Param: cooldown bars (int) for cooldown/composite; recovery fraction for
#        recovery; 0 otherwise
RE_CONFIGS = [
    ("Baseline (no SL)",  "NO_SL",        0    ),
    ("RE0: Immediate",    "none",         0    ),
    ("RE1: 3d Cooldown",  "cooldown",     3    ),
    ("RE2: 5d Cooldown",  "cooldown",     5    ),
    ("RE3: Recov +2%",    "recovery",     0.02 ),
    ("RE4: Recov +5%",    "recovery",     0.05 ),
    ("RE5: MA30 Mandatory","ma30",        0    ),
    ("RE6: Strong U1",    "strong_u1",    0    ),
    ("RE7: 3d+MA30",      "composite",    3    ),
    ("RE8: D2Reset+MA30", "d2reset_ma30", 0    ),
]

# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING  (identical to backtest_stop_loss.py)
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

    _cb_url  = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
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
        _cb_df = pd.DataFrame(_cb_rows, columns=["ts","low","high","open","close","volume"])
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
    feat = pd.DataFrame(index=df.index)
    c = df["btc_close"]; h = df["btc_high"]
    l_ = df["btc_low"];  v = df["btc_volume"]
    ret = np.log(c).diff()

    for k in [1,3,5,7,14,30]: feat[f"ret_{k}"] = ret.rolling(k).sum()
    for k in [5,10,20,30]:     feat[f"vol_{k}"] = ret.rolling(k).std()
    prev_c = c.shift(1)
    tr = pd.concat([(h-l_),(h-prev_c).abs(),(l_-prev_c).abs()],axis=1).max(axis=1)
    for k in [7,14,30]: feat[f"atr_{k}"] = tr.rolling(k).mean()/c
    feat["range_today"] = (h-l_)/c
    feat["range_ma7"]   = ((h-l_)/c).rolling(7).mean()
    feat["range_ma30"]  = ((h-l_)/c).rolling(30).mean()
    feat["range_std30"] = ((h-l_)/c).rolling(30).std()
    gain = c.diff().clip(lower=0).rolling(14).mean()
    loss = (-c.diff().clip(upper=0)).rolling(14).mean()
    feat["rsi_14"] = 100 - 100/(1+gain/loss.replace(0,np.nan))
    e12 = c.ewm(span=12,adjust=False).mean(); e26 = c.ewm(span=26,adjust=False).mean()
    macd = e12-e26
    feat["macd"]      = macd/c
    feat["macd_sig"]  = macd.ewm(span=9,adjust=False).mean()/c
    feat["macd_hist"] = (macd-macd.ewm(span=9,adjust=False).mean())/c
    ma20=c.rolling(20).mean(); sd20=c.rolling(20).std()
    feat["bb_width"]   = (4*sd20)/ma20
    feat["dist_hi_30"] = c/c.rolling(30).max()-1
    feat["dist_lo_30"] = c/c.rolling(30).min()-1
    feat["dist_hi_90"] = c/c.rolling(90).max()-1
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
    for nm,col in [("spx","spx_close"),("ndx","ndx_close"),
                   ("gold","gold_close"),("dxy","dxy_close")]:
        if col in df.columns:
            feat[f"btc_{nm}_corr_30"] = ret.rolling(30).corr(np.log(df[col]).diff())
    for col in [x for x in df.columns if x.startswith("oc_")]:
        s=df[col].astype(float); sl=np.log(s.replace(0,np.nan))
        feat[f"{col}_d1"] = sl.diff(1); feat[f"{col}_d7"] = sl.diff(7)
        feat[f"{col}_z30"] = (sl-sl.rolling(30).mean())/sl.rolling(30).std()
    nh,nl = h.shift(-1),l_.shift(-1)
    y_hi = (nh-c)/c; y_lo = (c-nl)/c
    feat["y_hi_ema3"] = y_hi.shift(1).ewm(span=3,adjust=False).mean()
    feat["y_lo_ema3"] = y_lo.shift(1).ewm(span=3,adjust=False).mean()
    feat["y_hi_ema7"] = y_hi.shift(1).ewm(span=7,adjust=False).mean()
    feat["y_lo_ema7"] = y_lo.shift(1).ewm(span=7,adjust=False).mean()
    p3h=h.shift(1).rolling(3).max(); p3l=l_.shift(1).rolling(3).min()
    feat["above_3d_high"] = (c>p3h).astype(float)
    feat["below_3d_low"]  = (c<p3l).astype(float)
    feat["bo_strength_up"] = (c/p3h-1).clip(lower=0)
    feat["bo_strength_dn"] = (1-c/p3l).clip(lower=0)
    yhl=y_hi.shift(1); yll=y_lo.shift(1)
    feat["y_hi_surprise"] = yhl-yhl.ewm(span=7,adjust=False).mean()
    feat["y_lo_surprise"] = yll-yll.ewm(span=7,adjust=False).mean()
    nr=ret.clip(upper=0)
    feat["dn_vol_5"]  = nr.rolling(5).std()
    feat["dn_vol_20"] = nr.rolling(20).std()
    sma50=c.rolling(50).mean()
    feat["below_sma50"]    = (c<sma50).astype(float)
    feat["below_sma50_5d"] = feat["below_sma50"].rolling(5).min().fillna(0)
    if "cb_premium" in df.columns and df["cb_premium"].notna().any():
        feat["cb_premium"]     = df["cb_premium"]
        feat["cb_premium_ma3"] = df["cb_premium_ma3"]
        feat["cb_premium_z7"]  = df["cb_premium_z7"]
    else:
        feat["cb_premium"]     = 0.0
        feat["cb_premium_ma3"] = 0.0
        feat["cb_premium_z7"]  = 0.0

    fc = AD["feat_cols"]
    feat = feat.replace([np.inf,-np.inf], np.nan)
    for col in fc:
        if col not in feat.columns: feat[col] = np.nan
    F = feat[fc].dropna()
    if F.empty:
        raise RuntimeError("Feature matrix is empty — check data coverage")

    if AD.get("ensemble") and AD.get("constituents"):
        yhi_arr = np.mean([con["m_hi"].predict(F) for con in AD["constituents"]], axis=0)
        ylo_arr = np.mean([con["m_lo"].predict(F) for con in AD["constituents"]], axis=0)
        if AD.get("blended") and float(AD.get("alpha",1.0)) < 1.0:
            a = float(AD["alpha"])
            yhi_arr = a*yhi_arr + (1-a)*float(AD.get("mu_hi",0))
            ylo_arr = a*ylo_arr + (1-a)*float(AD.get("mu_lo",0))
    else:
        yhi_arr = AD["hi_model"].predict(F)
        ylo_arr = AD["lo_model"].predict(F)

    cv = c.reindex(F.index).values
    ph = cv*(1+np.clip(yhi_arr,0,None))
    pl = cv*(1-np.clip(ylo_arr,0,None))
    idx = np.asarray(F.index, dtype="datetime64[ns]")
    nd = np.empty(len(F), dtype="datetime64[ns]")
    nd[:-1] = idx[1:]; nd[-1] = idx[-1]+np.timedelta64(1,"D")
    preds = pd.DataFrame(
        {"close_asof":cv,"pred_high":ph,"pred_low":pl},
        index=pd.DatetimeIndex(nd, name="target_date"),
    )
    return preds[~preds.index.duplicated(keep="last")]


def fetch_mstr(end_date: str) -> pd.Series:
    d = _yf_download("MSTR", FETCH_START,
                     (pd.Timestamp(end_date)+pd.Timedelta(days=2)).strftime("%Y-%m-%d"))
    raw = d["Close"].sort_index()
    return raw.reindex(
        pd.date_range(raw.index[0], max(raw.index[-1],pd.Timestamp(end_date)), freq="D")
    ).ffill()


def build_synthetic_mstu(end_date: str) -> pd.Series:
    end_dt = pd.Timestamp(end_date)
    _mstu_end = max(end_dt, MSTU_INCEPTION+pd.Timedelta(days=90))+pd.Timedelta(days=2)
    try:
        d_mstu = _yf_download("MSTU", MSTU_INCEPTION.strftime("%Y-%m-%d"),
                               _mstu_end.strftime("%Y-%m-%d"))
        mstu_actual = d_mstu["Close"].sort_index()
    except Exception:
        return pd.Series(dtype=float)
    try:
        d_mstr = _yf_download("MSTR", FETCH_START, _mstu_end.strftime("%Y-%m-%d"))
        mstr_all = d_mstr["Close"].sort_index()
    except Exception:
        return pd.Series(dtype=float)

    mstr_post = mstr_all[mstr_all.index >= MSTU_INCEPTION]
    common_idx = mstu_actual.index.intersection(mstr_post.index)
    beta, alpha_ols = 2.0, -0.0002
    if len(common_idx) >= 10:
        mstu_lr = np.log(mstu_actual.loc[common_idx]/
                         mstu_actual.loc[common_idx].shift(1)).dropna()
        mstr_lr = np.log(mstr_post.loc[mstu_lr.index]/
                         mstr_post.loc[mstu_lr.index].shift(1)).dropna()
        cidx = mstu_lr.index.intersection(mstr_lr.index)
        if len(cidx) >= 5:
            x=mstr_lr.loc[cidx].values; y=mstu_lr.loc[cidx].values
            xm=x-x.mean(); ym=y-y.mean()
            denom=float(np.dot(xm,xm))
            if denom > 1e-10:
                beta=float(np.dot(xm,ym)/denom)
                alpha_ols=float(y.mean()-beta*x.mean())

    mstr_pre = mstr_all[mstr_all.index < MSTU_INCEPTION].sort_index()
    if len(mstr_pre) < 2:
        mstu_full = mstu_actual.copy()
    else:
        mstr_lr_pre = np.log(mstr_pre/mstr_pre.shift(1)).fillna(0.0).values
        syn_lr = beta*mstr_lr_pre+alpha_ols; syn_lr[0] = 0.0
        inception_price = float(mstu_actual.iloc[0])
        cum_syn = np.cumsum(syn_lr)
        syn_px = inception_price*np.exp(cum_syn-cum_syn[-1])
        syn_series = pd.Series(syn_px, index=mstr_pre.index)
        mstu_full = pd.concat([syn_series, mstu_actual])
        mstu_full = mstu_full[~mstu_full.index.duplicated(keep="last")].sort_index()

    return mstu_full.reindex(
        pd.date_range(mstu_full.index[0], max(mstu_full.index[-1],end_dt), freq="D")
    ).ffill()


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATION  (extended to return raw arrays for re-entry criteria)
# ══════════════════════════════════════════════════════════════════════════════

def build_signals(comp: pd.DataFrame) -> dict:
    """Compute all TF2+V-Gate arrays.  Also returns raw ehma3/hb3 for RE6."""
    N       = len(comp)
    c_asof  = comp["close_asof"].values.astype(float)
    pred_hi = comp["pred_high"].values.astype(float)
    pred_lo = comp["pred_low"].values.astype(float)
    act_hi  = comp["actual_high"].values.astype(float)
    act_lo  = comp["actual_low"].values.astype(float)

    err_hi = (act_hi-pred_hi)/c_asof*100
    err_lo = (pred_lo-act_lo)/c_asof*100
    hi_brk = (act_hi>pred_hi).astype(int)
    lo_brk = (act_lo<pred_lo).astype(int)

    ehma3 = np.zeros(N); elma3 = np.zeros(N)
    hb3   = np.zeros(N, dtype=int); lb3 = np.zeros(N, dtype=int)
    for i in range(N):
        s=max(0,i-2)
        ehma3[i]=np.mean(err_hi[s:i+1]); elma3[i]=np.mean(err_lo[s:i+1])
        hb3[i]=int(np.sum(hi_brk[s:i+1])); lb3[i]=int(np.sum(lo_brk[s:i+1]))

    u1 = (ehma3>0.7) & (hb3>=2)
    d1 = (lb3>=2)    & (elma3>0.5)
    d2 = ehma3 < -0.75
    d3 = np.zeros(N, dtype=bool)
    for i in range(1,N):
        consec = 0
        for k in range(i-1,-1,-1):
            if hi_brk[k]: consec += 1
            else: break
        if consec >= 3 and lo_brk[i]:
            d3[i] = True

    ma30 = np.full(N, np.nan)
    for i in range(N):
        w=min(30,i+1)
        ma30[i]=np.mean(c_asof[max(0,i-w+1):i+1])
    above_ma30     = c_asof > ma30
    ma30_slope_pos = np.zeros(N, dtype=bool)
    for i in range(N):
        if i>=5 and np.isfinite(ma30[i]) and np.isfinite(ma30[i-5]):
            ma30_slope_pos[i] = ma30[i] > ma30[i-5]
    bull_regime = above_ma30 & ma30_slope_pos

    clean_10d = np.zeros(N, dtype=bool)
    for i in range(N):
        lo_i=max(0,i-7)
        clean_10d[i] = not bool(np.any(d1[lo_i:i] | d2[lo_i:i]))

    _DN_NORM_W    = 30
    roll_ehi_norm = np.array([
        float(np.mean(err_hi[max(0,i-_DN_NORM_W+1):i+1])) for i in range(N)
    ])
    dn_score_arr = np.zeros(N)
    for i in range(N):
        norm=max(abs(roll_ehi_norm[i]),0.01)
        dn_score_arr[i] = (
            (-ehma3[i]/norm)                           * 0.30 +
            (lb3[i]/3.0)                               * 0.30 +
            (elma3[i]/max(abs(elma3[i]),0.10))         * 0.20 +
            float(lo_brk[i])                           * 0.20
        )
    v_rev_bar = (dn_score_arr>0.8) & (err_lo>3.0)
    v_recent  = np.zeros(N, dtype=bool)
    for i in range(N):
        v_recent[i] = bool(np.any(v_rev_bar[max(0,i-2):i+1]))

    entry = u1 & (above_ma30 | clean_10d | v_recent)

    return dict(
        N=N, d1=d1, d2=d2, d3=d3, bull_regime=bull_regime,
        entry=entry, above_ma30=above_ma30, clean_10d=clean_10d, v_recent=v_recent,
        # Raw arrays for RE6 (strong U1) criterion
        ehma3=ehma3, hb3=hb3,
    )


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE WITH RE-ENTRY CRITERIA
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest_reentry(
    comp: pd.DataFrame,
    asset_px: np.ndarray,
    dates: pd.DatetimeIndex,
    _bt0: int,
    signals: dict,
    sl_type: str   = "trailing",
    sl_pct: float  = 0.07,
    reentry_mode: str  = "none",
    reentry_param: float = 0,
    initial_capital: float = INITIAL_CAPITAL,
) -> dict:
    """
    Backtest TF2+V-Gate with Trail-7% stop-loss and a pluggable re-entry filter.

    The re-entry filter is ONLY active after a stop-loss exit.  Normal D2/D3
    signal exits return to the standard U1 re-entry logic immediately.

    reentry_mode options:
      "NO_SL"        — no stop-loss at all (baseline reference)
      "none"         — SL applied, immediate re-entry after SL (current behaviour)
      "cooldown"     — block re-entry for `reentry_param` bars after SL
      "recovery"     — re-enter only when price > sl_exit_price × (1+reentry_param)
      "ma30"         — re-enter after SL only when BTC close is above MA30
      "strong_u1"    — re-enter after SL only when ehma3>1.2 AND hi_breaks>=3
      "composite"    — cooldown(`reentry_param` bars) AND above MA30
      "d2reset_ma30" — after SL, wait until D2 fires (clears bull momentum),
                       then re-enter on next U1 above MA30
    """
    N          = signals["N"]
    d1         = signals["d1"]
    d2         = signals["d2"]
    d3         = signals["d3"]
    bull       = signals["bull_regime"]
    entry      = signals["entry"]
    above_ma30 = signals["above_ma30"]
    clean_10d  = signals["clean_10d"]
    v_recent   = signals["v_recent"]
    ehma3      = signals["ehma3"]
    hb3        = signals["hb3"]

    nav      = initial_capital
    pos      = "CASH"
    qty      = 0.0
    e_price  = e_nav = e_date = e_trigger = None
    peak_px  = 0.0
    trades   = []
    nav_arr  = np.full(N, np.nan)

    # Re-entry state (only active after SL exits)
    sl_active          = False   # currently in post-SL re-entry filter mode
    sl_exit_bar        = -999    # bar index of last SL exit
    sl_exit_price      = 0.0    # asset price at last SL exit
    d2_fired_since_sl  = False   # D2 signal fired since last SL exit
    n_reentry_blocked  = 0       # count of U1 signals blocked by re-entry filter

    for i in range(N):
        si    = i - 1
        price = asset_px[i]

        if i < _bt0:
            nav_arr[i] = initial_capital
            continue

        if not np.isfinite(price) or price <= 0:
            nav_arr[i] = qty*(asset_px[i-1] if i>0 else price) if pos=="LONG" else nav
            continue

        if pos == "LONG":
            if sl_type == "trailing":
                peak_px = max(peak_px, price)

            # Evaluate stop-loss
            stop_hit = False; exit_lbl = ""
            if reentry_mode != "NO_SL" and sl_type == "fixed" and sl_pct > 0:
                if price < e_price*(1.0-sl_pct):
                    stop_hit = True; exit_lbl = f"SL-fixed-{sl_pct*100:.0f}%"
            elif reentry_mode != "NO_SL" and sl_type == "trailing" and sl_pct > 0:
                if price < peak_px*(1.0-sl_pct):
                    stop_hit = True; exit_lbl = f"SL-trail-{sl_pct*100:.0f}%"

            # Evaluate signal exit (1-bar lag)
            sig_exit = False
            if si >= 0:
                sig_exit = bool(d3[si] or (d2[si] and not bull[si]))
                sig_lbl  = "D3" if (si>=0 and d3[si]) else "D2(bear)"

            should_exit = stop_hit or sig_exit
            if should_exit:
                if not exit_lbl: exit_lbl = sig_lbl
                nav = qty * price
                trades.append(dict(
                    entry_date     = e_date,
                    entry_price    = e_price,
                    entry_trigger  = e_trigger,
                    entry_nav      = e_nav,
                    exit_date      = dates[i],
                    exit_price     = price,
                    exit_nav       = nav,
                    exit_signal    = exit_lbl,
                    pnl_pct        = (price/e_price - 1)*100,
                    pnl_abs        = nav - e_nav,
                    duration_days  = (dates[i]-e_date).days,
                    peak_price     = peak_px,
                    stop_triggered = stop_hit,
                ))
                pos = "CASH"; qty = 0.0
                e_price = e_nav = e_date = e_trigger = None

                if stop_hit:
                    sl_active         = True
                    sl_exit_bar       = i
                    sl_exit_price     = price
                    d2_fired_since_sl = False

        else:  # CASH
            # Track D2 firing after SL exit (used by d2reset_ma30)
            if sl_active and si >= 0 and d2[si]:
                d2_fired_since_sl = True

            if si >= 0 and entry[si]:
                # Determine whether the re-entry filter allows entry here
                can_enter = True

                if sl_active:
                    if reentry_mode == "cooldown":
                        cooldown = int(reentry_param)
                        can_enter = (i >= sl_exit_bar + cooldown + 1)

                    elif reentry_mode == "recovery":
                        can_enter = (price > sl_exit_price * (1.0 + float(reentry_param)))

                    elif reentry_mode == "ma30":
                        # MA30 mandatory (not just one of three gates)
                        can_enter = bool(above_ma30[si])

                    elif reentry_mode == "strong_u1":
                        # Stricter momentum: err_hi_ma3 > 1.2% AND 3 hi-breaks in 3d
                        can_enter = bool(ehma3[si] > 1.2 and hb3[si] >= 3)

                    elif reentry_mode == "composite":
                        cooldown = int(reentry_param)
                        can_enter = (
                            (i >= sl_exit_bar + cooldown + 1)
                            and bool(above_ma30[si])
                        )

                    elif reentry_mode == "d2reset_ma30":
                        # Wait until D2 fires post-SL (signals momentum has reset),
                        # then require above MA30 for re-entry
                        can_enter = d2_fired_since_sl and bool(above_ma30[si])

                    if not can_enter:
                        n_reentry_blocked += 1
                    else:
                        sl_active = False   # filter satisfied; exit SL mode

                if can_enter:
                    qty     = nav / price
                    e_price = price; e_date = dates[i]
                    e_nav   = nav;   peak_px = price
                    pos     = "LONG"
                    if v_recent[si] and not above_ma30[si] and not clean_10d[si]:
                        e_trigger = "U1+V-rev"
                    elif above_ma30[si] and clean_10d[si]:
                        e_trigger = "U1+↑MA30+c10d"
                    elif above_ma30[si]:
                        e_trigger = "U1+↑MA30"
                    else:
                        e_trigger = "U1+c10d"

        nav_arr[i] = qty*price if pos=="LONG" else nav

    if pos=="LONG" and np.isfinite(asset_px[N-1]) and asset_px[N-1] > 0:
        nav_arr[N-1] = qty*asset_px[N-1]

    nav_series = pd.Series(nav_arr[_bt0:], index=dates[_bt0:]).ffill()
    bh_series  = pd.Series(
        initial_capital*asset_px[_bt0:]/asset_px[_bt0], index=dates[_bt0:]
    )
    return dict(
        nav=nav_series, bh=bh_series, trades=trades,
        open_pos=(pos=="LONG"), open_entry_price=e_price, open_entry_date=e_date,
        n_reentry_blocked=n_reentry_blocked,
    )


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(res: dict, ic: float = INITIAL_CAPITAL) -> dict:
    nav    = res["nav"]; bh = res["bh"]; trades = res["trades"]
    final  = float(nav.iloc[-1])
    bh_fin = float(bh.iloc[-1])
    ret    = (final/ic - 1)*100
    bh_ret = (bh_fin/ic - 1)*100
    n_yr   = (nav.index[-1]-nav.index[0]).days/365.25
    cagr   = ((final/ic)**(1/n_yr)-1)*100 if n_yr > 0 else 0.0
    rm     = nav.cummax()
    max_dd = float(((nav-rm)/rm*100).min())
    dr     = nav.pct_change().fillna(0)
    rf_d   = (1.045)**(1/252)-1
    exc    = dr-rf_d
    sharpe = float(exc.mean()/exc.std()*np.sqrt(252)) if exc.std() > 0 else 0.0
    wins   = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    wr     = 100*len(wins)/len(trades) if trades else 0.0
    best   = max([t["pnl_pct"] for t in trades]) if trades else 0.0
    worst  = min([t["pnl_pct"] for t in trades]) if trades else 0.0
    avg_pnl= float(np.mean([t["pnl_pct"] for t in trades])) if trades else 0.0
    gp     = sum(t["pnl_abs"] for t in wins) if wins else 0.0
    gl     = abs(sum(t["pnl_abs"] for t in losses)) if losses else 1.0
    pf     = gp/gl if gl > 0 else float("inf")
    days_in= sum(t["duration_days"] for t in trades)
    tot_d  = max(1,(nav.index[-1]-nav.index[0]).days)
    n_sl   = sum(1 for t in trades if t.get("stop_triggered",False))
    n_blocked = res.get("n_reentry_blocked", 0)
    return dict(
        ret=ret, bh_ret=bh_ret, cagr=cagr,
        max_dd=max_dd, sharpe=sharpe,
        n_trades=len(trades), win_rate=wr, avg_pnl=avg_pnl,
        best=best, worst=worst, profit_factor=pf,
        time_in=100*days_in/tot_d,
        n_sl_exits=n_sl,
        n_reentry_blocked=n_blocked,
    )


# ══════════════════════════════════════════════════════════════════════════════
# REPORTING
# ══════════════════════════════════════════════════════════════════════════════

RE_LABELS = [cfg[0] for cfg in RE_CONFIGS]
COL_W = 18


def print_period_table(period_label: str, asset: str,
                       results_list: list, bh_ret: float):
    """Print a comparison table for one period × one asset across all re-entry configs."""
    metrics_list = [compute_metrics(r) for r in results_list]

    hdr = f"  Asset: {asset}   Period: {period_label}"
    print(f"\n{hdr}")
    print(f"  B&H return for period: {bh_ret:+.1f}%   "
          f"(Trail-7% SL used for RE0–RE8; no SL for Baseline)")
    sep = "  " + "─"*(32 + COL_W*len(RE_LABELS))
    print(sep)
    # Column headers (wrap at 17 chars)
    short_labels = [
        "Baseline\n(no SL)",
        "RE0\nImmediate",
        "RE1\n3d Cooldown",
        "RE2\n5d Cooldown",
        "RE3\nRecov+2%",
        "RE4\nRecov+5%",
        "RE5\nMA30 Mand.",
        "RE6\nStrong U1",
        "RE7\n3d+MA30",
        "RE8\nD2+MA30",
    ]
    # Print two-line header
    top    = f"  {'Metric':<32}" + "".join(f"{s.split(chr(10))[0]:>{COL_W}}" for s in short_labels)
    bottom = f"  {'':32}"        + "".join(f"{s.split(chr(10))[1]:>{COL_W}}" for s in short_labels)
    print(top); print(bottom); print(sep)

    base = metrics_list[0]   # Baseline (no SL) is the first config

    def row(label: str, key: str, fmt: str, higher_better: bool = True):
        vals = [m[key] for m in metrics_list]
        cells = []
        for j, v in enumerate(vals):
            s = f"{v:{fmt}}"
            if j == 0:
                cells.append(f"{s:>{COL_W}}")
            else:
                delta = v - vals[0]
                if abs(delta) < 0.05:
                    marker = " ="
                elif (delta > 0) == higher_better:
                    marker = " ▲"
                else:
                    marker = " ▼"
                cells.append(f"{s:>{COL_W-2}}{marker}")
        print(f"  {label:<32}" + "".join(cells))

    row("Total return (%)",        "ret",                "+.1f")
    row("CAGR (%)",                "cagr",               "+.1f")
    row("Max drawdown (%)",        "max_dd",             "+.1f", higher_better=True)
    row("Sharpe ratio",            "sharpe",             ".2f")
    row("Win rate (%)",            "win_rate",           ".1f")
    row("Profit factor",           "profit_factor",      ".2f")
    row("Avg trade P&L (%)",       "avg_pnl",            "+.2f")
    row("Best trade (%)",          "best",               "+.1f")
    row("Worst trade (%)",         "worst",              "+.1f", higher_better=True)
    row("# Trades",                "n_trades",           "d",    higher_better=False)
    row("# SL exits",              "n_sl_exits",         "d",    higher_better=False)
    row("# Re-entries blocked",    "n_reentry_blocked",  "d",    higher_better=False)
    row("Time in market (%)",      "time_in",            ".1f",  higher_better=False)
    print(sep)


def print_asset_summary(asset: str, all_period_metrics: dict):
    """Print Return | Max-DD | Sharpe summary across all periods for one asset."""
    print(f"\n{'═'*120}")
    print(f"  SUMMARY — {asset}  |  ▲=better vs Baseline(no-SL)  ▼=worse")
    print(f"{'═'*120}")
    hdr_line = f"  {'Period / Metric':<40}" + "".join(f"{c:>{COL_W}}" for c in RE_LABELS)
    print(hdr_line)
    print(f"  {'─'*40}" + "─"*COL_W*len(RE_LABELS))

    for period, mlist in all_period_metrics.items():
        short_p = period[:36]
        for key, label, fmt, hb in [
            ("ret",    "  Return (%)",    "+.1f", True),
            ("max_dd", "  Max DD (%)",    "+.1f", True),
            ("sharpe", "  Sharpe",        ".2f",  True),
        ]:
            vals = [m[key] for m in mlist]
            cells = []
            for j, v in enumerate(vals):
                s = f"{v:{fmt}}"
                if j == 0:
                    cells.append(f"{s:>{COL_W}}")
                else:
                    delta = v - vals[0]
                    if abs(delta) < 0.05:
                        marker = "="
                    elif (delta > 0) == hb:
                        marker = "▲"
                    else:
                        marker = "▼"
                    cells.append(f"{s:>{COL_W-1}}{marker}")
            period_prefix = f"  {short_p:<38}" if key=="ret" else f"  {'':38}"
            print(f"{period_prefix} {label:<1}" + "".join(cells))
        print()


def print_cross_asset_recommendation(all_asset_results: dict):
    """Print a concise recommendation table comparing best criteria per asset."""
    print("\n" + "═"*120)
    print("  CROSS-ASSET RECOMMENDATION SUMMARY")
    print("  Scoring: for each (asset × period), award 1 point if criterion beats RE0 on Return AND Sharpe.")
    print("  The criterion with the most points across all periods for a given asset is recommended.")
    print("═"*120)

    re_labels = RE_LABELS[1:]   # skip Baseline (no-SL); compare RE0–RE8

    for asset, per_asset in all_asset_results.items():
        scores = {lbl: 0 for lbl in re_labels}
        n_periods = 0

        for period, mlist in per_asset.items():
            if not mlist: continue
            n_periods += 1
            base_m  = mlist[0]  # Baseline (no SL)
            re0_m   = mlist[1]  # RE0 (immediate, current behaviour)

            for j, lbl in enumerate(re_labels):
                m = mlist[j+1]  # j+1 because index 0 is Baseline
                # Award point if this criterion beats RE0 on return AND sharpe
                if m["ret"] > re0_m["ret"] and m["sharpe"] > re0_m["sharpe"]:
                    scores[lbl] += 1

        best_crit = max(scores, key=lambda k: scores[k])
        best_score = scores[best_crit]

        print(f"\n  {asset}  ({n_periods} periods scored)")
        print(f"  {'Criterion':<22}" + "  ".join(f"{lbl[:16]:>16}" for lbl in re_labels))
        print(f"  {'Score / ' + str(n_periods):<22}" +
              "  ".join(f"{scores[lbl]:>16d}" for lbl in re_labels))
        if best_score > 0:
            print(f"  ► Best: {best_crit}  (beats RE0 in {best_score}/{n_periods} periods on Return+Sharpe)")
        else:
            print(f"  ► No criterion consistently beats RE0 across periods for {asset}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═"*80)
    print("  BTC RANGE MODEL — Re-Entry Criteria Evaluation (post Stop-Loss)")
    print("═"*80)
    print(f"  Base SL: Trail-{SL_PCT*100:.0f}% (trailing stop)")
    print(f"  Testing {len(RE_CONFIGS)} re-entry criteria × "
          f"{len(PERIODS)} periods × 3 assets\n")

    # ── Load model ─────────────────────────────────────────────────────────────
    model_path = _ROOT/"models"/"inference_assets_ct.joblib"
    print("Loading CT model …", end=" ", flush=True)
    AD = joblib.load(str(model_path))
    cm = AD.get("calibration_meta", {})
    oos_start = cm.get("test_start", "2026-03-01")
    print(f"OK  (trained through {cm.get('train_end','?')}, test_start={oos_start})")
    PERIODS["OOS Only   (Mar 2026–Jun 2026)"] = (oos_start, "2026-06-07")

    # ── Fetch data ──────────────────────────────────────────────────────────────
    print("\n[1/4] Fetching BTC + macro + on-chain data …")
    df = fetch_btc_and_macro("2026-06-09")

    print("\n[2/4] Building CT predictions …", end=" ", flush=True)
    preds = build_features_and_predictions(df, AD)
    print(f"{len(preds)} bars")

    print("\n[3/4] Fetching MSTR prices …", end=" ", flush=True)
    mstr_all = fetch_mstr("2026-06-09")
    print(f"{len(mstr_all)} bars")

    print("\n[4/4] Building synthetic MSTU prices …", end=" ", flush=True)
    mstu_all = build_synthetic_mstu("2026-06-09")
    print(f"{len(mstu_all)} bars")

    # ══════════════════════════════════════════════════════════════════════════
    # Run backtests
    # ══════════════════════════════════════════════════════════════════════════
    asset_defs = {
        "BTC":  None,
        "MSTR": mstr_all,
        "MSTU": mstu_all,
    }

    all_results: dict = {}  # asset → period → list[result_dict]

    for asset_name, asset_price_series in asset_defs.items():
        print(f"\n{'─'*80}")
        print(f"  Running backtests for {asset_name}")
        print(f"{'─'*80}")
        per_asset: dict = {}

        for period_label, (p_start, p_end) in PERIODS.items():
            start_dt = pd.Timestamp(p_start)
            end_dt   = pd.Timestamp(p_end)
            pre_dt   = start_dt - pd.Timedelta(days=60)

            p_slice = preds.loc[
                (preds.index >= pre_dt) & (preds.index <= end_dt)
            ].copy()
            if len(p_slice) < WARMUP+3:
                print(f"  SKIP {period_label}: insufficient prediction rows")
                continue

            p_slice["actual_high"]  = df["btc_high"].reindex(p_slice.index).values
            p_slice["actual_low"]   = df["btc_low"].reindex(p_slice.index).values
            p_slice["actual_close"] = df["btc_close"].reindex(p_slice.index).values
            comp = p_slice.dropna(
                subset=["actual_high","actual_low","actual_close"]
            ).reset_index()
            N = len(comp)
            if N < WARMUP+3: continue

            dates = pd.DatetimeIndex(comp["target_date"])
            _bt0  = max(WARMUP, int(dates.searchsorted(start_dt)))
            if N-_bt0 < 3: continue

            btc_px_arr = comp["actual_close"].values.astype(float)
            ax = (
                asset_price_series.reindex(dates).ffill().bfill().values.astype(float)
                if asset_price_series is not None else btc_px_arr
            )

            signals = build_signals(comp)

            period_results = []
            for re_label, re_mode, re_param in RE_CONFIGS:
                if re_mode == "NO_SL":
                    res = run_backtest_reentry(
                        comp, ax, dates, _bt0, signals,
                        sl_type="none", sl_pct=0.0,
                        reentry_mode="NO_SL",
                    )
                else:
                    res = run_backtest_reentry(
                        comp, ax, dates, _bt0, signals,
                        sl_type=SL_TYPE, sl_pct=SL_PCT,
                        reentry_mode=re_mode, reentry_param=re_param,
                    )
                period_results.append(res)

            bh_ret = (ax[N-1]/ax[_bt0]-1)*100 if ax[_bt0] > 0 else 0.0
            per_asset[period_label] = period_results
            baseline_ret = compute_metrics(period_results[0])["ret"]
            re0_ret      = compute_metrics(period_results[1])["ret"]
            print(f"  {period_label}  ({N-_bt0} bars)  "
                  f"B&H {bh_ret:+.1f}%  "
                  f"Baseline(no SL) {baseline_ret:+.1f}%  "
                  f"RE0(immediate) {re0_ret:+.1f}%")

        all_results[asset_name] = per_asset

    # ══════════════════════════════════════════════════════════════════════════
    # Print detailed tables
    # ══════════════════════════════════════════════════════════════════════════
    print("\n\n" + "═"*120)
    print("  RE-ENTRY CRITERIA EVALUATION RESULTS")
    print("  ▲ = improvement vs Baseline(no-SL)   ▼ = deterioration vs Baseline(no-SL)")
    print("═"*120)

    for asset_name, per_asset in all_results.items():
        print(f"\n\n{'█'*120}")
        print(f"  ASSET: {asset_name}")
        print(f"{'█'*120}")
        summary_data: dict = {}

        for period_label, period_results in per_asset.items():
            ax_bh = compute_metrics(period_results[0])["bh_ret"]
            print_period_table(period_label, asset_name, period_results, ax_bh)
            summary_data[period_label] = [compute_metrics(r) for r in period_results]

        print_asset_summary(asset_name, summary_data)

    # ══════════════════════════════════════════════════════════════════════════
    # Cross-asset recommendation
    # ══════════════════════════════════════════════════════════════════════════
    all_period_metrics_by_asset = {
        asset: {
            period: [compute_metrics(r) for r in results]
            for period, results in per_asset.items()
        }
        for asset, per_asset in all_results.items()
    }
    print_cross_asset_recommendation(all_period_metrics_by_asset)

    # ══════════════════════════════════════════════════════════════════════════
    # Interpretation guide
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "═"*120)
    print("  INTERPRETATION GUIDE")
    print("═"*120)
    print("""
  What each criterion does and when it helps:
  ────────────────────────────────────────────
  RE0  Immediate       — Re-enters on the very next U1 after SL. Can whipsaw in
                         bear trends where the SL fired for good reason.

  RE1  3d Cooldown     — Simple time gate: wait 3 bars. Misses fast V-recoveries
                         but avoids getting immediately re-stopped in trending
                         bear markets.

  RE2  5d Cooldown     — More conservative; better in sustained bear trends,
                         worse in fast recoveries.

  RE3  Recovery +2%    — Asset must recover 2% above the SL exit price. Low bar
                         that filters only the most immediate whipsaws. Works
                         because a 7% trailing SL exit means price is already
                         ~7% off the peak; another 2% up confirms a real bounce.

  RE4  Recovery +5%    — Higher bar: ~12% off peak before re-entry allowed.
                         Meaningful in BTC bear markets; may be too conservative
                         for MSTR/MSTU (leveraged, moves faster).

  RE5  MA30 Mandatory  — The U1 signal already accepts (above_ma30 OR clean_10d
                         OR v_reversal). This tightens it post-SL: MUST be above
                         MA30. Ensures regime has flipped back to bull. Most
                         effective when SL fires during a genuine bear turn.

  RE6  Strong U1       — Requires err_hi_ma3 > 1.2% (vs 0.7%) AND hi_breaks >= 3
                         (vs 2). Waits for a more convincing momentum reset. Can
                         miss moderate but genuine recoveries.

  RE7  3d+MA30         — Combined: time buffer AND regime confirmation. Typically
                         the most balanced for bear markets without sacrificing
                         too much bull-market upside.

  RE8  D2Reset+MA30    — The most patient: waits for D2 (momentum cleared) to
                         fire at least once after the SL, then requires MA30.
                         Best suited to MSTR/MSTU where stop-losses often fire
                         in the middle of a correction that needs to fully
                         exhaust before reversing.

  Statistical note:
  ─────────────────
  With 4 backtest periods and typically 5–20 trades per period, no single
  criterion achieves classical statistical significance in isolation. The
  scoring metric (beats RE0 on both Return AND Sharpe across N periods) is
  a multi-period consistency check, not a p-value. Consistent improvement
  across BOTH bull and bear periods is the strongest signal.
""")
    print("✓ Done.")


if __name__ == "__main__":
    main()
