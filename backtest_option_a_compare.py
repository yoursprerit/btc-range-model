#!/usr/bin/env python3
"""
Option A Entry Filter — Comparison vs Current Implementation
============================================================
Option A: require ma30_slope for the clean_7d path only.

  Current:  tf2_entry = u1 & ((above_ma30 ^ clean_7d) | v_recent)
  Option A: tf2_entry = u1 & ((above_ma30 ^ (clean_7d & ma30_slope)) | v_recent)

Runs SL5 (regime-adaptive, UI-recommended variant for MSTR/MSTU) for both entry
conditions across all three periods and prints a side-by-side comparison.

Assets: MSTR (fixed −3% stop) and MSTU (fixed −7% stop).
"""

import sys, warnings, requests, time as _time
warnings.filterwarnings("ignore")
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

# ── sklearn _loss fix (same workaround as the Streamlit app) ─────────────────
try:
    import sklearn._loss._loss as _sklearn_loss_ext
    if "_loss" not in sys.modules:
        sys.modules["_loss"] = _sklearn_loss_ext
    import sklearn._loss.loss   # noqa: F401
    import sklearn._loss.link   # noqa: F401
except Exception:
    pass

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
ASSET_CONFIG = {
    "MSTR": {"ticker": "MSTR",  "sl_type": "fixed", "sl_pct": 0.03, "name": "MicroStrategy"},
    "MSTU": {"ticker": "MSTU",  "sl_type": "fixed", "sl_pct": 0.07, "name": "T-Rex 2× Long MSTR"},
}
PERIODS = {
    "Bull (Jun24→Jun25)": ("2024-06-05", "2025-06-14", "2024-03-01"),
    "Bear (Jun25→May26)": ("2025-06-01", "2026-05-31", "2025-03-01"),
    "Full (Jun24→May26)": ("2024-06-01", "2026-05-31", "2024-03-01"),
}
INITIAL_CAPITAL = 100_000.0
WARMUP = 35

_MODEL_PATH = _ROOT / "models" / "inference_assets_ct.joblib"
if not _MODEL_PATH.exists():
    _MODEL_PATH = _ROOT / "artifacts" / "artifacts.pkl"

print(f"Loading CT model from: {_MODEL_PATH}")
AD = joblib.load(str(_MODEL_PATH))
print(f"  keys: {list(AD.keys())[:6]}\n")

# ─────────────────────────────────────────────────────────────────────────────
# Data helpers (trimmed from backtest_stop_loss_reentry.py)
# ─────────────────────────────────────────────────────────────────────────────
_MACRO_SYMS = {"eth":"ETH-USD","spx":"^GSPC","ndx":"^IXIC",
               "vix":"^VIX","gold":"GC=F","dxy":"DX-Y.NYB","tnx":"^TNX"}
_ONCHAIN = ["hash-rate","difficulty","n-transactions","miners-revenue",
            "n-unique-addresses","transaction-fees-usd","mempool-size",
            "estimated-transaction-volume-usd","market-cap",
            "avg-block-size","cost-per-transaction"]

def _yf_dl(ticker, start, end):
    d = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] for c in d.columns]
    d.index = pd.DatetimeIndex(d.index).tz_localize(None).normalize()
    return d

def fetch_data(fetch_start, fetch_end):
    print(f"  Fetching BTC+macro {fetch_start} → {fetch_end} …")
    d = _yf_dl("BTC-USD", fetch_start, fetch_end)
    frames = {"btc_close": d["Close"], "btc_high": d["High"],
              "btc_low": d["Low"],     "btc_volume": d["Volume"]}
    for nm, sym in _MACRO_SYMS.items():
        try:
            frames[f"{nm}_close"] = _yf_dl(sym, fetch_start, fetch_end)["Close"]
        except Exception:
            pass
    df = pd.DataFrame(frames).sort_index().ffill(limit=5)
    df.index.name = "date"
    ok = 0
    for m in _ONCHAIN:
        try:
            r = requests.get(f"https://api.blockchain.info/charts/{m}",
                             params={"timespan":"3years","format":"json","sampled":"true"},
                             timeout=20)
            vals = r.json().get("values", [])
            s = pd.Series(
                {pd.Timestamp(v["x"], unit="s").normalize(): v["y"] for v in vals},
                name=f"oc_{m.replace('-','_')}", dtype=float)
            s = s[~s.index.duplicated(keep="last")].sort_index()
            s.index = pd.DatetimeIndex(s.index).tz_localize(None)
            df[s.name] = s.reindex(df.index).ffill(limit=7)
            ok += 1
        except Exception:
            pass
    print(f"    On-chain: {ok}/{len(_ONCHAIN)} OK")
    return df

def fetch_asset_prices(ticker, fetch_start, fetch_end):
    try:
        d = _yf_dl(ticker, fetch_start, fetch_end)
        px = d["Close"].sort_index()
        all_days = pd.date_range(px.index[0], max(px.index[-1], pd.Timestamp(fetch_end)), freq="D")
        return px.reindex(all_days).ffill()
    except Exception as e:
        print(f"    [WARN] {ticker}: {e}")
        return pd.Series(dtype=float)

def build_ct_preds(df, fetch_start="2020-01-01", fetch_end="2030-01-01"):
    c, h, l_, v = df["btc_close"], df["btc_high"], df["btc_low"], df["btc_volume"]
    ret = np.log(c).diff()
    f = pd.DataFrame(index=df.index)
    for k in [1,3,5,7,14,30]: f[f"ret_{k}"] = ret.rolling(k).sum()
    for k in [5,10,20,30]:     f[f"vol_{k}"] = ret.rolling(k).std()
    prev_c = c.shift(1)
    tr = pd.concat([(h-l_),(h-prev_c).abs(),(l_-prev_c).abs()],axis=1).max(axis=1)
    for k in [7,14,30]: f[f"atr_{k}"] = tr.rolling(k).mean()/c
    f["range_today"] = (h-l_)/c; f["range_ma7"] = ((h-l_)/c).rolling(7).mean()
    f["range_ma30"] = ((h-l_)/c).rolling(30).mean()
    f["range_std30"] = ((h-l_)/c).rolling(30).std()
    g = c.diff().clip(lower=0).rolling(14).mean()
    ls = (-c.diff().clip(upper=0)).rolling(14).mean()
    f["rsi_14"] = 100 - 100/(1+g/ls.replace(0,np.nan))
    e12 = c.ewm(span=12,adjust=False).mean(); e26 = c.ewm(span=26,adjust=False).mean()
    macd = e12-e26
    f["macd"] = macd/c; f["macd_sig"] = macd.ewm(span=9,adjust=False).mean()/c
    f["macd_hist"] = (macd-macd.ewm(span=9,adjust=False).mean())/c
    ma20=c.rolling(20).mean(); sd20=c.rolling(20).std()
    f["bb_width"] = (4*sd20)/ma20
    f["dist_hi_30"] = c/c.rolling(30).max()-1; f["dist_lo_30"] = c/c.rolling(30).min()-1
    f["dist_hi_90"] = c/c.rolling(90).max()-1
    f["vol_chg_1"] = np.log(v).diff()
    f["vol_z_20"] = (np.log(v)-np.log(v).rolling(20).mean())/np.log(v).rolling(20).std()
    f["vol_ma_ratio"] = v/v.rolling(20).mean()
    dow = df.index.dayofweek
    for i in range(6): f[f"dow_{i}"] = (dow==i).astype(float)
    for nm in ["spx","ndx","vix","gold","dxy","tnx","eth"]:
        col = f"{nm}_close"
        if col not in df.columns: continue
        s = df[col]; lr = np.log(s).diff()
        for k in [1,5,20]: f[f"{nm}_ret_{k}"] = lr.rolling(k).sum()
        f[f"{nm}_vol_20"] = lr.rolling(20).std()
    for corr_nm, corr_col in [("spx","spx_close"),("ndx","ndx_close"),
                               ("gold","gold_close"),("dxy","dxy_close")]:
        if corr_col in df.columns:
            f[f"btc_{corr_nm}_corr_30"] = ret.rolling(30).corr(np.log(df[corr_col]).diff())
    for col in [x for x in df.columns if x.startswith("oc_")]:
        s = df[col].astype(float); sl = np.log(s.replace(0,np.nan))
        f[f"{col}_d1"] = sl.diff(1); f[f"{col}_d7"] = sl.diff(7)
        f[f"{col}_z30"] = (sl-sl.rolling(30).mean())/sl.rolling(30).std()
    nh=h.shift(-1); nl_=l_.shift(-1)
    y_hi=(nh-c)/c; y_lo=(c-nl_)/c
    f["y_hi_ema3"]=y_hi.shift(1).ewm(span=3,adjust=False).mean()
    f["y_lo_ema3"]=y_lo.shift(1).ewm(span=3,adjust=False).mean()
    f["y_hi_ema7"]=y_hi.shift(1).ewm(span=7,adjust=False).mean()
    f["y_lo_ema7"]=y_lo.shift(1).ewm(span=7,adjust=False).mean()
    p3h=h.shift(1).rolling(3).max(); p3l=l_.shift(1).rolling(3).min()
    f["above_3d_high"]=(c>p3h).astype(float); f["below_3d_low"]=(c<p3l).astype(float)
    f["bo_strength_up"]=(c/p3h-1).clip(lower=0); f["bo_strength_dn"]=(1-c/p3l).clip(lower=0)
    ya=y_hi.shift(1); yb=y_lo.shift(1)
    f["y_hi_surprise"]=ya-ya.ewm(span=7,adjust=False).mean()
    f["y_lo_surprise"]=yb-yb.ewm(span=7,adjust=False).mean()
    neg_ret=ret.clip(upper=0)
    f["dn_vol_5"]=neg_ret.rolling(5).std(); f["dn_vol_20"]=neg_ret.rolling(20).std()
    sma50=c.rolling(50).mean()
    f["below_sma50"]=(c<sma50).astype(float)
    f["below_sma50_5d"]=f["below_sma50"].rolling(5).min().fillna(0)
    # Coinbase premium (skip for speed — set to 0)
    f["cb_premium"] = f["cb_premium_ma3"] = f["cb_premium_z7"] = 0.0
    fc = AD["feat_cols"]
    f = f.replace([np.inf,-np.inf],np.nan)
    for col in fc:
        if col not in f.columns: f[col] = np.nan
    F = f[fc].dropna()
    if F.empty:
        raise RuntimeError("Feature matrix empty — check data")
    if AD.get("ensemble") and AD.get("constituents"):
        yhi = np.mean([con["m_hi"].predict(F) for con in AD["constituents"]],axis=0)
        ylo = np.mean([con["m_lo"].predict(F) for con in AD["constituents"]],axis=0)
        if AD.get("blended") and float(AD.get("alpha",1.0))<1.0:
            a=float(AD["alpha"])
            yhi=a*yhi+(1-a)*float(AD.get("mu_hi",0))
            ylo=a*ylo+(1-a)*float(AD.get("mu_lo",0))
    else:
        yhi=AD["hi_model"].predict(F); ylo=AD["lo_model"].predict(F)
    c_vals=c.reindex(F.index).values
    ph=c_vals*(1+np.clip(yhi,0,None)); pl=c_vals*(1-np.clip(ylo,0,None))
    idx=np.asarray(F.index,dtype="datetime64[ns]")
    nd=np.empty(len(F),dtype="datetime64[ns]")
    nd[:-1]=idx[1:]; nd[-1]=idx[-1]+np.timedelta64(1,"D")
    res=pd.DataFrame({"close_asof":c_vals,"pred_high":ph,"pred_low":pl},
                     index=pd.DatetimeIndex(nd,name="target_date"))
    return res[~res.index.duplicated(keep="last")]

def _prep_comp(df_raw, preds, start_iso, end_iso):
    sd, ed = pd.Timestamp(start_iso), pd.Timestamp(end_iso)
    p = preds.loc[(preds.index>=sd-pd.Timedelta(days=60))&(preds.index<=ed)].copy()
    p["actual_high"]  = df_raw["btc_high"].reindex(p.index).values
    p["actual_low"]   = df_raw["btc_low"].reindex(p.index).values
    p["actual_close"] = df_raw["btc_close"].reindex(p.index).values
    return p.dropna(subset=["actual_high","actual_low","actual_close"]).reset_index()

# ─────────────────────────────────────────────────────────────────────────────
# Signal computation — returns BOTH current and Option A entry arrays
# ─────────────────────────────────────────────────────────────────────────────
def compute_signals_both(comp):
    N      = len(comp)
    c_asof = comp["close_asof"].values.astype(float)
    ph     = comp["pred_high"].values.astype(float)
    pl     = comp["pred_low"].values.astype(float)
    ah     = comp["actual_high"].values.astype(float)
    al     = comp["actual_low"].values.astype(float)

    err_hi = (ah-ph)/c_asof*100; err_lo = (pl-al)/c_asof*100
    hi_brk = (ah>ph).astype(int); lo_brk = (al<pl).astype(int)

    ehma3=np.zeros(N); elma3=np.zeros(N)
    hb3=np.zeros(N,dtype=int); lb3=np.zeros(N,dtype=int)
    for i in range(N):
        s=max(0,i-2)
        ehma3[i]=np.mean(err_hi[s:i+1]); elma3[i]=np.mean(err_lo[s:i+1])
        hb3[i]=int(np.sum(hi_brk[s:i+1])); lb3[i]=int(np.sum(lo_brk[s:i+1]))

    u1=(ehma3>0.7)&(hb3>=2)
    d1=(lb3>=2)&(elma3>0.5)
    d2=ehma3<-0.75
    d3=np.zeros(N,dtype=bool)
    for i in range(1,N):
        c2=0
        for k in range(i-1,-1,-1):
            if hi_brk[k]: c2+=1
            else: break
        if c2>=3 and lo_brk[i]: d3[i]=True

    ma30=np.full(N,np.nan)
    for i in range(N):
        w=min(30,i+1); ma30[i]=np.mean(c_asof[max(0,i-w+1):i+1])
    above_ma30=c_asof>ma30
    ma30_slope=np.zeros(N,dtype=bool)
    for i in range(5,N):
        if np.isfinite(ma30[i]) and np.isfinite(ma30[i-5]):
            ma30_slope[i]=ma30[i]>ma30[i-5]
    bull_regime=above_ma30&ma30_slope

    clean_7d=np.zeros(N,dtype=bool)
    for i in range(N):
        lo_i=max(0,i-7)
        clean_7d[i]=not bool(np.any(d1[lo_i:i]|d2[lo_i:i]))

    roll_norm=np.array([float(np.mean(err_hi[max(0,i-29):i+1])) for i in range(N)])
    dn_score=np.zeros(N)
    for i in range(N):
        norm=max(abs(roll_norm[i]),0.01)
        dn_score[i]=((-ehma3[i]/norm)*0.30+(lb3[i]/3.0)*0.30
                     +(elma3[i]/max(abs(elma3[i]),0.10))*0.20+float(lo_brk[i])*0.20)
    v_rev_bar=(dn_score>0.8)&(err_lo>3.0)
    v_recent=np.zeros(N,dtype=bool)
    for i in range(N):
        v_recent[i]=bool(np.any(v_rev_bar[max(0,i-2):i+1]))

    ema10=pd.Series(c_asof).ewm(span=10,adjust=False).mean().values

    # Current entry gate
    tf2_current  = u1 & ((above_ma30 ^ clean_7d) | v_recent)
    # Option A: clean_7d path also requires rising MA30
    tf2_option_a = u1 & ((above_ma30 ^ (clean_7d & ma30_slope)) | v_recent)

    return dict(
        N=N, c_asof=c_asof,
        u1=u1, d1=d1, d2=d2, d3=d3,
        above_ma30=above_ma30, ma30_slope=ma30_slope, bull_regime=bull_regime,
        clean_7d=clean_7d, v_recent=v_recent, ema10=ema10,
        tf2_entry=tf2_current, tf2_option_a=tf2_option_a,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Backtest engine (SL5 only, supports custom entry array)
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest_sl5(dates, asset_px, btc_px, sigs, entry_arr,
                     sl_type, sl_pct, cap, bt_start):
    N    = len(dates)
    d2   = sigs["d2"]; d3 = sigs["d3"]; bull = sigs["bull_regime"]
    above_ma30=sigs["above_ma30"]; clean_7d=sigs["clean_7d"]; v_recent=sigs["v_recent"]

    _bt0 = max(WARMUP, int(pd.DatetimeIndex(dates).searchsorted(bt_start)))

    nav=cap; pos="CASH"; qty=0.0
    e_asset_price=None; e_date=None; e_nav=None; e_trigger=None; hwm=None
    from_sl=False; sl_exit_asset_px=None; bars_since_sl=0
    trades=[]; nav_arr=np.full(N,np.nan)

    for i in range(N):
        si=i-1
        price=asset_px[i]
        if i<_bt0: nav_arr[i]=cap; continue

        if pos=="CASH" and from_sl:
            bars_since_sl+=1

        if pos=="LONG":
            cur=qty*price
            stop_level=(hwm*(1-sl_pct) if sl_type=="trailing"
                        else e_asset_price*(1-sl_pct))
            if sl_type=="trailing": hwm=max(hwm,price)

            if price<=stop_level:
                nav=cur
                trades.append(dict(
                    entry_date=e_date, entry_price=e_asset_price, entry_nav=e_nav,
                    entry_trigger=e_trigger, exit_date=dates[i], exit_price=price,
                    exit_nav=nav, pnl_pct=(price/e_asset_price-1)*100,
                    pnl_abs=nav-e_nav,
                    exit_signal=f"SL-fixed-{sl_pct*100:.0f}%",
                    duration_days=(dates[i]-e_date).days, was_sl=True,
                    hwm=hwm if sl_type=="trailing" else e_asset_price,
                ))
                pos="CASH"; qty=0.0; from_sl=True
                sl_exit_asset_px=price; bars_since_sl=0; hwm=None
            elif si>=0:
                should_exit=bool(d3[si] or (d2[si] and not bull[si]))
                if should_exit:
                    nav=cur
                    xl="D3" if d3[si] else "D2"
                    trades.append(dict(
                        entry_date=e_date, entry_price=e_asset_price, entry_nav=e_nav,
                        entry_trigger=e_trigger, exit_date=dates[i], exit_price=price,
                        exit_nav=nav, pnl_pct=(price/e_asset_price-1)*100,
                        pnl_abs=nav-e_nav, exit_signal=xl,
                        duration_days=(dates[i]-e_date).days, was_sl=False, hwm=hwm,
                    ))
                    pos="CASH"; qty=0.0; from_sl=False; hwm=None
                else:
                    nav=cur
            else:
                nav=cur
        else:
            if si>=0 and entry_arr[si]:
                allow = bool(bull[si]) or (not from_sl) or (bars_since_sl>=10)
                if allow:
                    if not np.isfinite(price) or price<=0:
                        nav_arr[i]=nav; continue
                    qty=nav/price; e_asset_price=price; e_date=dates[i]
                    e_nav=nav; pos="LONG"; hwm=price
                    from_sl=False; bars_since_sl=0
                    if v_recent[si]:       e_trigger="U1+V-reversal"
                    elif above_ma30[si]:   e_trigger="U1+↑MA30"
                    else:                  e_trigger="U1+clean7d"

        nav_arr[i]=qty*price if pos=="LONG" else nav

    if pos=="LONG" and np.isfinite(asset_px[N-1]):
        nav_arr[N-1]=qty*asset_px[N-1]

    nav_s=pd.Series(nav_arr[_bt0:],index=dates[_bt0:]).ffill()
    bh_s =pd.Series(cap*asset_px[_bt0:]/asset_px[_bt0],index=dates[_bt0:])
    return dict(trades=trades,nav=nav_s,bh=bh_s,open_pos=(pos=="LONG"))

# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def metrics(res, cap=INITIAL_CAPITAL):
    nav=res["nav"]; bh=res["bh"]; trades=res["trades"]
    fn=float(nav.iloc[-1]); fb=float(bh.iloc[-1])
    n_y=(nav.index[-1]-nav.index[0]).days/365.25
    cagr=((fn/cap)**(1/n_y)-1)*100 if n_y>0 else 0
    dr=nav.pct_change().fillna(0)
    rfd=1.045**(1/252)-1; exc=dr-rfd
    sharpe=float(exc.mean()/exc.std()*np.sqrt(252)) if exc.std()>0 else 0
    rm=nav.cummax(); dd=(nav-rm)/rm*100; max_dd=float(dd.min())
    wins=[t for t in trades if t["pnl_pct"]>0]
    losses=[t for t in trades if t["pnl_pct"]<=0]
    sl_ex=[t for t in trades if t.get("was_sl")]
    win_rate=100*len(wins)/len(trades) if trades else 0
    gp=sum(t["pnl_abs"] for t in wins)  if wins   else 0
    gl=abs(sum(t["pnl_abs"] for t in losses)) if losses else 1e-9
    avg_win=float(np.mean([t["pnl_pct"] for t in wins]))   if wins   else 0
    avg_loss=float(np.mean([t["pnl_pct"] for t in losses])) if losses else 0
    days_in=sum(t["duration_days"] for t in trades)
    tot_days=max(1,(nav.index[-1]-nav.index[0]).days)
    return dict(
        strat_ret=(fn/cap-1)*100, bh_ret=(fb/cap-1)*100,
        alpha_pp=(fn/cap-1)*100-(fb/cap-1)*100,
        cagr=cagr, sharpe=sharpe, max_dd=max_dd,
        n_trades=len(trades), n_sl=len(sl_ex),
        n_wins=len(wins), n_losses=len(losses),
        win_rate=win_rate, avg_win=avg_win, avg_loss=avg_loss,
        profit_factor=gp/gl,
        time_in=100*days_in/tot_days,
        trades=trades,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Entry path breakdown
# ─────────────────────────────────────────────────────────────────────────────
def entry_paths(trades):
    paths={"U1+↑MA30":0,"U1+clean7d":0,"U1+V-reversal":0}
    for t in trades:
        trig=t.get("entry_trigger","")
        if "V-reversal" in trig: paths["U1+V-reversal"]+=1
        elif "↑MA30" in trig or "MA30" in trig: paths["U1+↑MA30"]+=1
        else: paths["U1+clean7d"]+=1
    return paths

# ─────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ─────────────────────────────────────────────────────────────────────────────
W=92
def hr(ch="─"): print("  "+ch*(W-2))

ROWS = [
    ("Strategy return %",    "strat_ret",  "{:+.1f}%",  False),
    ("B&H return %",         "bh_ret",     "{:+.1f}%",  False),
    ("Alpha vs B&H (pp)",    "alpha_pp",   "{:+.1f}pp", False),
    ("Max drawdown %",       "max_dd",     "{:.1f}%",   False),
    ("Sharpe ratio",         "sharpe",     "{:.2f}",    False),
    ("Win rate %",           "win_rate",   "{:.1f}%",   False),
    ("Avg winning trade %",  "avg_win",    "{:+.1f}%",  False),
    ("Avg losing trade %",   "avg_loss",   "{:+.1f}%",  False),
    ("Profit factor",        "profit_factor","{:.2f}",  False),
    ("# Trades",             "n_trades",   "{:d}",      True),
    ("# Winning trades",     "n_wins",     "{:d}",      True),
    ("# Losing trades",      "n_losses",   "{:d}",      True),
    ("# Stop-loss exits",    "n_sl",       "{:d}",      True),
    ("Time in market %",     "time_in",    "{:.1f}%",   False),
]

def print_table(period_name, asset, cur_m, opta_m, cur_paths, opta_paths,
                cur_signals, opta_signals, blocked, ma30_up_pct):
    hr("═")
    print(f"  PERIOD: {period_name}  |  ASSET: {asset}")
    hr("═")
    print(f"  {'Metric':<28}  {'Current (UI)':>16}  {'Option A':>16}  {'Δ':>12}")
    hr()
    for label, key, fmt, is_int in ROWS:
        cv = cur_m[key]; av = opta_m[key]
        if is_int:
            delta = int(av)-int(cv)
            print(f"  {label:<28}  {int(cv):>16}  {int(av):>16}  {delta:>+12d}")
        else:
            delta = av-cv
            cs=fmt.format(cv); as_=fmt.format(av)
            if "%" in fmt: ds=f"{delta:+.1f}pp"
            elif fmt=="{:.2f}": ds=f"{delta:+.2f}"
            else: ds=f"{delta:+.1f}"
            print(f"  {label:<28}  {cs:>16}  {as_:>16}  {ds:>12}")
    hr()
    print(f"  {'Entry path breakdown':}")
    for p in ["U1+↑MA30","U1+clean7d","U1+V-reversal"]:
        c=cur_paths[p]; a=opta_paths[p]
        print(f"  {'':>4}{p:<24}  {c:>16}  {a:>16}  {a-c:>+12d}")
    hr()
    print(f"  Signal-level stats (trading window):")
    print(f"  {'':>4}{'Entry signal bars (current):':<28}  {cur_signals}")
    print(f"  {'':>4}{'Entry signal bars (Option A):':<28}  {opta_signals}")
    print(f"  {'':>4}{'Blocked by Option A:':<28}  {blocked}  (clean_7d bars where MA30 declining)")
    print(f"  {'':>4}{'MA30 rising % of bars:':<28}  {ma30_up_pct:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n"+"═"*W)
    print("  OPTION A ENTRY FILTER — vs CURRENT IMPLEMENTATION  (SL5 variant)")
    print("  Current:  tf2_entry = u1 & ((above_ma30 ^ clean_7d) | v_recent)")
    print("  Option A: tf2_entry = u1 & ((above_ma30 ^ (clean_7d & ma30_slope)) | v_recent)")
    print("  MSTR: fixed −3% stop  |  MSTU: fixed −7% stop  |  SL5 regime-adaptive re-entry")
    print("═"*W+"\n")

    df_raw_cache: dict = {}

    for pname, (start, end, fs) in PERIODS.items():
        fetch_end = (pd.Timestamp(end)+pd.Timedelta(days=3)).strftime("%Y-%m-%d")
        cache_key = (fs, fetch_end)

        if cache_key not in df_raw_cache:
            print(f"\n── Fetching data: {pname} ──")
            df_raw       = fetch_data(fs, fetch_end)
            preds        = build_ct_preds(df_raw, fetch_start=fs, fetch_end=fetch_end)
            mstr_px_raw  = fetch_asset_prices("MSTR", fs, fetch_end)
            mstu_px_raw  = fetch_asset_prices("MSTU", fs, fetch_end)
            df_raw_cache[cache_key] = dict(
                df_raw=df_raw, preds=preds,
                mstr_px=mstr_px_raw, mstu_px=mstu_px_raw)
        cached=df_raw_cache[cache_key]
        df_raw=cached["df_raw"]; preds=cached["preds"]
        mstr_px_raw=cached["mstr_px"]; mstu_px_raw=cached["mstu_px"]

        comp=_prep_comp(df_raw, preds, start, end)
        sd=pd.Timestamp(start)
        if len(comp)<WARMUP+3:
            print(f"  [SKIP] {pname}: insufficient bars"); continue

        dates  = pd.DatetimeIndex(comp["target_date"])
        sigs   = compute_signals_both(comp)
        N      = sigs["N"]
        btc_px = comp["actual_close"].values.astype(float)
        _bt0   = max(WARMUP, int(dates.searchsorted(sd)))

        # Signal-level counts (trading window only)
        cur_sigs  = int(np.sum(sigs["tf2_entry"][_bt0:]))
        opta_sigs = int(np.sum(sigs["tf2_option_a"][_bt0:]))
        blocked   = int(np.sum(sigs["tf2_entry"][_bt0:] & ~sigs["tf2_option_a"][_bt0:]))
        ma30_up   = 100*np.mean(sigs["ma30_slope"][_bt0:])

        asset_px_map = {"MSTR": mstr_px_raw, "MSTU": mstu_px_raw}

        for asset_name in ["MSTR", "MSTU"]:
            px_raw=asset_px_map[asset_name]
            if px_raw is None or px_raw.empty:
                print(f"\n  [SKIP] {asset_name}: no price data"); continue
            a=px_raw.reindex(dates).ffill().bfill().values.astype(float)
            if np.sum(np.isfinite(a[_bt0:])&(a[_bt0:]>0))<5:
                print(f"\n  [SKIP] {asset_name}: insufficient valid prices"); continue

            cfg=ASSET_CONFIG[asset_name]; sl_t=cfg["sl_type"]; sl_pct=cfg["sl_pct"]

            cur_res  = run_backtest_sl5(dates, a, btc_px, sigs,
                                        entry_arr=sigs["tf2_entry"],
                                        sl_type=sl_t, sl_pct=sl_pct,
                                        cap=INITIAL_CAPITAL, bt_start=sd)
            opta_res = run_backtest_sl5(dates, a, btc_px, sigs,
                                        entry_arr=sigs["tf2_option_a"],
                                        sl_type=sl_t, sl_pct=sl_pct,
                                        cap=INITIAL_CAPITAL, bt_start=sd)
            cur_m  = metrics(cur_res)
            opta_m = metrics(opta_res)

            cp = entry_paths(cur_m["trades"])
            ap = entry_paths(opta_m["trades"])

            print()
            print_table(pname, asset_name, cur_m, opta_m, cp, ap,
                        cur_sigs, opta_sigs, blocked, ma30_up)

    print(f"\n{'═'*W}")
    print("  DONE")
    print(f"{'═'*W}\n")


if __name__ == "__main__":
    main()
