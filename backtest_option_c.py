#!/usr/bin/env python3
"""
Option C — Pure Regime Entry (strict gate) vs Current and Option B
====================================================================
Current:  tf2_entry = u1 & ((above_ma30 ^ clean_7d) | v_recent)
Option B: tf2_entry = u1 & ((bull_regime ^ clean_7d) | v_recent)   [Confirmed Uptrend]
Option C: tf2_entry = u1 & (bull_regime | (clean_7d & ~above_ma30) | v_recent)  [Pure Regime]

Option C separates the two entry paths cleanly:
  - bull_regime path:           BTC above MA30 AND MA30 rising (confirmed uptrend)
  - clean_7d & ~above_ma30:     BTC below MA30 but clean window (fresh breakout from below)
  - v_recent:                   V-reversal capitulation spike
  Blocked entirely: above_ma30=True + ma30_slope=False ("no-man's-land" dead-cat zone)

This eliminates the XOR side-effect that allowed Option B to accidentally fire new
entries in the no-man's-land zone (above_ma30=True, ma30_slope=False, clean_7d=True).

Runs SL5 regime-adaptive for MSTR (−3% stop) and MSTU (−7% stop) across all periods.
"""

import sys, warnings, requests
warnings.filterwarnings("ignore")
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

try:
    import sklearn._loss._loss as _sklearn_loss_ext
    if "_loss" not in sys.modules:
        sys.modules["_loss"] = _sklearn_loss_ext
    import sklearn._loss.loss
    import sklearn._loss.link
except Exception:
    pass

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

ASSET_CONFIG = {
    "MSTR": {"ticker": "MSTR", "sl_pct": 0.03},
    "MSTU": {"ticker": "MSTU", "sl_pct": 0.07},
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
        try: frames[f"{nm}_close"] = _yf_dl(sym, fetch_start, fetch_end)["Close"]
        except Exception: pass
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
        except Exception: pass
    print(f"    On-chain: {ok}/{len(_ONCHAIN)} OK")
    return df

def fetch_asset_prices(ticker, fetch_start, fetch_end):
    try:
        d = _yf_dl(ticker, fetch_start, fetch_end)
        px = d["Close"].sort_index()
        all_days = pd.date_range(px.index[0], max(px.index[-1], pd.Timestamp(fetch_end)), freq="D")
        return px.reindex(all_days).ffill()
    except Exception as e:
        print(f"    [WARN] {ticker}: {e}"); return pd.Series(dtype=float)

def build_ct_preds(df):
    c, h, l_, v = df["btc_close"], df["btc_high"], df["btc_low"], df["btc_volume"]
    ret = np.log(c).diff()
    f = pd.DataFrame(index=df.index)
    for k in [1,3,5,7,14,30]: f[f"ret_{k}"] = ret.rolling(k).sum()
    for k in [5,10,20,30]:     f[f"vol_{k}"] = ret.rolling(k).std()
    prev_c = c.shift(1)
    tr = pd.concat([(h-l_),(h-prev_c).abs(),(l_-prev_c).abs()],axis=1).max(axis=1)
    for k in [7,14,30]: f[f"atr_{k}"] = tr.rolling(k).mean()/c
    f["range_today"]=(h-l_)/c; f["range_ma7"]=((h-l_)/c).rolling(7).mean()
    f["range_ma30"]=((h-l_)/c).rolling(30).mean(); f["range_std30"]=((h-l_)/c).rolling(30).std()
    g=c.diff().clip(lower=0).rolling(14).mean(); ls=(-c.diff().clip(upper=0)).rolling(14).mean()
    f["rsi_14"]=100-100/(1+g/ls.replace(0,np.nan))
    e12=c.ewm(span=12,adjust=False).mean(); e26=c.ewm(span=26,adjust=False).mean()
    macd=e12-e26; f["macd"]=macd/c; f["macd_sig"]=macd.ewm(span=9,adjust=False).mean()/c
    f["macd_hist"]=(macd-macd.ewm(span=9,adjust=False).mean())/c
    ma20=c.rolling(20).mean(); sd20=c.rolling(20).std(); f["bb_width"]=(4*sd20)/ma20
    f["dist_hi_30"]=c/c.rolling(30).max()-1; f["dist_lo_30"]=c/c.rolling(30).min()-1
    f["dist_hi_90"]=c/c.rolling(90).max()-1; f["vol_chg_1"]=np.log(v).diff()
    f["vol_z_20"]=(np.log(v)-np.log(v).rolling(20).mean())/np.log(v).rolling(20).std()
    f["vol_ma_ratio"]=v/v.rolling(20).mean()
    dow=df.index.dayofweek
    for i in range(6): f[f"dow_{i}"]=(dow==i).astype(float)
    for nm in ["spx","ndx","vix","gold","dxy","tnx","eth"]:
        col=f"{nm}_close"
        if col not in df.columns: continue
        s=df[col]; lr=np.log(s).diff()
        for k in [1,5,20]: f[f"{nm}_ret_{k}"]=lr.rolling(k).sum()
        f[f"{nm}_vol_20"]=lr.rolling(20).std()
    for corr_nm, corr_col in [("spx","spx_close"),("ndx","ndx_close"),
                               ("gold","gold_close"),("dxy","dxy_close")]:
        if corr_col in df.columns:
            f[f"btc_{corr_nm}_corr_30"]=ret.rolling(30).corr(np.log(df[corr_col]).diff())
    for col in [x for x in df.columns if x.startswith("oc_")]:
        s=df[col].astype(float); sl=np.log(s.replace(0,np.nan))
        f[f"{col}_d1"]=sl.diff(1); f[f"{col}_d7"]=sl.diff(7)
        f[f"{col}_z30"]=(sl-sl.rolling(30).mean())/sl.rolling(30).std()
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
    f["below_sma50"]=(c<sma50).astype(float); f["below_sma50_5d"]=f["below_sma50"].rolling(5).min().fillna(0)
    f["cb_premium"]=f["cb_premium_ma3"]=f["cb_premium_z7"]=0.0
    fc=AD["feat_cols"]; f=f.replace([np.inf,-np.inf],np.nan)
    for col in fc:
        if col not in f.columns: f[col]=np.nan
    F=f[fc].dropna()
    if F.empty: raise RuntimeError("Feature matrix empty")
    if AD.get("ensemble") and AD.get("constituents"):
        yhi=np.mean([con["m_hi"].predict(F) for con in AD["constituents"]],axis=0)
        ylo=np.mean([con["m_lo"].predict(F) for con in AD["constituents"]],axis=0)
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
    sd,ed=pd.Timestamp(start_iso),pd.Timestamp(end_iso)
    p=preds.loc[(preds.index>=sd-pd.Timedelta(days=60))&(preds.index<=ed)].copy()
    p["actual_high"]=df_raw["btc_high"].reindex(p.index).values
    p["actual_low"]=df_raw["btc_low"].reindex(p.index).values
    p["actual_close"]=df_raw["btc_close"].reindex(p.index).values
    return p.dropna(subset=["actual_high","actual_low","actual_close"]).reset_index()

def compute_signals(comp):
    N=len(comp)
    c_asof=comp["close_asof"].values.astype(float)
    ph=comp["pred_high"].values.astype(float)
    pl=comp["pred_low"].values.astype(float)
    ah=comp["actual_high"].values.astype(float)
    al=comp["actual_low"].values.astype(float)
    err_hi=(ah-ph)/c_asof*100; err_lo=(pl-al)/c_asof*100
    hi_brk=(ah>ph).astype(int); lo_brk=(al<pl).astype(int)
    ehma3=np.zeros(N); elma3=np.zeros(N); hb3=np.zeros(N,dtype=int); lb3=np.zeros(N,dtype=int)
    for i in range(N):
        s=max(0,i-2)
        ehma3[i]=np.mean(err_hi[s:i+1]); elma3[i]=np.mean(err_lo[s:i+1])
        hb3[i]=int(np.sum(hi_brk[s:i+1])); lb3[i]=int(np.sum(lo_brk[s:i+1]))
    u1=(ehma3>0.7)&(hb3>=2); d1=(lb3>=2)&(elma3>0.5); d2=ehma3<-0.75
    d3=np.zeros(N,dtype=bool)
    for i in range(1,N):
        c2=0
        for k in range(i-1,-1,-1):
            if hi_brk[k]: c2+=1
            else: break
        if c2>=3 and lo_brk[i]: d3[i]=True
    ma30=np.full(N,np.nan)
    for i in range(N): ma30[i]=np.mean(c_asof[max(0,i-29):i+1])
    above_ma30=c_asof>ma30
    ma30_slope=np.zeros(N,dtype=bool)
    for i in range(5,N):
        if np.isfinite(ma30[i]) and np.isfinite(ma30[i-5]): ma30_slope[i]=ma30[i]>ma30[i-5]
    bull_regime=above_ma30&ma30_slope
    clean_7d=np.zeros(N,dtype=bool)
    for i in range(N):
        lo_i=max(0,i-7); clean_7d[i]=not bool(np.any(d1[lo_i:i]|d2[lo_i:i]))
    roll_norm=np.array([float(np.mean(err_hi[max(0,i-29):i+1])) for i in range(N)])
    dn_score=np.zeros(N)
    for i in range(N):
        norm=max(abs(roll_norm[i]),0.01)
        dn_score[i]=((-ehma3[i]/norm)*0.30+(lb3[i]/3.0)*0.30
                     +(elma3[i]/max(abs(elma3[i]),0.10))*0.20+float(lo_brk[i])*0.20)
    v_rev_bar=(dn_score>0.8)&(err_lo>3.0)
    v_recent=np.zeros(N,dtype=bool)
    for i in range(N): v_recent[i]=bool(np.any(v_rev_bar[max(0,i-2):i+1]))

    tf_current  = u1 & ((above_ma30 ^ clean_7d) | v_recent)
    tf_option_b = u1 & ((bull_regime ^ clean_7d) | v_recent)
    tf_option_c = u1 & (bull_regime | (clean_7d & ~above_ma30) | v_recent)

    return dict(N=N, c_asof=c_asof, u1=u1, d1=d1, d2=d2, d3=d3,
                above_ma30=above_ma30, ma30_slope=ma30_slope, bull_regime=bull_regime,
                clean_7d=clean_7d, v_recent=v_recent, err_lo=err_lo,
                tf_current=tf_current, tf_option_b=tf_option_b, tf_option_c=tf_option_c)

def run_backtest(dates, asset_px, sigs, entry_arr, sl_pct, cap, bt_start):
    N=len(dates); d2=sigs["d2"]; d3=sigs["d3"]; bull=sigs["bull_regime"]
    _bt0=max(WARMUP, int(pd.DatetimeIndex(dates).searchsorted(bt_start)))
    nav=cap; pos="CASH"; qty=0.0
    e_price=e_date=e_nav=e_trigger=None; from_sl=False; bars_since_sl=0
    stop_px=0.0; trades=[]; nav_arr=np.full(N,np.nan)
    for i in range(N):
        price=asset_px[i]
        if i<_bt0: nav_arr[i]=cap; continue
        if pos=="CASH" and from_sl: bars_since_sl+=1
        if pos=="LONG":
            if price<=stop_px:
                nav=qty*price
                trades.append(dict(entry_date=e_date,entry_price=e_price,
                    exit_date=dates[i],exit_price=price,
                    pnl_pct=(price/e_price-1)*100,stop=True,
                    entry_trigger=e_trigger,
                    duration_days=(dates[i]-e_date).days))
                pos="CASH"; qty=0.0; from_sl=True; bars_since_sl=0
            else:
                si=i-1
                bear_exit = (d2[si] or d3[si]) if si>=0 else False
                bull_exit = d3[si] if si>=0 else False
                exit_sig = bull_exit if bull[si] else bear_exit if si>=0 else False
                if exit_sig:
                    nav=qty*price
                    trades.append(dict(entry_date=e_date,entry_price=e_price,
                        exit_date=dates[i],exit_price=price,
                        pnl_pct=(price/e_price-1)*100,stop=False,
                        entry_trigger=e_trigger,
                        duration_days=(dates[i]-e_date).days))
                    pos="CASH"; qty=0.0; from_sl=False
            nav_arr[i]=qty*price if pos=="LONG" else nav
            continue
        # CASH
        si=i-1
        can_enter=True
        if from_sl:
            bullish_now=bool(bull[si]) if si>=0 else False
            can_enter = bullish_now or (bars_since_sl>=10)
        if can_enter and si>=0 and entry_arr[si] and np.isfinite(price) and price>0:
            pos="LONG"; qty=nav/price; stop_px=price*(1-sl_pct)
            e_price=price; e_date=dates[i]; e_nav=nav; from_sl=False; bars_since_sl=0
            trig=e_trigger=""
            if bull[si] and not sigs["clean_7d"][si]: trig="U1+↑MA30"
            elif sigs["clean_7d"][si] and not sigs["above_ma30"][si]: trig="U1+clean7d(below)"
            elif sigs["clean_7d"][si]: trig="U1+clean7d"
            elif sigs["v_recent"][si]: trig="U1+V-reversal"
            else: trig="U1+other"
            e_trigger=trig
        nav_arr[i]=qty*price if pos=="LONG" else nav
    # close open position at last price
    if pos=="LONG":
        nav=qty*asset_px[-1]
    return trades, pd.Series(nav_arr,index=pd.DatetimeIndex(dates))

def metrics(trades, nav_arr, cap):
    fn=nav_arr.dropna().iloc[-1]
    fb=cap  # B&H just use cap as baseline (we report strat_ret)
    wins=[t for t in trades if t["pnl_pct"]>0]; losses=[t for t in trades if t["pnl_pct"]<=0]
    wr=100*len(wins)/max(1,len(trades))
    avg_win=np.mean([t["pnl_pct"] for t in wins]) if wins else 0
    avg_loss=np.mean([t["pnl_pct"] for t in losses]) if losses else 0
    gp=sum(t["pnl_pct"] for t in wins); gl=abs(sum(t["pnl_pct"] for t in losses))
    pf=gp/gl if gl>0 else 999
    nav_clean=nav_arr.dropna()
    dd=0.0
    if len(nav_clean)>1:
        roll_max=nav_clean.cummax(); dd=float(((nav_clean-roll_max)/roll_max).min()*100)
    rets=nav_clean.pct_change().dropna()
    sh=float(rets.mean()/rets.std()*np.sqrt(252)) if rets.std()>0 else 0
    return dict(
        strat_ret=(fn/cap-1)*100,
        alpha_pp=(fn/cap-1)*100,  # vs B&H computed at call site
        n_trades=len(trades), n_wins=len(wins), n_losses=len(losses),
        n_sl=sum(1 for t in trades if t.get("stop")),
        win_rate=wr, avg_win=avg_win, avg_loss=avg_loss, profit_factor=pf,
        max_dd=dd, sharpe=sh, final_nav=fn,
    )

W=110
def hr(ch="─"): print("  "+ch*(W-2))

def print_row(label, cv, bv, cv2, fmt, is_int=False):
    if is_int:
        db=int(bv)-int(cv); dc=int(cv2)-int(cv)
        print(f"  {label:<30}  {int(cv):>12}  {int(bv):>12}  {db:>+8d}  {int(cv2):>12}  {dc:>+8d}")
    else:
        db=bv-cv; dc=cv2-cv
        cs=fmt.format(cv); bs=fmt.format(bv); c2s=fmt.format(cv2)
        if "%" in fmt: dbs=f"{db:+.1f}pp"; dcs=f"{dc:+.1f}pp"
        elif ".2f" in fmt: dbs=f"{db:+.2f}"; dcs=f"{dc:+.2f}"
        else: dbs=f"{db:+.1f}"; dcs=f"{dc:+.1f}"
        print(f"  {label:<30}  {cs:>12}  {bs:>12}  {dbs:>8}  {c2s:>12}  {dcs:>8}")

def main():
    print("\n"+"═"*W)
    print("  OPTION C — PURE REGIME ENTRY vs CURRENT vs OPTION B  (SL5 regime-adaptive)")
    print("  Current  (Standard):           tf2 = u1 & ((above_ma30 ^ clean_7d) | v_recent)")
    print("  Option B (Confirmed Uptrend):  tf2 = u1 & ((bull_regime ^ clean_7d) | v_recent)")
    print("  Option C (Pure Regime Entry):  tf2 = u1 & (bull_regime | (clean_7d & ~above_ma30) | v_recent)")
    print("  MSTR: fixed −3% stop  |  MSTU: fixed −7% stop  |  SL5 regime-adaptive re-entry")
    print("═"*W+"\n")

    cache: dict = {}

    for pname, (start, end, fs) in PERIODS.items():
        fetch_end=(pd.Timestamp(end)+pd.Timedelta(days=3)).strftime("%Y-%m-%d")
        ck=(fs, fetch_end)
        if ck not in cache:
            df_raw=fetch_data(fs, fetch_end)
            preds=build_ct_preds(df_raw)
            mstr_px=fetch_asset_prices("MSTR", fs, fetch_end)
            mstu_px=fetch_asset_prices("MSTU", fs, fetch_end)
            cache[ck]=dict(df_raw=df_raw,preds=preds,mstr_px=mstr_px,mstu_px=mstu_px)

        c=cache[ck]; df_raw=c["df_raw"]; preds=c["preds"]
        comp=_prep_comp(df_raw, preds, start, end)
        if len(comp)<WARMUP+3:
            print(f"  [SKIP] {pname}: insufficient bars"); continue

        dates=pd.DatetimeIndex(comp["target_date"])
        sigs=compute_signals(comp)
        sd=pd.Timestamp(start)
        _bt0=max(WARMUP, int(dates.searchsorted(sd)))

        bh_px_start=comp["actual_close"].values[_bt0] if _bt0<len(comp) else 1
        bh_ret=( comp["actual_close"].values[-1]/bh_px_start - 1)*100

        for asset_name in ["MSTR","MSTU"]:
            px_raw=c[f"{asset_name.lower()}_px"]
            if px_raw is None or px_raw.empty:
                print(f"  [SKIP] {asset_name}: no price data"); continue
            a=px_raw.reindex(dates).ffill().bfill().values.astype(float)
            if np.sum(np.isfinite(a[_bt0:])&(a[_bt0:]>0))<5:
                print(f"  [SKIP] {asset_name}: insufficient prices"); continue
            sl_pct=ASSET_CONFIG[asset_name]["sl_pct"]

            cur_trades,  cur_nav  = run_backtest(dates, a, sigs, sigs["tf_current"],  sl_pct, INITIAL_CAPITAL, sd)
            optb_trades, optb_nav = run_backtest(dates, a, sigs, sigs["tf_option_b"], sl_pct, INITIAL_CAPITAL, sd)
            optc_trades, optc_nav = run_backtest(dates, a, sigs, sigs["tf_option_c"], sl_pct, INITIAL_CAPITAL, sd)

            cm=metrics(cur_trades,  cur_nav,  INITIAL_CAPITAL)
            bm=metrics(optb_trades, optb_nav, INITIAL_CAPITAL)
            cc=metrics(optc_trades, optc_nav, INITIAL_CAPITAL)

            asset_bh_start=a[_bt0] if (_bt0<len(a) and np.isfinite(a[_bt0]) and a[_bt0]>0) else None
            asset_bh_ret=((a[-1]/asset_bh_start-1)*100) if asset_bh_start else 0
            asset_bh_nav=INITIAL_CAPITAL*(1+asset_bh_ret/100)

            hr("═")
            print(f"  PERIOD: {pname}  |  ASSET: {asset_name}  |  B&H {asset_name}: {asset_bh_ret:+.1f}%  (NAV ${asset_bh_nav:,.0f})")
            hr("═")
            print(f"  {'Metric':<30}  {'Current':>12}  {'Opt B (CU)':>12}  {'ΔB':>8}  {'Opt C (PR)':>12}  {'ΔC':>8}")
            hr()
            for label,key,fmt,ii in [
                ("Strategy return %",    "strat_ret",    "{:+.1f}%", False),
                ("Final NAV $",          "final_nav",    "${:,.0f}", False),
                ("Max drawdown %",       "max_dd",       "{:.1f}%",  False),
                ("Sharpe ratio",         "sharpe",       "{:.2f}",   False),
                ("Win rate %",           "win_rate",     "{:.1f}%",  False),
                ("# Trades",             "n_trades",     "{:d}",     True),
                ("# Wins",               "n_wins",       "{:d}",     True),
                ("# Losses",             "n_losses",     "{:d}",     True),
                ("# Stop-loss exits",    "n_sl",         "{:d}",     True),
                ("Avg win %",            "avg_win",      "{:+.1f}%", False),
                ("Avg loss %",           "avg_loss",     "{:+.1f}%", False),
                ("Profit factor",        "profit_factor","{:.2f}",   False),
            ]:
                print_row(label, cm[key], bm[key], cc[key], fmt, ii)
            hr()
            # signal counts
            cur_sigs=int(np.sum(sigs["tf_current"][_bt0:]))
            optb_sigs=int(np.sum(sigs["tf_option_b"][_bt0:]))
            optc_sigs=int(np.sum(sigs["tf_option_c"][_bt0:]))
            print(f"  {'Entry signal bars:':<30}  {cur_sigs:>12}  {optb_sigs:>12}  {optb_sigs-cur_sigs:>+8d}  {optc_sigs:>12}  {optc_sigs-cur_sigs:>+8d}")
            # no-man's-land bars (above_ma30=T, slope=F)
            nml=sigs["above_ma30"][_bt0:]&~sigs["ma30_slope"][_bt0:]
            nml_label = "No-man's-land bars:"
            print(f"  {nml_label:<30}  {int(np.sum(nml)):>12}  (above MA30 but MA30 declining)")
            print()

    print(f"{'═'*W}")
    print("  DONE — copy ΔC values into the UI strategy card improvement banner")
    print(f"{'═'*W}\n")

if __name__ == "__main__":
    main()
