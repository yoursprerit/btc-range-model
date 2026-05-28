"""
TF2 vs Buy & Hold — Full 2-Year Backtest
=========================================
Period: May 26, 2024 → May 26, 2026 (2 years)
Note: CT model trained through Sep 17, 2025.
      May 2024 → Sep 2025 = in-sample
      Sep 2025 → May 2026 = fully OOS

Metrics computed:
  Total return, CAGR, Max Drawdown, Sharpe, Sortino, Calmar,
  Win Rate, Profit Factor, Avg Win/Loss, Time in Market,
  Trades, Avg Duration, Volatility, Best/Worst Trade,
  Monthly return heatmap summary
"""
import sys, os, warnings, joblib, requests
warnings.filterwarnings("ignore")
from pathlib import Path
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import yfinance as yf

# ── Load model ────────────────────────────────────────────────────────────────
AD = joblib.load(str(_ROOT / "models" / "inference_assets_ct.joblib"))
print(f"Model loaded. feat_cols: {len(AD['feat_cols'])}")

# ── Data fetch (reuse from optimizer) ─────────────────────────────────────────
_MACRO_SYMS = {"eth":"ETH-USD","spx":"^GSPC","ndx":"^IXIC",
               "vix":"^VIX","gold":"GC=F","dxy":"DX-Y.NYB","tnx":"^TNX"}
_ONCHAIN = ["hash-rate","difficulty","n-transactions","miners-revenue",
            "n-unique-addresses","transaction-fees-usd","mempool-size",
            "estimated-transaction-volume-usd","market-cap",
            "avg-block-size","cost-per-transaction"]

def _yf_single(sym, start, end):
    d = yf.download(sym, start=start, end=end, progress=False, auto_adjust=False)
    if isinstance(d.columns, pd.MultiIndex): d.columns = [c[0] for c in d.columns]
    d.index = pd.DatetimeIndex(d.index).tz_localize(None).normalize()
    return d

def fetch_data(fetch_start, fetch_end):
    print(f"\nFetching {fetch_start} → {fetch_end} …")
    frames = {}
    d = _yf_single("BTC-USD", fetch_start, fetch_end)
    frames.update({"btc_close":d["Close"],"btc_high":d["High"],
                   "btc_low":d["Low"],"btc_volume":d["Volume"]})
    for name, sym in _MACRO_SYMS.items():
        try:
            frames[f"{name}_close"] = _yf_single(sym, fetch_start, fetch_end)["Close"]
        except: pass
    df = pd.DataFrame(frames).ffill(limit=5)
    df.index.name = "date"
    print("  On-chain …", end=" ", flush=True)
    ok = 0
    for m in _ONCHAIN:
        col = f"oc_{m.replace('-','_')}"
        try:
            r = requests.get(f"https://api.blockchain.info/charts/{m}",
                params={"timespan":"3years","format":"json","sampled":"true"}, timeout=20)
            vals = r.json().get("values",[])
            s = pd.Series({pd.Timestamp(v["x"],unit="s").normalize():v["y"] for v in vals},
                          name=col, dtype=float)
            s = s[~s.index.duplicated(keep="last")].sort_index()
            s.index = pd.DatetimeIndex(s.index).tz_localize(None)
            df[col] = s.reindex(df.index).ffill(limit=7)
            ok += 1
        except: pass
    print(f"{ok}/{len(_ONCHAIN)} OK")
    return df

def build_preds(df):
    c,h,l_,v = df["btc_close"],df["btc_high"],df["btc_low"],df["btc_volume"]
    ret = np.log(c).diff()
    f = pd.DataFrame(index=df.index)
    for k in [1,3,5,7,14,30]: f[f"ret_{k}"] = ret.rolling(k).sum()
    for k in [5,10,20,30]:    f[f"vol_{k}"] = ret.rolling(k).std()
    prev_c = c.shift(1)
    tr = pd.concat([(h-l_),(h-prev_c).abs(),(l_-prev_c).abs()],axis=1).max(axis=1)
    for k in [7,14,30]: f[f"atr_{k}"] = tr.rolling(k).mean()/c
    f["range_today"]=(h-l_)/c; f["range_ma7"]=((h-l_)/c).rolling(7).mean()
    f["range_ma30"]=((h-l_)/c).rolling(30).mean(); f["range_std30"]=((h-l_)/c).rolling(30).std()
    g=c.diff().clip(lower=0).rolling(14).mean(); ls=(-c.diff().clip(upper=0)).rolling(14).mean()
    f["rsi_14"]=100-100/(1+g/ls.replace(0,np.nan))
    e12=c.ewm(span=12,adjust=False).mean(); e26=c.ewm(span=26,adjust=False).mean()
    macd=e12-e26
    f["macd"]=macd/c; f["macd_sig"]=macd.ewm(span=9,adjust=False).mean()/c
    f["macd_hist"]=(macd-macd.ewm(span=9,adjust=False).mean())/c
    ma20=c.rolling(20).mean(); sd20=c.rolling(20).std()
    f["bb_width"]=(4*sd20)/ma20; f["dist_hi_30"]=c/c.rolling(30).max()-1
    f["dist_lo_30"]=c/c.rolling(30).min()-1; f["dist_hi_90"]=c/c.rolling(90).max()-1
    f["vol_chg_1"]=np.log(v).diff()
    f["vol_z_20"]=(np.log(v)-np.log(v).rolling(20).mean())/np.log(v).rolling(20).std()
    f["vol_ma_ratio"]=v/v.rolling(20).mean()
    dow=df.index.dayofweek
    for i in range(6): f[f"dow_{i}"]=(dow==i).astype(float)
    for nm,col in [("spx","spx_close"),("ndx","ndx_close"),("vix","vix_close"),
                   ("gold","gold_close"),("dxy","dxy_close"),("tnx","tnx_close"),
                   ("eth","eth_close")]:
        if col not in df.columns: continue
        s=df[col]; lr=np.log(s).diff()
        for k in [1,5,20]: f[f"{nm}_ret_{k}"]=lr.rolling(k).sum()
        f[f"{nm}_vol_20"]=lr.rolling(20).std()
    for cn,cc in [("spx","spx_close"),("ndx","ndx_close"),
                  ("gold","gold_close"),("dxy","dxy_close")]:
        if cc in df.columns:
            f[f"btc_{cn}_corr_30"]=ret.rolling(30).corr(np.log(df[cc]).diff())
    for col in [x for x in df.columns if x.startswith("oc_")]:
        s=df[col].astype(float); sl=np.log(s.replace(0,np.nan))
        f[f"{col}_d1"]=sl.diff(1); f[f"{col}_d7"]=sl.diff(7)
        f[f"{col}_z30"]=(sl-sl.rolling(30).mean())/sl.rolling(30).std()
    nh,nl=h.shift(-1),l_.shift(-1); yh=(nh-c)/c; yl=(c-nl)/c
    f["y_hi_ema3"]=yh.shift(1).ewm(span=3,adjust=False).mean()
    f["y_lo_ema3"]=yl.shift(1).ewm(span=3,adjust=False).mean()
    f["y_hi_ema7"]=yh.shift(1).ewm(span=7,adjust=False).mean()
    f["y_lo_ema7"]=yl.shift(1).ewm(span=7,adjust=False).mean()
    p3h=h.shift(1).rolling(3).max(); p3l=l_.shift(1).rolling(3).min()
    f["above_3d_high"]=(c>p3h).astype(float); f["below_3d_low"]=(c<p3l).astype(float)
    f["bo_strength_up"]=(c/p3h-1).clip(lower=0); f["bo_strength_dn"]=(1-c/p3l).clip(lower=0)
    ya=yh.shift(1); yb=yl.shift(1)
    f["y_hi_surprise"]=ya-ya.ewm(span=7,adjust=False).mean()
    f["y_lo_surprise"]=yb-yb.ewm(span=7,adjust=False).mean()
    nr=ret.clip(upper=0); f["dn_vol_5"]=nr.rolling(5).std(); f["dn_vol_20"]=nr.rolling(20).std()
    sma50=c.rolling(50).mean(); f["below_sma50"]=(c<sma50).astype(float)
    f["below_sma50_5d"]=f["below_sma50"].rolling(5).min().fillna(0)
    fc=AD["feat_cols"]
    f=f.replace([np.inf,-np.inf],np.nan)
    for col in fc:
        if col not in f.columns: f[col]=np.nan
    F=f[fc].dropna()
    if F.empty: raise RuntimeError("Feature matrix empty")
    print(f"  Predictions: {len(F)} rows")
    if AD.get("ensemble") and AD.get("constituents"):
        yhi=np.mean([con["m_hi"].predict(F) for con in AD["constituents"]],axis=0)
        ylo=np.mean([con["m_lo"].predict(F) for con in AD["constituents"]],axis=0)
        if AD.get("blended") and float(AD.get("alpha",1.0))<1.0:
            a=float(AD["alpha"])
            yhi=a*yhi+(1-a)*float(AD.get("mu_hi",0))
            ylo=a*ylo+(1-a)*float(AD.get("mu_lo",0))
    else:
        yhi=AD["hi_model"].predict(F); ylo=AD["lo_model"].predict(F)
    cv=c.reindex(F.index).values
    ph=cv*(1+np.clip(yhi,0,None)); pl=cv*(1-np.clip(ylo,0,None))
    idx=np.asarray(F.index,dtype="datetime64[ns]")
    nd=np.empty(len(F),dtype="datetime64[ns]")
    nd[:-1]=idx[1:]; nd[-1]=idx[-1]+np.timedelta64(1,"D")
    res=pd.DataFrame({"close_asof":cv,"pred_high":ph,"pred_low":pl},
                     index=pd.DatetimeIndex(nd,name="target_date"))
    return res[~res.index.duplicated(keep="last")]

# ── Backtest engine ────────────────────────────────────────────────────────────
def backtest(df_raw, preds, start_iso, end_iso, strategy="TF2", cap=100_000.0):
    WARMUP=35
    sd,ed=pd.Timestamp(start_iso),pd.Timestamp(end_iso)
    p=preds.loc[(preds.index>=sd-pd.Timedelta(days=45))&(preds.index<=ed)].copy()
    c=df_raw["btc_close"]; h=df_raw["btc_high"]; lo=df_raw["btc_low"]
    p["ah"]=h.reindex(p.index).values; p["al"]=lo.reindex(p.index).values
    p["ac"]=c.reindex(p.index).values
    comp=p.dropna(subset=["ah","al","ac"]).reset_index()
    comp=comp[comp["target_date"]>=sd].reset_index(drop=True)
    N=len(comp)
    if N<WARMUP+3: return None
    dates=pd.DatetimeIndex(comp["target_date"])
    ca=comp["close_asof"].values.astype(float)
    ep=comp["ac"].values.astype(float)
    ph=comp["pred_high"].values.astype(float)
    pl=comp["pred_low"].values.astype(float)
    ah=comp["ah"].values.astype(float)
    al=comp["al"].values.astype(float)
    ehi=(ah-ph)/ca*100; elo=(pl-al)/ca*100
    hb=(ah>ph).astype(int); lb=(al<pl).astype(int)
    ehma3=np.zeros(N); elma3=np.zeros(N)
    hb3=np.zeros(N,dtype=int); lb3=np.zeros(N,dtype=int)
    for i in range(N):
        s=max(0,i-2)
        ehma3[i]=np.mean(ehi[s:i+1]); elma3[i]=np.mean(elo[s:i+1])
        hb3[i]=int(np.sum(hb[s:i+1])); lb3[i]=int(np.sum(lb[s:i+1]))
    u1=(ehma3>0.5)&(hb3>=2); d1=(lb3>=2)&(elma3>0.5)
    d2=ehma3<-1.0
    d3=np.zeros(N,dtype=bool)
    for i in range(1,N):
        con=0
        for k in range(i-1,-1,-1):
            if hb[k]: con+=1
            else: break
        if con>=3 and lb[i]: d3[i]=True
    ma30=np.full(N,np.nan)
    for i in range(N):
        w=min(30,i+1); ma30[i]=np.mean(ca[max(0,i-w+1):i+1])
    abv=ca>ma30
    slp=np.zeros(N,dtype=bool)
    for i in range(5,N):
        if np.isfinite(ma30[i]) and np.isfinite(ma30[i-5]):
            slp[i]=ma30[i]>ma30[i-5]
    bull=abv&slp
    c10=np.zeros(N,dtype=bool)
    for i in range(N):
        li=max(0,i-10); c10[i]=not bool(np.any(d1[li:i]|d2[li:i]))
    entry=u1&(abv|c10)
    nav=cap; pos="CASH"; qty=0.0
    ep_r=en=ed_r=et=None
    trades=[]; nav_arr=np.full(N,np.nan)
    for i in range(N):
        si=i-1; price=ep[i]
        if i<WARMUP: nav_arr[i]=cap; continue
        if pos=="LONG":
            cur=qty*price
            if si>=0:
                if strategy=="TF2":
                    ex=bool(d3[si] or (d2[si] and not bull[si]))
                    xl="D3" if d3[si] else "D2"
                else:
                    ex=bool(d2[si] or d3[si])
                    xl="D3" if d3[si] else "D2"
            else: ex=False; xl="?"
            if ex:
                nav=cur
                trades.append(dict(entry_date=ed_r,entry_price=ep_r,
                    exit_date=dates[i],exit_price=price,
                    pnl_pct=(price/ep_r-1)*100,pnl_abs=nav-en,
                    exit_signal=xl,duration_days=(dates[i]-ed_r).days,
                    regime="BULL" if (si>=0 and bull[si]) else "BEAR"))
                pos="CASH"; qty=0.0
            else: nav=cur
        else:
            if si>=0 and entry[si]:
                qty=nav/price; ep_r=price; ed_r=dates[i]; en=nav; pos="LONG"
                et=("U1+↑MA30+c10d" if (abv[si] and c10[si]) else
                    "U1+↑MA30" if abv[si] else "U1+c10d")
        nav_arr[i]=qty*price if pos=="LONG" else nav
    if pos=="LONG": nav_arr[N-1]=qty*ep[N-1]
    navs=pd.Series(nav_arr[WARMUP:],index=dates[WARMUP:]).ffill()
    bhs=pd.Series(cap*ep[WARMUP:]/ep[WARMUP],index=dates[WARMUP:])
    return dict(strategy=strategy,trades=trades,nav=navs,bh=bhs,
                open=pos=="LONG",open_price=ep_r,open_date=ed_r,open_nav=en)

# ── Metrics engine ────────────────────────────────────────────────────────────
def metrics(res, label, rf_annual=0.045):
    nav=res["nav"]; bh=res["bh"]; trades=res["trades"]
    cap=100_000.0
    # Returns
    total_ret=(nav.iloc[-1]/cap-1)*100
    bh_ret=(bh.iloc[-1]/cap-1)*100
    n_years=(nav.index[-1]-nav.index[0]).days/365.25
    cagr=((nav.iloc[-1]/cap)**(1/n_years)-1)*100 if n_years>0 else 0
    bh_cagr=((bh.iloc[-1]/cap)**(1/n_years)-1)*100 if n_years>0 else 0
    # Daily returns (forward-fill so weekends are flat)
    dr=nav.pct_change().fillna(0)
    bh_dr=bh.pct_change().fillna(0)
    # Volatility (annualised, 252 trading days)
    vol_ann=dr.std()*np.sqrt(252)*100
    bh_vol=bh_dr.std()*np.sqrt(252)*100
    # Sharpe (annualised, daily Sharpe × √252)
    rf_daily=(1+rf_annual)**(1/252)-1
    excess=dr-rf_daily
    sharpe=(excess.mean()/excess.std()*np.sqrt(252)) if excess.std()>0 else 0
    bh_excess=bh_dr-rf_daily
    bh_sharpe=(bh_excess.mean()/bh_excess.std()*np.sqrt(252)) if bh_excess.std()>0 else 0
    # Sortino (penalise only downside)
    down=dr[dr<rf_daily]-rf_daily
    down_std=down.std()*np.sqrt(252) if len(down)>1 else 1e-9
    sortino=excess.mean()*252/down_std if down_std>0 else 0
    bh_down=bh_dr[bh_dr<rf_daily]-rf_daily
    bh_down_std=bh_down.std()*np.sqrt(252) if len(bh_down)>1 else 1e-9
    bh_sortino=bh_excess.mean()*252/bh_down_std if bh_down_std>0 else 0
    # Max drawdown
    rm=nav.cummax(); dd=(nav-rm)/rm*100
    max_dd=dd.min()
    rm_bh=bh.cummax(); dd_bh=(bh-rm_bh)/rm_bh*100
    bh_max_dd=dd_bh.min()
    # Calmar
    calmar=cagr/abs(max_dd) if max_dd!=0 else 0
    bh_calmar=bh_cagr/abs(bh_max_dd) if bh_max_dd!=0 else 0
    # Recovery time (days from trough to new high)
    dd_idx=dd.idxmin()
    after_trough=nav[dd_idx:]
    recovery_mask=after_trough>=nav[:dd_idx].max() if len(nav[:dd_idx])>0 else pd.Series(dtype=bool)
    recovery_days=int((recovery_mask.index[recovery_mask.argmax()]-dd_idx).days) if recovery_mask.any() else None
    # Trade metrics
    wins=[t for t in trades if t["pnl_pct"]>0]
    losses=[t for t in trades if t["pnl_pct"]<=0]
    win_rate=100*len(wins)/len(trades) if trades else 0
    avg_win=np.mean([t["pnl_pct"] for t in wins]) if wins else 0
    avg_loss=np.mean([t["pnl_pct"] for t in losses]) if losses else 0
    gross_profit=sum(t["pnl_abs"] for t in wins) if wins else 0
    gross_loss=abs(sum(t["pnl_abs"] for t in losses)) if losses else 1
    profit_factor=gross_profit/gross_loss if gross_loss>0 else float("inf")
    best=max([t["pnl_pct"] for t in trades]) if trades else 0
    worst=min([t["pnl_pct"] for t in trades]) if trades else 0
    avg_dur=np.mean([t["duration_days"] for t in trades]) if trades else 0
    days_in=sum(t["duration_days"] for t in trades)
    total_days=max(1,(nav.index[-1]-nav.index[0]).days)
    time_in=100*days_in/total_days
    # Monthly returns
    monthly_nav=nav.resample("ME").last()
    monthly_ret=monthly_nav.pct_change().fillna((monthly_nav.iloc[0]/cap)-1)*100
    monthly_bh=bh.resample("ME").last()
    monthly_bh_ret=monthly_bh.pct_change().fillna((monthly_bh.iloc[0]/cap)-1)*100
    # Positive months
    pos_months=int((monthly_ret>0).sum())
    neg_months=int((monthly_ret<=0).sum())
    best_month=float(monthly_ret.max())
    worst_month=float(monthly_ret.min())
    return dict(
        label=label,
        period=f"{nav.index[0].date()} → {nav.index[-1].date()}",
        n_years=n_years,
        final_nav=nav.iloc[-1], bh_final=bh.iloc[-1],
        total_ret=total_ret, bh_ret=bh_ret,
        cagr=cagr, bh_cagr=bh_cagr,
        vol_ann=vol_ann, bh_vol=bh_vol,
        sharpe=sharpe, bh_sharpe=bh_sharpe,
        sortino=sortino, bh_sortino=bh_sortino,
        max_dd=max_dd, bh_max_dd=bh_max_dd,
        calmar=calmar, bh_calmar=bh_calmar,
        recovery_days=recovery_days,
        n_trades=len(trades), n_wins=len(wins), n_losses=len(losses),
        win_rate=win_rate, avg_win=avg_win, avg_loss=avg_loss,
        profit_factor=profit_factor,
        best_trade=best, worst_trade=worst,
        avg_duration=avg_dur, time_in=time_in,
        pos_months=pos_months, neg_months=neg_months,
        best_month=best_month, worst_month=worst_month,
        alpha_abs=nav.iloc[-1]-bh.iloc[-1],
        monthly_ret=monthly_ret, monthly_bh_ret=monthly_bh_ret,
        open=res["open"], open_price=res.get("open_price"),
        open_date=res.get("open_date"),
        trades=trades,
    )

def print_report(m):
    W=70
    def row(lbl,strat,bh,unit="",better="higher"):
        sv=f"{strat:.2f}{unit}" if isinstance(strat,float) else str(strat)
        bv=f"{bh:.2f}{unit}"   if isinstance(bh,  float) else str(bh)
        if isinstance(strat,float) and isinstance(bh,float):
            if better=="higher": marker="▲" if strat>bh else ("▼" if strat<bh else "=")
            else:                marker="▲" if strat<bh else ("▼" if strat>bh else "=")
        else: marker=""
        print(f"  {lbl:<38} {sv:>12}  {bv:>12}  {marker}")
    def section(title):
        print(f"\n  {'─'*W}")
        print(f"  {title}")
        print(f"  {'─'*W}")
        print(f"  {'Metric':<38} {'TF2':>12}  {'Buy & Hold':>12}")
        print(f"  {'─'*W}")

    print(f"\n{'═'*72}")
    print(f"  TF2 STRATEGY vs BUY & HOLD — FULL 2-YEAR ANALYSIS")
    print(f"  Period : {m['period']}")
    print(f"  Capital: $100,000  |  RF rate: 4.5%  |  ⚠️ pre-Sep 2025 = in-sample")
    print(f"{'═'*72}")

    section("RETURN METRICS")
    row("Total return",          m["total_ret"],   m["bh_ret"],   "%")
    row("CAGR (annualised)",     m["cagr"],        m["bh_cagr"],  "%")
    row("Final NAV",             m["final_nav"]/1e3, m["bh_final"]/1e3, "k")
    row("Alpha vs B&H",         (m["final_nav"]-m["bh_final"])/1e3, 0.0, "k")

    section("RISK METRICS")
    row("Annualised volatility", m["vol_ann"],     m["bh_vol"],   "%", "lower")
    row("Max drawdown",          m["max_dd"],      m["bh_max_dd"],"%", "lower")
    row("DD recovery (days)",
        m["recovery_days"] if m["recovery_days"] else "not yet",
        "n/a", better="lower")

    section("RISK-ADJUSTED RETURN METRICS")
    row("Sharpe ratio",          m["sharpe"],      m["bh_sharpe"])
    row("Sortino ratio",         m["sortino"],     m["bh_sortino"])
    row("Calmar ratio",          m["calmar"],      m["bh_calmar"])

    section("TRADE METRICS  (strategy only)")
    print(f"  {'Closed trades':<38} {m['n_trades']:>12}")
    print(f"  {'Win rate':<38} {m['win_rate']:>11.1f}%")
    print(f"  {'Avg win / Avg loss':<38} {m['avg_win']:>+9.1f}%  /  {m['avg_loss']:>+.1f}%")
    print(f"  {'Profit factor':<38} {m['profit_factor']:>12.2f}")
    print(f"  {'Best trade':<38} {m['best_trade']:>+11.1f}%")
    print(f"  {'Worst trade':<38} {m['worst_trade']:>+11.1f}%")
    print(f"  {'Avg trade duration':<38} {m['avg_duration']:>9.0f} days")
    print(f"  {'Time in market':<38} {m['time_in']:>11.0f}%")
    if m["open"]:
        op_str = f"YES @ ${m['open_price']:,.0f}"
        print(f"  {'Open position (at period end)':<38} {op_str:>12}")

    section("MONTHLY RETURNS")
    print(f"  {'Positive months':<38} {m['pos_months']:>12}")
    print(f"  {'Negative months':<38} {m['neg_months']:>12}")
    print(f"  {'Best month':<38} {m['best_month']:>+11.1f}%  {m['monthly_bh_ret'].max():>+11.1f}%")
    print(f"  {'Worst month':<38} {m['worst_month']:>+11.1f}%  {m['monthly_bh_ret'].min():>+11.1f}%")

    print(f"\n  {'─'*W}")
    print("  MONTHLY RETURN TABLE  (TF2 | B&H)")
    print(f"  {'─'*W}")
    combined=pd.DataFrame({"TF2":m["monthly_ret"],"B&H":m["monthly_bh_ret"]})
    for dt,row2 in combined.iterrows():
        tf2=row2["TF2"]; bh2=row2["B&H"]
        bar_tf2="█"*max(0,int(abs(tf2)/2))
        bar_bh ="█"*max(0,int(abs(bh2)/2))
        sign_tf2="+" if tf2>0 else "-"
        sign_bh ="+" if bh2>0 else "-"
        print(f"  {dt.strftime('%b %Y')}  TF2:{tf2:>+6.1f}%  B&H:{bh2:>+6.1f}%  "
              f"{'▲' if tf2>bh2 else '▼'}")

    print(f"\n{'═'*72}")
    print("  TRADE LOG (all closed trades)")
    print(f"{'═'*72}")
    print(f"  {'#':>3}  {'Entry':12}  {'Exit':12}  {'P&L':>7}  {'Days':>5}  Regime  Exit")
    oos_start = pd.Timestamp("2026-03-01")
    for j,t in enumerate(m["trades"],1):
        tag = " (OOS)" if t["entry_date"] >= oos_start else " (IS) "
        mk  = "✓" if t["pnl_pct"]>0 else "✗"
        print(f"  {j:>3}  {str(t['entry_date'].date()):12}  "
              f"{str(t['exit_date'].date()):12}  "
              f"{mk}{t['pnl_pct']:>+5.1f}%  "
              f"{t['duration_days']:>5}  "
              f"{t.get('regime','?'):<7} {t['exit_signal']}{tag}")
    if m["open"]:
        tag=" (OOS)" if m["open_date"]>=oos_start else " (IS) "
        print(f"  {'':>3}  {str(m['open_date'].date()):12}  "
              f"{'⏳ OPEN':12}  {'(open)':>7}  {'–':>5}{tag}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
BACKTEST_START = "2024-05-26"   # 2 years ago from today
BACKTEST_END   = "2026-05-17"   # last complete daily bar available
FETCH_START    = "2024-02-01"   # 90+ days extra for warmup features
FETCH_END      = "2026-05-26"

print(f"\nRunning 2-year backtest: {BACKTEST_START} → {BACKTEST_END}")
df = fetch_data(FETCH_START, FETCH_END)
print("Building CT predictions …", end=" ", flush=True)
preds = build_preds(df)
print(f"→ {len(preds)} bars")

print("\nRunning TF2 backtest …")
r_tf2 = backtest(df, preds, BACKTEST_START, BACKTEST_END, "TF2")
print("Running TF1 backtest …")
r_tf1 = backtest(df, preds, BACKTEST_START, BACKTEST_END, "TF1")

m_tf2 = metrics(r_tf2, "TF2")
m_tf1 = metrics(r_tf1, "TF1")

print_report(m_tf2)

# Quick TF1 comparison summary
print(f"\n{'═'*72}")
print("  TF1 vs TF2 QUICK COMPARISON (same 2-year period)")
print(f"{'═'*72}")
print(f"  {'Metric':<30} {'TF1':>12}  {'TF2':>12}  {'B&H':>12}")
print(f"  {'─'*65}")
rows_cmp = [
    ("Total return",      m_tf1["total_ret"],  m_tf2["total_ret"],  m_tf2["bh_ret"],    "%"),
    ("CAGR",              m_tf1["cagr"],       m_tf2["cagr"],       m_tf2["bh_cagr"],   "%"),
    ("Sharpe ratio",      m_tf1["sharpe"],     m_tf2["sharpe"],     m_tf2["bh_sharpe"], ""),
    ("Sortino ratio",     m_tf1["sortino"],    m_tf2["sortino"],    m_tf2["bh_sortino"],""),
    ("Max drawdown",      m_tf1["max_dd"],     m_tf2["max_dd"],     m_tf2["bh_max_dd"], "%"),
    ("Calmar ratio",      m_tf1["calmar"],     m_tf2["calmar"],     m_tf2["bh_calmar"], ""),
    ("# trades",          m_tf1["n_trades"],   m_tf2["n_trades"],   "n/a",              ""),
    ("Win rate",          m_tf1["win_rate"],   m_tf2["win_rate"],   "n/a",              "%"),
    ("Time in market",    m_tf1["time_in"],    m_tf2["time_in"],    100.0,              "%"),
    ("Alpha vs B&H ($k)", m_tf1["alpha_abs"]/1e3, m_tf2["alpha_abs"]/1e3, 0.0,         "k"),
]
for r in rows_cmp:
    lbl,v1,v2,bh2,unit=r
    fv=lambda x: f"{x:+.2f}{unit}" if isinstance(x,float) else str(x)
    print(f"  {lbl:<30} {fv(v1):>12}  {fv(v2):>12}  {fv(bh2):>12}")

print("\n✓ Done.")
