#!/usr/bin/env python3
"""Bull-capture vs bear-defense frontier for MSTR/MSTU across entry-gate variants
and the exposure levers (U1 threshold, D2 threshold, stop policy, exit patience)."""
import sys, warnings, itertools
warnings.filterwarnings("ignore")
sys.path.insert(0,"/home/user/btc-range-model")

import numpy as np, pandas as pd
import backtest_trailing_stop as T
from backtest_retune_12utc import base_arrays  # reuse the exact base-array builder

PERIODS = {"Bull":("2024-06-01","2025-06-05","2024-03-01"),
           "Bear":("2025-06-01","2026-05-31","2025-03-01")}
BH = {"MSTR":{"Bull":127,"Bear":-57}, "MSTU":{"Bull":190,"Bear":-92}}  # true-window B&H

def gate_entry(B, u1_thr, hb, d2_thr, gate):
    N=B["N"]; u1=(B["ehma3"]>u1_thr)&(B["hb3"]>=hb)
    d1=(B["lb3"]>=2)&(B["elma3"]>0.5); d2=B["ehma3"]<d2_thr
    clean=np.zeros(N,bool)
    for i in range(N):
        lo=max(0,i-7); clean[i]=not bool((d1[lo:i]|d2[lo:i]).any())
    above=B["above"]; bull=B["bull"]; v=B["v"]
    if gate=="bull_regime":  tf=u1&((bull^clean)|v)
    elif gate=="above_ma30": tf=u1&((above^clean)|v)
    elif gate=="pure_regime":tf=u1&(bull|(clean&~above)|v)
    elif gate=="bull_always":  # trend-follow: long whenever bull regime (u1 seeds, hold by regime)
        tf=u1&(bull|v)
    return dict(d2=d2,d3=B["d3"],bull_regime=bull,tf2_entry=tf,above_ma30=above,clean_7d=clean,v_recent=v)

def run(dates,px,sigs,stop,bt_start,stop_mode="fixed",exit_mode="regime"):
    d2=sigs["d2"];d3=sigs["d3"];bull=sigs["bull_regime"];tf=sigs["tf2_entry"]
    N=len(dates); _bt0=max(35,int(pd.DatetimeIndex(dates).searchsorted(bt_start)))
    nav=1e5;pos="CASH";qty=0.0;e_px=e_nav=e_dt=None;hwm=0.0
    from_sl=False;bars=0;trades=[];arr=np.full(N,np.nan)
    for i in range(N):
        p=px[i]
        if i<_bt0: arr[i]=1e5; continue
        if not np.isfinite(p) or p<=0: arr[i]=qty*px[i-1] if pos=="LONG" and i>0 else nav; continue
        if pos=="LONG":
            cur=qty*p; hwm=max(hwm,p)
            hard=e_px*(1-stop)
            stop_on = (stop_mode=="fixed") or (stop_mode=="bull_off" and not bull[i])
            fired=stop_on and p<=hard
            if exit_mode=="regime": sig=bool(d3[i] or (d2[i] and not bull[i]))
            elif exit_mode=="d3only": sig=bool(d3[i])            # patient everywhere
            if fired:
                nav=cur; trades.append((e_dt,dates[i],(p/e_px-1)*100,"SL")); pos="CASH";qty=0;from_sl=True;bars=0;hwm=0
            elif sig:
                nav=cur; trades.append((e_dt,dates[i],(p/e_px-1)*100,"sig")); pos="CASH";qty=0;from_sl=False;hwm=0
            else: nav=cur
        else:
            if from_sl: bars+=1
            _ex=d3[i] or (d2[i] and not bull[i])
            ok=(not from_sl) or bool(bull[i]) or bars>=10
            if tf[i] and ok and (from_sl or not _ex):
                qty=nav/p;e_px=p;e_dt=dates[i];e_nav=nav;pos="LONG";hwm=p;from_sl=False;bars=0
        arr[i]=qty*p if pos=="LONG" else nav
    if pos=="LONG" and np.isfinite(px[N-1]): arr[N-1]=qty*px[N-1]
    ns=pd.Series(arr[_bt0:],index=dates[_bt0:]).ffill()
    ret=(float(ns.iloc[-1])/1e5-1)*100
    rm=ns.cummax(); dd=float(((ns-rm)/rm*100).min())
    tim=100*sum((pd.Timestamp(b)-pd.Timestamp(a)).days for a,b,_,_ in trades)/max(1,(ns.index[-1]-ns.index[0]).days)
    return ret,dd,len(trades),tim

# preds once
rf=T.load_raw_features(); preds=T.build_preds_offline(rf)
px={"MSTR":T.load_asset("MSTR"),"MSTU":T.load_asset("MSTU")}
STOP={"MSTR":0.03,"MSTU":0.07}
data={}
for pn,(s,e,fs) in PERIODS.items():
    comp=T.prep(rf,preds,s,e); data[pn]=dict(dates=pd.DatetimeIndex(comp["target_date"]),B=base_arrays(comp),sd=pd.Timestamp(s))

def evalcfg(asset,gate,u1,d2,stop_mode,exit_mode,stop=None):
    out={}
    for pn in PERIODS:
        d=data[pn]; sigs=gate_entry(d["B"],u1,2,d2,gate)
        a=px[asset].reindex(d["dates"]).ffill().bfill().values.astype(float)
        out[pn]=run(d["dates"],a,sigs,stop or STOP[asset],d["sd"],stop_mode,exit_mode)
    return out

GATES=["bull_regime","pure_regime","above_ma30","bull_always"]
U1G=[0.5,0.7,0.9,1.1]; D2G=[-0.75,-1.0,-1.3,-1.6]
STOPMODES=["fixed","bull_off"]; EXITS=["regime","d3only"]

for asset in ["MSTR","MSTU"]:
    print(f"\n{'='*96}\n {asset}   (Bull B&H {BH[asset]['Bull']:+d}% · Bear B&H {BH[asset]['Bear']:+d}%)   [stop {int(STOP[asset]*100)}%]\n{'='*96}")
    rows=[]
    for gate,u1,d2,sm,ex in itertools.product(GATES,U1G,D2G,STOPMODES,EXITS):
        o=evalcfg(asset,gate,u1,d2,sm,ex)
        rows.append((gate,u1,d2,sm,ex,o["Bull"][0],o["Bull"][1],o["Bull"][3],o["Bear"][0],o["Bear"][1]))
    # current live baseline
    base=[r for r in rows if r[:5]==("bull_regime",1.1,-1.3,"fixed","regime")][0]
    print(f"  LIVE baseline: gate=bull_regime u1>1.1 d2<-1.3 fixed/regime -> Bull {base[5]:+.0f}% (DD{base[6]:.0f}%,TiM{base[7]:.0f}%) | Bear {base[8]:+.0f}% (DD{base[9]:.0f}%)")
    # top 8 by bull return that keep bear >= -10%
    good=[r for r in rows if r[8]>=-10]
    good.sort(key=lambda r:-r[5])
    print(f"\n  --- Max BULL capture with Bear >= -10% (top 10) ---")
    print(f"  {'gate':13} {'u1':>4} {'d2':>5} {'stop':>8} {'exit':>7} | {'Bull%':>6} {'bDD%':>6} {'TiM%':>5} | {'Bear%':>6} {'rDD%':>6}")
    for r in good[:10]:
        print(f"  {r[0]:13} {r[1]:>4} {r[2]:>5} {r[3]:>8} {r[4]:>7} | {r[5]:>6.0f} {r[6]:>6.0f} {r[7]:>5.0f} | {r[8]:>6.0f} {r[9]:>6.0f}")
    # absolute max bull ignoring bear
    rows.sort(key=lambda r:-r[5])
    print(f"\n  --- Absolute max BULL (any bear) top 3 ---")
    for r in rows[:3]:
        print(f"  {r[0]:13} {r[1]:>4} {r[2]:>5} {r[3]:>8} {r[4]:>7} | Bull {r[5]:+.0f}% (DD{r[6]:.0f}%,TiM{r[7]:.0f}%) | Bear {r[8]:+.0f}% (DD{r[9]:.0f}%)")
