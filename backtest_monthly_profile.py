"""
Month-by-Month Portfolio Profile: TF2 vs TF1 vs Buy & Hold
============================================================
Produces:
  1. Console table: month | TF2 $NAV | TF2 monthly% | TF1 $NAV | TF1 monthly% | BH $NAV | BH monthly%
  2. PNG plot: equity curves + monthly return bars + drawdown panel
"""
import sys, os, warnings, joblib, requests
warnings.filterwarnings("ignore")
from pathlib import Path
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import yfinance as yf

# ── Load model ────────────────────────────────────────────────────────────────
AD = joblib.load(str(_ROOT / "models" / "inference_assets_ct.joblib"))

# ── Data fetch ────────────────────────────────────────────────────────────────
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
    print(f"Fetching {fetch_start} → {fetch_end} …")
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
    ep_r=en=ed_r=None
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
        nav_arr[i]=qty*price if pos=="LONG" else nav
    if pos=="LONG": nav_arr[N-1]=qty*ep[N-1]
    navs=pd.Series(nav_arr[WARMUP:],index=dates[WARMUP:]).ffill()
    bhs=pd.Series(cap*ep[WARMUP:]/ep[WARMUP],index=dates[WARMUP:])
    return dict(strategy=strategy,trades=trades,nav=navs,bh=bhs,
                open=pos=="LONG",open_price=ep_r,open_date=ed_r)

# ── Run ────────────────────────────────────────────────────────────────────────
BACKTEST_START = "2024-05-26"
BACKTEST_END   = "2026-05-17"
FETCH_START    = "2024-02-01"
FETCH_END      = "2026-05-26"
CAP = 100_000.0
OOS_START = pd.Timestamp("2026-03-01")

df  = fetch_data(FETCH_START, FETCH_END)
print("Building CT predictions …", end=" ", flush=True)
preds = build_preds(df)
print(f"→ {len(preds)} bars")

print("Running backtests …")
r_tf2 = backtest(df, preds, BACKTEST_START, BACKTEST_END, "TF2")
r_tf1 = backtest(df, preds, BACKTEST_START, BACKTEST_END, "TF1")

nav_tf2 = r_tf2["nav"]
nav_tf1 = r_tf1["nav"]
nav_bh  = r_tf2["bh"]

# ── Monthly table ──────────────────────────────────────────────────────────────
# Resample to month-end NAV
def monthly_profile(nav, cap=CAP):
    mn = nav.resample("ME").last()
    ret = mn.pct_change() * 100
    # First month return = relative to starting capital
    ret.iloc[0] = (mn.iloc[0] / cap - 1) * 100
    return mn, ret

mn_tf2, mr_tf2 = monthly_profile(nav_tf2)
mn_tf1, mr_tf1 = monthly_profile(nav_tf1)
mn_bh,  mr_bh  = monthly_profile(nav_bh)

# Combine on common monthly index
idx = mn_tf2.index.union(mn_tf1.index).union(mn_bh.index)
table = pd.DataFrame({
    "TF2_NAV":  mn_tf2.reindex(idx),
    "TF2_Ret":  mr_tf2.reindex(idx),
    "TF1_NAV":  mn_tf1.reindex(idx),
    "TF1_Ret":  mr_tf1.reindex(idx),
    "BH_NAV":   mn_bh.reindex(idx),
    "BH_Ret":   mr_bh.reindex(idx),
}).dropna(how="all")

# ── Print table ────────────────────────────────────────────────────────────────
print(f"\n{'═'*90}")
print("  MONTH-BY-MONTH PORTFOLIO PROFILE  |  $100,000 initial capital")
print(f"  ⚠️  Pre-Sep 18 2025 = IN-SAMPLE  |  Sep 18 2025 onwards = OOS")
print(f"{'═'*90}")
hdr = (f"  {'Month':<10}  {'TF2 $NAV':>12}  {'TF2 %':>8}  "
       f"{'TF1 $NAV':>12}  {'TF1 %':>8}  "
       f"{'B&H $NAV':>12}  {'B&H %':>8}  {'Leader':>8}")
print(hdr)
print(f"  {'─'*86}")

total_tf2 = total_tf1 = total_bh = 0.0
for dt, row in table.iterrows():
    tf2_nav = row["TF2_NAV"]
    tf1_nav = row["TF1_NAV"]
    bh_nav  = row["BH_NAV"]
    tf2_ret = row["TF2_Ret"]
    tf1_ret = row["TF1_Ret"]
    bh_ret  = row["BH_Ret"]

    # Leader of the month
    best_ret = max(tf2_ret, tf1_ret, bh_ret)
    if best_ret == tf2_ret:   leader = "TF2 🏆"
    elif best_ret == tf1_ret: leader = "TF1 🏆"
    else:                     leader = "B&H 🏆"

    oos_tag = " *OOS*" if dt >= OOS_START else ""
    sign_tf2 = "▲" if tf2_ret > 0 else "▼"
    sign_tf1 = "▲" if tf1_ret > 0 else "▼"
    sign_bh  = "▲" if bh_ret  > 0 else "▼"

    print(f"  {dt.strftime('%b %Y'):<10}  "
          f"${tf2_nav:>11,.0f}  {sign_tf2}{tf2_ret:>+6.1f}%  "
          f"${tf1_nav:>11,.0f}  {sign_tf1}{tf1_ret:>+6.1f}%  "
          f"${bh_nav:>11,.0f}  {sign_bh}{bh_ret:>+6.1f}%  "
          f"{leader}{oos_tag}")

print(f"  {'─'*86}")
# Final row
final_tf2 = nav_tf2.iloc[-1]
final_tf1 = nav_tf1.iloc[-1]
final_bh  = nav_bh.iloc[-1]
print(f"  {'FINAL':10}  "
      f"${final_tf2:>11,.0f}  {(final_tf2/CAP-1)*100:>+6.1f}%  "
      f"${final_tf1:>11,.0f}  {(final_tf1/CAP-1)*100:>+6.1f}%  "
      f"${final_bh:>11,.0f}  {(final_bh/CAP-1)*100:>+6.1f}%")
print(f"{'═'*90}\n")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("#0d1117")

gs = fig.add_gridspec(4, 1, height_ratios=[3, 1.4, 1.4, 1.2], hspace=0.08)
ax1 = fig.add_subplot(gs[0])   # equity curve
ax2 = fig.add_subplot(gs[1], sharex=ax1)   # TF2 monthly bars
ax3 = fig.add_subplot(gs[2], sharex=ax1)   # B&H monthly bars
ax4 = fig.add_subplot(gs[3], sharex=ax1)   # drawdown

DARK = "#0d1117"
for ax in [ax1, ax2, ax3, ax4]:
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#8b949e", labelsize=9)
    ax.spines[["top","right","left","bottom"]].set_color("#30363d")
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    ax.yaxis.label.set_color("#8b949e")
    ax.xaxis.label.set_color("#8b949e")

C_TF2 = "#58a6ff"   # blue
C_TF1 = "#f0883e"   # orange
C_BH  = "#3fb950"   # green
C_OOS = "#6e40c9"   # purple shade for OOS shading

# ── Panel 1: Equity curves ────────────────────────────────────────────────────
ax1.axvline(OOS_START, color=C_OOS, lw=1.2, ls="--", alpha=0.7, label="OOS start (Aug 2025)")
ax1.axhspan(0, CAP, alpha=0.04, color="white")  # starting capital band

ax1.plot(nav_bh.index,  nav_bh.values,  color=C_BH,  lw=1.5, alpha=0.85, label=f"Buy & Hold  (final: ${final_bh:,.0f})")
ax1.plot(nav_tf1.index, nav_tf1.values, color=C_TF1, lw=1.5, alpha=0.85, label=f"TF1 Strategy (final: ${final_tf1:,.0f})")
ax1.plot(nav_tf2.index, nav_tf2.values, color=C_TF2, lw=2.2, alpha=1.0,  label=f"TF2 Strategy (final: ${final_tf2:,.0f})")
ax1.axhline(CAP, color="#484f58", lw=0.8, ls=":")

# Entry/exit markers for TF2 trades
for t in r_tf2["trades"]:
    ax1.axvspan(t["entry_date"], t["exit_date"],
                alpha=0.06, color=C_TF2, zorder=0)

ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax1.set_ylabel("Portfolio Value", fontsize=10)
ax1.legend(loc="upper left", framealpha=0.2, fontsize=9,
           facecolor="#161b22", edgecolor="#30363d", labelcolor="white")
ax1.set_title("Month-by-Month Portfolio Profile  |  $100k Initial Capital  (⚠️ pre-Aug 2025 = in-sample)",
              color="white", fontsize=12, fontweight="bold", pad=10)
ax1.grid(axis="y", color="#21262d", lw=0.6)
ax1.grid(axis="x", color="#21262d", lw=0.4)

# ── Panel 2: TF2 monthly return bars ─────────────────────────────────────────
bar_dates = [dt.to_pydatetime() for dt in table.index]
bar_w = 20  # days width
bars2 = ax2.bar(bar_dates,
                table["TF2_Ret"].values,
                width=bar_w,
                color=[C_TF2 if v >= 0 else "#da3633" for v in table["TF2_Ret"].values],
                alpha=0.85, edgecolor="none")
ax2.axhline(0, color="#484f58", lw=0.8)
ax2.axvline(OOS_START, color=C_OOS, lw=1.2, ls="--", alpha=0.7)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.0f}%"))
ax2.set_ylabel("TF2 Monthly\nReturn", fontsize=8, color=C_TF2)
ax2.tick_params(axis="y", labelcolor=C_TF2)
ax2.grid(axis="y", color="#21262d", lw=0.6)
# Add value labels on bars
for bar, val in zip(bars2, table["TF2_Ret"].values):
    if abs(val) > 3:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.3 if val >= 0 else -1.2),
                 f"{val:+.0f}%", ha="center", va="bottom" if val >= 0 else "top",
                 fontsize=6.5, color="white", alpha=0.85)

# ── Panel 3: B&H monthly return bars ─────────────────────────────────────────
bars3 = ax3.bar(bar_dates,
                table["BH_Ret"].values,
                width=bar_w,
                color=[C_BH if v >= 0 else "#da3633" for v in table["BH_Ret"].values],
                alpha=0.7, edgecolor="none")
ax3.axhline(0, color="#484f58", lw=0.8)
ax3.axvline(OOS_START, color=C_OOS, lw=1.2, ls="--", alpha=0.7)
ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.0f}%"))
ax3.set_ylabel("B&H Monthly\nReturn", fontsize=8, color=C_BH)
ax3.tick_params(axis="y", labelcolor=C_BH)
ax3.grid(axis="y", color="#21262d", lw=0.6)
for bar, val in zip(bars3, table["BH_Ret"].values):
    if abs(val) > 5:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.4 if val >= 0 else -1.5),
                 f"{val:+.0f}%", ha="center", va="bottom" if val >= 0 else "top",
                 fontsize=6.5, color="white", alpha=0.85)

# ── Panel 4: Drawdown ─────────────────────────────────────────────────────────
def drawdown_series(nav):
    rm = nav.cummax()
    return (nav - rm) / rm * 100

dd_tf2 = drawdown_series(nav_tf2)
dd_tf1 = drawdown_series(nav_tf1)
dd_bh  = drawdown_series(nav_bh)

ax4.fill_between(dd_tf2.index, dd_tf2.values, 0, color=C_TF2, alpha=0.35, label="TF2 DD")
ax4.fill_between(dd_bh.index,  dd_bh.values,  0, color=C_BH,  alpha=0.25, label="B&H DD")
ax4.plot(dd_tf2.index, dd_tf2.values, color=C_TF2, lw=1.0, alpha=0.9)
ax4.plot(dd_bh.index,  dd_bh.values,  color=C_BH,  lw=1.0, alpha=0.7)
ax4.axhline(0, color="#484f58", lw=0.8)
ax4.axvline(OOS_START, color=C_OOS, lw=1.2, ls="--", alpha=0.7)
ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax4.set_ylabel("Drawdown", fontsize=8)
ax4.legend(loc="lower left", framealpha=0.2, fontsize=8,
           facecolor="#161b22", edgecolor="#30363d", labelcolor="white")
ax4.grid(axis="y", color="#21262d", lw=0.6)

# ── OOS annotation ────────────────────────────────────────────────────────────
ax1.text(OOS_START + pd.Timedelta(days=5), ax1.get_ylim()[1] * 0.97,
         "← in-sample  |  OOS →", color=C_OOS, fontsize=8, alpha=0.8)

# ── x-axis formatting ─────────────────────────────────────────────────────────
import matplotlib.dates as mdates
ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(ax4.xaxis.get_majorticklabels(), color="#8b949e", fontsize=8)
for ax in [ax1, ax2, ax3]:
    plt.setp(ax.xaxis.get_majorticklabels(), visible=False)

# ── Stats annotation box ──────────────────────────────────────────────────────
def sharpe(nav, rf=0.045):
    dr = nav.pct_change().fillna(0)
    rf_d = (1+rf)**(1/252)-1
    ex = dr - rf_d
    return ex.mean() / ex.std() * np.sqrt(252) if ex.std() > 0 else 0

def max_dd_val(nav):
    rm = nav.cummax()
    return ((nav - rm) / rm * 100).min()

stats_text = (
    f"{'':─<36}\n"
    f"{'Metric':<18} {'TF2':>7} {'B&H':>7}\n"
    f"{'─'*36}\n"
    f"{'Total Return':<18} {(final_tf2/CAP-1)*100:>+6.1f}% {(final_bh/CAP-1)*100:>+6.1f}%\n"
    f"{'Sharpe Ratio':<18} {sharpe(nav_tf2):>7.2f} {sharpe(nav_bh):>7.2f}\n"
    f"{'Max Drawdown':<18} {max_dd_val(nav_tf2):>+6.1f}% {max_dd_val(nav_bh):>+6.1f}%\n"
    f"{'# Trades':<18} {len(r_tf2['trades']):>7}    {'—':>5}\n"
    f"{'Time in Market':<18} {sum(t['duration_days'] for t in r_tf2['trades']) / max(1,(nav_tf2.index[-1]-nav_tf2.index[0]).days)*100:>5.0f}%    {'100%':>5}\n"
    f"{'─'*36}\n"
    f"⚠️ Pre-Sep-2025 = IN-SAMPLE"
)
ax1.text(0.76, 0.04, stats_text, transform=ax1.transAxes,
         fontsize=8, color="#e6edf3", family="monospace",
         verticalalignment="bottom",
         bbox=dict(facecolor="#161b22", edgecolor="#30363d",
                   alpha=0.9, boxstyle="round,pad=0.6"))

plt.tight_layout(rect=[0, 0, 1, 1])
out_path = str(_ROOT / "backtest_monthly_profile.png")
fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=DARK)
print(f"✓ Plot saved → {out_path}")
plt.close(fig)
