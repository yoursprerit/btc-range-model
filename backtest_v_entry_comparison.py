"""
TF2 V-Entry Gate — Backtest Comparison (all variants)
======================================================
Period: May 26, 2024 → May 26, 2026  |  OOS boundary: Sep 18, 2025

Variants tested
───────────────
  A  TF2 Baseline       U1 AND (above_MA30 OR clean_10d)
  B  V-Gate 3-bar       U1 AND (above_MA30 OR clean_10d OR v_recent_3)
  C  V-Gate 7-bar       U1 AND (above_MA30 OR clean_10d OR v_recent_7)
  D  V-Standalone       (dn_score > 1.5 AND err_lo > 5%)  — no U1 required
  E  V-Gate + D2-sup 5  V-Gate 3-bar, suppress D2 exit for first 5 bars on V-entries
  F  V-Gate + D2-sup 10 V-Gate 3-bar, suppress D2 exit for first 10 bars on V-entries
  G  V-Gate + D2-sup 15 V-Gate 3-bar, suppress D2 exit for first 15 bars on V-entries

D2-suppression rationale:
  Post-capitulation recovery is inherently noisy — err_hi_ma3 can dip
  below -1% during the first few days of a bounce, triggering a premature
  D2/bear exit before the trend re-establishes.  Suppressing D2 exits on
  V-initiated positions for N bars lets the recovery breathe while D3
  (the structural exhaustion signal) still provides a hard stop.

V-reversal signal (per bar):
  dn_score  = (-ehma3 / max(|cum_mean_errhi|, 0.01)) * 0.30
            + (lb3 / 3) * 0.30
            + (elma3 / max(|elma3|, 0.10)) * 0.20
            + lb * 0.20
  v_rev_bar = (dn_score > 0.8) AND (err_lo > 5.0%)
  v_recent_3[i] = any v_rev_bar in [i-2 .. i]
  v_recent_7[i] = any v_rev_bar in [i-6 .. i]
"""

import sys, os, warnings, joblib, requests
warnings.filterwarnings("ignore")
from pathlib import Path
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import yfinance as yf

BACKTEST_START  = "2024-05-26"
BACKTEST_END    = "2026-05-26"
FETCH_START     = "2024-02-01"
INITIAL_CAP     = 100_000.0
OOS_BOUNDARY    = pd.Timestamp("2025-09-18")
V_STANDALONE_THRESH = 1.5   # dn_score threshold for no-U1 entry

# ── Model ──────────────────────────────────────────────────────────────────────
AD = joblib.load(str(_ROOT / "models" / "inference_assets_ct.joblib"))
print(f"✓ Model loaded  |  feat_cols: {len(AD['feat_cols'])}")

# ── Data helpers ───────────────────────────────────────────────────────────────
_MACRO_SYMS = {"eth":"ETH-USD","spx":"^GSPC","ndx":"^IXIC",
               "vix":"^VIX","gold":"GC=F","dxy":"DX-Y.NYB","tnx":"^TNX"}
_ONCHAIN = ["hash-rate","difficulty","n-transactions","miners-revenue",
            "n-unique-addresses","transaction-fees-usd","mempool-size",
            "estimated-transaction-volume-usd","market-cap",
            "avg-block-size","cost-per-transaction"]

def _yf(sym, start, end):
    d = yf.download(sym, start=start, end=end, progress=False, auto_adjust=False)
    if isinstance(d.columns, pd.MultiIndex): d.columns = [c[0] for c in d.columns]
    d.index = pd.DatetimeIndex(d.index).tz_localize(None).normalize()
    return d

def fetch_data(start, end):
    print(f"\nFetching {start} → {end} …")
    frames = {}
    d = _yf("BTC-USD", start, end)
    frames.update({"btc_close":d["Close"],"btc_high":d["High"],
                   "btc_low":d["Low"],"btc_volume":d["Volume"]})
    for name, sym in _MACRO_SYMS.items():
        try: frames[f"{name}_close"] = _yf(sym, start, end)["Close"]
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
    ret = np.log(c).diff(); f = pd.DataFrame(index=df.index)
    for k in [1,3,5,7,14,30]: f[f"ret_{k}"] = ret.rolling(k).sum()
    for k in [5,10,20,30]:    f[f"vol_{k}"] = ret.rolling(k).std()
    prev_c = c.shift(1)
    tr = pd.concat([(h-l_),(h-prev_c).abs(),(l_-prev_c).abs()],axis=1).max(axis=1)
    for k in [7,14,30]: f[f"atr_{k}"] = tr.rolling(k).mean()/c
    f["range_today"]=(h-l_)/c; f["range_ma7"]=((h-l_)/c).rolling(7).mean()
    f["range_ma30"]=((h-l_)/c).rolling(30).mean(); f["range_std30"]=((h-l_)/c).rolling(30).std()
    g=c.diff().clip(lower=0).rolling(14).mean(); ls=(-c.diff().clip(upper=0)).rolling(14).mean()
    f["rsi_14"]=100-100/(1+g/ls.replace(0,np.nan))
    e12=c.ewm(span=12,adjust=False).mean(); e26=c.ewm(span=26,adjust=False).mean(); macd=e12-e26
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
    fc=AD["feat_cols"]; f=f.replace([np.inf,-np.inf],np.nan)
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


# ── Signal arrays ──────────────────────────────────────────────────────────────
def build_signals(comp):
    N   = len(comp)
    ca  = comp["close_asof"].values.astype(float)
    ep  = comp["actual_close"].values.astype(float)
    ph  = comp["pred_high"].values.astype(float)
    pl  = comp["pred_low"].values.astype(float)
    ah  = comp["actual_high"].values.astype(float)
    al  = comp["actual_low"].values.astype(float)

    ehi = (ah - ph) / ca * 100
    elo = (pl - al) / ca * 100
    hb  = (ah > ph).astype(int)
    lb  = (al < pl).astype(int)

    ehma3 = np.zeros(N); elma3 = np.zeros(N)
    hb3   = np.zeros(N, dtype=int); lb3 = np.zeros(N, dtype=int)
    for i in range(N):
        s = max(0, i-2)
        ehma3[i] = np.mean(ehi[s:i+1]); elma3[i] = np.mean(elo[s:i+1])
        hb3[i]   = int(np.sum(hb[s:i+1])); lb3[i] = int(np.sum(lb[s:i+1]))

    u1 = (ehma3 > 0.5) & (hb3 >= 2)
    d1 = (lb3 >= 2)    & (elma3 > 0.5)
    d2 = ehma3 < -1.0
    d3 = np.zeros(N, dtype=bool)
    for i in range(1, N):
        consec = 0
        for k in range(i-1, -1, -1):
            if hb[k]: consec += 1
            else: break
        if consec >= 3 and lb[i]: d3[i] = True

    ma30 = np.full(N, np.nan)
    for i in range(N):
        w = min(30, i+1); ma30[i] = np.mean(ca[max(0,i-w+1):i+1])
    abv = ca > ma30
    slp = np.zeros(N, dtype=bool)
    for i in range(5, N):
        if np.isfinite(ma30[i]) and np.isfinite(ma30[i-5]):
            slp[i] = ma30[i] > ma30[i-5]
    bull = abv & slp

    c10 = np.zeros(N, dtype=bool)
    for i in range(N):
        li = max(0, i-10); c10[i] = not bool(np.any(d1[li:i] | d2[li:i]))

    # ── V-reversal signal arrays ──────────────────────────────────────────────
    cum_ehi = np.zeros(N)
    for i in range(N):
        cum_ehi[i] = np.mean(ehi[:i+1])

    dn_score = np.zeros(N)
    for i in range(N):
        norm = max(abs(cum_ehi[i]), 0.01)
        dn_score[i] = (
            (-ehma3[i] / norm)                          * 0.30 +
            (lb3[i]    / 3.0)                           * 0.30 +
            (elma3[i]  / max(abs(elma3[i]), 0.10))      * 0.20 +
            float(lb[i])                                * 0.20
        )

    # Tier-1: standard v_reversal_likely (dn_score > 0.8 & err_lo > 5%)
    v_rev_bar = (dn_score > 0.8) & (elo > 5.0)
    # Tier-2: high-conviction standalone (dn_score > 1.5 & err_lo > 5%)
    v_hi_bar  = (dn_score > V_STANDALONE_THRESH) & (elo > 5.0)

    # Rolling lookback windows
    v_recent_3 = np.zeros(N, dtype=bool)
    v_recent_7 = np.zeros(N, dtype=bool)
    for i in range(N):
        v_recent_3[i] = bool(np.any(v_rev_bar[max(0,i-2):i+1]))  # 3-bar
        v_recent_7[i] = bool(np.any(v_rev_bar[max(0,i-6):i+1]))  # 7-bar

    return dict(
        N=N, ca=ca, ep=ep, ehi=ehi, elo=elo, hb=hb, lb=lb,
        ehma3=ehma3, elma3=elma3, hb3=hb3, lb3=lb3,
        u1=u1, d1=d1, d2=d2, d3=d3,
        ma30=ma30, abv=abv, slp=slp, bull=bull, c10=c10,
        dn_score=dn_score,
        v_rev_bar=v_rev_bar, v_hi_bar=v_hi_bar,
        v_recent_3=v_recent_3, v_recent_7=v_recent_7,
    )


# ── Unified backtest engine ────────────────────────────────────────────────────
VARIANTS = {
    "A": dict(label="TF2 Baseline",        use_v3=False, use_v7=False, use_vs=False, d2_sup=0),
    "B": dict(label="V-Gate 3-bar",         use_v3=True,  use_v7=False, use_vs=False, d2_sup=0),
    "C": dict(label="V-Gate 7-bar",         use_v3=False, use_v7=True,  use_vs=False, d2_sup=0),
    "D": dict(label="V-Standalone",         use_v3=False, use_v7=False, use_vs=True,  d2_sup=0),
    "E": dict(label="V-Gate+D2sup5",        use_v3=True,  use_v7=False, use_vs=False, d2_sup=5),
    "F": dict(label="V-Gate+D2sup10",       use_v3=True,  use_v7=False, use_vs=False, d2_sup=10),
    "G": dict(label="V-Gate+D2sup15",       use_v3=True,  use_v7=False, use_vs=False, d2_sup=15),
}

def run_backtest(dates, sigs, cfg: dict, cap: float = INITIAL_CAP):
    WARMUP  = 35
    N       = sigs["N"]
    ep      = sigs["ep"];  abv  = sigs["abv"];  c10  = sigs["c10"]
    u1      = sigs["u1"];  d2   = sigs["d2"];   d3   = sigs["d3"];  bull = sigs["bull"]
    v3      = sigs["v_recent_3"]
    v7      = sigs["v_recent_7"]
    v_hi    = sigs["v_hi_bar"]
    use_v3  = cfg["use_v3"]; use_v7 = cfg["use_v7"]; use_vs = cfg["use_vs"]
    d2_sup  = int(cfg.get("d2_sup", 0))   # bars to suppress D2 on V-entries (0 = no suppression)

    nav = cap; pos = "CASH"; qty = 0.0
    ep_r = en = ed_r = et = None
    is_v_pos   = False   # whether current open position was V-gate initiated
    bars_in_pos = 0      # bars elapsed since entry (for D2 suppression countdown)
    trades = []; nav_arr = np.full(N, np.nan)

    for i in range(N):
        si    = i - 1
        price = ep[i]
        if i < WARMUP:
            nav_arr[i] = cap; continue

        if pos == "LONG":
            bars_in_pos += 1
            cur = qty * price
            should_exit = False; xl = "?"
            if si >= 0:
                # D2 is suppressed for the first d2_sup bars of a V-initiated position
                d2_blocked = (d2_sup > 0 and is_v_pos and bars_in_pos <= d2_sup)
                should_exit = bool(d3[si] or (d2[si] and not bull[si] and not d2_blocked))
                xl = "D3" if d3[si] else ("D2(bear,sup)" if d2_blocked else "D2(bear)")
                # override: if D2 was the only trigger and we blocked it, don't exit
                if d2_blocked and not d3[si]:
                    should_exit = False
            if should_exit:
                nav = cur
                trades.append(dict(
                    entry_date    = ed_r,   entry_price = ep_r,
                    entry_nav     = en,     entry_trigger = et,
                    exit_date     = dates[i], exit_price  = price,
                    exit_nav      = nav,
                    pnl_pct       = (price / ep_r - 1) * 100,
                    pnl_abs       = nav - en,
                    exit_signal   = xl,
                    duration_days = (dates[i] - ed_r).days,
                    regime        = "BULL" if (si >= 0 and bull[si]) else "BEAR",
                    oos           = ed_r >= OOS_BOUNDARY,
                    v_entry       = et.startswith("V"),
                ))
                pos = "CASH"; qty = 0.0
                is_v_pos = False; bars_in_pos = 0
            else:
                nav = cur

        else:
            allow = False; trigger = ""
            if si >= 0:
                gate_abv = bool(abv[si])
                gate_c10 = bool(c10[si])
                gate_v3  = bool(v3[si])  if use_v3 else False
                gate_v7  = bool(v7[si])  if use_v7 else False
                gate_vs  = bool(v_hi[si]) if use_vs else False

                if use_vs and gate_vs:
                    allow   = True
                    trigger = f"V-standalone(score>{V_STANDALONE_THRESH:.1f})"

                if not allow and u1[si]:
                    if gate_abv or gate_c10 or gate_v3 or gate_v7:
                        allow = True
                        if (gate_v3 or gate_v7) and not gate_abv and not gate_c10:
                            trigger = "V-recent gate"
                        elif gate_abv and gate_c10:
                            trigger = "U1+↑MA30+clean10d"
                        elif gate_abv:
                            trigger = "U1+↑MA30"
                        else:
                            trigger = "U1+clean10d"

            if allow:
                qty = nav / price; ep_r = price; ed_r = dates[i]; en = nav; pos = "LONG"
                et = trigger
                is_v_pos    = trigger.startswith("V")
                bars_in_pos = 0

        nav_arr[i] = qty * price if pos == "LONG" else nav

    if pos == "LONG":
        nav_arr[N-1] = qty * ep[N-1]

    nav_s = pd.Series(nav_arr[WARMUP:], index=dates[WARMUP:]).ffill()
    bh_s  = pd.Series(cap * ep[WARMUP:] / ep[WARMUP], index=dates[WARMUP:])

    open_trade = None
    if pos == "LONG":
        open_trade = dict(entry_date=ed_r, entry_price=ep_r,
                          entry_nav=en, entry_trigger=et,
                          cur_price=ep[-1], cur_nav=qty*ep[-1],
                          pnl_pct=(ep[-1]/ep_r-1)*100,
                          v_entry=et.startswith("V"),
                          d2_suppressed_until=bars_in_pos if is_v_pos and bars_in_pos <= d2_sup else 0)

    return dict(trades=trades, nav=nav_s, bh=bh_s, open=open_trade)


# ── Metrics ────────────────────────────────────────────────────────────────────
def summarise(res, label, rf=0.045):
    nav = res["nav"]; bh = res["bh"]; trades = res["trades"]
    cap = INITIAL_CAP
    fn = nav.iloc[-1]; fb = bh.iloc[-1]
    n_y = (nav.index[-1] - nav.index[0]).days / 365.25
    dr  = nav.pct_change().fillna(0)
    rfd = (1+rf)**(1/252) - 1
    exc = dr - rfd
    sharpe  = exc.mean()/exc.std()*np.sqrt(252) if exc.std() > 0 else 0
    rm      = nav.cummax(); max_dd = float(((nav-rm)/rm*100).min())
    wins    = [t for t in trades if t["pnl_pct"] > 0]
    days_in = sum(t["duration_days"] for t in trades)
    tot_days = max(1, (nav.index[-1]-nav.index[0]).days)
    _ann = {}
    for t in trades:
        yr = pd.Timestamp(t["exit_date"]).year
        _ann[yr] = _ann.get(yr, 0.0) + t["pnl_abs"]
    tax = sum(0.35 * max(0.0, v) for v in _ann.values())

    # OOS-only sub-metrics
    oos_t    = [t for t in trades if t.get("oos")]
    oos_wins = [t for t in oos_t if t["pnl_pct"] > 0]

    return dict(
        label       = label,
        final_nav   = fn,
        bh_nav      = fb,
        ret_pct     = (fn/cap - 1)*100,
        bh_ret_pct  = (fb/cap - 1)*100,
        after_tax   = fn - tax,
        tax_paid    = tax,
        alpha       = fn - fb,
        cagr        = ((fn/cap)**(1/n_y) - 1)*100 if n_y > 0 else 0,
        sharpe      = sharpe,
        max_dd      = max_dd,
        n_trades    = len(trades),
        win_rate    = 100*len(wins)/len(trades) if trades else 0,
        time_in     = 100 * days_in / tot_days,
        avg_pnl     = float(np.mean([t["pnl_pct"] for t in trades])) if trades else 0,
        n_v_entries = sum(1 for t in trades if t.get("v_entry")),
        oos_trades  = len(oos_t),
        oos_wr      = 100*len(oos_wins)/len(oos_t) if oos_t else 0,
        oos_avg_pnl = float(np.mean([t["pnl_pct"] for t in oos_t])) if oos_t else 0,
    )


# ── Print helpers ──────────────────────────────────────────────────────────────
def print_v_signal_analysis(dates, sigs):
    ep_arr = sigs["ep"]
    v_bars = np.where(sigs["v_rev_bar"])[0]
    hi_bars = set(np.where(sigs["v_hi_bar"])[0])
    print("\n" + "═"*110)
    print("V-REVERSAL SIGNAL QUALITY")
    print(f"  Standard (dn_score>0.8 & err_lo>5%): {len(v_bars)} bars  |  "
          f"High-conviction (dn_score>{V_STANDALONE_THRESH} & err_lo>5%): {len(hi_bars)} bars")
    print("─"*110)
    print(f"  {'Date':>12}  {'dn_score':>9}  {'err_lo%':>8}  {'Tier':>5}  {'Price':>9}"
          f"  {'Fwd+3d':>7}  {'Fwd+5d':>7}  {'Fwd+10d':>8}  {'Fwd+20d':>8}  {'Period':6}"
          f"  {'U1 same bar':>11}")
    print("─"*110)
    n5=n10=n20=0
    for idx in v_bars:
        dt    = dates[idx]; price = ep_arr[idx]
        score = sigs["dn_score"][idx]; err_lo = sigs["elo"][idx]
        tier  = "HI" if idx in hi_bars else "std"
        period = "OOS" if dt >= OOS_BOUNDARY else " IS"
        u1_same = "✓" if sigs["u1"][idx] else "—"
        def fwd(k):
            j = idx + k
            return (ep_arr[j]/price-1)*100 if j < len(ep_arr) else float("nan")
        f3,f5,f10,f20 = fwd(3),fwd(5),fwd(10),fwd(20)
        if not np.isnan(f5)  and f5  > 0: n5  += 1
        if not np.isnan(f10) and f10 > 0: n10 += 1
        if not np.isnan(f20) and f20 > 0: n20 += 1
        print(f"  {dt.strftime('%b %d, %Y'):>12}  {score:>9.3f}  {err_lo:>+8.2f}%  {tier:>5}  ${price:>8,.0f}"
              f"  {f3:>+7.2f}%  {f5:>+7.2f}%  {f10:>+8.2f}%  {f20:>+8.2f}%  {period:6}  {u1_same:>11}")
    n = len(v_bars)
    print("─"*110)
    print(f"  Hit rate (fwd > 0):  +5d {n5}/{n} ({100*n5/n:.0f}%)  "
          f"+10d {n10}/{n} ({100*n10/n:.0f}%)  +20d {n20}/{n} ({100*n20/n:.0f}%)")
    print("═"*110)


def print_trade_log(results: dict, dates, sigs):
    """Unified trade log across all variants; marks which variant(s) each trade appears in."""
    # Collect all unique (entry_date, exit_date) pairs
    all_keys = set()
    for tag, res in results.items():
        for t in res["trades"]:
            all_keys.add(t["entry_date"])
    all_keys = sorted(all_keys)

    W = 140
    print("\n" + "═"*W)
    print("TRADE LOG  (all variants)")
    print("─"*W)
    print(f"  {'Entry':>12}  {'Exit':>12}  {'Entry $':>9}  {'Exit $':>9}"
          f"  {'PnL%':>7}  {'Days':>5}  {'Trigger':30}  {'IS/OOS':6}  {'Variants':30}")
    print("─"*W)

    for edt in all_keys:
        # Gather all variant entries matching this date
        variant_rows = {}
        for tag, res in results.items():
            for t in res["trades"]:
                if t["entry_date"] == edt:
                    variant_rows[tag] = t

        # Pick a canonical row to display (prefer baseline if present, else first)
        canon = variant_rows.get("A") or next(iter(variant_rows.values()))
        tags  = "".join(sorted(variant_rows.keys()))
        oos   = " OOS" if canon["oos"] else "  IS"

        # Highlight rows that differ across variants
        exits = {t["exit_date"] for t in variant_rows.values()}
        exit_note = "*" if len(exits) > 1 else " "

        v_flag = "★ " if canon.get("v_entry") else "  "
        present_flags = "  ".join(
            f"[{tag}]" if tag in variant_rows else f" {tag} "
            for tag in sorted(results.keys())
        )

        print(f"  {pd.Timestamp(canon['entry_date']).strftime('%b %d, %Y'):>12}"
              f"  {pd.Timestamp(canon['exit_date']).strftime('%b %d, %Y'):>12}"
              f"  ${canon['entry_price']:>8,.0f}  ${canon['exit_price']:>8,.0f}"
              f"  {canon['pnl_pct']:>+7.1f}%  {canon['duration_days']:>5}d"
              f"  {v_flag}{canon['entry_trigger']:28}  {oos:6}  {present_flags}{exit_note}")

        # If exits differ, show variant-specific rows
        if len(exits) > 1:
            for tag, t in sorted(variant_rows.items()):
                if t != canon:
                    print(f"  {'':>12}  {'↳ ['+tag+'] exit:':>14}"
                          f"  {'':>9}  ${t['exit_price']:>8,.0f}"
                          f"  {t['pnl_pct']:>+7.1f}%  {t['duration_days']:>5}d"
                          f"  {'':30}  {'':6}")

    # Open positions
    print("─"*W)
    for tag, res in results.items():
        op = res.get("open")
        if op:
            print(f"  [OPEN {tag}]  entered {pd.Timestamp(op['entry_date']).strftime('%b %d, %Y')}"
                  f"  @ ${op['entry_price']:,.0f}  cur PnL {op['pnl_pct']:+.1f}%"
                  f"  {'← V-entry' if op.get('v_entry') else ''}")
    print("═"*W)


def print_summary_table(metrics: list):
    labels = [m["label"] for m in metrics]
    W = 16
    sep = "─" * (28 + W * len(metrics) + 2)
    print("\n" + "═" * (28 + W * len(metrics) + 2))
    print(f"{'FOUR-VARIANT SUMMARY':^{28 + W*len(metrics)}}")
    print(sep)
    print(f"  {'Metric':<26}" + "".join(f"{l:>{W}}" for l in labels))
    print(sep)

    def row(lbl, key, fmt):
        vals = [fmt.format(m[key]) for m in metrics]
        print(f"  {lbl:<26}" + "".join(f"{v:>{W}}" for v in vals))

    row("Final NAV",         "final_nav",   "${:,.0f}")
    row("Total Return",      "ret_pct",     "{:+.1f}%")
    row("After-Tax NAV",     "after_tax",   "${:,.0f}")
    row("Tax Paid",          "tax_paid",    "${:,.0f}")
    row("Alpha vs B&H",      "alpha",       "${:+,.0f}")
    row("B&H NAV",           "bh_nav",      "${:,.0f}")
    row("CAGR",              "cagr",        "{:+.1f}%")
    row("Sharpe Ratio",      "sharpe",      "{:.2f}")
    row("Max Drawdown",      "max_dd",      "{:.1f}%")
    print(sep)
    row("# Trades (total)",  "n_trades",    "{:d}")
    row("  V-entries",       "n_v_entries", "{:d}")
    row("Win Rate",          "win_rate",    "{:.1f}%")
    row("Avg Trade PnL",     "avg_pnl",     "{:+.1f}%")
    row("Time in Market",    "time_in",     "{:.1f}%")
    print(sep)
    row("OOS Trades",        "oos_trades",  "{:d}")
    row("OOS Win Rate",      "oos_wr",      "{:.1f}%")
    row("OOS Avg PnL",       "oos_avg_pnl", "{:+.1f}%")
    print("═" * (28 + W * len(metrics) + 2))

    # Delta vs baseline
    base = metrics[0]
    print(f"\n  Δ vs Baseline [{base['label']}]:")
    for m in metrics[1:]:
        dn = m["final_nav"] - base["final_nav"]
        da = m["after_tax"] - base["after_tax"]
        ds = m["sharpe"]    - base["sharpe"]
        dd = m["max_dd"]    - base["max_dd"]
        print(f"    [{m['label']:15}]  NAV {dn:>+9,.0f}  "
              f"After-tax {da:>+9,.0f}  "
              f"Sharpe {ds:>+6.3f}  "
              f"MaxDD {dd:>+5.1f}pp"
              f"  {'✓ all-round improvement' if dn>0 and ds>0 and dd>0 else ''}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    df_raw = fetch_data(FETCH_START, BACKTEST_END)
    preds  = build_preds(df_raw)

    sd, ed = pd.Timestamp(BACKTEST_START), pd.Timestamp(BACKTEST_END)
    p = preds.loc[(preds.index >= sd - pd.Timedelta(days=45)) &
                  (preds.index <= ed)].copy()
    for col, src in [("actual_high","btc_high"),("actual_low","btc_low"),("actual_close","btc_close")]:
        p[col] = df_raw[src].reindex(p.index).values
    comp = p.dropna(subset=["actual_high","actual_low","actual_close"]).reset_index()
    comp = comp[comp["target_date"] >= sd].reset_index(drop=True)
    dates = pd.DatetimeIndex(comp["target_date"])

    print(f"\nBacktest window: {dates[0].strftime('%b %d, %Y')} → "
          f"{dates[-1].strftime('%b %d, %Y')}  ({len(comp)} bars)")
    print(f"OOS boundary: {OOS_BOUNDARY.strftime('%b %d, %Y')}")

    print("Computing signals …")
    sigs = build_signals(comp)
    print(f"  v_rev_bar (score>0.8 & lo_err>5%):    {sigs['v_rev_bar'].sum():3d} bars")
    print(f"  v_hi_bar  (score>{V_STANDALONE_THRESH} & lo_err>5%):  {sigs['v_hi_bar'].sum():3d} bars")

    # Signal quality table (shown once)
    print_v_signal_analysis(dates, sigs)

    # Run all variants
    results = {}
    for tag, cfg in VARIANTS.items():
        print(f"Running [{tag}] {cfg['label']} …")
        results[tag] = run_backtest(dates, sigs, cfg)

    # ── Part 1: Full trade log (all variants) ─────────────────────────────────
    print_trade_log(results, dates, sigs)

    # ── Part 2: Full summary (all seven variants) ──────────────────────────────
    all_metrics = [summarise(results[tag], f"[{tag}] {cfg['label']}")
                   for tag, cfg in VARIANTS.items()]
    print_summary_table(all_metrics)

    # ── Part 3: Focused D2-suppression summary (A, B, E, F, G only) ───────────
    focus_tags = ["A", "B", "E", "F", "G"]
    focus_metrics = [m for m in all_metrics if any(m["label"].startswith(f"[{t}]") for t in focus_tags)]

    W = 18
    sep = "─" * (28 + W * len(focus_metrics) + 2)
    print("\n" + "═" * (28 + W * len(focus_metrics) + 2))
    print(f"{'D2-SUPPRESSION FOCUSED VIEW  (A, B, E, F, G)':^{28 + W*len(focus_metrics)}}")
    print(sep)
    print(f"  {'Metric':<26}" + "".join(f"{m['label']:>{W}}" for m in focus_metrics))
    print(sep)

    def frow(lbl, key, fmt):
        vals = [fmt.format(m[key]) for m in focus_metrics]
        print(f"  {lbl:<26}" + "".join(f"{v:>{W}}" for v in vals))

    frow("Final NAV",        "final_nav",   "${:,.0f}")
    frow("Total Return",     "ret_pct",     "{:+.1f}%")
    frow("After-Tax NAV",    "after_tax",   "${:,.0f}")
    frow("Alpha vs B&H",     "alpha",       "${:+,.0f}")
    frow("Sharpe Ratio",     "sharpe",      "{:.3f}")
    frow("Max Drawdown",     "max_dd",      "{:.1f}%")
    print(sep)
    frow("# Trades",         "n_trades",    "{:d}")
    frow("  V-entries",      "n_v_entries", "{:d}")
    frow("Win Rate",         "win_rate",    "{:.1f}%")
    frow("Avg Trade PnL",    "avg_pnl",     "{:+.1f}%")
    frow("Time in Market",   "time_in",     "{:.1f}%")
    print(sep)
    frow("OOS Trades",       "oos_trades",  "{:d}")
    frow("OOS Win Rate",     "oos_wr",      "{:.1f}%")
    frow("OOS Avg PnL",      "oos_avg_pnl", "{:+.1f}%")
    print("═" * (28 + W * len(focus_metrics) + 2))

    base_m = focus_metrics[0]
    print(f"\n  Δ vs [A] Baseline:")
    for m in focus_metrics[1:]:
        dn = m["final_nav"]  - base_m["final_nav"]
        da = m["after_tax"]  - base_m["after_tax"]
        ds = m["sharpe"]     - base_m["sharpe"]
        dd = m["max_dd"]     - base_m["max_dd"]
        flag = "✓ best" if dn == max(x["final_nav"]-base_m["final_nav"] for x in focus_metrics[1:]) else ""
        print(f"    {m['label']:22}  NAV {dn:>+9,.0f}  "
              f"After-tax {da:>+9,.0f}  "
              f"Sharpe {ds:>+6.3f}  "
              f"MaxDD {dd:>+5.1f}pp  {flag}")
