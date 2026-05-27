"""
TF2 Baseline vs TF2 + V-Entry Gate — Backtest Comparison
=========================================================
Evaluates whether adding the V-shaped recovery (capitulation) signal
as a third entry gate improves TF2 performance.

Entry gate change:
  Baseline : U1  AND  (above_MA30  OR  clean_10d)
  V-gate   : U1  AND  (above_MA30  OR  clean_10d  OR  v_reversal_recent)

where v_reversal_recent[i] = True if v_reversal_likely fired on bar i
or any of the 2 preceding bars (3-bar lookback).

V-reversal signal (per bar):
  dn_score_raw = (
      (-ehma3 / max(|cumulative_mean_errhi|, 0.01)) * 0.30
      + (lb3 / 3) * 0.30
      + (elma3 / max(|elma3|, 0.10)) * 0.20
      + lb * 0.20
  )
  v_reversal_likely = (dn_score_raw > 0.8)  AND  (err_lo_today > 5.0%)

Signal analysis section shows all V-reversal dates and their forward returns.
"""

import sys, os, warnings, joblib, requests
warnings.filterwarnings("ignore")
from pathlib import Path
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import yfinance as yf

BACKTEST_START = "2024-05-26"
BACKTEST_END   = "2026-05-26"
FETCH_START    = "2024-02-01"   # 115-day warmup for 90-day rolling features
INITIAL_CAP    = 100_000.0
OOS_BOUNDARY   = pd.Timestamp("2025-09-18")   # Sep 18, 2025 = OOS start

# ── Load model ─────────────────────────────────────────────────────────────────
AD = joblib.load(str(_ROOT / "models" / "inference_assets_ct.joblib"))
print(f"✓ Model loaded  |  feat_cols: {len(AD['feat_cols'])}")

# ── Data helpers (identical to backtest_2year.py) ─────────────────────────────
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
    if F.empty: raise RuntimeError("Feature matrix empty — check feature engineering")
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


# ── Signal arrays (shared between both backtest variants) ──────────────────────
def build_signals(comp):
    """Return a dict of all signal arrays for a compiled prediction DataFrame."""
    N = len(comp)
    ca  = comp["close_asof"].values.astype(float)
    ep  = comp["actual_close"].values.astype(float)
    ph  = comp["pred_high"].values.astype(float)
    pl  = comp["pred_low"].values.astype(float)
    ah  = comp["actual_high"].values.astype(float)
    al  = comp["actual_low"].values.astype(float)

    ehi = (ah - ph) / ca * 100      # err_hi: how much actual high beat predicted
    elo = (pl - al) / ca * 100      # err_lo: how much actual low undershot predicted
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
        w = min(30, i+1)
        ma30[i] = np.mean(ca[max(0, i-w+1):i+1])
    abv = ca > ma30
    slp = np.zeros(N, dtype=bool)
    for i in range(5, N):
        if np.isfinite(ma30[i]) and np.isfinite(ma30[i-5]):
            slp[i] = ma30[i] > ma30[i-5]
    bull = abv & slp

    c10 = np.zeros(N, dtype=bool)
    for i in range(N):
        li = max(0, i-10); c10[i] = not bool(np.any(d1[li:i] | d2[li:i]))

    # ── V-reversal signal computation (vectorised) ────────────────────────────
    # dn_score_raw[i] — replicates the per-bar formula from btc_hourly_app.py
    # Normalisation denominator: cumulative mean of err_hi up to bar i
    cum_ehi_mean = np.zeros(N)
    for i in range(N):
        cum_ehi_mean[i] = np.mean(ehi[:i+1])

    dn_score = np.zeros(N)
    for i in range(N):
        norm    = max(abs(cum_ehi_mean[i]), 0.01)
        dn_score[i] = (
            (-ehma3[i] / norm)                                * 0.30 +
            (lb3[i]    / 3.0)                                 * 0.30 +
            (elma3[i]  / max(abs(elma3[i]), 0.10))            * 0.20 +
            float(lb[i])                                      * 0.20
        )

    v_rev_bar = (dn_score > 0.8) & (elo > 5.0)          # single-bar v_reversal_likely
    # 3-bar lookback: signal fires on current bar OR either of 2 prior bars
    v_recent  = np.zeros(N, dtype=bool)
    for i in range(N):
        li = max(0, i-2)
        v_recent[i] = bool(np.any(v_rev_bar[li:i+1]))

    return dict(
        N=N, ca=ca, ep=ep, ehi=ehi, elo=elo, hb=hb, lb=lb,
        ehma3=ehma3, elma3=elma3, hb3=hb3, lb3=lb3,
        u1=u1, d1=d1, d2=d2, d3=d3,
        ma30=ma30, abv=abv, slp=slp, bull=bull, c10=c10,
        dn_score=dn_score, v_rev_bar=v_rev_bar, v_recent=v_recent,
    )


# ── Backtest engine ────────────────────────────────────────────────────────────
def run_backtest(dates, sigs, use_v_gate: bool, cap: float = INITIAL_CAP):
    """Execute one backtest variant.
    use_v_gate=False → TF2 baseline
    use_v_gate=True  → TF2 + V-reversal entry gate
    """
    WARMUP = 35
    N = sigs["N"]
    ep = sigs["ep"]; abv = sigs["abv"]; c10 = sigs["c10"]
    u1 = sigs["u1"]; d2 = sigs["d2"]; d3 = sigs["d3"]; bull = sigs["bull"]
    v_recent = sigs["v_recent"]

    nav = cap; pos = "CASH"; qty = 0.0
    ep_r = en = ed_r = et = None
    trades = []; nav_arr = np.full(N, np.nan)

    for i in range(N):
        si    = i - 1
        price = ep[i]
        if i < WARMUP:
            nav_arr[i] = cap
            continue

        if pos == "LONG":
            cur = qty * price
            if si >= 0:
                should_exit = bool(d3[si] or (d2[si] and not bull[si]))
                xl = "D3" if d3[si] else "D2(bear)"
            else:
                should_exit = False; xl = "?"
            if should_exit:
                nav = cur
                oos_flag = ed_r >= OOS_BOUNDARY
                trades.append(dict(
                    entry_date    = ed_r,
                    entry_price   = ep_r,
                    entry_nav     = en,
                    entry_trigger = et,
                    exit_date     = dates[i],
                    exit_price    = price,
                    exit_nav      = nav,
                    pnl_pct       = (price / ep_r - 1) * 100,
                    pnl_abs       = nav - en,
                    exit_signal   = xl,
                    duration_days = (dates[i] - ed_r).days,
                    regime        = "BULL" if (si >= 0 and bull[si]) else "BEAR",
                    oos           = oos_flag,
                    v_entry       = et.startswith("V"),
                ))
                pos = "CASH"; qty = 0.0
            else:
                nav = cur
        else:
            if si >= 0 and u1[si]:
                # Entry gate check
                gate_abv     = bool(abv[si])
                gate_c10     = bool(c10[si])
                gate_v       = bool(v_recent[si]) if use_v_gate else False
                allow_entry  = gate_abv or gate_c10 or gate_v
                if allow_entry:
                    qty = nav / price; ep_r = price; ed_r = dates[i]; en = nav; pos = "LONG"
                    if gate_v and not gate_abv and not gate_c10:
                        et = "V-reversal gate"
                    elif gate_abv and gate_c10:
                        et = "U1+↑MA30+clean10d"
                    elif gate_abv:
                        et = "U1+↑MA30"
                    else:
                        et = "U1+clean10d"
        nav_arr[i] = qty * price if pos == "LONG" else nav

    if pos == "LONG":
        nav_arr[N-1] = qty * ep[N-1]

    nav_s = pd.Series(nav_arr[WARMUP:], index=dates[WARMUP:]).ffill()
    bh_s  = pd.Series(cap * ep[WARMUP:] / ep[WARMUP], index=dates[WARMUP:])

    # Mark open position
    open_trade = None
    if pos == "LONG":
        open_trade = dict(entry_date=ed_r, entry_price=ep_r,
                          entry_nav=en, entry_trigger=et,
                          cur_price=ep[-1], cur_nav=qty*ep[-1],
                          pnl_pct=(ep[-1]/ep_r-1)*100,
                          v_entry=et.startswith("V"))

    return dict(trades=trades, nav=nav_s, bh=bh_s, open=open_trade)


# ── Summary metrics ───────────────────────────────────────────────────────────
def summarise(res, label, rf=0.045):
    nav = res["nav"]; bh = res["bh"]; trades = res["trades"]
    cap = INITIAL_CAP
    fn  = nav.iloc[-1]; fb = bh.iloc[-1]
    ret = (fn / cap - 1) * 100
    bhr = (fb / cap - 1) * 100
    n_y = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = ((fn/cap)**(1/n_y) - 1)*100 if n_y > 0 else 0
    dr   = nav.pct_change().fillna(0)
    rfd  = (1+rf)**(1/252) - 1
    exc  = dr - rfd
    sharpe = exc.mean()/exc.std()*np.sqrt(252) if exc.std() > 0 else 0
    rm     = nav.cummax(); max_dd = float(((nav-rm)/rm*100).min())
    wins   = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    wr     = 100*len(wins)/len(trades) if trades else 0
    days_in = sum(t["duration_days"] for t in trades)
    tot_days = max(1, (nav.index[-1]-nav.index[0]).days)
    # Annual tax netting
    _ann = {}
    for t in trades:
        yr = pd.Timestamp(t["exit_date"]).year
        _ann[yr] = _ann.get(yr, 0.0) + t["pnl_abs"]
    tax = sum(0.35 * max(0, v) for v in _ann.values())

    return dict(
        label      = label,
        final_nav  = fn,
        bh_nav     = fb,
        ret_pct    = ret,
        bh_ret_pct = bhr,
        after_tax  = fn - tax,
        tax_paid   = tax,
        alpha      = fn - fb,
        cagr       = cagr,
        sharpe     = sharpe,
        max_dd     = max_dd,
        n_trades   = len(trades),
        win_rate   = wr,
        time_in    = 100 * days_in / tot_days,
        avg_pnl    = float(np.mean([t["pnl_pct"] for t in trades])) if trades else 0,
        n_v_entries = sum(1 for t in trades if t.get("v_entry")),
    )


# ── Pretty printers ───────────────────────────────────────────────────────────
def print_trade_comparison(base_trades, v_trades, open_base, open_v):
    """Side-by-side trade log: which trades differ between variants."""
    # Merge on entry_date to align trades that match; identify new V-only trades
    base_by_entry = {t["entry_date"]: t for t in base_trades}
    v_by_entry    = {t["entry_date"]: t for t in v_trades}

    all_entries = sorted(set(base_by_entry) | set(v_by_entry))

    print("\n" + "═"*120)
    print("TRADE-BY-TRADE COMPARISON")
    print("═"*120)
    hdr = (f"{'#':>3}  {'Entry':>12}  {'Exit':>12}  {'Entry $':>9}  {'Exit $':>9}"
           f"  {'PnL%':>7}  {'Days':>5}  {'Trigger / Gate':25}  {'OOS':4}  {'Status'}")
    print(hdr)
    print("─"*120)

    idx = 0
    for edt in all_entries:
        in_base = edt in base_by_entry
        in_v    = edt in v_by_entry
        idx += 1

        if in_base and in_v:
            t  = base_by_entry[edt]
            tv = v_by_entry[edt]
            # Same entry — highlight if exit differs
            exit_diff = abs(t["exit_price"] - tv["exit_price"]) > 1
            note = "  ← exits differ" if exit_diff else ""
            print(f"{'[B]':>3}  {pd.Timestamp(t['entry_date']).strftime('%b %d %Y'):>12}"
                  f"  {pd.Timestamp(t['exit_date']).strftime('%b %d %Y'):>12}"
                  f"  ${t['entry_price']:>8,.0f}  ${t['exit_price']:>8,.0f}"
                  f"  {t['pnl_pct']:>+7.1f}%  {t['duration_days']:>5}d"
                  f"  {t['entry_trigger']:25}  {'OOS' if t['oos'] else ' IS':>4}  Baseline{note}")
            if exit_diff:
                print(f"{'[V]':>3}  {pd.Timestamp(tv['entry_date']).strftime('%b %d %Y'):>12}"
                      f"  {pd.Timestamp(tv['exit_date']).strftime('%b %d %Y'):>12}"
                      f"  ${tv['entry_price']:>8,.0f}  ${tv['exit_price']:>8,.0f}"
                      f"  {tv['pnl_pct']:>+7.1f}%  {tv['duration_days']:>5}d"
                      f"  {tv['entry_trigger']:25}  {'OOS' if tv['oos'] else ' IS':>4}  V-variant")

        elif in_v and not in_base:
            tv = v_by_entry[edt]
            print(f"{'[V]':>3}  {pd.Timestamp(tv['entry_date']).strftime('%b %d %Y'):>12}"
                  f"  {pd.Timestamp(tv['exit_date']).strftime('%b %d %Y'):>12}"
                  f"  ${tv['entry_price']:>8,.0f}  ${tv['exit_price']:>8,.0f}"
                  f"  {tv['pnl_pct']:>+7.1f}%  {tv['duration_days']:>5}d"
                  f"  {tv['entry_trigger']:25}  {'OOS' if tv['oos'] else ' IS':>4}  ★ V-ONLY ENTRY")

        elif in_base and not in_v:
            t = base_by_entry[edt]
            print(f"{'[B]':>3}  {pd.Timestamp(t['entry_date']).strftime('%b %d %Y'):>12}"
                  f"  {pd.Timestamp(t['exit_date']).strftime('%b %d %Y'):>12}"
                  f"  ${t['entry_price']:>8,.0f}  ${t['exit_price']:>8,.0f}"
                  f"  {t['pnl_pct']:>+7.1f}%  {t['duration_days']:>5}d"
                  f"  {t['entry_trigger']:25}  {'OOS' if t['oos'] else ' IS':>4}  Baseline only")

    # Open positions
    print("─"*120)
    for label, op in [("Baseline open", open_base), ("V-gate open", open_v)]:
        if op:
            print(f"  {label}: entered {pd.Timestamp(op['entry_date']).strftime('%b %d %Y')} "
                  f"@ ${op['entry_price']:,.0f}  current PnL {op['pnl_pct']:+.1f}%"
                  f"{'  ← V-entry' if op.get('v_entry') else ''}")
    print("═"*120)


def print_v_signal_analysis(dates, sigs, ep_arr):
    """Show every V-reversal bar and its forward returns (signal quality)."""
    v_bars = np.where(sigs["v_rev_bar"])[0]
    if len(v_bars) == 0:
        print("\nNo V-reversal signals found in period.")
        return

    print("\n" + "═"*100)
    print("V-REVERSAL SIGNAL QUALITY ANALYSIS")
    print(f"Total V-reversal (bar-level) signals: {len(v_bars)}")
    print("─"*100)
    print(f"  {'Date':>12}  {'dn_score':>9}  {'err_lo%':>8}  {'Price':>9}"
          f"  {'Fwd+3d%':>8}  {'Fwd+5d%':>8}  {'Fwd+10d%':>8}  {'Fwd+20d%':>9}  {'OOS':4}")
    print("─"*100)

    n_pos_5d = 0; n_pos_10d = 0; n_pos_20d = 0
    for idx in v_bars:
        dt     = dates[idx]
        price  = ep_arr[idx]
        score  = sigs["dn_score"][idx]
        err_lo = sigs["elo"][idx]
        oos    = "OOS" if dt >= OOS_BOUNDARY else " IS"

        def fwd(k):
            j = idx + k
            if j < len(ep_arr):
                return (ep_arr[j] / price - 1) * 100
            return float("nan")

        f3  = fwd(3);  f5  = fwd(5)
        f10 = fwd(10); f20 = fwd(20)

        if not np.isnan(f5)  and f5  > 0: n_pos_5d  += 1
        if not np.isnan(f10) and f10 > 0: n_pos_10d += 1
        if not np.isnan(f20) and f20 > 0: n_pos_20d += 1

        print(f"  {dt.strftime('%b %d, %Y'):>12}  {score:>9.3f}  {err_lo:>+8.2f}%  ${price:>8,.0f}"
              f"  {f3:>+8.2f}%  {f5:>+8.2f}%  {f10:>+8.2f}%  {f20:>+9.2f}%  {oos}")

    n = len(v_bars)
    print("─"*100)
    print(f"  Signal hit rate (fwd price > 0):  "
          f"+5d: {n_pos_5d}/{n} ({100*n_pos_5d/n:.0f}%)   "
          f"+10d: {n_pos_10d}/{n} ({100*n_pos_10d/n:.0f}%)   "
          f"+20d: {n_pos_20d}/{n} ({100*n_pos_20d/n:.0f}%)")
    print("═"*100)


def print_summary_table(base_m, v_m):
    print("\n" + "═"*72)
    print(f"{'SUMMARY COMPARISON':^72}")
    print(f"{'Metric':<28} {'TF2 Baseline':>20} {'TF2 + V-Gate':>20}")
    print("─"*72)
    rows = [
        ("Final NAV",         f"${base_m['final_nav']:>18,.0f}",   f"${v_m['final_nav']:>18,.0f}"),
        ("Total Return",      f"{base_m['ret_pct']:>18.1f}%",      f"{v_m['ret_pct']:>18.1f}%"),
        ("After-Tax NAV",     f"${base_m['after_tax']:>18,.0f}",   f"${v_m['after_tax']:>18,.0f}"),
        ("Tax Paid",          f"${base_m['tax_paid']:>18,.0f}",    f"${v_m['tax_paid']:>18,.0f}"),
        ("Alpha vs B&H",      f"${base_m['alpha']:>+18,.0f}",      f"${v_m['alpha']:>+18,.0f}"),
        ("B&H NAV",           f"${base_m['bh_nav']:>18,.0f}",      f"${v_m['bh_nav']:>18,.0f}"),
        ("CAGR",              f"{base_m['cagr']:>18.1f}%",         f"{v_m['cagr']:>18.1f}%"),
        ("Sharpe Ratio",      f"{base_m['sharpe']:>18.2f}",        f"{v_m['sharpe']:>18.2f}"),
        ("Max Drawdown",      f"{base_m['max_dd']:>18.1f}%",       f"{v_m['max_dd']:>18.1f}%"),
        ("# Trades (closed)", f"{base_m['n_trades']:>20}",         f"{v_m['n_trades']:>20}"),
        ("  of which V-entry",f"{base_m['n_v_entries']:>20}",      f"{v_m['n_v_entries']:>20}"),
        ("Win Rate",          f"{base_m['win_rate']:>18.1f}%",     f"{v_m['win_rate']:>18.1f}%"),
        ("Avg Trade PnL",     f"{base_m['avg_pnl']:>+18.1f}%",     f"{v_m['avg_pnl']:>+18.1f}%"),
        ("Time in Market",    f"{base_m['time_in']:>18.1f}%",      f"{v_m['time_in']:>18.1f}%"),
    ]
    for lbl, bv, vv in rows:
        print(f"  {lbl:<26} {bv:>20} {vv:>20}")
    print("═"*72)

    # Delta row
    delta_nav   = v_m["final_nav"]  - base_m["final_nav"]
    delta_tax   = v_m["after_tax"]  - base_m["after_tax"]
    delta_alpha = v_m["alpha"]       - base_m["alpha"]
    print(f"\n  V-gate delta vs baseline:")
    print(f"    NAV change       : ${delta_nav:>+12,.0f}"
          f"  ({'improvement' if delta_nav > 0 else 'degradation'})")
    print(f"    After-tax NAV Δ  : ${delta_tax:>+12,.0f}")
    print(f"    Alpha Δ          : ${delta_alpha:>+12,.0f}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # 1. Fetch & build predictions
    df_raw = fetch_data(FETCH_START, BACKTEST_END)
    preds  = build_preds(df_raw)

    # 2. Assemble comparison window (May 26 2024 → today)
    sd, ed = pd.Timestamp(BACKTEST_START), pd.Timestamp(BACKTEST_END)
    p = preds.loc[(preds.index >= sd - pd.Timedelta(days=45)) &
                  (preds.index <= ed)].copy()
    c_raw = df_raw["btc_close"]; h_raw = df_raw["btc_high"]; lo_raw = df_raw["btc_low"]
    p["actual_high"]  = h_raw.reindex(p.index).values
    p["actual_low"]   = lo_raw.reindex(p.index).values
    p["actual_close"] = c_raw.reindex(p.index).values
    comp = p.dropna(subset=["actual_high","actual_low","actual_close"]).reset_index()
    comp = comp[comp["target_date"] >= sd].reset_index(drop=True)

    print(f"\nBacktest window: {comp['target_date'].iloc[0].strftime('%b %d, %Y')} "
          f"→ {comp['target_date'].iloc[-1].strftime('%b %d, %Y')} "
          f"({len(comp)} bars)")

    # 3. Build signal arrays
    print("Computing signals …")
    sigs  = build_signals(comp)
    dates = pd.DatetimeIndex(comp["target_date"])

    n_v_bars = int(sigs["v_rev_bar"].sum())
    print(f"  V-reversal bars (dn_score>0.8 & err_lo>5%): {n_v_bars}")

    # 4. Run both backtest variants
    print("Running TF2 baseline …")
    res_base = run_backtest(dates, sigs, use_v_gate=False)
    print("Running TF2 + V-gate …")
    res_v    = run_backtest(dates, sigs, use_v_gate=True)

    # 5. V-reversal signal quality analysis
    print_v_signal_analysis(dates, sigs, sigs["ep"])

    # 6. Trade-by-trade comparison
    print_trade_comparison(res_base["trades"], res_v["trades"],
                           res_base["open"], res_v["open"])

    # 7. Summary metrics table
    m_base = summarise(res_base, "TF2 Baseline")
    m_v    = summarise(res_v,    "TF2 + V-Gate")
    print_summary_table(m_base, m_v)
