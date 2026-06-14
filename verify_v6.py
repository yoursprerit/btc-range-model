#!/usr/bin/env python3
"""
Offline verification of V6 (Idea 5: SPX trend alignment gate) vs V1 baseline.

V6 rule: entry requires V1 signal AND (SPX MA20 slope positive OR v_recent).
SPX MA20 slope = today's 20-day MA > 5 days ago's 20-day MA.
V-reversal override: v_recent=True bypasses the SPX gate.

Correctness guarantees:
  1. Each period fetches its OWN dataset restricted to [warmup_start, bt_end].
  2. Loop exactly matches app's if/else structure (no same-bar re-entry).
  3. B&H and strat_ret computed over [bt_start, bt_end] only.
  4. MSTR and MSTU use per-asset bt_start (MSTU OOS starts 2025-06-04).
"""
import sys, warnings, joblib, requests
warnings.filterwarnings("ignore")
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import yfinance as yf

_MODEL_PATH = _ROOT / "models" / "inference_assets_ct.joblib"
print("Loading model …")
AD = joblib.load(str(_MODEL_PATH))

WARMUP   = 35
MSTR_SL  = 0.03
MSTU_SL  = 0.07
INIT_CAP = 100_000.0

_MACRO_SYMS = {"eth":"ETH-USD","spx":"^GSPC","ndx":"^IXIC",
               "vix":"^VIX","gold":"GC=F","dxy":"DX-Y.NYB","tnx":"^TNX"}
_ONCHAIN = ["hash-rate","difficulty","n-transactions","miners-revenue",
            "n-unique-addresses","transaction-fees-usd","mempool-size",
            "estimated-transaction-volume-usd","market-cap",
            "avg-block-size","cost-per-transaction"]

def _yf(ticker, start, end):
    d = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(d.columns, pd.MultiIndex): d.columns = [c[0] for c in d.columns]
    d.index = pd.DatetimeIndex(d.index).tz_localize(None).normalize()
    return d

def fetch_btc(start, end):
    d = _yf("BTC-USD", start, end)
    frames = {"btc_close":d["Close"],"btc_high":d["High"],"btc_low":d["Low"],"btc_volume":d["Volume"]}
    for nm, sym in _MACRO_SYMS.items():
        try: frames[f"{nm}_close"] = _yf(sym, start, end)["Close"]
        except: pass
    df = pd.DataFrame(frames).sort_index().ffill(limit=5)
    ok = 0
    for m in _ONCHAIN:
        try:
            r = requests.get(f"https://api.blockchain.info/charts/{m}",
                params={"timespan":"3years","format":"json","sampled":"true"}, timeout=20)
            vals = r.json().get("values",[])
            s = pd.Series({pd.Timestamp(v["x"],unit="s").normalize(): v["y"] for v in vals},
                name=f"oc_{m.replace('-','_')}", dtype=float)
            s = s[~s.index.duplicated(keep="last")].sort_index()
            s.index = pd.DatetimeIndex(s.index).tz_localize(None)
            df[s.name] = s.reindex(df.index).ffill(limit=7)
            ok += 1
        except: pass
    print(f"    on-chain {ok}/{len(_ONCHAIN)} OK")
    return df

def fetch_px(ticker, start, end):
    try:
        d = _yf(ticker, start, end)["Close"].sort_index()
        all_days = pd.date_range(d.index[0], max(d.index[-1], pd.Timestamp(end)), freq="D")
        return d.reindex(all_days).ffill()
    except Exception as e:
        print(f"  [WARN] {ticker}: {e}")
        return pd.Series(dtype=float)

def build_preds(df):
    c, h, l_, v = df["btc_close"], df["btc_high"], df["btc_low"], df["btc_volume"]
    ret = np.log(c).diff()
    f = pd.DataFrame(index=df.index)
    for k in [1,3,5,7,14,30]: f[f"ret_{k}"] = ret.rolling(k).sum()
    for k in [5,10,20,30]:    f[f"vol_{k}"] = ret.rolling(k).std()
    pc = c.shift(1)
    f["hl_ratio"] = (h-l_)/pc.clip(lower=1)
    f["hi_ret"]   = (h-pc)/pc.clip(lower=1)
    f["lo_ret"]   = (l_-pc)/pc.clip(lower=1)
    for nm in _MACRO_SYMS:
        col = f"{nm}_close"
        if col in df.columns:
            s = np.log(df[col]).diff()
            f[f"{nm}_ret1"] = s; f[f"{nm}_ret5"] = s.rolling(5).sum()
    for col in df.columns:
        if col.startswith("oc_"): f[col] = df[col]
    feat_cols = AD.get("feat_cols",[])
    for col in feat_cols:
        if col not in f.columns: f[col] = 0.0
    f = f.ffill(limit=5).fillna(0.0)
    X = f[feat_cols].values if feat_cols else np.zeros((len(f),1))
    constituents = AD.get("constituents",[])
    if constituents:
        hi_p, lo_p = [], []
        for sub in constituents:
            hm, lm = sub.get("hi_model"), sub.get("lo_model")
            if hm and lm:
                try:
                    hi_p.append(hm.predict(X)*sub.get("sigma_hi",1)+sub.get("mu_hi",0))
                    lo_p.append(lm.predict(X)*sub.get("sigma_lo",1)+sub.get("mu_lo",0))
                except: pass
        ph_r = np.mean(hi_p,axis=0) if hi_p else np.zeros(len(f))
        pl_r = np.mean(lo_p,axis=0) if lo_p else np.zeros(len(f))
    else:
        hm, lm = AD.get("hi_model"), AD.get("lo_model")
        ph_r = hm.predict(X)*AD.get("sigma_hi",1)+AD.get("mu_hi",0) if hm else np.zeros(len(f))
        pl_r = lm.predict(X)*AD.get("sigma_lo",1)+AD.get("mu_lo",0) if lm else np.zeros(len(f))
    out = pd.DataFrame({
        "close_asof": c.shift(1).values, "pred_high": np.roll(c.values*(1+ph_r/100),1),
        "pred_low":   np.roll(c.values*(1+pl_r/100),1),
        "actual_high":h.values, "actual_low":l_.values,
    }, index=df.index)
    return out.iloc[1:]

def compute_sigs(comp, raw_df):
    N = len(comp)
    ca = comp["close_asof"].values.astype(float)
    ph = comp["pred_high"].values.astype(float)
    pl = comp["pred_low"].values.astype(float)
    ah = comp["actual_high"].values.astype(float)
    al = comp["actual_low"].values.astype(float)
    err_hi = (ah-ph)/ca*100; err_lo = (pl-al)/ca*100
    hi_b = (ah>ph).astype(int); lo_b = (al<pl).astype(int)
    ehma3=np.zeros(N); elma3=np.zeros(N)
    hb3=np.zeros(N,dtype=int); lb3=np.zeros(N,dtype=int)
    for i in range(N):
        s=max(0,i-2)
        ehma3[i]=np.mean(err_hi[s:i+1]); elma3[i]=np.mean(err_lo[s:i+1])
        hb3[i]=int(np.sum(hi_b[s:i+1])); lb3[i]=int(np.sum(lo_b[s:i+1]))
    u1=(ehma3>0.7)&(hb3>=2)
    d1=(lb3>=2)&(elma3>0.5); d2=ehma3<-0.75
    d3=np.zeros(N,dtype=bool)
    for i in range(1,N):
        consec=0
        for k in range(i-1,-1,-1):
            if hi_b[k]: consec+=1
            else: break
        if consec>=3 and lo_b[i]: d3[i]=True
    ma30=np.full(N,np.nan)
    for i in range(N):
        w=min(30,i+1); ma30[i]=np.mean(ca[max(0,i-w+1):i+1])
    above_ma30=ca>ma30
    ms_pos=np.zeros(N,dtype=bool)
    for i in range(N):
        if i>=5 and np.isfinite(ma30[i]) and np.isfinite(ma30[i-5]):
            ms_pos[i]=ma30[i]>ma30[i-5]
    bull=above_ma30&ms_pos
    c10=np.zeros(N,dtype=bool)
    for i in range(N):
        li=max(0,i-7); c10[i]=not bool(np.any(d1[li:i]|d2[li:i]))
    rn=np.array([float(np.mean(err_hi[max(0,i-29):i+1])) for i in range(N)])
    dns=np.zeros(N)
    for i in range(N):
        nm=max(abs(rn[i]),0.01)
        dns[i]=(-ehma3[i]/nm)*0.30+(lb3[i]/3.0)*0.30+(elma3[i]/max(abs(elma3[i]),0.10))*0.20+float(lo_b[i])*0.20
    vrb=(dns>0.8)&(err_lo>3.0)
    vr=np.zeros(N,dtype=bool)
    for i in range(N): vr[i]=bool(np.any(vrb[max(0,i-2):i+1]))

    # ── SPX MA20 slope (for V6) ────────────────────────────────────────────
    dates = comp.index
    spx_raw = raw_df["spx_close"].reindex(dates).ffill() if "spx_close" in raw_df.columns \
              else pd.Series(np.nan, index=dates)
    spx_c = spx_raw.values.astype(float)
    spx_ma20 = pd.Series(spx_c).rolling(20, min_periods=10).mean().values
    spx_slope_pos = np.zeros(N, dtype=bool)
    for i in range(5, N):
        if np.isfinite(spx_ma20[i]) and np.isfinite(spx_ma20[i-5]):
            spx_slope_pos[i] = spx_ma20[i] > spx_ma20[i-5]

    # V1 entry
    tf1_v1 = u1 & ((above_ma30 ^ c10) | vr)
    # V6: SPX slope alignment — V-reversal overrides gate
    tf1_v6 = tf1_v1 & (spx_slope_pos | vr)

    return dict(N=N, dates=dates, ca=ca, above_ma30=above_ma30,
                bull=bull, vr=vr, d1=d1, d2=d2, d3=d3,
                u1=u1, spx_slope_pos=spx_slope_pos,
                tf1_v1=tf1_v1, tf1_v6=tf1_v6)

def run_bt(sigs, bt_start_iso, bt_end_iso, asset_px, sl_pct, label=""):
    N = sigs["N"]; dates = sigs["dates"]
    bt_start = pd.Timestamp(bt_start_iso); bt_end = pd.Timestamp(bt_end_iso)
    _bt_mask  = dates >= bt_start
    _end_mask = dates <= bt_end
    _bt0 = max(WARMUP, int(np.argmax(_bt_mask)) if _bt_mask.any() else N)
    _btN = int(np.sum(_end_mask))

    px = asset_px.reindex(dates).ffill().values.astype(float)

    nav=INIT_CAP; pos="CASH"; qty=0.0
    ep=en=ed=et=None; stop_px=0.0
    from_sl=False; bssl=0
    trades=[]; nav_arr=np.full(N, np.nan)

    d1=sigs["d1"]; d2=sigs["d2"]; d3=sigs["d3"]
    bull=sigs["bull"]; am30=sigs["above_ma30"]; vr=sigs["vr"]

    for i in range(_bt0, _btN):
        pr = px[i]
        if not np.isfinite(pr) or pr <= 0:
            nav_arr[i] = qty*px[i-1] if pos=="LONG" and i>0 else nav; continue
        entry_sig = sigs[_entry_key][i]
        if pos == "LONG":
            if pr <= stop_px:
                nav = qty*pr
                trades.append(dict(entry_date=ed, exit_date=dates[i],
                    entry_price=ep, exit_price=pr,
                    pnl_pct=(pr/ep-1)*100, exit_signal="SL", entry_trigger=et))
                pos="CASH"; qty=0.0; stop_px=0.0
                from_sl=True; bssl=0
            else:
                sx = bool(d3[i] or (d2[i] and not bull[i]))
                if sx:
                    nav = qty*pr
                    trades.append(dict(entry_date=ed, exit_date=dates[i],
                        entry_price=ep, exit_price=pr,
                        pnl_pct=(pr/ep-1)*100,
                        exit_signal="D3" if d3[i] else "D2",
                        entry_trigger=et))
                    pos="CASH"; qty=0.0; stop_px=0.0; from_sl=False
                else:
                    nav = qty*pr
        else:
            if from_sl: bssl+=1
            _ex = bool(d3[i] or (d2[i] and not bull[i]))
            _ok = not from_sl or bool(bull[i]) or bssl>=10
            if entry_sig and _ok and (from_sl or not _ex):
                qty=nav/pr; ep=pr; en=nav; ed=dates[i]
                et = "U1+V" if vr[i] else ("U1+MA30" if am30[i] else "U1+Clean")
                stop_px=pr*(1-sl_pct)
                pos="LONG"; from_sl=False; bssl=0
        nav_arr[i] = qty*pr if pos=="LONG" else nav

    nav_s = pd.Series(nav_arr[_bt0:_btN], index=dates[_bt0:_btN]).ffill()
    bh_s  = pd.Series(INIT_CAP*px[_bt0:_btN]/px[_bt0], index=dates[_bt0:_btN])
    final_nav = float(nav_s.iloc[-1])
    strat_ret = (final_nav/INIT_CAP-1)*100
    bh_ret    = (float(bh_s.iloc[-1])/INIT_CAP-1)*100
    rm = nav_s.cummax(); max_dd = float(((nav_s-rm)/rm*100).min())
    n_wins = sum(1 for t in trades if t["pnl_pct"]>0)
    n_tot  = len(trades)
    win_r  = 100*n_wins/n_tot if n_tot else 0.0
    return dict(trades=trades, strat_ret=strat_ret, bh_ret=bh_ret,
                max_dd=max_dd, win_rate=win_r, n_trades=n_tot, n_wins=n_wins)

def mstu_synth(mstr_px, mstu_px, from_date):
    launch = pd.Timestamp("2024-09-18")
    om = mstr_px.loc[launch:]; ou = mstu_px.loc[launch:]
    common = om.index.intersection(ou.index)
    if len(common) < 10: return mstu_px
    lrm = np.log(om.loc[common]).diff().dropna()
    lru = np.log(ou.loc[common]).diff().dropna()
    idx = lrm.index.intersection(lru.index); lrm,lru=lrm.loc[idx],lru.loc[idx]
    beta = float(np.cov(lrm,lru)[0,1]/np.var(lrm)) if np.var(lrm)>0 else 2.0
    alpha= float(np.mean(lru)-beta*np.mean(lrm))
    lp = mstu_px.loc[mstu_px.index>=launch].iloc[0]
    need = mstr_px.loc[from_date:launch].sort_index()
    synth={need.index[-1]:lp}
    for i in range(len(need)-2,-1,-1):
        lr_m = np.log(need.iloc[i+1]/need.iloc[i]) if need.iloc[i]>0 else 0
        synth[need.index[i]] = synth[need.index[i+1]]/np.exp(alpha+beta*lr_m)
    s = pd.Series(synth).sort_index()
    combined = pd.concat([s, mstu_px.loc[launch:]]).sort_index()
    return combined[~combined.index.duplicated(keep="last")]

# ── periods (matching the UI exactly) ─────────────────────────────────────────
# (label, warmup_start, mstr_bt_start, mstu_bt_start, bt_end, use_synth_mstu)
PERIODS = [
    ("Bear Jun25–May26", "2025-03-01", "2025-06-01", "2025-06-04", "2026-05-31", False),
    ("Bull Jun24–Jun25", "2024-03-01", "2024-06-05", "2024-06-05", "2025-06-14", True),
    ("OOS (rolling)",   "2024-03-01", "2024-05-26", "2025-06-04", "2026-06-13", False),
    ("Full Jun24–May26", "2024-03-01", "2024-06-01", "2024-06-01", "2026-05-31", True),
]

global _entry_key
_entry_key = "tf1_v1"

print("\n" + "="*70)
print("Running per-period backtests (V1 baseline vs V6: SPX trend gate) …")

all_results = {}

for label, warmup_start, mstr_bt_start, mstu_bt_start, bt_end, use_synth in PERIODS:
    print(f"\n{'─'*60}")
    print(f"Period: {label}  [MSTR:{mstr_bt_start} | MSTU:{mstu_bt_start} → {bt_end}]")
    fetch_end_dt = (pd.Timestamp(bt_end) + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    print(f"  Fetching BTC+SPX [{warmup_start} → {fetch_end_dt}] …")
    raw = fetch_btc(warmup_start, fetch_end_dt)
    mstr_px = fetch_px("MSTR", warmup_start, fetch_end_dt)
    mstu_raw = fetch_px("MSTU", "2024-09-01", fetch_end_dt)
    mstu_px  = mstu_synth(mstr_px, mstu_raw, warmup_start) if use_synth else mstu_raw

    print("  Building preds + signals …")
    comp = build_preds(raw)
    sigs = compute_sigs(comp, raw)

    # SPX gate coverage stats
    mstr_mask = (sigs["dates"] >= pd.Timestamp(mstr_bt_start)) & (sigs["dates"] <= pd.Timestamp(bt_end))
    spx_up_pct = 100.0 * sigs["spx_slope_pos"][mstr_mask].sum() / mstr_mask.sum()
    v1_in = int(sigs["tf1_v1"][mstr_mask].sum())
    v6_in = int(sigs["tf1_v6"][mstr_mask].sum())
    print(f"  SPX slope positive: {spx_up_pct:.0f}% of bars | "
          f"V1 signals: {v1_in} | V6 signals: {v6_in} | Blocked: {v1_in - v6_in}")

    _entry_key = "tf1_v1"
    rv1_mstr = run_bt(sigs, mstr_bt_start, bt_end, mstr_px, MSTR_SL, label)
    rv1_mstu = run_bt(sigs, mstu_bt_start, bt_end, mstu_px, MSTU_SL, label)

    _entry_key = "tf1_v6"
    rv6_mstr = run_bt(sigs, mstr_bt_start, bt_end, mstr_px, MSTR_SL, label)
    rv6_mstu = run_bt(sigs, mstu_bt_start, bt_end, mstu_px, MSTU_SL, label)

    all_results[label] = dict(v1_mstr=rv1_mstr, v6_mstr=rv6_mstr,
                              v1_mstu=rv1_mstu, v6_mstu=rv6_mstu)

    for asset, v1, v6 in [("MSTR", rv1_mstr, rv6_mstr), ("MSTU", rv1_mstu, rv6_mstu)]:
        d = v6["strat_ret"] - v1["strat_ret"]
        print(f"  {asset}: B&H {v1['bh_ret']:+.1f}% | "
              f"V1 {v1['strat_ret']:+.1f}% ({v1['n_trades']}t, {v1['win_rate']:.0f}%, DD {v1['max_dd']:.1f}%) | "
              f"V6 {v6['strat_ret']:+.1f}% ({v6['n_trades']}t, {v6['win_rate']:.0f}%, DD {v6['max_dd']:.1f}%) | "
              f"Δ {d:+.1f}pp")

    print(f"\n  V1 MSTR trades blocked by V6 (SPX slope negative, no v_recent):")
    v1_t = {(pd.Timestamp(t["entry_date"]).date() if not isinstance(t["entry_date"], pd.Timestamp)
              else t["entry_date"].date()): t for t in rv1_mstr["trades"]}
    v6_t = {(pd.Timestamp(t["entry_date"]).date() if not isinstance(t["entry_date"], pd.Timestamp)
              else t["entry_date"].date()): t for t in rv6_mstr["trades"]}
    blocked_trades = [(d, t) for d, t in v1_t.items() if d not in v6_t]
    if blocked_trades:
        for d, t in sorted(blocked_trades):
            # show SPX slope at entry
            entry_ts = pd.Timestamp(d)
            idx_match = sigs["dates"].searchsorted(entry_ts)
            spx_ok = bool(sigs["spx_slope_pos"][idx_match]) if idx_match < sigs["N"] else "?"
            vr_ok  = bool(sigs["vr"][idx_match]) if idx_match < sigs["N"] else "?"
            print(f"    {d}  trig={t['entry_trigger']:<12}  P&L={t['pnl_pct']:+.1f}%  "
                  f"exit={t['exit_signal']}  spx_slope={spx_ok}  v_recent={vr_ok}")
    else:
        print("    (none)")

print("\n" + "="*70)
print("SUMMARY V1 vs V6 (Idea 5: SPX Trend Alignment Gate)  [as of 2026-06-13]")
print(f"{'Period':<22} {'Asset':<6} {'B&H':>8} {'V1':>8} {'V6':>8} {'Δ':>8} "
      f"{'V1 Win%':>8} {'V6 Win%':>8} {'V1 #':>6} {'V6 #':>6} {'V1 DD':>8} {'V6 DD':>8}")
print("-"*115)
for label, *_ in PERIODS:
    r = all_results[label]
    for asset, v1, v6 in [("MSTR", r["v1_mstr"], r["v6_mstr"]),
                           ("MSTU", r["v1_mstu"], r["v6_mstu"])]:
        d = v6["strat_ret"] - v1["strat_ret"]
        print(f"{label:<22} {asset:<6} {v1['bh_ret']:>+7.1f}% {v1['strat_ret']:>+7.1f}% "
              f"{v6['strat_ret']:>+7.1f}% {d:>+6.1f}pp  {v1['win_rate']:>6.0f}%  "
              f"{v6['win_rate']:>6.0f}%  {v1['n_trades']:>5d}  {v6['n_trades']:>5d}  "
              f"{v1['max_dd']:>6.1f}%  {v6['max_dd']:>6.1f}%")
print("\nDone.")
