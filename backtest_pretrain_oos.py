"""
Pre-Training OOS Backtest  —  TF2 + V-Gate
===========================================
Tests the CURRENT live model on data from BEFORE its training window.

Model training window : 2019-12-22 → 2026-02-27
This backtest         : 2018-01-01 → 2019-12-21   (fully OOS — model never saw this data)

Why this matters: the model was fit entirely on 2019-2026 data.
Running signals on 2018-2019 gives a genuine out-of-sample stress test,
covering the 2018 bear market (-84%) and the 2019 recovery / consolidation.

Variants tested
───────────────
  A  TF2 Baseline       U1 AND (above_MA30 OR clean_10d)
  B  TF2+V-Gate 3-bar   U1 AND (above_MA30 OR clean_10d OR v_recent_3)   ← live strategy
"""

import sys, os, warnings, joblib, requests
warnings.filterwarnings("ignore")
from pathlib import Path
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import yfinance as yf

TRAIN_START     = pd.Timestamp("2019-12-22")   # first day in model training data
BACKTEST_START  = "2018-01-01"
BACKTEST_END    = "2019-12-21"                 # last day before training starts
FETCH_START     = "2017-06-01"                 # enough for 90-bar lookbacks + 35-bar warmup
INITIAL_CAP     = 100_000.0

print(f"Model training starts : {TRAIN_START.date()}")
print(f"Backtest window       : {BACKTEST_START} → {BACKTEST_END}  (ALL OOS)")

# ── Model ──────────────────────────────────────────────────────────────────────
AD = joblib.load(str(_ROOT / "models" / "inference_assets_ct.joblib"))
print(f"Model loaded  |  feat_cols: {len(AD['feat_cols'])}")
meta = AD.get("calibration_meta", {})
print(f"Model train_start={meta.get('train_start')}  train_end={meta.get('train_end')}")

# ── Data helpers ───────────────────────────────────────────────────────────────
_MACRO_SYMS = {
    "eth":"ETH-USD", "spx":"^GSPC", "ndx":"^IXIC",
    "vix":"^VIX", "gold":"GC=F", "dxy":"DX-Y.NYB", "tnx":"^TNX"
}
_ONCHAIN = [
    "hash-rate","difficulty","n-transactions","miners-revenue",
    "n-unique-addresses","transaction-fees-usd","mempool-size",
    "estimated-transaction-volume-usd","market-cap",
    "avg-block-size","cost-per-transaction",
]

def _yf(sym, start, end):
    d = yf.download(sym, start=start, end=end, progress=False, auto_adjust=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] for c in d.columns]
    d.index = pd.DatetimeIndex(d.index).tz_localize(None).normalize()
    return d

def fetch_data(start, end):
    print(f"\nFetching {start} → {end} …")
    frames = {}
    d = _yf("BTC-USD", start, end)
    frames.update({
        "btc_close": d["Close"], "btc_high": d["High"],
        "btc_low":   d["Low"],   "btc_volume": d["Volume"],
    })
    for name, sym in _MACRO_SYMS.items():
        try:
            frames[f"{name}_close"] = _yf(sym, start, end)["Close"]
        except Exception:
            pass
    df = pd.DataFrame(frames).ffill(limit=5)
    df.index.name = "date"
    # On-chain: use "all" to get maximum historical depth
    print("  On-chain (timespan=all) …", end=" ", flush=True)
    ok = 0
    for m in _ONCHAIN:
        col = f"oc_{m.replace('-','_')}"
        try:
            r = requests.get(
                f"https://api.blockchain.info/charts/{m}",
                params={"timespan": "all", "format": "json", "sampled": "true"},
                timeout=30,
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
    print(f"{ok}/{len(_ONCHAIN)} OK")
    return df

def build_preds(df):
    c, h, l_, v = (df["btc_close"], df["btc_high"],
                   df["btc_low"],   df["btc_volume"])
    ret = np.log(c).diff()
    f   = pd.DataFrame(index=df.index)
    for k in [1,3,5,7,14,30]: f[f"ret_{k}"]  = ret.rolling(k).sum()
    for k in [5,10,20,30]:     f[f"vol_{k}"]  = ret.rolling(k).std()
    prev_c = c.shift(1)
    tr = pd.concat([(h-l_),(h-prev_c).abs(),(l_-prev_c).abs()], axis=1).max(axis=1)
    for k in [7,14,30]: f[f"atr_{k}"] = tr.rolling(k).mean() / c
    f["range_today"]  = (h-l_)/c
    f["range_ma7"]    = ((h-l_)/c).rolling(7).mean()
    f["range_ma30"]   = ((h-l_)/c).rolling(30).mean()
    f["range_std30"]  = ((h-l_)/c).rolling(30).std()
    g  = c.diff().clip(lower=0).rolling(14).mean()
    ls = (-c.diff().clip(upper=0)).rolling(14).mean()
    f["rsi_14"] = 100 - 100 / (1 + g / ls.replace(0, np.nan))
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    f["macd"]      = macd / c
    f["macd_sig"]  = macd.ewm(span=9, adjust=False).mean() / c
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    ma20 = c.rolling(20).mean(); sd20 = c.rolling(20).std()
    f["bb_width"]    = (4 * sd20) / ma20
    f["dist_hi_30"]  = c / c.rolling(30).max() - 1
    f["dist_lo_30"]  = c / c.rolling(30).min() - 1
    f["dist_hi_90"]  = c / c.rolling(90).max() - 1
    f["vol_chg_1"]   = np.log(v).diff()
    f["vol_z_20"]    = (np.log(v) - np.log(v).rolling(20).mean()) / np.log(v).rolling(20).std()
    f["vol_ma_ratio"] = v / v.rolling(20).mean()
    dow = df.index.dayofweek
    for i in range(6): f[f"dow_{i}"] = (dow == i).astype(float)
    for nm, col in [
        ("spx","spx_close"),("ndx","ndx_close"),("vix","vix_close"),
        ("gold","gold_close"),("dxy","dxy_close"),("tnx","tnx_close"),("eth","eth_close"),
    ]:
        if col not in df.columns: continue
        s  = df[col]; lr = np.log(s).diff()
        for k in [1,5,20]: f[f"{nm}_ret_{k}"] = lr.rolling(k).sum()
        f[f"{nm}_vol_20"] = lr.rolling(20).std()
    for cn, cc in [("spx","spx_close"),("ndx","ndx_close"),
                   ("gold","gold_close"),("dxy","dxy_close")]:
        if cc in df.columns:
            f[f"btc_{cn}_corr_30"] = ret.rolling(30).corr(np.log(df[cc]).diff())
    for col in [x for x in df.columns if x.startswith("oc_")]:
        s  = df[col].astype(float)
        sl = np.log(s.replace(0, np.nan))
        f[f"{col}_d1"]  = sl.diff(1)
        f[f"{col}_d7"]  = sl.diff(7)
        f[f"{col}_z30"] = (sl - sl.rolling(30).mean()) / sl.rolling(30).std()
    nh, nl  = h.shift(-1), l_.shift(-1)
    yh      = (nh - c) / c; yl = (c - nl) / c
    f["y_hi_ema3"] = yh.shift(1).ewm(span=3, adjust=False).mean()
    f["y_lo_ema3"] = yl.shift(1).ewm(span=3, adjust=False).mean()
    f["y_hi_ema7"] = yh.shift(1).ewm(span=7, adjust=False).mean()
    f["y_lo_ema7"] = yl.shift(1).ewm(span=7, adjust=False).mean()
    p3h = h.shift(1).rolling(3).max(); p3l = l_.shift(1).rolling(3).min()
    f["above_3d_high"]   = (c > p3h).astype(float)
    f["below_3d_low"]    = (c < p3l).astype(float)
    f["bo_strength_up"]  = (c / p3h - 1).clip(lower=0)
    f["bo_strength_dn"]  = (1 - c / p3l).clip(lower=0)
    ya = yh.shift(1); yb = yl.shift(1)
    f["y_hi_surprise"] = ya - ya.ewm(span=7, adjust=False).mean()
    f["y_lo_surprise"] = yb - yb.ewm(span=7, adjust=False).mean()
    nr = ret.clip(upper=0)
    f["dn_vol_5"]        = nr.rolling(5).std()
    f["dn_vol_20"]       = nr.rolling(20).std()
    sma50 = c.rolling(50).mean()
    f["below_sma50"]     = (c < sma50).astype(float)
    f["below_sma50_5d"]  = f["below_sma50"].rolling(5).min().fillna(0)

    fc = AD["feat_cols"]
    f  = f.replace([np.inf, -np.inf], np.nan)
    for col in fc:
        if col not in f.columns: f[col] = np.nan
    F = f[fc].dropna()
    if F.empty: raise RuntimeError("Feature matrix is empty — not enough history")
    print(f"  Predictions: {len(F)} rows  ({F.index[0].date()} → {F.index[-1].date()})")

    if AD.get("ensemble") and AD.get("constituents"):
        yhi = np.mean([con["m_hi"].predict(F) for con in AD["constituents"]], axis=0)
        ylo = np.mean([con["m_lo"].predict(F) for con in AD["constituents"]], axis=0)
        if AD.get("blended") and float(AD.get("alpha", 1.0)) < 1.0:
            a   = float(AD["alpha"])
            yhi = a*yhi + (1-a)*float(AD.get("mu_hi", 0))
            ylo = a*ylo + (1-a)*float(AD.get("mu_lo", 0))
    else:
        yhi = AD["hi_model"].predict(F)
        ylo = AD["lo_model"].predict(F)

    cv  = c.reindex(F.index).values
    ph  = cv * (1 + np.clip(yhi, 0, None))
    pl  = cv * (1 - np.clip(ylo, 0, None))
    idx = np.asarray(F.index, dtype="datetime64[ns]")
    nd  = np.empty(len(F), dtype="datetime64[ns]")
    nd[:-1] = idx[1:]; nd[-1] = idx[-1] + np.timedelta64(1, "D")
    res = pd.DataFrame(
        {"close_asof": cv, "pred_high": ph, "pred_low": pl},
        index=pd.DatetimeIndex(nd, name="target_date"),
    )
    return res[~res.index.duplicated(keep="last")]


# ── Signal arrays (identical to backtest_v_entry_comparison.py) ────────────────
def build_signals(comp):
    N    = len(comp)
    ca   = comp["close_asof"].values.astype(float)
    ep   = comp["actual_close"].values.astype(float)
    ph   = comp["pred_high"].values.astype(float)
    pl   = comp["pred_low"].values.astype(float)
    ah   = comp["actual_high"].values.astype(float)
    al   = comp["actual_low"].values.astype(float)

    ehi  = (ah - ph) / ca * 100
    elo  = (pl - al) / ca * 100
    hb   = (ah > ph).astype(int)
    lb   = (al < pl).astype(int)

    ehma3 = np.zeros(N); elma3 = np.zeros(N)
    hb3   = np.zeros(N, dtype=int); lb3 = np.zeros(N, dtype=int)
    for i in range(N):
        s       = max(0, i-2)
        ehma3[i] = np.mean(ehi[s:i+1])
        elma3[i] = np.mean(elo[s:i+1])
        hb3[i]   = int(np.sum(hb[s:i+1]))
        lb3[i]   = int(np.sum(lb[s:i+1]))

    u1 = (ehma3 > 0.5) & (hb3 >= 2)
    d1 = (lb3 >= 2) & (elma3 > 0.5)
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
        w = min(30, i+1); ma30[i] = np.mean(ca[max(0, i-w+1):i+1])
    abv  = ca > ma30
    slp  = np.zeros(N, dtype=bool)
    for i in range(5, N):
        if np.isfinite(ma30[i]) and np.isfinite(ma30[i-5]):
            slp[i] = ma30[i] > ma30[i-5]
    bull = abv & slp

    c10 = np.zeros(N, dtype=bool)
    for i in range(N):
        li   = max(0, i-10)
        c10[i] = not bool(np.any(d1[li:i] | d2[li:i]))

    cum_ehi = np.array([np.mean(ehi[:i+1]) for i in range(N)])
    dn_score = np.zeros(N)
    for i in range(N):
        norm        = max(abs(cum_ehi[i]), 0.01)
        dn_score[i] = (
            (-ehma3[i] / norm)                         * 0.30 +
            (lb3[i]    / 3.0)                          * 0.30 +
            (elma3[i]  / max(abs(elma3[i]), 0.10))     * 0.20 +
            float(lb[i])                               * 0.20
        )

    v_rev_bar  = (dn_score > 0.8) & (elo > 5.0)
    v_recent_3 = np.zeros(N, dtype=bool)
    for i in range(N):
        v_recent_3[i] = bool(np.any(v_rev_bar[max(0, i-2):i+1]))

    return dict(
        N=N, ca=ca, ep=ep, ehi=ehi, elo=elo, hb=hb, lb=lb,
        ehma3=ehma3, elma3=elma3, hb3=hb3, lb3=lb3,
        u1=u1, d1=d1, d2=d2, d3=d3,
        ma30=ma30, abv=abv, slp=slp, bull=bull, c10=c10,
        dn_score=dn_score, v_rev_bar=v_rev_bar, v_recent_3=v_recent_3,
    )


# ── Backtest engine ────────────────────────────────────────────────────────────
def run_backtest(dates, sigs, use_vgate: bool, cap=INITIAL_CAP):
    WARMUP = 35
    N      = sigs["N"]
    ep     = sigs["ep"]; abv = sigs["abv"]; c10 = sigs["c10"]
    u1     = sigs["u1"]; d2  = sigs["d2"]; d3  = sigs["d3"]; bull = sigs["bull"]
    v3     = sigs["v_recent_3"]

    nav = cap; pos = "CASH"; qty = 0.0
    ep_r = en = ed_r = et = None
    trades = []; nav_arr = np.full(N, np.nan)

    for i in range(N):
        si    = i - 1
        price = ep[i]
        if i < WARMUP:
            nav_arr[i] = cap; continue

        if pos == "LONG":
            cur = qty * price
            if si >= 0:
                should_exit = bool(d3[si] or (d2[si] and not bull[si]))
                xl = "D3" if d3[si] else "D2(bear)"
            else:
                should_exit = False; xl = "?"
            if should_exit:
                nav = cur
                trades.append(dict(
                    entry_date    = ed_r,    entry_price = ep_r,
                    entry_nav     = en,      entry_trigger = et,
                    exit_date     = dates[i], exit_price  = price,
                    exit_nav      = nav,
                    pnl_pct       = (price / ep_r - 1) * 100,
                    pnl_abs       = nav - en,
                    exit_signal   = xl,
                    duration_days = (dates[i] - ed_r).days,
                    regime        = "BULL" if (si >= 0 and bull[si]) else "BEAR",
                    v_entry       = et.startswith("V") if et else False,
                ))
                pos = "CASH"; qty = 0.0
            else:
                nav = cur
        else:
            allow = False; trigger = ""
            if si >= 0 and u1[si]:
                gate_abv = bool(abv[si])
                gate_c10 = bool(c10[si])
                gate_v3  = bool(v3[si]) if use_vgate else False
                if gate_abv or gate_c10 or gate_v3:
                    allow = True
                    if gate_v3 and not gate_abv and not gate_c10:
                        trigger = "V-recent gate"
                    elif gate_abv and gate_c10:
                        trigger = "U1+↑MA30+clean10d"
                    elif gate_abv:
                        trigger = "U1+↑MA30"
                    else:
                        trigger = "U1+clean10d"

            if allow:
                qty = nav / price; ep_r = price; ed_r = dates[i]; en = nav
                pos = "LONG"; et = trigger

        nav_arr[i] = qty * price if pos == "LONG" else nav

    if pos == "LONG":
        nav_arr[N-1] = qty * ep[N-1]

    nav_s = pd.Series(nav_arr[WARMUP:], index=dates[WARMUP:]).ffill()
    bh_s  = pd.Series(cap * ep[WARMUP:] / ep[WARMUP], index=dates[WARMUP:])
    return dict(trades=trades, nav=nav_s, bh=bh_s,
                open_pos=(pos == "LONG"),
                open_entry=(dict(price=ep_r, date=ed_r, nav=en) if pos=="LONG" else None))


# ── Report helpers ─────────────────────────────────────────────────────────────
def summarise(res, label, rf=0.045):
    nav = res["nav"]; bh = res["bh"]; trades = res["trades"]
    cap = INITIAL_CAP
    fn  = nav.iloc[-1]; fb = bh.iloc[-1]
    n_y = (nav.index[-1] - nav.index[0]).days / 365.25
    dr  = nav.pct_change().fillna(0); rfd = (1+rf)**(1/252) - 1
    exc = dr - rfd
    sharpe  = exc.mean()/exc.std()*np.sqrt(252) if exc.std() > 0 else 0
    rm      = nav.cummax(); max_dd = float(((nav-rm)/rm*100).min())
    wins    = [t for t in trades if t["pnl_pct"] > 0]
    days_in = sum(t["duration_days"] for t in trades)
    tot_days = max(1, (nav.index[-1] - nav.index[0]).days)
    return dict(
        label      = label,
        final_nav  = fn,   bh_nav   = fb,
        ret_pct    = (fn/cap-1)*100,
        bh_ret_pct = (fb/cap-1)*100,
        alpha      = fn - fb,
        cagr       = ((fn/cap)**(1/n_y)-1)*100 if n_y > 0 else 0,
        sharpe     = sharpe,   max_dd   = max_dd,
        n_trades   = len(trades),
        win_rate   = 100*len(wins)/len(trades) if trades else 0,
        avg_pnl    = float(np.mean([t["pnl_pct"] for t in trades])) if trades else 0,
        time_in    = 100*days_in/tot_days,
        n_v        = sum(1 for t in trades if t.get("v_entry")),
    )


def print_trade_log(trades_a, trades_b, label="ALL"):
    all_dates = sorted(set(t["entry_date"] for t in trades_a + trades_b))
    W = 130
    print(f"\n{'═'*W}")
    print(f"TRADE LOG  —  {label}")
    print(f"{'─'*W}")
    print(f"  {'Entry':>12}  {'Exit':>12}  {'Entry $':>10}  {'Exit $':>10}"
          f"  {'PnL%':>7}  {'Days':>5}  {'Regime':6}  {'Trigger':30}  {'[A]':5}  {'[B]':5}")
    print(f"{'─'*W}")
    for edt in all_dates:
        ta = next((t for t in trades_a if t["entry_date"]==edt), None)
        tb = next((t for t in trades_b if t["entry_date"]==edt), None)
        canon = ta or tb
        a_flag = "  ✓  " if ta else "  —  "
        b_flag = "  ✓  " if tb else "  —  "
        v_flag = "★ " if canon.get("v_entry") else "  "
        pnl    = canon["pnl_pct"]
        tick   = "✓" if pnl > 0 else "✗"
        print(f"  {pd.Timestamp(canon['entry_date']).strftime('%b %d, %Y'):>12}"
              f"  {pd.Timestamp(canon['exit_date']).strftime('%b %d, %Y'):>12}"
              f"  ${canon['entry_price']:>9,.0f}  ${canon['exit_price']:>9,.0f}"
              f"  {pnl:>+6.1f}% {tick}  {canon['duration_days']:>4}d"
              f"  {canon['regime']:6}  {v_flag}{canon['entry_trigger']:28}"
              f"  {a_flag}  {b_flag}")
    print(f"{'═'*W}")


def print_summary(metrics):
    W = 22
    labels = [m["label"] for m in metrics]
    sep = "─" * (30 + W*len(metrics))
    print(f"\n{'═'*(30+W*len(metrics))}")
    print(f"{'SUMMARY  (Pre-Training OOS)':^{30+W*len(metrics)}}")
    print(sep)
    print(f"  {'Metric':<28}" + "".join(f"{l:>{W}}" for l in labels))
    print(sep)
    def row(lbl, key, fmt):
        print(f"  {lbl:<28}" + "".join(f"{fmt.format(m[key]):>{W}}" for m in metrics))
    row("Final NAV",        "final_nav",  "${:,.0f}")
    row("Total Return",     "ret_pct",    "{:+.1f}%")
    row("B&H Return",       "bh_ret_pct", "{:+.1f}%")
    row("Alpha vs B&H",     "alpha",      "${:+,.0f}")
    row("CAGR",             "cagr",       "{:+.1f}%")
    row("Sharpe Ratio",     "sharpe",     "{:.2f}")
    row("Max Drawdown",     "max_dd",     "{:.1f}%")
    print(sep)
    row("# Trades",         "n_trades",   "{:d}")
    row("  of which V-gate","n_v",        "{:d}")
    row("Win Rate",         "win_rate",   "{:.1f}%")
    row("Avg Trade PnL",    "avg_pnl",    "{:+.1f}%")
    row("Time in Market",   "time_in",    "{:.1f}%")
    print(f"{'═'*(30+W*len(metrics))}")
    base = metrics[0]
    print(f"\n  Δ [B] vs [A] Baseline:")
    for m in metrics[1:]:
        print(f"    {m['label']:22}  "
              f"NAV {m['final_nav']-base['final_nav']:>+9,.0f}  "
              f"Sharpe {m['sharpe']-base['sharpe']:>+6.3f}  "
              f"MaxDD {m['max_dd']-base['max_dd']:>+5.1f}pp")


def print_v_signal_quality(dates, sigs):
    ep_arr   = sigs["ep"]
    v_bars   = np.where(sigs["v_rev_bar"])[0]
    print(f"\n{'═'*100}")
    print(f"V-REVERSAL SIGNAL QUALITY  (dn_score>0.8 & err_lo>5%)  —  {len(v_bars)} bars")
    print(f"{'─'*100}")
    print(f"  {'Date':>12}  {'dn_score':>9}  {'err_lo%':>8}  {'Price':>10}"
          f"  {'Fwd+3d':>7}  {'Fwd+5d':>7}  {'Fwd+10d':>8}  {'Fwd+20d':>8}  {'U1 same bar':>11}")
    print(f"{'─'*100}")
    n5 = n10 = n20 = 0
    for idx in v_bars:
        dt    = dates[idx]; price = ep_arr[idx]
        score = sigs["dn_score"][idx]; el = sigs["elo"][idx]
        u1s   = "✓" if sigs["u1"][idx] else "—"
        def fwd(k):
            j = idx + k
            return (ep_arr[j]/price-1)*100 if j < len(ep_arr) else float("nan")
        f3,f5,f10,f20 = fwd(3),fwd(5),fwd(10),fwd(20)
        if not np.isnan(f5)  and f5  > 0: n5  += 1
        if not np.isnan(f10) and f10 > 0: n10 += 1
        if not np.isnan(f20) and f20 > 0: n20 += 1
        print(f"  {dt.strftime('%b %d, %Y'):>12}  {score:>9.3f}  {el:>+8.2f}%  ${price:>9,.0f}"
              f"  {f3:>+7.2f}%  {f5:>+7.2f}%  {f10:>+8.2f}%  {f20:>+8.2f}%  {u1s:>11}")
    if v_bars.size:
        print(f"{'─'*100}")
        print(f"  Hit rate (fwd > 0):  +5d {n5}/{len(v_bars)} ({100*n5/len(v_bars):.0f}%)  "
              f"+10d {n10}/{len(v_bars)} ({100*n10/len(v_bars):.0f}%)  "
              f"+20d {n20}/{len(v_bars)} ({100*n20/len(v_bars):.0f}%)")
    print(f"{'═'*100}")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    df_raw = fetch_data(FETCH_START, BACKTEST_END)
    preds  = build_preds(df_raw)

    sd = pd.Timestamp(BACKTEST_START)
    ed = pd.Timestamp(BACKTEST_END)
    p  = preds.loc[(preds.index >= sd - pd.Timedelta(days=45)) &
                   (preds.index <= ed)].copy()
    for col, src in [("actual_high","btc_high"),("actual_low","btc_low"),
                     ("actual_close","btc_close")]:
        p[col] = df_raw[src].reindex(p.index).values
    comp  = p.dropna(subset=["actual_high","actual_low","actual_close"]).reset_index()
    comp  = comp[comp["target_date"] >= sd].reset_index(drop=True)
    dates = pd.DatetimeIndex(comp["target_date"])

    print(f"\nBacktest window : {dates[0].strftime('%b %d, %Y')} → "
          f"{dates[-1].strftime('%b %d, %Y')}  ({len(comp)} bars)  — ALL OOS")
    print(f"Training starts : {TRAIN_START.date()}  (model has NEVER seen this data)")

    print("\nComputing signals …")
    sigs = build_signals(comp)
    v_count = sigs["v_rev_bar"].sum()
    print(f"  U1 fires:        {sigs['u1'].sum():4d} bars  ({100*sigs['u1'].sum()/len(comp):.1f}%)")
    print(f"  D2 fires:        {sigs['d2'].sum():4d} bars  ({100*sigs['d2'].sum()/len(comp):.1f}%)")
    print(f"  D3 fires:        {sigs['d3'].sum():4d} bars  ({100*sigs['d3'].sum()/len(comp):.1f}%)")
    print(f"  v_rev_bar fires: {v_count:4d} bars  ({100*v_count/len(comp):.1f}%)")

    print_v_signal_quality(dates, sigs)

    print("\nRunning [A] TF2 Baseline …")
    res_a = run_backtest(dates, sigs, use_vgate=False)
    print("Running [B] TF2+V-Gate 3-bar …")
    res_b = run_backtest(dates, sigs, use_vgate=True)

    print_trade_log(res_a["trades"], res_b["trades"],
                    label=f"Pre-Training OOS  ({BACKTEST_START} → {BACKTEST_END})")

    m_a = summarise(res_a, "[A] TF2 Baseline")
    m_b = summarise(res_b, "[B] TF2+V-Gate")
    print_summary([m_a, m_b])

    # Buy & Hold reference
    bh_final = res_a["bh"].iloc[-1]
    print(f"\n  Buy & Hold reference: ${bh_final:,.0f}  "
          f"({(bh_final/INITIAL_CAP-1)*100:+.1f}%)")

    # Open position warning
    for tag, res in [("[A]", res_a), ("[B]", res_b)]:
        if res["open_pos"]:
            op = res["open_entry"]
            cur_pnl = (res["nav"].iloc[-1]/op["nav"]-1)*100
            print(f"  {tag} OPEN position at end: entered {pd.Timestamp(op['date']).strftime('%b %d %Y')}"
                  f"  @ ${op['price']:,.0f}  unrealised {cur_pnl:+.1f}%")
