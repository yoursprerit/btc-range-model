#!/usr/bin/env python3
"""
Do free Binance order-book / derivatives microstructure features add value?
===========================================================================
Offline research script (no app/model/strategy code is modified).

Data (scripts/pull_orderbook_features.py -> data/orderbook/hourly_ob.csv):
  bookDepth +/-1/2/5 % depth imbalance (2023-01+), open interest, funding, perp
  premium, taker flow, long/short ratios.  All rolled to the SAME 12:00-UTC bars
  as data/backtest (bar D = [D 12:00, D+1 12:00); features known at bar close).

Three studies
  S1  Signal quality     : rank-IC of each feature vs next-bar return / H / L / range
  S2  Filters on live    : overlay entry-block / exit rules on the deployed CT engine
                           and on the BTC>SMA40 rule (BTC, MSTR, MSTU sleeves);
                           split-half robustness
  S3  ML feature value   : walk-forward quantile-GBM next-bar High/Low model,
                           baseline vs +features; loss + end-to-end strategy

Run: python research_orderbook_eval.py [s1|s2|s3|all]
"""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "app"))

OB_CSV = ROOT / "data" / "orderbook" / "hourly_ob.csv"
OUT = ROOT / "data" / "orderbook"
ANCHOR = 12

DEPTH_F = ["ob_imb1", "ob_imb2", "ob_imb5", "ob_imb1_l6", "ob_imb5_l6", "ob_imb5_d1",
           "ob_depth1_z", "ob_depth5_z"]
DERIV_F = ["oi_chg1", "oi_chg7", "oi_z30", "fund", "fund_z30", "prem", "prem_z30",
           "basis", "tk_spot", "tk_perp", "tk_spot_3d", "tk_ratio", "ls_top", "ls_all",
           "ls_all_z30", "trades_z30"]


def z(s, n=30):
    return (s - s.rolling(n).mean()) / s.rolling(n).std()


# ─────────────────────────────────────────────────────────────────────────────
def build_bars() -> tuple[pd.DataFrame, pd.DataFrame]:
    """12:00-UTC bars: (spot OHLCV bars, microstructure features)."""
    h = pd.read_csv(OB_CSV, index_col=0, parse_dates=True)
    h["bucket"] = (h.index - pd.Timedelta(hours=ANCHOR)).normalize()
    n = h.groupby("bucket")["spot_close"].transform("count")
    h = h[n == 24]
    g = h.groupby("bucket")
    spot = pd.DataFrame({
        "open": g["spot_open"].first(), "high": g["spot_high"].max(),
        "low": g["spot_low"].min(), "close": g["spot_close"].last(),
        "volume": g["spot_vol"].sum()})

    f = pd.DataFrame(index=spot.index)
    last6 = h[h.index.hour.isin(range(6, 12))]          # last 6 h of each bar (18:00-11:59)
    l6 = last6.groupby("bucket")
    for k in (1, 2, 5):
        f[f"ob_imb{k}"] = g[f"imb{k}"].mean()
    f["ob_imb1_l6"] = l6["imb1"].mean()
    f["ob_imb5_l6"] = l6["imb5"].mean()
    f["ob_imb5_d1"] = f["ob_imb5"].diff()
    f["ob_depth1_z"] = z(np.log(g["depth1"].mean()))
    f["ob_depth5_z"] = z(np.log(g["depth5"].mean()))
    oi = g["sum_open_interest"].last()
    f["oi_chg1"] = np.log(oi).diff(1); f["oi_chg7"] = np.log(oi).diff(7)
    f["oi_z30"] = z(np.log(oi))
    f["fund"] = g["funding"].mean() * 1e4
    f["fund_z30"] = z(f["fund"])
    f["prem"] = g["prem"].mean() * 1e4
    f["prem_z30"] = z(f["prem"])
    f["basis"] = (g["perp_close"].last() / g["spot_close"].last() - 1) * 1e4
    f["tk_spot"] = (2 * g["spot_tbuy"].sum() - g["spot_vol"].sum()) / g["spot_vol"].sum()
    f["tk_perp"] = (2 * g["perp_tbuy"].sum() - g["perp_vol"].sum()) / g["perp_vol"].sum()
    f["tk_spot_3d"] = f["tk_spot"].rolling(3).mean()
    f["tk_ratio"] = np.log(g["sum_taker_long_short_vol_ratio"].mean().clip(lower=1e-3))
    f["ls_top"] = np.log(g["count_toptrader_long_short_ratio"].mean())
    f["ls_all"] = np.log(g["count_long_short_ratio"].mean())
    f["ls_all_z30"] = z(f["ls_all"])
    f["trades_z30"] = z(np.log(g["spot_trades"].sum()))
    return spot, f.replace([np.inf, -np.inf], np.nan)


# ═════════════════════════════════════════════════════════════════════════════
# S1  signal quality
# ═════════════════════════════════════════════════════════════════════════════
def s1():
    spot, f = build_bars()
    c = spot["close"]
    tgt = pd.DataFrame({
        "next_ret": np.log(c.shift(-1) / c),
        "next_hi": (spot["high"].shift(-1) - c) / c,
        "next_lo": (c - spot["low"].shift(-1)) / c,
    })
    tgt["next_range"] = tgt["next_hi"] + tgt["next_lo"]
    d = f.join(tgt).loc["2022-03-01":].dropna(subset=["next_ret"])
    rows = []
    for col in DEPTH_F + DERIV_F:
        x = d[col]
        r = {"feature": col, "n": int(x.notna().sum())}
        for t in tgt.columns:
            r[f"IC_{t}"] = x.corr(d[t], method="spearman")
        for yr in (2022, 2023, 2024, 2025, 2026):
            dy = d.loc[str(yr)]
            r[f"ret_{yr}"] = dy[col].corr(dy["next_ret"], method="spearman") if dy[col].notna().sum() > 60 else np.nan
        rows.append(r)
    R = pd.DataFrame(rows).set_index("feature")
    n_eff = 1000
    print(f"\nS1  rank-IC vs next 12UTC bar (2022-03 → {d.index[-1].date()}); |IC|>~{2/np.sqrt(n_eff):.3f} is ~2σ for ~1000 bars\n")
    pd.set_option("display.width", 220); pd.set_option("display.float_format", lambda v: f"{v:+.3f}")
    print(R.drop(columns="n").to_string())
    R.to_csv(OUT / "s1_ic.csv")


# ═════════════════════════════════════════════════════════════════════════════
# S2  filters / overlays on live engine
# ═════════════════════════════════════════════════════════════════════════════
def _metrics(nav: pd.Series) -> dict:
    dr = nav.pct_change().fillna(0)
    sh = dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else 0.0
    dd = (nav / nav.cummax() - 1).min()
    return dict(ret=(nav.iloc[-1] / nav.iloc[0] - 1) * 100, sharpe=sh, mdd=dd * 100)


def _engine_setup():
    import btc_ct_engine as E, backtest_trailing_stop as T
    rf = T.load_raw_features()
    preds = T.build_preds_offline(rf)
    return E, T, rf, preds


def _run_engine(E, comp, feat, variant, start="2024-03-05", preds_override=None):
    """Run BTC/MSTR/MSTU sleeves with an overlay `variant(feat_row_arrays)->(block_entry, force_exit)`."""
    dates = pd.DatetimeIndex(comp["target_date"])
    sigs = E.compute_sigs_pure(comp)
    px = E._load_prices(dates, comp)
    fa = feat.reindex(dates)
    block, force = variant(fa) if variant else (np.zeros(len(dates), bool), np.zeros(len(dates), bool))
    out = {}
    for k in ("BTC", "MSTR", "MSTU"):
        s = dict(sigs)
        base_entry = sigs["tf2_entry_pure"] if k == "BTC" else sigs["tf2_entry_ma"]
        s["tf2_entry"] = base_entry & ~block
        s["u1"] = sigs["u1"] & ~block                       # also gates re-entry override
        s["d3"] = sigs["d3"] | force                         # force-exit rides the D3 exit path
        bt = E._run_bt(dates, px[k], s, E.STOP_PCT[k], pd.Timestamp(start))
        out[k] = (bt["nav"], len(bt["trades"]))
    return out


def _sma_rule(E, comp, feat, variant, n=40, start="2024-03-05", cost=0.001):
    """BTC > SMA(n) long/flat on all three sleeves (repo's simple rule), next-bar fills, 10 bps/switch."""
    dates = pd.DatetimeIndex(comp["target_date"])
    px = E._load_prices(dates, comp)
    btc = pd.Series(px["BTC"], index=dates)
    on = (btc > btc.rolling(n).mean()).to_numpy()
    fa = feat.reindex(dates)
    block, force = variant(fa) if variant else (np.zeros(len(dates), bool), np.zeros(len(dates), bool))
    pos = np.zeros(len(dates)); cur = 0.0
    for i in range(len(dates)):
        if cur and (not on[i] or force[i]):
            cur = 0.0
        elif not cur and on[i] and not block[i] and not force[i]:
            cur = 1.0
        pos[i] = cur
    pos = pd.Series(pos, index=dates).shift(1).fillna(0)       # signal at close i -> position over i→i+1
    out = {}
    for k in ("BTC", "MSTR", "MSTU"):
        r = pd.Series(px[k], index=dates).pct_change().fillna(0)
        sw = pos.diff().abs().fillna(0)
        net = pos * r - sw * cost
        nav = (1 + net.loc[start:]).cumprod()
        out[k] = (nav, int(sw.loc[start:].sum()))
    return out


def make_variants(feat: pd.DataFrame):
    """Pre-registered, parameter-light overlays (one threshold each — no grid search)."""
    def blk(cond):  return lambda fa: (cond(fa).fillna(False).to_numpy(bool), np.zeros(len(fa), bool))
    def frc(cond):  return lambda fa: (np.zeros(len(fa), bool), cond(fa).fillna(False).to_numpy(bool))
    zi5 = (feat["ob_imb5"] - feat["ob_imb5"].rolling(60).mean()) / feat["ob_imb5"].rolling(60).std()
    zi1 = (feat["ob_imb1"] - feat["ob_imb1"].rolling(60).mean()) / feat["ob_imb1"].rolling(60).std()
    F = feat.assign(zi5=zi5, zi1=zi1)
    V = {
        "base": None,
        # --- entry blocks ---
        "blk imb5_z<-1  (asks heavy)": blk(lambda fa: F.reindex(fa.index)["zi5"] < -1),
        "blk depth5_z<-1 (thin book)": blk(lambda fa: fa["ob_depth5_z"] < -1),
        "blk fund_z>2   (crowded long)": blk(lambda fa: fa["fund_z30"] > 2),
        "blk prem_z>2   (perp froth)": blk(lambda fa: fa["prem_z30"] > 2),
        "blk oi_chg7>+15% (lev build)": blk(lambda fa: fa["oi_chg7"] > 0.15),
        "blk taker_spot_3d<0 (sellers)": blk(lambda fa: fa["tk_spot_3d"] < 0),
        # --- force exits (liquidation / book-collapse proxies) ---
        "exit oi_chg1<-4% & ret<0 (flush)": frc(lambda fa: (fa["oi_chg1"] < -0.04)),
        "exit imb5_z<-2 (book flips)": frc(lambda fa: F.reindex(fa.index)["zi5"] < -2),
        "exit depth5_z<-2 (liq. drain)": frc(lambda fa: fa["ob_depth5_z"] < -2),
    }
    return V


def s2():
    E, T, rf, preds = _engine_setup()
    spot, feat = build_bars()
    start_data = "2024-03-01"
    comp = T.prep(rf, preds, start_data, str(rf.index[-1].date()))
    V = make_variants(feat)
    halves = [("full", "2024-03-05", None), ("H1 24-03→25-09", "2024-03-05", "2025-09-30"),
              ("H2 25-10→now", "2025-10-01", None)]
    results = []
    for engine_name, runner in (("CT-engine (deployed ML signal)", _run_engine),
                                ("BTC>SMA40 simple rule", _sma_rule)):
        print(f"\n{'═'*100}\nS2  {engine_name}\n{'═'*100}")
        for sleeve in ("BTC", "MSTR", "MSTU"):
            print(f"\n  {sleeve}   (ret% / Sharpe / MaxDD% / trades) — Δ vs base in Sharpe")
            print(f"  {'variant':<36}" + "".join(f"{h[0]:>26}" for h in halves))
            base_sh = {}
            for vn, v in V.items():
                res = runner(E, comp, feat, v)[sleeve]
                nav, ntr = res
                cells = []
                for hn, a, b in halves:
                    nv = nav.loc[a:b] if b else nav.loc[a:]
                    m = _metrics(nv / nv.iloc[0])
                    if vn == "base":
                        base_sh[hn] = m["sharpe"]
                    d = m["sharpe"] - base_sh[hn]
                    cells.append(f"{m['ret']:+6.0f}/{m['sharpe']:4.2f}/{m['mdd']:4.0f} Δ{d:+.2f}")
                    results.append(dict(engine=engine_name, sleeve=sleeve, variant=vn, window=hn, trades=ntr, **m, d_sharpe=d))
                print(f"  {vn:<36}" + "".join(f"{c:>26}" for c in cells) + f"   n={ntr}")
    pd.DataFrame(results).to_csv(OUT / "s2_filters.csv", index=False)


# ═════════════════════════════════════════════════════════════════════════════
# S3  ML feature value
# ═════════════════════════════════════════════════════════════════════════════
def baseline_features(rf: pd.DataFrame) -> pd.DataFrame:
    """Faithful trimmed copy of backtest_trailing_stop.build_preds_offline feature block."""
    df = rf
    c, h, l_, v = df["btc_close"], df["btc_high"], df["btc_low"], df["btc_volume"]
    ret = np.log(c).diff()
    f = pd.DataFrame(index=df.index)
    for k in [1, 3, 5, 7, 14, 30]: f[f"ret_{k}"] = ret.rolling(k).sum()
    for k in [5, 10, 20, 30]:      f[f"vol_{k}"] = ret.rolling(k).std()
    pc = c.shift(1)
    tr = pd.concat([(h - l_), (h - pc).abs(), (l_ - pc).abs()], axis=1).max(axis=1)
    for k in [7, 14, 30]: f[f"atr_{k}"] = tr.rolling(k).mean() / c
    rg = (h - l_) / c
    f["range_today"] = rg; f["range_ma7"] = rg.rolling(7).mean()
    f["range_ma30"] = rg.rolling(30).mean(); f["range_std30"] = rg.rolling(30).std()
    g = c.diff().clip(lower=0).rolling(14).mean(); ls = (-c.diff().clip(upper=0)).rolling(14).mean()
    f["rsi_14"] = 100 - 100 / (1 + g / ls.replace(0, np.nan))
    e12 = c.ewm(span=12, adjust=False).mean(); e26 = c.ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    f["macd"] = macd / c; f["macd_sig"] = macd.ewm(span=9, adjust=False).mean() / c
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    ma20 = c.rolling(20).mean(); sd20 = c.rolling(20).std()
    f["bb_width"] = 4 * sd20 / ma20
    f["dist_hi_30"] = c / c.rolling(30).max() - 1; f["dist_lo_30"] = c / c.rolling(30).min() - 1
    f["dist_hi_90"] = c / c.rolling(90).max() - 1
    f["vol_chg_1"] = np.log(v).diff()
    f["vol_z_20"] = (np.log(v) - np.log(v).rolling(20).mean()) / np.log(v).rolling(20).std()
    f["vol_ma_ratio"] = v / v.rolling(20).mean()
    dow = df.index.dayofweek
    for i in range(6): f[f"dow_{i}"] = (dow == i).astype(float)
    for nm in ["spx", "ndx", "vix", "gold", "dxy", "tnx", "eth"]:
        col = f"{nm}_close"
        if col not in df.columns: continue
        lr = np.log(df[col]).diff()
        for k in [1, 5, 20]: f[f"{nm}_ret_{k}"] = lr.rolling(k).sum()
        f[f"{nm}_vol_20"] = lr.rolling(20).std()
    for nm in ["spx", "ndx", "gold", "dxy"]:
        if f"{nm}_close" in df.columns:
            f[f"btc_{nm}_corr_30"] = ret.rolling(30).corr(np.log(df[f"{nm}_close"]).diff())
    for col in [x for x in df.columns if x.startswith("oc_")]:
        sl = np.log(df[col].astype(float).replace(0, np.nan))
        f[f"{col}_d1"] = sl.diff(1); f[f"{col}_d7"] = sl.diff(7)
        f[f"{col}_z30"] = (sl - sl.rolling(30).mean()) / sl.rolling(30).std()
    y_hi = (h.shift(-1) - c) / c; y_lo = (c - l_.shift(-1)) / c
    for k in (3, 7):
        f[f"y_hi_ema{k}"] = y_hi.shift(1).ewm(span=k, adjust=False).mean()
        f[f"y_lo_ema{k}"] = y_lo.shift(1).ewm(span=k, adjust=False).mean()
    p3h = h.shift(1).rolling(3).max(); p3l = l_.shift(1).rolling(3).min()
    f["above_3d_high"] = (c > p3h).astype(float); f["below_3d_low"] = (c < p3l).astype(float)
    f["bo_strength_up"] = (c / p3h - 1).clip(lower=0); f["bo_strength_dn"] = (1 - c / p3l).clip(lower=0)
    ya, yb = y_hi.shift(1), y_lo.shift(1)
    f["y_hi_surprise"] = ya - ya.ewm(span=7, adjust=False).mean()
    f["y_lo_surprise"] = yb - yb.ewm(span=7, adjust=False).mean()
    nr = ret.clip(upper=0)
    f["dn_vol_5"] = nr.rolling(5).std(); f["dn_vol_20"] = nr.rolling(20).std()
    sma50 = c.rolling(50).mean()
    f["below_sma50"] = (c < sma50).astype(float)
    f["below_sma50_5d"] = f["below_sma50"].rolling(5).min().fillna(0)
    for cb in ["cb_premium", "cb_premium_ma3", "cb_premium_z7"]:
        f[cb] = df[cb].fillna(0.0) if cb in df.columns else 0.0
    return f.replace([np.inf, -np.inf], np.nan)


def pinball(y, q, a):
    e = y - q
    return np.maximum(a * e, (a - 1) * e)


def walk_forward(X, yhi, ylo, first_test, step=21, alpha=0.70, seed=0):
    """Expanding-window, refit every `step` bars; returns OOS (q_hi, q_lo) Series."""
    from sklearn.ensemble import HistGradientBoostingRegressor as H
    idx = X.index
    qh = pd.Series(np.nan, index=idx); ql = pd.Series(np.nan, index=idx)
    i0 = int(idx.searchsorted(pd.Timestamp(first_test)))
    for s in range(i0, len(idx), step):
        tr = np.arange(0, s - 1)                            # embargo 1 bar (target = next bar)
        te = np.arange(s, min(s + step, len(idx)))
        Xtr = X.iloc[tr]
        for y, out in ((yhi, qh), (ylo, ql)):
            ytr = y.iloc[tr]; ok = ytr.notna()
            m = H(loss="quantile", quantile=alpha, max_iter=250, learning_rate=0.04,
                  max_depth=3, min_samples_leaf=20, l2_regularization=1.0, random_state=seed)
            m.fit(Xtr[ok], ytr[ok])
            out.iloc[te] = m.predict(X.iloc[te])
    return qh, ql


def s3():
    E, T, rf, _ = _engine_setup()
    spot, feat = build_bars()
    base = baseline_features(rf)
    f_all = base.join(feat, how="left")
    c, h, l_ = rf["btc_close"], rf["btc_high"], rf["btc_low"]
    yhi = ((h.shift(-1) - c) / c).rename("yhi"); ylo = ((c - l_.shift(-1)) / c).rename("ylo")

    sets = {
        "A  baseline (deployed feature set)": list(base.columns),
        "B  + derivatives (OI/funding/prem/taker/LS)": list(base.columns) + DERIV_F,
        "C  + depth imbalance only": list(base.columns) + DEPTH_F,
        "D  + all order-book/deriv features": list(base.columns) + DEPTH_F + DERIV_F,
    }
    first_test = "2025-02-01"
    valid = f_all.dropna(subset=list(base.columns)).index
    valid = valid[(valid >= "2024-02-15")]
    out_preds = {}
    print(f"\nS3  walk-forward HistGB quantile(0.70) next-bar High/Low; test from {first_test}\n")
    print(f"  {'feature set':<46}{'pinball_H':>10}{'pinball_L':>10}{'MAPE_H%':>9}{'MAPE_L%':>9}{'cov_H':>7}{'cov_L':>7}")
    L = {}
    for name, cols in sets.items():
        X = f_all.loc[valid, cols]
        yh, yl = yhi.loc[valid], ylo.loc[valid]
        qh, ql = walk_forward(X, yh, yl, first_test)
        m = qh.notna() & yh.notna()
        ph, pl_ = pinball(yh[m], qh[m], .70), pinball(yl[m], ql[m], .70)
        L[name] = (ph, pl_)
        mape_h = (np.abs(yh[m] - qh[m]) / c.loc[valid][m] * c.loc[valid][m]).mean()  # |Δ| in return units
        print(f"  {name:<46}{ph.mean()*1e4:10.2f}{pl_.mean()*1e4:10.2f}"
              f"{np.abs(yh[m]-qh[m]).mean()*100:9.3f}{np.abs(yl[m]-ql[m]).mean()*100:9.3f}"
              f"{(yh[m]<=qh[m]).mean():7.2f}{(yl[m]<=ql[m]).mean():7.2f}")
        out_preds[name] = (qh, ql)
    # paired loss differences vs baseline (block bootstrap, 10-bar blocks)
    print("\n  paired Δpinball vs A (bps of price; negative = better), block-bootstrap 95% CI")
    rng = np.random.default_rng(0)
    b0 = list(L)[0]
    for name in list(L)[1:]:
        for side, k in (("High", 0), ("Low", 1)):
            d = (L[name][k] - L[b0][k]).to_numpy() * 1e4
            nb = len(d) // 10
            means = [np.mean(np.concatenate([d[s:s + 10] for s in rng.integers(0, len(d) - 10, nb)])) for _ in range(1000)]
            print(f"    {name:<46}{side:<5} mean {d.mean():+7.3f}   CI [{np.percentile(means,2.5):+.3f}, {np.percentile(means,97.5):+.3f}]")

    # ── end-to-end: plug each walk-forward prediction set into the live signal/exec engine ──
    print("\n  END-TO-END through the live CT signal engine (same gates, stops, fills):")
    print(f"  {'feature set':<46}" + "".join(f"{k+' ret/Sh/MDD':>24}" for k in ("BTC", "MSTR", "MSTU")))
    for name, (qh, ql) in out_preds.items():
        m = qh.notna()
        idx = qh.index[m]
        cc = c.loc[idx].values
        ph = cc * (1 + np.clip(qh[m].values, 0, None)); pl_ = cc * (1 - np.clip(ql[m].values, 0, None))
        nd = np.append(np.asarray(idx[1:], dtype="datetime64[ns]"), np.datetime64(idx[-1] + pd.Timedelta(days=1)))
        preds = pd.DataFrame({"close_asof": cc, "pred_high": ph, "pred_low": pl_},
                             index=pd.DatetimeIndex(nd, name="target_date"))
        comp = T.prep(rf, preds, first_test, str(rf.index[-1].date()))
        res = _run_engine(E, comp, feat, None, start=first_test)
        cells = []
        for k in ("BTC", "MSTR", "MSTU"):
            nav, ntr = res[k]
            mm = _metrics(nav / nav.iloc[0])
            cells.append(f"{mm['ret']:+5.0f}/{mm['sharpe']:4.2f}/{mm['mdd']:4.0f} n{ntr}")
        print(f"  {name:<46}" + "".join(f"{x:>24}" for x in cells))
    # reference: deployed pretrained model on the same window
    preds_dep = T.build_preds_offline(rf)
    comp = T.prep(rf, preds_dep, first_test, str(rf.index[-1].date()))
    res = _run_engine(E, comp, feat, None, start=first_test)
    cells = []
    for k in ("BTC", "MSTR", "MSTU"):
        nav, ntr = res[k]; mm = _metrics(nav / nav.iloc[0])
        cells.append(f"{mm['ret']:+5.0f}/{mm['sharpe']:4.2f}/{mm['mdd']:4.0f} n{ntr}")
    print(f"  {'(ref) deployed pretrained model':<46}" + "".join(f"{x:>24}" for x in cells))


# ═════════════════════════════════════════════════════════════════════════════
# S4  event study: do the overlay conditions predict forward BTC returns?
# ═════════════════════════════════════════════════════════════════════════════
def s4():
    spot, feat = build_bars()
    c = spot["close"]
    V = make_variants(feat)
    dates = feat.loc["2023-03-01":].index
    fa = feat.reindex(dates)
    rng = np.random.default_rng(1)
    print("\nS4  forward BTC log-return (%) after the condition fires vs all other days, 2023-03→now")
    print(f"  {'condition':<36}{'days':>5}" + "".join(f"{'fwd'+str(k)+'d (cond/rest)  CI(diff)':>34}" for k in (1, 3, 7)))
    for vn, v in V.items():
        if v is None: continue
        b, f_ = v(fa)
        cond = pd.Series(b | f_, index=dates)
        row = f"  {vn:<36}{int(cond.sum()):>5}"
        for k in (1, 3, 7):
            fwd = np.log(c.shift(-k) / c).reindex(dates) * 100
            m = fwd.notna()
            x, y = fwd[m & cond], fwd[m & ~cond]
            # bootstrap CI of difference in means (resample days independently; overlap ignored → CI is optimistic for k>1)
            diffs = [rng.choice(x, len(x)).mean() - rng.choice(y, len(y)).mean() for _ in range(1000)]
            row += f"{x.mean():+8.2f}/{y.mean():+6.2f}  [{np.percentile(diffs,2.5):+5.2f},{np.percentile(diffs,97.5):+5.2f}]".rjust(34)
        print(row)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("s1", "all"): s1()
    if which in ("s2", "all"): s2()
    if which in ("s3", "all"): s3()
    if which in ("s4", "all"): s4()
