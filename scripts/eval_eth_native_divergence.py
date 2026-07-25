"""ETH-native regime-divergence engine vs keeping ETH on the BTC parent signal.

**Question.**  The shipped ETH sleeve trades spot ETH (12:00-UTC bars, executed
live via ETHA) off the **BTC** CT regime-divergence signal (Standard-MA gate,
−8% stop — see ETH_BMNR_STRATEGY_EVAL.md).  Would training a **new** regime-
divergence engine on **Ethereum's own prices** (an ETH H/L forecast model +
ETH-side U1/D2/D3 divergence signals, thresholds tuned the repo's own way)
beat that — and beat the simple BTC>SMA30 parent-trend rule that the earlier
memo found strongest on ETH?

**Evaluation only** — nothing here changes any live config.

Method (mirrors the repo's own conventions):

  * ETH H/L model = the CT recipe ported to ETH: same feature engineering as
    ``backtest_trailing_stop.build_preds_offline`` computed on ETH OHLCV, plus
    optional BTC-parent and macro/Coinbase-premium blocks; same 3-model
    ensemble (Huber + linear q0.70 + GBM q0.70).  A ridge H/L model (the
    ticker-app template, ``backtest_ticker.build_predictions``) is also swept.
  * ETH OHLCV comes from Binance on the SAME 12:00-UTC bar anchor as the BTC
    signal (``pull_backtest_data.fetch_12utc``), extended back to 2019 via the
    ``data-api.binance.vision`` mirror so the ETH model can train on a history
    comparable to the BTC model's (the committed CSV starts 2023-11; the BTC
    CT artifact trained on 2019-03 → 2026-02).  ``api.binance.us`` only serves
    2019-09 → and is deliberately NOT used for the long fetch.
  * Signals + backtest loop are byte-faithful ports of
    ``app/btc_ct_engine.compute_sigs_pure`` / ``_run_bt`` (same U1/D1/D2/D3
    construction, MA30 regime, Pure-Regime & Standard-MA gates, V-reversal,
    post-stop re-entry override, same-bar fill — legitimate here because ETH
    bars close exactly at the signal moment, as for BTC).
  * Divergence thresholds (U1 × D2 × V × gate × regime source × stop) are
    tuned on **2024-03-05 → 2025-08-01 only** (the repo's Aug-2025 convention,
    ``scripts/eval_regime_aug2025.py``).  Selection rule = the repo's own:
    maximise Sharpe s.t. MDD ≤ B&H, tie-broken by return.  The model itself is
    fit strictly on pre-Aug-2025 rows (1-bar embargo).  **Aug-2025 → now is a
    true holdout** for both the model and the thresholds.
  * A "regime source" axis lets the divergence stay ETH-native while the MA30
    regime gate reads BTC (the parent-trend hybrid), since the earlier memo
    showed parent trend beats own trend on ETH.
  * Walk-forward robustness: the champion combo is refit on an expanding
    window every 63 bars across the sleeve window, so every prediction is
    out-of-sample; the (fixed) chosen thresholds are then applied.
  * Baselines over the same windows: ETH B&H; the shipped CT-on-BTC sleeve
    (Standard-MA gate, no stop and −8%); simple BTC>SMA30 → ETH; ETH own
    SMA30 (both next-bar, 10 bps per switch — the earlier memo's cost model).

Run:  python scripts/eval_eth_native_divergence.py
      (network needed for the 2019→ Binance history; falls back to the
      committed 2023-11→ CSVs when offline.  Long fetch is cached in tmp/.)
"""
from __future__ import annotations

import sys
import time
import argparse
import itertools
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
for _p in (str(_REPO), str(_REPO / "app"), str(_REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import btc_ct_engine as E                 # noqa: E402  (loads the CT model; shims sklearn)
import backtest_trailing_stop as T        # noqa: E402
import pull_backtest_data as PBD          # noqa: E402

TD = 252
LONG_START = "2019-01-01"                 # ≈ the BTC artifact's train_start (2019-03-31)
SLEEVE_START = "2024-03-05"               # the shipped sleeve's window start
TRAIN_CUTOFF = "2025-08-01"               # repo convention: tune through Aug-2025 only
CACHE = _REPO / "tmp"                     # gitignored (/tmp/ in .gitignore)


# ── small helpers (same math as scripts/eval_eth_bmnr.py) ──────────────────
def metrics(eq: pd.Series) -> dict:
    eq = eq.dropna()
    if len(eq) < 2:
        return dict(total=np.nan, mdd=np.nan, sharpe=np.nan)
    r = eq.pct_change().dropna()
    return dict(total=eq.iloc[-1] / eq.iloc[0] - 1,
                mdd=float((eq / eq.cummax() - 1).min()),
                sharpe=float(r.mean() / (r.std() + 1e-12) * np.sqrt(TD)))


def simple_bt(asset: pd.Series, sig: pd.Series, cost=0.001) -> dict:
    """Long/flat next-bar rule with `cost` per switch (prior-eval methodology)."""
    s = (sig.astype(float).reindex(asset.index.union(sig.index)).ffill()
         .reindex(asset.index).fillna(0.0))
    pos = s.shift(1).fillna(0.0)
    ret = asset.pct_change().fillna(0.0)
    sw = pos.diff().abs().fillna(0.0)
    strat = pos * ret - sw * cost
    m = metrics((1 + strat).cumprod())
    m.update(trades=int((sw > 0).sum()), strat=strat)
    return m


def rule_sma(c: pd.Series, n: int) -> pd.Series:
    return c > c.rolling(n).mean()


def fmt(m, extra=""):
    return (f"tot {m['total']*100:+8.1f}%  MDD {m['mdd']*100:6.1f}%  "
            f"Sh {m['sharpe']:5.2f}{extra}")


# ── data ───────────────────────────────────────────────────────────────────
def fetch_long(symbol: str) -> pd.DataFrame | None:
    """12:00-UTC OHLCV bars from LONG_START, via the full-history Binance
    mirror.  Cached under tmp/ (gitignored); returns None when offline."""
    CACHE.mkdir(exist_ok=True)
    fp = CACHE / f"eval_eth_native_{symbol.lower()}_12utc.csv"
    if fp.exists():
        df = pd.read_csv(fp, index_col=0, parse_dates=True)
        if len(df) > 500:
            return df
    hosts0 = PBD._BINANCE_HOSTS
    try:      # vision mirror first — binance.us silently starts at 2019-09
        PBD._BINANCE_HOSTS = ("https://data-api.binance.vision",)
        df = PBD.fetch_12utc(LONG_START, symbol=symbol)
        df.to_csv(fp)
        return df
    except Exception as exc:
        print(f"  [warn] long Binance fetch failed for {symbol}: {exc}")
        return None
    finally:
        PBD._BINANCE_HOSTS = hosts0


def load_committed() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(eth OHLCV, raw features) from the committed CSVs (2023-11 →)."""
    eth = pd.read_csv(T.DATA / "eth_usd_daily.csv", index_col=0, parse_dates=True)
    rf = T.load_raw_features()
    return eth, rf


# ── ETH feature engineering — the CT recipe ported to ETH ──────────────────
def build_features(eth: pd.DataFrame, btc_close: pd.Series | None,
                   rf: pd.DataFrame | None, feature_set: str):
    """Feature matrix on ETH bars (row t = known at bar t's close) + targets
    for bar t+1.  Mirrors backtest_trailing_stop.build_preds_offline's blocks
    computed on ETH OHLCV; 'EB' adds the BTC-parent block; 'EBM' adds the
    macro / Coinbase-premium / on-chain block from the committed feature CSV."""
    c, h, l_, v = eth["close"], eth["high"], eth["low"], eth["volume"]
    ret = np.log(c).diff()
    f = pd.DataFrame(index=eth.index)
    for k in [1, 3, 5, 7, 14, 30]: f[f"ret_{k}"] = ret.rolling(k).sum()
    for k in [5, 10, 20, 30]:      f[f"vol_{k}"] = ret.rolling(k).std()
    prev_c = c.shift(1)
    tr = pd.concat([(h - l_), (h - prev_c).abs(), (l_ - prev_c).abs()], axis=1).max(axis=1)
    for k in [7, 14, 30]: f[f"atr_{k}"] = tr.rolling(k).mean() / c
    f["range_today"] = (h - l_) / c
    f["range_ma7"] = ((h - l_) / c).rolling(7).mean()
    f["range_ma30"] = ((h - l_) / c).rolling(30).mean()
    f["range_std30"] = ((h - l_) / c).rolling(30).std()
    g = c.diff().clip(lower=0).rolling(14).mean()
    ls = (-c.diff().clip(upper=0)).rolling(14).mean()
    f["rsi_14"] = 100 - 100 / (1 + g / ls.replace(0, np.nan))
    e12 = c.ewm(span=12, adjust=False).mean(); e26 = c.ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    f["macd"] = macd / c
    f["macd_sig"] = macd.ewm(span=9, adjust=False).mean() / c
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    ma20 = c.rolling(20).mean(); sd20 = c.rolling(20).std()
    f["bb_width"] = (4 * sd20) / ma20
    f["dist_hi_30"] = c / c.rolling(30).max() - 1
    f["dist_lo_30"] = c / c.rolling(30).min() - 1
    f["dist_hi_90"] = c / c.rolling(90).max() - 1
    f["vol_chg_1"] = np.log(v).diff()
    f["vol_z_20"] = (np.log(v) - np.log(v).rolling(20).mean()) / np.log(v).rolling(20).std()
    f["vol_ma_ratio"] = v / v.rolling(20).mean()
    dow = eth.index.dayofweek
    for i in range(6): f[f"dow_{i}"] = (dow == i).astype(float)
    nh, nl_ = h.shift(-1), l_.shift(-1)
    y_hi = (nh - c) / c
    y_lo = (c - nl_) / c
    f["y_hi_ema3"] = y_hi.shift(1).ewm(span=3, adjust=False).mean()
    f["y_lo_ema3"] = y_lo.shift(1).ewm(span=3, adjust=False).mean()
    f["y_hi_ema7"] = y_hi.shift(1).ewm(span=7, adjust=False).mean()
    f["y_lo_ema7"] = y_lo.shift(1).ewm(span=7, adjust=False).mean()
    p3h = h.shift(1).rolling(3).max(); p3l = l_.shift(1).rolling(3).min()
    f["above_3d_high"] = (c > p3h).astype(float)
    f["below_3d_low"] = (c < p3l).astype(float)
    f["bo_strength_up"] = (c / p3h - 1).clip(lower=0)
    f["bo_strength_dn"] = (1 - c / p3l).clip(lower=0)
    ya = y_hi.shift(1); yb = y_lo.shift(1)
    f["y_hi_surprise"] = ya - ya.ewm(span=7, adjust=False).mean()
    f["y_lo_surprise"] = yb - yb.ewm(span=7, adjust=False).mean()
    neg_ret = ret.clip(upper=0)
    f["dn_vol_5"] = neg_ret.rolling(5).std(); f["dn_vol_20"] = neg_ret.rolling(20).std()
    sma50 = c.rolling(50).mean()
    f["below_sma50"] = (c < sma50).astype(float)
    f["below_sma50_5d"] = f["below_sma50"].rolling(5).min().fillna(0)

    if feature_set in ("EB", "EBM") and btc_close is not None:
        b = btc_close.reindex(eth.index)
        br = np.log(b).diff()
        for k in [1, 5, 20]: f[f"btc_ret_{k}"] = br.rolling(k).sum()
        f["btc_vol_20"] = br.rolling(20).std()
        f["eth_btc_corr_30"] = ret.rolling(30).corr(br)
        f["btc_above_sma30"] = (b > b.rolling(30).mean()).astype(float)
        f["btc_dist_sma50"] = b / b.rolling(50).mean() - 1
        f["ethbtc_ratio_z30"] = ((c / b) - (c / b).rolling(30).mean()) / (c / b).rolling(30).std()

    if feature_set == "EBM" and rf is not None:
        for nm in ["spx", "ndx", "vix", "gold", "dxy", "tnx"]:
            s = rf[f"{nm}_close"].reindex(eth.index).ffill()
            lr = np.log(s).diff()
            for k in [1, 5, 20]: f[f"{nm}_ret_{k}"] = lr.rolling(k).sum()
            f[f"{nm}_vol_20"] = lr.rolling(20).std()
        for cbcol in ["cb_premium", "cb_premium_ma3", "cb_premium_z7"]:
            f[cbcol] = rf[cbcol].reindex(eth.index).fillna(0.0) if cbcol in rf else 0.0
        for col in [x for x in rf.columns if x.startswith("oc_")]:
            sl = np.log(rf[col].astype(float).replace(0, np.nan)).reindex(eth.index)
            f[f"{col}_d7"] = sl.diff(7)
            f[f"{col}_z30"] = (sl - sl.rolling(30).mean()) / sl.rolling(30).std()

    f = f.replace([np.inf, -np.inf], np.nan)
    keep = pd.concat([f, y_hi.rename("y_hi"), y_lo.rename("y_lo")], axis=1).dropna()
    feat_cols = [cc for cc in f.columns]
    return keep[feat_cols], keep["y_hi"], keep["y_lo"], c.reindex(keep.index)


# ── models ─────────────────────────────────────────────────────────────────
def _make_models(kind: str, quick: bool):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeCV, HuberRegressor, QuantileRegressor
    from sklearn.ensemble import GradientBoostingRegressor

    def _pipe(m):
        return Pipeline([("sc", StandardScaler()), ("m", m)])

    if kind == "ridge":
        return [("ridge", lambda: _pipe(RidgeCV(alphas=np.logspace(-3, 3, 13))))]
    n_est = 400 if quick else 1500
    return [
        ("huber", lambda: _pipe(HuberRegressor(alpha=1e-3, epsilon=1.35, max_iter=500))),
        ("quant_lin", lambda: _pipe(QuantileRegressor(alpha=1e-3, quantile=0.7,
                                                      solver="highs"))),
        ("gbm_quant", lambda: _pipe(GradientBoostingRegressor(
            loss="quantile", alpha=0.7, n_estimators=n_est, learning_rate=0.01,
            max_depth=3, subsample=0.8, min_samples_leaf=1, random_state=42))),
    ]


def fit_predict(F, y_hi, y_lo, close, model_kind, train_end, quick,
                predict_from=None) -> pd.DataFrame:
    """Fit on rows ≤ train_end (1-bar embargo, CT convention) and predict every
    bar (from `predict_from` when given).  Returns the comp-style pred frame
    indexed by target_date (bar t features → bar t+1 H/L), mirroring
    build_preds_offline's clipping and frame shape."""
    cut = pd.Timestamp(train_end) - pd.Timedelta(days=1)
    tr = F.index <= cut
    Ftr = F.loc[tr]
    Fpr = F if predict_from is None else F.loc[F.index >= pd.Timestamp(predict_from)]
    yhi = np.zeros(len(Fpr)); ylo = np.zeros(len(Fpr))
    specs = _make_models(model_kind, quick)
    for _, mk in specs:
        mh = mk().fit(Ftr, y_hi.loc[tr]); ml = mk().fit(Ftr, y_lo.loc[tr])
        yhi += mh.predict(Fpr) / len(specs)
        ylo += ml.predict(Fpr) / len(specs)
    c_vals = close.reindex(Fpr.index).values
    ph = c_vals * (1 + np.clip(yhi, 0, None))
    pl = c_vals * (1 - np.clip(ylo, 0, None))
    idx = np.asarray(Fpr.index, dtype="datetime64[ns]")
    nd = np.empty(len(Fpr), dtype="datetime64[ns]")
    nd[:-1] = idx[1:]; nd[-1] = idx[-1] + np.timedelta64(1, "D")
    out = pd.DataFrame({"close_asof": c_vals, "pred_high": ph, "pred_low": pl},
                       index=pd.DatetimeIndex(nd, name="target_date"))
    return out[~out.index.duplicated(keep="last")]


def make_comp(preds: pd.DataFrame, eth: pd.DataFrame, start, end=None) -> pd.DataFrame:
    """Attach realized ETH H/L/close (T.prep's shape, ETH substituted for BTC)."""
    sd = pd.Timestamp(start) - pd.Timedelta(days=60)
    p = preds.loc[preds.index >= sd].copy()
    if end is not None:
        p = p.loc[p.index <= pd.Timestamp(end)]
    p["actual_high"] = eth["high"].reindex(p.index).values
    p["actual_low"] = eth["low"].reindex(p.index).values
    p["actual_close"] = eth["close"].reindex(p.index).values
    return (p.dropna(subset=["actual_high", "actual_low", "actual_close"])
            .reset_index())


# ── signals — btc_ct_engine.compute_sigs_pure, parametrized ────────────────
def base_arrays(comp: pd.DataFrame, btc_close_asof: np.ndarray | None) -> dict:
    """Everything threshold-independent: raw errors, 3-bar aggregates, D3,
    down-score, and the MA30 regime for BOTH regime sources (ETH / BTC)."""
    N = len(comp)
    c = comp["close_asof"].values.astype(float)
    ph = comp["pred_high"].values.astype(float)
    pl = comp["pred_low"].values.astype(float)
    ah = comp["actual_high"].values.astype(float)
    al = comp["actual_low"].values.astype(float)
    err_hi = (ah - ph) / c * 100
    err_lo = (pl - al) / c * 100
    hi = (ah > ph).astype(int)
    lo = (al < pl).astype(int)
    ehma3 = np.zeros(N); elma3 = np.zeros(N)
    hb3 = np.zeros(N, int); lb3 = np.zeros(N, int)
    for i in range(N):
        s = max(0, i - 2)
        ehma3[i] = err_hi[s:i + 1].mean(); elma3[i] = err_lo[s:i + 1].mean()
        hb3[i] = hi[s:i + 1].sum(); lb3[i] = lo[s:i + 1].sum()
    d3 = np.zeros(N, bool)
    for i in range(1, N):
        cc = 0
        for k in range(i - 1, -1, -1):
            if hi[k]: cc += 1
            else: break
        if cc >= 3 and lo[i]:
            d3[i] = True
    roll = np.array([err_hi[max(0, i - 29):i + 1].mean() for i in range(N)])
    dn = np.zeros(N)
    for i in range(N):
        nrm = max(abs(roll[i]), 0.01)
        dn[i] = ((-ehma3[i] / nrm) * .30 + (lb3[i] / 3.) * .30
                 + (elma3[i] / max(abs(elma3[i]), .10)) * .20 + float(lo[i]) * .20)

    def _regime(cl):
        ma30 = np.array([cl[max(0, i - min(30, i + 1) + 1):i + 1].mean() for i in range(N)])
        above = cl > ma30
        slope = np.zeros(N, bool)
        for i in range(5, N):
            slope[i] = ma30[i] > ma30[i - 5]
        return above, above & slope

    reg = {"eth": _regime(c)}
    if btc_close_asof is not None:
        reg["btc"] = _regime(btc_close_asof.astype(float))
    return dict(err_hi=err_hi, err_lo=err_lo, hi=hi, lo=lo, ehma3=ehma3,
                elma3=elma3, hb3=hb3, lb3=lb3, d3=d3, dn=dn, reg=reg, N=N)


def make_sigs(B: dict, U1: float, D2: float, D1: float, V: float,
              regime: str) -> dict:
    """The threshold-dependent layer of compute_sigs_pure (identical logic)."""
    N = B["N"]
    u1 = (B["ehma3"] > U1) & (B["hb3"] >= 2)
    d1 = (B["lb3"] >= 2) & (B["elma3"] > D1)
    d2 = B["ehma3"] < D2
    above, bull = B["reg"][regime]
    clean = np.array([not bool((d1[max(0, i - 7):i] | d2[max(0, i - 7):i]).any())
                      for i in range(N)])
    vbar = (B["dn"] > 0.8) & (B["err_lo"] > V)
    v = np.array([bool(vbar[max(0, i - (E.V_RECENT_WIN - 1)):i + 1].any())
                  for i in range(N)])
    tf_pure = u1 & (bull | (clean & ~above) | v)
    tf_ma = u1 & ((above ^ clean) | v)
    return dict(d1=d1, d2=d2, d3=B["d3"], u1=u1, bull_regime=bull,
                tf2_entry_pure=tf_pure, tf2_entry_ma=tf_ma,
                above_ma30=above, clean_7d=clean, v_recent=v,
                err_hi=B["err_hi"], err_lo=B["err_lo"])


def run_div(dates, px, sigs, gate, stop, start, end=None) -> dict:
    """One divergence backtest through the app-faithful loop (E._run_bt)."""
    s = dict(sigs)
    s["tf2_entry"] = sigs["tf2_entry_ma"] if gate == "ma" else sigs["tf2_entry_pure"]
    if end is not None:
        i1 = int(pd.DatetimeIndex(dates).searchsorted(pd.Timestamp(end), side="right"))
        dates = dates[:i1]; px = px[:i1]
        s = {k: (v[:i1] if isinstance(v, np.ndarray) else v) for k, v in s.items()}
    bt = E._run_bt(dates, px, s, stop, pd.Timestamp(start))
    nav = bt["nav"][bt["nav"].index >= pd.Timestamp(start)]
    m = metrics(nav)
    m.update(trades=len(bt["trades"]), nav=nav,
             wr=(100 * np.mean([t["ret"] > 0 for t in bt["trades"]])
                 if bt["trades"] else 0.0))
    return m


# ── threshold sweep (train window only) ────────────────────────────────────
def sweep(dates, px, B, s3_hi, s3_lo, s_lo, has_btc_regime, start, cutoff):
    """Grid over U1 × D2 × V × gate × regime × stop on the TRAIN window only.
    Grids are scaled to the signal's own error std (backtest_ticker's
    divergence_scale convention — the BTC engine's ±1.3 constants belong to
    the BTC error scale).  Returns (best spec, all candidates)."""
    bh = metrics(pd.Series(px, index=dates).loc[start:cutoff])
    u1_grid = sorted({round(k * s3_hi, 2) for k in (0.4, 0.6, 0.8, 1.0, 1.3)})
    d2_grid = sorted({round(-k * s3_hi, 2) for k in (0.4, 0.6, 0.8, 1.0, 1.3)})
    v_grid = sorted({round(k * s_lo, 1) for k in (1.0, 1.5, 2.0)})
    d1_val = round(0.35 * s3_lo, 2)
    regimes = ["eth", "btc"] if has_btc_regime else ["eth"]
    cands = []
    for U1, D2, V, reg in itertools.product(u1_grid, d2_grid, v_grid, regimes):
        sigs = make_sigs(B, U1, D2, d1_val, V, reg)
        for gate, stop in itertools.product(("ma", "pure"), (None, .05, .08, .12)):
            m = run_div(dates, px, sigs, gate, stop, start, cutoff)
            ok = m["mdd"] >= bh["mdd"] - 1e-9
            spec = dict(U1=U1, D2=D2, V=V, D1=d1_val, regime=reg, gate=gate,
                        stop=stop, train=m, score=(ok, m["sharpe"], m["total"]))
            cands.append(spec)
    best = max(cands, key=lambda x: x["score"])
    return best, cands, bh


def spec_tag(b):
    stop = "none" if b["stop"] is None else "-%.0f%%" % (b["stop"] * 100)
    return (f"U1={b['U1']} D2={b['D2']} V={b['V']} gate={b['gate']} "
            f"regime={b['regime'].upper()} stop={stop}")


# ── main ───────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller GBM (400 trees)")
    ap.add_argument("--skip-walkforward", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    print("Loading committed data + fetching 2019→ Binance history …", flush=True)
    eth_c, rf = load_committed()
    eth_long = fetch_long("ETHUSDT")
    btc_long = fetch_long("BTCUSDT")
    if eth_long is not None:
        ov = eth_long["close"].reindex(eth_c.index).dropna()
        drift = (ov / eth_c["close"].reindex(ov.index) - 1).abs().max()
        print(f"  long ETH {eth_long.index[0].date()} → {eth_long.index[-1].date()} "
              f"({len(eth_long)} bars); overlap vs committed CSV: max |Δ| {drift*100:.3f}%")
        eth = eth_long
    else:
        print("  [offline] using committed 2023-11→ ETH bars only")
        eth = eth_c
    btc_close = (btc_long["close"] if btc_long is not None
                 else rf["btc_close"])

    # ═══ Part 0 — BTC-parent baselines (the incumbents) ════════════════════
    print("\n" + "═" * 96)
    print("PART 0 — incumbents on the shipped window "
          f"({SLEEVE_START} → now) and the Aug-2025 holdout")
    print("═" * 96, flush=True)
    rff = T.load_raw_features()
    preds_btc = T.build_preds_offline(rff)
    comp_btc = T.prep(rff, preds_btc, "2024-01-01", str(rff.index[-1].date()))
    dates_b = pd.DatetimeIndex(comp_btc["target_date"])
    sigs_btc = E.compute_sigs_pure(comp_btc)
    eth_12 = eth_c["close"].astype(float)
    eth_on_b = eth_12.reindex(dates_b).ffill().to_numpy(float)
    btc_ser = pd.Series(comp_btc["actual_close"].to_numpy(float), index=dates_b)

    incumbents = {}
    for lbl, gate, stop in (("CT-on-BTC MA (no stop)", "ma", None),
                            ("CT-on-BTC MA −8% (shipped)", "ma", 0.08)):
        s = dict(sigs_btc); s["tf2_entry"] = sigs_btc["tf2_entry_ma"]
        bt = E._run_bt(dates_b, eth_on_b, s, stop, pd.Timestamp(SLEEVE_START))
        incumbents[lbl] = bt["nav"]
    eth_sleeve = eth_12.loc[SLEEVE_START:]

    rows = [("ETH Buy & hold", metrics(eth_sleeve), metrics(eth_12.loc[TRAIN_CUTOFF:]))]
    for lbl, nav in incumbents.items():
        rows.append((lbl, metrics(nav[nav.index >= SLEEVE_START]),
                     metrics(nav[nav.index >= TRAIN_CUTOFF])))
    for lbl, src in (("BTC>SMA30 → ETH (10bps)", btc_ser), ("ETH own SMA30 (10bps)", eth_12)):
        mfull = simple_bt(eth_12.loc[SLEEVE_START:], rule_sma(src, 30))
        mhold = simple_bt(eth_12.loc[TRAIN_CUTOFF:], rule_sma(src, 30))
        rows.append((lbl, mfull, mhold))
    print(f"  {'':34s} {'— full window —':>44s}   {'— holdout Aug-25→ —':>40s}")
    for lbl, mf, mh in rows:
        print(f"  {lbl:34s} {fmt(mf)}   {fmt(mh)}")
    incumbent_full = rows[2][1]; incumbent_hold = rows[2][2]     # shipped −8% row

    # ═══ Part 1 — ETH-native engines: fit, sweep (≤ Aug-2025), holdout ═════
    print("\n" + "═" * 96)
    print(f"PART 1 — ETH-native divergence engines (model fit ≤ {TRAIN_CUTOFF}, "
          f"thresholds tuned {SLEEVE_START} → {TRAIN_CUTOFF}, holdout after)")
    print("═" * 96, flush=True)

    combos = []
    for fs in ("E", "EB", "EBM"):
        for mk in ("ridge", "ctens"):
            combos.append((fs, mk))

    results = {}
    champion = None
    for fs, mk in combos:
        t1 = time.time()
        F, y_hi, y_lo, close = build_features(eth, btc_close, rf, fs)
        preds = fit_predict(F, y_hi, y_lo, close, mk, TRAIN_CUTOFF, args.quick)
        comp = make_comp(preds, eth, "2024-01-01")
        dates = pd.DatetimeIndex(comp["target_date"])
        px = comp["actual_close"].to_numpy(float)
        btc_asof = btc_close.reindex(
            pd.DatetimeIndex(comp["target_date"]) - pd.Timedelta(days=1)
        ).ffill().to_numpy(float)
        B = base_arrays(comp, btc_asof)

        tr_mask = (dates >= pd.Timestamp(SLEEVE_START)) & (dates < pd.Timestamp(TRAIN_CUTOFF))
        s3_hi = float(np.std(B["ehma3"][tr_mask]))
        s3_lo = float(np.std(B["elma3"][tr_mask]))
        s_lo = float(np.std(B["err_lo"][tr_mask]))

        best, cands, bh_tr = sweep(dates, px, B, s3_hi, s3_lo, s_lo,
                                   True, SLEEVE_START, TRAIN_CUTOFF)
        sigs = make_sigs(B, best["U1"], best["D2"], best["D1"], best["V"], best["regime"])
        m_full = run_div(dates, px, sigs, best["gate"], best["stop"], SLEEVE_START)
        m_hold = run_div(dates, px, sigs, best["gate"], best["stop"], TRAIN_CUTOFF)

        # honest forecast quality on the holdout (model never saw these rows)
        hm = dates >= pd.Timestamp(TRAIN_CUTOFF)
        mape_h = float(np.mean(np.abs(comp["pred_high"] - comp["actual_high"])[hm]
                               / comp["actual_high"][hm]) * 100)
        mape_l = float(np.mean(np.abs(comp["pred_low"] - comp["actual_low"])[hm]
                               / comp["actual_low"][hm]) * 100)

        key = f"{fs}/{mk}"
        results[key] = dict(best=best, m_full=m_full, m_hold=m_hold,
                            mape=(mape_h, mape_l), n_train=int((F.index <= pd.Timestamp(
                                TRAIN_CUTOFF) - pd.Timedelta(days=1)).sum()),
                            comp=comp, B=B, sigs=sigs, dates=dates, px=px)
        print(f"\n  [{key}]  n_train={results[key]['n_train']}  "
              f"holdout MAPE H/L {mape_h:.2f}%/{mape_l:.2f}%  "
              f"({time.time()-t1:.0f}s)")
        print(f"    chosen: {spec_tag(best)}")
        print(f"    train  {SLEEVE_START}→{TRAIN_CUTOFF}: {fmt(best['train'])}  "
              f"tr {best['train']['trades']}  (B&H {fmt(bh_tr)})")
        print(f"    FULL   window: {fmt(m_full)}  tr {m_full['trades']}")
        print(f"    HOLDOUT      : {fmt(m_hold)}  tr {m_hold['trades']}", flush=True)
        if champion is None or best["score"] > results[champion]["best"]["score"]:
            champion = key

    ch = results[champion]
    print(f"\n  ── champion by the train-window rule: [{champion}] — {spec_tag(ch['best'])}")

    # BTC-thresholds ported verbatim (U1=+1.3 / D2=−1.3, shipped gate/stop) —
    # what "just point the BTC engine's constants at ETH errors" would do.
    sig_port = make_sigs(ch["B"], E.U1_ERRHI_MIN, E.D2_ERRHI_MAX, 0.5, 3.0, "eth")
    m_port_f = run_div(ch["dates"], ch["px"], sig_port, "ma", 0.08, SLEEVE_START)
    m_port_h = run_div(ch["dates"], ch["px"], sig_port, "ma", 0.08, TRAIN_CUTOFF)
    print(f"  (BTC constants ported verbatim on [{champion}] errors: "
          f"full {fmt(m_port_f)} · holdout {fmt(m_port_h)})")

    # ═══ Part 2 — walk-forward robustness for the champion ═════════════════
    if not args.skip_walkforward:
        print("\n" + "═" * 96)
        print("PART 2 — walk-forward (champion combo; expanding refit every 63 bars; "
              "same fixed thresholds)")
        print("═" * 96, flush=True)
        fs, mk = champion.split("/")
        F, y_hi, y_lo, close = build_features(eth, btc_close, rf, fs)
        pieces = []
        anchors = [d for i, d in enumerate(F.index)
                   if d >= pd.Timestamp("2024-02-01") and i % 63 == 0]
        if not anchors or anchors[0] > pd.Timestamp("2024-03-01"):
            first = F.index[F.index >= pd.Timestamp("2024-02-01")][0]
            anchors = [first] + anchors
        for j, a in enumerate(anchors):
            nxt = anchors[j + 1] if j + 1 < len(anchors) else None
            p = fit_predict(F, y_hi, y_lo, close, mk, str(a.date()), args.quick,
                            predict_from=str(a.date()))
            if nxt is not None:
                p = p.loc[p.index <= nxt]
            pieces.append(p)
        preds_wf = pd.concat(pieces)
        preds_wf = preds_wf[~preds_wf.index.duplicated(keep="first")].sort_index()
        comp_wf = make_comp(preds_wf, eth, "2024-01-01")
        dates_wf = pd.DatetimeIndex(comp_wf["target_date"])
        px_wf = comp_wf["actual_close"].to_numpy(float)
        btc_asof_wf = btc_close.reindex(dates_wf - pd.Timedelta(days=1)).ffill().to_numpy(float)
        B_wf = base_arrays(comp_wf, btc_asof_wf)
        b = ch["best"]
        sigs_wf = make_sigs(B_wf, b["U1"], b["D2"], b["D1"], b["V"], b["regime"])
        m_wf_full = run_div(dates_wf, px_wf, sigs_wf, b["gate"], b["stop"], SLEEVE_START)
        m_wf_hold = run_div(dates_wf, px_wf, sigs_wf, b["gate"], b["stop"], TRAIN_CUTOFF)
        print(f"  refits: {len(anchors)} ({anchors[0].date()} → {anchors[-1].date()})")
        print(f"  walk-forward FULL   : {fmt(m_wf_full)}  tr {m_wf_full['trades']}")
        print(f"  walk-forward HOLDOUT: {fmt(m_wf_hold)}  tr {m_wf_hold['trades']}", flush=True)

    # ═══ Part 3 — side-by-side + correlation to the BTC sleeve ═════════════
    print("\n" + "═" * 96)
    print("PART 3 — verdict table (full window / true holdout) + portfolio angle")
    print("═" * 96)
    print(f"  {'':38s} {'— full 2024-03-05 → now —':>42s}   {'— holdout Aug-25→ —':>40s}")
    for lbl, mf, mh in rows:
        print(f"  {lbl:38s} {fmt(mf)}   {fmt(mh)}")
    for key in results:
        r = results[key]
        star = " ◀ champion" if key == champion else ""
        print(f"  ETH-native div [{key:9s}]{'':13s} {fmt(r['m_full'])}   "
              f"{fmt(r['m_hold'])}{star}")
    if not args.skip_walkforward:
        print(f"  {'ETH-native div (walk-forward champ)':38s} {fmt(m_wf_full)}   {fmt(m_wf_hold)}")

    # correlation of daily strategy returns vs the live BTC CT sleeve
    s_btc = dict(sigs_btc); s_btc["tf2_entry"] = sigs_btc["tf2_entry_pure"]
    bt_btc = E._run_bt(dates_b, comp_btc["actual_close"].to_numpy(float), s_btc,
                       None, pd.Timestamp(SLEEVE_START))
    r_btc = bt_btc["nav"].pct_change().rename("BTC_sleeve")
    r_inc = incumbents["CT-on-BTC MA −8% (shipped)"].pct_change().rename("ETH_on_BTC")
    r_nat = ch["m_full"]["nav"].pct_change().rename("ETH_native")
    cor = pd.concat([r_btc, r_inc, r_nat], axis=1).dropna().corr()
    print("\n  strategy-return correlation to the live BTC sleeve "
          f"(diversification angle; ETH-on-BTC was the ETH sleeve's weak point):")
    print(f"    ETH-on-BTC (shipped): {cor.loc['BTC_sleeve','ETH_on_BTC']:+.2f}   "
          f"ETH-native (champion): {cor.loc['BTC_sleeve','ETH_native']:+.2f}")

    ch_h, ch_f = ch["m_hold"], ch["m_full"]
    print("\n  ── bottom line ──")
    print(f"  incumbent (shipped) : full {fmt(incumbent_full)} · holdout {fmt(incumbent_hold)}")
    print(f"  ETH-native champion : full {fmt(ch_f)} · holdout {fmt(ch_h)}")
    print(f"  champion beats shipped on holdout Sharpe? "
          f"{ch_h['sharpe'] > incumbent_hold['sharpe']}   return? "
          f"{ch_h['total'] > incumbent_hold['total']}")
    print(f"\nDone in {time.time()-t0:.0f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
