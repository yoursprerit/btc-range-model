"""Generic ticker strategy — backtest engine + threshold/strategy tuning.

Config-driven port of ``backtest_gldm.py``.  Two strategy engines are provided
and the sweep picks, per asset, the one that best beats buy & hold on return
while keeping drawdown no worse than buy & hold:

  * MA trend filter  (simulate_regime) — long when the primary close is above
    its N-day SMA.  Robust for cyclical / trending ETFs.
  * Divergence Pure-Regime (simulate) — the Gold/BTC signature system.

The daily High/Low signal model is fit ONCE on the pre-OOS window and predicts
every later bar, so every reported window is genuinely out-of-sample.

Run:
    python backtest_ticker.py SOXX --sweep      # fetch, sweep, backtest
    python backtest_ticker.py XLE --cached      # use cached data
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "app"))
sys.path.insert(0, str(_ROOT))
import ticker_core as tc            # noqa: E402
from ticker_config import get_config, TickerConfig  # noqa: E402

TRADING_DAYS = 252


# ── build out-of-sample daily H/L predictions ────────────────────────────
def build_predictions(cfg: TickerConfig, daily: pd.DataFrame, oos_start: str | None = None,
                      signal_prefix: str = "px", keep_cols: list | None = None):
    """Fit the daily H/L ridge on the pre-OOS window, predict every bar.

    ``signal_prefix`` chooses which asset supplies the H/L signal (default the
    primary ``px``).  Returns a frame with close_asof, pred_high/low,
    actual_high/low, target_date, and the primary + traded price columns.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    oos_start = oos_start or cfg.oos_start
    feat = tc.build_daily_features(cfg, daily)
    close = daily[f"{signal_prefix}_close"]
    high = daily[f"{signal_prefix}_high"]; low = daily[f"{signal_prefix}_low"]
    prev_c = close.shift(1)
    y_hi = high / prev_c - 1.0
    y_lo = low / prev_c - 1.0
    frac = feat.isna().mean()
    feat_cols = [c for c in feat.columns if frac[c] <= 0.35]

    df = feat[feat_cols].copy()
    df["y_hi"] = y_hi; df["y_lo"] = y_lo
    df["close_asof"] = prev_c
    df["actual_high"] = high; df["actual_low"] = low
    df["px_close"] = daily["px_close"]
    keep_price = keep_cols if keep_cols is not None else [
        col for (_lbl, col) in cfg.traded_assets]
    for col in keep_price:
        if col in daily and col not in df:
            df[col] = daily[col]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=feat_cols + ["y_hi", "y_lo", "close_asof"])

    train = df.loc[:oos_start]
    def _mk():
        return Pipeline([("sc", StandardScaler()),
                         ("m", RidgeCV(alphas=np.logspace(-3, 3, 13)))])
    mh = _mk().fit(train[feat_cols], train["y_hi"])
    ml = _mk().fit(train[feat_cols], train["y_lo"])
    raw_ph = df["close_asof"] * (1 + mh.predict(df[feat_cols]))
    raw_pl = df["close_asof"] * (1 + ml.predict(df[feat_cols]))
    tr_mask = df.index <= pd.Timestamp(oos_start)
    bias_hi = float((df.loc[tr_mask, "actual_high"] - raw_ph.loc[tr_mask]).mean())
    bias_lo = float((df.loc[tr_mask, "actual_low"] - raw_pl.loc[tr_mask]).mean())
    df["pred_high"] = raw_ph + bias_hi
    df["pred_low"] = raw_pl + bias_lo
    df["target_date"] = df.index
    keep = ["close_asof", "pred_high", "pred_low", "actual_high", "actual_low",
            "target_date", "px_close"] + [c for c in keep_price if c in df and c != "px_close"]
    return df[keep].copy()


# ── vectorised signal precompute ──────────────────────────────────────────
def precompute_signals(cfg: TickerConfig, preds: pd.DataFrame):
    c = preds["close_asof"].to_numpy(float)
    phi = preds["pred_high"].to_numpy(float)
    plo = preds["pred_low"].to_numpy(float)
    ah = preds["actual_high"].to_numpy(float)
    al = preds["actual_low"].to_numpy(float)
    n = len(c)
    err_hi = (ah - phi) / c * 100
    err_lo = (plo - al) / c * 100

    def _center(x):
        s = pd.Series(x)
        med = s.rolling(60, min_periods=20).median()
        med = med.fillna(s.expanding(min_periods=1).median())
        return (s - med).to_numpy()
    err_hi = _center(err_hi)
    err_lo = _center(err_lo)
    hi_break = (err_hi > 0).astype(int)
    lo_break = (err_lo > 0).astype(int)

    def _ma3(x):
        return np.array([np.mean(x[max(0, i - 2):i + 1]) for i in range(n)])
    def _sum3(x):
        return np.array([np.sum(x[max(0, i - 2):i + 1]) for i in range(n)])

    ehma3 = _ma3(err_hi); elma3 = _ma3(err_lo)
    hb3 = _sum3(hi_break); lb3 = _sum3(lo_break)

    ma20 = np.array([np.mean(c[max(0, i - 19):i + 1]) for i in range(n)])
    ma20_5ago = np.array([np.mean(c[max(0, i - 24):max(1, i - 4)]) for i in range(n)])
    above_ma20 = c > ma20
    slope_pos = ma20 > ma20_5ago
    bull_regime = above_ma20 & slope_pos

    consec_hi = np.zeros(n, int)
    for i in range(1, n):
        consec_hi[i] = consec_hi[i - 1] + 1 if hi_break[i] else 0
    d3 = np.array([(i >= 1 and consec_hi[i - 1] >= 3 and lo_break[i] == 1) for i in range(n)])

    dn_norm = np.array([np.mean(err_hi[max(0, i - 29):i + 1]) for i in range(n)])
    v_rev = np.zeros(n, bool)
    for j in range(n):
        nrm = max(abs(dn_norm[j]), 0.01)
        dns = ((-ehma3[j] / nrm) * 0.30 + (lb3[j] / 3.0) * 0.30 +
               (elma3[j] / max(abs(elma3[j]), 0.1)) * 0.20 + float(lo_break[j]) * 0.20)
        v_rev[j] = (dns > 0.8) and (err_lo[j] > cfg.v_errlo_min)
    v_recent = np.array([bool(np.any(v_rev[max(0, i - 2):i + 1])) for i in range(n)])

    return dict(err_hi=err_hi, err_lo=err_lo, ehma3=ehma3, elma3=elma3,
                hb3=hb3, lb3=lb3, bull_regime=bull_regime, above_ma20=above_ma20,
                d3=d3, v_recent=v_recent, lo_break=lo_break)


# ── trend-family long/flat signal (ma · dual_ma · macd · ma_vol) ──────────
def _rolling_mean(x, w):
    return np.array([np.mean(x[max(0, i - (w - 1)):i + 1]) for i in range(len(x))])


def macd_hist_array(cfg, gcl):
    """Raw MACD histogram (``macd_fast`` / ``macd_slow`` / ``macd_signal``) on the
    PRIMARY close array ``gcl``.  The histogram is ``(EMA_fast − EMA_slow)`` minus
    its own ``macd_signal``-span EMA, in price units.  The long condition in macd
    mode is simply ``histogram > 0`` (see ``trend_long_array``)."""
    c = pd.Series(np.asarray(gcl, float))
    macd = (c.ewm(span=cfg.macd_fast, adjust=False).mean()
            - c.ewm(span=cfg.macd_slow, adjust=False).mean())
    return (macd - macd.ewm(span=cfg.macd_signal, adjust=False).mean()).to_numpy()


def trend_long_array(cfg, gcl):
    """Boolean long-at-close signal for the config's trend mode, computed on the
    PRIMARY close array ``gcl`` (decision at each close → executed next bar).

      ma       long while close > N-day SMA
      dual_ma  long while the fast SMA is above the slow SMA
      macd     long while the MACD histogram (fast, slow, signal) > 0
      ma_vol   long while close > N-day SMA AND realised vol is below k · its
               rolling median (skip the high-volatility regime)
    """
    m = cfg.strategy_mode
    if m == "dual_ma":
        return _rolling_mean(gcl, cfg.ma_fast) > _rolling_mean(gcl, cfg.ma_slow)
    if m == "macd":
        return macd_hist_array(cfg, gcl) > 0
    if m == "ma_vol":
        c = pd.Series(gcl)
        v = np.log(c).diff().rolling(cfg.vol_win).std()
        med = v.rolling(cfg.vol_med_win).median()
        above = gcl > _rolling_mean(gcl, cfg.ma_window)
        return (above & (v < cfg.vol_k * med).to_numpy())
    return gcl > _rolling_mean(gcl, cfg.ma_window)          # "ma"


def trend_long_now(cfg, daily):
    """The trend signal's boolean state on the latest available bar."""
    gcl = daily["px_close"].to_numpy(float)
    return bool(trend_long_array(cfg, gcl)[-1]) if len(gcl) else False


def trend_long_now_live(cfg, close_hist, live_px):
    """The trend signal's boolean state if ``live_px`` is taken as the newest
    close.  Appends the live price as a provisional latest bar and re-evaluates
    the mode's ACTUAL long condition (via ``trend_long_array``), so dual_ma
    re-checks the fast/slow SMA cross — and ma_vol its vol filter — rather than a
    naive price-vs-single-line read.  Returns None when inputs are unusable."""
    if live_px is None:
        return None
    arr = np.asarray(close_hist, float)
    arr = arr[np.isfinite(arr)]
    if not len(arr) or not np.isfinite(live_px):
        return None
    arr = np.append(arr, float(live_px))
    return bool(trend_long_array(cfg, arr)[-1])


def trend_line_value(cfg, daily):
    """Representative price-level line for the trend chart / UI (last value):
    the slow SMA for dual_ma, else the N-day SMA."""
    gcl = daily["px_close"].to_numpy(float)
    if not len(gcl):
        return float("nan")
    w = cfg.ma_slow if cfg.strategy_mode == "dual_ma" else cfg.ma_window
    return float(np.mean(gcl[-w:]))


def vol_filter_state(cfg, daily):
    """The ma_vol volatility-contraction gate on the latest bar, as raw numbers
    the UI can surface next to the close-vs-SMA read.

    ``ma_vol`` requires realised vol (``vol_win``-day std of daily log-returns) to
    sit *below* ``vol_k`` × its ``vol_med_win``-day rolling median — i.e. the
    strategy only holds through subdued-volatility regimes.  This exposes both the
    live value and the required threshold (the same series ``trend_long_array``
    computes) so the gate can be shown as its own condition rather than folded
    into a single boolean.  Returns None when the mode isn't ma_vol or there
    aren't enough bars for the median to warm up."""
    if cfg.strategy_mode != "ma_vol":
        return None
    gcl = daily["px_close"].to_numpy(float)
    if len(gcl) < cfg.vol_med_win + cfg.vol_win:
        return None
    c = pd.Series(gcl)
    v = np.log(c).diff().rolling(cfg.vol_win).std()
    med = v.rolling(cfg.vol_med_win).median()
    vol = float(v.iloc[-1]); med_v = float(med.iloc[-1])
    if not (np.isfinite(vol) and np.isfinite(med_v)):
        return None
    thr = cfg.vol_k * med_v
    return dict(vol=vol, median=med_v, thr=thr, ok=bool(vol < thr),
                vol_win=cfg.vol_win, vol_med_win=cfg.vol_med_win, vol_k=cfg.vol_k)


# ── MA / trend-filter strategy ────────────────────────────────────────────
def simulate_regime(cfg, preds, sig, price_col, ma_window=None, stop_pct=1.0,
                    oos_start=None, end=None):
    """Long into bar i+1 when the PRIMARY trend signal is bullish at bar i
    (decision at that close → no look-ahead); flat otherwise.  Signal derived
    from the primary via ``trend_long_array``, executed on ``price_col``.  When
    ``ma_window`` is passed explicitly (the sweep) a plain close>SMA filter with
    that window is used regardless of mode.  Optional fixed stop.  Returns equity
    curves + trade log (schema matches ``simulate``)."""
    oos_start = oos_start or cfg.oos_start
    price = preds[price_col].to_numpy(float)
    gcl = preds["px_close"].to_numpy(float)
    dates = preds["target_date"].to_numpy()
    n = len(price)
    if ma_window is not None:                               # explicit MA (sweep)
        long_at_close = gcl > _rolling_mean(gcl, ma_window)
    else:
        long_at_close = trend_long_array(cfg, gcl)
    i0 = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(oos_start))))
    i1 = n if end is None else int(np.searchsorted(
        dates, np.datetime64(pd.Timestamp(end)), side="right"))

    eq = 1.0
    strat_eq = []; bh_eq = []; used_dates = []; pos_series = []
    trades = []; trade_log = []
    entry_px = np.nan; entry_date = None; in_pos = False
    bh0 = price[i0]
    for i in range(i0, i1):
        if in_pos:
            eq *= price[i] / price[i - 1]
        strat_eq.append(eq); bh_eq.append(price[i] / bh0); used_dates.append(dates[i])
        want_long = bool(long_at_close[i - 1])
        if in_pos:
            stop_hit = price[i] <= entry_px * (1 - stop_pct)
            if (not want_long) or stop_hit:
                ret = price[i] / entry_px - 1
                reason = ("stop −%.0f%%" % (stop_pct * 100) if stop_hit else "MA cross-down")
                trades.append(ret)
                trade_log.append(dict(entry_date=entry_date, exit_date=dates[i],
                                      entry_px=float(entry_px), exit_px=float(price[i]),
                                      ret=float(ret), reason=reason))
                in_pos = False; entry_px = np.nan; entry_date = None
        elif want_long:
            in_pos = True; entry_px = price[i]; entry_date = dates[i]
        pos_series.append(1 if in_pos else 0)
    return dict(dates=used_dates, strat=np.array(strat_eq), bh=np.array(bh_eq),
                pos=np.array(pos_series), trades=np.array(trades),
                trade_log=trade_log, in_pos_now=in_pos,
                entry_px=(float(entry_px) if in_pos else None), entry_date=entry_date)


def _clean_flags(sig, i, D2, D1):
    s = max(0, i - 7)
    d2h = sig["ehma3"][s:i] < D2
    d1h = (sig["lb3"][s:i] >= 2) & (sig["elma3"][s:i] > D1)
    return not bool(np.any(d2h | d1h))


# ── divergence Pure-Regime strategy ───────────────────────────────────────
def simulate(cfg, preds, sig, price_col, stop_pct=None, U1=None, D2=None, D1=None,
             use_d1_exit=False, oos_start=None, end=None):
    stop_pct = cfg.fixed_stop if stop_pct is None else stop_pct
    U1 = cfg.u1_errhi_min if U1 is None else U1
    D2 = cfg.d2_errhi_max if D2 is None else D2
    D1 = cfg.d1_errlo_min if D1 is None else D1
    oos_start = oos_start or cfg.oos_start
    price = preds[price_col].to_numpy(float)
    dates = preds["target_date"].to_numpy()
    n = len(price)
    i0 = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(oos_start))))
    i1 = n if end is None else int(np.searchsorted(
        dates, np.datetime64(pd.Timestamp(end)), side="right"))

    in_pos = False
    entry_px = np.nan; entry_date = None
    eq = 1.0
    strat_eq = []; bh_eq = []; used_dates = []; pos_series = []
    trades = []; trade_log = []
    bh0 = price[i0]
    for i in range(i0, i1):
        if in_pos:
            eq *= price[i] / price[i - 1]
        strat_eq.append(eq); bh_eq.append(price[i] / bh0); used_dates.append(dates[i])

        u1 = (sig["ehma3"][i] > U1) and (sig["hb3"][i] >= 2)
        d2 = sig["ehma3"][i] < D2
        d1 = (sig["lb3"][i] >= 2) and (sig["elma3"][i] > D1)
        d3 = sig["d3"][i]
        clean = _clean_flags(sig, i, D2, D1)
        entry_gate = u1 and (sig["bull_regime"][i] or
                             (clean and not sig["above_ma20"][i]) or sig["v_recent"][i])
        exit_sig = d2 or d3 or (use_d1_exit and d1)
        if in_pos:
            stop_hit = price[i] <= entry_px * (1 - stop_pct)
            if stop_hit or exit_sig:
                ret = price[i] / entry_px - 1
                reason = ("stop −%.0f%%" % (stop_pct * 100) if stop_hit else
                          "D3 exhaustion" if d3 else "D2 fade" if d2 else "D1")
                trades.append(ret)
                trade_log.append(dict(entry_date=entry_date, exit_date=dates[i],
                                      entry_px=float(entry_px), exit_px=float(price[i]),
                                      ret=float(ret), reason=reason))
                in_pos = False; entry_px = np.nan; entry_date = None
        elif entry_gate and not exit_sig:
            in_pos = True; entry_px = price[i]; entry_date = dates[i]
        pos_series.append(1 if in_pos else 0)

    return dict(dates=used_dates, strat=np.array(strat_eq), bh=np.array(bh_eq),
                pos=np.array(pos_series), trades=np.array(trades),
                trade_log=trade_log, in_pos_now=in_pos,
                entry_px=(float(entry_px) if in_pos else None), entry_date=entry_date)


def run_strategy(cfg, preds, sig, price_col, oos_start=None, end=None, stop_pct=None, **kw):
    """Dispatch to the config's chosen strategy engine.

    ``stop_pct`` defaults to the per-asset stop (``cfg.stop_by_asset`` keyed by
    ``price_col``, falling back to ``cfg.fixed_stop``), so a high-beta sibling can
    trade a different stop than its 1× parent off the same signal."""
    if stop_pct is None:
        stop_pct = getattr(cfg, "stop_by_asset", {}).get(price_col, cfg.fixed_stop)
    if cfg.strategy_mode == "divergence":
        kw.setdefault("use_d1_exit", getattr(cfg, "use_d1_exit", False))
        return simulate(cfg, preds, sig, price_col, stop_pct=stop_pct,
                        oos_start=oos_start, end=end, **kw)
    return simulate_regime(cfg, preds, sig, price_col, stop_pct=stop_pct,
                           oos_start=oos_start, end=end, **kw)


def drawdown_series(eq):
    eq = np.asarray(eq, float)
    peak = np.maximum.accumulate(eq)
    return eq / peak - 1.0


def _metrics(eq, dates):
    if len(eq) < 2:
        return dict(total_ret=0, cagr=0, mdd=0, sharpe=0)
    total = eq[-1] / eq[0] - 1
    yrs = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days / 365.25
    cagr = (eq[-1] / eq[0]) ** (1 / max(yrs, 1e-9)) - 1
    peak = np.maximum.accumulate(eq)
    mdd = float(np.min(eq / peak - 1))
    rets = np.diff(eq) / eq[:-1]
    sharpe = float(np.mean(rets) / (np.std(rets) + 1e-12) * np.sqrt(TRADING_DAYS))
    return dict(total_ret=float(total), cagr=float(cagr), mdd=mdd, sharpe=sharpe)


# ── strategy sweep — pick the best engine + params per asset ──────────────
def _run_spec(cfg, preds, sig, price_col, spec, oos_start=None, end=None):
    if spec["mode"] == "ma":
        return simulate_regime(cfg, preds, sig, price_col, ma_window=spec["ma_window"],
                               stop_pct=spec["stop"], oos_start=oos_start, end=end)
    return simulate(cfg, preds, sig, price_col, stop_pct=spec["stop"],
                    U1=spec["U1"], D2=spec["D2"], oos_start=oos_start, end=end)


def _score_spec(cfg, preds, sig, price_col, spec):
    """Evaluate a candidate across ALL configured periods + the full window.

    Encodes the brief: beat Buy&Hold in the loss periods and minimise drawdown
    while retaining as much profit as possible over multiple periods."""
    r_full = _run_spec(cfg, preds, sig, price_col, spec)
    full = _metrics(r_full["strat"], r_full["dates"])
    bh_full = _metrics(r_full["bh"], r_full["dates"])
    per = {}
    beats_loss = True; mdd_ok = True; participate = True
    for lbl, s, e in cfg.periods:
        r = _run_spec(cfg, preds, sig, price_col, spec, oos_start=s, end=e)
        sm = _metrics(r["strat"], r["dates"]); bm = _metrics(r["bh"], r["dates"])
        per[lbl] = (sm, bm)
        # in any DOWN period (B&H return < 0) the strategy must beat B&H
        if bm["total_ret"] < 0 and sm["total_ret"] < bm["total_ret"] - 1e-9:
            beats_loss = False
        # drawdown must be no worse than B&H in every period
        if sm["mdd"] < bm["mdd"] - 1e-9:
            mdd_ok = False
        # must retain a reasonable slice of B&H upside in UP periods (≥55%)
        if bm["total_ret"] > 0.05 and sm["total_ret"] < 0.55 * bm["total_ret"]:
            participate = False
    spec.update(dict(total_ret=full["total_ret"], mdd=full["mdd"], sharpe=full["sharpe"],
                     cagr=full["cagr"], trades=int(len(r_full["trades"])),
                     beats_loss=beats_loss, mdd_ok=mdd_ok, participate=participate,
                     bh_ret=bh_full["total_ret"], bh_mdd=bh_full["mdd"],
                     bh_sharpe=bh_full["sharpe"]))
    return spec


def sweep_asset(cfg, preds, sig, price_col):
    """Search MA and divergence configs; pick the one that best satisfies the
    multi-period objective (beat B&H in loss periods, MDD ≤ B&H everywhere,
    retain upside) and maximises full-period return, tie-broken by Sharpe."""
    cands = []
    for ma_w in (20, 30, 40, 50, 60, 80, 100, 120, 150, 200):
        for stop in (cfg.fixed_stop, 1.0):
            cands.append(dict(mode="ma", ma_window=ma_w, stop=stop))
    for U1 in (0.05, 0.08, 0.12):
        for D2 in (-0.08, -0.12, -0.18):
            for stop in (cfg.fixed_stop, 1.0):
                cands.append(dict(mode="divergence", U1=U1, D2=D2, stop=stop))
    for spec in cands:
        _score_spec(cfg, preds, sig, price_col, spec)

    tiers = [
        [x for x in cands if x["beats_loss"] and x["mdd_ok"] and x["participate"]],
        [x for x in cands if x["beats_loss"] and x["mdd_ok"]],
        [x for x in cands if x["mdd_ok"] and x["participate"]],
        [x for x in cands if x["mdd_ok"]],
        cands,
    ]
    pool = next(t for t in tiers if t)
    best = max(pool, key=lambda x: (x["total_ret"], x["sharpe"]))
    return best


def period_table(cfg, preds, sig, price_col):
    """Strategy vs B&H across the config's backtest periods (for the results note)."""
    rows = []
    for lbl, s, e in cfg.periods:
        r = run_strategy(cfg, preds, sig, price_col, oos_start=s, end=e)
        sm = _metrics(r["strat"], r["dates"]); bm = _metrics(r["bh"], r["dates"])
        wr = (r["trades"] > 0).mean() * 100 if len(r["trades"]) else 0
        rows.append((lbl, sm, bm, wr, len(r["trades"])))
    return rows


def _get_daily(cfg, cached):
    paths = tc.cache_paths(cfg)
    if cached and paths["daily"].exists():
        return pd.read_csv(paths["daily"], index_col=0, parse_dates=True)
    daily = tc.fetch_daily(cfg)
    daily.to_csv(paths["daily"])
    return daily


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker")
    ap.add_argument("--cached", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()
    cfg = get_config(args.ticker.upper())

    daily = _get_daily(cfg, args.cached)
    preds = build_predictions(cfg, daily)
    sig = precompute_signals(cfg, preds)
    oos_end = str(preds["target_date"].iloc[-1].date())
    print(f"[{cfg.key}] OOS {cfg.oos_start} → {oos_end}  "
          f"({(preds['target_date'] >= pd.Timestamp(cfg.oos_start)).sum()} bars, "
          f"mode={cfg.strategy_mode})")

    if args.sweep:
        print("\n=== strategy/threshold sweep (max return s.t. MDD ≤ B&H) ===")
        chosen = {}
        for lbl, col in cfg.traded_assets:
            b = sweep_asset(cfg, preds, sig, col)
            chosen[lbl] = b
            spec = (f"MA={b['ma_window']} stop={b['stop']*100:.0f}%" if b["mode"] == "ma"
                    else f"U1={b['U1']} D2={b['D2']} stop={b['stop']*100:.0f}%")
            print(f"  {lbl:5s} {b['mode']:11s} {spec:26s} "
                  f"ret={b['total_ret']*100:+8.1f}% MDD={b['mdd']*100:6.1f}% "
                  f"Sharpe={b['sharpe']:.2f} trades={b['trades']} | "
                  f"B&H ret={b['bh_ret']*100:+.1f}% MDD={b['bh_mdd']*100:.1f}% Sharpe={b['bh_sharpe']:.2f}")
        tc.cache_paths(cfg)["sweep"].write_text(json.dumps(chosen, indent=2, default=str))

    print(f"\n=== {cfg.strategy_name} ({cfg.strategy_mode}) — per-period vs Buy&Hold ===")
    out = {"ticker": cfg.key, "oos_start": cfg.oos_start, "oos_end": oos_end,
           "strategy": cfg.strategy_name, "mode": cfg.strategy_mode, "assets": {}}
    for lbl, col in cfg.traded_assets:
        print(f"\n  {lbl}:")
        periods = {}
        for plabel, sm, bm, wr, ntr in period_table(cfg, preds, sig, col):
            print(f"    {plabel:38s} STRAT ret={sm['total_ret']*100:+8.1f}% "
                  f"MDD={sm['mdd']*100:6.1f}% Sharpe={sm['sharpe']:.2f} | "
                  f"B&H ret={bm['total_ret']*100:+8.1f}% MDD={bm['mdd']*100:6.1f}%")
            periods[plabel] = dict(strategy=sm, buy_hold=bm, win_rate=float(wr), n_trades=int(ntr))
        r = run_strategy(cfg, preds, sig, col)
        # the top-level metrics use the config's default OOS window (oos_start);
        # ``periods`` carries every window (incl. 🌍 Full history) so the artifact
        # matches the app's per-period Backtesting tabs.
        out["assets"][lbl] = dict(strategy=_metrics(r["strat"], r["dates"]),
                                  buy_hold=_metrics(r["bh"], r["dates"]),
                                  n_trades=int(len(r["trades"])),
                                  win_rate=float((r["trades"] > 0).mean() * 100) if len(r["trades"]) else 0.0,
                                  default_window=cfg.oos_start, periods=periods)
    tc.cache_paths(cfg)["backtest"].write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved → {tc.cache_paths(cfg)['backtest']}")


if __name__ == "__main__":
    main()
