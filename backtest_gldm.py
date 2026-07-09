"""GLDM trend-signature strategy — backtest engine + threshold tuning.

Mirrors the BTC application's TF1/Pure-Regime strategy, adapted to gold:

  * The daily High/Low model is fit ONCE on the pre-OOS training window and
    predicts every subsequent bar, so all reported performance is genuinely
    out-of-sample (no look-ahead in the signals).
  * The GLDM-derived entry/exit signals are then applied to the 1x asset
    (GLDM) and to its leveraged / high-beta analogs (UGL 2x, GDX miners) —
    exactly how the BTC app runs one signal across BTC / MSTR / MSTU.
  * A frontier sweep over the U1 entry threshold, D2 exit threshold and the
    per-asset fixed stop finds the configuration that maximises full-period
    return while keeping the drawdown in check.

Run:
    python backtest_gldm.py            # fetch fresh
    python backtest_gldm.py --cached   # use data/gldm snapshots
    python backtest_gldm.py --sweep    # also run the threshold frontier sweep
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "app"))
sys.path.insert(0, str(_ROOT))
import gldm_core as gc  # noqa: E402

OOS_START = "2021-01-01"          # signals/trades reported from here on
TRADING_DAYS = 252
RESULTS_JSON = gc.GLDM_DATA_DIR / "backtest_results.json"


# ── build out-of-sample daily H/L predictions ────────────────────────────
def build_predictions(daily: pd.DataFrame, oos_start: str = OOS_START):
    """Fit the daily H/L ridge on the pre-OOS window, predict every bar.

    Returns a frame with columns: close_asof, pred_high, pred_low,
    actual_high, actual_low, target_date, gldm_close (+ analog closes),
    indexed by target date, restricted to bars with a prior close.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    feat = gc.build_daily_features(daily)
    close = daily["gldm_close"]; high = daily["gldm_high"]; low = daily["gldm_low"]
    prev_c = close.shift(1)
    y_hi = high / prev_c - 1.0
    y_lo = low / prev_c - 1.0
    frac = feat.isna().mean()
    feat_cols = [c for c in feat.columns if frac[c] <= 0.35]

    df = feat[feat_cols].copy()
    df["y_hi"] = y_hi; df["y_lo"] = y_lo
    df["close_asof"] = prev_c
    df["actual_high"] = high; df["actual_low"] = low
    df["gldm_close"] = close
    for a in ("ugl", "gdx", "nugt"):
        if f"{a}_close" in daily:
            df[f"{a}_close"] = daily[f"{a}_close"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feat_cols + ["y_hi", "y_lo", "close_asof"])

    train = df.loc[:oos_start]
    def _mk():
        return Pipeline([("sc", StandardScaler()),
                         ("m", RidgeCV(alphas=np.logspace(-3, 3, 13)))])
    mh = _mk().fit(train[feat_cols], train["y_hi"])
    ml = _mk().fit(train[feat_cols], train["y_lo"])
    raw_ph = df["close_asof"] * (1 + mh.predict(df[feat_cols]))
    raw_pl = df["close_asof"] * (1 + ml.predict(df[feat_cols]))
    # ── Calibration (fit on train, apply to all): center the divergence error
    #    at zero on the training window so hi/lo breaks occur ~50% of the time.
    #    Without this, ridge's raw band is biased and the U1/D2 signals never
    #    fire.  This is the gold analog of the BTC daily-H/L calibration_meta.
    tr_mask = df.index <= pd.Timestamp(oos_start)
    # symmetric centering: pred = raw + mean(actual − raw) on the train window,
    # so both err_hi=(actual_high−pred_high) and err_lo derived from
    # (actual_low−pred_low) sit at zero and hi/lo breaks occur ~50% of bars.
    bias_hi = float((df.loc[tr_mask, "actual_high"] - raw_ph.loc[tr_mask]).mean())
    bias_lo = float((df.loc[tr_mask, "actual_low"] - raw_pl.loc[tr_mask]).mean())
    df["pred_high"] = raw_ph + bias_hi
    df["pred_low"] = raw_pl + bias_lo
    df["target_date"] = df.index
    keep = ["close_asof", "pred_high", "pred_low", "actual_high", "actual_low",
            "target_date", "gldm_close"] + \
           [c for c in ("ugl_close", "gdx_close", "nugt_close") if c in df]
    return df[keep].copy()


# ── vectorised signal precompute (consistent with gldm_core) ──────────────
def precompute_signals(preds: pd.DataFrame):
    c = preds["close_asof"].to_numpy(float)
    phi = preds["pred_high"].to_numpy(float)
    plo = preds["pred_low"].to_numpy(float)
    ah = preds["actual_high"].to_numpy(float)
    al = preds["actual_low"].to_numpy(float)
    n = len(c)
    err_hi = (ah - phi) / c * 100
    err_lo = (plo - al) / c * 100
    # Regime-adaptive centering: the ridge predicts a symmetric range but gold's
    # daily H/L vs prior close is structurally asymmetric AND drifts across
    # regimes, so a static bias fails.  Subtract each error's rolling median so
    # the divergence signal is self-calibrating (breaks ~50%). Break flags are
    # then defined off the centered error for internal consistency.
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

    # MA20 regime
    ma20 = np.array([np.mean(c[max(0, i - 19):i + 1]) for i in range(n)])
    ma20_5ago = np.array([np.mean(c[max(0, i - 24):max(1, i - 4)]) for i in range(n)])
    above_ma20 = c > ma20
    slope_pos = ma20 > ma20_5ago
    bull_regime = above_ma20 & slope_pos

    # D3 exhaustion: first lo_break after >=3 consecutive hi_breaks
    consec_hi = np.zeros(n, int)
    for i in range(1, n):
        consec_hi[i] = consec_hi[i - 1] + 1 if hi_break[i] else 0
    d3 = np.array([(i >= 1 and consec_hi[i - 1] >= 3 and lo_break[i] == 1) for i in range(n)])

    # V-reversal per bar (dn_score>0.8 & single-bar low undershoot>V)
    dn_norm = np.array([np.mean(err_hi[max(0, i - 29):i + 1]) for i in range(n)])
    v_rev = np.zeros(n, bool)
    for j in range(n):
        nrm = max(abs(dn_norm[j]), 0.01)
        dns = ((-ehma3[j] / nrm) * 0.30 + (lb3[j] / 3.0) * 0.30 +
               (elma3[j] / max(abs(elma3[j]), 0.1)) * 0.20 + float(lo_break[j]) * 0.20)
        v_rev[j] = (dns > 0.8) and (err_lo[j] > gc.V_ERRLO_MIN)
    v_recent = np.array([bool(np.any(v_rev[max(0, i - 2):i + 1])) for i in range(n)])

    # MA50 trend-filter regime on the GLDM own close (primary gold strategy).
    # Computed per bar from close_asof so it is known at decision time.
    gcl = preds["gldm_close"].to_numpy(float)
    ma50 = np.array([np.mean(gcl[max(0, i - 49):i + 1]) for i in range(n)])
    above_ma50 = gcl > ma50

    return dict(err_hi=err_hi, err_lo=err_lo, ehma3=ehma3, elma3=elma3,
                hb3=hb3, lb3=lb3, bull_regime=bull_regime, above_ma20=above_ma20,
                d3=d3, v_recent=v_recent, lo_break=lo_break,
                ma50=ma50, above_ma50=above_ma50)


def simulate_regime(preds, sig, price_col, ma_window=50, stop_pct=1.0,
                    d2_exit=False, oos_start=OOS_START):
    """PRIMARY GLDM strategy — the price-vs-MA trend filter.

    Long into bar i+1 when the GLDM close at bar i is above its ``ma_window``
    SMA (decision made at that close → no look-ahead); flat otherwise.  The
    signal is derived from GLDM and applied to ``price_col`` (GLDM / UGL /
    GDX), mirroring how the BTC app runs one signal across BTC / MSTR / MSTU.
    Optional ``d2_exit`` forces flat when the divergence D2 signal fires, and
    an optional fixed stop caps single-position loss.
    """
    price = preds[price_col].to_numpy(float)
    gcl = preds["gldm_close"].to_numpy(float)
    dates = preds["target_date"].to_numpy()
    n = len(price)
    ma = np.array([np.mean(gcl[max(0, i - (ma_window - 1)):i + 1]) for i in range(n)])
    long_at_close = gcl > ma
    if d2_exit:
        long_at_close = long_at_close & ~(sig["ehma3"] < -0.15)
    oos_i0 = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(oos_start))))

    eq = 1.0
    strat_eq = []; bh_eq = []; used_dates = []
    trades = []; entry_px = np.nan; in_pos = False
    bh0 = price[oos_i0]
    for i in range(oos_i0, n):
        want_long = bool(long_at_close[i - 1])   # decided at prior close
        if in_pos:
            eq *= price[i] / price[i - 1]
            stop_hit = price[i] <= entry_px * (1 - stop_pct)
            if (not want_long) or stop_hit:
                trades.append(price[i] / entry_px - 1)
                in_pos = False; entry_px = np.nan
        elif want_long:
            in_pos = True; entry_px = price[i]
        strat_eq.append(eq); bh_eq.append(price[i] / bh0); used_dates.append(dates[i])
    return dict(dates=used_dates, strat=np.array(strat_eq), bh=np.array(bh_eq),
                trades=np.array(trades))


def _clean_flags(sig, i, U1, D2, D1):
    """clean_10d: no D1/D2 in the 7 bars before i."""
    s = max(0, i - 7)
    d2h = sig["ehma3"][s:i] < D2
    d1h = (sig["lb3"][s:i] >= 2) & (sig["elma3"][s:i] > D1)
    return not bool(np.any(d2h | d1h))


# ── strategy simulation on one price series ───────────────────────────────
def simulate(preds, sig, price_col, stop_pct, U1, D2, D1,
             use_d1_exit=False, oos_start=OOS_START, end=None):
    """Walk bars; long when the Pure-Regime entry gate fires, exit on D2/D3
    (+optional D1) or the fixed stop.  Window = [oos_start, end].

    Returns, over the window: strategy & buy&hold equity curves (start 1.0), a
    per-bar long/flat position array, the completed-trade log (entry/exit date,
    price, return, exit reason) and the trade-return array.
    """
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
        strat_eq.append(eq)
        bh_eq.append(price[i] / bh0)
        used_dates.append(dates[i])

        u1 = (sig["ehma3"][i] > U1) and (sig["hb3"][i] >= 2)
        d2 = sig["ehma3"][i] < D2
        d1 = (sig["lb3"][i] >= 2) and (sig["elma3"][i] > D1)
        d3 = sig["d3"][i]
        clean = _clean_flags(sig, i, U1, D2, D1)
        entry_gate = u1 and (sig["bull_regime"][i] or
                             (clean and not sig["above_ma20"][i]) or sig["v_recent"][i])

        exit_sig = d2 or d3 or (use_d1_exit and d1)
        if in_pos:
            stop_hit = price[i] <= entry_px * (1 - stop_pct)
            if stop_hit or exit_sig:
                ret = price[i] / entry_px - 1
                reason = ("stop −%.0f%%" % (stop_pct * 100) if stop_hit else
                          "D3 exhaustion" if d3 else
                          "D2 fade" if d2 else "D1")
                trades.append(ret)
                trade_log.append(dict(entry_date=entry_date, exit_date=dates[i],
                                      entry_px=float(entry_px), exit_px=float(price[i]),
                                      ret=float(ret), reason=reason))
                in_pos = False; entry_px = np.nan; entry_date = None
        # Exit ALWAYS overrides entry: never open a new position on a bar that is
        # simultaneously flashing an exit signal (keeps live UI == backtest).
        elif entry_gate and not exit_sig:
            in_pos = True; entry_px = price[i]; entry_date = dates[i]
        pos_series.append(1 if in_pos else 0)

    return dict(dates=used_dates, strat=np.array(strat_eq), bh=np.array(bh_eq),
                pos=np.array(pos_series), trades=np.array(trades),
                trade_log=trade_log, in_pos_now=in_pos,
                entry_px=(float(entry_px) if in_pos else None),
                entry_date=entry_date)


def drawdown_series(eq):
    """Running drawdown (fraction, ≤0) of an equity curve."""
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


def run_asset(preds, sig, asset, strategy="ma50", **kw):
    """Run one strategy on one asset.  strategy ∈ {"ma50", "divergence"}."""
    col = "gldm_close" if asset == "GLDM" else f"{asset.lower()}_close"
    if col not in preds:
        return None
    if strategy == "ma50":
        r = simulate_regime(preds, sig, col, ma_window=kw.get("ma_window", 50),
                            stop_pct=kw.get("stop_pct", 1.0), d2_exit=kw.get("d2_exit", False))
    else:
        r = simulate(preds, sig, col, kw.get("stop_pct", 1.0),
                     kw.get("U1", gc.U1_ERRHI_MIN), kw.get("D2", gc.D2_ERRHI_MAX),
                     kw.get("D1", gc.D1_ERRLO_MIN), kw.get("use_d1_exit", False))
    sm = _metrics(r["strat"], r["dates"]); bm = _metrics(r["bh"], r["dates"])
    trades = r["trades"]
    wr = float((trades > 0).mean() * 100) if len(trades) else 0.0
    return dict(asset=asset, strat=sm, bh=bm, n_trades=int(len(trades)),
                win_rate=wr, avg_trade=float(trades.mean() * 100) if len(trades) else 0.0)


def print_asset(res):
    a = res["asset"]; s = res["strat"]; b = res["bh"]
    print(f"\n  {a:5s}  STRATEGY  ret={s['total_ret']*100:+8.1f}%  "
          f"CAGR={s['cagr']*100:+6.1f}%  MDD={s['mdd']*100:6.1f}%  "
          f"Sharpe={s['sharpe']:.2f}  trades={res['n_trades']}  win={res['win_rate']:.0f}%")
    print(f"         BUY&HOLD  ret={b['total_ret']*100:+8.1f}%  "
          f"CAGR={b['cagr']*100:+6.1f}%  MDD={b['mdd']*100:6.1f}%  Sharpe={b['sharpe']:.2f}")


def sweep_ma(preds, sig, asset):
    """Frontier sweep over the MA window (and optional stop) — the primary
    gold strategy.  Best by Sharpe (risk-adjusted) among configs whose MDD is
    better than buy & hold."""
    col = "gldm_close" if asset == "GLDM" else f"{asset.lower()}_close"
    bh = _metrics(simulate_regime(preds, sig, col, ma_window=50)["bh"],
                  simulate_regime(preds, sig, col, ma_window=50)["dates"])
    best = None
    for ma_w in (20, 30, 40, 50, 60, 80, 100, 150, 200):
        for stop in (0.02, 0.03, 0.05, 1.0):
            r = simulate_regime(preds, sig, col, ma_window=ma_w, stop_pct=stop)
            m = _metrics(r["strat"], r["dates"])
            if m["mdd"] < bh["mdd"] - 1e-9:      # require deeper-drawdown configs skipped
                continue
            score = m["sharpe"]
            if best is None or score > best["score"]:
                best = dict(score=score, ma_window=ma_w, stop=stop, ret=m["total_ret"],
                            cagr=m["cagr"], mdd=m["mdd"], sharpe=m["sharpe"],
                            trades=int(len(r["trades"])))
    if best is None:   # fallback: pure best Sharpe
        for ma_w in (20, 30, 50, 100, 200):
            r = simulate_regime(preds, sig, col, ma_window=ma_w)
            m = _metrics(r["strat"], r["dates"])
            if best is None or m["sharpe"] > best["score"]:
                best = dict(score=m["sharpe"], ma_window=ma_w, stop=1.0, ret=m["total_ret"],
                            cagr=m["cagr"], mdd=m["mdd"], sharpe=m["sharpe"],
                            trades=int(len(r["trades"])))
    best["bh_ret"] = bh["total_ret"]; best["bh_mdd"] = bh["mdd"]; best["bh_sharpe"] = bh["sharpe"]
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cached", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    if args.cached and gc.DAILY_CACHE_CSV.exists():
        daily = pd.read_csv(gc.DAILY_CACHE_CSV, index_col=0, parse_dates=True)
    else:
        daily = gc.fetch_daily(start="2015-01-01")
        daily.to_csv(gc.DAILY_CACHE_CSV)

    preds = build_predictions(daily)
    sig = precompute_signals(preds)
    print(f"OOS window: {OOS_START} → {preds['target_date'].iloc[-1].date()}  "
          f"({(preds['target_date'] >= pd.Timestamp(OOS_START)).sum()} bars)")

    if args.sweep:
        print("\n=== MA-window frontier sweep (best Sharpe, MDD ≤ Buy&Hold) ===")
        chosen = {}
        for asset in ("GLDM", "UGL", "GDX"):
            b = sweep_ma(preds, sig, asset)
            chosen[asset] = b
            print(f"  {asset:5s} best: MA={b['ma_window']:3d} stop={b['stop']*100:.0f}%  "
                  f"ret={b['ret']*100:+.1f}% CAGR={b['cagr']*100:+.1f}% MDD={b['mdd']*100:.1f}% "
                  f"Sharpe={b['sharpe']:.2f} entries={b['trades']} | "
                  f"B&H ret={b['bh_ret']*100:+.1f}% MDD={b['bh_mdd']*100:.1f}% Sharpe={b['bh_sharpe']:.2f}")
        (gc.GLDM_DATA_DIR / "sweep_results.json").write_text(json.dumps(chosen, indent=2, default=str))

    out = {"oos_start": OOS_START, "oos_end": str(preds["target_date"].iloc[-1].date()),
           "primary_strategy": "MA50 trend filter (long when GLDM close > 50-day SMA)",
           "assets": {}}

    print("\n=== PRIMARY: MA50 trend filter — Strategy vs Buy&Hold ===")
    for asset in ("GLDM", "UGL", "GDX"):
        res = run_asset(preds, sig, asset, strategy="ma50", ma_window=50)
        if res is None:
            continue
        print_asset(res)
        out["assets"][asset] = dict(strategy=res["strat"], buy_hold=res["bh"],
                                    n_trades=res["n_trades"], win_rate=res["win_rate"])

    print(f"\n=== CHOSEN STRATEGY: {gc.STRATEGY_NAME} "
          f"(U1={gc.U1_ERRHI_MIN:+.2f} D2={gc.D2_ERRHI_MAX:+.2f} stop={gc.FIXED_STOP*100:.0f}%) "
          f"— traded on GDX & UGL ===")
    out["divergence_alt"] = {}
    for asset in ("GLDM", "UGL", "GDX"):
        res = run_asset(preds, sig, asset, strategy="divergence",
                        stop_pct=gc.FIXED_STOP, U1=gc.U1_ERRHI_MIN, D2=gc.D2_ERRHI_MAX)
        if res is None:
            continue
        print_asset(res)
        out["divergence_alt"][asset] = dict(strategy=res["strat"], buy_hold=res["bh"],
                                            n_trades=res["n_trades"])

    RESULTS_JSON.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved results → {RESULTS_JSON}")


if __name__ == "__main__":
    main()
