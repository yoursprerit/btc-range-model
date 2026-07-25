#!/usr/bin/env python3
"""
Trailing-stop exit evaluation for MSTR / MSTU (offline, reproducible).

Question: in a strong uptrend the rally is cut short by the D3 BTC-canary exit.
Can we let winners run using a trailing stop on the MSTR/MSTU price itself (in
BULL regime, replacing the D3 exit) while keeping the existing fixed stop
(-3% MSTR / -7% MSTU) for downside containment?

This script:
  * builds CT predictions fully offline from data/backtest/raw_features_daily.csv,
    applying the SAME feature engineering AND the direction_head classifier the
    live app uses in _build_backtest_preds (so U1/D2/D3 fire on the live bars).
  * reuses compute_signals() from backtest_stop_loss_reentry (BTC-derived TF2+V,
    live "bull_regime" XOR entry gate).
  * mirrors the LIVE app backtest loop (same-bar execution, SL5 regime-adaptive
    re-entry, fixed close-triggered stop) as BASELINE.
  * adds a TRAIL-IN-BULL variant that, in BULL regime, replaces the D3 signal
    exit with a price trailing stop on the execution asset; the fixed stop stays
    active at all times (downside containment unchanged).
  * runs Bull / Bear / Full periods for MSTR and MSTU and prints a comparison.

FINDINGS (this dataset, run 2026-07):
  * The mechanism works on the flagship trade: MSTR Oct-2024 rally exits at
    +68.6% under D3 vs +104.5% under a 15% trail (holds 7 more days).
  * At a well-chosen width it lifts BULL and FULL return for MSTR
    (bull +62.7% -> +97.3% @15%; full +89.9% -> +118.7% @15%).
  * BUT it hurts the BEAR period in every configuration, deepens max drawdown
    (badly for the 2x MSTU), is very sensitive to trail width (too-tight whipsaws
    below baseline everywhere), rides essentially ONE in-sample rally, and for
    MSTU is a wash-to-negative on total return. Treat as exploratory, not a
    validated upgrade.

Run:  python backtest_trailing_stop.py   (needs scikit-learn==1.8.0 to unpickle
      the CT artefact; plus numpy / pandas / joblib)
"""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

# Reuse the research module's model load + signal/feature logic (no network at import).
import backtest_stop_loss_reentry as R

DATA = _ROOT / "data" / "backtest"
INITIAL_CAPITAL = 100_000.0
WARMUP = 35

PERIODS = {
    "Bull (Jun24->Jun25)": ("2024-06-05", "2025-06-14", "2024-03-01"),
    "Bear (Jun25->May26)": ("2025-06-01", "2026-05-31", "2025-03-01"),
    "Full (Jun24->May26)": ("2024-06-01", "2026-05-31", "2024-03-01"),
}

# ─────────────────────────────────────────────────────────────────────────────
# Offline data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_raw_features() -> pd.DataFrame:
    df = pd.read_csv(DATA / "raw_features_daily.csv", index_col=0, parse_dates=True)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    return df.sort_index()


def build_preds_offline(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce R.build_ct_preds feature engineering, but take cb_premium* from CSV."""
    AD = R.AD
    c, h, l_, v = df["btc_close"], df["btc_high"], df["btc_low"], df["btc_volume"]
    ret = np.log(c).diff()
    f = pd.DataFrame(index=df.index)
    for k in [1, 3, 5, 7, 14, 30]: f[f"ret_{k}"] = ret.rolling(k).sum()
    for k in [5, 10, 20, 30]:      f[f"vol_{k}"] = ret.rolling(k).std()
    prev_c = c.shift(1)
    tr = pd.concat([(h - l_), (h - prev_c).abs(), (l_ - prev_c).abs()], axis=1).max(axis=1)
    for k in [7, 14, 30]: f[f"atr_{k}"] = tr.rolling(k).mean() / c
    f["range_today"] = (h - l_) / c
    f["range_ma7"]   = ((h - l_) / c).rolling(7).mean()
    f["range_ma30"]  = ((h - l_) / c).rolling(30).mean()
    f["range_std30"] = ((h - l_) / c).rolling(30).std()
    g  = c.diff().clip(lower=0).rolling(14).mean()
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
    dow = df.index.dayofweek
    for i in range(6): f[f"dow_{i}"] = (dow == i).astype(float)
    for nm in ["spx", "ndx", "vix", "gold", "dxy", "tnx", "eth"]:
        col = f"{nm}_close"
        if col not in df.columns: continue
        s = df[col]; lr = np.log(s).diff()
        for k in [1, 5, 20]: f[f"{nm}_ret_{k}"] = lr.rolling(k).sum()
        f[f"{nm}_vol_20"] = lr.rolling(20).std()
    for corr_nm, corr_col in [("spx", "spx_close"), ("ndx", "ndx_close"),
                              ("gold", "gold_close"), ("dxy", "dxy_close")]:
        if corr_col in df.columns:
            f[f"btc_{corr_nm}_corr_30"] = ret.rolling(30).corr(np.log(df[corr_col]).diff())
    for col in [x for x in df.columns if x.startswith("oc_")]:
        s = df[col].astype(float); sl = np.log(s.replace(0, np.nan))
        f[f"{col}_d1"] = sl.diff(1); f[f"{col}_d7"] = sl.diff(7)
        f[f"{col}_z30"] = (sl - sl.rolling(30).mean()) / sl.rolling(30).std()
    nh, nl_ = h.shift(-1), l_.shift(-1)
    y_hi = (nh - c) / c; y_lo = (c - nl_) / c
    f["y_hi_ema3"] = y_hi.shift(1).ewm(span=3, adjust=False).mean()
    f["y_lo_ema3"] = y_lo.shift(1).ewm(span=3, adjust=False).mean()
    f["y_hi_ema7"] = y_hi.shift(1).ewm(span=7, adjust=False).mean()
    f["y_lo_ema7"] = y_lo.shift(1).ewm(span=7, adjust=False).mean()
    p3h = h.shift(1).rolling(3).max(); p3l = l_.shift(1).rolling(3).min()
    f["above_3d_high"] = (c > p3h).astype(float)
    f["below_3d_low"]  = (c < p3l).astype(float)
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
    # Coinbase premium — take precomputed columns from the CSV (offline)
    for cbcol in ["cb_premium", "cb_premium_ma3", "cb_premium_z7"]:
        f[cbcol] = df[cbcol] if cbcol in df.columns else 0.0

    fc = AD["feat_cols"]
    f = f.replace([np.inf, -np.inf], np.nan)
    for col in fc:
        if col not in f.columns: f[col] = np.nan
    for cbcol in ["cb_premium", "cb_premium_ma3", "cb_premium_z7"]:
        if cbcol in f.columns: f[cbcol] = f[cbcol].fillna(0.0)
    F = f[fc].dropna()
    if AD.get("ensemble") and AD.get("constituents"):
        yhi = np.mean([con["m_hi"].predict(F) for con in AD["constituents"]], axis=0)
        ylo = np.mean([con["m_lo"].predict(F) for con in AD["constituents"]], axis=0)
        if AD.get("blended") and float(AD.get("alpha", 1.0)) < 1.0:
            a = float(AD["alpha"])
            yhi = a * yhi + (1 - a) * float(AD.get("mu_hi", 0))
            ylo = a * ylo + (1 - a) * float(AD.get("mu_lo", 0))
    else:
        yhi = AD["hi_model"].predict(F); ylo = AD["lo_model"].predict(F)
    # Direction head — identical to app _build_backtest_preds (matches LIVE signals)
    _dh = AD.get("direction_head")
    if _dh is not None and _dh.get("classifier") is not None:
        try:
            _clf = _dh["classifier"]
            _beta_base = float(_dh.get("beta", 1.0))
            _reduction = float(_dh.get("beta_trend_reduction", 0.0))
            _trend_sat = float(_dh.get("trend_saturation", 0.05))
            _trend_feat = _dh.get("trend_feature", "ret_5")
            _d_bull = float(_dh.get("d_bull_mean", 0.0))
            _d_bear = float(_dh.get("d_bear_mean", 0.0))
            _p_bull = _clf.predict_proba(F)[:, 1]
            _t_vals = F[_trend_feat].fillna(0).values if _trend_feat in F.columns else np.zeros(len(F))
            _t_str = np.minimum(np.abs(_t_vals) / _trend_sat, 1.0)
            _b_eff = np.clip(_beta_base * (1.0 - _reduction * _t_str), 0.0, 1.0)
            _m_pred = (yhi + ylo) / 2.0
            _d_pred = (yhi - ylo) / 2.0
            _d_dir = _p_bull * _d_bull + (1 - _p_bull) * _d_bear
            _d_blnd = _b_eff * _d_pred + (1 - _b_eff) * _d_dir
            yhi = _m_pred + _d_blnd
            ylo = _m_pred - _d_blnd
        except Exception:
            pass
    c_vals = c.reindex(F.index).values
    ph = c_vals * (1 + np.clip(yhi, 0, None))
    pl = c_vals * (1 - np.clip(ylo, 0, None))
    idx = np.asarray(F.index, dtype="datetime64[ns]")
    nd = np.empty(len(F), dtype="datetime64[ns]")
    nd[:-1] = idx[1:]; nd[-1] = idx[-1] + np.timedelta64(1, "D")
    res = pd.DataFrame({"close_asof": c_vals, "pred_high": ph, "pred_low": pl},
                       index=pd.DatetimeIndex(nd, name="target_date"))
    return res[~res.index.duplicated(keep="last")]


def load_asset(name: str) -> pd.Series:
    fn = {"MSTR": "mstr_daily.csv", "MSTU": "mstu_daily.csv",
          "ETHA": "etha_daily.csv"}[name]
    df = pd.read_csv(DATA / fn, parse_dates=["Date"]).set_index("Date").sort_index()
    return df["close"].astype(float)


# ─────────────────────────────────────────────────────────────────────────────
# App-faithful same-bar backtest engine (mirrors run_mstr/mstu_backtest)
#   exit_mode = "baseline"    -> exit d3 or (d2 & ~bull); fixed stop; SL5 reentry
#   exit_mode = "trail_bull"  -> in BULL, replace D3 with price trailing stop;
#                                in BEAR/neutral keep d2/d3; fixed stop always on
# ─────────────────────────────────────────────────────────────────────────────
def run_bt(dates, asset_px, sigs, fixed_pct, exit_mode="baseline",
           trail_pct=0.15, trail_from_entry_floor=True, bt_start=None,
           cap=INITIAL_CAPITAL):
    d2 = sigs["d2"]; d3 = sigs["d3"]; bull = sigs["bull_regime"]
    tf_entry = sigs["tf2_entry"]; above = sigs["above_ma30"]
    c7 = sigs["clean_7d"]; vr = sigs["v_recent"]
    N = len(dates)
    _bt0 = max(WARMUP, int(pd.DatetimeIndex(dates).searchsorted(bt_start))) if bt_start is not None else WARMUP

    nav = cap; pos = "CASH"; qty = 0.0
    e_price = e_nav = e_date = e_trig = None
    stop_px = 0.0; hwm = 0.0
    from_sl = False; bars_since_sl = 0
    trades = []; nav_arr = np.full(N, np.nan)

    for i in range(N):
        price = asset_px[i]
        if i < _bt0:
            nav_arr[i] = cap; continue
        if not np.isfinite(price) or price <= 0:
            nav_arr[i] = qty * asset_px[i - 1] if pos == "LONG" and i > 0 else nav
            continue
        if pos == "LONG":
            cur = qty * price
            hwm = max(hwm, price)
            # fixed stop always active (downside containment) — unchanged from live
            hard_stop = e_price * (1.0 - fixed_pct)
            # effective stop for the trailing exit (bull only)
            trail_stop = hwm * (1.0 - trail_pct)
            if trail_from_entry_floor:
                eff_trail = max(trail_stop, hard_stop)
            else:
                eff_trail = trail_stop

            fired_hard = price <= hard_stop
            if exit_mode == "trail_bull" and bull[i]:
                # In BULL: exit on trailing-stop breach (replaces D3). Bear D2 n/a here.
                sig_exit = price <= eff_trail
                exit_lbl = "TRAIL" if (price <= eff_trail and not fired_hard) else "SL-fixed"
            else:
                sig_exit = bool(d3[i] or (d2[i] and not bull[i]))
                exit_lbl = "D3" if d3[i] else "D2 (bear)"

            if fired_hard:
                nav = qty * price
                trades.append(dict(entry_date=e_date, entry_price=e_price, entry_nav=e_nav,
                                   entry_trigger=e_trig, exit_date=dates[i], exit_price=price,
                                   exit_nav=nav, pnl_pct=(price / e_price - 1) * 100,
                                   pnl_abs=nav - e_nav, exit_signal=f"SL-fixed-{fixed_pct*100:.0f}%",
                                   duration_days=(dates[i] - e_date).days, was_sl=True))
                pos = "CASH"; qty = 0.0; stop_px = 0.0
                from_sl = True; bars_since_sl = 0; hwm = 0.0
            elif sig_exit:
                nav = cur
                trades.append(dict(entry_date=e_date, entry_price=e_price, entry_nav=e_nav,
                                   entry_trigger=e_trig, exit_date=dates[i], exit_price=price,
                                   exit_nav=nav, pnl_pct=(price / e_price - 1) * 100,
                                   pnl_abs=nav - e_nav, exit_signal=exit_lbl,
                                   duration_days=(dates[i] - e_date).days, was_sl=False))
                pos = "CASH"; qty = 0.0; stop_px = 0.0
                from_sl = False; hwm = 0.0
            else:
                nav = cur
        else:
            if from_sl:
                bars_since_sl += 1
            _exit_at_i = d3[i] or (d2[i] and not bull[i])
            _reentry_ok = (not from_sl) or bool(bull[i]) or bars_since_sl >= 10
            if tf_entry[i] and _reentry_ok and (from_sl or not _exit_at_i):
                qty = nav / price; e_price = price; e_date = dates[i]
                e_nav = nav; pos = "LONG"; hwm = price
                from_sl = False; bars_since_sl = 0
                e_trig = "U1+V-rev" if vr[i] else ("U1+^MA30" if above[i] else "U1+clean7d")
        nav_arr[i] = qty * price if pos == "LONG" else nav

    if pos == "LONG" and np.isfinite(asset_px[N - 1]):
        nav_arr[N - 1] = qty * asset_px[N - 1]

    nav_s = pd.Series(nav_arr[_bt0:], index=dates[_bt0:]).ffill()
    bh_s = pd.Series(cap * asset_px[_bt0:] / asset_px[_bt0], index=dates[_bt0:])
    return dict(trades=trades, nav=nav_s, bh=bh_s, open_pos=(pos == "LONG"))


def stats(res):
    nav = res["nav"]; bh = res["bh"]; tr = res["trades"]
    fn = float(nav.iloc[-1]); fb = float(bh.iloc[-1])
    dr = nav.pct_change().fillna(0); rfd = 1.045 ** (1 / 252) - 1
    exc = dr - rfd
    sharpe = float(exc.mean() / exc.std() * np.sqrt(252)) if exc.std() > 0 else 0.0
    rm = nav.cummax(); max_dd = float(((nav - rm) / rm * 100).min())
    wins = [t for t in tr if t["pnl_pct"] > 0]
    days_in = sum(t["duration_days"] for t in tr)
    tot = max(1, (nav.index[-1] - nav.index[0]).days)
    return dict(ret=(fn / INITIAL_CAPITAL - 1) * 100, bh=(fb / INITIAL_CAPITAL - 1) * 100,
                final=fn, alpha=fn - fb, sharpe=sharpe, max_dd=max_dd,
                n=len(tr), n_sl=len([t for t in tr if t.get("was_sl")]),
                wr=100 * len(wins) / len(tr) if tr else 0.0,
                tim=100 * days_in / tot, trades=tr)


# ─────────────────────────────────────────────────────────────────────────────
def pure_regime_entry(sigs):
    """Live app default entry gate: u1 & (bull | (clean7d & ~above_ma30) | v_recent)."""
    u1 = sigs["u1"]; bull = sigs["bull_regime"]; c7 = sigs["clean_7d"]
    above = sigs["above_ma30"]; vr = sigs["v_recent"]
    return u1 & (bull | (c7 & ~above) | vr)


def prep(df_raw, preds, start_iso, end_iso):
    sd, ed = pd.Timestamp(start_iso), pd.Timestamp(end_iso)
    p = preds.loc[(preds.index >= sd - pd.Timedelta(days=60)) & (preds.index <= ed)].copy()
    p["actual_high"] = df_raw["btc_high"].reindex(p.index).values
    p["actual_low"]  = df_raw["btc_low"].reindex(p.index).values
    p["actual_close"] = df_raw["btc_close"].reindex(p.index).values
    return p.dropna(subset=["actual_high", "actual_low", "actual_close"]).reset_index()


def main():
    print("Loading offline data ...")
    df_raw = load_raw_features()
    preds = build_preds_offline(df_raw)
    print(f"  preds: {len(preds)} rows  {preds.index.min().date()} -> {preds.index.max().date()}")
    px = {"MSTR": load_asset("MSTR"), "MSTU": load_asset("MSTU")}
    fixed = {"MSTR": 0.03, "MSTU": 0.07}
    # trailing-width sweep per asset (MSTU is 2x -> wider)
    trail_grid = {"MSTR": [0.08, 0.10, 0.12, 0.15, 0.20],
                  "MSTU": [0.15, 0.20, 0.25, 0.30, 0.35]}

    all_rows = {}
    for pname, (s, e, fs) in PERIODS.items():
        comp = prep(df_raw, preds, s, e)
        if len(comp) < WARMUP + 3:
            print(f"[skip] {pname}"); continue
        dates = pd.DatetimeIndex(comp["target_date"])
        sigs = R.compute_signals(comp)
        sd = pd.Timestamp(s)
        _bt0 = max(WARMUP, int(dates.searchsorted(sd)))
        for asset in ["MSTR", "MSTU"]:
            a = px[asset].reindex(dates).ffill().bfill().values.astype(float)
            if asset == "MSTU":
                fv = int(np.argmax(np.isfinite(a) & (a > 0)))
                if fv > _bt0 + 10:
                    continue
            base = stats(run_bt(dates, a, sigs, fixed[asset], "baseline", bt_start=sd))
            all_rows[(pname, asset, "baseline")] = base
            for tp in trail_grid[asset]:
                r = stats(run_bt(dates, a, sigs, fixed[asset], "trail_bull",
                                 trail_pct=tp, bt_start=sd))
                all_rows[(pname, asset, f"trail{int(tp*100)}")] = r

    # ── print ────────────────────────────────────────────────────────────────
    hdr = f"{'variant':<14}{'Ret%':>9}{'B&H%':>9}{'Alpha$k':>9}{'MaxDD%':>9}{'Shrp':>7}{'Tr':>4}{'SL':>4}{'WR%':>6}{'TiM%':>6}"
    for asset in ["MSTR", "MSTU"]:
        print("\n" + "=" * 84)
        print(f"  {asset}  (fixed stop -{fixed[asset]*100:.0f}%, close-triggered — DOWNSIDE UNCHANGED)")
        print("=" * 84)
        for pname in PERIODS:
            print(f"\n  {pname}")
            print("  " + hdr)
            print("  " + "-" * 82)
            keys = [("baseline", "baseline (live)")] + \
                   [(f"trail{int(tp*100)}", f"trail-in-bull {int(tp*100)}%") for tp in trail_grid[asset]]
            base = all_rows.get((pname, asset, "baseline"))
            for vk, vlabel in keys:
                m = all_rows.get((pname, asset, vk))
                if m is None:
                    continue
                mark = ""
                if base is not None and vk != "baseline":
                    mark = " ▲" if m["final"] > base["final"] else " ▼"
                print(f"  {vlabel:<14}{m['ret']:>9.1f}{m['bh']:>9.1f}{m['alpha']/1e3:>9.1f}"
                      f"{m['max_dd']:>9.1f}{m['sharpe']:>7.2f}{m['n']:>4}{m['n_sl']:>4}"
                      f"{m['wr']:>6.0f}{m['tim']:>6.0f}{mark}")

    # ── bull-period trade-level comparison (where the effect lives) ───────────
    print("\n" + "=" * 84)
    print("  BULL PERIOD trade detail — baseline vs a representative trail width")
    print("=" * 84)
    rep = {"MSTR": 0.15, "MSTU": 0.25}
    for asset in ["MSTR", "MSTU"]:
        s, e, fs = PERIODS["Bull (Jun24->Jun25)"]
        comp = prep(df_raw, preds, s, e); dates = pd.DatetimeIndex(comp["target_date"])
        sigs = R.compute_signals(comp); sd = pd.Timestamp(s)
        a = px[asset].reindex(dates).ffill().bfill().values.astype(float)
        for mode, tp, lbl in [("baseline", 0.0, "BASELINE (D3 exit in bull)"),
                              ("trail_bull", rep[asset], f"TRAIL-IN-BULL {int(rep[asset]*100)}%")]:
            r = run_bt(dates, a, sigs, fixed[asset], mode, trail_pct=tp, bt_start=sd)
            m = stats(r)
            print(f"\n  {asset} — {lbl}   ret {m['ret']:+.1f}%  ({m['n']} trades, WR {m['wr']:.0f}%)")
            for t in r["trades"]:
                print(f"     {pd.Timestamp(t['entry_date']).date()} -> {pd.Timestamp(t['exit_date']).date()}"
                      f"  {t['pnl_pct']:+7.1f}%  {t['exit_signal']:<12} ({t['duration_days']}d)")


if __name__ == "__main__":
    main()
