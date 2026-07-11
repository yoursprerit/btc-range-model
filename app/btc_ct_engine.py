"""BTC / MSTR / MSTU via the BTC app's ACTUAL signal engine.

The BTC app (``app/btc_hourly_app.py``) is not importable (it runs Streamlit at
module top), but its H/L prediction + Pure-Regime backtest are byte-identically
reproduced by the importable research module ``backtest_trailing_stop`` (which
loads the trained CT ensemble ``models/inference_assets_ct.joblib`` and builds
the same 116-feature matrix — price + macro + Bitcoin on-chain + Coinbase
premium).  This module wraps it with the app's *live* per-asset entry gate and
thresholds (U1 err_hi>+1.3% + ≥2 high-breaks, D2 exit<−1.3%, MA30 regime gate,
regime-adaptive exit, per-asset stops, SL re-entry) so the Overall app's BTC
sleeve matches the BTC app in both live signal and back-test.  2026-07 retune:
BTC trades the Pure Regime gate; MSTR/MSTU trade the Standard MA (above-MA30)
gate — the most profitable and most stable gate for the two equities.

Verified: reproduces the BTC app's headline BTC +89% (Pure Regime) /
MSTR +148% / MSTU +419% (Standard MA), full period Jun 2024 → May 2026.

Caveat: the CT feature data (``data/backtest/raw_features_daily.csv``) spans
~2023-11 → the last pull, so BTC-sleeve returns begin ~2024 (not 2021).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

_APP_DIR = Path(__file__).resolve().parent
_REPO = _APP_DIR.parent
for _p in (str(_REPO), str(_APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:                                            # CT model pickle needs this shim
    import sklearn._loss._loss as _sl
    sys.modules.setdefault("_loss", _sl)
except Exception:
    pass

import numpy as np
import pandas as pd

import backtest_trailing_stop as T             # loads the CT model at import

# ── self-heal the CT feature data so the BTC sleeve updates daily ────────────
# BTC/MSTR/MSTU signals are built from a committed daily-features CSV
# (data/backtest/raw_features_daily.csv).  In a long-running deployment nothing
# refreshes it, so those three freeze at the CSV's last date while every other
# instrument fetches live.  When the CSV falls behind the latest completed CT
# bar we re-pull it (scripts/pull_backtest_data.py) — at most once every few
# hours, and fully non-fatal: any failure just keeps the existing CSV.
_PULL_SCRIPT = _REPO / "scripts" / "pull_backtest_data.py"
_FEATURES_CSV = T.DATA / "raw_features_daily.csv"
_PULL_GUARD = T.DATA / ".last_feature_pull"
_PULL_MIN_GAP = pd.Timedelta(hours=3)          # don't re-attempt more often than this


def _features_last_bar() -> "pd.Timestamp | None":
    try:
        idx = pd.read_csv(_FEATURES_CSV, index_col=0, parse_dates=True).index
        return pd.Timestamp(idx[-1]).tz_localize(None).normalize()
    except Exception:
        return None


def _ensure_fresh_features() -> None:
    try:
        if not _FEATURES_CSV.exists() or not _PULL_SCRIPT.exists():
            return
        now = pd.Timestamp.utcnow().tz_localize(None)
        today = now.normalize()
        # the CT "day" is anchored at 12:00 UTC; the prior day's bar is complete
        # once we're past that anchor, so that's the freshest bar we can expect.
        target = today if now.hour >= 12 else (today - pd.Timedelta(days=1))
        last = _features_last_bar()
        if last is not None and last >= target:
            return                                 # already current — nothing to do
        if _PULL_GUARD.exists():                    # rate-limit re-attempts
            try:
                if (now - pd.Timestamp(_PULL_GUARD.read_text().strip())) < _PULL_MIN_GAP:
                    return
            except Exception:
                pass
        _PULL_GUARD.write_text(now.isoformat())     # stamp before running
        import subprocess
        subprocess.run([sys.executable, str(_PULL_SCRIPT)], cwd=str(_REPO),
                       timeout=360, check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


# live thresholds (btc_hourly_app.py) and per-asset entry gate + stop.
# 2026-07 retune: BTC uses the Pure Regime gate; MSTR/MSTU the Standard MA
# (above-MA30) gate — the most profitable and most stable gate for the two
# equities on the current data (mirrors MSTR_STRATEGY_GATE / MSTU_STRATEGY_GATE
# in btc_hourly_app.py).
U1_ERRHI_MIN = 1.3
D2_ERRHI_MAX = -1.3
STOP_PCT = {"BTC": None, "MSTR": 0.03, "MSTU": 0.03}   # BTC: no fixed stop
GATE_BY_ASSET = {"BTC": "pure_regime", "MSTR": "above_ma30", "MSTU": "above_ma30"}
_META = {
    "BTC":  dict(name="Bitcoin",       kind="core", stop=0.0),
    "MSTR": dict(name="MicroStrategy",  kind="beta", stop=0.03),
    "MSTU": dict(name="2× MSTR",        kind="lev",  stop=0.03),
}
ACCENT = "#f7931a"
EMOJI = "₿"


# ── signals: both live entry gates (U1>1.3 / D2<-1.3) ────────────────────
#   tf2_entry_pure = Pure Regime (OR gate)   — used for BTC
#   tf2_entry_ma   = Standard MA (XOR gate)  — used for MSTR/MSTU
def compute_sigs_pure(comp: pd.DataFrame) -> dict:
    """Byte-identical to the BTC app's run_*_backtest signal arrays, exposing
    both live entry gates (Pure Regime + Standard MA) and current thresholds."""
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
    u1 = (ehma3 > U1_ERRHI_MIN) & (hb3 >= 2)
    d1 = (lb3 >= 2) & (elma3 > 0.5)
    d2 = ehma3 < D2_ERRHI_MAX
    d3 = np.zeros(N, bool)
    for i in range(1, N):
        cc = 0
        for k in range(i - 1, -1, -1):
            if hi[k]:
                cc += 1
            else:
                break
        if cc >= 3 and lo[i]:
            d3[i] = True
    ma30 = np.array([c[max(0, i - min(30, i + 1) + 1):i + 1].mean() for i in range(N)])
    above = c > ma30
    slope = np.zeros(N, bool)
    for i in range(5, N):
        slope[i] = ma30[i] > ma30[i - 5]
    bull = above & slope
    clean = np.array([not bool((d1[max(0, i - 7):i] | d2[max(0, i - 7):i]).any())
                      for i in range(N)])
    roll = np.array([err_hi[max(0, i - 29):i + 1].mean() for i in range(N)])
    dn = np.zeros(N)
    for i in range(N):
        nrm = max(abs(roll[i]), 0.01)
        dn[i] = ((-ehma3[i] / nrm) * .30 + (lb3[i] / 3.) * .30
                 + (elma3[i] / max(abs(elma3[i]), .10)) * .20 + float(lo[i]) * .20)
    vbar = (dn > 0.8) & (err_lo > 3.0)
    v = np.array([bool(vbar[max(0, i - 2):i + 1].any()) for i in range(N)])
    tf_pure = u1 & (bull | (clean & ~above) | v)        # PURE-REGIME (OR gate)  — BTC
    tf_ma   = u1 & ((above ^ clean) | v)                # STANDARD MA (XOR gate) — MSTR/MSTU
    return dict(d1=d1, d2=d2, d3=d3, u1=u1, bull_regime=bull,
                tf2_entry=tf_pure, tf2_entry_pure=tf_pure, tf2_entry_ma=tf_ma,
                above_ma30=above, clean_7d=clean, v_recent=v,
                err_hi=err_hi, err_lo=err_lo, ehma3=ehma3, elma3=elma3,
                hb3=hb3, lb3=lb3)


# ── backtest loop — faithful copy of backtest_trailing_stop.run_bt, extended
#    to also return the per-bar position series and the open-position entry ──
def _run_bt(dates, asset_px, sigs, fixed_pct, bt_start):
    d2 = sigs["d2"]; d3 = sigs["d3"]; bull = sigs["bull_regime"]
    tf_entry = sigs["tf2_entry"]; above = sigs["above_ma30"]; vr = sigs["v_recent"]
    N = len(dates)
    WARMUP = T.WARMUP
    _bt0 = max(WARMUP, int(pd.DatetimeIndex(dates).searchsorted(bt_start))) \
        if bt_start is not None else WARMUP
    cap = T.INITIAL_CAPITAL
    nav = cap; pos = "CASH"; qty = 0.0
    e_price = e_nav = e_date = e_trig = None
    from_sl = False; bars_since_sl = 0; hwm = 0.0
    trades = []; nav_arr = np.full(N, np.nan); pos_arr = np.zeros(N)
    no_stop = (fixed_pct is None)
    fp = 10.0 if no_stop else float(fixed_pct)       # huge pct ⇒ hard stop never fires

    for i in range(N):
        price = asset_px[i]
        if i < _bt0:
            nav_arr[i] = cap; continue
        if not np.isfinite(price) or price <= 0:
            nav_arr[i] = qty * asset_px[i - 1] if pos == "LONG" and i > 0 else nav
            pos_arr[i] = 1.0 if pos == "LONG" else 0.0
            continue
        if pos == "LONG":
            cur = qty * price
            hwm = max(hwm, price)
            hard_stop = e_price * (1.0 - fp)
            fired_hard = (not no_stop) and price <= hard_stop
            sig_exit = bool(d3[i] or (d2[i] and not bull[i]))
            exit_lbl = "D3 exhaustion" if d3[i] else "D2 fade (bear)"
            if fired_hard:
                nav = qty * price
                trades.append(dict(entry_date=e_date, entry_price=e_price,
                                   exit_date=dates[i], exit_price=price,
                                   ret=price / e_price - 1, reason="stop −%.0f%%" % (fp * 100),
                                   duration_days=(dates[i] - e_date).days, was_sl=True))
                pos = "CASH"; qty = 0.0; from_sl = True; bars_since_sl = 0; hwm = 0.0
            elif sig_exit:
                nav = cur
                trades.append(dict(entry_date=e_date, entry_price=e_price,
                                   exit_date=dates[i], exit_price=price,
                                   ret=price / e_price - 1, reason=exit_lbl,
                                   duration_days=(dates[i] - e_date).days, was_sl=False))
                pos = "CASH"; qty = 0.0; from_sl = False; hwm = 0.0
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
                e_trig = ("U1 + V-reversal" if vr[i]
                          else "U1 + ↑MA30" if above[i] else "U1 + Clean-7d")
        nav_arr[i] = qty * price if pos == "LONG" else nav
        pos_arr[i] = 1.0 if pos == "LONG" else 0.0

    if pos == "LONG" and np.isfinite(asset_px[N - 1]):
        nav_arr[N - 1] = qty * asset_px[N - 1]
    idx = pd.DatetimeIndex(dates[_bt0:])
    nav_s = pd.Series(nav_arr[_bt0:], index=idx).ffill()
    bh_s = pd.Series(cap * asset_px[_bt0:] / asset_px[_bt0], index=idx)
    pos_s = pd.Series(pos_arr[_bt0:], index=idx)
    open_entry = (dict(price=float(e_price), date=pd.Timestamp(e_date), trigger=e_trig)
                  if pos == "LONG" else None)
    return dict(trades=trades, nav=nav_s, bh=bh_s, pos=pos_s,
                open_pos=(pos == "LONG"), open_entry=open_entry)


def _load_prices(dates: pd.DatetimeIndex, comp: pd.DataFrame) -> dict:
    btc = comp["actual_close"].values.astype(float)
    mstr = T.load_asset("MSTR").reindex(dates).ffill().to_numpy(float)
    try:
        syn = pd.read_csv(T.DATA / "mstu_synthetic_daily.csv", parse_dates=["Date"])
        syn = syn.set_index("Date").sort_index()["close"].astype(float)
        mstu = syn.reindex(dates).ffill().to_numpy(float)
    except Exception:
        mstu = T.load_asset("MSTU").reindex(dates).ffill().to_numpy(float)
    return {"BTC": btc, "MSTR": mstr, "MSTU": mstu}


# ── result assembly (standard Overall-app result dicts) ──────────────────
_TRADING_DAYS = 252
CAP_BY_KIND = {"core": 0.30, "beta": 0.18, "lev": 0.10}
KIND_EMOJI = {"core": "", "beta": "⚡", "lev": "🔺"}


def _curve_metrics(equity: pd.Series) -> dict:
    eq = equity.to_numpy(float); idx = equity.index
    if len(eq) < 2:
        return dict(total_ret=0.0, cagr=0.0, mdd=0.0, sharpe=0.0, vol=0.0)
    total = eq[-1] / eq[0] - 1
    yrs = max((idx[-1] - idx[0]).days / 365.25, 1e-9)
    cagr = (eq[-1] / eq[0]) ** (1 / yrs) - 1
    peak = np.maximum.accumulate(eq)
    mdd = float(np.min(eq / peak - 1))
    rets = np.diff(eq) / eq[:-1]
    vol = float(np.std(rets) * np.sqrt(_TRADING_DAYS))
    sharpe = float(np.mean(rets) / (np.std(rets) + 1e-12) * np.sqrt(_TRADING_DAYS))
    return dict(total_ret=float(total), cagr=float(cagr), mdd=mdd, sharpe=sharpe, vol=vol)


def _decision(sigs, i, in_pos):
    d2 = bool(sigs["d2"][i]); d3 = bool(sigs["d3"][i]); bull = bool(sigs["bull_regime"][i])
    u1 = bool(sigs["u1"][i]); entry = bool(sigs["tf2_entry"][i])
    exit_sig = d3 or (d2 and not bull)                  # regime-adaptive (app logic)
    why = "D3 exhaustion" if d3 else "D2 fade (bear)"
    if in_pos:
        if exit_sig:
            return dict(state="EXIT", label=f"EXIT — {why}", ico="🔴", tone="exit")
        return dict(state="HOLD", label="LONG — HOLDING", ico="🟢", tone="hold")
    if exit_sig:
        return dict(state="AVOID", label=f"STAND ASIDE — EXIT ACTIVE ({why})",
                    ico="🟠", tone="watch")
    if entry:
        return dict(state="ENTRY", label="ENTER — STRATEGY BUY", ico="🟢", tone="buy")
    if u1:
        return dict(state="WATCH", label="WATCH — U1 (GATE PENDING)", ico="🟡", tone="watch")
    return dict(state="FLAT", label="FLAT — NO SIGNAL", ico="⬜", tone="flat")


def _alert(sigs, i, in_pos):
    if sigs["d3"][i] or (sigs["d2"][i] and not sigs["bull_regime"][i]):
        return "EXIT_DN"
    if sigs["tf2_entry"][i]:
        return "STRATEGY_BUY"
    if sigs["u1"][i]:
        return "WATCH_UP"
    return "NEUTRAL"


def run_btc_ct(start: str = "2024-01-01") -> list[dict]:
    """Run BTC/MSTR/MSTU through the BTC app's CT engine and return standard
    Overall-app result dicts (one per instrument)."""
    _ensure_fresh_features()          # refresh the feature CSV if it's behind
    rf = T.load_raw_features()
    preds = T.build_preds_offline(rf)
    end = str(rf.index[-1].date())
    comp = T.prep(rf, preds, start, end)
    dates = pd.DatetimeIndex(comp["target_date"])
    if len(dates) < 30:
        return []
    sigs = compute_sigs_pure(comp)
    px = _load_prices(dates, comp)
    sd = pd.Timestamp(start)
    last = len(dates) - 1
    bull_now = bool(sigs["bull_regime"][last])
    # cross-asset momentum (BTC distance above its 50-day SMA)
    bc = comp["actual_close"].to_numpy(float)
    ref = bc[max(0, len(bc) - 50):].mean()
    mom = (bc[-1] / ref - 1) if ref else 0.0
    as_of = pd.Timestamp(dates[last])

    out = []
    for key in ("BTC", "MSTR", "MSTU"):
        meta = _META[key]
        # per-asset entry gate: BTC → Pure Regime, MSTR/MSTU → Standard MA
        sigs_k = dict(sigs)
        sigs_k["tf2_entry"] = (sigs["tf2_entry_ma"]
                               if GATE_BY_ASSET[key] == "above_ma30"
                               else sigs["tf2_entry_pure"])
        bt = _run_bt(dates, px[key], sigs_k, STOP_PCT[key], sd)
        nav = bt["nav"]; bh = bt["bh"]
        ret = nav.pct_change().fillna(0.0).rename(key)
        pos_series = bt["pos"].rename(key)
        # r dict shaped like backtest_ticker.run_strategy for overall_core reuse
        r = dict(bh=bh.to_numpy(float), dates=list(nav.index), trades=bt["trades"],
                 in_pos_now=bt["open_pos"], strat=nav.to_numpy(float))
        last_px = float(px[key][last])
        prev_px = float(px[key][last - 1]) if last >= 1 else last_px
        dchg = (last_px / prev_px - 1) * 100 if prev_px else 0.0
        dec = _decision(sigs_k, last, bt["open_pos"])
        m = _curve_metrics(nav); bhm = _curve_metrics(bh)
        wr = 100.0 * np.mean([t["ret"] > 0 for t in bt["trades"]]) if bt["trades"] else 0.0
        pos = dict(in_pos=bt["open_pos"], entry_px=None, entry_date=None,
                   upnl=None, stop_px=None, days=None, dist_stop=None)
        if bt["open_pos"] and bt["open_entry"]:
            e_px = bt["open_entry"]["price"]; e_dt = bt["open_entry"]["date"]
            pos.update(entry_px=e_px, entry_date=e_dt,
                       upnl=(last_px / e_px - 1) * 100,
                       days=int((as_of - e_dt).days))
            if meta["stop"]:
                pos["stop_px"] = e_px * (1 - meta["stop"])
                pos["dist_stop"] = (last_px / pos["stop_px"] - 1) * 100
        last_trade = None
        if bt["trades"]:
            t = bt["trades"][-1]
            last_trade = dict(entry_date=t["entry_date"], exit_date=t["exit_date"],
                              entry_px=t["entry_price"], exit_px=t["exit_price"],
                              ret=t["ret"], reason=t["reason"])
        out.append(dict(
            key=key, parent="BTC", name=meta["name"], kind=meta["kind"],
            emoji=EMOJI, kemoji=KIND_EMOJI[meta["kind"]], accent=ACCENT,
            cap=CAP_BY_KIND[meta["kind"]],
            last_close=last_px, dchg=dchg, ma_val=None,
            sentiment=np.nan, decision=dec, alert=_alert(sigs_k, last, bt["open_pos"]),
            bull_regime=bull_now, pos=pos, last_trade=last_trade, mom=mom,
            metrics=m, bh_metrics=bhm, win_rate=wr, n_trades=len(bt["trades"]),
            ret=ret, pos_series=pos_series, strat=nav.to_numpy(float),
            dates=pd.Series(nav.index), r=r, as_of=as_of,
            mode="ct-divergence", ma_window=30,
            stop=meta["stop"], engine="ct",
        ))
    return out

