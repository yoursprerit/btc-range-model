"""GLDM / GDX / UGL / NUGT via the Gold app's ACTUAL engine.

The Gold app (``app/gldm_hourly_app.py``) trades the MIDDLE-PATH hybrid
(``gc.ENGINE_BY_ASSET``): a dual-MA 25/100 crossover on the GLDM close for the
smooth trenders GLDM & UGL (−3% stop) and the Divergence Pure-Regime for the
miners GDX & NUGT (−3% / −5%).  This module wraps the exact same
``backtest_gldm.run_asset_sim`` dispatch so the Overall app's gold sleeves
match the Gold app bar-for-bar.
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

import numpy as np
import pandas as pd

import gldm_core as gc
import backtest_gldm as bg

ACCENT = "#b8860b"
EMOJI = "🥇"
OOS = "2021-01-01"
_TRADING_DAYS = 252
CAP_BY_KIND = {"core": 0.30, "beta": 0.18, "lev": 0.10}
KIND_EMOJI = {"core": "", "beta": "⚡", "lev": "🔺"}
# parent = the individual app that surfaces the sleeve (the middle-path split):
#   GLDM → 🥇 Gold Trend (dual-MA)   GDXM → ⛏️ Gold Miners (divergence)
_META = {
    "GLDM": dict(name="Gold (GLDM)",  kind="core", col="gldm_close",
                 parent="GLDM", emoji="🥇"),
    "GDX":  dict(name="Gold Miners",  kind="beta", col="gdx_close",
                 parent="GDXM", emoji="⛏️"),
    "UGL":  dict(name="2× Gold",      kind="lev",  col="ugl_close",
                 parent="GLDM", emoji="🥇"),
    "NUGT": dict(name="2× Gold Miners", kind="lev", col="nugt_close",
                 parent="GDXM", emoji="⛏️"),
}


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


def _trend_decision(long_now: bool, in_pos: bool):
    """Dual-MA sleeve decision (GLDM & UGL): long/flat off the 25/100 cross.

    The cross decides at the close and acts on the NEXT bar, so an in-position
    EXIT here means "still open today, closes next bar" — the flag lets the
    Overall action table show its ⚠️ exits-next-bar banner even when the pill
    is pinned to a morning book published before the signal committed."""
    if in_pos:
        if long_now:
            return dict(state="HOLD", label="LONG — HOLDING", ico="🟢", tone="hold")
        return dict(state="EXIT", label="EXIT — MA CROSS-DOWN", ico="🔴", tone="exit",
                    exits_next_bar=True)
    if long_now:
        return dict(state="ENTRY", label="ENTER — DUAL-MA CROSS-UP", ico="🟢", tone="buy")
    return dict(state="FLAT", label="FLAT — BELOW TREND", ico="⬜", tone="flat")


def _decision(sigs, in_pos):
    """Divergence net decision (exit overrides entry) — same precedence as the
    Gold app's net_signal and the backtest simulate."""
    if not sigs:
        return dict(state="FLAT", label="NO DATA", ico="⬜", tone="flat")
    d1 = bool(sigs.get("d1_triggered")); d2 = bool(sigs.get("d2_triggered"))
    d3 = bool(sigs.get("d3_triggered"))
    exit_sig = d2 or d3                                  # gold: no D1 exit
    entry = bool(sigs.get("entry_triggered"))
    why = "D3 exhaustion" if d3 else "D2 momentum fade"
    if in_pos:
        if exit_sig:
            # decided at the close, executed next bar (backtest_gldm lags the
            # signals one bar) — flag it so the Overall action table's
            # exits-next-bar banner covers the miners sleeves (GDX/NUGT) too.
            return dict(state="EXIT", label=f"EXIT — {why}", ico="🔴", tone="exit",
                        exits_next_bar=True)
        return dict(state="HOLD", label="LONG — HOLDING", ico="🟢", tone="hold")
    if exit_sig:
        return dict(state="AVOID", label=f"STAND ASIDE — EXIT ACTIVE ({why})",
                    ico="🟠", tone="watch")
    if entry:
        return dict(state="ENTRY", label="ENTER — PURE-REGIME BUY", ico="🟢", tone="buy")
    if bool(sigs.get("u1_triggered")):
        return dict(state="WATCH", label="WATCH — U1 (GATE PENDING)", ico="🟡", tone="watch")
    if d1:
        return dict(state="AVOID", label="STAND ASIDE — DOWNTREND (D1)", ico="🟠", tone="watch")
    return dict(state="FLAT", label="FLAT — NO SIGNAL", ico="⬜", tone="flat")


def _attach_nugt(df: pd.DataFrame) -> pd.DataFrame:
    """Attach NUGT (2× gold miners) as an extra traded-price column, driven by
    the same gold signal — the leveraged-miners analog of UGL/GDX.  Fetched here
    in the Overall wrapper so ``gldm_core`` and the dedicated Gold app stay
    untouched.  Best-effort: a NUGT fetch failure leaves the gold sleeve intact."""
    if df is None or df.empty or "nugt_close" in df.columns:
        return df
    try:
        nugt = gc._merge({"nugt": "NUGT"}, "1d", start="2018-06-26")
        if not nugt.empty and "nugt_close" in nugt.columns:
            for suff in ("open", "high", "low", "close"):
                col = f"nugt_{suff}"
                if col in nugt:
                    df[col] = nugt[col].reindex(df.index).ffill()
    except Exception:
        pass
    return df


def _gldm_gate_spec():
    """GateSpec for the shared gold dataset (see app/data_gate.py) — the
    GLDM/UGL/GDX columns from gldm_core.fetch_daily plus the NUGT columns
    attached by ``_attach_nugt`` (all four sleeves trade off this one frame)."""
    import data_gate as dg
    traded = [f"gldm_{s}" for s in ("open", "high", "low", "close")]
    traded += [f"{t.lower()}_close" for t in gc.LEVERAGED_SYMBOLS]
    if "nugt_close" not in traded:      # older configs attach NUGT separately
        traded.append("nugt_close")
    syms = {"gldm_close": gc.PRIMARY_SYMBOL}
    syms.update({f"{t.lower()}_close": t for t in gc.LEVERAGED_SYMBOLS})
    syms.setdefault("nugt_close", "NUGT")
    return dg.GateSpec(key="GLDM", price_col="gldm_close",
                       traded_close_cols=traded,
                       macro_close_cols=[f"{n}_close" for n in gc.MACRO_SYMS],
                       snapshot_csv=gc.DAILY_CACHE_CSV,
                       symbol_by_col=syms)


def _fetch_recent_gldm() -> pd.DataFrame:
    """Last few daily rows (incl. the in-progress bar) for the live overlay
    on the pinned gold snapshot — mirrors ticker_core.fetch_recent_daily."""
    syms = {"gldm": gc.PRIMARY_SYMBOL}
    syms.update({t.lower(): t for t in gc.LEVERAGED_SYMBOLS})
    syms.update(gc.MACRO_SYMS)
    syms["nugt"] = "NUGT"
    df = gc._merge(syms, "1d", range_="5d")
    if df.empty:
        return df
    df.index = pd.to_datetime(df.index).normalize()
    return df[~df.index.duplicated(keep="last")]


def _load_daily() -> pd.DataFrame:
    # Quality-gated, snapshot-pinned load (app/data_gate.py): validated
    # completed history is pinned per session-roll so the gold back-tests are
    # identical between page loads; rejected fetches serve the last known-good
    # snapshot; every decision lands in runtime/dataset_audit.json.
    try:
        import data_gate as dg
        df = dg.gated_daily(
            _gldm_gate_spec(),
            fetch_full=lambda: _attach_nugt(gc.fetch_daily(start="2018-06-26")),
            fetch_recent=_fetch_recent_gldm,
            consumer="GLDM/GDXM")
        if (df is not None and not df.empty and "gldm_close" in df.columns
                and len(df) > 260):
            return _attach_nugt(df)       # no-op when nugt_close already there
    except Exception:
        pass
    df = pd.DataFrame()                    # gate unavailable — legacy path
    try:
        d = gc.fetch_daily(start="2018-06-26")
        if d is not None and not d.empty and "gldm_close" in d.columns and len(d) > 260:
            df = d
    except Exception:
        df = pd.DataFrame()
    if df.empty:
        csv = gc.GLDM_DATA_DIR / "gldm_macro_daily.csv"
        if Path(csv).exists():
            df = pd.read_csv(csv, index_col=0, parse_dates=True)
    return _attach_nugt(df)


import threading

_RUN_CACHE: dict = {}
_RUN_LOCK = threading.Lock()


def run_gldm_cached() -> list[dict]:
    """One shared engine run per process/day — the two gold parent apps
    (🥇 Gold Trend and ⛏️ Gold Miners) both consume it, so the Overall
    universe doesn't fetch/simulate the gold stack twice."""
    key = pd.Timestamp.utcnow().strftime("%Y-%m-%d-%H")   # hourly freshness
    with _RUN_LOCK:
        if _RUN_CACHE.get("key") != key:
            _RUN_CACHE["key"] = key
            _RUN_CACHE["out"] = run_gldm()
        return _RUN_CACHE["out"]


def run_gldm_trend() -> list[dict]:
    """🥇 Gold Trend sleeves (GLDM & UGL · dual-MA 25/100)."""
    return [r for r in run_gldm_cached() if r["parent"] == "GLDM"]


def run_gldm_miners() -> list[dict]:
    """⛏️ Gold Miners sleeves (GDX & NUGT · Divergence Pure-Regime)."""
    return [r for r in run_gldm_cached() if r["parent"] == "GDXM"]


def run_gldm() -> list[dict]:
    daily = _load_daily()
    if daily is None or daily.empty or "gldm_close" not in daily.columns:
        return []
    # COMMITTED signals from completed sessions only, exactly like the Gold app
    # (it builds predictions on freshness.drop_in_progress_us_bar(daily)).  The
    # gated frame overlays Yahoo's in-progress *today* row during US market
    # hours for live prices — but that bar's high is only "the high so far", so
    # scoring it as a completed bar biases err_hi downward and can fire a
    # spurious D2 momentum-fade EXIT intraday that the Gold Miners app (which
    # strips the partial bar) never shows.  ``daily`` keeps the partial bar for
    # the live display prices below.
    import freshness as _frs
    hist = _frs.drop_in_progress_us_bar(daily)
    if hist is None or hist.empty or "gldm_close" not in hist.columns:
        hist = daily
    preds = bg.build_predictions(hist)
    # build_predictions only keeps ugl/gdx closes; carry NUGT's price through so
    # the same gold signal can be simulated on the 2× miners sleeve.
    if "nugt_close" in daily.columns:
        preds = preds.copy()
        preds["nugt_close"] = daily["nugt_close"].reindex(preds.index)
    sig = bg.precompute_signals(preds)
    completed = preds[preds["actual_high"].notna() & preds["actual_low"].notna()]
    # tail(150), not 45: the 60-bar median centering + 30-bar capitulation
    # normaliser need ≥90 bars of history for the newest bar's signals to
    # match bg.precompute_signals (which the position simulation below uses).
    sigs = gc.compute_trend_signatures(completed.tail(150)) if len(completed) >= 3 else None
    bull_now = bool((sigs or {}).get("bull_regime", False))
    try:
        s = gc.macro_sentiment(daily) if hasattr(gc, "macro_sentiment") else None
        sent = float(s.dropna().iloc[-1]) if s is not None and s.notna().any() else np.nan
    except Exception:
        sent = np.nan
    gcl = daily["gldm_close"].to_numpy(float)
    ref = gcl[max(0, len(gcl) - 50):].mean()
    mom = (gcl[-1] / ref - 1) if ref else 0.0

    dual_long_now = bool(bg.dual_ma_long_array(preds)[-1])
    out = []
    for key, meta in _META.items():
        col = meta["col"]
        if col not in preds.columns:
            continue
        stop = gc.STOP_BY_ASSET.get(key, gc.FIXED_STOP)
        # middle-path dispatch: dual-MA 25/100 for GLDM & UGL, divergence for
        # GDX & NUGT — the same bg.run_asset_sim path the Gold app trades.
        r = bg.run_asset_sim(preds, sig, key, oos_start=OOS)
        dates = pd.to_datetime(pd.Series(r["dates"]))
        strat = np.asarray(r["strat"], float)
        eq = pd.Series(strat, index=dates)
        ret = pd.Series(np.diff(strat) / strat[:-1], index=dates.iloc[1:]).rename(key)
        pos_series = pd.Series(np.asarray(r["pos"], float), index=dates).rename(key)
        last_px = float(daily[col].dropna().iloc[-1])
        prev_px = float(daily[col].dropna().iloc[-2])
        dchg = (last_px / prev_px - 1) * 100 if prev_px else 0.0
        as_of = pd.Timestamp(dates.iloc[-1])
        if gc.engine_for(key) == "dual_ma":
            dec = _trend_decision(dual_long_now, bool(r.get("in_pos_now")))
        else:
            dec = _decision(sigs, bool(r.get("in_pos_now")))
        m = _curve_metrics(eq)
        bhm = _curve_metrics(pd.Series(r["bh"], index=dates))
        wr = float((r["trades"] > 0).mean() * 100) if len(r["trades"]) else 0.0
        pos = dict(in_pos=bool(r.get("in_pos_now")), entry_px=r.get("entry_px"),
                   entry_date=r.get("entry_date"), upnl=None, stop_px=None,
                   days=None, dist_stop=None)
        if pos["in_pos"] and pos["entry_px"]:
            e_px = float(pos["entry_px"]); e_dt = pd.Timestamp(pos["entry_date"])
            pos.update(upnl=(last_px / e_px - 1) * 100,
                       days=int((as_of - e_dt).days))
            if stop < 0.999:                       # no stop_px for a stop-less sibling (UGL)
                pos.update(stop_px=e_px * (1 - stop),
                           dist_stop=(last_px / (e_px * (1 - stop)) - 1) * 100)
        last_trade = r["trade_log"][-1] if r.get("trade_log") else None
        out.append(dict(
            key=key, parent=meta["parent"], name=meta["name"], kind=meta["kind"],
            emoji=meta["emoji"], kemoji=KIND_EMOJI[meta["kind"]], accent=ACCENT,
            cap=CAP_BY_KIND[meta["kind"]],
            last_close=last_px, dchg=dchg, ma_val=None,
            sentiment=sent, decision=dec, alert=(sigs or {}).get("alert_level", "NEUTRAL"),
            bull_regime=bull_now, pos=pos, last_trade=last_trade, mom=mom,
            metrics=m, bh_metrics=bhm, win_rate=wr, n_trades=int(len(r["trades"])),
            ret=ret, pos_series=pos_series, strat=strat, dates=dates,
            r=dict(bh=np.asarray(r["bh"], float), dates=list(dates), trades=r["trades"],
                   trade_log=r.get("trade_log"),   # dated closed trades — overall_core's
                   in_pos_now=bool(r.get("in_pos_now"))),  # since-start counter needs them
            as_of=as_of, mode=gc.engine_for(key),
            ma_window=(gc.DUAL_MA_SLOW if gc.engine_for(key) == "dual_ma"
                       else gc.MA_WINDOW_BY_ASSET.get(key, 50)),
            stop=stop, engine="gldm",
        ))
    return out
