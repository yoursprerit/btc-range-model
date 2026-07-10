"""GLDM / GDX / UGL via the Gold app's ACTUAL engine.

The Gold app (``app/gldm_hourly_app.py``) trades a Divergence Pure-Regime system
through ``backtest_gldm`` + ``gldm_core`` with per-asset regime windows
(GLDM 50 / UGL 40 / GDX 100) and −3% stops.  This module wraps those exact
modules so the Overall app's gold sleeve matches the Gold app bar-for-bar
(verified: reproduces the app's GDX +272% / UGL +211% / Sharpe ~1.3-1.4).
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
_META = {
    "GLDM": dict(name="Gold (GLDM)",  kind="core", col="gldm_close"),
    "GDX":  dict(name="Gold Miners",  kind="beta", col="gdx_close"),
    "UGL":  dict(name="2× Gold",      kind="lev",  col="ugl_close"),
    "NUGT": dict(name="2× Gold Miners", kind="lev", col="nugt_close"),
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
            return dict(state="EXIT", label=f"EXIT — {why}", ico="🔴", tone="exit")
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


def _load_daily() -> pd.DataFrame:
    df = pd.DataFrame()
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


def run_gldm() -> list[dict]:
    daily = _load_daily()
    if daily is None or daily.empty or "gldm_close" not in daily.columns:
        return []
    preds = bg.build_predictions(daily)
    # build_predictions only keeps ugl/gdx closes; carry NUGT's price through so
    # the same gold signal can be simulated on the 2× miners sleeve.
    if "nugt_close" in daily.columns:
        preds = preds.copy()
        preds["nugt_close"] = daily["nugt_close"].reindex(preds.index)
    sig = bg.precompute_signals(preds)
    completed = preds[preds["actual_high"].notna() & preds["actual_low"].notna()]
    sigs = gc.compute_trend_signatures(completed.tail(45)) if len(completed) >= 3 else None
    bull_now = bool((sigs or {}).get("bull_regime", False))
    try:
        s = gc.macro_sentiment(daily) if hasattr(gc, "macro_sentiment") else None
        sent = float(s.dropna().iloc[-1]) if s is not None and s.notna().any() else np.nan
    except Exception:
        sent = np.nan
    gcl = daily["gldm_close"].to_numpy(float)
    ref = gcl[max(0, len(gcl) - 50):].mean()
    mom = (gcl[-1] / ref - 1) if ref else 0.0

    out = []
    for key, meta in _META.items():
        col = meta["col"]
        if col not in preds.columns:
            continue
        stop = gc.STOP_BY_ASSET.get(key, gc.FIXED_STOP)
        r = bg.simulate(preds, sig, col, stop, gc.U1_ERRHI_MIN, gc.D2_ERRHI_MAX,
                        gc.D1_ERRLO_MIN, oos_start=OOS)
        dates = pd.to_datetime(pd.Series(r["dates"]))
        strat = np.asarray(r["strat"], float)
        eq = pd.Series(strat, index=dates)
        ret = pd.Series(np.diff(strat) / strat[:-1], index=dates.iloc[1:]).rename(key)
        pos_series = pd.Series(np.asarray(r["pos"], float), index=dates).rename(key)
        last_px = float(daily[col].dropna().iloc[-1])
        prev_px = float(daily[col].dropna().iloc[-2])
        dchg = (last_px / prev_px - 1) * 100 if prev_px else 0.0
        as_of = pd.Timestamp(dates.iloc[-1])
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
            key=key, parent="GLDM", name=meta["name"], kind=meta["kind"],
            emoji=EMOJI, kemoji=KIND_EMOJI[meta["kind"]], accent=ACCENT,
            cap=CAP_BY_KIND[meta["kind"]],
            last_close=last_px, dchg=dchg, ma_val=None,
            sentiment=sent, decision=dec, alert=(sigs or {}).get("alert_level", "NEUTRAL"),
            bull_regime=bull_now, pos=pos, last_trade=last_trade, mom=mom,
            metrics=m, bh_metrics=bhm, win_rate=wr, n_trades=int(len(r["trades"])),
            ret=ret, pos_series=pos_series, strat=strat, dates=dates,
            r=dict(bh=np.asarray(r["bh"], float), dates=list(dates), trades=r["trades"],
                   in_pos_now=bool(r.get("in_pos_now"))),
            as_of=as_of, mode="divergence", ma_window=gc.MA_WINDOW_BY_ASSET.get(key, 50),
            stop=stop, engine="gldm",
        ))
    return out
