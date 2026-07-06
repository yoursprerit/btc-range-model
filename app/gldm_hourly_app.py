"""Streamlit app — GLDM (SPDR Gold MiniShares) forecasting & trading.

The gold counterpart of btc_hourly_app.py, laid out to mirror it: a Live tab
with the same KPI rows / trend-signature alert cards / strategy description, a
Historical Replay tab, and per-asset Backtesting tabs.

Gold-specific logic lives in app/gldm_core.py; models in models/gldm/*.joblib
(trained by src/gldm/train_gldm.py).  This module never touches the BTC app.

The traded universe mirrors BTC → MSTR / MSTU: the 1x GLDM position only
supplies the signal and forecasts, while the strategy trades

    GDX  = VanEck Gold Miners ETF   (high-beta gold — ~MSTR analog)
    UGL  = ProShares Ultra Gold 2x  (leveraged gold — ~MSTU analog)

with ONE strategy — the gold-scaled Divergence Pure-Regime system.

Tabs
  🔴 Live              current forecast + trend-signature alert + strategy + positions
  🕒 Historical replay replay the signals/positions as of any past date (GDX + UGL)
  📊 GDX Backtesting   full / bull / chop performance vs buy & hold
  📈 UGL Backtesting   full / bull / chop performance vs buy & hold
  🗓️ H/L & Cones       daily High/Low + 7-day & 14-day close cones
  🧠 Explain           models, features, methodology, freshness, honest framing
"""
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

_APP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _APP_DIR.parent
sys.path.insert(0, str(_APP_DIR))
sys.path.insert(0, str(_REPO_ROOT))

try:
    import sklearn._loss._loss as _sk_loss_ext
    if "_loss" not in sys.modules:
        sys.modules["_loss"] = _sk_loss_ext
except Exception:
    pass

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go

import importlib
import gldm_core
import backtest_gldm

# Streamlit re-executes the main script from disk on every run, but `import`
# returns whatever is cached in sys.modules — so after a code update that adds
# new symbols, a long-lived server process (e.g. Streamlit Cloud that didn't
# fully restart) can hand back a STALE gldm_core / backtest_gldm and raise
# AttributeError (e.g. missing STRATEGY_NAME).  Self-heal by reloading from disk
# whenever an expected new symbol is absent.  Reload gldm_core first so
# backtest_gldm's `import gldm_core as gc` binds to the refreshed module.
if not hasattr(gldm_core, "STRATEGY_NAME"):
    gldm_core = importlib.reload(gldm_core)
if not hasattr(backtest_gldm, "drawdown_series"):
    backtest_gldm = importlib.reload(backtest_gldm)
gc = gldm_core
btg = backtest_gldm

st.set_page_config(page_title="GLDM Gold Forecaster", page_icon="🥇",
                   layout="wide", initial_sidebar_state="expanded")

ASSET_LABELS = {"GDX": "GDX · Gold Miners", "UGL": "UGL · 2× Gold"}

# ════════════════════════════════════════════════════════════════════════
# Sidebar — application selector (shared with the BTC app) + controls
# ════════════════════════════════════════════════════════════════════════
if "gldm_active_app" not in st.session_state:
    st.session_state["gldm_active_app"] = "GLDM"
with st.sidebar:
    st.radio("**Application**", options=["BTC", "GLDM"],
             format_func=lambda x: "₿  Bitcoin (BTC)" if x == "BTC" else "🥇  Gold (GLDM)",
             key="gldm_active_app")
    st.markdown("---")
    st.markdown("**Auto-refresh:** live data cached ~5 min.")
    if st.button("Refresh now", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    st.caption("_GLDM & macro pulled from Yahoo. Gold trades US market hours; "
               "intraday bars update through the session._")

st.title("🥇 GLDM — Gold forecast & trend strategy")
st.caption(
    "SPDR Gold MiniShares (GLDM) tracks spot gold. Models: ridge on log-returns "
    "(hourly close), ridge H/L bands & close cones, logistic day-type — driven by "
    "gold's real macro factors (USD index, 10-year yields, silver, VIX). The "
    f"**{gc.STRATEGY_NAME}** strategy trades **GDX** (miners) and **UGL** (2× gold) "
    "off the gold signal — the way the BTC app trades MSTR & MSTU."
)


# ════════════════════════════════════════════════════════════════════════
# Loaders
# ════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model(path_str: str):
    p = Path(path_str)
    if not p.exists():
        return None
    try:
        return joblib.load(p)
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner="Fetching GLDM daily data…")
def get_daily():
    d = gc.fetch_daily(start="2015-01-01")
    if d is None or d.empty:
        if gc.DAILY_CACHE_CSV.exists():
            d = pd.read_csv(gc.DAILY_CACHE_CSV, index_col=0, parse_dates=True)
    return d


@st.cache_data(ttl=300, show_spinner="Fetching GLDM hourly data…")
def get_hourly():
    h = gc.fetch_hourly(range_="730d")
    if h is None or h.empty:
        if gc.HOURLY_CACHE_CSV.exists():
            h = pd.read_csv(gc.HOURLY_CACHE_CSV, index_col=0, parse_dates=True)
    return h


@st.cache_data(ttl=600, show_spinner=False)
def get_predictions(_daily_key: str):
    daily = get_daily()
    preds = btg.build_predictions(daily)
    sig = btg.precompute_signals(preds)
    return preds, sig


def _fresh(art):
    if not art:
        return "_missing_"
    meta = art.get("calibration_meta", {})
    end = meta.get("train_end") or art.get("train_end")
    return end.split()[0] if isinstance(end, str) else "_unknown_"


M_HOURLY = load_model(str(gc.HOURLY_MODEL_GLDM))
M_HL     = load_model(str(gc.DAILY_HL_GLDM))
M_7D     = load_model(str(gc.CONE_7D_GLDM))
M_14D    = load_model(str(gc.CONE_14D_GLDM))
M_DT     = load_model(str(gc.DAY_TYPE_GLDM))

daily = get_daily()
if daily is None or daily.empty:
    st.error("Could not fetch GLDM data from Yahoo. Click **Refresh now** to retry.")
    st.stop()
_daily_key = f"{daily.index.max()}::{len(daily)}"
preds, sig = get_predictions(_daily_key)
completed_all = preds[preds["actual_high"].notna() & preds["actual_low"].notna()]


# ════════════════════════════════════════════════════════════════════════
# Inference helpers
# ════════════════════════════════════════════════════════════════════════
def predict_next_hour(hourly):
    if M_HOURLY is None or hourly is None or hourly.empty:
        return None
    feat = gc.build_hourly_features(hourly)
    cols = M_HOURLY["feat_cols"]
    x = feat[cols].ffill().iloc[[-1]]
    if x.isna().any(axis=None):
        return None
    r = float(M_HOURLY["model"].predict(x)[0])
    last_close = float(hourly["gldm_close"].iloc[-1])
    sigma = float(M_HOURLY["sigma"])
    return dict(last_close=last_close, pred_close=last_close * np.exp(r),
                lo=last_close * np.exp(r - 1.96 * sigma),
                hi=last_close * np.exp(r + 1.96 * sigma),
                ret=r, sigma=sigma, ts=hourly.index[-1])


def predict_next_daily_hl(daily_df):
    if M_HL is None:
        return None
    feat = gc.build_daily_features(daily_df)
    cols = M_HL["feat_cols"]
    x = feat[cols].ffill().iloc[[-1]]
    if x.isna().any(axis=None):
        return None
    last_close = float(daily_df["gldm_close"].iloc[-1])
    bh = M_HL.get("bias_high", 0.0); bl = M_HL.get("bias_low", 0.0)
    ph = last_close * (1 + float(M_HL["model_high"].predict(x)[0])) + bh * last_close
    pl = last_close * (1 + float(M_HL["model_low"].predict(x)[0])) + bl * last_close
    return dict(last_close=last_close, pred_high=ph, pred_low=pl,
                band_hi=M_HL["sigma_high"] * last_close,
                band_lo=M_HL["sigma_low"] * last_close)


def predict_cone(art, daily_df):
    if art is None:
        return None
    feat = gc.build_daily_features(daily_df)
    cols = art["feat_cols"]
    x = feat[cols].ffill().iloc[[-1]]
    if x.isna().any(axis=None):
        return None
    last_close = float(daily_df["gldm_close"].iloc[-1])
    central_r = float(art["model"].predict(x)[0])
    q = art["quantiles"]
    return dict(last_close=last_close, horizon=art["horizon"],
                central=last_close * np.exp(central_r),
                p5=last_close * np.exp(central_r + q[5]),
                p25=last_close * np.exp(central_r + q[25]),
                p75=last_close * np.exp(central_r + q[75]),
                p95=last_close * np.exp(central_r + q[95]))


def predict_day_type(daily_df):
    if M_DT is None:
        return None
    feat = gc.build_daily_features(daily_df)
    cols = M_DT["feat_cols"]
    x = feat[cols].ffill().iloc[[-1]]
    if x.isna().any(axis=None):
        return None
    proba = M_DT["model"].predict_proba(x)[0]
    classes = M_DT["model"].classes_
    label_map = M_DT["classes"]
    pm = {label_map[int(c)]: float(p) for c, p in zip(classes, proba)}
    return dict(probs=pm, top=max(pm, key=pm.get))


def signatures_asof(target_date):
    """Trend signatures using only completed bars up to (and including) target_date."""
    sub = completed_all[completed_all["target_date"] <= pd.Timestamp(target_date)].tail(45)
    return gc.compute_trend_signatures(sub)


def strategy_position(asset, end=None):
    """Run the chosen strategy for `asset` up to `end`; return current state,
    metrics, equity curve, drawdown and trade log."""
    col = f"{asset.lower()}_close"
    if col not in preds:
        return None
    r = btg.simulate(preds, sig, col, gc.FIXED_STOP, gc.U1_ERRHI_MIN,
                     gc.D2_ERRHI_MAX, gc.D1_ERRLO_MIN, end=end)
    r["metrics"] = btg._metrics(r["strat"], r["dates"])
    r["bh_metrics"] = btg._metrics(r["bh"], r["dates"])
    return r


# ════════════════════════════════════════════════════════════════════════
# Trend-signature alert renderer (mirrors the BTC Live-tab card layout)
# ════════════════════════════════════════════════════════════════════════
_ALERT_CFG = {
    "HIGH_DN":     {"bg": "#fee2e2", "border": "#dc2626", "badge": "#dc2626",
                    "txt": "⚠️ HIGH DOWNTREND ALERT", "col": "#7f1d1d"},
    "ELEVATED_DN": {"bg": "#fef3c7", "border": "#d97706", "badge": "#d97706",
                    "txt": "🟠 ELEVATED DOWNTREND RISK", "col": "#78350f"},
    "WATCH_DN":    {"bg": "#fffbeb", "border": "#f59e0b", "badge": "#f59e0b",
                    "txt": "👁 DOWNTREND WATCH", "col": "#92400e"},
    "STRATEGY_BUY":{"bg": "#fff7e6", "border": "#b8860b", "badge": "#b8860b",
                    "txt": "🎯 PURE-REGIME BUY", "col": "#7a5901"},
    "WATCH_UP":    {"bg": "#f0fdf4", "border": "#16a34a", "badge": "#16a34a",
                    "txt": "📈 UPTREND SIGNAL (U1)", "col": "#14532d"},
    "NEUTRAL":     {"bg": "#f8fafc", "border": "#94a3b8", "badge": "#64748b",
                    "txt": "⬜ NO ACTIVE SIGNAL", "col": "#334155"},
}


def _sig_card(title, active, subtitle, value_line, good):
    """One signature card as styled HTML (green if bullish-active, red if
    bearish-active, grey if idle) — matches the BTC 2×2 card grid."""
    if not active:
        bg, brd, ico = "#f1f5f9", "#cbd5e1", "○"
    elif good:
        bg, brd, ico = "#dcfce7", "#16a34a", "✅"
    else:
        bg, brd, ico = "#fee2e2", "#dc2626", "⚠️"
    return (f"<div style='background:{bg};border:1.5px solid {brd};border-radius:9px;"
            f"padding:10px 12px;height:100%;'>"
            f"<div style='font-weight:700;font-size:13px;'>{ico} {title}</div>"
            f"<div style='font-size:11px;color:#475569;margin:2px 0 4px;'>{subtitle}</div>"
            f"<div style='font-size:12px;font-family:monospace;'>{value_line}</div></div>")


def render_gldm_signatures(sigs):
    if not sigs:
        st.info("Not enough completed bars for trend signatures yet (need ≥ 3).")
        return
    cfg = _ALERT_CFG[sigs["alert_level"]]
    as_of = pd.Timestamp(sigs["as_of_date"]).strftime("%Y-%m-%d")
    entry = sigs["entry_triggered"]
    st.markdown(
        f"""<div style="background:{cfg['bg']};border:2px solid {cfg['border']};
        border-radius:10px;padding:12px 16px;margin:8px 0;">
        <span style="background:{cfg['badge']};color:white;font-weight:700;font-size:14px;
        padding:5px 14px;border-radius:20px;">{cfg['txt']}</span>
        <span style="color:{cfg['col']};font-size:13px;margin-left:10px;">
        <b>{sigs['dn_count']}/3</b> DN · <b>{sigs['up_count']}/1</b> UP ·
        Entry (Pure-Regime): <b>{'✅ ACTIVE' if entry else '○ inactive'}</b>
        (U1={'✓' if sigs['u1_triggered'] else '✗'}
        bull_regime={'✓' if sigs.get('bull_regime') else '✗'}
        clean7d={'✓' if sigs['clean_10d'] else '✗'}
        V-rev={'✓' if sigs.get('v_recent_gate') else '✗'})
        · as-of <b>{as_of}</b> · <b>{sigs['n_bars']}</b> bars</span></div>""",
        unsafe_allow_html=True)

    # 2×2 signature cards: D1 / D2 / D3 / U1
    r1c1, r1c2 = st.columns(2)
    r2c1, r2c2 = st.columns(2)
    r1c1.markdown(_sig_card(
        "D1 — Downtrend pressure", sigs["d1_triggered"],
        f"≥2 low-breaks (3d) & err_lo 3d-avg > +{gc.D1_ERRLO_MIN:.2f}%",
        f"err_lo_ma3={sigs['err_lo_ma3']:+.3f}%  lo_breaks_3d={sigs['lo_breaks_3d']}",
        good=False), unsafe_allow_html=True)
    r1c2.markdown(_sig_card(
        "D2 — Momentum fading", sigs["d2_triggered"],
        f"err_hi 3d-avg < {gc.D2_ERRHI_MAX:+.2f}%",
        f"err_hi_ma3={sigs['err_hi_ma3']:+.3f}%", good=False),
        unsafe_allow_html=True)
    r2c1.markdown(_sig_card(
        "D3 — Exhaustion canary", sigs["d3_triggered"],
        "first low-break after ≥3 high-break streak",
        f"consec_hi={sigs['consec_hi']}  exhaustion={'yes' if sigs['exhaustion_active'] else 'no'}",
        good=False), unsafe_allow_html=True)
    r2c2.markdown(_sig_card(
        "U1 — Bullish pressure", sigs["u1_triggered"],
        f"err_hi 3d-avg > +{gc.U1_ERRHI_MIN:.2f}% & ≥2 high-breaks",
        f"err_hi_ma3={sigs['err_hi_ma3']:+.3f}%  hi_breaks_3d={sigs['hi_breaks_3d']}",
        good=True), unsafe_allow_html=True)

    # Full-width action banner
    if sigs["d2_triggered"] or sigs["d3_triggered"]:
        bg, brd, ico, lab = "#fef2f2", "#dc2626", "🔴", "EXIT SIGNAL ACTIVE"
        sub = " · ".join([s for s, f in [("D3 exhaustion", sigs["d3_triggered"]),
                                         ("D2 fade", sigs["d2_triggered"])] if f])
    elif entry:
        bg, brd, ico, lab = "#f0fdf4", "#16a34a", "🟢", "ENTRY SIGNAL ACTIVE"
        gates = [g for g, f in [("🐂 Bull Regime", sigs.get("bull_regime")),
                                ("🧹 Clean Breakout", sigs["clean_10d"] and not sigs["above_ma20"]),
                                ("⚡ V-reversal", sigs.get("v_recent_gate"))] if f]
        sub = f"U1 confirmed · trend gate: {' + '.join(gates) or '—'}"
    elif sigs["u1_triggered"]:
        bg, brd, ico, lab = "#fefce8", "#ca8a04", "🟡", "U1 ACTIVE — TREND GATE NOT MET"
        sub = "bullish pressure, but no regime/clean/V-reversal confirmation yet"
    else:
        bg, brd, ico, lab = "#f8fafc", "#94a3b8", "⬜", "NO ACTION SIGNAL"
        sub = "waiting for a Pure-Regime entry or a regime-exit signal"
    st.markdown(
        f"""<div style="background:{bg};border:2px solid {brd};border-radius:10px;
        padding:10px 16px;margin:8px 0;"><b style="font-size:15px;">{ico} {lab}</b>
        <span style="color:#475569;font-size:13px;margin-left:8px;">{sub}</span></div>""",
        unsafe_allow_html=True)
    if sigs.get("v_reversal_likely"):
        st.markdown("⚡ **V-reversal likely** — capitulation low undershoot detected.")

    # Last-5-bars table
    rows = pd.DataFrame(sigs["detail_rows"])
    if not rows.empty:
        rows["date"] = pd.to_datetime(rows["date"]).dt.date
        show = rows[["date", "close", "pred_hi", "actual_hi", "err_hi_pct",
                     "pred_lo", "actual_lo", "err_lo_pct", "hi_break", "lo_break"]]
        st.dataframe(show.round(2), use_container_width=True, hide_index=True)


def position_panel(asset, col_container, end=None):
    """Open-position / current-signal panel for one traded asset."""
    r = strategy_position(asset, end=end)
    label = ASSET_LABELS[asset]
    col = f"{asset.lower()}_close"
    if r is None:
        col_container.info(f"{label}: price series unavailable.")
        return
    if end is None:
        px = float(preds[col].iloc[-1])
    else:
        sub = preds[preds["target_date"] <= pd.Timestamp(end)]
        px = float(sub[col].iloc[-1])
    if r["in_pos_now"] and r["entry_px"]:
        upnl = (px / r["entry_px"] - 1) * 100
        col_container.markdown(
            f"**{label} — 🟢 LONG**  \n"
            f"entry ${r['entry_px']:,.2f} on {pd.Timestamp(r['entry_date']).date()}  \n"
            f"now ${px:,.2f} · unrealized **{upnl:+.2f}%**")
    else:
        col_container.markdown(f"**{label} — ⚪ CASH**  \nlast close ${px:,.2f} · flat, awaiting entry")


# ════════════════════════════════════════════════════════════════════════
# Backtesting dashboard (mirrors BTC/MSTR backtesting tabs)
# ════════════════════════════════════════════════════════════════════════
_PERIODS = {"Full (2021 → now)": ("2021-01-01", None),
            "Chop (2021–2022)": ("2021-01-01", "2022-12-31"),
            "Bull (2023 → now)": ("2023-01-01", None)}


def render_backtest_dashboard(asset):
    col = f"{asset.lower()}_close"
    st.markdown(f"## {'📊' if asset == 'GDX' else '📈'} {ASSET_LABELS[asset]} — "
                "Gold Signal-Driven Backtesting")
    st.markdown(
        f"Trades **{asset}** off the **GLDM {gc.STRATEGY_NAME}** signal — entry when "
        f"U1 bullish divergence (> +{gc.U1_ERRHI_MIN:.2f}%) confirms inside the Pure-Regime "
        f"gate (Bull Regime *or* washed-out Clean Breakout *or* V-reversal); exit on "
        f"D2 (< {gc.D2_ERRHI_MAX:+.2f}%) / D3 exhaustion or a fixed **−{gc.FIXED_STOP*100:.0f}%** stop. "
        "Signals and forecasts come from gold (GLDM); execution is in the traded asset — "
        "the way the BTC app runs one BTC signal across MSTR / MSTU.")
    st.caption("All trades are out-of-sample: the daily H/L model is fit once on the "
               "pre-2021 window and predicts every later bar. Costs/slippage not modelled.")

    # KPI table across the three periods
    rows = []
    for lbl, (s, e) in _PERIODS.items():
        r = btg.simulate(preds, sig, col, gc.FIXED_STOP, gc.U1_ERRHI_MIN,
                         gc.D2_ERRHI_MAX, gc.D1_ERRLO_MIN, oos_start=s, end=e)
        sm = btg._metrics(r["strat"], r["dates"]); bm = btg._metrics(r["bh"], r["dates"])
        wr = (r["trades"] > 0).mean() * 100 if len(r["trades"]) else 0
        rows.append(dict(Period=lbl,
                         Strategy=f"{sm['total_ret']*100:+.1f}%",
                         **{"Buy&Hold": f"{bm['total_ret']*100:+.1f}%"},
                         **{"Strat MDD": f"{sm['mdd']*100:.1f}%",
                            "B&H MDD": f"{bm['mdd']*100:.1f}%",
                            "Sharpe": f"{sm['sharpe']:.2f}",
                            "Trades": len(r["trades"]),
                            "Win%": f"{wr:.0f}%"}))
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # headline metrics (full period)
    rf = btg.simulate(preds, sig, col, gc.FIXED_STOP, gc.U1_ERRHI_MIN,
                      gc.D2_ERRHI_MAX, gc.D1_ERRLO_MIN, oos_start="2021-01-01")
    sm = btg._metrics(rf["strat"], rf["dates"]); bm = btg._metrics(rf["bh"], rf["dates"])
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Strategy total return", f"{sm['total_ret']*100:+.1f}%", f"CAGR {sm['cagr']*100:+.1f}%")
    k2.metric("Buy & Hold return", f"{bm['total_ret']*100:+.1f}%", f"CAGR {bm['cagr']*100:+.1f}%")
    k3.metric("Max drawdown", f"{sm['mdd']*100:.1f}%", f"vs B&H {bm['mdd']*100:.1f}%",
              delta_color="inverse")
    k4.metric("Sharpe", f"{sm['sharpe']:.2f}", f"vs B&H {bm['sharpe']:.2f}")

    # equity + drawdown
    dts = pd.to_datetime(rf["dates"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dts, y=rf["strat"], name="Strategy",
                             line=dict(color="#b8860b", width=2)))
    fig.add_trace(go.Scatter(x=dts, y=rf["bh"], name="Buy & Hold",
                             line=dict(color="#888", dash="dash")))
    fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0),
                      yaxis_title="Growth of $1", legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    dd = btg.drawdown_series(rf["strat"]); ddb = btg.drawdown_series(rf["bh"])
    figd = go.Figure()
    figd.add_trace(go.Scatter(x=dts, y=dd * 100, name="Strategy DD", fill="tozeroy",
                              line=dict(color="#b8860b")))
    figd.add_trace(go.Scatter(x=dts, y=ddb * 100, name="Buy & Hold DD",
                              line=dict(color="#bbb", dash="dash")))
    figd.update_layout(height=200, margin=dict(l=0, r=0, t=6, b=0),
                       yaxis_title="Drawdown %", legend=dict(orientation="h"))
    st.plotly_chart(figd, use_container_width=True)

    # trade log
    st.markdown("#### Trade log (full period)")
    tl = pd.DataFrame(rf["trade_log"])
    if not tl.empty:
        tl["entry_date"] = pd.to_datetime(tl["entry_date"]).dt.date
        tl["exit_date"] = pd.to_datetime(tl["exit_date"]).dt.date
        tl["ret"] = (tl["ret"] * 100).round(2)
        tl = tl.rename(columns={"entry_date": "Entry", "exit_date": "Exit",
                                "entry_px": "Entry $", "exit_px": "Exit $",
                                "ret": "Return %", "reason": "Exit reason"})
        tl["Entry $"] = tl["Entry $"].round(2); tl["Exit $"] = tl["Exit $"].round(2)
        st.dataframe(tl[::-1], use_container_width=True, hide_index=True, height=300)
        wins = tl[tl["Return %"] > 0]["Return %"]; losses = tl[tl["Return %"] <= 0]["Return %"]
        st.caption(f"{len(tl)} trades · avg win {wins.mean():.2f}% · avg loss "
                   f"{losses.mean() if len(losses) else 0:.2f}% · "
                   f"best {tl['Return %'].max():.1f}% · worst {tl['Return %'].min():.1f}%")


# ════════════════════════════════════════════════════════════════════════
# Tabs
# ════════════════════════════════════════════════════════════════════════
tab_live, tab_hist, tab_gdx, tab_ugl, tab_cones, tab_explain = st.tabs(
    ["🔴 Live (rolling now+1h)", "🕒 Historical replay",
     "📊 GDX Backtesting", "📈 UGL Backtesting",
     "🗓️ H/L & Cones", "🧠 Explain"])


# ═════════════════════════════ LIVE ══════════════════════════════════════
def render_live_dashboard(as_of_date=None, is_live=True):
    """Shared Live / Historical-replay renderer (mirrors BTC render_dashboard)."""
    if is_live:
        d_df = daily
        hourly = get_hourly()
        ph = predict_next_hour(hourly)
    else:
        d_df = daily[daily.index <= pd.Timestamp(as_of_date)]
        ph = None
    hl = predict_next_daily_hl(d_df)
    dt = predict_day_type(d_df)
    sent = gc.gold_macro_sentiment(d_df).dropna()
    sent_now = float(sent.iloc[-1]) if len(sent) else np.nan
    sigs = signatures_asof(d_df.index[-1] if not is_live else completed_all["target_date"].iloc[-1])

    last_px = float(d_df["gldm_close"].iloc[-1])
    prev_px = float(d_df["gldm_close"].iloc[-2])

    # ── headline KPI row (5 cols) ──
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("GLDM close", f"${last_px:,.2f}", f"{(last_px/prev_px-1)*100:+.2f}% d/d")
    if ph:
        c2.metric("Next-hour close (pred)", f"${ph['pred_close']:,.2f}", f"{ph['ret']*100:+.2f}%")
    else:
        c2.metric("Next-hour close (pred)", "— (replay)")
    if ph:
        c3.metric("95% CI band", f"${ph['lo']:,.2f}–${ph['hi']:,.2f}",
                  f"±{1.96*ph['sigma']*100:.2f}%")
    else:
        c3.metric("95% CI band", "—")
    if not np.isnan(sent_now):
        mood = ("Bullish" if sent_now >= 60 else "Bearish" if sent_now <= 40 else "Neutral")
        c4.metric("Gold macro sentiment", f"{sent_now:.0f}/100", mood)
    _bull = sigs.get("bull_regime") if sigs else None
    c5.metric("Market regime", ("🐂 BULL" if _bull else "🐻 BEAR/NEUTRAL") if _bull is not None else "—",
              delta=("↑MA20 & rising" if _bull else "below MA20 or flat") if _bull is not None else None,
              delta_color="normal" if _bull else "inverse")

    # ── second row: MA20 · Daily High pred · Daily Low pred ──
    f1, f2, f3 = st.columns(3)
    _ma = sigs.get("ma20_value") if sigs else None
    if _ma:
        f1.metric("20-day MA", f"${_ma:,.2f}", f"${last_px-_ma:+,.2f} vs close",
                  delta_color="normal" if last_px >= _ma else "inverse")
    else:
        f1.metric("20-day MA", "—")
    if hl:
        f2.metric("Daily High — predicted", f"${hl['pred_high']:,.2f}",
                  f"{(hl['pred_high']/hl['last_close']-1)*100:+.2f}% vs close")
        f3.metric("Daily Low — predicted", f"${hl['pred_low']:,.2f}",
                  f"{(hl['pred_low']/hl['last_close']-1)*100:+.2f}% vs close")
        if dt:
            f2.caption(f"Tomorrow day-type: **{dt['top']}** ({max(dt['probs'].values())*100:.0f}%)")
    else:
        f2.metric("Daily High — predicted", "—"); f3.metric("Daily Low — predicted", "—")

    # ── trend-signature alert (the BTC-style card block) ──
    st.markdown("### 🔔 Trend-Signature Alert")
    render_gldm_signatures(sigs)

    # ── strategy + current positions ──
    st.markdown(f"### 🎯 Strategy — {gc.STRATEGY_NAME}  (GDX & UGL)")
    st.markdown(
        f"**Signal from gold, executed in the leveraged/high-beta names** "
        f"(GLDM 1× is not traded). **Entry:** U1 bullish divergence (> +{gc.U1_ERRHI_MIN:.2f}%) "
        f"inside the Pure-Regime gate — Bull Regime *or* washed-out Clean Breakout *or* "
        f"V-reversal. **Exit:** D2 (< {gc.D2_ERRHI_MAX:+.2f}%) / D3 exhaustion, or a fixed "
        f"**−{gc.FIXED_STOP*100:.0f}%** stop. Out-of-sample 2021→now this beats buy & hold on "
        f"**both return and drawdown**: GDX **+270%** (MDD −16% vs −47%, Sharpe 1.40), "
        f"UGL **+207%** (MDD −18% vs −49%, Sharpe 1.29).")
    p1, p2 = st.columns(2)
    end = None if is_live else as_of_date
    position_panel("GDX", p1, end=end)
    position_panel("UGL", p2, end=end)


with tab_live:
    render_live_dashboard(is_live=True)


# ═════════════════════════ HISTORICAL REPLAY ═════════════════════════════
with tab_hist:
    st.markdown("### 🕒 Historical replay — GDX & UGL")
    st.caption("Replay the gold trend-signals, forecasts and both positions exactly as "
               "they stood at the close of any past trading day.")
    # Restrict replay to the out-of-sample window: the daily H/L model is fit on
    # the pre-2021 window, so signals/positions before 2021 would be in-sample.
    dates_avail = pd.to_datetime(completed_all["target_date"])
    dates_avail = dates_avail[dates_avail >= pd.Timestamp(btg.OOS_START)]
    min_d, max_d = dates_avail.min().date(), dates_avail.max().date()
    if "gldm_hist_date" not in st.session_state:
        st.session_state["gldm_hist_date"] = max_d
    cprev, cpick, cnext = st.columns([1, 3, 1])
    with cprev:
        if st.button("◀ prev day", use_container_width=True):
            cur = st.session_state["gldm_hist_date"]
            prior = dates_avail[dates_avail < pd.Timestamp(cur)]
            if len(prior):
                st.session_state["gldm_hist_date"] = prior.max().date()
    with cnext:
        if st.button("next day ▶", use_container_width=True):
            cur = st.session_state["gldm_hist_date"]
            later = dates_avail[dates_avail > pd.Timestamp(cur)]
            if len(later):
                st.session_state["gldm_hist_date"] = later.min().date()
    with cpick:
        st.date_input("Replay date", min_value=min_d, max_value=max_d,
                      key="gldm_hist_date")
    picked = pd.Timestamp(st.session_state["gldm_hist_date"])
    # snap to nearest available completed bar ≤ picked
    avail_le = dates_avail[dates_avail <= picked]
    if len(avail_le) == 0:
        st.warning("No completed bar on or before that date.")
    else:
        snapped = avail_le.max()
        if snapped.date() != picked.date():
            st.caption(f"⚠️ Snapped to last completed bar: **{snapped.date()}**")
        render_live_dashboard(as_of_date=snapped, is_live=False)


# ═════════════════════════ BACKTESTING TABS ══════════════════════════════
with tab_gdx:
    render_backtest_dashboard("GDX")
with tab_ugl:
    render_backtest_dashboard("UGL")


# ═════════════════════════════ CONES ═════════════════════════════════════
with tab_cones:
    st.subheader("Daily High / Low band")
    hl = predict_next_daily_hl(daily)
    if hl and M_HL:
        m = M_HL["metrics_test"]
        st.caption(f"Out-of-sample: MAPE_high {m['mae_hi_pct']:.3f}%, "
                   f"MAPE_low {m['mae_lo_pct']:.3f}%, ±1σ band coverage {m['band_cov_pct']:.0f}%.")
    c7, c14 = st.columns(2)
    for col, art, label in ((c7, M_7D, "7-day"), (c14, M_14D, "14-day")):
        with col:
            st.subheader(f"{label} close cone")
            cone = predict_cone(art, daily)
            if cone:
                hist = daily["gldm_close"].tail(90)
                future = pd.date_range(hist.index[-1], periods=cone["horizon"] + 1, freq="B")[1:]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="GLDM",
                                         line=dict(color="#b8860b")))
                y0 = cone["last_close"]
                for key, clr, nm in (("p95", "rgba(192,57,43,0.25)", "P95"),
                                     ("p75", "rgba(192,57,43,0.15)", "P75"),
                                     ("central", "#c0392b", "central"),
                                     ("p25", "rgba(39,174,96,0.15)", "P25"),
                                     ("p5", "rgba(39,174,96,0.25)", "P5")):
                    yv = np.linspace(y0, cone[key], len(future) + 1)[1:]
                    fig.add_trace(go.Scatter(x=future, y=yv, name=nm,
                                             line=dict(color=clr,
                                                       dash="dot" if nm != "central" else "solid")))
                fig.update_layout(height=320, margin=dict(l=0, r=0, t=6, b=0),
                                  showlegend=True, legend=dict(orientation="h"))
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(f"**Central:** ${cone['central']:,.2f} "
                            f"({(cone['central']/cone['last_close']-1)*100:+.1f}%)  |  "
                            f"**P5–P95:** ${cone['p5']:,.2f} – ${cone['p95']:,.2f}")
                if art:
                    st.caption(f"OOS R²={art['metrics_test']['R2']:+.3f}, "
                               f"5–95 band coverage {art['metrics_test']['band_cov_pct']:.0f}%.")
            else:
                st.info(f"{label} cone unavailable.")


# ═════════════════════════════ EXPLAIN ═══════════════════════════════════
with tab_explain:
    st.subheader("How the GLDM app works")
    st.markdown(f"""
**Asset.** GLDM = SPDR Gold MiniShares, tracking spot gold. Gold's realised
volatility is roughly a quarter of Bitcoin's and it trades the US equity
calendar, so every feature and threshold is re-derived for gold.

**Gold-specific features** (replacing BTC's crypto inputs): US Dollar Index (DXY,
inverse), 10-year Treasury yield (`^TNX`, real-yield proxy, inverse), silver (SLV)
+ gold/silver ratio, gold futures (`GC=F`) basis, VIX & S&P 500, plus a
purpose-built **gold macro-sentiment 0–100** composite that replaces the crypto
Fear & Greed index.

**Five models** (`src/gldm/train_gldm.py`): hourly next-close (ridge + 95% CI),
daily High/Low (calibrated ridge bands), 7-day & 14-day close cones, and a
3-class day-type classifier.

**Strategy — {gc.STRATEGY_NAME} (one strategy, two assets).** Entry when U1
bullish divergence (3-day centered `err_hi` > +{gc.U1_ERRHI_MIN:.2f}% with ≥2
high-breaks) confirms inside the Pure-Regime gate (Bull Regime *or* a washed-out
Clean Breakout below the MA *or* a recent V-reversal). Exit on D2
(< {gc.D2_ERRHI_MAX:+.2f}%) / D3 exhaustion, or a fixed **−{gc.FIXED_STOP*100:.0f}%**
stop. The divergence error is regime-centered (rolling median) so the signal
self-calibrates to gold's low volatility. Signals come from **GLDM**; execution
is in **GDX** (miners) and **UGL** (2× gold) — the 1× GLDM position is not traded,
mirroring how the BTC app trades MSTR / MSTU rather than spot BTC.

**Out-of-sample results (2021→now)** — beats buy & hold on **both** return and
drawdown for both assets:

| Asset | Strategy | Buy & Hold | Strat MDD | B&H MDD | Sharpe |
|---|---|---|---|---|---|
| GDX | **+270%** | +104% | **−16%** | −47% | **1.40** |
| UGL | **+207%** | +161% | **−18%** | −49% | **1.29** |

**Honest framing.** Intraday gold direction is ~coin-flip (like BTC); the hourly
model's value is a tight CI, not a directional bet. The edge is in the trend
regime and risk control — quantified in the Backtesting tabs.
""")
    st.markdown("#### Model freshness (`train_end`) — GLDM")
    for label, art in (("hourly close", M_HOURLY), ("daily H/L", M_HL),
                       ("7-day cone", M_7D), ("14-day cone", M_14D),
                       ("3-class day type", M_DT)):
        st.caption(f"&bull; {label}: `{_fresh(art)}`", unsafe_allow_html=True)
    st.caption("Retrain with `python src/gldm/train_gldm.py`, then Refresh now.")
