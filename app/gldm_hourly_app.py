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

ASSET_LABELS = {"GDX": "GDX · Gold Miners", "UGL": "UGL · 2× Gold",
                "NUGT": "NUGT · 2× Gold Miners"}

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
    r = btg.simulate(preds, sig, col, gc.stop_for(asset), gc.U1_ERRHI_MIN,
                     gc.D2_ERRHI_MAX, gc.D1_ERRLO_MIN, end=end)
    r["metrics"] = btg._metrics(r["strat"], r["dates"])
    r["bh_metrics"] = btg._metrics(r["bh"], r["dates"])
    return r


# ════════════════════════════════════════════════════════════════════════
# Trend-signature alert renderer (mirrors the BTC Live-tab card layout)
# ════════════════════════════════════════════════════════════════════════


def _sig_card(title, icon, color, triggered, rows, interpretation):
    """One trend-signature card as styled HTML (mirrors the BTC 2×2 grid).

    ``rows`` = list of (label, current_value_str, threshold_str, fired_bool).
    Each row shows the condition's live value against its trigger threshold with
    a ✓ TRIGGERED / ○ badge, so the criteria AND current values are explicit.
    """
    border = color if triggered else "#cbd5e1"
    bg = ("#f0fdf4" if (triggered and color == "#16a34a")
          else "#fff7ed" if triggered else "#f8fafc")
    status = "● ACTIVE" if triggered else "○ CLEAR"
    badge = (f"<span style='background:{color};color:white;border-radius:12px;"
             f"padding:2px 10px;font-size:11px;font-weight:700;margin-left:8px;'>{status}</span>")
    rows_html = ""
    for label, val, thr, fired in rows:
        vcol = color if fired else "#64748b"
        bd = (f"<span style='color:{color};font-weight:700;margin-left:auto;'>✓ TRIGGERED</span>"
              if fired else "<span style='color:#94a3b8;margin-left:auto;'>○</span>")
        rows_html += (
            "<div style='display:flex;align-items:center;gap:8px;padding:4px 0;"
            "border-bottom:1px solid #e2e8f0;'>"
            f"<span style='font-size:12px;color:#64748b;width:150px;flex-shrink:0;'>{label}</span>"
            f"<span style='font-weight:700;color:{vcol};font-size:13px;min-width:74px;'>{val}</span>"
            f"<span style='font-size:11px;color:#94a3b8;'>need: {thr}</span>{bd}</div>")
    tcol = color if triggered else "#475569"
    return (
        f"<div style='background:{bg};border:2px solid {border};border-radius:10px;"
        f"padding:14px;height:100%;box-sizing:border-box;'>"
        f"<div style='font-size:14px;font-weight:700;color:{tcol};margin-bottom:8px;'>"
        f"{icon} {title}{badge}</div>"
        f"<div style='margin-bottom:8px;'>{rows_html}</div>"
        f"<div style='font-size:11.5px;color:#1e293b;background:rgba(0,0,0,0.04);"
        f"border-radius:6px;padding:7px 9px;line-height:1.45;'>"
        f"<b>📊 What it means:</b> {interpretation}</div></div>")


def _gate_card(title, icon, fired, rows, interpretation):
    """Render an ENTRY-GATE condition card. Each row is
    (label, current_value, criterion, met, kind) where kind ∈ {'threshold',
    'context'} — a threshold row is a hard numeric comparison, a context row is a
    regime/lookback state (no single cut-off)."""
    color = "#4f46e5"                       # indigo — an enabling gate, not a directional signal
    border = color if fired else "#cbd5e1"
    bg = "#eef2ff" if fired else "#f8fafc"
    status = "● SATISFIED" if fired else "○ NOT MET"
    badge = (f"<span style='background:{color if fired else '#94a3b8'};color:white;"
             f"border-radius:12px;padding:2px 10px;font-size:11px;font-weight:700;"
             f"margin-left:8px;'>{status}</span>")
    rows_html = ""
    for label, val, crit, met, kind in rows:
        vcol = color if met else "#64748b"
        tag_col = "#0e7490" if kind == "threshold" else "#a16207"
        tag = "THRESHOLD" if kind == "threshold" else "CONTEXT"
        chk = (f"<span style='color:{color};font-weight:700;margin-left:auto;'>✓</span>"
               if met else "<span style='color:#94a3b8;margin-left:auto;'>○</span>")
        rows_html += (
            "<div style='display:flex;align-items:center;gap:8px;padding:4px 0;"
            "border-bottom:1px solid #e2e8f0;'>"
            f"<span style='font-size:12px;color:#64748b;width:150px;flex-shrink:0;'>{label}</span>"
            f"<span style='font-weight:700;color:{vcol};font-size:13px;min-width:70px;'>{val}</span>"
            f"<span style='font-size:11px;color:#94a3b8;'>need: {crit}</span>"
            f"<span style='font-size:9px;font-weight:800;color:{tag_col};background:{tag_col}1a;"
            f"border-radius:5px;padding:1px 5px;margin-left:6px;'>{tag}</span>{chk}</div>")
    tcol = color if fired else "#475569"
    return (
        f"<div style='background:{bg};border:2px solid {border};border-radius:10px;"
        f"padding:14px;height:100%;box-sizing:border-box;'>"
        f"<div style='font-size:14px;font-weight:700;color:{tcol};margin-bottom:8px;'>"
        f"{icon} {title}{badge}</div>"
        f"<div style='margin-bottom:8px;'>{rows_html}</div>"
        f"<div style='font-size:11.5px;color:#1e293b;background:rgba(0,0,0,0.04);"
        f"border-radius:6px;padding:7px 9px;line-height:1.45;'>"
        f"<b>📊 What it means:</b> {interpretation}</div></div>")


def render_gldm_gate_signatures(sigs):
    """Entry-gate condition cards (divergence mode). A U1 pressure signal only
    opens a position when a trend gate also holds:
    ``U1 AND (Bull Regime OR Clean Breakout OR V-Reversal)``. Each card shows the
    criterion, where the live value stands, and whether it's a hard threshold or
    a regime/context read."""
    if not sigs:
        return
    close = sigs["detail_rows"][-1]["close"] if sigs.get("detail_rows") else None
    ma20 = sigs.get("ma20_value")
    above = bool(sigs.get("above_ma20")); slope = bool(sigs.get("ma20_slope_pos"))
    bull = bool(sigs.get("bull_regime")); clean = bool(sigs.get("clean_10d"))
    clean_gate = bool(clean and not above); vgate = bool(sigs.get("v_recent_gate"))
    u1 = bool(sigs.get("u1_triggered")); entry = bool(sigs.get("entry_triggered"))
    vdn = float(sigs.get("dn_score_raw") or 0.0)
    close_s = f"${close:,.2f}" if close is not None else "—"
    ma_s = f"${ma20:,.2f}" if ma20 is not None else "—"
    gate_ok = bull or clean_gate or vgate

    st.markdown(
        f"<div style='background:#eef2ff;border:2px solid #c7d2fe;border-radius:10px;"
        f"padding:10px 14px;margin:8px 0;font-size:13px;color:#3730a3;'>"
        f"<b>🚪 Entry gate</b> — U1 pressure only opens a position when a trend gate "
        f"also holds: <b>U1 AND (Bull Regime OR Clean Breakout OR V-Reversal)</b>. "
        f"Now: U1 <b>{'✓' if u1 else '✗'}</b> · gate <b>{'✓' if gate_ok else '✗'}</b> → "
        f"entry <b>{'✅ ARMED' if entry else '⛔ not armed'}</b>. "
        f"Exits (D2/D3) are ungated and override entry.</div>",
        unsafe_allow_html=True)

    g1, g2, g3 = st.columns(3)
    g1.markdown(_gate_card(
        "Bull Regime", "🐂", bull,
        [("close vs 20-day SMA", close_s, f"> {ma_s}", above, "threshold"),
         ("20-day SMA slope", "rising" if slope else "falling", "rising", slope, "context")],
        "Price sits above a rising 20-day average — an established uptrend. Satisfies "
        "the entry gate on its own."), unsafe_allow_html=True)
    g2.markdown(_gate_card(
        "Clean Breakout", "🧹", clean_gate,
        [("no D1/D2 last ~8 bars", "clean" if clean else "recent damage", "clean", clean, "context"),
         ("close vs 20-day SMA", close_s, f"< {ma_s}", (not above), "threshold")],
        "A fresh breakout from <i>below</i> the average with no recent downside damage — "
        "lets U1 fire early, before the regime formally turns bullish."),
        unsafe_allow_html=True)
    g3.markdown(_gate_card(
        "V-Reversal", "⚡", vgate,
        [("washout in last 3 bars", "yes — ≤3 bars ago" if vgate else "none",
          "capitulation ≤3 bars ago", vgate, "context"),
         ("capitulation score (today)", f"{vdn:.2f}", "> 0.80 + deep low", vgate, "context")],
        "A recent sharp washout / capitulation-low undershoot (V-shaped reversal setup) — "
        "also satisfies the entry gate. The gate arms when the capitulation score clears "
        "0.80 <i>and</i> the day's low deeply undershoots the model, within the last 3 bars."),
        unsafe_allow_html=True)


def net_signal(sigs):
    """Resolve the many raw signatures into ONE actionable state with a clear
    precedence — exit ALWAYS overrides entry — so the UI can never show an
    entry and an exit as simultaneously active (matches backtest execution).

    Returns dict(state, label, ico, bg, brd, reason).  state ∈
    {EXIT, ENTRY, WATCH_UP, WATCH_DN, NEUTRAL}.
    """
    if not sigs:
        return dict(state="NEUTRAL", label="NO DATA", ico="⬜",
                    bg="#f8fafc", brd="#94a3b8", reason="insufficient completed bars")
    d1, d2, d3 = sigs["d1_triggered"], sigs["d2_triggered"], sigs["d3_triggered"]
    u1, entry = sigs["u1_triggered"], sigs["entry_triggered"]
    if d2 or d3:                                   # EXIT overrides everything
        parts = [p for p, f in [("D3 exhaustion", d3), ("D2 momentum fade", d2)] if f]
        note = " — entry is blocked while an exit is active" if entry else ""
        return dict(state="EXIT", label="EXIT / STAND ASIDE", ico="🔴",
                    bg="#fef2f2", brd="#dc2626",
                    reason="Exit signal: " + " + ".join(parts) + note)
    if entry:
        gates = [g for g, f in [("🐂 Bull Regime", sigs.get("bull_regime")),
                                ("🧹 Clean Breakout", sigs["clean_10d"] and not sigs["above_ma20"]),
                                ("⚡ V-reversal", sigs.get("v_recent_gate"))] if f]
        return dict(state="ENTRY", label="ENTRY / GO LONG", ico="🟢",
                    bg="#f0fdf4", brd="#16a34a",
                    reason="U1 confirmed inside the Pure-Regime gate: " + " + ".join(gates))
    if u1:
        return dict(state="WATCH_UP", label="U1 WATCH — GATE NOT MET", ico="🟡",
                    bg="#fefce8", brd="#ca8a04",
                    reason="Bullish pressure, but no Bull-Regime / Clean-Breakout / V-reversal yet")
    if d1:
        return dict(state="WATCH_DN", label="DOWNTREND WATCH (D1)", ico="🟠",
                    bg="#fff7ed", brd="#f59e0b",
                    reason="Low-break pressure building — no exit trigger yet")
    return dict(state="NEUTRAL", label="NO ACTION SIGNAL", ico="⬜",
                bg="#f8fafc", brd="#94a3b8",
                reason="Flat — awaiting a Pure-Regime entry")


def _stops_line() -> str:
    """Per-asset fixed-stop summary for the strategy card. The leveraged siblings
    are looser than the 1×/high-beta names: GDX −3% · UGL signal-only · NUGT −5%
    (a tight 1× stop whipsaws a 2× ETF — see LEV_SIBLINGS_STOP_EVAL.md)."""
    parts = []
    for a in gc.TRADEABLE_ASSETS:
        s = gc.stop_for(a)
        parts.append(f"<b>{a}</b> " + ("signal-only" if s >= 0.999 else f"−{s * 100:.0f}%"))
    return " · ".join(parts)


def render_strategy_card():
    """Static BTC-style strategy-description card (gold theme)."""
    st.markdown(f"""
<div style='background:#fffaf0; border:2px solid #b8860b; border-radius:12px;
     padding:16px 20px; margin:4px 0 14px 0; font-family:sans-serif;'>
  <div style='font-size:15px; font-weight:800; color:#7a5901; margin-bottom:12px;
       letter-spacing:0.3px;'>
    🥇 {gc.STRATEGY_NAME} &nbsp;—&nbsp;
    <span style='color:#b8860b;'>GLDM signals · GDX, UGL &amp; NUGT execution</span>
  </div>
  <div style='background:#fdf0d5; border-radius:8px; padding:10px 14px; margin-bottom:12px;
       font-size:12.5px; color:#5c4400; font-weight:600;'>
    🔁 <b>Core idea:</b> Gold trends smoothly with shallow dips, so the same divergence
    signatures (U1 / D2 / D3 / V-reversal) that read <b>GLDM's</b> predicted-vs-actual
    highs/lows are used as the signal engine, then executed in gold's higher-octane
    proxies — <b>GDX</b> (miners, ~MSTR analog) and <b>UGL</b> (2× gold, ~MSTU analog).
    The 1× GLDM position itself is not traded.
  </div>
  <div style='background:#fdf0d5; border:1px solid #b8860b; border-radius:7px;
       padding:8px 13px; margin-bottom:12px; font-size:12px; color:#5c4400;'>
    🎯 <b>Pure-Regime entry:</b> a U1 bullish-divergence trigger must be confirmed by
    <b>one</b> of three non-overlapping trend paths — a bull regime (above a rising
    20-day MA), a washed-out clean breakout from below the MA, or a fresh V-reversal.
  </div>
  <div style='display:flex; gap:14px; flex-wrap:wrap;'>
    <div style='flex:1; min-width:230px;'>
      <div style='font-size:11px; font-weight:700; color:#15803d; text-transform:uppercase;
           letter-spacing:0.8px; margin-bottom:5px;'>📥 Entry — go long GDX, UGL &amp; NUGT</div>
      <div style='font-size:12px; color:#334155; line-height:1.7;'>
        ① <b>U1 active</b> — err_hi 3d-avg &gt; +{gc.U1_ERRHI_MIN:.2f}% &amp;&amp; ≥2 high-breaks<br>
        ② <b>one gate</b>: 🐂 Bull Regime · 🧹 Clean Breakout · ⚡ V-reversal
      </div>
    </div>
    <div style='flex:1; min-width:230px;'>
      <div style='font-size:11px; font-weight:700; color:#b91c1c; text-transform:uppercase;
           letter-spacing:0.8px; margin-bottom:5px;'>📤 Exit — go to cash (overrides entry)</div>
      <div style='font-size:12px; color:#334155; line-height:1.7;'>
        ① <b>D2 fade</b> — err_hi 3d-avg &lt; {gc.D2_ERRHI_MAX:+.2f}%<br>
        ② <b>D3 exhaustion</b> — first low-break after a ≥3 high-break streak<br>
        ③ <b>fixed stop</b> (per asset) — {_stops_line()}
      </div>
    </div>
  </div>
  <div style='margin-top:12px; font-size:11.5px; color:#7a5901;'>
    📈 <b>Out-of-sample 2021→now</b> (beats buy &amp; hold on return <i>and</i> drawdown):
    GDX <b>+272%</b> · MDD −16% · Sharpe 1.41 &nbsp;|&nbsp;
    UGL <i>(stop-less)</i> <b>+247%</b> · MDD −18% · Sharpe 1.37 &nbsp;|&nbsp;
    NUGT <i>(−5%)</i> <b>+1183%</b> · MDD −28% · Sharpe 1.48
  </div>
</div>""", unsafe_allow_html=True)


def render_conditions_box(sigs):
    """Dynamic checklist: every entry & exit condition with its live on/off
    state, and the single resolved net decision."""
    ns = net_signal(sigs)
    if not sigs:
        st.info("Strategy conditions unavailable — need ≥ 3 completed bars.")
        return

    def row(active, name, detail):
        ico = "✅" if active else "○"
        col = "#15803d" if active else "#94a3b8"
        weight = "700" if active else "500"
        return (f"<tr><td style='padding:3px 8px 3px 0;font-size:14px;'>{ico}</td>"
                f"<td style='padding:3px 10px 3px 0;font-weight:{weight};color:{col};"
                f"white-space:nowrap;'>{name}</td>"
                f"<td style='padding:3px 0;font-size:11.5px;color:#475569;'>{detail}</td></tr>")

    gate_bull = bool(sigs.get("bull_regime"))
    gate_clean = bool(sigs["clean_10d"] and not sigs["above_ma20"])
    gate_v = bool(sigs.get("v_recent_gate"))
    gate_any = gate_bull or gate_clean or gate_v
    entry_ready = sigs["u1_triggered"] and gate_any and not (sigs["d2_triggered"] or sigs["d3_triggered"])

    entry_html = "<table style='border-collapse:collapse;'>" + \
        row(sigs["u1_triggered"], "U1 trigger",
            f"err_hi 3d-avg = {sigs['err_hi_ma3']:+.3f}% (need &gt; +{gc.U1_ERRHI_MIN:.2f}%) "
            f"&amp; high-breaks = {sigs['hi_breaks_3d']}/3 (need ≥2)") + \
        row(gate_any, "Trend gate (any one)",
            f"{'🐂 Bull ' if gate_bull else ''}{'🧹 Clean ' if gate_clean else ''}"
            f"{'⚡ V-rev' if gate_v else ''}".strip() or "none active") + \
        "<tr><td></td><td colspan='2' style='padding-left:22px;font-size:11px;color:#64748b;'>" + \
        f"{'✓' if gate_bull else '○'} Bull Regime &nbsp; " + \
        f"{'✓' if gate_clean else '○'} Clean Breakout &nbsp; " + \
        f"{'✓' if gate_v else '○'} V-reversal</td></tr>" + \
        "</table>"

    exit_html = "<table style='border-collapse:collapse;'>" + \
        row(sigs["d2_triggered"], "D2 momentum fade",
            f"err_hi 3d-avg = {sigs['err_hi_ma3']:+.3f}% (exit &lt; {gc.D2_ERRHI_MAX:+.2f}%)") + \
        row(sigs["d3_triggered"], "D3 exhaustion",
            f"consec high-breaks = {sigs['consec_hi']} then a low-break") + \
        row(False, "Fixed stop (per asset)",
            _stops_line() + " — position-level, per open trade") + \
        "</table>"

    st.markdown(f"""
<div style='display:flex; gap:14px; flex-wrap:wrap; margin:2px 0 6px 0;'>
  <div style='flex:1; min-width:280px; background:#f0fdf4; border:1.5px solid #86efac;
       border-radius:10px; padding:10px 14px;'>
    <div style='font-weight:700; color:#15803d; margin-bottom:6px;'>
      📥 ENTRY conditions {'— ✅ ALL MET' if entry_ready else '— not met'}</div>
    {entry_html}
    <div style='font-size:11px;color:#64748b;margin-top:6px;'>Entry fires when
      <b>U1 AND one gate</b> are true and no exit is active.</div>
  </div>
  <div style='flex:1; min-width:280px; background:#fef2f2; border:1.5px solid #fca5a5;
       border-radius:10px; padding:10px 14px;'>
    <div style='font-weight:700; color:#b91c1c; margin-bottom:6px;'>
      📤 EXIT conditions {'— 🔴 ACTIVE' if (sigs['d2_triggered'] or sigs['d3_triggered']) else '— clear'}</div>
    {exit_html}
    <div style='font-size:11px;color:#64748b;margin-top:6px;'>Any one exit (or the stop)
      closes the position and <b>overrides</b> a simultaneous entry.</div>
  </div>
</div>
<div style='background:{ns['bg']}; border:2px solid {ns['brd']}; border-radius:10px;
     padding:10px 16px; margin:4px 0;'>
  <b style='font-size:15px;'>{ns['ico']} NET DECISION: {ns['label']}</b>
  <span style='color:#475569; font-size:12.5px; margin-left:8px;'>{ns['reason']}</span>
</div>""", unsafe_allow_html=True)


def render_gldm_signatures(sigs):
    if not sigs:
        st.info("Not enough completed bars for trend signatures yet (need ≥ 3).")
        return
    ns = net_signal(sigs)
    as_of = pd.Timestamp(sigs["as_of_date"]).strftime("%Y-%m-%d")
    # Single composite banner driven by the ONE resolved net signal (no separate
    # "entry active" text that could contradict an active exit).
    st.markdown(
        f"""<div style="background:{ns['bg']};border:2px solid {ns['brd']};
        border-radius:10px;padding:12px 16px;margin:8px 0;">
        <span style="background:{ns['brd']};color:white;font-weight:700;font-size:14px;
        padding:5px 14px;border-radius:20px;">{ns['ico']} {ns['label']}</span>
        <span style="color:#334155;font-size:13px;margin-left:10px;">
        <b>{sigs['dn_count']}/3</b> DN · <b>{sigs['up_count']}/1</b> UP ·
        raw flags: U1={'✓' if sigs['u1_triggered'] else '✗'}
        D2={'✓' if sigs['d2_triggered'] else '✗'}
        D3={'✓' if sigs['d3_triggered'] else '✗'}
        · bull_regime={'✓' if sigs.get('bull_regime') else '✗'}
        · as-of <b>{as_of}</b> · <b>{sigs['n_bars']}</b> bars</span></div>""",
        unsafe_allow_html=True)

    # 2×2 signature cards: U1 / D2 / D1 / D3 — each shows criteria + live values
    r1c1, r1c2 = st.columns(2)
    r2c1, r2c2 = st.columns(2)
    r1c1.markdown(_sig_card(
        "U1 — Bullish pressure", "📈", "#16a34a", sigs["u1_triggered"],
        [("err_hi 3d-avg", f"{sigs['err_hi_ma3']:+.3f}%", f"> +{gc.U1_ERRHI_MIN:.2f}%",
          sigs['err_hi_ma3'] > gc.U1_ERRHI_MIN),
         ("high-breaks (3d)", f"{sigs['hi_breaks_3d']}/3", "≥ 2", sigs['hi_breaks_3d'] >= 2)],
        "GLDM's actual highs keep beating the model's predicted highs → upside "
        "momentum. This is the entry trigger (needs a trend gate too)."),
        unsafe_allow_html=True)
    r1c2.markdown(_sig_card(
        "D2 — Momentum fading", "📉", "#dc2626", sigs["d2_triggered"],
        [("err_hi 3d-avg", f"{sigs['err_hi_ma3']:+.3f}%", f"< {gc.D2_ERRHI_MAX:+.2f}%",
          sigs['err_hi_ma3'] < gc.D2_ERRHI_MAX)],
        "Predicted highs are no longer being beaten — upside momentum is "
        "collapsing. Primary exit signal."), unsafe_allow_html=True)
    r2c1.markdown(_sig_card(
        "D1 — Downtrend pressure", "📉", "#dc2626", sigs["d1_triggered"],
        [("err_lo 3d-avg", f"{sigs['err_lo_ma3']:+.3f}%", f"> +{gc.D1_ERRLO_MIN:.2f}%",
          sigs['err_lo_ma3'] > gc.D1_ERRLO_MIN),
         ("low-breaks (3d)", f"{sigs['lo_breaks_3d']}/3", "≥ 2", sigs['lo_breaks_3d'] >= 2)],
        "GLDM's actual lows keep undershooting the predicted floor → the trend is "
        "deteriorating."), unsafe_allow_html=True)
    r2c2.markdown(_sig_card(
        "D3 — Exhaustion canary", "📉", "#dc2626", sigs["d3_triggered"],
        [("consec high-breaks", f"{sigs['consec_hi']}", "≥ 3 then a low-break",
          sigs['consec_hi'] >= 3),
         ("low-break today", "yes" if sigs['detail_rows'] and sigs['detail_rows'][-1]['lo_break'] else "no",
          "required", bool(sigs['exhaustion_active']))],
        "A first downside break after a run of upside breaks — classic blow-off / "
        "exhaustion reversal. Exit signal (fires even in a bull regime)."),
        unsafe_allow_html=True)

    if sigs.get("v_reversal_likely"):
        st.markdown("⚡ **V-reversal likely** — capitulation low undershoot detected "
                    "(a fresh V-reversal within 3 bars satisfies the entry trend gate).")

    # Last-5-bars mini-table (collapsible, all err columns) — like the BTC Live tab
    if sigs["detail_rows"]:
        with st.expander("📋 Last 5 bars — signal detail", expanded=False):
            st.caption(
                "Each row is a **completed** GLDM daily bar (actual H/L known); signals use "
                "these bars only. err_hi (bar) = (actual_high − pred_high)/close × 100 "
                "(regime-centered; + = bullish pressure). err_hi 3d-avg = rolling 3-bar mean "
                "(the bottom row equals the U1/D2 card value). Break = actual H > pred_H "
                "(Hi) or actual L < pred_L (Lo).")
            disp = []
            for r in sigs["detail_rows"]:
                disp.append({
                    "Date": pd.Timestamp(r["date"]).strftime("%Y-%m-%d"),
                    "Close": f"${r['close']:,.2f}",
                    "Pred H": f"${r['pred_hi']:,.2f}", "Actual H": f"${r['actual_hi']:,.2f}",
                    "err_hi (bar)": f"{r['err_hi_pct']:+.3f}%", "err_hi 3d-avg": f"{r['err_hi_ma3']:+.3f}%",
                    "Hi Brk": "✓" if r["hi_break"] else "–",
                    "Pred L": f"${r['pred_lo']:,.2f}", "Actual L": f"${r['actual_lo']:,.2f}",
                    "err_lo (bar)": f"{r['err_lo_pct']:+.3f}%", "err_lo 3d-avg": f"{r['err_lo_ma3']:+.3f}%",
                    "Lo Brk": "✓" if r["lo_break"] else "–",
                })
            st.dataframe(pd.DataFrame(disp[::-1]), hide_index=True, use_container_width=True)


def position_panel(asset, col_container, end=None):
    """Rich open-position / last-trade card for one traded asset — same look &
    fields as the BTC app's MSTR / MSTU panels (LONG green card, or CLOSED
    win/loss card, or a FLAT card)."""
    r = strategy_position(asset, end=end)
    label = ASSET_LABELS[asset]
    col = f"{asset.lower()}_close"
    if r is None:
        col_container.info(f"{label}: price series unavailable.")
        return
    if end is None:
        px = float(preds[col].iloc[-1]); as_of = pd.Timestamp(preds["target_date"].iloc[-1])
    else:
        sub = preds[preds["target_date"] <= pd.Timestamp(end)]
        px = float(sub[col].iloc[-1]); as_of = pd.Timestamp(end)

    def _tbl(rows):
        body = "".join(
            f"<tr><td style='color:#64748b;padding:1px 8px 1px 0;white-space:nowrap'>{k}</td>"
            f"<td style='font-weight:600'>{v}</td></tr>" for k, v in rows)
        return f"<table style='font-size:12px;color:#334155;width:100%;border-collapse:collapse'>{body}</table>"

    if r["in_pos_now"] and r["entry_px"]:
        e_px = r["entry_px"]; e_date = pd.Timestamp(r["entry_date"])
        upnl = (px / e_px - 1) * 100
        col_pnl = "#16a34a" if upnl >= 0 else "#dc2626"
        days = (as_of - e_date).days
        # per-asset stop: UGL (2× gold) trades signal-only (no fixed stop), so show
        # that instead of a −3%/$0.00 that doesn't apply to it.
        _stop = gc.stop_for(asset)
        _rows = [("Entry", f"{e_date.strftime('%b %d, %Y')} @ ${e_px:,.2f}"),
                 ("Trigger", "U1 + Pure-Regime gate"),
                 ("Live price", f"${px:,.2f}"),
                 ("Unrealized P&amp;L", f"<b style='color:{col_pnl}'>{upnl:+.2f}%</b>")]
        if _stop < 0.999:
            _rows.append(("Stop (−%.0f%%)" % (_stop * 100), f"${e_px * (1 - _stop):,.2f}"))
        else:
            _rows.append(("Exit", "signal-only — <b>no fixed stop</b>"))
        _rows.append(("Days held", f"{days}d"))
        html = (
            f"<div style='background:#f0fdf4;border:2px solid #16a34a;border-radius:10px;padding:12px 14px;'>"
            f"<div style='font-size:13px;font-weight:700;color:#15803d;margin-bottom:6px;'>"
            f"📍 {label} — LONG</div>" + _tbl(_rows) + "</div>")
        col_container.markdown(html, unsafe_allow_html=True)
        return

    # Flat: show the most recent CLOSED trade card if one exists
    tl = r.get("trade_log") or []
    if tl:
        lt = tl[-1]
        ret = lt["ret"] * 100; profit = ret > 0
        bg = "#f0fdf4" if profit else "#fef2f2"; brd = "#16a34a" if profit else "#dc2626"
        hdr = "#15803d" if profit else "#991b1b"; pcol = "#16a34a" if profit else "#dc2626"
        badge = "✅ CLOSED — PROFIT" if profit else "🔴 CLOSED — LOSS"
        e_date = pd.Timestamp(lt["entry_date"]); x_date = pd.Timestamp(lt["exit_date"])
        html = (
            f"<div style='background:{bg};border:2.5px solid {brd};border-radius:10px;padding:12px 14px;'>"
            f"<div style='font-size:12px;font-weight:700;color:{hdr};margin-bottom:6px;'>"
            f"{badge} — {label}</div>"
            f"<div style='background:#fff7ed;border:1.5px solid #ea580c;border-radius:6px;"
            f"padding:4px 9px;margin-bottom:7px;font-size:12px;font-weight:700;color:#9a3412;'>"
            f"📤 Exit: {lt['reason']}</div>"
            + _tbl([("Entry", f"{e_date.strftime('%b %d, %Y')} @ ${lt['entry_px']:,.2f}"),
                    ("Exit", f"{x_date.strftime('%b %d, %Y')} @ ${lt['exit_px']:,.2f}"),
                    ("Trade P&amp;L", f"<b style='color:{pcol}'>{ret:+.2f}%</b>"),
                    ("Days held", f"{(x_date - e_date).days}d"),
                    ("Now", f"⚪ FLAT · ${px:,.2f}, awaiting next entry")]) + "</div>")
        col_container.markdown(html, unsafe_allow_html=True)
    else:
        col_container.markdown(
            f"<div style='background:#f8fafc;border:2px solid #94a3b8;border-radius:10px;"
            f"padding:12px 14px;font-size:13px;'><b>⚪ {label} — FLAT</b><br>"
            f"<span style='color:#475569'>last close ${px:,.2f} · no trades yet, awaiting entry</span></div>",
            unsafe_allow_html=True)


# Shared plotly layout defaults so every chart matches the BTC app's look.
_PLOT_BG = dict(plot_bgcolor="#f8fafc", paper_bgcolor="#ffffff", hovermode="x unified")
_GRID = dict(showgrid=True, gridcolor="#e2e8f0", zeroline=False)
_HL_BAND_PCT = 0.006   # ±0.6% uncertainty tint (gold's daily range is ~⅓ of BTC's)


def _hl_forecast_fig(d_df, n_bars=8):
    """Daily H/L predictions-vs-actuals over the last N completed bars + the next
    forecast — same layout/colors as the BTC Live-tab H/L chart (green predicted
    HIGH line + tint, red predicted LOW line + tint, ✕ actual markers)."""
    sub = preds[preds["target_date"] <= d_df.index[-1]].tail(n_bars).copy()
    if sub.empty:
        return None
    hl = predict_next_daily_hl(d_df)
    if hl:
        nx = d_df.index[-1] + pd.tseries.offsets.BDay(1)
        sub = pd.concat([sub, pd.DataFrame([{
            "target_date": nx, "pred_high": hl["pred_high"], "pred_low": hl["pred_low"],
            "actual_high": np.nan, "actual_low": np.nan}])], ignore_index=True)
    x = pd.to_datetime(sub["target_date"])
    fig = go.Figure()
    # HIGH ±band (green tint)
    fig.add_trace(go.Scatter(x=x, y=sub["pred_high"] * (1 + _HL_BAND_PCT),
                             line=dict(color="rgba(34,139,34,0)"), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=sub["pred_high"] * (1 - _HL_BAND_PCT), fill="tonexty",
                             fillcolor="rgba(34,139,34,0.13)", line=dict(color="rgba(34,139,34,0)"),
                             name=f"HIGH ±{_HL_BAND_PCT*100:.1f}% band", hoverinfo="skip"))
    # LOW ±band (red tint)
    fig.add_trace(go.Scatter(x=x, y=sub["pred_low"] * (1 + _HL_BAND_PCT),
                             line=dict(color="rgba(220,20,60,0)"), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=sub["pred_low"] * (1 - _HL_BAND_PCT), fill="tonexty",
                             fillcolor="rgba(220,20,60,0.13)", line=dict(color="rgba(220,20,60,0)"),
                             name=f"LOW ±{_HL_BAND_PCT*100:.1f}% band", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=sub["pred_high"], mode="lines+markers",
                             line=dict(color="green", width=2.2, dash="dot"),
                             marker=dict(size=8), name="Predicted HIGH"))
    fig.add_trace(go.Scatter(x=x, y=sub["pred_low"], mode="lines+markers",
                             line=dict(color="red", width=2.2, dash="dot"),
                             marker=dict(size=8), name="Predicted LOW"))
    have = sub["actual_high"].notna()
    if have.any():
        fig.add_trace(go.Scatter(x=x[have], y=sub.loc[have, "actual_high"], mode="markers",
                                 marker=dict(symbol="x-thin", size=12, line=dict(width=3, color="darkgreen")),
                                 name="Actual HIGH"))
        fig.add_trace(go.Scatter(x=x[have], y=sub.loc[have, "actual_low"], mode="markers",
                                 marker=dict(symbol="x-thin", size=12, line=dict(width=3, color="darkred")),
                                 name="Actual LOW"))
    if hl:  # highlight the forward forecast point
        fig.add_trace(go.Scatter(x=[nx, nx], y=[hl["pred_low"], hl["pred_high"]],
                                 mode="markers", marker=dict(symbol="diamond", size=12, color="#b8860b"),
                                 name="Next-bar forecast"))
    fig.update_layout(height=340, margin=dict(l=0, r=10, t=44, b=0),
                      title=dict(text="📈 Daily H/L — predictions vs actuals (last bars + next forecast)",
                                 font=dict(size=13), x=0, xanchor="left"),
                      yaxis_title="Price ($)", yaxis_tickprefix="$", yaxis_tickformat=",.2f",
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
                      xaxis=_GRID, yaxis=_GRID, **_PLOT_BG)
    return fig


@st.cache_data(ttl=600, show_spinner=False)
def rolling_cone_series(as_of_iso, horizon, lookback):
    """Rolling H-day close prediction vs realized, for the cone charts.

    For each recent anchor bar (up to ``as_of_iso`` — so Historical Replay never
    leaks future data): predict the close ``horizon`` trading days out (central +
    P5/P95 band), map it to the target date, and attach the realized close if that
    date has already occurred. Mirrors the BTC rolling-cone chart.
    """
    art = M_7D if horizon == 7 else M_14D
    if art is None:
        return pd.DataFrame()
    d_df = daily[daily.index <= pd.Timestamp(as_of_iso)]
    close = d_df["gldm_close"]
    feat = gc.build_daily_features(d_df)[art["feat_cols"]].ffill()
    n = len(close); q = art["quantiles"]
    start = max(0, n - lookback - horizon)
    Xa = feat.iloc[start:]
    Xa = Xa[~Xa.isna().any(axis=1)]
    if Xa.empty:
        return pd.DataFrame()
    cr = art["model"].predict(Xa)
    rows = []
    for adate, c in zip(Xa.index, cr):
        pos = close.index.get_loc(adate)
        ac = float(close.iloc[pos]); tpos = pos + horizon
        pred = ac * np.exp(c)
        lo = ac * np.exp(c + q[5]); hi = ac * np.exp(c + q[95])
        if tpos < n:
            tdate = close.index[tpos]; realized = float(close.iloc[tpos]); future = False
        else:
            tdate = close.index[-1] + pd.tseries.offsets.BDay(tpos - (n - 1)); realized = None; future = True
        rows.append(dict(target_date=tdate, pred=pred, lo=lo, hi=hi,
                         realized=realized, future=future))
    return pd.DataFrame(rows)


def _cone_forecast_fig(as_of_iso, horizon, lookback, title, band_rgba, line_col):
    """Rolling H-day cone chart — realized close line, prediction band, and the
    forward forecast segment — styled like the BTC 7d/14d cone charts."""
    df = rolling_cone_series(as_of_iso, horizon, lookback)
    if df.empty:
        return None
    df["target_date"] = pd.to_datetime(df["target_date"])
    x = df["target_date"]
    band_pct = (df["hi"] / df["pred"] - 1).mean() * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=df["hi"], line=dict(color="rgba(0,0,0,0)"),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=df["lo"], fill="tonexty", fillcolor=band_rgba,
                             line=dict(color="rgba(0,0,0,0)"),
                             name=f"±{band_pct:.1f}% (P5–P95) band", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=df["pred"], mode="lines",
                             line=dict(color=line_col, width=1.8, dash="dot"),
                             name=f"{horizon}d prediction"))
    res = df[df["realized"].notna()]
    if not res.empty:
        within = (res["realized"] >= res["lo"]) & (res["realized"] <= res["hi"])
        fig.add_trace(go.Scatter(x=res["target_date"], y=res["realized"], mode="lines",
                                 line=dict(color="#1f2937", width=2.2), name="Realized close"))
        if within.any():
            fig.add_trace(go.Scatter(x=res.loc[within, "target_date"], y=res.loc[within, "realized"],
                                     mode="markers", marker=dict(symbol="circle", size=6, color="#16a34a",
                                     line=dict(color="white", width=1)), name="✅ within band"))
        if (~within).any():
            fig.add_trace(go.Scatter(x=res.loc[~within, "target_date"], y=res.loc[~within, "realized"],
                                     mode="markers", marker=dict(symbol="x", size=8, color="#dc2626",
                                     line=dict(color="#dc2626", width=2)), name="❌ outside band"))
    fut = df[df["future"]]
    if not fut.empty:
        fig.add_trace(go.Scatter(x=fut["target_date"], y=fut["pred"], mode="lines+markers",
                                 line=dict(color=line_col, width=2.4),
                                 marker=dict(symbol="diamond", size=8, color=line_col,
                                             line=dict(color="white", width=1)),
                                 name=f"+{horizon}d forecast"))
    within_pct = (((res["realized"] >= res["lo"]) & (res["realized"] <= res["hi"])).mean() * 100
                  if not res.empty else np.nan)
    subtitle = f"  ·  {within_pct:.0f}% of realized closes within band" if not np.isnan(within_pct) else ""
    fig.update_layout(height=320, margin=dict(l=0, r=10, t=44, b=0),
                      title=dict(text=title + subtitle, font=dict(size=13), x=0, xanchor="left"),
                      yaxis_title="Close ($)", yaxis_tickprefix="$", yaxis_tickformat=",.2f",
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
                      xaxis=_GRID, yaxis=_GRID, **_PLOT_BG)
    return fig


_HR_LOOKBACK_HOURS = 23   # rolling look-back window (hours). 23h actuals + ~1h
                          # forecast ≈ a 24h span, matching the BTC daily bar view.


def _hourly_forecast_fig(as_of_date, is_live, hl=None):
    """Rolling next-hour GLDM close forecast — a faithful gold copy of the BTC
    hourly chart: the last 23 hours of bars with the actual close (black), the
    per-bar past predictions (● purple line, colored green=correct-direction /
    red=miscalled), the realized closes (◆ teal line), a 95% CI tint, a khaki
    forecast zone with a darkorange connector to the ⭐ forecast star (±95% CI
    error bars), and a crimson 'now' line."""
    hourly = get_hourly()
    if hourly is None or hourly.empty or M_HOURLY is None:
        return None
    if not is_live and as_of_date is not None:
        cutoff = pd.Timestamp(as_of_date) + pd.Timedelta(hours=23, minutes=59)
        hourly = hourly[hourly.index <= cutoff]
    if len(hourly) < 30:
        return None

    close = hourly["gldm_close"]
    feat = gc.build_hourly_features(hourly)
    cols = M_HOURLY["feat_cols"]
    X = feat[cols]
    valid = X.index[~X.isna().any(axis=1)]
    if len(valid) < 5:
        return None
    # Rolling window: keep only bars from the last 23 hours (plus the anchor).
    anchor = valid[-1]
    look = valid[valid >= anchor - pd.Timedelta(hours=_HR_LOOKBACK_HOURS)]
    if len(look) < 3:                            # sparse data → keep a few bars
        look = valid[-8:]
    sigma = float(M_HOURLY["sigma"])
    yhat = M_HOURLY["model"].predict(X.loc[look])

    # Per-bar past predictions: bar look[k] predicts the NEXT bar's close.
    tgt_ts, pred_c, act_c, mcol = [], [], [], []
    for k in range(len(look) - 1):
        b = look[k]; nb = look[k + 1]
        c0 = float(close.loc[b]); pc = c0 * np.exp(yhat[k]); ac = float(close.loc[nb])
        tgt_ts.append(nb); pred_c.append(pc); act_c.append(ac)
        correct = np.sign(yhat[k]) == np.sign(np.log(ac / c0))
        mcol.append("seagreen" if correct else "indianred")
    tgt_ts = pd.DatetimeIndex(tgt_ts)

    last_ts = look[-1]; last_close = float(close.loc[last_ts])
    # nominal next-bar timestamp = median spacing of recent bars (market-hours aware)
    if len(valid) >= 3:
        step = pd.Series(valid[-6:]).diff().median()
        step = step if pd.notna(step) else pd.Timedelta(hours=1)
    else:
        step = pd.Timedelta(hours=1)
    next_ts = last_ts + step
    fc = last_close * np.exp(yhat[-1])
    fc_ret = float(yhat[-1])
    fc_hi = last_close * np.exp(yhat[-1] + 1.96 * sigma)
    fc_lo = last_close * np.exp(yhat[-1] - 1.96 * sigma)

    # ── convert every x-value to US Central Time (DST-aware) for display ──
    # Data timestamps are tz-naive UTC; localize → convert → drop tz. Same as BTC.
    def _ct(ts):
        idx = pd.DatetimeIndex(pd.to_datetime(np.atleast_1d(ts)))
        return idx.tz_localize("UTC").tz_convert("America/Chicago").tz_localize(None)
    tgt_ct = _ct(tgt_ts)
    act_win = close.loc[look]
    act_ct = _ct(act_win.index)
    last_ct = _ct(last_ts)[0]; next_ct = _ct(next_ts)[0]

    fig = go.Figure()
    # ±95% CI band around the past predictions (blue tint) — matches BTC band
    fig.add_trace(go.Scatter(x=tgt_ct, y=np.array(pred_c) * np.exp(1.96 * sigma),
                             line=dict(color="rgba(65,105,225,0)"), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=tgt_ct, y=np.array(pred_c) * np.exp(-1.96 * sigma), fill="tonexty",
                             fillcolor="rgba(65,105,225,0.18)", line=dict(color="rgba(65,105,225,0)"),
                             name=f"Pred ±{1.96*sigma*100:.2f}% (95% CI) band", hoverinfo="skip"))
    # Actual close line (last 23h of bars) — black
    fig.add_trace(go.Scatter(x=act_ct, y=act_win.values, mode="lines",
                             line=dict(color="black", width=2), name="Actual close (hourly)",
                             hovertemplate="%{x|%d-%b %H:%M} CT<br>$%{y:,.2f}<extra></extra>"))
    # Past predictions — purple line, ● markers colored by direction correctness
    fig.add_trace(go.Scatter(x=tgt_ct, y=pred_c, mode="lines+markers",
                             line=dict(color="#7c3aed", width=2),
                             marker=dict(color=mcol, size=8, symbol="circle", line=dict(width=1, color="white")),
                             name="● Past hourly predictions (green = correct dir.)",
                             hovertemplate="Past pred for %{x|%d-%b %H:%M} CT<br>$%{y:,.2f}<extra></extra>"))
    # Realized closes — teal line, ◆ markers colored by direction correctness
    fig.add_trace(go.Scatter(x=tgt_ct, y=act_c, mode="lines+markers",
                             line=dict(color="#0d9488", width=2),
                             marker=dict(color=mcol, size=10, symbol="diamond", line=dict(color="white", width=1.5)),
                             name="◆ Hourly realized close",
                             hovertemplate="Realized %{x|%d-%b %H:%M} CT<br>$%{y:,.2f}<extra></extra>"))
    # Forecast zone (khaki) + darkorange connector + ⭐ star with 95% CI error bars
    fig.add_vrect(x0=last_ct, x1=next_ct, fillcolor="khaki", opacity=0.30, line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=[last_ct, next_ct], y=[last_close, fc], mode="lines",
                             line=dict(color="darkorange", width=2), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[next_ct], y=[fc], mode="markers",
                             marker=dict(symbol="star", size=16, color="darkorange", line=dict(width=1, color="white")),
                             error_y=dict(type="data", array=[fc_hi - fc], arrayminus=[fc - fc_lo],
                                          color="darkorange", thickness=1.5, width=6),
                             name="⭐ Next-hour forecast",
                             hovertemplate=(f"Forecast {next_ct:%d-%b %H:%M} CT<br>"
                                            f"$%{{y:,.2f}} ({fc_ret*100:+.2f}%)<br>"
                                            f"95% CI ${fc_lo:,.2f} – ${fc_hi:,.2f}<extra></extra>")))
    fig.add_vline(x=last_ct, line=dict(color="crimson", width=1.5, dash="dash"))

    # ── overlay the predicted daily HIGH / LOW as full-width threshold lines
    #    with ±band, clipped at the midpoint (same as the BTC hourly chart) ──
    if hl:
        fig.add_hline(y=hl["pred_high"], line=dict(color="green", width=2.5, dash="dot"),
                      annotation_text=f"Daily Pred HIGH ${hl['pred_high']:,.2f}",
                      annotation_position="top right", annotation_font=dict(color="green", size=12),
                      annotation_bgcolor="rgba(255,255,255,0.92)", annotation_bordercolor="green",
                      annotation_borderwidth=1)
        fig.add_hline(y=hl["pred_low"], line=dict(color="red", width=2.5, dash="dot"),
                      annotation_text=f"Daily Pred LOW ${hl['pred_low']:,.2f}",
                      annotation_position="bottom right", annotation_font=dict(color="red", size=12),
                      annotation_bgcolor="rgba(255,255,255,0.92)", annotation_bordercolor="red",
                      annotation_borderwidth=1)
        mid = (hl["pred_high"] + hl["pred_low"]) / 2
        hi_dn = max(hl["pred_high"] - hl["band_hi"], mid)   # clip green band at mid
        lo_up = min(hl["pred_low"] + hl["band_lo"], mid)    # clip red band at mid
        if hl["pred_high"] + hl["band_hi"] > hi_dn:
            fig.add_hrect(y0=hi_dn, y1=hl["pred_high"] + hl["band_hi"],
                          fillcolor="rgba(0,170,0,0.12)", line_width=0, layer="below")
        if lo_up > hl["pred_low"] - hl["band_lo"]:
            fig.add_hrect(y0=hl["pred_low"] - hl["band_lo"], y1=lo_up,
                          fillcolor="rgba(220,30,30,0.12)", line_width=0, layer="below")

    title = ("🕐 GLDM — rolling next-hour close forecast (last 23 hours)"
             if is_live else f"🕐 GLDM hourly forecast as of {pd.Timestamp(as_of_date).date()}")
    fig.update_layout(template="plotly_white", height=420,
                      title=dict(text=title, font=dict(size=13), x=0, xanchor="left"),
                      yaxis_title="GLDM / USD", yaxis_tickprefix="$", yaxis_tickformat=",.2f",
                      margin=dict(l=0, r=10, t=46, b=0),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
                      xaxis=dict(title="Time (US Central)", tickformat="%d-%b %H:%M", **_GRID),
                      yaxis=_GRID, **_PLOT_BG)
    return fig


def render_prediction_plots(d_df, key_prefix, is_live=True, as_of_date=None):
    """Hourly + Daily H/L + 7-day + 14-day forecast charts (Live & Replay)."""
    st.markdown("### 🔮 Model forecast charts (GLDM)")
    hl = predict_next_daily_hl(d_df)   # overlaid on the hourly chart as H/L lines
    fhr = _hourly_forecast_fig(as_of_date, is_live, hl=hl)
    if fhr:
        st.plotly_chart(fhr, use_container_width=True, key=f"{key_prefix}_hr")
    else:
        st.caption("_Hourly forecast unavailable for this date (hourly history starts ~Aug 2023)._")
    fhl = _hl_forecast_fig(d_df)
    if fhl:
        st.plotly_chart(fhl, use_container_width=True, key=f"{key_prefix}_hl")
    as_of_iso = str(d_df.index[-1])
    f7 = _cone_forecast_fig(as_of_iso, 7, 90, "📅 7-day close-price cone", "rgba(147,197,253,0.28)", "#2563eb")
    f14 = _cone_forecast_fig(as_of_iso, 14, 120, "📆 14-day close-price cone", "rgba(196,181,253,0.30)", "#7c3aed")
    if f7:
        st.plotly_chart(f7, use_container_width=True, key=f"{key_prefix}_c7")
    if f14:
        st.plotly_chart(f14, use_container_width=True, key=f"{key_prefix}_c14")


# ════════════════════════════════════════════════════════════════════════
# Backtesting dashboard (mirrors BTC/MSTR backtesting tabs)
# ════════════════════════════════════════════════════════════════════════
# Four periods, mirroring the BTC/MSTR tabs.  NOTE: the daily H/L signal model
# is fit only on pre-2021 data, so EVERY window below is genuinely out-of-sample
# — the "OOS — Recent" tab simply isolates the most recent fully-blind slice.
_PERIODS = [
    ("🐻 Choppy / Bear (2021–2022)", "2021-01-01", "2022-12-31"),
    ("🐂 Bull Market (2023 → now)", "2023-01-01", None),
    ("🌐 Full Market (2021 → now)", "2021-01-01", None),
    ("🔬 OOS — Recent (2025 → now)", "2025-01-01", None),
]


def _equity_fig(r, asset, title):
    """Strategy-vs-B&H equity curve with entry (▲) and exit (▼ win/loss) markers,
    plus a marker for a still-open position — same idea as the BTC/MSTR plot."""
    INIT = 100_000.0
    dts = pd.to_datetime(r["dates"])
    nav = r["strat"] * INIT; bh = r["bh"] * INIT
    date_to_nav = dict(zip([pd.Timestamp(d) for d in r["dates"]], nav))
    fig = go.Figure()

    # ── regime shading (gold bull vs bear/neutral) — like the BTC chart ──
    reg = pd.Series(sig["bull_regime"], index=pd.to_datetime(preds["target_date"]))
    reg = reg.reindex(dts).ffill().fillna(False).values.astype(bool)
    if len(reg):
        changes = np.where(np.diff(reg.astype(int)) != 0)[0] + 1
        starts = np.concatenate([[0], changes]); ends = np.concatenate([changes, [len(reg)]])
        bull_leg = bear_leg = False
        for s_i, e_i in zip(starts, ends):
            is_bull = bool(reg[s_i])
            show = (is_bull and not bull_leg) or (not is_bull and not bear_leg)
            fig.add_vrect(x0=dts[s_i], x1=dts[min(e_i, len(dts) - 1)],
                          fillcolor="rgba(34,197,94,0.10)" if is_bull else "rgba(239,68,68,0.07)",
                          line_width=0, name="🐂 Bull Regime (GLDM)" if is_bull else "🐻 Bear/Neutral (GLDM)",
                          showlegend=show)
            bull_leg = bull_leg or is_bull; bear_leg = bear_leg or (not is_bull)

    fig.add_trace(go.Scatter(x=dts, y=bh, name=f"Buy & Hold {asset}",
                             line=dict(color="#94a3b8", width=1.5, dash="dot"),
                             hovertemplate="%{x|%b %d, %Y}: $%{y:,.0f}<extra>Buy & Hold</extra>"))
    fig.add_trace(go.Scatter(x=dts, y=nav, name=f"{gc.STRATEGY_NAME} ({asset})",
                             line=dict(color="#b8860b", width=2.5),
                             hovertemplate="%{x|%b %d, %Y}: $%{y:,.0f}<extra>Strategy</extra>"))
    # GLDM signal-source price on a secondary axis (like BTC price on the MSTR chart)
    gseries = preds.set_index(pd.to_datetime(preds["target_date"]))["gldm_close"].reindex(dts)
    fig.add_trace(go.Scatter(x=dts, y=gseries.values, name="GLDM price (signal)",
                             line=dict(color="#2563eb", width=1.2), yaxis="y2", opacity=0.7,
                             hovertemplate="%{x|%b %d, %Y}: $%{y:,.2f}<extra>GLDM</extra>"))
    fig.add_hline(y=INIT, line_dash="dash", line_color="#64748b", line_width=1, opacity=0.4,
                  annotation_text="  $100k start", annotation_position="bottom right")

    # entry / exit markers (▲ green entry · ▼ green|red exit)
    for t in r["trade_log"]:
        ed = pd.Timestamp(t["entry_date"]); xd = pd.Timestamp(t["exit_date"])
        win_col = "#16a34a" if t["ret"] > 0 else "#dc2626"
        if ed in date_to_nav:
            fig.add_trace(go.Scatter(x=[ed], y=[date_to_nav[ed]], mode="markers", showlegend=False,
                                     marker=dict(symbol="triangle-up", size=12, color="#16a34a",
                                                 line=dict(width=1.5, color="white")),
                                     hovertemplate=f"<b>BUY {asset}</b> {ed:%b %d} @ ${t['entry_px']:,.2f}<extra></extra>"))
        if xd in date_to_nav:
            fig.add_trace(go.Scatter(x=[xd], y=[date_to_nav[xd]], mode="markers", showlegend=False,
                                     marker=dict(symbol="triangle-down", size=12, color=win_col,
                                                 line=dict(width=1.5, color="white")),
                                     hovertemplate=(f"<b>SELL {asset}</b> {xd:%b %d} @ ${t['exit_px']:,.2f} "
                                                    f"({t['ret']*100:+.1f}%) — {t['reason']}<extra></extra>")))
    if r.get("in_pos_now") and r.get("entry_date") is not None:
        ed = pd.Timestamp(r["entry_date"])
        if ed in date_to_nav:
            fig.add_trace(go.Scatter(x=[ed], y=[date_to_nav[ed]], mode="markers", name="Open entry",
                                     marker=dict(symbol="triangle-up", size=14, color="#f59e0b",
                                                 line=dict(width=1.5, color="white"))))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13), x=0, xanchor="left"),
        height=380, margin=dict(l=0, r=70, t=54, b=0),
        yaxis_title="Portfolio value ($)", yaxis_tickprefix="$", yaxis_tickformat=",.0f",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
        xaxis=dict(domain=[0.0, 0.9], **_GRID), yaxis=_GRID,
        yaxis2=dict(title="GLDM ($)", overlaying="y", side="right", showgrid=False,
                    zeroline=False, anchor="x", tickprefix="$", tickformat=",.2f", color="#2563eb"),
        **_PLOT_BG)
    return fig


def _trade_log_table(r, asset):
    """BTC-style trade log: #, entry/exit dates & prices, P&L, emoji Result,
    days, exit signal, and running NAV After (start $100k)."""
    INIT = 100_000.0
    rows = []
    nav = INIT
    for i, t in enumerate(r["trade_log"], 1):
        nav *= (1 + t["ret"])
        if "stop" in t["reason"]:
            result = "🛑 SL EXIT"
        elif t["ret"] > 0:
            result = "✓ WIN"
        else:
            result = "✗ LOSS"
        rows.append({
            "#": i,
            "Entry": pd.Timestamp(t["entry_date"]).strftime("%b %d, %Y"),
            f"Buy {asset} @": f"${t['entry_px']:,.2f}",
            "Exit": pd.Timestamp(t["exit_date"]).strftime("%b %d, %Y"),
            f"Sell {asset} @": f"${t['exit_px']:,.2f}",
            "P&L": f"{t['ret']*100:+.1f}%",
            "Result": result,
            "Days": (pd.Timestamp(t["exit_date"]) - pd.Timestamp(t["entry_date"])).days,
            "Exit signal": t["reason"],
            "NAV After": f"${nav:,.0f}",
        })
    if r.get("in_pos_now") and r.get("entry_px"):
        last_px = float(preds[f"{asset.lower()}_close"].iloc[-1]) if f"{asset.lower()}_close" in preds else None
        unr = (last_px / r["entry_px"] - 1) * 100 if last_px else 0.0
        rows.append({
            "#": len(r["trade_log"]) + 1,
            "Entry": pd.Timestamp(r["entry_date"]).strftime("%b %d, %Y"),
            f"Buy {asset} @": f"${r['entry_px']:,.2f}",
            "Exit": "⏳ OPEN", f"Sell {asset} @": "—",
            "P&L": f"{unr:+.1f}% (unrlzd)", "Result": "🟡 OPEN",
            "Days": (pd.Timestamp(preds['target_date'].iloc[-1]) - pd.Timestamp(r["entry_date"])).days,
            "Exit signal": "—", "NAV After": "—",
        })
    if not rows:
        st.info("No trades in this period — strategy was in cash throughout.")
        return
    st.dataframe(pd.DataFrame(rows[::-1]), use_container_width=True, hide_index=True, height=300)
    tr = r["trades"]
    if len(tr):
        wins = tr[tr > 0]
        st.caption(f"💡 {len(tr)} trades · win rate {(tr>0).mean()*100:.0f}% · "
                   f"avg win {wins.mean()*100 if len(wins) else 0:.2f}% · "
                   f"avg loss {tr[tr<=0].mean()*100 if (tr<=0).any() else 0:.2f}% · "
                   f"best {tr.max()*100:+.1f}% · worst {tr.min()*100:+.1f}%. "
                   "P&L at execution prices; entry/exit from the GLDM signal.")


def _metrics_table_html(asset):
    """BTC-style colored metrics table: periods × (Strategy / Buy&Hold), with the
    Strategy cell green when it beats B&H on that metric, red when worse."""
    col = f"{asset.lower()}_close"

    def cell(txt, good=None, bold=False):
        color = "#16a34a" if good is True else "#dc2626" if good is False else "#334155"
        w = "700" if (bold or good is not None) else "500"
        return f"<td style='padding:6px 10px;text-align:center;color:{color};font-weight:{w};'>{txt}</td>"

    metric_rows = [("Total Return", "ret", True), ("CAGR", "cagr", True),
                   ("Max Drawdown", "mdd", True), ("Sharpe", "sharpe", True),
                   ("Win Rate", "wr", None), ("Trades", "n", None)]
    per = []
    for lbl, s, e in _PERIODS:
        r = btg.simulate(preds, sig, col, gc.stop_for(asset), gc.U1_ERRHI_MIN,
                         gc.D2_ERRHI_MAX, gc.D1_ERRLO_MIN, oos_start=s, end=e)
        sm = btg._metrics(r["strat"], r["dates"]); bm = btg._metrics(r["bh"], r["dates"])
        wr = (r["trades"] > 0).mean() * 100 if len(r["trades"]) else 0
        per.append((lbl, sm, bm, wr, len(r["trades"])))

    # header
    hdr = "<tr style='background:#7a5901;color:white;'><th style='padding:9px 10px;text-align:left;'>Metric</th>"
    for i, (lbl, *_ ) in enumerate(per):
        bg = "#14532d" if "OOS" in lbl else "#7a5901"
        hdr += (f"<th colspan='2' style='padding:9px 8px;text-align:center;background:{bg};"
                f"border-left:3px solid #fff;'>{lbl}</th>")
    hdr += "</tr>"
    sub = "<tr style='background:#a67c00;color:white;font-size:11px;'><th></th>"
    for lbl, *_ in per:
        bg = "#166534" if "OOS" in lbl else "#a67c00"
        sub += (f"<th style='padding:4px 8px;background:{bg};border-left:3px solid #fff;'>Strategy</th>"
                f"<th style='padding:4px 8px;background:{bg};'>Buy&amp;Hold</th>")
    sub += "</tr>"

    body = ""
    for ri, (lbl_m, key, higher_better) in enumerate(metric_rows):
        bg = "#fffaf0" if ri % 2 == 0 else "#ffffff"
        body += f"<tr style='background:{bg};'><td style='padding:6px 10px;font-weight:600;color:#334155;'>{lbl_m}</td>"
        for lbl, sm, bm, wr, ntr in per:
            if key == "ret":
                sv, bv = sm["total_ret"] * 100, bm["total_ret"] * 100
                good = sv > bv; body += cell(f"{sv:+.1f}%", good) + cell(f"{bv:+.1f}%")
            elif key == "cagr":
                sv, bv = sm["cagr"] * 100, bm["cagr"] * 100
                good = sv > bv; body += cell(f"{sv:+.1f}%", good) + cell(f"{bv:+.1f}%")
            elif key == "mdd":
                sv, bv = sm["mdd"] * 100, bm["mdd"] * 100
                good = sv > bv; body += cell(f"{sv:.1f}%", good) + cell(f"{bv:.1f}%")
            elif key == "sharpe":
                sv, bv = sm["sharpe"], bm["sharpe"]
                good = sv > bv; body += cell(f"{sv:.2f}", good) + cell(f"{bv:.2f}")
            elif key == "wr":
                body += cell(f"{wr:.0f}%") + cell("—")
            else:
                body += cell(f"{ntr}") + cell("—")
        body += "</tr>"

    return (f"<div style='overflow-x:auto;margin:8px 0;'>"
            f"<table style='width:100%;border-collapse:collapse;font-size:13px;"
            f"border:1px solid #e2e8f0;border-radius:8px;overflow:hidden;'>"
            f"<thead>{hdr}{sub}</thead><tbody>{body}</tbody></table></div>"
            f"<p style='font-size:11px;color:#64748b;margin:2px 0 10px;'>"
            f"🟢 green = Strategy beats Buy&amp;Hold on that metric · 🔴 red = worse · "
            f"Max Drawdown closer to 0 is better. All windows are out-of-sample.</p>")


def render_backtest_dashboard(asset):
    col = f"{asset.lower()}_close"
    st.markdown(f"## {'📊' if asset == 'GDX' else '📈'} {ASSET_LABELS[asset]} — "
                "Gold Signal-Driven Backtesting")
    render_strategy_card()
    st.caption("All trades are out-of-sample: the GLDM daily H/L signal model is fit once "
               "on the pre-2021 window and predicts every later bar, so all four periods "
               "below are genuinely blind. NAV starts at $100k; costs/slippage not modelled.")

    # ── colored period × Strategy/B&H metrics table (BTC-style) ──
    st.markdown(_metrics_table_html(asset), unsafe_allow_html=True)

    # ── one chart+log tab per period (mirrors the BTC/MSTR period tabs) ──
    period_tabs = st.tabs([lbl for lbl, _, _ in _PERIODS])
    for (lbl, s, e), tb in zip(_PERIODS, period_tabs):
        with tb:
            r = btg.simulate(preds, sig, col, gc.stop_for(asset), gc.U1_ERRHI_MIN,
                             gc.D2_ERRHI_MAX, gc.D1_ERRLO_MIN, oos_start=s, end=e)
            if len(r["strat"]) < 2:
                st.info("Not enough bars in this window.")
                continue
            sm = btg._metrics(r["strat"], r["dates"]); bm = btg._metrics(r["bh"], r["dates"])
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Strategy return", f"{sm['total_ret']*100:+.1f}%", f"CAGR {sm['cagr']*100:+.1f}%")
            k2.metric("Buy & Hold", f"{bm['total_ret']*100:+.1f}%", f"CAGR {bm['cagr']*100:+.1f}%")
            k3.metric("Max drawdown", f"{sm['mdd']*100:.1f}%", f"vs B&H {bm['mdd']*100:.1f}%",
                      delta_color="inverse")
            k4.metric("Sharpe", f"{sm['sharpe']:.2f}", f"vs B&H {bm['sharpe']:.2f}")

            st.plotly_chart(_equity_fig(r, asset, f"{gc.STRATEGY_NAME} ({asset}) vs Buy & Hold — {lbl}"),
                            use_container_width=True, key=f"{asset}_{s}_{e}_eq")
            dts = pd.to_datetime(r["dates"])
            figd = go.Figure()
            figd.add_trace(go.Scatter(x=dts, y=btg.drawdown_series(r["strat"]) * 100,
                                      name="Strategy DD", fill="tozeroy", line=dict(color="#b8860b")))
            figd.add_trace(go.Scatter(x=dts, y=btg.drawdown_series(r["bh"]) * 100,
                                      name="Buy & Hold DD", line=dict(color="#94a3b8", dash="dot")))
            figd.update_layout(height=190, margin=dict(l=0, r=70, t=6, b=0),
                               yaxis_title="Drawdown %", yaxis_ticksuffix="%",
                               legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
                               xaxis=dict(domain=[0.0, 0.9], **_GRID), yaxis=_GRID, **_PLOT_BG)
            st.plotly_chart(figd, use_container_width=True, key=f"{asset}_{s}_{e}_dd")
            st.markdown("#### 📋 Trade log")
            _trade_log_table(r, asset)


# ════════════════════════════════════════════════════════════════════════
# Tabs
# ════════════════════════════════════════════════════════════════════════
tab_live, tab_hist, tab_gdx, tab_ugl, tab_nugt, tab_explain = st.tabs(
    ["🔴 Live (rolling now+1h)", "🕒 Historical replay",
     "📊 GDX Backtesting", "📈 UGL Backtesting", "⛏️ NUGT Backtesting", "🧠 Explain"])


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

    def _px_chg(col):
        """(last, day-change%) for a price column in d_df, or (None,None)."""
        if col not in d_df or d_df[col].dropna().shape[0] < 2:
            return None, None
        s = d_df[col].dropna()
        return float(s.iloc[-1]), (float(s.iloc[-1]) / float(s.iloc[-2]) - 1) * 100

    gdx_px, gdx_chg = _px_chg("gdx_close")
    ugl_px, ugl_chg = _px_chg("ugl_close")

    # ── Row 1: the three asset prices + sentiment + regime ──
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("GLDM (gold) close", f"${last_px:,.2f}", f"{(last_px/prev_px-1)*100:+.2f}% d/d")
    c2.metric("GDX · miners (traded)", f"${gdx_px:,.2f}" if gdx_px else "—",
              f"{gdx_chg:+.2f}% d/d" if gdx_chg is not None else None)
    c3.metric("UGL · 2× gold (traded)", f"${ugl_px:,.2f}" if ugl_px else "—",
              f"{ugl_chg:+.2f}% d/d" if ugl_chg is not None else None)
    if not np.isnan(sent_now):
        mood = ("Bullish" if sent_now >= 60 else "Bearish" if sent_now <= 40 else "Neutral")
        c4.metric("Gold macro sentiment", f"{sent_now:.0f}/100", mood)
    else:
        c4.metric("Gold macro sentiment", "—")
    _bull = sigs.get("bull_regime") if sigs else None
    c5.metric("Market regime", ("🐂 BULL" if _bull else "🐻 BEAR/NEUTRAL") if _bull is not None else "—",
              delta=("↑MA20 & rising" if _bull else "below MA20 or flat") if _bull is not None else None,
              delta_color="normal" if _bull else "inverse")

    # ── Row 2: GLDM next-hour forecast + CI + MA20 + predicted daily H/L ──
    d1c, d2c, d3c, d4c, d5c = st.columns(5)
    if ph:
        d1c.metric("GLDM next-hour (pred)", f"${ph['pred_close']:,.2f}", f"{ph['ret']*100:+.2f}%")
        d2c.metric("95% CI band", f"${ph['lo']:,.2f}–${ph['hi']:,.2f}", f"±{1.96*ph['sigma']*100:.2f}%")
    else:
        d1c.metric("GLDM next-hour (pred)", "— (replay)")
        d2c.metric("95% CI band", "—")
    _ma = sigs.get("ma20_value") if sigs else None
    d3c.metric("GLDM 20-day MA", f"${_ma:,.2f}" if _ma else "—",
               (f"${last_px-_ma:+,.2f} vs close" if _ma else None),
               delta_color="normal" if (_ma and last_px >= _ma) else "inverse")
    if hl:
        d4c.metric("GLDM daily High (pred)", f"${hl['pred_high']:,.2f}",
                   f"{(hl['pred_high']/hl['last_close']-1)*100:+.2f}% vs close")
        d5c.metric("GLDM daily Low (pred)", f"${hl['pred_low']:,.2f}",
                   f"{(hl['pred_low']/hl['last_close']-1)*100:+.2f}% vs close")
        if dt:
            d4c.caption(f"Day-type: **{dt['top']}** ({max(dt['probs'].values())*100:.0f}%)")
    else:
        d4c.metric("GLDM daily High (pred)", "—"); d5c.metric("GLDM daily Low (pred)", "—")

    # ── trend-signature alert (the BTC-style card block) ──
    st.markdown("### 🔔 Trend-Signature Alert  ·  _signals derived from the GLDM daily H/L model_")
    render_gldm_signatures(sigs)
    st.markdown("#### 🚪 Entry-gate conditions  ·  _what turns a U1 pressure signal into an actual entry_")
    render_gldm_gate_signatures(sigs)

    # ── strategy description card + live conditions checklist + positions ──
    st.markdown(f"### 🎯 Strategy — {gc.STRATEGY_NAME}")
    render_strategy_card()
    st.markdown("#### Strategy conditions (live)")
    render_conditions_box(sigs)
    st.markdown("#### Current positions")
    p1, p2, p3 = st.columns(3)
    end = None if is_live else as_of_date
    position_panel("GDX", p1, end=end)
    position_panel("UGL", p2, end=end)
    position_panel("NUGT", p3, end=end)

    # ── model forecast charts (hourly + Daily H/L + 7d & 14d close cones) ──
    st.markdown("---")
    render_prediction_plots(d_df, key_prefix=("live" if is_live else "hist"),
                            is_live=is_live, as_of_date=as_of_date)


with tab_live:
    render_live_dashboard(is_live=True)


# ═════════════════════════ HISTORICAL REPLAY ═════════════════════════════
with tab_hist:
    st.markdown("### 🕒 Historical replay — GDX, UGL & NUGT")
    st.caption("Replay the gold trend-signals, forecasts and all three positions exactly as "
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
with tab_nugt:
    render_backtest_dashboard("NUGT")


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
is in **GDX** (miners), **UGL** (2× gold) and **NUGT** (2× gold miners) — the 1×
GLDM position is not traded, mirroring how the BTC app trades MSTR / MSTU rather
than spot BTC.

**Out-of-sample results (2021→now)** — beats buy & hold on **both** return and
drawdown for every sleeve:

| Asset | Strategy | Buy & Hold | Strat MDD | B&H MDD | Sharpe |
|---|---|---|---|---|---|
| GDX | **+272%** | +97% | **−16%** | −47% | **1.41** |
| UGL | **+211%** | +158% | **−18%** | −49% | **1.30** |
| NUGT | **+634%** | +48% | **−28%** | −74% | **1.29** |

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
