"""Generic multi-ticker Streamlit app (SOXX / GRID / XLE / REMX).

A single, config-driven port of ``app/gldm_hourly_app.py`` — it reproduces the
Gold app's exact layout, styling and colour coding for whichever ticker the
sidebar radio selects, reading everything asset-specific from
``ticker_config.TickerConfig``.  The BTC and GLDM apps are never imported here;
the root router (``streamlit_app.py``) dispatches to this module for the five
new apps and keeps the two originals byte-for-byte unchanged.

Two strategy engines are supported and chosen per ticker in the config:
  * "ma"          long-above-N-day-SMA trend filter (SOXX / GRID)
  * "divergence"  the Gold/BTC U1·D2·D3 Pure-Regime system (XLE / REMX)

Tabs mirror Gold: 🔴 Live · 🕒 Historical replay · 📊 per-asset Backtesting ·
🧠 Explain.
"""
import sys
import importlib
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

import ticker_core
import backtest_ticker
import ticker_config
# Self-heal a stale module cache on long-lived Streamlit servers (same guard as
# the Gold app): reload from disk if an expected new symbol is missing.
if not hasattr(ticker_core, "compute_trend_signatures"):
    ticker_core = importlib.reload(ticker_core)
if not hasattr(backtest_ticker, "run_strategy"):
    backtest_ticker = importlib.reload(backtest_ticker)
tc = ticker_core
bt = backtest_ticker

_ALL_APPS = (["OVERALL", "BTC", "GLDM", "GDXM"] + ticker_config.APP_KEYS
             + ["DAILYAUDIT", "HEALTH", "TARGETBOOK", "EXECUTEDBOOK"])
_APP_LABELS = {"OVERALL": "🧭  Overall Trading",
               "BTC": "₿  Bitcoin (BTC)", "GLDM": "🥇  Gold Trend (GLDM·UGL)",
               "GDXM": "⛏️  Gold Miners (GDX·NUGT)",
               "DAILYAUDIT": "🕵️  Daily Audit",
               "HEALTH": "🩺  Strategy Health",
               "TARGETBOOK": "📋  Target Book (IBKR)",
               "EXECUTEDBOOK": "✅  Executed Book (IBKR)"}
for _k, _c in ticker_config.CONFIGS.items():
    _APP_LABELS[_k] = f"{_c.emoji}  {_c.key} · {_c.name.split('(')[0].strip()[:22]}"

# ── which ticker is active? (set by the router via session state) ─────────
_active = st.session_state.get("gldm_active_app", "SOXX")
if _active not in ticker_config.CONFIGS:
    _active = ticker_config.APP_KEYS[0]
cfg = ticker_config.get_config(_active)

st.set_page_config(page_title=f"{cfg.key} Forecaster", page_icon=cfg.emoji,
                   layout="wide", initial_sidebar_state="expanded")

# Streamlit's default st.metric value font (~2.25rem) clips long values —
# e.g. the "95% CI band" $lo–$hi range in a 5-column row — so shrink it
# enough that the full value is always visible.
st.markdown("""
<style>
div[data-testid="stMetricValue"] { font-size: 1.4rem; }
</style>""", unsafe_allow_html=True)

# ── theme shorthands ──────────────────────────────────────────────────────
ACC = cfg.accent; ACCD = cfg.accent_dark; ACCBG = cfg.accent_bg; ACCBG2 = cfg.accent_bg2
IS_DIV = cfg.strategy_mode == "divergence"
TUI = None if IS_DIV else cfg.trend_ui()   # trend-rule UI descriptors (ma/dual_ma/macd/ma_vol)
SIGNAL_LINE = "#475569"     # secondary signal-price line (distinct from any accent)


# ════════════════════════════════════════════════════════════════════════
# Sidebar — unified application selector (all seven apps) + controls
# ════════════════════════════════════════════════════════════════════════
if "gldm_active_app" not in st.session_state:
    st.session_state["gldm_active_app"] = cfg.key
with st.sidebar:
    st.radio("**Application**", options=_ALL_APPS,
             format_func=lambda x: _APP_LABELS.get(x, x), key="gldm_active_app")
    st.markdown("---")
    st.markdown("**Auto-refresh:** live data cached ~5 min.")
    if st.button("Refresh now", use_container_width=True):
        st.cache_data.clear(); st.cache_resource.clear(); st.rerun()
    st.caption(f"_{cfg.key} & macro pulled from Yahoo. Equity trades US market "
               "hours; intraday bars update through the session._")

st.title(f"{cfg.emoji} {cfg.key} — {cfg.name}")
import strategy_version as _sv                 # noqa: E402
st.caption(_sv.badge_caption())                # visible above every tab
st.caption(cfg.blurb + f"  Models: ridge on log-returns (hourly close), ridge "
           "H/L bands & close cones, logistic day-type — driven by the asset's "
           f"macro factors. The **{cfg.strategy_name}** strategy is tuned to beat "
           "buy-&-hold on risk while keeping drawdowns shallow.")


# ════════════════════════════════════════════════════════════════════════
# Loaders (per-ticker cache keys so the five apps never collide)
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


@st.cache_data(ttl=300, show_spinner="Fetching daily data…")
def get_daily(key: str):
    c = ticker_config.get_config(key)
    d = tc.fetch_daily(c)
    paths = tc.cache_paths(c)
    if d is None or d.empty:
        if paths["daily"].exists():
            d = pd.read_csv(paths["daily"], index_col=0, parse_dates=True)
    return d


@st.cache_data(ttl=300, show_spinner="Fetching hourly data…")
def get_hourly(key: str):
    c = ticker_config.get_config(key)
    h = tc.fetch_hourly(c)
    paths = tc.cache_paths(c)
    if h is None or h.empty:
        if paths["hourly"].exists():
            h = pd.read_csv(paths["hourly"], index_col=0, parse_dates=True)
    return h


@st.cache_data(ttl=600, show_spinner=False)
def get_predictions(key: str, _daily_key: str):
    c = ticker_config.get_config(key)
    daily = get_daily(key)
    preds = bt.build_predictions(c, daily)
    sig = bt.precompute_signals(c, preds)
    return preds, sig


def _fresh(art):
    if not art:
        return "_missing_"
    meta = art.get("calibration_meta", {})
    end = meta.get("train_end") or art.get("train_end")
    return end.split()[0] if isinstance(end, str) else "_unknown_"


_MP = tc.model_paths(cfg)
M_HOURLY = load_model(str(_MP["hourly"]))
M_HL = load_model(str(_MP["daily_hl"]))
M_7D = load_model(str(_MP["cone_7d"]))
M_14D = load_model(str(_MP["cone_14d"]))
M_DT = load_model(str(_MP["day_type"]))

daily = get_daily(cfg.key)
if daily is None or daily.empty:
    st.error(f"Could not fetch {cfg.key} data from Yahoo. Click **Refresh now** to retry.")
    st.stop()
_daily_key = f"{daily.index.max()}::{len(daily)}"
preds, sig = get_predictions(cfg.key, _daily_key)
completed_all = preds[preds["actual_high"].notna() & preds["actual_low"].notna()]

# ── signal-freshness caption + 🕵️ Daily Audit bookkeeping ──────────────────
# Shared helpers (app/freshness.py) so the closing date/time shown here always
# matches the Daily Audit tab: these signals are generated upon the US market
# close (4:00 PM ET) of the newest daily bar.
try:
    import freshness as _fr
    _sig_asof = pd.Timestamp(daily.index.max())
    st.caption(_fr.signal_close_caption("us_equity", _sig_asof))
    _fr.record_refresh(cfg.key, kind="us_equity",
                       app_label=f"{cfg.emoji} {cfg.key}",
                       as_of=str(_sig_asof.date()),
                       close_label=_fr.close_label("us_equity", _sig_asof))
except Exception:
    pass

TRADED = cfg.traded_assets                      # [(label, col), ...]
PRIMARY_LABEL = TRADED[0][0]


# ── per-traded-asset fixed stop ──────────────────────────────────────────
# The app runs ONE signal but may trade several instruments off it (e.g. SOXX
# 1× plus the 3× SOXL sibling).  Each instrument can carry its OWN fixed stop:
# SOXX keeps its tuned −5% stop, while SOXL trades the SAME signal with NO fixed
# stop (a −5% stop is far too tight for a 3× ETF — it whipsaws it — so SOXL exits
# on the signal only).  These helpers read that per-asset stop so the strategy
# card, live position panel and backtest tabs describe each instrument correctly
# instead of implying the 1× stop applies to a leveraged sibling.  Read from the
# data field (not the ``stop_for`` method) so a hot-reloaded/stale TickerConfig
# instance without the method can't AttributeError.
def _asset_stop(col):
    return getattr(cfg, "stop_by_asset", {}).get(col, cfg.fixed_stop)


# Siblings whose stop differs from the primary's tuned stop — surfaced so the
# shared strategy card never implies the 1× stop governs a leveraged sibling.
_STOP_SIBLINGS = [(lbl, col) for lbl, col in TRADED[1:]
                  if _asset_stop(col) != cfg.fixed_stop]


def _sibling_stop_note():
    """One-line clarification when a traded sibling exits differently from the
    1× primary (e.g. SOXL trades the SOXX signal with no fixed stop)."""
    if not _STOP_SIBLINGS:
        return ""
    parts = []
    for lbl, col in _STOP_SIBLINGS:
        s = _asset_stop(col)
        how = ("exits on the signal only — <b>no fixed stop</b>" if s >= 0.999
               else f"uses a wider <b>−{s * 100:.0f}%</b> stop")
        parts.append(f"<b>{lbl}</b> {how}")
    return (f"The {cfg.stop_label} fixed stop applies to <b>{cfg.key}</b> (1×) only; "
            + "; ".join(parts)
            + " — a 1× stop is too tight for a leveraged sibling, so it whipsaws it.")


def _sibling_stop_note_md():
    """Markdown (not HTML) variant of ``_sibling_stop_note`` for st.markdown text."""
    if not _STOP_SIBLINGS:
        return ""
    parts = []
    for lbl, col in _STOP_SIBLINGS:
        s = _asset_stop(col)
        how = ("exits on the signal only — **no fixed stop**" if s >= 0.999
               else f"uses a wider **−{s * 100:.0f}%** stop")
        parts.append(f"**{lbl}** {how}")
    return (f"The {cfg.stop_label} fixed stop applies to **{cfg.key}** (1×) only; "
            + "; ".join(parts)
            + " — a 1× stop is too tight for a leveraged sibling (it whipsaws it into "
              "a much lower win-rate and Sharpe), so the sibling exits on the signal only.")


# ════════════════════════════════════════════════════════════════════════
# Inference helpers
# ════════════════════════════════════════════════════════════════════════
def predict_next_hour(hourly):
    if M_HOURLY is None or hourly is None or hourly.empty:
        return None
    feat = tc.build_hourly_features(cfg, hourly)
    cols = M_HOURLY["feat_cols"]
    x = feat[cols].ffill().iloc[[-1]]
    if x.isna().any(axis=None):
        return None
    r = float(M_HOURLY["model"].predict(x)[0])
    last_close = float(hourly["px_close"].iloc[-1])
    sigma = float(M_HOURLY["sigma"])
    return dict(last_close=last_close, pred_close=last_close * np.exp(r),
                lo=last_close * np.exp(r - 1.96 * sigma),
                hi=last_close * np.exp(r + 1.96 * sigma),
                ret=r, sigma=sigma, ts=hourly.index[-1])


def predict_next_daily_hl(daily_df):
    if M_HL is None:
        return None
    feat = tc.build_daily_features(cfg, daily_df)
    cols = M_HL["feat_cols"]
    x = feat[cols].ffill().iloc[[-1]]
    if x.isna().any(axis=None):
        return None
    last_close = float(daily_df["px_close"].iloc[-1])
    bh = M_HL.get("bias_high", 0.0); bl = M_HL.get("bias_low", 0.0)
    ph = last_close * (1 + float(M_HL["model_high"].predict(x)[0])) + bh * last_close
    pl = last_close * (1 + float(M_HL["model_low"].predict(x)[0])) + bl * last_close
    return dict(last_close=last_close, pred_high=ph, pred_low=pl,
                band_hi=M_HL["sigma_high"] * last_close,
                band_lo=M_HL["sigma_low"] * last_close)


def predict_cone(art, daily_df):
    if art is None:
        return None
    feat = tc.build_daily_features(cfg, daily_df)
    cols = art["feat_cols"]
    x = feat[cols].ffill().iloc[[-1]]
    if x.isna().any(axis=None):
        return None
    last_close = float(daily_df["px_close"].iloc[-1])
    central_r = float(art["model"].predict(x)[0]); q = art["quantiles"]
    return dict(last_close=last_close, horizon=art["horizon"],
                central=last_close * np.exp(central_r),
                p5=last_close * np.exp(central_r + q[5]),
                p25=last_close * np.exp(central_r + q[25]),
                p75=last_close * np.exp(central_r + q[75]),
                p95=last_close * np.exp(central_r + q[95]))


def predict_day_type(daily_df):
    if M_DT is None:
        return None
    feat = tc.build_daily_features(cfg, daily_df)
    cols = M_DT["feat_cols"]
    x = feat[cols].ffill().iloc[[-1]]
    if x.isna().any(axis=None):
        return None
    proba = M_DT["model"].predict_proba(x)[0]
    classes = M_DT["model"].classes_; label_map = M_DT["classes"]
    pm = {label_map[int(c)]: float(p) for c, p in zip(classes, proba)}
    return dict(probs=pm, top=max(pm, key=pm.get))


def signatures_asof(target_date):
    # tail(150), not 45: the 60-bar median centering + 30-bar capitulation
    # normaliser need ≥90 bars of history for the newest bar's signals to
    # reproduce backtest_ticker.precompute_signals exactly.
    sub = completed_all[completed_all["target_date"] <= pd.Timestamp(target_date)].tail(150)
    return tc.compute_trend_signatures(cfg, sub)


def ma_state(d_df):
    """Trend-filter state on the primary close (ma / dual_ma / macd / ma_vol).

    ``above`` is the engine's ACTUAL long condition on the latest bar (single
    source of truth), and ``ma`` is the representative price-level line the UI
    overlays (the slow SMA for dual_ma, the N-day SMA otherwise)."""
    if IS_DIV or d_df is None or not len(d_df):
        return None
    c = d_df["px_close"].to_numpy(float)
    w = cfg.ma_slow if cfg.strategy_mode == "dual_ma" else cfg.ma_window
    ma = float(np.mean(c[-w:])) if len(c) >= 1 else np.nan
    ma_prev = float(np.mean(c[-(w + 5):-5])) if len(c) >= w + 5 else ma
    close = float(c[-1])
    # dual_ma trades the fast-vs-slow SMA cross, so surface the fast SMA too — the
    # UI shows both the fast (25d) and slow (100d) levels against the cross
    # threshold rather than a single close-vs-line read.
    ma_fast_val = (float(np.mean(c[-cfg.ma_fast:]))
                   if cfg.strategy_mode == "dual_ma" and len(c) >= 1 else None)
    above = bt.trend_long_now(cfg, d_df)          # engine truth for this mode
    dist = (close / ma - 1) * 100 if ma == ma and ma else 0.0
    # config-driven MACD histogram value on the latest bar (macd mode only)
    macd_hist = None
    if cfg.strategy_mode == "macd" and len(c):
        macd_hist = float(bt.macd_hist_array(cfg, c)[-1])
    # human phrase describing the current signal state
    if cfg.strategy_mode == "macd":
        desc = (f"the MACD({cfg.macd_fast}/{cfg.macd_slow}/{cfg.macd_signal}) histogram "
                f"is {macd_hist:+.4f} ({'positive' if above else 'negative'})")
    elif cfg.strategy_mode == "dual_ma":
        desc = f"the {cfg.ma_fast}-day SMA is {'above' if above else 'below'} the {cfg.ma_slow}-day SMA ${ma:,.2f}"
    elif cfg.strategy_mode == "ma_vol":
        desc = (f"close ${close:,.2f} is {'above' if above else 'below/blocked-by-vol at'} the "
                f"{cfg.ma_window}-day SMA ${ma:,.2f} (vol filter {'clear' if above else 'active'})")
    elif cfg.strategy_mode == "crash_dd":
        hi52 = float(pd.Series(c).rolling(252, min_periods=60).max().iloc[-1])
        dd_now = close / hi52 - 1 if hi52 else 0.0
        desc = (f"close ${close:,.2f} sits {dd_now*100:+.1f}% off the 52-week high "
                f"${hi52:,.2f} (crash exit at −{cfg.dd_exit_pct*100:.0f}%) — "
                f"{'shield clear, LONG by default' if above else 'crash shield active (flat), re-enter above the SMA'}")
    else:
        desc = f"close ${close:,.2f} is {'above' if above else 'below'} the {w}-day SMA ${ma:,.2f}"
    # ma_vol carries a second mandatory gate: realised vol below k × its median.
    # Surface the live value + threshold so the UI can show it as its own row
    # rather than folding it into the single close-vs-SMA read.
    vol_state = bt.vol_filter_state(cfg, d_df) if cfg.strategy_mode == "ma_vol" else None
    return dict(ma=ma, ma_prev=ma_prev, close=close, above=above,
                slope_pos=ma > ma_prev, window=w, dist=dist, desc=desc,
                macd_hist=macd_hist, vol_state=vol_state, ma_fast_val=ma_fast_val,
                line_label=TUI["line"], cond=TUI["cond"], cond_short=TUI["cond_short"])


def strategy_position(col, end=None):
    if col not in preds:
        return None
    r = bt.run_strategy(cfg, preds, sig, col, end=end)
    r["metrics"] = bt._metrics(r["strat"], r["dates"])
    r["bh_metrics"] = bt._metrics(r["bh"], r["dates"])
    return r


# ════════════════════════════════════════════════════════════════════════
# Trend-signature alert renderer (identical card layout to Gold)
# ════════════════════════════════════════════════════════════════════════
def _sig_card(title, icon, color, triggered, rows, interpretation):
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
    'context'} — 'threshold' marks a MANDATORY condition the gate requires (every
    required component, whether a single-bar cut-off or a lookback-window state);
    'context' is reserved for genuinely non-required, informational rows."""
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


def net_signal_div(sigs):
    """Resolve raw signatures → ONE state (divergence mode). Exit overrides entry."""
    if not sigs:
        return dict(state="NEUTRAL", label="NO DATA", ico="⬜",
                    bg="#f8fafc", brd="#94a3b8", reason="insufficient completed bars")
    d1, d2, d3 = sigs["d1_triggered"], sigs["d2_triggered"], sigs["d3_triggered"]
    u1, entry = sigs["u1_triggered"], sigs["entry_triggered"]
    d1_exit = d1 and cfg.use_d1_exit
    if d2 or d3 or d1_exit:
        parts = [p for p, f in [("D3 exhaustion", d3), ("D2 momentum fade", d2),
                                ("D1 downtrend", d1_exit)] if f]
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


def net_signal_ma(mst, pos=None):
    """One resolved state for the MA-trend-filter mode.

    Reconciles the *instantaneous* MA read (today's close vs the SMA) with the
    strategy's *actual* executed position, so the NET DECISION can never silently
    contradict the position card.  The filter decides at each close and acts on
    the NEXT bar (no look-ahead), so a one-bar whipsaw — prior close dips below
    the SMA (→ exit today) while today's close pops back above it — is surfaced
    explicitly instead of showing a bare 'LONG' that disagrees with a just-closed
    position.
    """
    if not mst:
        return dict(state="NEUTRAL", label="NO DATA", ico="⬜",
                    bg="#f8fafc", brd="#94a3b8", reason="insufficient bars")
    above = mst["above"]; desc = mst["desc"]
    in_pos = bool(pos and pos.get("in_pos_now"))
    just_exit = None
    if pos and not in_pos and pos.get("trade_log"):
        lt = pos["trade_log"][-1]
        exit_d = pd.Timestamp(lt["exit_date"])
        last_bar = pd.Timestamp(preds["target_date"].iloc[-1])
        if exit_d >= last_bar - pd.Timedelta(days=6):     # exited on/near the latest bar
            just_exit = lt

    if in_pos:
        return dict(state="ENTRY", label="LONG — HOLDING", ico="🟢",
                    bg="#f0fdf4", brd="#16a34a",
                    reason=f"Strategy is long; {desc} — hold.")
    if above:
        # Flat but the trend signal is bullish again → the filter re-enters next bar.
        base = (f"Strategy is FLAT. The trend signal is bullish again ({desc}), so it "
                f"will RE-ENTER (go long) on the next bar.")
        if just_exit:
            base += (f" It exited on {pd.Timestamp(just_exit['exit_date']).strftime('%b %d')} "
                     f"({just_exit['reason']}) on the *prior* bar — a one-bar whipsaw; the "
                     f"signal has since recovered.")
        return dict(state="WATCH_UP", label="FLAT — RE-ENTRY PENDING NEXT BAR", ico="🟡",
                    bg="#fefce8", brd="#ca8a04", reason=base)
    return dict(state="EXIT", label="FLAT — BELOW TREND", ico="🔴",
                bg="#fef2f2", brd="#dc2626",
                reason=f"Strategy is FLAT; {desc} — stand aside in cash.")


def render_strategy_card():
    exec_line = (" · ".join(f"<b>{lbl}</b>" for lbl, _ in TRADED))
    if IS_DIV:
        st.markdown(f"""
<div style='background:{ACCBG}; border:2px solid {ACC}; border-radius:12px;
     padding:16px 20px; margin:4px 0 14px 0; font-family:sans-serif;'>
  <div style='font-size:15px; font-weight:800; color:{ACCD}; margin-bottom:12px; letter-spacing:0.3px;'>
    {cfg.emoji} {cfg.strategy_name} &nbsp;—&nbsp;
    <span style='color:{ACC};'>{cfg.key} signal · {exec_line} execution</span>
  </div>
  <div style='background:{ACCBG2}; border-radius:8px; padding:10px 14px; margin-bottom:12px;
       font-size:12.5px; color:{ACCD}; font-weight:600;'>
    🔁 <b>Core idea:</b> the divergence signatures (U1 / D2 / D3 / V-reversal) read
    <b>{cfg.key}'s</b> predicted-vs-actual daily highs/lows as the signal engine, and the
    strategy holds long only while momentum is confirmed inside a trend regime — sitting
    out the deep drawdowns that wreck buy-&-hold on this asset.
  </div>
  <div style='background:{ACCBG2}; border:1px solid {ACC}; border-radius:7px;
       padding:8px 13px; margin-bottom:12px; font-size:12px; color:{ACCD};'>
    🎯 <b>Pure-Regime entry:</b> a U1 bullish-divergence trigger must be confirmed by
    <b>one</b> of three non-overlapping trend paths — a bull regime (above a rising
    20-day MA), a washed-out clean breakout from below the MA, or a fresh V-reversal.
  </div>
  <div style='display:flex; gap:14px; flex-wrap:wrap;'>
    <div style='flex:1; min-width:230px;'>
      <div style='font-size:11px; font-weight:700; color:#15803d; text-transform:uppercase;
           letter-spacing:0.8px; margin-bottom:5px;'>📥 Entry — go long {exec_line}</div>
      <div style='font-size:12px; color:#334155; line-height:1.7;'>
        ① <b>U1 active</b> — err_hi 3d-avg &gt; +{cfg.u1_errhi_min:.2f}% &amp;&amp; ≥2 high-breaks<br>
        ② <b>one gate</b>: 🐂 Bull Regime · 🧹 Clean Breakout · ⚡ V-reversal
      </div>
    </div>
    <div style='flex:1; min-width:230px;'>
      <div style='font-size:11px; font-weight:700; color:#b91c1c; text-transform:uppercase;
           letter-spacing:0.8px; margin-bottom:5px;'>📤 Exit — go to cash (overrides entry)</div>
      <div style='font-size:12px; color:#334155; line-height:1.7;'>
        ① <b>D2 fade</b> — err_hi 3d-avg &lt; {cfg.d2_errhi_max:+.2f}%<br>
        ② <b>D3 exhaustion</b> — first low-break after a ≥3 high-break streak<br>
        {('③ <b>D1 downtrend</b> — err_lo 3d-avg &gt; +' + f'{cfg.d1_errlo_min:.2f}' + '% &amp; ≥2 low-breaks<br>' if cfg.use_d1_exit else '') + ((('④' if cfg.use_d1_exit else '③') + ' <b>fixed stop</b> — ' + cfg.stop_label + ' from entry') if cfg.has_stop else '<i>no fixed stop — exits are signal-driven</i>')}
      </div>
    </div>
  </div>
  {("<div style='margin-top:10px; background:#fff7ed; border:1px solid #fdba74; border-radius:7px; padding:7px 12px; font-size:11.5px; color:#9a3412;'>🔀 <b>Per-instrument exits:</b> " + _sibling_stop_note() + "</div>") if _STOP_SIBLINGS else ""}
  <div style='margin-top:12px; font-size:11.5px; color:{ACCD};'>📈 {cfg.results_note}</div>
</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
<div style='background:{ACCBG}; border:2px solid {ACC}; border-radius:12px;
     padding:16px 20px; margin:4px 0 14px 0; font-family:sans-serif;'>
  <div style='font-size:15px; font-weight:800; color:{ACCD}; margin-bottom:12px; letter-spacing:0.3px;'>
    {cfg.emoji} {cfg.strategy_name} &nbsp;—&nbsp;
    <span style='color:{ACC};'>{TUI['headline']}</span>
  </div>
  <div style='background:{ACCBG2}; border-radius:8px; padding:10px 14px; margin-bottom:12px;
       font-size:12.5px; color:{ACCD}; font-weight:600;'>
    🔁 <b>Core idea:</b> {cfg.key} trends in long cycles with vicious drawdowns. A robust
    trend filter — <b>{TUI['core']}</b> — captures the up-cycles and moves to
    cash for the down-cycles, beating buy-&-hold on risk (and often on return).
  </div>
  <div style='display:flex; gap:14px; flex-wrap:wrap;'>
    <div style='flex:1; min-width:230px;'>
      <div style='font-size:11px; font-weight:700; color:#15803d; text-transform:uppercase;
           letter-spacing:0.8px; margin-bottom:5px;'>📥 Entry — go long</div>
      <div style='font-size:12px; color:#334155; line-height:1.7;'>
        ① {TUI['entry']} (decided at the close)<br>
        ② enter on the next bar
      </div>
    </div>
    <div style='flex:1; min-width:230px;'>
      <div style='font-size:11px; font-weight:700; color:#b91c1c; text-transform:uppercase;
           letter-spacing:0.8px; margin-bottom:5px;'>📤 Exit — go to cash</div>
      <div style='font-size:12px; color:#334155; line-height:1.7;'>
        {('① ' + TUI['exit'] + ', or<br>② <b>fixed stop</b> — ' + cfg.stop_label + ' from entry') if cfg.has_stop else ('① ' + TUI['exit'])}
      </div>
    </div>
  </div>
  {("<div style='margin-top:10px; background:#fff7ed; border:1px solid #fdba74; border-radius:7px; padding:7px 12px; font-size:11.5px; color:#9a3412;'>🔀 <b>Per-instrument exits:</b> " + _sibling_stop_note() + "</div>") if _STOP_SIBLINGS else ""}
  <div style='margin-top:12px; font-size:11.5px; color:{ACCD};'>📈 {cfg.results_note}</div>
</div>""", unsafe_allow_html=True)


def render_conditions_box(sigs, mst, pos=None):
    if IS_DIV:
        ns = net_signal_div(sigs)
        if not sigs:
            st.info("Strategy conditions unavailable — need ≥ 3 completed bars.")
            return

        def row(active, name, detail):
            ico = "✅" if active else "○"
            col = "#15803d" if active else "#94a3b8"; weight = "700" if active else "500"
            return (f"<tr><td style='padding:3px 8px 3px 0;font-size:14px;'>{ico}</td>"
                    f"<td style='padding:3px 10px 3px 0;font-weight:{weight};color:{col};"
                    f"white-space:nowrap;'>{name}</td>"
                    f"<td style='padding:3px 0;font-size:11.5px;color:#475569;'>{detail}</td></tr>")

        gate_bull = bool(sigs.get("bull_regime"))
        gate_clean = bool(sigs["clean_10d"] and not sigs["above_ma20"])
        gate_v = bool(sigs.get("v_recent_gate"))
        gate_any = gate_bull or gate_clean or gate_v
        d1_exit = sigs["d1_triggered"] and cfg.use_d1_exit
        exit_active = sigs["d2_triggered"] or sigs["d3_triggered"] or d1_exit
        entry_ready = sigs["u1_triggered"] and gate_any and not exit_active
        entry_html = "<table style='border-collapse:collapse;'>" + \
            row(sigs["u1_triggered"], "U1 trigger",
                f"err_hi 3d-avg = {sigs['err_hi_ma3']:+.3f}% (need &gt; +{cfg.u1_errhi_min:.2f}%) "
                f"&amp; high-breaks = {sigs['hi_breaks_3d']}/3 (need ≥2)") + \
            row(gate_any, "Trend gate (any one)",
                f"{'🐂 Bull ' if gate_bull else ''}{'🧹 Clean ' if gate_clean else ''}"
                f"{'⚡ V-rev' if gate_v else ''}".strip() or "none active") + \
            "<tr><td></td><td colspan='2' style='padding-left:22px;font-size:11px;color:#64748b;'>" + \
            f"{'✓' if gate_bull else '○'} Bull Regime &nbsp; {'✓' if gate_clean else '○'} Clean Breakout &nbsp; " + \
            f"{'✓' if gate_v else '○'} V-reversal</td></tr></table>"
        exit_html = "<table style='border-collapse:collapse;'>" + \
            row(sigs["d2_triggered"], "D2 momentum fade",
                f"err_hi 3d-avg = {sigs['err_hi_ma3']:+.3f}% (exit &lt; {cfg.d2_errhi_max:+.2f}%)") + \
            row(sigs["d3_triggered"], "D3 exhaustion",
                f"consec high-breaks = {sigs['consec_hi']} then a low-break") + \
            (row(d1_exit, "D1 downtrend pressure",
                 f"err_lo 3d-avg = {sigs['err_lo_ma3']:+.3f}% (exit &gt; +{cfg.d1_errlo_min:.2f}%) "
                 f"&amp; ≥2 low-breaks") if cfg.use_d1_exit else "") + \
            (row(False, f"Fixed stop {cfg.stop_label}",
                 "position-level — checked per open trade") if cfg.has_stop else "") + "</table>"
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
      📤 EXIT conditions {'— 🔴 ACTIVE' if exit_active else '— clear'}</div>
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
    else:
        ns = net_signal_ma(mst, pos)
        if not mst:
            st.info("Strategy conditions unavailable.")
            return
        above = mst["above"]
        strat_state = ("🟢 LONG (holding)" if (pos and pos.get("in_pos_now"))
                       else "⚪ FLAT (in cash)")
        def rowm(active, name, detail):
            ico = "✅" if active else "○"
            col = "#15803d" if active else "#94a3b8"; weight = "700" if active else "500"
            return (f"<tr><td style='padding:3px 8px 3px 0;font-size:14px;'>{ico}</td>"
                    f"<td style='padding:3px 10px 3px 0;font-weight:{weight};color:{col};"
                    f"white-space:nowrap;'>{name}</td>"
                    f"<td style='padding:3px 0;font-size:11.5px;color:#475569;'>{detail}</td></tr>")
        # The trend filter trades on ONE regime condition (mode-specific) — except
        # ma_vol, which ANDs a close-vs-SMA cut with a volatility-contraction gate.
        # For ma_vol split the two mandatory cuts into their own rows so the
        # realised-vol value shows against its required threshold.
        vs = mst.get("vol_state")
        if vs:
            close_ok = mst["dist"] > 0
            entry_html = "<table style='border-collapse:collapse;'>" + \
                rowm(close_ok, f"close &gt; {cfg.ma_window}-day SMA",
                     f"close ${mst['close']:,.2f} vs SMA ${mst['ma']:,.2f} "
                     f"({mst['dist']:+.2f}%)") + \
                rowm(vs["ok"], f"{vs['vol_win']}-day realised vol subdued",
                     f"{vs['vol']*100:.2f}% (need &lt; {vs['vol_k']:g}× "
                     f"{vs['vol_med_win']}-day median = {vs['thr']*100:.2f}%)") + \
                "</table>"
        elif cfg.strategy_mode == "dual_ma" and mst.get("ma_fast_val") is not None:
            # dual_ma trades the fast-vs-slow SMA cross — surface both SMA levels
            # against the cross threshold (fast must sit above slow) so the card
            # shows the actual required comparison, not just close-vs-line.
            maf = mst["ma_fast_val"]; mas = mst["ma"]
            gap = (maf / mas - 1) * 100 if mas else 0.0
            entry_html = "<table style='border-collapse:collapse;'>" + \
                rowm(above, mst["cond"],
                     f"{cfg.ma_fast}-day SMA ${maf:,.2f} vs {cfg.ma_slow}-day SMA "
                     f"${mas:,.2f} ({gap:+.2f}%)") + \
                rowm(maf > mas, f"{cfg.ma_fast}-day SMA &gt; {cfg.ma_slow}-day SMA",
                     f"${maf:,.2f} &gt; ${mas:,.2f} (need &gt; 0%, now {gap:+.2f}%)") + \
                "</table>"
        else:
            entry_html = "<table style='border-collapse:collapse;'>" + \
                rowm(above, mst["cond"], mst["desc"]) + "</table>"
        exit_html = "<table style='border-collapse:collapse;'>" + \
            rowm(not above, mst["cond_short"], "→ move to cash") + \
            (rowm(False, f"Fixed stop {cfg.stop_label}",
                  "position-level — checked per open trade") if cfg.has_stop else "") + "</table>"
        st.markdown(f"""
<div style='display:flex; gap:14px; flex-wrap:wrap; margin:2px 0 6px 0;'>
  <div style='flex:1; min-width:280px; background:#f0fdf4; border:1.5px solid #86efac;
       border-radius:10px; padding:10px 14px;'>
    <div style='font-weight:700; color:#15803d; margin-bottom:6px;'>
      📥 LONG conditions {'— ✅ IN TREND' if above else '— not met'}</div>
    {entry_html}
    <div style='font-size:11px;color:#64748b;margin-top:6px;'>Hold long while the trend
      signal ({TUI['cond']}) is bullish. <b>Decisions are made at the daily
      close and executed on the next bar</b> (no look-ahead), so the actual
      position can lag this instantaneous read by one bar — e.g. a one-day flip
      exits even if the signal has since recovered.<br>
      Current strategy position: <b>{strat_state}</b>.</div>
  </div>
  <div style='flex:1; min-width:280px; background:#fef2f2; border:1.5px solid #fca5a5;
       border-radius:10px; padding:10px 14px;'>
    <div style='font-weight:700; color:#b91c1c; margin-bottom:6px;'>
      📤 CASH conditions {'— 🔴 ACTIVE' if not above else '— clear'}</div>
    {exit_html}
    <div style='font-size:11px;color:#64748b;margin-top:6px;'>A bearish flip of the trend
      signal{' (or the fixed stop)' if cfg.has_stop else ''} moves the position to cash.</div>
  </div>
</div>
<div style='background:{ns['bg']}; border:2px solid {ns['brd']}; border-radius:10px;
     padding:10px 16px; margin:4px 0;'>
  <b style='font-size:15px;'>{ns['ico']} NET DECISION: {ns['label']}</b>
  <span style='color:#475569; font-size:12.5px; margin-left:8px;'>{ns['reason']}</span>
</div>""", unsafe_allow_html=True)


def render_signatures(sigs):
    if not sigs:
        st.info("Not enough completed bars for trend signatures yet (need ≥ 3).")
        return
    ns = net_signal_div(sigs)
    as_of = pd.Timestamp(sigs["as_of_date"]).strftime("%Y-%m-%d")
    st.markdown(
        f"""<div style="background:{ns['bg']};border:2px solid {ns['brd']};
        border-radius:10px;padding:12px 16px;margin:8px 0;">
        <span style="background:{ns['brd']};color:white;font-weight:700;font-size:14px;
        padding:5px 14px;border-radius:20px;">{ns['ico']} {ns['label']}</span>
        <span style="color:#334155;font-size:13px;margin-left:10px;">
        <b>{sigs['dn_count']}/3</b> DN · <b>{sigs['up_count']}/1</b> UP ·
        raw flags: U1={'✓' if sigs['u1_triggered'] else '✗'}
        D2={'✓' if sigs['d2_triggered'] else '✗'} D3={'✓' if sigs['d3_triggered'] else '✗'}
        · bull_regime={'✓' if sigs.get('bull_regime') else '✗'}
        · as-of <b>{as_of}</b> · <b>{sigs['n_bars']}</b> bars</span></div>""",
        unsafe_allow_html=True)
    r1c1, r1c2 = st.columns(2)
    r2c1, r2c2 = st.columns(2)
    r1c1.markdown(_sig_card(
        "U1 — Bullish pressure", "📈", "#16a34a", sigs["u1_triggered"],
        [("err_hi 3d-avg", f"{sigs['err_hi_ma3']:+.3f}%", f"> +{cfg.u1_errhi_min:.2f}%",
          sigs['err_hi_ma3'] > cfg.u1_errhi_min),
         ("high-breaks (3d)", f"{sigs['hi_breaks_3d']}/3", "≥ 2", sigs['hi_breaks_3d'] >= 2)],
        f"{cfg.key}'s actual highs keep beating the model's predicted highs → upside "
        "momentum. This is the entry trigger (needs a trend gate too)."),
        unsafe_allow_html=True)
    r1c2.markdown(_sig_card(
        "D2 — Momentum fading", "📉", "#dc2626", sigs["d2_triggered"],
        [("err_hi 3d-avg", f"{sigs['err_hi_ma3']:+.3f}%", f"< {cfg.d2_errhi_max:+.2f}%",
          sigs['err_hi_ma3'] < cfg.d2_errhi_max)],
        "Predicted highs are no longer being beaten — upside momentum is collapsing. "
        "Primary exit signal."), unsafe_allow_html=True)
    r2c1.markdown(_sig_card(
        "D1 — Downtrend pressure", "📉", "#dc2626", sigs["d1_triggered"],
        [("err_lo 3d-avg", f"{sigs['err_lo_ma3']:+.3f}%", f"> +{cfg.d1_errlo_min:.2f}%",
          sigs['err_lo_ma3'] > cfg.d1_errlo_min),
         ("low-breaks (3d)", f"{sigs['lo_breaks_3d']}/3", "≥ 2", sigs['lo_breaks_3d'] >= 2)],
        f"{cfg.key}'s actual lows keep undershooting the predicted floor → the trend is "
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
    if sigs["detail_rows"]:
        with st.expander("📋 Last 5 bars — signal detail", expanded=False):
            st.caption(
                f"Each row is a **completed** {cfg.key} daily bar (actual H/L known); signals use "
                "these bars only. err_hi (bar) = (actual_high − pred_high)/close × 100 "
                "(regime-centered; + = bullish pressure). err_hi 3d-avg = rolling 3-bar mean. "
                "Break = actual H > pred_H (Hi) or actual L < pred_L (Lo).")
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


def render_gate_signatures(sigs):
    """Entry-gate condition cards (divergence mode). A U1 pressure signal only
    opens a position when a trend gate also holds:
    ``U1 AND (Bull Regime OR Clean Breakout OR V-Reversal)``. Each card shows the
    criterion, where the live value stands, and — since every row is a mandatory
    component of that gate — tags each as a required THRESHOLD."""
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
        f"Exits (D2/D3{' /D1' if cfg.use_d1_exit else ''}) are ungated and override entry.</div>",
        unsafe_allow_html=True)

    g1, g2, g3 = st.columns(3)
    g1.markdown(_gate_card(
        "Bull Regime", "🐂", bull,
        [("close vs 20-day SMA", close_s, f"> {ma_s}", above, "threshold"),
         ("20-day SMA slope", "rising" if slope else "falling", "rising", slope, "threshold")],
        "Price sits above a rising 20-day average — an established uptrend. Satisfies "
        "the entry gate on its own."), unsafe_allow_html=True)
    g2.markdown(_gate_card(
        "Clean Breakout", "🧹", clean_gate,
        [("no D1/D2 last ~8 bars", "clean" if clean else "recent damage", "clean", clean, "threshold"),
         ("close vs 20-day SMA", close_s, f"< {ma_s}", (not above), "threshold")],
        "A fresh breakout from <i>below</i> the average with no recent downside damage — "
        "lets U1 fire early, before the regime formally turns bullish."),
        unsafe_allow_html=True)
    g3.markdown(_gate_card(
        "V-Reversal", "⚡", vgate,
        [("washout in last 3 bars", "yes — ≤3 bars ago" if vgate else "none",
          "capitulation ≤3 bars ago", vgate, "threshold"),
         ("capitulation score (today)", f"{vdn:.2f}", "> 0.80 + deep low", vgate, "threshold")],
        "A recent sharp washout / capitulation-low undershoot (V-shaped reversal setup) — "
        "also satisfies the entry gate. The gate arms when the capitulation score clears "
        "0.80 <i>and</i> the day's low deeply undershoots the model, within the last 3 bars."),
        unsafe_allow_html=True)


def render_ma_signatures(mst, pos=None):
    """Signal cards for the MA-trend-filter mode — the conditions the strategy
    *actually* trades on: close vs the SMA (the sole entry/exit rule), the fixed
    stop, and the current position.  (The SMA's slope is deliberately NOT shown
    as a card: it plays no part in the entry/exit logic, so surfacing it as a
    'signature' would misrepresent the strategy.)"""
    if not mst:
        st.info("Not enough bars for the trend-filter read yet.")
        return
    ns = net_signal_ma(mst, pos)
    c = mst["close"]; ma = mst["ma"]; w = mst["window"]
    above = mst["above"]; line_lbl = mst["line_label"]
    dist = (c / ma - 1) * 100 if ma else 0.0
    in_pos = bool(pos and pos.get("in_pos_now"))
    is_macd = cfg.strategy_mode == "macd"
    mh = mst.get("macd_hist")

    # In macd mode surface the actual histogram value next to the signal — it IS
    # the traded quantity (long while histogram > 0), so the banner shows it
    # alongside the close/MA read.
    hist_html = ""
    if is_macd and mh is not None:
        hist_html = (f" · MACD({cfg.macd_fast}/{cfg.macd_slow}/{cfg.macd_signal}) hist "
                     f"<b>{mh:+.4f}</b>")
    # dual_ma: show the fast SMA next to the slow SMA in the banner — the signal
    # is the fast-vs-slow cross, not close-vs-line.
    maf_banner = mst.get("ma_fast_val")
    if cfg.strategy_mode == "dual_ma" and maf_banner is not None:
        hist_html = (f" · {cfg.ma_fast}d SMA <b>${maf_banner:,.2f}</b>"
                     f" vs {cfg.ma_slow}d SMA <b>${ma:,.2f}</b>" + hist_html)
    st.markdown(
        f"""<div style="background:{ns['bg']};border:2px solid {ns['brd']};
        border-radius:10px;padding:12px 16px;margin:8px 0;">
        <span style="background:{ns['brd']};color:white;font-weight:700;font-size:14px;
        padding:5px 14px;border-radius:20px;">{ns['ico']} {ns['label']}</span>
        <span style="color:#334155;font-size:13px;margin-left:10px;">
        close <b>${c:,.2f}</b> · {line_lbl} <b>${ma:,.2f}</b> ·
        distance <b>{dist:+.2f}%</b>{hist_html}</span></div>""",
        unsafe_allow_html=True)

    trend_rows = [("signal", "bullish" if above else "bearish", mst["cond"], above)]
    if is_macd and mh is not None:
        trend_rows.append(
            (f"MACD({cfg.macd_fast}/{cfg.macd_slow}/{cfg.macd_signal}) hist",
             f"{mh:+.4f}", "> 0", mh > 0))
    # dual_ma: the traded rule is fast SMA > slow SMA, so show BOTH SMA levels
    # against the cross threshold instead of a single close-vs-line read.
    maf = mst.get("ma_fast_val")
    if cfg.strategy_mode == "dual_ma" and maf is not None:
        gap = (maf / ma - 1) * 100 if ma else 0.0
        trend_rows.append(
            (f"{cfg.ma_fast}-day SMA", f"${maf:,.2f}",
             f"&gt; {cfg.ma_slow}-day SMA ${ma:,.2f} ({gap:+.2f}%)", maf > ma))
        trend_rows.append(
            (f"{cfg.ma_slow}-day SMA", f"${ma:,.2f}",
             f"&lt; {cfg.ma_fast}-day SMA ${maf:,.2f} to hold long", maf > ma))
    else:
        trend_rows.append(
            ("close vs " + line_lbl, f"${c:,.2f}", f"${ma:,.2f} ({dist:+.2f}%)", dist > 0))
    # ma_vol: the volatility-contraction gate is the SECOND mandatory long
    # condition — show the live realised vol against its required threshold so the
    # card reflects both cuts, not just the close-vs-SMA read.
    vs = mst.get("vol_state")
    if vs:
        trend_rows.append(
            (f"{vs['vol_win']}-day realised vol", f"{vs['vol']*100:.2f}%",
             f"&lt; {vs['vol_k']:g}× {vs['vol_med_win']}-day median "
             f"({vs['thr']*100:.2f}%)", vs["ok"]))
    col1, col2, col3 = st.columns(3)
    col1.markdown(_sig_card(
        f"Trend Filter — {TUI['headline']}", "📈", "#16a34a", above,
        trend_rows,
        f"The condition this strategy trades on: {TUI['core']}, and exit to cash "
        "on the next bar once it flips."), unsafe_allow_html=True)

    if not cfg.has_stop:
        col2.markdown(_sig_card(
            "Exit — signal-driven (no fixed stop)", "🚪", "#94a3b8", False,
            [("stop", "none", "signal exit only", False)],
            f"This config carries no fixed stop — the position closes when the "
            f"trend signal flips ({TUI['cond']} no longer holds)."),
            unsafe_allow_html=True)
    elif in_pos and pos.get("entry_px"):
        e_px = float(pos["entry_px"]); stop_px = e_px * (1 - cfg.fixed_stop)
        cushion = (c / stop_px - 1) * 100
        col2.markdown(_sig_card(
            f"Stop-Loss Guard — {cfg.stop_label} fixed stop", "🛑", "#dc2626", cushion < 3.0,
            [("stop level", f"${stop_px:,.2f}", "hold above", c > stop_px),
             ("cushion to stop", f"{cushion:+.2f}%", "> 0%", cushion > 0)],
            "A hard stop from the entry caps the single-trade loss if price gaps "
            "down faster than the trend signal can trigger the exit."), unsafe_allow_html=True)
    else:
        col2.markdown(_sig_card(
            f"Stop-Loss Guard — {cfg.stop_label} fixed stop", "🛑", "#94a3b8", False,
            [("status", "inactive (flat)", "opens with a position", False)],
            f"Inactive while in cash. On entry a hard {cfg.stop_label} stop from the fill "
            "protects against a fast breakdown before the trend signal can react."),
            unsafe_allow_html=True)

    if in_pos and pos.get("entry_px"):
        e_dt = pd.Timestamp(pos["entry_date"]); e_px = float(pos["entry_px"])
        days = (pd.Timestamp(preds["target_date"].iloc[-1]) - e_dt).days
        upnl = (c / e_px - 1) * 100
        col3.markdown(_sig_card(
            "Position — currently LONG", "📍", "#16a34a", True,
            [("entry", f"{e_dt.strftime('%b %d')} @ ${e_px:,.2f}", "—", True),
             ("unrealised P&L", f"{upnl:+.2f}%", "≥ 0", upnl >= 0),
             ("days held", f"{days}d", "—", True)],
            f"The trend filter is long {cfg.key}; it stays long until the trend "
            f"signal flips{' or the fixed stop' if cfg.has_stop else ''}."), unsafe_allow_html=True)
    else:
        col3.markdown(_sig_card(
            "Position — currently FLAT", "⚪", "#94a3b8", False,
            [("state", "in cash", "—", False)],
            f"No open position. The filter re-enters when the trend signal "
            f"({TUI['cond']}) turns bullish again."), unsafe_allow_html=True)


def position_panel(label, col, col_container, end=None):
    r = strategy_position(col, end=end)
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

    trig = "U1 + Pure-Regime gate" if IS_DIV else TUI["cond"]
    if r["in_pos_now"] and r["entry_px"]:
        e_px = r["entry_px"]; e_date = pd.Timestamp(r["entry_date"])
        upnl = (px / e_px - 1) * 100; col_pnl = "#16a34a" if upnl >= 0 else "#dc2626"
        days = (as_of - e_date).days
        rows_ = [("Entry", f"{e_date.strftime('%b %d, %Y')} @ ${e_px:,.2f}"),
                 ("Trigger", trig), ("Live price", f"${px:,.2f}"),
                 ("Unrealized P&amp;L", f"<b style='color:{col_pnl}'>{upnl:+.2f}%</b>")]
        _s = _asset_stop(col)
        if _s < 0.999:
            rows_.append(("Stop (−%.0f%%)" % (_s * 100), f"${e_px * (1 - _s):,.2f}"))
        else:
            rows_.append(("Exit", "signal-only — <b>no fixed stop</b>"))
        rows_.append(("Days held", f"{days}d"))
        html = (
            f"<div style='background:#f0fdf4;border:2px solid #16a34a;border-radius:10px;padding:12px 14px;'>"
            f"<div style='font-size:13px;font-weight:700;color:#15803d;margin-bottom:6px;'>"
            f"📍 {label} — LONG</div>" + _tbl(rows_) + "</div>")
        col_container.markdown(html, unsafe_allow_html=True)
        return
    tl = r.get("trade_log") or []
    if tl:
        lt = tl[-1]; ret = lt["ret"] * 100; profit = ret > 0
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


# Shared plotly layout defaults (match the Gold/BTC look).
_PLOT_BG = dict(plot_bgcolor="#f8fafc", paper_bgcolor="#ffffff", hovermode="x unified")
_GRID = dict(showgrid=True, gridcolor="#e2e8f0", zeroline=False)


def render_trend_regime_chart(d_df, key_prefix="live", pos=None):
    """Price + strategy-line chart with the LONG regime shaded green — the same
    visual the Gold Trend app carries for its dual-MA cross, generalised to
    every trend mode.  The shading always comes from ``bt.trend_long_array``
    (the engine's ACTUAL long condition — so for ma_vol it includes the vol
    gate, for crash_dd the stateful shield), and the overlaid lines are the
    mode's own reference levels:

      dual_ma   fast + slow SMA (the cross being traded)
      ma/ma_vol the N-day SMA (ma_vol shading also reflects the vol gate)
      macd      the context SMA (shading = histogram > 0)
      crash_dd  the re-entry SMA + the rolling 52-week high + the crash-exit
                level (52wk-high × (1−dd))

    ``pos`` is the primary asset's ``bt.run_strategy`` result; when given, its
    trade log is overlaid as ▲ entry / ▼ exit markers (GLDM-app style) so the
    executed trades are visible on top of the regime shading.
    """
    if not cfg.is_trend:
        return
    c = d_df["px_close"].dropna()
    if len(c) < 60:
        return
    arr = bt.trend_long_array(cfg, c.to_numpy(float))
    view = c.index >= (c.index.max() - pd.Timedelta(days=730))
    cv = c[view]; idx = cv.index
    long_v = pd.Series(arr, index=c.index)[view].astype(bool)

    m = cfg.strategy_mode
    lines = []          # (series, name, color, dash)
    if m == "dual_ma":
        lines = [(c.rolling(cfg.ma_fast).mean()[view],
                  f"{cfg.ma_fast}-day SMA (fast)", "#2563eb", None),
                 (c.rolling(cfg.ma_slow).mean()[view],
                  f"{cfg.ma_slow}-day SMA (slow)", "#7c3aed", "dot")]
        sub = f"dual-MA {cfg.ma_fast}/{cfg.ma_slow}"
    elif m == "crash_dd":
        hi52 = c.rolling(252, min_periods=60).max()
        lines = [(c.rolling(cfg.dd_reentry_ma).mean()[view],
                  f"{cfg.dd_reentry_ma}-day SMA (re-entry)", "#2563eb", None),
                 (hi52[view], "52-week high", "#f59e0b", "dash"),
                 ((hi52 * (1 - cfg.dd_exit_pct))[view],
                  f"crash exit (−{cfg.dd_exit_pct*100:.0f}% off 52wk high)",
                  "#dc2626", "dot")]
        sub = f"crash-shield {int(cfg.dd_exit_pct*100)}%/SMA{cfg.dd_reentry_ma}"
    elif m == "macd":
        lines = [(c.rolling(cfg.ma_window).mean()[view],
                  f"{cfg.ma_window}-day SMA (context)", "#2563eb", "dot")]
        sub = f"MACD {cfg.macd_fast}/{cfg.macd_slow}/{cfg.macd_signal} (shading = histogram > 0)"
    else:  # "ma" / "ma_vol"
        lines = [(c.rolling(cfg.ma_window).mean()[view],
                  f"{cfg.ma_window}-day SMA", "#2563eb", None)]
        sub = (f"MA{cfg.ma_window} + vol filter (shading includes the vol gate)"
               if m == "ma_vol" else f"MA{cfg.ma_window}")

    fig = go.Figure()
    in_span = False; x0 = None
    for i, (d, on) in enumerate(zip(idx, long_v)):
        if on and not in_span:
            in_span, x0 = True, d
        elif (not on or i == len(idx) - 1) and in_span:
            fig.add_vrect(x0=x0, x1=d, fillcolor="#16a34a", opacity=0.07, line_width=0)
            in_span = False
    fig.add_trace(go.Scatter(x=idx, y=cv, name=f"{cfg.key} close",
                             line=dict(color=cfg.accent, width=1.6)))
    for s, name, col, dash in lines:
        fig.add_trace(go.Scatter(x=idx, y=s, name=name,
                                 line=dict(color=col, width=1.3, dash=dash)))
    # ▲ entry / ▼ exit markers from the primary asset's executed trade log
    # (same styling as the GLDM app's NAV chart), clipped to the 2-year view.
    if pos:
        x_min = idx.min()
        for t in pos.get("trade_log") or []:
            ed = pd.Timestamp(t["entry_date"]); xd = pd.Timestamp(t["exit_date"])
            win_col = "#16a34a" if t["ret"] > 0 else "#dc2626"
            if ed >= x_min:
                fig.add_trace(go.Scatter(
                    x=[ed], y=[t["entry_px"]], mode="markers", showlegend=False,
                    marker=dict(symbol="triangle-up", size=11, color="#16a34a",
                                line=dict(width=1.5, color="white")),
                    hovertemplate=(f"<b>ENTRY</b> {ed:%b %d, %Y} @ "
                                   f"${t['entry_px']:,.2f}<extra></extra>")))
            if xd >= x_min:
                fig.add_trace(go.Scatter(
                    x=[xd], y=[t["exit_px"]], mode="markers", showlegend=False,
                    marker=dict(symbol="triangle-down", size=11, color=win_col,
                                line=dict(width=1.5, color="white")),
                    hovertemplate=(f"<b>EXIT</b> {xd:%b %d, %Y} @ "
                                   f"${t['exit_px']:,.2f} ({t['ret']*100:+.1f}%) "
                                   f"— {t['reason']}<extra></extra>")))
        if pos.get("in_pos_now") and pos.get("entry_date") is not None \
                and pos.get("entry_px") is not None:
            ed = pd.Timestamp(pos["entry_date"])
            if ed >= x_min:
                fig.add_trace(go.Scatter(
                    x=[ed], y=[pos["entry_px"]], mode="markers", name="▲ open entry",
                    marker=dict(symbol="triangle-up", size=13, color="#f59e0b",
                                line=dict(width=1.5, color="white")),
                    hovertemplate=(f"<b>OPEN ENTRY</b> {ed:%b %d, %Y} @ "
                                   f"${pos['entry_px']:,.2f}<extra></extra>")))
    marker_note = " · ▲ entry / ▼ exit" if pos else ""
    fig.update_layout(
        height=300, margin=dict(l=0, r=70, t=24, b=0),
        title=dict(text=(f"{cfg.key} {sub} — green shading = long regime "
                         f"(the traded signal){marker_note}"), font=dict(size=13)),
        yaxis_title="$", legend=dict(orientation="h", yanchor="bottom",
                                     y=1.02, xanchor="right", x=1),
        xaxis=dict(domain=[0.0, 0.9], **_GRID), yaxis=_GRID, **_PLOT_BG)
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_trend_regime")
_HL_BAND_PCT = cfg.hl_band_pct


# The Clean-Breakout gate flags an entry only when NO D1/D2 fired in the prior
# 7 bars (clean_10d). The daily-H/L chart therefore shows at least this many
# bars of history so the whole Clean-Breakout window is visible at a glance.
CLEAN_BREAKOUT_LOOKBACK = 7


def _add_clean_breakout_markers(fig, sub, sigs):
    """Overlay D1 / D2 / D3 / U1 signature markers on the daily-H/L chart.

    ``sigs`` is the compute_trend_signatures() dict; it carries per-bar
    ``d1_hist`` / ``d2_hist`` / ``d3_hist`` / ``u1_hist`` aligned to
    ``sig_dates``. D1 (downtrend / low-break pressure) is anchored to the
    predicted-LOW line and D2 (high-momentum fade) to the predicted-HIGH line —
    the exact arrays the Clean-Breakout gate scans. D3 (exhaustion canary) sits
    on the actual LOW that broke the floor and U1 (bullish breakout pressure)
    on the actual HIGH, so every Pure-Regime signature is visible at a glance."""
    if not sigs or sigs.get("d1_hist") is None or sigs.get("sig_dates") is None:
        return
    dts = pd.to_datetime(sigs["sig_dates"])
    d1_by = dict(zip(dts, [bool(x) for x in sigs["d1_hist"]]))
    d2_by = dict(zip(dts, [bool(x) for x in sigs["d2_hist"]]))
    d3_by = (dict(zip(dts, [bool(x) for x in sigs["d3_hist"]]))
             if sigs.get("d3_hist") is not None else {})
    u1_by = (dict(zip(dts, [bool(x) for x in sigs["u1_hist"]]))
             if sigs.get("u1_hist") is not None else {})
    d1x, d1y, d2x, d2y = [], [], [], []
    d3x, d3y, u1x, u1y = [], [], [], []
    for _, r in sub.iterrows():
        td = pd.Timestamp(r["target_date"])
        if d1_by.get(td):
            d1x.append(td); d1y.append(r["pred_low"])
        if d2_by.get(td):
            d2x.append(td); d2y.append(r["pred_high"])
        if d3_by.get(td):
            d3x.append(td)
            d3y.append(r["actual_low"] if pd.notna(r["actual_low"]) else r["pred_low"])
        if u1_by.get(td):
            u1x.append(td)
            u1y.append(r["actual_high"] if pd.notna(r["actual_high"]) else r["pred_high"])
    if d1x:
        fig.add_trace(go.Scatter(
            x=d1x, y=d1y, mode="markers+text", text=["D1"] * len(d1x),
            textposition="bottom center", textfont=dict(size=10, color="#b91c1c"),
            marker=dict(symbol="triangle-down", size=13, color="#dc2626",
                        line=dict(width=1.4, color="white")),
            name="D1 fired — downtrend pressure",
            hovertemplate="%{x|%b %d}: 🔻 D1 — downtrend pressure "
                          "(≥2 low-breaks in 3d)<extra></extra>"))
    if d2x:
        fig.add_trace(go.Scatter(
            x=d2x, y=d2y, mode="markers+text", text=["D2"] * len(d2x),
            textposition="top center", textfont=dict(size=10, color="#c2410c"),
            marker=dict(symbol="triangle-down", size=13, color="#f97316",
                        line=dict(width=1.4, color="white")),
            name="D2 fired — momentum fade",
            hovertemplate="%{x|%b %d}: 🔻 D2 — momentum fade "
                          "(3d-avg high undershoot)<extra></extra>"))
    if d3x:
        fig.add_trace(go.Scatter(
            x=d3x, y=d3y, mode="markers+text", text=["D3"] * len(d3x),
            textposition="bottom center", textfont=dict(size=10, color="#6d28d9"),
            marker=dict(symbol="triangle-down", size=13, color="#7c3aed",
                        line=dict(width=1.4, color="white")),
            name="D3 fired — exhaustion canary",
            hovertemplate="%{x|%b %d}: 🔻 D3 — exhaustion canary (first low-break "
                          "after ≥3 high-break streak)<extra></extra>"))
    if u1x:
        fig.add_trace(go.Scatter(
            x=u1x, y=u1y, mode="markers+text", text=["U1"] * len(u1x),
            textposition="top center", textfont=dict(size=10, color="#15803d"),
            marker=dict(symbol="triangle-up", size=13, color="#16a34a",
                        line=dict(width=1.4, color="white")),
            name="U1 fired — bullish breakout pressure",
            hovertemplate="%{x|%b %d}: 🔼 U1 — bullish pressure (3d-avg high "
                          "overshoot + ≥2 high-breaks)<extra></extra>"))


def _hl_forecast_fig(d_df, sigs=None, n_bars=None):
    # Show enough history to cover the Clean-Breakout lookback (prior 7 bars) plus
    # the current bar — max(lookback, 7) + 1, so the D1/D2 markers below always
    # span the full window the gate evaluates.
    if n_bars is None:
        n_bars = max(CLEAN_BREAKOUT_LOOKBACK, 7) + 1
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
    fig.add_trace(go.Scatter(x=x, y=sub["pred_high"] * (1 + _HL_BAND_PCT),
                             line=dict(color="rgba(34,139,34,0)"), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=sub["pred_high"] * (1 - _HL_BAND_PCT), fill="tonexty",
                             fillcolor="rgba(34,139,34,0.13)", line=dict(color="rgba(34,139,34,0)"),
                             name=f"HIGH ±{_HL_BAND_PCT*100:.1f}% band", hoverinfo="skip"))
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
    if hl:
        fig.add_trace(go.Scatter(x=[nx, nx], y=[hl["pred_low"], hl["pred_high"]],
                                 mode="markers", marker=dict(symbol="diamond", size=12, color=ACC),
                                 name="Next-bar forecast"))
    # (No SMA overlays here — the strategy's reference lines live on the
    #  dedicated trend-regime chart, keeping the forecast plots clean.)
    _add_clean_breakout_markers(fig, sub, sigs)
    fig.update_layout(height=340, margin=dict(l=0, r=10, t=44, b=0),
                      title=dict(text="📈 Daily H/L — predictions vs actuals · 🔻 D1/D2/D3 + 🔼 U1 "
                                      "signature markers (D1/D2 set the Clean-Breakout window, prior 7 bars)",
                                 font=dict(size=13), x=0, xanchor="left"),
                      yaxis_title="Price ($)", yaxis_tickprefix="$", yaxis_tickformat=",.2f",
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
                      xaxis=_GRID, yaxis=_GRID, **_PLOT_BG)
    return fig


@st.cache_data(ttl=600, show_spinner=False)
def rolling_cone_series(key, as_of_iso, horizon, lookback):
    art = M_7D if horizon == 7 else M_14D
    if art is None:
        return pd.DataFrame()
    d_df = daily[daily.index <= pd.Timestamp(as_of_iso)]
    close = d_df["px_close"]
    feat = tc.build_daily_features(cfg, d_df)[art["feat_cols"]].ffill()
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
        pred = ac * np.exp(c); lo = ac * np.exp(c + q[5]); hi = ac * np.exp(c + q[95])
        if tpos < n:
            tdate = close.index[tpos]; realized = float(close.iloc[tpos]); future = False
        else:
            tdate = close.index[-1] + pd.tseries.offsets.BDay(tpos - (n - 1)); realized = None; future = True
        rows.append(dict(target_date=tdate, pred=pred, lo=lo, hi=hi, realized=realized, future=future))
    return pd.DataFrame(rows)


def _cone_forecast_fig(as_of_iso, horizon, lookback, title, band_rgba, line_col):
    df = rolling_cone_series(cfg.key, as_of_iso, horizon, lookback)
    if df.empty:
        return None
    df["target_date"] = pd.to_datetime(df["target_date"]); x = df["target_date"]
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


_HR_LOOKBACK_HOURS = 23


def _hourly_rangebreaks(x_vals):
    """Rangebreaks that collapse x-axis spans with no bars (market closed:
    overnight, weekends, holidays) so consecutive market-hours bars plot as a
    continuous line instead of a flat overnight connector followed by a jump
    at the next open.  Data-driven — a 24/7 asset has no gaps, so no breaks."""
    idx = pd.DatetimeIndex(pd.to_datetime(np.atleast_1d(x_vals))).unique().sort_values()
    if len(idx) < 3:
        return []
    step = pd.Series(idx).diff().median()
    if pd.isna(step) or step <= pd.Timedelta(0):
        return []
    pad = step / 2
    return [dict(bounds=[prev + pad, nxt - pad])
            for prev, nxt in zip(idx[:-1], idx[1:]) if nxt - prev > step * 1.5]


def _hourly_forecast_fig(as_of_date, is_live, hl=None):
    hourly = get_hourly(cfg.key)
    if hourly is None or hourly.empty or M_HOURLY is None:
        return None
    if not is_live and as_of_date is not None:
        cutoff = pd.Timestamp(as_of_date) + pd.Timedelta(hours=23, minutes=59)
        hourly = hourly[hourly.index <= cutoff]
    if len(hourly) < 30:
        return None
    close = hourly["px_close"]
    feat = tc.build_hourly_features(cfg, hourly)
    cols = M_HOURLY["feat_cols"]; X = feat[cols]
    valid = X.index[~X.isna().any(axis=1)]
    if len(valid) < 5:
        return None
    anchor = valid[-1]
    look = valid[valid >= anchor - pd.Timedelta(hours=_HR_LOOKBACK_HOURS)]
    if len(look) < 3:
        look = valid[-8:]
    # Outside market hours the feed just repeats the last traded price, so any
    # bar whose close is unchanged from the previous bar is a stale off-hours
    # row — drop it from the plotted window.
    moved = close.loc[look].astype(float).diff().ne(0.0)
    moved.iloc[0] = True
    if moved.sum() >= 3:
        look = look[moved.to_numpy()]
    sigma = float(M_HOURLY["sigma"]); yhat = M_HOURLY["model"].predict(X.loc[look])
    tgt_ts, pred_c, act_c, mcol = [], [], [], []
    for k in range(len(look) - 1):
        b = look[k]; nb = look[k + 1]
        c0 = float(close.loc[b]); pc = c0 * np.exp(yhat[k]); ac = float(close.loc[nb])
        tgt_ts.append(nb); pred_c.append(pc); act_c.append(ac)
        correct = np.sign(yhat[k]) == np.sign(np.log(ac / c0))
        mcol.append("seagreen" if correct else "indianred")
    tgt_ts = pd.DatetimeIndex(tgt_ts)
    last_ts = look[-1]; last_close = float(close.loc[last_ts])
    if len(valid) >= 3:
        step = pd.Series(valid[-6:]).diff().median()
        step = step if pd.notna(step) else pd.Timedelta(hours=1)
    else:
        step = pd.Timedelta(hours=1)
    next_ts = last_ts + step
    fc = last_close * np.exp(yhat[-1]); fc_ret = float(yhat[-1])
    fc_hi = last_close * np.exp(yhat[-1] + 1.96 * sigma); fc_lo = last_close * np.exp(yhat[-1] - 1.96 * sigma)

    def _ct(ts):
        idx = pd.DatetimeIndex(pd.to_datetime(np.atleast_1d(ts)))
        return idx.tz_localize("UTC").tz_convert("America/Chicago").tz_localize(None)
    tgt_ct = _ct(tgt_ts); act_win = close.loc[look]; act_ct = _ct(act_win.index)
    last_ct = _ct(last_ts)[0]; next_ct = _ct(next_ts)[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tgt_ct, y=np.array(pred_c) * np.exp(1.96 * sigma),
                             line=dict(color="rgba(65,105,225,0)"), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=tgt_ct, y=np.array(pred_c) * np.exp(-1.96 * sigma), fill="tonexty",
                             fillcolor="rgba(65,105,225,0.18)", line=dict(color="rgba(65,105,225,0)"),
                             name=f"Pred ±{1.96*sigma*100:.2f}% (95% CI) band", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=act_ct, y=act_win.values, mode="lines",
                             line=dict(color="black", width=2), name="Actual close (hourly)",
                             hovertemplate="%{x|%d-%b %H:%M} CT<br>$%{y:,.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=tgt_ct, y=pred_c, mode="lines+markers",
                             line=dict(color="#7c3aed", width=2),
                             marker=dict(color=mcol, size=8, symbol="circle", line=dict(width=1, color="white")),
                             name="● Past hourly predictions (green = correct dir.)",
                             hovertemplate="Past pred for %{x|%d-%b %H:%M} CT<br>$%{y:,.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=tgt_ct, y=act_c, mode="lines+markers",
                             line=dict(color="#0d9488", width=2),
                             marker=dict(color=mcol, size=10, symbol="diamond", line=dict(color="white", width=1.5)),
                             name="◆ Hourly realized close",
                             hovertemplate="Realized %{x|%d-%b %H:%M} CT<br>$%{y:,.2f}<extra></extra>"))
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
    # Daily predicted HIGH/LOW bands belong to the divergence (H/L prediction)
    # strategy; trend-filter apps trade on their regime line (overlaid below)
    # instead, so the H/L lines are omitted there to keep the hourly plot clean.
    if hl and IS_DIV:
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
        hi_dn = max(hl["pred_high"] - hl["band_hi"], mid); lo_up = min(hl["pred_low"] + hl["band_lo"], mid)
        if hl["pred_high"] + hl["band_hi"] > hi_dn:
            fig.add_hrect(y0=hi_dn, y1=hl["pred_high"] + hl["band_hi"],
                          fillcolor="rgba(0,170,0,0.12)", line_width=0, layer="below")
        if lo_up > hl["pred_low"] - hl["band_lo"]:
            fig.add_hrect(y0=hl["pred_low"] - hl["band_lo"], y1=lo_up,
                          fillcolor="rgba(220,30,30,0.12)", line_width=0, layer="below")
    # (No SMA reference lines here — the strategy's SMA levels live on the
    #  dedicated trend-regime chart and in the signal cards, keeping the
    #  hourly forecast clean.)
    title = (f"🕐 {cfg.key} — rolling next-hour close forecast (last 23 hours)"
             if is_live else f"🕐 {cfg.key} hourly forecast as of {pd.Timestamp(as_of_date).date()}")
    fig.update_layout(template="plotly_white", height=420,
                      title=dict(text=title, font=dict(size=13), x=0, xanchor="left"),
                      yaxis_title=f"{cfg.key} / USD", yaxis_tickprefix="$", yaxis_tickformat=",.2f",
                      margin=dict(l=0, r=10, t=46, b=0),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
                      xaxis=dict(title="Time (US Central)", tickformat="%d-%b %H:%M",
                                 rangebreaks=_hourly_rangebreaks(list(act_ct) + [next_ct]),
                                 **_GRID),
                      yaxis=_GRID, **_PLOT_BG)
    return fig


def _macd_hist_fig(d_df, months=9):
    """MACD-histogram bar chart (macd mode only) — the exact quantity the trend
    filter trades on: long into the next bar while the histogram is > 0.  Green
    bars mark a positive (bullish) histogram, red a negative (bearish) one."""
    if cfg.strategy_mode != "macd" or d_df is None or not len(d_df):
        return None
    c = d_df["px_close"].to_numpy(float)
    if len(c) < cfg.macd_slow + cfg.macd_signal:
        return None
    hist = bt.macd_hist_array(cfg, c)          # config-driven fast/slow/signal
    dts = pd.to_datetime(d_df.index)
    n = min(len(hist), max(60, months * 21))   # a readable recent window
    dts, hist = dts[-n:], hist[-n:]
    colors = ["#16a34a" if h > 0 else "#dc2626" for h in hist]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=dts, y=hist, marker_color=colors, name="MACD histogram",
                         hovertemplate="%{x|%b %d, %Y}: %{y:+.4f}<extra>MACD hist</extra>"))
    fig.add_hline(y=0, line_color="#334155", line_width=1)
    fig.update_layout(
        title=dict(text=(f"📊 MACD histogram ({cfg.macd_fast}/{cfg.macd_slow}/{cfg.macd_signal}) "
                         f"— the traded signal (long while &gt; 0)"),
                   font=dict(size=13), x=0, xanchor="left"),
        height=270, margin=dict(l=0, r=10, t=46, b=0), bargap=0.15,
        yaxis_title="Histogram (price units)",
        xaxis=_GRID, yaxis=_GRID, **_PLOT_BG)
    return fig


def render_prediction_plots(d_df, key_prefix, is_live=True, as_of_date=None, sigs=None):
    st.markdown(f"### 🔮 Model forecast charts ({cfg.key})")
    hl = predict_next_daily_hl(d_df)
    fhr = _hourly_forecast_fig(as_of_date, is_live, hl=hl)
    if fhr:
        st.plotly_chart(fhr, use_container_width=True, key=f"{key_prefix}_hr")
    else:
        st.caption("_Hourly forecast unavailable for this date (hourly history is limited)._")
    fhl = _hl_forecast_fig(d_df, sigs=sigs)
    if fhl:
        st.plotly_chart(fhl, use_container_width=True, key=f"{key_prefix}_hl")
    as_of_iso = str(d_df.index[-1])
    f7 = _cone_forecast_fig(as_of_iso, 7, 90, "📅 7-day close-price cone", "rgba(147,197,253,0.28)", "#2563eb")
    f14 = _cone_forecast_fig(as_of_iso, 14, 120, "📆 14-day close-price cone", "rgba(196,181,253,0.30)", "#7c3aed")
    if f7:
        st.plotly_chart(f7, use_container_width=True, key=f"{key_prefix}_c7")
    if f14:
        st.plotly_chart(f14, use_container_width=True, key=f"{key_prefix}_c14")
    # MACD histogram — the traded signal — anchored below every other chart.
    fmh = _macd_hist_fig(d_df)
    if fmh:
        st.plotly_chart(fmh, use_container_width=True, key=f"{key_prefix}_macdhist")


# ════════════════════════════════════════════════════════════════════════
# Backtesting dashboard
# ════════════════════════════════════════════════════════════════════════
def _equity_fig(r, label, col, title):
    INIT = 100_000.0
    dts = pd.to_datetime(r["dates"]); nav = r["strat"] * INIT; bh = r["bh"] * INIT
    date_to_nav = dict(zip([pd.Timestamp(d) for d in r["dates"]], nav))
    fig = go.Figure()
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
                          line_width=0, name=f"🐂 Bull Regime ({cfg.key})" if is_bull else f"🐻 Bear/Neutral ({cfg.key})",
                          showlegend=show)
            bull_leg = bull_leg or is_bull; bear_leg = bear_leg or (not is_bull)
    fig.add_trace(go.Scatter(x=dts, y=bh, name=f"Buy & Hold {label}",
                             line=dict(color="#94a3b8", width=1.5, dash="dot"),
                             hovertemplate="%{x|%b %d, %Y}: $%{y:,.0f}<extra>Buy & Hold</extra>"))
    fig.add_trace(go.Scatter(x=dts, y=nav, name=f"{cfg.strategy_name} ({label})",
                             line=dict(color=ACC, width=2.5),
                             hovertemplate="%{x|%b %d, %Y}: $%{y:,.0f}<extra>Strategy</extra>"))
    gseries = preds.set_index(pd.to_datetime(preds["target_date"]))["px_close"].reindex(dts)
    fig.add_trace(go.Scatter(x=dts, y=gseries.values, name=f"{cfg.key} price (signal)",
                             line=dict(color=SIGNAL_LINE, width=1.2), yaxis="y2", opacity=0.7,
                             hovertemplate="%{x|%b %d, %Y}: $%{y:,.2f}<extra>" + cfg.key + "</extra>"))
    fig.add_hline(y=INIT, line_dash="dash", line_color="#64748b", line_width=1, opacity=0.4,
                  annotation_text="  $100k start", annotation_position="bottom right")
    for t in r["trade_log"]:
        ed = pd.Timestamp(t["entry_date"]); xd = pd.Timestamp(t["exit_date"])
        win_col = "#16a34a" if t["ret"] > 0 else "#dc2626"
        if ed in date_to_nav:
            fig.add_trace(go.Scatter(x=[ed], y=[date_to_nav[ed]], mode="markers", showlegend=False,
                                     marker=dict(symbol="triangle-up", size=12, color="#16a34a",
                                                 line=dict(width=1.5, color="white")),
                                     hovertemplate=f"<b>BUY {label}</b> {ed:%b %d} @ ${t['entry_px']:,.2f}<extra></extra>"))
        if xd in date_to_nav:
            fig.add_trace(go.Scatter(x=[xd], y=[date_to_nav[xd]], mode="markers", showlegend=False,
                                     marker=dict(symbol="triangle-down", size=12, color=win_col,
                                                 line=dict(width=1.5, color="white")),
                                     hovertemplate=(f"<b>SELL {label}</b> {xd:%b %d} @ ${t['exit_px']:,.2f} "
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
        yaxis2=dict(title=f"{cfg.key} ($)", overlaying="y", side="right", showgrid=False,
                    zeroline=False, anchor="x", tickprefix="$", tickformat=",.2f", color=SIGNAL_LINE),
        **_PLOT_BG)
    return fig


def _trade_log_table(r, label, col):
    INIT = 100_000.0; rows = []; nav = INIT
    for i, t in enumerate(r["trade_log"], 1):
        nav *= (1 + t["ret"])
        result = "🛑 SL EXIT" if "stop" in t["reason"] else ("✓ WIN" if t["ret"] > 0 else "✗ LOSS")
        rows.append({
            "#": i, "Entry": pd.Timestamp(t["entry_date"]).strftime("%b %d, %Y"),
            f"Buy {label} @": f"${t['entry_px']:,.2f}",
            "Exit": pd.Timestamp(t["exit_date"]).strftime("%b %d, %Y"),
            f"Sell {label} @": f"${t['exit_px']:,.2f}",
            "P&L": f"{t['ret']*100:+.1f}%", "Result": result,
            "Days": (pd.Timestamp(t["exit_date"]) - pd.Timestamp(t["entry_date"])).days,
            "Exit signal": t["reason"], "NAV After": f"${nav:,.0f}"})
    if r.get("in_pos_now") and r.get("entry_px"):
        last_px = float(preds[col].iloc[-1]) if col in preds else None
        unr = (last_px / r["entry_px"] - 1) * 100 if last_px else 0.0
        rows.append({
            "#": len(r["trade_log"]) + 1, "Entry": pd.Timestamp(r["entry_date"]).strftime("%b %d, %Y"),
            f"Buy {label} @": f"${r['entry_px']:,.2f}", "Exit": "⏳ OPEN", f"Sell {label} @": "—",
            "P&L": f"{unr:+.1f}% (unrlzd)", "Result": "🟡 OPEN",
            "Days": (pd.Timestamp(preds['target_date'].iloc[-1]) - pd.Timestamp(r["entry_date"])).days,
            "Exit signal": "—", "NAV After": "—"})
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
                   "P&L at execution prices; costs/slippage not modelled.")


def _metrics_table_html(label, col):
    def cell(txt, good=None, bold=False):
        color = "#16a34a" if good is True else "#dc2626" if good is False else "#334155"
        w = "700" if (bold or good is not None) else "500"
        return f"<td style='padding:6px 10px;text-align:center;color:{color};font-weight:{w};'>{txt}</td>"
    metric_rows = [("Total Return", "ret"), ("CAGR", "cagr"), ("Max Drawdown", "mdd"),
                   ("Sharpe", "sharpe"), ("Win Rate", "wr"), ("Trades", "n")]
    per = []
    for lbl, s, e in cfg.periods:
        r = bt.run_strategy(cfg, preds, sig, col, oos_start=s, end=e)
        sm = bt._metrics(r["strat"], r["dates"]); bm = bt._metrics(r["bh"], r["dates"])
        wr = (r["trades"] > 0).mean() * 100 if len(r["trades"]) else 0
        per.append((lbl, sm, bm, wr, len(r["trades"])))
    hdr = f"<tr style='background:{ACCD};color:white;'><th style='padding:9px 10px;text-align:left;'>Metric</th>"
    for lbl, *_ in per:
        bg = "#14532d" if "OOS" in lbl or "recent" in lbl else ACCD
        hdr += (f"<th colspan='2' style='padding:9px 8px;text-align:center;background:{bg};"
                f"border-left:3px solid #fff;'>{lbl}</th>")
    hdr += "</tr>"
    subr = f"<tr style='background:{ACC};color:white;font-size:11px;'><th></th>"
    for lbl, *_ in per:
        bg = "#166534" if "OOS" in lbl or "recent" in lbl else ACC
        subr += (f"<th style='padding:4px 8px;background:{bg};border-left:3px solid #fff;'>Strategy</th>"
                 f"<th style='padding:4px 8px;background:{bg};'>Buy&amp;Hold</th>")
    subr += "</tr>"
    body = ""
    for ri, (lbl_m, key) in enumerate(metric_rows):
        bg = ACCBG if ri % 2 == 0 else "#ffffff"
        body += f"<tr style='background:{bg};'><td style='padding:6px 10px;font-weight:600;color:#334155;'>{lbl_m}</td>"
        for lbl, sm, bm, wr, ntr in per:
            if key == "ret":
                sv, bv = sm["total_ret"] * 100, bm["total_ret"] * 100
                body += cell(f"{sv:+.1f}%", sv > bv) + cell(f"{bv:+.1f}%")
            elif key == "cagr":
                sv, bv = sm["cagr"] * 100, bm["cagr"] * 100
                body += cell(f"{sv:+.1f}%", sv > bv) + cell(f"{bv:+.1f}%")
            elif key == "mdd":
                sv, bv = sm["mdd"] * 100, bm["mdd"] * 100
                body += cell(f"{sv:.1f}%", sv > bv) + cell(f"{bv:.1f}%")
            elif key == "sharpe":
                sv, bv = sm["sharpe"], bm["sharpe"]
                body += cell(f"{sv:.2f}", sv > bv) + cell(f"{bv:.2f}")
            elif key == "wr":
                body += cell(f"{wr:.0f}%") + cell("—")
            else:
                body += cell(f"{ntr}") + cell("—")
        body += "</tr>"
    return (f"<div style='overflow-x:auto;margin:8px 0;'>"
            f"<table style='width:100%;border-collapse:collapse;font-size:13px;"
            f"border:1px solid #e2e8f0;border-radius:8px;overflow:hidden;'>"
            f"<thead>{hdr}{subr}</thead><tbody>{body}</tbody></table></div>"
            f"<p style='font-size:11px;color:#64748b;margin:2px 0 10px;'>"
            f"🟢 green = Strategy beats Buy&amp;Hold on that metric · 🔴 red = worse · "
            f"Max Drawdown closer to 0 is better. The H/L model is out-of-sample; "
            f"strategy parameters were tuned on these windows (in-sample-selected).</p>")


def render_backtest_dashboard(label, col):
    st.markdown(f"## 📊 {label} — {cfg.key} Signal-Driven Backtesting")
    render_strategy_card()
    # read the per-asset stop from the data field (not the stop_for method) so a
    # stale/hot-reloaded TickerConfig instance without the method can't crash.
    _s = getattr(cfg, "stop_by_asset", {}).get(col, cfg.fixed_stop)
    if _s != cfg.fixed_stop:
        _txt = "no fixed stop" if _s >= 0.999 else f"a wider −{_s*100:.0f}% stop"
        st.caption(f"↳ **{label}** trades the same {cfg.key} signal but with {_txt} "
                   f"(vs {cfg.stop_label} on {cfg.key}) — a 1× stop is too tight for this "
                   f"higher-beta sibling, so signal-driven exits give a higher win-rate and Sharpe.")
    st.caption(f"The {cfg.key} daily H/L signal model is fit once on the pre-{cfg.oos_start[:4]} "
               "window and predicts every later bar (model-OOS). The strategy layer on top — "
               "engine mode, MA/MACD windows, divergence thresholds and stops — was selected by "
               "sweeps run over these same periods, so the levels shown are in-sample-selected, "
               "not blind; periods that predate the model's fit window replay in-sample "
               "predictions. Divergence fills are post-signal (decide at close i−1, fill at "
               "close i) since the 2026-07-25 look-ahead fix. NAV starts at $100k; "
               "costs/slippage not modelled.")
    st.markdown(_metrics_table_html(label, col), unsafe_allow_html=True)
    period_tabs = st.tabs([lbl for lbl, _, _ in cfg.periods])
    for (lbl, s, e), tb in zip(cfg.periods, period_tabs):
        with tb:
            r = bt.run_strategy(cfg, preds, sig, col, oos_start=s, end=e)
            if len(r["strat"]) < 2:
                st.info("Not enough bars in this window.")
                continue
            sm = bt._metrics(r["strat"], r["dates"]); bm = bt._metrics(r["bh"], r["dates"])
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Strategy return", f"{sm['total_ret']*100:+.1f}%", f"CAGR {sm['cagr']*100:+.1f}%")
            k2.metric("Buy & Hold", f"{bm['total_ret']*100:+.1f}%", f"CAGR {bm['cagr']*100:+.1f}%")
            k3.metric("Max drawdown", f"{sm['mdd']*100:.1f}%", f"vs B&H {bm['mdd']*100:.1f}%",
                      delta_color="inverse")
            k4.metric("Sharpe", f"{sm['sharpe']:.2f}", f"vs B&H {bm['sharpe']:.2f}")
            st.plotly_chart(_equity_fig(r, label, col, f"{cfg.strategy_name} ({label}) vs Buy & Hold — {lbl}"),
                            use_container_width=True, key=f"{label}_{s}_{e}_eq")
            dts = pd.to_datetime(r["dates"]); figd = go.Figure()
            figd.add_trace(go.Scatter(x=dts, y=bt.drawdown_series(r["strat"]) * 100,
                                      name="Strategy DD", fill="tozeroy", line=dict(color=ACC)))
            figd.add_trace(go.Scatter(x=dts, y=bt.drawdown_series(r["bh"]) * 100,
                                      name="Buy & Hold DD", line=dict(color="#94a3b8", dash="dot")))
            figd.update_layout(height=190, margin=dict(l=0, r=70, t=6, b=0),
                               yaxis_title="Drawdown %", yaxis_ticksuffix="%",
                               legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
                               xaxis=dict(domain=[0.0, 0.9], **_GRID), yaxis=_GRID, **_PLOT_BG)
            st.plotly_chart(figd, use_container_width=True, key=f"{label}_{s}_{e}_dd")
            st.markdown("#### 📋 Trade log")
            _trade_log_table(r, label, col)


# ════════════════════════════════════════════════════════════════════════
# Tabs
# ════════════════════════════════════════════════════════════════════════
_bt_tab_labels = [f"📊 {lbl} Backtesting" for lbl, _ in TRADED]
_tabs = st.tabs(["🔴 Live (rolling now+1h)", "🕒 Historical replay", *_bt_tab_labels, "🧠 Explain"])
tab_live, tab_hist = _tabs[0], _tabs[1]
tab_bt = _tabs[2:2 + len(TRADED)]
tab_explain = _tabs[-1]


def render_live_dashboard(as_of_date=None, is_live=True):
    if is_live:
        d_df = daily
        hourly = get_hourly(cfg.key)
        ph = predict_next_hour(hourly)
    else:
        d_df = daily[daily.index <= pd.Timestamp(as_of_date)]
        ph = None
    hl = predict_next_daily_hl(d_df)
    dt = predict_day_type(d_df)
    sent = tc.macro_sentiment(cfg, d_df).dropna()
    sent_now = float(sent.iloc[-1]) if len(sent) else np.nan
    sigs = signatures_asof(d_df.index[-1] if not is_live else completed_all["target_date"].iloc[-1])
    mst = ma_state(d_df)

    last_px = float(d_df["px_close"].iloc[-1]); prev_px = float(d_df["px_close"].iloc[-2])

    def _px_chg(col):
        if col not in d_df or d_df[col].dropna().shape[0] < 2:
            return None, None
        s = d_df[col].dropna()
        return float(s.iloc[-1]), (float(s.iloc[-1]) / float(s.iloc[-2]) - 1) * 100

    # ── Row 1: primary + siblings + sentiment + regime ──
    cols = st.columns(5)
    cols[0].metric(f"{cfg.key} close", f"${last_px:,.2f}", f"{(last_px/prev_px-1)*100:+.2f}% d/d")
    ci = 1
    for lbl, col in TRADED:
        if col == "px_close":
            continue
        spx, schg = _px_chg(col)
        cols[ci].metric(f"{lbl} (traded)", f"${spx:,.2f}" if spx else "—",
                        f"{schg:+.2f}% d/d" if schg is not None else None)
        ci += 1
    if ci < 3 and not IS_DIV and mst:
        _ma = mst["ma"]
        cols[ci].metric(f"{cfg.key} {mst['line_label']}", f"${_ma:,.2f}",
                        f"${last_px-_ma:+,.2f} vs close",
                        delta_color="normal" if last_px >= _ma else "inverse")
        ci += 1
    if not np.isnan(sent_now):
        mood = "Bullish" if sent_now >= 60 else "Bearish" if sent_now <= 40 else "Neutral"
        cols[ci].metric(cfg.sentiment_label, f"{sent_now:.0f}/100", mood); ci += 1
    if ci <= 4:
        _bull = sigs.get("bull_regime") if sigs else None
        cols[4].metric("Market regime", ("🐂 BULL" if _bull else "🐻 BEAR/NEUTRAL") if _bull is not None else "—",
                       delta=("↑MA20 & rising" if _bull else "below MA20 or flat") if _bull is not None else None,
                       delta_color="normal" if _bull else "inverse")

    # ── Row 2: next-hour forecast + CI + MA + predicted daily H/L ──
    d1c, d2c, d3c, d4c, d5c = st.columns(5)
    if ph:
        d1c.metric(f"{cfg.key} next-hour (pred)", f"${ph['pred_close']:,.2f}", f"{ph['ret']*100:+.2f}%")
        d2c.metric("95% CI band", f"${ph['lo']:,.2f}–${ph['hi']:,.2f}", f"±{1.96*ph['sigma']*100:.2f}%")
    else:
        d1c.metric(f"{cfg.key} next-hour (pred)", "— (replay)"); d2c.metric("95% CI band", "—")
    _ma20 = sigs.get("ma20_value") if sigs else None
    d3c.metric(f"{cfg.key} 20-day MA", f"${_ma20:,.2f}" if _ma20 else "—",
               (f"${last_px-_ma20:+,.2f} vs close" if _ma20 else None),
               delta_color="normal" if (_ma20 and last_px >= _ma20) else "inverse")
    if hl:
        d4c.metric(f"{cfg.key} daily High (pred)", f"${hl['pred_high']:,.2f}",
                   f"{(hl['pred_high']/hl['last_close']-1)*100:+.2f}% vs close")
        d5c.metric(f"{cfg.key} daily Low (pred)", f"${hl['pred_low']:,.2f}",
                   f"{(hl['pred_low']/hl['last_close']-1)*100:+.2f}% vs close")
        if dt:
            d4c.caption(f"Day-type: **{dt['top']}** ({max(dt['probs'].values())*100:.0f}%)")
    else:
        d4c.metric(f"{cfg.key} daily High (pred)", "—"); d5c.metric(f"{cfg.key} daily Low (pred)", "—")

    end = None if is_live else as_of_date
    primary_pos = strategy_position(TRADED[0][1], end=end)

    if IS_DIV:
        st.markdown(f"### 🔔 Trend-Signature Alert  ·  _signals derived from the {cfg.key} daily H/L model_")
        render_signatures(sigs)
        st.markdown("#### 🚪 Entry-gate conditions  ·  _what turns a U1 pressure signal into an actual entry_")
        render_gate_signatures(sigs)
    else:
        st.markdown(f"### 🔔 Trend-Filter Signal  ·  _the {TUI['headline']} this app trades on_")
        render_ma_signatures(mst, primary_pos)
        render_trend_regime_chart(d_df, key_prefix=("live" if is_live else "hist"),
                                  pos=primary_pos)
        with st.expander("🔬 Divergence read (U1 / D2 / D3) — context only, not traded", expanded=False):
            st.caption(f"For parity with the Gold/BTC apps. {cfg.key} is traded by the "
                       f"{TUI['headline']} above, so these divergence "
                       "signatures do **not** open or close positions here.")
            render_signatures(sigs)

    st.markdown(f"### 🎯 Strategy — {cfg.strategy_name}")
    render_strategy_card()
    st.markdown("#### Strategy conditions (live)")
    render_conditions_box(sigs, mst, pos=primary_pos)
    st.markdown("#### Current positions")
    if _STOP_SIBLINGS:
        st.markdown(
            f"<div style='font-size:12px;color:#64748b;margin:-6px 0 8px;'>↳ {_sibling_stop_note()}</div>",
            unsafe_allow_html=True)
    pcols = st.columns(max(2, len(TRADED)))
    for (lbl, col), pc in zip(TRADED, pcols):
        position_panel(lbl, col, pc, end=end)

    st.markdown("---")
    render_prediction_plots(d_df, key_prefix=("live" if is_live else "hist"),
                            is_live=is_live, as_of_date=as_of_date, sigs=sigs)


with tab_live:
    render_live_dashboard(is_live=True)


with tab_hist:
    st.markdown(f"### 🕒 Historical replay — {' & '.join(l for l, _ in TRADED)}")
    st.caption("Replay the trend-signals, forecasts and positions exactly as they stood "
               "at the close of any past trading day.")
    dates_avail = pd.to_datetime(completed_all["target_date"])
    dates_avail = dates_avail[dates_avail >= pd.Timestamp(cfg.oos_start)]
    if len(dates_avail) == 0:
        st.info("No out-of-sample bars available yet.")
    else:
        min_d, max_d = dates_avail.min().date(), dates_avail.max().date()
        skey = f"{cfg.key}_hist_date"
        if skey not in st.session_state:
            st.session_state[skey] = max_d
        cprev, cpick, cnext = st.columns([1, 3, 1])
        with cprev:
            if st.button("◀ prev day", use_container_width=True, key=f"{cfg.key}_prev"):
                prior = dates_avail[dates_avail < pd.Timestamp(st.session_state[skey])]
                if len(prior):
                    st.session_state[skey] = prior.max().date()
        with cnext:
            if st.button("next day ▶", use_container_width=True, key=f"{cfg.key}_next"):
                later = dates_avail[dates_avail > pd.Timestamp(st.session_state[skey])]
                if len(later):
                    st.session_state[skey] = later.min().date()
        with cpick:
            st.date_input("Replay date", min_value=min_d, max_value=max_d, key=skey)
        picked = pd.Timestamp(st.session_state[skey])
        avail_le = dates_avail[dates_avail <= picked]
        if len(avail_le) == 0:
            st.warning("No completed bar on or before that date.")
        else:
            snapped = avail_le.max()
            if snapped.date() != picked.date():
                st.caption(f"⚠️ Snapped to last completed bar: **{snapped.date()}**")
            render_live_dashboard(as_of_date=snapped, is_live=False)


for (lbl, col), tb in zip(TRADED, tab_bt):
    with tb:
        render_backtest_dashboard(lbl, col)


with tab_explain:
    st.subheader(f"How the {cfg.key} app works")
    drivers = ", ".join(cfg.macro_syms.keys())
    st.markdown(f"""
**Asset.** {cfg.key} = {cfg.name}. {cfg.blurb}

**Asset-specific features** (replacing BTC's crypto inputs): the macro drivers
`{drivers}` plus a purpose-built **{cfg.sentiment_label} 0–100** composite (each
driver signed so *up = bullish for {cfg.key}*, then rank-scaled like a Fear & Greed
index), on top of the usual technical toolkit (returns, vol, ATR, RSI, MACD,
Bollinger width, distance from moving averages / extremes) and seasonality.

**Five models** (`src/tickers/train_ticker.py {cfg.key}`): hourly next-close
(ridge + 95% CI), daily High/Low (calibrated ridge bands), 7-day & 14-day close
cones, and a 3-class day-type classifier.

**Strategy — {cfg.strategy_name}.** {('The Gold/BTC divergence Pure-Regime system, re-tuned for this asset: enter on a U1 bullish divergence (3-day centered `err_hi` > +' + f'{cfg.u1_errhi_min:.2f}' + '%) confirmed inside the Pure-Regime gate, exit on D2 (< ' + f'{cfg.d2_errhi_max:+.2f}' + '%)' + (' / D1 downtrend' if cfg.use_d1_exit else '') + ' / D3 exhaustion' + (' or a fixed ' + cfg.stop_label + ' stop.' if cfg.has_stop else ' (no fixed stop).')) if IS_DIV else ('A ' + TUI['headline'] + ': ' + TUI['core'] + ', otherwise move to cash' + ((' (with a ' + cfg.stop_label + ' fixed stop).') if cfg.has_stop else '.'))} The strategy and its parameters were chosen by a per-asset search over the full price history to maximise risk-adjusted return without a drawdown worse than buy-&-hold.

{('**Per-instrument exits.** ' + _sibling_stop_note_md()) if _STOP_SIBLINGS else ''}

{cfg.results_note}

**Honest framing.** Intraday direction is ~coin-flip (like BTC and gold); the
hourly model's value is a tight, well-calibrated CI, not a directional bet. The
edge is in the trend regime and risk control — quantified in the Backtesting tabs.
""")
    if cfg.eval_note:
        _sibs = ", ".join(lbl for lbl, _ in TRADED[1:]) or "sibling"
        _eval_hdr = ("🛢️ XLE vs OIH — which signal drives OIH?" if cfg.key == "XLE"
                     else f"🔀 {_sibs} — driven off the {cfg.key} signal (and how it exits)")
        st.markdown(f"#### {_eval_hdr}")
        st.markdown(cfg.eval_note)
    st.markdown(f"#### Model freshness (`train_end`) — {cfg.key}")
    for label, art in (("hourly close", M_HOURLY), ("daily H/L", M_HL),
                       ("7-day cone", M_7D), ("14-day cone", M_14D),
                       ("3-class day type", M_DT)):
        st.caption(f"&bull; {label}: `{_fresh(art)}`", unsafe_allow_html=True)
    st.caption(f"Retrain with `python src/tickers/train_ticker.py {cfg.key}`, then Refresh now.")
