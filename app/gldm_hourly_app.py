"""Streamlit app — GLDM (SPDR Gold MiniShares) forecasting & trading.

The gold counterpart of btc_hourly_app.py.  Everything gold-specific lives in
app/gldm_core.py (data, features, strategy) and models/gldm/*.joblib (trained
by src/gldm/train_gldm.py).  This module is self-contained: it never touches
the BTC app's code or artefacts.

Tabs
  📡 Live        current price, next-hour close forecast, next daily H/L,
                 gold-macro sentiment, day-type, trend-signature alert
  🗓️ H/L & Cones daily High/Low forecast + 7-day & 14-day close cones
  📈 Strategy    MA-trend-filter dashboard (GLDM / UGL / GDX) vs buy & hold
  🔬 Signals     divergence trend-signature diagnostics (D1/D2/D3/U1/V)
  📚 Explain     models, features, methodology, model freshness, honest framing
"""
import os
import sys
import warnings
from pathlib import Path
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

_APP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _APP_DIR.parent
sys.path.insert(0, str(_APP_DIR))
sys.path.insert(0, str(_REPO_ROOT))

# sklearn's pickled models reference the compiled _loss extension under its
# bare name on some builds — register it before any joblib.load (see BTC app).
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

import gldm_core as gc
import backtest_gldm as btg   # single source of the prediction / signal / sim logic

st.set_page_config(page_title="GLDM Gold Forecaster", page_icon="🥇",
                   layout="wide", initial_sidebar_state="expanded")

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

st.title("🥇 GLDM — Gold ETF forecast & trend strategy")
st.caption(
    "SPDR Gold MiniShares (GLDM) tracks spot gold (~1/100 oz, 0.10% ER). "
    "Models: ridge on log-returns (hourly close), ridge H/L bands & close cones, "
    "logistic day-type — driven by gold's real macro factors (USD index, 10-year "
    "real yields, silver, VIX). **Honest framing:** intraday gold direction is "
    "near-random (~50%); the edge is in the *trend regime* and *risk control*."
)


# ════════════════════════════════════════════════════════════════════════
# Loaders (cached)
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


@st.cache_data(ttl=gc.CACHE_TTL if hasattr(gc, "CACHE_TTL") else 300, show_spinner="Fetching GLDM daily data…")
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
    """Calibrated daily H/L predictions + signals for the whole history."""
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


# load all artefacts
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


# ════════════════════════════════════════════════════════════════════════
# Prediction helpers (self-contained inference)
# ════════════════════════════════════════════════════════════════════════
def predict_next_hour(hourly: pd.DataFrame):
    if M_HOURLY is None or hourly is None or hourly.empty:
        return None
    feat = gc.build_hourly_features(hourly)
    cols = M_HOURLY["feat_cols"]
    x = feat[cols].iloc[[-1]]
    if x.isna().any(axis=None):
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


def predict_next_daily_hl(daily: pd.DataFrame):
    """Use the trained daily H/L model + its calibration to forecast the next
    bar's high & low from the latest close."""
    if M_HL is None:
        return None
    feat = gc.build_daily_features(daily)
    cols = M_HL["feat_cols"]
    x = feat[cols].ffill().iloc[[-1]]
    if x.isna().any(axis=None):
        return None
    last_close = float(daily["gldm_close"].iloc[-1])
    bh = M_HL.get("bias_high", 0.0); bl = M_HL.get("bias_low", 0.0)
    ph = last_close * (1 + float(M_HL["model_high"].predict(x)[0])) + bh * last_close
    pl = last_close * (1 + float(M_HL["model_low"].predict(x)[0])) + bl * last_close
    return dict(last_close=last_close, pred_high=ph, pred_low=pl,
                band_hi=M_HL["sigma_high"] * last_close,
                band_lo=M_HL["sigma_low"] * last_close)


def predict_cone(art, daily: pd.DataFrame):
    if art is None:
        return None
    feat = gc.build_daily_features(daily)
    cols = art["feat_cols"]
    x = feat[cols].ffill().iloc[[-1]]
    if x.isna().any(axis=None):
        return None
    last_close = float(daily["gldm_close"].iloc[-1])
    central_r = float(art["model"].predict(x)[0])
    q = art["quantiles"]
    return dict(last_close=last_close, horizon=art["horizon"],
                central=last_close * np.exp(central_r),
                p5=last_close * np.exp(central_r + q[5]),
                p25=last_close * np.exp(central_r + q[25]),
                p75=last_close * np.exp(central_r + q[75]),
                p95=last_close * np.exp(central_r + q[95]))


def predict_day_type(daily: pd.DataFrame):
    if M_DT is None:
        return None
    feat = gc.build_daily_features(daily)
    cols = M_DT["feat_cols"]
    x = feat[cols].ffill().iloc[[-1]]
    if x.isna().any(axis=None):
        return None
    proba = M_DT["model"].predict_proba(x)[0]
    classes = M_DT["model"].classes_
    label_map = M_DT["classes"]
    pm = {label_map[int(c)]: float(p) for c, p in zip(classes, proba)}
    top = max(pm, key=pm.get)
    return dict(probs=pm, top=top)


# ════════════════════════════════════════════════════════════════════════
# Tabs
# ════════════════════════════════════════════════════════════════════════
tab_live, tab_cones, tab_strat, tab_sig, tab_explain = st.tabs(
    ["📡 Live", "🗓️ H/L & Cones", "📈 Strategy", "🔬 Signals", "📚 Explain"])

# ── Common: latest trend signature ────────────────────────────────────────
completed = preds[preds["actual_high"].notna() & preds["actual_low"].notna()].tail(45)
ts = gc.compute_trend_signatures(completed)


# ═════════════════════════════ LIVE ══════════════════════════════════════
with tab_live:
    hourly = get_hourly()
    ph = predict_next_hour(hourly)
    hl = predict_next_daily_hl(daily)
    dt = predict_day_type(daily)
    sent = gc.gold_macro_sentiment(daily).dropna()
    sent_now = float(sent.iloc[-1]) if len(sent) else np.nan

    c1, c2, c3, c4 = st.columns(4)
    last_px = float(daily["gldm_close"].iloc[-1])
    prev_px = float(daily["gldm_close"].iloc[-2])
    c1.metric("GLDM last close", f"${last_px:,.2f}",
              f"{(last_px/prev_px-1)*100:+.2f}% d/d")
    if ph:
        c2.metric("Next-hour close (pred)", f"${ph['pred_close']:,.2f}",
                  f"{ph['ret']*100:+.2f}%")
        c2.caption(f"95% CI ${ph['lo']:,.2f} – ${ph['hi']:,.2f}")
    if not np.isnan(sent_now):
        mood = ("Bullish backdrop" if sent_now >= 60 else
                "Bearish backdrop" if sent_now <= 40 else "Neutral")
        c3.metric("Gold macro sentiment", f"{sent_now:.0f}/100", mood)
    if dt:
        c4.metric("Tomorrow — day type", dt["top"],
                  f"{max(dt['probs'].values())*100:.0f}% conf")

    st.markdown("---")
    lc, rc = st.columns([2, 1])
    with lc:
        st.subheader("Next daily High / Low forecast")
        if hl:
            st.markdown(
                f"- **Predicted High:** ${hl['pred_high']:,.2f}  "
                f"_(band ±${hl['band_hi']:,.2f})_\n"
                f"- **Last close:**    ${hl['last_close']:,.2f}\n"
                f"- **Predicted Low:**  ${hl['pred_low']:,.2f}  "
                f"_(band ±${hl['band_lo']:,.2f})_")
        # recent price with predicted band
        recent = daily.tail(60)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=recent.index, open=recent["gldm_open"], high=recent["gldm_high"],
            low=recent["gldm_low"], close=recent["gldm_close"], name="GLDM"))
        if hl:
            nx = recent.index[-1] + pd.Timedelta(days=1)
            fig.add_trace(go.Scatter(x=[nx], y=[hl["pred_high"]], mode="markers",
                                     marker=dict(symbol="triangle-down", size=12, color="#c0392b"),
                                     name="pred high"))
            fig.add_trace(go.Scatter(x=[nx], y=[hl["pred_low"]], mode="markers",
                                     marker=dict(symbol="triangle-up", size=12, color="#27ae60"),
                                     name="pred low"))
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0),
                          xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    with rc:
        st.subheader("Trend-signature alert")
        if ts:
            colours = {"STRATEGY_BUY": "🟢", "WATCH_UP": "🟩", "NEUTRAL": "⚪",
                       "WATCH_DN": "🟡", "ELEVATED_DN": "🟠", "HIGH_DN": "🔴"}
            st.markdown(f"### {colours.get(ts['alert_level'],'⚪')} {ts['alert_level'].replace('_',' ')}")
            st.markdown(
                f"- U1 (bullish): **{'✅' if ts['u1_triggered'] else '—'}**\n"
                f"- D2 (fading): **{'⚠️' if ts['d2_triggered'] else '—'}**\n"
                f"- D3 (exhaustion): **{'⚠️' if ts['d3_triggered'] else '—'}**\n"
                f"- Bull regime (>MA20 rising): **{'✅' if ts['bull_regime'] else '—'}**\n"
                f"- err_hi 3d avg: `{ts['err_hi_ma3']:+.3f}%`\n"
                f"- close streak: `{ts['streak']:+d}`")
        else:
            st.info("Not enough completed bars for signatures yet.")
        # MA50 primary signal
        ma50 = float(daily["gldm_close"].tail(50).mean())
        long_now = last_px > ma50
        st.markdown("---")
        st.markdown(f"**MA50 trend filter:** "
                    f"{'🟢 LONG' if long_now else '⚪ CASH'}  \n"
                    f"close ${last_px:,.2f} vs MA50 ${ma50:,.2f} "
                    f"({(last_px/ma50-1)*100:+.2f}%)")


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
                fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="GLDM", line=dict(color="#b8860b")))
                # linear interpolation from last close to each cone quantile at horizon
                x0, y0 = hist.index[-1], cone["last_close"]
                for key, clr, nm in (("p95", "rgba(192,57,43,0.25)", "P95"),
                                     ("p75", "rgba(192,57,43,0.15)", "P75"),
                                     ("central", "#c0392b", "central"),
                                     ("p25", "rgba(39,174,96,0.15)", "P25"),
                                     ("p5", "rgba(39,174,96,0.25)", "P5")):
                    yv = np.linspace(y0, cone[key], len(future) + 1)[1:]
                    fig.add_trace(go.Scatter(x=future, y=yv, name=nm,
                                             line=dict(color=clr, dash="dot" if nm != "central" else "solid")))
                fig.update_layout(height=320, margin=dict(l=0, r=0, t=6, b=0),
                                  showlegend=True, legend=dict(orientation="h"))
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(
                    f"**Central:** ${cone['central']:,.2f} "
                    f"({(cone['central']/cone['last_close']-1)*100:+.1f}%)  |  "
                    f"**P5–P95:** ${cone['p5']:,.2f} – ${cone['p95']:,.2f}")
                if art:
                    st.caption(f"OOS R²={art['metrics_test']['R2']:+.3f}, "
                               f"5–95 band coverage {art['metrics_test']['band_cov_pct']:.0f}%.")
            else:
                st.info(f"{label} cone unavailable.")


# ═════════════════════════════ STRATEGY ══════════════════════════════════
with tab_strat:
    st.subheader("Trading strategy — capture the trend, cut the drawdown")
    a0, a1 = st.columns([1.4, 1])
    strat_kind = a0.radio(
        "Strategy", ["MA trend filter", "Divergence Pure-Regime"], horizontal=True,
        help="MA filter: long above the N-day SMA (highest 1x return). "
             "Divergence: the BTC-style U1/D2/D3/V signal system, gold-scaled "
             "(best Sharpe & drawdown; beats buy & hold on GDX).")
    asset = a1.selectbox("Asset", ["GLDM", "UGL", "GDX"],
                         format_func=lambda a: {"GLDM": "GLDM · 1x gold",
                                                "UGL": "UGL · 2x gold",
                                                "GDX": "GDX · gold miners"}[a])
    col = "gldm_close" if asset == "GLDM" else f"{asset.lower()}_close"

    if strat_kind == "MA trend filter":
        b1, b2 = st.columns(2)
        ma_win = b1.slider("MA window (days)", 20, 200,
                           gc.MA_WINDOW_BY_ASSET.get(asset, 50), step=5)
        stop = b2.slider("Fixed stop (%)", 1.0, 15.0,
                         gc.STOP_BY_ASSET[asset] * 100, step=0.5) / 100.0
        r = btg.simulate_regime(preds, sig, col, ma_window=ma_win, stop_pct=stop) \
            if col in preds else None
        desc = (f"Long when the GLDM close is above its {ma_win}-day SMA (signal "
                f"derived from gold, applied to {asset}); flat otherwise; stop {stop*100:.1f}%.")
    else:
        b1, b2, b3 = st.columns(3)
        U1 = b1.slider("U1 entry (+%)", 0.02, 0.40, gc.U1_ERRHI_MIN, step=0.01)
        D2 = b2.slider("D2 exit (−%)", -0.40, -0.02, gc.D2_ERRHI_MAX, step=0.01)
        stop = b3.slider("Fixed stop (%)", 1.0, 15.0,
                         gc.STOP_BY_ASSET[asset] * 100, step=0.5) / 100.0
        r = btg.simulate(preds, sig, col, stop, U1, D2, gc.D1_ERRLO_MIN) \
            if col in preds else None
        desc = (f"Pure-Regime entry (U1>{U1:+.2f}% & regime gate), exit on D2<{D2:+.2f}% "
                f"/ D3 exhaustion; stop {stop*100:.1f}%. Signal from gold, applied to {asset}.")

    if r is None:
        st.warning(f"{asset} price series unavailable.")
    else:
        sm = btg._metrics(r["strat"], r["dates"]); bm = btg._metrics(r["bh"], r["dates"])
        trades = r["trades"]; wr = (trades > 0).mean() * 100 if len(trades) else 0
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Strategy return", f"{sm['total_ret']*100:+.1f}%",
                  f"CAGR {sm['cagr']*100:+.1f}%")
        k2.metric("Buy & Hold return", f"{bm['total_ret']*100:+.1f}%",
                  f"CAGR {bm['cagr']*100:+.1f}%")
        k3.metric("Max drawdown", f"{sm['mdd']*100:.1f}%",
                  f"vs B&H {bm['mdd']*100:.1f}%", delta_color="inverse")
        k4.metric("Sharpe", f"{sm['sharpe']:.2f}", f"vs B&H {bm['sharpe']:.2f}")

        fig = go.Figure()
        dts = pd.to_datetime(r["dates"])
        fig.add_trace(go.Scatter(x=dts, y=r["strat"], name="Strategy",
                                 line=dict(color="#b8860b", width=2)))
        fig.add_trace(go.Scatter(x=dts, y=r["bh"], name="Buy & Hold",
                                 line=dict(color="#888", dash="dash")))
        fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0),
                          yaxis_title="Growth of $1", legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Out-of-sample {btg.OOS_START} → {pd.to_datetime(r['dates'][-1]).date()}. "
                   f"{len(trades)} completed trades, {wr:.0f}% win rate. {desc}")


# ═════════════════════════════ SIGNALS ═══════════════════════════════════
with tab_sig:
    st.subheader("Divergence trend-signatures (diagnostic)")
    st.caption(
        "Prediction-vs-actual divergence signals, gold-scaled. These power the "
        "alert card and an alternative BTC-style entry/exit strategy. For gold "
        "the simple MA trend filter (Strategy tab) outperforms trading these "
        "aggressively — they are most useful as a *regime read*.")
    if ts:
        cA, cB, cC = st.columns(3)
        cA.metric("err_hi 3d avg", f"{ts['err_hi_ma3']:+.3f}%",
                  "U1 " + ("ON" if ts['u1_triggered'] else "off"))
        cB.metric("err_lo 3d avg", f"{ts['err_lo_ma3']:+.3f}%",
                  "D1 " + ("ON" if ts['d1_triggered'] else "off"))
        cC.metric("hi/lo breaks (3d)", f"{ts['hi_breaks_3d']} / {ts['lo_breaks_3d']}",
                  "D2 " + ("ON" if ts['d2_triggered'] else "off"))
        rows = pd.DataFrame(ts["detail_rows"])
        if not rows.empty:
            rows["date"] = pd.to_datetime(rows["date"]).dt.date
            show = rows[["date", "close", "pred_hi", "actual_hi", "err_hi_pct",
                         "pred_lo", "actual_lo", "err_lo_pct", "hi_break", "lo_break"]]
            st.dataframe(show.round(2), use_container_width=True, hide_index=True)
        st.markdown(
            f"**Thresholds (gold-scaled):** U1 > {gc.U1_ERRHI_MIN:+.2f}%, "
            f"D2 < {gc.D2_ERRHI_MAX:+.2f}%, D1 > {gc.D1_ERRLO_MIN:+.2f}%, "
            f"V-reversal low undershoot > {gc.V_ERRLO_MIN:.2f}%. "
            "BTC uses ±1.3% — gold's daily range is ~1/3 of BTC's.")
    else:
        st.info("Not enough completed bars.")


# ═════════════════════════════ EXPLAIN ═══════════════════════════════════
with tab_explain:
    st.subheader("How the GLDM app works")
    st.markdown(f"""
**Asset.** GLDM = SPDR Gold MiniShares, tracking spot gold (~1/100 oz, 0.10%
expense ratio). Gold's realised volatility is roughly a quarter of Bitcoin's and
it trades the US equity calendar, so every feature and threshold is re-derived
for gold rather than copied from BTC.

**Gold-specific features** (swapped in for BTC's crypto inputs):
- **US Dollar Index (DXY)** — strong *inverse* driver of gold
- **10-year Treasury yield (^TNX)** — real-yield proxy, *inverse* driver
- **Silver (SLV)** — precious-metals complex co-movement + gold/silver ratio
- **Gold futures (GC=F)** — basis / lead vs the ETF
- **VIX & S&P 500** — risk-off gold bid / stock-vs-gold rotation
- **Gold macro sentiment (0–100)** — a purpose-built composite (dollar weakness
  + falling real yields + risk-off + gold momentum), replacing the crypto
  Fear & Greed index

**Five models** (all trained by `src/gldm/train_gldm.py`):
1. **Hourly next-close** — ridge on log-returns + 95% CI
2. **Daily High/Low** — ridge ratio-to-prior-close bands (calibrated)
3. **7-day close cone** — central path + empirical quantile bands
4. **14-day close cone**
5. **3-class day type** — Trend Up / Chop / Trend Down (logistic)

**Strategy.** The backtest shows gold's persistent uptrends and shallow dips
reward a **simple price-vs-MA trend filter** over BTC's aggressive
divergence-exit complex: it keeps you long while the trend holds and steps aside
in confirmed downtrends. Applied to GLDM (1x), UGL (2x) and GDX (miners) it
captures most of buy-&-hold's return while roughly **halving max drawdown** and
raising Sharpe. The BTC-style divergence signals are retained as a diagnostic
(Signals tab) and an alternative strategy.
""")
    st.markdown("#### Model freshness (`train_end`) — GLDM")
    for label, art in (("hourly close", M_HOURLY), ("daily H/L", M_HL),
                       ("7-day cone", M_7D), ("14-day cone", M_14D),
                       ("3-class day type", M_DT)):
        st.caption(f"&bull; {label}: `{_fresh(art)}`", unsafe_allow_html=True)
    st.caption("Retrain with `python src/gldm/train_gldm.py`, then Refresh now.")
