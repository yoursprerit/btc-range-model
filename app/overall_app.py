"""Overall Trading — the combined cross-asset decision cockpit.

One screen that fuses the live signals, positions and back-tests of every other
app (BTC, Gold, SOXX, VEGN, GRID, XLE, REMX, WGMI) into a single view built for
one question: *where do I put money to work today?*

  🔴 Live — Decision Cockpit   what to CLOSE / OPEN / HOLD today, the optimal %
                                allocation, and a summary of the strategy's
                                current book across all assets.
  📊 Combined Backtesting       the historically-optimal blend of all the assets'
                                signal-driven strategies vs the obvious
                                benchmarks, with per-window and per-asset detail.
  🧠 Strategy & Methodology     how every number here is produced.

All maths lives in ``overall_core``; this module is the (thin, cached) Streamlit
layer.  The BTC and Gold apps are never imported — this app re-runs every asset
through one unified daily engine so the eight strategies are directly comparable
and can be blended into a single portfolio.
"""
from __future__ import annotations

import sys
import importlib
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

_APP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _APP_DIR.parent
for _p in (str(_APP_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import sklearn._loss._loss as _sk_loss_ext
    if "_loss" not in sys.modules:
        sys.modules["_loss"] = _sk_loss_ext
except Exception:
    pass

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import overall_core as ov
import ticker_config
if not hasattr(ov, "run_universe"):
    ov = importlib.reload(ov)


# ── app registry (shared with the router / other apps) ────────────────────
_ALL_APPS = ["OVERALL", "BTC", "GLDM"] + ticker_config.APP_KEYS
_APP_LABELS = {
    "OVERALL": "🧭  Overall Trading",
    "BTC": "₿  Bitcoin (BTC)",
    "GLDM": "🥇  Gold (GLDM)",
}
for _k, _c in ticker_config.CONFIGS.items():
    _APP_LABELS[_k] = f"{_c.emoji}  {_c.key} · {_c.name.split('(')[0].strip()[:22]}"

st.set_page_config(page_title="Overall Trading", page_icon="🧭",
                   layout="wide", initial_sidebar_state="expanded")

# colour tokens
C_BUY = "#16a34a"; C_HOLD = "#0ea5e9"; C_EXIT = "#dc2626"
C_WATCH = "#d97706"; C_FLAT = "#64748b"; C_CASH = "#94a3b8"
_TONE_COL = {"buy": C_BUY, "hold": C_HOLD, "exit": C_EXIT,
             "watch": C_WATCH, "flat": C_FLAT}
_ACTION_COL = {"CLOSE": C_EXIT, "OPEN": C_BUY, "HOLD": C_HOLD,
               "WATCH": C_WATCH, "STAND ASIDE": C_FLAT}


# ══════════════════════════════════════════════════════════════════════════
# Sidebar — unified application selector
# ══════════════════════════════════════════════════════════════════════════
if "gldm_active_app" not in st.session_state:
    st.session_state["gldm_active_app"] = "OVERALL"
with st.sidebar:
    st.radio("**Application**", options=_ALL_APPS,
             format_func=lambda x: _APP_LABELS.get(x, x), key="gldm_active_app")
    st.markdown("---")
    st.markdown("**Auto-refresh:** live data cached ~15 min.")
    if st.button("Refresh now", use_container_width=True):
        st.cache_data.clear(); st.cache_resource.clear(); st.rerun()
    st.caption("_Overall Trading fuses all eight asset apps into one portfolio "
               "view. Each asset is re-run through a single unified daily "
               "trend/divergence engine so the strategies are directly "
               "comparable and blendable._")


# ══════════════════════════════════════════════════════════════════════════
# Cached compute — run all 8 assets, then the portfolio maths
# ══════════════════════════════════════════════════════════════════════════
def _bucket() -> str:
    # cache key that rolls over ~ every 15 min
    now = pd.Timestamp.utcnow()
    return f"{now.date()}-{now.hour}-{now.minute // 15}"


@st.cache_data(ttl=900, show_spinner="Running all 8 strategies live (first load ~30–90s)…")
def get_results(bucket: str):
    return ov.run_universe()


@st.cache_data(ttl=900, show_spinner="Optimising the combined allocation…")
def get_portfolio(bucket: str):
    results = get_results(bucket)
    if not results:
        return None
    rets = ov.returns_matrix(results)
    opt = ov.optimize_weights(rets)
    bm = ov.benchmarks(rets, results)
    w_opt = np.array([opt["optimal"]["weights"][c] for c in opt["cols"]])
    per = ov.period_breakdown(rets, w_opt, ov.COMBINED_PERIODS)
    curves = {
        "Optimal blend": ov._equity(ov._combine(rets, w_opt)),
        "Equal-weight strategies": bm["strat_equal"]["equity"],
        "8× Buy & Hold (equal)": bm["bh_equal"]["equity"],
    }
    gate = ov.signal_gated_allocation(results, opt["optimal"]["weights"])
    return dict(results=results, rets=rets, opt=opt, bm=bm, per=per,
                curves=curves, gate=gate, w_opt=w_opt)


st.title("🧭 Overall Trading — Combined Decision Cockpit")
st.caption("Every asset app, fused into one portfolio. Live entry/exit signals, "
           "current positions, the historically-optimal cross-asset allocation, "
           "and one combined back-test — engineered around a single question: "
           "**where should capital go today?**")

_PF = get_portfolio(_bucket())
if not _PF:
    st.error("Could not fetch market data for any asset right now. "
             "Hit **Refresh now** in a moment.")
    st.stop()

results = _PF["results"]
by_key = {r["key"]: r for r in results}
opt = _PF["opt"]; gate = _PF["gate"]; bm = _PF["bm"]
as_of = max(r["as_of"] for r in results)


# ── small HTML helpers ─────────────────────────────────────────────────────
def _pill(text, color, bg=None):
    bg = bg or (color + "22")
    return (f"<span style='background:{bg};color:{color};font-weight:700;"
            f"padding:2px 9px;border-radius:999px;font-size:12px;white-space:nowrap'>"
            f"{text}</span>")


def _pct(x, digits=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.{digits}f}%"


tab_live, tab_bt, tab_explain = st.tabs(
    ["🔴 Live — Decision Cockpit", "📊 Combined Backtesting",
     "🧠 Strategy & Methodology"])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE DECISION COCKPIT
# ══════════════════════════════════════════════════════════════════════════
with tab_live:
    st.markdown(f"#### As of **{as_of.strftime('%b %d, %Y')}** · {len(results)} assets tracked")

    invested = 1.0 - gate["cash"]
    k = st.columns(5)
    k[0].metric("Positions open", f"{gate['n_active']}",
                help="Assets the strategy is currently long.")
    k[1].metric("Open today", f"{gate['n_open']}",
                delta="new entries" if gate["n_open"] else None)
    k[2].metric("Close today", f"{gate['n_close']}",
                delta="exit signal" if gate["n_close"] else None,
                delta_color="inverse")
    k[3].metric("Target invested", f"{invested*100:.0f}%")
    k[4].metric("Target cash", f"{gate['cash']*100:.0f}%")

    # ── 1. TODAY'S ACTION PLAN ──────────────────────────────────────────
    st.markdown("### 🎯 Today's action plan")
    st.caption("What to do now, ranked. **Close** exits first, then **open** new "
               "positions, then holds. Target % is the recommended book after "
               "today's moves.")
    hdr = ("<tr style='background:#f1f5f9;font-size:12px;text-align:left'>"
           "<th style='padding:7px 10px'>Action</th><th>Asset</th>"
           "<th>Live signal</th><th style='text-align:right'>Price</th>"
           "<th style='text-align:right'>Unreal. P&amp;L</th>"
           "<th style='text-align:right'>Target %</th></tr>")
    rows = []
    for a in gate["actions"]:
        ac = _ACTION_COL[a["action"]]
        tgt = a["target"]
        tgt_s = f"{tgt*100:.1f}%" if tgt > 0.0005 else "—"
        pnl = _pct(a["upnl"]) if a["in_pos"] else "—"
        pnl_col = (C_BUY if (a["upnl"] or 0) >= 0 else C_EXIT) if a["in_pos"] else "#94a3b8"
        bar = (f"<div style='height:7px;background:#e2e8f0;border-radius:4px;overflow:hidden;"
               f"margin-top:3px'><div style='height:7px;width:{min(tgt*100,100):.0f}%;"
               f"background:{a['emoji'] and by_key[a['key']]['accent']}'></div></div>"
               if tgt > 0.0005 else "")
        _r = by_key[a["key"]]
        if _r["mode"] == "ma" and _r.get("ma_val"):
            _dist = (_r["last_close"] / _r["ma_val"] - 1) * 100
            sub = f"close {_dist:+.1f}% vs MA{_r['ma_window']}"
        else:
            sub = f"alert: {a['alert']}"
        rows.append(
            f"<tr style='border-bottom:1px solid #eef2f7'>"
            f"<td style='padding:8px 10px'>{_pill(a['action'], ac)}</td>"
            f"<td style='font-weight:700'>{a['emoji']} {a['key']}"
            f"<div style='font-size:11px;color:#64748b;font-weight:400'>{a['name']}</div></td>"
            f"<td style='font-size:12px;color:{_TONE_COL.get(a['tone'], C_FLAT)}'>{a['decision']}"
            f"<div style='font-size:10px;color:#94a3b8'>{sub}</div></td>"
            f"<td style='text-align:right;font-variant-numeric:tabular-nums'>${a['last_close']:,.2f}</td>"
            f"<td style='text-align:right;color:{pnl_col};font-weight:600'>{pnl}</td>"
            f"<td style='text-align:right;font-weight:700'>{tgt_s}{bar}</td></tr>")
    st.markdown(f"<table style='width:100%;border-collapse:collapse'>{hdr}{''.join(rows)}</table>",
                unsafe_allow_html=True)

    # explicit call-outs
    closes = [a for a in gate["actions"] if a["action"] == "CLOSE"]
    opens = [a for a in gate["actions"] if a["action"] == "OPEN"]
    cc = st.columns(2)
    with cc[0]:
        if closes:
            st.error("**Close now:** " + ", ".join(
                f"{a['emoji']} {a['key']} ({_pct(a['upnl'])})" for a in closes))
        else:
            st.success("**No exits triggered today** — nothing to close.")
    with cc[1]:
        if opens:
            st.success("**Open now:** " + ", ".join(
                f"{a['emoji']} {a['key']} → {a['target']*100:.0f}%" for a in opens))
        else:
            st.info("**No fresh entries today** — no flat asset is signalling a buy.")

    st.markdown("---")

    # ── 2. OPTIMAL ALLOCATION TODAY vs CURRENT BOOK ─────────────────────
    st.markdown("### 📐 Optimal allocation for today")
    st.caption("Left: what the strategy holds **right now**. Right: the "
               "**recommended** book after today's signals, sized by the "
               "historically-optimal weights (capped 35%/asset, water-filled).")
    ac = st.columns([1, 1, 1])

    def _alloc_donut(alloc: dict, cash: float, title: str):
        labels, vals, colors = [], [], []
        for kk, vv in sorted(alloc.items(), key=lambda x: -x[1]):
            if vv <= 0.0005:
                continue
            labels.append(f"{by_key[kk]['emoji']} {kk}"); vals.append(vv * 100)
            colors.append(by_key[kk]["accent"])
        if cash > 0.005:
            labels.append("💵 Cash"); vals.append(cash * 100); colors.append(C_CASH)
        fig = go.Figure(go.Pie(labels=labels, values=vals, hole=0.58,
                               marker=dict(colors=colors), sort=False,
                               textinfo="label+percent", textfont_size=12,
                               hovertemplate="%{label}: %{value:.1f}%<extra></extra>"))
        fig.update_layout(title=dict(text=title, font_size=14), showlegend=False,
                          height=310, margin=dict(t=40, b=10, l=10, r=10))
        return fig

    ac[0].plotly_chart(_alloc_donut(gate["current"], gate["cash_now"],
                                    "Current book"), use_container_width=True)
    ac[1].plotly_chart(_alloc_donut(gate["target"], gate["cash"],
                                    "Recommended today"), use_container_width=True)
    with ac[2]:
        st.markdown("**Rebalancing moves**")
        moves = []
        allkeys = set(gate["current"]) | set(gate["target"])
        for kk in sorted(allkeys, key=lambda x: -(gate["target"].get(x, 0))):
            cur = gate["current"].get(kk, 0.0); tg = gate["target"].get(kk, 0.0)
            d = tg - cur
            if abs(d) < 0.005:
                continue
            arrow = "▲" if d > 0 else "▼"
            col = C_BUY if d > 0 else C_EXIT
            moves.append(f"<div style='font-size:13px;margin:3px 0'>{by_key[kk]['emoji']} "
                         f"<b>{kk}</b> {cur*100:.0f}% → {tg*100:.0f}% "
                         f"<span style='color:{col};font-weight:700'>{arrow}{abs(d)*100:.0f}pt</span></div>")
        dcash = gate["cash"] - gate["cash_now"]
        if abs(dcash) >= 0.005:
            col = C_EXIT if dcash > 0 else C_BUY
            moves.append(f"<div style='font-size:13px;margin:3px 0'>💵 <b>Cash</b> "
                         f"{gate['cash_now']*100:.0f}% → {gate['cash']*100:.0f}% "
                         f"<span style='color:{col};font-weight:700'>"
                         f"{'▲' if dcash>0 else '▼'}{abs(dcash)*100:.0f}pt</span></div>")
        st.markdown("".join(moves) if moves
                    else "_Book already at target — no rebalancing needed._",
                    unsafe_allow_html=True)

    st.markdown("---")

    # ── 3. PER-ASSET LIVE CARDS ─────────────────────────────────────────
    st.markdown("### 🛰️ Live signal & position — every asset")
    st.caption("The current read from each app's strategy. Green = long / buy, "
               "red = exit, amber = watch, grey = stand aside.")
    cards = st.columns(4)
    for i, res in enumerate(results):
        dec = res["decision"]; tone = dec["tone"]; col = _TONE_COL.get(tone, C_FLAT)
        pos = res["pos"]
        with cards[i % 4]:
            body = [
                f"<div style='border:1px solid #e2e8f0;border-left:5px solid {res['accent']};"
                f"border-radius:10px;padding:11px 13px;margin-bottom:12px;background:#fff'>",
                f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                f"<span style='font-weight:800;font-size:15px'>{res['emoji']} {res['key']}</span>"
                f"<span style='font-size:12px;color:#64748b'>${res['last_close']:,.2f} "
                f"<b style='color:{C_BUY if res['dchg']>=0 else C_EXIT}'>{res['dchg']:+.1f}%</b></span></div>",
                f"<div style='font-size:11px;color:#94a3b8;margin-bottom:6px'>{res['name']}</div>",
                f"<div style='margin:5px 0'>{_pill(dec['ico'] + ' ' + dec['label'], col)}</div>",
            ]
            if pos["in_pos"] and pos["entry_px"]:
                pcol = C_BUY if (pos["upnl"] or 0) >= 0 else C_EXIT
                body.append(
                    f"<div style='font-size:12px;line-height:1.55;margin-top:4px'>"
                    f"📍 <b>LONG</b> since {pd.Timestamp(pos['entry_date']).strftime('%b %d')} "
                    f"@ ${float(pos['entry_px']):,.2f}<br>"
                    f"P&amp;L <b style='color:{pcol}'>{_pct(pos['upnl'])}</b> · "
                    f"{pos['days']}d held<br>"
                    f"Stop ${pos['stop_px']:,.2f} "
                    f"<span style='color:#94a3b8'>({_pct(pos.get('dist_stop'))} away)</span></div>")
            elif res["last_trade"]:
                lt = res["last_trade"]; r_ = lt["ret"] * 100
                rc = C_BUY if r_ >= 0 else C_EXIT
                body.append(
                    f"<div style='font-size:12px;line-height:1.5;margin-top:4px;color:#475569'>"
                    f"⚪ FLAT · last trade "
                    f"<b style='color:{rc}'>{r_:+.1f}%</b> "
                    f"({pd.Timestamp(lt['exit_date']).strftime('%b %d')})<br>"
                    f"<span style='color:#94a3b8'>exit: {lt.get('reason','—')}</span></div>")
            else:
                body.append("<div style='font-size:12px;color:#475569;margin-top:4px'>⚪ FLAT · no open position</div>")
            sent = res["sentiment"]
            sent_s = f"{sent:.0f}/100" if sent == sent else "—"  # nan check
            body.append(
                f"<div style='font-size:11px;color:#94a3b8;margin-top:6px;border-top:1px dashed #e2e8f0;padding-top:5px'>"
                f"{'🐂 bull' if res['bull_regime'] else '🐻 bear'} regime · sentiment {sent_s} · "
                f"{res['mode'].upper()}{('/MA'+str(res['ma_window'])) if res['mode']=='ma' else ''}</div>")
            body.append("</div>")
            st.markdown("".join(body), unsafe_allow_html=True)

    st.info("These are unified **daily** reads. For the canonical hourly "
            "Pure-Regime view of BTC (BTC/MSTR/MSTU) or Gold (GDX/UGL), open the "
            "**₿ Bitcoin** or **🥇 Gold** app in the sidebar.")


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — COMBINED BACKTESTING
# ══════════════════════════════════════════════════════════════════════════
with tab_bt:
    o = opt["optimal"]
    st.markdown("## 📊 The optimal combined strategy")
    st.caption("Each asset's signal-driven strategy produces a daily return "
               "stream (long when its signal is on, cash otherwise). We search "
               "long-only blends of all eight — capped 35%/asset — for the mix "
               "that **maximises risk-adjusted return (Sharpe) while holding the "
               "drawdown shallow**. Out-of-sample from 2021; $100k start.")

    # KPI row: optimal vs benchmarks
    m = st.columns(4)
    m[0].metric("Optimal blend — total return", f"{o['total_ret']*100:,.0f}%",
                delta=f"CAGR {o['cagr']*100:.0f}%")
    m[1].metric("Max drawdown", f"{o['mdd']*100:.0f}%",
                delta=f"vs {bm['bh_equal']['mdd']*100:.0f}% buy&hold",
                delta_color="inverse")
    m[2].metric("Sharpe", f"{o['sharpe']:.2f}",
                delta=f"vs {bm['bh_equal']['sharpe']:.2f} buy&hold")
    m[3].metric("Volatility (ann.)", f"{o['vol']*100:.0f}%")

    st.markdown("#### Optimal weights")
    wc = st.columns([2, 1])
    with wc[0]:
        wk = [(c, o["weights"][c]) for c in opt["cols"]]
        wk.sort(key=lambda x: -x[1])
        fig = go.Figure(go.Bar(
            x=[w * 100 for _, w in wk], y=[f"{by_key[c]['emoji']} {c}" for c, _ in wk],
            orientation="h", marker=dict(color=[by_key[c]["accent"] for c, _ in wk]),
            text=[f"{w*100:.1f}%" for _, w in wk], textposition="outside"))
        fig.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=30),
                          xaxis_title="allocation %", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
    with wc[1]:
        st.markdown("**Allocation scheme comparison**")
        cmp_rows = []
        for nm, key in [("Optimal (max-Sharpe)", "optimal"),
                        ("Risk parity", "risk_parity"),
                        ("Equal weight", "equal")]:
            d = opt[key]
            cmp_rows.append(f"<tr><td style='padding:4px 6px'>{nm}</td>"
                            f"<td style='text-align:right'>{d['total_ret']*100:,.0f}%</td>"
                            f"<td style='text-align:right'>{d['mdd']*100:.0f}%</td>"
                            f"<td style='text-align:right'>{d['sharpe']:.2f}</td></tr>")
        st.markdown(
            "<table style='width:100%;font-size:13px;border-collapse:collapse'>"
            "<tr style='background:#f1f5f9'><th style='text-align:left;padding:4px 6px'>Scheme</th>"
            "<th style='text-align:right'>Return</th><th style='text-align:right'>MDD</th>"
            "<th style='text-align:right'>Sharpe</th></tr>" + "".join(cmp_rows) + "</table>",
            unsafe_allow_html=True)

    # equity curves
    st.markdown("#### Growth of $100k — combined vs benchmarks")
    fig = go.Figure()
    styles = {"Optimal blend": ("#111827", 3),
              "Equal-weight strategies": ("#0ea5e9", 1.7),
              "8× Buy & Hold (equal)": ("#94a3b8", 1.7)}
    for name, curve in _PF["curves"].items():
        cc2, wdt = styles.get(name, ("#888", 1.5))
        fig.add_trace(go.Scatter(x=curve.index, y=curve.to_numpy() * 100000,
                                 name=name, line=dict(color=cc2, width=wdt)))
    fig.update_layout(height=440, margin=dict(t=20, b=10, l=10, r=10),
                      yaxis_title="Portfolio value ($)", yaxis_type="log",
                      legend=dict(orientation="h", y=1.08),
                      hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # drawdown of the optimal blend
    opt_curve = _PF["curves"]["Optimal blend"]
    dd = opt_curve / opt_curve.cummax() - 1
    figd = go.Figure(go.Scatter(x=dd.index, y=dd.to_numpy() * 100, fill="tozeroy",
                                line=dict(color=C_EXIT, width=1)))
    figd.update_layout(height=230, margin=dict(t=10, b=10, l=10, r=10),
                       yaxis_title="drawdown %", title=dict(
                           text="Optimal blend — drawdown", font_size=13))
    st.plotly_chart(figd, use_container_width=True)

    # period breakdown
    st.markdown("#### Performance by market regime")
    per = _PF["per"]
    ph = ("<tr style='background:#f1f5f9'><th style='text-align:left;padding:6px 10px'>Window</th>"
          "<th style='text-align:right'>Return</th><th style='text-align:right'>CAGR</th>"
          "<th style='text-align:right'>Max DD</th><th style='text-align:right'>Sharpe</th></tr>")
    pr = []
    for row in per:
        pr.append(f"<tr style='border-bottom:1px solid #eef2f7'>"
                  f"<td style='padding:6px 10px'>{row['label']}</td>"
                  f"<td style='text-align:right;font-weight:600'>{row['total_ret']*100:,.0f}%</td>"
                  f"<td style='text-align:right'>{row['cagr']*100:.0f}%</td>"
                  f"<td style='text-align:right;color:{C_EXIT}'>{row['mdd']*100:.0f}%</td>"
                  f"<td style='text-align:right;font-weight:600'>{row['sharpe']:.2f}</td></tr>")
    st.markdown(f"<table style='width:100%;border-collapse:collapse'>{ph}{''.join(pr)}</table>",
                unsafe_allow_html=True)

    # per-asset standalone contribution
    st.markdown("#### Per-asset strategy (standalone, out-of-sample)")
    st.caption("Each asset's own signal-driven strategy vs buy-&-hold, and its "
               "weight in the optimal blend.")
    ah = ("<tr style='background:#f1f5f9'><th style='text-align:left;padding:6px 10px'>Asset</th>"
          "<th>Engine</th><th style='text-align:right'>Strat return</th>"
          "<th style='text-align:right'>Buy&amp;Hold</th><th style='text-align:right'>Max DD</th>"
          "<th style='text-align:right'>Sharpe</th><th style='text-align:right'>Win%</th>"
          "<th style='text-align:right'>Opt. weight</th></tr>")
    ar = []
    for res in sorted(results, key=lambda r: -opt["optimal"]["weights"].get(r["key"], 0)):
        mm = res["metrics"]; bb = res["bh_metrics"]
        beat = mm["total_ret"] >= bb["total_ret"]
        eng = f"MA{res['ma_window']}/−{res['stop']*100:.0f}%" if res["mode"] == "ma" else "Divergence"
        ar.append(f"<tr style='border-bottom:1px solid #eef2f7'>"
                  f"<td style='padding:6px 10px;font-weight:700'>{res['emoji']} {res['key']}"
                  f"<span style='font-size:11px;color:#94a3b8;font-weight:400'> {res['name']}</span></td>"
                  f"<td style='font-size:12px;color:#64748b'>{eng}</td>"
                  f"<td style='text-align:right;font-weight:600;color:{C_BUY if beat else '#111'}'>"
                  f"{mm['total_ret']*100:,.0f}%</td>"
                  f"<td style='text-align:right;color:#64748b'>{bb['total_ret']*100:,.0f}%</td>"
                  f"<td style='text-align:right;color:{C_EXIT}'>{mm['mdd']*100:.0f}%</td>"
                  f"<td style='text-align:right'>{mm['sharpe']:.2f}</td>"
                  f"<td style='text-align:right'>{res['win_rate']:.0f}%</td>"
                  f"<td style='text-align:right;font-weight:700'>"
                  f"{opt['optimal']['weights'].get(res['key'],0)*100:.1f}%</td></tr>")
    st.markdown(f"<table style='width:100%;border-collapse:collapse'>{ah}{''.join(ar)}</table>",
                unsafe_allow_html=True)

    st.success(
        f"**Bottom line.** The optimal blend returned "
        f"**{o['total_ret']*100:,.0f}%** at just **{o['mdd']*100:.0f}%** max "
        f"drawdown (Sharpe **{o['sharpe']:.2f}**) — versus an equal-weight "
        f"buy-&-hold of the same eight assets at "
        f"**{bm['bh_equal']['total_ret']*100:,.0f}%** but a punishing "
        f"**{bm['bh_equal']['mdd']*100:.0f}%** drawdown (Sharpe "
        f"**{bm['bh_equal']['sharpe']:.2f}**). Blending signal-driven, cash-when-"
        f"out strategies across uncorrelated assets is what buys the risk "
        f"reduction.")
    st.caption("⚠️ Optimal weights are fit on this same history (in-sample); "
               "treat them as the best *historical* blend, not a guarantee. "
               "Equal-weight and risk-parity need no fitting and are shown for "
               "comparison.")


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — METHODOLOGY
# ══════════════════════════════════════════════════════════════════════════
with tab_explain:
    st.markdown("## 🧠 How Overall Trading works")
    st.markdown(
        """
**The idea.** Every other app trades one asset off its own signal. *Overall
Trading* runs all eight through **one unified daily engine**, so their live
signals, positions and back-tests sit side-by-side and can be blended into a
single portfolio.

**The universe (one instrument per app).**

| App | Traded here | Engine | Why |
|---|---|---|---|
| ₿ Bitcoin | BTC | MA50 trend filter, −3% stop | Model-free daily trend regime; robustly beats hold on risk |
| 🥇 Gold | GLDM | MA50 trend filter, −3% stop | The Gold app's own *documented primary* strategy |
| 🖥️ SOXX | SOXX | MA40, −5% | Its tuned config |
| 🌱 VEGN | VEGN | MA200, −5% | Its tuned config |
| ⚡ GRID | GRID | MA150, −5% | Its tuned config |
| 🛢️ XLE | XLE | Divergence Pure-Regime, −8% | Its tuned config |
| 🧲 REMX | REMX | **Divergence Pure-Regime, −8%** | Divergence wins REMX's risk-adjusted OOS (+109%/−18%/0.93 vs MA's +73%/−56%/0.48) — see note below |
| ⛏️ WGMI | WGMI | MA30, −10% | Its tuned config |

The six ETF apps reuse their **exact** `ticker_config` entries, so their numbers
match those apps bar-for-bar. BTC and Gold use faithful trend-filter configs
(Gold's is its documented MA50 primary; BTC's daily divergence would need the
dedicated app's hourly CT model, so a robust daily trend filter is used instead).

**Live signals & positions.** For each asset we fetch data, fit the H/L band
model out-of-sample, replay the strategy bar-by-bar, and read off: the current
alert level (the same `STRATEGY_BUY / WATCH_UP / … / HIGH_DN` vocabulary every
app uses), whether we're long, the entry price/date, unrealised P&L, stop level
and days held. That drives the **action plan** (close / open / hold) and the
**allocation donuts**.

**The optimal allocation.** Each strategy is long when its signal is on and in
cash otherwise, producing a daily return stream. We Monte-Carlo long-only blends
(sum = 100%, ≤ 35% per asset) and pick the one that **maximises Sharpe while
keeping the drawdown shallow**. Equal-weight and inverse-vol (risk-parity)
blends are shown alongside as un-fitted references.

**Today's book.** Capital is deployed only to assets currently signalling long
(or firing a fresh entry), split by the optimal weights, water-filled to the
35% cap; whatever can't be deployed sits in cash. That's the *Recommended today*
donut and the rebalancing moves.

**Honest caveats.**
- Optimal weights are in-sample — the best *historical* blend, not a promise.
- This is a **daily** engine. The BTC and Gold apps' canonical **hourly**
  Pure-Regime signals (and their MSTR/MSTU, GDX/UGL siblings) live in those
  apps — open them from the sidebar for execution-grade reads.
- Nothing here is investment advice.
        """)


st.caption(f"Overall Trading · unified daily engine · data to "
           f"{as_of.strftime('%Y-%m-%d')} · not investment advice.")
