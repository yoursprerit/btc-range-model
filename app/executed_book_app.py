"""Executed Book (IBKR) — post-rebalance report of trades + current positions.

Selected from the sidebar **Application** radio ("✅ Executed Book (IBKR)"), right
after Target Book. It reads the execution report the laptop/VM executor commits
back after a rebalance (``data/overall/executed_book.json``) and shows, in a
human-readable way:

  * a run header (when it ran, account, mode, net-liq, cash) with signature +
    freshness badges,
  * the **trades executed** (buys/sells, quantity, fill price, status),
  * the **current IBKR positions** (shares, cost, market value, unrealised P&L,
    weight) with a donut.

The cloud app never connects to IBKR — this is the mirror of the Target Book
flow: the executor publishes the report, this page displays it.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

_APP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _APP_DIR.parent
for _p in (str(_APP_DIR), str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import overall_core as ov                 # noqa: E402
import ticker_config                       # noqa: E402
import target_book as tb                   # noqa: E402  (shared HMAC verify)
import executed_book as eb                 # noqa: E402
import ibkr_symbols as sym                 # noqa: E402  (BTC→IBIT mapping)

REPORT_PATH = _REPO_ROOT / "data" / "overall" / "executed_book.json"
REPORT_PATH_LIVE = _REPO_ROOT / "data" / "overall" / "executed_book_live.json"
TARGET_PATH = _REPO_ROOT / "data" / "overall" / "target_book.json"

try:
    st.set_page_config(page_title="Executed Book (IBKR)", page_icon="✅",
                       layout="wide", initial_sidebar_state="expanded")
except Exception:
    pass

# ── sidebar Application selector (same widget/key as every other app) ─────────
_ALL_APPS = (["OVERALL", "BTC", "GLDM", "GDXM"] + ticker_config.APP_KEYS
             + ["DAILYAUDIT", "HEALTH", "TARGETBOOK", "EXECUTEDBOOK"])
_APP_LABELS = {"OVERALL": "🧭  Overall Trading", "BTC": "₿  Bitcoin (BTC)",
               "GLDM": "🥇  Gold Trend (GLDM·UGL)", "GDXM": "⛏️  Gold Miners (GDX·NUGT)", "DAILYAUDIT": "🕵️  Daily Audit",
               "HEALTH": "🩺  Strategy Health",
               "TARGETBOOK": "📋  Target Book (IBKR)",
               "EXECUTEDBOOK": "✅  Executed Book (IBKR)"}
for _k, _c in ticker_config.CONFIGS.items():
    _APP_LABELS[_k] = f"{_c.emoji}  {_c.key} · {_c.name.split('(')[0].strip()[:22]}"
if st.session_state.get("gldm_active_app") not in _ALL_APPS:
    st.session_state["gldm_active_app"] = "EXECUTEDBOOK"
with st.sidebar:
    st.radio("**Application**", options=_ALL_APPS,
             format_func=lambda x: _APP_LABELS.get(x, x), key="gldm_active_app")
    st.markdown("---")
    st.caption("_The executed book is written back by the IBKR executor after a "
               "rebalance. See IBKR_PAPER_TRADING.md for the full flow._")

C_BUY = "#16a34a"; C_SELL = "#dc2626"; C_CASH = "#94a3b8"


def _secret() -> str | None:
    try:
        s = st.secrets.get("OVERALL_BOOK_SECRET")
        if s:
            return str(s)
    except Exception:
        pass
    return os.environ.get("OVERALL_BOOK_SECRET")


def _badge(text: str, color: str) -> str:
    return (f"<span style='background:{color}22;color:{color};font-weight:700;"
            f"padding:3px 11px;border-radius:999px;font-size:13px'>{text}</span>")


def _name(key: str, symbol: str) -> str:
    meta = ov.ASSET_META.get(key or "", {})
    kind = meta.get("kind", "core")
    return f"{ov.KIND_EMOJI.get(kind, '')} {meta.get('name', symbol)}".strip()


def _render(payload: dict, *, source: str) -> None:
    secret = _secret()
    today = pd.Timestamp.now(tz="America/New_York").tz_localize(None)
    ok_sig, sig_why = tb.verify_signature(payload, secret)
    ok_val, val_why = eb.validate(payload, today)

    mode = (payload.get("mode") or "").lower()
    net_liq = float(payload.get("net_liq") or 0.0)
    cash = float(payload.get("cash") or 0.0)
    trades = payload.get("trades") or []
    positions = payload.get("positions") or []

    st.caption(f"Source: {source}")
    c = st.columns(4)
    c[0].metric("Account", payload.get("account", "—"))
    c[1].metric("Net liquidation", f"${net_liq:,.0f}" if net_liq else "—")
    c[2].metric("Cash", f"${cash:,.0f}" if cash else "—")
    c[3].metric("Positions", f"{len(positions)}")

    acct_mode = (payload.get("account_mode") or "paper").lower()
    live = acct_mode == "live"
    acct_txt = "🔴 LIVE — real money" if live else "🧪 PAPER"
    acct_col = "#dc2626" if live else "#0ea5e9"
    mode_txt = ("🟢 EXECUTED" if mode == "execute"
                else "🟡 DRY-RUN (no orders)" if mode == "dry-run" else mode or "—")
    mode_col = "#16a34a" if mode == "execute" else "#d97706"
    if not secret:
        sig_col = "#64748b"; sig_txt = ("🔒 signed (no key to verify)"
                                        if payload.get("signature") else "🔓 unsigned")
    elif ok_sig:
        sig_col = "#16a34a"; sig_txt = "✅ signature verified"
    else:
        sig_col = "#dc2626"; sig_txt = "⛔ signature MISMATCH"
    st.markdown(
        _badge(acct_txt, acct_col) + "  " +
        _badge(mode_txt, mode_col) + "  " + _badge(sig_txt, sig_col) + "  " +
        _badge(("🟢 " if ok_val else "🟡 ") + val_why, "#16a34a" if ok_val else "#d97706") +
        (f"  {_badge('for signal bar ' + str(payload.get('as_of')), '#0ea5e9')}"
         if payload.get("as_of") else ""),
        unsafe_allow_html=True)
    st.markdown("")

    # ── trades executed ───────────────────────────────────────────────────────
    st.markdown("### 🔁 Trades executed")
    if not trades:
        st.info("No trades this run — the book was already within the no-trade band.")
    else:
        trows = []
        for t in sorted(trades, key=lambda x: (x.get("action") != "SELL",
                                               -abs(float(x.get("qty") or 0) *
                                                    float(x.get("price") or 0)))):
            filled = float(t.get("filled") or 0.0)
            avg = float(t.get("avg_fill_price") or 0.0)
            qty = float(t.get("qty") or 0.0)
            px = avg or float(t.get("price") or 0.0)
            trows.append(dict(
                Action=t.get("action"), Instrument=_name(t.get("key"), t.get("symbol")),
                IBKR=t.get("symbol"), Qty=qty,
                Fill=(f"${avg:,.2f}" if avg else (f"~${float(t.get('price') or 0):,.2f}")),
                Value=f"${qty * px:,.0f}",
                Status=t.get("status") or "—",
                Filled=(f"{filled:g}" if filled else ("—" if t.get('status') == 'PLANNED' else "0")),
            ))
        tdf = pd.DataFrame(trows)
        n_buy = sum(1 for t in trades if t.get("action") == "BUY")
        n_sell = sum(1 for t in trades if t.get("action") == "SELL")
        st.caption(f"**{n_sell}** sell(s) then **{n_buy}** buy(s).")
        st.dataframe(
            tdf, hide_index=True, use_container_width=True,
            column_config={
                "Qty": st.column_config.NumberColumn("Qty", format="%.0f"),
                "Action": st.column_config.TextColumn("Action")})

    st.markdown("---")

    # ── current positions ─────────────────────────────────────────────────────
    st.markdown("### 📊 Current IBKR positions")
    if not positions:
        st.info("No open positions reported.")
        _drift_section(payload)
        _download(payload, secret)
        return

    mv_total = sum(abs(float(p.get("market_value") or 0.0)) for p in positions)
    prows = []
    for p in sorted(positions, key=lambda x: -abs(float(x.get("market_value") or 0.0))):
        mv = float(p.get("market_value") or 0.0)
        upnl = float(p.get("unrealized_pnl") or 0.0)
        shares = float(p.get("shares") or 0.0)
        avg_cost = float(p.get("avg_cost") or 0.0)
        mpx = float(p.get("market_price") or 0.0)
        wt = (mv / mv_total * 100) if mv_total else 0.0
        prows.append(dict(
            Instrument=_name(p.get("key"), p.get("symbol")), IBKR=p.get("symbol"),
            Shares=shares, Avg_cost=(f"${avg_cost:,.2f}" if avg_cost else "—"),
            Price=(f"${mpx:,.2f}" if mpx else "—"),
            Value=(f"${mv:,.0f}" if mv else "—"),
            Unreal_PnL=(f"${upnl:,.0f}" if mv else "—"),
            Weight=wt))
    pdf = pd.DataFrame(prows)

    left, right = st.columns([3, 2])
    with left:
        st.dataframe(
            pdf, hide_index=True, use_container_width=True,
            column_config={
                "Shares": st.column_config.NumberColumn("Shares", format="%.0f"),
                "Avg_cost": st.column_config.TextColumn("Avg cost"),
                "Unreal_PnL": st.column_config.TextColumn("Unreal. P&L"),
                "Weight": st.column_config.NumberColumn("Weight %", format="%.1f%%")})
        tot_upnl = sum(float(p.get("unrealized_pnl") or 0.0) for p in positions)
        st.caption(f"Total unrealised P&L: **${tot_upnl:,.0f}** · "
                   f"invested **${mv_total:,.0f}**"
                   + (f" · cash **${cash:,.0f}**" if cash else ""))
    with right:
        if mv_total:
            labels = [p.get("symbol") for p in sorted(
                positions, key=lambda x: -abs(float(x.get("market_value") or 0.0)))]
            vals = [abs(float(p.get("market_value") or 0.0)) for p in sorted(
                positions, key=lambda x: -abs(float(x.get("market_value") or 0.0)))]
            if cash > 0:
                labels += ["CASH"]; vals += [cash]
            fig = go.Figure(go.Pie(labels=labels, values=vals, hole=0.55,
                                   sort=False, textinfo="label+percent"))
            fig.update_layout(height=340, margin=dict(t=10, b=10, l=10, r=10),
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("_No market values reported (paper account may lack a "
                       "market-data subscription) — sizing shown by shares/cost._")

    _drift_section(payload)
    _download(payload, secret)


def _drift_section(payload: dict) -> None:
    """Compare what the executor TARGETED (target_book weights) against what it
    actually HOLDS now (executed positions), per instrument, in percentage points.
    Small drifts are normal: whole-share rounding, the no-trade band, and fills
    landing away from the sizing price."""
    if not TARGET_PATH.exists():
        return
    try:
        tgt = tb.loads(TARGET_PATH.read_text())
    except Exception:
        return

    st.markdown("---")
    st.markdown("### 🎯 vs Target Book — allocation drift")
    if tgt.get("as_of") and payload.get("as_of") and tgt["as_of"] != payload["as_of"]:
        st.caption(f"⚠️ Target book bar (**{tgt['as_of']}**) differs from this "
                   f"report's bar (**{payload['as_of']}**) — comparing the latest of each.")

    t_weights = tgt.get("weights", {}) or {}
    t_cash = float(tgt.get("cash_weight", 0.0) or 0.0)
    net_liq = float(payload.get("net_liq") or 0.0)
    cash = float(payload.get("cash") or 0.0)
    positions = payload.get("positions") or []

    def _val(p: dict) -> float:
        mv = float(p.get("market_value") or 0.0)
        return mv or float(p.get("shares") or 0.0) * float(p.get("avg_cost") or 0.0)

    actual_val: dict[str, float] = {}
    for p in positions:
        k = p.get("key") or p.get("symbol")
        actual_val[k] = actual_val.get(k, 0.0) + _val(p)

    denom = net_liq if net_liq > 0 else (sum(actual_val.values()) + cash)
    if denom <= 0:
        st.caption("Not enough account data to compute drift on this run.")
        return

    seen, ordered = set(), []
    for k in list(t_weights) + list(actual_val):
        if k not in seen:
            seen.add(k); ordered.append(k)
    ordered.sort(key=lambda k: -float(t_weights.get(k, 0.0)))

    rows = []
    for k in ordered:
        tw = float(t_weights.get(k, 0.0)) * 100
        aw = actual_val.get(k, 0.0) / denom * 100
        rows.append(dict(Instrument=_name(k, k), IBKR=(sym.trade_symbol(k) or k),
                         Target=tw, Actual=aw, Drift=aw - tw))
    rows.append(dict(Instrument="💵 Cash", IBKR="—", Target=t_cash * 100,
                     Actual=cash / denom * 100, Drift=cash / denom * 100 - t_cash * 100))

    st.dataframe(
        pd.DataFrame(rows), hide_index=True, use_container_width=True,
        column_config={
            "Target": st.column_config.NumberColumn("Target %", format="%.1f%%"),
            "Actual": st.column_config.NumberColumn("Actual %", format="%.1f%%"),
            "Drift": st.column_config.NumberColumn("Drift (pp)", format="%+.1f")})
    max_drift = max((abs(r["Drift"]) for r in rows), default=0.0)
    st.caption(f"Largest drift **{max_drift:.1f} pp**. Expected sources: whole-share "
               "rounding, the no-trade band, and fills vs the sizing price. A large "
               "drift on a name means it didn't fill as intended — check the trades above.")


def _download(payload: dict, secret) -> None:
    text = (json.dumps(payload, indent=1) if payload.get("signature")
            else tb.dumps(payload, secret))
    with st.expander("Raw execution report (JSON)"):
        st.code(text, language="json")
    st.download_button("⬇️ Download executed_book.json", data=text,
                       file_name="executed_book.json", mime="application/json")


# ══════════════════════════════════════════════════════════════════════════
st.title("✅ Executed Book (IBKR)")
st.caption("What the IBKR executor actually did on the last rebalance — trades "
           "placed and the resulting positions.")

# Which account views are available? (paper always; live once a live run exists)
_avail = [("Paper", REPORT_PATH)] + (
    [("Live", REPORT_PATH_LIVE)] if REPORT_PATH_LIVE.exists() else [])
if len(_avail) > 1:
    _pick = st.radio("Account", [n for n, _ in _avail], horizontal=True,
                     help="Paper and live executions are kept as separate reports.")
    _path = dict(_avail)[_pick]
else:
    _path = REPORT_PATH

if _path.exists():
    try:
        payload = tb.loads(_path.read_text())
        _render(payload, source=f"`{_path.relative_to(_REPO_ROOT)}`")
    except Exception as e:
        st.error(f"Could not read the execution report: {e}")
else:
    st.warning("No execution report found at "
               f"`{_path.relative_to(_REPO_ROOT)}`.")
    st.markdown(
        "It appears here once the executor has run a rebalance and committed the "
        "report back to the branch (`scripts/ibkr_execute_book.py` writes it; the "
        "daily wrapper commits it). Until then, see **📋 Target Book (IBKR)** for "
        "the intended allocation.")
