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

Two tabs:

  * **✅ Latest run** — the report as just described,
  * **🕰️ Historical** — the same view for any PAST run, picked by date from the
    dated as-of records the executor archives beside the report
    (``data/overall/executed_archive/<as_of>[_live].json``).

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
TARGET_ARCHIVE_DIR = _REPO_ROOT / "data" / "overall" / tb.ARCHIVE_DIRNAME

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


def _render(payload: dict, *, source: str, historical: bool = False) -> None:
    """Render one execution report.

    ``historical`` marks an archived record: its age is the point rather than a
    problem, so the freshness badge becomes an "archived record" stamp and the
    drift comparison uses the target book of that same signal bar."""
    secret = _secret()
    # both tabs render on every script run, so every element that Streamlit
    # auto-IDs from its parameters needs a scope-unique key — two identical
    # charts/tables on one run is a DuplicateElementId error
    scope = "hist" if historical else "now"
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
    if historical:
        # an archived run is SUPPOSED to be old — flag the date, not the age
        gen = payload.get("generated_at_utc") or "—"
        fresh_badge = _badge(f"🗄️ archived run · executed {gen} UTC", "#6366f1")
    else:
        fresh_badge = _badge(("🟢 " if ok_val else "🟡 ") + val_why,
                             "#16a34a" if ok_val else "#d97706")
    st.markdown(
        _badge(acct_txt, acct_col) + "  " +
        _badge(mode_txt, mode_col) + "  " + _badge(sig_txt, sig_col) + "  " +
        fresh_badge +
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
            lmt = float(t.get("limit_price") or 0.0)
            otype = {"marketable-limit": "LMT", "moc": "MOC",
                     "market": "MKT"}.get(t.get("order_type") or "", "—")
            trows.append(dict(
                Action=t.get("action"), Instrument=_name(t.get("key"), t.get("symbol")),
                IBKR=t.get("symbol"), Qty=qty,
                Type=(f"{otype} ${lmt:,.2f}" if lmt else otype),
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
            tdf, hide_index=True, use_container_width=True, key=f"{scope}_trades",
            column_config={
                "Qty": st.column_config.NumberColumn("Qty", format="%.0f"),
                "Action": st.column_config.TextColumn("Action")})

    st.markdown("---")

    # ── current positions ─────────────────────────────────────────────────────
    st.markdown("### 📊 Current IBKR positions")
    if not positions:
        st.info("No open positions reported.")
        _drift_section(payload, historical=historical)
        _download(payload, secret, historical=historical)
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
            pdf, hide_index=True, use_container_width=True, key=f"{scope}_positions",
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
            st.plotly_chart(fig, use_container_width=True, key=f"{scope}_donut")
        else:
            st.caption("_No market values reported (paper account may lack a "
                       "market-data subscription) — sizing shown by shares/cost._")

    _drift_section(payload, historical=historical)
    _download(payload, secret, historical=historical)


def _drift_section(payload: dict, *, historical: bool = False) -> None:
    """Compare what the executor TARGETED (target_book weights) against what it
    actually HOLDS now (executed positions), per instrument, in percentage points.
    Small drifts are normal: whole-share rounding, the no-trade band, and fills
    landing away from the sizing price.

    For an archived run the comparison uses the target book published for that
    run's own signal bar (``book_archive/<as_of>.json``) — comparing a past
    execution against today's book would only measure the days in between."""
    as_of = payload.get("as_of")
    if historical:
        # today's book says nothing about a past run — use that bar's own book
        tgt_path = (TARGET_ARCHIVE_DIR / f"{pd.Timestamp(as_of).date()}.json"
                    if as_of else None)
        if tgt_path is None or not tgt_path.exists():
            st.markdown("---")
            st.markdown("### 🎯 vs Target Book — allocation drift")
            st.caption(f"No archived target book for signal bar **{as_of or '—'}** "
                       "(`data/overall/book_archive/`), so there is nothing to "
                       "compare this run against.")
            return
    else:
        tgt_path = TARGET_PATH
        if not tgt_path.exists():
            return
    try:
        tgt = tb.loads(tgt_path.read_text())
    except Exception:
        return

    st.markdown("---")
    st.markdown("### 🎯 vs Target Book — allocation drift")
    if tgt.get("as_of") and payload.get("as_of") and tgt["as_of"] != payload["as_of"]:
        st.caption(f"⚠️ Target book bar (**{tgt['as_of']}**) differs from this "
                   f"report's bar (**{payload['as_of']}**) — comparing the latest of each.")
    elif historical:
        st.caption(f"Against the book published for signal bar **{tgt.get('as_of')}** "
                   f"(`{tgt_path.relative_to(_REPO_ROOT)}`) — the one this run traded.")

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
        key=f"{'hist' if historical else 'now'}_drift",
        column_config={
            "Target": st.column_config.NumberColumn("Target %", format="%.1f%%"),
            "Actual": st.column_config.NumberColumn("Actual %", format="%.1f%%"),
            "Drift": st.column_config.NumberColumn("Drift (pp)", format="%+.1f")})
    max_drift = max((abs(r["Drift"]) for r in rows), default=0.0)
    st.caption(f"Largest drift **{max_drift:.1f} pp**. Expected sources: whole-share "
               "rounding, the no-trade band, and fills vs the sizing price. A large "
               "drift on a name means it didn't fill as intended — check the trades above.")


def _download(payload: dict, secret, *, historical: bool = False) -> None:
    text = (json.dumps(payload, indent=1) if payload.get("signature")
            else tb.dumps(payload, secret))
    name = "executed_book.json"
    if historical and payload.get("as_of"):
        name = f"executed_book_{pd.Timestamp(payload['as_of']).date()}.json"
    with st.expander(f"Raw execution report (JSON){' — ' + str(payload.get('as_of')) if historical else ''}"):
        st.code(text, language="json")
    st.download_button(f"⬇️ Download {name}", data=text, file_name=name,
                       mime="application/json",
                       key=f"dl_{'hist' if historical else 'latest'}_{name}")


# ══════════════════════════════════════════════════════════════════════════
# HISTORICAL TAB — browse the dated as-of records the executor archives
# ══════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def _records(report_path_str: str, sig: tuple) -> list[dict]:
    """Archived runs for this account mode, newest first.

    Cached on *sig*, a signature of the archive directory (names + mtimes +
    sizes), so a 45-second auto-refresh re-reads the files only when they
    actually change. (The name must NOT start with an underscore — Streamlit
    leaves underscored arguments out of the cache key, which would pin the
    first read forever.)"""
    return eb.archived_records(report_path_str)


def _archive_sig(report_path: Path) -> tuple:
    d = eb.archive_dir(report_path)
    if not d.is_dir():
        return ()
    return tuple(sorted((f.name, f.stat().st_mtime, f.stat().st_size)
                        for f in d.glob("*.json")))


def _history_index(recs: list[dict]) -> pd.DataFrame:
    """One row per archived run — the at-a-glance record of every rebalance."""
    rows = []
    for r in recs:
        pl = r["payload"]
        trades = pl.get("trades") or []
        filled = [t for t in trades if float(t.get("filled") or 0.0) > 0]
        rows.append(dict(
            Executed=r["executed_on"], Signal_bar=r["as_of"],
            Account=("🔴 live" if (pl.get("account_mode") or "paper").lower() == "live"
                     else "🧪 paper"),
            Mode=("executed" if (pl.get("mode") or "").lower() == "execute"
                  else (pl.get("mode") or "—")),
            Trades=len(trades), Filled=len(filled),
            Positions=len(pl.get("positions") or []),
            Net_liq=float(pl.get("net_liq") or 0.0),
            Cash=float(pl.get("cash") or 0.0)))
    return pd.DataFrame(rows)


def _render_history(report_path: Path) -> None:
    recs = _records(str(report_path), _archive_sig(report_path))
    reset = eb.read_reset(report_path)
    if not recs:
        st.info("No archived runs yet for this account.")
        if reset:
            st.markdown(
                f"The record was **reset on {str(reset.get('reset_at_utc'))[:10]}** "
                f"({reset.get('reason')}): runs up to and including signal bar "
                f"**{reset.get('cutoff_as_of')}** belong to the previous account and "
                "are not shown. The next execution report starts the record fresh.")
        else:
            st.markdown(
                "Every rebalance overwrites `data/overall/executed_book.json`, so past "
                "runs are kept as dated records beside it in "
                "`data/overall/executed_archive/<signal-bar>.json`. The executor writes "
                "one on each run (`scripts/ibkr_execute_book.py`) and the daily wrapper "
                "commits it; to seed the archive from runs that happened before it "
                "existed, run **`python scripts/backfill_executed_archive.py`** on an "
                "unshallowed clone — it rebuilds the records verbatim from git history.")
        return
    if reset:
        st.caption(f"↺ Record reset on {str(reset.get('reset_at_utc'))[:10]} "
                   f"({reset.get('reason')}) — runs up to signal bar "
                   f"**{reset.get('cutoff_as_of')}** were a previous account's and "
                   "are not shown.")

    dates = sorted(r["executed_on"] for r in recs)          # ascending
    first, last = pd.Timestamp(dates[0]).date(), pd.Timestamp(dates[-1]).date()

    st.markdown("### 📅 Pick a past execution")
    c1, c2 = st.columns([1, 2])
    with c1:
        picked = st.date_input(
            "Execution date", value=last, min_value=first, max_value=last,
            help="The day the executor ran. It trades the morning after the "
                 "signal bar, so a run dated the 18th executed the book "
                 "published from the 17th's close.")
    with c2:
        st.caption(f"**{len(recs)}** archived run(s) from **{first}** to "
                   f"**{last}**. Days without a run (weekends, holidays, a "
                   "skipped cycle) fall back to the most recent run on or "
                   "before the date you pick.")

    if isinstance(picked, (tuple, list)):        # defensive: never a range here
        picked = picked[-1] if picked else None
    if picked is None:                           # the input was cleared
        picked = last
        st.caption("No date selected — showing the most recent run.")
    key = str(picked)
    rec = eb.record_for(recs, picked)
    if rec is None:
        st.warning(f"No run on or before **{key}** — the archive starts at "
                   f"**{first}**.")
        return
    if rec["executed_on"] != key:            # snapped back to the standing run
        st.caption(f"No run on **{key}** — showing the most recent one before it, "
                   f"**{rec['executed_on']}**.")

    st.markdown(f"#### 🕰️ Run of {rec['executed_on']} "
                f"(signal bar {rec['as_of']})")
    _render(rec["payload"], source=f"`{rec['path'].relative_to(_REPO_ROOT)}`",
            historical=True)

    # ── the whole archive at a glance ────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🗂️ All archived runs")
    idx = _history_index(recs)
    shown = idx.assign(
        Net_liq=idx["Net_liq"].map(lambda v: f"${v:,.0f}" if v else "—"),
        Cash=idx["Cash"].map(lambda v: f"${v:,.0f}" if v else "—"))
    st.dataframe(
        shown, hide_index=True, use_container_width=True, key="hist_index",
        column_config={
            "Signal_bar": st.column_config.TextColumn("Signal bar"),
            "Net_liq": st.column_config.TextColumn("Net liq"),
            "Filled": st.column_config.NumberColumn(
                "Filled", help="Trades with a non-zero fill quantity.")})

    nl = idx[idx["Net_liq"] > 0].sort_values("Executed")
    if len(nl) > 1:
        fig = go.Figure(go.Scatter(
            x=pd.to_datetime(nl["Executed"]), y=nl["Net_liq"], mode="lines+markers",
            line=dict(color="#0ea5e9", width=2), name="Net liquidation",
            hovertemplate="%{x|%b %d, %Y}<br>$%{y:,.0f}<extra></extra>"))
        fig.add_vline(x=pd.Timestamp(rec["executed_on"]), line_dash="dot",
                      line_color="#dc2626")
        fig.update_layout(height=260, margin=dict(t=30, b=10, l=10, r=10),
                          yaxis_title="Net liquidation ($)", hovermode="x unified",
                          xaxis=dict(tickformat="%b %d", dtick="D1"),
                          title=dict(text="Account value as reported at each run",
                                     font_size=13))
        st.plotly_chart(fig, use_container_width=True, key="hist_netliq")
        st.caption("Account value as the executor read it on each run — a record "
                   "of the account, not a back-test: it moves with deposits and "
                   "withdrawals as well as with the strategy.")


# ══════════════════════════════════════════════════════════════════════════
st.title("✅ Executed Book (IBKR)")
import strategy_version as _sv                 # noqa: E402
if not (hasattr(_sv, "render_badge") and hasattr(_sv, "BADGE_COLOR")):
    import importlib                           # stale hot-loaded module (the
    _sv = importlib.reload(_sv)                # server kept an older import)
_sv.render_badge()
st.caption("What the IBKR executor actually did on the last rebalance — trades "
           "placed and the resulting positions — plus every earlier run, by date, "
           "under 🕰️ Historical.")

# Which account views are available? (paper always; live once a live run exists —
# either as the current report or as an archived one)
_live_seen = REPORT_PATH_LIVE.exists() or bool(
    eb.archive_dir(REPORT_PATH_LIVE).is_dir()
    and list(eb.archive_dir(REPORT_PATH_LIVE).glob("*_live.json")))
_avail = [("Paper", REPORT_PATH)] + ([("Live", REPORT_PATH_LIVE)] if _live_seen else [])
if len(_avail) > 1:
    _pick = st.radio("Account", [n for n, _ in _avail], horizontal=True,
                     help="Paper and live executions are kept as separate reports.")
    _path = dict(_avail)[_pick]
else:
    _path = REPORT_PATH

_tab_now, _tab_hist = st.tabs(["✅ Latest run", "🕰️ Historical"])

with _tab_now:
    if _path.exists():
        try:
            payload = tb.loads(_path.read_text())
            _render(payload, source=f"`{_path.relative_to(_REPO_ROOT)}`")
        except Exception as e:
            st.error(f"Could not read the execution report: {e}")
    else:
        _reset = eb.read_reset(_path)
        st.warning("No execution report found at "
                   f"`{_path.relative_to(_REPO_ROOT)}`.")
        st.markdown(
            (f"The record was **reset on {str(_reset.get('reset_at_utc'))[:10]}** "
             f"({_reset.get('reason')}) — the previous account's runs were retired "
             "up to signal bar **" + str(_reset.get("cutoff_as_of")) + "**. "
             if _reset else "") +
            "It appears here once the executor has run a rebalance and committed the "
            "report back to the branch (`scripts/ibkr_execute_book.py` writes it; the "
            "daily wrapper commits it). Until then, see **📋 Target Book (IBKR)** for "
            "the intended allocation.")

with _tab_hist:
    st.caption("Previously executed books — every run the executor archived, "
               "picked by date. Same view as the latest run: the trades it "
               "placed, the positions it ended with, and how that stood against "
               "the target book it was trading.")
    _render_history(_path)
