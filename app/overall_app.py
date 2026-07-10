"""Overall Trading — the combined cross-asset decision cockpit.

One screen that fuses the live signals, positions and back-tests of every other
app into a single portfolio view built for one question: *where do I put money to
work today?*  Each signal app trades its 1× primary plus higher-beta / leveraged
siblings (BTC→MSTR/MSTU, Gold→GDX/UGL, XLE→OIH), all steered off the parent
signal, so the combined book spans every instrument across all apps.

  🔴 Live — Decision Cockpit   what to CLOSE / OPEN / HOLD today, the optimal %
                                allocation, and the strategy's current book.
  📊 Combined Backtesting       the historically-optimal blend of all the
                                instruments' signal-driven strategies vs the
                                obvious benchmarks, per-window and per-instrument.
  🧠 Strategy & Methodology     how every number here is produced.

All maths lives in ``overall_core``; this module is the (thin, cached) Streamlit
layer.  The BTC and Gold apps are never imported — this app re-runs everything
through one unified daily engine so the strategies are directly comparable and
blendable.  The sidebar app-selector is rendered before any heavy work so it can
never be blanked by a slow or failed data fetch.
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
import inspect as _inspect


def _stale_core(mod) -> bool:
    """True if the imported overall_core is an old hot-reloaded copy that lacks a
    current capability. Streamlit reruns THIS entry script but reuses
    already-imported dependency modules, so after an overall_core update the app
    can otherwise call new code against a stale module (e.g. optimize_weights
    without the `fundamental` overlay arg) and crash until a full restart."""
    if not hasattr(mod, "run_universe") or not hasattr(mod, "optimize_weights"):
        return True
    try:
        if "fundamental" not in _inspect.signature(mod.optimize_weights).parameters:
            return True
        if "include_entries" not in _inspect.signature(mod.live_exit_keys).parameters:
            return True
        if not getattr(mod, "LIVE_EXIT_MODE_AWARE", False):
            return True
        return False
    except (ValueError, TypeError, AttributeError):
        return False


if _stale_core(ov):
    ov = importlib.reload(ov)

try:                                  # optional live auto-refresh (graceful if absent)
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

N_ALL = len(ov.SPOT_SYMBOLS)          # total instruments the universe spans
_AUTOREFRESH_SECS = 45                # re-run cadence so the live price column ticks


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
_KIND_TAG = {"lev": ("2×", "#7c3aed"), "beta": ("β", "#0891b2"), "core": ("", "")}


# ══════════════════════════════════════════════════════════════════════════
# Sidebar — unified application selector (rendered FIRST, always).
# ══════════════════════════════════════════════════════════════════════════
if st.session_state.get("gldm_active_app") not in _ALL_APPS:
    st.session_state["gldm_active_app"] = "OVERALL"
if st.session_state.get("overall_risk_profile") not in ov.RISK_PROFILES:
    st.session_state["overall_risk_profile"] = ov.DEFAULT_PROFILE
st.session_state.setdefault("overall_use_fundamental", True)
with st.sidebar:
    st.radio("**Application**", options=_ALL_APPS,
             format_func=lambda x: _APP_LABELS.get(x, x), key="gldm_active_app")
    st.markdown("---")
    st.markdown(f"**Risk profile:** `{st.session_state['overall_risk_profile']}` "
                "— switch it on the 🔴 **Live** tab.")
    st.markdown("---")
    st.markdown("**Auto-refresh:** live data cached ~15 min.")
    if st.button("Refresh now", use_container_width=True):
        st.cache_data.clear(); st.cache_resource.clear(); st.rerun()
    st.caption("_Overall Trading fuses all the asset apps into one portfolio. "
               "Each app's signal trades its 1× primary plus higher-beta / "
               "leveraged siblings, all re-run through one unified daily engine._")

# Auto-rerun the page every ~45s so the live spot-price column ticks without a
# click.  It only re-fetches quotes when the 60-second spot cache expires, and
# the heavy strategy/optimise stays on its 30-min cache — so each tick is cheap.
# Session state (risk profile, portfolio value) is preserved across the rerun.
if st_autorefresh is not None:
    st_autorefresh(interval=_AUTOREFRESH_SECS * 1000, key="overall_spot_tick")


# ══════════════════════════════════════════════════════════════════════════
# Cached compute — run all instruments, then the portfolio maths
# ══════════════════════════════════════════════════════════════════════════
def _bucket() -> str:
    # 30-minute granularity: the strategy runs on daily bars (positions/signals
    # are daily-stable), and the *prices* are overlaid live from a separate
    # 60-second spot cache — so the heavy compute needn't rerun more often, and
    # switching away to another app and back stays a warm-cache (instant) hit.
    now = pd.Timestamp.utcnow()
    return f"{now.date()}-{now.hour}-{now.minute // 30}"


@st.cache_data(ttl=1800, show_spinner="Running every strategy live (first load ~30–60s)…")
def get_results(bucket: str):
    return ov.run_universe()


@st.cache_data(ttl=60, show_spinner=False)
def get_spot(minute_bucket: str):
    """Live spot prices, refreshed ~every minute (separate from the 15-min
    strategy cache) so the action plan shows current prices, not stale bars."""
    return ov.fetch_spot()


@st.cache_data(ttl=60, show_spinner=False)
def get_sata_spot(minute_bucket: str):
    """Live SATA quote (price, day-change, P&L vs $100 par), refreshed ~1/min."""
    return ov.fetch_sata()


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_entry_closes(positions: tuple):
    """Real official close on each open position's entry date — the cost basis.
    Cached on the (key, symbol, entry-date) tuple so it only re-fetches when a
    position actually opens/closes, not on every price refresh."""
    from concurrent.futures import ThreadPoolExecutor
    if not positions:
        return {}
    def _one(item):
        k, sym, dt = item
        return k, ov._bar_close(sym, dt)
    with ThreadPoolExecutor(max_workers=min(13, len(positions))) as ex:
        return {k: v for k, v in ex.map(_one, positions) if v}


@st.cache_data(ttl=1800, show_spinner="Optimising the combined allocation…")
def get_all_profiles(bucket: str, fundamental: bool = True):
    """Compute the full portfolio for EVERY risk profile once, so switching
    profiles (and rendering the comparison table) is instant — no recompute.
    ``fundamental`` applies the mid-2026 sector forward-view overlay."""
    results = get_results(bucket)
    if not results:
        return None
    rets = ov.returns_matrix(results)
    pos = ov.position_matrix(results, rets.index)
    sata = ov.SATA_DAILY
    bm = ov.benchmarks(rets, results, pos=pos, sata_daily=sata)      # profile-independent
    base_curves = {"Equal-weight strategies": bm["strat_equal"]["equity"],
                   "Equal-weight Buy & Hold": bm["bh_equal"]["equity"]}
    profiles = {}
    for name, prof in ov.RISK_PROFILES.items():
        caps = ov.caps_for(name)
        opt = ov.optimize_weights(rets, caps=caps, pos=pos, sata_daily=sata,
                                  mdd_floor=prof["mdd_floor"], objective=prof["objective"],
                                  fundamental=fundamental)
        w_opt = np.array([opt["optimal"]["weights"][c] for c in opt["cols"]])
        per = ov.period_breakdown(rets, w_opt, ov.COMBINED_PERIODS, pos=pos, sata_daily=sata)
        curves = {"Optimal blend": ov._equity(ov._combine(rets, w_opt, pos, sata)),
                  **base_curves}
        gate = ov.signal_gated_allocation(results, opt["optimal"]["weights"], caps=caps)
        profiles[name] = dict(opt=opt, per=per, curves=curves, gate=gate, w_opt=w_opt)
    return dict(results=results, rets=rets, bm=bm, profiles=profiles)


def get_portfolio(bucket: str, profile: str, fundamental: bool = True):
    allp = get_all_profiles(bucket, fundamental)
    if not allp:
        return None
    p = allp["profiles"][profile]
    return dict(results=allp["results"], rets=allp["rets"], bm=allp["bm"],
                profile=profile, **p)


def get_profile_comparison(bucket: str, fundamental: bool = True):
    """Headline metrics for every risk profile (reads the shared computation)."""
    allp = get_all_profiles(bucket, fundamental)
    if not allp:
        return []
    rows = []
    for name, prof in ov.RISK_PROFILES.items():
        o = allp["profiles"][name]["opt"]["optimal"]
        bl = sum(v for k, v in o["weights"].items()
                 if ov.ASSET_META.get(k, {}).get("kind") in ("beta", "lev"))
        rows.append(dict(name=name, blurb=prof["blurb"], betalev=bl, **{
            k: o[k] for k in ("total_ret", "cagr", "mdd", "sharpe")}))
    return rows


st.title("🧭 Overall Trading — Combined Decision Cockpit")
st.caption(f"Every asset app, fused into one portfolio spanning {N_ALL} instruments "
           "(each app's 1× primary plus its higher-beta / leveraged siblings). "
           "Live entry/exit signals, current positions, the historically-optimal "
           "cross-asset allocation, and one combined back-test — built around a "
           "single question: **where should capital go today?**")

_profile = st.session_state["overall_risk_profile"]
_use_fund = st.session_state["overall_use_fundamental"]
try:
    _PF = get_portfolio(_bucket(), _profile, _use_fund)
except Exception as exc:                       # never blank the sidebar/selector
    st.error(f"Live data fetch hit an error: {exc}. Press **Refresh now** in a moment.")
    st.stop()
if not _PF:
    st.error("Could not fetch market data for any instrument right now. "
             "Press **Refresh now** in a moment.")
    st.stop()

results = _PF["results"]
by_key = {r["key"]: r for r in results}
opt = _PF["opt"]; gate = _PF["gate"]; bm = _PF["bm"]
as_of = max(r["as_of"] for r in results)
# group instruments by parent signal, preserving parent order
parents = []
for pk in ov.PARENT_KEYS:
    grp = [r for r in results if r["parent"] == pk]
    if grp:
        parents.append((pk, grp))

# ── diagnostic: make a silently-dropped app VISIBLE ─────────────────────────
# All apps' instruments should load. If one fails in a given environment
# (e.g. a data/model file missing or a library-version mismatch), its sleeve is
# dropped and every combined number changes — so surface it loudly instead of
# showing a quietly-reduced universe.
_present = {r["parent"] for r in results}
_missing = [k for k in ov.PARENT_KEYS if k not in _present]
if _missing:
    _errs = getattr(ov, "_LAST_ERRORS", {})
    _lines = "; ".join(f"**{k}** ({_errs.get(k, 'not loaded')})" for k in _missing)
    st.warning(
        f"⚠️ Only **{len(results)}/{N_ALL}** instruments loaded — the following "
        f"app(s) failed to load, so the combined strategy & back-test below are "
        f"computed on a **reduced universe** and won't match the full-universe "
        f"figures: {_lines}. Press **Refresh now**; if it persists, the app's "
        f"data/model files may be unavailable in this environment.")

# ── live spot prices — overlay onto the display (not the signals/back-test) ──
# Signals run on completed daily bars (some cached, so a few days stale for
# BTC/MSTR/MSTU); the action-plan price and open-position P&L must show the
# current spot, so fetch each instrument's live quote and overlay it.
_spot_ts = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M")
_spot = {}
_sata = {}
try:
    # cost basis = the real official close on each open position's entry bar
    _pos_key = tuple(sorted(
        (r["key"], ov.SPOT_SYMBOLS[r["key"]],
         str(pd.Timestamp(r["pos"]["entry_date"]).date()))
        for r in results
        if r["pos"].get("in_pos") and r["pos"].get("entry_date")
        and r["key"] in ov.SPOT_SYMBOLS))
    ov.apply_entry_basis(results, get_entry_closes(_pos_key))
    _spot = get_spot(_spot_ts)
    _sata = get_sata_spot(_spot_ts)
    ov.apply_spot(results, _spot)
    for _a in gate["actions"]:            # keep the action table price/P&L in sync
        _r = by_key.get(_a["key"])
        if _r:
            _a["last_close"] = _r["last_close"]; _a["upnl"] = _r["pos"]["upnl"]
    _n_spot = sum(1 for v in _spot.values() if v.get("price"))
except Exception:
    _n_spot = 0


# ── small HTML helpers ─────────────────────────────────────────────────────
def _pill(text, color, bg=None):
    bg = bg or (color + "22")
    return (f"<span style='background:{bg};color:{color};font-weight:700;"
            f"padding:2px 9px;border-radius:999px;font-size:12px;white-space:nowrap'>"
            f"{text}</span>")


def _kind_badge(kind):
    tag, col = _KIND_TAG.get(kind, ("", ""))
    if not tag:
        return ""
    return (f"<span style='background:{col};color:#fff;font-size:9px;font-weight:800;"
            f"padding:1px 5px;border-radius:5px;margin-left:5px;vertical-align:middle'>{tag}</span>")


def _pct(x, digits=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.{digits}f}%"


tab_live, tab_bt, tab_explain = st.tabs(
    ["🔴 Live — Decision Cockpit", "📊 Combined Backtesting",
     "🧠 Strategy & Methodology"])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE DECISION COCKPIT
# ══════════════════════════════════════════════════════════════════════════
with tab_live:
    _auto = (f" · auto-refreshing every {_AUTOREFRESH_SECS}s"
             if (st_autorefresh is not None and _n_spot) else "")
    _px_note = (f" · <span style='color:#16a34a'>● prices live (spot, {_n_spot}/{N_ALL})</span>{_auto}"
                if _n_spot else " · <span style='color:#dc2626'>spot quote unavailable — showing last bar</span>")
    st.markdown(f"#### Signals as of **{as_of.strftime('%b %d, %Y')}** · "
                f"{len(results)} instruments across {len(parents)} signals{_px_note}",
                unsafe_allow_html=True)

    # ── risk-profile switch — decide and trade accordingly ──────────────
    pcomp = {r["name"]: r for r in get_profile_comparison(_bucket(), _use_fund)}
    rp = st.columns([1.15, 2])
    with rp[0]:
        st.radio("⚙️ **Risk profile**", list(ov.RISK_PROFILES.keys()),
                 key="overall_risk_profile", horizontal=True)
        st.caption(ov.RISK_PROFILES[_profile]["blurb"])
    with rp[1]:
        cells = []
        for name in ov.RISK_PROFILES:
            r = pcomp.get(name)
            if not r:
                continue
            on = name == _profile
            cells.append(
                f"<div style='flex:1;border:2px solid {'#2563eb' if on else '#e2e8f0'};"
                f"background:{'#eff6ff' if on else '#fff'};border-radius:9px;"
                f"padding:7px 10px;min-width:120px'>"
                f"<div style='font-size:12px;font-weight:800;color:#1e293b'>{name}"
                f"{' ◄ active' if on else ''}</div>"
                f"<div style='font-size:12px;margin-top:2px'>ret "
                f"<b style='color:{C_BUY}'>{r['total_ret']*100:,.0f}%</b> · "
                f"dd <b style='color:{C_EXIT}'>{r['mdd']*100:.0f}%</b> · "
                f"Sharpe <b>{r['sharpe']:.2f}</b></div></div>")
        st.markdown(
            "<div style='display:flex;gap:8px;align-items:stretch'>" + "".join(cells) + "</div>",
            unsafe_allow_html=True)
    st.checkbox("🔭 **Apply fundamental view** (mid-2026 sector outlook)",
                key="overall_use_fundamental",
                help=ov.FUNDAMENTAL_VIEW_NOTE + " Tilts each profile's optimal blend "
                     "toward the strongest secular-growth sleeves (re-capped), then "
                     "re-runs the allocation and back-test. Uncheck for the pure "
                     "historical quant optimum.")
    if _use_fund:
        st.caption(f"🔭 **Fundamental overlay ON** — {ov.FUNDAMENTAL_VIEW_NOTE}")
        st.caption("Switching reruns today's targets and the back-test on the "
                   "chosen profile. β / 2× exposure rises Balanced → Aggressive: "
                   "more return, deeper drawdowns, lower Sharpe.")
    st.markdown("")

    invested = 1.0 - gate["sata"]
    k = st.columns(5)
    k[0].metric("Positions open", f"{gate['n_active']}")
    k[1].metric("Open today", f"{gate['n_open']}",
                delta="new entries" if gate["n_open"] else None)
    k[2].metric("Close today", f"{gate['n_close']}",
                delta="exit signal" if gate["n_close"] else None, delta_color="inverse")
    k[3].metric("Target in risk assets", f"{invested*100:.0f}%")
    k[4].metric("Target in SATA", f"{gate['sata']*100:.0f}%",
                help=f"{gate['sata_info']['name']} — idle cash parked at "
                     f"~{gate['sata_info']['annual_rate']*100:.0f}% yield.")

    # live-adjusted allocation (drops MA instruments whose live price is below the
    # trend filter → they exit next bar) — used by the action table's Live target
    # columns and by the "Recommended now" pie below. include_entries flags fresh
    # buys that would open into an already-broken trend as well as current holds.
    _live_exits = ov.live_exit_keys(results, _spot, include_entries=True)
    try:
        gate_live = ov.signal_gated_allocation(
            results, opt["optimal"]["weights"], caps=ov.caps_for(_profile),
            force_exit=_live_exits)
    except Exception:
        gate_live = gate

    # ── 1. OPTIMAL ALLOCATION TODAY vs CURRENT BOOK ─────────────────────
    st.markdown("### 📐 Optimal allocation for today")
    st.caption("**Current book** = what the strategy holds right now. "
               "**Recommended today** = the book after today's *committed* "
               "(last-close) signals — risk assets sized by the optimal weights "
               "tilted by entry priority (caps: 30% core, 18% high-beta, 10% "
               "leveraged); the remainder sits in **SATA**. **Recommended now "
               "(live-adjusted)** additionally drops any position whose *live* "
               "price has fallen below its trend filter — it still holds today "
               "but exits on the next bar — and reallocates that capital to the "
               "survivors and SATA.")
    # (_live_exits / gate_live computed above, before the action table)
    ac = st.columns([1, 1, 1])

    def _alloc_donut(alloc: dict, sata: float, title: str):
        labels, vals, colors = [], [], []
        for kk, vv in sorted(alloc.items(), key=lambda x: -x[1]):
            if vv <= 0.0005:
                continue
            tag = {"lev": " 2×", "beta": " β"}.get(by_key[kk]["kind"], "")
            labels.append(f"{kk}{tag}"); vals.append(vv * 100)
            colors.append(by_key[kk]["accent"])
        if sata > 0.005:
            labels.append("SATA"); vals.append(sata * 100); colors.append("#334155")
        if not vals:
            labels, vals, colors = ["SATA"], [100.0], ["#334155"]
        fig = go.Figure(go.Pie(labels=labels, values=vals, hole=0.58,
                               marker=dict(colors=colors,
                                           line=dict(color="#fff", width=1)),
                               sort=False, textinfo="label+percent", textfont_size=11,
                               hovertemplate="%{label}: %{value:.1f}%<extra></extra>"))
        fig.update_layout(title=dict(text=title, font_size=14), showlegend=False,
                          height=320, margin=dict(t=40, b=10, l=10, r=10))
        return fig

    _live_title = ("Recommended now (live-adjusted)" if _live_exits
                   else "Recommended now (live) — no pending exits")
    ac[0].plotly_chart(_alloc_donut(gate["current"], gate["sata_now"],
                                    "Current book"), use_container_width=True)
    ac[1].plotly_chart(_alloc_donut(gate["target"], gate["sata"],
                                    "Recommended today"), use_container_width=True)
    ac[2].plotly_chart(_alloc_donut(gate_live["target"], gate_live["sata"],
                                    _live_title), use_container_width=True)
    if _live_exits:
        st.warning("⚠️ **Live-adjusted:** " + ", ".join(sorted(_live_exits)) +
                   " " + ("is" if len(_live_exits) == 1 else "are") +
                   " signalling long off the last close but the **live price is now "
                   "below the trend filter**, so " +
                   ("it" if len(_live_exits) == 1 else "they") +
                   " won't hold — " + ("it exits" if len(_live_exits) == 1
                                        else "they exit") +
                   " on the next bar. The **Recommended now** pie removes "
                   + ("it" if len(_live_exits) == 1 else "them") +
                   " and reallocates to the survivors and SATA.")
    else:
        st.caption("No held position's live price is below its trend filter — the "
                   "live-adjusted book matches **Recommended today**.")

    st.markdown("**Rebalancing moves** — current book → live-adjusted target")
    moves = []
    for kk in sorted(set(gate["current"]) | set(gate_live["target"]),
                     key=lambda x: -(gate_live["target"].get(x, 0))):
        cur = gate["current"].get(kk, 0.0); tg = gate_live["target"].get(kk, 0.0)
        d = tg - cur
        if abs(d) < 0.005:
            continue
        col = C_BUY if d > 0 else C_EXIT
        moves.append(f"<span style='font-size:13px;margin-right:16px;white-space:nowrap'>"
                     f"<b>{kk}</b>{_kind_badge(by_key[kk]['kind'])} "
                     f"{cur*100:.0f}% → {tg*100:.0f}% "
                     f"<span style='color:{col};font-weight:700'>"
                     f"{'▲' if d>0 else '▼'}{abs(d)*100:.0f}pt</span></span>")
    dsata = gate_live["sata"] - gate["sata_now"]
    if abs(dsata) >= 0.005:
        col = C_EXIT if dsata > 0 else C_BUY
        moves.append(f"<span style='font-size:13px;margin-right:16px;white-space:nowrap'>"
                     f"💵 <b>SATA</b> "
                     f"{gate['sata_now']*100:.0f}% → {gate_live['sata']*100:.0f}% "
                     f"<span style='color:{col};font-weight:700'>"
                     f"{'▲' if dsata>0 else '▼'}{abs(dsata)*100:.0f}pt</span></span>")
    st.markdown("".join(moves) if moves
                else "_Book already at target — no rebalancing needed._",
                unsafe_allow_html=True)

    st.markdown("---")

    # ── 2. TODAY'S ACTION PLAN ──────────────────────────────────────────
    st.markdown("### 🎯 Today's action plan")
    st.caption("What to do now, ranked: **close** exits first, then **open** / "
               "**hold**, ordered by entry priority. β = higher-beta sibling · "
               "2× = leveraged (traded off the parent signal). The **priority** "
               "score (0–1) blends live momentum, macro sentiment, the strategy's "
               "back-tested win-rate and its risk-adjusted edge — it decides which "
               "signals get funded and how much. Held/opened risk assets total "
               "100%; any capped-out remainder is parked in **SATA**. Rows shaded "
               "**red** ⚠️ are holds **or fresh entries** whose trend has broken on "
               "the live price — they still **hold/open today** but **exit on the "
               "next bar** (either the last close already crossed the trend, or the "
               "live price has since slipped below it). **Unreal. P&L** is measured "
               "against each position's real cost basis — the official close on its "
               "entry bar. **Target % / $ (Last bar)** is the committed allocation "
               "from the last-close signals; **Target % / $ (Live)** re-runs it "
               "against the current live price, dropping any position exiting next "
               "bar and reallocating to the survivors and SATA (differences are "
               "coloured green/red).")
    _pv_cols = st.columns([1, 2])
    with _pv_cols[0]:
        portfolio_value = st.number_input(
            "💼 Portfolio value ($)", min_value=0.0, value=100000.0, step=1000.0,
            format="%.0f", key="overall_portfolio_value",
            help="Target $ per instrument = target % × this value.")
    hdr = ("<tr style='background:#f1f5f9;font-size:12px;text-align:left'>"
           "<th style='padding:7px 10px'>Action</th><th>Instrument</th>"
           "<th>Live signal</th><th style='text-align:center'>Priority</th>"
           "<th style='text-align:right'>Price</th>"
           "<th style='text-align:right'>Chg %</th>"
           "<th style='text-align:right'>Unreal. P&amp;L</th>"
           "<th style='text-align:right'>Target % (Last bar)</th>"
           "<th style='text-align:right'>Target $ (Last bar)</th>"
           "<th style='text-align:right'>Target % (Live)</th>"
           "<th style='text-align:right'>Target $ (Live)</th></tr>")
    rows = []
    for a in gate["actions"]:
        ac = _ACTION_COL[a["action"]]
        tgt = a["target"]                                    # last-bar (committed)
        tgt_live = gate_live["target"].get(a["key"], 0.0)    # live-adjusted
        tgt_s = f"{tgt*100:.1f}%" if tgt > 0.0005 else "—"
        pnl = _pct(a["upnl"]) if a["in_pos"] else "—"
        pnl_col = (C_BUY if (a["upnl"] or 0) >= 0 else C_EXIT) if a["in_pos"] else "#94a3b8"
        _r = by_key[a["key"]]
        off = "" if a["kind"] == "core" else f" · off {a['parent']} signal"
        if _r["mode"] in ("ma", "dual_ma", "ma_vol") and _r.get("ma_val"):
            dist = (_r["last_close"] / _r["ma_val"] - 1) * 100
            sub = f"{a['parent']} close {dist:+.1f}% vs {_r.get('engine_label', 'trend')}{off}"
        else:
            sub = f"{a['parent']} alert: {a['alert']}{off}"
        if a["priority"] is not None:
            p = a["priority"]; pcolor = C_BUY if p >= 0.6 else C_WATCH if p >= 0.4 else C_FLAT
            prio_cell = (f"<span style='font-weight:700;color:{pcolor}'>{p:.2f}</span>"
                         f"<div style='height:5px;background:#e2e8f0;border-radius:3px;margin-top:2px'>"
                         f"<div style='height:5px;width:{p*100:.0f}%;background:{pcolor};border-radius:3px'></div></div>")
        else:
            prio_cell = "<span style='color:#cbd5e1'>—</span>"
        def _tgt_bar(t):
            return (f"<div style='height:7px;background:#e2e8f0;border-radius:4px;overflow:hidden;"
                    f"margin-top:3px'><div style='height:7px;width:{min(t*100,100):.0f}%;"
                    f"background:{_r['accent']}'></div></div>" if t > 0.0005 else "")
        bar = _tgt_bar(tgt)
        amt_s = f"${tgt * portfolio_value:,.0f}" if tgt > 0.0005 else "—"
        # live-adjusted target (accounts for positions exiting next bar on live px)
        tgt_live_s = f"{tgt_live*100:.1f}%" if tgt_live > 0.0005 else "—"
        bar_live = _tgt_bar(tgt_live)
        amt_live_s = f"${tgt_live * portfolio_value:,.0f}" if tgt_live > 0.0005 else "—"
        # highlight a live target that has diverged from the last-bar target
        _live_moved = abs(tgt_live - tgt) > 0.005
        _live_col = (C_EXIT if tgt_live < tgt else C_BUY) if _live_moved else "inherit"
        # today's price change (%) — live day-change from the spot overlay
        _dchg = _r.get("dchg")
        if _dchg is None or (isinstance(_dchg, float) and np.isnan(_dchg)):
            chg_s, chg_col = "—", "#94a3b8"
        else:
            chg_s = f"{_dchg:+.2f}%"
            chg_col = C_BUY if _dchg >= 0 else C_EXIT
        # cost basis = the real close on the entry bar
        _cb = _r["pos"].get("entry_px")
        cb_sub = (f"<div style='font-size:10px;color:#94a3b8'>@ ${_cb:,.2f} cost</div>"
                  if a["in_pos"] and _cb else "")
        # a hold/entry whose trend has broken drops out on the NEXT bar — flag the
        # whole row red so it's clear the position is about to be closed out. This
        # covers two cases: the committed last-close exit (exits_next_bar), and the
        # live-price exit — a row that still holds/opens today but whose live price
        # has fallen below the trend filter (in _live_exits, incl. fresh entries).
        _committed_exit = bool(a.get("exits_next_bar"))
        _live_exit = a["key"] in _live_exits
        exit_next = _committed_exit or _live_exit
        row_style = ("border-bottom:1px solid #fecaca;background:#fef2f2;"
                     "box-shadow:inset 3px 0 0 #dc2626" if exit_next
                     else "border-bottom:1px solid #eef2f7")
        if not exit_next:
            warn = ""
        elif _committed_exit:                      # last close already below trend
            warn = ("<div style='font-size:10px;color:#dc2626;font-weight:700'>"
                    "⚠️ exits next bar</div>")
        else:                                      # live-driven (hold or fresh entry)
            warn = ("<div style='font-size:10px;color:#dc2626;font-weight:700'>"
                    "⚠️ live px below trend — exits next bar</div>")
        rows.append(
            f"<tr style='{row_style}'>"
            f"<td style='padding:8px 10px'>{_pill(a['action'], ac)}{warn}</td>"
            f"<td style='font-weight:700'>{a['emoji']} {a['key']}{_kind_badge(a['kind'])}"
            f"<div style='font-size:11px;color:#64748b;font-weight:400'>{a['name']}</div></td>"
            f"<td style='font-size:12px;color:{_TONE_COL.get(a['tone'], C_FLAT)}'>{a['decision']}"
            f"<div style='font-size:10px;color:#94a3b8'>{sub}</div></td>"
            f"<td style='text-align:center;font-size:12px;min-width:56px'>{prio_cell}</td>"
            f"<td style='text-align:right;font-variant-numeric:tabular-nums'>${a['last_close']:,.2f}</td>"
            f"<td style='text-align:right;font-weight:600;font-variant-numeric:tabular-nums;color:{chg_col}'>{chg_s}</td>"
            f"<td style='text-align:right;color:{pnl_col};font-weight:600'>{pnl}{cb_sub}</td>"
            f"<td style='text-align:right;font-weight:700'>{tgt_s}{bar}</td>"
            f"<td style='text-align:right;font-weight:700;font-variant-numeric:tabular-nums'>{amt_s}</td>"
            f"<td style='text-align:right;font-weight:700;color:{_live_col}'>{tgt_live_s}{bar_live}</td>"
            f"<td style='text-align:right;font-weight:700;font-variant-numeric:tabular-nums;color:{_live_col}'>{amt_live_s}</td></tr>")
    # SATA row — the idle-cash park absorbing whatever risk assets can't hold
    si = gate["sata_info"]; sata_pct = gate["sata"]; sata_live = gate_live["sata"]

    def _sata_bar(t):
        return (f"<div style='height:7px;background:#e2e8f0;border-radius:4px;overflow:hidden;"
                f"margin-top:3px'><div style='height:7px;width:{min(t*100,100):.0f}%;"
                f"background:#334155'></div></div>" if t > 0.0005 else "")
    sbar = _sata_bar(sata_pct); sbar_live = _sata_bar(sata_live)
    _sata_col = (C_BUY if sata_live > sata_pct else C_EXIT) if abs(sata_live - sata_pct) > 0.005 else "inherit"
    # live SATA quote — price, day-change, and P&L vs the $100 par cost basis
    _sa_px = _sata.get("price"); _sa_dc = _sata.get("dchg"); _sa_pnl = _sata.get("upnl")
    sa_px_s = f"${_sa_px:,.2f}" if _sa_px else f"${si['par']:,.0f}"
    if _sa_dc is None:
        sa_dc_s, sa_dc_col = "—", "#94a3b8"
    else:
        sa_dc_s, sa_dc_col = f"{_sa_dc:+.2f}%", (C_BUY if _sa_dc >= 0 else C_EXIT)
    if _sa_pnl is None:
        sa_pnl_s, sa_pnl_col = "—", "#94a3b8"
    else:
        sa_pnl_s, sa_pnl_col = f"{_sa_pnl:+.2f}%", (C_BUY if _sa_pnl >= 0 else C_EXIT)
    sa_pnl_sub = "<div style='font-size:10px;color:#94a3b8'>@ $100.00 par</div>"
    rows.append(
        f"<tr style='border-top:2px solid #cbd5e1;background:#f8fafc'>"
        f"<td style='padding:8px 10px'>{_pill('PARK', '#334155')}</td>"
        f"<td style='font-weight:700'>💵 SATA"
        f"<div style='font-size:11px;color:#64748b;font-weight:400'>{si['name']}</div></td>"
        f"<td style='font-size:12px;color:#334155'>Idle cash → SATA preferred"
        f"<div style='font-size:10px;color:#94a3b8'>~{si['annual_rate']*100:.0f}% daily-dividend yield · $100 par · +{si['annual_rate']*100:.0f}%/yr coupon</div></td>"
        f"<td style='text-align:center;color:#cbd5e1'>—</td>"
        f"<td style='text-align:right;font-variant-numeric:tabular-nums'>{sa_px_s}</td>"
        f"<td style='text-align:right;font-weight:600;font-variant-numeric:tabular-nums;color:{sa_dc_col}'>{sa_dc_s}</td>"
        f"<td style='text-align:right;font-weight:600;color:{sa_pnl_col}'>{sa_pnl_s}{sa_pnl_sub}</td>"
        f"<td style='text-align:right;font-weight:800'>{sata_pct*100:.1f}%{sbar}</td>"
        f"<td style='text-align:right;font-weight:800;font-variant-numeric:tabular-nums'>"
        f"${sata_pct*portfolio_value:,.0f}</td>"
        f"<td style='text-align:right;font-weight:800;color:{_sata_col}'>{sata_live*100:.1f}%{sbar_live}</td>"
        f"<td style='text-align:right;font-weight:800;font-variant-numeric:tabular-nums;color:{_sata_col}'>"
        f"${sata_live*portfolio_value:,.0f}</td></tr>")
    st.markdown(f"<table style='width:100%;border-collapse:collapse'>{hdr}{''.join(rows)}</table>",
                unsafe_allow_html=True)
    if gate["n_active"] == 0:
        st.warning("**No open positions today** — no instrument is signalling long, "
                   "so the entire book is parked in **SATA** earning its idle-cash "
                   "yield until a signal fires.")

    closes = [a for a in gate["actions"] if a["action"] == "CLOSE"]
    opens = [a for a in gate["actions"] if a["action"] == "OPEN"]
    cc = st.columns(2)
    with cc[0]:
        if closes:
            st.error("**Close now:** " + ", ".join(
                f"{a['key']} ({_pct(a['upnl'])})" for a in closes))
        else:
            st.success("**No exits triggered today** — nothing to close.")
    with cc[1]:
        if opens:
            st.success("**Open now:** " + ", ".join(
                f"{a['key']} → {a['target']*100:.0f}%" for a in opens))
        else:
            st.info("**No fresh entries today** — no flat instrument is signalling a buy.")

    st.markdown("---")

    # ── 3. PER-SIGNAL LIVE CARDS (grouped by parent) ────────────────────
    st.markdown("### 🛰️ Live signal & positions — by app")
    st.caption("Each app fires one signal; its 1× primary and higher-beta / "
               "leveraged siblings are all traded off it. Green = long/buy, "
               "red = exit, amber = watch, grey = stand aside.")
    for pk, grp in parents:
        head = grp[0]
        dec = head["decision"]; col = _TONE_COL.get(dec["tone"], C_FLAT)
        sent = head["sentiment"]
        sent_s = f"{sent:.0f}/100" if sent == sent else "—"
        eng = ("CT-Divergence" if head["mode"] == "ct-divergence"
               else head.get("engine_label")
               or (f"MA{head['ma_window']}" if head["mode"] == "ma" else "Divergence"))
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:10px;margin:10px 0 4px 0'>"
            f"<span style='font-size:16px;font-weight:800'>{head['emoji']} {pk}</span>"
            f"{_pill(dec['ico'] + ' ' + dec['label'], col)}"
            f"<span style='font-size:11px;color:#94a3b8'>{eng} · "
            f"{'🐂 bull' if head['bull_regime'] else '🐻 bear'} regime · sentiment {sent_s}</span></div>",
            unsafe_allow_html=True)
        cards = st.columns(max(3, len(grp)))
        for i, res in enumerate(grp):
            pos = res["pos"]
            with cards[i % max(3, len(grp))]:
                body = [
                    f"<div style='border:1px solid #e2e8f0;border-left:5px solid {res['accent']};"
                    f"border-radius:9px;padding:9px 11px;margin-bottom:8px;background:#fff'>",
                    f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                    f"<span style='font-weight:800;font-size:14px'>{res['key']}"
                    f"{_kind_badge(res['kind'])}</span>"
                    f"<span style='font-size:12px;color:#64748b'>${res['last_close']:,.2f} "
                    f"<b style='color:{C_BUY if res['dchg']>=0 else C_EXIT}'>{res['dchg']:+.1f}%</b></span></div>",
                    f"<div style='font-size:10px;color:#94a3b8;margin-bottom:4px'>{res['name']}</div>",
                ]
                if pos["in_pos"] and pos["entry_px"]:
                    pcol = C_BUY if (pos["upnl"] or 0) >= 0 else C_EXIT
                    # Per-asset exit: leveraged / no-stop sleeves (e.g. SOXL 3× and
                    # WGMI) trade the SAME signal with NO fixed stop, so there is no
                    # stop price to show — say "signal exit · no fixed stop" instead
                    # of rendering a misleading "$0.00" (or crashing on a None
                    # stop_px, which the old unconditional format did).
                    _stop = res.get("stop") or 1.0
                    if _stop < 0.999 and pos.get("stop_px"):
                        exit_txt = (f"stop ${pos['stop_px']:,.2f} "
                                    f"<span style='color:#94a3b8'>({_pct(pos.get('dist_stop'))})</span>")
                    else:
                        exit_txt = "<span style='color:#94a3b8'>signal exit · no fixed stop</span>"
                    body.append(
                        f"<div style='font-size:11.5px;line-height:1.5'>"
                        f"📍 <b>LONG</b> {pd.Timestamp(pos['entry_date']).strftime('%b %d')} "
                        f"@ ${float(pos['entry_px']):,.2f} · {pos['days']}d<br>"
                        f"P&amp;L <b style='color:{pcol}'>{_pct(pos['upnl'])}</b> · "
                        f"{exit_txt}</div>")
                elif res["last_trade"]:
                    lt = res["last_trade"]; r_ = lt["ret"] * 100
                    rc = C_BUY if r_ >= 0 else C_EXIT
                    body.append(
                        f"<div style='font-size:11.5px;line-height:1.45;color:#475569'>"
                        f"⚪ FLAT · last {r_:+.1f}% "
                        f"<span style='color:{rc}'></span>"
                        f"({pd.Timestamp(lt['exit_date']).strftime('%b %d')}, {lt.get('reason','—')})</div>")
                else:
                    body.append("<div style='font-size:11.5px;color:#475569'>⚪ FLAT · no position</div>")
                body.append("</div>")
                st.markdown("".join(body), unsafe_allow_html=True)

    st.info("Unified **daily** reads. For the canonical hourly Pure-Regime view of "
            "BTC (BTC/MSTR/MSTU) or Gold (GDX/UGL), open the **₿ Bitcoin** or "
            "**🥇 Gold** app in the sidebar.")


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — COMBINED BACKTESTING
# ══════════════════════════════════════════════════════════════════════════
with tab_bt:
    o = opt["optimal"]
    _oos_lo, _oos_hi = ov.oos_start_span()
    _oos_span = _oos_lo if _oos_lo == _oos_hi else f"{_oos_lo}–{_oos_hi}"
    st.markdown("## 📊 The optimal combined strategy")
    st.caption("Each instrument's signal-driven strategy produces a daily return "
               "stream (long when its parent signal is on, otherwise **idle "
               "capital earns the SATA yield ~13%/yr**). We search long-only "
               f"blends of all {N_ALL} — leveraged sleeves capped tighter — for the mix "
               "that **maximises return while keeping the drawdown shallow** "
               "(highest raw return among near-max-Sharpe blends). Out-of-sample "
               f"from {_oos_span} depending on the instrument (staggered starts — "
               "newer sleeves like BTC begin ~2024); $100k start.")
    if opt.get("fundamental") and "optimal_pretilt" in opt:
        _pt = opt["optimal_pretilt"]
        st.info(f"🔭 **Fundamental overlay applied** — these figures tilt the quant "
                f"optimum toward the mid-2026 sector view (toggle on the Live tab). "
                f"This profile: **{o['total_ret']*100:,.0f}%** return / "
                f"Sharpe **{o['sharpe']:.2f}** *with* the overlay vs "
                f"**{_pt['total_ret']*100:,.0f}%** / **{_pt['sharpe']:.2f}** without. "
                f"{ov.FUNDAMENTAL_VIEW_NOTE}")

    # ── risk-profile trade-off: leaning on β / 2× proxies ───────────────
    st.markdown(f"**Risk profile: `{_profile}`** — "
                f"{ov.RISK_PROFILES[_profile]['blurb']} "
                f"_(change it in the sidebar.)_")
    comp = get_profile_comparison(_bucket(), _use_fund)
    if comp:
        ch = ("<tr style='background:#f1f5f9;font-size:12px;text-align:left'>"
              "<th style='padding:6px 10px'>Profile</th>"
              "<th style='text-align:right'>Return</th><th style='text-align:right'>CAGR</th>"
              "<th style='text-align:right'>Max DD</th><th style='text-align:right'>Sharpe</th>"
              "<th style='text-align:right'>β + 2× weight</th></tr>")
        crows = []
        for row in comp:
            hi = "background:#eff6ff;" if row["name"] == _profile else ""
            crows.append(
                f"<tr style='border-bottom:1px solid #eef2f7;{hi}'>"
                f"<td style='padding:6px 10px;font-weight:700'>{row['name']}"
                f"{' ◄ active' if row['name']==_profile else ''}</td>"
                f"<td style='text-align:right;font-weight:600'>{row['total_ret']*100:,.0f}%</td>"
                f"<td style='text-align:right'>{row['cagr']*100:.0f}%</td>"
                f"<td style='text-align:right;color:{C_EXIT}'>{row['mdd']*100:.0f}%</td>"
                f"<td style='text-align:right;font-weight:600'>{row['sharpe']:.2f}</td>"
                f"<td style='text-align:right'>{row['betalev']*100:.0f}%</td></tr>")
        st.markdown(f"<table style='width:100%;border-collapse:collapse'>{ch}{''.join(crows)}</table>",
                    unsafe_allow_html=True)
        st.caption("Loading the high-beta / leveraged proxies (β + 2× weight) "
                   "**boosts return but lowers Sharpe** — the drawdown deepens "
                   "faster than the return. Growth roughly doubles the return of "
                   "Balanced for a still-respectable Sharpe; Aggressive pushes "
                   "return highest at the deepest drawdown.")
    st.markdown("")

    m = st.columns(4)
    m[0].metric("Optimal blend — total return", f"{o['total_ret']*100:,.0f}%",
                delta=f"CAGR {o['cagr']*100:.0f}%")
    m[1].metric("Max drawdown", f"{o['mdd']*100:.0f}%",
                delta=f"vs {bm['bh_equal']['mdd']*100:.0f}% buy&hold", delta_color="inverse")
    m[2].metric("Sharpe", f"{o['sharpe']:.2f}",
                delta=f"vs {bm['bh_equal']['sharpe']:.2f} buy&hold")
    m[3].metric("Volatility (ann.)", f"{o['vol']*100:.0f}%")

    st.markdown("#### Optimal weights")
    wc = st.columns([2, 1])
    with wc[0]:
        wk = [(c, o["weights"][c]) for c in opt["cols"] if o["weights"][c] > 0.002]
        wk.sort(key=lambda x: -x[1])
        fig = go.Figure(go.Bar(
            x=[w * 100 for _, w in wk],
            y=[f"{c}{'  2×' if by_key[c]['kind']=='lev' else '  β' if by_key[c]['kind']=='beta' else ''}"
               for c, _ in wk],
            orientation="h", marker=dict(color=[by_key[c]["accent"] for c, _ in wk]),
            text=[f"{w*100:.1f}%" for _, w in wk], textposition="outside"))
        fig.update_layout(height=360, margin=dict(t=10, b=10, l=10, r=30),
                          xaxis_title="allocation %", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
    with wc[1]:
        st.markdown("**Allocation scheme**")
        cmp_rows = []
        for nm, key in [("Optimal", "optimal"), ("Risk parity", "risk_parity"),
                        ("Equal weight", "equal")]:
            d = opt[key]
            cmp_rows.append(f"<tr><td style='padding:4px 6px'>{nm}</td>"
                            f"<td style='text-align:right'>{d['total_ret']*100:,.0f}%</td>"
                            f"<td style='text-align:right'>{d['mdd']*100:.0f}%</td>"
                            f"<td style='text-align:right'>{d['sharpe']:.2f}</td></tr>")
        st.markdown(
            "<table style='width:100%;font-size:13px;border-collapse:collapse'>"
            "<tr style='background:#f1f5f9'><th style='text-align:left;padding:4px 6px'>Scheme</th>"
            "<th style='text-align:right'>Ret</th><th style='text-align:right'>MDD</th>"
            "<th style='text-align:right'>Sharpe</th></tr>" + "".join(cmp_rows) + "</table>",
            unsafe_allow_html=True)
        st.caption("Leveraged sleeves capped at 10%, high-beta at 18%, core at "
                   "30% — so the optimiser only leans on the 2× / β names when "
                   "they improve risk-adjusted return.")

    st.markdown("#### Growth of $100k — combined vs benchmarks")
    fig = go.Figure()
    styles = {"Optimal blend": ("#111827", 3),
              "Equal-weight strategies": ("#0ea5e9", 1.7),
              "Equal-weight Buy & Hold": ("#94a3b8", 1.7)}
    for name, curve in _PF["curves"].items():
        cc2, wdt = styles.get(name, ("#888", 1.5))
        fig.add_trace(go.Scatter(x=curve.index, y=curve.to_numpy() * 100000,
                                 name=name, line=dict(color=cc2, width=wdt)))
    fig.update_layout(height=440, margin=dict(t=20, b=10, l=10, r=10),
                      yaxis_title="Portfolio value ($)", yaxis_type="log",
                      legend=dict(orientation="h", y=1.08), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    opt_curve = _PF["curves"]["Optimal blend"]
    dd = opt_curve / opt_curve.cummax() - 1
    figd = go.Figure(go.Scatter(x=dd.index, y=dd.to_numpy() * 100, fill="tozeroy",
                                line=dict(color=C_EXIT, width=1)))
    figd.update_layout(height=230, margin=dict(t=10, b=10, l=10, r=10),
                       yaxis_title="drawdown %",
                       title=dict(text="Optimal blend — drawdown", font_size=13))
    st.plotly_chart(figd, use_container_width=True)

    st.markdown("#### Performance by market regime")
    ph = ("<tr style='background:#f1f5f9'><th style='text-align:left;padding:6px 10px'>Window</th>"
          "<th style='text-align:right'>Return</th><th style='text-align:right'>CAGR</th>"
          "<th style='text-align:right'>Max DD</th><th style='text-align:right'>Sharpe</th></tr>")
    pr = []
    for row in _PF["per"]:
        pr.append(f"<tr style='border-bottom:1px solid #eef2f7'>"
                  f"<td style='padding:6px 10px'>{row['label']}</td>"
                  f"<td style='text-align:right;font-weight:600'>{row['total_ret']*100:,.0f}%</td>"
                  f"<td style='text-align:right'>{row['cagr']*100:.0f}%</td>"
                  f"<td style='text-align:right;color:{C_EXIT}'>{row['mdd']*100:.0f}%</td>"
                  f"<td style='text-align:right;font-weight:600'>{row['sharpe']:.2f}</td></tr>")
    st.markdown(f"<table style='width:100%;border-collapse:collapse'>{ph}{''.join(pr)}</table>",
                unsafe_allow_html=True)

    st.markdown("#### Per-instrument strategy (standalone, out-of-sample)")
    st.caption("Each instrument's signal-driven strategy vs buy-&-hold, and its "
               "weight in the optimal blend. Grouped by signal — β = high-beta "
               "sibling, 2× = leveraged. **Siblings (↳) are traded off their "
               "parent's signal**, not their own: MSTR/MSTU enter and exit on "
               "BTC's divergence signal, GDX/UGL on gold's, OIH on XLE's — the "
               "higher-beta name executes on its own price but is steered by the "
               "cleaner parent read. Every asset here runs its **own app's actual "
               "engine** — BTC/MSTR/MSTU via the BTC app's trained CT model, "
               "GLDM/GDX/UGL via the Gold app's Divergence Pure-Regime, the ETFs "
               "via their tuned configs — so these numbers match each source app. "
               "(BTC's CT features begin ~2024, so its sleeve covers a shorter "
               "window than the earliest-start ETFs.)")
    ah = ("<tr style='background:#f1f5f9'><th style='text-align:left;padding:6px 10px'>Instrument</th>"
          "<th>Signal / engine</th><th style='text-align:right'>Strat</th>"
          "<th style='text-align:right'>Buy&amp;Hold</th><th style='text-align:right'>Max DD</th>"
          "<th style='text-align:right'>Sharpe</th><th style='text-align:right'>Win%</th>"
          "<th style='text-align:right'>Opt. wt</th></tr>")
    ar = []
    for pk, grp in parents:
        for res in grp:
            mm = res["metrics"]; bb = res["bh_metrics"]
            beat = mm["total_ret"] >= bb["total_ret"]
            base_eng = ("CT-Divergence" if res["mode"] == "ct-divergence"
                        else res.get("engine_label")
                        or (f"MA{res['ma_window']}" if res["mode"] == "ma" else "Divergence"))
            if res["key"] == res["parent"]:          # this app's own signal instrument
                eng = f"{base_eng} signal"
            else:                                     # sibling traded off the parent signal
                eng = f"↳ off {res['parent']} {base_eng}"
            ar.append(f"<tr style='border-bottom:1px solid #eef2f7'>"
                      f"<td style='padding:6px 10px;font-weight:700'>{res['key']}"
                      f"{_kind_badge(res['kind'])}"
                      f"<span style='font-size:11px;color:#94a3b8;font-weight:400'> {res['name']}</span></td>"
                      f"<td style='font-size:12px;color:#64748b'>{eng}</td>"
                      f"<td style='text-align:right;font-weight:600;color:{C_BUY if beat else '#111'}'>"
                      f"{mm['total_ret']*100:,.0f}%</td>"
                      f"<td style='text-align:right;color:#64748b'>{bb['total_ret']*100:,.0f}%</td>"
                      f"<td style='text-align:right;color:{C_EXIT}'>{mm['mdd']*100:.0f}%</td>"
                      f"<td style='text-align:right'>{mm['sharpe']:.2f}</td>"
                      f"<td style='text-align:right'>{res['win_rate']:.0f}%</td>"
                      f"<td style='text-align:right;font-weight:700'>"
                      f"{o['weights'].get(res['key'],0)*100:.1f}%</td></tr>")
    st.markdown(f"<table style='width:100%;border-collapse:collapse'>{ah}{''.join(ar)}</table>",
                unsafe_allow_html=True)

    st.success(
        f"**Bottom line.** Blending signal-driven, cash-when-out strategies across "
        f"{N_ALL} instruments — including the higher-beta MSTR/MSTU, GDX/UGL and OIH "
        f"sleeves, used only when they earn their capped slots — the optimal blend "
        f"returned **{o['total_ret']*100:,.0f}%** at just **{o['mdd']*100:.0f}%** "
        f"max drawdown (Sharpe **{o['sharpe']:.2f}**), versus an equal-weight "
        f"buy-&-hold of the same instruments at **{bm['bh_equal']['total_ret']*100:,.0f}%** "
        f"but a punishing **{bm['bh_equal']['mdd']*100:.0f}%** drawdown "
        f"(Sharpe **{bm['bh_equal']['sharpe']:.2f}**).")
    st.caption("⚠️ Optimal weights are fit on this same history (in-sample) — the "
               "best *historical* blend, not a guarantee. Strategy curves park "
               "idle capital in SATA (~13%/yr, an assumed-constant series); the "
               "buy-&-hold benchmark is always fully invested with no SATA. "
               "Equal-weight and risk-parity need no fitting and are shown for "
               "comparison. Not investment advice.")


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — METHODOLOGY
# ══════════════════════════════════════════════════════════════════════════
with tab_explain:
    st.markdown("## 🧠 How Overall Trading works")
    st.markdown(
        """
**The idea.** Every other app trades one signal. *Overall Trading* runs all of
them — each app's 1× primary **plus its higher-beta / leveraged siblings** —
through **one unified daily engine**, so their signals, positions and back-tests
sit side-by-side and blend into a single portfolio.

**The universe.** Each sibling is traded off its **parent's**
signal (never its own), exactly as the dedicated apps do:

| App / signal | Traded instruments | Engine |
|---|---|---|
| ₿ **BTC** | BTC · MSTR (β) · MSTU (2×) | BTC app's CT-model Divergence Pure-Regime |
| 🥇 **Gold (GLDM)** | GLDM · GDX (β) · UGL (2×) | Gold app's Divergence Pure-Regime, −3% |
| 🛢️ **XLE** | XLE · OIH (β) | Divergence Pure-Regime, −8% |
| 🧲 **REMX** | REMX | Divergence Pure-Regime, −8% |
| 🖥️ **SOXX** | SOXX | MA40, −5% |
| ⚡ **GRID** | GRID | MA150, −5% |
| ⛏️ **WGMI** | WGMI (β) | MA30, −10% |

Every asset runs the **exact engine its own app trades**, so the Overall app's
signals, positions and back-tests match each source app:

- **BTC / MSTR / MSTU** run the **BTC app's own trained CT model**
  (`inference_assets_ct.joblib`, 116 features incl. Bitcoin on-chain + Coinbase
  premium) with the app's live Pure-Regime gate (U1>+1.3% + ≥2 high-breaks,
  regime-adaptive D2/D3 exit, MA30 gate, per-asset stops, SL re-entry). This
  reproduces the BTC app's headline **BTC +102% / MSTR +213% / MSTU +504%**. The
  CT feature data begins ~2023-11, so the BTC sleeve covers ~2024→now (the
  combined engine handles the staggered start).
- **GLDM / GDX / UGL** run the **Gold app's `backtest_gldm`** Divergence
  Pure-Regime with its per-asset regime windows (GLDM 50 / UGL 40 / GDX 100) and
  −3% stops — reproducing the Gold app's **GDX +272% / UGL +211%**.
- **SOXX / GRID / XLE / REMX / WGMI** reuse their **exact `ticker_config`**
  entries through the same `backtest_ticker` engine their apps use (SOXX 25/100
  dual-MA, GRID MACD 10/20/9, WGMI 50-day SMA + vol-filter, REMX &
  XLE divergence Pure-Regime). These match their apps bar-for-bar.

**Live signals & positions.** For each app we fetch data, fit the H/L band model
out-of-sample, replay the strategy bar-by-bar, and read off the current alert
level, whether we're long, entry price/date, unrealised P&L, stop (or a
signal-only exit for the no-stop sleeves — the 3× **SOXL** and **WGMI** carry no
fixed stop, since a 1× stop whipsaws a leveraged/high-beta name) and days held —
for the primary **and** each sibling (which shares the parent's entry/exit timing
but has its own fill price and P&L). That drives the **action plan** and the
**allocation donuts**.

**The optimal allocation.** Each strategy is long when its signal is on and
otherwise parks idle capital in **SATA** (see below), producing a daily return
stream. We Monte-Carlo long-only blends (sum = 100%) with **per-instrument caps**
— 30% core, 18% high-beta, 10% leveraged — and pick the **highest-return blend
among the near-max-Sharpe set**, so returns are maximised while drawdown stays
shallow. The tight caps mean the 2× / β sleeves only get weight when they
genuinely improve the risk-adjusted result.

**Entry priority.** When several instruments signal entry at once — or one fires
while others are already held — a **priority score (0–1)** decides which get
funded and how much. It blends four current-conditions reads, each normalised
across the competing candidates: **momentum** (0.28, distance above the 50-day
SMA), **macro sentiment** (0.24, the app's 0–100 gauge), the strategy's
**back-tested win-rate** (0.20), its **risk-adjusted edge** (0.18, OOS Sharpe)
and **regime** (0.10, bull vs bear). Each instrument's target = its optimal
weight × (0.5 + priority), water-filled to the caps — so the highest-priority
signals in today's tape get the largest slices.

**Today's book & SATA.** Capital is deployed only to instruments currently
signalling long (or firing a fresh entry); the held/opened **risk assets total
100%** when the caps allow, and **whatever can't be deployed is parked in SATA**
— *Strive's Variable-Rate Series A Perpetual Preferred*, a US-listed security
paying a ~13% annual coupon as a daily dividend on $100 par (≈13.88% effective
reinvested). With **no open positions the entire book sits in SATA**, earning
that yield until a signal fires. Idle capital earns SATA in the back-test too, so
the combined curve reflects cash working rather than sitting dead.

**Honest caveats.**
- Optimal weights are in-sample — the best *historical* blend, not a promise.
- SATA is modelled (per the BTC app's framing) as always having existed, flat at
  $100 par, paying its ~13% daily dividend across every period — an assumption,
  not a market-tested series.
- Leveraged 2× sleeves (MSTU, UGL) and high-beta names compound decay and gap
  risk; the caps bound but don't remove that.
- This is a **daily** engine. The BTC and Gold apps' canonical **hourly**
  Pure-Regime signals live in those apps — open them from the sidebar.
- Nothing here is investment advice.
        """)


st.caption(f"Overall Trading · unified daily engine · {len(results)} instruments · "
           f"data to {as_of.strftime('%Y-%m-%d')} · not investment advice.")
