"""Overall Trading — the combined cross-asset decision cockpit.

One screen that fuses the live signals, positions and back-tests of every other
app into a single portfolio view built for one question: *where do I put money to
work today?*  Each signal app trades its 1× primary plus higher-beta / leveraged
siblings (BTC→MSTR/MSTU/ETH, Gold→GDX/UGL, XLE→OIH), all steered off the parent
signal, so the combined book spans every instrument across all apps.

  🔴 Live — Decision Cockpit   what to CLOSE / OPEN / HOLD today, the optimal %
                                allocation, and the strategy's current book.
  🕰️ Historical View            the same overall view reconstructed for any
                                calendar date — the book, positions and open/
                                close events as they stood at that day's close.
  📊 Combined Backtesting       the walk-forward replay of the daily gated,
                                priority-tilted book (look-ahead-free) vs the
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
import freshness as fr
import inspect as _inspect

if not hasattr(fr, "audit_universe"):          # self-heal a stale module cache
    fr = importlib.reload(fr)


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
        if not hasattr(mod, "live_entry_keys"):
            return True
        if not getattr(mod, "LIVE_ENTRY_DIVERGENCE_AWARE", False):
            return True
        if not hasattr(mod, "adjust_for_selection"):
            return True
        if not hasattr(mod, "slice_metrics"):
            return True
        if not hasattr(mod, "per_asset_slice_metrics"):
            return True
        if not hasattr(mod, "overall_trade_stats"):
            return True
        if not hasattr(mod, "snapshot_asof"):
            return True
        if not hasattr(mod, "walkforward_gated_replay"):
            return True
        if not hasattr(mod, "pnl_attribution_replay"):
            return True
        if not hasattr(mod, "data_vintage"):
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
_ALL_APPS = (["OVERALL", "BTC", "GLDM", "GDXM"] + ticker_config.APP_KEYS
             + ["DAILYAUDIT", "HEALTH", "TARGETBOOK", "EXECUTEDBOOK"])
_APP_LABELS = {
    "OVERALL": "🧭  Overall Trading",
    "BTC": "₿  Bitcoin (BTC)",
    "GLDM": "🥇  Gold Trend (GLDM·UGL)", "GDXM": "⛏️  Gold Miners (GDX·NUGT)",
    "DAILYAUDIT": "🕵️  Daily Audit",
    "HEALTH": "🩺  Strategy Health",
    "TARGETBOOK": "📋  Target Book (IBKR)",
    "EXECUTEDBOOK": "✅  Executed Book (IBKR)",
}
for _k, _c in ticker_config.CONFIGS.items():
    _APP_LABELS[_k] = f"{_c.emoji}  {_c.key} · {_c.name.split('(')[0].strip()[:22]}"

st.set_page_config(page_title="Overall Trading", page_icon="🧭",
                   layout="wide", initial_sidebar_state="expanded")

# Trim the wide-layout gutter: Streamlit's default side padding leaves a wide
# dead strip between the sidebar and the content on large screens, while wide
# tables (e.g. the action-plan table) still had no room to breathe on the
# right — hence the overflow-x:auto wrapper used on every raw <table> below.
st.markdown(
    "<style>"
    "[data-testid='stMainBlockContainer'], .main .block-container, .block-container"
    "{padding-left:2rem;padding-right:2rem;max-width:100%;}"
    # st.metric clips long labels/deltas to one line with an ellipsis — in a
    # 6-column row that hides e.g. the winning-days confidence interval.
    # Let them wrap instead.
    "[data-testid='stMetricLabel'] p, [data-testid='stMetricDelta'],"
    "[data-testid='stMetricDelta'] div"
    "{white-space:normal !important;overflow:visible !important;"
    "text-overflow:clip !important;line-height:1.25;}"
    "</style>",
    unsafe_allow_html=True)

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
# Risk profiles offered in the UI — Aggressive is retired from the interface
# (−38% drawdown budget: too deep to publish); its definition stays in core
# for the CLI tools. A stale session pointing at it falls back to the default.
UI_PROFILES = [p for p in ov.RISK_PROFILES if p != "Aggressive"]
if st.session_state.get("overall_risk_profile") not in UI_PROFILES:
    st.session_state["overall_risk_profile"] = ov.DEFAULT_PROFILE
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
    # computed_at is cached alongside the results, so it records when the
    # signals were actually (re)generated — not when the page last rendered.
    return dict(results=ov.run_universe(), computed_at=pd.Timestamp.utcnow())


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


# display name of the combined back-test curve — the walk-forward gated replay
# (the daily gate/tilt book, look-ahead-free anchors), NOT a fixed-weight blend.
STRAT_CURVE = "Overall strategy (gated replay)"

# display name of the as-published record — the target books actually committed
# each day (data/overall/book_archive/), daily optimiser trims included.
STRAT_CURVE_ACTUAL = "Overall strategy (as published)"


@st.cache_data(ttl=900, show_spinner=False)
def get_published_books(bucket: str):
    """The archived as-published target books (one per signal day), re-read on
    the shared refresh bucket so a fresh publish shows up without a restart."""
    return ov.load_published_books()


@st.cache_data(ttl=900, show_spinner=False)
def get_book_version_map(bucket: str):
    """Backfilled strategy-version provenance for pre-stamp archived books."""
    return ov.load_book_version_map()


@st.cache_data(ttl=1800, show_spinner="Optimising the combined allocation…")
def get_all_profiles(bucket: str):
    """Compute the full portfolio for every UI risk profile once, so switching
    profiles (and rendering the comparison table) is instant — no recompute.

    ``opt``/``gate`` power TODAY'S live book (full-history optimal weights —
    all data to date is legitimately known when sizing today; the mid-2026
    fundamental overlay is retired everywhere, being a hindsight-formed view).
    Every BACK-TEST number (``wf``/``per``/``curves[STRAT_CURVE]``) instead
    comes from the walk-forward gated replay: daily gate/tilt/water-fill with
    expanding-window anchor refits and as-of priority inputs — no look-ahead
    anywhere in the history."""
    res = get_results(bucket)
    results, computed_at = res["results"], res["computed_at"]
    if not results:
        return None
    rets = ov.returns_matrix(results)
    pos = ov.position_matrix(results, rets.index)
    sata = ov.SATA_DAILY
    bm = ov.benchmarks(rets, results, pos=pos, sata_daily=sata)      # profile-independent
    base_curves = {"Equal-weight strategies": bm["strat_equal"]["equity"],
                   "Equal-weight Buy & Hold": bm["bh_equal"]["equity"]}
    profiles = {}
    for name in UI_PROFILES:
        prof = ov.RISK_PROFILES[name]
        caps = ov.caps_for(name)
        opt = ov.optimize_weights(rets, caps=caps, pos=pos, sata_daily=sata,
                                  mdd_floor=prof["mdd_floor"], objective=prof["objective"],
                                  fundamental=False)
        w_opt = np.array([opt["optimal"]["weights"][c] for c in opt["cols"]])
        wf = ov.walkforward_gated_replay(results, caps=caps,
                                         mdd_floor=prof["mdd_floor"],
                                         objective=prof["objective"],
                                         sata_daily=sata, tilt=True)
        per = ov.period_metrics_from_ret(wf["ret"], ov.COMBINED_PERIODS)
        curves = {STRAT_CURVE: wf["equity"], **base_curves}
        gate = ov.signal_gated_allocation(results, opt["optimal"]["weights"], caps=caps)
        profiles[name] = dict(opt=opt, per=per, curves=curves, gate=gate,
                              w_opt=w_opt, wf=wf)
    return dict(results=results, computed_at=computed_at, rets=rets, bm=bm,
                profiles=profiles)


def get_portfolio(bucket: str, profile: str):
    allp = get_all_profiles(bucket)
    if not allp:
        return None
    p = allp["profiles"][profile]
    return dict(results=allp["results"], computed_at=allp["computed_at"],
                rets=allp["rets"], bm=allp["bm"], profile=profile, **p)


def get_profile_comparison(bucket: str):
    """Headline metrics for every UI risk profile (reads the shared computation)."""
    allp = get_all_profiles(bucket)
    if not allp:
        return []
    rows = []
    for name in UI_PROFILES:
        prof = ov.RISK_PROFILES[name]
        o = allp["profiles"][name]["opt"]["optimal"]     # live anchors → β/2× posture
        wm = allp["profiles"][name]["wf"]["metrics"]     # back-test = gated replay
        bl = sum(v for k, v in o["weights"].items()
                 if ov.ASSET_META.get(k, {}).get("kind") in ("beta", "lev"))
        rows.append(dict(name=name, blurb=prof["blurb"], betalev=bl, **{
            k: wm[k] for k in ("total_ret", "cagr", "mdd", "sharpe")}))
    return rows


st.title("🧭 Overall Trading — Combined Decision Cockpit")
import strategy_version as _sv                 # noqa: E402
if not (hasattr(_sv, "render_badge") and hasattr(_sv, "BADGE_COLOR")):
    import importlib                           # stale hot-loaded module (the
    _sv = importlib.reload(_sv)                # server kept an older import)
_sv.render_badge()                             # visible above every tab
# 📦 data-vintage stamp — every back-test figure on this page is a pure
# function of (strategy version, these dataset hashes).  If a number changed
# since your last visit, this line changed with it: same stamp ⇒ same numbers.
try:
    _dv = ov.data_vintage()
    st.caption(f"📦 **Data vintage** — BTC features `{_dv['btc_sha']}` "
               f"(through {_dv['btc_to']}) · equity/gold snapshots "
               f"`{_dv['sleeves_sha']}` ({_dv['n_sleeves']} pinned datasets). "
               "Back-tests change only when this stamp changes "
               "(🕵️ Daily Audit → Dataset audit trail has the full detail).")
except Exception:
    pass
st.caption(f"Every asset app, fused into one portfolio spanning {N_ALL} instruments "
           "(each app's 1× primary plus its higher-beta / leveraged siblings). "
           "Live entry/exit signals, current positions, the historically-optimal "
           "cross-asset allocation, and one combined back-test — built around a "
           "single question: **where should capital go today?**")

# 🩺 one-line strategy-health badge — read from the committed nightly snapshot
# (never computes anything); a red flag finds the user on the page they
# actually open daily.  Full monitor: the 🩺 Strategy Health app.
try:
    import json as _json
    _hp = _json.loads((_REPO_ROOT / "data" / "overall"
                       / "strategy_health.json").read_text())
    _hv, _hi = _hp["verdict"], {"green": "🟢", "yellow": "🟡", "red": "🔴"}
    st.caption(f"🩺 Strategy health: {_hi.get(_hv['status'], '⬜')} "
               f"{_hv['headline']} (as-of {_hp['as_of']}) — details in the "
               "**🩺 Strategy Health** app in the sidebar.")
except Exception:
    pass

_profile = st.session_state["overall_risk_profile"]
try:
    _PF = get_portfolio(_bucket(), _profile)
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
# when the signals were last (re)generated — the cached run_universe() time,
# shown in US Central Time (CDT/CST picked automatically by the zone)
signals_updated = pd.Timestamp(_PF.get("computed_at") or pd.Timestamp.utcnow())
if signals_updated.tzinfo is None:
    signals_updated = signals_updated.tz_localize("UTC")
signals_updated = signals_updated.tz_convert("America/Chicago")
# group instruments by parent signal, preserving parent order
parents = []
for pk in ov.PARENT_KEYS:
    grp = [r for r in results if r["parent"] == pk]
    if grp:
        parents.append((pk, grp))


def _alloc_area_fig(weights: pd.DataFrame, sata: pd.Series, title: str):
    """Stacked-area figure of the replayed book's daily % allocation, kept
    deliberately simple to read: ONE band per **signal group** (each app's
    family of instruments — e.g. BTC·MSTR·MSTU·ETH — summed and drawn in the
    parent's accent colour, so every band has a distinct colour), largest
    group first, groups that never matter folded into a single *Other* band,
    and the SATA idle-cash remainder in light grey on top.  Every day stacks
    to 100%; hovering a band shows its per-asset split that day.  Used below
    both Growth charts."""
    fam: dict[str, list[str]] = {}                 # parent signal → member cols
    for c in weights.columns:
        if float(weights[c].max()) > 0.0005:
            fam.setdefault(by_key.get(c, {}).get("parent", c), []).append(c)
    groups = sorted(((p, mem, weights[mem].sum(axis=1)) for p, mem in fam.items()),
                    key=lambda g: -float(g[2].mean()))
    # a group that never exceeds 5% and averages <1% is visual noise — fold it
    small = [g for g in groups if g[2].mean() < 0.01 and g[2].max() < 0.05]
    big = [g for g in groups if g not in small]

    def _split(mem):                               # per-day member breakdown
        arr = weights[mem].to_numpy()
        return [" · ".join(f"{k} {v * 100:.1f}%" for k, v in zip(mem, row)
                           if v > 0.001) or "—" for row in arr]

    fig = go.Figure()
    # two apps can share an accent (GLDM and GDXM are both gold) — give any
    # repeat a fallback colour so no two bands ever look identical
    alt = iter(["#78350f", "#db2777", "#0ea5e9", "#84cc16", "#64748b"])
    used: set = set()
    for p, mem, g in big:
        meta = by_key.get(mem[0], {})
        label = f"{meta.get('emoji', '')} {p}".strip() + \
                (f" +{len(mem) - 1}" if len(mem) > 1 else "")
        color = meta.get("accent", "#888")
        if color in used:
            color = next(alt, "#334155")
        used.add(color)
        fig.add_trace(go.Scatter(
            x=g.index, y=g.to_numpy() * 100, name=label,
            mode="lines", stackgroup="book",
            line=dict(width=0, color=color),
            customdata=_split(mem),
            hovertemplate="%{fullData.name}: <b>%{y:.1f}%</b><br>"
                          "%{customdata}<extra></extra>"))
    if small:
        gs = sum(g[2] for g in small)
        mems = [c for g in small for c in g[1]]
        fig.add_trace(go.Scatter(
            x=gs.index, y=gs.to_numpy() * 100,
            name=f"Other ({len(mems)} small sleeves)",
            mode="lines", stackgroup="book", line=dict(width=0, color="#94a3b8"),
            customdata=_split(mems),
            hovertemplate="Other: <b>%{y:.1f}%</b><br>"
                          "%{customdata}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=sata.index, y=sata.to_numpy() * 100, name="💵 SATA (idle cash)",
        mode="lines", stackgroup="book", line=dict(width=0, color="#cbd5e1"),
        hovertemplate="SATA idle cash: <b>%{y:.1f}%</b><extra></extra>"))
    fig.update_layout(height=380, margin=dict(t=30, b=10, l=10, r=10),
                      yaxis=dict(title="% of book", range=[0, 100.5]),
                      legend=dict(orientation="h", y=-0.14),
                      hovermode="x unified",
                      title=dict(text=title, font_size=13))
    return fig

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

# ── SIGNAL-FRESHNESS AUDIT — validate every app's bar BEFORE trusting it ─────
# Each signal app's newest completed bar is compared against the freshest bar
# its asset class can possibly have right now (BTC: the last 12:00-UTC / 7am-CT
# bar close; equities: the last 4:00 PM ET market close).  If anything is stale
# the cached universe is thrown away ONCE and recomputed from live data — the
# "proper data/signal refresh" step — and only if the recompute is STILL stale
# does the cockpit raise the flashing stale-signals alert below.
_AUDIT = fr.audit_universe(results, parent_order=ov.PARENT_KEYS)
if not _AUDIT["passed"]:
    _retry_flag = f"overall_audit_retry::{_bucket()}"
    if not st.session_state.get(_retry_flag):
        st.session_state[_retry_flag] = True
        get_results.clear()
        get_all_profiles.clear()
        st.rerun()                      # recompute the whole universe fresh
if not _AUDIT["passed"]:
    _stale_rows = [r for r in _AUDIT["rows"] if not r["fresh"]]
    _items = "".join(
        f"<li><b>{r['app']}</b> — instruments {', '.join(r['instruments'])}: "
        f"signals stuck on the <b>{r['actual_close']}</b> bar; expected the "
        f"<b>{r['expected_close']}</b> bar ({r['age_days']}d behind).</li>"
        for r in _stale_rows)
    st.markdown(
        "<style>@keyframes ovstaleflash{0%,100%{opacity:1}50%{opacity:.45}}"
        ".ov-stale-alert{animation:ovstaleflash 1.1s ease-in-out infinite;"
        "background:#dc2626;color:#fff;border:3px solid #7f1d1d;border-radius:12px;"
        "padding:16px 20px;margin:10px 0;font-size:17px;font-weight:800}"
        ".ov-stale-alert ul{margin:8px 0 0 18px;font-size:14px;font-weight:600}"
        "</style>"
        f"<div class='ov-stale-alert'>🚨 STALE SIGNALS — audit FAILED for "
        f"{len(_stale_rows)} of {len(_AUDIT['rows'])} signal apps "
        f"({', '.join(_AUDIT['stale_apps'])}). A forced data refresh did not "
        f"bring them current, so the combined strategy below is using "
        f"out-of-date signals for the flagged assets — treat their "
        f"recommendations with caution.<ul>{_items}</ul></div>",
        unsafe_allow_html=True)

# what the Daily Audit tab reports for the Overall view — recorded every render
fr.record_refresh("OVERALL", kind="overall",
                  as_of=str(pd.Timestamp(as_of).date()),
                  signals_generated_at_utc=str(pd.Timestamp(_PF["computed_at"])),
                  audit_passed=bool(_AUDIT["passed"]),
                  stale_apps=_AUDIT["stale_apps"],
                  audit_rows=_AUDIT["rows"],
                  profile=_profile,
                  n_instruments=len(results))

# ── live spot prices — overlay onto the display (not the signals/back-test) ──
# Signals run on completed daily bars (some cached, so a few days stale for
# BTC/MSTR/MSTU/ETH); the action-plan price and open-position P&L must show the
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
            # _a["last_close"] stays the close of the last completed bar (set in
            # signal_gated_allocation, before the spot overlay); expose the live
            # spot separately so the action plan shows both side by side.
            _a["live_price"] = _r["last_close"]; _a["upnl"] = _r["pos"]["upnl"]
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


tab_live, tab_hist, tab_bt, tab_explain = st.tabs(
    ["🔴 Live — Decision Cockpit", "🕰️ Historical View",
     "📊 Combined Backtesting", "🧠 Strategy & Methodology"])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE DECISION COCKPIT
# ══════════════════════════════════════════════════════════════════════════
with tab_live:
    _auto = (f" · auto-refreshing every {_AUTOREFRESH_SECS}s"
             if (st_autorefresh is not None and _n_spot) else "")
    _px_note = (f" · <span style='color:#16a34a'>● prices live (spot, {_n_spot}/{N_ALL})</span>{_auto}"
                if _n_spot else " · <span style='color:#dc2626'>spot quote unavailable — showing last bar</span>")
    st.markdown(f"#### Signals as of **{as_of.strftime('%b %d, %Y')}** (daily close) · "
                f"generated **{signals_updated.strftime('%b %d, %Y %I:%M %p %Z')}** · "
                f"{len(results)} instruments across {len(parents)} signals{_px_note}",
                unsafe_allow_html=True)
    st.caption(f"🕒 Signals last updated **{signals_updated.strftime('%b %d, %Y at %I:%M:%S %p %Z')}** — "
               f"auto-recomputed every ~30 min, or press **Refresh now** in the sidebar.")

    # per-asset-class signal closes + the audit verdict, matching 🕵️ Daily Audit
    _btc_row = next((r for r in _AUDIT["rows"] if r["asset_class"] == "crypto"), None)
    _eq_row = next((r for r in _AUDIT["rows"] if r["asset_class"] == "us_equity"), None)
    _aud_ico = "✅ audit PASSED — all signals fresh" if _AUDIT["passed"] else \
               f"🚨 audit FAILED — stale: {', '.join(_AUDIT['stale_apps'])}"
    st.caption(
        ("₿ Bitcoin signals from the close of "
         f"**{_btc_row['actual_close']}**" if _btc_row else "") +
        (" · " if _btc_row and _eq_row else "") +
        ("📈 Equity-app signals from the close of "
         f"**{_eq_row['actual_close']}**" if _eq_row else "") +
        f" · {_aud_ico} (checked {_AUDIT['checked_at_ct']})")
    with st.expander("🕵️ Signal-freshness audit — per-app closing bars", expanded=not _AUDIT["passed"]):
        _adf = pd.DataFrame([
            dict(App=r["app"],
                 Instruments=", ".join(r["instruments"]),
                 **{"Signals from close of": r["actual_close"],
                    "Freshest possible close": r["expected_close"],
                    "Status": "✅ Fresh" if r["fresh"] else f"🚨 STALE ({r['age_days']}d behind)"})
            for r in _AUDIT["rows"]])
        st.dataframe(_adf, use_container_width=True, hide_index=True)
        st.caption("The same audit gates the scheduled once-daily publish "
                   "(≈7:15 AM CT): the audit runs ONCE, the Overall strategy "
                   "recomputes only after every app's signals pass this check, "
                   "and the Target Book is then published immediately — and "
                   "stays frozen until the next morning. See the 🕵️ **Daily "
                   "Audit** app for the full trail.")

    # ── risk-profile switch — decide and trade accordingly ──────────────
    pcomp = {r["name"]: r for r in get_profile_comparison(_bucket())}
    rp = st.columns([1.15, 2])
    with rp[0]:
        st.radio("⚙️ **Risk profile**", UI_PROFILES,
                 key="overall_risk_profile", horizontal=True)
        st.caption(ov.RISK_PROFILES[_profile]["blurb"])
    with rp[1]:
        cells = []
        for name in UI_PROFILES:
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
        # period behind the ret/dd/Sharpe figures above — the end date rolls
        # forward automatically as each new daily close lands in the data
        _per0, _per1 = _PF["rets"].index[0], _PF["rets"].index[-1]
        st.caption(f"📅 Figures computed over **{_per0.strftime('%b %d, %Y')} → "
                   f"{_per1.strftime('%b %d, %Y')}** — end date advances daily "
                   f"with each new close.")
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
    # columns and by the "Recommended Live Possible Targetbook" pie below.
    # include_entries flags fresh
    # buys that would open into an already-broken trend as well as current holds.
    _live_exits = ov.live_exit_keys(results, _spot, include_entries=True)
    # mirror set for likely ENTRIES: flat names with no committed buy whose live
    # price now satisfies the trend's real long condition — if it holds into the
    # close they signal today and open next bar. Used only to flag rows green in
    # the action table (the live book never pre-funds an uncommitted signal).
    _live_entries = ov.live_entry_keys(results, _spot)

    def _load_published_book(*paths: Path) -> dict | None:
        """First readable published-book payload among *paths* (None if none)."""
        import json as _json
        for _p in paths:
            try:
                if _p.exists():
                    return _json.loads(_p.read_text())
            except Exception:
                continue
        return None

    def _book_alloc(payload: dict) -> tuple[dict, float]:
        """(risk_weights, idle_weight) from a published book — LIVE books park
        idle capital as a SATA weight, PAPER books as cash_weight."""
        w = {k: float(v) for k, v in (payload.get("weights") or {}).items()}
        idle = w.pop("SATA", 0.0) + float(payload.get("cash_weight") or 0.0)
        return w, idle

    # published books loaded BEFORE the live gate: the closed-market freeze
    # below pins the weekend/holiday live recommendation to the current book
    _TB_DIR = _REPO_ROOT / "data" / "overall"
    _prev_book = _load_published_book(_TB_DIR / "target_book_live_prev.json",
                                      _TB_DIR / "target_book_prev.json")
    _cur_book = _load_published_book(_TB_DIR / "target_book_live.json",
                                     _TB_DIR / "target_book.json")

    try:
        gate_live = ov.signal_gated_allocation(
            results, opt["optimal"]["weights"], caps=ov.caps_for(_profile),
            force_exit=_live_exits)
        # weekend / holiday guard: on days the US market is closed, the daily
        # tilt must not re-size sleeves whose signal apps got no new bar (the
        # cross-set priority normalisation would let the crypto sleeves' fresh
        # weekend bars move every stale sleeve's weight).  Pin the live book
        # to the published Current Targetbook — only committed signal changes
        # (the BTC/ETH apps' bars close every day) can move it.
        if _cur_book:
            gate_live = ov.apply_closed_market_freeze(
                gate_live, _book_alloc(_cur_book)[0])
    except Exception:
        gate_live = gate

    # ── 1. TARGETBOOKS — previous & current (as published) vs live view ──
    st.markdown("### 📐 Targetbooks — published vs live recommendation")
    st.caption("**Previous Targetbook** = **yesterday's** published book — "
               "BTC · MSTR · MSTU · ETH from *yesterday's* 7:00 AM CT bar close, all "
               "other tickers from the market close of the **day before "
               "yesterday**. An intraday re-publish never replaces it: the prev "
               "slot only rolls forward at the first publish of a new day. "
               "**Current Targetbook** = the *officially published* book the "
               "IBKR executor trades — its data basis is pinned to the "
               "morning **7:15 AM CT anchor**: all tickers as of the market "
               "close **before** that anchor (yesterday's 4:00 PM ET close) "
               "and BTC · MSTR · MSTU · ETH as of the day's **7:00 AM CT bar "
               "close** (each donut's caption shows the exact dates). Signal "
               "changes after today's market close appear ONLY in the "
               "Recommended Live Possible Targetbook until tomorrow's "
               "7:15 AM CT publish — the Current Targetbook never moves on "
               "them, even on a manual 🚀 re-publish. It is published **once** daily at "
               "≈7:15 AM CT (right after the daily audit) and does **not** "
               "change during the day — only the 🚀 **Publish new target "
               "book** button in the 📋 Target Book app (or a manual workflow "
               "run) replaces it. **Recommended Live Possible Targetbook** = "
               "what today's committed signals plus *live* prices recommend "
               "right now; it can drift through the day because today's market "
               "bar and BTC bar have not closed yet.")
    # (_live_exits / gate_live computed above, before the action table)

    def _alloc_donut(alloc: dict, sata: float, title: str):
        labels, vals, colors = [], [], []
        for kk, vv in sorted(alloc.items(), key=lambda x: -x[1]):
            if vv <= 0.0005:
                continue
            meta = by_key.get(kk) or ov.ASSET_META.get(kk, {})
            tag = {"lev": " 2×", "beta": " β"}.get(meta.get("kind"), "")
            labels.append(f"{kk}{tag}"); vals.append(vv * 100)
            colors.append(meta.get("accent") or "#94a3b8")
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

    def _book_close_caption(payload: dict) -> str:
        """Explicit close DATES a published book is built from.  Prefers the
        book's own signature-covered ``signal_basis`` stamp (written by the
        publisher); legacy books fall back to deriving the basis from their
        publish day's 7:15-AM-CT anchor — NOT from the wall-clock publish
        moment, since the basis is pinned to the anchor even for a late or
        manual publish."""
        try:
            basis = payload.get("signal_basis") or {}
            if basis.get("equity_close") and basis.get("btc_bar_close_utc"):
                eq_d = pd.Timestamp(basis["equity_close"])
                btc_cm = pd.Timestamp(basis["btc_bar_close_utc"])
            else:
                gen = payload.get("generated_at_utc")
                if not gen:
                    return ""
                anchor = fr.publish_anchor_ct(gen)
                eq_d = fr.expected_equity_asof(anchor)
                btc_cm = fr.close_moment("crypto", fr.expected_crypto_asof(anchor))
            eq = eq_d.strftime("%b %d, %Y")
            btc = (btc_cm.tz_convert("America/Chicago")
                   .strftime("%b %d, %Y, %I:%M %p %Z").replace(" 0", " "))
        except Exception:
            return ""
        return (f"Other tickers as of market close **{eq}** · BTC · MSTR · "
                f"MSTU · ETH as of bar close **{btc}**")

    # (_prev_book / _cur_book loaded above, before the live gate)
    # Today's-publish-pending notice: past the 7:15-AM-CT anchor but the newest
    # published book still predates it (GitHub cron fires routinely arrive
    # late), the donuts below are necessarily YESTERDAY's books — say so
    # instead of letting them read as silently stuck.
    if _cur_book and fr.publish_pending(_cur_book.get("generated_at_utc")):
        st.warning(
            "⏳ **Today's ≈7:15 AM CT publish hasn't landed yet** — the "
            "Previous / Current Targetbooks below are still **yesterday's** "
            "books. GitHub's scheduler often delivers the publish cycle late; "
            "the donuts roll forward automatically once today's book commits. "
            "To publish right now, use the 🚀 button in the 📋 Target Book "
            "app (or run the *Publish target book* workflow manually).")

    ac = st.columns([1, 1, 1])
    with ac[0]:
        if _prev_book:
            _pw, _pidle = _book_alloc(_prev_book)
            st.plotly_chart(_alloc_donut(_pw, _pidle, "Previous Targetbook"),
                            use_container_width=True)
            st.caption(f"Yesterday's published book — "
                       f"{_book_close_caption(_prev_book)} · profile "
                       f"**{_prev_book.get('profile', '—')}** · published "
                       f"**{fr.fmt_ct(_prev_book.get('generated_at_utc'))}**")
        else:
            st.info("**Previous Targetbook** — none recorded yet. It appears "
                    "after the next publish rotates the current book out "
                    "(`target_book*_prev.json`).")
    with ac[1]:
        if _cur_book:
            _cw, _cidle = _book_alloc(_cur_book)
            st.plotly_chart(_alloc_donut(_cw, _cidle, "Current Targetbook"),
                            use_container_width=True)
            st.caption(f"{_book_close_caption(_cur_book)} — committed closes "
                       f"only, matching the action plan's last-close targets · "
                       f"profile **{_cur_book.get('profile', '—')}** · published "
                       f"**{fr.fmt_ct(_cur_book.get('generated_at_utc'))}** · "
                       f"frozen until the next 7:15 AM CT publish (or a manual "
                       f"🚀 publish).")
        else:
            st.info("**Current Targetbook** — no published book found "
                    "(`data/overall/target_book*.json`). Run the **Publish "
                    "target book** workflow or the 🚀 button in the 📋 Target "
                    "Book app.")
    with ac[2]:
        st.plotly_chart(_alloc_donut(gate_live["target"], gate_live["sata"],
                                     "Recommended Live Possible Targetbook"),
                        use_container_width=True)
        st.caption("⚡ Live recommendation — **could change** until today's "
                   "market-day bar (4:00 PM ET) and BTC bar (7:00 AM CT "
                   "tomorrow) close; it becomes official only when published "
                   "as a Targetbook."
                   + (" **Live-adjusted:** pending exits removed." if _live_exits
                      else " No pending live exits."))
    _frz = gate_live.get("freeze")
    if _frz:
        st.info("🧊 **US market closed today** (weekend/holiday) — the "
                "Recommended Live book is **pinned to the published Current "
                "Targetbook**. Sleeves whose signal apps got no new bar keep "
                "their published weight exactly; only **committed signal "
                "changes** from apps whose bar still closes today (the ₿ BTC "
                "and ⟠ ETH apps — 7:00 AM CT bar, every day) can move the "
                "book. No daily-tilt re-sizing of positions that cannot "
                "trade. "
                + (f"Dropped on signal: **{', '.join(_frz['closed'])}**. "
                   if _frz["closed"] else "")
                + (f"Opened on signal: **{', '.join(_frz['opened'])}**. "
                   if _frz["opened"] else "")
                + ("Today: no signal changes — the live book equals the "
                   "published book." if not (_frz["closed"] or _frz["opened"])
                   else ""))
    if _live_exits:
        st.warning("⚠️ **Live-adjusted:** " + ", ".join(sorted(_live_exits)) +
                   " " + ("is" if len(_live_exits) == 1 else "are") +
                   " signalling long off the last close but the **live price is now "
                   "below the trend filter**, so " +
                   ("it" if len(_live_exits) == 1 else "they") +
                   " won't hold — " + ("it exits" if len(_live_exits) == 1
                                        else "they exit") +
                   " on the next bar. The **Recommended Live Possible "
                   "Targetbook** pie removes "
                   + ("it" if len(_live_exits) == 1 else "them") +
                   " and reallocates to the survivors and SATA.")
    else:
        st.caption("No held position's live price is below its trend filter — the "
                   "**Recommended Live Possible Targetbook** matches the book "
                   "today's committed signals would publish.")
    if _live_entries:
        st.success("🟢 **Likely entries:** " + ", ".join(sorted(_live_entries)) +
                   " " + ("is" if len(_live_entries) == 1 else "are") +
                   " flat off the last close but the **live price now satisfies "
                   "the entry condition** (trend filter crossed, or a divergence "
                   "app's entry gate firing on the live bar) — if it holds into "
                   "the close, the signal fires today and " +
                   ("it enters" if len(_live_entries) == 1 else "they enter") +
                   " on the next bar. Flagged green in the action plan below; the "
                   "live book never pre-funds a signal before it commits at the "
                   "close.")
    # committed entries the frozen morning book predates: the ENGINE's latest
    # close fired the buy (action OPEN) but the published book still records a
    # different action — the signal committed after the morning publish (e.g. a
    # divergence pure-regime buy at today's close). The action table flags the
    # row green; say it here too so it isn't lost behind the book-pinned pill.
    _book_act_now = {x.get("key"): x.get("action")
                     for x in ((_cur_book or {}).get("actions") or [])}
    _new_commits = sorted(
        a["key"] for a in gate["actions"]
        if a["action"] == "OPEN" and not a["in_pos"]
        and _book_act_now.get(a["key"]) not in (None, "OPEN"))
    if _new_commits:
        st.success("🟢 **New committed " +
                   ("entry" if len(_new_commits) == 1 else "entries") +
                   ":** " + ", ".join(_new_commits) + " — the buy signal "
                   "**fired at today's close**, after the morning publish, so "
                   "the frozen Current Targetbook doesn't carry it yet. " +
                   ("It enters" if len(_new_commits) == 1 else "They enter") +
                   " on the **next bar** and " +
                   ("gets" if len(_new_commits) == 1 else "get") +
                   " funded at the next ≈7:15 AM CT publish (or a manual 🚀 "
                   "publish).")

    # ── 1b. USER INCLUDE / EXCLUDE overlay on the live-adjusted book ─────
    # Same control as the 📋 Target Book viewer: untick a position and its weight
    # moves to SATA (idle cash), NOT to the other names. The rebalancing-moves and
    # target below then reflect the user's selection; with nothing excluded the
    # result is identical to the pure live-adjusted book.
    _lt = dict(gate_live["target"])
    _sel_included = set(_lt)
    if _lt:
        with st.expander("🎛️ Adjust the recommended book — include / exclude positions"):
            st.caption("Untick a position to drop it; its weight moves to **SATA** "
                       "(idle cash), not to the other names.")
            _sel_df = pd.DataFrame([
                dict(Include=True, Signal=kk, Instrument=by_key[kk]["name"],
                     Weight=_lt[kk] * 100)
                for kk in sorted(_lt, key=lambda x: -_lt[x])])
            _edited = st.data_editor(
                _sel_df, hide_index=True, use_container_width=True,
                key="overall_live_include",
                disabled=["Signal", "Instrument", "Weight"],
                column_config={
                    "Include": st.column_config.CheckboxColumn("Include", default=True),
                    "Weight": st.column_config.NumberColumn("Weight %", format="%.1f%%")})
            _sel_included = set(_edited.loc[_edited["Include"], "Signal"])

    # apply the selection (excluded weight → SATA); identity when nothing excluded
    _adj_target, _adj_sata, _excluded, _adj_dep, _adj_moved = ov.adjust_for_selection(
        _lt, gate_live["sata"], _sel_included)
    if _excluded:
        _pv_sel = st.session_state.get("overall_portfolio_value", 100000.0)
        st.info(f"Excluded **{', '.join(sorted(_excluded))}** → moved "
                f"**{_adj_moved*100:.1f}%** (${_adj_moved*_pv_sel:,.0f}) to SATA. "
                f"Deployed {_adj_dep*100:.0f}% · SATA {_adj_sata*100:.0f}%.")
        ac2 = st.columns([1, 1, 1])
        ac2[1].plotly_chart(_alloc_donut(_adj_target, _adj_sata,
                            "Recommended Live (your selection)"),
                            use_container_width=True)

    # ── BASELINE — the book actually in force ────────────────────────────
    # These moves must be the trades that take you from what you HOLD to the
    # recommended target, so the "from" side has to be the *published* Current
    # Targetbook — the artifact the IBKR executor trades and the middle donut
    # draws.  It used to be ``gate["current"]``, which is a different thing: the
    # engine's synthetic "what we hold now" (optimal base weights over the
    # in-position names, water-filled to caps, with NO priority tilt).  That
    # construct is neither what was published nor what is held, so the line
    # contradicted the donut right above it — e.g. it showed REMX 17% and OIH 15%
    # while the published book held REMX 0.3% and OIH 18.0%.
    # Fall back to the synthetic book only when nothing has been published yet.
    if _cur_book:
        _base_w, _base_idle = _book_alloc(_cur_book)
        _base_src = "published Current Targetbook"
    else:
        _base_w, _base_idle = dict(gate["current"]), gate["sata_now"]
        _base_src = "current holdings (nothing published yet)"
    _moves_label = (f"{_base_src} → your selection" if _excluded
                    else f"{_base_src} → live-adjusted target")

    # One collapsible box holds both rebalancing views: the trades that take the
    # book in force to the recommended target, and the prev → current published
    # comparison with the optimizer's rationale.
    with st.expander("🔁 **Rebalancing moves**", expanded=False):
        st.markdown(f"**{_moves_label}**")
        moves = []

        # Rounding-consistent deltas + retired-key handling live in the core so
        # they are unit-tested; this loop only renders them.
        for _mv in ov.rebalancing_moves(_base_w, _base_idle, _adj_target, _adj_sata):
            _kk, _d = _mv["key"], _mv["delta_pct"]
            if _kk == "SATA":
                _label, _badge = "💵 <b>SATA</b>", ""
                _col = C_EXIT if _d > 0 else C_BUY  # more idle cash = de-risking
            else:
                # A book published before a universe change can name an
                # instrument the live universe no longer trades (an ETHA-era
                # book after the ETH swap); it still belongs here, since the
                # action is to sell it to 0.
                _meta = by_key.get(_kk)
                _badge = _kind_badge(_meta["kind"]) if _meta else ""
                _label = (f"<b>{_kk}</b>" if _meta else
                          f"<b>{_kk}</b><span style='font-size:10px;color:#94a3b8'>"
                          f" (retired)</span>")
                _col = C_BUY if _d > 0 else C_EXIT
            moves.append(
                f"<span style='font-size:13px;margin-right:16px;white-space:nowrap'>"
                f"{_label}{_badge} {_mv['from_pct']:.0f}% → {_mv['to_pct']:.0f}% "
                f"<span style='color:{_col};font-weight:700'>"
                f"{'▲' if _d > 0 else '▼'}{abs(_d):.0f}pt</span></span>")
        st.markdown("".join(moves) if moves
                    else "_Book already at target — no rebalancing needed._",
                    unsafe_allow_html=True)

        # ── prev → current published books: what changed at the last publish &
        # why. Same donut-to-donut comparison the two left pies invite, with the
        # optimizer's rationale spelled out per move. Reasons come from the two
        # published payloads themselves (recorded decisions + priority scores),
        # not a live recomputation — see ov.book_move_reasons.
        if _prev_book and _cur_book:
            st.markdown("---")
            st.markdown(f"**Previous → Current Targetbook** "
                        f"(as-of {_prev_book.get('as_of', '—')} → "
                        f"{_cur_book.get('as_of', '—')}: why the optimizer moved)")
            st.caption("Each published book's weights are the optimal base "
                       "blend tilted by that morning's **entry-priority** score "
                       "(live momentum · macro sentiment · back-tested win-rate "
                       "· risk-adjusted edge), then **water-filled** to the "
                       "per-instrument caps, with the undeployable residual "
                       "parked in **SATA**. So a weight only moves between two "
                       "publishes because a **signal fired** (fresh entry), a "
                       "**signal died** (exit / trend break / retired from the "
                       "universe), or the **priority tilt drifted** and the "
                       "water-fill re-spread the difference across the uncapped "
                       "names. Sub-point moves are omitted, matching the moves "
                       "line above.")
            _bm_rows = []
            for _mv in ov.book_move_reasons(_prev_book, _cur_book):
                _kk, _d = _mv["key"], _mv["delta_pct"]
                if _kk == "SATA":
                    _label, _badge = "💵 <b>SATA</b>", ""
                    _col = C_EXIT if _d > 0 else C_BUY
                else:
                    _meta = by_key.get(_kk)
                    _badge = _kind_badge(_meta["kind"]) if _meta else ""
                    _label = (f"<b>{_kk}</b>" if _meta else
                              f"<b>{_kk}</b><span style='font-size:10px;"
                              f"color:#94a3b8'> (retired)</span>")
                    _col = C_BUY if _d > 0 else C_EXIT
                _bm_rows.append(
                    f"<div style='font-size:13px;margin:4px 0'>"
                    f"{_label}{_badge} {_mv['from_pct']:.0f}% → "
                    f"{_mv['to_pct']:.0f}% "
                    f"<span style='color:{_col};font-weight:700'>"
                    f"{'▲' if _d > 0 else '▼'}{abs(_d):.0f}pt</span>"
                    f"<span style='color:#64748b'> — {_mv['reason']}</span>"
                    f"</div>")
            st.markdown("".join(_bm_rows) if _bm_rows else
                        "_No whole-point moves — the optimizer kept the "
                        "current book essentially unchanged from the previous "
                        "one (signals and priority tilts barely drifted)._",
                        unsafe_allow_html=True)

    st.markdown("---")

    # ── 2. TODAY'S ACTION PLAN ──────────────────────────────────────────
    # The "Target % / $ (Last bar)" columns must show the SAME committed
    # allocation the Current Targetbook donut draws — the *published* book.
    # They used to show ``gate["target"]``, the engine's re-run of the
    # committed signals: its priority tilt blends live momentum/sentiment, so
    # the column drifted through the day and contradicted the frozen donut
    # above it.  Fall back to the engine's committed target only when nothing
    # has been published yet (same rule as the rebalancing-moves baseline).
    if _cur_book:
        _committed_w, _committed_idle = _book_alloc(_cur_book)
    else:
        _committed_w, _committed_idle = None, None
    # Action / Signal / Priority must be pinned to the published book for the
    # same reason: ``gate["actions"]`` is the cockpit's ~30-min engine re-run,
    # whose entry-priority tilt blends LIVE momentum and sentiment — so those
    # cells (and the priority ranking) drifted through the day alongside the
    # Recommended Live Possible Targetbook and contradicted the frozen book.
    # The published payload records each instrument's action, decision and
    # priority at publish time; keys the book doesn't carry (an instrument
    # added since the last publish) fall back to the live engine's cells.
    _book_actions = {a["key"]: a
                     for a in ((_cur_book.get("actions") or []) if _cur_book else [])
                     if a.get("key")}
    _plan_actions = gate["actions"]
    if _book_actions:
        _book_rank = {k: i for i, k in enumerate(_book_actions)}
        _plan_actions = sorted(gate["actions"],
                               key=lambda a: _book_rank.get(a["key"], len(_book_rank)))
    with st.expander("🎯 **Today's action plan**", expanded=False):
        st.caption("What to do now, ranked: **close** exits first, then **open** / "
                   "**hold**, ordered by entry priority. β = higher-beta sibling · "
                   "2× = leveraged (traded off the parent signal). The **priority** "
                   "score (0–1) blends momentum, macro sentiment, the strategy's "
                   "back-tested win-rate and its risk-adjusted edge — it decides which "
                   "signals get funded and how much. **Action**, **Signal** and "
                   "**Priority** (and the row order) are the **published Current "
                   "Targetbook's** recorded decisions — frozen at the morning publish, "
                   "matching the donut above, never drifting intraday with the live "
                   "engine (they fall back to the live engine's values only when "
                   "nothing has been published yet, or for an instrument added since "
                   "the last publish). Intraday drift shows up only in the live "
                   "columns and the red ⚠️ / green 🟢 flags. Held/opened risk assets "
                   "total "
                   "100%; any capped-out remainder is parked in **SATA**. Rows shaded "
                   "**red** ⚠️ are holds **or fresh entries** whose trend has broken on "
                   "the live price — they still **hold/open today** but **exit on the "
                   "next bar** (either the last close already crossed the trend, or the "
                   "live price has since slipped below it). Rows shaded **green** 🟢 are "
                   "the mirror on the way in: committed fresh buys that **enter on the "
                   "next bar**, plus flat names whose **live price now satisfies the "
                   "entry condition** — a trend filter crossed, or a divergence app's "
                   "entry gate (e.g. ARTY's U1 + regime confirm) firing with the live "
                   "price as the provisional bar — if it holds into the close, the "
                   "signal fires today and they **likely enter next bar** (an "
                   "uncommitted signal is never pre-funded in the live columns). "
                   "**Price (Close of Last Bar)** "
                   "is the official close of the last completed daily bar the signals run "
                   "on; **Live Price** is the current spot quote (coloured green/red vs "
                   "that close). **Unreal. P&L** is measured "
                   "against each position's real cost basis — the official close on its "
                   "entry bar. **Target % / $ (Last bar)** is the committed allocation "
                   "of the **published Current Targetbook** — the same values as the "
                   "donut above (falling back to the engine's last-close targets only "
                   "when nothing has been published yet); **Target % / $ (Live)** re-runs it "
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
               "<th>Signal</th><th style='text-align:center'>Priority</th>"
               "<th style='text-align:right'>Price (Close of Last Bar)</th>"
               "<th style='text-align:right'>Live Price</th>"
               "<th style='text-align:right'>Chg %</th>"
               "<th style='text-align:right'>Unreal. P&amp;L</th>"
               "<th style='text-align:right'>Target % (Last bar)</th>"
               "<th style='text-align:right'>Target $ (Last bar)</th>"
               "<th style='text-align:right'>Target % (Live)</th>"
               "<th style='text-align:right'>Target $ (Live)</th></tr>")
        rows = []
        for a in _plan_actions:
            # published-book cells (frozen at publish) with live-engine fallback
            _ba = _book_actions.get(a["key"])
            _act = (_ba.get("action") or a["action"]) if _ba else a["action"]
            _dec = (_ba.get("decision") or a["decision"]) if _ba else a["decision"]
            _prio = _ba["priority"] if _ba else a["priority"]
            # colour the decision by the (possibly book-sourced) action so the
            # two cells never contradict; the action→colour map mirrors the
            # live tone map state for state.
            _dec_col = (_ACTION_COL.get(_act, C_FLAT) if _ba
                        else _TONE_COL.get(a["tone"], C_FLAT))
            ac = _ACTION_COL.get(_act, C_FLAT)
            # last-bar (committed) — the published book's weight, so the column
            # matches the Current Targetbook donut slice for slice
            tgt = (_committed_w.get(a["key"], 0.0) if _committed_w is not None
                   else a["target"])
            # live-adjusted target, further reduced by the user's include/exclude
            # selection (an excluded position shows 0 here, its weight in SATA).
            tgt_live = _adj_target.get(a["key"], 0.0)
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
            if _prio is not None:
                p = _prio; pcolor = C_BUY if p >= 0.6 else C_WATCH if p >= 0.4 else C_FLAT
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
            # live spot price (falls back to the last-bar close when no live quote);
            # coloured green/red vs the last-bar close to show the intraday move.
            _live_px = a.get("live_price")
            if _live_px is None:
                _live_px = a["last_close"]
            _live_px_s = f"${_live_px:,.2f}"
            _live_px_col = (C_BUY if _live_px > a["last_close"]
                            else C_EXIT if _live_px < a["last_close"] else "inherit")
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
            # …and the green mirror: a row that opens on the NEXT bar. Two cases,
            # exactly like the exits: the committed entry (the ENGINE's action is
            # OPEN — the latest close fired the buy signal; deliberately NOT the
            # book-pinned _act, since a signal that commits at today's close
            # AFTER the morning publish — e.g. ARTY's pure-regime buy — is
            # exactly what this flag must surface, like exits_next_bar it drifts
            # with the engine), and the live-driven likely entry — a flat row
            # with no committed buy whose live price now satisfies the entry
            # condition (in _live_entries), so if it holds into the close it
            # signals today and enters next bar. A pending exit always wins the
            # row colour (an OPEN into a re-broken trend stays red).
            _committed_entry = (not a["in_pos"]) and a["action"] == "OPEN"
            # a committed entry the frozen morning book predates (its recorded
            # action isn't OPEN) gets called out — the pill and flag disagree
            # on purpose: the pill is the book, the flag is today's signal.
            _book_masked_entry = _committed_entry and _ba and _act != "OPEN"
            _live_entry = a["key"] in _live_entries
            enter_next = (not exit_next) and (_committed_entry or _live_entry)
            if exit_next:
                row_style = ("border-bottom:1px solid #fecaca;background:#fef2f2;"
                             "box-shadow:inset 3px 0 0 #dc2626")
            elif enter_next:
                row_style = ("border-bottom:1px solid #bbf7d0;background:#f0fdf4;"
                             "box-shadow:inset 3px 0 0 #16a34a")
            else:
                row_style = "border-bottom:1px solid #eef2f7"
            if _committed_exit:                        # last close already below trend
                warn = ("<div style='font-size:10px;color:#dc2626;font-weight:700'>"
                        "⚠️ exits next bar</div>")
            elif _live_exit:                           # live-driven (hold or fresh entry)
                warn = ("<div style='font-size:10px;color:#dc2626;font-weight:700'>"
                        "⚠️ live px below trend — exits next bar</div>")
            elif _book_masked_entry:                   # fired after the morning publish
                warn = ("<div style='font-size:10px;color:#16a34a;font-weight:700'>"
                        "🟢 signal fired at today's close — enters next bar</div>")
            elif _committed_entry:                     # last close already fired the buy
                warn = ("<div style='font-size:10px;color:#16a34a;font-weight:700'>"
                        "🟢 enters next bar</div>")
            elif _live_entry:                          # live-driven likely entry
                _why = ("live px above trend" if _r["mode"] in ("ma", "dual_ma", "ma_vol")
                        else "live px fires entry signal")   # divergence apps (e.g. ARTY)
                warn = ("<div style='font-size:10px;color:#16a34a;font-weight:700'>"
                        f"🟢 {_why} — likely enters next bar</div>")
            else:
                warn = ""
            rows.append(
                f"<tr style='{row_style}'>"
                f"<td style='padding:8px 10px'>{_pill(_act, ac)}{warn}</td>"
                f"<td style='font-weight:700'>{a['emoji']} {a['key']}{_kind_badge(a['kind'])}"
                f"<div style='font-size:11px;color:#64748b;font-weight:400'>{a['name']}</div></td>"
                f"<td style='font-size:12px;color:{_dec_col}'>{_dec}"
                f"<div style='font-size:10px;color:#94a3b8'>{sub}</div></td>"
                f"<td style='text-align:center;font-size:12px;min-width:56px'>{prio_cell}</td>"
                f"<td style='text-align:right;font-variant-numeric:tabular-nums'>${a['last_close']:,.2f}</td>"
                f"<td style='text-align:right;font-weight:600;font-variant-numeric:tabular-nums;color:{_live_px_col}'>{_live_px_s}</td>"
                f"<td style='text-align:right;font-weight:600;font-variant-numeric:tabular-nums;color:{chg_col}'>{chg_s}</td>"
                f"<td style='text-align:right;color:{pnl_col};font-weight:600'>{pnl}{cb_sub}</td>"
                f"<td style='text-align:right;font-weight:700'>{tgt_s}{bar}</td>"
                f"<td style='text-align:right;font-weight:700;font-variant-numeric:tabular-nums'>{amt_s}</td>"
                f"<td style='text-align:right;font-weight:700;color:{_live_col}'>{tgt_live_s}{bar_live}</td>"
                f"<td style='text-align:right;font-weight:700;font-variant-numeric:tabular-nums;color:{_live_col}'>{amt_live_s}</td></tr>")
        # SATA row — the idle-cash park absorbing whatever risk assets can't hold;
        # the last-bar cell shows the published book's idle weight (the donut's
        # SATA slice), falling back to the engine's committed SATA when unpublished.
        si = gate["sata_info"]
        sata_pct = _committed_idle if _committed_idle is not None else gate["sata"]
        sata_live = gate_live["sata"]

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
            f"<td style='text-align:right;font-variant-numeric:tabular-nums'>${si['par']:,.2f}</td>"
            f"<td style='text-align:right;font-weight:600;font-variant-numeric:tabular-nums'>{sa_px_s}</td>"
            f"<td style='text-align:right;font-weight:600;font-variant-numeric:tabular-nums;color:{sa_dc_col}'>{sa_dc_s}</td>"
            f"<td style='text-align:right;font-weight:600;color:{sa_pnl_col}'>{sa_pnl_s}{sa_pnl_sub}</td>"
            f"<td style='text-align:right;font-weight:800'>{sata_pct*100:.1f}%{sbar}</td>"
            f"<td style='text-align:right;font-weight:800;font-variant-numeric:tabular-nums'>"
            f"${sata_pct*portfolio_value:,.0f}</td>"
            f"<td style='text-align:right;font-weight:800;color:{_sata_col}'>{sata_live*100:.1f}%{sbar_live}</td>"
            f"<td style='text-align:right;font-weight:800;font-variant-numeric:tabular-nums;color:{_sata_col}'>"
            f"${sata_live*portfolio_value:,.0f}</td></tr>")
        st.markdown(f"<div style='overflow-x:auto;margin:8px 0;'>"
                    f"<table style='width:100%;border-collapse:collapse'>{hdr}{''.join(rows)}</table></div>",
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
    with st.expander("🛰️ **Live signal & positions — by app**", expanded=False):
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
                "BTC (BTC/MSTR/MSTU/ETH) or Gold (GDX/UGL), open the **₿ Bitcoin** or "
                "**🥇 Gold** app in the sidebar.")

    st.markdown("---")

    # ── 4. OVERALL STRATEGY P&L SINCE A USER-CHOSEN START DATE ──────────
    # What has the combined strategy actually delivered for someone who put
    # capital in on a given date?  Two selectable sources feed one identical
    # pipeline: the AS-PUBLISHED record (the target books actually committed
    # each day — data/overall/book_archive/ — so every daily optimiser
    # trim/re-size is in every figure) or the WALK-FORWARD GATED REPLAY
    # simulation of the same rules over the full back-test history.  Either
    # curve is re-based at the chosen date so drawdown, Sharpe etc. are
    # measured from the entry point, not from inception.
    with st.expander("📈 **Overall strategy P&L — pick your start & end dates**", expanded=False):
        _books = get_published_books(_bucket())
        _vmap = get_book_version_map(_bucket())
        # full archive only for context (how many older-generation books
        # exist); the as-published VIEW is CURRENT-STRATEGY BOOKS ONLY —
        # stamped with STRATEGY_VERSION, signal days on/after its start date
        _bookrep_all = (ov.published_book_replay(_PF["rets"], _books,
                                                 version_map=_vmap)
                        if _books else None)
        _bookrep = (ov.published_book_replay(
                        _PF["rets"], _books, version_map=_vmap,
                        only_version=ov.STRATEGY_VERSION,
                        min_as_of=ov.STRATEGY_VERSION_START)
                    if _books else None)
        _ver_up = ov.STRATEGY_VERSION.upper()
        _ver_start = pd.Timestamp(ov.STRATEGY_VERSION_START).strftime("%b %d, %Y")
        _SRC_ACTUAL = "🎯 As-published record (actual books)"
        _SRC_REPLAY = "🧪 Walk-forward replay (simulated)"
        if _bookrep_all is not None:
            _pnl_src = st.radio(
                "Performance source", [_SRC_ACTUAL, _SRC_REPLAY],
                horizontal=True, key="overall_pnl_source",
                help="**As-published record** — compounds the target books the "
                     "publisher actually committed each day (one archived JSON "
                     "per signal day), so the daily optimizer's trims and "
                     "re-sizes are exactly the ones that really happened. "
                     f"**Only books published under the current strategy "
                     f"version ({_ver_up}, from {_ver_start}) are counted** — "
                     "older-generation books never mix in. "
                     "**Walk-forward replay** — a look-ahead-free simulation "
                     "of the current rules over the full back-test "
                     "history (longer window, but a reconstruction, not the "
                     "record).")
        else:
            _pnl_src = _SRC_REPLAY
            st.caption("ℹ️ No archived published books found "
                       "(`data/overall/book_archive/`) — showing the "
                       "walk-forward replay only. The as-published view "
                       "unlocks once the daily publisher has archived books.")
        _actual = _pnl_src == _SRC_ACTUAL
        if _actual and _bookrep is None:
            # the current version's record hasn't accumulated 2 bars yet —
            # say so explicitly and fall back to the replay, never to
            # old-generation books
            _n_old = len(_bookrep_all["books"])
            st.info(f"🎯 The as-published record under the current strategy "
                    f"version **{_ver_up}** starts accumulating on "
                    f"**{_ver_start}** — not enough {_ver_up}-stamped books "
                    f"yet. The {_n_old} archived "
                    f"book{'s' if _n_old != 1 else ''} from earlier strategy "
                    "generations are excluded by design, so old-logic "
                    "performance can never mix into this view. Showing the "
                    "🧪 walk-forward replay (the current implementation's "
                    "full-history back-test) until the record exists.")
            _actual = False
        if _actual:
            _n_books = len(_bookrep["books"])
            _n_old = len(_bookrep_all["books"]) - _n_books
            st.caption(f"P&L, performance and risk of the **as-published "
                       f"Overall strategy book** — the {_n_books} dated target "
                       f"book{'s' if _n_books != 1 else ''} committed by the "
                       f"daily publisher (`data/overall/book_archive/`) "
                       f"**under the current strategy version {_ver_up}** "
                       f"(records start {_ver_start}), compounded at "
                       "official closes. The optimizer re-sizes the book "
                       "every morning, so **every daily trim/re-size that "
                       "really happened flows through every figure and "
                       "toggled section below** (the 🔁 toggle lists each "
                       "book and its trims). This view follows the profile "
                       "each book was *actually published under* (shown per "
                       "book below) — the risk-profile selector above does "
                       "not rewrite history. "
                       + (f"**{_n_old} older-generation "
                          f"book{'s' if _n_old != 1 else ''}** (pre-{_ver_up} "
                          "logic) are excluded so the record reflects only "
                          "the currently-implemented strategy. "
                          if _n_old else "")
                       + "Dollar figures scale the 💼 portfolio value entered "
                         "above.")
            # no separate signal record exists for archived books — the trade
            # log falls back to weight-inferred signal-vs-tilt classification
            _wf = {"weights": _bookrep["weights"], "sata": _bookrep["sata"],
                   "active": None}
            _strat_label = STRAT_CURVE_ACTUAL
            _curves_view = {STRAT_CURVE_ACTUAL: _bookrep["equity"],
                            **_PF["curves"]}
            _curve_all = _bookrep["equity"]
            if _bookrep["dropped"]:
                st.caption("⚠️ Book keys not in today's universe (weight "
                           "earns nothing): "
                           + ", ".join(f"`{k}`" for k in _bookrep["dropped"]))
        else:
            st.caption(f"P&L, performance and risk of the **daily-gated Overall "
                       f"strategy** (walk-forward replay — the same gate/tilt "
                       f"logic the live book runs, anchors re-fit each quarter on "
                       f"prior data only) measured from the start date below, "
                       f"under the risk profile selected above (currently "
                       f"**`{_profile}`**). Change the profile or the date and "
                       "every figure recomputes. Dollar figures scale the 💼 "
                       "portfolio value entered above.")
            _wf = _PF["wf"]
            _strat_label = STRAT_CURVE
            _curves_view = _PF["curves"]
            _curve_all = _PF["curves"][STRAT_CURVE]
        _d0, _d1 = _curve_all.index[0].date(), _curve_all.index[-1].date()
        _default_start = pd.Timestamp("2026-03-01").date()
        _today = pd.Timestamp.now(tz="America/New_York").date()
        _end_max = max(_d1, _today)
        # switching source narrows the valid range (the as-published record is
        # younger than the replay) — clamp a remembered date or the widget errors
        if "overall_pnl_start" in st.session_state:
            _remember = st.session_state["overall_pnl_start"]
            if _remember < _d0 or _remember > _d1:
                st.session_state["overall_pnl_start"] = min(max(_remember, _d0), _d1)
        if "overall_pnl_end" in st.session_state:
            _remember_e = st.session_state["overall_pnl_end"]
            if _remember_e < _d0 or _remember_e > _end_max:
                st.session_state["overall_pnl_end"] = min(max(_remember_e, _d0),
                                                          _end_max)
        pnl_cols = st.columns([1, 1, 2])
        with pnl_cols[0]:
            _start_sel = st.date_input(
                "📅 Start date", value=min(max(_default_start, _d0), _d1),
                min_value=_d0, max_value=_d1, key="overall_pnl_start",
                help="First strategy bar on/after this date becomes the cost-basis "
                     "anchor (weekends/holidays roll forward). Defaults to "
                     "March 1, 2026.")
        with pnl_cols[1]:
            _end_sel = st.date_input(
                "📅 End date", value=min(max(_today, _d0), _end_max),
                min_value=_d0, max_value=_end_max, key="overall_pnl_end",
                help="Last strategy bar on/before this date closes the "
                     "measurement window (weekends/holidays roll back). "
                     "Defaults to today on every load, so the section keeps "
                     "measuring through the latest bar unless you pick an "
                     "earlier cut-off.")
        # everything below measures the [start, end] window — truncate the
        # curves / weight matrices / returns once here and every downstream
        # figure, chart and toggle follows.  _end_arg stays None while the end
        # date covers the latest bar so open positions keep counting as today.
        _end_ts = pd.Timestamp(_end_sel)
        _end_arg = None if _end_sel >= _d1 else _end_sel
        _curve_all = _curve_all.loc[:_end_ts]
        _curves_view = {k: v.loc[:_end_ts] for k, v in _curves_view.items()}
        _wf = {"weights": _wf["weights"].loc[:_end_ts],
               "sata": _wf["sata"].loc[:_end_ts],
               # replay source: per-day signal truth (funded set); absent on
               # older cached replays and the as-published record
               "active": (_wf.get("active").loc[:_end_ts]
                          if _wf.get("active") is not None else None)}
        _rets_win = _PF["rets"].loc[:_end_ts]
        # sleeve-inclusion gate + per-trade notionals under a daily-weight book
        _wmax = {k: float(_wf["weights"][k].max()) for k in _wf["weights"].columns}
        _sm = ov.slice_metrics(_curve_all, _start_sel)
        if _sm is None:
            st.warning("Not enough strategy history between those dates — pick "
                       "an earlier start or a later end "
                       f"(data runs {_d0} → {_d1}).")
        else:
            # window label for every toggle/chart below: "since <start>" while
            # the end date covers the latest bar, "<start> → <end>" otherwise
            _win_lbl = (f"since {_sm['start'].strftime('%b %d, %Y')}"
                        if _end_arg is None else
                        f"{_sm['start'].strftime('%b %d, %Y')} → "
                        f"{_sm['end'].strftime('%b %d, %Y')}")
            _pnl_d = _sm["total_ret"] * portfolio_value
            _end_v = portfolio_value * (1 + _sm["total_ret"])
            _src_note = (f"as-published <b style='color:{_sv.BADGE_COLOR}'>"
                         f"{_ver_up}</b> record · {_n_books} "
                         f"book{'s' if _n_books != 1 else ''}" if _actual
                         else f"profile <code>{_profile}</code>")
            with pnl_cols[2]:
                st.markdown(
                    f"<div style='font-size:13px;color:#64748b;padding-top:30px'>"
                    f"Measured <b>{_sm['start'].strftime('%b %d, %Y')} → "
                    f"{_sm['end'].strftime('%b %d, %Y')}</b> · {_sm['days']} trading "
                    f"days · {_src_note}</div>",
                    unsafe_allow_html=True)
            pm = st.columns(4)
            pm[0].metric("Strategy P&L", f"{_sm['total_ret']*100:+.1f}%",
                         delta=f"${_pnl_d:+,.0f} on ${portfolio_value:,.0f}",
                         delta_color="normal" if _pnl_d >= 0 else "inverse")
            pm[1].metric("Portfolio value now", f"${_end_v:,.0f}",
                         delta=f"CAGR {_sm['cagr']*100:+.1f}%")
            pm[2].metric("Max drawdown since start", f"{_sm['mdd']*100:.1f}%",
                         help="Deepest peak-to-trough fall measured from the chosen "
                              "start date, not from back-test inception.")
            pm[3].metric("Sharpe (ann.)", f"{_sm['sharpe']:.2f}",
                         delta=f"vol {_sm['vol']*100:.0f}%", delta_color="off")
            # per-asset since-start reads (also feed the blend-level trade stats)
            _pa = ov.per_asset_slice_metrics(results, _start_sel, end=_end_arg)
            _ots = ov.overall_trade_stats(_pa, _wmax)
            # winning-days read conditioned on days with risk on — SATA-only
            # days would otherwise count as tiny guaranteed wins
            _dw = ov.daily_win_stats(_curve_all, _start_sel,
                                     active=_wf["weights"].sum(axis=1))
            pm2 = st.columns(6)
            if _dw is None or _dw["win_rate"] is None:
                pm2[0].metric("Winning days (invested)", "—",
                              delta=f"all-days {_sm['win_days']*100:.0f}%",
                              delta_color="off",
                              help="No days with risk deployed in this window "
                                   "— the book sat fully in SATA cash.")
            else:
                pm2[0].metric(
                    "Winning days (invested)", f"{_dw['win_rate']*100:.0f}%",
                    delta=(f"{_dw['wins']}/{_dw['n_active']}d · CI "
                           f"{_dw['ci_lo']*100:.0f}–{_dw['ci_hi']*100:.0f}%"),
                    delta_color="off",
                    help="Share of days the blend closed up, counting ONLY days "
                         "it actually held a position — fully-in-cash days are "
                         "excluded so the steady SATA yield can't inflate the "
                         "rate. The Wilson 95% interval says how seriously to "
                         "take the headline: with few invested days a coin "
                         "flip sits inside it.")
            pm2[1].metric("Best day", f"{_sm['best_day']*100:+.2f}%")
            pm2[2].metric("Worst day", f"{_sm['worst_day']*100:+.2f}%")
            if _dw is None or _dw["win_rate"] is None:
                pm2[3].metric("Trading days", f"{_sm['days']}")
            else:
                pm2[3].metric(
                    "Daily edge (invested)", f"{_dw['expectancy']*100:+.2f}%",
                    delta=(f"W {_dw['avg_win']*100:+.2f}% · "
                           f"L {_dw['avg_loss']*100:+.2f}%"),
                    delta_color="off",
                    help="Mean return per invested day — win rate × average "
                         "winning day + loss rate × average losing day. The "
                         "number that actually compounds: a high win rate "
                         "with a negative edge still loses money.")
            pm2[4].metric("Trades (incl. open)", f"{_ots['n_trades']}",
                          delta=f"{_ots['n_open']} open" if _ots["n_open"] else "all closed",
                          delta_color="off",
                          help="Round-trip trades across every sleeve the optimal "
                               "blend holds (weight > 0) that were open at any point "
                               "since the start date — including trades entered "
                               "before it — plus currently-open positions.")
            pm2[5].metric("Win rate",
                          "—" if _ots["win_rate"] is None else f"{_ots['win_rate']*100:.0f}%",
                          delta=f"{_ots['wins']}/{_ots['n_trades']} won" if _ots["n_trades"] else None,
                          delta_color="off",
                          help="Winning trades out of all counted trades: closed "
                               "trades on their realised return, open positions on "
                               "their current unrealised P&L.")

            # same-window benchmark comparison — is the strategy earning its
            # keep?  In as-published mode the walk-forward replay stays in the
            # table as a benchmark row, so record-vs-simulation is one glance.
            _bench_rows = []
            for _nm, _cv in _curves_view.items():
                _bm_sm = ov.slice_metrics(_cv, _start_sel)
                if _bm_sm:
                    _bench_rows.append((_nm, _bm_sm))
            if len(_bench_rows) > 1:
                bh = ("<tr style='background:#f1f5f9;font-size:12px;text-align:left'>"
                      "<th style='padding:6px 10px'>Strategy (same window)</th>"
                      "<th style='text-align:right'>P&amp;L</th>"
                      "<th style='text-align:right'>$ on portfolio</th>"
                      "<th style='text-align:right'>Max DD</th>"
                      "<th style='text-align:right'>Sharpe</th></tr>")
                br = []
                for _nm, _m in _bench_rows:
                    _hi = "background:#eff6ff;font-weight:700;" if _nm == _strat_label else ""
                    _pc = C_BUY if _m["total_ret"] >= 0 else C_EXIT
                    br.append(
                        f"<tr style='border-bottom:1px solid #eef2f7;{_hi}'>"
                        f"<td style='padding:6px 10px'>{_nm}"
                        f"{' ◄ this strategy' if _nm == _strat_label else ''}</td>"
                        f"<td style='text-align:right;color:{_pc};font-weight:600'>"
                        f"{_m['total_ret']*100:+.1f}%</td>"
                        f"<td style='text-align:right;font-variant-numeric:tabular-nums'>"
                        f"${_m['total_ret']*portfolio_value:+,.0f}</td>"
                        f"<td style='text-align:right;color:{C_EXIT}'>{_m['mdd']*100:.1f}%</td>"
                        f"<td style='text-align:right'>{_m['sharpe']:.2f}</td></tr>")
                st.markdown(f"<div style='overflow-x:auto;margin:8px 0;'>"
                            f"<table style='width:100%;border-collapse:collapse'>{bh}{''.join(br)}</table></div>",
                            unsafe_allow_html=True)

            # equity curve re-based to the portfolio value at the chosen start
            _fig_pnl = go.Figure()
            _pnl_styles = {_strat_label: ("#111827", 3),
                           "Equal-weight strategies": ("#0ea5e9", 1.5),
                           "Equal-weight Buy & Hold": ("#94a3b8", 1.5)}
            if _actual:                        # replay demoted to a benchmark line
                _pnl_styles[STRAT_CURVE] = ("#7c3aed", 1.5)
            for _nm, _cv in _curves_view.items():
                _sub = _cv.loc[pd.Timestamp(_start_sel):]
                if len(_sub) < 2:
                    continue
                _c, _w = _pnl_styles.get(_nm, ("#888", 1.5))
                _fig_pnl.add_trace(go.Scatter(
                    x=_sub.index, y=(_sub / _sub.iloc[0]).to_numpy() * portfolio_value,
                    name=_nm, line=dict(color=_c, width=_w)))
            _fig_pnl.add_hline(y=portfolio_value, line_dash="dot",
                               line_color="#cbd5e1",
                               annotation_text="break-even", annotation_font_size=10)
            _src_title = (f"as-published books ({_ver_up})" if _actual
                          else f"`{_profile}` profile")
            _fig_pnl.update_layout(
                height=340, margin=dict(t=30, b=10, l=10, r=10),
                yaxis_title="Portfolio value ($)", hovermode="x unified",
                legend=dict(orientation="h", y=1.1),
                title=dict(text=f"Growth of ${portfolio_value:,.0f} "
                                f"{_win_lbl} — {_src_title}",
                           font_size=13))
            st.plotly_chart(_fig_pnl, use_container_width=True)
            if _actual:
                st.caption("📌 The **as-published record**: each day compounds "
                           "the archived book actually committed from the "
                           "previous bar's close — weights, daily optimizer "
                           "trims, cash remainder (earning SATA on business "
                           "days) exactly as published. Marked at official "
                           "closes, gross of commissions/fills (the IBKR "
                           "executed-book history is the fills-level record). "
                           "Not investment advice.")
            else:
                st.caption("⚠️ Simulated performance of the daily-gated strategy under "
                           "the selected risk profile (idle capital earning SATA), "
                           "assuming entry at the close of the anchor bar. The replay is "
                           "walk-forward — anchor weights re-fit each quarter start on "
                           "prior data only, priorities from as-of inputs lagged one bar — so "
                           "no figure uses information from after the day it describes. "
                           "Not investment advice.")

            # ── daily % allocation — the book behind the curve above ─────────
            if st.toggle(f"📊 Daily % allocation by asset {_win_lbl} — every "
                         "asset's share of the book through time (toggle on/off)",
                         key="overall_alloc_history_pnl"):
                st.plotly_chart(
                    _alloc_area_fig(
                        _wf["weights"].loc[pd.Timestamp(_start_sel):],
                        _wf["sata"].loc[pd.Timestamp(_start_sel):],
                        f"Daily % allocation {_win_lbl} — every asset "
                        f"plus SATA (stacks to 100%)"),
                    use_container_width=True)
                st.caption("The exact book the P&L above compounds each day, "
                           "**one colour per signal group** (e.g. ₿ BTC covers "
                           "BTC·MSTR·MSTU·ETH — hover any band for that day's "
                           "per-asset split); grey on top is idle cash in "
                           "**SATA**. Bands widen and narrow daily as "
                           + ("the published books trim and re-size positions "
                              "— even when no Action signal changes."
                              if _actual else
                              "priorities move — even when no Action signal "
                              "changes."))

            # per-day $ / % P&L off the same weights + returns as the P&L-by-
            # asset attribution — shared by the trade log's Day-P&L column and
            # the 📆 Daily P&L chart, and exact for both sources.  LAZY: the
            # series (and the trade-log diff) are computed only when a toggle
            # that needs them is actually on, so the default page load pays
            # nothing for these sections.
            _dpl_memo = {}

            def _get_dpl():
                if "v" not in _dpl_memo:
                    _dpl_memo["v"] = ov.pnl_daily_replay(
                        _rets_win, _wf["weights"], _wf["sata"], _start_sel,
                        sata_daily=ov.SATA_DAILY)
                return _dpl_memo["v"]

            # ── daily trade log — every buy/sell the daily book ordered ──────
            # (both sources: the source radio above decides which weight
            # matrix — as-published books or walk-forward replay — feeds it)
            if st.toggle(
                    f"🧾 Daily trade log {_win_lbl} — positions bought "
                    "& sold day by day, with each day's P&L (toggle on/off)",
                    key="overall_daily_trade_log"):
                _dtl = ov.daily_trade_log(_wf["weights"], _wf["sata"],
                                          _start_sel, returns=_rets_win,
                                          active=_wf.get("active"))
                _dtl_buys = sum(d["n_buys"] for d in _dtl)
                _dtl_sells = sum(d["n_sells"] for d in _dtl)
                _dtl_resz = sum(d["n_resize"] for d in _dtl)
                _dpl = _get_dpl()
                st.caption(f"**{_dtl_buys} "
                           f"buy{'s' if _dtl_buys != 1 else ''} · "
                           f"{_dtl_sells} "
                           f"sell{'s' if _dtl_sells != 1 else ''} · "
                           f"{_dtl_resz} tilt "
                           f"re-size{'s' if _dtl_resz != 1 else ''} across "
                           f"{len(_dtl)} trading "
                           f"day{'s' if len(_dtl) != 1 else ''}.** "
                           "Every position **bought and sold, day by day**, as "
                           "the "
                           + ("**as-published books**" if _actual
                              else "**walk-forward replay**")
                           + " (per the performance-source toggle above) "
                           "re-shaped the book: **🚦 signal changes** open "
                           "(green) or fully close (red) a sleeve when its "
                           "Action signal flips, while **⚖️ daily tilt "
                           "adjustments** re-size positions already held — "
                           "the daily optimizer's adds and trims"
                           + (" (inferred from the archived weights — no "
                              "separate signal record exists for published "
                              "books)" if _actual else
                              " (classified off the actual signal record, "
                              "so an optimizer re-size through zero stays a "
                              "tilt)")
                           + ". Tilt sales also carry the **realized P&L "
                           "on the slice sold** — measured over the "
                           "position's **average cost**: tilt adds raise "
                           "the basis at the price actually paid, daily "
                           "re-balance flows included, and a position that "
                           "fully closes restarts its basis on re-entry "
                           "(held-at-first-bar positions start theirs at "
                           "that bar's close) — plus the ≈ $ gain or loss "
                           "locked in on the dollars sold, scaled by the "
                           "💼 portfolio value. Each row is "
                           "the trade ticket executed at that bar's close "
                           "(the book earns from the next bar — the "
                           "decided-at-previous-close convention), newest "
                           "first; days with no changes are omitted. "
                           "**≈ $ traded** scales the day's gross weight "
                           "movement (Σ|Δweight|) by the 💼 portfolio value; "
                           "turnover is one-way (½·Σ|Δweight|, cash leg "
                           "included), matching the published-book trims. "
                           "**Day P&L** is what the whole book earned or lost "
                           "over that bar, in dollars on the 💼 portfolio "
                           "value and in % — set up by the *previous* close's "
                           "book; the row's own trades start earning from the "
                           "next bar. The anchor bar's opening book is the "
                           "cost basis, so the log starts with the changes "
                           "after it. (The 📆 Daily P&L toggle below charts "
                           "every day, quiet ones included.)")
                if not _dtl:
                    st.info("The book never changed in this window — no daily "
                            "trades to list.")
                else:
                    def _dtl_chip(a):
                        _cc = C_BUY if a["delta"] > 0 else C_EXIT
                        if a["action"] == "buy":
                            _lbl = f"{a['key']} <b>bought +{a['w1']*100:.1f}%</b>"
                        elif a["action"] == "sell":
                            _lbl = f"{a['key']} <b>sold −{a['w0']*100:.1f}%</b>"
                        else:
                            _lbl = (f"{a['key']} {a['w0']*100:.1f}→"
                                    f"{a['w1']*100:.1f}% "
                                    f"<b>({a['delta']*100:+.1f})</b>")
                        # every tilt-column sale — partial trim or a re-size
                        # through zero — shows what the slice realized
                        if (a["delta"] < 0 and not a["signal_change"]
                                and a.get("pnl") is not None
                                and a["pnl"] > -1):
                            # realized P&L on the slice sold: proceeds
                            # |Δw|·💼 minus their average-cost basis
                            _sold = abs(a["delta"]) * portfolio_value
                            _pd = _sold - _sold / (1 + a["pnl"])
                            _pc = C_BUY if a["pnl"] >= 0 else C_EXIT
                            _lbl += (f" · <span style='color:{_pc}'>"
                                     f"P&amp;L {a['pnl']*100:+.1f}% "
                                     f"≈ ${_pd:+,.0f}</span>")
                        return f"<span style='color:{_cc}'>{_lbl}</span>"

                    _dh = ("<tr style='background:#f1f5f9;font-size:12px;"
                           "text-align:left'>"
                           "<th style='padding:6px 10px'>Executed at close</th>"
                           "<th style='padding-left:10px'>🚦 Signal changes "
                           "(bought / sold)</th>"
                           "<th style='padding-left:10px'>⚖️ Daily tilt "
                           "adjustments (re-sizes)</th>"
                           "<th style='text-align:right'>Turnover</th>"
                           "<th style='text-align:right'>≈ $ traded</th>"
                           "<th style='text-align:right'>Day P&amp;L</th></tr>")
                    _drs = []
                    for _d in _dtl:
                        _sig = " · ".join(_dtl_chip(a) for a in _d["actions"]
                                          if a["signal_change"])
                        _tilt = " · ".join(_dtl_chip(a) for a in _d["actions"]
                                           if not a["signal_change"])
                        _none = "<span style='color:#94a3b8'>—</span>"
                        _pl_s = _none
                        if _dpl is not None and _d["date"] in _dpl["total"].index:
                            _pl_v = float(_dpl["total"].loc[_d["date"]]) \
                                * portfolio_value
                            _pl_r = float(_dpl["ret"].loc[_d["date"]])
                            _pl_c = C_BUY if _pl_v >= 0 else C_EXIT
                            _pl_s = (f"<span style='color:{_pl_c};"
                                     f"font-weight:700'>${_pl_v:+,.0f}</span>"
                                     f"<div style='font-size:10px;"
                                     f"color:#94a3b8'>{_pl_r*100:+.2f}%</div>")
                        _drs.append(
                            f"<tr style='border-bottom:1px solid #eef2f7'>"
                            f"<td style='padding:6px 10px;font-weight:700;"
                            f"font-variant-numeric:tabular-nums;"
                            f"white-space:nowrap'>"
                            f"{_d['date'].strftime('%b %d, %Y')}</td>"
                            f"<td style='padding:6px 10px;font-size:11.5px'>"
                            f"{_sig or _none}</td>"
                            f"<td style='padding:6px 10px;font-size:11.5px'>"
                            f"{_tilt or _none}</td>"
                            f"<td style='text-align:right;font-weight:600'>"
                            f"{_d['turnover']*100:.1f}%</td>"
                            f"<td style='text-align:right;"
                            f"font-variant-numeric:tabular-nums'>"
                            f"${_d['gross']*portfolio_value:,.0f}</td>"
                            f"<td style='text-align:right;"
                            f"font-variant-numeric:tabular-nums'>{_pl_s}</td>"
                            f"</tr>")
                    st.markdown(
                        f"<div style='overflow-x:auto;max-height:440px;"
                        f"overflow-y:auto;margin:8px 0;'>"
                        f"<table style='width:100%;border-collapse:collapse'>"
                        f"{_dh}{''.join(_drs)}</table></div>",
                        unsafe_allow_html=True)

            # ── realized P&L by asset — what the sell tickets locked in ──────
            # (the 🧾 trade log's ⚖️ tilt trims + 🚦 sell signals rolled up
            # per sleeve — realized-only, per-trade approximation: it does
            # NOT reconcile to the exact 📊 P&L-by-asset attribution)
            if st.toggle(
                    f"💰 Realized P&L by asset {_win_lbl} — % and $ locked "
                    "in by daily tilt trims & sell signals (toggle to show)",
                    key="overall_realized_pnl_by_asset"):
                _rlz = ov.realized_pnl_by_asset(
                    _wf["weights"], _wf["sata"], _start_sel, _rets_win,
                    active=_wf.get("active"))
                if not _rlz:
                    st.info("No position was trimmed or sold in this window — "
                            "nothing realized yet.")
                else:
                    st.caption("**Only the P&L locked in by sales**: every "
                               "**⚖️ daily tilt trim** and **🚦 sell-signal "
                               "close** the "
                               + ("**as-published books**" if _actual
                                  else "**walk-forward replay**")
                               + " ordered since the start date, rolled up "
                               "per asset — the 🧾 trade log's sell tickets "
                               "aggregated. Each bar sums slice proceeds "
                               "minus their **average-cost basis** (tilt "
                               "adds raise the basis at the price actually "
                               "paid), scaled "
                               "by the 💼 portfolio value; the % label is "
                               "the cost-weighted realized return "
                               "(Σproceeds/Σcost − 1). Hover for the "
                               "tilt-vs-signal split, trade counts and the "
                               "dollars sold. **Unrealized P&L on positions "
                               "still held is NOT included**, and dollars "
                               "use the trade-ticket convention "
                               "(un-compounded, static 💼 scaling) — so "
                               "these bars deliberately do **not** sum to "
                               "the exact 📊 P&L-by-asset attribution "
                               "below, which counts every day's earnings, "
                               "held or sold, off the compounding blend "
                               "curve.")
                    _rb = []
                    for k, r in _rlz.items():
                        _em = (by_key.get(k)
                               or ov.ASSET_META.get(k, {})).get("emoji", "")
                        _n_t, _n_s = r["n_trims"], r["n_sells"]
                        _rb.append((
                            f"{_em} {k}".strip(),
                            r["pnl"] * portfolio_value,
                            r["ret"],
                            C_BUY if r["pnl"] >= 0 else C_EXIT,
                            f"{_n_t} tilt trim{'s' if _n_t != 1 else ''} "
                            f"(${r['pnl_tilt']*portfolio_value:+,.0f}) · "
                            f"{_n_s} signal sell{'s' if _n_s != 1 else ''} "
                            f"(${r['pnl_signal']*portfolio_value:+,.0f}) · "
                            f"${r['proceeds']*portfolio_value:,.0f} sold"))
                    _rb.sort(key=lambda b: b[1])
                    _rlz_tot = sum(b[1] for b in _rb)
                    _rlz_cost = sum(r["cost"] for r in _rlz.values())
                    _rlz_ret = (sum(r["proceeds"] for r in _rlz.values())
                                / _rlz_cost - 1) if _rlz_cost > 0 else 0.0
                    fig_rlz = go.Figure(go.Bar(
                        y=[b[0] for b in _rb],
                        x=[b[1] for b in _rb],
                        orientation="h",
                        marker_color=[b[3] for b in _rb],
                        text=[f"${v:+,.0f} · {rr*100:+.1f}%"
                              for _, v, rr, _c, _h in _rb],
                        textposition="outside", cliponaxis=False,
                        customdata=[[b[4]] for b in _rb],
                        hovertemplate="%{y}: <b>$%{x:+,.0f}</b><br>"
                                      "%{customdata[0]}<extra></extra>"))
                    fig_rlz.add_vline(x=0, line_color="#94a3b8", line_width=1)
                    fig_rlz.update_layout(
                        height=max(280, 26 * len(_rb) + 80),
                        margin=dict(t=40, b=10, l=10, r=110),
                        xaxis_title="$ realized on sales since start",
                        title=dict(text=f"Realized P&L per asset — "
                                        f"${_rlz_tot:+,.0f} locked in "
                                        f"({_rlz_ret*100:+.1f}% on the "
                                        f"capital sold)",
                                   font_size=13))
                    st.plotly_chart(fig_rlz, use_container_width=True)

            # ── daily P&L — what the book made or lost, every single day ─────
            # (both sources, off the same day-axis attribution as the column
            # above — bars sum exactly to the headline Strategy P&L dollars)
            if st.toggle(
                    f"📆 Daily P&L {_win_lbl} — every day's $ "
                    "and % result, split by sleeve on hover (toggle on/off)",
                    key="overall_daily_pnl"):
                _dpl = _get_dpl()
                if _dpl is None:
                    st.info("Not enough blend history after that date — "
                            "no daily P&L to chart.")
                else:
                    st.caption("What the "
                               + ("**as-published book**" if _actual
                                  else "**walk-forward replay book**")
                               + " (per the performance-source toggle above) "
                               "**earned or lost each trading day**: green "
                               "bars are up days, red bars down days, "
                               "**every day included** (unlike the 🧾 trade "
                               "log, which lists only days that traded). "
                               "Each day's blend return is scaled by the "
                               "book's compounded value at the previous "
                               "close, so the bars **sum exactly to the "
                               "headline Strategy P&L dollar figure** for "
                               "this source and start date. Hover any bar "
                               "for the day's **% return and per-sleeve "
                               "split** (largest contributors, SATA "
                               "included); the anchor bar is the cost basis "
                               "and earns nothing. Dollar figures scale the "
                               "💼 portfolio value entered above.")
                    _dd = _dpl["total"].iloc[1:] * portfolio_value
                    _dr = _dpl["ret"].iloc[1:]
                    _dp_hov = []
                    for _dt in _dd.index:
                        _parts = [(k, float(_dpl["per_key"].at[_dt, k])
                                   * portfolio_value)
                                  for k in _dpl["per_key"].columns]
                        _parts = [p for p in _parts if abs(p[1]) >= 0.5]
                        _parts.sort(key=lambda p: -abs(p[1]))
                        _more = len(_parts) - 5
                        _txt = " · ".join(f"{k} ${v:+,.0f}"
                                          for k, v in _parts[:5])
                        if _more > 0:
                            _txt += f" · +{_more} more"
                        _sata_v = float(_dpl["sata"].loc[_dt]) * portfolio_value
                        if abs(_sata_v) >= 0.5:
                            _txt = ((_txt + " · ") if _txt else "") + \
                                f"💵 SATA ${_sata_v:+,.0f}"
                        _dp_hov.append(_txt or "book flat — no deployed P&L")
                    fig_dp = go.Figure(go.Bar(
                        x=_dd.index, y=_dd.to_numpy(),
                        marker_color=[C_BUY if v >= 0 else C_EXIT
                                      for v in _dd],
                        customdata=[[r * 100, h]
                                    for r, h in zip(_dr, _dp_hov)],
                        hovertemplate="%{x|%b %d, %Y}: <b>$%{y:+,.0f}</b> "
                                      "(%{customdata[0]:+.2f}%)<br>"
                                      "%{customdata[1]}<extra></extra>"))
                    fig_dp.add_hline(y=0, line_color="#94a3b8", line_width=1)
                    _n_up = int((_dd > 0).sum())
                    _n_all = int(len(_dd))
                    fig_dp.update_layout(
                        height=340, margin=dict(t=30, b=10, l=10, r=10),
                        yaxis_title="Day P&L ($)",
                        title=dict(
                            text=f"Daily P&L on ${portfolio_value:,.0f} — "
                                 f"sums to ${_dd.sum():+,.0f} · "
                                 f"{_n_up}/{_n_all} days up",
                            font_size=13))
                    st.plotly_chart(fig_dp, use_container_width=True)
                    _bd, _wd = _dd.idxmax(), _dd.idxmin()
                    st.caption(f"Best day **{_bd.strftime('%b %d, %Y')}** "
                               f"(${_dd.loc[_bd]:+,.0f} · "
                               f"{_dr.loc[_bd]*100:+.2f}%) · worst day "
                               f"**{_wd.strftime('%b %d, %Y')}** "
                               f"(${_dd.loc[_wd]:+,.0f} · "
                               f"{_dr.loc[_wd]*100:+.2f}%). Weekend bars "
                               "carry the crypto sleeves' P&L only (equity "
                               "sleeves hold through closed markets; SATA "
                               "accrues on business days).")

            # ── published books & daily trims — the record, book by book ─────
            # (as-published mode only: this IS the archive, one row per publish)
            if _actual:
                _bk_all = _bookrep["books"]
                _bk_win = [b for b in _bk_all
                           if _sm["start"].date() <= b["as_of"].date()
                           <= _sm["end"].date()]
                # include the book driving the anchor bar (published before it)
                _bk_pre = [b for b in _bk_all
                           if b["as_of"].date() < _sm["start"].date()]
                if _bk_pre:
                    _bk_win = [_bk_pre[-1]] + _bk_win
                _turns = [b["turnover"] for b in _bk_win
                          if b.get("turnover") is not None]
                _avg_to = (sum(_turns) / len(_turns)) if _turns else 0.0
                if st.toggle(
                        f"🔁 Published books & daily trims {_win_lbl} — "
                        f"{len(_bk_win)} book{'s' if len(_bk_win) != 1 else ''}"
                        + (f" · avg daily turnover {_avg_to*100:.1f}%"
                           if _turns else "")
                        + " (toggle to show)",
                        key="overall_published_books"):
                    st.caption("One row per **archived publish** (newest "
                               "first): the book's signal bar, the risk "
                               "profile it was actually published under, the "
                               "**strategy-logic version** that produced it "
                               "(with the publishing commit), "
                               "deployed vs idle capital, its **one-way "
                               "turnover** vs the previous book (½·Σ|Δweight|, "
                               "cash leg included — the size of that "
                               "morning's optimizer trims), and every "
                               "per-asset weight change. These books are "
                               "exactly what the P&L above compounds, so all "
                               "the trims listed here are already inside "
                               "every metric in this section.")
                    _bh = ("<tr style='background:#f1f5f9;font-size:12px;"
                           "text-align:left'>"
                           "<th style='padding:6px 10px'>Signal bar</th>"
                           "<th>Profile</th>"
                           "<th>Logic</th>"
                           "<th style='text-align:right'>Deployed</th>"
                           "<th style='text-align:right'>💵 Cash</th>"
                           "<th style='text-align:right'>Turnover</th>"
                           "<th style='padding-left:10px'>Weight changes vs "
                           "previous book</th></tr>")
                    _brs = []
                    for _prev_b, _b in zip([None] + _bk_win[:-1], _bk_win):
                        _dep = sum(_b["weights"].values())
                        _pw = (_prev_b or {}).get("weights", {})
                        _chg = []
                        for _k in sorted(set(_b["weights"]) | set(_pw),
                                         key=lambda k: -abs(_b["weights"].get(k, 0.0)
                                                            - _pw.get(k, 0.0))):
                            _w0, _w1 = _pw.get(_k, 0.0), _b["weights"].get(_k, 0.0)
                            _d = _w1 - _w0
                            if _prev_b is None:
                                _chg.append(f"{_k} {_w1*100:.1f}%")
                                continue
                            if abs(_d) < 0.0005:
                                continue
                            _cc = C_BUY if _d > 0 else C_EXIT
                            if _w0 < 0.0005:
                                _lbl = f"{_k} <b>new +{_w1*100:.1f}%</b>"
                            elif _w1 < 0.0005:
                                _lbl = f"{_k} <b>closed −{_w0*100:.1f}%</b>"
                            else:
                                _lbl = (f"{_k} {_w0*100:.1f}→{_w1*100:.1f}% "
                                        f"<b>({_d*100:+.1f})</b>")
                            _chg.append(f"<span style='color:{_cc}'>{_lbl}</span>")
                        _chg_s = (" · ".join(_chg) if _chg else
                                  "<span style='color:#94a3b8'>unchanged</span>")
                        if _prev_b is None:
                            _chg_s = ("<span style='color:#64748b'>opening book: "
                                      f"{' · '.join(_chg) or '—'}</span>")
                        _to = _b.get("turnover")
                        _to_s = ("—" if _prev_b is None or _to is None
                                 else f"{_to*100:.1f}%")
                        _brs.append(
                            f"<tr style='border-bottom:1px solid #eef2f7'>"
                            f"<td style='padding:6px 10px;font-weight:700;"
                            f"font-variant-numeric:tabular-nums'>"
                            f"{_b['as_of'].strftime('%b %d, %Y')}</td>"
                            f"<td><code>{_b['profile']}</code>"
                            f"<span style='font-size:10px;color:#94a3b8'> "
                            f"{_b['book_mode']}</span></td>"
                            f"<td><span style='background:{_sv.BADGE_COLOR};"
                            f"color:#fff;font-weight:800;font-size:11px;"
                            f"letter-spacing:.04em;padding:2px 9px;"
                            f"border-radius:999px;white-space:nowrap'>"
                            f"{_b['version'].upper()}</span>"
                            + (f"<div style='font-size:10px;color:#94a3b8'>"
                               f"{_b['code_sha']}</div>" if _b.get("code_sha")
                               else "") + "</td>"
                            f"<td style='text-align:right'>{_dep*100:.1f}%</td>"
                            f"<td style='text-align:right;color:#64748b'>"
                            f"{_b['cash']*100:.1f}%</td>"
                            f"<td style='text-align:right;font-weight:600'>{_to_s}</td>"
                            f"<td style='padding:6px 10px;font-size:11.5px'>"
                            f"{_chg_s}</td></tr>")
                    st.markdown(f"<div style='overflow-x:auto;margin:8px 0;'>"
                                f"<table style='width:100%;border-collapse:collapse'>"
                                f"{_bh}{''.join(reversed(_brs))}</table></div>",
                                unsafe_allow_html=True)

            # ── per-asset breakdown — opt-in via toggle so the page stays clean
            # (a nested st.expander is not allowed inside the section expander)
            # (_pa computed above, alongside the blend-level trade stats)
            if st.toggle(f"🔬 Per-asset breakdown {_win_lbl} — every sleeve's "
                         "return & risk (toggle to show)",
                         key="overall_pa_breakdown"):
                st.caption("Each instrument's **own signal-driven strategy** vs "
                           "simply buying & holding it, both measured from the same "
                           "start date. **In mkt** = share of days the sleeve was "
                           "actually long (it earns SATA when flat inside the blend); "
                           "**Peak wt** = the largest daily allocation the selected "
                           "book (as-published or replay, per the source toggle "
                           "above) ever gave it. Sleeves whose data begins after the "
                           "start date are measured from their own first bar "
                           "(noted inline).")
                if not _pa:
                    st.info("No instrument has enough history after that date.")
                else:
                    # quick visual: strategy vs buy & hold return per sleeve
                    _pa_sorted = sorted(_pa, key=lambda x: x["strat"]["total_ret"])
                    fig_pa = go.Figure()
                    fig_pa.add_trace(go.Bar(
                        y=[p["key"] for p in _pa_sorted],
                        x=[p["strat"]["total_ret"] * 100 for p in _pa_sorted],
                        name="Strategy", orientation="h",
                        marker_color=[p["accent"] for p in _pa_sorted],
                        text=[f"{p['strat']['total_ret']*100:+.1f}%" for p in _pa_sorted],
                        textposition="outside", cliponaxis=False))
                    fig_pa.add_trace(go.Bar(
                        y=[p["key"] for p in _pa_sorted],
                        x=[(p["bh"]["total_ret"] * 100 if p["bh"] else 0) for p in _pa_sorted],
                        name="Buy & Hold", orientation="h", marker_color="#cbd5e1"))
                    fig_pa.update_layout(
                        barmode="group", height=max(300, 26 * len(_pa_sorted) + 80),
                        margin=dict(t=30, b=10, l=10, r=40),
                        xaxis_title="return since start (%)",
                        legend=dict(orientation="h", y=1.06),
                        title=dict(text="Return by sleeve — strategy vs buy & hold",
                                   font_size=13))
                    fig_pa.add_vline(x=0, line_color="#94a3b8", line_width=1)
                    st.plotly_chart(fig_pa, use_container_width=True)

                    # detail table, grouped by parent signal like the rest of the app
                    pah = ("<tr style='background:#f1f5f9;font-size:12px;text-align:left'>"
                           "<th style='padding:6px 10px'>Instrument</th>"
                           "<th style='text-align:right'>Strat P&amp;L</th>"
                           "<th style='text-align:right'>Buy&amp;Hold</th>"
                           "<th style='text-align:right'>Max DD</th>"
                           "<th style='text-align:right'>Sharpe</th>"
                           "<th style='text-align:right'>Trades</th>"
                           "<th style='text-align:right'>Win rate</th>"
                           "<th style='text-align:right'>Win days</th>"
                           "<th style='text-align:right'>In mkt</th>"
                           "<th style='text-align:right'>Peak wt</th></tr>")
                    par = []
                    _wts = _wmax
                    _pa_by_key = {p["key"]: p for p in _pa}
                    for pk, grp in parents:
                        for res in grp:
                            p = _pa_by_key.get(res["key"])
                            if not p:
                                continue
                            sm_a, bh_a = p["strat"], p["bh"]
                            beat = bh_a is None or sm_a["total_ret"] >= bh_a["total_ret"]
                            _rc = C_BUY if sm_a["total_ret"] >= 0 else C_EXIT
                            _late = (f"<div style='font-size:10px;color:#94a3b8'>since "
                                     f"{sm_a['start'].strftime('%b %d, %Y')}</div>"
                                     if sm_a["start"] > _sm["start"] else "")
                            _tr = p["trades"]
                            _tr_s = (f"{_tr['n_trades']}"
                                     + (f" <span style='color:{C_HOLD};font-size:10px'>"
                                        f"({_tr['n_open']} open)</span>"
                                        if _tr["n_open"] else ""))
                            _wr_s = ("—" if _tr["win_rate"] is None
                                     else f"{_tr['win_rate']*100:.0f}%"
                                          f" <span style='color:#94a3b8;font-size:10px'>"
                                          f"({_tr['wins']}/{_tr['n_trades']})</span>")
                            _dw_a = p.get("daily")
                            _wd_s = ("—" if not _dw_a or _dw_a["win_rate"] is None
                                     else f"{_dw_a['win_rate']*100:.0f}%"
                                          f" <span style='color:#94a3b8;font-size:10px'>"
                                          f"({_dw_a['wins']}/{_dw_a['n_active']})</span>")
                            par.append(
                                f"<tr style='border-bottom:1px solid #eef2f7'>"
                                f"<td style='padding:6px 10px;font-weight:700'>"
                                f"{p['emoji']} {p['key']}{_kind_badge(p['kind'])}"
                                f"<span style='font-size:11px;color:#94a3b8;font-weight:400'>"
                                f" {p['name']}</span>{_late}</td>"
                                f"<td style='text-align:right;font-weight:700;color:{_rc}'>"
                                f"{sm_a['total_ret']*100:+.1f}%"
                                f"{' 🏆' if beat and bh_a is not None else ''}</td>"
                                f"<td style='text-align:right;color:#64748b'>"
                                f"{_pct(bh_a['total_ret']*100) if bh_a else '—'}</td>"
                                f"<td style='text-align:right;color:{C_EXIT}'>{sm_a['mdd']*100:.1f}%</td>"
                                f"<td style='text-align:right'>{sm_a['sharpe']:.2f}</td>"
                                f"<td style='text-align:right;font-weight:600'>{_tr_s}</td>"
                                f"<td style='text-align:right;font-weight:600'>{_wr_s}</td>"
                                f"<td style='text-align:right'>{_wd_s}</td>"
                                f"<td style='text-align:right'>{p['in_market']*100:.0f}%</td>"
                                f"<td style='text-align:right;font-weight:700'>"
                                f"{_wts.get(p['key'], 0)*100:.1f}%</td></tr>")
                    # SATA — the idle-cash sleeve every flat day drains into
                    _sata_m = ov.sata_slice_metrics(_curve_all.index, _start_sel)
                    if _sata_m:
                        par.append(
                            f"<tr style='border-top:2px solid #cbd5e1;background:#f8fafc'>"
                            f"<td style='padding:6px 10px;font-weight:700'>💵 SATA"
                            f"<span style='font-size:11px;color:#64748b;font-weight:400'>"
                            f" {ov.SATA['name']}</span></td>"
                            f"<td style='text-align:right;font-weight:700;color:{C_BUY}'>"
                            f"{_sata_m['total_ret']*100:+.1f}%</td>"
                            f"<td style='text-align:right;color:#64748b'>"
                            f"{_sata_m['total_ret']*100:+.1f}%</td>"
                            f"<td style='text-align:right;color:#64748b'>0.0%</td>"
                            f"<td style='text-align:right;color:#94a3b8'>—</td>"
                            f"<td style='text-align:right;color:#94a3b8'>—</td>"
                            f"<td style='text-align:right;color:#94a3b8'>—</td>"
                            f"<td style='text-align:right'>100%</td>"
                            f"<td style='text-align:right'>100%</td>"
                            f"<td style='text-align:right;color:#94a3b8'>idle bal.</td></tr>")
                    st.markdown(f"<div style='overflow-x:auto;margin:8px 0;'>"
                                f"<table style='width:100%;border-collapse:collapse'>"
                                f"{pah}{''.join(par)}</table></div>", unsafe_allow_html=True)
                    st.caption("🏆 = the sleeve's strategy beat buying & holding it "
                               "over this window. **Trades** counts round-trips open "
                               "at any point since the start date (open positions "
                               "included); **Win rate** judges closed trades on their "
                               "realised return and the open one on its current "
                               "unrealised P&L, while **Win days** is the share of "
                               "positive daily returns counting only the days the "
                               "sleeve actually held its position — flat days can't "
                               "pad the rate. **SATA** is the idle-cash "
                               "park: a steady ~13%/yr daily yield on a flat $100 "
                               "par — no drawdown, no trades, its weight is simply "
                               "whatever the risk sleeves leave undeployed each day. "
                               "Per-sleeve returns exclude that SATA yield (it "
                               "accrues at the blend level), so the sleeves won't "
                               "sum to the blend P&L.")

            # ── trade log — the individual trades behind the counts above ──────
            _tl = ov.trade_log_since(results, _wmax, _start_sel,
                                     weight_matrix=_wf["weights"],
                                     end=_end_arg)
            _n_open_tl = sum(1 for t in _tl if t["open"])
            if st.toggle(
                    f"📜 Trade log {_win_lbl} — "
                    f"{len(_tl)} trade{'s' if len(_tl) != 1 else ''}"
                    f"{f' · {_n_open_tl} open' if _n_open_tl else ''} (toggle to show)",
                    key="overall_trade_log"):
                st.caption("Every round-trip across the sleeves the selected book "
                           "holds (weight > 0) that was open at any point since the "
                           "start date — including trades entered before it — plus "
                           "any **currently-open position** (highlighted; its return "
                           "is unrealised, marked to the latest price). Newest "
                           "first. **≈ $ on blend** scales each trade's return by "
                           "the sleeve's blend weight and the 💼 portfolio value — "
                           "an approximation that ignores compounding.")
                if not _tl:
                    st.info("No sleeve the blend holds traded in this window.")
                else:
                    tlh = ("<tr style='background:#f1f5f9;font-size:12px;text-align:left'>"
                           "<th style='padding:6px 10px'>Instrument</th>"
                           "<th>Status</th>"
                           "<th style='text-align:right'>Entry</th>"
                           "<th style='text-align:right'>Entry px</th>"
                           "<th style='text-align:right'>Exit</th>"
                           "<th style='text-align:right'>Exit / last px</th>"
                           "<th style='text-align:right'>Days</th>"
                           "<th style='text-align:right'>Return</th>"
                           "<th style='text-align:right'>≈ $ on blend</th>"
                           "<th style='padding-left:10px'>Exit reason</th></tr>")

                    def _tl_px(v):
                        return "—" if v is None else (f"${v:,.2f}" if v < 1000
                                                      else f"${v:,.0f}")

                    def _tl_dt(v):
                        return "—" if v is None else pd.Timestamp(v).strftime("%b %d, %Y")

                    tlr = []
                    for t in _tl:
                        _ret = t["ret"]
                        _rc = "#94a3b8" if _ret is None else (C_BUY if _ret >= 0 else C_EXIT)
                        _bg = "background:#fffbeb;" if t["open"] else ""
                        _status = ((f"<span style='background:{C_HOLD}22;color:{C_HOLD};"
                                    f"font-weight:700;font-size:10px;padding:1px 6px;"
                                    f"border-radius:6px'>OPEN</span>") if t["open"]
                                   else "<span style='color:#94a3b8;font-size:11px'>closed</span>")
                        _imp = None if _ret is None else _ret * t["weight"] * portfolio_value
                        _imp_s = ("—" if _imp is None else
                                  f"<span style='color:{C_BUY if _imp >= 0 else C_EXIT}'>"
                                  f"${_imp:+,.0f}</span>")
                        _reason = ("<span style='color:#94a3b8;font-size:11px'>— still open</span>"
                                   if t["open"] else t["reason"])
                        _unreal = ("<div style='font-size:10px;color:#94a3b8;"
                                   "font-weight:400'>unrealised</div>" if t["open"] else "")
                        tlr.append(
                            f"<tr style='border-bottom:1px solid #eef2f7;{_bg}'>"
                            f"<td style='padding:6px 10px;font-weight:700'>"
                            f"{t['emoji']} {t['key']}{_kind_badge(t['kind'])}"
                            f"<span style='font-size:11px;color:#94a3b8;font-weight:400'>"
                            f" wt {t['weight']*100:.1f}%</span></td>"
                            f"<td>{_status}</td>"
                            f"<td style='text-align:right;font-variant-numeric:tabular-nums'>"
                            f"{_tl_dt(t['entry_date'])}</td>"
                            f"<td style='text-align:right;font-variant-numeric:tabular-nums'>"
                            f"{_tl_px(t['entry_px'])}</td>"
                            f"<td style='text-align:right;font-variant-numeric:tabular-nums'>"
                            f"{_tl_dt(t['exit_date'])}</td>"
                            f"<td style='text-align:right;font-variant-numeric:tabular-nums'>"
                            f"{_tl_px(t['exit_px'])}</td>"
                            f"<td style='text-align:right'>{'—' if t['days'] is None else t['days']}</td>"
                            f"<td style='text-align:right;font-weight:700;color:{_rc}'>"
                            f"{_pct(_ret * 100 if _ret is not None else None)}{_unreal}</td>"
                            f"<td style='text-align:right;font-variant-numeric:tabular-nums'>{_imp_s}</td>"
                            f"<td style='padding-left:10px;font-size:11px;color:#64748b'>{_reason}</td></tr>")
                    st.markdown(f"<div style='overflow-x:auto;margin:8px 0;'>"
                                f"<table style='width:100%;border-collapse:collapse'>"
                                f"{tlh}{''.join(tlr)}</table></div>", unsafe_allow_html=True)

            # ── capital traded by asset — where the strategy put the money ─────
            if st.toggle(
                    f"🥧 Capital traded by asset {_win_lbl} — % of total "
                    "deployed capital (toggle to show)",
                    key="overall_capital_by_asset"):
                st.caption("Each round-trip in the trade log above (open positions "
                           "included) deploys its sleeve's blend weight of the 💼 "
                           "portfolio value at entry. Summing those entry notionals "
                           "per instrument shows **where the strategy actually put "
                           "capital to work** since the start date. The percentage "
                           "shares are independent of the portfolio value entered — "
                           "only the dollar figures scale with it.")
                if not _tl:
                    st.info("No sleeve the blend holds traded in this window — "
                            "nothing was deployed.")
                else:
                    _cap, _cnt, _tl_meta = {}, {}, {}
                    for t in _tl:
                        _cap[t["key"]] = _cap.get(t["key"], 0.0) + t["weight"] * portfolio_value
                        _cnt[t["key"]] = _cnt.get(t["key"], 0) + 1
                        _tl_meta[t["key"]] = t
                    _ck = sorted(_cap, key=_cap.get, reverse=True)
                    _cap_tot = sum(_cap.values())
                    # fold sleeves under 2.5% of the total into one "Other" slice so
                    # the pie stays readable — their split lives in the Other hover
                    _big = [k for k in _ck if _cap[k] / _cap_tot >= 0.025]
                    _small = [k for k in _ck if k not in _big]
                    if len(_small) == 1:          # folding a single sleeve saves nothing
                        _big, _small = _ck, []
                    _lbls = [f"{_tl_meta[k]['emoji']} {k}" for k in _big]
                    _vals = [_cap[k] for k in _big]
                    _cols = [_tl_meta[k]["accent"] for k in _big]
                    _hov = [f"{_cnt[k]} trade{'s' if _cnt[k] != 1 else ''} · peak wt "
                            f"{_tl_meta[k]['weight']*100:.1f}%" for k in _big]
                    if _small:
                        _lbls.append(f"Other ({len(_small)} sleeves)")
                        _vals.append(sum(_cap[k] for k in _small))
                        _cols.append("#94a3b8")
                        _hov.append("<br>".join(
                            f"{_tl_meta[k]['emoji']} {k}: {_cap[k]/_cap_tot*100:.1f}%"
                            f" · {_cnt[k]} trade{'s' if _cnt[k] != 1 else ''}"
                            for k in _small))
                    fig_cap = go.Figure(go.Pie(
                        labels=_lbls, values=_vals,
                        customdata=[[h] for h in _hov],
                        marker=dict(colors=_cols, line=dict(color="#ffffff", width=2)),
                        hole=0.45, sort=False, direction="clockwise",
                        textinfo="label+percent",
                        hovertemplate="%{label}: <b>%{percent}</b> of deployed "
                                      "capital · ≈ $%{value:,.0f}<br>"
                                      "%{customdata[0]}<extra></extra>"))
                    fig_cap.update_layout(
                        height=430, margin=dict(t=40, b=10, l=10, r=10),
                        title=dict(text=f"Share of deployed capital per instrument — "
                                        f"total ≈ ${_cap_tot:,.0f} across {len(_tl)} "
                                        f"trade{'s' if len(_tl) != 1 else ''}",
                                   font_size=13))
                    st.plotly_chart(fig_cap, use_container_width=True)

            # ── P&L by asset — exact attribution of the blend's P&L ──────────
            if st.toggle(
                    f"📊 P&L by asset {_win_lbl} — how much each "
                    "instrument earned or lost (toggle to show)",
                    key="overall_pnl_share_by_asset"):
                st.caption("The headline **Strategy P&L above, split by sleeve**: "
                           "each day's blend return is decomposed into every "
                           "sleeve's weighted share (plus the **SATA** yield on "
                           "idle capital) and scaled by the blend's compounding "
                           "value, so the bars **sum exactly to the Strategy "
                           "P&L dollar figure** for the selected source and "
                           "start date. Green bars are net winners, red bars "
                           "net losers; each is labelled with its share of the "
                           "net total. (This is exact attribution off the blend "
                           "curve — unlike the trade log's **≈ $ on blend** "
                           "column, which is a per-trade approximation.)")
                _att = ov.pnl_attribution_replay(
                    _rets_win, _wf["weights"], _wf["sata"], _start_sel,
                    sata_daily=ov.SATA_DAILY)
                if _att is None:
                    st.info("Not enough blend history after that date — "
                            "no P&L to attribute.")
                else:
                    _pcnt = {}
                    for t in _tl:
                        _pcnt[t["key"]] = _pcnt.get(t["key"], 0) + 1
                    _pnl_by = {k: v * portfolio_value
                               for k, v in _att["per_key"].items()}
                    _net_tot = (sum(_pnl_by.values())
                                + _att["sata"] * portfolio_value)
                    # hide sleeves that contributed nothing (zero-weight);
                    # totals above already include their (≈ $0) share
                    _pnl_by = {k: v for k, v in _pnl_by.items() if abs(v) >= 0.5}
                    _bars = [(f"{by_key[k]['emoji']} {k}", v,
                              C_BUY if v >= 0 else C_EXIT,
                              f"{_pcnt.get(k, 0)} trade"
                              f"{'s' if _pcnt.get(k, 0) != 1 else ''} · peak wt "
                              f"{_wmax.get(k, 0)*100:.1f}%")
                             for k, v in _pnl_by.items()]
                    if _att["sata"]:
                        _bars.append(("💵 SATA", _att["sata"] * portfolio_value,
                                      "#334155", "idle-cash yield on undeployed "
                                      "capital · no trades"))
                    _win_tot = sum(v for _, v, _c, _h in _bars if v > 0)
                    _loss_tot = sum(v for _, v, _c, _h in _bars if v < 0)
                    # biggest earner on top, biggest loser at the bottom
                    _bars.sort(key=lambda b: b[1])
                    # share of the net total only reads sensibly when positive
                    _txt = [f"${v:+,.0f}"
                            + (f" · {v/_net_tot*100:+.0f}% of net"
                               if _net_tot > 0 else "")
                            for _, v, _c, _h in _bars]
                    fig_pnl_by = go.Figure(go.Bar(
                        y=[b[0] for b in _bars],
                        x=[b[1] for b in _bars],
                        orientation="h",
                        marker_color=[b[2] for b in _bars],
                        text=_txt, textposition="outside", cliponaxis=False,
                        customdata=[[b[3]] for b in _bars],
                        hovertemplate="%{y}: <b>$%{x:+,.0f}</b><br>"
                                      "%{customdata[0]}<extra></extra>"))
                    fig_pnl_by.add_vline(x=0, line_color="#94a3b8", line_width=1)
                    fig_pnl_by.update_layout(
                        height=max(300, 26 * len(_bars) + 80),
                        margin=dict(t=40, b=10, l=10, r=90),
                        xaxis_title="$ P&L on blend since start",
                        title=dict(text=f"P&L earned per instrument — "
                                        f"${_win_tot:+,.0f} profits"
                                        + (f" · ${_loss_tot:+,.0f} losses"
                                           if _loss_tot < 0 else "")
                                        + f" · net ${_net_tot:+,.0f} "
                                          f"({_att['total']*100:+.1f}%)",
                                   font_size=13))
                    st.plotly_chart(fig_pnl_by, use_container_width=True)

            # ── capital efficiency — P&L per dollar of capital deployed ──────
            if st.toggle(
                    f"⚡ Capital efficiency by asset {_win_lbl} — P&L earned "
                    "per $ of capital traded (toggle to show)",
                    key="overall_capital_efficiency"):
                st.caption("Each sleeve's **exact-attribution P&L** (the chart "
                           "above) divided by the **total capital it deployed** "
                           "— the sum of every round-trip's entry notional "
                           "(blend weight × 💼 portfolio value, as in the 🥧 "
                           "capital-traded chart; a sleeve that traded several "
                           "times redeploys capital, so its denominator can "
                           "exceed its blend weight). A high % means the sleeve "
                           "earned a lot per dollar it put at risk; the ratio "
                           "is independent of the portfolio value entered. "
                           "**SATA** isn't shown — its yield accrues on idle "
                           "cash, not traded capital.")
                _eff_att = ov.pnl_attribution_replay(
                    _rets_win, _wf["weights"], _wf["sata"], _start_sel,
                    sata_daily=ov.SATA_DAILY)
                if not _tl or _eff_att is None:
                    st.info("No sleeve the blend holds traded in this window — "
                            "nothing was deployed.")
                else:
                    _ecap, _ecnt = {}, {}
                    for t in _tl:
                        _ecap[t["key"]] = (_ecap.get(t["key"], 0.0)
                                           + t["weight"] * portfolio_value)
                        _ecnt[t["key"]] = _ecnt.get(t["key"], 0) + 1
                    _ebars = []
                    for k, cap in _ecap.items():
                        if cap <= 0:
                            continue
                        _pnl_k = _eff_att["per_key"].get(k, 0.0) * portfolio_value
                        _ebars.append((
                            f"{by_key[k]['emoji']} {k}", _pnl_k / cap * 100,
                            C_BUY if _pnl_k >= 0 else C_EXIT,
                            f"${_pnl_k:+,.0f} P&L on ${cap:,.0f} deployed · "
                            f"{_ecnt[k]} trade{'s' if _ecnt[k] != 1 else ''}"))
                    # most efficient sleeve on top, least efficient at the bottom
                    _ebars.sort(key=lambda b: b[1])
                    fig_eff = go.Figure(go.Bar(
                        y=[b[0] for b in _ebars],
                        x=[b[1] for b in _ebars],
                        orientation="h",
                        marker_color=[b[2] for b in _ebars],
                        text=[f"{b[1]:+.1f}%" for b in _ebars],
                        textposition="outside", cliponaxis=False,
                        customdata=[[b[3]] for b in _ebars],
                        hovertemplate="%{y}: <b>%{x:+.1f}%</b><br>"
                                      "%{customdata[0]}<extra></extra>"))
                    fig_eff.add_vline(x=0, line_color="#94a3b8", line_width=1)
                    fig_eff.update_layout(
                        height=max(300, 26 * len(_ebars) + 80),
                        margin=dict(t=40, b=10, l=10, r=70),
                        xaxis_title="P&L as % of capital deployed since start",
                        title=dict(text="Capital efficiency per instrument — "
                                        "attribution P&L ÷ capital traded",
                                   font_size=13))
                    st.plotly_chart(fig_eff, use_container_width=True)

            # ── win rate per asset — how often each sleeve's trades won ──────
            if st.toggle(
                    f"🎯 Win rate by asset {_win_lbl} — share of winning "
                    "trades per instrument (toggle to show)",
                    key="overall_win_rate_by_asset"):
                st.caption("Winning trades out of all counted trades per sleeve "
                           "the blend holds, since the start date — the same "
                           "per-sleeve stats behind the blend-level **Win rate** "
                           "metric above: closed trades judged on their realised "
                           "return, an open position on its current unrealised "
                           "P&L. Bars are green at ≥ 50%, red below; the dotted "
                           "line marks the 50% coin-flip. A sleeve with only a "
                           "trade or two can sit at 0% or 100% on tiny evidence "
                           "— the label shows the wins/trades count behind each "
                           "bar.")
                _wts_wr = _wmax
                _wr_bars = []
                for p in _pa:
                    tr = p["trades"]
                    if _wts_wr.get(p["key"], 0) <= 0.002 or not tr["n_trades"]:
                        continue
                    _wr = tr["win_rate"] * 100
                    _wr_bars.append((
                        f"{p['emoji']} {p['key']}", _wr,
                        C_BUY if _wr >= 50 else C_EXIT,
                        f"{_wr:.0f}% ({tr['wins']}/{tr['n_trades']})",
                        f"{tr['wins']}/{tr['n_trades']} trades won"
                        + (f" · {tr['n_open']} open" if tr["n_open"] else "")))
                if not _wr_bars:
                    st.info("No sleeve the blend holds traded in this window — "
                            "no win rate to measure.")
                else:
                    # best hit-rate on top, worst at the bottom
                    _wr_bars.sort(key=lambda b: b[1])
                    fig_wr = go.Figure(go.Bar(
                        y=[b[0] for b in _wr_bars],
                        x=[b[1] for b in _wr_bars],
                        orientation="h",
                        marker_color=[b[2] for b in _wr_bars],
                        text=[b[3] for b in _wr_bars],
                        textposition="outside", cliponaxis=False,
                        customdata=[[b[4]] for b in _wr_bars],
                        hovertemplate="%{y}: <b>%{x:.0f}%</b><br>"
                                      "%{customdata[0]}<extra></extra>"))
                    fig_wr.add_vline(x=50, line_dash="dot", line_color="#94a3b8",
                                     line_width=1, annotation_text="50%",
                                     annotation_font_size=10)
                    fig_wr.update_layout(
                        height=max(300, 26 * len(_wr_bars) + 80),
                        margin=dict(t=40, b=10, l=10, r=90),
                        xaxis=dict(title="share of winning trades since start (%)",
                                   range=[0, 112]),
                        title=dict(text="Win rate per instrument — trades won "
                                        "since start (open positions on "
                                        "unrealised P&L)",
                                   font_size=13))
                    st.plotly_chart(fig_wr, use_container_width=True)

            # ── trade efficiency — each trade's average share of total P&L ───
            if st.toggle(
                    f"💡 Trade efficiency by asset {_win_lbl} — average % of "
                    "total P&L earned per trade (toggle to show)",
                    key="overall_trade_efficiency"):
                st.caption("**(P&L per trade ÷ total P&L) × 100** for each "
                           "sleeve: its **exact-attribution P&L** (the 📊 chart "
                           "above) divided by its **number of counted trades** "
                           "since the start date (closed trades plus any open "
                           "position — the same counts as the 🎯 win-rate "
                           "chart), expressed as a share of the blend's **net "
                           "total P&L**. A 5% bar means one average trade in "
                           "that sleeve moved the whole strategy's P&L by 5%; "
                           "the ratio is independent of the 💼 portfolio value. "
                           "**SATA** isn't shown — its yield accrues daily on "
                           "idle cash, not from trades.")
                _pe_att = ov.pnl_attribution_replay(
                    _rets_win, _wf["weights"], _wf["sata"], _start_sel,
                    sata_daily=ov.SATA_DAILY)
                _wts_pe = _wmax
                _pe_bars = []
                if _pe_att and _pe_att["total"] > 0:
                    for p in _pa:
                        tr = p["trades"]
                        if _wts_pe.get(p["key"], 0) <= 0.002 or not tr["n_trades"]:
                            continue
                        _pnl_k = (_pe_att["per_key"].get(p["key"], 0.0)
                                  * portfolio_value)
                        _eff_pct = (_pe_att["per_key"].get(p["key"], 0.0)
                                    / tr["n_trades"] / _pe_att["total"] * 100)
                        _pe_bars.append((
                            f"{p['emoji']} {p['key']}", _eff_pct,
                            C_BUY if _eff_pct >= 0 else C_EXIT,
                            f"${_pnl_k:+,.0f} P&L over {tr['n_trades']} trade"
                            f"{'s' if tr['n_trades'] != 1 else ''}"
                            + (f" · {tr['n_open']} open" if tr["n_open"] else "")
                            + f" = ${_pnl_k/tr['n_trades']:+,.0f}/trade"))
                if _pe_att and _pe_att["total"] <= 0:
                    st.info("The blend's net P&L is not positive over this "
                            "window, so a per-trade **share of total P&L** "
                            "isn't meaningful — pick a different start date.")
                elif not _pe_bars:
                    st.info("No sleeve the blend holds traded in this window — "
                            "no per-trade P&L to measure.")
                else:
                    # most efficient sleeve on top, worst at the bottom
                    _pe_bars.sort(key=lambda b: b[1])
                    fig_pe = go.Figure(go.Bar(
                        y=[b[0] for b in _pe_bars],
                        x=[b[1] for b in _pe_bars],
                        orientation="h",
                        marker_color=[b[2] for b in _pe_bars],
                        text=[f"{b[1]:+.1f}%/trade" for b in _pe_bars],
                        textposition="outside", cliponaxis=False,
                        customdata=[[b[3]] for b in _pe_bars],
                        hovertemplate="%{y}: <b>%{x:+.1f}% of total P&L "
                                      "per trade</b><br>"
                                      "%{customdata[0]}<extra></extra>"))
                    fig_pe.add_vline(x=0, line_color="#94a3b8", line_width=1)
                    fig_pe.update_layout(
                        height=max(300, 26 * len(_pe_bars) + 80),
                        margin=dict(t=40, b=10, l=10, r=100),
                        xaxis_title="average % of total P&L per trade since start",
                        title=dict(text="Trade efficiency per instrument — "
                                        "(P&L per trade ÷ total P&L) × 100",
                                   font_size=13))
                    st.plotly_chart(fig_pe, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — HISTORICAL VIEW (the strategy as it stood on a chosen date)
# ══════════════════════════════════════════════════════════════════════════
with tab_hist:
    st.markdown("### 🕰️ Historical View — the strategy as it stood on a chosen date")
    st.caption("Pick any date from the calendar and see the overall strategy "
               "exactly as it stood at that day's completed close — the positions "
               "it held (with entry, days held and unrealised P&L **marked at that "
               "bar**, not at today's price), what it opened and closed on that "
               "bar, the book it was holding and its performance up to that point. "
               "Everything is read off the committed back-test series — the same "
               "engines the 🔴 Live tab runs — so the positions and trades shown are "
               "the engine's genuine bar-by-bar decisions. The book percentages come "
               "from the **walk-forward gated replay**: anchors fit only on data "
               "before that date, priority tilt from as-of inputs — an honest as-of "
               "reconstruction (per-sleeve strategy parameters remain today's). "
               "Where a Targetbook was actually **published** from the chosen "
               "bar's signals, the **Executed Playbook** section below shows "
               "it — the record of what actually traded. "
               "Follows the "
               f"risk profile (currently **`{_profile}`**) selected on the "
               "Live tab.")

    _h_curve = _PF["curves"][STRAT_CURVE]
    _h_d0, _h_d1 = _h_curve.index[0].date(), _h_curve.index[-1].date()
    _h_top = st.columns([1, 1, 2])
    with _h_top[0]:
        _h_sel = st.date_input(
            "📅 View date", value=_h_d1, min_value=_h_d0, max_value=_h_d1,
            key="overall_hist_date",
            help="Weekends and holidays roll back to each instrument's last "
                 "completed bar on or before this date. Data runs "
                 f"{_h_d0} → {_h_d1}.")
    with _h_top[1]:
        _h_pv = st.number_input(
            "💼 Portfolio value ($)", min_value=0.0,
            value=float(st.session_state.get("overall_portfolio_value", 100000.0)),
            step=1000.0, format="%.0f", key="overall_hist_pv",
            help="Book $ per instrument = book % × this value.")

    _snap = ov.snapshot_asof(results, _h_sel)
    if _snap is None:
        st.warning(f"No instrument has any strategy history on or before "
                   f"**{_h_sel}** — pick a later date (data runs "
                   f"{_h_d0} → {_h_d1}).")
    else:
        _h_bar = _snap["asof"]
        # the replayed book ON that bar — walk-forward anchors + as-of priority
        # tilt (falls back to the untilted water-fill for a bar the replay
        # doesn't cover, e.g. an instrument-only calendar day)
        _h_wf = _PF["wf"]
        if _h_bar in _h_wf["weights"].index:
            _h_wrow = _h_wf["weights"].loc[_h_bar]
            _h_book = {k: float(v) for k, v in _h_wrow.items() if v > 0.0005}
            _h_sata = float(_h_wf["sata"].loc[_h_bar])
        else:
            _h_alloc = ov.historical_allocation(_snap, opt["optimal"]["weights"],
                                                caps=ov.caps_for(_profile))
            _h_book, _h_sata = _h_alloc["book"], _h_alloc["sata"]
        with _h_top[2]:
            st.markdown(
                f"<div style='font-size:13px;color:#64748b;padding-top:30px'>"
                f"Signals bar used: <b>{_h_bar.strftime('%b %d, %Y')}</b> (daily "
                f"close) · {len(_snap['rows'])} instruments live · profile "
                f"<code>{_profile}</code></div>", unsafe_allow_html=True)
        if _snap["not_live"]:
            st.caption("⏳ Not yet live on that date (history starts later): "
                       + ", ".join(sorted(_snap["not_live"])) +
                       " — the strategy could not have traded "
                       + ("it" if len(_snap["not_live"]) == 1 else "them")
                       + ", so the book below excludes "
                       + ("it" if len(_snap["not_live"]) == 1 else "them") + ".")

        _h_k = st.columns(5)
        _h_k[0].metric("Positions open", f"{_snap['n_active']}")
        _h_k[1].metric("Opened that day", f"{_snap['n_open']}",
                       delta="new entries" if _snap["n_open"] else None)
        _h_k[2].metric("Closed that day", f"{_snap['n_close']}",
                       delta="exit signal" if _snap["n_close"] else None,
                       delta_color="inverse")
        _h_k[3].metric("In risk assets", f"{(1 - _h_sata)*100:.0f}%")
        _h_k[4].metric("In SATA", f"{_h_sata*100:.0f}%",
                       help=f"{ov.SATA['name']} — idle cash parked at "
                            f"~{ov.SATA['annual_rate']*100:.0f}% yield.")

        # ── the book on that date + the blend's record up to it ────────────
        # this donut stays the CURRENT strategy's view of that day; the as-of
        # RECORD (the book actually published from this bar's committed
        # signals, data/overall/book_archive/<as_of>.json) is rendered in the
        # Executed Playbook section below.
        _h_arch_dir = _TB_DIR / "book_archive"
        _h_pub = _load_published_book(_h_arch_dir / f"{_h_bar.date()}.json")
        _h_cols = st.columns([1, 1.4])
        with _h_cols[0]:
            st.plotly_chart(
                _alloc_donut(_h_book, _h_sata,
                             f"Strategy book — {_h_bar.strftime('%b %d, %Y')}"),
                use_container_width=True)
            st.caption("The **replayed book on that bar**: anchor weights fit "
                       "only on data before that date, water-filled over the "
                       "sleeves in position at that close and tilted by the "
                       "as-of entry-priority read (momentum, sentiment, "
                       "expanding win-rate/Sharpe, regime — all lagged one "
                       "bar); undeployed capital in **SATA**. The same daily "
                       "book the combined back-test compounds."
                       + (" What was **actually traded** that day is in the "
                          "🧾 Executed Playbook below." if _h_pub else ""))
        with _h_cols[1]:
            _h_cm = ov.curve_metrics(_h_curve.loc[:_h_bar])
            _h_m = st.columns(4)
            _h_m[0].metric("Blend return to date", f"{_h_cm['total_ret']*100:+,.0f}%",
                           delta=f"CAGR {_h_cm['cagr']*100:+.0f}%")
            _h_m[1].metric("Max DD to date", f"{_h_cm['mdd']*100:.1f}%")
            _h_m[2].metric("Sharpe to date", f"{_h_cm['sharpe']:.2f}")
            _h_fwd = ov.slice_metrics(_h_curve, _h_bar)
            _h_m[3].metric("Since then → today",
                           "—" if _h_fwd is None else f"{_h_fwd['total_ret']*100:+.1f}%",
                           delta=(None if _h_fwd is None
                                  else f"${_h_fwd['total_ret']*_h_pv:+,.0f} on ${_h_pv:,.0f}"),
                           delta_color=("off" if _h_fwd is None else
                                        "normal" if _h_fwd["total_ret"] >= 0 else "inverse"),
                           help="What the strategy went on to return from "
                                "this date to the latest close — the one "
                                "forward-looking figure on this page.")
            _h_fig = go.Figure()
            _h_styles = {STRAT_CURVE: ("#111827", 3),
                         "Equal-weight strategies": ("#0ea5e9", 1.5),
                         "Equal-weight Buy & Hold": ("#94a3b8", 1.5)}
            for _h_nm, _h_cv in _PF["curves"].items():
                _h_sub = _h_cv.loc[:_h_bar]
                if len(_h_sub) < 2:
                    continue
                _h_c, _h_w = _h_styles.get(_h_nm, ("#888", 1.5))
                _h_fig.add_trace(go.Scatter(x=_h_sub.index, y=_h_sub.to_numpy() * 100000,
                                            name=_h_nm, line=dict(color=_h_c, width=_h_w)))
            _h_fig.add_vline(x=_h_bar, line_dash="dot", line_color="#dc2626")
            _h_fig.update_layout(
                height=300, margin=dict(t=30, b=10, l=10, r=10),
                yaxis_title="Growth of $100k", yaxis_type="log",
                hovermode="x unified", legend=dict(orientation="h", y=1.12),
                title=dict(text=f"The record as known on {_h_bar.strftime('%b %d, %Y')}",
                           font_size=13))
            st.plotly_chart(_h_fig, use_container_width=True)

        # ── that day's action plan — what the strategy did on the bar ──────
        st.markdown(f"### 🎯 Action plan on {_h_bar.strftime('%b %d, %Y')}")
        st.caption("What the strategy did at that day's close, ranked: **closes** "
                   "first (with the trade's realised return and exit reason), then "
                   "fresh **opens**, **holds** and the flat sleeves. β = higher-beta "
                   "sibling · 2× = leveraged (traded off the parent signal). "
                   "**Close** is the official close of that instrument's bar on "
                   "the chosen date; **P&L** marks each open position at that bar. "
                   "**Book % / $** is the allocation shown in the donut above.")
        _h_order = {"CLOSE": 0, "OPEN": 1, "HOLD": 2, "STAND ASIDE": 3}
        _h_rows_sorted = sorted(
            _snap["rows"],
            key=lambda r: (_h_order[r["action"]], -_h_book.get(r["key"], 0.0)))
        _h_hdr = ("<tr style='background:#f1f5f9;font-size:12px;text-align:left'>"
                  "<th style='padding:7px 10px'>Action</th><th>Instrument</th>"
                  "<th style='text-align:right'>Bar</th>"
                  "<th style='text-align:right'>Close</th>"
                  "<th style='text-align:right'>Chg %</th>"
                  "<th>Position</th>"
                  "<th style='text-align:right'>P&amp;L at date</th>"
                  "<th style='text-align:right'>Book %</th>"
                  "<th style='text-align:right'>Book $</th></tr>")
        _h_tr = []
        for r in _h_rows_sorted:
            _h_ac = _ACTION_COL.get(r["action"], C_FLAT)
            _h_tgt = _h_book.get(r["key"], 0.0)
            _h_tgt_s = f"{_h_tgt*100:.1f}%" if _h_tgt > 0.0005 else "—"
            _h_amt_s = f"${_h_tgt*_h_pv:,.0f}" if _h_tgt > 0.0005 else "—"
            _h_bar_note = ("" if r["bar"] == _h_bar else
                           "<div style='font-size:10px;color:#d97706'>older bar</div>")
            if r["dchg"] is None:
                _h_chg_s, _h_chg_c = "—", "#94a3b8"
            else:
                _h_chg_s = f"{r['dchg']:+.2f}%"
                _h_chg_c = C_BUY if r["dchg"] >= 0 else C_EXIT
            if r["in_pos"]:
                _h_e_px = (f" @ ${r['entry_px']:,.2f}" if r["entry_px"] else "")
                _h_pos_s = (f"📍 LONG since {pd.Timestamp(r['entry_date']).strftime('%b %d, %Y')}"
                            f"{_h_e_px} · {r['days']}d")
                _h_pnl, _h_pnl_sub = r["upnl"], ""
            elif r["trade"]:
                t = r["trade"]
                _h_pos_s = (f"↩︎ closed — entered "
                            f"{t['entry_date'].strftime('%b %d, %Y')}"
                            + (f" @ ${t['entry_px']:,.2f}" if t["entry_px"] else ""))
                _h_pnl = t["ret"] * 100
                _h_pnl_sub = (f"<div style='font-size:10px;color:#94a3b8'>"
                              f"realised · {t['reason']}</div>")
            else:
                _h_pos_s = "<span style='color:#94a3b8'>⚪ flat</span>"
                _h_pnl, _h_pnl_sub = None, ""
            _h_pnl_s = _pct(_h_pnl)
            _h_pnl_c = ("#94a3b8" if _h_pnl is None
                        else C_BUY if _h_pnl >= 0 else C_EXIT)
            _h_tr.append(
                f"<tr style='border-bottom:1px solid #eef2f7'>"
                f"<td style='padding:8px 10px'>{_pill(r['action'], _h_ac)}</td>"
                f"<td style='font-weight:700'>{r['emoji']} {r['key']}{_kind_badge(r['kind'])}"
                f"<div style='font-size:11px;color:#64748b;font-weight:400'>{r['name']}"
                f"{'' if r['key'] == r['parent'] else ' · off ' + r['parent'] + ' signal'}</div></td>"
                f"<td style='text-align:right;font-size:12px;color:#64748b'>"
                f"{pd.Timestamp(r['bar']).strftime('%b %d')}{_h_bar_note}</td>"
                f"<td style='text-align:right;font-variant-numeric:tabular-nums'>${r['close']:,.2f}</td>"
                f"<td style='text-align:right;font-weight:600;font-variant-numeric:tabular-nums;"
                f"color:{_h_chg_c}'>{_h_chg_s}</td>"
                f"<td style='font-size:12px'>{_h_pos_s}</td>"
                f"<td style='text-align:right;font-weight:600;color:{_h_pnl_c}'>{_h_pnl_s}{_h_pnl_sub}</td>"
                f"<td style='text-align:right;font-weight:700'>{_h_tgt_s}</td>"
                f"<td style='text-align:right;font-weight:700;font-variant-numeric:tabular-nums'>{_h_amt_s}</td></tr>")
        # SATA row — whatever the risk sleeves left undeployed that day
        _h_tr.append(
            f"<tr style='border-top:2px solid #cbd5e1;background:#f8fafc'>"
            f"<td style='padding:8px 10px'>{_pill('PARK', '#334155')}</td>"
            f"<td style='font-weight:700'>💵 SATA"
            f"<div style='font-size:11px;color:#64748b;font-weight:400'>{ov.SATA['name']}</div></td>"
            f"<td style='text-align:right;color:#94a3b8'>—</td>"
            f"<td style='text-align:right;font-variant-numeric:tabular-nums'>${ov.SATA['par']:,.2f}</td>"
            f"<td style='text-align:right;color:#94a3b8'>—</td>"
            f"<td style='font-size:12px;color:#334155'>Idle cash → SATA "
            f"(~{ov.SATA['annual_rate']*100:.0f}%/yr daily dividend)</td>"
            f"<td style='text-align:right;color:#94a3b8'>—</td>"
            f"<td style='text-align:right;font-weight:800'>{_h_sata*100:.1f}%</td>"
            f"<td style='text-align:right;font-weight:800;font-variant-numeric:tabular-nums'>"
            f"${_h_sata*_h_pv:,.0f}</td></tr>")
        st.markdown(f"<div style='overflow-x:auto;margin:8px 0;'>"
                    f"<table style='width:100%;border-collapse:collapse'>"
                    f"{_h_hdr}{''.join(_h_tr)}</table></div>", unsafe_allow_html=True)
        if _snap["n_active"] == 0:
            st.info("**No open positions on that date** — no instrument was "
                    "signalling long, so the entire book sat in **SATA** earning "
                    "its idle-cash yield.")
        st.caption("⚠️ Reconstructed from the committed back-test series: bars, "
                   "fills and exits are exactly what the engines decided on that "
                   "date, and the **book weights are the walk-forward replay's** "
                   "— anchors fit only on data before that date, priority tilt "
                   "from as-of inputs — water-filled over that day's positions. "
                   "An as-of reconstruction, not a live record of a book "
                   "published that day (the 🧾 Executed Playbook below is the "
                   "record where one exists). Prices are official daily closes "
                   "(no live-spot overlay). Not investment advice.")

        # ── EXECUTED PLAYBOOK — the book actually published & traded ───────
        st.markdown(f"### 🧾 Executed Playbook — {_h_bar.strftime('%b %d, %Y')}")
        if not _h_pub:
            _h_first = min((p.stem for p in _h_arch_dir.glob("*.json")),
                           default=None)
            st.info("No published Targetbook was recorded for this bar, so the "
                    "Strategy book above is the only view."
                    + (f" Records start **{_h_first}** — the daily 7:15 AM CT "
                       f"publisher archives every book it hands to the "
                       f"executor." if _h_first else ""))
        else:
            st.caption("The **as-of record**: the Targetbook actually published "
                       "from this bar's committed signals and handed to the "
                       "IBKR executor — built with the blend weights and "
                       "entry-priority tilt of **that day**, unlike the "
                       "current-weights Strategy book above · "
                       f"{_book_close_caption(_h_pub)} · profile "
                       f"**{_h_pub.get('profile', '—')}** · published "
                       f"**{fr.fmt_ct(_h_pub.get('generated_at_utc'))}**")
            _ep_cols = st.columns([1, 1.4])
            _epw, _epidle = _book_alloc(_h_pub)
            with _ep_cols[0]:
                st.plotly_chart(
                    _alloc_donut(_epw, _epidle,
                                 f"Executed book — {_h_bar.strftime('%b %d, %Y')}"),
                    use_container_width=True)
            with _ep_cols[1]:
                _ep_px = _h_pub.get("exec_price") or {}
                _ep_order = {"CLOSE": 0, "OPEN": 1, "HOLD": 2, "WATCH": 3,
                             "STAND ASIDE": 4}
                _ep_acts = sorted(
                    (a for a in (_h_pub.get("actions") or []) if a.get("key")),
                    key=lambda a: (_ep_order.get(a.get("action"), 9),
                                   -float(a.get("target") or 0.0)))
                _ep_hdr = ("<tr style='background:#f1f5f9;font-size:12px;"
                           "text-align:left'>"
                           "<th style='padding:7px 10px'>Action</th>"
                           "<th>Instrument</th><th>Published decision</th>"
                           "<th style='text-align:right'>Priority</th>"
                           "<th style='text-align:right'>Exec px</th>"
                           "<th style='text-align:right'>Book %</th>"
                           "<th style='text-align:right'>Book $</th></tr>")
                _ep_tr = []
                for a in _ep_acts:
                    k = a["key"]
                    meta = by_key.get(k) or ov.ASSET_META.get(k, {})
                    _ep_w = float(a.get("target") or 0.0)
                    _ep_w_s = f"{_ep_w*100:.1f}%" if _ep_w > 0.0005 else "—"
                    _ep_amt = f"${_ep_w*_h_pv:,.0f}" if _ep_w > 0.0005 else "—"
                    _ep_p = a.get("priority")
                    _ep_p_s = f"{_ep_p:.2f}" if _ep_p is not None else "—"
                    _ep_x = _ep_px.get(k)
                    _ep_x_s = f"${_ep_x:,.2f}" if _ep_x else "—"
                    _ep_tr.append(
                        f"<tr style='border-bottom:1px solid #eef2f7'>"
                        f"<td style='padding:8px 10px'>"
                        f"{_pill(a.get('action') or '—', _ACTION_COL.get(a.get('action'), C_FLAT))}</td>"
                        f"<td style='font-weight:700'>{meta.get('emoji', '')} {k}"
                        f"{_kind_badge(meta.get('kind'))}"
                        f"<div style='font-size:11px;color:#64748b;font-weight:400'>"
                        f"{meta.get('name', '')}</div></td>"
                        f"<td style='font-size:12px'>{a.get('decision') or '—'}</td>"
                        f"<td style='text-align:right;font-variant-numeric:tabular-nums'>{_ep_p_s}</td>"
                        f"<td style='text-align:right;font-variant-numeric:tabular-nums'>{_ep_x_s}</td>"
                        f"<td style='text-align:right;font-weight:700'>{_ep_w_s}</td>"
                        f"<td style='text-align:right;font-weight:700;"
                        f"font-variant-numeric:tabular-nums'>{_ep_amt}</td></tr>")
                if _epidle > 0.0005:
                    _ep_sata_px = _ep_px.get("SATA")
                    _ep_tr.append(
                        f"<tr style='border-top:2px solid #cbd5e1;background:#f8fafc'>"
                        f"<td style='padding:8px 10px'>{_pill('PARK', '#334155')}</td>"
                        f"<td style='font-weight:700'>💵 SATA"
                        f"<div style='font-size:11px;color:#64748b;font-weight:400'>"
                        f"{ov.SATA['name']}</div></td>"
                        f"<td style='font-size:12px;color:#334155'>Idle capital "
                        f"parked (~{ov.SATA['annual_rate']*100:.0f}%/yr)</td>"
                        f"<td style='text-align:right;color:#94a3b8'>—</td>"
                        f"<td style='text-align:right;font-variant-numeric:tabular-nums'>"
                        f"{f'${_ep_sata_px:,.2f}' if _ep_sata_px else '—'}</td>"
                        f"<td style='text-align:right;font-weight:800'>{_epidle*100:.1f}%</td>"
                        f"<td style='text-align:right;font-weight:800;"
                        f"font-variant-numeric:tabular-nums'>${_epidle*_h_pv:,.0f}</td></tr>")
                st.markdown(f"<div style='overflow-x:auto;margin:8px 0;'>"
                            f"<table style='width:100%;border-collapse:collapse'>"
                            f"{_ep_hdr}{''.join(_ep_tr)}</table></div>",
                            unsafe_allow_html=True)
                st.caption("Published decisions, entry-priority scores and "
                           "execution prices exactly as archived at publish "
                           "time (`data/overall/book_archive/`) — signature-"
                           "covered, never recomputed. **Book $** applies the "
                           "portfolio value above to the published weights.")


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — COMBINED BACKTESTING
# ══════════════════════════════════════════════════════════════════════════
with tab_bt:
    o = opt["optimal"]
    _wf_bt = _PF["wf"]
    _wfm = _wf_bt["metrics"]
    _oos_lo, _oos_hi = ov.oos_start_span()
    _oos_span = _oos_lo if _oos_lo == _oos_hi else f"{_oos_lo}–{_oos_hi}"
    st.markdown("## 📊 The combined strategy — walk-forward gated replay")
    st.caption("Each instrument's signal-driven strategy produces a daily return "
               "stream (long when its parent signal is on, otherwise **idle "
               "capital earns the SATA yield ~13%/yr**). The back-test replays "
               "the SAME daily logic the live book runs: capital deployed only "
               "to the sleeves in the market, sized by anchor weights **tilted "
               "by the as-of entry-priority read** and water-filled to the "
               "profile caps. Anchors are **re-fit each quarter start on prior "
               "data only** (equal-weight before enough history exists), priority "
               "inputs are lagged one bar — nothing in the curve uses "
               "information from after the day it describes. Out-of-sample "
               f"from {_oos_span} depending on the instrument (staggered starts — "
               "newer sleeves like BTC begin ~2024); $100k start.")
    # ── risk-profile trade-off: leaning on β / 2× proxies ───────────────
    st.markdown(f"**Risk profile: `{_profile}`** — "
                f"{ov.RISK_PROFILES[_profile]['blurb']} "
                f"_(change it in the sidebar.)_")
    comp = get_profile_comparison(_bucket())
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
        st.markdown(f"<div style='overflow-x:auto;margin:8px 0;'>"
                    f"<table style='width:100%;border-collapse:collapse'>{ch}{''.join(crows)}</table></div>",
                    unsafe_allow_html=True)
        st.caption("Loading the high-beta / leveraged proxies (β + 2× weight) "
                   "**boosts return but lowers Sharpe** — the drawdown deepens "
                   "faster than the return. Growth leans harder on the β / 2× "
                   "sleeves than Balanced inside a −22% drawdown budget.")
    st.markdown("")

    m = st.columns(4)
    m[0].metric("Overall strategy — total return", f"{_wfm['total_ret']*100:,.0f}%",
                delta=f"CAGR {_wfm['cagr']*100:.0f}%")
    m[1].metric("Max drawdown", f"{_wfm['mdd']*100:.0f}%",
                delta=f"vs {bm['bh_equal']['mdd']*100:.0f}% buy&hold", delta_color="inverse")
    m[2].metric("Sharpe", f"{_wfm['sharpe']:.2f}",
                delta=f"vs {bm['bh_equal']['sharpe']:.2f} buy&hold")
    m[3].metric("Volatility (ann.)", f"{_wfm['vol']*100:.0f}%")

    st.markdown("#### Today's anchor weights (live book)")
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
            "<div style='overflow-x:auto;margin:8px 0;'>"
            "<table style='width:100%;font-size:13px;border-collapse:collapse'>"
            "<tr style='background:#f1f5f9'><th style='text-align:left;padding:4px 6px'>Scheme</th>"
            "<th style='text-align:right'>Ret</th><th style='text-align:right'>MDD</th>"
            "<th style='text-align:right'>Sharpe</th></tr>" + "".join(cmp_rows) + "</table></div>",
            unsafe_allow_html=True)
        st.caption("Leveraged sleeves capped at 10%, high-beta at 18%, core at "
                   "30% — so the optimiser only leans on the 2× / β names when "
                   "they improve risk-adjusted return. These weights anchor "
                   "**today's live book**; the back-test above re-fits its own "
                   "anchors each quarter on prior data only (this scheme table is "
                   "a static-blend diagnostic of the current fit, not the "
                   "back-test).")

    st.markdown("#### Growth of $100k — combined vs benchmarks")
    fig = go.Figure()
    styles = {STRAT_CURVE: ("#111827", 3),
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

    # ── daily % allocation through time — the replayed book, day by day ────
    if st.toggle("📊 Daily % allocation by asset — how the replayed book was "
                 "deployed through time (toggle on/off)",
                 value=True, key="overall_alloc_history"):
        st.plotly_chart(
            _alloc_area_fig(_wf_bt["weights"], _wf_bt["sata"],
                            "Daily % allocation — every asset plus SATA "
                            "(stacks to 100%)"),
            use_container_width=True)
        st.caption("The book the back-test compounds each day, **one colour "
                   "per signal group** (e.g. ₿ BTC covers BTC·MSTR·MSTU·ETH — "
                   "hover any band for that day's per-asset split); grey on "
                   "top is idle cash in **SATA**. Bands widen and narrow "
                   "daily as priorities move — even when no Action signal "
                   "changes — exactly like the live book.")

    opt_curve = _PF["curves"][STRAT_CURVE]
    dd = opt_curve / opt_curve.cummax() - 1
    figd = go.Figure(go.Scatter(x=dd.index, y=dd.to_numpy() * 100, fill="tozeroy",
                                line=dict(color=C_EXIT, width=1)))
    figd.update_layout(height=230, margin=dict(t=10, b=10, l=10, r=10),
                       yaxis_title="drawdown %",
                       title=dict(text="Overall strategy — drawdown", font_size=13))
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
    st.markdown(f"<div style='overflow-x:auto;margin:8px 0;'>"
                f"<table style='width:100%;border-collapse:collapse'>{ph}{''.join(pr)}</table></div>",
                unsafe_allow_html=True)

    st.markdown("#### Per-instrument strategy (standalone, model-OOS — "
                "strategy parameters tuned in-window)")
    st.caption("Each instrument's signal-driven strategy vs buy-&-hold, and its "
               "anchor weight in today's live book. Grouped by signal — β = high-beta "
               "sibling, 2× = leveraged. **Siblings (↳) are traded off their "
               "parent's signal**, not their own: MSTR/MSTU/ETH enter and exit on "
               "BTC's divergence signal, GDX/UGL on gold's, OIH on XLE's — the "
               "higher-beta name executes on its own price but is steered by the "
               "cleaner parent read. Every asset here runs its **own app's actual "
               "engine** — BTC/MSTR/MSTU/ETH via the BTC app's trained CT model, "
               "GLDM/GDX/UGL via the Gold app's Divergence Pure-Regime, the ETFs "
               "via their tuned configs — so these numbers match each source app. "
               "(BTC's CT features begin ~2024, so its sleeve covers a shorter "
               "window than the earliest-start ETFs.)")
    ah = ("<tr style='background:#f1f5f9'><th style='text-align:left;padding:6px 10px'>Instrument</th>"
          "<th>Signal / engine</th><th style='text-align:right'>Strat</th>"
          "<th style='text-align:right'>Buy&amp;Hold</th><th style='text-align:right'>Max DD</th>"
          "<th style='text-align:right'>Sharpe</th><th style='text-align:right'>Win%</th>"
          "<th style='text-align:right'>Anchor wt</th></tr>")
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
    st.markdown(f"<div style='overflow-x:auto;margin:8px 0;'>"
                f"<table style='width:100%;border-collapse:collapse'>{ah}{''.join(ar)}</table></div>",
                unsafe_allow_html=True)

    st.success(
        f"**Bottom line.** Replaying the daily gate/tilt book across "
        f"{N_ALL} instruments — including the higher-beta MSTR/MSTU, GDX/UGL and OIH "
        f"sleeves, used only when they earn their capped slots — the strategy "
        f"returned **{_wfm['total_ret']*100:,.0f}%** at **{_wfm['mdd']*100:.0f}%** "
        f"max drawdown (Sharpe **{_wfm['sharpe']:.2f}**), versus an equal-weight "
        f"buy-&-hold of the same instruments at **{bm['bh_equal']['total_ret']*100:,.0f}%** "
        f"but a **{bm['bh_equal']['mdd']*100:.0f}%** drawdown "
        f"(Sharpe **{bm['bh_equal']['sharpe']:.2f}**).")
    st.caption("⚠️ The replay is walk-forward (anchors re-fit each quarter start on prior "
               "data only, priority inputs lagged a bar, no fundamental overlay), "
               "but per-sleeve strategy *parameters* were tuned on history and "
               "no transaction costs are charged — a daily-rebalanced book "
               "trades often. Strategy curves park idle capital in SATA "
               "(~13%/yr, an assumed-constant series applied over the whole "
               "history); the buy-&-hold benchmark is always fully invested "
               "with no SATA. Not investment advice.")


# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — METHODOLOGY
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
signal (never its own), exactly as the dedicated apps do. Entry and exit rules
are summarised per app (all decided on **completed daily closes**):

| App / signal | Traded instruments | Engine | Entry criteria | Exit criteria |
|---|---|---|---|---|
| ₿ **BTC** | BTC · ETH · MSTR (β) · MSTU (2×) | CT-model Divergence · Standard-MA gate | U1 divergence — predicted-high error > +1.3% with ≥2 high-breaks (3d), price above the 30-day MA | Regime-adaptive D2/D3 (err_hi < −1.3%); BTC signal-only, MSTR −3% / MSTU −6% / ETH −8% stops (2026-07-25 honest-fill re-sweep) + post-stop re-entry (5-bar V-reversal, 12-bar U1 override) |
| 🥇 **Gold Trend (GLDM)** | GLDM · UGL (2×) | Dual-MA 25/100 | 25-day SMA of the GLDM close crosses above the 100-day (decided at the close) | 25-day SMA crosses back below the 100-day; −3% stops |
| ⛏️ **Gold Miners (GDXM)** | GDX (β) · NUGT (2×) | Divergence Pure-Regime | U1 divergence on the GLDM signal (err_hi 3d-avg > +0.10% with ≥2 high-breaks) + regime confirm | D2 fade (err_hi < −0.20%) or D3 exhaustion; GDX −5%, NUGT −8% |
| 🖥️ **SOXX** | SOXX · SOXL (3×) | Dual-MA 25/100 | 25-day SMA crosses above the 100-day SMA | 25-day SMA crosses back below the 100-day; SOXX −5% stop, SOXL signal-only |
| ⚡ **GRID** | GRID | MACD 10/20/9 | MACD histogram turns positive (MACD above its signal line) | MACD histogram turns negative; −5% stop |
| 🛢️ **XLE** | XLE · OIH (β) · ERX (2×) | Crash-shield quasi-B&H | Long by default; (re-)enter when the close is above the 50-day SMA and not in a crash state | Exit only while the close sits >30% below its rolling 52-week high (crash, not correction); no fixed stop |
| 🧲 **REMX** | REMX | Dual-MA 50/200 golden cross | 50-day SMA crosses above the 200-day SMA | 50-day SMA crosses back below the 200-day; −5% stop |
| ⛏️ **WGMI** | WGMI (β) | MA50 + vol filter | Close above the 50-day SMA AND 10-day realised vol < 0.95× its 189-day median | Close below the 50-day SMA or vol spikes above the filter; no fixed stop |
| ☀️ **PBW** | PBW | Divergence Pure-Regime | U1 divergence (err_hi 3d-avg > +0.42% with ≥2 high-breaks) + regime confirm | D2 fade (err_hi < −0.18%) or D3 exhaustion; −10% stop |
| 🤖 **ARTY** | ARTY | Divergence Pure-Regime | U1 divergence (err_hi 3d-avg > +0.32% with ≥2 high-breaks) + regime confirm | D2 fade (err_hi < −0.46%) or D3 exhaustion; signal-only (no fixed stop) |

*Shorthand:* **U1** = bullish divergence (the model's predicted daily high
overshoots while price breaks its recent highs) · **D1** = downtrend pressure
(repeated low-breaks) · **D2** = momentum fade · **D3** = exhaustion ·
**regime confirm** = price above a rising short MA, or a clean 10-day recovery,
or a V-reversal gate · **signal-only** = exits on the signal with no fixed
stop (kept where the 2026-07-25 honest-fill stop sweeps showed a fixed stop
only whipsaws). Siblings share the
parent's entry/exit **timing** but fill at their own price with their own stop.

Every asset runs the **exact engine its own app trades**, so the Overall app's
signals, positions and back-tests match each source app:

- **BTC / ETH / MSTR / MSTU** run the **BTC app's own trained CT model**
  (`inference_assets_ct.joblib`, 116 features incl. Bitcoin on-chain + Coinbase
  premium) with the app's live **Standard MA (above-MA30) entry gate** (U1>+1.3%
  + ≥2 high-breaks, regime-adaptive D2/D3 exit, MA30 gate, per-asset stops, SL
  re-entry) — all four assets share the same gate. Since the **2026-07-25
  look-ahead fix**, MSTR/MSTU fills land at the first exchange close *after*
  the 12:00-UTC signal moment (the old same-date fill preceded the signal by
  ~15 h and banked the overnight gap; config-unchanged the fix moved MSTR
  +296%→+184% and MSTU +685%→+402% — BTC/ETH unchanged). Stops were then
  re-swept on the honest fill (MSTR −3%, MSTU −6%, ETH −8%; BTC stop-less),
  giving **BTC +58% · MSTR +245% · MSTU +677% · ETH +40%** on the 2026-07-25
  vintage. Figures drift with data-vintage refreshes. The CT feature data
  begins ~2023-11, so the BTC sleeve covers ~2024→now (the combined engine
  handles the staggered start), and the CT model's training window extends
  into the displayed period — only bars after its `train_end` are model-blind.
- **GLDM / GDX / UGL / NUGT** run the **Gold app's `backtest_gldm`** middle-path
  split — dual-MA 25/100 for GLDM/UGL, Divergence Pure-Regime for GDX/NUGT —
  with per-asset stops (GLDM/UGL −3%, GDX −5%, NUGT −8% after the 2026-07-25
  re-sweep). Since the 2026-07-25 look-ahead fix the divergence engine decides
  at close i−1 and fills at close i (its signal needs bar i's realized
  high/low). Model-OOS 2021→now on the fix-date vintage: **GLDM +137% ·
  UGL +302% · GDX +110% · NUGT +217%** (strategy thresholds tuned on this
  window; older LEV_SIBLINGS_STOP_EVAL.md figures pre-date the causal H/L fix
  and are historical).
- **SOXX / GRID / XLE / REMX / WGMI / PBW / ARTY** reuse their **exact
  `ticker_config`** entries through the same `backtest_ticker` engine their apps
  use (SOXX 25/100 dual-MA driving the stop-less 3× SOXL, GRID MACD 10/20/9,
  WGMI 50-day SMA + vol-filter, REMX 50/200 golden cross, and XLE / PBW / ARTY
  divergence Pure-Regime — XLE's signal also driving OIH and the stop-less 2×
  ERX). These match their apps bar-for-bar.

**Live signals & positions.** For each app we fetch data, fit the H/L band model
out-of-sample, replay the strategy bar-by-bar, and read off the current alert
level, whether we're long, entry price/date, unrealised P&L, stop (or a
signal-only exit for the no-stop sleeves — **BTC, ARTY, the XLE sleeve
(XLE · OIH · ERX), the 3× SOXL** and **WGMI** carry no fixed stop, where the
stop sweeps showed a fixed stop only whipsaws) and days held —
for the primary **and** each sibling (which shares the parent's entry/exit timing
but has its own fill price and P&L). That drives the **action plan** and the
**allocation donuts**.

**Rebalancing moves.** The 🔁 collapsible box on the Live tab holds both
rebalancing views. The first lists the trades that take the book **actually in
force** — the published **Current Targetbook**, the artifact the IBKR executor
trades (never a synthetic reconstruction) — to the live-adjusted recommended
target, each delta routed to/from SATA. A 🎛️ include/exclude overlay can drop
a position first; its weight parks in SATA and is never redistributed to the
survivors. The second view compares the **previous → current published books**
with the optimizer's recorded rationale per move — a **signal fired** (fresh
entry), a **signal died** (exit / trend break / retired from the universe), or
the **priority tilt drifted** and the water-fill re-spread the difference —
read from the published payloads themselves, not a live recomputation.

**The optimal allocation.** Each strategy is long when its signal is on and
otherwise parks idle capital in **SATA** (see below), producing a daily return
stream. We Monte-Carlo long-only blends (sum = 100%) with **per-instrument caps
set by the active risk profile** (see below) and pick the winner by that
profile's objective, so returns are maximised while drawdown stays inside the
budget. The tight caps mean the 2× / β sleeves only get weight when they
genuinely improve the risk-adjusted result. These full-history weights anchor
**today's live book only**.

**The combined back-test = walk-forward gated replay.** Historical performance
is NOT a fixed-weight blend. Every day of the back-test rebuilds the book the
live logic would have held **using only information available at the previous
close**: the sleeves in the market that day get capital, sized by anchor
weights × (0.5 + entry-priority) and water-filled to the profile caps, with the
remainder in SATA. Anchor weights are **re-fit at every quarter start
(Jan/Apr/Jul/Oct 1) on the data before that date** (cap-normalised equal
weight before enough history exists — quarterly refits were adopted after the
adaptivity eval showed they beat annual on both return and Sharpe), the
priority inputs are their as-of analogues (momentum vs the 50-day SMA, the
rolling sentiment gauge, *expanding* win-rate and Sharpe, the MA20 bull-regime
rule) lagged one bar, and the fundamental overlay is excluded throughout.
Position state **carries across non-trading days** (the crypto sleeves put
weekends into the calendar; a carried equity position keeps its weight and
contributes 0 return until its next bar), and **SATA accrues on weekday bars
only** (its coupon is 0.13/250 per *business* day). The 📊 tab's
daily-allocation chart shows this replayed book through time.

**Risk profiles.** The ⚙️ switch on the Live tab bundles the per-kind caps, the
optimiser objective and a drawdown budget:

| Profile | Caps (core / β / 2×) | Objective |
|---|---|---|
| **Balanced** (default) | 30% / 18% / 10% | Hold Sharpe near its max — best risk-adjusted blend |
| **Growth** | 30% / 25% / 18% | Maximise return inside a **−22%** drawdown budget |

β / 2× exposure rises Balanced → Growth: more return, deeper drawdowns, lower
Sharpe. Every number on the Live and Backtesting tabs follows the selected
profile. (An *Aggressive* profile — 35/40/35 caps, −38% budget — exists in
`overall_core` for the CLI tools but is retired from this UI: its drawdown
budget was judged too deep to publish.)

**Fundamental overlay — retired.** Earlier builds offered a 🔭 toggle applying
a mid-2026 sector forward-view multiplier to the anchor weights. It has been
**removed everywhere** (app and the daily Target-Book publisher): the view was
formed knowing how 2021→2026 played out, so any use of it — even for today's
book — bakes a hindsight-formed conviction into the sizing. All allocations
are now the pure quant optimum under the profile caps.

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

**🩺 Strategy Health — the decay watchdog.** Every sleeve's config was
*selected* by sweeping candidates over the same history it is reported on, so
the live edge is expected to be smaller than the back-tested edge — and simple
technical rules (dual-MA, MACD, divergence gates) are exactly the class of
signal that dies when the regime that made them work ends. A nightly job
(right after the Target-Book publish) therefore rebuilds **four monitors** per
sleeve and per profile and commits the snapshot the 🩺 **Strategy Health** app
reads:

| Monitor | Question |
|---|---|
| **M1** Book-vs-replay tracking | Is the *published* book earning what the model says it should? |
| **M2** Drawdown tripwires | Has the curve gone deeper — or stayed underwater longer — than its own history ever did? |
| **M3** Edge vs Buy & Hold | Is the *timing* alpha alive, or is absolute P&L just the asset rallying? |
| **M4** Trade expectancy | Are the last 20 trades consistent with the sleeve's own trade distribution? |

Each monitor reports 🟢 healthy · 🟡 warning · 🔴 alarm · ⬜ warming-up (not
enough data — deliberately *not* green), with thresholds fixed **ex-ante** and
each breach's first date persisted in the committed artifact, so a one-day
flicker never escalates and every flag's history is auditable in git. The
worst state surfaces as the one-line badge under this app's title. **The
monitor never changes a weight** — the adaptivity eval's penalty-box variant
showed auto-de-rating cold sleeves was neutral-to-negative — so an alarm
triggers the *human* review protocol: re-run the sleeve's honest-fill re-sweep
and decide (retune, de-rate, or retire). With ~18 monitored sleeves about one
🟡 at any time is expected by chance; act on 🔴, or on a 🟡 that persists.
Full design + thresholds: `STRATEGY_HEALTH.md`.

**Honest caveats.**
- The back-test's *allocation layer* is walk-forward (anchors fit on prior data
  only, as-of priority inputs, no overlay), but each sleeve's **strategy
  parameters** (thresholds, stops, MA windows) were tuned on history, and the
  BTC CT model's training window extends into the displayed period — the
  replay removes the weight-level look-ahead, not parameter-tuning bias.
- No transaction costs or slippage are charged, and the daily-rebalanced book
  trades far more often than a fixed blend.
- SATA is modelled (per the BTC app's framing) as always having existed, flat at
  $100 par, paying its ~13% daily dividend across every period — an assumption,
  not a market-tested series.
- Leveraged sleeves (MSTU, UGL, NUGT, ERX 2× and SOXL 3×) and high-beta names
  compound decay and gap risk; the caps bound but don't remove that.
- This is a **daily** engine. The BTC and Gold apps' canonical **hourly**
  Pure-Regime signals live in those apps — open them from the sidebar.
- Nothing here is investment advice.
        """)


st.caption(f"Overall Trading · unified daily engine · {len(results)} instruments · "
           f"data to {as_of.strftime('%Y-%m-%d')} · not investment advice.")
