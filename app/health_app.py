"""🩺 Strategy Health — the decay monitor across every sleeve and the book.

One read-only page that answers, at a glance: **is the strategy still working,
or has it decayed?**  Four monitors (methodology + thresholds:
``STRATEGY_HEALTH.md``; compute: ``app/health_core.py``):

  M1  Published-book vs replay tracking (the live record vs the model)
  M2  Drawdown tripwires (current depth/duration vs the reference maximum)
  M3  Rolling 6-month edge vs Buy & Hold (is the *timing* alpha alive?)
  M4  Last-20-trade expectancy vs the sleeve's own bootstrap band

Like 🕵️ Daily Audit, this page is deliberately LIGHT: it never runs the heavy
engine.  Everything is read from ``data/overall/strategy_health.json`` and
``data/overall/health_history.csv``, computed nightly by
``scripts/build_strategy_health.py`` inside the publish-target-book GitHub
Action and committed — so every flag shown here is auditable in git history.

Alarms feed the human review protocol (re-run the sleeve's honest-fill
re-sweep and decide); nothing here changes any weight automatically — the
adaptivity study (V4 penalty box) showed auto-derating doesn't pay.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _APP_DIR.parent
for _p in (str(_APP_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ticker_config                        # noqa: E402
import freshness as fr                      # noqa: E402
import health_core as hc                    # noqa: E402

try:
    st.set_page_config(page_title="Strategy Health", page_icon="🩺",
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
    st.session_state["gldm_active_app"] = "HEALTH"
with st.sidebar:
    st.radio("**Application**", options=_ALL_APPS,
             format_func=lambda x: _APP_LABELS.get(x, x), key="gldm_active_app")
    st.markdown("---")
    if st.button("Refresh now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("_Computed nightly by the publish workflow and committed, so "
               "every flag is auditable in git history. Nothing here changes "
               "any weight — alarms feed the human re-evaluation protocol._")

_EMOJI = hc.STATUS_EMOJI
_HEALTH_JSON = _REPO_ROOT / "data" / "overall" / "strategy_health.json"
_HEALTH_CSV = _REPO_ROOT / "data" / "overall" / "health_history.csv"


@st.cache_data(ttl=300, show_spinner=False)
def _load(mtime: float, hist_mtime: float):
    snap = json.loads(_HEALTH_JSON.read_text())
    hist = pd.read_csv(_HEALTH_CSV) if _HEALTH_CSV.exists() else pd.DataFrame()
    return snap, hist


st.title("🩺 Strategy Health — Decay Monitor")
import strategy_version as _sv                 # noqa: E402
st.caption(_sv.badge_caption())

if not _HEALTH_JSON.exists():
    st.info("**No health snapshot yet.** The monitor is computed nightly by the "
            "publish-target-book workflow (`scripts/build_strategy_health.py`) "
            "and committed as `data/overall/strategy_health.json`. Run it once "
            "manually to seed this page:\n\n"
            "```bash\npython scripts/build_strategy_health.py\n```")
    st.stop()

snap, hist = _load(_HEALTH_JSON.stat().st_mtime,
                   _HEALTH_CSV.stat().st_mtime if _HEALTH_CSV.exists() else 0.0)
v = snap["verdict"]
th = snap.get("thresholds", hc.THRESHOLDS)

# ════════════════════════════════════════════════════════════════════════════
# Tier 1 · The verdict
# ════════════════════════════════════════════════════════════════════════════
_banner = {"green": st.success, "yellow": st.warning, "red": st.error}[v["status"]]
_banner(f"{_EMOJI[v['status']]} **{v['headline']}** · as-of **{snap['as_of']}** · "
        f"{v['n_sleeves']} sleeves ({v['n_warming']} ⬜ warming up) · "
        f"live record since {snap['live_anchor']}")
st.caption("_With ~18 monitored sleeves, roughly one 🟡 at any time is expected "
           "by pure chance — act on 🔴, or on a 🟡 that persists. ⬜ means not "
           "enough data to judge (deliberately not green). Thresholds were "
           "fixed ex-ante — see the methodology box below._")

# ════════════════════════════════════════════════════════════════════════════
# Tier 2a · Portfolio — published book vs replay, and drawdown tripwires
# ════════════════════════════════════════════════════════════════════════════
st.markdown("### 1 · Portfolio — the live record vs the model")
track = snap["portfolio"].get("tracking")
cL, cR = st.columns([3, 2])
with cL:
    st.markdown("**M1 · Published book vs walk-forward replay** "
                "(growth of $1, close-to-close, gross)")
    if track:
        ser = track["series"]
        df = pd.DataFrame({"Published books (live record)": ser["live"],
                           "Gated replay (model)": ser["replay"]},
                          index=pd.to_datetime(ser["dates"]))
        st.line_chart(df, height=260)
        m1, m2, m3 = st.columns(3)
        m1.metric("Cumulative gap (live − model)", f"{track['cum_gap'] * 100:+.2f}%")
        m2.metric(f"Trailing {track['window']}-day gap",
                  f"{track['gap_window'] * 100:+.2f}%",
                  help=f"Yellow below {th['track_yellow'] * 100:+.1f}%, "
                       f"red below {th['track_red'] * 100:+.1f}%")
        m3.metric("Status", f"{_EMOJI[track['status']]} {track['status']}",
                  help="Persistent shortfall vs the model = decay-in-practice "
                       "(publish timing, missed days, weight drift), even "
                       "when every signal still works.")
        st.caption(f"_{track['n_days']} live days since {track['first_date']}. "
                   "Both legs are gross of costs and close-to-close, so the gap "
                   "isolates what was actually **published** vs what the replay "
                   "assumes; real fills layer in once the executed-book history "
                   "is long enough._")
    else:
        st.info("⬜ Warming up — needs archived books in "
                "`data/overall/book_archive/` plus a replay leg. The gap "
                "series starts accruing on the next nightly build.")
with cR:
    st.markdown("**M2 · Profile drawdown tripwires**")
    for name, p in (snap["portfolio"]["profiles"] or {}).items():
        if p["status"] == "warming":
            st.markdown(f"⬜ **{name}** — {p.get('need', 'warming up')}")
            continue
        st.markdown(
            f"{_EMOJI[p['status']]} **{name}** — drawdown **{p['value'] * 100:+.1f}%** "
            f"vs reference max **{p['ref_mdd'] * 100:+.1f}%** · underwater "
            f"**{p['cur_uw_days']}d** vs max **{p['ref_max_uw_days']}d**")
    st.caption("_A breach of the reference maximum falsifies the back-test's "
               "risk model for that profile — the pre-committed response is "
               "de-risk and re-evaluate, decided by a human, not the "
               "optimiser._")

# ════════════════════════════════════════════════════════════════════════════
# Tier 2b · Sleeve health board (worst first)
# ════════════════════════════════════════════════════════════════════════════
st.markdown("### 2 · Sleeve health board")
st.caption("_One row per traded instrument, worst first. **M2** current "
           "drawdown vs its own pre-live maximum · **M3** rolling "
           f"{th['edge_window']}-bar strategy-minus-B&H (the timing edge) · "
           f"**M4** mean return of the last {th['exp_window']} trades as a "
           "percentile of the sleeve's own bootstrap band._")


def _dd_cell(s):
    d = s["dd"]
    if d["status"] == "warming":
        return f"⬜ {d.get('need', '')}"
    return (f"{_EMOJI[d['status']]} {d['value'] * 100:+.1f}% / "
            f"{d['ref_mdd'] * 100:+.1f}% · {d['cur_uw_days']}d/{d['ref_max_uw_days']}d")


def _edge_cell(s):
    e = s["edge"]
    if e["status"] == "warming":
        return f"⬜ {e.get('need', '')}"
    since = f" (since {e['first_breach_date']})" if e["first_breach_date"] else ""
    return f"{_EMOJI[e['status']]} {e['value_pp']:+.1f}pp{since}"


def _exp_cell(s):
    x = s["expectancy"]
    if x["status"] == "warming":
        return f"⬜ {x.get('need', '')}"
    return f"{_EMOJI[x['status']]} p{x['pctl']:.0f}"


_board = pd.DataFrame([{
    "Status": _EMOJI[s["status"]],
    "Sleeve": f"{s['emoji']} {s['key']} · {s['name']}",
    "Signal": s["parent"], "Kind": s["kind"],
    "M2 · DD vs limit": _dd_cell(s),
    "M3 · Edge 6m": _edge_cell(s),
    "M4 · Expectancy": _exp_cell(s),
    "Trades": s["n_trades"],
} for s in snap["sleeves"]])
st.dataframe(_board, hide_index=True, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# Tier 3 · Drill-downs
# ════════════════════════════════════════════════════════════════════════════
st.markdown("### 3 · Sleeve drill-downs")
_flagged = [s for s in snap["sleeves"] if s["status"] in ("red", "yellow")]
if not _flagged:
    st.caption("_No flagged sleeves — every drill-down is collapsed below._")
for s in snap["sleeves"]:
    label = (f"{_EMOJI[s['status']]} {s['emoji']} {s['key']} — {s['name']} "
             f"({s['parent']} signal, {s['kind']})")
    with st.expander(label, expanded=s["status"] == "red"):
        g1, g2 = st.columns(2)
        ser = s["series"]
        with g1:
            st.markdown("**Drawdown** (vs pre-live reference max)")
            if ser["dd"]:
                ddf = pd.DataFrame({"drawdown %": [x * 100 for x in ser["dd"]]},
                                   index=pd.to_datetime(ser["dates"]))
                if s["dd"].get("ref_mdd") is not None:
                    ddf["reference max %"] = s["dd"]["ref_mdd"] * 100
                st.line_chart(ddf, height=200)
        with g2:
            st.markdown(f"**Rolling {th['edge_window']}-bar edge vs Buy & Hold** (pp)")
            if ser["edge"]:
                edf = pd.DataFrame({"edge pp": ser["edge"], "zero": 0.0},
                                   index=pd.to_datetime(ser["edge_dates"]))
                st.line_chart(edf, height=200)
            else:
                st.caption("⬜ not enough joint history yet")
        x = s["expectancy"]
        if x["status"] == "warming":
            st.caption(f"**M4 expectancy:** ⬜ warming up — {x.get('need', '')} "
                       f"({x['n_trades']} closed trades so far).")
        else:
            st.caption(f"**M4 expectancy:** last {th['exp_window']} trades "
                       f"averaged {x['window_mean'] * 100:+.2f}%/trade — "
                       f"percentile **p{x['pctl']:.0f}** of the sleeve's own "
                       f"bootstrap band ({x['n_trades']} trades total). "
                       f"Yellow < p{th['exp_yellow_pctl']:.0f}, red < "
                       f"p{th['exp_red_pctl']:.0f} persisting {th['exp_red_days']}d.")
        if s["status"] == "red":
            st.error("**Pre-committed response:** the back-test no longer "
                     "describes this sleeve. Re-run its honest-fill re-sweep "
                     "(see the sleeve's eval doc) and decide — retune, "
                     "de-rate, or retire. Do not wait for the portfolio "
                     "curve to show it.")

# ════════════════════════════════════════════════════════════════════════════
# History + methodology
# ════════════════════════════════════════════════════════════════════════════
if not hist.empty:
    with st.expander("📈 Flag history (from health_history.csv)"):
        piv = (hist[hist["scope"] == "sleeve"]
               .pivot_table(index="date", columns="key", values="status",
                            aggfunc="first"))
        st.dataframe(piv.map(lambda x: _EMOJI.get(x, x) if isinstance(x, str)
                             else x), use_container_width=True)

with st.expander("🧪 Methodology & thresholds (fixed ex-ante)"):
    st.markdown(f"""
* **M1 tracking** — daily return of the latest **published** book (archive,
  decided-at-previous-close, SATA on the cash leg) minus the gated replay's
  same-day return. Trailing {th['track_window']}-day gap below
  **{th['track_yellow'] * 100:+.1f}%** → 🟡, below
  **{th['track_red'] * 100:+.1f}%** → 🔴; ⬜ until
  {th['track_min_days']} live days exist (a days-old record is variance,
  not signal).
* **M2 drawdown** — reference = each curve **before {snap['live_anchor']}**
  (the first published book). Deeper current drawdown, or a longer underwater
  spell, than the reference ever produced → 🔴; within
  {th['dd_approach'] * 100:.0f}% of either limit → 🟡. Needs
  {th['dd_min_ref_bars']} reference bars to arm.
* **M3 edge** — rolling {th['edge_window']}-bar strategy return minus Buy &
  Hold. Below −{th['edge_deadband_pp']}pp (a dead-band so hairline noise
  never flags) → 🟡; persisting {th['edge_red_days']} days → 🔴.
* **M4 expectancy** — mean return of the last {th['exp_window']} closed trades
  vs {th['boot_n']:,} bootstrap draws (seed {th['boot_seed']}) of
  {th['exp_window']}-trade samples from the sleeve's earlier trades. Below
  p{th['exp_yellow_pctl']:.0f} → 🟡; below p{th['exp_red_pctl']:.0f} for
  {th['exp_red_days']}+ days → 🔴. ⬜ until
  {th['exp_window'] + th['exp_min_ref']} closed trades exist.

Full design + rationale: **STRATEGY_HEALTH.md**. Persistence lives in the
committed snapshot (`first_breach_date`), so flags survive redeploys and are
auditable in git history.""")

st.caption(f"_Snapshot generated {snap['generated_at_utc']} UTC by "
           "`scripts/build_strategy_health.py` · this page only reads "
           "committed artifacts — it never runs the engine._")

# record this page's own render, like every other app
fr.record_refresh("HEALTH", kind="audit", app_label="🩺 Strategy Health")
