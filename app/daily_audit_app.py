"""🕵️ Daily Audit — the signal-freshness trail across every app.

One user-friendly page that answers, at a glance:

  * **Which close is each app's signal generated from, and when was it last
    generated?**  (BTC: the 12:00-UTC / 7:00-AM-CT daily-bar close; every other
    app: the 4:00 PM ET US market close.)
  * **When was the Overall strategy view last computed/refreshed, and did its
    freshness audit pass?**
  * **When was the Target Book generated and published?**

All times come from ONE shared source of truth (``app/freshness.py``) — the
same helpers that render the closing date/time captions inside each individual
app and that gate the scheduled once-daily 7:15-AM-CT publish — so the
timestamps shown here always match what the apps themselves display.

Data sources (all light — this page never runs the heavy engine):
  * the live clock → the freshest close each asset class can possibly have now;
  * ``runtime/freshness_log.json`` → when each app's page last refreshed in this
    deployment and which close it displayed (written by the apps on render);
  * ``data/overall/daily_audit.json`` → the audit trail written by the scheduled
    once-daily publisher run (committed by the GitHub Action);
  * ``data/overall/target_book(_live).json`` → the published books' timestamps.
"""
from __future__ import annotations

import json
import sys
import importlib
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

if not hasattr(fr, "audit_universe"):       # self-heal a stale module cache
    fr = importlib.reload(fr)

try:
    st.set_page_config(page_title="Daily Audit", page_icon="🕵️",
                       layout="wide", initial_sidebar_state="expanded")
except Exception:
    pass

# ── sidebar Application selector (same widget/key as every other app) ─────────
_ALL_APPS = (["OVERALL", "BTC", "GLDM", "GDXM"] + ticker_config.APP_KEYS
             + ["DAILYAUDIT", "TARGETBOOK", "EXECUTEDBOOK"])
_APP_LABELS = {"OVERALL": "🧭  Overall Trading", "BTC": "₿  Bitcoin (BTC)",
               "GLDM": "🥇  Gold Trend (GLDM·UGL)", "GDXM": "⛏️  Gold Miners (GDX·NUGT)", "DAILYAUDIT": "🕵️  Daily Audit",
               "TARGETBOOK": "📋  Target Book (IBKR)",
               "EXECUTEDBOOK": "✅  Executed Book (IBKR)"}
for _k, _c in ticker_config.CONFIGS.items():
    _APP_LABELS[_k] = f"{_c.emoji}  {_c.key} · {_c.name.split('(')[0].strip()[:22]}"
if st.session_state.get("gldm_active_app") not in _ALL_APPS:
    st.session_state["gldm_active_app"] = "DAILYAUDIT"
with st.sidebar:
    st.radio("**Application**", options=_ALL_APPS,
             format_func=lambda x: _APP_LABELS.get(x, x), key="gldm_active_app")
    st.markdown("---")
    if st.button("Refresh now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("_Every timestamp on this page comes from the same shared "
               "helpers the individual apps use, so they always match._")

# Signal apps in the Overall universe, with their display metadata.
_PARENTS = [("BTC", "₿", "Bitcoin (BTC → BTC · MSTR · MSTU)"),
            ("GLDM", "🥇", "Gold (GLDM → GLDM · GDX · UGL · NUGT)")] + \
           [(k, ticker_config.CONFIGS[k].emoji,
             f"{k} · {ticker_config.CONFIGS[k].name.split('(')[0].strip()}")
            for k in ticker_config.APP_KEYS]

_NOW = fr.now_utc()
_LOG = fr.read_refresh_log()

# The committed scheduled-run audit trail (written headlessly by the publisher
# GitHub Action) — the fallback source for sections 1–2 whenever an app page
# hasn't rendered in this deployment (or rendered longer ago than the last
# scheduled run).  This removes the page's dependency on the Streamlit apps
# being open: signals refreshed by the scheduled publisher show up here too.
_DA = fr.load_daily_audit() or {}
_DA_GEN = _DA.get("generated_at_utc")
_DA_ROWS = {r.get("app"): r for r in (_DA.get("audit") or {}).get("rows") or []}


def _scheduled_is_newer(entry: dict | None) -> bool:
    """True when the committed scheduled-run audit is fresher than a live
    page-render record (or no live record exists)."""
    if not _DA_GEN:
        return False
    if not entry or not entry.get("recorded_at_utc"):
        return True
    try:
        return pd.Timestamp(_DA_GEN) > pd.Timestamp(entry["recorded_at_utc"])
    except Exception:
        return False

st.title("🕵️ Daily Audit — signal freshness trail")
st.caption(f"🔄 Audit page refreshed **{fr.fmt_ct(_NOW, seconds=True)}**")

# ── the daily cycle, in plain words ──────────────────────────────────────────
st.markdown(
    "**The daily cycle:** every app except Bitcoin generates its signals upon "
    "the **US market close (4:00 PM ET)**. Bitcoin's daily bar closes at "
    "**12:00 UTC — 7:00 AM Central in summer (CDT), 6:00 AM in winter (CST)** — "
    "and its predictions/signals update then. The **Overall strategy** is "
    "recomputed headlessly on a schedule **once a day at ≈7:15 AM US Central** "
    "— ≈15 minutes after the Bitcoin bar close, so it sees BTC's fresh "
    "7:00-AM-CT bar and every equity app's prior 4:00-PM-ET close. The **daily "
    "audit runs ONCE, first**: every app's signals are validated as being on "
    "their freshest bar, and only then is the **Target Book published "
    "immediately**. The published book then stays **frozen until the next "
    "morning's cycle** — only the 🚀 *Publish new target book* button in the "
    "📋 Target Book app (a manual workflow run) replaces it intraday. The "
    "cycle doesn't need the Streamlit app to be open: the scheduled publisher "
    "retries until the day's book is committed.")
st.markdown("---")


def _status(fresh: bool | None, age: int = 0) -> str:
    if fresh is None:
        return "➖ not recorded yet"
    return "✅ Fresh" if fresh else f"🚨 STALE ({age}d behind)"


def _stat(col, label: str, value) -> None:
    """Compact stat card — like ``st.metric`` but with a value font sized for
    full CT timestamps (st.metric's ~2.25rem value clips them to the date)."""
    col.markdown(
        "<div style='border:1px solid rgba(128,128,128,.28);border-radius:10px;"
        "padding:9px 13px;margin-bottom:6px'>"
        f"<div style='font-size:12px;opacity:.65;font-weight:600'>{label}</div>"
        f"<div style='font-size:15.5px;font-weight:700;margin-top:2px;"
        f"line-height:1.35'>{value}</div></div>",
        unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# 1 · Individual app signals — closing bar & last generation
# ════════════════════════════════════════════════════════════════════════════
st.markdown("### 1 · Individual app signals")
st.caption("What each app's signals are generated from and the freshest close "
           "available right now. **Signals last refreshed** counts only a "
           "refresh computed from the *freshest close available now* (a live "
           "page render in this deployment or the scheduled headless publisher "
           "run, whichever is more recent) — an app still sitting on an older "
           "close shows no refresh time and a 🚨 STALE status.")

rows = []
for key, emoji, label in _PARENTS:
    kind = fr.PARENT_CLASS.get(key, "us_equity")
    exp = fr.expected_asof(kind, _NOW)
    entry = _LOG.get(key) or {}
    darow = _DA_ROWS.get(key)
    if darow is not None and _scheduled_is_newer(entry):
        # the committed scheduled-run audit is the freshest record for this app
        logged_asof = darow.get("actual_asof")
        refreshed = f"{_DA.get('generated_at_ct', '—')} · ⚙️ scheduled run"
    else:
        logged_asof = entry.get("as_of")
        refreshed = entry.get("recorded_at_ct") or "—"
    fresh = None
    age = 0
    if logged_asof:
        fresh = pd.Timestamp(logged_asof) >= exp
        age = max((exp - pd.Timestamp(logged_asof)).days, 0)
    if not fresh:
        # only a refresh computed from the freshest available close counts —
        # a stale (or unrecorded) app shows no refresh time at all
        refreshed = "— not refreshed on this close yet"
    rows.append({
        "App": f"{emoji} {label}",
        "Status": _status(fresh, age),
        "Signals update at": ("Bitcoin daily-bar close — 12:00 UTC (7:00 AM CDT / 6:00 AM CST)"
                              if kind == "crypto" else
                              "US market close — 4:00 PM ET"),
        "Freshest close available now": fr.close_label(kind, exp, _NOW),
        "Signals last refreshed": refreshed,
    })
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
st.caption("_Each app also shows this same closing date/time and page-refresh "
           "time in its own header caption — the values are produced by the "
           "same shared code, so they always agree with this table._")

# ════════════════════════════════════════════════════════════════════════════
# 2 · Overall strategy — last update & audit verdict
# ════════════════════════════════════════════════════════════════════════════
st.markdown("### 2 · Overall strategy")
_ov = _LOG.get("OVERALL")
if _DA and _scheduled_is_newer(_ov):
    # No live Overall render since the last scheduled publisher run — show the
    # committed headless run (same engine, same audit) instead of a blank.
    _aud = _DA.get("audit") or {}
    _ovr = _DA.get("overall") or {}
    _ov = dict(as_of=_ovr.get("as_of", "—"),
               recorded_at_ct=f"{_DA.get('generated_at_ct', '—')} · ⚙️ scheduled run",
               profile=_ovr.get("profile", "—"),
               n_instruments=_ovr.get("n_instruments", "—"),
               audit_passed=bool(_aud.get("passed")),
               stale_apps=_aud.get("stale_apps") or [],
               audit_rows=_aud.get("rows") or [])
if not _ov:
    st.info("No Overall run recorded yet — the scheduled publisher hasn't "
            "committed an audit trail and the 🧭 Overall Trading app hasn't "
            "rendered in this deployment.")
else:
    c = st.columns([1.1, 1.6, 0.8, 0.7])
    _stat(c[0], "Signals as-of (newest bar)", _ov.get("as_of", "—"))
    _stat(c[1], "Overall view last updated", _ov.get("recorded_at_ct", "—"))
    _stat(c[2], "Risk profile", _ov.get("profile", "—"))
    _stat(c[3], "Instruments", str(_ov.get("n_instruments", "—")))
    if _ov.get("audit_passed"):
        st.success("✅ The Overall strategy's signal-freshness audit **PASSED** on "
                   "its last run — every app's signals were on their freshest bar.")
    else:
        _stale = ", ".join(_ov.get("stale_apps") or []) or "unknown"
        st.error(f"🚨 The Overall strategy's signal-freshness audit **FAILED** on "
                 f"its last run — stale: **{_stale}**. The Overall app flags the "
                 f"affected assets with a flashing stale-signals alert.")
    _arows = _ov.get("audit_rows") or []
    if _arows:
        with st.expander("Per-app audit detail (from the last Overall run)"):
            st.dataframe(pd.DataFrame([
                dict(App=r.get("app"),
                     Instruments=", ".join(r.get("instruments") or []),
                     **{"Signals from close of": r.get("actual_close"),
                        "Freshest possible close": r.get("expected_close"),
                        "Status": _status(r.get("fresh"), r.get("age_days", 0))})
                for r in _arows]), hide_index=True, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# 3 · Scheduled once-daily publish — the committed audit trail
# ════════════════════════════════════════════════════════════════════════════
st.markdown("### 3 · Scheduled refresh & publish (once daily, ≈7:15 AM CT)")
_da = _DA or None
if not _da:
    st.info("No scheduled-run audit artifact found yet "
            "(`data/overall/daily_audit.json`). It is written by the "
            "**Publish target book** GitHub Action — once daily, ≈15 min "
            "after the 7:00-AM-CT Bitcoin bar close — and committed to the "
            "repo; it appears here after the first scheduled run.")
else:
    aud = _da.get("audit") or {}
    ovr = _da.get("overall") or {}
    tbk = _da.get("target_book") or {}
    c = st.columns([1.5, 0.9, 0.7, 1.5])
    _stat(c[0], "Overall computed", fr.fmt_ct(_da.get("generated_at_utc")))
    _stat(c[1], "Signals as-of", ovr.get("as_of", "—"))
    _stat(c[2], "Audit", "✅ PASSED" if aud.get("passed") else "🚨 FAILED")
    _stat(c[3], "Target book published",
          fr.fmt_ct(tbk.get("generated_at_utc")) if tbk.get("published") else "⛔ withheld")
    if not aud.get("passed"):
        st.error("🚨 The scheduled run's audit **FAILED** — stale: "
                 f"**{', '.join(aud.get('stale_apps') or [])}**. "
                 + ("The Target Book was **NOT** published from stale signals; "
                    "the executor keeps trading the previous verified book (its "
                    "own freshness guard skips if that book is too old)."
                    if not tbk.get("published") else
                    "See the run log for how the book was still produced."))
    elif tbk.get("published"):
        st.success("✅ Audit passed and the Target Book was published immediately "
                   "after the Overall strategy update.")
    if aud.get("rows"):
        with st.expander("Per-app audit detail (scheduled run)",
                         expanded=not aud.get("passed", True)):
            st.dataframe(pd.DataFrame([
                dict(App=r.get("app"),
                     Instruments=", ".join(r.get("instruments") or []),
                     **{"Signals from close of": r.get("actual_close"),
                        "Freshest possible close": r.get("expected_close"),
                        "Status": _status(r.get("fresh"), r.get("age_days", 0))})
                for r in aud["rows"]]), hide_index=True, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# 4 · Target Book — generation & publish times
# ════════════════════════════════════════════════════════════════════════════
st.markdown("### 4 · Target Book")
_books = [("🧪 Paper book", _REPO_ROOT / "data" / "overall" / "target_book.json"),
          ("🔴 Live book", _REPO_ROOT / "data" / "overall" / "target_book_live.json")]
bc = st.columns(len(_books))
for (_lbl, _p), _col in zip(_books, bc):
    with _col:
        try:
            _pl = json.loads(_p.read_text())
            st.markdown(f"**{_lbl}**")
            st.markdown(
                f"- Generated & published: **{fr.fmt_ct(_pl.get('generated_at_utc'))}**\n"
                f"- Signal bar (as-of): **{_pl.get('as_of', '—')}**\n"
                f"- Risk profile: **{_pl.get('profile', '—')}** · "
                f"{'🔏 signed' if _pl.get('signature') else '🔓 unsigned'}")
        except Exception:
            st.markdown(f"**{_lbl}** — _not published yet_")
st.caption("_Open the 📋 Target Book app for the full allocation, signature "
           "verification and downloads._")

# record this page's own render, like every other app
fr.record_refresh("DAILYAUDIT", kind="audit", app_label="🕵️ Daily Audit")
