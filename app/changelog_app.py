"""📜 Strategy Change Log — what changed in the strategy, under which version, and when.

One read-only page. It lists **Strategy Logic V2** (current) and **V1** with the
day each was in force, the GitHub pull requests that introduced them, a plain
summary, and a dated log of every change under its version. Different changes
inside one version have different release dates — the date shown is the day the
change reached ``main`` (the pull request's merge date, UTC).

The log itself is data, not code: ``data/changelog/strategy_changelog.json``
(schema ``strategy-changelog/v1``). Logging a change is one new entry under its
version; ``tests/test_strategy_changelog.py`` keeps the file consistent with
``app/strategy_version.py``. Keeping it out of ``app/*.py`` is deliberate — the
P&L snapshot's code fingerprint hashes the engine modules, and a changelog edit
must not invalidate it.

GitHub has no tags, releases or branches for V1 or V2, so the references next to
each version are the introducing pull requests and the version-history file.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "app"))
import ticker_config  # noqa: E402

try:
    st.set_page_config(page_title="Strategy Change Log", page_icon="📜",
                       layout="wide", initial_sidebar_state="expanded")
except Exception:
    pass

# ── sidebar Application selector (same widget/key as every other app) ─────────
_ALL_APPS = (["OVERALL", "BTC", "GLDM", "GDXM"] + ticker_config.APP_KEYS
             + ["LEVERAGED", "DAILYAUDIT", "HEALTH", "TARGETBOOK",
                "EXECUTEDBOOK", "ASSISTANT", "CHANGELOG"])
_APP_LABELS = {"OVERALL": "🧭  Overall Trading", "BTC": "₿  Bitcoin (BTC)",
               "GLDM": "🥇  Gold Trend (GLDM·UGL)", "GDXM": "⛏️  Gold Miners (GDX·NUGT)",
               "LEVERAGED": "⚡  Leveraged Assets",
               "DAILYAUDIT": "🕵️  Daily Audit",
               "HEALTH": "🩺  Strategy Health",
               "TARGETBOOK": "📋  Target Book (IBKR)",
               "EXECUTEDBOOK": "✅  Executed Book",
               "ASSISTANT": "🤖  AI Assistant",
               "CHANGELOG": "📜  Strategy Change Log"}
for _k, _c in ticker_config.CONFIGS.items():
    _APP_LABELS[_k] = f"{_c.emoji}  {_c.key} · {_c.name.split('(')[0].strip()[:22]}"
if st.session_state.get("gldm_active_app") not in _ALL_APPS:
    st.session_state["gldm_active_app"] = "CHANGELOG"
with st.sidebar:
    st.radio("**Application**", options=_ALL_APPS,
             format_func=lambda x: _APP_LABELS.get(x, x), key="gldm_active_app")
    st.markdown("---")
    if st.button("Reload log", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("_Read-only. Every strategy change is logged under its version "
               "in `data/changelog/strategy_changelog.json`._")

_LOG_PATH = _REPO_ROOT / "data" / "changelog" / "strategy_changelog.json"

# change types → (label, colour)
KINDS = {"sleeve": ("Sleeve logic", "#0ea5e9"),
         "allocator": ("Allocator", "#7c3aed"),
         "execution": ("Execution & publishing", "#d97706"),
         "data": ("Data & integrity", "#16a34a"),
         "analytics": ("Analytics & monitoring", "#64748b"),
         "ui": ("UI", "#94a3b8")}


@st.cache_data(ttl=300, show_spinner=False, max_entries=2)
def _load(mtime: float) -> dict:
    return json.loads(_LOG_PATH.read_text(encoding="utf-8"))


def _link(l: dict) -> str:
    return (f"<a href='{html.escape(l['url'], quote=True)}' target='_blank' "
            f"rel='noopener'>{html.escape(l['label'])}</a>")


def _pill(text: str, color: str) -> str:
    return (f"<span style='background:{color}22;color:{color};font-weight:700;"
            f"padding:2px 9px;border-radius:999px;font-size:12px;white-space:nowrap'>"
            f"{html.escape(text)}</span>")


def _entries_table(entries: list[dict]) -> str:
    rows = []
    for e in entries:
        label, color = KINDS.get(e["kind"], (e["kind"], "#64748b"))
        scope = (f" {_pill(e['profile'] + ' only', '#0f766e')}" if e.get("profile") else "")
        impact = (f"<div style='font-size:12px;color:#475569;margin-top:4px'>"
                  f"<b>Why it helps / evidence:</b> {html.escape(e['impact'])}</div>"
                  if e.get("impact") else "")
        refs = " · ".join(_link(l) for l in e.get("links", [])) or "—"
        rows.append(
            "<tr style='border-top:1px solid #e2e8f0;vertical-align:top'>"
            f"<td style='padding:8px 10px;white-space:nowrap;font-variant-numeric:tabular-nums'>"
            f"<b>{html.escape(e['date'])}</b></td>"
            f"<td style='padding:8px 6px'>{_pill(label, color)}</td>"
            f"<td style='padding:8px 10px'><b>{html.escape(e['title'])}</b>{scope}"
            f"<div style='font-size:13px;margin-top:3px'>{html.escape(e['description'])}</div>"
            f"{impact}</td>"
            f"<td style='padding:8px 10px;font-size:12px'>{refs}</td></tr>")
    head = ("<tr style='background:#f1f5f9;font-size:12px;text-align:left'>"
            "<th style='padding:7px 10px'>Released</th><th>Type</th>"
            "<th style='padding:7px 10px'>Change</th><th style='padding:7px 10px'>GitHub</th></tr>")
    return ("<div style='overflow-x:auto;margin:6px 0'>"
            f"<table style='width:100%;border-collapse:collapse'>{head}{''.join(rows)}</table></div>")


st.title("📜 Strategy Change Log")
import strategy_version as _sv                 # noqa: E402
if not (hasattr(_sv, "render_badge") and hasattr(_sv, "BADGE_COLOR")):
    import importlib                           # stale hot-loaded module
    _sv = importlib.reload(_sv)
_sv.render_badge()

if not _LOG_PATH.exists():
    st.info("**No change log yet.** Expected `data/changelog/strategy_changelog.json`.")
    st.stop()

_doc = _load(_LOG_PATH.stat().st_mtime)
_versions = _doc["versions"]
_entries = _doc["entries"]
_by_ver = {v["id"]: v for v in _versions}

st.caption("What changed in the strategy, under which version, and when. **Released** is the day a "
           "change reached `main` (its pull request's merge date, UTC); a version's **in force** "
           "dates are its first and last signal days. Different changes inside one version have "
           "different release dates.")

_kinds_present = [k for k in KINDS if any(e["kind"] == k for e in _entries)]
_pick = st.multiselect("Filter by type", options=_kinds_present,
                       default=_kinds_present, format_func=lambda k: KINDS[k][0],
                       key="changelog_kinds")


def _sel(vid: str) -> list[dict]:
    rows = [e for e in _entries if e["version"] == vid and e["kind"] in _pick]
    return sorted(rows, key=lambda e: e["date"], reverse=True)


tab_ver, tab_all = st.tabs(["By version", "Timeline (newest first)"])

with tab_ver:
    for v in _versions:                                 # newest version first (file order)
        cur = v["status"] == "current"
        span = (f"in force from **{v['effective']}**" if v.get("ended") is None
                else f"in force **{v['effective']} → {v['ended']}**")
        st.markdown(
            f"### {html.escape(v['label'])} &nbsp; "
            + _pill("CURRENT" if cur else "RETIRED", "#16a34a" if cur else "#64748b")
            + f" &nbsp; <span style='font-size:14px;color:#64748b;font-weight:400'>{span}</span>",
            unsafe_allow_html=True)
        st.markdown("**GitHub:** " + (" · ".join(_link(l) for l in v.get("links", [])) or "—"),
                    unsafe_allow_html=True)
        st.write(v["summary"])
        hl = v.get("headline")
        if hl:
            st.markdown(f"**V1 → V2 on the replay** — {hl['columns']}; {hl['as_of']}")
            st.dataframe(
                [{"Profile": r["profile"], "V1": r["v1"], "V2": r["v2"]} for r in hl["rows"]],
                hide_index=True, use_container_width=True)
        rows = _sel(v["id"])
        st.markdown(f"**Major features and changes ({len(rows)})**")
        if rows:
            st.markdown(_entries_table(rows), unsafe_allow_html=True)
        else:
            st.caption("_No entries match the type filter._")
        st.markdown("---")

with tab_all:
    rows = sorted([e for e in _entries if e["kind"] in _pick],
                  key=lambda e: (e["date"], e["version"]), reverse=True)
    st.markdown(f"**{len(rows)} changes** across "
                + ", ".join(f"{html.escape(v['label'])} ({sum(e['version'] == v['id'] for e in rows)})"
                            for v in _versions), unsafe_allow_html=True)
    tl = [dict(e, title=f"[{e['version'].upper()}] {e['title']}") for e in rows]
    st.markdown(_entries_table(tl), unsafe_allow_html=True)

st.markdown("##### Keeping it current")
st.caption(f"The log is `{_LOG_PATH.relative_to(_REPO_ROOT)}` — add one entry under the right "
           "version (`version`, `date`, `kind`, `title`, `description`, optional `impact`, "
           "`profile`, `links`) for every change; a new strategy-logic generation also adds a row "
           "to `VERSION_HISTORY` in `app/strategy_version.py`. "
           + _doc.get("note", ""))
