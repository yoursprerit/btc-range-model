"""Single source of truth for the deliberate strategy-logic version.

BUMP ``STRATEGY_VERSION`` whenever the Overall logic or any per-asset
strategy changes materially — gate/tilt/water-fill rules, optimiser
objective/caps, per-asset entry/exit logic or parameters, universe changes.
Display/UI-only changes do NOT warrant a bump.  The publisher stamps every
Targetbook it commits with this version (plus the publishing commit SHA), and
the Overall P&L section segments the as-published record wherever the stamp
changes, so old-logic and current-logic performance are never conflated.

Version history
---------------
* ``pre-v1`` — evolving logic before inline stamping (books 2026-07-16 →
  2026-07-30, labelled from the ``book_versions.json`` side-car).
* ``v1`` (**Strategy Logic V1**, from 2026-07-31) — the 2026-07-25 honest-fill
  retunes: dual-MA 25/100 gold (−3% fixed stops), WGMI SMA50 + vol filter,
  REMX 50/200 golden cross (−5%), daily priority tilt re-sizing every held
  sleeve.
* ``v2`` (**Strategy Logic V2**, from 2026-10-09) — the 2026-10 loss review:
  gold (GLDM/UGL) takes the 25/100 cross only while the 100-day SMA is rising
  and trades a 10%/12% trailing stop instead of the −3% fixed stop; WGMI
  enters only while BTC sits above its 50-day SMA; REMX holds only while its
  20-day SMA is above the 100-day; the allocator never trims a held sleeve
  except on its signal exit and only adds when the target rises by ≥ 8 pp
  ("adds-only" band); the sleeves sharing one parent signal (XLE → OIH/ERX,
  GLDM → UGL, SOXX → SOXL, GDX → NUGT, BTC → MSTR/MSTU/ETH) never exceed
  30 % of the book combined ("parent-cluster cap", added 2026-10-10 before
  the first V2 publish); a stop/trail hit publishes a CLOSE.  GRID, SOXX,
  XLE, BTC, PBW, ARTY, GDX/NUGT sleeves are identical in V1 and V2.

Kept dependency-light (stdlib only) so the read-only artifact apps
(📋 Targetbook, ✅ Executed Book, 🕵️ Daily Audit, 🩺 Strategy Health) can show
the badge without importing the heavy engine stack.  ``overall_core``
re-exports the constants for the engine/publisher side.
"""
STRATEGY_VERSION = "v2"

# First signal day (as_of) counted into the current version's as-published
# record.  BUMP TOGETHER with STRATEGY_VERSION — set it to the day of the
# bump.  The Overall P&L's as-published view only admits books stamped with
# the current version on/after this date, so a same-day re-publish of an
# older bar under new code can never pull old-strategy days into the record.
STRATEGY_VERSION_START = "2026-10-09"

# Every deliberate logic generation and the first signal day it governs,
# oldest first.  ``version_for_date`` reads this; the apps' V1 / V2 /
# Combined selectors and every "version in effect on that day" label derive
# from it, so a future bump is one row here plus the two constants above.
VERSION_HISTORY = [
    ("pre-v1", "2026-07-16"),
    ("v1", "2026-07-31"),
    ("v2", STRATEGY_VERSION_START),
]

# Human labels.  The pill, captions and radio options all read these.
VERSION_LABELS = {
    "pre-v1": "Pre-V1 (evolving logic)",
    "v1": "Strategy Logic V1",
    "v2": "Strategy Logic V2",
}

# The three analysis views every P&L / back-test / health surface offers.
VIEW_V1, VIEW_V2, VIEW_COMBINED = "Strategy V1", "Strategy V2", "Combined"
VIEW_OPTIONS = [VIEW_V1, VIEW_V2, VIEW_COMBINED]
VIEW_TO_VERSION = {VIEW_V1: "v1", VIEW_V2: "v2", VIEW_COMBINED: "combined"}
VERSION_TO_VIEW = {v: k for k, v in VIEW_TO_VERSION.items()}
# ``combined`` = each day under the logic that was actually in effect that
# day (V1 before the cut-over, V2 from it) — the live record's own history.
COMBINED_VERSIONS = ("v1", "v2")

BADGE_COLOR = "#7c3aed"        # the logic-provenance purple, used app-wide
V1_COLOR = "#0ea5e9"           # sky — V1 segments / markers on charts
V2_COLOR = "#7c3aed"           # purple — V2 segments / markers on charts
VERSION_COLORS = {"pre-v1": "#94a3b8", "v1": V1_COLOR, "v2": V2_COLOR}


def version_for_date(as_of) -> str:
    """The strategy-logic version in effect for a signal day (``as_of``, any
    date-like).  Days before the first stamped generation are ``pre-v1``."""
    d = str(as_of)[:10]
    cur = VERSION_HISTORY[0][0]
    for ver, start in VERSION_HISTORY:
        if d >= start:
            cur = ver
    return cur


def version_label(ver: str) -> str:
    """``v2`` → ``Strategy Logic V2`` (unknown tags are upper-cased as-is)."""
    return VERSION_LABELS.get(ver, f"Strategy Logic {str(ver).upper()}")


def versions_for_view(view_or_version: str) -> tuple:
    """Which stamped versions a selector choice admits: ``v1`` → ``('v1',)``,
    ``v2`` → ``('v2',)``, ``combined`` (or the Combined radio label) → both."""
    v = VIEW_TO_VERSION.get(view_or_version, view_or_version)
    if v == "combined":
        return tuple(COMBINED_VERSIONS)
    return (v,)


def cutover_dates() -> list:
    """Start days of every stamped generation after the first one — the
    points a Combined chart marks as strategy transitions."""
    return [start for _, start in VERSION_HISTORY[1:]]


def badge_caption() -> str:
    """Plain-text one-liner (markdown) — fallback where HTML can't render."""
    return (f"⚙️ Strategy logic version: **`{STRATEGY_VERSION}`** "
            f"({version_label(STRATEGY_VERSION)}, trading since "
            f"{STRATEGY_VERSION_START}) — bumped on material strategy "
            "changes; every published Targetbook is stamped with the version "
            "that produced it.")


def badge_html() -> str:
    """The high-visibility version pill every app shows under its title: the
    version number in a bold white-on-purple pill so it can't be missed, with
    the explanatory note in muted small print beside it."""
    return (
        "<div style='margin:2px 0 12px 0;display:flex;align-items:center;"
        "gap:10px;flex-wrap:wrap'>"
        f"<span style='background:{BADGE_COLOR};color:#ffffff;"
        "font-weight:800;font-size:16px;line-height:1;padding:7px 15px;"
        "border-radius:999px;letter-spacing:.04em;white-space:nowrap;"
        "box-shadow:0 1px 3px rgba(0,0,0,.2)'>"
        f"⚙️ STRATEGY LOGIC {STRATEGY_VERSION.upper()}</span>"
        f"<span style='color:#64748b;font-size:12px'>"
        f"{version_label(STRATEGY_VERSION)} trades from "
        f"{STRATEGY_VERSION_START} · bumped on material strategy changes · "
        "every published Targetbook is stamped with the version that "
        "produced it</span></div>")


def version_pill_html(ver: str, note: str = "") -> str:
    """A small inline pill naming the version in effect (used by historical
    replays, executed-book runs and chart captions)."""
    color = VERSION_COLORS.get(ver, BADGE_COLOR)
    extra = (f"<span style='color:#64748b;font-size:12px;margin-left:8px'>"
             f"{note}</span>" if note else "")
    return ("<span style='display:inline-flex;align-items:center;gap:6px'>"
            f"<span style='background:{color};color:#fff;font-weight:700;"
            "font-size:12px;line-height:1;padding:4px 10px;border-radius:999px;"
            f"white-space:nowrap'>⚙️ {version_label(ver).upper()}</span>"
            f"{extra}</span>")


def render_badge() -> None:
    """Render the pill in a Streamlit app (import kept local so this module
    stays importable engine-side without Streamlit)."""
    import streamlit as st
    st.markdown(badge_html(), unsafe_allow_html=True)


def render_version_pill(ver: str, note: str = "") -> None:
    import streamlit as st
    st.markdown(version_pill_html(ver, note), unsafe_allow_html=True)


def view_help(what: str = "figures") -> str:
    """Shared help text for the Strategy V1 / V2 / Combined selector."""
    return (f"**{VIEW_V1}** — every {what} under {version_label('v1')} only "
            f"(the rules traded 2026-07-31 → {STRATEGY_VERSION_START}). "
            f"**{VIEW_V2}** — the same {what} under {version_label('v2')} only "
            f"(the rules trading since {STRATEGY_VERSION_START}; on history "
            "before that date this is a what-if replay of the V2 rules). "
            f"**{VIEW_COMBINED}** — each day under the logic that was "
            "actually in effect on that day (V1 before the cut-over, V2 from "
            "it), with the transition marked.")


def render_view_radio(key: str, default: str = VIEW_COMBINED,
                      label: str = "⚙️ Strategy logic", what: str = "figures",
                      note: str | None = None, horizontal: bool = True) -> str:
    """The Strategy V1 / Strategy V2 / Combined radio every analysis surface
    shares.  Returns the chosen view label (one of ``VIEW_OPTIONS``)."""
    import streamlit as st
    if st.session_state.get(key) not in VIEW_OPTIONS:
        st.session_state.pop(key, None)
    choice = st.radio(label, VIEW_OPTIONS,
                      index=VIEW_OPTIONS.index(default),
                      horizontal=horizontal, key=key, help=view_help(what))
    if note:
        st.caption(note)
    return choice
