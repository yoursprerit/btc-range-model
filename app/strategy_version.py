"""Single source of truth for the deliberate strategy-logic version.

BUMP ``STRATEGY_VERSION`` whenever the Overall logic or any per-asset
strategy changes materially — gate/tilt/water-fill rules, optimiser
objective/caps, per-asset entry/exit logic or parameters, universe changes.
Display/UI-only changes do NOT warrant a bump.  The publisher stamps every
Targetbook it commits with this version (plus the publishing commit SHA), and
the Overall P&L section segments the as-published record wherever the stamp
changes, so old-logic and current-logic performance are never conflated.

Kept dependency-light (stdlib only) so the read-only artifact apps
(📋 Targetbook, ✅ Executed Book, 🕵️ Daily Audit, 🩺 Strategy Health) can show
the badge without importing the heavy engine stack.  ``overall_core``
re-exports the constant for the engine/publisher side.
"""
STRATEGY_VERSION = "v1"

# First signal day (as_of) counted into the current version's as-published
# record.  BUMP TOGETHER with STRATEGY_VERSION — set it to the day of the
# bump.  The Overall P&L's as-published view only admits books stamped with
# the current version on/after this date, so a same-day re-publish of an
# older bar under new code can never pull old-strategy days into the record.
STRATEGY_VERSION_START = "2026-07-31"

BADGE_COLOR = "#7c3aed"        # the logic-provenance purple, used app-wide


def badge_caption() -> str:
    """Plain-text one-liner (markdown) — fallback where HTML can't render."""
    return (f"⚙️ Strategy logic version: **`{STRATEGY_VERSION}`** — bumped on "
            "material strategy changes; every published Targetbook is "
            "stamped with the version that produced it.")


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
        "<span style='color:#64748b;font-size:12px'>bumped on material "
        "strategy changes · every published Targetbook is stamped with the "
        "version that produced it</span></div>")


def render_badge() -> None:
    """Render the pill in a Streamlit app (import kept local so this module
    stays importable engine-side without Streamlit)."""
    import streamlit as st
    st.markdown(badge_html(), unsafe_allow_html=True)
