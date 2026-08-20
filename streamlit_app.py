"""Root-level Streamlit entry-point — routes between all the ticker apps.

Streamlit Community Cloud expects the main file at the repo root named
``streamlit_app.py``.  The applications live under ``app/``:

  * ``app/btc_hourly_app.py``   — the original Bitcoin forecaster (unchanged)
  * ``app/gldm_hourly_app.py``  — the gold (GLDM) forecaster (unchanged)
  * ``app/ticker_app.py``       — the generic, config-driven app that serves the
                                  new tickers (SOXX / GRID / XLE / REMX / …)
  * ``app/assistant_app.py``    — the AI Assistant chat, grounded in the
                                  committed artifacts and the repo's own docs

A single **Application** radio at the top of the sidebar lists every app.  The
choice is stored in ``st.session_state['gldm_active_app']``; reading session
state emits no Streamlit UI command, so it is safe to do here *before* the
selected app calls ``st.set_page_config``.

Keeping the two original apps byte-for-byte unchanged
-----------------------------------------------------
The BTC and GLDM apps each render their OWN two-option (BTC / GLDM) selector via
``st.radio(..., key="gldm_active_app")``.  To surface the five new tickers in the
*same* radio without editing those files, this router temporarily wraps
``st.radio`` while it runs one of them: the wrapped call for the
``gldm_active_app`` widget is transparently upgraded to the full app list, and
every other ``st.radio`` call passes straight through.  The generic ticker app
renders the full list natively, so no wrapping is needed for it.

Each sub-app is executed as the top-level script with ``__file__`` set to its
*actual* path, which every app relies on to resolve the repo root.  The compiled
code object is cached across reruns (see ``_exec_app``) so the large sub-apps are
not re-parsed on every rerun.
"""
from pathlib import Path

import streamlit as st

import sys
_APP_DIR = Path(__file__).resolve().parent / "app"
sys.path.insert(0, str(_APP_DIR))
import ticker_config  # noqa: E402

_ALL_APPS = (["OVERALL", "BTC", "GLDM", "GDXM"] + ticker_config.APP_KEYS
             + ["DAILYAUDIT", "HEALTH", "TARGETBOOK", "EXECUTEDBOOK", "ASSISTANT"])
_LABELS = {"OVERALL": "🧭  Overall Trading",
           "BTC": "₿  Bitcoin (BTC)", "GLDM": "🥇  Gold Trend (GLDM·UGL)",
           "GDXM": "⛏️  Gold Miners (GDX·NUGT)",
           "DAILYAUDIT": "🕵️  Daily Audit",
           "HEALTH": "🩺  Strategy Health",
           "TARGETBOOK": "📋  Target Book (IBKR)",
           "EXECUTEDBOOK": "✅  Executed Book (IBKR)",
           "ASSISTANT": "🤖  AI Assistant"}
for _k, _c in ticker_config.CONFIGS.items():
    _LABELS[_k] = f"{_c.emoji}  {_c.key} · {_c.name.split('(')[0].strip()[:22]}"


# ── sub-app execution (compile once, exec on every rerun) ────────────────────
@st.cache_resource(show_spinner=False)
def _compiled_app(path_str: str, _mtime: float):
    """Compile a sub-app's source once and reuse the code object across reruns.

    Streamlit runs this router on every rerun — each ~45 s auto-refresh and every
    app switch.  The previous ``runpy.run_path`` re-read and re-compiled the target
    file each time; for ``btc_hourly_app.py`` (~15 k lines / ~760 KB) that recompile
    was a large, invisible tax on switch latency.  Caching on ``(path, mtime)``
    recompiles only when the file actually changes on disk."""
    return compile(Path(path_str).read_text(), path_str, "exec")


def _exec_app(path: Path):
    """Execute a sub-app as the top-level script, mirroring the ``runpy.run_path``
    contract the apps depend on: ``__name__ == '__main__'`` and ``__file__`` set to
    the real path (each app resolves the repo root from ``__file__``)."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    code = _compiled_app(str(path), mtime)
    exec(code, {"__name__": "__main__", "__file__": str(path),
                "__builtins__": __builtins__})


# ── routing ────────────────────────────────────────────────────────────────
# Every app is one script run of this router that executes the selected sub-app.
# The selected app is persisted in the URL (``?app=...``) so a shared or refreshed
# link lands on the right app, and the heavy per-app compute stays warm in
# ``st.cache_data`` across runs, so switching back and forth is cheap.
#
# Resolution order: the sidebar radio widget (``gldm_active_app`` in session
# state) wins, because on the run right after a click Streamlit has already
# committed the new selection there; otherwise fall back to the URL, then to a
# sane default.
_qp_app = st.query_params.get("app")
_widget_app = st.session_state.get("gldm_active_app")
_choice = _widget_app or _qp_app or "OVERALL"
if _choice not in _ALL_APPS:
    _choice = "OVERALL"

# Sync the URL + selector widget to the resolved choice, then render that app in
# the SAME run.  On a switch, ``_widget_app`` already holds the app the user just
# clicked, so rendering ``_choice`` now *is* the switch taking effect — a single
# pass, no round-trip.
#
# An earlier version issued an ``st.rerun()`` here whenever the widget and the URL
# disagreed, so every switch cost TWO full script executions: the first only
# re-pointed the URL and bailed via ``st.rerun`` *before rendering anything*,
# leaving an empty intermediate frame (the previous app "ghosting" under a
# spinner) and roughly doubling the time-to-switch — long enough that users
# re-clicked the radio, queuing more reruns and dropping clicks.  (Older still was
# an injected ``window.parent.location.reload()``, which a sandboxed Streamlit
# component iframe silently blocks — it froze the app on the previous page.)
# Rendering the resolved choice directly avoids both.
st.query_params["app"] = _choice                     # keep the URL in sync
st.session_state["gldm_active_app"] = _choice        # seed the selector widget

# Per-app page title / icon (derived from the label); no-op the sub-apps' own
# set_page_config so it doesn't error or override — the same transparent-patch
# approach already used for the app selector below.
_lbl = _LABELS.get(_choice, _choice).strip()
_icon = _lbl.split()[0] if _lbl.split() else "🧭"
_title = _lbl[len(_icon):].strip() or str(_choice)
try:
    st.set_page_config(page_title=_title, page_icon=_icon, layout="wide",
                       initial_sidebar_state="expanded")
except Exception:
    pass
_orig_spc = st.set_page_config
st.set_page_config = lambda *a, **k: None            # sub-apps' calls become no-ops

# ── main-pane root: wipe the previous app's screen on a switch ───────────────
# Streamlit keeps the previous run's elements visible (dimmed) until the new run
# overwrites them or finishes.  Within ONE app that's the right behaviour — the
# ~45 s auto-refresh updates in place with no flash.  Across an app SWITCH it is
# the "ghosting" users see: the heavy apps compute for seconds before emitting
# their first element, and the whole previous app lingers on screen meanwhile.
#
# Fix: render every sub-app inside a single root container that always occupies
# the main pane's first slot.  On a normal rerun the root is a plain
# ``st.container()`` — the frontend keeps its children until they're replaced,
# so same-app refreshes stay flicker-free.  On a switch the root is created via
# ``st.empty().container()``: the ``Empty`` delta lands in the same slot and the
# frontend drops the old app's entire subtree *immediately*, then the container
# re-establishes the identical delta path for the new app to render into.
#
# ``_rendered_app`` is recorded the moment the wipe delta is emitted (deltas
# stream to the browser as they are created), NOT after the sub-app finishes —
# the apps' own "Refresh now" buttons call ``st.rerun()`` mid-render, and
# recording late would misread that follow-up pass as a switch and blank the
# screen on every refresh click.
_prev_rendered = st.session_state.get("_rendered_app")
_root = st.empty().container() if _prev_rendered != _choice else st.container()
st.session_state["_rendered_app"] = _choice


def _run_choice():
    if _choice == "OVERALL":
        # The combined cross-asset cockpit renders its own full selector.
        _exec_app(_APP_DIR / "overall_app.py")
    elif _choice == "DAILYAUDIT":
        # Freshness trail: per-app signal closes, Overall update, book publish.
        _exec_app(_APP_DIR / "daily_audit_app.py")
    elif _choice == "HEALTH":
        # Decay monitor: book-vs-replay tracking, drawdown tripwires,
        # rolling edge vs B&H, trade expectancy — reads committed artifacts only.
        _exec_app(_APP_DIR / "health_app.py")
    elif _choice == "TARGETBOOK":
        # Friendly viewer for the published signed IBKR target book.
        _exec_app(_APP_DIR / "target_book_app.py")
    elif _choice == "EXECUTEDBOOK":
        # Post-rebalance report: trades executed + current IBKR positions.
        _exec_app(_APP_DIR / "executed_book_app.py")
    elif _choice == "ASSISTANT":
        # Chat grounded in the committed artifacts + the repo's own docs.
        _exec_app(_APP_DIR / "assistant_app.py")
    elif _choice in ("BTC", "GLDM", "GDXM"):
        # Upgrade the original app's built-in BTC/GLDM selector to the full list,
        # without touching the app source.  Only the ``gldm_active_app`` widget is
        # rewritten; all other st.radio calls are untouched.
        _orig_radio = st.radio

        def _patched_radio(label, options=None, *args, **kwargs):
            if kwargs.get("key") == "gldm_active_app":
                return _orig_radio(label, _ALL_APPS,
                                   format_func=lambda x: _LABELS.get(x, x),
                                   key="gldm_active_app")
            return _orig_radio(label, options, *args, **kwargs)

        st.radio = _patched_radio
        try:
            _target = "btc_hourly_app.py" if _choice == "BTC" else "gldm_hourly_app.py"
            _exec_app(_APP_DIR / _target)
        finally:
            st.radio = _orig_radio
    else:
        # One of the new tickers — the generic app renders the full selector.
        _exec_app(_APP_DIR / "ticker_app.py")


try:
    with _root:                                       # all main-pane output lands here
        _run_choice()
finally:
    st.set_page_config = _orig_spc                    # restore for the next run
