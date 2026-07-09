"""Root-level Streamlit entry-point — routes between all the ticker apps.

Streamlit Community Cloud expects the main file at the repo root named
``streamlit_app.py``.  The applications live under ``app/``:

  * ``app/btc_hourly_app.py``   — the original Bitcoin forecaster (unchanged)
  * ``app/gldm_hourly_app.py``  — the gold (GLDM) forecaster (unchanged)
  * ``app/ticker_app.py``       — the generic, config-driven app that serves the
                                  new tickers (SOXX / GRID / XLE / REMX / …)

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

``runpy.run_path`` preserves ``__file__`` as the *actual* path of the target
module, which every app relies on to resolve the repo root.
"""
import runpy
from pathlib import Path

import streamlit as st

import sys
_APP_DIR = Path(__file__).resolve().parent / "app"
sys.path.insert(0, str(_APP_DIR))
import ticker_config  # noqa: E402

_ALL_APPS = ["OVERALL", "BTC", "GLDM"] + ticker_config.APP_KEYS
_LABELS = {"OVERALL": "🧭  Overall Trading",
           "BTC": "₿  Bitcoin (BTC)", "GLDM": "🥇  Gold (GLDM)"}
for _k, _c in ticker_config.CONFIGS.items():
    _LABELS[_k] = f"{_c.emoji}  {_c.key} · {_c.name.split('(')[0].strip()[:22]}"

# ── routing ────────────────────────────────────────────────────────────────
# Every app is one script run of this router that runpy-executes the selected
# sub-app.  The selected app is persisted in the URL (``?app=...``) so a shared or
# refreshed link lands on the right app, and the heavy per-app compute stays warm
# in ``st.cache_data`` across runs, so switching back and forth is cheap.
_qp_app = st.query_params.get("app")
_widget_app = st.session_state.get("gldm_active_app")
_choice = _widget_app or _qp_app or "OVERALL"
if _choice not in _ALL_APPS:
    _choice = "OVERALL"

# A real switch = the user picked an app (widget) different from what the URL has
# currently loaded.  Sync the URL and re-run cleanly so *only* the newly-selected
# app renders.
#
# Earlier this tried to hard-reload the browser via an injected
# ``components.html("<script>window.parent.location.reload()</script>")``.  That can
# never work: Streamlit renders components in an iframe whose sandbox does NOT grant
# ``allow-top-navigation``, so the browser blocks any attempt to reload/navigate the
# parent window ("Unsafe attempt to initiate navigation … the frame … is sandboxed").
# The reload silently failed, the branch ``st.stop()``-ed before rendering anything,
# and the app was left frozen on the previous page — e.g. stuck on SOXX when switching
# to Overall Trading.  A server-side ``st.rerun`` re-runs the router from the top with
# the URL already pointing at the new app, so the switch always takes.
if _qp_app is not None and _widget_app is not None and _widget_app != _qp_app:
    st.query_params["app"] = _choice
    st.session_state["gldm_active_app"] = _choice
    st.rerun()

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


def _run_choice():
    if _choice == "OVERALL":
        # The combined cross-asset cockpit renders its own full selector.
        runpy.run_path(str(_APP_DIR / "overall_app.py"), run_name="__main__")
    elif _choice in ("BTC", "GLDM"):
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
            _target = "gldm_hourly_app.py" if _choice == "GLDM" else "btc_hourly_app.py"
            runpy.run_path(str(_APP_DIR / _target), run_name="__main__")
        finally:
            st.radio = _orig_radio
    else:
        # One of the new tickers — the generic app renders the full selector.
        runpy.run_path(str(_APP_DIR / "ticker_app.py"), run_name="__main__")


try:
    _run_choice()
finally:
    st.set_page_config = _orig_spc                    # restore for the next run
