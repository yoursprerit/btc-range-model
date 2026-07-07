"""Root-level Streamlit entry-point — routes between all the ticker apps.

Streamlit Community Cloud expects the main file at the repo root named
``streamlit_app.py``.  The applications live under ``app/``:

  * ``app/btc_hourly_app.py``   — the original Bitcoin forecaster (unchanged)
  * ``app/gldm_hourly_app.py``  — the gold (GLDM) forecaster (unchanged)
  * ``app/ticker_app.py``       — the generic, config-driven app that serves the
                                  five new tickers (SOXX / VEGN / GRID / XLE / REMX)

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

_choice = st.session_state.get("gldm_active_app", "OVERALL")
if _choice not in _ALL_APPS:
    _choice = "OVERALL"

# Switching apps used to leave the previous (often longer) app's trailing
# elements ghosting through as faded text at the bottom of the new page: because
# every app is one script run of THIS router, Streamlit reconciles elements by
# position and a shorter new app doesn't overwrite the old tail.  Fix: render
# each app inside a *route-keyed* container so every app has a distinct element
# identity — on switch, the previous route's subtree is dropped wholesale.
#
# A keyed container is a Streamlit command, so it must come *after*
# ``set_page_config``; we therefore set the page config once here (per-app title
# / icon derived from the label) and no-op the sub-apps' own calls — the same
# transparent-patch approach already used for the app selector below.
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
    with st.container(key=f"route_{_choice}"):
        _run_choice()
finally:
    st.set_page_config = _orig_spc                    # restore for the next run
