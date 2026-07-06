"""Root-level Streamlit entry-point — routes between the BTC and GLDM apps.

Streamlit Community Cloud expects the main file at the repo root named
``streamlit_app.py``.  Two applications live under ``app/``:

  * ``app/btc_hourly_app.py``   — the original Bitcoin forecaster (unchanged)
  * ``app/gldm_hourly_app.py``  — the gold (GLDM) forecaster

A radio button in each app's sidebar (grey panel) lets the user switch between
them.  The choice is stored in ``st.session_state['gldm_active_app']``; reading
session state does not emit any Streamlit UI command, so it is safe to do here
*before* the selected app calls ``st.set_page_config``.  Each app renders the
selector itself, so switching simply flips the state and reruns — this router
then dispatches to the other module.

``runpy.run_path`` preserves ``__file__`` as the *actual* path of the target
module, which both apps rely on to resolve the repo root.
"""
import runpy
from pathlib import Path

import streamlit as st

_APP_DIR = Path(__file__).resolve().parent / "app"
_choice = st.session_state.get("gldm_active_app", "BTC")
_target = "gldm_hourly_app.py" if _choice == "GLDM" else "btc_hourly_app.py"
runpy.run_path(str(_APP_DIR / _target), run_name="__main__")
