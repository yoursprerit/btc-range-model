"""The sidebar **Application** radio must offer the same apps everywhere.

Every page renders its own copy of the selector (``_ALL_APPS`` / ``_APP_LABELS``
with the shared ``gldm_active_app`` widget key), and ``streamlit_app.py`` holds
a ninth copy plus the routing branch that actually executes the chosen app.
That is nine places to keep in step by hand: miss one and the new app is either
invisible from that page, or selectable there and silently routed to the
default.

These tests read the literals straight out of the sources — no Streamlit
runtime — so adding an app without wiring it everywhere fails here rather than
in the browser.
"""
import ast
import re
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "app"))
import ticker_config as tc  # noqa: E402

#: Every source that renders the Application selector.
_SELECTOR_FILES = [
    _ROOT / "streamlit_app.py",
    *(_ROOT / "app" / n for n in (
        "overall_app.py", "ticker_app.py", "daily_audit_app.py", "health_app.py",
        "target_book_app.py", "executed_book_app.py", "assistant_app.py")),
]

#: The apps that are not config-driven tickers, in sidebar order.
_FIXED = ["OVERALL", "BTC", "GLDM", "GDXM", "DAILYAUDIT", "HEALTH",
          "TARGETBOOK", "EXECUTEDBOOK", "ASSISTANT"]

_LIST_RE = re.compile(
    r"_ALL_APPS\s*=\s*\((\[[^\]]*\])\s*\+\s*ticker_config\.APP_KEYS\s*\+\s*(\[[^\]]*\])\)",
    re.S)
_LABELS_RE = re.compile(r"_(?:APP_)?LABELS\s*=\s*(\{.*?\})\s*\n", re.S)


def _app_keys(path: Path) -> list[str]:
    m = _LIST_RE.search(path.read_text(encoding="utf-8"))
    assert m, f"{path.name}: could not find the _ALL_APPS literal"
    return ast.literal_eval(m.group(1)) + list(tc.APP_KEYS) + ast.literal_eval(m.group(2))


def _labels(path: Path) -> dict:
    m = _LABELS_RE.search(path.read_text(encoding="utf-8"))
    assert m, f"{path.name}: could not find the labels literal"
    return ast.literal_eval(m.group(1))


@pytest.mark.parametrize("path", _SELECTOR_FILES, ids=lambda p: p.name)
def test_every_page_offers_the_same_apps(path):
    assert _app_keys(path) == _app_keys(_ROOT / "streamlit_app.py")


@pytest.mark.parametrize("path", _SELECTOR_FILES, ids=lambda p: p.name)
def test_every_app_has_a_label_on_every_page(path):
    """A key with no label falls back to the bare key — a raw 'EXECUTEDBOOK' in
    the sidebar. Ticker labels are generated from ticker_config, so only the
    fixed apps need a literal."""
    labels = _labels(path)
    missing = [k for k in _FIXED if k not in labels]
    assert not missing, f"{path.name} is missing labels for {missing}"


def test_the_registry_is_what_we_expect():
    keys = _app_keys(_ROOT / "streamlit_app.py")
    assert set(keys) == set(_FIXED) | set(tc.APP_KEYS)
    assert keys[0] == "OVERALL", "the cockpit stays the first/default option"
    assert "ASSISTANT" in keys


def test_the_router_routes_every_app():
    """Each key must reach a branch in ``_run_choice`` — the ticker apps share
    the generic ``else``, everything else needs its own arm."""
    src = (_ROOT / "streamlit_app.py").read_text(encoding="utf-8")
    routed = set(re.findall(r'_choice\s*(?:==|in)\s*\(?([^\n:]+)', src))
    flat = {k.strip().strip('"\'') for chunk in routed
            for k in chunk.replace("(", "").replace(")", "").split(",")}
    for key in _FIXED:
        assert key in flat, f"streamlit_app.py never routes {key}"


def test_the_assistant_page_exists_and_is_routed():
    src = (_ROOT / "streamlit_app.py").read_text(encoding="utf-8")
    assert 'assistant_app.py' in src
    assert (_ROOT / "app" / "assistant_app.py").exists()
    assert (_ROOT / "app" / "assistant_core.py").exists()
