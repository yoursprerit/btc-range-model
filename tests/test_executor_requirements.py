"""The executor host installs requirements-ibkr.txt, NOT requirements.txt.

On 2026-10-12 the live run placed its orders, waited for the auction, and then
crashed with ``No module named 'cryptography'``: report_vault (imported the
moment the runner decides whether to publish) needs it, and it was listed only in
requirements.txt. This pins that class of bug: every third-party package the
executor's module chain imports (at module level OR lazily inside a function,
except optional ``try:`` imports) must be in requirements-ibkr.txt.
"""
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS, APP = ROOT / "scripts", ROOT / "app"

# the scheduled entry points (the run, the after-the-close reconcile, the health
# check). ibkr_rebalance (Option A) is deliberately absent: it imports the whole
# strategy engine and is run with BOTH requirements files installed.
ENTRY = ["ibkr_execute_accounts", "ibkr_execute_book", "ibkr_reconcile_accounts",
         "ibkr_gateway_healthcheck", "ibkr_heartbeat"]


def _local(name):
    for d in (SCRIPTS, APP):
        if (d / f"{name}.py").is_file():
            return d / f"{name}.py"
    return None


def _imports(path):
    """Module roots a file imports ANYWHERE — module level or lazily inside a
    function (the crash came from a lazy ``import report_vault`` inside the runner).
    Imports inside a ``try:`` body are skipped: those are the optional ones
    (``try: import streamlit …``) that are written to fail soft."""
    tree = ast.parse(path.read_text())
    roots, in_try = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for child in node.body:
                in_try |= {id(n) for n in ast.walk(child)}
    for node in ast.walk(tree):
        if id(node) in in_try:
            continue
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def _chain(entries):
    """(third-party roots, local modules) reachable from *entries* via module-level imports."""
    seen, third, todo = set(), set(), list(entries)
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        path = _local(name)
        if path is None:
            continue
        for root in _imports(path):
            if _local(root):
                todo.append(root)
            elif root not in sys.stdlib_module_names and root != "__future__":
                third.add(root)
    return third, seen


def _listed(req_file):
    pkgs = set()
    for line in (ROOT / req_file).read_text().splitlines():
        m = re.match(r"\s*([A-Za-z0-9_.\-]+)", line)
        if m and not line.lstrip().startswith("#"):
            pkgs.add(m.group(1).lower().replace("-", "_"))
    return pkgs


def test_requirements_ibkr_covers_everything_the_executor_imports():
    third, _ = _chain(ENTRY)
    listed = _listed("requirements-ibkr.txt")
    missing = sorted(p for p in third if p.lower() not in listed)
    assert not missing, (
        f"third-party modules imported by the executor chain but missing from "
        f"requirements-ibkr.txt: {missing} — the executor host would crash with "
        "ModuleNotFoundError after trading")


def test_the_chain_really_reaches_the_vault():
    """Guards the test above: it must actually see report_vault and its crypto import."""
    third, mods = _chain(ENTRY)
    assert "report_vault" in mods and "cryptography" in third


def test_cryptography_is_pinned_like_the_main_requirements():
    ibkr = (ROOT / "requirements-ibkr.txt").read_text()
    main = (ROOT / "requirements.txt").read_text()
    floor = lambda t: re.search(r"^cryptography\s*>=\s*([0-9.]+)", t, re.M).group(1)  # noqa: E731
    assert floor(ibkr) == floor(main)
