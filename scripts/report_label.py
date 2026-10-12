"""Display label for where an execution report was read from.

The Executed Book page labels each report with its path relative to the repo. A
report opened from the encrypted vault lives in a temp folder OUTSIDE the repo,
so ``Path.relative_to(repo)`` raised ``ValueError`` and took the whole page down
the first time the vault was used (2026-10-12). This keeps the label in one
tested place -- and out of ``app/*.py`` modules that are hashed into the P&L
snapshot's code fingerprint (only ``*_app.py`` page modules are exempt).
"""
from __future__ import annotations

from pathlib import Path


def source_label(path, repo_root) -> str:
    """Markdown code-span naming *path*: repo-relative when it is inside the
    repo; for a decrypted vault copy, ``accounts/<ACCT>/...`` under a lock marker
    (never the temp folder it happens to sit in); otherwise just the file name."""
    p = Path(path)
    try:
        return f"`{p.relative_to(Path(repo_root))}`"
    except ValueError:
        pass
    parts = p.parts
    if "accounts" in parts:
        i = len(parts) - 1 - parts[::-1].index("accounts")
        return "`🔒 encrypted vault: " + "/".join(parts[i:]) + "`"
    return f"`{p.name}`"
