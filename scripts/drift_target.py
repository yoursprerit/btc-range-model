"""Which target book an execution report's allocation-drift table is judged against.

The Executed Book page compared EVERY report with ``target_book.json`` -- the
PAPER book. A live account's report was therefore judged against a book with no
SATA (the paper book parks idle capital as cash) and 40 % cash, so each live
account showed SATA "target 0 %", cash "target 40 %" and drift to match, even
though the executor had traded the live book (SATA 30 %, cash 10 %).

Kept in ``scripts/`` rather than ``app/``: every ``app/*.py`` except ``*_app.py``
and ``pnl_snapshot.py`` is hashed into the P&L snapshot's code fingerprint.
"""
from __future__ import annotations

from pathlib import Path

PAPER_BOOK = "target_book.json"
LIVE_BOOK = "target_book_live.json"


def is_live(payload: dict) -> bool:
    return str((payload or {}).get("account_mode") or "paper").lower() == "live"


def current_target_path(payload: dict, data_dir) -> Path:
    """The book the executor traded for this report's account mode."""
    return Path(data_dir) / (LIVE_BOOK if is_live(payload) else PAPER_BOOK)


def book_mode_warning(book: dict, payload: dict) -> str | None:
    """A caption when the book being compared is not the report's own kind
    (a live report against a paper book, or the reverse), else None. A book with
    no ``book_mode`` stamp is not judged."""
    mode = str((book or {}).get("book_mode") or "").lower()
    want = "live" if is_live(payload) else "paper"
    if mode and mode != want:
        return (f"⚠️ This is a **{want}** report but the target book it is being "
                f"compared with is the **{mode}** book — the drift below is not "
                "meaningful.")
    return None
