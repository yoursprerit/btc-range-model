"""Live execution reports must be judged against the LIVE target book.

2026-10-12: every live IBKR account showed SATA "target 0 %" (and cash "target
40 %") in the allocation-drift table, with +30 pp / -30 pp drift to match: the page
compared every report with target_book.json -- the PAPER book, which parks idle
capital as cash -- while the executor had traded target_book_live.json (SATA 30 %,
cash 10 %).
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import drift_target as dt  # noqa: E402

DATA = ROOT / "data" / "overall"


def test_a_live_report_is_judged_against_the_live_book():
    assert dt.current_target_path({"account_mode": "live"}, DATA) == DATA / "target_book_live.json"
    assert dt.current_target_path({"account_mode": "LIVE"}, DATA) == DATA / "target_book_live.json"


def test_a_paper_or_unstamped_report_keeps_the_paper_book():
    assert dt.current_target_path({"account_mode": "paper"}, DATA) == DATA / "target_book.json"
    assert dt.current_target_path({}, DATA) == DATA / "target_book.json"
    assert dt.current_target_path(None, DATA) == DATA / "target_book.json"


def test_a_book_of_the_wrong_kind_is_called_out():
    live = {"account_mode": "live"}
    assert "paper" in dt.book_mode_warning({"book_mode": "paper"}, live)
    assert "live" in dt.book_mode_warning({"book_mode": "live"}, {"account_mode": "paper"})


def test_a_matching_or_unstamped_book_raises_no_warning():
    assert dt.book_mode_warning({"book_mode": "live"}, {"account_mode": "live"}) is None
    assert dt.book_mode_warning({"book_mode": "paper"}, {"account_mode": "paper"}) is None
    assert dt.book_mode_warning({}, {"account_mode": "live"}) is None
    assert dt.book_mode_warning(None, None) is None


def test_both_published_books_park_idle_capital_in_capped_sata():
    """Since the SATA cap both modes publish a SATA leg (capped) plus the
    excess in cash — the paper book no longer differs from the live one."""
    paper = json.loads((DATA / "target_book.json").read_text())
    live = json.loads((DATA / "target_book_live.json").read_text())
    assert paper.get("book_mode") == "paper" and live.get("book_mode") == "live"
    for book in (paper, live):
        assert 0 < book["weights"].get("SATA", 0) <= book["sata_cap"] + 1e-9
        assert book["cash_weight"] >= 0


def test_the_page_picks_its_target_through_the_helper():
    src = (ROOT / "app" / "executed_book_app.py").read_text()
    assert "_dt.current_target_path(payload" in src
    # the default branch must not fall back to the bare paper-book constant
    block = src[src.index("def _drift_section"):src.index("def _download")]
    assert not re.search(r"tgt_path = TARGET_PATH\b", block)


def test_the_helper_stays_out_of_the_hashed_app_modules():
    assert (ROOT / "scripts" / "drift_target.py").is_file()
    assert not (ROOT / "app" / "drift_target.py").exists()
