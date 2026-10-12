"""The Executed Book page must label a report it read from the encrypted vault.

2026-10-12: the first time the Streamlit app read the vault, every IBKR page died
with ``ValueError: '/tmp/reportvault-…/accounts/U…/executed_book_live.json' is not
in the subpath of '/mount/src/btc-range-model'`` -- the decrypted copy sits in a
temp folder outside the repo and the page labelled it with ``Path.relative_to``.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import report_label as rl  # noqa: E402

REPO = Path("/mount/src/btc-range-model")


def test_a_path_inside_the_repo_is_labelled_relative_as_before():
    p = REPO / "data" / "overall" / "executed_book.json"
    assert rl.source_label(p, REPO) == "`data/overall/executed_book.json`"


def test_a_vault_copy_outside_the_repo_no_longer_raises():
    p = Path("/tmp/reportvault-qg8yydsq/accounts/U1111111/executed_book_live.json")
    label = rl.source_label(p, REPO)
    assert label == "`🔒 encrypted vault: accounts/U1111111/executed_book_live.json`"
    assert "/tmp" not in label and "reportvault" not in label


def test_a_vault_archive_record_keeps_its_dated_name():
    p = Path("/tmp/vaultfetch-x/accounts/U1/executed_archive/2026-10-09_live.json")
    assert rl.source_label(p, REPO).endswith("accounts/U1/executed_archive/2026-10-09_live.json`")


def test_the_last_accounts_folder_wins():
    p = Path("/tmp/accounts/reportvault-1/accounts/U9/executed_book_live.json")
    assert rl.source_label(p, REPO) == "`🔒 encrypted vault: accounts/U9/executed_book_live.json`"


def test_any_other_outside_path_shows_only_the_file_name():
    assert rl.source_label("/somewhere/else/report.json", REPO) == "`report.json`"


def test_strings_and_paths_are_both_accepted():
    assert rl.source_label(str(REPO / "a" / "b.json"), str(REPO)) == "`a/b.json`"


def test_the_page_never_labels_a_report_path_with_a_bare_relative_to():
    """Static guard: the report/archive paths on this page can be vault copies, so
    only the repo-constant paths (target-book archive, C2 snapshot) may still use
    ``relative_to`` directly."""
    src = (ROOT / "app" / "executed_book_app.py").read_text()
    offenders = re.findall(r"(?:rec\[['\"]path['\"]\]|\b_path)\.relative_to\(", src)
    assert not offenders, f"report path labelled with relative_to again: {offenders}"
    assert "_rl.source_label(" in src


def test_the_helper_stays_out_of_the_hashed_app_modules():
    """app/*.py (except *_app.py and pnl_snapshot.py) is hashed into the P&L
    snapshot's code fingerprint, so a helper dropped there would invalidate it."""
    assert (ROOT / "scripts" / "report_label.py").is_file()
    assert not (ROOT / "app" / "report_label.py").exists()
