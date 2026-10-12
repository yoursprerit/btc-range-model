"""The 📜 Strategy Change Log — data integrity and a smoke run of the page.

The log (``data/changelog/strategy_changelog.json``) is the human-readable record
of every strategy change under its version and release date.  These tests keep it
consistent with ``app/strategy_version.py`` (the code's own version history), so a
new strategy-logic generation cannot ship without a changelog entry, and an entry
cannot name a version, type or link that does not exist.
"""
import datetime as dt
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
import strategy_version as sv  # noqa: E402

LOG = ROOT / "data" / "changelog" / "strategy_changelog.json"
REPO_URL = "https://github.com/yoursprerit/btc-range-model/"
KINDS = {"sleeve", "allocator", "execution", "data", "analytics", "ui"}


@pytest.fixture(scope="module")
def doc():
    return json.loads(LOG.read_text(encoding="utf-8"))


def _d(s):
    return dt.date.fromisoformat(s)


def test_schema_and_versions_match_the_code_history(doc):
    assert doc["schema"] == "strategy-changelog/v1"
    vers = {v["id"]: v for v in doc["versions"]}
    assert set(vers) == {"v1", "v2"}
    assert [v["id"] for v in doc["versions"]] == ["v2", "v1"]          # newest first
    hist = dict(sv.VERSION_HISTORY)
    for vid, v in vers.items():
        assert v["effective"] == hist[vid], f"{vid} effective date != strategy_version.VERSION_HISTORY"
        assert v["label"] == sv.VERSION_LABELS[vid]
        assert v["summary"].strip() and v["links"]
    current = [v["id"] for v in doc["versions"] if v["status"] == "current"]
    assert current == [sv.STRATEGY_VERSION]                            # exactly the live version
    assert vers[sv.STRATEGY_VERSION]["ended"] is None
    # the retired generation ends the day before the next one starts
    assert _d(vers["v1"]["ended"]) == _d(vers["v2"]["effective"]) - dt.timedelta(days=1)


def test_every_entry_is_complete_dated_and_filed_under_a_real_version(doc):
    vers = {v["id"]: v for v in doc["versions"]}
    seen = set()
    for e in doc["entries"]:
        assert e["version"] in vers, e
        assert e["kind"] in KINDS, e
        assert e["title"].strip() and e["description"].strip(), e
        day = _d(e["date"])
        assert day <= dt.date.today() + dt.timedelta(days=1), f"future-dated: {e['title']}"
        key = (e["version"], e["date"], e["title"])
        assert key not in seen, f"duplicate entry {key}"
        seen.add(key)
        v = vers[e["version"]]
        if v.get("ended"):                       # a retired version takes no new changes
            assert day <= _d(v["ended"]), f"{e['title']} is dated after {v['id']} ended"
        elif e["version"] == "v2":               # V2 did not exist before its first signal day
            assert day >= _d(v["effective"]), f"{e['title']} predates V2"
        assert "impact" not in e or e["impact"].strip()
    for vid in vers:
        assert sum(e["version"] == vid for e in doc["entries"]) >= 5, f"{vid} has too few entries"


def test_links_point_at_this_repository(doc):
    for holder in list(doc["versions"]) + list(doc["entries"]):
        for l in holder.get("links", []):
            assert l["label"].strip()
            assert l["url"].startswith(REPO_URL), l
            assert re.search(r"/(pull/\d+|blob/main/[\w./-]+|commit/[0-9a-f]{7,40})$", l["url"]), l
    # each version points at the pull request that introduced it
    v = {x["id"]: x for x in doc["versions"]}
    assert any(l["url"].endswith("/pull/188") for l in v["v2"]["links"])
    assert any(l["url"].endswith("/pull/114") for l in v["v1"]["links"])


def test_the_page_kinds_match_the_log():
    src = (ROOT / "app" / "changelog_app.py").read_text(encoding="utf-8")
    page_kinds = set(re.findall(r'(?:^\s*|\{)"(\w+)": \("[^"]+", "#[0-9a-f]{6}"\)', src, re.M))
    assert page_kinds == KINDS


def test_the_page_renders_both_versions_with_their_links():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app" / "changelog_app.py"), default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    body = " ".join(m.value for m in at.markdown)
    for needle in ("Strategy Logic V2", "Strategy Logic V1", "CURRENT", "RETIRED",
                   "/pull/188", "/pull/114", "Balanced: fixed crypto-cluster split",
                   "Look-ahead bias removed"):
        assert needle in body, needle
    assert at.sidebar.radio[0].value == "CHANGELOG"
