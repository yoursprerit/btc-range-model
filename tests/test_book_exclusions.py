"""Positions listed in book_exclusions.json are left out of the published books;
their weight goes to the idle bucket (SATA up to the cap, rest cash) and is NOT
redistributed to the other names."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

_REPO = Path(__file__).resolve().parent.parent
for p in (_REPO, _REPO / "app", _REPO / "scripts"):
    sys.path.insert(0, str(p))

import publish_target_book as ptb  # noqa: E402


def _book():
    return SimpleNamespace(
        as_of="2026-10-21", weights={"BTC": 0.30, "SOXX": 0.10, "SOXL": 0.05},
        cash_weight=0.55, sata_cap=0.40,
        exec_price={"BTC": 1.0, "SOXX": 2.0, "SOXL": 3.0},
        actions=[dict(key="SOXX", action="enter", target=0.1),
                 dict(key="BTC", action="hold", target=0.3)])


def _assemble(excl):
    ptb.oc.fetch_sata = lambda: {"price": 100.0}
    return ptb.assemble_payloads(_book(), "Balanced", audit_stamp={}, basis={},
                                 exclusions=excl)


def test_no_exclusions_is_identity():
    paper, live, w, cash = _assemble({})
    assert set(w) == {"BTC", "SOXX", "SOXL", "SATA"}
    assert abs(sum(w.values()) + cash - 1.0) < 1e-9
    assert "excluded" not in live


def test_excluded_names_dropped_and_weight_goes_idle():
    paper, live, w, cash = _assemble({"paper": ["SOXX", "SOXL"], "live": ["SOXX", "SOXL"]})
    assert set(w) == {"BTC", "SATA"} and w["BTC"] == 0.30
    assert abs(sum(w.values()) + cash - 1.0) < 1e-9
    assert live["excluded"] == ["SOXL", "SOXX"] and "SOXX" not in live["exec_price"]
    assert next(a for a in live["actions"] if a["key"] == "SOXX")["action"] == "EXCLUDED"


def test_modes_are_independent():
    paper, live, w, cash = _assemble({"live": ["SOXX"]})
    assert "SOXX" in paper["weights"] and "SOXX" not in live["weights"]


def test_missing_file_means_none(tmp_path):
    assert ptb.load_exclusions(tmp_path / "nope.json") == {"paper": set(), "live": set()}
    f = tmp_path / "e.json"; f.write_text(json.dumps({"live": ["soxx"]}))
    assert ptb.load_exclusions(f) == {"paper": set(), "live": {"SOXX"}}
