"""The 📈 P&L section's per-asset breakdown, trade log and realized-P&L tables must
read the SELECTED strategy generation's per-sleeve results (``_results_v``), not the
page-level ``results`` (always the live V2 universe) — otherwise the bar chart and table
never change when the V1 / V2 / Combined selector does (and V1 would show XOP, not OIH).
Source-level guard: the page is a Streamlit script, so it cannot be imported here."""
from pathlib import Path

SRC = (Path(__file__).resolve().parents[1] / "app" / "overall_app.py").read_text()


def test_selected_version_results_feed_the_per_asset_tables():
    assert "_results_v" in SRC and "_parents_v" in SRC
    assert "ov.per_asset_slice_metrics(_results_v," in SRC
    assert "ov.per_asset_slice_metrics(results," not in SRC
    assert "ov.trade_log_since(\n                _results_v," in SRC
    assert '_res_by_key = {r["key"]: r for r in _results_v}' in SRC


def test_per_asset_tables_group_by_the_selected_versions_parents():
    tail = SRC[SRC.index("# ── per-asset breakdown — opt-in via toggle"):]
    assert tail.count("for pk, grp in _parents_v:") == 2
    assert "for pk, grp in parents:" not in tail
