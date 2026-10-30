"""After-close price overlay: the pure pieces behind same-day executed P&L."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "app"))
sys.path.insert(0, str(_ROOT / "scripts"))
import overall_core as oc          # noqa: E402
import pull_close_overlay as pco   # noqa: E402


def _closes():
    idx = pd.to_datetime(["2026-10-28", "2026-10-29"])
    return pd.DataFrame({"XLE": [68.0, 69.0], "SATA": [96.0, 96.2]}, index=idx)


def _doc(session="2026-10-30", **closes):
    return {"schema": oc.CLOSE_OVERLAY_SCHEMA, "session": session,
            "closes": closes or {"XLE": 70.0}}


def test_overlay_appends_one_bar_and_leaves_input_untouched():
    c = _closes()
    out = oc.apply_close_overlay(c, _doc(XLE=70.0, SATA=96.4))
    assert out.index[-1] == pd.Timestamp("2026-10-30")
    assert out.loc["2026-10-30", "XLE"] == 70.0 and out.loc["2026-10-30", "SATA"] == 96.4
    assert len(c) == 2


def test_overlay_missing_key_stays_nan_for_carry_forward():
    out = oc.apply_close_overlay(_closes(), _doc(XLE=70.0))
    assert np.isnan(out.loc["2026-10-30", "SATA"])


def test_pinned_data_that_caught_up_wins():
    c = _closes()
    assert oc.apply_close_overlay(c, _doc(session="2026-10-29")) is c
    assert oc.apply_close_overlay(c, _doc(session="2026-10-27")) is c


def test_bad_print_is_dropped_and_empty_overlay_is_a_noop():
    c = _closes()
    assert oc.apply_close_overlay(c, _doc(XLE=7.0)) is c        # -90% = bad print
    assert oc.apply_close_overlay(c, None) is c
    assert oc.apply_close_overlay(None, _doc()) is None


def test_load_rejects_foreign_or_empty(tmp_path):
    p = tmp_path / "o.json"
    p.write_text('{"schema": "x", "session": "2026-10-30", "closes": {"XLE": 1}}')
    assert oc.load_close_overlay(p) is None
    p.write_text('{"schema": "close-overlay/v1", "session": "2026-10-30", "closes": {}}')
    assert oc.load_close_overlay(p) is None
    assert oc.load_close_overlay(tmp_path / "missing.json") is None


def test_session_guard_follows_the_four_ten_et_rule():
    assert pco.session_for(pd.Timestamp("2026-10-30 20:05", tz="UTC")) is None   # 4:05 ET
    assert pco.session_for(pd.Timestamp("2026-10-30 20:25", tz="UTC")) == pd.Timestamp("2026-10-30")
    assert pco.session_for(pd.Timestamp("2026-10-31 20:25", tz="UTC")) is None   # Saturday


def test_build_overlay_lists_missing_and_skips_unchanged():
    now = pd.Timestamp("2026-10-30 20:30", tz="UTC")
    doc = pco.build_overlay(pd.Timestamp("2026-10-30"),
                            {"XLE": 70.123, "ERX": None, "OIH": float("nan")}, now)
    assert doc["closes"] == {"XLE": 70.123} and doc["missing"] == ["ERX", "OIH"]
    assert pco.unchanged(dict(doc, fetched_at_utc="earlier"), doc)
    assert not pco.unchanged(None, doc)
    assert pco.build_overlay(pd.Timestamp("2026-10-30"), {"XLE": None}, now) is None


def test_overlay_keys_exclude_spot_signals_held_as_etfs():
    k = pco.overlay_keys()
    assert "BTC" not in k and "ETH" not in k and k["XLE"] == "XLE" and "SATA" in k


def test_ibkr_replay_reaches_the_overlay_session():
    runs = [dict(date=pd.Timestamp("2026-10-28"), shares={"XLE": 100.0},
                 price={"XLE": 68.0}, cash=3200.0),
            dict(date=pd.Timestamp("2026-10-30"), shares={"XLE": 150.0},
                 price={"XLE": 69.5}, cash=0.0)]
    base = _closes()[["XLE"]]
    assert oc.ibkr_executed_replay(runs, base)["equity"].index[-1] == pd.Timestamp("2026-10-29")
    ext = oc.apply_close_overlay(base, _doc(XLE=70.0))
    eq = oc.ibkr_executed_replay(runs, ext)["equity"]
    assert eq.index[-1] == pd.Timestamp("2026-10-30")
    # the 10/29 holding (100 sh + 3,200 cash) marked 69 → 70 earns the last bar
    assert abs(eq.iloc[-1] / eq.iloc[-2] - (100 * 70 + 3200) / (100 * 69 + 3200)) < 1e-9
