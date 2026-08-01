"""Unit tests for ``apply_closed_market_freeze`` (app/overall_core.py) — the
weekend/holiday guard on the Recommended Live book.

Synthetic gate dicts only (no network, no Streamlit): identity on US trading
days, published-weight pinning on closed days (no tilt re-sizing of sleeves
whose signal apps got no new bar), committed CLOSE dropping a sleeve to SATA,
committed OPEN funding from idle at the gate's tilted size (priority order,
idle-capped), per-action target rewriting, and the freeze metadata the UI
renders.  NYSE-holiday detection reuses ``freshness.us_market_holidays``.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import overall_core as oc  # noqa: E402

# Sat 2026-07-04 / Fri 2026-07-03 (July-4th NYSE closure) / Mon 2026-07-06
SATURDAY = "2026-07-04"
HOLIDAY = "2026-07-03"
TRADING_DAY = "2026-07-06"


def _gate(target, actions, sata=None):
    sata = sata if sata is not None else max(1.0 - sum(target.values()), 0.0)
    return dict(target=dict(target), sata=sata,
                actions=[dict(key=k, action=a) for k, a in actions],
                current={}, sata_now=0.0, priorities={}, priority_rank=[],
                n_active=0, n_open=0, n_close=0, sata_info=oc.SATA)


def test_identity_on_a_us_trading_day():
    g = _gate({"BTC": 0.5}, [("BTC", "HOLD")])
    assert oc.apply_closed_market_freeze(g, {"BTC": 0.4},
                                         today=TRADING_DAY) is g


def test_identity_without_a_published_book():
    g = _gate({"BTC": 0.5}, [("BTC", "HOLD")])
    assert oc.apply_closed_market_freeze(g, None, today=SATURDAY) is g
    assert oc.apply_closed_market_freeze(g, {}, today=SATURDAY) is g


def test_weekend_pins_every_sleeve_to_the_published_weight():
    # the tilt wants SOXX 0.22 / BTC 0.35 — but Saturday's book must keep the
    # published 0.18 / 0.30 exactly (no re-sizing of positions that can't trade)
    g = _gate({"BTC": 0.35, "SOXX": 0.22},
              [("BTC", "HOLD"), ("SOXX", "HOLD")])
    out = oc.apply_closed_market_freeze(g, {"BTC": 0.30, "SOXX": 0.18},
                                        today=SATURDAY)
    assert out["target"] == {"BTC": 0.30, "SOXX": 0.18}
    assert np.isclose(out["sata"], 0.52)
    assert out["freeze"]["pinned"] == ["BTC", "SOXX"]
    assert out["freeze"]["closed"] == [] and out["freeze"]["opened"] == []
    # the action rows' targets are rewritten to match the pinned book
    assert {a["key"]: a["target"] for a in out["actions"]} == \
        {"BTC": 0.30, "SOXX": 0.18}


def test_weekend_close_signal_drops_the_sleeve_to_sata():
    # a BTC-app exit on the Saturday bar — the one change that SHOULD move it
    g = _gate({"SOXX": 0.18}, [("BTC", "CLOSE"), ("SOXX", "HOLD")])
    out = oc.apply_closed_market_freeze(g, {"BTC": 0.30, "SOXX": 0.18},
                                        today=SATURDAY)
    assert "BTC" not in out["target"]
    assert out["target"] == {"SOXX": 0.18}
    assert np.isclose(out["sata"], 0.82)
    assert out["freeze"]["closed"] == ["BTC"]


def test_weekend_open_signal_funds_from_idle_at_the_gate_size():
    g = _gate({"BTC": 0.25, "SOXX": 0.18},
              [("BTC", "OPEN"), ("SOXX", "HOLD")])
    out = oc.apply_closed_market_freeze(g, {"SOXX": 0.18}, today=SATURDAY)
    # BTC wasn't in the book — funded at the gate's tilted 0.25 from idle
    assert np.isclose(out["target"]["BTC"], 0.25)
    assert np.isclose(out["target"]["SOXX"], 0.18)
    assert out["freeze"]["opened"] == ["BTC"]
    assert out["freeze"]["pinned"] == ["SOXX"]


def test_weekend_open_is_capped_by_remaining_idle():
    g = _gate({"BTC": 0.30}, [("BTC", "OPEN"), ("SOXX", "HOLD")])
    out = oc.apply_closed_market_freeze(g, {"SOXX": 0.95}, today=SATURDAY)
    # only 5% idle remains — the open cannot displace the pinned book
    assert np.isclose(out["target"]["BTC"], 0.05)
    assert np.isclose(out["target"]["SOXX"], 0.95)
    assert np.isclose(out["sata"], 0.0)


def test_nyse_holiday_freezes_like_a_weekend():
    # Fri Jul 3, 2026 — Independence Day observed; a weekday, market closed
    g = _gate({"BTC": 0.40}, [("BTC", "HOLD")])
    out = oc.apply_closed_market_freeze(g, {"BTC": 0.30}, today=HOLIDAY)
    assert out["target"] == {"BTC": 0.30}
    assert out["freeze"]["day"] == HOLIDAY


def test_negligible_published_weights_are_not_pinned():
    g = _gate({}, [("BTC", "HOLD")])
    out = oc.apply_closed_market_freeze(g, {"BTC": 0.0002}, today=SATURDAY)
    assert out["target"] == {}
    assert np.isclose(out["sata"], 1.0)
