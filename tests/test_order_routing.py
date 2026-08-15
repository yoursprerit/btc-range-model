"""Order routing / pricing for the IBKR executor (paper AND live).

These cover the pure helpers in ``scripts/ibkr_common.py`` — the marketable
limit's price, its tick rounding, and the MOC cutoff/wait arithmetic — so the
rules that decide what price hits the market are testable without a broker
connection.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import ibkr_common as ic  # noqa: E402


# ── marketable limit pricing ────────────────────────────────────────────────
def test_buy_prices_through_the_ask():
    # the side we must cross is the ask; 0.5% through 100.00 → 100.50
    lmt, basis = ic.marketable_limit_price(
        "BUY", {"bid": 99.90, "ask": 100.00, "last": 99.95, "close": 99.0}, 0.005)
    assert basis == "ask"
    assert lmt == pytest.approx(100.50)


def test_sell_prices_through_the_bid():
    lmt, basis = ic.marketable_limit_price(
        "SELL", {"bid": 100.00, "ask": 100.10, "last": 99.95, "close": 99.0}, 0.005)
    assert basis == "bid"
    assert lmt == pytest.approx(99.50)


def test_limit_is_marketable_against_the_touch():
    """The whole point: the limit must CROSS the spread, not sit inside it."""
    q = {"bid": 250.10, "ask": 250.40}
    buy, _ = ic.marketable_limit_price("BUY", q)
    sell, _ = ic.marketable_limit_price("SELL", q)
    assert buy > q["ask"], "a buy limit below the ask would rest, not fill"
    assert sell < q["bid"], "a sell limit above the bid would rest, not fill"


def test_falls_back_through_last_then_close():
    # IBKR reports absent fields as nan or -1 — both must be skipped
    lmt, basis = ic.marketable_limit_price(
        "BUY", {"bid": -1.0, "ask": float("nan"), "last": 50.0, "close": 49.0})
    assert basis == "last" and lmt == pytest.approx(50.25)
    lmt, basis = ic.marketable_limit_price(
        "BUY", {"ask": None, "last": 0.0, "close": 49.0})
    assert basis == "close" and lmt == pytest.approx(49.25, abs=0.01)


def test_unusable_quote_reports_why():
    lmt, why = ic.marketable_limit_price("BUY", {})
    assert lmt is None and "no usable" in why
    lmt, why = ic.marketable_limit_price("BUY", {"ask": 10.0}, cap=1.5)
    assert lmt is None and "out of range" in why
    lmt, why = ic.marketable_limit_price("HOLD", {"ask": 10.0})
    assert lmt is None and "unknown action" in why


def test_rounding_never_pulls_the_limit_inside_the_spread():
    # a raw buy limit of 10.0301 must round UP to 10.04, not down to 10.03
    assert ic.round_to_tick(10.0301, "BUY") == pytest.approx(10.04)
    assert ic.round_to_tick(10.0399, "SELL") == pytest.approx(10.03)
    # already on the grid → unchanged in both directions
    assert ic.round_to_tick(10.03, "BUY") == pytest.approx(10.03)
    assert ic.round_to_tick(10.03, "SELL") == pytest.approx(10.03)


def test_sub_dollar_names_use_the_finer_tick():
    lmt, _ = ic.marketable_limit_price("BUY", {"ask": 0.5000}, 0.005)
    assert lmt == pytest.approx(0.5025)          # 4-decimal grid, not 0.51


def test_cap_scales_the_distance_through_the_touch():
    tight, _ = ic.marketable_limit_price("BUY", {"ask": 100.0}, 0.001)
    wide, _ = ic.marketable_limit_price("BUY", {"ask": 100.0}, 0.02)
    assert tight == pytest.approx(100.10)
    assert wide == pytest.approx(102.00)


# ── MOC cutoff + wait ───────────────────────────────────────────────────────
def _et(stamp: str) -> pd.Timestamp:
    return pd.Timestamp(stamp, tz="America/New_York")


def test_moc_accepted_at_the_executor_slot():
    """2:30 PM CT = 3:30 PM ET — 20 min inside the 15:50 ET cutoff. This is
    what makes MOC usable at all; the old 8:45-AM-CT slot never could."""
    ok, why = ic.moc_entry_open(_et("2026-08-04 15:30"))
    assert ok, why


def test_moc_refused_after_the_cutoff():
    ok, why = ic.moc_entry_open(_et("2026-08-04 15:51"))
    assert not ok
    assert "cutoff" in why


def test_moc_cutoff_edge_is_inclusive():
    ok, _ = ic.moc_entry_open(_et("2026-08-04 15:50"))
    assert ok, "15:50:00 is the cutoff instant, not past it"


def test_moc_cutoff_tracks_eastern_not_utc():
    # 19:30 UTC is 3:30 PM EDT — inside the cutoff. A naive UTC comparison
    # against 15:50 would wrongly reject it.
    ok, _ = ic.moc_entry_open(pd.Timestamp("2026-08-04T19:30:00Z"))
    assert ok


def test_moc_wait_spans_the_close_plus_reporting_buffer():
    # 15:30 ET → 30 min to the close, plus the 120 s buffer
    secs = ic.moc_wait_seconds(_et("2026-08-04 15:30"), buffer_s=120.0)
    assert secs == pytest.approx(30 * 60 + 120)


def test_moc_wait_is_clamped_and_never_negative():
    assert ic.moc_wait_seconds(_et("2026-08-04 09:30"), max_wait_s=600.0) == 600.0
    # already past the close → no negative sleep, just the buffer
    assert ic.moc_wait_seconds(_et("2026-08-04 16:30"), buffer_s=120.0) == 0.0


def test_moc_wait_fits_inside_the_systemd_ceiling():
    """deploy/systemd/ibkr-executor.service caps a run at 3600 s; the MOC hold
    from the 2:30-PM-CT slot must fit with room for the pull and the fills."""
    assert ic.moc_wait_seconds(_et("2026-08-04 15:30")) < 3600


# ── limit → market escalation ───────────────────────────────────────────────
def test_escalates_the_whole_order_when_nothing_filled():
    assert ic.unfilled_remainder(100, 0) == 100.0


def test_escalates_only_the_unfilled_part_of_a_partial():
    assert ic.unfilled_remainder(100, 30) == 70.0


def test_no_escalation_when_fully_filled():
    assert ic.unfilled_remainder(100, 100) == 0.0
    assert ic.unfilled_remainder(100, 100.0000001) == 0.0


def test_rounding_crumbs_never_become_an_order():
    # a whole-share account must not send a 1-share order for a 0.4 remainder
    assert ic.unfilled_remainder(100, 99.6) == 0.0
    # the remainder is floored (not the fill): 100 − 0.5 = 99.5 → 99 shares
    assert ic.unfilled_remainder(100.0, 0.5) == 99.0


def test_fractional_accounts_keep_the_exact_remainder():
    assert ic.unfilled_remainder(10.5, 2.25, fractional=True) == pytest.approx(8.25)


# ── routing guards ──────────────────────────────────────────────────────────
class _FakeBroker:
    """Just the routing logic — no IB connection."""
    _resolve_order_type = ic.Broker._resolve_order_type


def test_moc_falls_back_to_a_limit_after_the_cutoff():
    got, frac = _FakeBroker()._resolve_order_type(
        ic.ORDER_MOC, False, _et("2026-08-04 15:55"))
    assert got == ic.ORDER_MARKETABLE_LIMIT
    assert frac is False


def test_moc_forces_whole_shares():
    got, frac = _FakeBroker()._resolve_order_type(
        ic.ORDER_MOC, True, _et("2026-08-04 15:30"))
    assert got == ic.ORDER_MOC
    assert frac is False, "MOC does not accept fractional quantities"


def test_non_moc_types_pass_through_untouched():
    for t in (ic.ORDER_MARKET, ic.ORDER_MARKETABLE_LIMIT):
        got, frac = _FakeBroker()._resolve_order_type(t, True, _et("2026-08-04 15:55"))
        assert (got, frac) == (t, True)


def test_unknown_order_type_is_rejected():
    with pytest.raises(ValueError, match="unknown order type"):
        _FakeBroker()._resolve_order_type("iceberg", False)


def test_default_is_the_protected_type():
    """A naked market order must never be what you get by not choosing."""
    assert ic.ORDER_TYPES[0] == ic.ORDER_MARKETABLE_LIMIT
