"""Unit tests for ``executed_book.current_positions`` — the cost-basis position
view behind the Overall app's 💼 **Current Positions** sections.

Every other position block in the Overall cockpit is the *engine's* book: the
entry is a daily close the strategy decided on. This one is the *broker's*: the
cost basis is the IBKR average fill the executor reported, and the mark is
whatever price the tab has (the live spot on 🔴 Live, the chosen bar's official
close on 🕰️ Historical). These tests pin that P&L definition, the fallback when
a mark is missing, and the aggregate totals the section's header shows.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import executed_book as eb  # noqa: E402


def _payload(positions, **kw):
    return eb.build_payload(
        as_of=kw.get("as_of", "2026-08-18"), profile="Balanced",
        mode=kw.get("mode", "execute"), account="DU1234567",
        net_liq=kw.get("net_liq", 100000.0), cash=kw.get("cash", 2500.0),
        trades=[], positions=positions,
        account_mode=kw.get("account_mode", "paper"),
        realized_pnl=kw.get("realized_pnl"))


def _pos(key, shares, avg_cost, market_price=0.0):
    return dict(key=key, symbol=key, shares=shares, avg_cost=avg_cost,
                market_price=market_price,
                market_value=shares * market_price,
                unrealized_pnl=shares * (market_price - avg_cost))


def test_pnl_is_mark_minus_cost_basis_not_the_reported_pnl():
    """The report's own unrealized_pnl is stale the moment it is written — the
    section re-marks the broker's cost basis at the price the tab holds."""
    cp = eb.current_positions(
        _payload([_pos("SOXX", 50.0, 500.0, market_price=501.0)]),
        {"SOXX": 600.0})
    (row,) = cp["rows"]
    assert row["price"] == 600.0 and row["price_src"] == "live"
    assert row["cost_basis"] == pytest.approx(25_000.0)
    assert row["market_value"] == pytest.approx(30_000.0)
    assert row["pnl"] == pytest.approx(5_000.0)
    assert row["pnl_pct"] == pytest.approx(20.0)
    # totals mirror the per-row maths
    assert cp["cost"] == pytest.approx(25_000.0)
    assert cp["value"] == pytest.approx(30_000.0)
    assert cp["pnl"] == pytest.approx(5_000.0)
    assert cp["pnl_pct"] == pytest.approx(20.0)
    assert (cp["n"], cp["n_live"], cp["n_stale"]) == (1, 1, 0)


def test_price_map_accepts_plain_numbers_and_spot_quote_dicts():
    """The Live tab hands {key: {"price", "dchg"}}; the Historical tab can hand
    a plain {key: close}. Both are the same price map to this function."""
    cp = eb.current_positions(
        _payload([_pos("SOXX", 10.0, 100.0), _pos("GDX", 20.0, 50.0)]),
        {"SOXX": {"price": 110.0, "dchg": 1.5}, "GDX": 55.0})
    by = {r["key"]: r for r in cp["rows"]}
    assert by["SOXX"]["price"] == 110.0 and by["SOXX"]["dchg"] == 1.5
    assert by["GDX"]["price"] == 55.0 and by["GDX"]["dchg"] is None
    assert by["SOXX"]["pnl_pct"] == pytest.approx(10.0)
    assert by["GDX"]["pnl_pct"] == pytest.approx(10.0)


def test_missing_mark_falls_back_to_the_reports_own_price_and_is_flagged():
    """A holding with no live quote is still shown — marked at the price the
    report carried — but flagged so the UI can say the P&L is not current."""
    cp = eb.current_positions(
        _payload([_pos("SOXX", 50.0, 500.0, market_price=520.0)]), {})
    (row,) = cp["rows"]
    assert row["price_src"] == "report" and row["price"] == 520.0
    assert row["pnl"] == pytest.approx(1_000.0)
    assert (cp["n_live"], cp["n_stale"]) == (0, 1)


def test_unpriced_holding_contributes_no_pnl_and_no_market_value():
    """No mark anywhere: the row renders (the shares are real) but must not be
    silently counted as a zero-value position, which would show a -100% loss."""
    cp = eb.current_positions(_payload([_pos("SOXX", 50.0, 500.0)]), {})
    (row,) = cp["rows"]
    assert row["price_src"] == "none"
    assert row["market_value"] == 0.0 and row["pnl"] == 0.0
    assert row["pnl_pct"] is None
    assert cp["cost"] == pytest.approx(25_000.0)   # the basis is still real…
    assert cp["value"] == 0.0                      # …but nothing is marked
    assert cp["pnl"] == 0.0 and cp["pnl_pct"] is None


def test_totals_only_compare_priced_holdings():
    """Mixing a priced and an unpriced holding must not let the unpriced one's
    cost basis drag the headline P&L to a fictional loss."""
    cp = eb.current_positions(
        _payload([_pos("SOXX", 50.0, 500.0), _pos("GDX", 100.0, 90.0)]),
        {"GDX": 99.0})
    assert cp["cost"] == pytest.approx(25_000.0 + 9_000.0)
    assert cp["value"] == pytest.approx(9_900.0)
    assert cp["pnl"] == pytest.approx(900.0)       # GDX only
    assert cp["pnl_pct"] == pytest.approx(10.0)    # vs GDX's basis, not both


def test_zero_share_and_zero_cost_lines_are_dropped():
    """A flattened line or one with no average cost is not a holding."""
    cp = eb.current_positions(
        _payload([_pos("SOXX", 0.0, 500.0), _pos("GDX", 100.0, 0.0),
                  _pos("XLE", 10.0, 60.0)]),
        {"SOXX": 500.0, "GDX": 90.0, "XLE": 66.0})
    assert [r["key"] for r in cp["rows"]] == ["XLE"]
    assert cp["n"] == 1


def test_rows_are_largest_position_first():
    cp = eb.current_positions(
        _payload([_pos("XLE", 10.0, 60.0), _pos("SOXX", 50.0, 500.0),
                  _pos("GDX", 100.0, 90.0)]),
        {"XLE": 60.0, "SOXX": 500.0, "GDX": 90.0})
    assert [r["key"] for r in cp["rows"]] == ["SOXX", "GDX", "XLE"]


def test_run_metadata_rides_along_for_the_section_caption():
    """The caption names the account mode, the signal bar and the cash — read
    them off the same result so the header can never disagree with the rows."""
    cp = eb.current_positions(
        _payload([_pos("SOXX", 50.0, 500.0)], as_of="2026-08-18",
                 account_mode="live", mode="dry-run", cash=1234.0,
                 net_liq=98765.0),
        {"SOXX": 500.0})
    assert cp["as_of"] == "2026-08-18"
    assert cp["account_mode"] == "live" and cp["mode"] == "dry-run"
    assert cp["cash"] == pytest.approx(1234.0)
    assert cp["net_liq"] == pytest.approx(98765.0)
    assert cp["generated_at_utc"]


def test_no_report_is_an_empty_view_not_a_crash():
    """Before the executor has ever run there is no report at all — the section
    must render its 'nothing yet' state, not raise."""
    for payload in (None, {}, {"positions": []}):
        cp = eb.current_positions(payload, {"SOXX": 500.0})
        assert cp["rows"] == [] and cp["n"] == 0
        assert cp["cost"] == 0 and cp["value"] == 0 and cp["pnl"] == 0
        assert cp["pnl_pct"] is None


def test_bad_price_entries_are_treated_as_no_quote():
    """A spot fetch that half-failed hands back None/'' prices — those must fall
    back to the report's mark, never blow up the whole section."""
    cp = eb.current_positions(
        _payload([_pos("SOXX", 50.0, 500.0, market_price=510.0),
                  _pos("GDX", 100.0, 90.0, market_price=91.0)]),
        {"SOXX": {"price": None, "dchg": "n/a"}, "GDX": "not-a-number"})
    assert {r["price_src"] for r in cp["rows"]} == {"report"}
    assert all(r["dchg"] is None for r in cp["rows"])


def test_short_position_pnl_has_the_right_sign():
    """Long-only in practice, but a negative share count must not report a gain
    when the price rises."""
    cp = eb.current_positions(
        _payload([_pos("SOXX", -10.0, 500.0)]), {"SOXX": 550.0})
    (row,) = cp["rows"]
    assert row["pnl"] == pytest.approx(-500.0)
    assert row["pnl_pct"] == pytest.approx(-10.0)


def test_real_committed_report_marks_against_live_prices():
    """End-to-end against the report actually committed in the repo."""
    import json
    p = (Path(__file__).resolve().parent.parent / "data" / "overall"
         / "executed_book.json")
    if not p.exists():                      # a checkout without the artifact
        pytest.skip("no committed execution report")
    payload = json.loads(p.read_text())
    keys = [x["key"] for x in payload["positions"]]
    cp = eb.current_positions(payload, {k: 1.0 for k in keys})
    assert cp["n"] == len(keys) and cp["n_live"] == len(keys)
    assert cp["value"] == pytest.approx(sum(
        float(x["shares"]) for x in payload["positions"]))
    assert cp["pnl"] == pytest.approx(cp["value"] - cp["cost"])


# ════════════════════════════════════════════════════════════════════════════
# REALISED P&L — what a tilt-driven trim actually banked
# ════════════════════════════════════════════════════════════════════════════
# A partial sell is invisible in the cost-basis view on its own: IBKR leaves the
# remaining shares' avg_cost untouched, so the position's open P&L just scales
# down with the share count and the gain on the sold slice disappears. These
# pin the field that recovers it — and, just as importantly, pin that a report
# which never carried the field reports "unknown" rather than a confident zero.

def _pos_r(key, shares, avg_cost, realized=None, market_price=0.0):
    p = _pos(key, shares, avg_cost, market_price)
    if realized is not None:
        p["realized_pnl"] = realized
    return p


def test_trim_plus_realised_reconciles_to_the_untrimmed_position():
    """The whole point of the field. 194 sh @ 96.94 marked at 99.84 is +563
    open. Trim to 150 and the open P&L drops to +435 — but 44 shares booked
    +127.60 on the way out, and the two must add back to the original +563."""
    full = eb.current_positions(
        _payload([_pos("GDX", 194.0, 96.94)]), {"GDX": 99.84})
    trimmed = eb.current_positions(
        _payload([_pos_r("GDX", 150.0, 96.94, realized=127.60)],
                 realized_pnl=127.60), {"GDX": 99.84})
    assert full["pnl"] == pytest.approx(562.60, abs=0.01)
    assert trimmed["pnl"] == pytest.approx(435.00, abs=0.01)
    assert trimmed["realized"] == pytest.approx(127.60)
    assert trimmed["total_pnl"] == pytest.approx(full["pnl"], abs=0.01)


def test_trim_leaves_the_percentage_untouched():
    """avg_cost does not move on a partial long sell, so P&L % is invariant to
    the trim — only the dollar figure scales. Guards against a future change
    that quietly re-bases the percentage on the remaining shares."""
    a = eb.current_positions(_payload([_pos("GDX", 194.0, 96.94)]), {"GDX": 99.84})
    b = eb.current_positions(_payload([_pos("GDX", 150.0, 96.94)]), {"GDX": 99.84})
    assert a["rows"][0]["pnl_pct"] == pytest.approx(b["rows"][0]["pnl_pct"])
    assert b["rows"][0]["pnl"] < a["rows"][0]["pnl"]


def test_a_report_without_the_field_is_unknown_not_zero():
    """Every archived record written before this field existed. Reporting $0
    realised would claim the rebalance banked nothing, which nobody knows."""
    cp = eb.current_positions(_payload([_pos("GDX", 150.0, 96.94)]), {"GDX": 99.84})
    assert cp["realized"] is None
    assert cp["realized_known"] is False
    assert cp["realized_account"] is None and cp["realized_positions"] is None
    # and the total must not be inflated by a phantom zero
    assert cp["total_pnl"] == pytest.approx(cp["pnl"])
    assert cp["rows"][0]["realized"] is None


def test_a_genuine_zero_is_recorded_as_known():
    """A rebalance that placed no sells really did bank nothing — that is a
    fact, and must read differently from an unrecorded field."""
    cp = eb.current_positions(
        _payload([_pos_r("GDX", 150.0, 96.94, realized=0.0)], realized_pnl=0.0),
        {"GDX": 99.84})
    assert cp["realized"] == 0.0 and cp["realized_known"] is True


def test_account_figure_covers_names_closed_out_entirely():
    """IBKR drops a position the moment it hits zero shares, so a full exit has
    no row and its realised P&L can only come from the account-level figure —
    which must therefore win over the sum of the surviving positions."""
    cp = eb.current_positions(
        _payload([_pos_r("GDX", 150.0, 96.94, realized=127.60)],
                 realized_pnl=940.0), {"GDX": 99.84})
    assert cp["realized_positions"] == pytest.approx(127.60)
    assert cp["realized_account"] == pytest.approx(940.0)
    assert cp["realized"] == pytest.approx(940.0)      # the superset wins
    assert cp["total_pnl"] == pytest.approx(cp["pnl"] + 940.0)


def test_per_position_sum_is_used_when_only_positions_carry_it():
    """A report whose account-level read failed but whose positions came
    through still yields a realised figure rather than falling back to None."""
    cp = eb.current_positions(
        _payload([_pos_r("GDX", 150.0, 96.94, realized=127.60),
                  _pos_r("XLE", 50.0, 63.63, realized=-12.40)]),
        {"GDX": 99.84, "XLE": 64.46})
    assert cp["realized_account"] is None
    assert cp["realized"] == pytest.approx(115.20)
    assert cp["realized_known"] is True


def test_realised_survives_the_json_round_trip_through_build_payload():
    """The report is written to disk and read back by a different process, so
    the field has to survive serialisation — including its null form."""
    import json
    payload = _payload([_pos_r("GDX", 150.0, 96.94, realized=127.60),
                        _pos("XLE", 50.0, 63.63)], realized_pnl=940.0)
    back = json.loads(json.dumps(payload))
    assert back["realized_pnl"] == pytest.approx(940.0)
    by = {p["key"]: p for p in back["positions"]}
    assert by["GDX"]["realized_pnl"] == pytest.approx(127.60)
    assert by["XLE"]["realized_pnl"] is None      # present, explicitly null
    cp = eb.current_positions(back, {"GDX": 99.84, "XLE": 64.46})
    assert cp["realized"] == pytest.approx(940.0)


def test_adding_the_field_does_not_break_signature_verification():
    """Records signed before the field existed must still verify — the HMAC
    covers each payload as it was written, so an added key cannot retro-break
    the archive."""
    import target_book as tb
    old = _payload([_pos("GDX", 194.0, 96.94)])
    old.pop("realized_pnl")                       # a genuinely pre-field record
    signed = tb.loads(tb.dumps(old, "s3cret"))
    ok, _ = tb.verify_signature(signed, "s3cret")
    assert ok
    assert eb.current_positions(signed, {"GDX": 99.84})["realized_known"] is False


@pytest.mark.parametrize("raw", [None, "", "n/a", float("nan")])
def test_unreadable_realised_values_become_none(raw):
    cp = eb.current_positions(
        _payload([_pos_r("GDX", 150.0, 96.94, realized=raw)]), {"GDX": 99.84})
    assert cp["rows"][0]["realized"] is None
    assert cp["realized_known"] is False


def test_opt_float_rejects_nan_and_blanks_but_keeps_zero():
    assert eb.opt_float(0.0) == 0.0               # a real zero survives
    assert eb.opt_float("12.5") == pytest.approx(12.5)
    for bad in (None, "", "abc", float("nan")):
        assert eb.opt_float(bad) is None
