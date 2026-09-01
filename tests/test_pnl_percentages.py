"""Unit tests for the executed book's P&L percentages — the ``Unreal. P&L %``
and ``Realised P&L %`` columns beside the dollar figures on the ✅ Executed Book
page's positions table.

A dollar P&L is unreadable across a book whose positions differ tenfold in size,
so each figure is also shown as a return on the cost basis that produced it. The
two bases are NOT the same: unrealised P&L is earned by the shares still held,
realised P&L by the shares this run SOLD. These tests pin both, and pin that a
missing basis stays "—" rather than collapsing into a confident 0 %.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import executed_book as eb  # noqa: E402


def _pos(key="SOXX", shares=100.0, avg_cost=200.0, unrealized_pnl=1_000.0,
         realized_pnl=None, market_price=210.0):
    return dict(key=key, symbol=key, shares=shares, avg_cost=avg_cost,
                market_price=market_price, market_value=shares * market_price,
                unrealized_pnl=unrealized_pnl, realized_pnl=realized_pnl)


def _sell(key="SOXX", filled=20.0, avg_fill_price=210.0):
    return dict(key=key, symbol=key, action="SELL", qty=filled, price=avg_fill_price,
                status="Filled", filled=filled, avg_fill_price=avg_fill_price)


# ── unrealised % ─────────────────────────────────────────────────────────────

def test_unrealised_pct_is_pnl_over_shares_times_avg_cost():
    u, _ = eb.position_pnl_pct(_pos(shares=100.0, avg_cost=200.0,
                                    unrealized_pnl=1_000.0), [])
    assert u == pytest.approx(5.0)          # 1,000 on a 20,000 basis


def test_unrealised_pct_is_negative_for_a_losing_position():
    u, _ = eb.position_pnl_pct(_pos(unrealized_pnl=-2_000.0), [])
    assert u == pytest.approx(-10.0)


def test_no_cost_basis_is_not_zero_percent():
    """A position with no average cost cannot be expressed as a return — the
    column must say "—", which is None here, never 0 %."""
    u, r = eb.position_pnl_pct(_pos(avg_cost=0.0, realized_pnl=50.0),
                               [_sell()])
    assert u is None and r is None


# ── realised % ───────────────────────────────────────────────────────────────

def test_realised_pct_is_measured_against_what_the_sold_shares_cost():
    """IBKR leaves the remaining shares' avg_cost untouched on a trim, so that
    same average is the sold slice's cost basis: $200 booked on 20 × $200."""
    _, r = eb.position_pnl_pct(_pos(avg_cost=200.0, realized_pnl=200.0),
                               [_sell(filled=20.0)])
    assert r == pytest.approx(5.0)          # 200 on a 4,000 basis


def test_realised_pct_ignores_buys_and_unfilled_orders():
    trades = [_sell(filled=20.0),
              dict(key="SOXX", symbol="SOXX", action="BUY", qty=10.0,
                   status="Filled", filled=10.0, avg_fill_price=210.0),
              dict(key="SOXX", symbol="SOXX", action="SELL", qty=40.0,
                   status="PLANNED", filled=0.0, avg_fill_price=0.0)]
    _, r = eb.position_pnl_pct(_pos(avg_cost=200.0, realized_pnl=200.0), trades)
    assert r == pytest.approx(5.0)          # only the 20 that actually printed


def test_realised_pct_only_counts_this_instruments_sells():
    trades = [_sell(key="SOXX", filled=20.0), _sell(key="ERX", filled=500.0)]
    _, r = eb.position_pnl_pct(_pos(key="SOXX", avg_cost=200.0,
                                    realized_pnl=200.0), trades)
    assert r == pytest.approx(5.0)


def test_realised_pct_is_none_when_the_report_predates_the_field():
    """realized_pnl absent means "not recorded" — the percentage must not
    invent a 0 % out of it."""
    _, r = eb.position_pnl_pct(_pos(realized_pnl=None), [_sell()])
    assert r is None


def test_realised_pct_is_none_without_a_fill_to_measure_against():
    _, r = eb.position_pnl_pct(_pos(realized_pnl=200.0), [])
    assert r is None


def test_a_genuine_zero_realised_still_reads_as_zero_percent():
    _, r = eb.position_pnl_pct(_pos(realized_pnl=0.0), [_sell(filled=20.0)])
    assert r == pytest.approx(0.0)


# ── book-level totals (the caption under the table) ──────────────────────────

def test_book_pct_aggregates_over_the_whole_book():
    payload = dict(
        positions=[_pos(key="SOXX", shares=100.0, avg_cost=200.0,
                        unrealized_pnl=1_000.0, realized_pnl=200.0),
                   _pos(key="ERX", shares=100.0, avg_cost=100.0,
                        unrealized_pnl=-500.0, realized_pnl=100.0)],
        trades=[_sell(key="SOXX", filled=20.0),
                _sell(key="ERX", filled=10.0, avg_fill_price=110.0)],
        realized_pnl=300.0)
    u, r = eb.book_pnl_pct(payload)
    assert u == pytest.approx(500.0 / 30_000.0 * 100)     # +1,000 −500 on 30,000
    assert r == pytest.approx(300.0 / 5_000.0 * 100)      # 20×200 + 10×100


def test_book_pct_prefers_the_account_realised_figure():
    """The account figure is the superset — it also sees a name closed out
    entirely, whose position row IBKR drops."""
    payload = dict(
        positions=[_pos(key="SOXX", shares=100.0, avg_cost=200.0,
                        unrealized_pnl=0.0, realized_pnl=200.0)],
        trades=[_sell(key="SOXX", filled=20.0),
                _sell(key="ERX", filled=100.0, avg_fill_price=50.0)],
        realized_pnl=700.0)
    _, r = eb.book_pnl_pct(payload)
    # closed-out ERX has no avg_cost left in the report, so its proceeds stand
    # in for its basis: 20×200 + 100×50 = 9,000
    assert r == pytest.approx(700.0 / 9_000.0 * 100)


def test_book_pct_is_none_when_nothing_was_recorded():
    payload = dict(positions=[_pos(realized_pnl=None)], trades=[])
    u, r = eb.book_pnl_pct(payload)
    assert u == pytest.approx(5.0) and r is None
    assert eb.book_pnl_pct({}) == (None, None)
