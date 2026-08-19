"""The sleeve ledger — trading a compounding SLICE of the account.

The property that makes a sleeve a sleeve, and the one ``--max-deploy-frac``
does not have, is that its equity moves ONE-FOR-ONE with the strategy's own
P&L: a winning run is not diluted back across the untouched cash, and a losing
run is not topped up from it.  Everything else here is a guard against the
sleeve quietly stopping being a slice — a double-booked run, a ledger pointed
at the wrong account, or one that has drifted outside the account it claims to
live inside.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import sleeve as sl  # noqa: E402


def _open(usd=10_000.0, net_liq=100_000.0):
    return sl.open_sleeve(usd=usd, account="DU123", account_mode="paper",
                          account_net_liq=net_liq, as_of="2026-08-18")


def _fill(symbol, action, qty, price):
    return {"symbol": symbol, "action": action, "qty": qty, "price": price,
            "filled": qty, "avg_fill_price": price, "status": "Filled"}


# ── the compounding property ────────────────────────────────────────────────
def test_a_gain_stays_in_the_sleeve_and_is_not_diluted():
    """$10k of a $100k account buys 100 SPY at $100; SPY goes to $110.

    The sleeve is now worth $11,000 — the full gain — even though the account
    only moved 1%. A --max-deploy-frac rule would hand back a 10%-of-$101k =
    $10,100 stake instead, keeping a tenth of what the strategy earned."""
    led = _open()
    led, _ = sl.apply_fills(led, [_fill("SPY", "BUY", 100, 100.0)], "2026-08-18")
    assert led["cash"] == pytest.approx(0.0)

    nav, unpriced = sl.nav(led, {"SPY": 100.0}, {"SPY": 110.0})
    assert not unpriced
    assert nav == pytest.approx(11_000.0)
    perf = sl.performance(led, nav)
    assert perf["contributed"] == pytest.approx(10_000.0)
    assert perf["pnl"] == pytest.approx(1_000.0)
    assert perf["return_pct"] == pytest.approx(10.0)


def test_a_loss_is_not_topped_up_from_the_rest_of_the_account():
    led = _open()
    led, _ = sl.apply_fills(led, [_fill("SPY", "BUY", 100, 100.0)], "2026-08-18")
    nav, _ = sl.nav(led, {"SPY": 100.0}, {"SPY": 80.0})
    assert nav == pytest.approx(8_000.0)          # NOT restored to 10% of NAV
    assert sl.performance(led, nav)["return_pct"] == pytest.approx(-20.0)


def test_it_compounds_across_runs():
    """Sell the winner, re-deploy: the second position is sized off the LARGER
    balance, which is what compounding means."""
    led = _open()
    led, _ = sl.apply_fills(led, [_fill("SPY", "BUY", 100, 100.0)], "2026-08-18")
    led, _ = sl.apply_fills(led, [_fill("SPY", "SELL", 100, 110.0)], "2026-08-19")
    assert led["cash"] == pytest.approx(11_000.0)
    nav, _ = sl.nav(led, {}, {})
    assert nav == pytest.approx(11_000.0)


# ── money that is not P&L stays separable ───────────────────────────────────
def test_a_deposit_raises_capital_but_not_the_return():
    led = _open()
    led, _ = sl.adjust(led, 5_000.0, note="top-up")
    nav, _ = sl.nav(led, {}, {})
    assert nav == pytest.approx(15_000.0)
    perf = sl.performance(led, nav)
    assert perf["contributed"] == pytest.approx(15_000.0)
    assert perf["return_pct"] == pytest.approx(0.0)   # added money is not profit


def test_a_dividend_is_profit_not_capital():
    """The subtle one. A dividend arrives as cash exactly like a deposit does,
    but the sleeve EARNED it — booking it as capital raises the base and the
    strategy's measured return goes DOWN on a day it made money."""
    led = _open()
    led, why = sl.adjust(led, 250.0, kind="income", note="GLDM dividend")
    perf = sl.performance(led, sl.nav(led, {}, {})[0])
    assert "P&L" in why
    assert perf["contributed"] == pytest.approx(10_000.0)   # base unmoved
    assert perf["pnl"] == pytest.approx(250.0)
    assert perf["return_pct"] == pytest.approx(2.5)

    same_cash_as_capital = sl.adjust(_open(), 250.0)[0]
    assert sl.performance(same_cash_as_capital,
                          10_250.0)["return_pct"] == pytest.approx(0.0)


def test_a_fee_debit_is_a_loss():
    led, _ = sl.adjust(_open(), -30.0, kind="income", note="data fee")
    perf = sl.performance(led, sl.nav(led, {}, {})[0])
    assert perf["contributed"] == pytest.approx(10_000.0)
    assert perf["pnl"] == pytest.approx(-30.0)


def test_older_ledgers_still_value_the_same():
    """`adjust` was the single pre-split kind name and meant capital."""
    led = _open()
    led["entries"].append({"kind": "adjust", "usd": 1_000.0})
    assert sl.contributed(led) == pytest.approx(11_000.0)


def test_an_unknown_adjustment_kind_is_refused():
    with pytest.raises(ValueError):
        sl.adjust(_open(), 100.0, kind="dividend")


# ── fills drive the cash, not intentions ────────────────────────────────────
def test_a_partial_fill_moves_only_what_filled():
    led = _open()
    t = _fill("SPY", "BUY", 100, 100.0)
    t["filled"], t["avg_fill_price"] = 40.0, 101.0     # 40 filled, above the plan
    led, _ = sl.apply_fills(led, [t], "2026-08-18")
    assert led["cash"] == pytest.approx(10_000.0 - 40 * 101.0)


def test_planned_rows_from_a_dry_run_move_nothing():
    planned = [{"symbol": "SPY", "action": "BUY", "qty": 100, "price": 100.0,
                "status": "PLANNED", "filled": 0.0, "avg_fill_price": 0.0}]
    delta, warn = sl.fills_cash_delta(planned)
    assert delta == 0.0 and not warn


def test_commissions_are_a_cost_on_both_sides():
    buy = _fill("SPY", "BUY", 10, 100.0) | {"commission": 1.0}
    sell = _fill("SPY", "SELL", 10, 100.0) | {"commission": 1.0}
    assert sl.fills_cash_delta([buy])[0] == pytest.approx(-1001.0)
    assert sl.fills_cash_delta([sell])[0] == pytest.approx(999.0)


def test_the_same_signal_bar_cannot_be_booked_twice():
    """The executed-book archive already locks a duplicate RUN; this locks the
    duplicate MONEY, so a --force-rerun cannot debit the sleeve a second time."""
    led = _open()
    led, _ = sl.apply_fills(led, [_fill("SPY", "BUY", 100, 100.0)], "2026-08-18")
    again, why = sl.apply_fills(led, [_fill("SPY", "BUY", 100, 100.0)], "2026-08-18")
    assert again["cash"] == pytest.approx(0.0)
    assert "already advanced" in why


# ── it must keep being a subset of a real account ───────────────────────────
def test_a_ledger_bigger_than_the_account_is_refused():
    ok, why = sl.check_within_account(60_000.0, 10_000.0,
                                      account_net_liq=50_000.0, account_cash=10_000.0)
    assert not ok and "drifted" in why


def test_a_sleeve_that_cannot_fund_its_buys_is_refused():
    """Ledger says $9k of cash, the account holds $500 — the buy leg would
    borrow. Refused unless margin is explicitly allowed."""
    ok, _ = sl.check_within_account(10_000.0, 9_000.0,
                                    account_net_liq=100_000.0, account_cash=500.0)
    assert not ok
    ok, _ = sl.check_within_account(10_000.0, 9_000.0, account_net_liq=100_000.0,
                                    account_cash=500.0, allow_margin=True)
    assert ok


def test_a_healthy_slice_passes():
    ok, why = sl.check_within_account(10_000.0, 2_000.0,
                                      account_net_liq=100_000.0, account_cash=90_000.0)
    assert ok and "10.0%" in why


def test_a_paper_ledger_will_not_trade_the_live_account():
    led = _open()
    assert not sl.check_account_match(led, "U7654321", "live")[0]
    assert not sl.check_account_match(led, "DU123", "live")[0]
    assert sl.check_account_match(led, "DU123", "paper")[0]


def test_a_stake_bigger_than_the_account_cannot_be_opened():
    with pytest.raises(ValueError):
        sl.open_sleeve(usd=200_000.0, account="DU1", account_mode="paper",
                       account_net_liq=100_000.0)
    with pytest.raises(ValueError):
        sl.open_sleeve(usd=0.0, account="DU1", account_mode="paper",
                       account_net_liq=100_000.0)


# ── valuation ───────────────────────────────────────────────────────────────
def test_a_name_the_book_no_longer_prices_falls_back_to_the_brokers_mark():
    """A position being closed out carries no book price; valuing it at zero
    would understate the sleeve and over-buy everything else."""
    led = _open()
    nav, unpriced = sl.nav(led, {"SPY": 10.0, "OIH": 5.0}, {"SPY": 100.0},
                           {"OIH": 40.0})
    assert not unpriced
    assert nav == pytest.approx(10_000.0 + 1_000.0 + 200.0)


def test_an_unpriceable_holding_is_reported_not_silently_zeroed():
    led = _open()
    _, unpriced = sl.nav(led, {"SPY": 10.0}, {}, {})
    assert unpriced == ["SPY"]


def test_ledger_round_trips_through_disk(tmp_path):
    led = _open()
    led, _ = sl.apply_fills(led, [_fill("SPY", "BUY", 10, 100.0)], "2026-08-18")
    path = sl.save(led, tmp_path / "sleeve.json", "secret")
    back = sl.load(path)
    assert sl.validate(back)[0]
    assert back["cash"] == pytest.approx(9_000.0)
    import target_book as tb
    assert tb.verify_signature(back, "secret")[0]
    assert not tb.verify_signature(back, "wrong-secret")[0]


def test_default_paths_follow_the_account_mode(tmp_path):
    assert sl.default_path(tmp_path / "executed_book.json").name == "sleeve.json"
    assert sl.default_path(tmp_path / "executed_book_live.json",
                           "live").name == "sleeve_live.json"


# ── the sizing base the executor hands to the plan and the guards ───────────
import ibkr_common as ic  # noqa: E402


def test_the_plan_is_sized_off_the_sleeve_not_the_account():
    """The whole point: a $10k sleeve of a $100k account buys a tenth as much."""
    weights, price = {"SOXX": 0.5}, {"SOXX": 100.0}
    full = ic.build_order_plan(weights, price, 100_000.0, {}, 0.01, False)
    slice_ = ic.build_order_plan(weights, price, 10_000.0, {}, 0.01, False)
    assert full[0].qty == pytest.approx(500.0)
    assert slice_[0].qty == pytest.approx(50.0)


def test_the_no_trade_band_scales_with_the_sleeve():
    """A 1% band on the account is a 10% band on a 10% sleeve — it would
    suppress every rebalance the sleeve ever wants to make."""
    weights, price = {"SOXX": 0.5}, {"SOXX": 100.0}
    held = {"SOXX": 48.0}          # $200 away from the sleeve's $5,000 target
    assert ic.build_order_plan(weights, price, 10_000.0, held, 0.01, False), \
        "a 2% drift must trade against a 1%-of-sleeve band"
    assert not ic.build_order_plan(weights, price, 10_000.0, held, 0.10, False)


def test_the_exposure_cap_measures_leverage_against_the_sleeve():
    """Left on the account's net liq, a 10% sleeve could gear itself 10:1 before
    the 1.02x cap noticed anything."""
    orders = [ic.Order("SOXX", "SOXX", "BUY", 100.0, 100.0, "")]
    held = {"SOXX": 100.0}                      # already $10k of a $10k sleeve
    on_account = ic.projected_exposure(orders, held, {"SOXX": 100.0}, 100_000.0)
    on_sleeve = ic.projected_exposure(orders, held, {"SOXX": 100.0}, 10_000.0)
    assert ic.check_exposure(on_account, ic.DEFAULT_MAX_GROSS_FRAC)[0]
    assert not ic.check_exposure(on_sleeve, ic.DEFAULT_MAX_GROSS_FRAC)[0]


def test_turnover_is_measured_against_the_sleeve_too():
    orders = [ic.Order("SOXX", "SOXX", "BUY", 200.0, 100.0, "")]   # $20k traded
    assert ic.check_turnover(orders, 100_000.0, 1.5)[0]
    assert not ic.check_turnover(orders, 10_000.0, 1.5)[0]


# ── money you cannot take out ───────────────────────────────────────────────
def test_a_withdrawal_from_a_fully_invested_sleeve_is_refused():
    """The sleeve's money is in its positions. Booking the withdrawal anyway
    would leave a negative cash balance that the NEXT run sizes its book
    against — a fiction, compounded."""
    led = _open()
    led, _ = sl.apply_fills(led, [_fill("SPY", "BUY", 100, 100.0)], "2026-08-18")
    with pytest.raises(ValueError, match="invested, not idle"):
        sl.adjust(led, -1_000.0, min_cash=0.0)
    assert sl.adjust(led, -1_000.0)[0]["cash"] == pytest.approx(-1_000.0), \
        "unfloored, it books — which is why the executor always passes a floor"


def test_a_withdrawal_within_idle_cash_goes_through():
    led = _open()
    led, msg = sl.adjust(led, -1_000.0, min_cash=0.0)
    assert led["cash"] == pytest.approx(9_000.0)
    assert sl.performance(led, 9_000.0)["contributed"] == pytest.approx(9_000.0)


def test_a_sleeve_on_margin_is_refused():
    ok, why = sl.check_within_account(10_000.0, -1_000.0,
                                      account_net_liq=100_000.0, account_cash=50_000.0)
    assert not ok and "carrying a loan" in why
    assert sl.check_within_account(10_000.0, -1_000.0, account_net_liq=100_000.0,
                                   account_cash=50_000.0, allow_margin=True)[0]


def test_a_few_dollars_of_fees_are_not_a_loan():
    """A fee debit against a fully-invested sleeve legitimately dips cash below
    zero; only a material balance is a loan."""
    assert sl.check_within_account(10_000.0, -12.50, account_net_liq=100_000.0,
                                   account_cash=50_000.0)[0]
