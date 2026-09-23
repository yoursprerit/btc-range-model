"""Collective2 publisher: sizing and the guards around SetDesiredPositions.

SetDesiredPositions closes every C2 position missing from the request, so the
tests that matter most pin the cases where a bad book or tiny capital would
turn into an accidental liquidation of the public strategy.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import publish_c2 as pc  # noqa: E402


# ── sizing ──────────────────────────────────────────────────────────────────
def test_shares_are_floored_after_the_cash_buffer():
    pos, dropped = pc.build_positions({"GLDM": 0.5}, {"GLDM": 86.33}, 100_000, 0.01)
    # 0.5 × 99,000 / 86.33 = 573.37 → 573
    assert pos == [{"symbol": "GLDM", "key": "GLDM", "quantity": 573,
                    "price": 86.33, "weight": 0.5}]
    assert dropped == []


def test_a_fully_invested_book_never_sizes_past_capital():
    weights = {"GLDM": 0.3, "UGL": 0.3, "ARTY": 0.4}
    prices = {"GLDM": 86.33, "UGL": 50.58, "ARTY": 79.81}
    pos, _ = pc.build_positions(weights, prices, 50_000)
    assert sum(p["quantity"] * p["price"] for p in pos) <= 50_000


def test_btc_sleeve_trades_through_ibit():
    pos, _ = pc.build_positions({"BTC": 0.2}, {"BTC": 60.0}, 10_000)
    assert pos[0]["symbol"] == "IBIT" and pos[0]["key"] == "BTC"


def test_unsizeable_names_are_reported_not_guessed():
    pos, dropped = pc.build_positions(
        {"GLDM": 0.001, "UGL": 0.2, "NOPE": 0.1}, {"GLDM": 86.33}, 10_000)
    assert pos == []
    assert any("GLDM" in d and "one share" in d for d in dropped)
    assert any("UGL" in d and "no price" in d for d in dropped)
    assert any("NOPE" in d and "no tradeable symbol" in d for d in dropped)


# ── liquidation guards ──────────────────────────────────────────────────────
def test_book_with_positions_that_sizes_to_nothing_is_refused():
    ok, why = pc.check_positions({"OIH": 0.05}, [], allow_flat=True)
    assert not ok and "liquidate" in why


def test_all_cash_book_needs_explicit_allow_flat():
    assert not pc.check_positions({}, [], allow_flat=False)[0]
    assert pc.check_positions({}, [], allow_flat=True)[0]


def test_overweight_book_is_refused():
    pos = [{"symbol": "GLDM", "quantity": 1}]
    ok, why = pc.check_positions({"GLDM": 0.7, "UGL": 0.5}, pos, allow_flat=False)
    assert not ok and "100%" in why


# ── request / response ──────────────────────────────────────────────────────
def test_payload_matches_the_v2_schema():
    body = pc.desired_positions_payload(123, [{"symbol": "XLE", "quantity": 10}])
    assert body == {"StrategyId": 123, "Positions": [{
        "StrategyId": 123, "Quantity": 10,
        "ExchangeSymbol": {"Symbol": "XLE", "Currency": "USD",
                           "SecurityExchange": "DEFAULT", "SecurityType": "CS"}}]}


def test_rejected_signal_fails_the_publish():
    ok, lines = pc.summarize_response({"Results": [{
        "NewSignals": [{"Side": "1", "OrderQuantity": 5,
                        "ExchangeSymbol": {"Symbol": "XLE"}}],
        "RejectedSignals": [{"Side": "1", "OrderQuantity": 3,
                             "ExchangeSymbol": {"Symbol": "ARTY"},
                             "RejectMessage": "symbol not supported"}]}]})
    assert not ok
    assert any("ARTY" in l and "not supported" in l for l in lines)


def test_error_status_fails_and_clean_response_passes():
    assert not pc.summarize_response(
        {"ResponseStatus": {"ErrorCode": "400", "Message": "bad"}})[0]
    ok, lines = pc.summarize_response(
        {"Results": [{"NewSignals": [], "RejectedSignals": []}],
         "ResponseStatus": {"ErrorCode": "200", "Message": "OK"}})
    assert ok and "already matches" in lines[0]


def test_fingerprint_changes_when_the_book_is_republished():
    a = {"as_of": "2026-09-22", "signature": {"value": "aa"}}
    b = {"as_of": "2026-09-22", "signature": {"value": "bb"}}
    assert pc.book_fingerprint(a) != pc.book_fingerprint(b)
    assert pc.book_fingerprint(a) == pc.book_fingerprint(dict(a))


# ── end to end (network stubbed) ────────────────────────────────────────────
def test_execute_sends_once_then_skips_the_same_book(tmp_path, monkeypatch):
    import json
    book = {"schema": "overall-target-book/v1", "as_of": None,
            "weights": {"XLE": 0.5}, "exec_price": {"XLE": 50.0}}
    import pandas as pd
    book["as_of"] = str(pd.Timestamp.now().normalize().date())
    f = tmp_path / "book.json"
    f.write_text(json.dumps(book))
    state = tmp_path / "state.json"

    sent = []
    monkeypatch.setattr(pc, "set_desired_positions",
                        lambda key, body: sent.append(body) or {"Results": []})
    monkeypatch.setattr(pc.ic, "is_trading_day", lambda d: (True, ""))
    monkeypatch.setattr(pc.ic, "market_session_open", lambda **k: (True, ""))
    monkeypatch.setenv("C2_API_KEY", "k")
    monkeypatch.delenv("OVERALL_BOOK_SECRET", raising=False)
    argv = ["--file", str(f), "--state", str(state), "--capital", "10000",
            "--strategy-id", "42", "--execute"]

    assert pc.main(argv) == 0
    assert sent[0]["Positions"][0]["Quantity"] == 99      # 0.5 × 9,900 / 50
    assert json.loads(state.read_text())["positions"] == {"XLE": 99}

    assert pc.main(argv) == 0                              # same book → skipped
    assert len(sent) == 1


def test_missing_credentials_abort_without_capital(tmp_path, monkeypatch, capsys):
    import json
    import pandas as pd
    f = tmp_path / "book.json"
    f.write_text(json.dumps({"schema": "overall-target-book/v1",
                             "as_of": str(pd.Timestamp.now().normalize().date()),
                             "weights": {"XLE": 0.5}, "exec_price": {"XLE": 50.0}}))
    monkeypatch.delenv("C2_API_KEY", raising=False)
    monkeypatch.delenv("C2_STRATEGY_ID", raising=False)
    monkeypatch.delenv("OVERALL_BOOK_SECRET", raising=False)
    assert pc.main(["--file", str(f), "--state", str(tmp_path / "s.json")]) == 2
    assert "C2_API_KEY" in capsys.readouterr().out
