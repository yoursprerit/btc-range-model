"""The manual-trade refresh helper: refresh-only argv + reconciliation table."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "app"))

import ibkr_execute_accounts as acc  # noqa: E402
import ibkr_manual_refresh as mr  # noqa: E402


def _cfg():
    return acc.Config(accounts=[acc.Account(id="U1234567", label="t")])


def test_child_argv_never_trades_and_refreshes():
    cfg = _cfg()
    argv = mr.child_argv(cfg, cfg.accounts[0], 2, "book.json")
    assert "--execute" not in argv
    assert "--refresh-report" in argv and "--no-push-report" in argv
    assert argv[argv.index("--account") + 1] == "U1234567"
    assert argv[argv.index("--expected-account") + 1] == "U1234567"
    assert argv[argv.index("--client-id") + 1] == str(cfg.client_id_base + mr.CLIENT_ID_OFFSET + 2)


def test_reconcile_flags_missing_and_unexpected():
    report = {"net_liq": 10000.0, "positions": [
        {"key": "SATA", "symbol": "SATA", "shares": 30, "market_value": 3000.0},
        {"key": "GRID", "symbol": "GRID", "shares": 5, "market_value": 800.0}]}
    lines, problems = mr.reconcile_lines(report, {"SATA": 0.3, "PBW": 0.3})
    assert any("PBW" in p and "holds none" in p for p in problems)
    assert any("GRID" in p and "no target" in p for p in problems)
    assert not any("SATA" in p for p in problems)
    assert len(lines) == 3


def test_reconcile_clean_when_aligned():
    report = {"net_liq": 10000.0, "positions": [
        {"key": "PBW", "symbol": "PBW", "shares": 100, "market_value": 3000.0}]}
    _, problems = mr.reconcile_lines(report, {"PBW": 0.3})
    assert problems == []


def test_manual_fills_detected():
    r = {"trades": [{"reason": "fill with no planned order (reconstructed from broker fills)"},
                    {"reason": "-> 30%"}]}
    assert len(mr.manual_fills(r)) == 1
