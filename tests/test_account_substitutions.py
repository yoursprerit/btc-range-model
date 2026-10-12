"""IRA accounts cannot buy UGL: its weight is traded as GLDM instead."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "app"))

import ibkr_common as ic  # noqa: E402
import ibkr_execute_accounts as acc  # noqa: E402
import ibkr_symbols as sym  # noqa: E402


def test_weight_moves_to_substitute():
    w, moves = ic.apply_substitutions({"UGL": 0.2, "XLE": 0.1}, {"UGL": "GLDM"})
    assert w == {"GLDM": 0.2, "XLE": 0.1}
    assert moves == [("UGL", "GLDM", 0.2)]


def test_adds_to_existing_substitute_and_keeps_total():
    book = {"UGL": 0.15, "GLDM": 0.25, "SATA": 0.3}
    w, _ = ic.apply_substitutions(book, {"UGL": "GLDM"})
    assert w["GLDM"] == pytest.approx(0.40) and "UGL" not in w
    assert sum(w.values()) == pytest.approx(sum(book.values()))


def test_noop_without_restricted_name_and_input_untouched():
    book = {"GLDM": 0.3}
    w, moves = ic.apply_substitutions(book, {"UGL": "GLDM"})
    assert w == book and moves == [] and book == {"GLDM": 0.3}


def test_plan_buys_gldm_not_ugl():
    w, _ = ic.apply_substitutions({"UGL": 0.2}, {"UGL": "GLDM"})
    orders = ic.build_order_plan(w, {"GLDM": 100.0}, 100_000.0, {}, 0.01, False)
    assert [(o.symbol, o.action, o.qty) for o in orders] == [("GLDM", "BUY", 200.0)]


def test_parse_substitutions_rejects_bad_input():
    assert ic.parse_substitutions(["ugl=gldm"]) == {"UGL": "GLDM"}
    for bad in (["UGL"], ["UGL=UGL"], ["UGL=NOPE"], ["UGL=GLDM", "GLDM=GDX"]):
        with pytest.raises(ValueError):
            ic.parse_substitutions(bad)


def _cfg(**acct):
    raw = {"accounts": [dict(id="U2222222", label="x", **acct)]}
    cfg, errs = acc.validate_config(raw)
    assert not errs, errs
    return cfg, cfg.accounts[0]


def test_ira_account_defaults_to_ugl_gldm_and_passes_flag():
    cfg, a = _cfg(type="ira")
    assert a.substitutions == {"UGL": "GLDM"}
    argv = acc.child_argv(cfg, a, 0, execute=False)
    assert argv[argv.index("--substitute") + 1] == "UGL=GLDM"


def test_margin_account_has_no_substitution_and_override_works():
    cfg, a = _cfg(type="margin")
    assert a.substitutions == {} and "--substitute" not in acc.child_argv(cfg, a, 0, execute=False)
    _, a = _cfg(type="ira", substitutions={})
    assert a.substitutions == {}


def test_bad_substitution_in_config_is_reported():
    cfg, errs = acc.validate_config(
        {"accounts": [dict(id="U2222222", type="ira", substitutions={"UGL": "NOPE"})]})
    assert cfg is None and any("substitutions" in e for e in errs)


def test_symbols_default_map():
    assert sym.substitutions_for("IRA") == {"UGL": "GLDM"}
    assert sym.substitutions_for("margin") == {}
