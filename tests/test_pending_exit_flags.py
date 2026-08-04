"""Pending ("exits next bar") exit consistency across engines and the gate.

Every engine decides at the close and executes on the NEXT bar, so an
in-position exit — a trend break OR a divergence/CT exit signal committed at
the latest close — is a *pending* exit: still open today, closed next bar.
These tests pin the two guarantees that keep the app surfaces consistent:

* every engine's in-position exit decision carries ``exits_next_bar=True``
  (the Overall action table's ⚠️ banner reads it, and it survives even when
  the Action pill is pinned to a morning book published before the signal
  committed — the GDX/NUGT case);
* ``signal_gated_allocation`` treats a committed pending exit exactly like a
  committed CLOSE — dropped from the recommended book (the REMX case: a
  death-cross committed at today's close previously stayed funded while its
  row was flagged "exits next bar") — and ``apply_closed_market_freeze``
  applies it as a committed signal change on closed days too.
"""
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import overall_core as oc  # noqa: E402
import gldm_engine as ge  # noqa: E402
import btc_ct_engine as be  # noqa: E402


# ── engine decision flags ────────────────────────────────────────────────
def _div_cfg(use_d1_exit=False):
    return SimpleNamespace(is_trend=False, use_d1_exit=use_d1_exit)


def _trend_cfg():
    return SimpleNamespace(is_trend=True)


def test_net_decision_trend_pending_exit_sets_flag():
    dec = oc._net_decision(_trend_cfg(), None, in_pos=True,
                           last_close=90.0, ma_val=100.0, long_now=False)
    # unified pending-exit convention: EXIT NEXT BAR — <cause>, red, tone exit
    assert dec["state"] == "EXIT" and dec["tone"] == "exit"
    assert dec["label"] == "EXIT NEXT BAR — BELOW TREND"
    assert dec.get("exits_next_bar") is True


def test_pending_exit_labels_share_one_convention():
    sigs = dict(d1_triggered=False, d2_triggered=True, d3_triggered=False,
                entry_triggered=False, u1_triggered=False)
    div = oc._net_decision(_div_cfg(), sigs, in_pos=True,
                           last_close=90.0, ma_val=None)
    gld = ge._decision(sigs, in_pos=True)
    gldt = ge._trend_decision(long_now=False, in_pos=True)
    btc = be._decision(dict(d2=[True], d3=[False], bull_regime=[False],
                            u1=[False], tf2_entry=[False]), 0, in_pos=True)
    for dec in (div, gld, gldt, btc):
        assert dec["label"].startswith("EXIT NEXT BAR — ")
        assert dec["tone"] == "exit" and dec.get("exits_next_bar") is True


def test_net_decision_trend_holding_above_has_no_flag():
    dec = oc._net_decision(_trend_cfg(), None, in_pos=True,
                           last_close=110.0, ma_val=100.0, long_now=True)
    assert not dec.get("exits_next_bar")


def test_net_decision_divergence_in_pos_exit_sets_flag():
    sigs = dict(d1_triggered=False, d2_triggered=True, d3_triggered=False,
                entry_triggered=False, u1_triggered=False)
    dec = oc._net_decision(_div_cfg(), sigs, in_pos=True,
                           last_close=90.0, ma_val=None)
    assert dec["state"] == "EXIT" and dec["tone"] == "exit"
    assert dec.get("exits_next_bar") is True


def test_net_decision_divergence_flat_exit_active_has_no_flag():
    # flat + active exit = STAND ASIDE — nothing is open, nothing exits
    sigs = dict(d1_triggered=False, d2_triggered=True, d3_triggered=False,
                entry_triggered=False, u1_triggered=False)
    dec = oc._net_decision(_div_cfg(), sigs, in_pos=False,
                           last_close=90.0, ma_val=None)
    assert dec["state"] == "AVOID" and not dec.get("exits_next_bar")


def test_gldm_engine_decisions_set_flag():
    sigs = dict(d1_triggered=False, d2_triggered=True, d3_triggered=False,
                entry_triggered=False, u1_triggered=False)
    div = ge._decision(sigs, in_pos=True)
    assert div["tone"] == "exit" and div.get("exits_next_bar") is True
    trend = ge._trend_decision(long_now=False, in_pos=True)
    assert trend["tone"] == "exit" and trend.get("exits_next_bar") is True
    assert not ge._decision(sigs, in_pos=False).get("exits_next_bar")
    assert not ge._trend_decision(long_now=True, in_pos=True).get("exits_next_bar")


def test_btc_engine_decision_sets_flag():
    sigs = dict(d2=[True], d3=[False], bull_regime=[False],
                u1=[False], tf2_entry=[False])
    dec = be._decision(sigs, 0, in_pos=True)
    assert dec["tone"] == "exit" and dec.get("exits_next_bar") is True
    assert not be._decision(sigs, 0, in_pos=False).get("exits_next_bar")


# ── gate: committed pending exits leave the recommended book ─────────────
def _res(key, in_pos=True, tone="hold", exits_next_bar=False, kind="core"):
    # real engines emit state EXIT whenever tone is "exit" (divergence/CT)
    dec = dict(state=("EXIT" if tone == "exit" else "HOLD"),
               label="LONG — HOLDING", ico="🟢", tone=tone)
    if exits_next_bar:
        dec["exits_next_bar"] = True
    return dict(key=key, parent=key, name=key, kind=kind, emoji="🧲",
                kemoji="", accent="#000", cap=0.30,
                last_close=100.0, dchg=0.0, ma_val=None,
                sentiment=50.0, decision=dec, alert="NEUTRAL",
                bull_regime=True, mom=0.05, win_rate=60.0,
                metrics=dict(sharpe=1.0),
                pos=dict(in_pos=in_pos, upnl=1.0))


def test_gate_drops_committed_pending_exit_from_target():
    held = _res("SOXX")
    pending = _res("REMX", exits_next_bar=True)          # trend death cross
    closing = _res("GDX", tone="exit")                   # divergence D2 exit
    gate = oc.signal_gated_allocation([held, pending, closing],
                                      {"SOXX": 0.3, "REMX": 0.3, "GDX": 0.3})
    assert gate["target"].get("REMX", 0.0) == 0.0
    assert gate["target"].get("GDX", 0.0) == 0.0
    assert gate["target"]["SOXX"] > 0
    assert gate["n_close"] == 2
    # the action row is an INSTRUCTION: a committed pending exit reads CLOSE
    # (the book zero-weights it and the executor sells it this session), and
    # carries the flag the table's ⚠️ banner renders
    by = {a["key"]: a for a in gate["actions"]}
    assert by["REMX"]["action"] == "CLOSE" and by["REMX"]["exits_next_bar"]
    assert by["GDX"]["action"] == "CLOSE"
    assert by["SOXX"]["action"] == "HOLD" and not by["SOXX"]["exits_next_bar"]


def test_freeze_drops_pending_exit_like_a_close():
    # Sat 2026-07-04 — a pending exit committed at Friday's close is a
    # committed signal change; the freeze must not pin its published weight
    gate = dict(target={"SOXX": 0.18}, sata=0.82,
                actions=[dict(key="REMX", action="HOLD", exits_next_bar=True),
                         dict(key="SOXX", action="HOLD")],
                current={}, sata_now=0.0, priorities={}, priority_rank=[],
                n_active=0, n_open=0, n_close=0, sata_info=oc.SATA)
    out = oc.apply_closed_market_freeze(gate, {"REMX": 0.30, "SOXX": 0.18},
                                        today="2026-07-04")
    assert "REMX" not in out["target"]
    assert out["target"] == {"SOXX": 0.18}
    assert out["freeze"]["closed"] == ["REMX"]
    assert np.isclose(out["sata"], 0.82)
