"""Live-price likely ENTRIES are funded in the recommended live book.

The Recommended Live Possible Targetbook must reflect the possible target
book **today's live prices point to** — on the way out AND on the way in:

* ``signal_gated_allocation(force_entry=…)`` funds a flat sleeve whose live
  price satisfies its entry condition exactly like a committed fresh open
  (priority-tilted, water-filled) and marks its action row
  ``live_entry=True`` so the UI can tell a live "likely" open from a
  committed one;
* a key in both ``force_entry`` and ``force_exit`` is treated as an exit
  (it would open into a broken trend and reverse straight back out);
* without ``force_entry`` nothing changes — the committed-only callers
  (the publisher, the last-bar gate) never pre-fund an uncommitted signal;
* ``apply_closed_market_freeze`` funds a ``live_entry`` row from idle
  capital like a committed OPEN — on a closed US day such a row can only
  come from an app whose signal asset still trades (BTC/ETH).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import overall_core as oc  # noqa: E402


def _res(key, in_pos=False, tone="flat", state=None, label="FLAT — NO SIGNAL"):
    dec = dict(state=(state or ("HOLD" if in_pos else "FLAT")),
               label=label, ico="🟢", tone=tone)
    return dict(key=key, parent=key, name=key, kind="core", emoji="🧲",
                kemoji="", accent="#000", cap=0.30,
                last_close=100.0, dchg=0.0, ma_val=None,
                sentiment=50.0, decision=dec, alert="NEUTRAL",
                bull_regime=True, mom=0.05, win_rate=60.0,
                metrics=dict(sharpe=1.0),
                pos=dict(in_pos=in_pos, upnl=1.0))


_W = {"SOXX": 0.3, "XLE": 0.3, "GDX": 0.3}


def test_force_entry_funds_flat_sleeve_like_an_open():
    held = _res("SOXX", in_pos=True, tone="hold", state="HOLD",
                label="LONG — HOLDING")
    flat = _res("XLE")                       # flat, no committed buy
    gate = oc.signal_gated_allocation([held, flat], _W,
                                      force_entry={"XLE"})
    assert gate["target"]["XLE"] > 0
    assert gate["n_open"] == 1
    by = {a["key"]: a for a in gate["actions"]}
    # the committed action/state is untouched — only the funding + flag change
    assert by["XLE"]["live_entry"] is True
    assert by["XLE"]["action"] == "STAND ASIDE"
    assert by["XLE"]["target"] == gate["target"]["XLE"]
    assert by["SOXX"]["live_entry"] is False


def test_force_exit_beats_force_entry():
    flat = _res("XLE")
    gate = oc.signal_gated_allocation([flat], _W,
                                      force_exit={"XLE"}, force_entry={"XLE"})
    assert gate["target"].get("XLE", 0.0) == 0.0
    assert gate["n_open"] == 0


def test_committed_buy_is_not_flagged_live_entry():
    buy = _res("GDX", tone="buy", state="ENTRY", label="BUY SIGNAL")
    gate = oc.signal_gated_allocation([buy], _W, force_entry={"GDX"})
    by = {a["key"]: a for a in gate["actions"]}
    assert by["GDX"]["action"] == "OPEN" and by["GDX"]["live_entry"] is False
    assert gate["target"]["GDX"] > 0


def test_default_call_never_prefunds_uncommitted_signal():
    # regression guard for the committed-only callers (publisher / last-bar
    # gate): with no force_entry a flat sleeve stays unfunded and unflagged
    flat = _res("XLE")
    gate = oc.signal_gated_allocation([flat], _W)
    assert gate["target"].get("XLE", 0.0) == 0.0
    by = {a["key"]: a for a in gate["actions"]}
    assert by["XLE"]["live_entry"] is False


def test_freeze_funds_live_entry_from_idle():
    # Sat 2026-07-04 — a live-price likely entry (necessarily from an app
    # whose asset trades on a closed US day) is funded from idle capital at
    # the gate's tilted size, like a committed OPEN
    gate = dict(target={"SOXX": 0.18, "BTC": 0.25}, sata=0.57,
                actions=[dict(key="SOXX", action="HOLD"),
                         dict(key="BTC", action="STAND ASIDE",
                              live_entry=True)],
                current={}, sata_now=0.0, priorities={}, priority_rank=[],
                n_active=1, n_open=1, n_close=0, sata_info=oc.SATA)
    out = oc.apply_closed_market_freeze(gate, {"SOXX": 0.18},
                                        today="2026-07-04")
    assert out["target"]["SOXX"] == 0.18
    assert np.isclose(out["target"]["BTC"], 0.25)
    assert out["freeze"]["opened"] == ["BTC"]
    assert np.isclose(out["sata"], 1.0 - 0.18 - 0.25)


def test_freeze_still_ignores_plain_stand_aside():
    gate = dict(target={"SOXX": 0.18}, sata=0.82,
                actions=[dict(key="SOXX", action="HOLD"),
                         dict(key="XLE", action="STAND ASIDE")],
                current={}, sata_now=0.0, priorities={}, priority_rank=[],
                n_active=1, n_open=0, n_close=0, sata_info=oc.SATA)
    out = oc.apply_closed_market_freeze(gate, {"SOXX": 0.18},
                                        today="2026-07-04")
    assert "XLE" not in out["target"]
    assert out["freeze"]["opened"] == []
