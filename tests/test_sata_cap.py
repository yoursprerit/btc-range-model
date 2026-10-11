"""Strategy Logic V2 — the SATA cap (both profiles).

Idle capital (whatever the signals leave undeployed) used to sit entirely in
SATA, a preferred whose price can fall — up to 84 % of the book in the replay.
Under V2 it is parked in SATA only up to ``SATA_CAP`` (30 %) of the book; the
excess stays in CASH earning 0 %.  SATA is the allocator's residual, so no sleeve
weight changes — only where the idle remainder sits, and so which part earns the
coupon.  V1 has no cap; the Combined view applies it from the cut-over.

These tests pin each surface on synthetic data (no live feeds): the helper, the
live gate, the weekend freeze, the historical book, the replay (+ Combined
splice), the as-published replay, the frictions simulator, the health tracker,
the book payload schema and the call-site guard.
"""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "app"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import health_core as hc            # noqa: E402
import overall_core as oc           # noqa: E402
import overall_frictions as fr      # noqa: E402
import target_book as tb            # noqa: E402
from test_strategy_logic_v2 import _res, _universe   # noqa: E402

CAP = oc.SATA_CAP


# ── the helper and the version gate ───────────────────────────────────────
def test_cap_config_and_version_gate():
    assert CAP == pytest.approx(0.30)
    assert oc.sata_cap_for() == pytest.approx(CAP)
    assert oc.sata_cap_for("v2") == pytest.approx(CAP)
    assert oc.sata_cap_for("combined") == pytest.approx(CAP)    # acts from the cut-over
    assert oc.sata_cap_for("v1") is None                        # V1: idle is all SATA


def test_split_idle():
    assert oc.split_idle(0.1, CAP) == (pytest.approx(0.1), 0.0)           # under the cap
    s, c = oc.split_idle(0.84, CAP)
    assert s == pytest.approx(0.30) and c == pytest.approx(0.54)          # over the cap
    assert oc.split_idle(1.0, CAP) == (pytest.approx(0.30), pytest.approx(0.70))
    assert oc.split_idle(0.84, None) == (pytest.approx(0.84), 0.0)        # V1
    assert oc.split_idle(-0.01, CAP) == (0.0, 0.0)                        # no negative idle
    assert oc.split_idle(0.5, 0.0) == (0.0, pytest.approx(0.5))           # cap 0 → all cash


# ── the live gate ──────────────────────────────────────────────────────────
_W = {"AAA": 0.3, "BBB": 0.3, "CCC": 0.3}


def _gate(**kw):
    held = _res("AAA", in_pos=True, tone="hold")          # one core sleeve: cap 0.30 → 70 % idle
    return oc.signal_gated_allocation([held, _res("BBB"), _res("CCC")], _W,
                                      caps={"AAA": 0.30, "BBB": 0.30, "CCC": 0.30},
                                      adds_only=0, **kw)


def test_gate_parks_idle_in_sata_up_to_the_cap_and_the_rest_in_cash():
    g = _gate(sata_cap=CAP)
    assert g["idle"] == pytest.approx(0.70)
    assert g["sata"] == pytest.approx(0.30) and g["cash"] == pytest.approx(0.40)
    assert sum(g["target"].values()) + g["sata"] + g["cash"] == pytest.approx(1.0)
    assert g["sata_cap"] == CAP
    # the book in force ("what we hold now") splits the same way
    assert g["sata_now"] + g["cash_now"] == pytest.approx(g["idle_now"])
    assert g["sata_now"] <= CAP + 1e-12
    # the cap moves no sleeve weight
    plain = _gate()
    assert g["target"] == plain["target"]
    assert plain["sata"] == pytest.approx(0.70) and plain["cash"] == 0.0 and plain["sata_cap"] is None


def test_gate_under_the_cap_keeps_everything_in_sata():
    two = [_res("AAA", in_pos=True, tone="hold"), _res("BBB", in_pos=True, tone="hold"),
           _res("CCC", in_pos=True, tone="hold")]
    g = oc.signal_gated_allocation(two, _W, caps={"AAA": 0.30, "BBB": 0.30, "CCC": 0.30},
                                   adds_only=0, sata_cap=CAP)
    assert g["idle"] == pytest.approx(0.10)
    assert g["sata"] == pytest.approx(0.10) and g["cash"] == 0.0


def test_gate_ignores_the_cap_for_v1_results():
    held = _res("AAA", in_pos=True, tone="hold", version="v1")
    g = oc.signal_gated_allocation([held, _res("BBB", version="v1")], _W,
                                   caps={"AAA": 0.30, "BBB": 0.30}, adds_only=0, sata_cap=CAP)
    assert g["sata_cap"] is None and g["cash"] == 0.0
    assert g["sata"] == pytest.approx(g["idle"]) and g["idle"] > CAP


def test_weekend_freeze_resplits_the_pinned_book_by_the_cap():
    g = _gate(sata_cap=CAP)
    frozen = oc.apply_closed_market_freeze(g, {"AAA": 0.10, "BBB": 0.10},
                                           today=pd.Timestamp("2026-10-10"))   # a Saturday
    assert sum(frozen["target"].values()) == pytest.approx(0.20)
    assert frozen["idle"] == pytest.approx(0.80)
    assert frozen["sata"] == pytest.approx(0.30) and frozen["cash"] == pytest.approx(0.50)
    assert sum(frozen["target"].values()) + frozen["sata"] + frozen["cash"] == pytest.approx(1.0)


def test_historical_allocation_takes_the_cap():
    snap = dict(rows=[dict(key="AAA", parent="AAA", in_pos=True),
                      dict(key="BBB", parent="BBB", in_pos=False)])
    caps = {"AAA": 0.30, "BBB": 0.30}
    capped = oc.historical_allocation(snap, _W, caps=caps, sata_cap=CAP)
    assert capped["idle"] == pytest.approx(0.70)
    assert capped["sata"] == pytest.approx(0.30) and capped["cash"] == pytest.approx(0.40)
    plain = oc.historical_allocation(snap, _W, caps=caps)
    assert plain["sata"] == pytest.approx(0.70) and plain["cash"] == 0.0


# ── the replay ─────────────────────────────────────────────────────────────
def _replay(**kw):
    res = _universe()
    bw = {k: 0.12 for k in ("AAA", "BBB", "CCC", "DDD")}
    caps = {k: 0.12 for k in bw}                                # book ~48 % deployed → ~52 % idle
    return oc.replay_gated_allocation(res, base_weights=bw, caps=caps, adds_only=0,
                                      cluster_cap=0, sata_daily=0.001, **kw)


def test_replay_caps_sata_and_prices_the_cash_at_zero():
    plain, capped = _replay(), _replay(sata_cap=CAP)
    assert capped["weights"].equals(plain["weights"])           # no sleeve moves
    assert (capped["sata"] <= CAP + 1e-12).all()
    assert (capped["weights"].sum(axis=1) + capped["sata"] + capped["cash"]).round(9).eq(1.0).all()
    assert plain["cash"].abs().max() == 0.0
    assert capped["cash"].max() > 0.2 and capped["sata_cap"] == CAP
    # the only return difference is the coupon on the cash leg (weekdays only)
    biz = (plain["ret"].index.dayofweek < 5).astype(float)
    expected = plain["ret"] - capped["cash"] * 0.001 * biz
    pd.testing.assert_series_equal(capped["ret"], expected, check_names=False, atol=1e-12)
    # cash is not traded: turnover counts only the SATA leg, so it cannot rise
    assert capped["turnover"]["mean"] <= plain["turnover"]["mean"] + 1e-12


def test_replay_fully_flat_day_holds_the_cap_in_sata_and_the_rest_in_cash():
    res = _universe()
    for r in res:                                               # flat for the first 30 bars
        r["pos_series"] = r["pos_series"].copy()
        r["pos_series"].iloc[:30] = 0.0
    bw = {k: 0.25 for k in ("AAA", "BBB", "CCC", "DDD")}
    rep = oc.replay_gated_allocation(res, base_weights=bw, adds_only=0, cluster_cap=0,
                                     sata_cap=CAP, sata_daily=0.001)
    flat = rep["weights"].sum(axis=1) == 0
    assert flat.sum() >= 20
    assert np.allclose(rep["sata"][flat], CAP) and np.allclose(rep["cash"][flat], 1 - CAP)


def test_replay_cap_switches_on_from_a_date():
    cut = _universe()[0]["dates"][60]
    plain = _replay()
    mixed = _replay(sata_cap=CAP, sata_cap_from=cut)
    pre = slice(None, cut - pd.Timedelta(days=1))
    pd.testing.assert_series_equal(mixed["sata"].loc[pre], plain["sata"].loc[pre])
    assert mixed["cash"].loc[pre].abs().max() == 0.0
    assert mixed["cash"].loc[cut:].max() > 0.0
    assert (mixed["sata"].loc[cut:] <= CAP + 1e-12).all()


def test_walkforward_replay_applies_the_cap_for_v2_ignores_it_for_v1():
    res = _universe(n_days=300)
    kw = dict(n_samples=200, min_hist=40)
    v1 = oc.walkforward_gated_replay(res, version="v1", sata_cap=CAP, **kw)
    v1_plain = oc.walkforward_gated_replay(res, version="v1", **kw)
    pd.testing.assert_series_equal(v1["ret"], v1_plain["ret"])
    assert v1["cash"].abs().max() == 0.0 and v1["sata"].max() > CAP
    v2 = oc.walkforward_gated_replay(res, version="v2", sata_cap=CAP, **kw)
    assert (v2["sata"] <= CAP + 1e-12).all() and v2["cash"].max() > 0.0
    assert v2["sata_cap"] == CAP
    assert v2["ret"].sum() < oc.walkforward_gated_replay(res, version="v2", **kw)["ret"].sum()


def test_combined_replay_splices_cash_applying_the_cap_from_the_cutover():
    import strategy_version as sv
    from test_strategy_logic_v2 import _sleeve
    cut = pd.Timestamp(sv.STRATEGY_VERSION_START)
    idx = pd.bdate_range(cut - pd.Timedelta(days=200), periods=300)

    def gen(version, seed):
        rng = np.random.default_rng(seed)
        out = []
        for k in ("AAA", "BBB", "CCC", "DDD"):
            r = _sleeve(k, k, idx, rng)
            r["version"] = version
            r["pos_series"] = pd.Series((np.arange(len(idx)) % 7 < 2).astype(float), index=idx)
            out.append(r)
        return out

    v1, v2 = gen("v1", 5), gen("v2", 6)
    comb = oc.combine_results(v1, v2)
    rep = oc.walkforward_gated_replay(comb, version="combined", gens=(v1, v2), sata_cap=CAP,
                                      n_samples=200, min_hist=40)
    assert rep["cash"].loc[:cut - pd.Timedelta(days=1)].abs().max() == 0.0      # V1 days: all SATA
    assert (rep["sata"].loc[:cut - pd.Timedelta(days=1)] > CAP).any()
    assert (rep["sata"].loc[cut:] <= CAP + 1e-12).all()
    assert rep["cash"].loc[cut:].max() > 0.0
    assert (rep["weights"].sum(axis=1) + rep["sata"] + rep["cash"]).round(9).eq(1.0).all()


# ── the as-published replay ────────────────────────────────────────────────
def _book(as_of, weights, cash, mode="live", **extra):
    return {"schema": "overall-target-book/v1", "as_of": as_of, "profile": "Balanced",
            "book_mode": mode, "weights": weights, "cash_weight": cash,
            "exec_price": {}, **extra}


_IDX = pd.date_range("2026-07-06", periods=5, freq="D")                 # Mon..Fri


def test_published_book_replay_credits_the_coupon_only_up_to_the_stamped_cap():
    rets = pd.DataFrame({"AAA": 0.0}, index=_IDX)
    sd = 0.001
    legacy = oc.published_book_replay(rets, [_book("2026-07-06", {"AAA": 0.2}, 0.8, mode="paper")],
                                      sata_daily=sd)
    capped_paper = oc.published_book_replay(
        rets, [_book("2026-07-06", {"AAA": 0.2}, 0.8, mode="paper", sata_cap=0.30)], sata_daily=sd)
    capped_live = oc.published_book_replay(
        rets, [_book("2026-07-06", {"AAA": 0.2, "SATA": 0.30}, 0.50, sata_cap=0.30)], sata_daily=sd)
    assert np.allclose(legacy["ret"].iloc[1:], 0.8 * sd)                  # no stamp → all idle is SATA
    for rep in (capped_paper, capped_live):
        assert np.allclose(rep["sata"].iloc[1:], 0.30) and np.allclose(rep["cash"].iloc[1:], 0.50)
        assert np.allclose(rep["ret"].iloc[1:], 0.30 * sd)                # coupon on the SATA leg only
    assert legacy["cash"].abs().max() == 0.0


def test_published_book_idle_under_the_cap_is_all_sata():
    rets = pd.DataFrame({"AAA": 0.0}, index=_IDX)
    rep = oc.published_book_replay(rets, [_book("2026-07-06", {"AAA": 0.9}, 0.1, mode="paper",
                                                sata_cap=0.30)], sata_daily=0.001)
    assert np.allclose(rep["sata"].iloc[1:], 0.1) and rep["cash"].abs().max() == 0.0


# ── the dollar re-play with costs / taxes ─────────────────────────────────
def _fx_book(n=400, seed=2):
    idx = pd.bdate_range("2023-01-02", periods=n)
    rng = np.random.default_rng(seed)
    rets = pd.DataFrame({"A": rng.normal(0.0008, 0.01, n)}, index=idx)
    on = (np.arange(n) // 40) % 2 == 0
    w = pd.DataFrame({"A": np.where(on, 0.3, 0.0)}, index=idx)
    idle = 1.0 - w.sum(axis=1)
    sata = idle.clip(upper=CAP)
    return rets, w, sata, idle - sata, pd.DataFrame({"A": 100.0}, index=idx)


def test_frictions_hold_the_over_cap_cash_at_zero_yield():
    rets, w, sata, cash, px = _fx_book()
    sd = 0.0005
    legacy = fr.simulate_frictions(rets, w, 1.0 - w.sum(axis=1), px, rets.index[5], sata_daily=sd)
    capped = fr.simulate_frictions(rets, w, sata, px, rets.index[5], sata_daily=sd, cash_w=cash)
    # frictions off: the dollar re-play reproduces the capped replay it is fed (net == gross)
    np.testing.assert_allclose(capped["net"].to_numpy(), capped["gross"].to_numpy(), rtol=1e-6)
    assert capped["net"].iloc[-1] < legacy["net"].iloc[-1]          # the cash earns no coupon
    # a zero cash leg reproduces the V1 plug exactly
    zero = fr.simulate_frictions(rets, w, 1.0 - w.sum(axis=1), px, rets.index[5], sata_daily=sd,
                                 cash_w=pd.Series(0.0, index=rets.index))
    np.testing.assert_allclose(zero["net"].to_numpy(), legacy["net"].to_numpy(), rtol=1e-9)
    t = fr.simulate_frictions(rets, w, sata, px, rets.index[5], sata_daily=sd, cash_w=cash,
                              costs=True, taxes=True)
    assert t["net"].iloc[-1] < capped["net"].iloc[-1]               # costs/taxes still bite
    assert np.isfinite(t["net"].to_numpy()).all()


# ── the health tracker ─────────────────────────────────────────────────────
def test_realized_book_returns_credits_the_coupon_up_to_the_books_cap():
    idx = pd.date_range("2026-07-13", periods=4, freq="D")                # Mon..Thu
    rets = pd.DataFrame({"AAA": [0.0] * 4}, index=idx)
    sd = 0.001
    books = [dict(as_of="2026-07-13", weights={"AAA": 0.2}, cash_weight=0.8, sata_cap=0.30)]
    live = hc.realized_book_returns(books, rets, sata_daily=sd)
    assert np.isclose(live.iloc[0], 0.30 * sd)
    legacy = hc.realized_book_returns([dict(as_of="2026-07-13", weights={"AAA": 0.2},
                                            cash_weight=0.8)], rets, sata_daily=sd)
    assert np.isclose(legacy.iloc[0], 0.8 * sd)                           # unstamped → unchanged
    live_book = hc.realized_book_returns(
        [dict(as_of="2026-07-13", weights={"AAA": 0.2, "SATA": 0.3}, cash_weight=0.5, sata_cap=0.30)],
        rets, sata_daily=sd)
    assert np.isclose(live_book.iloc[0], 0.30 * sd)                       # live form: same coupon


# ── the book payload ───────────────────────────────────────────────────────
def test_payload_stamps_the_cap_only_when_set():
    base = dict(as_of="2026-10-12", profile="Balanced", weights={"AAA": 0.2}, cash_weight=0.8,
                exec_price={"AAA": 10.0})
    assert "sata_cap" not in tb.build_payload(**base)                     # legacy shape unchanged
    p = tb.build_payload(sata_cap=0.30, **base)
    assert p["sata_cap"] == pytest.approx(0.30)
    # covered by the signature: tampering with the stamp is detected
    signed = tb.sign(p, "secret")
    assert tb.verify_signature(signed, "secret")[0]
    tampered = dict(signed, sata_cap=0.9)
    assert not tb.verify_signature(tampered, "secret")[0]


# ── every consumer must pass the cap ──────────────────────────────────────
_NEEDS_CAP = {"signal_gated_allocation", "walkforward_gated_replay"}


def test_every_allocator_call_site_passes_sata_cap():
    """A surface that forgot the cap would show or trade a different book from
    the one the replay validated."""
    bad = []
    for base in ("app", "scripts"):
        for p in sorted((REPO / base).glob("*.py")):
            if p.name == "overall_core.py":
                continue
            tree = ast.parse(p.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                f = node.func
                name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                if name in _NEEDS_CAP and "sata_cap" not in {k.arg for k in node.keywords}:
                    bad.append(f"{p.relative_to(REPO)}:{node.lineno} {name}")
    assert bad == []
