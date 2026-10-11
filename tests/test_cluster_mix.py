"""Strategy Logic V2 — fixed within-cluster split for the crypto cluster (Balanced).

The BTC parent cluster (BTC · MSTR · MSTU · ETH) trades one signal, so how its
weight is divided is purely a vehicle mix.  Under V2 the Balanced profile
divides the cluster's weight by a fixed ordered split — MSTR 45 % · MSTU 30 % ·
BTC 20 % · ETH 5 % — instead of the optimiser/tilt split.  The cluster's TOTAL
is still sized by the tilt and bound by the 30 % parent-cluster cap; the split
is applied to the tilt target before the adds-only pin (held names are never
re-split).  V1, and the Growth / Aggressive profiles, keep the tilt split.

These tests pin each part on synthetic data (no live feeds).
"""
import ast
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "app"))

import overall_core as oc           # noqa: E402
from test_strategy_logic_v2 import _res, _universe   # noqa: E402

_MIX = oc.CLUSTER_MIX_BALANCED
_SHARES = _MIX["BTC"]
_CL = {"BTC": "BTC", "MSTR": "BTC", "MSTU": "BTC", "ETH": "BTC", "XLE": "XLE"}


# ── configuration ──────────────────────────────────────────────────────────
def test_mix_config_is_ordered_complete_and_balanced_only():
    assert sum(_SHARES.values()) == pytest.approx(1.0)
    assert _SHARES == {"MSTR": 0.45, "MSTU": 0.30, "BTC": 0.20, "ETH": 0.05}
    order = sorted(_SHARES, key=lambda k: -_SHARES[k])
    assert order == ["MSTR", "MSTU", "BTC", "ETH"]               # high → low
    assert set(oc.CLUSTER_MIX) == {"Balanced"}
    assert oc.cluster_mix_for("Balanced") == _MIX
    assert oc.cluster_mix_for("Balanced", "v2") == _MIX
    assert oc.cluster_mix_for("Growth") is None
    assert oc.cluster_mix_for("Aggressive") is None
    assert oc.cluster_mix_for("Balanced", "v1") is None          # V1 never had it
    assert oc.cluster_mix_for("Balanced", "combined") == _MIX    # applied from the cut-over
    assert oc.cluster_mix_for("Growth", "combined") is None


def test_mix_fits_balanced_per_sleeve_caps_at_a_full_cluster():
    caps = oc.caps_for("Balanced")
    for k, share in _SHARES.items():
        assert share * oc.CLUSTER_CAP <= caps[k] + 1e-9, k       # e.g. MSTU 9 % ≤ 10 %
    assert set(_SHARES) <= set(oc.ASSET_META)                     # all four are real sleeves


# ── the helper ─────────────────────────────────────────────────────────────
def test_apply_cluster_mix_divides_the_cluster_total_and_leaves_the_rest():
    caps = oc.caps_for("Balanced")
    w = {"BTC": 0.05, "MSTR": 0.15, "MSTU": 0.05, "ETH": 0.05, "XLE": 0.2}   # cluster 0.30
    out = oc._apply_cluster_mix(w, _CL, _MIX, caps)
    assert out["XLE"] == pytest.approx(0.2)                       # other clusters untouched
    assert sum(out[k] for k in _SHARES) == pytest.approx(0.30)    # total preserved
    for k, s in _SHARES.items():
        assert out[k] == pytest.approx(0.30 * s)
    assert oc._apply_cluster_mix(w, _CL, None, caps) == w         # no mix → identity
    assert oc._apply_cluster_mix(w, _CL, {}, caps) == w


def test_apply_cluster_mix_renormalises_over_the_funded_members():
    caps = oc.caps_for("Balanced")
    w = {"MSTR": 0.10, "BTC": 0.10}                               # MSTU / ETH not funded
    out = oc._apply_cluster_mix(w, _CL, _MIX, caps)
    assert out["MSTR"] == pytest.approx(0.20 * 0.45 / 0.65)
    assert out["BTC"] == pytest.approx(0.20 * 0.20 / 0.65)
    assert "MSTU" not in out and "ETH" not in out                 # nothing is invented
    # a lone member keeps the cluster total (up to its own cap)
    solo = oc._apply_cluster_mix({"BTC": 0.12}, _CL, _MIX, caps)
    assert solo["BTC"] == pytest.approx(0.12)


def test_apply_cluster_mix_respects_sleeve_caps_and_strands_the_excess_in_sata():
    caps = oc.caps_for("Balanced")                                # MSTR 18 %, MSTU 10 %
    out = oc._apply_cluster_mix({"MSTR": 0.15, "MSTU": 0.15}, _CL, _MIX, caps)
    assert out["MSTU"] == pytest.approx(0.10)                     # capped, not 12 %
    assert out["MSTR"] <= 0.18 + 1e-9
    assert sum(out.values()) <= 0.30 + 1e-9                       # never more than before
    for k, v in out.items():
        assert v <= caps[k] + 1e-9


# ── the live gate ──────────────────────────────────────────────────────────
def _crypto_sleeves(version="v2"):
    return [_res(k, in_pos=True, tone="hold", parent="BTC", version=version)
            for k in ("BTC", "MSTR", "MSTU", "ETH")]


_BW = {"BTC": 0.30, "MSTR": 0.18, "MSTU": 0.10, "ETH": 0.30}


def test_gate_splits_the_cluster_by_the_mix_and_keeps_the_cluster_cap():
    caps = oc.caps_for("Balanced")
    sleeves = _crypto_sleeves() + [_res("XLE", in_pos=True, tone="hold")]
    base = oc.signal_gated_allocation(sleeves, {**_BW, "XLE": 0.3}, caps=caps)
    mixed = oc.signal_gated_allocation(sleeves, {**_BW, "XLE": 0.3}, caps=caps, cluster_mix=_MIX)
    tot = sum(mixed["target"][k] for k in _SHARES)
    assert tot == pytest.approx(sum(base["target"][k] for k in _SHARES))   # same cluster size
    assert tot <= oc.CLUSTER_CAP + 1e-9
    for k, s in _SHARES.items():
        assert mixed["target"][k] == pytest.approx(tot * s)
    assert mixed["target"]["XLE"] == pytest.approx(base["target"]["XLE"])
    assert mixed["cluster_mix"] == _MIX and base["cluster_mix"] is None
    assert mixed["tilt_target"] == mixed["target"]                # no prev book → nothing pinned
    cur = sum(mixed["current"][k] for k in _SHARES)
    for k, s in _SHARES.items():                                  # "what we hold now" splits too
        assert mixed["current"][k] == pytest.approx(cur * s)
    assert mixed["sata"] == pytest.approx(1.0 - sum(mixed["target"].values()))


def test_gate_never_re_splits_a_held_cluster_adds_only_wins():
    caps = oc.caps_for("Balanced")
    sleeves = _crypto_sleeves()
    prev = {"BTC": 0.08, "MSTR": 0.06, "MSTU": 0.04, "ETH": 0.02}  # an old, differently-split book
    g = oc.signal_gated_allocation(sleeves, _BW, caps=caps, prev_weights=prev, cluster_mix=_MIX)
    for k, v in prev.items():
        assert g["target"][k] == pytest.approx(v)                 # pinned, never trimmed or re-split
    assert set(g["held_pinned"]) == set(prev)


def test_gate_fresh_entry_is_sized_at_the_mix_even_with_a_prev_book():
    caps = oc.caps_for("Balanced")
    sleeves = [_res(k, tone="buy", state="ENTRY", label="BUY", parent="BTC")
               for k in ("BTC", "MSTR", "MSTU", "ETH")]
    g = oc.signal_gated_allocation(sleeves, _BW, caps=caps, prev_weights={"XLE": 0.3},
                                   cluster_mix=_MIX)
    tot = sum(g["target"][k] for k in _SHARES)
    assert tot > 0
    for k, s in _SHARES.items():
        assert g["target"][k] == pytest.approx(tot * s)


def test_gate_ignores_the_mix_for_v1_results():
    caps = oc.caps_for("Balanced")
    v1 = oc.signal_gated_allocation(_crypto_sleeves("v1"), _BW, caps=caps, cluster_mix=_MIX)
    plain = oc.signal_gated_allocation(_crypto_sleeves("v1"), _BW, caps=caps)
    assert v1["target"] == plain["target"] and v1["cluster_mix"] is None


def test_historical_allocation_takes_the_mix():
    caps = oc.caps_for("Balanced")
    snap = dict(rows=[dict(key=k, parent="BTC", in_pos=True) for k in _SHARES])
    book = oc.historical_allocation(snap, _BW, caps=caps, cluster_cap=0.30, cluster_mix=_MIX)["book"]
    for k, s in _SHARES.items():
        assert book[k] == pytest.approx(0.30 * s)


# ── the replay ─────────────────────────────────────────────────────────────
_PAR = {"BBB": "AAA", "CCC": "AAA", "DDD": "AAA"}
_RMIX = {"AAA": {"AAA": 0.4, "BBB": 0.3, "CCC": 0.2, "DDD": 0.1}}
_RBW = {k: 0.3 for k in ("AAA", "BBB", "CCC", "DDD")}


def _split_days(rep, funded_all):
    w = rep["weights"][["AAA", "BBB", "CCC", "DDD"]]
    return w.loc[funded_all]


def test_replay_applies_the_split_every_day_the_whole_cluster_is_funded():
    res = _universe(parents=_PAR)
    rep = oc.replay_gated_allocation(res, base_weights=_RBW, adds_only=0, cluster_cap=0.30,
                                     cluster_mix=_RMIX)
    act = rep["active"][["AAA", "BBB", "CCC", "DDD"]]
    full = act.all(axis=1)
    assert full.sum() > 20
    w = rep["weights"][["AAA", "BBB", "CCC", "DDD"]]
    tot = w.sum(axis=1)
    for k, s in _RMIX["AAA"].items():
        pd.testing.assert_series_equal((w[k] / tot)[full], pd.Series(s, index=w.index[full]),
                                       check_names=False, atol=1e-9)
    assert (tot <= 0.30 + 1e-9).all()
    assert (rep["weights"].sum(axis=1) + rep["sata"]).round(9).eq(1.0).all()
    assert rep["cluster_mix"] == _RMIX


def test_replay_split_keeps_the_cluster_total_of_the_unmixed_replay():
    res = _universe(parents=_PAR)
    kw = dict(base_weights=_RBW, adds_only=0, cluster_cap=0.30)
    plain = oc.replay_gated_allocation(res, **kw)
    mixed = oc.replay_gated_allocation(res, cluster_mix=_RMIX, **kw)
    cols = ["AAA", "BBB", "CCC", "DDD"]
    # with no adds-only pin the day's cluster total is unchanged by the split
    pd.testing.assert_series_equal(plain["weights"][cols].sum(axis=1),
                                   mixed["weights"][cols].sum(axis=1), atol=1e-9)
    assert plain["cluster_mix"] is None


def test_replay_mix_switches_on_from_a_date():
    res = _universe(parents=_PAR)
    cut = res[0]["dates"][60]
    kw = dict(base_weights=_RBW, adds_only=0, cluster_cap=0.30)
    plain = oc.replay_gated_allocation(res, **kw)
    mixed = oc.replay_gated_allocation(res, cluster_mix=_RMIX, cluster_mix_from=cut, **kw)
    pd.testing.assert_frame_equal(mixed["weights"].loc[:cut - pd.Timedelta(days=1)],
                                  plain["weights"].loc[:cut - pd.Timedelta(days=1)])
    assert not mixed["weights"].loc[cut:].equals(plain["weights"].loc[cut:])


def test_replay_adds_only_pins_held_names_through_the_split():
    res = _universe(parents=_PAR)
    mixed = oc.replay_gated_allocation(res, base_weights=_RBW, adds_only=0.08,
                                       cluster_cap=0.30, cluster_mix=_RMIX)
    w = mixed["weights"]
    assert (mixed["weights"].sum(axis=1) + mixed["sata"]).round(9).eq(1.0).all()
    assert (w[["AAA", "BBB", "CCC", "DDD"]].sum(axis=1) <= 0.30 + 1e-9).all()
    # a held sleeve's weight never falls while its position carries (adds-only)
    act = mixed["active"]
    for k in ("AAA", "BBB", "CCC", "DDD"):
        held = act[k] & act[k].shift(1, fill_value=False)
        step = w[k].diff()[held]
        # only the cluster cap / book-overflow scale-downs may shrink a pinned name
        assert (step > -0.30).all()


def test_walkforward_replay_ignores_the_mix_for_v1_and_applies_it_for_v2():
    res = _universe(n_days=300, parents=_PAR)
    kw = dict(n_samples=200, min_hist=40)
    v1 = oc.walkforward_gated_replay(res, version="v1", cluster_mix=_RMIX, **kw)
    v1_plain = oc.walkforward_gated_replay(res, version="v1", **kw)
    pd.testing.assert_series_equal(v1["ret"], v1_plain["ret"])
    assert v1["weights"].equals(v1_plain["weights"])
    v2 = oc.walkforward_gated_replay(res, version="v2", cluster_mix=_RMIX, **kw)
    v2_plain = oc.walkforward_gated_replay(res, version="v2", **kw)
    assert not v2["weights"].equals(v2_plain["weights"])
    assert v2["cluster_mix"] == _RMIX


# ── every consumer must pass the profile's mix ────────────────────────────
_GUARDED = {"signal_gated_allocation", "walkforward_gated_replay", "historical_allocation"}


def _unguarded_calls():
    """Call sites in app/ and scripts/ of the three allocator entry points that
    do not pass ``cluster_mix=`` — a surface that forgot the mix would show or
    trade a different book from the one the replay validated."""
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
                if name in _GUARDED and "cluster_mix" not in {k.arg for k in node.keywords}:
                    bad.append(f"{p.relative_to(REPO)}:{node.lineno} {name}")
    return bad


def test_every_allocator_call_site_passes_cluster_mix():
    assert _unguarded_calls() == []


def test_combined_replay_applies_the_mix_only_from_the_cutover():
    import numpy as np
    import strategy_version as sv
    from test_strategy_logic_v2 import _sleeve
    cut = pd.Timestamp(sv.STRATEGY_VERSION_START)
    idx = pd.bdate_range(cut - pd.Timedelta(days=200), periods=300)
    assert idx[0] < cut < idx[-1]

    def gen(version, seed):
        rng = np.random.default_rng(seed)
        out = []
        for k in ("AAA", "BBB", "CCC", "DDD"):
            r = _sleeve(k, _PAR.get(k, k), idx, rng)
            r["version"] = version
            out.append(r)
        return out

    v1, v2 = gen("v1", 5), gen("v2", 6)
    entry = idx[idx >= cut][3]                       # the cluster opens fresh, 3 bars after the cut-over
    for r in v1 + v2:
        r["pos_series"] = pd.Series((idx >= entry).astype(float), index=idx)
    comb = oc.combine_results(v1, v2)
    kw = dict(version="combined", gens=(v1, v2), n_samples=200, min_hist=40)
    mixed = oc.walkforward_gated_replay(comb, cluster_mix=_RMIX, **kw)
    plain = oc.walkforward_gated_replay(comb, **kw)
    pre = slice(None, cut - pd.Timedelta(days=1))
    pd.testing.assert_series_equal(mixed["ret"].loc[pre], plain["ret"].loc[pre])
    assert mixed["weights"].loc[pre].equals(plain["weights"].loc[pre])
    cols = ["AAA", "BBB", "CCC", "DDD"]
    w = mixed["weights"][cols].loc[entry]
    assert w.sum() > 0
    for k, sh in _RMIX["AAA"].items():                # the fresh entry is sized at the split
        assert w[k] / w.sum() == pytest.approx(sh)
    p = plain["weights"][cols].loc[entry]
    assert p.sum() == pytest.approx(w.sum())          # same cluster size, different split
    assert not (p / p.sum()).round(6).equals((w / w.sum()).round(6))
