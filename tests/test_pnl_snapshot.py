"""The precomputed P&L snapshot is only ever served when it is safe to."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
for _p in (str(_REPO / "app"), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pnl_snapshot as ps  # noqa: E402


def _payload(last_bar="2026-10-09", fingerprint=None, schema=ps.SCHEMA):
    return dict(schema=schema, built_at="2026-10-10T12:00:00",
                fingerprint=fingerprint or ps.code_fingerprint(),
                vintage={}, last_bar=last_bar, packs={}, shared={})


def test_roundtrip_and_fresh(tmp_path):
    f = tmp_path / "s.pkl.gz"
    ps.save(_payload(), f)
    assert ps.load(f, now=pd.Timestamp("2026-10-10")) is not None


def test_rejected_when_bar_is_stale(tmp_path):
    f = tmp_path / "s.pkl.gz"
    ps.save(_payload(), f)
    assert ps.load(f, now=pd.Timestamp("2026-10-20")) is None


def test_rejected_when_engine_code_differs(tmp_path):
    f = tmp_path / "s.pkl.gz"
    ps.save(_payload(fingerprint="deadbeef"), f)
    assert ps.load(f, now=pd.Timestamp("2026-10-10")) is None
    assert ps.load(f, now=pd.Timestamp("2026-10-10"), check_code=False) is not None


def test_rejected_when_missing_or_corrupt_or_wrong_schema(tmp_path):
    assert ps.load(tmp_path / "nope.pkl.gz") is None
    bad = tmp_path / "bad.pkl.gz"
    bad.write_bytes(b"not a pickle")
    assert ps.load(bad) is None
    f = tmp_path / "old.pkl.gz"
    ps.save(_payload(schema=0), f)
    assert ps.load(f, now=pd.Timestamp("2026-10-10")) is None


def test_rets_match_guard():
    idx = pd.date_range("2026-01-01", periods=5)
    a = pd.DataFrame(np.random.default_rng(0).normal(size=(5, 3)),
                     index=idx, columns=list("xyz"))
    assert ps.rets_match(a, a.copy())
    assert not ps.rets_match(a, a.iloc[:-1])                 # calendar differs
    assert not ps.rets_match(a, a.rename(columns={"z": "w"}))  # sleeves differ
    b = a.copy(); b.iloc[-1, 0] += 1e-4
    assert not ps.rets_match(a, b)                           # numbers differ


def test_get_pack_assembles_curves():
    eq = pd.Series([1.0, 1.1])
    snap = dict(packs={("v2", "Growth"): dict(wf=dict(equity=eq))},
                shared={"v2": dict(rets=pd.DataFrame(), base_curves={ps.BH_CURVE: eq})})
    p = ps.get_pack(snap, "V2", "Growth")
    assert set(p["curves"]) == {ps.STRAT_CURVE, ps.BH_CURVE}
    assert ps.get_pack(snap, "v1", "Growth") is None
