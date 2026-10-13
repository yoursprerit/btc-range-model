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


# ── download from the data-snapshots branch ──────────────────────────────────
class _Resp:
    def __init__(self, blob): self._b = blob
    def read(self): return self._b
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _xz_blob(tmp_path):
    f = tmp_path / "src.pkl.xz"
    ps.save(_payload(), f)
    return f.read_bytes()


def test_fetch_remote_downloads_then_stays_fresh(tmp_path, monkeypatch):
    blob, calls = _xz_blob(tmp_path), []
    monkeypatch.setattr(ps.urllib.request, "urlopen",
                        lambda req, timeout=0: calls.append(req.full_url) or _Resp(blob))
    monkeypatch.setattr(ps, "_last_failure", [0.0])
    dest = tmp_path / "local.pkl.xz"
    assert ps.fetch_remote(dest) == "downloaded"
    assert calls[0].startswith("https://raw.githubusercontent.com/")
    assert ps.load(dest, now=pd.Timestamp("2026-10-10")) is not None
    assert ps.fetch_remote(dest) == "fresh" and len(calls) == 1


def test_fetch_remote_with_token_uses_contents_api(tmp_path, monkeypatch):
    blob, seen = _xz_blob(tmp_path), []
    monkeypatch.setattr(ps.urllib.request, "urlopen",
                        lambda req, timeout=0: seen.append(req) or _Resp(blob))
    monkeypatch.setattr(ps, "_last_failure", [0.0])
    assert ps.fetch_remote(tmp_path / "l.pkl.xz", token="t") == "downloaded"
    assert "api.github.com" in seen[0].full_url
    assert seen[0].get_header("Authorization") == "Bearer t"


def test_fetch_remote_failure_keeps_old_copy_and_backs_off(tmp_path, monkeypatch):
    def boom(req, timeout=0): raise OSError("offline")
    monkeypatch.setattr(ps.urllib.request, "urlopen", boom)
    monkeypatch.setattr(ps, "_last_failure", [0.0])
    dest = tmp_path / "local.pkl.xz"
    ps.save(_payload(), dest)
    import os
    os.utime(dest, (0, 0))                       # old enough to need a refresh
    assert ps.fetch_remote(dest) == "failed"
    assert ps.fetch_remote(dest) == "backoff"    # no hammering GitHub
    assert ps.load(dest, now=pd.Timestamp("2026-10-10")) is not None


def test_fetch_remote_rejects_a_corrupt_download(tmp_path, monkeypatch):
    monkeypatch.setattr(ps.urllib.request, "urlopen",
                        lambda req, timeout=0: _Resp(b"<html>404</html>"))
    monkeypatch.setattr(ps, "_last_failure", [0.0])
    dest = tmp_path / "local.pkl.xz"
    assert ps.fetch_remote(dest) == "failed" and not dest.exists()


# ── optimiser output rides in each pack ──────────────────────────────────────
def _snap_with_pack(opt):
    rets = pd.DataFrame({"A": [0.01, -0.02]}, index=pd.to_datetime(["2026-10-09", "2026-10-12"]))
    wf = dict(equity=pd.Series([1.0, 1.01]))
    pack = dict(wf=wf) if opt is None else dict(wf=wf, opt=opt)
    return dict(packs={("v2", "Balanced"): pack},
                shared={"v2": dict(rets=rets, base_curves={})}), rets


def test_get_pack_exposes_the_stored_optimiser_output():
    snap, rets = _snap_with_pack({"optimal": {"weights": {"A": 1.0}}, "cols": ["A"]})
    got = ps.get_pack(snap, "V2", "Balanced")
    assert got["opt"]["cols"] == ["A"]
    assert ps.rets_match(got["rets"], rets)


def test_get_pack_opt_is_none_for_a_snapshot_built_before_it_was_stored():
    snap, _ = _snap_with_pack(None)
    got = ps.get_pack(snap, "v2", "Balanced")
    assert got is not None and got["opt"] is None
