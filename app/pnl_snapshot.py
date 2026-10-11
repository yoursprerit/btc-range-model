"""Precomputed 📈 Overall-strategy-P&L snapshot.

The P&L section needs, per strategy view (V1 / V2 / Combined) and risk profile,
a full walk-forward gated replay — plus the price-returns matrix, the
equal-weight buy-&-hold replay and the sleeve price table.  Those only change
when a new daily bar lands, yet the app used to recompute them (tens of seconds)
on every cold start / 30-minute cache expiry.

``scripts/build_pnl_snapshot.py`` (run daily by .github/workflows/
build-pnl-snapshot.yml) calls :func:`build` and commits the result to
``data/overall/pnl_snapshot.pkl``; the app calls :func:`load` and reads it back
in well under a second.  :func:`load` returns ``None`` — and the app falls back
to its original live computation — whenever the snapshot is missing, unreadable,
its newest bar is older than ``MAX_BAR_AGE_DAYS`` or it was built by different engine code
(``code_fingerprint``), so a stale file can never be served silently.

Only the daily-stable results are stored.  Dollar sizing, date windows, costs
and taxes are applied by the app on top, per user input.
"""
from __future__ import annotations

import gzip
import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import overall_core as ov

_APP_DIR = Path(__file__).resolve().parent
SNAPSHOT_PATH = _APP_DIR.parent / "data" / "overall" / "pnl_snapshot.pkl.gz"

SCHEMA = 1
MAX_BAR_AGE_DAYS = 4        # newest bar older than this = the daily pipeline is broken
VERSIONS = ("v1", "v2", "combined")

# curve labels — the app's STRAT_CURVE / BH_CURVE (overall_app imports these)
STRAT_CURVE = "Overall strategy (gated replay)"
BH_CURVE = "Equal-weight Buy & Hold"


def ui_profiles() -> list[str]:
    """Profiles the UI offers (Aggressive is retired from the interface)."""
    return [p for p in ov.RISK_PROFILES if p != "Aggressive"]


def code_fingerprint() -> str:
    """Hash of every engine module a replay depends on (all app/*.py except the
    Streamlit page modules).  A snapshot built by different code is rejected."""
    h = hashlib.sha256()
    for p in sorted(_APP_DIR.glob("*.py")):
        if p.name.endswith("_app.py") or p.name == "pnl_snapshot.py":
            continue
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


# ── compute (used by the builder; mirrors the app's live path exactly) ───────
def _results(version: str, cache: dict) -> list[dict]:
    if version not in cache:
        if version == "combined":
            cache[version] = ov.combine_results(_results("v1", cache),
                                                _results("v2", cache))
        else:
            cache[version] = ov.run_universe(version)
    return cache[version]


def _pack(version: str, name: str, cache: dict, rets_cache: dict) -> dict:
    # only the profile-specific replay is stored per pack; the returns matrix
    # and the benchmark curves are identical across profiles and live once per
    # version under ``shared`` (see :func:`get_pack`)
    results = _results(version, cache)
    if version not in rets_cache:
        rets = ov.returns_matrix(results)
        pos = ov.position_matrix(results, rets.index)
        bm = ov.benchmarks(rets, results, pos=pos, sata_daily=ov.SATA_DAILY)
        rets_cache[version] = (rets, bm)
    rets, bm = rets_cache[version]
    prof = ov.RISK_PROFILES[name]
    gens = ((_results("v1", cache), _results("v2", cache))
            if version == "combined" else None)
    wf = ov.walkforward_gated_replay(
        results, caps=ov.caps_for(name), mdd_floor=prof["mdd_floor"],
        objective=prof["objective"], sata_daily=ov.SATA_DAILY, tilt=True,
        version=version, gens=gens, cluster_mix=ov.cluster_mix_for(name, version),
        adds_only_cluster=ov.cluster_adds_band_for(name, version))
    return dict(wf=wf)


def _sleeve_prices(results: list[dict]):
    cols = {}
    for r in results:
        try:
            ser = ov.asset_close_series(r)
        except Exception:
            ser = None
        if ser is not None:
            cols[r["key"]] = ser[~ser.index.duplicated()]
    return (pd.DataFrame(cols).reindex(ov.returns_matrix(results).index)
            if cols else None)


def build(profiles: list[str] | None = None, versions=VERSIONS,
          log=print) -> dict:
    """Run every replay the P&L section can ask for and return the payload."""
    profiles = profiles or ui_profiles()
    cache: dict = {}
    rets_cache: dict = {}
    packs: dict = {}
    for v in versions:
        for name in profiles:
            log(f"  replay  {v:9s} {name}")
            packs[(v, name)] = _pack(v, name, cache, rets_cache)
    v2 = _results("v2", cache)
    rets_v2 = (rets_cache.get("v2") or (ov.returns_matrix(v2), None))[0]
    comb = _results("combined", cache)
    shared = {v: dict(rets=rets_cache[v][0],
                      base_curves={"Equal-weight strategies":
                                   rets_cache[v][1]["strat_equal"]["equity"],
                                   BH_CURVE: rets_cache[v][1]["bh_equal"]["equity"]})
              for v in versions}
    last_bar = max(sh["rets"].index[-1] for sh in shared.values())
    return dict(
        schema=SCHEMA,
        built_at=pd.Timestamp.utcnow().isoformat(),
        fingerprint=code_fingerprint(),
        vintage=ov.data_vintage(),
        last_bar=str(pd.Timestamp(last_bar).date()),
        profiles=list(profiles), versions=list(versions),
        packs=packs, shared=shared,
        price_returns=ov.bh_returns_matrix(v2),
        bh_replay=ov.equal_weight_bh_replay(v2, index=rets_v2.index),
        sleeve_prices=_sleeve_prices(comb),
    )


def get_pack(snap: dict, version: str, name: str):
    """One (view, profile) portfolio in the shape the P&L section reads
    (``wf`` / ``curves`` / ``rets``), or ``None`` if it was not snapshotted."""
    p = snap["packs"].get((str(version).lower(), name))
    sh = snap["shared"].get(str(version).lower())
    if not p or not sh:
        return None
    return dict(wf=p["wf"], rets=sh["rets"], version=version, profile=name,
                curves={STRAT_CURVE: p["wf"]["equity"], **sh["base_curves"]})


def rets_match(snap_rets: pd.DataFrame, live_rets: pd.DataFrame) -> bool:
    """True when a snapshot replay was built on the very same returns matrix the
    live universe just produced (same calendar, same sleeves, same numbers) —
    the guard that lets the app reuse a snapshotted replay next to live
    ``results`` without ever mixing two data vintages."""
    try:
        if (not snap_rets.index.equals(live_rets.index)
                or list(snap_rets.columns) != list(live_rets.columns)):
            return False
        return bool(np.allclose(snap_rets.to_numpy(float),
                                live_rets.to_numpy(float),
                                rtol=0, atol=1e-9, equal_nan=True))
    except Exception:
        return False


# ── storage ───────────────────────────────────────────────────────────────────
def content_key(payload: dict) -> str:
    """What makes two snapshots equivalent (the builder skips an unchanged one)."""
    return f"{payload['fingerprint']}|{payload['last_bar']}|{payload['vintage']}"


def save(payload: dict, path: Path = SNAPSHOT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as raw, \
            gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0) as fh:
        pickle.dump(payload, fh, protocol=4)
    tmp.replace(path)


def load(path: Path = SNAPSHOT_PATH, now: pd.Timestamp | None = None,
         check_code: bool = True):
    """The snapshot payload, or ``None`` if absent / unreadable / stale /
    built by different engine code."""
    try:
        with gzip.open(path, "rb") as fh:
            snap = pickle.load(fh)
        if snap.get("schema") != SCHEMA:
            return None
        if check_code and snap.get("fingerprint") != code_fingerprint():
            return None
        now = (now or pd.Timestamp.utcnow()).tz_localize(None)
        if (now.normalize() - pd.Timestamp(snap["last_bar"])).days > MAX_BAR_AGE_DAYS:
            return None
        return snap
    except Exception:
        return None
