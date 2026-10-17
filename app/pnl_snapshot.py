"""Precomputed 📈 Overall-strategy-P&L snapshot.

The P&L section needs, per strategy view (V1 / V2 / Combined) and risk profile,
a full walk-forward gated replay — plus the price-returns matrix, the
equal-weight buy-&-hold replay and the sleeve price table.  Those only change
when a new daily bar lands, yet the app used to recompute them (tens of seconds)
on every cold start / 30-minute cache expiry.

``scripts/build_pnl_snapshot.py`` (run daily by .github/workflows/
build-pnl-snapshot.yml) calls :func:`build` and publishes the result to the
single-commit ``data-snapshots`` branch — NOT to ``main``, so the repository
does not accrue a ~1 MB binary per build.  The app downloads it with
:func:`fetch_remote` into ``data/overall/pnl_snapshot.pkl.xz`` (git-ignored
local cache) and calls :func:`load` to read it back in well under a second.  :func:`load` returns ``None`` — and the app falls back
to its original live computation — whenever the snapshot is missing, unreadable,
its newest bar is older than ``MAX_BAR_AGE_DAYS`` or it was built by different engine code
(``code_fingerprint``), so a stale file can never be served silently.

The snapshot also carries the V1 / V2 ``run_universe`` results, so the live
cockpit loads them instead of re-running every instrument's engine.

Only the daily-stable results are stored.  Dollar sizing, date windows, costs
and taxes are applied by the app on top, per user input.
"""
from __future__ import annotations

import hashlib
import lzma
import os
import pickle
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import overall_core as ov

_APP_DIR = Path(__file__).resolve().parent
SNAPSHOT_PATH = _APP_DIR.parent / "data" / "overall" / "pnl_snapshot.pkl.xz"

# where CI publishes it (a one-commit branch, force-pushed each build)
REMOTE_REPO = "yoursprerit/btc-range-model"
REMOTE_BRANCH = "data-snapshots"
REMOTE_FILE = "pnl_snapshot.pkl.xz"
REMOTE_TTL_SECS = 30 * 60   # re-check the branch at most this often
_RETRY_SECS = 5 * 60        # …and wait this long after a failed download
_last_failure = [0.0]

SCHEMA = 2                  # 2: xz-compressed, results added (1 was gzip)
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
    sata = _sata_source()
    if version not in rets_cache:
        rets = ov.returns_matrix(results)
        pos = ov.position_matrix(results, rets.index)
        bm = ov.benchmarks(rets, results, pos=pos, sata_daily=sata)
        rets_cache[version] = (rets, bm)
    rets, bm = rets_cache[version]
    prof = ov.RISK_PROFILES[name]
    gens = ((_results("v1", cache), _results("v2", cache))
            if version == "combined" else None)
    wf = ov.walkforward_gated_replay(
        results, caps=ov.caps_for(name), mdd_floor=prof["mdd_floor"],
        objective=prof["objective"], sata_daily=sata, tilt=True,
        version=version, gens=gens, cluster_mix=ov.cluster_mix_for(name, version),
        adds_only_cluster=ov.cluster_adds_band_for(name, version),
        sata_cap=ov.sata_cap_for(version))
    # the weight optimiser (seeded Monte-Carlo, a few seconds per profile) is
    # the other heavy step the cockpit repeats on every cold cache; same inputs
    # as ``get_profile_pack`` feeds it, so the app can reuse it under the same
    # returns-matrix match as the replay
    pos = ov.position_matrix(results, rets.index)
    opt = ov.optimize_weights(rets, caps=ov.caps_for(name), pos=pos,
                              sata_daily=ov.SATA_DAILY,
                              mdd_floor=prof["mdd_floor"],
                              objective=prof["objective"], fundamental=False)
    return dict(wf=wf, opt=opt)


def _sata_source():
    """SATA's realised return series for the P&L paths (``load_sata_returns``),
    the coupon when no history is on disk."""
    sr = ov.load_sata_returns()
    return sr if sr is not None else ov.SATA_DAILY


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
    # the pinned-data stamp as checked out, BEFORE the engines run: they top up
    # stale datasets in their own workspace (uncommitted), so the end-of-run
    # ``vintage`` below never matches the repo the app is deployed from
    vintage_in = ov.data_vintage()
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
        vintage_in=vintage_in,
        last_bar=str(pd.Timestamp(last_bar).date()),
        profiles=list(profiles), versions=list(versions),
        packs=packs, shared=shared,
        # the per-generation universe runs themselves — the heaviest thing the
        # web process used to recompute (tens of seconds of CPU every 30 min)
        results={v: _results(v, cache) for v in ("v1", "v2")},
        price_returns=ov.bh_returns_matrix(v2),
        bh_replay=ov.equal_weight_bh_replay(v2, index=rets_v2.index),
        sleeve_prices=_sleeve_prices(comb),
        # the SATA leg's realised returns the replays above were priced on, so
        # the app's attribution / daily P&L reconcile to the snapshotted curves
        sata_returns=ov.load_sata_returns(),
    )


def get_pack(snap: dict, version: str, name: str):
    """One (view, profile) portfolio in the shape the P&L section reads
    (``wf`` / ``opt`` / ``curves`` / ``rets``), or ``None`` if it was not
    snapshotted.  ``opt`` is ``None`` for a snapshot built before it was stored."""
    p = snap["packs"].get((str(version).lower(), name))
    sh = snap["shared"].get(str(version).lower())
    if not p or not sh:
        return None
    return dict(wf=p["wf"], opt=p.get("opt"), rets=sh["rets"], version=version,
                profile=name, curves={STRAT_CURVE: p["wf"]["equity"], **sh["base_curves"]})


def get_results(snap: dict | None, version: str) -> dict | None:
    """The snapshotted ``run_universe`` output for a single generation
    (``"v1"`` / ``"v2"``) as ``dict(results=..., computed_at=...)``, or ``None``
    when it is absent or the pinned datasets have changed since the build
    (every number is a pure function of strategy version + the data vintage,
    so an equal vintage means the live computation would be identical).  The
    comparison is against the stamp taken before the build ran (``vintage_in``)."""
    try:
        res = (snap or {}).get("results", {}).get(str(version).lower())
        if not res or snap.get("vintage_in") != ov.data_vintage():
            return None
        return dict(results=res, computed_at=pd.Timestamp(snap["built_at"]))
    except Exception:
        return None


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
    # xz is ~3x smaller than gzip on these frames/arrays (1.0 vs 2.9 MB) and
    # lossless, so nothing the app reads changes
    with lzma.open(tmp, "wb", preset=6) as fh:
        pickle.dump(payload, fh, protocol=4)
    tmp.replace(path)


def read(path: Path = SNAPSHOT_PATH) -> dict:
    """The raw payload at *path* (no freshness checks); raises if unreadable."""
    with lzma.open(path, "rb") as fh:
        return pickle.load(fh)


def fetch_remote(dest: Path = SNAPSHOT_PATH, token: str | None = None,
                 repo: str = REMOTE_REPO, branch: str = REMOTE_BRANCH,
                 ttl: float = REMOTE_TTL_SECS, timeout: float = 30) -> str:
    """Make sure *dest* holds a recent copy of the CI-published snapshot.

    Returns ``"fresh"`` (local copy younger than *ttl*), ``"downloaded"``,
    ``"backoff"`` (a recent attempt failed) or ``"failed"``; on any failure the
    previous local copy, if any, is left untouched — :func:`load` and the app's
    live computation take it from there, so this never raises.  A public repo
    is read from raw.githubusercontent.com; with a *token* (private repo) the
    contents API is used instead."""
    now = time.time()
    try:
        if dest.exists() and now - dest.stat().st_mtime < ttl:
            return "fresh"
    except OSError:
        pass
    if now - _last_failure[0] < _RETRY_SECS:
        return "backoff"
    try:
        if token:
            req = urllib.request.Request(
                f"https://api.github.com/repos/{repo}/contents/{REMOTE_FILE}"
                f"?ref={branch}",
                headers={"Accept": "application/vnd.github.raw+json",
                         "Authorization": f"Bearer {token}"})
        else:
            req = urllib.request.Request(
                f"https://raw.githubusercontent.com/{repo}/{branch}/{REMOTE_FILE}")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            blob = r.read()
        pickle.loads(lzma.decompress(blob))        # reject a truncated / wrong file
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + f".{os.getpid()}.tmp")
        tmp.write_bytes(blob)
        tmp.replace(dest)
        return "downloaded"
    except Exception:
        _last_failure[0] = now
        return "failed"


def load(path: Path = SNAPSHOT_PATH, now: pd.Timestamp | None = None,
         check_code: bool = True):
    """The snapshot payload, or ``None`` if absent / unreadable / stale /
    built by different engine code."""
    try:
        snap = read(path)
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
