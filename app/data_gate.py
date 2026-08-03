"""Data-quality gate + pinned daily snapshots + dataset audit trail.

WHY THIS EXISTS
---------------
Every equity/gold sleeve used to re-fetch its FULL daily history live from
Yahoo on each cache expiry (~every 5–30 min), and the walk-forward backtests
were recomputed from whatever that fetch happened to return.  Yahoo's free
feed is not a stable historical source on shared egress IPs: symbols come
back empty (silently dropping macro feature columns), sessions go missing
(silently repaired from the hourly feed or a second provider at slightly
different prints), and rate-limited fetches fall back to stale committed
CSVs.  Two page loads hours apart could therefore simulate the SAME strategy
over DIFFERENT input histories — which is exactly how "+970% in the morning,
+1204% in the evening" happens.

WHAT IT DOES
------------
``gated_daily`` wraps a sleeve's daily fetch with four behaviours:

1. **Pin** — once a fetch has passed quality control and covers the most
   recent completed US session, its completed-bars history is persisted as
   the sleeve's snapshot (CSV + manifest with a content hash).  Until the
   next session completes, every subsequent load reuses that byte-identical
   snapshot — so the morning and evening backtests are computed from the
   SAME data, and no full-history fetch is repeated intraday.
2. **Validate** — a fresh fetch is accepted only if it passes the quality
   checks below (required columns present, sane index, positive finite
   prices, no absurd one-day traded-price moves, no regression versus the
   snapshot, and per-day return agreement with the snapshot on the
   overlapping history).  A failed fetch is retried once, then REJECTED.
3. **Fall back** — a rejected fetch never reaches the models: the last
   known-good snapshot is served instead, and the rejection is recorded.
4. **Audit** — every load appends a record (decision, source, span, rows,
   SHA-256 content hash, failed checks, consumer app) to
   ``runtime/dataset_audit.json``; the 🕵️ Daily Audit app renders the trail.

The in-progress *today* bar is handled separately from history: pinned
history is completed-bars only, and the live partial bar is re-attached on
top from a tiny ``range=5d`` fetch (never persisted, never audited as
history) so the live views keep their intraday freshness.  In publisher mode
(``OVERALL_COMPLETED_BARS_ONLY``) no partial bar is ever attached.

This module is Streamlit-free and thread-safe (the Overall engine loads all
sleeves concurrently); all persistence is atomic and best-effort — a failed
write never breaks a caller.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import freshness as _fr

_REPO_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_DIR = _REPO_ROOT / "runtime"
AUDIT_JSON = RUNTIME_DIR / "dataset_audit.json"

_AUDIT_LOCK = threading.Lock()
_MAX_AUDIT_ENTRIES = 60          # kept per dataset key (newest first)

# One-day |move| beyond this on a TRADED close is treated as corrupt data
# (a zeroed/garbage print, or a large-ratio split splice at ~90%).  The bound
# must clear the worst REAL day a 3× leveraged ETF has printed — ERX closed
# −60.1% on 2020-03-09 (COVID + oil-price-war crash) — so 60% is too tight;
# macro level series (VIX can jump >100%) are exempt from this check entirely.
MAX_ABS_TRADED_MOVE = 0.75

# Overlap agreement: per-day log returns of the primary close must match the
# snapshot on the shared completed history.  A few provider corrections are
# normal; a rewritten history is not.
_OVERLAP_DAYS = 120              # most recent shared sessions compared
_OVERLAP_RET_TOL = 1e-3          # |Δ log-return| below this is "agreeing"
_OVERLAP_MAX_BAD = 6             # >6 disagreeing days out of 120 → reject


@dataclass
class GateSpec:
    """Everything the gate needs to know about one daily dataset."""
    key: str                          # dataset key, e.g. "SOXX" or "GLDM"
    price_col: str                    # primary close column
    traded_close_cols: list[str]      # traded closes (extreme-move guarded)
    macro_close_cols: list[str]       # macro driver closes (presence-checked)
    snapshot_csv: Path                # pinned known-good history (CSV)
    manifest_json: Path | None = None # defaults next to the CSV
    min_rows: int = 260               # engines need ≥260 bars to run at all
    required_extra_cols: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.snapshot_csv = Path(self.snapshot_csv)
        if self.manifest_json is None:
            self.manifest_json = self.snapshot_csv.with_name(
                self.snapshot_csv.stem + "_manifest.json")

    @property
    def required_cols(self) -> list[str]:
        return (self.traded_close_cols + self.macro_close_cols
                + self.required_extra_cols)


def ticker_spec(cfg) -> GateSpec:
    """Build a GateSpec from a :class:`ticker_config.TickerConfig`."""
    import ticker_core as tc
    traded = [f"px_{s}" for s in ("open", "high", "low", "close")]
    traded += [f"{nm}_close" for nm in cfg.extra_syms]
    macro = [f"{nm}_close" for nm in cfg.macro_syms]
    return GateSpec(key=cfg.key, price_col="px_close",
                    traded_close_cols=traded, macro_close_cols=macro,
                    snapshot_csv=tc.cache_paths(cfg)["daily"])


# ════════════════════════════════════════════════════════════════════════
# Quality checks
# ════════════════════════════════════════════════════════════════════════
def _norm_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    out.index = idx.normalize()
    return out[~out.index.duplicated(keep="last")].sort_index()


def run_quality_checks(spec: GateSpec, df: pd.DataFrame | None,
                       snapshot: pd.DataFrame | None = None) -> dict:
    """Validate one freshly-fetched COMPLETED-BARS daily frame.

    Returns ``dict(passed, failed, checks=[{name, passed, detail}, ...])``.
    ``snapshot`` (the last known-good frame) powers the regression and
    history-agreement checks; pass ``None`` when there is none yet.
    """
    checks: list[dict] = []

    def add(name: str, passed: bool, detail: str = "") -> None:
        checks.append(dict(name=name, passed=bool(passed), detail=detail))

    if df is None or len(df) == 0:
        add("non_empty", False, "fetch returned no rows")
        return dict(passed=False, failed=["non_empty"], checks=checks)
    add("non_empty", True, f"{len(df)} rows")

    add("min_rows", len(df) >= spec.min_rows,
        f"{len(df)} rows (need ≥ {spec.min_rows})")

    idx = pd.DatetimeIndex(df.index)
    add("index_sane", idx.is_monotonic_increasing and idx.is_unique,
        "sorted, no duplicate sessions" if idx.is_monotonic_increasing
        and idx.is_unique else "unsorted or duplicated session dates")

    # primary close: positive and finite everywhere it is present
    px = (pd.to_numeric(df[spec.price_col], errors="coerce").dropna()
          if spec.price_col in df.columns else pd.Series(dtype=float))
    vals = px.to_numpy(float)
    ok_px = (len(px) >= spec.min_rows and bool(np.isfinite(vals).all())
             and bool((vals > 0).all()))
    add("price_positive_finite", ok_px,
        f"{spec.price_col}: {len(px)} non-null values"
        if ok_px else f"{spec.price_col} missing/non-positive/non-finite")

    # every declared column must be present with real data — a macro symbol
    # Yahoo silently failed to serve changes the FEATURE SET and therefore
    # every signal in the backtest, which is exactly the corruption to reject
    missing = [c for c in spec.required_cols
               if c not in df.columns or not df[c].notna().any()]
    add("required_columns", not missing,
        "all present" if not missing else f"missing/empty: {missing}")

    # absurd one-day moves on traded closes (bad prints, adjustment splices)
    offenders = []
    for c in spec.traded_close_cols:
        if not c.endswith("_close") and c != spec.price_col:
            continue
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) < 3:
            continue
        jumps = s.pct_change().abs()
        bad = jumps[jumps > MAX_ABS_TRADED_MOVE]
        if len(bad):
            offenders.append(f"{c}@{bad.index[-1].date()}"
                             f" ({bad.iloc[-1] * 100:+.0f}%)")
    add("no_extreme_traded_moves", not offenders,
        "all daily moves plausible" if not offenders
        else f"implausible one-day moves: {offenders}")

    # versus the last known-good snapshot: never accept a shrunken or
    # backdated history, nor one whose overlapping returns disagree
    if snapshot is not None and len(snapshot):
        add("no_row_regression", len(df) >= 0.98 * len(snapshot),
            f"{len(df)} rows vs snapshot {len(snapshot)}")
        add("no_span_regression", df.index.max() >= snapshot.index.max(),
            f"last {pd.Timestamp(df.index.max()).date()} vs snapshot "
            f"{pd.Timestamp(snapshot.index.max()).date()}")
        if (spec.price_col in df.columns
                and spec.price_col in snapshot.columns):
            a = pd.to_numeric(df[spec.price_col], errors="coerce")
            b = pd.to_numeric(snapshot[spec.price_col], errors="coerce")
            shared = a.index.intersection(b.index)[-_OVERLAP_DAYS:]
            if len(shared) > 10:
                ra = np.log(a.loc[shared]).diff().dropna()
                rb = np.log(b.loc[shared]).diff().dropna()
                common = ra.index.intersection(rb.index)
                dis = int((abs(ra.loc[common] - rb.loc[common])
                           > _OVERLAP_RET_TOL).sum())
                add("history_agreement", dis <= _OVERLAP_MAX_BAD,
                    f"{dis}/{len(common)} overlapping daily returns disagree "
                    f"with the snapshot (tolerance {_OVERLAP_RET_TOL})")

    failed = [c["name"] for c in checks if not c["passed"]]
    return dict(passed=not failed, failed=failed, checks=checks)


# ════════════════════════════════════════════════════════════════════════
# Snapshot persistence (CSV + manifest, atomic)
# ════════════════════════════════════════════════════════════════════════
def frame_sha(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.to_csv().encode()).hexdigest()[:16]


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def load_snapshot(spec: GateSpec) -> tuple[pd.DataFrame | None, dict]:
    """The pinned known-good frame + its manifest ({} when absent)."""
    try:
        # float_precision="round_trip": the default C parser is ~1 ULP off on
        # some floats, which would re-serialize differently and break the
        # manifest checksum ↔ CSV integrity match on a byte-identical file
        df = pd.read_csv(spec.snapshot_csv, index_col=0, parse_dates=True,
                         float_precision="round_trip")
        df = _norm_index(df)
    except Exception:
        return None, {}
    try:
        man = json.loads(Path(spec.manifest_json).read_text())
    except Exception:
        man = {}
    return (df if len(df) else None), (man if isinstance(man, dict) else {})


def save_snapshot(spec: GateSpec, df: pd.DataFrame, report: dict,
                  source: str = "yahoo-live") -> dict:
    """Persist a validated completed-bars frame + manifest. Best-effort."""
    man = dict(
        key=spec.key, source=source,
        saved_at_utc=_fr.now_utc().isoformat(timespec="seconds"),
        rows=int(len(df)),
        date_from=str(pd.Timestamp(df.index.min()).date()),
        date_to=str(pd.Timestamp(df.index.max()).date()),
        columns=[str(c) for c in df.columns],
        checksum_sha256=frame_sha(df),
        qc_passed=bool(report.get("passed")),
        qc_checks=report.get("checks", []),
    )
    try:
        spec.snapshot_csv.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(spec.snapshot_csv, df.to_csv())
        _atomic_write(Path(spec.manifest_json), json.dumps(man, indent=1))
    except Exception:
        pass
    return man


def read_manifest(spec_or_path) -> dict:
    p = (Path(spec_or_path.manifest_json)
         if isinstance(spec_or_path, GateSpec) else Path(spec_or_path))
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


# ════════════════════════════════════════════════════════════════════════
# Audit trail
# ════════════════════════════════════════════════════════════════════════
def _record_audit(spec: GateSpec, decision: str, source: str, consumer: str,
                  df: pd.DataFrame | None, report: dict | None,
                  note: str = "", partial_rows: int = 0) -> None:
    entry = dict(
        ts_utc=_fr.now_utc().isoformat(timespec="seconds"),
        ts_ct=_fr.fmt_ct(_fr.now_utc(), seconds=True),
        decision=decision, source=source, consumer=consumer,
        rows=int(len(df)) if df is not None else 0,
        date_from=(str(pd.Timestamp(df.index.min()).date())
                   if df is not None and len(df) else "—"),
        date_to=(str(pd.Timestamp(df.index.max()).date())
                 if df is not None and len(df) else "—"),
        sha256=(frame_sha(df) if df is not None and len(df) else "—"),
        qc_passed=bool(report.get("passed")) if report else None,
        failed_checks=(report or {}).get("failed", []),
        partial_bar_rows=int(partial_rows),
        note=note,
        snapshot_file=str(spec.snapshot_csv.relative_to(_REPO_ROOT))
        if str(spec.snapshot_csv).startswith(str(_REPO_ROOT))
        else str(spec.snapshot_csv),
    )
    try:
        with _AUDIT_LOCK:
            log = read_audit()
            rows = log.get(spec.key, [])
            rows.insert(0, entry)
            log[spec.key] = rows[:_MAX_AUDIT_ENTRIES]
            RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
            _atomic_write(AUDIT_JSON, json.dumps(log, indent=1, default=str))
    except Exception:
        pass


def read_audit() -> dict:
    """{dataset key: [entries, newest first]} — best-effort, {} on any error."""
    try:
        d = json.loads(AUDIT_JSON.read_text())
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


# ════════════════════════════════════════════════════════════════════════
# The gate
# ════════════════════════════════════════════════════════════════════════
def _append_partial_bar(hist: pd.DataFrame, recent: pd.DataFrame | None,
                        cut: pd.Timestamp, price_col: str) -> tuple[pd.DataFrame, int]:
    """Overlay any in-progress rows (session date > ``cut``) from a small
    recent fetch on top of the pinned completed history.  Guarded by the same
    60% jump check as the cross-provider merge; never persisted."""
    if recent is None or len(recent) == 0:
        return hist, 0
    try:
        r = _norm_index(recent)
        rows = r[r.index > cut]
        if rows.empty or price_col not in rows.columns:
            return hist, 0
        rows = rows.dropna(subset=[price_col])
        base = pd.to_numeric(hist.get(price_col), errors="coerce").dropna()
        if len(base) and float(base.iloc[-1]) > 0:
            keep = (rows[price_col].astype(float)
                    / float(base.iloc[-1]) - 1.0).abs() <= MAX_ABS_TRADED_MOVE
            rows = rows[keep]
        if rows.empty:
            return hist, 0
        out = pd.concat([hist, rows.reindex(columns=hist.columns)]).sort_index()
        out = out[~out.index.duplicated(keep="last")]
        # forward-fill macro columns onto the overlay row (mirrors fetch_daily)
        macro_like = [c for c in out.columns
                      if not str(c).startswith(price_col.split("_", 1)[0] + "_")]
        if macro_like:
            out[macro_like] = out[macro_like].ffill(limit=5)
        return out, int(len(rows))
    except Exception:
        return hist, 0


def _finish(spec: GateSpec, df: pd.DataFrame, decision: str, source: str,
            report: dict | None, cut) -> pd.DataFrame:
    df.attrs["data_gate"] = dict(
        key=spec.key, decision=decision, source=source,
        qc_passed=bool(report.get("passed")) if report else None,
        failed_checks=(report or {}).get("failed", []),
        completed_through=str(pd.Timestamp(cut).date()),
        sha256=frame_sha(df) if len(df) else "—",
    )
    return df


def gated_daily(spec: GateSpec, fetch_full, fetch_recent=None,
                consumer: str = "") -> pd.DataFrame:
    """Quality-gated, snapshot-pinned daily history for one sleeve.

    * ``fetch_full``   — zero-arg callable returning the full live daily frame
      (e.g. ``lambda: ticker_core.fetch_daily(cfg)``).
    * ``fetch_recent`` — optional zero-arg callable returning a small recent
      daily frame (``range=5d``) used only to overlay the in-progress bar on
      the pinned history; skipped entirely in publisher mode.
    * ``consumer``     — who is loading (an app key or "OVERALL"), recorded in
      the audit trail so each dataset's usage is traceable.
    """
    basis_now = _fr.publish_anchor_ct() if _fr.completed_bars_only() else None
    cut = _fr.expected_equity_asof(basis_now)
    snap, man = load_snapshot(spec)
    # the pinned fast path additionally requires the manifest's content hash to
    # match the CSV on disk — a snapshot rewritten by anything other than
    # save_snapshot (e.g. a CLI training run) must be re-validated, not trusted
    snap_ok = (snap is not None and man.get("qc_passed")
               and man.get("checksum_sha256") == frame_sha(snap)
               and not [c for c in spec.required_cols
                        if c not in snap.columns or not snap[c].notna().any()])
    snap_last = _fr.last_completed_session(snap.index, basis_now) \
        if snap_ok else None

    # ── fast path: the pinned snapshot already covers the last completed
    # session — reuse it byte-for-byte, no full-history refetch ─────────────
    if snap_ok and snap_last is not None and snap_last >= cut:
        out, n_part = snap, 0
        if fetch_recent is not None and not _fr.completed_bars_only():
            try:
                out, n_part = _append_partial_bar(snap, fetch_recent(), cut,
                                                  spec.price_col)
            except Exception:
                out, n_part = snap, 0
        _record_audit(spec, "pinned", man.get("source", "snapshot"), consumer,
                      snap, dict(passed=True, failed=[]),
                      note="snapshot already covers the last completed session",
                      partial_rows=n_part)
        return _finish(spec, out, "pinned", man.get("source", "snapshot"),
                       dict(passed=True, failed=[]), cut)

    # ── refresh path: fetch live, validate, pin on success ────────────────
    report: dict | None = None
    fresh = None
    for _attempt in (1, 2):
        try:
            fresh = fetch_full()
        except Exception:
            fresh = None
        if fresh is None or len(fresh) == 0:
            report = dict(passed=False, failed=["non_empty"],
                          checks=[dict(name="non_empty", passed=False,
                                       detail="fetch returned no rows")])
            continue
        fresh = _norm_index(fresh)
        hist = _fr.drop_in_progress_us_bar(fresh, basis_now)
        report = run_quality_checks(spec, hist, snapshot=snap)
        if report["passed"]:
            save_snapshot(spec, hist, report)
            out = hist
            n_part = 0
            if not _fr.completed_bars_only():
                partial = fresh[fresh.index > cut]
                if len(partial):
                    out = pd.concat([hist, partial]).sort_index()
                    out = out[~out.index.duplicated(keep="last")]
                    n_part = int(len(partial))
            _record_audit(spec, "refreshed", "yahoo-live", consumer, hist,
                          report, partial_rows=n_part)
            return _finish(spec, out, "refreshed", "yahoo-live", report, cut)

    # ── fetch rejected: serve the last known-good snapshot ────────────────
    if snap is not None:
        _record_audit(spec, "fallback_snapshot", man.get("source", "snapshot"),
                      consumer, snap, report,
                      note="live fetch failed quality control; serving the "
                           "last known-good snapshot instead")
        return _finish(spec, snap, "fallback_snapshot",
                       man.get("source", "snapshot"), report, cut)

    # ── no snapshot exists yet: never brick the app — serve the fetch as-is,
    # loudly flagged as unvalidated (the audit + UI make this visible) ──────
    out = fresh if fresh is not None else pd.DataFrame()
    _record_audit(spec, "live_unvalidated", "yahoo-live", consumer,
                  out if len(out) else None, report,
                  note="no known-good snapshot exists; serving the live fetch "
                       "despite failed quality control")
    return _finish(spec, out, "live_unvalidated", "yahoo-live", report, cut)
