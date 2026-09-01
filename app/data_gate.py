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
   snapshot, no completed session the snapshot already carries silently
   dropped, and per-day return agreement with the snapshot on the
   overlapping history).  A failed fetch is retried once, then REJECTED.
3. **Fall back** — a rejected fetch never reaches the models: the last
   known-good snapshot is served instead, and the rejection is recorded.
   A sleeve therefore stops advancing while the provider keeps serving a
   corrupt vintage — deliberately: a frozen dataset shows up as a stale app
   in the daily audit, whereas a silently altered one changes the signals.
   Should a session ever be *legitimately* withdrawn upstream (so the
   dropped-session check can never pass again), re-pin from scratch by
   deleting that sleeve's snapshot CSV + manifest; with no snapshot there is
   nothing to regress against and the next clean fetch becomes the baseline.
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

# Dropped-session guard: a completed session the snapshot already carries must
# still be there on the next fetch.  Yahoo intermittently serves a frame with a
# single recent session MISSING while back-filling older gaps in the same
# response, so the row count can RISE while recent history loses a bar — every
# other check passes (the total is bigger, the last date is newer, and the
# overlap comparison only ever looks at dates present in BOTH frames, so a hole
# is invisible to it).  That is not cosmetic: the signature engines read the
# last ~150 completed bars positionally, and one absent bar re-links a
# consecutive-break streak — e.g. ARTY 2026-08-14, where dropping 2026-08-11
# moved the high-break streak from 3 to 2 and flipped D3 exhaustion (the
# published book booked "EXIT NEXT BAR — D3 exhaustion" on a bar whose pinned
# vintage no longer carries it).  The window covers the whole signature
# lookback; older provider corrections stay the overlap check's business.
_RECENT_SESSION_DAYS = 150       # snapshot sessions that must not disappear

# Independent cross-check: a session being added to the snapshot for the first
# time has no prior reference to agree with, so its traded closes are verified
# against Nasdaq's independent tape (app/market_fallback.py) where served.
# Empirically the two providers agree to fractions of a basis point (see
# DATA_CONSISTENCY.md), so 50 bps is a generous bound that still catches any
# materially wrong day-one print.  Nasdaq is raw (split-unadjusted); the
# comparison first aligns bases via the median close ratio over the sessions
# just before the new ones, so a recent split cannot masquerade as an error.
NASDAQ_XCHECK_TOL = 0.005        # 0.5% relative close disagreement → reject
_XCHECK_MAX_SESSIONS = 5         # newest new sessions verified per refresh
_XCHECK_REF_DAYS = 10            # prior overlap sessions used to align bases


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
    # close column → listed ticker, for the independent Nasdaq cross-check of
    # newly-added sessions; empty dict disables the check for this dataset.
    symbol_by_col: dict = field(default_factory=dict)

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
    syms = {"px_close": cfg.primary_symbol}
    syms.update({f"{nm}_close": s for nm, s in cfg.extra_syms.items()})
    return GateSpec(key=cfg.key, price_col="px_close",
                    traded_close_cols=traded, macro_close_cols=macro,
                    snapshot_csv=tc.cache_paths(cfg)["daily"],
                    symbol_by_col=syms)


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
        # a session inside the signature lookback can never legitimately vanish
        # from a later fetch — see _RECENT_SESSION_DAYS.  Compared only over the
        # span the fetch actually covers, so a short/backdated frame is left to
        # the span/row regression checks above rather than double-reported.
        recent = pd.DatetimeIndex(snapshot.index)[-_RECENT_SESSION_DAYS:]
        recent = recent[recent <= df.index.max()]
        dropped = recent.difference(pd.DatetimeIndex(df.index))
        shown = ", ".join(str(pd.Timestamp(d).date()) for d in dropped[:5])
        add("no_missing_recent_sessions", len(dropped) == 0,
            f"all {len(recent)} recent snapshot sessions still served"
            if len(dropped) == 0 else
            f"{len(dropped)} session(s) the snapshot carries are missing from "
            f"the fetch: {shown}{' …' if len(dropped) > 5 else ''}")
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
# Independent cross-check of newly-added sessions (Nasdaq's free tape)
# ════════════════════════════════════════════════════════════════════════
def _nasdaq_crosscheck(spec: GateSpec, hist: pd.DataFrame,
                       prev: pd.DataFrame | None) -> dict | None:
    """Verify the traded closes of sessions being ADDED to the snapshot
    against Nasdaq's independent quote API — the one moment a print has no
    prior reference to agree with (`history_agreement` covers everything
    else from the next refresh onward).

    Best-effort by design: returns a check dict only when at least one
    (column, session) pair could actually be verified; returns ``None`` on
    bootstrap (no previous snapshot), nothing new, no symbol map, or when
    Nasdaq serves nothing — an unreachable second opinion must never block
    a refresh, it just isn't evidence either way (the skip is recorded in
    the check detail when a partial check ran)."""
    if not spec.symbol_by_col or prev is None or not len(prev):
        return None
    try:
        import market_fallback as mf
    except Exception:
        return None
    prev_last = pd.Timestamp(prev.index.max())
    new_idx = hist.index[hist.index > prev_last]
    if not len(new_idx):
        return None
    new_idx = new_idx[-_XCHECK_MAX_SESSIONS:]
    start = (pd.Timestamp(new_idx.min())
             - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    checked, skipped, bad = 0, 0, []
    for col, sym in spec.symbol_by_col.items():
        if col not in hist.columns or not mf.supports(sym):
            skipped += len(new_idx)
            continue
        try:
            nq = mf.daily_ohlcv(sym, start)
        except Exception:
            nq = pd.DataFrame()
        if nq is None or nq.empty:
            skipped += len(new_idx)
            continue
        nq = nq.copy()
        nq.index = pd.DatetimeIndex(nq.index).normalize()
        y = pd.to_numeric(hist[col], errors="coerce")
        # align adjustment bases (Nasdaq is raw, Yahoo split-adjusted) on the
        # sessions just before the new ones — a recent split shows up as a
        # constant ratio there and is divided out, not flagged
        ref = nq.index.intersection(y.dropna().index)
        ref = ref[ref < new_idx.min()][-_XCHECK_REF_DAYS:]
        if len(ref) < 3:
            skipped += len(new_idx)
            continue
        scale = float((y.loc[ref] / nq.loc[ref, "close"]).median())
        if not np.isfinite(scale) or scale <= 0:
            skipped += len(new_idx)
            continue
        for d in new_idx:
            yd = y.get(d)
            if d not in nq.index or yd is None or pd.isna(yd) or yd <= 0:
                skipped += 1
                continue
            rel = abs(float(yd) - float(nq.loc[d, "close"]) * scale) / float(yd)
            checked += 1
            if rel > NASDAQ_XCHECK_TOL:
                bad.append(f"{col}@{pd.Timestamp(d).date()} "
                           f"({rel * 100:.2f}% vs Nasdaq)")
    if checked == 0:
        return None                       # no independent evidence available
    detail = (f"{checked} new-session close(s) verified against Nasdaq"
              + (f", {skipped} skipped (not served)" if skipped else ""))
    if bad:
        detail = f"disagrees with Nasdaq beyond {NASDAQ_XCHECK_TOL*100:.1f}%: {bad}"
    return dict(name="nasdaq_crosscheck", passed=not bad, detail=detail)


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


def _restore_from_snapshot(hist: pd.DataFrame, snap: pd.DataFrame | None,
                           cut, price_col: str) -> tuple[pd.DataFrame, list]:
    """Prefer the pinned snapshot's own values wherever this fetch lost or only
    RECONSTRUCTED a session.  Returns the frame plus the sessions it restored.

    Two cases, both about the same failure — a provider WITHDRAWING a close it
    already published (Yahoo began serving a null for the 2026-08-28 bar of nine
    tickers on 2026-09-01, and `_chart` drops null-close rows):

    1. The session is missing from the fetch outright.  Without this the fetch
       keeps failing ``no_missing_recent_sessions`` forever and the sleeve is
       frozen on that same snapshot, one session behind, with every retry
       hitting the identical hole — no path back without a human.
    2. The session was rebuilt upstream by ``freshness.repair_daily_frame`` (it
       stamps ``repaired_sessions`` / ``repaired_cells``).  That is real data,
       but an hourly aggregate closes on the last intraday bar rather than the
       official 4:00-PM-ET print (~0.1% apart) and a second provider carries its
       own adjustment basis.  The snapshot holds the OFFICIAL close, pinned when
       the feed still served it and cross-checked against Nasdaq's independent
       tape at that moment — so it wins, and a repair can never silently restate
       history this module has already pinned.

    Strictly bounded: only sessions inside the guard window, only at or before
    the last completed close, and never past the span the fetch itself covers —
    so a genuinely truncated or backdated fetch is still caught by the span/row
    regression checks instead of being papered over.

    NOTE the deliberate asymmetry with the "re-pin from scratch" escape hatch in
    this module's docstring: if a session is *legitimately* withdrawn upstream,
    this keeps restoring it, and deleting the snapshot CSV + manifest remains the
    way to force a clean re-baseline.  That is the safe default — a session
    vanishing from a feed is overwhelmingly a feed defect, not a correction, and
    silently dropping it re-links the signature engines' consecutive-bar streaks
    (see _RECENT_SESSION_DAYS).
    """
    if (hist is None or not len(hist) or snap is None or not len(snap)
            or price_col not in hist.columns or price_col not in snap.columns):
        return hist, []
    # read the repair stamp BEFORE any concat below can drop frame attrs
    rebuilt = set(hist.attrs.get("repaired_sessions") or [])
    rebuilt_cells = dict(hist.attrs.get("repaired_cells") or {})

    window = pd.DatetimeIndex(snap.index)[-_RECENT_SESSION_DAYS:]
    window = window[(window <= pd.Timestamp(cut))
                    & (window <= pd.Timestamp(hist.index.max()))]
    absent = window.difference(pd.DatetimeIndex(hist.index))
    reconstructed = pd.DatetimeIndex(
        [d for d in window if str(pd.Timestamp(d).date()) in rebuilt])

    out, restored = hist, []
    take = absent.union(reconstructed)
    if len(take):
        rows = snap.loc[take].reindex(columns=hist.columns).dropna(
            subset=[price_col])
        if len(rows):
            out = pd.concat([hist.drop(index=rows.index, errors="ignore"), rows])
            out = out.sort_index()
            out = out[~out.index.duplicated(keep="last")]
            restored = [str(pd.Timestamp(d).date()) for d in rows.index]

    # sibling closes rebuilt cell-by-cell on a session the frame already had
    for col, days in rebuilt_cells.items():
        if col not in snap.columns or col not in out.columns:
            continue
        stem = str(col).rsplit("_", 1)[0]
        for ds in days:
            d = pd.Timestamp(ds)
            if d not in out.index or d not in snap.index or d > pd.Timestamp(cut):
                continue
            for suffix in ("open", "high", "low", "close", "volume"):
                c2 = f"{stem}_{suffix}"
                if c2 in out.columns and c2 in snap.columns and pd.notna(
                        snap.at[d, c2]):
                    out.at[d, c2] = snap.at[d, c2]
            if ds not in restored:
                restored.append(ds)

    return out, sorted(restored)


def _finish(spec: GateSpec, df: pd.DataFrame, decision: str, source: str,
            report: dict | None, cut, spliced: list | None = None) -> pd.DataFrame:
    df.attrs["data_gate"] = dict(
        key=spec.key, decision=decision, source=source,
        qc_passed=bool(report.get("passed")) if report else None,
        failed_checks=(report or {}).get("failed", []),
        completed_through=str(pd.Timestamp(cut).date()),
        spliced_sessions=list(spliced or []),
        sha256=frame_sha(df) if len(df) else "—",
    )
    return df


def gated_daily(spec: GateSpec, fetch_full, fetch_recent=None,
                consumer: str = "", read_only: bool = False) -> pd.DataFrame:
    """Quality-gated, snapshot-pinned daily history for one sleeve.

    * ``fetch_full``   — zero-arg callable returning the full live daily frame
      (e.g. ``lambda: ticker_core.fetch_daily(cfg)``).
    * ``fetch_recent`` — optional zero-arg callable returning a small recent
      daily frame (``range=5d``) used only to overlay the in-progress bar on
      the pinned history; skipped entirely in publisher mode.
    * ``consumer``     — who is loading (an app key or "OVERALL"), recorded in
      the audit trail so each dataset's usage is traceable.
    * ``read_only``    — do not re-pin: a passing fetch is served but NOT written
      to the snapshot.  Pinning is a side effect of merely LOADING a sleeve, so
      any diagnostic, backtest or CLI run that touches an engine silently
      re-baselines the committed vintage from whatever the feed happened to
      return on that machine.  Callers that are not the app or the publisher
      should pass ``read_only=True``.
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
    spliced: list = []
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
        # A completed session the feed has since WITHDRAWN is taken back from the
        # pinned snapshot before QC judges the frame — both when the fetch lost
        # it outright (else the sleeve freezes forever) and when it was rebuilt
        # upstream from hourly/second-provider data (the pinned OFFICIAL close
        # wins, so a repair never restates already-pinned history).
        hist, spliced = _restore_from_snapshot(hist, snap, cut, spec.price_col)
        report = run_quality_checks(spec, hist, snapshot=snap)
        if report["passed"]:
            # sessions entering the snapshot for the first time get a second
            # opinion from Nasdaq's independent tape (best-effort; a failed
            # verification rejects the fetch like any other check)
            try:
                xc = _nasdaq_crosscheck(spec, hist, snap)
            except Exception:
                xc = None
            if xc is not None:
                report["checks"].append(xc)
                if not xc["passed"]:
                    report["failed"] = list(report["failed"]) + [xc["name"]]
                    report["passed"] = False
                    continue                    # retry once, then fall back
            if not read_only:
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
                          report, partial_rows=n_part,
                          note=("restored from the pinned snapshot: "
                                + ", ".join(spliced)) if spliced else "")
            return _finish(spec, out, "refreshed", "yahoo-live", report, cut,
                           spliced)

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
