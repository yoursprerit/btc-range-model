"""Unit tests for app/data_gate.py — the data-quality gate, pinned snapshots
and the dataset audit trail.

No network, no Streamlit: fetches are stubbed with synthetic frames whose
dates are anchored to the REAL ``freshness.expected_equity_asof()`` clock, so
the completed/in-progress split behaves exactly as in production.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import freshness as fr    # noqa: E402
import data_gate as dg    # noqa: E402

CUT = fr.expected_equity_asof()          # last completed US session, real clock


def _frame(n=320, end=CUT, seed=7, cols_extra=("soxl_close", "vix_close")):
    """Synthetic completed-bars daily frame ending at ``end``."""
    idx = pd.bdate_range(end=end, periods=n)
    rng = np.random.default_rng(seed)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df = pd.DataFrame(index=idx)
    df["px_open"] = px * (1 + rng.normal(0, 0.002, n))
    df["px_high"] = px * 1.01
    df["px_low"] = px * 0.99
    df["px_close"] = px
    df["px_volume"] = 1e6
    for c in cols_extra:
        df[c] = px * (2.0 if c.startswith("soxl") else 0.2)
    return df


def _spec(tmp_path, key="TEST"):
    return dg.GateSpec(
        key=key, price_col="px_close",
        traded_close_cols=["px_open", "px_high", "px_low", "px_close",
                           "soxl_close"],
        macro_close_cols=["vix_close"],
        snapshot_csv=tmp_path / "macro_daily.csv")


@pytest.fixture(autouse=True)
def _isolated_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(dg, "AUDIT_JSON", tmp_path / "dataset_audit.json")
    monkeypatch.setattr(dg, "RUNTIME_DIR", tmp_path)
    yield


# ── quality checks ───────────────────────────────────────────────────────────
def test_checks_pass_on_good_frame(tmp_path):
    rep = dg.run_quality_checks(_spec(tmp_path), _frame())
    assert rep["passed"], rep["failed"]


def test_checks_fail_on_empty_and_short(tmp_path):
    spec = _spec(tmp_path)
    assert not dg.run_quality_checks(spec, None)["passed"]
    rep = dg.run_quality_checks(spec, _frame(n=50))
    assert "min_rows" in rep["failed"]


def test_checks_fail_on_missing_macro_column(tmp_path):
    df = _frame().drop(columns=["vix_close"])
    rep = dg.run_quality_checks(_spec(tmp_path), df)
    assert "required_columns" in rep["failed"]


def test_checks_fail_on_all_nan_column(tmp_path):
    df = _frame()
    df["vix_close"] = np.nan            # present but silently dead feed
    rep = dg.run_quality_checks(_spec(tmp_path), df)
    assert "required_columns" in rep["failed"]


def test_checks_fail_on_extreme_traded_move(tmp_path):
    df = _frame()
    df.iloc[-5, df.columns.get_loc("soxl_close")] *= 3.0   # +200% one-day print
    rep = dg.run_quality_checks(_spec(tmp_path), df)
    assert "no_extreme_traded_moves" in rep["failed"]


def test_checks_fail_on_regression_vs_snapshot(tmp_path):
    spec = _spec(tmp_path)
    snap = _frame()
    shrunk = snap.iloc[:200]
    rep = dg.run_quality_checks(spec, shrunk, snapshot=snap)
    assert "no_row_regression" in rep["failed"]
    assert "no_span_regression" in rep["failed"]
    # min_rows also fires at 200 < 260 — but the regression checks are the point


def test_checks_fail_on_dropped_recent_session(tmp_path):
    """A completed session the snapshot carries must not vanish from a later
    fetch — the ARTY 2026-08-11 case.  Yahoo served a frame that back-filled
    two older gaps while dropping one recent session, so the row count ROSE and
    every other check passed; positionally-read signature math then re-linked a
    high-break streak and fired a phantom D3 exhaustion exit."""
    spec = _spec(tmp_path)
    base = _frame(n=322)
    snap = base.iloc[2:]                          # 320 rows, ends at CUT
    fresh = base.drop(base.index[-4])             # 321: +2 old, −1 recent
    rep = dg.run_quality_checks(spec, fresh, snapshot=snap)
    assert "no_missing_recent_sessions" in rep["failed"]
    # …and it is the ONLY check that catches it: this is the blind spot
    assert set(rep["failed"]) == {"no_missing_recent_sessions"}
    detail = next(c["detail"] for c in rep["checks"]
                  if c["name"] == "no_missing_recent_sessions")
    assert str(pd.Timestamp(base.index[-4]).date()) in detail


def test_dropped_session_check_passes_on_growing_history(tmp_path):
    """The normal case must not trip it: a fetch that keeps every snapshot
    session and adds a new one."""
    spec = _spec(tmp_path)
    base = _frame(n=321)
    rep = dg.run_quality_checks(spec, base, snapshot=base.iloc[1:])
    assert rep["passed"], rep["failed"]


def _aged_snapshot(spec, base):
    """Pin ``base`` then roll the snapshot back two sessions, so the next
    ``gated_daily`` MUST take the refresh path rather than the pinned fast path."""
    dg.gated_daily(spec, fetch_full=lambda: base.iloc[2:], consumer="UNIT")
    man_p = Path(spec.manifest_json)
    man = json.loads(man_p.read_text())
    stale = dg._norm_index(pd.read_csv(spec.snapshot_csv, index_col=0,
                                       parse_dates=True)).iloc[:-2]
    dg._atomic_write(spec.snapshot_csv, stale.to_csv())
    man["checksum_sha256"] = dg.frame_sha(stale)
    man_p.write_text(json.dumps(man))
    return stale


def test_gate_restores_a_session_the_feed_withdrew(tmp_path):
    """Regression for 2026-09-01: Yahoo started serving a NULL close for the
    already-published 2026-08-28 bar of nine tickers, `_chart` dropped the row,
    and `no_missing_recent_sessions` then refused every fetch — freezing seven
    sleeves on their pinned snapshot and withholding the target book with no
    path back, because each retry hit the identical hole.

    The withdrawn session must be restored from the snapshot (which still holds
    it, Nasdaq-verified from when it was added) so the frame the models see has
    NO hole and the sleeve keeps advancing."""
    spec = _spec(tmp_path)
    base = _frame(n=322)
    stale = _aged_snapshot(spec, base)

    dropped_date = stale.index[-4]
    out = dg.gated_daily(spec, fetch_full=lambda: base.drop(dropped_date),
                         consumer="UNIT")
    gi = out.attrs["data_gate"]
    assert gi["decision"] == "refreshed"
    assert gi["failed_checks"] == []
    assert str(dropped_date.date()) in gi["spliced_sessions"]
    # restored from the snapshot byte-for-byte, not re-derived or interpolated
    assert dropped_date in out.index
    assert out.loc[dropped_date, "px_close"] == stale.loc[dropped_date, "px_close"]
    # and the sleeve actually moves forward instead of freezing
    assert out.index.max() > stale.index.max()


def test_splice_does_not_mask_a_truncated_fetch(tmp_path):
    """The splice is a repair, not a blanket pass: a fetch that is genuinely
    short/backdated must still be rejected and the snapshot served."""
    spec = _spec(tmp_path)
    base = _frame(n=322)
    _aged_snapshot(spec, base)
    out = dg.gated_daily(spec, fetch_full=lambda: base.iloc[:-40], consumer="UNIT")
    gi = out.attrs["data_gate"]
    assert gi["decision"] == "fallback_snapshot"
    assert "no_span_regression" in gi["failed_checks"]


def test_pinned_official_close_beats_an_upstream_repair(tmp_path):
    """A session the fetcher REBUILT (from hourly bars or a second provider) is
    real data, but not the official 4:00-PM-ET print — an hourly aggregate lands
    ~0.1% off it and a second provider carries its own adjustment basis.  When
    the snapshot already holds that session's official close, the pinned value
    must win, so a repair can never silently restate pinned history.

    freshness.repair_daily_frame stamps what it rebuilt; the gate reads it.
    """
    spec = _spec(tmp_path)
    base = _frame(n=322)
    stale = _aged_snapshot(spec, base)
    session = stale.index[-4]
    official = float(stale.loc[session, "px_close"])

    rebuilt = base.copy()
    rebuilt.loc[session, "px_close"] = official * 1.001      # hourly-ish drift
    rebuilt.attrs["repaired_sessions"] = [str(session.date())]

    out = dg.gated_daily(spec, fetch_full=lambda: rebuilt, consumer="UNIT",
                         read_only=True)
    gi = out.attrs["data_gate"]
    assert gi["decision"] == "refreshed"
    assert str(session.date()) in gi["spliced_sessions"]
    assert float(out.loc[session, "px_close"]) == official   # pinned value wins
    # sessions it did NOT rebuild are left exactly as the feed served them
    untouched = base.index[-1]
    assert float(out.loc[untouched, "px_close"]) == float(base.loc[untouched, "px_close"])


def test_read_only_never_repins_the_snapshot(tmp_path):
    """Pinning is a side effect of merely LOADING a sleeve, so a diagnostic or
    backtest run used to re-baseline the committed vintage from whatever the
    feed returned on that machine.  read_only=True must serve the fetch without
    writing it."""
    spec = _spec(tmp_path)
    base = _frame(n=322)
    stale = _aged_snapshot(spec, base)
    before = dg.frame_sha(dg.load_snapshot(spec)[0])
    out = dg.gated_daily(spec, fetch_full=lambda: base, consumer="UNIT",
                         read_only=True)
    assert out.attrs["data_gate"]["decision"] == "refreshed"
    assert out.index.max() > stale.index.max()          # caller still gets it
    assert dg.frame_sha(dg.load_snapshot(spec)[0]) == before   # but nothing pinned


def test_checks_fail_on_rewritten_history(tmp_path):
    spec = _spec(tmp_path)
    snap = _frame()
    rewritten = snap.copy()
    wob = 1 + 0.01 * np.sin(np.arange(60))          # jitters 60 recent returns
    rewritten.iloc[-60:, rewritten.columns.get_loc("px_close")] *= wob
    rep = dg.run_quality_checks(spec, rewritten, snapshot=snap)
    assert "history_agreement" in rep["failed"]


# ── the gate: refresh → pin → fallback ──────────────────────────────────────
def test_refresh_pins_snapshot_and_manifest(tmp_path):
    spec = _spec(tmp_path)
    good = _frame()
    out = dg.gated_daily(spec, fetch_full=lambda: good, consumer="UNIT")
    assert out.attrs["data_gate"]["decision"] == "refreshed"
    assert spec.snapshot_csv.exists() and Path(spec.manifest_json).exists()
    man = json.loads(Path(spec.manifest_json).read_text())
    assert man["qc_passed"] is True
    snap, _ = dg.load_snapshot(spec)
    assert man["checksum_sha256"] == dg.frame_sha(snap)
    assert pd.Timestamp(snap.index.max()) <= CUT     # completed bars only


def test_pinned_path_never_refetches(tmp_path):
    spec = _spec(tmp_path)
    dg.gated_daily(spec, fetch_full=lambda: _frame(), consumer="UNIT")

    def _boom():
        raise AssertionError("full fetch must not run on the pinned path")

    out = dg.gated_daily(spec, fetch_full=_boom, consumer="UNIT")
    assert out.attrs["data_gate"]["decision"] == "pinned"
    assert pd.Timestamp(out.index.max()) == pd.Timestamp(
        fr.last_completed_session(out.index))


def test_pinned_path_is_byte_identical(tmp_path):
    spec = _spec(tmp_path)
    a = dg.gated_daily(spec, fetch_full=lambda: _frame(), consumer="UNIT")
    b = dg.gated_daily(spec, fetch_full=lambda: _frame(seed=99), consumer="X")
    pd.testing.assert_frame_equal(a, b)              # seed-99 fetch never ran


def test_in_progress_bar_split_from_history(tmp_path):
    spec = _spec(tmp_path)
    full = _frame()
    partial_date = CUT + pd.Timedelta(days=1)
    partial = full.iloc[[-1]].copy()
    partial.index = [partial_date]
    with_partial = pd.concat([full, partial])
    out = dg.gated_daily(spec, fetch_full=lambda: with_partial, consumer="U")
    # served frame keeps the live in-progress row …
    assert pd.Timestamp(out.index.max()) == partial_date
    # … but the pinned snapshot must not contain it
    snap = pd.read_csv(spec.snapshot_csv, index_col=0, parse_dates=True)
    assert pd.Timestamp(snap.index.max()) <= CUT


def test_partial_overlay_on_pinned_snapshot(tmp_path):
    spec = _spec(tmp_path)
    good = _frame()
    dg.gated_daily(spec, fetch_full=lambda: good, consumer="UNIT")
    recent = good.iloc[[-1]].copy()
    recent.index = [CUT + pd.Timedelta(days=1)]
    out = dg.gated_daily(spec, fetch_full=lambda: _frame(seed=1),
                         fetch_recent=lambda: recent, consumer="UNIT")
    assert out.attrs["data_gate"]["decision"] == "pinned"
    assert pd.Timestamp(out.index.max()) == CUT + pd.Timedelta(days=1)
    snap = pd.read_csv(spec.snapshot_csv, index_col=0, parse_dates=True)
    assert pd.Timestamp(snap.index.max()) <= CUT     # overlay never persisted


def test_rejected_fetch_falls_back_to_snapshot(tmp_path):
    spec = _spec(tmp_path)
    good = _frame()
    dg.gated_daily(spec, fetch_full=lambda: good, consumer="UNIT")
    # invalidate the manifest freshness so the gate MUST try the fetch again
    man_p = Path(spec.manifest_json)
    man = json.loads(man_p.read_text())
    snap = pd.read_csv(spec.snapshot_csv, index_col=0, parse_dates=True)
    old = dg._norm_index(snap).iloc[:-2]             # snapshot now 2 bars stale
    dg._atomic_write(spec.snapshot_csv, old.to_csv())
    man["checksum_sha256"] = dg.frame_sha(old)
    man_p.write_text(json.dumps(man))

    corrupt = _frame().drop(columns=["vix_close"])   # fails required_columns
    out = dg.gated_daily(spec, fetch_full=lambda: corrupt, consumer="UNIT")
    gi = out.attrs["data_gate"]
    assert gi["decision"] == "fallback_snapshot"
    assert "required_columns" in gi["failed_checks"]
    assert "vix_close" in out.columns                # the snapshot, not the fetch


def test_unvalidated_live_when_no_snapshot(tmp_path):
    spec = _spec(tmp_path)
    corrupt = _frame().drop(columns=["vix_close"])
    out = dg.gated_daily(spec, fetch_full=lambda: corrupt, consumer="UNIT")
    assert out.attrs["data_gate"]["decision"] == "live_unvalidated"
    assert len(out)                                   # still served — flagged


def test_audit_trail_records_every_decision(tmp_path):
    spec = _spec(tmp_path)
    dg.gated_daily(spec, fetch_full=lambda: _frame(), consumer="OVERALL")
    dg.gated_daily(spec, fetch_full=lambda: _frame(), consumer="SOXX")
    trail = dg.read_audit()
    assert spec.key in trail
    decisions = [e["decision"] for e in trail[spec.key]]
    assert decisions[0] == "pinned" and decisions[1] == "refreshed"
    assert {e["consumer"] for e in trail[spec.key]} == {"OVERALL", "SOXX"}
    for e in trail[spec.key]:
        assert e["sha256"] != "—" and e["rows"] > 0


# ── Nasdaq cross-check of newly-added sessions ──────────────────────────────
def _make_stale_snapshot(spec, nbars=2):
    """Pin a good snapshot, then trim its newest ``nbars`` sessions so the next
    gated_daily call MUST take the refresh path (and cross-check the re-added
    sessions)."""
    good = _frame()
    dg.gated_daily(spec, fetch_full=lambda: good, consumer="UNIT")
    man_p = Path(spec.manifest_json)
    man = json.loads(man_p.read_text())
    snap, _ = dg.load_snapshot(spec)
    old = snap.iloc[:-nbars]
    dg._atomic_write(spec.snapshot_csv, old.to_csv())
    man["checksum_sha256"] = dg.frame_sha(old)
    man_p.write_text(json.dumps(man))
    return good, old


def _fake_nasdaq(close: pd.Series):
    """market_fallback.daily_ohlcv stand-in serving the given close series."""
    def _daily_ohlcv(symbol, start, end=None):
        s = close[close.index >= pd.Timestamp(start)]
        return pd.DataFrame({"open": s, "high": s, "low": s,
                             "close": s, "volume": 1e6}, index=s.index)
    return _daily_ohlcv


def test_crosscheck_agreement_refreshes(tmp_path, monkeypatch):
    import market_fallback as mf
    spec = _spec(tmp_path)
    spec.symbol_by_col = {"px_close": "TST"}
    good, _ = _make_stale_snapshot(spec)
    monkeypatch.setattr(mf, "daily_ohlcv", _fake_nasdaq(good["px_close"]))
    out = dg.gated_daily(spec, fetch_full=lambda: good, consumer="UNIT")
    assert out.attrs["data_gate"]["decision"] == "refreshed"
    man = json.loads(Path(spec.manifest_json).read_text())
    xc = [c for c in man["qc_checks"] if c["name"] == "nasdaq_crosscheck"]
    assert xc and xc[0]["passed"]                 # verification ran & recorded


def test_crosscheck_disagreement_rejects(tmp_path, monkeypatch):
    import market_fallback as mf
    spec = _spec(tmp_path)
    spec.symbol_by_col = {"px_close": "TST"}
    good, old = _make_stale_snapshot(spec)
    # Nasdaq agrees on history but disputes the newly-added sessions by 2%
    disputed = good["px_close"].copy()
    disputed.loc[disputed.index > old.index.max()] *= 1.02
    monkeypatch.setattr(mf, "daily_ohlcv", _fake_nasdaq(disputed))
    out = dg.gated_daily(spec, fetch_full=lambda: good, consumer="UNIT")
    gi = out.attrs["data_gate"]
    assert gi["decision"] == "fallback_snapshot"
    assert "nasdaq_crosscheck" in gi["failed_checks"]
    assert pd.Timestamp(out.index.max()) == old.index.max()   # kept last good


def test_crosscheck_unavailable_skips(tmp_path, monkeypatch):
    import market_fallback as mf
    spec = _spec(tmp_path)
    spec.symbol_by_col = {"px_close": "TST"}
    good, _ = _make_stale_snapshot(spec)
    monkeypatch.setattr(mf, "daily_ohlcv",
                        lambda *a, **k: pd.DataFrame())       # Nasdaq down
    out = dg.gated_daily(spec, fetch_full=lambda: good, consumer="UNIT")
    # no independent evidence available → must not block the refresh
    assert out.attrs["data_gate"]["decision"] == "refreshed"


def test_crosscheck_split_rebase_tolerated(tmp_path, monkeypatch):
    """A raw (unadjusted) Nasdaq series after a split differs from Yahoo by a
    constant factor — the base alignment must divide it out, not flag it."""
    import market_fallback as mf
    spec = _spec(tmp_path)
    spec.symbol_by_col = {"px_close": "TST"}
    good, _ = _make_stale_snapshot(spec)
    monkeypatch.setattr(mf, "daily_ohlcv",
                        _fake_nasdaq(good["px_close"] * 15.0))  # 15:1 raw basis
    out = dg.gated_daily(spec, fetch_full=lambda: good, consumer="UNIT")
    assert out.attrs["data_gate"]["decision"] == "refreshed"


def test_tampered_snapshot_not_pinned(tmp_path):
    """A snapshot rewritten outside save_snapshot (hash mismatch) must be
    re-validated via a fresh fetch instead of being trusted."""
    spec = _spec(tmp_path)
    dg.gated_daily(spec, fetch_full=lambda: _frame(), consumer="UNIT")
    snap = pd.read_csv(spec.snapshot_csv, index_col=0, parse_dates=True)
    snap.iloc[-1, snap.columns.get_loc("px_close")] *= 1.001   # silent edit
    dg._atomic_write(spec.snapshot_csv, dg._norm_index(snap).to_csv())
    out = dg.gated_daily(spec, fetch_full=lambda: _frame(), consumer="UNIT")
    assert out.attrs["data_gate"]["decision"] == "refreshed"   # not "pinned"
