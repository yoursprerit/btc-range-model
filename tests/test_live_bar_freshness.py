"""The LIVE BTC app must not strand its daily views a bar behind.

Two independent defects put "⚠️ Daily signal bar is 2 bar(s) behind" on the
page on 2026-09-01 while `data/backtest/raw_features_daily.csv` held both of
the bars it said were missing:

1. **The versioned seed never ran.**  `_seed_daily_raw_from_versioned` calls
   `_load_raw_features`, which was defined ~3,200 lines FURTHER DOWN the
   module.  Streamlit executes the script top-to-bottom and the
   freshness-caption block calls `_fetch_daily_raw()` long before that point,
   so the call raised NameError, the seeder's `except Exception` swallowed it,
   and the repaired daily bars never reached the live frame.

2. **The live app had no gap repair at all.**  `_rebucket_12utc` keeps a daily
   bar only when all 24 of its hourly klines are present; the offline puller
   heals such holes but the app rebuilt its own bars from its own hourly pull,
   and its only fallback trigger was a STALE TAIL — which an INTERIOR hole
   never trips, because `max(index)` stays perfectly current.

`app/btc_hourly_app.py` runs Streamlit at import, so the ordering half is
pinned by parsing the source; the healing half lives in `app/hourly_gaps.py`
and is exercised for real.  Fully offline.
"""
import ast
import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "app"))

import hourly_gaps as H  # noqa: E402

APP = _REPO / "app" / "btc_hourly_app.py"
VENUE_RATIO = 300.0          # donor (deep venue) volume ÷ primary (thin venue)


# ════════════════════════════════════════════════════════════════════════════
# 1. The versioned seed must be reachable when the caption block runs
# ════════════════════════════════════════════════════════════════════════════
_TREE = ast.parse(APP.read_text())
_DEFS = {n.name: n.lineno for n in _TREE.body if isinstance(n, ast.FunctionDef)}


def _first_module_level_call(name: str) -> int:
    """Line of the first call to `name` OUTSIDE any def — i.e. the first one
    the interpreter actually reaches while executing the script."""
    lines = [sub.lineno
             for node in _TREE.body
             if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
             for sub in ast.walk(node)
             if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
             and sub.func.id == name]
    assert lines, f"no module-level call to {name}()"
    return min(lines)


def test_seed_helper_is_defined_before_the_script_first_builds_daily_bars():
    assert _DEFS["_load_raw_features"] < _DEFS["_seed_daily_raw_from_versioned"]
    assert _DEFS["_seed_daily_raw_from_versioned"] < _first_module_level_call("_fetch_daily_raw")


def test_the_seeder_does_not_swallow_a_name_error():
    """The safety net may hide a missing/short CSV — never an ordering bug."""
    seeder = next(n for n in _TREE.body
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "_seed_daily_raw_from_versioned")
    handlers = [h for node in ast.walk(seeder) if isinstance(node, ast.Try)
                for h in node.handlers]
    reraises = [h for h in handlers
                if isinstance(h.type, ast.Name) and h.type.id == "NameError"
                and any(isinstance(s, ast.Raise) for s in h.body)]
    assert reraises, "_seed_daily_raw_from_versioned must re-raise NameError"


def test_seed_actually_lifts_the_frame_to_the_versioned_dataset():
    """End to end on the real committed CSV: a live frame stuck two bars back
    is carried forward to the newest bar the dataset holds."""
    csv = _REPO / "data" / "backtest" / "raw_features_daily.csv"
    seed = pd.read_csv(csv, index_col=0, parse_dates=True)
    seed.index = pd.DatetimeIndex(seed.index).tz_localize(None).normalize()
    newest = seed.index.max()
    live = seed.loc[seed.index <= newest - pd.Timedelta(days=2)].copy()
    assert live.index.max() == newest - pd.Timedelta(days=2)
    assert live.combine_first(seed).sort_index().index.max() == newest


# ════════════════════════════════════════════════════════════════════════════
# 2. Interior-gap healing in the live hourly feed
# ════════════════════════════════════════════════════════════════════════════
def _now() -> pd.Timestamp:
    return pd.Timestamp("2026-09-01 16:00")


def _series(now=None, days: int = 12, scale: float = 1.0,
            px_factor: float = 1.0, drop=()) -> pd.DataFrame:
    """Synthetic hourly OHLCV for the `days` before `now`, minus `drop`."""
    now = _now() if now is None else now
    first = H.newest_closed_bar(now) - pd.Timedelta(days=days) \
        + pd.Timedelta(hours=H.ANCHOR_HOUR_UTC)
    idx = pd.date_range(first, now, freq="h", inclusive="left")
    idx = idx.difference(pd.DatetimeIndex(drop))
    close = pd.Series([(80_000.0 + (i % 97)) * px_factor
                       for i in range(len(idx))], index=idx)
    vol = pd.Series([(1.0 + (i % 5)) * scale for i in range(len(idx))], index=idx)
    return pd.DataFrame({"open": close, "high": close * 1.001,
                         "low": close * 0.999, "close": close, "volume": vol})


def _rebucket_12utc(hourly: pd.DataFrame) -> pd.DataFrame:
    """The app's bar builder, verbatim in contract: 24 hours or no bar."""
    h = hourly.copy()
    h["bucket"] = (h.index - pd.Timedelta(hours=H.ANCHOR_HOUR_UTC)).normalize()
    g = h.groupby("bucket").agg(close=("close", "last"), n=("close", "size"))
    return g[g["n"] == 24].drop(columns="n")


# ── which hours count as "missing" ───────────────────────────────────────────
def test_a_complete_feed_has_nothing_to_heal():
    assert not len(H.missing_hours(_series(), now=_now()))


def test_a_hole_in_a_closed_bar_is_reported():
    holes = H.bar_hours(H.newest_closed_bar(_now()))[4:12]      # 8 hours
    got = H.missing_hours(_series(drop=holes), now=_now())
    assert list(got) == list(holes)


def test_an_interior_hole_leaves_the_tail_current():
    """Why the app's tail-only staleness guard never fired for this."""
    holes = H.bar_hours(H.newest_closed_bar(_now()))[4:12]
    gappy = _series(drop=holes)
    assert gappy.index.max() == _series().index.max()           # tail looks fine
    assert len(H.missing_hours(gappy, now=_now())) == 8         # but a bar is short


def test_a_bar_still_in_progress_is_not_healed():
    open_bar = H.newest_closed_bar(_now()) + pd.Timedelta(days=1)
    gaps = H.missing_hours(_series(), now=_now())
    assert not [t for t in gaps if H.bar_start(t) == open_bar]


def test_history_beyond_the_lookback_is_left_alone():
    old = H.newest_closed_bar(_now()) - pd.Timedelta(days=H.HEAL_LOOKBACK_DAYS + 2)
    holes = H.bar_hours(old)[:3]
    assert not len(H.missing_hours(_series(days=14, drop=holes), now=_now()))


def test_a_bar_the_venue_barely_traded_is_not_imported_wholesale():
    holes = H.bar_hours(H.newest_closed_bar(_now()))[:20]       # 4 hours left
    assert not len(H.missing_hours(_series(drop=holes), now=_now()))


def test_an_all_nan_row_counts_as_a_hole_not_a_bar():
    holes = H.bar_hours(H.newest_closed_bar(_now()))[4:12]
    s = _series()
    s.loc[holes, "close"] = float("nan")
    assert list(H.missing_hours(s, now=_now())) == list(holes)


# ── healing itself ──────────────────────────────────────────────────────────
def test_gap_is_filled_and_volume_rescaled_onto_the_primary_scale():
    holes = H.bar_hours(H.newest_closed_bar(_now()))[4:12]
    primary = _series(drop=holes)
    donor = _series(scale=VENUE_RATIO)
    healed = H.heal_hourly_gaps(primary, [("donor", donor)], now=_now())

    assert not len(H.missing_hours(healed, now=_now()))
    for t in holes:
        assert healed.loc[t, "close"] == pytest.approx(donor.loc[t, "close"])
        assert healed.loc[t, "volume"] == pytest.approx(donor.loc[t, "volume"] / VENUE_RATIO,
                                                        rel=1e-6)
    own = primary.loc[[t for t in primary.index
                       if H.bar_start(t) == H.newest_closed_bar(_now())], "volume"]
    assert healed.loc[holes, "volume"].max() <= own.max() * 1.5


def test_a_donor_whose_prices_disagree_is_refused():
    holes = H.bar_hours(H.newest_closed_bar(_now()))[4:12]
    primary = _series(drop=holes)
    healed = H.heal_hourly_gaps(
        primary, [("bad", _series(scale=VENUE_RATIO, px_factor=1.2))], now=_now())
    assert list(H.missing_hours(healed, now=_now())) == list(holes)


def test_a_donor_with_too_little_overlap_is_refused():
    holes = H.bar_hours(H.newest_closed_bar(_now()))[4:12]
    primary = _series(drop=holes)
    thin = _series(scale=VENUE_RATIO)
    keep = pd.DatetimeIndex(holes).union(thin.index[-(H.HEAL_MIN_OVERLAP - 1):])
    healed = H.heal_hourly_gaps(primary, [("thin", thin.loc[keep])], now=_now())
    assert list(H.missing_hours(healed, now=_now())) == list(holes)


def test_healing_never_overwrites_an_hour_the_primary_already_has():
    holes = H.bar_hours(H.newest_closed_bar(_now()))[4:12]
    primary = _series(drop=holes)
    healed = H.heal_hourly_gaps(primary, [("donor", _series(scale=VENUE_RATIO))],
                                now=_now())
    kept = primary.index
    pd.testing.assert_frame_equal(healed.loc[kept], primary.loc[kept])


def test_donors_are_only_fetched_when_a_completed_bar_is_short():
    calls = []

    def donor():
        calls.append(1)
        return _series(scale=VENUE_RATIO)

    H.heal_hourly_gaps(_series(), [("lazy", donor)], now=_now())
    assert calls == []                                   # healthy feed: no fetch
    H.heal_hourly_gaps(_series(drop=H.bar_hours(H.newest_closed_bar(_now()))[4:12]),
                       [("lazy", donor)], now=_now())
    assert calls == [1]


def test_a_failing_donor_does_not_break_the_heal():
    holes = H.bar_hours(H.newest_closed_bar(_now()))[4:12]
    primary = _series(drop=holes)

    def boom():
        raise RuntimeError("host down")

    healed = H.heal_hourly_gaps(
        primary, [("dead", boom), ("good", _series(scale=VENUE_RATIO))], now=_now())
    assert not len(H.missing_hours(healed, now=_now()))


# ── end to end: the completed bar survives into the daily frame ─────────────
def test_the_bar_a_host_gap_would_have_dropped_reaches_the_daily_frame():
    bar = H.newest_closed_bar(_now())
    holes = H.bar_hours(bar)[4:12]
    primary = _series(drop=holes)
    assert bar not in _rebucket_12utc(primary).index          # the reported bug

    healed = H.heal_hourly_gaps(primary, [("donor", _series(scale=VENUE_RATIO))],
                                now=_now())
    daily = _rebucket_12utc(healed)
    assert daily.index.max() == bar                           # signal bar caught up
    assert bar + pd.Timedelta(days=1) not in daily.index      # no in-progress bar
