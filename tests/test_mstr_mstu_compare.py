"""The arithmetic behind the BTC app's **MSTR-MSTU Plot** tab.

``app/mstr_mstu_compare.py`` deliberately holds no Streamlit, so everything the
tab computes — the aligned frame, the window slice, the three spread definitions,
the summary statistics and the figure assembly — is exercised here without a
browser. The cases below are the ones that were easy to get quietly wrong:
calendar misalignment between the two tickers, rebasing against the window
(not the whole history), the ratio's neutral point being 1.0 rather than 0, and
annotation coordinates on a log axis.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "app"))
import mstr_mstu_compare as mm  # noqa: E402


def _frame(dates, mstr, mstu) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two OHLCV-shaped frames in the same form the app's CSV loader returns."""
    idx = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    return (pd.DataFrame({"close": mstr}, index=idx[: len(mstr)]),
            pd.DataFrame({"close": mstu}, index=idx[: len(mstu)]))


@pytest.fixture
def sample() -> pd.DataFrame:
    """Five aligned trading days; MSTU moves exactly 2× MSTR's daily return.

    MSTR runs +10%, −10%, +10%, +10% (100 → 119.79); MSTU therefore runs +20%,
    −20%, +20%, +20% (50 → 69.12). Note the fund ends **+38.24%** against MSTR's
    **+19.79%** — a perfect 2× *daily* tracker, nowhere near 2× the period return.
    """
    dates = ["2024-09-18", "2024-09-19", "2024-09-20", "2024-09-23", "2024-09-24"]
    mstr = [100.0, 110.0, 99.0, 108.9, 119.79]
    mstu = [50.0, 60.0, 48.0, 57.6, 69.12]
    a, b = _frame(dates, mstr, mstu)
    return mm.build_comparison_frame(a, b)


# ── alignment ───────────────────────────────────────────────────────────────
def test_only_days_both_tickers_traded_survive():
    """A day present in one frame only is a data gap, not a holiday — drop it.

    Carrying it forward would print a spread move on a day one leg never traded.
    """
    idx_a = pd.DatetimeIndex(["2024-09-18", "2024-09-19", "2024-09-20"])
    idx_b = pd.DatetimeIndex(["2024-09-18", "2024-09-20"])
    a = pd.DataFrame({"close": [100.0, 110.0, 120.0]}, index=idx_a)
    b = pd.DataFrame({"close": [50.0, 60.0]}, index=idx_b)

    out = mm.build_comparison_frame(a, b)

    assert list(out.index.strftime("%Y-%m-%d")) == ["2024-09-18", "2024-09-20"]
    assert out.loc["2024-09-20", "MSTU"] == 60.0


def test_rows_before_mstu_inception_are_dropped():
    """MSTR has years of history the tab must not show — MSTU did not exist yet."""
    idx = pd.DatetimeIndex(["2023-11-01", "2024-09-18", "2024-09-19"])
    a = pd.DataFrame({"close": [42.0, 100.0, 110.0]}, index=idx)
    b = pd.DataFrame({"close": [1.0, 50.0, 60.0]}, index=idx)

    out = mm.build_comparison_frame(a, b)

    assert out.index.min() == mm.MSTU_INCEPTION
    assert len(out) == 2


@pytest.mark.parametrize("a,b", [(None, None), (None, "df"), ("df", None)])
def test_missing_inputs_return_an_empty_frame_not_none(a, b):
    """One shape for callers to handle — the tab renders an error, never a crash."""
    df = pd.DataFrame({"close": [1.0]}, index=pd.DatetimeIndex(["2024-09-18"]))
    out = mm.build_comparison_frame(df if a else None, df if b else None)

    assert out.empty and list(out.columns) == ["MSTR", "MSTU"]


def test_non_positive_prices_are_dropped():
    """A zero or negative close would divide by zero in the ratio and the rebase."""
    idx = pd.DatetimeIndex(["2024-09-18", "2024-09-19", "2024-09-20"])
    a = pd.DataFrame({"close": [100.0, 0.0, 120.0]}, index=idx)
    b = pd.DataFrame({"close": [50.0, 60.0, -1.0]}, index=idx)

    out = mm.build_comparison_frame(a, b)

    assert list(out.index.strftime("%Y-%m-%d")) == ["2024-09-18"]


# ── window ──────────────────────────────────────────────────────────────────
def test_end_bound_is_today_even_when_the_last_close_is_older(sample):
    """The picker must still offer *today* over a weekend, or the default is stale."""
    lo, hi = mm.window_bounds(sample, pd.Timestamp("2026-09-20"))

    assert lo == mm.MSTU_INCEPTION
    assert hi == pd.Timestamp("2026-09-20")


def test_window_bounds_on_empty_frame_collapse_to_today():
    lo, hi = mm.window_bounds(pd.DataFrame(), pd.Timestamp("2026-09-20"))
    assert lo == hi == pd.Timestamp("2026-09-20")


def test_slice_is_inclusive_of_both_endpoints(sample):
    out = mm.slice_window(sample, "2024-09-19", "2024-09-23")
    assert list(out.index.strftime("%Y-%m-%d")) == ["2024-09-19", "2024-09-20", "2024-09-23"]


def test_reversed_dates_are_swapped_rather_than_returning_nothing(sample):
    """A viewer who picks the dates backwards gets the window, not a blank tab."""
    forwards = mm.slice_window(sample, "2024-09-19", "2024-09-23")
    backwards = mm.slice_window(sample, "2024-09-23", "2024-09-19")
    pd.testing.assert_frame_equal(forwards, backwards)


@pytest.mark.parametrize("preset,expected", [
    ("1M", "2026-08-20"), ("3M", "2026-06-20"), ("6M", "2026-03-20"),
    ("YTD", "2026-01-01"), ("1Y", "2025-09-20"), ("MAX", "2024-09-18"),
])
def test_quick_range_presets(preset, expected):
    got = mm.quick_range_start(pd.Timestamp("2026-09-20"), preset,
                               mm.MSTU_INCEPTION)
    assert got == pd.Timestamp(expected)


def test_quick_range_never_starts_before_inception():
    """A 1-year preset a month after launch must clamp, not ask for missing data."""
    got = mm.quick_range_start(pd.Timestamp("2024-10-18"), "1Y", mm.MSTU_INCEPTION)
    assert got == mm.MSTU_INCEPTION


def test_unknown_preset_falls_back_to_max_instead_of_raising():
    """A stale session-state value from an older build must not break the tab."""
    assert mm.quick_range_start(pd.Timestamp("2026-09-20"), "17Q",
                                mm.MSTU_INCEPTION) == mm.MSTU_INCEPTION


# ── derived columns ─────────────────────────────────────────────────────────
def test_rebasing_anchors_on_the_window_not_the_full_history(sample):
    """Change the start date and the indexed lines re-anchor to it."""
    win = mm.add_derived(mm.slice_window(sample, "2024-09-20", "2024-09-24"))

    assert win["MSTR_idx"].iloc[0] == pytest.approx(100.0)
    assert win["MSTU_idx"].iloc[0] == pytest.approx(100.0)
    # MSTR 99 → 119.79 over the slice is +21%, not the full history's +19.79%.
    assert win["MSTR_idx"].iloc[-1] == pytest.approx(119.79 / 99.0 * 100.0)


def test_spread_columns(sample):
    win = mm.add_derived(sample)

    assert win["spread_usd"].iloc[-1] == pytest.approx(119.79 - 69.12)
    assert win["ratio"].iloc[-1] == pytest.approx(119.79 / 69.12)
    # Both rebased to 100 at the start, so the pp gap starts at exactly zero.
    assert win["spread_pp"].iloc[0] == pytest.approx(0.0)
    assert win["spread_pp"].iloc[-1] == pytest.approx(
        win["MSTR_idx"].iloc[-1] - win["MSTU_idx"].iloc[-1])


def test_add_derived_on_empty_frame_still_has_every_column():
    out = mm.add_derived(pd.DataFrame(columns=["MSTR", "MSTU"], dtype="float64"))
    for col in ("MSTR_idx", "MSTU_idx", "spread_usd", "spread_pp", "ratio"):
        assert col in out.columns
    assert out.empty


# ── statistics ──────────────────────────────────────────────────────────────
def test_beta_recovers_the_funds_target_leverage(sample):
    """MSTU doubles MSTR's move every day in the fixture, so β must come out 2.00."""
    s = mm.summary_stats(mm.add_derived(sample))

    assert s["beta"] == pytest.approx(2.0, abs=1e-6)
    assert s["corr"] == pytest.approx(1.0, abs=1e-6)


def test_ideal_replication_benchmark_matches_perfect_daily_doubling(sample):
    """A fund that doubles each daily move perfectly has zero tracking shortfall.

    This is the yardstick the tab quotes: 2× the *daily* move compounded, which is
    emphatically not 2× the period return — here MSTR returns +20% while the ideal
    2× path returns +38.24%.
    """
    s = mm.summary_stats(mm.add_derived(sample))

    assert s["mstr_ret"] == pytest.approx(19.79)
    assert s["lev_ideal_ret"] == pytest.approx(38.24, abs=1e-6)
    assert s["mstu_ret"] == pytest.approx(38.24, abs=1e-6)
    assert s["lev_gap_pp"] == pytest.approx(0.0, abs=1e-6)


def test_leverage_decay_shows_up_as_a_negative_shortfall():
    """Round-tripping MSTR to flat leaves a 2× daily fund down — the whole point."""
    idx = pd.DatetimeIndex(["2024-09-18", "2024-09-19", "2024-09-20"])
    a = pd.DataFrame({"close": [100.0, 110.0, 100.0]}, index=idx)
    b = pd.DataFrame({"close": [50.0, 60.0, 49.09090909]}, index=idx)

    s = mm.summary_stats(mm.add_derived(mm.build_comparison_frame(a, b)))

    assert s["mstr_ret"] == pytest.approx(0.0)
    assert s["lev_ideal_ret"] < 0
    assert s["lev_gap_pp"] == pytest.approx(0.0, abs=1e-6)


def test_stats_on_a_single_day_are_defined_but_not_invented(sample):
    """One close is a valid window; a daily-return beta from it is not."""
    s = mm.summary_stats(mm.add_derived(mm.slice_window(sample, "2024-09-18", "2024-09-18")))

    assert s["n_days"] == 1
    assert s["mstr_ret"] == pytest.approx(0.0)
    assert np.isnan(s["beta"]) and np.isnan(s["corr"])


def test_stats_on_empty_window_are_all_nan():
    s = mm.summary_stats(mm.add_derived(pd.DataFrame(columns=["MSTR", "MSTU"],
                                                     dtype="float64")))
    assert s["n_days"] == 0 and s["start"] is None
    assert np.isnan(s["beta"])


# ── "widest gap" ────────────────────────────────────────────────────────────
def test_widest_gap_is_measured_from_the_modes_own_neutral_point():
    """For the ratio, "furthest apart" means furthest from 1.0, not the largest ×.

    Ratios of 1.5, 0.4 and 1.2: the biggest NUMBER is 1.5, but 0.4 is the day the
    two were furthest apart. Measuring from zero would pick the wrong one.
    """
    idx = pd.DatetimeIndex(["2024-09-18", "2024-09-19", "2024-09-20"])
    a = pd.DataFrame({"close": [150.0, 40.0, 120.0]}, index=idx)
    b = pd.DataFrame({"close": [100.0, 100.0, 100.0]}, index=idx)
    win = mm.add_derived(mm.build_comparison_frame(a, b))

    value, day = mm.extreme_spread(win, "ratio")

    assert value == pytest.approx(0.4)
    assert pd.Timestamp(day) == pd.Timestamp("2024-09-19")


def test_widest_gap_for_differences_is_the_largest_magnitude(sample):
    win = mm.add_derived(sample)
    value, day = mm.extreme_spread(win, "usd")
    assert value == pytest.approx(float(win["spread_usd"].abs().max()))
    assert day is not None


def test_widest_gap_on_empty_window_is_nan():
    value, day = mm.extreme_spread(
        mm.add_derived(pd.DataFrame(columns=["MSTR", "MSTU"], dtype="float64")), "usd")
    assert np.isnan(value) and day is None


@pytest.mark.parametrize("value,mode,expected", [
    (12.5, "usd", "+$12.50"), (-12.5, "usd", "−$12.50"),
    (3.25, "pp", "+3.2 pp"), (-3.25, "pp", "−3.2 pp"),
    (1.234, "ratio", "1.23×"),
    (float("nan"), "usd", "—"), (None, "pp", "—"),
])
def test_format_spread(value, mode, expected):
    assert mm.format_spread(value, mode) == expected


# ── figure ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("scale", list(mm.SCALE_MODES))
@pytest.mark.parametrize("spread", list(mm.SPREAD_MODES))
def test_every_mode_combination_builds(sample, scale, spread):
    fig = mm.make_figure(mm.add_derived(sample), scale, spread)
    names = [t.name for t in fig.data if t.showlegend is not False]
    assert names[:2] == ["MSTR", "MSTU"]
    # Two series, both directly labelled at their last point.
    assert len(fig.layout.annotations) == 2


def test_price_panel_never_uses_a_second_y_axis(sample):
    """Two y-scales would make every crossing an artefact of the scaling."""
    fig = mm.make_figure(mm.add_derived(sample), "price", "usd")
    axes = {k for k in fig.layout.to_plotly_json() if k.startswith("yaxis")}
    # One axis per subplot row — never two axes overlaid on the same row.
    assert axes == {"yaxis", "yaxis2"}
    assert all(getattr(fig.layout[a], "overlaying", None) is None for a in axes)


def test_log_scale_places_annotations_in_log10_space(sample):
    """Plotly reads annotation y on a log axis as an exponent.

    Passing the raw price there once pushed the axis range out to 1e162 and left
    the panel blank, so pin the conversion down.
    """
    win = mm.add_derived(sample)
    fig = mm.make_figure(win, "log", "usd")

    ys = sorted(a["y"] for a in fig.layout.annotations)
    assert ys == pytest.approx(sorted([np.log10(119.79), np.log10(69.12)]))
    # The text still quotes real dollars, not the exponent.
    assert any("$119.79" in a["text"] for a in fig.layout.annotations)
    assert fig.layout.yaxis.type == "log"


def test_indexed_scale_plots_the_rebased_columns(sample):
    fig = mm.make_figure(mm.add_derived(sample), "indexed", "pp")
    assert fig.data[0].y[0] == pytest.approx(100.0)
    assert fig.data[1].y[0] == pytest.approx(100.0)


def test_ratio_panel_shades_around_parity_not_zero(sample):
    """``tozeroy`` would fill from the line down to 0; the ratio's neutral is 1.0."""
    fig = mm.make_figure(mm.add_derived(sample), "price", "ratio")
    fills = [t for t in fig.data if t.fill == "tonexty"]

    assert len(fills) == 2
    baselines = [t for t in fig.data if t.fill is None and t.line.width == 0]
    assert all(set(np.asarray(t.y)) == {1.0} for t in baselines)


def test_hiding_the_gap_panel_leaves_a_single_row(sample):
    fig = mm.make_figure(mm.add_derived(sample), "price", "usd", show_spread=False)
    axes = {k for k in fig.layout.to_plotly_json() if k.startswith("yaxis")}

    assert axes == {"yaxis"}
    assert [t.name for t in fig.data] == ["MSTR", "MSTU"]


def test_unknown_modes_fall_back_instead_of_raising(sample):
    """Stale session state must not be able to blank the tab."""
    fig = mm.make_figure(mm.add_derived(sample), "holographic", "furlongs")
    assert fig.layout.yaxis.type != "log"
    assert [t.name for t in fig.data][:2] == ["MSTR", "MSTU"]


def test_empty_window_still_produces_a_figure():
    fig = mm.make_figure(mm.add_derived(pd.DataFrame(columns=["MSTR", "MSTU"],
                                                     dtype="float64")))
    assert fig is not None and len(fig.layout.annotations) == 0


def test_export_frame_is_human_readable(sample):
    table = mm.export_frame(mm.add_derived(sample))

    assert table.index.name == "Date"
    assert "MSTR close ($)" in table.columns
    assert "MSTR − MSTU (pp)" in table.columns
    assert len(table) == len(sample)
