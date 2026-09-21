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


def test_every_gap_is_mstu_minus_mstr(sample):
    """Orientation is MSTU-first throughout, so positive always means MSTU ahead.

    The fixture ends with MSTR at 119.79 and MSTU at 69.12 — MSTR is ahead on
    price, so the dollar gap and the ratio must both come out below parity, while
    the rebased paths put MSTU ahead and the pp gap positive.
    """
    win = mm.add_derived(sample)

    assert win["spread_usd"].iloc[-1] == pytest.approx(69.12 - 119.79)
    assert win["spread_usd"].iloc[-1] < 0
    assert win["ratio"].iloc[-1] == pytest.approx(69.12 / 119.79)
    assert win["ratio"].iloc[-1] < 1.0
    # Both rebased to 100 at the start, so the pp gap starts at exactly zero.
    assert win["spread_pp"].iloc[0] == pytest.approx(0.0)
    assert win["spread_pp"].iloc[-1] == pytest.approx(
        win["MSTU_idx"].iloc[-1] - win["MSTR_idx"].iloc[-1])
    # MSTU compounded 2x daily to +38.24% against MSTR's +19.79%.
    assert win["spread_pp"].iloc[-1] > 0


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

    MSTU ÷ MSTR of 1.5, 0.4 and 1.2: the biggest NUMBER is 1.5, but 0.4 is the day
    the two were furthest apart. Measuring from zero would pick the wrong one.
    """
    idx = pd.DatetimeIndex(["2024-09-18", "2024-09-19", "2024-09-20"])
    a = pd.DataFrame({"close": [100.0, 100.0, 100.0]}, index=idx)
    b = pd.DataFrame({"close": [150.0, 40.0, 120.0]}, index=idx)
    win = mm.add_derived(mm.build_comparison_frame(a, b))

    value, day = mm.extreme_spread(win, "ratio")

    assert value == pytest.approx(0.4)
    assert pd.Timestamp(day) == pd.Timestamp("2024-09-19")


def test_widest_gap_for_differences_is_the_largest_magnitude(sample):
    """Largest distance from zero — returned SIGNED, so it says who was ahead."""
    win = mm.add_derived(sample)

    value, day = mm.extreme_spread(win, "usd")

    assert abs(value) == pytest.approx(float(win["spread_usd"].abs().max()))
    assert value == pytest.approx(float(win["spread_usd"].loc[day]))
    # MSTU trades below MSTR all through the fixture, so the widest gap is negative.
    assert value < 0


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


def test_gap_fill_above_parity_wears_mstus_colour(sample):
    """Colour follows whoever is ahead, and the gap is MSTU-minus-MSTR.

    So the ABOVE-parity half is MSTU's magenta and the below-parity half MSTR's
    blue. Inverting the subtraction without swapping these would silently colour
    each half with the wrong name.
    """
    fig = mm.make_figure(mm.add_derived(sample), "price", "usd")
    fills = [t for t in fig.data if t.fill == "tonexty"]

    assert len(fills) == 2
    assert mm.COLOR_MSTU.lstrip("#") in _rgba_hexish(fills[0].fillcolor)
    assert mm.COLOR_MSTR.lstrip("#") in _rgba_hexish(fills[1].fillcolor)
    # The first fill is the one clipped to the positive side.
    assert min(fills[0].y) >= 0 and max(fills[1].y) <= 0


def _rgba_hexish(rgba: str) -> str:
    """``rgba(r,g,b,a)`` → the equivalent ``rrggbb``, for comparing to a token."""
    nums = rgba[rgba.index("(") + 1:rgba.index(")")].split(",")
    return "".join(f"{int(float(n)):02x}" for n in nums[:3])


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
    assert "MSTU − MSTR (pp)" in table.columns
    assert len(table) == len(sample)


# ── vehicle analytics ───────────────────────────────────────────────────────
# The rating is a COST grade, not a forecast, and these tests pin that down:
# the leverage identity it rests on, the hurdle rearrangement, and the fact
# that `calibration` scores it causally and can report that it does not predict.
def _synthetic_pair(n=400, mu=0.0, sigma=0.02, carry=0.0, seed=7):
    """MSTR as a random walk, MSTU as an EXACT 2×-daily tracker minus a carry.

    Building MSTU from MSTR's own returns rather than from data means the true
    beta (2.00) and the true carry are known, so the estimators can be checked
    against a right answer instead of against themselves.
    """
    rng = np.random.default_rng(seed)
    r = rng.normal(mu, sigma, n)
    idx = pd.bdate_range("2024-09-18", periods=n)
    mstr = 100.0 * np.cumprod(1.0 + r)
    mstu = 50.0 * np.cumprod(1.0 + 2.0 * r - carry)
    return pd.DataFrame({"MSTR": mstr, "MSTU": mstu}, index=idx)


def test_tracking_fit_recovers_known_leverage_and_carry():
    df = _synthetic_pair(carry=0.001)

    fit = mm.tracking_fit(df)

    assert fit["beta"] == pytest.approx(2.0, abs=0.02)
    assert fit["alpha_daily"] == pytest.approx(-0.001, abs=2e-4)
    assert fit["alpha_ann"] == pytest.approx(fit["alpha_daily"] * mm.TRADING_DAYS)
    assert fit["r2"] > 0.99


def test_tracking_slippage_is_zero_for_a_perfect_tracker():
    """A frictionless 2×-daily fund has no slippage against its own mandate."""
    slip = mm.tracking_slippage(_synthetic_pair(carry=0.0))
    assert abs(float(slip.mean())) < 1e-9


def test_slippage_picks_up_the_carry():
    slip = mm.tracking_slippage(_synthetic_pair(carry=0.002, sigma=0.01))
    # log(1 + 2r − c) − log(1 + 2r) ≈ −c for small moves.
    assert float(slip.mean()) == pytest.approx(-0.002, abs=3e-4)


def test_leverage_identity_matches_a_simulated_path():
    """`2L − H·σ² − H·carry` must reproduce what the fund actually did.

    This is the identity the whole verdict rests on; if it drifts, the hurdle,
    the breakeven and the projection are all wrong together.
    """
    n, sigma, carry = 300, 0.02, 0.0005
    df = _synthetic_pair(n=n, sigma=sigma, carry=carry)
    H = 21
    logret = np.log(df).diff()
    actual = float(logret["MSTU"].iloc[-H:].sum())

    realised_sigma = float(df["MSTR"].pct_change().iloc[-H:].std())
    realised_carry = float(-mm.tracking_slippage(df).iloc[-H:].mean())
    predicted = (2 * float(logret["MSTR"].iloc[-H:].sum())
                 - H * realised_sigma ** 2 - H * realised_carry)

    assert actual == pytest.approx(predicted, abs=0.02)


def test_projected_mstu_is_below_twice_mstr_because_of_decay():
    """The decay term is what `2 × MSTR` leaves out — it must bite."""
    got = mm.projected_mstu(0.10, sigma_daily=0.05, drag_daily=0.0015, horizon=21)
    naive = (1.10 ** 2) - 1

    assert got < naive
    assert mm.projected_mstu(0.0, 0.05, 0.0015, 21) < 0      # flat MSTR still bleeds


def test_projected_mstu_accepts_arrays_and_scalars():
    arr = mm.projected_mstu(np.array([-0.1, 0.0, 0.1]), 0.05, 0.0015, 21)
    assert arr.shape == (3,) and np.all(np.diff(arr) > 0)
    assert isinstance(mm.projected_mstu(0.1, 0.05, 0.0015, 21), float)


def test_projected_mstu_is_nan_when_inputs_are_unknown():
    assert np.isnan(mm.projected_mstu(0.1, np.nan, 0.0015))


def test_breakeven_move_is_where_mstu_exactly_matches_mstr():
    """The definition, checked by substitution rather than restated."""
    sigma, drag, H = 0.05, 0.0015, 21
    be = mm.breakeven_move(sigma, drag, H)

    assert mm.projected_mstu(be, sigma, drag, H) == pytest.approx(be, abs=1e-9)
    # And it is the ONLY crossing: below loses, above wins.
    assert mm.projected_mstu(be - 0.02, sigma, drag, H) < be - 0.02
    assert mm.projected_mstu(be + 0.02, sigma, drag, H) > be + 0.02


def test_breakeven_rises_with_volatility_and_with_holding_period():
    """Both terms scale the decay, so both raise the bar MSTR has to clear."""
    base = mm.breakeven_move(0.04, 0.0015, 21)
    assert mm.breakeven_move(0.06, 0.0015, 21) > base     # more vol
    assert mm.breakeven_move(0.04, 0.0015, 63) > base     # longer hold
    assert mm.breakeven_move(0.04, 0.0030, 21) > base     # costlier fund


def test_breakeven_curve_crosses_zero_edge_exactly_once():
    curve = mm.breakeven_curve(0.05, 0.0015, 21)
    signs = np.sign(curve["edge"].to_numpy())
    assert (np.diff(signs) != 0).sum() == 1


# ── the rating ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("edge,expected", [
    (-0.30, "STRONG SELL"), (-0.050001, "STRONG SELL"),
    (-0.05, "SELL"), (-0.02, "SELL"),
    (-0.01, "NEUTRAL"), (0.0, "NEUTRAL"), (0.009, "NEUTRAL"),
    (0.01, "BUY"), (0.04, "BUY"),
    (0.05, "STRONG BUY"), (0.50, "STRONG BUY"),
])
def test_rating_thresholds(edge, expected):
    assert mm.rate(edge)["label"] == expected


def test_rating_of_an_unknown_edge_says_nothing_rather_than_something():
    assert mm.rate(float("nan"))["label"] == "NEUTRAL"
    assert mm.rate(None)["label"] == "NEUTRAL"


def test_every_rating_level_carries_an_icon_and_a_word():
    """Status must never rest on colour alone."""
    assert len(mm.RATING_LEVELS) == 5
    for lvl in mm.RATING_LEVELS:
        assert lvl["icon"] and lvl["label"] and lvl["gloss"]
        assert lvl["color"].startswith("#")
    floors = [l["floor"] for l in mm.RATING_LEVELS[1:]]
    assert floors == sorted(floors)          # ordered worst → best
    assert mm.RATING_LEVELS[0]["floor"] is None


def test_vehicle_read_rates_a_calm_uptrend_above_a_violent_one():
    """Same drift, more volatility → higher breakeven → worse vehicle rating.

    This is the whole mechanism in one assertion: decay scales with σ², so the
    identical drift stops covering the carry once the path gets rough.
    """
    calm = mm.vehicle_read(_synthetic_pair(mu=0.004, sigma=0.01, carry=0.0005))
    wild = mm.vehicle_read(_synthetic_pair(mu=0.004, sigma=0.05, carry=0.0005))

    assert calm["ready"] and wild["ready"]
    assert wild["breakeven"] > calm["breakeven"]
    assert wild["edge"] < calm["edge"]
    assert calm["rating"]["label"] == "STRONG BUY"
    assert wild["rating"]["label"] == "STRONG SELL"


def test_vehicle_read_figures_are_internally_consistent():
    """The edge must be exactly the difference of the two figures shown above it.

    The earlier build quoted a log-space hurdle beside a simple-return
    breakeven, so the margin did not equal the visible subtraction. One frame,
    one arithmetic.
    """
    read = mm.vehicle_read(_synthetic_pair(mu=0.002, sigma=0.02, carry=0.0005))

    assert read["edge"] == pytest.approx(read["mstu_at_pace"] - read["mstr_pace"])
    assert read["mstu_at_pace"] == pytest.approx(mm.projected_mstu(
        read["mstr_pace"], read["sigma_daily"], read["drag_daily"], read["horizon"]))
    assert read["breakeven"] == pytest.approx(mm.breakeven_move(
        read["sigma_daily"], read["drag_daily"], read["horizon"]))
    assert read["rating"] is mm.rate(read["edge"])


@pytest.mark.parametrize("horizon", [1, 2, 5, 7, 10, 20, 21, 42, 63, 100, 126, 252, 365])
def test_vehicle_read_scales_every_figure_with_the_holding_period(horizon):
    """The slider drives the whole verdict, not just the chart underneath it.

    Decay compounds, so a longer hold needs a bigger MSTR move to break even and
    bleeds further if MSTR goes nowhere — and the edge, which the rating grades,
    moves further from neutral in whichever direction it already points.
    """
    df = _synthetic_pair(mu=0.003, sigma=0.02, carry=0.0005)
    base = mm.vehicle_read(df, horizon=21)
    read = mm.vehicle_read(df, horizon=horizon)

    assert read["horizon"] == horizon
    assert read["ready"]
    if horizon > 21:
        assert read["breakeven"] > base["breakeven"]
        assert read["flat_outcome"] < base["flat_outcome"]
        assert abs(read["edge"]) > abs(base["edge"])
    elif horizon < 21:
        assert read["breakeven"] < base["breakeven"]
        assert abs(read["edge"]) < abs(base["edge"])


def test_vehicle_read_defaults_to_the_one_month_horizon():
    assert mm.vehicle_read(_synthetic_pair())["horizon"] == mm.HORIZON_DAYS


def test_vehicle_read_reports_not_ready_rather_than_guessing():
    """Six observations must not produce a confident-looking verdict."""
    read = mm.vehicle_read(_synthetic_pair(n=30))
    assert read["ready"] is False and read["n_obs"] == 30
    assert np.isnan(read["edge"]) and np.isnan(read["breakeven"])


def test_vehicle_read_on_empty_frame():
    read = mm.vehicle_read(pd.DataFrame(columns=["MSTR", "MSTU"], dtype="float64"))
    assert read["ready"] is False and read["asof"] is None


def test_vehicle_read_respects_asof_and_ignores_later_rows():
    """Dragging the end date back must replay the past, not peek at the future."""
    df = _synthetic_pair(n=400)
    cut = df.index[300]

    a = mm.vehicle_read(df, asof=cut)
    b = mm.vehicle_read(df.loc[:cut])

    assert a["asof"] == cut
    assert a["edge"] == pytest.approx(b["edge"])


def test_breakeven_figure_uses_one_axis_and_shows_both_choices():
    sigma, drag = 0.05, 0.0015
    fig = mm.make_breakeven_figure(mm.breakeven_curve(sigma, drag),
                                   mm.breakeven_move(sigma, drag))

    assert [t.name for t in fig.data] == ["Hold MSTR", "Hold MSTU (projected)"]
    axes = {k for k in fig.layout.to_plotly_json() if k.startswith("yaxis")}
    assert axes == {"yaxis"}


def test_breakeven_figure_on_empty_curve_returns_a_figure():
    fig = mm.make_breakeven_figure(pd.DataFrame(columns=["mstr", "mstu", "edge"]),
                                   float("nan"))
    assert fig is not None and len(fig.data) == 0


@pytest.mark.parametrize("horizon", [1, 5, 21, 63, 126, 252, 365])
def test_breakeven_curve_always_contains_its_own_crossing(horizon):
    """The marker must stay inside the plotted range at every holding period.

    At long holds the breakeven runs past +100%; a fixed ±40% grid stranded the
    annotation off the end of the curve, hiding the one point worth showing.
    """
    sigma, drag = 0.056, 0.0014
    be = mm.breakeven_move(sigma, drag, horizon)
    curve = mm.breakeven_curve(sigma, drag, horizon)

    assert curve["mstr"].min() <= be <= curve["mstr"].max()
    assert (np.diff(np.sign(curve["edge"].to_numpy())) != 0).sum() == 1


def test_every_whole_session_count_up_to_the_ceiling_is_usable():
    """The slider is continuous, so every integer in range must actually compute.

    It replaced a six-value preset picker, and the presets were the only counts
    anything had ever been run at. A single horizon that returns NaN, overflows
    or produces a curve the breakeven falls outside of would be a dead position
    on the slider — invisible until someone dragged onto it.
    """
    df = _synthetic_pair(n=400, mu=0.002, sigma=0.02, carry=0.0005)

    for horizon in range(1, mm.MAX_HORIZON_DAYS + 1):
        read = mm.vehicle_read(df, horizon=horizon)
        assert read["ready"], horizon
        assert read["horizon"] == horizon
        for key in ("breakeven", "mstr_pace", "mstu_at_pace", "edge", "flat_outcome"):
            assert np.isfinite(read[key]), (horizon, key)
        assert read["edge"] == pytest.approx(read["mstu_at_pace"] - read["mstr_pace"])
        assert read["rating"] in mm.RATING_LEVELS
        curve = mm.breakeven_curve(read["sigma_daily"], read["drag_daily"], horizon)
        assert np.isfinite(curve[["mstu", "edge"]].to_numpy()).all(), horizon
        assert curve["mstr"].min() <= read["breakeven"] <= curve["mstr"].max(), horizon


def test_breakeven_and_flat_outcome_are_monotone_in_the_holding_period():
    """Decay only accumulates, so the bar rises and the flat outcome sinks."""
    sigma, drag = 0.03, 0.0008
    horizons = list(range(1, mm.MAX_HORIZON_DAYS + 1))
    be = [mm.breakeven_move(sigma, drag, h) for h in horizons]
    flat = [mm.projected_mstu(0.0, sigma, drag, h) for h in horizons]

    assert all(be[i] < be[i + 1] for i in range(len(be) - 1))
    assert all(flat[i] > flat[i + 1] for i in range(len(flat) - 1))
    assert flat[-1] > -1.0          # a total return can never be worse than −100%


def test_max_horizon_is_the_advertised_ceiling():
    assert mm.MAX_HORIZON_DAYS == 365
    assert mm.HORIZON_DAYS < mm.MAX_HORIZON_DAYS
