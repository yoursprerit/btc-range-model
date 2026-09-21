"""Leveraged-pair comparison — data + figure helpers for the **<BASE>-<LEV> Plot**
tabs.

Every app that trades a 1× underlying alongside a leveraged daily-target sibling
gets the same tab from this one module: the two closes overlaid since the pair's
start, the gap between them underneath on a shared time axis, and a verdict on
whether the wrapper is currently an efficient way to hold the underlying.

The pairs are registered in ``PAIRS``:

    BTC app    MSTR → MSTU   2×    since MSTU's inception
    GLDM app   GLDM → UGL    2×    full history
    GDXM app   GDX  → NUGT   2×    since NUGT's 3×→2× change
    SOXX app   SOXX → SOXL   3×    full history
    XLE app    XLE  → ERX    2×    since ERX's 3×→2× change

Everything here is pure ``pandas``/``plotly``: aligning the two series, slicing
the window, the three spread definitions and the summary statistics.  Keeping it
out of the app modules means the arithmetic is importable and unit-testable
without a Streamlit runtime (see ``tests/test_lev_pair_compare.py``); each app
keeps only the widgets and the layout, and they share one implementation so the
tabs stay identical as this evolves.

Internally the two legs are always the columns ``BASE`` and ``LEV`` — never the
tickers — so the arithmetic is written once and the labels come from the pair.

A note on the y-axis
--------------------
The two legs trade at very different levels (a leveraged fund decays away from
its underlying), which is the classic temptation to reach for a second y-axis.
We do not: two scales on one plot make any crossing point a visual coincidence
and invite false "they diverged here" readings.  Both series are dollars, so
``price`` puts them on ONE dollar axis; ``log`` keeps that single axis but makes
equal percentage moves equal distances; ``indexed`` rebases both to 100 at the
window start, which is the honest way to compare their *paths*.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass(frozen=True)
class LevPair:
    """One underlying and its leveraged daily-target sibling.

    ``leverage`` is the fund's STATED daily multiple, not a fitted one: it is
    what the decay identity needs, and the tab reports the realised beta against
    it so a drifting tracker shows up rather than being absorbed.

    ``start`` is the first date the pair is comparable.  For MSTU that is its
    inception; for NUGT and ERX it is the date their current multiple took
    effect, because a fund that changed its target is a different instrument
    for this arithmetic and splicing the eras would blend 3× decay into a 2×
    reading.
    """
    key: str
    base: str
    lev: str
    leverage: float
    start: pd.Timestamp
    start_note: str
    source: str                 # CSV under data/, or "backtest" for the BTC pair
    base_col: str = ""
    lev_col: str = ""
    base_name: str = ""
    lev_name: str = ""

    @property
    def title(self) -> str:
        return f"{self.base}-{self.lev} Plot"

    @property
    def lev_label(self) -> str:
        return f"{self.leverage:g}× {self.base}"


#: Every registered pair, keyed by the app that shows it.  Start dates for NUGT
#: and ERX were measured, not recalled: a 40-session rolling beta on each pair
#: sits at ~3.0 before 2020 and ~2.0 after, and from these dates the full-sample
#: fit is beta 1.98 / 1.99 at R² 0.996 / 0.993.
PAIRS: dict[str, LevPair] = {
    "BTC": LevPair("BTC", "MSTR", "MSTU", 2.0, pd.Timestamp("2024-09-18"),
                   "MSTU's first trading day", "backtest",
                   base_name="MicroStrategy",
                   lev_name="T-Rex 2× Long MSTR Daily Target ETF"),
    "GLDM": LevPair("GLDM", "GLDM", "UGL", 2.0, pd.Timestamp("2018-06-26"),
                    "the start of the shared price history",
                    "gldm/gldm_macro_daily.csv", "gldm_close", "ugl_close",
                    base_name="SPDR Gold MiniShares",
                    lev_name="ProShares Ultra Gold"),
    "GDXM": LevPair("GDXM", "GDX", "NUGT", 2.0, pd.Timestamp("2020-09-01"),
                    "NUGT's 3× → 2× change (measured: rolling beta drops to ~2.0)",
                    "gldm/gldm_macro_daily.csv", "gdx_close", "nugt_close",
                    base_name="VanEck Gold Miners ETF",
                    lev_name="Direxion Daily Gold Miners Bull 2×"),
    "SOXX": LevPair("SOXX", "SOXX", "SOXL", 3.0, pd.Timestamp("2015-01-02"),
                    "the start of the shared price history",
                    "soxx/macro_daily.csv", "px_close", "soxl_close",
                    base_name="iShares Semiconductor ETF",
                    lev_name="Direxion Daily Semiconductor Bull 3×"),
    "XLE": LevPair("XLE", "XLE", "ERX", 2.0, pd.Timestamp("2020-04-01"),
                   "ERX's 3× → 2× change (measured: rolling beta drops to ~2.0)",
                   "xle/macro_daily.csv", "px_close", "erx_close",
                   base_name="Energy Select Sector SPDR",
                   lev_name="Direxion Daily Energy Bull 2×"),
}

#: One hue per leg, held fixed across every pair and every tab (chart, spread
#: shading, metric chips) so colour always means the same thing.  Validated as a
#: categorical pair: worst-case CVD separation ΔE 18.7 (protan), normal-vision
#: ΔE 32.9, both ≥ 3:1 against the light chart surface.
COLOR_BASE = "#2563eb"   # blue — the 1× underlying
COLOR_LEV = "#db2777"    # magenta — the leveraged sibling
COLOR_INK = "#334155"    # slate — the spread line itself (never a series hue)
COLOR_GRID = "#e2e8f0"
COLOR_AXIS = "#94a3b8"
SURFACE = "#f8fafc"

#: Scale modes for the price panel.  ``key → (label, help)``, as templates that
#: ``scale_modes(pair)`` fills with the pair's tickers.
_SCALE_MODES: dict[str, tuple[str, str]] = {
    "price": (
        "💵 Actual price ($)",
        "Both closes in dollars on one shared axis — what each share actually cost.",
    ),
    "log": (
        "📐 Actual price — log scale",
        "Same dollars, log axis: equal percentage moves take equal vertical space, "
        "so {base} and {lev} stay comparable despite trading at different levels.",
    ),
    "indexed": (
        "📊 Indexed — both = 100 at window start",
        "Rebases both series to 100 on the first day of the window, so the lines "
        "compare performance rather than price level.",
    ),
}

#: Spread definitions for the lower panel, as templates.  ``spread_modes(pair)``
#: fills them.  Orientation is LEVERAGED-first throughout, so a positive number
#: always reads "the leveraged fund is ahead".
_SPREAD_MODES: dict[str, dict[str, str]] = {
    "usd": {
        "label": "➖ Price difference ({lev} − {base}), $",
        "column": "spread_usd",
        "axis": "{lev} − {base} ($)",
        "unit": "$",
        "hover": "$%{{y:,.2f}}",
        "tickformat": "$,.0f",
        "help": (
            "Literal difference of the two closes. It moves with the *price levels*, "
            "so a share-count change (split or reverse split) steps it without any "
            "economic event behind it."
        ),
    },
    "pp": {
        "label": "📊 Performance gap ({lev} − {base}), pp",
        "column": "spread_pp",
        "axis": "{lev} − {base} (pp)",
        "unit": "pp",
        "hover": "%{{y:+,.1f}} pp",
        "tickformat": "+,.0f",
        "help": (
            "Both series rebased to 100 at the window start, then subtracted: how far "
            "apart the two *paths* have travelled, in percentage points. Immune to "
            "price level, so this is the one to read for leverage decay."
        ),
    },
    "ratio": {
        "label": "➗ Price ratio ({lev} ÷ {base}), ×",
        "column": "ratio",
        "axis": "{lev} ÷ {base} (×)",
        "unit": "×",
        "hover": "%{{y:,.3f}}×",
        "tickformat": ",.2f",
        "help": (
            "How many {base} shares one {lev} share buys. A flat line means the two "
            "moved in step; a rising line means {lev} gained on {base} *per share*. "
            "Like the dollar difference this is a share-price ratio, so a split or "
            "reverse split rescales it — read the performance gap for the clean "
            "comparison."
        ),
    },
}


def scale_modes(pair: LevPair) -> dict[str, tuple[str, str]]:
    """``_SCALE_MODES`` with the pair's tickers filled in."""
    return {k: (lbl, hlp.format(base=pair.base, lev=pair.lev))
            for k, (lbl, hlp) in _SCALE_MODES.items()}


def spread_modes(pair: LevPair) -> dict[str, dict[str, str]]:
    """``_SPREAD_MODES`` with the pair's tickers filled in."""
    return {k: {f: (v.format(base=pair.base, lev=pair.lev)
                    if isinstance(v, str) else v)
                for f, v in spec.items()}
            for k, spec in _SPREAD_MODES.items()}


_DEFAULT_SCALE = "price"
_DEFAULT_SPREAD = "usd"


def load_pair_csv(pair: LevPair, data_root) -> pd.DataFrame:
    """Build a pair frame from the app's own committed macro CSV.

    Every non-BTC pair already has both legs side by side in the app's daily
    macro file, so no extra fetch is needed.  The BTC pair is the exception —
    it lives in the versioned ``data/backtest`` CSVs and is topped up from
    yfinance by its app, which builds the frame itself.
    """
    import pathlib as _pl
    empty = pd.DataFrame(columns=["BASE", "LEV"], dtype="float64")
    if pair.source == "backtest" or not pair.base_col or not pair.lev_col:
        return empty
    path = _pl.Path(data_root) / pair.source
    try:
        raw = pd.read_csv(path, index_col=0, parse_dates=True)
    except (OSError, ValueError):
        return empty
    if pair.base_col not in raw or pair.lev_col not in raw:
        return empty
    base = raw[pair.base_col].rename("close").to_frame()
    lev = raw[pair.lev_col].rename("close").to_frame()
    for frame in (base, lev):
        frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    return build_comparison_frame(base, lev, pair.start)


def identity_fit(df: pd.DataFrame, pair: LevPair,
                 horizon: int = 21) -> dict:
    """How well ``k·L − (k(k−1)/2)·H·σ² − H·carry`` reproduces the fund.

    Measured on THIS pair rather than quoted from the one it was first derived
    on, so every tab's claim about the identity is about its own data.  Reports
    the fit against the naive ``k × underlying`` too, which is the thing the
    identity is an improvement on.
    """
    out = {"r2": np.nan, "resid_sd": np.nan, "naive_sd": np.nan,
           "naive_bias": np.nan, "n": 0, "horizon": horizon}
    if df.empty or len(df) < horizon + 5:
        return out
    k = pair.leverage
    logret = np.log(df[["BASE", "LEV"]]).diff()
    r = df[["BASE", "LEV"]].pct_change()
    slip = tracking_slippage(df, pair)
    lev_h = logret["LEV"].rolling(horizon).sum()
    base_h = logret["BASE"].rolling(horizon).sum()
    var_h = (r["BASE"] ** 2).rolling(horizon).mean()
    carry_h = (-slip).rolling(horizon).mean()
    pred = k * base_h - horizon * (k * (k - 1.0) / 2.0 * var_h + carry_h)
    both = pd.concat([lev_h.rename("a"), pred.rename("p"),
                      (k * base_h).rename("naive")], axis=1).dropna()
    if len(both) < 10:
        return out
    corr = float(both["a"].corr(both["p"]))
    out.update(r2=corr ** 2, resid_sd=float((both["a"] - both["p"]).std()),
               naive_sd=float((both["a"] - both["naive"]).std()),
               naive_bias=float((both["a"] - both["naive"]).mean()), n=len(both))
    return out


# ── data ────────────────────────────────────────────────────────────────────
def build_comparison_frame(
    mstr: pd.DataFrame | None,
    mstu: pd.DataFrame | None,
    inception: pd.Timestamp,
) -> pd.DataFrame:
    """Align MSTR and MSTU closes onto the days BOTH actually traded.

    Takes the daily OHLCV frames the app already loads (lower-cased columns,
    tz-naive normalised ``DatetimeIndex``) and returns a two-column frame
    ``[MSTR, MSTU]`` from ``inception`` onward.

    The inner join is deliberate: these are two US-listed securities on the same
    exchange calendar, so a day present in only one frame is a data gap, not a
    holiday, and carrying it forward would invent a spread move on a day one leg
    never printed.  Returns an empty frame (not ``None``) when either side is
    missing, so callers have one shape to handle.
    """
    empty = pd.DataFrame(columns=["BASE", "LEV"], dtype="float64")
    if mstr is None or mstu is None or "close" not in mstr or "close" not in mstu:
        return empty

    joined = pd.concat(
        [
            pd.to_numeric(mstr["close"], errors="coerce").rename("BASE"),
            pd.to_numeric(mstu["close"], errors="coerce").rename("LEV"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    if joined.empty:
        return empty

    joined = joined[joined.index >= pd.Timestamp(inception)]
    joined = joined[(joined["BASE"] > 0) & (joined["LEV"] > 0)]
    return joined.sort_index()


def window_bounds(df: pd.DataFrame, today: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the ``(min, max)`` dates the pickers may offer.

    The floor is the first aligned trading day (MSTU's inception).  The ceiling is
    *today* — never the last row — so the end picker still says "today" on a
    weekend or holiday, which is what a viewer expects to see preselected; the
    chart simply ends at the last close inside that window.
    """
    today = pd.Timestamp(today).normalize()
    if df.empty:
        return today, today
    first = df.index.min()
    return first, max(today, first)


def slice_window(df: pd.DataFrame, start, end) -> pd.DataFrame:
    """Rows between ``start`` and ``end`` inclusive (both may be ``date`` or str)."""
    if df.empty:
        return df
    lo = pd.Timestamp(start).normalize()
    hi = pd.Timestamp(end).normalize()
    if lo > hi:
        lo, hi = hi, lo
    return df.loc[(df.index >= lo) & (df.index <= hi)]


def add_derived(win: pd.DataFrame) -> pd.DataFrame:
    """Add the rebased series and the three spread columns to a windowed frame.

    Rebasing is anchored on the FIRST row of the window, so every derived column
    is relative to what the viewer picked — change the start date and the indexed
    lines and the ``pp`` spread re-anchor with it.
    """
    out = win.copy()
    if out.empty:
        for col in ("BASE_idx", "LEV_idx", "spread_usd", "spread_pp", "ratio"):
            out[col] = pd.Series(dtype="float64")
        return out

    base_mstr = float(out["BASE"].iloc[0])
    base_mstu = float(out["LEV"].iloc[0])
    out["BASE_idx"] = out["BASE"] / base_mstr * 100.0
    out["LEV_idx"] = out["LEV"] / base_mstu * 100.0
    # MSTU first in every gap, so a positive number always reads "the leveraged
    # fund is ahead" — the direction a viewer of this tab is asking about.
    out["spread_usd"] = out["LEV"] - out["BASE"]
    out["spread_pp"] = out["LEV_idx"] - out["BASE_idx"]
    out["ratio"] = out["LEV"] / out["BASE"]
    return out


def summary_stats(win: pd.DataFrame, pair: LevPair) -> dict:
    """Headline numbers for the metric row above the chart.

    ``beta`` / ``corr`` are computed on daily *simple* returns, because that is the
    basis MSTU's own 2× daily objective is stated on — a beta measured on log
    returns is slightly under 2 even for a fund that hits its target perfectly, so
    it would not be comparable to the 2.00× the chip quotes it against.
    ``lev_ideal_ret`` compounds ``2×`` MSTR's daily simple return — the payoff a
    perfect, frictionless daily-target fund would have produced, which is the only
    fair yardstick for MSTU (a 2× *daily* target is not a 2× total return).
    """
    stats: dict = {
        "n_days": int(len(win)),
        "start": None, "end": None,
        "base_start": np.nan, "base_end": np.nan, "base_ret": np.nan,
        "lev_start": np.nan, "lev_end": np.nan, "lev_ret": np.nan,
        "beta": np.nan, "corr": np.nan,
        "lev_ideal_ret": np.nan, "lev_gap_pp": np.nan,
    }
    if win.empty:
        return stats

    stats["start"] = win.index.min()
    stats["end"] = win.index.max()
    for tic in ("BASE", "LEV"):
        first = float(win[tic].iloc[0])
        last = float(win[tic].iloc[-1])
        stats[f"{tic.lower()}_start"] = first
        stats[f"{tic.lower()}_end"] = last
        stats[f"{tic.lower()}_ret"] = (last / first - 1.0) * 100.0 if first else np.nan

    # Daily-return statistics need at least two returns, i.e. three closes.
    if len(win) >= 3:
        rets = win[["BASE", "LEV"]].pct_change().dropna()
        if len(rets) >= 2:
            var_mstr = float(rets["BASE"].var())
            if var_mstr > 0:
                stats["beta"] = float(rets["LEV"].cov(rets["BASE"]) / var_mstr)
            corr = float(rets["BASE"].corr(rets["LEV"]))
            stats["corr"] = corr if np.isfinite(corr) else np.nan

    if len(win) >= 2:
        simple = win["BASE"].pct_change().dropna()
        ideal = float(np.prod(1.0 + pair.leverage * simple.to_numpy()) - 1.0) * 100.0
        stats["lev_ideal_ret"] = ideal
        if np.isfinite(stats["lev_ret"]):
            stats["lev_gap_pp"] = stats["lev_ret"] - ideal
    return stats


def quick_range_start(end: pd.Timestamp, key: str, floor: pd.Timestamp) -> pd.Timestamp:
    """Start date for a quick-range preset, never earlier than ``floor``.

    ``key`` is one of ``1M``/``3M``/``6M``/``YTD``/``1Y``/``MAX``; anything else
    falls back to ``MAX`` rather than raising, so a stale session-state value
    from an older build cannot break the tab.
    """
    end = pd.Timestamp(end).normalize()
    floor = pd.Timestamp(floor).normalize()
    offsets = {
        "1M": pd.DateOffset(months=1),
        "3M": pd.DateOffset(months=3),
        "6M": pd.DateOffset(months=6),
        "1Y": pd.DateOffset(years=1),
    }
    if key == "YTD":
        start = pd.Timestamp(year=end.year, month=1, day=1)
    elif key in offsets:
        start = end - offsets[key]
    else:
        start = floor
    return max(start, floor)


def format_spread(value: float, spread_mode: str) -> str:
    """Render one spread value in its own unit, for metric chips and captions."""
    if value is None or not np.isfinite(value):
        return "—"
    if spread_mode == "ratio":
        return f"{value:,.2f}×"
    # Typographic minus throughout, matching the axis ticks plotly draws.
    if spread_mode == "pp":
        return f"{value:+,.1f} pp".replace("-", "−")
    return f"{value:+,.2f}".replace("+", "+$").replace("-", "−$")


def extreme_spread(win: pd.DataFrame, spread_mode: str,
                   pair: LevPair | None = None) -> tuple[float, object]:
    """The window's widest gap (largest distance from parity) and the day it hit.

    "Widest" is measured from the mode's own neutral point — zero for the two
    differences, 1.0 for the ratio — so the answer means "furthest apart", not
    "largest number", which for the ratio are not the same thing.
    """
    spec = _SPREAD_MODES.get(spread_mode, _SPREAD_MODES[_DEFAULT_SPREAD])
    col = spec["column"]
    if win.empty or col not in win or win[col].dropna().empty:
        return float("nan"), None
    baseline = 1.0 if spread_mode == "ratio" else 0.0
    series = win[col].dropna()
    idx = (series - baseline).abs().idxmax()
    return float(series.loc[idx]), idx


def export_frame(win: pd.DataFrame, pair: LevPair) -> pd.DataFrame:
    """The windowed data as a display/download table with readable column names."""
    cols = {
        "BASE": f"{pair.base} close ($)",
        "LEV": f"{pair.lev} close ($)",
        "BASE_idx": f"{pair.base} indexed (=100)",
        "LEV_idx": f"{pair.lev} indexed (=100)",
        "spread_usd": f"{pair.lev} − {pair.base} ($)",
        "spread_pp": f"{pair.lev} − {pair.base} (pp)",
        "ratio": f"{pair.lev} ÷ {pair.base} (×)",
    }
    present = [c for c in cols if c in win.columns]
    out = win[present].rename(columns=cols).copy()
    # Plain dates, not midnight timestamps: the app renders this frame directly,
    # and "2026-09-10 00:00:00" in every row is noise. CSV export is unaffected —
    # a normalised DatetimeIndex already serialises as a bare date.
    out.index = pd.DatetimeIndex(out.index).date
    out.index.name = "Date"
    return out.round(4)


# ── figure ──────────────────────────────────────────────────────────────────
def _series_columns(scale_mode: str) -> tuple[str, str, str, str, str]:
    """``(base_col, lev_col, axis_title, tickformat, hover_fmt)`` for a scale mode."""
    if scale_mode == "indexed":
        return "BASE_idx", "LEV_idx", "Indexed (start = 100)", ",.0f", "%{y:,.1f}"
    return "BASE", "LEV", "Close ($)", "$,.0f", "$%{y:,.2f}"


def _last_point_labels(win: pd.DataFrame, base_col: str, lev_col: str,
                       value_fmt, pair: LevPair,
                       log_axis: bool = False) -> list[dict]:
    """Direct labels pinned to each series' final point.

    Two series is well under the four the legend-plus-direct-label rule allows, and
    a label at the end of the line means identity never rests on colour alone. When
    the two endpoints nearly coincide the labels are nudged apart vertically so they
    cannot overprint each other.

    ``log_axis`` matters: plotly positions annotations on a log axis in log10
    space, so a raw price there would be read as an exponent and drag the axis
    range out to absurdity. The label TEXT always shows the real value.
    """
    if win.empty:
        return []
    x_last = win.index[-1]
    y_base = float(win[base_col].iloc[-1])
    y_lev = float(win[lev_col].iloc[-1])
    text_base, text_lev = value_fmt(y_base), value_fmt(y_lev)
    if log_axis:
        y_base = float(np.log10(y_base)) if y_base > 0 else 0.0
        y_lev = float(np.log10(y_lev)) if y_lev > 0 else 0.0

    span = abs(y_base - y_lev) if log_axis else float(
        np.nanmax(win[[base_col, lev_col]].to_numpy())
        - np.nanmin(win[[base_col, lev_col]].to_numpy()))
    if log_axis:
        col_max = float(np.nanmax(win[[base_col, lev_col]].to_numpy()))
        col_min = float(np.nanmin(win[[base_col, lev_col]].to_numpy()))
        span = (float(np.log10(col_max)) - float(np.log10(col_min))
                if col_min > 0 else 0.0)
    too_close = span > 0 and abs(y_base - y_lev) < 0.06 * span
    shift_base = shift_lev = 0
    if too_close:
        shift_base, shift_lev = (9, -9) if y_base >= y_lev else (-9, 9)

    return [
        dict(x=x_last, y=y_base, text=f" {pair.base} {text_base}", showarrow=False,
             xanchor="left", yanchor="middle", xshift=6, yshift=shift_base,
             font=dict(size=11, color=COLOR_BASE)),
        dict(x=x_last, y=y_lev, text=f" {pair.lev} {text_lev}", showarrow=False,
             xanchor="left", yanchor="middle", xshift=6, yshift=shift_lev,
             font=dict(size=11, color=COLOR_LEV)),
    ]


def make_figure(win: pd.DataFrame, pair: LevPair,
                scale_mode: str = _DEFAULT_SCALE,
                spread_mode: str = _DEFAULT_SPREAD,
                show_spread: bool = True, height: int = 620) -> go.Figure:
    """Build the tab's chart: prices on top, the chosen spread underneath.

    The two panels are one figure with a shared x-axis rather than two charts, so
    a zoom, a pan or a hover on either panel lands on the same dates in the other —
    reading "the gap blew out *here*" off the price line is the whole point of the
    lower panel, and that only works if the two are locked together.

    ``show_spread=False`` collapses it to the price panel alone.
    """
    scale_mode = scale_mode if scale_mode in _SCALE_MODES else _DEFAULT_SCALE
    spread_mode = spread_mode if spread_mode in _SPREAD_MODES else _DEFAULT_SPREAD
    spec = spread_modes(pair)[spread_mode]
    base_col, lev_col, y_title, y_tickfmt, hover_fmt = _series_columns(scale_mode)

    rows = 2 if show_spread else 1
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.07,
        row_heights=[0.66, 0.34] if rows == 2 else [1.0],
    )

    for name, col, color in ((pair.base, base_col, COLOR_BASE),
                             (pair.lev, lev_col, COLOR_LEV)):
        fig.add_trace(go.Scatter(
            x=win.index, y=win[col] if col in win else [], name=name, mode="lines",
            line=dict(color=color, width=2),
            hovertemplate=f"<b>{name}</b> {hover_fmt}<extra></extra>",
            legendgroup=name,
        ), row=1, col=1)

    if show_spread and not win.empty and spec["column"] in win:
        spread = win[spec["column"]]
        # A ratio has no meaningful zero, so it is shaded around 1.0 (parity)
        # instead — the level at which one MSTU share buys one MSTR share.
        baseline = 1.0 if spread_mode == "ratio" else 0.0
        rel = spread - baseline
        # Two clipped fills rather than one: the sign of the gap is the point, so
        # "MSTU ahead" and "MSTR ahead" get the colour of whichever name is ahead.
        # Every gap is MSTU-minus-MSTR, so ABOVE parity is MSTU's magenta.
        # Each half is filled with ``tonexty`` against an invisible constant trace
        # at the baseline — ``tozeroy`` would fill to y=0, which is the wrong
        # reference for the ratio (parity is 1.0, not 0).
        flat = pd.Series(baseline, index=win.index)
        for clipped, color in ((rel.clip(lower=0), COLOR_LEV),
                               (rel.clip(upper=0), COLOR_BASE)):
            fig.add_trace(go.Scatter(
                x=win.index, y=flat, mode="lines", line=dict(width=0),
                hoverinfo="skip", showlegend=False,
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=win.index, y=clipped + baseline, mode="lines",
                line=dict(width=0), fill="tonexty",
                fillcolor=_rgba(color, 0.16),
                hoverinfo="skip", showlegend=False,
            ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=win.index, y=spread, name=spec["axis"],
            mode="lines", line=dict(color=COLOR_INK, width=2),
            hovertemplate=f"<b>Gap</b> {spec['hover']}<extra></extra>",
        ), row=2, col=1)
        fig.add_hline(y=baseline, line_dash="dash", line_color=COLOR_AXIS,
                      line_width=1, opacity=0.7, row=2, col=1)

    # Direct end-of-line labels; the right margin below reserves room for them.
    if scale_mode == "indexed":
        def _fmt(v): return f"{v:,.1f}"
    else:
        def _fmt(v): return f"${v:,.2f}"
    annotations = _last_point_labels(win, base_col, lev_col, _fmt, pair,
                                     log_axis=scale_mode == "log")

    fig.update_layout(
        height=height,
        margin=dict(l=0, r=104, t=52, b=0),
        hovermode="x unified",
        plot_bgcolor=SURFACE, paper_bgcolor="#ffffff",
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="left", x=0, font=dict(size=12)),
        annotations=annotations,
        dragmode="pan",
    )
    # Ask for ONE tooltip spanning both panels. Only recent plotly.js builds
    # honour it; where they do not, each panel shows its own unified tooltip and
    # the crosshair still runs across both, which reads fine either way.
    try:
        fig.update_layout(hoversubplots="axis")
    except (ValueError, TypeError):
        pass

    fig.update_xaxes(
        showgrid=True, gridcolor=COLOR_GRID, zeroline=False,
        showspikes=True, spikemode="across", spikethickness=1,
        spikecolor=COLOR_AXIS, spikedash="dot",
        tickfont=dict(size=11, color="#475569"),
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=COLOR_GRID, zeroline=False,
        tickfont=dict(size=11, color="#475569"),
        title_font=dict(size=12, color="#475569"),
    )
    if scale_mode == "log":
        # "D2" labels 1/2/5 per decade. Plotly's default labels every minor tick
        # once the range is under ~2 decades, which stacks $20…$100 into an
        # unreadable ladder down the axis.
        fig.update_yaxes(title_text=y_title, tickformat=y_tickfmt, type="log",
                         dtick="D2", row=1, col=1)
    else:
        fig.update_yaxes(title_text=y_title, tickformat=y_tickfmt, type="linear",
                         row=1, col=1)
    if show_spread:
        fig.update_yaxes(title_text=spec["axis"], tickformat=spec["tickformat"],
                         row=2, col=1)
        fig.update_xaxes(title_text=None, row=2, col=1)
    return fig


def _rgba(hex_color: str, alpha: float) -> str:
    """``#rrggbb`` → ``rgba(r,g,b,alpha)`` for translucent fills."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# ════════════════════════════════════════════════════════════════════════
# Vehicle analytics — is MSTU an efficient way to hold MSTR right now?
# ════════════════════════════════════════════════════════════════════════
# MSTU is not an independent security with its own fair value: it is a
# mechanical 2×-daily wrapper around MSTR.  Regressed on the full aligned
# history its daily return is ``1.969 × MSTR − 0.149%/day`` at R² = 0.998 —
# the beta is nailed to target and barely moves (20-day rolling β spans
# 1.91–2.00), so there is no dispersion to trade.  What there IS, is a
# quantifiable cost of carry, and a closed-form condition for when the
# leverage pays for itself.
#
# The identity these helpers rest on, for a k× daily fund over H sessions:
#
#     log(fund_H) ≈ k·log(underlying_H) − (k(k−1)/2)·H·σ² − H·drag
#
# and for k = 2 that is ``2L − H·σ² − H·drag``.  Checked against every
# 21-session window in the sample it lands at **R² = 0.9999**, residual
# sd 0.80% — against 4.48% for the naive "MSTU = 2× MSTR" that most people
# carry in their head.  Rearranged, MSTU beats simply holding MSTR only when
#
#     L_MSTR  >  H · (σ² + drag)            ← the hurdle
#
# which is arithmetic, not a forecast, and is what the rating below grades.
#
# WHAT THE RATING IS NOT
# ----------------------
# It is not a price forecast, and it must never be presented as one.  Before
# writing any of this a predictive version WAS built and backtested — six
# weightings (hurdle margin, volatility, trend efficiency, tracking quality,
# and blends) × three horizons.  Not one was monotonic; at 42 and 63 sessions
# the ordering INVERTS, and the top bucket beat MSTR 0% of the time at 63
# sessions.  Every bucket, every horizon, every weighting had a negative mean.
# The features that looked predictive were proxies for MSTR mean-reverting
# after a selloff — a directional bet on MSTR wearing a costume, fitted to one
# crypto cycle.  So the rating grades the *vehicle's cost*, which is knowable,
# rather than direction, which on this evidence is not.  The UI carries that
# caveat in prose beside the chip; the measurements behind it are in the commit
# that introduced this block, and in the numbers above.  If you are tempted to
# re-weight these components into something predictive, re-run that experiment
# first — it is cheap, and it says no.

#: Sessions in the "one month" horizon every carry number defaults to.
HORIZON_DAYS = 21
#: The longest holding period the UI offers, in sessions.  Past roughly a year
#: of sessions, extrapolating a trailing drift stops meaning anything at all —
#: the caller warns about that well before this ceiling.
MAX_HORIZON_DAYS = 365

#: Bounds for the volatility the UI lets you dial in, annualised.  MSTR's own
#: trailing-20 reading has spanned 33–156% since MSTU launched (14–257% on a
#: 5-session window), so this brackets everything observed with room for a
#: calmer or a more violent regime than it has yet produced.
MIN_VOL_ANN = 0.20
MAX_VOL_ANN = 2.50
#: The slider's resolution, annualised.  Differences finer than this cannot be
#: dialled in, so they are measurement, not intent.
VOL_STEP_ANN = 0.01

#: Lookbacks for the drift ladder: how far MSTR has actually moved, read over
#: several trailing windows.  It exists because the verdict extrapolates ONE
#: drift estimate and the choice of window moves that estimate a great deal —
#: the ladder makes the sensitivity visible instead of leaving it buried in a
#: parameter.  Counted in SESSIONS rather than calendar periods, matching the
#: holding-period slider and the identity's own unit.  Two rungs are anchors
#: rather than round numbers: 21 is ``HORIZON_DAYS``, the holding period the
#: verdict defaults to, and 60 is ``vehicle_read``'s default ``drift_win``, so
#: the ladder ends on the number the verdict actually extrapolates rather than
#: near it.
DRIFT_LADDER_WINDOWS: tuple[tuple[int, str], ...] = (
    (1, "1 session"), (5, "5 sessions"), (21, "21 sessions"),
    (30, "30 sessions"), (60, "60 sessions"),
)
#: Sessions per year, for annualising volatility.
TRADING_DAYS = 252


#: The five levels, ordered worst → best, each with the lower bound of the
#: monthly margin that selects it (``None`` = no lower bound).  Cuts are round
#: numbers on a directly meaningful axis — percentage points per month of
#: expected shortfall — NOT fitted to the sample, because a sample this thin
#: cannot support fitting.  Every level ships with an icon and a word so the
#: verdict never rests on colour alone.
RATING_LEVELS: tuple[dict, ...] = (
    {"key": "strong_sell", "label": "STRONG SELL", "icon": "⛔",
     "color": "#b91c1c", "floor": None,
     "gloss": "the leverage is deeply underwater — {base} is nowhere near the "
              "drift {lev} needs to justify its carry"},
    {"key": "sell", "label": "SELL", "icon": "🔻",
     "color": "#dc2626", "floor": -0.05,
     "gloss": "{base} is drifting below {lev}'s hurdle; the wrapper is costing "
              "more than the leverage is adding"},
    {"key": "neutral", "label": "NEUTRAL", "icon": "⚪",
     "color": "#64748b", "floor": -0.01,
     "gloss": "{base}'s drift is within a point a month of {lev}'s hurdle — "
              "the two vehicles are close to a wash"},
    {"key": "buy", "label": "BUY", "icon": "🟢",
     "color": "#16a34a", "floor": 0.01,
     "gloss": "{base} is compounding above {lev}'s hurdle; the leverage is "
              "currently paying for its own decay"},
    {"key": "strong_buy", "label": "STRONG BUY", "icon": "✅",
     "color": "#15803d", "floor": 0.05,
     "gloss": "{base} is compounding far above {lev}'s hurdle — the conditions "
              "leveraged ETFs are built for"},
)


def daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns for both legs, NaNs dropped."""
    if df.empty or len(df) < 2:
        return pd.DataFrame(columns=["BASE", "LEV"], dtype="float64")
    return df[["BASE", "LEV"]].pct_change().dropna()


def tracking_slippage(df: pd.DataFrame, pair: LevPair) -> pd.Series:
    """Per-session slippage of MSTU against a frictionless 2×-daily tracker.

    ``e_t = log(1 + r_MSTU) − log(1 + 2·r_MSTR)``, i.e. what the fund lost (or
    gained) on the day relative to the mandate it advertises.  Its mean is the
    fund's true cost of carry — fees, swap financing and the slippage of
    rebalancing a 2× book daily in a name this volatile — and on this sample it
    runs about −15 bp a session.
    """
    r = daily_returns(df)
    if r.empty:
        return pd.Series(dtype="float64")
    return np.log1p(r["LEV"]) - np.log1p(pair.leverage * r["BASE"])


def tracking_fit(df: pd.DataFrame) -> dict:
    """OLS of MSTU's daily return on MSTR's: delivered leverage and carry.

    ``beta`` is the leverage the fund actually delivered against the 2.00× it
    targets; ``alpha`` is the daily cost that survives after the leverage is
    accounted for, reported per session, per month and annualised.
    """
    out = {"beta": np.nan, "alpha_daily": np.nan, "alpha_month": np.nan,
           "alpha_ann": np.nan, "r2": np.nan, "n": 0}
    r = daily_returns(df)
    if len(r) < 10:
        return out
    x = r["BASE"].to_numpy(dtype="float64")
    y = r["LEV"].to_numpy(dtype="float64")
    if not np.isfinite(x).all() or float(np.var(x)) <= 0:
        return out
    beta, alpha = np.polyfit(x, y, 1)
    corr = float(np.corrcoef(x, y)[0, 1])
    out.update(beta=float(beta), alpha_daily=float(alpha),
               alpha_month=float(alpha) * HORIZON_DAYS,
               alpha_ann=float(alpha) * TRADING_DAYS,
               r2=corr ** 2, n=len(r))
    return out


def rate(edge: float, pair: LevPair | None = None) -> dict:
    """Map an edge (the fund's projected outcome minus the underlying's) to a level.

    With a ``pair`` the returned dict's ``gloss`` names that pair's tickers;
    without one the template is returned untouched, which keeps the pure
    threshold logic callable from anywhere.
    """
    if edge is None or not np.isfinite(edge):
        chosen = RATING_LEVELS[2]        # NEUTRAL — say nothing, not something
    else:
        chosen = RATING_LEVELS[0]
        for level in RATING_LEVELS:
            if level["floor"] is not None and edge >= level["floor"]:
                chosen = level
    if pair is None:
        return chosen
    return {**chosen,
            "gloss": chosen["gloss"].format(base=pair.base, lev=pair.lev)}


def vehicle_read(df: pd.DataFrame, pair: LevPair, asof=None,
                 horizon: int = HORIZON_DAYS,
                 sigma_ann_override: float | None = None,
                 vol_win: int = 20, drag_win: int = 60,
                 drift_win: int = 60) -> dict:
    """Grade MSTU's cost of carry against MSTR, over a chosen holding period.

    The inputs are measured on the trailing sessions up to ``asof`` (default:
    the last row), NOT on the chart's window — a viewer zoomed to two weeks
    should still get a hurdle built from enough history to mean something.
    Quoting ``asof`` from the picked end date does mean dragging the end date
    back replays the verdict as it stood then, which is the useful behaviour.

    Everything comes out in ONE frame: simple total returns over ``horizon``
    sessions, all pushed through ``projected_lev``.  That matters because the
    alternative — quoting a log-space hurdle beside a simple-return breakeven —
    puts two numbers for the same quantity on screen that do not agree, and a
    margin that does not equal the difference of the two figures above it.

        breakeven     what MSTR must gain for MSTU to merely match it
        base_pace     what MSTR makes over the period at its trailing drift
        lev_at_pace  what MSTU makes at that same pace, decay included
        edge          lev_at_pace − base_pace, in points — what the rating grades

    ``edge`` scales with ``horizon``, and deliberately so: the decay compounds,
    so the same conditions grade further from neutral the longer you intend to
    hold — in whichever direction they already point.  Extrapolating a trailing
    drift over six months is a heroic assumption, and the caller says so.

    ``sigma_ann_override`` swaps in a what-if volatility (annualised) for the
    trailing reading.  Decay scales with σ², so it is the input the verdict is
    most sensitive to and the one most worth stress-testing; the measured value
    is still returned as ``sigma_ann_measured`` so the caller can show both and
    never pass a hypothetical off as an observation.  Note it does NOT touch
    ``base_pace``, which comes from the drift and is independent of volatility.

    Returns ``ready=False`` when the trailing history is too short to compute
    the inputs, so the caller can say so rather than render a confident-looking
    number built on six observations.
    """
    read = {
        "ready": False, "asof": None, "n_obs": 0, "horizon": int(horizon),
        "sigma_daily": np.nan, "sigma_ann": np.nan,
        "sigma_ann_measured": np.nan, "sigma_is_override": False,
        "drag_daily": np.nan, "drag_period": np.nan,
        "hurdle_daily": np.nan, "drift_daily": np.nan,
        "breakeven": np.nan, "base_pace": np.nan, "lev_at_pace": np.nan,
        "edge": np.nan, "flat_outcome": np.nan, "rating": RATING_LEVELS[2],
        "vol_win": vol_win, "drag_win": drag_win, "drift_win": drift_win,
    }
    if df.empty:
        return read
    hist = df if asof is None else df.loc[df.index <= pd.Timestamp(asof)]
    need = max(vol_win, drag_win, drift_win) + 1
    read["asof"] = hist.index.max() if len(hist) else None
    read["n_obs"] = len(hist)
    if len(hist) < need:
        return read

    r = daily_returns(hist)
    logret = np.log(hist[["BASE", "LEV"]]).diff().dropna()
    slip = tracking_slippage(hist, pair)

    sigma_measured = float(r["BASE"].iloc[-vol_win:].std())
    sigma = sigma_measured
    is_override = False
    if sigma_ann_override is not None and np.isfinite(sigma_ann_override):
        sigma_ann = float(np.clip(sigma_ann_override, MIN_VOL_ANN, MAX_VOL_ANN))
        sigma = sigma_ann / np.sqrt(TRADING_DAYS)
        # Flagged as a what-if only when it differs by MORE than the slider can
        # express.  The control steps in whole percent, so a live reading of
        # 111.88% arrives back as 112% and an exact comparison would brand the
        # untouched default a hypothetical — which it is not.
        # bool(): numpy comparisons yield np.bool_, which fails an `is False`
        # check and pickles oddly into Streamlit's cache.
        is_override = bool(
            abs(sigma_ann - sigma_measured * np.sqrt(TRADING_DAYS)) > VOL_STEP_ANN / 2)
    # Cost of carry as a POSITIVE number of log-points per session.
    drag = float(-slip.iloc[-drag_win:].mean())
    # The hurdle per session: the fund beats the underlying only once the
    # underlying's log drift clears k/2·σ² + carry/(k−1).  Over the period that
    # becomes the breakeven move.
    k = pair.leverage
    hurdle = k / 2.0 * sigma ** 2 + drag / (k - 1.0)
    drift = float(logret["BASE"].iloc[-drift_win:].mean())

    base_pace = float(np.expm1(drift * horizon))
    lev_at_pace = float(projected_lev(base_pace, sigma, drag, horizon, k))

    read.update(
        ready=True, sigma_daily=sigma, sigma_ann=sigma * np.sqrt(TRADING_DAYS),
        sigma_ann_measured=sigma_measured * np.sqrt(TRADING_DAYS),
        sigma_is_override=is_override,
        drag_daily=drag, drag_period=drag * horizon,
        hurdle_daily=hurdle, drift_daily=drift,
        breakeven=breakeven_move(sigma, drag, horizon, k),
        base_pace=base_pace, lev_at_pace=lev_at_pace,
        edge=lev_at_pace - base_pace,
        flat_outcome=float(projected_lev(0.0, sigma, drag, horizon, k)),
        rating=rate(lev_at_pace - base_pace, pair),
    )
    return read


def drift_ladder(df: pd.DataFrame, asof=None,
                 windows: tuple[tuple[int, str], ...] = DRIFT_LADDER_WINDOWS
                 ) -> list[dict]:
    """MSTR's realised move over each trailing lookback.

    Each entry carries the total return over the window and the average
    log return per session inside it.  Both are measurements of what already
    happened — nothing here is extrapolated.

    Deliberately NOT annualised.  Scaling a single session to a year is
    arithmetically legal and completely meaningless: MSTR's last session was
    +16.4%, which annualises to about 4×10¹⁸ %.  A number that large on screen
    reads as a bug, and a reader who takes it at face value has been actively
    misled, so the per-session average is as far as this scales.

    Entries whose window is longer than the available history come back with
    ``ready=False`` rather than a value computed from fewer sessions than the
    label claims.
    """
    out: list[dict] = []
    hist = df if asof is None else df.loc[df.index <= pd.Timestamp(asof)]
    closes = hist["BASE"] if "BASE" in hist else pd.Series(dtype="float64")
    for sessions, label in windows:
        row = {"label": label, "sessions": sessions, "ready": False,
               "start": None, "end": None,
               "total_ret": np.nan, "per_session_log": np.nan}
        if len(closes) >= sessions + 1:
            first = float(closes.iloc[-1 - sessions])
            last = float(closes.iloc[-1])
            if first > 0 and last > 0:
                row.update(ready=True,
                           start=closes.index[-1 - sessions], end=closes.index[-1],
                           total_ret=last / first - 1.0,
                           per_session_log=float(np.log(last / first)) / sessions)
        out.append(row)
    return out


# ── B: the breakeven calculator ─────────────────────────────────────────────
def projected_lev(base_total_ret, sigma_daily: float, drag_daily: float,
                  horizon: int = HORIZON_DAYS, leverage: float = 2.0):
    """MSTU's total return for a given MSTR total return over ``horizon``.

    Applies the identity at the top of this section, so it answers the question
    a holder actually has — "if MSTR does X over the next month, where does
    MSTU land?" — including the decay that ``2 × X`` silently omits.  Accepts a
    scalar or an array; returns the same shape, as a simple (not log) return.
    """
    arr = np.asarray(base_total_ret, dtype="float64")
    if not np.isfinite(sigma_daily) or not np.isfinite(drag_daily):
        return np.full(arr.shape, np.nan) if arr.ndim else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        underlying_log = np.log1p(np.clip(arr, -0.9999, None))
    k = leverage
    decay = horizon * (k * (k - 1.0) / 2.0 * sigma_daily ** 2 + drag_daily)
    fund_log = k * underlying_log - decay
    out = np.expm1(fund_log)
    return out if arr.ndim else float(out)


def breakeven_move(sigma_daily: float, drag_daily: float,
                   horizon: int = HORIZON_DAYS, leverage: float = 2.0) -> float:
    """The underlying's total return at which the fund exactly matches it.

    Below it the leverage loses to the plain shares; above it the leverage
    wins.  Setting ``projected_lev(L) = L`` and solving gives

        L* = exp(H · (k/2 · σ² + carry/(k−1))) − 1

    which is ``exp(H·(σ² + carry)) − 1`` at k = 2 and a materially higher bar at
    k = 3 — SOXL needs SOXX to clear 1.5σ² a session where MSTU needs 1.0σ².
    It carries no forecast of whether the underlying will get there.
    """
    if (not np.isfinite(sigma_daily) or not np.isfinite(drag_daily)
            or leverage <= 1.0):
        return np.nan
    k = leverage
    per_session = k / 2.0 * sigma_daily ** 2 + drag_daily / (k - 1.0)
    return float(np.expm1(horizon * per_session))


def breakeven_curve(sigma_daily: float, drag_daily: float,
                    horizon: int = HORIZON_DAYS, leverage: float = 2.0,
                    moves=None) -> pd.DataFrame:
    """``base`` / ``lev`` / ``edge`` over a grid of underlying moves.

    ``edge`` is the fund minus the underlying: negative means the plain shares
    won.  It
    crosses zero exactly once, at ``breakeven_move``.

    The default grid STRETCHES to contain that crossing.  At a 126-session hold
    the breakeven runs past +100%, and a fixed ±40% grid left the marker
    stranded off the end of the curve with an empty gulf between them — the one
    point the chart exists to show, pushed out of frame.
    """
    if moves is None:
        be = breakeven_move(sigma_daily, drag_daily, horizon, leverage)
        top = 0.40 if not np.isfinite(be) else max(0.40, be * 1.25)
        moves = np.linspace(-0.40, top, 60)
    moves = np.asarray(moves, dtype="float64")
    lev = projected_lev(moves, sigma_daily, drag_daily, horizon, leverage)
    return pd.DataFrame({"base": moves, "lev": lev, "edge": lev - moves})


def make_breakeven_figure(curve: pd.DataFrame, breakeven: float,
                          pair: LevPair, horizon: int = HORIZON_DAYS,
                          height: int = 330) -> go.Figure:
    """Plot MSTU's projected outcome against MSTR's move over ``horizon``.

    Both series are total returns over the same window, so they share one axis —
    the 1:1 line IS holding MSTR, and the gap to the curve is what the leverage
    adds or costs.  The crossing point is the breakeven: left of it the plain
    shares win, right of it the leverage does.
    """
    fig = go.Figure()
    if curve.empty:
        return fig
    x = curve["base"].to_numpy()

    fig.add_trace(go.Scatter(
        x=x, y=x, name=f"Hold {pair.base}", mode="lines",
        line=dict(color=COLOR_BASE, width=2, dash="dot"),
        hovertemplate=f"<b>{pair.base}</b> %{{y:+.1%}}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=curve["lev"], name=f"Hold {pair.lev} (projected)", mode="lines",
        line=dict(color=COLOR_LEV, width=2.5),
        hovertemplate=f"<b>{pair.lev}</b> %{{y:+.1%}}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color=COLOR_AXIS, line_width=1, opacity=0.5)
    if np.isfinite(breakeven):
        fig.add_vline(
            x=breakeven, line_dash="dash", line_color=COLOR_INK, line_width=1.5,
            annotation_text=f"  breakeven {breakeven:+.1%}",
            annotation_position="top left",
            annotation_font=dict(size=11, color=COLOR_INK),
        )
    fig.update_layout(
        height=height, margin=dict(l=0, r=16, t=42, b=0),
        hovermode="x unified", plot_bgcolor=SURFACE, paper_bgcolor="#ffffff",
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="left", x=0, font=dict(size=12)),
        dragmode="pan",
    )
    fig.update_xaxes(
        title_text=f"{pair.base} total return over the next {horizon} sessions",
        tickformat="+.0%", showgrid=True, gridcolor=COLOR_GRID, zeroline=False,
        tickfont=dict(size=11, color="#475569"),
        title_font=dict(size=12, color="#475569"),
    )
    fig.update_yaxes(
        title_text="Your total return", tickformat="+.0%",
        showgrid=True, gridcolor=COLOR_GRID, zeroline=False,
        tickfont=dict(size=11, color="#475569"),
        title_font=dict(size=12, color="#475569"),
    )
    return fig
