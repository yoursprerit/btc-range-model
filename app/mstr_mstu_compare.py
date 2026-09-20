"""MSTR vs MSTU price comparison — data + figure helpers for the BTC app's
**MSTR-MSTU Plot** tab.

The tab overlays MSTR (MicroStrategy) and MSTU (T-Rex 2× Long MSTR Daily Target
ETF) closes since MSTU's inception, over a user-picked date window, and plots the
gap between them underneath on a shared time axis.

Why this lives outside ``btc_hourly_app.py``
--------------------------------------------
Everything here is pure ``pandas``/``plotly``: aligning the two series, slicing
the window, the three spread definitions and the summary statistics.  Keeping it
in its own module means the arithmetic is importable and unit-testable without a
Streamlit runtime (see ``tests/test_mstr_mstu_compare.py``); the app module keeps
only the widgets and the layout.

A note on the y-axis
--------------------
MSTR and MSTU trade at very different levels (MSTU is a 2× daily-target fund that
has decayed hard since launch), which is the classic temptation to reach for a
second y-axis.  We do not: two scales on one plot make any crossing point a
visual coincidence and invite false "they diverged here" readings.  Both series
are dollars, so ``price`` puts them on ONE dollar axis; ``log`` keeps that single
axis but makes equal percentage moves equal distances; ``indexed`` rebases both
to 100 at the window start, which is the honest way to compare their *paths*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── constants ───────────────────────────────────────────────────────────────
#: MSTU's first trading day (T-Rex 2× Long MSTR Daily Target ETF).  The window
#: pickers floor here; ``data/backtest/mstu_daily.csv`` starts on the same day.
MSTU_INCEPTION = pd.Timestamp("2024-09-18")

#: MSTU's daily leverage target — used for the "ideal replication" benchmark.
MSTU_TARGET_LEVERAGE = 2.0

#: One hue per instrument, held fixed everywhere in the tab (chart, spread
#: shading, metric chips) so colour always means the same thing.  Validated as a
#: categorical pair: worst-case CVD separation ΔE 18.7 (protan), normal-vision
#: ΔE 32.9, both ≥ 3:1 against the light chart surface.
COLOR_MSTR = "#2563eb"   # blue
COLOR_MSTU = "#db2777"   # magenta
COLOR_INK = "#334155"    # slate — the spread line itself (never a series hue)
COLOR_GRID = "#e2e8f0"
COLOR_AXIS = "#94a3b8"
SURFACE = "#f8fafc"

#: Scale modes for the price panel.  ``key → (label, help)``.
SCALE_MODES: dict[str, tuple[str, str]] = {
    "price": (
        "💵 Actual price ($)",
        "Both closes in dollars on one shared axis — what each share actually cost.",
    ),
    "log": (
        "📐 Actual price — log scale",
        "Same dollars, log axis: equal percentage moves take equal vertical space, "
        "so MSTR and MSTU stay comparable despite trading at different levels.",
    ),
    "indexed": (
        "📊 Indexed — both = 100 at window start",
        "Rebases both series to 100 on the first day of the window, so the lines "
        "compare performance rather than price level.",
    ),
}

#: Spread definitions for the lower panel.  ``key → spec``.
SPREAD_MODES: dict[str, dict[str, str]] = {
    "usd": {
        "label": "➖ Price difference (MSTR − MSTU), $",
        "column": "spread_usd",
        "axis": "MSTR − MSTU ($)",
        "unit": "$",
        "hover": "$%{y:,.2f}",
        "tickformat": "$,.0f",
        "help": (
            "Literal difference of the two closes. It moves with the *price levels*, "
            "so a share-count change (split or reverse split) steps it without any "
            "economic event behind it."
        ),
    },
    "pp": {
        "label": "📊 Performance gap (MSTR − MSTU), pp",
        "column": "spread_pp",
        "axis": "MSTR − MSTU (pp)",
        "unit": "pp",
        "hover": "%{y:+,.1f} pp",
        "tickformat": "+,.0f",
        "help": (
            "Both series rebased to 100 at the window start, then subtracted: how far "
            "apart the two *paths* have travelled, in percentage points. Immune to "
            "price level, so this is the one to read for leverage decay."
        ),
    },
    "ratio": {
        "label": "➗ Price ratio (MSTR ÷ MSTU), ×",
        "column": "ratio",
        "axis": "MSTR ÷ MSTU (×)",
        "unit": "×",
        "hover": "%{y:,.3f}×",
        "tickformat": ",.2f",
        "help": (
            "How many MSTU shares one MSTR share buys. A flat line means the two "
            "moved in step; a rising line means MSTR gained on MSTU *per share*. "
            "Like the dollar difference this is a share-price ratio, so a split or "
            "reverse split rescales it — read the performance gap for the clean "
            "comparison."
        ),
    },
}

_DEFAULT_SCALE = "price"
_DEFAULT_SPREAD = "usd"


# ── data ────────────────────────────────────────────────────────────────────
def build_comparison_frame(
    mstr: pd.DataFrame | None,
    mstu: pd.DataFrame | None,
    inception: pd.Timestamp = MSTU_INCEPTION,
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
    empty = pd.DataFrame(columns=["MSTR", "MSTU"], dtype="float64")
    if mstr is None or mstu is None or "close" not in mstr or "close" not in mstu:
        return empty

    joined = pd.concat(
        [
            pd.to_numeric(mstr["close"], errors="coerce").rename("MSTR"),
            pd.to_numeric(mstu["close"], errors="coerce").rename("MSTU"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    if joined.empty:
        return empty

    joined = joined[joined.index >= pd.Timestamp(inception)]
    joined = joined[(joined["MSTR"] > 0) & (joined["MSTU"] > 0)]
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
        for col in ("MSTR_idx", "MSTU_idx", "spread_usd", "spread_pp", "ratio"):
            out[col] = pd.Series(dtype="float64")
        return out

    base_mstr = float(out["MSTR"].iloc[0])
    base_mstu = float(out["MSTU"].iloc[0])
    out["MSTR_idx"] = out["MSTR"] / base_mstr * 100.0
    out["MSTU_idx"] = out["MSTU"] / base_mstu * 100.0
    out["spread_usd"] = out["MSTR"] - out["MSTU"]
    out["spread_pp"] = out["MSTR_idx"] - out["MSTU_idx"]
    out["ratio"] = out["MSTR"] / out["MSTU"]
    return out


def summary_stats(win: pd.DataFrame) -> dict:
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
        "mstr_start": np.nan, "mstr_end": np.nan, "mstr_ret": np.nan,
        "mstu_start": np.nan, "mstu_end": np.nan, "mstu_ret": np.nan,
        "beta": np.nan, "corr": np.nan,
        "lev_ideal_ret": np.nan, "lev_gap_pp": np.nan,
    }
    if win.empty:
        return stats

    stats["start"] = win.index.min()
    stats["end"] = win.index.max()
    for tic in ("MSTR", "MSTU"):
        first = float(win[tic].iloc[0])
        last = float(win[tic].iloc[-1])
        stats[f"{tic.lower()}_start"] = first
        stats[f"{tic.lower()}_end"] = last
        stats[f"{tic.lower()}_ret"] = (last / first - 1.0) * 100.0 if first else np.nan

    # Daily-return statistics need at least two returns, i.e. three closes.
    if len(win) >= 3:
        rets = win[["MSTR", "MSTU"]].pct_change().dropna()
        if len(rets) >= 2:
            var_mstr = float(rets["MSTR"].var())
            if var_mstr > 0:
                stats["beta"] = float(rets["MSTU"].cov(rets["MSTR"]) / var_mstr)
            corr = float(rets["MSTR"].corr(rets["MSTU"]))
            stats["corr"] = corr if np.isfinite(corr) else np.nan

    if len(win) >= 2:
        simple = win["MSTR"].pct_change().dropna()
        ideal = float(np.prod(1.0 + MSTU_TARGET_LEVERAGE * simple.to_numpy()) - 1.0) * 100.0
        stats["lev_ideal_ret"] = ideal
        if np.isfinite(stats["mstu_ret"]):
            stats["lev_gap_pp"] = stats["mstu_ret"] - ideal
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


def extreme_spread(win: pd.DataFrame, spread_mode: str) -> tuple[float, object]:
    """The window's widest gap (largest distance from parity) and the day it hit.

    "Widest" is measured from the mode's own neutral point — zero for the two
    differences, 1.0 for the ratio — so the answer means "furthest apart", not
    "largest number", which for the ratio are not the same thing.
    """
    spec = SPREAD_MODES.get(spread_mode, SPREAD_MODES[_DEFAULT_SPREAD])
    col = spec["column"]
    if win.empty or col not in win or win[col].dropna().empty:
        return float("nan"), None
    baseline = 1.0 if spread_mode == "ratio" else 0.0
    series = win[col].dropna()
    idx = (series - baseline).abs().idxmax()
    return float(series.loc[idx]), idx


def export_frame(win: pd.DataFrame) -> pd.DataFrame:
    """The windowed data as a display/download table with readable column names."""
    cols = {
        "MSTR": "MSTR close ($)",
        "MSTU": "MSTU close ($)",
        "MSTR_idx": "MSTR indexed (=100)",
        "MSTU_idx": "MSTU indexed (=100)",
        "spread_usd": "MSTR − MSTU ($)",
        "spread_pp": "MSTR − MSTU (pp)",
        "ratio": "MSTR ÷ MSTU (×)",
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
    """``(mstr_col, mstu_col, axis_title, tickformat, hover_fmt)`` for a scale mode."""
    if scale_mode == "indexed":
        return "MSTR_idx", "MSTU_idx", "Indexed (start = 100)", ",.0f", "%{y:,.1f}"
    return "MSTR", "MSTU", "Close ($)", "$,.0f", "$%{y:,.2f}"


def _last_point_labels(win: pd.DataFrame, mstr_col: str, mstu_col: str,
                       value_fmt, log_axis: bool = False) -> list[dict]:
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
    y_mstr = float(win[mstr_col].iloc[-1])
    y_mstu = float(win[mstu_col].iloc[-1])
    text_mstr, text_mstu = value_fmt(y_mstr), value_fmt(y_mstu)
    if log_axis:
        y_mstr = float(np.log10(y_mstr)) if y_mstr > 0 else 0.0
        y_mstu = float(np.log10(y_mstu)) if y_mstu > 0 else 0.0

    span = abs(y_mstr - y_mstu) if log_axis else float(
        np.nanmax(win[[mstr_col, mstu_col]].to_numpy())
        - np.nanmin(win[[mstr_col, mstu_col]].to_numpy()))
    if log_axis:
        col_max = float(np.nanmax(win[[mstr_col, mstu_col]].to_numpy()))
        col_min = float(np.nanmin(win[[mstr_col, mstu_col]].to_numpy()))
        span = (float(np.log10(col_max)) - float(np.log10(col_min))
                if col_min > 0 else 0.0)
    too_close = span > 0 and abs(y_mstr - y_mstu) < 0.06 * span
    shift_mstr = shift_mstu = 0
    if too_close:
        shift_mstr, shift_mstu = (9, -9) if y_mstr >= y_mstu else (-9, 9)

    return [
        dict(x=x_last, y=y_mstr, text=f" MSTR {text_mstr}", showarrow=False,
             xanchor="left", yanchor="middle", xshift=6, yshift=shift_mstr,
             font=dict(size=11, color=COLOR_MSTR)),
        dict(x=x_last, y=y_mstu, text=f" MSTU {text_mstu}", showarrow=False,
             xanchor="left", yanchor="middle", xshift=6, yshift=shift_mstu,
             font=dict(size=11, color=COLOR_MSTU)),
    ]


def make_figure(win: pd.DataFrame, scale_mode: str = _DEFAULT_SCALE,
                spread_mode: str = _DEFAULT_SPREAD,
                show_spread: bool = True, height: int = 620) -> go.Figure:
    """Build the tab's chart: prices on top, the chosen spread underneath.

    The two panels are one figure with a shared x-axis rather than two charts, so
    a zoom, a pan or a hover on either panel lands on the same dates in the other —
    reading "the gap blew out *here*" off the price line is the whole point of the
    lower panel, and that only works if the two are locked together.

    ``show_spread=False`` collapses it to the price panel alone.
    """
    scale_mode = scale_mode if scale_mode in SCALE_MODES else _DEFAULT_SCALE
    spread_mode = spread_mode if spread_mode in SPREAD_MODES else _DEFAULT_SPREAD
    spec = SPREAD_MODES[spread_mode]
    mstr_col, mstu_col, y_title, y_tickfmt, hover_fmt = _series_columns(scale_mode)

    rows = 2 if show_spread else 1
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.07,
        row_heights=[0.66, 0.34] if rows == 2 else [1.0],
    )

    for name, col, color in (("MSTR", mstr_col, COLOR_MSTR),
                             ("MSTU", mstu_col, COLOR_MSTU)):
        fig.add_trace(go.Scatter(
            x=win.index, y=win[col] if col in win else [], name=name, mode="lines",
            line=dict(color=color, width=2),
            hovertemplate=f"<b>{name}</b> {hover_fmt}<extra></extra>",
            legendgroup=name,
        ), row=1, col=1)

    if show_spread and not win.empty and spec["column"] in win:
        spread = win[spec["column"]]
        # A ratio has no meaningful zero, so it is shaded around 1.0 (parity)
        # instead — the level at which one MSTR share buys one MSTU share.
        baseline = 1.0 if spread_mode == "ratio" else 0.0
        rel = spread - baseline
        # Two clipped fills rather than one: the sign of the gap is the point, so
        # "MSTR ahead" and "MSTU ahead" get the colour of whichever name is ahead.
        # Each half is filled with ``tonexty`` against an invisible constant trace
        # at the baseline — ``tozeroy`` would fill to y=0, which is the wrong
        # reference for the ratio (parity is 1.0, not 0).
        flat = pd.Series(baseline, index=win.index)
        for clipped, color in ((rel.clip(lower=0), COLOR_MSTR),
                               (rel.clip(upper=0), COLOR_MSTU)):
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
    annotations = _last_point_labels(win, mstr_col, mstu_col, _fmt,
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
