"""The **<BASE>-<LEV> Plot** tab, rendered identically for every registered pair.

One underlying, its leveraged daily-target sibling, and the same three things in
the same order: the two closes overlaid over a window you pick, the gap between
them underneath on a shared time axis, and a verdict on whether the wrapper is
currently an efficient way to hold the underlying.

Four apps call this: the BTC app (MSTR/MSTU), the GLDM app (GLDM/UGL and
GDX/NUGT) and the generic ticker app (SOXX/SOXL, XLE/ERX).  Every label, hurdle
and colour comes from ``lev_pair_compare.PAIRS``, so the tabs cannot drift apart
as this evolves — a change lands in all of them at once, which is the whole
reason the UI lives here rather than being copied into each app.

The caller supplies the aligned frame (apps differ in how they load prices: the
BTC pair tops its versioned CSVs up from yfinance, the rest read their app's own
committed macro CSV) and everything else is shared.  Streamlit session-state
keys are scoped per pair so two tabs never collide in the one router process.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import lev_pair_compare as L


def _key(pair: L.LevPair, name: str) -> str:
    """Session-state / widget key, scoped to the pair."""
    return f"lp_{pair.key}_{name}"


#: Quick-range presets, in button order. ``MAX`` is the default window.
_PRESETS = (("1M", "1 month back"), ("3M", "3 months back"),
               ("6M", "6 months back"), ("YTD", "January 1 to today"),
               ("1Y", "1 year back"), ("MAX", "The pair's whole shared history"))


def _set_range(preset: str, floor, ceil, k_start: str, k_end: str) -> None:
    """Quick-range button callback — rewrites both pickers before the rerun.

    Runs as an ``on_click`` callback, i.e. *before* the date widgets are
    instantiated on the next run, which is the only point at which their
    session-state values may still be assigned.
    """
    start = L.quick_range_start(pd.Timestamp(ceil), preset, pd.Timestamp(floor))
    st.session_state[k_start] = start.date()
    st.session_state[k_end] = ceil


def _chart(fig, key: str) -> None:
    """``st.plotly_chart`` with scroll-zoom, degrading on older Streamlit.

    The explicit ``config=`` parameter landed in Streamlit 1.36; requirements.txt
    only floors at 1.30, so fall back to the default config rather than blanking
    the tab on an older deployment.
    """
    cfg = {"scrollZoom": True, "displaylogo": False,
           "modeBarButtonsToRemove": ["select2d", "lasso2d"]}
    try:
        st.plotly_chart(fig, use_container_width=True, key=key, config=cfg)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key)


def _pct(x, digits=1, signed=True):
    """Percent with a typographic minus, or an em dash when undefined."""
    if x is None or not np.isfinite(x):
        return "—"
    s = f"{x * 100:+.{digits}f}%" if signed else f"{x * 100:.{digits}f}%"
    return s.replace("-", "−")


def _reset_vol(live_pct: int, k_vol: str) -> None:
    """Snap the volatility slider back onto the underlying's measured reading.

    ASSIGNING the widget's key is the supported way to move a slider from code.
    Deleting it instead resets the value Python sees but leaves the thumb where
    the user dragged it — the figures snapped back to the live reading while the
    control still read 163%, which is worse than not offering the button.
    """
    st.session_state[k_vol] = int(live_pct)


def _render_verdict(pair: L.LevPair, df: pd.DataFrame, asof) -> None:
    """The vehicle verdict: is the fund an efficient way to hold the underlying?

    Two controls drive everything beneath them, and both come first.

    The holding period matters because the decay compounds with it, so the same
    conditions grade further from neutral the longer you intend to hold.

    The volatility matters more than anything else on the panel: decay scales
    with σ², so it is the single input the verdict is most sensitive to — across
    the underlying's own historical range the same drift swings the rating from
    STRONG BUY to STRONG SELL. It defaults to the live trailing reading and is dialable
    from there, because "what if the next month is calmer than the last" is the
    question a 20-session estimate invites and cannot answer.
    """
    k_vol = _key(pair, "vol")
    # Measured first, with no override, so the volatility slider can default to
    # the live reading and the caption can always name it.
    live = L.vehicle_read(df, pair, asof=asof, horizon=L.HORIZON_DAYS)
    if not live["ready"]:
        st.info(
            f"📐 Vehicle verdict needs about {max(live['vol_win'], live['drag_win'], live['drift_win']) + 1} "
            f"sessions of history to compute; only {live['n_obs']} are available up to this date."
        )
        return

    c_hz, c_vol = st.columns(2)
    with c_hz:
        # A continuous slider, not a handful of presets: the decay is smooth in
        # the holding period, so any session count is a legitimate question.
        horizon = st.slider(
            "⏳ Holding period (trading sessions)",
            min_value=1, max_value=L.MAX_HORIZON_DAYS, value=L.HORIZON_DAYS,
            step=1, key=_key(pair, "hz"),
            help="Trading sessions, not calendar days: 21 ≈ one month, 63 ≈ a quarter, "
                 "126 ≈ six months, 252 ≈ a year. The leverage decay compounds with the "
                 f"holding period, so a longer hold needs a proportionally bigger "
                 f"{pair.base} move to break even.",
        )
    with c_vol:
        vol_pct = st.slider(
            f"📈 {pair.base} volatility, annualised",
            min_value=int(round(L.MIN_VOL_ANN * 100)),
            max_value=int(round(L.MAX_VOL_ANN * 100)),
            value=int(np.clip(round(live["sigma_ann_measured"] * 100),
                              round(L.MIN_VOL_ANN * 100), round(L.MAX_VOL_ANN * 100))),
            step=1, key=k_vol, format="%d%%",
            help=f"Defaults to {pair.base}'s live trailing-20-session reading. Decay "
                 "scales with the SQUARE of this, so it is the input the verdict is most "
                 "sensitive to — worth stress-testing. Moving it changes every figure "
                 f"below EXCEPT *{pair.base} at recent pace*, which comes from the drift "
                 "and is independent of volatility.",
        )

    read = L.vehicle_read(df, pair, asof=asof, horizon=horizon,
                            sigma_ann_override=vol_pct / 100.0)
    lvl = read["rating"]
    fit = L.tracking_fit(df.loc[df.index <= pd.Timestamp(asof)] if asof is not None else df)

    st.caption(
        f"≈ **{horizon / L.TRADING_DAYS * 12:.1f} months** of market time "
        f"({horizon} trading sessions) · volatility "
        f"**{_pct(read['sigma_ann'], 0, signed=False)}** annualised."
    )
    if read["sigma_is_override"]:
        w, b = st.columns([5, 1])
        w.warning(
            f"🧪 **What-if volatility.** You are pricing the decay off "
            f"**{_pct(read['sigma_ann'], 0, signed=False)}** rather than {pair.base}'s live "
            f"trailing-{read['vol_win']} reading of "
            f"**{_pct(read['sigma_ann_measured'], 0, signed=False)}**. Every figure below "
            f"moves with it except *{pair.base} at recent pace*."
        )
        b.button("↺ Live vol", key=_key(pair, "vol_reset"), on_click=_reset_vol,
                 args=(int(round(read["sigma_ann_measured"] * 100)), k_vol),
                 use_container_width=True,
                 help=f"Snap the slider back to {pair.base}'s measured trailing volatility.")

    st.markdown(
        f"""
<div style="border-left:6px solid {lvl['color']};background:#f8fafc;
            border-radius:6px;padding:0.85rem 1.1rem;margin:0.4rem 0 0.2rem 0;">
  <div style="font-size:1.45rem;font-weight:700;color:{lvl['color']};
              letter-spacing:0.02em;">{lvl['icon']} {lvl['label']}</div>
  <div style="font-size:0.94rem;color:#334155;margin-top:0.25rem;">
    {pair.lev} as a vehicle for {pair.base} exposure over <b>{horizon} sessions</b> at
    <b>{_pct(read['sigma_ann'], 0, signed=False)}</b> volatility —
    {lvl['gloss']}.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    g1, g2, g3, g4 = st.columns(4)
    g1.metric(f"{pair.base} must gain", _pct(read["breakeven"]),
              help=f"The breakeven: what {pair.base} has to return over {horizon} sessions "
                   f"just for {pair.lev} to match it. It is the volatility drag plus the "
                   "fund's carry, compounded over the period — arithmetic, not a forecast.")
    g2.metric(f"{pair.base} at recent pace", _pct(read["base_pace"]),
              help=f"Where {pair.base} lands in {horizon} sessions if it keeps its trailing "
                   f"{read['drift_win']}-session drift. An extrapolation, and a heroic "
                   "one over the longer periods — not a prediction. The volatility "
                   "slider does not move this.")
    g3.metric(f"{pair.lev} at that pace", _pct(read["lev_at_pace"]),
              help=f"The same path run through the {pair.leverage:g}×-daily identity, decay "
                   f"included. Note it is NOT {pair.leverage:g}× the figure to its left: "
                   f"multiplying each DAY's move raises the period return to the power "
                   f"{pair.leverage:g} — (1+m)^{pair.leverage:g} − 1 — and the decay term then "
                   "multiplies the whole thing down by e^(−H(k(k−1)/2·σ²+carry)).")
    g4.metric(f"Edge ({pair.lev} − {pair.base})", _pct(read["edge"]),
              help="The difference of the two figures to the left, in points. This is "
                   "what the rating grades: positive means the leverage is paying for "
                   "its own decay at the current pace.")

    # ── how far the underlying has actually moved, by lookback ───────────────
    # The verdict extrapolates ONE drift estimate; the window it uses moves that
    # estimate a lot, so show the ladder rather than leave the choice implicit.
    ladder = L.drift_ladder(df, asof=asof)
    if any(row["ready"] for row in ladder):
        st.markdown(f"###### {pair.base}'s realised drift, by lookback")
        d_cols = st.columns(len(ladder))
        for col, row in zip(d_cols, ladder):
            if not row["ready"]:
                col.metric(row["label"], "—",
                           help="Not enough history up to this date for this lookback.")
                continue
            # Plain period labels: an abbreviated session count ("21s") reads as
            # seconds. The help below names the sessions explicitly.
            col.metric(
                row["label"],
                _pct(row["total_ret"]),
                help=(f"{pair.base}'s total move from the "
                      f"{pd.Timestamp(row['start']):%b %d, %Y} close to the "
                      f"{pd.Timestamp(row['end']):%b %d, %Y} close — "
                      f"{_pct(row['per_session_log'], 2)} per session on average, "
                      "in log terms."),
            )
        _per = " · ".join(
            f"**{r['label']}** {_pct(r['per_session_log'], 2)}"
            for r in ladder if r["ready"])
        _last = ladder[-1]
        _tail = (
            f" The verdict above extrapolates the **{read['drift_win']}-session** drift, "
            f"so the last rung is the figure it actually uses"
            + (f" ({_pct(_last['per_session_log'], 2)} per session)"
               if _last["ready"] and _last["sessions"] == read["drift_win"] else "")
            + " — read the rest as the spread of answers a different window would have given."
        )
        st.caption(
            f"📐 Realised moves, not forecasts. Per session: {_per}.{_tail} Deliberately not "
            "annualised: scaling one session to a year is legal arithmetic and "
            f"meaningless — {pair.base}'s last session annualises to "
            f"{_pct(np.expm1(ladder[0]['per_session_log'] * L.TRADING_DAYS), 0) if ladder[0]['ready'] else '—'}."
        )

    if horizon > 126:
        st.warning(
            f"📏 **{horizon} sessions is a long extrapolation.** The breakeven is still exact "
            "arithmetic — it only needs the volatility set above and the measured carry. "
            f"*{pair.base} at recent pace* is not: it compounds a trailing {read['drift_win']}-session "
            f"drift out {horizon / L.TRADING_DAYS:.1f} years, which no drift estimate "
            "survives. Read the breakeven at these lengths and treat the pace, the projection "
            "and the rating as illustration."
        )

    st.caption(
        f"⚠️ **This grades the vehicle, not the direction.** It says how expensive "
        f"{pair.lev} is as a way to hold {pair.base}, assuming {pair.base} keeps its recent "
        f"pace — it makes no claim about where {pair.base} actually goes, and it is not a "
        f"trade signal. A predictive version of this rating was built and backtested first "
        f"on the MSTR/MSTU pair across six weightings and three horizons: none was "
        f"monotonic, beyond about a month the ordering inverted, and every bucket had a "
        f"negative mean, so this grades **cost**, which is knowable, instead. Over the whole "
        f"sample {pair.lev} tracked {pair.base} at **β {fit['beta']:.2f}** "
        f"(target {pair.leverage:.2f}) with **{_pct(fit['alpha_ann'], 1)}/yr** of carry, "
        f"R² {fit['r2']:.3f}. As of **{pd.Timestamp(read['asof']):%b %d, %Y}**."
    )

    with st.expander("🔬 The arithmetic behind it", expanded=False):
        idf = L.identity_fit(df, pair)
        _k = pair.leverage
        _hurdle = (f"{_k / 2:g}·σ² + carry" if _k != 2 else "σ² + carry")
        st.markdown(
            f"For a **k× daily** fund over **H** sessions, "
            "`log(fund) ≈ k·log(underlying) − (k(k−1)/2)·H·σ² − H·carry`. "
            f"With k = {_k:g} that is "
            f"`{_k:g}L − {(_k * (_k - 1) / 2):g}·H·σ² − H·carry`."
            + (f" Checked against every {idf['horizon']}-session window of this pair it lands "
               f"at **R² = {idf['r2']:.4f}** (residual sd {idf['resid_sd'] * 100:.2f}%), against "
               f"{idf['naive_sd'] * 100:.2f}% for the naive *{pair.lev} = {_k:g} × {pair.base}* "
               "most people carry in their head." if np.isfinite(idf["r2"]) else "")
            + f" Rearranged, {pair.lev} beats simply holding {pair.base} only when "
              f"**{pair.base}'s log return clears H·({_hurdle}"
            + (f"/{_k - 1:g}" if _k != 2 else "") + ")** — the breakeven above."
        )
        st.metric(f"If {pair.base} is flat for {horizon} sessions, {pair.lev} returns",
                  _pct(read["flat_outcome"]),
                  help="Pure decay: the cost of holding the wrapper through a sideways stretch.")
        _chart(
            L.make_breakeven_figure(
                L.breakeven_curve(read["sigma_daily"], read["drag_daily"],
                                  horizon, pair.leverage),
                read["breakeven"], pair, horizon),
            key=_key(pair, "breakeven_chart"))
        st.caption(
            f"The dotted blue line is holding {pair.base}; the magenta curve is {pair.lev}'s projection at "
            f"the volatility set above ({_pct(read['sigma_ann'], 0, signed=False)} annualised"
            f"{', a what-if' if read['sigma_is_override'] else f', {pair.base}&apos;s live reading'}) and "
            f"its recent carry ({_pct(read['drag_daily'] * -L.TRADING_DAYS)}/yr, from the "
            f"trailing {read['drag_win']} sessions — the caption above quotes the full-sample "
            "figure, which differs). They cross once, at breakeven. Volatility is held fixed "
            "across the curve, so treat it as the shape of the trade-off rather than a price target."
        )


def render_lev_pair_tab(pair: L.LevPair, df: pd.DataFrame,
                        today: pd.Timestamp,
                        dataset_note: str = "") -> None:
    """The **<BASE>-<LEV> Plot** tab: price overlay + gap panel over a picked window.

    ``df`` is the aligned pair frame the app has already loaded; ``dataset_note``
    is the provenance line the app wants in the footer.  Everything else is the
    same for every pair, which is the point — one implementation, identical tabs.
    """
    k_start, k_end = _key(pair, "start"), _key(pair, "end")
    SCALE, SPREAD = L.scale_modes(pair), L.spread_modes(pair)
    _k = pair.leverage

    st.markdown(f"## 📉 {pair.base} vs {pair.lev} — Price Overlay")
    st.markdown(
        f"**{pair.base}**{f' ({pair.base_name})' if pair.base_name else ''} and "
        f"**{pair.lev}**{f' ({pair.lev_name})' if pair.lev_name else ''} closes on one "
        f"time axis, from **{pair.start:%b %d %Y}** — {pair.start_note} — to today. "
        f"Pick any window with the dates below. {pair.lev} targets "
        f"**{_k:g}× {pair.base}'s *daily* move**, which is not the same promise as "
        f"{_k:g}× {pair.base}'s return over a period: each day's move compounds off the "
        f"previous day's result, so a choppy stretch bleeds the fund even when "
        f"{pair.base} ends flat. The lower panel plots the gap between the two so that "
        "drift is visible rather than inferred."
    )

    if df.empty:
        st.error(
            f"No overlapping {pair.base} / {pair.lev} price history available "
            f"(source: `{pair.source}`)."
        )
        return

    lo_ts, hi_ts = L.window_bounds(df, today)
    lo, hi = lo_ts.date(), hi_ts.date()

    # Defaults — full history on first view; the viewer's own picks persist across
    # the app's 60s auto-refresh because they live in session state. Stored values
    # are clamped in case the bounds moved (a new trading day, a re-pulled CSV).
    st.session_state.setdefault(k_start, lo)
    st.session_state.setdefault(k_end, hi)
    st.session_state[k_start] = min(max(st.session_state[k_start], lo), hi)
    st.session_state[k_end] = min(max(st.session_state[k_end], lo), hi)

    # ── controls: the window, then the quick ranges that rewrite it ───────────
    c_start, c_end, c_quick = st.columns([1.1, 1.1, 2.8])
    with c_start:
        st.date_input("📅 Start date", min_value=lo, max_value=hi,
                      key=k_start, format="YYYY-MM-DD",
                      help=f"Earliest available is {lo:%b %d, %Y} — {pair.start_note}.")
    with c_end:
        st.date_input("📅 End date", min_value=lo, max_value=hi,
                      key=k_end, format="YYYY-MM-DD",
                      help="Defaults to today; the chart ends at the last close on or before it.")
    with c_quick:
        st.markdown("<div style='height:1.85rem'></div>", unsafe_allow_html=True)
        btn_cols = st.columns(len(_PRESETS))
        for (preset, tip), col in zip(_PRESETS, btn_cols):
            col.button(preset, key=_key(pair, f"preset_{preset}"), help=tip,
                       use_container_width=True,
                       on_click=_set_range, args=(preset, lo, hi, k_start, k_end))

    start_d, end_d = st.session_state[k_start], st.session_state[k_end]
    if start_d > end_d:
        st.warning("⚠️ Start date is after the end date — showing the window reversed.")
    win = L.add_derived(L.slice_window(df, start_d, end_d))
    if win.empty:
        st.warning(
            f"No {pair.base}/{pair.lev} closes between **{start_d:%b %d, %Y}** and "
            f"**{end_d:%b %d, %Y}** — both tickers are US-listed, so a window "
            "landing entirely on a weekend or market holiday comes back empty. "
            "Widen it, or press **MAX**."
        )
        return

    # ── view controls, directly above the chart they steer ───────────────────
    v_scale, v_spread, v_toggle = st.columns([1.5, 1.5, 0.8])
    with v_scale:
        scale_mode = st.radio(
            "Price axis", options=list(SCALE),
            format_func=lambda k: SCALE[k][0],
            index=0, key=_key(pair, "scale"),
            help=f"{pair.base} and {pair.lev} trade at very different levels. Rather than give "
                 "each its own y-axis — which makes every crossing point a "
                 "coincidence of scaling — pick how to put them on one.",
        )
    with v_spread:
        spread_mode = st.radio(
            "Gap measured as", options=list(SPREAD),
            format_func=lambda k: SPREAD[k]["label"],
            index=0, key=_key(pair, "spread"),
        )
    with v_toggle:
        st.markdown("<div style='height:1.85rem'></div>", unsafe_allow_html=True)
        show_spread = st.checkbox("Show gap panel", value=True, key=_key(pair, "showgap"),
                                  help="Hide it to give the price overlay the full height.")

    st.caption(f"ℹ️ {SCALE[scale_mode][1]}")

    # ── headline numbers ─────────────────────────────────────────────────────
    s = L.summary_stats(win, pair)
    gap_now = float(win[SPREAD[spread_mode]["column"]].iloc[-1])
    gap_ext, gap_ext_day = L.extreme_spread(win, spread_mode)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(f"{pair.base} close", f"${s['base_end']:,.2f}",
              f"{s['base_ret']:+.1f}% in window")
    k2.metric(f"{pair.lev} close", f"${s['lev_end']:,.2f}",
              f"{s['lev_ret']:+.1f}% in window")
    k3.metric("Gap now", L.format_spread(gap_now, spread_mode),
              help=SPREAD[spread_mode]["help"])
    k4.metric("Widest gap" if gap_ext_day is None
              else f"Widest gap · {pd.Timestamp(gap_ext_day):%b %d, %Y}",
              L.format_spread(gap_ext, spread_mode),
              help="The furthest apart the two got inside this window — measured "
                   "from parity, so for the ratio it is the day furthest from 1.00×, "
                   "not the largest multiple.")
    k5.metric(f"{pair.lev} β vs {pair.base}",
              "—" if not np.isfinite(s["beta"]) else f"{s['beta']:.2f}×",
              help=f"Slope of {pair.lev}'s daily return on {pair.base}'s, inside this window — "
                   "the leverage the fund actually delivered against the 2.00× it "
                   "targets. Measured on simple returns, the basis that target is "
                   "stated on.")

    _corr = "—" if not np.isfinite(s["corr"]) else f"{s['corr']:.4f}"
    if np.isfinite(s["lev_ideal_ret"]) and np.isfinite(s["lev_gap_pp"]):
        _decay = (
            f" · A frictionless fund tracking {_k:g}× {pair.base}'s move **every day** would "
            f"have returned **{s['lev_ideal_ret']:+.1f}%** over this window; {pair.lev} returned "
            f"**{s['lev_ret']:+.1f}%**, a **{s['lev_gap_pp']:+.1f} pp** shortfall to "
            f"fees, financing and daily rebalancing."
        )
    else:
        _decay = ""
    st.caption(
        f"📊 **{s['n_days']} trading days** · {pd.Timestamp(s['start']):%b %d, %Y} → "
        f"{pd.Timestamp(s['end']):%b %d, %Y} · daily-return correlation **{_corr}**"
        f"{_decay}"
    )

    # ── vehicle verdict (rating + breakeven + regime audit) ──────────────────
    # Placed above the chart because it is the question a viewer arrives with;
    # `asof` is the window's end date, so dragging that back replays the verdict
    # as it stood then rather than always quoting today.
    st.markdown(f"#### 🧭 Vehicle verdict — {pair.lev} vs {pair.base}")
    _render_verdict(pair, df, s["end"])
    st.divider()

    # ── the chart ────────────────────────────────────────────────────────────
    fig = L.make_figure(win, pair, scale_mode=scale_mode, spread_mode=spread_mode,
                          show_spread=show_spread, height=620 if show_spread else 460)
    _chart(fig, key=_key(pair, "overlay_chart"))
    if show_spread:
        st.caption(
            f"Lower panel — every gap is **{pair.lev} minus {pair.base}**, shaded **magenta "
            f"above the dashed parity line, where {pair.lev} is ahead**, and **blue below "
            f"it, where {pair.base} is ahead**. "
            f"{SPREAD[spread_mode]['help']} Drag to pan, scroll to zoom, "
            "double-click to reset; both panels share the time axis."
        )

    st.caption(
        f"📦 {dataset_note or f'Source `{pair.source}`'} · {pair.base} & {pair.lev} daily "
        "closes (split- and dividend-adjusted) · aligned on days both traded"
    )

    # ── the numbers behind the picture ───────────────────────────────────────
    with st.expander("📄 Data table & CSV download", expanded=False):
        table = L.export_frame(win, pair)
        st.dataframe(table.iloc[::-1], use_container_width=True, height=320)
        st.download_button(
            "⬇️ Download this window as CSV",
            data=table.to_csv().encode("utf-8"),
            file_name=(f"mstr_mstu_{pd.Timestamp(s['start']):%Y%m%d}_"
                       f"{pd.Timestamp(s['end']):%Y%m%d}.csv"),
            mime="text/csv",
            key=_key(pair, "csv"),
        )


