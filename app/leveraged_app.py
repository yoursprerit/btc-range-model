"""Leveraged Assets — every leveraged wrapper's Vehicle Verdict on one page.

Selected from the sidebar **Application** radio ("⚡ Leveraged Assets").

Each app that trades a 1× underlying alongside a leveraged daily-target sibling
carries a ``<BASE>-<LEV> Plot`` tab whose **Vehicle verdict** panel grades how
expensive that wrapper currently is.  Answering "which of them is the cheapest
way to hold its underlying right now?" meant opening five tabs and holding the
numbers in your head.  This page puts the same figures side by side.

It computes nothing of its own: every row is ``lev_pair_compare.board_row`` —
the identical ``vehicle_read`` call the tab makes, at a holding period you set
once for all five — so a number here and the same number on its own tab can
never disagree.  Stats only; the prose, the breakeven curve and the arithmetic
walkthrough stay on the per-pair tabs, one click away via the buttons below the
board.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _APP_DIR.parent
for _p in (str(_APP_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import lev_pair_compare as L       # noqa: E402
import ticker_config               # noqa: E402

try:
    st.set_page_config(page_title="Leveraged Assets", page_icon="⚡",
                       layout="wide", initial_sidebar_state="expanded")
except Exception:
    pass


# ── sidebar Application selector (same widget/key as every other app) ────────
_ALL_APPS = (["OVERALL", "BTC", "GLDM", "GDXM"] + ticker_config.APP_KEYS
             + ["LEVERAGED", "DAILYAUDIT", "HEALTH", "TARGETBOOK",
                "EXECUTEDBOOK", "ASSISTANT"])
_APP_LABELS = {"OVERALL": "🧭  Overall Trading", "BTC": "₿  Bitcoin (BTC)",
               "GLDM": "🥇  Gold Trend (GLDM·UGL)",
               "GDXM": "⛏️  Gold Miners (GDX·NUGT)",
               "LEVERAGED": "⚡  Leveraged Assets",
               "DAILYAUDIT": "🕵️  Daily Audit",
               "HEALTH": "🩺  Strategy Health",
               "TARGETBOOK": "📋  Target Book (IBKR)",
               "EXECUTEDBOOK": "✅  Executed Book",
               "ASSISTANT": "🤖  AI Assistant"}
for _k, _c in ticker_config.CONFIGS.items():
    _APP_LABELS[_k] = f"{_c.emoji}  {_c.key} · {_c.name.split('(')[0].strip()[:22]}"
if st.session_state.get("gldm_active_app") not in _ALL_APPS:
    st.session_state["gldm_active_app"] = "LEVERAGED"
with st.sidebar:
    st.radio("**Application**", options=_ALL_APPS,
             format_func=lambda x: _APP_LABELS.get(x, x), key="gldm_active_app")
    st.markdown("---")
    st.caption("_Grades the vehicle, not the direction: how costly each "
               "leveraged wrapper is as a way to hold its underlying._")


# ── prices ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False, max_entries=8)
def _frame(pair_key: str, end_iso: str) -> pd.DataFrame:
    """One pair's aligned closes, matching what that pair's own tab plots.

    Four of the five read their app's committed macro CSV and stop there, which
    is exactly what those tabs do.  The BTC pair is the exception: its tab tops
    the versioned ``data/backtest`` CSVs up from yfinance so the chart reaches
    the latest close rather than waiting on the nightly data job, so this does
    the same — otherwise MSTU's verdict here would sit a session behind the
    identical verdict on the BTC app's tab, with nothing on screen to explain
    the difference.  The top-up is best-effort: if the fetch fails, the
    committed frame is served and the board's *as of* column says so.
    """
    pair = L.PAIRS[pair_key]
    df = L.load_pair_csv(pair, _REPO_ROOT / "data")
    if pair.source != "backtest" or df.empty:
        return df
    return _yf_topup(df, pair, end_iso)


def _yf_topup(df: pd.DataFrame, pair: L.LevPair, end_iso: str) -> pd.DataFrame:
    """Extend both legs with yfinance closes past the committed snapshot."""
    end_ts = pd.Timestamp(end_iso)
    if df.index.max() >= end_ts:
        return df
    try:
        import yfinance as yf
    except Exception:
        return df
    legs = {}
    for ticker, col in ((pair.base, "BASE"), (pair.lev, "LEV")):
        try:
            raw = yf.download(ticker,
                              start=(df.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                              end=(end_ts + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
                              progress=False, auto_adjust=True)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0] for c in raw.columns]
            raw.index = pd.DatetimeIndex(raw.index).tz_localize(None).normalize()
            legs[col] = raw["Close"].dropna()
        except Exception:
            return df
    extra = pd.concat(legs, axis=1).dropna()
    extra = extra.loc[(extra.index > df.index.max()) & (extra.index <= end_ts)]
    if extra.empty:
        return df
    return pd.concat([df, extra[["BASE", "LEV"]]]).sort_index()


def _jump(app_key: str) -> None:
    """Send the router to that pair's own app — its tab has the full panel."""
    st.session_state["gldm_active_app"] = app_key


def _pct_col(label: str, help_text: str, digits: int = 1, signed: bool = True):
    fmt = f"%{'+' if signed else ''}.{digits}f%%"
    return st.column_config.NumberColumn(label, format=fmt, help=help_text)


# ── the board ───────────────────────────────────────────────────────────────
st.markdown("## ⚡ Leveraged Assets — vehicle verdict board")

_today = pd.Timestamp.now(tz="America/Chicago").normalize().tz_localize(None)
_frames = {k: _frame(k, _today.strftime("%Y-%m-%d")) for k in L.PAIRS}

c_hz, c_note = st.columns([1.6, 2.4])
with c_hz:
    horizon = st.slider("⏳ Holding period (trading sessions)",
                        min_value=1, max_value=L.MAX_HORIZON_DAYS,
                        value=L.HORIZON_DAYS, step=1, key="lev_board_hz",
                        help="Applied to every pair at once. 21 ≈ one month, "
                             "63 ≈ a quarter, 252 ≈ a year. The decay compounds "
                             "with the holding period, so every figure below "
                             "moves with this.")
with c_note:
    st.markdown("<div style='height:1.85rem'></div>", unsafe_allow_html=True)
    st.caption(f"≈ **{horizon / L.TRADING_DAYS * 12:.1f} months** of market time · "
               f"each pair priced at its own live trailing-20-session volatility · "
               f"verdicts sorted best vehicle first.")

board = L.verdict_board(_frames, horizon=horizon)
if board.empty:
    st.error("No leveraged pairs could be loaded — check the committed data files.")
    st.stop()

# ── verdict strip: one card per pair, colour + icon + word, never colour alone ─
cards = st.columns(len(board))
for col, (_, row) in zip(cards, board.iterrows()):
    edge = "—" if not np.isfinite(row["edge"]) else f"{row['edge'] * 100:+.1f}%".replace("-", "−")
    col.markdown(
        f"""
<div style="border-left:6px solid {row['color']};background:#f8fafc;border-radius:6px;
            padding:0.6rem 0.75rem;margin-bottom:0.35rem;">
  <div style="font-size:0.78rem;color:#64748b;letter-spacing:0.03em;">{row['pair']}</div>
  <div style="font-size:1.05rem;font-weight:700;color:{row['color']};line-height:1.35;">
    {row['icon']} {row['rating']}</div>
  <div style="font-size:0.86rem;color:#334155;">edge <b>{edge}</b> · {row['leverage']:g}×</div>
</div>
""", unsafe_allow_html=True)

st.caption("**Edge** = what the leveraged fund returns over the holding period "
           "at the underlying's recent pace, minus what the underlying itself "
           "returns. Positive means the leverage is paying for its own decay.")

# ── the numbers ─────────────────────────────────────────────────────────────
_PCT = ["breakeven", "base_pace", "lev_at_pace", "edge", "flat_outcome",
        "sigma_ann", "carry_ann"]
table = board.assign(**{c: board[c] * 100 for c in _PCT})
# Icon and word in one cell: a separate icon column needs a blank header, and
# the verdict must never read as colour (or a glyph) alone.
table["rating"] = table["icon"] + "  " + table["rating"]
table = table[["pair", "leverage", "rating", "edge", "breakeven",
               "base_pace", "lev_at_pace", "flat_outcome", "sigma_ann",
               "carry_ann", "beta", "r2", "asof"]]
st.dataframe(
    table, use_container_width=True, hide_index=True,
    column_config={
        "pair": st.column_config.TextColumn("Pair", help="Underlying → leveraged sibling"),
        "leverage": st.column_config.NumberColumn("k", format="%.0f×",
                                                  help="The fund's stated DAILY multiple."),
        "rating": st.column_config.TextColumn("Verdict", width="medium"),
        "edge": _pct_col(f"Edge ({horizon}s)",
                         "Leveraged return minus the underlying's over the holding "
                         "period, at the underlying's recent pace. This is what the "
                         "verdict grades."),
        "breakeven": _pct_col("Must gain",
                              "What the underlying has to return over the holding "
                              "period just for the fund to MATCH it — volatility "
                              "drag plus carry, compounded. Arithmetic, not a forecast."),
        "base_pace": _pct_col("Base at pace",
                              "Where the underlying lands over the holding period if "
                              "it keeps its trailing 60-session drift. An extrapolation."),
        "lev_at_pace": _pct_col("Fund at pace",
                                "The same path run through the k×-daily identity, decay "
                                "included — NOT k× the column to its left."),
        "flat_outcome": _pct_col("Base flat",
                                 "Pure decay: what the fund returns if the underlying "
                                 "goes nowhere for the whole holding period."),
        "sigma_ann": _pct_col("Vol",
                              "The underlying's trailing-20-session volatility, "
                              "annualised. Decay scales with its SQUARE.", 0, False),
        "carry_ann": _pct_col("Carry /yr",
                              "Fees, financing and rebalancing slippage measured from "
                              "the fund's tracking error over the trailing 60 sessions.",
                              1),
        "beta": st.column_config.NumberColumn(
            "β", format="%.2f×",
            help="Slope of the fund's daily return on the underlying's over the whole "
                 "shared history — the leverage actually delivered against its target."),
        "r2": st.column_config.NumberColumn("R²", format="%.3f"),
        "asof": st.column_config.DateColumn("As of", format="YYYY-MM-DD",
                                            help="Last close used. Sources refresh on "
                                                 "their own schedules, so these can differ."),
    })

# ── the verdict drawn, beside the drift it rests on ─────────────────────────
# Side by side for a reason beyond density: the scatter pins its y-axis to its
# x-axis 1:1 so the parity line is a true 45°, and plotly honours that by
# widening whichever axis has the spare pixels.  Full-container width turns a
# 2 pp spread of data into a 5 pp axis and buries the five dots in whitespace;
# a half-width column keeps the panel near-square and the dots apart.
c_fig, c_drift = st.columns([1.05, 1])
with c_fig:
    st.markdown("##### 📐 Drift against the hurdle it has to clear")
    st.plotly_chart(L.make_board_figure(board), use_container_width=True,
                    key="lev_board_scatter",
                    config={"scrollZoom": True, "displaylogo": False,
                            "modeBarButtonsToRemove": ["select2d", "lasso2d"]})
    st.caption("Both axes are log return per session on one shared scale, so the "
               "dashed 45° line IS breakeven: which side of it a pair sits on is "
               "the sign of its edge, and the distance is how much per session. "
               "Per session is why this panel alone does not move with the "
               "holding period — a longer hold scales every edge by the same "
               "factor without moving a dot.")
with c_drift:
    st.markdown("##### 📊 Realised drift by lookback")
    _drift = L.drift_matrix(board) * 100
    st.dataframe(
        _drift, use_container_width=True,
        column_config={c: st.column_config.NumberColumn(c, format="%+.1f%%")
                       for c in _drift.columns})
    st.caption("Total move of each underlying over the trailing window — what "
               "already happened, not a forecast. The verdict extrapolates the "
               "**60-session** column, so a row whose columns disagree is a "
               "verdict resting on one choice of window.")

# ── download + the standing caveat, kept to one line ────────────────────────
d1, d2 = st.columns([1, 3])
with d1:
    st.download_button("⬇️ Download board as CSV",
                       data=board.drop(columns=["color"]).to_csv(index=False).encode("utf-8"),
                       file_name=f"leveraged_board_{horizon}s_{_today:%Y%m%d}.csv",
                       mime="text/csv", use_container_width=True)
with d2:
    st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)
    st.caption("⚠️ This grades **cost, not direction** — how expensive each wrapper "
               "is as a way to hold its underlying, assuming that underlying keeps "
               "its recent pace. It is not a trade signal.")

# Streamlit cannot preselect a tab from code, so these switch apps and the
# viewer clicks the tab — the help text says so rather than promising a landing
# spot the button cannot deliver.
st.markdown("###### Open a pair's own app")
jump = st.columns(len(board))
for col, (_, row) in zip(jump, board.iterrows()):
    col.button(f"{row['base']}-{row['lev']} →", key=f"lev_jump_{row['key']}",
               use_container_width=True, on_click=_jump, args=(row["key"],),
               help=f"Switch to the {_APP_LABELS.get(row['key'], row['key']).strip()} "
                    f"app, then open its **📉 {row['base']}-{row['lev']} Plot** tab: "
                    "price overlay, gap panel, breakeven curve and the arithmetic "
                    "behind this verdict.")
