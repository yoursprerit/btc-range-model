"""Overall Trading — the cross-asset combined engine.

This module is the shared backbone of the **Overall Trading** app
(``app/overall_app.py``).  It runs *every instrument the individual apps trade*
through ONE unified daily engine (``ticker_core`` + ``backtest_ticker``) so the
signals, positions and back-tests are directly comparable on a single screen and
can be blended into one portfolio.

Each app produces ONE signal but may trade several instruments off it — its 1×
primary plus higher-beta / leveraged siblings — exactly as the dedicated apps do:

    App     Signal     Traded off that signal
    ─────   ──────     ────────────────────────────────────────────
    BTC     BTC        BTC (1×) · MSTR (BTC-proxy) · MSTU (2× MSTR) · ETH (spot ETH)
    Gold    GLDM       GLDM (1×) · GDX (miners) · UGL (2× gold)
    XLE     XLE        XLE (1×) · OIH (oil services, high-beta)
    SOXX    SOXX       SOXX
    GRID    GRID       GRID
    REMX    REMX       REMX
    WGMI    WGMI       WGMI

Every sibling is driven by its PARENT's signal (never its own) — the higher-beta
name is steered by the cleaner primary read, the way the Gold app trades GDX/UGL
off gold and the BTC app trades MSTR/MSTU off Bitcoin.  The leveraged sleeves are
included in the optimiser "as and when they make sense": each carries a tighter
weight cap (see ``CAP_BY_KEY``) so the max-Sharpe search only leans on them when
the extra beta actually improves risk-adjusted return.

The six ETF apps reuse their exact ``ticker_config`` entries.  Gold reuses the
Gold app's actual Divergence Pure-Regime strategy; BTC uses a 30-day trend
filter (the MA30 window its app references) because the BTC app's hourly
divergence signal cannot be reproduced daily.  None of the original app files are
imported or modified.
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

_APP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _APP_DIR.parent
for _p in (str(_APP_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ticker_core as tc                     # noqa: E402
import backtest_ticker as bt                 # noqa: E402
import ticker_config                         # noqa: E402
import freshness as _frs                     # noqa: E402
from ticker_config import TickerConfig, get_config, _STD_PERIODS   # noqa: E402

# Capability flag: this build's live_exit_keys re-runs each mode's real trend
# condition (dual_ma fast/slow cross, etc.) instead of a naive price-vs-line
# proxy, and _asset_result attaches cfg/close_hist for it.  The app's _stale_core
# guard checks this so a hot-reload against an older cached module is refreshed.
LIVE_EXIT_MODE_AWARE = True

# Capability flag: this build's live_entry_keys also scores DIVERGENCE apps —
# it re-runs the real entry gate (compute_trend_signatures) with the live price
# as a provisional completed bar, using the pending-bar model bands run_asset
# attaches (sig_completed / pending_pred).  Checked by the app's _stale_core
# guard alongside LIVE_EXIT_MODE_AWARE.
LIVE_ENTRY_DIVERGENCE_AWARE = True


def _warmup_imports() -> None:
    """Force every heavy, lazily-imported module into ``sys.modules`` on a
    SINGLE thread, so the concurrent per-app thread pool in ``run_universe`` can
    never trigger a first-time import from two threads at once — which Python's
    import lock turns into a deadlock (observed on ``sklearn.linear_model._huber``
    and the CT-model engine).  Idempotent and best-effort."""
    try:
        from sklearn.linear_model import (RidgeCV, HuberRegressor,   # noqa: F401
                                          QuantileRegressor)
        from sklearn.ensemble import GradientBoostingRegressor       # noqa: F401
        from sklearn.preprocessing import StandardScaler             # noqa: F401
        from sklearn.pipeline import Pipeline                        # noqa: F401
    except Exception:
        pass
    for _eng in ("btc_ct_engine", "gldm_engine"):                    # load CT/gold engines + models
        try:
            __import__(_eng)
        except Exception:
            pass


_warmup_imports()   # run at import time (single-threaded), before any thread pool


# ════════════════════════════════════════════════════════════════════════
# CONFIGS — one per signal app; each may trade its 1× primary + siblings.
# ════════════════════════════════════════════════════════════════════════
# BTC — daily MA30 trend filter (model-free, robust regime). Per-asset stops
# (2026-07e): BTC & MSTR signal-exit-only (no fixed stop), MSTU −6%.
# The dedicated BTC app runs an *hourly* divergence Pure-Regime system that uses
# a 30-day moving average (``above_ma30``) as its trend-regime gate; that hourly
# CT-anchored signal can't be reproduced in this daily engine, so BTC is traded
# on a daily 30-day trend filter — the SAME 30-bar window the BTC app references,
# so the number is consistent with what that app shows.  Trades BTC (1×) plus
# MSTR (BTC-proxy equity) and MSTU (2× MSTR) off the signal.
BTC_CFG = TickerConfig(
    key="BTC", name="Bitcoin", emoji="₿",
    blurb=("Spot Bitcoin drives the signal; the strategy trades BTC and its "
           "higher-beta siblings MSTR and MSTU off it."),
    accent="#f7931a", accent_dark="#b45309", accent_bg="#fff7ed", accent_bg2="#ffedd5",
    primary_symbol="BTC-USD",
    macro_syms={"eth": "ETH-USD", "spx": "^GSPC", "ndx": "^NDX",
                "vix": "^VIX", "gold": "GC=F", "dxy": "DX-Y.NYB", "tnx": "^TNX"},
    extra_syms={"mstr": "MSTR", "mstu": "MSTU", "eth": "ETH-USD"},
    sentiment=[("spx_close", "mom", +1.0), ("ndx_close", "mom", +1.0),
               ("vix_close", "lvl", -1.0), ("px_close", "mom", +1.0)],
    sentiment_label="Crypto/risk macro sentiment",
    traded_assets=[("BTC", "px_close"), ("MSTR", "mstr_close"), ("MSTU", "mstu_close"),
                   ("ETH", "eth_close")],
    asset_labels={"px_close": "BTC · Bitcoin", "mstr_close": "MSTR · MicroStrategy",
                  "mstu_close": "MSTU · 2× MSTR", "eth_close": "ETH · Ethereum"},
    strategy_mode="divergence", strategy_name="BTC Divergence Pure-Regime",
    ma_window=30, fixed_stop=0.03,
    u1_errhi_min=0.013, d2_errhi_max=-0.013, d1_errlo_min=0.005, v_errlo_min=0.50,
    use_d1_exit=False, hl_band_pct=0.02,
    fetch_start="2015-01-01", oos_start="2021-01-01", periods=_STD_PERIODS,
    day_up_thresh=0.02, day_down_thresh=-0.02,
    results_note=("Divergence Pure-Regime — the SAME logic and thresholds as the "
                  "BTC app (U1 err_hi>+1.3% + ≥2 high-breaks, D2 exit err_hi<−1.3%, "
                  "MA30 regime gate; per-asset stops 2026-07e — BTC & MSTR "
                  "signal-exit-only, MSTU −6%), so the live entry/exit signal "
                  "tracks the BTC app. NOTE: those thresholds are calibrated for "
                  "the app's HOURLY CT-model; on a daily RidgeCV rebuild the H/L "
                  "predictions are noisier, so the daily back-test is weak and "
                  "BTC/MSTR/MSTU earn ~0 weight in the blend — the alignment is "
                  "for signal consistency, not daily performance."),
)

# GOLD — the middle-path split, surfaced as TWO parent apps (both driven by
# the shared GLDM signal via gldm_engine; these configs are display metadata —
# the dispatch in run_universe routes both to the real gold engine):
#   🥇 GLDM (Gold Trend)  — GLDM (1×) & UGL (2×) on a dual-MA 25/100 crossover
#   ⛏️ GDXM (Gold Miners) — GDX & NUGT on the Divergence Pure-Regime signal
GLDM_CFG = TickerConfig(
    key="GLDM", name="Gold Trend (SPDR Gold MiniShares)", emoji="🥇",
    blurb=("Spot gold via GLDM drives the signal; the Gold Trend app trades "
           "GLDM (1× core) and UGL (2× gold) on a SOXX-style dual-MA 25/100 "
           "crossover of the gold close."),
    accent="#b8860b", accent_dark="#8b6508", accent_bg="#fffbeb", accent_bg2="#fef3c7",
    primary_symbol="GLDM",
    macro_syms={"gc": "GC=F", "slv": "SLV", "dxy": "DX-Y.NYB",
                "tnx": "^TNX", "vix": "^VIX", "spx": "^GSPC"},
    extra_syms={"ugl": "UGL"},
    sentiment=[("gc_close", "mom", +1.0), ("dxy_close", "mom", -1.0),
               ("vix_close", "lvl", -1.0), ("px_close", "mom", +1.0)],
    sentiment_label="Gold macro sentiment",
    traded_assets=[("GLDM", "px_close"), ("UGL", "ugl_close")],
    asset_labels={"px_close": "GLDM · Gold", "ugl_close": "UGL · 2× Gold"},
    strategy_mode="dual_ma", strategy_name="Gold Dual-MA Trend",
    ma_window=100, ma_fast=25, ma_slow=100, fixed_stop=0.03,
    hl_band_pct=0.008,
    fetch_start="2018-06-26", oos_start="2021-01-01", periods=_STD_PERIODS,
    day_up_thresh=0.006, day_down_thresh=-0.006,
    results_note=("Gold Trend — half of the gold middle path: GLDM (1×) and "
                  "UGL (2×) ride a SOXX-style 25/100 dual-MA crossover on the "
                  "GLDM close with −3% stops. OOS 2021→now: GLDM +137% / −19% / "
                  "Sharpe 1.08, UGL +302% / −38% / 0.96 — beats both B&H and "
                  "the divergence variant on return AND Sharpe for these two "
                  "smooth-trending sleeves. The miners half (GDX/NUGT) trades "
                  "the divergence signal under the ⛏️ Gold Miners app."),
)

GDXM_CFG = TickerConfig(
    key="GDXM", name="Gold Miners (GDX & NUGT)", emoji="⛏️",
    blurb=("The gold-miners half of the gold middle path: GDX and NUGT trade "
           "the Divergence Pure-Regime signal derived from GLDM's causal daily "
           "H/L model."),
    accent="#8b6508", accent_dark="#6b4e06", accent_bg="#fffbeb", accent_bg2="#fef3c7",
    primary_symbol="GDX",
    macro_syms={"gc": "GC=F", "slv": "SLV", "dxy": "DX-Y.NYB",
                "tnx": "^TNX", "vix": "^VIX", "spx": "^GSPC"},
    extra_syms={"nugt": "NUGT"},
    sentiment=[("gc_close", "mom", +1.0), ("dxy_close", "mom", -1.0),
               ("vix_close", "lvl", -1.0), ("px_close", "mom", +1.0)],
    sentiment_label="Gold macro sentiment",
    traded_assets=[("GDX", "px_close"), ("NUGT", "nugt_close")],
    asset_labels={"px_close": "GDX · Gold Miners", "nugt_close": "NUGT · 2× Gold Miners"},
    strategy_mode="divergence", strategy_name="Gold-Miners Divergence Pure-Regime",
    ma_window=20, fixed_stop=0.03,
    stop_by_asset={"nugt_close": 0.05},
    u1_errhi_min=0.15, d2_errhi_max=-0.10, d1_errlo_min=0.15, v_errlo_min=1.00,
    use_d1_exit=False, hl_band_pct=0.012,
    fetch_start="2018-06-26", oos_start="2021-01-01", periods=_STD_PERIODS,
    day_up_thresh=0.008, day_down_thresh=-0.008,
    results_note=("Gold Miners — the divergence half of the gold middle path: "
                  "GDX (−3%) and NUGT (−5%) trade the causal GLDM divergence "
                  "signal (U1 +0.15 / D2 −0.10 / V 1.0). OOS 2021→now: GDX "
                  "+103% / −22% / Sharpe 0.85, NUGT +276% / −38% / 0.90 — and "
                  "the system stayed net positive (+12%/+11%) through the "
                  "2021-22 chop, hedging the trend sleeves' dips. The smooth "
                  "trenders (GLDM/UGL) ride the dual-MA under 🥇 Gold Trend."),
)

def overall_config(key: str) -> TickerConfig:
    """Return the unified parent config for one signal app.

    Every app is aligned to the strategy its own app trades: BTC via the CT
    engine (see run_universe), GLDM via its Divergence Pure-Regime config, and
    the six ETF apps via their exact ``ticker_config`` entries — SOXX on its
    25/100 dual-MA crossover, GRID on its MACD 10/20/9 filter, WGMI on its
    50-day SMA + volatility filter, REMX on its 50/200 dual-MA golden cross, XLE
    on its energy divergence.  The Overall app therefore matches each source app
    bar-for-bar.
    """
    if key == "BTC":
        return BTC_CFG
    if key == "GLDM":
        return GLDM_CFG
    if key == "GDXM":
        return GDXM_CFG
    return get_config(key)


# Parent (signal-app) display order.
PARENT_KEYS = ["BTC", "GLDM", "GDXM", "SOXX", "GRID", "XLE", "REMX", "WGMI",
               "PBW", "ARTY"]


def all_configs() -> list[TickerConfig]:
    return [overall_config(k) for k in PARENT_KEYS]


def oos_start_span() -> tuple[str, str]:
    """(earliest, latest) out-of-sample start YEAR across the traded universe,
    read from each instrument's ``cfg.oos_start`` — so the app's "out-of-sample
    from …" framing tracks the configs instead of a hardcoded year and correctly
    reflects that starts are staggered (most ETFs 2021, newer sleeves later)."""
    yrs = sorted({c.oos_start[:4] for c in all_configs() if getattr(c, "oos_start", None)})
    return (yrs[0], yrs[-1]) if yrs else ("", "")


# ── per-instrument metadata (label, class, weight cap) ────────────────────
# kind ∈ {"core", "beta" (high-beta equity), "lev" (leveraged 2×)}.
# Leveraged sleeves get the tightest cap so the optimiser only leans on them
# when the extra beta genuinely improves risk-adjusted return.
ASSET_META = {
    "BTC":  dict(name="Bitcoin",        kind="core"),
    "MSTR": dict(name="MicroStrategy",  kind="beta"),
    "MSTU": dict(name="2× MSTR",        kind="lev"),
    "ETH":  dict(name="Ethereum",        kind="core"),
    "GLDM": dict(name="Gold (GLDM)",    kind="core"),
    "GDX":  dict(name="Gold Miners",    kind="beta"),
    "UGL":  dict(name="2× Gold",        kind="lev"),
    "NUGT": dict(name="2× Gold Miners", kind="lev"),
    "SOXX": dict(name="Semiconductors", kind="core"),
    "SOXL": dict(name="3× Semiconductors", kind="lev"),
    "GRID": dict(name="Grid Infra",     kind="core"),
    "XLE":  dict(name="Energy",         kind="core"),
    "OIH":  dict(name="Oil Services",   kind="beta"),
    "ERX":  dict(name="2× Energy",      kind="lev"),
    "REMX": dict(name="Rare-Earth Metals", kind="core"),
    "WGMI": dict(name="Bitcoin Miners", kind="beta"),
    "PBW":  dict(name="Clean Energy",   kind="core"),
    "ARTY": dict(name="AI & Tech",      kind="core"),
}
KIND_EMOJI = {"core": "", "beta": "⚡", "lev": "🔺"}
CAP_BY_KIND = {"core": 0.30, "beta": 0.18, "lev": 0.10}
CAP_BY_KEY = {k: CAP_BY_KIND[m["kind"]] for k, m in ASSET_META.items()}


# ── fundamental forward-view overlay (mid-2026 sector outlook) ─────────────
# A per-instrument conviction multiplier (>1 overweight · <1 underweight) that
# tilts each profile's historically-optimal blend toward the strongest secular
# growth themes, then re-water-fills to the same caps.  It is applied on top of
# the quant optimum (see ``optimize_weights(fundamental=True)``) so the strategy,
# allocation and back-test reflect the forward view, not just past risk/return.
FUNDAMENTAL_VIEW = {
    "SOXX": 1.40, "SOXL": 1.40, "ARTY": 1.40,   # AI / semiconductor supercycle (SOXL = 3× semis)
    "BTC": 1.40, "MSTR": 1.40, "MSTU": 1.40,    # crypto institutional era
    "ETH": 1.40,                                 # spot ETH — same crypto thesis
    "GLDM": 1.30, "GDX": 1.40, "UGL": 1.40, "NUGT": 1.40,   # structural gold bull (NUGT = 2× miners)
    "GRID": 1.40,                                # electrification / grid capex
    "WGMI": 1.30,                                # miners' AI/HPC pivot
    "REMX": 1.10,                                # rare-earth supply squeeze
    "XLE": 1.00, "ERX": 1.00,                    # energy — gas ok, oil soft (ERX = 2× energy)
    "OIH": 0.50, "PBW": 0.40,                    # no catalyst / policy headwinds
}
FUNDAMENTAL_VIEW_NOTE = (
    "Mid-2026 sector outlook: overweight AI/semis (SOXX, ARTY), the crypto "
    "institutional era (BTC/MSTR/MSTU/ETH, WGMI), the structural gold bull "
    "(GLDM/GDX/UGL) and electrification (GRID); underweight clean energy (PBW) "
    "and oil services (OIH).")


# ── risk profiles — how hard to lean on the high-beta / leveraged proxies ──
# Beta/leveraged sleeves carry huge returns but poor stand-alone Sharpe (deep
# drawdowns), so loading them boosts raw return at the cost of risk-adjusted
# return.  A profile bundles the per-kind caps, the optimiser objective and the
# drawdown budget so the user can dial that trade-off:
#   Balanced   — hold Sharpe near its max (~unchanged behaviour; default)
#   Growth     — maximise return inside a −22% drawdown budget (more β / 2×)
#   Aggressive — maximise return inside a −38% budget (heavy β / 2×)
RISK_PROFILES = {
    "Balanced": dict(
        caps={"core": 0.30, "beta": 0.18, "lev": 0.10},
        objective="balanced", mdd_floor=-0.35,
        blurb="Holds Sharpe near its max — the historically best risk-adjusted blend."),
    "Growth": dict(
        caps={"core": 0.30, "beta": 0.25, "lev": 0.18},
        objective="max_return", mdd_floor=-0.22,
        blurb="Leans harder on the high-beta / leveraged proxies to maximise "
              "return inside a −22% drawdown budget."),
    "Aggressive": dict(
        caps={"core": 0.35, "beta": 0.40, "lev": 0.35},
        objective="max_return", mdd_floor=-0.38,
        blurb="Maximum return inside a −38% budget — heavy β / 2× exposure; "
              "highest return, deepest drawdowns, lower Sharpe."),
}
DEFAULT_PROFILE = "Balanced"


def caps_for(profile: str) -> dict:
    kc = RISK_PROFILES.get(profile, RISK_PROFILES[DEFAULT_PROFILE])["caps"]
    return {k: kc[m["kind"]] for k, m in ASSET_META.items()}


def _label(key: str) -> str:
    return ASSET_META.get(key, {}).get("name", key)


# ── SATA — the idle-cash park (Strive Variable Rate Series A Perpetual Pfd) ──
# The same instrument the BTC app parks idle capital in: a US-listed preferred
# paying a ~13% annual coupon as a daily dividend on $100 par (≈13.88% effective
# when reinvested).  Any capital not deployed to a live position sits in SATA and
# earns this daily yield, rather than dead cash.
SATA = dict(ticker="SATA", name="Strive Variable-Rate Preferred (idle cash)",
            annual_rate=0.13, biz_days=250, par=100.0)
SATA_DAILY = SATA["annual_rate"] / SATA["biz_days"]     # ≈0.00052 / business day


# ── entry-priority weighting ──────────────────────────────────────────────
# When several instruments signal entry at once (or one fires while others are
# already held), rank them by their ability to maximise return in the CURRENT
# tape and size the book toward the winners.  A blend of live momentum, macro
# sentiment, the strategy's back-tested win-rate and its risk-adjusted edge.
PRIORITY_WEIGHTS = dict(momentum=0.28, sentiment=0.24, win_rate=0.20,
                        sharpe=0.18, regime=0.10)


def _minmax(vals: dict) -> dict:
    xs = [v for v in vals.values() if v == v]          # drop NaN
    if not xs:
        return {k: 0.5 for k in vals}
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-12:
        return {k: 0.5 for k in vals}
    return {k: (0.5 if v != v else (v - lo) / (hi - lo)) for k, v in vals.items()}


def compute_priorities(results: list[dict], keys: list[str]) -> dict:
    """Priority score in [0,1] for each candidate key, plus its components.
    Normalised across the competing candidate set so the ranking is relative to
    what is actually on the table today."""
    by = {r["key"]: r for r in results}
    keys = [k for k in keys if k in by]
    if not keys:
        return {}
    mom = _minmax({k: by[k].get("mom", np.nan) for k in keys})
    shp = _minmax({k: by[k]["metrics"].get("sharpe", np.nan) for k in keys})
    out = {}
    for k in keys:
        r = by[k]
        sent = r.get("sentiment", np.nan)
        sent01 = 0.5 if sent != sent else min(max(sent / 100.0, 0.0), 1.0)
        wr01 = min(max(r.get("win_rate", 0.0) / 100.0, 0.0), 1.0)
        reg = 1.0 if r.get("bull_regime") else 0.0
        comp = dict(momentum=mom[k], sentiment=sent01, win_rate=wr01,
                    sharpe=shp[k], regime=reg)
        score = sum(PRIORITY_WEIGHTS[c] * comp[c] for c in PRIORITY_WEIGHTS)
        out[k] = dict(score=float(score), **comp)
    return out


# ════════════════════════════════════════════════════════════════════════
# PER-ASSET RUN — fetch → predict → simulate each traded instrument
# ════════════════════════════════════════════════════════════════════════
def _load_daily(cfg: TickerConfig) -> pd.DataFrame:
    """Daily OHLCV+macro for one config, via the QUALITY-GATED pinned loader
    (``app/data_gate.py``): once a fetch passes quality control and covers the
    last completed session it is pinned as the sleeve's snapshot, and every
    later load that day reuses the identical history — so the back-tests stop
    drifting between page loads as Yahoo's free feed wobbles.  A fetch that
    fails validation (missing macro columns, shrunken history, rewritten
    returns, absurd prints) is rejected and the last known-good snapshot is
    served instead; every decision lands in ``runtime/dataset_audit.json``.
    Traded-sibling price columns are forward-filled so equity assets don't leave
    NaN gaps on days their market is shut (e.g. weekends when BTC still trades)."""
    df = pd.DataFrame()
    try:
        import data_gate as dg
        df = dg.gated_daily(dg.ticker_spec(cfg),
                            fetch_full=lambda: tc.fetch_daily(cfg),
                            fetch_recent=lambda: tc.fetch_recent_daily(cfg),
                            consumer="OVERALL")
    except Exception:
        try:
            df = tc.fetch_daily(cfg)          # gate unavailable — legacy path
        except Exception:
            df = pd.DataFrame()
    if df is None or df.empty or "px_close" not in df.columns or len(df) < 260:
        csv = tc.cache_paths(cfg)["daily"]
        if Path(csv).exists():
            df = pd.read_csv(csv, index_col=0, parse_dates=True)
            if "px_close" not in df.columns:
                pref = f"{cfg.primary_symbol.lower()}_"
                df = df.rename(columns={c: "px_" + c[len(pref):]
                                        for c in df.columns if c.startswith(pref)})
    if df is None or df.empty:
        return pd.DataFrame()
    # forward-fill traded-sibling prices over their market's closed days
    for nm in cfg.extra_syms:
        for suff in ("open", "high", "low", "close"):
            col = f"{nm}_{suff}"
            if col in df.columns:
                df[col] = df[col].ffill()
    return df


def _net_decision(cfg: TickerConfig, sigs: dict | None, in_pos: bool,
                  last_close: float, ma_val: float | None,
                  long_now: bool | None = None) -> dict:
    """Resolve a single actionable state for the PARENT signal, reconciling the
    instantaneous read with the strategy's actual executed position.
    tone ∈ {buy, hold, exit, watch, flat}."""
    if cfg.is_trend:
        # ``long_now`` is the engine's actual trend signal for this bar (the one
        # source of truth across ma / dual_ma / macd / ma_vol); fall back to the
        # close-vs-line read only if it wasn't supplied.
        above = long_now if long_now is not None else (
            (ma_val is not None) and (last_close > ma_val))
        if in_pos:
            # The MA filter decides at the close and acts on the NEXT bar, so a
            # position is still open the day its close first drops below the
            # trend (it exits next bar).  One convention for that state across
            # EVERY engine family and app: "EXIT NEXT BAR — <cause>", red,
            # tone "exit" — matching the divergence engines and the source
            # apps' net signals word for word, so no two surfaces can describe
            # the identical committed pending exit differently.
            if above:
                return dict(state="HOLD", label="LONG — HOLDING", ico="🟢", tone="hold")
            return dict(state="EXIT", label="EXIT NEXT BAR — BELOW TREND",
                        ico="🔴", tone="exit", exits_next_bar=True)
        if above:
            return dict(state="ENTRY", label="ENTER — ABOVE TREND", ico="🟢", tone="buy")
        return dict(state="FLAT", label="FLAT — BELOW TREND", ico="⬜", tone="flat")
    # Divergence — replicate the backtest's exit-overrides-entry precedence
    # (backtest_ticker.simulate: exit on d2/d3/d1-exit; enter only when the gate
    # fires AND no exit is active) so the displayed decision can never say "buy"
    # while an exit signal is live — matching each app's net_signal.
    d1 = bool((sigs or {}).get("d1_triggered"))
    d2 = bool((sigs or {}).get("d2_triggered"))
    d3 = bool((sigs or {}).get("d3_triggered"))
    exit_sig = d2 or d3 or (d1 and cfg.use_d1_exit)
    entry_gate = bool((sigs or {}).get("entry_triggered"))
    why = ("D3 exhaustion" if d3 else "D2 momentum fade" if d2 else "D1 downtrend")
    if in_pos:
        if exit_sig:
            # Divergence exits are decided at the close and executed on the NEXT
            # bar (backtest_ticker.simulate lags signals one bar), exactly like
            # the trend family's pending exit above — same flag, same
            # "EXIT NEXT BAR — <cause>" label convention.
            return dict(state="EXIT", label=f"EXIT NEXT BAR — {why}",
                        ico="🔴", tone="exit", exits_next_bar=True)
        return dict(state="HOLD", label="LONG — HOLDING", ico="🟢", tone="hold")
    # flat — an active exit signal blocks entry (net-exit), even if U1 is firing
    if exit_sig:
        return dict(state="AVOID", label=f"STAND ASIDE — EXIT ACTIVE ({why})",
                    ico="🟠", tone="watch")
    if entry_gate:
        return dict(state="ENTRY", label="ENTER — PURE-REGIME BUY", ico="🟢", tone="buy")
    if bool((sigs or {}).get("u1_triggered")):
        return dict(state="WATCH", label="WATCH — UPTREND (GATE PENDING)", ico="🟡", tone="watch")
    if d1:
        return dict(state="AVOID", label="STAND ASIDE — DOWNTREND (D1)", ico="🟠", tone="watch")
    return dict(state="FLAT", label="FLAT — NO SIGNAL", ico="⬜", tone="flat")


def _asset_result(cfg, label, col, r, daily, dec, alert, bull, sent, ma_val, dchg, mom,
                  hist=None):
    """Assemble one traded-instrument result dict from a run_strategy output."""
    dates = pd.to_datetime(pd.Series(r["dates"]))
    strat = np.asarray(r["strat"], float)
    ret = pd.Series(np.diff(strat) / strat[:-1], index=dates.iloc[1:]).rename(label)
    pos_series = pd.Series(np.asarray(r["pos"], float), index=dates).rename(label)
    last_px = float(daily[col].dropna().iloc[-1]) if col in daily else np.nan
    as_of = pd.Timestamp(dates.iloc[-1])

    # per-asset stop (e.g. SOXL trades no stop) read from the data field, not the
    # stop_for method, so a hot-reloaded/stale config instance can't AttributeError.
    _stop = getattr(cfg, "stop_by_asset", {}).get(col, cfg.fixed_stop)
    pos = dict(in_pos=bool(r.get("in_pos_now")), entry_px=r.get("entry_px"),
               entry_date=r.get("entry_date"), upnl=None, stop_px=None,
               days=None, dist_stop=None)
    if pos["in_pos"] and pos["entry_px"]:
        e_px = float(pos["entry_px"]); e_dt = pd.Timestamp(pos["entry_date"])
        pos["upnl"] = (last_px / e_px - 1) * 100
        pos["days"] = int((as_of - e_dt).days)
        if _stop < 0.999:
            pos["stop_px"] = e_px * (1 - _stop)
            pos["dist_stop"] = (last_px / pos["stop_px"] - 1) * 100
    last_trade = r["trade_log"][-1] if r.get("trade_log") else None

    meta = ASSET_META.get(label, dict(name=label, kind="core"))
    m = bt._metrics(strat, r["dates"]); bh = bt._metrics(r["bh"], r["dates"])
    wr = float((r["trades"] > 0).mean() * 100) if len(r["trades"]) else 0.0
    return dict(
        key=label, parent=cfg.key, name=meta["name"], kind=meta["kind"],
        emoji=cfg.emoji, kemoji=KIND_EMOJI[meta["kind"]], accent=cfg.accent,
        cap=CAP_BY_KEY.get(label, 0.30),
        last_close=last_px, dchg=dchg, ma_val=ma_val,
        sentiment=sent, decision=dec, alert=alert, bull_regime=bull,
        pos=pos, last_trade=last_trade, mom=mom,
        metrics=m, bh_metrics=bh, win_rate=wr, n_trades=int(len(r["trades"])),
        ret=ret, pos_series=pos_series, strat=strat, dates=dates, r=r, as_of=as_of,
        mode=cfg.strategy_mode, ma_window=cfg.ma_window, stop=_stop,
        engine_label=cfg.engine_label(),
        # committed close history + config so the live-price exit check can re-run
        # the mode's real trend condition (e.g. dual_ma's fast/slow SMA cross)
        # rather than a naive price-vs-line proxy.  Completed sessions only —
        # trend_long_now_live APPENDS the live price as the provisional newest
        # close, so an in-progress bar left here would count today twice.
        cfg=cfg,
        close_hist=(hist if hist is not None else daily)["px_close"].to_numpy(float),
    )


def run_asset(cfg: TickerConfig) -> list[dict]:
    """Run every instrument this app trades. Returns one result dict per traded
    asset (primary + siblings), all sharing the parent's signal/decision."""
    daily = _load_daily(cfg)
    if daily is None or daily.empty or "px_close" not in daily.columns:
        return []
    # COMMITTED signals come from completed sessions only, exactly like each
    # dedicated app (they build predictions on drop_in_progress_us_bar(daily)).
    # The gated frame overlays Yahoo's in-progress *today* row during US market
    # hours for live prices — feeding that half-formed bar into the signature
    # math lets D2/D3 (and the trend cross) flip intraday off a partial
    # high/low, so the Overall card would show an EXIT the source app doesn't.
    # The live-price provisional layer (live_exit_keys / live_entry_keys)
    # handles intraday reads separately.  ``daily`` keeps the partial bar for
    # display prices.
    hist = _frs.drop_in_progress_us_bar(daily)
    if hist is None or hist.empty or "px_close" not in hist.columns:
        hist = daily
    preds = bt.build_predictions(cfg, hist)
    sig = bt.precompute_signals(cfg, preds)

    # parent-level signal snapshot (shared by all traded instruments)
    last_close = float(daily["px_close"].iloc[-1])
    prev_close = float(daily["px_close"].iloc[-2]) if len(daily) > 1 else last_close
    dchg = (last_close / prev_close - 1) * 100 if prev_close else 0.0
    # trend line for display / live-exit, and the engine's actual long signal —
    # both on the completed frame (decisions are made at the close; the naive
    # partial-bar read would flicker intraday)
    ma_val = bt.trend_line_value(cfg, hist) if cfg.is_trend else None
    long_now = bt.trend_long_now(cfg, hist) if cfg.is_trend else None
    # common momentum read (distance above a 50-day SMA) for cross-asset priority
    ref_ma = float(daily["px_close"].tail(50).mean())
    mom = (last_close / ref_ma - 1) if ref_ma else 0.0
    completed = preds[preds["actual_high"].notna() & preds["actual_low"].notna()]
    # tail(150), not 45: the 60-bar median centering needs ≥90 bars of history
    # for the newest bar's signals to match bt.precompute_signals.
    sigs = tc.compute_trend_signatures(cfg, completed.tail(150)) if len(completed) >= 3 else None
    alert = (sigs or {}).get("alert_level", "NEUTRAL")
    bull = bool((sigs or {}).get("bull_regime",
                                 ma_val is not None and last_close > (ma_val or 0)))
    try:
        s = tc.macro_sentiment(cfg, daily)
        sent = float(s.dropna().iloc[-1]) if s.notna().any() else np.nan
    except Exception:
        sent = np.nan

    out = []
    for label, col in cfg.traded_assets:
        if col not in preds.columns or preds[col].notna().sum() < 60:
            continue
        # start each instrument at max(configured OOS, its first traded bar)
        valid = preds.loc[preds[col].notna(), "target_date"]
        start = max(pd.Timestamp(cfg.oos_start), pd.Timestamp(valid.iloc[0]))
        try:
            r = bt.run_strategy(cfg, preds, sig, col, oos_start=str(start.date()))
        except Exception:
            continue
        if len(r["dates"]) < 30:
            continue
        # each traded instrument shares the parent decision but has its own pos
        dec = _net_decision(cfg, sigs, bool(r.get("in_pos_now")), last_close, ma_val,
                            long_now=long_now)
        res = _asset_result(cfg, label, col, r, daily, dec, alert, bull,
                            sent, ma_val, dchg, mom, hist=hist)
        if cfg.strategy_mode == "divergence":
            # completed-signature tail + the model's bands for the pending bar,
            # so live_entry_keys can re-run the REAL divergence entry gate with
            # the live price as a provisional completed bar (the divergence
            # mirror of trend_long_now_live).  tail(150) matches the signature
            # computation above — the 60-bar median centering needs the runway.
            res["sig_completed"] = (
                completed.tail(150)[["close_asof", "pred_high", "pred_low",
                                     "actual_high", "actual_low",
                                     "target_date"]].copy()
                if len(completed) >= 3 else None)
            res["pending_pred"] = getattr(preds, "attrs", {}).get("pending_pred")
        out.append(res)
    return out


def run_universe() -> list[dict]:
    """Run every instrument across all apps; skip any that fail.

    BTC/MSTR/MSTU are run through the BTC app's ACTUAL CT-model engine
    (``btc_ct_engine``); every other app is run through the shared daily engine
    with its own tuned config, which is the exact engine those apps use.
    """
    from concurrent.futures import ThreadPoolExecutor
    import traceback

    _warmup_imports()   # ensure heavy imports are cached before threading

    def _one(cfg):
        try:
            if cfg.key == "BTC":
                import btc_ct_engine
                return cfg.key, btc_ct_engine.run_btc_ct(), None
            if cfg.key == "GLDM":
                import gldm_engine
                return cfg.key, gldm_engine.run_gldm_trend(), None
            if cfg.key == "GDXM":
                import gldm_engine
                return cfg.key, gldm_engine.run_gldm_miners(), None
            return cfg.key, run_asset(cfg), None
        except Exception:
            return cfg.key, [], traceback.format_exc().strip().splitlines()[-1]

    cfgs = all_configs()
    # Each app is network-bound (independent Yahoo fetches) — run them
    # concurrently.  ex.map preserves order, so the universe stays in app order.
    with ThreadPoolExecutor(max_workers=len(cfgs)) as ex:
        rows = list(ex.map(_one, cfgs))
    _LAST_ERRORS.clear()
    out = []
    for key, g, err in rows:           # aggregate in the main thread (no races)
        if err or not g:
            _LAST_ERRORS[key] = err or "no instruments returned"
        out.extend(g or [])
    return out


# per-app load errors from the most recent run_universe() (best-effort; the app
# surfaces these so a silently-dropped sleeve is visible, not hidden).
_LAST_ERRORS: dict[str, str] = {}


# ════════════════════════════════════════════════════════════════════════
# PORTFOLIO MATHS — combine the per-instrument strategy return streams.
# ════════════════════════════════════════════════════════════════════════
TRADING_DAYS = 252


def returns_matrix(results: list[dict]) -> pd.DataFrame:
    """Align each instrument's daily strategy returns into one frame (NaN where
    the instrument has no history yet)."""
    return pd.DataFrame({res["key"]: res["ret"] for res in results}).sort_index()


def position_matrix(results: list[dict], index: pd.Index,
                    carry: bool = False) -> pd.DataFrame:
    """0/1 in-market flag per instrument, aligned to ``index`` (0 where flat or
    no data).

    ``carry=True`` forward-fills each sleeve's state across its NON-trading
    days on a union calendar (weekends/holidays for equities once the crypto
    sleeves put Saturdays into the index): a position held at a sleeve's last
    bar is still *held* on days its market is shut — it merely has no return —
    rather than counted as sold.  Inception gaps stay 0 either way (ffill
    starts at each sleeve's first bar).  The default (``carry=False``) keeps
    the raw own-calendar view used by the optimiser and benchmarks."""
    cols = {res["key"]: res["pos_series"] for res in results}
    df = pd.DataFrame(cols).reindex(index)
    if carry:
        df = df.ffill()
    return df.fillna(0.0)


def _combine(returns: pd.DataFrame, weights: np.ndarray,
             pos: pd.DataFrame | None = None, sata_daily: float = 0.0) -> pd.Series:
    """Daily portfolio return for fixed target weights, renormalising over the
    instruments that actually have data each day (handles staggered inception).

    When a ``pos`` matrix and ``sata_daily`` are supplied, deployed weight that
    is sitting *out of the market* (its sleeve flat) — and any fully-cash day —
    earns the SATA idle-cash yield instead of nothing."""
    w = np.asarray(weights, float)
    avail = returns.notna().to_numpy()
    wt = avail * w
    denom = wt.sum(axis=1, keepdims=True)
    zero = denom[:, 0] == 0
    denom[denom == 0] = np.nan
    ew = wt / denom                              # deployed weights, sum→1 among available
    filled = np.nan_to_num(returns.to_numpy())
    port = np.nansum(np.nan_to_num(ew) * filled, axis=1)
    if pos is not None and sata_daily:
        P = np.nan_to_num(pos.reindex(returns.index).to_numpy())
        idle = np.nansum(np.nan_to_num(ew) * (1.0 - P) * avail, axis=1)
        port = port + sata_daily * idle
        port[zero] = sata_daily                  # fully-cash day → all in SATA
    else:
        port[zero] = 0.0
    return pd.Series(port, index=returns.index)


def _equity(daily_ret: pd.Series) -> pd.Series:
    return (1.0 + daily_ret.fillna(0.0)).cumprod()


def curve_metrics(equity: pd.Series) -> dict:
    eq = equity.to_numpy(float); idx = equity.index
    if len(eq) < 2:
        return dict(total_ret=0.0, cagr=0.0, mdd=0.0, sharpe=0.0, vol=0.0)
    total = eq[-1] / eq[0] - 1
    yrs = max((idx[-1] - idx[0]).days / 365.25, 1e-9)
    cagr = (eq[-1] / eq[0]) ** (1 / yrs) - 1
    peak = np.maximum.accumulate(eq)
    mdd = float(np.min(eq / peak - 1))
    rets = np.diff(eq) / eq[:-1]
    vol = float(np.std(rets) * np.sqrt(TRADING_DAYS))
    sharpe = float(np.mean(rets) / (np.std(rets) + 1e-12) * np.sqrt(TRADING_DAYS))
    return dict(total_ret=float(total), cagr=float(cagr), mdd=mdd, sharpe=sharpe, vol=vol)


def slice_metrics(equity: pd.Series, start, end=None) -> dict | None:
    """P&L / performance / risk metrics for an equity curve **re-based at
    ``start``** — i.e. what an investor who put capital into the strategy on
    that date has experienced since.  Re-basing (dividing the slice by its first
    value) matters: drawdown, Sharpe and total return are all measured from the
    entry point, not from the back-test's inception.  Returns ``None`` when the
    curve has fewer than 2 bars on/after ``start`` (nothing to measure); the
    first bar on/after ``start`` becomes the actual anchor (weekends/holidays
    roll forward).  An optional ``end`` closes the window at the last bar
    on/before it (``None`` = through the latest bar)."""
    sub = equity.loc[pd.Timestamp(start):
                     (pd.Timestamp(end) if end is not None else None)]
    if len(sub) < 2:
        return None
    sub = sub / sub.iloc[0]
    # daily win-rate and best/worst day round out the risk read
    rets = sub.pct_change().dropna()
    return dict(start=sub.index[0], end=sub.index[-1], days=len(sub),
                win_days=float((rets > 0).mean()),
                best_day=float(rets.max()), worst_day=float(rets.min()),
                **curve_metrics(sub))


def _wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion — behaves sanely at
    small ``n`` and near 0%/100%, where the naive normal interval breaks."""
    p = wins / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * np.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n))
    return float((centre - margin) / denom), float((centre + margin) / denom)


def daily_win_stats(equity: pd.Series, start, end=None,
                    active: pd.Series | None = None) -> dict | None:
    """Winning-days read on an equity curve, conditioned on the days the
    strategy actually held risk.

    ``slice_metrics``' ``win_days`` counts positive days out of ALL trading
    days — but idle capital parks in SATA at a constant daily yield, so every
    fully-cash day is a tiny guaranteed "win" and the all-days rate drifts
    toward the in-market share instead of measuring edge.  Here a day only
    enters the denominator when ``active`` (a daily position flag / deployed-
    weight series, reindexed to the return days) is > 0; without ``active``
    every day counts, matching the old behaviour.

    Win rate alone can't be judged without the payoff side (a 60% rate with
    fat losing days still loses money), so the dict carries the full first-
    order read on the active-day distribution:

      n_days      trading days in the window (denominator of ``slice_metrics``)
      n_active    days with risk on — the denominator here
      wins        active days with return > 0 (flat active days are NOT wins)
      win_rate    wins / n_active, ``None`` with no active days
      ci_lo/ci_hi Wilson 95% interval on ``win_rate`` — with few active days
                  a coin flip is inside the interval and the headline rate
                  should not be trusted
      avg_win     mean return of positive active days (0.0 if none)
      avg_loss    mean return of negative active days (≤ 0; 0.0 if none)
      payoff      avg_win / |avg_loss|, ``None`` with no losing days
      expectancy  mean active-day return — exactly
                  ``p·avg_win + (1−p)·avg_loss`` with flat days folded in;
                  the per-day edge that compounds

    Windowing matches ``slice_metrics``: first bar on/after ``start`` anchors,
    optional ``end`` closes at the last bar on/before it, ``None`` with <2
    bars.  Daily returns are autocorrelated in trending tapes, so the Wilson
    interval is mildly optimistic — read it as a floor on the uncertainty."""
    sub = equity.loc[pd.Timestamp(start):
                     (pd.Timestamp(end) if end is not None else None)]
    if len(sub) < 2:
        return None
    rets = sub.pct_change().dropna()
    n_days = len(rets)
    if active is not None:
        on = active.reindex(rets.index).fillna(0.0).astype(float) > 0
        rets = rets[on]
    n = len(rets)
    if n == 0:
        return dict(n_days=n_days, n_active=0, wins=0, win_rate=None,
                    ci_lo=None, ci_hi=None, avg_win=None, avg_loss=None,
                    payoff=None, expectancy=None)
    wins = int((rets > 0).sum())
    ci_lo, ci_hi = _wilson(wins, n)
    up, dn = rets[rets > 0], rets[rets < 0]
    avg_win = float(up.mean()) if len(up) else 0.0
    avg_loss = float(dn.mean()) if len(dn) else 0.0
    return dict(n_days=n_days, n_active=n, wins=wins, win_rate=wins / n,
                ci_lo=ci_lo, ci_hi=ci_hi, avg_win=avg_win, avg_loss=avg_loss,
                payoff=(avg_win / abs(avg_loss)) if avg_loss < 0 else None,
                expectancy=float(rets.mean()))


def per_asset_slice_metrics(results: list[dict], start, end=None) -> list[dict]:
    """Per-instrument read since ``start``: each sleeve's strategy equity and
    its buy-&-hold equity, both re-based at the anchor (see ``slice_metrics``),
    plus the share of days it actually spent in the market and its invested-
    days winning read (``daily`` — see ``daily_win_stats``).  An instrument whose
    history begins after ``start`` is measured from its own first bar (its
    ``strat['start']`` says so); one with <2 bars since ``start`` is skipped.
    Rows come back in ``results`` order (grouped by parent signal).  An
    optional ``end`` closes every window at the last bar on/before it
    (``None`` = through the latest bar)."""
    end_ts = pd.Timestamp(end) if end is not None else None
    rows = []
    for res in results:
        eq = _equity(res["ret"])
        sm = slice_metrics(eq, start, end)
        if sm is None:
            continue
        bh_eq = pd.Series(np.asarray(res["r"]["bh"], float),
                          index=pd.DatetimeIndex(res["dates"]))
        pos_sub = res["pos_series"].loc[pd.Timestamp(start):end_ts]
        rows.append(dict(
            key=res["key"], name=res["name"], kind=res["kind"],
            parent=res["parent"], accent=res["accent"], emoji=res["emoji"],
            in_market=float(pos_sub.mean()) if len(pos_sub) else 0.0,
            strat=sm, bh=slice_metrics(bh_eq, start, end),
            daily=daily_win_stats(eq, start, end, active=res["pos_series"]),
            trades=trade_stats_since(res, start, end)))
    return rows


def trade_stats_since(res: dict, start, end=None) -> dict:
    """Trade count and win rate for one sleeve since ``start``.  A trade counts
    when it was open at any point on/after the anchor — i.e. every closed trade
    whose exit lands on/after ``start`` (even if entered before it) plus the
    currently-open position.  ``trade_log`` holds *closed* trades only and the
    open position lives in ``res['pos']``, so there is no double count.  A
    closed trade wins on its realised return; the open one on its current
    unrealised P&L.  ``win_rate`` is a 0–1 fraction, ``None`` with no trades.

    With an ``end``, only trades open at some point within ``[start, end]``
    count: closed trades additionally need an entry on/before ``end`` (still
    judged on their full realised return, even when the exit lands after
    ``end``), and the currently-open position counts only when entered
    on/before ``end`` (an undated open position can't be assigned to the
    window, so it's conservatively not counted).

    Not every engine exposes ``trade_log``: the BTC CT engine's ``trades`` ARE
    dicts of the same shape, so fall back to them when ``trade_log`` is absent
    (e.g. a stale hot-loaded engine module).  The dict filter also skips
    engines whose ``trades`` is a bare array of returns — undated trades can't
    be assigned to the window, so they're (conservatively) not counted."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end) if end is not None else None
    log = res["r"].get("trade_log")
    if not log:
        raw = res["r"].get("trades")           # may be a numpy array — no `or`
        log = [t for t in (raw if raw is not None else []) if isinstance(t, dict)]
    wins = n = 0
    for t in log:
        if pd.Timestamp(t["exit_date"]) >= start and \
           (end is None or pd.Timestamp(t["entry_date"]) <= end):
            n += 1
            wins += t["ret"] > 0
    n_open = 0
    pos = res.get("pos") or {}
    if pos.get("in_pos") and (end is None or (
            pos.get("entry_date") is not None
            and pd.Timestamp(pos["entry_date"]) <= end)):
        n_open = 1
        n += 1
        wins += (pos.get("upnl") or 0) > 0
    return dict(n_trades=n, n_open=n_open, wins=int(wins),
                win_rate=(wins / n) if n else None)


def overall_trade_stats(pa_rows: list[dict], weights: dict,
                        min_w: float = 0.002) -> dict:
    """Blend-level trade count / win rate since the anchor: the sum over every
    sleeve the optimal blend actually holds (weight > ``min_w`` — a zero-weight
    sleeve never trades the book), open positions included."""
    sel = [r["trades"] for r in pa_rows if weights.get(r["key"], 0) > min_w]
    n = sum(t["n_trades"] for t in sel)
    wins = sum(t["wins"] for t in sel)
    return dict(n_trades=n, wins=wins, n_open=sum(t["n_open"] for t in sel),
                win_rate=(wins / n) if n else None, n_sleeves=len(sel))


def trade_log_since(results: list[dict], weights: dict, start,
                    min_w: float = 0.002,
                    weight_matrix: pd.DataFrame | None = None,
                    end=None) -> list[dict]:
    """The individual trades behind ``overall_trade_stats``: every round-trip
    across the sleeves the optimal blend holds (weight > ``min_w``) that was
    open at any point on/after ``start`` — closed trades whose exit lands on/
    after the anchor plus any currently-open position (``open=True``, no exit
    date, ``ret`` = its unrealised P&L at the latest price incl. any live-spot
    overlay).  Engines disagree on price key names (``entry_px``/``exit_px``
    vs the BTC CT engine's ``entry_price``/``exit_price``) — normalised here.
    Rows come back newest-first: open positions, then closed by exit date.

    ``weights`` gates which sleeves count (weight > ``min_w``) and is each
    row's default ``weight``.  With a ``weight_matrix`` (the replay's daily
    weight frame) each trade instead carries the PEAK weight the book gave
    that sleeve while the trade was open — the honest deployed notional under
    a daily-rebalanced book.

    With an ``end``, only trades open at some point within ``[start, end]``
    are listed — same window rule as ``trade_stats_since``: closed trades
    additionally need an entry on/before ``end``, and the currently-open
    position is listed only when entered on/before ``end``."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end) if end is not None else None

    def _px(t: dict, *keys):
        for k in keys:
            v = t.get(k)
            if v is not None and np.isfinite(v):
                return float(v)
        return None

    def _trade_w(key: str, default: float, entry_d, exit_d) -> float:
        if weight_matrix is None or key not in weight_matrix.columns:
            return float(default)
        col = weight_matrix[key].loc[entry_d:(exit_d if exit_d is not None else None)]
        return float(col.max()) if len(col) and np.isfinite(col.max()) else float(default)

    rows = []
    for res in results:
        w = weights.get(res["key"], 0)
        if w <= min_w:
            continue
        log = res["r"].get("trade_log")
        if not log:                            # same fallback as trade_stats_since
            raw = res["r"].get("trades")       # may be a numpy array — no `or`
            log = [t for t in (raw if raw is not None else []) if isinstance(t, dict)]
        base = dict(key=res["key"], name=res["name"], kind=res["kind"],
                    emoji=res["emoji"], accent=res["accent"], weight=float(w))
        for t in log:
            exit_d = pd.Timestamp(t["exit_date"])
            entry_d = pd.Timestamp(t["entry_date"])
            if exit_d < start or (end is not None and entry_d > end):
                continue
            base["weight"] = _trade_w(res["key"], w, entry_d, exit_d)
            rows.append(dict(base, open=False, entry_date=entry_d, exit_date=exit_d,
                             entry_px=_px(t, "entry_px", "entry_price"),
                             exit_px=_px(t, "exit_px", "exit_price"),
                             ret=float(t["ret"]), reason=str(t.get("reason") or "signal exit"),
                             days=int((exit_d - entry_d).days)))
        pos = res.get("pos") or {}
        if pos.get("in_pos"):
            e_dt = pd.Timestamp(pos["entry_date"]) if pos.get("entry_date") else None
            if end is None or (e_dt is not None and e_dt <= end):
                upnl = pos.get("upnl")         # % — refreshed by apply_spot
                last_px = res.get("last_close")
                base["weight"] = _trade_w(res["key"], w, e_dt, None)
                rows.append(dict(base, open=True, entry_date=e_dt, exit_date=None,
                                 entry_px=_px(pos, "entry_px"),
                                 exit_px=(float(last_px) if last_px is not None
                                          and np.isfinite(last_px) else None),
                                 ret=(upnl / 100 if upnl is not None else None),
                                 reason="open", days=pos.get("days")))
    _far_past = pd.Timestamp("1900-01-01")
    rows.sort(key=lambda r: (r["open"],
                             r["exit_date"] or r["entry_date"] or _far_past,
                             r["entry_date"] or _far_past),
              reverse=True)
    return rows


def daily_trade_log(weights: pd.DataFrame, sata: pd.Series, start,
                    min_delta: float = 0.0005,
                    returns: pd.DataFrame | None = None,
                    active: pd.DataFrame | None = None,
                    sata_daily: float = SATA_DAILY) -> list[dict]:
    """Day-by-day BUY/SELL log implied by the daily-weight book since
    ``start`` — the positions bought and sold at each close as Action
    signals open/close sleeves and the daily optimizer's tilt adjustments
    re-size the ones already held.  Source-agnostic: feed it either the
    walk-forward replay's weight matrix or the as-published record's
    (``replay_gated_allocation`` and ``published_book_replay`` return the
    same ``weights``/``sata`` shape), so the P&L section's source toggle
    drives this log unchanged.

    Both sources share the decided-at-previous-close convention (row *d*
    is the book earning day *d*, put on at the previous bar's close), so
    the diff between consecutive rows IS the trade ticket executed at the
    earlier bar's close — each log entry is dated at that execution bar.
    Every day-over-day weight change beyond ``min_delta`` becomes one
    action:

    * ``buy``  — 0 → w  (position opened — a signal change)
    * ``sell`` — w → 0  (position closed — a signal change)
    * ``add`` / ``trim`` — re-sized while held (a daily tilt adjustment)

    Returns one dict per day that traded (newest first): ``date`` (the
    execution close), ``actions`` sorted largest |Δweight| first (each
    with ``key``/``w0``/``w1``/``delta``/``action``/``signal_change``),
    ``gross`` (Σ|Δw| across the risk sleeves — the fraction of the book
    traded at that close), one-way ``turnover`` (½·(Σ|Δw| + |Δcash|) —
    the published-book trim convention), the cash weight before/after,
    and per-day ``n_buys``/``n_sells``/``n_resize`` counts.  Days where
    nothing moved beyond ``min_delta`` are omitted.

    With ``returns`` (per-sleeve daily % returns on the same keys), every
    sell-side action (``sell`` and ``trim``) also carries ``pnl`` and
    ``entry_date`` — the realized P&L on the slice sold, from an
    AVERAGE-COST lot ledger.  The ledger simulates the book's implied
    daily dollar flows on the FULL weight matrix (weights are fractions of
    the compounding blend, so every close's re-balance buys or sells
    dollars — even between listed trades): buys add cost at that close's
    value (so tilt adds RAISE the basis at the price actually paid), sells
    remove cost pro-rata, and a position that empties (weight below
    ``min_delta``) closes its lot — the next buy starts a fresh basis at
    ``entry_date``.  ``pnl`` is the slice's value over its average cost,
    minus 1, at the execution close.  A position already held at the
    matrix's first row starts its basis at that first close (the
    anchor-bar-is-cost-basis convention); ``sata_daily`` feeds the blend's
    value path the same business-day SATA accrual as the replays.
    Sell-side actions additionally carry the ledger's EXACT dollar flows,
    per $1 invested at the anchor close (scale by the portfolio value for
    dollars): ``sold`` — the slice's value at the execution close, on the
    blend's compounded value path — and ``basis`` — the average-cost basis
    removed by the sale (pro-rata for a trim, the whole lot for a full
    close), so ``sold − basis`` is the realized gain and
    ``sold/basis − 1 == pnl``.  These are exact where a naive
    ``|Δweight| × anchor portfolio value`` is not: weights are fractions
    of the COMPOUNDING blend, so once the blend has moved since the
    anchor the two diverge.
    Buy-side actions — and every action when ``returns`` is omitted or the
    key has no return column — carry ``pnl``/``entry_date``/``sold``/
    ``basis`` of ``None``.

    With ``active`` (bool DataFrame, per-day signal/position state on the
    same keys — ``replay_gated_allocation`` returns one),
    ``signal_change`` reflects the ACTUAL signal flips: a weight move with
    no flip is a daily tilt adjustment even when it opens or fully closes
    the position (the optimizer re-sized through zero).  Without it —
    e.g. the as-published record, where the archived books are all that is
    known — signal changes are inferred from the weights: any full open or
    close counts as one."""
    wf = weights.fillna(0.0)
    w = wf.loc[pd.Timestamp(start):]
    if len(w) < 2:
        return []
    cash = sata.reindex(wf.index).fillna(0.0).astype(float)
    act_f = active.reindex(index=wf.index).fillna(False) \
        if active is not None else None

    # ── average-cost lot ledger ──────────────────────────────────────────
    # snap_pnl[k][j]: sleeve k's % gain over its average cost at the close
    # of row j, BEFORE that close's re-balance — exactly what a sale at
    # that close realizes.  lot_entry[k][j]: row where the open lot began.
    # sell_val/sell_cost[k][j]: the EXACT dollars (per $1 at the anchor
    # close) the re-balance at close j sold, and the average-cost basis
    # those dollars carried — the actual realized flows on the compounded
    # value path.
    snap_pnl: dict[str, np.ndarray] = {}
    lot_entry: dict[str, np.ndarray] = {}
    sell_val: dict[str, np.ndarray] = {}
    sell_cost: dict[str, np.ndarray] = {}
    if returns is not None:
        wz = wf.where(wf >= min_delta, 0.0)      # same flat floor as actions
        kcols = [k for k in wf.columns if k in returns.columns]
        R = returns.reindex(wf.index).fillna(0.0).astype(float)
        biz = np.asarray(wf.index.dayofweek < 5)
        port = (wz[kcols].to_numpy(float) * R[kcols].to_numpy(float)) \
            .sum(axis=1) + cash.to_numpy(float) * sata_daily * biz
        port[0] = 0.0                            # anchor bar = cost basis
        V = np.cumprod(1.0 + port)               # blend value at each close
        for k in kcols:
            wk = wz[k].to_numpy(float)
            rk = R[k].to_numpy(float)
            sp = np.full(len(wk), np.nan)
            le = np.full(len(wk), -1, dtype=int)
            sv = np.full(len(wk), np.nan)
            sc = np.full(len(wk), np.nan)
            cost = wk[0]                         # basis = value at row-0 close
            ent = 0 if wk[0] > 0 else -1
            for j in range(len(wk)):
                a_val = wk[0] if j == 0 else \
                    wk[j] * V[j - 1] * (1.0 + rk[j])   # pre-re-balance value
                if a_val > 0 and cost > 0:
                    sp[j] = a_val / cost - 1.0
                    le[j] = ent
                if j + 1 < len(wk):              # re-balance at this close
                    h_val = wk[j + 1] * V[j]     # post-re-balance value
                    if a_val <= 0:
                        if h_val > 0:            # fresh lot
                            cost, ent = h_val, j
                    elif h_val <= 0:             # emptied — lot closed
                        sv[j], sc[j] = a_val, cost
                        cost, ent = 0.0, -1
                    elif h_val > a_val:          # buy: cost added at value
                        cost += h_val - a_val
                    else:                        # sell: cost removed pro-rata
                        sv[j] = a_val - h_val
                        sc[j] = cost * (1.0 - h_val / a_val)
                        cost *= h_val / a_val
            snap_pnl[k] = sp
            lot_entry[k] = le
            sell_val[k] = sv
            sell_cost[k] = sc

    days = []
    off = wf.index.get_loc(w.index[0])
    for i in range(off + 1, len(wf)):
        prev, cur = wf.iloc[i - 1], wf.iloc[i]
        actions = []
        for k in wf.columns:
            w0, w1 = float(prev[k]), float(cur[k])
            d = w1 - w0
            if abs(d) < min_delta:
                continue
            if w0 < min_delta:
                act = "buy"
            elif w1 < min_delta:
                act = "sell"
            else:
                act = "add" if d > 0 else "trim"
            if act_f is not None and k in act_f.columns:
                sig = bool(act_f[k].iloc[i]) != bool(act_f[k].iloc[i - 1])
            else:
                sig = act in ("buy", "sell")
            pnl = entry = sold = basis = None
            if d < 0 and k in snap_pnl:
                v = snap_pnl[k][i - 1]
                if np.isfinite(v):
                    pnl = float(v)
                    e = int(lot_entry[k][i - 1])
                    entry = wf.index[e] if e >= 0 else None
                # the exact flows the re-balance at this close sold
                sv = sell_val[k][i - 1]
                if np.isfinite(sv):
                    sold = float(sv)
                    basis = float(sell_cost[k][i - 1])
            actions.append(dict(key=k, w0=w0, w1=w1, delta=d, action=act,
                                signal_change=sig, pnl=pnl, entry_date=entry,
                                sold=sold, basis=basis))
        if not actions:
            continue
        actions.sort(key=lambda a: -abs(a["delta"]))
        gross = float((cur - prev).abs().sum())
        d_cash = float(cash.iloc[i] - cash.iloc[i - 1])
        days.append(dict(
            date=wf.index[i - 1], actions=actions, gross=gross,
            turnover=0.5 * (gross + abs(d_cash)),
            cash0=float(cash.iloc[i - 1]), cash1=float(cash.iloc[i]),
            n_buys=sum(a["action"] == "buy" and a["signal_change"]
                       for a in actions),
            n_sells=sum(a["action"] == "sell" and a["signal_change"]
                        for a in actions),
            n_resize=sum(not a["signal_change"] for a in actions)))
    days.reverse()
    return days


def trade_log_win_stats(log: list[dict]) -> dict | None:
    """Hit rate of the tickets that ACTUALLY realized P&L — the sale side of
    ``daily_trade_log``, judged one ticket at a time.

    The headline ``Win rate`` (``overall_trade_stats``) counts each sleeve's
    round-trip strategy trades: what the per-asset ENGINES did, whether or
    not the blend had money in that sleeve, and with open positions marked
    at their unrealised P&L.  This instead counts what the BOOK did — every
    🚦 exit-signal close and every ⚖️ daily-tilt trim the selected P&L source
    actually executed in the window, each carrying the realized return on
    the slice sold over its average cost.  It is therefore the win rate
    *behind the realized P&L*: the same tickets the 🧾 trade log lists and
    the 💰 realized-P&L-by-asset table rolls up, and it moves with the
    performance-source toggle (as-published books vs walk-forward replay).

    Positions still open contribute nothing — their P&L is unrealised, so
    there is no ticket and nothing to win or lose yet.

    Returns ``None`` when no sale ticket carries P&L (nothing sold in the
    window, or the log was built without ``returns``); otherwise:

      n_tickets   sale tickets counted — the denominator
      wins        tickets closed above their average cost (flat is not a win)
      losses      tickets closed below it
      win_rate    wins / n_tickets
      ci_lo/ci_hi Wilson 95% interval on that rate — a handful of tickets is
                  a wide interval, and the headline should be read with it
      n_sells/sell_wins   the 🚦 exit-signal closes inside those totals
      n_trims/trim_wins   the ⚖️ tilt trims inside them
      avg_win/avg_loss    mean realized % on winning / losing tickets
      payoff      avg_win / |avg_loss|, ``None`` with no losing ticket
      net         realized gain summed over every ticket, per $1 invested at
                  the anchor close (scale by the portfolio value for dollars)
    """
    tickets = [a for d in (log or []) for a in d["actions"]
               if a["delta"] < 0 and a.get("pnl") is not None]
    if not tickets:
        return None
    wins = [a for a in tickets if a["pnl"] > 0]
    losses = [a for a in tickets if a["pnl"] < 0]
    sells = [a for a in tickets if a["action"] == "sell"]
    trims = [a for a in tickets if a["action"] != "sell"]
    ci_lo, ci_hi = _wilson(len(wins), len(tickets))
    avg_win = float(np.mean([a["pnl"] for a in wins])) if wins else 0.0
    avg_loss = float(np.mean([a["pnl"] for a in losses])) if losses else 0.0
    return dict(
        n_tickets=len(tickets), wins=len(wins), losses=len(losses),
        win_rate=len(wins) / len(tickets), ci_lo=ci_lo, ci_hi=ci_hi,
        n_sells=len(sells), sell_wins=sum(1 for a in sells if a["pnl"] > 0),
        n_trims=len(trims), trim_wins=sum(1 for a in trims if a["pnl"] > 0),
        avg_win=avg_win, avg_loss=avg_loss,
        payoff=(avg_win / abs(avg_loss)) if avg_loss < 0 else None,
        net=float(sum(a["sold"] - a["basis"] for a in tickets
                      if a.get("sold") is not None
                      and a.get("basis") is not None)))


def realized_pnl_by_asset(weights: pd.DataFrame, sata: pd.Series, start,
                          returns: pd.DataFrame,
                          min_delta: float = 0.0005,
                          active: pd.DataFrame | None = None,
                          sata_daily: float = SATA_DAILY) -> dict:
    """Per-asset realized + unrealized P&L of the daily-weight book since
    ``start`` under WINDOW-ANCHORED exact accounting — built to reconcile
    with ``pnl_attribution_replay`` to float precision.

    An average-cost ledger runs per sleeve over the window: a position
    already held at the anchor takes its basis at the anchor close (the
    same anchor-is-cost-basis convention as the attribution), buys add
    cost at that close's value, sells remove cost pro-rata — and EVERY
    implied daily re-balance flow is counted, including the sub-ticket
    drift the 🧾 trade log's ``min_delta`` floor hides, all on the
    blend's compounded value path (figures are per $1 at the anchor
    close — scale by the portfolio value for dollars).  Consequently,
    per sleeve::

        pnl (realized)  +  unrealized (still held)  ==  attribution $

    exactly, and a sleeve flat at the window end has ``pnl`` equal to
    its 📊 P&L-by-asset bar.  (The 🧾 trade-log chips instead measure
    each ticket from the position's own entry — trade-LIFETIME
    accounting — so individual chips need not sum to these figures.)

    Returns ``{key: {...}}`` with ``pnl`` (realized), ``unrealized``,
    ``total`` (their sum — the attribution dollars), ``ret``
    (Σproceeds/Σcost-sold − 1, the money-weighted realized % return on
    the capital sold), ``proceeds``/``cost``, ``deployed`` (the TOTAL
    cost the ledger ever put into the sleeve — the anchor basis plus
    every buy flow: signal opens, tilt adds and daily re-balance buys —
    i.e. the gross capital cycled through it; always equals ``cost`` +
    ``cost_open``, the basis still invested at the window end), the
    signal-vs-tilt split of the realized dollars (``pnl_signal`` —
    flows executed at closes where the listed ticket was a real signal
    sale, per ``active`` when the source provides it; ``pnl_tilt`` —
    everything else, re-balance drift included) and the listed
    sale-ticket counts ``n_trims``/``n_sells``.  EVERY sleeve held at
    some point in the window is reported (a bought-and-held sleeve has
    zero ticket counts but real ``deployed``/``unrealized``) — callers
    wanting only sleeves that sold filter on the ticket counts; ``{}``
    with <2 bars on/after ``start``."""
    sub = returns.loc[pd.Timestamp(start):]
    if len(sub) < 2:
        return {}
    W = weights.reindex(index=sub.index, columns=sub.columns) \
        .fillna(0.0).to_numpy(float)
    R = np.nan_to_num(sub.to_numpy(float))
    # identical blend-value path to pnl_attribution_replay, or the parts
    # stop reconciling
    biz = np.asarray(sub.index.dayofweek < 5)
    sata_term = (sata.reindex(sub.index).fillna(1.0).to_numpy(float)
                 * sata_daily * biz)
    contrib = W * R
    contrib[0, :] = 0.0
    sata_term[0] = 0.0
    port = contrib.sum(axis=1) + sata_term
    V = np.cumprod(1.0 + port)                  # blend value at each close

    # listed sale tickets: the visibility gate + the signal/tilt labels
    sig_sale: set[tuple] = set()
    listed: dict[str, list] = {}
    for day in daily_trade_log(weights, sata, start, min_delta=min_delta,
                               active=active):
        for a in day["actions"]:
            if a["delta"] < 0:
                listed.setdefault(a["key"], []).append(a)
                if a["signal_change"]:
                    sig_sale.add((day["date"], a["key"]))

    out: dict[str, dict] = {}
    n = len(sub)
    for ci, k in enumerate(sub.columns):
        wk = W[:, ci]
        if not (wk > 0).any():
            continue
        rk = R[:, ci]
        cost = a_val = wk[0]                    # basis = the anchor close
        deployed = wk[0]                        # anchor holding = cost in
        realized = rl_sig = proceeds = cost_sold = 0.0
        for j in range(n):
            if j > 0:                           # pre-re-balance value
                a_val = wk[j] * V[j - 1] * (1.0 + rk[j])
            if j + 1 >= n:                      # no re-balance after last close
                break
            h_val = wk[j + 1] * V[j]            # post-re-balance value
            if a_val <= 0:
                cost = h_val if h_val > 0 else 0.0   # fresh lot (or stay flat)
                deployed += cost
                continue
            if h_val >= a_val:
                cost += h_val - a_val           # buy: cost added at value
                deployed += h_val - a_val
                continue
            sold = a_val - h_val                # sell flow at this close
            gain = sold * (1.0 - cost / a_val) if cost > 0 else sold
            realized += gain
            proceeds += sold
            cost_sold += sold - gain
            if (sub.index[j], k) in sig_sale:
                rl_sig += gain
            cost *= h_val / a_val               # pro-rata cost removal
        unreal = (a_val - cost) if a_val > 0 else 0.0
        tickets = listed.get(k, [])
        n_sig = sum(a["signal_change"] for a in tickets)
        out[k] = dict(pnl=realized, unrealized=unreal,
                      total=realized + unreal,
                      ret=(proceeds / cost_sold - 1.0) if cost_sold > 0
                      else 0.0,
                      proceeds=proceeds, cost=cost_sold,
                      deployed=deployed, cost_open=cost,
                      pnl_signal=rl_sig, pnl_tilt=realized - rl_sig,
                      n_trims=len(tickets) - n_sig, n_sells=n_sig)
    return out


def sata_slice_metrics(index: pd.Index, start) -> dict | None:
    """The SATA idle-cash sleeve since ``start`` on the blend's calendar: a
    constant ``SATA_DAILY`` yield on a flat $100 par — so no drawdown, every
    day a (tiny) win, and Sharpe/vol meaningless (returned as ``None``/0).
    The coupon is per BUSINESS day (0.13/250), so weekend bars on the crypto
    calendar accrue nothing — matching the replay."""
    sub = index[index >= pd.Timestamp(start)]
    if len(sub) < 2:
        return None
    n_biz = int((sub[1:].dayofweek < 5).sum())      # anchor bar = cost basis
    total = float((1 + SATA_DAILY) ** n_biz - 1)
    yrs = max((sub[-1] - sub[0]).days / 365.25, 1e-9)
    return dict(start=sub[0], end=sub[-1], days=len(sub), total_ret=total,
                cagr=float((1 + total) ** (1 / yrs) - 1), mdd=0.0, sharpe=None,
                vol=0.0, win_days=1.0, best_day=SATA_DAILY, worst_day=SATA_DAILY)


def pnl_attribution_since(returns: pd.DataFrame, weights, start,
                          pos: pd.DataFrame | None = None,
                          sata_daily: float = 0.0) -> dict | None:
    """Exact per-sleeve attribution of the blend's P&L since ``start``.

    Re-runs the same daily maths as ``_combine`` on the slice from the anchor
    bar, but keeps each sleeve's term separate and scales every day's return
    by the blend's compounded value at the previous close.  Because the day's
    dollar P&L is exactly ``V_prev × (Σ_k ew_k·r_k + sata·idle)``, the per-key
    dollars plus the SATA term sum *exactly* (to float precision) to the
    blend's re-based total return since the anchor — i.e. the same number
    ``slice_metrics`` reports for the blend curve.  All figures are per $1
    invested at the anchor close (the anchor bar itself is the cost basis, so
    its return is not counted).  Returns ``None`` with <2 bars since ``start``.
    """
    sub = returns.loc[pd.Timestamp(start):]
    if len(sub) < 2:
        return None
    w = np.asarray(weights, float)
    avail = sub.notna().to_numpy()
    wt = avail * w
    denom = wt.sum(axis=1, keepdims=True)
    zero = denom[:, 0] == 0
    denom[denom == 0] = np.nan
    ew = np.nan_to_num(wt / denom)               # deployed weights, sum→1
    contrib = ew * np.nan_to_num(sub.to_numpy()) # per-sleeve daily return term
    sata_term = np.zeros(len(sub))
    if pos is not None and sata_daily:
        P = np.nan_to_num(pos.reindex(sub.index).to_numpy())
        idle = (ew * (1.0 - P) * avail).sum(axis=1)
        sata_term = sata_daily * idle
        sata_term[zero] = sata_daily             # fully-cash day → all in SATA
    contrib[0, :] = 0.0                          # anchor bar = cost basis
    sata_term[0] = 0.0
    port = contrib.sum(axis=1) + sata_term
    v_prev = np.concatenate([[1.0], np.cumprod(1.0 + port)[:-1]])
    dollars = (contrib * v_prev[:, None]).sum(axis=0)
    return dict(per_key={c: float(v) for c, v in zip(sub.columns, dollars)},
                sata=float((sata_term * v_prev).sum()),
                total=float(np.prod(1.0 + port) - 1.0),
                start=sub.index[0], end=sub.index[-1])


def _metrics_batch(returns: pd.DataFrame, cand: np.ndarray,
                   pos: pd.DataFrame | None = None, sata_daily: float = 0.0,
                   chunk: int | None = None):
    """Vectorised ``curve_metrics(_equity(_combine(...)))`` over MANY weight
    vectors at once.

    Evaluating the optimiser's ~20 000-sample Monte-Carlo one candidate at a
    time — each building a ``pd.Series`` in ``_combine``/``_equity`` — cost ~10 s
    per profile (×3 profiles at first load).  This does the identical maths in
    batched numpy: a handful of array passes instead of 20 000 Python/pandas
    round-trips.  For a single row it is bit-for-bit equal to ``_combine`` →
    ``_equity`` → ``curve_metrics``.  Returns five arrays aligned to ``cand``'s
    rows: ``(total_ret, cagr, mdd, sharpe, vol)``.  Candidates are processed in
    chunks so the transient ``(chunk, T, N)`` array stays small."""
    R = returns.to_numpy(float)
    A = returns.notna().to_numpy()                       # (T, N) availability
    F = np.nan_to_num(R)                                 # (T, N) returns, 0 where NaN
    idx = returns.index
    yrs = max((idx[-1] - idx[0]).days / 365.25, 1e-9)
    use_sata = pos is not None and bool(sata_daily)
    if use_sata:
        idle_w = (1.0 - np.nan_to_num(pos.reindex(idx).to_numpy())) * A   # (T, N)
    C = np.asarray(cand, float)
    M, N = C.shape
    T = R.shape[0]
    if chunk is None:                                    # bound the (b, T, N) temp
        chunk = max(1, 3_000_000 // max(T * N, 1))
    tot = np.empty(M); cag = np.empty(M); mdd = np.empty(M)
    shp = np.empty(M); vlt = np.empty(M)
    sq = np.sqrt(TRADING_DAYS)
    for s0 in range(0, M, chunk):
        Wb = C[s0:s0 + chunk]                            # (b, N)
        b = slice(s0, s0 + Wb.shape[0])
        wt = A[None, :, :] * Wb[:, None, :]              # (b, T, N)
        denom = wt.sum(axis=2)                           # (b, T)
        zero = denom == 0
        # Long-only weights ⇒ a zero row has every ``wt`` zero, so dividing by 1
        # there yields ew = 0 (and ``port[zero]`` is overwritten below regardless).
        # Avoiding the NaN sentinel lets us skip nan_to_num / nansum — the two
        # dominant costs — and use plain division + sum on already-NaN-free arrays.
        ew = wt / np.where(zero, 1.0, denom)[:, :, None]
        port = (ew * F[None, :, :]).sum(axis=2)          # (b, T)
        if use_sata:
            port = port + sata_daily * (ew * idle_w[None, :, :]).sum(axis=2)
            port[zero] = sata_daily
        else:
            port[zero] = 0.0
        eq = np.cumprod(1.0 + port, axis=1)              # (b, T)
        e0 = eq[:, 0]; eL = eq[:, -1]
        tot[b] = eL / e0 - 1.0
        cag[b] = (eL / e0) ** (1.0 / yrs) - 1.0
        mdd[b] = np.min(eq / np.maximum.accumulate(eq, axis=1) - 1.0, axis=1)
        drets = np.diff(eq, axis=1) / eq[:, :-1]         # (b, T-1)
        sd = np.std(drets, axis=1)
        vlt[b] = sd * sq
        shp[b] = np.mean(drets, axis=1) / (sd + 1e-12) * sq
    return tot, cag, mdd, shp, vlt


def optimize_weights(returns: pd.DataFrame, caps: dict | None = None,
                     n_samples: int = 20000, seed: int = 7, mdd_floor: float = -0.35,
                     pos: pd.DataFrame | None = None, sata_daily: float = 0.0,
                     objective: str = "balanced", fundamental: bool = True) -> dict:
    """Search long-only weights (sum=1, per-instrument ≤ its cap).  Objective:
    ``"balanced"`` — highest return among near-max-Sharpe blends (holds Sharpe
    high); ``"max_return"`` — highest return within the ``mdd_floor`` budget
    (leans harder on the high-return β / 2× sleeves); ``"max_sharpe"`` — pure
    Sharpe.  Idle (out-of-market) capital earns the SATA yield when
    ``pos``/``sata_daily`` are supplied.  Pure-numpy MC."""
    cols = list(returns.columns)
    n = len(cols)
    cap_vec = np.array([(caps or CAP_BY_KEY).get(c, 0.30) for c in cols])
    rng = np.random.default_rng(seed)

    def _cm(w):
        return curve_metrics(_equity(_combine(returns, w, pos, sata_daily)))

    samples = rng.dirichlet(np.ones(n), size=n_samples)
    keep = (samples <= cap_vec + 1e-9).all(axis=1)
    samples = samples[keep]

    ew = np.full(n, 1.0 / n)
    vols = returns.std().to_numpy()
    inv = np.nan_to_num(1.0 / np.where(vols > 0, vols, np.nan))
    rp = inv / inv.sum() if inv.sum() > 0 else ew
    rp = _cap_normalise(rp, cap_vec)
    ew_c = _cap_normalise(ew, cap_vec)
    cand = np.vstack([samples, ew_c, rp]) if len(samples) else np.vstack([ew_c, rp])

    # Evaluate every candidate once (vectorised — see _metrics_batch), then choose
    # in two stages so the result both maximises risk-adjusted return AND grabs the
    # most raw return among near-optimal blends — matching the "maximise returns,
    # minimise losses" mandate rather than a purely defensive max-Sharpe point.
    # np.argmax returns the FIRST maximiser, so ties break in candidate order,
    # exactly as the previous ``max(pool, key=...)`` did.
    tot, cag, mdd_a, shp, vlt = _metrics_batch(returns, cand, pos, sata_daily)
    feas = np.nonzero(mdd_a >= mdd_floor)[0]
    pool = feas if feas.size else np.arange(len(cand))
    if objective == "max_return":
        # highest raw return inside the drawdown budget — leans on β / 2×
        sel = pool[int(np.argmax(tot[pool]))]
    elif objective == "max_sharpe":
        sel = pool[int(np.argmax(shp[pool]))]
    else:  # "balanced" — highest return among near-max-Sharpe blends
        best_sharpe = float(shp[pool].max())
        near = pool[shp[pool] >= 0.92 * best_sharpe]
        sel = near[int(np.argmax(tot[near]))]
    w_opt = cand[sel]
    m_opt = dict(total_ret=float(tot[sel]), cagr=float(cag[sel]), mdd=float(mdd_a[sel]),
                 sharpe=float(shp[sel]), vol=float(vlt[sel]))

    def _pack(w):
        return {c: float(x) for c, x in zip(cols, np.asarray(w, float))}

    base_optimal = dict(weights=_pack(w_opt), **m_opt)
    optimal = base_optimal
    if fundamental and FUNDAMENTAL_VIEW:
        # tilt the quant-optimal blend by forward conviction, re-water-fill to caps
        caps_d = {c: (caps or CAP_BY_KEY).get(c, 0.30) for c in cols}
        raw = {c: base_optimal["weights"][c] * FUNDAMENTAL_VIEW.get(c, 1.0) for c in cols}
        for c in cols:                       # let a high-conviction name enter even if ignored
            mult = FUNDAMENTAL_VIEW.get(c, 1.0)
            if mult > 1.2:
                raw[c] = max(raw[c], 0.02 * mult)
        w_t = _waterfill({c: v for c, v in raw.items() if v > 0}, caps_d)
        wv = np.array([w_t.get(c, 0.0) for c in cols])
        optimal = dict(weights=w_t, **_cm(wv))
    return dict(
        cols=cols,
        optimal=optimal,
        optimal_pretilt=base_optimal,
        fundamental=bool(fundamental and FUNDAMENTAL_VIEW),
        equal=dict(weights=_pack(ew_c), **_cm(ew_c)),
        risk_parity=dict(weights=_pack(rp), **_cm(rp)),
    )


def _cap_normalise(w: np.ndarray, cap_vec: np.ndarray) -> np.ndarray:
    """Normalise to sum 1 then water-fill against a per-element cap vector.

    Sums to 1 whenever the caps allow it (``cap_vec.sum() >= 1``); only an
    infeasible cap set leaves a remainder (→ cash/SATA in the callers)."""
    w = np.asarray(w, float).copy()
    if w.sum() <= 0:
        return w
    w = w / w.sum()
    # each pass pins at least one more name at its cap, so n+1 passes suffice
    for _ in range(len(w) + 1):
        over = w > cap_vec + 1e-9
        if not over.any():
            break
        spill = (w[over] - cap_vec[over]).sum()
        w[over] = cap_vec[over]
        room = w < cap_vec - 1e-9
        base = w[room].sum()
        if not room.any() or base <= 0:
            break
        # a recipient pushed past its own cap is NOT clipped here — the next
        # pass re-caps it and re-spills the excess to the names still below
        # cap, so no weight is silently dropped while room remains
        w[room] += spill * (w[room] / base)
    return w


def _waterfill(raw: dict[str, float], caps: dict) -> dict[str, float]:
    """Normalise ``raw`` to sum 1, capping each key at its cap and redistributing
    the overflow; leftover (if the keys can't absorb the book) → cash."""
    keys = [k for k, v in raw.items() if v > 0]
    if not keys:
        return {}
    cap_vec = np.array([caps.get(k, 0.30) for k in keys])
    w = _cap_normalise(np.array([raw[k] for k in keys]), cap_vec)
    # _cap_normalise keeps sum=1 when feasible; if n·cap<1 it can't, leaving room
    total = w.sum()
    if total > 1 + 1e-9:
        w = w / total
    return {k: float(v) for k, v in zip(keys, w)}


def signal_gated_allocation(results: list[dict], base_weights: dict[str, float],
                            caps: dict | None = None,
                            force_exit: set | None = None,
                            force_entry: set | None = None) -> dict:
    """Today's actionable allocation.

    Capital is deployed only to instruments the strategy is long (or opening a
    fresh entry).  Among those, the size of each slice is the historically
    optimal weight **tilted by a live entry-priority score** (momentum,
    sentiment, back-tested win-rate, risk-adjusted edge, regime), then
    water-filled to the per-instrument caps.  The deployed book sums to 100% when
    the caps allow; whatever cannot be deployed is parked in **SATA** (the idle
    -cash preferred, ~13% yield).  With no open positions the whole book is SATA.

    ``force_exit`` / ``force_entry`` are the LIVE-price overrides (from
    :func:`live_exit_keys` / :func:`live_entry_keys`) that make the result the
    *possible* target book the live prices point to: a forced exit is dropped
    from the book exactly like a committed close, and a forced entry — a flat
    name with no committed buy whose live price satisfies the mode's real entry
    condition — is funded exactly like a committed fresh open (priority-tilted,
    water-filled).  Force-funded entries are marked ``live_entry=True`` on their
    action row so callers can tell a live "likely" open from a committed one;
    a key in both sets is treated as an exit (it would open into a broken
    trend).  Committed-only callers (the publisher) pass neither.
    """
    caps = caps or CAP_BY_KEY
    force_exit = force_exit or set()
    force_entry = (force_entry or set()) - force_exit

    def _b(k):
        return max(base_weights.get(k, 0.0), 1e-6)

    in_pos = [res for res in results if res["pos"]["in_pos"]]
    # closing = committed exits — a tone-"exit" decision (divergence/BTC EXIT)
    # OR a committed pending exit (the trend family's HOLD + exits_next_bar:
    # the last close already broke the trend, so the position drops out on the
    # next bar) — plus any caller-forced exits (e.g. positions the live price
    # says will drop out on the next bar).  Both committed forms mean the same
    # thing — "this position will not be held after the next bar" — so both are
    # excluded from the recommended book; previously only tone-"exit" sleeves
    # were, leaving a trend sleeve whose death-cross committed at today's close
    # (e.g. REMX) still funded while its row was flagged "exits next bar".
    closing = {res["key"] for res in in_pos
               if res["decision"]["tone"] == "exit"
               or res["decision"].get("exits_next_bar")
               or res["key"] in force_exit}
    keep = [res for res in in_pos if res["key"] not in closing]
    # a fresh entry whose live price has already broken the trend (in force_exit)
    # would reverse straight back out next bar — don't fund it in the live book.
    # force_entry mirrors that on the way in: a flat name whose LIVE price
    # already satisfies the entry condition is funded like a committed open, so
    # the live book reflects the possible target book the live prices point to.
    opens = [res for res in results
             if (not res["pos"]["in_pos"]) and res["key"] not in force_exit
             and (res["decision"]["tone"] == "buy" or res["key"] in force_entry)]
    live_open = {res["key"] for res in opens if res["decision"]["tone"] != "buy"}

    target_keys = [res["key"] for res in keep] + [res["key"] for res in opens]
    prio = compute_priorities(results, target_keys)
    # priority-tilted raw weights: optimal anchor × (0.5 + priority) ∈ [0.5,1.5]×
    raw = {k: _b(k) * (0.5 + prio.get(k, {}).get("score", 0.5)) for k in target_keys}
    target = _waterfill(raw, caps)
    sata = max(1.0 - sum(target.values()), 0.0)

    # current book (what we hold now) — optimal weights, no forward tilt
    current = _waterfill({res["key"]: _b(res["key"]) for res in in_pos}, caps)
    sata_now = max(1.0 - sum(current.values()), 0.0)

    actions = []
    for res in results:
        k = res["key"]; dec = res["decision"]; stt = dec["state"]
        if res["pos"]["in_pos"] and (stt == "EXIT" or dec.get("exits_next_bar")):
            # The action column is an INSTRUCTION list ("what to do now"), and
            # for a committed pending exit the instruction is CLOSE: the book
            # zero-weights it and the executor sells it this session, exactly
            # like a tone-"exit" divergence close.  (Previously the action
            # stayed HOLD, so a sleeve the book itself had dropped to 0% still
            # rendered a HOLD pill.)
            act = "CLOSE"
        elif res["pos"]["in_pos"]:
            act = "HOLD"
        elif stt == "ENTRY":
            act = "OPEN"
        elif stt == "WATCH":            # U1 pressure, gate not yet met
            act = "WATCH"
        else:                            # AVOID (exit active / downtrend) or FLAT
            act = "STAND ASIDE"
        p = prio.get(k)
        # ``state`` rides along so the action table can tell a live flat-side
        # AVOID (exit signal active / D1 downtrend — entry blocked) from a plain
        # FLAT without parsing labels: both render the same STAND ASIDE action,
        # but only AVOID must be flagged when a frozen morning book still shows
        # WATCH for the sleeve (the PBW case).  The published payload trims to
        # its fixed field set, so the extra key never reaches the artifact.
        # ``live_entry`` marks a row funded ONLY because the live price
        # satisfies the entry condition (no committed buy yet) — the freeze
        # funds it like an OPEN, and the UI can label it a likely entry.
        actions.append(dict(key=k, name=res["name"], emoji=res["emoji"],
                            kemoji=res["kemoji"], kind=res["kind"], parent=res["parent"],
                            action=act, state=stt, tone=dec["tone"], decision=dec["label"],
                            target=target.get(k, 0.0), in_pos=res["pos"]["in_pos"],
                            upnl=res["pos"]["upnl"], alert=res["alert"],
                            last_close=res["last_close"],
                            exits_next_bar=bool(dec.get("exits_next_bar")),
                            live_entry=(k in live_open),
                            priority=(p["score"] if p else None),
                            prio_comp=(p if p else None)))
    order = {"CLOSE": 0, "OPEN": 1, "HOLD": 2, "WATCH": 3, "STAND ASIDE": 4}
    # within an action group, rank by priority (entries) then target size
    actions.sort(key=lambda a: (order[a["action"]],
                                -(a["priority"] if a["priority"] is not None else -1),
                                -a["target"]))
    ranked = sorted(prio.items(), key=lambda kv: -kv[1]["score"])
    return dict(target=target, sata=sata, current=current, sata_now=sata_now,
                actions=actions, priorities=prio, priority_rank=[k for k, _ in ranked],
                n_active=len(in_pos), n_open=len(opens), n_close=len(closing),
                sata_info=SATA)


def apply_closed_market_freeze(gate: dict, book_weights: dict | None,
                               today=None) -> dict:
    """Weekend/holiday guard for the LIVE recommended book.

    On a US trading day this is the identity — the gate's normal intraday
    behaviour is returned untouched.  On days the US equity market is
    CLOSED (weekend or NYSE holiday) the live gate must not re-rank or
    re-size the book: most sleeves' signal apps get no new bar (their data
    is stale from the last close), yet the daily tilt keeps recomputing —
    and because priorities are min-max normalised ACROSS the funded set,
    the crypto sleeves' fresh weekend bars shift every stale sleeve's
    normalised score, re-sizing positions that cannot even trade.  The fix
    pins the book to the ``book_weights`` actually published (the Current
    Targetbook's risk weights) and applies ONLY committed signal changes
    on top:

    * a sleeve whose gate action is CLOSE (committed exit, or a live-price
      forced exit) is dropped — its weight goes to SATA;
    * a sleeve whose gate action is OPEN — or a flat sleeve the gate
      force-funded on a live-price entry (``live_entry``, necessarily from
      an app whose signal asset still trades on a closed US day) — is
      funded from idle capital at the gate's own tilted target size (in
      the gate's priority order, capped by the idle remaining);
    * every other sleeve keeps its published weight EXACTLY — no tilt
      re-sizing on a day it cannot trade.

    Equity engines produce no new bar on closed days, so any OPEN/CLOSE
    seen here necessarily comes from an app whose signal asset does trade
    (the BTC/ETH apps — their bars close 12:00 UTC every day, weekends
    included) or was committed at the last real close and simply predates
    the published book — exactly the changes that SHOULD move the book.
    With no published book to pin to (``book_weights`` falsy) the gate is
    returned unchanged.  Returns a new gate dict of the same shape with
    per-action ``target`` values rewritten to match and a ``freeze``
    record (``day``/``closed``/``opened``/``pinned``) for the UI."""
    import freshness as fr
    today = (pd.Timestamp(today).date() if today is not None
             else pd.Timestamp.now(tz="America/New_York").date())
    if fr.is_us_trading_day(today) or not book_weights:
        return gate
    act_by = {a["key"]: a["action"] for a in gate["actions"]}
    # committed pending exits (HOLD + exits_next_bar — the trend family's "last
    # close broke the trend, exits next bar") are committed signal changes just
    # like a CLOSE, so the freeze applies them too instead of pinning a weight
    # the strategy has already decided to drop.
    pending_exit = {a["key"] for a in gate["actions"] if a.get("exits_next_bar")}
    target, closed = {}, []
    for k, w in book_weights.items():
        if w <= 0.0005:
            continue
        if act_by.get(k) == "CLOSE" or k in pending_exit:
            closed.append(k)
            continue
        target[k] = float(w)
    idle = max(1.0 - sum(target.values()), 0.0)
    opened = []
    for a in gate["actions"]:                 # OPENs come priority-ranked
        k = a["key"]
        if (a["action"] != "OPEN" and not a.get("live_entry")) or k in target:
            continue
        w = min(float(gate["target"].get(k, 0.0)), idle)
        if w <= 0.0005:
            continue
        target[k] = w
        idle -= w
        opened.append(k)
    sata = max(1.0 - sum(target.values()), 0.0)
    actions = [dict(a, target=target.get(a["key"], 0.0))
               for a in gate["actions"]]
    return dict(gate, target=target, sata=sata, actions=actions,
                freeze=dict(day=str(today), closed=sorted(closed),
                            opened=opened,
                            pinned=sorted(k for k in target
                                          if k not in opened)))


def book_phantom_actions(book_payload: dict | None, gate: dict,
                         live_exits=(), live_entries=()) -> dict:
    """Published-book instructions today's engine cannot reproduce.

    The action plan pins its Action / Signal / Priority cells to the *published*
    Current Targetbook so they match the donut and never drift intraday.  The
    app's flags already cover the book running BEHIND the engine (a signal that
    committed at a close after the morning publish — ``_book_masked_exit`` /
    ``_book_masked_entry`` / ``_book_masked_avoid``).  This is the other
    direction: the book carries an instruction the engine does not derive from
    the SAME as-of bar at all, which happens when the publish ran on a different
    data vintage than the pinned snapshot the apps now read (see
    ``data_gate._RECENT_SESSION_DAYS`` — a session served at publish time and
    dropped from a later fetch re-links a break streak and flips a hair-trigger
    signature).  Observed: ARTY 2026-08-14, published ``CLOSE`` /
    ``EXIT NEXT BAR — D3 exhaustion`` while every live surface read
    ``LONG — HOLDING`` with no D3 anywhere.

    Two mismatches are reported, both restricted to states the ordinary
    publish→execute lifecycle cannot produce:

    * ``exits``   — the book says CLOSE (or records ``exits_next_bar``) while
      the engine still holds the position and sees no exit, committed or live.
      A genuinely executed book exit leaves the sim flat on the next bar, so a
      still-held position means the exit never existed on today's data.
    * ``entries`` — the book says OPEN while the engine has no position and no
      entry, committed or live.  A genuinely executed book entry fills at the
      next bar's close regardless of that bar's signal, so a still-flat sleeve
      means the entry never existed either.

    Returns ``dict(exits=[...], entries=[...])``, sorted keys.  An unpublished
    book (or one without an ``actions`` list) yields empty lists.
    """
    book = {a.get("key"): a for a in ((book_payload or {}).get("actions") or [])
            if a.get("key")}
    if not book:
        return dict(exits=[], entries=[])
    live_exits, live_entries = set(live_exits or ()), set(live_entries or ())
    exits, entries = [], []
    for a in gate.get("actions") or []:
        k = a.get("key")
        ba = book.get(k)
        if not ba:
            continue
        b_act = ba.get("action")
        if (b_act == "CLOSE" or ba.get("exits_next_bar")) and a.get("in_pos") \
                and a.get("action") != "CLOSE" and not a.get("exits_next_bar") \
                and k not in live_exits:
            exits.append(k)
        elif b_act == "OPEN" and not a.get("in_pos") \
                and a.get("action") != "OPEN" and k not in live_entries:
            entries.append(k)
    return dict(exits=sorted(exits), entries=sorted(entries))


def rebalancing_moves(base_weights: dict, base_idle: float,
                      target: dict, target_idle: float,
                      idle_key: str = "SATA") -> list[dict]:
    """The trades that take the book you HOLD to the recommended ``target``.

    ``base_weights``/``base_idle`` must describe the book actually in force — the
    *published* Target Book the executor trades — not the engine's synthetic
    "what we hold now" (``signal_gated_allocation()["current"]``, which is the
    untilted optimal weights over the in-position names and matches neither what
    was published nor what is held).

    Weights are reported to whole percentage points, and the delta is the
    difference of those ROUNDED endpoints so a row always adds up.  Computing it
    from the raw weights instead let the endpoints and the arrow disagree
    (17% → 10% labelled ▼8pt, 18% → 22% labelled ▲5pt).  A move that rounds to
    0pt is omitted: it is invisible at display precision.

    Keys present only in ``base_weights`` are kept — a book published before a
    universe change can name a retired instrument, and the required action is to
    sell it down to zero.  Returns dicts of ``key``/``from_pct``/``to_pct``/
    ``delta_pct``, risk assets first (by target weight), idle cash last.
    """
    out = []
    for k in sorted(set(base_weights) | set(target),
                    key=lambda x: -float(target.get(x, 0.0))):
        c = round(float(base_weights.get(k, 0.0)) * 100)
        t = round(float(target.get(k, 0.0)) * 100)
        if t - c:
            out.append(dict(key=k, from_pct=c, to_pct=t, delta_pct=t - c))
    c = round(float(base_idle or 0.0) * 100)
    t = round(float(target_idle or 0.0) * 100)
    if t - c:
        out.append(dict(key=idle_key, from_pct=c, to_pct=t, delta_pct=t - c))
    return out


def book_move_reasons(prev_payload: dict, cur_payload: dict,
                      caps: dict | None = None,
                      idle_key: str = "SATA") -> list[dict]:
    """Why each Previous → Current Targetbook weight moved.

    Works from the two *published* payloads alone (schema
    ``overall-target-book/v1``) so the explanation describes what the optimizer
    actually committed at publish time, not a live recomputation.  Extends each
    ``rebalancing_moves`` row with a short ``reason`` derived from the books'
    recorded per-key actions and priority scores.  A published book's weights
    are the optimal base blend tilted by that morning's entry-priority score
    (momentum · sentiment · win-rate · risk-adjusted edge), water-filled to the
    per-instrument caps — so between two publishes a weight can only move
    because:

      * a signal fired → fresh entry, funded at its recorded priority;
      * a signal died → exit (the current book's recorded decision says why) or
        the instrument was retired from the universe entirely;
      * the priority tilt drifted → the water-fill re-spread weight across the
        uncapped names (a weight pinned at its cap is reported as cap-bound;
        a weight that moved *against* its own priority drift moved because the
        competing names' tilts shifted more);
      * everything the risk sleeve can't hold is the idle-cash (SATA) residual.

    Legacy books without an ``actions`` list degrade to weight-only reasons.
    Sub-point moves are omitted, same as ``rebalancing_moves``.
    """
    caps = caps or CAP_BY_KEY

    def _alloc(payload):
        w = {k: float(v) for k, v in (payload.get("weights") or {}).items()}
        idle = w.pop(idle_key, 0.0) + float(payload.get("cash_weight") or 0.0)
        return w, idle

    prev_w, prev_idle = _alloc(prev_payload)
    cur_w, cur_idle = _alloc(cur_payload)
    prev_act = {a.get("key"): a for a in (prev_payload.get("actions") or [])}
    cur_act = {a.get("key"): a for a in (cur_payload.get("actions") or [])}

    moves = rebalancing_moves(prev_w, prev_idle, cur_w, cur_idle, idle_key)
    for m in moves:
        k = m["key"]
        if k == idle_key:
            m["reason"] = ("the risk signals deploy less capital — the residual "
                           "parks in idle cash" if m["delta_pct"] > 0 else
                           "newly funded risk positions absorb previously idle "
                           "cash")
            continue
        fw, tw = float(prev_w.get(k, 0.0)), float(cur_w.get(k, 0.0))
        pa, ca = prev_act.get(k), cur_act.get(k)
        if fw < 5e-4 < tw:                               # flat → funded
            p = (ca or {}).get("priority")
            m["reason"] = ("fresh entry — "
                           + ((ca or {}).get("decision") or "signal fired long")
                           + (f" · funded at priority {p:.2f}"
                              if p is not None else ""))
        elif tw < 5e-4 < fw:                             # funded → flat
            if ca is None and cur_act:
                m["reason"] = ("retired from the trading universe — sold down "
                               "to zero")
            else:
                m["reason"] = ("signal exited — "
                               + ((ca or {}).get("decision")
                                  or "no longer signalling long"))
        else:                                            # resized hold
            cap = float(caps.get(k, 0.30))
            if abs(tw - cap) < 5e-4:
                m["reason"] = (f"filled to its {cap*100:.0f}% cap — takes all "
                               f"the weight the cap allows")
                continue
            pp = (pa or {}).get("priority")
            cp = (ca or {}).get("priority")
            if pp is not None and cp is not None and (cp - pp) * (tw - fw) > 0:
                up = cp > pp
                m["reason"] = (f"priority {'rose' if up else 'fell'} "
                               f"{pp:.2f} → {cp:.2f} (momentum · sentiment · "
                               f"win-rate · edge) — the optimizer re-tilted "
                               f"its slice {'up' if up else 'down'}")
            elif pp is not None and cp is not None:
                m["reason"] = (f"own priority only moved {pp:.2f} → {cp:.2f}; "
                               f"competing names' tilts shifted more, so the "
                               f"water-fill re-spread weight "
                               f"{'into' if tw > fw else 'out of'} this "
                               f"uncapped name")
            else:
                m["reason"] = ("re-tilted by the water-fill as competing "
                               "priorities shifted")
    return moves


def adjust_for_selection(weights: dict, cash: float, included) -> tuple:
    """Apply a user include/exclude choice to an allocation.

    Unticked (excluded) instruments are dropped and *their* weight is added to
    ``cash`` (SATA idle-cash in the Overall app) — the kept instruments' weights
    are left exactly as they were, with NO redistribution to the survivors.  So
    ``deployed + cash`` is invariant.  Shared by the Overall app's Live tab and
    the Target Book viewer so both behave identically.  Returns
    ``(adj_weights, adj_cash, excluded_keys, deployed, moved_to_cash)``.
    """
    inc = set(included)
    adj = {k: float(w) for k, w in weights.items() if k in inc}
    excluded = [k for k in weights if k not in inc]
    moved = sum(float(weights[k]) for k in excluded)
    return adj, float(cash) + moved, excluded, sum(adj.values()), moved


# ════════════════════════════════════════════════════════════════════════
# HISTORICAL SNAPSHOT — the strategy exactly as it stood on a past date.
# Positions and trades are read off the committed back-test series
# (``pos_series``, the normalised buy-&-hold price curve and the dated trade
# log), so those are the engine's genuine bar-by-bar decisions.  Caveat: the
# blend weights and per-sleeve strategy parameters applied to that date are
# TODAY'S (full-sample-fit) — the book percentages are a current-weights
# projection, not the as-of record (the app captions this).
# ════════════════════════════════════════════════════════════════════════
def asset_close_series(res: dict) -> pd.Series | None:
    """Actual daily close series over one instrument's back-test window.

    ``r['bh']`` is proportional to price in every engine (price/price₀ in the
    ticker/gold engines, capital×price/price₀ in the BTC CT engine), so one
    real fill anchors the scale.  Trade fills are preferred as anchors —
    ``last_close`` is mutated to the live spot quote by ``apply_spot``, which
    would skew the whole reconstructed history — falling back to the open
    position's fill, then to ``last_close``/bh[-1] for a sleeve that has never
    traded.  Returns ``None`` when no anchor exists or shapes disagree."""
    bh = np.asarray(res["r"]["bh"], float)
    idx = pd.DatetimeIndex(pd.Series(res["dates"]))
    if not len(bh) or len(bh) != len(idx):
        return None
    s = pd.Series(bh, index=idx)

    def _anchor() -> float | None:
        log = [t for t in (res["r"].get("trade_log") or []) if isinstance(t, dict)]
        for t in reversed(log):                      # newest fills first
            for dk, pks in (("exit_date", ("exit_px", "exit_price")),
                            ("entry_date", ("entry_px", "entry_price"))):
                d = t.get(dk)
                px = next((t[k] for k in pks if t.get(k) is not None), None)
                if d is None or px is None or not np.isfinite(px) or px <= 0:
                    continue
                d = pd.Timestamp(d)
                if d in s.index and float(s.loc[d]) > 0:
                    return float(px) / float(s.loc[d])
        pos = res.get("pos") or {}
        e_px = pos.get("entry_px_model") or pos.get("entry_px")
        if pos.get("in_pos") and e_px and pos.get("entry_date") is not None:
            d = pd.Timestamp(pos["entry_date"])
            if d in s.index and float(s.loc[d]) > 0:
                return float(e_px) / float(s.loc[d])
        last = res.get("last_close")
        if last and np.isfinite(last) and float(s.iloc[-1]) > 0:
            return float(last) / float(s.iloc[-1])
        return None

    sc = _anchor()
    return s * sc if sc else None


def snapshot_asof(results: list[dict], date) -> dict | None:
    """Every instrument's strategy state as of the last completed bar on/before
    ``date``.  Per row: the bar actually used (each sleeve's own calendar —
    weekends/holidays roll back), its official close and day-change, the
    position state (entry date/px, days held, unrealised P&L **marked at that
    bar**, not at today's price), and what happened ON that bar — ``OPEN``
    (entered), ``CLOSE`` (exited, with the trade's realised return and exit
    reason), ``HOLD``, or ``STAND ASIDE`` (flat).  Sleeves whose history starts
    after ``date`` are listed in ``not_live``.  ``None`` when no instrument has
    data by then."""
    date = pd.Timestamp(date)
    rows, not_live = [], []
    for res in results:
        pos_sub = res["pos_series"].loc[:date]
        closes = asset_close_series(res)
        px_sub = closes.loc[:date].dropna() if closes is not None else pd.Series(dtype=float)
        if not len(pos_sub) or not len(px_sub):
            not_live.append(res["key"])
            continue
        bar = pos_sub.index[-1]
        v = pos_sub.to_numpy(float)
        in_pos = bool(v[-1])
        prev_in = bool(v[-2]) if len(v) > 1 else False
        opened = in_pos and not prev_in
        closed = (not in_pos) and prev_in
        close_px = float(px_sub.iloc[-1])
        prev_px = float(px_sub.iloc[-2]) if len(px_sub) > 1 else np.nan
        dchg = ((close_px / prev_px - 1) * 100) if np.isfinite(prev_px) and prev_px else None

        entry_date = entry_px = upnl = days = None
        if in_pos:
            # entry bar = the start of the current 1-run (fills are that bar's close)
            z = np.nonzero(v == 0)[0]
            e_i = int(z[-1]) + 1 if len(z) else 0
            entry_date = pos_sub.index[e_i]
            e_val = float(closes.loc[entry_date]) if entry_date in closes.index else np.nan
            if np.isfinite(e_val) and e_val > 0:
                entry_px = e_val
                if np.isfinite(close_px):
                    upnl = (close_px / e_val - 1) * 100
            days = int((bar - entry_date).days)

        trade = None
        if closed:                              # the round-trip that exited on this bar
            log = [t for t in (res["r"].get("trade_log") or []) if isinstance(t, dict)]
            for t in reversed(log):
                if pd.Timestamp(t["exit_date"]) == bar:
                    trade = dict(
                        entry_date=pd.Timestamp(t["entry_date"]),
                        entry_px=next((float(t[k]) for k in ("entry_px", "entry_price")
                                       if t.get(k) is not None), None),
                        exit_px=next((float(t[k]) for k in ("exit_px", "exit_price")
                                      if t.get(k) is not None), None),
                        ret=float(t["ret"]),
                        reason=str(t.get("reason") or "signal exit"))
                    break
        action = ("CLOSE" if closed else "OPEN" if opened
                  else "HOLD" if in_pos else "STAND ASIDE")
        rows.append(dict(
            key=res["key"], name=res["name"], kind=res["kind"], parent=res["parent"],
            emoji=res["emoji"], accent=res["accent"], bar=bar, close=close_px,
            dchg=dchg, in_pos=in_pos, opened=opened, closed=closed, action=action,
            entry_date=entry_date, entry_px=entry_px, upnl=upnl, days=days,
            trade=trade))
    if not rows:
        return None
    return dict(date=date, asof=max(r["bar"] for r in rows), rows=rows,
                not_live=not_live,
                n_active=int(sum(r["in_pos"] for r in rows)),
                n_open=int(sum(r["opened"] for r in rows)),
                n_close=int(sum(r["closed"] for r in rows)))


def historical_allocation(snap: dict, base_weights: dict,
                          caps: dict | None = None) -> dict:
    """The book the strategy held on the snapshot bar: the blend's optimal
    weights water-filled (to the profile caps) over the sleeves in position on
    that bar, the undeployed remainder parked in SATA — the same construction
    as the Live tab's *current book*.  The entry-priority tilt uses live-only
    inputs (spot momentum, today's sentiment), so it is not applied
    retrospectively."""
    caps = caps or CAP_BY_KEY
    held = {r["key"]: max(base_weights.get(r["key"], 0.0), 1e-6)
            for r in snap["rows"] if r["in_pos"]}
    book = _waterfill(held, caps)
    return dict(book=book, sata=max(1.0 - sum(book.values()), 0.0))


# ── historical replay of the signal-gated, priority-tilted allocation ──────
def priority_component_history(results: list[dict], index: pd.Index,
                               wr_window: int | None = None,
                               sharpe_window: int | None = None
                               ) -> dict[str, pd.DataFrame]:
    """Causal daily history of every ``compute_priorities`` input, per key.

    The live gate scores candidates on five components; each has an exact
    as-of-that-day analogue reconstructable from data available at that bar's
    close (no look-ahead):

      momentum   parent signal close vs its 50-day SMA — the same read
                 ``run_asset`` takes live, computed on the parent group's
                 primary price path (``r['bh']`` is proportional to price, and
                 close/SMA is scale-invariant, so no fill anchor is needed).
      sentiment  ``tc.macro_sentiment`` is already a causal rolling series
                 (252-day rolling z-scores → rolling percentile rank); the
                 live gate just reads its last value, so the historical value
                 IS the series at that bar.  Parent config resolved via
                 ``overall_config``; sleeves whose macro data can't be loaded
                 fall back to the 0.5 neutral the live scorer uses for NaN.
      win_rate   expanding win fraction over the sleeve's CLOSED trades as of
                 each bar (the live gate uses the full-sample win rate, which
                 an investor mid-history could not have known).  Neutral 0.5
                 before the first closed trade.
      sharpe     expanding annualised Sharpe of the sleeve's strategy returns
                 (≥60 bars; NaN before that → 0.5 via ``_minmax``).  Raw value
                 — min-max across the day's candidates happens at replay time,
                 exactly like ``compute_priorities``.
      regime     close > 20-day SMA and SMA rising vs 5 bars ago — the same
                 ``bull_regime`` rule ``compute_trend_signatures`` applies,
                 evaluated on the parent's daily closes.

    Returns ``{key: DataFrame(mom, sent, wr, sharpe_raw, regime)}`` reindexed
    and forward-filled onto ``index`` (each value stands until the next bar of
    that sleeve's own calendar).  NOT shifted — the caller decides the
    information lag.

    Adaptivity experiments (defaults reproduce the published behaviour):
    ``wr_window=N`` scores win-rate over only the LAST N closed trades and
    ``sharpe_window=M`` scores Sharpe over only the last M bars — rolling
    reads that respond to recent form instead of lifetime averages.  Still
    strictly as-of."""
    # parent-level reads (momentum / sentiment / regime), shared by siblings
    groups: dict[str, list[dict]] = {}
    for res in results:
        groups.setdefault(res["parent"], []).append(res)
    parent_comp: dict[str, dict[str, pd.Series]] = {}
    for parent, members in groups.items():
        prim = next((m for m in members if m["key"] == parent), members[0])
        px = pd.Series(np.asarray(prim["r"]["bh"], float),
                       index=pd.DatetimeIndex(pd.Series(prim["dates"])))
        sma50 = px.rolling(50, min_periods=20).mean()
        mom = px / sma50 - 1.0
        sma20 = px.rolling(20, min_periods=5).mean()
        reg = ((px > sma20) & (sma20 > sma20.shift(5))).astype(float)
        try:
            cfg = overall_config(parent)
            import ticker_core as _tc
            s = _tc.macro_sentiment(cfg, _load_daily(cfg))
            sent = (s / 100.0).clip(0.0, 1.0)
        except Exception:
            sent = pd.Series(0.5, index=px.index)
        parent_comp[parent] = dict(mom=mom, sent=sent, reg=reg)

    out = {}
    for res in results:
        pc = parent_comp[res["parent"]]
        # expanding win rate stepped at each closed trade's exit bar
        log = [t for t in (res["r"].get("trade_log") or res["r"].get("trades") or [])
               if isinstance(t, dict) and t.get("exit_date") is not None]
        if log:
            ex = pd.Series([float(t["ret"]) > 0 for t in log],
                           index=pd.DatetimeIndex([pd.Timestamp(t["exit_date"])
                                                   for t in log])).sort_index()
            if wr_window:                       # rolling: last N closed trades
                wr = ex.astype(float).rolling(wr_window, min_periods=1).mean() \
                       .groupby(level=0).last()  # same-day exits → one bar
            else:                                # expanding lifetime win rate
                wr = ex.astype(float).groupby(level=0).sum().cumsum() / \
                     ex.groupby(level=0).size().cumsum()
        else:
            wr = pd.Series(dtype=float)
        r = res["ret"]
        if sharpe_window:
            mp = min(60, sharpe_window)
            shp = (r.rolling(sharpe_window, min_periods=mp).mean()
                   / r.rolling(sharpe_window, min_periods=mp).std()) * np.sqrt(TRADING_DAYS)
        else:
            shp = (r.expanding(60).mean() / r.expanding(60).std()) * np.sqrt(TRADING_DAYS)
        df = pd.DataFrame(dict(
            mom=pc["mom"].reindex(index).ffill(),
            sent=pc["sent"].reindex(index).ffill().fillna(0.5),
            wr=wr.reindex(index).ffill().fillna(0.5),
            sharpe_raw=shp.reindex(index).ffill(),
            regime=pc["reg"].reindex(index).ffill().fillna(0.0),
        ))
        out[res["key"]] = df
    return out


def walkforward_anchors(rets: pd.DataFrame, pos: pd.DataFrame | None = None,
                        caps: dict | None = None, mdd_floor: float = -0.35,
                        objective: str = "balanced",
                        sata_daily: float = SATA_DAILY, min_hist: int = 250,
                        n_samples: int = 8000, seed: int = 7,
                        refit: str = "Q",
                        fit_window: int | None = None) -> list[tuple]:
    """Expanding-window anchor-weight schedule with NO look-ahead.

    The live gate anchors position sizes on the optimiser's weights; fitting
    those on the full history and applying them retroactively is exactly the
    look-ahead the fixed-weight back-test carried.  This schedule removes it:

      * from inception until ``min_hist`` bars exist, the anchors are the
        cap-normalised EQUAL weights — a constant that uses no return data;
      * at each QUARTER start thereafter (Jan/Apr/Jul/Oct 1 — the published
        cadence, adopted from the V3 adaptivity eval where it beat annual
        refits on both return and Sharpe) the optimiser is re-fit on the data
        strictly BEFORE that date (same caps / drawdown budget / objective as
        the profile, ``fundamental=False`` — the mid-2026 forward view is
        hindsight relative to history, so it must not tilt the replay), and
        those weights anchor the book until the next refit.

    Returns ``[(effective_date, weights_dict), …]`` sorted ascending; entry 0
    covers the warm-up.  A sleeve with little or no data inside a fit window
    may carry an arbitrary anchor — harmless, since the replay water-fills
    over the sleeves actually in the market and the anchor only sets relative
    size once the sleeve is live.

    Adaptivity experiments (defaults reproduce the published behaviour):
    ``refit="A"`` reverts to the pre-V3 annual (each Jan 1) cadence;
    ``fit_window=N`` fits on only the trailing N bars (a ROLLING window that
    forgets old regimes) instead of the expanding full history.  Both remain
    strictly causal — every fit still sees only data before its effective
    date."""
    caps = caps or CAP_BY_KEY
    idx = rets.index
    cols = list(rets.columns)
    cap_vec = np.array([caps.get(c, 0.30) for c in cols])
    ew = _cap_normalise(np.full(len(cols), 1.0 / len(cols)), cap_vec)
    sched = [(idx[0], {c: float(w) for c, w in zip(cols, ew)})]
    if refit == "Q":
        refit_dates = [pd.Timestamp(f"{y}-{m:02d}-01")
                       for y in range(idx[0].year, idx[-1].year + 1)
                       for m in (1, 4, 7, 10)]
    else:
        refit_dates = [pd.Timestamp(f"{y}-01-01")
                       for y in range(idx[0].year + 1, idx[-1].year + 1)]
    for d in refit_dates:
        prior = idx[idx < d]
        if len(prior) < min_hist or d > idx[-1]:
            continue
        fit_r = rets.loc[:prior[-1]]
        if fit_window:
            fit_r = fit_r.iloc[-fit_window:]
        o = optimize_weights(fit_r, caps=caps, n_samples=n_samples,
                             seed=seed, mdd_floor=mdd_floor,
                             pos=(pos.loc[fit_r.index] if pos is not None else None),
                             sata_daily=sata_daily, objective=objective,
                             fundamental=False)
        sched.append((d, o["optimal"]["weights"]))
    return sched


def replay_gated_allocation(results: list[dict],
                            base_weights: dict[str, float] | None = None,
                            caps: dict | None = None,
                            sata_daily: float = SATA_DAILY,
                            tilt: bool = True,
                            anchors: list[tuple] | None = None,
                            wr_window: int | None = None,
                            sharpe_window: int | None = None,
                            penalty: dict | None = None) -> dict:
    """Historical replay of ``signal_gated_allocation`` — what the daily
    gate/tilt/water-fill book would have earned, decided each day from data
    available at the PREVIOUS bar's close.

    Construction, mirroring the live gate:
      * the funded set on day *t* is the sleeves the engines hold IN THE
        MARKET on day *t* (``pos_series`` — position on *t* was decided by
        signals at the close of *t−1*, so the set is causal and already
        encodes next-bar execution, committed exits and fresh entries).
        Position state is CARRIED across a sleeve's non-trading days
        (``position_matrix(carry=True)``): on the union calendar the crypto
        sleeves put weekends into the index, and without the carry every
        equity position looked "sold" each Friday close — water-filling the
        whole weekend book into crypto (≈84% vs ≈35% on weekdays) or 100%
        SATA, and manufacturing ~46% phantom turnover into weekend bars.
        Carried sleeves keep their weight and contribute 0 return until
        their next bar — exactly what holding through a closed market is;
      * each funded sleeve's raw weight is its optimal anchor ×
        ``(0.5 + priority)``, priority scored by ``PRIORITY_WEIGHTS`` over the
        as-of components of ``priority_component_history`` **lagged one bar**
        and min-max normalised across that day's funded set — exactly
        ``compute_priorities``;  ``tilt=False`` skips the tilt (anchor × 1.0),
        isolating the pure gate/concentration effect (the same construction as
        ``historical_allocation``'s per-snapshot book);
      * raw weights are water-filled to the profile caps (deployed book sums
        to 1 when the caps allow), the remainder — and every fully-flat day —
        earns the SATA daily yield.  SATA's coupon is ``0.13/250`` per
        BUSINESS day, so it accrues only on weekday bars — crediting it on
        the crypto calendar's weekend bars would compound ~13%/250 × 365
        ≈ 19%/yr instead of the intended ~13%.

    No transaction costs are modelled; ``turnover`` (mean and total daily
    one-way weight movement, SATA leg included) is reported so the cost
    surface is visible.  Returns daily returns/equity/metrics plus the full
    daily weight matrix (``weights`` columns = instruments, ``sata`` series).

    Anchors come from ``base_weights`` (one static dict — carries whatever
    fitting bias produced it) OR ``anchors`` (a ``walkforward_anchors``
    schedule; each day uses the latest entry effective on/before it, so the
    anchors are look-ahead-free too).  Exactly one must be supplied.

    Adaptivity experiments (defaults reproduce the published behaviour):
    ``wr_window``/``sharpe_window`` make the priority's win-rate/Sharpe reads
    rolling instead of lifetime (see ``priority_component_history``);
    ``penalty=dict(window=126, floor=0.0, mult=0.5)`` is a *penalty box* —
    a funded sleeve whose rolling ``window``-bar Sharpe (lagged one bar) is
    below ``floor`` has its raw weight multiplied by ``mult`` before the
    water-fill, automatically de-rating persistent underperformers.  All
    strictly as-of.
    """
    caps = caps or CAP_BY_KEY
    rets = returns_matrix(results)
    idx = rets.index
    keys = list(rets.columns)
    biz = np.asarray(idx.dayofweek < 5)       # SATA pays on business days only
    if (base_weights is None) == (anchors is None):
        raise ValueError("supply exactly one of base_weights / anchors")
    if anchors:
        a_dates = np.array([pd.Timestamp(d).to_datetime64() for d, _ in anchors])
        a_pick = np.maximum(np.searchsorted(a_dates, idx.values, side="right") - 1, 0)
        day_anchor = [anchors[i][1] for i in a_pick]
    else:
        day_anchor = None
    pos = position_matrix(results, idx, carry=True)
    comp = priority_component_history(results, idx, wr_window=wr_window,
                                      sharpe_window=sharpe_window)
    # decision at close t−1 drives the book that earns day t's return
    lag = {k: comp[k].shift(1) for k in keys}

    R = np.nan_to_num(rets.to_numpy(float))
    P = pos.to_numpy(float)
    mom_m = np.column_stack([lag[k]["mom"].to_numpy(float) for k in keys])
    sent_m = np.column_stack([lag[k]["sent"].to_numpy(float) for k in keys])
    wr_m = np.column_stack([lag[k]["wr"].to_numpy(float) for k in keys])
    shp_m = np.column_stack([lag[k]["sharpe_raw"].to_numpy(float) for k in keys])
    reg_m = np.column_stack([lag[k]["regime"].to_numpy(float) for k in keys])
    pw = PRIORITY_WEIGHTS
    pen_m = None
    if penalty:
        p_win = int(penalty.get("window", 126))
        p_mp = min(60, p_win)
        pen_m = np.column_stack([
            ((rets[k].rolling(p_win, min_periods=p_mp).mean()
              / rets[k].rolling(p_win, min_periods=p_mp).std())
             * np.sqrt(TRADING_DAYS)).shift(1).to_numpy(float)
            for k in keys])
        p_floor = float(penalty.get("floor", 0.0))
        p_mult = float(penalty.get("mult", 0.5))

    port = np.zeros(len(idx))
    W = np.zeros((len(idx), len(keys)))
    sata_w = np.zeros(len(idx))
    for t in range(len(idx)):
        act = [j for j in range(len(keys)) if P[t, j] > 0]
        if not act:
            port[t] = sata_daily if biz[t] else 0.0
            sata_w[t] = 1.0
            continue
        bw = day_anchor[t] if day_anchor is not None else base_weights
        if tilt:
            mom = _minmax({j: mom_m[t, j] for j in act})
            shp = _minmax({j: shp_m[t, j] for j in act})
            raw = {}
            for j in act:
                s01 = sent_m[t, j]
                s01 = 0.5 if s01 != s01 else s01
                w01 = wr_m[t, j]
                w01 = 0.5 if w01 != w01 else w01
                score = (pw["momentum"] * mom[j] + pw["sentiment"] * s01 +
                         pw["win_rate"] * w01 + pw["sharpe"] * shp[j] +
                         pw["regime"] * (reg_m[t, j] if reg_m[t, j] == reg_m[t, j] else 0.0))
                raw[j] = max(bw.get(keys[j], 0.0), 1e-6) * (0.5 + score)
        else:
            raw = {j: max(bw.get(keys[j], 0.0), 1e-6) for j in act}
        if pen_m is not None:                   # penalty box: de-rate cold sleeves
            for j in act:
                v = pen_m[t, j]
                if v == v and v < p_floor:
                    raw[j] *= p_mult
        target = _waterfill(raw, {j: caps.get(keys[j], 0.30) for j in raw})
        dep = 0.0
        for j, w in target.items():
            W[t, j] = w
            dep += w
        sata_w[t] = max(1.0 - dep, 0.0)
        port[t] = float(np.dot(W[t], R[t])) + \
            (sata_w[t] * sata_daily if biz[t] else 0.0)

    daily = pd.Series(port, index=idx)
    eq = _equity(daily)
    full = np.column_stack([W, sata_w])
    turno = 0.5 * np.abs(np.diff(full, axis=0)).sum(axis=1)
    return dict(
        ret=daily, equity=eq, metrics=curve_metrics(eq),
        weights=pd.DataFrame(W, index=idx, columns=keys),
        sata=pd.Series(sata_w, index=idx),
        # per-day signal/position truth (the funded set) — lets the trade
        # log tell real signal flips from optimizer re-sizes through zero
        active=pd.DataFrame(P > 0, index=idx, columns=keys),
        turnover=dict(mean=float(turno.mean()) if len(turno) else 0.0,
                      p95=float(np.percentile(turno, 95)) if len(turno) else 0.0,
                      days_traded=float((turno > 0.005).mean()) if len(turno) else 0.0),
        tilt=tilt)


def walkforward_gated_replay(results: list[dict], caps: dict | None = None,
                             mdd_floor: float = -0.35,
                             objective: str = "balanced",
                             sata_daily: float = SATA_DAILY, tilt: bool = True,
                             min_hist: int = 250, n_samples: int = 8000,
                             seed: int = 7, refit: str = "Q",
                             fit_window: int | None = None,
                             wr_window: int | None = None,
                             sharpe_window: int | None = None,
                             penalty: dict | None = None) -> dict:
    """The look-ahead-free Overall back-test: ``replay_gated_allocation`` run
    on a ``walkforward_anchors`` schedule.  Every input to a given day's book —
    anchor weights, funded set, priority components — is computable from data
    available at the previous close.  This is what the app publishes as the
    combined back-test; the full-sample optimiser weights remain in use only
    for TODAY'S live book, where all history to date is legitimately known.
    The schedule is attached as ``anchors``.

    The adaptivity-experiment knobs (``refit``/``fit_window``/``wr_window``/
    ``sharpe_window``/``penalty``) pass straight through to
    ``walkforward_anchors`` / ``replay_gated_allocation``; the defaults
    reproduce the published strategy exactly (see
    ``scripts/eval_adaptive_variants.py`` for the comparison harness)."""
    rets = returns_matrix(results)
    pos = position_matrix(results, rets.index)
    anchors = walkforward_anchors(rets, pos=pos, caps=caps, mdd_floor=mdd_floor,
                                  objective=objective, sata_daily=sata_daily,
                                  min_hist=min_hist, n_samples=n_samples, seed=seed,
                                  refit=refit, fit_window=fit_window)
    rep = replay_gated_allocation(results, caps=caps, sata_daily=sata_daily,
                                  tilt=tilt, anchors=anchors,
                                  wr_window=wr_window, sharpe_window=sharpe_window,
                                  penalty=penalty)
    rep["anchors"] = anchors
    return rep


def period_metrics_from_ret(daily_ret: pd.Series, periods: list[tuple]) -> list[dict]:
    """``period_breakdown`` for an already-built daily return stream (the
    replay's) — same row shape, sliced instead of recombined."""
    rows = []
    for lbl, s, e in periods:
        sub = daily_ret.loc[pd.Timestamp(s):(pd.Timestamp(e) if e else None)]
        if len(sub) < 5:
            continue
        rows.append(dict(label=lbl, start=s, end=e, **curve_metrics(_equity(sub))))
    return rows


def pnl_attribution_replay(returns: pd.DataFrame, weights: pd.DataFrame,
                           sata_w: pd.Series, start,
                           sata_daily: float = SATA_DAILY) -> dict | None:
    """Exact per-sleeve attribution of the REPLAYED book's P&L since ``start``
    — the daily-weight analogue of ``pnl_attribution_since`` (same output
    shape, same to-float-precision additivity: per-key dollars + SATA = the
    replay curve's re-based total return).  ``weights``/``sata_w`` are the
    matrices ``replay_gated_allocation`` returns; the anchor bar is the cost
    basis, so its return is not counted."""
    sub = returns.loc[pd.Timestamp(start):]
    if len(sub) < 2:
        return None
    W = weights.reindex(sub.index).fillna(0.0).to_numpy(float)
    contrib = W * np.nan_to_num(sub.to_numpy(float))
    # same business-day-only SATA accrual as the replay, or the parts stop
    # summing to the curve
    biz = np.asarray(sub.index.dayofweek < 5)
    sata_term = (sata_w.reindex(sub.index).fillna(1.0).to_numpy(float)
                 * sata_daily * biz)
    contrib[0, :] = 0.0
    sata_term[0] = 0.0
    port = contrib.sum(axis=1) + sata_term
    v_prev = np.concatenate([[1.0], np.cumprod(1.0 + port)[:-1]])
    dollars = (contrib * v_prev[:, None]).sum(axis=0)
    return dict(per_key={c: float(v) for c, v in zip(sub.columns, dollars)},
                sata=float((sata_term * v_prev).sum()),
                total=float(np.prod(1.0 + port) - 1.0),
                start=sub.index[0], end=sub.index[-1])


def pnl_daily_replay(returns: pd.DataFrame, weights: pd.DataFrame,
                     sata_w: pd.Series, start,
                     sata_daily: float = SATA_DAILY) -> dict | None:
    """Day-by-day dollar P&L of the daily-weight book since ``start`` —
    ``pnl_attribution_replay`` with the day axis KEPT instead of summed
    away.  Same maths, same conventions (anchor bar = cost basis, earns
    nothing; business-day-only SATA accrual; each day's return scaled by
    the blend's compounded value at the previous close), so the figures
    telescope exactly: every day's ``total`` is the change in the blend's
    re-based equity over that bar, and the per-day values sum over days to
    ``pnl_attribution_replay``'s ``per_key``/``sata``/``total`` to float
    precision.  Source-agnostic like the attribution: the replay's and the
    as-published record's weight matrices both feed it unchanged.

    All dollar figures are per $1 invested at the anchor close — multiply
    by the portfolio value for dollars.  Returns ``per_key`` (DataFrame:
    one column per sleeve, one row per day), ``sata`` (Series), ``total``
    (Series — the day's whole-blend P&L) and ``ret`` (Series — the day's
    blend % return); ``None`` with <2 bars on/after ``start``."""
    sub = returns.loc[pd.Timestamp(start):]
    if len(sub) < 2:
        return None
    W = weights.reindex(sub.index).fillna(0.0).to_numpy(float)
    contrib = W * np.nan_to_num(sub.to_numpy(float))
    biz = np.asarray(sub.index.dayofweek < 5)
    sata_term = (sata_w.reindex(sub.index).fillna(1.0).to_numpy(float)
                 * sata_daily * biz)
    contrib[0, :] = 0.0
    sata_term[0] = 0.0
    port = contrib.sum(axis=1) + sata_term
    v_prev = np.concatenate([[1.0], np.cumprod(1.0 + port)[:-1]])
    per_key = pd.DataFrame(contrib * v_prev[:, None],
                           index=sub.index, columns=sub.columns)
    sata_d = pd.Series(sata_term * v_prev, index=sub.index)
    return dict(per_key=per_key, sata=sata_d,
                total=per_key.sum(axis=1) + sata_d,
                ret=pd.Series(port, index=sub.index))


# ── as-published record — the books that ACTUALLY traded ────────────────────
# The publisher archives every daily target book it commits (one JSON per
# signal day, ``data/overall/book_archive/<as_of>.json`` — see
# ``target_book.archive_book``).  Those books ARE the strategy's real past:
# the optimiser re-sizes the book every morning, so every daily trim/re-size
# it ever ordered is baked into the archived weights.  Replaying them gives
# the exact historical performance record, as opposed to the walk-forward
# replay's simulation of the same rules.
BOOK_ARCHIVE_DIR = _REPO_ROOT / "data" / "overall" / "book_archive"

# Deliberate strategy-logic version — single source of truth (and bump
# instructions) in ``app/strategy_version.py``, dependency-light so every UI
# can badge it; re-exported here for the publisher and the replay.  The
# publisher stamps it (plus the publishing commit's SHA) into every book it
# commits, and the P&L section segments the as-published record wherever the
# stamp changes, so performance from different generations of the strategy is
# never silently conflated.  Books published before stamping existed are
# labelled from the committed side-car ``data/overall/book_versions.json``
# (see ``scripts/backfill_book_versions.py``) as ``pre-v1``.
from strategy_version import (STRATEGY_VERSION,        # noqa: E402,F401
                              STRATEGY_VERSION_START)  # noqa: E402,F401
BOOK_VERSIONS_JSON = _REPO_ROOT / "data" / "overall" / "book_versions.json"


def data_vintage() -> dict:
    """Compact provenance stamp for the headline back-test numbers: the BTC
    vintage (data/backtest manifest) plus one combined hash over every pinned
    sleeve snapshot manifest.  Every number the Overall app shows is a pure
    function of (strategy version, these hashes) — so when a figure changes,
    this stamp changes with it, and an unchanged stamp guarantees unchanged
    back-tests.  Best-effort: missing manifests appear as '—'."""
    import hashlib as _hl
    import json as _json
    out = dict(btc_sha="—", btc_to="—", sleeves_sha="—", n_sleeves=0)
    try:
        man = _json.loads((_REPO_ROOT / "data" / "backtest"
                           / "manifest.json").read_text())
        rf = (man.get("datasets") or {}).get("raw_features_daily") or {}
        out["btc_sha"] = str(rf.get("checksum_sha256", "—"))[:8]
        out["btc_to"] = rf.get("date_to", "—")
    except Exception:
        pass
    try:
        import data_gate as _dg
        parts = []
        for key in ticker_config.APP_KEYS:
            m = _dg.read_manifest(_dg.ticker_spec(get_config(key)))
            if m.get("checksum_sha256"):
                parts.append(f"{key}:{m['checksum_sha256']}")
        gman = _dg.read_manifest(_REPO_ROOT / "data" / "gldm"
                                 / "gldm_macro_daily_manifest.json")
        if gman.get("checksum_sha256"):
            parts.append(f"GLDM:{gman['checksum_sha256']}")
        if parts:
            out["sleeves_sha"] = _hl.sha256(
                "|".join(sorted(parts)).encode()).hexdigest()[:8]
            out["n_sleeves"] = len(parts)
    except Exception:
        pass
    return out


def load_book_version_map(path: Path | None = None) -> dict:
    """The backfilled ``as_of → {code_sha, strategy_version}`` side-car for
    archived books that predate inline version stamping.  ``{}`` when absent —
    unmatched books simply render as ``unstamped``."""
    import json
    try:
        payload = json.loads(Path(path or BOOK_VERSIONS_JSON).read_text())
        return dict(payload.get("books") or {})
    except Exception:
        return {}


def load_published_books(archive_dir: Path | None = None) -> list[dict]:
    """Every archived as-published target book, parsed and sorted by
    ``as_of``.  Unreadable files are skipped; a record needs at least an
    ``as_of`` and a ``weights`` dict to count."""
    import json
    books = []
    for p in sorted(Path(archive_dir or BOOK_ARCHIVE_DIR).glob("*.json")):
        try:
            b = json.loads(p.read_text())
            if b.get("as_of") and isinstance(b.get("weights"), dict):
                books.append(b)
        except Exception:
            continue
    books.sort(key=lambda b: pd.Timestamp(b["as_of"]))
    return books


def published_book_replay(returns: pd.DataFrame, books: list[dict],
                          sata_daily: float = SATA_DAILY,
                          version_map: dict | None = None,
                          only_version: str | None = None,
                          min_as_of=None) -> dict | None:
    """Compound the ACTUALLY-PUBLISHED daily books — the exact historical
    record, daily optimiser trims included — into the same shapes the
    walk-forward replay returns (``ret``/``equity``/``weights``/``sata``), so
    the P&L section's every calculation, toggle and metric can run off either
    source unchanged.

    Day *d* is earned by the latest archived book with ``as_of`` strictly
    before *d* — the decided-at-previous-close convention shared with the
    replay and the 🩺 health monitor's ``realized_book_returns`` (a book
    published from bar *D*'s committed signals trades at *D*+1).  The first
    archived ``as_of`` bar is the cost basis: it displays the first book (put
    on at that close) but earns nothing.  Sleeve returns come from the same
    ``returns_matrix`` as the replay; the book's cash remainder earns the
    SATA coupon on business days.  Gross of costs/fills, like the replay.

    A LIVE book parks its idle remainder as a real ``SATA`` weights entry
    (``cash_weight`` 0 — see the publisher); that leg is folded back into the
    cash/SATA term here so it accrues the coupon instead of being dropped as
    an unknown sleeve.

    Strategy-logic provenance: each book is labelled with its
    ``strategy_version`` — the inline stamp the publisher writes (signature-
    covered) when present, else the backfilled ``version_map`` side-car
    (``load_book_version_map``), else ``unstamped`` — and consecutive
    same-version books are grouped into ``version_spans`` so the UI can mark
    every point where the record switches strategy generations.
    ``only_version`` restricts the replay to books carrying that label — the
    "current-logic books only" view, guaranteed free of old-logic
    conflation — and ``min_as_of`` additionally drops books from earlier
    signal days (``STRATEGY_VERSION_START``: a same-day re-publish of an
    older bar under new code stays out of the current version's record).

    Extras beyond the replay shape: ``books`` — one record per archived
    publish (as_of / profile / book_mode / version / code_sha / weights /
    cash / one-way ``turnover`` vs the previous book, cash leg included) for
    the daily-trims table — ``version_spans`` as above, and ``dropped``, any
    book key absent from ``returns`` (its weight earns nothing rather than
    silently borrowing another sleeve's return).  Returns ``None`` when
    there is nothing replayable (<2 bars covered)."""
    version_map = version_map or {}
    parsed = []
    for b in books or []:
        try:
            as_of = pd.Timestamp(b["as_of"])
            w = {k: float(v) for k, v in (b.get("weights") or {}).items()}
            # live books hold idle capital as an actual SATA position — that
            # IS the cash/SATA leg, not a market sleeve
            cash = float(b.get("cash_weight", 0.0) or 0.0) + w.pop("SATA", 0.0)
            side = version_map.get(str(as_of.date()), {})
            parsed.append(dict(
                as_of=as_of, weights=w, cash=cash,
                profile=str(b.get("profile") or "—"),
                book_mode=str(b.get("book_mode") or "paper"),
                version=str(b.get("strategy_version")
                            or side.get("strategy_version") or "unstamped"),
                code_sha=(str(b.get("code_sha") or side.get("code_sha") or "")
                          [:12] or None)))
        except Exception:
            continue
    parsed.sort(key=lambda r: r["as_of"])
    if only_version is not None:
        parsed = [r for r in parsed if r["version"] == only_version]
    if min_as_of is not None:
        parsed = [r for r in parsed if r["as_of"] >= pd.Timestamp(min_as_of)]
    if not parsed or returns is None or returns.empty:
        return None
    idx = returns.index[returns.index >= parsed[0]["as_of"]]
    if len(idx) < 2:
        return None
    a_dates = np.array([r["as_of"].to_datetime64() for r in parsed])
    keys = list(returns.columns)
    col = {k: j for j, k in enumerate(keys)}
    R = np.nan_to_num(returns.reindex(idx).to_numpy(float))
    biz = np.asarray(idx.dayofweek < 5)
    W = np.zeros((len(idx), len(keys)))
    sata_w = np.zeros(len(idx))
    port = np.zeros(len(idx))
    dropped = set()
    for t, d in enumerate(idx):
        i = int(np.searchsorted(a_dates, d.to_datetime64(), side="left")) - 1
        b = parsed[max(i, 0)]                  # t=0 → first book, earns 0 below
        for k, wt in b["weights"].items():
            j = col.get(k)
            if j is None:
                dropped.add(k)
                continue
            W[t, j] = wt
        sata_w[t] = b["cash"]
        if t > 0:
            port[t] = float(np.dot(W[t], R[t])) + \
                (sata_w[t] * sata_daily if biz[t] else 0.0)
    # one-way turnover between consecutive publishes — the daily trims, made
    # visible (0.5 × Σ|Δw| over the union of keys, cash leg included)
    parsed[0]["turnover"] = None
    for prev, cur in zip(parsed, parsed[1:]):
        ks = set(prev["weights"]) | set(cur["weights"])
        cur["turnover"] = 0.5 * (
            sum(abs(cur["weights"].get(k, 0.0) - prev["weights"].get(k, 0.0))
                for k in ks) + abs(cur["cash"] - prev["cash"]))
    # consecutive same-version books → one span each; a span boundary is a
    # strategy-generation switch the UI must surface, never silently blend
    spans = []
    for r in parsed:
        if spans and spans[-1]["version"] == r["version"]:
            spans[-1]["end"] = r["as_of"]
            spans[-1]["n_books"] += 1
        else:
            spans.append(dict(version=r["version"], start=r["as_of"],
                              end=r["as_of"], n_books=1))
    daily = pd.Series(port, index=idx)
    eq = _equity(daily)
    return dict(ret=daily, equity=eq, metrics=curve_metrics(eq),
                weights=pd.DataFrame(W, index=idx, columns=keys),
                sata=pd.Series(sata_w, index=idx),
                books=parsed, version_spans=spans, dropped=sorted(dropped))


def _book_legs(b: dict) -> tuple:
    """``(as_of, {sleeve: weight}, cash)`` for either book shape — a raw
    archived JSON record or a ``published_book_replay`` parsed entry.  The
    live publisher parks idle capital as a real ``SATA`` weights entry, so
    that leg is folded into cash exactly as the replay folds it."""
    as_of = pd.Timestamp(b["as_of"])
    w = {k: float(v) for k, v in (b.get("weights") or {}).items()}
    cash = float(b.get("cash", b.get("cash_weight", 0.0)) or 0.0) \
        + w.pop("SATA", 0.0)
    return as_of, w, cash


def _book_ticket(prev: dict, cur: dict, min_delta: float) -> dict | None:
    """The trade ticket between two consecutive published books, in the
    exact per-day shape ``daily_trade_log`` emits (dated at the later book's
    ``as_of`` — the close its weights are put on under the
    decided-at-previous-close convention).  ``None`` when no sleeve moved
    beyond ``min_delta``, matching the log's quiet-day omission.  Signal
    changes are inferred from the weights (a full open or close), the only
    classification the archive supports."""
    _, w0, c0 = _book_legs(prev)
    as_of, w1, c1 = _book_legs(cur)
    actions = []
    for k in sorted(set(w0) | set(w1)):
        was, now = w0.get(k, 0.0), w1.get(k, 0.0)
        d = now - was
        if abs(d) < min_delta:
            continue
        if was < min_delta:
            act = "buy"
        elif now < min_delta:
            act = "sell"
        else:
            act = "add" if d > 0 else "trim"
        actions.append(dict(key=k, w0=was, w1=now, delta=d, action=act,
                            signal_change=act in ("buy", "sell"),
                            pnl=None, entry_date=None, sold=None, basis=None))
    if not actions:
        return None
    actions.sort(key=lambda a: -abs(a["delta"]))
    gross = sum(abs(w1.get(k, 0.0) - w0.get(k, 0.0))
                for k in set(w0) | set(w1))
    return dict(
        date=as_of, actions=actions, gross=gross,
        turnover=0.5 * (gross + abs(c1 - c0)), cash0=c0, cash1=c1,
        n_buys=sum(a["action"] == "buy" for a in actions),
        n_sells=sum(a["action"] == "sell" for a in actions),
        n_resize=sum(not a["signal_change"] for a in actions))


def pending_book_tickets(books: list[dict], covered_through,
                         min_delta: float = 0.0005) -> list[dict]:
    """Tickets from archived books the replay's weight matrix does not cover
    yet — newest first, in ``daily_trade_log``'s per-day shape plus
    ``pending=True``.

    ``published_book_replay`` maps day *d* to the latest book with ``as_of``
    strictly before *d*, so a book only reaches the weight matrix once a
    LATER bar has printed: the book published this morning (``as_of`` =
    the last completed bar) is in nobody's weights and its ticket cannot
    appear in the log until tomorrow's bar exists.  That one-bar blind spot
    reads as "the trade log stopped updating" on the very days it matters —
    the morning a position is opened or closed.  Feeding these rows into the
    log closes it: they carry no P&L (the bar they will earn has not
    happened), which is why they stay a separate, clearly-marked list rather
    than being folded into the replay's matrices.

    ``books`` is either the archive as loaded or ``published_book_replay``'s
    ``books``; ``covered_through`` is the last bar of the weight matrix
    (``None`` → nothing is covered, so every book after the first is
    pending).  Quiet publishes — an identical book, the common case while a
    position simply runs — yield no row, exactly as in the log."""
    parsed = sorted((b for b in books or []),
                    key=lambda b: pd.Timestamp(b["as_of"]))
    if len(parsed) < 2:
        return []
    cut = (pd.Timestamp(covered_through)
           if covered_through is not None else None)
    # the book in the matrix's last row is the newest one published strictly
    # before that bar; everything from it onwards is still unrepresented
    rep = [i for i, b in enumerate(parsed)
           if cut is not None and pd.Timestamp(b["as_of"]) < cut]
    start = rep[-1] if rep else 0
    out = [t for t in (_book_ticket(prev, cur, min_delta)
                       for prev, cur in zip(parsed[start:],
                                            parsed[start + 1:]))
           if t is not None]
    for t in out:
        t["pending"] = True
    out.reverse()
    return out


def book_change_status(books: list[dict],
                       min_delta: float = 0.0005) -> dict | None:
    """Why the as-published trade log ends where it does: ``n_books``,
    ``latest`` (newest ``as_of`` archived), ``last_change`` (newest book
    that actually moved a sleeve — the log's newest possible ticket),
    ``quiet`` (how many publishes since carried an identical book) and
    ``quiet_from`` (the first of them).

    A book whose weights repeat the previous one produces no ticket, so a
    run of unchanged publishes leaves the log frozen at an older date while
    the record itself is perfectly current — indistinguishable, without this,
    from a broken feed.  ``None`` when there are no books."""
    parsed = sorted((b for b in books or []),
                    key=lambda b: pd.Timestamp(b["as_of"]))
    if not parsed:
        return None
    last_change = quiet_from = None
    quiet = 0
    for prev, cur in zip(parsed, parsed[1:]):
        if _book_ticket(prev, cur, min_delta) is not None:
            last_change = pd.Timestamp(cur["as_of"])
            quiet, quiet_from = 0, None
        else:
            quiet += 1
            quiet_from = quiet_from or pd.Timestamp(cur["as_of"])
    return dict(n_books=len(parsed), latest=pd.Timestamp(parsed[-1]["as_of"]),
                last_change=last_change, quiet=quiet, quiet_from=quiet_from)


# ── equal-weight buy & hold — the "do nothing" alternative, in replay shape ──
# The strategy sections all run off one triple: a per-day WEIGHT matrix, a
# per-day SATA (idle-cash) weight, and a per-asset RETURN matrix.  Buy & hold
# is the same thing with the signals removed: every instrument in the universe
# carries the same target weight every day and earns its own underlying move,
# whatever the signal said, with nothing ever parked in cash.  Building it in
# the replay's shape lets the whole P&L section — headline metrics, benchmark
# table, allocation area, daily P&L, exact per-sleeve attribution — run off it
# unchanged, and guarantees the curve is byte-for-byte the ``bh_equal``
# benchmark line the same charts already draw.
def bh_returns_matrix(results: list[dict]) -> pd.DataFrame:
    """Per-instrument daily BUY & HOLD returns — the underlying's own bar-to-bar
    move, ignoring the signal.  The buy-&-hold analogue of ``returns_matrix``
    (which carries each sleeve's *strategy* returns), aligned into one frame
    with NaN where an instrument has no history yet."""
    cols = {}
    for res in results:
        px = pd.Series(np.asarray(res["r"]["bh"], float),
                       index=pd.to_datetime(pd.Series(res["r"]["dates"])))
        cols[res["key"]] = px.pct_change()
    return pd.DataFrame(cols).sort_index()


def equal_weight_bh_replay(results: list[dict],
                           index: pd.Index | None = None) -> dict | None:
    """Equal-weight buy & hold of the whole universe, returned in the same
    shape as ``walkforward_gated_replay`` / ``published_book_replay``
    (``ret``/``equity``/``metrics``/``weights``/``sata``) so every figure and
    toggle in the P&L section can run off it with no special-casing.

    Same construction as ``benchmarks()['bh_equal']``: an equal TARGET weight
    in every instrument, renormalised each day over the sleeves that actually
    have data (so a sleeve whose history starts later joins at its first bar
    instead of dragging a cash stub through the early window), held through
    every signal — no entries, no exits, no idle cash, so ``sata`` is flat
    zero.  Passing ``index`` (the strategy returns matrix's index) aligns the
    curve to exactly the same calendar the strategy sources use, which is what
    makes this identical to the ``Equal-weight Buy & Hold`` benchmark curve.

    Also returns ``rets`` — the buy-&-hold returns matrix the weights earn,
    which is what the attribution/daily-P&L helpers must be fed instead of the
    strategy ``returns_matrix`` — and ``n_assets``.  ``None`` when there is
    nothing to replay (<2 bars)."""
    bh = bh_returns_matrix(results)
    if bh.empty:
        return None
    if index is not None:
        bh = bh.reindex(index)
    if len(bh.index) < 2:
        return None
    avail = bh.notna().to_numpy()
    n = bh.shape[1]
    wt = avail * (1.0 / n)
    denom = wt.sum(axis=1, keepdims=True)
    denom[denom == 0] = np.nan
    W = np.nan_to_num(wt / denom)          # sums to 1 on any day with data
    port = (W * np.nan_to_num(bh.to_numpy(float))).sum(axis=1)
    daily = pd.Series(port, index=bh.index)
    eq = _equity(daily)
    return dict(ret=daily, equity=eq, metrics=curve_metrics(eq),
                weights=pd.DataFrame(W, index=bh.index, columns=bh.columns),
                sata=pd.Series(0.0, index=bh.index),
                rets=bh, n_assets=n)


def benchmarks(returns: pd.DataFrame, results: list[dict],
               pos: pd.DataFrame | None = None, sata_daily: float = 0.0) -> dict:
    """Reference curves: equal-weight buy&hold of the underlyings (always
    invested, no SATA), equal-weight of the strategies (idle→SATA), and the best
    single strategy by Sharpe."""
    out = {}
    bh_cols = {}
    for res in results:
        px = pd.Series(res["r"]["bh"], index=pd.to_datetime(pd.Series(res["r"]["dates"])))
        bh_cols[res["key"]] = px.pct_change()
    bh_df = pd.DataFrame(bh_cols).reindex(returns.index)
    n = bh_df.shape[1]
    eqbh = _equity(_combine(bh_df, np.full(n, 1.0 / n)))
    out["bh_equal"] = curve_metrics(eqbh); out["bh_equal"]["equity"] = eqbh
    eqw = _equity(_combine(returns, np.full(returns.shape[1], 1.0 / returns.shape[1]),
                           pos, sata_daily))
    out["strat_equal"] = curve_metrics(eqw); out["strat_equal"]["equity"] = eqw
    best_key = best_m = best_eq = None
    for res in results:
        eq = _equity(res["ret"].reindex(returns.index)); m = curve_metrics(eq)
        if best_m is None or m["sharpe"] > best_m["sharpe"]:
            best_key, best_m, best_eq = res["key"], m, eq
    out["best_single"] = dict(key=best_key, **best_m); out["best_single"]["equity"] = best_eq
    return out


def period_breakdown(returns: pd.DataFrame, weights: np.ndarray,
                     periods: list[tuple], pos: pd.DataFrame | None = None,
                     sata_daily: float = 0.0) -> list[dict]:
    rows = []
    for lbl, s, e in periods:
        sub = returns.loc[pd.Timestamp(s):(pd.Timestamp(e) if e else None)]
        if len(sub) < 5:
            continue
        subpos = pos.loc[sub.index] if pos is not None else None
        rows.append(dict(label=lbl, start=s, end=e,
                         **curve_metrics(_equity(_combine(sub, weights, subpos, sata_daily)))))
    return rows


# combined-history windows (per-instrument return streams begin at the common
# OOS start — out-of-sample for the H/L models, but strategy parameters and
# blend weights are tuned/fit on these same windows, and the BTC CT sleeves'
# model training extends into them; treat levels as in-sample-selected).
COMBINED_PERIODS = [
    ("🌐 Full OOS (2021 → now)", "2021-01-01", None),
    ("🐻 Bear (2021–2022)", "2021-01-01", "2022-12-31"),
    ("🐂 Bull (2023 → now)", "2023-01-01", None),
    ("🔬 Recent (2025 → now)", "2025-01-01", None),
]


# ════════════════════════════════════════════════════════════════════════
# LIVE SPOT PRICES — the traded-instrument last price, for the action plan.
# The strategy/signals run on completed daily bars (some from cached CSVs, so
# their last close can be days stale — notably BTC/MSTR/MSTU); the *displayed*
# price must be the current spot, so we pull each instrument's live quote
# separately from the Yahoo chart meta (regularMarketPrice + previous close).
# ════════════════════════════════════════════════════════════════════════
# Derived from ASSET_META so every traded instrument — including newly-added
# leveraged siblings (SOXL, ERX, NUGT, …) — always gets a live quote. A missing
# entry silently leaves that asset's Price / Chg % on the stale daily-bar close,
# so we build the map from the single source of truth rather than by hand. The
# Yahoo symbol equals the key for every asset except BTC (spot ticker BTC-USD).
# Display/spot quote symbol per key.  The two crypto sleeves quote their SIGNAL
# asset (spot BTC / spot ETH); the instrument actually traded on IBKR is a
# separate mapping (BTC→IBIT, ETH→ETHA — see scripts/ibkr_symbols.py), and the
# executor sizes on that vehicle's own live price.
_SPOT_OVERRIDE = {"BTC": "BTC-USD", "ETH": "ETH-USD"}
SPOT_SYMBOLS = {k: _SPOT_OVERRIDE.get(k, k) for k in ASSET_META}


def _quote(symbol: str) -> tuple:
    """(spot price, previous-session close) for a symbol.

    Spot = the chart meta's ``regularMarketPrice`` (true live quote); the
    previous close is the last *completed* daily bar strictly before today
    (Yahoo's ``previousClose`` is often null and ``chartPreviousClose`` is the
    close before the whole range, so neither gives a correct 1-day change)."""
    params = {"interval": "1d", "range": "7d"}
    for host in tc._YH_HOSTS:
        try:
            r = requests.get(f"{host}/v8/finance/chart/{symbol}",
                             params=params, headers=tc._UA, timeout=15)
            if r.status_code != 200:
                continue
            res = r.json()["chart"]["result"][0]
            meta = res.get("meta", {})
            px = meta.get("regularMarketPrice")
            ts = res.get("timestamp") or []
            closes = (res.get("indicators", {}).get("quote", [{}])[0] or {}).get("close") or []
            s = pd.Series(closes, index=pd.to_datetime(ts, unit="s").normalize()).dropna()
            if px is None and len(s):
                px = float(s.iloc[-1])
            # prev = the last close strictly BEFORE the session the quote
            # belongs to, so the day-change always compares the live/most-recent
            # session against the one before it (during market hours, after
            # hours and on weekends alike).  Anchoring on the quote's own
            # timestamp rather than on ``s.iloc[-2]`` keeps that true when the
            # daily series lags the quote — early in a session the chart can
            # still be missing today's bar while ``regularMarketPrice`` is
            # already live, and the positional pick then measured against the
            # session before last, overstating the change by a whole day.
            prev = None
            q_ts = meta.get("regularMarketTime")
            if q_ts:
                try:
                    q_day = (pd.Timestamp(int(q_ts), unit="s", tz="UTC")
                             .tz_convert(meta.get("exchangeTimezoneName") or "UTC")
                             .normalize().tz_localize(None))
                    earlier = s[s.index < q_day]
                    if len(earlier):
                        prev = float(earlier.iloc[-1])
                except Exception:
                    prev = None
            if prev is None and len(s) >= 2:
                prev = float(s.iloc[-2])
            if px:
                return float(px), prev
        except Exception:
            continue
    return None, None


def fetch_spot(symbols: dict | None = None) -> dict:
    """Live spot price + day-change % for each instrument, fetched concurrently.
    Returns {key: {"price": float|None, "dchg": float|None}}."""
    from concurrent.futures import ThreadPoolExecutor
    symbols = symbols or SPOT_SYMBOLS

    def _one(item):
        k, sym = item
        px, prev = _quote(sym)
        dchg = ((px / prev - 1) * 100) if (px and prev) else None
        return k, dict(price=px, dchg=dchg)

    items = list(symbols.items())
    with ThreadPoolExecutor(max_workers=min(13, len(items))) as ex:
        return {k: v for k, v in ex.map(_one, items)}


def fetch_sata() -> dict:
    """Live SATA quote: current price, day-change %, and unrealised P&L measured
    against its $100 par cost basis (the price idle cash is parked at)."""
    px, prev = _quote(SATA["ticker"])
    dchg = ((px / prev - 1) * 100) if (px and prev) else None
    upnl = ((px / SATA["par"] - 1) * 100) if px else None
    return dict(price=px, prev=prev, dchg=dchg, upnl=upnl)


def live_change_pct(bar_close, live_price) -> float | None:
    """Percent move from a completed bar's close to the live spot price.

    This is the ONLY change the action plan's ``Chg %`` column may show: it is
    printed between ``Price (Close of Last Bar)`` and ``Live Price``, so it has
    to be the move between exactly those two numbers — anything else (notably
    the spot feed's session day-change, which measures live vs the previous
    *calendar* session's close) contradicts the prices either side of it and
    the green/red tint on the live cell.

    ``None`` when either side is missing or unusable — a row with no live quote
    reports "—", never a fabricated 0.00%."""
    try:
        b, p = float(bar_close), float(live_price)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(b) and np.isfinite(p)) or b <= 0:
        return None
    return (p / b - 1) * 100


def apply_spot(results: list[dict], spot: dict) -> None:
    """Overlay live spot prices onto result dicts in place — updates the
    displayed last price, day-change and any open-position unrealised P&L /
    distance-to-stop, without touching the strategy signals or back-test."""
    for r in results:
        s = spot.get(r["key"])
        if not s or not s.get("price"):
            continue
        r["last_close"] = s["price"]
        if s.get("dchg") is not None:
            r["dchg"] = s["dchg"]
        p = r.get("pos") or {}
        if p.get("in_pos") and p.get("entry_px"):
            e = float(p["entry_px"])
            p["upnl"] = (s["price"] / e - 1) * 100
            if p.get("stop_px"):
                p["dist_stop"] = (s["price"] / p["stop_px"] - 1) * 100


# ── real cost basis: the official close on the entry bar ─────────────────────
def _bar_close(symbol: str, date) -> float | None:
    """Official daily close of ``symbol`` on ``date`` (or the last trading day
    at/before it), from the live chart feed — used as the real cost basis so
    unrealised P&L is measured against a genuine market close on the entry bar,
    not the strategy's (possibly synthetic/cached) fill price."""
    d = pd.Timestamp(date).normalize()
    params = {"interval": "1d",
              "period1": int((d - pd.Timedelta(days=8)).timestamp()),
              "period2": int((d + pd.Timedelta(days=3)).timestamp())}
    for host in tc._YH_HOSTS:
        try:
            r = requests.get(f"{host}/v8/finance/chart/{symbol}",
                             params=params, headers=tc._UA, timeout=15)
            if r.status_code != 200:
                continue
            res = r.json()["chart"]["result"][0]
            ts = res.get("timestamp") or []
            closes = (res.get("indicators", {}).get("quote", [{}])[0] or {}).get("close") or []
            s = pd.Series(closes, index=pd.to_datetime(ts, unit="s").normalize()).dropna()
            s = s[s.index <= d]
            if len(s):
                return float(s.iloc[-1])
        except Exception:
            continue
    return None


def fetch_entry_closes(results: list[dict], symbols: dict | None = None) -> dict:
    """Real official close on each OPEN position's entry date, fetched
    concurrently. Returns {key: close_float} (only successful lookups)."""
    from concurrent.futures import ThreadPoolExecutor
    symbols = symbols or SPOT_SYMBOLS
    need = []
    for r in results:
        p = r.get("pos") or {}
        sym = symbols.get(r["key"])
        if p.get("in_pos") and p.get("entry_date") and sym:
            need.append((r["key"], sym, p["entry_date"]))
    if not need:
        return {}

    def _one(item):
        k, sym, dt = item
        return k, _bar_close(sym, dt)

    with ThreadPoolExecutor(max_workers=min(13, len(need))) as ex:
        return {k: v for k, v in ex.map(_one, need) if v}


def apply_entry_basis(results: list[dict], entry_closes: dict) -> None:
    """Overlay the real entry-date close as the displayed cost basis, recomputing
    the stop level, unrealised P&L and distance-to-stop against the current price.
    Keeps the strategy's own fill under ``pos['entry_px_model']``. Idempotent —
    safe to call on cached result dicts across reruns."""
    for r in results:
        c = entry_closes.get(r["key"])
        p = r.get("pos") or {}
        if not c or not p.get("in_pos"):
            continue
        p.setdefault("entry_px_model", p.get("entry_px"))   # preserve strategy fill
        p["entry_px"] = float(c)
        stop_frac = r.get("stop") or 0.0
        last = r.get("last_close")
        if stop_frac:
            p["stop_px"] = float(c) * (1 - stop_frac)
        if last:
            p["upnl"] = (last / float(c) - 1) * 100
            if p.get("stop_px"):
                p["dist_stop"] = (last / p["stop_px"] - 1) * 100


def _live_is_last_close(r: dict, plive, tol: float = 1e-4) -> bool:
    """True when the live quote IS the parent's last committed close, just
    re-served (weekends, market holidays, and after-hours — Yahoo's
    ``regularMarketPrice`` simply repeats the last session close then).

    Why it matters: the live entry/exit checks append the live price to the
    committed close history as a PROVISIONAL NEW bar.  When that price equals
    the last close, the appended bar adds no information — it merely
    duplicates the final close and shifts every rolling window by one bar,
    which can flip a marginal read (e.g. a dual-MA golden cross whose fast/
    slow SMAs sit close together) and manufacture phantom "exits next bar" /
    "likely enters next bar" flags that contradict the committed decision —
    and the instrument's own ticker app, which shows committed bars only
    (observed: REMX flagged "exits next bar" in the Overall action table on
    a Saturday while the REMX app correctly showed LONG — HOLDING).  The
    committed engine state already answers "what if the next close equals
    the last one", so the live overlay should stay silent until a genuinely
    NEW price exists.  ``tol`` is relative (default 1 bp — quotes that close
    to the last close carry no actionable new information either).  Returns
    False when the history is unavailable (the caller's fallback reads are
    consistent with the committed state by construction)."""
    ch = r.get("close_hist")
    if ch is None or plive is None:
        return False
    arr = np.asarray(ch, float)
    arr = arr[np.isfinite(arr)]
    if not len(arr) or not np.isfinite(plive):
        return False
    last = float(arr[-1])
    return abs(float(plive) - last) <= abs(last) * tol


def live_exit_keys(results: list[dict], spot: dict,
                   include_entries: bool = False) -> set:
    """Keys of trend instruments the *live* price says will drop out on the next
    bar.  The mode's REAL long condition is re-evaluated with the live price as
    the newest close (via ``trend_long_now_live``): for ``ma``/``ma_vol`` that's
    close-vs-SMA, and for ``dual_ma`` it's the fast/slow SMA cross — NOT a naive
    price-vs-line read (so a golden-cross name whose price merely dips below its
    slow SMA is not mis-flagged).  These filters decide at the close and act on
    the next bar.  By default this covers positions currently **held** (long
    today, but exit next bar).  With ``include_entries=True`` it also covers fresh
    **entries** signalled off the last close whose live price has since broken the
    trend — they'd open into a broken trend and reverse right back out next bar.
    MACD (oscillator) and divergence exits are signal-triggered, not simple
    price-crossings, so they're not included."""
    out = set()
    for r in results:
        mode = r.get("mode")
        if mode not in ("ma", "dual_ma", "ma_vol"):
            continue
        p = r.get("pos") or {}
        in_pos = bool(p.get("in_pos"))
        tone = (r.get("decision") or {}).get("tone")
        is_hold = in_pos
        is_entry = (not in_pos) and tone == "buy"
        if not (is_hold or (include_entries and is_entry)):
            continue
        plive = (spot.get(r.get("parent")) or {}).get("price")
        if plive is None:
            continue
        if _live_is_last_close(r, plive):
            continue                 # stale re-served quote — no new information
        # Re-run the mode's ACTUAL long condition with the live price as the newest
        # close.  Critically, dual_ma exits on a fast/slow SMA cross (a "death
        # cross"), NOT on price dropping below the slow SMA — so a naive
        # `price < ma_val` proxy would falsely flag a golden-cross name (e.g. REMX)
        # whose live price dips below its 200-day line while 50-SMA > 200-SMA.
        cfg = r.get("cfg"); close_hist = r.get("close_hist")
        _live_fn = getattr(bt, "trend_long_now_live", None)  # absent on a stale reload
        if cfg is not None and close_hist is not None and _live_fn is not None:
            long_live = _live_fn(cfg, close_hist, plive)
            if long_live is False:          # trend genuinely flips flat on live px
                out.add(r["key"])
        elif mode == "ma":
            # fallback for results lacking cfg/close_hist (e.g. stale cache): the
            # naive price-vs-SMA read IS the real exit rule for plain `ma` only.
            ma = r.get("ma_val")
            if ma is not None and plive < ma:
                out.add(r["key"])
    return out


def divergence_entry_now_live(cfg, completed, pending: dict | None,
                              live_px) -> bool | None:
    """Would the divergence engine's flat-side ENTRY fire if today's bar closed
    at ``live_px``?  The divergence mirror of ``trend_long_now_live``: appends a
    provisional completed bar — the model's pending-bar bands (``pending``, from
    ``build_predictions``) scored against the live price standing in for the
    bar's high AND low — re-runs the REAL signature engine
    (``compute_trend_signatures``) on the extended frame, and applies the same
    exit-overrides-entry precedence as ``_net_decision``'s flat branch.  Using
    the live price as the provisional high understates ``err_hi`` (the true
    intraday high is ≥ live), so a True here is a conservative read — the entry
    fires even on the understated bar.  Returns None when inputs are unusable
    (missing bands / too little history), never guessing."""
    if (completed is None or len(completed) < 3 or not pending
            or live_px is None or not np.isfinite(live_px)):
        return None
    try:
        row = pd.DataFrame([dict(
            close_asof=float(pending["close_asof"]),
            pred_high=float(pending["pred_high"]),
            pred_low=float(pending["pred_low"]),
            actual_high=float(live_px), actual_low=float(live_px),
            target_date=pd.Timestamp(completed["target_date"].iloc[-1])
            + pd.Timedelta(days=1))])
        frame = pd.concat([completed.reset_index(drop=True), row],
                          ignore_index=True)
        sigs = tc.compute_trend_signatures(cfg, frame)
    except Exception:
        return None
    if not sigs:
        return None
    # same net-exit precedence as _net_decision's flat branch: an active exit
    # signal blocks the entry, so this can never say "enters" while D2/D3 fire.
    exit_sig = (bool(sigs["d2_triggered"]) or bool(sigs["d3_triggered"])
                or (bool(sigs["d1_triggered"]) and cfg.use_d1_exit))
    return bool(sigs["entry_triggered"]) and not exit_sig


def live_entry_keys(results: list[dict], spot: dict) -> set:
    """Mirror of :func:`live_exit_keys` for FRESH ENTRIES: keys of instruments
    that are flat off the last close (no committed buy signal) but whose *live*
    price satisfies the mode's REAL entry condition — if it holds into the
    close, the signal fires today and the position **opens next bar**.

    Trend modes (``ma``/``dual_ma``/``ma_vol``) re-evaluate the long condition
    with the live price as the newest close (via ``trend_long_now_live``):
    close-vs-SMA (plus the vol gate) for ``ma``/``ma_vol``, the fast/slow SMA
    cross for ``dual_ma`` — NOT a naive price-vs-line read (so a name whose
    price merely pops above its slow SMA is not mis-flagged before a golden
    cross).  DIVERGENCE apps (e.g. ARTY) re-run the real signature engine with
    the live price scored against the model's pending-bar bands — see
    :func:`divergence_entry_now_live` — so a live rally strong enough to fire
    U1 + the regime gate at today's close is flagged too.  Instruments already
    signalling buy off the last close (committed entries, action OPEN) are
    excluded — they are flagged separately as certain, not "likely", entries.
    MACD (oscillator) entries are signal-triggered with no live equivalent, so
    they're not included."""
    out = set()
    for r in results:
        mode = r.get("mode")
        if mode not in ("ma", "dual_ma", "ma_vol", "divergence"):
            continue
        p = r.get("pos") or {}
        if p.get("in_pos"):
            continue
        if (r.get("decision") or {}).get("tone") == "buy":
            continue                    # committed entry — not a live "likely"
        plive = (spot.get(r.get("parent")) or {}).get("price")
        if plive is None:
            continue
        if _live_is_last_close(r, plive):
            continue                 # stale re-served quote — no new information
        cfg = r.get("cfg"); close_hist = r.get("close_hist")
        if mode == "divergence":
            if cfg is not None and divergence_entry_now_live(
                    cfg, r.get("sig_completed"), r.get("pending_pred"),
                    plive) is True:
                out.add(r["key"])
            continue
        _live_fn = getattr(bt, "trend_long_now_live", None)  # absent on a stale reload
        if cfg is not None and close_hist is not None and _live_fn is not None:
            if _live_fn(cfg, close_hist, plive) is True:  # trend flips long on live px
                out.add(r["key"])
        elif mode == "ma":
            # fallback for results lacking cfg/close_hist: the naive
            # price-vs-SMA read IS the real entry rule for plain `ma` only.
            ma = r.get("ma_val")
            if ma is not None and plive > ma:
                out.add(r["key"])
    return out
