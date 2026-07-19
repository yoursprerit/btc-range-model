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
    BTC     BTC        BTC (1×) · MSTR (BTC-proxy) · MSTU (2× MSTR)
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
from ticker_config import TickerConfig, get_config, _STD_PERIODS   # noqa: E402

# Capability flag: this build's live_exit_keys re-runs each mode's real trend
# condition (dual_ma fast/slow cross, etc.) instead of a naive price-vs-line
# proxy, and _asset_result attaches cfg/close_hist for it.  The app's _stale_core
# guard checks this so a hot-reload against an older cached module is refreshed.
LIVE_EXIT_MODE_AWARE = True


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
    extra_syms={"mstr": "MSTR", "mstu": "MSTU"},
    sentiment=[("spx_close", "mom", +1.0), ("ndx_close", "mom", +1.0),
               ("vix_close", "lvl", -1.0), ("px_close", "mom", +1.0)],
    sentiment_label="Crypto/risk macro sentiment",
    traded_assets=[("BTC", "px_close"), ("MSTR", "mstr_close"), ("MSTU", "mstu_close")],
    asset_labels={"px_close": "BTC · Bitcoin", "mstr_close": "MSTR · MicroStrategy",
                  "mstu_close": "MSTU · 2× MSTR"},
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

# GLDM — Divergence Pure-Regime, the Gold app's ACTUAL strategy.
# gldm_core trades one strategy: a BTC-style U1/D2/D3 bullish-divergence system
# (U1=0.08, D2=−0.10, D1=0.10, −3% stop) confirmed inside a 50-day trend-regime
# gate.  It reproduces daily far better than it does for BTC (gold trends
# smoothly), so we run the divergence system here — not the MA50 comparison
# variant.  Trades GLDM (1×) plus GDX (miners) and UGL (2× gold) off the signal.
GLDM_CFG = TickerConfig(
    key="GLDM", name="SPDR Gold MiniShares", emoji="🥇",
    blurb=("Spot gold via GLDM drives the signal; the strategy trades GLDM and "
           "its higher-beta siblings GDX (miners), UGL (2× gold) and NUGT "
           "(2× gold miners) off it."),
    accent="#b8860b", accent_dark="#8b6508", accent_bg="#fffbeb", accent_bg2="#fef3c7",
    primary_symbol="GLDM",
    macro_syms={"gc": "GC=F", "slv": "SLV", "dxy": "DX-Y.NYB",
                "tnx": "^TNX", "vix": "^VIX", "spx": "^GSPC"},
    extra_syms={"gdx": "GDX", "ugl": "UGL", "nugt": "NUGT"},
    sentiment=[("gc_close", "mom", +1.0), ("dxy_close", "mom", -1.0),
               ("vix_close", "lvl", -1.0), ("px_close", "mom", +1.0)],
    sentiment_label="Gold macro sentiment",
    traded_assets=[("GLDM", "px_close"), ("GDX", "gdx_close"), ("UGL", "ugl_close"),
                   ("NUGT", "nugt_close")],
    asset_labels={"px_close": "GLDM · Gold", "gdx_close": "GDX · Gold Miners",
                  "ugl_close": "UGL · 2× Gold", "nugt_close": "NUGT · 2× Gold Miners"},
    strategy_mode="divergence", strategy_name="Gold Divergence Pure-Regime",
    ma_window=50, fixed_stop=0.03,
    u1_errhi_min=0.08, d2_errhi_max=-0.10, d1_errlo_min=0.10, v_errlo_min=0.50,
    use_d1_exit=False, hl_band_pct=0.008,
    fetch_start="2018-06-26", oos_start="2021-01-01", periods=_STD_PERIODS,
    day_up_thresh=0.006, day_down_thresh=-0.006,
    results_note=("Divergence Pure-Regime — the Gold app's actual strategy: a U1 "
                  "bullish-divergence entry confirmed inside a 50-day regime "
                  "gate, D2/D3 exits, per-asset stops (GLDM/GDX −3%, but the "
                  "leveraged siblings are looser — UGL signal-only, NUGT −5%). The "
                  "same gold signal steers GDX and the 2× UGL. OOS 2021→now: GLDM "
                  "+73% / −11% / Sharpe 1.29, GDX +156% / −19% / 1.12, UGL (now "
                  "stop-less) +247% / −18% / 1.37. The same signal drives the 2× "
                  "gold-miners NUGT sleeve — retuned to a −5% stop (vs −3%): "
                  "+1183% / −28% / Sharpe 1.48 OOS, a leveraged win-win in the "
                  "blend (see LEV_SIBLINGS_STOP_EVAL.md)."),
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
    return get_config(key)


# Parent (signal-app) display order.
PARENT_KEYS = ["BTC", "GLDM", "SOXX", "GRID", "XLE", "REMX", "WGMI",
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
    "GLDM": 1.30, "GDX": 1.40, "UGL": 1.40, "NUGT": 1.40,   # structural gold bull (NUGT = 2× miners)
    "GRID": 1.40,                                # electrification / grid capex
    "WGMI": 1.30,                                # miners' AI/HPC pivot
    "REMX": 1.10,                                # rare-earth supply squeeze
    "XLE": 1.00, "ERX": 1.00,                    # energy — gas ok, oil soft (ERX = 2× energy)
    "OIH": 0.50, "PBW": 0.40,                    # no catalyst / policy headwinds
}
FUNDAMENTAL_VIEW_NOTE = (
    "Mid-2026 sector outlook: overweight AI/semis (SOXX, ARTY), the crypto "
    "institutional era (BTC/MSTR/MSTU, WGMI), the structural gold bull "
    "(GLDM/GDX/UGL) and electrification (GRID); underweight clean energy (PBW) "
    "and oil services (OIH).")


# ── risk profiles — how hard to lean on the high-beta / leveraged proxies ──
# Beta/leveraged sleeves carry huge returns but poor stand-alone Sharpe (deep
# drawdowns), so loading them boosts raw return at the cost of risk-adjusted
# return.  A profile bundles the per-kind caps, the optimiser objective and the
# drawdown budget so the user can dial that trade-off:
#   Balanced   — hold Sharpe near its max (default; ~unchanged behaviour)
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
DEFAULT_PROFILE = "Aggressive"


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
    """Daily OHLCV+macro for one config, live from Yahoo with a CSV fallback.
    Traded-sibling price columns are forward-filled so equity assets don't leave
    NaN gaps on days their market is shut (e.g. weekends when BTC still trades)."""
    df = pd.DataFrame()
    try:
        df = tc.fetch_daily(cfg)
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
            # position is still open the day its close first drops below the SMA
            # (it exits next bar). Mirror the source app's net_signal_ma: while
            # in position the state is always HOLD — flag the pending exit rather
            # than closing a bar early, so the Overall app agrees with the app.
            if above:
                return dict(state="HOLD", label="LONG — HOLDING", ico="🟢", tone="hold")
            return dict(state="HOLD",
                        label="LONG — HOLDING (below trend → exits next bar)",
                        ico="🟡", tone="hold", exits_next_bar=True)
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
            return dict(state="EXIT", label=f"EXIT — {why}", ico="🔴", tone="exit")
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


def _asset_result(cfg, label, col, r, daily, dec, alert, bull, sent, ma_val, dchg, mom):
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
        # rather than a naive price-vs-line proxy.
        cfg=cfg, close_hist=daily["px_close"].to_numpy(float),
    )


def run_asset(cfg: TickerConfig) -> list[dict]:
    """Run every instrument this app trades. Returns one result dict per traded
    asset (primary + siblings), all sharing the parent's signal/decision."""
    daily = _load_daily(cfg)
    if daily is None or daily.empty or "px_close" not in daily.columns:
        return []
    preds = bt.build_predictions(cfg, daily)
    sig = bt.precompute_signals(cfg, preds)

    # parent-level signal snapshot (shared by all traded instruments)
    last_close = float(daily["px_close"].iloc[-1])
    prev_close = float(daily["px_close"].iloc[-2]) if len(daily) > 1 else last_close
    dchg = (last_close / prev_close - 1) * 100 if prev_close else 0.0
    # trend line for display / live-exit, and the engine's actual long signal
    ma_val = bt.trend_line_value(cfg, daily) if cfg.is_trend else None
    long_now = bt.trend_long_now(cfg, daily) if cfg.is_trend else None
    # common momentum read (distance above a 50-day SMA) for cross-asset priority
    ref_ma = float(daily["px_close"].tail(50).mean())
    mom = (last_close / ref_ma - 1) if ref_ma else 0.0
    completed = preds[preds["actual_high"].notna() & preds["actual_low"].notna()]
    sigs = tc.compute_trend_signatures(cfg, completed.tail(45)) if len(completed) >= 3 else None
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
        out.append(_asset_result(cfg, label, col, r, daily, dec, alert, bull,
                                 sent, ma_val, dchg, mom))
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
                return cfg.key, gldm_engine.run_gldm(), None
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


def position_matrix(results: list[dict], index: pd.Index) -> pd.DataFrame:
    """0/1 in-market flag per instrument, aligned to ``index`` (0 where flat or
    no data)."""
    cols = {res["key"]: res["pos_series"] for res in results}
    return pd.DataFrame(cols).reindex(index).fillna(0.0)


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


def slice_metrics(equity: pd.Series, start) -> dict | None:
    """P&L / performance / risk metrics for an equity curve **re-based at
    ``start``** — i.e. what an investor who put capital into the strategy on
    that date has experienced since.  Re-basing (dividing the slice by its first
    value) matters: drawdown, Sharpe and total return are all measured from the
    entry point, not from the back-test's inception.  Returns ``None`` when the
    curve has fewer than 2 bars on/after ``start`` (nothing to measure); the
    first bar on/after ``start`` becomes the actual anchor (weekends/holidays
    roll forward)."""
    sub = equity.loc[pd.Timestamp(start):]
    if len(sub) < 2:
        return None
    sub = sub / sub.iloc[0]
    # daily win-rate and best/worst day round out the risk read
    rets = sub.pct_change().dropna()
    return dict(start=sub.index[0], end=sub.index[-1], days=len(sub),
                win_days=float((rets > 0).mean()),
                best_day=float(rets.max()), worst_day=float(rets.min()),
                **curve_metrics(sub))


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
    """Normalise to sum 1 then water-fill against a per-element cap vector."""
    w = np.asarray(w, float).copy()
    if w.sum() <= 0:
        return w
    w = w / w.sum()
    for _ in range(30):
        over = w > cap_vec + 1e-9
        if not over.any():
            break
        spill = (w[over] - cap_vec[over]).sum()
        w[over] = cap_vec[over]
        room = w < cap_vec - 1e-9
        base = w[room].sum()
        if not room.any() or base <= 0:
            break
        w[room] = np.minimum(cap_vec[room], w[room] + spill * (w[room] / base))
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
                            force_exit: set | None = None) -> dict:
    """Today's actionable allocation.

    Capital is deployed only to instruments the strategy is long (or opening a
    fresh entry).  Among those, the size of each slice is the historically
    optimal weight **tilted by a live entry-priority score** (momentum,
    sentiment, back-tested win-rate, risk-adjusted edge, regime), then
    water-filled to the per-instrument caps.  The deployed book sums to 100% when
    the caps allow; whatever cannot be deployed is parked in **SATA** (the idle
    -cash preferred, ~13% yield).  With no open positions the whole book is SATA.
    """
    caps = caps or CAP_BY_KEY
    force_exit = force_exit or set()

    def _b(k):
        return max(base_weights.get(k, 0.0), 1e-6)

    in_pos = [res for res in results if res["pos"]["in_pos"]]
    # closing = committed exits, plus any caller-forced exits (e.g. positions the
    # live price says will drop out on the next bar).
    closing = {res["key"] for res in in_pos
               if res["decision"]["tone"] == "exit" or res["key"] in force_exit}
    keep = [res for res in in_pos if res["key"] not in closing]
    # a fresh entry whose live price has already broken the trend (in force_exit)
    # would reverse straight back out next bar — don't fund it in the live book.
    opens = [res for res in results
             if (not res["pos"]["in_pos"]) and res["decision"]["tone"] == "buy"
             and res["key"] not in force_exit]

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
        if res["pos"]["in_pos"] and stt == "EXIT":
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
        actions.append(dict(key=k, name=res["name"], emoji=res["emoji"],
                            kemoji=res["kemoji"], kind=res["kind"], parent=res["parent"],
                            action=act, tone=dec["tone"], decision=dec["label"],
                            target=target.get(k, 0.0), in_pos=res["pos"]["in_pos"],
                            upnl=res["pos"]["upnl"], alert=res["alert"],
                            last_close=res["last_close"],
                            exits_next_bar=bool(dec.get("exits_next_bar")),
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
# OOS start, so every window below is genuinely out-of-sample).
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
SPOT_SYMBOLS = {k: ("BTC-USD" if k == "BTC" else k) for k in ASSET_META}


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
            # prev = the session-before-last close, so day-change compares the
            # live/most-recent session against the one before it (works during
            # market hours, after hours, and on weekends alike).
            prev = float(s.iloc[-2]) if len(s) >= 2 else None
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
    return dict(price=px, dchg=dchg, upnl=upnl)


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
