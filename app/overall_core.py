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
    VEGN    VEGN       VEGN
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


# ════════════════════════════════════════════════════════════════════════
# CONFIGS — one per signal app; each may trade its 1× primary + siblings.
# ════════════════════════════════════════════════════════════════════════
# BTC — daily MA30 trend filter (model-free, robust regime), −3% stop.
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
                  "MA30 regime gate, −3% stop), so the live entry/exit signal "
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
           "its higher-beta siblings GDX (miners) and UGL (2× gold) off it."),
    accent="#b8860b", accent_dark="#8b6508", accent_bg="#fffbeb", accent_bg2="#fef3c7",
    primary_symbol="GLDM",
    macro_syms={"gc": "GC=F", "slv": "SLV", "dxy": "DX-Y.NYB",
                "tnx": "^TNX", "vix": "^VIX", "spx": "^GSPC"},
    extra_syms={"gdx": "GDX", "ugl": "UGL"},
    sentiment=[("gc_close", "mom", +1.0), ("dxy_close", "mom", -1.0),
               ("vix_close", "lvl", -1.0), ("px_close", "mom", +1.0)],
    sentiment_label="Gold macro sentiment",
    traded_assets=[("GLDM", "px_close"), ("GDX", "gdx_close"), ("UGL", "ugl_close")],
    asset_labels={"px_close": "GLDM · Gold", "gdx_close": "GDX · Gold Miners",
                  "ugl_close": "UGL · 2× Gold"},
    strategy_mode="divergence", strategy_name="Gold Divergence Pure-Regime",
    ma_window=50, fixed_stop=0.03,
    u1_errhi_min=0.08, d2_errhi_max=-0.10, d1_errlo_min=0.10, v_errlo_min=0.50,
    use_d1_exit=False, hl_band_pct=0.008,
    fetch_start="2018-06-26", oos_start="2021-01-01", periods=_STD_PERIODS,
    day_up_thresh=0.006, day_down_thresh=-0.006,
    results_note=("Divergence Pure-Regime — the Gold app's actual strategy: a U1 "
                  "bullish-divergence entry confirmed inside a 50-day regime "
                  "gate, D2/D3 exits, −3% stop. The same gold signal steers GDX "
                  "and the 2× UGL. OOS 2021→now: GLDM +73% / −11% / Sharpe 1.29, "
                  "GDX +156% / −19% / 1.12, UGL +184% / −20% / 1.28."),
)

def overall_config(key: str) -> TickerConfig:
    """Return the unified parent config for one signal app.

    Every app is aligned to the strategy its own app trades: BTC via the CT
    engine (see run_universe), GLDM via its Divergence Pure-Regime config, and
    the six ETF apps via their exact ``ticker_config`` entries — including
    **REMX on its MA150 trend filter** (its app's strategy; an earlier
    divergence override was reverted so the Overall app matches each source app).
    """
    if key == "BTC":
        return BTC_CFG
    if key == "GLDM":
        return GLDM_CFG
    return get_config(key)


# Parent (signal-app) display order.
PARENT_KEYS = ["BTC", "GLDM", "SOXX", "VEGN", "GRID", "XLE", "REMX", "WGMI"]


def all_configs() -> list[TickerConfig]:
    return [overall_config(k) for k in PARENT_KEYS]


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
    "SOXX": dict(name="Semiconductors", kind="core"),
    "VEGN": dict(name="ESG Large-Cap",  kind="core"),
    "GRID": dict(name="Grid Infra",     kind="core"),
    "XLE":  dict(name="Energy",         kind="core"),
    "OIH":  dict(name="Oil Services",   kind="beta"),
    "REMX": dict(name="Rare-Earth Metals", kind="core"),
    "WGMI": dict(name="Bitcoin Miners", kind="beta"),
}
KIND_EMOJI = {"core": "", "beta": "⚡", "lev": "🔺"}
CAP_BY_KIND = {"core": 0.30, "beta": 0.18, "lev": 0.10}
CAP_BY_KEY = {k: CAP_BY_KIND[m["kind"]] for k, m in ASSET_META.items()}


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
                  last_close: float, ma_val: float | None) -> dict:
    """Resolve a single actionable state for the PARENT signal, reconciling the
    instantaneous read with the strategy's actual executed position.
    tone ∈ {buy, hold, exit, watch, flat}."""
    if cfg.strategy_mode == "ma":
        above = (ma_val is not None) and (last_close > ma_val)
        if in_pos and above:
            return dict(state="HOLD", label="LONG — HOLDING", ico="🟢", tone="hold")
        if in_pos and not above:
            return dict(state="EXIT", label="EXIT — TREND BROKEN", ico="🔴", tone="exit")
        if (not in_pos) and above:
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

    pos = dict(in_pos=bool(r.get("in_pos_now")), entry_px=r.get("entry_px"),
               entry_date=r.get("entry_date"), upnl=None, stop_px=None,
               days=None, dist_stop=None)
    if pos["in_pos"] and pos["entry_px"]:
        e_px = float(pos["entry_px"]); e_dt = pd.Timestamp(pos["entry_date"])
        pos["upnl"] = (last_px / e_px - 1) * 100
        pos["stop_px"] = e_px * (1 - cfg.fixed_stop)
        pos["days"] = int((as_of - e_dt).days)
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
        mode=cfg.strategy_mode, ma_window=cfg.ma_window, stop=cfg.fixed_stop,
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
    ma_val = float(daily["px_close"].tail(cfg.ma_window).mean()) if cfg.strategy_mode == "ma" else None
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
        dec = _net_decision(cfg, sigs, bool(r.get("in_pos_now")), last_close, ma_val)
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


def optimize_weights(returns: pd.DataFrame, caps: dict | None = None,
                     n_samples: int = 20000, seed: int = 7, mdd_floor: float = -0.35,
                     pos: pd.DataFrame | None = None, sata_daily: float = 0.0,
                     objective: str = "balanced") -> dict:
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

    # Evaluate every candidate once, then choose in two stages so the result
    # both maximises risk-adjusted return AND grabs the most raw return among
    # near-optimal blends — matching the "maximise returns, minimise losses"
    # mandate rather than a purely defensive max-Sharpe point.
    evals = [(w, _cm(w)) for w in cand]
    feasible = [(w, mm) for w, mm in evals if mm["mdd"] >= mdd_floor]
    pool = feasible if feasible else evals
    if objective == "max_return":
        # highest raw return inside the drawdown budget — leans on β / 2×
        w_opt, m_opt = max(pool, key=lambda x: x[1]["total_ret"])
    elif objective == "max_sharpe":
        w_opt, m_opt = max(pool, key=lambda x: x[1]["sharpe"])
    else:  # "balanced" — highest return among near-max-Sharpe blends
        best_sharpe = max(mm["sharpe"] for _, mm in pool)
        near = [(w, mm) for w, mm in pool if mm["sharpe"] >= 0.92 * best_sharpe]
        w_opt, m_opt = max(near, key=lambda x: x[1]["total_ret"])

    def _pack(w):
        return {c: float(x) for c, x in zip(cols, np.asarray(w, float))}
    return dict(
        cols=cols,
        optimal=dict(weights=_pack(w_opt), **m_opt),
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
                            caps: dict | None = None) -> dict:
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

    def _b(k):
        return max(base_weights.get(k, 0.0), 1e-6)

    in_pos = [res for res in results if res["pos"]["in_pos"]]
    closing = {res["key"] for res in in_pos if res["decision"]["tone"] == "exit"}
    keep = [res for res in in_pos if res["key"] not in closing]
    opens = [res for res in results
             if (not res["pos"]["in_pos"]) and res["decision"]["tone"] == "buy"]

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
SPOT_SYMBOLS = {
    "BTC": "BTC-USD", "MSTR": "MSTR", "MSTU": "MSTU",
    "GLDM": "GLDM", "GDX": "GDX", "UGL": "UGL",
    "SOXX": "SOXX", "VEGN": "VEGN", "GRID": "GRID",
    "XLE": "XLE", "OIH": "OIH", "REMX": "REMX", "WGMI": "WGMI",
}


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
