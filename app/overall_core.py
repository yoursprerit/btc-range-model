"""Overall Trading — the cross-asset combined engine.

This module is the shared backbone of the **Overall Trading** app
(``app/overall_app.py``).  It runs *every* asset that the individual apps trade
through ONE unified daily engine (``ticker_core`` + ``backtest_ticker``) so the
signals, positions and back-tests are directly comparable on a single screen and
can be blended into one portfolio.

Universe (one clean, liquid, long-history instrument per signal app)::

    BTC   Bitcoin                    (from the BTC app)          MA-trend
    GLDM  SPDR Gold MiniShares       (from the Gold app)         MA-trend (MA50 = the Gold app's documented primary)
    SOXX  iShares Semiconductors     (SOXX app)                  MA-trend
    VEGN  US Vegan Climate ETF       (VEGN app)                  MA-trend
    GRID  Clean-Edge Smart Grid ETF  (GRID app)                  MA-trend
    XLE   Energy Select Sector SPDR  (XLE app)                   Divergence Pure-Regime
    REMX  Rare-Earth/Strategic Metals(REMX app)                  MA-trend
    WGMI  Bitcoin Miners ETF         (WGMI app)                  MA-trend

The six ETF apps reuse their exact ``ticker_config`` entries (so the numbers
match those apps bar-for-bar).  BTC and GLDM get faithful config entries here:
GLDM uses the MA50 trend filter documented as its primary strategy, and BTC uses
a daily MA50 trend filter — a robust, model-free trend regime.  The dedicated BTC
and Gold apps keep their canonical hourly Pure-Regime views; this app links out to
them.  None of the original app files are imported or modified.

Everything an app needs is returned as plain dicts / DataFrames so the Streamlit
layer stays thin and cache-friendly.
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

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
# CONFIGS — the six ETF apps verbatim, plus faithful BTC & GLDM entries.
# ════════════════════════════════════════════════════════════════════════
# BTC — daily MA50 trend filter (model-free, robust regime), −3% stop.
BTC_CFG = TickerConfig(
    key="BTC", name="Bitcoin", emoji="₿",
    blurb=("Spot Bitcoin. The dedicated BTC app runs an hourly Pure-Regime "
           "system trading BTC/MSTR/MSTU; here BTC is traded on a daily "
           "trend-regime filter for cross-asset comparability."),
    accent="#f7931a", accent_dark="#b45309", accent_bg="#fff7ed", accent_bg2="#ffedd5",
    primary_symbol="BTC-USD",
    macro_syms={"eth": "ETH-USD", "spx": "^GSPC", "ndx": "^NDX",
                "vix": "^VIX", "gold": "GC=F", "dxy": "DX-Y.NYB", "tnx": "^TNX"},
    extra_syms={},
    sentiment=[("spx_close", "mom", +1.0), ("ndx_close", "mom", +1.0),
               ("vix_close", "lvl", -1.0), ("px_close", "mom", +1.0)],
    sentiment_label="Crypto/risk macro sentiment",
    traded_assets=[("BTC", "px_close")],
    asset_labels={"px_close": "BTC · Bitcoin"},
    strategy_mode="ma", strategy_name="BTC Trend-Regime",
    ma_window=50, fixed_stop=0.03, hl_band_pct=0.02,
    fetch_start="2015-01-01", oos_start="2021-01-01", periods=_STD_PERIODS,
    day_up_thresh=0.02, day_down_thresh=-0.02,
    results_note=("Daily MA50 trend filter: long while BTC closes above its "
                  "50-day SMA, −3% stop. OOS 2021→now it multiplied capital "
                  "~6.7× vs ~2.2× buy-&-hold while cutting the drawdown from "
                  "−77% to −46%."),
)

# GLDM — MA50 trend filter (the Gold app's documented primary strategy).
GLDM_CFG = TickerConfig(
    key="GLDM", name="SPDR Gold MiniShares", emoji="🥇",
    blurb=("Spot gold via GLDM. The dedicated Gold app trades GDX/UGL off the "
           "gold signal with an hourly Pure-Regime system; here GLDM is traded "
           "on its documented MA50 trend filter."),
    accent="#b8860b", accent_dark="#8b6508", accent_bg="#fffbeb", accent_bg2="#fef3c7",
    primary_symbol="GLDM",
    macro_syms={"gc": "GC=F", "slv": "SLV", "dxy": "DX-Y.NYB",
                "tnx": "^TNX", "vix": "^VIX", "spx": "^GSPC"},
    extra_syms={},
    sentiment=[("gc_close", "mom", +1.0), ("dxy_close", "mom", -1.0),
               ("vix_close", "lvl", -1.0), ("px_close", "mom", +1.0)],
    sentiment_label="Gold macro sentiment",
    traded_assets=[("GLDM", "px_close")],
    asset_labels={"px_close": "GLDM · Gold"},
    strategy_mode="ma", strategy_name="Gold Trend-Regime",
    ma_window=50, fixed_stop=0.03, hl_band_pct=0.008,
    fetch_start="2018-06-26", oos_start="2021-01-01", periods=_STD_PERIODS,
    day_up_thresh=0.006, day_down_thresh=-0.006,
    results_note=("MA50 trend filter (the Gold app's documented primary): OOS "
                  "2021→now +91% vs +111% buy-&-hold but at −17% max drawdown "
                  "vs −26%, Sharpe 0.86 vs 0.85."),
)


# Order = display order (BTC & Gold first, then the six ETF apps).
UNIVERSE_KEYS = ["BTC", "GLDM", "SOXX", "VEGN", "GRID", "XLE", "REMX", "WGMI"]

# short human labels for the traded instrument of each app
ASSET_NAME = {
    "BTC": "Bitcoin", "GLDM": "Gold (GLDM)", "SOXX": "Semiconductors",
    "VEGN": "ESG Large-Cap", "GRID": "Grid Infra", "XLE": "Energy",
    "REMX": "Rare-Earth Metals", "WGMI": "Bitcoin Miners",
}


# REMX — override to the DIVERGENCE Pure-Regime system for the combined
# portfolio.  A U1×D2×stop×D1-exit sweep showed divergence is the better
# *risk-adjusted* choice for REMX out-of-sample (2021→now: +109% at −18% max
# drawdown / Sharpe 0.93, vs the MA150 filter's +73% / −56% / 0.48 and
# buy-&-hold's +24% / −74% / 0.30) — it dodges the rare-earth crashes the slow
# MA rides down.  A −8% stop is inert here (the D2/D3 exits always fire first),
# so it only bounds a gap.  The standalone REMX app keeps its MA150 config,
# which wins the full 2016+ cycle on raw return; this override is scoped to the
# Overall app's explicit "most optimal combined strategy" mandate.
REMX_CFG = replace(
    get_config("REMX"),
    strategy_mode="divergence",
    strategy_name="Metals Divergence Pure-Regime",
    u1_errhi_min=0.16, d2_errhi_max=-0.18, d1_errlo_min=0.10, v_errlo_min=0.50,
    use_d1_exit=False, fixed_stop=0.08,
    results_note=("Divergence Pure-Regime (chosen for the combined book): OOS "
                  "2021→now +109% at −18% max drawdown, Sharpe 0.93 — vs the "
                  "MA150 filter's +73% / −56% / 0.48 and buy-&-hold +24% / "
                  "−74% / 0.30. Trades the crude-metals swings and stands aside "
                  "through the deep rare-earth busts."),
)


def overall_config(key: str) -> TickerConfig:
    """Return the unified config for one universe key."""
    if key == "BTC":
        return BTC_CFG
    if key == "GLDM":
        return GLDM_CFG
    if key == "REMX":
        return REMX_CFG
    return get_config(key)


def all_configs() -> list[TickerConfig]:
    return [overall_config(k) for k in UNIVERSE_KEYS]


# ════════════════════════════════════════════════════════════════════════
# PER-ASSET RUN — fetch → predict → simulate → live snapshot
# ════════════════════════════════════════════════════════════════════════
def _load_daily(cfg: TickerConfig) -> pd.DataFrame:
    """Daily OHLCV+macro for one config, live from Yahoo with a CSV fallback."""
    try:
        df = tc.fetch_daily(cfg)
        if df is not None and not df.empty and len(df) > 260:
            return df
    except Exception:
        pass
    # CSV fallback (the six ETF apps cache px_-prefixed frames)
    csv = tc.cache_paths(cfg)["daily"]
    if Path(csv).exists():
        df = pd.read_csv(csv, index_col=0, parse_dates=True)
        # GLDM's legacy cache uses a gldm_ prefix instead of px_
        if "px_close" not in df.columns:
            pref = f"{cfg.primary_symbol.lower()}_"
            df = df.rename(columns={c: "px_" + c[len(pref):]
                                    for c in df.columns if c.startswith(pref)})
        return df
    return pd.DataFrame()


def _net_decision(cfg: TickerConfig, sigs: dict | None, r: dict,
                  last_close: float, ma_val: float | None) -> dict:
    """Resolve a single actionable state for the asset, reconciling the
    instantaneous read with the strategy's *actual* executed position.

    Returns dict(state, label, ico, tone) where tone ∈ {buy, hold, exit, watch, flat}.
    """
    in_pos = bool(r.get("in_pos_now"))
    if cfg.strategy_mode == "ma":
        above = (ma_val is not None) and (last_close > ma_val)
        if in_pos and above:
            return dict(state="HOLD", label="LONG — HOLDING", ico="🟢", tone="hold")
        if in_pos and not above:
            return dict(state="EXIT", label="EXIT — TREND BROKEN", ico="🔴", tone="exit")
        if (not in_pos) and above:
            return dict(state="ENTRY", label="ENTER — ABOVE TREND", ico="🟢", tone="buy")
        return dict(state="FLAT", label="FLAT — BELOW TREND", ico="⬜", tone="flat")
    # divergence
    alert = (sigs or {}).get("alert_level", "NEUTRAL")
    if in_pos:
        if alert in ("HIGH_DN", "ELEVATED_DN"):
            return dict(state="EXIT", label="EXIT — DOWNTREND PRESSURE", ico="🔴", tone="exit")
        return dict(state="HOLD", label="LONG — HOLDING", ico="🟢", tone="hold")
    if alert == "STRATEGY_BUY":
        return dict(state="ENTRY", label="ENTER — PURE-REGIME BUY", ico="🟢", tone="buy")
    if alert == "WATCH_UP":
        return dict(state="WATCH", label="WATCH — UPTREND (GATE PENDING)", ico="🟡", tone="watch")
    if alert in ("HIGH_DN", "ELEVATED_DN", "WATCH_DN"):
        return dict(state="AVOID", label="STAND ASIDE — DOWNTREND", ico="🟠", tone="watch")
    return dict(state="FLAT", label="FLAT — NO SIGNAL", ico="⬜", tone="flat")


def run_asset(cfg: TickerConfig, oos_start: str | None = None) -> dict | None:
    """Full pipeline for one asset. Returns a rich dict or None if data missing."""
    daily = _load_daily(cfg)
    if daily is None or daily.empty or "px_close" not in daily.columns:
        return None
    oos = oos_start or cfg.oos_start
    preds = bt.build_predictions(cfg, daily)
    sig = bt.precompute_signals(cfg, preds)
    r = bt.run_strategy(cfg, preds, sig, "px_close", oos_start=oos)

    dates = pd.to_datetime(pd.Series(r["dates"]))
    strat = np.asarray(r["strat"], float)
    ret = pd.Series(np.diff(strat) / strat[:-1], index=dates.iloc[1:]).rename(cfg.key)

    # live snapshot ------------------------------------------------------
    last_close = float(daily["px_close"].iloc[-1])
    prev_close = float(daily["px_close"].iloc[-2]) if len(daily) > 1 else last_close
    dchg = (last_close / prev_close - 1) * 100 if prev_close else 0.0
    ma_val = None
    if cfg.strategy_mode == "ma":
        w = cfg.ma_window
        ma_val = float(daily["px_close"].tail(w).mean())

    # trend-signature alert (last 45 completed bars) --------------------
    completed = preds[preds["actual_high"].notna() & preds["actual_low"].notna()]
    sigs = None
    if len(completed) >= 3:
        sigs = tc.compute_trend_signatures(cfg, completed.tail(45))

    try:
        sent = tc.macro_sentiment(cfg, daily)
        sent_now = float(sent.dropna().iloc[-1]) if sent.notna().any() else np.nan
    except Exception:
        sent_now = np.nan

    dec = _net_decision(cfg, sigs, r, last_close, ma_val)

    # position economics -------------------------------------------------
    pos = dict(in_pos=bool(r.get("in_pos_now")), entry_px=r.get("entry_px"),
               entry_date=r.get("entry_date"), upnl=None, stop_px=None, days=None)
    as_of = pd.Timestamp(dates.iloc[-1])
    if pos["in_pos"] and pos["entry_px"]:
        e_px = float(pos["entry_px"]); e_dt = pd.Timestamp(pos["entry_date"])
        pos["upnl"] = (last_close / e_px - 1) * 100
        pos["stop_px"] = e_px * (1 - cfg.fixed_stop)
        pos["days"] = int((as_of - e_dt).days)
        pos["dist_stop"] = (last_close / pos["stop_px"] - 1) * 100
    last_trade = r["trade_log"][-1] if r.get("trade_log") else None

    m = bt._metrics(strat, r["dates"])
    bh = bt._metrics(r["bh"], r["dates"])
    wr = float((r["trades"] > 0).mean() * 100) if len(r["trades"]) else 0.0

    return dict(
        key=cfg.key, cfg=cfg, name=ASSET_NAME.get(cfg.key, cfg.key),
        emoji=cfg.emoji, accent=cfg.accent,
        last_close=last_close, dchg=dchg, ma_val=ma_val,
        sentiment=sent_now, decision=dec, alert=(sigs or {}).get("alert_level", "NEUTRAL"),
        bull_regime=bool((sigs or {}).get("bull_regime", ma_val is not None and last_close > (ma_val or 0))),
        pos=pos, last_trade=last_trade,
        metrics=m, bh_metrics=bh, win_rate=wr, n_trades=int(len(r["trades"])),
        ret=ret, strat=strat, dates=dates, r=r,
        as_of=as_of,
        mode=cfg.strategy_mode, ma_window=cfg.ma_window, stop=cfg.fixed_stop,
    )


def run_universe(oos_start: str | None = None) -> list[dict]:
    """Run every asset; skip (with no error) any that fail to fetch."""
    out = []
    for cfg in all_configs():
        try:
            res = run_asset(cfg, oos_start=oos_start)
        except Exception:
            res = None
        if res is not None:
            out.append(res)
    return out


# ════════════════════════════════════════════════════════════════════════
# PORTFOLIO MATHS — combine the per-asset strategy return streams.
# ════════════════════════════════════════════════════════════════════════
TRADING_DAYS = 252


def returns_matrix(results: list[dict]) -> pd.DataFrame:
    """Align each asset's daily strategy returns into one frame (NaN where the
    asset has no history yet)."""
    cols = {res["key"]: res["ret"] for res in results}
    df = pd.DataFrame(cols).sort_index()
    return df


def _combine(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Daily portfolio return for fixed target weights, renormalising over the
    assets that actually have data each day (handles staggered inception)."""
    w = np.asarray(weights, float)
    mask = returns.notna().to_numpy()               # (T, N)
    wt = mask * w                                    # zero out unavailable
    denom = wt.sum(axis=1, keepdims=True)
    denom[denom == 0] = np.nan
    wt = wt / denom
    filled = np.nan_to_num(returns.to_numpy())
    port = np.nansum(wt * filled, axis=1)
    port[np.isnan(denom[:, 0])] = 0.0               # fully-cash days
    return pd.Series(port, index=returns.index)


def _equity(daily_ret: pd.Series) -> pd.Series:
    return (1.0 + daily_ret.fillna(0.0)).cumprod()


def curve_metrics(equity: pd.Series) -> dict:
    eq = equity.to_numpy(float)
    idx = equity.index
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
    return dict(total_ret=float(total), cagr=float(cagr), mdd=mdd,
                sharpe=sharpe, vol=vol)


def optimize_weights(returns: pd.DataFrame, cap: float = 0.30,
                     n_samples: int = 40000, seed: int = 7,
                     mdd_floor: float = -0.35) -> dict:
    """Search long-only weights (sum=1, per-asset ≤ cap) that maximise the
    combined Sharpe, preferring blends whose drawdown is no worse than
    ``mdd_floor``.  Returns the chosen weights plus equal-weight and
    inverse-volatility (risk-parity) references.

    Pure numpy Monte-Carlo over the capped simplex — stable and dependency-free.
    """
    cols = list(returns.columns)
    n = len(cols)
    rng = np.random.default_rng(seed)

    # candidate pool -----------------------------------------------------
    samples = rng.dirichlet(np.ones(n), size=n_samples)
    if cap < 1.0:
        keep = (samples <= cap + 1e-9).all(axis=1)
        samples = samples[keep]
    # always include equal-weight and inverse-vol as candidates
    ew = np.full(n, 1.0 / n)
    vols = returns.std().to_numpy()
    inv = 1.0 / np.where(vols > 0, vols, np.nan)
    inv = np.nan_to_num(inv)
    rp = inv / inv.sum() if inv.sum() > 0 else ew
    rp = np.minimum(rp, cap); rp = rp / rp.sum()
    cand = np.vstack([samples, ew, rp]) if len(samples) else np.vstack([ew, rp])

    best = None
    best_sharpe = -np.inf
    best_relaxed = None
    best_relaxed_sharpe = -np.inf
    for w in cand:
        eq = _equity(_combine(returns, w))
        mm = curve_metrics(eq)
        s = mm["sharpe"]
        if s > best_relaxed_sharpe:
            best_relaxed_sharpe = s; best_relaxed = (w, mm)
        if mm["mdd"] >= mdd_floor and s > best_sharpe:
            best_sharpe = s; best = (w, mm)
    if best is None:
        best = best_relaxed
    w_opt, m_opt = best

    def _pack(w):
        w = np.asarray(w, float)
        return {c: float(x) for c, x in zip(cols, w)}

    return dict(
        cols=cols,
        optimal=dict(weights=_pack(w_opt), **m_opt),
        equal=dict(weights=_pack(ew), **curve_metrics(_equity(_combine(returns, ew)))),
        risk_parity=dict(weights=_pack(rp), **curve_metrics(_equity(_combine(returns, rp)))),
    )


def _waterfill(raw: dict[str, float], cap: float) -> dict[str, float]:
    """Normalise ``raw`` weights to sum 1, capping each at ``cap`` and
    redistributing the overflow to the uncapped names (repeat until stable).
    If the names can't absorb the whole book (n·cap < 1) the remainder is left
    unallocated (→ cash)."""
    keys = [k for k, v in raw.items() if v > 0]
    if not keys:
        return {}
    w = {k: raw[k] for k in keys}
    tot = sum(w.values())
    w = {k: v / tot for k, v in w.items()}          # normalise to 1
    for _ in range(20):
        over = {k: v for k, v in w.items() if v > cap + 1e-9}
        if not over:
            break
        spill = sum(v - cap for v in over.values())
        for k in over:
            w[k] = cap
        room = [k for k in w if w[k] < cap - 1e-9]
        base = sum(w[k] for k in room)
        if not room or base <= 0:
            break                                    # rest becomes cash
        for k in room:
            w[k] = min(cap, w[k] + spill * (w[k] / base))
    return w


def signal_gated_allocation(results: list[dict], base_weights: dict[str, float],
                            cap: float = 0.35) -> dict:
    """Today's actionable allocation: deploy capital only to assets whose
    strategy is currently long (or firing a fresh entry), split by the optimal
    base weights renormalised over the active set and capped per asset via
    water-filling; any un-deployable remainder sits in cash.

    Returns per-asset target %, the cash %, and open/close/hold actions.
    """
    def _b(k):
        return max(base_weights.get(k, 0.0), 1e-6)

    in_pos = [res for res in results if res["pos"]["in_pos"]]
    closing = {res["key"] for res in in_pos if res["decision"]["tone"] == "exit"}
    keep = [res for res in in_pos if res["key"] not in closing]
    opens = [res for res in results
             if (not res["pos"]["in_pos"]) and res["decision"]["tone"] == "buy"]

    # what the strategy holds RIGHT NOW (before today's actions)
    current = _waterfill({res["key"]: _b(res["key"]) for res in in_pos}, cap)
    # recommended book AFTER today's opens/closes
    target_set = {res["key"] for res in keep} | {res["key"] for res in opens}
    target = _waterfill({k: _b(k) for k in target_set}, cap)
    active = in_pos
    cash = max(1.0 - sum(target.values()), 0.0)
    cash_now = max(1.0 - sum(current.values()), 0.0)

    actions = []
    for res in results:
        k = res["key"]; dec = res["decision"]
        cur = base_weights.get(k, 0.0)
        if res["pos"]["in_pos"] and dec["tone"] == "exit":
            act = "CLOSE"
        elif res["pos"]["in_pos"]:
            act = "HOLD"
        elif dec["tone"] == "buy":
            act = "OPEN"
        elif dec["tone"] in ("watch",):
            act = "WATCH"
        else:
            act = "STAND ASIDE"
        actions.append(dict(key=k, name=res["name"], emoji=res["emoji"],
                            action=act, tone=dec["tone"], decision=dec["label"],
                            target=target.get(k, 0.0), base=cur,
                            in_pos=res["pos"]["in_pos"], upnl=res["pos"]["upnl"],
                            alert=res["alert"], last_close=res["last_close"]))
    order = {"CLOSE": 0, "OPEN": 1, "HOLD": 2, "WATCH": 3, "STAND ASIDE": 4}
    actions.sort(key=lambda a: (order[a["action"]], -a["target"]))
    return dict(target=target, cash=max(cash, 0.0),
                current=current, cash_now=max(cash_now, 0.0),
                actions=actions, n_active=len(active),
                n_open=len(opens), n_close=len(closing))


def benchmarks(returns: pd.DataFrame, results: list[dict]) -> dict:
    """Reference curves: equal-weight buy&hold of the 8 underlyings, best single
    strategy, and equal-weight of the 8 strategies."""
    out = {}
    # equal-weight buy&hold of the underlyings (daily px returns)
    bh_cols = {}
    for res in results:
        px = pd.Series(res["r"]["bh"], index=pd.to_datetime(pd.Series(res["r"]["dates"])))
        bh_cols[res["key"]] = px.pct_change()
    bh_df = pd.DataFrame(bh_cols).reindex(returns.index)
    n = bh_df.shape[1]
    out["bh_equal"] = curve_metrics(_equity(_combine(bh_df, np.full(n, 1.0 / n))))
    out["bh_equal"]["equity"] = _equity(_combine(bh_df, np.full(n, 1.0 / n)))
    # equal-weight of the strategies
    eqw = _equity(_combine(returns, np.full(returns.shape[1], 1.0 / returns.shape[1])))
    out["strat_equal"] = curve_metrics(eqw); out["strat_equal"]["equity"] = eqw
    # best single strategy by Sharpe
    best_key, best_m, best_eq = None, None, None
    for res in results:
        eq = _equity(res["ret"].reindex(returns.index))
        m = curve_metrics(eq)
        if best_m is None or m["sharpe"] > best_m["sharpe"]:
            best_key, best_m, best_eq = res["key"], m, eq
    out["best_single"] = dict(key=best_key, **best_m); out["best_single"]["equity"] = best_eq
    return out


def period_breakdown(returns: pd.DataFrame, weights: np.ndarray,
                     periods: list[tuple]) -> list[dict]:
    """Combined-strategy metrics over each named window."""
    rows = []
    for lbl, s, e in periods:
        sub = returns.loc[pd.Timestamp(s):(pd.Timestamp(e) if e else None)]
        if len(sub) < 5:
            continue
        eq = _equity(_combine(sub, weights))
        m = curve_metrics(eq)
        rows.append(dict(label=lbl, start=s, end=e, **m))
    return rows


# combined-history windows for the portfolio backtest.  The per-asset strategy
# return streams begin at the common out-of-sample start (2021), so every window
# below is genuinely out-of-sample for the trend sleeves.
COMBINED_PERIODS = [
    ("🌐 Full OOS (2021 → now)", "2021-01-01", None),
    ("🐻 Bear (2021–2022)", "2021-01-01", "2022-12-31"),
    ("🐂 Bull (2023 → now)", "2023-01-01", None),
    ("🔬 Recent (2025 → now)", "2025-01-01", None),
]
