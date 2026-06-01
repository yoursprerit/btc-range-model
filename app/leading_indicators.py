"""
Leading Indicators tab — BTC price correlation & next-day / next-week forecast.

10 indicators tracked:
  1. Short-Term Holder Cost Basis  (155-day rolling avg — standard proxy)
  2. True Market Mean              (365d/730d MA blend — balanced-price proxy)
  3. 200-Day Moving Average
  4. Dealer Negative Gamma         (live GEX via Deribit public API)
  5. 200-Week Moving Average
  6. Coinbase Premium              (Coinbase BTC − Binance BTC, already fetched)
  7. 30-Day Net Flow Proxy         (rolling tx-volume momentum, blockchain.info)
  8. Stablecoin Supply Change      (USDT+USDC 30d delta via CoinGecko)
  9. Sell-Side Risk Proxy          (volume spike vs 365d avg — SSRR approximation)
 10. 180-Day Realized Price Change (momentum)

For metrics that require paid on-chain APIs (Glassnode/CryptoQuant), the best
publicly available free approximations are used and clearly labeled.
"""

from __future__ import annotations

import math
import time
import warnings
from datetime import timezone, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

# ── palette ──────────────────────────────────────────────────────────────────
BTC_ORANGE = "#F7931A"
BULL_GREEN = "#22c55e"
BEAR_RED   = "#ef4444"
NEUTRAL    = "#94a3b8"
GRID_CLR   = "#e2e8f0"
BG_LIGHT   = "#f8fafc"

# ── indicator metadata ────────────────────────────────────────────────────────
INDICATORS: dict[str, dict] = {
    "sth_cost_basis": dict(
        label="STH Cost Basis",
        short="STH CB",
        unit="$",
        description=(
            "Short-Term Holder cost basis — approximated as the 155-day rolling "
            "average BTC price (coins moved in the last 155 days). "
            "BTC below this level → STH unrealized loss → capitulation risk. "
            "Source: price-derived (official via Glassnode)."
        ),
        bullish_when="price_above",
    ),
    "true_market_mean": dict(
        label="True Market Mean",
        short="TMM",
        unit="$",
        description=(
            "Balanced-price proxy — average of the 365-day and 730-day simple "
            "moving averages, approximating the midpoint between realized price "
            "and transfer price. Acts as a long-term fair-value anchor. "
            "Source: price-derived (official via Glassnode)."
        ),
        bullish_when="price_above",
    ),
    "ma_200d": dict(
        label="200-Day Moving Average",
        short="200d MA",
        unit="$",
        description=(
            "Classic bull/bear regime separator. Price above 200d MA → bull "
            "market structure; below → bear. Also used as dynamic support/resistance."
        ),
        bullish_when="price_above",
    ),
    "dealer_gex": dict(
        label="Dealer Gamma Exposure",
        short="Dealer GEX",
        unit="BTC",
        description=(
            "Net Gamma Exposure of market makers in BTC options (Deribit). "
            "GEX > 0 → dealers long gamma (stabilizing: buy dips, sell rips). "
            "GEX < 0 → dealers short gamma (amplifying: forced to sell on drops, "
            "buy on rallies). Computed from Black-Scholes gamma × open interest."
        ),
        bullish_when="positive",
    ),
    "ma_200w": dict(
        label="200-Week Moving Average",
        short="200w MA",
        unit="$",
        description=(
            "The historical Bitcoin bear market floor. Price below the 200-week MA "
            "has historically been the best long-term accumulation zone."
        ),
        bullish_when="price_above",
    ),
    "coinbase_premium": dict(
        label="Coinbase Premium",
        short="CB Prem",
        unit="$",
        description=(
            "BTC/USD price on Coinbase minus BTC/USDT price on Binance. "
            "Positive → US institutional/retail demand outpacing global; "
            "negative → selling pressure from US participants."
        ),
        bullish_when="positive",
    ),
    "net_flow_30d": dict(
        label="30D Net Flow Proxy",
        short="Flow30",
        unit="%",
        description=(
            "30-day momentum of estimated on-chain transaction volume relative to "
            "its own 90-day baseline. Rising → elevated exchange inflows (sell "
            "pressure); falling → outflows / HODLing. "
            "Source: blockchain.info tx-volume (proxy for net exchange flows)."
        ),
        bullish_when="negative",
    ),
    "stablecoin_flow_30d": dict(
        label="Stablecoin Supply Δ (30d)",
        short="Stable Δ",
        unit="$B",
        description=(
            "30-day change in combined USDT + USDC market cap (in billions). "
            "Growing stablecoin supply = fresh 'dry powder' capital parked on the "
            "sidelines, historically bullish for BTC when deployed. "
            "Source: CoinGecko free API."
        ),
        bullish_when="positive",
    ),
    "sell_side_risk": dict(
        label="Sell-Side Risk Proxy",
        short="SSRP",
        unit="×",
        description=(
            "30-day average daily volume divided by the 365-day average daily "
            "volume. Values >1 indicate elevated trading activity — a proxy for "
            "the official Sell-Side Risk Ratio (which requires UTXO data). "
            "High values near cycle tops/bottoms often signal local extremes."
        ),
        bullish_when="low",
    ),
    "realized_chg_180d": dict(
        label="180-Day Price Change",
        short="180d Chg",
        unit="%",
        description=(
            "BTC close price change over the past 180 days. Captures the "
            "medium-term momentum regime and mirrors the '180-Day Realized Price "
            "Change' metric used by on-chain analysts."
        ),
        bullish_when="positive",
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# Data fetching
# ══════════════════════════════════════════════════════════════════════════════

def _bs_gamma(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Black-Scholes gamma (same for calls and puts)."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        return math.exp(-0.5 * d1**2) / (math.sqrt(2 * math.pi) * S * sigma * math.sqrt(T))
    except Exception:
        return 0.0


@st.cache_data(ttl=3600 * 2, show_spinner=False)
def _fetch_deribit_gex() -> tuple[float | None, float | None]:
    """Fetch BTC option book summary from Deribit and compute net GEX.

    GEX = Σ_calls(gamma × OI) − Σ_puts(gamma × OI), in BTC units.
    Positive GEX → dealers net-long gamma (price-stabilizing).
    Negative GEX → dealers net-short gamma (move-amplifying).
    """
    try:
        url = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
        r = requests.get(url, params={"currency": "BTC", "kind": "option"}, timeout=25)
        if r.status_code != 200:
            return None, None
        items = r.json().get("result", [])
        if not items:
            return None, None

        now_utc = datetime.now(timezone.utc)
        spot = None
        call_gex = 0.0
        put_gex = 0.0

        for item in items:
            name = item.get("instrument_name", "")
            parts = name.split("-")
            if len(parts) != 4:
                continue
            # BTC-DDMMMYY-STRIKE-C/P
            try:
                strike = float(parts[2])
                opt_type = parts[3]  # "C" or "P"
            except ValueError:
                continue

            # Expiry — parse "26JAN24" → datetime
            try:
                exp = datetime.strptime(parts[1], "%d%b%y").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            T = max((exp - now_utc).total_seconds() / (365.25 * 86400), 0)

            underlying = item.get("underlying_price") or 0
            if not spot and underlying:
                spot = float(underlying)
            S = spot or 100_000.0

            # Implied vol from mark_iv (annualised %, convert to decimal)
            iv = item.get("mark_iv")
            if not iv:
                continue
            sigma = float(iv) / 100.0

            oi = float(item.get("open_interest") or 0)
            if oi <= 0:
                continue

            g = _bs_gamma(S, strike, T, sigma)
            if opt_type == "C":
                call_gex += g * oi
            elif opt_type == "P":
                put_gex += g * oi

        net_gex = call_gex - put_gex
        return net_gex, spot
    except Exception:
        return None, None


@st.cache_data(ttl=3600 * 6, show_spinner=False)
def _fetch_stablecoin_mcap() -> pd.Series:
    """Fetch combined USDT + USDC daily market cap from CoinGecko (free API).

    Returns a Series indexed by UTC date with values in USD billions.
    Returns empty Series on failure.
    """
    combined: dict[pd.Timestamp, float] = {}
    for coin in ["tether", "usd-coin"]:
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
            r = requests.get(
                url,
                params={"vs_currency": "usd", "days": "400", "interval": "daily"},
                timeout=25,
            )
            if r.status_code != 200:
                continue
            data = r.json()
            for ts_ms, cap in data.get("market_caps", []):
                dt = pd.Timestamp(ts_ms, unit="ms").normalize()
                combined[dt] = combined.get(dt, 0.0) + cap / 1e9
            time.sleep(0.5)  # CoinGecko free tier rate limit
        except Exception:
            continue
    if not combined:
        return pd.Series(dtype=float, name="stablecoin_mcap_b")
    s = pd.Series(combined, name="stablecoin_mcap_b").sort_index()
    return s[~s.index.duplicated(keep="last")]


@st.cache_data(ttl=3600 * 6, show_spinner=False)
def _fetch_blockchain_tx_volume() -> pd.Series:
    """Fetch estimated-transaction-volume-usd from blockchain.info."""
    try:
        j = requests.get(
            "https://api.blockchain.info/charts/estimated-transaction-volume-usd"
            "?timespan=all&format=json&sampled=false",
            timeout=30,
        ).json()
        idx = pd.to_datetime([x["x"] for x in j["values"]], unit="s").normalize()
        vals = [x["y"] for x in j["values"]]
        s = pd.Series(vals, index=idx, name="btc_tx_vol_usd")
        return s[~s.index.duplicated(keep="last")].sort_index()
    except Exception:
        return pd.Series(dtype=float, name="btc_tx_vol_usd")


def compute_price_indicators(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Compute all price-derived leading indicators from daily OHLCV data.

    Parameters
    ----------
    daily_df : DataFrame with at minimum columns ['btc_close'] and optionally
               ['btc_volume', 'coinbase_close'] indexed by bar_start dates.

    Returns
    -------
    DataFrame of same index with indicator columns.
    """
    close = daily_df["btc_close"].copy()
    vol = daily_df.get("btc_volume", pd.Series(dtype=float, index=daily_df.index))

    df = pd.DataFrame(index=daily_df.index)

    # 1. STH Cost Basis (≈155-day rolling mean — standard on-chain proxy)
    df["sth_cost_basis"] = close.rolling(155, min_periods=50).mean()

    # 2. True Market Mean (balanced-price proxy: avg of 365d and 730d SMA)
    df["true_market_mean"] = (
        close.rolling(365, min_periods=120).mean()
        + close.rolling(730, min_periods=250).mean()
    ) / 2.0

    # 3. 200-Day Moving Average
    df["ma_200d"] = close.rolling(200, min_periods=100).mean()

    # 4. 200-Week Moving Average (200 weeks ≈ 1400 calendar days)
    df["ma_200w"] = close.rolling(1400, min_periods=400).mean()

    # 5. Coinbase Premium (USD)
    if "coinbase_close" in daily_df.columns:
        df["coinbase_premium"] = (daily_df["coinbase_close"] - close).fillna(0)
    else:
        df["coinbase_premium"] = 0.0

    # 6. 30-Day Net Flow Proxy — computed once tx_vol is joined
    #    Placeholder filled later in fetch_all_leading_indicators
    df["net_flow_30d"] = np.nan

    # 7. Stablecoin flow — filled later
    df["stablecoin_flow_30d"] = np.nan

    # 8. Sell-Side Risk Proxy: 30d avg vol / 365d avg vol (>1 = high activity)
    if vol.notna().sum() > 30:
        vol30  = vol.rolling(30,  min_periods=10).mean()
        vol365 = vol.rolling(365, min_periods=120).mean()
        df["sell_side_risk"] = (vol30 / vol365).replace([np.inf, -np.inf], np.nan)
    else:
        df["sell_side_risk"] = np.nan

    # 9. 180-Day Realized Price Change (%)
    df["realized_chg_180d"] = (close / close.shift(180) - 1) * 100

    # 10. Dealer GEX — live only, not historical; filled in render
    df["dealer_gex"] = np.nan

    return df


@st.cache_data(ttl=3600 * 6, show_spinner="Building leading indicator dataset …")
def fetch_all_leading_indicators(daily_df_json: str) -> pd.DataFrame:
    """Assemble the full leading-indicators DataFrame.

    `daily_df_json` is a JSON-serialised snapshot of the daily OHLCV + macro
    frame (from _fetch_daily_raw) — used as cache key so the cache busts
    whenever new daily data arrives.
    """
    import io
    daily_df = pd.read_json(io.StringIO(daily_df_json), orient="split")
    daily_df.index = pd.to_datetime(daily_df.index)

    ind = compute_price_indicators(daily_df)

    # ── On-chain tx volume → net_flow_30d ────────────────────────────────
    tx_vol = _fetch_blockchain_tx_volume()
    if not tx_vol.empty:
        tx_aligned = tx_vol.reindex(ind.index).ffill(limit=7)
        baseline = tx_aligned.rolling(90, min_periods=30).mean()
        tx_norm = (tx_aligned / baseline - 1) * 100  # % vs 90d baseline
        # 30d momentum: current 30d avg vs prior 30d avg
        flow30 = tx_norm.rolling(30, min_periods=10).mean()
        flow_prev = tx_norm.shift(30).rolling(30, min_periods=10).mean()
        ind["net_flow_30d"] = flow30 - flow_prev

    # ── Stablecoin supply → stablecoin_flow_30d ──────────────────────────
    sc_mcap = _fetch_stablecoin_mcap()
    if not sc_mcap.empty:
        sc_aligned = sc_mcap.reindex(ind.index).ffill(limit=7)
        ind["stablecoin_flow_30d"] = sc_aligned.diff(30)

    return ind


# ══════════════════════════════════════════════════════════════════════════════
# Predictive model
# ══════════════════════════════════════════════════════════════════════════════

_FEAT_COLS = [
    "sth_basis_ratio",
    "tmm_ratio",
    "ma_200d_ratio",
    "ma_200w_ratio",
    "coinbase_premium_pct",
    "net_flow_30d",
    "sell_side_risk",
    "realized_chg_180d",
    # stablecoin_flow and dealer_gex added when available
]


def _build_model_features(close: pd.Series, ind: pd.DataFrame) -> pd.DataFrame:
    """Construct ratio-based features for the predictive model.

    Using ratios (price / indicator) rather than raw levels avoids spurious
    correlation from shared long-run trends.
    """
    f = pd.DataFrame(index=close.index)
    f["sth_basis_ratio"]    = (close / ind["sth_cost_basis"]  - 1) * 100
    f["tmm_ratio"]          = (close / ind["true_market_mean"] - 1) * 100
    f["ma_200d_ratio"]      = (close / ind["ma_200d"]          - 1) * 100
    f["ma_200w_ratio"]      = (close / ind["ma_200w"]          - 1) * 100
    f["coinbase_premium_pct"] = ind["coinbase_premium"] / close * 100
    f["net_flow_30d"]       = ind["net_flow_30d"]
    f["sell_side_risk"]     = ind["sell_side_risk"]
    f["realized_chg_180d"]  = ind["realized_chg_180d"]

    if ind["stablecoin_flow_30d"].notna().any():
        f["stablecoin_flow_30d"] = ind["stablecoin_flow_30d"]
    return f


def build_forecast_models(close: pd.Series, ind: pd.DataFrame, test_days: int = 180):
    """Train GBM models to predict 1-day and 7-day forward returns.

    Returns
    -------
    dict with keys:
      model_1d, model_7d, scaler, feat_cols,
      test_metrics_1d, test_metrics_7d,
      X_test, y1_test, y7_test, y1_pred, y7_pred, test_index
    """
    if not _SKLEARN:
        return None

    feat_df = _build_model_features(close, ind)
    feat_df["target_1d"] = (close.shift(-1) / close - 1) * 100
    feat_df["target_7d"] = (close.shift(-7) / close - 1) * 100

    all_feat_cols = [c for c in feat_df.columns if c not in ("target_1d", "target_7d")]
    # Keep only columns that have enough non-null data (≥30% coverage)
    min_rows = max(60, len(feat_df) * 0.30)
    feat_cols = [c for c in all_feat_cols if feat_df[c].notna().sum() >= min_rows]
    if not feat_cols:
        return None
    # Require targets and at least the core price-ratio columns to be non-NaN
    core_cols = [c for c in ("sth_basis_ratio", "ma_200d_ratio") if c in feat_cols]
    drop_subset = (core_cols or feat_cols[:1]) + ["target_1d", "target_7d"]
    feat_df = feat_df.dropna(subset=drop_subset)

    if len(feat_df) < test_days + 60:
        test_days = max(30, len(feat_df) // 4)

    train = feat_df.iloc[:-test_days]
    test  = feat_df.iloc[-test_days:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feat_cols].fillna(0))
    X_test  = scaler.transform(test[feat_cols].fillna(0))

    y1_tr, y7_tr = train["target_1d"].values, train["target_7d"].values
    y1_te, y7_te = test["target_1d"].values,  test["target_7d"].values

    gbm_params = dict(n_estimators=150, max_depth=3, learning_rate=0.05,
                      subsample=0.8, random_state=42)
    m1 = GradientBoostingRegressor(**gbm_params)
    m7 = GradientBoostingRegressor(**gbm_params)
    m1.fit(X_train, y1_tr)
    m7.fit(X_train, y7_tr)

    p1 = m1.predict(X_test)
    p7 = m7.predict(X_test)

    def _dir_acc(y, yhat):
        mask = np.sign(y) == np.sign(yhat)
        return mask.mean() * 100

    def _mae(y, yhat):
        return np.abs(y - yhat).mean()

    return dict(
        model_1d=m1, model_7d=m7, scaler=scaler, feat_cols=feat_cols,
        test_metrics_1d=dict(dir_acc=_dir_acc(y1_te, p1), mae=_mae(y1_te, p1)),
        test_metrics_7d=dict(dir_acc=_dir_acc(y7_te, p7), mae=_mae(y7_te, p7)),
        X_test=X_test, y1_test=y1_te, y7_test=y7_te,
        y1_pred=p1, y7_pred=p7, test_index=test.index,
        feat_df=feat_df,
    )


def current_forecast(close: pd.Series, ind: pd.DataFrame, model_bundle: dict, gex: float | None):
    """Apply trained models to the latest available features.

    Returns (pred_1d_pct, pred_7d_pct, feat_row_dict)
    """
    if model_bundle is None:
        return None, None, {}

    feat_df = _build_model_features(close, ind)
    if gex is not None:
        feat_df["dealer_gex_norm"] = gex  # level at last data point

    feat_cols = model_bundle["feat_cols"]
    # Use only columns that exist in both
    common = [c for c in feat_cols if c in feat_df.columns]
    latest = feat_df[common].dropna()
    if latest.empty:
        return None, None, {}

    row = latest.iloc[[-1]]
    # Fill any missing model features with 0
    X = pd.DataFrame(0.0, index=row.index, columns=feat_cols)
    X[common] = row[common]

    scaler = model_bundle["scaler"]
    Xs = scaler.transform(X.fillna(0))

    pred_1d = float(model_bundle["model_1d"].predict(Xs)[0])
    pred_7d = float(model_bundle["model_7d"].predict(Xs)[0])
    return pred_1d, pred_7d, row.to_dict(orient="records")[0]


# ══════════════════════════════════════════════════════════════════════════════
# Plotly helpers
# ══════════════════════════════════════════════════════════════════════════════

def _base_layout(**kw) -> dict:
    return dict(
        plot_bgcolor=BG_LIGHT,
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=12),
        margin=dict(l=10, r=10, t=40, b=30),
        hovermode="x unified",
        xaxis=dict(showgrid=True, gridcolor=GRID_CLR, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=GRID_CLR, zeroline=False),
        legend=dict(orientation="h", y=-0.15),
        **kw,
    )


def _dual_axis_chart(
    dates, price, indicator, ind_label: str, ind_unit: str, title: str
) -> go.Figure:
    """Price (right axis) vs indicator (left axis), last 2 years."""
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=730)
    mask = pd.Series(dates) >= cutoff
    d = pd.Series(dates)[mask]
    p = pd.Series(price)[mask]
    v = pd.Series(indicator)[mask]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d, y=v, name=ind_label,
        line=dict(color="#6366f1", width=2),
        yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=d, y=p, name="BTC Price",
        line=dict(color=BTC_ORANGE, width=1.5, dash="dot"),
        yaxis="y2",
    ))
    layout = _base_layout(title=title)
    layout.update(
        yaxis=dict(title=f"{ind_label} ({ind_unit})", showgrid=True, gridcolor=GRID_CLR),
        yaxis2=dict(title="BTC Price ($)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=1.05, x=0),
    )
    fig.update_layout(**layout)
    return fig


def _correlation_heatmap(corr_df: pd.DataFrame) -> go.Figure:
    """Heatmap of leading indicator → forward BTC return correlations."""
    z = corr_df.values
    fig = go.Figure(go.Heatmap(
        z=z,
        x=corr_df.columns.tolist(),
        y=corr_df.index.tolist(),
        colorscale=[
            [0.0,  "#ef4444"],
            [0.5,  "#f8fafc"],
            [1.0,  "#22c55e"],
        ],
        zmid=0, zmin=-1, zmax=1,
        text=np.round(z, 2),
        texttemplate="%{text}",
        hovertemplate="%{y} → %{x}: %{z:.3f}<extra></extra>",
        colorbar=dict(title="r"),
    ))
    layout = _base_layout(title="Pearson Correlation: Indicator → Forward BTC Return")
    layout.update(margin=dict(l=180, r=20, t=50, b=80), xaxis=dict(tickangle=-30))
    fig.update_layout(**layout)
    return fig


def _backtest_scatter(y_true, y_pred, horizon: str, index) -> go.Figure:
    """Predicted vs actual return scatter for the test set."""
    colors = [BULL_GREEN if np.sign(p) == np.sign(a) else BEAR_RED
              for p, a in zip(y_pred, y_true)]
    fig = go.Figure(go.Scatter(
        x=list(y_true), y=list(y_pred),
        mode="markers",
        marker=dict(color=colors, size=5, opacity=0.7),
        text=[str(d.date()) for d in index],
        hovertemplate="Date: %{text}<br>Actual: %{x:.2f}%<br>Pred: %{y:.2f}%<extra></extra>",
    ))
    mn = min(y_true.min(), y_pred.min()) * 1.1
    mx = max(y_true.max(), y_pred.max()) * 1.1
    fig.add_shape(type="line", x0=mn, x1=mx, y0=mn, y1=mx,
                  line=dict(color=NEUTRAL, dash="dash"))
    fig.update_layout(
        **_base_layout(title=f"{horizon} — Predicted vs Actual Return (test set)"),
        xaxis_title="Actual return (%)",
        yaxis_title="Predicted return (%)",
    )
    return fig


def _time_series_predictions(test_index, y_true, y_pred, horizon: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_index, y=y_true, name="Actual",
                             line=dict(color=BTC_ORANGE, width=1.5)))
    fig.add_trace(go.Scatter(x=test_index, y=y_pred, name="Predicted",
                             line=dict(color="#6366f1", width=1.5, dash="dash")))
    fig.add_hline(y=0, line=dict(color=NEUTRAL, dash="dot", width=1))
    fig.update_layout(
        **_base_layout(title=f"{horizon} Returns — Actual vs Model (test set)"),
        yaxis_title="Return (%)",
    )
    return fig


def _feature_importance_chart(model, feat_cols: list[str]) -> go.Figure:
    imp = pd.Series(model.feature_importances_, index=feat_cols).sort_values()
    fig = go.Figure(go.Bar(
        x=imp.values, y=imp.index, orientation="h",
        marker_color=[BTC_ORANGE if v > 0 else NEUTRAL for v in imp.values],
    ))
    layout = _base_layout(title="Feature Importance — 1-Day Model")
    layout.update(margin=dict(l=160, r=20, t=40, b=30), xaxis_title="Importance")
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Main render function
# ══════════════════════════════════════════════════════════════════════════════

def render_leading_indicators(daily_df: pd.DataFrame) -> None:
    """Render the full Leading Indicators tab.

    Parameters
    ----------
    daily_df : the DataFrame returned by _fetch_daily_raw() — daily bars with
               BTC OHLCV, macro columns, and optionally 'coinbase_close'.
    """
    st.markdown("## 📡 Bitcoin Leading Indicators")
    st.caption(
        "Correlation analysis and next-day / next-week ML forecasts built on "
        "10 on-chain and technical indicators. "
        "Metrics sourced from free public APIs; on-chain proxies are noted where "
        "the official metric requires paid data (Glassnode/CryptoQuant)."
    )

    if daily_df is None or daily_df.empty:
        st.error("Daily price data unavailable — cannot build leading indicators.")
        return

    close = daily_df["btc_close"].dropna()
    current_price = float(close.iloc[-1])

    # ── Fetch indicators ─────────────────────────────────────────────────────
    with st.spinner("Fetching leading indicators …"):
        daily_json = daily_df.to_json(orient="split", date_format="iso")
        ind = fetch_all_leading_indicators(daily_json)

    # ── Live Deribit GEX (not cached with daily data — fetched fresh) ────────
    with st.spinner("Fetching live Deribit options data …"):
        live_gex, live_spot = _fetch_deribit_gex()

    if live_gex is not None:
        ind["dealer_gex"] = np.nan
        if not ind.empty:
            ind.iloc[-1, ind.columns.get_loc("dealer_gex")] = live_gex

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 1 — Current indicator snapshot (metric cards)
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### Current Indicator Snapshot")
    st.caption(
        f"Based on latest data · BTC reference price: **${current_price:,.0f}**"
    )

    def _signal_icon(key: str, val: float | None) -> str:
        if val is None or np.isnan(val):
            return "⚪"
        meta = INDICATORS[key]
        bw = meta["bullish_when"]
        if bw == "price_above":
            return "🟢" if current_price > val else "🔴"
        elif bw == "positive":
            return "🟢" if val > 0 else "🔴"
        elif bw == "negative":
            return "🟢" if val < 0 else "🔴"
        elif bw == "low":
            return "🟢" if val < 1.0 else "🔴"
        return "⚪"

    def _fmt_val(key: str, val: float | None) -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        unit = INDICATORS[key]["unit"]
        if unit == "$":
            return f"${val:,.0f}"
        elif unit == "%":
            return f"{val:+.1f}%"
        elif unit == "$B":
            return f"{val:+.2f}B"
        elif unit == "BTC":
            return f"{val:,.0f}"
        elif unit == "×":
            return f"{val:.2f}×"
        return f"{val:.4g}"

    def _get_current(key: str) -> float | None:
        col = ind[key] if key in ind.columns else pd.Series(dtype=float)
        vals = col.dropna()
        return float(vals.iloc[-1]) if not vals.empty else None

    keys = list(INDICATORS.keys())
    row1_keys = keys[:5]
    row2_keys = keys[5:]

    for row_keys in [row1_keys, row2_keys]:
        cols = st.columns(len(row_keys))
        for col, key in zip(cols, row_keys):
            meta  = INDICATORS[key]
            val   = _get_current(key)
            icon  = _signal_icon(key, val)
            label = f"{icon} {meta['short']}"
            fval  = _fmt_val(key, val)

            # Delta vs 30 days ago
            if key in ind.columns:
                hist = ind[key].dropna()
                if len(hist) >= 30:
                    old = float(hist.iloc[-30])
                    if meta["unit"] == "$":
                        delta = f"{(val - old):+,.0f}" if val is not None else None
                    elif meta["unit"] in ("%", "×"):
                        delta = f"{(val - old):+.2f}" if val is not None else None
                    else:
                        delta = None
                else:
                    delta = None
            else:
                delta = None

            with col:
                st.metric(label=label, value=fval, delta=delta,
                          help=meta["description"])

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 2 — Individual indicator charts vs BTC price
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### Indicator Time-Series vs BTC Price")
    st.caption("All charts show the last 2 years. Dotted orange line = BTC price (right axis).")

    chart_keys = [k for k in keys if k != "dealer_gex"]  # GEX is live-only
    chart_tabs = st.tabs([INDICATORS[k]["short"] for k in chart_keys])

    dates = ind.index
    price_arr = close.reindex(ind.index).values

    for tab, key in zip(chart_tabs, chart_keys):
        with tab:
            meta = INDICATORS[key]
            if key not in ind.columns or ind[key].isna().all():
                st.info(f"No historical data available for {meta['label']}.")
                continue
            fig = _dual_axis_chart(
                dates, price_arr, ind[key].values,
                ind_label=meta["label"], ind_unit=meta["unit"],
                title=f"{meta['label']} vs BTC Price",
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("ℹ️ About this indicator"):
                st.markdown(meta["description"])

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 3 — Correlation Analysis
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### Correlation Analysis")
    st.caption(
        "Pearson correlation between each indicator's current value and "
        "BTC's forward return over the next 1d / 7d / 30d. "
        "Computed on the full available history."
    )

    fwd_ret = pd.DataFrame(index=close.index)
    fwd_ret["1d"] = (close.shift(-1) / close - 1) * 100
    fwd_ret["7d"] = (close.shift(-7) / close - 1) * 100
    fwd_ret["30d"] = (close.shift(-30) / close - 1) * 100

    corr_rows = {}
    for key in keys:
        if key == "dealer_gex":
            continue
        if key not in ind.columns:
            continue
        col_vals = ind[key].reindex(close.index)
        row = {}
        for horizon, ret_col in fwd_ret.items():
            valid = col_vals.notna() & ret_col.notna()
            if valid.sum() < 30:
                row[horizon] = np.nan
            else:
                row[horizon] = float(np.corrcoef(col_vals[valid], ret_col[valid])[0, 1])
        corr_rows[INDICATORS[key]["short"]] = row

    if corr_rows:
        corr_df = pd.DataFrame(corr_rows).T
        corr_df.columns = ["→ 1d Return", "→ 7d Return", "→ 30d Return"]
        corr_df = corr_df.sort_values("→ 1d Return", ascending=False)

        col_heat, col_bar = st.columns([1.4, 1])
        with col_heat:
            st.plotly_chart(_correlation_heatmap(corr_df),
                            use_container_width=True)

        with col_bar:
            st.markdown("**Correlations — 1d Forward Return (ranked)**")
            for idx_name, row in corr_df.iterrows():
                r = row["→ 1d Return"]
                if np.isnan(r):
                    continue
                bar_color = BULL_GREEN if r > 0 else BEAR_RED
                pct = int(abs(r) * 100)
                sign = "+" if r > 0 else "−"
                st.markdown(
                    f"<div style='margin:2px 0'>"
                    f"<span style='display:inline-block;width:130px;font-size:12px'>{idx_name}</span>"
                    f"<span style='display:inline-block;background:{bar_color};"
                    f"width:{max(pct,2)}px;height:12px;border-radius:2px;vertical-align:middle'></span>"
                    f" <span style='font-size:12px;color:{bar_color}'>{sign}{abs(r):.3f}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # Rolling 90-day correlation
        st.markdown("#### Rolling 90-Day Correlation (indicator → next-week BTC return)")
        selected_ind = st.selectbox(
            "Select indicator",
            [k for k in keys if k != "dealer_gex" and k in ind.columns],
            format_func=lambda k: INDICATORS[k]["label"],
            key="li_roll_corr_sel",
        )
        if selected_ind:
            ret7 = fwd_ret["7d"].reindex(ind.index)
            ind_col = ind[selected_ind].reindex(ind.index)
            roll_corr = []
            roll_dates = []
            window = 90
            for i in range(window, len(ind.index)):
                sl_i = ind_col.iloc[i - window:i]
                sl_r = ret7.iloc[i - window:i]
                valid = sl_i.notna() & sl_r.notna()
                if valid.sum() >= 30:
                    r = float(np.corrcoef(sl_i[valid], sl_r[valid])[0, 1])
                    roll_corr.append(r)
                    roll_dates.append(ind.index[i])
            if roll_dates:
                fig_rc = go.Figure()
                fig_rc.add_hline(y=0, line=dict(color=NEUTRAL, dash="dot"))
                fig_rc.add_trace(go.Scatter(
                    x=roll_dates, y=roll_corr,
                    fill="tozeroy",
                    line=dict(color="#6366f1", width=1.5),
                    name="90d rolling corr",
                ))
                _rc_layout = _base_layout(
                    title=f"Rolling 90d Correlation: {INDICATORS[selected_ind]['label']} → 7d BTC Return"
                )
                _rc_layout.update(yaxis=dict(range=[-1, 1], showgrid=True, gridcolor=GRID_CLR))
                fig_rc.update_layout(**_rc_layout)
                st.plotly_chart(fig_rc, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 4 — Predictive Model
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### ML Price Forecast")

    if not _SKLEARN:
        st.warning("scikit-learn not installed — ML forecast unavailable.")
        return

    with st.spinner("Training / loading forecast models …"):
        bundle = build_forecast_models(close, ind)

    if bundle is None:
        st.warning("Insufficient data to train the forecast model.")
        return

    pred_1d, pred_7d, feat_row = current_forecast(close, ind, bundle, live_gex)
    latest_date = close.index[-1]

    # ── Current forecast cards ────────────────────────────────────────────
    st.markdown("#### Current Prediction")
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        st.metric("BTC Now", f"${current_price:,.0f}")
    with fc2:
        if pred_1d is not None:
            p1_price = current_price * (1 + pred_1d / 100)
            icon = "🟢" if pred_1d > 0 else "🔴"
            st.metric(
                f"{icon} Predicted Tomorrow",
                f"${p1_price:,.0f}",
                delta=f"{pred_1d:+.2f}%",
                help="GBM model prediction based on current leading indicator readings",
            )
        else:
            st.metric("Predicted Tomorrow", "N/A")
    with fc3:
        if pred_7d is not None:
            p7_price = current_price * (1 + pred_7d / 100)
            icon = "🟢" if pred_7d > 0 else "🔴"
            st.metric(
                f"{icon} Predicted in 7 Days",
                f"${p7_price:,.0f}",
                delta=f"{pred_7d:+.2f}%",
                help="GBM model prediction based on current leading indicator readings",
            )
        else:
            st.metric("Predicted in 7 Days", "N/A")

    # Live GEX display
    if live_gex is not None:
        gex_icon = "🟢" if live_gex > 0 else "🔴"
        gex_label = "Positive (stabilizing)" if live_gex > 0 else "Negative (amplifying)"
        st.info(
            f"{gex_icon} **Live Dealer GEX (Deribit):** {live_gex:,.0f} BTC — {gex_label}. "
            f"{'Dealers are net-long gamma; expect range-bound price action.' if live_gex > 0 else 'Dealers are net-short gamma; expect amplified moves in either direction.'}"
        )
    else:
        st.caption("⚪ Live Deribit GEX unavailable (API timeout or no active options data).")

    # ── Model accuracy metrics ────────────────────────────────────────────
    m1d = bundle["test_metrics_1d"]
    m7d = bundle["test_metrics_7d"]
    st.markdown("#### Model Accuracy (Out-of-Sample Test Set)")
    st.caption(
        f"Trained on data up to ~{bundle['test_index'][0].date()} · "
        f"Tested on the last {len(bundle['test_index'])} trading days"
    )
    ma1, ma2, ma3, ma4 = st.columns(4)
    ma1.metric("1d Direction Accuracy", f"{m1d['dir_acc']:.1f}%",
               help="% of days where model correctly predicted up or down")
    ma2.metric("1d MAE", f"{m1d['mae']:.2f}%",
               help="Mean absolute error of 1-day return prediction")
    ma3.metric("7d Direction Accuracy", f"{m7d['dir_acc']:.1f}%")
    ma4.metric("7d MAE", f"{m7d['mae']:.2f}%")

    st.info(
        "⚠️ **Honest framing:** Direction accuracy of 55–65% is typical for "
        "leading-indicator models on daily BTC returns. The model's value is "
        "in aggregating regime signals, not in precise price-level prediction."
    )

    # ── Backtest charts ───────────────────────────────────────────────────
    tab_1d, tab_7d = st.tabs(["📅 1-Day Forecast Backtest", "📅 7-Day Forecast Backtest"])

    with tab_1d:
        c_ts, c_sc = st.columns(2)
        with c_ts:
            fig = _time_series_predictions(
                bundle["test_index"], bundle["y1_test"], bundle["y1_pred"], "1-Day"
            )
            st.plotly_chart(fig, use_container_width=True)
        with c_sc:
            fig = _backtest_scatter(
                bundle["y1_test"], bundle["y1_pred"], "1-Day", bundle["test_index"]
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab_7d:
        c_ts, c_sc = st.columns(2)
        with c_ts:
            fig = _time_series_predictions(
                bundle["test_index"], bundle["y7_test"], bundle["y7_pred"], "7-Day"
            )
            st.plotly_chart(fig, use_container_width=True)
        with c_sc:
            fig = _backtest_scatter(
                bundle["y7_test"], bundle["y7_pred"], "7-Day", bundle["test_index"]
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Feature importance ────────────────────────────────────────────────
    st.markdown("#### Feature Importance — 1-Day Model")
    st.caption(
        "Which indicators contributed most to the GBM model's 1-day predictions."
    )
    fig_imp = _feature_importance_chart(bundle["model_1d"], bundle["feat_cols"])
    st.plotly_chart(fig_imp, use_container_width=True)

    # ── Current feature values ────────────────────────────────────────────
    with st.expander("🔬 Current feature values fed into the model"):
        if feat_row:
            feat_display = pd.DataFrame(
                [(k, f"{v:+.4f}") for k, v in feat_row.items()],
                columns=["Feature", "Current Value"],
            ).set_index("Feature")
            st.dataframe(feat_display, use_container_width=True)
        else:
            st.info("No feature data available for the latest date.")

    # ── Data source notes ─────────────────────────────────────────────────
    with st.expander("📚 Data sources & methodology"):
        st.markdown(
            """
| Indicator | Source | Notes |
|---|---|---|
| STH Cost Basis | Price-derived | 155-day rolling avg (official: Glassnode UTXO data) |
| True Market Mean | Price-derived | (365d MA + 730d MA) / 2 (official: Glassnode balanced price) |
| 200-Day MA | Price-derived | Exact |
| Dealer GEX | Deribit public API | Live only; B-S gamma × OI per strike |
| 200-Week MA | Price-derived | Exact |
| Coinbase Premium | Coinbase + Binance APIs | Daily OHLC close difference |
| 30D Net Flow Proxy | blockchain.info | Tx-volume momentum vs 90d baseline |
| Stablecoin Supply Δ | CoinGecko free API | USDT + USDC combined market cap |
| Sell-Side Risk Proxy | Price-derived | 30d / 365d volume ratio (official: SSRR via Glassnode) |
| 180D Price Change | Price-derived | Exact |

**Model:** Gradient Boosting Regressor (sklearn) trained on price-ratio features.
All features are normalized to BTC price ratios to remove trend co-integration bias.
Train/test split: chronological, last 180 days held out.
            """
        )
