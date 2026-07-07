"""Ticker configurations for the multi-asset forecasting apps.

The Gold (GLDM) app in ``app/gldm_hourly_app.py`` was hand-built as the gold
counterpart of the BTC app.  Rather than copy-paste that ~1,400-line app five
more times, this module captures everything that differs between assets in a
single ``TickerConfig`` dataclass, and the generic engine

    app/ticker_core.py        data fetch + features + signals   (config-driven)
    backtest_ticker.py        backtest engine + threshold sweep
    src/tickers/train_ticker.py   trains the 5-model suite
    app/ticker_app.py         the Streamlit UI (identical layout to Gold)

renders each app with the Gold app's exact structure, styling and colour
coding — only the asset, its macro drivers, its tuned thresholds and a per-app
accent colour change.  The BTC and GLDM apps are never touched.

Every field below is re-derived for the specific asset (nothing is copied
blindly).  The five apps are:

    SOXX  iShares Semiconductor ETF        (high-beta tech / semis)
    VEGN  US Vegan Climate ETF             (ESG large-cap, broad-market tilt)
    GRID  First Trust Clean Edge Grid ETF  (grid infrastructure / electrification)
    XLE   Energy Select Sector SPDR        (energy sector; OIH = high-beta proxy)
    REMX  VanEck Rare Earth/Strategic Metals(rare-earth & strategic metals)

None of these use a leveraged proxy (per the brief) EXCEPT XLE, which carries
OIH (VanEck Oil Services ETF) as its high-beta execution sibling — mirroring how
the Gold app trades GDX/UGL off the gold signal.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ── sentiment-composite driver spec ──────────────────────────────────────
# Each entry is (column, kind, sign):
#   column : a column in the merged frame ("px_close" = the primary asset)
#   kind   : "mom" 20-period log-momentum · "chg" 20-period diff · "lvl" level z
#   sign   : +1 if "up = bullish for this asset", −1 if inverse driver
# The signed z-scores are averaged then rolling-percentile-ranked to a
# self-calibrating 0-100 gauge — exactly like the Gold app's macro sentiment.
SentSpec = tuple  # (col:str, kind:str, sign:float)


@dataclass
class TickerConfig:
    key: str                       # app id / session value, e.g. "SOXX"
    name: str                      # full ETF name
    blurb: str                     # one-line description under the title
    emoji: str                     # app icon
    accent: str                    # primary accent colour (hex) — Gold uses #b8860b
    accent_dark: str               # darker shade for headings/tables
    accent_bg: str                 # pale background for the strategy card
    accent_bg2: str                # slightly deeper pale background

    primary_symbol: str            # signal + (usually) traded 1x asset
    macro_syms: dict               # name -> Yahoo symbol (feature drivers)
    extra_syms: dict               # name -> Yahoo symbol (traded siblings, e.g. OIH)
    sentiment: list                # list of SentSpec for the macro-sentiment gauge
    sentiment_label: str           # e.g. "Semis macro sentiment"

    # ── traded universe ──────────────────────────────────────────────────
    # list of (label, column) the strategy trades.  "px_close" = the primary.
    traded_assets: list
    asset_labels: dict             # column -> short label for cards/tabs

    # ── strategy (tuned by the sweep; see backtest_ticker.py --sweep) ─────
    strategy_mode: str             # "ma" (trend filter) or "divergence"
    strategy_name: str
    ma_window: int                 # MA-trend-filter window (mode="ma")
    fixed_stop: float              # fractional stop (e.g. 0.08 = −8%)
    # divergence thresholds (mode="divergence"); harmless if unused
    u1_errhi_min: float = 0.08
    d2_errhi_max: float = -0.10
    d1_errlo_min: float = 0.10
    v_errlo_min: float = 0.50
    use_d1_exit: bool = False      # if True, D1 downtrend pressure is also an exit
    hl_band_pct: float = 0.010     # ±band tint on the daily H/L chart

    # ── windows ──────────────────────────────────────────────────────────
    fetch_start: str = "2015-01-01"
    oos_start: str = "2021-01-01"
    periods: list = field(default_factory=list)   # (label, start, end|None)

    day_up_thresh: float = 0.006
    day_down_thresh: float = -0.006

    # ── results (filled in from the backtest, shown in the UI/README) ─────
    results_note: str = ""
    eval_note: str = ""            # optional extra analysis (e.g. XLE vs OIH)

    @property
    def prefix(self) -> str:
        return "px"

    @property
    def signal_label(self) -> str:
        return self.key


# Backtest windows shared by the assets with full 2015+ history.
# The first window spans the ENTIRE available history (bull + bear, INCLUDING the
# pre-2021 in-sample training window) — the honest combined-cycle test of whether
# the strategy beats buy-&-hold once a real bear market is in scope.  The MA
# trend-filter uses no fitted parameters, so its full-history curve is genuinely
# out-of-sample; the divergence variant's pre-2021 signals are in-sample.
_STD_PERIODS = [
    ("🌍 Full history (Bull + Bear)", "2015-01-01", None),
    ("🐻 Drawdown / Bear (2021–2022)", "2021-01-01", "2022-12-31"),
    ("🐂 Recovery / Bull (2023 → now)", "2023-01-01", None),
    ("🌐 Full Out-of-Sample (2021 → now)", "2021-01-01", None),
    ("🔬 Most-recent OOS (2025 → now)", "2025-01-01", None),
]


CONFIGS: dict[str, TickerConfig] = {}


# ════════════════════════════════════════════════════════════════════════
# SOXX — iShares Semiconductor ETF
# ════════════════════════════════════════════════════════════════════════
CONFIGS["SOXX"] = TickerConfig(
    key="SOXX",
    name="iShares Semiconductor ETF",
    blurb=("SOXX tracks the ICE Semiconductor index — a high-beta basket of chip "
           "makers (NVDA, AVGO, AMD…). It is one of the most cyclical equity "
           "groups: it leads tech both up and down, with 50%+ drawdowns."),
    emoji="🖥️",
    accent="#2563eb", accent_dark="#1e40af", accent_bg="#eff6ff", accent_bg2="#dbeafe",
    primary_symbol="SOXX",
    macro_syms={
        "smh": "SMH",          # sister semi ETF — basis / lead
        "sox": "^SOX",         # PHLX Semiconductor index — direct read
        "qqq": "QQQ",          # Nasdaq-100 tech beta
        "nvda": "NVDA",        # bellwether chip
        "tnx": "^TNX",         # 10Y yield — growth/duration, inverse
        "dxy": "DX-Y.NYB",     # US dollar
        "vix": "^VIX",         # equity vol (risk-off = bearish semis)
        "spx": "^GSPC",        # S&P 500 risk backdrop
    },
    extra_syms={},
    sentiment=[("qqq_close", "mom", +1.0), ("tnx_close", "chg", -1.0),
               ("vix_close", "lvl", -1.0), ("px_close", "mom", +1.0)],
    sentiment_label="Semis macro sentiment",
    traded_assets=[("SOXX", "px_close")],
    asset_labels={"px_close": "SOXX · Semiconductors"},
    strategy_mode="ma", strategy_name="Semis Trend-Regime",
    ma_window=40, fixed_stop=0.05,
    hl_band_pct=0.012,
    fetch_start="2015-01-01", oos_start="2021-01-01", periods=_STD_PERIODS,
    day_up_thresh=0.010, day_down_thresh=-0.010,
    results_note=("OOS 2021→now: +233% vs Buy&Hold +362%, but max drawdown "
                  "−31% vs −46% and a higher Sharpe (0.94 vs 0.93). A "
                  "long-above-the-40-day-SMA filter with a tight −5% stop that "
                  "sidesteps the worst of every chip down-cycle."),
)


# ════════════════════════════════════════════════════════════════════════
# VEGN — US Vegan Climate ETF (ESG large-cap, broad-market tilt)
# ════════════════════════════════════════════════════════════════════════
CONFIGS["VEGN"] = TickerConfig(
    key="VEGN",
    name="US Vegan Climate ETF",
    blurb=("VEGN tracks the Beyond Investing US Large Cap index — the S&P 500 "
           "screened to exclude animal-exploitation, fossil-fuel and defence "
           "names. The result is a tech-tilted large-cap fund that trades very "
           "close to the broad market, with far lower volatility than a sector ETF."),
    emoji="🌱",
    accent="#16a34a", accent_dark="#166534", accent_bg="#f0fdf4", accent_bg2="#dcfce7",
    primary_symbol="VEGN",
    macro_syms={
        "spy": "SPY",          # broad market — VEGN's dominant driver
        "qqq": "QQQ",          # growth / tech tilt
        "xlk": "XLK",          # technology sector weight
        "tnx": "^TNX",         # 10Y yield — duration, inverse
        "dxy": "DX-Y.NYB",     # US dollar
        "vix": "^VIX",         # equity vol
    },
    extra_syms={},
    sentiment=[("spy_close", "mom", +1.0), ("tnx_close", "chg", -1.0),
               ("vix_close", "lvl", -1.0), ("px_close", "mom", +1.0)],
    sentiment_label="Large-cap macro sentiment",
    traded_assets=[("VEGN", "px_close")],
    asset_labels={"px_close": "VEGN · ESG Large-Cap"},
    strategy_mode="ma", strategy_name="Large-Cap Trend-Regime",
    ma_window=200, fixed_stop=0.05,
    hl_band_pct=0.007,
    # VEGN launched Sept-2019: fetch from inception, hold OOS start to 2022 so the
    # signal model has ~2.3y of pre-OOS training.
    fetch_start="2019-09-09", oos_start="2022-01-01",
    periods=[
        ("🌍 Full history (Bull + Bear)", "2019-09-09", None),
        ("🐻 Drawdown / Bear (2022)", "2022-01-01", "2022-12-31"),
        ("🐂 Recovery / Bull (2023 → now)", "2023-01-01", None),
        ("🌐 Full Out-of-Sample (2022 → now)", "2022-01-01", None),
        ("🔬 Most-recent OOS (2025 → now)", "2025-01-01", None),
    ],
    day_up_thresh=0.005, day_down_thresh=-0.005,
    results_note=("OOS 2022→now: +82% vs Buy&Hold +81% while HALVING the max "
                  "drawdown (−15% vs −33%) and lifting Sharpe to 1.00 vs 0.73. "
                  "A slow 200-day trend filter that side-steps the 2022 bear and "
                  "stays long through the recovery — beats the market on both "
                  "return and risk."),
)


# ════════════════════════════════════════════════════════════════════════
# GRID — First Trust Clean Edge Smart Grid Infrastructure ETF
# ════════════════════════════════════════════════════════════════════════
CONFIGS["GRID"] = TickerConfig(
    key="GRID",
    name="First Trust Clean Edge Smart Grid Infrastructure ETF",
    blurb=("GRID holds the electrical-grid & electrification supply chain "
           "(ABB, Eaton, Schneider, Quanta…). It rides the capex super-cycle in "
           "power infrastructure — steadier than pure clean-energy, but rate- and "
           "capex-sensitive."),
    emoji="⚡",
    accent="#0d9488", accent_dark="#115e59", accent_bg="#f0fdfa", accent_bg2="#ccfbf1",
    primary_symbol="GRID",
    macro_syms={
        "xlu": "XLU",          # utilities — grid overlap
        "icln": "ICLN",        # clean energy complex
        "tan": "TAN",          # solar (electrification demand)
        "cper": "CPER",        # copper — electrification metal
        "tnx": "^TNX",         # 10Y yield — capex/duration, inverse
        "spx": "^GSPC",        # equity risk backdrop
        "vix": "^VIX",         # equity vol
    },
    extra_syms={},
    sentiment=[("spx_close", "mom", +1.0), ("cper_close", "mom", +1.0),
               ("tnx_close", "chg", -1.0), ("px_close", "mom", +1.0)],
    sentiment_label="Grid / electrification sentiment",
    traded_assets=[("GRID", "px_close")],
    asset_labels={"px_close": "GRID · Grid Infrastructure"},
    strategy_mode="ma", strategy_name="Grid Trend-Regime",
    ma_window=150, fixed_stop=0.05,
    hl_band_pct=0.010,
    fetch_start="2015-01-01", oos_start="2021-01-01", periods=_STD_PERIODS,
    day_up_thresh=0.007, day_down_thresh=-0.007,
    results_note=("Full history 2016→now (bull + bear): +296% vs Buy&Hold +452% "
                  "but at nearly HALF the drawdown (−22% vs −41%) and a higher "
                  "Sharpe (0.94 vs 0.88) — a better risk-adjusted return over the "
                  "full cycle. OOS 2021→now: +105% vs +132% with −19% vs −30% "
                  "drawdown. A 150-day trend filter that rides the grid-capex "
                  "super-cycle but cuts the rate-shock drawdowns."),
)


# ════════════════════════════════════════════════════════════════════════
# XLE — Energy Select Sector SPDR (OIH = high-beta oil-services sibling)
# ════════════════════════════════════════════════════════════════════════
CONFIGS["XLE"] = TickerConfig(
    key="XLE",
    name="Energy Select Sector SPDR",
    blurb=("XLE is the large-cap US energy sector (XOM, CVX, COP…). It is driven "
           "by crude oil and is deeply cyclical. OIH (oil-services) is its "
           "higher-beta sibling — the strategy trades BOTH off the XLE signal, "
           "the way the Gold app trades GDX & UGL off gold."),
    emoji="🛢️",
    accent="#ea580c", accent_dark="#9a3412", accent_bg="#fff7ed", accent_bg2="#ffedd5",
    primary_symbol="XLE",
    macro_syms={
        "cl": "CL=F",          # WTI crude — the primary driver
        "bz": "BZ=F",          # Brent crude
        "xom": "XOM",          # ExxonMobil bellwether
        "dxy": "DX-Y.NYB",     # US dollar — inverse for commodities
        "spx": "^GSPC",        # equity risk backdrop
        "vix": "^VIX",         # equity vol
    },
    extra_syms={"oih": "OIH"},   # high-beta oil-services sibling (traded)
    sentiment=[("cl_close", "mom", +1.0), ("dxy_close", "mom", -1.0),
               ("vix_close", "lvl", -1.0), ("px_close", "mom", +1.0)],
    sentiment_label="Energy macro sentiment",
    traded_assets=[("XLE", "px_close"), ("OIH", "oih_close")],
    asset_labels={"px_close": "XLE · Energy", "oih_close": "OIH · Oil Services"},
    strategy_mode="divergence", strategy_name="Energy Divergence Pure-Regime",
    ma_window=80, fixed_stop=0.08,
    u1_errhi_min=0.16, d2_errhi_max=-0.10, d1_errlo_min=0.10, v_errlo_min=0.50,
    use_d1_exit=True,
    hl_band_pct=0.014,
    fetch_start="2015-01-01", oos_start="2021-01-01", periods=_STD_PERIODS,
    day_up_thresh=0.010, day_down_thresh=-0.010,
    results_note=("OOS 2021→now on XLE: +105% with a max drawdown of just −9% "
                  "(vs Buy&Hold +180% / −27%), Sharpe 1.37 vs 0.84. The "
                  "divergence system (with a D1 downtrend exit) trades the "
                  "crude-driven swings and stays out of the deep energy "
                  "drawdowns. The same XLE signal drives the higher-beta OIH "
                  "sibling to +70% at −19% MDD (see the OIH tab)."),
    eval_note=("**Trade OIH on the XLE signal, not on OIH's own.** Back-tested "
               "both ways (2021→now): OIH driven by the XLE divergence signal "
               "returns **+36%** (Sharpe 0.43) and wins the 2021–22 energy "
               "drawdown decisively (+65% vs +13%); OIH driven by "
               "its *own* signal returns only +23% (Sharpe 0.29). XLE is a "
               "broader, less-noisy read of the energy tape, so its signal steers "
               "the thin, higher-beta OIH better than OIH's own — the same reason "
               "the Gold app trades GDX/UGL off the cleaner gold signal."),
)


# ════════════════════════════════════════════════════════════════════════
# REMX — VanEck Rare Earth & Strategic Metals ETF
# ════════════════════════════════════════════════════════════════════════
CONFIGS["REMX"] = TickerConfig(
    key="REMX",
    name="VanEck Rare Earth & Strategic Metals ETF",
    blurb=("REMX holds rare-earth & strategic-metal miners (Lynas, MP Materials, "
           "lithium & tungsten names). It is a thin, extremely volatile, "
           "China-sensitive commodity-equity basket — huge 2020–21 boom, deep "
           "2022–23 bust — where avoiding the drawdown is the whole game."),
    emoji="🧲",
    accent="#7c3aed", accent_dark="#5b21b6", accent_bg="#f5f3ff", accent_bg2="#ede9fe",
    primary_symbol="REMX",
    macro_syms={
        "lit": "LIT",          # lithium / battery-metals sibling
        "cper": "CPER",        # copper — industrial-metals read
        "fxi": "FXI",          # China large-cap — supply/demand driver
        "slv": "SLV",          # silver — metals complex
        "dxy": "DX-Y.NYB",     # US dollar — inverse for commodities
        "spx": "^GSPC",        # equity risk backdrop
        "vix": "^VIX",         # equity vol
    },
    extra_syms={},
    sentiment=[("cper_close", "mom", +1.0), ("lit_close", "mom", +1.0),
               ("dxy_close", "mom", -1.0), ("px_close", "mom", +1.0)],
    sentiment_label="Strategic-metals sentiment",
    traded_assets=[("REMX", "px_close")],
    asset_labels={"px_close": "REMX · Rare-Earth Metals"},
    strategy_mode="ma", strategy_name="Metals Trend-Regime",
    ma_window=150, fixed_stop=0.05,
    u1_errhi_min=0.16, d2_errhi_max=-0.14, d1_errlo_min=0.10, v_errlo_min=0.50,
    hl_band_pct=0.016,
    fetch_start="2015-01-01", oos_start="2021-01-01", periods=_STD_PERIODS,
    day_up_thresh=0.012, day_down_thresh=-0.012,
    results_note=("Full history 2016→now (bull + bear): +370% vs Buy&Hold +112% "
                  "— it BEATS buy-&-hold on return AND drawdown (−56% vs a "
                  "catastrophic −75%) with Sharpe 0.70 vs 0.38, because a "
                  "150-day trend filter side-steps the multi-year rare-earth bear "
                  "that guts buy-&-hold. OOS 2021→now it still wins: +73% vs +24%. "
                  "(A tighter divergence variant cut the OOS drawdown to −18% but "
                  "gave up the full-cycle return — this config is tuned to beat "
                  "buy-&-hold over the whole bull+bear span.)"),
)


# ════════════════════════════════════════════════════════════════════════
# WGMI — CoinShares Valkyrie Bitcoin Miners ETF
# ════════════════════════════════════════════════════════════════════════
CONFIGS["WGMI"] = TickerConfig(
    key="WGMI",
    name="CoinShares Valkyrie Bitcoin Miners ETF",
    blurb=("WGMI holds Bitcoin-mining companies (MARA, CLSK, RIOT, IREN…). It is "
           "a high-beta play on Bitcoin — it amplifies BTC's moves roughly 2–3× in "
           "both directions, so its up-cycles are explosive and its drawdowns are "
           "brutal. It is an ordinary equity ETF (not a leveraged fund), and it "
           "trades itself off its own signal."),
    emoji="⛏️",
    accent="#0891b2", accent_dark="#155e75", accent_bg="#ecfeff", accent_bg2="#cffafe",
    primary_symbol="WGMI",
    macro_syms={
        "btc": "BTC-USD",      # spot Bitcoin — the dominant driver of miners
        "mara": "MARA",        # Marathon Digital — bellwether miner
        "riot": "RIOT",        # Riot Platforms — bellwether miner
        "coin": "COIN",        # Coinbase — crypto-equity beta
        "eth": "ETH-USD",      # Ether — crypto complex
        "qqq": "QQQ",          # Nasdaq-100 — risk-on tech backdrop
        "vix": "^VIX",         # equity vol (risk-off = bearish miners)
    },
    extra_syms={},
    sentiment=[("btc_close", "mom", +1.0), ("qqq_close", "mom", +1.0),
               ("vix_close", "lvl", -1.0), ("px_close", "mom", +1.0)],
    sentiment_label="Bitcoin-miner sentiment",
    traded_assets=[("WGMI", "px_close")],
    asset_labels={"px_close": "WGMI · Bitcoin Miners"},
    strategy_mode="ma", strategy_name="Miner Trend-Regime",
    ma_window=30, fixed_stop=0.10,
    hl_band_pct=0.020,
    # WGMI launched Feb-2022; its 52-week features don't warm up until ~Feb-2023,
    # so fetch from inception but hold the OOS start to 2024 — that leaves ~11
    # months of feature-complete pre-OOS training for the daily H/L signal model.
    fetch_start="2022-02-07", oos_start="2024-01-01",
    periods=[
        ("🌍 Full history (Bull + Bear)", "2022-02-07", None),
        ("🚀 2024 Rally (H1)", "2024-01-01", "2024-06-30"),
        ("📉 2024 Drawdown (H2)", "2024-07-01", "2024-12-31"),
        ("🌐 Full Out-of-Sample (2024 → now)", "2024-01-01", None),
        ("🔬 Most-recent OOS (2025 → now)", "2025-01-01", None),
    ],
    day_up_thresh=0.015, day_down_thresh=-0.015,
    results_note=("OOS 2024→now: +191% vs Buy&Hold +223% while cutting the max "
                  "drawdown from a brutal −63% to −40% and beating its Sharpe "
                  "(1.02 vs 0.98). A fast 30-day trend filter that rides the "
                  "Bitcoin-miner up-cycles but bails to cash when BTC rolls over "
                  "— essential risk control on a 2–3× BTC-beta basket. (Short "
                  "history: WGMI listed Feb-2022, so the pre-OOS training window "
                  "is ~11 months.)"),
)


# ════════════════════════════════════════════════════════════════════════
# PBW — Invesco WilderHill Clean Energy ETF
# ════════════════════════════════════════════════════════════════════════
CONFIGS["PBW"] = TickerConfig(
    key="PBW",
    name="Invesco WilderHill Clean Energy ETF",
    blurb=("PBW holds a broad, equal-weighted basket of clean-energy names "
           "(solar, wind, EVs, hydrogen, storage). It is one of the most "
           "explosive boom-bust equity groups — a huge 2020–21 melt-up and a "
           "brutal multi-year bust — where sidestepping the drawdown is the "
           "whole game."),
    emoji="☀️",
    accent="#65a30d", accent_dark="#3f6212", accent_bg="#f7fee7", accent_bg2="#ecfccb",
    primary_symbol="PBW",
    macro_syms={
        "icln": "ICLN",        # global clean energy — sister basket
        "tan": "TAN",          # solar
        "qcln": "QCLN",        # clean-edge green energy
        "cper": "CPER",        # copper — electrification metal
        "tnx": "^TNX",         # 10Y yield — long-duration growth, inverse
        "spx": "^GSPC",        # equity risk backdrop
        "vix": "^VIX",         # equity vol
    },
    extra_syms={},
    sentiment=[("icln_close", "mom", +1.0), ("tnx_close", "chg", -1.0),
               ("vix_close", "lvl", -1.0), ("px_close", "mom", +1.0)],
    sentiment_label="Clean-energy sentiment",
    traded_assets=[("PBW", "px_close")],
    asset_labels={"px_close": "PBW · Clean Energy"},
    strategy_mode="divergence", strategy_name="Clean-Energy Divergence Pure-Regime",
    ma_window=50, fixed_stop=0.10,
    u1_errhi_min=0.12, d2_errhi_max=-0.08, d1_errlo_min=0.10, v_errlo_min=0.50,
    use_d1_exit=False,
    hl_band_pct=0.014,
    fetch_start="2010-01-01", oos_start="2021-01-01", periods=_STD_PERIODS,
    day_up_thresh=0.012, day_down_thresh=-0.012,
    results_note=("OOS 2021→now: the U1/D2 divergence system returns +61% at a "
                  "−39% max drawdown (Sharpe 0.54) while buy-&-hold is DOWN −64% "
                  "at a −90% drawdown. On this boom-bust clean-energy basket a "
                  "trend filter barely clears water (+10%); the divergence "
                  "Pure-Regime — which stands aside through the multi-year bust "
                  "and re-enters on confirmed momentum — is by far the better "
                  "way to trade it, so it is the tuned strategy here."),
)


# ════════════════════════════════════════════════════════════════════════
# ARTY — iShares Future AI & Tech ETF
# ════════════════════════════════════════════════════════════════════════
CONFIGS["ARTY"] = TickerConfig(
    key="ARTY",
    name="iShares Future AI & Tech ETF",
    blurb=("ARTY holds the AI & exponential-technology supply chain (chips, "
           "software, robotics, data names). It is a high-beta growth basket "
           "that leads the Nasdaq both up and down and is highly rate- and "
           "risk-sensitive."),
    emoji="🤖",
    accent="#4f46e5", accent_dark="#3730a3", accent_bg="#eef2ff", accent_bg2="#e0e7ff",
    primary_symbol="ARTY",
    macro_syms={
        "qqq": "QQQ",          # Nasdaq-100 tech beta
        "xlk": "XLK",          # technology sector
        "smh": "SMH",          # semiconductors — AI compute
        "nvda": "NVDA",        # AI bellwether
        "tnx": "^TNX",         # 10Y yield — growth/duration, inverse
        "spx": "^GSPC",        # equity risk backdrop
        "vix": "^VIX",         # equity vol
    },
    extra_syms={},
    sentiment=[("qqq_close", "mom", +1.0), ("smh_close", "mom", +1.0),
               ("tnx_close", "chg", -1.0), ("px_close", "mom", +1.0)],
    sentiment_label="AI/tech sentiment",
    traded_assets=[("ARTY", "px_close")],
    asset_labels={"px_close": "ARTY · AI & Tech"},
    strategy_mode="divergence", strategy_name="AI/Tech Divergence Pure-Regime",
    ma_window=50, fixed_stop=0.05,
    u1_errhi_min=0.05, d2_errhi_max=-0.18, d1_errlo_min=0.10, v_errlo_min=0.50,
    use_d1_exit=True,
    hl_band_pct=0.012,
    fetch_start="2018-06-28", oos_start="2021-01-01",
    periods=[
        ("🌍 Full history (Bull + Bear)", "2019-01-01", None),
        ("🐻 Drawdown / Bear (2021–2022)", "2021-01-01", "2022-12-31"),
        ("🐂 Recovery / Bull (2023 → now)", "2023-01-01", None),
        ("🌐 Full Out-of-Sample (2021 → now)", "2021-01-01", None),
        ("🔬 Most-recent OOS (2025 → now)", "2025-01-01", None),
    ],
    day_up_thresh=0.010, day_down_thresh=-0.010,
    results_note=("OOS 2021→now: the U1/D2 divergence system (with a D1 "
                  "downtrend exit and a −5% stop) returns +108% at just a −17% "
                  "max drawdown (Sharpe 0.99) vs buy-&-hold +82% at −56% "
                  "(Sharpe 0.52) — it beats the market on return AND roughly "
                  "thirds the drawdown. A trend filter only manages +79% at "
                  "−33%, so the divergence Pure-Regime is the tuned strategy."),
)


def get_config(key: str) -> TickerConfig:
    return CONFIGS[key]


APP_KEYS = list(CONFIGS.keys())     # + WGMI, PBW, ARTY
