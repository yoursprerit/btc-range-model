"""Secondary daily-price feed — Nasdaq's public quote API — used when Yahoo is
unavailable or stale.

WHY THIS EXISTS
---------------
Every app fetches daily bars from Yahoo (``ticker_core._chart`` /
``gldm_core._chart``).  Yahoo's *daily* series lags its own intraday series and,
on heavily-shared egress IPs (Streamlit Community Cloud, GitHub Actions), the
responses are rate-limited or served from a stale cache for hours.  Because
``_chart`` drops null-close rows, the whole universe then sits one session behind
and the Overall cockpit raises its flashing STALE-SIGNALS alert — observed
2026-07-25, when nine of ten signal apps were stuck on Jul 23 while Jul 24 had
long since closed, both in the Streamlit UI and in the publish workflow.

Nasdaq's ``api.nasdaq.com/api/quote/<SYM>/historical`` endpoint needs no API key
and is a genuinely independent provider, so it recovers exactly the sessions
Yahoo is withholding.  Verified against Yahoo on 2026-07-25: identical OHLC for
SOXX/XLE/GRID/GLDM/GDX/WGMI (e.g. SOXX Jul-24 close 527.01, open 544.36, high
548.00, low 522.21 on both).

WHY YAHOO STAYS PRIMARY
-----------------------
This is deliberately a *fallback*, not a replacement:

1. **Coverage.**  Nasdaq serves listed equities and ETFs only.  The macro columns
   every divergence/sentiment feature needs — ``^GSPC``, ``^NDX``, ``^VIX``,
   ``GC=F``, ``DX-Y.NYB``, ``^TNX`` — are not available, so Yahoo is required no
   matter what.
2. **Depth.**  Its history stops ~10 years back (SOXX: 2016-07-25), shallower
   than the configs' fetch windows and the committed ~11.5-year macro CSVs.
   Promoting it would silently truncate the long back-tests.
3. **Adjustment provenance.**  Every committed artifact and headline figure is
   built on Yahoo *auto-adjusted* (split/dividend) closes.  Nasdaq's rows are
   raw, so wholesale substitution could shift historical series (e.g. across
   MSTR's Aug-2024 10-for-1 split).  Topping up only the newest *completed*
   sessions avoids that, and the caller's split-scale guard rejects a row whose
   level disagrees with the existing series.
4. **Politeness.**  It is an undocumented endpoint with unpublished rate limits;
   used as a narrow top-up it is hit rarely, not on every render.

So the resolution order per app is: Yahoo daily → Yahoo hourly (rebuild the
session) → **Nasdaq daily (this module)** → committed CSV snapshot.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request

import pandas as pd

_HOST = "https://api.nasdaq.com/api/quote"
# The endpoint 403s a bare urllib UA; it wants a browser-ish one.
_UA = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "application/json",
}
# Most of this universe is ETFs; a wrong assetclass returns no rows rather than an
# error, so try the likely one first and fall through.
_ASSET_CLASSES = ("etf", "stocks")
_TIMEOUT = 20

# Symbols this provider cannot serve — indices, futures and FX quotes.  Skipped
# without a request so a fallback sweep costs nothing for them.
def supports(symbol: str) -> bool:
    """True only for plain US listed tickers (Nasdaq has no indices/futures/FX)."""
    s = (symbol or "").strip().upper()
    if not s or any(ch in s for ch in "^=."):
        return False
    return s.replace("-", "").isalnum() and "-" not in s


def _num(v) -> float:
    """Parse Nasdaq's display numbers: '$91.67', '10,952,030', 'N/A'."""
    if v is None:
        return float("nan")
    t = str(v).replace("$", "").replace(",", "").strip()
    if not t or t.upper() in ("N/A", "--"):
        return float("nan")
    try:
        return float(t)
    except ValueError:
        return float("nan")


_cache: dict[tuple, pd.DataFrame] = {}


def daily_ohlcv(symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Daily OHLCV for one US-listed ``symbol`` from Nasdaq, tz-naive DatetimeIndex.

    Returns an EMPTY frame on any problem (unsupported symbol, HTTP/JSON error,
    no rows) so every caller can treat it as "nothing to add" and keep whatever
    it already had.  Results are memoised per (symbol, start, end) for the life of
    the process — a fallback sweep touching the same symbol twice costs one call.
    """
    if not supports(symbol):
        return pd.DataFrame()
    end = end or pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    key = (symbol.upper(), str(start), str(end))
    if key in _cache:
        return _cache[key]
    out = pd.DataFrame()
    for cls in _ASSET_CLASSES:
        q = urllib.parse.urlencode({"assetclass": cls, "fromdate": str(start),
                                    "todate": str(end), "limit": "9999"})
        try:
            req = urllib.request.Request(f"{_HOST}/{symbol.upper()}/historical?{q}",
                                         headers=_UA)
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as fh:
                payload = json.load(fh)
            rows = (((payload.get("data") or {}).get("tradesTable") or {})
                    .get("rows")) or []
            if not rows:
                continue
            df = pd.DataFrame({
                "open":   [_num(r.get("open"))   for r in rows],
                "high":   [_num(r.get("high"))   for r in rows],
                "low":    [_num(r.get("low"))    for r in rows],
                "close":  [_num(r.get("close"))  for r in rows],
                "volume": [_num(r.get("volume")) for r in rows],
            }, index=pd.to_datetime([r.get("date") for r in rows],
                                    format="%m/%d/%Y", errors="coerce"))
            df = df[df.index.notna()].sort_index()
            df = df[~df.index.duplicated(keep="last")].dropna(subset=["close"])
            if not df.empty:
                out = df
                break
        except Exception:
            continue
    _cache[key] = out
    return out


def merge_frame(symbol_map: dict, start: str, end: str | None = None) -> pd.DataFrame:
    """``{name: yahoo_symbol}`` → one column-merged frame of ``name_open`` …
    ``name_volume``, mirroring ``ticker_core._merge`` so the result can be handed
    straight to the session-backfill helper.  Unsupported symbols are skipped."""
    frames = []
    for name, sym in (symbol_map or {}).items():
        d = daily_ohlcv(sym, start, end)
        if not d.empty:
            frames.append(d.add_prefix(f"{name}_"))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()


# ════════════════════════════════════════════════════════════════════════════
# LIVE QUOTES — backup for the action plan's spot prices
# ════════════════════════════════════════════════════════════════════════════
# ``overall_core._quote`` reads each instrument's live price from Yahoo's chart
# meta.  That single feed is the same one this module already backstops for
# daily bars, and it fails the same ways: rate-limited or 403'd on shared egress
# IPs, geo-blocked, or simply returning a null ``regularMarketPrice``.  When it
# does, every row of 🎯 Today's action plan falls back to its last completed bar
# close — Live Price freezes, Chg % reads "—", and the live entry/exit flags and
# the Target % / $ (Live) columns have nothing to react to.  The cockpit is then
# blind exactly when the market is moving.
#
# So each quote has a provider chain, tried in order until one returns a usable
# price.  Every provider here is keyless, public, and independent of Yahoo:
#
#   US listed equities & ETFs   Yahoo → Nasdaq (``/api/quote/<SYM>/info``)
#   spot crypto (BTC-USD, …)    Yahoo → Coinbase Exchange → Binance
#
# Each provider costs ONE request and returns both halves of a quote (the live
# price and the previous close) from that one response, so a failover never
# multiplies the request count.  The caller records which source served each
# quote (``fetch_spot`` → ``src``) and the cockpit says so on screen — a backup
# price is never passed off as the primary feed's.
#
# What the fallbacks are NOT used for: signals, bars, or anything persisted.
# They mark the live column only, exactly like the Yahoo quote they replace.
_CRYPTO_PRODUCTS = {          # spot symbol → (Coinbase product, Binance pair)
    "BTC-USD": ("BTC-USD", "BTCUSDT"),
    "ETH-USD": ("ETH-USD", "ETHUSDT"),
}
_BINANCE_HOSTS = ("https://api.binance.us", "https://api.binance.com")


def _get_json(url: str, headers: dict | None = None, timeout: int = _TIMEOUT):
    """GET → parsed JSON, or ``None`` on any HTTP/JSON/network failure."""
    try:
        req = urllib.request.Request(url, headers=headers or {"User-Agent": _UA["User-Agent"]})
        with urllib.request.urlopen(req, timeout=timeout) as fh:
            return json.load(fh)
    except Exception:
        return None


def _pair(px, prev) -> tuple:
    """(price, previous close) with unusable values normalised to ``None``."""
    def _ok(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f if f == f and f > 0 else None      # finite and positive
    return _ok(px), _ok(prev)


def nasdaq_quote(symbol: str) -> tuple:
    """Live (price, previous close) for a US listed symbol from Nasdaq.

    ``primaryData`` is the REGULAR-session quote — the same thing Yahoo's
    ``regularMarketPrice`` is — so the two are interchangeable; the extended
    hours print (``secondaryData``) is deliberately ignored.  The previous close
    is derived as ``lastSalePrice − netChange``, which is what that percentage
    change is quoted against."""
    if not supports(symbol):
        return None, None
    for cls in _ASSET_CLASSES:
        j = _get_json(f"{_HOST}/{symbol.upper()}/info?"
                      + urllib.parse.urlencode({"assetclass": cls}))
        d = ((j or {}).get("data") or {}).get("primaryData") or {}
        px = _num(d.get("lastSalePrice"))
        if px != px or px <= 0:                     # NaN / absent → try next class
            continue
        chg = _num(d.get("netChange"))
        return _pair(px, (px - chg) if chg == chg else None)
    return None, None


def coinbase_quote(symbol: str) -> tuple:
    """Live (price, 24 h-ago price) for spot crypto from Coinbase Exchange.

    USD-denominated, so it needs no stablecoin basis adjustment.  ``open`` is
    the rolling 24-hour open rather than the previous UTC day's close — close
    enough for the day-change readouts a backup feed serves, and the action
    plan's Chg % does not use it at all (it measures against the strategy's own
    last bar close)."""
    prod = (_CRYPTO_PRODUCTS.get(symbol.upper()) or (None, None))[0]
    if not prod:
        return None, None
    j = _get_json(f"https://api.exchange.coinbase.com/products/{prod}/stats")
    return _pair((j or {}).get("last"), (j or {}).get("open"))


def binance_quote(symbol: str) -> tuple:
    """Live (price, previous close) for spot crypto from Binance's 24 h ticker.

    Quoted in USDT, not USD — a basis of a few basis points against the Yahoo /
    Coinbase USD print. Last in the crypto chain for that reason, and because
    ``api.binance.com`` answers 451 from US-hosted infrastructure (so the ``.us``
    host is tried first, as in the BTC app)."""
    pair = (_CRYPTO_PRODUCTS.get(symbol.upper()) or (None, None))[1]
    if not pair:
        return None, None
    for host in _BINANCE_HOSTS:
        j = _get_json(f"{host}/api/v3/ticker/24hr?symbol={pair}")
        if j and j.get("lastPrice"):
            return _pair(j.get("lastPrice"), j.get("prevClosePrice"))
    return None, None


# provider chains, in the order they are tried, per instrument type
_CHAINS = (
    ("nasdaq", nasdaq_quote),
    ("coinbase", coinbase_quote),
    ("binance", binance_quote),
)


def live_quote(symbol: str) -> tuple:
    """Backup live quote: ``(price, previous close, source)`` from the first
    provider that serves ``symbol``, or ``(None, None, None)``.

    Only the price is guaranteed — a provider that gives no previous close
    still counts as a served quote, since the action plan measures its change
    against the strategy's last bar, not against the feed's previous session."""
    for name, fn in _CHAINS:
        try:
            px, prev = fn(symbol)
        except Exception:
            continue
        if px:
            return px, prev, name
    return None, None, None
