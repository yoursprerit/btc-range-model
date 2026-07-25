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
