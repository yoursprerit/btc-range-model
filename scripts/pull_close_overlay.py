"""After-close price overlay: today's official closes, the evening they print.

The pinned datasets (``data/<sleeve>/…``, the P&L snapshot) deliberately stop at
the previous session's close until the next morning's publish (see
``app/freshness.py``: post-close rows must not leak into a published book). The
P&L of what IBKR / Collective2 actually HELD needs no signals — only prices — so
this writes just the newest completed session's closes to
``data/overall/close_overlay.json`` and ``overall_core.apply_close_overlay``
adds them as one extra bar for the executed-fills replays alone.

    python scripts/pull_close_overlay.py            # after 4:10 PM ET on a trading day
    python scripts/pull_close_overlay.py --force    # skip the time guard

Writes nothing before 4:10 PM ET, on a non-trading day, or when no symbol has a
bar for the session; rewrites only when the closes changed. Yahoo first, then
Nasdaq (``market_fallback``) for a symbol Yahoo lacks. Never raises on a single
bad symbol — a partial overlay is still useful, and the missing keys are listed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "app"))
sys.path.insert(0, str(_REPO / "scripts"))

import freshness as fr                       # noqa: E402
import ibkr_symbols as sym                    # noqa: E402

OUT = _REPO / "data" / "overall" / "close_overlay.json"
SCHEMA = "close-overlay/v1"
AFTER_CLOSE = (16, 10)        # ET: the close has printed and Yahoo has finalised it


def overlay_keys() -> dict[str, str]:
    """``{key: symbol}`` for every sleeve priced as its own listed ticker. BTC /
    ETH are signalled on spot but held as IBIT / ETHA, so their price column is
    not the held ETF's close — left out (the replays carry those forward)."""
    return {k: v for k, v in sym.TRADE_SYMBOL.items() if k == v}


def session_for(now: pd.Timestamp) -> pd.Timestamp | None:
    """The US session whose close is in, or None (before 4:10 PM ET / closed)."""
    et = now.tz_convert(fr.ET) if now.tzinfo else now.tz_localize("UTC").tz_convert(fr.ET)
    if not fr.is_us_trading_day(et.date()) or (et.hour, et.minute) < AFTER_CLOSE:
        return None
    return pd.Timestamp(et.date())


def build_overlay(session: pd.Timestamp, closes: dict[str, float | None],
                  now: pd.Timestamp) -> dict | None:
    """The overlay document, or None when no key has a usable close."""
    good = {k: round(float(v), 6) for k, v in closes.items()
            if v is not None and pd.notna(v) and float(v) > 0}
    if not good:
        return None
    return {"schema": SCHEMA, "session": str(session.date()),
            "fetched_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "closes": dict(sorted(good.items())),
            "missing": sorted(k for k in closes if k not in good)}


def unchanged(old: dict | None, new: dict) -> bool:
    return bool(old) and old.get("session") == new["session"] \
        and old.get("closes") == new["closes"]


def _close_on(df: pd.DataFrame | None, session: pd.Timestamp, col: str) -> float | None:
    if df is None or not len(df):
        return None
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    hit = df[col][idx.normalize() == session]
    hit = hit.dropna()
    return float(hit.iloc[-1]) if len(hit) else None


def fetch_close(symbol: str, session: pd.Timestamp) -> float | None:
    """The official (unadjusted) close of *symbol* on *session*: Yahoo, else Nasdaq."""
    try:
        import yfinance as yf
        h = yf.Ticker(symbol).history(period="7d", interval="1d", auto_adjust=False)
        v = _close_on(h, session, "Close")
        if v:
            return v
    except Exception as e:                    # noqa: BLE001 — fall through to Nasdaq
        print(f"  [yahoo] {symbol}: {type(e).__name__}: {e}")
    try:
        import market_fallback as mf
        start = (session - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        end = (session + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        return _close_on(mf.daily_ohlcv(symbol, start, end), session, "close")
    except Exception as e:                    # noqa: BLE001
        print(f"  [nasdaq] {symbol}: {type(e).__name__}: {e}")
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--force", action="store_true",
                    help="ignore the 4:10 PM ET guard (uses the latest completed session)")
    args = ap.parse_args(argv)

    now = pd.Timestamp.now(tz="UTC")
    session = session_for(now)
    if session is None and args.force:
        session = fr.expected_equity_asof(now)
    if session is None:
        print("Before 4:10 PM ET or not a trading day — nothing to do.")
        return 0

    keys = overlay_keys()
    closes = {k: fetch_close(s, session) for k, s in keys.items()}
    doc = build_overlay(session, closes, now)
    if doc is None:
        print(f"No closes found for {session.date()} yet — leaving the overlay as is.")
        return 0
    try:
        old = json.loads(args.out.read_text())
    except (OSError, ValueError):
        old = None
    if unchanged(old, doc):
        print(f"Overlay for {doc['session']} unchanged ({len(doc['closes'])} closes).")
        return 0
    args.out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"Wrote {args.out}: {len(doc['closes'])} closes for {doc['session']}"
          + (f"; missing {', '.join(doc['missing'])}" if doc["missing"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
