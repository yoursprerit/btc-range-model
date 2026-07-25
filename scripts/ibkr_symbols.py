"""Signal-key → IBKR contract mapping for the Overall Trading paper rebalancer.

The Overall engine (``app/overall_core.py``) keys every traded instrument by a
short signal key (``ASSET_META``): BTC, MSTR, MSTU, ETH, GLDM, GDX, UGL, NUGT,
SOXX, SOXL, GRID, XLE, OIH, ERX, REMX, WGMI, PBW, ARTY.  Every one of these is a
US-listed ETF / equity that Interactive Brokers can trade **except the two spot
crypto sleeves** — IBKR has no spot-Bitcoin or spot-Ether product, so the BTC
signal sleeve is traded via the **IBIT** spot-Bitcoin ETF and the ETH sleeve via
the **ETHA** spot-Ether ETF, exactly as the user specified.

This module is the single source of truth for that key → tradeable-symbol map so
both the rebalancer and the price-sizing step agree.  ``ib_async`` is imported
lazily inside :func:`ibkr_contract` so the module (and its ``TRADE_SYMBOL`` map)
can be imported for dry-runs / tests on a host that does not have ``ib_async``
installed.
"""
from __future__ import annotations

# ── key → the symbol we actually trade on IBKR ────────────────────────────────
# Identity for everything except BTC, whose signal is expressed through the IBIT
# spot-Bitcoin ETF.  SATA (the idle-cash preferred) is tradeable: the PAPER book
# leaves idle capital as cash (no SATA weight → never traded on paper), while the
# LIVE book parks it in a real SATA position — so SATA maps to itself here.
TRADE_SYMBOL: dict[str, str] = {
    "BTC":  "IBIT",   # spot-BTC signal → iShares Bitcoin Trust ETF (no spot BTC on IBKR)
    "ETH":  "ETHA",   # spot-ETH signal → iShares Ethereum Trust ETF (no spot ETH on IBKR)
    "SATA": "SATA",   # idle-cash park (Strive preferred) — used by the LIVE book only
    "MSTR": "MSTR",
    "MSTU": "MSTU",

    "GLDM": "GLDM",
    "GDX":  "GDX",
    "UGL":  "UGL",
    "NUGT": "NUGT",
    "SOXX": "SOXX",
    "SOXL": "SOXL",
    "GRID": "GRID",
    "XLE":  "XLE",
    "OIH":  "OIH",
    "ERX":  "ERX",
    "REMX": "REMX",
    "WGMI": "WGMI",
    "PBW":  "PBW",
    "ARTY": "ARTY",
}

# reverse map: IBKR symbol → signal key, so positions read back from the broker
# can be matched to a target weight.
KEY_FOR_SYMBOL: dict[str, str] = {v: k for k, v in TRADE_SYMBOL.items()}


def trade_symbol(key: str) -> str | None:
    """The IBKR ticker traded for a signal ``key`` (``None`` if unmapped)."""
    return TRADE_SYMBOL.get(key)


def key_for_symbol(symbol: str) -> str | None:
    """The signal key a held IBKR ``symbol`` belongs to (``None`` if foreign)."""
    return KEY_FOR_SYMBOL.get(symbol.upper())


def ibkr_contract(symbol: str):
    """A qualified-ready IBKR ``Stock`` contract for a US-listed ``symbol``.

    Routed ``SMART`` in USD — the correct venue for every ETF/equity in the
    universe (IBIT, ETHA, the leveraged sleeves, and the sector ETFs all trade as
    ordinary US stocks).  ``ib_async`` is imported here (not at module top) so
    this file stays importable for offline dry-runs and unit tests.
    """
    from ib_async import Stock   # lazy: only needed when actually trading
    return Stock(symbol, "SMART", "USD")
