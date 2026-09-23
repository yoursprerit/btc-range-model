"""Mirror the published target book onto a Collective2 strategy.

Reads the same signed artifact the IBKR executor trades
(``data/overall/target_book.json``), converts its weights into share counts
against the C2 strategy's model-account value, and sends them to C2's
``SetDesiredPositions`` endpoint (API v4). C2 then places whatever market
orders move its model account onto those positions — the declarative twin of
our own rebalance, so the public track record follows the book exactly.

    # preview (default — nothing is sent); needs no API key with --capital:
    python scripts/publish_c2.py --capital 100000

    # read the model-account value from C2 and preview against it:
    C2_API_KEY=… C2_STRATEGY_ID=… python scripts/publish_c2.py

    # actually send the desired positions:
    OVERALL_BOOK_SECRET=… C2_API_KEY=… C2_STRATEGY_ID=… \\
        python scripts/publish_c2.py --execute

Safety rails
------------
* Dry-run is the DEFAULT; sending requires ``--execute``.
* SetDesiredPositions CLOSES every C2 position left out of the list, and an
  empty list liquidates the strategy. So a book that sizes to zero positions is
  refused, and a genuinely all-cash book needs ``--allow-flat``.
* Same book checks as the executor: the HMAC signature must verify when
  ``OVERALL_BOOK_SECRET`` is set (or ``--require-signature``), and the book must
  pass ``target_book.validate`` (schema, freshness, publish-time audit).
* Only sends on a US trading day inside regular hours — C2 cannot adjust
  positions while the market is shut (``--outside-rth`` overrides the hours).
* Idempotent per book: a book already sent is skipped (``--force`` re-sends),
  so a re-run cannot churn the model account on intraday price moves.
* Share counts are floored and a cash buffer is held back, so the model account
  never sizes onto margin from stale publish-time prices.

Stdlib + pandas only (no ``requests``), so it runs on the executor host.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "app"))
sys.path.insert(0, str(_REPO / "scripts"))

import target_book as tb                           # noqa: E402
import ibkr_symbols as sym                          # noqa: E402
import ibkr_common as ic                            # noqa: E402

C2_API_BASE = "https://api4-general.collective2.com"
DEFAULT_BOOK = _REPO / "data" / "overall" / "target_book.json"
DEFAULT_STATE = _REPO / "logs" / "c2_publish_state.json"
DEFAULT_CASH_BUFFER = 0.01      # keep 1% of model capital uninvested
MAX_WEIGHT_SUM = 1.0 + 1e-6     # a book summing past 100% would size onto margin


class C2Error(RuntimeError):
    """C2 refused or failed a request — never treat the publish as done."""


# ════════════════════════════════════════════════════════════════════════════
# PURE HELPERS (unit-tested, no network)
# ════════════════════════════════════════════════════════════════════════════
def build_positions(weights: dict[str, float], exec_price: dict[str, float],
                    capital: float, cash_buffer: float = DEFAULT_CASH_BUFFER
                    ) -> tuple[list[dict], list[str]]:
    """Target weights → C2 desired positions (whole shares, floored).

    Returns ``(positions, dropped)``: ``positions`` is ``[{symbol, key, quantity,
    price, weight}]`` sorted by symbol; ``dropped`` names the keys that could not
    be sized (no tradeable symbol, no price, or less than one share)."""
    investable = capital * (1.0 - cash_buffer)
    positions, dropped = [], []
    for key in sorted(weights):
        w = float(weights[key] or 0.0)
        if w <= 0:
            continue
        symbol = sym.trade_symbol(key)
        price = exec_price.get(key)
        if not symbol:
            dropped.append(f"{key} (no tradeable symbol)")
            continue
        if not price or price <= 0:
            dropped.append(f"{key} (no price)")
            continue
        qty = math.floor(w * investable / price)
        if qty < 1:
            dropped.append(f"{key} ({w*100:.2f}% < one share at ${price:,.2f})")
            continue
        positions.append({"symbol": symbol, "key": key, "quantity": qty,
                          "price": float(price), "weight": w})
    positions.sort(key=lambda p: p["symbol"])
    return positions, dropped


def check_positions(weights: dict[str, float], positions: list[dict],
                    allow_flat: bool) -> tuple[bool, str]:
    """(ok, reason) — refuse any request that would wrongly liquidate the strategy.

    An empty ``Positions`` list tells C2 to close everything, so it is only
    allowed when the book itself is all cash AND the caller said so."""
    total = sum(max(float(w or 0.0), 0.0) for w in weights.values())
    if total > MAX_WEIGHT_SUM:
        return False, f"book weights sum to {total*100:.2f}% (> 100%) — refusing"
    if positions:
        return True, f"{len(positions)} positions"
    if total > 0:
        return False, ("book holds positions but none sized to a whole share "
                       "— sending would liquidate the C2 strategy; raise capital")
    if not allow_flat:
        return False, "book is all cash — pass --allow-flat to close every C2 position"
    return True, "all cash — closing every C2 position (--allow-flat)"


def desired_positions_payload(strategy_id: int, positions: list[dict]) -> dict:
    """The SetDesiredPositionsV2 request body (US stocks/ETFs only)."""
    return {
        "StrategyId": int(strategy_id),
        "Positions": [{
            "StrategyId": int(strategy_id),
            "Quantity": p["quantity"],
            "ExchangeSymbol": {"Symbol": p["symbol"], "Currency": "USD",
                               "SecurityExchange": "DEFAULT", "SecurityType": "CS"},
        } for p in positions],
    }


def book_fingerprint(payload: dict) -> str:
    """Identity of a book for the already-sent check: its bar plus its signature
    (or, unsigned, its weights) — a re-published book for the same bar re-sends."""
    sig = (payload.get("signature") or {}).get("value")
    body = sig or json.dumps(payload.get("weights") or {}, sort_keys=True)
    return f"{payload.get('as_of')}|{body}"


def summarize_response(resp: dict) -> tuple[bool, list[str]]:
    """(ok, lines) from a SetDesiredPositionsV2 response. Any rejected signal or
    a ResponseStatus error code makes the publish a failure."""
    lines, ok = [], True
    status = resp.get("ResponseStatus") or {}
    code = str(status.get("ErrorCode") or "")
    if (code.isdigit() and int(code) >= 400) or status.get("Errors"):
        ok = False
        lines.append(f"C2 error {code}: {status.get('Message')}")
        for e in status.get("Errors") or []:
            lines.append(f"  {e.get('FieldName')}: {e.get('Message')}")
    side = {"1": "BUY", "2": "SELL"}
    for res in resp.get("Results") or []:
        for label, key in (("new", "NewSignals"), ("canceled", "CanceledSignals"),
                           ("REJECTED", "RejectedSignals")):
            for s in res.get(key) or []:
                sym_ = ((s.get("ExchangeSymbol") or {}).get("Symbol")
                        or (s.get("C2Symbol") or {}).get("FullSymbol") or "?")
                msg = f"  {label:<9}{side.get(str(s.get('Side')), '?'):<5}" \
                      f"{s.get('OrderQuantity')!s:>8} {sym_}"
                if key == "RejectedSignals":
                    ok = False
                    msg += f" — {s.get('RejectMessage')}"
                lines.append(msg)
    if ok and len(lines) == 0:
        lines.append("  no new signals (C2 model account already matches)")
    return ok, lines


# ════════════════════════════════════════════════════════════════════════════
# C2 API
# ════════════════════════════════════════════════════════════════════════════
def _c2_request(method: str, path: str, api_key: str, *, params: str = "",
                body: dict | None = None, timeout: float = 30.0) -> dict:
    url = f"{C2_API_BASE}{path}" + (f"?{params}" if params else "")
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:500]
        raise C2Error(f"{method} {path} → HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise C2Error(f"{method} {path} → {e.reason}") from e


def model_account_value(api_key: str, strategy_id: int) -> float:
    resp = _c2_request("GET", "/Strategies/GetStrategyDetails", api_key,
                       params=f"StrategyId={int(strategy_id)}")
    results = resp.get("Results") or []
    value = results[0].get("ModelAccountValue") if results else None
    if not value or value <= 0:
        raise C2Error(f"no ModelAccountValue for strategy {strategy_id}: {resp}")
    return float(value)


def set_desired_positions(api_key: str, body: dict) -> dict:
    return _c2_request("POST", "/v2/Strategies/SetDesiredPositions", api_key, body=body)


# ════════════════════════════════════════════════════════════════════════════
# STATE
# ════════════════════════════════════════════════════════════════════════════
def _load_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=1))


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--file", default=str(DEFAULT_BOOK), help="target book JSON")
    ap.add_argument("--execute", action="store_true",
                    help="send to C2 (default is a dry-run preview)")
    ap.add_argument("--strategy-id", type=int,
                    default=int(os.environ["C2_STRATEGY_ID"])
                    if os.environ.get("C2_STRATEGY_ID") else None)
    ap.add_argument("--capital", type=float, default=None,
                    help="size against this capital instead of C2's model-account value")
    ap.add_argument("--cash-buffer", type=float, default=DEFAULT_CASH_BUFFER)
    ap.add_argument("--allow-flat", action="store_true",
                    help="allow an all-cash book to close every C2 position")
    ap.add_argument("--outside-rth", action="store_true",
                    help="send outside regular trading hours")
    ap.add_argument("--require-signature", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-send a book already published to C2")
    ap.add_argument("--state", default=str(DEFAULT_STATE))
    args = ap.parse_args(argv)

    api_key = os.environ.get("C2_API_KEY", "")
    secret = os.environ.get("OVERALL_BOOK_SECRET")
    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(f"── Collective2 publish ({mode}) ──")

    try:
        payload = tb.loads(Path(args.file).read_text())
    except (OSError, ValueError) as e:
        print(f"ABORT: cannot read book {args.file}: {e}")
        return 2

    if args.require_signature and not secret:
        print("ABORT: --require-signature but OVERALL_BOOK_SECRET is not set")
        return 2
    ok, why = tb.verify_signature(payload, secret)
    print(f"signature: {why}")
    if not ok:
        print("ABORT: book signature did not verify")
        return 2

    today = pd.Timestamp.now(tz=ic.ET_TZ).tz_localize(None).normalize()
    ok, why = tb.validate(payload, today)
    print(f"book: {why} · profile {payload.get('profile')} · mode {payload.get('book_mode')}")
    if not ok:
        print("ABORT: book failed validation")
        return 2

    if args.execute:
        ok, why = ic.is_trading_day(today)
        if not ok:
            print(f"SKIP: {why}")
            return 0
        ok, why = ic.market_session_open(allow_outside=args.outside_rth)
        if not ok:
            print(f"SKIP: {why}")
            return 0

    state_path = Path(args.state)
    fp = book_fingerprint(payload)
    if args.execute and not args.force and _load_state(state_path).get("fingerprint") == fp:
        print(f"SKIP: book for {payload.get('as_of')} already sent to C2 (--force to re-send)")
        return 0

    if args.capital:
        capital = args.capital
        print(f"capital: ${capital:,.0f} (--capital)")
    else:
        if not (api_key and args.strategy_id):
            print("ABORT: set C2_API_KEY and C2_STRATEGY_ID, or pass --capital to preview")
            return 2
        try:
            capital = model_account_value(api_key, args.strategy_id)
        except C2Error as e:
            print(f"ABORT: {e}")
            return 1
        print(f"capital: ${capital:,.0f} (C2 model account value)")

    weights = payload.get("weights") or {}
    positions, dropped = build_positions(weights, payload.get("exec_price") or {},
                                         capital, args.cash_buffer)
    print(f"\n{'symbol':<7}{'weight':>8}{'shares':>9}{'price':>11}{'value':>13}")
    for p in positions:
        print(f"{p['symbol']:<7}{p['weight']*100:>7.1f}%{p['quantity']:>9}"
              f"{p['price']:>11,.2f}{p['quantity']*p['price']:>13,.0f}")
    invested = sum(p["quantity"] * p["price"] for p in positions)
    print(f"{'total':<7}{'':>8}{'':>9}{'':>11}{invested:>13,.0f}"
          f"  ({invested/capital*100:.1f}% of capital)")
    for d in dropped:
        print(f"  dropped: {d}")

    ok, why = check_positions(weights, positions, args.allow_flat)
    if not ok:
        print(f"ABORT: {why}")
        return 2

    if not args.execute:
        print("\nDRY-RUN — nothing sent. Re-run with --execute to publish.")
        return 0
    if not (api_key and args.strategy_id):
        print("ABORT: --execute needs C2_API_KEY and C2_STRATEGY_ID")
        return 2

    body = desired_positions_payload(args.strategy_id, positions)
    try:
        resp = set_desired_positions(api_key, body)
    except C2Error as e:
        print(f"ABORT: {e}")
        return 1
    ok, lines = summarize_response(resp)
    print("\nC2 response:")
    print("\n".join(lines))
    if not ok:
        print("FAILED: C2 rejected part of the request — not recording as sent")
        return 1

    _save_state(state_path, {
        "fingerprint": fp, "as_of": payload.get("as_of"),
        "sent_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "strategy_id": args.strategy_id, "capital": capital,
        "positions": {p["symbol"]: p["quantity"] for p in positions},
    })
    print(f"\npublished {len(positions)} positions to C2 strategy {args.strategy_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
