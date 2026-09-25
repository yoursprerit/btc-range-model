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

    # read-only: save what the C2 model account holds (shares + average fill)
    # to data/overall/c2_positions.json for the Overall app's Current Positions:
    C2_API_KEY=… C2_STRATEGY_ID=… python scripts/publish_c2.py --snapshot

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
  so a re-run cannot churn the model account on intraday price moves. The
  marker is committed (``data/overall/c2_publish_state.json``) so the
  scheduled workflow's fresh checkouts see it.
* Share counts are floored and a cash buffer is held back, so the model account
  never sizes onto margin from stale publish-time prices.
* No-trade band (``--band``, default 1% of capital, the executor's
  ``IBKR_BAND``): a name C2 already holds keeps its current quantity when the
  resize would move less than the band, so an unchanged book sends no orders.
  New positions and exits always go through.

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
DEFAULT_STATE = _REPO / "data" / "overall" / "c2_publish_state.json"
DEFAULT_SNAPSHOT = _REPO / "data" / "overall" / "c2_positions.json"
SNAPSHOT_SCHEMA = "c2-positions/v1"
DEFAULT_CASH_BUFFER = 0.01      # keep 1% of model capital uninvested
MAX_WEIGHT_SUM = 1.0 + 1e-6     # a book summing past 100% would size onto margin
DEFAULT_BAND = 0.01             # no-trade band, fraction of capital (= IBKR executor)


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


def apply_band(positions: list[dict], current: dict[str, float], capital: float,
               band: float = DEFAULT_BAND) -> tuple[list[dict], list[str]]:
    """Hold C2's current quantity for names whose resize is inside the band.

    ``current`` is C2's open positions, symbol → shares. A target for a symbol
    already held whose ``|target − held| × price`` is below ``band × capital``
    is replaced by the held quantity, so SetDesiredPositions places no order for
    it. New names (not held) and exits (held, no target) are never banded.
    If keeping the held quantities would put the book above 100% of capital,
    the band is dropped for this run rather than size onto margin.

    Returns ``(positions, banded_symbols)``."""
    if band <= 0 or not current:
        return positions, []
    limit = band * capital
    out, banded = [], []
    for p in positions:
        held = current.get(p["symbol"], 0.0)
        if held > 0 and held != p["quantity"] \
                and abs(p["quantity"] - held) * p["price"] < limit:
            out.append({**p, "quantity": held, "banded_from": p["quantity"]})
            banded.append(p["symbol"])
        else:
            out.append(p)
    if sum(p["quantity"] * p["price"] for p in out) > capital:
        return positions, []
    return out, banded


def orders_needed(positions: list[dict], current: dict[str, float]) -> bool:
    """Would SetDesiredPositions trade anything? False when every target equals
    C2's held quantity and nothing held is left out (which would close it)."""
    target = {p["symbol"]: p["quantity"] for p in positions}
    held = {s: q for s, q in current.items() if q}
    return target != held


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


def build_snapshot(resp: dict, strategy_id: int, account_value: float | None,
                   fetched_at_utc: str, book_as_of: str | None = None) -> dict:
    """GetStrategyOpenPositions response → the positions snapshot the Overall
    app's 💼 Current Positions reads. Shaped like an execution report's
    ``positions`` (``key, symbol, shares, avg_cost``) so the app's cost-basis
    P&L code reads both; ``avg_cost`` is C2's ``AvgPx`` — the model account's
    real average fill. Stocks only, one row per symbol, largest first."""
    agg: dict[str, dict] = {}
    for pos in resp.get("Results") or []:
        sym_ = ((pos.get("ExchangeSymbol") or {}).get("Symbol")
                or (pos.get("C2Symbol") or {}).get("FullSymbol"))
        try:
            qty = float(pos.get("Quantity") or 0.0)
            px = float(pos.get("AvgPx") or 0.0)
        except (TypeError, ValueError):
            continue
        if not sym_ or not qty:
            continue
        sym_ = sym_.upper()
        row = agg.setdefault(sym_, {"shares": 0.0, "cost": 0.0, "opened": None})
        row["shares"] += qty
        row["cost"] += qty * px
        opened = pos.get("OpenedDate") or pos.get("OpenedDateTime")
        if opened and (row["opened"] is None or str(opened) < row["opened"]):
            row["opened"] = str(opened)
    positions = []
    for sym_, r in agg.items():
        if not r["shares"]:
            continue
        positions.append({
            "key": sym.key_for_symbol(sym_) or sym_, "symbol": sym_,
            "shares": r["shares"],
            "avg_cost": round(r["cost"] / r["shares"], 6),
            "opened": r["opened"],
        })
    positions.sort(key=lambda p: -abs(p["shares"] * p["avg_cost"]))
    return {
        "schema": SNAPSHOT_SCHEMA, "fetched_at_utc": fetched_at_utc,
        "strategy_id": int(strategy_id), "book_as_of": book_as_of,
        "model_account_value": account_value, "positions": positions,
    }


def snapshot_changed(old: dict, new: dict) -> bool:
    """Whether *new* is worth committing over *old*: the holdings or the
    book changed, or the old one is from an earlier (UTC) day — so the
    15-minute workflow slots do not commit a fresh timestamp every run, but
    the account value still refreshes at least daily."""
    if not old or old.get("schema") != new.get("schema"):
        return True
    if old.get("positions") != new.get("positions"):
        return True
    if old.get("book_as_of") != new.get("book_as_of"):
        return True
    return str(old.get("fetched_at_utc"))[:10] != str(new.get("fetched_at_utc"))[:10]


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


def parse_open_positions(resp: dict) -> dict[str, float]:
    """GetStrategyOpenPositions response → {symbol: net shares} (stocks only)."""
    out: dict[str, float] = {}
    for pos in resp.get("Results") or []:
        sym_ = ((pos.get("ExchangeSymbol") or {}).get("Symbol")
                or (pos.get("C2Symbol") or {}).get("FullSymbol"))
        qty = pos.get("Quantity")
        if sym_ and qty:
            out[sym_.upper()] = out.get(sym_.upper(), 0.0) + float(qty)
    return out


def open_positions(api_key: str, strategy_id: int) -> dict[str, float]:
    resp = _c2_request("GET", "/Strategies/GetStrategyOpenPositions", api_key,
                       params=f"StrategyIds={int(strategy_id)}&SecurityType=CS")
    return parse_open_positions(resp)


def fetch_snapshot(api_key: str, strategy_id: int,
                   book_as_of: str | None = None) -> dict:
    resp = _c2_request("GET", "/Strategies/GetStrategyOpenPositions", api_key,
                       params=f"StrategyIds={int(strategy_id)}&SecurityType=CS")
    try:
        value = model_account_value(api_key, strategy_id)
    except C2Error:
        value = None                     # holdings are the point; value is extra
    return build_snapshot(resp, strategy_id, value,
                          datetime.now(timezone.utc).isoformat(timespec="seconds"),
                          book_as_of)


def run_snapshot(api_key: str, strategy_id: int | None, path: Path,
                 state_path: Path) -> int:
    """``--snapshot``: read-only — record what the C2 model account holds."""
    if not (api_key and strategy_id):
        print("ABORT: --snapshot needs C2_API_KEY and C2_STRATEGY_ID")
        return 2
    book_as_of = _load_state(state_path).get("as_of")
    try:
        snap = fetch_snapshot(api_key, strategy_id, book_as_of)
    except C2Error as e:
        print(f"ABORT: {e}")
        return 1
    print(f"C2 holds {len(snap['positions'])} position(s)"
          + (f" · model account ${snap['model_account_value']:,.0f}"
             if snap["model_account_value"] else ""))
    for p in snap["positions"]:
        print(f"  {p['symbol']:<7}{p['shares']:>9g} @ {p['avg_cost']:,.2f}")
    if not snapshot_changed(_load_state(path), snap):
        print("snapshot unchanged — not rewritten")
        return 0
    _save_state(path, snap)
    print(f"snapshot written to {path}")
    return 0


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
    ap.add_argument("--band", type=float, default=DEFAULT_BAND,
                    help="no-trade band as a fraction of capital (0 disables)")
    ap.add_argument("--allow-flat", action="store_true",
                    help="allow an all-cash book to close every C2 position")
    ap.add_argument("--outside-rth", action="store_true",
                    help="send outside regular trading hours")
    ap.add_argument("--require-signature", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-send a book already published to C2")
    ap.add_argument("--state", default=str(DEFAULT_STATE))
    ap.add_argument("--snapshot", action="store_true",
                    help="only save C2's open positions (read-only, sends nothing)")
    ap.add_argument("--snapshot-file", default=str(DEFAULT_SNAPSHOT))
    args = ap.parse_args(argv)

    api_key = os.environ.get("C2_API_KEY", "")
    if args.snapshot:
        print("── Collective2 positions snapshot ──")
        return run_snapshot(api_key, args.strategy_id, Path(args.snapshot_file),
                            Path(args.state))
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

    # C2's current holdings drive the no-trade band. Without credentials (a
    # --capital preview) there is nothing to compare against, so no band.
    current: dict[str, float] = {}
    if api_key and args.strategy_id:
        try:
            current = open_positions(api_key, args.strategy_id)
        except C2Error as e:
            print(f"ABORT: {e}")
            return 1
        print(f"C2 holds {len(current)} position(s)")

    weights = payload.get("weights") or {}
    positions, dropped = build_positions(weights, payload.get("exec_price") or {},
                                         capital, args.cash_buffer)
    positions, banded = apply_band(positions, current, capital, args.band)
    print(f"\n{'symbol':<7}{'weight':>8}{'shares':>9}{'price':>11}{'value':>13}")
    for p in positions:
        note = (f"  held (target {p['banded_from']}, inside {args.band*100:g}% band)"
                if "banded_from" in p else "")
        print(f"{p['symbol']:<7}{p['weight']*100:>7.1f}%{p['quantity']:>9g}"
              f"{p['price']:>11,.2f}{p['quantity']*p['price']:>13,.0f}{note}")
    targets = {p["symbol"] for p in positions}
    for s_ in sorted(set(current) - targets):
        print(f"  close: {s_} ({current[s_]:g} shares, no longer in the book)")
    invested = sum(p["quantity"] * p["price"] for p in positions)
    print(f"{'total':<7}{'':>8}{'':>9}{'':>11}{invested:>13,.0f}"
          f"  ({invested/capital*100:.1f}% of capital)")
    for d in dropped:
        print(f"  dropped: {d}")

    ok, why = check_positions(weights, positions, args.allow_flat)
    if not ok:
        print(f"ABORT: {why}")
        return 2

    needed = orders_needed(positions, current) if current else True
    if not args.execute:
        if not needed:
            print("\nNo orders needed — C2 already holds these positions.")
        print("\nDRY-RUN — nothing sent. Re-run with --execute to publish.")
        return 0
    if not (api_key and args.strategy_id):
        print("ABORT: --execute needs C2_API_KEY and C2_STRATEGY_ID")
        return 2

    if not needed:
        # every change was inside the band: nothing to trade, but the book is
        # handled — record it so later slots skip it too
        print("\nNo orders needed — C2 already holds these positions (nothing sent).")
    else:
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
    if needed:
        print(f"\npublished {len(positions)} positions to C2 strategy {args.strategy_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
