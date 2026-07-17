"""Execution-report artifact — the reverse of the target book.

After the executor rebalances the IBKR paper account it writes this report and
commits it back to the branch, so the cloud Streamlit app can show what actually
happened (the app itself never connects to IBKR). Same transport and signing as
the target book — just flowing laptop/VM → cloud instead of cloud → laptop.

Schema ``executed-book/v1``:

    {
      "schema": "executed-book/v1",
      "generated_at_utc": "…",     # when the rebalance ran
      "as_of": "…",                # the target book's signal bar it executed
      "profile": "Aggressive",
      "mode": "execute" | "dry-run",
      "account": "DU1234567",
      "net_liq": 100000.0,
      "cash": 42000.0,
      "trades":    [ {key, symbol, action, qty, price, value, status, filled,
                      avg_fill_price, reason} … ],
      "positions": [ {key, symbol, shares, avg_cost, market_price, market_value,
                      unrealized_pnl} … ],
      "signature": { "alg": "HMAC-SHA256", "value": "…" }   # optional
    }

Signing/verifying reuses ``target_book.sign`` / ``verify_signature`` (they are
schema-agnostic), so one shared secret covers both artifacts.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

SCHEMA = "executed-book/v1"


def build_payload(*, as_of: str, profile: str, mode: str, account: str,
                  net_liq: float, cash: float, trades: list[dict],
                  positions: list[dict], generated_at_utc: str | None = None) -> dict:
    """Assemble a v1 execution-report payload (unsigned)."""
    def _trade(t: dict) -> dict:
        qty = float(t.get("qty") or 0.0)
        price = float(t.get("price") or 0.0)
        return {
            "key": t.get("key"), "symbol": t.get("symbol"),
            "action": t.get("action"), "qty": qty, "price": price,
            "value": qty * price,
            "status": t.get("status"), "filled": float(t.get("filled") or 0.0),
            "avg_fill_price": float(t.get("avg_fill_price") or 0.0),
            "reason": t.get("reason"),
        }

    def _pos(p: dict) -> dict:
        return {
            "key": p.get("key"), "symbol": p.get("symbol"),
            "shares": float(p.get("shares") or 0.0),
            "avg_cost": float(p.get("avg_cost") or 0.0),
            "market_price": float(p.get("market_price") or 0.0),
            "market_value": float(p.get("market_value") or 0.0),
            "unrealized_pnl": float(p.get("unrealized_pnl") or 0.0),
        }

    return {
        "schema": SCHEMA,
        "generated_at_utc": generated_at_utc or datetime.now(timezone.utc)
                                                        .isoformat(timespec="seconds"),
        "as_of": as_of,
        "profile": profile,
        "mode": mode,
        "account": account,
        "net_liq": float(net_liq or 0.0),
        "cash": float(cash or 0.0),
        "trades": [_trade(t) for t in trades],
        "positions": [_pos(p) for p in positions],
    }


def validate(payload: dict, today: pd.Timestamp, *,
             max_gen_age_hours: float = 48.0) -> tuple[bool, str]:
    """(ok, reason) — schema check + how recently the report was produced.

    A wider default window than the target book (48h) so a Friday execution is
    still "recent" over a weekend when someone looks on Monday."""
    if payload.get("schema") != SCHEMA:
        return False, f"unexpected schema {payload.get('schema')!r} (want {SCHEMA})"
    gen = payload.get("generated_at_utc")
    if not gen:
        return True, "no timestamp"
    gen_ts = pd.Timestamp(gen)
    if gen_ts.tzinfo is None:
        gen_ts = gen_ts.tz_localize("UTC")
    now = pd.Timestamp(datetime.now(timezone.utc))
    age_h = (now - gen_ts).total_seconds() / 3600.0
    if age_h > max_gen_age_hours:
        return False, f"report is {age_h/24:.1f} days old (> {max_gen_age_hours/24:.0f}d)"
    return True, f"executed {gen} UTC ({age_h:.1f}h ago)"
