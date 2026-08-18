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
                      avg_fill_price, order_type, limit_price, reason} … ],
      "positions": [ {key, symbol, shares, avg_cost, market_price, market_value,
                      unrealized_pnl} … ],
      "signature": { "alg": "HMAC-SHA256", "value": "…" }   # optional
    }

Signing/verifying reuses ``target_book.sign`` / ``verify_signature`` (they are
schema-agnostic), so one shared secret covers both artifacts.

Each run also lands a dated copy of the report in
``data/overall/executed_archive/<as_of>[_live].json`` (see the archive section
at the bottom of this module) — the live report file only ever holds the LAST
run, and the app's 🕰️ Historical tab needs the earlier ones.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCHEMA = "executed-book/v1"


def build_payload(*, as_of: str, profile: str, mode: str, account: str,
                  net_liq: float, cash: float, trades: list[dict],
                  positions: list[dict], generated_at_utc: str | None = None,
                  account_mode: str = "paper") -> dict:
    """Assemble a v1 execution-report payload (unsigned).

    ``mode`` is execute/dry-run; ``account_mode`` is paper/live."""
    def _trade(t: dict) -> dict:
        qty = float(t.get("qty") or 0.0)
        price = float(t.get("price") or 0.0)
        return {
            "key": t.get("key"), "symbol": t.get("symbol"),
            "action": t.get("action"), "qty": qty, "price": price,
            "value": qty * price,
            "status": t.get("status"), "filled": float(t.get("filled") or 0.0),
            "avg_fill_price": float(t.get("avg_fill_price") or 0.0),
            # what was actually SENT: marketable-limit / moc / market, plus the
            # limit that capped it (0.0 for market and MOC). Reports written
            # before these existed simply omit them — readers use .get().
            "order_type": t.get("order_type") or "",
            "limit_price": float(t.get("limit_price") or 0.0),
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
        "account_mode": account_mode,
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


# ════════════════════════════════════════════════════════════════════════════
# DATED ARCHIVE  (executed_archive/<as_of>[_live].json)
# ════════════════════════════════════════════════════════════════════════════
# The live report files (``executed_book.json`` / ``executed_book_live.json``)
# only ever hold the LAST run — every rebalance overwrites them, so the record
# of what was traded on earlier days survived only in git history.  The
# executor now also drops a dated copy next to them, keyed by the signal bar it
# executed (the same key the target book's ``book_archive/`` uses), which the
# Executed Book page's **🕰️ Historical** tab browses.
#
# Paper and live runs are kept apart by the same ``_live`` suffix the report
# files use — a paper fill is not a live fill and the two must never merge.
# A re-run for the same signal bar (a late ``--refresh-report``, a manual
# after-hours top-up) replaces that bar's record: last run wins, matching the
# account's end state.
ARCHIVE_DIRNAME = "executed_archive"


def variant(report_path) -> str:
    """``''`` for the paper report, ``'_live'`` for the live one."""
    return "_live" if Path(report_path).stem.endswith("_live") else ""


def archive_path(report_path, as_of) -> Path:
    """``…/executed_book_live.json`` + as_of → ``…/executed_archive/<as_of>_live.json``."""
    report_path = Path(report_path)
    return (report_path.parent / ARCHIVE_DIRNAME
            / f"{pd.Timestamp(as_of).date()}{variant(report_path)}.json")


def archive_report(payload: dict, report_path, secret: str | None = None) -> Path | None:
    """Persist *payload* as the dated as-of record next to *report_path*.

    Signed with *secret* when given (same HMAC as the live report), so archived
    records stay verifiable.  Best-effort: a failure to archive never fails a
    rebalance.  Returns the path written, or None."""
    try:
        import target_book as tb                     # local: keeps this module
        as_of = payload.get("as_of")                 # importable on its own
        if not as_of:
            return None
        p = archive_path(report_path, as_of)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(tb.dumps(payload, secret))
        return p
    except Exception:
        return None


def archive_dir(report_path) -> Path:
    return Path(report_path).parent / ARCHIVE_DIRNAME


def archived_records(report_path) -> list[dict]:
    """Every archived run for *report_path*'s account mode, newest first.

    Each entry is ``{as_of, executed_on, generated_at_utc, path, payload}``,
    where ``executed_on`` is the New-York date the run actually happened (the
    executor trades the morning after the signal bar, so it is normally
    ``as_of`` + 1 session).  Unreadable/foreign records are skipped rather than
    breaking the page."""
    d, want = archive_dir(report_path), variant(report_path)
    if not d.is_dir():
        return []
    out: list[dict] = []
    for p in sorted(d.glob("*.json")):
        stem = p.stem
        if ("_live" if stem.endswith("_live") else "") != want:
            continue
        try:
            payload = json.loads(p.read_text())
        except Exception:
            continue
        if payload.get("schema") != SCHEMA:
            continue
        as_of = payload.get("as_of") or stem.replace("_live", "")
        out.append({
            "as_of": str(pd.Timestamp(as_of).date()),
            "executed_on": _executed_on(payload) or str(pd.Timestamp(as_of).date()),
            "generated_at_utc": payload.get("generated_at_utc") or "",
            "path": p,
            "payload": payload,
        })
    out.sort(key=lambda r: (r["executed_on"], r["as_of"]), reverse=True)
    return out


def _executed_on(payload: dict) -> str | None:
    """New-York calendar date of the run, from ``generated_at_utc``."""
    gen = payload.get("generated_at_utc")
    if not gen:
        return None
    try:
        ts = pd.Timestamp(gen)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return str(ts.tz_convert("America/New_York").date())
    except Exception:
        return None


def record_for(records: list[dict], day) -> dict | None:
    """The run executed ON *day*, else the most recent one BEFORE it.

    The executor runs on trading days only, so a date picker over a plain
    calendar lands on weekends, holidays and skipped cycles; on those the book
    that was standing is the last one executed, which is what this returns.
    None when *day* precedes the whole archive.  *records* is
    :func:`archived_records` output (any order)."""
    ts = pd.Timestamp(day)
    if ts is pd.NaT or pd.isna(ts):        # cleared date input / unparsable day
        return None
    key = str(ts.date())
    on_or_before = [r for r in records if r["executed_on"] <= key]
    if not on_or_before:
        return None
    return max(on_or_before, key=lambda r: (r["executed_on"], r["as_of"]))
