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
      "realized_pnl": 127.6,       # account, this session (null = not recorded)
      "trades":    [ {key, symbol, action, qty, price, value, status, filled,
                      avg_fill_price, order_type, limit_price, reason} … ],
      "positions": [ {key, symbol, shares, avg_cost, market_price, market_value,
                      unrealized_pnl, realized_pnl} … ],
      "signature": { "alg": "HMAC-SHA256", "value": "…" }   # optional
    }

``realized_pnl`` (added after v1 shipped) is what a rebalance actually BANKED.
It matters because a partial trim is otherwise invisible here: IBKR leaves the
remaining shares' ``avg_cost`` untouched when part of a long is sold, so the
position's unrealised P&L just scales down with the share count and the gain on
the sold slice vanishes.  It is carried in two places because neither alone is
complete — per position (what that name's trim booked) and per account (the
only figure that can include a name closed out ENTIRELY, since IBKR drops a
zero-size position from the portfolio read).  Both are **nullable**: reports
written before the field existed omit it, and null must never be read as a real
zero.  The figure is per trading SESSION, not since inception; the executor
snapshots it right after its rebalance, so in normal operation it is that
rebalance's own realised result.

Adding the field needs no schema bump — old readers ignore an unknown key, new
readers use ``.get()`` — and archived records signed before it existed still
verify, since each signature covers the payload as it was written.

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


def opt_float(raw) -> float | None:
    """A nullable numeric field: the value as a float, else None.

    Used for the P&L fields, where None ("the broker did not report this", or
    "this report predates the field") must stay distinguishable from a genuine
    0.0.  Coercing the two together would let an old record claim a rebalance
    banked nothing when in fact nobody ever asked."""
    if raw is None or raw == "":
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    return None if val != val else val          # NaN is not a value either


def build_payload(*, as_of: str, profile: str, mode: str, account: str,
                  net_liq: float, cash: float, trades: list[dict],
                  positions: list[dict], generated_at_utc: str | None = None,
                  account_mode: str = "paper",
                  realized_pnl: float | None = None) -> dict:
    """Assemble a v1 execution-report payload (unsigned).

    ``mode`` is execute/dry-run; ``account_mode`` is paper/live.

    ``realized_pnl`` is the ACCOUNT's session realized P&L at the moment the
    report was written (see ``ibkr_common.IBKRBroker.realized_pnl``); each
    position may carry its own ``realized_pnl`` too.  ``None`` throughout means
    "not recorded" — reports written before the field existed simply omit it,
    and that must never read as a real zero."""
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
            # session realized P&L for this name — what a trim booked. Kept
            # nullable: "IBKR didn't report it" is not "it booked nothing".
            "realized_pnl": opt_float(p.get("realized_pnl")),
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
        # account-level session realized P&L: the only figure that can see a
        # name closed out entirely (IBKR drops a zero-size position from the
        # portfolio, so its realized P&L is in no per-position row)
        "realized_pnl": opt_float(realized_pnl),
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

    Only an **executed** run is archived.  A dry-run is a preview of orders that
    were never sent, and archiving one would overwrite the record of what the
    account actually traded on that bar with a list of PLANNED trades — which is
    exactly what an after-hours dry-run did to the 2026-08-17 record on
    2026-08-18.  The Historical tab is the record of executions; previews do not
    belong in it.

    Signed with *secret* when given (same HMAC as the live report), so archived
    records stay verifiable.  Best-effort: a failure to archive never fails a
    rebalance.  Returns the path written, or None."""
    try:
        import target_book as tb                     # local: keeps this module
        as_of = payload.get("as_of")                 # importable on its own
        if not as_of:
            return None
        if (payload.get("mode") or "").lower() != "execute":
            return None                              # a preview, not a record
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
    reset = read_reset(report_path)
    cutoff = (reset or {}).get("cutoff_as_of") or ""
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
        if cutoff and str(pd.Timestamp(as_of).date()) <= cutoff:
            continue                       # a previous account's run — not ours
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


# ── reset marker ─────────────────────────────────────────────────────────────
# Resetting the broker account (a fresh paper account, a wiped balance) makes
# every earlier run a record of a DIFFERENT account: same schema, unrelated
# money. Deleting the files is not enough on its own — the reports live on in
# git history, and ``backfill_executed_archive.py`` would happily restore them.
# A marker records the cutoff so the reset survives: readers ignore anything at
# or before it, and the backfill refuses to re-create it.
RESET_MARKER = "_reset"
RESET_SCHEMA = "executed-archive-reset/v1"


def reset_marker_path(report_path) -> Path:
    """``_reset.json`` for paper, ``_reset_live.json`` for live.

    Paper and live share the archive directory, so the marker carries the same
    ``_live`` suffix its records do — resetting a paper account must never
    retire the live account's history."""
    return archive_dir(report_path) / f"{RESET_MARKER}{variant(report_path)}.json"


def read_reset(report_path) -> dict | None:
    """The reset marker for this account mode, or None."""
    try:
        p = reset_marker_path(report_path)
        if not p.exists():
            return None
        payload = json.loads(p.read_text())
    except Exception:
        return None
    return payload if payload.get("schema") == RESET_SCHEMA else None


def write_reset(report_path, cutoff_as_of, reason: str = "account reset") -> Path:
    """Record that runs up to and including *cutoff_as_of* belong to a previous
    account and must never be shown or restored."""
    p = reset_marker_path(report_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "schema": RESET_SCHEMA,
        "reset_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cutoff_as_of": str(pd.Timestamp(cutoff_as_of).date()),
        "reason": reason,
    }, indent=1))
    return p


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


def completed_run(report_path, as_of) -> dict | None:
    """The archived execution report for *as_of* on this account mode, if any.

    The executor uses it as a duplicate-run lock: an archived record means the
    book was already traded, so a second execute run for the same signal bar is
    refused (``--force-rerun`` overrides).  Dry-runs place no orders and are not
    treated as an execution.  Returns the payload, or None."""
    try:
        reset = read_reset(report_path)
        cutoff = (reset or {}).get("cutoff_as_of") or ""
        if cutoff and str(pd.Timestamp(as_of).date()) <= cutoff:
            return None                    # pre-reset: a different account
        p = archive_path(report_path, as_of)
        if not p.exists():
            return None
        payload = json.loads(p.read_text())
    except Exception:
        return None
    if payload.get("schema") != SCHEMA:
        return None
    if (payload.get("mode") or "").lower() != "execute":
        return None                        # a dry-run traded nothing
    return payload


# ════════════════════════════════════════════════════════════════════════════
# COST-BASIS POSITIONS  (what the account actually holds, marked to a price)
# ════════════════════════════════════════════════════════════════════════════
# The execution report is the ONLY record of what the account really paid: the
# strategy engines know an entry BAR (a daily close), the broker knows the fill.
# ``current_positions`` turns a report into per-instrument rows whose cost basis
# is the broker's ``avg_cost`` and whose mark is whatever price the caller
# supplies — the live spot on the 🔴 Live tab, that bar's official close on the
# 🕰️ Historical one.  Pure and price-source-agnostic so both tabs (and the
# tests) share one P&L definition.

def current_positions(payload: dict | None, prices: dict | None = None) -> dict:
    """Cost-basis positions from an execution report, marked at *prices*.

    *prices* maps instrument key → either a plain number or a mapping with a
    ``price`` (and optional ``dchg`` day-change %) — so a spot-quote dict and a
    ``{key: close}`` dict both work.  A key with no usable mark falls back to
    the ``market_price`` the report itself carried at execution time, flagged
    ``price_src='report'`` so the UI can say the quote is not current; a
    position with neither is ``price_src='none'`` and contributes no P&L.

    Returns ``{rows, cost, value, pnl, pnl_pct, n, n_live, n_stale, realized,
    realized_account, realized_known, total_pnl, as_of, cash, net_liq,
    account_mode, mode}``; ``rows`` are largest-position-first and each carries
    ``key, symbol, shares, avg_cost, cost_basis, price, price_src, dchg,
    market_value, pnl, pnl_pct, realized``.

    On realized P&L: ``realized`` sums what the open positions booked this
    session (a trim's gain), ``realized_account`` is the account-level figure
    that ALSO covers names closed out entirely, and ``total_pnl`` adds the
    better of the two to the open P&L — because a trim-and-re-add cycle
    otherwise reports only the smaller open profit and silently drops what was
    already banked.  ``realized_known`` is False for a report written before
    the field existed, so the UI can say "not recorded" rather than "$0".
    """
    payload = payload or {}
    prices = prices or {}

    def _mark(key: str) -> tuple[float, float | None]:
        """(price, day-change %) for *key* from the caller's price map."""
        q = prices.get(key)
        if q is None:
            return 0.0, None
        if isinstance(q, dict):
            try:
                px = float(q.get("price") or 0.0)
            except (TypeError, ValueError):
                px = 0.0
            d = q.get("dchg")
            try:
                d = float(d) if d is not None else None
            except (TypeError, ValueError):
                d = None
            return px, d
        try:
            return float(q), None
        except (TypeError, ValueError):
            return 0.0, None

    rows = []
    for p in payload.get("positions") or []:
        key = p.get("key") or p.get("symbol") or ""
        shares = float(p.get("shares") or 0.0)
        avg_cost = float(p.get("avg_cost") or 0.0)
        if not shares or avg_cost <= 0:
            continue                        # a closed / unpriced line, not a holding
        px, dchg = _mark(key)
        src = "live"
        if px <= 0:                          # no live quote — the report's own mark
            px = float(p.get("market_price") or 0.0)
            src = "report" if px > 0 else "none"
        cost_basis = shares * avg_cost
        mv = shares * px if px > 0 else 0.0
        pnl = (mv - cost_basis) if px > 0 else 0.0
        pnl_pct = (pnl / abs(cost_basis) * 100) if (px > 0 and cost_basis) else None
        rows.append(dict(
            key=key, symbol=p.get("symbol") or key, shares=shares,
            avg_cost=avg_cost, cost_basis=cost_basis, price=px, price_src=src,
            dchg=dchg, market_value=mv, pnl=pnl, pnl_pct=pnl_pct,
            realized=opt_float(p.get("realized_pnl"))))

    rows.sort(key=lambda r: -abs(r["market_value"] or r["cost_basis"]))
    cost = sum(r["cost_basis"] for r in rows)
    value = sum(r["market_value"] for r in rows if r["price_src"] != "none")
    priced_cost = sum(r["cost_basis"] for r in rows if r["price_src"] != "none")
    pnl = value - priced_cost

    # realized: per-position sums cover trims of names still held; the
    # account figure additionally covers names closed out entirely (which have
    # no row at all).  Prefer the account figure when it exists — it is the
    # superset — and fall back to the per-position sum when it does not.
    per_pos = [r["realized"] for r in rows if r["realized"] is not None]
    acct = opt_float(payload.get("realized_pnl"))
    realized_sum = sum(per_pos) if per_pos else None
    realized = acct if acct is not None else realized_sum
    return dict(
        rows=rows, cost=cost, value=value, pnl=pnl,
        pnl_pct=(pnl / abs(priced_cost) * 100) if priced_cost else None,
        n=len(rows),
        n_live=sum(1 for r in rows if r["price_src"] == "live"),
        n_stale=sum(1 for r in rows if r["price_src"] != "live"),
        realized=realized, realized_account=acct,
        realized_positions=realized_sum,
        realized_known=realized is not None,
        total_pnl=pnl + (realized or 0.0),
        as_of=payload.get("as_of"), cash=float(payload.get("cash") or 0.0),
        net_liq=float(payload.get("net_liq") or 0.0),
        account_mode=(payload.get("account_mode") or "paper"),
        mode=(payload.get("mode") or ""),
        generated_at_utc=payload.get("generated_at_utc") or "")
