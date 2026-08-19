"""Sleeve ledger — trade the book with only a SLICE of the account, compounded.

The executor sizes every order off the account's net liquidation value, so it
always trades the *whole* account.  This module lets you hand the strategy a
fixed dollar stake once and have it trade **only that stake**, which then
compounds on its own P&L while the rest of the account is never touched.

Why not just ``--max-deploy-frac``
----------------------------------
``--max-deploy-frac 0.10`` scales the book's weights to 10% of net liq — but it
re-derives that 10% from the *full account* on every run, so it is a
constant-fraction rule, not a stake:

* a winning week is **diluted** — the gain is re-spread over the whole account,
  and the sleeve only keeps 10% of it;
* a losing week is **topped up** — fresh capital is pulled from the untouched
  cash to restore the 10%;
* a deposit or withdrawal *elsewhere in the account* silently resizes it.

A sleeve does the opposite on all three: the stake is set once, and from then on
its equity moves one-for-one with the strategy's own P&L.

    sleeve_nav = sleeve_cash + Σ shares_k × price_k        (k over the book)

Only ``sleeve_cash`` is persisted here.  Positions are read from the broker on
every run and verified by the executor's existing positions-trust check, so the
ledger can never disagree with reality about *what is held* — only about how
much cash backs it, and that is driven by actual fills.

THE ONE ASSUMPTION
------------------
The sleeve must be the **only** strategy trading these symbols in this account.
The executor already closes anything held but not in the book, so it behaves as
the sole owner of the book universe; the ledger takes the account's holdings in
those names to be the sleeve's.  If you also trade those tickers by hand in the
same account, use a separate IBKR account instead — the broker's own accounting
beats any ledger.

Dependency-light (stdlib + pandas) and signed with the same HMAC helpers as the
target / executed books, so it travels the same way.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "sleeve-ledger/v1"

#: how many audit entries to keep in the file (newest last)
MAX_ENTRIES = 400

#: how far the ledger may sit outside the account before a run is refused
DEFAULT_TOL = 0.02


def default_path(report_out: str | Path, account_mode: str = "paper") -> Path:
    """``…/data/overall/executed_book.json`` → ``…/data/overall/sleeve[_live].json``."""
    stem = "sleeve_live" if account_mode == "live" else "sleeve"
    return Path(report_out).parent / f"{stem}.json"


# ════════════════════════════════════════════════════════════════════════════
# BUILD / MUTATE
# ════════════════════════════════════════════════════════════════════════════
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _entry(kind: str, usd: float, cash_after: float, *, as_of: str = "",
           note: str = "") -> dict:
    return {"utc": _now(), "kind": kind, "as_of": as_of,
            "usd": round(float(usd), 6), "cash_after": round(float(cash_after), 6),
            "note": note}


def open_sleeve(*, usd: float, account: str, account_mode: str,
                account_net_liq: float, as_of: str = "",
                note: str = "") -> dict:
    """Create a ledger funded with *usd* of cash.  Day one is 100% cash: the
    first rebalance is what deploys it, so nothing is assumed about positions."""
    usd = float(usd)
    if usd <= 0:
        raise ValueError(f"sleeve stake must be positive (got {usd})")
    if account_net_liq > 0 and usd > account_net_liq:
        raise ValueError(f"stake ${usd:,.2f} exceeds the account's net liq "
                         f"${account_net_liq:,.2f}")
    frac = (usd / account_net_liq) if account_net_liq > 0 else 0.0
    return {
        "schema": SCHEMA,
        "generated_at_utc": _now(),
        "account": account,
        "account_mode": account_mode,
        "inception": {"utc": _now(), "as_of": as_of, "usd": usd,
                      "account_net_liq": float(account_net_liq),
                      "frac_of_account": frac},
        "cash": usd,
        "last_as_of": "",
        "entries": [_entry("inception", usd, usd, as_of=as_of,
                           note=note or f"{frac*100:.1f}% of net liq")],
    }


def fills_cash_delta(trades: list[dict]) -> tuple[float, list[str]]:
    """(cash delta, warnings) from an execution report's trade list.

    Driven by what actually **filled**, never by what was planned: a partial
    fill moves the ledger by the part that filled, and slippage lands in the
    ledger because the fill price is the fill price.  ``PLANNED`` rows from a
    dry-run contribute nothing.
    """
    delta, warn = 0.0, []
    for t in trades or []:
        filled = float(t.get("filled") or 0.0)
        if filled <= 0:
            continue
        px = float(t.get("avg_fill_price") or 0.0)
        if px <= 0:
            px = float(t.get("price") or 0.0)
            if px <= 0:
                warn.append(f"{t.get('symbol')}: {filled:g} filled with no usable "
                            "price — cash NOT adjusted for this leg")
                continue
            warn.append(f"{t.get('symbol')}: no average fill price reported; "
                        f"booked at the sizing price ${px:,.2f}")
        gross = filled * px
        # commissions are a cost either way; IBKR reports them per-execution and
        # the report may or may not carry them through.
        comm = abs(float(t.get("commission") or 0.0))
        if str(t.get("action", "")).upper() == "SELL":
            delta += gross - comm
        else:
            delta -= gross + comm
    return delta, warn


def apply_fills(ledger: dict, trades: list[dict], as_of: str,
                *, force: bool = False) -> tuple[dict, str]:
    """Advance the sleeve's cash by a completed run's fills.  (ledger, reason).

    Refuses to apply the same signal bar twice — the same lock the executed-book
    archive puts on a duplicate run, applied to the money.  A dry-run must never
    reach this function; only a run that really sent orders does.
    """
    out = dict(ledger)
    if not force and as_of and out.get("last_as_of") == as_of:
        return out, (f"sleeve already advanced for signal bar {as_of} — cash "
                     "left unchanged (this run's fills were NOT double-counted)")
    delta, warn = fills_cash_delta(trades)
    cash = float(out.get("cash", 0.0)) + delta
    out["cash"] = cash
    out["last_as_of"] = as_of or out.get("last_as_of", "")
    out["generated_at_utc"] = _now()
    out["entries"] = (list(out.get("entries", []))
                      + [_entry("fills", delta, cash, as_of=as_of,
                                note="; ".join(warn))])[-MAX_ENTRIES:]
    return out, (f"sleeve cash {delta:+,.2f} → ${cash:,.2f}"
                 + (" [" + "; ".join(warn) + "]" if warn else ""))


#: entry kinds that move the sleeve's CAPITAL BASE rather than its P&L
CAPITAL_KINDS = ("inception", "reset", "capital", "adjust")


def adjust(ledger: dict, usd: float, note: str = "",
           min_cash: float | None = None,
           kind: str = "capital") -> tuple[dict, str]:
    """Move cash into (+) or out of (−) the sleeve for a reason a fill cannot
    explain.  Two kinds, and the difference decides what "compounded return"
    means:

    * ``capital`` — a deposit or a withdrawal.  It changes the **base** the
      return is measured against, so topping the sleeve up never flatters it.
    * ``income`` — a dividend, an interest credit, a fee or commission sweep.
      It is **P&L**: the sleeve earned (or lost) it by holding what it holds, so
      it lifts the return exactly the way a price move does.

    Booking a dividend as capital is the subtle way to under-report a strategy
    on dividend-paying ETFs — the cash arrives, the base rises with it, and the
    return does not move.

    ``min_cash`` floors the resulting balance.  Taking money out of a sleeve
    that is fully invested does not free anything — the cash is in the
    positions — so the withdrawal has to be refused rather than booked as a
    balance the sleeve does not have and would then size its next book against.
    """
    if kind not in ("capital", "income"):
        raise ValueError(f"unknown adjustment kind {kind!r}")
    out = dict(ledger)
    if min_cash is not None and float(out.get("cash", 0.0)) + float(usd) < min_cash:
        raise ValueError(
            f"withdrawing ${abs(float(usd)):,.2f} would leave the sleeve at "
            f"${float(out.get('cash', 0.0)) + float(usd):,.2f} of cash (floor "
            f"${min_cash:,.2f}) — it is invested, not idle. Cut the book's "
            "exposure first, or reduce the stake with --sleeve-reset once the "
            "positions are sold.")
    cash = float(out.get("cash", 0.0)) + float(usd)
    out["cash"] = cash
    out["generated_at_utc"] = _now()
    out["entries"] = (list(out.get("entries", []))
                      + [_entry(kind, usd, cash, note=note)])[-MAX_ENTRIES:]
    return out, (f"{kind} {float(usd):+,.2f} → cash ${cash:,.2f}"
                 + ("" if kind == "capital" else " (counts as P&L)"))


# ════════════════════════════════════════════════════════════════════════════
# VALUE / PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════
def positions_value(current: dict[str, float], price: dict[str, float],
                    fallback: dict[str, float] | None = None) -> tuple[float, list[str]]:
    """(value, names priced by nothing) for the sleeve's holdings.

    Marked at the book's ``exec_price`` — the SAME prices the orders are sized
    at, so the sleeve's NAV and its target dollar allocations are internally
    consistent.  *fallback* (the broker's own marks) covers a name that is held
    but no longer in the book, which carries no book price.
    """
    value, unpriced = 0.0, []
    for key, shares in (current or {}).items():
        if not shares:
            continue
        px = float((price or {}).get(key) or 0.0)
        if px <= 0:
            px = float((fallback or {}).get(key) or 0.0)
        if px <= 0:
            unpriced.append(key)
            continue
        value += float(shares) * px
    return value, unpriced


def nav(ledger: dict, current: dict[str, float], price: dict[str, float],
        fallback: dict[str, float] | None = None) -> tuple[float, list[str]]:
    """(sleeve net liquidation value, unpriced names)."""
    value, unpriced = positions_value(current, price, fallback)
    return float(ledger.get("cash", 0.0)) + value, unpriced


def contributed(ledger: dict) -> float:
    """Total capital put IN — inception plus every later deposit, minus
    withdrawals.  The denominator for a return that is not flattered by a
    top-up.  Income entries are deliberately excluded: they are what the sleeve
    earned, not what you gave it.  (``adjust`` is the pre-split kind name and
    still reads as capital, so older ledgers value the same as before.)"""
    total = 0.0
    for e in ledger.get("entries", []):
        if e.get("kind") in CAPITAL_KINDS:
            total += float(e.get("usd") or 0.0)
    return total


def performance(ledger: dict, sleeve_nav: float) -> dict:
    """Compounded P&L of the sleeve since inception, net of contributions."""
    base = contributed(ledger)
    pnl = sleeve_nav - base
    return {"nav": float(sleeve_nav), "cash": float(ledger.get("cash", 0.0)),
            "contributed": base, "pnl": pnl,
            "return_pct": (pnl / base * 100.0) if base > 0 else 0.0,
            "inception": (ledger.get("inception") or {}).get("utc", ""),
            "inception_usd": (ledger.get("inception") or {}).get("usd", 0.0)}


# ════════════════════════════════════════════════════════════════════════════
# GUARDS
# ════════════════════════════════════════════════════════════════════════════
def check_within_account(sleeve_nav: float, sleeve_cash: float,
                         account_net_liq: float, account_cash: float,
                         *, allow_margin: bool = False,
                         tol: float = DEFAULT_TOL) -> tuple[bool, str]:
    """(ok, reason) — is the ledger still a believable SUBSET of the account?

    A sleeve is a claim on part of a real account, and the two can only drift
    apart for reasons worth stopping for: a run whose fills were never booked, a
    withdrawal from the account, a manual trade in the book's tickers, or a
    corrupted ledger.  Sizing a fresh book against a ledger that has drifted is
    how a "10% sleeve" quietly becomes the whole account.
    """
    if sleeve_nav <= 0:
        return False, f"sleeve NAV is ${sleeve_nav:,.2f} — nothing left to trade"
    if account_net_liq <= 0:
        return False, "account net liquidation is zero or unreadable"
    if sleeve_nav > account_net_liq * (1.0 + tol):
        return False, (f"sleeve NAV ${sleeve_nav:,.0f} exceeds the account's "
                       f"${account_net_liq:,.0f} — the ledger has drifted; "
                       "reconcile before trading")
    if not allow_margin and sleeve_cash < -tol * sleeve_nav:
        return False, (f"sleeve cash is ${sleeve_cash:,.2f} — the sleeve is "
                       "carrying a loan against the rest of the account "
                       "(--allow-margin to permit it deliberately)")
    if not allow_margin and sleeve_cash > account_cash + tol * sleeve_nav:
        return False, (f"sleeve claims ${sleeve_cash:,.0f} cash but the account "
                       f"holds ${account_cash:,.0f} — the sleeve cannot fund its "
                       "buys without margin; reconcile before trading")
    return True, (f"sleeve ${sleeve_nav:,.0f} of account ${account_net_liq:,.0f} "
                  f"({sleeve_nav/account_net_liq*100:.1f}%)")


def validate(payload: dict) -> tuple[bool, str]:
    if payload.get("schema") != SCHEMA:
        return False, f"unexpected schema {payload.get('schema')!r} (want {SCHEMA})"
    if "cash" not in payload:
        return False, "ledger has no cash balance"
    if not (payload.get("inception") or {}).get("usd"):
        return False, "ledger has no inception stake"
    return True, "sleeve ledger OK"


def check_account_match(payload: dict, account: str,
                        account_mode: str) -> tuple[bool, str]:
    """A ledger belongs to ONE account.  Pointing a paper ledger at the live
    account (or the reverse) would size real orders off a fictional stake."""
    want_a, want_m = payload.get("account"), payload.get("account_mode")
    if want_a and account and want_a != account:
        return False, (f"ledger belongs to account {want_a}, connected to "
                       f"{account} — refusing to trade")
    if want_m and account_mode and want_m != account_mode:
        return False, (f"ledger is a {want_m} sleeve but this run is "
                       f"{account_mode} — refusing to trade")
    return True, f"ledger matches account {account} ({account_mode})"


# ════════════════════════════════════════════════════════════════════════════
# LOAD / SAVE  (signed with the same helpers as the books)
# ════════════════════════════════════════════════════════════════════════════
def load(path: str | Path) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def save(payload: dict, path: str | Path, secret: str | None = None) -> Path:
    import target_book as tb                     # same HMAC, schema-agnostic
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(tb.dumps(payload, secret))
    return p
