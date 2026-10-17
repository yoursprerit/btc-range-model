"""Dollar P&L for one IBKR account's "as-published record", with monthly deposits.

The as-published record is a TIME-WEIGHTED series: it compounds the daily target
books, so adding cash to an account never moves it.  What a deposit DOES change
is the dollar figures — "portfolio value" and "P&L $" scaled from one fixed
starting amount are wrong once money keeps arriving.  This module replays the
account in dollars from the day it first traded the book:

    value_t = value_{t-1} × (curve_t / curve_{t-1}) + deposit_t

A deposit lands at that day's CLOSE (after the day's return) and so earns from
the next bar.  Dollar P&L = value − starting value − deposits made since.

Pure functions, no Streamlit — unit-tested in ``tests/test_account_pnl.py``.
The user supplies, per account, a recurring monthly amount and the day of the
month it arrives (``data/overall/account_deposits.json`` holds saved defaults).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import executed_book as eb

MAX_DEPOSIT_DAY = 28          # a day every month has — "the 31st" would skip months


# ── what the account has actually executed ──────────────────────────────────
def record_info(report_path) -> dict | None:
    """What an account's executed record says, or None when it has none.

    Only an EXECUTED run counts (``eb.archived_records`` holds those alone — a
    dry-run archives nothing), so an account that has only ever been previewed
    stays hidden.  ``start`` is the New-York date of the FIRST execution: the
    day the account began holding the book, and so the first day its record can
    be measured from."""
    recs = eb.archived_records(report_path)          # newest first
    if not recs:
        return None
    first, last = recs[-1], recs[0]

    def _nl(r):
        try:
            v = float(r["payload"].get("net_liq") or 0.0)
        except (TypeError, ValueError):
            return 0.0
        return v if v > 0 else 0.0
    return dict(account=eb.account_id_of(report_path) or "",
                start=pd.Timestamp(first["executed_on"]).normalize(),
                first_as_of=first["as_of"], first_net_liq=_nl(first),
                last_executed_on=pd.Timestamp(last["executed_on"]).normalize(),
                last_net_liq=_nl(last), n_runs=len(recs),
                holdings=holdings_from_records(recs))


def holdings_from_records(recs: list[dict]) -> list[dict]:
    """What the account held after each executed run, oldest first.

    ``recs`` is ``eb.archived_records`` output (newest first).  Each entry is
    ``{date, net_liq, cash, shares, price}``: ``shares``/``price`` by sleeve
    key from the run's positions, and ``cash`` = net liquidation − the
    positions' market value.  The report's own ``TotalCashValue`` tag is a
    separate read that can lag an MOC fill, so the implied figure is the one
    that makes the account add up.  A run with no usable net liquidation is
    skipped."""
    out = []
    for r in reversed(recs):
        pl = r["payload"]
        try:
            nl = float(pl.get("net_liq") or 0.0)
        except (TypeError, ValueError):
            continue
        if nl <= 0:
            continue
        shares, price, held = {}, {}, 0.0
        for p in pl.get("positions") or []:
            try:
                key = str(p.get("key") or p.get("symbol"))
                q, px = float(p["shares"]), float(p["market_price"])
            except (KeyError, TypeError, ValueError):
                continue
            if q == 0 or not np.isfinite(px) or px <= 0:
                continue
            shares[key] = shares.get(key, 0.0) + q
            price[key] = px
            held += q * px
        out.append(dict(date=pd.Timestamp(r["executed_on"]).normalize(),
                        net_liq=nl, cash=nl - held, shares=shares, price=price))
    return out


def executed_marks(report_paths, keys=("SATA",)) -> dict[str, pd.Series]:
    """Daily price marks for sleeves the close table does not carry (SATA),
    read off the executed IBKR reports: ``{key: Series}`` indexed by the
    New-York date of each run, the median across accounts when several ran
    the same day.  These are the run-time marks of the same security another
    account held, so a broker whose own prices are unavailable (Collective2)
    can value that position at something other than its fill price."""
    rows: dict[str, dict] = {k: {} for k in keys}
    for rp in report_paths:
        try:
            recs = eb.archived_records(rp)
        except Exception:
            continue
        for r in recs:
            day = pd.Timestamp(r["executed_on"]).normalize()
            for p in (r["payload"].get("positions") or []):
                k = str(p.get("key") or p.get("symbol"))
                try:
                    px = float(p["market_price"])
                except (KeyError, TypeError, ValueError):
                    continue
                if k in rows and np.isfinite(px) and px > 0:
                    rows[k].setdefault(day, []).append(px)
    return {k: pd.Series({d: float(np.median(v)) for d, v in sorted(m.items())},
                         dtype=float)
            for k, m in rows.items() if m}


def load_deposit_defaults(path) -> dict[str, dict]:
    """Saved per-account deposit settings: ``{"U123": {"monthly": 500, "day": 1}}``.
    A missing or unreadable file means "no deposits", never an error."""
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    out = {}
    for acct, v in (raw.items() if isinstance(raw, dict) else []):
        if not isinstance(v, dict) or str(acct).startswith("_"):
            continue
        try:
            out[str(acct)] = dict(monthly=max(0.0, float(v.get("monthly", 0.0))),
                                  day=clamp_day(v.get("day", 1)))
        except (TypeError, ValueError):
            continue
    return out


def clamp_day(day) -> int:
    try:
        d = int(day)
    except (TypeError, ValueError):
        d = 1
    return min(max(d, 1), MAX_DEPOSIT_DAY)


# ── the deposit schedule ────────────────────────────────────────────────────
def deposit_bars(index: pd.DatetimeIndex, first_bar: pd.Timestamp, day: int
                 ) -> list[pd.Timestamp]:
    """The bar each monthly deposit lands on.

    One deposit per calendar month, on *day* — rolled FORWARD to the first bar
    on or after that date (a weekend or holiday deposit is credited the next
    session).  The month of *first_bar* has none: the account was funded at its
    start, and the record opens with that balance.  A month whose deposit date
    falls after the last bar is not yet due."""
    idx = pd.DatetimeIndex(index)
    if len(idx) == 0:
        return []
    day = clamp_day(day)
    first_bar = pd.Timestamp(first_bar)
    out: list[pd.Timestamp] = []
    month = (first_bar + pd.offsets.MonthBegin(1)).normalize()   # next month's 1st
    last = idx[-1]
    while month <= last:
        due = month + pd.Timedelta(days=day - 1)
        pos = idx.searchsorted(due)                     # first bar on/after due
        if pos < len(idx):
            out.append(idx[pos])        # (two months may share a bar after a gap)
        month = month + pd.offsets.MonthBegin(1)
    return out


def simulate_account(curve: pd.Series, start, initial: float, monthly: float,
                     day: int = 1) -> pd.DataFrame:
    """The account in dollars from its first bar on/after *start*.

    *curve* is any positive equity series (the as-published record); only its
    day-to-day ratios matter.  Returns a frame indexed by bar with ``value``
    (account value at the close, deposit included), ``deposit`` and
    ``contributed`` (starting value + every deposit so far)."""
    c = curve.dropna()
    c = c[c > 0]
    c = c.loc[pd.Timestamp(start):]
    if len(c) == 0 or initial <= 0:
        return pd.DataFrame(columns=["value", "deposit", "contributed"], dtype=float)
    dep = pd.Series(0.0, index=c.index)
    if monthly > 0:
        for bar in deposit_bars(c.index, c.index[0], day):
            dep.loc[bar] += float(monthly)
    ratio = (c / c.shift(1)).fillna(1.0).to_numpy()
    vals = np.empty(len(c))
    v = float(initial)
    for i in range(len(c)):
        v = v * ratio[i] + dep.iloc[i]
        vals[i] = v
    out = pd.DataFrame({"value": vals, "deposit": dep.to_numpy()}, index=c.index)
    out["contributed"] = float(initial) + out["deposit"].cumsum()
    return out


def window_summary(sim: pd.DataFrame, start, end=None) -> dict | None:
    """P&L over [start, end] of a simulated account, deposits taken out.

    The first bar on/after *start* is the anchor, as everywhere else on the P&L
    page: its closing value is the starting balance, and deposits count only
    from the NEXT bar on.  ``pnl = value_end − value_start − deposits``."""
    if sim is None or len(sim) == 0:
        return None
    s = sim.loc[pd.Timestamp(start):]
    if end is not None:
        s = s.loc[:pd.Timestamp(end)]
    if len(s) < 1:
        return None
    v0, v1 = float(s["value"].iloc[0]), float(s["value"].iloc[-1])
    deposits = float(s["deposit"].iloc[1:].sum())
    pnl = v1 - v0 - deposits
    base = v0 + deposits
    return dict(start=s.index[0], end=s.index[-1], start_value=v0, end_value=v1,
                deposits=deposits, n_deposits=int((s["deposit"].iloc[1:] > 0).sum()),
                contributed=base, pnl=pnl, pnl_pct_contributed=(pnl / base if base else 0.0),
                mwr=xirr(s))


def xirr(s: pd.DataFrame) -> float | None:
    """Money-weighted (internal-rate) annualised return of a window, or None
    when it cannot be solved or is too short to mean anything."""
    if s is None or len(s) < 2:
        return None
    d0 = s.index[0]
    flows = [(d0, -float(s["value"].iloc[0]))]
    flows += [(t, -float(a)) for t, a in s["deposit"].iloc[1:].items() if a > 0]
    flows.append((s.index[-1], float(s["value"].iloc[-1])))
    if (s.index[-1] - d0).days < 30:
        return None
    t = np.array([(f[0] - d0).days / 365.25 for f in flows])
    a = np.array([f[1] for f in flows])

    def npv(r):
        return float(np.sum(a / (1.0 + r) ** t))
    lo, hi = -0.99, 10.0
    try:
        f_lo, f_hi = npv(lo), npv(hi)
        if f_lo * f_hi > 0:
            return None
        for _ in range(200):
            mid = (lo + hi) / 2
            f_mid = npv(mid)
            if f_lo * f_mid <= 0:
                hi = mid
            else:
                lo, f_lo = mid, f_mid
        return (lo + hi) / 2
    except (OverflowError, ZeroDivisionError, FloatingPointError):
        return None
