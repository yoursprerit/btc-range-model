"""Market-stress and execution-quality guards for the IBKR executor.

``ibkr_common`` already refuses the wrong-book / wrong-share-count class of
error (stale bars, duplicate runs, levered plans, split-sized orders).  This
module covers the other half: a market that is broken *right now*.  Every
function here is pure (a quote dict or a few numbers in, a reason string or
``None`` out) so each rule can be tested without a broker.

The rules, and what each one is for:

* :func:`quote_problem` — a per-symbol quote sanity check applied just before an
  order is built: halted, crossed, or a spread so wide that a marketable limit
  would pay it.  A failing leg is skipped and reported, never sent blind.
* :func:`fills_tripwire` — after a leg fills, compare each fill with the touch
  the order was priced off.  Fills far through it mean the run is getting bad
  prices, so the remaining legs are not sent.
* :func:`stress_assessment` — a market-wide gate (SPY move, VIX level) plus a
  per-name move check.  In stress the executor de-risks only: sells go,
  buys wait for tomorrow.
* :func:`nav_drop` — the account is far below where the last run left it: same
  de-risk-only treatment.
* :func:`halt_reason` — the kill switch, re-read between orders rather than only
  at start-up.

Every threshold is a flag with an environment fallback and ``0`` disables it.
Defaults are deliberately extreme — they should be silent on an ordinary bad
day and fire only on a genuinely broken one.  Anything that cannot be measured
(no market-data subscription, no VIX feed) passes: these guards add caution,
they never turn a missing data feed into a missed rebalance.
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

# ── defaults (each one: flag --x, env IBKR_X, 0 = off) ───────────────────────
DEFAULT_MAX_SPREAD = 0.02          # buys refuse a bid/ask spread wider than 2%
DEFAULT_SELL_SPREAD_MULT = 2.0     # …sells tolerate twice that: exiting matters more
DEFAULT_MAX_FILL_SLIPPAGE = 0.01   # fills >1% through the touch trip the wire
                                   # (the marketable-limit cap is 0.5%, so this
                                   # only fires on an escalation into a gap)
DEFAULT_STRESS_SPY_MOVE = 0.08     # |SPY vs prior close| beyond 8%
DEFAULT_STRESS_VIX = 50.0          # VIX above 50 (Apr-2025 peak close was ~52)
DEFAULT_STRESS_NAME_MOVE = 0.40    # a single ordered name moved >40% on the day
DEFAULT_MAX_NAV_DROP = 0.15        # net-liq >15% below where the last run left it
BASELINE_MAX_AGE_DAYS = 7          # an older report is not a "yesterday" baseline


def env_float(name: str, default: float) -> float:
    """A float from the environment, falling back to *default* on unset/garbage."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class SafetyLimits:
    """The thresholds plus the two hooks the broker needs at run time."""
    max_spread: float = DEFAULT_MAX_SPREAD
    sell_spread_mult: float = DEFAULT_SELL_SPREAD_MULT
    max_fill_slippage: float = DEFAULT_MAX_FILL_SLIPPAGE
    stress_spy_move: float = DEFAULT_STRESS_SPY_MOVE
    stress_vix: float = DEFAULT_STRESS_VIX
    stress_name_move: float = DEFAULT_STRESS_NAME_MOVE
    max_nav_drop: float = DEFAULT_MAX_NAV_DROP
    outside_rth: bool = False
    # → reason string while trading is halted, else None.  Called between orders.
    halt_check: Callable[[], str | None] | None = field(default=None, repr=False)
    # → delivers a message (webhook + stderr).  Best-effort.
    alert: Callable[[str], None] | None = field(default=None, repr=False)

    @classmethod
    def from_args(cls, args, halt_check=None, alert=None) -> "SafetyLimits":
        s = cls(halt_check=halt_check, alert=alert)
        for f in ("max_spread", "max_fill_slippage", "stress_spy_move",
                  "stress_vix", "stress_name_move", "max_nav_drop"):
            v = getattr(args, f, None)
            if v is not None:
                setattr(s, f, float(v))
        s.outside_rth = bool(getattr(args, "outside_rth", False))
        return s


def add_arguments(ap) -> None:
    """Register the safety flags on an argparse parser (shared by both entry
    points so they expose — and default — the same limits)."""
    g = ap.add_argument_group("market-stress & execution-quality guards (0 = off)")
    g.add_argument("--max-spread", type=float,
                   default=env_float("IBKR_MAX_SPREAD", DEFAULT_MAX_SPREAD),
                   help="skip a BUY whose live bid/ask spread is wider than this "
                        "fraction of the mid, or whose book is crossed (sells "
                        "tolerate %gx). A halted symbol is always skipped "
                        "(default %%(default)s; env IBKR_MAX_SPREAD)"
                        % DEFAULT_SELL_SPREAD_MULT)
    g.add_argument("--max-fill-slippage", type=float,
                   default=env_float("IBKR_MAX_FILL_SLIPPAGE",
                                     DEFAULT_MAX_FILL_SLIPPAGE),
                   help="after the sell leg, if any fill is this far THROUGH the "
                        "touch it was priced off, send no buys and alert "
                        "(default %(default)s; env IBKR_MAX_FILL_SLIPPAGE)")
    g.add_argument("--stress-spy-move", type=float,
                   default=env_float("IBKR_STRESS_SPY_MOVE",
                                     DEFAULT_STRESS_SPY_MOVE),
                   help="stress gate: |SPY vs prior close| beyond this blocks "
                        "all buys (sells still go) (default %(default)s; env "
                        "IBKR_STRESS_SPY_MOVE)")
    g.add_argument("--stress-vix", type=float,
                   default=env_float("IBKR_STRESS_VIX", DEFAULT_STRESS_VIX),
                   help="stress gate: VIX above this blocks all buys "
                        "(default %(default)s; env IBKR_STRESS_VIX)")
    g.add_argument("--stress-name-move", type=float,
                   default=env_float("IBKR_STRESS_NAME_MOVE",
                                     DEFAULT_STRESS_NAME_MOVE),
                   help="stress gate: skip the BUY of any name that has moved "
                        "more than this vs its prior close (default "
                        "%(default)s; env IBKR_STRESS_NAME_MOVE)")
    g.add_argument("--max-nav-drop", type=float,
                   default=env_float("IBKR_MAX_NAV_DROP", DEFAULT_MAX_NAV_DROP),
                   help="de-risk only (no buys) when net liquidation is more "
                        "than this fraction below where the last execution "
                        "report left it (default %(default)s; env "
                        "IBKR_MAX_NAV_DROP). A large withdrawal trips it too — "
                        "the alert says so")


# ── small helpers ────────────────────────────────────────────────────────────
def _pos(x) -> float | None:
    """A usable positive number, or None (IBKR reports absent fields as nan/-1)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0 else None


def touch_price(action: str, quote: dict) -> float | None:
    """The side of the book an order must cross (ask for BUY, bid for SELL),
    falling back to last — the reference a fill's slippage is measured from."""
    side = "ask" if action == "BUY" else "bid"
    for field_ in (side, "last"):
        v = _pos(quote.get(field_))
        if v is not None:
            return v
    return None


def send_alert(msg: str) -> None:
    """Webhook + stderr; never raises."""
    try:
        from ibkr_gateway_healthcheck import send_alert as _send
        _send(msg, os.environ.get("IBKR_ALERT_WEBHOOK"))
    except Exception:
        print(msg, file=sys.stderr, flush=True)


# ── 2. per-symbol quote sanity ───────────────────────────────────────────────
def quote_problem(quote: dict, action: str, max_spread: float,
                  sell_mult: float = DEFAULT_SELL_SPREAD_MULT,
                  outside_rth: bool = False) -> str | None:
    """Why *quote* is not safe to trade against, or None.

    An empty or partial quote is "unknown", not "bad": the pricing path already
    degrades on its own when a name cannot be quoted, and these checks must not
    turn a missing market-data subscription into a skipped rebalance.

    Outside regular hours books are routinely crossed and wide, so only a trading
    halt counts there.
    """
    if not quote:
        return None
    halted = quote.get("halted")
    try:
        if halted is not None and math.isfinite(float(halted)) and float(halted) > 0:
            kind = {1: "general halt", 2: "volatility halt"}.get(int(float(halted)),
                                                                  "halt")
            return f"trading halted ({kind})"
    except (TypeError, ValueError):
        pass
    if outside_rth or max_spread <= 0:
        return None
    bid, ask = _pos(quote.get("bid")), _pos(quote.get("ask"))
    if bid is None or ask is None:
        return None
    if ask < bid:
        return f"crossed book (bid {bid:,.2f} > ask {ask:,.2f})"
    spread = (ask - bid) / ((ask + bid) / 2.0)
    limit = max_spread * (sell_mult if action == "SELL" else 1.0)
    if spread > limit:
        return (f"spread {spread*100:.2f}% (bid {bid:,.2f} / ask {ask:,.2f}) "
                f"is wider than the {limit*100:.2f}% limit")
    return None


# ── 5. fill-quality tripwire ─────────────────────────────────────────────────
def adverse_slippage(action: str, ref: float, avg_fill: float) -> float:
    """Fraction by which a fill was WORSE than *ref* (positive = bad for us)."""
    if not (ref and avg_fill) or ref <= 0 or avg_fill <= 0:
        return 0.0
    return (avg_fill - ref) / ref if action == "BUY" else (ref - avg_fill) / ref


def fills_tripwire(results: list[dict], refs: dict[tuple[str, str], float],
                   max_slip: float) -> str | None:
    """Reason the run should stop sending orders, or None.

    *refs* maps ``(action, symbol)`` to the touch price the order was priced off.
    Only fills with a recorded reference are judged; a MOC fill is never judged
    (it prints at the auction price by design)."""
    if max_slip <= 0:
        return None
    worst: tuple[float, str] = (0.0, "")
    for r in results:
        if r.get("order_type") == "moc" or float(r.get("filled") or 0) <= 0:
            continue
        ref = refs.get((r.get("action"), r.get("symbol")))
        slip = adverse_slippage(r.get("action"), ref or 0.0,
                                float(r.get("avg_fill_price") or 0.0))
        if slip > worst[0]:
            worst = (slip, f"{r['action']} {r['symbol']} filled at "
                           f"${float(r['avg_fill_price']):,.2f} against a "
                           f"${ref:,.2f} touch ({slip*100:.2f}% adverse)")
    if worst[0] > max_slip:
        return f"{worst[1]} — beyond the {max_slip*100:.2f}% fill-quality limit"
    return None


# ── 1. stress gate ───────────────────────────────────────────────────────────
def day_move(quote: dict) -> float | None:
    """Signed move of the live price versus the prior close, or None."""
    close = _pos(quote.get("close"))
    bid, ask = _pos(quote.get("bid")), _pos(quote.get("ask"))
    live = _pos(quote.get("last"))
    if live is None and bid is not None and ask is not None:
        live = (bid + ask) / 2.0
    if close is None or live is None:
        return None
    return live / close - 1.0


@dataclass
class Stress:
    block_buys: str | None = None                 # market-wide reason, or None
    names: dict[str, str] = field(default_factory=dict)   # symbol -> reason
    notes: list[str] = field(default_factory=list)        # what was measured

    def buy_block_for(self, symbol: str) -> str | None:
        return self.block_buys or self.names.get(symbol)


def stress_assessment(spy_quote: dict | None, vix_quote: dict | None,
                      name_quotes: dict[str, dict], lim: SafetyLimits) -> Stress:
    """Evaluate the market-wide and per-name stress rules on already-fetched
    quotes (the broker does the fetching).  Unmeasurable inputs are noted and
    skipped — see the module docstring."""
    out = Stress()
    reasons: list[str] = []
    if lim.stress_spy_move > 0:
        mv = day_move(spy_quote or {})
        if mv is None:
            out.notes.append("SPY move not measured (no quote)")
        else:
            out.notes.append(f"SPY {mv*100:+.2f}% on the day")
            if abs(mv) > lim.stress_spy_move:
                reasons.append(f"SPY {mv*100:+.1f}% on the day "
                               f"(limit ±{lim.stress_spy_move*100:.0f}%)")
    if lim.stress_vix > 0:
        vix = None
        for k in ("last", "close"):
            vix = _pos((vix_quote or {}).get(k))
            if vix is not None:
                break
        if vix is None:
            out.notes.append("VIX not measured (no feed)")
        else:
            out.notes.append(f"VIX {vix:.1f}")
            if vix > lim.stress_vix:
                reasons.append(f"VIX {vix:.1f} (limit {lim.stress_vix:g})")
    if reasons:
        out.block_buys = "market stress — " + "; ".join(reasons)
    if lim.stress_name_move > 0:
        for symbol, q in name_quotes.items():
            mv = day_move(q or {})
            if mv is not None and abs(mv) > lim.stress_name_move:
                out.names[symbol] = (f"{symbol} {mv*100:+.1f}% on the day "
                                     f"(limit ±{lim.stress_name_move*100:.0f}%)")
    return out


# ── 6. NAV brake + kill switch ───────────────────────────────────────────────
def read_baseline_net_liq(report_path, now: datetime | None = None,
                          max_age_days: int = BASELINE_MAX_AGE_DAYS
                          ) -> float | None:
    """Net liquidation recorded by the previous execution report at *report_path*,
    or None when there is none, it is unreadable, or it is too old to mean
    "where the last run left the account"."""
    try:
        data = json.loads(Path(report_path).read_text())
        nl = _pos(data.get("net_liq"))
        stamp = datetime.fromisoformat(str(data.get("generated_at_utc")))
    except Exception:
        return None
    if nl is None:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    if (now - stamp).total_seconds() > max_age_days * 86400:
        return None
    return nl


def nav_drop(net_liq: float, baseline: float | None, max_drop: float
             ) -> str | None:
    """Reason to go de-risk-only, or None."""
    if max_drop <= 0 or not baseline or baseline <= 0 or not net_liq or net_liq <= 0:
        return None
    drop = 1.0 - net_liq / baseline
    if drop > max_drop:
        return (f"net liquidation ${net_liq:,.0f} is {drop*100:.1f}% below the "
                f"${baseline:,.0f} the last run left (limit {max_drop*100:.0f}%) "
                "— a crash, or a large withdrawal")
    return None


def halt_reason(kill_switch_file: str | None) -> str | None:
    """The kill switch, exactly as the start-up check reads it."""
    if os.environ.get("IBKR_TRADING_DISABLED"):
        return "IBKR_TRADING_DISABLED is set"
    if kill_switch_file and Path(kill_switch_file).exists():
        return f"kill-switch file {kill_switch_file} exists"
    return None


def arm(broker, args, net_liq: float, report_path=None,
        kill_switch_file: str | None = None) -> None:
    """Attach the safety limits to *broker* for this run, and apply the NAV
    brake.  Both entry points call this right after reading net liquidation."""
    broker.safety = SafetyLimits.from_args(
        args,
        halt_check=(lambda: halt_reason(kill_switch_file)),
        alert=send_alert)
    if report_path:
        why = nav_drop(net_liq, read_baseline_net_liq(report_path),
                       broker.safety.max_nav_drop)
        if why:
            broker.buy_block = f"NAV brake — {why}"
            send_alert(f"⚠️ {getattr(broker, 'account', '?')}: {why}. "
                       "Running de-risk only: buys will be withheld, sells proceed.")
