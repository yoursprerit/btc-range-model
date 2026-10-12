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
DEFAULT_MIN_MARGIN_CUSHION = 0.05  # excess liquidity under 5% of net-liq: don't trade
DEFAULT_MAX_BOOK_ADD = 0.60        # >60% of NAV of NEW target weight in one day: suspicious
DEFAULT_SOFT_PRICE_DRIFT = 0.10    # unlevered name 10%+ off its book price: hold its buy
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
    min_margin_cushion: float = DEFAULT_MIN_MARGIN_CUSHION
    soft_price_drift: float = DEFAULT_SOFT_PRICE_DRIFT
    outside_rth: bool = False
    # → reason string while trading is halted, else None.  Called between orders.
    halt_check: Callable[[], str | None] | None = field(default=None, repr=False)
    # → delivers a message (webhook + stderr).  Best-effort.
    alert: Callable[[str], None] | None = field(default=None, repr=False)

    @classmethod
    def from_args(cls, args, halt_check=None, alert=None) -> "SafetyLimits":
        s = cls(halt_check=halt_check, alert=alert)
        for f in ("max_spread", "max_fill_slippage", "stress_spy_move",
                  "stress_vix", "stress_name_move", "max_nav_drop",
                  "min_margin_cushion", "soft_price_drift"):
            v = getattr(args, f, None)
            if v is not None:
                setattr(s, f, float(v))
        s.outside_rth = bool(getattr(args, "outside_rth", False))
        return s


def add_arguments(ap, book_checks: bool = True) -> None:
    """Register the safety flags on an argparse parser (shared by both entry
    points so they expose — and default — the same limits).  *book_checks* adds
    the published-book flags; the paper rebalancer builds its own book and has
    no previous one to compare with, so it leaves them out."""
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
    g.add_argument("--min-margin-cushion", type=float,
                   default=env_float("IBKR_MIN_MARGIN_CUSHION",
                                     DEFAULT_MIN_MARGIN_CUSHION),
                   help="abort the run, with an alert, when the account's excess "
                        "liquidity is at or below zero (a margin deficiency — "
                        "IBKR may be liquidating) or under this fraction of net "
                        "liquidation, so the executor never trades against the "
                        "broker's own risk system (default %(default)s; env "
                        "IBKR_MIN_MARGIN_CUSHION; unreadable = pass)")
    g.add_argument("--soft-price-drift", type=float,
                   default=env_float("IBKR_SOFT_PRICE_DRIFT",
                                     DEFAULT_SOFT_PRICE_DRIFT),
                   help="hold the BUY (sells proceed, the run is NOT aborted) of "
                        "an unlevered ETF whose live price is this far from the "
                        "book's sizing price; leveraged, single-stock and crypto "
                        "names are exempt. The hard --max-price-drift abort is "
                        "unchanged (default %(default)s; env IBKR_SOFT_PRICE_DRIFT)")
    if not book_checks:
        return
    g.add_argument("--expected-profile", default=os.environ.get(
                       "IBKR_EXPECTED_PROFILE", "").strip() or None,
                   help="abort unless the published book is for this risk profile "
                        "(e.g. Balanced). Without it a profile switch at the "
                        "publisher silently re-targets the account (env "
                        "IBKR_EXPECTED_PROFILE)")
    g.add_argument("--max-book-add", type=float,
                   default=env_float("IBKR_MAX_BOOK_ADD", DEFAULT_MAX_BOOK_ADD),
                   help="when today's book ADDS more than this fraction of NAV of "
                        "new target weight versus the last book this account saw "
                        "(or the profile changed), hold the buys for a day and "
                        "alert; cuts are never held (default %(default)s; env "
                        "IBKR_MAX_BOOK_ADD; 0 disables)")
    g.add_argument("--accept-large-change", action="store_true",
                   default=os.environ.get("IBKR_ACCEPT_LARGE_CHANGE", "") == "1",
                   help="trade today's book even though it trips --max-book-add "
                        "(env IBKR_ACCEPT_LARGE_CHANGE=1). Meant for ONE day — "
                        "unset it afterwards")


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
    spy_halt = (spy_quote or {}).get("halted")
    try:
        if spy_halt is not None and math.isfinite(float(spy_halt)) and float(spy_halt) > 0:
            reasons.append("SPY is halted (market-wide halt?)")
    except (TypeError, ValueError):
        pass
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


# ── account health / run outcome ─────────────────────────────────────────────
def margin_problem(excess_liquidity, net_liq: float, min_cushion: float
                   ) -> str | None:
    """Why the account is too close to (or in) a margin call to trade, or None.

    Measured on ExcessLiquidity — the figure IBKR itself liquidates on — as a
    share of net liquidation.  A cash or IRA account with no leverage sits near
    100 %.  An unreadable figure is "unknown", never a reason to stop."""
    if min_cushion <= 0:
        return None
    try:
        ex = float(excess_liquidity)
        nl = float(net_liq)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(ex) and math.isfinite(nl)) or nl <= 0:
        return None
    if ex <= 0:
        return (f"excess liquidity is ${ex:,.0f} — a margin deficiency; IBKR may "
                "be liquidating positions and this run must not fight it")
    if ex / nl < min_cushion:
        return (f"excess liquidity ${ex:,.0f} is only {ex/nl*100:.1f}% of net "
                f"liquidation (minimum {min_cushion*100:.0f}%) — close to a "
                "margin call")
    return None


def nothing_sent(results: list[dict], planned: int) -> str | None:
    """Reason to call the run out loud when orders were planned but not one was
    actually sent (every leg withheld or skipped), else None."""
    if planned <= 0:
        return None
    sent = [r for r in results if not str(r.get("status", "")).startswith("SKIPPED")]
    if sent:
        return None
    kinds = sorted({str(r.get("status", "")) for r in results})
    return (f"{planned} order(s) were planned but NONE was sent "
            f"({', '.join(kinds) or 'no legs reported'})")


# ── published-book change guard ──────────────────────────────────────────────
# The signature proves the book came from the publisher; it cannot prove the
# publisher was right.  These compare today's book with the last one THIS
# account saw: a profile switch, or a jump in new target weight, is held for a
# day (buys only) and reported, so a corrupted or mis-generated book that is
# still arithmetically valid cannot buy a large position unseen.  Cuts are never
# held — a stop-out that moves to cash is the safe direction and must go through.
def read_book_state(path) -> dict | None:
    try:
        d = json.loads(Path(path).read_text())
        return d if isinstance(d, dict) else None
    except Exception:
        return None


def _clean(weights: dict) -> dict[str, float]:
    out = {}
    for k, v in (weights or {}).items():
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out[str(k)] = f
    return out


def book_change(state: dict | None, as_of: str, profile: str, weights: dict
                ) -> dict | None:
    """How today's book differs from the last distinct book this account saw, or
    None when there is nothing to compare with.  A re-run of the same ``as_of``
    is compared with the book BEFORE it, so a second run cannot slip past a
    first run's hold."""
    if not state or not as_of:
        return None
    if state.get("as_of") == as_of:
        base_w, base_p, base_d = (state.get("prev_weights"),
                                  state.get("prev_profile"), state.get("prev_as_of"))
    else:
        base_w, base_p, base_d = (state.get("weights"), state.get("profile"),
                                  state.get("as_of"))
    if not isinstance(base_w, dict):
        return None
    old, new = _clean(base_w), _clean(weights)
    keys = set(old) | set(new)
    adds = sum(max(new.get(k, 0.0) - old.get(k, 0.0), 0.0) for k in keys)
    cuts = sum(max(old.get(k, 0.0) - new.get(k, 0.0), 0.0) for k in keys)
    return dict(adds=adds, cuts=cuts, base_as_of=base_d, base_profile=base_p,
                new_profile=profile,
                profile_changed=bool(base_p and profile and base_p != profile))


def next_book_state(state: dict | None, as_of: str, profile: str, weights: dict
                    ) -> dict:
    """The state to store after seeing this book (keeps the previous distinct
    book so a same-day re-run is still judged against it)."""
    cur = dict(as_of=as_of, profile=profile or "", weights=_clean(weights))
    if state and state.get("as_of") == as_of:
        cur.update(prev_as_of=state.get("prev_as_of"),
                   prev_profile=state.get("prev_profile"),
                   prev_weights=state.get("prev_weights"))
    elif state and state.get("as_of"):
        cur.update(prev_as_of=state.get("as_of"), prev_profile=state.get("profile"),
                   prev_weights=state.get("weights"))
    return cur


def write_book_state(path, state: dict) -> bool:
    """Atomic best-effort write; a failure must never stop a run."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(state, indent=1, sort_keys=True))
        os.replace(tmp, p)
        return True
    except Exception:
        return False


def book_state_path(report_path) -> Path:
    """Dot-named, so the report vault (which skips dotfiles) never seals it."""
    p = Path(report_path)
    return p.with_name("." + p.stem + ".bookstate.json")


def book_hold(change: dict | None, max_add: float, accepted: bool,
              profile_pinned: bool) -> tuple[str | None, list[str]]:
    """(reason to hold this run's BUYS or None, notes for the log/alert).

    A pinned profile that matches makes a profile change impossible, so the
    profile rule only applies to an unpinned account."""
    if max_add <= 0:
        return None, ["book check disabled"]
    if not change:
        return None, ["no previous book on record for this account — nothing to compare"]
    notes = [f"book vs {change['base_as_of']}: adds {change['adds']*100:.1f}% / "
             f"cuts {change['cuts']*100:.1f}% of NAV"]
    reasons = []
    if change["profile_changed"] and not profile_pinned:
        reasons.append(f"profile changed {change['base_profile']} → "
                       f"{change.get('new_profile')}")
    if change["adds"] > max_add:
        reasons.append(f"the book adds {change['adds']*100:.0f}% of NAV in new "
                       f"target weight (limit {max_add*100:.0f}%)")
    if reasons and accepted:
        notes.append("large change ACCEPTED by --accept-large-change / "
                     "IBKR_ACCEPT_LARGE_CHANGE — unset it after today")
        return None, notes
    if reasons:
        return "book check — " + "; ".join(reasons), notes
    if change["cuts"] > max_add:
        notes.append(f"large de-risk ({change['cuts']*100:.0f}% of NAV cut) — "
                     "proceeding; sells are never held")
    return None, notes
