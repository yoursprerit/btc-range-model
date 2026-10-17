"""After-the-close reconcile for the live multi-account MOC run.

    python scripts/ibkr_reconcile_accounts.py            # what the 4:15 PM ET task runs
    python scripts/ibkr_reconcile_accounts.py --dry-run  # look, change nothing

The 2:20 PM run sends the orders and then waits ~30 minutes through the 4:00 PM
closing auction in the SAME process to write the execution report.  If that PC
sleeps, the gateway drops or Windows restarts in those 30 minutes, the orders are
fine (they live at IBKR) but the report is never written -- and nobody knows
whether everything filled.  This job closes that gap.  For every account that had
orders sent today (the send-lock written at transmit time) it asks the broker what
really happened and checks:

  * the execution report for the bar exists -- if not, it is REBUILT from the
    broker's fills (and the alert says the run did not finish);
  * every planned leg filled -- unfilled / partly filled legs are reported;
  * the report matches the broker's fills -- late fills are written into it;
  * no order is still working at the broker.

Rewrites are surgical (planned reasons and order types are kept; only fills are
updated) and are skipped when the broker shows NO executions at all while the
report lists trades -- IBKR only serves the current session's fills, so an empty
answer must never overwrite a real record.  Nothing is ever re-sent or cancelled.

Exit codes: 0 all reconciled · 1 a problem to look at · 2 gateway / config
problem · 3 nothing to do (no orders were sent today, or the auction has not
printed yet).  A heartbeat goes to ``IBKR_RECONCILE_HEARTBEAT_URL`` when set.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "app"))

import pandas as pd                                    # noqa: E402

import executed_book as eb                             # noqa: E402
import ibkr_common as ic                               # noqa: E402
import ibkr_execute_accounts as acc                    # noqa: E402
import ibkr_heartbeat as hbeat                         # noqa: E402

EXIT_OK, EXIT_PROBLEM, EXIT_SETUP, EXIT_NOTHING = 0, 1, 2, 3
CLIENT_ID_OFFSET = 60            # keeps clear of the run's ids (base + index) and the check's
AUCTION_SETTLE_MIN = 5           # wait this long past the close before judging fills
FILL_TOL = 0.5                   # shares; whole-share orders, so half a share is "equal"


# ── pure helpers (unit-tested without a broker) ─────────────────────────────
@dataclass
class Finding:
    account: str
    as_of: str = ""
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    rewritten: bool = False

    @property
    def ok(self) -> bool:
        return not self.problems


def todays_locks(accounts, now_et: pd.Timestamp) -> dict[str, str]:
    """``{account id: signal bar}`` for accounts whose orders were transmitted
    today (New York date), read from the send-lock written at transmit time.
    With several locks for the day (a forced re-run) the newest wins."""
    out: dict[str, tuple[str, str]] = {}
    for a in accounts:
        lock_dir = acc.account_dir(a.id) / ".send_locks"
        if not lock_dir.is_dir():
            continue
        for p in lock_dir.glob("*.lock"):
            try:
                d = json.loads(p.read_text())
                when = pd.Timestamp(d["locked_at_utc"])
                if when.tzinfo is None:
                    when = when.tz_localize("UTC")
                if when.tz_convert(ic.ET_TZ).date() != now_et.date():
                    continue
                as_of = str(d.get("as_of") or p.stem)
            except (OSError, ValueError, KeyError):
                continue
            if a.id not in out or when.isoformat() > out[a.id][1]:
                out[a.id] = (as_of, when.isoformat())
    return {k: v[0] for k, v in out.items()}


def fills_map(fills: list[dict]) -> dict[tuple[str, str], tuple[float, float]]:
    """``{(symbol, action): (filled, avg price)}`` from the broker's fills."""
    return {(f["symbol"], f["action"]): (float(f.get("filled", 0.0) or 0.0),
                                         float(f.get("avg_fill_price", 0.0) or 0.0))
            for f in fills}


def merge_trades(prior_trades: list[dict], fills: list[dict]) -> list[dict]:
    """The report's trade list brought in line with the broker's fills.

    Planned legs keep everything they had (reason, order type, planned qty);
    their ``filled`` / ``avg_fill_price`` / ``status`` come from the broker.
    Legs the plan skipped (``SKIPPED-*``) are left as they were.  A fill with no
    planned leg (a manual trade the same day) is appended and labelled."""
    fm = fills_map(fills)
    used, out = set(), []
    for t in prior_trades:
        t = dict(t)
        if not str(t.get("status", "")).startswith("SKIPPED"):
            key = (t.get("symbol"), t.get("action"))
            got, avg = fm.get(key, (0.0, 0.0))
            used.add(key)
            t["filled"], t["avg_fill_price"] = got, avg
            want = float(t.get("qty", 0.0) or 0.0)
            t["status"] = ("Filled" if got + FILL_TOL >= want and want > 0
                           else "Partial" if got > 0 else "Unfilled")
        out.append(t)
    for f in fills:
        key = (f["symbol"], f["action"])
        if key not in used:
            f = dict(f)
            f["reason"] = "fill with no planned order (reconstructed from broker fills)"
            out.append(f)
    return out


def unfilled_legs(trades: list[dict], fills: list[dict]) -> list[str]:
    """Human-readable lines for planned legs that did not fully fill."""
    fm = fills_map(fills)
    gaps = []
    for t in trades:
        if str(t.get("status", "")).startswith("SKIPPED"):
            continue
        want = float(t.get("qty", 0.0) or 0.0)
        got = fm.get((t.get("symbol"), t.get("action")), (0.0, 0.0))[0]
        if want > 0 and got + FILL_TOL < want:
            gaps.append(f"{t.get('action')} {t.get('symbol')}: {got:g} of {want:g} filled")
    return gaps


def needs_rewrite(prior_trades: list[dict], fills: list[dict]) -> bool:
    """Does the report disagree with the broker about what was filled?"""
    fm = fills_map(fills)
    for t in prior_trades:
        if str(t.get("status", "")).startswith("SKIPPED"):
            continue
        got = fm.get((t.get("symbol"), t.get("action")), (0.0, 0.0))[0]
        if abs(float(t.get("filled", 0.0) or 0.0) - got) > FILL_TOL:
            return True
    planned = {(t.get("symbol"), t.get("action")) for t in prior_trades}
    return any(k not in planned for k in fm)


def describe_working(trades) -> str:
    out = []
    for t in trades[:5]:
        qty = getattr(t.order, "totalQuantity", None)
        out.append(f"{t.order.action} {qty:g} {t.contract.symbol}"
                   if isinstance(qty, (int, float)) else str(t.contract.symbol))
    more = f" (+{len(trades) - 5} more)" if len(trades) > 5 else ""
    return ", ".join(out) + more


# ── one account ─────────────────────────────────────────────────────────────
def reconcile_account(cfg: acc.Config, a: acc.Account, as_of: str, client_id: int,
                      *, dry_run: bool, secret: str | None,
                      broker_factory=None) -> Finding:
    f = Finding(a.id, as_of)
    report = acc.account_dir(a.id) / "executed_book_live.json"
    prior = eb.completed_run(report, as_of)
    make = broker_factory or ic.Broker
    try:
        broker = make(cfg.host, cfg.port, client_id, account_mode="live",
                      expected_account=a.id, confirm_live=True,
                      market_data_type=cfg.market_data_type, account=a.id)
    except Exception as e:
        f.problems.append(f"could not connect to the account ({e})")
        return f
    try:
        fills = broker.day_fills_by_symbol()
        try:
            working = broker.working_orders()
        except RuntimeError as e:
            working = []
            f.problems.append(f"could not read the account's working orders ({e})")
        if working:
            f.problems.append(f"{len(working)} order(s) still working after the "
                              f"close: {describe_working(working)}")

        prior_trades = list((prior or {}).get("trades") or [])
        if prior is None:
            if fills:
                f.problems.append("no execution report was written for this bar "
                                  "(the run did not finish) — rebuilt from the "
                                  "broker's fills; the plan is unknown, so check "
                                  "TWS that every leg filled")
            else:
                f.problems.append("no execution report AND the broker shows no "
                                  "executions today — check TWS")
            merged, rewrite = merge_trades([], fills), bool(fills)
        else:
            f.problems += unfilled_legs(prior_trades, fills)
            if not fills and any(not str(t.get("status", "")).startswith("SKIPPED")
                                 for t in prior_trades):
                f.problems.append("the broker shows NO executions today although "
                                  "the report lists trades — the report was left "
                                  "untouched (IBKR only serves the current session)")
                merged, rewrite = prior_trades, False
            else:
                merged = merge_trades(prior_trades, fills)
                rewrite = needs_rewrite(prior_trades, fills)
                if rewrite:
                    f.notes.append("late or changed fills written into the report")

        if rewrite and dry_run:
            f.notes.append("(dry-run) the report would be rewritten")
        elif rewrite:
            if not secret:
                f.problems.append("report needs updating but OVERALL_BOOK_SECRET is "
                                  "not set, so it cannot be signed — not rewritten")
            else:
                import ibkr_execute_book as ex          # the executor's own writer
                payload = {"as_of": as_of, "profile": (prior or {}).get("profile", "")}
                ex._write_report(broker, payload, "execute", merged, str(report),
                                 secret, "live")
                f.rewritten = True
    except Exception as e:                      # one account must not stop the rest
        f.problems.append(f"reconcile error: {type(e).__name__}: {e}")
    finally:
        try:
            broker.disconnect()
        except Exception:
            pass
    return f


# ── main ────────────────────────────────────────────────────────────────────
def _now_et() -> pd.Timestamp:
    return pd.Timestamp.now(tz=ic.ET_TZ)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Reconcile today's MOC run against the "
                                              "broker, after the close")
    ap.add_argument("--config", default=str(acc.DEFAULT_CONFIG))
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--dry-run", action="store_true",
                    help="look and report; rewrite nothing, push nothing, no heartbeat")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="reconcile even before the closing auction has printed")
    args = ap.parse_args(argv)

    def ping(kind, msg=""):
        if not args.dry_run:
            hbeat.ping(kind, msg, env_var="IBKR_RECONCILE_HEARTBEAT_URL")

    ping("start", "reconcile started")
    try:
        code, detail, benign = _reconcile(args)
    except BaseException as e:
        ping("fail", f"crashed: {type(e).__name__}: {str(e)[:200]}")
        raise
    ping("ok" if (code == EXIT_OK or (code == EXIT_NOTHING and benign)) else "fail",
         detail)
    return code


def _reconcile(args) -> tuple[int, str, bool]:
    cfg, errs = acc.load_config(Path(args.config))
    if errs or cfg is None:
        print("CONFIG ERROR:")
        for e in errs:
            print(f"  - {e}")
        return EXIT_SETUP, "config error", False
    if args.only:
        for a in cfg.accounts:
            a.enabled = a.enabled and a.id in args.only
    print("Live reconcile" + (" (dry-run)" if args.dry_run else ""))
    print("Runtime: " + ic.runtime_versions())

    now_et = _now_et()
    h, m = ic.session_close_et(now_et)
    ready = now_et.normalize() + pd.Timedelta(hours=h, minutes=m + AUCTION_SETTLE_MIN)
    if ic.is_trading_day(now_et.tz_localize(None))[0] and now_et < ready and not args.force:
        print(f"Too early: the closing auction has not printed yet ({now_et:%H:%M} ET; "
              f"reconcile from {ready:%H:%M} ET, or pass --force). Nothing done.")
        return EXIT_NOTHING, "too early — closing auction not printed", False

    locks = todays_locks(cfg.enabled(), now_et)
    if not locks:
        print("No orders were sent for any account today — nothing to reconcile.")
        return EXIT_NOTHING, "nothing to reconcile (no orders sent today)", True
    todo = [a for a in cfg.enabled() if a.id in locks]
    print(f"{len(todo)} account(s) had orders sent today: "
          + ", ".join(f"{a.id} (bar {locks[a.id]})" for a in todo))

    managed, why = acc.verify_gateway(cfg, retry_s=acc.GATEWAY_RETRY_S,
                                      respect_cutoff=False)
    print(f"Gateway: {why}")
    if managed is None:
        acc.alert(f"🔴 IBKR reconcile could not reach the gateway — {why}. Today's "
                  "fills were NOT verified; check TWS.")
        return EXIT_SETUP, f"gateway: {why[:300]}", False

    secret = os.environ.get("OVERALL_BOOK_SECRET")
    index_of = {a.id: i for i, a in enumerate(cfg.accounts)}
    findings: list[Finding] = []
    for a in todo:
        if a.id not in managed:
            fnd = Finding(a.id, locks[a.id], problems=["not managed by this login"])
        else:
            fnd = reconcile_account(cfg, a, locks[a.id],
                                    cfg.client_id_base + CLIENT_ID_OFFSET + index_of[a.id],
                                    dry_run=args.dry_run, secret=secret)
        findings.append(fnd)
        print(f"  {a.id}  {a.label[:24]:<24} "
              + ("ok" if fnd.ok else "PROBLEM")
              + (" · report rewritten" if fnd.rewritten else ""))
        for p in fnd.problems:
            print(f"      ✗ {p}")
        for n in fnd.notes:
            print(f"      · {n}")

    changed = [a for a in todo if any(x.account == a.id and x.rewritten for x in findings)]
    if changed and not args.dry_run and acc.publishing_enabled(args.no_push):
        print("\nPublishing the rebuilt reports …")
        acc.push_reports(changed)

    bad = [x for x in findings if not x.ok]
    detail = ("; ".join(f"{hbeat.mask_account(x.account)}: {x.problems[0][:120]}"
                        for x in bad) if bad
              else f"{len(findings)} account(s) reconciled")
    if bad:
        acc.alert("🔴 IBKR reconcile found problems — " + "; ".join(
            f"{x.account}: " + " | ".join(x.problems) for x in bad))
        return EXIT_PROBLEM, detail, False
    print("\nAll accounts reconciled.")
    return EXIT_OK, detail, False


if __name__ == "__main__":
    raise SystemExit(main())
