"""Re-state live IBKR accounts after MANUAL trades, then publish the reports.

    python scripts/ibkr_manual_refresh.py                 # read + write reports locally
    python scripts/ibkr_manual_refresh.py --publish       # ... and commit + push them
    python scripts/ibkr_manual_refresh.py --only U123 --publish
    python scripts/ibkr_manual_refresh.py --pull --publish   # git pull --ff-only first

For when you bought/sold by hand in TWS (the 2:20 PM executor did not run, or
you topped up afterwards).  It NEVER places an order: for every enabled account
in the multi-account config it runs the executor's ``--refresh-report`` path,
which reads the account's real positions, cash, net-liq and TODAY'S fills from
the gateway and rewrites that account's signed ``executed_book_live.json`` (and
its dated ``executed_archive`` record).  Then it prints a reconciliation of what
each account holds against the book's target weights, flags orders still working
at the broker, and -- with ``--publish`` -- commits all the reports in ONE push
(or the encrypted vault, when ``IBKR_VAULT_KEY`` is set).

Why not ``ibkr_reconcile_accounts.py``?  It only looks at accounts with a
send-lock written by an executor run today; manual trades leave none, so it
exits 3 ("nothing to do").

Limits you should know about
  * IBKR serves the CURRENT session's fills only -- run this the same day as the
    trades.  An account with no fills today is NOT rewritten over a report that
    has trades (the executor refuses); it is listed as a problem here.
  * The report is stamped with the book you point it at (default: the config's
    ``book``, i.e. ``data/overall/target_book_live.json``).  Manual fills are
    recorded as "fill with no planned order" -- there was no plan to attach.
  * Needs OVERALL_BOOK_SECRET (reports are signed; an unsigned one shows a
    signature error in the app).

Exit codes: 0 all reconciled · 1 a problem to look at · 2 gateway/config problem.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO / "app"))

import ibkr_common as ic                               # noqa: E402
import ibkr_execute_accounts as acc                    # noqa: E402

EXIT_OK, EXIT_PROBLEM, EXIT_SETUP = 0, 1, 2
CLIENT_ID_OFFSET = 80        # clear of the run (base+i), the check and the reconcile (+60)
CHILD_TIMEOUT_S = 300.0
WEIGHT_TOL = 0.01            # |actual - target| above this share of NAV is flagged


# ── pure helpers (unit-tested without a broker) ─────────────────────────────
def child_argv(cfg: acc.Config, a: acc.Account, index: int, book: str,
               python: str = sys.executable) -> list[str]:
    """The refresh-only command for one account.  Contains no ``--execute``."""
    return [python, str(acc.EXECUTOR),
            "--file", book,
            "--host", cfg.host, "--port", str(cfg.port),
            "--client-id", str(cfg.client_id_base + CLIENT_ID_OFFSET + index),
            "--account", a.id, "--expected-account", a.id,
            "--account-mode", "live", "--confirm-live",
            "--market-data-type", str(cfg.market_data_type),
            "--require-signature", "--no-push-report", "--refresh-report"]


def reconcile_lines(report: dict, weights: dict, tol: float = WEIGHT_TOL
                    ) -> tuple[list[str], list[str]]:
    """(table lines, problems): what the account holds vs the book's targets.

    Weights are shares of the report's net-liq, so an account that deploys less
    than the book (a ``max_deploy_frac`` cap) will show a uniform shortfall --
    that is informational; a name held that the book does not want, or wanted
    and missing, is what gets flagged."""
    nav = float(report.get("net_liq") or 0.0)
    held = {}
    for p in report.get("positions") or []:
        k = p.get("key") or p.get("symbol")
        if k and float(p.get("shares") or 0.0) != 0.0:
            held[k] = held.get(k, 0.0) + float(p.get("market_value") or 0.0)
    lines, problems = [], []
    def _size(k: str) -> float:
        return float(weights.get(k) or 0.0) or (held.get(k, 0.0) / nav if nav else 0.0)

    for k in sorted(set(weights) | set(held), key=lambda k: -_size(k)):
        want = float(weights.get(k, 0.0) or 0.0)
        have = held.get(k, 0.0) / nav if nav else 0.0
        flag = ""
        if want and not have:
            flag = "  <- MISSING"
            problems.append(f"{k}: book wants {want*100:.1f}% but the account holds none")
        elif have and not want:
            flag = "  <- NOT IN BOOK"
            problems.append(f"{k}: held at {have*100:.1f}% but the book has no target")
        elif abs(have - want) > tol:
            flag = f"  <- off by {(have - want)*100:+.1f}%"
        lines.append(f"    {k:<6} target {want*100:5.1f}%   held {have*100:5.1f}%{flag}")
    return lines, problems


def manual_fills(report: dict) -> list[dict]:
    return [t for t in report.get("trades") or []
            if "no planned order" in str(t.get("reason") or "")]


# ── orchestration ───────────────────────────────────────────────────────────
def _git(*a: str) -> subprocess.CompletedProcess:
    return subprocess.run(("git",) + a, cwd=_REPO, capture_output=True, text=True)


def run(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=str(acc.DEFAULT_CONFIG))
    ap.add_argument("--only", action="append", default=[],
                    help="restrict to this account id (repeatable)")
    ap.add_argument("--book", help="book to stamp the report with "
                    "(default: the config's `book`)")
    ap.add_argument("--publish", action="store_true",
                    help="commit + push the refreshed reports (one commit)")
    ap.add_argument("--pull", action="store_true",
                    help="git pull --ff-only origin <branch> first, so the book is current")
    args = ap.parse_args(argv)

    cfg, errs = acc.load_config(Path(args.config))
    if errs or cfg is None:
        print("CONFIG ERROR:\n" + "\n".join(f"  - {e}" for e in errs))
        return EXIT_SETUP
    if args.only:
        unknown = [o for o in args.only if o not in {a.id for a in cfg.accounts}]
        if unknown:
            print(f"CONFIG ERROR: --only names unknown account(s): {', '.join(unknown)}")
            return EXIT_SETUP
        for a in cfg.accounts:
            a.enabled = a.enabled and a.id in args.only
    accounts = cfg.enabled()
    if not accounts:
        print("CONFIG ERROR: no enabled accounts.")
        return EXIT_SETUP
    if not os.environ.get("OVERALL_BOOK_SECRET"):
        print("ABORT: OVERALL_BOOK_SECRET is not set -- the book cannot be verified "
              "and the reports would go out unsigned. Nothing written.")
        return EXIT_SETUP

    if args.pull:
        branch = (_git("rev-parse", "--abbrev-ref", "HEAD").stdout or "").strip()
        if branch in ("", "HEAD"):
            print("WARN: detached HEAD -- skipping --pull")
        else:
            r = _git("pull", "--ff-only", "origin", branch)
            print(f"git pull: {(r.stdout or r.stderr).strip().splitlines()[-1:] or 'ok'}")

    book = args.book or (cfg.book if Path(cfg.book).is_absolute() else str(_REPO / cfg.book))
    payload = json.loads(Path(book).read_text())
    weights = dict(payload.get("weights") or {})
    print(f"Book: {book}\n  as-of {payload.get('as_of')}, profile {payload.get('profile')}, "
          f"generated {payload.get('generated_at_utc')}, mode {payload.get('book_mode')}")
    print(f"  targets: " + ", ".join(f"{k} {v*100:.1f}%" for k, v in
                                      sorted(weights.items(), key=lambda kv: -kv[1])))

    managed, why = acc.verify_gateway(cfg)
    print(f"Gateway: {why}")
    if managed is None:
        print("ABORT: cannot read the accounts. Nothing written.")
        return EXIT_SETUP
    missing = [a.id for a in accounts if a.id not in managed]
    if missing:
        print(f"ABORT: not managed by this login: {', '.join(missing)}. Nothing written.")
        return EXIT_SETUP

    problems_total: list[str] = []
    refreshed: list[acc.Account] = []
    for i, a in enumerate(accounts):
        print(f"\n=== {a.id}  {a.label} " + "=" * 40)
        argv_c = child_argv(cfg, a, i, book)
        assert "--execute" not in argv_c           # this tool must never trade
        try:
            r = subprocess.run(argv_c, cwd=_REPO, capture_output=True, text=True,
                               timeout=CHILD_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            problems_total.append(f"{a.id}: refresh timed out after {CHILD_TIMEOUT_S:.0f}s")
            print("  TIMED OUT")
            continue
        tail = [l for l in (r.stdout or "").splitlines()
                if any(s in l for s in ("Wrote execution", "archived", "WARN", "ABORT",
                                        "executions today", "Signature"))]
        print("\n".join("  " + l.strip() for l in tail) or "  (no output)")
        if r.returncode != 0:
            problems_total.append(f"{a.id}: refresh failed (exit {r.returncode}) -- "
                                  + (tail[-1].strip() if tail else "see output"))
            continue
        path = acc.account_dir(a.id) / "executed_book_live.json"
        try:
            report = json.loads(path.read_text())
        except (OSError, ValueError) as e:
            problems_total.append(f"{a.id}: report unreadable after refresh ({e})")
            continue
        refreshed.append(a)
        mf = manual_fills(report)
        print(f"  net-liq ${report.get('net_liq', 0):,.0f}  cash ${report.get('cash', 0):,.0f}  "
              f"{len(report.get('trades') or [])} trade(s) today, {len(mf)} without a plan")
        for t in report.get("trades") or []:
            print(f"    {t.get('action'):<4} {t.get('symbol'):<6} {t.get('filled'):g} "
                  f"@ {t.get('avg_fill_price'):.2f}  [{t.get('status')}]")
        lines, probs = reconcile_lines(report, weights)
        print("  vs book:\n" + "\n".join(lines))
        problems_total += [f"{a.id}: {p}" for p in probs]
        # an order still working means the fill (and so this report) is not final
        try:
            b = ic.Broker(cfg.host, cfg.port, cfg.client_id_base + CLIENT_ID_OFFSET + 20 + i,
                          account_mode="live", expected_account=a.id, confirm_live=True,
                          market_data_type=cfg.market_data_type, account=a.id)
            try:
                w = b.working_orders()
            finally:
                b.disconnect()
            if w:
                problems_total.append(f"{a.id}: {len(w)} order(s) still working at IBKR -- "
                                      "not filled yet; re-run once they are")
        except Exception as e:
            problems_total.append(f"{a.id}: could not read working orders ({e})")

    if args.publish:
        print("\n=== publish ===")
        if refreshed:
            if not acc.push_reports(refreshed):
                problems_total.append("publish failed -- reports are on disk, not pushed")
        else:
            print("  nothing refreshed -- nothing to publish")
    else:
        print("\n(not published -- add --publish to commit + push the reports)")

    print("\n" + ("All accounts refreshed." if not problems_total
                  else "PROBLEMS:\n" + "\n".join(f"  - {p}" for p in problems_total)))
    return EXIT_OK if not problems_total else EXIT_PROBLEM


if __name__ == "__main__":
    sys.exit(run())
