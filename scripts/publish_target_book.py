"""Publish today's target book (Option C, publisher side).

Runs the full Overall engine and writes the live-adjusted target allocation as a
compact, optionally-signed JSON artifact that a local executor
(``scripts/ibkr_execute_book.py``) consumes to trade an IBKR paper account —
without ever running the model itself.

This is the "cloud decides" half of Option C.  It's transport-agnostic: it just
writes a file.  Wire it to a transport however you like — e.g. the GitHub Action
``.github/workflows/publish-target-book.yml`` runs this and commits the artifact
to the feature branch, so the executor can ``git pull`` (or fetch the raw URL)
and act on it.

    OVERALL_BOOK_SECRET=… python scripts/publish_target_book.py            # default profile
    python scripts/publish_target_book.py --profile Balanced --stdout

Signing: set ``OVERALL_BOOK_SECRET`` (env) to HMAC-sign the artifact so the
executor can verify authenticity before trading.  Without it the book is written
unsigned (a warning is printed).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "app"))
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO))

import pandas as pd                               # noqa: E402
import overall_core as oc                         # noqa: E402
import target_book as tb                          # noqa: E402
import freshness as fr                            # noqa: E402
from ibkr_rebalance import compute_target_book     # noqa: E402  (reuse the exact app path)

DEFAULT_OUT = _REPO / "data" / "overall" / "target_book.json"
DEFAULT_OUT_LIVE = _REPO / "data" / "overall" / "target_book_live.json"
DEFAULT_AUDIT_OUT = _REPO / "data" / "overall" / "daily_audit.json"
SATA_KEY = "SATA"
AUDIT_SCHEMA = "overall-daily-audit/v1"
AUDIT_RETRY_WAIT_S = 90          # let a detached BTC feature re-pull land, then retry

# Signal app → data-gate dataset key.  Everything not listed shares its own name;
# the ⛏️ miners sleeve reads the SAME gold dataset as 🥇 GLDM (see
# gldm_engine._load_daily), so a hole there strands both apps at once.
GATE_DATASET = {"GDXM": "GLDM"}


def _gate_decisions() -> dict:
    """Newest data-gate decision per dataset key (runtime/dataset_audit.json).

    Without this the publish log says only "STALE (3d behind)" and gives no clue
    WHY — whether the fetch came back empty, was rejected by quality control, or
    was served from a pinned snapshot — because every gate decision went to a
    gitignored runtime file that is never printed, committed or uploaded.  That
    is what turned the 2026-09-01 stale-withhold into a code-archaeology
    exercise.  Best-effort: reporting must never break a publish.
    """
    try:
        import data_gate as dg
        log = dg.read_audit() or {}
    except Exception:
        return {}
    out = {}
    for key, entries in log.items():
        if not entries:
            continue
        e = entries[0]
        out[key] = dict(decision=e.get("decision"), source=e.get("source"),
                        served_through=e.get("date_to"),
                        qc_passed=e.get("qc_passed"),
                        failed_checks=list(e.get("failed_checks") or []),
                        note=e.get("note") or "")
    return out


def _attach_gate_decisions(audit: dict) -> dict:
    """Fold each app's gate decision into its audit row, so the committed audit
    trail (and the 🕵️ Daily Audit tab) explains a stale app instead of just
    reporting one."""
    gates = _gate_decisions()
    for row in audit.get("rows", []):
        key = GATE_DATASET.get(row.get("app"), row.get("app"))
        if key in gates:
            row["data_gate"] = gates[key]
    return audit


def _withheld_streak(path: Path, published: bool, now) -> tuple:
    """``(withheld_since_ct, consecutive_withheld_days)`` for the current streak.

    Lets the workflow tell a ONE-CYCLE withhold (a feed that is briefly behind —
    the designed safe path, and usually fixed by the next catch-up slot) from a
    withhold that is STUCK, which is a real outage needing a human.  Both used to
    surface as the same red X, so the safe path and a genuine engine crash were
    indistinguishable at a glance.

    Counted in CT calendar days, not runs, because several catch-up slots fire
    per cycle and must not inflate the streak.
    """
    today = fr._as_utc(now).tz_convert(fr.CT).date()
    if published:
        return None, 0
    prev = {}
    try:
        prev = (json.loads(path.read_text()) or {}).get("target_book") or {}
    except Exception:
        prev = {}
    since = prev.get("withheld_since_ct") if not prev.get("published", True) else None
    try:
        since_d = pd.Timestamp(since).date() if since else today
    except Exception:
        since_d = today
    if since_d > today:
        since_d = today
    return str(since_d), int((today - since_d).days) + 1


def _write_audit(path: Path, *, audit: dict, results: list, profile: str,
                 book_published: bool, book_info: dict | None,
                 skip_reason: str | None) -> None:
    """Persist the scheduled-run audit trail the 🕵️ Daily Audit tab reads."""
    now = fr.now_utc()
    withheld_since, withheld_days = _withheld_streak(path, book_published, now)
    as_of = max((str(pd.Timestamp(r["as_of"]).date()) for r in results), default="")
    payload = dict(
        schema=AUDIT_SCHEMA,
        generated_at_utc=now.isoformat(timespec="seconds"),
        generated_at_ct=fr.fmt_ct(now, seconds=True),
        scheduled_publish=fr.SCHEDULED_PUBLISH_CT,
        overall=dict(as_of=as_of, profile=profile,
                     n_instruments=len(results),
                     n_apps=len({r.get("parent") for r in results}),
                     computed_at_utc=now.isoformat(timespec="seconds")),
        audit=audit,
        target_book=dict(published=bool(book_published),
                         skip_reason=skip_reason,
                         withheld_since_ct=withheld_since,
                         consecutive_withheld_days=withheld_days,
                         **(book_info or {})),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, default=str))
    print(f"Wrote audit trail {path}")


def _publish_sha() -> str:
    """Short SHA of the code publishing this book: the Action's GITHUB_SHA,
    else the local HEAD, else ``unknown`` — provenance only, never fatal."""
    sha = os.environ.get("GITHUB_SHA")
    if not sha:
        try:
            import subprocess
            sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO,
                                 capture_output=True, text=True,
                                 timeout=10).stdout.strip()
        except Exception:
            sha = ""
    return (sha or "unknown")[:12]


def main() -> int:
    ap = argparse.ArgumentParser(description="Publish the Overall target books (Option C)")
    ap.add_argument("--profile", default=oc.DEFAULT_PROFILE,
                    choices=list(oc.RISK_PROFILES), help="risk profile (default: app default)")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help=f"PAPER book output path (default {DEFAULT_OUT})")
    ap.add_argument("--live-out", default=str(DEFAULT_OUT_LIVE),
                    help=f"LIVE book output path (default {DEFAULT_OUT_LIVE})")
    ap.add_argument("--stdout", action="store_true",
                    help="also print the paper artifact to stdout")
    ap.add_argument("--no-sign", action="store_true",
                    help="do not sign even if OVERALL_BOOK_SECRET is set")
    ap.add_argument("--audit-out", default=str(DEFAULT_AUDIT_OUT),
                    help=f"audit-trail output path (default {DEFAULT_AUDIT_OUT})")
    ap.add_argument("--allow-stale", action="store_true",
                    help="publish the book even when the signal-freshness audit "
                         "fails (default: REFUSE — stale signals are never "
                         "published to the target book)")
    args = ap.parse_args()

    # The book is published EVERY day, weekends and US market holidays included:
    # Bitcoin trades continuously, so BTC-driven signals (and the resulting
    # target weights) keep moving while the US equity market is closed. The
    # weekend/holiday guard lives solely on the EXECUTOR side
    # (ibkr_execute_book.py / ibkr_rebalance.py via ibkr_common.is_trading_day),
    # so orders are still only ever placed on US market days — but when the
    # executor wakes up on the next trading morning it acts on a book computed
    # from the freshest data, not one frozen at the prior week's close.
    secret = None if args.no_sign else os.environ.get("OVERALL_BOOK_SECRET")

    # ── 1. Run the universe ONCE, then AUDIT its signal freshness ────────────
    # The Overall strategy must only be computed from every app's freshest
    # COMPLETED bar: equities from the prior 4:00 PM ET close, Bitcoin from the
    # 12:00-UTC (7:00 AM CT) bar that closed just before this scheduled run.
    # The completed-bars-only flag makes the daily fetchers trim any
    # in-progress *today* equity bar, so a delayed catch-up run or a manual 🚀
    # publish fired mid-session still produces the same close-based book the
    # 7:15-AM-CT run would have.  A failed audit gets ONE full re-run (the BTC
    # engine kicks off a background feature re-pull on the first pass; the wait
    # lets it land) — the "proper data / signal refresh" step.  Still stale ⇒
    # the book is NOT published (unless --allow-stale): stale signals never
    # reach the target book.
    os.environ[fr.COMPLETED_BARS_ENV] = "1"
    # The book's data basis is pinned to TODAY's 7:15-AM-CT anchor whatever
    # the wall-clock time of this run: equities from the pre-anchor session
    # close, BTC from the day's 7:00-AM-CT bar. The audit expects exactly that
    # basis (expected_now=anchor), so a post-close run neither ingests nor
    # demands the close that just landed.
    anchor = fr.publish_anchor_ct()
    print(f"Running the Overall universe (live fetch, ~30–90s, completed bars "
          f"only, basis anchored {fr.fmt_ct(anchor)}) — profile "
          f"{args.profile}…", flush=True)
    results = oc.run_universe()
    if not results:
        raise RuntimeError("run_universe() returned no instruments — check data feeds")
    audit = fr.audit_universe(results, parent_order=oc.PARENT_KEYS,
                              expected_now=anchor)
    if not audit["passed"]:
        print(f"Signal-freshness audit FAILED (stale: {audit['stale_apps']}) — "
              f"forcing a data refresh and re-running once in "
              f"{AUDIT_RETRY_WAIT_S}s…", flush=True)
        time.sleep(AUDIT_RETRY_WAIT_S)
        results = oc.run_universe() or results
        audit = fr.audit_universe(results, parent_order=oc.PARENT_KEYS,
                                  expected_now=anchor)
    audit = _attach_gate_decisions(audit)
    for row in audit["rows"]:
        mark = "✅ fresh" if row["fresh"] else f"🚨 STALE ({row['age_days']}d behind)"
        print(f"  audit {row['app']:5s} as-of {row['actual_asof']} "
              f"(expected ≥ {row['expected_asof']})  {mark}")
        g = row.get("data_gate")
        if g:
            bits = [f"{g['decision']} ({g['source']})",
                    f"through {g['served_through']}"]
            if g["failed_checks"]:
                bits.append("FAILED: " + ", ".join(g["failed_checks"]))
            if g["note"]:
                bits.append(g["note"])
            print(f"        └ data gate: {' — '.join(bits)}")

    if not audit["passed"] and not args.allow_stale:
        print(f"\nAUDIT FAILED — stale signal apps: {audit['stale_apps']}. "
              "REFUSING to publish a target book from stale signals "
              "(the executor keeps trading the previous verified book; its own "
              "freshness guard skips if that book is too old). "
              "Run with --allow-stale to override.", file=sys.stderr)
        _write_audit(Path(args.audit_out), audit=audit, results=results,
                     profile=args.profile, book_published=False, book_info=None,
                     skip_reason=f"signal-freshness audit failed: "
                                 f"stale {audit['stale_apps']}")
        return 2

    # ── 2. Audit passed → compute the book from the SAME audited results ─────
    # live_adjust=False: the PUBLISHED book is the committed last-close
    # allocation only (no live-spot exit override), so it matches the Overall
    # app's action plan and stays reproducible for the whole frozen day.
    print(f"\nAudit {'PASSED' if audit['passed'] else 'overridden (--allow-stale)'}"
          " — computing target book (committed close signals)…", flush=True)
    book = compute_target_book(args.profile, results=results, live_adjust=False)

    if not secret:
        print("WARNING: OVERALL_BOOK_SECRET not set — writing UNSIGNED books. "
              "The executor cannot verify authenticity.", file=sys.stderr)
    else:
        print("Signed with OVERALL_BOOK_SECRET (HMAC-SHA256).")

    # ── PAPER book — undeployed remainder is held as CASH (no SATA leg) ──────
    paper = tb.build_payload(
        as_of=book.as_of, profile=args.profile, weights=book.weights,
        cash_weight=book.cash_weight, exec_price=book.exec_price,
        actions=book.actions, book_mode="paper")

    # ── LIVE book — the SAME risk weights, but the idle remainder is parked in
    # a real SATA position (Strive preferred, ~13% yield) instead of cash. ─────
    live_weights = dict(book.weights)
    live_exec = dict(book.exec_price)
    if book.cash_weight > 1e-6:
        live_weights[SATA_KEY] = live_weights.get(SATA_KEY, 0.0) + book.cash_weight
        try:
            sata_px = oc.fetch_sata().get("price")
        except Exception:
            sata_px = None
        live_exec[SATA_KEY] = float(sata_px) if sata_px else float(oc.SATA["par"])
    live = tb.build_payload(
        as_of=book.as_of, profile=args.profile, weights=live_weights,
        cash_weight=0.0, exec_price=live_exec, actions=book.actions,
        book_mode="live", generated_at_utc=paper["generated_at_utc"])

    # Stamp the audit verdict INTO both books (covered by the HMAC signature),
    # so any consumer can see the signals were verified fresh at publish time.
    _audit_stamp = dict(passed=bool(audit["passed"]),
                        checked_at_utc=audit["checked_at_utc"],
                        stale_apps=list(audit["stale_apps"]))
    paper["signal_audit"] = _audit_stamp
    live["signal_audit"] = dict(_audit_stamp)

    # Stamp the DATA BASIS too (also signature-covered): the exact closes this
    # book was built from — the UI donut captions read this instead of
    # guessing from the publish time.
    _basis = dict(
        anchor_ct=fr.fmt_ct(anchor),
        equity_close=str(fr.expected_equity_asof(anchor).date()),
        btc_bar_close_utc=fr.close_moment(
            "crypto", fr.expected_crypto_asof(anchor)).isoformat())
    paper["signal_basis"] = _basis
    live["signal_basis"] = dict(_basis)

    # Stamp the STRATEGY-LOGIC PROVENANCE (signature-covered): the deliberate
    # strategy version plus the publishing commit, so the as-published P&L
    # record can segment on logic changes instead of silently conflating
    # books produced by different generations of the strategy.
    for _bk in (paper, live):
        _bk["strategy_version"] = oc.STRATEGY_VERSION
        _bk["code_sha"] = _publish_sha()

    outdir = Path(args.out).parent
    outdir.mkdir(parents=True, exist_ok=True)
    for _bp in (Path(args.out), Path(args.live_out)):
        # outgoing books → *_prev.json (the "Previous Targetbook" the UI shows)
        if tb.rotate_prev(_bp):
            print(f"Rotated previous book → {tb.prev_path(_bp)}")
    Path(args.out).write_text(tb.dumps(paper, secret))
    Path(args.live_out).write_text(tb.dumps(live, secret))
    # dated as-of record (book_archive/<as_of>.json) — the Historical View's
    # source for "the book actually published from this bar's signals"
    _arch = tb.archive_book(live, Path(args.live_out), secret)
    if _arch:
        print(f"Archived as-of record → {_arch}")

    _write_audit(Path(args.audit_out), audit=audit, results=results,
                 profile=args.profile, book_published=True,
                 book_info=dict(generated_at_utc=paper["generated_at_utc"],
                                as_of=book.as_of, profile=args.profile,
                                signed=bool(secret)),
                 skip_reason=None)

    print(f"\nTarget books (as-of {book.as_of}, profile {args.profile}):")
    print("  PAPER (idle → cash):")
    for k in sorted(book.weights, key=lambda k: -book.weights[k]):
        print(f"    {k:5s} {book.weights[k]*100:5.1f}%")
    print(f"    CASH  {book.cash_weight*100:5.1f}%")
    print("  LIVE  (idle → SATA):")
    for k in sorted(live_weights, key=lambda k: -live_weights[k]):
        tag = "  ← idle park" if k == SATA_KEY else ""
        print(f"    {k:5s} {live_weights[k]*100:5.1f}%{tag}")
    print(f"\nWrote {args.out}\n      {args.live_out}")
    if args.stdout:
        print("\n" + tb.dumps(paper, secret))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
