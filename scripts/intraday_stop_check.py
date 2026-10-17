"""Intraday stop check — the 2:00-PM-CT amendment to the day's published books.

The daily books are built at 7:15 AM CT from the previous close, so a stop
(a fixed −5 %/−10 % stop or a 10 %/12 % trailing stop on the sleeves that have
one) can only be seen by the book the morning AFTER the close that breached it.
This script closes most of that gap: ~20 minutes before the executor's 2:20 PM
CT slot it prices every HELD sleeve that carries a stop at its LIVE quote and,
if any is at/below its stop level, re-publishes BOTH books (paper + live) with
those sleeves taken out — exactly as tomorrow's book would show them — so the
executor, the Collective2 publisher (2:30 PM CT) and the as-published record
all see the stop-out today.

    OVERALL_BOOK_SECRET=… python scripts/intraday_stop_check.py            # check + republish
    python scripts/intraday_stop_check.py --dry-run                         # report only
    python scripts/intraday_stop_check.py --force                           # ignore the time window

What it does, in order
----------------------
1. Guards: a US trading day, inside the CT window (default 13:45–14:12, i.e.
   it must finish BEFORE the executor starts — a book that changes after the
   executor traded it would no longer describe the account), and a published
   book that is TODAY's (CT date), signature-verified.
2. Re-runs the universe on the same completed-bar basis as the morning publish
   (a failed freshness audit aborts: no stop is acted on from stale signals).
3. For each sleeve that is held in the published book and has a stop level
   (``pos.stop_px`` for a fixed stop, ``pos.trail_px`` for a trailing stop —
   both read off the last completed close, the trail level only ever ratchets
   up) compares the sleeve's own live quote with that level.  A quote is only
   acted on if it is fresh, from an open session and within 35 % of the last
   close — a dead feed or a bad print can never stop a position out.
4. Nothing breached → the books are left untouched; only the small status
   marker ``data/overall/intraday_stop_check.json`` is written (the executor
   reads it).  Something breached → the allocation is recomputed with those
   sleeves force-exited (V2 adds-only pinned to the morning book, so nothing
   else is trimmed), both books are re-signed and rewritten in place with an
   ``intraday_stop`` stamp, and the dated as-of archive is replaced.  The
   outgoing book is NOT rotated into ``*_prev.json`` (same-day re-publish).

Honest limits (by design of the request)
----------------------------------------
* The engine's stop is a CLOSE rule; this fires on the 2:00 PM price.  A sleeve
  that is below its stop at 2:00 but recovers into the close is sold anyway
  (and tomorrow's close-based book may buy it back); one that breaches only in
  the last hour is caught the next day as before.
* A manual 🚀 publish after this job re-builds the committed-signal book and
  drops the amendment — re-run this job afterwards (it is idempotent).

Exit code: 0 checked (amended or nothing to do / skipped by a guard) ·
1 the check could not be completed (stale audit, bad signature, engine error).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "app"))
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO))

import pandas as pd                               # noqa: E402

import freshness as fr                            # noqa: E402
import overall_core as oc                         # noqa: E402
import target_book as tb                          # noqa: E402

DEFAULT_PAPER = _REPO / "data" / "overall" / "target_book.json"
DEFAULT_LIVE = _REPO / "data" / "overall" / "target_book_live.json"
DEFAULT_MARKER = _REPO / "data" / "overall" / "intraday_stop_check.json"
MARKER_SCHEMA = "overall-intraday-stop-check/v1"

# The job must be done before the executor starts (2:20 PM CT) and have time to
# land its commit; the early bound keeps a stray early fire off a thin pre-2 PM
# tape.  Both overridable (env) when the executor slot moves.
DEFAULT_EARLIEST = "13:45"
DEFAULT_LATEST = "14:12"
QUOTE_MAX_AGE_MIN = 20.0      # a live quote older than this is not "live"
SANE_BAND = 0.35              # |live / last close − 1| beyond this is a bad print
WATCH_ROWS = 8                # nearest-to-stop rows kept in the marker


# ── pure logic (unit-tested) ────────────────────────────────────────────────
def stop_level(res: dict) -> tuple[float | None, str | None]:
    """(level, kind) of a held sleeve's stop — ``("trail"|"fixed")`` — or
    ``(None, None)`` for a sleeve that is flat or has no stop.  A trailing stop
    is reported as ``trail`` (the engine also mirrors its level into
    ``stop_px``)."""
    p = res.get("pos") or {}
    if not p.get("in_pos"):
        return None, None
    trail = p.get("trail_px")
    if trail:
        return float(trail), "trail"
    stop = p.get("stop_px")
    if stop:
        return float(stop), "fixed"
    return None, None


def _quote_problem(q: dict, bar_close, now) -> str | None:
    """Why a live quote must NOT be acted on (None = usable)."""
    px = q.get("price")
    try:
        px = float(px)
    except (TypeError, ValueError):
        return "no live quote"
    if not math.isfinite(px) or px <= 0:
        return "no live quote"
    if q.get("session_closed"):
        return "session closed"
    qt = q.get("quote_time")
    if qt is not None:
        try:
            age = (pd.Timestamp(now) - pd.Timestamp(qt)).total_seconds() / 60.0
            if age > QUOTE_MAX_AGE_MIN:
                return f"quote {age:.0f} min old"
        except Exception:
            pass
    try:
        b = float(bar_close)
        if math.isfinite(b) and b > 0 and abs(px / b - 1.0) > SANE_BAND:
            return f"implausible vs last close ({(px / b - 1) * 100:+.0f} %)"
    except (TypeError, ValueError):
        pass
    return None


def evaluate_stops(results: list[dict], spot: dict, *, held_keys=None,
                   now=None) -> dict:
    """Compare every held, stop-carrying sleeve's live price with its level.

    ``held_keys`` limits the check to sleeves in the published book (None = all
    engine-held sleeves).  Returns ``{"triggered": [...], "watching": [...],
    "skipped": [...]}``; each row carries ``key, kind, level, live, dist_pct,
    entry, src`` (``skipped`` rows carry ``reason`` instead of a verdict)."""
    now = now if now is not None else fr.now_utc()
    out = {"triggered": [], "watching": [], "skipped": []}
    for r in results:
        key = r["key"]
        level, kind = stop_level(r)
        if level is None:
            continue
        if held_keys is not None and key not in held_keys:
            continue
        p = r.get("pos") or {}
        q = (spot or {}).get(key) or {}
        row = dict(key=key, kind=kind, level=round(level, 4),
                   entry=p.get("entry_px"))
        why = _quote_problem(q, r.get("bar_close"), now)
        if why:
            out["skipped"].append(dict(row, reason=why))
            continue
        px = float(q["price"])
        row.update(live=round(px, 4), dist_pct=round((px / level - 1.0) * 100, 2),
                   src=q.get("src"))
        (out["triggered"] if px <= level else out["watching"]).append(row)
    out["watching"].sort(key=lambda x: x["dist_pct"])
    return out


def in_window(ct_now, earliest: str, latest: str) -> tuple[bool, str]:
    """(inside?, reason) for the CT wall-clock window ``earliest``–``latest``."""
    def _hm(s):
        h, m = str(s).split(":")
        return int(h) * 60 + int(m)
    cur = ct_now.hour * 60 + ct_now.minute
    stamp = ct_now.strftime("%H:%M CT")
    if cur < _hm(earliest):
        return False, f"{stamp} is before the {earliest} CT window"
    if cur > _hm(latest):
        return False, (f"{stamp} is after the {latest} CT cut-off — the executor "
                       "may already be trading this book")
    return True, f"{stamp} inside the {earliest}–{latest} CT window"


def amend_actions(actions: list[dict], triggered: list[dict]) -> list[dict]:
    """Mark each stopped sleeve's action row CLOSE (the allocator leaves a
    force-exited, still-engine-held sleeve as HOLD with a zero target)."""
    by = {t["key"]: t for t in triggered}
    out = []
    for a in actions:
        a = dict(a)
        t = by.get(a.get("key"))
        if t:
            a.update(action="CLOSE", target=0.0, exits_next_bar=False,
                     decision=(f"STOP — live {t['live']:,.2f} ≤ "
                               f"{t['kind']} stop {t['level']:,.2f}"))
        out.append(a)
    return out


def write_marker(path: Path, **fields) -> None:
    payload = dict(schema=MARKER_SCHEMA, **fields)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, default=str))


# ── the job ─────────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="2:00 PM CT intraday stop check")
    ap.add_argument("--out", default=str(DEFAULT_PAPER), help="PAPER book path")
    ap.add_argument("--live-out", default=str(DEFAULT_LIVE), help="LIVE book path")
    ap.add_argument("--marker-out", default=str(DEFAULT_MARKER))
    ap.add_argument("--earliest", default=os.environ.get("INTRADAY_STOP_EARLIEST",
                                                         DEFAULT_EARLIEST))
    ap.add_argument("--latest", default=os.environ.get("INTRADAY_STOP_LATEST",
                                                       DEFAULT_LATEST))
    ap.add_argument("--dry-run", action="store_true",
                    help="evaluate and report; write nothing")
    ap.add_argument("--force", action="store_true",
                    help="ignore the trading-day / time-window / already-applied guards")
    args = ap.parse_args(argv)

    now = fr.now_utc()
    ct = now.tz_convert(fr.CT)
    marker = Path(args.marker_out)
    base = dict(checked_at_utc=now.isoformat(timespec="seconds"),
                checked_at_ct=fr.fmt_ct(now, seconds=True))

    def done(status: str, rc: int = 0, **extra) -> int:
        print(f"intraday stop check: {status}" +
              (f" — {extra['reason']}" if extra.get("reason") else ""))
        if not args.dry_run:
            write_marker(marker, status=status, **base, **extra)
        return rc

    if not args.force:
        if not fr.is_us_trading_day(ct.date()):
            print(f"Not a US trading day ({ct.date()}) — nothing to check.")
            return 0
        ok, why = in_window(ct, args.earliest, args.latest)
        if not ok:
            print(f"Skipped: {why}.")
            return 0

    paper_path, live_path = Path(args.out), Path(args.live_out)
    secret = os.environ.get("OVERALL_BOOK_SECRET")
    try:
        cur = tb.loads(paper_path.read_text())
    except Exception as e:
        return done("no_book", reason=f"cannot read {paper_path.name}: {e}")
    ok_sig, sig_why = tb.verify_signature(cur, secret)
    if not ok_sig or (cur.get("signature") and not secret):
        return done("error", 1, reason=f"book signature: {sig_why if not ok_sig else 'signed but OVERALL_BOOK_SECRET is not set'}")
    gen = pd.Timestamp(cur.get("generated_at_utc"))
    if gen.tzinfo is None:
        gen = gen.tz_localize("UTC")
    if gen.tz_convert(fr.CT).date() != ct.date():
        return done("no_current_book",
                    reason=f"the published book is from {gen.tz_convert(fr.CT).date()}, "
                           "not today — the morning publish was withheld or is late")
    profile = cur.get("profile") or oc.DEFAULT_PROFILE
    if profile not in oc.RISK_PROFILES:
        return done("error", 1, reason=f"unknown profile {profile!r} in the book")
    held_w = {k: float(w) for k, w in (cur.get("weights") or {}).items()
              if k != "SATA" and w and float(w) > 0}
    prior = cur.get("intraday_stop") or {}

    # ── same completed-bar basis as the morning publish ──────────────────────
    os.environ[fr.COMPLETED_BARS_ENV] = "1"
    anchor = fr.publish_anchor_ct()
    try:
        results = oc.run_universe()
        if not results:
            raise RuntimeError("run_universe() returned no instruments")
        audit = fr.audit_universe(results, parent_order=oc.PARENT_KEYS,
                                  expected_now=anchor)
    except Exception as e:
        return done("error", 1, reason=f"engine run failed: {e}")
    if not audit["passed"]:
        return done("audit_failed", 1,
                    reason=f"signal-freshness audit failed (stale: {audit['stale_apps']}) "
                           "— no stop is acted on from stale signals")

    spot = oc.fetch_spot()
    # a sleeve stopped out by an earlier run of this job is no longer in the
    # book but must still be recognised, so a re-run reports "already applied"
    ev = evaluate_stops(results, spot, now=now,
                        held_keys=set(held_w) | set(prior.get("stopped_keys") or []))
    trig = ev["triggered"]
    as_of = cur.get("as_of")
    for t in trig:
        print(f"  🛑 {t['key']:5s} live {t['live']:>10,.2f} ≤ {t['kind']} stop "
              f"{t['level']:>10,.2f} ({t['dist_pct']:+.2f} %, {t['src']})")
    for w in ev["watching"][:WATCH_ROWS]:
        print(f"  ·  {w['key']:5s} live {w['live']:>10,.2f}  {w['kind']} stop "
              f"{w['level']:>10,.2f}  ({w['dist_pct']:+.2f} % above)")
    for s in ev["skipped"]:
        print(f"  ?  {s['key']:5s} not evaluated — {s['reason']}")
    summary = dict(as_of=as_of, profile=profile, triggered=trig,
                   watching=ev["watching"][:WATCH_ROWS], skipped=ev["skipped"])

    if not trig:
        return done("clear", books_rewritten=False, **summary)
    keys = sorted(t["key"] for t in trig)
    if not args.force and set(keys) <= set(prior.get("stopped_keys") or []) \
            and all(held_w.get(k, 0) == 0 for k in keys):
        return done("already_applied", books_rewritten=False, **summary)
    if args.dry_run:
        print(f"[dry-run] would re-publish both books without: {', '.join(keys)}")
        return 0

    # ── rebuild: morning book pinned (adds-only), stopped sleeves forced out ──
    from ibkr_rebalance import compute_target_book
    import publish_target_book as ptb
    book = compute_target_book(profile, results=results, live_adjust=False,
                               prev_weights=held_w, force_exit=set(keys))
    book.actions = amend_actions(book.actions, trig)
    audit_stamp = dict(passed=True, checked_at_utc=audit["checked_at_utc"],
                       stale_apps=list(audit["stale_apps"]))
    basis = dict(anchor_ct=fr.fmt_ct(anchor),
                 equity_close=str(fr.expected_equity_asof(anchor).date()),
                 btc_bar_close_utc=fr.close_moment(
                     "crypto", fr.expected_crypto_asof(anchor)).isoformat())
    paper, live, live_w, live_cash = ptb.assemble_payloads(
        book, profile, audit_stamp=audit_stamp, basis=basis,
        generated_at_utc=base["checked_at_utc"])
    stamp = dict(checked_at_utc=base["checked_at_utc"],
                 checked_at_ct=base["checked_at_ct"],
                 stopped_keys=sorted(set(keys) | set(prior.get("stopped_keys") or [])),
                 stopped=[{k: t.get(k) for k in ("key", "kind", "level", "live",
                                                 "entry", "src")} for t in trig],
                 morning_generated_at_utc=(prior.get("morning_generated_at_utc")
                                           or cur.get("generated_at_utc")))
    paper["intraday_stop"] = stamp
    live["intraday_stop"] = dict(stamp)
    paper_path.write_text(tb.dumps(paper, secret))
    live_path.write_text(tb.dumps(live, secret))
    arch = tb.archive_book(live, live_path, secret)
    if arch:
        print(f"Replaced the dated as-of record → {arch}")
    print(f"\nRe-published both books without {', '.join(keys)} "
          f"(as-of {book.as_of}, profile {profile}):")
    for k in sorted(live_w, key=lambda k: -live_w[k]):
        print(f"    {k:5s} {live_w[k]*100:5.1f}%")
    print(f"    CASH  {live_cash*100:5.1f}%")
    return done("stopped", books_rewritten=True, **summary)


if __name__ == "__main__":
    raise SystemExit(main())
