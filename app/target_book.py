"""Target-book artifact — the JSON contract between the *publisher* (the cloud
side that runs the model) and the *executor* (the local box next to IB Gateway).

This is the heart of **Option C**: the strategy decision (which instruments, at
what weights) is computed once by ``scripts/publish_target_book.py`` and written
as a small, self-contained, optionally-signed JSON document.  A lightweight
executor (``scripts/ibkr_execute_book.py``) later reads that document and trades
it against an IBKR paper account — WITHOUT re-running the model.

Design notes
------------
* **Self-contained.** The book carries everything the executor needs to size and
  place orders: target weights, the publish-time execution prices, the signal's
  as-of bar, and a generation timestamp.  The executor never runs the universe.
* **Dependency-light.** This module imports only the stdlib + pandas, so the
  executor host needs none of the model stack.  It does NOT import
  ``overall_core``.
* **Optionally signed.** Because the transport may be public (a commit on a
  branch, a raw URL), an HMAC-SHA256 signature over the canonical payload lets
  the executor verify the book was produced by someone holding the shared secret
  before it places a single order.  Signing is skipped (with a warning) when no
  secret is configured.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCHEMA = "overall-target-book/v1"
SIG_ALG = "HMAC-SHA256"


def prev_path(path: Path) -> Path:
    """``target_book.json`` → ``target_book_prev.json`` (same directory)."""
    path = Path(path)
    return path.with_name(path.stem + "_prev" + path.suffix)


# ── dated as-of archive ──────────────────────────────────────────────────────
# One JSON per signal day (keyed by the book's ``as_of``), so the UI's
# Historical View can show the book that was ACTUALLY published from a past
# bar's committed signals — the as-of record — instead of only a current-
# weights projection.  An intraday re-publish for the same ``as_of`` replaces
# that day's record: last publish wins, matching what the executor traded.
ARCHIVE_DIRNAME = "book_archive"


def archive_path(book_path: Path, as_of) -> Path:
    """``…/target_book_live.json`` + as_of → ``…/book_archive/<as_of>.json``."""
    return (Path(book_path).parent / ARCHIVE_DIRNAME
            / f"{pd.Timestamp(as_of).date()}.json")


def archive_book(payload: dict, book_path: Path,
                 secret: str | None = None) -> Path | None:
    """Persist *payload* as the dated as-of record next to *book_path*.

    Signed with *secret* when given (same HMAC as the live book), so archived
    records stay verifiable.  Best-effort: a failure to archive never blocks a
    publish.  Returns the path written, or None."""
    try:
        as_of = payload.get("as_of")
        if not as_of:
            return None
        p = archive_path(book_path, as_of)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(dumps(payload, secret))
        return p
    except Exception:
        return None


_ROTATE_TZ = "America/Chicago"     # the publish cycle's anchor timezone


def rotate_prev(path: Path, now=None) -> bool:
    """Preserve the outgoing book at *path* as ``*_prev.json`` before a new one
    lands, so the UI's *Previous Targetbook* always shows **yesterday's** book.

    Day-aware (America/Chicago, the 7:15-AM-CT publish anchor): the outgoing
    book is rotated only when it was generated on an EARLIER Central-time day
    than *now*.  An intraday re-publish (the UI's 🚀 button) therefore replaces
    today's book WITHOUT clobbering the previous-day book in the prev slot —
    "Previous Targetbook" keeps meaning yesterday's, not "an hour ago's".

    Returns True when the prev file was (re)written.  Best-effort: a
    missing/unreadable old book never blocks a publish."""
    path = Path(path)
    try:
        if not path.exists():
            return False
        text = path.read_text()
        try:
            gen = json.loads(text).get("generated_at_utc")
            gen_ts = pd.Timestamp(gen)
            if gen_ts.tzinfo is None:
                gen_ts = gen_ts.tz_localize("UTC")
            now_ts = pd.Timestamp(now) if now is not None else \
                pd.Timestamp(datetime.now(timezone.utc))
            if now_ts.tzinfo is None:
                now_ts = now_ts.tz_localize("UTC")
            if gen_ts.tz_convert(_ROTATE_TZ).date() >= \
                    now_ts.tz_convert(_ROTATE_TZ).date():
                return False               # same-day re-publish — keep yesterday's prev
        except Exception:
            pass                           # unparsable stamp → rotate (old behaviour)
        prev_path(path).write_text(text)
        return True
    except Exception:
        pass
    return False


# ════════════════════════════════════════════════════════════════════════════
# BUILD / SERIALISE
# ════════════════════════════════════════════════════════════════════════════
def build_payload(*, as_of: str, profile: str, weights: dict[str, float],
                  cash_weight: float, exec_price: dict[str, float],
                  actions: list[dict] | None = None,
                  generated_at_utc: str | None = None,
                  book_mode: str = "paper") -> dict:
    """Assemble a v1 target-book payload (unsigned).

    ``actions`` are trimmed to the fields the executor / audit log care about, so
    the artifact stays compact and stable regardless of engine-internal extras.
    ``book_mode`` is ``paper`` (idle → cash) or ``live`` (idle → a SATA position).
    """
    trimmed = [
        {k: a.get(k) for k in ("key", "action", "decision", "target",
                               "in_pos", "priority", "exits_next_bar")}
        for a in (actions or [])
    ]
    return {
        "schema": SCHEMA,
        "generated_at_utc": generated_at_utc or datetime.now(timezone.utc)
                                                        .isoformat(timespec="seconds"),
        "as_of": as_of,
        "profile": profile,
        "book_mode": book_mode,
        "weights": {k: float(v) for k, v in weights.items()},
        "cash_weight": float(cash_weight),
        "exec_price": {k: float(v) for k, v in exec_price.items()},
        "actions": trimmed,
    }


def _canonical(payload: dict) -> bytes:
    """Deterministic byte-encoding of the payload MINUS its signature, so signer
    and verifier hash exactly the same bytes regardless of key order."""
    body = {k: v for k, v in payload.items() if k != "signature"}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()


def sign(payload: dict, secret: str | None) -> dict:
    """Return ``payload`` with a ``signature`` block attached (or unchanged if no
    secret).  Mutates a copy — the input is not modified."""
    out = dict(payload)
    out.pop("signature", None)
    if not secret:
        return out
    mac = hmac.new(secret.encode(), _canonical(out), hashlib.sha256).hexdigest()
    out["signature"] = {"alg": SIG_ALG, "value": mac}
    return out


def dumps(payload: dict, secret: str | None = None) -> str:
    return json.dumps(sign(payload, secret), indent=1)


# ════════════════════════════════════════════════════════════════════════════
# LOAD / VERIFY / VALIDATE
# ════════════════════════════════════════════════════════════════════════════
def loads(text: str) -> dict:
    return json.loads(text)


def verify_signature(payload: dict, secret: str | None) -> tuple[bool, str]:
    """(ok, reason).  With a secret: recomputes the HMAC and constant-time
    compares.  Without a secret: accepts, but flags that the book is unverified."""
    sig = payload.get("signature")
    if not secret:
        return True, ("signed but no secret provided to verify" if sig
                      else "unsigned (no secret configured)")
    if not sig or "value" not in sig:
        return False, "no signature present but a secret is configured"
    expect = hmac.new(secret.encode(), _canonical(payload), hashlib.sha256).hexdigest()
    if hmac.compare_digest(expect, str(sig.get("value", ""))):
        return True, "signature OK"
    return False, "signature MISMATCH — book was tampered with or wrong secret"


def validate(payload: dict, today: pd.Timestamp, *, max_bar_age_days: int = 4,
             max_gen_age_hours: float = 36.0) -> tuple[bool, str]:
    """(ok, reason) structural + freshness checks, independent of the broker.

    The book is published ONCE daily (≈7:15 AM CT) and intentionally frozen
    until the next morning, so the generation-age window must span a full
    publish cycle plus slack — measured from the publish anchor to the
    executor's slot.  With the executor at 2:30 PM CT, yesterday's book is
    31.25 h old when today's publish was withheld by a failed audit (7:15 AM
    CT → 2:30 PM CT next day), so the window is 36 h: it accepts that
    documented fallback with slack, while still rejecting a two-day-old book
    (55 h+).  It was 30 h while the executor ran at 8:45 AM CT; the two move
    together, so re-derive it if the slot changes again.

    Rejects a wrong schema, a stale signal bar (dead feed), a book generated
    too long ago (so a forgotten/queued artifact can't trade an old decision),
    or a book whose publish-time **signal-freshness audit failed** (the
    publisher normally withholds those; if one is produced anyway — e.g. via
    ``--allow-stale`` — it must never be traded).  Books published before the
    audit existed carry no ``signal_audit`` stamp and are accepted."""
    if payload.get("schema") != SCHEMA:
        return False, f"unexpected schema {payload.get('schema')!r} (want {SCHEMA})"
    aud = payload.get("signal_audit")
    if aud is not None and not aud.get("passed"):
        stale = ", ".join(aud.get("stale_apps") or []) or "unknown"
        return False, f"signal-freshness audit FAILED at publish (stale: {stale})"
    as_of = payload.get("as_of")
    if not as_of:
        return False, "no as_of in book"
    bar_age = (today.normalize() - pd.Timestamp(as_of).normalize()).days
    if bar_age > max_bar_age_days:
        return False, f"signal bar {as_of} is {bar_age}d old (> {max_bar_age_days}d)"
    gen = payload.get("generated_at_utc")
    if gen:
        gen_ts = pd.Timestamp(gen)
        now = pd.Timestamp(datetime.now(timezone.utc))
        if gen_ts.tzinfo is None:
            gen_ts = gen_ts.tz_localize("UTC")
        age_h = (now - gen_ts).total_seconds() / 3600.0
        if age_h > max_gen_age_hours:
            return False, (f"book generated {age_h:.1f}h ago "
                           f"(> {max_gen_age_hours}h) — stale, refusing")
    return True, f"valid (bar {as_of}, {bar_age}d old)"
