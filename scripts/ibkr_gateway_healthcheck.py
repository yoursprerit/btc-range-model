"""Healthcheck: is IB Gateway up, logged in, and serving the right account(s)?

Connects to the gateway's API port and confirms it returns a paper (DU…) managed
account.  With ``--live`` it instead expects REAL (non-DU) accounts, and with
``--expect U1234567`` (repeatable, or ``--accounts-config``) that every named
account is managed by the login — the multi-account MOC executor's pre-check::

    python scripts/ibkr_gateway_healthcheck.py --live --port 4001 --accounts-config deploy/ibkr_accounts.json
 Exits 0 when healthy, 1 when not — so systemd (or cron) marks the run
failed. On failure it also POSTs an alert to ``IBKR_ALERT_WEBHOOK`` if set
(Slack / Discord / Mattermost / generic-compatible: it sends both ``text`` and
``content`` keys, so whichever the endpoint reads is present).

    # health probe (used by the systemd timer):
    python scripts/ibkr_gateway_healthcheck.py --port 4004

    # send an arbitrary alert message (used by the OnFailure alert unit):
    python scripts/ibkr_gateway_healthcheck.py --message "ibkr-executor.service failed"

Detecting "logged in": a gateway process may accept a socket connection yet not
be authenticated. So we don't just check the port — we open an IBKR API session
and require a non-empty managed-account list, and that it's a paper account.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import urllib.request


def send_alert(msg: str, webhook: str | None) -> None:
    """Print the message (→ journald) and POST it to the webhook if configured."""
    print(msg, file=sys.stderr, flush=True)
    if not webhook:
        return
    try:
        data = json.dumps({"text": msg, "content": msg}).encode()
        req = urllib.request.Request(
            webhook, data=data, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=15)   # noqa: S310 (user-configured URL)
        print("alert delivered to IBKR_ALERT_WEBHOOK")
    except Exception as e:                          # never let alerting crash the check
        print(f"could not deliver webhook alert: {e}", file=sys.stderr)


def check(host: str, port: int, client_id: int, paper_prefix: str,
          webhook: str | None, live: bool = False,
          expect: list[str] | None = None) -> int:
    where = f"{host}:{port}"

    # 1) fast socket probe — is anything listening at all?
    try:
        with socket.create_connection((host, port), timeout=5):
            pass
    except OSError as e:
        send_alert(f"🔴 IB Gateway healthcheck FAILED — nothing listening on {where} "
                   f"({e}). Gateway is down or still starting.", webhook)
        return 1

    # 2) real API session — a socket alone doesn't mean it's authenticated.
    try:
        from ib_async import IB
    except Exception as e:
        send_alert(f"🔴 IB Gateway healthcheck ERROR — ib_async not importable ({e}).",
                   webhook)
        return 1

    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id, timeout=20)
    except Exception as e:
        send_alert(f"🔴 IB Gateway healthcheck FAILED — cannot open an API session on "
                   f"{where} ({e}). The gateway is likely not logged in.", webhook)
        return 1
    try:
        accts = ib.managedAccounts()
        if not accts:
            send_alert(f"🔴 IB Gateway on {where} connected but returned NO accounts — "
                       f"the session is not authenticated (login/2FA failed?).", webhook)
            return 1
        if live:
            paper = [a for a in accts if a.startswith(paper_prefix)]
            if paper:
                send_alert(f"🟠 IB Gateway on {where} is logged in to PAPER account(s) "
                           f"{', '.join(paper)} but --live expects real accounts. "
                           f"Check TRADING_MODE / the login.", webhook)
                return 1
            missing = [a for a in (expect or []) if a not in accts]
            if missing:
                send_alert(f"🔴 IB Gateway on {where} does not manage {', '.join(missing)} "
                           f"(manages: {', '.join(accts)}). Wrong login, or the "
                           f"account is not linked to it.", webhook)
                return 1
            print(f"🟢 healthy — {where} logged in, live account(s) {', '.join(accts)}")
            return 0
        acct = accts[0]
        if not acct.startswith(paper_prefix):
            send_alert(f"🟠 IB Gateway on {where} is logged in, but account '{acct}' is "
                       f"NOT a paper account ({paper_prefix}…). Check TRADING_MODE.", webhook)
            return 1
        print(f"🟢 healthy — {where} logged in, paper account {acct}")
        return 0
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description="IB Gateway login healthcheck + alerter")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=4004,
                    help="gateway paper API port (Docker image maps paper→4004)")
    ap.add_argument("--client-id", type=int, default=19,
                    help="distinct from the executor's client id to avoid clashes")
    ap.add_argument("--paper-prefix", default="DU")
    ap.add_argument("--live", action="store_true",
                    help="expect REAL accounts (no DU…) instead of a paper one; "
                         "use with --port 4001")
    ap.add_argument("--expect", action="append", default=[],
                    help="with --live: an account id the login must manage "
                         "(repeatable)")
    ap.add_argument("--accounts-config",
                    help="with --live: take the expected ids from the enabled "
                         "accounts in this deploy/ibkr_accounts.json")
    ap.add_argument("--message",
                    help="don't probe — just send this text to IBKR_ALERT_WEBHOOK "
                         "(used by the systemd OnFailure alert unit) and exit 1")
    args = ap.parse_args()

    webhook = os.environ.get("IBKR_ALERT_WEBHOOK")
    if args.message:
        send_alert(f"🔴 {args.message}", webhook)
        return 1
    expect = list(args.expect)
    if args.accounts_config:
        try:
            raw = json.loads(open(args.accounts_config).read())
            expect += [a["id"] for a in raw.get("accounts", [])
                       if a.get("enabled", True)]
        except (OSError, ValueError, KeyError) as e:
            send_alert(f"🔴 healthcheck cannot read {args.accounts_config}: {e}", webhook)
            return 1
    return check(args.host, args.port, args.client_id, args.paper_prefix, webhook,
                 live=args.live, expect=expect)


if __name__ == "__main__":
    raise SystemExit(main())
