"""Dead-man's switch: tell an outside monitor that a scheduled run happened.

Every other alert in this project fires from a run that STARTED.  If the PC is
off or asleep, Task Scheduler never fires, the gateway login lapsed before the
run, or Windows restarted for an update, no run exists to complain -- the day is
lost in silence.  A heartbeat inverts that: each run pings an external monitor,
and the MONITOR raises the alarm when an expected ping does not arrive.

    IBKR_HEARTBEAT_URL            URL of the run's check (a secret: it is a bearer
                                  token -- keep it in an environment variable)
    IBKR_RECONCILE_HEARTBEAT_URL  the same for the after-the-close reconcile job
    IBKR_HEARTBEAT_STYLE          healthchecks (default) | kuma

Styles
    healthchecks   healthchecks.io and compatible services: ``<url>/start`` when a
                   run begins, ``<url>`` when it succeeds, ``<url>/fail`` when it
                   does not, with a short text body.
    kuma           Uptime Kuma "push" monitors: ``<url>?status=up|down&msg=...``
                   (no start signal).

Set the monitor up as a DAILY check (the task runs every day; a weekend or a
market holiday is a normal, successful "nothing to do" run that still pings),
with a grace period of 30-45 minutes.  Messages carry counts and statuses with
masked account ids -- never balances or holdings.

Pure stdlib; ``ping`` never raises -- monitoring must not be able to stop a run.
"""
from __future__ import annotations

import os
import urllib.parse
import urllib.request
from urllib.error import URLError

STYLES = ("healthchecks", "kuma")
MAX_BODY = 4000                     # healthchecks keeps 100 KB; a summary is short


def mask_account(acct: str) -> str:
    """``U1234567`` -> ``…4567``: enough to tell accounts apart, not to name one."""
    acct = str(acct or "")
    return f"…{acct[-4:]}" if len(acct) > 4 else acct


def valid_url(url: str | None) -> bool:
    """https only -- the URL is a bearer token and must not travel in clear text
    (plain http is allowed for localhost, which the tests use)."""
    if not url:
        return False
    try:
        u = urllib.parse.urlparse(url)
    except ValueError:
        return False
    if u.scheme == "https" and u.netloc:
        return True
    return u.scheme == "http" and u.hostname in ("127.0.0.1", "localhost")


def _request_url(url: str, kind: str, style: str, message: str) -> tuple[str, bytes | None]:
    """(final URL, POST body or None) for a ping of *kind* (start / ok / fail)."""
    url = url.rstrip("/")
    msg = (message or "")[:MAX_BODY]
    if style == "kuma":
        status = "down" if kind == "fail" else "up"
        q = urllib.parse.urlencode({"status": status, "msg": msg[:200] or kind})
        sep = "&" if "?" in url else "?"
        return f"{url}{sep}{q}", None
    suffix = {"start": "/start", "fail": "/fail"}.get(kind, "")
    return url + suffix, msg.encode("utf-8")


def ping(kind: str, message: str = "", *, url: str | None = None,
         style: str | None = None, env_var: str = "IBKR_HEARTBEAT_URL",
         timeout: float = 10.0, retries: int = 2, opener=None) -> bool:
    """Send one heartbeat.  Returns True if the monitor acknowledged it.

    No URL configured, an unusable URL, or any network failure -> False, quietly:
    a monitoring outage must never stop a trading run.  *kind* is ``start``,
    ``ok`` or ``fail``."""
    url = url if url is not None else os.environ.get(env_var, "").strip()
    if not valid_url(url):
        return False
    style = (style or os.environ.get("IBKR_HEARTBEAT_STYLE", "") or "healthchecks").lower()
    if style not in STYLES:
        style = "healthchecks"
    if style == "kuma" and kind == "start":
        return False                                   # no start signal in that API
    target, body = _request_url(url, kind, style, message)
    open_ = opener or urllib.request.urlopen
    for _ in range(max(retries, 0) + 1):
        try:
            req = urllib.request.Request(
                target, data=body, method="POST" if body is not None else "GET",
                headers={"User-Agent": "btc-range-executor",
                         "Content-Type": "text/plain; charset=utf-8"})
            with open_(req, timeout=timeout) as r:     # noqa: S310 (https, validated)
                if 200 <= getattr(r, "status", 200) < 300:
                    return True
        except (URLError, OSError, ValueError):
            continue
    return False
