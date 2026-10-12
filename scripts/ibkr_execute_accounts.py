"""Multi-account Option-C runner — trade the published live book in EVERY
configured IBKR account on this PC, as market-on-close orders.

    # preview (default — no orders):
    python scripts/ibkr_execute_accounts.py
    # check the config and that the gateway manages every account, nothing else:
    python scripts/ibkr_execute_accounts.py --check
    # trade (what the scheduled wrapper runs):
    python scripts/ibkr_execute_accounts.py --execute

Why a runner and not a loop in the executor.  A MOC order has to be ENTERED
before the exchange cutoff (15:50 ET) but does not fill until the 16:00
auction, and ``ibkr_execute_book.py`` holds its connection open through the
auction to write a truthful report (~30 min).  Running N accounts one after
another would therefore push account 2 past the cutoff.  So every account runs
in its own executor PROCESS, all started together against the one IB Gateway
(one client id each): all orders go in within minutes of each other, and the
auction fills them all.  An account that fails or hangs cannot touch another.

Each child is the ordinary single-account executor, pinned with ``--account`` /
``--expected-account``, ``--order-type moc --strict-moc`` (no limit/market
fallback; a late run skips the session) and ``--require-signature``.  They keep
every guard the paper trial hardened: verified positions read, current-bar
check, duplicate-run lock, working-orders check, cash-funded buys, exposure /
turnover / price-drift caps.  Reports go to ``data/overall/accounts/<ACCT>/`` and
are committed in ONE push at the end (the children are started with
``--no-push-report``).

Exit code: 0 all accounts fine · 3 whole session skipped (MOC cutoff passed /
not a trading day) · 2 gateway or config problem, nothing traded ·
1 at least one account failed or did not fully fill.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO / "app"))

import pandas as pd                                    # noqa: E402

import ibkr_common as ic                               # noqa: E402
import ibkr_symbols as sym                             # noqa: E402
import ibkr_heartbeat as hbeat                         # noqa: E402

DEFAULT_CONFIG = _REPO / "deploy" / "ibkr_accounts.json"
EXECUTOR = _HERE / "ibkr_execute_book.py"
ACCOUNTS_DIR = _REPO / "data" / "overall" / "accounts"
LOG_DIR = _REPO / "logs" / "accounts"

EXIT_OK, EXIT_FAILED, EXIT_SETUP, EXIT_SKIPPED = 0, 1, 2, 3

ACCOUNT_TYPES = ("margin", "ira", "cash")
LIVE_ID = re.compile(r"^[UFI]\d{5,}$")      # a REAL account (U individual/joint/IRA, F/I advisor); paper is D…

# Per-account settings and their defaults.  Anything here can be overridden
# globally under "defaults" and again per account.
DEFAULT_LIMITS = {
    "band": 0.01,                # no-trade band, fraction of the account's net-liq
    "max_deploy_frac": 1.0,      # cap on the TOTAL invested, all assets incl. SATA; rest = cash
    "max_order_notional": 0.0,   # $ cap on any single order (0 = none)
    "max_gross_frac": ic.DEFAULT_MAX_GROSS_FRAC,
    "max_turnover_frac": 1.5,
    "max_price_drift": 0.25,
}


# ── config ──────────────────────────────────────────────────────────────────
@dataclass
class Account:
    id: str
    label: str = ""
    type: str = "margin"
    enabled: bool = True
    allow_margin: bool = False
    limits: dict = field(default_factory=dict)
    # restricted key -> key traded instead (an IRA cannot buy UGL -> GLDM)
    substitutions: dict = field(default_factory=dict)
    # the risk profile this account is meant to follow ("" = not pinned). A book
    # published for any other profile aborts this account's run.
    expected_profile: str = ""


@dataclass
class Config:
    host: str = "127.0.0.1"
    port: int = ic.DEFAULT_LIVE_PORT
    client_id_base: int = 40
    book: str = "data/overall/target_book_live.json"
    child_timeout_s: float = 3900.0   # = ic.MOC_MAX_WAIT_S + 300 (enforced below)
    stagger_s: float = 10.0
    market_data_type: int = 1      # 1 live, 2 frozen, 3 delayed, 4 delayed-frozen
    kill_switch_file: str = ""
    accounts: list[Account] = field(default_factory=list)

    def enabled(self) -> list[Account]:
        return [a for a in self.accounts if a.enabled]


def validate_config(raw: dict) -> tuple[Config | None, list[str]]:
    """(Config, errors) — every problem is reported at once, and any error
    means no Config.  Conservative on purpose: this file decides which REAL
    accounts get orders."""
    errs: list[str] = []
    if not isinstance(raw, dict):
        return None, ["config root must be a JSON object"]
    cfg = Config()
    for k, cast in (("host", str), ("book", str), ("kill_switch_file", str),
                    ("port", int), ("client_id_base", int),
                    ("child_timeout_s", float), ("stagger_s", float),
                    ("market_data_type", int)):
        if k in raw:
            try:
                setattr(cfg, k, cast(raw[k]))
            except (TypeError, ValueError):
                errs.append(f"'{k}' must be {cast.__name__}")
    if not (1 <= cfg.port <= 65535):
        errs.append(f"port {cfg.port} is not a valid TCP port")
    if cfg.client_id_base < 1:
        errs.append("client_id_base must be >= 1")
    if cfg.market_data_type not in (1, 2, 3, 4):
        errs.append("market_data_type must be 1 (live), 2, 3 (delayed) or 4")
    if cfg.child_timeout_s < 600:
        errs.append("child_timeout_s must allow for the closing auction (>= 600)")
    # Never kill an executor that is still waiting for the auction: whatever the
    # config says, the timeout is at least the MOC wait ceiling plus time to
    # start, read the account and write the report. (A config copied from the
    # old example carried 3000 s, which is shorter than a full wait.)
    cfg.child_timeout_s = max(cfg.child_timeout_s, ic.MOC_MAX_WAIT_S + 300.0)

    defaults = dict(DEFAULT_LIMITS)
    for k, v in (raw.get("defaults") or {}).items():
        if k not in DEFAULT_LIMITS:
            errs.append(f"defaults: unknown setting '{k}'")
        else:
            defaults[k] = v

    seen: set[str] = set()
    items = raw.get("accounts")
    if not isinstance(items, list) or not items:
        errs.append("'accounts' must be a non-empty list")
        items = []
    for i, it in enumerate(items):
        where = f"accounts[{i}]"
        if not isinstance(it, dict):
            errs.append(f"{where}: must be an object")
            continue
        aid = str(it.get("id", "")).strip()
        where = f"accounts[{i}] ({aid or '?'})"
        if aid.startswith("D"):
            errs.append(f"{where}: a paper (D…) account — this runner is live-only")
        elif not LIVE_ID.match(aid):
            errs.append(f"{where}: id must look like a LIVE account id, e.g. U1234567")
        if aid in seen:
            errs.append(f"{where}: duplicate account id")
        seen.add(aid)
        a = Account(id=aid, label=str(it.get("label", "")),
                    type=str(it.get("type", "margin")).lower(),
                    enabled=bool(it.get("enabled", True)),
                    allow_margin=bool(it.get("allow_margin", False)))
        ep = it.get("expected_profile", "")
        if not isinstance(ep, str):
            errs.append(f"{where}: expected_profile must be a string")
        else:
            a.expected_profile = ep.strip()
        if a.type not in ACCOUNT_TYPES:
            errs.append(f"{where}: type must be one of {', '.join(ACCOUNT_TYPES)}")
        if a.allow_margin and a.type != "margin":
            errs.append(f"{where}: allow_margin is only valid for a margin account")
        # type defaults (ira: UGL -> GLDM), overridable per account; an empty
        # {} turns the default off.
        subs = it.get("substitutions", sym.substitutions_for(a.type))
        try:
            a.substitutions = ic.parse_substitutions(
                [f"{k}={v}" for k, v in dict(subs).items()])
        except (ValueError, TypeError) as e:
            errs.append(f"{where}: substitutions: {e}")
        lim = dict(defaults)
        for k, v in it.items():
            if k in DEFAULT_LIMITS:
                lim[k] = v
            elif k not in ("id", "label", "type", "enabled", "allow_margin",
                           "substitutions", "expected_profile"):
                errs.append(f"{where}: unknown setting '{k}'")
        for k, v in lim.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                errs.append(f"{where}: {k} must be a number")
        try:
            if not 0 < float(lim["max_deploy_frac"]) <= 1:
                errs.append(f"{where}: max_deploy_frac must be in (0, 1]")
            if not 0 <= float(lim["band"]) <= 0.2:
                errs.append(f"{where}: band must be in [0, 0.2]")
            if float(lim["max_order_notional"]) < 0:
                errs.append(f"{where}: max_order_notional must be >= 0")
            if not 1.0 <= float(lim["max_gross_frac"]) <= 1.25:
                errs.append(f"{where}: max_gross_frac must be in [1.0, 1.25]")
            if float(lim["max_turnover_frac"]) <= 0:
                errs.append(f"{where}: max_turnover_frac must be > 0")
        except (TypeError, ValueError):
            pass                                  # already reported as non-numeric
        a.limits = lim
        cfg.accounts.append(a)
    if not errs and not cfg.enabled():
        errs.append("no account is enabled")
    return (None if errs else cfg), errs


def load_config(path: Path) -> tuple[Config | None, list[str]]:
    try:
        raw = json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None, [f"config not found: {path} — copy "
                      "deploy/ibkr_accounts.example.json to it and edit"]
    except ValueError as e:
        return None, [f"config is not valid JSON: {e}"]
    return validate_config(raw)


def account_dir(acct: str) -> Path:
    return ACCOUNTS_DIR / acct


def child_argv(cfg: Config, acct: Account, index: int, *, execute: bool,
               python: str = sys.executable) -> list[str]:
    """The command line for one account's executor."""
    lim = acct.limits
    argv = [python, str(EXECUTOR),
            "--file", str(_REPO / cfg.book) if not Path(cfg.book).is_absolute()
            else cfg.book,
            "--host", cfg.host, "--port", str(cfg.port),
            "--client-id", str(cfg.client_id_base + index),
            "--account", acct.id, "--expected-account", acct.id,
            "--account-mode", "live", "--confirm-live",
            "--order-type", ic.ORDER_MOC, "--strict-moc",
            "--require-signature", "--no-push-report",
            "--market-data-type", str(cfg.market_data_type),
            "--band", str(lim["band"]),
            "--max-deploy-frac", str(lim["max_deploy_frac"]),
            "--max-order-notional", str(lim["max_order_notional"]),
            "--max-gross-frac", str(lim["max_gross_frac"]),
            "--max-turnover-frac", str(lim["max_turnover_frac"]),
            "--max-price-drift", str(lim["max_price_drift"])]
    if cfg.kill_switch_file:
        argv += ["--kill-switch-file", cfg.kill_switch_file]
    if acct.expected_profile:
        argv += ["--expected-profile", acct.expected_profile]
    if acct.allow_margin and acct.type == "margin":
        argv.append("--allow-margin")
    for src, dst in sorted(acct.substitutions.items()):
        argv += ["--substitute", f"{src}={dst}"]
    if execute:
        argv.append("--execute")
    else:
        # A preview must not overwrite the account's last REAL execution report
        # with a list of planned trades.
        argv.append("--no-report")
    return argv


# ── gateway check ───────────────────────────────────────────────────────────
def _connect_once(cfg: Config) -> tuple[list[str] | None, str]:
    """One short-lived API session: (managed accounts, reason) or (None, reason)."""
    try:
        from ib_async import IB
    except Exception as e:
        return None, f"ib_async not importable ({e}) — pip install -r requirements-ibkr.txt"
    ib = IB()
    try:
        ib.connect(cfg.host, cfg.port,
                   clientId=cfg.client_id_base + len(cfg.accounts) + 5, timeout=20)
        accts = list(ib.managedAccounts() or [])
    except Exception as e:
        return None, (f"cannot open an API session on {cfg.host}:{cfg.port} ({e}) — "
                      "IB Gateway is down or not logged in")
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass
    if not accts:
        return None, "the gateway returned no accounts (login/2FA incomplete?)"
    return accts, f"gateway managed accounts: {', '.join(accts)}"


GATEWAY_RETRY_S = 600.0        # keep trying this long (execute runs only)
GATEWAY_RETRY_EVERY_S = 30.0
MOC_RETRY_MARGIN_MIN = 6       # stop retrying this close to the exchange cutoff


def verify_gateway(cfg: Config, retry_s: float = 0.0,
                   every_s: float = GATEWAY_RETRY_EVERY_S,
                   sleep=time.sleep, now=None, clock=time.monotonic,
                   respect_cutoff: bool = True) -> tuple[list[str] | None, str]:
    """(managed accounts, reason), retrying for up to *retry_s* seconds.

    A gateway that is momentarily unavailable (restarting, a slow login) must not
    cost the whole session: the run starts ~35 minutes before the MOC cutoff, so
    there is time to try again.  Retries stop early once an order placed after
    the next attempt could no longer beat the cutoff."""
    t0 = clock()
    attempt = 0
    while True:
        attempt += 1
        accts, why = _connect_once(cfg)
        if accts is not None or retry_s <= 0:
            return accts, why
        waited = clock() - t0
        late = respect_cutoff and not ic.moc_entry_open(
            (pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now))
            + pd.Timedelta(minutes=MOC_RETRY_MARGIN_MIN))[0]
        if waited + every_s > retry_s or late:
            return None, (f"{why} (gave up after {attempt} attempt(s), "
                          f"{waited:.0f}s)")
        print(f"  gateway not ready — {why}\n  retrying in {every_s:.0f}s "
              f"(attempt {attempt + 1})…", flush=True)
        sleep(every_s)


def split_by_availability(cfg: Config, managed: list[str]
                          ) -> tuple[list[Account], list[Account]]:
    """(runnable, missing): enabled accounts the login manages, and those it
    does not.  A missing account is skipped and reported — it never blocks the
    others, and it is never "approximated" by trading a different one."""
    ok, missing = [], []
    for a in cfg.enabled():
        (ok if a.id in managed else missing).append(a)
    return ok, missing


# ── children ────────────────────────────────────────────────────────────────
@dataclass
class Result:
    acct: Account
    rc: int | None = None
    note: str = ""
    seconds: float = 0.0


def describe(rc: int | None) -> str:
    return {0: "ok", 1: "ABORTED/FAILED — a guard refused or an error occurred",
            3: "skipped (MOC cutoff passed)",
            4: "orders sent but NOT fully filled — check TWS",
            5: "orders filled but the account did NOT land on target — check TWS",
            None: "did not finish (timed out and was killed)"}.get(rc, f"exit {rc}")


def _tag(acct: Account) -> str:
    return f"[{acct.id}]"


def rotate_log(path: Path, max_bytes: int = 5_000_000, keep: int = 3) -> None:
    """Size-based rotation (path -> path.1 -> path.2 ...), so a per-account log
    that holds account ids and balances does not grow without bound."""
    try:
        if not path.exists() or path.stat().st_size < max_bytes:
            return
        for i in range(keep - 1, 0, -1):
            src = path.with_name(f"{path.name}.{i}")
            if src.exists():
                src.replace(path.with_name(f"{path.name}.{i + 1}"))
        path.replace(path.with_name(f"{path.name}.1"))
    except OSError:
        pass                               # never let log housekeeping stop a run


def run_child(cfg: Config, acct: Account, index: int, *, execute: bool,
              echo_lock: threading.Lock, results: dict[str, Result]) -> None:
    res = results[acct.id] = Result(acct)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rotate_log(LOG_DIR / f"{acct.id}.log")
    log = (LOG_DIR / f"{acct.id}.log").open("a", encoding="utf-8")
    argv = child_argv(cfg, acct, index, execute=execute)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               PYTHONUNBUFFERED="1")  # a piped child otherwise holds its log until exit
    t0 = time.monotonic()
    log.write(f"\n===== {pd.Timestamp.now('UTC'):%Y-%m-%d %H:%M:%SZ} "
              f"{' '.join(argv[2:])}\n")
    try:
        p = subprocess.Popen(argv, cwd=_REPO, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace")
    except OSError as e:
        res.rc, res.note = 1, f"could not start executor: {e}"
        log.close()
        return

    def _kill():
        res.note = f"killed after {cfg.child_timeout_s:.0f}s"
        p.kill()
    timer = threading.Timer(cfg.child_timeout_s, _kill)
    timer.start()
    try:
        for line in p.stdout:                       # type: ignore[union-attr]
            log.write(line)
            with echo_lock:
                print(f"{_tag(acct)} {line.rstrip()}", flush=True)
        p.wait()
    finally:
        timer.cancel()
        log.close()
    res.seconds = time.monotonic() - t0
    res.rc = None if res.note.startswith("killed") else p.returncode


def run_all(cfg: Config, accounts: list[Account], *, execute: bool
            ) -> list[Result]:
    """Start every account's executor together (staggered a few seconds so the
    gateway is not hit by simultaneous logins), wait for all of them."""
    results: dict[str, Result] = {}
    lock = threading.Lock()
    threads = []
    index_of = {a.id: i for i, a in enumerate(cfg.accounts)}   # stable client ids
    for n, acct in enumerate(accounts):
        t = threading.Thread(target=run_child, name=acct.id, daemon=True,
                             args=(cfg, acct, index_of[acct.id]),
                             kwargs=dict(execute=execute, echo_lock=lock,
                                         results=results))
        t.start()
        threads.append(t)
        if n < len(accounts) - 1:
            time.sleep(cfg.stagger_s)
    for t in threads:
        t.join()
    return [results[a.id] for a in accounts if a.id in results]


def overall_exit(results: list[Result], missing: list[Account]) -> int:
    """Fold per-account outcomes into the runner's exit code."""
    if missing:
        return EXIT_FAILED
    if not results:
        return EXIT_SKIPPED
    if any(r.rc not in (0, 3) for r in results):
        return EXIT_FAILED
    if all(r.rc == 3 for r in results):
        return EXIT_SKIPPED
    return EXIT_OK


def summary(results: list[Result], missing: list[Account]) -> str:
    lines = ["", "Account summary:"]
    for r in results:
        lines.append(f"  {r.acct.id:<10} {r.acct.label[:22]:<22} "
                     f"{describe(r.rc)}" + (f"  ({r.note})" if r.note else "")
                     + (f"  [{r.seconds / 60:.0f} min]" if r.seconds else ""))
    for a in missing:
        lines.append(f"  {a.id:<10} {a.label[:22]:<22} NOT MANAGED by this login — "
                     "skipped (check the gateway login / account id)")
    return "\n".join(lines)


# ── publishing ──────────────────────────────────────────────────────────────
VAULT_DIR = _REPO / "data" / "overall" / "vault"


def vault_enabled() -> bool:
    """True when ``IBKR_VAULT_KEY`` is set: reports are then published only as an
    AES-256-GCM encrypted vault (``app/report_vault.py``), never in the clear."""
    import report_vault as rv
    return bool(rv.passphrase())


def publishing_enabled(no_push: bool = False) -> bool:
    """Reports hold account numbers, balances and holdings, so they stay on this
    PC unless you explicitly opt in with ``IBKR_PUBLISH_REPORTS=1``."""
    opted = (os.environ.get("IBKR_PUBLISH_REPORTS", "").strip() == "1"
             or vault_enabled())
    return (opted and not no_push and not os.environ.get("IBKR_NO_PUSH_REPORT"))


def push_reports(accounts: list[Account], branch: str | None = None) -> bool:
    """Commit every account's report + archive in ONE commit and push it.

    One push, not N: N concurrent children each pushing to the same branch is a
    race that loses reports.  On a rejected push it rebases the single report
    commit once onto the remote (account paths are ours alone, so it is a clean
    replay), and failing that it un-commits but KEEPS the files — a later run
    overwrites them anyway, and nothing is left diverged."""
    rels = []
    if vault_enabled():
        import report_vault as rv
        # seal EVERY account's report (the vault is rebuilt whole, so a partial
        # seal would drop the other accounts) and publish only the ciphertext
        if rv.seal_tree(ACCOUNTS_DIR.parent, VAULT_DIR, rv.passphrase()):
            rels.append(str(VAULT_DIR.relative_to(_REPO)))
    else:
        for a in accounts:
            d = account_dir(a.id)
            for p in (d / "executed_book_live.json", d / "executed_archive"):
                if p.exists():
                    rels.append(str(p.relative_to(_REPO)))
    if not rels:
        print("  no reports to publish")
        return True

    def git(*a):
        return subprocess.run(("git",) + a, cwd=_REPO, capture_output=True, text=True)
    branch = branch or (git("rev-parse", "--abbrev-ref", "HEAD").stdout or "main").strip()
    if branch == "HEAD":
        # detached: "HEAD:HEAD" would create a branch literally named HEAD
        print("  WARN: the repo is on a detached HEAD — not pushing the reports "
              "(git checkout main)")
        return False
    git("add", "-f", *rels)       # the report dir is git-ignored; publishing is an explicit opt-in
    if git("diff", "--cached", "--quiet", "--", *rels).returncode == 0:
        print("  reports unchanged — nothing to publish")
        return True
    msg = f"chore(ibkr): live execution reports {date.today()}"
    c = git("-c", "user.name=ibkr-executor", "-c", "user.email=executor@localhost",
            "commit", "-q", "-m", msg, "--", *rels)    # ONLY the reports: anything
    if c.returncode != 0:                              # else staged stays staged
        print(f"  WARN: could not commit reports ({c.stderr.strip()[:200]})")
        return False
    r = git("push", "origin", f"HEAD:{branch}")
    if r.returncode != 0:
        git("fetch", "--quiet", "origin", branch)
        rb = git("-c", "user.name=ibkr-executor", "-c", "user.email=executor@localhost",
                 "rebase", f"origin/{branch}")
        if rb.returncode == 0:
            r = git("push", "origin", f"HEAD:{branch}")
        else:
            git("rebase", "--abort")
    if r.returncode == 0:
        print(f"  published {len(rels)} report path(s) to origin/{branch}")
        return True
    print("  WARN: could not push the reports (this host needs git write access) — "
          "un-committing, files kept on disk")
    git("reset", "--mixed", "HEAD~1")
    return False


# ── alerting ────────────────────────────────────────────────────────────────
def alert(msg: str) -> None:
    try:
        from ibkr_gateway_healthcheck import send_alert
        send_alert(msg, os.environ.get("IBKR_ALERT_WEBHOOK"))
    except Exception:
        print(msg, file=sys.stderr, flush=True)


# ── main ────────────────────────────────────────────────────────────────────
class Heartbeat:
    """The run's dead-man's-switch pings (see ``ibkr_heartbeat``).

    Pings only for a real scheduled-style run (``--execute``): a preview or a
    ``--check`` typed by hand must never stand in for the missing daily run.  A
    run that had nothing to do for a normal reason (weekend, market holiday, kill
    switch) is a SUCCESS -- the task fired as scheduled -- while a skipped
    session (late), a gateway/config failure or any failed account is a FAIL."""

    def __init__(self):
        self.active = False
        self.benign = False          # a normal "nothing to do" skip
        self.detail = ""

    def begin(self):
        self.active = True
        hbeat.ping("start", "run started")

    def finish(self, code: int):
        if not self.active:
            return
        ok = code == EXIT_OK or (code == EXIT_SKIPPED and self.benign)
        hbeat.ping("ok" if ok else "fail",
                   self.detail or {EXIT_OK: "all accounts ok", EXIT_FAILED: "failed",
                                   EXIT_SETUP: "gateway or config problem",
                                   EXIT_SKIPPED: "session skipped"}.get(code, f"exit {code}"))


def main(argv: list[str] | None = None) -> int:
    """Run, then report the outcome to the dead-man's switch."""
    hb = Heartbeat()
    try:
        code = _run(argv, hb)
    except SystemExit:
        raise
    except BaseException as e:
        if hb.active:
            hbeat.ping("fail", f"crashed: {type(e).__name__}: {str(e)[:200]}")
        raise
    hb.finish(code)
    return code


def _run(argv: list[str] | None, hb: Heartbeat) -> int:
    ap = argparse.ArgumentParser(
        description="Trade the live Option-C book in every configured IBKR "
                    "account as market-on-close orders")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--execute", action="store_true",
                    help="place orders (default: preview only — no orders)")
    ap.add_argument("--check", action="store_true",
                    help="validate the config and the gateway login, then stop")
    ap.add_argument("--only", action="append", default=[],
                    help="restrict to this account id (repeatable)")
    ap.add_argument("--no-push", action="store_true",
                    help="do not commit/push the reports (env IBKR_NO_PUSH_REPORT=1 "
                         "does the same)")
    args = ap.parse_args(argv)
    if args.execute and not args.check:
        hb.begin()

    cfg, errs = load_config(Path(args.config))
    if errs or cfg is None:
        print("CONFIG ERROR:")
        for e in errs:
            print(f"  - {e}")
        return EXIT_SETUP
    if args.only:
        unknown = [o for o in args.only if o not in {a.id for a in cfg.accounts}]
        if unknown:
            print(f"CONFIG ERROR: --only names unknown account(s): {', '.join(unknown)}")
            return EXIT_SETUP
        for a in cfg.accounts:
            a.enabled = a.enabled and a.id in args.only

    mode = "EXECUTE" if args.execute else "preview (no orders)"
    print(f"Multi-account MOC runner — {mode}")
    print("Runtime: " + ic.runtime_versions())
    print(f"  gateway {cfg.host}:{cfg.port}, book {cfg.book}, "
          f"{len(cfg.enabled())} account(s) enabled")
    for a in cfg.enabled():
        lim = a.limits
        print(f"    {a.id}  {a.type:<6} {a.label:<24} deploy≤{lim['max_deploy_frac']*100:.0f}% "
              f"order≤{('$%s' % format(lim['max_order_notional'], ',.0f')) if lim['max_order_notional'] else 'no cap'}"
              f"{'  margin allowed' if a.allow_margin else ''}")

    if os.environ.get("IBKR_TRADING_DISABLED") or (
            cfg.kill_switch_file and Path(cfg.kill_switch_file).exists()):
        print("ABORT: trading is DISABLED (kill switch active). Nothing sent.")
        # Benign for the heartbeat (the task did fire), but never silent: a
        # switch thrown in a panic and forgotten loses every later session.
        alert("🛑 IBKR live run skipped: the kill switch is ACTIVE "
              f"({cfg.kill_switch_file or 'IBKR_TRADING_DISABLED'}). No orders "
              "were placed for any account — clear it to resume trading.")
        hb.benign, hb.detail = True, "kill switch active — nothing sent"
        return EXIT_SKIPPED

    if args.execute and not args.check:
        if not os.environ.get("OVERALL_BOOK_SECRET"):
            print("ABORT: OVERALL_BOOK_SECRET is not set — a live book must be "
                  "signature-verified before any order. Nothing sent.")
            return EXIT_SETUP
        today = pd.Timestamp.now(tz=ic.ET_TZ).tz_localize(None)
        if not ic.calendar_covers(today):
            print(f"ABORT: {ic.calendar_warning(today)}. Nothing sent.")
            alert("🔴 IBKR live run REFUSED — the market-holiday table has run out; "
                  "extend US_MARKET_HOLIDAYS / US_EARLY_CLOSES in scripts/ibkr_common.py.")
            return EXIT_SETUP
        _note = ic.calendar_warning(today)
        if _note:
            print(f"NOTE: {_note}")
        ok_day, why_day = ic.is_trading_day(today)
        if not ok_day:
            print(f"Not trading: {why_day}.")
            hb.benign, hb.detail = True, f"not trading: {why_day}"
            return EXIT_SKIPPED
        ok_moc, why_moc = ic.moc_entry_open()
        print(f"MOC window: {why_moc}")
        if not ok_moc:
            print("SKIP SESSION: strict MOC and the exchange cutoff has passed — "
                  "no account is touched (no limit/market fallback by design).")
            alert("🟠 IBKR live MOC run SKIPPED — started after the MOC cutoff "
                  f"({why_moc}). No orders were placed today.")
            hb.detail = f"SKIPPED — started after the MOC cutoff ({why_moc})"
            return EXIT_SKIPPED

    managed, why = verify_gateway(
        cfg, retry_s=GATEWAY_RETRY_S if (args.execute and not args.check) else 0.0)
    print(f"Gateway: {why}")
    if managed is None:
        alert(f"🔴 IBKR live MOC run ABORTED — {why}. Nothing was traded.")
        hb.detail = f"gateway: {why[:300]}"
        return EXIT_SETUP
    runnable, missing = split_by_availability(cfg, managed)
    for a in missing:
        print(f"  ✗ {a.id} ({a.label}) is NOT managed by this login — skipped")
    if missing:
        alert("🔴 IBKR live run: account(s) not managed by the gateway login — "
              + ", ".join(a.id for a in missing))
    if args.check:
        print("\nCheck complete — " + ("all accounts reachable." if not missing
              else f"{len(missing)} account(s) missing."))
        return EXIT_FAILED if missing else EXIT_OK
    if not runnable:
        print("ABORT: none of the enabled accounts is managed by this login.")
        return EXIT_SETUP

    print(f"\nStarting {len(runnable)} account(s) together …")
    results = run_all(cfg, runnable, execute=args.execute)
    print(summary(results, missing))
    code = overall_exit(results, missing)

    if args.execute and publishing_enabled(args.no_push):
        traded = [r.acct for r in results if r.rc in (0, 4, 5)]
        print("\nPublishing execution reports (IBKR_PUBLISH_REPORTS=1) …")
        push_reports(traded)
    elif args.execute:
        print("\nReports stay on this PC (account data is not published to GitHub).")
    if code == EXIT_FAILED:
        bad = [f"{r.acct.id}: {describe(r.rc)}" for r in results if r.rc not in (0, 3)]
        bad += [f"{a.id}: not managed by login" for a in missing]
        alert("🔴 IBKR live MOC run finished with problems — " + "; ".join(bad))
    hb.detail = "; ".join(
        [f"{hbeat.mask_account(r.acct.id)} {describe(r.rc)}" for r in results]
        + [f"{hbeat.mask_account(a.id)} not managed by login" for a in missing])
    return code


if __name__ == "__main__":
    raise SystemExit(main())
