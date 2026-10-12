"""Guided one-shot setup of LIVE multi-account MOC trading on a Windows PC.

Run it through ``scripts\\setup_live.ps1`` (which first makes sure Python, Git and
the virtualenv exist), or directly::

    .venv\\Scripts\\python scripts\\live_setup.py            # every step, in order
    .venv\\Scripts\\python scripts\\live_setup.py --only accounts
    .venv\\Scripts\\python scripts\\live_setup.py --check     # read-only health report

Steps (each is idempotent -- safe to re-run, it never overwrites without asking):

    gateway   find the installed IB Gateway, its version and its bundled Java
    ibc       download IBC, write C:\\IBC\\config-live.ini (live, port 4001)
    startbat  build C:\\IBC\\StartGateway-live.bat with the REAL gateway version,
              config path and Java -- the three things that went wrong by hand
    secret    store OVERALL_BOOK_SECRET (and optional monitors) for your user
    accounts  log in once, discover the accounts, write deploy\\ibkr_accounts.json
    tasks     register the 2:15 PM run, the 4:15 PM reconcile and the gateway
              auto-start + watchdog; retire the paper tasks
    verify    run the live pre-flight

Everything that needs a password reads it hidden; nothing secret is printed.
The decision logic is in plain functions so it is unit-tested
(``tests/test_live_setup.py``) -- only the prompts and subprocess calls are not.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

IBC_DIR = Path(r"C:\IBC")
JTS_DIR = Path(r"C:\Jts")
LIVE_PORT = 4001
ACCOUNT_TYPES = ("margin", "ira", "cash")
STEPS = ("gateway", "ibc", "startbat", "secret", "accounts", "tasks", "verify")


# ═══ pure helpers (unit-tested) ═════════════════════════════════════════════
def find_gateway(jts_dir: Path) -> dict | None:
    """The newest installed IB Gateway under *jts_dir*, or None.

    The standalone installer puts it in ``<Jts>\\ibgateway\\<major>`` (e.g. 1051).
    IBC must be told that exact number (``TWS_MAJOR_VRSN``) -- a mismatch is the
    classic "cannot find Gateway" startup failure."""
    cands: list[tuple[int, Path]] = []
    for parent in (jts_dir / "ibgateway", jts_dir):
        if not parent.is_dir():
            continue
        for d in parent.iterdir():
            if d.is_dir() and d.name.isdigit() and (
                    (d / "ibgateway.exe").exists() or (d / "jars").is_dir()
                    or (d / "jre").is_dir() or (d / ".install4j").is_dir()):
                cands.append((int(d.name), d))
    if not cands:
        return None
    ver, d = max(cands, key=lambda c: c[0])
    jre_bin = d / "jre" / "bin"
    return dict(version=str(ver), dir=d, jre=d / "jre",
                jre_bin=jre_bin if (jre_bin / "java.exe").exists() else None,
                jts=jts_dir)


def jre_cfg_problem(gw: dict) -> str | None:
    """Why ``.install4j\\inst_jre.cfg`` would send the launcher to a missing Java
    ("Can't find suitable Java installation"), or None when it is fine/absent."""
    cfg = gw["dir"] / ".install4j" / "inst_jre.cfg"
    if not cfg.exists():
        return None
    try:
        target = cfg.read_text(errors="replace").strip().splitlines()[0].strip()
    except (OSError, IndexError):
        return None
    if not target:
        return "inst_jre.cfg is empty"
    if not (Path(target) / "bin" / "java.exe").exists():
        return f"inst_jre.cfg points at {target}, which has no bin\\java.exe"
    return None


def fix_jre_cfg(gw: dict) -> str:
    """Point inst_jre.cfg at the bundled JRE (after backing the old one up)."""
    cfg = gw["dir"] / ".install4j" / "inst_jre.cfg"
    if gw["jre_bin"] is None:
        return "no bundled jre\\bin\\java.exe found -- reinstall IB Gateway"
    shutil.copy2(cfg, cfg.with_suffix(".cfg.bak"))
    cfg.write_text(str(gw["jre"]) + "\n")
    return f"inst_jre.cfg now points at {gw['jre']} (old one saved as .bak)"


BAT_VARS = ("TWS_MAJOR_VRSN", "IBC_INI", "TRADING_MODE", "IBC_PATH", "TWS_PATH",
            "JAVA_PATH", "TWOFA_TIMEOUT_ACTION")


def patch_start_bat(text: str, values: dict[str, str]) -> tuple[str, list[str]]:
    """Set ``set NAME=value`` lines in IBC's StartGateway.bat.  Returns the new
    text (CRLF kept) and the names that had no ``set`` line to patch."""
    missing = []
    for name, val in values.items():
        pat = re.compile(rf"(?im)^([ \t]*set[ \t]+\"?{re.escape(name)}=)[^\r\n\"]*(\"?)[ \t]*(?=\r?\n|\Z)")
        if pat.search(text):
            text = pat.sub(lambda m: f"{m.group(1)}{val}{m.group(2)}", text, count=1)
        else:
            missing.append(name)
    return text, missing


def render_ibc_config(template: str, user: str, password: str, port: int = LIVE_PORT) -> str:
    """The live IBC config: the repo template with the login and port filled in."""
    out = template
    for key, val in (("IbLoginId", user), ("IbPassword", password),
                     ("TradingMode", "live"), ("OverrideTwsApiPort", str(port)),
                     ("ReadOnlyApi", "no")):
        pat = re.compile(rf"(?m)^{key}=.*$")
        out = pat.sub(lambda m: f"{key}={val}", out, count=1) if pat.search(out) \
            else out + f"\n{key}={val}\n"
    return out


def build_accounts_file(example: dict, chosen: list[dict], port: int = LIVE_PORT) -> dict:
    """The accounts config: the shipped example (limits, defaults) with the
    discovered accounts.  *chosen*: ``[{id, label, type, enabled}]``."""
    cfg = {k: v for k, v in example.items() if k != "accounts"}
    cfg["port"] = port
    cfg["accounts"] = [dict(id=c["id"], label=c.get("label", ""),
                            type=c.get("type", "cash"), enabled=bool(c.get("enabled")))
                       for c in chosen]
    return cfg


def merge_accounts(existing: dict | None, managed: list[str]) -> tuple[list[str], list[str]]:
    """(accounts already configured and still managed, newly discovered ones)."""
    have = [a["id"] for a in (existing or {}).get("accounts", [])]
    return [a for a in have if a in managed], [a for a in managed if a not in have]


def mask(v: str) -> str:
    return f"set ({len(v)} chars)" if v else "NOT set"


# ═══ console helpers ════════════════════════════════════════════════════════
def say(tag: str, msg: str) -> None:
    print(f"  [{tag}] {msg}")


def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        v = input(f"  ? {prompt}{suffix}: ").strip()
    except EOFError:
        v = ""
    return v or default


def yes(prompt: str, default: bool = False) -> bool:
    v = ask(prompt + (" (Y/n)" if default else " (y/N)")).lower()
    return default if not v else v.startswith("y")


def is_admin() -> bool:
    try:
        import ctypes
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def powershell(*args: str) -> int:
    return subprocess.call(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", *args])


# ═══ steps ══════════════════════════════════════════════════════════════════
def step_gateway(a) -> bool:
    print("\n== IB Gateway ==")
    gw = find_gateway(Path(a.jts_dir))
    if gw is None:
        say("FAIL", f"IB Gateway not found under {a.jts_dir}.")
        say("DO", "Install it: https://www.interactivebrokers.com/en/trading/ibgateway-stable.php "
                  "(Stable, Windows, default folder C:\\Jts), then re-run this step.")
        return False
    say("OK", f"IB Gateway {gw['version']} at {gw['dir']}")
    if gw["jre_bin"] is None:
        say("WARN", "no bundled Java (jre\\bin\\java.exe) -- reinstall IB Gateway")
    problem = jre_cfg_problem(gw)
    if problem:
        say("WARN", problem + "  (this is what causes \"Can't find suitable Java installation\")")
        if a.yes or yes("Fix it now?", True):
            say("OK", fix_jre_cfg(gw))
    elif gw["jre_bin"] is not None:
        say("OK", f"bundled Java found: {gw['jre_bin']}")
    a.gw = gw
    return True


def _download_ibc(ibc_dir: Path) -> None:
    req = urllib.request.Request("https://api.github.com/repos/IbcAlpha/IBC/releases/latest",
                                 headers={"User-Agent": "live-setup"})
    rel = json.load(urllib.request.urlopen(req, timeout=30))
    asset = next(x for x in rel["assets"] if re.search(r"IBCWin.*\.zip$", x["name"]))
    zp = Path(os.environ.get("TEMP", ".")) / asset["name"]
    say("..", f"downloading {asset['name']}")
    urllib.request.urlretrieve(asset["browser_download_url"], zp)
    ibc_dir.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(zp), str(ibc_dir))


def step_ibc(a) -> bool:
    print("\n== IBC (unattended login) ==")
    ibc = Path(a.ibc_dir)
    if not (ibc / "StartGateway.bat").exists():
        try:
            _download_ibc(ibc)
        except Exception as e:
            say("FAIL", f"could not download IBC ({e}). Download IBCWin-*.zip from "
                        f"https://github.com/IbcAlpha/IBC/releases and unzip it into {ibc}")
            return False
    say("OK", f"IBC present in {ibc}")
    dest = ibc / "config-live.ini"
    if dest.exists() and not a.force and not yes(f"{dest} exists. Rewrite it?", False):
        say("OK", "keeping the existing config-live.ini")
        return True
    user = a.ib_user or ask("IBKR LIVE username (the one login that manages all accounts)")
    pw = a.ib_password or getpass.getpass("  ? IBKR password (hidden): ")
    if not user or not pw:
        say("FAIL", "username and password are both required")
        return False
    tmpl = (ROOT / "deploy" / "ibc" / "config.live.ini.example").read_text()
    dest.write_text(render_ibc_config(tmpl, user, pw), encoding="ascii", errors="replace")
    if os.name == "nt":                       # only you (and SYSTEM) may read the clear-text password
        subprocess.call(["icacls", str(dest), "/inheritance:r", "/grant:r",
                         f"{os.environ.get('USERNAME', '')}:(R,W)", "SYSTEM:(R)"],
                        stdout=subprocess.DEVNULL)
    say("OK", f"wrote {dest} (live, API port {LIVE_PORT}); permissions limited to your user")
    say("NOTE", "IBC keeps the password in clear text in that file -- never commit or share it")
    return True


def step_startbat(a) -> bool:
    print("\n== StartGateway-live.bat ==")
    gw = getattr(a, "gw", None) or find_gateway(Path(a.jts_dir))
    if gw is None:
        say("FAIL", "IB Gateway not found -- run the gateway step first")
        return False
    ibc = Path(a.ibc_dir)
    src = ibc / "StartGateway.bat"
    if not src.exists():
        say("FAIL", f"{src} missing -- run the ibc step first")
        return False
    text = src.read_text(errors="replace")
    vals = {"TWS_MAJOR_VRSN": gw["version"], "IBC_INI": str(ibc / "config-live.ini"),
            "TRADING_MODE": "live", "IBC_PATH": str(ibc), "TWS_PATH": str(gw["jts"]),
            "TWOFA_TIMEOUT_ACTION": "restart"}
    if gw["jre_bin"] is not None:
        vals["JAVA_PATH"] = str(gw["jre_bin"])
    new, missing = patch_start_bat(text, vals)
    out = ibc / "StartGateway-live.bat"
    out.write_text(new.replace("\r\n", "\n").replace("\n", "\r\n"), encoding="ascii", errors="replace")
    say("OK", f"wrote {out}")
    say("OK", f"  gateway version {gw['version']}, config {vals['IBC_INI']}, trading mode live")
    if missing:
        say("WARN", "no 'set' line to patch for: " + ", ".join(missing) + " -- check the file by hand")
    if not (ibc / "config-live.ini").exists():
        say("WARN", "config-live.ini does not exist yet (ibc step)")
    return True


def step_secret(a) -> bool:
    print("\n== Secrets and monitors ==")
    cur = os.environ.get("OVERALL_BOOK_SECRET", "")
    say("..", f"OVERALL_BOOK_SECRET is {mask(cur)} in this session")
    ok = True
    if not cur or yes("Replace it?", False):
        v = getpass.getpass("  ? Paste the SAME secret as the GitHub repo secret "
                            "OVERALL_BOOK_SECRET (hidden, Enter to skip): ").strip()
        if v:
            subprocess.call(["setx", "OVERALL_BOOK_SECRET", v], stdout=subprocess.DEVNULL)
            say("OK", "stored for your user (visible after you sign out and in again)")
        elif not cur:
            say("WARN", "skipped -- live refuses an unsigned book, so nothing will trade until it is set")
            ok = False
    for var, what in (("IBKR_ALERT_WEBHOOK", "Slack/Discord alert webhook URL"),
                      ("IBKR_HEARTBEAT_URL", "healthchecks.io URL for the 2:15 PM run"),
                      ("IBKR_RECONCILE_HEARTBEAT_URL", "healthchecks.io URL for the 4:15 PM reconcile")):
        if os.environ.get(var):
            say("OK", f"{var} {mask(os.environ[var])}")
            continue
        v = ask(f"{what} (optional, Enter to skip)")
        if v:
            subprocess.call(["setx", var, v], stdout=subprocess.DEVNULL)
            say("OK", f"{var} stored")
    return ok


def _discover(port: int) -> list[str] | None:
    import ibkr_execute_accounts as acc
    cfg, _ = acc.validate_config({"port": port, "accounts": [{"id": "U0000001"}]})
    accts, why = acc._connect_once(cfg)
    say("OK" if accts else "FAIL", why)
    return accts


def step_accounts(a) -> bool:
    print("\n== Accounts ==")
    path = ROOT / "deploy" / "ibkr_accounts.json"
    existing = None
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except ValueError as e:
            say("FAIL", f"{path} is not valid JSON ({e}). Delete it and re-run this step.")
            return False
    say("DO", f"Start the gateway now:  {a.ibc_dir}\\StartGateway-live.bat   "
              "and approve the IBKR Mobile push on your phone.")
    ask("Press Enter once the gateway window shows you are logged in")
    managed = _discover(LIVE_PORT)
    if not managed:
        return False
    keep, new = merge_accounts(existing, managed)
    if existing and not new:
        say("OK", f"{path.name} already lists every managed account -- unchanged")
        return True
    example = json.loads((ROOT / "deploy" / "ibkr_accounts.example.json").read_text())
    chosen = [a_ for a_ in (existing or {}).get("accounts", []) if a_["id"] in managed]
    print("\n  Decide per NEW account. 'Trade it?' = N keeps it listed but disabled.\n"
          "  type: margin (taxable, may hold margin) / ira (Roth or Traditional) / cash.\n")
    for acct in new:
        trade = yes(f"Trade account {acct}?", False)
        typ = ask(f"  type of {acct} ({'/'.join(ACCOUNT_TYPES)})", "cash")
        while typ not in ACCOUNT_TYPES:
            typ = ask(f"  type must be one of {ACCOUNT_TYPES}", "cash")
        chosen.append(dict(id=acct, label=ask(f"  label for {acct}", acct),
                           type=typ, enabled=trade))
    cfg = build_accounts_file(existing or example, chosen)
    import ibkr_execute_accounts as acc
    _, errs = acc.validate_config(cfg)
    if errs:
        say("FAIL", "generated config is invalid: " + "; ".join(errs))
        return False
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    say("OK", f"wrote {path}  ({sum(1 for c in chosen if c['enabled'])} enabled)")
    say("NOTE", "defaults are conservative: max_deploy_frac 0.25 (25% of each account in total), "
                "max_order_notional 5000. Raise them only after several clean days.")
    return True


def step_tasks(a) -> bool:
    print("\n== Scheduled tasks ==")
    if os.name != "nt":
        say("WARN", "not Windows -- skipping")
        return True
    if not is_admin():
        say("WARN", "not running as Administrator; if a task fails to register, "
                    "re-run from an elevated PowerShell")
    s = ROOT / "scripts"
    rc = powershell("-File", str(s / "setup_windows_option_c.ps1"), "-LiveTask", "-ReconcileTask")
    rc |= powershell("-File", str(s / "register_gateway_tasks.ps1"), "-IbcDir", str(a.ibc_dir),
                     "-RetirePaper")
    subprocess.call(["powercfg", "/change", "standby-timeout-ac", "0"])
    say("OK" if rc == 0 else "FAIL", "tasks registered; PC set to never sleep on AC power"
        if rc == 0 else "a task failed to register -- read the lines above")
    return rc == 0


def step_verify(a) -> bool:
    print("\n== Pre-flight ==")
    if os.name != "nt":
        say("WARN", "not Windows -- run scripts/preflight_option_c.py --live by hand")
        return True
    rc = powershell("-File", str(ROOT / "scripts" / "setup_windows_option_c.ps1"), "-LivePreflight")
    if rc == 0:
        say("OK", "pre-flight passed. Next: -LiveRehearse (full dress rehearsal, no orders).")
    return rc == 0


def check(a) -> int:
    """Read-only health report of everything this wizard sets up."""
    print("== Live setup health ==")
    gw = find_gateway(Path(a.jts_dir))
    bad = 0

    def line(ok, msg, hard=True):
        nonlocal bad
        say("OK" if ok else ("FAIL" if hard else "WARN"), msg)
        bad += (not ok) and hard
    line(gw is not None, f"IB Gateway installed ({gw['version']})" if gw else "IB Gateway not found")
    if gw:
        line(jre_cfg_problem(gw) is None, jre_cfg_problem(gw) or "Java config fine")
        bat = Path(a.ibc_dir) / "StartGateway-live.bat"
        txt = bat.read_text(errors="replace") if bat.exists() else ""
        line(bat.exists(), f"{bat} exists" if bat.exists() else f"{bat} missing")
        if txt:
            line(re.search(rf"(?im)^\s*set\s+TWS_MAJOR_VRSN={gw['version']}\s*$", txt) is not None,
                 f"StartGateway-live.bat uses gateway version {gw['version']}")
    line((Path(a.ibc_dir) / "config-live.ini").exists(), "config-live.ini present")
    line(bool(os.environ.get("OVERALL_BOOK_SECRET")), "OVERALL_BOOK_SECRET set")
    ap_ = ROOT / "deploy" / "ibkr_accounts.json"
    if not ap_.exists():
        line(False, "deploy\\ibkr_accounts.json missing (accounts step)")
    else:
        import ibkr_execute_accounts as acc
        try:
            cfg, errs = acc.validate_config(json.loads(ap_.read_text()))
        except ValueError as e:
            cfg, errs = None, [f"not valid JSON: {e}"]
        line(not errs, (f"ibkr_accounts.json valid, {len(cfg.enabled())} account(s) enabled" if not errs
                        else "ibkr_accounts.json: " + "; ".join(errs)))
    line(bool(os.environ.get("IBKR_HEARTBEAT_URL")), "heartbeat monitor configured", hard=False)
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", action="append", choices=STEPS, help="run just this step (repeatable)")
    ap.add_argument("--check", action="store_true", help="read-only health report")
    ap.add_argument("--yes", action="store_true", help="accept safe fixes without asking")
    ap.add_argument("--force", action="store_true", help="rewrite config-live.ini without asking")
    ap.add_argument("--ibc-dir", default=str(IBC_DIR))
    ap.add_argument("--jts-dir", default=str(JTS_DIR))
    ap.add_argument("--ib-user", default="")
    ap.add_argument("--ib-password", default="")
    a = ap.parse_args(argv)
    if a.check:
        return check(a)
    fns = dict(gateway=step_gateway, ibc=step_ibc, startbat=step_startbat,
               secret=step_secret, accounts=step_accounts, tasks=step_tasks, verify=step_verify)
    for name in (a.only or STEPS):
        if not fns[name](a):
            print(f"\nStopped at step '{name}'. Fix the line marked FAIL, then re-run:\n"
                  f"  python scripts\\live_setup.py --only {name}   (or without --only to continue)")
            return 1
    print("\nAll steps done. Sign out and back in (or reboot) so scheduled tasks see the "
          "new environment variables, log in to the gateway, and read docs/LIVE_SETUP_WINDOWS.md "
          "section 'First live day'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
