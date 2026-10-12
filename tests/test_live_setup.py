"""The live-setup wizard's decisions, and the guard that keeps account data off GitHub."""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import live_setup as ls  # noqa: E402
import ibkr_execute_accounts as acc  # noqa: E402


# ═══ gateway discovery and the Java problem ═════════════════════════════════
def _install(jts: Path, ver="1051", jre=True, cfg_target=None):
    d = jts / "ibgateway" / ver
    (d / ".install4j").mkdir(parents=True)
    (d / "jars").mkdir()
    if jre:
        (d / "jre" / "bin").mkdir(parents=True)
        (d / "jre" / "bin" / "java.exe").write_text("")
    if cfg_target is not None:
        (d / ".install4j" / "inst_jre.cfg").write_text(cfg_target + "\n")
    return d


def test_picks_the_newest_gateway(tmp_path):
    _install(tmp_path, "1045")
    _install(tmp_path, "1051")
    gw = ls.find_gateway(tmp_path)
    assert gw["version"] == "1051" and gw["jre_bin"] == gw["dir"] / "jre" / "bin"


def test_no_gateway(tmp_path):
    assert ls.find_gateway(tmp_path) is None
    (tmp_path / "ibgateway" / "notes").mkdir(parents=True)
    assert ls.find_gateway(tmp_path) is None


def test_a_stale_jre_cfg_is_found_and_fixed(tmp_path):
    d = _install(tmp_path, cfg_target=r"C:\old\jre\gone")
    gw = ls.find_gateway(tmp_path)
    assert "no bin" in ls.jre_cfg_problem(gw)
    msg = ls.fix_jre_cfg(gw)
    assert "inst_jre.cfg now points" in msg
    assert (d / ".install4j" / "inst_jre.cfg.bak").exists()
    assert ls.jre_cfg_problem(ls.find_gateway(tmp_path)) is None


def test_a_good_or_absent_jre_cfg_is_left_alone(tmp_path):
    d = _install(tmp_path, cfg_target="X")
    (d / ".install4j" / "inst_jre.cfg").write_text(str(d / "jre") + "\n")
    assert ls.jre_cfg_problem(ls.find_gateway(tmp_path)) is None
    d2 = _install(tmp_path / "b")
    assert ls.jre_cfg_problem(ls.find_gateway(tmp_path / "b")) is None and d2


# ═══ StartGateway-live.bat ══════════════════════════════════════════════════
BAT = ("@echo off\r\nset TWS_MAJOR_VRSN=981\r\nset IBC_INI=%USERPROFILE%\\Documents\\IBC\\config.ini\r\n"
       "set TRADING_MODE=\r\nset IBC_PATH=C:\\IBC\r\nset TWS_PATH=C:\\Jts\r\nset JAVA_PATH=\r\n"
       "rem comment set TWS_MAJOR_VRSN=1\r\ncall start.bat\r\n")


def test_bat_gets_the_real_version_config_and_java():
    out, missing = ls.patch_start_bat(BAT, {
        "TWS_MAJOR_VRSN": "1051", "IBC_INI": r"C:\IBC\config-live.ini", "TRADING_MODE": "live",
        "JAVA_PATH": r"C:\Jts\ibgateway\1051\jre\bin", "TWOFA_TIMEOUT_ACTION": "restart"})
    assert "set TWS_MAJOR_VRSN=1051\r\n" in out
    assert r"set IBC_INI=C:\IBC\config-live.ini" in out and "set TRADING_MODE=live" in out
    assert r"set JAVA_PATH=C:\Jts\ibgateway\1051\jre\bin" in out
    assert "rem comment set TWS_MAJOR_VRSN=1" in out            # comments untouched
    assert missing == ["TWOFA_TIMEOUT_ACTION"]
    assert "981" not in out


# ═══ IBC config ═════════════════════════════════════════════════════════════
def test_ibc_config_is_live_with_the_login():
    tmpl = (ROOT / "deploy" / "ibc" / "config.live.ini.example").read_text()
    cfg = ls.render_ibc_config(tmpl, "me", "p@ss=word")
    assert re.search(r"(?m)^IbLoginId=me$", cfg) and "IbPassword=p@ss=word" in cfg
    assert re.search(r"(?m)^TradingMode=live$", cfg)
    assert re.search(r"(?m)^OverrideTwsApiPort=4001$", cfg)
    assert re.search(r"(?m)^ReadOnlyApi=no$", cfg)


# ═══ accounts file ══════════════════════════════════════════════════════════
def test_generated_accounts_file_validates_and_is_conservative():
    ex = json.loads((ROOT / "deploy" / "ibkr_accounts.example.json").read_text())
    cfg = ls.build_accounts_file(ex, [
        dict(id="U5555551", label="Taxable", type="margin", enabled=True),
        dict(id="U5555552", label="Roth", type="ira", enabled=False)])
    parsed, errs = acc.validate_config(cfg)
    assert errs == [] and [a.id for a in parsed.enabled()] == ["U5555551"]
    assert parsed.port == 4001
    assert cfg["defaults"]["max_deploy_frac"] <= 0.25
    assert not any(a.allow_margin for a in parsed.accounts)


def test_merge_finds_new_and_dropped_accounts():
    existing = {"accounts": [{"id": "U1"}, {"id": "U2"}]}
    assert ls.merge_accounts(existing, ["U2", "U3"]) == (["U2"], ["U3"])
    assert ls.merge_accounts(None, ["U9"]) == ([], ["U9"])


def test_mask_never_reveals_a_value():
    assert ls.mask("abcdef") == "set (6 chars)" and ls.mask("") == "NOT set"


def test_check_reports_a_missing_install(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OVERALL_BOOK_SECRET", raising=False)
    rc = ls.main(["--check", "--jts-dir", str(tmp_path), "--ibc-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 1 and "FAIL" in out and "IB Gateway not found" in out


# ═══ scripts stay plain ASCII (Windows PowerShell 5.1 reads them as ANSI) ═══
@pytest.mark.parametrize("name", ["setup_live.ps1", "register_gateway_tasks.ps1",
                                  "ibkr_gateway_watchdog.ps1"])
def test_powershell_scripts_are_ascii(name):
    assert all(b < 128 for b in (ROOT / "scripts" / name).read_bytes())


def test_gateway_tasks_run_with_normal_rights():
    src = (ROOT / "scripts" / "register_gateway_tasks.ps1").read_text()
    assert "RunLevel Limited" in src and "Highest" not in src


# ═══ privacy: nothing about the accounts may reach GitHub ═══════════════════
PLACEHOLDERS = {"U1111111", "U2222222", "U3333333", "U1234567", "U9999999", "U0000001",
                "U5555551", "U5555552", "DU1234567", "U7777777"}
ID_RE = re.compile(r"\b(?:[UFI]\d{6,9}|DU[A-Z]?\d{6,8})\b")
# paper-trading report committed before live trading existed (a paper account)
LEGACY = ("data/overall/executed_book.json", "data/overall/executed_archive/")


def _tracked():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    return [f for f in out.stdout.splitlines() if (ROOT / f).is_file()]


def test_no_real_account_ids_in_tracked_files():
    hits = []
    for f in _tracked():
        if f.startswith(LEGACY) or f.endswith((".png", ".parquet", ".pkl", ".joblib", ".gz", ".zip")):
            continue
        try:
            text = (ROOT / f).read_text(errors="ignore")
        except OSError:
            continue
        for m in ID_RE.findall(text):
            if m not in PLACEHOLDERS:
                hits.append(f"{f}: {m}")
    assert hits == [], "real-looking account ids are tracked:\n" + "\n".join(hits[:10])


def test_private_paths_are_git_ignored():
    for path in ("deploy/ibkr_accounts.json", "data/overall/accounts/U1111111/executed_book_live.json",
                 "data/overall/accounts/U1111111/.send_locks/2026-10-09.lock",
                 "C:/IBC/config-live.ini", "config-live.ini", "logs/ibkr_live.log"):
        r = subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT)
        assert r.returncode == 0, f"{path} is not git-ignored"


def test_nothing_under_the_account_report_dir_is_tracked():
    assert not [f for f in _tracked() if f.startswith("data/overall/accounts/")]


def test_publishing_reports_is_opt_in(monkeypatch):
    monkeypatch.delenv("IBKR_PUBLISH_REPORTS", raising=False)
    monkeypatch.delenv("IBKR_NO_PUSH_REPORT", raising=False)
    assert acc.publishing_enabled() is False
    monkeypatch.setenv("IBKR_PUBLISH_REPORTS", "1")
    assert acc.publishing_enabled() is True
    assert acc.publishing_enabled(no_push=True) is False
    monkeypatch.setenv("IBKR_NO_PUSH_REPORT", "1")
    assert acc.publishing_enabled() is False
