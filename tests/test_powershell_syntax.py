"""Guard against the PowerShell mistake that silently stopped the live executor.

PR #267 wrote ``"... (exit $GuardExit: $GuardOut)"`` in the live scripts.  In a
double-quoted PowerShell string ``$name:`` is read as a drive-qualified variable
(like ``$env:``), so the file failed to PARSE: Task Scheduler "ran" it, it exited
1 before logging anything, and no trade happened from 2026-11-03.  CI is Linux and
cannot run PowerShell, so this scans for the pattern statically.  Write
``${name}:`` instead.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# $name followed by ':' that is not '::' and not a name char (so $env:X / $script:X
# scope prefixes, which the parser accepts, are matched separately below).
BAD = re.compile(r"\$([A-Za-z_]\w*):(?![:\w])")
SCOPES = {"env", "script", "global", "local", "private", "using", "variable", "function", "alias"}


def _ps1_files():
    out = subprocess.run(["git", "ls-files", "*.ps1"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout.split()
    return [ROOT / f for f in out]


def _bad_in_double_quoted(text: str) -> list[tuple[int, str]]:
    hits = []
    for n, line in enumerate(text.splitlines(), 1):
        # only inspect the double-quoted segments of the line (single quotes are literal)
        for seg in re.findall(r'"((?:[^"`]|`.)*)"', line):
            for m in BAD.finditer(seg):
                if m.group(1).lower() not in SCOPES:
                    hits.append((n, m.group(0)))
    return hits


def test_there_are_powershell_scripts_to_check():
    assert _ps1_files()


@pytest.mark.parametrize("path", _ps1_files(), ids=lambda p: p.name)
def test_no_drive_qualified_variable_in_double_quoted_strings(path):
    hits = _bad_in_double_quoted(path.read_text(encoding="utf-8", errors="ignore"))
    assert hits == [], (f"{path.name}: '$name:' inside a double-quoted string is a PowerShell "
                        f"parse error - use ${{name}}: instead: {hits}")


def test_the_scanner_catches_the_original_bug():
    assert _bad_in_double_quoted('Log "WARN: (exit $GuardExit: $GuardOut) x"') == [(1, "$GuardExit:")]
    assert _bad_in_double_quoted('Log "exit ${GuardExit}: ${GuardOut}"') == []
    assert _bad_in_double_quoted('Log "path $env:TEMP and $script:X"') == []
    assert _bad_in_double_quoted("Log 'literal $x: single'") == []
