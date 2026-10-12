<#
.SYNOPSIS
    ONE command to set up LIVE multi-account MOC trading on this Windows PC.

.DESCRIPTION
    Run from the repo folder in PowerShell:

        powershell -ExecutionPolicy Bypass -File scripts\setup_live.ps1

    It makes sure Git and Python exist (winget), creates the virtualenv, installs
    the broker dependencies, then hands over to scripts\live_setup.py which walks
    through: IB Gateway check, IBC + live config, StartGateway-live.bat, secrets,
    account discovery, scheduled tasks, pre-flight.  Safe to re-run.

    -Check   read-only health report, changes nothing
    -Only    run a single step: gateway, ibc, startbat, secret, accounts, tasks, verify
    Full guide: docs\LIVE_SETUP_WINDOWS.md
#>
param(
    [switch]$Check,
    [string]$Only = ''
)
$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

function Have($cmd) { return [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }

Write-Host "== Prerequisites =="
foreach ($pkg in @(@('git', 'Git.Git'), @('python', 'Python.Python.3.12'))) {
    if (-not (Have $pkg[0])) {
        if (-not (Have 'winget')) { throw "$($pkg[0]) is missing and winget is not available - install it by hand, then re-run." }
        Write-Host "  installing $($pkg[1]) ..."
        winget install --id $pkg[1] -e --accept-package-agreements --accept-source-agreements
        $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
        if (-not (Have $pkg[0])) { throw "installed $($pkg[1]) but it is not on PATH yet - close this window, open a new PowerShell and re-run." }
    }
}
Write-Host "  git + python present"

$venvpy = Join-Path $RepoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $venvpy)) {
    Write-Host "  creating .venv ..."
    python -m venv .venv
}
& $venvpy -m pip install --quiet --upgrade pip
& $venvpy -m pip install --quiet -r requirements-ibkr.txt
if ($LASTEXITCODE -ne 0) { throw "pip install failed - read the error above" }
Write-Host "  virtualenv ready"

$argv = @()
if ($Check) { $argv += '--check' }
if ($Only)  { $argv += @('--only', $Only) }
& $venvpy (Join-Path $ScriptDir 'live_setup.py') @argv
exit $LASTEXITCODE
