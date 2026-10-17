<#
.SYNOPSIS
    After-the-close RECONCILE for the live multi-account MOC run (Windows PC).

.DESCRIPTION
    Runs scripts\ibkr_reconcile_accounts.py about 15 minutes after the 4:00 PM ET
    closing auction. For every account that had orders sent today it asks the
    broker what really filled and checks the execution report against it: a
    report the 2:20 PM run never got to write is rebuilt from the broker's fills,
    unfilled or partly filled legs and orders still working are reported, and an
    alert is raised for anything that needs a look. It never sends or cancels an
    order.

    Schedule it at 4:15 PM US Eastern = 3:15 PM US Central. See
    docs\LIVE_MULTI_ACCOUNT_MOC.md.

.PARAMETER Config
    Accounts config (env: IBKR_ACCOUNTS_CONFIG, default deploy\ibkr_accounts.json).
.PARAMETER DryRun
    Look and report only: rewrite nothing, push nothing, no heartbeat.

    Env: IBKR_PYTHON, OVERALL_BOOK_SECRET (to re-sign a rebuilt report),
    IBKR_NO_PUSH_REPORT=1, IBKR_ALERT_WEBHOOK, IBKR_RECONCILE_HEARTBEAT_URL.

    Exit code: 0 all reconciled | 1 a problem to look at | 2 gateway or config
    problem | 3 nothing to do (no orders sent today / auction not printed yet).

    KEEP THIS FILE PURE ASCII -- see the note in setup_windows_option_c.ps1.
#>
param(
    [string]$Config = $env:IBKR_ACCOUNTS_CONFIG,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Split-Path -Parent $ScriptDir

if ($env:IBKR_PYTHON) { $Python = $env:IBKR_PYTHON }
else { $Python = Join-Path $RepoRoot '.venv\Scripts\python.exe' }

$env:PYTHONUTF8       = '1'
$env:PYTHONIOENCODING = 'utf-8'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

$LogDir = Join-Path $RepoRoot 'logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Log = Join-Path $LogDir 'ibkr_live_reconcile.log'

# Rotate at 5 MB (the log holds account ids): .log -> .1 -> .2 -> .3
try {
    if ((Test-Path $Log) -and ((Get-Item $Log).Length -gt 5MB)) {
        foreach ($i in 2, 1) {
            $src = "$Log.$i"
            if (Test-Path $src) { Move-Item -Force $src "$Log.$($i + 1)" }
        }
        Move-Item -Force $Log "$Log.1"
    }
} catch { }

function Log($msg) {
    $line = "[{0}] {1}" -f (Get-Date).ToUniversalTime().ToString('u'), $msg
    Write-Output $line
    Add-Content -Path $Log -Value $line -Encoding UTF8
}

Set-Location $RepoRoot
Log "----------------------------------------------"
Log "starting IBKR LIVE reconcile (after the close)"

$RunArgs = @('scripts\ibkr_reconcile_accounts.py')
if ($Config) { $RunArgs += @('--config', $Config) }
if ($DryRun) { $RunArgs += '--dry-run' }

$prevEAP = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& $Python @RunArgs 2>&1 | ForEach-Object { Log $_ }
$RunExit = $LASTEXITCODE
$ErrorActionPreference = $prevEAP

switch ($RunExit) {
    0 { Log "all accounts reconciled" }
    3 { Log "nothing to reconcile (no orders sent today, or the auction has not printed yet)" }
    2 { Log "ERROR: gateway or config problem - today's fills were NOT verified; check TWS" }
    default { Log "WARN: reconcile found a problem - read the account lines above" }
}
Log "done (reconcile exit $RunExit)"
exit $RunExit
