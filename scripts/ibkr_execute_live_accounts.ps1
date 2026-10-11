<#
.SYNOPSIS
    Daily LIVE executor for ALL configured IBKR accounts (Option C, Windows PC).

.DESCRIPTION
    Pulls the latest published live target book, then runs
    scripts\ibkr_execute_accounts.py --execute, which trades the book in every
    account listed in deploy\ibkr_accounts.json as MARKET-ON-CLOSE orders only
    (no limit/market fallback; a run that starts after the 15:50 ET exchange
    cutoff skips the session and places nothing).

    All accounts run TOGETHER, one executor process each, against the single
    IB Gateway live login - so every MOC order is in before the cutoff and one
    closing auction fills them all. The runner commits and pushes every
    account's execution report in ONE push at the end.

    Schedule it ~15 minutes before the cutoff-safe slot, e.g. 2:15 PM US
    Central (3:15 PM ET): MOC orders fill at the close whenever they are
    entered, so starting earlier costs nothing and buys margin against a slow
    gateway or git pull. See IBKR_OPTION_C_WINDOWS.md section 13.

.PARAMETER Branch
    Git branch carrying the published book (env: IBKR_BRANCH, default main).
.PARAMETER Config
    Accounts config (env: IBKR_ACCOUNTS_CONFIG, default deploy\ibkr_accounts.json).
.PARAMETER Preview
    Run the runner WITHOUT --execute: connects, reads every account, prints the
    order plan, places no orders.
.PARAMETER NoPull
    Skip the git pull (use the book already on disk).

    Env: IBKR_PYTHON (interpreter), IBKR_NO_PUSH_REPORT=1 (do not push reports),
    IBKR_TRADING_DISABLED=1 (kill switch), OVERALL_BOOK_SECRET (required),
    IBKR_ALERT_WEBHOOK (optional Slack/Discord alert URL).

    Exit code: 0 ok | 1 an account failed or did not fully fill | 2 gateway or
    config problem, nothing traded | 3 session skipped (past the MOC cutoff,
    holiday, or kill switch).

    KEEP THIS FILE PURE ASCII -- see the note in setup_windows_option_c.ps1.
#>
param(
    [string]$Branch = $env:IBKR_BRANCH,
    [string]$Config = $env:IBKR_ACCOUNTS_CONFIG,
    [switch]$Preview,
    [switch]$NoPull
)

$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Split-Path -Parent $ScriptDir
if (-not $Branch) { $Branch = 'main' }

if ($env:IBKR_PYTHON) { $Python = $env:IBKR_PYTHON }
else { $Python = Join-Path $RepoRoot '.venv\Scripts\python.exe' }

# Task Scheduler runs this detached, so stdout is a pipe: force UTF-8 on both
# sides or the first arrow in the book printout kills the run.
$env:PYTHONUTF8       = '1'
$env:PYTHONIOENCODING = 'utf-8'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

$LogDir = Join-Path $RepoRoot 'logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Log = Join-Path $LogDir 'ibkr_live_executor.log'

function Log($msg) {
    # Convert explicitly: Get-Date -Format 'u' labels LOCAL time as UTC.
    $line = "[{0}] {1}" -f (Get-Date).ToUniversalTime().ToString('u'), $msg
    Write-Output $line
    Add-Content -Path $Log -Value $line -Encoding UTF8
}

# The run holds the gateway connection through the 4:00 PM ET closing auction
# (~30 minutes after the orders go in). A PC that sleeps in that window drops the
# connection and the report is never written, so ask Windows to stay awake for
# as long as this script runs. Best-effort: never let it stop the run.
$KeepAwake = $false
try {
    Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);' -ErrorAction Stop
    # ES_CONTINUOUS (0x80000000) | ES_SYSTEM_REQUIRED (0x00000001)
    [void][Win32.Power]::SetThreadExecutionState([uint32]2147483649)
    $KeepAwake = $true
} catch { }

Set-Location $RepoRoot
Log "----------------------------------------------"
if ($KeepAwake) { Log "sleep inhibited for the duration of this run" }
else { Log "WARN: could not inhibit sleep - make sure the PC cannot sleep before 4:10 PM ET" }
$Mode = if ($Preview) { 'PREVIEW (no orders)' } else { 'EXECUTE' }
Log "starting IBKR LIVE multi-account MOC executor ($Mode, branch=$Branch)"

if (-not $NoPull) {
    Log "pulling latest live book from origin/$Branch"
    # Native git writes progress to stderr; do not let that terminate the run.
    $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    git fetch --quiet origin $Branch 2>&1 | ForEach-Object { Log $_ }
    git merge --ff-only "origin/$Branch" 2>&1 | ForEach-Object { Log $_ }
    if ($LASTEXITCODE -ne 0) { Log "WARN: ff-merge failed - using the book already on disk" }
    $ErrorActionPreference = $prev
}

$RunArgs = @('scripts\ibkr_execute_accounts.py')
if ($Config)   { $RunArgs += @('--config', $Config) }
if (-not $Preview) { $RunArgs += '--execute' }

# Judge the outcome by $LASTEXITCODE, not by stderr records (see
# ibkr_execute_daily.ps1 for why 'Stop' + '2>&1' would abort the log early).
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& $Python @RunArgs 2>&1 | ForEach-Object { Log $_ }
$RunExit = $LASTEXITCODE
$ErrorActionPreference = $prevEAP

switch ($RunExit) {
    0 { Log "all accounts OK" }
    3 { Log "session skipped (past the MOC cutoff, not a trading day, or kill switch) - nothing was traded" }
    2 { Log "ERROR: gateway or config problem - NOTHING was traded. Check IB Gateway login/2FA and deploy\ibkr_accounts.json" }
    default { Log "WARN: runner exited $RunExit - at least one account failed or did not fully fill; read the account summary above" }
}
Log "done (runner exit $RunExit)"
if ($KeepAwake) { try { [void][Win32.Power]::SetThreadExecutionState([uint32]2147483648) } catch { } }
exit $RunExit
