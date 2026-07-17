<#
.SYNOPSIS
    Daily IBKR paper executor (Option C) for a Windows laptop / Task Scheduler.

.DESCRIPTION
    Pulls the latest published target book from the feature branch, then runs the
    lightweight executor to rebalance the IBKR *paper* account to match it. The
    model never runs here — this host only consumes the pre-computed book.

    The executor itself guards against weekends / holidays / stale books and only
    trades a paper (DU…) account, so a stray or early run is a safe no-op.

.PARAMETER Branch
    Git branch carrying the published target_book.json (env: IBKR_BRANCH).
.PARAMETER Book
    Path to the target book JSON (env: IBKR_BOOK).
.PARAMETER Band
    No-trade band as a fraction of net-liq (env: IBKR_BAND, default 0.01).
.PARAMETER Port
    IB Gateway API port (env: IBKR_PORT, default 4002 = paper).
.PARAMETER NoPull
    Skip the `git pull` (use the book already on disk).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File C:\btc-range-model\scripts\ibkr_execute_daily.ps1

.NOTES
    Requires OVERALL_BOOK_SECRET to be set (same value used to publish) so the
    book's signature can be verified before trading. Set it as a *user* or
    *system* environment variable so Task Scheduler inherits it.
#>
param(
    [string]$Branch = $env:IBKR_BRANCH,
    [string]$Book   = $env:IBKR_BOOK,
    [string]$Band   = $env:IBKR_BAND,
    [string]$Port   = $env:IBKR_PORT,
    [switch]$NoPull
)

$ErrorActionPreference = 'Stop'

# Resolve the repo root from this script's location (…\repo\scripts\*.ps1).
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Split-Path -Parent $ScriptDir

if (-not $Branch) { $Branch = 'claude/trading-signals-ibkr-paper-jwyvrc' }
if (-not $Book)   { $Book   = Join-Path $RepoRoot 'data\overall\target_book.json' }
if (-not $Band)   { $Band   = '0.01' }
if (-not $Port)   { $Port   = '4002' }

# Point at the venv interpreter (override via IBKR_PYTHON).
if ($env:IBKR_PYTHON) { $Python = $env:IBKR_PYTHON }
else { $Python = Join-Path $RepoRoot '.venv\Scripts\python.exe' }

$LogDir = Join-Path $RepoRoot 'logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Log = Join-Path $LogDir 'ibkr_executor.log'

function Log($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'u'), $msg
    Write-Output $line
    Add-Content -Path $Log -Value $line
}

Set-Location $RepoRoot
Log "──────────────────────────────────────────────"
Log "starting IBKR paper executor (branch=$Branch port=$Port band=$Band)"

if (-not $NoPull) {
    Log "pulling latest target book from origin/$Branch"
    git fetch --quiet origin $Branch
    # Fast-forward only: never clobber local work, just take the newest book.
    git merge --ff-only "origin/$Branch" 2>&1 | ForEach-Object { Log $_ }
}

# --execute places orders; the executor's own guards decide if it actually trades.
& $Python "scripts\ibkr_execute_book.py" --file $Book --execute --band $Band --port $Port 2>&1 |
    ForEach-Object { Log $_ }

# Publish the execution report back so the cloud app's "Executed Book" tab shows
# it. Requires git WRITE credentials on this host. On push failure, reset to
# origin so the branch never diverges. Set env IBKR_NO_PUSH_REPORT=1 to skip.
$Report = Join-Path $RepoRoot 'data\overall\executed_book.json'
if ($env:IBKR_NO_PUSH_REPORT -ne '1' -and (Test-Path $Report)) {
    git add $Report
    git diff --cached --quiet -- $Report
    if ($LASTEXITCODE -ne 0) {
        git -c user.name="ibkr-executor" -c user.email="executor@localhost" `
            commit -q -m "chore(ibkr): execution report $(Get-Date -Format 'yyyy-MM-dd')"
        git push origin "HEAD:$Branch"
        if ($LASTEXITCODE -eq 0) { Log "published execution report to origin/$Branch" }
        else {
            Log "WARN: could not push execution report (this host needs git write access) - rolling back"
            git reset --hard "origin/$Branch" | Out-Null
        }
    } else { Log "no change to executed_book.json - nothing to publish" }
}

Log "done (exit $LASTEXITCODE)"
exit $LASTEXITCODE
