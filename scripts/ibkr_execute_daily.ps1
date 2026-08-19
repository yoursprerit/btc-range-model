<#
.SYNOPSIS
    Daily IBKR paper executor (Option C) for a Windows laptop / Task Scheduler.

.DESCRIPTION
    Pulls the latest published target book from the feature branch, then runs the
    lightweight executor to rebalance the IBKR *paper* account to match it. The
    model never runs here - this host only consumes the pre-computed book.

    The executor itself guards against weekends / holidays / stale books and only
    trades a paper (DU...) account, so a stray or early run is a safe no-op.

.PARAMETER Branch
    Git branch carrying the published target_book.json (env: IBKR_BRANCH,
    default main - where the publisher commits the daily book).
.PARAMETER Book
    Path to the target book JSON (env: IBKR_BOOK).
.PARAMETER Band
    No-trade band as a fraction of net-liq (env: IBKR_BAND, default 0.01).
.PARAMETER Port
    IB Gateway API port (env: IBKR_PORT, default 4002 = paper).
.PARAMETER NoPull
    Skip the `git pull` (use the book already on disk).

.NOTES
    Order routing (env only - see IBKR_PAPER_TRADING.md section Order routing):
      IBKR_ORDER_TYPE    marketable-limit (default) | moc | market
                         Default sends a LIMIT priced through the touch, so a
                         bad quote can't run away with the order; anything left
                         unfilled escalates to a market order in the same run.
                         moc fills in the 4:00 PM ET closing auction instead -
                         the price the backtest books against - and must be
                         entered before the 15:50 ET cutoff (the 2:30 PM CT
                         task clears it by 20 min).
      IBKR_SLIPPAGE_CAP  how far through the touch a marketable limit prices
                         (default 0.005 = 0.5%). Widen it on a laptop with
                         delayed-only market data.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File C:\btc-range-model\scripts\ibkr_execute_daily.ps1

.NOTES
    Requires OVERALL_BOOK_SECRET to be set (same value used to publish) so the
    book's signature can be verified before trading. Set it as a *user* or
    *system* environment variable so Task Scheduler inherits it. If it is NOT
    set the executor SKIPS verification rather than failing -- it logs
    "signed but no secret provided to verify" and trades anyway.

    The report push at the end needs git WRITE access on this host, and runs
    detached (no console), so the credential must resolve without a prompt --
    an SSH deploy key or a stored PAT. Set IBKR_NO_PUSH_REPORT=1 to skip it.
    See IBKR_OPTION_C_WINDOWS.md section 8.

    KEEP THIS FILE PURE ASCII -- see the note in setup_windows_option_c.ps1.
#>
param(
    [string]$Branch = $env:IBKR_BRANCH,
    [string]$Book   = $env:IBKR_BOOK,
    [string]$Band   = $env:IBKR_BAND,
    [string]$Port   = $env:IBKR_PORT,
    [switch]$NoPull
)

$ErrorActionPreference = 'Stop'

# Resolve the repo root from this script's location (...\repo\scripts\*.ps1).
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Split-Path -Parent $ScriptDir

$AccountMode = if ($env:IBKR_ACCOUNT_MODE) { $env:IBKR_ACCOUNT_MODE } else { 'paper' }
if (-not $Branch) { $Branch = 'main' }
if (-not $Book)   {
    # live parks idle capital in SATA (target_book_live.json); paper holds cash.
    $BookName = if ($AccountMode -eq 'live') { 'target_book_live.json' } else { 'target_book.json' }
    $Book = Join-Path $RepoRoot "data\overall\$BookName"
}
if (-not $Band)   { $Band   = '0.01' }
if (-not $Port)   { $Port   = '4002' }

# Point at the venv interpreter (override via IBKR_PYTHON).
if ($env:IBKR_PYTHON) { $Python = $env:IBKR_PYTHON }
else { $Python = Join-Path $RepoRoot '.venv\Scripts\python.exe' }

# --- console/pipe encoding -------------------------------------------------
# When Task Scheduler runs this detached, the executor's stdout is a PIPE, not a
# console. Python then encodes stdout with the ANSI codepage (cp1252) instead of
# using the console's UTF-16 API, and the first '->' arrow (U+2192) in the book
# printout raises UnicodeEncodeError and kills the run -- while the same command
# typed at an interactive prompt works fine, because a console stdout takes the
# Unicode path. Force UTF-8 on both sides so redirected output matches the
# console.
$env:PYTHONUTF8      = '1'
$env:PYTHONIOENCODING = 'utf-8'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

$LogDir = Join-Path $RepoRoot 'logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Log = Join-Path $LogDir 'ibkr_executor.log'

function Log($msg) {
    # Get-Date -Format 'u' appends 'Z' but formats LOCAL time -- it does not
    # convert. That mislabels every line as UTC and makes the log impossible to
    # line up against IBKR's genuinely-UTC order stamps. Convert explicitly.
    $line = "[{0}] {1}" -f (Get-Date).ToUniversalTime().ToString('u'), $msg
    Write-Output $line
    Add-Content -Path $Log -Value $line -Encoding UTF8
}

Set-Location $RepoRoot
Log "----------------------------------------------"
Log "starting IBKR paper executor (branch=$Branch port=$Port band=$Band)"

if (-not $NoPull) {
    Log "pulling latest target book from origin/$Branch"
    git fetch --quiet origin $Branch
    # Fast-forward only: never clobber local work, just take the newest book.
    git merge --ff-only "origin/$Branch" 2>&1 | ForEach-Object { Log $_ }
}

# For live (env IBKR_ACCOUNT_MODE=live + a live gateway port) the wrapper adds
# --confirm-live and the safety limits.
$ExecArgs = @('--file', $Book, '--execute', '--band', $Band, '--port', $Port,
              '--account-mode', $AccountMode)
if ($AccountMode -eq 'live') {
    $ExecArgs += '--confirm-live'
    if ($env:IBKR_EXPECTED_ACCOUNT)   { $ExecArgs += @('--expected-account', $env:IBKR_EXPECTED_ACCOUNT) }
    if ($env:IBKR_MAX_DEPLOY_FRAC)    { $ExecArgs += @('--max-deploy-frac', $env:IBKR_MAX_DEPLOY_FRAC) }
    if ($env:IBKR_MAX_ORDER_NOTIONAL) { $ExecArgs += @('--max-order-notional', $env:IBKR_MAX_ORDER_NOTIONAL) }
}
# Sleeve ledger: only the PATH belongs in the env, and only when it is not the
# default beside the execution report -- once the ledger exists the executor
# finds it and keeps compounding it. --sleeve-init-* / --sleeve-adjust are
# one-off operator commands; set here they would re-run every single day.
if ($env:IBKR_SLEEVE_FILE)      { $ExecArgs += @('--sleeve-file', $env:IBKR_SLEEVE_FILE) }
if ($env:IBKR_KILL_SWITCH_FILE) { $ExecArgs += @('--kill-switch-file', $env:IBKR_KILL_SWITCH_FILE) }
# Order routing: marketable-limit (default) | moc | market - see IBKR_PAPER_TRADING.md.
if ($env:IBKR_ORDER_TYPE)   { $ExecArgs += @('--order-type', $env:IBKR_ORDER_TYPE) }
if ($env:IBKR_SLIPPAGE_CAP) { $ExecArgs += @('--slippage-cap', $env:IBKR_SLIPPAGE_CAP) }
# Extra flags passed straight through, the way ibkr_execute_daily.sh does. The
# rehearsal uses it to reach the order stage on a day whose book is stale or
# already executed (IBKR_EXTRA='--allow-stale-bar --force-rerun'), which the
# pre-flight guards would otherwise abort well before the kill switch.
if ($env:IBKR_EXTRA) { $ExecArgs += ($env:IBKR_EXTRA -split '\s+' | Where-Object { $_ }) }
Log "account mode: $AccountMode"

# --execute places orders; the executor's own guards decide if it actually trades.
# '2>&1' turns the executor's stderr into ErrorRecords, and under
# $ErrorActionPreference='Stop' the FIRST such record is a TERMINATING error --
# it would abort this script on the spot, skipping the exit-code log line and the
# report push, leaving a log that just stops mid-run with no explanation. Drop to
# 'Continue' for the call and judge the outcome by $LASTEXITCODE instead.
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& $Python "scripts\ibkr_execute_book.py" @ExecArgs 2>&1 | ForEach-Object { Log $_ }
$ExecExit = $LASTEXITCODE
$ErrorActionPreference = $prevEAP
if ($ExecExit -ne 0) { Log "WARN: executor exited $ExecExit - see the lines above" }

# Publish the execution report back so the cloud app's "Executed Book" tab shows
# it. Requires git WRITE credentials on this host. On push failure, reset to
# origin so the branch never diverges. Set env IBKR_NO_PUSH_REPORT=1 to skip.
$ReportName = if ($AccountMode -eq 'live') { 'executed_book_live.json' } else { 'executed_book.json' }
$Report = Join-Path $RepoRoot "data\overall\$ReportName"
# The dated as-of copy the executor writes beside the report feeds the Executed
# Book page's Historical tab, so it ships in the same commit.
$Archive = Join-Path $RepoRoot "data\overall\executed_archive"
if ($env:IBKR_NO_PUSH_REPORT -ne '1' -and (Test-Path $Report)) {
    git add $Report
    if (Test-Path $Archive) { git add $Archive }
    git diff --cached --quiet -- $Report $Archive
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

Log "done (executor exit $ExecExit)"
exit $ExecExit
