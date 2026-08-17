<#
.SYNOPSIS
    One-shot Windows setup for Option C (IBKR paper executor). Automates the
    scriptable parts of steps 2-7 of IBKR_OPTION_C_WINDOWS.md.

.DESCRIPTION
    Idempotent - safe to re-run. Each phase is individually selectable, or use
    -All for the full run. What it does vs what stays manual:

      Automated here                                   Manual (by design)
      -----------------------------------------        ---------------------------
      * install Python 3.12 + Git (winget)             * IBKR license click-through
      * create .venv, pip install broker deps          * first IB Gateway login / 2FA
      * set OVERALL_BOOK_SECRET (setx)                 * deciding to --execute live
      * download + launch IB Gateway installer
      * download IBC + template its config.ini
      * register the daily Task Scheduler job
      * verify the whole chain

    Anything requiring an IBKR credential prompt or a license agreement cannot
    (and should not) be scripted - the script pauses and tells you exactly what
    to click.

.PARAMETER All
    Run every phase (Python, venv, secret, gateway, IBC, task, verify).
.PARAMETER InstallPython   Install Python 3.12 + Git via winget if missing.
.PARAMETER Venv            Create .venv and install requirements-ibkr.txt.
.PARAMETER Secret          Set OVERALL_BOOK_SECRET (uses -BookSecret or generates one).
.PARAMETER Gateway         Download + launch the IB Gateway installer.
.PARAMETER IBC             Download IBC and template its config.ini (paper).
.PARAMETER Task            Register the daily Task Scheduler job.
.PARAMETER Verify          Check the end-to-end chain and report.

.PARAMETER BookSecret      Value for OVERALL_BOOK_SECRET (must match the publisher).
.PARAMETER IbUser          IBKR paper username, written into the IBC config.
.PARAMETER IbPassword      IBKR paper password, written into the IBC config.
.PARAMETER IbcDir          Where to install IBC (default C:\IBC).
.PARAMETER TaskTime        Local time for the daily task (default 14:30 = the
                           2:30 PM US Central slot on a Central-time machine;
                           on another zone pass the local equivalent, e.g.
                           15:30 on an Eastern box).
.PARAMETER Branch          Branch carrying the published target book (default
                           main - where the publisher commits the daily book).

.EXAMPLE
    # full setup, generate a secret, wire IBC with paper creds:
    powershell -ExecutionPolicy Bypass -File scripts\setup_windows_option_c.ps1 -All -IbUser myPaperUser -IbPassword 's3cret'

.EXAMPLE
    # just (re)create the venv and re-register the scheduled task:
    powershell -ExecutionPolicy Bypass -File scripts\setup_windows_option_c.ps1 -Venv -Task

.NOTES
    Run from an ELEVATED PowerShell (Run as Administrator) for the installer and
    scheduled-task phases. Re-open PowerShell after the secret phase so the new
    environment variable is visible.

    -Secret with no -BookSecret GENERATES a new random secret. That value must
    then be copied into the PUBLISHER's GitHub repo secret AND a fresh book
    published, or every verifier rejects the book. Pass -BookSecret to avoid it.
    See IBKR_OPTION_C_WINDOWS.md section 4.

    This script does NOT start IB Gateway. Task Scheduler fires the executor
    whether or not the gateway is up. See IBKR_OPTION_C_WINDOWS.md section 6.

    KEEP THIS FILE PURE ASCII. Windows PowerShell 5.1 decodes a BOM-less .ps1
    using the machine's ANSI codepage, so a UTF-8 em dash (E2 80 94) arrives as
    'a-circumflex, euro, RIGHT DOUBLE QUOTATION MARK' -- and PowerShell accepts
    that curly quote as a string delimiter. One em dash inside a double-quoted
    message silently opens a string and the whole file fails to parse with
    "The string is missing the terminator" plus bogus "Missing closing '}'".
#>
[CmdletBinding()]
param(
    [switch]$All,
    [switch]$InstallPython,
    [switch]$Venv,
    [switch]$Secret,
    [switch]$Gateway,
    [switch]$IBC,
    [switch]$Task,
    [switch]$Verify,

    [string]$BookSecret,
    [string]$IbUser,
    [string]$IbPassword,
    [string]$IbcDir   = 'C:\IBC',
    [string]$TaskTime = '14:30',
    [string]$Branch   = 'main',
    [string]$Port     = '4002'
)

$ErrorActionPreference = 'Stop'

# Resolve repo root from this script's location (...\repo\scripts\*.ps1).
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Split-Path -Parent $ScriptDir
$Venvpy    = Join-Path $RepoRoot '.venv\Scripts\python.exe'

# If no phase flags were passed, default to -All (nice bare-invocation UX).
$anyPhase = $InstallPython -or $Venv -or $Secret -or $Gateway -or $IBC -or $Task -or $Verify
if (-not $anyPhase) { $All = $true }
if ($All) { $InstallPython=$true; $Venv=$true; $Secret=$true; $Gateway=$true; $IBC=$true; $Task=$true; $Verify=$true }

function Section($t) { Write-Host "`n=== $t ===" -ForegroundColor Cyan }
function Info($t)    { Write-Host "  $t" }
function Ok($t)      { Write-Host "  [ok] $t" -ForegroundColor Green }
function Warn($t)    { Write-Host "  [!] $t" -ForegroundColor Yellow }
function Have($cmd)  { [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }

# -- 2a. Python 3.12 + Git (winget) -------------------------------------------
if ($InstallPython) {
    Section "Python 3.12 + Git"
    if (-not (Have winget)) {
        Warn "winget not found. Install 'App Installer' from the Microsoft Store, then re-run, "
        Warn "or install Python 3.12 and Git manually (see IBKR_OPTION_C_WINDOWS.md section 2)."
    } else {
        if (Have py) { Ok "Python launcher present ($(py -3.12 --version 2>$null))" }
        else {
            Info "installing Python 3.12 via winget..."
            winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
            Ok "Python 3.12 installed (re-open PowerShell if 'py' isn't found yet)"
        }
        if (Have git) { Ok "Git present ($(git --version))" }
        else {
            Info "installing Git via winget..."
            winget install -e --id Git.Git --accept-source-agreements --accept-package-agreements
            Ok "Git installed"
        }
    }
}

# -- 2b. venv + broker deps ----------------------------------------------------
if ($Venv) {
    Section "Python venv + executor dependencies"
    if (-not (Test-Path $Venvpy)) {
        Info "creating .venv (Python 3.12)..."
        if (Have py) { py -3.12 -m venv (Join-Path $RepoRoot '.venv') }
        else { python -m venv (Join-Path $RepoRoot '.venv') }
    }
    Info "installing requirements-ibkr.txt..."
    & $Venvpy -m pip install --upgrade pip | Out-Null
    & $Venvpy -m pip install -r (Join-Path $RepoRoot 'requirements-ibkr.txt')
    # A failing native command does NOT throw in PowerShell, so check the exit
    # code explicitly -- otherwise a broken install still reported success and
    # the failure only surfaced later as ModuleNotFoundError at run time.
    if ($LASTEXITCODE -ne 0) {
        Warn "pip install FAILED (exit $LASTEXITCODE) -- scroll up for the reason."
        Warn "the executor will not run until this succeeds; re-run with -Venv."
    } else {
        # prove the broker layer AND pandas import
        & $Venvpy -c "import ib_async, pandas; print('ib_async', ib_async.__version__, '| pandas', pandas.__version__)"
        if ($LASTEXITCODE -ne 0) { Warn "dependencies installed but do not import -- re-run with -Venv" }
        else { Ok "executor environment ready" }
    }
}

# -- 4. shared secret ----------------------------------------------------------
if ($Secret) {
    Section "OVERALL_BOOK_SECRET"
    $existing = [Environment]::GetEnvironmentVariable('OVERALL_BOOK_SECRET', 'User')
    if ($BookSecret) {
        $val = $BookSecret
    } elseif ($existing) {
        $val = $existing; Info "keeping the existing user secret"
    } else {
        # generate a 40-char URL-safe random secret
        $chars = (48..57) + (65..90) + (97..122)
        $val = -join ($chars | Get-Random -Count 40 | ForEach-Object {[char]$_})
        Warn "generated a new secret - set the SAME value on the PUBLISHER"
        Warn "(GitHub repo secret OVERALL_BOOK_SECRET):"
        Write-Host "      $val" -ForegroundColor Magenta
    }
    setx OVERALL_BOOK_SECRET "$val" | Out-Null
    Ok "OVERALL_BOOK_SECRET set for your user (re-open PowerShell to see it)"
}

# -- 3. IB Gateway installer ---------------------------------------------------
if ($Gateway) {
    Section "IB Gateway (paper)"
    $exe = Join-Path $env:TEMP 'ibgateway-stable-standalone-windows-x64.exe'
    $url = 'https://download2.interactivebrokers.com/installers/ibgateway/stable-standalone/ibgateway-stable-standalone-windows-x64.exe'
    if (Test-Path 'C:\Jts\ibgateway') {
        Ok "IB Gateway appears to be installed already (C:\Jts\ibgateway)"
    } else {
        Info "downloading IB Gateway installer..."
        Invoke-WebRequest -Uri $url -OutFile $exe
        Info "launching the installer - accept the licence and finish the wizard."
        Start-Process -FilePath $exe -Wait
        Ok "installer closed"
    }
    Warn "MANUAL: launch IB Gateway, choose *Paper Trading*, log in, then in"
    Warn "  Configure -> Settings -> API -> Settings set: Enable Socket Clients,"
    Warn "  Socket port $Port, add 127.0.0.1 to Trusted IPs, uncheck Read-Only API."
    Warn "  (IBC below can also enforce the API port automatically.)"
}

# -- 6. IBC auto-login ---------------------------------------------------------
if ($IBC) {
    Section "IBC (unattended auto-login)"
    New-Item -ItemType Directory -Force -Path $IbcDir | Out-Null
    if (Test-Path (Join-Path $IbcDir 'StartGateway.bat')) {
        Ok "IBC already present at $IbcDir"
    } else {
        Info "finding the latest IBC Windows release..."
        $rel = Invoke-RestMethod -Uri 'https://api.github.com/repos/IbcAlpha/IBC/releases/latest' `
                                  -Headers @{ 'User-Agent' = 'setup-script' }
        $asset = $rel.assets | Where-Object { $_.name -match 'IBCWin.*\.zip$' } | Select-Object -First 1
        if (-not $asset) { throw "could not find an IBCWin*.zip asset in the latest IBC release" }
        $zip = Join-Path $env:TEMP $asset.name
        Info "downloading $($asset.name)..."
        Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $zip
        Expand-Archive -Path $zip -DestinationPath $IbcDir -Force
        Ok "IBC unzipped to $IbcDir"
    }
    # template config.ini from the repo example (paper, port, optional creds)
    $tmpl = Join-Path $RepoRoot 'deploy\ibc\config.ini.example'
    $dest = Join-Path $IbcDir 'config.ini'
    if (Test-Path $dest) {
        Warn "config.ini already exists at $dest - leaving it untouched"
    } else {
        $cfg = Get-Content $tmpl -Raw
        if ($IbUser)     { $cfg = $cfg -replace '(?m)^IbLoginId=.*',  "IbLoginId=$IbUser" }
        if ($IbPassword) { $cfg = $cfg -replace '(?m)^IbPassword=.*', "IbPassword=$IbPassword" }
        $cfg = $cfg -replace '(?m)^TradingMode=.*',        'TradingMode=paper'
        $cfg = $cfg -replace '(?m)^OverrideTwsApiPort=.*', "OverrideTwsApiPort=$Port"
        Set-Content -Path $dest -Value $cfg -Encoding ASCII
        Ok "wrote $dest (TradingMode=paper, API port $Port)"
        if (-not $IbPassword) { Warn "no -IbPassword given - edit $dest to add credentials" }
        else { Warn "your IBKR password is stored in plaintext in $dest - protect this file" }
    }
    Info "start the gateway under IBC with: $IbcDir\StartGateway.bat"
}

# -- 7. Task Scheduler ---------------------------------------------------------
if ($Task) {
    Section "Task Scheduler - daily executor"
    $wrapper = Join-Path $ScriptDir 'ibkr_execute_daily.ps1'
    if (-not (Test-Path $wrapper)) { throw "missing $wrapper" }
    # Pass -Branch through explicitly: the task runs detached from this shell,
    # so pinning the branch in the action is what makes -Branch mean anything.
    $arg = "-NoProfile -ExecutionPolicy Bypass -File `"$wrapper`" -Branch `"$Branch`""
    $act = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $arg -WorkingDirectory $RepoRoot
    $trg = New-ScheduledTaskTrigger -Daily -At ([datetime]$TaskTime)
    $set = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable `
            -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries
    Register-ScheduledTask -TaskName 'IBKR Option C executor' -Action $act -Trigger $trg `
        -Settings $set -RunLevel Highest -Force `
        -Description 'Pull the published target book and rebalance the IBKR paper account.' | Out-Null
    Ok "registered task 'IBKR Option C executor' at $TaskTime daily (-WakeToRun), branch $Branch"
    Info "test it now: Start-ScheduledTask -TaskName 'IBKR Option C executor'"
}

# -- verify --------------------------------------------------------------------
if ($Verify) {
    Section "Verify"
    if (Test-Path $Venvpy) { Ok "venv python present" } else { Warn "venv python MISSING - run -Venv" }
    # native exit codes, not exceptions -- try/catch does not see these
    & $Venvpy -c "import ib_async" 2>$null
    if ($LASTEXITCODE -eq 0) { Ok "ib_async importable" } else { Warn "ib_async not importable - run -Venv" }
    & $Venvpy -c "import pandas" 2>$null
    if ($LASTEXITCODE -eq 0) { Ok "pandas importable" } else { Warn "pandas not importable - run -Venv" }
    if ([Environment]::GetEnvironmentVariable('OVERALL_BOOK_SECRET','User')) { Ok "OVERALL_BOOK_SECRET set" }
    else { Warn "OVERALL_BOOK_SECRET not set - run -Secret" }
    # gateway reachability on the API port
    $tcp = Test-NetConnection -ComputerName 127.0.0.1 -Port ([int]$Port) -WarningAction SilentlyContinue
    if ($tcp.TcpTestSucceeded) { Ok "IB Gateway reachable on 127.0.0.1:$Port" }
    else { Warn "nothing listening on 127.0.0.1:$Port - start IB Gateway (paper) / IBC" }
    if (Get-ScheduledTask -TaskName 'IBKR Option C executor' -ErrorAction SilentlyContinue) {
        Ok "scheduled task registered"
    } else { Warn "scheduled task not found - run -Task" }
    $book = Join-Path $RepoRoot 'data\overall\target_book.json'
    if (Test-Path $book) { Ok "a target book is present ($book)" }
    else { Warn "no target book yet - publish one and 'git pull' (guide section 5)" }
}

Section "Next steps"
Info "1. Finish any MANUAL items flagged above (IB Gateway login, API settings)."
Info "2. Ensure the PUBLISHER has the SAME OVERALL_BOOK_SECRET, then run the"
Info "   'Publish target book' GitHub Action and 'git pull' on this laptop."
Info "3. Dry-run once:"
Info "     .\.venv\Scripts\python.exe scripts\ibkr_execute_book.py --file data\overall\target_book.json"
Info "4. When it looks right, add --execute (during US market hours)."
Write-Host ""
