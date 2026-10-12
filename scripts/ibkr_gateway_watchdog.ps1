<#
.SYNOPSIS
    Start the LIVE IB Gateway if it is not running.  Run every 5 minutes by the
    'IB Gateway LIVE watchdog' task (registered by register_gateway_tasks.ps1).

    It only launches when ALL of these hold, so it can never stack a second
    gateway on top of a login that is still waiting for your 2FA approval:
      * nothing is listening on the live API port,
      * no Java process of IB Gateway / IBC is already running,
      * it has not launched one in the last 15 minutes.
    It cannot approve the IBKR Mobile push: after a launch you still tap it.
#>
param(
    [string]$IbcDir = 'C:\IBC',
    [int]$Port = 4001,
    [int]$CooldownMin = 15
)
$ErrorActionPreference = 'Continue'
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$LogDir = Join-Path $RepoRoot 'logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Log = Join-Path $LogDir 'gateway_watchdog.log'
$Stamp = Join-Path $IbcDir 'watchdog.stamp'

function Log($m) {
    if ((Test-Path $Log) -and ((Get-Item $Log).Length -gt 1MB)) { Move-Item -Force $Log "$Log.1" }
    Add-Content -Path $Log -Value ("{0} {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $m)
}

$listening = (Test-NetConnection -ComputerName 127.0.0.1 -Port $Port -WarningAction SilentlyContinue).TcpTestSucceeded
if ($listening) { exit 0 }

$gw = Get-CimInstance Win32_Process -Filter "Name='java.exe'" -ErrorAction SilentlyContinue |
      Where-Object { $_.CommandLine -match 'ibgateway|IBC|ibcalpha' }
if ($gw) { Log "port $Port closed but a gateway/IBC java process is running (waiting for login/2FA?) - not launching"; exit 0 }

if (Test-Path $Stamp) {
    $age = (New-TimeSpan -Start (Get-Item $Stamp).LastWriteTime -End (Get-Date)).TotalMinutes
    if ($age -lt $CooldownMin) { Log "last launch $([int]$age) min ago - cooling down"; exit 0 }
}

$bat = Join-Path $IbcDir 'StartGateway-live.bat'
if (-not (Test-Path $bat)) { Log "FAIL: $bat missing"; exit 2 }
Log "gateway is down - launching $bat"
Set-Content -Path $Stamp -Value (Get-Date -Format o)
Start-Process -FilePath $bat -WorkingDirectory $IbcDir
if ($env:IBKR_ALERT_WEBHOOK) {
    try {
        Invoke-RestMethod -Uri $env:IBKR_ALERT_WEBHOOK -Method Post -ContentType 'application/json' `
            -Body (@{ text = "IB Gateway LIVE was down - watchdog relaunched it. Approve the IBKR Mobile push." } | ConvertTo-Json) | Out-Null
    } catch { Log "alert webhook failed: $($_.Exception.Message)" }
}
exit 0
