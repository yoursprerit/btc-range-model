<#
.SYNOPSIS
    Register the tasks that keep the LIVE IB Gateway running, and (with
    -RetirePaper) switch off everything left over from paper trading.

      'IB Gateway LIVE (logon)'   start StartGateway-live.bat when you sign in
      'IB Gateway LIVE watchdog'  every 5 min: relaunch it if it is not running

    Both run as YOU, interactively (the gateway needs a desktop session), with
    normal rights.  Turn on Windows auto-login (netplwiz) so a reboot brings the
    gateway back without you.
#>
param(
    [string]$IbcDir = 'C:\IBC',
    [int]$EveryMin = 5,
    [switch]$RetirePaper
)
$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Split-Path -Parent $ScriptDir
$bat = Join-Path $IbcDir 'StartGateway-live.bat'
if (-not (Test-Path $bat)) { throw "$bat not found - run: scripts\setup_live.ps1 -Only startbat" }

$me = "$env:USERDOMAIN\$env:USERNAME"
$prin = New-ScheduledTaskPrincipal -UserId $me -LogonType Interactive -RunLevel Limited

# 1. start the gateway at logon
$act = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument "/c `"$bat`"" -WorkingDirectory $IbcDir
$trg = New-ScheduledTaskTrigger -AtLogOn -User $me
$trg.Delay = 'PT1M'
$set = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 5) -ExecutionTimeLimit (New-TimeSpan -Days 3650)
Register-ScheduledTask -TaskName 'IB Gateway LIVE (logon)' -Action $act -Trigger $trg -Settings $set `
    -Principal $prin -Force -Description 'Start the live IB Gateway under IBC at logon.' | Out-Null
Write-Host "  [OK] task 'IB Gateway LIVE (logon)' registered"

# 2. watchdog
$wd  = Join-Path $ScriptDir 'ibkr_gateway_watchdog.ps1'
$act = New-ScheduledTaskAction -Execute 'powershell.exe' `
        -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$wd`" -IbcDir `"$IbcDir`"" `
        -WorkingDirectory $RepoRoot
$trg = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
        -RepetitionInterval (New-TimeSpan -Minutes $EveryMin) -RepetitionDuration (New-TimeSpan -Days 3650)
$set = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 5)
Register-ScheduledTask -TaskName 'IB Gateway LIVE watchdog' -Action $act -Trigger $trg -Settings $set `
    -Principal $prin -Force -Description 'Relaunch the live IB Gateway if it is not running.' | Out-Null
Write-Host "  [OK] task 'IB Gateway LIVE watchdog' registered (every $EveryMin min)"

# 3. retire paper trading so nothing trades or starts the paper gateway
if ($RetirePaper) {
    Write-Host "  retiring paper trading ..."
    foreach ($name in @('IBKR Option C executor')) {
        $t = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
        if ($t) { Disable-ScheduledTask -TaskName $name | Out-Null; Write-Host "  [OK] disabled task '$name'" }
        else    { Write-Host "  [OK] no task '$name'" }
    }
    $lnk = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs\Startup\StartGateway.lnk'
    if (Test-Path $lnk) { Remove-Item $lnk -Force; Write-Host "  [OK] removed paper gateway Startup shortcut" }
    $paper = Test-NetConnection -ComputerName 127.0.0.1 -Port 4002 -WarningAction SilentlyContinue
    if ($paper.TcpTestSucceeded) { Write-Host "  [WARN] something still listens on paper port 4002 - close the paper IB Gateway window" }
    else { Write-Host "  [OK] nothing listens on paper port 4002" }
    [Environment]::SetEnvironmentVariable('IBKR_ACCOUNT_MODE', $null, 'User')
}
