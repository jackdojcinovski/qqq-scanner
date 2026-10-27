# Registers a Windows scheduled task that starts the scanner at 8:25 on weekdays.
# Run once from the project folder:  powershell -ExecutionPolicy Bypass -File scripts\install_task.ps1
# Remove it later with:              Unregister-ScheduledTask -TaskName "QQQ Scanner" -Confirm:$false

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Error "No .venv found at $py. Create it first (see README)."; exit 1 }

$action   = New-ScheduledTaskAction -Execute $py -Argument "`"$root\run_scanner.py`"" -WorkingDirectory $root
$trigger  = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At 8:25am
$settings = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable -AllowStartIfOnBatteries `
            -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 9)
Register-ScheduledTask -TaskName "QQQ Scanner" -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
Write-Host "Scheduled 'QQQ Scanner' for 8:25 AM on weekdays. It wakes the laptop if it is asleep (not shut down)."
