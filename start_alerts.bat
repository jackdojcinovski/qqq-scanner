@echo off
REM Double-click before 9:30 ET. Leave this window open until it says it's done (by ~12:30 at the latest).
cd /d "%~dp0"
echo Updating...
git pull --rebase
echo.
echo Starting QQQ alerts (scans 9:30-11:00, one trade a day). Keep this window open.
".venv\Scripts\python.exe" run_alerts.py
echo.
echo Done for today. You can close this window.
pause
