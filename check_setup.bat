@echo off
REM Double-click any time to test keys, data, option quotes and Discord.
cd /d "%~dp0"
git pull --rebase
".venv\Scripts\python.exe" run_alerts.py --check
pause
