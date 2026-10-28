@echo off
REM Double-click after the scanner finishes: saves today's journal to GitHub so Claude can see the results.
cd /d "%~dp0"
git add journal
git commit -m "journal %date%"
git pull --rebase
git push
echo.
echo Journal saved. You can close this window.
pause
