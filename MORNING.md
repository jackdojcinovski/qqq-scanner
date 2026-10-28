# Daily routine (QQQ alerts)

## Easiest: double-click
In `C:\dev\qqq-scanner` (tip: right-click each file -> Send to -> Desktop (create shortcut)):

| When | Double-click | What it does |
|---|---|---|
| Before 9:30 ET | `start_alerts.bat` | updates the code, then scans 9:30-11:00 and sends Discord alerts. **Leave the window open** until it says "Done for today" (by ~12:30 at the latest). |
| After it's done | `end_of_day.bat` | saves today's journal (`journal/trades.csv`) to GitHub |
| Any time | `check_setup.bat` | tests keys, data, option quotes and Discord |

## Or paste into PowerShell
Morning (before 9:30 ET):
```powershell
cd C:\dev\qqq-scanner
.venv\Scripts\Activate.ps1
git pull --rebase
python run_alerts.py
```
After it finishes:
```powershell
cd C:\dev\qqq-scanner
git add journal; git commit -m "journal"; git pull --rebase; git push
```

## Have ready
- Laptop **plugged in, lid open**, internet on, the scanner window left open (it keeps the laptop awake)
- **Discord** on your phone (QQQ Alerts channel, notifications on)
- **Wealthsimple** (USD account) open and logged in, so you can buy the contract fast
- Use YOUR fill price: stop = fill x 0.65 (-35%), target = fill x 1.5 (+50%) unless the alert says strong plan (+100% / -25%)

## What you'll get
1. "Scanner running" when you start it
2. ENTER alert (contract, buy price + time, size, stop, target, why) - or "no setup by 11:00" - or "no trade, red-folder news"
3. Exit briefing when the stop / target / time limit hits (with $ and %)
4. Daily recap (today, this week, all time)
