# QQQ Setup Scanner

Watches QQQ on 1-minute bars, scores ICT + indicator setups out of 100, and
(from Phase 4) posts alerts to Discord. Alerts only; trades are placed by hand.

## Layout

| Path | What it does |
| --- | --- |
| `config.yaml` | Every weight, threshold and window. Tune here, not in code. |
| `blackout.yaml` | Dates to sit out (big earnings, anything you choose). |
| `scanner/indicators.py` | VWAP, EMA, RSI, ATR, 5-min bars, opening range, relative volume |
| `scanner/levels.py` | Prior day high/low/close, premarket high/low, opening range |
| `scanner/structure.py` | Swings, equal highs/lows, sweeps, structure shifts, FVG/IFVG, order blocks |
| `scanner/news.py` | Forex Factory red-folder USD events + blackout dates |
| `scanner/scorer.py` | Gates, 100-point score, entry / stop / target |
| `scanner/replay.py` | Runs a day bar by bar through the scorer |
| `tools/replay_day.py` | Command-line replay of a past day from Alpaca data |
| `scanner/history.py` | Monthly CSV store for downloaded bars |
| `scanner/backtest.py` | Trade simulator, live-rule day runner, statistics |
| `tools/download_history.py` | Downloads history from Alpaca |
| `tools/backtest.py` | Runs the backtest and prints the report |
| `tools/compare.py` | Runs config variants side by side on one period |
| `scanner/research.py` | Candidate cache, variant grid, option cost model, ranking |
| `tools/research.py` | Full research run: 192 variants, tune/check, writes `reports/` |
| `scanner/options.py` + `tools/option_check.py` | Replays trades with real option prices |
| `scanner/orb.py` + `tools/orb_study.py` | ORB deep study: losers, stop widths, symbols, graded top 25%, option costs |
| `scanner/quant.py` + `tools/quant_grade.py` | 8 quantitative signs known by 9:35; grades trades, keeps the top 25%, max 1/day |
| `scanner/live.py` | Live scanner: same rules as the backtest, plus news; tracks trades |
| `scanner/discord.py` | Discord webhook and message formats |
| `scanner/feeds.py` | Alpaca history, live bars, market calendar, option quotes |
| `scanner/store.py` | SQLite log of alerts and outcomes (`data/scanner.db`) |
| `run_scanner.py` | Runs the live scanner for the day |
| `scripts/install_task.ps1` | Schedules it for 8:25 AM on weekdays |
| `check_setup.py` | Phase 1 connection check |

## Commands (PowerShell, inside the project with `.venv` active)

```powershell
pip install -r requirements.txt     # once, after updating
python -m pytest -q                 # run the tests
python tools/replay_day.py 2026-10-26             # alerts the scanner would have sent
python tools/replay_day.py 2026-10-26 --show-all  # plus near misses (60-74)

python tools/download_history.py                  # ~2 years of QQQ 1-min bars (once)
python tools/backtest.py --split 2025-10-27       # tune period vs check period
python tools/compare.py                           # config variants side by side (tune period)
python tools/compare.py --period check --only D   # confirm the chosen one, once
```

### Research workflow

```powershell
python tools/download_history.py --symbols QQQ,SPY --start 2019-01-01
python tools/research.py                 # ~10-15 min first time, ~1 min after (cached)
git add reports; git commit -m "research results"; git push
python tools/option_check.py --variant "<winner from the report>"
```

`research.py` ranks every variant on the tune period (everything before the
last 12 months) by the 90% lower bound of average R after estimated option
costs, then tests only the top 3 on the untouched last 12 months.

The backtest saves every setup to `data/backtest/trades_*.csv` (opens in
Excel). Results are in R on the QQQ share price; option P&L will differ.

## How a setup is scored

Gates (all must pass): 9:35-11:30 ET, no news block, under the daily alert
and loss limits, a recent liquidity sweep (of a low for longs, a high for
shorts) followed by a structure shift, or an opening range break; an FVG or
inverse FVG to enter from; and reward to risk of at least 1.5.

| Component | Points |
| --- | --- |
| Liquidity sweep of a key level | 15 |
| Market structure shift with displacement | 15 |
| Entry from IFVG (10) or plain FVG (7) | 10 |
| Order block overlapping the entry zone | 5 |
| Trade side of VWAP | 15 |
| Beyond the opening range | 10 |
| 5-min EMA 9 vs EMA 20 agrees | 10 |
| Trigger volume 1.5x its 20-bar average | 5 |
| 5-min RSI not stretched | 5 |
| Reward to risk 2.0 or better | 5 |
| 9:45-11:00 ET (2 points until 11:30) | 5 |

75+ = alert, 60-74 = logged for the backtest, under 60 = ignored.
Entry is the zone midpoint, the stop sits just beyond the sweep wick, and the
target is the nearest key level at least 1.5R away (otherwise a fixed 2R).

## Live alerts (Phase 4)

```powershell
python run_scanner.py --dry-run     # prints messages instead of posting (test it any time)
python run_scanner.py               # posts to Discord; start before 9:14 ET ideally
powershell -ExecutionPolicy Bypass -File scripts\install_task.ps1   # run it automatically
```

Each day: a start message (session, news, settings), alerts with entry, stop,
target, suggested contract and size, follow-ups when a trade fills, takes a
partial, or finishes, a news warning if a red-folder event is close while a
trade is open, a heartbeat if things are quiet, and a summary once the last
trade is done. Messages say [PAPER] until `live.paper` is set to false.
