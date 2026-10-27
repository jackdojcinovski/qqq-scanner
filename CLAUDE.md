# QQQ/SPY options scanner: project brief for Claude

Jack's personal project (he is a 3rd year CS student, knows Java best). Python 3.14 on Windows, repo at
`C:\dev\qqq-scanner`, GitHub `jackdojcinovski/qqq-scanner`.

## What it is
A local scanner that watches **QQQ and SPY only** during the session and posts trade alerts to a Discord
webhook. Each alert: entry, grade/score, what was seen, context, stop, take-profit, and the suggested
option contract. A follow-up message reports stop / target / close. Jack places trades **manually on
Wealthsimple, buying OPTIONS (calls/puts), usually 1 contract, never more than 5.** No trades on
Forex Factory red-folder (high-impact USD) news days.

## Goals (in Jack's words, summarised)
- **Profit, not win rate.** "As long as I can make money we have a successful model." It has to make
  enough money, often enough.
- Frequency: about 3 trades/week, at most 1/day, only the top-graded setups.
- Likes asymmetric trades: winners ~3R, losers ~1R. Win rate nearer 50% would be nice, not required.
- **Instruments: QQQ and SPY options only.** Do not expand to single stocks or other ETFs; find more
  trades by adding more quant strategies on QQQ/SPY instead.
- ICT concepts (FVG, order blocks, sweeps) are legacy; focus is quantitative/indicator strategies.

## Where things stand (Oct 2026)
- 5-minute opening-range breakout (`scanner/orb.py`, variant `mid|with|long|classic|fixedeod|checklist`)
  held to close has an edge in **shares** (~+0.3R/trade tune, +0.12R check, ~1.6 trades/week).
- After option costs: 1-2 DTE and weekly ATM lose; deep ITM monthly is about breakeven. Option choice
  (deep ITM, longer-dated, high delta) matters as much as the signal.
- Quant grade (`scanner/quant.py`): only "tight opening range" clearly helps; the grade did not hold up
  on the check year (small sample).
- ORB study (reports/orb_study_2026-10-27.md): QQQ/SPY longs ~+0.06R after deep ITM costs; shorts lose;
  wider stops raise win rate but shrink winners (same profit); 28% of losers are noise stop-outs, 25%
  were +1R first. Real deep ITM option prices: 26 QQQ trades, +$307 total. Verdict: ORB alone is not enough.
- Strategy study (reports/strategies_2026-10-27.md): noise/VWAP/last-half all ~flat or negative after
  deep ITM costs (best check -0.08R). Last-half-hour is dead; drop it.
- Jack's rules (Oct 27): judge in DOLLARS with 5 contracts per trade; Wealthsimple CAD = 1.5% FX each way
  (~3%), $0 per contract; take-profit on PREMIUM (+50% / +100% is fine, 1R for him = premium doubled);
  LONGS by default, shorts (puts) only when the prior 2-3 days show a down trend.
- `scanner/optionsim.py` + `tools/money_study.py`: Black-Scholes option repricing per minute, premium
  TP/SL grid, 6 contract types (0DTE..monthly 3% ITM), regime rules, $ per week, tune/check.
- New: `scanner/intraday.py` (noise-boundary, last-half-hour, VWAP trend; 1/day, decide on a closed
  bar, fill next open) + `tools/strategy_study.py` + `option_check.py --strategy NAME`.

## Next work (in order)
1. DONE: ORB study + option check (see above).
2. Money study (reports/money_2026-10-27.md, MODEL prices): only 0DTE ATM makes money; held to the setup's
   exit (no TP) beats every TP (+100% best of the TPs). Check year, 5 contracts: ORB [long] +$521/wk,
   ORB [long_up_k1] +$368/wk, noise [regime_k1] +$123/wk. Longer-dated contracts all lose.
   IN PROGRESS: confirm on REAL 0DTE prices with `tools/real_options.py --setup orb --rule long --dte 0`
   (and noise regime_k1, and --tp 1.0). Model may be too kind to 0DTE.
   REAL results (Feb 2024 on, 5 contracts): real 0DTE contracts cost ~1.44x the model (model IV too low;
   use --iv-mult ~1.6). ORB long +100% TP: +$4,962 total (+$35/wk), but 2024 -2.8k, 2025 -1.2k, 2026 +8.9k.
   Noise regime_k1 hold: -$12,895 (its puts on down days made +$2.4k on 34 trades). ORB hold: not yet run. Earlier: `python tools/strategy_study.py`, read reports/strategies_<date>.md, then price the
   shortlist with `option_check.py --strategy ... --itm-pct 3 --dte 20`. Strategies built so far:
   - noise-boundary intraday momentum (Zarattini, Aziz & Barbon 2024: entries on :00/:30, trailing
     stop at the boundary/VWAP, exit at close);
   - last-half-hour momentum (Gao et al.: first 30 min return predicts last 30 min);
   - end-of-day hedging-flow momentum (Baltussen et al.);
   - VWAP trend.
3. Judge every strategy on **net R after option costs** (`research.CostModel`) and real option prices,
   tune on all but the last 12 months, check once on the last 12. Rank by the bootstrap 90% lower bound.
4. Combine the strategies that survive into one daily pick (max 1/day), then switch `live.py` to it,
   with the grade and suggested contract in the Discord alert.

## How to work
- Set up: `.venv\Scripts\Activate.ps1`, then `pip install -r requirements.txt`.
- Tests: `python -m pytest -q` (keep them passing; add tests for new engines, check no look-ahead).
- Data: `python tools/download_history.py` (Alpaca free IEX feed, history from 2020-07-27, the latest
  15 minutes are not available over REST). Cached under `data/`.
- Reports go to `reports/`; commit them. Always `git pull --rebase` before `git push`.
- Dry run of the live scanner: `python run_scanner.py --dry-run`.

## Hard rules
- Never read, print, type or commit secrets: `.env` holds the Alpaca keys and Discord webhook. Jack
  enters them himself. `.env` is git-ignored; keep it that way.
- Discord: only the "QQQ Alerts" server. Never touch any other server.
- Never place, modify or cancel trades. The scanner only alerts; Jack trades by hand.
- Live config stays `paper: true` for the Alpaca data account.
- Ask Jack before deleting data, changing the scheduled task, or anything irreversible.
