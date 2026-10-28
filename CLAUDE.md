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

## JACK'S STRATEGY SPEC (Oct 27, firm - build every test around this)
- CONTRACT: plain 0DTE calls/puts at the money ONLY. No debit/credit spreads, no ITM variants (Jack, Oct 27).
- The coin-flip is a CONTROL only (never traded) - explain it as a lie detector for the backtest.
- Enter 9:30-11:00 ONLY. Hold no longer than 60-90 minutes ("hot potato" - limit premium decay).
- Stop at most -50% of premium (test -25/-35/-50). Looking for LARGE moves: what causes them decides entries.
- ~1 trade a day, ~$1,000 per trade, QQQ/SPY 0DTE. Results in dollars, real option prices.
- `tools/morning_study.py` implements this: library setups <= 11:00 + coin-flip CONTROL, big-move signs
  (`scanner/bigmove.py`: rvol, gap, gap_with, nr7, inside, prev_big, wide_or, moving, trend) chosen on tune by
  lift in P(0.35%+ move within 60 min), grade filter, atm/itm2/ds2/ds3, TP +50/+100/+150/ride, SL, hold 60/90.
- Exits (Jack, Oct 27): base target +50% with a tight stop (-25/-35%); +100/+150% ONLY when big-move signs line
  up, with a TIGHTER stop; or scale out (half at +50%, breakeven stop on the rest). `scanner/debit.py` plans()/run_plan;
  morning_study tests fixed, scale-out and ADAPTIVE (grade-based) plans.
- Entries include ICT-style setups (library.ict_signals): silver_bullet (sweep + FVG 10-11), judas_pd / judas_or
  (sweep of PDH/PDL or OR15 then reclaim + MSS), orb_retest, first_pull (opening drive -> pullback holding VWAP -> HOD).
- Bias guards: debit spreads must cost 20-90% of width; spread TP only counts when both legs traded within 2 min;
  max 50 contracts; reports show median and total without best 5 days. If the coin control profits, distrust it.
- MORNING STUDY v2 RESULT (reports/morning_2026-10-27.md): coin control was PROFITABLE (+$156/wk tune, +$183 check)
  -> test was flattering itself. ds2 (debit spreads) profitable in 97% of settings, atm ~ -$4/-72/wk: spread leg
  prices look biased (off-market / mismatched leg prints). Fixes: take-profits fill AT the target (not minute
  close); every setup now scored vs the coin traded the same way (edge per trade); coin shown per structure.
- v3 RESULT: coin per structure: atm -$65/-95/wk (correct), itm2 +5/-56, ds2 +$234/+339 -> ds2 pricing biased.
  ROOT CAUSE found: spread value was floored at the spread's NET intrinsic; an ITM vertical with time left is worth
  LESS than intrinsic, so debit spreads were overstated (and credit-spread buy-backs overstated = credit study too
  pessimistic). FIX: floor each LEG at its own intrinsic (debit.py + spreads.py). Rerun morning_study + credit_study.
  Single-leg (trustworthy) results v3: no setup robust; top tune configs (vwap10 atm/itm2) lost ~-$250/wk on check.
- v4 RESULT (after leg-floor fix): coin loses with EVERY structure (atm -65/-95, itm2 +5/-56, ds2 -99/-168/wk) ->
  test is honest. Real setups: atm avg -$42/-$115/wk, ds2 -$93/-$164; settings profitable AND beating coin in both
  periods: 248 of 13,200 (~2%, chance level). Combined model check: -$64/wk, 33% win. Credit spreads after fix:
  still lose (check: 0-4% of settings profitable). VERDICT: no edge in 9:30-11:00 0DTE from these price/volume setups.
- Unverified: old buyer study (12:00 time stop) showed ds2 +$1.8k/wk tune and check - likely stale-print bias; rerun.

## REBUILD (Oct 27, Jack's direction) - history
- Jack's format: ~$1,000 per trade, QQQ/SPY 0DTE ATM calls/puts, target about +100% (double), stop -25..-50%
  of premium, ~1 trade a DAY (not a week). USD account at Wealthsimple: no FX, $0 per contract.
- Target: entries that move ~+0.5% our way before ~-0.25% against, 50-60% of the time, ~5x/week.
- Don't reinvent: `scanner/library.py` = 12 published/common setups (orb5, orb15, orb30, vwap10, vwap_cross,
  vwap_pull, noise, gap_go, gap_fade, pdh, momo_1030, rsi_dip), entries all day until 14:30, no look-ahead.
- `tools/model_study.py`: every setup x sides (both/long/regime_k1) x 0DTE/1DTE x TP x SL on tune; combine
  the profitable ones (first signal of the day wins, max 1/day); check year once. Writes model_trades.csv.
- `tools/real_model.py`: re-prices the combined model's trades with REAL option bars.
- RESULT (reports/model_2026-10-27.md, real_model_2026-10-27.md): every setup slightly + on MODEL (hold to
  15:59, both sides, 0DTE ATM best); TP +50/+100 and stops worse. Combined = orb5 nearly every day (fires
  first at 9:35). REAL prices Feb 2024-Oct 2026: 668 trades, 35% win, -$18/trade, -$12.3k total (-$86/wk),
  DD $33.7k; check year +$15k but the 16 months before -$27k. Real is ~$77/trade worse than model (constant-IV
  model misses intraday 0DTE vol decay). VERDICT: no directional 0DTE-buying edge from price-only setups.
- PIVOT (Jack approved Oct 27): SELL 0DTE premium (credit spreads / iron condors) on QQQ/SPY, our signals as
  filters. 100% real option prices, no model. `tools/download_chains.py` caches daily 0DTE chains (+-3% strikes)
  to data/options/; `scanner/spreads.py` replays verticals/condors (last trade ffill, intrinsic floor, exit 15:50,
  TP keep 50/75% of credit, stop 2x/3x credit); `tools/credit_study.py` grid: 6 rules (put/call/condor/trend/
  trend_ic/regime) x 5 entry times x 5 strike distances x $1/2/3 widths x TP x SL, sized to ~$1,000 max loss.
- CREDIT RESULT (reports/credit_2026-10-27.md, v2 after bad-print fix): every rule loses averaged over its
  settings; on check only ~1% of 4,050 settings profitable; every entry-time x distance cell negative on check.
  Selling 0DTE spreads is not the answer either.
- NOW: `tools/buyer_study.py` + `scanner/debit.py`: cheaper ways to BUY the 12 setups on REAL chains: atm / itm2 /
  otm1 / debit spreads ds2/ds3/ds5, cheap-option filter (ATM straddle vs prior-10-day remaining move),
  TP +50/+100/hold, SL -50/none, time stop 12:00/13:30/none. Compare tables average over all other settings.
- Note: independent replications show published VWAP/ORB headline returns mostly vanish with 1-2c slippage.

## Next work (in order)
1. DONE: ORB study + option check (see above).
2. Money study (reports/money_2026-10-27.md, MODEL prices): only 0DTE ATM makes money; held to the setup's
   exit (no TP) beats every TP (+100% best of the TPs). Check year, 5 contracts: ORB [long] +$521/wk,
   ORB [long_up_k1] +$368/wk, noise [regime_k1] +$123/wk. Longer-dated contracts all lose.
   IN PROGRESS: confirm on REAL 0DTE prices with `tools/real_options.py --setup orb --rule long --dte 0`
   (and noise regime_k1, and --tp 1.0). Model may be too kind to 0DTE.
   REAL results (Feb 2024 on, 5 contracts): real 0DTE contracts cost ~1.44x the model (model IV too low;
   use --iv-mult ~1.6). ORB long +100% TP: +$4,962 total (+$35/wk), but 2024 -2.8k, 2025 -1.2k, 2026 +8.9k.
   Noise regime_k1 hold: -$12,895 (its puts on down days made +$2.4k on 34 trades). ORB hold REAL: +$9,369 (206 trades, 26% win, +$66/wk, DD $12.4k) but 2024 -4.2k, 2025 -4.9k,
   2026 +18.4k; top 5 trades = all the profit. Recalibrated money study (iv-mult 1.6): best ORB long 0DTE
   no TP, -30% stop: +$90/wk tune, +$129/wk check; fixed TPs +25..+100% ~breakeven or worse; VWAP and
   noise lose. VERDICT: no robust edge yet; ORB long 0DTE is the only candidate and depends on trend days. Earlier: `python tools/strategy_study.py`, read reports/strategies_<date>.md, then price the
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
