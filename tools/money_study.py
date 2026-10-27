"""How much MONEY each setup would have made, trading 5 option contracts every time.

    python tools/money_study.py
    python tools/money_study.py --contracts 1

For every setup x contract type x take-profit x stop x regime rule:
  * buys --contracts calls (puts for shorts) at the ask, sells at the bid
  * Wealthsimple CAD account: 1.5% FX on the buy and 1.5% on the sell (~3%)
  * take-profit / stop are on the premium: +100% = the contract doubled
  * at most 1 trade a day across QQQ and SPY (first signal wins)
  * longs by default; shorts (puts) only on "down" days: below the 20-day
    average AND the last 3 days fell more than k x their normal move

Option prices are modelled (Black-Scholes on recent realised volatility,
scanner/optionsim.py), so treat the dollars as estimates and confirm the
winner with tools/option_check.py on real option prices.

Every choice is made on the TUNE period; the last 12 months (CHECK) are
looked at once for the shortlist. Writes reports/money_<date>.md.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scanner import backtest as bt  # noqa: E402
from scanner import intraday as ix  # noqa: E402
from scanner import optionsim as om  # noqa: E402
from scanner import setups  # noqa: E402
from scanner.config import ROOT, load_config  # noqa: E402
from scanner.history import load_history, trading_days  # noqa: E402

TPS = [0.25, 0.5, 0.75, 1.0, 1.5, None]
SLS = [None, 0.3, 0.5]
MODES = {
    "long": "longs only, every day",
    "long_up_k1": "longs only, skip down days",
    "regime_k1": "longs; puts on down days (k=1)",
    "regime_k0.5": "longs; puts on down days (k=0.5, more puts)",
}


def pct(x):
    return "none" if x is None else f"+{x:.0%}" if x >= 0 else f"{x:.0%}"


def summarise(res: list[om.Result], n_days: int, n_contracts: int) -> dict:
    pnl = np.array([r.pnl(n_contracts) for r in res]) if res else np.array([])
    weeks = max(n_days / 5, 1)
    if len(pnl) == 0:
        return dict(n=0, per_week=0, win=0, avg_win=0, avg_loss=0, per_trade=0, per_week_usd=0, total=0,
                    lo=float("nan"), dd=0, streak=0, cost=0, tgt=0, stops=0, avg_ret=0, years="0/0", by_year={})
    eq = np.cumsum(pnl)
    dd = float(np.max(np.maximum.accumulate(np.concatenate([[0], eq])) - np.concatenate([[0], eq])))
    streak = worst = 0
    for x in pnl:
        streak = streak + 1 if x < 0 else 0
        worst = max(worst, streak)
    lo, _ = bt.bootstrap_ci(list(pnl)) if len(pnl) >= 2 else (float("nan"), 0)
    by_year: dict[int, list[float]] = {}
    for r, x in zip(res, pnl):
        by_year.setdefault(r.signal.day.year, []).append(x)
    return dict(
        n=len(pnl), per_week=len(pnl) / weeks, win=float(np.mean(pnl > 0)),
        avg_win=float(pnl[pnl > 0].mean()) if (pnl > 0).any() else 0.0,
        avg_loss=float(pnl[pnl <= 0].mean()) if (pnl <= 0).any() else 0.0,
        per_trade=float(pnl.mean()), per_week_usd=float(pnl.sum() / weeks), total=float(pnl.sum()),
        lo=lo, dd=dd, streak=worst,
        cost=float(np.mean([r.paid for r in res]) * 100 * n_contracts),
        tgt=float(np.mean([r.outcome == "target" for r in res])),
        stops=float(np.mean([r.outcome == "stop" for r in res])),
        avg_ret=float(np.mean([r.ret for r in res])),
        years=f"{sum(1 for v in by_year.values() if sum(v) > 0)}/{len(by_year)}",
        by_year={y: (len(v), float(sum(v))) for y, v in sorted(by_year.items())},
    )


HEAD = ("| setup | contract | take profit | stop | trades | per week | win | avg win $ | avg loss $ | $ per trade "
        "| $ per week | total $ | worst drawdown $ | losing streak | hit target | cost of 5 contracts | years + |")
SEP = "|" + "---|" * 17


def line(key, s: dict, bold=False) -> str:
    setup, contract, tp, sl = key
    b = "**" if bold else ""
    return (f"| {b}{setup}{b} | {contract} | {pct(tp)} | {'strategy' if sl is None else '-' + format(sl, '.0%')} "
            f"| {s['n']} | {s['per_week']:.2f} | {s['win']:.0%} | ${s['avg_win']:+,.0f} | ${s['avg_loss']:+,.0f} "
            f"| ${s['per_trade']:+,.0f} | **${s['per_week_usd']:+,.0f}** | ${s['total']:+,.0f} | ${s['dd']:,.0f} "
            f"| {s['streak']} | {s['tgt']:.0%} | ${s['cost']:,.0f} | {s['years']} |")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", default="QQQ,SPY")
    p.add_argument("--contracts", type=int, default=5)
    p.add_argument("--split", help="check period start (default: 12 months before the last day)")
    p.add_argument("--iv-mult", type=float, default=1.1, help="implied vol = realised vol x this")
    p.add_argument("--min-trades", type=int, default=80, help="min tune trades for a config to count")
    args = p.parse_args()
    N = args.contracts

    cfg = load_config()
    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    bars = {s: load_history(s, date(2015, 1, 1), date.today()) for s in symbols}
    for s, b in bars.items():
        if b.empty:
            raise SystemExit(f"No history for {s}. Run tools/download_history.py first.")
    all_days = sorted(set().union(*(set(trading_days(b)) for b in bars.values())))
    split = date.fromisoformat(args.split) if args.split else all_days[-1] - timedelta(days=365)
    tune = {d for d in all_days if d < split}
    check = {d for d in all_days if d >= split}

    print("building day grids and regime ...")
    grids = {s: ix.build_grid(s, b) for s, b in bars.items()}
    ctx = {s: om.daily_context(g) for s, g in grids.items()}

    print("collecting signals (ORB uses the scorer, takes a minute) ...")
    signals = {setups.SETUPS[k]: v for k, v in setups.collect(bars, grids, cfg).items()}
    print("pricing option paths ...")
    paths = {}            # (setup, contract) -> {(symbol, day, m_in, direction): (sig, path)}
    for setup, sigs in signals.items():
        for c in om.CONTRACTS:
            paths[(setup, c.name)] = {(s.symbol, s.day, s.m_in, s.direction): (s, p)
                                      for s, p in om.build_paths(sigs, grids, ctx, c, args.iv_mult)}
        print(f"  {setup}: {len(sigs)} signals")

    print("replaying take-profit / stop / regime combinations ...")
    results = {}          # (setup [mode], contract, tp, sl) -> (tune summary, check summary)
    for setup, sigs in signals.items():
        for mode in MODES:
            chosen = om.one_per_day(om.apply_regime(sigs, ctx, mode))
            label = f"{setup} [{mode}]"
            for c in om.CONTRACTS:
                book = paths[(setup, c.name)]
                pp = [book[k] for k in ((s.symbol, s.day, s.m_in, s.direction) for s in chosen) if k in book]
                pt = [x for x in pp if x[0].day in tune]
                pc = [x for x in pp if x[0].day in check]
                for tp in TPS:
                    for sl in SLS:
                        results[(label, c.name, tp, sl)] = (summarise(om.run(pt, c.name, tp, sl), len(tune), N),
                                                            summarise(om.run(pc, c.name, tp, sl), len(check), N))

    def score(k):
        s = results[k][0]
        return s["lo"] * s["per_week"] if s["n"] >= args.min_trades and s["lo"] == s["lo"] else -1e18

    ranked = sorted(results, key=score, reverse=True)
    L = [f"# Money study {datetime.now():%Y-%m-%d}\n",
         f"- **{N} contracts every trade**, QQQ and SPY, at most 1 trade a day. Dollars are USD after the "
         "bid/ask spread and Wealthsimple's 1.5% FX fee each way (about 3% of the premium per trade).",
         f"- Tune {min(tune)} to {max(tune)} ({len(tune)} days); check {min(check)} to {max(check)} ({len(check)} days, never used to choose).",
         "- Take profit / stop are on the premium: +100% = the contract doubled. 'strategy' stop = no premium stop, "
         "the setup's own stop / trail / 4 pm exit.",
         f"- Option prices are modelled (Black-Scholes, implied vol = 20-day realised vol x {args.iv_mult:g}). "
         "Confirm the winner on real option prices before trading it.",
         "- Regime rules: " + "; ".join(f"`{k}` = {v}" for k, v in MODES.items()) + "\n",
         "## Top 15 on the tune period (ranked by the cautious estimate of $ per week)\n", HEAD, SEP]
    for k in ranked[:15]:
        L.append(line(k, results[k][0]))

    L += ["\n## Shortlist on the unseen CHECK year\n", HEAD, SEP]
    for k in ranked[:8]:
        L.append(line(k, results[k][0]))
        L.append(line((f"check: {k[0]}",) + k[1:], results[k][1], bold=True))

    best = ranked[0]
    setup_mode, contract = best[0], best[1]
    L += [f"\n## Take-profit sweep for the best setup: {setup_mode}, {contract}\n",
          "Which take profit is sustainable? Same setup and contract, every take-profit and stop. Tune / check.\n",
          "| take profit | stop | tune: trades/wk | tune: win | tune: $/trade | tune: $/week | check: win | check: $/trade | check: $/week |",
          "|---|---|---|---|---|---|---|---|---|"]
    for tp in TPS:
        for sl in SLS:
            t, c = results[(setup_mode, contract, tp, sl)]
            L.append(f"| {pct(tp)} | {'strategy' if sl is None else '-' + format(sl, '.0%')} | {t['per_week']:.2f} "
                     f"| {t['win']:.0%} | ${t['per_trade']:+,.0f} | ${t['per_week_usd']:+,.0f} | {c['win']:.0%} "
                     f"| ${c['per_trade']:+,.0f} | ${c['per_week_usd']:+,.0f} |")

    L += [f"\n## Contract choice for {setup_mode} (take profit {pct(best[2])}, stop "
          f"{'strategy' if best[3] is None else '-' + format(best[3], '.0%')})\n", HEAD, SEP]
    for c in om.CONTRACTS:
        k = (setup_mode, c.name, best[2], best[3])
        L.append(line(k, results[k][0]))
        L.append(line((f"check: {k[0]}",) + k[1:], results[k][1], bold=True))

    L += ["\n## Regime rule comparison (best contract / take profit for each setup + rule, tune vs check)\n",
          "| setup + rule | contract | take profit | stop | tune $/week | tune win | check $/week | check win | check trades |",
          "|---|---|---|---|---|---|---|---|---|"]
    seen = set()
    for k in ranked:
        if k[0] in seen or results[k][0]["n"] < args.min_trades:
            continue
        seen.add(k[0])
        t, c = results[k]
        L.append(f"| {k[0]} | {k[1]} | {pct(k[2])} | {'strategy' if k[3] is None else '-' + format(k[3], '.0%')} "
                 f"| ${t['per_week_usd']:+,.0f} | {t['win']:.0%} | ${c['per_week_usd']:+,.0f} | {c['win']:.0%} | {c['n']} |")

    t, c = results[best]
    L += [f"\n## By year: {setup_mode}, {contract}, take profit {pct(best[2])}\n",
          "| year | trades | total $ |", "|---|---|---|"]
    for y, (n, tot) in sorted({**t["by_year"], **c["by_year"]}.items()):
        L.append(f"| {y} | {n} | ${tot:+,.0f} |")

    L += ["\n## Bottom line\n",
          f"- Best on tune: **{setup_mode}**, {contract}, take profit {pct(best[2])}, stop "
          f"{'strategy' if best[3] is None else '-' + format(best[3], '.0%')}: {t['per_week']:.2f} trades/week, "
          f"win {t['win']:.0%}, ${t['per_trade']:+,.0f}/trade, **${t['per_week_usd']:+,.0f}/week** with {N} contracts.",
          f"- Unseen check year: {c['n']} trades, win {c['win']:.0%}, ${c['per_trade']:+,.0f}/trade, "
          f"**${c['per_week_usd']:+,.0f}/week**, total ${c['total']:+,.0f}, worst drawdown ${c['dd']:,.0f}.",
          "- " + ("Held up on the unseen year." if c["per_week_usd"] > 0 else "Did NOT hold up on the unseen year.")]
    out = ROOT / "reports" / f"money_{datetime.now():%Y-%m-%d}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[-4:]))
    print(f"\nreport written: {out}\nNext: git add reports; git commit -m \"money study\"; git pull --rebase; git push")


if __name__ == "__main__":
    main()
