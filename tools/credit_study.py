"""Backtest 0DTE credit spreads on QQQ / SPY with REAL option prices.

    python tools/download_chains.py      # once (resumable), real 0DTE option bars
    python tools/credit_study.py
    python tools/credit_study.py --slip 0.02 --budget 1000

What it sells (all 0DTE, closed by 15:50 at the latest):
  put        bull put spread every day (the market drifts up)
  call       bear call spread every day
  condor     iron condor (both) every day
  trend      at entry: above VWAP and above the open -> put spread, below both -> call spread, else no trade
  trend_ic   same, but an iron condor when the direction is unclear
  regime     put spread, except call spread on "down" days (below 20-day avg + sharp 3-day drop)

x entry time (10:00 ... 13:30) x short strike distance (0.2% ... 1.2% out of the money)
x width ($1 / $2 / $3) x take profit (keep 50% / 75% of the credit, or hold) x stop
(buy back at 2x / 3x the credit, or none). One trade a day: QQQ, or SPY if QQQ can't be priced.

Sizing: as many spreads as keep the MAX possible loss near --budget ($1,000).
Wealthsimple: $0 per contract, USD account = no FX. --slip is paid per leg on entry AND exit.
Every choice is made on the TUNE period; the last 12 months (CHECK) are looked at once.
Writes reports/credit_<date>.md.
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scanner import intraday as ix  # noqa: E402
from scanner import optionsim as om  # noqa: E402
from scanner import spreads as sp  # noqa: E402
from scanner.config import ROOT  # noqa: E402
from scanner.history import ET, load_history  # noqa: E402

OPT_DIR = ROOT / "data" / "options"
RULES = ["put", "call", "condor", "trend", "trend_ic", "regime"]
TIMES = {30: "10:00", 60: "10:30", 120: "11:30", 180: "12:30", 240: "13:30"}
DISTS = [0.002, 0.004, 0.006, 0.008, 0.012]
WIDTHS = [1, 2, 3]
TPS = [0.5, 0.75, None]
SLS = [1.0, 2.0, None]          # stop when buy-back >= credit x (1 + sl): 2x / 3x the credit


def structure_for(rule: str, g: ix.DayGrid, i: int, m: int, down: bool) -> str | None:
    if rule in ("put", "call", "condor"):
        return rule
    if rule == "regime":
        return "call" if down else "put"
    c, v, o = g.close[i, m - 1], g.vwap[i, m - 1], g.day_open[i]
    if c > v and c > o:
        return "put"
    if c < v and c < o:
        return "call"
    return "condor" if rule == "trend_ic" else None


def stats(pnls: list[float], days: list[date], n_days: int) -> dict:
    weeks = max(n_days / 5, 1)
    p = np.array(pnls, float)
    if len(p) == 0:
        return dict(n=0, wk=0.0, win=0.0, aw=0.0, al=0.0, per=0.0, usd_wk=0.0, total=0.0, lo=-1e9, dd=0.0,
                    worst=0.0, streak=0, years="0/0")
    eq = np.concatenate([[0], np.cumsum(p)])
    streak = worst_s = 0
    for x in p:
        streak = streak + 1 if x < 0 else 0
        worst_s = max(worst_s, streak)
    yrs: dict[int, float] = {}
    for d, x in zip(days, p):
        yrs[d.year] = yrs.get(d.year, 0.0) + x
    sd = p.std(ddof=1) if len(p) > 1 else 0.0
    return dict(n=len(p), wk=len(p) / weeks, win=float((p > 0).mean()),
                aw=float(p[p > 0].mean()) if (p > 0).any() else 0.0,
                al=float(p[p <= 0].mean()) if (p <= 0).any() else 0.0,
                per=float(p.mean()), usd_wk=float(p.sum() / weeks), total=float(p.sum()),
                lo=float(p.mean() - 1.645 * sd / math.sqrt(len(p))) if len(p) > 1 else -1e9,
                dd=float((np.maximum.accumulate(eq) - eq).max()), worst=float(p.min()), streak=worst_s,
                years=f"{sum(v > 0 for v in yrs.values())}/{len(yrs)}")


HEAD = ("| rule | entry | short strike | width | take profit | stop | trades | per week | win | avg win | avg loss "
        "| worst | $ per trade | **$ per week** | total | worst drawdown | losing streak | years + |")
SEP = "|" + "---|" * 18


def label(cfg):
    rule, m, dist, width, tp, sl = cfg
    return (f"{rule} | {TIMES[m]} | {dist:.1%} OTM | ${width} | {'hold' if tp is None else f'keep {tp:.0%}'} "
            f"| {'none' if sl is None else f'{sl + 1:g}x credit'}")


def line(name, cfg, s, bold=False):
    b = "**" if bold else ""
    return (f"| {b}{name}{b}{label(cfg)[len(cfg[0]):]} | {s['n']} | {s['wk']:.2f} | {s['win']:.0%} | ${s['aw']:+,.0f} "
            f"| ${s['al']:+,.0f} | ${s['worst']:+,.0f} | ${s['per']:+,.0f} | **${s['usd_wk']:+,.0f}** | ${s['total']:+,.0f} "
            f"| ${s['dd']:,.0f} | {s['streak']} | {s['years']} |")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", default="QQQ,SPY", help="first = preferred each day")
    p.add_argument("--start", default="2024-02-05")
    p.add_argument("--budget", type=float, default=1000.0, help="max possible loss per trade, $")
    p.add_argument("--slip", type=float, default=0.01, help="$ per leg per side")
    p.add_argument("--split", help="check period start (default: 12 months before the last day)")
    p.add_argument("--min-trades", type=int, default=100)
    args = p.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    start = date.fromisoformat(args.start)
    grids, ctx = {}, {}
    for s in symbols:
        b = load_history(s, start - timedelta(days=60), date.today())
        if b.empty:
            raise SystemExit(f"No stock history for {s}.")
        grids[s] = ix.build_grid(s, b)
        ctx[s] = om.daily_context(grids[s])
    days = sorted(d for d in set().union(*(set(g.days) for g in grids.values())) if d >= start)
    files = {(s, d): OPT_DIR / f"{s}_{d}_0dte.csv.gz" for s in symbols for d in days}
    have = sorted({d for (s, d), f in files.items() if f.exists()})
    if not have:
        raise SystemExit("No option chains. Run tools/download_chains.py first.")
    split = date.fromisoformat(args.split) if args.split else have[-1] - timedelta(days=365)
    tune_days = [d for d in have if d < split]
    check_days = [d for d in have if d >= split]

    configs = list(itertools.product(RULES, TIMES, DISTS, WIDTHS, TPS, SLS))
    pnl = {c: [] for c in configs}                      # (day, $, symbol, outcome)
    pos = {s: {d: i for i, d in enumerate(g.days)} for s, g in grids.items()}
    print(f"{len(have)} days with option data; {len(configs)} configurations ...")
    for n, d in enumerate(have, 1):
        done_cfg = set()
        for s in symbols:                               # preferred symbol first; fall back if it can't trade
            f = files[(s, d)]
            if not f.exists() or d not in pos[s]:
                continue
            i = pos[s][d]
            g = grids[s]
            chain = sp.load_chain(pd.read_csv(f), d, ET)
            if not chain.price:
                continue
            S = g.close[i]
            down = bool(ctx[s].at[d, "down_k1"]) if d in ctx[s].index else False
            prepared = {}
            for rule, m, dist, width, tp, sl in configs:
                cfg = (rule, m, dist, width, tp, sl)
                if cfg in done_cfg:
                    continue
                st = structure_for(rule, g, i, m, down)
                if st is None:
                    done_cfg.add(cfg)                   # rule says: no trade today (same for every symbol)
                    continue
                key = (st, m, dist, width)
                if key not in prepared:
                    legs = sp.legs_for(st, float(g.close[i, m - 1]), dist, width)
                    prepared[key] = sp.prepare(chain, S, legs, m, width, args.slip)
                got = prepared[key]
                if got is None:
                    continue                            # try the next symbol
                credit, cost = got
                j, why = sp.exit_on(credit, cost, tp, sl)
                per = (credit - float(cost[j])) * 100
                max_loss = max(width - credit, 0.01) * 100
                k = max(1, int(args.budget // max_loss))
                pnl[cfg].append((d, per * k, s, why))
                done_cfg.add(cfg)
        if n % 25 == 0 or n == len(have):
            print(f"  {n}/{len(have)} days")

    def summ(cfg, period):
        rows = [r for r in pnl[cfg] if r[0] in period]
        return stats([r[1] for r in rows], [r[0] for r in rows], len(period))

    tset, cset = set(tune_days), set(check_days)
    tune = {c: summ(c, tset) for c in configs}
    ok = [c for c in configs if tune[c]["n"] >= args.min_trades]
    ranked = sorted(ok, key=lambda c: tune[c]["lo"] * tune[c]["wk"], reverse=True)

    L = [f"# 0DTE credit spreads on REAL option prices ({datetime.now():%Y-%m-%d})\n",
         f"- {', '.join(symbols)} ({symbols[0]} preferred), one trade a day, closed by 15:50. "
         f"Sized so the max possible loss is about ${args.budget:,.0f} per trade.",
         f"- Real traded option prices (Alpaca), ${args.slip:.2f} slippage per leg on entry and exit, $0 commission, no FX.",
         f"- Tune {tune_days[0]} to {tune_days[-1]} ({len(tune_days)} days); check {check_days[0]} to "
         f"{check_days[-1]} ({len(check_days)} days, never used to choose).",
         f"- {len(configs)} combinations tested; ranked by a cautious estimate of $ per week on tune.\n",
         "## Top 20 on the tune period\n", HEAD, SEP]
    for c in ranked[:20]:
        L.append(line(c[0], c, tune[c]))
    L += ["\n## The top 10, on the unseen CHECK year\n", HEAD, SEP]
    for c in ranked[:10]:
        L.append(line(c[0], c, tune[c]))
        L.append(line(f"check: {c[0]}", c, summ(c, cset), bold=True))

    # robustness: average over every take-profit / stop for each rule x time x distance x width
    L += ["\n## Robustness: each rule's average $ per week across ALL its settings (tune / check)\n",
          "If a rule only works with one exact setting, it is probably luck. This averages every setting.\n",
          "| rule | settings | tune: avg $/week | tune: share of settings profitable | check: avg $/week | check: share profitable |",
          "|---|---|---|---|---|---|"]
    for rule in RULES:
        cs = [c for c in ok if c[0] == rule]
        if not cs:
            continue
        tw = [tune[c]["usd_wk"] for c in cs]
        cw = [summ(c, cset)["usd_wk"] for c in cs]
        L.append(f"| {rule} | {len(cs)} | ${np.mean(tw):+,.0f} | {np.mean([x > 0 for x in tw]):.0%} "
                 f"| ${np.mean(cw):+,.0f} | {np.mean([x > 0 for x in cw]):.0%} |")

    L += ["\n## By short-strike distance and entry time (best rule/width/exit for each cell, tune $ per week)\n",
          "| entry | " + " | ".join(f"{x:.1%} OTM" for x in DISTS) + " |", "|---|" + "---|" * len(DISTS)]
    for m in TIMES:
        cells = []
        for dist in DISTS:
            cs = [c for c in ok if c[1] == m and c[2] == dist]
            cells.append(f"${max(tune[c]['usd_wk'] for c in cs):+,.0f}" if cs else "-")
        L.append(f"| {TIMES[m]} | " + " | ".join(cells) + " |")

    if ranked:
        best = ranked[0]
        rows = pnl[best]
        L += [f"\n## Best setting in detail: {label(best)}\n", "| year | trades | total $ |", "|---|---|---|"]
        yrs: dict = {}
        for d, x, s, why in rows:
            n0, t0 = yrs.get(d.year, (0, 0.0))
            yrs[d.year] = (n0 + 1, t0 + x)
        L += [f"| {y} | {n0} | ${t0:+,.0f} |" for y, (n0, t0) in sorted(yrs.items())]
        outs: dict = {}
        for _, x, s, why in rows:
            outs.setdefault(why, []).append(x)
        L += ["", "| how it ended | trades | avg $ |", "|---|---|---|"]
        L += [f"| {k} | {len(v)} | ${np.mean(v):+,.0f} |" for k, v in outs.items()]
        worst = sorted(rows, key=lambda r: r[1])[:10]
        L += ["", "Worst 10 days: " + ", ".join(f"{d} {s} ${x:+,.0f}" for d, x, s, _ in worst)]
        c = summ(best, cset)
        L += ["\n## Bottom line\n",
              f"- Best on tune: **{label(best)}**: {tune[best]['wk']:.2f}/week, win {tune[best]['win']:.0%}, "
              f"${tune[best]['per']:+,.0f}/trade, **${tune[best]['usd_wk']:+,.0f}/week**.",
              f"- Unseen check year: {c['n']} trades, win {c['win']:.0%}, ${c['per']:+,.0f}/trade, "
              f"**${c['usd_wk']:+,.0f}/week**, total ${c['total']:+,.0f}, worst day ${c['worst']:+,.0f}, "
              f"worst drawdown ${c['dd']:,.0f}.",
              "- " + ("Held up on the unseen year." if c["usd_wk"] > 0 else "Did NOT hold up on the unseen year.")]
    out = ROOT / "reports" / f"credit_{datetime.now():%Y-%m-%d}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[-4:]))
    print(f"\nreport written: {out}\nNext: git add reports; git commit -m \"credit study\"; git pull --rebase; git push")


if __name__ == "__main__":
    main()
