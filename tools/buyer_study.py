"""Can we pay less for the option? Every way of BUYING our signals, on REAL 0DTE prices.

    python tools/buyer_study.py              # needs data/options (tools/download_chains.py)
    python tools/buyer_study.py --budget 1000 --slip 0.01

Signals: the 12 setups in scanner/library.py (calls on long signals, puts on
short ones; 'long' = calls only). One trade a day per setup (QQQ preferred).

Ways to buy (scanner/debit.py), all real traded option prices:
  atm / itm2 / otm1      plain option at the money / $2 in the money / $1 out of the money
  ds2 / ds3 / ds5        debit spread: buy at the money, sell $2 / $3 / $5 further out
Cheap-option filter: the at-the-money straddle at entry (what the market charges
for the rest of today's move) vs how much QQQ/SPY actually moved over the same
part of the day on the previous 10 days. 'cheap' = below the median (set on tune).
Exits: take profit +25% / +50% / +100% / hold, stop -25% / -50% / none, max hold 15 / 30 / 60 minutes or none
(else sold at 15:50). About --budget $ per trade. $0 commission, no FX.

Choices on the TUNE period; the last 12 months (CHECK) looked at once.
Writes reports/buyer_<date>.md.
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
from scanner import debit as db  # noqa: E402
from scanner import intraday as ix  # noqa: E402
from scanner import library as lib  # noqa: E402
from scanner import optionsim as om  # noqa: E402
from scanner import spreads as sp  # noqa: E402
from scanner.config import ROOT  # noqa: E402
from scanner.history import ET, load_history  # noqa: E402

OPT_DIR = ROOT / "data" / "options"
MODES = ["both", "long"]
TPS = [0.25, 0.5, 1.0, None]
SLS = [0.25, 0.5, None]
TIME = {15: "15 min", 30: "30 min", 60: "60 min", None: "to 15:50"}   # max minutes held
FILTERS = ["all", "cheap", "dear"]


def realized_remaining(g: ix.DayGrid, lookback: int = 10) -> np.ndarray:
    """For day i, minute m: mean over the PRIOR `lookback` days of |close(15:59) / close(m-1) - 1|."""
    rem = np.abs(g.close[:, -1:] / g.close - 1)                      # day x minute (uses that day's close)
    out = np.full_like(rem, np.nan)
    for i in range(lookback, len(g.days)):
        out[i] = rem[i - lookback:i].mean(axis=0)
    return out


def stats(rows, n_days: int) -> dict:
    weeks = max(n_days / 5, 1)
    p = np.array([r[1] for r in rows], float)
    if len(p) == 0:
        return dict(n=0, wk=0.0, win=0.0, aw=0.0, al=0.0, per=0.0, usd_wk=0.0, total=0.0, lo=-1e9, dd=0.0, years="0/0",
                    hold=0.0)
    eq = np.concatenate([[0], np.cumsum(p)])
    yrs: dict = {}
    for r, x in zip(rows, p):
        yrs[r[0].year] = yrs.get(r[0].year, 0.0) + x
    sd = p.std(ddof=1) if len(p) > 1 else 0.0
    return dict(n=len(p), wk=len(p) / weeks, win=float((p > 0).mean()),
                aw=float(p[p > 0].mean()) if (p > 0).any() else 0.0,
                al=float(p[p <= 0].mean()) if (p <= 0).any() else 0.0,
                per=float(p.mean()), usd_wk=float(p.sum() / weeks), total=float(p.sum()),
                lo=float(p.mean() - 1.645 * sd / math.sqrt(len(p))) if len(p) > 1 else -1e9,
                dd=float((np.maximum.accumulate(eq) - eq).max()),
                years=f"{sum(v > 0 for v in yrs.values())}/{len(yrs)}",
                hold=float(np.mean([r[3] for r in rows])))


def name(c) -> str:
    setup, mode, st, flt, tp, sl, tb = c
    return (f"{setup} | {mode} | {st} | {flt} | {'hold' if tp is None else f'+{tp:.0%}'} | "
            f"{'none' if sl is None else f'-{sl:.0%}'} | {TIME[tb]}")


HEAD = ("| setup | sides | how bought | filter | take profit | stop | max hold | trades | per week | win | avg win "
        "| avg loss | avg minutes held | $ per trade | **$ per week** | total | worst drawdown | years + |")
SEP = "|" + "---|" * 18


def line(c, s, prefix="", bold=False) -> str:
    b = "**" if bold else ""
    return (f"| {b}{prefix}{name(c)}{b} | {s['n']} | {s['wk']:.2f} | {s['win']:.0%} | ${s['aw']:+,.0f} | ${s['al']:+,.0f} "
            f"| {s['hold']:.0f} | ${s['per']:+,.0f} | **${s['usd_wk']:+,.0f}** | ${s['total']:+,.0f} | ${s['dd']:,.0f} | {s['years']} |")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", default="QQQ,SPY")
    p.add_argument("--start", default="2024-02-05")
    p.add_argument("--budget", type=float, default=1000.0)
    p.add_argument("--slip", type=float, default=0.01, help="$ per leg per side")
    p.add_argument("--split")
    p.add_argument("--min-trades", type=int, default=80)
    args = p.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    start = date.fromisoformat(args.start)
    grids, ctx, real = {}, {}, {}
    for s in symbols:
        b = load_history(s, start - timedelta(days=60), date.today())
        if b.empty:
            raise SystemExit(f"No stock history for {s}.")
        grids[s] = ix.build_grid(s, b)
        ctx[s] = om.daily_context(grids[s])
        real[s] = realized_remaining(grids[s])
    pos = {s: {d: i for i, d in enumerate(g.days)} for s, g in grids.items()}
    have = sorted({d for s in symbols for d in grids[s].days
                   if d >= start and (OPT_DIR / f"{s}_{d}_0dte.csv.gz").exists()})
    if not have:
        raise SystemExit("No option chains. Run tools/download_chains.py first.")
    split = date.fromisoformat(args.split) if args.split else have[-1] - timedelta(days=365)
    tune_days, check_days = [d for d in have if d < split], [d for d in have if d >= split]

    print("signals ...")
    sigs = lib.all_signals(grids, ctx)
    chosen = {}                                           # (setup, mode) -> {day: [signals, preferred first]}
    for setup in lib.SETUPS:
        for mode in MODES:
            v = sigs[setup] if mode == "both" else [s for s in sigs[setup] if s.direction == "long"]
            byday: dict = {}
            for s in sorted(v, key=lambda s: (s.day, s.m_in, s.symbol != symbols[0])):
                byday.setdefault(s.day, []).append(s)
            chosen[(setup, mode)] = byday

    base = list(itertools.product(lib.SETUPS, MODES, db.STRUCTURES, TPS, SLS, TIME))
    rows = {c: [] for c in base}
    moves: list = []                                       # (minutes held, underlying move our way, option return)                           # (day, $, ratio)
    print(f"{len(have)} days with option data; {len(base) * len(FILTERS)} combinations ...")
    for n, d in enumerate(have, 1):
        chains = {}
        for s in symbols:
            f = OPT_DIR / f"{s}_{d}_0dte.csv.gz"
            if f.exists():
                chains[s] = sp.load_chain(pd.read_csv(f), d, ET)
        cache = {}
        seen: set = set()
        for setup in lib.SETUPS:
            for mode in MODES:
                for st in db.STRUCTURES:
                    got = None
                    for sig in chosen[(setup, mode)].get(d, []):        # first signal that can be priced
                        ch = chains.get(sig.symbol)
                        if ch is None or not ch.price:
                            continue
                        key = (sig.symbol, sig.m_in, sig.direction, st)
                        if key not in cache:
                            g, i = grids[sig.symbol], pos[sig.symbol][d]
                            S0 = float(g.close[i, sig.m_in - 1])
                            legs, cap = db.legs_for(st, S0, sig.direction)
                            path = db.prepare(ch, g.close[i], legs, sig.m_in, cap, args.slip)
                            ratio = np.nan
                            k = float(round(S0))
                            c_, p_ = ("C", k), ("P", k)
                            if c_ in ch.price and p_ in ch.price and ch.age[c_][sig.m_in - 1] <= sp.MAX_AGE \
                                    and ch.age[p_][sig.m_in - 1] <= sp.MAX_AGE:
                                straddle = ch.price[c_][sig.m_in - 1] + ch.price[p_][sig.m_in - 1]
                                rr = real[sig.symbol][i, sig.m_in]
                                if rr == rr and rr > 0:
                                    ratio = (straddle / S0) / rr
                            cache[key] = (path, ratio, sig)
                        if cache[key][0] is not None:
                            got = cache[key]
                            break
                    if got is None:
                        continue
                    (debit, value), ratio, sig = got
                    if st == "atm" and (sig.symbol, sig.m_in, sig.direction) not in seen:
                        seen.add((sig.symbol, sig.m_in, sig.direction))
                        g, i = grids[sig.symbol], pos[sig.symbol][d]
                        sgn = 1 if sig.direction == "long" else -1
                        for h in (15, 30, 60):
                            if h < len(value) and sig.m_in + h < 390:
                                mv = sgn * (g.close[i, sig.m_in + h] / g.close[i, sig.m_in - 1] - 1)
                                moves.append((h, mv, float(value[h]) / debit - 1))
                    k_n = max(1, int(args.budget // (debit * 100)))
                    for tp, sl, tb in itertools.product(TPS, SLS, TIME):
                        j, _ = db.exit_on(debit, value, sig.m_in, tp, sl, None if tb is None else sig.m_in + tb)
                        rows[(setup, mode, st, tp, sl, tb)].append((d, k_n * 100 * (float(value[j]) - debit), ratio, j))
        if n % 25 == 0 or n == len(have):
            print(f"  {n}/{len(have)} days")

    tset, cset = set(tune_days), set(check_days)
    ratios = [r[2] for c in base for r in rows[c] if r[0] in tset and r[2] == r[2]]
    cut = float(np.median(ratios)) if ratios else 1.0

    def pick(c, flt):
        rs = rows[c]
        if flt == "cheap":
            rs = [r for r in rs if r[2] == r[2] and r[2] < cut]
        elif flt == "dear":
            rs = [r for r in rs if r[2] == r[2] and r[2] >= cut]
        return rs

    configs = [(a, b, st, flt, tp, sl, tb) for (a, b, st, tp, sl, tb) in base for flt in FILTERS]

    def summ(c, period):
        a, b, st, flt, tp, sl, tb = c
        return stats([r for r in pick((a, b, st, tp, sl, tb), flt) if r[0] in period], len(period))

    tune = {c: summ(c, tset) for c in configs}
    check = {c: summ(c, cset) for c in configs}
    ok = [c for c in configs if tune[c]["n"] >= args.min_trades]
    ranked = sorted(ok, key=lambda c: tune[c]["lo"] * tune[c]["wk"], reverse=True)

    L = [f"# Buying our signals for less: REAL 0DTE option prices ({datetime.now():%Y-%m-%d})\n",
         f"- {', '.join(symbols)} ({symbols[0]} preferred), one trade a day per setup, about ${args.budget:,.0f} per trade, "
         f"${args.slip:.2f} slippage per leg each way, $0 commission, no FX. Sold by 15:50 at the latest.",
         f"- Tune {tune_days[0]} to {tune_days[-1]} ({len(tune_days)} days); check {check_days[0]} to {check_days[-1]} "
         f"({len(check_days)} days, never used to choose).",
         f"- Cheap-option filter cut: straddle / normal remaining move = {cut:.2f} (median on tune; above 1 means the "
         "market charges more than it usually moves).",
         f"- {len(configs)} combinations; ranked by a cautious estimate of $ per week on tune.\n"]

    def compare(title, idx, values, labels=None):
        out = [f"## {title}: average over every other setting (tune -> check)\n",
               "| choice | settings | tune avg $/week | tune share profitable | check avg $/week | check share profitable "
               "| best on tune | that one on check |", "|---|---|---|---|---|---|---|---|"]
        for v in values:
            cs = [c for c in ok if c[idx] == v]
            if not cs:
                continue
            b = max(cs, key=lambda c: tune[c]["usd_wk"])
            out.append(f"| {labels[v] if labels else v} | {len(cs)} | ${np.mean([tune[c]['usd_wk'] for c in cs]):+,.0f} "
                       f"| {np.mean([tune[c]['usd_wk'] > 0 for c in cs]):.0%} "
                       f"| ${np.mean([check[c]['usd_wk'] for c in cs]):+,.0f} "
                       f"| {np.mean([check[c]['usd_wk'] > 0 for c in cs]):.0%} "
                       f"| ${tune[b]['usd_wk']:+,.0f} | ${check[b]['usd_wk']:+,.0f} |")
        return out + [""]

    buckets = [(-1, -0.002), (-0.002, -0.001), (-0.001, 0), (0, 0.001), (0.001, 0.002), (0.002, 0.003),
               (0.003, 0.005), (0.005, 1)]
    L += ["## What a short hold really costs (plain at-the-money 0DTE, real prices)\n",
          "Every signal's option, bought at the ask and valued at the bid after 15 / 30 / 60 minutes, grouped by how far "
          "QQQ/SPY moved our way in that time. Median option return in each group (and how often that move happened).\n",
          "| underlying move our way | after 15 min | after 30 min | after 60 min |", "|---|---|---|---|"]
    for lo, hi in buckets:
        cells = []
        for h in (15, 30, 60):
            v = [r for hh, mv, r in moves if hh == h and lo <= mv < hi]
            n_h = sum(1 for hh, _, _ in moves if hh == h)
            cells.append(f"{np.median(v):+.0%} ({len(v) / max(n_h, 1):.0%} of trades)" if v else "-")
        lab = (f"below {hi:+.1%}" if lo == -1 else f"above {lo:+.1%}" if hi == 1 else f"{lo:+.1%} to {hi:+.1%}")
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    for h in (15, 30, 60):
        v = [(mv, r) for hh, mv, r in moves if hh == h]
        if len(v) > 20:
            mv_arr, r_arr = np.array(v).T
            flat = r_arr[np.abs(mv_arr) < 0.0005]
            slope, icpt = np.polyfit(mv_arr, r_arr, 1)
            L.append(f"\n- {h} min hold: if price doesn't move (within 0.05%) the option loses about "
                     f"**{-np.median(flat):.0%}** (median, spread + decay). Break-even move ~**{-icpt / slope:+.2%}** "
                     "our way (the table shows how big a move it takes to double).")
    L.append("")
    L += compare("How to buy", 2, db.STRUCTURES)
    L += compare("Cheap-option filter", 3, FILTERS)
    L += compare("Max hold", 6, list(TIME), TIME)
    L += compare("Setup", 0, lib.SETUPS)
    L += ["## Top 15 on tune, and the same settings on the unseen CHECK year\n", HEAD, SEP]
    for c in ranked[:15]:
        L.append(line(c, tune[c]))
        L.append(line(c, check[c], "check: ", bold=True))
    if ranked:
        b = ranked[0]
        both = sorted([r for r in pick(b[:3] + b[4:], b[3]) if r[0] in tset | cset], key=lambda r: r[0])
        yrs: dict = {}
        for d, x, _, _ in both:
            nn, tt = yrs.get(d.year, (0, 0.0))
            yrs[d.year] = (nn + 1, tt + x)
        L += [f"\n## Best on tune, by year: {name(b)}\n", "| year | trades | total $ |", "|---|---|---|"]
        L += [f"| {y} | {nn} | ${tt:+,.0f} |" for y, (nn, tt) in sorted(yrs.items())]
        L += ["\n## Bottom line\n",
              f"- Best on tune: **{name(b)}**: {tune[b]['wk']:.2f}/week, win {tune[b]['win']:.0%}, "
              f"${tune[b]['per']:+,.0f}/trade, **${tune[b]['usd_wk']:+,.0f}/week**.",
              f"- Unseen check year: {check[b]['n']} trades, win {check[b]['win']:.0%}, ${check[b]['per']:+,.0f}/trade, "
              f"**${check[b]['usd_wk']:+,.0f}/week**, total ${check[b]['total']:+,.0f}, worst drawdown ${check[b]['dd']:,.0f}.",
              f"- Settings profitable on BOTH tune and check: "
              f"{sum(1 for c in ok if tune[c]['usd_wk'] > 0 and check[c]['usd_wk'] > 0)} of {len(ok)}."]
    out = ROOT / "reports" / f"buyer_{datetime.now():%Y-%m-%d}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[-4:]))
    print(f"\nreport written: {out}\nNext: git add reports; git commit -m \"buyer study\"; git pull --rebase; git push")


if __name__ == "__main__":
    main()
