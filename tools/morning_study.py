"""Jack's strategy, tested on REAL 0DTE option prices:

  * enter between 9:30 and 11:00 only
  * hold no longer than 60 or 90 minutes ("hot potato" - limits time decay)
  * stop at -25% / -35% / -50% of the premium at most
  * aim for LARGE moves: take profit +50% / +100% / +150%, or ride it until the time limit
  * one trade a day, ~$1,000 per trade, QQQ (SPY if QQQ can't be priced)

    python tools/morning_study.py        # needs data/options from tools/download_chains.py

Signals: the published setups in scanner/library.py, restricted to 9:30-11:00, plus
'coin' - a coin-flip direction at 9:45. The coin is a CONTROL: it has no edge, so if
it shows a profit, the backtest is flattering itself and nothing else can be trusted.

Big-move signs (scanner/bigmove.py): relative volume, gap, NR7 / inside day,
yesterday's big move, wide opening range, already moving, daily trend. Which signs
count is decided on the TUNE period by how much they raise the chance of a 0.35%+
move within the hour; then 'grade' = how many of them a trade has.

Ways to buy: at the money, $2 in the money, debit spreads $2 / $3 wide.
Choices on TUNE; the last 12 months (CHECK) looked at once. Writes reports/morning_<date>.md.
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
from scanner import bigmove as bm  # noqa: E402
from scanner import debit as db  # noqa: E402
from scanner import intraday as ix  # noqa: E402
from scanner import library as lib  # noqa: E402
from scanner import optionsim as om  # noqa: E402
from scanner import spreads as sp  # noqa: E402
from scanner.config import ROOT  # noqa: E402
from scanner.history import ET, load_history  # noqa: E402

OPT_DIR = ROOT / "data" / "options"
LAST_ENTRY = 90                                  # bar 90 = 11:00
SETUPS = lib.SETUPS + ["coin"]
MODES = ["both", "long", "regime_k1"]
STRUCTS = ["atm", "itm2", "ds2", "ds3"]
TPS = [0.5, 1.0, 1.5, None]
SLS = [0.25, 0.35, 0.5]
HOLDS = [60, 90]
GRADES = [0, 1, 2, 3]
BIG = 0.0035                                     # "big move": 0.35% our way within the hour
SIGN_KEYS = list(bm.SIGNS)


def coin_signals(grids) -> list[om.Signal]:
    out = []
    for sym, g in grids.items():
        for i in range(21, len(g.days)):
            d = g.days[i]
            rng = np.random.default_rng(int(d.strftime("%Y%m%d")))
            direction = "long" if rng.random() < 0.5 else "short"
            out.append(om.Signal("coin", sym, d, direction, 15, ix.MINUTES - 1, float(g.close[i, -1]), ix.MINUTES - 0.5))
    return out


def stats(rows, n_days: int) -> dict:
    weeks = max(n_days / 5, 1)
    p = np.array([r[1] for r in rows], float)
    if len(p) == 0:
        return dict(n=0, wk=0.0, win=0.0, aw=0.0, al=0.0, per=0.0, med=0.0, usd_wk=0.0, total=0.0, ex5=0.0,
                    lo=-1e9, dd=0.0, hold=0.0, years="0/0")
    eq = np.concatenate([[0], np.cumsum(p)])
    yrs: dict = {}
    for r, x in zip(rows, p):
        yrs[r[0].year] = yrs.get(r[0].year, 0.0) + x
    sd = p.std(ddof=1) if len(p) > 1 else 0.0
    top5 = np.sort(p)[-5:].sum() if len(p) >= 5 else p.sum()
    return dict(n=len(p), wk=len(p) / weeks, win=float((p > 0).mean()),
                aw=float(p[p > 0].mean()) if (p > 0).any() else 0.0,
                al=float(p[p <= 0].mean()) if (p <= 0).any() else 0.0,
                per=float(p.mean()), med=float(np.median(p)), usd_wk=float(p.sum() / weeks), total=float(p.sum()),
                ex5=float(p.sum() - top5),
                lo=float(p.mean() - 1.645 * sd / math.sqrt(len(p))) if len(p) > 1 else -1e9,
                dd=float((np.maximum.accumulate(eq) - eq).max()), hold=float(np.mean([r[3] for r in rows])),
                years=f"{sum(v > 0 for v in yrs.values())}/{len(yrs)}")


def cname(c) -> str:
    setup, mode, st, tp, sl, hold, gr = c
    return (f"{setup} | {mode} | {st} | {'ride' if tp is None else f'+{tp:.0%}'} | -{sl:.0%} | {hold} min | "
            f"grade {gr}+")


HEAD = ("| setup | sides | how bought | take profit | stop | max hold | big-move grade | trades | per week | win "
        "| avg win | avg loss | median | minutes held | $ per trade | **$ per week** | total | total without best 5 "
        "| worst drawdown | years + |")
SEP = "|" + "---|" * 20


def line(c, s, prefix="", bold=False, label=None) -> str:
    b = "**" if bold else ""
    return (f"| {b}{prefix}{label or cname(c)}{b} | {s['n']} | {s['wk']:.2f} | {s['win']:.0%} | ${s['aw']:+,.0f} | ${s['al']:+,.0f} "
            f"| ${s['med']:+,.0f} | {s['hold']:.0f} | ${s['per']:+,.0f} | **${s['usd_wk']:+,.0f}** | ${s['total']:+,.0f} "
            f"| ${s['ex5']:+,.0f} | ${s['dd']:,.0f} | {s['years']} |")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", default="QQQ,SPY")
    p.add_argument("--start", default="2024-02-05")
    p.add_argument("--budget", type=float, default=1000.0)
    p.add_argument("--slip", type=float, default=0.01, help="$ per leg per side")
    p.add_argument("--max-contracts", type=int, default=50)
    p.add_argument("--split")
    p.add_argument("--min-trades", type=int, default=60)
    args = p.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    start = date.fromisoformat(args.start)
    grids, ctx, feats = {}, {}, {}
    for s in symbols:
        b = load_history(s, start - timedelta(days=90), date.today())
        if b.empty:
            raise SystemExit(f"No stock history for {s}.")
        grids[s] = ix.build_grid(s, b)
        ctx[s] = om.daily_context(grids[s])
        feats[s] = bm.Features(grids[s])
    pos = {s: {d: i for i, d in enumerate(g.days)} for s, g in grids.items()}
    have = sorted({d for s in symbols for d in grids[s].days
                   if d >= start and (OPT_DIR / f"{s}_{d}_0dte.csv.gz").exists()})
    if not have:
        raise SystemExit("No option chains. Run tools/download_chains.py first.")
    split = date.fromisoformat(args.split) if args.split else have[-1] - timedelta(days=365)
    tune_days, check_days = [d for d in have if d < split], [d for d in have if d >= split]
    tset, cset = set(tune_days), set(check_days)

    print("signals ...")
    sigs = lib.all_signals(grids, ctx)
    sigs["coin"] = coin_signals(grids)
    chosen = {}
    for setup in SETUPS:
        for mode in MODES:
            out: dict = {}
            for s in sorted(sigs[setup], key=lambda s: (s.day, s.m_in, s.symbol != symbols[0])):
                if s.m_in > LAST_ENTRY:
                    continue
                if mode == "long" and s.direction != "long":
                    continue
                if mode == "regime_k1":
                    c = ctx[s.symbol]
                    down = bool(c.at[s.day, "down_k1"]) if s.day in c.index else False
                    if (s.direction == "long") == down:
                        continue
                out.setdefault(s.day, []).append(s)
            chosen[(setup, mode)] = out

    base = list(itertools.product(SETUPS, MODES, STRUCTS, TPS, SLS, HOLDS))
    rows = {c: [] for c in base}                    # (day, $, signs, minutes held, m_in)
    labels = []                                     # (day, setup, signs, big move?, $ ref)
    print(f"{len(have)} days with option data ...")
    for n, d in enumerate(have, 1):
        chains = {}
        for s in symbols:
            f = OPT_DIR / f"{s}_{d}_0dte.csv.gz"
            if f.exists():
                chains[s] = sp.load_chain(pd.read_csv(f), d, ET)
        cache, sign_cache = {}, {}
        for setup in SETUPS:
            for mode in MODES:
                for st in STRUCTS:
                    got = None
                    for sig in chosen[(setup, mode)].get(d, []):
                        ch = chains.get(sig.symbol)
                        if ch is None or not ch.price:
                            continue
                        key = (sig.symbol, sig.m_in, sig.direction, st)
                        if key not in cache:
                            g, i = grids[sig.symbol], pos[sig.symbol][d]
                            legs, cap = db.legs_for(st, float(g.close[i, sig.m_in - 1]), sig.direction)
                            cache[key] = db.prepare_full(ch, g.close[i], legs, sig.m_in, cap, args.slip)
                        if cache[key] is not None:
                            got = (cache[key], sig)
                            break
                    if got is None:
                        continue
                    (debit, value, fresh), sig = got
                    skey = (sig.symbol, sig.m_in, sig.direction)
                    if skey not in sign_cache:
                        i = pos[sig.symbol][d]
                        sg = feats[sig.symbol].signs(i, sig.m_in, sig.direction)
                        sign_cache[skey] = tuple(sg[k] for k in SIGN_KEYS) if sg else None
                    signs = sign_cache[skey]
                    if signs is None:
                        continue
                    k_n = min(args.max_contracts, max(1, int(args.budget // (debit * 100))))
                    for tp, sl, hold in itertools.product(TPS, SLS, HOLDS):
                        j, _ = db.exit_on(debit, value, sig.m_in, tp, sl, sig.m_in + hold, fresh)
                        pnl = k_n * 100 * (float(value[j]) - debit)
                        rows[(setup, mode, st, tp, sl, hold)].append((d, pnl, signs, j, sig.m_in))
                        if mode == "both" and st == "atm" and tp is None and sl == 0.5 and hold == 60 \
                                and setup != "coin":
                            g, i = grids[sig.symbol], pos[sig.symbol][d]
                            labels.append((d, setup, signs, bm.max_move(g, i, sig.m_in, sig.direction, 60) >= BIG, pnl))
        if n % 25 == 0 or n == len(have):
            print(f"  {n}/{len(have)} days")

    # which big-move signs count (tune only)
    lt = [r for r in labels if r[0] in tset]
    sign_rows, used = [], []
    for k_i, k in enumerate(SIGN_KEYS):
        w = [r for r in lt if r[2][k_i]]
        wo = [r for r in lt if not r[2][k_i]]
        if len(w) < 40 or len(wo) < 40:
            sign_rows.append((k, len(w), None, None, None, None, False))
            continue
        bw, bwo = np.mean([r[3] for r in w]), np.mean([r[3] for r in wo])
        pw, pwo = np.mean([r[4] for r in w]), np.mean([r[4] for r in wo])
        use = (bw - bwo) >= 0.05 and pw > pwo
        if use:
            used.append(k_i)
        sign_rows.append((k, len(w), bw, bwo, pw, pwo, use))

    def grade(signs) -> int:
        return sum(1 for k_i in used if signs[k_i])

    configs = [b + (gr,) for b in base for gr in GRADES]

    def subset(c, period):
        b, gr = c[:6], c[6]
        return [r for r in rows[b] if r[0] in period and grade(r[2]) >= gr]

    tune = {c: stats(subset(c, tset), len(tune_days)) for c in configs}
    check = {c: stats(subset(c, cset), len(check_days)) for c in configs}
    ok = [c for c in configs if tune[c]["n"] >= args.min_trades]
    real = [c for c in ok if c[0] != "coin"]
    ranked = sorted(real, key=lambda c: tune[c]["lo"] * tune[c]["wk"], reverse=True)

    L = [f"# Morning hot-potato strategy on REAL 0DTE option prices ({datetime.now():%Y-%m-%d})\n",
         "- Enter 9:30-11:00 only, hold at most 60 or 90 minutes, stop at -25% / -35% / -50% of the premium, "
         "take profit +50% / +100% / +150% or ride to the time limit.",
         f"- {', '.join(symbols)} ({symbols[0]} preferred), one trade a day per setup, about ${args.budget:,.0f} per trade "
         f"(max {args.max_contracts} contracts), ${args.slip:.2f} slippage per leg each way, $0 commission, no FX.",
         "- Real traded option prices. A take profit on a spread only counts in a minute where both legs traded.",
         f"- Tune {tune_days[0]} to {tune_days[-1]} ({len(tune_days)} days); check {check_days[0]} to {check_days[-1]} "
         f"({len(check_days)} days, never used to choose).\n",
         "## What causes a big move? (tune; big = 0.35%+ our way within the hour)\n",
         "| sign | trades with it | big-move rate with | without | $ per trade with (ref) | without | counts? |",
         "|---|---|---|---|---|---|---|"]
    for k, nw, bw, bwo, pw, pwo, use in sign_rows:
        if bw is None:
            L.append(f"| {bm.SIGNS[k]} | {nw} | - | - | - | - | too few |")
        else:
            L.append(f"| {bm.SIGNS[k]} | {nw} | {bw:.0%} | {bwo:.0%} | ${pw:+,.0f} | ${pwo:+,.0f} | {'**yes**' if use else ''} |")
    L.append(f"\nGrade = how many of the {len(used)} counted signs a trade has. "
             "Reference trade for this table: at the money, -50% stop, 60-minute max hold, no take profit.\n")

    coin = [c for c in ok if c[0] == "coin"]
    if coin:
        L += ["## Control: coin-flip entries (should LOSE about the cost of trading)\n",
              f"- Average over all {len(coin)} coin settings: tune ${np.mean([tune[c]['usd_wk'] for c in coin]):+,.0f}/week, "
              f"check ${np.mean([check[c]['usd_wk'] for c in coin]):+,.0f}/week; profitable on both: "
              f"{sum(1 for c in coin if tune[c]['usd_wk'] > 0 and check[c]['usd_wk'] > 0)} of {len(coin)}.",
              "- If the coin looks profitable, the backtest is too kind and the results below are not trustworthy.\n"]

    def compare(title, idx, values, fmt=str):
        out = [f"## {title} (average over every other setting, real setups only)\n",
               "| choice | settings | tune avg $/week | tune share profitable | check avg $/week | check share profitable |",
               "|---|---|---|---|---|---|"]
        for v in values:
            cs = [c for c in real if c[idx] == v]
            if cs:
                out.append(f"| {fmt(v)} | {len(cs)} | ${np.mean([tune[c]['usd_wk'] for c in cs]):+,.0f} "
                           f"| {np.mean([tune[c]['usd_wk'] > 0 for c in cs]):.0%} "
                           f"| ${np.mean([check[c]['usd_wk'] for c in cs]):+,.0f} "
                           f"| {np.mean([check[c]['usd_wk'] > 0 for c in cs]):.0%} |")
        return out + [""]

    L += compare("Setup", 0, lib.SETUPS)
    L += compare("Sides", 1, MODES)
    L += compare("How bought", 2, STRUCTS)
    L += compare("Take profit", 3, TPS, lambda v: "ride" if v is None else f"+{v:.0%}")
    L += compare("Stop", 4, SLS, lambda v: f"-{v:.0%}")
    L += compare("Max hold", 5, HOLDS, lambda v: f"{v} min")
    L += compare("Big-move grade", 6, GRADES, lambda v: f"{v}+")
    L += ["## Top 15 on tune, and the same settings on the unseen CHECK year\n", HEAD, SEP]
    for c in ranked[:15]:
        L.append(line(c, tune[c]))
        L.append(line(c, check[c], "check: ", bold=True))

    # combined: best settings per setup (tune), first qualifying signal of the day wins
    best = {}
    for c in ranked:
        best.setdefault(c[0], c)
    keep = [c for c in best.values() if tune[c]["per"] > 0 and tune[c]["lo"] > -1e8]
    L.append("\n## Combined model: best settings per setup (chosen on tune), first signal of the day wins\n")
    if keep:
        trades = [(r, c) for c in keep for r in rows[c[:6]] if grade(r[2]) >= c[6]]
        byday: dict = {}
        for r, c in sorted(trades, key=lambda x: (x[0][0], x[0][4])):
            byday.setdefault(r[0], (r, c))
        combo = [byday[d] for d in sorted(byday)]
        ct = stats([r for r, _ in combo if r[0] in tset], len(tune_days))
        cc = stats([r for r, _ in combo if r[0] in cset], len(check_days))
        L.append("Uses: " + "; ".join(f"`{cname(c)}`" for c in keep) + "\n")
        lab = "combined | mixed | mixed | mixed | mixed | mixed | mixed"
        L += [HEAD, SEP, line(None, ct, label=lab), line(None, cc, "check: ", bold=True, label=lab)]
        mix: dict = {}
        for r, c in combo:
            mix[c[0]] = mix.get(c[0], 0) + 1
        L.append("\nWho took the day: " + ", ".join(f"{k} {v}" for k, v in sorted(mix.items(), key=lambda x: -x[1])))
        by_m: dict = {}
        for r, _ in combo:
            by_m.setdefault(f"{r[0]:%Y-%m}", []).append(r[1])
        L += ["", "| month | trades | $ |", "|---|---|---|"]
        L += [f"| {m} | {len(v)} | ${sum(v):+,.0f} |" for m, v in sorted(by_m.items())]
        L += ["\n## Bottom line\n",
              f"- Combined, tune: {ct['wk']:.2f} trades/week, win {ct['win']:.0%}, ${ct['per']:+,.0f}/trade, "
              f"**${ct['usd_wk']:+,.0f}/week**.",
              f"- Combined, unseen check year: {cc['n']} trades ({cc['wk']:.2f}/week), win {cc['win']:.0%}, "
              f"${cc['per']:+,.0f}/trade (median ${cc['med']:+,.0f}), **${cc['usd_wk']:+,.0f}/week**, total "
              f"${cc['total']:+,.0f} (${cc['ex5']:+,.0f} without its best 5 days), worst drawdown ${cc['dd']:,.0f}.",
              f"- Real-setup settings profitable on BOTH tune and check: "
              f"{sum(1 for c in real if tune[c]['usd_wk'] > 0 and check[c]['usd_wk'] > 0)} of {len(real)} "
              f"(coin control: {sum(1 for c in coin if tune[c]['usd_wk'] > 0 and check[c]['usd_wk'] > 0)} of {len(coin)})."]
    else:
        L.append("No setup was profitable on tune. Nothing to combine.")
    out = ROOT / "reports" / f"morning_{datetime.now():%Y-%m-%d}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[-5:]))
    print(f"\nreport written: {out}\nNext: git add reports; git commit -m \"morning study\"; git pull --rebase; git push")


if __name__ == "__main__":
    main()
