"""Test the hold-for-hours quant strategies on QQQ and SPY, priced as options.

    python tools/strategy_study.py
    python tools/strategy_study.py --split 2025-10-27

Strategies (scanner/intraday.py): noise-boundary momentum, last-half-hour
momentum, VWAP trend; a few variants each. Every variant is limited to one
trade a day across QQQ and SPY (first signal wins).

Every choice is made on the TUNE period (all but the last 12 months). The
CHECK period is looked at once, for the shortlist only.

Writes reports/strategies_<date>.md. Then: git add reports; git commit; git push
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scanner import intraday as ix  # noqa: E402
from scanner import research as rs  # noqa: E402
from scanner.config import ROOT  # noqa: E402
from scanner.history import load_history, trading_days  # noqa: E402

COST = {
    "deep ITM monthly": rs.CostModel(0.10, 0.00004, 0.80),
    "weekly ATM": rs.CostModel(0.05, 0.00011, 0.50),
    "1-2 DTE ATM": rs.CostModel(0.03, 0.0003, 0.45),
    "shares": rs.CostModel(0.01, 0.0, 1.0),
}
MAIN = "deep ITM monthly"


def summary(trades: list[ix.Trade], n_days: int) -> dict:
    rows = [t.row() for t in trades]
    m = rs.metrics(rows, n_days, COST[MAIN])
    wins = [r.r for r in rows if r.r > 0]
    losses = [r.r for r in rows if r.r <= 0]
    usd = [ix.dollars(r, COST[MAIN]) for r in rows]
    weeks = max(n_days / 5, 1)
    eq = np.cumsum(usd) if usd else np.array([0.0])
    dd = float(np.max(np.maximum.accumulate(eq) - eq)) if usd else 0.0
    return dict(
        m=m, n=len(rows), per_week=len(rows) / weeks,
        win=m.win_rate, avg_win=float(np.mean(wins)) if wins else 0.0,
        avg_loss=float(np.mean(losses)) if losses else 0.0,
        gross=m.avg_r, net={k: rs.metrics(rows, n_days, cm).avg_net for k, cm in COST.items()},
        usd=float(np.mean(usd)) if usd else 0.0, usd_week=sum(usd) / weeks, usd_dd=dd,
        hold=float(np.mean([r.hold_min for r in rows])) if rows else 0.0,
    )


HEAD = ("| strategy | trades | per week | win | avg win | avg loss | gross R | net R deep ITM (90% CI) "
        "| net R weekly | net R 1-2 DTE | $/contract deep ITM | $/week (1 contract) | max DD $ | avg hold | years + |")
SEP = "|" + "---|" * 15


def line(label: str, s: dict) -> str:
    m = s["m"]
    return (f"| {label} | {s['n']} | {s['per_week']:.2f} | {s['win']:.0%} | {s['avg_win']:+.2f} | {s['avg_loss']:+.2f} "
            f"| {s['gross']:+.2f} | {s['net'][MAIN]:+.2f} ({m.ci_lo:+.2f} to {m.ci_hi:+.2f}) "
            f"| {s['net']['weekly ATM']:+.2f} | {s['net']['1-2 DTE ATM']:+.2f} | ${s['usd']:+.0f} "
            f"| ${s['usd_week']:+.0f} | ${s['usd_dd']:,.0f} | {s['hold'] / 60:.1f}h | {m.years_positive} |")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", default="QQQ,SPY")
    p.add_argument("--split", help="check period start (default: 12 months before the last day)")
    p.add_argument("--shortlist", type=int, default=6)
    args = p.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    bars = {s: load_history(s, date(2015, 1, 1), date.today()) for s in symbols}
    for s, b in bars.items():
        if b.empty:
            raise SystemExit(f"No history for {s}. Run tools/download_history.py first.")
    print("building day grids ...")
    grids = {s: ix.build_grid(s, b) for s, b in bars.items()}
    all_days = sorted(set().union(*(set(trading_days(b)) for b in bars.values())))
    split = date.fromisoformat(args.split) if args.split else all_days[-1] - timedelta(days=365)
    tune = {d for d in all_days if d < split}
    check = {d for d in all_days if d >= split}

    print("running strategies ...")
    res = {k: ix.one_per_day(v) for k, v in ix.run_all(grids).items()}
    t_sum = {k: summary([t for t in v if t.day in tune], len(tune)) for k, v in res.items()}

    L = [f"# Strategy study {datetime.now():%Y-%m-%d}\n",
         f"- Symbols: {', '.join(symbols)}; at most 1 trade a day across them (first signal wins)",
         f"- Tune {min(tune)} to {max(tune)} ({len(tune)} days); check {min(check)} to {max(check)} ({len(check)} days)",
         "- R = profit / distance to the stop at entry. $/contract = 100 x delta x move - option costs "
         "(deep ITM monthly: delta 0.8, $0.10 spread, small decay). Weekly and 1-2 DTE shown in R for comparison.",
         "- Strategies: **noise** = noise-boundary momentum (Zarattini et al. 2024), **lasthalf** = last-half-hour "
         "momentum (Gao et al. 2018), **vwap** = VWAP trend. See scanner/intraday.py.\n",
         "## Every variant, tune period (ranked by the 90% lower bound, deep ITM)\n", HEAD, SEP]
    ranked = sorted(t_sum, key=lambda k: t_sum[k]["m"].rank_key, reverse=True)
    for k in ranked:
        L.append(line(f"`{k}`", t_sum[k]))

    short = [k for k in ranked if t_sum[k]["n"] >= 60][: args.shortlist]
    L += ["\n## Shortlist: tune vs CHECK (unseen year)\n", HEAD, SEP]
    c_sum = {}
    for k in short:
        c_sum[k] = summary([t for t in res[k] if t.day in check], len(check))
        L.append(line(f"`{k}` tune", t_sum[k]))
        L.append(line(f"**`{k}` check**", c_sum[k]))

    # combine: best variant per strategy family that is positive on tune (lower bound > 0 preferred)
    fam: dict[str, str] = {}
    for k in ranked:
        f = k.split("|")[0]
        if f not in fam and t_sum[k]["net"][MAIN] > 0 and t_sum[k]["n"] >= 60:
            fam[f] = k
    combo_names = list(fam.values())
    L.append("\n## Combined: best of each strategy, 1 trade a day\n")
    if combo_names:
        combo = ix.one_per_day([t for k in combo_names for t in res[k]])
        ct = summary([t for t in combo if t.day in tune], len(tune))
        cc = summary([t for t in combo if t.day in check], len(check))
        L.append("Uses: " + ", ".join(f"`{k}`" for k in combo_names) + "\n")
        L += [HEAD, SEP, line("combined tune", ct), line("**combined check**", cc)]
        L += ["", "| year | trades | net R deep ITM | $ total, 1 contract |", "|---|---|---|---|"]
        for y in sorted({t.day.year for t in combo}):
            ys = [t for t in combo if t.day.year == y]
            L.append(f"| {y} | {len(ys)} | {np.mean([t.r - COST[MAIN].cost_r(t.row()) for t in ys]):+.2f} "
                     f"| ${sum(ix.dollars(t, COST[MAIN]) for t in ys):+,.0f} |")
        by_fam = {}
        for t in combo:
            by_fam.setdefault(t.strategy.split('|')[0], []).append(t)
        L.append("\nWhich strategy took the day: " + ", ".join(f"{f} {len(v)}" for f, v in by_fam.items()))
    else:
        ct = cc = None
        L.append("No variant was profitable after deep ITM option costs on the tune period.")

    L.append("\n## Bottom line\n")
    best = short[0] if short else None
    if best:
        b, bc = t_sum[best], c_sum[best]
        L.append(f"- Best single variant on tune: `{best}`: {b['per_week']:.2f}/week, win {b['win']:.0%}, "
                 f"{b['net'][MAIN]:+.2f}R net, about ${b['usd']:+.0f} per deep ITM contract.")
        L.append(f"- On the unseen check year: {bc['n']} trades, {bc['per_week']:.2f}/week, win {bc['win']:.0%}, "
                 f"{bc['net'][MAIN]:+.2f}R net (90% CI {bc['m'].ci_lo:+.2f} to {bc['m'].ci_hi:+.2f}), "
                 f"${bc['usd_week']:+.0f}/week on 1 contract.")
        L.append("- " + ("CONFIRMED on unseen data (lower bound above zero)." if bc["m"].ci_lo > 0 else
                         "Positive on unseen data but not yet statistically confirmed." if bc["net"][MAIN] > 0 else
                         "NOT confirmed on unseen data."))
    if cc:
        L.append(f"- Combined check: {cc['n']} trades ({cc['per_week']:.2f}/week), win {cc['win']:.0%}, "
                 f"{cc['net'][MAIN]:+.2f}R net, ${cc['usd_week']:+.0f}/week on 1 contract.")

    out = ROOT / "reports" / f"strategies_{datetime.now():%Y-%m-%d}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[-6:]))
    print(f"\nreport written: {out}\nNext: git add reports; git commit -m \"strategy study\"; git pull --rebase; git push")


if __name__ == "__main__":
    main()
