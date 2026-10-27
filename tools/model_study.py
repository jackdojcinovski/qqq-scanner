"""Test every setup in the library the way Jack trades, then build the combined model.

    python tools/model_study.py
    python tools/model_study.py --budget 1000 --max-per-day 1

Jack's format: about $1,000 per trade in QQQ / SPY 0DTE (or 1DTE) at-the-money
calls (puts for shorts), take profit on the premium (+50% / +100% / +150% /
hold), stop on the premium (-25% / -50% / none), otherwise sell at 15:59.
USD account: no FX fee, $0 per contract; costs = the bid/ask spread.

1. Every setup x side rule x contract x take profit x stop, on the TUNE years.
2. The combined model: each setup that was profitable on tune (with its best
   settings), first signal of the day wins, at most --max-per-day trades.
3. The combined model on the unseen CHECK year (last 12 months), once.

Option prices are modelled (Black-Scholes, implied vol = 20-day realised x
--iv-mult; 1.6 matched real 0DTE prices). Confirm with tools/real_options.py.
Writes reports/model_<date>.md.
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
from scanner import library as lib  # noqa: E402
from scanner import optionsim as om  # noqa: E402
from scanner.config import ROOT  # noqa: E402
from scanner.history import load_history, trading_days  # noqa: E402

TPS = [0.5, 1.0, 1.5, None]
SLS = [0.25, 0.5, None]
MODES = {"both": "calls and puts every day", "long": "calls only",
         "regime_k1": "calls; puts only on down days"}
CONTRACTS = [om.Contract("0DTE ATM", 0), om.Contract("1DTE ATM", 1)]


def n_contracts(paid: float, budget: float) -> int:
    return max(1, int(budget // (paid * 100)))


def pnl(r: om.Result, budget: float, fx: float) -> float:
    return n_contracts(r.paid, budget) * 100 * ((r.got - r.paid) - fx * (r.paid + r.got))


def summary(res: list[om.Result], n_days: int, budget: float, fx: float) -> dict:
    weeks = max(n_days / 5, 1)
    p = np.array([pnl(r, budget, fx) for r in res])
    if len(p) == 0:
        return dict(n=0, wk=0.0, win=0.0, aw=0.0, al=0.0, per=0.0, usd_wk=0.0, total=0.0, lo=-1e9, dd=0.0,
                    streak=0, years="0/0", by_year={}, spent=0.0, tgt=0.0)
    eq = np.concatenate([[0], np.cumsum(p)])
    streak = worst = 0
    for x in p:
        streak = streak + 1 if x < 0 else 0
        worst = max(worst, streak)
    by_year: dict[int, float] = {}
    for r, x in zip(res, p):
        by_year[r.signal.day.year] = by_year.get(r.signal.day.year, 0.0) + x
    lo = bt.bootstrap_ci(list(p))[0] if len(p) > 1 else -1e9
    return dict(n=len(p), wk=len(p) / weeks, win=float((p > 0).mean()),
                aw=float(p[p > 0].mean()) if (p > 0).any() else 0.0,
                al=float(p[p <= 0].mean()) if (p <= 0).any() else 0.0,
                per=float(p.mean()), usd_wk=float(p.sum() / weeks), total=float(p.sum()), lo=lo,
                dd=float((np.maximum.accumulate(eq) - eq).max()), streak=worst,
                years=f"{sum(v > 0 for v in by_year.values())}/{len(by_year)}", by_year=by_year,
                spent=float(np.mean([n_contracts(r.paid, budget) * r.paid * 100 for r in res])),
                tgt=float(np.mean([r.outcome == "target" for r in res])))


def fmt_x(x):
    return "hold" if x is None else f"+{x:.0%}"


def fmt_sl(x):
    return "none" if x is None else f"-{x:.0%}"


HEAD = ("| setup | sides | contract | take profit | stop | trades | per week | win | avg win | avg loss "
        "| $ per trade | **$ per week** | total | worst drawdown | losing streak | years + |")
SEP = "|" + "---|" * 16


def row(label, mode, cname, tp, sl, s, bold=False) -> str:
    b = "**" if bold else ""
    return (f"| {b}{label}{b} | {mode} | {cname} | {fmt_x(tp)} | {fmt_sl(sl)} | {s['n']} | {s['wk']:.2f} | {s['win']:.0%} "
            f"| ${s['aw']:+,.0f} | ${s['al']:+,.0f} | ${s['per']:+,.0f} | **${s['usd_wk']:+,.0f}** | ${s['total']:+,.0f} "
            f"| ${s['dd']:,.0f} | {s['streak']} | {s['years']} |")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", default="QQQ,SPY")
    p.add_argument("--budget", type=float, default=1000.0, help="$ per trade")
    p.add_argument("--fx", type=float, default=0.0, help="FX fee each way (0 = USD account)")
    p.add_argument("--iv-mult", type=float, default=1.6)
    p.add_argument("--max-per-day", type=int, default=1)
    p.add_argument("--min-trades", type=int, default=100, help="min tune trades for a setup to count")
    p.add_argument("--split", help="check period start (default: 12 months before the last day)")
    args = p.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    bars = {s: load_history(s, date(2015, 1, 1), date.today()) for s in symbols}
    for s, b in bars.items():
        if b.empty:
            raise SystemExit(f"No history for {s}. Run tools/download_history.py first.")
    all_days = sorted(set().union(*(set(trading_days(b)) for b in bars.values())))
    split = date.fromisoformat(args.split) if args.split else all_days[-1] - timedelta(days=365)
    tune = {d for d in all_days if d < split}
    check = {d for d in all_days if d >= split}

    print("building grids, regime and signals ...")
    grids = {s: ix.build_grid(s, b) for s, b in bars.items()}
    ctx = {s: om.daily_context(g) for s, g in grids.items()}
    sigs = lib.all_signals(grids, ctx)
    for k, v in sigs.items():
        print(f"  {k}: {len(v)} signals")

    print("pricing option paths ...")
    book = {}
    for k, v in sigs.items():
        for c in CONTRACTS:
            book[(k, c.name)] = {(s.symbol, s.day, s.m_in, s.direction): (s, pth)
                                 for s, pth in om.build_paths(v, grids, ctx, c, args.iv_mult)}

    def chosen(setup, mode):
        v = sigs[setup] if mode == "both" else om.apply_regime(sigs[setup], ctx, mode)
        return om.one_per_day(v)

    print("replaying every setup / side rule / contract / take profit / stop ...")
    res = {}
    for setup in lib.SETUPS:
        for mode in MODES:
            ch = chosen(setup, mode)
            for c in CONTRACTS:
                b = book[(setup, c.name)]
                pp = [b[k] for k in ((s.symbol, s.day, s.m_in, s.direction) for s in ch) if k in b]
                for tp in TPS:
                    for sl in SLS:
                        out = om.run(pp, c.name, tp, sl)
                        res[(setup, mode, c.name, tp, sl)] = (
                            summary([r for r in out if r.signal.day in tune], len(tune), args.budget, args.fx),
                            summary([r for r in out if r.signal.day in check], len(check), args.budget, args.fx),
                            out)

    def score(k):
        t = res[k][0]
        return t["lo"] * t["wk"] if t["n"] >= args.min_trades else -1e12

    # best config per setup (chosen on tune only)
    best = {}
    for k in sorted(res, key=score, reverse=True):
        best.setdefault(k[0], k)
    order = sorted(best.values(), key=score, reverse=True)

    L = [f"# Combined model study {datetime.now():%Y-%m-%d}\n",
         f"- About **${args.budget:,.0f} per trade** (as many contracts as that buys), QQQ and SPY, "
         f"FX {args.fx:.1%} each way, $0 per contract, bid/ask spread included.",
         f"- Tune {min(tune)} to {max(tune)} ({len(tune)} days); check {min(check)} to {max(check)} "
         f"({len(check)} days, never used to choose anything).",
         f"- Option prices modelled with implied vol = realised x {args.iv_mult:g} (matched real 0DTE prices). "
         "Take profit / stop are on the premium; otherwise the option is sold at 15:59.",
         "- Each setup: at most one trade a day across QQQ and SPY. No entries after 14:30.\n",
         "## Each setup with its best settings (chosen on tune), and how it did on the unseen year\n",
         HEAD, SEP]
    for k in order:
        t, c, _ = res[k]
        L.append(row(k[0], k[1], k[2], k[3], k[4], t))
        L.append(row(f"check: {k[0]}", k[1], k[2], k[3], k[4], c, bold=True))

    keep = [k for k in order if res[k][0]["n"] >= args.min_trades and res[k][0]["per"] > 0
            and int(res[k][0]["years"].split("/")[0]) >= max(1, int(res[k][0]["years"].split("/")[1]) - 2)]
    L.append("\n## The combined model\n")
    if not keep:
        L.append("No setup was profitable on the tune years with these option costs. Nothing to combine.")
        combo_t = combo_c = None
    else:
        rank = {k[0]: n for n, k in enumerate(keep)}
        trades = [r for k in keep for r in res[k][2]]
        by_day: dict = {}
        for r in sorted(trades, key=lambda r: (r.signal.day, r.signal.m_in, rank[r.signal.strategy])):
            lst = by_day.setdefault(r.signal.day, [])
            if len(lst) < args.max_per_day and all(x.exit_bar < r.signal.m_in for x in lst):
                lst.append(r)
        combo = [r for d in sorted(by_day) for r in by_day[d]]
        combo_t = summary([r for r in combo if r.signal.day in tune], len(tune), args.budget, args.fx)
        combo_c = summary([r for r in combo if r.signal.day in check], len(check), args.budget, args.fx)
        L.append("Setups used (each with its own settings above): " + ", ".join(
            f"`{k[0]}` ({k[1]}, {k[2]}, TP {fmt_x(k[3])}, stop {fmt_sl(k[4])})" for k in keep))
        L.append(f"\nRule: first signal of the day wins; at most {args.max_per_day} trade(s) a day, never two at once.\n")
        L += [HEAD, SEP, row("combined: tune", "mixed", "mixed", None, None, combo_t),
              row("combined: CHECK (unseen)", "mixed", "mixed", None, None, combo_c, bold=True)]
        L += ["", "| year | trades | total $ |", "|---|---|---|"]
        yrs = {}
        for r in combo:
            n, tot = yrs.get(r.signal.day.year, (0, 0.0))
            yrs[r.signal.day.year] = (n + 1, tot + pnl(r, args.budget, args.fx))
        for y, (n, tot) in sorted(yrs.items()):
            L.append(f"| {y} | {n} | ${tot:+,.0f} |")
        import csv
        cfg_of = {k[0]: k for k in keep}
        with open(ROOT / "reports" / "model_trades.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["day", "symbol", "setup", "direction", "m_in", "dte", "tp", "sl", "model_paid", "model_got",
                        "model_pnl", "period"])
            for r in combo:
                k = cfg_of[r.signal.strategy]
                dte = next(c.dte for c in CONTRACTS if c.name == k[2])
                w.writerow([r.signal.day, r.signal.symbol, r.signal.strategy, r.signal.direction, r.signal.m_in, dte,
                            "" if k[3] is None else k[3], "" if k[4] is None else k[4], f"{r.paid:.4f}", f"{r.got:.4f}",
                            f"{pnl(r, args.budget, args.fx):.2f}", "check" if r.signal.day in check else "tune"])
        mix = {}
        for r in combo:
            mix[r.signal.strategy] = mix.get(r.signal.strategy, 0) + 1
        L.append("\nWho took the day: " + ", ".join(f"{k} {v}" for k, v in sorted(mix.items(), key=lambda x: -x[1])))

    L += ["\n## Take-profit / stop map (tune $ per week, best side rule + contract per setup)\n",
          "| setup | " + " | ".join(f"TP {fmt_x(tp)} / stop {fmt_sl(sl)}" for tp in TPS for sl in SLS) + " |",
          "|---|" + "---|" * (len(TPS) * len(SLS))]
    for k in order:
        cells = [f"${res[(k[0], k[1], k[2], tp, sl)][0]['usd_wk']:+,.0f}" for tp in TPS for sl in SLS]
        L.append(f"| {k[0]} ({k[1]}, {k[2]}) | " + " | ".join(cells) + " |")

    L.append("\n## Bottom line\n")
    if combo_t:
        L.append(f"- Combined model, tune: {combo_t['wk']:.2f} trades/week, win {combo_t['win']:.0%}, "
                 f"${combo_t['per']:+,.0f}/trade, **${combo_t['usd_wk']:+,.0f}/week**.")
        L.append(f"- Combined model, unseen year: {combo_c['n']} trades ({combo_c['wk']:.2f}/week), win {combo_c['win']:.0%}, "
                 f"${combo_c['per']:+,.0f}/trade, **${combo_c['usd_wk']:+,.0f}/week**, total ${combo_c['total']:+,.0f}, "
                 f"worst drawdown ${combo_c['dd']:,.0f}.")
        L.append("- " + ("Held up on the unseen year." if combo_c["usd_wk"] > 0 else "Did NOT hold up on the unseen year."))
    out = ROOT / "reports" / f"model_{datetime.now():%Y-%m-%d}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[-4:]))
    print(f"\nreport written: {out}\nNext: git add reports; git commit -m \"model study\"; git pull --rebase; git push")


if __name__ == "__main__":
    main()
