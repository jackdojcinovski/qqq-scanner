"""Full research run: scan history once, test every variant, rank on the tune
period, confirm the top few on the untouched check period, write a report.

    python tools/research.py                               # QQQ+SPY, check = last 12 months
    python tools/research.py --symbols QQQ --split 2025-10-27
    python tools/research.py --top 5 --min-filled 80

Output: reports/research_<date>.md (+ a CSV of every variant's tune metrics).
Commit and push the reports folder so it can be reviewed:
    git add reports; git commit -m "research results"; git push
"""
from __future__ import annotations

import argparse
import os
import sys
import time as clock
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scanner import backtest as bt  # noqa: E402
from scanner import research as rs  # noqa: E402
from scanner.config import ROOT, load_config  # noqa: E402
from scanner.history import load_history, trading_days  # noqa: E402

COLS = ("| # | variant | alerts/wk | filled | fill | win | avg R | net R | 90% CI net | PF net "
        "| max DD | streak | years + | worst yr |")
SEP = "|" + "---|" * 14


def row(i, name, m: rs.Metrics) -> str:
    pf = "inf" if m.profit_factor == float("inf") else f"{m.profit_factor:.2f}"
    return (f"| {i} | `{name}` | {m.per_week:.2f} | {m.filled} | {m.fill_rate:.0%} | {m.win_rate:.0%} "
            f"| {m.avg_r:+.2f} | {m.avg_net:+.2f} | {m.ci_lo:+.2f} to {m.ci_hi:+.2f} | {pf} "
            f"| {m.max_dd:.1f}R | {m.worst_streak} | {m.years_positive} | {m.min_year:+.2f} |")


def detail(lines: list[str], title: str, recs: list[bt.TradeRecord], cost: rs.CostModel) -> None:
    taken = [r for r in recs if r.taken]
    rows = rs._to_rows(recs)
    lines.append(f"\n#### {title}\n")
    if not taken:
        lines.append("No trades.")
        return

    def grp(fn):
        g: dict = {}
        for rr in rows:
            g.setdefault(fn(rr), []).append(rr)
        return g

    lines.append("| group | value | alerts | filled | win (net) | avg R | net R |")
    lines.append("|---|---|---|---|---|---|---|")
    for gname, fn in [("year", lambda r: r.day.year), ("symbol", lambda r: r.symbol),
                      ("direction", lambda r: r.direction), ("trigger", lambda r: r.trigger),
                      ("hour", lambda r: f"{r.hour}:00"), ("outcome", lambda r: r.outcome)]:
        for k, v in sorted(grp(fn).items(), key=lambda kv: str(kv[0])):
            f = [x for x in v if x.filled]
            net = [x.r - cost.cost_r(x) for x in f]
            win = sum(1 for x in net if x > 0) / len(net) if net else 0
            avg = sum(x.r for x in f) / len(f) if f else 0
            avgn = sum(net) / len(net) if net else 0
            lines.append(f"| {gname} | {k} | {len(v)} | {len(f)} | {win:.0%} | {avg:+.2f} | {avgn:+.2f} |")
    holds = [x.hold_min for x in rows if x.filled]
    risks = [x.risk for x in rows if x.filled]
    if holds:
        lines.append(f"\nMedian hold {sorted(holds)[len(holds) // 2]:.0f} min, "
                     f"median stop distance ${sorted(risks)[len(risks) // 2]:.2f}.")


def yaml_overrides(name: str) -> str:
    v = rs.variant(name)
    out = []

    def walk(d, indent=0):
        for k, val in d.items():
            if isinstance(val, dict):
                out.append(" " * indent + f"{k}:")
                walk(val, indent + 2)
            else:
                out.append(" " * indent + f"{k}: {val}")
    walk(v.overrides)
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", default="QQQ,SPY")
    p.add_argument("--split", help="check period starts here (default: 12 months before the last day)")
    p.add_argument("--top", type=int, default=3, help="how many tune winners to confirm on check")
    p.add_argument("--min-filled", type=int, default=100, help="min filled tune trades to be ranked")
    p.add_argument("--scan-workers", type=int, default=None, help="default: CPU count - 1")
    p.add_argument("--grid-workers", type=int, default=None, help="default: min(CPU - 1, 6), ~350MB RAM each")
    p.add_argument("--spread", type=float, default=0.03)
    p.add_argument("--theta", type=float, default=0.0003, help="option decay per hour, fraction of share price")
    p.add_argument("--delta", type=float, default=0.45)
    args = p.parse_args()

    cfg = load_config()
    cost = rs.CostModel(args.spread, args.theta, args.delta)
    cpu = os.cpu_count() or 2
    scan_workers = args.scan_workers or max(1, cpu - 1)
    grid_workers = args.grid_workers or max(1, min(cpu - 1, 6))
    t_all = clock.time()

    def say(msg):
        print(f"[{clock.time() - t_all:6.0f}s] {msg}", flush=True)

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    bars = {}
    for s in symbols:
        b = load_history(s, date(2015, 1, 1), date.today())
        if b.empty:
            raise SystemExit(f"No history for {s}. Run: python tools/download_history.py "
                             f"--symbols {args.symbols} --start 2019-01-01")
        bars[s] = b
        say(f"{s}: {len(trading_days(b))} days, {b.index[0].date()} to {b.index[-1].date()}")

    all_days = sorted(set().union(*(set(trading_days(b)) for b in bars.values())))
    split = date.fromisoformat(args.split) if args.split else all_days[-1] - timedelta(days=365)
    tune = {d for d in all_days if d < split}
    check = {d for d in all_days if d >= split}
    say(f"tune {min(tune)} to {max(tune)} ({len(tune)} days), check {min(check)} to {max(check)} ({len(check)} days)")

    say("scanning history (first run is slow, later runs reuse the cache)")
    paths = rs.build_cache(bars, cfg, scan_workers, progress=say)

    variants = rs.grid()
    say(f"running {len(variants)} variants on {len(paths)} chunks with {grid_workers} workers")
    results = rs.run_grid(paths, cfg, variants, grid_workers, progress=say)

    models = {
        "options 1-2 DTE": cost,
        "options weekly": rs.CostModel(0.05, 0.00011, 0.50),
        "shares": rs.CostModel(0.01, 0.0, 1.0),
    }

    # ---------------------------------------------------------------- report
    today = datetime.now().strftime("%Y-%m-%d")
    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    L: list[str] = [f"# Research run {today}\n"]
    L.append(f"- Symbols: {', '.join(symbols)}; tune {min(tune)} to {max(tune)} ({len(tune)} days); "
             f"check {min(check)} to {max(check)} ({len(check)} days)")
    L.append(f"- {len(variants)} variants; ranked by the 90% lower bound of net R per trade; "
             f"needs at least {args.min_filled} filled tune trades to be ranked; alerts/wk is per symbol")
    L.append("- Ranked separately under three ways of trading the signal:")
    for name, cm in models.items():
        L.append(f"  - **{name}**: ${cm.spread:.2f} spread + {cm.theta_pct_per_hour:.3%} of price per hour held, "
                 f"delta {cm.delta}")
    L.append(f"- Config signature {rs.signature(cfg)}; runtime {clock.time() - t_all:.0f}s\n")

    summary = []
    csv_ranked = None
    for mname, cm in models.items():
        ranked = rs.rank(results, tune, args.min_filled, cm)
        csv_ranked = csv_ranked or ranked
        base_tune = next(m for n, m in ranked if n == rs.BASELINE)
        top = [n for n, m in ranked if m.filled >= args.min_filled][:args.top]
        marg = rs.marginal(ranked, args.min_filled)

        L.append(f"\n# Trading it with {mname}\n")
        L.append("## Tune period: top 10\n")
        L += [COLS, SEP]
        for i, (n, m) in enumerate(ranked[:10], 1):
            L.append(row(i, n, m))
        L.append(row("base", rs.BASELINE, base_tune))

        L.append("\n## Which choices help (median net R across ranked variants)\n")
        width = max(len(v) for v in marg.values())
        L.append("| setting | " + " | ".join(["option: median net R"] * width) + " |")
        L.append("|---|" + "---|" * width)
        for dim, vals in marg.items():
            cells = [f"{k}: {v:+.3f}" for k, v in vals.items()]
            cells += [""] * (width - len(cells))
            L.append(f"| {dim} | " + " | ".join(cells) + " |")

        L.append("\n## Check period (untouched): top picks and baseline\n")
        L += [COLS, SEP]
        for i, n in enumerate(top + [rs.BASELINE], 1):
            m = rs.metrics(rs.split_rows(results[n], check), len(check) * len(symbols), cm)
            L.append(row(i if n != rs.BASELINE else "base", n, m))
        if top:
            mc = rs.metrics(rs.split_rows(results[top[0]], check), len(check) * len(symbols), cm)
            if mc.filled < 30:
                verdict = f"only {mc.filled} check trades, too few to confirm"
            elif mc.ci_lo > 0:
                verdict = "CONFIRMED: positive after costs, 90% lower bound above zero"
            elif mc.avg_net > 0:
                verdict = "positive on average but not conclusive"
            else:
                verdict = "NOT confirmed: zero or negative after costs"
            L.append(f"\nVerdict for #1 `{top[0]}`: {verdict}.")
            summary.append((mname, top[0], mc, verdict))

    L.append("\n# Bottom line\n")
    L.append("| trading with | best tune variant | check filled | check net R | check 90% CI | verdict |")
    L.append("|---|---|---|---|---|---|")
    for mname, best, mc, verdict in summary:
        L.append(f"| {mname} | `{best}` | {mc.filled} | {mc.avg_net:+.2f} | {mc.ci_lo:+.2f} to {mc.ci_hi:+.2f} "
                 f"| {verdict} |")

    if summary:
        mname, best, _, _ = max(summary, key=lambda s: s[2].ci_lo if s[2].filled >= 30 else -99)
        cm = models[mname]
        L.append(f"\n## Detail for `{best}` ({mname})")
        detail(L, "Tune period", rs.full_records(paths, cfg, best, tune), cm)
        detail(L, "Check period", rs.full_records(paths, cfg, best, check), cm)
        L.append(f"\nConfig changes for `{best}`:\n\n```yaml\n{yaml_overrides(best)}\n```")

    ranked = csv_ranked
    md = out_dir / f"research_{today}.md"
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    csv = out_dir / f"research_{today}_tune.csv"
    with open(csv, "w", encoding="utf-8") as f:
        f.write("variant,filled,per_week,fill_rate,win_rate,avg_r,avg_net,ci_lo,ci_hi,pf,max_dd,streak,years_pos,min_year\n")
        for n, m in ranked:
            f.write(f"{n},{m.filled},{m.per_week:.3f},{m.fill_rate:.3f},{m.win_rate:.3f},{m.avg_r:.4f},"
                    f"{m.avg_net:.4f},{m.ci_lo:.4f},{m.ci_hi:.4f},{m.profit_factor:.3f},{m.max_dd:.2f},"
                    f"{m.worst_streak},{m.years_positive},{m.min_year:.4f}\n")
    say(f"report written: {md}")
    print("\nNext: git add reports; git commit -m \"research results\"; git push")


if __name__ == "__main__":
    main()
