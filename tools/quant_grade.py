"""Grade the opening-range trades with quantitative signs and keep only the best.

    python tools/quant_grade.py
    python tools/quant_grade.py --top 0.25 --variant "mid|with|long|classic|fixedeod|checklist"

Steps (all choices made on the TUNE period only):
  1. Take every trade the variant would have sent (QQQ and SPY).
  2. Check 8 quantitative signs known by 9:35 (see scanner/quant.py).
  3. Keep the signs that raised average R on the tune period.
  4. Grade = how many kept signs a trade has. Trade only grades in the top
     --top fraction, at most one trade a day across both symbols.
  5. Run the same rules once on the untouched CHECK period.

Writes reports/quant_grade_<date>.md. Then: git add reports; git commit; git push
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scanner import quant as q  # noqa: E402
from scanner import research as rs  # noqa: E402
from scanner.config import ROOT, load_config  # noqa: E402
from scanner.history import load_history, trading_days  # noqa: E402

DEFAULT_VARIANT = "mid|with|long|classic|fixedeod|checklist"

COST_MODELS = {
    "shares": rs.CostModel(0.01, 0.0, 1.0),
    "TQQQ / 3x ETF": rs.CostModel(0.02, 0.0, 1.0),
    "deep ITM monthly call": rs.CostModel(0.10, 0.00004, 0.80),
    "options weekly": rs.CostModel(0.05, 0.00011, 0.50),
    "options 1-2 DTE": rs.CostModel(0.03, 0.0003, 0.45),
}

HEAD = "| set | trades | per week | win | avg R | net R | 90% CI net | PF net | max DD | streak | years + |"
SEP = "|---|---|---|---|---|---|---|---|---|---|---|"


def line(label: str, rows: list, n_days: int, cm: rs.CostModel) -> str:
    m = rs.metrics(rows, n_days, cm)
    pf = "inf" if m.profit_factor == float("inf") else f"{m.profit_factor:.2f}"
    return (f"| {label} | {m.filled} | {m.per_week:.2f} | {m.win_rate:.0%} | {m.avg_r:+.2f} | {m.avg_net:+.2f} "
            f"| {m.ci_lo:+.2f} to {m.ci_hi:+.2f} | {pf} | {m.max_dd:.1f}R | {m.worst_streak} | {m.years_positive} |")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", default=DEFAULT_VARIANT)
    p.add_argument("--symbols", default="QQQ,SPY")
    p.add_argument("--split", help="check period start (default: 12 months before the last day)")
    p.add_argument("--top", type=float, default=0.25, help="fraction of tune trades to keep")
    p.add_argument("--min-edge", type=float, default=0.05, help="min R improvement for a sign to count")
    p.add_argument("--workers", type=int, default=None)
    args = p.parse_args()

    cfg = load_config()
    rs.variant(args.variant)
    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    bars = {s: load_history(s, date(2015, 1, 1), date.today()) for s in symbols}
    for s, b in bars.items():
        if b.empty:
            raise SystemExit(f"No history for {s}. Run tools/download_history.py first.")
    all_days = sorted(set().union(*(set(trading_days(b)) for b in bars.values())))
    split = date.fromisoformat(args.split) if args.split else all_days[-1] - timedelta(days=365)
    tune = {d for d in all_days if d < split}
    check = {d for d in all_days if d >= split}

    print("loading scan cache (reuses research.py's cache) ...")
    paths = rs.build_cache(bars, cfg, args.workers or max(1, (os.cpu_count() or 2) - 1))
    res = rs.run_grid(paths, cfg, [rs.variant(args.variant)], workers=1)
    rows = res[args.variant]
    tables = {s: q.daily_table(b) for s, b in bars.items()}
    graded = q.attach_signs(rows, tables)
    g_tune = [t for t in graded if t.day in tune]
    g_check = [t for t in graded if t.day in check]

    edges = q.sign_edges(g_tune)
    used = q.choose_signs(edges, min_edge=args.min_edge)
    cutoff = q.cutoff_for_top(g_tune, used, args.top) if used else 0
    pick_t = q.one_per_day(g_tune, used, cutoff)
    pick_c = q.one_per_day(g_check, used, cutoff)
    all_t = q.one_per_day(g_tune, [], 0)
    all_c = q.one_per_day(g_check, [], 0)

    L = [f"# Quant grade {datetime.now():%Y-%m-%d}\n",
         f"- Variant `{args.variant}` on {', '.join(symbols)}; tune {min(tune)} to {max(tune)}, "
         f"check {min(check)} to {max(check)}",
         f"- Keep the top {args.top:.0%} of tune trades by grade, at most one trade a day across symbols\n",
         "## Signs on the tune period (R before costs)\n",
         "| sign | with: trades | with: win | with: avg R | without: trades | without: win | without: avg R | edge | used |",
         "|---|---|---|---|---|---|---|---|---|"]
    for e in sorted(edges, key=lambda e: e.edge, reverse=True):
        L.append(f"| {q.SIGN_TEXT[e.sign]} | {e.n_hit} | {e.win_hit:.0%} | {e.r_hit:+.2f} | {e.n_miss} | "
                 f"{e.win_miss:.0%} | {e.r_miss:+.2f} | {e.edge:+.2f} | {'yes' if e.sign in used else ''} |")
    L.append(f"\nGrade = number of the {len(used)} used signs present. Trade when grade >= **{cutoff}**.\n")

    L.append("## Results by grade (shares costs)\n")
    L.append("| grade | tune trades | tune win | tune net R | check trades | check win | check net R |")
    L.append("|---|---|---|---|---|---|---|")
    cm0 = COST_MODELS["shares"]
    for g in range(len(used), -1, -1):
        rt = [t.row for t in g_tune if t.filled and t.grade(used) == g]
        rc = [t.row for t in g_check if t.filled and t.grade(used) == g]
        mt, mc = rs.metrics(rt, len(tune), cm0), rs.metrics(rc, len(check), cm0)
        L.append(f"| {g} | {mt.filled} | {mt.win_rate:.0%} | {mt.avg_net:+.2f} | {mc.filled} | "
                 f"{mc.win_rate:.0%} | {mc.avg_net:+.2f} |")

    for name, cm in COST_MODELS.items():
        L += [f"\n## Trading it with {name}\n", HEAD, SEP]
        L.append(line("tune: every trade, 1/day", [t.row for t in all_t], len(tune), cm))
        L.append(line(f"tune: grade >= {cutoff}", [t.row for t in pick_t], len(tune), cm))
        L.append(line("**check: every trade, 1/day**", [t.row for t in all_c], len(check), cm))
        L.append(line(f"**check: grade >= {cutoff}**", [t.row for t in pick_c], len(check), cm))

    mc = rs.metrics([t.row for t in pick_c], len(check), COST_MODELS["shares"])
    mall = rs.metrics([t.row for t in all_c], len(check), COST_MODELS["shares"])
    L.append("\n## Verdict (shares)\n")
    if mc.filled < 30:
        L.append(f"- Only {mc.filled} graded check trades: too few to confirm.")
    else:
        better = mc.avg_net > mall.avg_net
        L.append(f"- Graded check trades: {mc.filled}, win {mc.win_rate:.0%}, net {mc.avg_net:+.2f}R "
                 f"(90% CI {mc.ci_lo:+.2f} to {mc.ci_hi:+.2f}) vs {mall.avg_net:+.2f}R ungraded.")
        L.append("- The grade " + ("IMPROVED" if better else "did NOT improve") + " results on unseen data.")
        if mc.ci_lo > 0:
            L.append("- Lower bound above zero: the edge is confirmed on the check period.")
    L.append(f"\nSigns used: {', '.join(q.SIGN_TEXT[s] for s in used) or 'none'}")

    out = ROOT / "reports" / f"quant_grade_{datetime.now():%Y-%m-%d}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[-6:]))
    print(f"\nreport written: {out}\nNext: git add reports; git commit -m \"quant grade\"; git push")


if __name__ == "__main__":
    main()
