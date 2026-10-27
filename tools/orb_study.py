"""Deep study of the opening-range breakout (ORB) model for options trading.

    python tools/orb_study.py                 # QQQ and SPY only (the instruments we trade)
    python tools/orb_study.py --top 0.25

Answers, using the TUNE period for every choice and the CHECK period once:
  1. What do the losers have in common? (signs, noise stop-outs, time to stop)
  2. Does a wider stop raise the win rate without killing the winners?
  3. How does each symbol do?
  4. Grade every trade with the quant signs; keep the top fraction, at most
     --per-day trades a day across all symbols. How often, how good, after
     real-world option costs?

Writes reports/orb_study_<date>.md. Then: git add reports; git commit; git push
"""
from __future__ import annotations

import argparse
import statistics
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scanner import orb  # noqa: E402
from scanner import quant as q  # noqa: E402
from scanner import research as rs  # noqa: E402
from scanner.config import ROOT, load_config  # noqa: E402
from scanner.history import load_history, trading_days  # noqa: E402

DEFAULT_SYMBOLS = "QQQ,SPY"
COSTS = {   # name -> model; options first, the main way you'll trade it
    "deep ITM monthly": rs.CostModel(0.10, 0.00004, 0.80),
    "weekly ATM": rs.CostModel(0.05, 0.00011, 0.50),
    "1-2 DTE ATM": rs.CostModel(0.03, 0.0003, 0.45),
    "shares": rs.CostModel(0.01, 0.0, 1.0),
}
PRIMARY = "deep ITM monthly"


def peer_of(sym: str) -> str:
    return "SPY" if sym == "QQQ" else "QQQ"


def graded(trades: list[orb.OrbTrade], tables) -> list[q.GradedTrade]:
    out = []
    for t in trades:
        tab = tables.get(t.symbol)
        if tab is None or t.day not in tab.index:
            continue
        peer = tables.get(peer_of(t.symbol))
        peer_ret = peer.loc[t.day, "o5_ret"] if peer is not None and t.day in peer.index else None
        out.append(q.GradedTrade(t.symbol, t.day, t.direction, t.r, t.filled,
                                 q.signs_for(tab.loc[t.day], t.direction, peer_ret), t.row()))
    return out


def wl(rows) -> tuple[float, float]:
    wins = [r.r for r in rows if r.filled and r.r > 0]
    losses = [r.r for r in rows if r.filled and r.r <= 0]
    return (statistics.mean(wins) if wins else 0.0, statistics.mean(losses) if losses else 0.0)


def stats_cells(rows, n_days: int) -> str:
    """trades | per week | win | avg win | avg loss | gross R | net R per cost model (+CI for primary)"""
    base = rs.metrics(rows, n_days, COSTS["shares"])
    aw, al = wl(rows)
    cells = [f"{base.filled}", f"{base.per_week:.2f}", f"{sum(1 for r in rows if r.filled and r.r > 0) / max(base.filled, 1):.0%}",
             f"{aw:+.2f}", f"{al:+.2f}", f"{base.avg_r:+.2f}"]
    for name, cm in COSTS.items():
        m = rs.metrics(rows, n_days, cm)
        cells.append(f"{m.avg_net:+.2f}" + (f" ({m.ci_lo:+.2f} to {m.ci_hi:+.2f})" if name == PRIMARY else ""))
    return " | ".join(cells)


HEAD = ("| trades | per week | win | avg win | avg loss | gross R | " +
        " | ".join(f"net R {n}" + (" (90% CI)" if n == PRIMARY else "") for n in COSTS) + " |")


def table(label_head: str, rows_by_label, n_days: int) -> list[str]:
    """rows_by_label: (label, rows) or (label, rows, n_days) to override the day count."""
    out = [f"| {label_head} " + HEAD, "|---" * (2 + 6 + len(COSTS) - 1) + "|"]
    for item in rows_by_label:
        label, rows = item[0], item[1]
        out.append(f"| {label} | {stats_cells(rows, item[2] if len(item) > 2 else n_days)} |")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    p.add_argument("--split", help="check period start (default: 12 months before the last day)")
    p.add_argument("--scales", default="1,1.25,1.5,2", help="stop widths to test, x the opening range")
    p.add_argument("--top", type=float, default=0.25, help="fraction of tune trades to keep by grade")
    p.add_argument("--per-day", type=int, default=1)
    p.add_argument("--workers", type=int, default=None)
    args = p.parse_args()

    cfg = load_config()
    scales = tuple(float(x) for x in args.scales.split(","))
    want = [s.strip().upper() for s in args.symbols.split(",")]
    bars, missing = {}, []
    for s in want:
        b = load_history(s, date(2015, 1, 1), date.today())
        (bars.__setitem__(s, b) if not b.empty else missing.append(s))
    if not bars:
        raise SystemExit("No history. Run tools/download_history.py first.")
    if missing:
        print(f"no history for {', '.join(missing)} (skipped). To add: python tools/download_history.py "
              f"--symbols {','.join(missing)} --start 2019-01-01")
    syms = list(bars)
    all_days = sorted(set().union(*(set(trading_days(b)) for b in bars.values())))
    split = date.fromisoformat(args.split) if args.split else all_days[-1] - timedelta(days=365)
    tune = {d for d in all_days if d < split}
    check = {d for d in all_days if d >= split}
    nt, nc = len(tune), len(check)
    tables = {s: q.daily_table(b) for s, b in bars.items()}

    L = [f"# ORB study {datetime.now():%Y-%m-%d}\n",
         f"- Symbols: {', '.join(syms)}" + (f" (missing: {', '.join(missing)})" if missing else ""),
         f"- Tune {min(tune)} to {max(tune)} ({nt} days); check {min(check)} to {max(check)} ({nc} days)",
         "- Model: first 5-minute candle sets the direction, 5-min trend and VWAP must agree, "
         "market entry 9:36, stop beyond the opening range (x the stop width), hold to the close or 10R",
         "- 'per week' counts trades across all symbols. Net R = after the cost of trading it that way:",
         *(f"  - {n}: ${c.spread:.2f} spread, {c.theta_pct_per_hour:.3%} of price/hour decay, delta {c.delta}"
           for n, c in COSTS.items()), ""]

    best_choice = None
    for side, variant in orb.ORB_VARIANTS.items():
        print(f"running {side} ({variant}) on {len(syms)} symbols ...")
        trades = orb.orb_trades(bars, cfg, variant, scales, args.workers)
        L.append(f"\n# {'Longs only' if side == 'long' else 'Longs and shorts'}\n")

        # ---- 1. stop width
        L.append("## Stop width (tune period, every trade)\n")
        L += table("stop", [(f"{sc:g}x range", [t.row() for t in trades if t.stop_scale == sc and t.day in tune])
                            for sc in scales], nt)

        base = [t for t in trades if t.stop_scale == 1.0]
        bt_tune = [t for t in base if t.day in tune and t.filled]
        losers = [t for t in bt_tune if t.r <= 0]
        winners = [t for t in bt_tune if t.r > 0]

        # ---- 2. loser profile
        L.append("\n## What the losers have in common (tune, 1x stop)\n")
        if losers:
            noise = sum(1 for t in losers if t.noise_stop)
            mins = [(t.res.exit_time - t.res.fill_time).total_seconds() / 60 for t in losers
                    if t.res.exit_time is not None]
            touched = sum(1 for t in losers if t.res.mfe_r >= 1.0)
            L += [f"- {len(losers)} losers vs {len(winners)} winners ({len(winners) / len(bt_tune):.0%} win rate).",
                  f"- **{noise / len(losers):.0%} of losers were stopped out and then closed green anyway** "
                  "(stopped by noise, not a wrong call). A wider stop catches some of these.",
                  f"- Median time to the stop: {statistics.median(mins):.0f} min "
                  f"(a quarter stop within {np.percentile(mins, 25):.0f} min).",
                  f"- {touched / len(losers):.0%} of losers were up at least +1R before reversing to the stop.",
                  f"- Median opening range: losers ${statistics.median(t.or_range for t in losers):.2f}, "
                  f"winners ${statistics.median(t.or_range for t in winners):.2f}.", ""]
        g_all = graded(base, tables)
        g_tune = [t for t in g_all if t.day in tune]
        edges = q.sign_edges(g_tune)
        L += ["| sign | win with | win without | avg R with | avg R without | trades with | edge |",
              "|---|---|---|---|---|---|---|"]
        for e in sorted(edges, key=lambda e: e.edge, reverse=True):
            L.append(f"| {q.SIGN_TEXT[e.sign]} | {e.win_hit:.0%} | {e.win_miss:.0%} | {e.r_hit:+.2f} | "
                     f"{e.r_miss:+.2f} | {e.n_hit} | {e.edge:+.2f} |")

        # ---- 3. per symbol
        L.append("\n## By symbol (tune, 1x stop)\n")
        L += table("symbol", [(s, [t.row() for t in base if t.symbol == s and t.day in tune]) for s in syms], nt)

        # ---- 4. grade: choose stop width + signs + cutoff on TUNE, apply once to CHECK
        choices = []
        for sc in scales:
            g = graded([t for t in trades if t.stop_scale == sc], tables)
            gt = [t for t in g if t.day in tune]
            used = q.choose_signs(q.sign_edges(gt))
            cut = q.cutoff_for_top(gt, used, args.top) if used else 0
            pick = _per_day(gt, used, cut, args.per_day)
            m = rs.metrics([t.row for t in pick], nt, COSTS[PRIMARY])
            choices.append(((m.filled >= 40, m.ci_lo if m.filled >= 2 else -99), sc, used, cut, g))
        _, sc, used, cut, g = max(choices, key=lambda c: c[0])
        gt = [t for t in g if t.day in tune]
        gc = [t for t in g if t.day in check]
        pt, pc = _per_day(gt, used, cut, args.per_day), _per_day(gc, used, cut, args.per_day)
        at, ac = _per_day(gt, [], 0, args.per_day), _per_day(gc, [], 0, args.per_day)
        L.append(f"\n## Graded: top {args.top:.0%}, max {args.per_day}/day across symbols\n")
        kept = sum(1 for t in gt if t.grade(used) >= cut) / max(len(gt), 1)
        L.append(f"Chosen on tune: stop {sc:g}x range; signs used: "
                 f"{', '.join(q.SIGN_TEXT[s] for s in used) or 'none'}; trade when grade >= {cut} of {len(used)} "
                 f"(keeps {kept:.0%} of tune trades before the per-day cap).\n")
        L += table("set", [("tune: all trades", [t.row for t in at]), ("tune: graded", [t.row for t in pt]),
                           ("**check: all trades**", [t.row for t in ac], nc),
                           ("**check: graded**", [t.row for t in pc], nc)], nt)
        # per-week for check uses check days
        mcheck = rs.metrics([t.row for t in pc], nc, COSTS[PRIMARY])
        L.append(f"\nCheck period graded: {mcheck.filled} trades, {mcheck.per_week:.2f} per week, "
                 f"win {mcheck.win_rate:.0%}, net {mcheck.avg_net:+.2f}R ({PRIMARY}), "
                 f"90% CI {mcheck.ci_lo:+.2f} to {mcheck.ci_hi:+.2f}.")
        if best_choice is None or mcheck.ci_lo > best_choice[0]:
            best_choice = (mcheck.ci_lo, side, sc, used, cut, mcheck)

    L.append("\n# Bottom line\n")
    if best_choice:
        lo, side, sc, used, cut, m = best_choice
        L.append(f"- Best: **{'longs only' if side == 'long' else 'longs and shorts'}**, stop {sc:g}x the opening "
                 f"range, grade >= {cut} of {len(used)} signs.")
        L.append(f"- Check period: {m.filled} trades ({m.per_week:.2f}/week), win {m.win_rate:.0%}, "
                 f"net {m.avg_net:+.2f}R per trade with {PRIMARY} options (90% CI {m.ci_lo:+.2f} to {m.ci_hi:+.2f}).")
        L.append("- " + ("CONFIRMED on unseen data (lower bound above zero)." if lo > 0 else
                         "Not yet conclusive on unseen data." if m.avg_net > 0 else
                         "NOT confirmed on unseen data."))

    out = ROOT / "reports" / f"orb_study_{datetime.now():%Y-%m-%d}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[-4:]))
    print(f"\nreport written: {out}\nNext: git add reports; git commit -m \"orb study\"; git push")


def _per_day(trades, used, cut, per_day):
    if per_day == 1:
        return q.one_per_day(trades, used, cut)
    by_day: dict = {}
    for t in trades:
        if t.grade(used) >= cut:
            by_day.setdefault(t.day, []).append(t)
    out = []
    for d, v in sorted(by_day.items()):
        v.sort(key=lambda t: (t.grade(used), t.symbol in ("QQQ", "SPY")), reverse=True)
        out += v[:per_day]
    return out


if __name__ == "__main__":
    main()
