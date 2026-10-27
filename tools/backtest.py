"""Backtest the scanner on downloaded history and print a report.

    python tools/download_history.py                     # once, ~2 years of bars
    python tools/backtest.py                             # everything downloaded
    python tools/backtest.py --split 2025-11-01          # tune before, check after
    python tools/backtest.py --start 2025-01-01 --end 2025-06-30

Every trade and near miss is saved to data/backtest/ as a CSV you can open
in Excel. Results are in R on the QQQ share price; option P&L will differ.
"""
from __future__ import annotations

import argparse
import sys
import time as clock
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scanner import backtest as bt  # noqa: E402
from scanner.config import ROOT, load_config  # noqa: E402
from scanner.history import load_history, trading_days  # noqa: E402


MIN_SAMPLE = 30


def fmt(s: bt.Stats) -> str:
    pf = "inf" if s.profit_factor == float("inf") else f"{s.profit_factor:.2f}"
    small = "  (small sample)" if s.filled < MIN_SAMPLE else ""
    return (f"{s.filled:>4} filled  win {s.win_rate:>5.1%}  avg {s.avg_r:+.2f}R  "
            f"total {s.total_r:+7.1f}R  PF {pf:>5}  worst streak {s.max_losing_streak:>2}  "
            f"max DD {s.max_drawdown_r:.1f}R{small}")


def section(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


def report(records: list[bt.TradeRecord], days: int, label: str) -> None:
    taken = [r for r in records if r.taken]
    weeks = max(days / 5, 1)
    print(f"\n=== {label}: {days} trading days ===")

    section("What you would have traded (live rules)")
    s = bt.stats(taken)
    print(f"  {len(taken)} alerts ({len(taken) / weeks:.1f} per week), "
          f"{s.filled} filled ({s.filled / len(taken):.0%} fill rate)" if taken else "  no alerts")
    if taken:
        print("  " + fmt(s))
        outcomes = {}
        for r in taken:
            outcomes[r.outcome] = outcomes.get(r.outcome, 0) + 1
        print("  outcomes: " + ", ".join(f"{k} {v}" for k, v in sorted(outcomes.items())))

    section("Every setup by score band (ignores daily limits)")
    for band, st in bt.by_score_band(records):
        print(f"  {band:>7}  " + fmt(st))

    section("If the alert threshold were...")
    for t, st in bt.by_threshold(records):
        print(f"  >= {t:<3}  " + fmt(st))

    section("Components: setups that scored it vs didn't (avg R)")
    for name, hit, miss in bt.by_component(records):
        print(f"  {name:<20} hit {hit.filled:>4} @ {hit.avg_r:+.2f}R   "
              f"miss {miss.filled:>4} @ {miss.avg_r:+.2f}R   edge {hit.avg_r - miss.avg_r:+.2f}R")

    section("By time of day")
    for k, st in bt.by_key(records, bt.time_bucket):
        print(f"  {k:<12} " + fmt(st))

    section("By direction")
    for k, st in bt.by_key(records, lambda r: r.direction):
        print(f"  {k:<6} " + fmt(st))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", help="YYYY-MM-DD (default: earliest downloaded)")
    p.add_argument("--end", help="YYYY-MM-DD (default: latest downloaded)")
    p.add_argument("--split", help="YYYY-MM-DD: report the tune period (before) and check period (from) separately")
    p.add_argument("--workers", type=int, default=None, help="parallel processes (default: CPU count - 1)")
    args = p.parse_args()

    cfg = load_config()
    start = date.fromisoformat(args.start) if args.start else date(2000, 1, 1)
    end = date.fromisoformat(args.end) if args.end else date(2100, 1, 1)
    bars = load_history(cfg["symbol"], start if args.start else date(2015, 1, 1), end if args.end else date.today())
    if bars.empty:
        raise SystemExit("No history found. Run: python tools/download_history.py")
    days = [d for d in trading_days(bars) if start <= d <= end]
    print(f"{cfg['symbol']}: {len(days)} trading days, {days[0]} to {days[-1]}")

    t0 = clock.time()
    last = [0.0]

    def progress(n, total):
        if clock.time() - last[0] > 2 or n == total:
            last[0] = clock.time()
            print(f"\r  replaying day {n}/{total}  ({clock.time() - t0:.0f}s)", end="", flush=True)

    records = bt.run_backtest(bars, days, cfg, workers=args.workers, progress=progress)
    print()

    out_dir = ROOT / "data" / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"trades_{days[0]}_{days[-1]}.csv"
    bt.to_frame(records).to_csv(out, index=False)

    if args.split:
        split = date.fromisoformat(args.split)
        tune = [r for r in records if r.day < split]
        check = [r for r in records if r.day >= split]
        report(tune, len([d for d in days if d < split]), f"TUNE period (before {split})")
        report(check, len([d for d in days if d >= split]), f"CHECK period (from {split})")
    else:
        report(records, len(days), "ALL days")
    print(f"\nSaved {len(records)} setups to {out}")
    print(f"Rows marked (small sample) have under {MIN_SAMPLE} filled trades: don't tune on them.")


if __name__ == "__main__":
    main()
