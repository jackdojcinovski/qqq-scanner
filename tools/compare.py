"""Compare config variants side by side on one period of the backtest.

    python tools/compare.py                       # all variants on the TUNE period
    python tools/compare.py --period check --only D   # confirm one variant on CHECK

Tune on the tune period, pick a variant, then run it ONCE on the check period.
Re-running the check period until it looks good defeats the point of having it.

Each variant is a set of overrides on top of config.yaml. "Live" columns are
the trades you would actually have taken (2 alerts a day, stop after a loss,
one position at a time). "after cost" subtracts --cost-r per filled trade as a
rough allowance for option spreads and time decay.
"""
from __future__ import annotations

import argparse
import copy
import sys
import time as clock
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scanner import backtest as bt  # noqa: E402
from scanner.config import load_config  # noqa: E402
from scanner.history import load_history, trading_days  # noqa: E402

TREND = {"trade": {"require_trend": True}}
EDGE = {"trade": {"entry_at": "near_edge"}}
LONGS = {"trade": {"directions": ["long"]}}
NO_OB = {"weights": {"order_block": 0, "orb_break": 15}}
T80 = {"thresholds": {"alert_score": 80}}


def merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def combine(*parts: dict) -> dict:
    out: dict = {}
    for p in parts:
        out = merge(out, p)
    return out


VARIANTS: dict[str, tuple[str, dict]] = {
    "A": ("baseline (current config.yaml)", {}),
    "B": ("trend gate", TREND),
    "C": ("near-edge entry", EDGE),
    "D": ("trend gate + near-edge entry", combine(TREND, EDGE)),
    "E": ("D + longs only", combine(TREND, EDGE, LONGS)),
    "F": ("D + order block points moved to ORB", combine(TREND, EDGE, NO_OB)),
    "G": ("D + alert threshold 80", combine(TREND, EDGE, T80)),
}


def summarize(records: list[bt.TradeRecord], days: int, cost_r: float) -> dict[str, Any]:
    taken = [r for r in records if r.taken]
    s = bt.stats(taken)
    after = s.avg_r - cost_r if s.filled else 0.0
    ran = sum(1 for r in taken if r.outcome == "ran away")
    return {
        "alerts/wk": len(taken) / max(days / 5, 1),
        "fill": s.filled / len(taken) if taken else 0.0,
        "filled": s.filled, "win": s.win_rate, "avgR": s.avg_r, "after": after,
        "PF": s.profit_factor, "DD": s.max_drawdown_r, "streak": s.max_losing_streak, "ran": ran,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="2025-10-27", help="tune before this date, check from it")
    p.add_argument("--period", choices=["tune", "check", "all"], default="tune")
    p.add_argument("--only", help="comma-separated variant letters, e.g. A,D")
    p.add_argument("--cost-r", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=None)
    args = p.parse_args()

    base = load_config()
    bars = load_history(base["symbol"], date(2015, 1, 1), date.today())
    if bars.empty:
        raise SystemExit("No history found. Run: python tools/download_history.py")
    split = date.fromisoformat(args.split)
    all_days = trading_days(bars)
    days = {"tune": [d for d in all_days if d < split],
            "check": [d for d in all_days if d >= split],
            "all": all_days}[args.period]
    wanted = args.only.split(",") if args.only else list(VARIANTS)

    print(f"{base['symbol']} {args.period.upper()} period: {len(days)} days, {days[0]} to {days[-1]}")
    if args.period == "check":
        print("CHECK period: run your chosen variant here once, then stop tuning.")
    print()
    rows = []
    for key in wanted:
        name, over = VARIANTS[key]
        cfg = merge(base, over)
        t0 = clock.time()
        print(f"  running {key}: {name} ...", end="", flush=True)
        recs = bt.run_backtest(bars, days, cfg, workers=args.workers)
        print(f" {clock.time() - t0:.0f}s")
        rows.append((key, name, summarize(recs, len(days), args.cost_r)))

    print(f"\n{'':2}{'variant':<40}{'alerts/wk':>10}{'fill':>6}{'filled':>7}{'win':>7}"
          f"{'avg R':>8}{'after cost':>11}{'PF':>6}{'max DD':>8}{'streak':>7}{'ran away':>9}")
    for key, name, m in rows:
        pf = "inf" if m["PF"] == float("inf") else f"{m['PF']:.2f}"
        small = " *" if m["filled"] < 30 else ""
        print(f"{key:<2}{name[:39]:<40}{m['alerts/wk']:>10.1f}{m['fill']:>6.0%}{m['filled']:>7}"
              f"{m['win']:>7.0%}{m['avgR']:>+8.2f}{m['after']:>+11.2f}{pf:>6}{m['DD']:>7.1f}R"
              f"{m['streak']:>7}{m['ran']:>9}{small}")
    print(f"\n* under 30 filled trades: treat as noise. 'after cost' = avg R - {args.cost_r}R per trade.")


if __name__ == "__main__":
    main()
