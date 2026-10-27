"""Check a money-study winner against REAL option prices (Alpaca option bars, Feb 2024 on).

    python tools/real_options.py --setup orb --rule long --dte 0
    python tools/real_options.py --setup noise --rule regime_k1 --dte 0
    python tools/real_options.py --setup orb --rule long --dte 0 --tp 1.0

For every trade the setup would have alerted (QQQ and SPY, max 1 a day) it
looks up the actual option contract (same strike / expiry the model used),
buys at the traded price in the fill minute + half the spread, sells at the
exit minute (or at the take-profit, if a bar traded there) - half the spread,
and charges Wealthsimple's 1.5% FX each way. It also prints the MODEL's
estimate for the same trades, so we can see if the model is too kind.

Writes reports/real_options_<setup>_<rule>_<dte>dte.md.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time as clock
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scanner import intraday as ix  # noqa: E402
from scanner import optionsim as om  # noqa: E402
from scanner import options as op  # noqa: E402
from scanner import setups  # noqa: E402
from scanner.config import ROOT, load_config  # noqa: E402
from scanner.history import ET, load_history  # noqa: E402


def ts(day: date, minutes_after_open: float) -> pd.Timestamp:
    return pd.Timestamp(day.year, day.month, day.day, 9, 30, tz=ET) + pd.Timedelta(minutes=minutes_after_open)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--setup", default="orb", choices=list(setups.SETUPS))
    p.add_argument("--rule", default="long", choices=["long", "long_up_k1", "regime_k1", "regime_k0.5"])
    p.add_argument("--symbols", default="QQQ,SPY")
    p.add_argument("--dte", type=int, default=0, help="trading days to expiry (0 = same day)")
    p.add_argument("--itm-pct", type=float, default=0.0)
    p.add_argument("--tp", type=float, default=None, help="take profit on premium, e.g. 1.0 = +100%%")
    p.add_argument("--contracts", type=int, default=5)
    p.add_argument("--start", default="2024-02-05")
    p.add_argument("--iv-mult", type=float, default=1.1)
    args = p.parse_args()

    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.requests import OptionBarsRequest
    from alpaca.data.timeframe import TimeFrame

    cfg = load_config()
    start = date.fromisoformat(args.start)
    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    bars = {s: load_history(s, start - timedelta(days=60), date.today()) for s in symbols}
    grids = {s: ix.build_grid(s, b) for s, b in bars.items()}
    ctx = {s: om.daily_context(g) for s, g in grids.items()}
    print("collecting signals ...")
    sigs = setups.collect(bars, grids, cfg, [args.setup])[args.setup]
    sigs = [s for s in om.one_per_day(om.apply_regime(sigs, ctx, args.rule)) if s.day >= start]
    contract = om.Contract(f"{args.dte}DTE {args.itm_pct:g}% ITM", args.dte, args.itm_pct)
    model = {(s.symbol, s.day): (s, pth) for s, pth in om.build_paths(sigs, grids, ctx, contract, args.iv_mult)}
    print(f"{len(sigs)} trades since {start}; fetching real option bars ...")

    load_dotenv(ROOT / ".env")
    client = OptionHistoricalDataClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET"))
    rows, missing = [], 0
    for n, s in enumerate(sigs, 1):
        mp = model.get((s.symbol, s.day))
        if mp is None:
            missing += 1
            continue
        path = mp[1]
        sym = op.occ_symbol(s.symbol, path.expiry, s.right, path.strike)
        t_in, t_out = ts(s.day, s.m_in), ts(s.day, s.exit_after_min)
        try:
            df = client.get_option_bars(OptionBarsRequest(
                symbol_or_symbols=sym, timeframe=TimeFrame.Minute,
                start=(t_in - pd.Timedelta(minutes=10)).to_pydatetime(),
                end=(t_out + pd.Timedelta(minutes=2)).to_pydatetime())).df
        except Exception as exc:                                         # noqa: BLE001
            print(f"  {sym}: {exc}")
            df = pd.DataFrame()
        clock.sleep(0.35)
        if df is None or df.empty:
            missing += 1
            continue
        df = df.xs(sym, level="symbol") if "symbol" in df.index.names else df
        df.index = pd.to_datetime(df.index).tz_convert(ET)
        buy_px = op.price_at(df, t_in)
        sell_px = op.price_at(df, t_out)
        if buy_px is None or sell_px is None or buy_px <= 0:
            missing += 1
            continue
        paid = buy_px + om.half_spread(buy_px)
        got, why = sell_px - om.half_spread(sell_px), "exit"
        if args.tp is not None:
            target = paid * (1 + args.tp)
            during = df[(df.index > t_in) & (df.index < t_out)]
            hit = during[during["high"] - om.half_spread(target) >= target]
            if not hit.empty:
                got, why = target, "target"
        got = max(got, 0.0)
        real = args.contracts * 100 * ((got - paid) - om.FX * (paid + got))
        m_got, m_why, _ = om.replay(path, args.tp, None)
        mod = args.contracts * 100 * ((m_got - path.paid) - om.FX * (path.paid + m_got))
        rows.append(dict(day=s.day, symbol=s.symbol, side=s.direction, option=sym, paid=paid, got=got,
                         why=why, real=real, model=mod, model_paid=path.paid))
        if n % 10 == 0 or n == len(sigs):
            print(f"  {n}/{len(sigs)} trades, {len(rows)} priced, {missing} without option data")

    L = [f"# Real option prices: {setups.SETUPS[args.setup]} [{args.rule}], {contract.name}, "
         f"{args.contracts} contracts ({datetime.now():%Y-%m-%d})\n",
         f"- Trades since {start} on {', '.join(symbols)}, max 1 a day; take profit "
         f"{'none (hold to the setup exit)' if args.tp is None else f'+{args.tp:.0%}'}",
         f"- {len(sigs)} trades, {len(rows)} priced with real option bars, {missing} skipped (no data)",
         "- Real = actual traded option prices + half spread each side + 1.5% FX each way. "
         "Model = the money study's estimate for the same trades.\n"]
    if rows:
        real = [r["real"] for r in rows]
        mod = [r["model"] for r in rows]
        weeks = max(len(pd.bdate_range(start, date.today())) / 5, 1)
        eq = peak = dd = 0.0
        for x in real:
            eq += x
            peak = max(peak, eq)
            dd = max(dd, peak - eq)
        L += ["| | real | model |", "|---|---|---|",
              f"| trades priced | {len(rows)} | {len(rows)} |",
              f"| win rate | {sum(x > 0 for x in real) / len(real):.0%} | {sum(x > 0 for x in mod) / len(mod):.0%} |",
              f"| avg $ per trade | ${statistics.mean(real):+,.0f} | ${statistics.mean(mod):+,.0f} |",
              f"| total $ | ${sum(real):+,.0f} | ${sum(mod):+,.0f} |",
              f"| $ per week (all weeks since start) | ${sum(real) / weeks:+,.0f} | ${sum(mod) / weeks:+,.0f} |",
              f"| avg winner | ${statistics.mean([x for x in real if x > 0] or [0]):+,.0f} | "
              f"${statistics.mean([x for x in mod if x > 0] or [0]):+,.0f} |",
              f"| avg loser | ${statistics.mean([x for x in real if x <= 0] or [0]):+,.0f} | "
              f"${statistics.mean([x for x in mod if x <= 0] or [0]):+,.0f} |",
              f"| worst drawdown | ${dd:,.0f} | |",
              f"| avg cost of {args.contracts} contracts | ${statistics.mean(r['paid'] for r in rows) * 100 * args.contracts:,.0f} | "
              f"${statistics.mean(r['model_paid'] for r in rows) * 100 * args.contracts:,.0f} |", ""]
        by_m: dict[str, list[float]] = {}
        for r in rows:
            by_m.setdefault(f"{r['day']:%Y-%m}", []).append(r["real"])
        L += ["| month | trades | real $ |", "|---|---|---|"]
        for m, v in sorted(by_m.items()):
            L.append(f"| {m} | {len(v)} | ${sum(v):+,.0f} |")
        L += ["", "| day | symbol | side | option | paid | sold | exit | real $ | model $ |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            L.append(f"| {r['day']} | {r['symbol']} | {r['side']} | {r['option']} | {r['paid']:.2f} | {r['got']:.2f} "
                     f"| {r['why']} | ${r['real']:+,.0f} | ${r['model']:+,.0f} |")
        verdict = "real prices CONFIRM a profit." if sum(real) > 0 else "real prices do NOT show a profit."
        L.append(f"\n**Verdict:** {verdict} Real total ${sum(real):+,.0f} vs model ${sum(mod):+,.0f} "
                 f"on the same {len(rows)} trades.")
    out = ROOT / "reports" / f"real_options_{args.setup}_{args.rule}_{args.dte}dte.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[4:16]))
    print(f"\nreport written: {out}")


if __name__ == "__main__":
    main()
