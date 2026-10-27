"""Check a variant's trades against REAL option prices (Alpaca, Feb 2024 on).

    python tools/option_check.py --orb --itm-pct 3 --dte 20              # ORB model, deep ITM monthly
    python tools/option_check.py --orb --symbol NVDA --stop-scale 1.5 --itm-pct 3 --dte 20
    python tools/option_check.py --variant "edge|with|long|orb|fixed30|checklist" --dte 2
    python tools/option_check.py --strategy "noise|both|band1|vwap" --itm-pct 3 --dte 20   # see strategy_study.py

For every filled trade the variant would have sent since --start, it buys the
at-the-money call (long) or put (short) at the option's price in the fill
minute and sells it in the exit minute, minus --spread per share. Writes
reports/options_<date>.md. Needs history downloaded and the research cache.
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
from scanner import options as op  # noqa: E402
from scanner import research as rs  # noqa: E402
from scanner.config import ROOT, load_config  # noqa: E402
from scanner.history import ET, load_history, trading_days  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", help="a research variant (uses the scan cache)")
    p.add_argument("--orb", action="store_true", help="use the ORB model (fast engine)")
    p.add_argument("--strategy", help="an intraday strategy variant from tools/strategy_study.py")
    p.add_argument("--sides", default="long", choices=["long", "both"], help="ORB: longs only or both")
    p.add_argument("--stop-scale", type=float, default=1.0, help="ORB: stop width x the opening range")
    p.add_argument("--itm-pct", type=float, default=0.0, help="strike this %% in the money (e.g. 3)")
    p.add_argument("--symbol", default="QQQ")
    p.add_argument("--start", default="2024-02-05")
    p.add_argument("--dte", type=int, default=1, help="minimum trading days to expiry (1 = next day)")
    p.add_argument("--otm", type=int, default=0, help="strikes out of the money (0 = at the money)")
    p.add_argument("--spread", type=float, default=0.03, help="round-trip cost per share of option")
    args = p.parse_args()

    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.requests import OptionBarsRequest
    from alpaca.data.timeframe import TimeFrame

    cfg = load_config()
    if not args.orb and not args.variant and not args.strategy:
        raise SystemExit("Pass --orb, --strategy or --variant")
    bars = load_history(args.symbol, date(2015, 1, 1), date.today())
    if bars.empty:
        raise SystemExit("No history. Run tools/download_history.py first.")
    start = date.fromisoformat(args.start)
    days = {d for d in trading_days(bars) if d >= start}
    if args.strategy:
        from scanner import intraday as ix
        if args.strategy not in ix.variants():
            raise SystemExit(f"Unknown strategy. Choose from: {', '.join(ix.variants())}")
        label = f"strategy `{args.strategy}`"
        grid = ix.build_grid(args.symbol, bars[bars.index.date >= start - timedelta(days=40)])
        recs = [t for t in ix.run_all({args.symbol: grid}, [args.strategy])[args.strategy] if t.day in days]
    elif args.orb:
        from types import SimpleNamespace
        from scanner import orb
        label = f"ORB {args.sides}, stop {args.stop_scale:g}x range"
        ots = orb.orb_trades({args.symbol: bars[bars.index.date >= start - timedelta(days=7)]}, cfg,
                             orb.ORB_VARIANTS[args.sides], (args.stop_scale,), workers=1)
        recs = [SimpleNamespace(day=t.day, direction=t.direction, entry=t.res.fill_price, r=t.r,
                                fill_time=t.res.fill_time, exit_time=t.res.exit_time)
                for t in ots if t.filled and t.day in days]
    else:
        rs.variant(args.variant)
        label = f"`{args.variant}`"
        print(f"building/reusing the scan cache for {args.symbol} ...")
        paths = rs.build_cache({args.symbol: bars}, cfg)
        recs = [r for r in rs.full_records(paths, cfg, args.variant, days) if r.taken and r.filled]
    print(f"{len(recs)} filled trades since {start}; fetching option bars ...")

    load_dotenv(ROOT / ".env")
    client = OptionHistoricalDataClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET"))
    trades: list[op.OptionTrade] = []
    missing = 0
    for n, rec in enumerate(recs, 1):
        right = "C" if rec.direction == "long" else "P"
        ref = rec.entry * (1 - args.itm_pct / 100) if right == "C" else rec.entry * (1 + args.itm_pct / 100)
        got = None
        tries = [(e, step) for e in op.expiry_candidates(rec.day, args.dte) for step in (1.0, 2.5, 5.0)]
        for expiry, step in tries:
            strike = op.atm_strike(ref, step, args.otm, right)
            sym = op.occ_symbol(args.symbol, expiry, right, strike)
            req = OptionBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame.Minute,
                                    start=(rec.fill_time - pd.Timedelta(minutes=10)).to_pydatetime(),
                                    end=(rec.exit_time + pd.Timedelta(minutes=2)).to_pydatetime())
            try:
                df = client.get_option_bars(req).df
            except Exception as exc:
                print(f"  {sym}: {exc}")
                df = pd.DataFrame()
            clock.sleep(0.35)
            if df is not None and not df.empty:
                df = df.xs(sym, level="symbol") if "symbol" in df.index.names else df
                df.index = pd.to_datetime(df.index).tz_convert(ET)
                got = op.option_trade(rec, df, sym, args.spread)
                if got:
                    break
        if got:
            trades.append(got)
        else:
            missing += 1
        if n % 10 == 0 or n == len(recs):
            print(f"  {n}/{len(recs)} trades, {len(trades)} priced, {missing} without option data")

    L = [f"# Option price check {datetime.now():%Y-%m-%d}\n",
         f"- {label} on {args.symbol}, trades since {start}",
         f"- {f'{args.itm_pct:g}% in the money' if args.itm_pct else ('ATM' if not args.otm else f'{args.otm} strike(s) OTM')}, "
         f"expiry at least {args.dte} trading day(s) out, "
         f"${args.spread:.2f}/share round-trip spread",
         f"- {len(recs)} filled trades, {len(trades)} priced with real option bars, {missing} skipped (no data)\n"]
    if trades:
        pnl = [t.pnl for t in trades]
        wins = [t for t in trades if t.win]
        gains = sum(x for x in pnl if x > 0)
        losses = -sum(x for x in pnl if x < 0)
        eq = peak = dd = 0.0
        for x in pnl:
            eq += x
            peak = max(peak, eq)
            dd = max(dd, peak - eq)
        share_wins = sum(1 for t in trades if t.share_r > 0)
        L += ["| metric | value |", "|---|---|",
              f"| win rate | {len(wins) / len(trades):.0%} (share trades: {share_wins / len(trades):.0%}) |",
              f"| avg P&L per contract | ${statistics.mean(pnl):+.0f} |",
              f"| median P&L per contract | ${statistics.median(pnl):+.0f} |",
              f"| total P&L, 1 contract each | ${sum(pnl):+,.0f} |",
              f"| profit factor | {gains / losses if losses else float('inf'):.2f} |",
              f"| max drawdown, 1 contract | ${dd:,.0f} |",
              f"| avg premium paid | ${statistics.mean(t.entry for t in trades) * 100:,.0f} |",
              f"| avg return on premium | {statistics.mean(t.ret for t in trades):+.1%} |",
              f"| worst trade | ${min(pnl):+,.0f} |", f"| best trade | ${max(pnl):+,.0f} |", ""]
        by_year: dict[int, list[float]] = {}
        for t in trades:
            by_year.setdefault(t.day.year, []).append(t.pnl)
        L += ["| year | trades | avg $/contract | total $ |", "|---|---|---|---|"]
        for y, v in sorted(by_year.items()):
            L.append(f"| {y} | {len(v)} | {statistics.mean(v):+.0f} | {sum(v):+,.0f} |")
        L += ["", "| day | side | option | buy | sell | share R | $ P&L |", "|---|---|---|---|---|---|---|"]
        for t in trades:
            L.append(f"| {t.day} | {t.direction} | {t.symbol} | {t.entry:.2f} | {t.exit:.2f} "
                     f"| {t.share_r:+.2f} | {t.pnl:+.0f} |")
    out = ROOT / "reports" / f"options_{datetime.now():%Y-%m-%d}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"report written: {out}")


if __name__ == "__main__":
    main()
