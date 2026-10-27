"""Check a variant's trades against REAL option prices (Alpaca, Feb 2024 on).

    python tools/option_check.py --variant "edge|with|long|orb|fixed30|checklist"
    python tools/option_check.py --variant "..." --dte 2 --otm 1

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
    p.add_argument("--variant", required=True)
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
    rs.variant(args.variant)                      # validates the name early
    bars = load_history(args.symbol, date(2015, 1, 1), date.today())
    if bars.empty:
        raise SystemExit("No history. Run tools/download_history.py first.")
    start = date.fromisoformat(args.start)
    days = {d for d in trading_days(bars) if d >= start}
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
        strike = op.atm_strike(rec.entry, 1.0, args.otm, right)
        got = None
        for expiry in op.expiry_candidates(rec.day, args.dte):
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
         f"- Variant `{args.variant}` on {args.symbol}, trades since {start}",
         f"- {'ATM' if not args.otm else f'{args.otm} strike(s) OTM'}, expiry at least {args.dte} trading day(s) out, "
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
