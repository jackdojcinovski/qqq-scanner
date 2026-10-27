"""Download QQQ 1-minute bars from Alpaca (free IEX feed) into data/history/.

    python tools/download_history.py                 # last 2 years
    python tools/download_history.py --years 1
    python tools/download_history.py --start 2024-11-01 --end 2026-10-26

Months already on disk are skipped, except the current month, which is
refreshed. Re-run any time to top up.
"""
from __future__ import annotations

import argparse
import os
import sys
import time as clock
from datetime import date, datetime, time, timedelta
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scanner.config import ROOT, load_config  # noqa: E402
from scanner.history import ET, month_path, save_month  # noqa: E402


def month_range(start: date, end: date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--years", type=float, default=2.0)
    p.add_argument("--start")
    p.add_argument("--end")
    args = p.parse_args()

    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    cfg = load_config()
    symbol = cfg["symbol"]
    today = datetime.now(ET).date()
    end = date.fromisoformat(args.end) if args.end else today - timedelta(days=1)
    start = date.fromisoformat(args.start) if args.start else end - timedelta(days=int(365 * args.years))

    load_dotenv(ROOT / ".env")
    client = StockHistoricalDataClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET"))

    months = list(month_range(start, end))
    for n, (y, m) in enumerate(months, 1):
        path = month_path(symbol, y, m)
        current = (y, m) == (today.year, today.month)
        if path.exists() and not current:
            print(f"[{n:>2}/{len(months)}] {y}-{m:02d} already downloaded")
            continue
        m_start = max(date(y, m, 1), start)
        m_end_day = (date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1))
        m_end = min(m_end_day, end)
        req = StockBarsRequest(
            symbol_or_symbols=symbol, timeframe=TimeFrame.Minute, feed=DataFeed.IEX,
            start=datetime.combine(m_start, time(4, 0), ET),
            end=datetime.combine(m_end, time(20, 0), ET),
        )
        for attempt in range(3):
            try:
                raw = client.get_stock_bars(req).df
                break
            except Exception as exc:  # network hiccup or rate limit
                if attempt == 2:
                    raise
                print(f"    retrying after error: {exc}")
                clock.sleep(5 * (attempt + 1))
        if raw.empty:
            print(f"[{n:>2}/{len(months)}] {y}-{m:02d} no bars")
            continue
        df = raw.xs(symbol, level="symbol")[["open", "high", "low", "close", "volume"]]
        df.index = df.index.tz_convert(ET)
        save_month(df, symbol, y, m)
        days = len(set(df.index.date))
        print(f"[{n:>2}/{len(months)}] {y}-{m:02d} {len(df):>7,} bars over {days} days")
        clock.sleep(0.4)   # stay well under the free plan's 200 requests/min
    print("Done.")


if __name__ == "__main__":
    main()
