"""Replay a past trading day through the scanner and print every setup it
would have logged or alerted. Use this to check the engine against days you
marked by hand on TradingView.

    python tools/replay_day.py 2026-10-26
    python tools/replay_day.py 2026-10-26 --show-all    # include 60-74 scores

Bars are cached in data/bars/ so re-running a day is instant. News blocks
are not applied (the free calendar only covers the current week).
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scanner.config import ROOT, load_config  # noqa: E402
from scanner.replay import format_hit, replay  # noqa: E402

ET = ZoneInfo("America/New_York")


def fetch_bars(day: date, symbol: str) -> pd.DataFrame:
    cache = ROOT / "data" / "bars" / f"{symbol}_{day:%Y-%m-%d}.csv"
    if cache.exists():
        df = pd.read_csv(cache, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True).tz_convert(ET)
        return df

    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    load_dotenv(ROOT / ".env")
    client = StockHistoricalDataClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET"))
    start = datetime.combine(day - timedelta(days=5), time(4, 0), ET)
    end = datetime.combine(day, time(16, 0), ET)
    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Minute,
                           start=start, end=end, feed=DataFeed.IEX)
    raw = client.get_stock_bars(req).df
    if raw.empty:
        raise SystemExit(f"No bars returned for {day}. Is it a trading day?")
    df = raw.xs(symbol, level="symbol")[["open", "high", "low", "close", "volume"]]
    df.index = df.index.tz_convert(ET)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache)
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("day", help="YYYY-MM-DD, a past trading day")
    p.add_argument("--show-all", action="store_true", help="also print 60-74 scores")
    args = p.parse_args()

    cfg = load_config()
    day = date.fromisoformat(args.day)
    bars = fetch_bars(day, cfg["symbol"])
    if not (bars.index.date == day).any():
        raise SystemExit(f"No bars on {day}. Weekend or holiday?")
    hits = replay(bars, cfg, day)
    shown = [h for h in hits if args.show_all or h.band == "alert"]
    print(f"{cfg['symbol']} {day}: {len([h for h in hits if h.band == 'alert'])} alert(s), "
          f"{len([h for h in hits if h.band == 'log'])} logged (60-74)\n")
    for h in shown:
        print(format_hit(h) + "\n")
    if not shown:
        print("No setups. Try --show-all to see the near misses.")


if __name__ == "__main__":
    main()
