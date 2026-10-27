"""Alpaca connections for the live scanner: history, live bars, calendar, option quotes."""
from __future__ import annotations

import logging
import sys
from datetime import date, datetime, timedelta

import pandas as pd

from .history import ET

log = logging.getLogger(__name__)


class Alpaca:
    def __init__(self, key: str, secret: str):
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.historical.option import OptionHistoricalDataClient
        from alpaca.trading.client import TradingClient
        self.key, self.secret = key, secret
        self.stocks = StockHistoricalDataClient(key, secret)
        self.options = OptionHistoricalDataClient(key, secret)
        self.trading = TradingClient(key, secret, paper=True)

    def bars(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Minute, feed=DataFeed.IEX,
                               start=start.to_pydatetime(), end=end.to_pydatetime())
        raw = self.stocks.get_stock_bars(req).df
        if raw.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = raw.xs(symbol, level="symbol")[["open", "high", "low", "close", "volume"]]
        df.index = df.index.tz_convert(ET)
        return df

    def session(self, day: date) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        """(open, close) for a trading day, or None if the market is closed."""
        from alpaca.trading.requests import GetCalendarRequest
        cal = self.trading.get_calendar(GetCalendarRequest(start=day, end=day))
        if not cal:
            return None
        c = cal[0]
        return (pd.Timestamp(datetime.combine(c.date, c.open.time()) if hasattr(c.open, "time") else c.open, tz=ET),
                pd.Timestamp(datetime.combine(c.date, c.close.time()) if hasattr(c.close, "time") else c.close, tz=ET))

    def previous_session(self, day: date) -> date:
        from alpaca.trading.requests import GetCalendarRequest
        cal = self.trading.get_calendar(GetCalendarRequest(start=day - timedelta(days=10), end=day))
        days = [c.date for c in cal if c.date < day]
        return days[-1]

    def option_mid(self, occ: str) -> float | None:
        from alpaca.data.enums import OptionsFeed
        from alpaca.data.requests import OptionLatestQuoteRequest
        q = self.options.get_option_latest_quote(
            OptionLatestQuoteRequest(symbol_or_symbols=occ, feed=OptionsFeed.INDICATIVE))
        quote = q.get(occ)
        if not quote or not quote.ask_price:
            return None
        return round((quote.bid_price + quote.ask_price) / 2, 2)

    def stream(self):
        from alpaca.data.enums import DataFeed
        from alpaca.data.live import StockDataStream
        return StockDataStream(self.key, self.secret, feed=DataFeed.IEX)


def keep_awake(on: bool = True) -> None:
    """Stop Windows from sleeping while the scanner runs (no-op elsewhere)."""
    if sys.platform != "win32":
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))
