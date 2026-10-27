"""Synthetic bar builders for tests."""
from __future__ import annotations

from datetime import datetime

import pandas as pd

ET = "America/New_York"


def bars_from(rows: list[tuple], start: str) -> pd.DataFrame:
    """rows = [(open, high, low, close, volume), ...] one per minute from `start`."""
    idx = pd.date_range(pd.Timestamp(start, tz=ET), periods=len(rows), freq="1min")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx)


def flat_day(day: str, price: float = 500.0, high: float | None = None,
             low: float | None = None, minutes: int = 390, vol: int = 1000) -> pd.DataFrame:
    """A quiet regular session; optionally pin the day's high and low."""
    rows = []
    for i in range(minutes):
        p = price + (0.05 if i % 2 else -0.05)
        rows.append((p, p + 0.05, p - 0.05, p, vol))
    df = bars_from(rows, f"{day} 09:30")
    if high is not None:
        df.iloc[100, df.columns.get_loc("high")] = high
    if low is not None:
        df.iloc[200, df.columns.get_loc("low")] = low
    return df


def _candle(o: float, c: float, wick: float = 0.05, vol: int = 1000,
           high: float | None = None, low: float | None = None) -> tuple:
    return (o, high if high is not None else max(o, c) + wick,
            low if low is not None else min(o, c) - wick, c, vol)


def bullish_sweep_day() -> pd.DataFrame:
    """Prior day 495-505 (close ~500), premarket 499.0-501.5, then:
    opening range ~500.3-501.0, drift down with a swing high at 500.9,
    a sweep of the premarket low, a displacement candle through the swing
    high that leaves a bullish FVG, and a hold above it."""
    prior = flat_day("2026-10-27", 500.0, high=505.0, low=495.0)
    pre_rows = [_candle(500.0, 500.1)] * 30
    pre = bars_from(pre_rows, "2026-10-28 08:30")
    pre.iloc[5, pre.columns.get_loc("high")] = 501.5
    pre.iloc[10, pre.columns.get_loc("low")] = 499.0

    def candle(o, c, wick=0.15, **kw):   # wider wicks so the drift leaves no gaps
        return _candle(o, c, wick=wick, **kw)

    s = []
    # 9:30-9:34 opening range 500.3-501.0
    s += [candle(500.5, 500.7, high=501.0, low=500.3), candle(500.7, 500.6), candle(500.6, 500.5),
          candle(500.5, 500.6), candle(500.6, 500.5)]
    # 9:35-9:39 drift down
    s += [candle(500.5, 500.4), candle(500.4, 500.3), candle(500.3, 500.4), candle(500.4, 500.5),
          candle(500.5, 500.6)]
    # 9:40-9:44 bump to swing high 500.9 then fade
    s += [candle(500.6, 500.7), candle(500.7, 500.8, high=500.9), candle(500.8, 500.6),
          candle(500.6, 500.4), candle(500.4, 500.2)]
    # 9:45-9:49 sell toward the premarket low
    s += [candle(500.2, 500.0), candle(500.0, 499.8), candle(499.8, 499.6), candle(499.6, 499.5),
          candle(499.5, 499.4)]
    # 9:50 sweep: wick to 498.85, close 499.2 (back above 499.0)
    s += [candle(499.4, 499.2, low=498.85, vol=2500)]
    # 9:51 small up, 9:52 bearish candle (order block), 9:53 displacement, 9:54 gap holds
    s += [candle(499.2, 499.4), candle(499.5, 499.35, high=499.6, low=499.3),
          candle(499.4, 501.3, high=501.35, low=499.35, vol=4000),
          candle(501.3, 501.25, high=501.4, low=500.0)]
    # 9:55 hold above
    s += [candle(501.25, 501.2, low=501.1)]
    session = bars_from(s, "2026-10-28 09:30")
    return pd.concat([prior, pre, session])


candle = _candle
