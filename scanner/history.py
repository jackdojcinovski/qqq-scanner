"""Stores and loads historical 1-minute bars as one CSV per month in data/history/."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .config import ROOT

ET = ZoneInfo("America/New_York")
HISTORY_DIR = ROOT / "data" / "history"


def month_path(symbol: str, year: int, month: int, base: Path = HISTORY_DIR) -> Path:
    return base / f"{symbol}_{year:04d}-{month:02d}.csv"


def save_month(df: pd.DataFrame, symbol: str, year: int, month: int, base: Path = HISTORY_DIR) -> Path:
    p = month_path(symbol, year, month, base)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p)
    return p


def read_csv_bars(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(ET)
    return df[["open", "high", "low", "close", "volume"]]


def load_history(symbol: str, start: date, end: date, base: Path = HISTORY_DIR) -> pd.DataFrame:
    """All stored bars from `start` to `end` inclusive, sorted, duplicates dropped."""
    frames = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        p = month_path(symbol, y, m, base)
        if p.exists():
            frames.append(read_csv_bars(p))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    if not frames:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    d = df.index.date
    return df[(d >= start) & (d <= end)]


def trading_days(bars: pd.DataFrame) -> list[date]:
    """Days that have regular-session bars."""
    t = bars.index.time
    from datetime import time
    rth = bars[(t >= time(9, 30)) & (t < time(16, 0))]
    return sorted(set(rth.index.date))
