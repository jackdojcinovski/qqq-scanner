"""Key price levels used for liquidity sweeps and targets."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time

import pandas as pd

from .indicators import opening_range, regular_session


@dataclass(frozen=True)
class Level:
    name: str
    price: float
    valid_from: pd.Timestamp   # sweeps only count after this time
    side: str = "both"         # "low" = sell-side liquidity (longs), "high" = buy-side (shorts)


def session_date(history: pd.DataFrame) -> date:
    return history.index[-1].date()


def build_levels(history: pd.DataFrame, or_minutes: int = 5,
                 premarket_start: time = time(4, 0), open_t: time = time(9, 30)) -> list[Level]:
    """Prior day high/low/close, premarket high/low and the opening range,
    taken from a history frame that ends at the current bar."""
    today = session_date(history)
    tz = history.index.tz
    day_open = pd.Timestamp.combine(today, open_t).tz_localize(tz)
    levels: list[Level] = []

    rth = regular_session(history)
    prior = rth[rth.index.date < today]
    if not prior.empty:
        last_day = prior.index[-1].date()
        pday = prior[prior.index.date == last_day]
        levels += [
            Level("prior day high", float(pday["high"].max()), day_open, "high"),
            Level("prior day low", float(pday["low"].min()), day_open, "low"),
            Level("prior day close", float(pday["close"].iloc[-1]), day_open),
        ]

    t = history.index.time
    pre = history[(history.index.date == today) & (t >= premarket_start) & (t < open_t)]
    if not pre.empty:
        levels += [
            Level("premarket high", float(pre["high"].max()), day_open, "high"),
            Level("premarket low", float(pre["low"].min()), day_open, "low"),
        ]

    session = rth[rth.index.date == today]
    rng = opening_range(session, or_minutes, open_t)
    if rng is not None:
        or_end = day_open + pd.Timedelta(minutes=or_minutes)
        levels += [
            Level("opening range high", rng[0], or_end, "high"),
            Level("opening range low", rng[1], or_end, "low"),
        ]
    return levels
