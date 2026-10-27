"""Signs that a BIG move is coming, known at the moment of entry (no look-ahead).

Published day-trading research points to the same few causes of large intraday
moves: unusual volume (Zarattini, Barbon & Aziz 2024 "stocks in play": relative
volume in the first minutes), opening gaps, volatility expansion after quiet
days (Crabel's NR7 / inside days), volatility clustering (yesterday was a big
day), a wide opening range, and price already moving hard in the trade's
direction. For options we add whether the option is cheap vs normal movement.

Every sign uses only bars that have closed before the entry bar m, plus earlier days.
"""
from __future__ import annotations

import math

import numpy as np

from .intraday import DayGrid

SIGNS = {
    "rvol": "volume so far is 1.3x+ its usual for this time of day",
    "gap": "opened with a gap of 0.5+ normal daily moves",
    "gap_with": "the gap is in the trade's direction",
    "nr7": "yesterday was the narrowest day of the last 7 (squeeze)",
    "inside": "yesterday was an inside day",
    "prev_big": "yesterday moved 1.5+ normal daily moves",
    "wide_or": "first 15 minutes' range is wide vs the normal day",
    "moving": "price already moved 0.3+ normal daily moves our way since the open",
    "trend": "daily trend (20-day average) agrees with the trade",
}


class Features:
    def __init__(self, g: DayGrid, lookback: int = 14):
        self.g = g
        n = len(g.days)
        v = g.volume if g.volume is not None else np.ones_like(g.close)
        self.cumvol = np.cumsum(v, axis=1)
        self.avg_cumvol = np.full_like(self.cumvol, np.nan)
        for i in range(lookback, n):
            self.avg_cumvol[i] = self.cumvol[i - lookback:i].mean(axis=0)
        self.hi = g.high.max(axis=1)
        self.lo = g.low.min(axis=1)
        self.rng = self.hi - self.lo
        close = g.close[:, -1]
        ret = np.full(n, np.nan)
        ret[1:] = np.log(close[1:] / close[:-1])
        self.ret = ret
        self.vol20 = np.full(n, np.nan)            # daily return std of the PRIOR 20 days
        self.sma20 = np.full(n, np.nan)
        self.adr = np.full(n, np.nan)              # average daily range (prior 14), as a fraction of price
        for i in range(21, n):
            self.vol20[i] = np.nanstd(ret[i - 20:i], ddof=1)
            self.sma20[i] = close[i - 20:i].mean()
        for i in range(lookback, n):
            self.adr[i] = np.mean(self.rng[i - lookback:i] / close[i - lookback:i])
        self.close = close

    def signs(self, i: int, m: int, direction: str) -> dict[str, bool] | None:
        g = self.g
        if i < 21 or m < 1 or np.isnan(self.vol20[i]) or np.isnan(self.adr[i]):
            return None
        s = 1 if direction == "long" else -1
        o, pc = g.day_open[i], self.close[i - 1]
        sig = self.vol20[i]
        gap = o / pc - 1
        av = self.avg_cumvol[i, m - 1]
        rng = self.rng
        n15 = min(m, 15)
        or_rng = (g.high[i, :n15].max() - g.low[i, :n15].min()) / o
        return {
            "rvol": bool(av > 0 and self.cumvol[i, m - 1] / av >= 1.3),
            "gap": abs(gap) >= 0.5 * sig,
            "gap_with": s * gap > 0.25 * sig,
            "nr7": bool(i >= 8 and rng[i - 1] <= rng[i - 7:i].min()),
            "inside": bool(i >= 2 and self.hi[i - 1] <= self.hi[i - 2] and self.lo[i - 1] >= self.lo[i - 2]),
            "prev_big": bool(abs(self.ret[i - 1]) >= 1.5 * sig),
            "wide_or": bool(m >= 15 and or_rng >= 0.35 * self.adr[i]),
            "moving": s * (g.close[i, m - 1] / o - 1) >= 0.3 * sig,
            "trend": s * (pc - self.sma20[i]) > 0,
        }


def max_move(g: DayGrid, i: int, m: int, direction: str, minutes: int) -> float:
    """Best move our way within `minutes` after entry (label only - never used to decide a trade)."""
    s = 1 if direction == "long" else -1
    e = g.open[i, m]
    end = min(m + minutes, g.close.shape[1])
    path = g.high[i, m:end] if s > 0 else g.low[i, m:end]
    return float(s * (path.max() if s > 0 else path.min()) / e - s) if len(path) else 0.0


def sigma_daily(f: Features, i: int) -> float:
    v = f.vol20[i]
    return float(v) if v == v else math.nan
