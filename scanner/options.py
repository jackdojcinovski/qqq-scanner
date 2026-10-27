"""Turn share-based backtest trades into option trades using real option bars.

For each filled trade: buy the at-the-money call (long) or put (short) on the
nearest expiry at least `min_dte` trading days out, at the option's price in
the fill minute, and sell it in the exit minute. Alpaca has option bars from
February 2024.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd


def occ_symbol(root: str, expiry: date, right: str, strike: float) -> str:
    """OCC option symbol, e.g. QQQ250117C00500000."""
    return f"{root}{expiry:%y%m%d}{right}{int(round(strike * 1000)):08d}"


def atm_strike(price: float, step: float = 1.0, offset_steps: int = 0, right: str = "C") -> float:
    """Nearest listed strike; offset_steps > 0 moves out of the money."""
    k = round(price / step) * step
    shift = offset_steps * step
    return k + shift if right == "C" else k - shift


def expiry_candidates(day: date, min_dte: int, max_tries: int = 6) -> list[date]:
    """Weekdays from `min_dte` trading days after `day` onward (holidays are
    skipped naturally: no bars come back for them)."""
    out = []
    d = day
    steps = 0
    while len(out) < max_tries:
        d += timedelta(days=1)
        if d.weekday() >= 5:
            continue
        steps += 1
        if steps >= min_dte:
            out.append(d)
    return out


def price_at(bars: pd.DataFrame, ts: pd.Timestamp, max_gap_min: int = 5) -> float | None:
    """Close of the last option bar at or before `ts`, if recent enough."""
    if bars.empty:
        return None
    upto = bars[bars.index <= ts]
    if upto.empty:
        return None
    last = upto.index[-1]
    if (ts - last) > pd.Timedelta(minutes=max_gap_min):
        return None
    return float(upto["close"].iloc[-1])


@dataclass
class OptionTrade:
    day: date
    direction: str
    symbol: str
    entry: float
    exit: float
    share_r: float
    pnl: float           # $ per contract after spread
    ret: float           # return on premium after spread

    @property
    def win(self) -> bool:
        return self.pnl > 0


def option_trade(rec, opt_bars: pd.DataFrame, symbol: str, spread: float) -> OptionTrade | None:
    entry = price_at(opt_bars, rec.fill_time)
    exit_ = price_at(opt_bars, rec.exit_time)
    if entry is None or exit_ is None or entry <= 0:
        return None
    pnl = (exit_ - entry - spread) * 100
    return OptionTrade(rec.day, rec.direction, symbol, entry, exit_, rec.r, pnl, (exit_ - entry - spread) / entry)
