"""Indicators on 1-minute bars.

Bars are a DataFrame indexed by tz-aware Eastern timestamps (bar start time)
with columns open, high, low, close, volume.
"""
from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd

OHLCV = ["open", "high", "low", "close", "volume"]


def session_vwap(bars: pd.DataFrame) -> pd.Series:
    """VWAP that resets each calendar day. Pass regular-session bars only."""
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3
    day = bars.index.date
    pv = (typical * bars["volume"]).groupby(day).cumsum()
    vol = bars["volume"].groupby(day).cumsum()
    return (pv / vol.replace(0, np.nan)).ffill()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    out = 100 - 100 / (1 + rs)
    out = out.where(avg_loss != 0, 100.0)
    return out.where(avg_gain.notna())


def atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's ATR. Starts from the first bar so early-session values exist."""
    prev_close = bars["close"].shift(1)
    tr = pd.concat([
        bars["high"] - bars["low"],
        (bars["high"] - prev_close).abs(),
        (bars["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def resample_5m(bars: pd.DataFrame, complete_only: bool = True) -> pd.DataFrame:
    """1-minute bars to 5-minute bars aligned to :00/:05. Drops the last bar
    if its 5 minutes aren't finished yet."""
    if bars.empty:
        return bars.copy()
    out = bars[OHLCV].resample("5min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open"])
    if complete_only and not out.empty:
        last_minute = bars.index[-1]
        if last_minute < out.index[-1] + pd.Timedelta(minutes=4):
            out = out.iloc[:-1]
    return out


def relative_volume(bars: pd.DataFrame, period: int = 20) -> pd.Series:
    """Each bar's volume divided by the average of the previous `period` bars."""
    avg = bars["volume"].rolling(period, min_periods=5).mean().shift(1)
    return bars["volume"] / avg


def regular_session(bars: pd.DataFrame, open_t: time = time(9, 30),
                    close_t: time = time(16, 0)) -> pd.DataFrame:
    t = bars.index.time
    return bars[(t >= open_t) & (t < close_t)]


def opening_range(session: pd.DataFrame, minutes: int = 5,
                  open_t: time = time(9, 30)) -> tuple[float, float] | None:
    """(high, low) of the first `minutes` of the session, or None until it's complete."""
    if session.empty:
        return None
    day_open = session.index[0].normalize() + pd.Timedelta(hours=open_t.hour, minutes=open_t.minute)
    end = day_open + pd.Timedelta(minutes=minutes)
    if session.index[-1] < end - pd.Timedelta(minutes=1):
        return None
    window = session[(session.index >= day_open) & (session.index < end)]
    if window.empty:
        return None
    return float(window["high"].max()), float(window["low"].min())
