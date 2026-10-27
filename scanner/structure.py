"""ICT-style structure on 1-minute session bars: swing points, liquidity
sweeps, market structure shifts, fair value gaps (and their inversions),
and order blocks.

All functions take the regular-session bars for one day and work with
integer bar positions (0 = 9:30 bar). "long"/"short" is the trade direction
a pattern supports.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .levels import Level


# --------------------------------------------------------------------------- swings

@dataclass(frozen=True)
class Swing:
    idx: int
    price: float
    kind: str          # "high" or "low"
    confirmed_idx: int  # first bar at which the swing is known


def swing_points(bars: pd.DataFrame, strength: int = 2) -> list[Swing]:
    """A swing high beats the `strength` bars on each side (strictly)."""
    highs = bars["high"].to_numpy()
    lows = bars["low"].to_numpy()
    out: list[Swing] = []
    for i in range(strength, len(bars) - strength):
        left_h, right_h = highs[i - strength:i], highs[i + 1:i + strength + 1]
        if highs[i] > left_h.max() and highs[i] > right_h.max():
            out.append(Swing(i, float(highs[i]), "high", i + strength))
        left_l, right_l = lows[i - strength:i], lows[i + 1:i + strength + 1]
        if lows[i] < left_l.min() and lows[i] < right_l.min():
            out.append(Swing(i, float(lows[i]), "low", i + strength))
    return out


def equal_levels(bars: pd.DataFrame, swings: list[Swing], tolerance_pct: float) -> list[Level]:
    """Two swing highs (or lows) within `tolerance_pct` of each other mark resting
    liquidity. Valid once the second swing is confirmed."""
    out: list[Level] = []
    for kind, name in (("high", "equal highs"), ("low", "equal lows")):
        pts = [s for s in swings if s.kind == kind]
        for a_i, a in enumerate(pts):
            for b in pts[a_i + 1:]:
                if abs(a.price - b.price) <= tolerance_pct * a.price:
                    price = max(a.price, b.price) if kind == "high" else min(a.price, b.price)
                    valid = min(b.confirmed_idx + 1, len(bars) - 1)
                    out.append(Level(name, price, bars.index[valid], kind))
    return out


# --------------------------------------------------------------------------- sweeps

@dataclass(frozen=True)
class Sweep:
    direction: str      # "long" = swept a low, "short" = swept a high
    level_name: str
    level: float
    idx: int            # bar that pierced the level
    reclaim_idx: int    # bar that closed back inside
    extreme: float      # lowest low / highest high of the sweep


def find_sweeps(bars: pd.DataFrame, levels: list[Level], pierce_pct: float = 0.0002,
                reclaim_bars: int = 2) -> list[Sweep]:
    """A wick beyond a level by `pierce_pct`, approached from the other side,
    followed by a close back inside within `reclaim_bars` bars."""
    o = bars["open"].to_numpy()
    h = bars["high"].to_numpy()
    lo = bars["low"].to_numpy()
    c = bars["close"].to_numpy()
    n = len(bars)
    out: list[Sweep] = []
    for lvl in levels:
        start = int(bars.index.searchsorted(lvl.valid_from))
        i = max(start, 0)
        while i < n:
            prev_close = c[i - 1] if i > 0 else o[i]
            # sell-side sweep of a low (supports a long)
            if lvl.side in ("low", "both") and prev_close > lvl.price and lo[i] < lvl.price * (1 - pierce_pct):
                j = _reclaim(c, i, reclaim_bars, lambda x: x > lvl.price)
                if j is not None:
                    out.append(Sweep("long", lvl.name, lvl.price, i, j, float(lo[i:j + 1].min())))
                    i = j + 1
                    continue
            # buy-side sweep of a high (supports a short)
            if lvl.side in ("high", "both") and prev_close < lvl.price and h[i] > lvl.price * (1 + pierce_pct):
                j = _reclaim(c, i, reclaim_bars, lambda x: x < lvl.price)
                if j is not None:
                    out.append(Sweep("short", lvl.name, lvl.price, i, j, float(h[i:j + 1].max())))
                    i = j + 1
                    continue
            i += 1
    out.sort(key=lambda s: s.idx)
    return out


def _reclaim(closes: np.ndarray, i: int, bars: int, inside) -> int | None:
    for j in range(i, min(i + bars, len(closes))):
        if inside(closes[j]):
            return j
    return None


# --------------------------------------------------------------------------- market structure shift

@dataclass(frozen=True)
class StructureShift:
    direction: str
    idx: int              # displacement candle
    broken_level: float   # swing point it closed beyond
    body_atr: float       # candle body as a multiple of ATR


def find_mss(bars: pd.DataFrame, sweep: Sweep, swings: list[Swing], atr: pd.Series,
             max_bars: int = 15, body_mult: float = 1.5) -> StructureShift | None:
    """After a sweep, the first candle that closes beyond the last opposite swing
    point with a body of at least `body_mult` x ATR."""
    want = "high" if sweep.direction == "long" else "low"
    ref = [s for s in swings if s.kind == want and s.idx < sweep.idx]
    if not ref:
        return None
    level = ref[-1].price
    o = bars["open"].to_numpy()
    c = bars["close"].to_numpy()
    a = atr.to_numpy()
    for j in range(sweep.reclaim_idx, min(len(bars), sweep.idx + max_bars + 1)):
        body = c[j] - o[j]
        if sweep.direction == "long":
            ok = c[j] > level and body > 0
        else:
            ok = c[j] < level and body < 0
        if ok and abs(body) >= body_mult * a[j]:
            return StructureShift(sweep.direction, j, level, float(abs(body) / a[j]))
    return None


# --------------------------------------------------------------------------- fair value gaps

@dataclass
class Gap:
    kind: str            # "FVG" or "IFVG"
    direction: str       # trade direction the zone supports
    top: float
    bottom: float
    created_idx: int     # third candle of the gap
    active_from: int     # created_idx for FVG, inversion bar for IFVG

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2


def find_gaps(bars: pd.DataFrame, atr: pd.Series, min_atr_mult: float = 0.25,
              expiry_bars: int = 60) -> list[Gap]:
    """Fair value gaps still active at the last bar, including FVGs that price
    closed through and flipped into inverse FVGs.

    Bullish FVG: candle 3's low above candle 1's high; supports longs.
    It dies if price trades back to its bottom, and inverts (now supporting
    shorts) if a candle closes below its bottom. Bearish is the mirror.
    """
    h = bars["high"].to_numpy()
    lo = bars["low"].to_numpy()
    c = bars["close"].to_numpy()
    a = atr.to_numpy()
    n = len(bars)
    last = n - 1
    active: list[Gap] = []
    for i in range(2, n):
        min_size = min_atr_mult * a[i - 1]
        if lo[i] > h[i - 2] and lo[i] - h[i - 2] >= min_size:
            g = _track(Gap("FVG", "long", float(lo[i]), float(h[i - 2]), i, i), h, lo, c, expiry_bars, last)
            if g:
                active.append(g)
        elif h[i] < lo[i - 2] and lo[i - 2] - h[i] >= min_size:
            g = _track(Gap("FVG", "short", float(lo[i - 2]), float(h[i]), i, i), h, lo, c, expiry_bars, last)
            if g:
                active.append(g)
    return active


def _track(g: Gap, h, lo, c, expiry: int, last: int) -> Gap | None:
    """Walk the gap forward to the last bar; return its live state or None if dead."""
    for j in range(g.created_idx + 1, last + 1):
        if g.kind == "FVG":
            if j - g.created_idx > expiry:
                return None
            if g.direction == "long":
                if c[j] < g.bottom:
                    g = Gap("IFVG", "short", g.top, g.bottom, g.created_idx, j)
                elif lo[j] <= g.bottom:
                    return None
            else:
                if c[j] > g.top:
                    g = Gap("IFVG", "long", g.top, g.bottom, g.created_idx, j)
                elif h[j] >= g.top:
                    return None
        else:  # IFVG: dies on expiry or a close back through the far side
            if j - g.active_from > expiry:
                return None
            if g.direction == "long" and c[j] < g.bottom:
                return None
            if g.direction == "short" and c[j] > g.top:
                return None
    return g


# --------------------------------------------------------------------------- order blocks

@dataclass(frozen=True)
class OrderBlock:
    direction: str
    idx: int
    top: float
    bottom: float


def find_order_block(bars: pd.DataFrame, sweep: Sweep, mss: StructureShift) -> OrderBlock | None:
    """Last opposite-colour candle between the sweep and the displacement."""
    o = bars["open"].to_numpy()
    c = bars["close"].to_numpy()
    for j in range(mss.idx - 1, sweep.idx - 1, -1):
        bearish = c[j] < o[j]
        if (sweep.direction == "long" and bearish) or (sweep.direction == "short" and not bearish and c[j] > o[j]):
            return OrderBlock(sweep.direction, j, float(bars["high"].iloc[j]), float(bars["low"].iloc[j]))
    return None


def overlaps(a_top: float, a_bottom: float, b_top: float, b_bottom: float) -> bool:
    return a_bottom <= b_top and b_bottom <= a_top
