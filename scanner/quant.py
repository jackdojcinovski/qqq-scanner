"""Quantitative signs known by 9:35 ET, for grading opening-range trades.

Every value uses only data available at the decision time: prior sessions'
daily bars and today's first five 1-minute bars (9:30-9:34). Nothing from
later in the day leaks in.

Signs (each True/False, from the trade's point of view, long or short):
  gap_with       today's open gapped in the trade direction vs yesterday's close
  daily_trend    yesterday's close on the trade side of its 20-day average
  strong_open    first-5-minute move bigger than its usual size (14-day average
                 of |9:35 move|), i.e. outside the normal noise band
  opening_volume first-5-minute volume at least 1.2x its 14-day average
  calm_regime    14-day realized volatility below its 1-year median
  tight_range    opening range under 25% of the 14-day daily ATR (small risk)
  prior_day_with yesterday's candle closed in the trade direction
  peers_agree    QQQ and SPY first candles point the same way
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time

import numpy as np
import pandas as pd

SIGNS = ["gap_with", "daily_trend", "strong_open", "opening_volume", "calm_regime",
         "tight_range", "prior_day_with", "peers_agree"]

SIGN_TEXT = {
    "gap_with": "opened with a gap in the trade direction",
    "daily_trend": "daily trend agrees (vs 20-day average)",
    "strong_open": "first 5 minutes moved more than normal",
    "opening_volume": "opening volume 1.2x its usual",
    "calm_regime": "volatility below its 1-year median",
    "tight_range": "tight opening range (small risk vs daily ATR)",
    "prior_day_with": "yesterday closed in the trade direction",
    "peers_agree": "QQQ and SPY agree",
}


def daily_table(bars: pd.DataFrame) -> pd.DataFrame:
    """One row per session with the raw inputs: regular-session OHLC, the
    9:30-9:34 open window, and rolling stats that use PRIOR days only."""
    t = bars.index.time
    rth = bars[(t >= time(9, 30)) & (t < time(16, 0))]
    d = rth.groupby(rth.index.date)
    daily = pd.DataFrame({
        "open": d["open"].first(), "high": d["high"].max(), "low": d["low"].min(),
        "close": d["close"].last(),
    })
    first = rth[rth.index.time < time(9, 35)].groupby(rth[rth.index.time < time(9, 35)].index.date)
    daily["o5_close"] = first["close"].last()
    daily["o5_high"] = first["high"].max()
    daily["o5_low"] = first["low"].min()
    daily["o5_vol"] = first["volume"].sum()
    daily.index = pd.Index(daily.index, name="day")

    prev_close = daily["close"].shift(1)
    tr = pd.concat([daily["high"] - daily["low"], (daily["high"] - prev_close).abs(),
                    (daily["low"] - prev_close).abs()], axis=1).max(axis=1)
    ret = daily["close"].pct_change()
    o5_move = (daily["o5_close"] / daily["open"] - 1).abs()
    out = pd.DataFrame(index=daily.index)
    out["open"] = daily["open"]
    out["o5_ret"] = daily["o5_close"] / daily["open"] - 1
    out["o5_range"] = daily["o5_high"] - daily["o5_low"]
    out["prev_close"] = prev_close
    out["prev_open"] = daily["open"].shift(1)
    out["sma20_prev"] = daily["close"].rolling(20).mean().shift(1)
    out["atr14_prev"] = tr.rolling(14).mean().shift(1)
    out["rv14_prev"] = ret.rolling(14).std().shift(1)
    out["rv14_med_prev"] = out["rv14_prev"].rolling(252, min_periods=60).median().shift(1)
    out["noise_prev"] = o5_move.rolling(14).mean().shift(1)
    out["o5_vol"] = daily["o5_vol"]
    out["o5_vol_avg_prev"] = daily["o5_vol"].rolling(14).mean().shift(1)
    return out


def signs_for(row: pd.Series, direction: str, peer_o5_ret: float | None) -> dict[str, bool]:
    s = 1 if direction == "long" else -1

    def ok(x) -> bool:
        return bool(pd.notna(x) and x)

    return {
        "gap_with": ok(s * (row["open"] - row["prev_close"]) > 0),
        "daily_trend": ok(s * (row["prev_close"] - row["sma20_prev"]) > 0),
        "strong_open": ok(abs(row["o5_ret"]) > row["noise_prev"]) and ok(s * row["o5_ret"] > 0),
        "opening_volume": ok(row["o5_vol"] >= 1.2 * row["o5_vol_avg_prev"]),
        "calm_regime": ok(row["rv14_prev"] < row["rv14_med_prev"]),
        "tight_range": ok(row["o5_range"] < 0.25 * row["atr14_prev"]),
        "prior_day_with": ok(s * (row["prev_close"] - row["prev_open"]) > 0),
        "peers_agree": peer_o5_ret is not None and pd.notna(peer_o5_ret) and s * peer_o5_ret > 0,
    }


@dataclass
class GradedTrade:
    symbol: str
    day: date
    direction: str
    r: float
    filled: bool
    signs: dict[str, bool]
    row: object = None          # the research Row, for cost-aware metrics

    def grade(self, used: list[str]) -> int:
        return sum(1 for k in used if self.signs.get(k))


def attach_signs(trades, tables: dict[str, pd.DataFrame]) -> list[GradedTrade]:
    """trades: research Rows. tables: symbol -> daily_table."""
    out = []
    for t in trades:
        tab = tables.get(t.symbol)
        if tab is None or t.day not in tab.index:
            continue
        row = tab.loc[t.day]
        peers = [s for s in tables if s != t.symbol and t.day in tables[s].index]
        peer = tables[peers[0]].loc[t.day, "o5_ret"] if peers else None
        out.append(GradedTrade(t.symbol, t.day, t.direction, t.r, t.filled,
                               signs_for(row, t.direction, peer), t))
    return out


@dataclass
class SignEdge:
    """Average R (before costs, so it doesn't depend on how you trade it) and win
    rate for trades with and without one sign."""
    sign: str
    n_hit: int
    n_miss: int
    r_hit: float
    r_miss: float
    win_hit: float
    win_miss: float

    @property
    def edge(self) -> float:
        return self.r_hit - self.r_miss


def sign_edges(trades: list[GradedTrade]) -> list[SignEdge]:
    filled = [t for t in trades if t.filled]
    out = []
    for k in SIGNS:
        hit = [t.r for t in filled if t.signs[k]]
        miss = [t.r for t in filled if not t.signs[k]]
        out.append(SignEdge(k, len(hit), len(miss),
                            float(np.mean(hit)) if hit else 0.0, float(np.mean(miss)) if miss else 0.0,
                            float(np.mean([x > 0 for x in hit])) if hit else 0.0,
                            float(np.mean([x > 0 for x in miss])) if miss else 0.0))
    return out


def choose_signs(edges: list[SignEdge], min_side: int = 40, min_edge: float = 0.05) -> list[str]:
    """Signs that helped on the tune period, with enough trades on both sides."""
    return [e.sign for e in edges if e.n_hit >= min_side and e.n_miss >= min_side and e.edge >= min_edge]


def cutoff_for_top(trades: list[GradedTrade], used: list[str], top_frac: float = 0.25) -> int:
    """Smallest grade that keeps at most `top_frac` of tune trades."""
    grades = sorted((t.grade(used) for t in trades), reverse=True)
    if not grades:
        return 0
    k = max(1, int(len(grades) * top_frac))
    g = grades[k - 1]
    # if ties push us over the fraction, require one more sign
    if sum(1 for x in grades if x >= g) > len(grades) * top_frac * 1.15 and g < len(used):
        g += 1
    return g


def one_per_day(trades: list[GradedTrade], used: list[str], cutoff: int,
                prefer: tuple[str, ...] = ("QQQ", "SPY")) -> list[GradedTrade]:
    """Keep trades with grade >= cutoff, at most one per day (highest grade,
    then the preferred symbol order)."""
    best: dict[date, GradedTrade] = {}
    for t in trades:
        g = t.grade(used)
        if g < cutoff:
            continue
        cur = best.get(t.day)
        rank = (g, -prefer.index(t.symbol) if t.symbol in prefer else -99)
        if cur is None or rank > (cur.grade(used), -prefer.index(cur.symbol) if cur.symbol in prefer else -99):
            best[t.day] = t
    return sorted(best.values(), key=lambda t: t.day)
