"""Replay 0DTE credit spreads (verticals and iron condors) on REAL option minute bars.

Prices: the last traded price of each leg, carried forward minute to minute.
To stay honest when a leg hasn't traded for a while, the position is never
valued below its intrinsic value (what it is worth if it expired right now).

  entry     at the open of bar m, using each leg's last trade up to bar m-1;
            every leg must have traded in the 15 minutes before, else no trade
  credit    sum(short legs - slip) - sum(long legs + slip)
  buy back  sum(short legs + slip) - sum(long legs - slip)  (>= intrinsic)
  target    buy back <= credit x (1 - tp)       e.g. tp=0.5: keep half the credit
  stop      buy back >= credit x (1 + sl)       e.g. sl=1.0: lose about the credit
  exit      otherwise at 15:50 (bar 380), well before expiry, no assignment risk
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

MINUTES = 390
EXIT_BAR = 380            # 15:50
MAX_AGE = 15              # minutes since a leg's last trade, at entry
MAX_CREDIT_FRAC = 0.8     # reject credits above 80% of the width (bad / stale prints)


@dataclass
class Chain:
    """One symbol-day of 0DTE options: last-trade price and its age per minute."""
    day: date
    price: dict[tuple[str, float], np.ndarray]     # (right, strike) -> last trade carried forward (nan before any)
    age: dict[tuple[str, float], np.ndarray]       # minutes since that last trade


def parse_occ(sym: str) -> tuple[str, float]:
    return sym[-9], int(sym[-8:]) / 1000


def load_chain(df: pd.DataFrame, day: date, tz) -> Chain:
    price, age = {}, {}
    if df.empty:
        return Chain(day, price, age)
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(tz)
    mins = (ts.dt.hour * 60 + ts.dt.minute - 570).to_numpy()
    for sym, idx in df.groupby("symbol").indices.items():
        m = mins[idx]
        ok = (m >= 0) & (m < MINUTES)
        if not ok.any():
            continue
        arr = np.full(MINUTES, np.nan)
        arr[m[ok]] = df["close"].to_numpy()[idx][ok]
        traded = ~np.isnan(arr)
        last = np.where(traded, np.arange(MINUTES), -1)
        last = np.maximum.accumulate(last)
        a = np.where(last >= 0, np.arange(MINUTES) - last, 10_000)
        price[parse_occ(sym)] = pd.Series(arr).ffill().to_numpy()
        age[parse_occ(sym)] = a
    return Chain(day, price, age)


@dataclass(frozen=True)
class Leg:
    right: str        # C / P
    strike: float
    short: bool


@dataclass
class SpreadResult:
    day: date
    structure: str
    entry_bar: int
    exit_bar: int
    credit: float          # per share
    cost: float            # buy-back per share (incl. slip)
    width: float
    outcome: str           # target / stop / close

    @property
    def pnl(self) -> float:          # per spread, $
        return (self.credit - self.cost) * 100

    @property
    def max_loss(self) -> float:     # per spread, $ (also the margin)
        return max(self.width - self.credit, 0.01) * 100


def legs_for(structure: str, S: float, dist: float, width: int) -> list[Leg]:
    """Short strike `dist` (fraction) out of the money, long `width` dollars further out."""
    kp = math.floor(S * (1 - dist))
    kc = math.ceil(S * (1 + dist))
    put = [Leg("P", kp, True), Leg("P", kp - width, False)]
    call = [Leg("C", kc, True), Leg("C", kc + width, False)]
    return {"put": put, "call": call, "condor": put + call}[structure]


def _intrinsic(leg: Leg, S: np.ndarray) -> np.ndarray:
    return np.maximum(S - leg.strike, 0) if leg.right == "C" else np.maximum(leg.strike - S, 0)


def prepare(chain: Chain, S: np.ndarray, legs: list[Leg], m: int, width: float, slip: float = 0.01,
            min_credit: float = 0.05, exit_bar: int = EXIT_BAR) -> tuple[float, np.ndarray] | None:
    """(credit, buy-back cost per minute from bar m to exit_bar) or None if it can't be traded."""
    if m < 1 or m >= exit_bar:
        return None
    for lg in legs:
        k = (lg.right, float(lg.strike))
        if k not in chain.price or chain.age[k][m - 1] > MAX_AGE or np.isnan(chain.price[k][m - 1]):
            return None
    sign = np.array([1.0 if lg.short else -1.0 for lg in legs])
    px = np.vstack([chain.price[(lg.right, float(lg.strike))] for lg in legs])          # legs x minutes
    credit = float(np.sum(sign * px[:, m - 1]) - slip * len(legs))
    if credit < min_credit or credit > MAX_CREDIT_FRAC * width:
        return None            # a credit near the width is impossible in a real market: stale / bad prints
    intr = np.vstack([_intrinsic(lg, S) for lg in legs])
    window = slice(m, exit_bar + 1)
    mark = np.nansum(sign[:, None] * px[:, window], axis=0)
    floor = np.sum(sign[:, None] * intr[:, window], axis=0)
    cost = np.clip(np.maximum(mark, floor) + slip * len(legs), 0, width + slip * len(legs))
    return credit, cost


def exit_on(credit: float, cost: np.ndarray, tp: float | None, sl: float | None) -> tuple[int, str]:
    """(offset into cost, outcome): stop first when both happen in the same minute."""
    n = len(cost)
    hit_t = np.flatnonzero(cost <= credit * (1 - tp)) if tp is not None else ()
    hit_s = np.flatnonzero(cost >= credit * (1 + sl)) if sl is not None else ()
    jt = hit_t[0] if len(hit_t) else n
    js = hit_s[0] if len(hit_s) else n
    if js < n and js <= jt:
        return int(js), "stop"
    if jt < n:
        return int(jt), "target"
    return n - 1, "close"


def replay(chain: Chain, S: np.ndarray, legs: list[Leg], m: int, width: float, tp: float | None,
           sl: float | None, slip: float = 0.01, structure: str = "", min_credit: float = 0.05,
           exit_bar: int = EXIT_BAR) -> SpreadResult | None:
    got = prepare(chain, S, legs, m, width, slip, min_credit, exit_bar)
    if got is None:
        return None
    credit, cost = got
    j, why = exit_on(credit, cost, tp, sl)
    return SpreadResult(chain.day, structure, m, m + j, credit, float(cost[j]), float(width), why)
