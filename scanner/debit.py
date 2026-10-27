"""Long option positions (plain calls/puts and debit spreads) on REAL 0DTE option bars.

Same pricing rules as scanner/spreads.py: last trade carried forward, legs must
have traded in the 15 minutes before entry, never valued below intrinsic,
pay the ask (last + slip) on the way in and get the bid (last - slip) on the way out.

Structures (strikes from the underlying price at entry, $1 strikes):
  atm        buy the at-the-money option
  itm2       buy the option $2 in the money
  otm1       buy the option $1 out of the money
  ds2/3/5    debit spread: buy at the money, sell the one $2 / $3 / $5 further out
"""
from __future__ import annotations

import numpy as np

from .spreads import EXIT_BAR, MAX_AGE, Chain, Leg, _intrinsic

STRUCTURES = ["atm", "itm2", "otm1", "ds2", "ds3", "ds5"]


def legs_for(structure: str, S: float, direction: str) -> tuple[list[Leg], float | None]:
    """(legs, max value) — max value only for spreads."""
    s = 1 if direction == "long" else -1
    r = "C" if direction == "long" else "P"
    k = float(round(S))
    if structure == "atm":
        return [Leg(r, k, False)], None
    if structure == "itm2":
        return [Leg(r, k - 2 * s, False)], None
    if structure == "otm1":
        return [Leg(r, k + s, False)], None
    w = int(structure[2:])
    return [Leg(r, k, False), Leg(r, k + w * s, True)], float(w)


def prepare(chain: Chain, S: np.ndarray, legs: list[Leg], m: int, cap: float | None, slip: float = 0.01,
            min_debit: float = 0.05, exit_bar: int = EXIT_BAR) -> tuple[float, np.ndarray] | None:
    """(debit paid, value you could sell for at each minute m..exit_bar) or None."""
    if m < 1 or m >= exit_bar:
        return None
    for lg in legs:
        key = (lg.right, float(lg.strike))
        if key not in chain.price or chain.age[key][m - 1] > MAX_AGE or np.isnan(chain.price[key][m - 1]):
            return None
    sign = np.array([-1.0 if lg.short else 1.0 for lg in legs])          # long legs +
    px = np.vstack([chain.price[(lg.right, float(lg.strike))] for lg in legs])
    debit = float(np.sum(sign * px[:, m - 1]) + slip * len(legs))
    if debit < min_debit or (cap is not None and debit > 0.9 * cap):
        return None
    intr = np.vstack([_intrinsic(lg, S) for lg in legs])
    w = slice(m, exit_bar + 1)
    mark = np.sum(sign[:, None] * px[:, w], axis=0)
    floor = np.sum(sign[:, None] * intr[:, w], axis=0)
    value = np.maximum(mark, floor) - slip * len(legs)
    if cap is not None:
        value = np.minimum(value, cap - slip * len(legs))
    return debit, np.maximum(value, 0.0)


def exit_on(debit: float, value: np.ndarray, m: int, tp: float | None, sl: float | None,
            time_bar: int | None) -> tuple[int, str]:
    """Offset into value: stop first, then target, then the time stop, else the 15:50 exit."""
    n = len(value)
    end = n if time_bar is None else max(1, min(n, time_bar - m + 1))
    v = value[:end]
    hs = np.flatnonzero(v <= debit * (1 - sl)) if sl is not None else ()
    ht = np.flatnonzero(v >= debit * (1 + tp)) if tp is not None else ()
    js = hs[0] if len(hs) else end
    jt = ht[0] if len(ht) else end
    if js < end and js <= jt:
        return int(js), "stop"
    if jt < end:
        return int(jt), "target"
    return end - 1, "time" if end < n else "close"
