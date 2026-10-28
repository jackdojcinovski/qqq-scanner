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
MIN_SPREAD_FRAC = 0.2


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
    got = prepare_full(chain, S, legs, m, cap, slip, min_debit, exit_bar)
    return None if got is None else got[:2]


def prepare_full(chain: Chain, S: np.ndarray, legs: list[Leg], m: int, cap: float | None, slip: float = 0.01,
                 min_debit: float = 0.10, exit_bar: int = EXIT_BAR):
    """(debit, value per minute, fresh per minute) - fresh = every leg traded within the last 2 minutes,
    so a take-profit there is a price you could really have got, not two mismatched old prints."""
    if m < 1 or m >= exit_bar:
        return None
    for lg in legs:
        key = (lg.right, float(lg.strike))
        if key not in chain.price or chain.age[key][m - 1] > MAX_AGE or np.isnan(chain.price[key][m - 1]):
            return None
    sign = np.array([-1.0 if lg.short else 1.0 for lg in legs])          # long legs +
    px = np.vstack([chain.price[(lg.right, float(lg.strike))] for lg in legs])
    debit = float(np.sum(sign * px[:, m - 1]) + slip * len(legs))
    if debit < min_debit or (cap is not None and not (MIN_SPREAD_FRAC * cap <= debit <= 0.9 * cap)):
        return None            # an at-the-money debit spread can't really cost <20% or >90% of its width
    intr = np.vstack([_intrinsic(lg, S) for lg in legs])
    w = slice(m, exit_bar + 1)
    mark = np.sum(sign[:, None] * px[:, w], axis=0)
    floor = np.sum(sign[:, None] * intr[:, w], axis=0)
    value = np.maximum(mark, floor) - slip * len(legs)
    if cap is not None:
        value = np.minimum(value, cap - slip * len(legs))
    ages = np.vstack([chain.age[(lg.right, float(lg.strike))][w] for lg in legs])
    fresh = (ages <= 2).all(axis=0)
    return debit, np.maximum(value, 0.0), fresh


def exit_on(debit: float, value: np.ndarray, m: int, tp: float | None, sl: float | None,
            time_bar: int | None, fresh: np.ndarray | None = None) -> tuple[int, str]:
    """Offset into value: stop first, then target, then the time stop, else the 15:50 exit."""
    n = len(value)
    end = n if time_bar is None else max(1, min(n, time_bar - m + 1))
    v = value[:end]
    hs = np.flatnonzero(v <= debit * (1 - sl)) if sl is not None else ()
    ok_t = v >= debit * (1 + tp) if tp is not None else None
    if ok_t is not None and fresh is not None:
        ok_t = ok_t & fresh[:end]
    ht = np.flatnonzero(ok_t) if ok_t is not None else ()
    js = hs[0] if len(hs) else end
    jt = ht[0] if len(ht) else end
    if js < end and js <= jt:
        return int(js), "stop"
    if jt < end:
        return int(jt), "target"
    return end - 1, "time" if end < n else "close"


# ------------------------------------------------------------------ trade plans
# ("fixed", tp, sl)       all out at +tp or -sl (or the time limit)
# ("scale", tp2, sl)      half out at +50%, stop on the rest moves to breakeven, rest out at +tp2 (None = ride)

def plans() -> list[tuple]:
    out = [("fixed", tp, sl) for tp in (0.3, 0.5, 0.75, 1.0, 1.5) for sl in (0.2, 0.25, 0.35, 0.5)]
    out += [("scale", tp2, sl) for tp2 in (1.0, 1.5, None) for sl in (0.25, 0.35)]
    return out


def plan_label(p: tuple) -> str:
    kind, tp, sl = p
    if kind == "fixed":
        return f"+{tp:.0%} / -{sl:.0%}"
    return f"half +50%, rest {'ride' if tp is None else f'+{tp:.0%}'} (BE stop) / -{sl:.0%}"


def run_plan(debit: float, value: np.ndarray, m: int, plan: tuple, time_bar: int | None,
             fresh: np.ndarray | None = None) -> tuple[float, int, str]:
    """(average price received per unit, minutes held until fully out, how it ended)."""
    kind, tp, sl = plan
    if kind == "fixed":
        j, why = exit_on(debit, value, m, tp, sl, time_bar, fresh)
        return float(value[j]), j, why
    j1, why = exit_on(debit, value, m, 0.5, sl, time_bar, fresh)
    if why != "target":
        return float(value[j1]), j1, why                    # stopped or timed out before +50%: all out
    first = float(value[j1])
    rest_v = value[j1 + 1:]
    if len(rest_v) == 0:
        return first, j1, "target"
    rest_f = fresh[j1 + 1:] if fresh is not None else None
    tb = None if time_bar is None else time_bar
    # breakeven stop on the remainder: sl such that debit*(1 - sl) == debit -> sl = 0
    j2, why2 = exit_on(debit, rest_v, m + j1 + 1, tp, 0.0, tb, rest_f)
    rest = {"stop": "breakeven", "target": "target 2", "time": "time", "close": "time"}[why2]
    return 0.5 * first + 0.5 * float(rest_v[j2]), j1 + 1 + j2, f"half +50%, rest {rest}"
