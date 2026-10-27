"""A library of published / widely used intraday entry setups on QQQ and SPY.

Every setup looks only at bars that have CLOSED (bar m-1) and enters at the
open of the next bar (m), so nothing peeks at the future. Each gives at most
one entry per symbol-day: the first time it triggers inside its window.
Exits are not decided here: the money study manages the option with a
take-profit / stop on the premium, else sells near the close.

Bar index m: 0 = 9:30, 30 = 10:00, 60 = 10:30, 300 = 14:30, 389 = 15:59.

Setups (sources in docs, kept to the textbook parameters, not tuned):
  orb5        first 5-minute candle direction, enter 9:35 (Zarattini & Aziz 2023)
  orb15/orb30 close beyond the first 15 / 30 minutes' high or low (classic Crabel-style ORB)
  vwap10      side of VWAP at 10:00 with price beyond the open (Zarattini & Aziz VWAP trend, once a day)
  vwap_cross  first cross of VWAP after 10:00 in the direction of the day's move from the open
  vwap_pull   trend day pulls back to VWAP and holds: buy the bounce (sell the rejection)
  noise       noise-boundary breakout, checked every 30 min (Zarattini, Aziz & Barbon 2024)
  gap_go      gap > 0.3% and at 10:00 price is beyond the first 30 min in the gap direction
  gap_fade    gap > 0.3%, first 5 minutes move against the gap: trade toward yesterday's close
  pdh         first close above yesterday's high (below yesterday's low)
  momo_1030   first hour moved > 0.3% and price on the same side of VWAP: follow it at 10:30
  rsi_dip     5-min RSI(2) < 10 while the daily trend is up (> 95 while the daily trend is down)
"""
from __future__ import annotations

import numpy as np

from . import intraday as ix
from .optionsim import Signal

LAST_ENTRY = 300          # no new entries after 14:30 (0DTE decay gets too steep)


def _sig(g: ix.DayGrid, i: int, name: str, s: int, m: int) -> Signal | None:
    if m >= ix.MINUTES - 1 or m < 1:
        return None
    return Signal(name, g.symbol, g.days[i], "long" if s > 0 else "short", m, ix.MINUTES - 1,
                  float(g.close[i, ix.MINUTES - 1]), ix.MINUTES - 0.5)


def _first(cond_long: np.ndarray, cond_short: np.ndarray, lo: int, hi: int) -> tuple[int, int]:
    """First bar k in [lo, hi) where a condition is true -> (sign, entry bar k+1)."""
    for k in range(lo, min(hi, ix.MINUTES - 1)):
        if cond_long[k]:
            return 1, k + 1
        if cond_short[k]:
            return -1, k + 1
    return 0, 0


def _rsi(x: np.ndarray, n: int) -> np.ndarray:
    d = np.diff(x, prepend=x[0])
    up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
    out = np.full(len(x), 50.0)
    au = ad = None
    for k in range(1, len(x)):
        if k < n:
            continue
        if au is None:
            au, ad = up[1:k + 1].mean(), dn[1:k + 1].mean()
        else:
            au = (au * (n - 1) + up[k]) / n
            ad = (ad * (n - 1) + dn[k]) / n
        out[k] = 100.0 if ad == 0 else 100 - 100 / (1 + au / ad)
    return out


def day_signals(g: ix.DayGrid, i: int, ctx_row=None) -> list[Signal]:
    H, L, C, V = g.high[i], g.low[i], g.close[i], g.vwap[i]
    o, pc = g.day_open[i], g.prev_close[i]
    out: list[Signal] = []

    def add(name, s, m):
        if s:
            x = _sig(g, i, name, s, m)
            if x:
                out.append(x)

    # orb5
    c5 = C[4]
    add("orb5", 1 if c5 > o else (-1 if c5 < o else 0), 5)
    # orb15 / orb30
    for n in (15, 30):
        hi, lo = H[:n].max(), L[:n].min()
        s, m = _first(C > hi, C < lo, n, LAST_ENTRY)
        add(f"orb{n}", s, m)
    # vwap10
    c, v = C[29], V[29]
    add("vwap10", 1 if (c > v and c > o) else (-1 if (c < v and c < o) else 0), 30)
    # vwap_cross after 10:00, in the direction of the move from the open
    above = C > V
    cross_up = np.zeros(ix.MINUTES, bool)
    cross_dn = np.zeros(ix.MINUTES, bool)
    cross_up[1:] = above[1:] & ~above[:-1] & (C[1:] > o)
    cross_dn[1:] = ~above[1:] & above[:-1] & (C[1:] < o)
    s, m = _first(cross_up, cross_dn, 30, LAST_ENTRY)
    add("vwap_cross", s, m)
    # vwap_pull: last 30 closes all on one side, this bar touches VWAP and closes back on that side
    pl = np.zeros(ix.MINUTES, bool)
    ps = np.zeros(ix.MINUTES, bool)
    for k in range(60, LAST_ENTRY):
        win_c, win_v = C[k - 30:k], V[k - 30:k]
        if (win_c > win_v).all() and L[k] <= V[k] * 1.0005 and C[k] > V[k] and C[k] > o:
            pl[k] = True
        elif (win_c < win_v).all() and H[k] >= V[k] * 0.9995 and C[k] < V[k] and C[k] < o:
            ps[k] = True
    s, m = _first(pl, ps, 60, LAST_ENTRY)
    add("vwap_pull", s, m)
    # noise boundary (first half-hour mark beyond the band)
    if not np.isnan(pc) and not np.isnan(g.noise[i]).any():
        ub = max(o, pc) * (1 + g.noise[i])
        lb = min(o, pc) * (1 - g.noise[i])
        for m in range(30, LAST_ENTRY + 1, 30):
            if C[m - 1] > ub[m - 1]:
                add("noise", 1, m)
                break
            if C[m - 1] < lb[m - 1]:
                add("noise", -1, m)
                break
    # gaps
    if not np.isnan(pc):
        gap = o / pc - 1
        if abs(gap) > 0.003:
            gs = 1 if gap > 0 else -1
            hi30, lo30 = H[:30].max(), L[:30].min()
            if (gs > 0 and C[29] > hi30 * 0.9995) or (gs < 0 and C[29] < lo30 * 1.0005):
                add("gap_go", gs, 30)
            if gs * (C[4] - o) < 0:
                add("gap_fade", -gs, 5)
    # prior day high / low break
    if i > 0:
        pdh, pdl = g.high[i - 1].max(), g.low[i - 1].min()
        s, m = _first(C > pdh, C < pdl, 5, LAST_ENTRY)
        add("pdh", s, m)
    # first-hour momentum at 10:30
    r60 = C[59] / o - 1
    add("momo_1030", 1 if (r60 > 0.003 and C[59] > V[59]) else (-1 if (r60 < -0.003 and C[59] < V[59]) else 0), 60)
    # rsi_dip on 5-minute closes, with the daily trend
    if ctx_row is not None and ctx_row.get("sma20_prev") == ctx_row.get("sma20_prev"):
        up_trend = ctx_row["prev_close"] > ctx_row["sma20_prev"]
        c5s = C[4::5]                                     # close of each 5-min bar (bar 5j+4)
        r = _rsi(c5s, 2)
        for j in range(6, len(c5s)):                      # from 10:00
            m = 5 * j + 5                                 # enter at the next bar's open
            if m > LAST_ENTRY:
                break
            if up_trend and r[j] < 10:
                add("rsi_dip", 1, m)
                break
            if not up_trend and r[j] > 90:
                add("rsi_dip", -1, m)
                break
    return out


SETUPS = ["orb5", "orb15", "orb30", "vwap10", "vwap_cross", "vwap_pull", "noise",
          "gap_go", "gap_fade", "pdh", "momo_1030", "rsi_dip"]


def all_signals(grids: dict[str, ix.DayGrid], ctx: dict) -> dict[str, list[Signal]]:
    out: dict[str, list[Signal]] = {k: [] for k in SETUPS}
    for sym, g in grids.items():
        c = ctx[sym]
        for i in range(ix.LOOKBACK, len(g.days)):
            row = c.loc[g.days[i]] if g.days[i] in c.index else None
            for s in day_signals(g, i, row):
                out[s.strategy].append(s)
    return out
