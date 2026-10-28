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
    out += ict_signals(g, i, ctx_row)
    return out


ICT_END = 90              # ICT-style entries only until 11:00


def _bars5(x: np.ndarray, how: str) -> np.ndarray:
    r = x[: (len(x) // 5) * 5].reshape(-1, 5)
    return {"open": r[:, 0], "high": r.max(axis=1), "low": r.min(axis=1), "close": r[:, -1]}[how]


def ict_signals(g: ix.DayGrid, i: int, ctx_row=None) -> list[Signal]:
    """Day-trading entries in the ICT style, mechanical and on closed bars only.

    silver_bullet  10:00-11:00: a liquidity sweep (wick through the last 30 min's swing low/high, close back
                   inside), then a displacement candle leaving a fair value gap (5-min); enter when price
                   pulls back into the gap and holds it
    judas_pd       first hour: price runs through yesterday's low (high), then a 5-min close back above the
                   level AND above the sweep candle's high (market structure shift) -> reverse
    judas_or       same, but the liquidity is the first 15 minutes' high / low (failed breakout)
    orb_retest     5-min close beyond the first 15 minutes' range, then the first 1-min retest of that edge
                   that holds -> go with the breakout
    first_pull     strong opening drive (9:30-9:45), a pullback of 30%+ that stays on the VWAP side, then a
                   new high (low) of day -> continuation
    """
    H, L, C, O, V = g.high[i], g.low[i], g.close[i], g.open[i], g.vwap[i]
    o = g.day_open[i]
    O5, H5, L5, C5 = (_bars5(x, k) for x, k in ((O, "open"), (H, "high"), (L, "low"), (C, "close")))
    out: list[Signal] = []

    def add(name, s, m):
        if s and 1 <= m <= ICT_END:
            x = _sig(g, i, name, s, m)
            if x:
                out.append(x)

    body = np.abs(C5 - O5)
    # silver bullet: FVG bar k completes between 10:00 and 10:55 (5-min bars 5..16)
    best = None
    for k in range(5, 17):
        for s in (1, -1):
            if s > 0:
                fvg = L5[k] > H5[k - 2]
                disp = C5[k - 1] > O5[k - 1]
                top, bot = L5[k], H5[k - 2]
            else:
                fvg = H5[k] < L5[k - 2]
                disp = C5[k - 1] < O5[k - 1]
                top, bot = H5[k], L5[k - 2]
            if not (fvg and disp and body[k - 1] >= 1.5 * np.median(body[:k - 1])):
                continue
            swept = False
            for j in range(max(1, k - 6), k - 1):              # sweep before the displacement
                swing = L5[max(0, j - 6):j].min() if s > 0 else H5[max(0, j - 6):j].max()
                if (s > 0 and L5[j] < swing and C5[j] > swing) or (s < 0 and H5[j] > swing and C5[j] < swing):
                    swept = True
                    break
            if not swept:
                continue
            for t in range(5 * k + 5, ICT_END):                   # first 1-min retrace into the gap that holds
                if (s > 0 and L[t] <= top and C[t] >= bot) or (s < 0 and H[t] >= top and C[t] <= bot):
                    if best is None or t + 1 < best[1]:
                        best = (s, t + 1)
                    break
            break
    if best:
        add("silver_bullet", *best)

    # judas swing through yesterday's high / low, or the first-15-minute range
    if i > 0:
        levels = {"judas_pd": (g.high[i - 1].max(), g.low[i - 1].min(), 0),
                  "judas_or": (H5[:3].max(), L5[:3].min(), 3)}
        for name, (hi, lo, j0) in levels.items():
            done = False
            for j in range(j0, 12):                               # sweep bar in the first hour
                for k in range(j, min(j + 4, 17)):
                    if L5[j] < lo and C5[k] > lo and C5[k] > H5[j]:
                        add(name, 1, 5 * k + 5)
                        done = True
                    elif H5[j] > hi and C5[k] < hi and C5[k] < L5[j]:
                        add(name, -1, 5 * k + 5)
                        done = True
                    if done:
                        break
                if done:
                    break

    # opening-range retest
    orh, orl = H5[:3].max(), L5[:3].min()
    for j in range(3, 17):
        s = 1 if C5[j] > orh else (-1 if C5[j] < orl else 0)
        if s:
            edge = orh if s > 0 else orl
            for t in range(5 * j + 5, ICT_END):
                if (s > 0 and L[t] <= edge * 1.0005 and C[t] > edge) or (s < 0 and H[t] >= edge * 0.9995 and C[t] < edge):
                    add("orb_retest", s, t + 1)
                    break
            break

    # first pullback after the opening drive
    sig = float(ctx_row["vol20_prev"]) if ctx_row is not None and ctx_row.get("vol20_prev") == ctx_row.get("vol20_prev") \
        else None
    if sig:
        drive = C[14] / o - 1
        s = 1 if drive >= 0.4 * sig else (-1 if drive <= -0.4 * sig else 0)
        if s:
            ext = H[:15].max() if s > 0 else L[:15].min()
            pulled = False
            far = ext
            for t in range(15, ICT_END - 1):
                if s > 0:
                    far = min(far, L[t])
                    if ext - far >= 0.3 * (ext - o) and L[t] > V[t]:
                        pulled = True
                    if pulled and C[t] > ext:
                        add("first_pull", 1, t + 1)
                        break
                    if L[t] <= V[t]:
                        break                                     # lost VWAP: no longer a clean pullback
                else:
                    far = max(far, H[t])
                    if far - ext >= 0.3 * (o - ext) and H[t] < V[t]:
                        pulled = True
                    if pulled and C[t] < ext:
                        add("first_pull", -1, t + 1)
                        break
                    if H[t] >= V[t]:
                        break
    return out


SETUPS = ["orb5", "orb15", "orb30", "vwap10", "vwap_cross", "vwap_pull", "noise",
          "gap_go", "gap_fade", "pdh", "momo_1030", "rsi_dip",
          "silver_bullet", "judas_pd", "judas_or", "orb_retest", "first_pull"]


def all_signals(grids: dict[str, ix.DayGrid], ctx: dict) -> dict[str, list[Signal]]:
    out: dict[str, list[Signal]] = {k: [] for k in SETUPS}
    for sym, g in grids.items():
        c = ctx[sym]
        for i in range(ix.LOOKBACK, len(g.days)):
            row = c.loc[g.days[i]] if g.days[i] in c.index else None
            for s in day_signals(g, i, row):
                out[s.strategy].append(s)
    return out
