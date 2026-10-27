"""Intraday quant strategies on QQQ / SPY that hold for hours, not minutes.

All three decide on a finished bar and enter at the NEXT bar's open, so
nothing uses data it would not have had live. Each makes at most one trade
per symbol-day (the first signal), which is what a 1-trade-a-day alert needs.

noise   Noise-boundary momentum (Zarattini, Aziz & Barbon 2024, "Beat the
        Market"). For each minute of the day, the average absolute move from
        the open over the last 14 days is the "noise". Upper band =
        max(open, prior close) x (1 + noise), lower = min(...) x (1 - noise).
        Checked every 30 min from 10:00: close above the upper band -> long,
        below the lower -> short. Trailing stop at max(band, VWAP) (longs),
        checked at the same half-hour marks. Otherwise out at the close.

lastHalf  Intraday momentum (Gao, Han, Li & Zhou 2018, JFE; Baltussen et
        al. 2021). The return from the prior close to 10:00 ("first30") or
        to 15:30 ("day") predicts the last half hour. Enter 15:30 in that
        direction, out at the close. Only when the move is big vs its noise.

vwap    VWAP trend. Checked every 30 min from 10:30 (or 11:00) to 14:00:
        price above VWAP, VWAP rising over the last 30 min, and through the
        first-30-minute high -> long (mirror for shorts). Trailing stop at
        VWAP on the half-hour marks, else out at the close.

R: profit divided by the distance from entry to the stop in force at entry
(floored at `min_risk_pct` of price so a hair-thin first stop can't blow up
R). The research CostModel then turns R into option P&L.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, time

import numpy as np
import pandas as pd

from . import research as rs

MINUTES = 390                     # 9:30 .. 15:59 bars
LOOKBACK = 14                     # days for the noise estimate


# ---------------------------------------------------------------- day grid

@dataclass
class DayGrid:
    """One symbol's regular sessions as days x 390 arrays (bar m starts 9:30+m)."""
    symbol: str
    days: list[date]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    vwap: np.ndarray
    day_open: np.ndarray          # first bar's open
    prev_close: np.ndarray        # prior session's last close (nan on the first day)
    noise: np.ndarray             # mean |close/day_open - 1| at each minute, prior LOOKBACK days
    last30_noise: np.ndarray      # mean |close[389]/close[359] - 1| over prior LOOKBACK days
    tz: object = field(default=None, repr=False)
    volume: np.ndarray | None = field(default=None, repr=False)     # days x 390 (0 where no bar)

    def ts(self, i: int, m: int) -> pd.Timestamp:
        d = self.days[i]
        return pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=9, minute=30, tz=self.tz) \
            + pd.Timedelta(minutes=m)


def build_grid(symbol: str, bars: pd.DataFrame, min_bars: int = 300) -> DayGrid:
    t = bars.index.time
    rth = bars[(t >= time(9, 30)) & (t < time(16, 0))]
    tz = rth.index.tz
    mins = (rth.index.hour * 60 + rth.index.minute - 570).to_numpy()
    dates = rth.index.date
    uniq = sorted(set(dates))
    counts = pd.Series(dates).value_counts()
    keep = [d for d in uniq if counts.get(d, 0) >= min_bars]       # skip half days / broken data
    pos = {d: i for i, d in enumerate(keep)}
    n = len(keep)
    O, H, L, C, V = (np.full((n, MINUTES), np.nan) for _ in range(5))
    row = np.array([pos.get(d, -1) for d in dates])
    ok = row >= 0
    r, m = row[ok], mins[ok]
    for arr, col in ((O, "open"), (H, "high"), (L, "low"), (C, "close"), (V, "volume")):
        arr[r, m] = rth[col].to_numpy()[ok]
    # fill missing minutes: carry the last close, a flat zero-volume bar
    Cf = pd.DataFrame(C).ffill(axis=1).bfill(axis=1).to_numpy()
    miss = np.isnan(O)
    O[miss], H[miss], L[miss] = Cf[miss], Cf[miss], Cf[miss]
    C = Cf
    V = np.nan_to_num(V)
    typical = (H + L + C) / 3
    cv = np.cumsum(V, axis=1)
    vwap = np.where(cv > 0, np.cumsum(typical * V, axis=1) / np.where(cv > 0, cv, 1), C)
    day_open = O[:, 0]
    prev_close = np.concatenate([[np.nan], C[:-1, -1]])
    move = np.abs(C / day_open[:, None] - 1)
    last30 = np.abs(C[:, 389] / C[:, 359] - 1)
    noise = np.full_like(move, np.nan)
    l30 = np.full(n, np.nan)
    for i in range(LOOKBACK, n):
        noise[i] = move[i - LOOKBACK:i].mean(axis=0)
        l30[i] = last30[i - LOOKBACK:i].mean()
    return DayGrid(symbol, keep, O, H, L, C, vwap, day_open, prev_close, noise, l30, tz, V)


# ---------------------------------------------------------------- trades

@dataclass
class Trade:
    strategy: str                 # variant name
    symbol: str
    day: date
    direction: str
    entry: float
    exit: float
    risk: float
    entry_min: int                # bar index of the fill (0 = 9:30)
    exit_min: int                 # bar index of the exit (fill at its open, or 390 = the close)
    outcome: str                  # close / trail / stop
    fill_time: pd.Timestamp
    exit_time: pd.Timestamp

    @property
    def sign(self) -> int:
        return 1 if self.direction == "long" else -1

    @property
    def r(self) -> float:
        return self.sign * (self.exit - self.entry) / self.risk

    @property
    def filled(self) -> bool:
        return True

    @property
    def hold_min(self) -> float:
        return float(self.exit_min - self.entry_min)

    def row(self) -> rs.Row:
        return rs.Row(self.symbol, self.day, True, self.r, self.risk, self.entry, self.hold_min,
                      self.direction, self.strategy.split("|")[0], self.outcome, 9 + (30 + self.entry_min) // 60)


def _floor_risk(entry: float, dist: float, min_risk_pct: float) -> float:
    return max(dist, entry * min_risk_pct)


def _exit_close(g: DayGrid, i: int) -> tuple[float, int]:
    return float(g.close[i, MINUTES - 1]), MINUTES


def _make(g, i, name, direction, entry, risk, m_in, exit_px, m_out, outcome) -> Trade:
    exit_ts = g.ts(i, m_out) if m_out < MINUTES else g.ts(i, MINUTES - 1) + pd.Timedelta(minutes=1)
    return Trade(name, g.symbol, g.days[i], direction, float(entry), float(exit_px), float(risk),
                 m_in, m_out, outcome, g.ts(i, m_in), exit_ts)


def _trail(g: DayGrid, i: int, s: int, m_in: int, step: int, level) -> tuple[float, int, str]:
    """Half-hour checkpoint trailing exit: at each mark k after entry, if the
    close of bar k-1 is on the wrong side of level(k-1), exit at bar k's open."""
    for k in range(m_in + step, MINUTES, step):
        if s * (g.close[i, k - 1] - level(k - 1)) < 0:
            return float(g.open[i, k]), k, "trail"
    px, m = _exit_close(g, i)
    return px, m, "close"


def noise_trades(g: DayGrid, sides: str = "both", band_mult: float = 1.0, stop: str = "band_vwap",
                 step: int = 30, first: int = 30, last_entry: int = 330,
                 min_risk_pct: float = 0.001, name: str = "noise") -> list[Trade]:
    out = []
    for i in range(LOOKBACK, len(g.days)):
        if np.isnan(g.prev_close[i]) or np.isnan(g.noise[i]).any():
            continue
        o, pc = g.day_open[i], g.prev_close[i]
        ub = max(o, pc) * (1 + band_mult * g.noise[i])
        lb = min(o, pc) * (1 - band_mult * g.noise[i])
        for m in range(first, last_entry + 1, step):
            c = g.close[i, m - 1]
            s = 1 if c > ub[m - 1] else (-1 if c < lb[m - 1] else 0)
            if s == 0 or (s < 0 and sides == "long"):
                continue
            band = ub if s > 0 else lb
            if stop == "vwap":
                level = lambda k: g.vwap[i, k]                                       # noqa: E731
            else:
                pick = max if s > 0 else min
                level = lambda k: pick(band[k], g.vwap[i, k])                         # noqa: E731
            entry = g.open[i, m]
            risk = _floor_risk(entry, s * (entry - level(m - 1)), min_risk_pct)
            px, mo, why = _trail(g, i, s, m, step, level)
            out.append(_make(g, i, name, "long" if s > 0 else "short", entry, risk, m, px, mo, why))
            break
    return out


def last_half_trades(g: DayGrid, signal: str = "first30", min_noise: float = 1.0, sides: str = "both",
                     stop_r: float | None = None, name: str = "lasthalf") -> list[Trade]:
    """Enter at 15:30 (bar 360) in the direction of the early/day move, out at the close."""
    out = []
    for i in range(LOOKBACK, len(g.days)):
        pc = g.prev_close[i]
        if np.isnan(pc) or np.isnan(g.last30_noise[i]):
            continue
        mark = 29 if signal == "first30" else 359                    # close of 9:59 or 15:29 bar
        ret = g.close[i, mark] / pc - 1
        if abs(ret) < min_noise * g.noise[i, mark]:
            continue
        s = 1 if ret > 0 else -1
        if s < 0 and sides == "long":
            continue
        entry = g.open[i, 360]
        risk = entry * max(g.last30_noise[i], 0.0005)
        px, mo, why = _exit_close(g, i)[0], MINUTES, "close"
        if stop_r is not None:
            stop_px = entry - s * stop_r * risk
            for k in range(360, MINUTES):
                if (s > 0 and g.low[i, k] <= stop_px) or (s < 0 and g.high[i, k] >= stop_px):
                    gap = s * (g.open[i, k] - stop_px) <= 0
                    px, mo, why = (g.open[i, k] if gap else stop_px), k, "stop"
                    break
        out.append(_make(g, i, name, "long" if s > 0 else "short", entry, risk, 360, px, mo, why))
    return out


def vwap_trades(g: DayGrid, sides: str = "both", first: int = 30, last_entry: int = 270, step: int = 30,
                min_risk_pct: float = 0.001, name: str = "vwap") -> list[Trade]:
    out = []
    for i in range(LOOKBACK, len(g.days)):
        hi30, lo30 = g.high[i, :30].max(), g.low[i, :30].min()
        for m in range(max(first, 30), last_entry + 1, step):
            c, v, v_then = g.close[i, m - 1], g.vwap[i, m - 1], g.vwap[i, m - 30]      # VWAP 29 min earlier
            s = 0
            if c > v and v > v_then and c > hi30:
                s = 1
            elif c < v and v < v_then and c < lo30:
                s = -1
            if s == 0 or (s < 0 and sides == "long"):
                continue
            level = lambda k: g.vwap[i, k]                                           # noqa: E731
            entry = g.open[i, m]
            risk = _floor_risk(entry, s * (entry - v), min_risk_pct)
            px, mo, why = _trail(g, i, s, m, step, level)
            out.append(_make(g, i, name, "long" if s > 0 else "short", entry, risk, m, px, mo, why))
            break
    return out


# ---------------------------------------------------------------- variants

def variants() -> dict[str, tuple]:
    """name -> (function, kwargs). Kept small on purpose: fewer knobs, less overfitting."""
    v: dict[str, tuple] = {}
    for sides in ("both", "long"):
        for bm in (1.0, 1.5):
            for stop in ("band_vwap", "vwap"):
                n = f"noise|{sides}|band{bm:g}|{stop}"
                v[n] = (noise_trades, dict(sides=sides, band_mult=bm, stop=stop, name=n))
        for sig in ("first30", "day"):
            for k in (0.0, 1.0):
                n = f"lasthalf|{sides}|{sig}|noise{k:g}"
                v[n] = (last_half_trades, dict(signal=sig, min_noise=k, sides=sides, stop_r=None, name=n))
        for first in (60, 90):          # 10:00 can never clear the 9:30-10:00 range itself
            n = f"vwap|{sides}|from{first}"
            v[n] = (vwap_trades, dict(sides=sides, first=first, name=n))
    return v


def run_all(grids: dict[str, DayGrid], names: list[str] | None = None) -> dict[str, list[Trade]]:
    vs = variants()
    out = {}
    for name in names or list(vs):
        fn, kw = vs[name]
        trades = [t for g in grids.values() for t in fn(g, **kw)]
        trades.sort(key=lambda t: (t.day, t.entry_min, t.symbol))
        out[name] = trades
    return out


def one_per_day(trades: list[Trade], prefer: str = "QQQ") -> list[Trade]:
    """First signal of the day wins (that's the alert you'd act on); ties -> QQQ."""
    best: dict[date, Trade] = {}
    for t in sorted(trades, key=lambda t: (t.day, t.entry_min, t.symbol != prefer)):
        best.setdefault(t.day, t)
    return [best[d] for d in sorted(best)]


def dollars(t: Trade | rs.Row, cm: rs.CostModel) -> float:
    """Approximate $ P&L for 1 option contract (100 shares x delta), after costs."""
    row = t.row() if isinstance(t, Trade) else t
    move = row.r * row.risk                                   # $ per share of the underlying
    cost = cm.spread + cm.theta_pct_per_hour * row.price * row.hold_min / 60
    return 100 * (cm.delta * move - cost)
