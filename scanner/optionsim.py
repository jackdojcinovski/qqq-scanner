"""What a signal makes in DOLLARS when traded with real-style option contracts.

Each signal (an underlying entry and the strategy's own exit) is replayed
minute by minute with the option re-priced by Black-Scholes:

  * volatility = the prior 20 days' realised volatility x `iv_mult`
    (options usually price a bit above realised; check against real option
    prices with tools/option_check.py)
  * the clock runs to the expiry's 4 pm, so decay (theta) is built in and
    gets brutal near expiry exactly as it does in real life
  * you pay the ask (mid + half the spread) and sell at the bid
  * Wealthsimple charges $0 per contract on equity/ETF options but converts
    CAD -> USD at 1.5% each way: 1.5% of what you pay plus 1.5% of what you
    get back (about 3% of the premium per round trip)

Take-profit and stop are on the PREMIUM: tp=1.0 means sell when the contract
is worth double what you paid (+100%), sl=0.5 means sell if it loses half.
Stop is checked before target inside a minute (conservative). Whatever is
left exits with the strategy (its own stop / trail / the close).

Regime for shorts: a day is "down" when, using only the days before it, the
close is under its 20-day average AND the last 3 days fell more than
k x the normal 3-day move. Longs are the default; shorts only on down days.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .intraday import MINUTES, DayGrid

RATE = 0.04
YEAR_MIN = 365 * 24 * 60
FX = 0.015                      # Wealthsimple CAD account: 1.5% each way


# ---------------------------------------------------------------- pricing

def _erf(x):
    """Vectorised erf (Abramowitz & Stegun 7.1.26, error < 1.5e-7)."""
    x = np.asarray(x, dtype=float)
    s = np.sign(x)
    a = np.abs(x)
    t = 1 / (1 + 0.3275911 * a)
    y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
             + 0.254829592) * t * np.exp(-a * a)
    return s * y


def _ncdf(x):
    return 0.5 * (1 + _erf(np.asarray(x, dtype=float) / math.sqrt(2)))


def bs_price(S, K: float, T, sigma: float, right: str, r: float = RATE):
    """Black-Scholes price; S and T may be arrays. T in years (floored at 1 minute)."""
    S = np.asarray(S, dtype=float)
    T = np.maximum(np.asarray(T, dtype=float), 1 / YEAR_MIN)
    sq = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / sq
    d2 = d1 - sq
    disc = K * np.exp(-r * T)
    if right == "C":
        return S * _ncdf(d1) - disc * _ncdf(d2)
    return disc * _ncdf(-d2) - S * _ncdf(-d1)


def half_spread(premium: float) -> float:
    """Half the bid/ask: QQQ/SPY near-the-money ~$0.01-0.02, deep ITM ~0.5% of the price."""
    return max(0.01, 0.005 * premium)


@dataclass(frozen=True)
class Contract:
    name: str
    dte: int                # trading days to expiry (0 = expires today)
    itm_pct: float = 0.0    # strike this % in the money (0 = at the money)


CONTRACTS = [
    Contract("0DTE ATM", 0),
    Contract("1DTE ATM", 1),
    Contract("2DTE ATM", 2),
    Contract("weekly ATM", 5),
    Contract("weekly 1% ITM", 5, 1.0),
    Contract("monthly 3% ITM", 20, 3.0),
]


def expiry_date(day: date, dte: int) -> date:
    d, n = day, 0
    while n < dte:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n += 1
    return d


# ---------------------------------------------------------------- daily context

def daily_context(g: DayGrid, k_values=(0.5, 1.0)) -> pd.DataFrame:
    """Per day, from PRIOR days only: realised vol, 20-day average, 3-day move, down-regime flags."""
    close = pd.Series(g.close[:, -1], index=g.days)
    ret = np.log(close).diff()
    vol20 = ret.rolling(20).std().shift(1)                       # daily, prior 20 days
    sma20 = close.rolling(20).mean().shift(1)
    prev = close.shift(1)
    ret3 = (close.shift(1) / close.shift(4) - 1)
    out = pd.DataFrame({"prev_close": prev, "sma20_prev": sma20, "ret3_prev": ret3,
                        "vol20_prev": vol20, "ann_vol": vol20 * math.sqrt(252)})
    for k in k_values:
        out[f"down_k{k:g}"] = (prev < sma20) & (ret3 < -k * vol20 * math.sqrt(3))
    return out


# ---------------------------------------------------------------- signals and replay

@dataclass
class Signal:
    strategy: str
    symbol: str
    day: date
    direction: str
    m_in: int               # fill at the open of this bar
    m_end: int              # bars m_in .. m_end-1 are watched for target / stop
    exit_px: float          # strategy's exit price on the underlying
    exit_after_min: float   # minutes after 9:30 when the strategy exit happens

    @property
    def right(self) -> str:
        return "C" if self.direction == "long" else "P"


@dataclass
class Path:
    """Option bid/ask path of one signal with one contract (pre-computed once, reused for every tp/sl)."""
    paid: float                     # ask at entry
    worst_bid: np.ndarray           # per watched bar
    best_bid: np.ndarray
    open_bid: np.ndarray
    final_bid: float                # at the strategy exit
    strike: float
    expiry: date


@dataclass
class Result:
    signal: Signal
    contract: str
    paid: float
    got: float
    outcome: str            # target / stop / strategy
    exit_bar: int

    @property
    def ret(self) -> float:
        return (self.got - self.paid) / self.paid

    def pnl(self, contracts: int = 5, fx: float = FX) -> float:
        """$ after the spread (already in paid/got) and the FX fee, for `contracts` contracts."""
        return contracts * 100 * ((self.got - self.paid) - fx * (self.paid + self.got))


def _bar_time(day: date, minutes_after_open: float) -> pd.Timestamp:
    return pd.Timestamp(day) + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=minutes_after_open)


def make_path(g: DayGrid, i: int, sig: Signal, c: Contract, sigma: float) -> Path | None:
    if not (sigma > 0) or sig.m_end < sig.m_in:
        return None
    S0 = float(g.open[i, sig.m_in])
    right = sig.right
    ref = S0 * (1 - c.itm_pct / 100) if right == "C" else S0 * (1 + c.itm_pct / 100)
    K = float(round(ref))
    exp = expiry_date(sig.day, c.dte)
    exp_t = pd.Timestamp(exp) + pd.Timedelta(hours=16)

    t0 = (exp_t - _bar_time(sig.day, 0)).total_seconds() / 60 / YEAR_MIN    # years left at 9:30

    def T(minutes_after_open):
        return t0 - np.asarray(minutes_after_open, dtype=float) / YEAR_MIN

    mid0 = float(bs_price(S0, K, T(sig.m_in), sigma, right))
    if mid0 <= 0.05:
        return None
    paid = mid0 + half_spread(mid0)
    bars = np.arange(sig.m_in, sig.m_end)
    if len(bars):
        t_open, t_mid = T(bars), T(bars + 1)          # high/low counted at the bar's end (conservative on decay)
        hi, lo, op = g.high[i, bars], g.low[i, bars], g.open[i, bars]
        best_S, worst_S = (hi, lo) if right == "C" else (lo, hi)
        best = bs_price(best_S, K, t_mid, sigma, right)
        worst = bs_price(worst_S, K, t_mid, sigma, right)
        opn = bs_price(op, K, t_open, sigma, right)
        best_bid = best - np.maximum(0.01, 0.005 * best)
        worst_bid = worst - np.maximum(0.01, 0.005 * worst)
        open_bid = opn - np.maximum(0.01, 0.005 * opn)
    else:
        best_bid = worst_bid = open_bid = np.array([])
    fmid = float(bs_price(sig.exit_px, K, T(sig.exit_after_min), sigma, right))
    final_bid = max(fmid - half_spread(fmid), 0.0)
    return Path(paid, np.maximum(worst_bid, 0), np.maximum(best_bid, 0), np.maximum(open_bid, 0),
                final_bid, K, exp)


def replay(path: Path, tp: float | None, sl: float | None) -> tuple[float, str, int]:
    """(price received, outcome, bar offset) for one take-profit / stop on the premium."""
    target = path.paid * (1 + tp) if tp is not None else math.inf
    stop = path.paid * (1 - sl) if sl is not None else -math.inf
    n = len(path.best_bid)
    if n:
        hit_s = np.flatnonzero(path.worst_bid <= stop)
        hit_t = np.flatnonzero(path.best_bid >= target)
        js = hit_s[0] if len(hit_s) else n
        jt = hit_t[0] if len(hit_t) else n
        if js < n and js <= jt:                                   # stop first when both in the same bar
            return min(stop, path.open_bid[js]), "stop", int(js)
        if jt < n:
            return max(target, path.open_bid[jt]), "target", int(jt)
    return path.final_bid, "strategy", n


def run(paths: list[tuple[Signal, Path]], contract: str, tp: float | None, sl: float | None) -> list[Result]:
    out = []
    for sig, p in paths:
        got, why, j = replay(p, tp, sl)
        out.append(Result(sig, contract, p.paid, got, why, sig.m_in + j))
    return out


# ---------------------------------------------------------------- converters

def from_intraday(trades, grids: dict[str, DayGrid]) -> list[Signal]:
    out = []
    for t in trades:
        if t.exit_min >= MINUTES:          # held to the close
            m_end, after = MINUTES, MINUTES - 0.5          # sell at 15:59:30
        else:                              # trailing exit at the open of bar exit_min
            m_end, after = t.exit_min, t.exit_min
        out.append(Signal(t.strategy, t.symbol, t.day, t.direction, t.entry_min, m_end, t.exit, after))
    return out


def from_orb(orb_trades, name: str = "orb") -> list[Signal]:
    out = []
    for t in orb_trades:
        if not t.filled or t.res.exit_time is None:
            continue
        open_ts = t.res.fill_time.normalize() + pd.Timedelta(hours=9, minutes=30)
        m_in = int((t.res.fill_time - open_ts).total_seconds() // 60)
        m_x = int((t.res.exit_time - open_ts).total_seconds() // 60)
        m_x = min(m_x, MINUTES - 1)
        # the exit bar itself is not watched (its stop could come before its high); sell at its end
        out.append(Signal(name, t.symbol, t.day, t.direction, m_in, m_x, float(t.res.exit_price),
                          min(m_x + 1, MINUTES - 0.5)))
    return out


def one_per_day(signals: list[Signal], prefer: str = "QQQ") -> list[Signal]:
    best: dict[date, Signal] = {}
    for s in sorted(signals, key=lambda s: (s.day, s.m_in, s.symbol != prefer)):
        best.setdefault(s.day, s)
    return [best[d] for d in sorted(best)]


def apply_regime(signals: list[Signal], ctx: dict[str, pd.DataFrame], mode: str) -> list[Signal]:
    """mode: 'long' (longs only), 'long_up' (longs only, skip down days),
    'regime_k<k>' (longs on normal days, shorts only on down days)."""
    out = []
    for s in signals:
        c = ctx[s.symbol]
        if s.day not in c.index:
            continue
        if mode == "long":
            ok = s.direction == "long"
        else:
            k = mode.split("_k")[-1] if "_k" in mode else "1"
            down = bool(c.at[s.day, f"down_k{k}"])
            ok = (s.direction == "long" and not down) or (mode.startswith("regime") and s.direction == "short" and down)
        if ok:
            out.append(s)
    return out


def build_paths(signals: list[Signal], grids: dict[str, DayGrid], ctx: dict[str, pd.DataFrame],
                c: Contract, iv_mult: float = 1.1) -> list[tuple[Signal, Path]]:
    pos = {s: {d: i for i, d in enumerate(g.days)} for s, g in grids.items()}
    out = []
    for sig in signals:
        i = pos[sig.symbol].get(sig.day)
        if i is None or sig.day not in ctx[sig.symbol].index:
            continue
        v = ctx[sig.symbol].at[sig.day, "ann_vol"]
        if not (v == v):           # nan
            continue
        p = make_path(grids[sig.symbol], i, sig, c, float(v) * iv_mult)
        if p is not None:
            out.append((sig, p))
    return out
