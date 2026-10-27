"""Fast engine for the opening-range breakout (ORB) model.

The ORB decision is made once a day at 9:35, so instead of scanning every
minute this builds just that one candidate per symbol-day, using the same
scorer (candidates + select) and the same trade simulator as everything
else. That makes it cheap to test many symbols and stop widths.
"""
from __future__ import annotations

import copy
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date, time
from typing import Any

import pandas as pd

from . import backtest as bt
from . import research as rs
from .scorer import candidates, select

ORB_VARIANTS = {
    "long": "mid|with|long|classic|fixedeod|checklist",
    "both": "mid|with|both|classic|fixedeod|checklist",
}


@dataclass
class OrbTrade:
    symbol: str
    day: date
    direction: str
    entry_ref: float          # 9:35 close (fill is the 9:36 open)
    stop: float
    stop_scale: float
    res: bt.TradeResult
    eod_close: float          # session close, to tell noise stop-outs from real losses
    or_range: float

    @property
    def filled(self) -> bool:
        return self.res.filled

    @property
    def r(self) -> float:
        return self.res.r

    def row(self) -> rs.Row:
        hold = ((self.res.exit_time - self.res.fill_time).total_seconds() / 60
                if self.res.filled and self.res.exit_time is not None else 0.0)
        risk = abs((self.res.fill_price or self.entry_ref) - self.stop)
        return rs.Row(self.symbol, self.day, self.res.filled, self.res.r, risk, self.entry_ref, hold,
                      self.direction, "orb_classic", self.res.outcome, 9)

    @property
    def noise_stop(self) -> bool:
        """Stopped out, but the position would have been green at the close."""
        if self.res.outcome != "stop" or not self.res.filled:
            return False
        s = 1 if self.direction == "long" else -1
        return s * (self.eod_close - self.res.fill_price) > 0


def scale_stop(setup, scale: float):
    """Widen (scale > 1) the stop distance; the 10R target moves with it."""
    if scale == 1.0:
        return setup
    s = copy.copy(setup)
    sign = 1 if s.direction == "long" else -1
    risk = sign * (s.entry - s.stop) * scale
    s.stop = round(s.entry - sign * risk, 2)
    s.target = round(s.entry + sign * s.rr * risk, 2)
    return s


def _symbol_job(args):
    symbol, bars, cfg, scales = args
    t = bars.index.time
    days = sorted(set(bars.index.date))
    d_idx = bars.index.date
    out: list[OrbTrade] = []
    for i in range(1, len(days)):
        d, prev = days[i], days[i - 1]
        today = bars[d_idx == d]
        tt = today.index.time
        session = today[(tt >= time(9, 30)) & (tt < time(16, 0))]
        if len(session) < 30:
            continue
        decision = session.index[0] + pd.Timedelta(minutes=5)          # the 9:35 bar
        if decision not in session.index:
            continue
        hist = bars[(d_idx == prev) | ((d_idx == d) & (bars.index <= decision))]
        cands, _ = candidates(hist, cfg, ("mid",))
        cands = [c for c in cands if c.trigger == "orb_classic"]
        s, _ = select(cands, cfg)
        if s is None:
            continue
        after = session[session.index > decision]
        eod = float(session["close"].iloc[-1])
        rng = float(session.iloc[:5]["high"].max() - session.iloc[:5]["low"].min())
        for sc in scales:
            ss = scale_stop(s, sc)
            res = bt.simulate(ss, after, cfg)
            out.append(OrbTrade(symbol, d, ss.direction, ss.entry, ss.stop, sc, res, eod, rng))
    return out


def orb_trades(bars_by_symbol: dict[str, pd.DataFrame], cfg: dict[str, Any], variant: str,
               scales=(1.0,), workers: int | None = None) -> list[OrbTrade]:
    vcfg = rs.merge(cfg, rs.variant(variant).overrides)
    jobs = [(s, b, vcfg, tuple(scales)) for s, b in bars_by_symbol.items()]
    workers = workers if workers is not None else max(1, min(len(jobs), (os.cpu_count() or 2) - 1))
    if workers <= 1 or len(jobs) == 1:
        parts = list(map(_symbol_job, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_symbol_job, jobs))
    out = [t for p in parts for t in p]
    out.sort(key=lambda t: (t.day, t.symbol, t.stop_scale))
    return out
