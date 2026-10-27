"""Backtest: replay every day through the scorer and play each setup forward.

Every setup that reaches the log band is simulated, so the report can compare
score bands and components. Separately, `taken` marks the trades you would
actually have received live: alerts only, at most `max_alerts_per_day`, none
after a losing trade, and one position at a time.

Results are in R (multiples of the risk to the stop) on the QQQ share price.
Option P&L will differ: delta, time decay and spreads are not modelled.
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import date, time
from typing import Any, Iterable

import pandas as pd

from . import indicators as ind
from .config import parse_time
from .scorer import Setup, evaluate


# --------------------------------------------------------------------------- trade simulation

@dataclass
class TradeResult:
    outcome: str                 # target / stop / time / close / no fill / ran away
    r: float = 0.0
    fill_time: pd.Timestamp | None = None
    fill_price: float | None = None
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None

    @property
    def filled(self) -> bool:
        return self.fill_time is not None


def simulate(setup: Setup, after: pd.DataFrame, cfg: dict[str, Any]) -> TradeResult:
    """Play a setup forward on the bars that come after it (same session).

    Conservative assumptions: a bar that touches both stop and target counts
    as a stop; a bar that fills the entry and touches the stop is a stop; a
    target touched in the fill bar is ignored until the next bar; stops that
    gap through exit at the bar's open.
    """
    t = cfg["trade"]
    long = setup.direction == "long"
    entry, stop, target = setup.entry, setup.stop, setup.target
    valid_until = setup.time + pd.Timedelta(minutes=t["entry_valid_minutes"])
    time_exit = pd.Timedelta(minutes=t["time_exit_minutes"])

    fill_t = fill_px = None
    for ts, o, h, lo, c in after[["open", "high", "low", "close"]].itertuples():
        if fill_t is None:
            if ts > valid_until:
                return TradeResult("no fill")
            touched = lo <= entry if long else h >= entry
            if not touched:
                ran = h >= target if long else lo <= target
                if ran:
                    return TradeResult("ran away")
                continue
            fill_t = ts
            fill_px = min(o, entry) if long else max(o, entry)
            if (lo <= stop) if long else (h >= stop):
                return _close(fill_t, fill_px, stop, "stop", ts, setup)
            continue

        if (lo <= stop) if long else (h >= stop):
            px = min(o, stop) if long else max(o, stop)
            return _close(fill_t, fill_px, px, "stop", ts, setup)
        if (h >= target) if long else (lo <= target):
            px = max(o, target) if long else min(o, target)
            return _close(fill_t, fill_px, px, "target", ts, setup)
        if ts - fill_t >= time_exit:
            return _close(fill_t, fill_px, c, "time", ts, setup)

    if fill_t is None:
        return TradeResult("no fill")
    last = after.iloc[-1]
    return _close(fill_t, fill_px, float(last["close"]), "close", after.index[-1], setup)


def _close(fill_t, fill_px, exit_px, outcome, exit_t, setup: Setup) -> TradeResult:
    sign = 1 if setup.direction == "long" else -1
    risk = abs(fill_px - setup.stop)
    r = sign * (exit_px - fill_px) / risk if risk > 0 else 0.0
    return TradeResult(outcome, round(r, 3), fill_t, float(fill_px), exit_t, float(exit_px))


# --------------------------------------------------------------------------- day runner

@dataclass
class TradeRecord:
    day: date
    time: pd.Timestamp
    key: str
    direction: str
    score: int
    band: str
    taken: bool
    skip_reason: str
    entry: float
    stop: float
    target: float
    target_name: str
    rr: float
    entry_zone: str
    outcome: str
    r: float
    filled: bool
    fill_time: pd.Timestamp | None
    exit_time: pd.Timestamp | None
    components: dict[str, int] = field(default_factory=dict)


def run_day(history: pd.DataFrame, day: date, cfg: dict[str, Any]) -> list[TradeRecord]:
    """`history` holds at least the prior session and all of `day`."""
    w = cfg["window"]
    start, end = parse_time(w["start"]), parse_time(w["end"])
    lim = cfg["limits"]
    session = ind.regular_session(history)
    day_bars = session[session.index.date == day]
    if day_bars.empty:
        return []
    pos = {ts: i for i, ts in enumerate(history.index)}
    # the scorer sees no daily limits here; they're applied below to decide `taken`
    eval_cfg = {**cfg, "limits": {**lim, "max_alerts_per_day": 10**6, "stop_after_losses": 10**6}}

    records: dict[str, TradeRecord] = {}
    alerts = losses = 0
    busy_until: pd.Timestamp | None = None
    settled: set[str] = set()

    for ts in day_bars.index:
        if not (start <= ts.time() <= end):
            continue
        # settle taken trades that have finished by now
        for k, rec in records.items():
            if rec.taken and k not in settled and rec.exit_time is not None and rec.exit_time <= ts:
                settled.add(k)
                if rec.r < 0:
                    losses += 1
        ev = evaluate(history.iloc[:pos[ts] + 1], eval_cfg)
        s = ev.setup
        if s is None:
            continue
        band = s.band(cfg)
        if band == "ignore":
            continue
        prev = records.get(s.key)
        if prev is not None and not (prev.band == "log" and band == "alert"):
            continue

        after = day_bars[day_bars.index > ts]
        res = simulate(s, after, cfg)

        skip = ""
        if band != "alert":
            skip = "score below alert"
        elif alerts >= lim["max_alerts_per_day"]:
            skip = "daily alert limit"
        elif losses >= lim["stop_after_losses"]:
            skip = "stopped after a loss"
        elif busy_until is not None and ts < busy_until:
            skip = "trade already open"
        taken = skip == ""
        if taken:
            alerts += 1
            valid = ts + pd.Timedelta(minutes=cfg["trade"]["entry_valid_minutes"])
            busy_until = res.exit_time if res.filled else valid

        records[s.key] = TradeRecord(
            day=day, time=ts, key=s.key, direction=s.direction, score=s.score, band=band,
            taken=taken, skip_reason=skip, entry=s.entry, stop=s.stop, target=s.target,
            target_name=s.target_name, rr=s.rr, entry_zone=s.entry_zone,
            outcome=res.outcome, r=res.r, filled=res.filled,
            fill_time=res.fill_time, exit_time=res.exit_time,
            components={c.name: c.points for c in s.components},
        )
    return list(records.values())


def day_slices(bars: pd.DataFrame, days: Iterable[date]) -> list[tuple[date, pd.DataFrame]]:
    """For each day, the bars from the previous trading day through that day."""
    all_days = sorted(set(bars.index.date))
    want = set(days)
    out = []
    for i, d in enumerate(all_days):
        if d not in want or i == 0:
            continue
        prev = all_days[i - 1]
        d_idx = bars.index.date
        out.append((d, bars[(d_idx == prev) | (d_idx == d)]))
    return out


def _run_one(args):
    d, hist, cfg = args
    return run_day(hist, d, cfg)


def run_backtest(bars: pd.DataFrame, days: Iterable[date], cfg: dict[str, Any],
                 workers: int | None = None, progress=None) -> list[TradeRecord]:
    jobs = [(d, h, cfg) for d, h in day_slices(bars, days)]
    workers = workers if workers is not None else max(1, (os.cpu_count() or 2) - 1)
    out: list[TradeRecord] = []
    if workers <= 1:
        for n, j in enumerate(jobs, 1):
            out += _run_one(j)
            if progress:
                progress(n, len(jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for n, recs in enumerate(pool.map(_run_one, jobs, chunksize=4), 1):
                out += recs
                if progress:
                    progress(n, len(jobs))
    out.sort(key=lambda r: r.time)
    return out


# --------------------------------------------------------------------------- statistics

@dataclass
class Stats:
    setups: int
    filled: int
    wins: int
    win_rate: float
    avg_r: float
    total_r: float
    profit_factor: float
    max_losing_streak: int
    max_drawdown_r: float


def stats(records: list[TradeRecord]) -> Stats:
    filled = [r for r in records if r.filled]
    rs = [r.r for r in filled]
    wins = sum(1 for x in rs if x > 0)
    gains = sum(x for x in rs if x > 0)
    losses = -sum(x for x in rs if x < 0)
    streak = worst = 0
    for x in rs:
        streak = streak + 1 if x < 0 else 0
        worst = max(worst, streak)
    equity = peak = dd = 0.0
    for x in rs:
        equity += x
        peak = max(peak, equity)
        dd = max(dd, peak - equity)
    return Stats(
        setups=len(records), filled=len(filled), wins=wins,
        win_rate=wins / len(filled) if filled else 0.0,
        avg_r=sum(rs) / len(rs) if rs else 0.0,
        total_r=sum(rs),
        profit_factor=gains / losses if losses else (math.inf if gains else 0.0),
        max_losing_streak=worst, max_drawdown_r=dd,
    )


def to_frame(records: list[TradeRecord]) -> pd.DataFrame:
    rows = []
    for r in records:
        d = asdict(r)
        comps = d.pop("components")
        d.update({f"pts_{k.lower().replace(' ', '_')}": v for k, v in comps.items()})
        rows.append(d)
    return pd.DataFrame(rows)


SCORE_BANDS = [(60, 69), (70, 74), (75, 79), (80, 84), (85, 89), (90, 100)]


def by_score_band(records: list[TradeRecord]) -> list[tuple[str, Stats]]:
    out = []
    for lo, hi in SCORE_BANDS:
        sub = [r for r in records if lo <= r.score <= hi]
        if sub:
            out.append((f"{lo}-{hi}", stats(sub)))
    return out


def by_threshold(records: list[TradeRecord], thresholds=(60, 65, 70, 75, 80, 85, 90)) -> list[tuple[int, Stats]]:
    """What every setup at or above each score would have done (ignores daily limits)."""
    return [(t, stats([r for r in records if r.score >= t])) for t in thresholds]


def by_component(records: list[TradeRecord]) -> list[tuple[str, Stats, Stats]]:
    """For each scored component: stats when it scored points vs when it didn't."""
    names = sorted({k for r in records for k in r.components})
    out = []
    for n in names:
        hit = [r for r in records if r.components.get(n, 0) > 0]
        miss = [r for r in records if r.components.get(n, 0) == 0]
        out.append((n, stats(hit), stats(miss)))
    return out


def by_key(records: list[TradeRecord], fn) -> list[tuple[str, Stats]]:
    groups: dict[str, list[TradeRecord]] = {}
    for r in records:
        groups.setdefault(fn(r), []).append(r)
    return [(k, stats(v)) for k, v in sorted(groups.items())]


def time_bucket(r: TradeRecord) -> str:
    t = r.time.time()
    edges = [time(9, 35), time(10, 0), time(10, 30), time(11, 0), time(11, 30)]
    for lo, hi in zip(edges, edges[1:]):
        if t < hi:
            return f"{lo:%H:%M}-{hi:%H:%M}"
    return "11:00-11:30"
