"""Backtest: replay every day through the scanner and play each setup forward.

Two stages, so many variants can be tested cheaply:

1. `day_candidates()` scans a day minute by minute and stores every candidate
   setup (the slow part, done once and cached by `scanner.research`).
2. `run_variant()` applies one config's preferences to those candidates,
   applies the live rules and simulates each trade (fast).

Every setup that reaches the log band is simulated, so reports can compare
score bands and components. `taken` marks the trades you would actually have
received live: alerts only, at most `max_alerts_per_day`, none after a losing
trade, and one position at a time.

Results are in R (multiples of the risk to the stop) on the share price.
Option P&L differs: see tools/option_check.py.
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import date, time
from typing import Any, Iterable

import numpy as np
import pandas as pd

from . import indicators as ind
from .config import parse_time
from .scorer import ENTRY_MODES, Setup, candidates, select


# --------------------------------------------------------------------------- trade simulation

@dataclass
class TradeResult:
    outcome: str      # target / stop / breakeven / trail / time / close / no fill / ran away
    r: float = 0.0
    fill_time: pd.Timestamp | None = None
    fill_price: float | None = None
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    mfe_r: float = 0.0   # best open profit reached, in R
    mae_r: float = 0.0   # worst open loss reached, in R
    events: list[tuple[pd.Timestamp, str, float]] = field(default_factory=list)  # fill / partial
    stop_now: float | None = None   # live only: current stop after breakeven / trailing moves

    @property
    def filled(self) -> bool:
        return self.fill_time is not None

    @property
    def final(self) -> bool:
        return self.outcome not in ("pending", "open")


def exit_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    e = dict(cfg.get("exit", {}))
    e.setdefault("mode", "fixed")
    e.setdefault("time_exit_minutes", cfg["trade"].get("time_exit_minutes", 60))
    e.setdefault("partial_at_r", 1.0)
    e.setdefault("partial_fraction", 0.5)
    e.setdefault("trail_after_r", 1.0)
    e.setdefault("trail_distance_r", 1.0)
    return e


def simulate(setup: Setup, after: pd.DataFrame, cfg: dict[str, Any], live: bool = False) -> TradeResult:
    """Play a setup forward on the bars after it (same session).

    Exit modes (config `exit.mode`):
      fixed   - full position to the target or stop, time exit after N minutes
      partial - take `partial_fraction` off at `partial_at_r`, stop to breakeven,
                rest to the target / breakeven stop / time exit
      trail   - no fixed target; once `trail_after_r` is reached the stop trails
                `trail_distance_r` behind the best price (never below breakeven)

    Conservative assumptions: within a bar the stop is checked before anything
    favourable; the fill bar can stop out but can't take profit; stops that gap
    through exit at the bar's open; trailing stops move only on the next bar.

    live=True is for a trade still in progress: when the bars run out it returns
    "pending" (not filled yet) or "open" (in the trade) instead of closing it.
    """
    e = exit_settings(cfg)
    long = setup.direction == "long"
    sign = 1 if long else -1
    entry, stop0, target = setup.entry, setup.stop, setup.target
    valid_until = setup.time + pd.Timedelta(minutes=cfg["trade"]["entry_valid_minutes"])
    time_exit = pd.Timedelta(minutes=e["time_exit_minutes"])
    mode = e["mode"]

    fill_t = fill_px = None
    stop = stop0
    risk = 0.0
    realized = 0.0          # R banked by a partial exit
    remaining = 1.0
    partial_done = False
    best = worst = 0.0      # extremes in R
    peak = None             # best price since fill
    events: list[tuple[pd.Timestamp, str, float]] = []

    def finish(px: float, outcome: str, ts) -> TradeResult:
        r = realized + remaining * sign * (px - fill_px) / risk
        return TradeResult(outcome, round(r, 3), fill_t, float(fill_px), ts, float(px),
                           round(best, 3), round(worst, 3), events)

    for ts, o, h, lo, c in after[["open", "high", "low", "close"]].itertuples():
        if fill_t is None:
            if ts > valid_until:
                return TradeResult("no fill", exit_time=ts)
            touched = lo <= entry if long else h >= entry
            if not touched:
                if (h >= target) if long else (lo <= target):
                    return TradeResult("ran away", exit_time=ts)
                continue
            fill_t = ts
            fill_px = min(o, entry) if long else max(o, entry)
            risk = abs(fill_px - stop0)
            if risk <= 0:
                return TradeResult("no fill")
            events.append((ts, "fill", float(fill_px)))
            peak = fill_px
            worst = min(worst, sign * ((lo if long else h) - fill_px) / risk)
            if (lo <= stop) if long else (h >= stop):
                return finish(stop, "stop", ts)
            continue

        fav = h if long else lo
        adv = lo if long else h
        best = max(best, sign * (fav - fill_px) / risk)
        worst = min(worst, sign * (adv - fill_px) / risk)

        # 1. stop (checked first: the conservative order within a bar)
        if (adv <= stop) if long else (adv >= stop):
            px = min(o, stop) if long else max(o, stop)
            if stop == stop0:
                outcome = "stop"
            elif mode == "trail" and sign * (stop - fill_px) > 0:
                outcome = "trail"
            else:
                outcome = "breakeven"
            return finish(px, outcome, ts)

        # 2. favourable events
        if mode == "partial" and not partial_done:
            level = fill_px + sign * e["partial_at_r"] * risk
            if (fav >= level) if long else (fav <= level):
                realized += e["partial_fraction"] * e["partial_at_r"]
                remaining -= e["partial_fraction"]
                partial_done = True
                stop = fill_px
                events.append((ts, "partial", float(level)))
        if mode in ("fixed", "partial") and ((fav >= target) if long else (fav <= target)):
            px = max(o, target) if long else min(o, target)
            return finish(px, "target", ts)

        # 3. time exit
        if ts - fill_t >= time_exit:
            return finish(c, "time", ts)

        # 4. trailing stop for the next bar
        if mode == "trail":
            peak = max(peak, fav) if long else min(peak, fav)
            if sign * (peak - fill_px) >= e["trail_after_r"] * risk:
                trail = peak - sign * e["trail_distance_r"] * risk
                trail = max(trail, fill_px) if long else min(trail, fill_px)
                stop = max(stop, trail) if long else min(stop, trail)

    if fill_t is None:
        return TradeResult("pending" if live else "no fill")
    if live:
        px = float(after["close"].iloc[-1]) if len(after) else fill_px
        res = finish(px, "open", None)
        res.exit_price = None
        res.stop_now = stop          # current stop (moves with partial / trail)
        return res
    return finish(float(after["close"].iloc[-1]), "close", after.index[-1])


# --------------------------------------------------------------------------- candidates per day

@dataclass
class DayCandidates:
    symbol: str
    day: date
    bars: pd.DataFrame                                   # that day's regular-session bars
    minutes: list[tuple[pd.Timestamp, list[Setup]]]      # only minutes with candidates


def day_candidates(history: pd.DataFrame, day: date, cfg: dict[str, Any],
                   symbol: str = "QQQ", entry_modes=ENTRY_MODES) -> DayCandidates:
    """`history` holds at least the prior session and all of `day`."""
    w = cfg["window"]
    start, end = parse_time(w["start"]), parse_time(w["end"])
    session = ind.regular_session(history)
    day_bars = session[session.index.date == day]
    pos = {ts: i for i, ts in enumerate(history.index)}
    minutes = []
    for ts in day_bars.index:
        if not (start <= ts.time() <= end):
            continue
        cands, _ = candidates(history.iloc[:pos[ts] + 1], cfg, entry_modes)
        if cands:
            minutes.append((ts, cands))
    return DayCandidates(symbol, day, day_bars[["open", "high", "low", "close"]].copy(), minutes)


# --------------------------------------------------------------------------- variant runner

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
    symbol: str = "QQQ"
    trigger: str = ""
    mfe_r: float = 0.0
    mae_r: float = 0.0
    components: dict[str, int] = field(default_factory=dict)


def run_variant(dc: DayCandidates, cfg: dict[str, Any]) -> list[TradeRecord]:
    w = cfg["window"]
    start, end = parse_time(w["start"]), parse_time(w["end"])
    lim = cfg["limits"]
    records: dict[str, TradeRecord] = {}
    alerts = losses = 0
    busy_until: pd.Timestamp | None = None
    settled: set[str] = set()

    for ts, cands in dc.minutes:
        if not (start <= ts.time() <= end):
            continue
        for k, rec in records.items():
            if rec.taken and k not in settled and rec.exit_time is not None and rec.exit_time <= ts:
                settled.add(k)
                if rec.filled and rec.r < 0:
                    losses += 1
        s, _ = select(cands, cfg)
        if s is None:
            continue
        band = s.band(cfg)
        if band == "ignore":
            continue
        prev = records.get(s.key)
        if prev is not None and not (prev.band == "log" and band == "alert"):
            continue

        res = simulate(s, dc.bars[dc.bars.index > ts], cfg)

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
            busy_until = res.exit_time if res.exit_time is not None else valid

        records[s.key] = TradeRecord(
            day=dc.day, time=ts, key=s.key, direction=s.direction, score=s.score, band=band,
            taken=taken, skip_reason=skip, entry=s.entry, stop=s.stop, target=s.target,
            target_name=s.target_name, rr=s.rr, entry_zone=s.entry_zone,
            outcome=res.outcome, r=res.r, filled=res.filled,
            fill_time=res.fill_time, exit_time=res.exit_time,
            symbol=dc.symbol, trigger=s.trigger, mfe_r=res.mfe_r, mae_r=res.mae_r,
            components={c.name: c.points for c in s.components},
        )
    return list(records.values())


def run_day(history: pd.DataFrame, day: date, cfg: dict[str, Any], symbol: str = "QQQ") -> list[TradeRecord]:
    """Scan and run one day with one config (used by tools and tests)."""
    dc = day_candidates(history, day, cfg, symbol, (cfg["trade"].get("entry_at", "mid"),))
    return run_variant(dc, cfg)


def day_slices(bars: pd.DataFrame, days: Iterable[date]) -> list[tuple[date, pd.DataFrame]]:
    """For each day, the bars from the previous trading day through that day."""
    d_idx = bars.index.date
    all_days = sorted(set(d_idx))
    want = set(days)
    out = []
    for i, d in enumerate(all_days):
        if d not in want or i == 0:
            continue
        prev = all_days[i - 1]
        out.append((d, bars[(d_idx == prev) | (d_idx == d)]))
    return out


def _cands_job(args):
    d, hist, cfg, symbol, modes = args
    return day_candidates(hist, d, cfg, symbol, modes)


def build_candidates(bars: pd.DataFrame, days: Iterable[date], cfg: dict[str, Any],
                     symbol: str = "QQQ", entry_modes=ENTRY_MODES,
                     workers: int | None = None, progress=None) -> list[DayCandidates]:
    jobs = [(d, h, cfg, symbol, tuple(entry_modes)) for d, h in day_slices(bars, days)]
    workers = workers if workers is not None else max(1, (os.cpu_count() or 2) - 1)
    out: list[DayCandidates] = []
    if workers <= 1 or len(jobs) < 4:
        for n, j in enumerate(jobs, 1):
            out.append(_cands_job(j))
            if progress:
                progress(n, len(jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for n, dc in enumerate(pool.map(_cands_job, jobs, chunksize=4), 1):
                out.append(dc)
                if progress:
                    progress(n, len(jobs))
    return out


def run_on(day_cands: Iterable[DayCandidates], cfg: dict[str, Any]) -> list[TradeRecord]:
    out: list[TradeRecord] = []
    for dc in day_cands:
        out += run_variant(dc, cfg)
    out.sort(key=lambda r: (r.time, r.symbol))
    return out


def run_backtest(bars: pd.DataFrame, days: Iterable[date], cfg: dict[str, Any],
                 workers: int | None = None, progress=None, symbol: str = "QQQ") -> list[TradeRecord]:
    dcs = build_candidates(bars, days, cfg, symbol, (cfg["trade"].get("entry_at", "mid"),),
                           workers, progress)
    return run_on(dcs, cfg)


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


def bootstrap_ci(values: list[float], n: int = 2000, level: float = 0.90,
                 seed: int = 7) -> tuple[float, float]:
    """Confidence interval for the mean by resampling trades."""
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = rng.choice(arr, size=(n, len(arr)), replace=True).mean(axis=1)
    lo = (1 - level) / 2
    return float(np.quantile(means, lo)), float(np.quantile(means, 1 - lo))


def to_frame(records: list[TradeRecord]) -> pd.DataFrame:
    rows = []
    for r in records:
        d = asdict(r)
        comps = d.pop("components")
        d.update({f"pts_{k.lower().replace(' ', '_')}": v for k, v in comps.items()})
        rows.append(d)
    return pd.DataFrame(rows)


SCORE_BANDS = [(0, 59), (60, 69), (70, 74), (75, 79), (80, 84), (85, 89), (90, 100)]


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
