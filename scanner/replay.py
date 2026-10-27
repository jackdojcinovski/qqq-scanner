"""Replay a day bar by bar through the scorer, as the live scanner would see it."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from .news import NewsStatus
from .scorer import DayState, Setup, evaluate


@dataclass
class ReplayHit:
    setup: Setup
    band: str


@dataclass
class ReplayResult(list):
    """List of hits plus a count of why each other minute had no setup."""
    reasons: Counter = field(default_factory=Counter)
    below_log: int = 0          # minutes with a setup that scored under the log band
    best_below: Setup | None = None


def replay(history: pd.DataFrame, cfg: dict[str, Any], day: date,
           news: NewsStatus | None = None) -> ReplayResult:
    """Return each setup the first time it reaches the log or alert band, and
    again if it later upgrades from log to alert."""
    day_idx = [i for i, ts in enumerate(history.index) if ts.date() == day]
    hits = ReplayResult()
    if not day_idx:
        return hits
    seen: dict[str, str] = {}
    state = DayState()   # alert limit applies; the loss stop needs outcomes (Phase 3)
    for i in day_idx:
        ev = evaluate(history.iloc[:i + 1], cfg, news=news, day=state)
        if ev.setup is None:
            for part in ev.reason.split(" / "):
                hits.reasons[part.split(" at ")[0]] += 1
            continue
        band = ev.setup.band(cfg)
        if band == "ignore":
            hits.below_log += 1
            if hits.best_below is None or ev.setup.score > hits.best_below.score:
                hits.best_below = ev.setup
            continue
        prev = seen.get(ev.setup.key)
        if prev is None or (prev == "log" and band == "alert"):
            seen[ev.setup.key] = band
            hits.append(ReplayHit(ev.setup, band))
            if band == "alert":
                state.alerts_today += 1
    return hits


def format_hit(h: ReplayHit) -> str:
    s = h.setup
    lines = [f"{s.time:%H:%M}  {s.direction.upper():5}  score {s.score:>3}  [{h.band}]  "
             f"entry {s.entry:.2f} ({s.entry_zone})  stop {s.stop:.2f}  "
             f"target {s.target:.2f} ({s.target_name})  R:R {s.rr:.1f}"]
    for c in s.components:
        mark = "+" if c.hit else "-"
        lines.append(f"        {mark} {c.name} ({c.points}/{c.max_points}): {c.detail}")
    return "\n".join(lines)


def format_reasons(r: ReplayResult) -> str:
    lines = ["Why other minutes had no setup (a minute can count under more than one):"]
    for reason, n in r.reasons.most_common():
        lines.append(f"  {n:>4} min  {reason}")
    if r.below_log:
        b = r.best_below
        lines.append(f"  {r.below_log:>4} min  setup found but scored under the log band "
                     f"(best: {b.direction} {b.score} at {b.time:%H:%M})")
    return "\n".join(lines)
