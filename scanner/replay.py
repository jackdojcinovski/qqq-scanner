"""Replay a day bar by bar through the scorer, as the live scanner would see it."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from .news import NewsStatus
from .scorer import DayState, Setup, evaluate


@dataclass
class ReplayHit:
    setup: Setup
    band: str


def replay(history: pd.DataFrame, cfg: dict[str, Any], day: date,
           news: NewsStatus | None = None) -> list[ReplayHit]:
    """Return each setup the first time it reaches the log or alert band, and
    again if it later upgrades from log to alert."""
    day_idx = [i for i, ts in enumerate(history.index) if ts.date() == day]
    if not day_idx:
        return []
    seen: dict[str, str] = {}
    hits: list[ReplayHit] = []
    state = DayState()   # alert limit applies; the loss stop needs outcomes (Phase 3)
    for i in day_idx:
        ev = evaluate(history.iloc[:i + 1], cfg, news=news, day=state)
        if ev.setup is None:
            continue
        band = ev.setup.band(cfg)
        if band == "ignore":
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
