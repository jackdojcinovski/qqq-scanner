"""News filter: Forex Factory's weekly calendar plus your own blackout.yaml.

Rules (all times Eastern, red USD events only):
  * full-day block  - FOMC rate decision, CPI, Nonfarm Payrolls
  * late start      - any other red event before the open blocks until 10:00
  * window block    - red events during the session block 15 min before to 30 min after
  * warning         - an event starting within 15 minutes (for open trades)
  * manual block    - any date listed in blackout.yaml
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests
import yaml

from .config import ROOT, parse_time

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Event:
    title: str
    when: datetime
    impact: str
    currency: str


@dataclass(frozen=True)
class NewsStatus:
    blocked: bool
    reason: str = ""
    warning: str = ""      # event starting soon, for open trades


def parse_events(raw: list[dict[str, Any]], tz: ZoneInfo) -> list[Event]:
    out = []
    for item in raw:
        try:
            when = datetime.fromisoformat(item["date"]).astimezone(tz)
        except (KeyError, ValueError):
            continue
        out.append(Event(item.get("title", ""), when, item.get("impact", ""), item.get("country", "")))
    return out


def fetch_calendar(url: str, cache: Path, timeout: int = 10) -> list[dict[str, Any]] | None:
    """Download the weekly feed and cache it; fall back to the cache on failure."""
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "qqq-scanner/1.0"})
        r.raise_for_status()
        data = r.json()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data), encoding="utf-8")
        return data
    except (requests.RequestException, ValueError) as exc:
        log.warning("calendar fetch failed (%s); trying cache", exc)
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    return None


def load_blackouts(path: Path) -> dict[date, str]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out = {}
    for k, v in (data.get("dates") or {}).items():
        d = k if isinstance(k, date) else date.fromisoformat(str(k))
        out[d] = str(v)
    return out


class NewsFilter:
    def __init__(self, cfg: dict[str, Any], events: list[Event] | None,
                 blackouts: dict[date, str] | None = None):
        n = cfg["news"]
        self.tz = ZoneInfo(cfg["timezone"])
        self.fail_closed = n["fail_closed"]
        self.keywords = [k.lower() for k in n["full_day_keywords"]]
        self.before = timedelta(minutes=n["window_before_minutes"])
        self.after = timedelta(minutes=n["window_after_minutes"])
        self.warn = timedelta(minutes=n["warn_minutes"])
        self.late_until = parse_time(n["late_start_until"])
        self.open_t = parse_time(cfg["session"]["open"])
        self.close_t = parse_time(cfg["session"]["close"])
        self.available = events is not None
        self.events = [e for e in (events or [])
                       if e.currency == n["currency"] and e.impact == n["impact"]]
        self.blackouts = blackouts or {}

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "NewsFilter":
        n = cfg["news"]
        tz = ZoneInfo(cfg["timezone"])
        raw = fetch_calendar(n["feed_url"], ROOT / n["cache_file"])
        events = parse_events(raw, tz) if raw is not None else None
        return cls(cfg, events, load_blackouts(ROOT / n["blackout_file"]))

    def is_full_day(self, e: Event) -> bool:
        t = e.title.lower()
        return any(k in t for k in self.keywords)

    def day_events(self, d: date) -> list[Event]:
        return [e for e in self.events if e.when.date() == d]

    def status(self, now: datetime) -> NewsStatus:
        now = now.astimezone(self.tz)
        today = now.date()
        if today in self.blackouts:
            return NewsStatus(True, f"blackout: {self.blackouts[today]}")
        if not self.available:
            if self.fail_closed:
                return NewsStatus(True, "economic calendar unavailable")
            return NewsStatus(False, warning="economic calendar unavailable")

        events = self.day_events(today)
        for e in events:
            if self.is_full_day(e):
                return NewsStatus(True, f"{e.title} at {e.when:%H:%M} - standing down today")

        warning = ""
        for e in events:
            t = e.when.time()
            if t < self.open_t:
                if now.time() < self.late_until:
                    return NewsStatus(True, f"{e.title} at {e.when:%H:%M} - no alerts before "
                                            f"{self.late_until:%H:%M}")
                continue
            if t >= self.close_t:
                continue
            if e.when - self.before <= now <= e.when + self.after:
                soon = f"{e.title} at {e.when:%H:%M}" if now < e.when else ""
                return NewsStatus(True, f"{e.title} at {e.when:%H:%M}", warning=soon)
            if now < e.when <= now + self.warn and not warning:
                warning = f"{e.title} at {e.when:%H:%M}"
        return NewsStatus(False, warning=warning)
