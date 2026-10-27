from datetime import date, datetime
from zoneinfo import ZoneInfo

from scanner.config import load_config
from scanner.news import NewsFilter, parse_events

CFG = load_config()
ET = ZoneInfo("America/New_York")

RAW = [
    {"title": "CPI m/m", "country": "USD", "date": "2026-11-10T08:30:00-05:00", "impact": "High"},
    {"title": "PPI m/m", "country": "USD", "date": "2026-11-11T08:30:00-05:00", "impact": "High"},
    {"title": "ISM Services PMI", "country": "USD", "date": "2026-11-12T10:00:00-05:00", "impact": "High"},
    {"title": "German ZEW", "country": "EUR", "date": "2026-11-12T05:00:00-05:00", "impact": "High"},
    {"title": "Crude Oil Inventories", "country": "USD", "date": "2026-11-13T10:30:00-05:00", "impact": "Medium"},
]


def nf(blackouts=None, raw=RAW):
    events = parse_events(raw, ET) if raw is not None else None
    return NewsFilter(CFG, events, blackouts)


def at(day, hh, mm):
    return datetime(2026, 11, day, hh, mm, tzinfo=ET)


def test_cpi_blocks_whole_day():
    s = nf().status(at(10, 13, 0))
    assert s.blocked and "standing down" in s.reason


def test_premarket_event_is_late_start():
    assert nf().status(at(11, 9, 45)).blocked
    assert not nf().status(at(11, 10, 1)).blocked


def test_session_event_window_and_warning():
    f = nf()
    assert not f.status(at(12, 9, 40)).blocked
    early = f.status(at(12, 9, 44))
    assert not early.blocked and early.warning == ""
    inside = f.status(at(12, 9, 50))
    assert inside.blocked and "ISM" in inside.warning
    assert f.status(at(12, 10, 25)).blocked
    assert not f.status(at(12, 10, 31)).blocked


def test_other_currency_and_medium_impact_ignored():
    assert not nf().status(at(13, 10, 30)).blocked


def test_blackout_file_dates():
    s = nf({date(2026, 11, 16): "NVDA earnings"}).status(at(16, 10, 0))
    assert s.blocked and "NVDA" in s.reason


def test_fail_closed_without_calendar():
    assert nf(raw=None).status(at(16, 10, 0)).blocked
