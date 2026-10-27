import copy
from datetime import date

import pandas as pd
import pytest

from scanner import backtest as bt
from scanner.config import load_config
from scanner.discord import MemoryNotifier, contract_plan
from scanner.live import LiveScanner
from scanner.news import NewsFilter, parse_events
from scanner.store import Store
from tests.helpers import bars_from, bullish_sweep_day, candle

CFG = load_config()
DAY = date(2026, 10, 28)
ET = "America/New_York"


def winning_day():
    """The sweep day, then a pullback into the FVG (fills 499.80 and 500.00 entries)
    and a rally through the premarket-high target."""
    hist = bullish_sweep_day()
    tail = bars_from([candle(501.2, 500.6, low=500.5), candle(500.6, 500.1, low=499.95),
                      candle(500.1, 499.85, low=499.75), candle(499.85, 500.4, low=499.8),
                      candle(500.4, 501.0, low=500.3), candle(501.0, 501.6, high=501.7),
                      candle(501.6, 501.6)], "2026-10-28 09:56")
    return pd.concat([hist, tail])


def replay_live(full: pd.DataFrame, cfg=CFG, news=None, start="2026-10-28 09:30"):
    t0 = pd.Timestamp(start, tz=ET)
    hist, feed = full[full.index < t0], full[full.index >= t0]
    note = MemoryNotifier()
    sc = LiveScanner(cfg, "QQQ", note, hist, news=news, store=Store(":memory:"))
    for ts, row in feed.iterrows():
        sc.on_bar(ts, row.open, row.high, row.low, row.close, row.volume)
    sc.close_open_trades()
    return sc, note


def titles(note):
    return [m["embeds"][0]["title"] if m["embeds"] else m["content"] for m in note.sent]


def test_live_alerts_match_backtest_taken_trades():
    full = winning_day()
    for name_over in [{}, {"trade": {"entry_at": "near_edge"}}, {"exit": {"mode": "partial"}}]:
        cfg = copy.deepcopy(CFG)
        for k, v in name_over.items():
            cfg[k].update(v)
        sc, note = replay_live(full, cfg)
        live = [(tr.setup.key, tr.setup.entry, tr.outcome, tr.r or 0.0)
                for tr in sc.trades.values() if tr.taken]
        back = [(r.key, r.entry, r.outcome, r.r) for r in bt.run_day(full, DAY, cfg) if r.taken]
        assert live, name_over
        assert live == back, name_over


def test_alert_then_outcome_messages():
    sc, note = replay_live(winning_day())
    t = titles(note)
    assert any("LONG | Score" in x for x in t)
    assert any("FILLED" in x for x in t)
    assert any("TARGET HIT" in x for x in t)
    alert = next(m for m in note.sent if "LONG | Score" in m["embeds"][0]["title"])
    fields = {f["name"]: f["value"] for f in alert["embeds"][0]["fields"]}
    assert "499.80" in fields["Entry"] and fields["Stop"] and "premarket high" in fields["Target"]
    assert "Call" in fields["Contract"] and "What we see" in fields
    assert alert["content"].startswith("[PAPER]")
    assert sc.wins == 1 and sc.net_r > 0


def test_news_blocks_alerts_and_warns():
    raw = [{"title": "ISM Services PMI", "country": "USD", "date": "2026-10-28T10:00:00-04:00", "impact": "High"}]
    news = NewsFilter(CFG, parse_events(raw, __import__("zoneinfo").ZoneInfo(ET)))
    sc, note = replay_live(winning_day(), news=news)
    assert not any("Score" in x for x in titles(note))      # 9:45-10:30 blocked


def test_one_alert_per_setup_and_daily_limit():
    cfg = copy.deepcopy(CFG)
    cfg["limits"]["max_alerts_per_day"] = 0
    sc, note = replay_live(winning_day(), cfg)
    assert not any("Score" in x for x in titles(note))
    assert any(not tr.taken and tr.setup.band(cfg) == "alert" for tr in sc.trades.values())
    assert "skipped" in sc.summary()[1][0]["description"]


def test_no_alerts_while_session_data_missing_then_backfill():
    full = winning_day()
    t_start = pd.Timestamp("2026-10-28 09:50", tz=ET)
    hist, feed = full[full.index < pd.Timestamp("2026-10-28 09:30", tz=ET)], full[full.index >= t_start]
    calls = []

    def backfill(a, b):
        calls.append((a, b))
        return full[(full.index >= a) & (full.index < b)]

    note = MemoryNotifier()
    sc = LiveScanner(CFG, "QQQ", note, hist, backfill=backfill)
    sc.add_gap(pd.Timestamp("2026-10-28 09:30", tz=ET), t_start)
    for ts, row in feed.iterrows():
        sc.on_bar(ts, row.open, row.high, row.low, row.close, row.volume)
    assert not any("Score" in x for x in titles(note))       # gap not fillable within 16 min
    assert sc.gaps and not calls
    later = pd.Timestamp("2026-10-28 10:07", tz=ET)
    sc._fill_gaps(later)
    assert calls and not sc.gaps and (sc.bars.index == pd.Timestamp("2026-10-28 09:40", tz=ET)).any()


def test_contract_plan_sizing():
    from scanner.scorer import Setup
    s = Setup("k", "long", 500.4, 499.4, 502, "t", 1.6, "FVG", pd.Timestamp("2026-10-30 10:00", tz=ET))
    p = contract_plan(s, CFG, "QQQ")
    assert p["strike"] == 500 and p["right"] == "C"
    assert p["expiry"] == date(2026, 11, 2)                  # Friday -> Monday
    assert p["risk_per_contract"] == pytest.approx(45.0)
    assert p["max_contracts"] == 2                            # $100 limit / $45


def test_heartbeat_and_stale_warning():
    from datetime import datetime, timedelta
    sc, note = replay_live(winning_day())
    n = len(note.sent)
    sc.tick(pd.Timestamp("2026-10-28 10:10", tz=ET))          # last bar 10:02 -> stale
    assert len(note.sent) == n + 1 and "No market data" in titles(note)[-1]
    sc.tick(pd.Timestamp("2026-10-28 10:11", tz=ET))
    assert len(note.sent) == n + 1                            # warned once only
    sc.last_bar = pd.Timestamp("2026-10-28 10:10", tz=ET)
    sc.last_post = datetime.now() - timedelta(minutes=31)
    sc.tick(pd.Timestamp("2026-10-28 10:11", tz=ET))
    assert "Still running" in titles(note)[-1]


def test_missed_session_minutes_become_a_gap():
    full = winning_day()
    t0 = pd.Timestamp("2026-10-28 09:30", tz=ET)
    hist, feed = full[full.index < t0], full[full.index >= t0]
    sc = LiveScanner(CFG, "QQQ", MemoryNotifier(), hist)
    rows = list(feed.iterrows())
    for ts, row in rows[:10] + rows[15:]:                   # drop 09:40-09:44
        sc.on_bar(ts, row.open, row.high, row.low, row.close, row.volume)
    assert sc.gaps and sc.gaps[0][0] == pd.Timestamp("2026-10-28 09:40", tz=ET)
    assert sc.alerts == 0                                     # no alerts on incomplete data


def test_run_scanner_imports():
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_scanner", "run_scanner.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert callable(mod.main)
