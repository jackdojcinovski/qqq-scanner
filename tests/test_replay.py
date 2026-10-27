from datetime import date

from scanner.config import load_config
from scanner.replay import format_hit, replay
from tests.helpers import bullish_sweep_day

CFG = load_config()


def test_replay_reports_setup_once():
    hits = replay(bullish_sweep_day(), CFG, date(2026, 10, 28))
    keys = [h.setup.key for h in hits]
    assert "2026-10-28-long-0950" in keys
    # one entry per key per band, never a repeat of the same band
    assert len(keys) == len({(h.setup.key, h.band) for h in hits})
    text = format_hit(hits[-1])
    assert "LONG" in text and "entry" in text


def test_replay_unknown_day_is_empty():
    assert replay(bullish_sweep_day(), CFG, date(2026, 10, 29)) == []


def test_replay_explains_empty_minutes():
    from scanner.replay import format_reasons
    hits = replay(bullish_sweep_day(), CFG, date(2026, 10, 28))
    assert any("outside window" in r for r in hits.reasons)   # 9:30-9:34
    assert "Why other minutes" in format_reasons(hits)
