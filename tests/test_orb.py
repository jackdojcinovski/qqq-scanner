from datetime import date

import pandas as pd
import pytest

from scanner import backtest as bt
from scanner import orb
from scanner import research as rs
from scanner.config import load_config
from tests.test_live import winning_day

CFG = load_config()


def orb_day():
    """The live-test day with an up first candle so the classic ORB long triggers."""
    full = winning_day().copy()
    et = "America/New_York"
    full.loc[pd.Timestamp("2026-10-28 09:34", tz=et), "close"] = 500.8
    full.loc[pd.Timestamp("2026-10-28 09:35", tz=et), ["high", "close"]] = [501.0, 500.95]   # above VWAP
    return full


def test_fast_engine_matches_cached_research_path():
    full = orb_day()
    name = orb.ORB_VARIANTS["both"]
    fast = orb.orb_trades({"QQQ": full}, CFG, name, workers=1)
    cfg = rs.merge(CFG, rs.variant(name).overrides)
    dc = bt.day_candidates(full, date(2026, 10, 28), CFG)
    slow = [r for r in bt.run_variant(dc, cfg) if r.taken and r.trigger == "orb_classic"]
    assert len(fast) == 1 and len(slow) == 1
    assert fast[0].res.r == pytest.approx(slow[0].r) and fast[0].res.outcome == slow[0].outcome


def test_stop_scaling_and_noise_stop():
    full = orb_day()
    trades = orb.orb_trades({"QQQ": full}, CFG, orb.ORB_VARIANTS["both"], scales=(1.0, 2.0), workers=1)
    t1, t2 = [t for t in trades if t.stop_scale == 1.0][0], [t for t in trades if t.stop_scale == 2.0][0]
    s = 1 if t1.direction == "long" else -1
    assert s * (t1.entry_ref - t2.stop) == pytest.approx(2 * s * (t1.entry_ref - t1.stop), abs=0.02)
    # noise stop: a stopped trade that closed beyond the fill
    fake = orb.OrbTrade("QQQ", date(2026, 1, 2), "long", 100, 99, 1.0,
                        bt.TradeResult("stop", -1.0, pd.Timestamp("2026-01-02 09:36"), 100.0,
                                       pd.Timestamp("2026-01-02 10:00"), 99.0), 101.0, 1.0)
    assert fake.noise_stop
    fake.eod_close = 98.0
    assert not fake.noise_stop
    assert isinstance(t1.row(), rs.Row)
