from datetime import date

import pandas as pd
import pytest

from scanner import backtest as bt
from scanner.config import load_config
from scanner.history import load_history, save_month, trading_days
from scanner.scorer import Setup
from tests.helpers import bars_from, bullish_sweep_day, candle, flat_day

CFG = load_config()
T0 = pd.Timestamp("2026-10-28 10:00", tz="America/New_York")


def long_setup(entry=100.0, stop=99.0, target=102.0):
    return Setup(key="k", direction="long", score=80, entry=entry, stop=stop, target=target,
                 target_name="t", rr=2.0, entry_zone="FVG midpoint", time=T0)


def after(rows):
    return bars_from(rows, "2026-10-28 10:01")


def test_win():
    r = bt.simulate(long_setup(), after([candle(100.5, 100.0, low=99.9), candle(100.0, 101.5),
                                         candle(101.5, 102.2, high=102.3)]), CFG)
    assert r.outcome == "target" and r.r == pytest.approx(2.0)


def test_loss_and_gap_through_stop():
    r = bt.simulate(long_setup(), after([candle(100.5, 100.0, low=99.9), candle(100.0, 99.5),
                                         candle(98.5, 98.4, high=98.6)]), CFG)
    assert r.outcome == "stop" and r.r == pytest.approx(-1.5)   # gapped to 98.5


def test_both_in_one_bar_counts_as_stop():
    r = bt.simulate(long_setup(), after([candle(100.5, 100.0, low=99.9),
                                         candle(100.0, 100.0, high=102.5, low=98.9)]), CFG)
    assert r.outcome == "stop"


def test_fill_and_stop_same_bar():
    r = bt.simulate(long_setup(), after([candle(100.5, 99.0, low=98.9)]), CFG)
    assert r.outcome == "stop" and r.r < 0


def test_no_fill_and_ran_away():
    never = [candle(100.6, 100.7, low=100.5)] * 20
    assert bt.simulate(long_setup(), after(never), CFG).outcome == "no fill"
    ran = [candle(100.6, 101.5, low=100.5), candle(101.5, 102.4, low=101.4)]
    assert bt.simulate(long_setup(), after(ran), CFG).outcome == "ran away"


def test_time_exit():
    rows = [candle(100.5, 100.0, low=99.9)] + [candle(100.3, 100.4, low=100.2, high=100.6)] * 70
    r = bt.simulate(long_setup(), after(rows), CFG)
    assert r.outcome == "time" and r.r == pytest.approx(0.4)


def test_better_fill_when_bar_opens_below_entry():
    r = bt.simulate(long_setup(), after([candle(99.8, 99.9, low=99.7), candle(100, 102.1, high=102.2)]), CFG)
    assert r.fill_price == 99.8
    assert r.r == pytest.approx((102.0 - 99.8) / (99.8 - 99.0))


def test_stats():
    recs = [bt.TradeRecord(date(2026, 1, 1), T0, str(i), "long", 80, "alert", True, "", 1, 0, 2, "t",
                           2, "FVG", "x", r, True, T0, T0) for i, r in enumerate([2, -1, -1, -1, 2])]
    s = bt.stats(recs)
    assert s.filled == 5 and s.wins == 2 and s.win_rate == pytest.approx(0.4)
    assert s.total_r == pytest.approx(1.0) and s.profit_factor == pytest.approx(4 / 3)
    assert s.max_losing_streak == 3 and s.max_drawdown_r == pytest.approx(3.0)


def test_run_day_and_backtest_on_synthetic_day():
    hist = bullish_sweep_day()
    # extend the day with bars that run to the premarket-high target
    tail = bars_from([candle(501.2, 501.0, low=500.9)] * 3 + [candle(501.0, 501.6, high=501.7)] +
                     [candle(501.6, 501.6)] * 5, "2026-10-28 09:56")
    hist = pd.concat([hist, tail])
    recs = bt.run_day(hist, date(2026, 10, 28), CFG)
    taken = [r for r in recs if r.taken]
    assert len(taken) >= 1
    assert all(r.band == "alert" for r in taken)
    assert len({r.key for r in recs}) == len(recs)
    multi = bt.run_backtest(hist, [date(2026, 10, 28)], CFG, workers=1)
    assert [r.key for r in multi] == [r.key for r in recs]
    df = bt.to_frame(recs)
    assert "pts_liquidity_sweep" in df.columns


def test_daily_alert_limit_and_one_position():
    import copy
    cfg = copy.deepcopy(CFG)
    cfg["limits"]["max_alerts_per_day"] = 0
    recs = bt.run_day(bullish_sweep_day(), date(2026, 10, 28), cfg)
    assert recs and not any(r.taken for r in recs)
    assert {r.skip_reason for r in recs if r.band == "alert"} == {"daily alert limit"}


def test_history_round_trip(tmp_path):
    df = pd.concat([flat_day("2026-09-30", minutes=10), flat_day("2026-10-01", minutes=10)])
    save_month(df[df.index.month == 9], "QQQ", 2026, 9, tmp_path)
    save_month(df[df.index.month == 10], "QQQ", 2026, 10, tmp_path)
    back = load_history("QQQ", date(2026, 9, 1), date(2026, 10, 31), tmp_path)
    assert len(back) == 20 and str(back.index.tz) == "America/New_York"
    assert trading_days(back) == [date(2026, 9, 30), date(2026, 10, 1)]
    assert len(load_history("QQQ", date(2026, 10, 1), date(2026, 10, 1), tmp_path)) == 10
