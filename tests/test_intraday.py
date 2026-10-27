import numpy as np
import pandas as pd
import pytest

from scanner import intraday as ix
from scanner import research as rs
from tests.helpers import bars_from


def make_days(n=20, start="2026-01-05", seed=3, trend_last=0.0, vol=0.03):
    """n weekdays of 390 1-minute bars; the last day gets an extra drift per minute."""
    rng = np.random.default_rng(seed)
    frames, p, d, made = [], 100.0, pd.Timestamp(start), 0
    while made < n:
        if d.weekday() < 5:
            drift = trend_last if made == n - 1 else 0.0
            rows = []
            for i in range(390):
                o = p
                c = o + rng.normal(drift, vol)
                rows.append((o, max(o, c) + 0.01, min(o, c) - 0.01, c, 1000))
                p = c
            frames.append(bars_from(rows, f"{d.date()} 09:30"))
            made += 1
        d += pd.Timedelta(days=1)
    return pd.concat(frames)


def test_grid_shapes_and_noise_uses_prior_days_only():
    bars = make_days(20)
    g = ix.build_grid("QQQ", bars)
    assert g.close.shape == (20, 390)
    assert np.isnan(g.noise[ix.LOOKBACK - 1]).all() and not np.isnan(g.noise[ix.LOOKBACK]).any()
    # changing the last day must not change its own noise estimate
    last = bars.index.date == g.days[-1]
    b2 = bars.copy()
    b2.loc[last, ["open", "high", "low", "close"]] *= 1.3
    g2 = ix.build_grid("QQQ", b2)
    assert np.allclose(g.noise[-1], g2.noise[-1])
    assert g.prev_close[-1] == pytest.approx(bars[bars.index.date == g.days[-2]]["close"].iloc[-1])


def test_strong_up_day_goes_long_and_rides_to_close():
    g = ix.build_grid("QQQ", make_days(20, trend_last=0.02))
    t = [x for x in ix.noise_trades(g) if x.day == g.days[-1]]
    assert len(t) == 1 and t[0].direction == "long"
    assert t[0].entry_min % 30 == 0 and t[0].entry == pytest.approx(g.open[-1, t[0].entry_min])
    assert t[0].outcome == "close" and t[0].r > 0
    assert ix.noise_trades(g, sides="long")[-1].direction == "long"


def test_no_lookahead_entry_unchanged_by_later_bars():
    bars = make_days(20, trend_last=0.02)
    g = ix.build_grid("QQQ", bars)
    t = [x for x in ix.noise_trades(g) if x.day == g.days[-1]][0]
    # wreck everything after the fill bar: same entry decision and price
    b2 = bars.copy()
    after = (bars.index.date == g.days[-1]) & (bars.index > t.fill_time)
    b2.loc[after, ["open", "high", "low", "close"]] *= 0.9
    g2 = ix.build_grid("QQQ", b2)
    t2 = [x for x in ix.noise_trades(g2) if x.day == g.days[-1]][0]
    assert (t2.entry, t2.entry_min, t2.direction) == (t.entry, t.entry_min, t.direction)
    assert t2.r < t.r


def test_trailing_exit_on_checkpoint():
    bars = make_days(20, trend_last=0.02)
    last = bars.index.date == bars.index.date[-1]
    idx = np.where(last)[0]
    # crash from 13:00 onward on the last day
    crash = idx[210:]
    bars.iloc[crash, :4] = bars.iloc[crash, :4].to_numpy() - np.linspace(0, 30, len(crash))[:, None]
    g = ix.build_grid("QQQ", bars)
    t = [x for x in ix.noise_trades(g) if x.day == g.days[-1]][0]
    assert t.outcome == "trail" and t.exit_min % 30 == 0 and t.exit_min > 210
    assert t.exit == pytest.approx(g.open[-1, t.exit_min])


def test_last_half_direction_and_times():
    g = ix.build_grid("QQQ", make_days(20, trend_last=0.02))
    t = [x for x in ix.last_half_trades(g, signal="day") if x.day == g.days[-1]][0]
    assert t.direction == "long" and t.entry_min == 360 and t.exit_min == 390
    assert t.entry == g.open[-1, 360] and t.exit == g.close[-1, 389]
    assert t.fill_time.hour == 15 and t.fill_time.minute == 30


def test_one_per_day_and_dollars():
    g = ix.build_grid("QQQ", make_days(20, trend_last=0.02))
    s = ix.build_grid("SPY", make_days(20, seed=9, trend_last=0.02))
    trades = ix.noise_trades(g) + ix.noise_trades(s) + ix.vwap_trades(g)
    picked = ix.one_per_day(trades)
    assert len({t.day for t in picked}) == len(picked)
    t = picked[-1]
    cm = rs.CostModel(0.10, 0.0, 0.8)
    assert ix.dollars(t, cm) == pytest.approx(100 * (0.8 * t.sign * (t.exit - t.entry) - 0.10))


def test_variants_all_run():
    g = ix.build_grid("QQQ", make_days(20, trend_last=0.02))
    res = ix.run_all({"QQQ": g})
    assert set(res) == set(ix.variants())
    for trades in res.values():
        for t in trades:
            assert t.entry_min < t.exit_min <= 390 and t.risk > 0
