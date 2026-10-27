import numpy as np

from scanner import bigmove as bm
from scanner import intraday as ix
from tests.test_intraday import make_days


def test_signs_ignore_the_future():
    bars = make_days(40, seed=4, trend_last=0.02, vol=0.05)
    g = ix.build_grid("QQQ", bars)
    i, m = len(g.days) - 1, 45
    a = bm.Features(g).signs(i, m, "long")
    assert a is not None and set(a) == set(bm.SIGNS)
    b2 = bars.copy()
    day = g.days[-1]
    t_m = bars.index[bars.index.date == day][m]
    later = (bars.index.date == day) & (bars.index >= t_m)
    b2.loc[later, ["open", "high", "low", "close"]] *= 1.05
    b2.loc[later, "volume"] *= 10
    assert bm.Features(ix.build_grid("QQQ", b2)).signs(i, m, "long") == a


def test_rvol_and_moving_fire_on_a_busy_trend_day():
    bars = make_days(40, seed=4, trend_last=0.03, vol=0.05)
    last = bars.index.date == bars.index.date[-1]
    bars.loc[last, "volume"] *= 3
    g = ix.build_grid("QQQ", bars)
    s = bm.Features(g).signs(len(g.days) - 1, 60, "long")
    assert s["rvol"] and s["moving"]
    assert not bm.Features(g).signs(len(g.days) - 1, 60, "short")["moving"]


def test_max_move():
    g = ix.build_grid("QQQ", make_days(30, trend_last=0.02))
    i = len(g.days) - 1
    up = bm.max_move(g, i, 30, "long", 60)
    assert up == np.max(g.high[i, 30:90]) / g.open[i, 30] - 1 and up > 0
