import numpy as np

from scanner import intraday as ix
from scanner import library as lib
from scanner import optionsim as om
from tests.test_intraday import make_days


def _sigs(bars):
    g = ix.build_grid("QQQ", bars)
    ctx = om.daily_context(g)
    return g, lib.day_signals(g, len(g.days) - 1, ctx.loc[g.days[-1]])


def test_every_setup_fires_somewhere_and_is_well_formed():
    seen = set()
    for seed in range(12):
        for drift in (0.02, -0.02, 0.0):
            g, sigs = _sigs(make_days(30, seed=seed, trend_last=drift, vol=0.05))
            for s in sigs:
                seen.add(s.strategy)
                assert 1 <= s.m_in < ix.MINUTES - 1 and s.m_end == ix.MINUTES - 1
                assert s.direction in ("long", "short")
    assert seen >= {"orb5", "orb15", "orb30", "vwap10", "vwap_cross", "noise", "pdh", "momo_1030"}


def test_no_lookahead_for_any_setup():
    for seed in range(8):
        bars = make_days(30, seed=seed, trend_last=0.02 if seed % 2 else -0.02, vol=0.05)
        g, sigs = _sigs(bars)
        day = g.days[-1]
        for s in sigs:
            # scramble every bar from the entry bar on: the decision must not change
            b2 = bars.copy()
            t_in = bars.index[bars.index.date == day][s.m_in]
            later = (bars.index.date == day) & (bars.index >= t_in)
            rng = np.random.default_rng(99)
            b2.loc[later, ["open", "high", "low", "close"]] *= rng.uniform(0.95, 1.05, size=(later.sum(), 1))
            _, sigs2 = _sigs(b2)
            same = [x for x in sigs2 if x.strategy == s.strategy]
            assert same and (same[0].direction, same[0].m_in) == (s.direction, s.m_in), s.strategy


def test_orb5_direction():
    bars = make_days(30, trend_last=0.03)
    _, sigs = _sigs(bars)
    orb = [s for s in sigs if s.strategy == "orb5"]
    assert orb and orb[0].direction == "long" and orb[0].m_in == 5


def test_first_pullback_and_orb_retest_shapes():
    import numpy as np
    from tests.helpers import bars_from
    bars = make_days(30, seed=2, vol=0.03)
    last = bars.index.date[-1]
    day = bars[bars.index.date == last]
    p0 = float(day["open"].iloc[0])
    path = np.concatenate([np.linspace(p0, p0 * 1.006, 15),        # drive up 0.6% by 9:45
                           np.linspace(p0 * 1.006, p0 * 1.0040, 5),   # quick 33% pullback, above VWAP
                           np.linspace(p0 * 1.0040, p0 * 1.009, 40),  # new high of day
                           np.full(330, p0 * 1.009)])
    rows = [(a, max(a, b) + 0.005, min(a, b) - 0.005, b, 1000) for a, b in zip(np.r_[p0, path[:-1]], path)]
    new = bars_from(rows, f"{last} 09:30")
    bars = bars[bars.index.date != last]
    import pandas as pd
    g = ix.build_grid("QQQ", pd.concat([bars, new]))
    ctx = om.daily_context(g)
    sigs = {s.strategy: s for s in lib.day_signals(g, len(g.days) - 1, ctx.loc[g.days[-1]])}
    assert "first_pull" in sigs and sigs["first_pull"].direction == "long"
    assert 20 < sigs["first_pull"].m_in <= 60
