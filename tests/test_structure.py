import pandas as pd
import pytest

from scanner import indicators as ind
from scanner import structure as st
from scanner.levels import Level
from tests.helpers import bars_from, bullish_sweep_day, candle


def session_of(hist):
    s = ind.regular_session(hist)
    return s[s.index.date == hist.index[-1].date()]


def test_swing_points():
    highs = [1, 2, 5, 2, 1, 2, 1]
    rows = [(h, h, h - 0.5, h, 1) for h in highs]
    df = bars_from(rows, "2026-10-28 09:30")
    sw = st.swing_points(df, 2)
    hs = [s for s in sw if s.kind == "high"]
    assert [(s.idx, s.price) for s in hs] == [(2, 5)]
    assert hs[0].confirmed_idx == 4


def test_equal_highs():
    rows = [(1, h, h - 1, 1, 1) for h in [1, 2, 10.0, 2, 1, 2, 10.001, 2, 1]]
    df = bars_from(rows, "2026-10-28 09:30")
    lv = st.equal_levels(df, st.swing_points(df, 2), 0.0002)
    assert len(lv) == 1 and lv[0].name == "equal highs"
    assert lv[0].price == pytest.approx(10.001)


def test_sweep_needs_close_back_inside():
    lvl = [Level("pm low", 100.0, pd.Timestamp("2026-10-28 09:30", tz="America/New_York"))]
    swept = bars_from([candle(100.5, 100.4), candle(100.4, 100.2, low=99.9)], "2026-10-28 09:30")
    s = st.find_sweeps(swept, lvl)
    assert len(s) == 1 and s[0].direction == "long" and s[0].extreme == 99.9
    broke = bars_from([candle(100.5, 100.4), candle(100.4, 99.8, low=99.7), candle(99.8, 99.7),
                       candle(99.7, 99.6)], "2026-10-28 09:30")
    assert st.find_sweeps(broke, lvl) == []


def test_sweep_reclaim_on_next_bar_and_short_side():
    t0 = pd.Timestamp("2026-10-28 09:30", tz="America/New_York")
    df = bars_from([candle(100.5, 100.4), candle(100.4, 99.9, low=99.8), candle(99.9, 100.3)],
                   "2026-10-28 09:30")
    s = st.find_sweeps(df, [Level("x", 100.0, t0)])
    assert s[0].reclaim_idx == 2
    up = bars_from([candle(99.5, 99.6), candle(99.6, 99.8, high=100.2)], "2026-10-28 09:30")
    s2 = st.find_sweeps(up, [Level("y", 100.0, t0)])
    assert s2[0].direction == "short" and s2[0].extreme == 100.2


def test_sweep_ignores_pierce_below_threshold():
    t0 = pd.Timestamp("2026-10-28 09:30", tz="America/New_York")
    df = bars_from([candle(100.5, 100.4), candle(100.4, 100.2, low=99.99)], "2026-10-28 09:30")
    assert st.find_sweeps(df, [Level("x", 100.0, t0)], pierce_pct=0.0002) == []


def test_fvg_detection_fill_and_inversion():
    # bullish FVG between bar0 high 100.1 and bar2 low 100.5
    base = [candle(100.0, 100.05, wick=0.05), candle(100.1, 100.9, wick=0.05),
            candle(100.9, 101.0, low=100.5)]
    df = bars_from(base, "2026-10-28 09:30")
    a = ind.atr(df)
    gaps = st.find_gaps(df, a, 0.25)
    assert len(gaps) == 1 and gaps[0].kind == "FVG" and gaps[0].direction == "long"
    assert (gaps[0].bottom, gaps[0].top) == (100.1, 100.5)

    filled = bars_from(base + [candle(100.9, 100.3, low=100.05)], "2026-10-28 09:30")
    # close 100.3 is inside the gap but the low traded through the bottom -> dead
    assert st.find_gaps(filled, ind.atr(filled), 0.25) == []

    inverted = bars_from(base + [candle(100.9, 99.9, low=99.8)], "2026-10-28 09:30")
    g = st.find_gaps(inverted, ind.atr(inverted), 0.25)
    assert len(g) == 1 and g[0].kind == "IFVG" and g[0].direction == "short"
    assert g[0].active_from == 3


def test_fvg_expiry():
    base = [candle(100.0, 100.05), candle(100.1, 100.9), candle(100.9, 101.0, low=100.5)]
    tail = [candle(101.0, 101.0)] * 61
    df = bars_from(base + tail, "2026-10-28 09:30")
    assert st.find_gaps(df, ind.atr(df), 0.25, expiry_bars=60) == []


def test_full_pattern_sweep_mss_fvg_ob():
    hist = bullish_sweep_day()
    sess = session_of(hist)
    from scanner.levels import build_levels
    lv = build_levels(hist)
    sweeps = [s for s in st.find_sweeps(sess, lv) if s.direction == "long"]
    sweep = next(s for s in sweeps if s.level_name == "premarket low")
    assert sweep.idx == 20 and sweep.extreme == 498.85

    atr = ind.atr(sess)
    swings = st.swing_points(sess)
    mss = st.find_mss(sess, sweep, swings, atr)
    assert mss is not None and mss.idx == 23 and mss.broken_level == 500.95

    gaps = [g for g in st.find_gaps(sess, atr) if g.direction == "long"]
    fvg = next(g for g in gaps if g.created_idx == 24)
    assert (fvg.bottom, fvg.top) == (499.6, 500.0)

    ob = st.find_order_block(sess, sweep, mss)
    assert ob is not None and ob.idx == 22
    assert st.overlaps(ob.top, ob.bottom, fvg.top, fvg.bottom)


def test_high_level_retest_from_above_is_not_a_long_sweep():
    t0 = pd.Timestamp("2026-10-28 09:30", tz="America/New_York")
    df = bars_from([candle(100.5, 100.4), candle(100.4, 100.2, low=99.9)], "2026-10-28 09:30")
    assert st.find_sweeps(df, [Level("premarket high", 100.0, t0, "high")]) == []
    assert len(st.find_sweeps(df, [Level("premarket low", 100.0, t0, "low")])) == 1
