import numpy as np
import pandas as pd
import pytest

from scanner import indicators as ind
from scanner.levels import build_levels
from tests.helpers import bars_from, bullish_sweep_day, candle


def test_vwap_matches_hand_calc():
    df = bars_from([(10, 11, 9, 10, 100), (10, 13, 11, 12, 300)], "2026-10-28 09:30")
    vwap = ind.session_vwap(df)
    assert vwap.iloc[0] == pytest.approx(10.0)
    assert vwap.iloc[1] == pytest.approx((10 * 100 + 12 * 300) / 400)


def test_vwap_resets_each_day():
    a = bars_from([(10, 10, 10, 10, 100)], "2026-10-27 15:59")
    b = bars_from([(20, 20, 20, 20, 100)], "2026-10-28 09:30")
    vwap = ind.session_vwap(pd.concat([a, b]))
    assert vwap.iloc[1] == pytest.approx(20.0)


def test_vwap_zero_volume_carries_forward():
    df = bars_from([(10, 10, 10, 10, 100), (12, 12, 12, 12, 0)], "2026-10-28 09:30")
    assert ind.session_vwap(df).iloc[1] == pytest.approx(10.0)


def test_ema_known_values():
    s = pd.Series([1.0, 2.0, 3.0])
    # span 3 -> alpha 0.5
    assert ind.ema(s, 3).tolist() == pytest.approx([1.0, 1.5, 2.25])


def test_rsi_extremes_and_range():
    up = pd.Series(np.arange(1, 40, dtype=float))
    assert ind.rsi(up, 14).iloc[-1] == pytest.approx(100.0)
    down = pd.Series(np.arange(40, 1, -1, dtype=float))
    assert ind.rsi(down, 14).iloc[-1] == pytest.approx(0.0)
    rng = np.random.default_rng(1)
    noisy = pd.Series(100 + rng.normal(0, 1, 200).cumsum())
    r = ind.rsi(noisy, 14).dropna()
    assert ((r >= 0) & (r <= 100)).all()
    assert ind.rsi(noisy, 14).iloc[:13].isna().all()


def test_atr_uses_gaps():
    df = bars_from([(10, 11, 9, 10, 1), (15, 16, 14, 15, 1)], "2026-10-28 09:30")
    a = ind.atr(df, 14)
    assert a.iloc[0] == pytest.approx(2.0)
    # bar 2 true range = 16 - 10 = 6 -> 2 + (6 - 2)/14
    assert a.iloc[1] == pytest.approx(2 + 4 / 14)


def test_resample_5m_and_drops_incomplete():
    rows = [candle(100 + i, 100.5 + i, vol=10) for i in range(7)]
    df = bars_from(rows, "2026-10-28 09:30")
    five = ind.resample_5m(df)
    assert len(five) == 1                     # 9:35 bar only has 2 of 5 minutes
    assert five["open"].iloc[0] == 100
    assert five["close"].iloc[0] == pytest.approx(104.5)
    assert five["volume"].iloc[0] == 50
    assert len(ind.resample_5m(df, complete_only=False)) == 2


def test_opening_range_waits_for_completion():
    rows = [candle(100, 101, high=102, low=99)] + [candle(100, 100.5)] * 3
    df = bars_from(rows, "2026-10-28 09:30")
    assert ind.opening_range(df, 5) is None
    df2 = bars_from(rows + [candle(100, 100.2)], "2026-10-28 09:30")
    assert ind.opening_range(df2, 5) == (102, 99)


def test_relative_volume():
    rows = [candle(100, 100, vol=100)] * 20 + [candle(100, 100, vol=300)]
    df = bars_from(rows, "2026-10-28 09:30")
    assert ind.relative_volume(df, 20).iloc[-1] == pytest.approx(3.0)


def test_build_levels():
    hist = bullish_sweep_day()
    names = {l.name: l.price for l in build_levels(hist)}
    assert names["prior day high"] == 505.0
    assert names["prior day low"] == 495.0
    assert names["premarket high"] == 501.5
    assert names["premarket low"] == 499.0
    assert names["opening range high"] == 501.0
    assert names["opening range low"] == 500.3
