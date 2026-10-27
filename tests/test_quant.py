from datetime import date

import numpy as np
import pandas as pd
import pytest

from scanner import quant as q
from scanner import research as rs
from tests.helpers import bars_from


def days_of_bars(n_days=40, start="2026-01-05", seed=1, drift=0.0):
    rng = np.random.default_rng(seed)
    frames, p = [], 100.0
    d = pd.Timestamp(start)
    made = 0
    while made < n_days:
        if d.weekday() < 5:
            rows = []
            for i in range(390):
                o = p
                c = o + rng.normal(drift, 0.05)
                rows.append((o, max(o, c) + 0.02, min(o, c) - 0.02, c, 1000 + i))
                p = c
            frames.append(bars_from(rows, f"{d.date()} 09:30"))
            made += 1
        d += pd.Timedelta(days=1)
    return pd.concat(frames)


def test_daily_table_has_no_lookahead():
    bars = days_of_bars(40)
    tab = q.daily_table(bars)
    day = tab.index[30]
    # change everything from `day` on: prior-day stats for `day` must not move
    cut = bars[bars.index.date < day]
    later = bars[bars.index.date >= day].copy()
    later[["open", "high", "low", "close"]] *= 1.5
    tab2 = q.daily_table(pd.concat([cut, later]))
    for col in ["prev_close", "sma20_prev", "atr14_prev", "rv14_prev", "noise_prev", "o5_vol_avg_prev"]:
        assert tab.loc[day, col] == pytest.approx(tab2.loc[day, col]), col


def test_daily_table_opening_window_only():
    bars = days_of_bars(25)
    tab = q.daily_table(bars)
    d = tab.index[-1]
    day = bars[bars.index.date == d]
    first = day[day.index.time < pd.Timestamp("09:35").time()]
    assert tab.loc[d, "o5_ret"] == pytest.approx(first["close"].iloc[-1] / first["open"].iloc[0] - 1)
    assert tab.loc[d, "o5_vol"] == first["volume"].sum()


def test_signs_direction_aware():
    row = pd.Series({"open": 101, "prev_close": 100, "sma20_prev": 95, "o5_ret": 0.004, "noise_prev": 0.002,
                     "o5_vol": 1500, "o5_vol_avg_prev": 1000, "rv14_prev": 0.01, "rv14_med_prev": 0.02,
                     "o5_range": 0.5, "atr14_prev": 4, "prev_open": 99})
    up = q.signs_for(row, "long", 0.003)
    assert all(up.values())
    down = q.signs_for(row, "short", 0.003)
    assert not down["gap_with"] and not down["daily_trend"] and not down["peers_agree"]
    assert down["opening_volume"] and down["calm_regime"] and down["tight_range"]   # direction-free signs
    nan_row = row.copy()
    nan_row["sma20_prev"] = float("nan")
    assert q.signs_for(nan_row, "long", None)["daily_trend"] is False


def _gt(day, sym, r, signs):
    row = rs.Row(sym, day, True, r, 1.0, 100.0, 60, "long", "orb_classic", "close", 9)
    return q.GradedTrade(sym, day, "long", r, True, {k: (k in signs) for k in q.SIGNS}, row)


def test_choose_signs_cutoff_and_one_per_day():
    trades = []
    for i in range(200):
        d = date(2025, 1, 1) + pd.Timedelta(days=i).to_pytimedelta()
        good = i % 2 == 0
        trades.append(_gt(d, "QQQ", 1.0 if good else -0.5, {"gap_with"} if good else {"calm_regime"}))
    edges = {e.sign: e for e in q.sign_edges(trades)}
    assert edges["gap_with"].edge > 1 and edges["calm_regime"].edge < 0
    used = q.choose_signs(list(edges.values()), min_side=40)
    assert used == ["gap_with"]
    assert q.cutoff_for_top(trades, used, 0.25) == 1      # 50% have the sign: closest to 25%
    assert q.cutoff_for_top(trades, used, 0.9) == 0
    d = date(2025, 6, 1)
    picked = q.one_per_day([_gt(d, "SPY", 1, {"gap_with"}), _gt(d, "QQQ", 1, {"gap_with"}),
                            _gt(d, "SPY", 1, set())], used, 1)
    assert len(picked) == 1 and picked[0].symbol == "QQQ"       # tie on grade -> QQQ preferred
