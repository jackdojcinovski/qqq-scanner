import math

import numpy as np
import pytest

from scanner import intraday as ix
from scanner import optionsim as om
from tests.test_intraday import make_days


def test_black_scholes_parity_and_decay():
    S, K, T, v = 500.0, 495.0, 10 / 365, 0.2
    c, p = om.bs_price(S, K, T, v, "C"), om.bs_price(S, K, T, v, "P")
    assert c - p == pytest.approx(S - K * math.exp(-om.RATE * T), abs=1e-4)
    assert om.bs_price(S, K, 1 / 365, v, "C") < c                    # less time, less value
    assert om.bs_price(S, 510, 1e-9, v, "C") == pytest.approx(0, abs=1e-6)   # expired OTM is worthless
    assert om.bs_price(S, 490, 1e-9, v, "C") == pytest.approx(10, abs=0.01)


def test_expiry_skips_weekends():
    from datetime import date
    assert om.expiry_date(date(2026, 10, 23), 1) == date(2026, 10, 26)      # Fri + 1 -> Mon
    assert om.expiry_date(date(2026, 10, 27), 0) == date(2026, 10, 27)


def test_regime_uses_prior_days_only():
    bars = make_days(30)
    g = ix.build_grid("QQQ", bars)
    ctx = om.daily_context(g)
    b2 = bars.copy()
    last = bars.index.date == g.days[-1]
    b2.loc[last, ["open", "high", "low", "close"]] *= 0.5
    ctx2 = om.daily_context(ix.build_grid("QQQ", b2))
    assert ctx.iloc[-1].equals(ctx2.iloc[-1])


def _path(paid, best, worst, opn, final):
    return om.Path(paid, np.array(worst, float), np.array(best, float), np.array(opn, float), final, 500, None)


def test_replay_target_stop_and_strategy_exit():
    p = _path(2.0, best=[2.5, 4.2, 5.0], worst=[1.8, 3.0, 4.0], opn=[2.0, 2.6, 4.5], final=3.0)
    assert om.replay(p, 1.0, None) == (4.0, "target", 1)               # +100% hit in bar 1
    assert om.replay(p, 2.0, None) == (3.0, "strategy", 3)             # +200% never hit
    assert om.replay(p, 0.25, None) == (2.5, "target", 0)
    s = _path(2.0, best=[2.1, 2.2], worst=[1.5, 0.8], opn=[2.0, 1.4], final=1.0)
    assert om.replay(s, 1.0, 0.5) == (1.0, "stop", 1)
    gap = _path(2.0, best=[2.1, 0.7], worst=[1.5, 0.5], opn=[2.0, 0.6], final=1.0)
    assert om.replay(gap, None, 0.5) == (0.6, "stop", 1)              # gapped through the stop: worse fill
    both = _path(2.0, best=[4.5], worst=[0.9], opn=[2.0], final=1.0)
    assert om.replay(both, 1.0, 0.5)[1] == "stop"                     # same minute: stop first


def test_pnl_includes_canadian_fx():
    sig = om.Signal("x", "QQQ", None, "long", 30, 60, 0, 60)
    r = om.Result(sig, "c", paid=2.0, got=4.0, outcome="target", exit_bar=40)
    assert r.ret == pytest.approx(1.0)
    assert r.pnl(5) == pytest.approx(5 * 100 * (2.0 - 0.015 * 6.0))     # $1,000 gross - $45 FX


def test_end_to_end_up_day_call_makes_money():
    bars = make_days(30, trend_last=0.02)
    bars[["open", "high", "low", "close"]] *= 5          # QQQ-like price level
    g = ix.build_grid("QQQ", bars)
    grids = {"QQQ": g}
    ctx = {"QQQ": om.daily_context(g)}
    sigs = om.from_intraday([t for t in ix.noise_trades(g, sides="long") if t.day == g.days[-1]], grids)
    assert len(sigs) == 1
    for c in om.CONTRACTS:
        paths = om.build_paths(sigs, grids, ctx, c)
        assert len(paths) == 1
        res = om.run(paths, c.name, None, None)[0]
        assert res.got > res.paid, c.name
        assert om.run(paths, c.name, 0.5, None)[0].ret == pytest.approx(0.5, rel=1e-6) or \
            om.run(paths, c.name, 0.5, None)[0].outcome == "strategy"
    # 0DTE held to 15:59:30 is worth about its intrinsic value
    p0 = om.build_paths(sigs, grids, ctx, om.CONTRACTS[0])[0][1]
    intrinsic = max(sigs[0].exit_px - p0.strike, 0)
    assert p0.final_bid == pytest.approx(intrinsic - om.half_spread(intrinsic), abs=0.02)


def test_apply_regime_modes():
    from datetime import date
    import pandas as pd
    d1, d2 = date(2026, 1, 5), date(2026, 1, 6)
    ctx = {"QQQ": pd.DataFrame({"down_k1": [False, True]}, index=[d1, d2])}
    S = lambda d, side: om.Signal("x", "QQQ", d, side, 30, 60, 0, 60)   # noqa: E731
    sigs = [S(d1, "long"), S(d1, "short"), S(d2, "long"), S(d2, "short")]
    assert [(s.day, s.direction) for s in om.apply_regime(sigs, ctx, "long")] == [(d1, "long"), (d2, "long")]
    assert [(s.day, s.direction) for s in om.apply_regime(sigs, ctx, "long_up_k1")] == [(d1, "long")]
    assert [(s.day, s.direction) for s in om.apply_regime(sigs, ctx, "regime_k1")] == [(d1, "long"), (d2, "short")]
