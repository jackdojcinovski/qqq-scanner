from datetime import date

import numpy as np
import pandas as pd
import pytest

from scanner import spreads as sp
from scanner.history import ET
from scanner.options import occ_symbol

DAY = date(2026, 10, 27)


def chain_df(prices: dict) -> pd.DataFrame:
    """prices: (right, strike) -> list of (minute, close)."""
    rows = []
    for (r, k), pts in prices.items():
        for m, c in pts:
            t = pd.Timestamp(2026, 10, 27, 9, 30, tz=ET) + pd.Timedelta(minutes=m)
            rows.append((occ_symbol("QQQ", DAY, r, k), t.tz_convert("UTC"), c, c, c, c, 10))
    return pd.DataFrame(rows, columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"])


def test_load_chain_ffill_and_age():
    ch = sp.load_chain(chain_df({("P", 595): [(10, 1.0), (20, 0.8)]}), DAY, ET)
    p, a = ch.price[("P", 595.0)], ch.age[("P", 595.0)]
    assert np.isnan(p[9]) and p[10] == 1.0 and p[19] == 1.0 and p[25] == 0.8
    assert a[10] == 0 and a[19] == 9 and a[25] == 5 and a[5] > 1000


def test_put_spread_expires_worthless_keeps_credit():
    S = np.full(390, 600.0)
    legs = sp.legs_for("put", 600.0, 0.005, 2)          # short 597 P, long 595 P
    assert [(l.strike, l.short) for l in legs] == [(597, True), (595, False)]
    ch = sp.load_chain(chain_df({("P", 597): [(m, max(0.6 - m / 600, 0.01)) for m in range(0, 390)],
                                 ("P", 595): [(m, max(0.3 - m / 1200, 0.01)) for m in range(0, 390)]}), DAY, ET)
    r = sp.replay(ch, S, legs, 30, 2, None, None, slip=0.01, structure="put")
    assert r.credit == pytest.approx((0.6 - 29 / 600) - (0.3 - 29 / 1200) - 0.02)
    assert r.outcome == "close" and r.exit_bar == sp.EXIT_BAR and r.pnl > 0
    t = sp.replay(ch, S, legs, 30, 2, 0.5, None, slip=0.01)
    assert t.outcome == "target" and t.cost <= t.credit * 0.5 + 1e-9


def test_crash_hits_stop_and_intrinsic_floor_counts():
    S = np.full(390, 600.0)
    S[100:] = 590.0                                     # through both strikes
    legs = sp.legs_for("put", 600.0, 0.005, 2)
    # stale quotes: legs stop trading after minute 40 at low prices - intrinsic must still count
    ch = sp.load_chain(chain_df({("P", 597): [(m, 0.5) for m in range(0, 41)],
                                 ("P", 595): [(m, 0.2) for m in range(0, 41)]}), DAY, ET)
    r = sp.replay(ch, S, legs, 30, 2, None, 1.0, slip=0.01)
    assert r.outcome == "stop" and r.exit_bar == 100
    no_stop = sp.replay(ch, S, legs, 30, 2, None, None, slip=0.01)
    assert no_stop.cost == pytest.approx(2.02) and no_stop.pnl == pytest.approx((0.28 - 2.02) * 100)
    assert no_stop.max_loss == pytest.approx((2 - 0.28) * 100)


def test_no_trade_when_leg_is_stale_or_missing():
    S = np.full(390, 600.0)
    legs = sp.legs_for("put", 600.0, 0.005, 2)
    ch = sp.load_chain(chain_df({("P", 597): [(0, 0.6)], ("P", 595): [(0, 0.3)]}), DAY, ET)
    assert sp.replay(ch, S, legs, 60, 2, None, None) is None              # last trade 60 min ago
    ch2 = sp.load_chain(chain_df({("P", 597): [(m, 0.6) for m in range(60)]}), DAY, ET)
    assert sp.replay(ch2, S, legs, 30, 2, None, None) is None             # long leg missing


def test_condor_legs():
    legs = sp.legs_for("condor", 600.0, 0.005, 1)
    assert [(l.right, l.strike, l.short) for l in legs] == [("P", 597, True), ("P", 596, False),
                                                            ("C", 603, True), ("C", 604, False)]


def test_rejects_impossible_credit():
    S = np.full(390, 600.0)
    legs = sp.legs_for("put", 600.0, 0.005, 1)          # $1 wide: a $0.95 credit can't be real
    ch = sp.load_chain(chain_df({("P", 597): [(m, 1.20) for m in range(60)],
                                 ("P", 596): [(m, 0.20) for m in range(60)]}), DAY, ET)
    assert sp.replay(ch, S, legs, 30, 1, None, None) is None
