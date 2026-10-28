import numpy as np
import pytest

from scanner import debit as db
from scanner import spreads as sp
from tests.test_spreads import DAY, ET, chain_df


def test_legs():
    legs, cap = db.legs_for("ds3", 600.4, "long")
    assert [(l.right, l.strike, l.short) for l in legs] == [("C", 600, False), ("C", 603, True)] and cap == 3
    legs, cap = db.legs_for("ds2", 600.4, "short")
    assert [(l.right, l.strike, l.short) for l in legs] == [("P", 600, False), ("P", 598, True)]
    assert db.legs_for("itm2", 600.0, "long")[0][0].strike == 598
    assert db.legs_for("itm2", 600.0, "short")[0][0].strike == 602
    assert db.legs_for("otm1", 600.0, "short")[0][0].strike == 599


def test_call_spread_rally_target_and_cap():
    S = np.full(390, 600.0)
    S[100:] = 606.0
    ch = sp.load_chain(chain_df({("C", 600): [(m, 1.5 if m < 100 else 6.0) for m in range(390)],
                                 ("C", 603): [(m, 0.4 if m < 100 else 3.1) for m in range(390)]}), DAY, ET)
    legs, cap = db.legs_for("ds3", 600.0, "long")
    debit, value = db.prepare(ch, S, legs, 30, cap, slip=0.01)
    assert debit == pytest.approx(1.5 - 0.4 + 0.02)
    assert value.max() <= 3 - 0.02 + 1e-9                     # never worth more than the width
    j, why = db.exit_on(debit, value, 30, 1.0, None, None)
    assert why == "target" and 30 + j == 100
    j, why = db.exit_on(debit, value, 30, None, None, 60)
    assert why == "time" and 30 + j == 60


def test_plain_call_decays_and_stops():
    S = np.full(390, 600.0)
    ch = sp.load_chain(chain_df({("C", 600): [(m, max(1.6 * (1 - m / 390), 0.01)) for m in range(390)]}), DAY, ET)
    legs, cap = db.legs_for("atm", 600.0, "long")
    debit, value = db.prepare(ch, S, legs, 30, cap)
    j, why = db.exit_on(debit, value, 30, None, 0.5, None)
    assert why == "stop" and value[j] <= debit * 0.5
    j, why = db.exit_on(debit, value, 30, None, None, None)
    assert why == "close" and value[j] < debit * 0.1


def test_rejects_impossible_spread_price():
    S = np.full(390, 600.0)
    # short leg's stale print almost equals the long leg: a $0.05 debit on a $2 spread is not real
    ch = sp.load_chain(chain_df({("C", 600): [(m, 1.50) for m in range(60)],
                                 ("C", 602): [(m, 1.47) for m in range(60)]}), DAY, ET)
    legs, cap = db.legs_for("ds2", 600.0, "long")
    assert db.prepare(ch, S, legs, 30, cap) is None


def test_scale_out_plan():
    debit = 1.0
    #           +30%  +60%(half out)  +20%  -0% (BE stop)
    value = np.array([1.3, 1.6, 1.2, 0.99, 3.0])
    avg, held, why = db.run_plan(debit, value, 30, ("scale", 1.0, 0.25), None)
    assert avg == pytest.approx(0.5 * 1.5 + 0.5 * 0.99) and held == 3 and why == "half +50%, rest breakeven"
    avg, held, why = db.run_plan(debit, np.array([1.3, 1.6, 2.1]), 30, ("scale", 1.0, 0.25), None)
    assert avg == pytest.approx(0.5 * 1.5 + 0.5 * 2.0) and why == "half +50%, rest target 2"   # fills AT targets
    avg, held, why = db.run_plan(debit, np.array([0.9, 0.7, 2.0]), 30, ("scale", 1.0, 0.25), None)
    assert avg == pytest.approx(0.7) and held == 1 and why == "stop"   # stopped before the first target
    avg, _, why = db.run_plan(debit, np.array([1.2, 1.55, 0.6]), 30, ("fixed", 0.5, 0.25), None)
    assert avg == pytest.approx(1.5) and why == "target"
