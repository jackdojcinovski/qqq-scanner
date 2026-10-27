import copy
from datetime import date

import pandas as pd
import pytest

from scanner import backtest as bt
from scanner import options as op
from scanner import research as rs
from scanner.config import load_config
from tests.helpers import bullish_sweep_day

CFG = load_config()


def test_grid_names_parse_back():
    g = rs.grid()
    assert len(g) == 2 * 2 * 2 * 4 * 6 * 2
    assert rs.BASELINE in {v.name for v in g}
    for v in g[:10]:
        assert rs.variant(v.name).overrides == v.overrides
    with pytest.raises(ValueError):
        rs.variant("mid|any|both")
    with pytest.raises(ValueError):
        rs.variant("mid|any|both|all|fixed60|nope")


def test_baseline_variant_matches_config_defaults():
    cfg = rs.merge(CFG, rs.variant(rs.BASELINE).overrides)
    assert cfg["trade"]["entry_at"] == CFG["trade"]["entry_at"]
    assert cfg["trade"]["require_trend"] == CFG["trade"]["require_trend"]
    assert cfg["exit"]["mode"] == CFG["exit"]["mode"]
    assert cfg["thresholds"]["alert_score"] == CFG["thresholds"]["alert_score"]


def test_signature_tracks_detection_settings_only():
    a = rs.signature(CFG)
    c = copy.deepcopy(CFG)
    c["thresholds"]["alert_score"] = 90           # selection only
    c["exit"]["mode"] = "trail"
    c["trade"]["require_trend"] = True
    assert rs.signature(c) == a
    c["structure"]["mss_body_atr_mult"] = 1.2      # changes detection
    assert rs.signature(c) != a


def test_cached_path_equals_direct_path():
    """Candidates cached with both entry modes + select == scanning with one config."""
    hist = bullish_sweep_day()
    day = date(2026, 10, 28)
    dc = bt.day_candidates(hist, day, CFG)        # both entry modes, like the cache
    for name in ["mid|any|both|all|fixed60|score75", "edge|with|long|all|partial|checklist",
                 "edge|any|both|orb|trail|score75"]:
        cfg = rs.merge(CFG, rs.variant(name).overrides)
        cached = [(r.key, r.score, r.entry, r.r, r.taken) for r in bt.run_variant(dc, cfg)]
        direct = [(r.key, r.score, r.entry, r.r, r.taken) for r in bt.run_day(hist, day, cfg)]
        assert cached == direct, name


def test_cache_build_and_reuse(tmp_path):
    bars = {"QQQ": bullish_sweep_day()}
    p1 = rs.build_cache(bars, CFG, workers=1, base=tmp_path)
    mtime = p1[0].stat().st_mtime
    p2 = rs.build_cache(bars, CFG, workers=1, base=tmp_path)
    assert p1 == p2 and p2[0].stat().st_mtime == mtime
    res = rs.run_grid(p2, CFG, [rs.variant(rs.BASELINE)], workers=1)
    assert rs.BASELINE in res


def _row(r, risk=1.0, hold=30.0, year=2025, price=500.0):
    return rs.Row("QQQ", date(year, 3, 3), True, r, risk, price, hold, "long", "orb", "target", 10)


def test_cost_model():
    cm = rs.CostModel(spread=0.03, theta_pct_per_hour=0.0003, delta=0.45)
    # $0.03 + 0.0003*500*0.5h = $0.105 per share of option; / (0.45 * $1 risk) = 0.233R
    assert cm.cost_r(_row(1.0)) == pytest.approx(0.105 / 0.45)
    assert cm.cost_r(_row(1.0, risk=2.0)) == pytest.approx(0.105 / 0.9)   # wider stop, smaller cost in R


def test_metrics_and_ranking_prefers_consistent():
    cm = rs.CostModel(0, 0, 1)
    steady = [_row(0.3, year=y) for y in (2023, 2024, 2025) for _ in range(40)]
    lucky = [_row(-1.0, year=y) for y in (2023, 2024, 2025) for _ in range(38)] + [_row(25.0)] * 3
    m = rs.metrics(steady, 300, cm)
    assert m.avg_net == pytest.approx(0.3) and m.years_positive == "3/3"
    days = {r.day for r in steady + lucky}
    ranked = rs.rank({"steady": steady, "lucky": lucky}, days, min_filled=10, cost=cm)
    assert ranked[0][0] == "steady"                     # lucky has higher mean but wide CI


def test_occ_symbol_and_strikes():
    assert op.occ_symbol("QQQ", date(2025, 1, 17), "C", 500) == "QQQ250117C00500000"
    assert op.occ_symbol("SPY", date(2026, 3, 2), "P", 612.5) == "SPY260302P00612500"
    assert op.atm_strike(500.4) == 500 and op.atm_strike(500.6) == 501
    assert op.atm_strike(500.4, offset_steps=1, right="C") == 501
    assert op.atm_strike(500.4, offset_steps=1, right="P") == 499
    assert op.expiry_candidates(date(2026, 10, 23), 1)[0] == date(2026, 10, 26)   # Fri -> Mon


def test_option_trade_pricing():
    idx = pd.date_range("2026-10-28 10:00", periods=5, freq="1min", tz="America/New_York")
    bars = pd.DataFrame({"close": [2.00, 2.10, 2.40, 2.60, 2.50]}, index=idx)
    rec = bt.TradeRecord(date(2026, 10, 28), idx[0], "k", "long", 80, "alert", True, "", 1, 0, 2, "t", 2,
                         "FVG", "target", 1.5, True, idx[1], idx[3])
    t = op.option_trade(rec, bars, "QQQ261029C00500000", spread=0.03)
    assert t.entry == 2.10 and t.exit == 2.60
    assert t.pnl == pytest.approx((2.60 - 2.10 - 0.03) * 100)
    late = bt.TradeRecord(**{**rec.__dict__, "exit_time": idx[-1] + pd.Timedelta(minutes=30)})
    assert op.option_trade(late, bars, "x", 0.03) is None        # stale price -> skipped


def test_itm_strike_reference():
    entry = 700.0
    call_ref, put_ref = entry * (1 - 0.03), entry * (1 + 0.03)
    assert op.atm_strike(call_ref, 1.0) == 679 and op.atm_strike(put_ref, 1.0) == 721
    assert op.atm_strike(203.4 * 0.97, 2.5) == 197.5
