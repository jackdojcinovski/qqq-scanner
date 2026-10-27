import pandas as pd
import pytest

from scanner.config import load_config
from scanner.news import NewsStatus
from scanner.scorer import DayState, evaluate
from tests.helpers import bars_from, bullish_sweep_day, candle, flat_day

CFG = load_config()


def test_bullish_sweep_setup_scores_alert():
    ev = evaluate(bullish_sweep_day(), CFG)
    s = ev.setup
    assert s is not None, ev.reason
    assert s.direction == "long"
    assert s.band(CFG) == "alert", [(c.name, c.points, c.detail) for c in s.components]
    hit = {c.name for c in s.components if c.hit}
    assert {"Liquidity sweep", "Structure shift", "Entry zone", "Order block", "VWAP side"} <= hit
    assert s.entry == pytest.approx(499.8)       # FVG midpoint
    assert s.stop < 498.85                        # beyond the sweep wick
    assert s.target > s.entry and s.rr >= CFG["trade"]["min_rr"]
    assert s.key == "2026-10-28-long-0950"
    assert sum(c.max_points for c in s.components) == 100


def test_outside_window_gate():
    hist = bullish_sweep_day()
    early = hist[hist.index <= pd.Timestamp("2026-10-28 09:33", tz="America/New_York")]
    assert "outside window" in evaluate(early, CFG).reason


def test_news_gate():
    ev = evaluate(bullish_sweep_day(), CFG, news=NewsStatus(True, "CPI at 08:30"))
    assert ev.setup is None and "CPI" in ev.reason


def test_daily_limits():
    assert "limit" in evaluate(bullish_sweep_day(), CFG, day=DayState(alerts_today=2)).reason
    assert "loss" in evaluate(bullish_sweep_day(), CFG, day=DayState(losses_today=1)).reason


def test_quiet_day_has_no_setup():
    prior = flat_day("2026-10-27")
    today = flat_day("2026-10-28", minutes=60)
    ev = evaluate(pd.concat([prior, today]), CFG)
    assert ev.setup is None


def test_setup_invalidated_when_price_below_stop():
    hist = bullish_sweep_day()
    crash = bars_from([candle(501.2, 498.5, low=498.4)], "2026-10-28 09:56")
    ev = evaluate(pd.concat([hist, crash]), CFG)
    assert ev.setup is None or ev.setup.direction == "short"


def test_weights_total_100():
    from scanner.config import SCORED_WEIGHTS
    assert sum(CFG["weights"][k] for k in SCORED_WEIGHTS) == 100


def test_sweep_without_structure_shift_is_not_a_setup():
    import copy
    cfg = copy.deepcopy(CFG)
    cfg["structure"]["mss_body_atr_mult"] = 50      # no candle can displace this much
    ev = evaluate(bullish_sweep_day(), cfg)
    assert ev.setup is None or "Liquidity sweep" not in {c.name for c in ev.setup.components if c.hit}


def test_replay_respects_daily_alert_limit():
    from datetime import date
    from scanner.replay import replay
    import copy
    cfg = copy.deepcopy(CFG)
    cfg["limits"]["max_alerts_per_day"] = 0
    assert [h for h in replay(bullish_sweep_day(), cfg, date(2026, 10, 28)) if h.band == "alert"] == []


def test_stop_sits_beyond_entry_zone_and_sweep():
    s = evaluate(bullish_sweep_day(), CFG).setup
    zone_bottom = 499.6
    assert s.stop < min(498.85, zone_bottom)


def test_reweighting_changes_score_without_new_candidates():
    import copy
    from scanner.scorer import candidates, select
    hist = bullish_sweep_day()
    cands, _ = candidates(hist, CFG)
    base, _ = select(cands, CFG)
    cfg = copy.deepcopy(CFG)
    cfg["weights"]["order_block"] = 0
    cfg["weights"]["orb_break"] = 15
    alt, _ = select(cands, cfg)
    assert alt.key == base.key and alt.score == base.score   # OB and ORB both hit: 5 moved
    cfg["weights"]["vwap_side"] = 0
    assert select(cands, cfg)[0].score == base.score - 15


def test_required_components_and_trigger_filter():
    import copy
    from scanner.scorer import candidates, select
    cands, _ = candidates(bullish_sweep_day(), CFG)
    cfg = copy.deepcopy(CFG)
    cfg["trade"]["required_components"] = ["Volume", "Order block"]
    assert select(cands, cfg)[0] is not None
    cfg["trade"]["triggers"] = ["orb"]
    best, why = select(cands, cfg)
    assert best is None or best.trigger == "orb"
    assert best is not None or "sweep trigger not traded" in why


def test_trend_gate_direction_filter_and_near_edge_entry():
    import copy
    hist = bullish_sweep_day()
    cfg = copy.deepcopy(CFG)
    cfg["trade"]["entry_at"] = "near_edge"
    s = evaluate(hist, cfg).setup
    assert s.entry == pytest.approx(500.0) and "near edge" in s.entry_zone   # FVG top for a long
    cfg["trade"]["directions"] = ["short"]
    assert evaluate(hist, cfg).setup is None
    cfg2 = copy.deepcopy(CFG)
    cfg2["trade"]["require_trend"] = True
    assert evaluate(hist, cfg2).setup.direction == "long"     # this day's 5-min trend is up


def test_classic_orb_candidate():
    import copy
    from scanner.scorer import candidates, select
    hist = bullish_sweep_day().copy()
    hist.loc[pd.Timestamp("2026-10-28 09:34", tz="America/New_York"), "close"] = 500.8   # first candle up
    t = pd.Timestamp("2026-10-28 09:35", tz="America/New_York")
    cands, _ = candidates(hist[hist.index <= t], CFG)
    classic = [c for c in cands if c.trigger == "orb_classic"]
    assert classic and all(c.market for c in classic)
    c = classic[0]
    # first 5-min candle 500.5 -> 500.5 open/close? use direction from the data
    first = hist[(hist.index >= pd.Timestamp("2026-10-28 09:30", tz="America/New_York")) & (hist.index < t)]
    up = first["close"].iloc[-1] > first["open"].iloc[0]
    assert c.direction == ("long" if up else "short")
    assert c.stop == pytest.approx((500.3 * (1 - 0.0002)) if up else (501.0 * 1.0002), abs=0.01)
    assert c.rr == 10.0 and c.key.endswith("orb5")
    later, _ = candidates(hist[hist.index <= t + pd.Timedelta(minutes=1)], CFG)
    assert not any(x.trigger == "orb_classic" for x in later)      # offered once
    cfg = copy.deepcopy(CFG)
    cfg["trade"]["triggers"] = ["orb_classic"]
    cfg["thresholds"]["alert_score"] = 0
    assert select(cands, cfg)[0].trigger == "orb_classic"
