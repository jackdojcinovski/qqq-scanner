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
    from scanner.scorer import _Context, _score_direction
    from scanner import indicators as ind
    hist = bullish_sweep_day()
    rth = ind.regular_session(hist)
    sess = rth[rth.index.date == hist.index[-1].date()]
    s = _score_direction(_Context.build(hist, sess, CFG), "long")
    zone_bottom = 499.6
    assert s.stop < min(498.85, zone_bottom)
