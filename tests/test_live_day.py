import numpy as np
import pandas as pd

from scanner import day_setups as ds
from scanner import live_day as ld
from scanner import orb_alert as oa
from tests.test_orb_alert import setup_day


def run(d, bars, chain, tmp_path, **kw):
    sent = []
    api = oa.ReplayAPI(d, bars, chain)
    res = ld.run_day(api, lambda c, e: sent.append(e[0]), "QQQ", journal=tmp_path / "j.csv",
                     log=lambda *a: None, **kw)
    return res, sent


def trend_path(i, p0):
    # quiet first 15 minutes, then a steady push up with heavier volume
    return p0 + (0.02 * np.sin(i) if i < 15 else 0.06 * (i - 14))


def test_setup_found_alerted_managed_and_logged(tmp_path):
    d, bars, chain, _ = setup_day(trend_path, lambda m: 2.0 if m < 18 else 3.5)
    day = bars.index.date == d
    t = bars.index[day]
    bars.loc[t[15:60], "volume"] *= 4
    res, sent = run(d, bars, chain, tmp_path)
    assert res["status"] in ("traded", "no quote"), res
    if res["status"] == "traded":
        entry = [e for e in sent if e["title"].startswith("ENTER")][0]
        assert "CALL" in entry["title"] and "Why this trade" in entry["description"]
        assert "confirmations" in entry["description"]
        assert any("TAKE PROFIT" in e["title"] or "STOP" in e["title"] or "TIME" in e["title"] for e in sent)
        assert sent[-1]["title"] == "Daily recap"
        rows = (tmp_path / "j.csv").read_text().splitlines()
        assert len(rows) == 2 and "traded" in rows[1]


def test_quiet_day_no_setup_is_logged(tmp_path):
    d, bars, chain, _ = setup_day(lambda i, p0: p0 + 0.01 * np.sin(i), lambda m: 2.0)
    res, sent = run(d, bars, chain, tmp_path)
    assert res["status"] == "no setup"
    assert "no trade today" in sent[-2]["title"] and sent[-1]["title"] == "Daily recap"
    assert "no setup" in (tmp_path / "j.csv").read_text()


def test_news_day(tmp_path):
    d, bars, chain, _ = setup_day(trend_path, lambda m: 2.0)
    res, sent = run(d, bars, chain, tmp_path, news_events=["08:30 CPI y/y"])
    assert res["status"] == "news" and "red-folder" in sent[0]["title"]


def test_evaluate_needs_vwap_side_and_four_confirmations():
    n = 40
    C = np.r_[np.full(15, 100.0), 100 + 0.1 * np.arange(1, n - 14)]
    O = np.r_[C[0], C[:-1]]
    H, L = np.maximum(O, C) + 0.02, np.minimum(O, C) - 0.02
    V = np.r_[np.full(15, 1000.0), np.full(n - 16, 3000.0), [9000.0]]
    ctx = {"avg_cum": np.cumsum(np.full(390, 1000.0)), "prev_high": 101.0, "prev_low": 99.0, "prev_close": 100.0}
    m = 26                                                      # the candle that first closes above 101 (PDH)
    V = V.copy()
    V[m - 1] = 9000.0
    sigs = ds.evaluate(O[:m], H[:m], L[:m], C[:m], V[:m], ctx)
    assert any(x.setup == "pdh_break" for x in sigs)
    assert sigs and all(s.direction == "long" and s.confirms["vwap"] and s.score >= 4 for s in sigs)
    down = C[:m][::-1].copy()
    assert not [x for x in ds.evaluate(O[:m], H[:m], L[:m], down, V[:m], ctx) if x.direction == "long"]
