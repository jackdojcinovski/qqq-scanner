from datetime import date

import numpy as np
import pandas as pd

from scanner import orb_alert as oa
from scanner.history import ET
from tests.test_intraday import make_days

DAY = None


def setup_day(path_fn, price_fn):
    """30 quiet days, then a last day whose minute path and ATM option price we control."""
    bars = make_days(30, seed=5, vol=0.03)
    bars[["open", "high", "low", "close"]] *= 6          # QQQ-like ~600
    d = bars.index.date[-1]
    day = bars[bars.index.date == d].copy()
    p0 = float(day["open"].iloc[0])
    path = np.array([path_fn(i, p0) for i in range(390)])
    opn = np.r_[p0, path[:-1]]
    day["open"], day["close"] = opn, path
    day["high"], day["low"] = np.maximum(opn, path) + 0.02, np.minimum(opn, path) - 0.02
    bars = pd.concat([bars[bars.index.date != d], day])
    strike = round(path[4])
    right = "C" if path[4] > p0 else "P"
    sym = oa.occ("QQQ", d, right, strike)
    rows = [(sym, (pd.Timestamp(d.year, d.month, d.day, 9, 30, tz=ET) + pd.Timedelta(minutes=m)).tz_convert("UTC"),
             price_fn(m)) for m in range(390)]
    chain = pd.DataFrame(rows, columns=["symbol", "timestamp", "close"])
    return d, bars, chain, strike


def run(d, bars, chain, **kw):
    sent = []
    api = oa.ReplayAPI(d, bars, chain)
    res = oa.run_day(api, lambda c, e: sent.append(e[0]), "QQQ", oa.Plan(poll_s=5), log=lambda *a: None, **kw)
    return res, sent


def test_green_candle_buys_call_and_hits_take_profit():
    d, bars, chain, k = setup_day(lambda i, p0: p0 + 0.05 * min(i, 60), lambda m: 2.0 if m < 10 else 3.2)
    res, sent = run(d, bars, chain)
    assert res["status"] == "traded" and res["direction"] == "long" and res["kind"] == "target"
    entry = sent[1]
    assert "BUY QQQ" in entry["title"] and "CALL" in entry["title"]
    f = {x["name"]: x["value"] for x in entry["fields"]}
    assert "$2.01" in f["Buy price"] and "-35% -> **$1.31**" in f["Stop loss"] and "+50% -> **$3.02**" in f["Take profit"]
    assert f["Size"].startswith("4 contract")                    # $1,000 / $201
    assert "TAKE PROFIT HIT" in sent[2]["title"] and res["pnl"] > 0


def test_red_candle_buys_put_and_stop_briefing():
    d, bars, chain, k = setup_day(lambda i, p0: p0 - 0.05 * min(i, 5) + 0.08 * max(0, i - 5),
                                  lambda m: 2.0 if m < 10 else 1.1)
    res, sent = run(d, bars, chain)
    assert res["direction"] == "short" and res["kind"] == "stop" and res["pnl"] < 0
    assert "PUT" in sent[1]["title"] and "STOP LOSS HIT" in sent[2]["title"]
    assert "against us" in sent[2]["description"]


def test_time_exit_after_90_minutes():
    d, bars, chain, k = setup_day(lambda i, p0: p0 + 0.05 * min(i, 5), lambda m: 2.0)
    res, sent = run(d, bars, chain)
    assert res["kind"] == "time" and "TIME EXIT" in sent[2]["title"]


def test_doji_and_news_days_stand_down():
    d, bars, chain, k = setup_day(lambda i, p0: p0 + (0.03 if i % 2 else 0.0) * 0, lambda m: 2.0)
    res, sent = run(d, bars, chain)
    assert res["status"] == "doji" and "no trade" in sent[-1]["title"]
    d, bars, chain, k = setup_day(lambda i, p0: p0 + 0.05 * i, lambda m: 2.0)
    res, sent = run(d, bars, chain, news_events=["08:30 CPI m/m"])
    assert res["status"] == "news" and "CPI" in sent[0]["fields"][0]["value"]
