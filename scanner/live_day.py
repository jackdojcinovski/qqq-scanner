"""Live day: scan QQQ 9:30-11:00 for the common day-trading setups, alert ONE trade, manage it, log it.

  * every closed 1-minute candle from 9:35 to 11:00 is checked (scanner/day_setups.py)
  * first setup with price on the right side of VWAP and >= 4/5 confirmations -> alert
  * at-the-money 0DTE call (long) / put (short), ~$1,000
  * normal setup: +50% take profit / -35% stop. Strong setup (5/5 and volume >= 1.5x normal):
    +100% / -25% (bigger target, tighter stop)
  * out within 90 minutes no matter what; no trade on red-folder news days
  * every day is written to journal/trades.csv and a recap with running totals is posted
"""
from __future__ import annotations

import csv
import math
from datetime import date, time
from pathlib import Path

import numpy as np
import pandas as pd

from . import day_setups as ds
from .orb_alert import AMBER, BLUE, GREEN, GREY, RED, at, embed, occ

LAST_ENTRY = time(11, 0)
MAX_HOLD = 90
BUDGET = 1000.0
JOURNAL_FIELDS = ["date", "status", "setup", "direction", "contract", "contracts", "entry_time", "entry",
                  "exit_time", "exit", "how", "pnl", "pnl_pct", "score", "note"]


def profile(hist: pd.DataFrame, day: date) -> dict:
    """Yesterday's levels and a normal morning's cumulative volume by minute (last 14 sessions)."""
    if hist.empty:
        return {}
    t = hist.index.time
    rth = hist[(t >= time(9, 30)) & (t < time(16, 0)) & (hist.index.date < day)]
    if rth.empty:
        return {}
    days = sorted(set(rth.index.date))
    y = rth[rth.index.date == days[-1]]
    cums = []
    for d in days[-14:]:
        x = rth[rth.index.date == d]
        mins = ((x.index.hour * 60 + x.index.minute) - 570).to_numpy()
        v = np.zeros(390)
        v[mins[(mins >= 0) & (mins < 390)]] = x["volume"].to_numpy()[(mins >= 0) & (mins < 390)]
        cums.append(np.cumsum(v))
    return {"prev_high": float(y["high"].max()), "prev_low": float(y["low"].min()),
            "prev_close": float(y["close"].iloc[-1]), "avg_cum": np.mean(cums, axis=0) if cums else None}


def plan_for(sig: ds.Signal) -> tuple[float, float, str]:
    strong = sig.score == 5 and sig.facts.get("rvol", 0) == sig.facts.get("rvol", 0) and sig.facts.get("rvol", 0) >= 1.5
    return (1.0, 0.25, "strong setup: bigger target, tighter stop") if strong else (0.5, 0.35, "standard plan")


def journal_write(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=JOURNAL_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in JOURNAL_FIELDS})


def journal_totals(path: Path, day: date) -> dict:
    if not path.exists():
        return {"trades": 0, "wins": 0, "total": 0.0, "week": 0.0, "days": 0}
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    traded = [r for r in rows if r["status"] == "traded"]
    wk = pd.Timestamp(day).isocalendar()[:2]
    week = [r for r in traded if pd.Timestamp(r["date"]).isocalendar()[:2] == wk]
    return {"trades": len(traded), "wins": sum(1 for r in traded if float(r["pnl"]) > 0),
            "total": sum(float(r["pnl"]) for r in traded), "week": sum(float(r["pnl"]) for r in week),
            "days": len(rows)}


def recap(path: Path, day: date, today_line: str) -> list[dict]:
    t = journal_totals(path, day)
    win = f"{t['wins'] / t['trades']:.0%}" if t["trades"] else "-"
    return embed("Daily recap", BLUE,
                 [("Today", today_line),
                  ("This week", f"${t['week']:+,.0f}"),
                  ("All time", f"${t['total']:+,.0f} over {t['trades']} trade(s), win rate {win}, {t['days']} day(s) logged")])


def run_day(api, notify, symbol: str = "QQQ", news_events=None, news_ok: bool = True,
            journal: Path = Path("journal/trades.csv"), budget: float = BUDGET, poll_s: float = 5.0, log=print) -> dict:
    day = api.now().date()
    if not api.is_open(day):
        log(f"Market closed on {day}.")
        return {"status": "closed"}
    if news_events:
        notify("", embed(f"{symbol} - no trade today (red-folder news)", GREY, [("News", ", ".join(news_events))],
                         "Your rule: no trades on red-folder days."))
        journal_write(journal, {"date": day, "status": "news", "note": "; ".join(news_events)})
        notify("", recap(journal, day, "No trade (news day)"))
        return {"status": "news"}
    ctx = profile(api.history(symbol, at(day, 4, 0) - pd.Timedelta(days=30), at(day, 0, 0)), day)
    notify("", embed(f"{symbol} scanner running 9:30-11:00", BLUE,
                     [("Watching for", "Opening-range breakout, VWAP reclaim, VWAP pullback bounce, flag breakout, "
                                       "yesterday's high/low break, 9/20 EMA cross"),
                      ("Rule", "Price on the right side of VWAP + at least 4 of 5 confirmations "
                               "(VWAP, 9/20 EMA curl, volume vs normal, volume spike, vs today's open)"),
                      ("Trade", f"ATM 0DTE call/put, ~${budget:,.0f}, +50%/-35% (strong: +100%/-25%), out within "
                                f"{MAX_HOLD} min, one trade a day")],
                     "" if news_ok else "Couldn't load the economic calendar - check Forex Factory yourself."))

    while api.now() < at(day, 9, 30):
        api.sleep(min(30.0, max(1.0, (at(day, 9, 30) - api.now()).total_seconds())))
    bars: dict = {}
    seen = -1
    sig = None
    while api.now() < at(day, LAST_ENTRY.hour, LAST_ENTRY.minute) + pd.Timedelta(minutes=1):
        got = api.latest_bar(symbol)
        if got and got[0] >= at(day, 9, 30):
            bars[got[0]] = got[1]
        keys = sorted(bars)
        if keys:
            m = int((keys[-1] - at(day, 9, 30)).total_seconds() // 60)
            if m != seen and m >= 4:
                seen = m
                df = pd.DataFrame([bars[k] for k in keys], index=keys)
                full = df.reindex(pd.date_range(at(day, 9, 30), keys[-1], freq="1min")).ffill()
                full["volume"] = full["volume"].fillna(0)
                found = ds.evaluate(*(full[c].to_numpy(dtype=float) for c in ("open", "high", "low", "close", "volume")), ctx)
                if found:
                    sig = found[0]
                    break
        api.sleep(2.0)
    if sig is None:
        notify("", embed(f"{symbol} - no setup by 11:00, no trade today", GREY, [],
                         "Nothing lined up with enough confirmations. Sitting out is a position."))
        journal_write(journal, {"date": day, "status": "no setup"})
        notify("", recap(journal, day, "No trade (no setup)"))
        return {"status": "no setup"}

    right = "C" if sig.direction == "long" else "P"
    strike = float(round(sig.price))
    sym_opt = occ(symbol, day, right, strike)
    q = None
    for _ in range(6):
        q = api.option_quote(sym_opt)
        if q and q.ask > 0:
            break
        api.sleep(3.0)
    if not q or q.ask <= 0:
        notify("", embed(f"{symbol} - setup found but no option price", AMBER, [("Contract", sym_opt)]))
        journal_write(journal, {"date": day, "status": "no quote", "setup": sig.setup})
        return {"status": "no quote"}
    tp, sl, why_plan = plan_for(sig)
    n = max(1, int(budget // (q.ask * 100)))
    buy = q.ask
    stop = round(buy * (1 - sl), 2)
    target = math.ceil(round(buy * (1 + tp) * 100, 6)) / 100
    word = "CALL" if right == "C" else "PUT"
    out_by = q.ts + pd.Timedelta(minutes=MAX_HOLD)
    notify("", embed(
        f"ENTER: BUY {symbol} ${strike:g} {word} (0DTE) - {ds.NAMES[sig.setup]}", GREEN if right == "C" else RED,
        [("Buy price", f"**${buy:.2f}** (ask) at {q.ts:%H:%M:%S} ET - mid ${q.mid:.2f}"),
         ("Contracts", f"{n} = ${n * buy * 100:,.0f}"),
         ("Stop loss", f"-{sl:.0%} -> **${stop:.2f}** (lose about ${n * (buy - stop) * 100:,.0f})"),
         ("Take profit", f"+{tp:.0%} -> **${target:.2f}** (make about ${n * (target - buy) * 100:,.0f}) - {why_plan}"),
         ("Time limit", f"out by {out_by:%H:%M} ET"),
         (f"{symbol} at entry", f"{sig.price:.2f}")],
        "**Why this trade**\n" + ds.explain(sig, ctx),
        "Alert prices are Alpaca indicative quotes - set your stop/target from your actual Wealthsimple fill."))

    best = worst = q.bid
    last, kind = q, None
    while True:
        api.sleep(poll_s)
        qq = api.option_quote(sym_opt)
        if qq and qq.bid > 0:
            last = qq
            best, worst = max(best, qq.bid), min(worst, qq.bid)
            if qq.bid <= stop:
                kind = "stop"
            elif qq.bid >= target:
                kind = "target"
        if kind is None and api.now() >= out_by:
            kind = "time"
        if kind:
            break
    bar = api.latest_bar(symbol)
    s_out = float(bar[1]["close"]) if bar else sig.price
    sell = target if kind == "target" else last.bid
    pnl = n * (sell - buy) * 100
    pct = sell / buy - 1
    held = int((last.ts - q.ts).total_seconds() // 60)
    move = s_out / sig.price - 1
    ours = move if right == "C" else -move
    story = {"target": "The move followed through and the option reached the target.",
             "stop": ("Price turned against the setup and we cut it at the stop." if ours < 0 else
                      "Price stalled and time decay pulled the option down to the stop."),
             "time": "Neither target nor stop hit inside the time limit - out before decay gets worse."}[kind]
    title = {"target": "TAKE PROFIT HIT", "stop": "STOP LOSS HIT", "time": "TIME EXIT"}[kind]
    notify("", embed(f"{title}: {'+' if pnl >= 0 else '-'}${abs(pnl):,.0f} ({pct:+.0%}) - {symbol} ${strike:g} {word}",
                     GREEN if pnl > 0 else (RED if pnl < 0 else GREY),
                     [("Sold", f"${sell:.2f} at {last.ts:%H:%M:%S} ET (bought ${buy:.2f} at {q.ts:%H:%M:%S})"),
                      ("Result", f"**{'+' if pnl >= 0 else '-'}${abs(pnl):,.0f}** on {n} contract(s), {pct:+.1%}, held {held} min")],
                     f"**What happened**\n{symbol} {move:+.2%} ({sig.price:.2f} -> {s_out:.2f}), "
                     f"{'our way' if ours > 0 else 'against us'}. {story}\n"
                     f"Best the option got: {best / buy - 1:+.0%}. Worst: {worst / buy - 1:+.0%}."))
    journal_write(journal, {"date": day, "status": "traded", "setup": ds.NAMES[sig.setup], "direction": sig.direction,
                            "contract": f"{symbol} {strike:g}{right}", "contracts": n, "entry_time": f"{q.ts:%H:%M:%S}",
                            "entry": f"{buy:.2f}", "exit_time": f"{last.ts:%H:%M:%S}", "exit": f"{sell:.2f}", "how": kind,
                            "pnl": f"{pnl:.2f}", "pnl_pct": f"{pct:.4f}", "score": sig.score})
    notify("", recap(journal, day, f"{ds.NAMES[sig.setup]} {word}: {'+' if pnl >= 0 else '-'}${abs(pnl):,.0f} ({pct:+.0%})"))
    return {"status": "traded", "setup": sig.setup, "direction": sig.direction, "kind": kind, "pnl": pnl, "n": n}
