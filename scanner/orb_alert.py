"""Live TEST alerts: the 5-minute opening-range breakout on QQQ, traded with a plain 0DTE option.

The plan (Jack's rules):
  * 9:35: the first 5-minute candle (9:30-9:34) sets the direction. Green candle -> buy the at-the-money
    0DTE CALL. Red candle -> buy the at-the-money 0DTE PUT. Tiny candle (doji) -> no trade today.
  * about $1,000 per trade (as many contracts as that buys)
  * take profit +50% of the premium, stop loss -35%, and out by 11:05 (90 minutes) no matter what
  * no trade on red-folder (high-impact USD) news days
  * one trade a day; it watches the option every few seconds and briefs you when it exits

`run_day(api, notifier, ...)` does the whole day. `api` is either the live Alpaca connection
(LiveAPI) or a replay of a past day from downloaded data (ReplayAPI), so the exact same code can be
checked against history before it runs live.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, time
from typing import Protocol

import numpy as np
import pandas as pd

from .history import ET

GREEN, RED, GREY, BLUE, AMBER = 0x2ECC71, 0xE74C3C, 0x95A5A6, 0x3498DB, 0xF39C12


@dataclass
class Plan:
    budget: float = 1000.0
    take_profit: float = 0.50
    stop_loss: float = 0.35
    max_hold_min: int = 90
    doji_pct: float = 0.0002           # first candle body under 0.02% of price = no direction
    poll_s: float = 5.0


@dataclass
class Quote:
    bid: float
    ask: float
    ts: pd.Timestamp

    @property
    def mid(self) -> float:
        return round((self.bid + self.ask) / 2, 2)


class API(Protocol):
    def now(self) -> pd.Timestamp: ...
    def sleep(self, seconds: float) -> None: ...
    def is_open(self, day: date) -> bool: ...
    def latest_bar(self, symbol: str) -> tuple[pd.Timestamp, dict] | None: ...
    def option_quote(self, occ: str) -> Quote | None: ...
    def history(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame: ...


def occ(symbol: str, day: date, right: str, strike: float) -> str:
    return f"{symbol}{day:%y%m%d}{right}{int(round(strike * 1000)):08d}"


def at(day: date, hh: int, mm: int) -> pd.Timestamp:
    return pd.Timestamp(day.year, day.month, day.day, hh, mm, tz=ET)


# ------------------------------------------------------------------ context (yesterday and before)

def context(hist: pd.DataFrame, day: date) -> dict:
    """Facts known before today's open, from minute history (any extended hours are ignored)."""
    if hist.empty:
        return {}
    t = hist.index.time
    rth = hist[(t >= time(9, 30)) & (t < time(16, 0)) & (hist.index.date < day)]
    if rth.empty:
        return {}
    g = rth.groupby(rth.index.date)
    daily = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                          "close": g["close"].last()})
    f5 = rth[rth.index.time < time(9, 35)]
    first5 = f5.groupby(f5.index.date)["volume"].sum()
    rets = np.log(daily["close"]).diff().dropna()
    out = {"prev_close": float(daily["close"].iloc[-1]), "prev_high": float(daily["high"].iloc[-1]),
           "prev_low": float(daily["low"].iloc[-1]),
           "vol20": float(rets.tail(20).std()) if len(rets) >= 5 else math.nan,
           "avg_first5_vol": float(first5.tail(14).mean()) if len(first5) else math.nan,
           "prev_ret": float(rets.iloc[-1]) if len(rets) else math.nan}
    if len(daily) >= 2:
        y, b = daily.iloc[-1], daily.iloc[-2]
        out["inside"] = bool(y["high"] <= b["high"] and y["low"] >= b["low"])
    return out


def analysis(direction: str, bars: pd.DataFrame, ctx: dict) -> tuple[str, list[str]]:
    o, c = float(bars["open"].iloc[0]), float(bars["close"].iloc[-1])
    h, low = float(bars["high"].max()), float(bars["low"].min())
    vol = float(bars["volume"].sum())
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3
    vwap = float((typical * bars["volume"]).sum() / max(bars["volume"].sum(), 1))
    word = "green (buyers in control)" if direction == "long" else "red (sellers in control)"
    lines = [f"First 5-minute candle was **{word}**: open {o:.2f} -> close {c:.2f} ({(c / o - 1):+.2%}), "
             f"high {h:.2f}, low {low:.2f}.",
             "Opening-range breakout: the direction of the first 5 minutes often carries into the morning, "
             f"so we {'buy a call' if direction == 'long' else 'buy a put'} in that direction."]
    signs = []
    pc, sig = ctx.get("prev_close"), ctx.get("vol20")
    if pc:
        gap = o / pc - 1
        lines.append(f"Gap from yesterday's close ({pc:.2f}): {gap:+.2%}.")
        if sig and sig == sig and abs(gap) >= 0.5 * sig:
            signs.append("big gap")
    avg = ctx.get("avg_first5_vol")
    if avg and avg == avg and avg > 0:
        rv = vol / avg
        lines.append(f"Opening volume: {rv:.1f}x a normal day's first 5 minutes.")
        if rv >= 1.3:
            signs.append("heavy volume")
    if ctx.get("inside"):
        signs.append("yesterday was an inside day")
    pr = ctx.get("prev_ret")
    if pr is not None and sig and pr == pr and sig == sig and abs(pr) >= 1.5 * sig:
        signs.append("yesterday was a big day")
    side = "above" if c >= vwap else "below"
    lines.append(f"Price is {side} the opening VWAP ({vwap:.2f}).")
    if ctx.get("prev_high") and c > ctx["prev_high"]:
        lines.append(f"Already above yesterday's high ({ctx['prev_high']:.2f}).")
    if ctx.get("prev_low") and c < ctx["prev_low"]:
        lines.append(f"Already below yesterday's low ({ctx['prev_low']:.2f}).")
    lines.append("Big-move signs: " + (", ".join(signs) if signs else "none today (expect a smaller move)") + ".")
    return "\n".join(lines), signs


# ------------------------------------------------------------------ messages

def embed(title: str, color: int, fields: list[tuple[str, str]], desc: str = "", footer: str = "") -> list[dict]:
    e = {"title": title, "color": color, "fields": [{"name": n, "value": v, "inline": False} for n, v in fields]}
    if desc:
        e["description"] = desc[:4000]
    if footer:
        e["footer"] = {"text": footer}
    return [e]


def entry_message(sym: str, day: date, direction: str, strike: float, q: Quote, n: int, plan: Plan,
                  why: str) -> list[dict]:
    right = "CALL" if direction == "long" else "PUT"
    buy = q.ask
    stop = round(buy * (1 - plan.stop_loss), 2)
    target = math.ceil(round(buy * (1 + plan.take_profit) * 100, 6)) / 100   # at least the full +50%
    out_by = (q.ts + pd.Timedelta(minutes=plan.max_hold_min)).strftime("%H:%M")
    return embed(
        f"TEST TRADE - BUY {sym} {strike:g} {right} (0DTE)", GREEN if direction == "long" else RED,
        [("Contract", f"{sym} {day:%b %d} ${strike:g} {right} - expires today"),
         ("Buy price", f"**${buy:.2f}** (ask) at {q.ts:%H:%M:%S} ET  -  mid ${q.mid:.2f}, bid ${q.bid:.2f}"),
         ("Size", f"{n} contract(s) = ${n * buy * 100:,.0f}"),
         ("Stop loss", f"-{plan.stop_loss:.0%} -> **${stop:.2f}**  (lose about ${n * (buy - stop) * 100:,.0f})"),
         ("Take profit", f"+{plan.take_profit:.0%} -> **${target:.2f}**  (make about ${n * (target - buy) * 100:,.0f})"),
         ("Time limit", f"Out by {out_by} ET if neither is hit")],
        "**Why this trade**\n" + why,
        "TEST alert - this setup has NOT shown a reliable edge in backtests. Prices are Alpaca indicative quotes; "
        "use your Wealthsimple fill.")


def exit_message(sym: str, strike: float, direction: str, kind: str, buy: float, sell: float, n: int,
                 t_in: pd.Timestamp, t_out: pd.Timestamp, s_in: float, s_out: float, best: float,
                 worst: float) -> list[dict]:
    right = "CALL" if direction == "long" else "PUT"
    pnl = n * (sell - buy) * 100
    pct = sell / buy - 1
    held = int((t_out - t_in).total_seconds() // 60)
    title = {"target": f"TAKE PROFIT HIT: +${pnl:,.0f} ({pct:+.0%})",
             "stop": f"STOP LOSS HIT: -${-pnl:,.0f} ({pct:+.0%})",
             "time": f"TIME EXIT: {'+' if pnl >= 0 else '-'}${abs(pnl):,.0f} ({pct:+.0%})"}[kind]
    color = GREEN if pnl > 0 else (RED if pnl < 0 else GREY)
    move = s_out / s_in - 1
    our = move if direction == "long" else -move
    what = (f"{sym} moved {move:+.2%} ({s_in:.2f} -> {s_out:.2f}) in {held} minutes, "
            f"{'our way' if our > 0 else 'against us'}. ")
    if kind == "target":
        what += "The breakout carried far enough for the option to reach the +50% target."
    elif kind == "stop":
        what += ("The opening direction didn't hold - price reversed and the option lost 35%, so we cut it."
                 if our < 0 else "Price went nowhere and time decay ate the premium down to the stop.")
    else:
        what += "Neither target nor stop was hit within the time limit, so we got out before decay gets worse."
    what += f"\nBest the option got: {best / buy - 1:+.0%}. Worst: {worst / buy - 1:+.0%}."
    return embed(f"{sym} {strike:g} {right} - {title}", color,
                 [("Sold", f"${sell:.2f} (bid) at {t_out:%H:%M:%S} ET  -  bought ${buy:.2f} at {t_in:%H:%M:%S}"),
                  ("Result", f"**{'+' if pnl >= 0 else '-'}${abs(pnl):,.0f}** on {n} contract(s) ({pct:+.1%})"),
                  ("Held", f"{held} minutes")],
                 "**What happened**\n" + what, "TEST alert - not a recommendation to trade.")


# ------------------------------------------------------------------ the day

def run_day(api: API, notify, symbol: str = "QQQ", plan: Plan = Plan(), news_events=None,
            news_ok: bool = True, log=print) -> dict:
    """Runs one trading day. Returns a small summary dict (also used by tests)."""
    day = api.now().date()
    if not api.is_open(day):
        log(f"Market closed on {day}.")
        return {"status": "closed"}
    if news_events:
        names = ", ".join(news_events)
        notify("", embed(f"{symbol} TEST alerts - no trade today", GREY,
                         [("Red-folder news", names)], "Your rule: no trades on red-folder days. Back tomorrow."))
        return {"status": "news", "events": news_events}
    note = "" if news_ok else "\nCouldn't load the economic calendar - check Forex Factory yourself before trading."
    notify("", embed(f"{symbol} TEST alerts running", BLUE,
                     [("Plan", f"9:35 opening-range direction -> ATM 0DTE call/put, ~${plan.budget:,.0f}, "
                               f"+{plan.take_profit:.0%} target / -{plan.stop_loss:.0%} stop, out within "
                               f"{plan.max_hold_min} min")], "Watching the first 5 minutes." + note))

    hist = api.history(symbol, at(day, 4, 0) - pd.Timedelta(days=35), at(day, 0, 0))
    ctx = context(hist, day)

    # wait for the open, then collect the 9:30-9:34 one-minute bars
    while api.now() < at(day, 9, 30):
        api.sleep(min(30.0, max(1.0, (at(day, 9, 30) - api.now()).total_seconds())))
    bars: dict = {}
    while True:
        got = api.latest_bar(symbol)
        if got:
            ts, b = got
            if at(day, 9, 30) <= ts <= at(day, 9, 34):
                bars[ts] = b
        if at(day, 9, 34) in bars or api.now() >= at(day, 9, 37):
            break
        api.sleep(2.0)
    if len(bars) < 3:
        notify("", embed(f"{symbol} - no trade today", AMBER, [("Why", "Couldn't get the first 5 minutes of data.")]))
        return {"status": "no data"}
    df = pd.DataFrame([bars[k] for k in sorted(bars)], index=sorted(bars))
    o, c = float(df["open"].iloc[0]), float(df["close"].iloc[-1])
    if abs(c - o) < plan.doji_pct * o:
        notify("", embed(f"{symbol} - no trade today", GREY,
                         [("Why", f"First 5-minute candle had no direction (open {o:.2f}, close {c:.2f}).")]))
        return {"status": "doji"}
    direction = "long" if c > o else "short"
    right = "C" if direction == "long" else "P"
    strike = float(round(c))
    sym_opt = occ(symbol, day, right, strike)
    q = None
    for _ in range(6):
        q = api.option_quote(sym_opt)
        if q and q.ask > 0:
            break
        api.sleep(3.0)
    if not q or q.ask <= 0:
        notify("", embed(f"{symbol} - no trade today", AMBER, [("Why", f"No price for {sym_opt}.")]))
        return {"status": "no quote"}
    n = max(1, int(plan.budget // (q.ask * 100)))
    why, _signs = analysis(direction, df, ctx)
    notify("", entry_message(symbol, day, direction, strike, q, n, plan, why))

    buy, t_in, s_in = q.ask, q.ts, c
    stop = buy * (1 - plan.stop_loss)
    target = buy * (1 + plan.take_profit)
    deadline = t_in + pd.Timedelta(minutes=plan.max_hold_min)
    best = worst = q.bid
    last_q, kind = q, None
    while True:
        api.sleep(plan.poll_s)
        qq = api.option_quote(sym_opt)
        if qq and qq.bid > 0:
            last_q = qq
            best, worst = max(best, qq.bid), min(worst, qq.bid)
            if qq.bid <= stop:
                kind = "stop"
            elif qq.bid >= target:
                kind = "target"
        if kind is None and api.now() >= deadline:
            kind = "time"
        if kind:
            break
    bar = api.latest_bar(symbol)
    s_out = float(bar[1]["close"]) if bar else s_in
    sell = last_q.bid
    notify("", exit_message(symbol, strike, direction, kind, buy, sell, n, t_in, last_q.ts, s_in, s_out, best, worst))
    return {"status": "traded", "direction": direction, "strike": strike, "buy": buy, "sell": sell, "n": n,
            "kind": kind, "pnl": n * (sell - buy) * 100}


# ------------------------------------------------------------------ a past day, replayed from downloaded data

class ReplayAPI:
    """Pretends it's `day`: stock bars from data/history, option prices from data/options (last trades)."""

    def __init__(self, day: date, stock: pd.DataFrame, chain: pd.DataFrame, start: pd.Timestamp | None = None,
                 half_spread: float = 0.01):
        self.day = day
        self.stock = stock.sort_index()
        self.t = start or at(day, 9, 25)
        self.hs = half_spread
        self.opt = {}
        if not chain.empty:
            ts = pd.to_datetime(chain["timestamp"], utc=True).dt.tz_convert(ET)
            for sym, idx in chain.groupby("symbol").indices.items():
                s = pd.Series(chain["close"].to_numpy()[idx], index=ts.iloc[idx].to_numpy()).sort_index()
                self.opt[sym] = s

    def now(self) -> pd.Timestamp:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t = self.t + pd.Timedelta(seconds=seconds)

    def is_open(self, day: date) -> bool:
        return not self.stock[self.stock.index.date == day].empty

    def latest_bar(self, symbol: str):
        done = self.stock[(self.stock.index + pd.Timedelta(minutes=1)) <= self.t]
        if done.empty:
            return None
        ts = done.index[-1]
        return ts, done.iloc[-1].to_dict()

    def option_quote(self, occ_sym: str) -> Quote | None:
        s = self.opt.get(occ_sym)
        if s is None:
            return None
        s = s[(s.index + pd.Timedelta(minutes=1)) <= self.t]
        if s.empty:
            return None
        px = float(s.iloc[-1])
        return Quote(max(px - self.hs, 0.01), px + self.hs, self.t)

    def history(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        s = self.stock
        return s[(s.index >= start) & (s.index < end)]


class FetchingReplayAPI(ReplayAPI):
    """ReplayAPI that downloads each option's minute bars the first time it's asked for (for replaying today)."""

    def __init__(self, day: date, stock: pd.DataFrame, fetch_option, half_spread: float = 0.01):
        super().__init__(day, stock, pd.DataFrame(columns=["symbol", "timestamp", "close"]), half_spread=half_spread)
        self._fetch = fetch_option            # occ -> pd.Series of closes indexed by ET timestamps (or None)
        self._tried: set = set()

    def option_quote(self, occ_sym: str) -> Quote | None:
        if occ_sym not in self.opt and occ_sym not in self._tried:
            self._tried.add(occ_sym)
            s = self._fetch(occ_sym)
            if s is not None and len(s):
                self.opt[occ_sym] = s.sort_index()
        return super().option_quote(occ_sym)
