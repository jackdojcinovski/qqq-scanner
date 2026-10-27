"""Live scanner core: takes 1-minute bars, sends alerts, tracks trades.

It applies exactly the backtest's rules (scorer.candidates + select, the same
trade simulator, 2 alerts a day, stop after a loss, one trade at a time), plus
the news filter, which the backtest can't apply. Market data and Discord are
passed in, so the whole thing can be replayed in tests.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Any, Callable

import pandas as pd

from . import backtest as bt
from . import discord as dz
from .config import parse_time
from .news import NewsFilter, NewsStatus
from .scorer import Setup, candidates, gate_reason, select
from .store import Store

log = logging.getLogger(__name__)
OHLCV = ["open", "high", "low", "close", "volume"]
REST_DELAY = pd.Timedelta(minutes=16)   # free plan: REST can't return the latest 15 minutes


@dataclass
class Tracked:
    trade_no: str
    setup: Setup
    taken: bool
    posted_events: set[str] = field(default_factory=set)
    warned_news: bool = False
    result: bt.TradeResult | None = None
    done: bool = False
    outcome: str = ""
    r: float | None = None


class LiveScanner:
    def __init__(self, cfg: dict[str, Any], symbol: str, notifier, history: pd.DataFrame,
                 news: NewsFilter | None = None,
                 backfill: Callable[[pd.Timestamp, pd.Timestamp], pd.DataFrame | None] | None = None,
                 option_quote: Callable[[dict], float | None] | None = None,
                 store: Store | None = None):
        self.cfg = cfg
        self.symbol = symbol
        self.notify = notifier
        self.bars = history[OHLCV].copy().sort_index()
        self.news = news
        self.backfill = backfill
        self.option_quote = option_quote
        self.store = store
        self.gaps: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        self.trades: dict[str, Tracked] = {}      # by setup key
        self.alerts = 0
        self.losses = 0
        self.wins = 0
        self.net_r = 0.0
        self.count = 0
        self.last_bar: pd.Timestamp | None = None
        self.last_post: datetime | None = None
        self.stale_warned = False
        self.summary_sent = False
        w = cfg["window"]
        self.start_t, self.end_t = parse_time(w["start"]), parse_time(w["end"])

    # ------------------------------------------------------------------ messages
    def post(self, msg: tuple[str, list[dict]]) -> None:
        content, embeds = msg
        self.notify.send(content, embeds)
        self.last_post = datetime.now()

    def today_line(self) -> str:
        return (f"Today: {self.alerts} alert(s), {self.wins} win(s), {self.losses} loss(es), "
                f"net {self.net_r:+.2f}R")

    # ------------------------------------------------------------------ data intake
    def add_gap(self, start: pd.Timestamp, end: pd.Timestamp) -> None:
        if end > start:
            self.gaps.append((start, end))

    def on_bar(self, ts: pd.Timestamp, o: float, h: float, lo: float, c: float, v: float,
               now: pd.Timestamp | None = None) -> None:
        # QQQ trades on IEX every minute of the session, so a jump means bars were missed
        if self.last_bar is not None and ts - self.last_bar > pd.Timedelta(minutes=2) \
                and time(9, 30) <= ts.time() < time(16, 0):
            self.add_gap(self.last_bar + pd.Timedelta(minutes=1), ts)
        self.bars.loc[ts, OHLCV] = [o, h, lo, c, v]
        if len(self.bars) > 1 and self.bars.index[-2] > ts:
            self.bars.sort_index(inplace=True)
        self.last_bar = ts
        self.stale_warned = False
        self._fill_gaps(now or ts + pd.Timedelta(minutes=1))
        self.process(ts)

    def _fill_gaps(self, now: pd.Timestamp) -> None:
        if not self.backfill:
            return
        still = []
        for a, b in self.gaps:
            if now - REST_DELAY >= b:
                try:
                    df = self.backfill(a, b)
                except Exception as exc:  # keep running; try again next minute
                    log.warning("backfill failed: %s", exc)
                    still.append((a, b))
                    continue
                if df is not None and not df.empty:
                    new = df[OHLCV][~df.index.isin(self.bars.index)]
                    self.bars = pd.concat([self.bars, new]).sort_index()
            else:
                still.append((a, b))
        self.gaps = still

    def session_gap_open(self, ts: pd.Timestamp) -> bool:
        """True while data from today's session is still missing."""
        day_open = ts.normalize() + pd.Timedelta(hours=9, minutes=30)
        return any(b > day_open - pd.Timedelta(minutes=30) and a < ts for a, b in self.gaps)

    # ------------------------------------------------------------------ core
    def news_status(self, ts: pd.Timestamp) -> NewsStatus:
        return self.news.status(ts.to_pydatetime()) if self.news else NewsStatus(False)

    def process(self, ts: pd.Timestamp) -> None:
        today = ts.date()
        day_bars = self.bars[(self.bars.index.date == today)]
        t = day_bars.index.time
        session = day_bars[(t >= time(9, 30)) & (t < time(16, 0))]
        news = self.news_status(ts)

        # 1. follow open trades
        for tr in self.trades.values():
            if not tr.taken or tr.done:
                continue
            res = bt.simulate(tr.setup, session[session.index > tr.setup.time], self.cfg, live=True)
            tr.result = res
            for ev_ts, kind, px in res.events:
                if kind not in tr.posted_events:
                    tr.posted_events.add(kind)
                    note = {"fill": f"stop {tr.setup.stop:.2f}, target {tr.setup.target:.2f}",
                            "partial": "rest rides with the stop at breakeven"}.get(kind, "")
                    self.post(dz.event_message(tr.trade_no, kind, ev_ts, px, note))
            if res.final:
                self._resolve(tr, res.outcome, res.r if res.filled else None, res.exit_time, res.exit_price)
            elif news.warning and not tr.warned_news:
                tr.warned_news = True
                self.post(dz.simple(f"Trade #{tr.trade_no}: news soon",
                                    f"{news.warning}. Consider closing or tightening the stop.", dz.AMBER))

        # 2. look for a new setup
        if not (self.start_t <= ts.time() <= self.end_t):
            return
        # limits are applied below (like the backtest) so skipped setups are remembered
        if gate_reason(ts, self.cfg, news, None) or self.session_gap_open(ts):
            return
        cands, _ = candidates(self.bars[self.bars.index <= ts], self.cfg,
                              (self.cfg["trade"].get("entry_at", "mid"),))
        s, _ = select(cands, self.cfg)
        if s is None:
            return
        band = s.band(self.cfg)
        if band == "ignore":
            return
        prev = self.trades.get(s.key)
        if prev is not None and not (prev.setup.band(self.cfg) == "log" and band == "alert"):
            return

        lim = self.cfg["limits"]
        busy = any(tr.taken and not tr.done for tr in self.trades.values())
        skip = ""
        if band != "alert":
            skip = "score below alert"
        elif self.alerts >= lim["max_alerts_per_day"]:
            skip = "daily alert limit"
        elif self.losses >= lim["stop_after_losses"]:
            skip = "stopped after a loss"
        elif busy:
            skip = "trade already open"
        taken = skip == ""
        self.count += 1
        trade_no = f"{ts:%Y-%m-%d}-{self.count:02d}"
        self.trades[s.key] = Tracked(trade_no, s, taken)
        if self.store:
            self.store.alert(trade_no, s, self.symbol, band, taken, skip)
        if taken:
            self.alerts += 1
            plan = dz.contract_plan(s, self.cfg, self.symbol)
            if self.option_quote:
                try:
                    plan["option_price"] = self.option_quote(plan)
                except Exception as exc:
                    log.warning("option quote failed: %s", exc)
            ctx = [news.warning and f"News soon: {news.warning}" or "No red-folder news in the next 15 min",
                   f"{ts:%H:%M} ET; alert {self.alerts} of {lim['max_alerts_per_day']} today"]
            self.post(dz.alert_message(s, self.cfg, self.symbol, trade_no, plan, ctx))

    def _resolve(self, tr: Tracked, outcome: str, r: float | None, when, price: float | None) -> None:
        tr.done = True
        tr.outcome, tr.r = outcome, r
        if r is not None:
            self.net_r += r
            if r < 0:
                self.losses += 1
            elif r > 0:
                self.wins += 1
        if self.store:
            self.store.outcome(tr.trade_no, when, outcome, r, price)
        self.post(dz.update_message(self.symbol, tr.trade_no, tr.setup, outcome, r,
                                    when or self.last_bar, price, self.today_line()))

    # ------------------------------------------------------------------ housekeeping
    def tick(self, now: pd.Timestamp) -> None:
        """Called every ~30 s by a timer: stale-data warning, heartbeat, summary."""
        mkt_open = now.normalize() + pd.Timedelta(hours=9, minutes=30)
        close = now.normalize() + pd.Timedelta(hours=16)
        if mkt_open + pd.Timedelta(minutes=2) <= now <= close and self.last_bar is not None \
                and now - self.last_bar > pd.Timedelta(minutes=4) and not self.stale_warned:
            self.stale_warned = True
            self.post(dz.simple("No market data", f"No bar since {self.last_bar:%H:%M}. "
                                "Check the laptop's connection; alerts are paused until data returns.", dz.AMBER))
        hb = self.cfg["live"]["heartbeat_minutes"]
        hb_end = now.normalize() + pd.Timedelta(hours=self.end_t.hour, minutes=self.end_t.minute) \
            + pd.Timedelta(minutes=bt.exit_settings(self.cfg)["time_exit_minutes"])
        if mkt_open <= now <= hb_end and self.last_post is not None \
                and datetime.now() - self.last_post >= timedelta(minutes=hb):
            last = f"{self.last_bar:%H:%M}" if self.last_bar is not None else "none yet"
            self.post(dz.simple("Still running", f"Last bar {last}. {self.today_line()}", dz.GREY))

    def close_open_trades(self) -> None:
        last_px = float(self.bars["close"].iloc[-1]) if len(self.bars) else None
        for tr in self.trades.values():
            if tr.taken and not tr.done:
                res = tr.result
                if res is not None and res.filled:
                    sign = 1 if tr.setup.direction == "long" else -1
                    risk = abs(res.fill_price - tr.setup.stop)
                    r = sign * (last_px - res.fill_price) / risk if risk else 0.0
                    self._resolve(tr, "close", round(r, 3), self.last_bar, last_px)
                else:
                    self._resolve(tr, "no fill", None, self.last_bar, None)

    def summary(self) -> tuple[str, list[dict]]:
        lines = []
        for tr in self.trades.values():
            if not tr.taken:
                continue
            res = tr.result
            outcome = tr.outcome or (res.outcome if res else "?")
            r = f"{tr.r:+.2f}R" if tr.r is not None else "-"
            lines.append(f"#{tr.trade_no} {tr.setup.direction} {tr.setup.time:%H:%M} "
                         f"score {tr.setup.score}: {outcome} {r}")
        skipped = sum(1 for tr in self.trades.values() if not tr.taken and tr.setup.band(self.cfg) == "alert")
        body = "\n".join(lines) or "No alerts today."
        if skipped:
            body += f"\n{skipped} more setup(s) skipped by the daily rules."
        return dz.simple(f"{self.symbol} daily summary", f"{body}\n\n{self.today_line()}", dz.BLUE)
