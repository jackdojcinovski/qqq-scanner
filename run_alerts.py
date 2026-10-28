"""Live alerts for today: scans QQQ 9:30-11:00 for the common day-trading setups, alerts ONE 0DTE trade.

    python run_alerts.py --check          # tonight: checks your keys, data, option quotes and Discord
    python run_alerts.py --replay 2026-10-23   # replay a past day from downloaded data (prints, no Discord)
    python run_alerts.py                  # tomorrow morning: start any time before 9:30 ET and leave it running
    python run_alerts.py --dry-run        # live data, but print messages instead of posting to Discord
    python run_alerts.py --sim            # after the fact: replay TODAY's real data and post it to Discord as a SIMULATION
    python run_alerts.py --sim --ignore-news   # same, but show the trade even on a red-folder news day

Setups: opening-range breakout, VWAP reclaim, VWAP pullback bounce, flag breakout, yesterday's high/low
break, 9/20 EMA cross - only when price is on the right side of VWAP and 4 of 5 confirmations agree
(VWAP, 9/20 EMA curl, volume vs normal, volume spike, vs today's open). See scanner/day_setups.py.
Trade: at-the-money 0DTE call/put, ~$1,000, +50% / -35% (strong setups +100% / -25%), out within 90 min,
no trade on red-folder days. Posts the entry, watches the option every 5 s, posts the exit and a
recap with running totals. Every day is logged to journal/trades.csv.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time as clock
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scanner import live_day as ld  # noqa: E402
from scanner import orb_alert as oa  # noqa: E402
from scanner.config import ROOT, load_config  # noqa: E402
from scanner.discord import Notifier  # noqa: E402
from scanner.feeds import Alpaca, keep_awake  # noqa: E402
from scanner.history import ET, load_history  # noqa: E402

log = logging.getLogger("alerts")


class LiveAPI:
    def __init__(self, key: str, secret: str):
        self.a = Alpaca(key, secret)

    def now(self) -> pd.Timestamp:
        return pd.Timestamp.now(tz=ET)

    def sleep(self, seconds: float) -> None:
        clock.sleep(seconds)

    def is_open(self, day: date) -> bool:
        return self.a.session(day) is not None

    def latest_bar(self, symbol: str):
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockLatestBarRequest
        try:
            b = self.a.stocks.get_stock_latest_bar(StockLatestBarRequest(symbol_or_symbols=symbol, feed=DataFeed.IEX))[symbol]
        except Exception as exc:                                    # noqa: BLE001
            log.warning("latest bar failed: %s", exc)
            return None
        ts = pd.Timestamp(b.timestamp).tz_convert(ET)
        return ts, {"open": b.open, "high": b.high, "low": b.low, "close": b.close, "volume": b.volume}

    def option_quote(self, occ: str):
        from alpaca.data.enums import OptionsFeed
        from alpaca.data.requests import OptionLatestQuoteRequest
        try:
            q = self.a.options.get_option_latest_quote(
                OptionLatestQuoteRequest(symbol_or_symbols=occ, feed=OptionsFeed.INDICATIVE)).get(occ)
        except Exception as exc:                                    # noqa: BLE001
            log.warning("option quote failed: %s", exc)
            return None
        if not q or not q.ask_price:
            return None
        return oa.Quote(float(q.bid_price or 0), float(q.ask_price), pd.Timestamp(q.timestamp).tz_convert(ET))

    def history(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        return self.a.bars(symbol, start, end)


class PrintNotifier:
    def send(self, content="", embeds=None):
        for e in embeds or []:
            print("\n=== " + e.get("title", ""))
            for f in e.get("fields", []):
                print(f"  {f['name']}: {f['value']}")
            if e.get("description"):
                print("  " + e["description"].replace("\n", "\n  "))
            if e.get("footer"):
                print("  (" + e["footer"]["text"] + ")")
        return True


def news_today(cfg, day: date) -> tuple[list[str], bool]:
    from scanner.news import NewsFilter
    try:
        nf = NewsFilter.from_config(cfg)
    except Exception as exc:                                        # noqa: BLE001
        log.warning("calendar failed: %s", exc)
        return [], False
    return [f"{e.when:%H:%M} {e.title}" for e in nf.day_events(day)], nf.available


def option_series(api: "LiveAPI", day: date):
    """occ -> minute closes for `day` from Alpaca's historical option bars."""
    from alpaca.data.requests import OptionBarsRequest
    from alpaca.data.timeframe import TimeFrame

    def fetch(occ: str):
        try:
            df = api.a.options.get_option_bars(OptionBarsRequest(
                symbol_or_symbols=occ, timeframe=TimeFrame.Minute,
                start=oa.at(day, 9, 30).to_pydatetime(), end=oa.at(day, 16, 0).to_pydatetime())).df
        except Exception as exc:                                    # noqa: BLE001
            log.warning("option bars failed for %s: %s", occ, exc)
            return None
        if df is None or df.empty:
            return None
        df = df.xs(occ, level="symbol") if "symbol" in df.index.names else df
        return pd.Series(df["close"].to_numpy(), index=pd.to_datetime(df.index).tz_convert(ET))
    return fetch


def run_sim(api: "LiveAPI", notify, args) -> None:
    day = api.now().date() if args.sim == "today" else date.fromisoformat(args.sim)
    end = min(oa.at(day, 16, 0), api.now() - pd.Timedelta(minutes=16))     # free plan: nothing newer than 15 min
    if end < oa.at(day, 11, 0):
        raise SystemExit("Too early - wait until about 11:16 ET so the 9:30-11:00 data is available.")
    stock = api.history(args.symbol, oa.at(day, 4, 0) - pd.Timedelta(days=35), end)
    if stock.empty or stock[stock.index.date == day].empty:
        raise SystemExit(f"No {args.symbol} data for {day}.")
    sim = oa.FetchingReplayAPI(day, stock, option_series(api, day))
    events, avail = ([], True) if args.ignore_news else news_today(load_config(), day)

    def sim_notify(content="", embeds=None):
        for e in embeds or []:
            e["title"] = "SIMULATION - " + e.get("title", "")
            e["footer"] = {"text": f"Replayed on {pd.Timestamp.now(tz=ET):%b %d %H:%M} ET from today's real data. "
                                   "Not a live alert."}
        return notify(content, embeds)

    result = ld.run_day(sim, sim_notify, args.symbol, news_events=events, news_ok=avail,
                        journal=ROOT / "journal" / "simulated.csv", budget=args.budget, log=log.info)
    print(result)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="test keys, data, quotes and Discord, then exit")
    p.add_argument("--dry-run", action="store_true", help="print instead of posting to Discord")
    p.add_argument("--replay", help="YYYY-MM-DD: replay a past day from downloaded data")
    p.add_argument("--sim", nargs="?", const="today", help="replay today (or YYYY-MM-DD) from Alpaca data, post to Discord")
    p.add_argument("--ignore-news", action="store_true", help="with --sim: don't stand down on red-folder days")
    p.add_argument("--symbol", default="QQQ")
    p.add_argument("--budget", type=float, default=1000.0)
    args = p.parse_args()

    (ROOT / "data").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(ROOT / "data" / "alerts.log", encoding="utf-8")])

    if args.replay:
        d = date.fromisoformat(args.replay)
        stock = load_history(args.symbol, d - timedelta(days=40), d)
        f = ROOT / "data" / "options" / f"{args.symbol}_{d}_0dte.csv.gz"
        if stock.empty or not f.exists():
            raise SystemExit("Need data/history and data/options for that day (download_history / download_chains).")
        api = oa.ReplayAPI(d, stock, pd.read_csv(f))
        print(ld.run_day(api, PrintNotifier().send, args.symbol, journal=ROOT / "data" / "replay_journal.csv",
                         budget=args.budget, log=print))
        return

    load_dotenv(ROOT / ".env")
    key, secret, hook = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET"), os.getenv("DISCORD_WEBHOOK_URL")
    if not key or not secret or (not hook and not args.dry_run):
        raise SystemExit("Fill ALPACA_API_KEY, ALPACA_API_SECRET and DISCORD_WEBHOOK_URL in .env")
    api = LiveAPI(key, secret)
    notify = PrintNotifier().send if args.dry_run else Notifier(hook, username="QQQ TEST alerts").send

    if args.sim:
        run_sim(api, notify, args)
        return

    if args.check:
        ok = True
        bar = api.latest_bar(args.symbol)
        print(f"latest {args.symbol} bar: {bar[0]:%Y-%m-%d %H:%M} close {bar[1]['close']:.2f}" if bar else "NO stock bar")
        ok &= bar is not None
        d = api.now().date() + timedelta(days=1)
        while not api.is_open(d):
            d += timedelta(days=1)
        if bar:
            sym = oa.occ(args.symbol, d, "C", round(bar[1]["close"]))
            q = api.option_quote(sym)
            print(f"option quote {sym}: " + (f"bid {q.bid:.2f} ask {q.ask:.2f} at {q.ts:%H:%M}" if q else "NONE"))
            ok &= q is not None
        hist = api.history(args.symbol, pd.Timestamp.now(tz=ET) - pd.Timedelta(days=10), pd.Timestamp.now(tz=ET) - pd.Timedelta(minutes=20))
        print(f"history bars (last 10 days): {len(hist)}")
        ok &= len(hist) > 0
        events, avail = news_today(load_config(), d)
        print(f"calendar for {d}: " + ("unavailable" if not avail else (", ".join(events) or "no red-folder USD events")))
        sent = notify("", oa.embed("QQQ TEST alerts - setup check", oa.BLUE,
                                   [("Status", "Everything connected." if ok else "Something failed - see the terminal."),
                                    ("Next session", f"{d:%A %b %d}: start `python run_alerts.py` before 9:30 ET")]))
        print("discord message sent" if sent else "DISCORD FAILED")
        print("\nALL GOOD - ready for tomorrow." if ok and sent else "\nFix the item(s) marked above.")
        return

    keep_awake(True)
    try:
        events, avail = news_today(load_config(), api.now().date())
        result = ld.run_day(api, notify, args.symbol, news_events=events, news_ok=avail,
                            journal=ROOT / "journal" / "trades.csv", budget=args.budget, log=log.info)
        log.info("done: %s", result)
    finally:
        keep_awake(False)


if __name__ == "__main__":
    main()
