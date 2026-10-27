"""Run the live QQQ scanner for today.

    python run_scanner.py            # normal run
    python run_scanner.py --dry-run  # print messages instead of posting to Discord

Start it before the open (8:30 ET is ideal). It sends a start message, alerts
and follow-ups during the session, a daily summary after the trades are done,
and then exits. Ctrl+C stops it cleanly. Keep the laptop plugged in; the
script stops Windows from sleeping while it runs.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time as clock
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scanner import discord as dz  # noqa: E402
from scanner.backtest import exit_settings  # noqa: E402
from scanner.config import ROOT, load_config, parse_time  # noqa: E402
from scanner.feeds import Alpaca, keep_awake  # noqa: E402
from scanner.history import ET  # noqa: E402
from scanner.live import REST_DELAY, LiveScanner  # noqa: E402
from scanner.news import NewsFilter  # noqa: E402
from scanner.store import Store  # noqa: E402


class PrintNotifier:
    def send(self, content="", embeds=None):
        print("\n--- DISCORD ---\n" + content)
        for e in embeds or []:
            print(e.get("title", ""))
            for f in e.get("fields", []):
                print(f"  {f['name']}: {f['value']}")
            if e.get("description"):
                print("  " + e["description"])
        return True


def now_et() -> pd.Timestamp:
    return pd.Timestamp.now(tz=ET).floor("s")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    (ROOT / "data").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(ROOT / "data" / "scanner.log", encoding="utf-8")])
    load_dotenv(ROOT / ".env")
    key, secret, hook = (os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET"),
                         os.getenv("DISCORD_WEBHOOK_URL"))
    if not key or not secret or (not hook and not args.dry_run):
        raise SystemExit("Fill ALPACA_API_KEY, ALPACA_API_SECRET and DISCORD_WEBHOOK_URL in .env")

    cfg = load_config()
    symbol = cfg["symbol"]
    notifier = PrintNotifier() if args.dry_run else dz.Notifier(hook)
    api = Alpaca(key, secret)
    today = now_et().date()

    session = api.session(today)
    if session is None:
        print(f"Market closed on {today}. Nothing to do.")
        return
    open_ts, close_ts = session

    # wait for the configured start time if launched very early
    start_at = pd.Timestamp.combine(today, parse_time(cfg["live"]["start_time"])).tz_localize(ET)
    while now_et() < start_at - pd.Timedelta(minutes=1):
        clock.sleep(20)

    news = NewsFilter.from_config(cfg)
    status = news.status(now_et().to_pydatetime())
    events = [f"{e.when:%H:%M} {e.title}" for e in news.day_events(today)]

    # history: previous session + today so far (REST can't return the last 15 minutes)
    prev = api.previous_session(today)
    t_now = now_et()
    hist_start = pd.Timestamp.combine(prev, parse_time("04:00")).tz_localize(ET)
    rest_end = t_now - REST_DELAY
    history = api.bars(symbol, hist_start, rest_end)

    sc = LiveScanner(cfg, symbol, notifier, history, news=news,
                     backfill=lambda a, b: api.bars(symbol, a, b),
                     option_quote=lambda plan: api.option_mid(
                         f"{symbol}{plan['expiry']:%y%m%d}{plan['right']}{int(plan['strike'] * 1000):08d}"),
                     store=Store())
    sc.add_gap(rest_end, t_now + pd.Timedelta(minutes=1))   # filled automatically once REST catches up

    late = t_now > open_ts - pd.Timedelta(minutes=14)
    lines = [f"Session {open_ts:%H:%M}-{close_ts:%H:%M} ET; alerts {cfg['window']['start']}-{cfg['window']['end']}",
             f"Exit: {exit_settings(cfg)['mode']}; up to {cfg['limits']['max_alerts_per_day']} alerts, "
             f"stop after {cfg['limits']['stop_after_losses']} loss",
             "Red-folder USD today: " + (", ".join(events) if events else "none")]
    if status.blocked:
        lines.append(f"Standing down: {status.reason}")
    if late:
        lines.append(f"Started late: alerts wait until today's data is backfilled (~{t_now + REST_DELAY:%H:%M}).")
    sc.post(dz.simple(f"{'[PAPER] ' if cfg['live'].get('paper', True) else ''}{symbol} scanner started",
                      "\n".join(lines), dz.BLUE))
    if status.blocked and "standing down" in status.reason:
        print("Full-day news block. Exiting.")
        return

    keep_awake(True)
    lock = threading.Lock()
    stop = threading.Event()
    summary_at = close_ts if close_ts.hour < 16 else None
    trades_done_at = (pd.Timestamp.combine(today, parse_time(cfg["window"]["end"])).tz_localize(ET)
                      + pd.Timedelta(minutes=exit_settings(cfg)["time_exit_minutes"]
                                     + cfg["trade"]["entry_valid_minutes"]
                                     + cfg["live"]["summary_after_close_minutes"]))
    summary_at = min(summary_at or close_ts, trades_done_at)

    stream = api.stream()

    async def on_bar(bar):
        ts = pd.Timestamp(bar.timestamp).tz_convert(ET)
        with lock:
            try:
                sc.on_bar(ts, bar.open, bar.high, bar.low, bar.close, bar.volume, now=now_et())
            except Exception:
                logging.exception("error processing bar %s", ts)

    stream.subscribe_bars(on_bar, symbol)

    def housekeeping():
        while not stop.is_set():
            t = now_et()
            with lock:
                try:
                    sc.tick(t)
                    if t >= summary_at and not sc.summary_sent:
                        sc.close_open_trades()
                        sc.post(sc.summary())
                        sc.summary_sent = True
                        stop.set()
                except Exception:
                    logging.exception("housekeeping error")
            stop.wait(30)
        try:
            stream.stop()
        except Exception:
            pass

    hk = threading.Thread(target=housekeeping, daemon=True)
    hk.start()
    try:
        while not stop.is_set():
            gap_start = now_et()
            try:
                stream.run()          # blocks until stopped or the connection drops
            except Exception as exc:
                logging.warning("stream error: %s", exc)
            if stop.is_set():
                break
            # connection dropped: remember the gap so it gets backfilled, then reconnect
            with lock:
                sc.add_gap(gap_start if sc.last_bar is None else sc.last_bar, now_et())
            clock.sleep(5)
            stream = api.stream()
            stream.subscribe_bars(on_bar, symbol)
    except KeyboardInterrupt:
        with lock:
            sc.post(dz.simple("Scanner stopped", "Stopped by hand. Open trades are no longer tracked.\n"
                              + sc.today_line(), dz.GREY))
    finally:
        stop.set()
        keep_awake(False)
    print("Done for today.")


if __name__ == "__main__":
    main()
