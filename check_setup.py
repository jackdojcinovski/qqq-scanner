"""Phase 1 check: confirms your keys work, Discord receives a message,
and live QQQ 1-minute bars stream in.

Run:  python check_setup.py
Stop: Ctrl+C
"""
import os
import sys
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

load_dotenv()
KEY = os.getenv("ALPACA_API_KEY")
SECRET = os.getenv("ALPACA_API_SECRET")
WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL")
SYMBOL = "QQQ"


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def check_env() -> None:
    missing = [n for n, v in [("ALPACA_API_KEY", KEY), ("ALPACA_API_SECRET", SECRET),
                              ("DISCORD_WEBHOOK_URL", WEBHOOK)] if not v or "your_" in v or "xxxx" in v]
    if missing:
        fail(f"Fill these in your .env file: {', '.join(missing)}")
    print("[ OK ] .env loaded")


def check_discord() -> None:
    r = requests.post(WEBHOOK, json={"content": "QQQ scanner: setup check, Discord is connected."}, timeout=10)
    if r.status_code not in (200, 204):
        fail(f"Discord returned {r.status_code}: {r.text[:200]}")
    print("[ OK ] Discord message sent - check your channel")


def check_history() -> None:
    # Free plan can't query the latest 15 minutes, so ask for the previous day's daily bar.
    client = StockHistoricalDataClient(KEY, SECRET)
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    req = StockBarsRequest(symbol_or_symbols=SYMBOL, timeframe=TimeFrame.Day,
                           start=end - timedelta(days=7), end=end, feed=DataFeed.IEX)
    bars = client.get_stock_bars(req).df
    if bars.empty:
        fail("No historical bars returned - check your keys")
    last = bars.iloc[-1]
    print(f"[ OK ] Historical data works. Last daily bar: high {last['high']:.2f}, "
          f"low {last['low']:.2f}, close {last['close']:.2f}")


async def on_bar(bar) -> None:
    t = bar.timestamp.astimezone().strftime("%H:%M")
    print(f"[BAR ] {t}  O {bar.open:.2f}  H {bar.high:.2f}  L {bar.low:.2f}  "
          f"C {bar.close:.2f}  V {int(bar.volume)}")


def check_stream() -> None:
    print("[ .. ] Connecting to live stream. Bars arrive about once a minute "
          "while the market is open (9:30-16:00 ET). Ctrl+C to stop.")
    stream = StockDataStream(KEY, SECRET, feed=DataFeed.IEX)
    stream.subscribe_bars(on_bar, SYMBOL)
    stream.run()


if __name__ == "__main__":
    check_env()
    check_discord()
    check_history()
    check_stream()
