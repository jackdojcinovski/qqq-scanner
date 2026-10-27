"""Download REAL 0DTE option minute bars for QQQ / SPY, one file per symbol-day.

    python tools/download_chains.py                      # Feb 2024 -> today, QQQ and SPY
    python tools/download_chains.py --start 2025-01-01 --pct 3

For every trading day: the options expiring THAT day, every $1 strike within
--pct % of the day's open, calls and puts, 9:30-16:00 minute bars. Saved to
data/options/<SYM>_<date>_0dte.csv.gz. Already-downloaded days are skipped,
so you can stop it and run it again to resume. Needs the 1-minute stock
history (tools/download_history.py) for the open price.
"""
from __future__ import annotations

import argparse
import os
import sys
import time as clock
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scanner import options as op  # noqa: E402
from scanner.config import ROOT  # noqa: E402
from scanner.history import ET, load_history, trading_days  # noqa: E402

OPT_DIR = ROOT / "data" / "options"


def chain_path(symbol: str, day: date) -> Path:
    return OPT_DIR / f"{symbol}_{day}_0dte.csv.gz"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", default="QQQ,SPY")
    p.add_argument("--start", default="2024-02-05")
    p.add_argument("--end", default=str(date.today()))
    p.add_argument("--pct", type=float, default=3.0, help="strike range around the open, in %%")
    p.add_argument("--batch", type=int, default=80, help="option symbols per request")
    args = p.parse_args()

    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.requests import OptionBarsRequest
    from alpaca.data.timeframe import TimeFrame

    load_dotenv(ROOT / ".env")
    client = OptionHistoricalDataClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET"))
    OPT_DIR.mkdir(parents=True, exist_ok=True)
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    for sym in [s.strip().upper() for s in args.symbols.split(",")]:
        bars = load_history(sym, start, end)
        if bars.empty:
            raise SystemExit(f"No stock history for {sym}. Run tools/download_history.py first.")
        days = trading_days(bars)
        todo = [d for d in days if not chain_path(sym, d).exists()]
        print(f"{sym}: {len(days)} days, {len(todo)} to download")
        t0 = clock.time()
        for n, d in enumerate(todo, 1):
            day_bars = bars[bars.index.date == d]
            rth = day_bars[(day_bars.index.hour * 60 + day_bars.index.minute) >= 570]
            if rth.empty:
                continue
            o = float(rth["open"].iloc[0])
            lo, hi = int(o * (1 - args.pct / 100)), int(o * (1 + args.pct / 100)) + 1
            syms = [op.occ_symbol(sym, d, r, k) for k in range(lo, hi + 1) for r in ("C", "P")]
            s_t = pd.Timestamp(d.year, d.month, d.day, 9, 30, tz=ET)
            frames = []
            for i in range(0, len(syms), args.batch):
                req = OptionBarsRequest(symbol_or_symbols=syms[i:i + args.batch], timeframe=TimeFrame.Minute,
                                        start=s_t.to_pydatetime(),
                                        end=(s_t + pd.Timedelta(hours=6, minutes=31)).to_pydatetime())
                for attempt in range(3):
                    try:
                        df = client.get_option_bars(req).df
                        break
                    except Exception as exc:                             # noqa: BLE001
                        print(f"  {d}: {exc}; retrying")
                        clock.sleep(3 * (attempt + 1))
                        df = pd.DataFrame()
                if df is not None and not df.empty:
                    frames.append(df.reset_index())
                clock.sleep(0.3)
            out = pd.concat(frames) if frames else pd.DataFrame(columns=["symbol", "timestamp", "open", "high",
                                                                         "low", "close", "volume"])
            out = out[[c for c in ["symbol", "timestamp", "open", "high", "low", "close", "volume"] if c in out.columns]]
            out.to_csv(chain_path(sym, d), index=False, compression="gzip")
            if n % 10 == 0 or n == len(todo):
                rate = (clock.time() - t0) / n
                print(f"  {sym} {d}: {n}/{len(todo)} days, {len(out)} bars, ~{rate * (len(todo) - n) / 60:.0f} min left")
    print("done. Next: python tools/credit_study.py")


if __name__ == "__main__":
    main()
