"""Re-price the combined model's trades with REAL option prices (Alpaca, Feb 2024 on).

    python tools/model_study.py          # writes reports/model_trades.csv
    python tools/real_model.py           # prices those trades with real option bars

For every trade the model took since --start: same symbol, same at-the-money
strike (nearest $1), same expiry, bought at the traded price in the entry
minute + half the spread; sold at the take profit / stop on the premium if a
real bar traded there (stop checked first), else at 15:59 - half the spread.
Sizing: as many contracts as --budget buys. FX --fx each way (0 = USD account).

Writes reports/real_model_<date>.md.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time as clock
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scanner import optionsim as om  # noqa: E402
from scanner import options as op  # noqa: E402
from scanner.config import ROOT  # noqa: E402
from scanner.history import ET, load_history  # noqa: E402


def ts(day: date, minutes_after_open: float) -> pd.Timestamp:
    return pd.Timestamp(day.year, day.month, day.day, 9, 30, tz=ET) + pd.Timedelta(minutes=minutes_after_open)


def real_exit(df: pd.DataFrame, t_in, t_out, paid: float, tp, sl) -> tuple[float, str] | None:
    """Walk real option bars after the entry minute: stop first, then target, else the close."""
    target = paid * (1 + tp) if tp is not None else None
    stop = paid * (1 - sl) if sl is not None else None
    during = df[(df.index > t_in) & (df.index < t_out)]
    for _, b in during.iterrows():
        if stop is not None and b["low"] - om.half_spread(b["low"]) <= stop:
            return min(stop, b["open"] - om.half_spread(b["open"])), "stop"
        if target is not None and b["high"] - om.half_spread(b["high"]) >= target:
            return max(target, b["open"] - om.half_spread(b["open"])), "target"
    px = op.price_at(df, t_out)
    if px is None:
        return None
    return max(px - om.half_spread(px), 0.0), "close"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trades", default=str(ROOT / "reports" / "model_trades.csv"))
    p.add_argument("--budget", type=float, default=1000.0)
    p.add_argument("--fx", type=float, default=0.0)
    p.add_argument("--start", default="2024-02-05")
    args = p.parse_args()

    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.requests import OptionBarsRequest
    from alpaca.data.timeframe import TimeFrame

    start = date.fromisoformat(args.start)
    with open(args.trades, encoding="utf-8") as f:
        trades = [t for t in csv.DictReader(f) if date.fromisoformat(t["day"]) >= start]
    if not trades:
        raise SystemExit("No trades. Run tools/model_study.py first.")
    syms = sorted({t["symbol"] for t in trades})
    bars = {s: load_history(s, start, date.today()) for s in syms}

    load_dotenv(ROOT / ".env")
    client = OptionHistoricalDataClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET"))
    rows, missing = [], 0
    print(f"{len(trades)} model trades since {start}; fetching real option bars ...")
    for n, t in enumerate(trades, 1):
        day = date.fromisoformat(t["day"])
        m_in = int(t["m_in"])
        t_in, t_out = ts(day, m_in), ts(day, 389.5)
        b = bars[t["symbol"]]
        at = b[b.index == t_in]
        if at.empty:
            missing += 1
            continue
        right = "C" if t["direction"] == "long" else "P"
        strike = float(round(float(at["open"].iloc[0])))
        expiry = om.expiry_date(day, int(t["dte"]))
        sym = op.occ_symbol(t["symbol"], expiry, right, strike)
        try:
            df = client.get_option_bars(OptionBarsRequest(
                symbol_or_symbols=sym, timeframe=TimeFrame.Minute,
                start=(t_in - pd.Timedelta(minutes=10)).to_pydatetime(),
                end=(t_out + pd.Timedelta(minutes=2)).to_pydatetime())).df
        except Exception as exc:                                         # noqa: BLE001
            print(f"  {sym}: {exc}")
            df = pd.DataFrame()
        clock.sleep(0.35)
        if df is None or df.empty:
            missing += 1
            continue
        df = df.xs(sym, level="symbol") if "symbol" in df.index.names else df
        df.index = pd.to_datetime(df.index).tz_convert(ET)
        buy = op.price_at(df, t_in)
        if buy is None or buy <= 0:
            missing += 1
            continue
        paid = buy + om.half_spread(buy)
        tp = float(t["tp"]) if t["tp"] else None
        sl = float(t["sl"]) if t["sl"] else None
        ex = real_exit(df, t_in, t_out, paid, tp, sl)
        if ex is None:
            missing += 1
            continue
        got, why = ex
        k = max(1, int(args.budget // (paid * 100)))
        real = k * 100 * ((got - paid) - args.fx * (paid + got))
        rows.append(dict(day=day, symbol=t["symbol"], setup=t["setup"], side=t["direction"], option=sym, n=k,
                         paid=paid, got=got, why=why, real=real, model=float(t["model_pnl"]),
                         model_paid=float(t["model_paid"]), period=t["period"]))
        if n % 20 == 0 or n == len(trades):
            print(f"  {n}/{len(trades)} trades, {len(rows)} priced, {missing} without option data")

    L = [f"# Combined model on REAL option prices ({datetime.now():%Y-%m-%d})\n",
         f"- {len(trades)} model trades since {start}; {len(rows)} priced with real option bars, {missing} skipped",
         f"- About ${args.budget:,.0f} per trade, FX {args.fx:.1%} each way, half the spread paid on each side\n"]
    if rows:
        weeks = max(len(pd.bdate_range(start, max(r["day"] for r in rows))) / 5, 1)

        def block(rs, label):
            real = [r["real"] for r in rs]
            mod = [r["model"] for r in rs]
            eq = peak = dd = 0.0
            for x in real:
                eq += x
                peak = max(peak, eq)
                dd = max(dd, peak - eq)
            wins = [x for x in real if x > 0]
            loss = [x for x in real if x <= 0]
            return [f"### {label}\n", "| | real | model |", "|---|---|---|",
                    f"| trades | {len(rs)} | {len(rs)} |",
                    f"| win rate | {len(wins) / len(rs):.0%} | {sum(x > 0 for x in mod) / len(mod):.0%} |",
                    f"| avg winner / loser | ${statistics.mean(wins or [0]):+,.0f} / ${statistics.mean(loss or [0]):+,.0f} | |",
                    f"| $ per trade | ${statistics.mean(real):+,.0f} | ${statistics.mean(mod):+,.0f} |",
                    f"| total | ${sum(real):+,.0f} | ${sum(mod):+,.0f} |",
                    f"| worst drawdown | ${dd:,.0f} | |", ""]

        L += block(rows, "All priced trades")
        L.append(f"$ per week over the whole period: **${sum(r['real'] for r in rows) / weeks:+,.0f}** "
                 f"(model ${sum(r['model'] for r in rows) / weeks:+,.0f})\n")
        chk = [r for r in rows if r["period"] == "check"]
        if chk:
            L += block(chk, "Unseen check year only")
        ratio = statistics.median(r["paid"] / r["model_paid"] for r in rows)
        L.append(f"**Calibration:** real contracts cost {ratio:.2f}x the model's price.\n")
        for key, title in (("setup", "By setup"), ("symbol", "By symbol")):
            agg: dict = {}
            for r in rows:
                agg.setdefault(r[key], []).append(r["real"])
            L += [f"### {title}\n", "| | trades | win | real $ total |", "|---|---|---|---|"]
            for k2, v in sorted(agg.items(), key=lambda x: -sum(x[1])):
                L.append(f"| {k2} | {len(v)} | {sum(x > 0 for x in v) / len(v):.0%} | ${sum(v):+,.0f} |")
            L.append("")
        by_m: dict = {}
        for r in rows:
            by_m.setdefault(f"{r['day']:%Y-%m}", []).append(r["real"])
        L += ["### By month\n", "| month | trades | real $ |", "|---|---|---|"]
        L += [f"| {m} | {len(v)} | ${sum(v):+,.0f} |" for m, v in sorted(by_m.items())]
        L += ["", "### Every trade\n", "| day | symbol | setup | side | option | contracts | paid | sold | exit | real $ | model $ |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            L.append(f"| {r['day']} | {r['symbol']} | {r['setup']} | {r['side']} | {r['option']} | {r['n']} | {r['paid']:.2f} "
                     f"| {r['got']:.2f} | {r['why']} | ${r['real']:+,.0f} | ${r['model']:+,.0f} |")
    out = ROOT / "reports" / f"real_model_{datetime.now():%Y-%m-%d}.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:16]))
    print(f"\nreport written: {out}")


if __name__ == "__main__":
    main()
