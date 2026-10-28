"""Jack's day-trading strategy on REAL 0DTE option prices.

    python tools/morning_study.py        # needs data/options from tools/download_chains.py

The rules being tested:
  * get in between 9:30 and 11:00 only; out within 60 or 90 minutes ("hot potato")
  * base plan: +50% target, tight stop (-20% / -25% / -35%; -50% at most)
  * bigger targets (+100% / +150%) only when the big-move signs line up, with a TIGHTER stop
  * or scale out: half at +50%, stop to breakeven on the rest, let it run
  * ~$1,000 per trade, one trade a day, QQQ (SPY if QQQ can't be priced)

Entries: the setups in scanner/library.py that fire before 11:00 - opening range (5/15/30 min),
VWAP, noise boundary, gaps, yesterday's high/low, 10:30 momentum, and the ICT-style ones:
Silver Bullet (sweep + FVG, 10-11), Judas swing (yesterday's / opening-range high or low swept and
reclaimed), opening-range retest, first pullback after the opening drive. Plus 'coin': a coin-flip
CONTROL that must lose - if it doesn't, the backtest is flattering itself.

Big-move signs (scanner/bigmove.py) are chosen on the TUNE period by how much they raise the chance
of a 0.35%+ move our way within the hour. 'Adaptive' plans use them: when a trade has >= N signs it
goes for +100% / +150% with a tight stop, otherwise the +50% base plan.

Choices on TUNE; the last 12 months (CHECK) looked at once. Writes reports/morning_<date>.md.
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scanner import bigmove as bm  # noqa: E402
from scanner import debit as db  # noqa: E402
from scanner import intraday as ix  # noqa: E402
from scanner import library as lib  # noqa: E402
from scanner import optionsim as om  # noqa: E402
from scanner import spreads as sp  # noqa: E402
from scanner.config import ROOT  # noqa: E402
from scanner.history import ET, load_history  # noqa: E402

OPT_DIR = ROOT / "data" / "options"
LAST_ENTRY = 90                                  # 11:00
SETUPS = lib.SETUPS + ["coin"]
MODES = ["both", "long", "regime_k1"]
STRUCTS = ["atm", "itm2", "ds2"]
HOLDS = [60, 90]
PLANS = db.plans()
PH = list(itertools.product(range(len(PLANS)), HOLDS))       # (plan index, hold) columns
BIG = 0.0035
SIGN_KEYS = list(bm.SIGNS)
# adaptive = (big plan, base plan, signs needed)
BIG_PLANS = [("fixed", 1.5, 0.2), ("fixed", 1.5, 0.25), ("fixed", 1.0, 0.2), ("fixed", 1.0, 0.25),
             ("scale", 1.5, 0.25), ("scale", None, 0.25)]
BASE_PLANS = [("fixed", 0.5, 0.25), ("fixed", 0.5, 0.35)]
NEEDED = [2, 3]


def coin_signals(grids) -> list[om.Signal]:
    out = []
    for sym, g in grids.items():
        for i in range(21, len(g.days)):
            d = g.days[i]
            rng = np.random.default_rng(int(d.strftime("%Y%m%d")))
            direction = "long" if rng.random() < 0.5 else "short"
            out.append(om.Signal("coin", sym, d, direction, 15, ix.MINUTES - 1, float(g.close[i, -1]), ix.MINUTES - 0.5))
    return out


def stats(days, pnl, held, n_days: int) -> dict:
    weeks = max(n_days / 5, 1)
    p = np.asarray(pnl, float)
    if len(p) == 0:
        return dict(n=0, wk=0.0, win=0.0, aw=0.0, al=0.0, per=0.0, med=0.0, usd_wk=0.0, total=0.0, ex5=0.0,
                    lo=-1e9, dd=0.0, hold=0.0, years="0/0")
    eq = np.concatenate([[0], np.cumsum(p)])
    yrs: dict = {}
    for d, x in zip(days, p):
        yrs[d.year] = yrs.get(d.year, 0.0) + x
    sd = p.std(ddof=1) if len(p) > 1 else 0.0
    return dict(n=len(p), wk=len(p) / weeks, win=float((p > 0).mean()),
                aw=float(p[p > 0].mean()) if (p > 0).any() else 0.0,
                al=float(p[p <= 0].mean()) if (p <= 0).any() else 0.0,
                per=float(p.mean()), med=float(np.median(p)), usd_wk=float(p.sum() / weeks), total=float(p.sum()),
                ex5=float(p.sum() - np.sort(p)[-5:].sum()),
                lo=float(p.mean() - 1.645 * sd / math.sqrt(len(p))) if len(p) > 1 else -1e9,
                dd=float((np.maximum.accumulate(eq) - eq).max()), hold=float(np.mean(held)),
                years=f"{sum(v > 0 for v in yrs.values())}/{len(yrs)}")


HEAD = ("| setup | sides | how bought | plan (target / stop) | max hold | trades | per week | win | avg win | avg loss "
        "| median | minutes held | $ per trade | **$ per week** | total | total without best 5 | worst drawdown | years + |")
SEP = "|" + "---|" * 18


def line(label, s, bold=False) -> str:
    b = "**" if bold else ""
    return (f"| {b}{label}{b} | {s['n']} | {s['wk']:.2f} | {s['win']:.0%} | ${s['aw']:+,.0f} | ${s['al']:+,.0f} "
            f"| ${s['med']:+,.0f} | {s['hold']:.0f} | ${s['per']:+,.0f} | **${s['usd_wk']:+,.0f}** | ${s['total']:+,.0f} "
            f"| ${s['ex5']:+,.0f} | ${s['dd']:,.0f} | {s['years']} |")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", default="QQQ,SPY")
    p.add_argument("--start", default="2024-02-05")
    p.add_argument("--budget", type=float, default=1000.0)
    p.add_argument("--slip", type=float, default=0.01, help="$ per leg per side")
    p.add_argument("--fx", type=float, default=0.0, help="FX fee each way (0 = USD account; 0.015 = CAD account)")
    p.add_argument("--max-contracts", type=int, default=50)
    p.add_argument("--split")
    p.add_argument("--min-trades", type=int, default=60)
    args = p.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    start = date.fromisoformat(args.start)
    grids, ctx, feats = {}, {}, {}
    for s in symbols:
        b = load_history(s, start - timedelta(days=90), date.today())
        if b.empty:
            raise SystemExit(f"No stock history for {s}.")
        grids[s] = ix.build_grid(s, b)
        ctx[s] = om.daily_context(grids[s])
        feats[s] = bm.Features(grids[s])
    pos = {s: {d: i for i, d in enumerate(g.days)} for s, g in grids.items()}
    have = sorted({d for s in symbols for d in grids[s].days
                   if d >= start and (OPT_DIR / f"{s}_{d}_0dte.csv.gz").exists()})
    if not have:
        raise SystemExit("No option chains. Run tools/download_chains.py first.")
    split = date.fromisoformat(args.split) if args.split else have[-1] - timedelta(days=365)
    tune_days, check_days = [d for d in have if d < split], [d for d in have if d >= split]

    print("signals ...")
    sigs = lib.all_signals(grids, ctx)
    sigs["coin"] = coin_signals(grids)
    chosen = {}
    for setup in SETUPS:
        for mode in MODES:
            out: dict = {}
            for s in sorted(sigs[setup], key=lambda s: (s.day, s.m_in, s.symbol != symbols[0])):
                if s.m_in > LAST_ENTRY or (mode == "long" and s.direction != "long"):
                    continue
                if mode == "regime_k1":
                    c = ctx[s.symbol]
                    down = bool(c.at[s.day, "down_k1"]) if s.day in c.index else False
                    if (s.direction == "long") == down:
                        continue
                out.setdefault(s.day, []).append(s)
            chosen[(setup, mode)] = out

    groups = list(itertools.product(SETUPS, MODES, STRUCTS))
    gidx = {g: k for k, g in enumerate(groups)}
    rec_group, rec_day, rec_mask, rec_m, rec_big, pnl_rows, held_rows = [], [], [], [], [], [], []
    rec_info, avg_rows, why_rows, rec_mv30, rec_r30 = [], [], [], [], []
    print(f"{len(have)} days with option data; {len(groups)} setup/side/contract groups x {len(PH)} plans ...")
    for n, d in enumerate(have, 1):
        chains = {}
        for s in symbols:
            f = OPT_DIR / f"{s}_{d}_0dte.csv.gz"
            if f.exists():
                chains[s] = sp.load_chain(pd.read_csv(f), d, ET)
        cache, sign_cache = {}, {}
        for setup, mode, st in groups:
            got = None
            for sig in chosen[(setup, mode)].get(d, []):
                ch = chains.get(sig.symbol)
                if ch is None or not ch.price:
                    continue
                key = (sig.symbol, sig.m_in, sig.direction, st)
                if key not in cache:
                    g, i = grids[sig.symbol], pos[sig.symbol][d]
                    legs, cap = db.legs_for(st, float(g.close[i, sig.m_in - 1]), sig.direction)
                    cache[key] = db.prepare_full(ch, g.close[i], legs, sig.m_in, cap, args.slip)
                if cache[key] is not None:
                    got = (cache[key], sig)
                    break
            if got is None:
                continue
            (debit, value, fresh), sig = got
            skey = (sig.symbol, sig.m_in, sig.direction)
            if skey not in sign_cache:
                i = pos[sig.symbol][d]
                sg = feats[sig.symbol].signs(i, sig.m_in, sig.direction)
                mask = sum(1 << k for k, name in enumerate(SIGN_KEYS) if sg and sg[name]) if sg else -1
                big = bm.max_move(grids[sig.symbol], i, sig.m_in, sig.direction, 60) >= BIG
                sign_cache[skey] = (mask, big)
            mask, big = sign_cache[skey]
            if mask < 0:
                continue
            k_n = min(args.max_contracts, max(1, int(args.budget // (debit * 100))))
            prow, hrow, arow, wrow = [], [], [], []
            for pi, hold in PH:
                avg, held, why = db.run_plan(debit, value, sig.m_in, PLANS[pi], sig.m_in + hold, fresh)
                prow.append(k_n * 100 * ((avg - debit) - args.fx * (avg + debit)))
                hrow.append(held)
                arow.append(avg)
                wrow.append(why)
            g, i = grids[sig.symbol], pos[sig.symbol][d]
            legs, cap = db.legs_for(st, float(g.close[i, sig.m_in - 1]), sig.direction)
            rec_info.append((sig.symbol, sig.direction, st, [(lg.right, lg.strike, lg.short) for lg in legs], debit, k_n,
                             len(legs)))
            avg_rows.append(arow)
            why_rows.append(wrow)
            if len(value) > 30 and sig.m_in + 30 < ix.MINUTES:
                rec_mv30.append(abs(g.close[i, sig.m_in + 30] / g.close[i, sig.m_in - 1] - 1))
                rec_r30.append(float(value[30]) / debit - 1)
            else:
                rec_mv30.append(np.nan)
                rec_r30.append(np.nan)
            rec_group.append(gidx[(setup, mode, st)])
            rec_day.append(d)
            rec_mask.append(mask)
            rec_m.append(sig.m_in)
            rec_big.append(big)
            pnl_rows.append(prow)
            held_rows.append(hrow)
        if n % 25 == 0 or n == len(have):
            print(f"  {n}/{len(have)} days")

    G = np.array(rec_group)
    D = np.array(rec_day)
    MASK = np.array(rec_mask)
    BIGM = np.array(rec_big)
    PNL = np.array(pnl_rows)
    HELD = np.array(held_rows)
    AVG = np.array(avg_rows)
    MV30, R30 = np.array(rec_mv30), np.array(rec_r30)
    is_tune = np.array([d < split for d in D])
    ph_index = {ph: k for k, ph in enumerate(PH)}
    plan_index = {pl: k for k, pl in enumerate(PLANS)}

    # ---- big-move signs: which count (tune, pooled over real setups, 'both' sides, at the money)
    ref = np.array([groups[g][0] != "coin" and groups[g][1] == "both" and groups[g][2] == "atm" for g in G]) & is_tune
    ref_col = ph_index[(plan_index[("fixed", 0.5, 0.35)], 60)]
    sign_rows, used = [], []
    for k, name in enumerate(SIGN_KEYS):
        has = (MASK & (1 << k)) > 0
        w, wo = ref & has, ref & ~has
        if w.sum() < 40 or wo.sum() < 40:
            sign_rows.append((name, int(w.sum()), None, None, None, None, False))
            continue
        bw, bwo = BIGM[w].mean(), BIGM[wo].mean()
        pw, pwo = PNL[w, ref_col].mean(), PNL[wo, ref_col].mean()
        use = (bw - bwo) >= 0.05
        if use:
            used.append(k)
        sign_rows.append((name, int(w.sum()), bw, bwo, pw, pwo, use))
    umask = sum(1 << k for k in used)
    GRADE = np.array([bin(int(m) & umask).count("1") for m in MASK])

    # ---- every configuration: (group, plan label, hold) -> per-trade pnl column
    configs = []                                            # (label parts, column getter)
    for gi, (setup, mode, st) in enumerate(groups):
        rows = np.flatnonzero(G == gi)
        if len(rows) == 0:
            continue
        for pi, hold in PH:
            col = ph_index[(pi, hold)]
            configs.append(((setup, mode, st, db.plan_label(PLANS[pi]), hold), rows, np.full(len(rows), col)))
        for bp, basep, need in itertools.product(BIG_PLANS, BASE_PLANS, NEEDED):
            for hold in HOLDS:
                cb = ph_index[(plan_index[bp], hold)]
                cs = ph_index[(plan_index[basep], hold)]
                cols = np.where(GRADE[rows] >= need, cb, cs)
                lab = f"ADAPTIVE: {need}+ signs -> {db.plan_label(bp)}, else {db.plan_label(basep)}"
                configs.append(((setup, mode, st, lab, hold), rows, cols))

    results = []
    for (setup, mode, st, lab, hold), rows, cols in configs:
        pnl, held = PNL[rows, cols], HELD[rows, cols]
        t = is_tune[rows]
        ts = stats(D[rows][t], pnl[t], held[t], len(tune_days))
        cs = stats(D[rows][~t], pnl[~t], held[~t], len(check_days))
        results.append(((setup, mode, st, lab, hold), ts, cs, rows, cols))
    real = [r for r in results if r[0][0] != "coin" and r[1]["n"] >= args.min_trades]
    coin = [r for r in results if r[0][0] == "coin" and r[1]["n"] >= args.min_trades]
    ranked = sorted(real, key=lambda r: r[1]["lo"] * r[1]["wk"], reverse=True)

    def lab(key):
        setup, mode, st, plan, hold = key
        return f"{setup} | {mode} | {st} | {plan} | {hold} min"

    L = [f"# Morning hot-potato strategy on REAL 0DTE option prices ({datetime.now():%Y-%m-%d})\n",
         "- Entries 9:30-11:00 only; out within 60 or 90 minutes. Base plan +50% with a tight stop; bigger targets "
         "only when big-move signs line up (ADAPTIVE plans), or half out at +50% and let the rest run.",
         f"- {', '.join(symbols)} ({symbols[0]} preferred), one trade a day per setup, about ${args.budget:,.0f} per trade "
         f"(max {args.max_contracts} contracts), ${args.slip:.2f} slippage per leg each way, $0 commission, no FX.",
         "- Real traded option prices; a spread's take profit only counts in a minute where both legs traded.",
         f"- Tune {tune_days[0]} to {tune_days[-1]} ({len(tune_days)} days); check {check_days[0]} to {check_days[-1]} "
         f"({len(check_days)} days, never used to choose).\n",
         "## What causes a big move? (tune; big = 0.35%+ our way within the hour)\n",
         "| sign | trades with it | big-move rate with | without | $ per trade with (+50%/-35%, 60 min) | without | counts? |",
         "|---|---|---|---|---|---|---|"]
    for name, nw, bw, bwo, pw, pwo, use in sign_rows:
        if bw is None:
            L.append(f"| {bm.SIGNS[name]} | {nw} | - | - | - | - | too few |")
        else:
            L.append(f"| {bm.SIGNS[name]} | {nw} | {bw:.0%} | {bwo:.0%} | ${pw:+,.0f} | ${pwo:+,.0f} | {'**yes**' if use else ''} |")
    L.append(f"\nSigns that count: {len(used)}. A trade's grade = how many it has.\n")

    if coin:
        both = sum(1 for r in coin if r[1]["usd_wk"] > 0 and r[2]["usd_wk"] > 0)
        L += ["## Control: coin-flip entries (must LOSE)\n",
              f"- Average over {len(coin)} coin settings: tune ${np.mean([r[1]['usd_wk'] for r in coin]):+,.0f}/week, "
              f"check ${np.mean([r[2]['usd_wk'] for r in coin]):+,.0f}/week; profitable on both: {both} of {len(coin)}.",
              "- " + ("OK: no edge from randomness, the test isn't flattering itself." if both <= 0.05 * len(coin) else
                      "WARNING: coin flips look profitable too often - treat every result below with suspicion.") + "\n"]

    def compare(title, fn, values):
        out = [f"## {title} (average over every other setting)\n",
               "| choice | settings | tune avg $/week | tune share profitable | check avg $/week | check share profitable |",
               "|---|---|---|---|---|---|"]
        for v in values:
            cs = [r for r in real if fn(r[0]) == v]
            if cs:
                out.append(f"| {v} | {len(cs)} | ${np.mean([r[1]['usd_wk'] for r in cs]):+,.0f} "
                           f"| {np.mean([r[1]['usd_wk'] > 0 for r in cs]):.0%} | ${np.mean([r[2]['usd_wk'] for r in cs]):+,.0f} "
                           f"| {np.mean([r[2]['usd_wk'] > 0 for r in cs]):.0%} |")
        return out + [""]

    L += compare("Entry setup", lambda k: k[0], lib.SETUPS)
    L += compare("Sides", lambda k: k[1], MODES)
    L += compare("How bought", lambda k: k[2], STRUCTS)
    plan_names = [db.plan_label(pl) for pl in PLANS] + sorted({k[3] for k, *_ in real if k[3].startswith("ADAPTIVE")})
    L += compare("Trade plan (target / stop)", lambda k: k[3], plan_names)
    L += compare("Max hold", lambda k: k[4], HOLDS)

    L += ["## Top 15 on tune, and the same settings on the unseen CHECK year\n", HEAD, SEP]
    for key, ts, cs, *_ in ranked[:15]:
        L.append(line(lab(key), ts))
        L.append(line("check: " + lab(key), cs, bold=True))

    # combined model: best settings per entry setup (tune), first signal of the day wins
    best = {}
    for r in ranked:
        best.setdefault(r[0][0], r)
    keep = [r for r in best.values() if r[1]["per"] > 0]
    L.append("\n## Combined model: best settings per entry setup (chosen on tune), first signal of the day wins\n")
    if keep:
        picks: dict = {}
        M_ARR = np.array(rec_m)
        for key, ts, cs, rows, cols in keep:
            for rr, col in zip(rows, cols):
                d = D[rr]
                cand = (M_ARR[rr], PNL[rr, col], HELD[rr, col], key[0], rr, col)
                if d not in picks or cand[0] < picks[d][0]:
                    picks[d] = cand
        days = sorted(picks)
        t_days = [d for d in days if d < split]
        c_days = [d for d in days if d >= split]
        ct = stats(t_days, [picks[d][1] for d in t_days], [picks[d][2] for d in t_days], len(tune_days))
        cc = stats(c_days, [picks[d][1] for d in c_days], [picks[d][2] for d in c_days], len(check_days))
        L.append("Uses: " + "; ".join(f"`{lab(r[0])}`" for r in keep) + "\n")
        L += [HEAD, SEP, line("combined | mixed | mixed | mixed | mixed", ct),
              line("check: combined | mixed | mixed | mixed | mixed", cc, bold=True)]
        mix: dict = {}
        for d in days:
            mix[picks[d][3]] = mix.get(picks[d][3], 0) + 1
        L.append("\nWho took the day: " + ", ".join(f"{k} {v}" for k, v in sorted(mix.items(), key=lambda x: -x[1])))
        by_m: dict = {}
        for d in days:
            by_m.setdefault(f"{d:%Y-%m}", []).append(picks[d][1])
        L += ["", "| month | trades | $ |", "|---|---|---|"]
        L += [f"| {m} | {len(v)} | ${sum(v):+,.0f} |" for m, v in sorted(by_m.items())]
        # ---- cost of trading, and the day-by-day trade log
        flat_spread = {}
        for st in STRUCTS:
            sel = np.array([rec_info[r][2] == st for r in range(len(rec_info))]) & (MV30 < 0.0005)
            sel &= ~np.isnan(R30)
            if sel.sum() >= 20:
                spread_frac = np.median([2 * args.slip * rec_info[r][6] / rec_info[r][4] for r in np.flatnonzero(sel)])
                flat_spread[st] = (float(np.median(R30[sel])), float(spread_frac), int(sel.sum()))
        L += ["\n## What a trade costs you (real prices)\n",
              f"- Commission: $0 per contract (Wealthsimple ETF options). FX: {args.fx:.1%} each way "
              "(0 with a USD account).",
              f"- Spread / slippage: ${args.slip:.2f} per leg on the way in AND on the way out, on top of the traded price.",
              "- Time decay is already inside the real option prices; the table estimates its size from trades where "
              "QQQ/SPY went nowhere for 30 minutes.\n",
              "| how bought | flat-price trades | lost in 30 min if price doesn't move (spread + decay) | of which spread "
              "| of which time decay | decay per 10 min held |", "|---|---|---|---|---|---|"]
        decay_rate = {}
        for st, (r30, spf, nn) in flat_spread.items():
            decay = max(0.0, -r30 - spf)
            decay_rate[st] = decay / 30
            L.append(f"| {st} | {nn} | {-r30:.1%} | {spf:.1%} | {decay:.1%} | {decay / 3:.1%} |")
        log_rows = []
        for d in days:
            m_in, x, h, setup, rr, col = picks[d]
            sym, direction, st, legs, debit, k_n, nlegs = rec_info[rr]
            pi, hold = PH[col]
            spread_cost = k_n * 100 * 2 * args.slip * nlegs
            decay_cost = k_n * 100 * debit * decay_rate.get(st, 0.0) * h
            t_in = (datetime(2000, 1, 1, 9, 30) + timedelta(minutes=int(m_in))).strftime("%H:%M")
            t_out = (datetime(2000, 1, 1, 9, 30) + timedelta(minutes=int(m_in + h))).strftime("%H:%M")
            opt = " / ".join(f"{'-' if sh else '+'}{int(k)}{r}" for r, k, sh in legs) + f" ({st})"
            log_rows.append((d, sym, setup, direction, t_in, opt, k_n, debit, t_out, float(AVG[rr, col]),
                             why_rows[rr][col], x, spread_cost, decay_cost, "check" if d >= split else "tune",
                             db.plan_label(PLANS[pi])))
        import csv
        with open(ROOT / "reports" / "morning_trades.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["day", "symbol", "setup", "side", "entry", "option", "contracts", "paid", "exit", "sold",
                        "how it ended", "pnl", "spread_cost", "decay_cost_est", "period", "plan"])
            for r in log_rows:
                w.writerow([r[0], r[1], r[2], r[3], r[4], r[5], r[6], f"{r[7]:.2f}", r[8], f"{r[9]:.2f}", r[10],
                            f"{r[11]:.2f}", f"{r[12]:.2f}", f"{r[13]:.2f}", r[14], r[15]])
        chk = [r for r in log_rows if r[14] == "check"]
        if chk:
            L += ["", f"Unseen year, per trade on average: spread/slippage ${np.mean([r[12] for r in chk]):,.0f}, "
                  f"time decay (est.) ${np.mean([r[13] for r in chk]):,.0f}, result ${np.mean([r[11] for r in chk]):+,.0f}."]
            L += ["\n## Every trade on the unseen year (one per day, exactly as it would have played out)\n",
                  "| day | symbol | setup | side | in | option (+ bought, - sold) | contracts | paid | out | sold at "
                  "| how it ended | $ result | spread cost | decay (est.) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for r in chk:
                L.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | {r[7]:.2f} | {r[8]} | {r[9]:.2f} "
                         f"| {r[10]} | ${r[11]:+,.0f} | ${r[12]:,.0f} | ${r[13]:,.0f} |")
            L.append("\nAll trades (both periods) are in reports/morning_trades.csv.")
        L += ["\n## Bottom line\n",
              f"- Combined, tune: {ct['wk']:.2f} trades/week, win {ct['win']:.0%}, ${ct['per']:+,.0f}/trade, "
              f"**${ct['usd_wk']:+,.0f}/week**.",
              f"- Combined, unseen check year: {cc['n']} trades ({cc['wk']:.2f}/week), win {cc['win']:.0%}, "
              f"${cc['per']:+,.0f}/trade (median ${cc['med']:+,.0f}), **${cc['usd_wk']:+,.0f}/week**, total "
              f"${cc['total']:+,.0f} (${cc['ex5']:+,.0f} without its best 5 days), worst drawdown ${cc['dd']:,.0f}.",
              f"- Settings profitable on BOTH tune and check: {sum(1 for r in real if r[1]['usd_wk'] > 0 and r[2]['usd_wk'] > 0)}"
              f" of {len(real)} (coin control: {sum(1 for r in coin if r[1]['usd_wk'] > 0 and r[2]['usd_wk'] > 0)} of {len(coin)})."]
    else:
        L.append("No entry setup was profitable on tune with these plans. Nothing to combine.")
    out = ROOT / "reports" / f"morning_{datetime.now():%Y-%m-%d}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[-5:]))
    print(f"\nreport written: {out}\nNext: git add reports; git commit -m \"morning study\"; git pull --rebase; git push")


if __name__ == "__main__":
    main()
