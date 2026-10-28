"""Common day-trading setups for the 9:30-11:00 window, checked on every closed 1-minute candle.

Indicators (the ones every day trader watches):
  VWAP, 9 EMA and 20 EMA (the 9 "curling" up/down), relative volume (today vs a normal day at the
  same minute), volume spike on the trigger candle, the opening range (first 15 minutes), yesterday's
  high / low, and where price is vs today's open.

Triggers (long side; shorts are the mirror image):
  orb_break    first close above the 15-minute opening-range high
  vwap_reclaim close back above VWAP after being below it, on a volume spike
  vwap_bounce  uptrend (mostly above VWAP), pullback touches VWAP / 20 EMA, green candle closes back up
  hod_break    bull flag: opening push, a pullback, then a new high of the day
  pdh_break    first close above yesterday's high
  ema_cross    9 EMA crosses above the 20 EMA while price is above VWAP

Confirmations (each one counts 1):
  right side of VWAP | 9 EMA above 20 EMA and curling up | relative volume >= 1.2x |
  trigger candle volume >= 1.5x the last 10 candles | price above today's open
A setup only counts when price is on the right side of VWAP AND at least 4 of the 5 confirmations agree.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

NAMES = {
    "orb_break": "Opening-range breakout", "vwap_reclaim": "VWAP reclaim", "vwap_bounce": "VWAP pullback bounce",
    "hod_break": "Bull/bear flag breakout", "pdh_break": "Yesterday's high/low break", "ema_cross": "9/20 EMA cross",
}
MIN_CONFIRM = 4


@dataclass
class Signal:
    setup: str
    direction: str            # long / short
    bar: int                  # index of the closed candle that triggered (0 = 9:30)
    price: float
    confirms: dict = field(default_factory=dict)
    facts: dict = field(default_factory=dict)

    @property
    def score(self) -> int:
        return sum(self.confirms.values())


def ema(x: np.ndarray, n: int) -> np.ndarray:
    a = 2 / (n + 1)
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def evaluate(O, H, L, C, V, ctx: dict) -> list[Signal]:
    """All setups triggering on the LAST closed candle. Arrays hold today's candles from 9:30 up to now."""
    m = len(C) - 1
    if m < 4:
        return []
    typ = (H + L + C) / 3
    cv = np.cumsum(V)
    vwap = np.cumsum(typ * V) / np.where(cv > 0, cv, 1)
    e9, e20 = ema(C, 9), ema(C, 20)
    o = O[0]
    avg_cum = ctx.get("avg_cum")
    rvol = float(cv[m] / avg_cum[m]) if avg_cum is not None and m < len(avg_cum) and avg_cum[m] > 0 else float("nan")
    spike = float(V[m] / max(V[max(0, m - 10):m].mean(), 1)) if m >= 3 else float("nan")
    out = []
    for s in (1, -1):
        d = "long" if s > 0 else "short"
        side = s * (C[m] - vwap[m]) > 0
        if not side:
            continue
        conf = {
            "vwap": side,
            "ema": bool(s * (e9[m] - e20[m]) > 0 and s * (e9[m] - e9[max(0, m - 3)]) > 0),
            "rvol": bool(rvol == rvol and rvol >= 1.2),
            "spike": bool(spike == spike and spike >= 1.5),
            "open": bool(s * (C[m] - o) > 0),
        }
        if sum(conf.values()) < MIN_CONFIRM:
            continue
        trig = []
        if m >= 15:
            orh, orl = H[:15].max(), L[:15].min()
            lvl = orh if s > 0 else orl
            if s * (C[m] - lvl) > 0 and s * (C[m - 1] - lvl) <= 0:
                trig.append(("orb_break", {"level": lvl}))
        if s * (C[m - 1] - vwap[m - 1]) <= 0 and conf["spike"]:
            trig.append(("vwap_reclaim", {"vwap": vwap[m]}))
        if m >= 15:
            trend = np.mean(s * (C[m - 15:m] - vwap[m - 15:m]) > 0) >= 0.8
            touch = any((L[k] <= max(vwap[k], e20[k]) * 1.0005) if s > 0 else (H[k] >= min(vwap[k], e20[k]) * 0.9995)
                        for k in range(m - 2, m + 1))
            if trend and touch and s * (C[m] - O[m]) > 0:
                trig.append(("vwap_bounce", {"vwap": vwap[m]}))
        if m >= 10:
            prior = H[:m].max() if s > 0 else L[:m].min()
            move = s * (prior - o)
            pull = (prior - L[:m][np.argmax(H[:m]):].min()) if s > 0 else (H[:m][np.argmin(L[:m]):].max() - prior)
            if move > 0 and pull >= 0.3 * move and s * (C[m] - prior) > 0:
                trig.append(("hod_break", {"level": prior}))
        lvl = ctx.get("prev_high") if s > 0 else ctx.get("prev_low")
        if lvl and s * (C[m] - lvl) > 0 and s * (C[m - 1] - lvl) <= 0:
            trig.append(("pdh_break", {"level": lvl}))
        if m >= 20 and s * (e9[m] - e20[m]) > 0 and s * (e9[m - 1] - e20[m - 1]) <= 0:
            trig.append(("ema_cross", {}))
        for name, extra in trig:
            facts = {"vwap": float(vwap[m]), "ema9": float(e9[m]), "ema20": float(e20[m]), "rvol": rvol,
                     "spike": spike, "open": float(o), "hod": float(H[:m + 1].max()), "lod": float(L[:m + 1].min()),
                     **{k: float(v) for k, v in extra.items()}}
            out.append(Signal(name, d, m, float(C[m]), conf, facts))
    return sorted(out, key=lambda x: -x.score)


def explain(sig: Signal, ctx: dict) -> str:
    f, s = sig.facts, sig.direction == "long"
    up = "above" if s else "below"
    lines = {
        "orb_break": f"Broke {'above the high' if s else 'below the low'} of the first 15 minutes "
                     f"({f.get('level', 0):.2f}) - the opening range just gave way.",
        "vwap_reclaim": f"Price was {'under' if s else 'over'} VWAP and just closed back {up} it "
                        f"({f['vwap']:.2f}) on a volume spike - {'buyers' if s else 'sellers'} took control back.",
        "vwap_bounce": f"Trending {up} VWAP all morning; it pulled back to VWAP / the 20 EMA and bounced with a "
                       f"{'green' if s else 'red'} candle - classic trend pullback entry.",
        "hod_break": f"Opening push, a pullback, and now a new {'high' if s else 'low'} of the day "
                     f"({f.get('level', 0):.2f}) - {'bull' if s else 'bear'} flag breakout.",
        "pdh_break": f"Just broke yesterday's {'high' if s else 'low'} ({f.get('level', 0):.2f}) - a key level gave way.",
        "ema_cross": f"9 EMA just crossed {'above' if s else 'below'} the 20 EMA with price {up} VWAP - momentum turning.",
    }[sig.setup]
    c = sig.confirms
    checks = [f"{'✅' if c['vwap'] else '❌'} price {up} VWAP ({f['vwap']:.2f})",
              f"{'✅' if c['ema'] else '❌'} 9 EMA {'above' if s else 'below'} 20 EMA and curling {'up' if s else 'down'} "
              f"({f['ema9']:.2f} vs {f['ema20']:.2f})",
              f"{'✅' if c['rvol'] else '❌'} volume {f['rvol']:.1f}x a normal morning" if f['rvol'] == f['rvol']
              else "❌ volume vs normal: unknown",
              f"{'✅' if c['spike'] else '❌'} trigger candle volume {f['spike']:.1f}x the last 10 candles",
              f"{'✅' if c['open'] else '❌'} {up} today's open ({f['open']:.2f})"]
    extra = []
    pc = ctx.get("prev_close")
    if pc:
        extra.append(f"Gap from yesterday: {f['open'] / pc - 1:+.2%}.")
    extra.append(f"Day range so far: {f['lod']:.2f} - {f['hod']:.2f}.")
    return (f"**{NAMES[sig.setup]}** ({sig.score}/5 confirmations)\n{lines}\n\n" + "\n".join(checks)
            + "\n\n" + " ".join(extra))
