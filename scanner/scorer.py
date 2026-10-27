"""Gates and the 100-point setup score.

`evaluate(history, cfg, ...)` looks at the market as of the last bar in
`history` (1-minute bars from at least the prior session through now) and
returns the best long or short setup, or the reason there isn't one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from typing import Any

import pandas as pd

from . import indicators as ind
from . import structure as st
from .config import parse_time
from .levels import Level, build_levels
from .news import NewsStatus


@dataclass
class Component:
    name: str
    points: int
    max_points: int
    detail: str

    @property
    def hit(self) -> bool:
        return self.points > 0


@dataclass
class Setup:
    key: str                  # stable id so the same setup isn't alerted twice
    direction: str            # "long" / "short"
    score: int
    entry: float
    stop: float
    target: float
    target_name: str
    rr: float
    entry_zone: str
    time: pd.Timestamp
    components: list[Component] = field(default_factory=list)
    context: list[str] = field(default_factory=list)

    @property
    def risk(self) -> float:
        return abs(self.entry - self.stop)

    def band(self, cfg: dict[str, Any]) -> str:
        if self.score >= cfg["thresholds"]["alert_score"]:
            return "alert"
        if self.score >= cfg["thresholds"]["log_score"]:
            return "log"
        return "ignore"


@dataclass
class Evaluation:
    setup: Setup | None
    reason: str = ""          # why there's no setup (a failed gate)


@dataclass
class DayState:
    alerts_today: int = 0
    losses_today: int = 0


def evaluate(history: pd.DataFrame, cfg: dict[str, Any],
             news: NewsStatus | None = None, day: DayState | None = None) -> Evaluation:
    day = day or DayState()
    news = news or NewsStatus(False)
    now = history.index[-1]
    w = cfg["window"]
    start, end = parse_time(w["start"]), parse_time(w["end"])

    # ---------------------------------------------------------------- gates
    if not (start <= now.time() <= end):
        return Evaluation(None, f"outside window {w['start']}-{w['end']}")
    if news.blocked:
        return Evaluation(None, f"news: {news.reason}")
    lim = cfg["limits"]
    if day.alerts_today >= lim["max_alerts_per_day"]:
        return Evaluation(None, "daily alert limit reached")
    if day.losses_today >= lim["stop_after_losses"]:
        return Evaluation(None, "stopped for the day after a loss")

    rth = ind.regular_session(history, parse_time(cfg["session"]["open"]),
                              parse_time(cfg["session"]["close"]))
    session = rth[rth.index.date == now.date()]
    if len(session) < cfg["session"]["opening_range_minutes"]:
        return Evaluation(None, "not enough session bars")

    ctx = _Context.build(history, session, cfg)
    candidates = [s for s in (_score_direction(ctx, d) for d in ("long", "short")) if s]
    if not candidates:
        why = " / ".join(sorted(ctx.rejects)) or "no sweep or opening range break"
        return Evaluation(None, why)
    valid = [s for s in candidates if s.rr >= cfg["trade"]["min_rr"]]
    if not valid:
        return Evaluation(None, "reward to risk below minimum")
    return Evaluation(max(valid, key=lambda s: s.score))


# ---------------------------------------------------------------------- internals

@dataclass
class _Context:
    cfg: dict[str, Any]
    session: pd.DataFrame
    levels: list[Level]
    vwap: pd.Series
    atr: pd.Series
    relvol: pd.Series
    ema_fast: float | None
    ema_slow: float | None
    rsi5: float | None
    orange: tuple[float, float] | None
    swings: list[st.Swing]
    sweeps: list[st.Sweep]
    gaps: list[st.Gap]
    rejects: set[str] = field(default_factory=set)

    @classmethod
    def build(cls, history: pd.DataFrame, session: pd.DataFrame, cfg: dict[str, Any]) -> "_Context":
        i, s = cfg["indicators"], cfg["structure"]
        levels = build_levels(history, cfg["session"]["opening_range_minutes"],
                              parse_time(cfg["session"]["premarket_start"]),
                              parse_time(cfg["session"]["open"]))
        rth_hist = ind.regular_session(history)
        five = ind.resample_5m(rth_hist)
        ef = ind.ema(five["close"], i["ema_fast"]) if len(five) else None
        es = ind.ema(five["close"], i["ema_slow"]) if len(five) else None
        r5 = ind.rsi(five["close"], i["rsi_period"]) if len(five) else None
        atr = ind.atr(session, i["atr_period"])
        swings = st.swing_points(session, s["swing_strength"])
        levels = levels + st.equal_levels(session, swings, s["equal_level_pct"])
        return cls(
            cfg=cfg,
            session=session,
            levels=levels,
            vwap=ind.session_vwap(session),
            atr=atr,
            relvol=ind.relative_volume(rth_hist, i["rel_volume_period"]).reindex(session.index),
            ema_fast=_last(ef),
            ema_slow=_last(es),
            rsi5=_last(r5),
            orange=ind.opening_range(session, cfg["session"]["opening_range_minutes"]),
            swings=swings,
            sweeps=st.find_sweeps(session, levels, s["sweep_pierce_pct"], s["sweep_reclaim_bars"]),
            gaps=st.find_gaps(session, atr, s["fvg_min_atr_mult"], s["fvg_expiry_bars"]),
        )


def _last(series: pd.Series | None) -> float | None:
    if series is None or series.empty or pd.isna(series.iloc[-1]):
        return None
    return float(series.iloc[-1])


def _score_direction(ctx: _Context, direction: str) -> Setup | None:
    cfg, s = ctx.cfg, ctx.cfg["structure"]
    W, T = cfg["weights"], cfg["thresholds"]
    bars = ctx.session
    last = len(bars) - 1
    close = float(bars["close"].iloc[-1])
    long = direction == "long"
    sign = 1 if long else -1

    # ------------------------------------------------ trigger: recent sweep or ORB
    sweeps = [x for x in ctx.sweeps if x.direction == direction
              and last - x.idx <= s["trigger_lookback_bars"]]

    orb_idx = None
    orb_now = False
    if ctx.orange:
        edge = ctx.orange[0] if long else ctx.orange[1]
        closes = bars["close"].to_numpy()
        beyond = [j for j in range(len(bars)) if sign * (closes[j] - edge) > 0
                  and bars.index[j] >= bars.index[0] + pd.Timedelta(minutes=cfg["session"]["opening_range_minutes"])]
        orb_now = sign * (close - edge) > 0
        if beyond and orb_now:
            orb_idx = beyond[0]
    orb_recent = orb_idx is not None and last - orb_idx <= s["orb_break_lookback_bars"]

    # score each recent sweep (and the ORB alone) and keep the best valid one
    triggers: list[st.Sweep | None] = list(sweeps) + ([None] if orb_recent else [])
    if not triggers:
        ctx.rejects.add("no sweep or opening range break")
    best = None
    for sweep in triggers:
        cand = _score_trigger(ctx, direction, sweep, orb_idx, orb_now)
        if cand and (best is None or cand.score > best.score):
            best = cand
    return best


def _score_trigger(ctx: _Context, direction: str, sweep: st.Sweep | None,
                   orb_idx: int | None, orb_now: bool) -> Setup | None:
    cfg, s = ctx.cfg, ctx.cfg["structure"]
    W, T = cfg["weights"], cfg["thresholds"]
    bars = ctx.session
    close = float(bars["close"].iloc[-1])
    long = direction == "long"
    sign = 1 if long else -1
    comps: list[Component] = []

    # ------------------------------------------------ ICT structure
    mss = None
    if sweep:
        comps.append(Component("Liquidity sweep", W["sweep"], W["sweep"],
                               f"swept {sweep.level_name} {sweep.level:.2f}, closed back inside"))
        mss = st.find_mss(bars, sweep, ctx.swings, ctx.atr, s["mss_max_bars"], s["mss_body_atr_mult"])
    else:
        comps.append(Component("Liquidity sweep", 0, W["sweep"], "no sweep"))
    if mss:
        comps.append(Component("Structure shift", W["mss"], W["mss"],
                               f"closed {'above' if long else 'below'} {mss.broken_level:.2f}, "
                               f"displacement {mss.body_atr:.1f}x ATR"))
    else:
        if sweep and cfg["trade"].get("require_mss_after_sweep", True):
            ctx.rejects.add("sweep but no structure shift")
            return None
        comps.append(Component("Structure shift", 0, W["mss"], "no displacement through structure"))

    anchor = sweep.idx if sweep else orb_idx
    zones = [g for g in ctx.gaps if g.direction == direction and g.active_from >= anchor
             and (g.bottom <= close if long else g.top >= close)]
    zone = None
    if zones:
        zones.sort(key=lambda g: (g.kind == "IFVG", g.active_from))
        zone = zones[-1]
    if zone and zone.kind == "IFVG":
        comps.append(Component("Entry zone", W["fvg_ifvg"], W["fvg_ifvg"],
                               f"inverse FVG {zone.bottom:.2f}-{zone.top:.2f}"))
    elif zone:
        comps.append(Component("Entry zone", W["fvg_plain"], W["fvg_ifvg"],
                               f"FVG {zone.bottom:.2f}-{zone.top:.2f}"))
    else:
        if cfg["trade"].get("require_entry_zone", True):
            ctx.rejects.add("trigger but no FVG to enter from")
            return None
        comps.append(Component("Entry zone", 0, W["fvg_ifvg"], "no FVG to enter from"))

    ob = st.find_order_block(bars, sweep, mss) if (sweep and mss) else None
    if ob and zone and st.overlaps(ob.top, ob.bottom, zone.top, zone.bottom):
        comps.append(Component("Order block", W["order_block"], W["order_block"],
                               f"order block {ob.bottom:.2f}-{ob.top:.2f} overlaps entry"))
    else:
        comps.append(Component("Order block", 0, W["order_block"], "no overlapping order block"))

    # ------------------------------------------------ indicators
    vwap = float(ctx.vwap.iloc[-1])
    side_ok = sign * (close - vwap) > 0
    comps.append(Component("VWAP side", W["vwap_side"] if side_ok else 0, W["vwap_side"],
                           f"{'above' if close > vwap else 'below'} VWAP {vwap:.2f}"))

    comps.append(Component("Opening range break", W["orb_break"] if orb_now else 0, W["orb_break"],
                           f"beyond opening range {'high' if long else 'low'}" if orb_now
                           else "inside opening range"))

    if ctx.ema_fast is not None and ctx.ema_slow is not None:
        trend_ok = sign * (ctx.ema_fast - ctx.ema_slow) > 0
        comps.append(Component("EMA trend", W["ema_trend"] if trend_ok else 0, W["ema_trend"],
                               f"5-min EMA {cfg['indicators']['ema_fast']} "
                               f"{'>' if ctx.ema_fast > ctx.ema_slow else '<'} "
                               f"EMA {cfg['indicators']['ema_slow']}"))
    else:
        comps.append(Component("EMA trend", 0, W["ema_trend"], "not enough 5-min bars"))

    trig = mss.idx if mss else (sweep.reclaim_idx if sweep else orb_idx)
    rv = ctx.relvol.iloc[trig]
    rv_ok = pd.notna(rv) and rv >= T["rel_volume_min"]
    comps.append(Component("Volume", W["rel_volume"] if rv_ok else 0, W["rel_volume"],
                           f"trigger volume {rv:.1f}x average" if pd.notna(rv) else "volume n/a"))

    if ctx.rsi5 is not None:
        rsi_ok = ctx.rsi5 < T["rsi_long_max"] if long else ctx.rsi5 > T["rsi_short_min"]
        comps.append(Component("RSI", W["rsi_ok"] if rsi_ok else 0, W["rsi_ok"],
                               f"5-min RSI {ctx.rsi5:.0f}" + ("" if rsi_ok else " (stretched)")))
    else:
        comps.append(Component("RSI", 0, W["rsi_ok"], "RSI n/a"))

    # ------------------------------------------------ trade plan
    t = cfg["trade"]
    buf = t["stop_buffer_pct"]
    entry = zone.mid if zone else close
    entry = min(entry, close) if long else max(entry, close)
    # stop goes beyond the sweep wick (or opening range) AND the entry zone's far edge,
    # so a normal fill into the zone can't stop the trade out
    if sweep:
        anchor_px = sweep.extreme
    else:
        anchor_px = ctx.orange[1] if long else ctx.orange[0]
    far_edge = zone.bottom if long else zone.top
    anchor_px = min(anchor_px, far_edge) if long else max(anchor_px, far_edge)
    stop = anchor_px * (1 - buf) if long else anchor_px * (1 + buf)
    risk = sign * (entry - stop)
    if risk <= 0 or sign * (close - stop) <= 0:
        ctx.rejects.add("price already through the stop")
        return None

    beyond = sorted((l for l in ctx.levels if sign * (l.price - entry) >= t["min_rr"] * risk),
                    key=lambda l: abs(l.price - entry))
    if beyond:
        target, target_name = beyond[0].price, beyond[0].name
    else:
        target, target_name = entry + sign * t["default_rr"] * risk, f"{t['default_rr']:g}R"
    rr = abs(target - entry) / risk
    comps.append(Component("Reward to risk", W["rr"] if rr >= T["rr_points_min"] else 0, W["rr"],
                           f"R:R {rr:.1f}"))

    prime = parse_time(cfg["window"]["prime_start"]) <= bars.index[-1].time() <= parse_time(cfg["window"]["prime_end"])
    late = bars.index[-1].time() > parse_time(cfg["window"]["prime_end"])
    tp = W["timing_prime"] if prime else (W["timing_late"] if late else 0)
    comps.append(Component("Timing", tp, W["timing_prime"], f"{bars.index[-1]:%H:%M} ET"))

    trigger_idx = sweep.idx if sweep else orb_idx
    key = f"{bars.index[-1]:%Y-%m-%d}-{direction}-{bars.index[trigger_idx]:%H%M}"
    zone_desc = (f"{zone.kind} midpoint" if zone else "market")
    return Setup(
        key=key, direction=direction, score=sum(c.points for c in comps),
        entry=round(entry, 2), stop=round(stop, 2), target=round(target, 2),
        target_name=target_name, rr=round(rr, 2), entry_zone=zone_desc,
        time=bars.index[-1], components=comps,
    )
