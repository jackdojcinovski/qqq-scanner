"""Gates, candidate setups and the 100-point score.

Two steps, shared by the live scanner and the backtest:

1. `candidates(history, cfg)` finds every possible setup at the last bar:
   each direction, each recent trigger (liquidity sweep or opening range
   break) and each entry style. It records which components were present
   but applies no preferences.
2. `select(cands, cfg)` applies the configurable preferences (directions,
   trend gate, required components, entry style...), scores what's left with
   the config weights and returns the best.

`evaluate()` = time/news/limit gates + both steps, for one moment in time.
Because preferences and weights are applied in step 2, the backtest can try
many variants on the same cached candidates.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from . import indicators as ind
from . import structure as st
from .config import parse_time
from .levels import Level, build_levels
from .news import NewsStatus

ENTRY_MODES = ("mid", "near_edge")


@dataclass
class Component:
    name: str
    key: str              # weights key used when present
    max_key: str          # weights key for the maximum (differs for partial credit)
    hit: bool
    detail: str
    points: int = 0
    max_points: int = 0


@dataclass
class Setup:
    key: str                  # stable id so the same setup isn't alerted twice
    direction: str            # "long" / "short"
    entry: float
    stop: float
    target: float
    target_name: str
    rr: float
    entry_zone: str
    time: pd.Timestamp
    trigger: str = "sweep"    # "sweep" or "orb"
    has_mss: bool = False
    trend_ok: bool = False
    entry_mode: str = "mid"
    score: int = 0
    components: list[Component] = field(default_factory=list)

    @property
    def risk(self) -> float:
        return abs(self.entry - self.stop)

    def band(self, cfg: dict[str, Any]) -> str:
        if self.score >= cfg["thresholds"]["alert_score"]:
            return "alert"
        if self.score >= cfg["thresholds"]["log_score"]:
            return "log"
        return "ignore"

    def hit(self, name: str) -> bool:
        return any(c.name == name and c.hit for c in self.components)


@dataclass
class Evaluation:
    setup: Setup | None
    reason: str = ""          # why there's no setup


@dataclass
class DayState:
    alerts_today: int = 0
    losses_today: int = 0


# ---------------------------------------------------------------------- public API

def evaluate(history: pd.DataFrame, cfg: dict[str, Any],
             news: NewsStatus | None = None, day: DayState | None = None) -> Evaluation:
    day = day or DayState()
    news = news or NewsStatus(False)
    reason = gate_reason(history.index[-1], cfg, news, day)
    if reason:
        return Evaluation(None, reason)
    cands, rejects = candidates(history, cfg, (cfg["trade"].get("entry_at", "mid"),))
    best, why = select(cands, cfg)
    if best is None:
        reasons = sorted(rejects | why) or ["no sweep or opening range break"]
        return Evaluation(None, " / ".join(reasons))
    return Evaluation(best)


def gate_reason(now: pd.Timestamp, cfg: dict[str, Any], news: NewsStatus, day: DayState) -> str:
    w = cfg["window"]
    if not (parse_time(w["start"]) <= now.time() <= parse_time(w["end"])):
        return f"outside window {w['start']}-{w['end']}"
    if news.blocked:
        return f"news: {news.reason}"
    lim = cfg["limits"]
    if day.alerts_today >= lim["max_alerts_per_day"]:
        return "daily alert limit reached"
    if day.losses_today >= lim["stop_after_losses"]:
        return "stopped for the day after a loss"
    return ""


def candidates(history: pd.DataFrame, cfg: dict[str, Any],
               entry_modes=ENTRY_MODES) -> tuple[list[Setup], set[str]]:
    """Every structurally valid setup at the last bar, unscored."""
    now = history.index[-1]
    rth = ind.regular_session(history, parse_time(cfg["session"]["open"]),
                              parse_time(cfg["session"]["close"]))
    session = rth[rth.index.date == now.date()]
    if len(session) < cfg["session"]["opening_range_minutes"]:
        return [], {"not enough session bars"}
    ctx = _Context.build(history, session, cfg)
    out: list[Setup] = []
    for direction in ("long", "short"):
        out += _direction_candidates(ctx, direction, entry_modes)
    return out, ctx.rejects


def select(cands: list[Setup], cfg: dict[str, Any]) -> tuple[Setup | None, set[str]]:
    """Apply the config's preferences, score with its weights, return the best."""
    t = cfg["trade"]
    sides = t.get("directions", ["long", "short"])
    mode = t.get("entry_at", "mid")
    triggers = t.get("triggers", ["sweep", "orb"])
    required = t.get("required_components", [])
    why: set[str] = set()
    best: Setup | None = None
    for c in cands:
        if c.entry_mode != mode:
            continue
        if c.direction not in sides:
            why.add("direction not traded")
            continue
        if c.trigger not in triggers:
            why.add(f"{c.trigger} trigger not traded")
            continue
        if c.trigger == "sweep" and not c.has_mss and t.get("require_mss_after_sweep", True):
            why.add("sweep but no structure shift")
            continue
        if t.get("require_trend", False) and not c.trend_ok:
            why.add("against the 5-min trend")
            continue
        missing = [n for n in required if not c.hit(n)]
        if missing:
            why.add(f"missing {missing[0]}")
            continue
        if c.rr < t["min_rr"]:
            why.add("reward to risk below minimum")
            continue
        score = sum(cfg["weights"][x.key] for x in c.components if x.hit)
        if best is None or score > best.score:
            best = _scored(c, cfg)
    return best, why


def _scored(c: Setup, cfg: dict[str, Any]) -> Setup:
    s = copy.copy(c)
    W = cfg["weights"]
    s.components = [copy.copy(x) for x in c.components]
    for x in s.components:
        x.points = W[x.key] if x.hit else 0
        x.max_points = W[x.max_key]
    s.score = sum(x.points for x in s.components)
    return s


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


def _direction_candidates(ctx: _Context, direction: str, entry_modes) -> list[Setup]:
    cfg, s = ctx.cfg, ctx.cfg["structure"]
    bars = ctx.session
    last = len(bars) - 1
    close = float(bars["close"].iloc[-1])
    long = direction == "long"
    sign = 1 if long else -1

    sweeps = [x for x in ctx.sweeps if x.direction == direction
              and last - x.idx <= s["trigger_lookback_bars"]]

    orb_idx = None
    orb_now = False
    if ctx.orange:
        edge = ctx.orange[0] if long else ctx.orange[1]
        closes = bars["close"].to_numpy()
        or_end = bars.index[0] + pd.Timedelta(minutes=cfg["session"]["opening_range_minutes"])
        beyond = [j for j in range(len(bars)) if sign * (closes[j] - edge) > 0 and bars.index[j] >= or_end]
        orb_now = sign * (close - edge) > 0
        if beyond and orb_now:
            orb_idx = beyond[0]
    orb_recent = orb_idx is not None and last - orb_idx <= s["orb_break_lookback_bars"]

    triggers: list[st.Sweep | None] = list(sweeps) + ([None] if orb_recent else [])
    if not triggers:
        ctx.rejects.add("no sweep or opening range break")
    out: list[Setup] = []
    for sweep in triggers:
        out += _trigger_candidates(ctx, direction, sweep, orb_idx, orb_now, entry_modes)
    return out


def _trigger_candidates(ctx: _Context, direction: str, sweep: st.Sweep | None,
                        orb_idx: int | None, orb_now: bool, entry_modes) -> list[Setup]:
    cfg, s, T = ctx.cfg, ctx.cfg["structure"], ctx.cfg["thresholds"]
    bars = ctx.session
    close = float(bars["close"].iloc[-1])
    long = direction == "long"
    sign = 1 if long else -1
    comps: list[Component] = []

    # ------------------------------------------------ ICT structure
    mss = None
    if sweep:
        comps.append(Component("Liquidity sweep", "sweep", "sweep", True,
                               f"swept {sweep.level_name} {sweep.level:.2f}, closed back inside"))
        mss = st.find_mss(bars, sweep, ctx.swings, ctx.atr, s["mss_max_bars"], s["mss_body_atr_mult"])
    else:
        comps.append(Component("Liquidity sweep", "sweep", "sweep", False, "no sweep"))
    comps.append(Component("Structure shift", "mss", "mss", mss is not None,
                           f"closed {'above' if long else 'below'} {mss.broken_level:.2f}, "
                           f"displacement {mss.body_atr:.1f}x ATR" if mss
                           else "no displacement through structure"))

    anchor = sweep.idx if sweep else orb_idx
    zones = [g for g in ctx.gaps if g.direction == direction and g.active_from >= anchor
             and (g.bottom <= close if long else g.top >= close)]
    if not zones:
        # entry, stop and zone description all hang off the zone, so it's required
        ctx.rejects.add("trigger but no FVG to enter from")
        return []
    zones.sort(key=lambda g: (g.kind == "IFVG", g.active_from))
    zone = zones[-1]
    if zone.kind == "IFVG":
        comps.append(Component("Entry zone", "fvg_ifvg", "fvg_ifvg", True,
                               f"inverse FVG {zone.bottom:.2f}-{zone.top:.2f}"))
    else:
        comps.append(Component("Entry zone", "fvg_plain", "fvg_ifvg", True,
                               f"FVG {zone.bottom:.2f}-{zone.top:.2f}"))

    ob = st.find_order_block(bars, sweep, mss) if (sweep and mss) else None
    ob_hit = bool(ob and st.overlaps(ob.top, ob.bottom, zone.top, zone.bottom))
    comps.append(Component("Order block", "order_block", "order_block", ob_hit,
                           f"order block {ob.bottom:.2f}-{ob.top:.2f} overlaps entry" if ob_hit
                           else "no overlapping order block"))

    # ------------------------------------------------ indicators
    vwap = float(ctx.vwap.iloc[-1])
    comps.append(Component("VWAP side", "vwap_side", "vwap_side", sign * (close - vwap) > 0,
                           f"{'above' if close > vwap else 'below'} VWAP {vwap:.2f}"))
    comps.append(Component("Opening range break", "orb_break", "orb_break", orb_now,
                           f"beyond opening range {'high' if long else 'low'}" if orb_now
                           else "inside opening range"))

    have_ema = ctx.ema_fast is not None and ctx.ema_slow is not None
    trend_ok = have_ema and sign * (ctx.ema_fast - ctx.ema_slow) > 0
    comps.append(Component("EMA trend", "ema_trend", "ema_trend", trend_ok,
                           f"5-min EMA {cfg['indicators']['ema_fast']} "
                           f"{'>' if ctx.ema_fast > ctx.ema_slow else '<'} "
                           f"EMA {cfg['indicators']['ema_slow']}" if have_ema else "not enough 5-min bars"))

    trig = mss.idx if mss else (sweep.reclaim_idx if sweep else orb_idx)
    rv = ctx.relvol.iloc[trig]
    comps.append(Component("Volume", "rel_volume", "rel_volume",
                           bool(pd.notna(rv) and rv >= T["rel_volume_min"]),
                           f"trigger volume {rv:.1f}x average" if pd.notna(rv) else "volume n/a"))

    if ctx.rsi5 is not None:
        rsi_ok = ctx.rsi5 < T["rsi_long_max"] if long else ctx.rsi5 > T["rsi_short_min"]
        comps.append(Component("RSI", "rsi_ok", "rsi_ok", rsi_ok,
                               f"5-min RSI {ctx.rsi5:.0f}" + ("" if rsi_ok else " (stretched)")))
    else:
        comps.append(Component("RSI", "rsi_ok", "rsi_ok", False, "RSI n/a"))

    now_t = bars.index[-1].time()
    w = cfg["window"]
    prime = parse_time(w["prime_start"]) <= now_t <= parse_time(w["prime_end"])
    late = now_t > parse_time(w["prime_end"])
    timing = Component("Timing", "timing_prime" if prime or not late else "timing_late", "timing_prime",
                       prime or late, f"{bars.index[-1]:%H:%M} ET")

    # ------------------------------------------------ trade plans, one per entry style
    t = cfg["trade"]
    buf = t["stop_buffer_pct"]
    anchor_px = sweep.extreme if sweep else (ctx.orange[1] if long else ctx.orange[0])
    far_edge = zone.bottom if long else zone.top
    anchor_px = min(anchor_px, far_edge) if long else max(anchor_px, far_edge)
    stop = anchor_px * (1 - buf) if long else anchor_px * (1 + buf)
    trigger_idx = sweep.idx if sweep else orb_idx
    key = f"{bars.index[-1]:%Y-%m-%d}-{direction}-{bars.index[trigger_idx]:%H%M}"

    out = []
    for mode in entry_modes:
        entry = (zone.top if long else zone.bottom) if mode == "near_edge" else zone.mid
        entry = min(entry, close) if long else max(entry, close)
        risk = sign * (entry - stop)
        if risk <= 0 or sign * (close - stop) <= 0:
            ctx.rejects.add("price already through the stop")
            continue
        beyond = sorted((l for l in ctx.levels if sign * (l.price - entry) >= t["min_rr"] * risk),
                        key=lambda l: abs(l.price - entry))
        if beyond:
            target, target_name = beyond[0].price, beyond[0].name
        else:
            target, target_name = entry + sign * t["default_rr"] * risk, f"{t['default_rr']:g}R"
        rr = abs(target - entry) / risk
        rr_c = Component("Reward to risk", "rr", "rr", rr >= T["rr_points_min"], f"R:R {rr:.1f}")
        out.append(Setup(
            key=key, direction=direction,
            entry=round(entry, 2), stop=round(stop, 2), target=round(target, 2),
            target_name=target_name, rr=round(rr, 2),
            entry_zone=f"{zone.kind} {'near edge' if mode == 'near_edge' else 'midpoint'}",
            time=bars.index[-1], trigger="sweep" if sweep else "orb", has_mss=mss is not None,
            trend_ok=trend_ok, entry_mode=mode,
            components=[copy.copy(c) for c in comps] + [rr_c, copy.copy(timing)],
        ))
    return out
