"""Research engine: cached candidates, a grid of variants, costs and ranking.

Workflow (see tools/research.py):
  1. Scan every day once per symbol and cache the candidate setups, one file
     per symbol-year under data/research/<signature>/. The signature changes
     whenever a setting that affects detection changes, so stale caches are
     never reused.
  2. Run every variant over every cached chunk in parallel. Each variant is a
     set of overrides on config.yaml (entry style, trend gate, sides, trigger
     types, exit style, score gate).
  3. Rank variants on the TUNE period only, by the lower bound of a bootstrap
     confidence interval for average R after estimated option costs. That
     favours variants that are both profitable and consistent over ones that
     got lucky on a few big trades.
  4. Evaluate only the top few (plus the baseline) on the untouched CHECK period.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from . import backtest as bt
from .config import ROOT
from .scorer import ENTRY_MODES

CACHE_ROOT = ROOT / "data" / "research"


# --------------------------------------------------------------------------- config helpers

def merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def signature(cfg: dict[str, Any]) -> str:
    """Hash of every setting that changes which candidates are found."""
    t = cfg["trade"]
    parts = {
        "session": cfg["session"], "window": cfg["window"], "indicators": cfg["indicators"],
        "structure": cfg["structure"],
        "thresholds": {k: v for k, v in cfg["thresholds"].items() if k not in ("alert_score", "log_score")},
        "trade": {k: t[k] for k in ("stop_buffer_pct", "min_rr", "default_rr")},
        "version": 2,
    }
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:10]


# --------------------------------------------------------------------------- candidate cache

def chunk_path(sig: str, symbol: str, year: int, base: Path = CACHE_ROOT) -> Path:
    return base / sig / f"{symbol}_{year}.pkl"


def build_cache(bars_by_symbol: dict[str, pd.DataFrame], cfg: dict[str, Any],
                workers: int | None = None, progress=None, base: Path = CACHE_ROOT) -> list[Path]:
    """Make sure every symbol-year chunk is cached. Returns the chunk paths."""
    sig = signature(cfg)
    paths = []
    for symbol, bars in bars_by_symbol.items():
        days = sorted(set(bars.index.date))
        years = sorted({d.year for d in days})
        for y in years:
            path = chunk_path(sig, symbol, y, base)
            want = [d for d in days[1:] if d.year == y]          # first day has no prior day
            if path.exists():
                with open(path, "rb") as f:
                    have = {dc.day for dc in pickle.load(f)}
                if set(want) <= have:
                    paths.append(path)
                    continue
            if progress:
                progress(f"scanning {symbol} {y} ({len(want)} days)")
            dcs = bt.build_candidates(bars, want, cfg, symbol, ENTRY_MODES, workers)
            for dc in dcs:
                _slim(dc)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(dcs, f, protocol=pickle.HIGHEST_PROTOCOL)
            paths.append(path)
    return paths


def _slim(dc: bt.DayCandidates) -> None:
    """Drop display-only text and share identical components to keep caches small."""
    pool: dict[tuple, Any] = {}
    for _, cands in dc.minutes:
        for s in cands:
            s.target_name = ""
            comps = []
            for c in s.components:
                c.detail = ""
                k = (c.name, c.key, c.max_key, c.hit)
                comps.append(pool.setdefault(k, c))
            s.components = comps


# --------------------------------------------------------------------------- variants

@dataclass(frozen=True)
class Variant:
    name: str
    overrides: dict


DIMENSIONS: dict[str, dict[str, dict]] = {
    "entry": {
        "mid": {"trade": {"entry_at": "mid"}},
        "edge": {"trade": {"entry_at": "near_edge"}},
    },
    "trend": {
        "any": {"trade": {"require_trend": False}},
        "with": {"trade": {"require_trend": True}},
    },
    "sides": {
        "both": {"trade": {"directions": ["long", "short"]}},
        "long": {"trade": {"directions": ["long"]}},
    },
    "trigger": {
        "all": {"trade": {"triggers": ["sweep", "orb"]}},
        "orb": {"trade": {"triggers": ["orb"]}},
        "sweep": {"trade": {"triggers": ["sweep"]}},
        "classic": {"trade": {"triggers": ["orb_classic"]}},
    },
    "exit": {
        "fixed60": {"exit": {"mode": "fixed", "time_exit_minutes": 60}},
        "fixed30": {"exit": {"mode": "fixed", "time_exit_minutes": 30}},
        "partial": {"exit": {"mode": "partial", "time_exit_minutes": 60}},
        "trail": {"exit": {"mode": "trail", "time_exit_minutes": 60}},
        "fixedeod": {"exit": {"mode": "fixed", "time_exit_minutes": 400}},
        "traileod": {"exit": {"mode": "trail", "time_exit_minutes": 400}},
    },
    "gate": {
        "score75": {"thresholds": {"alert_score": 75}, "trade": {"required_components": []}},
        "checklist": {"thresholds": {"alert_score": 0, "log_score": 0},
                      "trade": {"required_components": ["VWAP side"]}},
    },
}
BASELINE = "mid|any|both|all|fixed60|score75"


def grid() -> list[Variant]:
    names = list(DIMENSIONS)
    out = []
    for combo in itertools.product(*(DIMENSIONS[n].keys() for n in names)):
        over: dict = {}
        for dim, val in zip(names, combo):
            over = merge(over, DIMENSIONS[dim][val])
        out.append(Variant("|".join(combo), over))
    return out


def variant(name: str) -> Variant:
    """Build one variant from its name, e.g. 'edge|with|long|orb|trail|checklist'."""
    vals = name.split("|")
    if len(vals) != len(DIMENSIONS):
        raise ValueError(f"variant needs {len(DIMENSIONS)} parts: {'|'.join(DIMENSIONS)}")
    over: dict = {}
    for dim, val in zip(DIMENSIONS, vals):
        if val not in DIMENSIONS[dim]:
            raise ValueError(f"unknown {dim} '{val}', choose from {list(DIMENSIONS[dim])}")
        over = merge(over, DIMENSIONS[dim][val])
    return Variant(name, over)


# --------------------------------------------------------------------------- running

# compact per-trade row used for ranking (only taken trades are kept)
@dataclass(frozen=True)
class Row:
    symbol: str
    day: date
    filled: bool
    r: float
    risk: float          # entry-to-stop distance in $
    price: float         # entry price
    hold_min: float      # minutes from fill to exit
    direction: str
    trigger: str
    outcome: str
    hour: int


def _to_rows(recs: list[bt.TradeRecord]) -> list[Row]:
    out = []
    for r in recs:
        if not r.taken:
            continue
        hold = ((r.exit_time - r.fill_time).total_seconds() / 60) if r.filled and r.exit_time is not None else 0.0
        out.append(Row(r.symbol, r.day, r.filled, r.r, abs(r.entry - r.stop), r.entry, hold,
                       r.direction, r.trigger, r.outcome, r.time.hour))
    return out


def _chunk_job(args):
    path, base_cfg, variants = args
    with open(path, "rb") as f:
        dcs = pickle.load(f)
    out = {}
    for v in variants:
        cfg = merge(base_cfg, v.overrides)
        out[v.name] = _to_rows(bt.run_on(dcs, cfg))
    return out


def run_grid(paths: list[Path], base_cfg: dict[str, Any], variants: list[Variant],
             workers: int | None = None, progress=None) -> dict[str, list[Row]]:
    workers = workers if workers is not None else max(1, (os.cpu_count() or 2) - 1)
    jobs = [(p, base_cfg, variants) for p in paths]
    results: dict[str, list[Row]] = {v.name: [] for v in variants}
    if workers <= 1 or len(jobs) == 1:
        it = map(_chunk_job, jobs)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        it = pool.map(_chunk_job, jobs)
    try:
        for n, part in enumerate(it, 1):
            for k, rows in part.items():
                results[k] += rows
            if progress:
                progress(f"ran chunk {n}/{len(jobs)}")
    finally:
        if pool:
            pool.shutdown()
    return results


def full_records(paths: list[Path], base_cfg: dict[str, Any], name: str,
                 days: set[date] | None = None) -> list[bt.TradeRecord]:
    """Every record (taken and not) for one variant, for detailed reports."""
    cfg = merge(base_cfg, variant(name).overrides)
    out = []
    for p in paths:
        with open(p, "rb") as f:
            dcs = pickle.load(f)
        if days is not None:
            dcs = [dc for dc in dcs if dc.day in days]
        out += bt.run_on(dcs, cfg)
    out.sort(key=lambda r: (r.time, r.symbol))
    return out


# --------------------------------------------------------------------------- costs and metrics

@dataclass(frozen=True)
class CostModel:
    """Rough cost of trading 1-2 DTE near-the-money options instead of shares,
    expressed in R so it can be subtracted from share-based results.

    cost per share of option = spread + theta_pct_per_hour * price * hours held
    option P&L ~ delta * underlying move, so in underlying-R units:
        cost_R = cost / (delta * risk)
    Defaults: $0.03 round-trip spread, ~0.03% of the share price per hour of
    decay (about $0.21/hour at QQQ $700), delta 0.45.
    """
    spread: float = 0.03
    theta_pct_per_hour: float = 0.0003
    delta: float = 0.45
    min_risk: float = 0.05

    def cost_r(self, row: Row) -> float:
        cost = self.spread + self.theta_pct_per_hour * row.price * row.hold_min / 60
        return cost / (self.delta * max(row.risk, self.min_risk))


@dataclass
class Metrics:
    alerts: int
    filled: int
    per_week: float
    fill_rate: float
    win_rate: float
    avg_r: float
    avg_net: float          # after modelled option costs
    ci_lo: float            # 90% bootstrap lower bound of avg_net
    ci_hi: float
    profit_factor: float    # after costs
    max_dd: float           # after costs, in R
    worst_streak: int
    years_positive: str     # e.g. "5/6"
    min_year: float         # worst calendar-year avg_net
    ran_away: int
    by_year: dict[int, tuple[int, float]]

    @property
    def rank_key(self) -> float:
        return self.ci_lo if self.filled >= 2 else -99.0


def metrics(rows: list[Row], n_days: int, cost: CostModel = CostModel()) -> Metrics:
    filled = [r for r in rows if r.filled]
    gross = [r.r for r in filled]
    net = [r.r - cost.cost_r(r) for r in filled]
    lo, hi = bt.bootstrap_ci(net) if len(net) >= 2 else (float("nan"), float("nan"))
    gains = sum(x for x in net if x > 0)
    losses = -sum(x for x in net if x < 0)
    eq = peak = dd = 0.0
    streak = worst = 0
    for x in net:
        eq += x
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
        streak = streak + 1 if x < 0 else 0
        worst = max(worst, streak)
    years: dict[int, list[float]] = {}
    for r, x in zip(filled, net):
        years.setdefault(r.day.year, []).append(x)
    by_year = {y: (len(v), float(np.mean(v))) for y, v in sorted(years.items())}
    pos = sum(1 for _, (n, m) in by_year.items() if m > 0)
    return Metrics(
        alerts=len(rows), filled=len(filled), per_week=len(rows) / max(n_days / 5, 1),
        fill_rate=len(filled) / len(rows) if rows else 0.0,
        win_rate=sum(1 for x in net if x > 0) / len(net) if net else 0.0,
        avg_r=float(np.mean(gross)) if gross else 0.0,
        avg_net=float(np.mean(net)) if net else 0.0,
        ci_lo=lo, ci_hi=hi,
        profit_factor=gains / losses if losses else (float("inf") if gains else 0.0),
        max_dd=dd, worst_streak=worst,
        years_positive=f"{pos}/{len(by_year)}",
        min_year=min((m for _, (n, m) in by_year.items()), default=0.0),
        ran_away=sum(1 for r in rows if r.outcome == "ran away"),
        by_year=by_year,
    )


def split_rows(rows: Iterable[Row], days: set[date]) -> list[Row]:
    return [r for r in rows if r.day in days]


def rank(results: dict[str, list[Row]], days: set[date], min_filled: int,
         cost: CostModel = CostModel()) -> list[tuple[str, Metrics]]:
    scored = []
    for name, rows in results.items():
        # days x symbols, so alerts per week is per symbol scanned
        m = metrics(split_rows(rows, days), len(days) * _symbols(rows), cost)
        scored.append((name, m))
    eligible = [s for s in scored if s[1].filled >= min_filled]
    rest = [s for s in scored if s[1].filled < min_filled]
    eligible.sort(key=lambda s: s[1].rank_key, reverse=True)
    rest.sort(key=lambda s: s[1].rank_key, reverse=True)
    return eligible + rest


def _symbols(rows: list[Row]) -> int:
    return len({r.symbol for r in rows}) or 1


def marginal(ranked: list[tuple[str, Metrics]], min_filled: int) -> dict[str, dict[str, float]]:
    """Median after-cost avg R of eligible variants for each value of each dimension:
    shows which choices help no matter what else is chosen."""
    out: dict[str, dict[str, float]] = {}
    elig = [(n, m) for n, m in ranked if m.filled >= min_filled]
    for i, dim in enumerate(DIMENSIONS):
        out[dim] = {}
        for val in DIMENSIONS[dim]:
            vals = [m.avg_net for n, m in elig if n.split("|")[i] == val]
            out[dim][val] = float(np.median(vals)) if vals else float("nan")
    return out
