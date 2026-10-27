"""Loads config.yaml into a plain dict and checks the weights."""
from __future__ import annotations

from datetime import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "config.yaml"

SCORED_WEIGHTS = ["vwap_side", "orb_break", "ema_trend", "rel_volume", "sweep", "mss",
                  "fvg_ifvg", "order_block", "rsi_ok", "rr", "timing_prime"]


def load_config(path: str | Path = DEFAULT_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    total = sum(cfg["weights"][k] for k in SCORED_WEIGHTS)
    if total != 100:
        raise ValueError(f"config weights must total 100, got {total}")
    return cfg


def parse_time(value: str) -> time:
    h, m = value.split(":")
    return time(int(h), int(m))
