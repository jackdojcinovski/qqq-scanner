"""The trade setups the money studies test, as option-ready signals."""
from __future__ import annotations

import os
from typing import Any

import pandas as pd

from . import intraday as ix
from . import optionsim as om
from . import orb

SETUPS = {
    "orb": "opening range (ORB)",
    "noise": "noise boundary",
    "noise_wide": "noise boundary, wide band",
    "vwap": "VWAP trend",
}


def collect(bars: dict[str, pd.DataFrame], grids: dict[str, ix.DayGrid], cfg: dict[str, Any],
            names: list[str] | None = None) -> dict[str, list[om.Signal]]:
    """All signals (longs and shorts) per setup key; regime rules are applied later."""
    out: dict[str, list[om.Signal]] = {}
    for key in names or list(SETUPS):
        if key == "orb":
            ots = orb.orb_trades(bars, cfg, orb.ORB_VARIANTS["both"], (1.0,),
                                 workers=min(len(bars), max(1, (os.cpu_count() or 2) - 1)))
            out[key] = om.from_orb(ots, "orb")
        elif key == "noise":
            out[key] = om.from_intraday([t for g in grids.values()
                                         for t in ix.noise_trades(g, sides="both", band_mult=1.0, stop="vwap")], grids)
        elif key == "noise_wide":
            out[key] = om.from_intraday([t for g in grids.values()
                                         for t in ix.noise_trades(g, sides="both", band_mult=1.5, stop="band_vwap")], grids)
        elif key == "vwap":
            out[key] = om.from_intraday([t for g in grids.values()
                                         for t in ix.vwap_trades(g, sides="both", first=90)], grids)
        else:
            raise KeyError(key)
    return out
