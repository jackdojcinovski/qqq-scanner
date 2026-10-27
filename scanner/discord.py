"""Discord webhook messages for the live scanner."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import requests

from .scorer import Setup

log = logging.getLogger(__name__)

GREEN, RED, GREY, BLUE, AMBER = 0x2ECC71, 0xE74C3C, 0x95A5A6, 0x3498DB, 0xF39C12


class Notifier:
    """Posts to one webhook. Retries on rate limits and network errors; never raises."""

    def __init__(self, url: str, username: str = "QQQ Scanner", timeout: int = 10):
        self.url = url
        self.username = username
        self.timeout = timeout

    def send(self, content: str = "", embeds: list[dict] | None = None) -> bool:
        payload: dict[str, Any] = {"username": self.username, "allowed_mentions": {"parse": []}}
        if content:
            payload["content"] = content[:2000]
        if embeds:
            payload["embeds"] = embeds[:10]
        for attempt in range(4):
            try:
                r = requests.post(self.url, json=payload, timeout=self.timeout)
            except requests.RequestException as exc:
                log.warning("discord send failed (%s), retrying", exc)
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code == 429:
                wait = float(r.json().get("retry_after", 1)) if r.headers.get("content-type", "").startswith("application/json") else 1.0
                time.sleep(min(wait, 10))
                continue
            if r.status_code in (200, 204):
                return True
            log.error("discord returned %s: %s", r.status_code, r.text[:200])
            return False
        return False


@dataclass
class MemoryNotifier:
    """Collects messages instead of sending them (tests, dry runs)."""
    sent: list[dict] = field(default_factory=list)

    def send(self, content: str = "", embeds: list[dict] | None = None) -> bool:
        self.sent.append({"content": content, "embeds": embeds or []})
        return True


# --------------------------------------------------------------------------- formatting

def next_expiry(day: date, min_dte: int) -> date:
    d, n = day, 0
    while n < min_dte:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n += 1
    return d


def contract_plan(setup: Setup, cfg: dict[str, Any], symbol: str,
                  option_price: float | None = None) -> dict[str, Any]:
    live = cfg["live"]
    right = "C" if setup.direction == "long" else "P"
    strike = round(setup.entry)
    expiry = next_expiry(setup.time.date(), live["option_min_dte"])
    risk_per_contract = abs(setup.entry - setup.stop) * live["option_delta"] * 100
    max_n = int(live["max_risk_per_trade_usd"] // risk_per_contract) if risk_per_contract > 0 else 0
    max_n = max(0, min(max_n, live["max_contracts"]))
    return {
        "label": f"{symbol} {expiry:%b %d} {strike} {'Call' if right == 'C' else 'Put'}",
        "expiry": expiry, "strike": strike, "right": right,
        "risk_per_contract": risk_per_contract, "max_contracts": max_n,
        "option_price": option_price,
    }


def alert_message(setup: Setup, cfg: dict[str, Any], symbol: str, trade_no: str,
                  plan: dict[str, Any], context: list[str]) -> tuple[str, list[dict]]:
    long = setup.direction == "long"
    paper = "[PAPER] " if cfg["live"].get("paper", True) else ""
    side = "LONG" if long else "SHORT"
    e = cfg.get("exit", {})
    mode = e.get("mode", "fixed")
    valid = setup.time + timedelta(minutes=cfg["trade"]["entry_valid_minutes"] + 1)
    exit_rule = {
        "fixed": f"full size to target or stop, close after {e.get('time_exit_minutes', 60)} min",
        "partial": f"take {e.get('partial_fraction', 0.5):.0%} off at +{e.get('partial_at_r', 1):g}R, "
                   f"stop to breakeven, rest to target; close after {e.get('time_exit_minutes', 60)} min",
        "trail": f"no fixed target: after +{e.get('trail_after_r', 1):g}R trail the stop "
                 f"{e.get('trail_distance_r', 1):g}R behind the best price; close after "
                 f"{e.get('time_exit_minutes', 60)} min",
    }[mode]
    target_txt = f"{setup.target:.2f} ({setup.target_name}), R:R {setup.rr:.1f}" if mode != "trail" \
        else f"trailing (reference {setup.target:.2f})"
    price_txt = f" ~${plan['option_price']:.2f} indicative" if plan.get("option_price") else ""
    seen = "\n".join(f"{'+' if c.hit else '-'} {c.name} ({c.points}): {c.detail}"
                     for c in setup.components)
    fields = [
        {"name": "Entry", "value": f"{setup.entry:.2f} limit ({setup.entry_zone})\nvalid until {valid:%H:%M} ET",
         "inline": True},
        {"name": "Stop", "value": f"{setup.stop:.2f}", "inline": True},
        {"name": "Target", "value": target_txt, "inline": True},
        {"name": "Contract", "value": f"{plan['label']}{price_txt}\n"
                                      f"risk ~${plan['risk_per_contract']:.0f}/contract, "
                                      f"max {plan['max_contracts']} at your limit", "inline": False},
        {"name": "Exit plan", "value": exit_rule, "inline": False},
        {"name": "What we see", "value": f"```diff\n{seen[:990]}\n```", "inline": False},
        {"name": "Context", "value": "\n".join(context)[:1024] or "-", "inline": False},
    ]
    content = f"{paper}**{symbol} {side}** score {setup.score}/100, entry {setup.entry:.2f}, stop {setup.stop:.2f}"
    embed = {"title": f"{paper}{symbol} {side} | Score {setup.score}/100 | Trade #{trade_no}",
             "color": GREEN if long else RED, "fields": fields,
             "footer": {"text": "Alerts only. You place and manage the trade."}}
    return content, [embed]


OUTCOME_TEXT = {
    "target": ("TARGET HIT", GREEN), "stop": ("STOPPED OUT", RED), "breakeven": ("OUT AT BREAKEVEN", GREY),
    "trail": ("TRAILING STOP HIT", GREEN), "time": ("TIME EXIT", AMBER), "close": ("CLOSED AT SESSION END", AMBER),
    "no fill": ("EXPIRED, NOT FILLED", GREY), "ran away": ("CANCELLED, PRICE LEFT WITHOUT A FILL", GREY),
    "news": ("CLOSE NOW: NEWS WINDOW", AMBER), "stopped": ("SCANNER STOPPED, TRADE NOT TRACKED", GREY),
}


def update_message(symbol: str, trade_no: str, setup: Setup, outcome: str, r: float | None,
                   when, price: float | None, today_line: str) -> tuple[str, list[dict]]:
    title, color = OUTCOME_TEXT.get(outcome, (outcome.upper(), BLUE))
    if outcome in ("target", "trail") or (r is not None and r > 0 and outcome in ("time", "close")):
        color = GREEN
    elif r is not None and r < 0:
        color = RED
    lines = []
    if price is not None:
        lines.append(f"{symbol} {setup.entry:.2f} -> {price:.2f}" + (f" ({r:+.2f}R)" if r is not None else ""))
    lines.append(today_line)
    content = f"Trade #{trade_no} {title}" + (f" {r:+.2f}R" if r is not None else "")
    return content, [{"title": f"Trade #{trade_no} {title} at {when:%H:%M}", "color": color,
                      "description": "\n".join(lines)}]


def event_message(trade_no: str, kind: str, when, price: float, note: str) -> tuple[str, list[dict]]:
    title = {"fill": "FILLED", "partial": "PARTIAL TAKEN, STOP TO BREAKEVEN"}.get(kind, kind.upper())
    return f"Trade #{trade_no} {title}", [{"title": f"Trade #{trade_no} {title} at {when:%H:%M}",
                                            "color": BLUE, "description": f"at {price:.2f}. {note}"}]


def simple(title: str, text: str, color: int = BLUE) -> tuple[str, list[dict]]:
    return title, [{"title": title, "description": text[:4000], "color": color}]
