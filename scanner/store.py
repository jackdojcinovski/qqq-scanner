"""SQLite log of live alerts and their outcomes (data/scanner.db)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .config import ROOT

DB_PATH = ROOT / "data" / "scanner.db"


class Store:
    def __init__(self, path: Path | str = DB_PATH):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS alerts (
                trade_no TEXT PRIMARY KEY, key TEXT, symbol TEXT, time TEXT, direction TEXT,
                score INTEGER, band TEXT, taken INTEGER, skip_reason TEXT,
                entry REAL, stop REAL, target REAL, rr REAL, detail TEXT);
            CREATE TABLE IF NOT EXISTS outcomes (
                trade_no TEXT, time TEXT, outcome TEXT, r REAL, price REAL);
        """)
        self.db.commit()

    def alert(self, trade_no: str, setup, symbol: str, band: str, taken: bool, skip: str) -> None:
        detail = json.dumps({c.name: [c.points, c.detail] for c in setup.components})
        self.db.execute("INSERT OR REPLACE INTO alerts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (trade_no, setup.key, symbol, setup.time.isoformat(), setup.direction, setup.score,
                         band, int(taken), skip, setup.entry, setup.stop, setup.target, setup.rr, detail))
        self.db.commit()

    def outcome(self, trade_no: str, when, outcome: str, r: float | None, price: float | None) -> None:
        self.db.execute("INSERT INTO outcomes VALUES (?,?,?,?,?)",
                        (trade_no, when.isoformat() if when is not None else None, outcome, r, price))
        self.db.commit()

    def close(self) -> None:
        self.db.close()
