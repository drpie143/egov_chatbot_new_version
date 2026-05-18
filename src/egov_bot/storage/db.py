from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS query_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    sources_json TEXT NOT NULL,
    latency_ms INTEGER NOT NULL,
    cached INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT,
    session_id TEXT,
    rating TEXT NOT NULL,
    comment TEXT,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS popular_procedures (
    name TEXT PRIMARY KEY,
    url TEXT,
    total_queries INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL
);
"""


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def import_legacy_json(self, popular_path: Path, feedback_path: Path) -> None:
        with self.connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM popular_procedures").fetchone()[0]
            if count == 0 and popular_path.exists():
                try:
                    data = json.loads(popular_path.read_text(encoding="utf-8"))
                    for item in data.get("popular_procedures", []):
                        name = str(item.get("name") or "").strip()
                        if not name:
                            continue
                        total = int(item.get("total_queries") or 0)
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO popular_procedures(name, url, total_queries, updated_at)
                            VALUES (?, ?, ?, ?)
                            """,
                            (name, "", total, time.time()),
                        )
                except Exception:
                    pass

            feedback_count = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
            if feedback_count == 0 and feedback_path.exists():
                try:
                    data = json.loads(feedback_path.read_text(encoding="utf-8"))
                    summary = data.get("feedback_summary", {})
                    for rating, total in [("like", summary.get("likes", 0)), ("dislike", summary.get("dislikes", 0))]:
                        for _ in range(int(total or 0)):
                            conn.execute(
                                "INSERT INTO feedback(request_id, session_id, rating, comment, created_at) VALUES (?, ?, ?, ?, ?)",
                                ("legacy", "legacy", rating, "", time.time()),
                            )
                except Exception:
                    pass

