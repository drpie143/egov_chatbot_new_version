from __future__ import annotations

import json
import time
from typing import Any

from egov_bot.schemas.common import Source
from egov_bot.storage.db import Database


class AppRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def log_query(
        self,
        request_id: str,
        session_id: str,
        question: str,
        answer: str,
        sources: list[Source],
        latency_ms: int,
        cached: bool,
    ) -> None:
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO query_logs(request_id, session_id, question, answer, sources_json, latency_ms, cached, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    session_id,
                    question,
                    answer,
                    json.dumps([source.to_dict() for source in sources], ensure_ascii=False),
                    latency_ms,
                    1 if cached else 0,
                    time.time(),
                ),
            )

    def save_feedback(
        self,
        rating: str,
        request_id: str | None = None,
        session_id: str | None = None,
        comment: str | None = None,
    ) -> None:
        rating = (rating or "").strip().lower()
        if rating not in {"like", "dislike", "neutral"}:
            raise ValueError("rating must be one of: like, dislike, neutral")
        with self.db.connect() as conn:
            conn.execute(
                "INSERT INTO feedback(request_id, session_id, rating, comment, created_at) VALUES (?, ?, ?, ?, ?)",
                (request_id, session_id, rating, comment or "", time.time()),
            )

    def increment_popular(self, name: str, url: str = "") -> None:
        if not name:
            return
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO popular_procedures(name, url, total_queries, updated_at)
                VALUES (?, ?, 1, ?)
                ON CONFLICT(name) DO UPDATE SET
                    total_queries = total_queries + 1,
                    url = CASE WHEN excluded.url != '' THEN excluded.url ELSE popular_procedures.url END,
                    updated_at = excluded.updated_at
                """,
                (name, url or "", time.time()),
            )

    def popular(self, limit: int = 10) -> list[dict[str, Any]]:
        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT name, url, total_queries
                FROM popular_procedures
                ORDER BY total_queries DESC, name ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def feedback_summary(self) -> dict[str, int]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT rating, COUNT(*) AS total FROM feedback GROUP BY rating"
            ).fetchall()
        summary = {"likes": 0, "dislikes": 0, "neutral": 0}
        for row in rows:
            if row["rating"] == "like":
                summary["likes"] = row["total"]
            elif row["rating"] == "dislike":
                summary["dislikes"] = row["total"]
            elif row["rating"] == "neutral":
                summary["neutral"] = row["total"]
        return summary

