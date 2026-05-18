from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Source:
    title: str
    url: str
    agency: str = ""
    score: float = 0.0
    snippet: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "agency": self.agency,
            "score": round(float(self.score), 6),
            "snippet": self.snippet,
        }


@dataclass(frozen=True)
class ChatResult:
    answer: str
    sources: list[Source] = field(default_factory=list)
    request_id: str = ""
    latency_ms: int = 0
    cached: bool = False
    context_source: str | None = None
    timings: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": [source.to_dict() for source in self.sources],
            "request_id": self.request_id,
            "latency_ms": self.latency_ms,
            "cached": self.cached,
            "context_source": self.context_source,
            "timings": self.timings,
        }

