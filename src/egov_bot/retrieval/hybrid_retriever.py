from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from egov_bot.config import Settings
from egov_bot.data.procedure_store import ProcedureStore
from egov_bot.schemas.common import Source

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    index: int
    parent_id: str
    title: str
    url: str
    text: str
    score: float
    dense_score: float = 0.0
    sparse_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_source(self, store: ProcedureStore) -> Source:
        snippet = (self.text or "")[:280]
        source = store.source_for(self.parent_id, score=self.score, snippet=snippet)
        if source:
            return source
        return Source(
            title=self.title or "Thu tuc hanh chinh",
            url=self.url or self.parent_id,
            score=self.score,
            snippet=snippet,
        )


class HybridRetriever:
    def __init__(
        self,
        settings: Settings,
        procedure_store: ProcedureStore,
        metadatas: list[dict[str, Any]] | None = None,
        faiss_index: Any = None,
        bm25: Any = None,
        embedding_model: Any = None,
    ) -> None:
        self.settings = settings
        self.procedure_store = procedure_store
        self.metadatas = metadatas or []
        self.faiss_index = faiss_index
        self.bm25 = bm25
        self.embedding_model = embedding_model

    @property
    def has_vector_index(self) -> bool:
        return self.faiss_index is not None and self.embedding_model is not None and bool(self.metadatas)

    def retrieve(self, query: str, top_k: int | None = None, mode: str = "hybrid") -> list[RetrievalResult]:
        """Retrieve documents for a query.

        Args:
            query: The search query.
            top_k: Maximum number of results to return.
            mode: Retrieval mode - "hybrid" (default), "bm25", or "dense".
        """
        top_k = top_k or self.settings.top_k

        if mode == "bm25":
            sparse = self._sparse_search(query)
            if sparse:
                return self._scores_to_results(sparse)[:top_k]
        elif mode == "dense":
            dense = self._dense_search(query)
            if dense:
                return self._scores_to_results(dense)[:top_k]
        else:
            # hybrid mode (default)
            dense = self._dense_search(query)
            sparse = self._sparse_search(query)
            if dense or sparse:
                fused = self._fuse(dense, sparse)
                return fused[:top_k]

        fallback_sources = self.procedure_store.search(query, limit=top_k)
        return [
            RetrievalResult(
                index=-1,
                parent_id=source.url,
                title=source.title,
                url=source.url,
                text=source.snippet,
                score=source.score,
            )
            for source in fallback_sources
        ]

    def search(self, query: str, limit: int | None = None) -> list[Source]:
        limit = limit or self.settings.search_limit
        results = self.retrieve(query, top_k=limit)
        if results:
            return [result.to_source(self.procedure_store) for result in results]
        return self.procedure_store.search(query, limit=limit)

    def _dense_search(self, query: str) -> dict[int, float]:
        if not self.has_vector_index:
            return {}
        try:
            qv = self.embedding_model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype("float32")
            candidates = max(self.settings.faiss_candidates, self.settings.top_k * 5)
            distances, indices = self.faiss_index.search(qv, candidates)
            raw_scores = [float(value) for value in distances[0].tolist()]
            scaled = _minmax(raw_scores)
            dense: dict[int, float] = {}
            for idx, score in zip(indices[0].tolist(), scaled, strict=False):
                if 0 <= idx < len(self.metadatas):
                    dense[idx] = float(score)
            return dense
        except Exception as exc:
            logger.warning("Dense search failed: %s", exc)
            return {}

    def _sparse_search(self, query: str) -> dict[int, float]:
        if self.bm25 is None or not self.metadatas:
            return {}
        try:
            scores = self.bm25.get_scores(query.split())
            if scores is None:
                return {}
            values = [float(value) for value in list(scores)]
            if not values:
                return {}
            top_n = min(max(self.settings.bm25_candidates, self.settings.top_k * 5), len(values))
            order = sorted(range(len(values)), key=lambda idx: values[idx], reverse=True)[:top_n]
            scaled_scores = _minmax([values[idx] for idx in order])
            return {
                int(idx): float(score)
                for idx, score in zip(order, scaled_scores, strict=False)
                if 0 <= int(idx) < len(self.metadatas)
            }
        except Exception as exc:
            logger.warning("Sparse search failed: %s", exc)
            return {}

    def _build_result(
        self,
        idx: int,
        score: float,
        dense_score: float = 0.0,
        sparse_score: float = 0.0,
    ) -> RetrievalResult:
        """Build a RetrievalResult from a metadata index."""
        metadata = self.metadatas[idx]
        parent_id = str(metadata.get("parent_id") or metadata.get("nguon") or "")
        text = str(metadata.get("text") or metadata.get("raw") or metadata.get("content") or "")
        title = str(metadata.get("ten_thu_tuc") or metadata.get("title") or "")
        url = str(metadata.get("nguon") or parent_id)
        return RetrievalResult(
            index=idx,
            parent_id=parent_id or url,
            title=title,
            url=url,
            text=text,
            score=float(score),
            dense_score=float(dense_score),
            sparse_score=float(sparse_score),
            metadata=metadata,
        )

    def _scores_to_results(self, scores: dict[int, float]) -> list[RetrievalResult]:
        """Convert a score dict {index: score} to sorted RetrievalResult list."""
        return [
            self._build_result(idx, score)
            for idx, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        ]

    def _fuse(self, dense: dict[int, float], sparse: dict[int, float]) -> list[RetrievalResult]:
        ranks: dict[int, float] = {}
        dense_order = sorted(dense, key=lambda idx: dense[idx], reverse=True)
        sparse_order = sorted(sparse, key=lambda idx: sparse[idx], reverse=True)

        for rank, idx in enumerate(dense_order, start=1):
            ranks[idx] = ranks.get(idx, 0.0) + 1.0 / (60 + rank) + 0.35 * dense[idx]
        for rank, idx in enumerate(sparse_order, start=1):
            ranks[idx] = ranks.get(idx, 0.0) + 1.0 / (60 + rank) + 0.35 * sparse[idx]

        return [
            self._build_result(
                idx, fused_score,
                dense_score=dense.get(idx, 0.0),
                sparse_score=sparse.get(idx, 0.0),
            )
            for idx, fused_score in sorted(ranks.items(), key=lambda item: item[1], reverse=True)
        ]


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high == low:
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]

