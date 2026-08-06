from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from egov_bot.config import Settings
from egov_bot.data.procedure_store import ProcedureStore
from egov_bot.retrieval.query_rewriter import strip_boilerplate
from egov_bot.retrieval.reranker import CrossEncoderReranker
from egov_bot.retrieval.sparse_index import SparseIndex
from egov_bot.schemas.common import Source
from egov_bot.utils.normalizer import normalize_text

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
    #: How many chunks of this procedure were in the candidate pool. 1 unless
    #: results were grouped by procedure.
    chunk_matches: int = 1
    #: Cross-encoder score, set only when the rerank stage ran.
    rerank_score: float | None = None

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
        self.bm25 = bm25 if isinstance(bm25, SparseIndex) else SparseIndex.load(bm25)
        self.embedding_model = embedding_model
        self.reranker = (
            CrossEncoderReranker(settings.rerank_model) if settings.rerank_enabled else None
        )
        self._check_sparse_alignment()

    def _check_sparse_alignment(self) -> None:
        """Warn when the sparse index was not built against the loaded corpus.

        Sparse hits are positional lookups into ``self.metadatas``. If the two
        come from different corpus builds the retriever still returns results,
        they just describe unrelated procedures — so this has to be loud.
        """
        if self.bm25 is None or not self.metadatas:
            return
        if self.bm25.doc_count != len(self.metadatas):
            logger.error(
                "Sparse index covers %s docs but %s chunks are loaded. The index was built "
                "against a different corpus; sparse hits will point at unrelated procedures. "
                "Rebuild with scripts/rebuild_bm25_index.py.",
                f"{self.bm25.doc_count:,}",
                f"{len(self.metadatas):,}",
            )

    @property
    def has_vector_index(self) -> bool:
        return self.faiss_index is not None and self.embedding_model is not None and bool(self.metadatas)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        mode: str = "hybrid",
        group_by_procedure: bool | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve documents for a query.

        Args:
            query: The search query.
            top_k: Maximum number of results to return.
            mode: Retrieval mode - "hybrid" (default), "bm25", or "dense".
            group_by_procedure: Keep only the best-scoring chunk per procedure
                before truncating to ``top_k``. Defaults to the configured
                value. Without it a single procedure can fill most of the
                top-k with its own chunks.
        """
        top_k = top_k or self.settings.top_k
        if group_by_procedure is None:
            group_by_procedure = self.settings.group_by_procedure
        if self.settings.strip_query_boilerplate:
            query = strip_boilerplate(query)

        # Grouping collapses many chunks into one result, so the candidate pool
        # has to be deep enough to still contain top_k *distinct* procedures.
        want = top_k * self.settings.group_candidate_multiplier if group_by_procedure else top_k

        ranked: list[RetrievalResult] | None = None
        if mode == "bm25":
            sparse = self._sparse_search(query, want)
            if sparse:
                ranked = self._scores_to_results(sparse)
        elif mode == "dense":
            dense = self._dense_search(query, want)
            if dense:
                ranked = self._scores_to_results(dense)
        else:
            # hybrid mode (default)
            dense = self._dense_search(query, want)
            sparse = self._sparse_search(query, want)
            if dense or sparse:
                ranked = self._fuse(dense, sparse)

        if ranked is not None:
            if group_by_procedure:
                ranked = _collapse_to_procedures(ranked, self.settings.group_key)
            if self.reranker is not None:
                shortlist = max(top_k, self.settings.rerank_candidates)
                ranked = self.reranker.rerank(query, ranked[:shortlist], top_k=top_k)
            return ranked[:top_k]

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

    def _dense_search(self, query: str, want: int | None = None) -> dict[int, float]:
        if not self.has_vector_index:
            return {}
        try:
            qv = self.embedding_model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype("float32")
            candidates = max(self.settings.faiss_candidates, (want or self.settings.top_k) * 5)
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

    def _sparse_search(self, query: str, want: int | None = None) -> dict[int, float]:
        if self.bm25 is None or not self.metadatas:
            return {}
        try:
            top_n = max(self.settings.bm25_candidates, (want or self.settings.top_k) * 5)
            pairs = [
                (idx, score)
                for idx, score in self.bm25.top_n(query, top_n)
                if 0 <= idx < len(self.metadatas)
            ]
            if not pairs:
                return {}
            scaled = _minmax([score for _, score in pairs])
            return {
                int(idx): float(score)
                for (idx, _), score in zip(pairs, scaled, strict=False)
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

    def _fuse(
        self,
        dense: dict[int, float],
        sparse: dict[int, float],
        strategy: str | None = None,
    ) -> list[RetrievalResult]:
        strategy = (strategy or self.settings.fusion_strategy).lower()
        if strategy == "rrf":
            fused = _fuse_rrf(dense, sparse, k=self.settings.rrf_k)
        elif strategy == "weighted":
            fused = _fuse_weighted(dense, sparse, alpha=self.settings.hybrid_alpha)
        elif strategy == "legacy":
            fused = _fuse_legacy(dense, sparse)
        else:
            logger.warning("Unknown fusion strategy %r; falling back to rrf.", strategy)
            fused = _fuse_rrf(dense, sparse, k=self.settings.rrf_k)

        return [
            self._build_result(
                idx, fused_score,
                dense_score=dense.get(idx, 0.0),
                sparse_score=sparse.get(idx, 0.0),
            )
            for idx, fused_score in sorted(fused.items(), key=lambda item: item[1], reverse=True)
        ]


def _collapse_to_procedures(
    results: list[RetrievalResult],
    key_mode: str = "parent",
) -> list[RetrievalResult]:
    """Keep the best-scoring chunk per procedure, preserving rank order.

    Results arrive ranked, so the first chunk seen for a procedure is its best
    one. The number of chunks that matched is recorded on the surviving result
    rather than discarded, since it is a useful relevance signal.
    """
    best: dict[str, RetrievalResult] = {}
    order: list[str] = []
    for result in results:
        if key_mode == "title":
            key = normalize_text(result.title) or result.parent_id or result.url
        else:
            key = result.parent_id or result.url or result.title
        existing = best.get(key)
        if existing is None:
            best[key] = result
            order.append(key)
        else:
            existing.chunk_matches += 1
    return [best[key] for key in order]


def _rank_order(scores: dict[int, float]) -> list[int]:
    return sorted(scores, key=lambda idx: scores[idx], reverse=True)


def _fuse_rrf(dense: dict[int, float], sparse: dict[int, float], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion: score = sum over rankers of 1 / (k + rank).

    Only ranks are used, so the two rankers do not need comparable score
    scales. That matters here because both score sets are min-max scaled
    within their own candidate window, which makes their magnitudes
    incomparable across queries.
    """
    fused: dict[int, float] = {}
    for scores in (dense, sparse):
        for rank, idx in enumerate(_rank_order(scores), start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return fused


def _fuse_weighted(dense: dict[int, float], sparse: dict[int, float], alpha: float = 0.5) -> dict[int, float]:
    """Convex combination of the normalised dense and sparse scores."""
    fused: dict[int, float] = {}
    for idx, score in dense.items():
        fused[idx] = fused.get(idx, 0.0) + alpha * score
    for idx, score in sparse.items():
        fused[idx] = fused.get(idx, 0.0) + (1.0 - alpha) * score
    return fused


def _fuse_legacy(dense: dict[int, float], sparse: dict[int, float]) -> dict[int, float]:
    """The original fusion, kept so published numbers stay reproducible."""
    fused: dict[int, float] = {}
    for scores in (dense, sparse):
        for rank, idx in enumerate(_rank_order(scores), start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (60 + rank) + 0.35 * scores[idx]
    return fused


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high == low:
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]

