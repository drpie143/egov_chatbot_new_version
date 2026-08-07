"""Cross-encoder reranking stage.

Bi-encoder retrieval scores the query and the document separately, so it never
sees them together. A cross-encoder does, at the cost of one forward pass per
candidate — which is why it runs over a shortlist rather than the corpus.

Measured on the 167-question FAQ set, recall saturates at depth 50
(Recall@50 = Recall@100 = 0.9162) while Recall@1 sits at 0.7365. Reranking
cannot raise the ceiling; its headroom is entirely in moving the correct
procedure from somewhere in the shortlist up to rank 1.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Reorders retrieval results with a cross-encoder, loaded on first use."""

    def __init__(self, model_name: str, device: str | None = None, max_length: int = 512) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self._model: Any = None
        self._failed = False

    @property
    def available(self) -> bool:
        return not self._failed

    def _load(self) -> Any:
        if self._model is not None or self._failed:
            return self._model
        try:
            import torch
            from sentence_transformers import CrossEncoder

            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = CrossEncoder(self.model_name, device=device, max_length=self.max_length)
            logger.info("Loaded reranker %s on %s", self.model_name, device)
        except Exception as exc:
            # Retrieval must keep working without the reranker, but silently
            # returning the unranked order is how the earlier tokenizer and
            # fuzzy-matching failures went unnoticed, so say so once and mark
            # the stage unavailable rather than retrying on every query.
            logger.error("Reranker %s unavailable, keeping retrieval order: %s", self.model_name, exc)
            self._failed = True
        return self._model

    def rerank(self, query: str, results: list, top_k: int | None = None) -> list:
        """Return ``results`` reordered by cross-encoder relevance."""
        if not results:
            return results
        model = self._load()
        if model is None:
            return results[:top_k] if top_k else results

        pairs = [(query, self._document_text(result)) for result in results]
        try:
            scores = model.predict(pairs, show_progress_bar=False)
        except Exception as exc:
            logger.error("Reranker scoring failed, keeping retrieval order: %s", exc)
            return results[:top_k] if top_k else results

        order = sorted(range(len(results)), key=lambda i: float(scores[i]), reverse=True)
        reranked = []
        for position in order:
            result = results[position]
            result.rerank_score = float(scores[position])
            reranked.append(result)
        return reranked[:top_k] if top_k else reranked

    @staticmethod
    def _document_text(result: Any) -> str:
        """Title plus body: the title alone is often the discriminating part."""
        title = (getattr(result, "title", "") or "").strip()
        text = (getattr(result, "text", "") or "").strip()
        if title and not text.startswith(title):
            return f"{title}\n{text}"
        return text or title
